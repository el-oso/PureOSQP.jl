"""
    PureOSQPKrylovExt

The matrix-free linear-system backend, loaded when Krylov.jl is.

Every other backend forms the reduced matrix `P̃ + σI + Ãᵀ diag(ρ) Ã` and factors it. This
one never forms it: it applies it through the same `mul_A!`, `mul_At!` and `mul_P!` the
iteration already uses, so a problem that can only supply matrix-vector products is
solvable. Selected with `linsys = :indirect`.

Krylov.jl is a weak dependency, not a core one. It has no non-stdlib dependencies of its
own, but it costs about 185 ms to load against the 76 ms of this whole package, and a
direct factorization is some two orders of magnitude faster whenever the matrix *can* be
formed. Nobody should pay that who is not using this backend.
"""
module PureOSQPKrylovExt

using PureOSQP: PureOSQP, LinearSystem, Workspace, mul_A!, mul_At!, mul_P!, norm_inf
using Krylov: Krylov, CgWorkspace, cg!
using LinearAlgebra: LinearAlgebra, mul!

"""
    ReducedOperator{W}

`x ↦ (P̃ + σI + Ãᵀ diag(ρ) Ã) x`, applied without forming anything.

Built inside `solve_system!` from the workspace it is given, so it is concretely typed at
the call site rather than holding an abstractly typed back-reference.
"""
struct ReducedOperator{T <: Real, W <: Workspace}
    ws::W
end

# `T` is carried in the operator's own type. Krylov checks `eltype(A)` against the vectors'
# and falls back to a slower, allocating path when they disagree, so this cannot be left to
# a generic `eltype` of the workspace type.
ReducedOperator(ws::Workspace{T}) where {T} = ReducedOperator{T, typeof(ws)}(ws)

Base.size(op::ReducedOperator) = (op.ws.n, op.ws.n)
Base.size(op::ReducedOperator, d::Integer) = op.ws.n
Base.eltype(::ReducedOperator{T}) where {T} = T

function LinearAlgebra.mul!(y::AbstractVector, op::ReducedOperator, x::AbstractVector)
    ws = op.ws
    # `mul_A!` and `mul_At!` use `ws.tmp_m`/`ws.tmp_n` as scratch, so `work_m` carries the
    # intermediate here rather than aliasing theirs.
    if ws.m > 0
        mul_A!(ws.work_m, ws, x)
        for i in eachindex(ws.work_m, ws.rho_vec)
            ws.work_m[i] *= ws.rho_vec[i]
        end
        mul_At!(y, ws, ws.work_m)
    else
        fill!(y, zero(eltype(y)))
    end
    mul_P!(ws.work_n, ws, x)
    σ = ws.settings.sigma
    for i in eachindex(y, x, ws.work_n)
        y[i] += ws.work_n[i] + σ * x[i]
    end
    return y
end

"""
    IndirectCG{T,V,K} <: LinearSystem

Conjugate gradients on the reduced system, with a Jacobi preconditioner.

`factorize!` builds the preconditioner rather than a factorization: the diagonal of the
reduced matrix, which is computable column by column without assembling the matrix itself.
The Krylov workspace is allocated once and reused, so the per-iteration solve allocates
nothing.
"""
mutable struct IndirectCG{T <: Real, V <: AbstractVector{T}, K} <: LinearSystem
    kws::K              # Krylov's CgWorkspace, reused across solves
    rhs::V
    prec::V             # the reduced diagonal, inverted
    tol_scale::T        # shrinks as the ADMM residuals do
end

function PureOSQP.indirect_backend(proto::AbstractVector{T}, n::Integer, m::Integer) where {T <: Real}
    kws = CgWorkspace(n, n, typeof(similar(proto, T, n)))
    # `cg!` allocates its preconditioned vector on first use, when it finds the field empty.
    # Filling it here is what makes the very first solve allocation-free, not merely every
    # solve after the first.
    kws.z = similar(proto, T, n)
    return IndirectCG{T, typeof(similar(proto, T, n)), typeof(kws)}(
        kws, similar(proto, T, n), fill!(similar(proto, T, n), one(T)), one(T)
    )
end

PureOSQP.backend_name(::IndirectCG) = :indirect

"""
    factorize!(ls::IndirectCG, ws) -> Bool

Rebuild the Jacobi preconditioner for the current `ρ`. The reduced diagonal is
`c·D[j]²·P[j,j] + σ + Σᵢ ρᵢ (E[i] A[i,j] D[j])²`, which each column yields directly.

Always succeeds: there is nothing here that can be singular, since `σ > 0` keeps every
diagonal entry positive.
"""
function PureOSQP.factorize!(ls::IndirectCG{T}, ws)::Bool where {T}
    n, m = ws.n, ws.m
    σ, c = ws.settings.sigma, ws.c
    for j in 1:n
        dj = ws.D[j]
        d = c * dj * T(ws.P[j, j]) * dj + σ
        for i in 1:m
            a = ws.E[i] * T(ws.A[i, j]) * dj
            d += ws.rho_vec[i] * a * a
        end
        ls.prec[j] = inv(max(d, sqrt(eps(T))))
    end
    return true
end

"""
    solve_system!(ls::IndirectCG, ws, rhs_x, rhs_z) -> Nothing

Solve the reduced system by preconditioned CG.

The tolerance follows the ADMM residuals rather than being fixed: an early iterate does not
deserve an exact inner solve, and a late one does. It is `cg_tol_fraction` of the current
residual level, tightened by `cg_tol_reduction` as the outer iteration converges, and
floored so it cannot chase zero. This makes the solve *inexact*, so iterates differ from
the direct backends in the last digits even though both converge to the same solution.
"""
function PureOSQP.solve_system!(ls::IndirectCG{T}, ws, rhs_x, rhs_z)::Nothing where {T}
    m = ws.m
    if m > 0
        for i in eachindex(ws.work_m, ws.rho_vec, rhs_z)
            ws.work_m[i] = ws.rho_vec[i] * rhs_z[i]
        end
        mul_At!(ls.rhs, ws, ws.work_m)
        for i in eachindex(ls.rhs, rhs_x)
            ls.rhs[i] += rhs_x[i]
        end
    else
        copyto!(ls.rhs, rhs_x)
    end

    s = ws.settings
    level = max(ws.scaled_prim_res, ws.scaled_dual_res)
    atol = max(s.cg_tol_fraction * level / s.cg_tol_reduction, sqrt(eps(T)))
    op = ReducedOperator(ws)
    cg!(
        ls.kws, op, ls.rhs;
        M = LinearAlgebra.Diagonal(ls.prec), ldiv = false,
        atol = atol, rtol = zero(T), itmax = s.cg_max_iter,
    )
    copyto!(ws.xtilde, ls.kws.x)
    m > 0 && mul_A!(ws.ztilde, ws, ws.xtilde)
    return nothing
end

end # module PureOSQPKrylovExt
