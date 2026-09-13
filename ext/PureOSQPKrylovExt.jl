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
        PureOSQP.multiply!(ws.work_m, ws.work_m, ws.rho_vec)
        mul_At!(y, ws, ws.work_m)
    else
        fill!(y, zero(eltype(y)))
    end
    mul_P!(ws.work_n, ws, x)
    PureOSQP.add_scaled!(y, ws.work_n, ws.settings.sigma, x)
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
    reduction::T        # multiplies the tolerance; halved when CG stops iterating
    idle_solves::Int    # consecutive solves in which CG took no iteration
end

function PureOSQP.indirect_backend(proto::AbstractVector{T}, n::Integer, m::Integer) where {T <: Real}
    kws = CgWorkspace(n, n, typeof(similar(proto, T, n)))
    # `cg!` allocates its preconditioned vector on first use, when it finds the field empty.
    # Filling it here is what makes the very first solve allocation-free, not merely every
    # solve after the first.
    kws.z = similar(proto, T, n)
    return IndirectCG{T, typeof(similar(proto, T, n)), typeof(kws)}(
        kws, similar(proto, T, n), fill!(similar(proto, T, n), one(T)), one(T), 0
    )
end

PureOSQP.backend_name(::IndirectCG) = :indirect

# Matrix-free: the preconditioner is a diagonal, not a factorization, so there is no factor
# to count.
PureOSQP.backend_info(ls::IndirectCG) = PureOSQP.BackendInfo(
    PureOSQP.backend_name(ls), false, :reduced, length(ls.rhs), 0
)

"""
    factorize!(ls::IndirectCG, ws) -> Bool

Rebuild the Jacobi preconditioner for the current `ρ`. The reduced diagonal is
`c·D[j]²·P[j,j] + σ + Σᵢ ρᵢ (E[i] A[i,j] D[j])²`, which each column yields directly.

Always succeeds: there is nothing here that can be singular, since `σ > 0` keeps every
diagonal entry positive.
"""
function PureOSQP.factorize!(ls::IndirectCG{T}, ws)::Bool where {T}
    PureOSQP.reduced_diagonal!(
        ls.prec, T, ws.P, ws.A, ws.rho_vec, ws.E, ws.D, ws.settings.sigma, ws.c
    )
    return true
end

"""
    solve_system!(ls::IndirectCG, ws, rhs_x, rhs_z) -> Nothing

Solve the reduced system by preconditioned CG.

The tolerance follows the ADMM residuals rather than being fixed: an early iterate does not
deserve an exact inner solve, and a late one does. It is `cg_tol_fraction` of the current
residual level, floored at `eps(T)` relative to the right-hand side so it cannot chase zero.
This makes the solve *inexact*, so iterates differ from the direct backends in the last
digits even though both converge to the same solution.

CG starts from the previous `x̃`. Late in a solve that guess already meets the tolerance, CG
takes no step, and the iterate stops moving; after `cg_tol_reduction` such solves in a row
the tolerance is halved, which is what that setting counts, as in libosqp. The floor is
relative rather than a fixed `sqrt(eps)` because a fixed floor sits above tight outer
tolerances, and no amount of halving gets below it.
"""
function PureOSQP.solve_system!(ls::IndirectCG{T}, ws, rhs_x, rhs_z)::Nothing where {T}
    m = ws.m
    if m > 0
        PureOSQP.multiply!(ws.work_m, ws.rho_vec, rhs_z)
        mul_At!(ls.rhs, ws, ws.work_m)
        PureOSQP.increment!(ls.rhs, rhs_x)
    else
        copyto!(ls.rhs, rhs_x)
    end

    s = ws.settings
    level = max(ws.scaled_prim_res, ws.scaled_dual_res)
    floor = eps(T) * max(one(T), norm_inf(ls.rhs))
    atol = max(ls.reduction * s.cg_tol_fraction * level, floor)
    op = ReducedOperator(ws)
    # The previous step's `x̃` is the best available guess: consecutive ADMM subproblems
    # differ by one relaxation step, so starting from zero discards most of the work and
    # the inner budget is spent recovering it.
    Krylov.warm_start!(ls.kws, ws.xtilde)
    cg!(
        ls.kws, op, ls.rhs;
        M = LinearAlgebra.Diagonal(ls.prec), ldiv = false,
        atol = atol, rtol = zero(T), itmax = s.cg_max_iter,
    )
    ls.idle_solves = iszero(ls.kws.stats.niter) ? ls.idle_solves + 1 : 0
    if ls.idle_solves >= s.cg_tol_reduction
        ls.reduction /= 2
        ls.idle_solves = 0
    end
    copyto!(ws.xtilde, ls.kws.x)
    m > 0 && mul_A!(ws.ztilde, ws, ws.xtilde)
    return nothing
end

end # module PureOSQPKrylovExt
