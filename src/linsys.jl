"""
    LinearSystem

Interface for the factorization that solves the ADMM subproblem

    ⎡P̃ + σI      Ãᵀ   ⎤ ⎡x̃⎤   ⎡rhs_x⎤
    ⎣Ã       −diag(ρ⁻¹)⎦ ⎣ν⎦ = ⎣rhs_z⎦

once per iteration. A backend owns its own storage and factorization object; the workspace
holds one, chosen at [`setup`](@ref) and fixed for the workspace's life, so every call
dispatches statically.

Implementations must provide the two methods below; the contract is enforced at
precompilation.
"""
abstract type LinearSystem end

function factorize! end
function solve_system! end

@contract LinearSystem begin
    factorize!(::Self, ::Any)::Bool
    solve_system!(::Self, ::Any, ::Any, ::Any)::Nothing
end

"""
    ReducedCholesky{T,M} <: LinearSystem

Eliminates `ν` and solves the `n×n` symmetric positive definite reduced system

    (P̃ + σI + Ãᵀ diag(ρ) Ã) x̃ = rhs_x + Ãᵀ(ρ ⊙ rhs_z),   z̃ = Ã x̃

`W` holds the scaled `Ã` with `sqrt(ρ)` folded in, so the reduced matrix is one `syrk`.
`Rinv` holds that matrix's inverse in its upper triangle, so each iteration's solve is a
single `symv` rather than the two triangular solves of a Cholesky `ldiv!`. Both cost `2n²`
flops, but a triangular solve computes its entries in sequence while `symv` does not, which
makes it about seven times faster at `n = 200`. Inverting is sound here because the reduced
matrix carries the `σI` regularization: its conditioning is bounded, and the residuals of
the two forms agree to within a small factor.

This is the default: the reduced matrix is smaller than the full system for every `m`, and
measurement puts it faster in every dense regime.
"""
struct ReducedCholesky{T <: Real, M <: AbstractMatrix{T}} <: LinearSystem
    W::M
    Rinv::M
end

"""
    ReducedCholesky(proto::AbstractVector, n, m)

Build the backend's storage as `similar(proto, ...)`, so it follows the array type of the
data it was given rather than always being a `Matrix`. `proto` is a dense vector, not one
of the problem matrices: both buffers are dense even when `P` and `A` are not.
"""
function ReducedCholesky(proto::AbstractVector{T}, n::Integer, m::Integer) where {T <: Real}
    W = similar(proto, T, m, n)
    return ReducedCholesky{T, typeof(W)}(W, similar(proto, T, n, n))
end

"""
    FullKKT{T,M,V,F} <: LinearSystem

Factors the full `(n+m)×(n+m)` quasi-definite matrix with `bunchkaufman!`, which is what
the reference implementation does. Slower than [`ReducedCholesky`](@ref), but it does not
square the conditioning of `Ã`, so it is the more accurate factorization at moderate
conditioning and the better choice when a result is in question.
"""
mutable struct FullKKT{T <: Real, M <: AbstractMatrix{T}, V <: AbstractVector{T}, F} <: LinearSystem
    K::M
    rhs::V
    fact::F
end

"""
    FullKKT(proto::AbstractVector, n, m)

Build the backend's storage as `similar(proto, ...)`, following the array type of the data
it was given. See [`ReducedCholesky`](@ref) on why `proto` is a vector.
"""
function FullKKT(proto::AbstractVector{T}, n::Integer, m::Integer) where {T <: Real}
    K = similar(proto, T, n + m, n + m)
    rhs = similar(proto, T, n + m)
    fact = bunchkaufman!(Symmetric(fill(one(T), 1, 1)))
    return FullKKT{T, typeof(K), typeof(rhs), typeof(fact)}(K, rhs, fact)
end

"Name of the backend, for reporting."
backend_name(::ReducedCholesky) = :cholesky
backend_name(::FullKKT) = :bunchkaufman

# Overwrite the Cholesky factor occupying `R` with the inverse it factors. `potri!` does
# this in place and touches only the upper triangle; the fallback covers element types
# LAPACK has no method for.
invert_spd!(R::StridedMatrix{<:LinearAlgebra.BlasFloat}, F) = LAPACK.potri!('U', R)
invert_spd!(R::AbstractMatrix, F) = copyto!(R, inv(F))

"""
    factorize!(ls, ws) -> Bool

Rebuild the factorization for the workspace's current `ρ`. Returns `false` if the matrix
turned out not to be factorizable by this backend, which for [`ReducedCholesky`](@ref)
means it was not positive definite.
"""
function factorize!(ls::ReducedCholesky{T}, ws)::Bool where {T}
    n, m = ws.n, ws.m
    R = ls.Rinv
    # `scaled_col!` writes only the entries the matrix actually has, so W is zeroed first.
    fill!(ls.W, zero(T))
    rho, E, D = ws.rho_vec, ws.E, ws.D
    for j in 1:n
        dj = D[j]
        scaled_col!(T, ls.W, ws.A, j, (a, i) -> sqrt(rho[i]) * E[i] * a * dj)
    end
    if m > 0
        mul!(R, ls.W', ls.W)
    else
        fill!(R, zero(T))
    end
    c = ws.c
    for j in 1:n
        dj = D[j]
        add_scaled_col!(T, R, ws.P, j, (p, i) -> c * D[i] * p * dj)
    end
    for i in 1:n
        R[i, i] += ws.settings.sigma
    end
    F = cholesky!(Symmetric(R); check = false)
    issuccess(F) || return false
    invert_spd!(R, F)
    return true
end

function factorize!(ls::FullKKT{T}, ws)::Bool where {T}
    n, m = ws.n, ws.m
    fill!(ls.K, zero(T))
    for j in 1:n
        dj = ws.D[j]
        for i in 1:n
            ls.K[i, j] = ws.c * ws.D[i] * T(ws.P[i, j]) * dj
        end
        ls.K[j, j] += ws.settings.sigma
        for i in 1:m
            aij = ws.E[i] * T(ws.A[i, j]) * dj
            ls.K[n + i, j] = aij
            ls.K[j, n + i] = aij
        end
    end
    for i in 1:m
        ls.K[n + i, n + i] = -ws.rho_inv_vec[i]
    end
    F = bunchkaufman!(Symmetric(ls.K, :L); check = false)
    issuccess(F) || return false
    ls.fact = F
    return true
end

"""
    solve_system!(ls, ws, rhs_x, rhs_z) -> Nothing

Solve the subproblem, writing `x̃` into `ws.xtilde` and `z̃` into `ws.ztilde`.
"""
function solve_system!(ls::ReducedCholesky, ws, rhs_x, rhs_z)::Nothing
    m = ws.m
    # The right-hand side is assembled in `work_n`: `symv` may not alias its two vectors.
    if m > 0
        for i in eachindex(ws.work_m, ws.rho_vec, rhs_z)
            ws.work_m[i] = ws.rho_vec[i] * rhs_z[i]
        end
        mul_At!(ws.work_n, ws, ws.work_m)
        for i in eachindex(ws.work_n, rhs_x)
            ws.work_n[i] += rhs_x[i]
        end
    else
        copyto!(ws.work_n, rhs_x)
    end
    mul!(ws.xtilde, Symmetric(ls.Rinv, :U), ws.work_n)
    m > 0 && mul_A!(ws.ztilde, ws, ws.xtilde)
    return nothing
end

function solve_system!(ls::FullKKT, ws, rhs_x, rhs_z)::Nothing
    n, m = ws.n, ws.m
    # Indexed rather than `copyto!(view(...), ...)`: the views leave allocation sites that
    # AllocCheck reports, and the loops make the no-allocation property provable.
    for i in 1:n
        ls.rhs[i] = rhs_x[i]
    end
    for i in 1:m
        ls.rhs[n + i] = rhs_z[i]
    end
    ldiv!(ls.fact, ls.rhs)
    for i in 1:n
        ws.xtilde[i] = ls.rhs[i]
    end
    for i in 1:m
        ws.ztilde[i] = rhs_z[i] + ws.rho_inv_vec[i] * ls.rhs[n + i]
    end
    return nothing
end

@verify ReducedCholesky trim_compat = true
@verify FullKKT trim_compat = true

"""
    refactor!(ws)

Refresh the workspace's factorization after `ρ` or the problem data changed.

The backend is fixed at [`setup`](@ref), so that every solve dispatches statically and the
workspace stays concretely typed. If the reduced Cholesky ever fails here — which no
measured problem has produced once equilibration is on — the remedy is to rebuild the
workspace with `linsys = :kkt` rather than to switch backend underneath the caller.
"""
function refactor!(ws)
    ok = factorize!(ws.linsys, ws)
    ok || throw(
        ArgumentError(
            "the linear system could not be factorized with the $(backend_name(ws.linsys)) backend. " *
                "Rebuild the workspace with linsys = :kkt, which does not square the conditioning of A."
        )
    )
    ws.refactor_count += 1
    return ws
end

"""
    indirect_backend(proto, n, m) -> LinearSystem

Build the matrix-free backend selected by `linsys = :indirect`. Supplied by the Krylov
extension; without Krylov loaded there is no such backend and this says so.
"""
function indirect_backend(proto::AbstractVector, n::Integer, m::Integer)
    throw(
        ArgumentError(
            "linsys = :indirect needs Krylov.jl, which is a weak dependency: run " *
                "`using Krylov` before `setup`. It is not a core dependency because the " *
                "backend is only worth reaching for when the reduced matrix cannot be formed."
        )
    )
end
