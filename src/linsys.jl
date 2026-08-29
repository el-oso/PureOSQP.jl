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
    ReducedInverse <: LinearSystem

Backends that eliminate `ν` and solve the reduced `n×n` system by multiplying against its
stored inverse.

They differ only in how the reduced matrix is *formed*, which is where the representation
of `A` matters; once formed, the factorization, the inversion and the per-iteration `symv`
are the same work regardless. `solve_system!` is therefore written once, here.
"""
abstract type ReducedInverse <: LinearSystem end

"""
    ReducedCholesky{T,M} <: ReducedInverse

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
struct ReducedCholesky{T <: Real, M <: AbstractMatrix{T}} <: ReducedInverse
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

"""
    is_convex(T, P, sigma) -> Bool

Whether `P + σI` is positive definite, which is what OSQP requires of `P` — not merely that
it be positive semidefinite. A positive semidefinite `P` always passes; only an indefinite
one fails. Without the check the reduced matrix `P + σI + Ãᵀ diag(ρ) Ã` can still factor and
an indefinite `P` would be accepted silently.

The generic method densifies, because a factorization it can rely on for an arbitrary
`AbstractMatrix` is the dense one. That is `O(n³)` and `O(n²)` in memory whatever `P` was,
so a representation with a cheaper test overrides this — `ext/PureOSQPSparseArraysExt.jl`
does, where the dense test measures 93× slower at `n = 2000`.
"""
function is_convex(::Type{T}, P::AbstractMatrix, sigma) where {T}
    isempty(P) && return true
    return issuccess(cholesky!(Symmetric(Matrix{T}(P) + sigma * I); check = false))
end

"""
    choose_backend(P, A, proto, n, m, D, E, c, rho_vec, sigma) -> (LinearSystem, Bool)

The backend `linsys = :auto` builds for these matrices, and whether it already carries a
factorization of the current data.

Dispatching on `typeof(P)` and `typeof(A)` is the point: a representation that admits a
cheaper way to form the reduced matrix is served by adding a method here rather than by
branching inside `factorize!`. The choice is made once, and the backend then becomes part
of the workspace's type, so the per-iteration solve still dispatches statically.

The equilibration factors and `ρ` are passed in because deciding well means building the
reduced matrix and reading the fill its factorization produces — and once that is done with
the values the solver will actually use, the factorization is the setup factorization. A
method that works that way returns `true` and [`setup`](@ref) does not factor again; one
that only picks a representation returns `false`.

The default forms the reduced matrix densely, which suits a dense or unrecognized `A`, and
leaves the factorizing to `factorize!`.
"""
choose_backend(
    P, A, proto::AbstractVector, n::Integer, m::Integer, D, E, c, rho_vec, sigma
) = (ReducedCholesky(proto, n, m), false)

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
    reduced_rhs!(ws, rhs_x, rhs_z) -> ws.work_n

Assemble `rhs_x + Ãᵀ(ρ ⊙ rhs_z)`, the right-hand side of the reduced system.

Written into `work_n` rather than over an argument because the solves that consume it may
not alias their input and output — `symv` in particular.
"""
function reduced_rhs!(ws, rhs_x, rhs_z)
    if ws.m > 0
        multiply!(ws.work_m, ws.rho_vec, rhs_z)
        mul_At!(ws.work_n, ws, ws.work_m)
        increment!(ws.work_n, rhs_x)
    else
        copyto!(ws.work_n, rhs_x)
    end
    return ws.work_n
end

"""
    solve_system!(ls, ws, rhs_x, rhs_z) -> Nothing

Solve the subproblem, writing `x̃` into `ws.xtilde` and `z̃` into `ws.ztilde`.
"""
function solve_system!(ls::ReducedInverse, ws, rhs_x, rhs_z)::Nothing
    reduced_rhs!(ws, rhs_x, rhs_z)
    mul!(ws.xtilde, Symmetric(ls.Rinv, :U), ws.work_n)
    ws.m > 0 && mul_A!(ws.ztilde, ws, ws.xtilde)
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

"""
    ldl_backend(gram, proto, n, fill_limit) -> LinearSystem or nothing

A backend that factors the already-assembled reduced matrix `gram.R` with an `LDLᵀ` other
than the one SparseArrays supplies, or `nothing` if no such factorization is available or
its fill exceeds `fill_limit * n^2`.

The reduced matrix is passed in already built, so the extension answering this needs to know
nothing about how it was assembled — and the sparse extension, in turn, needs no dependency
on whatever does the factoring.

Only the factorization is delegated. The substitutions and the diagonal scaling on the
per-iteration path stay in this package, over whatever `L` and `D` the backend exposes,
because they are as fast as any library's and they carry the allocation guarantee.
"""
ldl_backend(gram, proto::AbstractVector, n::Integer, fill_limit::Real) = nothing

"""
    ldl_kkt_backend(K, proto, n, m, fill_limit) -> LinearSystem or nothing

The same delegation for the full quasi-definite KKT matrix `K`, which is factored `LDLᵀ`
without pivoting because a quasi-definite matrix admits one under any symmetric permutation.

Separate from [`ldl_backend`](@ref) because the solve differs, not the factorization: the
full system yields `z̃` from the eliminated multiplier where the reduced one recovers it with
a product against `A`.
"""
ldl_kkt_backend(gram, proto::AbstractVector, n::Integer, m::Integer, fill_limit::Real) = nothing

"""
    require_host(v, what)

Throw unless `v` is host memory, naming what needs it.

`polish!` and the derivatives build a dense `(n+k)×(n+k)` matrix and factor it with
`bunchkaufman!`, which has no GPU implementation. Without this the caller gets
`GPUArraysCore`'s scalar-indexing error from somewhere inside the factorization, which says
nothing about what to do.
"""
function require_host(v::AbstractVector, what::String)
    v isa Vector || throw(
        ArgumentError(
            "$what runs on the host and this workspace holds $(typeof(v)): it factors a " *
                "dense matrix with `bunchkaufman!`, which has no GPU counterpart. Move the " *
                "problem to the host with `Array`, or leave `polish = false` and take the " *
                "ADMM iterate."
        )
    )
    return nothing
end
