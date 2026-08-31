"""
    BackendInfo

What a [`LinearSystem`](@ref) backend is and how big the object it solves through is,
reported by [`backend_info`](@ref).

- `name` — the backend's name, the same symbol [`backend_name`](@ref) returns.
- `direct` — `true` when a factorization is stored, `false` for a matrix-free iterative
  backend, whose `factor_nnz` is `0` because there is nothing stored to count.
- `system` — `:reduced` for the `n×n` system that eliminates `ν`, `:kkt` for the full
  `(n+m)×(n+m)` quasi-definite system.
- `dim` — the side length of that system: `n` when `system` is `:reduced`, `n+m` when it is
  `:kkt`.
- `factor_nnz` — the size of the stored factorization as one triangle, in that factorization's
  own convention: `nnz(L)` for a sparse factor, `dim(dim+1)/2` for a dense one, and the
  inverse's triangle for a backend that stores an inverse instead.

It is a fill measure, not a memory total, and the conventions differ in ways that matter when
comparing two backends directly: an `LDLᵀ` factor's `L` is strictly lower with a unit diagonal
held elsewhere, so it counts `dim` fewer scalars than an `LLᵀ` factor of the same matrix, and a
backend may physically hold more than it reports — `SparseCholmod` keeps `L` and `Lᵀ` both,
`FullKKT` keeps a pivot vector beside its factor.

Comparing fills across backends needs the problem's `n` rather than `dim`, because `dim` is
`n+m` for a `:kkt` backend and `n` for a `:reduced` one. [`factor_fill`](@ref) takes a
workspace and does that normalization; it is the quantity the sparse selection thresholds are
stated in.

The sparse backends' factors are empty until `factorize!` has run. [`setup`](@ref) always
factorizes, so a backend reached through a workspace is populated.
"""
struct BackendInfo
    name::Symbol
    direct::Bool
    system::Symbol
    dim::Int
    factor_nnz::Int
end

"""
    backend_info(ls::LinearSystem) -> BackendInfo

Describe the backend a workspace holds: `backend_info(ws.linsys)`.
"""
function backend_info end

"""
    factor_fill(ws) -> Float64

The stored factorization's size as a fraction of `n²`, which is how the sparse selection
thresholds are stated and the only form in which two backends' fills compare.

`BackendInfo`'s own `dim` is `n+m` for a backend solving the full KKT system and `n` for one
solving the reduced system, so normalizing by `dim²` would divide the two families by
different denominators. This takes `n` from the workspace instead.
"""
factor_fill(ws) = backend_info(ws.linsys).factor_nnz / ws.n^2

"""
    LinearSystem

Interface for the factorization that solves the ADMM subproblem

    ⎡P̃ + σI      Ãᵀ   ⎤ ⎡x̃⎤   ⎡rhs_x⎤
    ⎣Ã       −diag(ρ⁻¹)⎦ ⎣ν⎦ = ⎣rhs_z⎦

once per iteration. A backend owns its own storage and factorization object; the workspace
holds one, chosen at [`setup`](@ref) and fixed for the workspace's life, so every call
dispatches statically.

Implementations must provide the three methods below; the contract is enforced at
precompilation. [`refactor_rho!`](@ref) is optional, and rebuilding from scratch is a correct
answer to it.
"""
abstract type LinearSystem end

function factorize! end
function solve_system! end

@contract LinearSystem begin
    factorize!(::Self, ::Any)::Bool
    solve_system!(::Self, ::Any, ::Any, ::Any)::Nothing
    backend_info(::Self)::BackendInfo
end

"""
    refactor_rho!(ls, ws) -> Bool

Refresh the factorization after `ρ` moved and nothing else, returning whether it succeeded.

Separate from [`factorize!`](@ref) because the two events are not the same: `ρ` changes on
its own every time [`adapt_rho!`](@ref) fires, while `P`, `A`, `D`, `E` and `c` change only
through [`setup`](@ref) and [`update!`](@ref). A backend whose factorization is partly
independent of `ρ` can keep that part.

Rebuilding everything is correct, and is what the default does. An override may assume the
`ρ`-independent parts are current, since every path that invalidates them calls
[`refactor!`](@ref) instead.
"""
refactor_rho!(ls::LinearSystem, ws) = factorize!(ls, ws)

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
    DiagonalReduced{T,V} <: LinearSystem

The reduced system when it is diagonal, which it is when `P` and `A` both are:

    R = c D P D + σI + Ãᵀ diag(ρ) Ã

is a sum of diagonal terms, so there is nothing to factor and each solve is `n` divisions.
`dinv` holds `R`'s reciprocal diagonal.

`Ãᵀ diag(ρ) Ã` fills in for any other `A`, which is why this is keyed on `A`'s type and not
`P`'s: a `Diagonal` `P` with a general `A` still has a dense reduced matrix.
"""
struct DiagonalReduced{T <: Real, V <: AbstractVector{T}} <: LinearSystem
    dinv::V
end

"""
    DiagonalReduced(proto::AbstractVector, n)

Build the backend's storage as `similar(proto, ...)`, following the array type of the data
it was given. See [`ReducedCholesky`](@ref) on why `proto` is a vector.
"""
function DiagonalReduced(proto::AbstractVector{T}, n::Integer) where {T <: Real}
    dinv = similar(proto, T, n)
    return DiagonalReduced{T, typeof(dinv)}(dinv)
end

"""
    TridiagonalReduced{T,V,F} <: LinearSystem

The reduced system when it is tridiagonal. Diagonal scaling preserves a bandwidth and
`Ãᵀ diag(ρ) Ã` doubles `A`'s, so

    bandwidth(R) = max(bandwidth(P), 2 bandwidth(A))

which is 1 for a `SymTridiagonal` or `Tridiagonal` `P` with a `Diagonal` `A`, for a
`Diagonal` `P` with a `Bidiagonal` `A`, and for the two together. `ldlt` factors that in
`O(n)` and its `ldiv!` allocates nothing.

`dv` and `ev` hold `R`'s two bands. They are computed entry by entry rather than by forming
`c D P D + σI + Ãᵀ diag(ρ) Ã`: that product returns a dense `Array` for a `Bidiagonal` `A`
even though the result has bandwidth 1, so the structure has to be established here rather
than recovered from the arithmetic. `fdv` and `fev` are the copies `ldlt!` overwrites, which
keeps a refactorization from allocating.
"""
mutable struct TridiagonalReduced{T <: Real, V <: AbstractVector{T}, F} <: LinearSystem
    dv::V
    ev::V
    fdv::V
    fev::V
    fact::F
end

"""
    TridiagonalReduced(proto::AbstractVector, n)

Build the backend's storage as `similar(proto, ...)`, following the array type of the data
it was given. See [`ReducedCholesky`](@ref) on why `proto` is a vector.
"""
function TridiagonalReduced(proto::AbstractVector{T}, n::Integer) where {T <: Real}
    dv, ev = similar(proto, T, n), similar(proto, T, max(n - 1, 0))
    fdv, fev = similar(dv), similar(ev)
    fact = ldlt!(SymTridiagonal(fill(one(T), 1), fill(one(T), 0)))
    return TridiagonalReduced{T, typeof(dv), typeof(fact)}(dv, ev, fdv, fev, fact)
end

"""
    band_columns(A, k) -> UnitRange

The columns row `k` of `A` can hold a nonzero in. Each structured `A` the tridiagonal
backend accepts answers this in `O(1)`, which is what keeps forming `Ãᵀ diag(ρ) Ã` linear.
"""
band_columns(A::Diagonal, k::Integer) = k:k
band_columns(A::Bidiagonal, k::Integer) =
    A.uplo == 'U' ? (k:min(k + 1, size(A, 2))) : (max(k - 1, 1):k)

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

# A diagonal matrix is positive definite exactly when its diagonal is, so the test is a
# pass over `n` entries rather than a factorization of an `n×n` densification of them.
is_convex(::Type{T}, P::Diagonal, sigma) where {T} = all(d -> d + sigma > zero(T), P.diag)

# `ldlt` of a tridiagonal is `O(n)` and its pivots decide definiteness: all positive is
# positive definite, and a zero pivot throws rather than reporting.
function is_convex(::Type{T}, P::SymTridiagonal, sigma) where {T}
    isempty(P) && return true
    S = SymTridiagonal(P.dv .+ sigma, copy(P.ev))
    fact = try
        ldlt!(S)
    catch e
        e isa LinearAlgebra.ZeroPivotException || rethrow()
        return false
    end
    return all(>(zero(T)), fact.data.dv)
end

# `validate` has established that `P` is symmetric before this runs, so a `Tridiagonal`
# describes the same band as the `SymTridiagonal` built from its diagonal and superdiagonal,
# and gets the same `O(n)` test rather than the generic densification.
is_convex(::Type{T}, P::Tridiagonal, sigma) where {T} =
    is_convex(T, SymTridiagonal(diag(P), diag(P, 1)), sigma)

"""
    is_materializable(M) -> Bool

Whether `M`'s entries can be read one at a time. True unless the representation says
otherwise.

Forming the reduced matrix, [`polish!`](@ref) and the derivatives all read entries; an
operator that supplies only `mul!` declares `false` here and is refused by those paths by
name rather than by a `MethodError` from inside a factorization. Declining is a statement
by the operator's author about what it can answer, not a measured threshold.

The method body is a literal, so a call against a type with no override folds away and the
rungs that consult it stay concretely typed.
"""
is_materializable(M) = true

"""
    require_entries(P, A, what, remedy)

Throw unless both operators can be read entry by entry, naming what needs it.

[`polish!`](@ref) and the derivatives copy `P` and `A` into a dense matrix one entry at a
time and factor it, which an operator supplying only products cannot serve. Without this
the caller gets a `MethodError` from inside the copy, which says nothing about what to do.

The message names no type. Interpolating one goes through `show(::IO, ::Type)`, which is a
runtime dispatch that `--trim` cannot resolve — and for an operator that declines, the
condition folds to `false`, so this branch is live code rather than the dead one it is for
every materializable pair.
"""
function require_entries(P, A, what::String, remedy::String)
    (is_materializable(P) && is_materializable(A)) || throw(
        ArgumentError(
            "$what reads the entries of P and A one at a time, and one of them declares " *
                "`PureOSQP.is_materializable` false: it supplies products only. $remedy"
        )
    )
    return nothing
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

A `(P, A)` pair with no method of its own descends [`select_backend`](@ref)'s ladder, whose
named terminal rung is the dense reduced matrix.
"""
choose_backend(
    P, A, proto::AbstractVector, n::Integer, m::Integer, D, E, c, rho_vec, sigma
) = select_backend(P, A, proto, n, m, D, E, c, rho_vec, sigma)

"""
    select_backend(P, A, proto, n, m, D, E, c, rho_vec, sigma) -> (LinearSystem, Bool)

Descend the selection ladder, returning the first rung that serves this `(P, A)` pair and
whether that rung already carries a factorization of the current data.

The rungs, in order:

1. [`density_gate_rung`](@ref) — the reduced matrix is dense enough that assembling it from
   stored entries loses to the dense product, so the pair goes straight to the terminal.
2. [`kkt_rung`](@ref) — factor the full `(n+m)×(n+m)` quasi-definite matrix sparsely.
3. [`reduced_rung`](@ref) — factor the `n×n` reduced matrix sparsely.
4. [`lowrank_rung`](@ref) — the reduced matrix is a structured core plus a low-rank
   correction, solved through the correction rather than formed.
5. [`formed_rung`](@ref) — assemble the reduced matrix from stored entries and invert it
   densely.
6. [`dense_rung`](@ref) — form the reduced matrix densely and invert it. Every pair that can
   be materialized at all stops here.
7. [`indirect_rung`](@ref) — matrix-free, for an operator the terminal cannot materialize.

Rungs 2 and 3 decide by factoring the real equilibrated matrix and reading the fill, so the
factorization they produce is the setup factorization and they return `true`. That is why a
rung returns the backend rather than a verdict: a query answering only "does this fit" would
throw that factorization away and pay for it twice.

This ladder is not the whole of selection, and reading it alone will mislead. Three things sit
outside it:

- `linsys = :kkt`, `:dense` and `:indirect` are handled in [`setup`](@ref) before the ladder
  is reached, so a caller who names a backend never descends it. `:indirect` in particular
  reaches [`indirect_backend`](@ref) directly and not through rung 6.
- A [`choose_backend`](@ref) method for a specific `(P, A)` pair wins over this ladder by
  dispatch, which is how the structured and banded backends are chosen. The ladder is the
  body of the *fallback* method.
- Any backend that arrives unfactored and whose `factorize!` then fails — an indefinite `P`
  that `σ` does not lift, whichever rung or `choose_backend` method produced it — is rebuilt
  on [`FullKKT`](@ref) by [`setup`](@ref). That is the last word on selection, and it is not
  a rung.

Each rung is a generic function whose default declines, so an extension adds itself to the
ladder by defining the method its representation needs. The order is fixed here, in one
place, rather than emerging from where each gate happens to sit.
"""
function select_backend(
        P, A, proto::AbstractVector, n::Integer, m::Integer, D, E, c, rho_vec, sigma
    )
    rung = density_gate_rung(P, A, proto, n, m)
    isnothing(rung) || return rung
    rung = kkt_rung(P, A, proto, n, m, D, E, c, rho_vec, sigma)
    isnothing(rung) || return rung
    rung = reduced_rung(P, A, proto, n, m, D, E, c, rho_vec, sigma)
    isnothing(rung) || return rung
    rung = block_rung(P, A, proto, n, m, D, E, c, rho_vec, sigma)
    isnothing(rung) || return rung
    rung = lowrank_rung(P, A, proto, n, m, D, E, c, rho_vec, sigma)
    isnothing(rung) || return rung
    rung = formed_rung(P, A, proto, n, m)
    isnothing(rung) || return rung
    rung = dense_rung(P, A, proto, n, m)
    isnothing(rung) || return rung
    return indirect_rung(P, A, proto, n, m)
end

"""
    density_gate_rung(P, A, proto, n, m) -> (LinearSystem, Bool) or nothing

Ladder rung 1: send a pair whose stored entries are too dense for sparse assembly to pay
straight to [`dense_rung`](@ref), skipping the rungs between.

Declines for a representation with no density to measure.
"""
density_gate_rung(P, A, proto::AbstractVector, n::Integer, m::Integer) = nothing

"""
    kkt_rung(P, A, proto, n, m, D, E, c, rho_vec, sigma) -> (LinearSystem, Bool) or nothing

Ladder rung 2: factor the full quasi-definite KKT matrix sparsely, when its factor stays
sparse enough to clear the gate. Decides by factoring, so what it returns is already factored.

The gate is a fill threshold, not a comparison against the dense path: it accepts where the
sparse factor is small, which is a sufficient condition for the sparse route to win and not a
necessary one. A pair it declines is not thereby known to be better served densely.
"""
kkt_rung(P, A, proto::AbstractVector, n::Integer, m::Integer, D, E, c, rho_vec, sigma) = nothing

"""
    reduced_rung(P, A, proto, n, m, D, E, c, rho_vec, sigma) -> (LinearSystem, Bool) or nothing

Ladder rung 3: factor the reduced matrix sparsely, when its factor stays sparse. Decides by
factoring, so what it returns is already factored.
"""
reduced_rung(P, A, proto::AbstractVector, n::Integer, m::Integer, D, E, c, rho_vec, sigma) = nothing

"""
    formed_rung(P, A, proto, n, m) -> (LinearSystem, Bool) or nothing

Ladder rung 5: form the reduced matrix by accumulating over stored entries, then invert it
densely — the same dense arithmetic as [`dense_rung`](@ref) reached without the `m×n` buffer
its product needs.

Accumulating reads entries, so a method here declines an operand that answers
[`is_materializable`](@ref) with `false`, as rung 6 does.
"""
formed_rung(P, A, proto::AbstractVector, n::Integer, m::Integer) = nothing

"""
    dense_rung(P, A, proto, n, m) -> (LinearSystem, Bool) or nothing

Ladder rung 6, the terminal: [`ReducedCholesky`](@ref), which forms the reduced matrix with
one dense product and inverts it. It serves any pair of materializable matrices, which is
why every rung above it may decline freely.

Declines when either operand answers [`is_materializable`](@ref) with `false`, since forming
the product reads entries. The ladder then falls through to [`indirect_rung`](@ref).
"""
function dense_rung(
        P::AbstractMatrix, A::AbstractMatrix, proto::AbstractVector, n::Integer, m::Integer
    )
    (is_materializable(P) && is_materializable(A)) || return nothing
    return (ReducedCholesky(proto, n, m), false)
end

# Every rung declines rather than erroring on a pair it does not serve, so the ladder reaches
# its next rung instead of the caller reaching a `MethodError`. This is the terminal rung's
# share of that: an operand outside `AbstractMatrix` is served below, not here.
dense_rung(P, A, proto::AbstractVector, n::Integer, m::Integer) = nothing

"""
    indirect_rung(P, A, proto, n, m) -> (LinearSystem, Bool)

Ladder rung 7, below the terminal: conjugate gradients, which needs only products with `P`
and `A` and so serves an operator no other rung can materialize.

It has no gate: reaching it means nothing above could serve. Without Krylov loaded there is
no such backend and [`indirect_backend`](@ref) says so.
"""
indirect_rung(P, A, proto::AbstractVector, n::Integer, m::Integer) =
    (indirect_backend(proto, n, m), false)

choose_backend(
    P::Diagonal, A::Diagonal, proto::AbstractVector, n::Integer, m::Integer,
    D, E, c, rho_vec, sigma
) = (DiagonalReduced(proto, n), false)

# The pairs whose reduced matrix has bandwidth 1. `Bidiagonal` is as wide an `A` as this
# reaches: a `Tridiagonal` one squares to bandwidth 2, which no symmetric type in
# LinearAlgebra stores.
#
# A `Tridiagonal` `P` names the same band a `SymTridiagonal` one does, and `factorize!`
# reads it through `P[j, j]` and `P[j, j+1]` alone; `validate` has already established that
# `P` is symmetric, so the subdiagonal it also stores holds the same numbers.
choose_backend(
    P::Union{SymTridiagonal, Tridiagonal}, A::Diagonal, proto::AbstractVector,
    n::Integer, m::Integer, D, E, c, rho_vec, sigma
) = (TridiagonalReduced(proto, n), false)

choose_backend(
    P::Union{Diagonal, SymTridiagonal, Tridiagonal}, A::Bidiagonal, proto::AbstractVector,
    n::Integer, m::Integer, D, E, c, rho_vec, sigma
) = (TridiagonalReduced(proto, n), false)

"Name of the backend, for reporting."
backend_name(::ReducedCholesky) = :cholesky
backend_name(::FullKKT) = :bunchkaufman
backend_name(::DiagonalReduced) = :diagonal
backend_name(::TridiagonalReduced) = :tridiagonal

# `Rinv` and `K` are dense, so their triangle is what the factorization occupies. The
# diagonal and tridiagonal backends store their bands and nothing else.
dense_triangle(dim::Integer) = dim * (dim + 1) ÷ 2

function backend_info(ls::ReducedCholesky)
    dim = size(ls.Rinv, 1)
    return BackendInfo(backend_name(ls), true, :reduced, dim, dense_triangle(dim))
end

function backend_info(ls::FullKKT)
    dim = size(ls.K, 1)
    return BackendInfo(backend_name(ls), true, :kkt, dim, dense_triangle(dim))
end

backend_info(ls::DiagonalReduced) =
    BackendInfo(backend_name(ls), true, :reduced, length(ls.dinv), length(ls.dinv))

function backend_info(ls::TridiagonalReduced)
    dim = length(ls.dv)
    return BackendInfo(backend_name(ls), true, :reduced, dim, dim + length(ls.ev))
end

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

function factorize!(ls::DiagonalReduced{T}, ws)::Bool where {T}
    n, m = ws.n, ws.m
    P, A, D, E = ws.P, ws.A, ws.D, ws.E
    c, rho, sigma = ws.c, ws.rho_vec, ws.settings.sigma
    for j in 1:n
        dj = D[j]
        r = c * dj * P[j, j] * dj + sigma
        if j <= m
            a = E[j] * A[j, j] * dj
            r += rho[j] * a * a
        end
        # The reduced matrix carries `σI`, so a non-positive entry means `P` was indefinite
        # by more than `σ` absorbs -- the same condition a Cholesky reports as a failure.
        r > zero(T) || return false
        ls.dinv[j] = inv(r)
    end
    return true
end

function factorize!(ls::TridiagonalReduced{T}, ws)::Bool where {T}
    n, m = ws.n, ws.m
    P, A, D, E = ws.P, ws.A, ws.D, ws.E
    c, rho, sigma = ws.c, ws.rho_vec, ws.settings.sigma
    dv, ev = ls.dv, ls.ev
    for j in 1:n
        dv[j] = c * D[j] * P[j, j] * D[j] + sigma
    end
    for j in 1:(n - 1)
        ev[j] = c * D[j] * P[j, j + 1] * D[j + 1]
    end
    # `Ãᵀ diag(ρ) Ã` row by row: row `k` reaches only the columns in its band, so each row
    # contributes to at most two diagonal entries and one off-diagonal one.
    for k in 1:m
        w = rho[k] * E[k] * E[k]
        cols = band_columns(A, k)
        for j in cols
            akj = A[k, j] * D[j]
            dv[j] += w * akj * akj
            if j + 1 in cols
                ev[j] += w * akj * A[k, j + 1] * D[j + 1]
            end
        end
    end
    copyto!(ls.fdv, dv)
    copyto!(ls.fev, ev)
    # An `ldlt` reports neither indefiniteness nor a zero pivot the way a Cholesky does: it
    # returns a factorization with a negative pivot in the first case and throws in the
    # second. Both mean this backend cannot solve the system, so both are refused here.
    ls.fact = try
        ldlt!(SymTridiagonal(ls.fdv, ls.fev))
    catch e
        e isa LinearAlgebra.ZeroPivotException || rethrow()
        return false
    end
    return all(>(zero(T)), ls.fact.data.dv)
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

function solve_system!(ls::DiagonalReduced, ws, rhs_x, rhs_z)::Nothing
    reduced_rhs!(ws, rhs_x, rhs_z)
    multiply!(ws.xtilde, ls.dinv, ws.work_n)
    ws.m > 0 && mul_A!(ws.ztilde, ws, ws.xtilde)
    return nothing
end

function solve_system!(ls::TridiagonalReduced, ws, rhs_x, rhs_z)::Nothing
    reduced_rhs!(ws, rhs_x, rhs_z)
    copyto!(ws.xtilde, ws.work_n)
    ldiv!(ls.fact, ws.xtilde)
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
@verify DiagonalReduced trim_compat = true
@verify TridiagonalReduced trim_compat = true

"""
    refactor!(ws)

Refresh the workspace's factorization after `ρ` or the problem data changed.

The backend is fixed at [`setup`](@ref), so that every solve dispatches statically and the
workspace stays concretely typed. If the reduced Cholesky ever fails here — which no
measured problem has produced once equilibration is on — the remedy is to rebuild the
workspace with `linsys = :kkt` rather than to switch backend underneath the caller.
"""
function refactor!(ws)
    return refactored!(ws, factorize!(ws.linsys, ws))
end

"""
    refactor_rho!(ws)

Refresh the workspace's factorization after `ρ` alone changed, through the backend's
[`refactor_rho!`](@ref) rather than a full rebuild. Counted and reported like any other
refactorization.
"""
function refactor_rho!(ws)
    return refactored!(ws, refactor_rho!(ws.linsys, ws))
end

function refactored!(ws, ok::Bool)
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
