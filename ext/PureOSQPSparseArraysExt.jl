"""
    PureOSQPSparseArraysExt

Sparse-aware traversals and a sparse-forming reduced backend, for `SparseMatrixCSC`.

PureOSQP's per-iteration products already go through `mul!`, which sparse matrices handle
well on their own. Equilibration and the factorization are different: they walk the
caller's matrices entry by entry, and the generic loop visits every structural zero and
reaches each one through `M[i, j]`, which on CSC is a binary search within the column. On a
400×200 matrix at 1% density that is 80 000 searches per sweep where 800 direct reads would
do, and it made equilibration ten times slower on a sparse matrix than on a dense one.

The four column traversals below walk `nzrange` instead. The extension also supplies a
two reduced backends: [`SparseFormedInverse`](@ref), which forms `Ãᵀ diag(ρ) Ã` from the
stored entries rather than through a
dense `m×n` product but still factors densely, and [`SparseCholmod`](@ref), which also
factors sparsely when the factor stays sparse enough to pay. Nothing else in
the solver needs to know the storage.
"""
module PureOSQPSparseArraysExt

using PureOSQP: PureOSQP
using TypeContracts: TypeContracts, @verify
using LinearAlgebra: Symmetric, Diagonal, LowerTriangular, UpperTriangular, I,
    cholesky, cholesky!, issuccess, ldiv!, transpose
using SparseArrays: SparseMatrixCSC, issparse, nnz, nzrange, rowvals, nonzeros, sparse

@inline function PureOSQP.weighted_colmax(
        ::Type{T}, M::SparseMatrixCSC, j::Integer, w::AbstractVector
    ) where {T}
    r = zero(T)
    rows, vals = rowvals(M), nonzeros(M)
    for k in nzrange(M, j)
        r = max(r, w[rows[k]] * abs(T(vals[k])))
    end
    return r
end

@inline function PureOSQP.weighted_colmax_rowmax!(
        ::Type{T}, e::AbstractVector, M::SparseMatrixCSC, j::Integer,
        w::AbstractVector, s
    ) where {T}
    r = zero(T)
    rows, vals = rowvals(M), nonzeros(M)
    for k in nzrange(M, j)
        i = rows[k]
        v = abs(T(vals[k]))
        r = max(r, w[i] * v)
        e[i] = max(e[i], s * v)
    end
    return r
end

@inline function PureOSQP.scaled_col!(
        ::Type{T}, dest::AbstractMatrix, M::SparseMatrixCSC, j::Integer, f::F
    ) where {T, F}
    rows, vals = rowvals(M), nonzeros(M)
    for k in nzrange(M, j)
        i = rows[k]
        dest[i, j] = f(T(vals[k]), i)
    end
    return dest
end

@inline function PureOSQP.add_scaled_col!(
        ::Type{T}, dest::AbstractMatrix, M::SparseMatrixCSC, j::Integer, f::F
    ) where {T, F}
    rows, vals = rowvals(M), nonzeros(M)
    for k in nzrange(M, j)
        i = rows[k]
        dest[i, j] += f(T(vals[k]), i)
    end
    return dest
end


"""
    SparseFormedInverse{T,M} <: PureOSQP.ReducedInverse

The reduced backend for a sparse `A`, which forms `Ãᵀ diag(ρ) Ã` from the stored entries
instead of through a dense product.

[`PureOSQP.ReducedCholesky`](@ref) writes the scaled `Ã` into an `m×n` dense buffer so that
one `syrk` produces the reduced matrix. That buffer is mostly zeros when `A` is sparse, and
the `syrk` does `mn²` flops to multiply them: on a 2000×4000 problem at 0.25% density it is
about 63% of a refactorization and 65% of the workspace. Accumulating over the stored
entries instead costs `Σᵢ nnzᵢ²` and needs no buffer, so this backend holds only the
inverse.

The reduced matrix itself is still dense, and everything after it is forming — the
Cholesky, the inversion, the per-iteration `symv` — is exactly what the dense backend does.
"""
struct SparseFormedInverse{T <: Real, M <: AbstractMatrix{T}} <: PureOSQP.ReducedInverse
    Rinv::M
end

"""
Density above which the dense product forms the reduced matrix faster than accumulating
over stored entries, so a `SparseMatrixCSC` is served by [`PureOSQP.ReducedCholesky`](@ref)
anyway.

Accumulation does `Σᵢ nnzᵢ²` flops against the dense product's `mn²`, but it writes into `R`
by scattered index where `syrk` streams contiguous memory, and that constant is worth about
an order of magnitude. Measured, the two cross near 20% density; this is set below that, and
matches the density above which callers are already advised to hand over dense copies.
"""
const DENSE_FORM_DENSITY = 0.1

function PureOSQP.choose_backend(
        P, A::SparseMatrixCSC, proto::AbstractVector{T}, n::Integer, m::Integer
    ) where {T <: Real}
    if n > 0 && m > 0 && nnz(A) > DENSE_FORM_DENSITY * m * n
        return PureOSQP.ReducedCholesky(proto, n, m)
    end
    # Factoring sparsely beats inverting densely only when the factor stays sparse, which
    # is a property of the pattern and so is measured here rather than assumed.
    ldl = cholmod_backend(P, A, proto, n, m)
    isnothing(ldl) || return ldl
    Rinv = similar(proto, T, n, n)
    return SparseFormedInverse{T, typeof(Rinv)}(Rinv)
end

PureOSQP.backend_name(::SparseFormedInverse) = :sparse_formed

"""
    csr_rows(A) -> (rowptr, colind, nzval)

Group `A`'s stored entries by row, as `A[i, colind[p]] == nzval[p]` for
`p in rowptr[i]:(rowptr[i + 1] - 1)`, with `colind` ascending within each row.

The accumulation needs each row's nonzeros together and CSC stores columns, so something
has to transpose. `copy(transpose(A))` would, but it goes through the `SparseMatrixCSC`
constructor, whose dimension validation raises through a closure that formats its message —
enough runtime dispatch on a branch that never fires to cost `factorize!` the type-stability
guarantee. Three plain vectors carry everything the accumulation reads.
"""
function csr_rows(A::SparseMatrixCSC{Tv}) where {Tv}
    m, n = size(A)
    rows, vals = rowvals(A), nonzeros(A)
    rowptr = Vector{Int}(undef, m + 1)
    fill!(rowptr, 0)
    for k in eachindex(rows)
        rowptr[rows[k]] += 1
    end
    total = 1
    for i in 1:m
        count = rowptr[i]
        rowptr[i] = total
        total += count
    end
    rowptr[m + 1] = total
    colind = Vector{Int}(undef, length(rows))
    nzval = Vector{Tv}(undef, length(vals))
    # Columns are visited in ascending order, which is what leaves `colind` ascending
    # within each row and lets `gram_upper!` take the upper triangle as `k >= j`.
    pos = copy(rowptr)
    for j in 1:n
        for k in nzrange(A, j)
            i = rows[k]
            p = pos[i]
            colind[p] = j
            nzval[p] = vals[k]
            pos[i] = p + 1
        end
    end
    return (rowptr, colind, nzval)
end

"""
    gram_upper!(R, rowptr, colind, nzval, rho, E, D)

Add `Ãᵀ diag(ρ) Ã` to the upper triangle of `R`, where `Ã = diag(E) A diag(D)`.

Each row of `A` contributes the outer product of its own nonzeros, so the work is
`Σᵢ nnzᵢ²` rather than the `mn²` of a dense product against a mostly-zero buffer.

Only the upper triangle is written, which is all that is ever read: `cholesky!` is handed a
`Symmetric(R, :U)`, `potri!` writes the inverse into the same triangle, and the solve
multiplies by `Symmetric(Rinv, :U)`. The inner loop starts at `p` rather than at the row's
first entry because [`csr_rows`](@ref) leaves each row's columns ascending.
"""
function gram_upper!(
        R::AbstractMatrix{T}, rowptr::Vector{Int}, colind::Vector{Int},
        nzval::AbstractVector, rho::AbstractVector, E::AbstractVector, D::AbstractVector
    ) where {T}
    for i in eachindex(rho, E)
        ei = E[i]
        w = rho[i] * ei * ei
        stop = rowptr[i + 1] - 1
        for p in rowptr[i]:stop
            j = colind[p]
            wvj = w * T(nzval[p]) * D[j]
            for q in p:stop
                k = colind[q]
                R[j, k] += wvj * T(nzval[q]) * D[k]
            end
        end
    end
    return R
end

function PureOSQP.factorize!(ls::SparseFormedInverse{T}, ws)::Bool where {T}
    n, m = ws.n, ws.m
    R = ls.Rinv
    fill!(R, zero(T))
    if m > 0
        # Rebuilt rather than cached: `update!` may replace `ws.A`, and a cached grouping
        # would then describe a matrix the solver no longer holds. It costs O(nnz), against
        # the O(mn²) product it replaces.
        rowptr, colind, nzval = csr_rows(ws.A)
        gram_upper!(R, rowptr, colind, nzval, ws.rho_vec, ws.E, ws.D)
    end
    c, D = ws.c, ws.D
    for j in 1:n
        dj = D[j]
        PureOSQP.add_scaled_col!(T, R, ws.P, j, (p, i) -> c * D[i] * p * dj)
    end
    for i in 1:n
        R[i, i] += ws.settings.sigma
    end
    F = cholesky!(Symmetric(R); check = false)
    issuccess(F) || return false
    PureOSQP.invert_spd!(R, F)
    return true
end

@verify SparseFormedInverse trim_compat = true


"""
Fill fraction `nnz(L)/n²` above which a sparse factorization stops paying, so a sparse `R`
is factored densely anyway.

Measured at `n = 2000` over a bandwidth sweep: against the dense inverse's `symv`, the pair
of sparse triangular solves is 12.6× faster at a fill of 0.0025, 1.53× at 0.044, and loses
at 0.086 — crossing near 0.06. The limit sits below that crossing so the accepted region
wins on *both* the factorization and the per-iteration solve, which keeps the choice from
depending on how a particular run divides its time between the two. The factorization alone
favors sparse much further out, to 5.2× at a fill of 0.28.
"""
const DENSE_FACTOR_FILL = 0.05

"""
    SparseCholmod{T,V,F} <: PureOSQP.LinearSystem

Forms the reduced matrix sparsely *and* factors it sparsely, through SparseArrays.

`cholesky(Symmetric(R))` on a `SparseMatrixCSC` is CHOLMOD, and `cholesky!(F, R)` reuses the
symbolic analysis it already did, so every refactorization after the first pays only for the
numeric phase. The factorization is `L Lᵀ`, not `L D Lᵀ`; the solve below depends on that and
[`cholmod_backend`](@ref) checks it before selecting this backend.

This cannot be a [`PureOSQP.ReducedInverse`](@ref): that stores `R⁻¹` and solves with one
`symv`, and the inverse of a sparse matrix is dense. The Cholesky factor is kept instead and
each solve is a pair of sparse triangular solves. On a banded `R` that is the better trade by
a wide margin — at `n = 4000` a refactorization goes from 1063 ms to 0.66 ms and a solve from
1430 µs to 71 µs — and on a filled-in `R` it is worse, which [`DENSE_FACTOR_FILL`](@ref)
decides.

`L` and `perm` are extracted from the factorization rather than solved through it, because
CHOLMOD's `ldiv!` allocates a result and workspace on every call — 64 KB per solve at
`n = 2000`, which the hot path may not do. Applying the permutation and the two triangular
solves over preallocated buffers allocates nothing and measures slightly faster besides.
"""
mutable struct SparseCholmod{T <: Real, V <: AbstractVector{T}, F} <: PureOSQP.LinearSystem
    R::SparseMatrixCSC{T, Int}
    fact::F
    L::SparseMatrixCSC{T, Int}
    perm::Vector{Int}
    permuted::V
end

PureOSQP.backend_name(::SparseCholmod) = :cholmod

"""
    reduced_sparse(T, P, A, rho, E, D, c, sigma) -> SparseMatrixCSC

The reduced matrix `P̃ + σI + Ãᵀ diag(ρ) Ã`, built and kept sparse.

Equilibration comes out of the products: with `Ã = diag(E) A diag(D)` and
`P̃ = c diag(D) P diag(D)`, the matrix is `diag(D) (Aᵀ diag(ρE²) A + cP) diag(D) + σI`, so
the sandwich is applied once at the end rather than inside the sparse product.

The pattern this produces depends only on the patterns of `P` and `A`, never on the values
of `ρ`, `E`, `D`, `c` or `σ`, which are positive. That is what lets the backend be chosen at
setup, before equilibration has run, and what lets every later refactorization reuse the
symbolic factorization.
"""
function reduced_sparse(::Type{T}, P, A, rho, E, D, c, sigma) where {T}
    w = rho .* E .^ 2
    inner = transpose(A) * (Diagonal(w) * A) + c * P
    Dg = Diagonal(D)
    return SparseMatrixCSC{T, Int}(Dg * inner * Dg + sigma * I)
end

function PureOSQP.factorize!(ls::SparseCholmod{T}, ws)::Bool where {T}
    R = reduced_sparse(T, ws.P, ws.A, ws.rho_vec, ws.E, ws.D, ws.c, ws.settings.sigma)
    if R.colptr == ls.R.colptr && R.rowval == ls.R.rowval
        # Same pattern, so the symbolic factorization still describes it and only the
        # numeric values need redoing. This is the case every time `ρ` moves.
        cholesky!(ls.fact, Symmetric(R); check = false)
    else
        # `update!` replaced P or A with a differently shaped matrix.
        ls.fact = cholesky(Symmetric(R); check = false)
    end
    issuccess(ls.fact) || return false
    ls.R = R
    # `F.L` is defined only for an LLᵀ factorization. `choose_backend` selects this backend
    # only after checking that CHOLMOD produces one for this pattern, and the pattern is
    # what decides it, so this holds for the workspace's life.
    # Asserted, not assumed: `getproperty` on a CHOLMOD factor branches on the symbol and
    # is not inferrable, and an unannotated result costs `factorize!` type stability.
    ls.L = sparse(ls.fact.L)::SparseMatrixCSC{T, Int}
    ls.perm = ls.fact.p::Vector{Int}
    return true
end

function PureOSQP.solve_system!(ls::SparseCholmod{T}, ws, rhs_x, rhs_z)::Nothing where {T}
    rhs = PureOSQP.reduced_rhs!(ws, rhs_x, rhs_z)
    perm, work = ls.perm, ls.permuted
    # R[perm, perm] = L Lᵀ, so the solve is a permutation, two triangular solves, and the
    # inverse permutation -- all over buffers this backend owns.
    for i in eachindex(perm)
        work[i] = rhs[perm[i]]
    end
    ldiv!(LowerTriangular(ls.L), work)
    ldiv!(UpperTriangular(transpose(ls.L)), work)
    x = ws.xtilde
    for i in eachindex(perm)
        x[perm[i]] = work[i]
    end
    ws.m > 0 && PureOSQP.mul_A!(ws.ztilde, ws, x)
    return nothing
end

"""
    cholmod_backend(P, A, proto, n, m) -> SparseCholmod or nothing

Build the sparse-factorization backend if it is the right one for these matrices, and
return `nothing` if it is not.

The decision is measured rather than guessed at: CHOLMOD is asked to factor the reduced
matrix's pattern, and the fill it reports settles it. That costs one factorization, which is
not wasted when the answer is yes — it *is* the first factorization, whose symbolic part
every later one reuses. When the answer is no the cost is bounded by a pre-screen on
`nnz(R)`, since `L` is at least as dense as `R`'s triangle.
"""
function cholmod_backend(P, A, proto::AbstractVector{T}, n::Integer, m::Integer) where {T <: Real}
    (issparse(P) && n > 0) || return nothing
    ones_m = fill(one(T), m)
    ones_n = fill(one(T), n)
    R = reduced_sparse(T, P, A, ones_m, ones_m, ones_n, one(T), one(T))
    nnz(R) < 2 * DENSE_FACTOR_FILL * n^2 || return nothing
    F = cholesky(Symmetric(R); check = false)
    issuccess(F) || return nothing
    # An LDLᵀ factorization has no `F.L`, and this backend's solve assumes `L Lᵀ`.
    L = try
        sparse(F.L)
    catch
        return nothing
    end
    nnz(L) < DENSE_FACTOR_FILL * n^2 || return nothing
    return SparseCholmod{T, typeof(proto), typeof(F)}(R, F, L, F.p, similar(proto, T, n))
end

# No `trim_compat` claim: the solve reaches CHOLMOD through `ccall`, and the trim entry
# points cover the dense path.
@verify SparseCholmod

end # module PureOSQPSparseArraysExt
