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
reduced backend that forms `Ãᵀ diag(ρ) Ã` from the stored entries rather than through a
dense `m×n` product -- see [`SparseReducedCholesky`](@ref). Nothing else in
the solver needs to know the storage.
"""
module PureOSQPSparseArraysExt

using PureOSQP: PureOSQP
using TypeContracts: TypeContracts, @verify
using LinearAlgebra: Symmetric, cholesky!, issuccess
using SparseArrays: SparseMatrixCSC, nnz, nzrange, rowvals, nonzeros

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
    SparseReducedCholesky{T,M} <: PureOSQP.ReducedInverse

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
struct SparseReducedCholesky{T <: Real, M <: AbstractMatrix{T}} <: PureOSQP.ReducedInverse
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
    Rinv = similar(proto, T, n, n)
    return SparseReducedCholesky{T, typeof(Rinv)}(Rinv)
end

PureOSQP.backend_name(::SparseReducedCholesky) = :sparse_cholesky

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

function PureOSQP.factorize!(ls::SparseReducedCholesky{T}, ws)::Bool where {T}
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

@verify SparseReducedCholesky trim_compat = true

end # module PureOSQPSparseArraysExt
