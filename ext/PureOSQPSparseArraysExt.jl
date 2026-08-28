"""
    PureOSQPSparseArraysExt

Column traversals for `SparseMatrixCSC`.

PureOSQP's per-iteration products already go through `mul!`, which sparse matrices handle
well on their own. Equilibration and the factorization are different: they walk the
caller's matrices entry by entry, and the generic loop visits every structural zero and
reaches each one through `M[i, j]`, which on CSC is a binary search within the column. On a
400×200 matrix at 1% density that is 80 000 searches per sweep where 800 direct reads would
do, and it made equilibration ten times slower on a sparse matrix than on a dense one.

These four methods walk `nzrange` instead. They are the whole extension; nothing else in
the solver needs to know the storage.
"""
module PureOSQPSparseArraysExt

using PureOSQP: PureOSQP
using SparseArrays: SparseMatrixCSC, nzrange, rowvals, nonzeros

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

end # module PureOSQPSparseArraysExt
