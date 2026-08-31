@inline function limit_scaling(v::T) where {T}
    return v < MIN_SCALING(T) ? one(T) : min(v, MAX_SCALING(T))
end

# ── column traversals ───────────────────────────────────────────────────────────────────
# Equilibration and the dense formation both walk the caller's matrices one column at a
# time. These four functions are the only places that do, so a matrix type that can
# enumerate a column faster than by index needs to override just them, or just
# `structural_rows`. `ext/PureOSQPSparseArraysExt.jl` overrides all four for
# `SparseMatrixCSC`, where indexing `M[i, j]` is a binary search and the generic loop visits
# every structural zero.
#
# All four skip entries the matrix cannot have, which assumes `f(0, i) == 0` for the
# functions passed to `scaled_col!` and `add_scaled_col!`, and that the running maxima start
# at zero. Both hold for every caller here, and the sparse overrides already depend on it.

"""
    structural_rows(M, j) -> iterable of row indices

The rows in which column `j` of `M` can hold a nonzero, in increasing order and without
repeats.

Every row, in general. A banded matrix has a much shorter answer, and taking the generic one
is what makes a `Diagonal` `P` cost `O(n)` per column rather than `O(1)` — enough on its own
to make structured storage slower than dense.

The iterator's state must be concretely typed: this runs on the equilibration path, where a
dynamic dispatch costs an order of magnitude and fails `--trim`.

A range for every representation whose nonzeros are contiguous down a column, which is most
of them; [`RowCoupled`](@ref) is the exception, since its dense rows sit above scattered
single-entry ones.
"""
@inline structural_rows(M::AbstractMatrix, j::Integer) = axes(M, 1)
@inline structural_rows(M::Diagonal, j::Integer) = j:j
@inline function structural_rows(M::Bidiagonal, j::Integer)
    lo, hi = firstindex(M, 1), lastindex(M, 1)
    return M.uplo == 'U' ? (max(lo, j - 1):min(hi, j)) : (max(lo, j):min(hi, j + 1))
end
@inline function structural_rows(M::Union{Tridiagonal, SymTridiagonal}, j::Integer)
    lo, hi = firstindex(M, 1), lastindex(M, 1)
    return max(lo, j - 1):min(hi, j + 1)
end

"`max(w[i] * |M[i, j]|)` over the column."
@inline function weighted_colmax(::Type{T}, M::AbstractMatrix, j::Integer, w::AbstractVector) where {T}
    r = zero(T)
    for i in structural_rows(M, j)
        r = max(r, w[i] * abs(T(M[i, j])))
    end
    return r
end

"""
    weighted_colmax_rowmax!(T, e, M, j, w, s) -> max(w[i] * |M[i,j]|)

The column maximum, and at the same time `e[i] = max(e[i], s * |M[i,j]|)` — the row norms
accumulated in the same pass rather than gathered by a second traversal.
"""
@inline function weighted_colmax_rowmax!(
        ::Type{T}, e::AbstractVector, M::AbstractMatrix, j::Integer,
        w::AbstractVector, s
    ) where {T}
    r = zero(T)
    for i in structural_rows(M, j)
        v = abs(T(M[i, j]))
        r = max(r, w[i] * v)
        e[i] = max(e[i], s * v)
    end
    return r
end

"`dest[i, j] = f(M[i, j], i)` over the column, with `dest` already zeroed."
@inline function scaled_col!(
        ::Type{T}, dest::AbstractMatrix, M::AbstractMatrix, j::Integer, f::F
    ) where {T, F}
    for i in structural_rows(M, j)
        dest[i, j] = f(T(M[i, j]), i)
    end
    return dest
end

"`dest[i, j] += f(M[i, j], i)` over the column."
@inline function add_scaled_col!(
        ::Type{T}, dest::AbstractMatrix, M::AbstractMatrix, j::Integer, f::F
    ) where {T, F}
    for i in structural_rows(M, j)
        dest[i, j] += f(T(M[i, j]), i)
    end
    return dest
end

"""
    column_norms!(d, e, T, pcol, A, D, E, c)

One Ruiz sweep's column measurements: `d[j]` becomes
`max(c·D[j]·pcol[j], D[j]·‖E ⊙ A[:,j]‖∞)` and `e` accumulates the row norms of the scaled
`A` in the same pass.

`pcol` holds `‖D ⊙ P[:,j]‖∞` for the current `D`, computed by [`cost_norms!`](@ref) at the
end of the previous sweep. `D` does not change between the two, so recomputing it here would
be a second full pass over `P` for the same numbers.

The row norms are accumulated here rather than gathered in a second loop over `A[i, j]` with
`j` innermost: that walks a column-major matrix across its rows, and on a 400×200 problem
the strided reads cost more than everything else in setup put together.
"""
function column_norms!(d, e, ::Type{T}, pcol, A, D, E, c) where {T}
    fill!(e, zero(T))
    for j in eachindex(d)
        dj = D[j]
        aj = weighted_colmax_rowmax!(T, e, A, j, E, dj)
        d[j] = limit_scaling(max(c * dj * pcol[j], dj * aj))
    end
    return d
end

"""
    cost_norms!(pcol, T, P, D, c, n) -> mean column norm

Fill `pcol[j]` with `‖D ⊙ P[:,j]‖∞` and return their `c`-weighted mean, which is what the
cost normalization compares against `‖q̃‖∞`.

One pass serves two purposes: the mean, and the column norms the next sweep needs.

A representation that cannot be indexed overrides this with whole-matrix reductions — see
`ext/PureOSQPGPUArraysCoreExt.jl`.
"""
function cost_norms!(pcol, ::Type{T}, P, D, c, n) where {T}
    acc = zero(T)
    for j in 1:n
        pj = weighted_colmax(T, P, j, D)
        pcol[j] = pj
        acc += c * D[j] * pj
    end
    return acc / n
end

"""
    equilibrate!(T, P, A, q0, l0, u0, q, l, u, D, E, d, e, pcol, n, sweeps) -> c

Run modified Ruiz equilibration: `D` and `E` become the column and row factors, `q`, `l` and
`u` the scaled data, and the returned `c` the cost factor.

Takes the arrays rather than a [`Workspace`](@ref) because the factors must exist before one
does. The backend is part of the workspace's type, so it is chosen first, and it is chosen by
building and factoring the reduced matrix the solver will actually use — which needs `D`, `E`
and `c`. `d`, `e` and `pcol` are scratch of length `n`, `m` and `n`.
"""
function equilibrate!(
        ::Type{T}, P, A, q0, l0, u0, q, l, u, D, E, d, e, pcol, n, sweeps
    ) where {T}
    fill!(D, one(T))
    fill!(E, one(T))
    c = one(T)
    copyto!(q, q0)
    copyto!(l, l0)
    copyto!(u, u0)
    if sweeps <= 0
        return c
    end
    # Seeds `pcol` for the first sweep; every later one gets it from the cost normalization
    # at the end of the sweep before, which reads `P` with the same `D`.
    cost_norms!(pcol, T, P, D, c, n)
    for _ in 1:sweeps
        column_norms!(d, e, T, pcol, A, D, E, c)
        e .= limit_scaling.(E .* e)
        d .= inv.(sqrt.(d))
        e .= inv.(sqrt.(e))
        D .*= d
        E .*= e
        q .*= d
        # Cost normalization: average column ∞-norm of the scaled P, against ‖q̃‖∞.
        ct = max(
            cost_norms!(pcol, T, P, D, c, n),
            limit_scaling(maximum(abs, q; init = zero(T))),
        )
        ct = inv(limit_scaling(ct))
        q .*= ct
        c *= ct
    end
    l .= E .* l0
    u .= E .* u0
    return c
end


"""
    mul_A!(out, ws, x)

`out = Ã x = E ⊙ (A (D ⊙ x))`, using the caller's `A` unchanged.

`out` must not alias `ws.tmp_n`, which is used as scratch.
"""
function mul_A!(out::AbstractVector{T}, ws::Workspace{T}, x::AbstractVector{T}) where {T}
    multiply!(ws.tmp_n, ws.D, x)
    mul!(out, ws.A, ws.tmp_n)
    # `scale_by!` rather than `out .*= ws.E`: an in-place broadcast has `out` on both
    # sides, which leaves an `unaliascopy` branch that AllocCheck reports as a possible
    # allocation even though it never fires. See src/elementwise.jl.
    scale_by!(out, ws.E)
    return out
end

"""
    mul_At!(out, ws, y)

`out = Ãᵀ y = D ⊙ (Aᵀ(E ⊙ y))`, using the caller's `A` unchanged.

`out` must not alias `ws.tmp_m`, which is used as scratch.
"""
function mul_At!(out::AbstractVector{T}, ws::Workspace{T}, y::AbstractVector{T}) where {T}
    multiply!(ws.tmp_m, ws.E, y)
    mul!(out, ws.A', ws.tmp_m)
    scale_by!(out, ws.D)
    return out
end

# `mul!` against a `Tridiagonal`'s adjoint allocates in LinearAlgebra, which on a path that
# runs every iteration is the difference between holding the no-allocation guarantee and
# losing it. The three bands give `Aᵀ t` directly: `(Aᵀt)[j] = d[j]t[j] + dl[j]t[j+1] +
# du[j-1]t[j-1]`.
function mul_At!(
        out::AbstractVector{T}, ws::Workspace{T, <:AbstractMatrix, <:Tridiagonal},
        y::AbstractVector{T}
    ) where {T}
    multiply!(ws.tmp_m, ws.E, y)
    A, t, n = ws.A, ws.tmp_m, ws.n
    dl, d, du = A.dl, A.d, A.du
    # Summed in ascending `i`, which is the order `mul!` against the adjoint uses. Any other
    # order rounds differently, and a representation is supposed to change how the entries
    # are reached, not what comes out.
    for j in 1:n
        v = j > 1 ? du[j - 1] * t[j - 1] : zero(T)
        v += d[j] * t[j]
        j < n && (v += dl[j] * t[j + 1])
        out[j] = v
    end
    scale_by!(out, ws.D)
    return out
end

"""
    mul_P!(out, ws, x)

`out = P̃ x = c (D ⊙ (P (D ⊙ x)))`, using the caller's `P` unchanged.

`out` must not alias `ws.tmp_n`, which is used as scratch.
"""
function mul_P!(out::AbstractVector{T}, ws::Workspace{T}, x::AbstractVector{T}) where {T}
    multiply!(ws.tmp_n, ws.D, x)
    mul!(out, ws.P, ws.tmp_n)
    scale_by!(out, ws.D, ws.c)
    return out
end

"""
    reduced_diagonal!(dest, T, P, A, rho, E, D, sigma, c)

The diagonal of the reduced matrix, `c·D[j]²·P[j,j] + σ + Σᵢ ρᵢ(E[i]·A[i,j]·D[j])²`.

The matrix-free backend preconditions with this, and it is the one thing that backend needs
from `P` and `A` other than their products. The row sum follows
[`structural_rows`](@ref)`(A, j)`, so a declared structure costs only its own entries; a
representation that cannot be indexed at all overrides this function with whole-matrix
reductions.
"""
function reduced_diagonal!(dest, ::Type{T}, P, A, rho, E, D, sigma, c) where {T}
    # `rho` and `E` are workspace vectors indexed by the same `i` that indexes `A`'s rows, so
    # a representation whose rows are not counted from one would read the wrong weights.
    Base.require_one_based_indexing(rho, E)
    for j in eachindex(dest)
        dj = D[j]
        d = c * dj * T(P[j, j]) * dj + sigma
        for i in structural_rows(A, j)
            a = E[i] * T(A[i, j]) * dj
            d += rho[i] * a * a
        end
        dest[j] = inv(max(d, sqrt(eps(T))))
    end
    return dest
end
