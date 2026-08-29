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
    structural_rows(M, j) -> AbstractUnitRange

The rows in which column `j` of `M` can hold a nonzero.

Every row, in general. A banded matrix has a much shorter answer, and taking the generic one
is what makes a `Diagonal` `P` cost `O(n)` per column rather than `O(1)` — the reason
structured storage measured slower than dense before these methods existed.
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
    column_norms!(d, e, T, P, A, D, E, c)

One Ruiz sweep's column measurements: `d[j]` becomes
`max(c·D[j]·‖D ⊙ P[:,j]‖∞, D[j]·‖E ⊙ A[:,j]‖∞)` and `e` accumulates the row norms of the
scaled `A` in the same pass.

The row norms are accumulated here rather than gathered in a second loop over `A[i, j]`
with `j` innermost: that walks a column-major matrix across its rows, and on a 400×200
problem the strided reads cost more than everything else in setup put together.

A representation that cannot be indexed answers this with whole-matrix reductions instead —
see `ext/PureOSQPGPUArraysCoreExt.jl`.
"""
function column_norms!(d, e, ::Type{T}, P, A, D, E, c) where {T}
    fill!(e, zero(T))
    for j in eachindex(d)
        pj = weighted_colmax(T, P, j, D)
        dj = D[j]
        aj = weighted_colmax_rowmax!(T, e, A, j, E, dj)
        d[j] = limit_scaling(max(c * dj * pj, dj * aj))
    end
    return d
end

"""
    mean_column_norm(T, P, D, c, n)

The average column ∞-norm of the scaled `P`, which is what the cost normalization compares
against `‖q̃‖∞`.
"""
function mean_column_norm(::Type{T}, P, D, c, n) where {T}
    acc = zero(T)
    for j in 1:n
        acc += c * D[j] * weighted_colmax(T, P, j, D)
    end
    return acc / n
end

"""
    scale!(ws)

Modified Ruiz equilibration of the KKT matrix `[P Aᵀ; A 0]`, storing the result as the
factors `D`, `E` and the cost scaling `c` rather than modifying `P` and `A`. The scaled
problem is

    P̃ = c D P D,   Ã = E A D,   q̃ = c D q,   l̃ = E l,   ũ = E u

Column and row norms of the scaled blocks are evaluated from the original entries and the
running factors, so no scaled matrix is ever formed.
"""
function scale!(ws::Workspace{T}) where {T}
    n, m = ws.n, ws.m
    fill!(ws.D, one(T))
    fill!(ws.E, one(T))
    ws.c = one(T)
    copyto!(ws.q, ws.q0)
    copyto!(ws.l, ws.l0)
    copyto!(ws.u, ws.u0)
    if ws.settings.scaling <= 0
        return ws
    end
    d = ws.tmp_n
    e = ws.tmp_m
    P, A, D, E = ws.P, ws.A, ws.D, ws.E
    for _ in 1:ws.settings.scaling
        column_norms!(d, e, T, P, A, D, E, ws.c)
        e .= limit_scaling.(E .* e)
        d .= inv.(sqrt.(d))
        e .= inv.(sqrt.(e))
        ws.D .*= d
        ws.E .*= e
        ws.q .*= d
        # Cost normalization: average column ∞-norm of the scaled P, against ‖q̃‖∞.
        ct = max(
            mean_column_norm(T, P, D, ws.c, n),
            limit_scaling(maximum(abs, ws.q; init = zero(T))),
        )
        ct = inv(limit_scaling(ct))
        ws.q .*= ct
        ws.c *= ct
    end
    ws.l .= ws.E .* ws.l0
    ws.u .= ws.E .* ws.u0
    return ws
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
from `P` and `A` other than their products. A representation that cannot be indexed
overrides it with whole-matrix reductions.
"""
function reduced_diagonal!(dest, ::Type{T}, P, A, rho, E, D, sigma, c) where {T}
    for j in eachindex(dest)
        dj = D[j]
        d = c * dj * T(P[j, j]) * dj + sigma
        for i in eachindex(rho, E)
            a = E[i] * T(A[i, j]) * dj
            d += rho[i] * a * a
        end
        dest[j] = inv(max(d, sqrt(eps(T))))
    end
    return dest
end
