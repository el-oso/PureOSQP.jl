@inline function limit_scaling(v::T) where {T}
    return v < MIN_SCALING(T) ? one(T) : min(v, MAX_SCALING(T))
end

# ── column traversals ───────────────────────────────────────────────────────────────────
# Equilibration and the factorization both walk the caller's matrices one column at a time.
# These three functions are the only places that do, so a matrix type that can enumerate a
# column faster than by index needs to override just them. `ext/PureOSQPSparseArraysExt.jl`
# does exactly that for `SparseMatrixCSC`, where indexing `M[i, j]` is a binary search and
# the generic loop visits every structural zero.

"`max(w[i] * |M[i, j]|)` over the column."
@inline function weighted_colmax(::Type{T}, M::AbstractMatrix, j::Integer, w::AbstractVector) where {T}
    r = zero(T)
    for i in axes(M, 1)
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
    for i in axes(M, 1)
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
    for i in axes(M, 1)
        dest[i, j] = f(T(M[i, j]), i)
    end
    return dest
end

"`dest[i, j] += f(M[i, j], i)` over the column."
@inline function add_scaled_col!(
        ::Type{T}, dest::AbstractMatrix, M::AbstractMatrix, j::Integer, f::F
    ) where {T, F}
    for i in axes(M, 1)
        dest[i, j] += f(T(M[i, j]), i)
    end
    return dest
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
        # One pass over A, column by column. The row norms are accumulated here rather
        # than gathered in a second loop over `A[i, j]` with `j` innermost: that walks a
        # column-major matrix across its rows, and on a 400x200 problem the strided reads
        # cost more than everything else in setup put together.
        fill!(e, zero(T))
        for j in 1:n
            pj = weighted_colmax(T, P, j, D)
            dj = D[j]
            aj = weighted_colmax_rowmax!(T, e, A, j, E, dj)
            d[j] = limit_scaling(max(ws.c * dj * pj, dj * aj))
        end
        for i in 1:m
            e[i] = limit_scaling(E[i] * e[i])
        end
        d .= inv.(sqrt.(d))
        e .= inv.(sqrt.(e))
        ws.D .*= d
        ws.E .*= e
        ws.q .*= d
        # Cost normalization: average column ∞-norm of the scaled P, against ‖q̃‖∞.
        acc = zero(T)
        for j in 1:n
            acc += ws.c * D[j] * weighted_colmax(T, P, j, D)
        end
        ct = max(acc / n, limit_scaling(maximum(abs, ws.q; init = zero(T))))
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
    for i in eachindex(ws.tmp_n, ws.D, x)
        ws.tmp_n[i] = ws.D[i] * x[i]
    end
    mul!(out, ws.A, ws.tmp_n)
    # Written as a loop, not `out .*= ws.E`: an in-place broadcast has `out` on both sides,
    # which leaves an `unaliascopy` branch that AllocCheck reports as a possible allocation
    # even though it never fires. The loop makes the no-allocation property provable.
    for i in eachindex(out, ws.E)
        out[i] *= ws.E[i]
    end
    return out
end

"""
    mul_At!(out, ws, y)

`out = Ãᵀ y = D ⊙ (Aᵀ(E ⊙ y))`, using the caller's `A` unchanged.

`out` must not alias `ws.tmp_m`, which is used as scratch.
"""
function mul_At!(out::AbstractVector{T}, ws::Workspace{T}, y::AbstractVector{T}) where {T}
    for i in eachindex(ws.tmp_m, ws.E, y)
        ws.tmp_m[i] = ws.E[i] * y[i]
    end
    mul!(out, ws.A', ws.tmp_m)
    for i in eachindex(out, ws.D)
        out[i] *= ws.D[i]
    end
    return out
end

"""
    mul_P!(out, ws, x)

`out = P̃ x = c (D ⊙ (P (D ⊙ x)))`, using the caller's `P` unchanged.

`out` must not alias `ws.tmp_n`, which is used as scratch.
"""
function mul_P!(out::AbstractVector{T}, ws::Workspace{T}, x::AbstractVector{T}) where {T}
    for i in eachindex(ws.tmp_n, ws.D, x)
        ws.tmp_n[i] = ws.D[i] * x[i]
    end
    mul!(out, ws.P, ws.tmp_n)
    for i in eachindex(out, ws.D)
        out[i] *= ws.D[i] * ws.c
    end
    return out
end
