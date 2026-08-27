@inline function limit_scaling(v::T) where {T}
    return v < MIN_SCALING(T) ? one(T) : min(v, MAX_SCALING(T))
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
    for _ in 1:ws.settings.scaling
        for j in 1:n
            pj = zero(T)
            for i in 1:n
                pj = max(pj, ws.D[i] * abs(T(ws.P[i, j])))
            end
            aj = zero(T)
            for i in 1:m
                aj = max(aj, ws.E[i] * abs(T(ws.A[i, j])))
            end
            d[j] = limit_scaling(max(ws.c * ws.D[j] * pj, ws.D[j] * aj))
        end
        for i in 1:m
            ai = zero(T)
            for j in 1:n
                ai = max(ai, ws.D[j] * abs(T(ws.A[i, j])))
            end
            e[i] = limit_scaling(ws.E[i] * ai)
        end
        d .= inv.(sqrt.(d))
        e .= inv.(sqrt.(e))
        ws.D .*= d
        ws.E .*= e
        ws.q .*= d
        # Cost normalization: average column ∞-norm of the scaled P, against ‖q̃‖∞.
        acc = zero(T)
        for j in 1:n
            pj = zero(T)
            for i in 1:n
                pj = max(pj, ws.D[i] * abs(T(ws.P[i, j])))
            end
            acc += ws.c * ws.D[j] * pj
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
    ws.tmp_n .= ws.D .* x
    mul!(out, ws.A, ws.tmp_n)
    out .*= ws.E
    return out
end

"""
    mul_At!(out, ws, y)

`out = Ãᵀ y = D ⊙ (Aᵀ(E ⊙ y))`, using the caller's `A` unchanged.

`out` must not alias `ws.tmp_m`, which is used as scratch.
"""
function mul_At!(out::AbstractVector{T}, ws::Workspace{T}, y::AbstractVector{T}) where {T}
    ws.tmp_m .= ws.E .* y
    mul!(out, ws.A', ws.tmp_m)
    out .*= ws.D
    return out
end

"""
    mul_P!(out, ws, x)

`out = P̃ x = c (D ⊙ (P (D ⊙ x)))`, using the caller's `P` unchanged.

`out` must not alias `ws.tmp_n`, which is used as scratch.
"""
function mul_P!(out::AbstractVector{T}, ws::Workspace{T}, x::AbstractVector{T}) where {T}
    ws.tmp_n .= ws.D .* x
    mul!(out, ws.P, ws.tmp_n)
    out .*= ws.D
    out .*= ws.c
    return out
end
