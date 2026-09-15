"""
    residuals_at!(prob, x, y, z, Ax, Px, Aty) -> (prim_res, dual_res, obj_val)

Residuals of an arbitrary scaled point, reported in problem space, writing `Ax = Ãx`,
`Px = P̃x` and `Aty = Ãᵀy` into the given buffers. This is how [`polish_kernel!`](@ref)
decides whether a polished point is an improvement.
"""
function residuals_at!(
        prob::Problem{T}, x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T},
        Ax::AbstractVector{T}, Px::AbstractVector{T}, Aty::AbstractVector{T}
    ) where {T}
    scaled = prob.scaling > 0
    pr = zero(T)
    if prob.m > 0
        mul_A!(Ax, prob, x)
        subtract!(prob.work_m, Ax, z)
        pr = scaled ? invscaled_norm_inf(prob.E, prob.work_m) : norm_inf(prob.work_m)
    end
    mul_P!(Px, prob, x)
    add!(prob.work_n, prob.q, Px)
    if prob.m > 0
        mul_At!(Aty, prob, y)
        increment!(prob.work_n, Aty)
    end
    dr = scaled ? invscaled_norm_inf(prob.D, prob.work_n) / prob.c : norm_inf(prob.work_n)
    obj = (dot(Px, x) / 2 + dot(prob.q, x)) / prob.c
    return (pr, dr, obj)
end

"""
    polish_kernel!(prob, x, y, z, prim_res, dual_res, Ax, Px, Aty; delta, refine_iter) ->
        (status, xpol, ypol, zpol)

Guess the active set from the iterate `(x, y, z)`, solve the resulting
equality-constrained QP exactly, and accept it only if it improves on `prim_res` and
`dual_res`. `Ax`, `Px` and `Aty` are scratch for [`residuals_at!`](@ref); the caller's `x`,
`y` and `z` are only read.

The returned [`PolishStatus`](@ref) separates the ways this can decline: an empty active
set gives `POLISH_NO_ACTIVE_SET_FOUND`, which means there was nothing to polish rather
than that anything went wrong; a singular reduced KKT gives `POLISH_LINSYS_ERROR`; and a
point that was computed but is no improvement gives `POLISH_FAILED`. Only
`POLISH_SUCCESS` reports a genuinely polished `(xpol, ypol, zpol)`; every other status
returns the caller's own `x`, `y`, `z` unchanged, which the caller must not mistake for a
polished point.

The reduced KKT system is regularized by `delta` and corrected by `refine_iter` steps of
iterative refinement against the unregularized operator.
"""
function polish_kernel!(
        prob::Problem{T}, x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T},
        prim_res::T, dual_res::T, Ax::AbstractVector{T}, Px::AbstractVector{T},
        Aty::AbstractVector{T}; delta::T, refine_iter::Int
    ) where {T}
    require_host(x, "polishing")
    require_entries(prob.P, prob.A, "polishing", "Leave `polishing = false` and take the ADMM iterate.")
    n, m = prob.n, prob.m
    active = Int[]
    lower = Bool[]
    for i in 1:m
        if z[i] - prob.l[i] < -y[i] || prob.l[i] == prob.u[i]
            push!(active, i)
            push!(lower, true)
        elseif prob.u[i] - z[i] < y[i]
            push!(active, i)
            push!(lower, false)
        end
    end
    k = length(active)
    iszero(k) && return (POLISH_NO_ACTIVE_SET_FOUND, x, y, z)
    Ared = Matrix{T}(undef, k, n)
    for j in 1:n
        dj = prob.D[j]
        for (r, i) in enumerate(active)
            Ared[r, j] = prob.E[i] * T(prob.A[i, j]) * dj
        end
    end
    Kp = zeros(T, n + k, n + k)
    for j in 1:n
        dj = prob.D[j]
        for i in 1:n
            Kp[i, j] = prob.c * prob.D[i] * T(prob.P[i, j]) * dj
        end
        Kp[j, j] += delta
        for r in 1:k
            Kp[n + r, j] = Ared[r, j]
            Kp[j, n + r] = Ared[r, j]
        end
    end
    for r in 1:k
        Kp[n + r, n + r] = -delta
    end
    F = bunchkaufman!(Symmetric(copy(Kp), :L); check = false)
    issuccess(F) || return (POLISH_LINSYS_ERROR, x, y, z)
    rhs = Vector{T}(undef, n + k)
    for j in 1:n
        rhs[j] = -prob.q[j]
    end
    for r in 1:k
        i = active[r]
        rhs[n + r] = lower[r] ? prob.l[i] : prob.u[i]
    end
    sol = F \ rhs
    # Iterative refinement against the unregularized operator [P̃ Aredᵀ; Ared 0].
    res = Vector{T}(undef, n + k)
    xv = view(sol, 1:n)
    yv = view(sol, (n + 1):(n + k))
    for _ in 1:refine_iter
        copyto!(res, rhs)
        mul_P!(prob.work_n, prob, xv)
        for j in 1:n
            res[j] -= prob.work_n[j]
        end
        mul!(view(res, 1:n), Ared', yv, -one(T), one(T))
        mul!(view(res, (n + 1):(n + k)), Ared, xv, -one(T), one(T))
        ldiv!(F, res)
        sol .+= res
    end
    xpol = collect(xv)
    ypol = zeros(T, m)
    for r in 1:k
        ypol[active[r]] = sol[n + r]
    end
    zpol = zeros(T, m)
    m > 0 && mul_A!(zpol, prob, xpol)
    # Put z in [l,u] and y in the normal cone at z.
    ypol .+= zpol
    zpol .= clamp.(ypol, prob.l, prob.u)
    ypol .-= zpol
    pr, dr, obj = residuals_at!(prob, xpol, ypol, zpol, Ax, Px, Aty)
    tiny = T(1.0e-10)
    ok = (pr < prim_res && dr < dual_res) ||
        (pr < prim_res && dual_res < tiny) ||
        (dr < dual_res && prim_res < tiny)
    ok || return (POLISH_FAILED, x, y, z)
    return (POLISH_SUCCESS, xpol, ypol, zpol)
end

"""
    polish!(ws) -> PolishStatus

Guess the active set from the ADMM iterates, solve the resulting equality-constrained QP
exactly, and adopt the result only if both residuals improve. Delegates to
[`polish_kernel!`](@ref); on `POLISH_SUCCESS` it copies the polished point into `ws.x`,
`ws.y` and `ws.z` and recomputes the residuals.
"""
function polish!(ws::Workspace{T}) where {T}
    prob = ws.prob
    status, xpol, ypol, zpol = polish_kernel!(
        prob, ws.x, ws.y, ws.z, ws.prim_res, ws.dual_res, ws.Ax, ws.Px, ws.Aty;
        delta = ws.settings.delta, refine_iter = ws.settings.polish_refine_iter
    )
    status === POLISH_SUCCESS || return status
    copyto!(ws.x, xpol)
    copyto!(ws.y, ypol)
    copyto!(ws.z, zpol)
    # Recompute rather than copy `pr`/`dr`/`obj` across: the residuals, the objectives and
    # the duality gap must all describe the polished point, and this is the one place they
    # are derived together.
    update_residuals!(ws)
    return POLISH_SUCCESS
end
