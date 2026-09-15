"""
    residuals_at(ws, x, y, z) -> (prim_res, dual_res, obj_val)

Residuals of an arbitrary scaled point, reported in problem space. Used to decide whether
a polished point is an improvement.
"""
function residuals_at(
        ws::Workspace{T}, x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T}
    ) where {T}
    prob = ws.prob
    scaled = prob.scaling > 0
    pr = zero(T)
    if prob.m > 0
        mul_A!(prob.work_m, prob, x)
        prob.work_m .-= z
        pr = scaled ? invscaled_norm_inf(prob.E, prob.work_m) : norm_inf(prob.work_m)
    end
    mul_P!(ws.Px, prob, x)
    prob.work_n .= prob.q .+ ws.Px
    if prob.m > 0
        mul_At!(ws.Aty, prob, y)
        prob.work_n .+= ws.Aty
    end
    dr = scaled ? invscaled_norm_inf(prob.D, prob.work_n) / prob.c : norm_inf(prob.work_n)
    obj = (dot(ws.Px, x) / 2 + dot(prob.q, x)) / prob.c
    return (pr, dr, obj)
end

"""
    polish!(ws) -> PolishStatus

Guess the active set from the ADMM iterates, solve the resulting equality-constrained QP
exactly, and adopt the result only if both residuals improve.

The returned [`PolishStatus`](@ref) separates the ways this can decline. An empty active
set gives `POLISH_NO_ACTIVE_SET_FOUND`, which means there was nothing to polish rather
than that anything went wrong; a singular reduced KKT gives `POLISH_LINSYS_ERROR`; and a
point that was computed but is no improvement gives `POLISH_FAILED`. Only
`POLISH_SUCCESS` replaces the iterates.

The reduced KKT system is regularized by `δ` and corrected by `polish_refine_iter` steps
of iterative refinement against the unregularized operator.
"""
function polish!(ws::Workspace{T}) where {T}
    prob = ws.prob
    require_host(ws.x, "polishing")
    require_entries(prob.P, prob.A, "polishing", "Leave `polishing = false` and take the ADMM iterate.")
    n, m = prob.n, prob.m
    δ = ws.settings.delta
    active = Int[]
    lower = Bool[]
    for i in 1:m
        if ws.z[i] - prob.l[i] < -ws.y[i] || prob.l[i] == prob.u[i]
            push!(active, i)
            push!(lower, true)
        elseif prob.u[i] - ws.z[i] < ws.y[i]
            push!(active, i)
            push!(lower, false)
        end
    end
    k = length(active)
    iszero(k) && return POLISH_NO_ACTIVE_SET_FOUND
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
        Kp[j, j] += δ
        for r in 1:k
            Kp[n + r, j] = Ared[r, j]
            Kp[j, n + r] = Ared[r, j]
        end
    end
    for r in 1:k
        Kp[n + r, n + r] = -δ
    end
    F = bunchkaufman!(Symmetric(copy(Kp), :L); check = false)
    issuccess(F) || return POLISH_LINSYS_ERROR
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
    for _ in 1:ws.settings.polish_refine_iter
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
    pr, dr, obj = residuals_at(ws, xpol, ypol, zpol)
    tiny = T(1.0e-10)
    ok = (pr < ws.prim_res && dr < ws.dual_res) ||
        (pr < ws.prim_res && ws.dual_res < tiny) ||
        (dr < ws.dual_res && ws.prim_res < tiny)
    ok || return POLISH_FAILED
    copyto!(ws.x, xpol)
    copyto!(ws.y, ypol)
    copyto!(ws.z, zpol)
    # Recompute rather than copy `pr`/`dr`/`obj` across: the residuals, the objectives and
    # the duality gap must all describe the polished point, and this is the one place they
    # are derived together.
    update_residuals!(ws)
    return POLISH_SUCCESS
end
