"""
    residuals_at(ws, x, y, z) -> (prim_res, dual_res, obj_val)

Residuals of an arbitrary scaled point, reported in problem space. Used to decide whether
a polished point is an improvement.
"""
function residuals_at(ws::Workspace{T}, x::Vector{T}, y::Vector{T}, z::Vector{T}) where {T}
    scaled = ws.settings.scaling > 0
    pr = zero(T)
    if ws.m > 0
        mul_A!(ws.work_m, ws, x)
        ws.work_m .-= z
        pr = scaled ? invscaled_norm_inf(ws.E, ws.work_m) : norm_inf(ws.work_m)
    end
    mul_P!(ws.Px, ws, x)
    ws.work_n .= ws.q .+ ws.Px
    if ws.m > 0
        mul_At!(ws.Aty, ws, y)
        ws.work_n .+= ws.Aty
    end
    dr = scaled ? invscaled_norm_inf(ws.D, ws.work_n) / ws.c : norm_inf(ws.work_n)
    obj = (dot(ws.Px, x) / 2 + dot(ws.q, x)) / ws.c
    return (pr, dr, obj)
end

"""
    polish!(ws) -> Bool

Guess the active set from the ADMM iterates, solve the resulting equality-constrained QP
exactly, and adopt the result only if both residuals improve. Returns whether the polished
point was accepted.

The reduced KKT system is regularized by `δ` and corrected by `polish_refine_iter` steps
of iterative refinement against the unregularized operator.
"""
function polish!(ws::Workspace{T}) where {T}
    n, m = ws.n, ws.m
    δ = ws.settings.delta
    active = Int[]
    lower = Bool[]
    for i in 1:m
        if ws.z[i] - ws.l[i] < -ws.y[i] || ws.l[i] == ws.u[i]
            push!(active, i)
            push!(lower, true)
        elseif ws.u[i] - ws.z[i] < ws.y[i]
            push!(active, i)
            push!(lower, false)
        end
    end
    k = length(active)
    iszero(k) && return false
    Ared = Matrix{T}(undef, k, n)
    for j in 1:n
        dj = ws.D[j]
        for (r, i) in enumerate(active)
            Ared[r, j] = ws.E[i] * T(ws.A[i, j]) * dj
        end
    end
    Kp = zeros(T, n + k, n + k)
    for j in 1:n
        dj = ws.D[j]
        for i in 1:n
            Kp[i, j] = ws.c * ws.D[i] * T(ws.P[i, j]) * dj
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
    issuccess(F) || return false
    rhs = Vector{T}(undef, n + k)
    for j in 1:n
        rhs[j] = -ws.q[j]
    end
    for r in 1:k
        i = active[r]
        rhs[n + r] = lower[r] ? ws.l[i] : ws.u[i]
    end
    sol = F \ rhs
    # Iterative refinement against the unregularized operator [P̃ Aredᵀ; Ared 0].
    res = Vector{T}(undef, n + k)
    xv = view(sol, 1:n)
    yv = view(sol, (n + 1):(n + k))
    for _ in 1:ws.settings.polish_refine_iter
        copyto!(res, rhs)
        mul_P!(ws.work_n, ws, collect(xv))
        for j in 1:n
            res[j] -= ws.work_n[j]
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
    m > 0 && mul_A!(zpol, ws, xpol)
    # Put z in [l,u] and y in the normal cone at z.
    ypol .+= zpol
    zpol .= clamp.(ypol, ws.l, ws.u)
    ypol .-= zpol
    pr, dr, obj = residuals_at(ws, xpol, ypol, zpol)
    tiny = T(1.0e-10)
    ok = (pr < ws.prim_res && dr < ws.dual_res) ||
        (pr < ws.prim_res && ws.dual_res < tiny) ||
        (dr < ws.dual_res && ws.prim_res < tiny)
    ok || return false
    copyto!(ws.x, xpol)
    copyto!(ws.y, ypol)
    copyto!(ws.z, zpol)
    ws.prim_res = pr
    ws.dual_res = dr
    ws.obj_val = obj
    return true
end
