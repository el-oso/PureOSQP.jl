"""
    build_solution(ws) -> Solution

Package the workspace's point as a [`Solution`](@ref), with the residuals recomputed from the
caller's own data.

Several fields are structurally zero here and stay that way: there is no `ρ` to report, no
accelerator, no conjugate-gradient count and no primal-dual integral, because none of those
exist in an active-set method.
"""
function build_solution(ws::ActiveSetWorkspace{T}) where {T}
    prob = ws.prob
    run_time = ws.setup_time + ws.update_time + ws.solve_time
    if has_solution(ws.status)
        obj = obj_value(ws)
        prim, dual = residuals(ws)
        gap = duality_gap(ws)
    else
        obj = ws.status == PRIMAL_INFEASIBLE ? T(Inf) : T(-Inf)
        prim = dual = gap = T(NaN)
    end
    return Solution{T}(
        copy(ws.x), copy(ws.y), ws.status, obj, obj - gap, gap,
        prim, dual, max(prim, dual, abs(gap)), ws.iter,
        0.0, 0.0, zero(T), 0, 0, 0,
        ws.polished, ws.status_polish,
        ws.setup_time, ws.update_time, ws.solve_time, 0.0, run_time,
        T[], T[],
    )
end

"½ xᵀPx + qᵀx at the current point, from the caller's data."
function obj_value(ws::ActiveSetWorkspace{T}) where {T}
    prob = ws.prob
    Px = prob.P * ws.x
    return T(0.5) * dot(ws.x, Px) + dot(prob.q0, ws.x)
end

"""
    residuals(ws) -> (primal, dual)

`‖max(Ax−u,0) + max(l−Ax,0)‖∞` and `‖Px + q + Aᵀy‖∞`, both in the caller's own scaling.
"""
function residuals(ws::ActiveSetWorkspace{T}) where {T}
    prob = ws.prob
    Ax = prob.A * ws.x
    prim = zero(T)
    for i in eachindex(Ax)
        prim = max(prim, max(Ax[i] - prob.u0[i], zero(T)), max(prob.l0[i] - Ax[i], zero(T)))
    end
    r = prob.P * ws.x + prob.q0 + prob.A' * ws.y
    return prim, maximum(abs, r; init = zero(T))
end

"""
    duality_gap(ws) -> gap

`xᵀPx + qᵀx + uᵀmax(y,0) + lᵀmin(y,0)`, the gap between the primal objective and the dual
one. An exact active-set answer drives this to rounding.
"""
function duality_gap(ws::ActiveSetWorkspace{T}) where {T}
    prob = ws.prob
    support = zero(T)
    for i in eachindex(ws.y)
        yi = ws.y[i]
        if yi > 0
            isfinite(prob.u0[i]) && (support += prob.u0[i] * yi)
        elseif yi < 0
            isfinite(prob.l0[i]) && (support += prob.l0[i] * yi)
        end
    end
    Px = prob.P * ws.x
    return dot(ws.x, Px) + dot(prob.q0, ws.x) + support
end
