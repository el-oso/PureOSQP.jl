"""
    update_residuals!(ws)

Recompute `‖Ãx − z‖∞` and `‖P̃x + q̃ + Ãᵀy‖∞`, both in scaled space (used by the ρ
estimate) and unscaled (used for termination and reporting), plus the objective value.
"""
function update_residuals!(ws::OperatorSplittingWorkspace{T}) where {T}
    prob = ws.prob
    m = prob.m
    scaled = prob.scaling > 0
    if m > 0
        mul_A!(ws.Ax, prob, ws.x)
        subtract!(prob.work_m, ws.Ax, ws.z)
        ws.scaled_prim_res = norm_inf(prob.work_m)
        ws.prim_res = scaled ? invscaled_norm_inf(prob.E, prob.work_m) : ws.scaled_prim_res
    else
        ws.prim_res = zero(T)
        ws.scaled_prim_res = zero(T)
    end
    mul_P!(ws.Px, prob, ws.x)
    add!(prob.work_n, prob.q, ws.Px)
    if m > 0
        mul_At!(ws.Aty, prob, ws.y)
        increment!(prob.work_n, ws.Aty)
    else
        fill!(ws.Aty, zero(T))
    end
    ws.scaled_dual_res = norm_inf(prob.work_n)
    ws.dual_res = if scaled
        invscaled_norm_inf(prob.D, prob.work_n) / prob.c
    else
        ws.scaled_dual_res
    end
    quad, lin, sup = gap_terms(prob, ws.y, ws.Px, ws.x)
    ws.xtPx = quad
    ws.qtx = lin
    ws.SCy = sup
    ws.scaled_duality_gap = quad + lin + sup
    cinv = inv(prob.c)
    ws.obj_val = (quad / 2 + lin) * cinv
    ws.dual_obj_val = (-quad / 2 - sup) * cinv
    ws.duality_gap = ws.scaled_duality_gap * cinv
    ws.rel_kkt_error = max(ws.prim_res, ws.dual_res, abs(ws.duality_gap))
    return ws
end

eps_prim(ws::OperatorSplittingWorkspace) = eps_prim(ws.prob, ws.options, ws.z, ws.Ax)

eps_dual(ws::OperatorSplittingWorkspace) = eps_dual(ws.prob, ws.options, ws.Aty, ws.Px)

eps_duality_gap(ws::OperatorSplittingWorkspace) = eps_duality_gap(ws.prob, ws.options, ws.xtPx, ws.qtx, ws.SCy)

function is_primal_infeasible(ws::OperatorSplittingWorkspace{T}, eps::T) where {T}
    return is_primal_infeasible(ws.prob, ws.delta_y, eps)
end

function is_dual_infeasible(ws::OperatorSplittingWorkspace{T}, eps::T) where {T}
    return is_dual_infeasible(ws.prob, ws.delta_x, eps)
end

"""
    check_termination(ws, approximate = false) -> Status

`UNSOLVED` means keep iterating. With `approximate = true` every tolerance is relaxed by a
factor of ten and the returned statuses are the `*_INACCURATE` variants; that is the retry
the reference implementation makes once the iteration limit is hit, before declaring the
run unconverged.
"""
function check_termination(ws::OperatorSplittingWorkspace{T}, approximate::Bool = false) where {T}
    s = ws.options
    inf = INFTY(T)
    # `NaN` compares false against everything, so it passes a `> inf` test and would reach
    # the iteration limit as a point with `has_solution` true. It is caught by name, and
    # reported as the divergence it is.
    (isnan(ws.prim_res) || isnan(ws.dual_res)) && return NON_CONVEX
    (ws.prim_res > inf || ws.dual_res > inf) && return NON_CONVEX
    f = approximate ? T(10) : one(T)
    scaled_term = s.scaled_termination && ws.prob.scaling > 0
    pres = scaled_term ? ws.scaled_prim_res : ws.prim_res
    dres = scaled_term ? ws.scaled_dual_res : ws.dual_res
    prim_ok = iszero(ws.prob.m) || pres < f * eps_prim(ws)
    if !prim_ok && is_primal_infeasible(ws, f * s.eps_prim_inf)
        return approximate ? PRIMAL_INFEASIBLE_INACCURATE : PRIMAL_INFEASIBLE
    end
    dual_ok = dres < f * eps_dual(ws)
    if !dual_ok && is_dual_infeasible(ws, f * s.eps_dual_inf)
        return approximate ? DUAL_INFEASIBLE_INACCURATE : DUAL_INFEASIBLE
    end
    (prim_ok && dual_ok) || return UNSOLVED
    # The gap is checked only once the residuals pass, so it can delay convergence but
    # never declare it: a point with a small gap and a large residual is not a solution.
    if s.check_dualgap
        gap = scaled_term ? ws.scaled_duality_gap : ws.duality_gap
        abs(gap) < f * eps_duality_gap(ws) || return UNSOLVED
    end
    return approximate ? SOLVED_INACCURATE : SOLVED
end

"""
    accumulate_primdual!(ws) -> Nothing

Add this iteration's slice to the primal-dual integral, `∫|gap| dt` over the solve.

Two accumulators run over the same samples, because the rule matters and libosqp's is not
readable from its published headers. `primdual_int` joins consecutive samples with a straight
line, the trapezoid. `primdual_int_log` joins them with an exponential, which is what a
geometrically decaying gap does between samples, and integrates that exactly: the slice is
the interval times the *logarithmic* mean of the endpoints rather than their arithmetic mean.
The logarithmic mean is the smaller of the two whenever the endpoints differ, so trapezoid
overstates a convex decaying gap and the pair brackets the truth.

The exponential model needs both endpoints strictly positive, and the gap is not sign-definite
before convergence, so the log rule falls back to the trapezoid on any slice where it is
undefined. Both are wall-clock quantities and neither is reproducible across machines.

Called from [`solve!`](@ref) after [`update_residuals!`](@ref) rather than from inside it.
`time_ns` compiles to a runtime call that AllocCheck counts as an allocation, and
`update_residuals!` carries the allocation-free guarantee where `solve!` does not — the same
reason the `time_limit` clock lives in the loop.
"""
function accumulate_primdual!(ws::OperatorSplittingWorkspace{T})::Nothing where {T}
    now = (time_ns() - ws.loop_start) / 1.0e9
    gap = abs(ws.duality_gap)
    # The first sample opens the interval and closes nothing.
    if ws.iter > 0
        dt = now - ws.last_gap_time
        if dt > 0.0
            prev = ws.last_gap
            trap = 0.5 * (Float64(prev) + Float64(gap)) * dt
            ws.primdual_int += trap
            ws.primdual_int_log += logmean_slice(Float64(prev), Float64(gap), dt, trap)
        end
    end
    ws.last_gap_time = now
    ws.last_gap = gap
    return nothing
end

"""
    logmean_slice(a, b, dt, trap) -> Float64

`dt` times the logarithmic mean of `a` and `b`, which is `∫` of the exponential through them.

Returns `trap` where that model does not apply: a non-positive endpoint leaves the exponential
undefined, and equal endpoints make the difference quotient `0/0` while the two means coincide.
Near-equal endpoints are handled by the same branch, since the quotient loses its significant
digits before the means differ enough to matter.
"""
function logmean_slice(a::Float64, b::Float64, dt::Float64, trap::Float64)
    (a > 0.0 && b > 0.0) || return trap
    r = a / b
    (isfinite(r) && abs(r - 1.0) > 1.0e-8) || return trap
    return dt * (a - b) / log(r)
end
