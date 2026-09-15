@inline norm_inf(v::AbstractVector{T}) where {T} = maximum(abs, v; init = zero(T))

"`max|s[i] v[i]|`. See `src/elementwise.jl` on why there are two schedules."
@inline function scaled_norm_inf(s::Array{T}, v::Array{T}) where {T}
    r = zero(T)
    for i in eachindex(s, v)
        r = max(r, abs(s[i] * v[i]))
    end
    return r
end

@inline function scaled_norm_inf(s::AbstractVector{T}, v::AbstractVector{T}) where {T}
    return mapreduce((a, b) -> abs(a * b), max, s, v; init = zero(T))
end

"`max|v[i] / s[i]|`."
@inline function invscaled_norm_inf(s::Array{T}, v::Array{T}) where {T}
    r = zero(T)
    for i in eachindex(s, v)
        r = max(r, abs(v[i] / s[i]))
    end
    return r
end

@inline function invscaled_norm_inf(s::AbstractVector{T}, v::AbstractVector{T}) where {T}
    return mapreduce((a, b) -> abs(b / a), max, s, v; init = zero(T))
end
"""
    update_residuals!(ws)

Recompute `‖Ãx − z‖∞` and `‖P̃x + q̃ + Ãᵀy‖∞`, both in scaled space (used by the ρ
estimate) and unscaled (used for termination and reporting), plus the objective value.
"""
function update_residuals!(ws::Workspace{T}) where {T}
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
    # Objectives and the duality gap. `SC(y) = uᵀmax(y,0) + lᵀmin(y,0)`, taken after
    # projecting `y` onto the polar of the recession cone of `[l, u]`: that projection is
    # what stops a row with an infinite bound from contributing `Inf * 0`. Multipliers
    # under the deadzone are dropped first, since a `1e-20` `y` against a large bound is
    # noise that would otherwise dominate the sum.
    #
    # `tmp_m` is free here: it is `mul_At!`'s scratch, and the last call to it is above.
    quad = dot(ws.Px, ws.x)
    lin = dot(prob.q, ws.x)
    sup = zero(T)
    if m > 0
        copyto!(prob.tmp_m, ws.y)
        project_polar_reccone!(prob.tmp_m, prob.l, prob.u)
        sup = support_sum(prob.tmp_m, prob.l, prob.u)
    end
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

"""
    support_sum(y, l, u) -> T

`uᵀ max(y, 0) + lᵀ min(y, 0)`, the support function of `[l, u]` at `y`.

Multipliers below the deadzone contribute nothing: a `1e-20` multiplier against a bound of
`1e20` is noise that would otherwise dominate the sum. `y` must already be projected onto
the polar recession cone, which is what keeps an infinite bound from meeting a nonzero
multiplier here.
"""
@inline function support_sum(y::Array{T}, l::Array{T}, u::Array{T}) where {T}
    dead = ZERO_DEADZONE(T)
    s = zero(T)
    for i in eachindex(y)
        v = y[i]
        abs(v) < dead && continue
        s += (v > zero(T) ? u[i] : l[i]) * v
    end
    return s
end

@inline function support_sum(y::AbstractVector{T}, l::AbstractVector{T}, u::AbstractVector{T}) where {T}
    dead = ZERO_DEADZONE(T)
    return mapreduce(
        (v, lo, hi) -> abs(v) < dead ? zero(T) : (v > zero(T) ? hi : lo) * v,
        +, y, l, u; init = zero(T)
    )
end

function eps_prim(ws::Workspace{T}) where {T}
    s = ws.settings
    mx = if ws.prob.scaling > 0
        max(invscaled_norm_inf(ws.prob.E, ws.z), invscaled_norm_inf(ws.prob.E, ws.Ax))
    else
        max(norm_inf(ws.z), norm_inf(ws.Ax))
    end
    return s.eps_abs + s.eps_rel * mx
end

function eps_dual(ws::Workspace{T}) where {T}
    s, prob = ws.settings, ws.prob
    mx = if prob.scaling > 0
        max(invscaled_norm_inf(prob.D, prob.q), invscaled_norm_inf(prob.D, ws.Aty), invscaled_norm_inf(prob.D, ws.Px)) / prob.c
    else
        max(norm_inf(prob.q), norm_inf(ws.Aty), norm_inf(ws.Px))
    end
    return s.eps_abs + s.eps_rel * mx
end

"""
    eps_duality_gap(ws)

Tolerance for the duality-gap test, relative to the size of the terms that make up the
gap. Without the relative part a problem whose objective is `1e8` could never pass.
"""
function eps_duality_gap(ws::Workspace{T}) where {T}
    s = ws.settings
    mx = max(abs(ws.xtPx), abs(ws.qtx), abs(ws.SCy))
    # The stored terms are scaled; unscale unless termination is being judged scaled.
    (ws.prob.scaling > 0 && !s.scaled_termination) && (mx /= ws.prob.c)
    return s.eps_abs + s.eps_rel * mx
end

"The polar recession cone projection, elementwise."
@inline function polar_reccone(v::T, lo::T, hi::T, loose::T) where {T}
    if hi > loose
        return lo < -loose ? zero(T) : min(v, zero(T))
    elseif lo < -loose
        return max(v, zero(T))
    end
    return v
end

"""
    project_polar_reccone!(v, l, u)

Project `v` onto the polar of the recession cone of `[l, u]`, in place.
"""
function project_polar_reccone!(v::Array{T}, l::Array{T}, u::Array{T}) where {T}
    loose = INFTY(T) * MIN_SCALING(T)
    for i in eachindex(v)
        v[i] = polar_reccone(v[i], l[i], u[i], loose)
    end
    return v
end

function project_polar_reccone!(v::AbstractVector{T}, l::AbstractVector{T}, u::AbstractVector{T}) where {T}
    loose = INFTY(T) * MIN_SCALING(T)
    v .= polar_reccone.(v, l, u, loose)
    return v
end

"`uᵀ max(v, 0) + lᵀ min(v, 0)`, the support function evaluated without a deadzone."
@inline function support_plain(v::Vector{T}, l, u) where {T}
    s = zero(T)
    for i in eachindex(v)
        dy = v[i]
        s += u[i] * max(dy, zero(T)) + l[i] * min(dy, zero(T))
    end
    return s
end

@inline function support_plain(v::AbstractVector{T}, l, u) where {T}
    return mapreduce(
        (dy, lo, hi) -> hi * max(dy, zero(T)) + lo * min(dy, zero(T)),
        +, v, l, u; init = zero(T)
    )
end

"""
    is_primal_infeasible(ws, eps) -> Bool

Certificate test on `δy`: after projecting `δy` onto the polar of the recession cone of
`[l, u]`, the problem is primal infeasible when `uᵀ max(δy,0) + lᵀ min(δy,0) < 0` and
`‖Aᵀδy‖ < ε‖δy‖`. Overwrites `ws.delta_y` with the projected vector, which then becomes the
certificate.

The support function is tested against zero rather than against `ε‖δy‖`. A tolerance there
admits directions that do not separate, and a certificate is a proof or it is nothing.
"""
function is_primal_infeasible(ws::Workspace{T}, eps::T) where {T}
    prob = ws.prob
    iszero(prob.m) && return false
    project_polar_reccone!(ws.delta_y, prob.l, prob.u)
    ndy = prob.scaling > 0 ? scaled_norm_inf(prob.E, ws.delta_y) : norm_inf(ws.delta_y)
    ndy > DIVISION_TOL(T) || return false
    # Strict, as libosqp 1.0 has it: the support function of the direction must be
    # negative, not merely under a tolerance that scales with the direction's own norm.
    support_plain(ws.delta_y, prob.l, prob.u) < zero(T) || return false
    mul_At!(prob.work_n, prob, ws.delta_y)
    if prob.scaling > 0
        # mul_At! applies D; the unscaled test is Aᵀ(E ⊙ δy), so divide it back out once.
        divide!(prob.work_n, prob.work_n, prob.D)
    end
    return norm_inf(prob.work_n) < eps * ndy
end

"Whether `Aδx` leaves the recession cone of `[l, u]` at any row, to within `tol`."
@inline function leaves_reccone(v::Vector{T}, l, u, loose, tol) where {T}
    for i in eachindex(v)
        vi = v[i]
        u[i] < loose && vi > tol && return true
        l[i] > -loose && vi < -tol && return true
    end
    return false
end

@inline function leaves_reccone(v::AbstractVector{T}, l, u, loose, tol) where {T}
    return mapreduce(
        (vi, lo, hi) -> (hi < loose && vi > tol) || (lo > -loose && vi < -tol),
        |, v, l, u; init = false
    )
end

"""
    is_dual_infeasible(ws, eps) -> Bool

Certificate test on `δx`: `qᵀδx < 0`, `‖Pδx‖ < ε‖δx‖`, and `Aδx` in the recession cone
of `[l, u]` to within `ε‖δx‖`.
"""
function is_dual_infeasible(ws::Workspace{T}, eps::T) where {T}
    prob = ws.prob
    scaled = prob.scaling > 0
    ndx = scaled ? scaled_norm_inf(prob.D, ws.delta_x) : norm_inf(ws.delta_x)
    cost = scaled ? prob.c : one(T)
    ndx > DIVISION_TOL(T) || return false
    # A strict sign test, as libosqp 1.0 has it. Allowing `qᵀδx` up to `+ε‖δx‖` certifies a
    # direction that does not descend, and on an ill-conditioned `A` the near-null directions
    # clear the two remaining tests, so a bounded problem is declared unbounded.
    dot(prob.q, ws.delta_x) < zero(T) || return false
    mul_P!(prob.work_n, prob, ws.delta_x)
    scaled && divide!(prob.work_n, prob.work_n, prob.D)
    norm_inf(prob.work_n) < cost * eps * ndx || return false
    iszero(prob.m) && return true
    mul_A!(prob.work_m, prob, ws.delta_x)
    scaled && divide!(prob.work_m, prob.work_m, prob.E)
    return !leaves_reccone(prob.work_m, prob.l, prob.u, INFTY(T) * MIN_SCALING(T), eps * ndx)
end

"""
    check_termination(ws, approximate = false) -> Status

`UNSOLVED` means keep iterating. With `approximate = true` every tolerance is relaxed by a
factor of ten and the returned statuses are the `*_INACCURATE` variants; that is the retry
the reference implementation makes once the iteration limit is hit, before declaring the
run unconverged.
"""
function check_termination(ws::Workspace{T}, approximate::Bool = false) where {T}
    s = ws.settings
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
function accumulate_primdual!(ws::Workspace{T})::Nothing where {T}
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
