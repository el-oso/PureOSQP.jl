@inline norm_inf(v::AbstractVector{T}) where {T} = maximum(abs, v; init = zero(T))

@inline function scaled_norm_inf(s::AbstractVector{T}, v::AbstractVector{T}) where {T}
    r = zero(T)
    for i in eachindex(s, v)
        r = max(r, abs(s[i] * v[i]))
    end
    return r
end

@inline function invscaled_norm_inf(s::AbstractVector{T}, v::AbstractVector{T}) where {T}
    r = zero(T)
    for i in eachindex(s, v)
        r = max(r, abs(v[i] / s[i]))
    end
    return r
end

"""
    update_residuals!(ws)

Recompute `‖Ãx − z‖∞` and `‖P̃x + q̃ + Ãᵀy‖∞`, both in scaled space (used by the ρ
estimate) and unscaled (used for termination and reporting), plus the objective value.
"""
function update_residuals!(ws::Workspace{T}) where {T}
    n, m = ws.n, ws.m
    scaled = ws.settings.scaling > 0
    if m > 0
        mul_A!(ws.Ax, ws, ws.x)
        ws.work_m .= ws.Ax .- ws.z
        ws.scaled_prim_res = norm_inf(ws.work_m)
        ws.prim_res = if scaled
            r = zero(T)
            for i in 1:m
                r = max(r, abs(ws.work_m[i] / ws.E[i]))
            end
            r
        else
            ws.scaled_prim_res
        end
    else
        ws.prim_res = zero(T)
        ws.scaled_prim_res = zero(T)
    end
    mul_P!(ws.Px, ws, ws.x)
    ws.work_n .= ws.q .+ ws.Px
    if m > 0
        mul_At!(ws.Aty, ws, ws.y)
        ws.work_n .+= ws.Aty
    else
        fill!(ws.Aty, zero(T))
    end
    ws.scaled_dual_res = norm_inf(ws.work_n)
    ws.dual_res = if scaled
        r = zero(T)
        for i in 1:n
            r = max(r, abs(ws.work_n[i] / ws.D[i]))
        end
        r / ws.c
    else
        ws.scaled_dual_res
    end
    ws.obj_val = (dot(ws.Px, ws.x) / 2 + dot(ws.q, ws.x)) / ws.c
    return ws
end

function eps_prim(ws::Workspace{T}) where {T}
    s = ws.settings
    mx = if s.scaling > 0
        max(invscaled_norm_inf(ws.E, ws.z), invscaled_norm_inf(ws.E, ws.Ax))
    else
        max(norm_inf(ws.z), norm_inf(ws.Ax))
    end
    return s.eps_abs + s.eps_rel * mx
end

function eps_dual(ws::Workspace{T}) where {T}
    s = ws.settings
    mx = if s.scaling > 0
        max(invscaled_norm_inf(ws.D, ws.q), invscaled_norm_inf(ws.D, ws.Aty), invscaled_norm_inf(ws.D, ws.Px)) / ws.c
    else
        max(norm_inf(ws.q), norm_inf(ws.Aty), norm_inf(ws.Px))
    end
    return s.eps_abs + s.eps_rel * mx
end

"""
    project_polar_reccone!(v, l, u)

Project `v` onto the polar of the recession cone of `[l, u]`, in place.
"""
function project_polar_reccone!(v::AbstractVector{T}, l::AbstractVector{T}, u::AbstractVector{T}) where {T}
    loose = INFTY(T) * MIN_SCALING(T)
    for i in eachindex(v, l, u)
        if u[i] > loose
            v[i] = l[i] < -loose ? zero(T) : min(v[i], zero(T))
        elseif l[i] < -loose
            v[i] = max(v[i], zero(T))
        end
    end
    return v
end

"""
    is_primal_infeasible(ws, eps) -> Bool

Certificate test on `δy`, from libosqp 0.6.2: after projecting `δy` onto the polar of the
recession cone of `[l, u]`, the problem is primal infeasible when
`uᵀ max(δy,0) + lᵀ min(δy,0) < ε‖δy‖` and `‖Aᵀδy‖ < ε‖δy‖`. Overwrites `ws.delta_y` with
the projected vector, which then becomes the certificate.
"""
function is_primal_infeasible(ws::Workspace{T}, eps::T) where {T}
    ws.m == 0 && return false
    project_polar_reccone!(ws.delta_y, ws.l, ws.u)
    ndy = ws.settings.scaling > 0 ? scaled_norm_inf(ws.E, ws.delta_y) : norm_inf(ws.delta_y)
    ndy > DIVISION_TOL(T) || return false
    lhs = zero(T)
    for i in 1:ws.m
        dy = ws.delta_y[i]
        lhs += ws.u[i] * max(dy, zero(T)) + ws.l[i] * min(dy, zero(T))
    end
    lhs < eps * ndy || return false
    mul_At!(ws.work_n, ws, ws.delta_y)
    if ws.settings.scaling > 0
        # mul_At! applies D; the unscaled test is Aᵀ(E ⊙ δy), so divide it back out once.
        for i in 1:ws.n
            ws.work_n[i] /= ws.D[i]
        end
    end
    return norm_inf(ws.work_n) < eps * ndy
end

"""
    is_dual_infeasible(ws, eps) -> Bool

Certificate test on `δx`: `qᵀδx < 0`, `‖Pδx‖ < ε‖δx‖`, and `Aδx` in the recession cone
of `[l, u]` to within `ε‖δx‖`.
"""
function is_dual_infeasible(ws::Workspace{T}, eps::T) where {T}
    scaled = ws.settings.scaling > 0
    ndx = scaled ? scaled_norm_inf(ws.D, ws.delta_x) : norm_inf(ws.delta_x)
    cost = scaled ? ws.c : one(T)
    ndx > DIVISION_TOL(T) || return false
    # v0.6.2 uses a tolerance here, not a strict sign test; master tightened it to < 0.
    dot(ws.q, ws.delta_x) < cost * eps * ndx || return false
    mul_P!(ws.work_n, ws, ws.delta_x)
    if scaled
        for i in 1:ws.n
            ws.work_n[i] /= ws.D[i]
        end
    end
    norm_inf(ws.work_n) < cost * eps * ndx || return false
    ws.m == 0 && return true
    mul_A!(ws.work_m, ws, ws.delta_x)
    if scaled
        for i in 1:ws.m
            ws.work_m[i] /= ws.E[i]
        end
    end
    loose = INFTY(T) * MIN_SCALING(T)
    tol = eps * ndx
    for i in 1:ws.m
        v = ws.work_m[i]
        if ws.u[i] < loose && v > tol
            return false
        end
        if ws.l[i] > -loose && v < -tol
            return false
        end
    end
    return true
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
    (ws.prim_res > inf || ws.dual_res > inf) && return NON_CONVEX
    f = approximate ? T(10) : one(T)
    prim_ok = ws.m == 0 || ws.prim_res < f * eps_prim(ws)
    if !prim_ok && is_primal_infeasible(ws, f * s.eps_prim_inf)
        return approximate ? PRIMAL_INFEASIBLE_INACCURATE : PRIMAL_INFEASIBLE
    end
    dual_ok = ws.dual_res < f * eps_dual(ws)
    if !dual_ok && is_dual_infeasible(ws, f * s.eps_dual_inf)
        return approximate ? DUAL_INFEASIBLE_INACCURATE : DUAL_INFEASIBLE
    end
    (prim_ok && dual_ok) || return UNSOLVED
    return approximate ? SOLVED_INACCURATE : SOLVED
end
