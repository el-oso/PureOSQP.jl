"""
    admm_step!(ws)

One ADMM iteration:

    (x̃, z̃) ← solve of the subproblem for  (σx − q,  z − ρ⁻¹ ⊙ y)
    x ← α x̃ + (1−α) x
    z ← Π_[l,u](α z̃ + (1−α) z + ρ⁻¹ ⊙ y)
    y ← y + ρ ⊙ (α z̃ + (1−α) z_prev − z)

`x_prev` and `z_prev` are swapped rather than copied.
"""
function admm_step!(ws::Workspace{T}) where {T}
    a = ws.settings.alpha
    ws.x, ws.x_prev = ws.x_prev, ws.x
    ws.z, ws.z_prev = ws.z_prev, ws.z
    ws.rhs_x .= ws.settings.sigma .* ws.x_prev .- ws.q
    if ws.m > 0
        ws.rhs_z .= ws.z_prev .- ws.rho_inv_vec .* ws.y
    end
    solve_kkt!(ws, ws.rhs_x, ws.rhs_z)
    ws.x .= a .* ws.xtilde .+ (one(T) - a) .* ws.x_prev
    ws.delta_x .= ws.x .- ws.x_prev
    if ws.m > 0
        ws.z .= a .* ws.ztilde .+ (one(T) - a) .* ws.z_prev .+ ws.rho_inv_vec .* ws.y
        ws.z .= clamp.(ws.z, ws.l, ws.u)
        ws.delta_y .= ws.rho_vec .* (a .* ws.ztilde .+ (one(T) - a) .* ws.z_prev .- ws.z)
        ws.y .+= ws.delta_y
    end
    return ws
end

"""
    solve!(ws) -> Solution

Run the ADMM loop on an existing workspace. Safe to call repeatedly; with
`warm_starting = true` the previous iterates are the starting point.

If the iteration limit is reached, the termination tests are retried once at ten times the
requested tolerances before the run is declared unconverged, which is where the
`*_INACCURATE` statuses come from.
"""
function solve!(ws::Workspace{T}) where {T}
    s = ws.settings
    s.warm_starting || cold_start!(ws)
    ws.status = UNSOLVED
    ws.polished = false
    ws.iter = 0
    for iter in 1:s.max_iter
        ws.iter = iter
        admm_step!(ws)
        adapting = s.adaptive_rho && s.adaptive_rho_interval > 0 &&
            iter % s.adaptive_rho_interval == 0
        checking = s.check_termination > 0 && iter % s.check_termination == 0
        (adapting || checking || iter == 1) || continue
        update_residuals!(ws)
        if checking
            st = check_termination(ws, false)
            if st != UNSOLVED
                ws.status = st
                break
            end
        end
        adapting && adapt_rho!(ws)
    end
    if ws.status == UNSOLVED
        update_residuals!(ws)
        st = check_termination(ws, false)
        if st == UNSOLVED
            st = check_termination(ws, true)
        end
        ws.status = st == UNSOLVED ? MAX_ITER_REACHED : st
    end
    if (ws.status == SOLVED || ws.status == SOLVED_INACCURATE) && s.polish
        ws.polished = polish!(ws)
    end
    sol = build_solution(ws)
    # An infeasible run leaves the iterates on a diverging ray; a later solve on this
    # workspace must not resume from there.
    has_solution(ws.status) || cold_start!(ws)
    return sol
end

function build_solution(ws::Workspace{T}) where {T}
    n, m = ws.n, ws.m
    if ws.status == PRIMAL_INFEASIBLE || ws.status == PRIMAL_INFEASIBLE_INACCURATE
        cert = ws.settings.scaling > 0 ? ws.E .* ws.delta_y : copy(ws.delta_y)
        nc = norm_inf(cert)
        nc > zero(T) && (cert ./= nc)
        return Solution{T}(
            fill(T(NaN), n), fill(T(NaN), m), ws.status, T(Inf),
            ws.prim_res, ws.dual_res, ws.iter, false, cert, T[]
        )
    elseif ws.status == DUAL_INFEASIBLE || ws.status == DUAL_INFEASIBLE_INACCURATE
        cert = ws.settings.scaling > 0 ? ws.D .* ws.delta_x : copy(ws.delta_x)
        nc = norm_inf(cert)
        nc > zero(T) && (cert ./= nc)
        return Solution{T}(
            fill(T(NaN), n), fill(T(NaN), m), ws.status, T(-Inf),
            ws.prim_res, ws.dual_res, ws.iter, false, T[], cert
        )
    elseif !has_solution(ws.status)
        # NON_CONVEX and anything else without a meaningful point: no number here would
        # mean anything, so do not hand back one that looks like a solution.
        return Solution{T}(
            fill(T(NaN), n), fill(T(NaN), m), ws.status, T(NaN),
            ws.prim_res, ws.dual_res, ws.iter, false, T[], T[]
        )
    end
    x = ws.D .* ws.x
    y = (ws.E .* ws.y) ./ ws.c
    return Solution{T}(
        x, y, ws.status, ws.obj_val, ws.prim_res, ws.dual_res,
        ws.iter, ws.polished, T[], T[]
    )
end

"""
    solve(P, q, A, l, u; x0 = nothing, y0 = nothing, kwargs...) -> Solution

Solve `min ½xᵀPx + qᵀx  s.t.  l ≤ Ax ≤ u` in one call.
"""
function solve(
        P::AbstractMatrix, q::AbstractVector, A::AbstractMatrix,
        l::AbstractVector, u::AbstractVector;
        x0 = nothing, y0 = nothing, kwargs...
    )
    ws = setup(P, q, A, l, u; kwargs...)
    (isnothing(x0) && isnothing(y0)) || warm_start!(ws; x = x0, y = y0)
    return solve!(ws)
end
