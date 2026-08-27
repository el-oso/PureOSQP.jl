"""
    set_rho_vec!(ws, rho) -> Bool

Classify each constraint and fill `rho_vec`/`rho_inv_vec`. Rows with both bounds
effectively infinite get `RHO_MIN`, rows with `ũ - l̃ < RHO_TOL` are equalities and get
`1e3 ρ`, everything else gets `ρ`. Returns `true` if any classification changed, which is
what forces a refactorization.
"""
function set_rho_vec!(ws::Workspace{T}, rho::T) where {T}
    ws.rho = clamp(rho, RHO_MIN(T), RHO_MAX(T))
    loose = INFTY(T) * MIN_SCALING(T)
    changed = false
    for i in 1:ws.m
        t = if ws.l[i] < -loose && ws.u[i] > loose
            Int8(-1)
        elseif ws.u[i] - ws.l[i] < RHO_TOL(T)
            Int8(1)
        else
            Int8(0)
        end
        changed |= (t != ws.constr_type[i])
        ws.constr_type[i] = t
        ws.rho_vec[i] = t == Int8(-1) ? RHO_MIN(T) :
            t == Int8(1) ? RHO_EQ_OVER_INEQ(T) * ws.rho : ws.rho
        ws.rho_inv_vec[i] = inv(ws.rho_vec[i])
    end
    return changed
end

"""
    adapt_rho!(ws) -> Bool

Rescale `ρ` from the current scaled residuals and refactorize if the estimate moved by
more than `adaptive_rho_tolerance`. Returns `true` when a refactorization happened.
"""
function adapt_rho!(ws::Workspace{T}) where {T}
    tol = DIVISION_TOL(T)
    pnorm = max(norm_inf(ws.z), norm_inf(ws.Ax))
    dnorm = max(norm_inf(ws.q), norm_inf(ws.Aty), norm_inf(ws.Px))
    pr = ws.scaled_prim_res / (pnorm + tol)
    dr = ws.scaled_dual_res / (dnorm + tol)
    rho_new = clamp(ws.rho * sqrt(pr / dr), RHO_MIN(T), RHO_MAX(T))
    (isfinite(rho_new) && rho_new > zero(T)) || return false
    band = ws.settings.adaptive_rho_tolerance
    if rho_new > ws.rho * band || rho_new < ws.rho / band
        set_rho_vec!(ws, rho_new)
        refactor!(ws)
        return true
    end
    return false
end
