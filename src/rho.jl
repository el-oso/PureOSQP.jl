"Which `ρ` class a row's bounds put it in: -1 free, 1 equality, 0 inequality."
@inline function rho_class(lo::T, hi::T, loose::T, split::Bool) where {T}
    split || return Int8(0)
    lo < -loose && hi > loose && return Int8(-1)
    hi - lo < RHO_TOL(T) && return Int8(1)
    return Int8(0)
end

"The `ρ` a class gets."
@inline function rho_for(t::Int8, rho::T) where {T}
    t == Int8(-1) && return RHO_MIN(T)
    t == Int8(1) && return RHO_EQ_OVER_INEQ(T) * rho
    return rho
end

"""
    set_rho_vec!(ws, rho) -> Bool

Classify each constraint and fill `rho_vec`/`rho_inv_vec`. Rows with both bounds
effectively infinite get `RHO_MIN`, rows with `ũ - l̃ < RHO_TOL` are equalities and get
`1e3 ρ`, everything else gets `ρ`. Returns `true` if any classification changed, which is
what forces a refactorization.
"""
function set_rho_vec!(ws::Workspace{T}, rho::T) where {T}
    ws.rho = clamp(rho, RHO_MIN(T), RHO_MAX(T))
    # With `rho_is_vec = false` every row is treated as a plain inequality, so `ρ` is
    # uniform. The classification still runs, because it is what decides whether a
    # refactorization is needed when bounds move between classes.
    return classify_rho!(
        ws.constr_type, ws.rho_vec, ws.rho_inv_vec, ws.l, ws.u, ws.rho,
        INFTY(T) * MIN_SCALING(T), ws.settings.rho_is_vec
    )
end

function classify_rho!(ct::Vector{Int8}, rho_vec, rho_inv_vec, l, u, rho::T, loose, split) where {T}
    changed = false
    for i in eachindex(ct)
        t = rho_class(l[i], u[i], loose, split)
        changed |= (t != ct[i])
        ct[i] = t
        rho_vec[i] = rho_for(t, rho)
        rho_inv_vec[i] = inv(rho_vec[i])
    end
    return changed
end

function classify_rho!(ct, rho_vec, rho_inv_vec, l, u, rho::T, loose, split) where {T}
    # One whole-array comparison stands in for the loop's running `|=`, which is one device
    # synchronization rather than `m` of them.
    changed = mapreduce(
        (lo, hi, t) -> rho_class(lo, hi, loose, split) != t,
        |, l, u, ct; init = false
    )
    ct .= rho_class.(l, u, loose, split)
    rho_vec .= rho_for.(ct, rho)
    rho_inv_vec .= inv.(rho_vec)
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
    # Reported whether or not it is adopted: the estimate is what the residuals imply, and
    # seeing it sit just inside the band explains why `ρ` did not move.
    ws.rho_estimate = rho_new
    band = ws.settings.adaptive_rho_tolerance
    if rho_new > ws.rho * band || rho_new < ws.rho / band
        set_rho_vec!(ws, rho_new)
        refactor!(ws)
        ws.rho_updates += 1
        return true
    end
    return false
end
