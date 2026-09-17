"""
    polish!(ws) -> PolishStatus

Guess the active set from the ADMM iterates, solve the resulting equality-constrained QP
exactly, and adopt the result only if both residuals improve. Delegates to
[`polish_kernel!`](@ref); on `POLISH_SUCCESS` it copies the polished point into `ws.x`,
`ws.y` and `ws.z` and recomputes the residuals.
"""
function polish!(ws::OperatorSplittingWorkspace{T}) where {T}
    prob = ws.prob
    status, xpol, ypol, zpol = polish_kernel!(
        prob, ws.x, ws.y, ws.z, ws.prim_res, ws.dual_res, ws.Ax, ws.Px, ws.Aty;
        delta = ws.options.delta, refine_iter = ws.options.polish_refine_iter
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
