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
    ws.x, ws.x_prev = ws.x_prev, ws.x
    ws.z, ws.z_prev = ws.z_prev, ws.z
    scale_subtract!(ws.rhs_x, ws.settings.sigma, ws.x_prev, ws.q)
    subtract_scaled!(ws.rhs_z, ws.z_prev, ws.rho_inv_vec, ws.y)
    solve_system!(ws.linsys, ws, ws.rhs_x, ws.rhs_z)
    update_x!(ws.x, ws.delta_x, ws.xtilde, ws.x_prev, ws.settings.alpha)
    ws.m > 0 && update_zy!(
        ws.z, ws.y, ws.delta_y, ws.ztilde, ws.z_prev,
        ws.rho_vec, ws.rho_inv_vec, ws.l, ws.u, ws.settings.alpha
    )
    return ws
end

"Name of a status, for `verbose` output."
function status_name(s::Status)
    s === SOLVED && return "solved"
    s === SOLVED_INACCURATE && return "solved inaccurate"
    s === PRIMAL_INFEASIBLE && return "primal infeasible"
    s === PRIMAL_INFEASIBLE_INACCURATE && return "primal infeasible inaccurate"
    s === DUAL_INFEASIBLE && return "dual infeasible"
    s === DUAL_INFEASIBLE_INACCURATE && return "dual infeasible inaccurate"
    s === MAX_ITER_REACHED && return "maximum iterations reached"
    s === TIME_LIMIT_REACHED && return "time limit reached"
    s === INTERRUPTED && return "interrupted"
    s === NON_CONVEX && return "problem non convex"
    return "unsolved"
end

# The `verbose` output.
#
# Everything here writes to `Core.stdout` and formats by hand. That is not a style choice:
# `--trim` analyses this code whether or not `verbose` is ever set, and it rejects both
# Printf (its format specifications carry type parameters that do not infer) and bare
# `println(x)` (`Base.stdout` is an abstractly typed global). `Core.stdout` is a concrete
# singleton, so calls through it resolve statically; `redirect_stdout` still captures it,
# since that redirects the file descriptor.
const VERBOSE_RULE = "------------------------------------------------------------------"

"Right-align `s` in `width` columns."
function print_padded(s::String, width::Int)
    for _ in (ncodeunits(s) + 1):width
        print(Core.stdout, " ")
    end
    print(Core.stdout, s)
    return nothing
end

print_padded(v, width::Int, digits::Int) = print_padded(string(round(v; sigdigits = digits)), width)

function print_header(ws::Workspace)
    println(Core.stdout, VERBOSE_RULE)
    println(Core.stdout, "            PureOSQP - operator splitting QP solver")
    print(Core.stdout, "     n = ")
    print(Core.stdout, ws.n)
    print(Core.stdout, ", m = ")
    print(Core.stdout, ws.m)
    print(Core.stdout, ", backend = ")
    println(Core.stdout, backend_name(ws.linsys) === :cholesky ? "cholesky" : "bunchkaufman")
    print(Core.stdout, "     eps_abs = ")
    print(Core.stdout, ws.settings.eps_abs)
    print(Core.stdout, ", eps_rel = ")
    print(Core.stdout, ws.settings.eps_rel)
    print(Core.stdout, ", max_iter = ")
    print(Core.stdout, ws.settings.max_iter)
    print(Core.stdout, ", polish = ")
    println(Core.stdout, ws.settings.polish ? "on" : "off")
    println(Core.stdout, VERBOSE_RULE)
    println(Core.stdout, " iter      objective      prim res      dual res           rho")
    return nothing
end

function print_row(ws::Workspace)
    print_padded(string(ws.iter), 5)
    print_padded(ws.obj_val, 15, 6)
    print_padded(ws.prim_res, 14, 3)
    print_padded(ws.dual_res, 14, 3)
    print_padded(ws.rho, 14, 3)
    print(Core.stdout, "\n")
    return nothing
end

function print_footer(ws::Workspace)
    println(Core.stdout, VERBOSE_RULE)
    print(Core.stdout, "status:               ")
    println(Core.stdout, status_name(ws.status))
    if ws.settings.polish
        print(Core.stdout, "polish:               ")
        println(Core.stdout, ws.polished ? "successful" : "unsuccessful")
    end
    print(Core.stdout, "number of iterations: ")
    println(Core.stdout, ws.iter)
    if has_solution(ws.status)
        print(Core.stdout, "optimal objective:    ")
        println(Core.stdout, round(ws.obj_val; sigdigits = 6))
        print(Core.stdout, "primal residual:      ")
        println(Core.stdout, round(ws.prim_res; sigdigits = 3))
        print(Core.stdout, "dual residual:        ")
        println(Core.stdout, round(ws.dual_res; sigdigits = 3))
    end
    println(Core.stdout, VERBOSE_RULE)
    return nothing
end

"""
    solve!(ws) -> Solution

Run the ADMM loop on an existing workspace. Safe to call repeatedly; with
`warm_starting = true` the previous iterates are the starting point.

If the iteration limit is reached, the termination tests are retried once at ten times the
requested tolerances before the run is declared unconverged, which is where the
`*_INACCURATE` statuses come from.

`time_limit` bounds this loop and returns `TIME_LIMIT_REACHED`. It measures the loop only:
equilibration and the first factorization happen in [`setup`](@ref) and are not counted,
so on a fresh workspace the wall-clock cost of `solve` exceeds the limit by however long
setup took. The status is returned as soon as the budget is spent, without re-checking the
tolerances, so a run that stops this way reports `TIME_LIMIT_REACHED` even if its last
point would have passed. The iterates are still meaningful and
[`has_solution`](@ref PureOSQP.has_solution) accepts it, as it does `MAX_ITER_REACHED`.

An `InterruptException` raised during the loop — a `Ctrl-C` — returns `INTERRUPTED` with
the point reached rather than losing the run; its residuals are recomputed first, since an
interrupt lands wherever it lands and not on a scheduled check. Every other exception
propagates.
"""
function solve!(ws::Workspace{T}) where {T}
    s = ws.settings
    s.warm_starting || cold_start!(ws)
    ws.status = UNSOLVED
    ws.polished = false
    ws.status_polish = POLISH_NOT_PERFORMED
    ws.iter = 0
    # Per-run counters, so a second `solve!` on this workspace reports its own numbers
    # rather than the sum of both. `refactor_count` is deliberately not reset: it is a
    # property of the workspace's whole life.
    ws.rho_updates = 0
    ws.last_rel_kkt = INFTY(T)
    ws.solve_time = 0.0
    ws.polish_time = 0.0
    s.verbose && print_header(ws)
    # `time_ns` is monotonic and costs tens of nanoseconds against a per-iteration cost of
    # microseconds, but the whole check is skipped when no limit is set, so the default
    # path is exactly what it was. A limit makes the iteration count machine-dependent,
    # which is why it is off unless asked for.
    limited = isfinite(s.time_limit)
    started = time_ns()
    budget = limited ? round(UInt64, Float64(s.time_limit) * 1.0e9) : typemax(UInt64)
    try
        for iter in 1:s.max_iter
            ws.iter = iter
            admm_step!(ws)
            if limited && time_ns() - started >= budget
                # Report the residuals of the point actually reached, not the stale ones
                # from the last scheduled check.
                update_residuals!(ws)
                ws.status = TIME_LIMIT_REACHED
                s.verbose && print_row(ws)
                break
            end
            adapting = s.adaptive_rho !== :disabled && s.adaptive_rho_interval > 0 &&
                iszero(iter % s.adaptive_rho_interval)
            checking = s.check_termination > 0 && iszero(iter % s.check_termination)
            (adapting || checking || isone(iter)) || continue
            update_residuals!(ws)
            # Only on a termination check: the residuals and objective a row reports are
            # the ones that check just used, so a printed row always explains the decision
            # made alongside it.
            s.verbose && checking && print_row(ws)
            if checking
                st = check_termination(ws, false)
                if st != UNSOLVED
                    ws.status = st
                    break
                end
            end
            # The interval decides when the test is made. Under `:kkt_error` the test
            # itself is whether the error has fallen to `adaptive_rho_fraction` of what it
            # was when `ρ` last moved, so a run whose error stops falling stops retuning
            # `ρ` instead of paying for refactorizations that are not helping.
            if adapting
                allowed = s.adaptive_rho !== :kkt_error ||
                    ws.rel_kkt_error <= s.adaptive_rho_fraction * ws.last_rel_kkt
                allowed && adapt_rho!(ws) && (ws.last_rel_kkt = ws.rel_kkt_error)
            end
        end
    catch e
        e isa InterruptException || rethrow()
        # The iterates reached are a valid, if unconverged, point, so hand them back rather
        # than lose the run. The residuals are refreshed because an interrupt lands
        # wherever it lands, not on a scheduled check.
        update_residuals!(ws)
        ws.status = INTERRUPTED
    end
    if ws.status == UNSOLVED
        update_residuals!(ws)
        st = check_termination(ws, false)
        if st == UNSOLVED
            st = check_termination(ws, true)
        end
        ws.status = st == UNSOLVED ? MAX_ITER_REACHED : st
    end
    ws.solve_time = (time_ns() - started) / 1.0e9
    if (ws.status == SOLVED || ws.status == SOLVED_INACCURATE) && s.polish
        t_polish = time_ns()
        ws.status_polish = polish!(ws)
        ws.polished = ws.status_polish === POLISH_SUCCESS
        ws.polish_time = (time_ns() - t_polish) / 1.0e9
    end
    s.verbose && print_footer(ws)
    sol = build_solution(ws)
    ws.first_run = false
    # The updates belonged to this run and are now reported; the next solve counts only the
    # ones made after it.
    ws.update_time = 0.0
    # An infeasible run leaves the iterates on a diverging ray; a later solve on this
    # workspace must not resume from there.
    has_solution(ws.status) || cold_start!(ws)
    return sol
end

"""
    solution_from(ws, x, y, obj, dual_obj, gap, prim_cert, dual_cert) -> Solution

Assemble a [`Solution`](@ref), taking everything that does not depend on the outcome
directly from the workspace. The objectives and the gap are passed in because a run
without a meaningful point must not report them.
"""
function solution_from(
        ws::Workspace{T}, x, y, obj::T, dual_obj::T, gap::T, prim_cert, dual_cert
    ) where {T}
    # `Solution` holds plain `Vector`s whatever the workspace was built from: it is the
    # result a caller reads, not a buffer the solver iterates on, and leaving a GPU array
    # here would make every field access a device transfer.
    return Solution{T}(
        Vector{T}(x), Vector{T}(y), ws.status, obj, dual_obj, gap,
        ws.prim_res, ws.dual_res, ws.rel_kkt_error, ws.iter,
        ws.rho_estimate, ws.rho_updates, ws.polished, ws.status_polish,
        ws.setup_time, ws.update_time, ws.solve_time, ws.polish_time,
        # Setup is charged to the first run only; a re-solve did not pay it again. The
        # updates since the previous solve are charged here, because they are what this
        # run cost the caller.
        (ws.first_run ? ws.setup_time : 0.0) + ws.update_time + ws.solve_time + ws.polish_time,
        Vector{T}(prim_cert), Vector{T}(dual_cert),
    )
end

function build_solution(ws::Workspace{T}) where {T}
    n, m = ws.n, ws.m
    nan = T(NaN)
    if ws.status == PRIMAL_INFEASIBLE || ws.status == PRIMAL_INFEASIBLE_INACCURATE
        cert = ws.settings.scaling > 0 ? ws.E .* ws.delta_y : copy(ws.delta_y)
        nc = norm_inf(cert)
        nc > zero(T) && (cert ./= nc)
        return solution_from(
            ws, fill(nan, n), fill(nan, m), T(Inf), nan, nan, cert, T[]
        )
    elseif ws.status == DUAL_INFEASIBLE || ws.status == DUAL_INFEASIBLE_INACCURATE
        cert = ws.settings.scaling > 0 ? ws.D .* ws.delta_x : copy(ws.delta_x)
        nc = norm_inf(cert)
        nc > zero(T) && (cert ./= nc)
        return solution_from(
            ws, fill(nan, n), fill(nan, m), T(-Inf), nan, nan, T[], cert
        )
    elseif !has_solution(ws.status)
        # NON_CONVEX and anything else without a meaningful point: no number here would
        # mean anything, so do not hand back one that looks like a solution.
        return solution_from(ws, fill(nan, n), fill(nan, m), nan, nan, nan, T[], T[])
    end
    x = ws.D .* ws.x
    y = (ws.E .* ws.y) ./ ws.c
    return solution_from(
        ws, x, y, ws.obj_val, ws.dual_obj_val, ws.duality_gap, T[], T[]
    )
end

# `@constprop :aggressive` for the same reason [`setup`](@ref) carries it, at the second of two
# barriers: without it the keyword values arrive at `setup`'s call site as runtime values, so
# `setup`'s own annotation has no constants to propagate. Both are needed; either alone leaves
# the widening in place.
"""
    solve(P, q, A, l, u; x0 = nothing, y0 = nothing, kwargs...) -> Solution

Solve `min ½xᵀPx + qᵀx  s.t.  l ≤ Ax ≤ u` in one call.
"""
Base.@constprop :aggressive function solve(
        P::AbstractMatrix, q::AbstractVector, A::AbstractMatrix,
        l::AbstractVector, u::AbstractVector;
        x0 = nothing, y0 = nothing, kwargs...
    )
    ws = setup(P, q, A, l, u; kwargs...)
    (isnothing(x0) && isnothing(y0)) || warm_start!(ws; x = x0, y = y0)
    return solve!(ws)
end
