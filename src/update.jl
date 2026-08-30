"""
    update!(ws; q = nothing, l = nothing, u = nothing, P = nothing, A = nothing) -> Workspace

Replace problem data in an existing workspace, keeping the equilibration factors, the
buffers and the current iterates.

This is the sequential-resolve path: a model-predictive or sequential-quadratic loop
changes `q`, `l` and `u` every step while `P` and `A` stay fixed, and re-running [`setup`](@ref)
would repeat the equilibration sweeps and the factorization for nothing.

What each update costs:

- `q` alone: rescaling one vector. No factorization.
- `l`, `u`: rescaling, plus reclassifying every row as equality, inequality or free. A
  factorization is needed only if that classification changed, because the classification
  is what sets `ρ`.
- `P`, `A`: always a factorization.

Equilibration is **not** recomputed — the factors `D`, `E`, `c` from `setup` are reused, as
in the reference implementation. They stay appropriate while the data keeps roughly the
same scale; after a large change in magnitude, build a fresh workspace instead.

`P` and `A` must keep their dimensions, and `P` must stay symmetric with `P + σI` positive
definite, which is checked.
"""
function update!(
        ws::Workspace{T}; q = nothing, l = nothing, u = nothing,
        P = nothing, A = nothing
    ) where {T}
    t0 = time_ns()
    n, m = ws.n, ws.m
    refactor_needed = false

    if !isnothing(P)
        size(P) == (n, n) || throw(ArgumentError("P must stay $(n)×$(n), got $(size(P))"))
        issymmetric(P) || throw(ArgumentError("P must be symmetric"))
        is_convex(T, P, ws.settings.sigma) ||
            throw(ArgumentError("P + sigma*I is not positive definite: P is indefinite, so the problem is not convex."))
        ws.P = P
        refactor_needed = true
    end
    if !isnothing(A)
        size(A) == (m, n) || throw(ArgumentError("A must stay $(m)×$(n), got $(size(A))"))
        ws.A = A
        refactor_needed = true
    end
    if !isnothing(q)
        length(q) == n || throw(ArgumentError("length(q) must be $n, got $(length(q))"))
        all(isfinite, q) || throw(ArgumentError("q must be finite, found NaN or Inf"))
        ws.q0 .= q
    end
    if !isnothing(l) || !isnothing(u)
        inf = INFTY(T)
        if !isnothing(l)
            length(l) == m || throw(ArgumentError("length(l) must be $m, got $(length(l))"))
            any(isnan, l) && throw(ArgumentError("l contains NaN"))
            any(li -> li == Inf, l) && throw(ArgumentError("l may not be +Inf"))
            ws.l0 .= max.(T.(l), -inf)
        end
        if !isnothing(u)
            length(u) == m || throw(ArgumentError("length(u) must be $m, got $(length(u))"))
            any(isnan, u) && throw(ArgumentError("u contains NaN"))
            any(ui -> ui == -Inf, u) && throw(ArgumentError("u may not be -Inf"))
            ws.u0 .= min.(T.(u), inf)
        end
        for i in 1:m
            ws.l0[i] <= ws.u0[i] ||
                throw(ArgumentError("l must be elementwise ≤ u, violated at index $i: $(ws.l0[i]) > $(ws.u0[i])"))
        end
    end

    # Reapply the existing equilibration to whatever changed.
    isnothing(q) || (ws.q .= ws.c .* ws.D .* ws.q0)
    if !isnothing(l) || !isnothing(u)
        ws.l .= ws.E .* ws.l0
        ws.u .= ws.E .* ws.u0
        # A row that becomes (or stops being) an equality or a free row changes its rho,
        # and rho is baked into the factorization.
        refactor_needed |= set_rho_vec!(ws, ws.rho)
    end

    refactor_needed && refactor!(ws)
    # Accumulated, not assigned: a caller typically makes several calls before solving --
    # `q`, then `l` and `u`, then perhaps `P` -- and all of them belong to the next solve,
    # which reports the total and resets it.
    ws.update_time += (time_ns() - t0) / 1.0e9
    return ws
end
