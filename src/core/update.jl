"""
    update!(ws; q = nothing, l = nothing, u = nothing, P = nothing, A = nothing) -> OperatorSplittingWorkspace

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

`P` and `A` must also keep the representation the workspace was built with. The backend is
part of the workspace's type, and the structured backends in particular read only the
structure they were selected on — a denser or differently structured replacement would be
solved through a structure it does not have, which is a quiet wrong answer rather than an
error. Rebuild the workspace with [`setup`](@ref) to change the representation. The
structured backends' data-level invariants are re-checked as well: the Kronecker backend's
scalar `P` (its `μ` is refreshed, not re-guessed), its uniform `ρ`, the block partition,
and the coupling rank.
"""
function update!(
        ws::OperatorSplittingWorkspace{T}; q = nothing, l = nothing, u = nothing, P = nothing, A = nothing
    ) where {T}
    t0 = time_ns()
    prob = ws.prob
    validate_update!(prob, ws.linsys; P, A, q, l, u)

    # What ADMM requires beyond the data being well formed: convexity at its `σ`, and the
    # uniform `ρ` the kronecker backend's diagonalization needs.
    !isnothing(P) && !is_convex(T, P, ws.algorithm.sigma) &&
        throw(ArgumentError("P + sigma*I is not positive definite: P is indefinite, so the problem is not convex."))
    if (!isnothing(l) || !isnothing(u)) && ws.linsys isa KroneckerReduced && prob.m > 0
        loose = INFTY(T) * MIN_SCALING(T)
        split = ws.algorithm.rho_is_vec
        lprop = isnothing(l) ? prob.l0 : max.(T.(l), -INFTY(T))
        uprop = isnothing(u) ? prob.u0 : min.(T.(u), INFTY(T))
        first_class = rho_class(prob.E[1] * lprop[1], prob.E[1] * uprop[1], loose, split)
        for i in 2:prob.m
            rho_class(prob.E[i] * lprop[i], prob.E[i] * uprop[i], loose, split) !=
                first_class && throw(
                ArgumentError(
                    "the new bounds put constraint rows in different ρ classes: the " *
                        "kronecker backend requires a uniform ρ and has no form for " *
                        "a split one. Rebuild the workspace with setup."
                )
            )
        end
    end

    adopt_update!(prob; P, A, q, l, u)
    refactor_needed = !isnothing(P) || !isnothing(A)
    if !isnothing(l) || !isnothing(u)
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

"""
    validate_update!(prob, ls; P, A, q, l, u) -> Nothing

Throw unless the proposed data can replace what `prob` holds: dimensions, representation,
symmetry, finiteness and bound ordering, plus whatever the backend `ls` needs through
[`check_update`](@ref). Reads the arguments and `prob`, and writes nothing.

Convexity is not checked here, because the shift it is checked at belongs to the algorithm.
"""
function validate_update!(
        prob::Problem{T, MP, MA}, ls; P = nothing, A = nothing, q = nothing, l = nothing,
        u = nothing
    ) where {T, MP, MA}
    n, m = prob.n, prob.m
    if !isnothing(P)
        size(P) == (n, n) || throw(ArgumentError("P must stay $(n)×$(n), got $(size(P))"))
        P isa MP || throw(
            ArgumentError(
                "P must keep the representation the workspace was built with: its linear-" *
                    "system backend is built for that representation and would read only the " *
                    "structure it implies. Rebuild the workspace with setup to change it."
            )
        )
        is_symmetric(P) || throw(ArgumentError("P must be symmetric"))
        is_materializable(P) || check_symmetric_products(P, prob.q0)
        is_materializable(P) && check_finite(P, n, n, "P")
        check_storage(P, n, n)
    end
    if !isnothing(A)
        size(A) == (m, n) || throw(ArgumentError("A must stay $(m)×$(n), got $(size(A))"))
        A isa MA || throw(
            ArgumentError(
                "A must keep the representation the workspace was built with: its linear-" *
                    "system backend is built for that representation and would read only the " *
                    "structure it implies. Rebuild the workspace with setup to change it."
            )
        )
        is_materializable(A) && check_finite(A, m, n, "A")
        check_storage(A, m, n)
    end
    if !isnothing(P) || !isnothing(A)
        check_update(ls, isnothing(P) ? prob.P : P, isnothing(A) ? prob.A : A)
    end
    if !isnothing(q)
        length(q) == n || throw(ArgumentError("length(q) must be $n, got $(length(q))"))
        all(isfinite, q) || throw(ArgumentError("q must be finite, found NaN or Inf"))
    end
    if !isnothing(l) || !isnothing(u)
        # Lengths first: the walks below index every row of both proposals, and a short one
        # would reach the end of a vector rather than this message.
        isnothing(l) || length(l) == m ||
            throw(ArgumentError("length(l) must be $m, got $(length(l))"))
        isnothing(u) || length(u) == m ||
            throw(ArgumentError("length(u) must be $m, got $(length(u))"))
        inf = INFTY(T)
        if !isnothing(l)
            any(isnan, l) && throw(ArgumentError("l contains NaN"))
            any(li -> li == Inf, l) && throw(ArgumentError("l may not be +Inf"))
        end
        if !isnothing(u)
            any(isnan, u) && throw(ArgumentError("u contains NaN"))
            any(ui -> ui == -Inf, u) && throw(ArgumentError("u may not be -Inf"))
        end
        # The ordering test runs on the clamped proposals, not on what the workspace holds,
        # so a pair that fails it leaves the old bounds in place.
        for i in 1:m
            li = isnothing(l) ? prob.l0[i] : max(T(l[i]), -inf)
            ui = isnothing(u) ? prob.u0[i] : min(T(u[i]), inf)
            li <= ui ||
                throw(ArgumentError("l must be elementwise ≤ u, violated at index $i: $li > $ui"))
        end
    end
    return nothing
end

"""
    adopt_update!(prob; P, A, q, l, u) -> prob

Replace what `prob` holds with the data given, and reapply the existing equilibration to it.
Runs only after [`validate_update!`](@ref) and every algorithm-specific check have passed.
"""
function adopt_update!(
        prob::Problem{T}; P = nothing, A = nothing, q = nothing, l = nothing, u = nothing
    ) where {T}
    # Every check reads the arguments, never the problem's own fields, so the matrices are
    # adopted only once none of them can refuse. A refusal that had already replaced `P` or
    # `A` would leave the workspace holding a matrix its factorization and its buffers were
    # not built for, and the next solve reads out of range.
    isnothing(P) || (prob.P = P)
    isnothing(A) || (prob.A = A)
    isnothing(q) || (prob.q0 .= q)
    isnothing(l) || (prob.l0 .= max.(T.(l), -INFTY(T)))
    isnothing(u) || (prob.u0 .= min.(T.(u), INFTY(T)))

    # Reapply the existing equilibration to whatever changed.
    isnothing(q) || (prob.q .= prob.c .* prob.D .* prob.q0)
    if !isnothing(l) || !isnothing(u)
        prob.l .= prob.E .* prob.l0
        prob.u .= prob.E .* prob.u0
    end
    return prob
end
