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
