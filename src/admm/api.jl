function update_settings!(ws::OperatorSplittingWorkspace{T}, alg::OperatorSplitting) where {T}
    old = ws.algorithm
    new = element_typed(alg, T, ws.options)
    # Compared field by field rather than by looping over a tuple of symbols: `getfield`
    # with a symbol the compiler cannot see is a dynamic call, and `--trim` rejects it.
    refactor_needed = new.rho != old.rho || new.sigma != old.sigma ||
        new.rho_is_vec != old.rho_is_vec
    ws.algorithm = new
    adopt_settings!(ws.linsys, new, ws.options)
    if refactor_needed
        # `σ` is held by value in the weights, so a new one needs a new weights object; the
        # vectors are shared and refilled in place by `set_rho_vec!`.
        wt = ws.weights
        ws.weights = SystemWeights(wt.w, wt.w_inv, new.sigma)
        set_rho_vec!(ws, new.rho)
        refactor!(ws)
    end
    return ws
end

"""
    update_rho!(ws, rho) -> ws

Set the workspace's `ρ` and refactorize. `rho` is clamped to `[1e-6, 1e6]` and then split
across the constraint classes exactly as adaptive `ρ` does, so this is the same operation
the solver performs on itself, made available to the caller.

`ws.algorithm.rho` keeps the value [`setup`](@ref) was given; the live value is `ws.rho`.
"""
function update_rho!(ws::OperatorSplittingWorkspace{T}, rho::Real) where {T}
    rho > 0 || throw(ArgumentError("rho must be positive, got $rho"))
    set_rho_vec!(ws, T(rho))
    refactor!(ws)
    return ws
end

"""
    dimensions(ws) -> (n, m)

Number of variables and of constraint rows.
"""
dimensions(ws::QPWorkspace) = (ws.prob.n, ws.prob.m)

"""
    capabilities() -> NamedTuple

What this build of the solver supports, reported for the packages currently loaded rather
than for the package alone: `indirect_solver` is true once Krylov.jl is loaded and the
matrix-free extension exists. The names mirror libosqp's `osqp_capabilities`
bit-flags, so a caller porting from the C API can check the same things.

There is no `error_message` counterpart: this package throws exceptions carrying their own
messages rather than returning codes to be looked up.
"""
capabilities() = (
    direct_solver = true,
    indirect_solver = !isnothing(Base.get_extension(@__MODULE__, :PureOSQPKrylovExt)),
    codegen = false,
    update_matrices = true,
    derivatives = true,
)

"""
    constraint_violation!(out, ws) -> out

Write how far each row of `l ≤ Ax ≤ u` is from being satisfied, in the caller's units.

`out[i]` is `max(l[i] - (Ax)[i], (Ax)[i] - u[i], 0)`: zero where the row holds, and the
distance to the nearer bound where it does not. `‖out‖∞` is the primal residual
[`solve!`](@ref) reports as `prim_res`, so this is that number broken out by row.

The iterate `z` is projected into `[l, u]` every iteration and is feasible by construction;
`Ax` is what can miss, and is what this measures. A row that was one-sided on input has a
bound of `±1e30` here, far enough that it never reports a violation of its own.

`out` must have one entry per constraint row. Nothing is allocated.
"""
function constraint_violation!(out::AbstractVector{T}, ws::OperatorSplittingWorkspace{T}) where {T}
    prob = ws.prob
    length(out) == prob.m || throw(
        DimensionMismatch("out must have one entry per constraint row")
    )
    iszero(prob.m) && return out
    mul_A!(ws.Ax, prob, ws.x)
    scaled = prob.scaling > 0
    for i in eachindex(out)
        # `l`, `u` and `Ax` are all equilibrated by the same row factor, so the violation
        # comes back to the caller's units by dividing it out once.
        gap = max(prob.l[i] - ws.Ax[i], ws.Ax[i] - prob.u[i], zero(T))
        out[i] = scaled ? gap / prob.E[i] : gap
    end
    return out
end

"""
    constraint_violation(ws) -> Vector

How far each row of `l ≤ Ax ≤ u` is from being satisfied, in the caller's units.

Allocates the result; [`constraint_violation!`](@ref) writes into a vector you supply and
carries the description of what the entries mean.
"""
function constraint_violation(ws::OperatorSplittingWorkspace{T}) where {T}
    return constraint_violation!(similar(ws.x, T, ws.prob.m), ws)
end
