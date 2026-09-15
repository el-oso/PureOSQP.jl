"""
    update_settings!(ws; kwargs...) -> ws
    update_settings!(ws, alg) -> ws

Replace the workspace's [`Options`](@ref), keeping every option not named in `kwargs`, or
replace its algorithm parameters with `alg`, an algorithm object of the workspace's own
algorithm whose unnamed parameters take their defaults. Either is validated exactly as at
[`setup`](@ref), so an out-of-range value throws and leaves the workspace untouched, and a
keyword that is an algorithm parameter throws, naming the algorithm object it belongs in.

On an [`OperatorSplittingWorkspace`](@ref), `rho`, `sigma` and `rho_is_vec` are built into the
factorization, so changing any of them refactorizes; everything else is free. On an
[`InteriorPointWorkspace`](@ref) nothing refactorizes: a solve resets the regularization from
the algorithm parameters before its first iteration and refactorizes every iteration after.
The settings a backend reads while solving — the `cg_*` settings of the matrix-free backend —
reach it at once.

`linsys` and `scaling` are rejected rather than honored. The backend is part of the
workspace's *type*, so assigning new options cannot change it — accepting `linsys = :kkt`
and then continuing to run the Cholesky would be a quiet lie. `scaling` is worse: the
equilibration factors are computed once, from the data `setup` saw, so turning it off
afterwards would leave `D`, `E` and `c` at their equilibrated values while flipping every
branch that tests it, and the residuals and the returned `x` and `y` would come back in
scaled space. Build a new workspace to change either.
"""
function update_settings!(ws::QPWorkspace{T}; kwargs...) where {T}
    check_option_names(kwargs)
    old = ws.options
    new = Options{T}(; settings_tuple(old)..., kwargs...)
    new.linsys === old.linsys || throw(
        ArgumentError(
            "linsys is fixed once the workspace is built, because the backend is part of " *
                "its type. Call setup again to change it."
        )
    )
    new.scaling == old.scaling || throw(
        ArgumentError(
            "scaling is fixed once the workspace is built, because the equilibration " *
                "factors come from the data setup saw. Call setup again to change it."
        )
    )
    ws.options = new
    adopt_settings!(ws.linsys, ws.algorithm, new)
    return ws
end

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

function update_settings!(ws::QPWorkspace, alg::QPAlgorithm)
    throw(
        ArgumentError(
            lazy"this workspace runs $(nameof(typeof(ws.algorithm))), not $(nameof(typeof(alg))): the algorithm is fixed once the workspace is built. Call setup again to change it."
        )
    )
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
