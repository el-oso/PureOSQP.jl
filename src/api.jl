"""
    settings_tuple(s) -> NamedTuple

`s` as a keyword-ready `NamedTuple`. `Val(fieldcount(...))` keeps the loop unrolled, so the
field accesses resolve statically.
"""
function settings_tuple(s::Settings{T}) where {T}
    return NamedTuple{fieldnames(Settings{T})}(
        ntuple(i -> getfield(s, i), Val(fieldcount(Settings{T})))
    )
end

"""
    update_settings!(ws; kwargs...) -> ws

Replace the workspace's settings, keeping every field not named in `kwargs`. The keywords
are those of [`Settings`](@ref) and are validated exactly as at [`setup`](@ref), so an
out-of-range value throws and leaves the workspace untouched.

`rho`, `sigma` and `rho_is_vec` are built into the factorization, so changing any of them
refactorizes; everything else is free.

`linsys` and `scaling` are rejected rather than honored. The backend is part of the
workspace's *type*, so assigning a new `Settings` cannot change it — accepting
`linsys = :kkt` and then continuing to run the Cholesky would be a quiet lie. `scaling` is
worse: the equilibration factors are computed once, from the data `setup` saw, so turning
it off afterwards would leave `D`, `E` and `c` at their equilibrated values while flipping
every branch that tests it, and the residuals and the returned `x` and `y` would come back
in scaled space. Build a new workspace to change either.
"""
function update_settings!(ws::Workspace{T}; kwargs...) where {T}
    old = ws.settings
    new = Settings{T}(; settings_tuple(old)..., kwargs...)
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
    # Compared field by field rather than by looping over a tuple of symbols: `getfield`
    # with a symbol the compiler cannot see is a dynamic call, and `--trim` rejects it.
    refactor_needed = new.rho != old.rho || new.sigma != old.sigma ||
        new.rho_is_vec != old.rho_is_vec
    ws.settings = new
    if refactor_needed
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

`ws.settings.rho` keeps the value [`setup`](@ref) was given; the live value is `ws.rho`.
"""
function update_rho!(ws::Workspace{T}, rho::Real) where {T}
    rho > 0 || throw(ArgumentError("rho must be positive, got $rho"))
    set_rho_vec!(ws, T(rho))
    refactor!(ws)
    return ws
end

"""
    dimensions(ws) -> (n, m)

Number of variables and of constraint rows.
"""
dimensions(ws::Workspace) = (ws.n, ws.m)

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
