"""
    init_accelerator(spec, ::Type{T}, n, m)

Turn whatever the caller passed as `accelerator` into the object the workspace holds.

`nothing` is the default and stays `nothing`, which makes every hook below a no-op resolved
at compile time. Anything else is a request the core cannot serve on its own: an extension
supplies the method that recognizes it, and without that extension the request is refused by
name rather than silently ignored.
"""
init_accelerator(::Nothing, ::Type{T}, n::Integer, m::Integer) where {T <: Real} = nothing

function init_accelerator(spec, ::Type{T}, n::Integer, m::Integer) where {T <: Real}
    return no_accelerator()
end

"""
    no_accelerator()

Throw the refusal an unrecognized `accelerator` owes its caller.

The message interpolates nothing: `show(::IO, ::Type)` is a runtime dispatch `--trim` cannot
resolve, and this branch is live code whenever an accelerator is asked for.
"""
no_accelerator() = throw(
    ArgumentError(
        "this `accelerator` is not one the solver recognizes. Acceleration comes from " *
            "COSMOAccelerators.jl, which is a weak dependency: run `using COSMOAccelerators` " *
            "before `setup`, and pass `accelerator = PureOSQP.anderson(Float64, n + m)`."
    )
)

"""
    accelerate_pre!(accel, ws, iter) -> ws

Replace the iterates with an extrapolated point before the step that follows.

ADMM is a fixed-point iteration on `w = [x; ρ⁻¹ ⊙ y + z]`. An accelerator reads the sequence
of those vectors and proposes a better one; the step then runs from the proposal instead of
from the plain iterate.

With no accelerator this is `nothing` and the solver runs the plain iteration.
"""
accelerate_pre!(::Nothing, ws, iter::Integer) = ws

"""
    accelerate_post!(accel, ws, iter) -> ws

Keep or discard the extrapolated point, after the step that used it.

ADMM never lets the fixed-point residual grow; extrapolation can. A proposal whose residual
came out worse than the plain step's by more than the accelerator's tolerance is dropped and
the step is retaken from the last point the accelerator did not touch.
"""
accelerate_post!(::Nothing, ws, iter::Integer) = ws

"""
    pack_fixed_point!(w, ws) -> w

Write `[x; ρ⁻¹ ⊙ y + z]`, the vector the ADMM iteration is a fixed point of.
"""
function pack_fixed_point!(w::AbstractVector{T}, ws::Workspace{T}) where {T}
    n = ws.prob.n
    @views w[1:n] .= ws.x
    @views w[(n + 1):end] .= ws.weights.w_inv .* ws.y .+ ws.z
    return w
end

"""
    unpack_fixed_point!(ws, w) -> ws

Split `w` back into `x`, `z` and `y`.

The second block is `ρ⁻¹ ⊙ y + z`, and projecting it onto the bounds separates the two: `z`
is the part inside, `y` is what is left scaled by `ρ`. Carrying the previous `y` across
instead leaves the pair inconsistent, and the iteration does not recover from that.
"""
function unpack_fixed_point!(ws::Workspace{T}, w::AbstractVector{T}) where {T}
    prob = ws.prob
    n = prob.n
    @views ws.x .= w[1:n]
    v = @view w[(n + 1):end]
    @. ws.z = clamp(v, prob.l, prob.u)
    @. ws.y = ws.weights.w * (v - ws.z)
    return ws
end

"""
    anderson(T, dim; memory = 10, safeguard_tol = 2.0)

Build an Anderson accelerator over element type `T` for a fixed-point vector of length
`dim`, which is `n + m`.

Supplied by the COSMOAccelerators extension, which defines this for the floating-point types
it can accelerate; without that package loaded, or for an element type it cannot take, this
says so rather than returning something that quietly does nothing.
"""
function anderson(::Type, dim::Integer; kwargs...)
    return throw(
        ArgumentError(
            "acceleration needs COSMOAccelerators.jl, which is a weak dependency, and a " *
                "floating-point element type: run `using COSMOAccelerators` before building " *
                "one, and build it over the type the solve is carried out in."
        )
    )
end

"""
    accelerator_reset!(accel) -> accel

Drop the history an accelerator has built up.

Called wherever the workspace refactorizes, since `ρ` or the data changing makes the
iteration a fixed point of a different map and a window spanning both describes neither.
With no accelerator this is `nothing`.
"""
accelerator_reset!(::Nothing) = nothing

"""
    accelerator_declined(accel) -> Int

How many accelerated steps `accel` has discarded over its life. `solve!` reports the
difference across one solve as `Solution.accel_declined`. Zero with no accelerator.
"""
accelerator_declined(::Nothing) = 0
