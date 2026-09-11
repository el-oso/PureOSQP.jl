"""
Accelerates the ADMM iteration with Anderson extrapolation.

ADMM converges because its step is non-expansive: the fixed-point residual never grows.
That also bounds how fast it can go. Anderson extrapolation takes a window of past iterates
and solves a small least-squares problem for a combination that ought to be closer to the
fixed point, which reaches the same tolerance in fewer steps and gives up the non-expansive
guarantee to do it.

The guarantee is bought back by checking each proposal: one whose residual came out worse
than the plain step's by more than `safeguard_tol` is discarded, and the step is retaken from
the last point the accelerator did not touch. Without that an accelerated run can stall where
the plain one converges.

Pass `accelerator = PureOSQP.anderson(Float64, n + m)` to [`PureOSQP.setup`](@ref). Nothing is
accelerated unless you ask.
"""
module PureOSQPCOSMOAcceleratorsExt

using PureOSQP
using COSMOAccelerators
using LinearAlgebra

const CA = COSMOAccelerators

"""
    AndersonState{T,A}

An `AndersonAccelerator` and the buffers the solver needs to hand it a fixed-point vector.

`w` holds the current `[x; ρ⁻¹ ⊙ y + z]` and `w_prev` the one the last step ran from, which
is the pair Anderson works on. `nrm_plain` carries the residual the plain step would have
had from the proposal to the check that judges it; it is read off the accelerator's own `f`,
so producing it costs nothing.
"""
mutable struct AndersonState{T <: AbstractFloat, A}
    accel::A
    w::Vector{T}
    w_prev::Vector{T}
    safeguard_tol::T
    nrm_plain::T
    guarding::Bool
    declined::Int
end

"""
    PureOSQP.anderson(T, dim; memory = 10, safeguard_tol = 2.0)

Build an Anderson accelerator over element type `T` for a problem whose fixed-point vector
has length `dim`, which is `n + m`.

`memory` is how many past iterates the extrapolation is drawn from. `safeguard_tol` is how
much worse than the plain step a proposal may be and still be kept: at the default of `2` a
proposal may double the residual, which sounds loose but suits a method that is erratic step
to step and wins over a window. `Inf` keeps every proposal, which is not safe on a problem
you have not already run.

`T` must be a floating-point type. An element type outside `AbstractFloat` —
`ForwardDiff.Dual` among them — reaches the refusal in the core instead.
"""
function PureOSQP.anderson(
        ::Type{T}, dim::Integer; memory::Integer = 10, safeguard_tol::Real = 2.0
    ) where {T <: AbstractFloat}
    accel = CA.AndersonAccelerator{T}(Int(dim); mem = Int(memory))
    return AndersonState{T, typeof(accel)}(
        accel, zeros(T, dim), zeros(T, dim), T(safeguard_tol), zero(T), false, 0
    )
end

"An `AndersonState` is already what the workspace holds; sizes are checked against it here."
function PureOSQP.init_accelerator(
        state::AndersonState{T}, ::Type{T}, n::Integer, m::Integer
    ) where {T <: AbstractFloat}
    length(state.w) == n + m || throw(
        DimensionMismatch("the accelerator was built for a different problem size")
    )
    return state
end

function PureOSQP.accelerate_pre!(state::AndersonState{T}, ws, iter::Integer) where {T}
    state.guarding = false
    PureOSQP.pack_fixed_point!(state.w, ws)
    if iter > 1
        CA.update!(state.accel, state.w, state.w_prev, iter)
        CA.accelerate!(state.w, state.w_prev, state.accel, iter)
        if CA.was_successful(state.accel)
            # `f` is `x - g`, the residual the plain step would have had.
            state.nrm_plain = norm(state.accel.f, 2)
            state.guarding = isfinite(state.safeguard_tol)
            PureOSQP.unpack_fixed_point!(ws, state.w)
            PureOSQP.pack_fixed_point!(state.w, ws)
        end
    end
    copyto!(state.w_prev, state.w)
    return ws
end

function PureOSQP.accelerate_post!(state::AndersonState{T}, ws, iter::Integer) where {T}
    state.guarding || return ws
    PureOSQP.pack_fixed_point!(state.w, ws)
    state.w .= state.w_prev .- state.w
    if norm(state.w, 2) > state.safeguard_tol * state.nrm_plain
        state.declined += 1
        copyto!(state.w, state.accel.g_last)
        PureOSQP.unpack_fixed_point!(ws, state.w)
        copyto!(state.w_prev, state.w)
        PureOSQP.admm_step!(ws)
    end
    return ws
end

end
