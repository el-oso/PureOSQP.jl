"""
Makes [`PureOSQP.solve`](@ref) differentiable for every AD backend that consumes ChainRules.

One `rrule` and one `frule` over the implicit derivatives this package already computes, so a
QP can sit inside a loss and be trained through. Zygote reaches this; Mooncake needs an
explicit `Mooncake.@from_rrule`, and Enzyme its own `EnzymeRules` shim.

**These rules differentiate the solution, not the iteration.** Both call
[`PureOSQP.adjoint_derivative`](@ref) or [`PureOSQP.forward_derivative`](@ref), which
differentiate the KKT conditions at the active set — one linear solve, reusing a
factorization the solve already produced, and independent of how many iterations were taken.
Differentiating the ADMM loop instead would tape every iteration, cost memory in proportion,
and return the derivative of the iterate rather than of the solution.

Two consequences follow, and both are refusals rather than approximations. The rules require
a converged solve, because the KKT conditions they differentiate hold at the solution and
nowhere else. And they inherit `adjoint_derivative`'s refusal on a degenerate active set: a
least-squares answer there would carry the right shape and units while being a different
quantity, and nothing downstream could tell.
"""
module PureOSQPChainRulesCoreExt

using PureOSQP: PureOSQP
using ChainRulesCore: ChainRulesCore, NoTangent, ZeroTangent, unthunk, @thunk

"""
    differentiable_workspace(P, q, A, l, u; kwargs...)

Solve, and refuse to hand back a workspace whose solution cannot be differentiated.

`polish = true` unless the caller said otherwise: the derivative is taken at the active set,
and polishing is what identifies it exactly. Without it the active set is whatever the ADMM
iterate happened to be near, and the gradient is of a nearby problem.
"""
function differentiable_workspace(P, q, A, l, u; kwargs...)
    ws = PureOSQP.setup(P, q, A, l, u; polish = true, kwargs...)
    sol = PureOSQP.solve!(ws)
    sol.status === PureOSQP.SOLVED || throw(
        ArgumentError(
            "cannot differentiate a solve that ended $(sol.status): the KKT conditions the " *
                "derivative differentiates hold at the solution. Tighten eps_abs/eps_rel or " *
                "raise max_iter."
        )
    )
    return ws, sol
end

function ChainRulesCore.rrule(
        ::typeof(PureOSQP.solve), P, q, A, l, u; kwargs...
    )
    ws, sol = differentiable_workspace(P, q, A, l, u; kwargs...)
    function solve_pullback(Δsol)
        # A loss reads `sol.x`, or `sol.y`, or both; whatever it did not read arrives as a
        # zero tangent rather than an array.
        Δ = unthunk(Δsol)
        dx = tangent_or_zeros(Δ, :x, length(sol.x))
        dy = tangent_or_zeros(Δ, :y, length(sol.y))
        g = PureOSQP.adjoint_derivative(ws, dx, dy)
        return (
            NoTangent(), @thunk(g.dP), @thunk(g.dq), @thunk(g.dA), @thunk(g.dl), @thunk(g.du),
        )
    end
    return sol, solve_pullback
end

function ChainRulesCore.frule(
        (_, ΔP, Δq, ΔA, Δl, Δu), ::typeof(PureOSQP.solve), P, q, A, l, u; kwargs...
    )
    ws, sol = differentiable_workspace(P, q, A, l, u; kwargs...)
    dx, dy = PureOSQP.forward_derivative(
        ws;
        dP = as_perturbation(ΔP, P), dq = as_perturbation(Δq, q),
        dA = as_perturbation(ΔA, A), dl = as_perturbation(Δl, l),
        du = as_perturbation(Δu, u),
    )
    # `Solution` is immutable and holds more than the two vectors that move, so the tangent
    # is a `Tangent` over the fields a perturbation of the data actually reaches.
    return sol, ChainRulesCore.Tangent{typeof(sol)}(x = dx, y = dy)
end

"`Δ.field` as a plain vector, or zeros when the loss did not read that field."
function tangent_or_zeros(Δ, field::Symbol, n::Int)
    Δ isa Union{ZeroTangent, NoTangent} && return zeros(n)
    v = getproperty(Δ, field)
    (v isa Union{ZeroTangent, NoTangent}) && return zeros(n)
    return collect(unthunk(v))
end

"A tangent as something `forward_derivative` can use, or `nothing` for no perturbation."
as_perturbation(Δ, ::Any) = Δ isa Union{ZeroTangent, NoTangent} ? nothing : unthunk(Δ)

end # module PureOSQPChainRulesCoreExt
