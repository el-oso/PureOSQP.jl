"""
    conforms(alg::QPAlgorithm; eps, slow_iters, problems = nothing)

Assert the guarantees [`Solution`](@ref) and [`Status`](@ref) carry, for `alg`.

The [`QPAlgorithm`](@ref) and [`QPWorkspace`](@ref) contracts say which methods an algorithm
must define. They cannot say what the values those methods return have to mean, and that is
what this checks: that stopping early is never reported as success, that a run with no point
fills `x` and `y` with `NaN` rather than a plausible number, that the reported objective is
the objective, and that the timings are real. An algorithm that got any of these wrong would
still satisfy every contract and still pass a suite written around its own behavior.

`eps` is a tolerance the algorithm can reach on a small dense QP, and `slow_iters` an
iteration budget too small to converge on one. Both differ by algorithm — operator splitting
needs thousands of iterations where an interior-point method needs ten — so they are given by
the caller rather than assumed here. `problems` overrides the built-in fixtures with a
`(feasible, infeasible)` pair of `(P, q, A, l, u)` tuples.

Requires Test to be loaded, and runs inside the caller's `@testset`, so a failure is reported
against the test item that called it.

```julia
using Test, PureQPBase, PureOSQP
PureQPBase.conforms(OperatorSplitting(); eps = 1.0e-9, slow_iters = 5)
```
"""
function conforms end
