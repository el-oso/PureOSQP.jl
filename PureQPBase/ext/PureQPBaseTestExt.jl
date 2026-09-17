"""
The behavioral half of the algorithm contract: what [`PureQPBase.Solution`](@ref) and
[`PureQPBase.Status`](@ref) mean, asserted against an algorithm.

This lives in an extension because the assertions need Test, which a solver needs only while
testing. Nothing here is reachable from a solve, so the entry points and their `--trim`
verification are untouched.
"""
module PureQPBaseTestExt

using PureQPBase
using LinearAlgebra
using Test

"""
    conformance_problems() -> (feasible, infeasible)

Two small dense QPs as `(P, q, A, l, u)`. The first has a strictly convex objective and an
interior point, so every algorithm reaches it. The second bounds one row above and below on
opposite sides of the same value, so no `x` satisfies it and the run must end without a point.
"""
function conformance_problems()
    n = 6
    X = Float64[(i * 7 + j * 3) % 5 + (i == j) * 6 for i in 1:n, j in 1:n]
    P = Matrix(Symmetric(X'X / n + 2I))
    q = Float64[(-1)^i * (i / n) for i in 1:n]
    A = Float64[(i + 2j) % 4 - 1 for i in 1:(n + 2), j in 1:n]
    x0 = Float64[i / n - 0.5 for i in 1:n]
    b = A * x0
    feasible = (P, q, A, b .- 0.5, b .+ 0.5)
    # One row asked for two values at once: `a'x <= -1` together with `a'x >= 1`.
    Ai = [A; A[1:1, :]]
    li = [b .- 0.5; b[1] + 1.0]
    ui = [b .+ 0.5; Inf]
    li[1] = -Inf
    ui[1] = b[1] - 1.0
    infeasible = (P, q, Ai, li, ui)
    return (; feasible, infeasible)
end

function PureQPBase.conforms(
        alg::PureQPBase.QPAlgorithm; eps::Real, slow_iters::Integer, problems = nothing
    )
    probs = isnothing(problems) ? conformance_problems() : problems
    P, q, A, l, u = probs.feasible
    tol = (eps_abs = eps, eps_rel = eps)

    # Stopping before convergence is never reported as success, and the point reached is
    # still returned rather than discarded.
    early = PureQPBase.solve(P, q, A, l, u, alg; tol..., max_iter = slow_iters)
    @test early.status !== PureQPBase.SOLVED
    @test PureQPBase.has_solution(early.status)
    @test all(isfinite, early.x)
    @test all(isfinite, early.y)
    @test early.iter <= slow_iters

    # A converged run agrees with its own reported numbers.
    ok = PureQPBase.solve(P, q, A, l, u, alg; tol..., max_iter = 200_000)
    @test ok.status === PureQPBase.SOLVED
    @test PureQPBase.has_solution(ok.status)
    @test ok.obj_val ≈ 0.5 * dot(ok.x, P, ok.x) + dot(q, ok.x) rtol = 1.0e-7
    # One number bounds how far the point is from optimal, so it cannot be under either
    # residual it summarizes.
    @test ok.rel_kkt_error >= max(ok.prim_res, ok.dual_res) - 1.0e-12
    @test ok.prim_res >= 0 && ok.dual_res >= 0

    # The timings are measured, not left at zero, and the parts do not exceed the whole.
    @test ok.setup_time > 0
    @test ok.solve_time > 0
    @test ok.run_time >= ok.setup_time + ok.solve_time - 1.0e-9

    # `Solution` renders on one line, whatever produced it.
    @test iszero(count('\n', sprint(show, ok)))
    @test iszero(count('\n', sprint(show, early)))

    # A run that reaches no point says so, and fills the answer with NaN rather than with a
    # number a caller might use.
    Pi, qi, Ai, li, ui = probs.infeasible
    bad = PureQPBase.solve(Pi, qi, Ai, li, ui, alg; tol..., max_iter = 200_000)
    @test !PureQPBase.has_solution(bad.status)
    @test all(isnan, bad.x)
    @test all(isnan, bad.y)
    # The objective is not a number the caller can use, and its sign says which way: `Inf`
    # is the infimum over an empty feasible set, `-Inf` an objective unbounded below.
    @test !isfinite(bad.obj_val)
    if bad.status === PureQPBase.PRIMAL_INFEASIBLE ||
            bad.status === PureQPBase.PRIMAL_INFEASIBLE_INACCURATE
        @test bad.obj_val == Inf
    elseif bad.status === PureQPBase.DUAL_INFEASIBLE ||
            bad.status === PureQPBase.DUAL_INFEASIBLE_INACCURATE
        @test bad.obj_val == -Inf
    end

    # The element type of the answer is the element type of the data, concretely.
    @test ok isa PureQPBase.Solution{Float64}
    return nothing
end

end # module PureQPBaseTestExt
