# Problems, reference solutions and assertions ported from the OSQP C test suite
# (tests/{basic_qp,basic_qp2,primal_dual_infeasibility,unconstrained,non_cvx}, v0.6.2).
# Data and expected solutions are upstream's; the solver settings are the ones each
# upstream test sets; TOL is upstream's TESTS_TOL.

@testitem "C suite: basic_qp solve" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P = [4.0 1.0; 1.0 2.0]
    q = [1.0, 1.0]
    A = [1.0 1.0; 1.0 0.0; 0.0 1.0; 0.0 1.0]
    l = [1.0, 0.0, 0.0, -Inf]
    u = [1.0, 0.7, 0.7, Inf]
    TOL = 1.0e-4
    # upstream: max_iter 2000, alpha 1.6, polish 1, scaling 0, warm_start 0
    s = PureOSQP.solve(
        P, q, A, l, u; max_iter = 2000, alpha = 1.6, polish = true,
        scaling = 0, warm_starting = false
    )
    @test s.status == SOLVED
    @test norm(s.x .- [0.3, 0.7], Inf) < TOL
    @test norm(s.y .- [-2.9, 0.0, 0.2, 0.0], Inf) < TOL
    @test abs(s.obj_val - 1.88) < TOL
end

@testitem "C suite: basic_qp check_termination" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P = [4.0 1.0; 1.0 2.0]
    q = [1.0, 1.0]
    A = [1.0 1.0; 1.0 0.0; 0.0 1.0; 0.0 1.0]
    l = [1.0, 0.0, 0.0, -Inf]
    u = [1.0, 0.7, 0.7, Inf]
    # upstream: max_iter 200, alpha 1.6, polish 0, scaling 0, check_termination 0,
    # warm_start 0. With the check disabled the loop always runs to max_iter, and the
    # post-loop check is what assigns the status.
    s = PureOSQP.solve(
        P, q, A, l, u; max_iter = 200, alpha = 1.6, polish = false,
        scaling = 0, check_termination = 0, warm_starting = false
    )
    @test s.iter == 200
    @test s.status == SOLVED
    @test norm(s.x .- [0.3, 0.7], Inf) < 1.0e-4
end

@testitem "C suite: basic_qp update_rho" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P = [4.0 1.0; 1.0 2.0]
    q = [1.0, 1.0]
    A = [1.0 1.0; 1.0 0.0; 0.0 1.0; 0.0 1.0]
    l = [1.0, 0.0, 0.0, -Inf]
    u = [1.0, 0.7, 0.7, Inf]
    # Upstream solves from several starting rho values and requires the same solution.
    for rho in (0.1, 0.7, 1.0e-4, 10.0)
        s = PureOSQP.solve(
            P, q, A, l, u; rho = rho, max_iter = 5000, alpha = 1.6,
            polish = true, scaling = 0, warm_starting = false
        )
        @test s.status == SOLVED
        @test norm(s.x .- [0.3, 0.7], Inf) < 1.0e-4
    end
end

@testitem "C suite: basic_qp warm start" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P = [4.0 1.0; 1.0 2.0]
    q = [1.0, 1.0]
    A = [1.0 1.0; 1.0 0.0; 0.0 1.0; 0.0 1.0]
    l = [1.0, 0.0, 0.0, -Inf]
    u = [1.0, 0.7, 0.7, Inf]
    opts = (max_iter = 2000, alpha = 1.6, scaling = 0, check_termination = 1)
    cold = PureOSQP.solve(P, q, A, l, u; warm_starting = false, opts...)
    ws = setup(P, q, A, l, u; opts...)
    warm_start!(ws; x = cold.x, y = cold.y)
    warm = PureOSQP.solve!(ws)
    @test warm.status == SOLVED
    @test warm.iter < cold.iter
    @test norm(warm.x .- cold.x, Inf) < 1.0e-3
end

@testitem "C suite: basic_qp2" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P = [11.0 0.0; 0.0 0.0]
    q = [3.0, 4.0]
    A = [-1.0 0.0; 0.0 -1.0; -1.0 3.0; 2.0 5.0; 3.0 4.0]
    l = fill(-Inf, 5)
    u = [0.0, 0.0, -15.0, 100.0, 80.0]
    TOL = 1.0e-4
    # upstream: rho 0.1, alpha 1.6, polish 1, remaining settings default
    s = PureOSQP.solve(P, q, A, l, u; rho = 0.1, alpha = 1.6, polish = true, max_iter = 4000)
    @test s.status == SOLVED
    @test norm(s.x .- [15.0, 0.0], Inf) < TOL
    @test norm(s.y .- [0.0, 508.0, 168.0, 0.0, 0.0], Inf) < TOL
    @test abs(s.obj_val - 1282.5) < TOL
end

@testitem "C suite: primal_dual_infeasibility, all four variants" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P = [1.0 0.0; 0.0 0.0]
    q = [1.0, -1.0]
    A12 = [1.0 1.0; 1.0 0.0; 0.0 1.0]
    A34 = [1.0 0.0; 1.0 0.0; 0.0 1.0]
    l = [0.0, 1.0, 1.0]
    # upstream: max_iter 2000, alpha 1.6, scaling 0; polish 1 for the feasible case only
    s1 = PureOSQP.solve(
        P, q, A12, l, [5.0, 3.0, 3.0];
        max_iter = 2000, alpha = 1.6, scaling = 0, polish = true
    )
    @test s1.status == SOLVED
    @test norm(s1.x .- [1.0, 3.0], Inf) < 1.0e-4
    @test norm(s1.y .- [0.0, -2.0, 1.0], Inf) < 1.0e-4
    @test abs(s1.obj_val - (-1.5)) < 1.0e-4

    opts = (max_iter = 2000, alpha = 1.6, scaling = 0, polish = false)
    @test PureOSQP.solve(P, q, A12, l, [0.0, 3.0, 3.0]; opts...).status == PRIMAL_INFEASIBLE
    @test PureOSQP.solve(P, q, A34, l, [2.0, 3.0, Inf]; opts...).status == DUAL_INFEASIBLE
    @test PureOSQP.solve(P, q, A34, l, [0.0, 3.0, Inf]; opts...).status == PRIMAL_INFEASIBLE
end

@testitem "C suite: unconstrained" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P = Matrix(Diagonal([0.617022, 0.92032449, 0.20011437, 0.50233257, 0.34675589]))
    q = [-1.10593508, -1.65451545, -2.3634686, 1.13534535, -1.01701414]
    x_test = [1.79237542, 1.79775228, 11.81058885, -2.26014678, 2.93293975]
    # upstream: all settings default
    s = PureOSQP.solve(
        P, q, zeros(0, 5), Float64[], Float64[]; max_iter = 10_000,
        eps_abs = 1.0e-9, eps_rel = 1.0e-9
    )
    @test s.status == SOLVED
    @test norm(s.x .- x_test, Inf) < 1.0e-4
    @test abs(s.obj_val - (-19.209752026813277)) < 1.0e-4
end

@testitem "C suite: non_cvx" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P = [2.0 5.0; 5.0 1.0]          # eigenvalues 6.52 and -3.52
    q = [3.0, 4.0]
    A = [-1.0 0.0; 0.0 -1.0; -1.0 3.0; 2.0 5.0; 3.0 4.0]
    l = fill(-Inf, 5)
    u = [0.0, 0.0, -15.0, 100.0, 80.0]
    # sigma = 1e-6 leaves P + sigma*I indefinite; upstream returns OSQP_NONCVX_ERROR
    # from setup, so setup must fail here too.
    @test_throws "not positive definite" setup(P, q, A, l, u; sigma = 1.0e-6)
    # sigma = 5 makes P + sigma*I positive definite, so setup succeeds; upstream then
    # reports NON_CVX from the diverging residuals.
    s = PureOSQP.solve(P, q, A, l, u; sigma = 5.0, max_iter = 10_000)
    @test s.status == NON_CONVEX
    @test isnan(s.obj_val)
    @test all(isnan, s.x)
end

@testitem "C suite: an indefinite P is always refused at setup" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    # Upstream's QDLDL checks the KKT inertia and errors at setup. The reduced Cholesky
    # here cannot see that on its own, because rho*AᵀA can make the reduced matrix
    # positive definite even when P is not.
    Random.seed!(77)
    n, m = 6, 10
    for trial in 1:20
        X = randn(n, n)
        P = Matrix(Symmetric(X + X'))
        eigmin(Symmetric(P)) < -1.0e-6 || continue
        A = randn(m, n)
        @test_throws "not positive definite" setup(P, randn(n), A, fill(-1.0e6, m), fill(1.0e6, m))
    end
end
