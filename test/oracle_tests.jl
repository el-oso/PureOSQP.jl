@testitem "iterates match the C library at machine precision" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(8)
    n, m = 6, 10
    P = (X = randn(n, n); Matrix(X'X))
    q = randn(n)
    A = randn(m, n)
    l, u = -rand(m), rand(m)
    # Both linear-system backends and both scaled and unscaled space: the transcription
    # gate is only meaningful if it covers the algebra the solver actually runs.
    for backend in (:auto, :kkt), sc in (0, 10), k in (1, 2, 3, 5, 25)
        c = osqp_ref(
            P, q, A, l, u; max_iter = k, scaling = sc, adaptive_rho = false,
            eps_abs = 1.0e-12, eps_rel = 1.0e-12
        )
        j = PureOSQP.solve(
            P, q, A, l, u; max_iter = k, scaling = sc, adaptive_rho = false,
            check_termination = 0, eps_abs = 1.0e-12, eps_rel = 1.0e-12,
            linsys = backend
        )
        @test j.x ≈ c.x atol = 1.0e-10
        @test j.y ≈ c.y atol = 1.0e-10
    end
end

@testitem "objective and iteration count agree with the C library" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    for (n, m) in ((10, 20), (25, 50), (40, 15), (30, 200))
        P, q, A, l, u = random_qp(n, m; seed = n + m)
        c = osqp_ref(P, q, A, l, u; eps_abs = 1.0e-6, eps_rel = 1.0e-6, max_iter = 20_000)
        j = PureOSQP.solve(P, q, A, l, u; eps_abs = 1.0e-6, eps_rel = 1.0e-6, max_iter = 20_000)
        @test j.status == SOLVED
        @test c.info.status == :Solved
        @test abs(j.obj_val - c.info.obj_val) <= 1.0e-6 * max(1, abs(c.info.obj_val))
        @test j.iter == c.info.iter
    end
end

@testitem "status agrees with the C library on infeasible instances" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    cases = Any[
        (zeros(1, 1), [0.0], reshape([1.0, 1.0], 2, 1), [1.0, -Inf], [Inf, 0.0], PRIMAL_INFEASIBLE),
        (zeros(1, 1), [-1.0], reshape([1.0], 1, 1), [0.0], [Inf], DUAL_INFEASIBLE),
    ]
    for (P, q, A, l, u, expected) in cases
        j = PureOSQP.solve(P, q, A, l, u)
        c = osqp_ref(P, q, A, l, u)
        @test j.status == expected
        @test string(c.info.status) == (expected == PRIMAL_INFEASIBLE ? "Primal_infeasible" : "Dual_infeasible")
    end
end

@testitem "infeasibility certificates satisfy their defining inequalities" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    A = reshape([1.0, 1.0], 2, 1)
    l, u = [1.0, -Inf], [Inf, 0.0]
    s = PureOSQP.solve(zeros(1, 1), [0.0], A, l, u)
    dy = s.prim_inf_cert
    @test !isempty(dy)
    @test norm(A' * dy, Inf) < 1.0e-4 * norm(dy, Inf)
    lhs = sum(
        i -> (isfinite(u[i]) ? u[i] * max(dy[i], 0) : 0.0) +
            (isfinite(l[i]) ? l[i] * min(dy[i], 0) : 0.0), eachindex(dy)
    )
    @test lhs < 1.0e-4 * norm(dy, Inf)

    P, q = zeros(1, 1), [-1.0]
    Ad, ld, ud = reshape([1.0], 1, 1), [0.0], [Inf]
    s2 = PureOSQP.solve(P, q, Ad, ld, ud)
    dx = s2.dual_inf_cert
    @test !isempty(dx)
    @test norm(P * dx, Inf) < 1.0e-4 * norm(dx, Inf)
    @test dot(q, dx) < 0
    @test (Ad * dx)[1] > -1.0e-4 * norm(dx, Inf)
end
