@testitem "update! matches a fresh setup" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(10, 24; seed = 60)
    opts = (eps_abs = 1.0e-9, eps_rel = 1.0e-9, max_iter = 100_000)
    ws = setup(P, q, A, l, u; opts...)
    PureOSQP.solve!(ws)
    Random.seed!(61)
    for _ in 1:5
        q2 = randn(10)
        b = A * randn(10)
        l2, u2 = b .- rand(24), b .+ rand(24)
        update!(ws; q = q2, l = l2, u = u2)
        got = PureOSQP.solve!(ws)
        want = PureOSQP.solve(P, q2, A, l2, u2; opts...)
        @test got.status == SOLVED
        @test want.status == SOLVED
        @test got.x ≈ want.x rtol = 1.0e-5
        @test abs(got.obj_val - want.obj_val) <= 1.0e-6 * max(1, abs(want.obj_val))
        @test maximum(kkt_residuals(P, q2, A, l2, u2, got.x, got.y)) < 1.0e-5
    end
end

@testitem "update! refactorizes only when it must" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P = [4.0 1.0; 1.0 2.0]
    A = [1.0 1.0; 1.0 0.0]
    ws = setup(P, [1.0, 1.0], A, [0.0, 0.0], [1.0, 1.0])
    @test ws.refactor_count == 1
    # A linear-cost change touches no factorization.
    update!(ws; q = [2.0, 3.0])
    @test ws.refactor_count == 1
    # Bounds that keep every row an inequality do not either.
    update!(ws; l = [0.0, 0.1], u = [1.0, 0.9])
    @test ws.refactor_count == 1
    # Turning a row into an equality changes its rho, which is baked into the factorization.
    update!(ws; l = [0.5, 0.1], u = [0.5, 0.9])
    @test ws.refactor_count == 2
    @test ws.constr_type[1] == Int8(1)
    # Turning it back changes it again.
    update!(ws; l = [0.0, 0.1], u = [1.0, 0.9])
    @test ws.refactor_count == 3
    # Matrix updates always do.
    update!(ws; P = [5.0 1.0; 1.0 3.0])
    @test ws.refactor_count == 4
    update!(ws; A = [1.0 2.0; 1.0 0.0])
    @test ws.refactor_count == 5
end

@testitem "update! of P and A gives the same answer as a fresh setup" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(8, 20; seed = 62)
    P2, _, A2, _, _ = random_qp(8, 20; seed = 63)
    opts = (eps_abs = 1.0e-9, eps_rel = 1.0e-9, max_iter = 100_000)
    ws = setup(P, q, A, l, u; opts...)
    PureOSQP.solve!(ws)
    b = A2 * randn(8)
    l2, u2 = b .- rand(20), b .+ rand(20)
    update!(ws; P = P2, A = A2, l = l2, u = u2)
    got = PureOSQP.solve!(ws)
    want = PureOSQP.solve(P2, q, A2, l2, u2; opts...)
    @test got.status == SOLVED
    @test got.x ≈ want.x rtol = 1.0e-5
    @test maximum(kkt_residuals(P2, q, A2, l2, u2, got.x, got.y)) < 1.0e-5
end

@testitem "update! warm starts the next solve" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(12, 30; seed = 64)
    opts = (eps_abs = 1.0e-8, eps_rel = 1.0e-8, max_iter = 100_000)
    ws = setup(P, q, A, l, u; opts...)
    first = PureOSQP.solve!(ws)
    # A small perturbation should be cheap from the previous solution.
    q2 = q .+ 1.0e-3 .* randn(12)
    update!(ws; q = q2)
    second = PureOSQP.solve!(ws)
    cold = PureOSQP.solve(P, q2, A, l, u; opts...)
    @test second.status == SOLVED
    @test second.iter < cold.iter
    @test second.x ≈ cold.x rtol = 1.0e-4
end

@testitem "update! validates its arguments" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P = [4.0 1.0; 1.0 2.0]
    A = [1.0 1.0; 1.0 0.0]
    ws = setup(P, [1.0, 1.0], A, [0.0, 0.0], [1.0, 1.0])
    @test_throws "length(q) must be 2" update!(ws; q = [1.0, 2.0, 3.0])
    @test_throws "q must be finite" update!(ws; q = [NaN, 1.0])
    @test_throws "length(l) must be 2" update!(ws; l = [0.0])
    @test_throws "l must be elementwise ≤ u" update!(ws; l = [2.0, 0.0])
    @test_throws "l may not be +Inf" update!(ws; l = [Inf, 0.0])
    @test_throws "P must stay 2×2" update!(ws; P = Matrix(1.0I, 3, 3))
    @test_throws "P must be symmetric" update!(ws; P = [1.0 2.0; 0.0 1.0])
    @test_throws "not positive definite" update!(ws; P = [2.0 5.0; 5.0 1.0])
    @test_throws "A must stay 2×2" update!(ws; A = ones(3, 2))
end
