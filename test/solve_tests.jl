@testitem "solves the reference QP" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P = [4.0 1.0; 1.0 2.0]
    q = [1.0, 1.0]
    A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
    l = [1.0, 0.0, 0.0]
    u = [1.0, 0.7, 0.7]
    s = PureOSQP.solve(P, q, A, l, u; eps_abs = 1.0e-9, eps_rel = 1.0e-9, max_iter = 20_000)
    @test s.status == SOLVED
    @test maximum(kkt_residuals(P, q, A, l, u, s.x, s.y)) < 1.0e-6
    @test s.obj_val ≈ 1.88 atol = 1.0e-6
end

@testitem "the referee accepts every solved random instance" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    solved = Ref(0)
    for trial in 1:30
        Random.seed!(100 + trial)
        n, m = rand(3:15), rand(1:30)
        P = (X = randn(n, n); Matrix(X'X))
        A = randn(m, n) * Diagonal(exp10.(rand(-3:3, n)))
        q = randn(n)
        l = -rand(m) .- 0.1
        u = rand(m) .+ 0.1
        s = PureOSQP.solve(P, q, A, l, u; eps_abs = 1.0e-9, eps_rel = 1.0e-9, max_iter = 100_000)
        s.status == SOLVED || continue
        solved[] += 1
        @test maximum(kkt_residuals(P, q, A, l, u, s.x, s.y)) < 1.0e-5
    end
    @test solved[] >= 25
end

@testitem "equilibration is what makes a badly scaled problem tractable" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(10, 25; seed = 12, colscale = 4)
    off = PureOSQP.solve(
        P, q, A, l, u; scaling = 0, eps_abs = 1.0e-10, eps_rel = 1.0e-10,
        max_iter = 200_000
    )
    on = PureOSQP.solve(
        P, q, A, l, u; scaling = 10, eps_abs = 1.0e-10, eps_rel = 1.0e-10,
        max_iter = 200_000
    )
    @test on.status == SOLVED
    @test maximum(kkt_residuals(P, q, A, l, u, on.x, on.y)) < 1.0e-6
    # Without equilibration the same problem does not converge in 200_000 iterations.
    @test off.status == MAX_ITER_REACHED
end

@testitem "hitting max_iter is never reported as solved" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(30, 60; seed = 13)
    s = PureOSQP.solve(P, q, A, l, u; max_iter = 2, eps_abs = 1.0e-14, eps_rel = 1.0e-14)
    @test s.status == MAX_ITER_REACHED
end

@testitem "adaptive rho reduces the iteration count from a bad rho" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(25, 60; seed = 14)
    fixed = PureOSQP.solve(
        P, q, A, l, u; adaptive_rho = false, rho = 1.0e-4,
        eps_abs = 1.0e-8, eps_rel = 1.0e-8, max_iter = 100_000
    )
    adapt = PureOSQP.solve(
        P, q, A, l, u; adaptive_rho = true, rho = 1.0e-4,
        eps_abs = 1.0e-8, eps_rel = 1.0e-8, max_iter = 100_000
    )
    @test adapt.status == SOLVED
    @test adapt.iter < fixed.iter
end

@testitem "rho vector classifies equalities, inequalities and free rows" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    A = [1.0 0.0; 0.0 1.0; 1.0 1.0]
    l = [1.0, -Inf, 0.0]
    u = [1.0, Inf, 1.0]
    ws = setup(zeros(2, 2), zeros(2), A, l, u; scaling = 0, rho = 0.1)
    @test ws.rho_vec[1] ≈ 1.0e3 * 0.1
    @test ws.rho_vec[2] ≈ 1.0e-6
    @test ws.rho_vec[3] ≈ 0.1
    @test ws.rho_inv_vec ≈ 1 ./ ws.rho_vec
    @test ws.constr_type == Int8[1, -1, 0]
end

@testitem "warm starting from the solution cuts the iteration count" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(12, 30; seed = 15)
    cold = PureOSQP.solve(P, q, A, l, u; eps_abs = 1.0e-8, eps_rel = 1.0e-8, max_iter = 100_000)
    ws = setup(P, q, A, l, u; eps_abs = 1.0e-8, eps_rel = 1.0e-8, max_iter = 100_000)
    warm_start!(ws; x = cold.x, y = cold.y)
    warm = PureOSQP.solve!(ws)
    @test warm.status == SOLVED
    @test warm.iter < cold.iter
    @test warm.x ≈ cold.x rtol = 1.0e-5
end
