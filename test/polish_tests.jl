@testitem "polishing sharpens a loose solution" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(12, 30; seed = 16)
    loose = PureOSQP.solve(P, q, A, l, u; eps_abs = 1.0e-3, eps_rel = 1.0e-3, polish = false)
    sharp = PureOSQP.solve(P, q, A, l, u; eps_abs = 1.0e-3, eps_rel = 1.0e-3, polish = true)
    @test sharp.polished
    r_loose = maximum(kkt_residuals(P, q, A, l, u, loose.x, loose.y))
    r_sharp = maximum(kkt_residuals(P, q, A, l, u, sharp.x, sharp.y))
    @test r_sharp < r_loose / 100
end

@testitem "polishing never makes the answer worse" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    checked = Ref(0)
    for trial in 1:15
        Random.seed!(200 + trial)
        n, m = rand(3:12), rand(3:25)
        X = randn(n, n)
        P = trial % 3 == 0 ? Matrix(X'X) : Matrix(X'X + I)
        q = randn(n)
        A = randn(m, n)
        Ax = A * randn(n)
        l, u = Ax .- rand(m), Ax .+ rand(m)
        a = PureOSQP.solve(P, q, A, l, u; eps_abs = 1.0e-4, eps_rel = 1.0e-4, polish = false)
        b = PureOSQP.solve(P, q, A, l, u; eps_abs = 1.0e-4, eps_rel = 1.0e-4, polish = true)
        a.status == SOLVED || continue
        checked[] += 1
        @test b.status == SOLVED
        # Judged by the independent referee, not by the solver's own reported residuals,
        # which are exactly what polish!'s acceptance rule already compares.
        ra = maximum(kkt_residuals(P, q, A, l, u, a.x, a.y))
        rb = maximum(kkt_residuals(P, q, A, l, u, b.x, b.y))
        @test rb <= max(ra * (1 + 1.0e-6), 1.0e-9)
    end
    @test checked[] >= 10
end

@testitem "no active set means polishing is skipped, not failed" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(18)
    n = 8
    P = (X = randn(n, n); Matrix(X'X + I))
    q = randn(n)
    A = Matrix(1.0I, n, n)
    s = PureOSQP.solve(
        P, q, A, fill(-1.0e3, n), fill(1.0e3, n);
        polish = true, eps_abs = 1.0e-8, eps_rel = 1.0e-8, max_iter = 100_000
    )
    @test s.status == SOLVED
    @test !s.polished
    @test s.x ≈ -(P \ q) rtol = 1.0e-5
end
