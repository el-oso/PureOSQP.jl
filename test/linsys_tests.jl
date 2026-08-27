@testitem "the reduced solve reproduces the full KKT system" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(5)
    n, m = 9, 14
    P = (X = randn(n, n); Matrix(X'X))
    A = randn(m, n)
    ws = setup(P, randn(n), A, -rand(m), rand(m); scaling = 0, sigma = 1.0e-6, rho = 0.1)
    @test ws.backend === :cholesky
    K = [P + ws.settings.sigma * I  A'; A  -Diagonal(1 ./ ws.rho_vec)]
    bx, bz = randn(n), randn(m)
    PureOSQP.solve_kkt!(ws, bx, bz)
    ref = K \ [bx; bz]
    @test ws.xtilde ≈ ref[1:n] rtol = 1.0e-9
    @test ws.ztilde ≈ A * ws.xtilde rtol = 1.0e-9
end

@testitem "an ill-conditioned A falls back to Bunch-Kaufman" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(6)
    n, m = 40, 120
    U = Matrix(qr(randn(m, n)).Q)[:, 1:n]
    V = Matrix(qr(randn(n, n)).Q)
    A = U * Diagonal(exp10.(range(0, 11; length = n))) * V'
    P = zeros(n, n)
    ws = setup(P, randn(n), A, -ones(m), ones(m); scaling = 0)
    @test ws.backend === :bunchkaufman
    bx, bz = randn(n), randn(m)
    PureOSQP.solve_kkt!(ws, bx, bz)
    # At this conditioning a Float64 `K \ b` is no more trustworthy than the solver, so the
    # reference is computed in extended precision.
    K = [P + ws.settings.sigma * I  A'; A  -Diagonal(1 ./ ws.rho_vec)]
    ref = Float64.(big.(K) \ big.([bx; bz]))[1:n]
    @test norm(ws.xtilde .- ref, Inf) < 1.0e-4 * norm(ref, Inf)
end

@testitem "both backends reach the same solution" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(15, 40; seed = 7)
    opts = (eps_abs = 1.0e-9, eps_rel = 1.0e-9, max_iter = 100_000)
    chol = PureOSQP.solve(P, q, A, l, u; linsys = :auto, opts...)
    kkt = PureOSQP.solve(P, q, A, l, u; linsys = :kkt, opts...)
    @test chol.status == SOLVED
    @test kkt.status == SOLVED
    @test chol.x ≈ kkt.x rtol = 1.0e-5
    @test abs(chol.obj_val - kkt.obj_val) <= 1.0e-6 * max(1, abs(kkt.obj_val))
end

@testitem "linsys rejects an unknown backend" begin
    @test_throws "linsys must be :auto or :kkt" setup([1.0;;], [0.0], [1.0;;], [0.0], [1.0]; linsys = :magic)
end
