@testitem "lazy scaled products match explicit scaling" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(1)
    n, m = 7, 11
    P = (X = randn(n, n); Matrix(Symmetric(X'X)))
    A = randn(m, n) * Diagonal(exp10.(range(-3, 3; length = n)))
    q = randn(n)
    l = -rand(m)
    u = rand(m)
    ws = setup(P, q, A, l, u; scaling = 10)
    Pt = ws.c .* (Diagonal(ws.D) * P * Diagonal(ws.D))
    At = Diagonal(ws.E) * A * Diagonal(ws.D)
    x = randn(n)
    y = randn(m)
    @test PureOSQP.mul_A!(similar(y), ws, x) ≈ At * x
    @test PureOSQP.mul_At!(similar(x), ws, y) ≈ At' * y
    @test PureOSQP.mul_P!(similar(x), ws, x) ≈ Pt * x
    @test ws.q ≈ ws.c .* (ws.D .* q)
    @test ws.l ≈ ws.E .* l
    @test ws.u ≈ ws.E .* u
end

@testitem "equilibration reduces the column-norm spread" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(2)
    n, m = 20, 60
    A = randn(m, n) * Diagonal(exp10.(range(-4, 4; length = n)))
    ws = setup(zeros(n, n), zeros(n), A, -ones(m), ones(m); scaling = 10)
    At = Diagonal(ws.E) * A * Diagonal(ws.D)
    spread(M) = (v = [maximum(abs, view(M, :, j)) for j in axes(M, 2)]; maximum(v) / minimum(v))
    @test spread(A) > 1.0e6
    @test spread(At) < 10
end

@testitem "P and A are never mutated" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(3)
    A = randn(5, 3)
    P = (X = randn(3, 3); Matrix(X'X))
    A0, P0 = copy(A), copy(P)
    PureOSQP.solve(P, zeros(3), A, -ones(5), ones(5); scaling = 10, max_iter = 50)
    @test A == A0
    @test P == P0
end

@testitem "scaling = 0 leaves the factors at identity" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(4)
    A = randn(5, 3)
    P = (X = randn(3, 3); Matrix(X'X))
    q = randn(3)
    ws = setup(P, q, A, -ones(5), ones(5); scaling = 0)
    @test all(isone, ws.D)
    @test all(isone, ws.E)
    @test isone(ws.c)
    @test ws.q == q
end
