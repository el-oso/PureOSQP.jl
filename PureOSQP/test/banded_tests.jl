@testitem "the banded backend agrees with the dense one end to end" begin
    using LinearAlgebra, BandedMatrices, Random
    Random.seed!(32)
    n = 40
    P = SymTridiagonal(rand(n) .+ 4, rand(n - 1) ./ 8)
    A = Tridiagonal(rand(n - 1) ./ 4, rand(n) .+ 1, rand(n - 1) ./ 4)
    q, l, u = randn(n), -rand(n), rand(n)
    opts = (eps_abs = 1.0e-9, eps_rel = 1.0e-9)
    banded = solve(P, q, A, l, u; opts...)
    dense = solve(Matrix(P), q, Matrix(A), l, u; opts...)
    @test banded.status == PureOSQP.SOLVED
    @test banded.x ≈ dense.x rtol = 1.0e-6
    @test banded.iter == dense.iter
end
