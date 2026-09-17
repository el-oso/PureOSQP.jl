@testitem "reduced_diagonal! matches the full walk for structured A" begin
    using PureQPBase, LinearAlgebra, Random, Krylov
    Random.seed!(91)
    n, k = 24, 3
    # `A` is only read at entries it can hold, so a structured spelling and its dense form
    # must give the same reciprocals to the last bit. It is the only thing the matrix-free
    # backend reads from `A` besides its products, so an inexact match would precondition
    # the two spellings of one problem differently.
    structured = (
        Diagonal(randn(n) ./ 2),
        Bidiagonal(randn(n) ./ 2, randn(n - 1) ./ 4, :U),
        PureQPBase.RowCoupled(randn(k, n) ./ 4, ones(n - k), collect(1:(n - k))),
    )
    P = Diagonal(rand(n) .+ 0.5)
    D, sigma, c = rand(n) .+ 0.5, 1.0e-6, 1.3
    for A in structured
        m = size(A, 1)
        rho, E = rand(m) .+ 0.1, rand(m) .+ 0.5
        args = (Float64, P, A, rho, E, D, sigma, c)
        dense = (Float64, P, Matrix(A), rho, E, D, sigma, c)
        @test PureQPBase.reduced_diagonal!(zeros(n), args...) ==
            PureQPBase.reduced_diagonal!(zeros(n), dense...)
    end
end
