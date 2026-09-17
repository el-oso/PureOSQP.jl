@testitem "a KroneckerOperator agrees with the matrix it stands for" begin
    using PureQPBase, LinearAlgebra, Random
    Random.seed!(61)
    A1, A2 = randn(5, 4), randn(3, 6)
    K = PureQPBase.KroneckerOperator(A1, A2)
    dense = kron(A1, A2)

    @test size(K) == size(dense)
    @test Matrix(K) ≈ dense
    x = randn(size(K, 2))
    y = randn(size(K, 1))
    @test K * x ≈ dense * x
    @test K' * y ≈ dense' * y

    # The products run off the factors, so they must not depend on the scratch's contents.
    fill!(K.scratch1, NaN)
    fill!(K.scratch2, NaN)
    @test K * x ≈ dense * x
    @test K' * y ≈ dense' * y
end

@testitem "the Kronecker backend solves the system the dense one does" begin
    using PureQPBase, LinearAlgebra, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(62)
    n1, n2 = 12, 10
    A1, A2 = randn(n1, n1), randn(n2, n2)
    K = PureQPBase.KroneckerOperator(A1, A2)
    n = n1 * n2
    P = Diagonal(fill(2.0, n))
    q = randn(n)
    b = kron(A1, A2) * randn(n)
    l, u = b .- rand(n), b .+ rand(n)

    prob, wt, ls = backend_for(P, q, K, l, u; scaling = 0)
    @test PureQPBase.backend_name(ls) === :kronecker

    bx, bz = randn(n), randn(n)
    x, z = zeros(n), zeros(n)
    PureQPBase.solve_system!(ls, prob, wt, bx, bz, x, z)
    ref = kkt_matrix(P, kron(A1, A2), wt) \ [bx; bz]
    @test x ≈ ref[1:n] rtol = 1.0e-7
    @test z ≈ K * x rtol = 1.0e-7

    # Two eigenbases and a diagonal, against a dense inverse's triangle.
    info = PureQPBase.backend_info(ls)
    @test info.factor_nnz == n1^2 + n2^2 + n1 * n2
    @test info.factor_nnz < n * (n + 1) ÷ 2
end

@testitem "the Kronecker rung declines what it cannot diagonalize" begin
    using PureQPBase, LinearAlgebra, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(63)
    n1, n2 = 8, 6
    n = n1 * n2
    A1, A2 = randn(n1, n1), randn(n2, n2)
    K = PureQPBase.KroneckerOperator(A1, A2)
    q = randn(n)
    b = kron(A1, A2) * randn(n)
    l, u = b .- rand(n), b .+ rand(n)
    scalar = Diagonal(fill(2.0, n))
    name(P; kwargs...) = PureQPBase.backend_name(last(backend_for(P, q, K, l, u; kwargs...)))

    # Each of these breaks the diagonalization. A rung that accepted any of them would return
    # a wrong answer, not a slow one.
    @test name(scalar; scaling = 0) === :kronecker
    # Equilibration puts `c·μ·D²` in the reduced matrix: diagonal, but not scalar.
    @test name(scalar) === :cholesky
    # A `P` that is not a multiple of the identity, including a Kronecker one.
    @test name(Diagonal(rand(n) .+ 1); scaling = 0) === :cholesky
    # A second weight among the rows, which an equality row is one way to produce.
    @test name(scalar; scaling = 0, rho = vcat(1.0e3, fill(0.1, n - 1))) === :cholesky

    # Predicate and value are separate so neither returns a union; the rung checks the first
    # before reading the second.
    @test PureQPBase.is_scalar_multiple(Diagonal(fill(3.0, 4)))
    @test PureQPBase.scalar_multiple(Diagonal(fill(3.0, 4))) == 3.0
    @test !PureQPBase.is_scalar_multiple(Diagonal([1.0, 2.0]))
    @test !PureQPBase.is_scalar_multiple(randn(3, 3))
end

@testitem "a non-finite Kronecker factor is refused by naming the factor" begin
    using PureQPBase, LinearAlgebra
    include(joinpath(@__DIR__, "helpers.jl"))
    # The check reads the two factors, not the product, so the entry it reports is the
    # factor's own.
    A1 = [1.0 2.0; 3.0 NaN]
    A2 = [1.0 0.0; 0.0 1.0]
    @test_throws "A's first Kronecker factor is not finite at entry (2, 2)" backend_for(
        Diagonal(ones(4)), zeros(4), PureQPBase.KroneckerOperator(A1, A2), fill(-1.0, 4), fill(1.0, 4)
    )
    @test_throws "A's second Kronecker factor" backend_for(
        Diagonal(ones(4)), zeros(4), PureQPBase.KroneckerOperator(A2, A1), fill(-1.0, 4), fill(1.0, 4)
    )
end
