@testitem "a KroneckerOperator agrees with the matrix it stands for" begin
    using LinearAlgebra, Random
    Random.seed!(61)
    A1, A2 = randn(5, 4), randn(3, 6)
    K = PureOSQP.KroneckerOperator(A1, A2)
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

@testitem "the Kronecker backend solves what the dense path solves" begin
    using LinearAlgebra, Random
    Random.seed!(62)
    n1, n2 = 12, 10
    A1, A2 = randn(n1, n1), randn(n2, n2)
    K = PureOSQP.KroneckerOperator(A1, A2)
    n = n1 * n2
    P = Diagonal(fill(2.0, n))
    q = randn(n)
    b = kron(A1, A2) * randn(n)
    l, u = b .- rand(n), b .+ rand(n)
    opts = (scaling = 0, eps_abs = 1.0e-9, eps_rel = 1.0e-9)

    ws = setup(P, q, K, l, u; opts...)
    @test PureOSQP.backend_name(ws.linsys) === :kronecker

    kronecker = solve(P, q, K, l, u; opts...)
    dense = solve(Matrix(P), q, kron(A1, A2), l, u; opts...)
    @test kronecker.iter == dense.iter
    @test kronecker.x ≈ dense.x rtol = 1.0e-6

    # Two eigenbases and a diagonal, against a dense inverse's triangle.
    info = PureOSQP.backend_info(ws.linsys)
    @test info.factor_nnz == n1^2 + n2^2 + n1 * n2
    @test info.factor_nnz < n * (n + 1) ÷ 2
end

@testitem "the Kronecker rung declines what it cannot diagonalize" begin
    using LinearAlgebra, Random
    Random.seed!(63)
    n1, n2 = 8, 6
    n = n1 * n2
    A1, A2 = randn(n1, n1), randn(n2, n2)
    K = PureOSQP.KroneckerOperator(A1, A2)
    q = randn(n)
    b = kron(A1, A2) * randn(n)
    l, u = b .- rand(n), b .+ rand(n)
    scalar = Diagonal(fill(2.0, n))

    # Each of these breaks the diagonalization, and the log carries the measurement for it.
    # A rung that accepted any of them would return a wrong answer, not a slow one.
    @test PureOSQP.backend_name(setup(scalar, q, K, l, u; scaling = 0).linsys) === :kronecker
    # Equilibration puts `c·μ·D²` in the reduced matrix: diagonal, but not scalar.
    @test PureOSQP.backend_name(setup(scalar, q, K, l, u).linsys) === :cholesky
    # A `P` that is not a multiple of the identity, including a Kronecker one.
    @test PureOSQP.backend_name(
        setup(Diagonal(rand(n) .+ 1), q, K, l, u; scaling = 0).linsys
    ) === :cholesky
    # One equality row gives `ρ` a second value.
    @test PureOSQP.backend_name(
        setup(scalar, q, K, vcat(b[1], l[2:end]), vcat(b[1], u[2:end]); scaling = 0).linsys
    ) === :cholesky

    # Predicate and value are separate so neither returns a union; the rung checks the first
    # before reading the second.
    @test PureOSQP.is_scalar_multiple(Diagonal(fill(3.0, 4)))
    @test PureOSQP.scalar_multiple(Diagonal(fill(3.0, 4))) == 3.0
    @test !PureOSQP.is_scalar_multiple(Diagonal([1.0, 2.0]))
    @test !PureOSQP.is_scalar_multiple(randn(3, 3))
end
