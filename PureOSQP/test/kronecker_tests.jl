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

    @test PureOSQP.backend_name(setup(P, q, K, l, u; opts...).linsys) === :kronecker
    kronecker = solve(P, q, K, l, u; opts...)
    dense = solve(Matrix(P), q, kron(A1, A2), l, u; opts...)
    @test kronecker.iter == dense.iter
    @test kronecker.x ≈ dense.x rtol = 1.0e-6
end

@testitem "an equality row moves the Kronecker pair off its rung" begin
    using LinearAlgebra, Random
    Random.seed!(63)
    n1, n2 = 8, 6
    n = n1 * n2
    K = PureOSQP.KroneckerOperator(randn(n1, n1), randn(n2, n2))
    q = randn(n)
    b = Matrix(K) * randn(n)
    l, u = b .- rand(n), b .+ rand(n)
    scalar = Diagonal(fill(2.0, n))

    # The rung needs one ρ for every row, and the split gives an equality row its own.
    @test PureOSQP.backend_name(setup(scalar, q, K, l, u; scaling = 0).linsys) === :kronecker
    @test PureOSQP.backend_name(
        setup(scalar, q, K, vcat(b[1], l[2:end]), vcat(b[1], u[2:end]); scaling = 0).linsys
    ) === :cholesky
end
