@testitem "the block backend solves what the dense path solves" begin
    using LinearAlgebra, Random
    Random.seed!(51)
    K, nb, mb = 5, 12, 8
    P = PureOSQP.BlockDiagonal(
        [
            Matrix(
                Symmetric(
                    let S = randn(nb, nb)
                        S'S ./ nb + 3I
                    end
                )
            ) for _ in 1:K
        ]
    )
    A = PureOSQP.BlockDiagonal([randn(mb, nb) ./ sqrt(nb) for _ in 1:K])
    n, m = size(P, 1), size(A, 1)
    q = randn(n)
    b = A * randn(n)
    l, u = b .- rand(m), b .+ rand(m)
    opts = (eps_abs = 1.0e-9, eps_rel = 1.0e-9)

    @test PureOSQP.backend_name(setup(P, q, A, l, u; opts...).linsys) === :block
    block = solve(P, q, A, l, u; opts...)
    dense = solve(Matrix(P), q, Matrix(A), l, u; opts...)
    @test block.iter == dense.iter
    @test block.x ≈ dense.x rtol = 1.0e-6
end
