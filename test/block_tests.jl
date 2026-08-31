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

    ws = setup(P, q, A, l, u; opts...)
    @test PureOSQP.backend_name(ws.linsys) === :block

    block = solve(P, q, A, l, u; opts...)
    dense = solve(Matrix(P), q, Matrix(A), l, u; opts...)
    @test block.iter == dense.iter
    @test block.x ≈ dense.x rtol = 1.0e-6

    # Equilibration reads the blocks and nothing else, so it produces the same factors the
    # dense form does rather than merely close ones.
    @test ws.D == setup(Matrix(P), q, Matrix(A), l, u; opts...).D
    @test ws.E == setup(Matrix(P), q, Matrix(A), l, u; opts...).E

    # `Σ nᵢ²` stored against `n²`, which is what the tier is for.
    info = PureOSQP.backend_info(ws.linsys)
    @test info.factor_nnz == K * nb * (nb + 1) ÷ 2
    @test info.factor_nnz < n * (n + 1) ÷ 2
end

@testitem "the block rung declines what it cannot decouple" begin
    using LinearAlgebra, Random
    Random.seed!(52)
    proto = zeros(20)
    square(k) = Matrix(
        Symmetric(
            let S = randn(k, k)
                S'S ./ k + 3I
            end
        )
    )

    "The rung's verdict for a `P` and `A` built from these block sizes."
    function rung(psizes, asizes)
        P = PureOSQP.BlockDiagonal([square(k) for k in psizes])
        A = PureOSQP.BlockDiagonal([randn(k, k) for k in asizes])
        n, m = size(P, 2), size(A, 1)
        return PureOSQP.block_rung(
            P, A, proto, n, m, ones(n), ones(m), 1.0, ones(m), 1.0e-6
        )
    end

    # One block is the dense terminal wearing a wrapper.
    @test isnothing(rung([6], [6]))
    # Different column partitions do not decouple onto the same blocks.
    @test isnothing(rung([4, 8], [6, 6]))
    ls, factored = rung([4, 8], [4, 8])
    @test ls isa PureOSQP.BlockReduced
    @test !factored
end

@testitem "a BlockDiagonal agrees with the matrix it stands for" begin
    using LinearAlgebra, Random
    Random.seed!(53)
    blocks = [randn(3, 4), randn(2, 5), randn(4, 2)]
    A = PureOSQP.BlockDiagonal(blocks)
    dense = Matrix(A)

    @test size(A) == (9, 11)
    @test dense == cat(blocks...; dims = (1, 2))

    x = randn(size(A, 2))
    y = randn(size(A, 1))
    @test A * x ≈ dense * x
    @test A' * y ≈ dense' * y

    # A column's nonzeros are its own block's rows, which is what keeps equilibration to the
    # entries the blocks hold.
    for j in axes(A, 2)
        rows = PureOSQP.structural_rows(A, j)
        @test all(iszero, dense[setdiff(axes(A, 1), rows), j])
    end
end
