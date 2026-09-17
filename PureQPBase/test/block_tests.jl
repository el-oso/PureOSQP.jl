@testitem "the block backend solves the system the dense one does" begin
    using PureQPBase, LinearAlgebra, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(51)
    K, nb, mb = 5, 12, 8
    square(k) = Matrix(
        Symmetric(
            let S = randn(k, k)
                S'S ./ k + 3I
            end
        )
    )
    P = PureQPBase.BlockDiagonal([square(nb) for _ in 1:K])
    A = PureQPBase.BlockDiagonal([randn(mb, nb) ./ sqrt(nb) for _ in 1:K])
    n, m = size(P, 1), size(A, 1)
    q = randn(n)
    b = A * randn(n)
    l, u = b .- rand(m), b .+ rand(m)

    prob, wt, ls = backend_for(P, q, A, l, u; scaling = 0)
    @test PureQPBase.backend_name(ls) === :block

    bx, bz = randn(n), randn(m)
    x, z = zeros(n), zeros(m)
    PureQPBase.solve_system!(ls, prob, wt, bx, bz, x, z)
    ref = kkt_matrix(P, A, wt) \ [bx; bz]
    @test x ≈ ref[1:n] rtol = 1.0e-9
    @test z ≈ A * x rtol = 1.0e-9

    # Equilibration reads the blocks and nothing else, so it produces the same factors the
    # dense form does rather than merely close ones.
    blocked = PureQPBase.validated_problem(Float64, n, m, P, q, A, l, u, 10)
    dense = PureQPBase.validated_problem(Float64, n, m, Matrix(P), q, Matrix(A), l, u, 10)
    @test blocked.D == dense.D
    @test blocked.E == dense.E

    # `Σ nᵢ²` stored against `n²`, which is what the tier is for.
    info = PureQPBase.backend_info(ls)
    @test info.factor_nnz == K * nb * (nb + 1) ÷ 2
    @test info.factor_nnz < n * (n + 1) ÷ 2
end

@testitem "the block rung declines what it cannot decouple" begin
    using PureQPBase, LinearAlgebra, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(52)
    square(k) = Matrix(
        Symmetric(
            let S = randn(k, k)
                S'S ./ k + 3I
            end
        )
    )

    "The rung's verdict for a `P` and `A` built from these block sizes."
    function rung(psizes, asizes)
        P = PureQPBase.BlockDiagonal([square(k) for k in psizes])
        A = PureQPBase.BlockDiagonal([randn(k, k) for k in asizes])
        n, m = size(P, 2), size(A, 1)
        prob = raw_problem(P, A, n, m)
        wt = raw_weights(ones(m), 1.0e-6)
        return PureQPBase.block_rung(P, A, prob, wt, PureQPBase.ADMMSelection())
    end

    # One block is the dense terminal wearing a wrapper.
    @test isnothing(rung([6], [6]))
    # Different column partitions do not decouple onto the same blocks.
    @test isnothing(rung([4, 8], [6, 6]))
    ls, factored = rung([4, 8], [4, 8])
    @test ls isa PureQPBase.BlockReduced
    @test !factored
end

@testitem "a BlockDiagonal agrees with the matrix it stands for" begin
    using PureQPBase, LinearAlgebra, Random
    Random.seed!(53)
    blocks = [randn(3, 4), randn(2, 5), randn(4, 2)]
    A = PureQPBase.BlockDiagonal(blocks)
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
        rows = PureQPBase.structural_rows(A, j)
        @test all(iszero, dense[setdiff(axes(A, 1), rows), j])
    end
end
