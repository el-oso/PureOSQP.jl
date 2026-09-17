@testitem "the interior-point method solves with nothing else loaded" begin
    using LinearAlgebra
    P = [4.0 1.0; 1.0 2.0]
    q = [1.0, 1.0]
    A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
    l = [1.0, 0.0, 0.0]
    u = [1.0, 0.7, 0.7]

    sol = solve(P, q, A, l, u, InteriorPoint())
    @test sol.status == SOLVED
    @test sol.x ≈ [0.3, 0.7] atol = 1.0e-6
    @test sol.obj_val ≈ 1.88 atol = 1.0e-6
    @test 0 < sol.iter <= 20

    ws = setup(P, q, A, l, u, InteriorPoint(); max_iter = 50)
    @test ws isa InteriorPointWorkspace
    @test dimensions(ws) == (2, 3)
    @test solve!(ws).status == SOLVED
    update!(ws; q = [1.0, 2.0])
    @test solve!(ws).status == SOLVED
end

@testitem "the algorithm extends PureQPBase's solve rather than defining its own" begin
    using PureQPBase: PureQPBase
    m = which(
        solve,
        Tuple{
            Matrix{Float64}, Vector{Float64}, Matrix{Float64}, Vector{Float64},
            Vector{Float64}, InteriorPoint,
        },
    )
    @test m.module === PureQPBase
end

@testitem "no default algorithm is claimed here" begin
    # `OperatorSplitting` is PureOSQP's, and the five-argument forms that run it come with it:
    # this package adds an algorithm, it does not make itself the default.
    P = [4.0 1.0; 1.0 2.0]
    q = [1.0, 1.0]
    A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
    l = [1.0, 0.0, 0.0]
    u = [1.0, 0.7, 0.7]
    @test_throws MethodError solve(P, q, A, l, u)
    @test_throws MethodError setup(P, q, A, l, u)
end
