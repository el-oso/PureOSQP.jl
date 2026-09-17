# What this package is on its own: the problem and the contracts, with no algorithm. Solving
# is tested where the algorithms are, against a referee this package has no reason to depend on.

@testitem "no algorithm is defined here" begin
    using PureQPBase
    using InteractiveUtils: subtypes
    # Both are declared, so an algorithm package has a supertype to extend and `solve` has a
    # function to add a method to; neither has an instance until one is loaded.
    @test isabstracttype(PureQPBase.QPAlgorithm)
    @test isabstracttype(PureQPBase.QPWorkspace)
    @test isempty(subtypes(PureQPBase.QPAlgorithm))
    @test isempty(subtypes(PureQPBase.QPWorkspace))

    P = [4.0 1.0; 1.0 2.0]
    q = [1.0, 1.0]
    A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
    l = [1.0, 0.0, 0.0]
    u = [1.0, 0.7, 0.7]
    # The five-argument forms belong to the package that defines the default algorithm.
    @test_throws MethodError solve(P, q, A, l, u)
    @test_throws MethodError setup(P, q, A, l, u)
end

@testitem "the problem is built and validated" begin
    using PureQPBase
    P = [4.0 1.0; 1.0 2.0]
    q = [1.0, 1.0]
    A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
    l = [1.0, 0.0, 0.0]
    u = [1.0, 0.7, 0.7]
    prob = PureQPBase.validated_problem(Float64, 2, 3, P, q, A, l, u, 10)
    @test prob.n == 2
    @test prob.m == 3
    @test length(prob.D) == 2
    @test length(prob.E) == 3

    @test_throws "l must be elementwise" PureQPBase.validate(P, q, A, [1.0, 0.0, 1.0], u)
    @test_throws "q" PureQPBase.validate(P, [1.0, NaN], A, l, u)
end

@testitem "a backend factors and solves its system" begin
    using PureQPBase, LinearAlgebra
    include(joinpath(@__DIR__, "helpers.jl"))
    # `ReducedCholesky` is this package's own, and needs no algorithm to drive it: the weights
    # stand in for what one would set.
    P = [4.0 1.0; 1.0 2.0]
    q = [1.0, 1.0]
    A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
    l = [1.0, 0.0, 0.0]
    u = [1.0, 0.7, 0.7]
    prob, wt, ls = backend_for(P, q, A, l, u; scaling = 0)
    @test ls isa PureQPBase.LinearSystem
    @test PureQPBase.backend_name(ls) isa Symbol

    bx, bz = [1.0, 2.0], [0.5, 0.5, 0.5]
    x, z = zeros(2), zeros(3)
    PureQPBase.solve_system!(ls, prob, wt, bx, bz, x, z)
    ref = kkt_matrix(P, A, wt) \ [bx; bz]
    @test x ≈ ref[1:2] rtol = 1.0e-9
    @test z ≈ A * x rtol = 1.0e-9
end

@testitem "the option names are checked against a backend name" begin
    using PureQPBase
    @test :auto in PureQPBase.LINSYS_OPTIONS
    @test :kkt in PureQPBase.LINSYS_OPTIONS
    @test !(:nonsense in PureQPBase.LINSYS_OPTIONS)
    @test_throws "linsys must be one of" PureQPBase.check_linsys(:nonsense)
    @test PureQPBase.check_linsys(:kkt) === Val(:kkt)
end
