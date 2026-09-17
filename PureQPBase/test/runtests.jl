# What this package is on its own: the problem, the backends and the contracts, with no
# algorithm. Solving is tested where the algorithms are, against a referee this package has no
# reason to depend on.
using PureQPBase
using InteractiveUtils: subtypes
using LinearAlgebra
using Test

@testset "PureQPBase" begin
    @testset "no algorithm is defined here" begin
        # Both are declared, so an algorithm package has a supertype to extend and `solve` has
        # a function to add a method to; neither has an instance until one is loaded.
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

    @testset "the problem is built and validated" begin
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

    @testset "a backend factors and solves its system" begin
        # `ReducedCholesky` is this package's own, and needs no algorithm to drive it: the
        # weights stand in for what one would set.
        P = [4.0 1.0; 1.0 2.0]
        q = [1.0, 1.0]
        A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
        l = [1.0, 0.0, 0.0]
        u = [1.0, 0.7, 0.7]
        prob = PureQPBase.validated_problem(Float64, 2, 3, P, q, A, l, u, 0)
        rho = fill(0.1, 3)
        wt = PureQPBase.SystemWeights(rho, 1 ./ rho, 1.0e-6)
        ls, factored = PureQPBase.select_backend(P, A, prob, wt, PureQPBase.ADMMSelection())
        @test ls isa PureQPBase.LinearSystem
        @test PureQPBase.backend_name(ls) isa Symbol
        # A rung reports whether it left the backend factorized, so that `setup` factorizes
        # only what has not been.
        factored || @test PureQPBase.factorize!(ls, prob, wt)

        x, z = zeros(2), zeros(3)
        PureQPBase.solve_system!(ls, prob, wt, [1.0, 2.0], [0.5, 0.5, 0.5], x, z)
        @test all(isfinite, x)
        @test all(isfinite, z)
    end

    @testset "the option names are checked against a backend name" begin
        @test :auto in PureQPBase.LINSYS_OPTIONS
        @test :kkt in PureQPBase.LINSYS_OPTIONS
        @test !(:nonsense in PureQPBase.LINSYS_OPTIONS)
        @test_throws "linsys must be one of" PureQPBase.check_linsys(:nonsense)
        @test PureQPBase.check_linsys(:kkt) === Val(:kkt)
    end
end
