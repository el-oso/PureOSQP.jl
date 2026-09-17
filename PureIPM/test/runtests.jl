# That this package solves on its own, with no other algorithm loaded. What the method does
# — its iteration counts, its backends, its certificates, its derivatives — is tested against
# a referee in the suite that has one.
using PureIPM
using PureQPBase: PureQPBase
using LinearAlgebra
using Test

@testset "PureIPM" begin
    P = [4.0 1.0; 1.0 2.0]
    q = [1.0, 1.0]
    A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
    l = [1.0, 0.0, 0.0]
    u = [1.0, 0.7, 0.7]

    @testset "it solves alone" begin
        sol = solve(P, q, A, l, u, InteriorPoint())
        @test sol.status == SOLVED
        @test sol.x ≈ [0.3, 0.7] atol = 1.0e-6
        @test sol.obj_val ≈ 1.88 atol = 1.0e-6
        @test 0 < sol.iter <= 20
        # The core's `solve` is the one being extended, not a second function of the same name.
        @test which(solve, Tuple{Matrix{Float64}, Vector{Float64}, Matrix{Float64}, Vector{Float64}, Vector{Float64}, InteriorPoint}).module === PureQPBase
    end

    @testset "the workspace is reusable" begin
        ws = setup(P, q, A, l, u, InteriorPoint(); max_iter = 50)
        @test ws isa InteriorPointWorkspace
        @test dimensions(ws) == (2, 3)
        @test solve!(ws).status == SOLVED
        update!(ws; q = [1.0, 2.0])
        @test solve!(ws).status == SOLVED
    end

    @testset "no default algorithm is claimed here" begin
        # `OperatorSplitting` is PureOSQP's, and the five-argument forms that run it come with
        # it: this package adds an algorithm, it does not make itself the default.
        @test_throws MethodError solve(P, q, A, l, u)
    end

    @testset "what it cannot do it refuses by name" begin
        @test_throws "kronecker" setup(P, q, A, l, u, InteriorPoint(); linsys = :kronecker)
        @test_throws "lowrank" setup(P, q, A, l, u, InteriorPoint(); linsys = :lowrank)
        @test_throws "caller-supplied" setup(P, q, A, l, u, InteriorPoint(); linsys = :indirect)
        @test_throws "rho is not an option" solve(P, q, A, l, u, InteriorPoint(); rho = 0.2)
    end
end
