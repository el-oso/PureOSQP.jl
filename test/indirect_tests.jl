@testitem "the matrix-free backend reaches the same solution as the direct one" begin
    using LinearAlgebra, SparseArrays, OSQP, Random, Krylov
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(30, 60; seed = 3)
    opts = (eps_abs = 1.0e-8, eps_rel = 1.0e-8, max_iter = 100_000)

    direct = PureOSQP.solve(P, q, A, l, u; opts...)
    indirect = PureOSQP.solve(P, q, A, l, u; opts..., linsys = :indirect)

    @test direct.status == SOLVED
    @test indirect.status == SOLVED
    # The inner solve is inexact, so the iterates are not identical -- demanding equality
    # here would be demanding the wrong thing. Both must solve the original problem.
    @test indirect.x ≈ direct.x atol = 1.0e-5
    @test indirect.obj_val ≈ direct.obj_val atol = 1.0e-5
    @test maximum(kkt_residuals(P, q, A, l, u, indirect.x, indirect.y)) < 1.0e-5

    @test PureOSQP.backend_name(setup(P, q, A, l, u; linsys = :indirect).linsys) == :indirect
end

@testitem "the matrix-free solve allocates nothing per iteration" begin
    using LinearAlgebra, SparseArrays, OSQP, Random, Krylov
    include(joinpath(@__DIR__, "helpers.jl"))
    # The reason for a preallocated Krylov workspace, and the reason `ReducedOperator`
    # carries its element type: Krylov compares `eltype(A)` against the vectors' and drops
    # to an allocating path when they disagree, which is silent apart from a warning.
    P, q, A, l, u = random_qp(20, 40; seed = 8)
    ws = setup(P, q, A, l, u; eps_abs = 1.0e-8, eps_rel = 1.0e-8, linsys = :indirect)
    PureOSQP.solve!(ws)
    PureOSQP.solve_system!(ws.linsys, ws, ws.rhs_x, ws.rhs_z)      # warm up
    allocs = [(@allocated PureOSQP.solve_system!(ws.linsys, ws, ws.rhs_x, ws.rhs_z)) for _ in 1:4]
    @test all(iszero, allocs)
end

@testitem "asking for the matrix-free backend without Krylov says so" begin
    # The core cannot build it, and the error has to name the remedy rather than surface a
    # MethodError from somewhere inside `setup`.
    P = [4.0 1.0; 1.0 2.0]
    q = [1.0, 1.0]
    A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
    l = [1.0, 0.0, 0.0]
    u = [1.0, 0.7, 0.7]
    if isnothing(Base.get_extension(PureOSQP, :PureOSQPKrylovExt))
        @test_throws "needs Krylov.jl" setup(P, q, A, l, u; linsys = :indirect)
    else
        # Krylov is loaded by the other items in this file, so the backend exists here.
        @test PureOSQP.backend_name(setup(P, q, A, l, u; linsys = :indirect).linsys) == :indirect
    end
    @test_throws "linsys must be" setup(P, q, A, l, u; linsys = :nonsense)
    @test_throws "cg_max_iter must be positive" setup(P, q, A, l, u; cg_max_iter = 0)
    @test_throws "cg_tol_fraction must lie in (0, 1]" setup(P, q, A, l, u; cg_tol_fraction = 2.0)
end
