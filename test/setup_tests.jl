@testitem "setup rejects invalid input" begin
    P = [1.0 0.0; 0.0 1.0]
    q = [0.0, 0.0]
    A = [1.0 0.0]
    l = [0.0]
    u = [1.0]
    @test_throws "l must be elementwise ≤ u" setup(P, q, A, [2.0], [1.0])
    @test_throws "P must be square" setup([1.0 0.0], q, A, l, u)
    @test_throws "must equal size(P, 1)" setup(P, q, [1.0 0.0 0.0], l, u)
    @test_throws "q must be finite" setup(P, [NaN, 0.0], A, l, u)
    @test_throws "q must be finite" setup(P, [Inf, 0.0], A, l, u)
    @test_throws "P must be symmetric" setup([1.0 2.0; 0.0 1.0], q, A, l, u)
    @test_throws "l may not be +Inf" setup(P, q, A, [Inf], [Inf])
    @test_throws "u contains NaN" setup(P, q, A, l, [NaN])
end

@testitem "workspace fields are concrete" begin
    ws = setup([4.0 1.0; 1.0 2.0], [1.0, 1.0], [1.0 1.0], [0.0], [1.0])
    W = typeof(ws)
    for f in fieldnames(W)
        @test isconcretetype(fieldtype(W, f))
    end
end

@testitem "solve returns a concretely typed Solution" begin
    for T in (Float32, Float64)
        s = PureOSQP.solve(Matrix{T}([4 1; 1 2]), T[1, 1], Matrix{T}([1 1]), T[0], T[1])
        @test s isa Solution{T}
    end
    @test Base.return_types(PureOSQP.solve!, (typeof(setup([1.0;;], [0.0], [1.0;;], [0.0], [1.0])),)) ==
        [Solution{Float64}]
end

@testitem "referee scores the known solution of the reference QP" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P = [4.0 1.0; 1.0 2.0]
    q = [1.0, 1.0]
    A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
    l = [1.0, 0.0, 0.0]
    u = [1.0, 0.7, 0.7]
    rp, rd, rc = kkt_residuals(P, q, A, l, u, [0.3, 0.7], [-2.9, 0.0, 0.2])
    @test max(rp, rd, rc) < 1.0e-8
end

@testitem "settings are validated, not silently accepted" begin
    P = [4.0 1.0; 1.0 2.0]
    q = [1.0, 1.0]
    A = [1.0 1.0]
    l = [0.0]
    u = [1.0]
    @test_throws "sigma must be positive" setup(P, q, A, l, u; sigma = 0.0)
    @test_throws "sigma must be positive" setup(P, q, A, l, u; sigma = -1.0)
    @test_throws "rho must be positive" setup(P, q, A, l, u; rho = -1.0)
    @test_throws "alpha must lie in (0, 2)" setup(P, q, A, l, u; alpha = 0.0)
    @test_throws "alpha must lie in (0, 2)" setup(P, q, A, l, u; alpha = 5.0)
    @test_throws "max_iter must be positive" setup(P, q, A, l, u; max_iter = 0)
    @test_throws "max_iter must be positive" setup(P, q, A, l, u; max_iter = -5)
    @test_throws "eps_abs and eps_rel must be non-negative" setup(P, q, A, l, u; eps_abs = -1.0)
    @test_throws "scaling must be non-negative" setup(P, q, A, l, u; scaling = -3)
    @test_throws "check_termination must be non-negative" setup(P, q, A, l, u; check_termination = -1)
    @test_throws "polish_refine_iter must be non-negative" setup(P, q, A, l, u; polish_refine_iter = -1)
    @test_throws "delta must be positive" setup(P, q, A, l, u; delta = 0.0)
end

@testitem "a run without a solution returns NaN, not a plausible point" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P = [2.0 5.0; 5.0 1.0]
    q = [3.0, 4.0]
    A = [-1.0 0.0; 0.0 -1.0; -1.0 3.0; 2.0 5.0; 3.0 4.0]
    s = PureOSQP.solve(
        P, q, A, fill(-Inf, 5), [0.0, 0.0, -15.0, 100.0, 80.0];
        sigma = 5.0, max_iter = 10_000
    )
    @test s.status == NON_CONVEX
    @test all(isnan, s.x)
    @test all(isnan, s.y)
    @test isnan(s.obj_val)
end

@testitem "an infeasible solve cold starts the workspace for the next one" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    A = reshape([1.0, 1.0], 2, 1)
    ws = setup(zeros(1, 1), [0.0], A, [1.0, -Inf], [Inf, 0.0])
    s1 = PureOSQP.solve!(ws)
    @test s1.status == PRIMAL_INFEASIBLE
    @test all(iszero, ws.y)          # not left on the diverging ray
    @test all(iszero, ws.x)
    s2 = PureOSQP.solve!(ws)
    @test s2.status == PRIMAL_INFEASIBLE
    @test s2.iter == s1.iter          # the second run is not a continuation of the first
end

@testitem "Float16 still classifies infinite bounds correctly" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    # INFTY(T) must stay finite for narrow float types, or the free-row test can never
    # be true and a free row silently gets the inequality rho.
    for T in (Float16, Float32, Float64)
        @test isfinite(PureOSQP.INFTY(T))
        A = Matrix{T}([1 0; 0 1; 1 1])
        l = T[-Inf, 1, 0]
        u = T[Inf, 1, 1]
        ws = setup(T, zeros(T, 2, 2), zeros(T, 2), A, l, u; scaling = 0, rho = 0.1)
        @test ws.constr_type == Int8[-1, 1, 0]
    end
end

@testitem "every Status value is exported" begin
    # A status a caller can receive but cannot name is useless: comparing against it
    # would be an UndefVarError. This caught the *_INACCURATE tier being unexported.
    for s in instances(Status)
        @test Base.isexported(PureOSQP, Symbol(s))
    end
end
