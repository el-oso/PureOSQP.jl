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
    # The factorizations run with `check = false`, so a non-finite stored entry is refused
    # here rather than answered with.
    @test_throws "P is not finite at entry (2, 2)" setup([1.0 0.0; 0.0 NaN], q, A, l, u)
    @test_throws "A is not finite at entry (1, 1)" setup(P, q, [Inf 0.0], l, u)
end

@testitem "Solution and OperatorSplittingWorkspace render as one line" begin
    ws = setup([4.0 1.0; 1.0 2.0], [1.0, 1.0], [1.0 1.0; 1.0 0.0], [0.0, 0.0], [1.0, 1.0])
    s = PureOSQP.solve!(ws)
    txt_ws = sprint(show, ws)
    txt_sol = sprint(show, s)
    @test occursin("PureOSQP OperatorSplittingWorkspace", txt_ws)
    @test occursin("backend", txt_ws)
    @test occursin("Solution", txt_sol)
    @test occursin("iterations", txt_sol)
    @test iszero(count('\n', txt_ws))
    @test iszero(count('\n', txt_sol))
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
    @test_throws "sigma must be positive" OperatorSplitting(sigma = 0.0)
    @test_throws "sigma must be positive" OperatorSplitting(sigma = -1.0)
    @test_throws "rho must be positive" OperatorSplitting(rho = -1.0)
    @test_throws "alpha must lie in (0, 2)" OperatorSplitting(alpha = 0.0)
    @test_throws "alpha must lie in (0, 2)" OperatorSplitting(alpha = 5.0)
    @test_throws "max_iter must be positive" setup(P, q, A, l, u; max_iter = 0)
    @test_throws "max_iter must be positive" setup(P, q, A, l, u; max_iter = -5)
    @test_throws "eps_abs and eps_rel must be non-negative" setup(P, q, A, l, u; eps_abs = -1.0)
    @test_throws "scaling must be non-negative" setup(P, q, A, l, u; scaling = -3)
    @test_throws "check_termination must be non-negative" setup(P, q, A, l, u; check_termination = -1)
    @test_throws "polish_refine_iter must be non-negative" setup(P, q, A, l, u; polish_refine_iter = -1)
    @test_throws "delta must be positive" setup(P, q, A, l, u; delta = 0.0)
end

@testitem "algorithm objects and shared options" begin
    using PureIPM
    using LinearAlgebra, ForwardDiff
    P = [4.0 1.0; 1.0 2.0]
    q = [1.0, 1.0]
    A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
    l = [1.0, 0.0, 0.0]
    u = [1.0, 0.7, 0.7]

    # A setting passed in the wrong place names where it belongs, under the algorithm in use.
    @test_throws "rho is a parameter of OperatorSplitting, not an option" solve(P, q, A, l, u; rho = 0.2)
    @test_throws "InteriorPoint(reg_primal = ...)" setup(P, q, A, l, u, InteriorPoint(); reg_primal = 1.0e-7)
    @test_throws ArgumentError update_settings!(setup(P, q, A, l, u); sigma = 1.0e-5)
    # A parameter of the other algorithm is not an option either, and is named as the
    # unrecognized keyword it is rather than reaching Options.
    @test_throws "rho is not an option, and not a parameter of InteriorPoint" solve(P, q, A, l, u, InteriorPoint(); rho = 0.2)
    @test_throws "reg_primal is not an option, and not a parameter of OperatorSplitting" setup(P, q, A, l, u; reg_primal = 1.0e-7)
    # An option is not a parameter, and an unknown name is refused too.
    @test_throws MethodError InteriorPoint(rho = 0.2)
    @test_throws MethodError OperatorSplitting(max_iter = 10)
    @test_throws "not_a_setting is not an option" setup(P, q, A, l, u; not_a_setting = 1)

    # The defaults that differ by algorithm come from the algorithm; a value given is kept.
    admm = default_options(OperatorSplitting(), Float64)
    ipm = default_options(InteriorPoint(), Float64)
    @test admm isa Options{Float64}
    @test (admm.max_iter, admm.eps_abs, admm.eps_prim_inf, admm.check_termination) == (4000, 1.0e-3, 1.0e-4, 25)
    @test (admm.cg_max_iter, admm.cg_tol_fraction) == (20, 0.15)
    @test (ipm.max_iter, ipm.eps_abs, ipm.eps_prim_inf, ipm.check_termination) == (100, 1.0e-8, 1.0e-8, 1)
    @test (ipm.cg_max_iter, ipm.cg_tol_fraction) == (500, 0.1)
    @test default_options(InteriorPoint(), Float32).eps_abs == sqrt(eps(Float32))
    @test setup(P, q, A, l, u).options == admm
    ws = setup(P, q, A, l, u, InteriorPoint(); max_iter = 50)
    @test ws.options.max_iter == 50
    @test ws.options.eps_abs == 1.0e-8

    # `update_settings!` in both forms, on the operator-splitting workspace.
    ws = setup(P, q, A, l, u, OperatorSplitting(rho = 0.2, adaptive_rho = :kkt_error); eps_abs = 1.0e-6)
    @test ws.algorithm == OperatorSplitting{Float64}(OperatorSplitting(rho = 0.2, adaptive_rho = :kkt_error))
    update_settings!(ws; eps_abs = 1.0e-8)
    @test ws.options.eps_abs == 1.0e-8
    @test ws.algorithm.rho == 0.2
    count = ws.refactor_count
    update_settings!(ws, OperatorSplitting(sigma = 1.0e-5))
    @test ws.algorithm.sigma == 1.0e-5
    @test ws.algorithm.rho == 0.1                 # a parameter left out takes its default
    @test ws.algorithm.adaptive_rho === :iterations
    @test ws.weights.sigma == 1.0e-5
    @test ws.refactor_count == count + 1
    @test ws.options.eps_abs == 1.0e-8            # the options are untouched
    @test solve!(ws).status == SOLVED
    @test_throws "the algorithm is fixed once the workspace is built" update_settings!(ws, InteriorPoint())

    # Each algorithm object takes the solve's element type, and the defaults that depend on it.
    for T in (Float32, BigFloat, ForwardDiff.Dual{Nothing, Float64, 1})
        c(v) = T.(v)
        os = setup(c(P), c(q), c(A), c(l), c(u), OperatorSplitting())
        @test os.algorithm isa OperatorSplitting{T}
        @test os.options isa Options{T}
        @test os.algorithm.rho == T(0.1)
        ip = setup(c(P), c(q), c(A), c(l), c(u), InteriorPoint())
        @test ip.algorithm isa InteriorPoint{T, T, T, Int}
        @test ip.options isa Options{T}
        @test ip.algorithm.reg_primal == PureIPM.ipm_floor(T)
        @test ip.options.eps_abs == PureIPM.ipm_floor(T)
        @test ip.algorithm.refine_iter == 1
    end
    @test setup(Float32.(P), Float32.(q), Float32.(A), Float32.(l), Float32.(u), InteriorPoint(reg_primal = 1.0e-6)).algorithm.reg_primal == 1.0f-6
    @test InteriorPoint(reg_primal = big"1e-40").reg_primal == big"1e-40"
end

@testitem "a run without a solution returns NaN, not a plausible point" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P = [2.0 5.0; 5.0 1.0]
    q = [3.0, 4.0]
    A = [-1.0 0.0; 0.0 -1.0; -1.0 3.0; 2.0 5.0; 3.0 4.0]
    s = PureOSQP.solve(
        P, q, A, fill(-Inf, 5), [0.0, 0.0, -15.0, 100.0, 80.0], OperatorSplitting(sigma = 5.0);
        max_iter = 10_000
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
        ws = setup(T, zeros(T, 2, 2), zeros(T, 2), A, l, u, OperatorSplitting(rho = 0.1); scaling = 0)
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

@testitem "a non-BLAS eltype solves on both backends" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    # Float16 is not a BLAS float, so it exercises LinearAlgebra's generic factorizations
    # rather than LAPACK. Both linear-system backends must still work.
    T = Float16
    P = Matrix{T}([4 1; 1 2])
    q = T[1, 1]
    A = Matrix{T}([1 1; 1 0; 0 1])
    l = T[1, 0, 0]
    u = T[1, 0.7, 0.7]
    s = PureOSQP.solve(P, q, A, l, u; eps_abs = T(1.0e-2), eps_rel = T(1.0e-2))
    @test s isa Solution{T}
    @test s.status == SOLVED
    @test setup(P, q, A, l, u; linsys = :kkt).linsys isa PureOSQP.FullKKT
    @test PureOSQP.solve(P, q, A, l, u; linsys = :kkt, eps_abs = T(1.0e-2), eps_rel = T(1.0e-2)).status == SOLVED
end

@testitem "the unchecked column traversals run behind a guard that fires" begin
    using LinearAlgebra, SparseArrays
    # The four sparse column traversals index a weight vector at a row read out of the
    # matrix, which no compiler can prove is in range, so they drop the check and
    # `check_storage` establishes the property once per `setup` instead. `validate` calls it
    # on `P` and `A`, so a caller cannot reach those loops without it having passed.
    @test PureOSQP.check_storage(sparse(1.0I, 3, 3), 3, 3) === nothing
    # A stored row past the end of the matrix: the access the loops no longer check.
    @test_throws "outside 1:3" PureOSQP.check_storage(SparseMatrixCSC(3, 3, [1, 2, 2, 2], [7], [1.0]), 3, 3)
    @test_throws "outside 1:3" PureOSQP.check_storage(SparseMatrixCSC(3, 3, [1, 2, 2, 2], [0], [1.0]), 3, 3)
    # Dimensions that disagree with the system would size the weight vectors wrongly.
    @test_throws "expected a 4×3 matrix" PureOSQP.check_storage(sparse(1.0I, 3, 3), 4, 3)
    # A dense matrix has nothing to establish: its traversals are provably in bounds.
    @test PureOSQP.check_storage(zeros(3, 3), 3, 3) === nothing
end

@testitem "a Symmetric wrapper is accepted for sparse and dense P" begin
    # `:ldl_kkt` and `:sparse_kkt` exist only once LDLFactorizations is loaded. Without this
    # import the item asserts whichever backends a sibling item happened to make reachable in
    # the same worker, which is a different assertion on every run.
    using LinearAlgebra, SparseArrays, LDLFactorizations
    n = m = 60
    M = sparse(Diagonal(range(1.0, 3.0; length = n)))
    M[1, 2] = M[2, 1] = 0.4
    # One dense row over an identity block: the row spans every column, so the reduced form
    # is out and the pair takes the KKT one.
    A = sparse(
        vcat(fill(1, n), 2:m), vcat(1:n, 1:(m - 1)),
        vcat(fill(1.0, n), fill(1.0, m - 1)), m, n
    )
    q = collect(range(-1.0, 1.0; length = n))
    l = fill(-1.0, m)
    u = fill(2.0, m)
    unwrapped = setup(M, q, A, l, u)
    @test PureOSQP.backend_name(unwrapped.linsys) in (:sparse_kkt, :ldl_kkt)
    base = solve!(unwrapped)
    @test base.status == SOLVED

    # A wrapper is not a `SparseMatrixCSC`, so the two sparse rungs decline it and the
    # ladder carries it to a backend that reaches `P` only through the column traversals.
    ws = setup(Symmetric(M), q, A, l, u)
    @test PureOSQP.backend_name(ws.linsys) == :sparse_formed
    wrapped = solve!(ws)
    @test wrapped.status == SOLVED
    # A different backend solves the same system in a different order, so the two agree to
    # rounding rather than bit for bit.
    @test wrapped.x ≈ base.x

    dense = solve!(setup(Symmetric(Matrix(M)), q, A, l, u))
    @test dense.status == SOLVED
    # Both wrappers descend to the same backend and are walked entrywise by the same
    # generic traversals, so these two agree exactly.
    @test dense.x == wrapped.x

    # A wrapper over a stored triangle is read through the wrapper's own mirroring
    # `getindex`, so it describes the same matrix and solves to the same point.
    triangle = solve!(setup(Symmetric(sparse(triu(M))), q, A, l, u))
    @test triangle.status == SOLVED
    @test triangle.x == wrapped.x
end
