@testitem "the QPWorkspace contract is enforced, not decorative" begin
    using TypeContracts
    W = PureOSQP.QPWorkspace
    spec = TypeContracts.list_contract(W)
    required = [s.description for s in spec if !s.optional]
    @test required == [
        "solve!(::Self) :: Solution",
        "warm_start!(::Self) :: Self",
        "cold_start!(::Self) :: Self",
        "update!(::Self) :: Self",
        "update_settings!(::Self) :: Self",
        "update_settings!(::Self, ::QPAlgorithm) :: Self",
        "dimensions(::Self) :: Tuple{Int, Int}",
        "derivative_ready(::Self) :: Nothing",
    ]
    @test sort([nameof(s.f) for s in spec if s.optional]) ==
        [:constraint_violation, :update_rho!]

    for T in (OperatorSplittingWorkspace, InteriorPointWorkspace)
        @test TypeContracts.satisfies(T, W).satisfied
    end
    @test isempty(TypeContracts.satisfies(OperatorSplittingWorkspace, W).missing_optional)

    # A workspace that declares the supertype and implements nothing inherits only the
    # methods written for every `QPWorkspace`; the rest are reported by name.
    @eval struct IncompleteWorkspace <: PureOSQP.QPWorkspace{Float64} end
    r = TypeContracts.satisfies(IncompleteWorkspace, W)
    @test !r.satisfied
    @test r.missing_methods == [
        "solve!(::Self) :: Solution",
        "warm_start!(::Self) :: Self",
        "cold_start!(::Self) :: Self",
        "update!(::Self) :: Self",
    ]
    @test_throws "solve!(::Self) :: Solution" TypeContracts.check_contract(IncompleteWorkspace, W)
end

@testitem "the QPAlgorithm contract is enforced, not decorative" begin
    using TypeContracts
    Alg = PureOSQP.QPAlgorithm
    spec = TypeContracts.list_contract(Alg)
    @test [nameof(s.f) for s in spec if !s.optional] ==
        [:setup_backend, :algorithm_defaults, :default_options, :element_typed]
    @test [nameof(s.f) for s in spec if s.optional] == [:adopt_settings!]

    for T in (OperatorSplitting, InteriorPoint)
        @test TypeContracts.satisfies(T, Alg).satisfied
    end

    @eval struct IncompleteAlgorithm <: PureOSQP.QPAlgorithm end
    r = TypeContracts.satisfies(IncompleteAlgorithm, Alg)
    @test !r.satisfied
    @test [first(split(m, '(')) for m in r.missing_methods] ==
        ["setup_backend", "algorithm_defaults", "element_typed"]
    @test_throws "setup_backend(::Self" TypeContracts.check_contract(IncompleteAlgorithm, Alg)
end

@testitem "the Preconditioner contract covers built-in and caller preconditioners" begin
    using LinearAlgebra, SparseArrays, Random, Krylov, TypeContracts
    include(joinpath(@__DIR__, "helpers.jl"))
    Pre = PureOSQP.Preconditioner
    @test [s.description for s in TypeContracts.list_contract(Pre)] == [
        "update_preconditioner!(::Self, ::Problem, ::SystemWeights, ::Int) :: Self",
        "LinearAlgebra.ldiv!(::AbstractVector, ::Self, ::AbstractVector)",
    ]
    for T in (IdentityPreconditioner, JacobiPreconditioner)
        @test T <: Pre
        @test TypeContracts.satisfies(T, Pre).satisfied
    end

    # Caller preconditioners are not subtypes and satisfy the same contract: a factorization
    # object as it is, and the reference preconditioners of the benchmarks. The benchmark file
    # is loaded without its `using` line, which names a package the tests do not depend on;
    # the preconditioner bodies that need it are not run here.
    P, q, A, l, u = random_qp(10, 15; seed = 5)
    chol = cholesky(Symmetric(P + I))
    @test TypeContracts.check_contract(typeof(chol), Pre).passed
    bench = Module(:BenchPreconditioners)
    Core.eval(bench, :(using PureOSQP, LinearAlgebra, SparseArrays))
    Base.include(
        ex -> Meta.isexpr(ex, :using) ? nothing : ex, bench,
        joinpath(@__DIR__, "..", "bench", "ipm_preconditioners.jl"),
    )
    @test TypeContracts.check_contract(bench.LaggedCholesky, Pre).passed
    @test TypeContracts.check_contract(bench.IncompleteLDL, Pre).passed

    @eval struct NoApply end
    r = TypeContracts.satisfies(NoApply, Pre)
    @test r.missing_methods == ["LinearAlgebra.ldiv!(::AbstractVector, ::Self, ::AbstractVector)"]
    @test_throws "LinearAlgebra.ldiv!(::AbstractVector, ::Self, ::AbstractVector)" TypeContracts.check_contract(NoApply, Pre)

    # `setup` refuses it by name under both algorithms, before building anything.
    for alg in (OperatorSplitting(), InteriorPoint())
        @test_throws "has no method LinearAlgebra.ldiv!" setup(
            P, q, A, l, u, alg; linsys = :indirect, scaling = 0, preconditioner = NoApply()
        )
        ws = setup(P, q, A, l, u, alg; linsys = :indirect, scaling = 0, preconditioner = chol)
        @test PureOSQP.backend_name(ws.linsys) === :indirect
    end
end
