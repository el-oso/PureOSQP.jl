@testitem "the MathOptInterface wrapper passes MOI.Test" begin
    using PureIPM
    using MathOptInterface, LinearAlgebra, SparseArrays
    const MOI = MathOptInterface

    # A narrower subset than the one PureOSQP runs for operator splitting: the interior-point
    # method is measured end to end against the structural corpus and the c-suite
    # (`PureIPM/test/ipm_tests.jl`), and this only checks that the MOI wrapper itself
    # dispatches to it and reports its numbers, not a second full pass of MOI.Test.
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        MOI.instantiate(PureIPM.Optimizer; with_bridge_type = Float64),
    )
    MOI.set(model, MOI.Silent(), true)
    # The default tolerances (`1e-8`) already clear the `1e-4` MOI.Test checks, so nothing is
    # tightened here.
    #
    # The three exclusions are the ones neither algorithm can satisfy: neither tracks a basis,
    # and this solver has no bound on the objective to report. No exclusion beyond those was
    # needed to pass this subset.
    MOI.Test.runtests(
        model,
        MOI.Test.Config(;
            atol = 1.0e-4, rtol = 1.0e-4,
            exclude = Any[MOI.ConstraintBasisStatus, MOI.VariableBasisStatus, MOI.ObjectiveBound],
        ),
        include = ["test_linear_", "test_quadratic_"],
    )
end

@testitem "the optimizer reaches optimize! and reports its iterations" begin
    using PureIPM
    using MathOptInterface
    const MOI = MathOptInterface

    o = PureIPM.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    src = MOI.Utilities.Model{Float64}()
    x = MOI.add_variables(src, 2)
    MOI.add_constraint.(src, x, MOI.GreaterThan(0.0))
    obj = MOI.ScalarQuadraticFunction(
        MOI.ScalarQuadraticTerm.([2.0, 2.0], x, x), MOI.ScalarAffineTerm.([-2.0, -4.0], x), 5.0,
    )
    MOI.set(src, MOI.ObjectiveFunction{typeof(obj)}(), obj)
    MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.copy_to(o, src)
    MOI.optimize!(o)
    @test MOI.get(o, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(o, MOI.BarrierIterations()) > 0
end
