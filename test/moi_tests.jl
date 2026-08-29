@testitem "the MathOptInterface wrapper passes MOI.Test" begin
    using MathOptInterface, LinearAlgebra, SparseArrays
    const MOI = MathOptInterface

    # `MOI.Test` is the point of this test item: it is a far more thorough suite than
    # anything written here would be, and it is the same one every registered solver runs.
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        MOI.instantiate(PureOSQP.Optimizer; with_bridge_type = Float64),
    )
    MOI.set(model, MOI.Silent(), true)
    # The solver's defaults are `1e-3`, matching upstream, and `MOI.Test` checks answers to
    # `1e-4`. An ADMM solver asked for three digits and judged on four fails on the digit
    # it was never asked to produce, so the tolerances are tightened rather than the
    # comparison loosened.
    MOI.set(model, MOI.RawOptimizerAttribute("eps_abs"), 1.0e-9)
    MOI.set(model, MOI.RawOptimizerAttribute("eps_rel"), 1.0e-9)
    MOI.Test.runtests(
        model,
        MOI.Test.Config(;
            atol = 1.0e-4, rtol = 1.0e-4,
            exclude = Any[MOI.ConstraintBasisStatus, MOI.VariableBasisStatus, MOI.ObjectiveBound],
        ),
    )
end

@testitem "the wrapper reports the solver's own numbers" begin
    using MathOptInterface, LinearAlgebra, SparseArrays
    const MOI = MathOptInterface

    # minimize (x-1)^2 + (y-2)^2  s.t.  x + y <= 2,  x >= 0,  y >= 0
    o = PureOSQP.Optimizer()
    src = MOI.Utilities.Model{Float64}()
    x = MOI.add_variables(src, 2)
    MOI.add_constraint.(src, x, MOI.GreaterThan(0.0))
    MOI.add_constraint(
        src,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(1.0, x), 0.0),
        MOI.LessThan(2.0),
    )
    obj = MOI.ScalarQuadraticFunction(
        MOI.ScalarQuadraticTerm.([2.0, 2.0], x, x),
        MOI.ScalarAffineTerm.([-2.0, -4.0], x),
        5.0,
    )
    MOI.set(src, MOI.ObjectiveFunction{typeof(obj)}(), obj)
    MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.copy_to(o, src)
    MOI.optimize!(o)

    @test MOI.get(o, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(o, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(o, MOI.DualStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(o, MOI.ResultCount()) == 1
    @test MOI.get(o, MOI.SolverName()) == "PureOSQP"

    # The constrained optimum of this problem: x + y = 2 is active.
    xv = MOI.get.(o, MOI.VariablePrimal(), x)
    @test sum(xv) ≈ 2.0 atol = 1.0e-4
    @test MOI.get(o, MOI.ObjectiveValue()) ≈ 0.5 atol = 1.0e-4

    # These come straight from `Solution`, so they double as a check that the fields added
    # for reporting are wired through rather than recomputed.
    @test MOI.get(o, MOI.SolveTimeSec()) > 0
    @test MOI.get(o, MOI.BarrierIterations()) > 0
    @test MOI.get(o, MOI.DualObjectiveValue()) ≈ MOI.get(o, MOI.ObjectiveValue()) atol = 1.0e-3

    # Settings reach the solver by their own names.
    @test MOI.supports(o, MOI.RawOptimizerAttribute("eps_abs"))
    @test !MOI.supports(o, MOI.RawOptimizerAttribute("not_a_setting"))
    MOI.set(o, MOI.RawOptimizerAttribute("eps_abs"), 1.0e-10)
    @test MOI.get(o, MOI.RawOptimizerAttribute("eps_abs")) == 1.0e-10
    MOI.set(o, MOI.TimeLimitSec(), 5.0)
    @test MOI.get(o, MOI.TimeLimitSec()) == 5.0
end
