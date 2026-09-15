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

@testitem "the MathOptInterface wrapper passes MOI.Test at algorithm = :ipm" begin
    using MathOptInterface, LinearAlgebra, SparseArrays
    const MOI = MathOptInterface

    # A narrower subset than the ADMM run above: the interior-point method is measured
    # end to end against the structural corpus and the c-suite (`test/ipm_tests.jl`), and
    # this only checks that the MOI wrapper itself dispatches to it and reports its numbers,
    # not a second full pass of MOI.Test.
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        MOI.instantiate(PureOSQP.Optimizer; with_bridge_type = Float64),
    )
    MOI.set(model, MOI.Silent(), true)
    MOI.set(model, MOI.RawOptimizerAttribute("algorithm"), :ipm)
    # Default IPM tolerances (`1e-8`) already clear the `1e-4` MOI.Test checks, unlike the
    # ADMM run above, whose `1e-3` defaults do not, so nothing is tightened here.
    #
    # The three exclusions are the same the ADMM run needs (neither algorithm tracks a basis,
    # and this solver has no bound on the objective to report). No exclusion beyond those was
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

@testitem "raw settings are checked when set and read back their defaults" begin
    using MathOptInterface
    const MOI = MathOptInterface

    o = PureOSQP.Optimizer()
    @test MOI.get(o, MOI.RawOptimizerAttribute("max_iter")) == 4000
    @test MOI.get(o, MOI.RawOptimizerAttribute("linsys")) === :auto
    @test_throws MOI.UnsupportedAttribute MOI.get(o, MOI.RawOptimizerAttribute("not_a_setting"))

    @test_throws "linsys must be one of" MOI.set(o, MOI.RawOptimizerAttribute("linsys"), :nope)
    @test_throws "alpha must lie in (0, 2)" MOI.set(o, MOI.RawOptimizerAttribute("alpha"), 3.0)
    # A refused value leaves the setting as it was.
    @test MOI.get(o, MOI.RawOptimizerAttribute("alpha")) == 1.6

    MOI.set(o, MOI.RawOptimizerAttribute("linsys"), "kkt")
    @test MOI.get(o, MOI.RawOptimizerAttribute("linsys")) === :kkt
end

@testitem "the algorithm attribute selects which settings the raw attributes validate against" begin
    using MathOptInterface
    const MOI = MathOptInterface

    o = PureOSQP.Optimizer()
    @test MOI.supports(o, MOI.RawOptimizerAttribute("algorithm"))
    @test MOI.get(o, MOI.RawOptimizerAttribute("algorithm")) === :admm
    @test_throws "algorithm must be :admm or :ipm" MOI.set(
        o, MOI.RawOptimizerAttribute("algorithm"), :nope
    )

    # An ADMM-only field, checked against `Settings` by default.
    MOI.set(o, MOI.RawOptimizerAttribute("rho"), 0.2)
    @test MOI.get(o, MOI.RawOptimizerAttribute("rho")) == 0.2

    MOI.set(o, MOI.RawOptimizerAttribute("algorithm"), "ipm")
    @test MOI.get(o, MOI.RawOptimizerAttribute("algorithm")) === :ipm
    # `rho` belongs to `Settings`, not `IPMSettings`, so it is no longer a raw attribute here.
    @test !MOI.supports(o, MOI.RawOptimizerAttribute("rho"))
    # An IPM-only field, checked against `IPMSettings` once `algorithm` names it.
    @test MOI.supports(o, MOI.RawOptimizerAttribute("max_reg_bumps"))
    MOI.set(o, MOI.RawOptimizerAttribute("max_reg_bumps"), 3)
    @test MOI.get(o, MOI.RawOptimizerAttribute("max_reg_bumps")) == 3
    @test_throws "must be non-negative" MOI.set(o, MOI.RawOptimizerAttribute("max_reg_bumps"), -1)
end

@testitem "NUMERICAL_ERROR maps to MOI.NUMERICAL_ERROR with no result" begin
    using MathOptInterface
    const MOI = MathOptInterface

    # ADMM never ends this way, so the status is placed on a solution directly.
    o = PureOSQP.Optimizer()
    s = PureOSQP.solve([2.0;;], [1.0], [1.0;;], [-1.0], [1.0])
    fields = ntuple(i -> getfield(s, i), fieldcount(typeof(s)))
    o.sol = typeof(s)(fields[1:2]..., PureOSQP.NUMERICAL_ERROR, fields[4:end]...)
    @test MOI.get(o, MOI.TerminationStatus()) == MOI.NUMERICAL_ERROR
    @test MOI.get(o, MOI.ResultCount()) == 0
    @test MOI.get(o, MOI.PrimalStatus()) == MOI.NO_SOLUTION
    @test MOI.get(o, MOI.DualStatus()) == MOI.NO_SOLUTION
    @test MOI.get(o, MOI.RawStatusString()) == "NUMERICAL_ERROR"
end
