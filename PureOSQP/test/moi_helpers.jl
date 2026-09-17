using MathOptInterface
const MOI = MathOptInterface

"""
    moi_model(; kwargs...) -> MOI.Utilities.CachingOptimizer

The wrapper as a caller reaches it, with tolerances tight enough for `MOI.Test`. The
solver's defaults are `1e-3`, matching upstream, and `MOI.Test` checks answers to `1e-4`.
An ADMM solver asked for three digits and judged on four fails on the digit it was never
asked to produce, so the tolerances are tightened rather than the comparison loosened.
"""
function moi_model()
    m = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        MOI.instantiate(PureOSQP.Optimizer; with_bridge_type = Float64),
    )
    MOI.set(m, MOI.Silent(), true)
    MOI.set(m, MOI.RawOptimizerAttribute("eps_abs"), 1.0e-9)
    MOI.set(m, MOI.RawOptimizerAttribute("eps_rel"), 1.0e-9)
    return m
end

"""
    moi_config() -> MOI.Test.Config

Neither algorithm tracks a basis, and this solver reports no bound on the objective, so
those three attributes are excluded.
"""
moi_config() = MOI.Test.Config(;
    atol = 1.0e-4, rtol = 1.0e-4,
    exclude = Any[MOI.ConstraintBasisStatus, MOI.VariableBasisStatus, MOI.ObjectiveBound],
)

"""
The name prefixes of the first seven `MOI.Test` groups. The eighth group is everything these
do not match, so the eight together run each test exactly once.

The grouping is by compile cost, which is what dominates a `MOI.Test` run. `MOI.Test`
generates a `test_basic_<function>_<set>` test for every function and set it knows, which
is most of the suite, so those are split by function type.
"""
const MOI_GROUPS = (
    ["test_linear_", "test_quadratic_"],
    ["test_conic_"],
    ["test_model_", "test_solve_", "test_modification_", "test_attribute_"],
    ["test_basic_VariableIndex_", "test_basic_Scalar"],
    ["test_basic_VectorOfVariables_"],
    ["test_basic_VectorAffineFunction_"],
    ["test_basic_VectorQuadraticFunction_", "test_basic_VectorNonlinearFunction_"],
)
