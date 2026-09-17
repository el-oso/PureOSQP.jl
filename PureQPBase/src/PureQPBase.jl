"""
    PureQPBase

The algorithm-independent core of PureOSQP: the problem representation, the linear-system
backends, equilibration, termination, polishing kernels, and the `QPAlgorithm`/`QPWorkspace`
contracts an algorithm implements.

This package solves nothing by itself — [`setup`](@ref) and [`solve`](@ref) both take a
mandatory `alg::QPAlgorithm`, and no concrete algorithm is defined here. PureOSQP.jl supplies
the algorithms (`OperatorSplitting` and `InteriorPoint`) and is the package most callers want.
"""
module PureQPBase

using LinearAlgebra
using TypeContracts: TypeContracts, @contract, @verify

include("blockdiagonal.jl")
include("kronecker.jl")
include("rowcoupled.jl")
include("problem.jl")
include("options.jl")
include("weights.jl")
include("linsys.jl")
include("preconditioner.jl")
include("operator.jl")
include("lowrank.jl")
include("block.jl")
include("kronsolve.jl")
include("types.jl")
include("elementwise.jl")
include("scaling.jl")
include("termination.jl")
include("polish.jl")
include("derivative.jl")
include("recommend.jl")
include("api.jl")
include("conformance.jl")

export setup, solve, solve!, update!, update_settings!, update_rho!, warm_start!, cold_start!
export dimensions, capabilities, constraint_violation
export Solution, Status, Options, default_options
export QPAlgorithm
export QPWorkspace
export has_solution, status_name
export recommend_linsys, LinsysAdvice
export backend_info, backend_name, factor_fill, BackendInfo
export PolishStatus
export adjoint_derivative, forward_derivative
export LinearSystem, ReducedCholesky, FullKKT
export Preconditioner, IdentityPreconditioner, JacobiPreconditioner, update_preconditioner!
export SOLVED, PRIMAL_INFEASIBLE, DUAL_INFEASIBLE, MAX_ITER_REACHED, NON_CONVEX, UNSOLVED
export TIME_LIMIT_REACHED, INTERRUPTED, NUMERICAL_ERROR
export PolishStatus, POLISH_SUCCESS, POLISH_FAILED, POLISH_NOT_PERFORMED
export POLISH_NO_ACTIVE_SET_FOUND, POLISH_LINSYS_ERROR
export SOLVED_INACCURATE, PRIMAL_INFEASIBLE_INACCURATE, DUAL_INFEASIBLE_INACCURATE

# Every `LinearSystem` and built-in preconditioner defined by the time this module finishes
# must satisfy its contract and be `--trim` compatible, asserted here rather than type by
# type: a per-type `@verify` is opt-in, so a new type acquires the guarantee only if whoever
# wrote it remembered to ask. This sees every subtype defined here, so forgetting is not
# possible. An extension's backends load later and carry the same declaration at the end of
# the extension. The `QPAlgorithm` and `QPWorkspace` contracts are declared here and asserted
# by the packages that implement them, since this one defines no concrete subtype of either.
@verify LinearSystem subtypes = true trim_compat = true
@verify Preconditioner subtypes = true trim_compat = true

end # module PureQPBase
