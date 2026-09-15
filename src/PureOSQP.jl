"""
    PureOSQP

Pure-Julia implementation of the OSQP operator-splitting solver for

    minimize    ½ xᵀPx + qᵀx
    subject to  l ≤ Ax ≤ u

`P` and `A` may be any `AbstractMatrix`; there is no sparse-matrix dependency and the
caller's matrices are never modified. The algorithm follows Stellato, Banjac, Goulart,
Bemporad and Boyd, *OSQP: an operator splitting solver for quadratic programs*,
Mathematical Programming Computation 12(4):637–672, 2020.
"""
module PureOSQP

using LinearAlgebra
using TypeContracts: TypeContracts, @contract, @verify

export setup, solve, solve!, update!, update_settings!, update_rho!, warm_start!, cold_start!
export dimensions, capabilities, constraint_violation, constraint_violation!
export Optimizer, Settings, Solution, Status, Workspace
export has_solution, status_name
export backend_info, backend_name, factor_fill, BackendInfo
export PolishStatus
export adjoint_derivative, forward_derivative
export LinearSystem, ReducedCholesky, FullKKT
export IdentityPreconditioner, JacobiPreconditioner, update_preconditioner!
export SOLVED, PRIMAL_INFEASIBLE, DUAL_INFEASIBLE, MAX_ITER_REACHED, NON_CONVEX, UNSOLVED
export TIME_LIMIT_REACHED, INTERRUPTED
export PolishStatus, POLISH_SUCCESS, POLISH_FAILED, POLISH_NOT_PERFORMED
export POLISH_NO_ACTIVE_SET_FOUND, POLISH_LINSYS_ERROR
export SOLVED_INACCURATE, PRIMAL_INFEASIBLE_INACCURATE, DUAL_INFEASIBLE_INACCURATE

include("core/blockdiagonal.jl")
include("core/kronecker.jl")
include("core/rowcoupled.jl")
include("core/problem.jl")
include("core/weights.jl")
include("core/linsys.jl")
include("core/preconditioner.jl")
include("core/operator.jl")
include("core/lowrank.jl")
include("core/block.jl")
include("core/kronsolve.jl")
include("types.jl")
include("admm/accelerate.jl")
include("core/elementwise.jl")
include("core/scaling.jl")
include("admm/rho.jl")
include("admm/termination.jl")
include("admm/admm.jl")
include("core/polish.jl")
include("core/derivative.jl")
include("core/update.jl")
include("admm/api.jl")

"""
    Optimizer(; kwargs...)

MathOptInterface optimizer, available once MathOptInterface is loaded. Keyword arguments
are the fields of [`Settings`](@ref).

The wrapper lives in a package extension, so it costs nothing to a caller who does not use
it; this name is the only part of it the core owns.
"""
function Optimizer end

# Every [`LinearSystem`](@ref) in this module must be `--trim` compatible, asserted here rather
# than backend by backend: a per-backend `@verify` is opt-in, so a new backend acquires the
# guarantee only if whoever wrote it remembered to ask. This sees every subtype defined by the
# time the module finishes, so forgetting is not possible. An extension's backends load later
# and carry the same declaration at the end of the extension.
@verify LinearSystem subtypes = true trim_compat = true

end # module PureOSQP
