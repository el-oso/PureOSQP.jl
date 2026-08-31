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
export dimensions, capabilities, Optimizer, Settings, Solution, Status
export adjoint_derivative, forward_derivative
export LinearSystem, ReducedCholesky, FullKKT
export SOLVED, PRIMAL_INFEASIBLE, DUAL_INFEASIBLE, MAX_ITER_REACHED, NON_CONVEX, UNSOLVED
export TIME_LIMIT_REACHED, INTERRUPTED
export PolishStatus, POLISH_SUCCESS, POLISH_FAILED, POLISH_NOT_PERFORMED
export POLISH_NO_ACTIVE_SET_FOUND, POLISH_LINSYS_ERROR
export SOLVED_INACCURATE, PRIMAL_INFEASIBLE_INACCURATE, DUAL_INFEASIBLE_INACCURATE

include("blockdiagonal.jl")
include("kronecker.jl")
include("rowcoupled.jl")
include("linsys.jl")
include("operator.jl")
include("lowrank.jl")
include("block.jl")
include("kronsolve.jl")
include("types.jl")
include("elementwise.jl")
include("scaling.jl")
include("rho.jl")
include("termination.jl")
include("admm.jl")
include("polish.jl")
include("derivative.jl")
include("update.jl")
include("api.jl")

"""
    Optimizer(; kwargs...)

MathOptInterface optimizer, available once MathOptInterface is loaded. Keyword arguments
are the fields of [`Settings`](@ref).

The wrapper lives in a package extension, so it costs nothing to a caller who does not use
it; this name is the only part of it the core owns.
"""
function Optimizer end

end # module PureOSQP
