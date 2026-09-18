"""
    PureDAQP

A dual active-set method for

    minimize    ½ xᵀPx + qᵀx
    subject to  l ≤ Ax ≤ u

The problem representation and the generic `setup`/`solve`/`solve!` come from PureQPBase.jl
and are re-exported here, so `using PureDAQP` is enough to solve a problem:

    solve(P, q, A, l, u, ActiveSet())

The method reduces the problem to a least-distance problem and maintains an `LDLᵀ` of the
working set's Gram matrix under rank-one updates, so each iteration costs `O(k²)` in the size
of the working set rather than a fresh factorization. It follows Algorithm 1 of Arnström,
Bemporad and Axehill, *A dual active-set solver for embedded quadratic programming using
recursive LDLᵀ updates*, IEEE Transactions on Automatic Control 67(8):4362-4369, 2022, with
that paper's Algorithm 2 as the proximal-point outer loop. The reference implementation is
MIT-licensed and was read alongside the paper.

Unlike the other algorithms in this repository it reads `P` and `A` as dense matrices and
exploits neither sparsity nor declared structure: the reduction forms `A R⁻¹`, which is dense
whatever `A` was. It is at its best where an active-set method always is, with few rows
active at the solution.
"""
module PureDAQP

using LinearAlgebra
using TypeContracts: TypeContracts, @contract, @verify
using PureQPBase

import PureQPBase:
    setup, solve, solve!, warm_start!, cold_start!, update!, update_settings!,
    derivative_ready, setup_backend, algorithm_defaults, element_typed, dimensions,
    Problem, Options, Solution, Status, QPAlgorithm, QPWorkspace, PolishStatus,
    adopt_update!, check_update, has_solution, is_convex, is_materializable, validate,
    validated_problem, validate_update!

export setup, solve, solve!, update!, update_settings!, warm_start!, cold_start!
export dimensions, capabilities
export Solution, Status, Options, default_options
export QPAlgorithm, ActiveSet
export QPWorkspace, ActiveSetWorkspace
export has_solution, status_name
export SOLVED, PRIMAL_INFEASIBLE, DUAL_INFEASIBLE, MAX_ITER_REACHED, NON_CONVEX, UNSOLVED
export TIME_LIMIT_REACHED, INTERRUPTED, NUMERICAL_ERROR
export SOLVED_INACCURATE, PRIMAL_INFEASIBLE_INACCURATE, DUAL_INFEASIBLE_INACCURATE

include("ldl.jl")
include("ldp.jl")
include("settings.jl")
include("workspace.jl")
include("solution.jl")

# Every workspace and algorithm this package defines must satisfy its contract, asserted for
# each subtype defined by the time the module finishes rather than type by type, so a new
# type cannot acquire the guarantee only by someone remembering to ask for it.
@verify QPWorkspace subtypes = true
@verify QPAlgorithm subtypes = true

end # module PureDAQP
