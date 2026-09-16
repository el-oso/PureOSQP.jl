# Interfaces

A new algorithm, a new linear-system backend or a new preconditioner plugs into the package by
implementing a short list of methods for a subtype of an abstract type. Each list is declared
with [TypeContracts.jl](https://github.com/el-oso/TypeContracts.jl)'s `@contract`, and every
subtype in the package is checked against it when the package precompiles, including whether
each method's inferred return type matches. `TypeContracts.describe(T)` prints the list for
`T`; the tables below state the same lists.

A required method must exist for the subtype. An optional method either has a default that
serves every subtype or is specific to one algorithm. `TypeContracts.satisfies(S, T)` reports
what a type `S` is missing, and `TypeContracts.check_contract(S, T)` throws naming it.

## An algorithm

An algorithm is an object holding the parameters only that method reads, a subtype of
[`QPAlgorithm`](@ref). [`setup`](@ref) calls these on it:

| method | returns | what it does |
|---|---|---|
| `setup_backend(alg, Val(linsys), T, P, q, A, l, u; kwargs...)` | a workspace | validates the data, builds the options, the problem and the backend, and factorizes |
| `algorithm_defaults(alg, T)` | `NamedTuple` | the [`Options`](@ref) defaults of this algorithm that have no common default |
| `default_options(alg, T)` | `Options` | the options a solve runs with when no keyword is passed; a default serves every algorithm |
| `element_typed(alg, T, options)` | `QPAlgorithm` | the object in the solve's element type, with every default resolved |

Optional: `adopt_settings!(ls, alg, options)`, which copies the parameters a backend reads into
it. Only the matrix-free backend reads any, so an algorithm that runs on it defines this method
for that backend. It is optional because a direct backend reads none; the matrix-free one
throws if it factorizes before its settings are copied in, rather than silently solving with a
zero iteration budget.

`setup_backend` declares no return type in the contract: inferred through the abstract data
arguments of the contract's signature it is `Any`, although a call with concrete arguments
returns a concrete workspace.

An algorithm also needs a [`PureOSQP.SelectionFor`](@ref) tag of its own and the four
selection methods that have no algorithm-independent answer:
[`select_backend`](@ref PureOSQP.select_backend), the order of its ladder;
[`dense_rung`](@ref PureOSQP.dense_rung), its terminal;
[`indirect_rung`](@ref PureOSQP.indirect_rung), what sits below the terminal; and, if the
sparse rungs are in that ladder, the SparseArrays extension's `sparse_form`. Every other
selection method already takes any tag: the rungs that, by default, are skipped so selection
moves to the next one, the `choose_backend` methods for a structured pair, and the errors the
GPU extension raises when a GPU array reaches a backend that cannot serve it. A tag with one of
the four missing gets an error naming the method, not a `MethodError`.

## A workspace

The solver state an algorithm's `setup_backend` builds is a subtype of
[`QPWorkspace`](@ref). Code outside the algorithm calls these on it:

| method | returns | what it does |
|---|---|---|
| [`solve!(ws)`](@ref solve!) | [`Solution`](@ref) | runs the algorithm from the workspace's state |
| [`warm_start!(ws; x, y)`](@ref warm_start!) | `ws` | seeds the next solve, in problem space |
| [`cold_start!(ws)`](@ref cold_start!) | `ws` | discards the iterates |
| [`update!(ws; q, l, u, P, A)`](@ref update!) | `ws` | replaces problem data |
| [`update_settings!(ws; kwargs...)`](@ref update_settings!) | `ws` | merges options; a default serves every workspace |
| [`update_settings!(ws, alg)`](@ref update_settings!) | `ws` | replaces the algorithm parameters; the default throws for another algorithm's object |
| [`dimensions(ws)`](@ref dimensions) | `Tuple{Int, Int}` | the number of variables and of rows; a default serves every workspace |

Optional, because only [`OperatorSplittingWorkspace`](@ref) implements them:
[`update_rho!`](@ref) and [`constraint_violation`](@ref).

The contract checks the positional signature of a method taking keywords, which is all it can
see. It does not check fields, and the methods written for every workspace ([`dimensions`](@ref),
the keyword form of [`update_settings!`](@ref), [`adjoint_derivative`](@ref) and
[`forward_derivative`](@ref)) read `prob`, `linsys`, `algorithm`, `options`, `x`, `y`, `z`,
`status` and `polished`.

## A linear-system backend

A backend is a subtype of [`LinearSystem`](@ref); the workspace holds one and hands it the
[`PureOSQP.Problem`](@ref) and the [`PureOSQP.SystemWeights`](@ref) on every call.

| method | returns | what it does |
|---|---|---|
| [`factorize!(ls, prob, wt)`](@ref PureOSQP.factorize!) | `Bool` | rebuilds the factorization; `false` when it cannot |
| [`solve_system!(ls, prob, wt, rhs_x, rhs_z, x, z)`](@ref PureOSQP.solve_system!) | `Nothing` | solves for `x` and writes `z = Ãx` |
| [`backend_info(ls)`](@ref backend_info) | [`BackendInfo`](@ref) | what the backend is and how large its factorization is |

Optional, each with a default for every backend:

| method | returns | default |
|---|---|---|
| [`refactor_weights!(ls, prob, wt)`](@ref PureOSQP.refactor_weights!) | `Bool` | calls `factorize!` |
| [`solve_multiplier!(ls, prob, wt, rhs_x, rhs_z, x, nu)`](@ref PureOSQP.solve_multiplier!) | `Nothing` | derives `ν` from `solve_system!` |
| [`check_update(ls, P, A)`](@ref PureOSQP.check_update) | `Nothing` | accepts |
| [`set_tolerance_level!(ls, level)`](@ref PureOSQP.set_tolerance_level!) | `Nothing` | ignores it |
| [`set_refresh_index!(ls, k)`](@ref PureOSQP.set_refresh_index!) | `Nothing` | ignores it |
| [`adopt_settings!(ls, alg, options)`](@ref PureOSQP.adopt_settings!) | `Nothing` | ignores them |
| [`use_residual_stop!(ls, on)`](@ref PureOSQP.use_residual_stop!) | `Nothing` | ignores it |
| [`last_solve_converged(ls)`](@ref PureOSQP.last_solve_converged) | `Bool` | `true` |
| [`inner_iterations(ls)`](@ref PureOSQP.inner_iterations) | `Int` | `0` |

A backend defined in an extension is checked when the extension loads.

## A preconditioner

The matrix-free backend (`linsys = :indirect`) applies a preconditioner `M` through two
methods:

| method | returns | what it does |
|---|---|---|
| [`update_preconditioner!(M, prob, wt, k)`](@ref update_preconditioner!) | `M`'s type | refreshes for the current weights; the default returns `M` unchanged |
| `LinearAlgebra.ldiv!(y, M, x)` | | writes the preconditioned vector into `y`; no default |

[`IdentityPreconditioner`](@ref) and [`JacobiPreconditioner`](@ref) are subtypes of
[`Preconditioner`](@ref). A caller's preconditioner need not be: a `Cholesky` or any other
factorization object has `ldiv!` already, and `TypeContracts.check_contract(typeof(M),
Preconditioner)` checks one against the same list. [`setup`](@ref) throws, naming the missing
method, when a preconditioner has no `ldiv!` method for the backend's vectors.
