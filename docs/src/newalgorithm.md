# Adding an algorithm

PureQPBase defines no algorithm. What it holds is everything an algorithm needs and would
otherwise write itself: the problem representation, equilibration, every linear-system backend
and the order they are tried in, the termination tests, the infeasibility certificates, the
polishing and derivative kernels, and the generic `setup`, `solve` and `solve!`.

PureIPM is the worked example. It was added to a package that already had operator splitting,
and the split between what it wrote and what it inherited is the argument for the base
existing:

| | lines |
|---|---|
| `ipm.jl` — the Mehrotra iteration itself | 835 |
| `workspace.jl` — the state it iterates on, and the methods the contract asks for | 460 |
| `settings.jl` — the algorithm object and its defaults | 135 |
| `PureIPM.jl` — the module, and what it imports from the base | 88 |

It wrote its own method. It wrote no backend, no equilibration, no certificate test, no
polishing, and no `Problem`. Adding a third algorithm is the same shape.

## What you write

[Interfaces](@ref) states each list in full. In outline, four pieces:

**An algorithm object**, a subtype of [`QPAlgorithm`](@ref PureQPBase.QPAlgorithm), holding
the parameters only your iteration reads. PureIPM's is `InteriorPoint`, with the
regularization and the refinement count; operator splitting's holds `rho`, `sigma` and
`alpha`. Everything both algorithms read — tolerances, `max_iter`, `scaling`, `linsys` — is
[`Options`](@ref PureQPBase.Options) and is already there. `algorithm_defaults` is where you
say which of those options your method wants a different default for: the interior-point
method asks for `eps_abs = 1e-8` and `max_iter = 100` where operator splitting takes `1e-3`
and `4000`.

**A workspace**, a subtype of [`QPWorkspace`](@ref PureQPBase.QPWorkspace), holding the
iterates and the scratch your method needs. It carries the [`Problem`](@ref
PureQPBase.Problem), the [`SystemWeights`](@ref PureQPBase.SystemWeights) and the backend that
`setup_backend` chose, and those three are what the shared kernels read. Keep every field
concretely typed: the per-iteration guarantees depend on it.

**`setup_backend`**, which validates the data, builds the options and the problem, chooses a
backend and factorizes. PureIPM's is 90 lines and most of it is refusing what its method
cannot do — an operator that supplies only products, `linsys = :kronecker`, `linsys =
:lowrank` — each with a message naming the remedy rather than a `MethodError` from somewhere
inside.

**A selection tag**, a subtype of [`PureQPBase.SelectionFor`](@ref), and the three selection
methods whose answer depends on the algorithm: `select_backend`, the order your method tries
candidates in; `dense_rung`, where a pair that can be formed comes to rest; and
`indirect_rung`, what serves an operator that cannot. Every other selection method already
accepts any tag. One of the three missing gives an error naming the method you have to write.

## What you inherit

Everything else, and it is most of it.

**Every backend.** Dense and sparse, banded, block, Kronecker, diagonal-plus-low-rank and
matrix-free, with the machinery that picks one from the types of `P` and `A`. Your method
calls [`factorize!`](@ref PureQPBase.factorize!) and [`solve_system!`](@ref
PureQPBase.solve_system!) and does not care which one answered. Adding an algorithm gives it
all of them at once — see [How a backend is chosen](@ref).

**The problem.** [`Problem`](@ref PureQPBase.Problem) validates the data, runs Ruiz
equilibration, and keeps `P` and `A` by reference so every product reaches the matrix the
caller passed. `mul_A!`, `mul_At!` and `mul_P!` apply the scaling lazily.

**Termination and certificates.** [`check_termination`](@ref PureQPBase.check_termination) is
declared in the base and extended by each algorithm, so both report the same
[`Status`](@ref PureQPBase.Status) values with the same meanings. The infeasibility tests are
algorithm-independent: given any direction, [`is_primal_infeasible`](@ref
PureQPBase.is_primal_infeasible) and [`is_dual_infeasible`](@ref
PureQPBase.is_dual_infeasible) decide, whatever produced it.

**Polishing and derivatives.** [`polish!`](@ref PureQPBase.polish!) and
[`adjoint_derivative`](@ref PureQPBase.adjoint_derivative) work from the active set at the
iterate, so an algorithm that reaches a solution gets both. What you owe them is
`derivative_ready`: throw unless your multipliers are ones the active-set test can read.
PureIPM's refuses without polishing, because an interior-point method holds inactive rows at
the barrier parameter rather than at zero.

**The MathOptInterface wrapper.** One wrapper in the base carries the algorithm as a field, so
your package supplies an `Optimizer()` that names yours and inherits the rest.

## What the contracts buy you

The four abstract types are declared with
[TypeContracts.jl](https://github.com/el-oso/TypeContracts.jl), so a type that claims one and
does not implement it is rejected when the package precompiles, naming what is missing rather
than failing at a call site much later.

The method lists cannot say what the values *mean*, which is what
[`PureQPBase.conforms`](@ref) is for. It asserts the guarantees
[`Solution`](@ref PureQPBase.Solution) and [`Status`](@ref PureQPBase.Status) carry — that
stopping early is never reported as solved, that a run with no point fills `x` and `y` with
`NaN` rather than a plausible number, that the reported objective is the objective, that the
timings are real. Run it against your algorithm:

```julia
using Test, PureQPBase, MyQPAlgorithm
PureQPBase.conforms(MyAlgorithm(); eps = 1e-8, slow_iters = 5)
```

`eps` is a tolerance your method can reach on a small dense QP and `slow_iters` an iteration
budget too small to converge on one; both differ by algorithm, which is why they are given
rather than assumed.

## Where to read

`PureIPM/src/settings.jl` for the algorithm object and its defaults, `workspace.jl` for the
workspace and the four methods `setup` calls, and `ipm.jl` for a method that uses the shared
kernels throughout. `docs/design/modular-ipm.md` records why it is shaped as it is.
