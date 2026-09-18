# Adding an algorithm

PureQPBase has no algorithm of its own. It gives you the parts around one: the problem, the
linear-system backends, equilibration, the termination tests, the certificates, polishing, the
derivatives, and `setup`, `solve` and `solve!`.

PureIPM shows what that saves. We added it to a package that already had operator splitting.
Here is what we had to write:

| file | what it holds | lines |
|---|---|---|
| `ipm.jl` | the Mehrotra method | 835 |
| `workspace.jl` | the state it works on, and the methods the contract asks for | 460 |
| `settings.jl` | the algorithm object and its defaults | 135 |
| `PureIPM.jl` | the module, and what it takes from the base | 88 |

We wrote the method. We wrote no backend, no equilibration, no certificate test, no polishing,
and no problem type. Your algorithm takes the same shape.

## You write four things

[Interfaces](@ref) lists every method. Here is the outline.

**1. An algorithm object.** Make it a subtype of [`QPAlgorithm`](@ref PureQPBase.QPAlgorithm).
Put in it only the parameters your method reads. PureIPM puts the regularization and the
refinement count in `InteriorPoint`. Operator splitting puts `rho`, `sigma` and `alpha` in
its own.

Settings more than one algorithm reads are already there. `max_iter`, the tolerances, `scaling`
and `linsys` all live in [`Options`](@ref PureQPBase.Options). Use `algorithm_defaults` to
change a default your method needs. PureIPM asks for `eps_abs = 1e-8` and `max_iter = 100`.
Operator splitting takes `1e-3` and `4000`. A method that cannot honour a shared option should
throw when it is set rather than ignore it, as PureDAQP does for `linsys`, `scaling` and
`polishing`.

**2. A workspace.** Make it a subtype of [`QPWorkspace`](@ref PureQPBase.QPWorkspace). Put the
iterates and your scratch space in it. It also carries three things the shared code reads: the
problem, the [`SystemWeights`](@ref PureQPBase.SystemWeights), and the backend. Give every
field a concrete type. The speed guarantees depend on it.

**3. A `setup_backend` method.** It checks the data, builds the options and the problem, picks
a backend, and factors it. PureIPM's runs to 90 lines. Most of those lines refuse work its
method cannot do: an operator with no entries, `linsys = :kronecker`, `linsys = :lowrank`.
Each refusal names the way out. None of them lets a `MethodError` escape from deeper down.

**4. A selection tag.** Make it a subtype of [`PureQPBase.SelectionFor`](@ref). Then write the
three methods whose answer changes with the algorithm:

- `select_backend` — the order your method tries candidates in
- `dense_rung` — where a pair it can build comes to rest
- `indirect_rung` — what serves an operator it cannot build

Every other selection method takes any tag already. Miss one of the three and you get an error
that names it.

## You get the rest

**Every backend.** Dense, sparse, banded, block, Kronecker, diagonal plus low rank, and
matrix-free. The code that picks one from the types of `P` and `A` comes with them. Your
method calls [`factorize!`](@ref PureQPBase.factorize!) and
[`solve_system!`](@ref PureQPBase.solve_system!). It never asks which backend answered. One
algorithm gets all of them — see [How a backend is chosen](@ref).

**The problem, in two pieces.** [`QPData`](@ref PureQPBase.QPData) is what the caller gave:
`P` and `A` by reference, so each product uses the matrix the caller passed, and the linear
term and bounds clamped. [`Problem`](@ref PureQPBase.Problem) is built around one of those and
adds the Ruiz equilibration and the scratch the scaled products and the backends need;
`mul_A!`, `mul_At!` and `mul_P!` apply the scaling as they go.

Ask for the one your method works on. An algorithm with no backend that multiplies `P` and
`A` itself holds the `QPData` and allocates none of the rest — reaching for `prob.D` or
`prob.work_n` is then an error naming the field, rather than an array that happens to be
there.

**Termination and certificates.** The base declares
[`check_termination`](@ref PureQPBase.check_termination) and each algorithm extends it. Every
algorithm reports the same [`Status`](@ref PureQPBase.Status) values, and they mean the same
thing. The infeasibility tests do not care which method made the direction they read:
[`is_primal_infeasible`](@ref PureQPBase.is_primal_infeasible) and
[`is_dual_infeasible`](@ref PureQPBase.is_dual_infeasible) work on any of them.

**Polishing and derivatives.** [`polish!`](@ref PureQPBase.polish!) and
[`adjoint_derivative`](@ref PureQPBase.adjoint_derivative) read the active set at the point
you reached. Reach a solution and you get both. You owe them one method: `derivative_ready`
throws unless your multipliers are ones the active-set test can read. PureIPM refuses without
polishing, because it holds inactive rows at the barrier parameter, not at zero.

**The MathOptInterface wrapper.** The base has one wrapper. It carries the algorithm as a
field. Your package supplies an `Optimizer()` that names yours.

## What the contracts give you

[TypeContracts.jl](https://github.com/el-oso/TypeContracts.jl) declares the four abstract
types. Claim one and leave a method out, and the package refuses to precompile. The error
names what is missing. You do not find out later, at some call site.

A method list says which methods exist. It cannot say what their values mean. That is what
[`PureQPBase.conforms`](@ref) checks. It tests the promises
[`Solution`](@ref PureQPBase.Solution) and [`Status`](@ref PureQPBase.Status) make:

- a run that stops early never reports `SOLVED`
- a run with no answer fills `x` and `y` with `NaN`, not with a number that looks usable
- the objective it reports is the objective
- the times it reports are real

Run it on your algorithm:

```julia
using Test, PureQPBase, MyQPAlgorithm
PureQPBase.conforms(MyAlgorithm(); eps = 1e-8, slow_iters = 5)
```

Give it `eps`, a tolerance your method can reach on a small dense problem. Give it
`slow_iters`, an iteration count too small to converge on one. These differ by algorithm, so
you pass them rather than let the check guess.

## Where to read

- `PureIPM/src/settings.jl` — the algorithm object and its defaults
- `PureIPM/src/workspace.jl` — the workspace, and the methods `setup` calls
- `PureIPM/src/ipm.jl` — a method that uses the shared parts throughout
- `docs/design/modular-ipm.md` — why we shaped it this way
