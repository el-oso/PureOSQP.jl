# Three packages in one repository

What the split is, what belongs where, and the constraints that shaped it. `design/modular-ipm.md`
sketched this layout as step 3; this records what was built.

## 1. The packages

| package | holds | depends on |
|---|---|---|
| `PureQPBase` | `Problem`, `SystemWeights`, `Options`, `Solution`, `Status`, every linear-system backend and the selection ladder, equilibration, termination, the polishing and derivative kernels, the `QPAlgorithm` and `QPWorkspace` contracts, and the generic `setup`, `solve`, `solve!` and `build_workspace` | LinearAlgebra, TypeContracts |
| `PureOSQP` | `OperatorSplitting`, its workspace, adaptive `ρ`, the accelerator, and the five-argument `setup`, `solve` and `recommend_linsys` | + PureQPBase |
| `PureIPM` | `InteriorPoint`, its workspace, regularization, certificates and polishing | + PureQPBase |

`PureQPBase` defines no algorithm and solves nothing on its own: `setup` and `solve` take one as
their sixth positional argument. Each solver package re-exports it, so `using` either is enough,
and `using` both puts both algorithms on the same `solve`.

The matrix-support extensions — SparseArrays, BandedMatrices, LDLFactorizations, Krylov,
LinearMaps, SciMLOperators, GPUArraysCore — and the ChainRulesCore rules belong to `PureQPBase`:
they extend backends and kernels, not an algorithm.

## 2. Decisions

- **No umbrella package.** A caller who wants both loads both; MathOptInterface is what a
  modeling layer talks to. A fourth package would exist only to re-export two.
- **The core owns `solve`.** Both packages add methods to one generic function rather than
  defining their own, so `solve(P, q, A, l, u, alg)` means the same thing whichever is loaded.
- **The five-argument forms live in `PureOSQP`.** They are what makes `OperatorSplitting` the
  default, and a default is a property of a solver package, not of the core.
- **One MathOptInterface wrapper, in `PureQPBase`.** It carries the algorithm as a field, and
  each solver package's extension supplies the `Optimizer()` name and the algorithm it runs.
  `PureOSQP.Optimizer` and `PureIPM.Optimizer` each accept the shared options and only their own
  algorithm's parameters; there is no attribute that switches algorithms.
- **Shared generic functions are declared in the core.** `check_termination` and `polish!` are
  declared in `PureQPBase` and extended by each algorithm. Without the declaration each package
  defines a function of its own name, and `PureOSQP.check_termination` is then a different
  function from `PureIPM.check_termination`.
- **The verbose printing helpers are not shared.** Each algorithm keeps its own `VERBOSE_RULE`
  and `print_padded` (§3).

## 3. Two inference constraints the split exposed

Both were found by the `--trim` gate, and both would otherwise have been silent.

**The backend name must become a `Val` in the frame that sees the caller's keyword.** `setup` and
`solve` each call `check_linsys(linsys)`, which validates the name and returns `Val(linsys)`, and
pass that down by position. Deciding it any deeper leaves the name to reach `setup_backend` by
constant propagation, whose budget a solve with four keywords exhausts; the `Val` is then
constructed at run time and the call is unresolved. `setup_backend` takes the options, the
preconditioner and the accelerator by position for the same reason: a keyword call carries a
`NamedTuple` whose names inference loses track of once several keywords survive to it.

**A helper shared by two algorithms is inferred over both callers at once.** `print_padded(v,
width, digits)` reaches `string(round(v; sigdigits))`. With one caller `v` is concrete; with two
it widens, and the `string` becomes a call `--trim` rejects. Each package keeps its own copy.

## 4. What is not done

- **Tests.** All 218 items live in `PureOSQP/test`, and the ones that exercise `InteriorPoint`
  make that suite depend on PureIPM. Splitting them per package needs a home for the shared
  helpers (`helpers.jl` and `bench/suite_problems.jl`) and for the items that test both
  algorithms against each other.
- **The MathOptInterface precompile workload.** Solving one model while an extension precompiles
  caches the bridge code specialized on the optimizer, which is most of what `MOI.Test` spends
  its time compiling. It cannot run where it is now: `PureQPBase`'s extension has no algorithm to
  solve with, and a solver package's extension cannot see `PureQPBase`'s at precompile time,
  since a sibling extension is not one of its dependencies. `warm_up(factory)` is available for a
  caller to run.
- **Registration.** Nothing is registered. The core must be registered before the two solvers,
  and each package's entry needs the `subdir` it now lives in.
