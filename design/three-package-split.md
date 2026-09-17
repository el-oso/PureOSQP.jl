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

- **The MathOptInterface precompile workload.** Solving one model while an extension precompiles
  caches the bridge code specialized on the optimizer, which is most of what `MOI.Test` spends
  its time compiling. It cannot run where it is now: `PureQPBase`'s extension has no algorithm to
  solve with, and a solver package's extension cannot see `PureQPBase`'s at precompile time,
  since a sibling extension is not one of its dependencies. `warm_up(factory)` is available for a
  caller to run.
- **Registration.** Nothing is registered. The core must be registered before the two solvers,
  and each package's entry needs the `subdir` it now lives in.

## 5. A structured augmented backend, proposed

Every structured backend solves the reduced matrix. The interior-point method cannot use the
reduced matrix. Those two facts leave a gap, and nothing fills it today.

### What the gap costs

A `Diagonal` `P` with a `RowCoupled` `A` has a backend under operator splitting,
`DiagonalLowRank`. Under the interior-point method the same pair reaches `FullKKT`, which
factors a dense `(n+m)` matrix. Handing the same numbers over as `SparseMatrixCSC` reaches
the sparse KKT factorization instead, at the same iteration count and the same answer.
Measured in `PureIPM/bench/ipm_lowrank_terminal.jl`, with the cost of building the sparse
pair charged to the sparse side:

| n | dense terminal | convert and solve sparsely | ratio |
|---|---|---|---|
| 120 | 2.92 ms | 0.22 ms | 13.4× |
| 240 | 15.0 ms | 0.45 ms | 33.3× |
| 480 | 87.2 ms | 0.87 ms | 99.9× |

So a caller who declares the structure is slower than a caller who does not, and the penalty
grows with size.

### Why the structured backends cannot serve it

The reduced matrix is `P̃ + σI + Ãᵀ diag(w) Ã`. An interior-point method drives an active
row's weight to `1/δ_d`, so the term that weight enters arrives orders of magnitude above the
rest of the matrix. The small directions are then rounded away as the matrix is built, before
any solve begins.

Measured on the low-rank families at their converged weights, against a reference in extended
precision: the reduced matrix solved through the Woodbury identity reaches `9e-9`, the same
matrix factored densely reaches `9e-8`, and the augmented factorization reaches `9e-16`.

Two explanations are ruled out by measurement. Conditioning is not the cause: the reduced
matrix has `cond = 1.1e10` and the augmented one `cond = 2.1e10`. The spread of the weights
is not the cause either: with weights spread smoothly over `1e16` and no row dominating, the
same low-rank backend stays at `3.6e-16`. What separates the cases is one active coupling row
whose weight reaches `1e8` against a diagonal core of `0.19`.

This rules out fixing the low-rank backend. Any backend that forms the reduced matrix loses
the same digits, however it represents or solves it.

### The proposal

Add backends that assemble the **augmented** matrix from a structured pair, rather than the
reduced one:

```
[ P̃ + σI    Ãᵀ  ]
[ Ã     -diag(w)⁻¹ ]
```

For `RowCoupled` the assembly is direct: `A` is `k` dense rows above one entry per remaining
row, so the pattern is known without inspecting a single value. The same holds for
`Diagonal`, `SymTridiagonal`, `BandedMatrix` and `BlockDiagonal`.

This is what the sparse route already does. `sparse_form` answers `:kkt`, the pair is factored
in augmented form, and the accuracy matches `FullKKT`. The sparse route wins because it is
augmented, not because it is sparse. A structured augmented backend would reach the same form
without requiring the caller to store the problem as `SparseMatrixCSC`.

The rung would sit between `reduced_rung` and `block_rung` in the interior-point ladder, and
would be absent from the operator-splitting one, whose weights stay inside `[1e-6, 1e6]` and
which is served well by the reduced form it already uses.

### What has to be measured before building it

- **Whether the structured assembly beats the sparse one.** The sparse route already reaches
  the augmented form at 0.87 ms where the dense terminal takes 87 ms. A structured backend has
  to beat the sparse route, not the dense one, or it earns nothing over converting.
- **Which structured types pay.** `RowCoupled` has a pattern that is trivial to state.
  `KroneckerOperator` does not: `A₁ ⊗ A₂` is dense, so its augmented form is dense too.
- **Where the crossover sits.** A structured augmented factor is larger than a structured
  reduced one, `(n+m)` against `n`. For operator splitting that trade is already settled in
  the reduced form's favor, and the interior-point answer may differ by problem size.
