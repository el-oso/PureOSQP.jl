# Guarantees

This page provides evidence for three claims about PureOSQP and defines their limits.

| claim | meaning | limit |
|---|---|---|
| **no allocation on the hot path** | No memory allocation occurs during iterations, avoiding garbage collection. | Guaranteed for `Vector`-backed workspaces. Not claimed for custom operators. |
| **type stability** | Performance is consistent because there is no hidden dynamic dispatch. | Checked for all reachable backends. |
| **compiles under `juliac --trim`** | Can be built into a standalone binary with no Julia runtime. | Entry points are enumerated; sparse operands require a named backend. |

All claims are machine-checked.

## The `LinearSystem` contract

The factorization backend is a declared interface, enforced at precompilation via [TypeContracts.jl](https://github.com/el-oso/TypeContracts.jl).

```@eval
using PureOSQP, TypeContracts, Markdown
Markdown.parse(replace(contract_md_string(PureOSQP.LinearSystem), r"\A# [^\n]*\n+" => ""))
```

`PureOSQP.refactor_weights!` is not part of the contract, as its default is to rebuild the factorization.

@verify ensures that backends implement the required methods and return the correct types at precompilation.

To add a backend, subtype `LinearSystem`, implement the mandatory methods, and use `@verify`. The backend is fixed when the workspace is built, so every per-iteration call dispatches statically.

## No allocation, no type instability

`bench/strictmode_audit.jl` uses [StrictMode.jl](https://github.com/el-oso/StrictMode.jl) to verify that the following functions are type-stable and allocation-free:

| function | guarantees |
|---|---|
| `admm_step!` | type-stable, allocation-free |
| `update_residuals!` | type-stable, allocation-free |
| `solve_system!` | type-stable, allocation-free |
| `check_termination` | type-stable |
| `factorize!` | type-stable |
| `refactor_weights!` | type-stable |
| `solve!` | type-stable |

The interior-point method's own hot kernels carry the same guarantees, checked over every
backend its selection ladder reaches (`FullKKT`, the sparse KKT family, the structured
reduced backends, and `:indirect` with a caller preconditioner):

| function | guarantees |
|---|---|
| `ipm_step!` | type-stable, allocation-free |
| `ipm_residuals!` | type-stable, allocation-free |
| `solve_multiplier!` | type-stable, allocation-free |
| `check_termination` | type-stable |
| `factorize!` | type-stable |
| `refactor_weights!` | type-stable |
| `solve!` | type-stable |

Two notes:
- Sparse arithmetic backends may perform allocations within their own libraries. This applies
  to `factorize!`, `refactor_weights!` and `solve!` on the sparse KKT family under both
  algorithms; `solve_multiplier!` there is this package's own code and keeps the full
  guarantee regardless.
- The matrix-free backend is checked by measurement; its static analysis shows potential
  branches that are never taken. Under the interior-point method, the `try`/`catch` that turns
  a non-positive-definite preconditioner into a missed solve (rather than an exception) sits
  in a helper outside the audited kernel, so it does not affect what is measured.

An operator provided by the user is only as efficient as its own `mul!` method.

## `--trim` compatibility

`juliac --trim` requires every function call to be resolved statically.

**Every backend is checked.** `src/PureOSQP.jl` uses `@verify LinearSystem subtypes = true trim_compat = true`. This check runs during precompilation and for every subtype.

**Entry points are enumerated.** `test/trim_tests.jl` validates concrete calls for every public path:
- `solve` with various settings and sparse/dense operands.
- Every structured representation (diagonal, tridiagonal, banded, low-rank, block-diagonal, Kronecker, and products).
- `setup` $\to$ `solve!` $\to$ `update!` $\to$ `solve!`, and similar sequences with `warm_start!`.
- `update_settings!`, `update_rho!`, and `cold_start!`.
- The derivatives.
- `InteriorPoint()` on `FullKKT` (its default and its named KKT backend), the sparse KKT
  family, a diagonal pair, `:indirect` with a caller preconditioner on both a matrix pair and
  a `ProductOperator` pair, and a `setup` $\to$ `solve!` $\to$ `update!` $\to$ `solve!`
  sequence and the derivatives, each under `InteriorPoint()`.

For sparse problems, we require the backend to be explicitly named (e.of `:kkt`, `:dense`, or `:indirect`) to ensure compatibility.
