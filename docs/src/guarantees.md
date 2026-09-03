# Guarantees

This page provides evidence for three claims about PureOSQP and defines their limits.

| claim | meaning | limit |
|---|---|---|
| **no allocation on the hot path** | No memory allocation occurs during iterations, avoiding garbage collection. | Guaranteed for `Vector`-backed workspaces. Not claimed for GPU arrays or custom operators. |
| **type stability** | Performance is consistent because there is no hidden dynamic dispatch. | Checked for all reachable backends. |
| **compiles under `juliac --trim`** | Can be built into a standalone binary with no Julia runtime. | Entry points are enumerated; sparse operands require a named backend. |

All claims are machine-checked.

## The `LinearSystem` contract

The factorization backend is a declared interface, enforced at precompilation via [TypeContracts.jl](https://github.com/el-oso/TypeContracts.jl).

```@eval
using PureOSQP, TypeContracts, Markdown
Markdown.parse(replace(contract_md_string(PureOSQP.LinearSystem), r"\A# [^\n]*\n+" => ""))
```

`PureOSQP.refactor_rho!` is not part of the contract, as its default is to rebuild the factorization.

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
| `refactor_rho!` | type-stable |
| `solve!` | type-stable |

Two notes:
- Sparse arithmetic backends may perform allocations within their own libraries.
- The matrix-free backend is checked by measurement; its static analysis shows potential branches that are never taken.

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

For sparse problems, we require the backend to be explicitly named (e.of `:kkt`, `:dense`, or `:indirect`) to ensure compatibility.

## What the guarantees cover on GPU arrays

GPU support is limited to matrix-free operations. `linsys = :indirect` is the only supported backend for GPUs.

- **Type stability** and **`--trim` compatibility** are maintained for GPU arrays.
- **Allocation-free** claims do not apply to GPU workspaces, as kernel launches and GPU communications involve allocations.

The elementwise operations are implemented to work efficiently on both CPU and GPU.
