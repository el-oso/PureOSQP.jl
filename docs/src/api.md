# API

```@docs
PureOSQP.solve
PureOSQP.setup
PureOSQP.solve!
PureOSQP.update!
PureOSQP.warm_start!
PureOSQP.cold_start!
PureOSQP.update_settings!
PureOSQP.update_rho!
PureOSQP.dimensions
PureOSQP.capabilities
PureOSQP.Optimizer
PureOSQP.adjoint_derivative
PureOSQP.forward_derivative
PureOSQP.Settings
PureOSQP.Solution
PureOSQP.Workspace
```

## Status values

```@docs
PureOSQP.Status
```

## Settings

`Settings` documents each field. This section shows how the fields correspond to the reference implementation.

| upstream | default | here | default | note |
|---|---|---|---|---|
| `device` | `0` | — | | GPU device selection; no counterpart |
| `osqp_linsys_solver_type` | direct | `linsys` | `:auto` | upstream picks a *library*, this picks a *formulation* — see [Algebra backends](@ref) |
| `allocate_solution` | `true` | — | | an embedded-allocation concern; no counterpart |
| `verbose` | `true` | `verbose` | `false` | a library that prints by default is the wrong default for a package |
| `profiler_level` | `0` | `profile_primdual` | `false` | one switch over the one measurement that needs a clock |
| `warm_starting` | `true` | `warm_starting` | `true` | |
| `scaling` | `10` | `scaling` | `10` | |
| `polishing` | `false` | `polish` | `false` | renamed only |
| `rho` | `0.1` | `rho` | `0.1` | |
| `rho_is_vec` | `true` | `rho_is_vec` | `true` | |
| `sigma` | `1e-6` | `sigma` | `1e-6` | |
| `alpha` | `1.6` | `alpha` | `1.6` | |
| `cg_max_iter` | `20` | `cg_max_iter` | `20` | |
| `cg_tol_reduction` | `10` | `cg_tol_reduction` | `10` | |
| `cg_tol_fraction` | `0.15` | `cg_tol_fraction` | `0.15` | |
| `cg_precond` | diagonal | — | | always diagonal here; nothing to select |
| `adaptive_rho` | `true` | `adaptive_rho` | `:iterations` | a mode, not a flag; `true` is accepted and means `:iterations` |
| `adaptive_rho_interval` | `50` | `adaptive_rho_interval` | `50` | the published table says `0`; the library ships `50` |
| `adaptive_rho_fraction` | `0.4` | `adaptive_rho_fraction` | `0.4` | **same name, different meaning** — see below |
| `adaptive_rho_tolerance` | `5` | `adaptive_rho_tolerance` | `5` | |
| `max_iter` | `4000` | `max_iter` | `4000` | |
| `eps_abs` | `1e-3` | `eps_abs` | `1e-3` | |
| `eps_rel` | `1e-3` | `eps_rel` | `1e-3` | |
| `eps_prim_inf` | `1e-4` | `eps_prim_inf` | `1e-4` | |
| `eps_dual_inf` | `1e-4` | `eps_dual_inf` | `1e-4` | |
| `scaled_termination` | `false` | `scaled_termination` | `false` | |
| `check_termination` | `25` | `check_termination` | `25` | |
| `check_dualgap` | `true` | `check_dualgap` | `true` | in the C header and the library's defaults, absent from the published table |
| `time_limit` | `1e10` | `time_limit` | `Inf` | the same "no limit", spelled as the thing it means |
| `delta` | `1e-6` | `delta` | `1e-6` | |
| `polish_refine_iter` | `3` | `polish_refine_iter` | `3` | |

Of the thirty-one settings: twenty-one match upstream, three have no counterpart, six are renamed or defaulted differently, and one is a trap.

**Same name, different meaning — the one to watch.** `adaptive_rho_fraction` means different things in upstream and here. Upstream it is a fraction of *setup time*. Here it is a fraction of the *previous KKT error*. Under `adaptive_rho = :kkt_error`, $\rho$ is retuned only when the relative KKT error falls to `adaptive_rho_fraction` of its previous value. Porting tuned values without reading this will cause quiet errors.

**Upstream only.** `device` and `allocate_solution` are for GPU and embedded systems. `cg_precond` always uses the diagonal preconditioner.

## Algebra backends

The reference implementation selects an *algebra* (`builtin` for CPU, `mkl` for Intel, `cuda` for NVIDIA GPUs).

In this package, the library is determined by what is loaded in the Julia session. A GPU array is a matrix type, and reaches the solver just like any other array.

The `linsys` setting selects the *formulation* of the linear system:

| | selects | values |
|---|---|---|
| upstream algebra | which library does the arithmetic | `builtin`, `mkl`, `cuda` |
| `linsys` here | which system is formed and how it is factored | `:auto`, `:dense`, `:kkt`, `:indirect` |

## Internals

```@docs
PureOSQP.equilibrate!
PureOSQP.mul_A!
PureOSQP.mul_At!
PureOSQP.mul_P!
PureOSQP.refactor!
PureOSQP.refactor_rho!
PureOSQP.factorize!
PureOSQP.solve_system!
PureOSQP.admm_step!
PureOSQP.set_rho_vec!
PureOSQP.adapt_rho!
PureOSQP.update_residuals!
PureOSQP.check_termination
PureOSQP.is_primal_infeasible
PureOSQP.is_dual_infeasible
PureOSQP.polish!
PureOSQP.has_solution
PureOSQP.PolishStatus
```

## Linear-system backends

```@docs
PureOSQP.LinearSystem
PureOSQP.choose_backend
PureOSQP.ReducedInverse
PureOSQP.ReducedCholesky
PureOSQP.DiagonalReduced
PureOSQP.TridiagonalReduced
PureOSQP.DiagonalLowRank
PureOSQP.BlockReduced
PureOSQP.KroneckerReduced
PureOSQP.FullKKT
PureOSQP.indirect_backend
```

## Structured inputs

```@docs
PureOSQP.RowCoupled
PureOSQP.coupling_rank
PureOSQP.BlockDiagonal
PureOSQP.KroneckerOperator
PureOSQP.factors
PureOSQP.is_scalar_multiple
PureOSQP.scalar_multiple
PureOSQP.nblocks
PureOSQP.rowrange
PureOSQP.colrange
PureOSQP.structural_rows
PureOSQP.is_convex
PureOSQP.is_symmetric
PureOSQP.is_materializable
PureOSQP.reduced_diagonal!
PureOSQP.reduced_rhs!
PureOSQP.ProductOperator
PureOSQP.unpreconditioned!
PureOSQP.probe_column!
PureOSQP.no_entries
```

## The selection ladder

```@docs
PureOSQP.select_backend
PureOSQP.density_gate_rung
PureOSQP.kkt_rung
PureOSQP.reduced_rung
PureOSQP.block_rung
PureOSQP.kronecker_rung
PureOSQP.lowrank_rung
PureOSQP.formed_rung
PureOSQP.dense_rung
PureOSQP.indirect_rung
```

## Backend introspection

```@docs
PureOSQP.backend_name
PureOSQP.backend_info
PureOSQP.factor_fill
PureOSQP.BackendInfo
```
