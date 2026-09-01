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

Every value [`Solution`](@ref) can report, and which of them carry a usable point.

```@docs
PureOSQP.Status
```

## Settings

[`Settings`](@ref) documents each field. This section is the correspondence with the
reference implementation's settings, for anyone porting a tuned configuration across.

Defaults below are read from the reference library itself through `bench/osqp_v1.jl` rather
than from its documentation, which is stale on one of them: the published table gives
`adaptive_rho_interval` as `0`, where the library ships `50`.

**Same name, same default.** `rho` `0.1`, `sigma` `1e-6`, `alpha` `1.6`, `scaling` `10`,
`rho_is_vec` `true`, `max_iter` `4000`, `eps_abs` and `eps_rel` `1e-3`, `eps_prim_inf` and
`eps_dual_inf` `1e-4`, `scaled_termination` `false`, `check_termination` `25`,
`adaptive_rho_interval` `50`, `adaptive_rho_tolerance` `5`, `cg_max_iter` `20`,
`cg_tol_fraction` `0.15`, `cg_tol_reduction` `10`, `delta` `1e-6`, `polish_refine_iter` `3`,
`warm_starting` `true`, `check_dualgap` `true`.

`check_dualgap` is in the 1.x C header and in the library's defaults, but not on that settings
page; this package follows the library.

**Renamed, or defaulted differently.**

| upstream | here | why |
|---|---|---|
| `polishing` | `polish` | name only; both default off |
| `verbose` = `true` | `verbose` = `false` | a library that prints by default is the wrong default for a package |
| `time_limit` = `1e10` | `time_limit` = `Inf` | the same "no limit", spelled as the thing it means |
| `profiler_level` | `profile_primdual` | one switch over the one measurement that needs a clock |
| `osqp_linsys_solver_type` | `linsys` | upstream selects a *library*, this selects a *formulation* — see below |
| `adaptive_rho` = `true` | `adaptive_rho` = `:iterations` | a mode rather than a flag; `true` is accepted and means `:iterations` |

**Same name, different meaning — the one to watch.** `adaptive_rho_fraction` is `0.4` in both,
and means different things. Upstream it is a fraction of *setup time*, feeding the wall-clock
adaptation mode. Here it is a fraction of the *previous KKT error*: under
`adaptive_rho = :kkt_error`, `ρ` is retuned only once `rel_kkt_error` has fallen to
`adaptive_rho_fraction` of what it was at the last look. Porting a tuned value across without
reading this will not error — it will quietly do something else.

**Upstream only.** `device` and `allocate_solution` are GPU and embedded concerns;
`cg_precond` selects a preconditioner where this package always uses the diagonal one.


## Algebra backends

The reference implementation selects an *algebra*: `builtin` for CPU, `mkl` for Intel's
library, `cuda` for NVIDIA GPUs, each compiled as a separate library and chosen at build or
load time.

There is no counterpart here, because the axis does not exist. This package holds one
implementation and gets its dense kernels from whatever BLAS the session has loaded, so
`using MKL` is the whole of the MKL story — no separate build, no setting. A GPU array is a
matrix type rather than a backend, and reaches the solver the same way any other array type
does.

What `linsys` selects is a different axis: the *formulation* of the linear system, not the
library underneath it.

| | selects | values |
|---|---|---|
| upstream algebra | which compiled library does the arithmetic | `builtin`, `mkl`, `cuda` |
| `linsys` here | which system is formed and how it is factored | `:auto`, `:dense`, `:kkt`, `:indirect` |

Both axes exist here, but only one is a setting: the library is whatever is loaded, and the
formulation is `linsys`. See [Choosing the linear system](@ref) for what each value does, and
[Guarantees](@ref) for which of them a GPU array can reach.

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

