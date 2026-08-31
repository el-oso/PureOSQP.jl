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
