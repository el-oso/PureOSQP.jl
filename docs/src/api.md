# API

```@docs
PureOSQP.solve
PureOSQP.setup
PureOSQP.solve!
PureOSQP.update!
PureOSQP.warm_start!
PureOSQP.cold_start!
PureOSQP.Settings
PureOSQP.Solution
PureOSQP.Workspace
```

## Internals

```@docs
PureOSQP.scale!
PureOSQP.mul_A!
PureOSQP.mul_At!
PureOSQP.mul_P!
PureOSQP.refactor!
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
```

## Linear-system backends

```@docs
PureOSQP.LinearSystem
PureOSQP.ReducedCholesky
PureOSQP.FullKKT
PureOSQP.refactor!
```
