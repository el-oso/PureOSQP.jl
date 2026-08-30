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
PureOSQP.FullKKT
```
