# PureIPM.jl

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://el-oso.github.io/PureQP.jl/dev/)

A pure-Julia interior-point solver for convex quadratic programs:

```
minimize    ½ xᵀPx + qᵀx
subject to  l ≤ Ax ≤ u
```

It uses the Mehrotra predictor–corrector method. It takes the problem type, the linear-system
backends and the equilibration from
[PureQPBase.jl](https://github.com/el-oso/PureQP.jl/tree/main/PureQPBase) and re-exports it,
so `using PureIPM` is all you need.

```julia
using PureIPM

P = [4.0 1.0; 1.0 2.0]
q = [1.0, 1.0]
A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
l = [1.0, 0.0, 0.0]
u = [1.0, 0.7, 0.7]

sol = solve(P, q, A, l, u, InteriorPoint())
sol.status   # SOLVED
sol.iter     # single digits
```

## When to use it

It reaches `1e-8` in eight to twelve iterations. Use it when you want more than a few digits,
or when the problem is badly conditioned.

Each iteration factors a matrix. So for a warm-started loop, or a very large problem you solve
loosely, a first-order method costs less.
[Choosing an algorithm](https://el-oso.github.io/PureQP.jl/dev/algorithms) compares the two.

We measured it against [Clarabel.jl](https://github.com/oxfordcontrol/Clarabel.jl) on random
problems at the same tolerance. The iteration counts agree to within one. The speed gap grows
with size:

| | n = 50 | n = 400 |
|---|---|---|
| dense | 2.3× | 4.3× |
| sparse | 1.4× | 2.3× |

Full tables in [Benchmarks](https://el-oso.github.io/PureQP.jl/dev/benchmarks).

## What it does

- primal and dual regularization, raised when a factorization fails
- one path for equality, one-sided and free rows, through a row classification
- Mehrotra's corrector and centering, with a safeguard
- infeasibility certificates, and warm starts
- a MathOptInterface wrapper as an extension, so `Model(PureIPM.Optimizer)` works from JuMP

It factors the full KKT matrix rather than the smaller reduced one. Here is why. An active
row's weight climbs to `1/δ_d`. Building the reduced matrix at that spread rounds away the
small directions before any solve starts. We measured `9e-9` for the reduced form against
`9e-16` for the full one.

## License

**MIT.** This package is not derived from OSQP. The algorithm follows Mehrotra, *On the
implementation of a primal-dual interior point method*, SIAM Journal on Optimization
2(4):575–601, 1992, and we wrote it from the paper. Clarabel.jl checks our answers. It is not
a source for our code. See
[Attribution](https://el-oso.github.io/PureQP.jl/dev/attribution).
