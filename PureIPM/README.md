# PureIPM.jl

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://el-oso.github.io/PureQP.jl/dev/)

A pure-Julia Mehrotra predictor–corrector interior-point method for convex quadratic programs,
`minimize ½ xᵀPx + qᵀx subject to l ≤ Ax ≤ u`. It takes its problem representation,
linear-system backends and equilibration from
[PureQPBase.jl](https://github.com/el-oso/PureQP.jl/tree/main/PureQPBase), which it
re-exports, so `using PureIPM` is enough.

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

**What it is good for.** It reaches `1e-8` in eight to twelve iterations, so it wins whenever
the answer is wanted to more than a few digits, or the problem is badly conditioned. Each
iteration factors a matrix, so a first-order method is the better choice for a warm-started
loop or a very large problem solved loosely.
[Choosing an algorithm](https://el-oso.github.io/PureQP.jl/dev/algorithms) compares them.

Measured against [Clarabel.jl](https://github.com/oxfordcontrol/Clarabel.jl) on random QPs at
the same tolerance, iteration counts match to within one and the gap widens with size: 2.3× at
`n = 50` to 4.3× at `n = 400` dense, 1.4× to 2.3× sparse. Full tables in
[Benchmarks](https://el-oso.github.io/PureQP.jl/dev/benchmarks).

**What it implements.** Primal and dual regularization with an increase on a failed
factorization, the row classification that lets equality, one-sided and free rows share one
path, Mehrotra's corrector and centering with a safeguard, infeasibility certificates, and
warm starts. Its terminal backend is the full quasi-definite KKT factorization rather than a
reduced one: an active row's weight reaches `1/δ_d`, and forming the reduced matrix at that
spread loses the accuracy the method needs — measured at `9e-9` against `9e-16` for the
augmented form. A MathOptInterface wrapper ships as an extension, so
`Model(PureIPM.Optimizer)` works from JuMP.

## License

**MIT.** This package is not a derivative of OSQP. The algorithm follows Mehrotra, *On the
implementation of a primal-dual interior point method*, SIAM Journal on Optimization
2(4):575–601, 1992, written from the paper. Clarabel.jl is the independent implementation
used to validate the answers, not a source for the code. See
[Attribution](https://el-oso.github.io/PureQP.jl/dev/attribution).
