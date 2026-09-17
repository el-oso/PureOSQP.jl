# PureQP.jl

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://el-oso.github.io/PureQP.jl/dev/)
[![Build Status](https://github.com/el-oso/PureQP.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/el-oso/PureQP.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://coveralls.io/repos/github/el-oso/PureQP.jl/badge.svg?branch=main)](https://coveralls.io/github/el-oso/PureQP.jl?branch=main)

Pure-Julia solvers for convex quadratic programs, `minimize ½ xᵀPx + qᵀx subject to
l ≤ Ax ≤ u`. Three packages live here, each registered on its own and each with its own
README. They handle every matrix representation — dense, sparse, structured, lazy, anything
satisfying `AbstractMatrix` — over any `Real` element type, allocation-free on the hot path
and compiling under `juliac --trim`.

| package | what it is | license |
|---|---|---|
| [**PureQPBase**](PureQPBase) | the problem, the linear-system backends and their selection, equilibration, the contracts. No algorithm. | MIT |
| [**PureOSQP**](PureOSQP) | operator splitting — [OSQP](https://osqp.org)'s ADMM iteration | Apache-2.0 |
| [**PureIPM**](PureIPM) | a Mehrotra predictor–corrector interior-point method | MIT |

Either solver re-exports the base, so one `using` is enough; loading both puts both
algorithms on the same `solve`, named as the sixth argument:

```julia
using PureOSQP, PureIPM

P = [4.0 1.0; 1.0 2.0]
q = [1.0, 1.0]
A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
l = [1.0, 0.0, 0.0]
u = [1.0, 0.7, 0.7]

solve(P, q, A, l, u, OperatorSplitting())   # ADMM: warm starts, loose tolerances
solve(P, q, A, l, u, InteriorPoint())       # interior point: few iterations, tight answers
```

Each solver is measured against the established implementation of its own method, at the same
tolerance and with the iteration counts checked to make sure the comparison is of solvers
rather than of stopping rules:

| | against | problems | faster by | iterations |
|---|---|---|---|---|
| **PureOSQP** | libosqp 1.0 (C) | OSQP's suite, 7 classes | 1.08× – 1.92× | identical |
| **PureIPM** | [Clarabel.jl](https://github.com/oxfordcontrol/Clarabel.jl) | random QPs, `n` = 50 … 400 | 1.4× – 4.3× | within one |

The interior-point margin grows with size — 2.3× to 4.3× dense from `n` = 50 to 400, 1.4× to
2.3× sparse. Passing a structured `A` rather than its sparsity pattern is worth another 1.7×
to 52× on top. Full tables, and the ill-conditioned and matrix-free families, in
[Benchmarks](https://el-oso.github.io/PureQP.jl/dev/benchmarks).

The three packages are under two licenses, because PureOSQP is a derivative of OSQP and the
other two are not:
[Attribution](https://el-oso.github.io/PureQP.jl/dev/attribution) has the reasoning, the
credit and the citations.

These packages are developed with the assistance of Claude Code. Generated code is reviewed
before it lands, and the design decisions, the measurements behind them, and the released
behaviour are the maintainer's own.
