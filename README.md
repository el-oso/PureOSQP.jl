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
algorithms on the same `solve`, with the algorithm as the optional sixth argument:

```julia
using PureOSQP, PureIPM

P = [4.0 1.0; 1.0 2.0]
q = [1.0, 1.0]
A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
l = [1.0, 0.0, 0.0]
u = [1.0, 0.7, 0.7]

solve(P, q, A, l, u)                    # operator splitting, the default
solve(P, q, A, l, u, InteriorPoint())   # the interior-point method
```

**The licenses differ because the provenance does.** PureOSQP is a derivative work of
[OSQP](https://github.com/osqp/osqp) — written against its paper and reference C
implementation, with its C unit tests ported — so it carries its parent's Apache-2.0 terms.
PureQPBase and PureIPM are not derivatives of it, and are MIT: they were written from
published papers, Ruiz 2001 for the equilibration, Banjac et al. 2019 for the infeasibility
certificates, Mehrotra 1992 for the interior-point method. Depending on PureQPBase places
your package under neither license. Each package directory carries the license governing it,
[`LICENSE`](LICENSE) at the root is the map, and
[Attribution](https://el-oso.github.io/PureQP.jl/dev/attribution) has the full credit and
citations.

These packages are developed with the assistance of Claude Code. Generated code is reviewed
before it lands, and the design decisions, the measurements behind them, and the released
behaviour are the maintainer's own.
