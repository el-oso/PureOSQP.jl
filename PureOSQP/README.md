# PureOSQP.jl

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://el-oso.github.io/PureQP.jl/dev/)
[![Build Status](https://github.com/el-oso/PureQP.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/el-oso/PureQP.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://coveralls.io/repos/github/el-oso/PureQP.jl/badge.svg?branch=main)](https://coveralls.io/github/el-oso/PureQP.jl?branch=main)

A pure-Julia operator-splitting solver for convex quadratic programs:

```
minimize    ½ xᵀPx + qᵀx
subject to  l ≤ Ax ≤ u
```

It runs [OSQP](https://osqp.org)'s ADMM iteration. It takes the problem type, the
linear-system backends and the equilibration from
[PureQPBase.jl](https://github.com/el-oso/PureQP.jl/tree/main/PureQPBase) and re-exports it,
so `using PureOSQP` is all you need.

```julia
using PureOSQP

P = [4.0 1.0; 1.0 2.0]
q = [1.0, 1.0]
A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
l = [1.0, 0.0, 0.0]
u = [1.0, 0.7, 0.7]

sol = solve(P, q, A, l, u, OperatorSplitting())
sol.status   # SOLVED
sol.x        # [0.3, 0.7]

# The algorithm is the sixth argument. Shared options are keywords.
sol = solve(P, q, A, l, u, OperatorSplitting(rho = 0.2); eps_abs = 1e-6)

# Keep the workspace and re-solve through `update!` without factoring again.
ws = setup(P, q, A, l, u, OperatorSplitting(); max_iter = 50)
sol = solve!(ws)
```

Give it any matrix type: dense, sparse, structured, lazy, anything that is an
`AbstractMatrix`, over any `Real` element type. It holds `P` and `A` by reference and never
changes them, so each product calls `mul!` on the matrix you passed.

## When to use it

Use it for repeated solves through `update!`, for warm starts, and for matrix-free operators.
In all three each iteration is a product, not a factorization.

It reaches a few digits fast and the last digits slowly. For one solve at a tight tolerance, an
interior-point method costs less.
[Choosing an algorithm](https://el-oso.github.io/PureQP.jl/dev/algorithms) compares the two.

## What it does

- modified Ruiz equilibration
- adaptive ρ, one value per row, split between equality and inequality rows
- primal and dual infeasibility certificates
- warm starts, and active-set polishing
- duality-gap termination, on by default

These are the parts of OSQP that change how it behaves, not just the ADMM recurrence.

Three more ship as extensions:

- a MathOptInterface wrapper, so `Model(PureOSQP.Optimizer)` works from JuMP. `MOI.Test`
  passes, apart from basis statuses and objective bounds, which an ADMM solver does not produce
- solution derivatives, by implicit differentiation of the KKT conditions. Dual numbers run
  them directly
- a matrix-free backend over
  [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl), which solves a
  product-only matrix with preconditioned conjugate gradients

## How it differs from the C library

By default it reduces the inner system to an `n×n` positive definite one. libosqp factors the
larger `(n+m)×(n+m)` system instead. We invert the reduced matrix once per factorization, so
each iteration costs one `symv`.

The matrices decide which form to use. A dense row in `A` sends the problem to a sparse
factorization of the full system, which is what libosqp does throughout. The
[Algorithm](https://el-oso.github.io/PureQP.jl/dev/algorithm) page gives the reasoning.

## Speed

These are the seven problem classes from OSQP's own benchmark suite, against libosqp 1.0.
Both solvers get the same sparse problem and stop after the same number of iterations. libosqp
always factors sparsely. PureOSQP picks: a dense Cholesky where `P` is dense, a sparse
factorization otherwise. It wins all seven.

| class | n | m | nnz(`A`) | iterations | backend | PureOSQP | libosqp 1.0 | vs libosqp |
|---|---|---|---|---|---|---|---|---|
| Random QP | 50 | 500 | 3 782 | 925 | `sparse_formed` | 4.94 ms | 9.47 ms | **1.92×** |
| Eq QP | 200 | 100 | 2 881 | 50 | `sparse_formed` | 2.37 ms | 4.13 ms | **1.74×** |
| SVM | 808 | 1600 | 2 549 | 300 | `ldlfactorizations` | 3.63 ms | 5.82 ms | **1.61×** |
| Portfolio | 505 | 506 | 2 294 | 450 | `ldl_kkt` | 3.52 ms | 4.31 ms | 1.22× |
| Control | 320 | 540 | 6 540 | 325 | `sparse_formed` | 6.18 ms | 7.44 ms | 1.20× |
| Lasso | 816 | 816 | 1 786 | 100 | `ldlfactorizations` | 1.51 ms | 1.65 ms | 1.09× |
| Huber | 1806 | 1800 | 3 526 | 125 | `ldlfactorizations` | 3.47 ms | 3.77 ms | 1.08× |

Measured on Julia 1.13.0, one BLAS thread, pinned to core 15. The objectives agree to `1e-13`
or better in six classes, and to `1e-9` in the seventh. Both solvers run with the duality-gap
test off, because each one computes the gap at a different point in the iteration. We time
libosqp on its setup and solve calls, from arrays we build first.

Each iteration is 1.17× to 5.44× faster. libosqp sets up faster in five of the seven. Full
tables, sparse and dense, in
[Benchmarks](https://el-oso.github.io/PureQP.jl/dev/benchmarks).

### Structure a sparsity pattern cannot express

libosqp reads `P` and `A` only as sparse matrices. The only structure it can use is where the
zeros are. PureOSQP reads those too, and structured types as well.

Running one problem three ways separates two things. PureOSQP sparse against libosqp compares
the two implementations. PureOSQP structured against PureOSQP sparse shows what declaring the
structure buys. All three use the same settings, take the same iterations, and agree on the
objective to `1e-9`.

| structure | nnz(`A`) | libosqp 1.0 (sparse) | PureOSQP (sparse) | PureOSQP (structured) | sparse vs libosqp | structured vs sparse |
|---|---|---|---|---|---|---|
| Kronecker `A₁ ⊗ A₂` | 100% | 120 ms | 54.6 ms | 1.06 ms | 2.19× | **51.6×** |
| tridiagonal | 0.2% | 0.58 ms | 0.47 ms | 0.19 ms | 1.26× | **2.49×** |
| low-rank coupling | 0.7% | 0.53 ms | 0.51 ms | 0.22 ms | 1.04× | **2.29×** |
| block-diagonal | 12.5% | 2.55 ms | 2.24 ms | 1.34 ms | 1.14× | **1.67×** |
| banded | 0.5% | 0.66 ms | 0.66 ms | 0.86 ms | 1.00× | 0.77× |

On sparse input PureOSQP is 1.0× to 2.2× faster than libosqp. Declaring the structure adds
another 1.7× to 2.5×, and 52× on the Kronecker problem. That `A` has no zeros, so a sparse
factorization has nothing to skip. Pass the same matrix as its two 20×20 factors and it is
solved through their eigenvectors instead.

The banded row is slower with the structure declared. The sparse factor of a banded matrix is
already banded, so declaring the band buys nothing at this size and bandwidth. A different
size or bandwidth can change that.

## Correctness

We check it against libosqp 0.6.2 and 1.x, the second through `ccall`. The first 25 iterates
match the C library to `1e-10`. The iteration count matches both on random problems. The
objectives agree to about `1e-15`. OSQP's own C test suite runs here too, and `setup` throws
if `P` is indefinite.

## Upstream

This is a Julia implementation of the OSQP algorithm. We wrote it against the OSQP paper and
the Apache-2.0 reference implementation, which we read for the details the paper leaves out.
It is a **derivative work**, not clean-room.

The original OSQP C library is by Bartolomeo Stellato, Goran Banjac and Paul Goulart
([osqp/osqp](https://github.com/osqp/osqp), Apache-2.0). The algorithm is theirs. Full credit
and citations in [Attribution](https://el-oso.github.io/PureQP.jl/dev/attribution).

## Development

We develop this package with help from Claude Code. Someone reviews generated code before it
lands. The design decisions, the measurements behind them, and the released behaviour are the
maintainer's own.

## License

**Apache-2.0**, matching upstream. This package derives from
[OSQP](https://github.com/osqp/osqp) and carries its license. See
[Attribution](https://el-oso.github.io/PureQP.jl/dev/attribution).
