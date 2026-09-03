# PureOSQP.jl

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://el-oso.github.io/PureOSQP.jl/dev/)
[![Build Status](https://github.com/el-oso/PureOSQP.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/el-oso/PureOSQP.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://coveralls.io/repos/github/el-oso/PureOSQP.jl/badge.svg?branch=main)](https://coveralls.io/github/el-oso/PureOSQP.jl?branch=main)

A pure-Julia implementation of the [OSQP](https://osqp.org) operator-splitting solver for convex quadratic programs:

```
minimize    ½ xᵀPx + qᵀx
subject to  l ≤ Ax ≤ u
```

It handles every matrix representation — dense, sparse, structured, lazy, anything that satisfies `AbstractMatrix` — over any `Real` element type. `P` and `A` are kept by reference and never mutated, so each product calls `mul!` on the matrix you passed. The numerics are just `LinearAlgebra`; the only other dependency is [TypeContracts.jl](https://github.com/el-oso/TypeContracts.jl), which declares the linear-system backend. The hot path is allocation-free and type-stable, and every entry point compiles under `juliac --trim` — [Guarantees](https://el-oso.github.io/PureOSQP.jl/dev/guarantees).

```julia
using PureOSQP

P = [4.0 1.0; 1.0 2.0]
q = [1.0, 1.0]
A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
l = [1.0, 0.0, 0.0]
u = [1.0, 0.7, 0.7]

sol = solve(P, q, A, l, u)
sol.status   # SOLVED
sol.x        # [0.3, 0.7]
```

## What it implements

Modified Ruiz equilibration, vector-valued adaptive ρ with an equality/inequality split, primal and dual infeasibility certificates, warm starts, and active-set polishing — the parts of OSQP that change how it behaves, not just the ADMM recurrence. Duality-gap termination is on by default.

Also shipped as an extension: a MathOptInterface wrapper (`Model(PureOSQP.Optimizer)` from JuMP, all of `MOI.Test` passes), solution derivatives by implicit differentiation of the KKT conditions (dual numbers run the solver directly), and a matrix-free backend over [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl) that solves a product-only matrix with preconditioned conjugate gradients.

## How it differs

The inner system is reduced to an `n×n` symmetric positive-definite system by default, instead of factoring the `(n+m)×(n+m)` quasi-definite one upstream uses, and that reduced matrix is inverted once per factorization so each iteration is a single `symv`. The form is chosen from the matrices — a dense row of `A` routes to a sparse factorization of the full system, which is upstream's own. The [Algorithm](https://el-oso.github.io/PureOSQP.jl/dev/algorithm) page has the full rationale.

## Performance

OSQP is a sparse solver — it takes CSC input and factors sparsely, so it never sees denseness. Both solvers below are given the same problem in the same CSC form and stop after the same number of iterations, so the only thing that differs is the per-iteration solve. PureOSQP picks that solve per problem — a dense Cholesky where `P` is dense, a sparse factorization where it isn't — and on OSQP's own seven benchmark classes it leads all seven:

| class | n | m | nnz(`A`) | iterations | backend | PureOSQP | OSQP | vs OSQP |
|---|---|---|---|---|---|---|---|---|
| Eq QP | 200 | 100 | 2 881 | 50 | `cholesky` | 1.60 ms | 3.31 ms | **2.06×** |
| Random QP | 50 | 500 | 3 782 | 925 | `cholesky` | 3.58 ms | 6.66 ms | **1.86×** |
| SVM | 808 | 1600 | 2 549 | 300 | `ldlfactorizations` | 2.69 ms | 3.95 ms | 1.47× |
| Control | 320 | 540 | 6 540 | 325 | `sparse_formed` | 4.46 ms | 5.33 ms | 1.20× |
| Portfolio | 505 | 506 | 2 294 | 450 | `ldl_kkt` | 2.75 ms | 2.96 ms | 1.08× |
| Lasso | 816 | 816 | 1 786 | 100 | `ldlfactorizations` | 1.07 ms | 1.15 ms | 1.08× |
| Huber | 1806 | 1800 | 3 526 | 125 | `ldlfactorizations` | 2.52 ms | 2.62 ms | 1.04× |

The iteration count is the same on both sides of every row, and the objectives agree to between `1e-16` and `1e-13`. The backend column is what PureOSQP chose without being told: the two ~2× rows are where `P` is dense and it picks a dense Cholesky, and the rest reach one sparse factorization or another. Full tables, plus the synthetic-sparse and dense cases: [Benchmarks](https://el-oso.github.io/PureOSQP.jl/dev/benchmarks).

### Structure a sparsity pattern cannot express

Sparsity says where the zeros are. These problems say more — the blocks decouple, the band is a band, the coupling is rank 2, the matrix is a Kronecker product — and declaring it is worth something a sparse factorization cannot recover. Same problem both sides, OSQP given it as CSC, same settings, and every row runs the **same number of iterations** and agrees on the objective to `1e-9`:

| structure | n | nnz(`A`) | iterations | backend | PureOSQP | OSQP | vs OSQP |
|---|---|---|---|---|---|---|---|
| Kronecker `A₁ ⊗ A₂` | 400 | 100% | 250 | `kronecker` | 1.10 ms | 121 ms | **110×** |
| tridiagonal | 400 | 0.2% | 75 | `tridiagonal` | 0.20 ms | 0.69 ms | **3.48×** |
| low-rank coupling | 400 | 0.7% | 75 | `lowrank` | 0.21 ms | 0.65 ms | **3.02×** |
| block-diagonal | 200 | 12.5% | 175 | `block` | 1.26 ms | 2.61 ms | **2.07×** |
| banded | 400 | 0.5% | 75 | `banded` | 0.84 ms | 0.78 ms | 0.93× |

The Kronecker row is the point in one line: that `A` has **no zeros at all**, so sparsity offers nothing, while the same matrix declared as its two 20×20 factors is solved through their eigenbases. The banded row is the honest other end — at bandwidth 3 a sparse factor is already banded, so there is nothing left to win.

## Correctness

Validated against libosqp 0.6.2 and 1.x (the latter by `ccall`): the first 25 iterates match the C library to `1e-10`, the iteration count is identical to both on random QPs, and objectives agree to about `1e-15`. OSQP's own C test suite is ported, and an indefinite `P` is refused at setup.

## Upstream

A Julia implementation of the algorithm, written against the OSQP paper and the Apache-2.0 reference (read for the details the paper leaves out). A **derivative work**, not clean-room. The original OSQP C library is by Bartolomeo Stellato, Goran Banjac and Paul Goulart ([osqp/osqp](https://github.com/osqp/osqp), Apache-2.0), and the algorithm is theirs — full credit and citations in [Attribution](https://el-oso.github.io/PureOSQP.jl/dev/attribution).

## Development

PureOSQP is developed with the assistance of Claude Code. Generated code is reviewed before it lands, and the design decisions, the measurements behind them, and the released behaviour are the maintainer's own.

## License

Apache-2.0, matching upstream.
