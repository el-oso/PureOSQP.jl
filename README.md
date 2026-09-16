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

# The algorithm is the optional sixth argument; shared options are keywords.
sol = solve(P, q, A, l, u, OperatorSplitting(rho = 0.2); eps_abs = 1e-6)
sol = solve(P, q, A, l, u, InteriorPoint(); eps_abs = 1e-9)

ws = setup(P, q, A, l, u, InteriorPoint(); max_iter = 50)
sol = solve!(ws)
```

`OperatorSplitting()`, the default, is OSQP's ADMM iteration; `InteriorPoint()` is a Mehrotra
interior-point method whose defaults aim at `1e-8` accuracy. [Choosing an algorithm](https://el-oso.github.io/PureOSQP.jl/dev/#Choosing-an-algorithm) compares them.

## What it implements

Modified Ruiz equilibration, vector-valued adaptive ρ with an equality/inequality split, primal and dual infeasibility certificates, warm starts, and active-set polishing — the parts of OSQP that change how it behaves, not just the ADMM recurrence. Duality-gap termination is on by default.

Also shipped as extensions: a MathOptInterface wrapper, so `Model(PureOSQP.Optimizer)` works from JuMP (and `MOI.Test` passes, except for basis statuses and objective bounds, which an ADMM solver does not produce); solution derivatives, by implicit differentiation of the KKT conditions, which dual numbers run directly; and a matrix-free backend over [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl) that solves a product-only matrix with preconditioned conjugate gradients.

## How it differs

By default it reduces the inner system to an `n×n` symmetric positive-definite one, rather than factoring the `(n+m)×(n+m)` quasi-definite system upstream uses. That reduced matrix is inverted once per factorization, so each iteration is a single `symv`. Which form to use is chosen from the matrices — a dense row of `A` routes to a sparse factorization of the full system, which is upstream's own. The [Algorithm](https://el-oso.github.io/PureOSQP.jl/dev/algorithm) page has the rationale.

## Performance

These are the seven problem classes from OSQP's own benchmark suite, compared with libosqp 1.0, OSQP's C library. Both solvers get the same sparse (CSC) problem and stop after the same number of iterations. libosqp always factors sparsely. PureOSQP picks the linear-system solve for each problem: a dense Cholesky where `P` is dense, a sparse factorization otherwise. It is faster in all seven classes:

| class | n | m | nnz(`A`) | iterations | backend | PureOSQP | libosqp 1.0 | vs libosqp |
|---|---|---|---|---|---|---|---|---|
| Random QP | 50 | 500 | 3 782 | 925 | `sparse_formed` | 4.94 ms | 9.47 ms | **1.92×** |
| Eq QP | 200 | 100 | 2 881 | 50 | `sparse_formed` | 2.37 ms | 4.13 ms | **1.74×** |
| SVM | 808 | 1600 | 2 549 | 300 | `ldlfactorizations` | 3.63 ms | 5.82 ms | **1.61×** |
| Portfolio | 505 | 506 | 2 294 | 450 | `ldl_kkt` | 3.52 ms | 4.31 ms | 1.22× |
| Control | 320 | 540 | 6 540 | 325 | `sparse_formed` | 6.18 ms | 7.44 ms | 1.20× |
| Lasso | 816 | 816 | 1 786 | 100 | `ldlfactorizations` | 1.51 ms | 1.65 ms | 1.09× |
| Huber | 1806 | 1800 | 3 526 | 125 | `ldlfactorizations` | 3.47 ms | 3.77 ms | 1.08× |

Measured on Julia 1.13.0, single-threaded BLAS, pinned to core 15. The objectives agree to `1e-13` or better in six classes and to `1e-9` in the seventh. Both solvers run with the duality-gap test (`check_dualgap`) off, because each computes the gap at a different point in the iteration. libosqp is timed on its setup and solve calls, from CSC arrays built beforehand. The backend column is what PureOSQP chose on its own: Random QP and Eq QP accumulate the reduced matrix from the stored entries rather than forming it with a dense product, which is why both now read `sparse_formed`. Each iteration is 1.17× to 5.44× faster; libosqp's setup is faster in five classes. Full tables, including sparse and dense families: [Benchmarks](https://el-oso.github.io/PureOSQP.jl/dev/benchmarks).

### Structure a sparsity pattern cannot express

libosqp accepts `P` and `A` only as sparse matrices, so the only structure it can use is where the zeros are. PureOSQP accepts the same sparse matrices and also structured types. Running one problem three ways separates two effects: PureOSQP sparse against libosqp compares the implementations, and PureOSQP structured against PureOSQP sparse shows what passing the structure is worth.

All three use the same settings, take the same number of iterations, and agree on the objective to `1e-9`.

| structure | nnz(`A`) | libosqp 1.0 (sparse) | PureOSQP (sparse) | PureOSQP (structured) | sparse vs libosqp | structured vs sparse |
|---|---|---|---|---|---|---|
| Kronecker `A₁ ⊗ A₂` | 100% | 120 ms | 54.6 ms | 1.06 ms | 2.19× | **51.6×** |
| tridiagonal | 0.2% | 0.58 ms | 0.47 ms | 0.19 ms | 1.26× | **2.49×** |
| low-rank coupling | 0.7% | 0.53 ms | 0.51 ms | 0.22 ms | 1.04× | **2.29×** |
| block-diagonal | 12.5% | 2.55 ms | 2.24 ms | 1.34 ms | 1.14× | **1.67×** |
| banded | 0.5% | 0.66 ms | 0.66 ms | 0.86 ms | 1.00× | 0.77× |

With sparse input, PureOSQP is 1.0× to 2.2× faster than libosqp. Passing the structure makes it another 1.7× to 2.5× faster, and 52× faster on the Kronecker problem. That `A` has no zeros, so a sparse factorization has nothing to skip, while the same matrix passed as its two 20×20 factors is solved through their eigenvectors.

The banded row is slower with the structure passed. The sparse factor of a banded matrix is already banded, so declaring the band gains nothing at this size and bandwidth. The result can change with the bandwidth and the problem size.

## Correctness

Validated against libosqp 0.6.2 and 1.x (the latter by `ccall`): the first 25 iterates match the C library to `1e-10`, the iteration count is identical to both on random QPs, and objectives agree to about `1e-15`. OSQP's own C test suite is ported, and an indefinite `P` is refused at setup.

## Upstream

A Julia implementation of the algorithm, written against the OSQP paper and the Apache-2.0 reference (read for the details the paper leaves out). A **derivative work**, not clean-room. The original OSQP C library is by Bartolomeo Stellato, Goran Banjac and Paul Goulart ([osqp/osqp](https://github.com/osqp/osqp), Apache-2.0), and the algorithm is theirs — full credit and citations in [Attribution](https://el-oso.github.io/PureOSQP.jl/dev/attribution).

## Development

PureOSQP is developed with the assistance of Claude Code. Generated code is reviewed before it lands, and the design decisions, the measurements behind them, and the released behaviour are the maintainer's own.

## License

Apache-2.0, matching upstream.
