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

Also shipped as extensions: a MathOptInterface wrapper, so `Model(PureOSQP.Optimizer)` works from JuMP (and all of `MOI.Test` passes); solution derivatives, by implicit differentiation of the KKT conditions, which dual numbers run directly; and a matrix-free backend over [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl) that solves a product-only matrix with preconditioned conjugate gradients.

## How it differs

By default it reduces the inner system to an `n×n` symmetric positive-definite one, rather than factoring the `(n+m)×(n+m)` quasi-definite system upstream uses. That reduced matrix is inverted once per factorization, so each iteration is a single `symv`. Which form to use is chosen from the matrices — a dense row of `A` routes to a sparse factorization of the full system, which is upstream's own. The [Algorithm](https://el-oso.github.io/PureOSQP.jl/dev/algorithm) page has the rationale.

## Performance

OSQP is a sparse solver: it takes CSC and factors sparsely, so it never sees denseness. Both solvers below get the same CSC problem and stop after the same number of iterations, so only the per-iteration solve differs. PureOSQP picks that solve per problem — a dense Cholesky where `P` is dense, a sparse factorization otherwise — and it leads all seven of OSQP's benchmark classes:

| class | n | m | nnz(`A`) | iterations | backend | PureOSQP | OSQP | vs OSQP |
|---|---|---|---|---|---|---|---|---|
| Eq QP | 200 | 100 | 2 881 | 50 | `cholesky` | 1.60 ms | 3.31 ms | **2.06×** |
| Random QP | 50 | 500 | 3 782 | 925 | `cholesky` | 3.58 ms | 6.66 ms | **1.86×** |
| SVM | 808 | 1600 | 2 549 | 300 | `ldlfactorizations` | 2.69 ms | 3.95 ms | 1.47× |
| Control | 320 | 540 | 6 540 | 325 | `sparse_formed` | 4.46 ms | 5.33 ms | 1.20× |
| Portfolio | 505 | 506 | 2 294 | 450 | `ldl_kkt` | 2.75 ms | 2.96 ms | 1.08× |
| Lasso | 816 | 816 | 1 786 | 100 | `ldlfactorizations` | 1.07 ms | 1.15 ms | 1.08× |
| Huber | 1806 | 1800 | 3 526 | 125 | `ldlfactorizations` | 2.52 ms | 2.62 ms | 1.04× |

The objectives agree to between `1e-16` and `1e-13`. The backend column is what PureOSQP chose on its own: the two ~2× rows have a dense `P`, so it picks a dense Cholesky; the rest get a sparse factorization. Full tables, plus the synthetic-sparse and dense cases: [Benchmarks](https://el-oso.github.io/PureOSQP.jl/dev/benchmarks).

### Structure a sparsity pattern cannot express

OSQP takes CSC and nothing else — `osqp_setup` accepts `P` and `A` only as sparse matrices — so what it can exploit is *where the zeros are*. PureOSQP accepts the same CSC, and also the structured type. Running one problem all three ways separates the two effects: **sparse against sparse is the implementation, structured against sparse is what declaring the structure buys.**

Same problem, same settings; every row runs the **same number of iterations** on all three and agrees on the objective to `1e-9`.

| structure | nnz(`A`) | OSQP (sparse) | PureOSQP (sparse) | PureOSQP (structured) | sparse vs OSQP | structured vs sparse |
|---|---|---|---|---|---|---|
| Kronecker `A₁ ⊗ A₂` | 100% | 120 ms | 55.0 ms | 1.09 ms | 2.19× | **50.3×** |
| tridiagonal | 0.2% | 0.69 ms | 0.49 ms | 0.20 ms | 1.41× | **2.45×** |
| low-rank coupling | 0.7% | 0.64 ms | 0.50 ms | 0.21 ms | 1.28× | **2.34×** |
| block-diagonal | 12.5% | 2.62 ms | 2.41 ms | 1.27 ms | 1.09× | **1.90×** |
| banded | 0.5% | 0.77 ms | 0.65 ms | 0.84 ms | 1.18× | 0.78× |

Read the last two columns. The sparse path is 1.09–2.19× ahead of OSQP on the same input, which is implementation. Declaring the structure is worth 1.9–2.5× on top of that, and 50× on the Kronecker problem, whose `A` has **no zeros at all** — sparsity has nothing to work with there, while the same matrix given as its two 20×20 factors is solved through their eigenbases.

The banded row is the counterexample, and it's kept: at this size and bandwidth the banded backend is *slower than PureOSQP's own sparse path* — CHOLMOD's factor of a banded matrix is already banded, so the declaration added nothing here. There may be cases where declaring the band is faster; the outcome moves with the band width and problem size.

## Correctness

Validated against libosqp 0.6.2 and 1.x (the latter by `ccall`): the first 25 iterates match the C library to `1e-10`, the iteration count is identical to both on random QPs, and objectives agree to about `1e-15`. OSQP's own C test suite is ported, and an indefinite `P` is refused at setup.

## Upstream

A Julia implementation of the algorithm, written against the OSQP paper and the Apache-2.0 reference (read for the details the paper leaves out). A **derivative work**, not clean-room. The original OSQP C library is by Bartolomeo Stellato, Goran Banjac and Paul Goulart ([osqp/osqp](https://github.com/osqp/osqp), Apache-2.0), and the algorithm is theirs — full credit and citations in [Attribution](https://el-oso.github.io/PureOSQP.jl/dev/attribution).

## Development

PureOSQP is developed with the assistance of Claude Code. Generated code is reviewed before it lands, and the design decisions, the measurements behind them, and the released behaviour are the maintainer's own.

## License

Apache-2.0, matching upstream.
