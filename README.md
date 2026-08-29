# PureOSQP.jl

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://el-oso.github.io/PureOSQP.jl/dev/)
[![Build Status](https://github.com/el-oso/PureOSQP.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/el-oso/PureOSQP.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://coveralls.io/repos/github/el-oso/PureOSQP.jl/badge.svg?branch=main)](https://coveralls.io/github/el-oso/PureOSQP.jl?branch=main)

A pure-Julia implementation of the [OSQP](https://osqp.org) operator-splitting solver for
convex quadratic programs:

```
minimize    ½ xᵀPx + qᵀx
subject to  l ≤ Ax ≤ u
```

**The goal is to support every matrix representation** — dense, sparse, structured, lazy,
and anything else satisfying the `AbstractMatrix` interface — over any `Real` element type.
That is the design aim, not a caveat attached to one favoured format. `P` and `A` are held
by reference and never copied or modified, and every per-iteration product runs `mul!` on
the matrix you passed, so a `Diagonal`, a `SubArray` or a `SparseMatrixCSC` keeps its own
product rather than being flattened into a dense copy.

The numerics
use `LinearAlgebra` alone; the only other dependency is
[TypeContracts.jl](https://github.com/el-oso/TypeContracts.jl), which declares the
linear-system backend interface and checks it at precompilation. Sparse `P` and `A` are
accepted as they are: a `SparseArrays` **weak** dependency lets equilibration walk only
their stored entries, the per-iteration products use their own `mul!`, the reduced matrix is
accumulated over the stored entries rather than through a dense copy of `A`, and it is
factored by CHOLMOD when its factor stays sparse. Eliminating to the reduced system squares
`A`, so a pattern that fills in is routed elsewhere: to a sparse factorization of the full
quasi-definite system when one dense row is the cause, and to the dense factorization
otherwise.

The hot path is proven allocation-free and type-stable, and every public entry point
compiles under `juliac --trim`. See [Guarantees](https://el-oso.github.io/PureOSQP.jl/dev/guarantees).

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

Modified Ruiz equilibration, vector-valued adaptive ρ with an equality/inequality split,
primal and dual infeasibility certificates, warm starting, and active-set solution
polishing — that is, the parts of OSQP that determine how it actually behaves, not just the
ADMM recurrence.

Duality-gap termination is implemented and on by default, following libosqp 1.x.

A MathOptInterface wrapper ships as a package extension: `Model(PureOSQP.Optimizer)` works
from JuMP, and the whole of `MOI.Test` passes.

Solution derivatives ship too, by implicit differentiation of the KKT conditions rather
than through the ADMM loop, checked against `ForwardDiff` to `1e-13`. Dual numbers run the
solver directly, since the element type is `Real`.

A matrix-free backend ships as a second extension, over
[Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl): `linsys = :indirect` runs
preconditioned conjugate gradients on the reduced system and never forms it, so a matrix
that only supplies products is solvable.
Which backend is faster depends on the problem and the answer turns over: on dense QPs the
factorization wins by 15–49×, and on sparse ones it depends on size — the matrix-free
backend is 0.09× at `n = 200` and 1.90× at `n = 4000, m = 8000`, on 33× less memory, because
the direct backend still factors a dense `n×n` inverse however sparse the input.

The backend is chosen by representation, and for a sparse `A` there are two. One accumulates
the reduced matrix over the stored entries instead of writing `A` into an `m×n` buffer for a
dense product — at `n = 2000, m = 4000` and 0.25% density that takes a refactorization from
400 ms to 148 ms, a whole solve from 1192 ms to 777 ms, and the workspace from 93 MiB to
32 MiB. The other also *factors* sparsely, by CHOLMOD through SparseArrays, whenever the
Cholesky factor stays sparse enough to pay: on banded problems that is worth 20× at
`n = 1000` and 37× at `n = 2000` against the dense factorization, on identical iterates.
Both gates are measured rather than guessed — the second asks CHOLMOD what the fill actually
is — and both choices happen once, at `setup`.

A structured `P` is read through traversals that visit only the rows a band type can hold a
nonzero in, so a `Diagonal` or `SymTridiagonal` equilibrates about 2.2× faster than the same
matrix stored densely rather than, as before, slower.

See the [Roadmap](https://el-oso.github.io/PureOSQP.jl/dev/roadmap) for what remains.

## How it differs from the reference implementation

The inner KKT system is eliminated to an `n×n` symmetric positive definite system by
default, rather than factoring the `(n+m)×(n+m)` quasi-definite system as upstream does.
Upstream avoids the reduction because `AᵀA` destroys sparsity, and that is a real cost: the
reduced matrix is dense for a dense `A`, and one dense row of a sparse `A` is enough to make
it dense too. So the form is chosen from the matrices rather than fixed — a dense row routes
to a sparse factorization of the full system, which is upstream's formulation. Pass
`linsys = :kkt` to force the dense quasi-definite factorization, which is slower but more
accurate at moderate conditioning, or `linsys = :dense` to force the reduced one; the whole
test corpus runs through the backends.

That reduced matrix is then inverted, once per factorization, and each iteration solves by
a single `symv` instead of a Cholesky `ldiv!`. Both are `2n²` flops, but a triangular solve
produces its entries in sequence and a symmetric product does not, which is worth about 7×
at `n = 200` — the factor is small and cache-resident, so the sequential dependency, not
bandwidth, is what bounds it. Inverting is safe here specifically because the reduced
matrix carries the `σI` regularization that bounds its conditioning; the residuals of the
two forms agree to within a small factor.

Equilibration is stored as factors rather than applied to the matrices, so every
per-iteration product runs through the caller's own `P` and `A` and a structured or lazy
matrix keeps its fast product.

ρ adaptation fires on a fixed iteration interval rather than on wall-clock time, so
iteration counts are reproducible across machines.

## Correctness

Validated against **libosqp 0.6.2** (OSQP.jl 0.8.1) and **libosqp 1.x**
(`bench/headtohead_v1.jl`):

- the first 25 iterates match the C library to `1e-10`, across both linear-system backends
  and both scaled and unscaled space, with adaptive ρ disabled;
- on dense random QPs the **iteration count is identical** to both 0.6.2 and 1.x, and the
  objective agrees to about `1e-15`. Note the count is quantized by `check_termination`, so
  it shows the termination and ρ logic track upstream — it is not iterate-level agreement,
  which the transcription test covers separately;
- OSQP's own C test suite is ported: `basic_qp`, `basic_qp2`,
  `primal_dual_infeasibility` (all four variants), `unconstrained` and `non_cvx`, with
  upstream's data, expected solutions, per-test settings and tolerance;
- status agrees on primal- and dual-infeasible instances, and certificates are checked
  against their defining inequalities rather than against the C library's (non-unique)
  certificate vectors;
- every solved instance is judged by a referee that computes KKT residuals and the duality
  gap from the original problem data alone, so it cannot inherit a scaling bug from the
  solver. The referee is itself calibrated by a permanent test that runs the C library's
  own solution through it.

An indefinite `P` is refused at setup, as upstream does. Settings and problem data are
validated rather than silently accepted, and a run without a meaningful solution returns
`NaN` rather than a plausible-looking point.

## Benchmarks

Dense random QPs, `eps_abs = eps_rel = 1e-6`, single-threaded BLAS, against
[OSQP.jl](https://github.com/osqp/OSQP.jl) (libosqp 0.6.2). PureOSQP is timed on OpenBLAS
and again on [PureBLAS](https://github.com/el-oso/PureBLAS.jl):

| n | m | PureOSQP | + PureBLAS | OSQP | vs | vs (PureBLAS) | iterations |
|---|---|---|---|---|---|---|---|
| 25 | 50 | 0.460 ms | 0.415 ms | 1.39 ms | 3.02× | 3.34× | 675 |
| 50 | 100 | 1.37 ms | 1.29 ms | 5.04 ms | 3.69× | 3.90× | 900 |
| 200 | 400 | 16.3 ms | 13.5 ms | 128 ms | 7.81× | 9.44× | 1175 |
| 400 | 800 | 40.9 ms | 40.2 ms | 618 ms | **15.12×** | 15.39× | 625 |
| 200 | 2000 | 229 ms | 211 ms | 1302 ms | 5.69× | 6.16× | 3025 |
| 100 | 50 | 0.347 ms | 0.334 ms | 1.16 ms | 3.36× | 3.49× | 50 |

**Iteration counts are identical to OSQP in every case**, and objectives agree to about
`1e-15`. That is measured with `check_dualgap = false`, which is what libosqp 0.6.2 does:
PureOSQP defaults the duality-gap termination test *on*, following libosqp 1.x, and a
solver stopping on three criteria cannot be expected to match one stopping on two. The same holds against libosqp 1.x, checked separately through a subprocess
oracle since it has no Julia wrapper. Switching BLAS changes neither: PureBLAS gives
`|Δx| ≈ 1e-14` on identical iteration counts.

PureBLAS is the faster of the two on every case in this table.

This table is **dense** problems, which is the reference implementation's worst case: a
sparse solver handed dense matrices, paying scalar sparse LDLᵀ where PureOSQP gets BLAS-3
dense Cholesky. Read it as a storage-format comparison on one format, not a solver-quality
one, and not a statement about which format this solver is for.

### Sparse against libosqp, both sides sparse

`bench/sparse_headtohead.jl` removes that caveat: both solvers hold `SparseMatrixCSC` and
neither gets a dense copy of anything. Iteration counts are identical in every row, and the
benchmark refuses to report a row whose objectives disagree.

| family | n | m | nnz(A) | PureOSQP | OSQP | vs OSQP |
|---|---|---|---|---|---|---|
| banded | 200 | 400 | 2775 | 9.70 ms | 11.0 ms | 1.13× |
| banded | 500 | 1000 | 6975 | 19.4 ms | 22.6 ms | 1.17× |
| banded | 1000 | 2000 | 13975 | 73.7 ms | 87.2 ms | 1.18× |
| banded | 2000 | 4000 | 27975 | 58.0 ms | 69.4 ms | 1.20× |
| random | 200 | 400 | 753 | 2.19 ms | 2.31 ms | 1.06× |
| random | 200 | 400 | 4014 | 10.6 ms | 30.1 ms | 2.84× |
| random | 500 | 1000 | 4994 | 31.5 ms | 128 ms | 4.06× |
| random | 1000 | 2000 | 9926 | 112 ms | 412 ms | 3.69× |
| random | 2000 | 4000 | 20104 | 698 ms | 2416 ms | 3.46× |

Two families, because they give opposite answers and reporting one would be choosing the
answer. **Banded** is a model-predictive control horizon: the reduced matrix keeps the band,
PureOSQP factors it with CHOLMOD, and the margin is narrow — 1.13–1.20× — because this is
exactly what libosqp's sparse LDLᵀ of the full KKT is designed for, and it is a good design
there. **Random** sparsity suits neither a sparse factorization nor a dense one, and forming
the reduced matrix from stored entries then factoring it densely wins by 2.8–4.1×.

### The OSQP benchmark suite

Both families above are synthetic. `bench/osqp_suite.jl` runs the seven problem classes
OSQP's own benchmark suite uses, which carry the structure real problems have, and the
answer there is harder — **PureOSQP leads four of the seven**, on identical iteration counts:

| class | PureOSQP | OSQP | vs OSQP |
|---|---|---|---|
| Random QP | 3.74 ms | 6.69 ms | **1.79×** |
| Eq QP | 3.37 ms | 4.03 ms | **1.19×** |
| Portfolio | 4.28 ms | 3.48 ms | 0.81× |
| Lasso | 1.79 ms | 1.30 ms | 0.72× |
| SVM | 4.13 ms | 4.37 ms | **1.06×** |
| Huber | 4.20 ms | 2.90 ms | 0.69× |
| Control | 5.16 ms | 5.68 ms | **1.10×** |

Portfolio is what a dense row costs. Eliminating to the reduced system squares `A`, so one
dense row makes the result dense: its `A` is 0.9% dense and the reduced matrix would be 99%
dense. A sparse factorization of the full quasi-definite system keeps that row as one sparse
row, which is what `sparse_kkt` does and what took the class from 16.2 ms to 4.28 ms.

Lasso and Huber are the two still behind. Both pick the right backend on a genuinely sparse
reduced matrix; what is left is that the reduced form needs a product with `A` and one with
`Aᵀ` every iteration where the full-KKT form gets both from the factorization.


Two more benchmarks cover the rest:

- **sparse `A`, storage choice** — hand PureOSQP the sparse matrices rather than dense
  copies below ~10% density: a `SparseArrays` package extension walks the stored entries
  during equilibration, worth 2.3–2.5× overall;
- **a dense QP solver** — [DAQP](https://github.com/darnstrom/daqp), active-set, is 30×
  faster at `n = 10`; PureOSQP is 1.7× ahead by `n = 200, m = 400`. On one small dense QP,
  use DAQP. ADMM earns its place on sequences, through warm starts and cheap re-solves.

Full tables, plus structured-storage and matrix-type results and the libosqp 1.x agreement
check: `bench/headtohead.jl`, `bench/sparse_headtohead.jl`, `bench/solvers.jl`,
`bench/matrix_types.jl`, `bench/osqp_suite.jl`, `bench/headtohead_v1.jl`,
`bench/kkt_backend.jl` and `bench/indirect_backend.jl`, with every sample saved under
`bench/results/`.
## Relationship to upstream OSQP

A Julia implementation of the algorithm, written against the OSQP paper and against the
Apache-2.0 reference implementation, which was read directly for the details the paper
leaves out (equilibration, the ρ schedule, polishing, the certificate thresholds); OSQP's
own C unit tests are ported too. It is a **derivative work**, not an independent
reinvention, and **not** a clean-room implementation — clean room means reimplementing
without source access, which is a precaution for code you are not licensed to copy, and
Apache-2.0 already grants that right in exchange for attribution.

Full citations, the deliberate differences from upstream, and why the license stays
Apache-2.0 rather than becoming MIT: see
[Attribution](https://el-oso.github.io/PureOSQP.jl/dev/attribution).

## Reference

Stellato, Banjac, Goulart, Bemporad and Boyd, *OSQP: an operator splitting solver for
quadratic programs*, Mathematical Programming Computation 12(4):637–672, 2020.

## License

Apache-2.0, matching upstream OSQP.
