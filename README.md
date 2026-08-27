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

`P` and `A` may be **any `AbstractMatrix`**, are never copied or modified, and there is no
sparse-matrix dependency anywhere. The numerics use `LinearAlgebra` alone; the only other
dependency is [TypeContracts.jl](https://github.com/el-oso/TypeContracts.jl), which
declares the linear-system backend interface and checks it at precompilation.

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

Not implemented: a MathOptInterface wrapper, a matrix-free/indirect inner solve, a sparse
linear-algebra backend, and the duality-gap termination check added in libosqp 1.x.

## How it differs from the reference implementation

The inner KKT system is eliminated to an `n×n` symmetric positive definite system and
factored with `cholesky!`, rather than factoring the `(n+m)×(n+m)` quasi-definite system
with a sparse LDLᵀ. Upstream avoids that because `AᵀA` destroys sparsity; with sparsity off
the table the objection does not apply, and the reduced form is faster in every dense
regime measured. Pass `linsys = :kkt` to force the full quasi-definite factorization, which
is slower but more accurate at moderate conditioning; the whole test corpus runs through
both backends.

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

Dense random QPs, `eps_abs = eps_rel = 1e-6`, single-threaded BLAS, versus libosqp 0.6.2:

| n | m | PureOSQP | libosqp 0.6.2 | speedup | iters | objective rel. Δ |
|---|---|---|---|---|---|---|
| 50 | 100 | 2.71 ms | 5.81 ms | 2.14× | 900 / 900 | 1.8e-15 |
| 200 | 400 | 38.4 ms | 130 ms | 3.39× | 1175 / 1175 | 4.4e-15 |
| 400 | 800 | 134 ms | 1747 ms | 13.0× | 625 / 625 | 3.9e-15 |
| 200 | 2000 | 299 ms | 2044 ms | 6.84× | 3025 / 3025 | 7.1e-15 |
| 100 | 50 | 1.63 ms | 1.17 ms | **0.72×** | 50 / 50 | 4.4e-15 |
| 10 | 20 | 0.153 ms | 0.081 ms | **0.53×** | 100 / 100 | 3.8e-15 |

These are **dense** problems, which is the reference implementation's worst case: a sparse
solver handed dense matrices, paying scalar sparse LDLᵀ where PureOSQP gets BLAS-3 dense
Cholesky. Read it as a storage-format comparison, not a solver-quality one — on sparse
problems the reference implementation is expected to win, and PureOSQP has no sparse
backend. It already loses here on small problems and when `m < n`.

Full tables, the libosqp 1.x agreement check and the linear-system backend comparison:
`bench/headtohead.jl`, `bench/headtohead_v1.jl` and `bench/kkt_backend.jl`, with every
sample saved under `bench/results/`.

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
