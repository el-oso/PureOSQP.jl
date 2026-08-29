# Benchmarks

Reproduce with `julia --project=bench bench/headtohead.jl`. Every sample is written to
`bench/results/headtohead.json`.

## Against OSQP

Dense random QPs, `eps_abs = eps_rel = 1e-6`, single-threaded BLAS, adaptive ρ pinned to a
50-iteration interval on both sides so the comparison is deterministic. **OSQP** here is
[OSQP.jl](https://github.com/osqp/OSQP.jl), which wraps libosqp 0.6.2 — the reference
implementation as a Julia user gets it. PureOSQP is timed on OpenBLAS and again on
[PureBLAS](https://github.com/el-oso/PureBLAS.jl) (commit `ea79919`).

| n | m | PureOSQP | PureOSQP + PureBLAS | OSQP | vs | vs (PureBLAS) | iterations | objective rel. Δ |
|---|---|---|---|---|---|---|---|---|
| 10 | 20 | 0.038 ms | 0.035 ms | 0.064 ms | 1.70× | 1.82× | 100 | 3.3e-15 |
| 25 | 50 | 0.460 ms | 0.415 ms | 1.39 ms | 3.02× | 3.34× | 675 | 1.3e-15 |
| 50 | 100 | 1.37 ms | 1.29 ms | 5.04 ms | 3.69× | 3.90× | 900 | 1.3e-15 |
| 100 | 200 | 3.52 ms | 2.99 ms | 20.5 ms | 5.82× | 6.87× | 850 | 1.3e-15 |
| 200 | 400 | 16.3 ms | 13.5 ms | 128 ms | 7.81× | 9.44× | 1175 | 4.1e-15 |
| 400 | 800 | 40.9 ms | 40.2 ms | 618 ms | **15.12×** | 15.39× | 625 | 4.5e-15 |
| 100 | 50 | 0.347 ms | 0.334 ms | 1.16 ms | 3.36× | 3.49× | 50 | 7.5e-16 |
| 200 | 100 | 1.18 ms | 1.15 ms | 7.29 ms | 6.16× | 6.37× | 50 | 1.1e-14 |
| 100 | 1000 | 120 ms | 112 ms | 679 ms | 5.67× | 6.05× | 7475 | 1.7e-11 |
| 200 | 2000 | 229 ms | 211 ms | 1302 ms | 5.69× | 6.16× | 3025 | 7.9e-15 |

**The iteration count is identical to OSQP in every case**, measured with
`check_dualgap = false` to match libosqp 0.6.2, which has no duality-gap termination test.
PureOSQP defaults that test on, following libosqp 1.x. That is the result worth caring
about: the equilibration, the ρ schedule and the termination tests reproduce the reference
exactly, not approximately. The objective agrees to about `1e-15`, and switching BLAS
changes neither — PureBLAS gives bit-comparable answers (`|Δx| ≈ 1e-14`) on the same
iteration counts.


**On `check_dualgap = false`.** PureOSQP defaults the duality-gap termination test *on*,
following libosqp 1.x; 0.6.2 has no such test, and a solver stopping on three criteria
cannot be expected to match one stopping on two. Pinning it off is what makes the comparison
a comparison. It is not a formality: across a sweep of badly scaled objectives the test
changed the iteration count in 114 of 600 comparable runs. Everything here that compares
counts against 0.6.2 pins it — the oracle tests, the ported C-suite cases and every
benchmark script.

The same agreement holds against **libosqp 1.x**, which has no Julia wrapper and so is
checked separately by `bench/headtohead_v1.jl` through a subprocess oracle: identical
iteration counts on all seven of its cases, objectives to `1e-13`. It is left out of the
timing table because timing a subprocess measures the subprocess.

PureOSQP is ahead of OSQP on every case in this table.

## How to read this

**The timings are not a claim that this solver is better than OSQP.** This table is dense
problems, which is the reference implementation's worst case: it is a *sparse* solver being
handed dense matrices, so it pays sparse-format overhead and scalar sparse LDLᵀ where
PureOSQP gets BLAS-3 dense Cholesky. The right reading is a *storage-format* comparison on
one format, not a solver-quality one — and not a statement about which format this solver
is for, since it aims to serve all of them. [Sparse A](@ref "Sparse A") and
[Matrix types](@ref "Matrix types") are the same solver on the others.

**What the matching iteration counts do and do not show.** They mean the algorithm's
control logic tracks the reference. But the count is *quantized*: with
`check_termination = 25`, matching counts say both crossed the threshold within the same
25-iteration window, not that the iterates agree. That the iterates agree is a separate
result — the transcription test, matching to `1e-10` over the first 25 iterations across
both linear-system backends and both scaled and unscaled space. And the returned `y` is a
third thing again, since the objective does not contain `y`; that one comes from the
referee.

**Run-to-run spread.** Consecutive runs of this benchmark agree within about 5% on every
row.

## Sequential re-solves

`P` and `A` fixed, `q`, `l` and `u` changing every step — the receding-horizon loop OSQP is
most used for. 20 solves per case. Reproduce with
`julia --project=bench bench/update_bench.jl`, or `--project=bench/pureblas` to include the
PureBLAS column.

| n | m | `update!` | fresh `setup` each step | saved | libosqp 1.x | vs | `update!` + PureBLAS | vs | factorizations |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 20 | 0.96 ms | 1.40 ms | 1.47× | 1.84 ms | 1.92× | 0.82 ms | 1.17× | 8 |
| 25 | 50 | 3.07 ms | 6.43 ms | 2.09× | 9.95 ms | 3.24× | 2.70 ms | 1.14× | 14 |
| 50 | 100 | 15.8 ms | 23.9 ms | 1.51× | 58.7 ms | 3.72× | 15.0 ms | 1.05× | 7 |
| 100 | 200 | 89.3 ms | 81.9 ms | **0.92×** | 487 ms | 5.45× | 82.0 ms | 1.09× | 3 |
| 200 | 400 | 343 ms | 359 ms | 1.05× | 2387 ms | 6.96× | 290 ms | 1.18× | 1 |

"Factorizations" counts the whole 20-step sequence: one from `setup`, plus one per step
whose constraint classification changed.

**`update!` is worth much less than it used to be, and that is a consequence of making
`setup` faster.** Its whole benefit is skipping setup; once setup dropped sevenfold (see
[Sparse A](@ref "Sparse A")) there was far less left to skip. At `n = 100` it is now
level with rebuilding — the two runs also differ in warm starting, so they take different
numbers of iterations and the comparison is no longer purely setup-versus-setup. It pays
where solves are short and setup is a large share, and it remains the right call when you
want warm starting; at the larger sizes it is close to a wash.

**This loop is the case where the BLAS choice is least predictable from the single-solve
numbers.** A re-solve sequence refactorizes far more often per iteration than one long
solve does, and `potrf` and `potri` are where the two libraries differ most and in opposite
directions — PureBLAS is 1.98× faster on the first and 0.67× on the second, so at `n = 200`
it is about 1.15× *behind* on the combined factorization step while still ahead per
iteration. It nonetheless comes out ahead on every row here, by 1.05–1.18×, so at these
factorization counts the per-iteration advantage still dominates. The margin does not track
the factorization count cleanly, which is what you would expect when refactorization cost
grows as `n³` and the loop's cost grows as iterations times `n²`.

The factorization count is not always 1: equilibration rescales the bounds, so a row whose
scaled gap `ũ - l̃` falls under `RHO_TOL` is reclassified as an equality and its `ρ`
changes. That is upstream's rule applied to the scaled bounds exactly as upstream applies
it, and it means `update!` is not unconditionally factorization-free.

## Linear-system backend

`bench/kkt_backend.jl` reproduces the cost and accuracy comparison between the reduced
Cholesky and the full-KKT Bunch-Kaufman factorization described under
[Algorithm](@ref "The linear system"), including the near-parallel-row family that
equilibration cannot fix.

## On PureBLAS instead of OpenBLAS

[PureBLAS.jl](https://github.com/el-oso/PureBLAS.jl) is a pure-Julia BLAS/LAPACK. Its
`activate()` overlays per-symbol forwards onto libblastrampoline, so PureOSQP runs on it
**with no code changes** — the same `mul!`, `cholesky!` and `symv` calls are rerouted in
process. Reproduce with `bench/pureblas_backend.jl` (setup instructions in its header).

Note `BLAS.get_config()` cannot show this: OpenBLAS stays loaded and the forwards sit on
top of it. `PureBLAS.is_active()` is the check, and the benchmark asserts it at every
measurement — otherwise a rerouting that silently failed would look like a clean tie.

Measured against PureBLAS at commit `ea79919`, which `bench/results/pureblas_backend.json`
records alongside the timings.

| n | m | OpenBLAS | PureBLAS | ratio | iterations | \|Δx\| |
|---|---|---|---|---|---|---|
| 25 | 50 | 0.198 ms | 0.186 ms | 1.07× | 225 | 4.2e-15 |
| 50 | 100 | 0.611 ms | 0.582 ms | 1.05× | 350 | 4.3e-15 |
| 100 | 200 | 5.94 ms | 5.20 ms | 1.14× | 1500 | 8.7e-15 |
| 200 | 400 | 9.44 ms | 8.12 ms | 1.16× | 650 | 1.0e-14 |
| 100 | 50 | 0.346 ms | 0.333 ms | 1.04× | 50 | 8.4e-15 |

**Correctness is exact** — identical iteration counts and solutions agreeing to `1e-14`, so
PureBLAS is a faithful drop-in — **and it is faster than OpenBLAS on every case**:

| operation (n=200, m=400) | OpenBLAS | PureBLAS | ratio | runs |
|---|---|---|---|---|
| `gemv A*x` | 3.38 µs | 3.07 µs | 1.10× | every iteration |
| `gemv Aᵀy` (transposed) | 3.91 µs | 2.53 µs | **1.55×** | every iteration |
| `gemv At*y` (materialized transpose) | 3.79 µs | 3.00 µs | 1.26× | — |
| `syrk WᵀW` | 313 µs | 233 µs | **1.34×** | on a ρ update |
| `potrf` | 114 µs | 57.7 µs | **1.98×** | on a ρ update |
| `potri` | 216 µs | 322 µs | **0.67×** | on a ρ update |
| `symv R⁻¹b` | 1.58 µs | 1.35 µs | 1.17× | every iteration |
| `trsv F\b` | 11.6 µs | 12.1 µs | 0.96× | — |

PureOSQP's inner loop is Level-2-bound: it does a handful of `gemv` and one `symv` per
iteration and touches `syrk`/`potrf`/`potri` only when ρ changes, so the whole-solve ratio
tracks `gemv`. `potri` is the one operation where PureBLAS is behind, and inverting the
factor put it on the ρ-update path — but at one call per update against hundreds of
iterations, it does not move the whole-solve number.

`trsv` is listed for reference only: it is what `symv` replaced. The two are the same
`2n²` flops on either BLAS, and the 7× between them is the sequential dependency, not the
library.

### On the threading

The two sides are not symmetric. OpenBLAS is pinned with `BLAS.set_num_threads(1)`;
PureBLAS is plain Julia, so it is bounded by `Threads.nthreads()` — 12 in these runs —
which `BLAS.set_num_threads` does not affect. Giving OpenBLAS more threads does not change
the picture at these sizes: at n=200, m=400 it took 25.4 ms on 1 thread, 25.8 ms on 4 and
26.3 ms on 8.



## Against other solvers

The tables above compare PureOSQP with OSQP. These are three other solvers on the same
dense QPs, one per algorithm
family. Reproduce with `julia --project=bench bench/solvers.jl`.

| n | m | PureOSQP | OSQP | DAQP | Clarabel |
|---|---|---|---|---|---|
| 10 | 20 | 0.091 ms | 0.155 ms | **0.003 ms** | 0.129 ms |
| 25 | 50 | 0.197 ms | 0.538 ms | **0.035 ms** | 0.655 ms |
| 50 | 100 | 0.611 ms | 2.15 ms | **0.208 ms** | 2.90 ms |
| 100 | 200 | 5.81 ms | 34.3 ms | **1.87 ms** | 17.2 ms |
| 200 | 400 | **9.49 ms** | 76.7 ms | 16.1 ms | 112 ms |
| 100 | 50 | **0.348 ms** | 1.15 ms | 0.359 ms | 6.06 ms |

DAQP is a dense active-set solver, and on small problems it is the right tool by a wide
margin — 30× faster at `n = 10`. The gap closes with size, and by `n = 200, m = 400`
PureOSQP is 1.7× ahead. That shape is what the algorithms predict: an active-set method
walks a handful of expensive iterations and terminates exactly, which is unbeatable when
the active set is small, while ADMM's cost per iteration grows more slowly. If you have one
small dense QP and no sequence, reach for DAQP.

What ADMM buys instead is the behaviour the rest of this page measures: iteration counts
that do not depend on the active set, warm starts, cheap re-solves, and a factorization
reused across hundreds of iterations. Clarabel is an interior-point method and lands
between the two, ahead at the smallest size and behind from `n = 50` up.

Solutions agree to `1e-4` across all four, the expected spread at `eps_abs = eps_rel =
1e-6`: DAQP and Clarabel terminate on exact optimality conditions while ADMM stops at a
residual tolerance.

## Matrix types

PureOSQP holds the caller's `P` and `A` and applies equilibration lazily, so every
per-iteration product runs `mul!` on whatever was passed in. Each row below is **one
problem solved twice** — same numbers, once as a plain `Matrix` and once in a structured
type — so the iteration counts match exactly and the difference is purely the cost of the
product. Reproduce with `julia --project=bench bench/matrix_types.jl`.

| problem | as `Matrix` | structured | speedup | iterations |
|---|---|---|---|---|
| dense `P` (vs `Symmetric`) | 35.2 ms | 36.6 ms | 0.96× | 4375 |
| diagonal `P` (vs `Diagonal`) | 5.25 ms | 4.95 ms | 1.06× | 550 |
| tridiagonal `P` (vs `SymTridiagonal`) | 21.4 ms | 20.9 ms | 1.02× | 2550 |
| `A` as a `SubArray` | 8.04 ms | 9.49 ms | 0.85× | 900 |

**Structured storage helps, but only a little, and it is worth being precise about why.**
The column traversals visit only the rows a band type can hold a nonzero in, which makes
equilibration 2.3× faster on a `Diagonal` `P` (453 µs to 200 µs at `n = 150`) and 2.2× on a
`SymTridiagonal`. End to end that is 1.06× and 1.02×, because equilibration is a modest share
of a run: each iteration also does two products with `A`, which at `m = 2n` is four times the
work of the `P` product, and `A` is dense in every row above.

Two earlier readings of this table were wrong and are worth recording. The diagonal row read
0.80× — a genuine pessimisation — because the traversals walked every structural zero and
each read was a branch returning zero where a dense array does a straight load. And the
tridiagonal row measured `Symmetric` wrapped around a *dense* array, which is not a band type
at all: it measured the wrapper's indexing overhead rather than the structure. It now uses
`SymTridiagonal`.

The `SubArray` row is unrelated to any of that: a view's indexing is simply dearer than its
parent's, and nothing in the traversals changes that.

So the honest claim stays narrow: the caller's matrices are never copied or mutated, any
`AbstractMatrix` works, a cheaper `mul!` is used when one exists, and a band type is no
longer read as though it were dense. On a dense `A` none of this is a speed feature.

## Sparse A

`A` and `P` genuinely sparse, so OSQP's sparse LDLᵀ is playing to its strength. PureOSQP is
measured both ways: handed dense copies, and handed the sparse matrices directly, where a
package extension walks `nzrange` instead of indexing.

| n | m | density | densified | sparse input | OSQP | best vs OSQP |
|---|---|---|---|---|---|---|
| 200 | 400 | 1% | 8.19 ms | 3.56 ms | 4.83 ms | 1.36× |
| 200 | 400 | 5% | 13.2 ms | 8.53 ms | 24.6 ms | 2.88× |
| 200 | 400 | 20% | 13.2 ms | 18.5 ms | 37.9 ms | 2.88× |
| 400 | 800 | 1% | 42.3 ms | 16.7 ms | 39.2 ms | 2.34× |
| 400 | 800 | 5% | 59.4 ms | 38.3 ms | 110 ms | 2.87× |
| 400 | 800 | 20% | 135 ms | 181 ms | 380 ms | 2.82× |

Iteration counts are identical everywhere — the storage changes the speed, never the path,
and the benchmark asserts that rather than assuming it.

**PureOSQP is ahead on every row.** The narrowest margin is the case OSQP is built for — a
small, very sparse problem where the KKT matrix is nearly empty — and even there it is
1.36×.

**Which storage to pass depends on density.** Below about 10%, hand PureOSQP the sparse
matrices: equilibration walks only the stored entries, worth 2.3–2.5× overall. Above it,
densify: sparse `mul!` loses to dense BLAS-2 once there is enough work per row, and at 20%
the sparse path is about 1.4× *slower* than the dense one. The solver does not choose for
you, because the right choice depends on a density it would have to compute.


## Sparse against libosqp, both sides sparse

`bench/sparse_headtohead.jl`. Both solvers are handed `SparseMatrixCSC`; neither gets a
dense copy of anything. Iteration counts are identical in every row, and the benchmark
refuses to report a row whose objectives disagree by more than `1e-6`.

Two families, because they give opposite answers and reporting one would be choosing the
answer.

**Banded**, as in a model-predictive control horizon. The reduced matrix keeps the band, so
PureOSQP factors it with CHOLMOD — the regime libosqp's own sparse LDLᵀ is built for:

| n | m | nnz(A) | PureOSQP backend | PureOSQP | OSQP | vs OSQP | iterations |
|---|---|---|---|---|---|---|---|
| 200 | 400 | 2775 | `cholmod` | 9.70 ms | 11.0 ms | 1.13× | 1400 |
| 500 | 1000 | 6975 | `cholmod` | 19.4 ms | 22.6 ms | 1.17× | 1125 |
| 1000 | 2000 | 13975 | `cholmod` | 73.7 ms | 87.2 ms | 1.18× | 2200 |
| 2000 | 4000 | 27975 | `cholmod` | 58.0 ms | 69.4 ms | 1.20× | 825 |

**Uniformly random**, with no structure for a sparse factorization to exploit. The reduced
matrix fills in, so PureOSQP forms it sparsely and factors it densely:

| n | m | nnz(A) | PureOSQP backend | PureOSQP | OSQP | vs OSQP | iterations |
|---|---|---|---|---|---|---|---|
| 200 | 400 | 753 | `sparse_formed` | 2.19 ms | 2.31 ms | 1.06× | 225 |
| 200 | 400 | 4014 | `sparse_formed` | 10.6 ms | 30.1 ms | 2.84× | 1100 |
| 500 | 1000 | 4994 | `sparse_formed` | 31.5 ms | 128 ms | 4.06× | 1100 |
| 1000 | 2000 | 9926 | `sparse_formed` | 112 ms | 412 ms | 3.69× | 850 |
| 2000 | 4000 | 20104 | `sparse_formed` | 698 ms | 2416 ms | 3.46× | 800 |

The margin is narrow on the banded family and wide on the random one, and that ordering is
the honest one: banded is what libosqp's sparse LDLᵀ of the full KKT is designed for, and it
is a good design there. Where PureOSQP pulls ahead is the case that suits neither a sparse
factorization nor a dense one, where forming the reduced matrix from stored entries and
factoring it densely turns out to beat factoring a KKT matrix that fills in.

The banded family also found a real defect rather than confirming a belief. At
`n = 2000, m = 4000` PureOSQP first measured **0.61×** — libosqp 1.6× ahead. The cause was
not the iteration: `setup` was 49.7% of the run, because the convexity test densified `P`
into an `n×n` array for a dense Cholesky, `O(n³)` on a tridiagonal matrix. Testing it with
CHOLMOD instead is 93× faster on that matrix and took `setup` from 56.1 ms to 3.06 ms.


## The OSQP benchmark suite

`bench/osqp_suite.jl`. The seven problem classes OSQP's own benchmark suite uses, ported
from its problem definitions. Both solvers hold `SparseMatrixCSC`; iteration counts are
identical in every row, so what these measure is per-iteration cost.

| class | n | m | PureOSQP backend | PureOSQP | OSQP | vs OSQP | per iteration | setup |
|---|---|---|---|---|---|---|---|---|
| Random QP | 50 | 500 | `cholesky` | 3.62 ms | 6.69 ms | **1.85×** | **1.69×** | 3.15× |
| SVM | 808 | 1600 | `ldlfactorizations` | 2.82 ms | 4.08 ms | **1.45×** | **1.53×** | 0.82× |
| Eq QP | 200 | 100 | `cholesky` | 3.16 ms | 3.77 ms | **1.19×** | **5.19×** | 0.75× |
| Control | 320 | 540 | `sparse_formed` | 4.77 ms | 5.39 ms | **1.13×** | **1.59×** | 0.28× |
| Lasso | 816 | 816 | `ldlfactorizations` | 1.13 ms | 1.19 ms | **1.05×** | **1.16×** | 0.85× |
| Portfolio | 505 | 506 | `ldl_kkt` | 2.99 ms | 3.02 ms | **1.01×** | **1.03×** | 0.77× |
| Huber | 1806 | 1800 | `ldlfactorizations` | 2.75 ms | 2.69 ms | 0.98× | **1.08×** | 0.73× |

**Per iteration PureOSQP is ahead on all seven**, and six of seven are ahead on total time.
This is the corpus that matters, the one with the block and band structure real problems
have; the synthetic families elsewhere on this page are uniformly random, which is the worst
case for any *sparse factorization* and therefore flatters a solver that does not have one.

**Every remaining deficit is setup, and Huber is the one it costs a row.** Its loop is 1.08×
but its setup 0.73×, and setup is a quarter of a run that only takes 125 iterations. The
shorter the solve, the more setup decides — which is why Control, at 0.28× on setup, still
wins overall on 325 iterations.

**Portfolio is what a dense row costs.** Its `A` is 0.9% dense, and the reduced matrix
`R = P̃ + σI + Ãᵀ diag(ρ) Ã` would be **99% dense**: one row of `A` — the budget constraint
`1ᵀx = 1` — touches 99% of the columns, and a single dense row makes `AᵀA` dense however
sparse the rest of it is. `SparseKKT` factors the full quasi-definite system instead, which
keeps that row as one sparse row, and takes the class from 16.2 ms to 5.47 ms.

| class | nnz(A)/mn | nnz(R)/n² | densest row of A | backend |
|---|---|---|---|---|
| Portfolio | 0.009 | **0.990** | **0.990** | `ldl_kkt` |
| Lasso | 0.003 | 0.004 | 0.007 | `ldlfactorizations` |
| Huber | 0.001 | 0.003 | 0.004 | `ldlfactorizations` |
| Control | 0.038 | 0.209 | 0.097 | `sparse_formed` |

**Eq QP was a storage cost, and equilibration now absorbs most of it.** Its `P` has
`nnz(P)/n² = 0.99` — a matrix that is 99% dense, handed over as a `SparseMatrixCSC` — so the
column traversals gather where they could stream. Two things follow. Handing the same matrix
over as a `Matrix` still takes `scale!` from 2.98 ms to 0.304 ms, which is the rule under
[Sparse A](@ref "Sparse A") costing an order of magnitude when ignored. And half the passes
over `P` were redundant: the cost normalization at the end of a sweep computes the column
norms the next sweep needs, with the same `D`, so `cost_norms!` now returns both and the
class went from 0.77× to 1.19×.

### What closed the gap

Four changes, in the order they were found, each measured against the class it was aimed at.

**The substitutions paid a bounds check per nonzero.** `x[rows[p]] -= vals[p] * xj` indexes
by a row read out of the factor, which no compiler can prove is in range, where QDLDL's
identical C loop checks nothing. On factors holding two to three nonzeros per column that
check is a large fraction of the work: removing it is worth 1.18× to 1.32× on the pair.
[`check_factor`](@ref) establishes the property once per factorization instead, so the loops
run unchecked behind a guard rather than an assumption. This was the largest single item and
took Portfolio from 0.88× to 1.00×.

**Selection and the setup factorization were separate.** Equilibration and the `ρ` split now
run before the backend is chosen, so the factorization that measures a candidate's fill is
the one the solver keeps. Setup fell 24–29% on the four sparse classes.

**Refactorization rebuilt the reduced matrix through four allocating sparse products.** It is
now a single pass over a stored slot map — see [`ReducedGram`](@ref) — which allocates
nothing. A refactorization runs every time `ρ` is retuned, inside the solve loop, so this is
not a setup-only saving.

**Back-substitution gathered where it could scatter.** Storing the factor transposed lets it
scatter down a column like the forward solve rather than accumulating into a scalar.

One thing deliberately *not* done: `cholmod_backend` still forms the reduced matrix through
sparse products to measure its fill, and then builds the slot map separately rather than
reusing that pattern. Reusing it looks like free work and is not sound — a sparse product
drops an entry that cancels to exactly zero, whether one cancels depends on `ρ`, and a map
built from the cancelled pattern would silently lose that term the moment `ρ` moved. The
duplication is the price of a pattern that stays valid.

### The SparseArrays extension

PureOSQP's per-iteration products go through `mul!`, which sparse matrices already handle
well. Equilibration and the factorization were the problem: they walk the caller's matrices
entry by entry, and the generic loop visits every structural zero and reaches each through
`M[i, j]`, which on CSC is a binary search. Equilibration was **10.7× slower on a sparse
matrix than on a dense one** — the opposite of what the storage should give.

The four column traversals are now overridable, and `ext/PureOSQPSparseArraysExt.jl`
specialises them for `SparseMatrixCSC`. `SparseArrays` is a weak dependency, so the core
still has none beyond `LinearAlgebra` and TypeContracts, and the extension can only load
when the caller already has sparse matrices to pass.

| at 1% density, n=200, m=400 | dense input | sparse input |
|---|---|---|
| `scale!` | 548 µs | **67.6 µs** |
| `factorize!` | 767 µs | 626 µs |
| `setup` | 1475 µs | **844 µs** |
| full solve | 8.16 ms | **3.54 ms** |

A test asserts the two storages produce **identical** `D`, `E` and `c`, so the extension
cannot drift into being a behaviour change.

## The matrix-free backend

`linsys = :indirect` never forms the reduced matrix; it applies it through the caller's own
products and solves by preconditioned conjugate gradients. Whether that is a good trade is
a question about the problem, and the answer turns over.

On dense QPs it is not close: the inner solve costs about 23× more per iteration, and being
inexact it costs iterations too, for 15× total at `n = 50, m = 100` and 49× at
`n = 200, m = 400`.

On sparse QPs it crosses over. The direct backend now forms the reduced matrix from the
stored entries, so what remains irreducibly dense is the `n×n` matrix it factors and
inverts — `O(n³)`, however sparse the input was. The matrix-free backend pays `O(nnz)` per
CG iteration and stores only vectors. Holding about five nonzeros per row of `A` and growing
the problem (`bench/indirect_backend.jl`, `eps_abs = eps_rel = 1e-6`, single-threaded BLAS):

| n | m | density | direct | matrix-free | speedup | direct memory | matrix-free memory |
|---|---|---|---|---|---|---|---|
| 200 | 400 | 2.5% | 6.59 ms | 76.8 ms | 0.09× | 0.5 MiB | 0.19 MiB |
| 500 | 1000 | 1% | 31.0 ms | 209 ms | 0.15× | 2.4 MiB | 0.49 MiB |
| 1000 | 2000 | 0.5% | 103 ms | 326 ms | 0.32× | 8.5 MiB | 0.97 MiB |
| 2000 | 4000 | 0.25% | 702 ms | 640 ms | **1.10×** | 32.4 MiB | 1.96 MiB |
| 3000 | 6000 | 0.17% | 2244 ms | 1336 ms | **1.68×** | 71.5 MiB | 2.98 MiB |
| 4000 | 8000 | 0.125% | 4424 ms | 2328 ms | **1.90×** | 125.7 MiB | 3.86 MiB |

The crossover sits near `n = 1900`, and memory is the more durable half: the workspace
shrinks 3× at the top of the table and 33× at the bottom, because the direct backend holds
an `n×n` inverse where the matrix-free one holds vectors. That gap keeps widening with `n`,
and it decides which problems fit at all.

Density decides it at fixed size. At `n = 1000, m = 2000`:

| density | direct | matrix-free | speedup |
|---|---|---|---|
| 0.2% | 82.1 ms | 73.5 ms | **1.12×** |
| 0.5% | 104 ms | 330 ms | 0.31× |
| 1% | 118 ms | 954 ms | 0.12× |
| 2% | 172 ms | 2313 ms | 0.07× |

Both costs follow `nnz` now, since the direct backend also accumulates over stored entries;
the matrix-free one simply climbs far faster, because it pays per CG iteration where the
direct backend pays per refactorization and there were two of those against 850 iterations.

Two cautions on reading this. The iteration counts differ between backends because the
inner solve is inexact, so these are end-to-end times rather than per-iteration ones; the
benchmark asserts the two objectives agree, which they do to about eight digits. And these
numbers moved once already: before the direct backend formed `R` sparsely, matrix-free
measured 3.61× at `n = 4000` rather than 1.90×, and 96× smaller rather than 33×. A sparse
*factorization* of `R`, still open, would move them again.
