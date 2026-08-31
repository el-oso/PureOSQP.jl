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

**`update!` is worth about as much as `setup` costs, which is not much.** Its whole benefit
is skipping setup, and setup is fast (see [Sparse A](@ref "Sparse A")), so there is little
left to skip. At `n = 100` it is level with rebuilding — the two runs also differ in warm
starting, so they take different iteration counts and the row is not purely
setup-versus-setup. It pays where solves are short and setup is a large share of them, and it
remains the right call when you want warm starting; at the larger sizes it is close to a wash.

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

## Structured backends

A structured `A` is where it becomes one, and by a great deal, because then the *reduced
matrix* is structured rather than merely the input. Eliminating `ν` gives
`R = c D P D + σI + Ãᵀ diag(ρ) Ã`; diagonal scaling preserves a bandwidth and `ÃᵀρÃ` doubles
`A`'s, so

```math
\mathrm{bandwidth}(R) = \max\bigl(\mathrm{bandwidth}(P),\; 2\,\mathrm{bandwidth}(A)\bigr)
```

Each row below is one problem solved twice, once in the structured types and once as a
`Matrix`, so the iteration counts match exactly and the difference is the backend.
Reproduce with `julia --project=bench bench/structured_backends.jl`; samples are written to
`bench/results/structured_backends.json`.

| `P`, `A` | n | bw(`R`) | backend | setup | dense setup | setup× | total | dense total | total× |
|---|---|---|---|---|---|---|---|---|---|
| `Diagonal`, `Diagonal` | 400 | 0 | `diagonal` | 49.6 µs | 5.37 ms | **108×** | 0.08 ms | 7.03 ms | **83×** |
| `SymTridiagonal`, `Diagonal` | 400 | 1 | `tridiagonal` | 65.1 µs | 5.40 ms | **83×** | 0.23 ms | 12.1 ms | **52×** |
| `SymTridiagonal`, `Tridiagonal` | 400 | 2 | `banded` | 114 µs | 5.37 ms | **47×** | 0.62 ms | 11.3 ms | **18×** |
| `Diagonal`, `Diagonal` | 2000 | 0 | `diagonal` | 230 µs | 393 ms | **1710×** | 0.40 ms | 488 ms | **1216×** |
| `SymTridiagonal`, `Diagonal` | 2000 | 1 | `tridiagonal` | 306 µs | 389 ms | **1270×** | 1.15 ms | 831 ms | **725×** |
| `SymTridiagonal`, `Tridiagonal` | 2000 | 2 | `banded` | 549 µs | 384 ms | **699×** | 3.08 ms | 778 ms | **252×** |

The ratios grow with `n` because the two sides have different exponents, not because the
constant is better: the dense path factors in `O(n³)` and applies in `O(n²)`, and these are
`O(n b²)` and `O(n b)`. At bandwidth 0 there is no factorization at all — a solve is `n`
divisions.

**What is selected, and by what.** `Diagonal` with `Diagonal` gives a diagonal `R`;
`SymTridiagonal` or `Tridiagonal` with `Diagonal`, and any of the three with a `Bidiagonal`
`A`, give bandwidth 1 and an `ldlt`. Both are core LinearAlgebra. Bandwidth 2 and up needs
BandedMatrices.jl loaded, since LinearAlgebra stores no symmetric banded type past
`SymTridiagonal`; without it those problems take the dense path, correctly but densely.
Selection is dispatch on the pair of types — no setting, no density gate — and the banded
backend declines in both directions, below bandwidth 2 where the LinearAlgebra backends are
cheaper and above a quarter of the matrix where the dense path wins per iteration.

**Keyed on `A`, not `P`.** `ÃᵀρÃ` is dense for a general `A` whatever `P` looked like, so a
`Diagonal` `P` with a dense `A` has a dense reduced matrix and correctly gets the dense
backend. Widening `A` costs twice what widening `P` does.

**The structure has to be established, not inherited.** Julia's arithmetic does not carry it
through: `D P D` on a `SymTridiagonal` returns a `Tridiagonal`, which `cholesky` rejects as
not Hermitian though it is symmetric to `1e-17`, and a `Diagonal` `P` with a `Bidiagonal` `A`
returns a dense `Array` despite having bandwidth 1. The bands are computed entry by entry for
that reason. An `ldlt` is likewise no substitute for a Cholesky's failure reporting — it
returns a negative pivot for an indefinite matrix and throws only on an exact zero — so the
tridiagonal backend tests the pivots itself, where the banded one can rely on `issuccess`.

## Block-diagonal structure

A [`PureOSQP.BlockDiagonal`](@ref) `P` and `A` decouple the reduced matrix into `K`
independent blocks, factored one at a time and never assembled whole. The factor cost falls
from `n³` to `Σnᵢ³` and the storage from `n²` to `Σnᵢ²`, so both improve as the same `n`
splits further. Each row is one problem solved twice, once in `BlockDiagonal` and once as a
`Matrix`, so the iteration count is shared and the difference is the backend. Reproduce with
`julia --project=bench bench/block_backend.jl`; samples in
`bench/results/block_backend.json`, single-threaded BLAS, `n = 240`.

| `K` | block | iters | `block` | dense | speedup | factor words | dense words | memory× |
|---|---|---|---|---|---|---|---|---|
| 2 | 120 | 50 | 4.28 ms | 4.79 ms | 1.12× | 14 520 | 28 920 | 2.0× |
| 3 | 80 | 50 | 2.98 ms | 3.68 ms | 1.24× | 9 720 | 28 920 | 3.0× |
| 4 | 60 | 50 | 2.44 ms | 3.23 ms | 1.32× | 7 320 | 28 920 | 4.0× |
| 6 | 40 | 150 | 2.23 ms | 3.11 ms | 1.40× | 4 920 | 28 920 | 5.9× |
| 8 | 30 | 100 | 1.86 ms | 2.68 ms | 1.44× | 3 720 | 28 920 | 7.8× |
| 12 | 20 | 250 | 2.12 ms | 3.14 ms | 1.48× | 2 520 | 28 920 | 11.5× |
| 20 | 12 | 100 | 1.66 ms | 3.18 ms | **1.91×** | 1 560 | 28 920 | **18.5×** |

Read the two halves differently. The time column improves modestly, because at `n = 240` a
dense factorization is already cheap and the per-iteration `symv` is what dominates. The
storage column improves exactly as `Σnᵢ²/n²` predicts and does not depend on the size being
small. Iteration counts vary down the table because each `K` is a different problem; the
comparison that holds is across a row, not down a column.

The matrix-free backend on the same problems takes 4.9 ms to 15.2 ms — worse than either
direct path at every split, since the blocks are dense and CG gains nothing from a structure
it cannot see.

## Low-rank coupling

A `Diagonal` `P` with a [`PureOSQP.RowCoupled`](@ref) `A` makes the reduced matrix a diagonal
plus a rank-`k` correction, which Woodbury solves without forming it: two `gemv`s against a
`k×n` block and one `k×k` solve, in `O(nk)` time and storage rather than `O(n²)`. Reproduce
with `julia --project=bench bench/lowrank_backend.jl`; samples in
`bench/results/lowrank_backend.json`, single-threaded BLAS.

| n | k | iters | setup | dense setup | total | dense total | total× |
|---|---|---|---|---|---|---|---|
| 500 | 1 | 75 | 80.3 µs | 9.31 ms | 0.24 ms | 19.2 ms | **81×** |
| 500 | 2 | 75 | 88.5 µs | 9.34 ms | 0.26 ms | 19.3 ms | **75×** |
| 500 | 6 | 100 | 138 µs | 9.34 ms | 0.58 ms | 14.3 ms | **25×** |
| 500 | 16 | 125 | 282 µs | 9.34 ms | 0.79 ms | 15.5 ms | **20×** |
| 1000 | 1 | 75 | 154 µs | 57.7 ms | 0.45 ms | 119 ms | **267×** |
| 1000 | 6 | 100 | 270 µs | 56.4 ms | 1.11 ms | 79.1 ms | **71×** |
| 1000 | 16 | 150 | 557 µs | 55.8 ms | 1.64 ms | 98.2 ms | **60×** |
| 2000 | 1 | 75 | 304 µs | 391 ms | 0.87 ms | 806 ms | **923×** |
| 2000 | 6 | 100 | 533 µs | 385 ms | 2.16 ms | 559 ms | **259×** |
| 2000 | 16 | 175 | 1.11 ms | 392 ms | 3.71 ms | 691 ms | **186×** |

The ratio grows with `n` at fixed `k` and shrinks as `k` climbs, which is what `O(nk)` against
`O(n²)` predicts. The rung declines once `10k > n`, below the measured crossing so the limit
holds at any BLAS thread count — see the gate discussion in
`bench/results/gate_crossover_lowrank.json`. Each row is a different problem, so read across
a row rather than down a column.

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

The banded family is what makes the convexity test's representation matter. Testing a
tridiagonal `P` by densifying it into an `n×n` array for a dense Cholesky is `O(n³)` on a
matrix holding `O(n)` entries, and at `n = 2000, m = 4000` that alone is half the run.
Testing it through CHOLMOD instead is 93× faster on that matrix — hence the sparse method in
`ext/PureOSQPSparseArraysExt.jl`.


## The OSQP benchmark suite

`bench/osqp_suite.jl`. The seven problem classes OSQP's own benchmark suite uses, ported
from its problem definitions. Both solvers hold `SparseMatrixCSC`; iteration counts are
identical in every row, so what these measure is per-iteration cost.

| class | n | m | PureOSQP backend | PureOSQP | OSQP | vs OSQP | per iteration | setup |
|---|---|---|---|---|---|---|---|---|
| Eq QP | 200 | 100 | `cholesky` | 1.60 ms | 3.31 ms | **2.06×** | **5.19×** | **1.51×** |
| Random QP | 50 | 500 | `cholesky` | 3.58 ms | 6.66 ms | **1.86×** | **1.70×** | **3.58×** |
| SVM | 808 | 1600 | `ldlfactorizations` | 2.69 ms | 3.95 ms | **1.47×** | **1.56×** | **1.11×** |
| Control | 320 | 540 | `sparse_formed` | 4.46 ms | 5.33 ms | **1.20×** | **1.60×** | 0.33× |
| Portfolio | 505 | 506 | `ldl_kkt` | 2.75 ms | 2.96 ms | **1.08×** | **1.08×** | 0.99× |
| Lasso | 816 | 816 | `ldlfactorizations` | 1.07 ms | 1.15 ms | **1.08×** | **1.09×** | **1.07×** |
| Huber | 1806 | 1800 | `ldlfactorizations` | 2.52 ms | 2.62 ms | **1.04×** | **1.08×** | 0.98× |

**PureOSQP is ahead on every class, and the weakest is 1.04×.** This is the corpus that
matters, the one with the block and band structure real problems have; the synthetic families
elsewhere on this page are uniformly random, which is the worst case for any *sparse
factorization* and therefore flatters a solver that does not have one.

Every figure here is a median over ten seconds of samples, and the two solvers are timed in
the order pure, libosqp, libosqp, pure. Timing one solver to completion before starting the
other charges any drift across the row — the clock ramping, the caches filling — entirely to
whichever went second, which is precisely the bias a ratio cannot survive; interleaving puts
each of them both early and late. The medians of the two turns are taken over the pooled
samples rather than averaged, so a turn that ran under a transient does not get half the
weight of a clean one.

**Setup leads on four of the seven, and Portfolio and Huber sit on parity.** Control's 0.33×
is the dense path, where the deficit is bought rather than suffered: forming and inverting `R`
costs `O(n³)` once and makes every iteration a single `symv`, worth 1.60× per iteration over
325 of them. Taking it sparse instead was measured and loses more in the loop than it recovers
in setup.

Within that factorization, forming `R` from the stored entries is the small part and the
explicit inverse is the large one: `potrf` costs `n³/3` and the `potri` that follows a further
`2n³/3`. The inverse is what makes each iteration one `symv`. Keeping the factor and applying
it as two `trsv` is the same flop count but serial; inverting only the triangle with `trtri`
and applying two `trmv` halves the setup cost but doubles the loop's. Over 325 iterations
neither trade pays.

**Some setup figures are far more measurable than others.** A setup dominated by one large
compute-bound operation times reliably: Control's is a dense Cholesky and inverse on
`n = 320`, and its deficit is reproducible rather than an artifact. A setup made of many small
operations and workspace allocation does not, because the ratio divides two sub-millisecond
numbers that sit within a few percent of each other. Portfolio's 0.99× and Huber's 0.98× are
parity, not deficits — read them, and Lasso's, as the interval they are rather than the digits
they print.

What dominates that interval is other load on the machine, not sampling: a pool of idle
processes left running moved a setup ratio further than any change to the code in this table
did. Run these with the machine otherwise quiet.

**Portfolio is what a dense row costs.** Its `A` is 0.9% dense, and the reduced matrix
`R = P̃ + σI + Ãᵀ diag(ρ) Ã` would be **99% dense**: one row of `A` — the budget constraint
`1ᵀx = 1` — touches 99% of the columns, and a single dense row makes `AᵀA` dense however
sparse the rest of it is. The `ldl_kkt` backend factors the full quasi-definite system
instead, which keeps that row as one sparse row: its factor holds 2322 nonzeros against the
KKT's 3305, so the elimination fills in nothing. Forming the reduced matrix here would mean a
dense `505×505` factorization in place of a sparse `1011×1011` one.

| class | nnz(A)/mn | nnz(R)/n² | densest row of A | backend |
|---|---|---|---|---|
| Portfolio | 0.009 | **0.990** | **0.990** | `ldl_kkt` |
| Lasso | 0.003 | 0.004 | 0.007 | `ldlfactorizations` |
| Huber | 0.001 | 0.003 | 0.004 | `ldlfactorizations` |
| Control | 0.038 | 0.209 | 0.097 | `sparse_formed` |

**Eq QP is a storage cost, and equilibration absorbs most of it.** Its `P` has
`nnz(P)/n² = 0.99` — a matrix that is 99% dense, handed over as a `SparseMatrixCSC` — so the
column traversals gather where they could stream. Handing the same matrix over as a `Matrix`
takes the scaling pass from 2.98 ms to 0.304 ms, which is the rule under
[Sparse A](@ref "Sparse A") costing an order of magnitude when ignored. A sweep also needs
only one traversal of `P`, not two: its closing cost normalization computes the column norms
the next sweep opens with, under the same `D`, so `cost_norms!` returns both.

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

## An operator that is never materialized

The sections above all hand the solver a matrix. This one hands it something that has no
entries to hand over: an operator supplying only `mul!`, wrapped in
[`PureOSQP.ProductOperator`](@ref). Nothing can be formed, so the density gates and the
direct rungs are not merely slower — they do not apply, and the ladder lands on the
matrix-free backend by elimination.

The comparison is against the same operator materialized into a `Matrix`, which is the choice
a caller has when the entries do exist. Reproduce with
`julia --project=bench bench/operator_protocol.jl`; samples in
`bench/results/operator_protocol.json`.

| n | BLAS threads | lazy setup | dense setup | setup× | lazy step | dense step | step× |
|---|---|---|---|---|---|---|---|
| 500 | 1 | 167 µs | 8.99 ms | **54×** | 303 µs | 44.9 µs | 0.15× |
| 1000 | 1 | 677 µs | 56.1 ms | **83×** | 1.08 ms | 259 µs | 0.24× |
| 500 | 8 | 179 µs | 5.54 ms | **31×** | 310 µs | 42.2 µs | 0.14× |
| 1000 | 8 | 682 µs | 25.1 ms | **37×** | 568 µs | 94.6 µs | 0.17× |

The trade is the whole story and it points both ways. Setup is 31–83× cheaper because there
is no `O(n³)` factorization to pay for — only the preconditioner. A step is 4–7× dearer,
because a preconditioned CG solve against the operator replaces one `symv` against a stored
inverse. Which wins is decided by how many iterations the problem takes, and the setup saving
grows with `n` while the per-step penalty does not.

Two things this measures that the matrix-free numbers above do not. The setup ratio narrows
as BLAS threads go up (54× to 31× at `n = 500`), because the dense side is the half that
threads. And the lazy step is essentially thread-independent, since the operator's `mul!`
here is the caller's own code rather than a BLAS call.

An operator supplied this way carries the solver's guarantees only as far as its own `mul!`
does: a product that allocates makes the iteration allocate.

## Ill-conditioned problems

At `n = 300` with `κ(P) = κ(A) = 1e12`, taken from a real case. Reproduce with
`julia --project=bench bench/illconditioned.jl`; samples in
`bench/results/illconditioned.json`, single-threaded BLAS, `eps_abs = eps_rel = 1e-8`.

**Read the statuses before the timings.** ADMM does not converge on these problems within
`max_iter`: every row below stopped at `MAX_ITER_REACHED` after 4000 iterations. So the time
column compares what four backends cost for the *same* number of iterations, not four
answers, and the objective column is a consistency check between backends rather than
evidence that any of them is near the optimum.

| shape | `linsys` | backend | 4000 iterations | factor words | objective |
|---|---|---|---|---|---|
| dense | `:auto` | `cholesky` | 75.2 ms | 45 150 | 182 375.90 |
| dense | `:kkt` | `bunchkaufman` | 320 ms | 180 300 | 182 375.34 |
| dense | `:indirect` | `indirect` | 751 ms | 0 | 0.0169 |
| blocks of 50 | `:auto` | `block` | **22.6 ms** | **7 650** | 27 139.9885 |
| blocks of 50 | `:dense` | `cholesky` | 35.6 ms | 45 150 | 27 139.9926 |
| blocks of 50 | `:kkt` | `bunchkaufman` | 330 ms | 180 300 | 27 140.019 |
| blocks of 50 | `:indirect` | `indirect` | 381 ms | 0 | 1.25e9 |

The finding that survives non-convergence is in the last column. The three direct backends
track each other — six significant figures on the block problem, five on the dense one — so
they are all in the same place after 4000 iterations, whatever place that is. The matrix-free
backend is not with them, and not by a little: `0.0169` where the direct backends read
`182 375`, and `1.25e9` where they read `27 140`. Conjugate gradients on a reduced matrix
whose conditioning is `κ(A)²` is not solving the same problem to a looser tolerance; its
iterates are somewhere else entirely.

So `linsys = :indirect` is the one backend that ill-conditioning disqualifies rather than
merely slows, and the declared structure is what pays here: the block backend is both the
fastest per iteration and the smallest, on the problem whose structure it can see.
