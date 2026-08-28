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
| 10 | 20 | 0.050 ms | 0.042 ms | 0.064 ms | 1.28× | 1.50× | 100 | 3.8e-15 |
| 25 | 50 | 0.713 ms | 0.587 ms | 1.39 ms | 1.94× | 2.36× | 675 | 1.4e-15 |
| 50 | 100 | 2.18 ms | 1.97 ms | 5.02 ms | 2.30× | 2.55× | 900 | 1.8e-15 |
| 100 | 200 | 5.93 ms | 5.34 ms | 20.5 ms | 3.46× | 3.84× | 850 | 1.3e-15 |
| 200 | 400 | 29.2 ms | 25.5 ms | 126 ms | 4.31× | 4.93× | 1175 | 4.4e-15 |
| 400 | 800 | 62.3 ms | 66.8 ms | 604 ms | **9.70×** | 9.05× | 625 | 4.1e-15 |
| 100 | 50 | 0.453 ms | 0.412 ms | 1.15 ms | 2.53× | 2.79× | 50 | 4.4e-15 |
| 200 | 100 | 1.49 ms | 1.36 ms | 7.22 ms | 4.86× | 5.31× | 50 | 1.1e-14 |
| 100 | 1000 | 145 ms | 131 ms | 693 ms | 4.79× | 5.29× | 7475 | 2.6e-11 |
| 200 | 2000 | 241 ms | 224 ms | 1302 ms | 5.41× | 5.80× | 3025 | 7.1e-15 |

**The iteration count is identical to OSQP in every case.** That is the result worth caring
about: the equilibration, the ρ schedule and the termination tests reproduce the reference
exactly, not approximately. The objective agrees to about `1e-15`, and switching BLAS
changes neither — PureBLAS gives bit-comparable answers (`|Δx| ≈ 1e-14`) on the same
iteration counts.

The same agreement holds against **libosqp 1.x**, which has no Julia wrapper and so is
checked separately by `bench/headtohead_v1.jl` through a subprocess oracle: identical
iteration counts on all seven of its cases, objectives to `1e-13`. It is left out of the
timing table because timing a subprocess measures the subprocess.
PureOSQP is now ahead of OSQP on every case in this table, including `n = 100, m = 50`,
which it used to lose. PureBLAS helps everywhere except `n = 400, m = 800`, where it is
marginally behind OpenBLAS.

## How to read this

**The timings are not a claim that this solver is better than OSQP.** These are dense
problems, which is the reference implementation's worst case: it is a *sparse* solver being
handed dense matrices, so it pays sparse-format overhead and scalar sparse LDLᵀ where
PureOSQP gets BLAS-3 dense Cholesky. The right reading is a *storage-format* comparison,
not a solver-quality one. [Sparse A](@ref "Sparse A") shows the other side, and
[Against other solvers](@ref) shows what a solver built for dense QPs does.

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
`julia --project=bench bench/update_bench.jl`.

| n | m | `update!` | fresh `setup` each step | saved | libosqp 1.x | vs | factorizations |
|---|---|---|---|---|---|---|---|
| 10 | 20 | 1.36 ms | 1.86 ms | 1.37× | 1.81 ms | 1.33× | 8 |
| 25 | 50 | 4.77 ms | 9.25 ms | 1.94× | 9.63 ms | 2.02× | 14 |
| 50 | 100 | 25.8 ms | 37.0 ms | 1.43× | 57.6 ms | 2.23× | 7 |
| 100 | 200 | 158 ms | 144 ms | **0.91×** | 484 ms | 3.07× | 3 |
| 200 | 400 | 631 ms | 643 ms | 1.02× | 2374 ms | 3.76× | 1 |

"Factorizations" counts the whole 20-step sequence: one from `setup`, plus one per step
whose constraint classification changed.

**`update!` is worth much less than it used to be, and that is a consequence of making
`setup` faster.** Its whole benefit is skipping setup; once setup dropped sevenfold (see
[Sparse A](@ref "Sparse A")) there was far less left to skip. At `n = 100` it is now
slightly *behind* rebuilding — the two runs also differ in warm starting, so they take
different numbers of iterations and the comparison is no longer purely setup-versus-setup.
It still pays where solves are short and setup is a large share, and it remains the right
call when you want warm starting, but the 2× figures this table used to show are gone.

The factorization count is not always 1: equilibration rescales the bounds, so a row whose
scaled gap `ũ - l̃` falls under `RHO_TOL` is reclassified as an equality and its `ρ`
changes. That is upstream's rule applied to the scaled bounds exactly as upstream applies
it, and it means `update!` is not unconditionally factorization-free.

The same sequences under PureBLAS rather than OpenBLAS:

| n | m | OpenBLAS | PureBLAS | ratio |
|---|---|---|---|---|
| 25 | 50 | 4.72 ms | 3.85 ms | 1.23× |
| 50 | 100 | 17.3 ms | 15.2 ms | 1.14× |
| 100 | 200 | 127 ms | 116 ms | 1.09× |
| 200 | 400 | 531 ms | 474 ms | 1.12× |

## Linear-system backend

`bench/kkt_backend.jl` reproduces the cost and accuracy comparison between the reduced
Cholesky and the full-KKT Bunch-Kaufman factorization described under
[Algorithm](@ref "The linear system"), including the near-parallel-row family that
equilibration cannot fix.

## On PureBLAS instead of OpenBLAS

[PureBLAS.jl](https://github.com/el-oso/PureBLAS.jl) is a pure-Julia BLAS/LAPACK. Its
`activate()` overlays per-symbol forwards onto libblastrampoline, so PureOSQP runs on it
**with no code changes** — the same `mul!`, `cholesky!` and `ldiv!` calls are rerouted in
process. Reproduce with `bench/pureblas_backend.jl` (setup instructions in its header).

Note `BLAS.get_config()` cannot show this: OpenBLAS stays loaded and the forwards sit on
top of it. `PureBLAS.is_active()` is the check, and the benchmark asserts it at every
measurement — otherwise a rerouting that silently failed would look like a clean tie.

Measured against PureBLAS at commit `ea79919`, which `bench/results/pureblas_backend.json`
records alongside the timings.

| n | m | OpenBLAS | PureBLAS | ratio | iterations | \|Δx\| |
|---|---|---|---|---|---|---|
| 25 | 50 | 0.279 ms | 0.238 ms | 1.17× | 225 | 4.8e-15 |
| 50 | 100 | 0.922 ms | 0.832 ms | 1.11× | 350 | 4.0e-15 |
| 100 | 200 | 10.2 ms | 9.30 ms | 1.10× | 1500 | 8.7e-15 |
| 200 | 400 | 16.8 ms | 14.6 ms | 1.15× | 650 | 1.1e-14 |
| 100 | 50 | 0.453 ms | 0.413 ms | 1.10× | 50 | 8.1e-15 |

**Correctness is exact** — identical iteration counts and solutions agreeing to `1e-14`, so
PureBLAS is a faithful drop-in — **and it is faster than OpenBLAS on every case**:

| operation (n=200, m=400) | OpenBLAS | PureBLAS | ratio | runs |
|---|---|---|---|---|
| `gemv A*x` | 3.26 µs | 2.67 µs | 1.22× | every iteration |
| `gemv Aᵀy` (transposed) | 4.02 µs | 2.51 µs | **1.60×** | every iteration |
| `gemv At*y` (materialized transpose) | 3.74 µs | 2.75 µs | 1.36× | — |
| `syrk WᵀW` | 309 µs | 233 µs | **1.33×** | on a ρ update |
| `potrf` | 116 µs | 58.2 µs | **2.00×** | on a ρ update |
| `trsv F\b` | 11.7 µs | 12.0 µs | 0.97× | every iteration |

PureOSQP's inner loop is Level-2-bound: it does a handful of `gemv` and one `trsv` per
iteration and touches `syrk`/`potrf` only when ρ changes, so the whole-solve ratio tracks
`gemv`.

### What an earlier measurement got wrong

A previous version of this page reported PureBLAS at 0.49–0.96×, with transposed `gemv`
**10× slower** than OpenBLAS. That was real, and it was a genuine bug rather than a tuning
problem: PureBLAS's BLAS-2 SIMD path was unreachable *through `activate()`* specifically,
so every measurement taken via the libblastrampoline reroute — which is how this benchmark
runs — fell back to a scalar path. The same kernels called directly were fine. It is fixed
upstream; transposed `gemv` went from 41.6 µs to 2.51 µs.

The mistaken diagnosis is worth recording too. The obvious suspicion was per-machine
tuning: PureBLAS autotunes, and two of its seven knobs (`gemvt_percol_window`, `gemvt_pf`)
govern exactly this path. That hypothesis was wrong. Running `PureBLAS.tune!(unlocked=true)`
here — three independent calibration runs on an idle machine — pins **nothing**:
`sytrf_cmult` disagreed across runs (`[1, 2, 2]`) and every other knob tied, so the report
is "the in-code defaults are adequate here". Tuning was never what separated the two.

### On the threading

The two sides are not symmetric. OpenBLAS is pinned with `BLAS.set_num_threads(1)`;
PureBLAS is plain Julia, so it is bounded by `Threads.nthreads()` — 12 in these runs —
which `BLAS.set_num_threads` does not affect. Giving OpenBLAS more threads does not change
the picture at these sizes: at n=200, m=400 it took 25.4 ms on 1 thread, 25.8 ms on 4 and
26.3 ms on 8.

## Against other solvers

Everything above compares PureOSQP with OSQP, which puts a sparse solver on dense data —
its worst case. These are three other solvers on the same dense QPs, one per algorithm
family. Reproduce with `julia --project=bench bench/solvers.jl`.

| n | m | PureOSQP | OSQP | DAQP | Clarabel |
|---|---|---|---|---|---|
| 10 | 20 | 0.132 ms | 0.156 ms | **0.003 ms** | 0.128 ms |
| 25 | 50 | 0.274 ms | 0.538 ms | **0.035 ms** | 0.646 ms |
| 50 | 100 | 0.921 ms | 2.15 ms | **0.206 ms** | 2.86 ms |
| 100 | 200 | 10.3 ms | 34.7 ms | **1.86 ms** | 17.2 ms |
| 200 | 400 | 16.9 ms | 76.1 ms | 16.0 ms | 111 ms |
| 100 | 50 | 0.455 ms | 1.16 ms | **0.359 ms** | 6.02 ms |

DAQP is a dense active-set solver, and on small problems it is the right tool by a wide
margin — 44× faster at `n = 10`. The gap closes with size, and by `n = 200, m = 400` the
two are level. That shape is what the algorithms predict: an active-set method walks a
handful of expensive iterations and terminates exactly, which is unbeatable when the active
set is small, while ADMM's cost per iteration grows more slowly. If you have one small
dense QP and no sequence, reach for DAQP.

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
| dense `P` (vs `Symmetric`) | 62.0 ms | 63.7 ms | 0.97× | 4375 |
| diagonal `P` (vs `Diagonal`) | 8.53 ms | 9.78 ms | 0.87× | 550 |
| tridiagonal `P` (vs `Symmetric`) | 36.8 ms | 38.1 ms | 0.96× | 2550 |
| `A` as a `SubArray` | 13.5 ms | 14.9 ms | 0.90× | 900 |

**Structured storage does not help here — it costs a little.** The mechanism works: a
`Diagonal` `P` really does get an `O(n)` product instead of `O(n²)`. But `P` is not where
the time goes. Each iteration also does two products with `A`, which at `m = 2n` is four
times the work of the dense `P` product, and `A` is dense in every row above. Meanwhile
equilibration reads `P[i, j]` over the whole matrix, and on a `Diagonal` each off-diagonal
read is a branch returning zero where the dense array is a straight load — which is where
the 0.87× comes from.

So the honest claim is narrow: the caller's matrices are never copied or mutated, any
`AbstractMatrix` works, and a cheaper `mul!` is used when one exists. On a dense `A` that
is not a speed feature.

## Sparse A

`A` and `P` genuinely sparse, so OSQP's sparse LDLᵀ is playing to its strength. PureOSQP is
measured both ways: handed dense copies, and handed the sparse matrices directly, where a
package extension walks `nzrange` instead of indexing.

| n | m | density | densified | sparse input | OSQP | best vs OSQP |
|---|---|---|---|---|---|---|
| 200 | 400 | 1% | 13.2 ms | 8.08 ms | 4.81 ms | **0.59×** |
| 200 | 400 | 5% | 22.8 ms | 17.1 ms | 24.5 ms | 1.43× |
| 200 | 400 | 20% | 22.9 ms | 27.3 ms | 37.9 ms | 1.65× |
| 400 | 800 | 1% | 59.0 ms | 34.9 ms | 38.7 ms | 1.11× |
| 400 | 800 | 5% | 90.1 ms | 69.5 ms | 111 ms | 1.59× |
| 400 | 800 | 20% | 222 ms | 268 ms | 381 ms | 1.72× |

Iteration counts are identical everywhere — the storage changes the speed, never the path,
and the benchmark asserts that rather than assuming it.

**PureOSQP wins every row except one: `n = 200, m = 400` at 1% density, where OSQP is 1.7×
faster.** That is the one cell in this whole page where the reference implementation is
still ahead, and it is the case it is built for — a small, very sparse problem where the
KKT matrix is nearly empty.

**Which storage to pass depends on density.** Below about 10%, hand PureOSQP the sparse
matrices: the products are 4–8× cheaper and equilibration walks only the stored entries,
worth 1.6–1.7× overall. Above it, densify: sparse `mul!` loses to dense BLAS-2 once there
is enough work per row, and at 20% the sparse path is 1.2× *slower* than the dense one.
The solver does not choose for you, because the right choice depends on a density it would
have to compute.

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
| `scale!` | 571 µs | **67.5 µs** |
| `factorize!` | 549 µs | 423 µs |
| `setup` | 1265 µs | **633 µs** |
| full solve | 13.1 ms | **8.01 ms** |

A test asserts the two storages produce **identical** `D`, `E` and `c`, so the extension
cannot drift into being a behaviour change.

### Why OSQP still wins the sparsest case

Splitting the 1% case, `n = 200`, `m = 400`: PureOSQP's setup is now 0.63 ms against OSQP's
0.44 ms, so setup is no longer the story. What remains is the iteration loop — and within
it, the triangular solve. The reduced system is `n×n` and **dense whatever `A` looked
like**, so `ldiv!` costs ~12 µs per iteration, which at 475 iterations is most of the
remaining time. OSQP's sparse LDLᵀ on a nearly-empty KKT matrix is far cheaper.

Nothing in the column traversals fixes that. Closing it needs a **sparse factorization
backend** — and the reference implementation's own design is the hint: for sparse problems
libosqp does not form `P + σI + AᵀρA` at all, because `AᵀA` fills in catastrophically when
`A` has even one dense row. It factors the full quasi-definite KKT instead. Since the
linear-system backend here is already a declared, checked interface
([`LinearSystem`](@ref)), that is a third implementation rather than a patch — it is not
built, and it is the honest answer to "make it competitive at 1%".
