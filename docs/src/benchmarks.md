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
| 10 | 20 | 0.051 ms | 0.044 ms | 0.063 ms | 1.23× | 1.42× | 100 | 3.8e-15 |
| 25 | 50 | 0.744 ms | 0.636 ms | 1.38 ms | 1.86× | 2.18× | 675 | 1.4e-15 |
| 50 | 100 | 2.57 ms | 2.34 ms | 5.00 ms | 1.95× | 2.13× | 900 | 1.8e-15 |
| 100 | 200 | 7.95 ms | 7.40 ms | 20.4 ms | 2.56× | 2.75× | 850 | 1.3e-15 |
| 200 | 400 | 37.6 ms | 33.7 ms | 126 ms | 3.35× | 3.74× | 1175 | 4.4e-15 |
| 400 | 800 | 96.6 ms | 101 ms | 600 ms | **6.21×** | 5.96× | 625 | 4.1e-15 |
| 100 | 50 | 1.33 ms | 1.29 ms | 1.15 ms | **0.87×** | 0.89× | 50 | 4.4e-15 |
| 200 | 100 | 5.49 ms | 5.35 ms | 7.24 ms | 1.32× | 1.35× | 50 | 1.1e-14 |
| 100 | 1000 | 152 ms | 131 ms | 672 ms | 4.43× | **5.13×** | 7475 | 2.6e-11 |
| 200 | 2000 | 273 ms | 267 ms | 1288 ms | 4.72× | 4.82× | 3025 | 7.1e-15 |

**The iteration count is identical to OSQP in every case.** That is the result worth caring
about: the equilibration, the ρ schedule and the termination tests reproduce the reference
exactly, not approximately. The objective agrees to about `1e-15`, and switching BLAS
changes neither — PureBLAS gives bit-comparable answers (`|Δx| ≈ 1e-14`) on the same
iteration counts.

The same agreement holds against **libosqp 1.x**, which has no Julia wrapper and so is
checked separately by `bench/headtohead_v1.jl` through a subprocess oracle: identical
iteration counts on all seven of its cases, objectives to `1e-13`. It is left out of the
timing table because timing a subprocess measures the subprocess.

PureOSQP is genuinely slower than OSQP at `n = 100, m = 50` (0.87×), where the reduced
system is barely smaller than the full one and there is no dense-linear-algebra advantage
to collect. PureBLAS helps everywhere except `n = 400, m = 800`, where it is marginally
behind OpenBLAS.

## How to read this

**The timings are not a claim that this solver is better than OSQP.** These are dense
problems, which is the reference implementation's worst case: it is a *sparse* solver being
handed dense matrices, so it pays sparse-format overhead and scalar sparse LDLᵀ where
PureOSQP gets BLAS-3 dense Cholesky. The right reading is a *storage-format* comparison,
not a solver-quality one. On genuinely sparse problems the reference implementation is
expected to win, and PureOSQP has no answer for a sparse `A` beyond treating it as dense.

**What the matching iteration counts do and do not show.** They mean the algorithm's
control logic tracks the reference. But the count is *quantized*: with
`check_termination = 25`, matching counts say both crossed the threshold within the
same 25-iteration window, not that the iterates agree. That the iterates agree is a
separate result — the transcription test, matching to `1e-10` over the first 25 iterations
across both linear-system backends and both scaled and unscaled space. And the returned `y`
is a third thing again, since the objective does not contain `y`; that one comes from the
referee.

**Run-to-run spread.** Two consecutive runs of this benchmark agreed within about 5% on
every row. An earlier published version of this table reported 13.0× at `n = 400`; that
figure came from a run taken while other heavy jobs were on the machine, which inflated the
reference column. The reproducible figure is 6.3×.

## Sequential re-solves

`P` and `A` fixed, `q`, `l` and `u` changing every step — the receding-horizon loop OSQP is
most used for. 20 solves per case, `eps_abs = eps_rel = 1e-6`. Reproduce with
`julia --project=bench bench/update_bench.jl`.

| n | m | `update!` | fresh `setup` each step | saved | libosqp 1.x | vs | factorizations |
|---|---|---|---|---|---|---|---|
| 10 | 20 | 1.51 ms | 2.10 ms | 1.39× | 1.79 ms | 1.19× | 8 |
| 25 | 50 | 4.84 ms | 10.3 ms | 2.13× | 9.73 ms | 2.01× | 14 |
| 50 | 100 | 25.8 ms | 44.8 ms | 1.73× | 54.8 ms | 2.12× | 7 |
| 100 | 200 | 160 ms | 176 ms | 1.11× | 486 ms | 3.04× | 3 |
| 200 | 400 | 630 ms | 809 ms | 1.28× | 2358 ms | 3.74× | 1 |

"Factorizations" counts the whole 20-step sequence: one from `setup`, plus one per step
whose constraint classification changed.

Two things worth reading off this. The saving from `update!` is bounded by how much of the
run is setup: at `n = 200` the ADMM iterations dominate, so skipping 19 of 20
factorizations is worth only 1.28×, while at `n = 25` — short solves, setup a large share —
it is 2.13×. And the factorization count is not always 1: equilibration rescales the bounds,
so a row whose scaled gap `ũ - l̃` falls under `RHO_TOL` is reclassified as an equality and
its `ρ` changes. That is upstream's rule applied to the scaled bounds exactly as upstream
applies it, and it means `update!` is not unconditionally factorization-free.

The same sequences under PureBLAS rather than OpenBLAS, 20 solves each:

| n | m | OpenBLAS | PureBLAS | ratio |
|---|---|---|---|---|
| 25 | 50 | 4.72 ms | 3.85 ms | 1.23× |
| 50 | 100 | 17.3 ms | 15.2 ms | 1.14× |
| 100 | 200 | 127 ms | 116 ms | 1.09× |
| 200 | 400 | 531 ms | 474 ms | 1.12× |

Slightly better than the single-solve ratios, which is what the composition predicts:
`update!` removes most of the factorizations, so a larger share of the run is the
`gemv`-bound iteration loop, and that is where PureBLAS is ahead.

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
| 25 | 50 | 0.324 ms | 0.281 ms | **1.15×** | 225 / 225 | 4.8e-15 |
| 50 | 100 | 1.31 ms | 1.22 ms | **1.08×** | 350 / 350 | 4.0e-15 |
| 100 | 200 | 12.2 ms | 11.4 ms | **1.07×** | 1500 / 1500 | 8.7e-15 |
| 200 | 400 | 25.2 ms | 22.9 ms | **1.10×** | 650 / 650 | 1.1e-14 |
| 100 | 50 | 1.32 ms | 1.28 ms | **1.03×** | 50 / 50 | 8.1e-15 |

**Correctness is exact** — identical iteration counts and solutions agreeing to `1e-14`, so
PureBLAS is a faithful drop-in — **and it is faster than OpenBLAS on every case**. The
per-operation breakdown shows where that comes from:

| operation (n=200, m=400) | OpenBLAS | PureBLAS | ratio | runs |
|---|---|---|---|---|
| `gemv A*x` | 3.22 µs | 2.43 µs | **1.32×** | every iteration |
| `gemv Aᵀy` (transposed) | 3.80 µs | 2.46 µs | **1.54×** | every iteration |
| `gemv At*y` (materialized transpose) | 3.73 µs | 2.76 µs | 1.35× | — |
| `syrk WᵀW` | 311 µs | 233 µs | **1.33×** | on a ρ update |
| `potrf` | 116 µs | 57.2 µs | **2.02×** | on a ρ update |
| `trsv F\b` | 11.7 µs | 12.1 µs | 0.97× | every iteration |

PureOSQP's inner loop is Level-2-bound: it does a handful of `gemv` and one `trsv` per
iteration and touches `syrk`/`potrf` only when ρ changes, which on these problems is a few
times in hundreds of iterations. So the whole-solve ratio tracks `gemv` — the 1.33× and
2.00× on the Level-3 kernels contribute far less than their size suggests.

### What an earlier measurement got wrong

A previous version of this page reported PureBLAS at 0.49–0.96×, with transposed `gemv`
**10× slower** than OpenBLAS. That was real, and it was a genuine bug rather than a tuning
problem: PureBLAS's BLAS-2 SIMD path was unreachable *through `activate()`* specifically,
so every measurement taken via the libblastrampoline reroute — which is how this benchmark
runs — fell back to a scalar path. The same kernels called directly were fine. It is fixed
upstream; transposed `gemv` went from 41.6 µs to 2.46 µs, a 17× improvement, and the
whole-solve ratio from 0.50× to 1.10×.

The mistaken diagnosis is worth recording too. The obvious suspicion was per-machine
tuning: PureBLAS autotunes, and two of its seven knobs (`gemvt_percol_window`, `gemvt_pf`)
govern exactly this path. That hypothesis was wrong. Running `PureBLAS.tune!(unlocked=true)`
here — three independent calibration runs on an idle machine — pins **nothing**:
`sytrf_cmult` disagreed across runs (`[1, 2, 2]`) and every other knob tied, so the report
is "the in-code defaults are adequate here". The numbers above are therefore already the
tuned numbers, and tuning was never what stood between the two libraries.

### On the threading

The two sides are not symmetric. OpenBLAS is pinned with `BLAS.set_num_threads(1)`;
PureBLAS is plain Julia, so it is bounded by `Threads.nthreads()` — 12 in these runs —
which `BLAS.set_num_threads` does not affect. Giving OpenBLAS more threads does not change
the picture at these sizes: at n=200, m=400 it took 25.4 ms on 1 thread, 25.8 ms on 4 and
26.3 ms on 8.

## Against other solvers

Everything above compares PureOSQP with OSQP, which puts a sparse solver on dense data —
its worst case. These are three other solvers on the same dense QPs, one per algorithm
family, so the comparison is not rigged by storage format. Reproduce with
`julia --project=bench bench/solvers.jl`.

| n | m | PureOSQP | OSQP | DAQP | Clarabel |
|---|---|---|---|---|---|
| 10 | 20 | 0.140 ms | 0.155 ms | **0.003 ms** | 0.127 ms |
| 25 | 50 | 0.327 ms | 0.535 ms | **0.035 ms** | 0.653 ms |
| 50 | 100 | 1.33 ms | 2.15 ms | **0.206 ms** | 2.86 ms |
| 100 | 200 | 12.1 ms | 34.1 ms | **1.86 ms** | 17.1 ms |
| 200 | 400 | 25.2 ms | 75.8 ms | **16.0 ms** | 110 ms |
| 100 | 50 | 1.33 ms | 1.14 ms | **0.361 ms** | 6.03 ms |

DAQP is a dense active-set solver, and on these problems it is the right tool by a wide
margin — 1.6× faster than PureOSQP at the largest case and 47× at the smallest. That is
not a defect in this implementation; it is what the algorithms are. ADMM converges linearly
and needs hundreds to thousands of cheap iterations; an active-set method walks a handful
of expensive ones and terminates exactly. On small, moderately constrained dense QPs the
active-set method wins, and the gap closes as the problem grows.

What ADMM buys instead is the behaviour the rest of this page measures: iteration counts
that do not depend on the active set, warm starts that actually help, cheap re-solves
through [`update!`](@ref), and a factorization that is reused across hundreds of
iterations. If you have one dense QP to solve and no sequence, reach for DAQP.

Clarabel is an interior-point method and lands between the two, ahead at the smallest size
and behind from `n = 50` up.

Solutions agree to `1e-4` across all four, which is the expected spread at
`eps_abs = eps_rel = 1e-6`: DAQP and Clarabel terminate on exact optimality conditions
while ADMM stops at a residual tolerance.

## Matrix types

PureOSQP holds the caller's `P` and `A` and applies equilibration lazily, so every
per-iteration product runs `mul!` on whatever was passed in. The question is what that is
worth. Each row below is **one problem solved twice** — same numbers, once as a plain
`Matrix` and once in a structured type — so the iteration counts match exactly and the
difference is purely the cost of the product. Reproduce with
`julia --project=bench bench/matrix_types.jl`.

| problem | as `Matrix` | structured | speedup | iterations |
|---|---|---|---|---|
| dense `P` (vs `Symmetric`) | 66.2 ms | 66.3 ms | 1.00× | 4375 |
| diagonal `P` (vs `Diagonal`) | 13.0 ms | 12.9 ms | 1.01× | 550 |
| tridiagonal `P` (vs `Symmetric`) | 41.4 ms | 41.5 ms | 1.00× | 2550 |
| `A` as a `SubArray` | 17.9 ms | 17.8 ms | 1.00× | 900 |

**Structured storage buys nothing measurable here**, and that is worth stating plainly
rather than leaving the reader to infer a benefit from the design. The mechanism works —
a `Diagonal` `P` really does get an `O(n)` product instead of `O(n²)` — but on these
shapes `P` is not where the time goes. Each iteration also does two products with `A`,
which at `m = 2n` is four times the work of the dense `P` product, and `A` is dense in
every row above. The structure would have to be in `A` to show up, and a structurally
sparse `A` is the next section.

So the honest claim is narrower than "structured matrices are faster": the caller's
matrices are never copied or mutated, any `AbstractMatrix` works, and a cheaper `mul!` is
used when one exists — but on a dense `A` that is not a speed feature.

## Sparse A

The case the rest of this page avoids. `A` and `P` genuinely sparse, so OSQP's sparse
LDLᵀ is playing to its strength and PureOSQP has no answer but to treat them as dense.

| n | m | density | PureOSQP | OSQP | vs | iterations |
|---|---|---|---|---|---|---|
| 200 | 400 | 1% | 21.5 ms | 4.81 ms | **0.22×** | 475 |
| 200 | 400 | 5% | 31.2 ms | 24.4 ms | 0.78× | 875 |
| 200 | 400 | 20% | 31.2 ms | 37.5 ms | **1.20×** | 875 |
| 400 | 800 | 1% | 93.4 ms | 38.7 ms | **0.41×** | 550 |
| 400 | 800 | 5% | 124 ms | 110 ms | 0.88× | 900 |
| 400 | 800 | 20% | 256 ms | 382 ms | **1.49×** | 2350 |

At 1% density OSQP is 2.4–4.5× faster and the answer is simply "use OSQP". The crossover
sits near 10%: below it the sparse factorization wins, above it the dense one does. The
iteration counts are identical at every density, which is the same agreement the dense
tables show — the two solvers take the same path and only the linear algebra underneath
differs.

This is the boundary of what PureOSQP is for. It is a dense solver; if `A` is sparse and
stays sparse, the reference implementation is the better tool and this package has no
sparse backend to offer.

### Does keeping `A` sparse help?

The table above hands PureOSQP dense copies. It does not have to: any `AbstractMatrix`
works, and lazy scaling means the per-iteration products call `mul!` on whatever was
passed, so a `SparseMatrixCSC` keeps sparse products. Only the factorization buffers are
dense either way. Passing the sparse matrices straight through:

| n | m | density | densified | kept sparse | gain |
|---|---|---|---|---|---|
| 200 | 400 | 1% | 21.4 ms | 17.5 ms | 1.22× |
| 200 | 400 | 5% | 31.3 ms | 36.4 ms | 0.86× |
| 200 | 400 | 20% | 31.3 ms | 48.2 ms | 0.65× |
| 400 | 800 | 1% | 93.6 ms | 76.0 ms | 1.23× |
| 400 | 800 | 5% | 125 ms | 167 ms | 0.75× |
| 400 | 800 | 20% | 255 ms | 360 ms | 0.71× |

Only at 1% density, and only by about 1.2×. From 5% up, keeping the matrix sparse is
actively *worse* — sparse `mul!` loses to dense BLAS-2 well before the flop count says it
should, because the dense kernel streams contiguous memory while the sparse one chases
indices. So densifying at the door, which is what the main table does, is the better
default; PureOSQP just never forces it on you.

### Where the gap below 5% actually is

Splitting the same runs into setup and iterations, `n = 200`, `m = 400`:

| density | | setup | iterations | total | iters |
|---|---|---|---|---|---|
| 1% | PureOSQP | **9.43 ms** | 4.03 ms | 13.5 ms | 150 |
| | OSQP | **0.44 ms** | 1.43 ms | 1.87 ms | 150 |
| 5% | PureOSQP | 9.56 ms | 11.2 ms | 20.7 ms | 450 |
| | OSQP | 2.21 ms | 11.5 ms | 13.7 ms | 450 |
| 20% | PureOSQP | 9.53 ms | **20.1 ms** | 29.7 ms | 825 |
| | OSQP | 3.66 ms | **31.8 ms** | 35.5 ms | 825 |

**It is setup, not the iteration loop.** PureOSQP's setup is flat at ~9.5 ms across all
three densities, because it forms `AᵀρA` and factors it densely no matter how sparse `A`
was: `O(mn²) + O(n³)`, a cost that cannot see sparsity. OSQP's setup tracks the nonzero
count instead — 0.44 ms at 1%, 3.66 ms at 20%. At 1% that one difference is 9.0 ms of an
11.6 ms total gap, about 80% of it.

The iteration loop is a much smaller effect and turns over: 2.8× behind at 1%, level at
5%, and 1.6× *ahead* at 20%, where dense BLAS-2 beats sparse products on the same
iteration count. So the fix for sparse problems would be a sparse factorization, not
faster products — and that is precisely the thing this package does not have.
