# Benchmarks

Reproduce with `julia --project=bench bench/headtohead.jl`. Every sample is written to
`bench/results/headtohead.json`.

## Against both reference versions

Dense random QPs, `eps_abs = eps_rel = 1e-6`, single-threaded BLAS, adaptive ρ pinned to a
50-iteration interval everywhere so all three are deterministic.

libosqp 0.6.2 is timed through OSQP.jl with BenchmarkTools. libosqp 1.x has no Julia
wrapper, so it is timed **inside** the subprocess oracle (`bench/oracle_v1/`) around the
same `setup + solve` span, which excludes interpreter startup and the JSON round trip.

| n | m | PureOSQP | libosqp 0.6.2 | vs | libosqp 1.x | vs | iterations (all three) | objective rel. Δ |
|---|---|---|---|---|---|---|---|---|
| 10 | 20 | 0.060 ms | 0.070 ms | 1.16× | 0.376 ms | 6.22× | 100 | 3.8e-15 |
| 25 | 50 | 0.842 ms | 1.51 ms | 1.79× | 1.86 ms | 2.20× | 675 | 1.4e-15 |
| 50 | 100 | 2.85 ms | 5.51 ms | 1.93× | 5.70 ms | 2.00× | 900 | 1.8e-15 |
| 100 | 200 | 8.47 ms | 22.4 ms | 2.64× | 22.3 ms | 2.63× | 850 | 1.3e-15 |
| 200 | 400 | 42.1 ms | 141 ms | 3.35× | 128 ms | 3.05× | 1175 | 4.4e-15 |
| 400 | 800 | 107 ms | 675 ms | 6.29× | 630 ms | 5.87× | 625 | 4.1e-15 |
| 100 | 50 | 1.36 ms | 1.27 ms | **0.94×** | 1.73 ms | 1.27× | 50 | 4.4e-15 |
| 200 | 100 | 5.96 ms | 7.83 ms | 1.31× | 7.34 ms | 1.23× | 50 | 1.1e-14 |
| 100 | 1000 | 173 ms | 764 ms | 4.43× | 718 ms | 4.16× | 7475 | 2.6e-11 |
| 200 | 2000 | 336 ms | 1445 ms | 4.31× | 1314 ms | 3.91× | 3025 | 7.1e-15 |

**The iteration counts are identical across all three implementations, in every case.**
That is the result worth caring about: the equilibration, the ρ schedule and the
termination tests reproduce the reference exactly, not approximately, and against both the
version this port was transcribed from and the current one. The objective agrees to about
`1e-15`.

Two caveats on the numbers themselves. At `n = 10` the 1.x column is dominated by SciPy
matrix construction inside the timed span, not by libosqp — it overstates 1.x there; the
0.6.2 column is the fair small-problem comparison. And PureOSQP is genuinely slower than
0.6.2 at `n = 100, m = 50`, where the reduced system is barely smaller than the full one.

## How to read this

**The timings are not a claim that this solver is better than OSQP.** These are dense
problems, which is the reference implementation's worst case: it is a *sparse* solver being
handed dense matrices, so it pays sparse-format overhead and scalar sparse LDLᵀ where
PureOSQP gets BLAS-3 dense Cholesky. The right reading is a *storage-format* comparison,
not a solver-quality one. On genuinely sparse problems the reference implementation is
expected to win, and PureOSQP has no answer for a sparse `A` beyond treating it as dense.

**What the matching iteration counts do and do not show.** They mean the algorithm's
control logic tracks the reference. But the count is *quantized*: with
`check_termination = 25`, matching counts say all three crossed the threshold within the
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
| 10 | 20 | 1.54 ms | 2.16 ms | 1.40× | 2.17 ms | 1.41× | 8 |
| 25 | 50 | 5.29 ms | 11.5 ms | 2.16× | 11.4 ms | 2.15× | 14 |
| 50 | 100 | 29.5 ms | 50.6 ms | 1.71× | 62.6 ms | 2.12× | 7 |
| 100 | 200 | 182 ms | 201 ms | 1.11× | 546 ms | 3.00× | 3 |
| 200 | 400 | 742 ms | 908 ms | 1.22× | 2687 ms | 3.62× | 1 |

"Factorizations" counts the whole 20-step sequence: one from `setup`, plus one per step
whose constraint classification changed.

Two things worth reading off this. The saving from `update!` is bounded by how much of the
run is setup: at `n = 200` the ADMM iterations dominate, so skipping 19 of 20
factorizations is worth only 1.22×, while at `n = 25` — short solves, setup a large share —
it is 2.16×. And the factorization count is not always 1: equilibration rescales the bounds,
so a row whose scaled gap `ũ - l̃` falls under `RHO_TOL` is reclassified as an equality and
its `ρ` changes. That is upstream's rule applied to the scaled bounds exactly as upstream
applies it, and it means `update!` is not unconditionally factorization-free.

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

| n | m | OpenBLAS | PureBLAS | ratio | iterations | \|Δx\| |
|---|---|---|---|---|---|---|
| 25 | 50 | 0.318 ms | 0.286 ms | **1.11×** | 225 / 225 | 4.8e-15 |
| 50 | 100 | 1.31 ms | 1.23 ms | **1.06×** | 350 / 350 | 4.0e-15 |
| 100 | 200 | 12.2 ms | 11.5 ms | **1.06×** | 1500 / 1500 | 8.7e-15 |
| 200 | 400 | 25.1 ms | 23.6 ms | **1.06×** | 650 / 650 | 1.1e-14 |
| 100 | 50 | 1.33 ms | 1.30 ms | **1.02×** | 50 / 50 | 8.1e-15 |

**Correctness is exact** — identical iteration counts and solutions agreeing to `1e-14`, so
PureBLAS is a faithful drop-in — **and it is faster than OpenBLAS on every case**. The
per-operation breakdown shows where that comes from:

| operation (n=200, m=400) | OpenBLAS | PureBLAS | ratio | runs |
|---|---|---|---|---|
| `gemv A*x` | 3.32 µs | 3.14 µs | 1.06× | every iteration |
| `gemv Aᵀy` (transposed) | 3.90 µs | 2.48 µs | **1.57×** | every iteration |
| `gemv At*y` (materialized transpose) | 3.24 µs | 2.92 µs | 1.11× | — |
| `syrk WᵀW` | 311 µs | 233 µs | **1.33×** | on a ρ update |
| `potrf` | 115 µs | 57.6 µs | **2.00×** | on a ρ update |
| `trsv F\b` | 11.5 µs | 12.3 µs | 0.93× | every iteration |

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
upstream; transposed `gemv` went from 41.6 µs to 2.48 µs, a 17× improvement, and the
whole-solve ratio from 0.50× to 1.06×.

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
