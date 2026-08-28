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
| 25 | 50 | 0.337 ms | 0.390 ms | 0.86× | 225 / 225 | 4.7e-15 |
| 50 | 100 | 1.34 ms | 1.91 ms | 0.70× | 350 / 350 | 6.2e-15 |
| 100 | 200 | 12.3 ms | 24.9 ms | 0.50× | 1500 / 1500 | 1.3e-14 |
| 200 | 400 | 25.7 ms | 51.2 ms | 0.50× | 650 / 650 | 2.2e-14 |
| 100 | 50 | 1.34 ms | 1.39 ms | 0.96× | 50 / 50 | 8.7e-15 |

Measured against PureBLAS at commit `3ba7d47`, **uncalibrated** — see the note below the
breakdown, which is the main thing to know before reading the ratios.

**Correctness is exact**: identical iteration counts and solutions agreeing to `1e-14`, so
PureBLAS is a faithful drop-in. **Speed is 1.0–2.0× worse here**, and the per-operation
breakdown says precisely where it goes:

| operation (n=200, m=400) | OpenBLAS | PureBLAS | ratio | runs |
|---|---|---|---|---|
| `gemv A*x` | 3.62 µs | 4.75 µs | 0.76× | every iteration |
| **`gemv Aᵀy` (transposed)** | 4.29 µs | **42.1 µs** | **0.10×** | every iteration |
| `gemv At*y` (materialized transpose) | 3.59 µs | 5.49 µs | 0.66× | — |
| `syrk WᵀW` | 314 µs | 235 µs | **1.34×** | on a ρ update |
| `potrf` | 119 µs | 58.4 µs | **2.03×** | on a ρ update |
| `trsv F\b` | 11.7 µs | 12.2 µs | 0.96× | every iteration |

PureBLAS *wins* on the Level-3 work — `syrk` 1.34× and `potrf` 2.03× — which is what its
own benchmarks target. But PureOSQP's inner loop is Level-2-bound: it does a handful of
`gemv` and one `trsv` per iteration and touches `syrk`/`potrf` only when ρ changes, which
on these problems is a few times in hundreds of iterations. The whole-solve ratio is
therefore set by `gemv`, and specifically by the **transposed** path.

**On whether tuning would close that gap: it was tried, and it does not.** PureBLAS
autotunes per machine and two of its seven knobs — `gemvt_percol_window` and `gemvt_pf` —
govern this path, so the obvious suspicion is that an unpinned default is the whole story.
Running its calibrator says otherwise: the window knob reports *"NO window reproduces the
measured winners — the knob's
SHAPE is wrong for this box, not just its value. Reporting, not pinning"*, so it pins
nothing at all.

The gap is also not confined to the sizes PureOSQP happens to use. Measured on square `A`,
gemv-T with PureBLAS is 7.1× slower at n=64, 9.5× at n=200, 9.7× at n=1024 and 3.5× at
n=4096 — worst in the middle of the range and never better than 3.5×. The calibrator tunes
at n=512–4096, so PureOSQP's n=200 sits below the tuned range, but the gap is present
across all of it.

So the honest reading is narrower than "PureBLAS is slower": on this machine its
**transposed** gemv is several times slower than OpenBLAS across every size measured, its
non-transposed gemv is only ~1.5× slower, and its Level-3 kernels are *faster*. PureOSQP is
simply Level-2-bound and lands on the one path where that gap lives.

### On the threading

The two sides are not symmetric, and the numbers should be read knowing it. OpenBLAS is
pinned with `BLAS.set_num_threads(1)`; PureBLAS is plain Julia, so it is bounded by
`Threads.nthreads()` — 12 in the run above — which `BLAS.set_num_threads` does not affect.
The comparison is therefore tilted *toward* PureBLAS, and it still loses.

Giving OpenBLAS more threads does not change the verdict, because at these sizes the
workload does not parallelize: at n=200, m=400 it took 25.4 ms on 1 thread, 25.8 ms on 4
and 26.3 ms on 8, against PureBLAS's 51.3 ms.
