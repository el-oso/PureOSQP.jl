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
