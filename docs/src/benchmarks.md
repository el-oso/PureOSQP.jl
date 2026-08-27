# Benchmarks

Reproduce with `julia --project=bench bench/headtohead.jl`. Every sample is written to
`bench/results/headtohead.json`.

## Speed, against libosqp 0.6.2

PureOSQP versus OSQP.jl 0.8.1 (libosqp 0.6.2) on dense random QPs, `eps_abs = eps_rel =
1e-6`, single-threaded BLAS, adaptive ρ pinned to a 50-iteration interval on both sides so
the comparison is deterministic.

| n | m | PureOSQP | libosqp 0.6.2 | speedup | iters (PureOSQP / C) | objective rel. Δ |
|---|---|---|---|---|---|---|
| 10 | 20 | 0.153 ms | 0.081 ms | 0.53× | 100 / 100 | 3.8e-15 |
| 25 | 50 | 0.986 ms | 1.67 ms | 1.69× | 675 / 675 | 1.4e-15 |
| 50 | 100 | 2.71 ms | 5.81 ms | 2.14× | 900 / 900 | 1.8e-15 |
| 100 | 200 | 7.90 ms | 20.9 ms | 2.65× | 850 / 850 | 1.3e-15 |
| 200 | 400 | 38.4 ms | 130 ms | 3.39× | 1175 / 1175 | 4.4e-15 |
| 400 | 800 | 134 ms | 1747 ms | 13.0× | 625 / 625 | 3.9e-15 |
| 100 | 50 | 1.63 ms | 1.17 ms | 0.72× | 50 / 50 | 4.4e-15 |
| 200 | 100 | 5.41 ms | 16.6 ms | 3.07× | 50 / 50 | 1.1e-14 |
| 100 | 1000 | 191 ms | 902 ms | 4.71× | 7475 / 7475 | 2.6e-11 |
| 200 | 2000 | 299 ms | 2044 ms | 6.84× | 3025 / 3025 | 7.1e-15 |

## Agreement, against libosqp 1.x

`OSQP.jl` wraps 0.6.2 only. `bench/headtohead_v1.jl` compares against **libosqp 1.x**
through a subprocess oracle (see `bench/oracle_v1/README.md`); it reports agreement, not
timing, because each oracle call pays interpreter startup.

| n | m | status | iters (PureOSQP / 1.x) | objective rel. Δ |
|---|---|---|---|---|
| 10 | 20 | solved / solved | 125 / 125 | 0.0 |
| 25 | 50 | solved / solved | 300 / 300 | 1.5e-15 |
| 50 | 100 | solved / solved | 1400 / 1400 | 8.1e-16 |
| 100 | 200 | solved / solved | 2625 / 2625 | 2.5e-15 |
| 200 | 400 | solved / solved | 700 / 700 | 3.4e-15 |
| 100 | 50 | solved / solved | 50 / 50 | 5.1e-16 |
| 50 | 500 | solved / solved | 1875 / 1875 | 2.4e-13 |

Identical iteration counts against both the version this port was transcribed from and the
current one.

## How to read this

**What the matching iteration counts do and do not show.** They mean the equilibration, the
ρ schedule and the termination tests track the reference implementation, which is the part
of OSQP's behavior a port most easily gets subtly wrong. But the count is *quantized*: with
`check_termination = 25`, matching counts say both solvers crossed the threshold within the
same 25-iteration window, not that the iterates agree. The evidence that they agree
iterate-by-iterate is separate — the transcription test, which matches to `1e-10` over the
first 25 iterations with adaptive ρ disabled — and the evidence about the returned `y` is
separate again, since the objective does not contain `y`. That one comes from the referee.

The corpus behind these numbers is feasible, bounded, cold-started, two-sided and dense.
Infeasibility detection, warm starting, polishing, `±Inf` bounds, equality rows and
degenerate `P` are covered in the test suite, not here.

**The timings are not a claim that this solver is better than OSQP.** These are dense
problems, which is the reference implementation's worst case: it is a *sparse* solver being
handed dense matrices, so it pays sparse-format overhead and scalar sparse LDLᵀ where
PureOSQP gets BLAS-3 dense Cholesky. The right reading is a *storage-format* comparison,
not a solver-quality one. On genuinely sparse problems the reference implementation is
expected to win, and PureOSQP has no answer for a sparse `A` beyond treating it as dense.
PureOSQP already loses here on the smallest problems (0.53× at `n = 10`), where setup
dominates, and at `n = 100, m = 50`, where the reduced system is barely smaller than the
full one.

## Linear-system backend

`bench/kkt_backend.jl` reproduces the cost and accuracy comparison between the reduced
Cholesky and the full-KKT Bunch-Kaufman factorization described under
[Algorithm](@ref "The linear system"), including the near-parallel-row family that
equilibration cannot fix.

## Sequential re-solves

`P` and `A` fixed, `q`, `l` and `u` changing every step — the receding-horizon loop OSQP is
most used for. 20 solves per case, `eps_abs = eps_rel = 1e-6`. libosqp 1.x is timed inside
its own process, so interpreter startup and the JSON round trip are excluded. Reproduce
with `julia --project=bench bench/update_bench.jl`.

| n | m | `update!` | fresh `setup` each step | saved | libosqp 1.x | ratio | factorizations | objective Δ |
|---|---|---|---|---|---|---|---|---|
| 10 | 20 | 2.00 ms | 2.66 ms | 1.33× | 1.82 ms | **0.91×** | 8 | 1.5e-14 |
| 25 | 50 | 5.38 ms | 11.3 ms | 2.10× | 9.77 ms | 1.82× | 14 | 4.3e-12 |
| 50 | 100 | 26.1 ms | 45.3 ms | 1.73× | 59.3 ms | 2.27× | 7 | 7.6e-14 |
| 100 | 200 | 157 ms | 180 ms | 1.15× | 491 ms | 3.13× | 3 | 3.9e-13 |
| 200 | 400 | 634 ms | 803 ms | 1.27× | 2420 ms | 3.81× | 1 | 1.9e-15 |

"Factorizations" counts the whole 20-step sequence: one from `setup`, plus one per step
whose constraint classification changed.

Two things worth reading off this. The saving from `update!` is bounded by how much of the
run is setup: at `n = 200` the ADMM iterations dominate, so skipping 19 of 20
factorizations is worth only 1.27×, while at `n = 25` — short solves, setup a large share —
it is 2.10×. And the factorization count is not always 1: equilibration rescales the bounds,
so a row whose scaled gap `ũ - l̃` falls under `RHO_TOL` is reclassified as an equality and
its `ρ` changes. That is upstream's rule, applied to the scaled bounds exactly as upstream
applies it, and it means `update!` is not unconditionally factorization-free.

PureOSQP is slower than libosqp 1.x on the smallest case here, as it is in the single-solve
comparison, for the same reason: at `n = 10` the per-solve overhead dominates and there is
no dense-linear-algebra advantage to collect.
