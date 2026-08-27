# PureOSQP.jl — implementation status

Plan: `2026-08-27-pureosqp.md`. Spec: `../specs/2026-08-27-pureosqp-design.md`.

All ten tasks implemented, plus `update!`. `Pkg.test()`: **478 passed, 0 failed**.
`runic` clean. ~1115 lines of source, ~950 of tests.

## Comparison against OSQP — the deliverable

Two oracles, because `OSQP.jl` wraps only libosqp 0.6.2:

| | oracle | what it checks |
|---|---|---|
| `test/oracle_tests.jl`, `test/corpus_tests.jl` | OSQP.jl 0.8.1 → **libosqp 0.6.2** | iterates, objective, iteration count, status |
| `test/c_suite_tests.jl` | upstream's own C tests, ported | `basic_qp`, `basic_qp2`, `primal_dual_infeasibility`, `unconstrained`, `non_cvx` |
| `bench/headtohead.jl` | libosqp 0.6.2 | wall clock and iteration count |
| `bench/headtohead_v1.jl` | **libosqp 1.x**, subprocess oracle | status, iteration count, objective |
| `bench/update_bench.jl` | **libosqp 1.x**, subprocess oracle | sequential re-solve wall clock |

Results: iteration counts **identical to both 0.6.2 (10/10 cases) and 1.x (7/7)**; objective
agrees to ~1e-15; the first 25 iterates match to 1e-10 across both linear-system backends
and both scaled and unscaled space.

## `update!`

Added after review flagged it as the most likely first request. `update!(ws; q, l, u, P, A)`
reuses the equilibration factors, buffers and iterates, and refactorizes only when it must:
never for `q`; for `l`/`u` only when a row changes constraint class; always for `P`/`A`.
`ws.refactor_count` makes that observable, and the tests assert it.

Measured over 20-step sequences (`bench/update_bench.jl`): **1.15–2.10×** saved versus a
fresh `setup` per step, and **0.91–3.81×** versus libosqp 1.x. The saving is bounded by
setup's share of the run, so it is largest where solves are short. Note the factorization
count is not always 1 — equilibration rescales the bounds, so a row whose scaled gap falls
under `RHO_TOL` is reclassified as an equality, which changes its `ρ`.

## What review changed

Three reviews ran: a design consult before implementation, an adversarial design review,
and an adversarial code audit against the C source.

**Measurement overrode the design in one place.** The plan justified the Bunch-Kaufman
fallback as the mechanism handling ill-conditioning ("the Cholesky fails past cond(A) ≈
1e9"). That holds only *unscaled*. Equilibrated, the reduced Cholesky holds ~1e-8 out to
cond(A) = 1e16 and was never observed to fail — including on near-parallel rows, the family
no diagonal scaling fixes. So the fallback is a safety net measurement never triggers.
`linsys` became a user-facing setting and the corpus now runs through both backends, rather
than leaving an untested path in the code.

**Bugs found and fixed:** an indefinite `P` was accepted and a non-optimal point returned as
`SOLVED` (14/20 random indefinite instances); no settings validation at all (`max_iter = 0`
returned `SOLVED` with `iter = 0`); no symmetry check on `P`; `NON_CONVEX` returned finite
garbage instead of `NaN`; `INFTY` overflowed to `Inf` for `Float16`, disabling all free-row
handling; no cold start after an infeasible solve; the dual-infeasibility test used master's
threshold, not 0.6.2's; a missing `SOLVED_INACCURATE` tier; `warm_start!` multiplied `y` by
`E` where the unscaling divides by it.

**The referee was wrong first.** It scored complementarity as "`y_i ≠ 0` implies the bound is
active", which at a first-order solution reports full-magnitude violations on correct
answers. Caught only by running the C library's own solution through it: 23/30 failures,
against 24/30 for PureOSQP. Replaced with the duality gap plus a sign residual on rows with
an infinite bound — no deadzone. That calibration is now a permanent test.

**Two tests were silently vacuous** — a damaged `@testitem` header and a manually forced
backend that the next refactorization discarded. Both were invisible because the suite still
reported "passed", just with fewer items. `test/meta_tests.jl` now asserts the test inventory
and that every test file parses.

## Not implemented

MathOptInterface wrapper; matrix-free / indirect (CG) inner solve; the duality-gap
termination check from libosqp 1.x; a sparse linear-algebra backend; Maros–Mészáros
regression instances.
