# PureOSQP.jl — design spec

Pure-Julia, stdlib-only implementation of the OSQP operator-splitting QP solver, for
dense and structured `AbstractMatrix` data. A port of the algorithm described
in Stellato, Banjac, Goulart, Bemporad & Boyd, *OSQP: an operator splitting solver for
quadratic programs*, Math. Prog. Comp. 12(4):637–672, 2020, transcribed against the
Apache-2.0 reference implementation.

## Problem

    minimize    ½ xᵀPx + qᵀx
    subject to  l ≤ Ax ≤ u

`P` is n×n symmetric positive semidefinite, `A` is m×n, and `l`, `u` may contain ∓Inf.
Equality constraints are rows with `l == u`.

## Scope of the port

| feature | v0.1 | rationale |
|---|---|---|
| ADMM iteration with over-relaxation α | yes | the algorithm |
| modified Ruiz equilibration (D, E, c) | yes | without it the solver loses every head-to-head; also the only thing that makes the reduced linear solve usable (measured below) |
| adaptive ρ, vector-valued, equality/inequality split | yes | load-bearing for iteration count |
| primal and dual infeasibility certificates | yes | fail-fast: the alternative is a silent spin to `max_iter` on infeasible input |
| warm starting | yes | nearly free |
| solution polishing | yes | OSQP's default tolerance is 1e-3; polish is how a caller gets more |
| duality-gap termination check | no | added in libosqp 1.x; absent from the 0.6.2 oracle this port is validated against |
| MathOptInterface wrapper | no | separate concern, additive |
| matrix-free / indirect (CG) inner solve | no | needs an inner-tolerance schedule that interacts with adaptive ρ |

## Linear-system backend

Each ADMM step solves

    ⎡P + σI      Aᵀ    ⎤ ⎡x̃⎤   ⎡σx − q      ⎤
    ⎣A       −diag(ρ⁻¹)⎦ ⎣ν⎦ = ⎣z − ρ⁻¹⊙y   ⎦

Upstream forms this (n+m)×(n+m) quasi-definite matrix and factors it with a sparse
pivot-free LDLᵀ. Eliminating ν analytically gives the equivalent reduced system

    (P + σI + Aᵀ diag(ρ) A) x̃ = σx − q + Aᵀ(ρ⊙z − y),     z̃ = A x̃

which is symmetric positive definite for σ > 0, ρ > 0, P ⪰ 0, so `cholesky!` applies.

**Decision: the reduced system is the default; the full KKT with `bunchkaufman!` is
selectable via `linsys = :kkt` and is also the automatic fallback if the Cholesky reports
the matrix is not positive definite.** Both halves are measured, not assumed
(`bench/kkt_backend.jl`).

Cost, dense, single-threaded, one factorization:

| n | m | full KKT `bunchkaufman!` | reduced `cholesky!` | ratio |
|---|---|---|---|---|
| 50 | 50 | 2.99e-5 s | 9.67e-6 s | 3.09 |
| 50 | 500 | 1.48e-3 s | 4.22e-5 s | 34.9 |
| 200 | 2000 | 7.07e-2 s | 1.63e-3 s | 43.3 |
| 500 | 100 | 2.36e-3 s | 1.62e-3 s | 1.45 |

The reduced form wins in every dense regime measured, including m < n.

Accuracy is the counter-pressure: forming `AᵀρA` squares the conditioning of `A`. Relative
error of the inner solve against a `BigFloat` reference, n=60, m=200, P=0:

| cond(A) | reduced, unscaled | reduced, equilibrated | full KKT |
|---|---|---|---|
| 1e6 | 1.1e-05 | 4.5e-09 | 8.8e-12 |
| 1e8 | 9.5e-02 | 8.0e-09 | 3.1e-10 |
| 1e10 | fails | 4.8e-09 | 3.2e-08 |
| 1e16 | fails | 1.2e-08 | 3.4e-02 |

**This measurement revised the design.** The original plan claimed the Cholesky "fails
loudly past cond(A) ≈ 1e9", justifying the fallback as the mechanism that handles
ill-conditioning. That is only true *unscaled*. With equilibration on — the default — the
reduced solve holds around 1e-8 across the whole range and was never observed to fail,
including on the harder family of near-parallel rows that no diagonal scaling can fix
(2.7e-8 at cond(A) = 3.4e10, Cholesky succeeding). Past cond(A) ≈ 1e10 it is *more*
accurate than the full KKT factorization.

Consequences: equilibration is load-bearing, not optional. The `PosDefException` fallback
is a safety net that measurement never triggers, so it cannot be relied on as tested
insurance — which is why `linsys` was made a user-facing setting and the entire structural
corpus runs through both backends.
## Genericity

`P` and `A` are accepted as any `AbstractMatrix` and are **never mutated**. Ruiz
equilibration is stored as factors `D`, `E`, `c` and applied lazily, so every
per-iteration product runs through `mul!` on the caller's original matrix:

    Ãx  = E ⊙ (A (D ⊙ x))
    Ãᵀy = D ⊙ (Aᵀ(E ⊙ y))
    P̃x  = c (D ⊙ (P (D ⊙ x)))

A structured or lazy `A` therefore keeps its fast product in the hot loop. Only the
direct solve materializes dense buffers: an m×n scaled copy of `A` and the n×n reduced
matrix, both rebuilt on a ρ update. This is inherent to a direct method, not an
oversight.

There is no factorization *interface*. `refactor!(ws)` and the `ldiv!` at its call sites
are the seam; if a structured-operator or matrix-free backend is ever wanted, it is a
dispatch change to those two functions.

## Numerical types

Everything is generic over `T<:AbstractFloat`, taken from `eltype`. All constants are
`convert`ed to `T`. Struct fields are concrete; workspace types are parametric in the
matrix types.

## Reference version

Two oracles are used.

**libosqp 0.6.2**, via `OSQP.jl` 0.8.1 — the newest Julia wrapper in General. This is what
the port was transcribed against and what the test suite compares to.

**libosqp 1.x**, via a subprocess oracle (`bench/oracle_v1/`). There is no Julia wrapper;
`OSQP_jll` v100 ships the library but its `OSQPSettings` is `__attribute__((packed))` with
a build-dependent integer and float width, which is not safe to mirror for `ccall` by hand.
`bench/headtohead_v1.jl` therefore drives the `osqp` Python wheel as a subprocess speaking
JSON. There is no in-process interop and the package itself has no Python dependency.

The ADMM steps, Ruiz equilibration, residuals, tolerances and polishing are byte-identical
between 0.6.2 and master. Two deliberate deviations from 0.6.2:

1. **Primal-infeasibility threshold.** 0.6.2 accepts the certificate when
   `uᵀ max(δy,0) + lᵀ min(δy,0) < ε‖δy‖`; master tightened this to `< 0`. PureOSQP uses the
   0.6.2 form. The dual-infeasibility test likewise uses 0.6.2's
   `qᵀδx < c·ε·‖δx‖` rather than master's `qᵀδx < 0`.
2. **ρ-adaptation trigger.** 0.6.2 defaults to *time-based* adaptation (fire once 0.4 ×
   setup time has elapsed). That makes iteration counts depend on machine speed and is
   untestable. PureOSQP adapts on a fixed iteration interval (default 50, master's
   default), and every head-to-head pins the oracle to `adaptive_rho_interval = 50`.

## Correctness contract

Iterate-by-iterate agreement over a full run is not a target: hundreds of ADMM iterations
amplify roundoff. The contract is four-tiered.

1. **Transcription gate.** Iterates match the C library to 1e-10 over the first 25
   iterations, across both linear-system backends and both scaled and unscaled space, with
   adaptive ρ disabled. This catches sign, index and α-relaxation errors where they are
   cheapest to find.
2. **Ported C suite.** OSQP's own tests — `basic_qp` (solve, check_termination, update_rho,
   warm_start), `basic_qp2`, `primal_dual_infeasibility` (four variants), `unconstrained`,
   `non_cvx` — with upstream's data, expected solutions, per-test settings and `TESTS_TOL`.
3. **Acceptance gate.** An independent referee computes KKT residuals and the duality gap
   at the returned `(x, y)` **from the caller's original `P, q, A, l, u`**, never through a
   solver-internal scaled quantity. It runs on every corpus instance, and is itself
   calibrated by a permanent test that runs the C library's own solution through it.
4. **Status and certificates.** Status must match the C library on infeasible instances,
   and the certificate is checked against its defining inequalities rather than against
   C's certificate vector, which is not unique.

Iteration-count parity is reported and asserted against both 0.6.2 and 1.x, but read with
the caveat that the count is quantized by `check_termination`.

Corpus: rank-deficient `P`, `P = 0` (an LP), `l == u` rows, fixed variables, one- and
two-sided `±Inf` bounds, free rows, m < n, m ≫ n, m = 0, n = 1, diagonal and `Symmetric`
`P`, `A` as a `SubArray` and as a `SparseMatrixCSC`, and `Float16`/`Float32`/`Float64`.
Maros–Mészáros instances are **not** included; that remains open.

## Known failure mode

The solver iterates in Ruiz-scaled space but must report residuals, adapt ρ, terminate and
emit certificates in problem space. Each crossing can silently converge to the wrong
tolerance while every easy test passes. The referee of gate 3 exists to catch this.

The referee itself proved to be the first thing that went wrong this way: it originally
scored complementarity as "`y_i ≠ 0` implies the bound is active", which at a first-order
solution reports full-magnitude violations on correct answers. It was caught only by
running the C library's own solution through it — 23 of 30 failures, against 24 of 30 for
PureOSQP. The general rule that follows: **a predicate quantized from a float is not a
check; residuals and gaps are**, and an oracle is untrusted until it has judged a
known-good answer.

Secondary: `Inf` bound arithmetic in scaling and in the equality/inequality ρ
classification. Input validation rejects `NaN`, non-finite `q`, non-symmetric `P`, `l > u`,
`l = +Inf`, `u = -Inf`, an indefinite `P`, and out-of-range settings, and fails rather than
continuing.

## Relationship to PureLSQP.jl

Complementary, not overlapping. PureLSQP is a high-accuracy active-set solver for a
specific bound-constrained least-squares structure under a hard latency budget; PureOSQP is
a general `l ≤ Ax ≤ u` first-order method at moderate accuracy. No shared code.

## License

Apache-2.0, matching upstream OSQP. This is a from-scratch Julia implementation written
against the paper and against the Apache-2.0 reference implementation, which was read for
the details the paper omits — a derivation of that work, not an independent reinvention.
