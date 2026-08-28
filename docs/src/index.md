# PureOSQP.jl

A pure-Julia implementation of the OSQP operator-splitting solver for convex quadratic
programs:

```math
\begin{aligned}
\text{minimize}   \quad & \tfrac12 x^\top P x + q^\top x \\
\text{subject to} \quad & l \le A x \le u
\end{aligned}
```

`P` is symmetric positive semidefinite, `A` is `m×n`, and `l`, `u` may contain `∓Inf`.
Rows with `l == u` are equality constraints.

## What makes this different

`P` and `A` may be **any `AbstractMatrix`** and are never copied or modified. There is no
sparse linear algebra — a sparse `A` is factored densely, though a `SparseArrays` weak
dependency lets equilibration walk only the stored entries when you pass one. The numerics
use `LinearAlgebra` alone, and the only
other dependency is TypeContracts.jl, which declares the linear-system backend interface
and checks it at precompilation. The solver is
built for dense and structured data, where the reference implementation's sparse machinery
is a liability rather than a help.

## Quick start

```julia
using PureOSQP

P = [4.0 1.0; 1.0 2.0]
q = [1.0, 1.0]
A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
l = [1.0, 0.0, 0.0]
u = [1.0, 0.7, 0.7]

sol = solve(P, q, A, l, u)

sol.status    # SOLVED
sol.x         # [0.3, 0.7]
sol.obj_val   # 1.88
```

For repeated solves, build the workspace once and reuse it — the factorization and all
buffers are retained, and the previous iterates warm-start the next solve:

```julia
ws = setup(P, q, A, l, u; eps_abs = 1e-8, eps_rel = 1e-8, polish = true)
sol = solve!(ws)
sol = solve!(ws)     # warm started from the previous solution
```

To seed from a solution you already have:

```julia
warm_start!(ws; x = x0, y = y0)
solve!(ws)
```

## Re-solving with new data

A receding-horizon or sequential-quadratic loop keeps `P` and `A` fixed and changes `q`,
`l` and `u` every step. Use [`update!`](@ref) rather than building a new workspace: it
reuses the equilibration factors, the buffers and the current iterates, and refactorizes
only when it has to.

```julia
ws = setup(P, q, A, l, u)
for step in 1:horizon
    update!(ws; q = q_k, l = l_k, u = u_k)
    sol = solve!(ws)          # warm started from the previous step
end
```

A change to `q` alone never refactorizes. A change to `l` or `u` refactorizes only if it
moves a row between the equality, inequality and free classes, since that is what sets `ρ`.
A change to `P` or `A` always does.

`ws.refactor_count` reports how many factorizations the workspace has done over its whole
life. Note that it counts factorizations from **adaptive ρ** as well, so it will keep
growing across a loop even when every `update!` is factorization-free; read it immediately
before and after a call to see what that call did.

Equilibration is not recomputed — the factors from `setup` are reused, as upstream does.
They stay appropriate while the data keeps roughly the same scale; after a large change in
magnitude, build a fresh workspace.

## Accuracy

The default tolerances are `eps_abs = eps_rel = 1e-3`, matching the reference
implementation. That is loose for many uses. Two ways to tighten it:

- lower `eps_abs`/`eps_rel`, which costs iterations;
- set `polish = true`, which guesses the active set at the ADMM solution and solves the
  resulting equality-constrained QP exactly. On a solution converged only to `1e-3`,
  polishing typically brings the KKT residual to around machine precision at the cost of
  one extra factorization.

Polishing is only ever accepted when it improves both residuals, so it cannot make an
answer worse.

## Choosing a linear-system backend

`linsys = :auto` (default) takes an `n×n` Cholesky of the reduced system and inverts it in
place, so each iteration's solve is one `symv`; it falls back to a Bunch-Kaufman
factorization of the full `(n+m)×(n+m)` quasi-definite system if the Cholesky reports the
matrix is not positive definite. `linsys = :kkt` forces the full factorization: slower, but
more accurate at moderate conditioning and closer to what the reference implementation
does, which makes it useful when a result is in question.

## Watching a solve

`verbose = true` prints a progress report: a header with the problem size and the settings
that matter, one row per termination check, and a footer with the status, iteration count
and final residuals.

```
 iter      objective      prim res      dual res           rho
   25        1.39217        0.0543       0.00297           0.1
   50        1.45711       0.00245      0.000415           0.1
  ...
  125        1.46211      0.000374      0.000174         0.549
```

The `rho` column is the useful one: a change there is an adaptive-ρ update, which is also
a refactorization. Rows appear at the `check_termination` interval, so setting that to `0`
disables them along with the termination tests themselves.

Output goes to `Core.stdout` rather than `Base.stdout`, because the abstractly typed
global does not survive `--trim`. `redirect_stdout` still captures it.

## What a solve reports

Beyond `x`, `y` and `status`, [`Solution`](@ref) carries:

| field | meaning |
|---|---|
| `obj_val`, `dual_obj_val` | primal and dual objectives; equal at an exact solution |
| `duality_gap` | `xᵀPx + qᵀx + SC(y)`, zero at an exact solution, reported unscaled |
| `prim_res`, `dual_res` | the two residuals, in problem space |
| `rel_kkt_error` | the largest of the residuals and the gap — one number for "how far from optimal" |
| `iter`, `rho_estimate`, `rho_updates` | iterations, the last `ρ` the residuals implied, and how many times `ρ` actually moved |
| `status_polish` | which of the five polishing outcomes occurred; `polished` is the narrower `POLISH_SUCCESS` |
| `setup_time`, `solve_time`, `polish_time`, `run_time` | seconds |

`rho_updates` counts only adaptive-`ρ` changes, where `ws.refactor_count` also counts
refactorizations forced by new data — read the former to understand a solve, the latter to
understand a workspace's whole life.

## Status values

| status | meaning |
|---|---|
| `SOLVED` | primal and dual residuals are below tolerance |
| `SOLVED_INACCURATE` | the iteration limit was hit, but the residuals meet ten times the requested tolerances |
| `PRIMAL_INFEASIBLE` | a certificate of primal infeasibility was found; `sol.prim_inf_cert` holds it |
| `DUAL_INFEASIBLE` | a certificate of dual infeasibility was found; `sol.dual_inf_cert` holds it |
| `PRIMAL_INFEASIBLE_INACCURATE`, `DUAL_INFEASIBLE_INACCURATE` | as above, found only at ten times the requested tolerances |
| `MAX_ITER_REACHED` | the iteration limit was hit and even the relaxed check failed |
| `TIME_LIMIT_REACHED` | `time_limit` was spent before the residuals converged |
| `NON_CONVEX` | residuals diverged |

`time_limit` bounds the ADMM loop and defaults to `Inf`. It measures the loop only —
equilibration and the first factorization happen in `setup` and are not counted — so a
`solve` on fresh data takes longer than the limit by however long setup ran. The budget is
checked every iteration, and the status is returned as soon as it is spent without
re-testing the tolerances, so a run can report `TIME_LIMIT_REACHED` at a point that would
have passed. Setting it makes the iteration count depend on the machine, which is why it
is off by default.

An unconverged result is never reported as `SOLVED`. Whenever there is no meaningful
primal-dual point — an infeasibility, or `NON_CONVEX` — `x` and `y` are filled with `NaN`
rather than a plausible-looking number.

## What is rejected rather than accepted

`setup` fails, rather than continuing, on: a non-symmetric `P`; an indefinite `P`, meaning
`P + σI` is not positive definite (this is upstream's condition, and a positive
semidefinite `P` always passes); `NaN` or `Inf` in `q`; `NaN` in `l` or `u`; `l > u`
elementwise; `l = +Inf` or `u = -Inf`; mismatched dimensions; and out-of-range settings
such as `sigma ≤ 0`, `alpha ∉ (0, 2)` or `max_iter ≤ 0`.

## Reference

Stellato, Banjac, Goulart, Bemporad and Boyd, *OSQP: an operator splitting solver for
quadratic programs*, Mathematical Programming Computation 12(4):637–672, 2020.
