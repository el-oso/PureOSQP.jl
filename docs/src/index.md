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
sparse-matrix dependency anywhere: the only dependency is `LinearAlgebra`. The solver is
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

`linsys = :auto` (default) uses an `n×n` Cholesky of the reduced system, falling back to a
Bunch-Kaufman factorization of the full `(n+m)×(n+m)` quasi-definite system if the Cholesky
reports the matrix is not positive definite. `linsys = :kkt` forces the full
factorization: slower, but more accurate at moderate conditioning and closer to what the
reference implementation does, which makes it useful when a result is in question.

## Status values

| status | meaning |
|---|---|
| `SOLVED` | primal and dual residuals are below tolerance |
| `SOLVED_INACCURATE` | the iteration limit was hit, but the residuals meet ten times the requested tolerances |
| `PRIMAL_INFEASIBLE` | a certificate of primal infeasibility was found; `sol.prim_inf_cert` holds it |
| `DUAL_INFEASIBLE` | a certificate of dual infeasibility was found; `sol.dual_inf_cert` holds it |
| `PRIMAL_INFEASIBLE_INACCURATE`, `DUAL_INFEASIBLE_INACCURATE` | as above, found only at ten times the requested tolerances |
| `MAX_ITER_REACHED` | the iteration limit was hit and even the relaxed check failed |
| `NON_CONVEX` | residuals diverged |

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
