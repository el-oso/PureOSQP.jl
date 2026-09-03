# PureOSQP.jl

A pure-Julia implementation of the [OSQP](https://osqp.org) solver for convex quadratic programs:

```math
\begin{aligned}
\text{minimize}   \quad & \tfrac12 x^\top P x + q^\top x \\
\text{subject to} \quad & l \le A x \le u
\end{aligned}
```

`P` is symmetric positive semidefinite, `A` is `m×n`, and `l`, `u` may contain `∓Inf`. Rows where `l == u` are equality constraints.

## Your first solve

Install it, describe the problem with five arrays, and call [`solve`](@ref):

```@example first
using PureOSQP

# minimize  x₁² + x₂² - x₁ - 2x₂     subject to   x₁ + x₂ ≤ 1,  x ≥ 0
P = [2.0 0.0; 0.0 2.0]      # the quadratic term, as ½xᵀPx -- note the 2s
q = [-1.0, -2.0]            # the linear term
A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
l = [-Inf, 0.0, 0.0]        # lower bounds
u = [1.0, Inf, Inf]         # upper bounds

sol = solve(P, q, A, l, u)
(sol.status, round.(sol.x; digits = 4))
```

`sol.status` is `SOLVED`, and `sol.x` is the answer. The interface is minimal: there is no model object or configuration.

Three things to know:
* **The $P$ matrix:** It represents the $\frac{1}{2}x^\top P x$ term, so a plain $x_1^2 + x_2^2$ objective needs $2$s on the diagonal.
* **Constraints:** Every constraint is a row of $l \leq Ax \leq u$. Use $l = u$ for equalities, and $\pm\infty$ for one-sided constraints.
* **Setup:** `solve` handles everything at once. For many similar problems, use [`setup`](@ref) to build a workspace once and reuse it.

For more, see [Examples](@ref "Building a workspace once") or the implementation details below.

## What makes this different

We support any `AbstractMatrix` (dense, sparse, structured, or lazy) for any `Real` type. `P` and `A` are held by reference and never modified; every iteration calls `mul!` on your input. This means types like `Diagonal` or `SparseMatrixCSC` keep their fast product instead of being copied to a dense matrix.

`P` and `A` are held by reference and never copied or modified. Every per-iteration product
runs `mul!` on the matrix you passed, so a `Diagonal`, a `Tridiagonal`, a `SubArray` or a
`SparseMatrixCSC` keeps its own product instead of being flattened into a dense copy.
Equilibration reaches the entries through four overridable column traversals, and a
`SparseArrays` weak dependency specialises them to walk only the stored entries.

One matrix in the solver is always dense: the $n \times n$ reduced system formed by eliminating $\nu$. This is a property of the reduction, not a limit on your input. If you have a dense row in $A$, the solver uses a sparse factorization of the full KKT system to avoid squaring the matrix.

The code uses `LinearAlgebra` and `TypeContracts.jl` for the linear-system backend interface.

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

To start from a known solution:

```julia
warm_start!(ws; x = x0, y = y0)
solve!(ws)
```

## Re-solving with new data

For loops like Model Predictive Control, keep $P$ and $A$ fixed and update $q$, $l$, and $u$. Use [`update!`](@ref) to reuse the workspace; it reuses equilibration, buffers, and iterates, refactorizing only when necessary.

```julia
ws = setup(P, q, A, l, u)
for step in 1:horizon
    update!(ws; q = q_k, l = l_k, u = u_k)
    sol = solve!(ws)          # warm started from the previous step
end
```

Updating $q$ never requires a refactorization. Changing $l$ or $u$ only does if it moves a row between equality, inequality, or free classes. Changing $P$ or $A$ always does.

`ws.refactor_count` tracks total factorizations. This includes those from adaptive $\rho$, so the count grows even when using `update!` if $\rho$ changes.

Equilibration is done once in `setup`. If your data changes magnitude significantly, build a new workspace.

## Accuracy

Default tolerances are `eps_abs = eps_rel = 1e-3`. To improve accuracy:
* Lower `eps_abs`/`eps_rel` (more iterations).
* Set `polish = true` to solve the resulting equality-constrained QP exactly. This brings KKT residuals to machine precision at the cost of one extra factorization.

Polishing only runs if it improves both residuals, so it cannot make the solution worse.

## Choosing a backend

`linsys = :auto` picks the first compatible backend. For two dense matrices, this is an $n \times n$ Cholesky of the reduced system. Structured matrices (diagonal, banded, etc.) are caught earlier. Matrix-free operators use the matrix-free backend. If a dense Cholesky fails because the matrix isn't positive definite, it falls back to a Bunch-Kaufman factorization of the full $(n+m) \times (n+m)$ system. Use ``linsys = :kkt`` for a full factorization, which is more accurate for ill-conditioned problems.

## Watching a solve

`verbose = true` prints progress: a header, one line per termination check, and a footer with status, iterations, and residuals.

```
 iter      objective      prim res      dual res           rho
   25        1.39217        0.0543       0.00297           0.1
   50        1.45711       0.00245      0.000415           0.1
  ...
  125        1.46211      0.000374      0.000174         0.549
```

The `rho` column shows adaptive $\rho$ updates, which trigger refactorizations.

Output goes to `Core.stdout` rather than `Base.stdout` to support `--trim` compilation. Use `redirect_stdout` to capture it.

## What a solve reports

The [`Solution`](@ref) object carries the objective, duality gap, both residuals, `rel_kkt_error`, iteration and $\rho$ counts, the polishing result, and four timings.

`rho_updates` counts only adaptive $\rho$ changes; `ws.refactor_count` counts all refactorizations (including data changes).

## Status values

There are eleven status values in [`Status`](@ref PureOSQP.Status). Key facts:
1. An unconverged result is never marked `SOLVED`.
2. If there is no meaningful primal-dual point, `x` and `y` are `NaN`.

Use [`has_solution`](@ref PureOSQP.has_solution) to check.

## What is rejected

`setup` fails if:
* $P$ is not symmetric or is indefinite.
* `q` contains `NaN` or `Inf`.
* `l` or `u` contains `NaN`.
* $l > u$ elementwise.
* $l = +\infty$ or $u = -\infty$.
* Mismatched dimensions.
* Invalid settings (e.g., $\sigma \le 0$, $\alpha \notin (0, 2)$, or `max_iter` $\le 0$).

## Citation

Stellato, Banjac, Goulart, Bemporad and Boyd, *OSQP: an operator splitting solver for
quadratic programs*, Mathematical Programming Computation 12(4):637–672, 2020.
