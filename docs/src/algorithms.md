# Choosing an algorithm

Two algorithms solve the same problem. The sixth argument of [`solve`](@ref) and
[`setup`](@ref) picks one and holds the settings only that algorithm reads; everything both
read is a keyword argument of [`Options`](@ref).

```julia
ws = setup(P, q, A, l, u, InteriorPoint(); max_iter = 50)
sol = solve!(ws)
ws.algorithm     # InteriorPoint{Float64, Float64, Float64, Int64}: parameters in the solve's element type
ws.options       # Options{Float64}: max_iter, the tolerances, linsys, polishing, …
update_settings!(ws; eps_abs = 1e-10)                    # change an option
update_settings!(ws, InteriorPoint(reg_primal = 1e-6))   # replace the algorithm parameters
```

The keyword arguments are the fields of [`Options`](@ref). The two methods default some of
them differently — `max_iter` is `4000` for `OperatorSplitting` and `100` for
`InteriorPoint`, and the tolerances `1e-3` and `1e-8` — and [`default_options`](@ref) shows
the full set for either. A value you pass is always used as given. A setting passed in the
wrong place throws, naming where it belongs:

```julia
InteriorPoint(rho = 0.2)                           # MethodError: rho is not an InteriorPoint parameter
solve(P, q, A, l, u, InteriorPoint(); rho = 0.2)   # ArgumentError: rho is a parameter of OperatorSplitting
```

## Accuracy and iteration count

[`OperatorSplitting`](@ref) is OSQP's ADMM iteration: many cheap iterations sharing one
factorization. [`InteriorPoint`](@ref) is a Mehrotra predictor–corrector method: a few
iterations, each factoring a new Newton system.

`bench/ipm_vs_clarabel.jl` runs both on the smallest instance of each OSQP suite problem
class, `InteriorPoint` at `eps_abs = eps_rel = 1e-8` and `OperatorSplitting` at `1e-6`, the
tightest tolerance ADMM reaches in a modest iteration count on these problems
(`bench/results/ipm_vs_clarabel.json`):

| class | ADMM iterations (`1e-6`) | IPM iterations (`1e-8`) |
|---|---|---|
| Random QP | 200 | 9 |
| Eq QP | 50 | 2 |
| Portfolio | 125 | 10 |
| Lasso | 100 | 6 |
| SVM | 375 | 9 |
| Huber | 125 | 9 |
| Control | 50 | 7 |

`InteriorPoint` reaches a tighter tolerance in 2 to 10 outer iterations; `OperatorSplitting`
takes 50 to 375 at a looser one on the same problems. The gap is not just a fixed offset: the
benchmark suite's full-size problems, run through [`bench/osqp_suite.jl`](@ref "The OSQP
benchmark suite") at `eps_abs = eps_rel = 1e-5` and through [`bench/rho_schedule.jl`](@ref "The
ρ schedule") at `1e-6`, take more ADMM iterations at the tighter tolerance on every class that
changes at all — Random QP 925 → 1225, Portfolio 450 → 600, Lasso 100 → 125, SVM 300 → 325,
Control 325 → 450. `InteriorPoint`'s iteration count is set by Newton's method converging
locally, which is far less sensitive to how tight the tolerance is; `OperatorSplitting`'s is a
first-order method's, which is not.

This is not a claim that `InteriorPoint` is always faster: each of its iterations costs a
factorization, where ADMM's iterations only apply the one it already has, so the crossover
depends on the problem. [Benchmarks](@ref "The interior-point method against Clarabel") has
the times, not just the iteration counts.

## Re-solving a sequence

Both algorithms accept [`update!`](@ref), [`warm_start!`](@ref), [`cold_start!`](@ref) and
[`update_settings!`](@ref), and both start a re-solve from the previous point when
`warm_starting = true` (the default).

**`OperatorSplitting` can skip the factorization entirely.** Updating `q` alone never
refactorizes; updating `l` or `u` only does if a row moves between equality, inequality and
free; `P` or `A` always does. The Model Predictive Control loop in
[Examples](@ref "Model predictive control") is built on this: fifteen closed-loop solves,
one factorization, because only the initial-state bounds move and none of them cross a class.
Each solve is then also short, because it starts from the previous step's iterates — this is
what the warm start buys under ADMM: fewer iterations on top of no refactorization.

**`InteriorPoint` refactors every outer iteration regardless of `update!`.** Every iteration
solves a fresh Newton system at that iteration's row weights, so there is no factorization for
`update!` to preserve — it saves the equilibration and the buffers, not a solve. `warm_start!`
still seeds the first iterate from a point you supply, and a re-solve takes at most as many
outer iterations as a cold one (checked in `test/ipm_tests.jl`), but there is little to save:
the count is already 2 to 10 at the default tolerance, so a warm start shortens an already
short run rather than replacing hundreds of iterations with dozens.

## What each algorithm throws on

Neither algorithm restricts the matrix types in [Matrix types](matrices.md) or
[Structured operators](@ref) beyond what `linsys` demands. The differences are in what a
caller-supplied operator needs, and in which `linsys` backends each algorithm accepts.

| | `OperatorSplitting` | `InteriorPoint` |
|---|---|---|
| matrix-free operators (`linsys = :indirect`) | works with the built-in Jacobi preconditioner, or none | needs `linsys = :indirect`, a **caller-supplied** preconditioner, and `scaling = 0`; passing the built-in preconditioners or equilibration throws, naming the remedy ([Operators under the interior-point method](@ref)) |
| `linsys = :kronecker` | works | throws: the Kronecker backend needs one weight for every row, and the interior-point method's weights are per-row |
| `linsys = :lowrank` | works | throws: the Woodbury solve misses the tolerance on linear programs ([Algorithm](@ref "Backends under the interior-point method")) |

The reason `InteriorPoint` needs a caller's own preconditioner on an operator is measured, not
assumed: its row weights reach `1/reg_dual` (`1e8` by default) on equality and active rows and
change every outer iteration, so a fixed diagonal preconditioner does not keep conjugate
gradients within budget the way it does under ADMM's fixed `ρ`. `bench/ipm_matrixfree.jl`
measures this on 24 dense planted instances with a lagged Cholesky preconditioner the caller
supplies, refreshed every third outer iteration: all 24 solve at `eps = 1e-6` with a referee
residual of at most `7.9e-7`, in the same outer iterations as the dense full-KKT factorization
(`bench/results/ipm_matrixfree.json`). One sparse instance with a limited-memory incomplete
`LDLᵀ` preconditioner fails instead — a preconditioner has to keep the inner iteration count
bounded as the weights spread, and not every cheap one does.

## Polishing, derivatives and infeasibility

**Polishing** runs the same way under both: `polishing = true` guesses the active set from the
iterate, solves the resulting equality-constrained QP with `bunchkaufman!` and three steps of
iterative refinement, and replaces the answer only if both residuals improve. The `polishing`,
`polish_refine_iter` and `delta` options apply to either algorithm.

**Derivatives require polishing under `InteriorPoint`, not under `OperatorSplitting`.**
[`adjoint_derivative`](@ref) and [`forward_derivative`](@ref) read the active set from a row's
multiplier being far from zero. ADMM projects its multipliers onto the feasible box directly,
so an inactive row's multiplier is already at (or near) zero and there is nothing extra to
check. An interior-point solution holds an inactive row's multiplier at the barrier parameter
`μ_final` instead, which the active-set test cannot tell apart from a genuinely active one — so
taking a derivative from an unpolished `InteriorPointWorkspace` throws, asking for
`polishing = true` first, and polishing is what brings that multiplier down before the
derivative is taken.

**Infeasibility certificates are the same test under both.** `InteriorPoint` does not implement
its own primal- and dual-infeasibility check; it reuses ADMM's certificate test, applied to its
own last step and its normalized iterate. Both report `PRIMAL_INFEASIBLE` and
`DUAL_INFEASIBLE` (and their `*_INACCURATE` variants) with a certificate in
`Solution.prim_inf_cert` or `Solution.dual_inf_cert`.

**What `Solution` carries differs in what reads zero.** The struct is shared, so every field
exists under both algorithms, but `rho_estimate`, `rho_updates`, `accel_declined`,
`primdual_int` and `primdual_int_log` are always zero under `InteriorPoint` — there is no `ρ`,
no accelerator and no primal-dual integral to report. `cg_iters` is nonzero for either
algorithm only under `linsys = :indirect`. `NUMERICAL_ERROR` is a status only `InteriorPoint`
returns — a stalled Newton system, a non-finite residual, or conjugate gradients missing too
many solves in a row; ADMM never reports it.

## Summary

| | `OperatorSplitting` (default) | `InteriorPoint` |
|---|---|---|
| iteration cost | many cheap iterations, one factorization reused until `ρ` changes | a few iterations, a fresh factorization each |
| default tolerance | `1e-3` | `1e-8` |
| `update!` | can skip refactorization entirely (`q`-only updates always do) | refactorizes every outer iteration regardless |
| matrix-free operators | no restriction | needs a caller-supplied preconditioner and `scaling = 0` |
| `linsys = :kronecker`, `:lowrank` | works | throws |
| derivatives | ready from the iterate as it stands | require `polishing = true` first |
| infeasibility certificates | yes | yes, through the same test |

Reach for `InteriorPoint` when the tolerance you need is tighter than ADMM reaches in a modest
iteration count, or when you are solving once rather than in a loop that reuses a
factorization. Reach for `OperatorSplitting` for repeated solves through `update!` and for
matrix-free operators without a preconditioner of your own.

Timing tables, not just iteration counts, are in [Benchmarks](@ref "The interior-point method
against Clarabel") and the benchmark suite tables above it; this page states which algorithm
qualifies for a given problem, not how many milliseconds it takes.
