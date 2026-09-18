# Choosing an algorithm

Three algorithms solve the same problem. PureOSQP.jl supplies [`OperatorSplitting`](@ref),
PureIPM.jl supplies [`InteriorPoint`](@ref), PureDAQP.jl supplies [`ActiveSet`](@ref). Each
re-exports PureQPBase.jl, so `using` any one is enough; `using` several puts all their
algorithms on the same [`solve`](@ref).

## Start here

Read down the table and take the first row that describes your problem.

| if | use | because |
|---|---|---|
| you re-solve a sequence, changing only `q`, `l` or `u` | `OperatorSplitting` | it keeps the factorization across [`update!`](@ref) and warm starts from the last answer |
| `P` or `A` is matrix-free, and you have no preconditioner | `OperatorSplitting` | the only one that takes an operator with the built-in Jacobi preconditioner |
| the data is dense **and** few rows are active at the solution | `ActiveSet` | it costs about one iteration per active row and returns the exact answer |
| you want `1e-8` or better from a single solve | `InteriorPoint` | a handful of Newton steps reach it whatever the conditioning |
| `P` or `A` is large and sparse or structured | `InteriorPoint` | it factors the pattern; `ActiveSet` would densify it |
| none of the above | `OperatorSplitting` | the default, and the cheapest per iteration |

**"Few rows active" is the whole of the active-set question**, and it is a property of the
problem, not a setting. Each iteration adds or drops one row, so the iteration count lands near
the number of rows active at the solution. Where that is a small fraction of `m` nothing here
beats it; where most rows end up active it is the slowest of the three, by a wide margin. The
[measurements below](@ref "Accuracy and iteration count") show both ends on the same suite. If
you do not know how many rows will be active, solve once and read `sol.iter`.

The sixth argument of [`solve`](@ref) and [`setup`](@ref) picks the algorithm and holds the
settings only that algorithm reads. Everything shared is a keyword argument of
[`Options`](@ref). The five-argument form runs [`OperatorSplitting`](@ref).

```julia
using PureIPM
ws = setup(P, q, A, l, u, InteriorPoint(); max_iter = 50)
sol = solve!(ws)
ws.algorithm     # InteriorPoint{Float64, Float64, Float64, Int64}: parameters in the solve's element type
ws.options       # Options{Float64}: max_iter, the tolerances, linsys, polishing, …
update_settings!(ws; eps_abs = 1e-10)                    # change an option
update_settings!(ws, InteriorPoint(reg_primal = 1e-6))   # replace the algorithm parameters
```

The keyword arguments are the fields of [`Options`](@ref). The algorithms default some of
them differently: `max_iter` is `4000` for `OperatorSplitting`, `100` for `InteriorPoint` and
`1000` for `ActiveSet`, and the tolerances are `1e-3`, `1e-8` and `sqrt(eps)`.
[`default_options`](@ref) shows the full set for any of them. `ActiveSet` reads only `max_iter`
from this set. It refuses `linsys`, `scaling` and `polishing` outright, and the rest do not
reach it: its tolerances are its own parameters, `primal_tol` and `zero_tol`, because it prices
normalized rows rather than measuring a residual against `eps_abs`, and it carries its working
set across a re-solve whatever `warm_starting` says. A value you pass is always used as you
gave it. A setting passed in the wrong place throws, and names where it belongs:

```julia
InteriorPoint(rho = 0.2)                           # MethodError: rho is not an InteriorPoint parameter
solve(P, q, A, l, u; rho = 0.2)                    # ArgumentError: rho is a parameter of OperatorSplitting
solve(P, q, A, l, u, InteriorPoint(); rho = 0.2)   # ArgumentError: rho is not an option of InteriorPoint
```

## Accuracy and iteration count

[`OperatorSplitting`](@ref) is OSQP's ADMM iteration: many cheap iterations that share one
factorization. [`InteriorPoint`](@ref) is a Mehrotra predictor–corrector method: a few
iterations, each factoring a new Newton system. For both, the iteration count is the price of a
tolerance. For [`ActiveSet`](@ref) it is not: it adds or drops one row per iteration and stops
at the exact solution over the active rows, so the count measures the problem's active set
rather than the accuracy asked for.

`PureIPM/bench/ipm_vs_clarabel.jl` runs all three on the smallest instance of each OSQP suite
problem class: `InteriorPoint` at `eps_abs = eps_rel = 1e-8`, `OperatorSplitting` at `1e-6` —
the tightest tolerance ADMM reaches in a modest iteration count on these problems — and
`ActiveSet` on dense copies of the same data
(`PureIPM/bench/results/ipm_vs_clarabel.json`). Every solver agrees on `x` to `2e-5` or better.

| class | m | ADMM (`1e-6`) | IPM (`1e-8`) | ActiveSet | ADMM | IPM | ActiveSet |
|---|---|---|---|---|---|---|---|
| Random QP | 60 | 200 | 9 | **7** | 0.054 ms | 0.075 ms | **0.005 ms** |
| Eq QP | 10 | 50 | 2 | **1** | 0.026 ms | 0.033 ms | **0.004 ms** |
| Control | 108 | 50 | 7 | **3** | 0.097 ms | 0.228 ms | **0.062 ms** |
| Portfolio | 102 | 125 | 10 | 99 | **0.177 ms** | 0.240 ms | 0.394 ms |
| Lasso | 204 | 100 | 6 | 6 | 0.214 ms | **0.195 ms** | 1.051 ms |
| SVM | 400 | 375 | 9 | 204 | 0.715 ms | **0.340 ms** | 3.950 ms |
| Huber | 600 | 125 | 9 | 397 | 0.788 ms | **0.836 ms** | 45.408 ms |

The first three columns are iteration counts, the last three wall clock.

**The active-set rows split in two, and the iteration count says which half.** Where it settles
in a handful of iterations — 7 of 60 rows on Random QP, 3 of 108 on Control — it is the fastest
of the three by 4× to 15×. Where most rows enter the working set — 204 of 400 on SVM, 397 of 600
on Huber — each iteration is a rank-one update of a working set that keeps growing, and it ends
up 12× to 54× slower. Portfolio sits between, at 99 active rows out of 102.

That is the same measurement the [Benchmarks](@ref "Against other solvers") page makes from the
other direction: dense problems with few active rows are where an active-set method wins, and
these classes were built to exercise sparsity, so four of the seven have a singular `P` and need
`eps_prox = 1e-4` before it will run at all.

`InteriorPoint` reaches a tighter tolerance in 2 to 10 outer iterations. `OperatorSplitting`
takes 50 to 375 at a looser one on the same problems. The gap is not a fixed offset. Run the
benchmark suite's full-size problems through
[`PureOSQP/bench/osqp_suite.jl`](@ref "The OSQP benchmark suite") at `eps_abs = eps_rel = 1e-5`
and through [`PureOSQP/bench/rho_schedule.jl`](@ref "The ρ schedule") at `1e-6`, and every class
that changes at all takes more ADMM iterations at the tighter tolerance: Random QP 925 → 1225,
Portfolio 450 → 600, Lasso 100 → 125, SVM 300 → 325, Control 325 → 450. What sets
`InteriorPoint`'s iteration count is Newton's method converging locally, which cares far less
about how tight the tolerance is. `OperatorSplitting` is a first-order method, which cares a
lot.

This does not say `InteriorPoint` is always faster. Each of its iterations costs a
factorization, while ADMM's iterations only apply the one it already has, so where they cross
depends on the problem. [Benchmarks](@ref "The interior-point method against Clarabel") has the
times, not just the iteration counts, and
[Against other solvers](@ref "Against other solvers") times all three on the same problems.

## Re-solving a sequence

All three accept [`update!`](@ref), [`warm_start!`](@ref), [`cold_start!`](@ref) and
[`update_settings!`](@ref). `OperatorSplitting` and `InteriorPoint` start a re-solve from the
previous point when `warm_starting = true` (the default); `ActiveSet` restarts from the
previous working set instead, which [`cold_start!`](@ref) is what drops.

**`OperatorSplitting` can skip the factorization altogether.** Updating `q` alone never
refactorizes. Updating `l` or `u` refactorizes only when a row moves between equality,
inequality and free. Updating `P` or `A` always does. The Model Predictive Control loop in
[Examples](@ref "Model predictive control") is built on this: fifteen closed-loop solves, one
factorization, because only the initial-state bounds move and none of them cross a class. Each
solve is short as well, because it starts from the previous step's iterates. That is what the
warm start buys under ADMM: fewer iterations on top of no refactorization.

**`InteriorPoint` refactors every outer iteration, whatever `update!` did.** Every iteration
solves a fresh Newton system at that iteration's row weights, so there is no factorization for
`update!` to keep. It saves the equilibration and the buffers, not a solve. `warm_start!` still
seeds the first iterate from a point you supply, and a re-solve takes at most as many outer
iterations as a cold one (`PureIPM/test/ipm_tests.jl` checks that). But there is little to save.
The count is already 2 to 10 at the default tolerance, so a warm start shortens a run that was
already short. It does not replace hundreds of iterations with dozens.

**`ActiveSet` keeps the working set, and hands back the same `Solution` every time.** Updating
`q`, `l` or `u` keeps the Cholesky factor of `P` and the transformed constraint matrix, so only
the right-hand side is rebuilt; updating `P` or `A` rebuilds the reduction. The re-solve starts
from the previous answer's active rows, which is the whole of what a warm start means here.

!!! warning
    The `Solution` an `ActiveSet` solve returns is the workspace's own object, and its `x` and
    `y` are the workspace's own arrays. That is what makes a solve allocate nothing, and it
    means a result held across the next `solve!` is overwritten in place. Copy what you need
    before re-solving. The other two algorithms return a fresh `Solution` each time.

## What each algorithm throws on

`ActiveSet` takes dense `P` and `A` and nothing else: the reduction forms `A R⁻¹`, which an
operator cannot supply and which is dense whatever `A` was, so a matrix it cannot read entry by
entry throws at [`setup`](@ref). The other two limit none of the matrix types in
[Matrix types](matrices.md) or [Structured operators](@ref) beyond what `linsys` asks for. They
differ in what an operator you supply needs, and in which `linsys` backends each one accepts.

| | `OperatorSplitting` | `InteriorPoint` | `ActiveSet` |
|---|---|---|---|
| matrix-free operators (`linsys = :indirect`) | works with the built-in Jacobi preconditioner, or none | needs `linsys = :indirect`, a **caller-supplied** preconditioner, and `scaling = 0`; passing the built-in preconditioners or equilibration throws, naming the remedy ([Operators under the interior-point method](@ref)) | throws: the reduction needs entries |
| `linsys = :kronecker` | works | throws: the Kronecker backend needs one weight for every row, and the interior-point method's weights are per-row | throws: no backend to select |
| `linsys = :lowrank` | works | throws: the Woodbury solve misses the tolerance on linear programs ([Algorithm](@ref "Backends under the interior-point method")) | throws: no backend to select |
| `scaling` | any value | any value | throws unless `0`: the reduction normalizes its own rows |
| `polishing = true` | works | works, and is required before a derivative | throws: the answer is already exact over the working set |
| an indefinite `P` | throws at `setup` | throws at `setup` | throws at `setup`, from the Cholesky of `P + eps_prox*I` |

We measured why `InteriorPoint` needs a preconditioner of your own on an operator. We did not
assume it. Its row weights reach `1/reg_dual`, `1e8` by default, on equality and active rows,
and they change every outer iteration. A fixed diagonal preconditioner cannot keep conjugate
gradients inside its budget at that spread, though it can under ADMM's fixed `ρ`.
`PureIPM/bench/ipm_matrixfree.jl` measures this on 24 dense planted instances with a lagged
Cholesky preconditioner the caller supplies, refreshed every third outer iteration. All 24 solve
at `eps = 1e-6` with a referee residual of at most `7.9e-7`, in the same outer iterations as the
dense full-KKT factorization
(`PureIPM/bench/results/ipm_matrixfree.json`). One sparse instance with a limited-memory
incomplete `LDLᵀ` preconditioner fails instead. A preconditioner must keep the inner iteration
count bounded as the weights spread, and not every cheap one does.

## Polishing, derivatives and infeasibility

**Polishing runs the same way under `OperatorSplitting` and `InteriorPoint`.** `polishing =
true` guesses the active set from the iterate, solves the equality-constrained QP that comes out
of it with `bunchkaufman!` and three steps of iterative refinement, and replaces the answer only
if both residuals improve. The `polishing`, `polish_refine_iter` and `delta` options work with
either of those two. `ActiveSet` refuses `polishing = true`: it already ends on an exact
solution of the equality-constrained QP over its working set, which is what polishing computes.

**Derivatives need polishing under `InteriorPoint` only.**
[`adjoint_derivative`](@ref) and [`forward_derivative`](@ref) read the active set by asking
which multipliers sit far from zero. ADMM projects its multipliers onto the feasible box
directly, so an inactive row's multiplier is already at or near zero and there is nothing extra
to check. `ActiveSet` puts every inactive row's multiplier at exactly zero, which is the same
test's best case. An interior-point solution holds an inactive row's multiplier at the barrier
parameter `μ_final` instead, and the active-set test cannot tell that apart from a truly active
row. So taking a derivative from an unpolished `InteriorPointWorkspace` throws and asks for
`polishing = true` first. Polishing brings that multiplier down before you take the derivative.

**`OperatorSplitting` and `InteriorPoint` use the same infeasibility test.** `InteriorPoint` has
no primal- and dual-infeasibility check of its own. It reuses ADMM's certificate test on its own
last step and its normalized iterate. Both report `PRIMAL_INFEASIBLE` and `DUAL_INFEASIBLE`, and
their `*_INACCURATE` versions, with a certificate in `Solution.prim_inf_cert` or
`Solution.dual_inf_cert`. `ActiveSet` finds primal infeasibility differently: a dual step along
the null direction of a singular working-set Gram matrix with no row to block it is an unbounded
dual ray, and the dual of a convex QP is unbounded exactly when the primal is infeasible. It
reports `PRIMAL_INFEASIBLE` without populating either certificate field, and has no
dual-infeasibility test and no `*_INACCURATE` status.

**What `Solution` carries differs in which fields read zero.** The struct is shared, so every
field exists under every algorithm. But `rho_estimate`, `rho_updates`, `accel_declined`,
`primdual_int` and `primdual_int_log` are zero under `InteriorPoint` and under `ActiveSet`,
because there is no `ρ`, no accelerator and no primal-dual integral to report. `cg_iters` is
nonzero only under `linsys = :indirect`, which neither of those two accepts. `InteriorPoint`
returns `NUMERICAL_ERROR` for a stalled Newton system, a non-finite residual, or conjugate
gradients missing too many solves in a row; `ActiveSet` returns it when a dual step toward the
working set's multipliers finds no row to block it through a nonsingular Gram matrix, which the
method's own argument rules out and so signals a numerical breakdown. ADMM never reports it.

## Summary

| | `OperatorSplitting` (default) | `InteriorPoint` | `ActiveSet` |
|---|---|---|---|
| iteration cost | many cheap iterations, one factorization reused until `ρ` changes | a few iterations, a fresh factorization each | a few iterations, each a rank-one update of the working set's `LDLᵀ` |
| default tolerance | `1e-3` | `1e-8` | exact at the working set; `primal_tol` decides which rows enter |
| matrices | any `AbstractMatrix`, structure and sparsity exploited | the same | dense only, structure and sparsity ignored |
| `update!` | can skip refactorization entirely (`q`-only updates always do) | refactorizes every outer iteration regardless | keeps the reduction unless `P` or `A` changes |
| matrix-free operators | no restriction | needs a caller-supplied preconditioner and `scaling = 0` | not supported |
| `linsys` | every backend | all but `:kronecker` and `:lowrank` | none: it has no backend to choose |
| derivatives | ready from the iterate as it stands | require `polishing = true` first | ready: inactive multipliers are exactly zero |
| infeasibility certificates | yes | yes, through the same test | primal only, and without a certificate |

The [table at the top](@ref "Start here") is the short version of this one.

[Benchmarks](@ref "The interior-point method against Clarabel") and the suite tables above it
give times, not just iteration counts. This page says which algorithm fits a given problem, not
how many milliseconds it takes.
