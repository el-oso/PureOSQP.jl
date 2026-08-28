# Roadmap

What the reference implementation does that PureOSQP does not. The list is derived from
libosqp's public API — `osqp_api.c`, `osqp_api_types.h` and `osqp_api_constants.h` for
1.x, plus the 0.6.2 `osqp.c` surface — compared against this package's exports,
[`Settings`](@ref) and [`Solution`](@ref).

Some entries are deliberate and will stay; those are marked as such and explained under
[What is deliberately different](@ref). The rest are open.

## Missing capabilities

**Solution derivatives.** Upstream computes derivatives of the solution with respect to
`P`, `q`, `A`, `l` and `u` (`osqp_adjoint_derivative_compute`, `..._get_mat`,
`..._get_vec`), which is what lets a QP sit inside a differentiable program as a layer.
Nothing here corresponds to it. This is the largest capability gap.

**Time limit and interruption.** The solver does no timing at all, so there is no
`time_limit` setting, no `TIME_LIMIT_REACHED` status, no interrupt checking and no
`SIGINT` status. A run that takes `max_iter = 20000` iterations cannot be bounded by
wall-clock.

**Code generation.** `osqp_codegen` emits a self-contained C solver for a fixed problem
structure, with embedded-mode, float-type and printing/profiling/interrupt toggles. The
comparable capability here is `juliac --trim`, which the reference has no equivalent of,
but the two are not interchangeable: trimming produces a Julia binary, not embeddable C.

**Indirect (matrix-free) solve.** The whole conjugate-gradient path is absent:
`cg_max_iter`, `cg_tol_reduction`, `cg_tol_fraction` and the diagonal preconditioner.
Every backend here factors an explicit matrix, so a problem that can only supply
matrix-vector products cannot be solved.

**Duality-gap termination.** libosqp 1.x can terminate on the duality gap
(`check_dualgap`) in addition to the primal and dual residuals. Only the residual tests
are implemented.

**`ρ` adaptation on KKT error.** Upstream offers four modes: disabled, fixed iteration
interval, wall-clock fraction, and relative KKT-error decrease. This package implements the
first two. The wall-clock mode is *deliberately* omitted — it makes iteration counts depend
on the machine, and reproducible counts are what the oracle tests check. The KKT-error mode
carries no such problem and is simply not built.

## Missing reported information

[`Solution`](@ref) reports `x`, `y`, `status`, `obj_val`, `prim_res`, `dual_res`, `iter`
and `polished`, plus the infeasibility certificates. `OSQPInfo` also carries:

| field | note |
|---|---|
| `dual_obj_val`, `duality_gap` | the test referee computes the gap from problem data; the solver never reports it |
| `rho_estimate`, `rho_updates` | `ws.refactor_count` conflates `ρ` updates with data-driven refactorizations, so neither can be recovered from it |
| `setup_time`, `solve_time`, `update_time`, `polish_time`, `run_time` | no timing is collected |
| `rel_kkt_error`, `primdual_int` | 1.x convergence diagnostics |
| `status_polish` | upstream distinguishes five polish outcomes; `polished::Bool` cannot separate "no active set found, skipped" from "attempted and failed" |

## Missing settings

- `scaled_termination` — the scaled primal residual is already computed for the `ρ` rule,
  but termination always uses the unscaled residuals.
- `rho_is_vec` — `ρ` is always a vector here; upstream can run it as a scalar.
- `linsys_solver` selection across QDLDL, MKL Pardiso and CUDA backends. The equivalent
  choice here is `linsys = :auto | :kkt`, which selects a formulation rather than a library.
- `device`, `profiler_level`, `allocate_solution` — embedded and GPU concerns with no
  counterpart.

## Missing API surface

- `osqp_update_settings` — settings are fixed once [`setup`](@ref) returns. 0.6.2 exposed
  about fifteen individual updaters (`osqp_update_max_iter`, `_eps_abs`, `_alpha`,
  `_polish`, …) which 1.x consolidated into one call. Note that changing `rho` or `sigma`
  requires a refactorization, so this is not merely a field assignment.
- `osqp_update_rho` as a public entry point.
- Introspection: `osqp_version`, `osqp_capabilities`, `osqp_error_message`,
  `osqp_get_dimensions`.

[`update!`](@ref) does cover upstream's whole data-update family — `osqp_update_lin_cost`,
`osqp_update_bounds`, `osqp_update_lower_bound`, `osqp_update_upper_bound`,
`osqp_update_P`, `osqp_update_A` and `osqp_update_P_A` — and `warm_start!(; x, y)` covers
`osqp_warm_start_x` and `osqp_warm_start_y`.

## Ecosystem

**No MathOptInterface wrapper.** In practice this is a larger barrier to use than any
algorithmic gap above, since it is what makes a solver reachable from JuMP.

**No sparse linear-algebra backend.** Deliberate, and measured: forming the reduced matrix
densifies, and a sparse factorization of it does not pay. See
[How the sparsest case was closed](@ref "How the sparsest case was closed") for the fill-in
measurements. A `SparseArrays` extension does specialise equilibration's column traversals.

## Suggested order

The small items close real holes for little work: add `time_limit`, separate `ρ` updates
from refactorizations in the reported counts, and report `duality_gap` and a real polish
status. Derivatives and a MathOptInterface wrapper are projects rather than fixes.

Note that anything printing or timing has to respect the `--trim` guarantee, which
analyses code whether or not the branch that reaches it is ever taken. `verbose` is the
worked example: Printf and bare `println` both fail there, so the output is written
through `Core.stdout` by hand.
