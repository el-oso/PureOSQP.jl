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

**Interruption.** There is no interrupt checking and no `SIGINT` status, so a long run
cannot be stopped from the keyboard and reported as such. `time_limit` exists and bounds
the ADMM loop, but not setup, since setup is not timed either — upstream's budget covers
the whole run because it times every phase.

**Code generation** — a documented difference rather than work to be done.

`osqp_codegen` writes four files: `workspace.h`, `workspace.c`, `osqp_configure.h` and an
`emosqp.c` example `main`. Into them it bakes the settings, the scaled `P`, `q`, `A`, `l`,
`u`, the equilibration vectors, `rho_vec`, the starting iterate and the numeric
factorization. Every scratch vector becomes a fixed-size file-scope array, so the generated
program has no allocation site at all. Under `embedded_mode = 2` it also bakes the KKT
matrix, its index maps and the symbolic factorization, which is what lets `P` and `A`
change and be refactorized with no symbolic phase and no heap; `embedded_mode = 1` fixes
the matrices and `ρ` and updates only `q`, `l` and `u`.

Most of that has a counterpart here. Baked settings, data and factorization are what
`setup` already returns in a concretely typed workspace that `solve!` and `update!` reuse.
Static dispatch with no reflection is what `test/trim_tests.jl` proves, negative control
included. There is no symbolic phase or sparsity pattern to bake, because the factored
matrix is dense. And the deployment story is closer than it sounds: `juliac --output-lib
--compile-ccallable --experimental --trim=safe` turns a `Base.@ccallable` wrapper around
[`solve`](@ref) into a 3.7 MB shared library exporting a C symbol, callable from a plain C
`main` with no `jl_init`.

What does not close is the difference underneath: OSQP emits C source, Julia emits machine
code that needs the Julia runtime. The trimmed binary wants roughly 89 MB of shared
libraries beside it — libjulia, libjulia-internal, and the OpenBLAS the dense
factorizations genuinely use — plus about 19 ms of one-time initialization, and the garbage
collector is in the address space whether or not the hot path touches it. Decisively,
Julia does not exist for the targets code generation aims at: a Cortex-M part, a
fixed-point DSP, a toolchain qualified for DO-178C. A C source file compiles for all of
them.

So this stays listed as a difference, not a gap to close. Emitting C from this package
would be a second implementation of the solver, and for the dense case what it emitted
would be a pair of loops and a Cholesky — not what anyone reaching for a generated solver
wants. If matching it ever became the goal, the honest form is a separate C library.

**Indirect (matrix-free) solve.** The whole conjugate-gradient path is absent:
`cg_max_iter`, `cg_tol_reduction`, `cg_tol_fraction` and the diagonal preconditioner.
Every backend here factors an explicit matrix, so a problem that can only supply
matrix-vector products cannot be solved.

**Duality-gap termination is implemented** and on by default, matching libosqp 1.x. It
binds rarely on well-scaled problems, where the residuals already imply a small gap, but
it does bind: across a sweep of badly scaled objectives it changed the iteration count in
114 of 600 comparable runs. Comparisons against libosqp 0.6.2 pin it off, since 0.6.2 has
no such test.

**`ρ` adaptation on KKT error.** Upstream offers four modes: disabled, fixed iteration
interval, wall-clock fraction, and relative KKT-error decrease. This package implements the
first two. The wall-clock mode is *deliberately* omitted — it makes iteration counts depend
on the machine, and reproducible counts are what the oracle tests check. The KKT-error mode
carries no such problem and is simply not built.

## Missing reported information

[`Solution`](@ref) now reports the objectives, the duality gap, `rel_kkt_error`,
`rho_estimate`, `rho_updates`, `status_polish` and the timings. What remains from
`OSQPInfo`:

| field | note |
|---|---|
| `update_time` | `update!` is not timed; `setup_time`, `solve_time`, `polish_time` and `run_time` are |
| `primdual_int` | the primal-dual integral, a 1.x convergence diagnostic requiring per-iteration profiling |

## Missing settings

`check_dualgap`, `scaled_termination` and `rho_is_vec` are implemented. What remains:

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

The small items close real holes for little work: separate `ρ` updates from
refactorizations in the reported counts, and report `duality_gap` and a real polish status.
Derivatives and a MathOptInterface wrapper are projects rather than fixes.

Note that anything printing or timing has to respect the `--trim` guarantee, which
analyses code whether or not the branch that reaches it is ever taken. `verbose` is the
worked example: Printf and bare `println` both fail there, so the output is written
through `Core.stdout` by hand.
