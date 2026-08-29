# Roadmap

What the reference implementation does that PureOSQP does not. The list is derived from
libosqp's public API — `osqp_api.c`, `osqp_api_types.h` and `osqp_api_constants.h` for
1.x, plus the 0.6.2 `osqp.c` surface — compared against this package's exports,
[`Settings`](@ref) and [`Solution`](@ref).

Some entries are deliberate and will stay; those are marked as such and explained under
[What is deliberately different](@ref). The rest are open.

## A backend per matrix representation

**The aim is to support every matrix representation well, not merely to accept it.** The
gap between those two is the largest open item here, and it is larger than the section
below.

What works today: `P` and `A` are held by reference, never copied, and every per-iteration
product runs `mul!` on the matrix the caller passed, so a `Diagonal`, a `SubArray` or a
`SparseMatrixCSC` keeps its own product. Equilibration reaches their entries through four
overridable column traversals, which the `SparseArrays` extension specialises.

What does not: **every linear-system backend materialises dense buffers**, whatever was
passed in.

| buffer | size | file |
|---|---|---|
| `ReducedCholesky.W` — a dense copy of `A` | `m×n` | `src/linsys.jl` |
| `ReducedCholesky.Rinv` | `n×n` | `src/linsys.jl` |
| `FullKKT.K` | `(n+m)×(n+m)` | `src/linsys.jl` |
| `polish!`'s reduced KKT | `(n+k)×(n+k)` | `src/polish.jl` |
| `active_kkt`'s `M` | `(n+k)×(n+k)` | `src/derivative.jl` |

Each is `similar(q0, ...)` off a dense vector, so each is dense by construction. `W` is the
sharpest case: `factorize!` zeroes an `m×n` buffer and writes `A`'s scaled entries into it
purely to hand `syrk` a dense argument, which is waste in proportion to how sparse `A` was.

[`LinearSystem`](@ref) is already the seam for fixing this — a declared, contract-checked
interface whose implementation is chosen once at [`setup`](@ref) and then fixed in the
workspace's type. Choosing it on `typeof(P)` and `typeof(A)` is what it is for. Three
concrete directions:

- **Sparse `A`** — form `R = P̃ + σI + Ãᵀ diag(ρ) Ã` through sparse products rather than a
  dense `W`. Note this is a different question from the one already measured: what was
  tested and rejected was a sparse LDLᵀ of the *full* KKT against the dense reduced solve.
  At 1% density `R` itself is only 6.3% dense, so forming it sparsely is untested and
  plausible even though factoring it sparsely is not.
- **Structured `P`** — a `Diagonal` or `Tridiagonal` is currently read entry by entry into a
  dense `R`, which is why [Matrix types](@ref "Matrix types") measures 0.80× rather than a
  speedup. The mechanism works; the backend throws the structure away.
- **Matrix-free** — the case where nothing can be materialised at all, covered below.

## Capabilities against upstream

**Solution derivatives: implemented.** [`adjoint_derivative`](@ref) and
[`forward_derivative`](@ref) differentiate the KKT conditions at the solution rather than
the ADMM loop, so one solve with the active-set KKT matrix gives the derivative whatever
the iteration did to get there. Both are checked against central differences through whole
solves, in all five parameters.

A derivative that does not exist is refused rather than approximated: a weakly active row,
more active rows than variables, a singular active-set matrix, or one ill-conditioned
enough that the solve fails a relative-residual gate. `polish!` is allowed a regularized
guess because it tests whether the guess helped; a gradient consumer cannot, and a
least-squares answer would carry the right shape and units while being a different
quantity.


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

**`ρ` adaptation.** Upstream offers four modes: disabled, fixed iteration
interval, wall-clock fraction, and relative KKT-error decrease. This package implements
three of them via `adaptive_rho = :disabled | :iterations | :kkt_error`, a `Bool` still
naming the first two. The wall-clock mode is *deliberately* omitted — it makes counts depend
on the machine, and reproducible counts are what the oracle tests check.

## Missing reported information

[`Solution`](@ref) now reports the objectives, the duality gap, `rel_kkt_error`,
`rho_estimate`, `rho_updates`, `status_polish` and the timings. What remains from
`OSQPInfo`:

| field | note |
|---|---|
| `update_time` | `update!` is not timed; `setup_time`, `solve_time`, `polish_time` and `run_time` are |
| `primdual_int` | the primal-dual integral, a 1.x convergence diagnostic requiring per-iteration profiling |

Interruption is implemented: a `Ctrl-C` inside the loop returns `INTERRUPTED` with the
point reached, and every other exception propagates. Note that `time_limit` bounds the ADMM
loop and not `setup`, because `setup` is timed but not budgeted — upstream's limit covers
the whole run, since it times every phase against it.

## Missing settings

`check_dualgap`, `scaled_termination` and `rho_is_vec` are implemented.

`check_dualgap` defaults on, matching libosqp 1.x. It binds rarely on well-scaled problems,
where the residuals already imply a small gap, but it does bind: across a sweep of badly
scaled objectives it changed the iteration count in 114 of 600 comparable runs. Anything
comparing iteration counts against libosqp 0.6.2 pins it off, since 0.6.2 has no such test
— that includes the oracle tests, the ported C-suite case and the benchmark scripts.

What remains:

- `linsys_solver` selection across QDLDL, MKL Pardiso and CUDA backends. The equivalent
  choice here is `linsys = :auto | :kkt`, which selects a formulation rather than a library.
- `device`, `profiler_level`, `allocate_solution` — embedded and GPU concerns with no
  counterpart.

## API surface

Complete, with one deliberate omission.

[`update_settings!`](@ref) covers `osqp_update_settings` and the fifteen individual
updaters 0.6.2 exposed. It refactorizes only for `rho`, `sigma` and `rho_is_vec`, which are
the settings the factorization contains, and it *rejects* `linsys` and `scaling` rather
than appearing to accept them: the backend is part of the workspace's type, and the
equilibration factors were computed once from the data `setup` saw, so honoring either
would leave the solver quietly running something other than what was asked for.

[`update_rho!`](@ref) is `osqp_update_rho`. [`dimensions`](@ref) and
[`capabilities`](@ref) are `osqp_get_dimensions` and `osqp_capabilities`.

There is no `osqp_error_message`, and there should not be: it exists to turn an error code
into a string, and this package throws exceptions that carry their own messages.

[`update!`](@ref) covers upstream's whole data-update family — `osqp_update_lin_cost`,
`osqp_update_bounds`, `osqp_update_lower_bound`, `osqp_update_upper_bound`,
`osqp_update_P`, `osqp_update_A` and `osqp_update_P_A` — and `warm_start!(; x, y)` covers
`osqp_warm_start_x` and `osqp_warm_start_y`.

## Ecosystem

**MathOptInterface wrapper: implemented**, as a package extension, so it costs nothing to
a caller who does not load MOI. `PureOSQP.Optimizer` is reachable from JuMP, and the whole
of `MOI.Test` passes with no excluded test names. Note it needs tighter tolerances than the
solver defaults to: `MOI.Test` checks to `1e-4` and the defaults are `1e-3`.


**No sparse linear-algebra backend.** Deliberate, and measured: forming the reduced matrix
densifies, and a sparse factorization of it does not pay. See
[How the sparsest case was closed](@ref "How the sparsest case was closed") for the fill-in
measurements. A `SparseArrays` extension does specialise equilibration's column traversals.

## Suggested order

One project remains: the matrix-free backend, and it is last for a reason — every problem
the dense backends handle well is one it would handle worse. What is left otherwise is
`update_time`, the primal-dual integral, and the settings that select a linear-algebra
library or a GPU, none of which have a counterpart here.

Two notes for whoever picks these up, both learned the hard way:

Anything that prints or times has to respect the `--trim` guarantee, which analyses code
whether or not the branch reaching it is ever taken. `verbose` is the worked example:
Printf, bare `println` and `lpad` all fail there, so its output is written through
`Core.stdout` by hand — see `print_padded` in `src/admm.jl`.

The element type is `Real`, not `AbstractFloat`, so dual numbers run the solver and AD can
differentiate straight through it. `Real` rather than `Number`, because the solver orders
`l ≤ Ax ≤ u` and compares residuals derived from the iterates, so the element type needs
`<` everywhere it appears; `Complex` cannot satisfy that.

What that buys is the derivative's oracle. `ForwardDiff.jacobian` through `solve` gives the
exact Jacobian, with no step-size error and no restriction to one direction, and
[`forward_derivative`](@ref) matches it to `1e-13` — against the `1e-5` that central
differencing on a single direction can support.

One limit remains: `bunchkaufman!` is LAPACK-only for BLAS floats, with no generic fallback
in `LinearAlgebra`, and it is reached from `polish!`, from the `FullKKT` backend and from
[`adjoint_derivative`](@ref). So dual numbers work on the reduced backend with
`polish = false`. Note `INFTY` is not a blocker, despite calling `prevfloat(typemax(T))`:
ForwardDiff defines both for `Dual`.

An AD integration would be a `ChainRulesCore` extension supplying `rrule` and `frule` over
[`adjoint_derivative`](@ref) and [`forward_derivative`](@ref). That reaches Zygote, which
consumes ChainRules; it does *not* reach Mooncake, which requires an explicit
`Mooncake.@from_rrule` and does not pick up ChainRules rules on its own.
