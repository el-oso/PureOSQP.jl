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

[`PureOSQP.choose_backend`](@ref) is the seam: it dispatches on `typeof(P)` and `typeof(A)`,
runs once inside [`setup`](@ref), and fixes the backend in the workspace's type, so the
per-iteration solve still dispatches statically. A representation that admits a cheaper way
to form the reduced matrix is served by adding a method to it.

**Sparse `A`: done, in two backends.** Three things can happen to a sparse `A`, and which
one does is decided at [`setup`](@ref) from the matrices themselves.

`SparseFormedInverse` accumulates `Ãᵀ diag(ρ) Ã` over the stored entries — `Σᵢ nnzᵢ²` work,
no buffer — where the dense backend wrote `A` into an `m×n` array so that one `syrk` could do
`mn²` flops against mostly zeros. That product was 63% of a refactorization and that buffer
65% of the workspace. Measured at `n = 2000, m = 4000`, 0.25% density: refactorization 400 ms
to 148 ms, a whole solve 1192 ms to 777 ms, and the workspace 93 MiB to 32 MiB. It still
factors densely and stores `R⁻¹`, so the per-iteration solve is the same `symv`.

`SparseCholmod` also factors sparsely, through SparseArrays — `cholesky(Symmetric(R))` is
CHOLMOD, and `cholesky!(F, R)` reuses its symbolic analysis so later refactorizations pay
only the numeric phase. It cannot be a `ReducedInverse`, because the inverse of a sparse
matrix is dense: it keeps the factor and solves with a pair of sparse triangular solves
instead. On banded problems — MPC, and what OSQP's own sparse LDLᵀ is built for — that is
worth 20× at `n = 1000, m = 2000` and 37× at `n = 2000, m = 4000` against the dense
factorization, on identical iterates and a workspace of 2 MiB.

The two gates are measured, not assumed:

- **density**, for whether to form sparsely. Accumulation writes into `R` by scattered index
  where `syrk` streams contiguous memory, worth about an order of magnitude in constant
  factor, so the two cross near 20% density. Above 10% a `SparseMatrixCSC` is served by the
  dense-forming backend anyway; below it the gain runs 1.8–2.7×.
- **fill**, for whether to factor sparsely. CHOLMOD is asked to factor the reduced matrix's
  pattern and the `nnz(L)` it reports settles it — a decision made from the problem rather
  than from a density heuristic, which matters because fill is what actually drives the cost
  and density only correlates with it. Against the dense inverse's `symv`, the sparse
  triangular solves measure 12.6× faster at a fill of `nnz(L)/n² = 0.0025`, 1.53× at 0.044
  and slower at 0.086; the limit is 0.05, below the crossing, so the accepted region wins on
  the factorization *and* the solve. Random sparsity fills in to about 0.42 and is refused,
  which is the right answer and matches what was measured of sparse factorizations here
  before.

The probe factorization is not wasted when it is accepted: it *is* the first factorization,
whose symbolic part every later one reuses. When it is refused, a pre-screen on `nnz(R)`
bounds the cost, since `L` is at least as dense as `R`'s triangle.

One guarantee is narrower for `SparseCholmod` and `bench/strictmode_audit.jl` says so rather
than passing quietly. Its `factorize!` builds the reduced matrix through SparseArrays' sparse
products, whose constructors validate dimensions and format the message through code static
analysis cannot see past, so neither it nor `solve!` carries a type-stability claim. The hot
path is unaffected and fully checked: `admm_step!`, `update_residuals!` and `solve_system!`
are type-stable and allocation-free, which needed the solve to apply the permutation and the
two triangular solves itself — CHOLMOD's own `ldiv!` allocates 64 KB per call at `n = 2000`.

What remains dense whatever was passed in:

| buffer | size | file | worth fixing? |
|---|---|---|---|
| `ReducedCholesky.W` — a dense copy of `A` | `m×n` | `src/linsys.jl` | gone for sparse `A` |
| the reduced matrix and its inverse | `n×n` | `src/linsys.jl` | only if `R` is itself structured |
| `FullKKT.K` | `(n+m)×(n+m)` | `src/linsys.jl` | no — it is the accuracy fallback, chosen rarely |
| `polish!`'s reduced KKT | `(n+k)×(n+k)` | `src/polish.jl` | no — one-shot, and `k` is the active set |
| `active_kkt`'s `M` | `(n+k)×(n+k)` | `src/derivative.jl` | no — same |

The last two carry a consequence worth knowing rather than fixing: on a problem large and
sparse enough that `linsys = :indirect` was chosen because nothing dense fits, `polish =
true` or a derivative call will materialise a dense `(n+k)×(n+k)` anyway.

The opening item is now closed on both axes:

- **Structured `P`: done.** `structural_rows` gives the column traversals the rows a band
  type can hold a nonzero in, so a `Diagonal` costs `O(1)` per column rather than `O(n)`.
  Equilibration measures 2.3× faster on a `Diagonal` `P` and 2.2× on a `SymTridiagonal`,
  which is 1.06× and 1.02× end to end — bounded by equilibration's share of a run, but no
  longer the 0.80× that made structured storage a pessimisation. A *backend* would pay only
  where `R` inherits the structure, which needs `A` structured too, since `Ãᵀ diag(ρ) Ã`
  otherwise fills it in whatever `P` looked like.


A note on the corpus, since it decides what any of this measures. Random sparse patterns are
the worst case for a sparse factorization: no separators, near-maximal fill. They are what
the benchmarks here generate, and it is why the fill gate refuses them — correctly. A
structured corpus (OSQP's own benchmark suite, or Maros–Mészáros) would say more about where
the CHOLMOD backend is selected on real problems than the banded family used to test it.
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

**Indirect (matrix-free) solve: implemented**, as a package extension over
[Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl), so it costs nothing to a
caller who does not load Krylov. `linsys = :indirect` applies the reduced matrix through
the same `mul_A!`, `mul_At!` and `mul_P!` the iteration already uses and never forms it, so
a problem that can only supply matrix-vector products is solvable. `cg_max_iter`,
`cg_tol_fraction` and `cg_tol_reduction` control the inner solve, which follows the ADMM
residuals rather than running to a fixed tolerance; the preconditioner is the reduced
diagonal, assembled column by column without the matrix.

Which backend is faster is a question about the problem, not about the solver, and the
answer turns over. On **dense** problems the direct solve wins everywhere measured: the
inner solve costs about **23×** more per iteration at both `n = 50, m = 100` and
`n = 200, m = 400`, and being inexact it can cost iterations too, for 15× and 49× total.

On **sparse** problems it crosses near `n = 1900`. The direct backend now forms the reduced
matrix from the stored entries, so what remains irreducibly dense is the `n×n` matrix it
factors and inverts, at `O(n³)` however sparse the input; the matrix-free backend pays
`O(nnz)` per CG iteration and stores only vectors. Holding about five nonzeros per row of
`A` and growing the problem (`bench/indirect_backend.jl`, `eps = 1e-6`, single-threaded):

| n | m | density | direct | matrix-free | speedup | direct | matrix-free |
|---|---|---|---|---|---|---|---|
| 200 | 400 | 2.5% | 6.59 ms | 76.8 ms | 0.09× | 0.5 MiB | 0.19 MiB |
| 500 | 1000 | 1% | 31.0 ms | 209 ms | 0.15× | 2.4 MiB | 0.49 MiB |
| 1000 | 2000 | 0.5% | 103 ms | 326 ms | 0.32× | 8.5 MiB | 0.97 MiB |
| 2000 | 4000 | 0.25% | 702 ms | 640 ms | **1.10×** | 32.4 MiB | 1.96 MiB |
| 3000 | 6000 | 0.17% | 2244 ms | 1336 ms | **1.68×** | 71.5 MiB | 2.98 MiB |
| 4000 | 8000 | 0.125% | 4424 ms | 2328 ms | **1.90×** | 125.7 MiB | 3.86 MiB |

The last two columns are the workspace, and they are the more durable point: it shrinks by
3× at the top of the table and 33× at the bottom, because the direct backend stores an `n×n`
inverse where the matrix-free one stores vectors. Density decides it at fixed size — at
`n = 1000, m = 2000` the matrix-free backend is 1.12× ahead at 0.2% density and 0.07× at 2%.

So the rule is: dense or small, use the factorization; large and genuinely sparse, the
matrix-free backend is faster and much smaller; and when the matrix cannot be formed at all
it is the only option. Objectives agree to about eight digits throughout, which is the
inexactness showing up where it should.

Note these numbers have already moved once. Before the direct backend formed `R` sparsely
the matrix-free backend measured 3.61× at `n = 4000` rather than 1.90×, and 96× smaller
rather than 33×. A sparse *factorization* of `R` would move them again.

Two properties survive the extension. The per-iteration solve allocates nothing, which
needs `cg!`'s workspace preallocated *and* its lazily-allocated preconditioned vector filled
in at construction, since `cg!` otherwise allocates that one on first use. And the whole
path stays `--trim` compatible, which is not automatic: Krylov formats its verbose output
with `Printf`, and that is only acceptable to the trimmer because it writes to a concretely
typed `Core.CoreSTDOUT` rather than to `Base.stdout`.

One guarantee is checked differently here. `bench/strictmode_audit.jl` cannot ask AllocCheck
to clear `cg!`: it times itself through an opaque `ccall` to `jl_hrtime`, and its
verbose-reporting and residual-history branches are guarded by runtime values, so their
`Printf` and `resize!` calls are live code to a static analyzer even though no solve takes
them. Rather than whitelist those findings, the audit proves what this package owns and
measures what it does not — `ReducedOperator`'s `mul!` carries the full static `noalloc`
guarantee with no exemption, and `solve_system!` is measured at zero bytes.

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

`check_dualgap`, `scaled_termination`, `rho_is_vec` and the three `cg_*` settings are
implemented.

`check_dualgap` defaults on, matching libosqp 1.x. It binds rarely on well-scaled problems,
where the residuals already imply a small gap, but it does bind: across a sweep of badly
scaled objectives it changed the iteration count in 114 of 600 comparable runs. Anything
comparing iteration counts against libosqp 0.6.2 pins it off, since 0.6.2 has no such test
— that includes the oracle tests, the ported C-suite case and the benchmark scripts.

What remains:

- `linsys_solver` selection across QDLDL, MKL Pardiso and CUDA backends. The equivalent
  choice here is `linsys = :auto | :kkt | :indirect`, which selects a formulation rather
  than a library.
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


**Sparse linear algebra: implemented, on both axes.** The reduced matrix is *formed* from
the stored entries for a sparse `A` below 10% density, and *factored* by CHOLMOD through
SparseArrays whenever its Cholesky factor stays sparse enough to beat a dense inverse —
worth 37× on a banded problem at `n = 2000`. Both gates are measured, and the top of this
page gives the numbers.

Note the earlier verdict this does not contradict: what was measured and rejected was a
sparse LDLᵀ of the *full* KKT against the dense reduced solve, which is a different matrix.
See [How the sparsest case was closed](@ref "How the sparsest case was closed") for those
fill-in measurements.

## Suggested order

What is left of the page's opening item is structured `P`: a `Diagonal` or `Tridiagonal` is
still read through generic column traversals that walk its structural zeros. A sparse `A` is
now carried all the way — formed from its stored entries and, when the factor stays sparse,
factored by CHOLMOD rather than densified.

The matrix-free backend remains the fallback for a matrix that cannot be formed at all, and
its measured margin has now narrowed twice as the direct path improved.

What is left otherwise is `update_time`, the primal-dual integral, and the settings that
select a linear-algebra library or a GPU, none of which have a counterpart here.

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
