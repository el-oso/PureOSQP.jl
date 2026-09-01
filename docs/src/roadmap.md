# Roadmap

What libosqp does that PureOSQP does not, derived from its public API — `osqp_api.c`,
`osqp_api_types.h` and `osqp_api_constants.h` for 1.x, plus the 0.6.2 `osqp.c` surface —
against this package's exports, [`Settings`](@ref) and [`Solution`](@ref).

Implemented capabilities are described where they are demonstrated rather than here:
[Benchmarks](@ref) for what the backends cost, [Guarantees](@ref) for what is proven about
them, [Algorithm](@ref) for how the solver works.

## Open

| item | note |
|---|---|
| CUDA in practice | the GPU path is designed for it and tested only against JLArrays; nobody has run it on a device |
| `polish` and derivatives on GPU | both build a dense `(n+k)×(n+k)` matrix and factor it with `bunchkaufman!`, so both stay on the host |
| a pure-Julia factorization by default | the `LDLᵀ` backends need LDLFactorizations.jl loaded; without it the sparse path is CHOLMOD, which is C and GPL |
| setup parity on the dense path | Control's `setup` is 0.33× libosqp's, against 1.20× on the run as a whole |

**CUDA in practice.** GPU arrays solve through `linsys = :indirect` only — see
[Guarantees](@ref) for why the other backends are refused at [`setup`](@ref).
`test/gpu_tests.jl` runs against JLArrays, which gates no-scalar-indexing and nothing else:
`JLArray <: StridedArray` is true, so `cholesky!` on one succeeds through CPU LAPACK. What a
real device would exercise is streams, `Krylov.cg!`'s device behavior, and the per-iteration
synchronizations in `check_termination`.

cuOSQP, by the OSQP authors, targets `nnz ≥ 1e4` and peaks near `nnz ≈ 1e8`. Below that a
single CPU core wins, which is why the direct backends were not ported.

**A pure-Julia factorization by default.** A sparse problem is factored by CHOLMOD — C and
GPL — unless LDLFactorizations.jl happens to be loaded. The `LDLᵀ` backends are also the
faster path, 2.3–3.1× on the numeric factorization and allocating nothing, so the gap is which
one a caller gets without asking.

Closing it needs a sparse `LDLᵀ` this package can depend on outright; LDLFactorizations is
LGPL-3, hence the weak dependency. PureSparse.jl was measured and does not fit: it is
supernodal, built for dense panels, where these factors hold two to three nonzeros per column
— `solve!` 4–13× slower than the substitutions here, factorization 1.5–2.7× slower. A
simplicial path there would change that.

**Setup parity on the dense path.** Forming and inverting `R` costs `potrf` (`n³/3`) plus
`potri` (`2n³/3`), and the inverse is what makes each iteration one `symv`. Measured and
rejected: a sparse `LDLᵀ` of the reduced matrix or of the KKT reaches setup parity and loses
more in the loop; two `trsv` is the same flop count but serial; `trtri` with two `trmv` halves
the setup and doubles the loop. Equilibration is within 1.13× of libosqp's per sweep. The
convexity test and the fill gate are fixed costs libosqp does not pay.

## Settings, against libosqp 1.x

Every upstream setting, and what it is called here. Defaults on the right are read from the
library through `bench/osqp_v1.jl`, not from
[the settings page](https://osqp.org/docs/interfaces/solver_settings.html), which is stale on
one of them: it gives `adaptive_rho_interval` as `0`, where the library ships `50`.

**Same name, same default.** `rho` `0.1`, `sigma` `1e-6`, `alpha` `1.6`, `scaling` `10`,
`rho_is_vec` `true`, `max_iter` `4000`, `eps_abs` and `eps_rel` `1e-3`, `eps_prim_inf` and
`eps_dual_inf` `1e-4`, `scaled_termination` `false`, `check_termination` `25`,
`adaptive_rho_interval` `50`, `adaptive_rho_tolerance` `5`, `cg_max_iter` `20`,
`cg_tol_fraction` `0.15`, `cg_tol_reduction` `10`, `delta` `1e-6`, `polish_refine_iter` `3`,
`warm_starting` `true`, `check_dualgap` `true`.

`check_dualgap` is in the 1.x C header and in the library's defaults, but not on that settings
page; this package follows the library.

**Renamed, or defaulted differently.**

| upstream | here | why |
|---|---|---|
| `polishing` | `polish` | name only; both default off |
| `verbose` = `true` | `verbose` = `false` | a library that prints by default is the wrong default for a package |
| `time_limit` = `1e10` | `time_limit` = `Inf` | the same "no limit", spelled as the thing it means |
| `profiler_level` | `profile_primdual` | one switch over the one measurement that needs a clock |
| `osqp_linsys_solver_type` | `linsys` | upstream selects a *library*, this selects a *formulation* — see below |
| `adaptive_rho` = `true` | `adaptive_rho` = `:iterations` | a mode rather than a flag; `true` is accepted and means `:iterations` |

**Same name, different meaning — the one to watch.** `adaptive_rho_fraction` is `0.4` in both,
and means different things. Upstream it is a fraction of *setup time*, feeding the wall-clock
adaptation mode. Here it is a fraction of the *previous KKT error*: under
`adaptive_rho = :kkt_error`, `ρ` is retuned only once `rel_kkt_error` has fallen to
`adaptive_rho_fraction` of what it was at the last look. Porting a tuned value across without
reading this will not error — it will quietly do something else.

**Upstream only.** `device` and `allocate_solution` are GPU and embedded concerns;
`cg_precond` selects a preconditioner where this package always uses the diagonal one.

## Deliberate differences

**Code generation.** `osqp_codegen` emits C source with the settings, scaled data, `ρ` and
numeric factorization baked into fixed-size arrays, for a toolchain that is not Julia's.
`--trim` gets close — `juliac --output-lib --compile-ccallable --experimental --trim=safe`
turns a `Base.@ccallable` wrapper around [`solve`](@ref) into a 3.7 MB shared library callable
from a plain C `main` with no `jl_init` — but it needs about 89 MB of Julia runtime beside it,
and Julia does not exist for a Cortex-M part or a DO-178C toolchain. Matching it means a
separate C library, not a generator here.

The narrower target this package is shaped for: the dense reduced form with a pre-inverted `R`
emits a baked `n×n` inverse, one `symv` and `O(n+m)` vector operations — a few hundred lines
of dependency-free C, with none of the symbolic factorization that makes upstream's large.

**Wall-clock `ρ` adaptation.** Upstream has four modes; this package implements three, as
`adaptive_rho = :disabled | :iterations | :kkt_error`. The wall-clock mode is omitted because
it makes iteration counts depend on the machine, and reproducible counts are what the oracle
tests check.

**`linsys_solver`, `device`, `profiler_level`, `allocate_solution`.** Selection across QDLDL,
MKL Pardiso and CUDA backends, and embedded and GPU concerns. The counterpart here is
`linsys = :auto | :dense | :kkt | :indirect`, which selects a formulation; the library
underneath is whatever BLAS is loaded, so `using MKL` is the whole of the MKL story.

`profiler_level` gates upstream's timing annotations by level. The counterpart here is
`profile_primdual`, a single switch over the one measurement that needs a clock the solver
would not otherwise read — see [Benchmarks](@ref "The primal-dual integral").

**Driving `ρ` from the gap's decay rate. Measured and rejected.** The primal-dual integral's
log-mean rule computes `log(gₖ/gₖ₊₁)` — the duality gap's local decay rate — as a byproduct,
which is a more direct measure of progress than the relative KKT error `adaptive_rho =
:kkt_error` already triggers on. It buys nothing: on the seven benchmark classes that trigger
reaches the same tolerance in the same iterations, with the same refactorization count, as
retuning on a fixed interval ([Benchmarks](@ref "The ρ schedule")). Adapting `ρ` is worth
3.8×; *when* it is adapted is worth zero, so a better progress signal has nothing to improve.

**`osqp_error_message`.** Exists to turn an error code into a string. This package throws
exceptions carrying their own messages.

## Notes for whoever picks these up

Anything that prints or times has to respect the `--trim` guarantee, which analyses code
whether or not the branch reaching it is ever taken. Printf, bare `println` and `lpad` all
fail there; `verbose` is the worked example, writing through `Core.stdout` by hand — see
`print_padded` in `src/admm.jl`. A *concretely typed* stream is what matters: Krylov's
`@printf` to `Core.CoreSTDOUT` passes where `Base.stdout` would not.

`bunchkaufman!` is LAPACK-only for BLAS floats, with no generic fallback in `LinearAlgebra`,
and it is reached from `polish!`, from the `FullKKT` backend and from
[`adjoint_derivative`](@ref). So dual numbers run the solver on the reduced backend with
`polish = false`. `INFTY` is not a blocker despite calling `prevfloat(typemax(T))`:
ForwardDiff defines both for `Dual`.

An AD integration would be a `ChainRulesCore` extension supplying `rrule` and `frule` over
[`adjoint_derivative`](@ref) and [`forward_derivative`](@ref). That reaches Zygote; it does
*not* reach Mooncake, which requires an explicit `Mooncake.@from_rrule`.
