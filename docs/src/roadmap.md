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
| agreement with libosqp 1.x | no check exists; it has no Julia wrapper, and `OSQP_jll` v100 ships the library and its headers, so the route is a `ccall` |
| `polish` and derivatives on GPU | both build a dense `(n+k)×(n+k)` matrix and factor it with `bunchkaufman!`, so both stay on the host |
| a pure-Julia factorization by default | the `LDLᵀ` backends need LDLFactorizations.jl loaded; without it the sparse path is CHOLMOD, which is C and GPL |
| setup parity on the dense path | Control's `setup` is 0.33× libosqp's, against 1.20× on the run as a whole |

**Agreement with libosqp 1.x.** Everything this package checks against a reference checks
against 0.6.2, through OSQP.jl. 1.x has no Julia wrapper, so nothing here is evidence about
it, and the duality-gap default this package follows is taken from its documented behaviour
rather than from a measurement.

The route is a `ccall`. `OSQP_jll` v100 is libosqp 1.0.0 and its artifact ships
`libosqp_builtin_double.so` beside the full headers, so `OSQPSettings` — 65 fields — can be
mirrored from the header rather than guessed, and the mirroring checked by comparing
`sizeof` against the library. What that would buy is the iteration-count comparison the 0.6.2
tests already make, against the version whose termination rules this package actually
implements.

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
