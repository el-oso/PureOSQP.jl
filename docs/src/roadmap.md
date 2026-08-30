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
| `primdual_int` | the primal-dual integral, a 1.x convergence diagnostic requiring per-iteration profiling |
| a structured backend | a `Diagonal` or band type is read efficiently, but the reduced matrix it forms is dense unless `A` is structured too |
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

**A structured backend.** `Ãᵀ diag(ρ) Ã` fills in whatever `P` looked like, so structure in
`P` alone does not survive into the reduced matrix. It pays only where both are structured —
`Diagonal` `P` with `Diagonal` `A` makes the whole solve `O(n)` — and no such problem has
turned up to justify the backend. [`choose_backend`](@ref PureOSQP.choose_backend) is where
it would go.

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
