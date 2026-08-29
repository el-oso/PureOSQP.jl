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
| `update_time` | `update!` is not timed; `setup_time`, `solve_time`, `polish_time` and `run_time` are |
| `primdual_int` | the primal-dual integral, a 1.x convergence diagnostic requiring per-iteration profiling |
| a structured backend | a `Diagonal` or band type is read efficiently, but the reduced matrix it forms is dense unless `A` is structured too |
| a sparse full-KKT factorization | a single dense row in `A` makes the reduced matrix dense however sparse `A` is; see below |
| `polish` and derivatives on GPU | both build a dense `(n+k)×(n+k)` matrix and factor it with `bunchkaufman!`, so both stay on the host |

**CUDA in practice.** GPU arrays solve through `linsys = :indirect`, and only through it —
see [Guarantees](@ref) for the scope and why the other backends are refused at
[`setup`](@ref). What is untested is a real device: `test/gpu_tests.jl` runs against
JLArrays, which enforces the same no-scalar-indexing discipline CUDA.jl does and is a sound
gate for exactly that. It is not a gate for anything else, because `JLArray <: StridedArray`
is true and `cholesky!` on one succeeds through CPU LAPACK. Running it on a CuArray is the
step nobody has taken, and the likely findings are in the pieces JLArrays cannot exercise:
streams, `Krylov.cg!`'s device behavior, and how much the per-iteration device
synchronizations in `check_termination` cost.

Note also what a GPU is worth here. The OSQP authors' own CUDA port, cuOSQP, targets
`nnz ≥ 1e4` and reaches its peak speedup only near `nnz ≈ 1e8`. Below that a single CPU core
wins, which is why the direct backends were not ported rather than why they could not be.

**A sparse full-KKT factorization.** Eliminating to the reduced system squares `A`, so one
dense row makes `R` dense however sparse the rest of it is. Measured on the OSQP benchmark
suite's Portfolio class, whose budget constraint `1ᵀx = 1` is exactly that row: `A` is 0.9%
dense, `R` is 99% dense, and PureOSQP runs at 0.18× of libosqp there. Upstream does not have
the problem, because a sparse LDLᵀ of the full `(n+m)×(n+m)` system keeps that row as one
sparse row.

**A structured backend.** `Ãᵀ diag(ρ) Ã` fills in whatever `P` looked like, so structure in
`P` alone does not survive into the reduced matrix. It pays only where both are structured —
`Diagonal` `P` with `Diagonal` `A` makes the whole solve `O(n)` — and no such problem has
turned up to justify the backend. [`choose_backend`](@ref PureOSQP.choose_backend) is where
it would go.

## Deliberate differences

**Code generation.** `osqp_codegen` emits C source with the settings, scaled data, `ρ` and
numeric factorization baked into fixed-size file-scope arrays, for a toolchain that is not
Julia's. `--trim` gets close on the Julia side — `juliac --output-lib --compile-ccallable
--experimental --trim=safe` turns a `Base.@ccallable` wrapper around [`solve`](@ref) into a
3.7 MB shared library callable from a plain C `main` with no `jl_init` — but it emits machine
code needing about 89 MB of Julia runtime beside it, and Julia does not exist for a Cortex-M
part, a fixed-point DSP, or a DO-178C-qualified toolchain. Matching it honestly means a
separate C library, not a code generator in this package.

There is a narrower target this package *is* shaped for, if it is ever wanted: the dense
reduced form with a pre-inverted `R` would emit a baked `n×n` inverse, one `symv` and `O(n+m)`
vector operations — a couple of hundred lines of dependency-free C, with none of the symbolic
factorization that makes upstream's generator large.

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

Both were learned the hard way.

Anything that prints or times has to respect the `--trim` guarantee, which analyses code
whether or not the branch reaching it is ever taken. Printf, bare `println` and `lpad` all
fail there; `verbose` is the worked example, writing through `Core.stdout` by hand — see
`print_padded` in `src/admm.jl`. Note a *concretely typed* stream is what matters: Krylov's
`@printf` to `Core.CoreSTDOUT` passes where `Base.stdout` would not.

`bunchkaufman!` is LAPACK-only for BLAS floats, with no generic fallback in `LinearAlgebra`,
and it is reached from `polish!`, from the `FullKKT` backend and from
[`adjoint_derivative`](@ref). So dual numbers run the solver on the reduced backend with
`polish = false`. `INFTY` is not a blocker despite calling `prevfloat(typemax(T))`:
ForwardDiff defines both for `Dual`.

An AD integration would be a `ChainRulesCore` extension supplying `rrule` and `frule` over
[`adjoint_derivative`](@ref) and [`forward_derivative`](@ref). That reaches Zygote; it does
*not* reach Mooncake, which requires an explicit `Mooncake.@from_rrule`.
