# Guarantees

Three properties of this package are machine-checked rather than asserted: the
linear-system interface is a declared contract, the hot path is proven allocation-free and
type-stable, and the public entry points compile under `juliac --trim`.

## The `LinearSystem` contract

The factorization backend is an interface, declared with
[TypeContracts.jl](https://github.com/el-oso/TypeContracts.jl) and enforced at
precompilation:

```julia
@contract LinearSystem begin
    factorize!(::Self, ::Any)::Bool
    solve_system!(::Self, ::Any, ::Any, ::Any)::Nothing
end
```

`@verify ReducedCholesky trim_compat=true` and the same for [`FullKKT`](@ref) run when the
module precompiles, so a backend that is missing a method — or whose method infers to the
wrong return type — fails to load rather than failing at solve time.

The contract is covered by a test that gives it something to reject: a type that declares
the supertype and implements none of it must be reported as unsatisfied. Without that, a
contract can be satisfied vacuously and nobody notices.

To add a backend, subtype `LinearSystem`, implement the two methods, and `@verify` it.
Note the backend is fixed when the workspace is built, so it is part of the workspace's
type and every per-iteration call dispatches statically — there is no runtime branch on
which backend is in use.

## No allocation, no type instability

`bench/strictmode_audit.jl` uses [StrictMode.jl](https://github.com/el-oso/StrictMode.jl)
to check, for **both** backends:

| function | guarantees |
|---|---|
| `admm_step!` | type-stable, allocation-free |
| `update_residuals!` | type-stable, allocation-free |
| `solve_system!` | type-stable, allocation-free |
| `check_termination` | type-stable |
| `factorize!` | type-stable |
| `solve!` | type-stable |

Run it with `cd bench && julia --project=. strictmode_audit.jl`.

Two things about that audit are worth knowing, because both were learned the hard way:

**It runs in `:full` mode, not `:fast`.** The fast heuristic reports `admm_step!` and
`update_residuals!` as allocating when AllocCheck proves they do not. For this package the
cheap tier produces false positives, so it cannot be the gate.

**It asserts `checks_enabled()` before doing anything.** With StrictMode's checks disabled
every assertion expands to the bare call and the audit passes vacuously, printing exactly
like a clean run. `bench/LocalPreferences.toml` enables them, and the script refuses to
report a pass without confirming it.

Getting to a provable `noalloc` required rewriting the hot path's broadcasts as explicit
loops. Any `a .= b .+ c` leaves an `unaliascopy` branch — a copy that would only happen if
the destination aliased a source — which never fires at runtime but which AllocCheck must
count. Measured with `@allocated` the broadcasts were already 0 bytes; the loops are what
make that a *proof* rather than an observation.

## `--trim` compatibility

`juliac --trim` needs every call resolved statically. All public entry points are checked
with [TrimCheck.jl](https://github.com/JuliaLang/TrimCheck.jl), in
`test/trim_tests.jl`, so a dynamic dispatch or a reflective call cannot creep in unnoticed:

- `solve` with default settings, with `polish = true`, with `linsys = :kkt`, and with
  `scaling = 0`;
- `setup` → `solve!` → `update!` → `solve!`;
- `setup` → `solve!` → `warm_start!` → `solve!`.

The same test includes a **negative control** — an entry point that calls
`Base.return_types`, which the trimmer cannot resolve. It must be reported as *not*
compatible. A checker that passed everything would tell you nothing, and this is what
proves it is discriminating.

TypeContracts' `@verify` at module top level does not interfere: it runs at precompilation
and is unreachable from any entry point, so the trimmer removes it.

Requires Julia ≥ 1.12, which is also this package's `[compat]` floor — `--trim` does not
exist before it.

## What the guarantees cover on GPU arrays

GPU support is **matrix-free only**. `linsys = :indirect` is the one backend whose inner
solve has a GPU counterpart — Krylov.jl's `cg!` is GPU-native, `bunchkaufman!` has no GPU
implementation, CHOLMOD is CPU by construction, and `potri!` reaches LAPACK. Any other
backend is refused at [`setup`](@ref) with a message naming the remedy, rather than left to
fail inside `factorize!`. `polish` and the derivatives stay on the host for the same
reason, and say so: both refuse a GPU workspace by name rather than surfacing a
scalar-indexing error from inside `bunchkaufman!`.

Two of the three guarantees above carry over unchanged, and one does not:

- **Type stability** holds and is checked.
- **`--trim`** is unaffected. Its entry points are concrete `Matrix{Float64}`/
  `Vector{Float64}` calls, so GPU code — reachable only through extension methods on GPU
  types — never enters the binary, the same scoping that keeps CHOLMOD's `ccall`s out.
- **Allocation-free does not extend to GPU workspaces**, and is not claimed for them. Every
  kernel launch allocates, and the broadcast and `mapreduce` schedules the GPU path uses
  allocate host-side besides. The claim is about `Vector`-backed workspaces, which is what
  `bench/strictmode_audit.jl` checks.

The elementwise work is written once as scalar functions with two schedules — an indexed
loop for `Vector`, a broadcast or reduction for everything else (`src/elementwise.jl`). The
loops are not a style preference: AllocCheck reports a broadcast as allocating even when its
destination is distinct from its sources, because `Base.Broadcast` keeps an `unaliascopy`
branch it cannot rule out, and it reports a multi-argument `mapreduce` the same way. Keeping
both schedules is what lets the CPU path stay provably allocation-free while the GPU path
avoids indexing.

**JLArrays is the correctness gate, and it is not a claim about CUDA.** `test/gpu_tests.jl`
runs the solver under `JLArrays.allowscalar(false)`, which enforces exactly the discipline
CUDA.jl enforces, so a green run certifies that nothing on the solve path indexes an array
elementwise. It certifies nothing else: `JLArray <: StridedArray` is true and `cholesky!` on
one *succeeds* through CPU LAPACK, so a JLArrays test of a direct backend would pass while
the same dispatch died inside `unsafe_convert` on a CuArray. CuArray is designed for and
untested; the refusal above is what keeps that gap from being discovered as a wrong answer.
