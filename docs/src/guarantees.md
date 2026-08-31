# Guarantees

Three properties of this package are machine-checked rather than asserted: the
linear-system interface is a declared contract, the hot path is proven allocation-free and
type-stable, and the public entry points compile under `juliac --trim`.

## The `LinearSystem` contract

The factorization backend is an interface, declared with
[TypeContracts.jl](https://github.com/el-oso/TypeContracts.jl) and enforced at
precompilation:

```@eval
using PureOSQP, TypeContracts, Markdown
Markdown.parse(replace(contract_md_string(PureOSQP.LinearSystem), r"\A# [^\n]*\n+" => ""))
```

[`PureOSQP.refactor_rho!`](@ref) is outside the contract: it refreshes the factorization when
`ρ` alone has moved, and rebuilding from scratch is a correct answer, so its default is
[`PureOSQP.factorize!`](@ref) and no backend has to implement it.

`@verify` runs when the module precompiles, so a backend that is missing a method — or whose
method infers to the wrong return type — fails to load rather than failing at solve time.
The backends verified in the core module, which is what is loaded here (each extension
verifies its own):

```@eval
using PureOSQP, TypeContracts, Markdown
verified = String[]
for m in methods(verified_trait)
    m.sig isa DataType && length(m.sig.parameters) == 3 || continue
    p = m.sig.parameters[3]
    p isa DataType && p <: Type || continue
    t = p.parameters[1]
    t isa TypeVar && continue
    push!(verified, string(nameof(t)))
end
Markdown.parse(join(("  - `" * v * "`" for v in sort!(verified)), "\n"))
```

The contract is covered by a test that gives it something to reject: a type that declares
the supertype and implements none of it must be reported as unsatisfied. Without that, a
contract can be satisfied vacuously and nobody notices.

To add a backend, subtype `LinearSystem`, implement the mandatory methods above, and
`@verify` it.
Note the backend is fixed when the workspace is built, so it is part of the workspace's
type and every per-iteration call dispatches statically — there is no runtime branch on
which backend is in use.

## No allocation, no type instability

`bench/strictmode_audit.jl` uses [StrictMode.jl](https://github.com/el-oso/StrictMode.jl)
to check, for every backend the audit can reach:

| function | guarantees |
|---|---|
| `admm_step!` | type-stable, allocation-free |
| `update_residuals!` | type-stable, allocation-free |
| `solve_system!` | type-stable, allocation-free |
| `check_termination` | type-stable |
| `factorize!` | type-stable |
| `refactor_rho!` | type-stable |
| `solve!` | type-stable |

Two qualifications the audit makes and this page inherits. A backend reaching sparse
arithmetic gets no static allocation claim for its factorization, because the sparse
libraries it calls carry allocation sites of their own. And the matrix-free backend's hot
path is checked by **measurement** rather than statically: Krylov's `cg!` holds timing and
buffer-allocation branches that are visible to a static analysis and never taken, so the
static check reports them and a measured run reports zero bytes.

An operator supplied by the caller is guaranteed only as far as its own `mul!` is: these
properties are the solver's, and a product that allocates makes the iteration allocate.

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

`juliac --trim` needs every call resolved statically. Two things enforce that, and they cover
different halves of the surface.

**Every backend, as a filter.** `src/PureOSQP.jl` carries

```julia
@verify LinearSystem subtypes = true trim_compat = true
```

which checks every [`PureOSQP.LinearSystem`](@ref) subtype at precompilation. An extension's
backends load after that sweep and cannot be swept the same way (re-registering the core
backends is method overwriting during precompilation), so `test/coverage_tests.jl` runs the
same check over every subtype with all extensions loaded, with no exemption list.

Be precise about what that buys, because it is less than it looks. The check scans each
backend's declared method bodies and does not follow calls out of them, so it catches a
backend that itself names something unresolvable and nothing deeper — the CHOLMOD backends
pass it while reaching SuiteSparse through `ccall`. It also reports by warning rather than by
raising, so it is the test that enforces it and not the load. The test pins both properties: it
reads the returned verdict rather than waiting for an exception, and it checks a deliberately
trim-unsafe backend is rejected, so a sweep that stopped discriminating would be caught.

The load-bearing guarantee is the entry-point enumeration below, which runs the trimmer.

**Entry points, per path.** `test/trim_tests.jl` validates a concrete call for each public
path with [TrimCheck.jl](https://github.com/JuliaLang/TrimCheck.jl):

- `solve` with default settings, with `polish = true`, `linsys = :kkt`, `linsys = :indirect`,
  `scaling = 0`, `verbose = true`, `max_iter` and `time_limit`;
- one per structured representation — diagonal, tridiagonal (both spellings), banded,
  low-rank, block-diagonal, Kronecker, and a products-only operator;
- sparse operands: `SparseMatrixCSC` on both sides with `linsys = :kkt` and with
  `linsys = :dense`, and a sparse `A` under a diagonal `P` on `:auto`;
- `setup` → `solve!` → `update!` → `solve!`, and the same with `warm_start!`;
- `update_settings!`/`update_rho!`/`cold_start!`, and the derivatives.

That list is an enumeration, so it proves what it names and no more. The backend sweep above
is what makes coverage of the *backends* structural rather than remembered.

**What a sparse problem needs to be AOT-compilable.** Two conditions, both about keeping
SuiteSparse out of the binary rather than about the backends, which pass the sweep above like
every other.

*Name the backend.* `linsys` reaches [`setup`](@ref) as a type parameter, so `:kkt`, `:dense`
and `:indirect` eliminate the selection ladder instead of leaving it unused. Left on `:auto`, a
pair that is sparse on both sides is incompatible by construction, and for two independent
reasons — removing either one leaves the other. That ladder has to consider the reduced rung,
which factors with CHOLMOD, whose bindings reach their C structs through `getproperty` on
pointer-backed wrappers that no static analysis resolves. And it reaches more backend types
than inference will merge: `Base.Compiler.MAX_TYPEUNION_LENGTH` is 3, so the workspace type
widens to `Workspace{…} where LS` and every later `solve!` becomes a dynamic dispatch. A sparse
`A` under a structured `P` is compatible on `:auto`, because the rungs that pair descends reach
neither a sparse factorization nor a fourth backend type.

*Load LDLFactorizations.* [`PureOSQP.is_convex`](@ref) needs a factorization to answer for a
sparse `P` that is not diagonal, and only a pure-Julia one is resolvable. With the extension
loaded, dispatch settles which is used before the trimmer runs, so the CHOLMOD fallback is
unreachable; without it, that fallback is the only answer and the path is not compatible.

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
