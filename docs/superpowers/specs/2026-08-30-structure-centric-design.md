# PureOSQP.jl — structure-centric design

A change of organizing principle. Today the solver asks *"does sparsity pay here, and if not,
densify"*. It should ask *"what structure does this problem have, and what is the cheapest way
to solve with it — preferably without forming a matrix at all"*, with dense and sparse as the
last two rungs rather than the destination.

## Objective

**Structure-centric: choose the cheapest representation the structure admits, and treat forming
an `O(n²)` object as a cost to be justified rather than the default.**

Forming a dense matrix is *permitted*, not prohibited — it is sometimes the cheapest thing
available, and the package's own measurements say so: `ext/PureOSQPKrylovExt.jl` records that a
direct factorization is "some two orders of magnitude faster whenever the matrix *can* be
formed", and the matrix-free backend takes different iteration counts because its inner solve
is inexact. What is wrong today is not that dense formation exists; it is that dense formation
is the **default reached by falling through**, so its `O(n²)` memory and `O(n³)` factorization
get paid whenever nothing else volunteers — including on problems whose structure would have
made them `O(nb)`, `O(nk)`, or free.

So the objective is a change of *default and ordering*, not a prohibition. Dense stays, named
and last — **permitted, but discouraged: it should have to earn the choice rather than receive
it by default or by a tie.**

That has a concrete consequence, because **today's gates are biased the other way.** The fill
gate is `nnz(L) < DENSE_FACTOR_FILL · n²` with `DENSE_FACTOR_FILL = 0.05`: a sparse factor must
be under 5% of the dense size to win, so anything between 5% and 100% goes dense. The banded
backend defers at `b >= n ÷ 2`, i.e. dense wins from half-bandwidth upward. Both thresholds put
the thumb on dense's side of the scale.

Discouraging dense therefore means **re-deriving those thresholds against measurement rather
than inheriting them**, and the direction of the correction is testable: find, for each gate,
the crossover where the structured path actually stops winning, and place the threshold there
instead of where caution originally put it.

**Measured 2026-08-30. Both thresholds are too conservative, and the cost is large.**
Reproduce with `bench/gate_crossover_fill.jl` and `bench/gate_crossover_band.jl`; samples in
`bench/results/`.

| gate | current | measured | cost of the current setting |
|---|---|---|---|
| `nnz(L) < DENSE_FACTOR_FILL·n²` | `0.05` | parity near **0.23** | at `n = 1000`, `:auto` is **1.5–2× slower** than the sparse backend it declines — fill 0.086: 60 ms vs 31 ms; 0.121: 88 vs 57; 0.166: 85 vs 59 |
| banded declines at `b >= n ÷ 2` | `0.5·n` | **no crossover found** | banded still wins **1.6×** at `b/n = 0.8`, well past the cutoff |

So the fill gate is throwing away roughly a factor of two across a wide band of problems, and
the banded gate declines in a regime where the banded path is still winning by 1.6×. Neither
threshold is defended by measurement; both were caution.

**Moving them is a separate, deliberate change** — the ladder refactor is behavior-preserving
by construction, so these findings are recorded here and acted on next, not folded in.

Structure is a lattice, not a binary:

| level | what structure means | how it is known | cost to exploit |
|---|---|---|---|
| algebraic | banded, block-diagonal, Kronecker, low-rank + diagonal, circulant | **declared by type** | free — dispatch |
| sparse | a nonzero *pattern* | **discovered** by symbolic analysis | paid once at setup |
| dense | none | — | — |

Sparse is a weak structure: pattern-only, not algebraic. You cannot reason about a sparsity
pattern, only measure its fill, which is why every sparse path needs a symbolic analysis and a
gate. Algebraic structure states the *form of the factorization* up front, so exploiting it is
free and frequently means never materializing anything. The per-*refactorization* cost of the
sparse path was already engineered away — `ReducedGram`'s slot map pays the pattern analysis
once and refills in a single allocation-free pass — so the distinction is about the *selection*
cost, not the steady state.

**The lattice is a description, not a preference order.** A cost gate cuts across it: the
banded backend defers to dense at `b >= n ÷ 2`, so declared algebraic structure still loses to
dense when the band is wide enough. Structure says which *forms* are available; value-dependent
cost still decides among them. Any ladder built on this must carry both, and a design that
treats "algebraic beats sparse beats dense" as an ordering will pick wrong at the extremes.

The measured version of that distinction, from `bench/structured_backends.jl` at `n = 2000`:
`Diagonal`/`Diagonal` is **1710×** faster to set up than the dense path on the same problem —
not because diagonal matrices are small, but because the structure was declared, so there was
nothing to discover and nothing to form.

## Where the field is

No QP solver is organized this way. Surveyed 2026-08-30 against repository sources, not
recollection; all three Julia solvers below are Apache-2.0, as is this package.

| solver | linear-system choice | algebraic structure | source |
|---|---|---|---|
| libosqp | sparse QDLDL only | no | reference implementation |
| COSMO.jl | 8 KKT solvers, direct vs indirect | no; no user-supplied-solver interface | `src/linear_solver/kktsolver.jl`; docs `lin_solver` |
| Clarabel.jl | LDL registry, `:auto` selection | stated as *future* work | `src/kktsolvers/direct-ldl/directldl_defaults.jl` |
| PIQP | dense **and** sparse backends | no — the same binary this package has today | PREDICT-EPFL/piqp |
| MadNLP.jl | `AbstractKKTSystem`: dense/sparse/condensed, GPU | KKT *formulation* and hardware, not matrix structure | `docs/src/man/kkt.md` |

The honest exception is the MPC family — HPIPM, FORCES, acados — which genuinely exploit the
OCP staircase via Riccati recursion. They exploit *one* structure, hard-coded; adding another
means forking the solver.

So the claim to stake is not "exploits structure" but **"structure is an open lattice, extended
by dispatch rather than by forking"**. That is a Julia-shaped claim: structure-centric selection
*is* multiple dispatch on matrix types, which is why a C++ solver has not done it.

## Why this package can

`choose_backend` already dispatches on `typeof(P)` and `typeof(A)`, and the caller's matrices
are held by reference, uncopied, so the type survives to the point of decision. Every other
solver takes a `SparseMatrixCSC` or a dense array at the API boundary and erases the structure
before the solver sees it.

This is the mistake to avoid rather than copy. Clarabel declares the intent to exploit special
structure, then defines its LDL abstraction as *"update entries in the KKT matrix using the
given index into its CSC representation"* — an abstraction that forecloses every structure that
is not a sparsity pattern. Take their selection machinery; refuse their data representation.

## What exists already

| capability | state |
|---|---|
| backend interface that does not require a matrix | **done** — `LinearSystem` is `factorize!` + `solve_system!`, neither mentions a matrix |
| unmaterialized solve | **done** — `IndirectCG`: `ReducedOperator` applies `x ↦ (P̃ + σI + ÃᵀρÃ)x`; `factorize!` builds a Jacobi preconditioner from the reduced diagonal, computed column by column without assembling |
| algebraic backends | **done, 3** — `DiagonalReduced`, `TridiagonalReduced`, `BandedReduced` |
| structure survives to selection | **done** — dispatch on `typeof(P)`, `typeof(A)`; matrices never copied |
| entry-access as override points | **done** — `reduced_diagonal!`, `weighted_colmax_rowmax!`, `scaled_col!` are all generic-on-`AbstractMatrix` with extension overrides |

`reduced_diagonal!`'s docstring already anticipates this work: *"it is the one thing that
backend needs from `P` and `A` other than their products. A representation that cannot be
indexed overrides it with whole-matrix reductions."*

## The gap

Narrower than "everything materializes", and worth stating precisely.

- **Two backends form an `O(n²)` object**: `ReducedCholesky` and `SparseFormedInverse`. The
  structured three store `n` reciprocals, two bands, and `O(nb)` respectively — compressed, not
  dense. The LDLᵀ pair store sparse factors.
- **No block, low-rank or Kronecker backend exists.** That is the missing tier.
- **Every gate's failure branch densifies**: `(b < 2 || b >= n ÷ 2) && return ReducedCholesky`,
  plus `|| return nothing` gates falling through to the dense default.
- **`setup` densifies `P` before any backend is chosen.** `is_convex`'s generic method is
  `cholesky!(Symmetric(Matrix{T}(P) + σI))` (`src/linsys.jl`), called unconditionally at
  `src/types.jl:346`, while the backend branch is at `:392`. **This runs for
  `linsys = :indirect` too**, so the matrix-free path already pays an `O(n²)`-memory,
  `O(n³)`-time densification at setup for any `P` without an override. `IndirectCG`'s own
  `factorize!` "always succeeds", so it cannot absorb the check. Overrides exist for
  `Diagonal`, `SymTridiagonal`, `SparseMatrixCSC` and `BandedMatrix`; anything else pays.
- **`validate` calls `issymmetric(P)`** (`src/types.jl:296`) — an `O(n²)` entrywise scan
  generically, and a `MethodError` for a non-`AbstractMatrix` operator.
- **`polish!` indexes `P[i,j]`/`A[i,j]` and factors densely** (`src/polish.jl`).
- **`IndirectCG` is not reachable from `choose_backend` at all** — only by setting
  `linsys = :indirect` (`src/linsys.jl`). Whether it joins `:auto` is a decision this plan must
  make, not assume.

Selection ordering is explicit *within* the sparse method (density gate → KKT gate → CHOLMOD
gate → `SparseFormedInverse` → `FullKKT` on failure); what is implicit is only cross-extension
method specificity, and the type unions are currently disjoint by construction.

## Requirements

Tracked as a checklist. **Build** = must be written; **Reuse** = exists, needs wiring.

### S1 — Explicit selection ladder, dense terminal (Build)

**Not a `Val`-keyed registry.** An earlier draft proposed Clarabel's
`backend_constructor(::Val{:name})` / `backend_is_available` shape. That is wrong here, for
three reasons:

1. It contradicts this plan's own thesis. The claim staked below is that structure-centric
   selection *is* multiple dispatch — then replacing dispatch with a hand-rolled runtime
   registry throws away the mechanism being claimed as the advantage.
2. Enumerating an open key set at runtime is dynamic dispatch, a `--trim` risk, and against
   this project's standing rule to resolve selection at compile time rather than through a
   runtime table.
3. **A pure `admits`/`cost` query cannot express what the gates do.** `sparse_kkt_backend` and
   `cholmod_backend` *factor the real equilibrated matrix* as their gate and return
   `(backend, true)` so that `setup` skips `factorize!` — deliberately, and documented in
   `choose_backend`'s docstring. A query with no channel for "and here is the factorization"
   forces either a second factorization after selection (roughly doubling setup on every
   sparse-factored class, which would push SVM 1.11×, Lasso 1.07× and Huber 0.98× setup below
   parity) or caching inside `admits`, which is the current gate with a new name.

   Related: `backend_cost` as a pure function of a structure description **cannot exist** for
   the sparse rungs. Their decision variable is `nnz(L)` of a *completed* factorization
   (`nnz(LD) < DENSE_FACTOR_FILL * n²`). Nothing short of factoring predicts it.

**What S1 actually is:** keep `choose_backend`'s dispatch as the registry — it already is one —
and extract the gate chain currently buried inside the sparse method into a shared, explicit
ladder whose steps carry the existing `(backend, factored)` return, so a gate that factored
keeps its factorization. `ReducedCholesky` becomes the ladder's named terminal rung rather than
the thing reached by falling through.

**Decided: `IndirectCG` joins the ladder as its terminal fallback, below dense.** It is selected
only when nothing else can serve — a lazy operator with no matching structured backend — so no
problem that reaches a materializing backend today changes path, and no existing benchmark row
moves. Competing it against dense on a measured crossover is a later question, deliberately not
taken now, because its inexact inner solve changes iteration counts and would touch the
parity-with-libosqp story.

**Decided, defaults taken:**

- `polish` is **refused for lazy operators** with a clear error naming the reason, rather than
  reaching `polish!`'s entrywise indexing and failing with a `MethodError`.
- `scaling = 0` is the **documented interim restriction** for lazy operators between S4 and S6,
  not a silent degradation.
- **S5 lands with S3**, not before it: adding update-channel methods before a backend benefits
  from them is speculative interface work, so the Woodbury backend shapes the interface.
- **Gate-threshold re-derivation joins phase 1**, as measurement only — the thresholds
  themselves change in a separate, deliberate commit, because S1 must stay behavior-preserving.

**Verification** — not "benchmark figures unchanged", which is not a checkable procedure given
that idle machine load has moved setup ratios more than code changes have. Instead: assert the
*chosen backend per suite class* in a test, and establish by construction that no new work is
added to the selection path.

### S2 — A structure description (Build)
One vocabulary replacing the ad-hoc `banded_bandwidth`, `reduced_bandwidth`, `densest_row`,
`reduced_nnz`: bandwidth, block partition, low-rank coupling rows, Kronecker factors, none.

Governing rule, because setup time is already a deficit (Control is 0.33× libosqp's, an open
roadmap item, and the existing fill gate alone costs 44 µs there — 14% of libosqp's entire
setup):

> **Structure declared by type is free. Structure discovered by analysis is paid for, once,
> behind a gate.**

Type-level detection first; analysis only for representations that carry no structure in their
type.

### S3 — Unmaterialized structured backends (Build)
The new tier. No interface change needed; `IndirectCG` is the existence proof.

- **block-diagonal** — disjoint column support decouples `R` into K independent systems; K small
  solves, never an `n×n` object
- **low-rank + structured → Woodbury / Sherman–Morrison** — k coupling rows give
  `Ãᵀdiag(ρ)Ã = (structured part) + Uᵀdiag(ρ_k)U`; solve the structured part, correct through a
  k×k system, `O(n·k)` rather than `O(n³)`
- **Kronecker** — factored solves via small eigendecompositions

### S4 — Relax the `AbstractMatrix` bound and define the operator protocol (Build, **not small**)
`Workspace{T, MP <: AbstractMatrix, MA <: AbstractMatrix, …}` rejects any operator that is not
an `AbstractMatrix` subtype — `LinearMap`, `AbstractOperator`, `SciMLOperator` are all separate
hierarchies.

The protocol is larger than the products. A lazy operator must satisfy, or the path must be
closed off for it:

| need | where | today |
|---|---|---|
| `size` | everywhere | — |
| `mul!`, adjoint `mul!` | `mul_A!`, `mul_At!`, `mul_P!` | matrix-free already |
| `reduced_diagonal!` | the CG preconditioner | override point exists, documented |
| column primitives | `equilibrate!` | override point exists — **but see S6** |
| **`is_convex`** | `setup`, unconditionally, before backend choice | **densifies generically** |
| **`issymmetric`** | `validate` | `O(n²)` scan; `MethodError` on an operator |
| **entry access + dense factor** | `polish!` | no path for an operator |

The last three were missing from an earlier draft and are why this is not a small change.
`polish!` may legitimately be refused for lazy operators (`polish = false`), but that must be a
stated decision with a clear error, not a `MethodError`.

**S4 is inert until S6.** A relaxed bound reaches `equilibrate!`'s column walks immediately, so
until the equilibration protocol lands, S4 delivers a usable path only under `scaling = 0`.
Either reorder S4 after S6, or accept and document that restriction.

### S5 — Split the cheap update paths out of `factorize!` (Build, small)
Take COSMO's distinction: "only ρ moved, do the cheap update" versus "data changed, rebuild".
`factorize!` currently conflates them and gets away with it because `ReducedGram`'s slot map
makes the rebuild cheap. A block or Kronecker factorization may update in `O(n)` when only ρ
moves, and there is no way to express that today.

**Also `update!` of `P`/`A` values**, which an earlier draft omitted. The sparse backends
already validate their slot maps against pattern changes; a structured or lazy backend needs a
defined response to changed data — refill, rebuild, or refuse.

### S6 — Equilibration protocol for lazy operators (Build, hardest — defer)
Ruiz needs row and column ∞-norms of `P` and `A`. Materialized: walk entries. Lazy: cannot.
Four options:

- **deterministic basis probing** — `A·eⱼ` yields column `j` exactly, so `n + m` matvecs give
  exact ∞-norms, **bit-identical to walking the entries**, with no addition to the operator
  protocol. Paid once at setup, and cheap per product for exactly the structured operators in
  question. Whether `n + m` products is affordable is measurable, not decidable here — but this
  belongs first on the list, not omitted as it was in an earlier draft.
- the operator supplies its norms — trivial for `Diagonal`, per-block for block operators;
  needs a protocol addition
- random probing — changes results, and is therefore worse than basis probing for no gain
- `scaling = 0` — measurable loss on badly scaled problems

**Which invariant is at risk, corrected.** An earlier draft claimed probing would break
"iteration counts identical to libosqp". It cannot: libosqp can only receive materialized
input, so a lazy operator has no libosqp baseline to differ from. What probing actually
threatens is **lazy-vs-materialized self-consistency** — that the same problem expressed two
ways solves identically. That is a valuable invariant and should be an explicit test; it is not
the libosqp one.

### S7 — Backend introspection (Reuse + Build, trivial)
Clarabel's `linear_solver_info` is richer than `backend_name`. Useful for the docs and the
benchmark table.

## Invariants that must not break

- **iteration counts identical to libosqp 0.6.2** — the strongest correctness evidence, and it
  depends on equilibration staying bit-comparable, hence S6's risk
- **`noalloc` + `typestable` on the hot path** — subtle: a block backend holding
  `Vector{Cholesky{…}}` is concretely typed and fine; one holding *heterogeneous* block types is
  not, and would lose the guarantee
- **`--trim` compatibility** on every public path
- **the dense-regime numbers** — structure-first selection must not tax the dense path; S1 being
  behavior-preserving is what protects this

## Phasing

| phase | work | risk | delivers |
|---|---|---|---|
| 1 | S1 + S7 — the ladder, dense terminal | low; checked by asserting the backend chosen per suite class | no behavior change at all |
| 2 | S2 — structure vocabulary, type-level first | low | selection reads one description |
| 3 | S5 — split the ρ and data update paths | low | cheap-update channel for later backends |
| 4 | S3 — first structured backend (Woodbury over a diagonal core) | medium | the new tier, on `AbstractMatrix` types |
| 5 | S4 — relax the bound, operator protocol | medium | **lazy operators, but `scaling = 0` only** |
| 6 | S6 — equilibration protocol | high | lazy operators with scaling |

**S4 moved after S3 and is marked partial deliberately.** A relaxed bound reaches
`equilibrate!`'s column walks immediately, so between phases 5 and 6 a lazy operator is usable
only unscaled. Doing S3 first means the new backend tier is proved on ordinary `AbstractMatrix`
types — where equilibration, `is_convex`, `issymmetric` and `polish!` all still work — before
the protocol work destabilizes any of them.

For phase 4, start with **rank-`(k+1)` Woodbury over a diagonal core**, and be precise about
the motivating problem, because an earlier draft got it wrong.

Portfolio (`bench/suite_problems.jl`, `k = 5`) is:

    P = blockdiag(2D, 2I)              # D = spdiagm(...) — P is diagonal
    A = [ 1ᵀ  0 ;  Fᵀ  −I ;  I  0 ]    # F = sprandn(500, 5, 0.5)

so `Ãᵀ diag(ρ) Ã` decomposes as **diagonal core** (from the `[I 0]` block, and `P` is already
diagonal) **plus rank 1** (the budget row) **plus rank `k = 5`** (the factor rows, each ~250
nonzeros). That is rank **6**, not rank 1. A rank-1 Sherman–Morrison correction leaves the five
factor rows behind, and their outer products alone make the remainder roughly 76% dense — the
"structured part" would still be dense, and the exploit would buy nothing.

Two further corrections to the earlier draft, both checked against the code:

- **The dense budget row is not "why the class is routed to `ldl_kkt`".** The gate is
  `densest_row(A)² < DENSE_FACTOR_FILL · n²` with `DENSE_FACTOR_FILL = 0.05`. A single factor
  row (~250 nonzeros, 250² = 62 500) trips it on its own against 0.05·505² ≈ 12 751. Delete the
  budget row and Portfolio still routes to `ldl_kkt`.
- **Reachability is zero as the benchmark stands.** The suite hands over `SparseMatrixCSC`, so a
  backend keyed on a *declared* low-rank type never fires on that row. Making it fire would
  require discovering low-rank coupling in a sparse matrix — paid analysis, on the one class
  whose setup is already only at parity (0.99×), which is exactly what S2's governing rule
  warns against.

**So the phase-4 deliverable is a declared operator type, validated against Portfolio
re-expressed in it — not a change to how the benchmark's CSC input is handled.** The benchmark
row stays on `ldl_kkt`; the new backend is measured against the same problem given its
structure explicitly. State the expected margin honestly: the incumbent is already 1.08×
per-iteration with a factor holding 3 305 nonzeros and essentially no fill, and a rank-6
Woodbury solve costs `O(n·k)` ≈ 6–7k flops against the sparse solve's ~2·nnz ≈ 6.6k. **Per
iteration this is close to a wash.** The win, if there is one, is in setup and memory, and the
phase should be judged on that.

## Validation

- **S1 is a refactor**: assert the *backend chosen per suite class* in a test, and establish by
  construction that the selection path does no new work. Not "benchmark figures unchanged" —
  medians are not a CI-checkable invariant, and this repository's own notes record idle machine
  load moving setup ratios further than code changes did.
- **Gate thresholds are findings, not settings.** Each retuned threshold needs the crossover
  measurement that justifies it, saved under `bench/results/` like every other datapoint.
  Assert the chosen backend per suite class in a test.
- **Each new backend**: bands or blocks checked against the densely-formed reduced matrix, and
  the solve checked against the full KKT system solved by `\`, as `test/banded_tests.jl` does.
- **Iteration counts** against libosqp on the OSQP suite, unchanged.
- **StrictMode**: every backend carries `typestable, noalloc` on `admm_step!`,
  `update_residuals!` and `solve_system!`, or states its exemption tier explicitly.
- **A structure-centric crossover study.** The ~10% density figure is where *sparse* stops
  paying. The structure-centric claim needs its own evidence: for which classes does declared
  structure beat both dense and sparse, and by how much. Three synthetic families is a start,
  not a corpus.

## Open questions

- **What destroys the structure being exploited?** Two threats, not one.
  - **`ρ`**: `AᵀA → AᵀρA`. ADMM's `ρ` is split by constraint type, so it is piecewise-constant
    rather than arbitrary — that may survive for Kronecker, and it is the difference between
    viable and not.
  - **Equilibration's `D`/`E`**, which an earlier draft wrongly cleared as benign. They *are*
    benign for banded, block and diagonal-plus-low-rank: diagonal scaling preserves bandwidth
    and block partition. They are **not** benign for Kronecker — `diag(E)(B ⊗ C)diag(D)` is not
    a Kronecker product unless `E` and `D` themselves factor as Kronecker, which Ruiz will not
    produce. **Equilibration threatens the Kronecker rung before `ρ` ever does.**

  `σI` is harmless to everything.
- **Is detection worth its setup cost on the sparse path?** S2's rule says type-declared
  structure is free, but a richer analysis for `SparseMatrixCSC` makes an already-deficient
  setup worse.
- **Does a heterogeneous block backend cost the type-stability guarantee**, and is a homogeneous
  restriction acceptable?
- **How much of this belongs in PureOSQP versus a separate operator package?** The lazy
  structured operator is useful to more than one solver; the backends are not.

## Resolved: the operator library

**LinearMaps.jl first, as a weak dependency, used at the leaves only. SciMLOperators.jl kept as
a second adapter.** The protocol itself stays library-agnostic.

Measured 2026-08-30, `n = 400`, `m = 600`, sparse `A` at 5% density:

| | leaf `mul!` alloc | leaf `mul!` time | vs raw | `--trim` | transitive packages |
|---|---|---|---|---|---|
| raw `SparseMatrixCSC` | 0 B | 3.63 µs | — | OK | — |
| LinearMaps `WrappedMap` | **0 B** | **3.64 µs** | +0.3% | **OK** | 45 |
| SciMLOperators `MatrixOperator` | **0 B** | **3.73 µs** | +0% | **OK** | 56 |

Both are free at the leaf and both are trim-compatible; the expectation that SciMLOperators'
`DiffEqBase` dependency would break trim was wrong. The remaining differences are a 24% larger
dependency tree and, decisively for this plan, that **LinearMaps names both structures the
thesis needs** — `KroneckerMap` *and* `BlockDiagonalMap` — where SciMLOperators has
`TensorProductOperator` but no block type.

**Leaf-only, not composition.** Letting the library assemble `R` puts its whole algebraic
structure in one type, which is attractive, but composed application allocates:

| | alloc per apply | time |
|---|---|---|
| `LinearCombination`/`CompositeMap` assembling `R` | **9 744 B** | 10.34 µs |
| `ReducedOperator` assembling `R` from `Workspace` scratch | **0 B** | 10.06 µs |

There is no cache API in LinearMaps to remove those intermediates — that is what
SciMLOperators' `cache_operator` exists for — so composition would cost the `noalloc`
guarantee on every CG iteration, and is slower besides. `ReducedOperator` already composes `R`
from preallocated `Workspace` buffers; the library is only ever called at a leaf. `R`'s
structure is then derived from the leaves, which is what `bandwidth(R) = max(bw(P), 2 bw(A))`
already does.

**Consequence for S5.** A protocol built on SciMLOperators' `update_coefficients!` cannot serve
a LinearMaps user. Library neutrality therefore *forces* the cheap-update channel to be
PureOSQP's own concept — see S5. That is a cost of neutrality, accepted deliberately.

**Ambiguity risk.** Two operator extensions each adding `choose_backend` methods are fine while
their type unions are disjoint, but a generic "any operator" fallback would collide — exactly
the ambiguity the banded backend threw on its first test run. Keep the unions disjoint and test
with both libraries loaded.
