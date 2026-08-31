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

That has a concrete consequence, because **an inherited gate is biased the other way.** The fill
gate is `nnz(L) < DENSE_FACTOR_FILL · n²` with `DENSE_FACTOR_FILL = 0.05`: a sparse factor must
be under 5% of the dense size to win, so anything between 5% and 100% goes dense. The banded
backend inherited a deferral at `b >= n ÷ 2`, dense winning from half-bandwidth upward.

Discouraging dense therefore means **deriving each threshold from measurement rather than
inheriting it**, and the direction of the correction is testable: find, for each gate, the
crossover where the structured path actually stops winning, and place the threshold below it.
Both gates now stand on that rule. The band limit is `4b > n`, measured, and the fill gate keeps
`0.05` because its measurement supports it — the log carries both, and neither number in this
paragraph is a live setting except `DENSE_FACTOR_FILL`.

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
banded backend defers to dense once the band is wide enough, so declared algebraic structure
still loses when the values say it should. Structure says which *forms* are available; value-dependent
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
- **Every gate's failure branch densifies**: `(b < 2 || 4b > n) && return dense_rung(…)`, plus
  `|| return nothing` gates falling through to the dense terminal.
- **`setup` densifies `P` before any backend is chosen.** `is_convex`'s generic method is
  `cholesky!(Symmetric(Matrix{T}(P) + σI))` (`src/linsys.jl`), called unconditionally at
  `src/types.jl:346`, while the backend branch is at `:392`. **This runs for
  `linsys = :indirect` too**, so the matrix-free path already pays an `O(n²)`-memory,
  `O(n³)`-time densification at setup for any `P` without an override. `IndirectCG`'s own
  `factorize!` "always succeeds", and a direct backend's cannot absorb the check either: it
  factors `P + σI + Ãᵀ diag(ρ) Ã`, which a large enough `ρ` makes positive definite over an
  indefinite `P`. Overrides exist for `Diagonal`, `SymTridiagonal`, `Tridiagonal`,
  `SparseMatrixCSC`, `BandedMatrix` and `Symmetric{<:Any,<:BandedMatrix}`; anything else,
  a GPU matrix included, pays.
- **`validate` tests symmetry through `is_symmetric`**, whose generic method is `issymmetric`
  — an `O(n²)` entrywise scan, and a `MethodError` for a non-`AbstractMatrix` operator.
  `BandedMatrix` overrides it to compare the band alone.
- **`polish!` indexes `P[i,j]`/`A[i,j]` and factors densely** (`src/polish.jl`).
- **`IndirectCG` is not reachable from `choose_backend` at all** — only by setting
  `linsys = :indirect` (`src/linsys.jl`). Whether it joins `:auto` is a decision this plan must
  make, not assume.

Selection ordering is explicit *within* the sparse method (density gate → KKT gate → CHOLMOD
gate → `SparseFormedInverse` → `FullKKT` on failure); what is implicit is only cross-extension
method specificity, and the type unions are currently disjoint by construction.

## Requirements

| item | state |
|---|---|
| S1 — explicit selection ladder, dense terminal | **done** |
| S2 — one structure description | **dropped** — `structural_rows` already is the shared description, and each of the four terms it would replace is read only inside the extension that defines it |
| S3 — unmaterialized structured backends | **done** for Woodbury over a diagonal core; block-diagonal and Kronecker unbuilt |
| S4 — relax the `AbstractMatrix` bound | **done, by a different route** — the bound is unchanged; `ProductOperator` presents a foreign hierarchy as an `AbstractMatrix`, `ext/PureOSQPLinearMapsExt.jl` accepts a `LinearMap`, and `is_materializable` authorizes the refusals |
| S5 — split the cheap update paths out of `factorize!` | **done** — `refactor_rho!`, defaulting to a full rebuild; `DiagonalLowRank` overrides it |
| S6 — equilibration protocol for lazy operators | **dropped** — replaced by the two documented seam levels, which an operator overrides at whichever level it can answer, and by the representation-independence test that checks the seam |
| S7 — backend introspection | **done** — `BackendInfo`, `backend_info`, `factor_fill` |


Tracked as a checklist. **Build** = must be written; **Reuse** = exists, needs wiring.

Defects against types the package already ships are tracked separately, since they are not new
capability. Each is confirmed by measurement before being fixed, and may close as "not a
defect"; the numbers are in the log. The last row is not a defect against the code — it is the
state of these two documents, which no test checks.

| item | state |
|---|---|
| banded-structural-rows — the column walk visits every row of a `BandedMatrix` | **done** |
| tridiagonal-rung — a `Tridiagonal` `P` reaches the dense terminal where the same band spelled `SymTridiagonal` reaches `TridiagonalReduced` | **done** |
| symmetric-sparse-admission — the sparse rungs admit a `Symmetric` wrapper their consumers cannot read | **done** |
| is-convex-coverage — two types the package ships a backend for reach `is_convex`'s densifying method | **done** |
| reduced-diagonal-structural-rows — the matrix-free preconditioner sums every row of `A` at every rho adaptation | **done** |
| lazy-materialized-agreement-test — no standing test asserts that one problem in two spellings equilibrates and solves alike | **done** |
| design-log-corrections — these documents record deferrals where decisions were taken, and quote figures no saved sample backs | **done** |

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
- `scaling = 0` is the **documented restriction** for an operator that overrides neither
  equilibration seam level, not a silent degradation.
- **S5 follows S3**, not the reverse: adding update-channel methods before a backend benefits
  from them is speculative interface work, so the Woodbury backend shapes the interface. It is
  that backend's `factorize!` — which rebuilds the whole correction on a ρ move that changes
  only the core and the capacitance — that the split is designed against.
- **Gate-threshold re-derivation joins phase 1**, as measurement only — the thresholds
  themselves change in a separate, deliberate commit, because S1 must stay behavior-preserving.

**Verification** — not "benchmark figures unchanged", which is not a checkable procedure given
that idle machine load has moved setup ratios more than code changes have. Instead: assert the
*chosen backend per suite class* in a test, and establish by construction that no new work is
added to the selection path.

### S2 — A structure description (dropped)
One vocabulary was to replace `banded_bandwidth`, `reduced_bandwidth`, `densest_row` and
`reduced_nnz`: bandwidth, block partition, low-rank coupling rows, Kronecker factors, none.

It is not built, and the shared description it would have introduced already exists in another
shape. `structural_rows` is what the four column traversals, the dense formation and the
matrix-free preconditioner all read, and it answers per column rather than naming a
whole-matrix noun, which is why every structured type here serves it and no type needs a
second description to be selected. The four terms above are each read only inside the extension
that defines them — `banded_bandwidth` and `reduced_bandwidth` in the banded extension,
`densest_row` and `reduced_nnz` in the sparse one — so a common vocabulary would have no second
consumer.

Its governing rule stands whatever carries it, and the rest of this document rests on it:

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
  k×k system, `O(n·k)` rather than `O(n³)`. **Built** as `RowCoupled` + `DiagonalLowRank`
  over a diagonal core; `bench/results/lowrank_backend.json` records 20–920× end to end
  against the **dense** path on the same problem. Against a **sparse KKT** path the margin is
  much smaller and storage goes the wrong way: on Portfolio re-expressed in the type, 1.87×
  setup and 1.18× per iteration, but 0.58× total under the default adaptive `ρ`, and 3556
  words stored against `ldl_kkt`'s 2322 (`bench/results/lowrank_portfolio.json`). The
  structured part is `Diagonal` only — a banded or tridiagonal core is the next step, and
  reuses the same capacitance apply.
- **Kronecker** — factored solves via small eigendecompositions

### S4 — The operator protocol (replaces the relaxed bound)
An operator declares itself `<: AbstractMatrix{T}` and supplies `size`, `mul!` and adjoint
`mul!`. Nothing about `Workspace{T, MP <: AbstractMatrix, MA <: AbstractMatrix, …}` changes,
and no operator library is a required dependency.

`LinearMap`, `AbstractOperator` and `SciMLOperator` are separate hierarchies, none below
`AbstractMatrix`, so none can be passed to `setup` directly. [`ProductOperator`](@ref) is the
wrapper that presents one as an `AbstractMatrix`: it holds the operator, forwards `size` and
both `mul!`s, and carries the two answers no product can give — whether the operator is
symmetric and whether `P + σI` is positive definite, both declared by the author. It lives in
`src/` and depends on nothing, so a second hierarchy costs one more small extension rather
than a rewrite.

`ext/PureOSQPLinearMapsExt.jl` is that extension for LinearMaps: `setup` and `solve` accept a
`LinearMap` in either position, mixing with matrices freely, and the wrapper reads `symmetric`
and `posdef` from the map's own `issymmetric`/`isposdef` traits rather than asking the caller.

The protocol is larger than the products. Each row below is an override point an operator
answers if it can, or a path closed off for it by name:

| need | where | how it is served |
|---|---|---|
| `size` | everywhere | required |
| `mul!`, adjoint `mul!` | `mul_A!`, `mul_At!`, `mul_P!` | required |
| `reduced_diagonal!` | the CG preconditioner | override point |
| column primitives | `equilibrate!` | two seam levels — per column (`structural_rows` and the four traversals) or per sweep (`column_norms!`, `cost_norms!`); without either, **refused by name** for a `ProductOperator`, which names `scaling = 0` and both seam levels |
| the reduced diagonal | the CG preconditioner | override point; a `ProductOperator` with no method runs unpreconditioned rather than failing, since Jacobi scaling costs iterations and not the answer |
| `is_convex` | `setup` (`src/types.jl:364`), unconditionally, before backend choice | override point; the generic method densifies |
| `is_symmetric` | `validate` (`src/types.jl:308`) | override point; the generic method is `issymmetric` |
| both again, on changed data | `update!` (`src/update.jl:36-37`) re-validates a new `P` | the same two override points, at a second call site |
| entry access + dense factor | `polish!` (`src/polish.jl:45`) | **refused by name** for an operator that declares `is_materializable` false |
| entry access + dense factor | `active_kkt` (`src/derivative.jl:29`), which `adjoint_derivative` (`:141`) and `forward_derivative` (`:192`) both reach | **refused by name**, at `active_kkt` so both derivative entry points inherit it |

`is_materializable(M)` is that declaration: true by default, `false` for an operator that
supplies only products. `dense_rung` and `formed_rung` decline such an operator, so
`linsys = :auto` falls through to `indirect_rung` instead of failing inside a factorization,
and `polish!` and `active_kkt` throw a named `ArgumentError` rather than a `MethodError`. It
is a statement by the operator's author, not a measured threshold, so it introduces no gate.

**What `scaling = 0` actually clears, which is less than the whole path.** `equilibrate!`
returns at `src/scaling.jl:159-161` before touching either matrix, so the column traversals are
skipped and an operator that overrides neither seam level gets past them. It does not get past
`is_symmetric` (`src/types.jl:308`) or `is_convex` (`src/types.jl:364`): both run before
`equilibrate!` and neither is conditioned on the scaling setting, so an operator with no
override for them fails there regardless. `scaling = 0` is the interim restriction on the
equilibration seam alone, and it is documented rather than silent.

### S5 — Split the cheap update paths out of `factorize!` (Build, small)
Take COSMO's distinction: "only ρ moved, do the cheap update" versus "data changed, rebuild".
`factorize!` currently conflates them and gets away with it because `ReducedGram`'s slot map
makes the rebuild cheap. A block or Kronecker factorization may update in `O(n)` when only ρ
moves, and there is no way to express that today.

**Also `update!` of `P`/`A` values**, which an earlier draft omitted. The sparse backends
already validate their slot maps against pattern changes; a structured or lazy backend needs a
defined response to changed data — refill, rebuild, or refuse.

**Built** as `refactor_rho!(ls, ws)`, whose default is `factorize!` — so rebuilding everything
stays a correct answer and no backend is obliged to override. `adapt_rho!` calls it;
`setup` and `update!` keep calling `refactor!`, which is what makes the override's
precondition (the `ρ`-independent parts are current) hold. Changed `P`/`A` values reach every
backend as a full rebuild, which is the "rebuild" response of the three.
`bench/results/rho_update.json` records what the one override saves: 1.14–1.44× for
`DiagonalLowRank`, and exactly 1.00× for the backends that take the default.

### S6 — Equilibration protocol for lazy operators (dropped)
Ruiz needs row and column ∞-norms of `P` and `A`. Materialized: walk entries. Lazy: cannot.
The option taken is the first below; the others are recorded because the choice between them is
the whole content of this item.

- **the operator supplies its norms** — trivial for `Diagonal`, per-block for a block operator,
  and it needs no addition to the protocol, because the seam it overrides is already there.
  `ext/PureOSQPGPUArraysCoreExt.jl` ships it: `column_norms!` and `cost_norms!` are replaced by
  whole-matrix reductions for a device matrix that must not be indexed, and the same two
  methods are what an operator overrides to answer per sweep. The per-column level —
  `structural_rows` and the four traversals — is the finer of the two seams and serves an
  operator that can enumerate a column.
- **deterministic basis probing** — `A·eⱼ` yields column `j`, so the ∞-norms can be read off
  products alone. The cost is not `n + m` products paid once. `equilibrate!` calls `cost_norms!`
  once to seed (`src/scaling.jl:164`) and again per sweep (`:175`), and `column_norms!` once per
  sweep (`:166`); `weighted_colmax_rowmax!` accumulates the row norms inside the column pass, so
  no `Aᵀ` probes are needed at all. That is `(2·sweeps + 1)·n` matvecs — **21n** at the default
  `scaling = 10`. Caching the probed columns instead costs `n(m + n)` words, which is exactly
  the dense backend's `W` and `Rinv` footprint under another name, so probing-with-a-cache
  spends the memory the structured tier exists to avoid.
- random probing — changes results, and is therefore worse than basis probing for no gain
- `scaling = 0` — measurable loss on badly scaled problems

**What probing agrees with, precisely.** It is exact for an operator whose `mul!` selects
stored entries, since `A·eⱼ` then copies column `j`. For an operator that applies a factored or
composed form it agrees only to that operator's own rounding, because the product recomputes
the entry rather than reading it. So probing is bit-exact in the case where it is unnecessary —
the entries exist and could be walked — and inexact in the case that motivated it.

**Which invariant is at risk, corrected.** An earlier draft claimed probing would break
"iteration counts identical to libosqp". It cannot: libosqp can only receive materialized
input, so a lazy operator has no libosqp baseline to differ from. What probing actually
threatens is **lazy-vs-materialized self-consistency** — that the same problem expressed two
ways solves identically. That invariant now has a standing test in `test/scaling_tests.jl` and
is listed under Validation.

### S7 — Backend introspection (Reuse + Build, trivial)
Clarabel's `linear_solver_info` is richer than `backend_name`. Useful for the docs and the
benchmark table.

## Invariants that must not break

- **iteration counts identical to libosqp 0.6.2** — the strongest correctness evidence, and it
  depends on equilibration staying bit-comparable, which is what the equilibration seam is
  designed to preserve: an override answers the same question about the same numbers, it does
  not substitute an estimate of them
- **`noalloc` + `typestable` on the hot path** — subtle: a block backend holding
  `Vector{Cholesky{…}}` is concretely typed and fine; one holding *heterogeneous* block types is
  not, and would lose the guarantee. For a caller-supplied operator the guarantees on
  `admm_step!`, `update_residuals!` and `solve_system!` are **inherited from that operator's
  `mul!`** rather than established by this package: the hot path runs the caller's code, and a
  broadcast or a type-interpolated error message in it loses both. `bench/lazy_operator.jl` is
  the operator written to hold them and `bench/strictmode_audit.jl`'s `:operator` row is where
  that is checked.
- **`--trim` compatibility** on every public path
- **the dense-regime numbers** — structure-first selection must not tax the dense path; S1 being
  behavior-preserving is what protects this

## Phasing

| phase | work | risk | delivers |
|---|---|---|---|
| 1 | S1 + S7 — the ladder, dense terminal | low; checked by asserting the backend chosen per suite class | no behavior change at all |
| 2 | S3 — first structured backend (Woodbury over a diagonal core) | medium | the new tier, on `AbstractMatrix` types |
| 3 | S5 — split the ρ and data update paths | low | cheap-update channel, with `DiagonalLowRank` as its first consumer |
| 4 | the six defect items, in the order of the table above | low; each reproduced by measurement before it is fixed | types the package already ships reach the backends it already has |
| 5 | the operator protocol — the `is_materializable` declaration and its named refusals | medium | lazy operators, at `scaling = 0` or with a seam override |
| 6 | this document and the log restated against the code now in force | none | S2, S4-as-written and S6 recorded as decisions rather than as deferrals |

**S4 moved after S3 deliberately.** Doing S3 first means the new backend tier is proved on
ordinary `AbstractMatrix` types — where equilibration, `is_convex`, `is_symmetric` and `polish!`
all still work — before the protocol work destabilizes any of them. The defect items sit between
them because each fires on a type the package already ships, so they are corrections to the base
the protocol lands on rather than new capability.

Phase 2 starts with **rank-`(k+1)` Woodbury over a diagonal core**, and the motivating problem
has to be stated precisely, because an earlier draft got it wrong.

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

**So the phase-2 deliverable is a declared operator type, validated against Portfolio
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
- **Representation independence**: one problem in two spellings — structure declared by type,
  and the same entries with no structure to declare — must equilibrate to bit-identical `D`,
  `E` and `c` and, at a fixed `ρ`, take the same iteration count to the same objective.
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


Measurements, reversals and review findings are in
`2026-08-30-structure-centric-log.md`. Nothing there changes a requirement here.