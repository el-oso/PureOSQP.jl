# Structure-centric work — log

Findings, measurements and reversals. The requirements and their status are in
`2026-08-30-structure-centric-design.md`, which does not depend on this file.

## The two gate limits, measured

**Measured 2026-08-30. Both thresholds are too conservative, and the cost is large.**
Reproduce with `bench/gate_crossover_fill.jl` and `bench/gate_crossover_band.jl`; samples in
`bench/results/`.

| gate | setting measured against | measured | cost of that setting |
|---|---|---|---|
| `nnz(L) < DENSE_FACTOR_FILL·n²` | `0.05` | parity near **0.23** | at `n = 1000`, `:auto` is **1.5–2× slower** than the sparse backend it declines — fill 0.086: 60 ms vs 31 ms; 0.121: 88 vs 57; 0.166: 85 vs 59 |
| banded deferral at `b >= n ÷ 2` | `0.5·n`, inherited | **no crossover found** | banded still wins **1.6×** at `b/n = 0.8`, well past the cutoff |

**The two are not in the same position, and an earlier draft of this section was wrong to say
both were caution.**

**Both gates now stand on the same rule — accept only where the structured path wins the
factorization *and* the per-iteration solve — and both are measured.** `bench/gate_band_beyond.jl`
puts the band's per-iteration crossing near `b = 0.3n`, and the limit sits below it at `n/4`.
`bench/gate_fill_periteration.jl` puts the sparse factor's crossing between a fill of 0.01 and
0.04, and the limit sits below it at 0.05 — close enough to the boundary that loosening it is
not supported. Neither choice depends on how many iterations a problem runs.

One trap governs every one of these numbers: **pin BLAS to one thread when measuring them.**
`symv` parallelizes and a banded triangular solve does not, so an eight-thread `symv` is some
eightfold faster and moves the band's crossing below `b = 0.05n`. Three sweeps disagreed with
each other before that was found, and a limit was set twice from the contaminated readings.

The fill gate **is** defended by measurement, documented at its definition: a bandwidth sweep
at `n = 2000` put the sparse triangular solves 12.6× ahead at a fill of 0.0025 and losing at
0.086, crossing near 0.06, and the limit was deliberately set *below* that crossing so the
accepted region wins on the factorization *and* the per-iteration solve — which keeps the
choice from depending on how a run divides its time between the two. The new figure measures
something different, end to end, and relaxing to it would trade that robustness property for
average-case speed. **Left unchanged; the trade is a decision, not a correction.**

The banded gate was not measured, and the comment it carried — "at half the matrix or wider,
the dense path wins outright" — is false. The limit in force is `4b > n`: the band is accepted
for `2 <= b <= n/4`, below the per-iteration crossing measured at one thread. One condition
covers what an earlier two-condition form spelled out separately, because `b <= n/4` already
puts the band's `(2b+1)n` words below the dense backend's `mn + n²` and already implies
`b < n - 1`, so neither storage nor a bandwidth wider than the matrix needs a test of its own.

## The operator library


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


## The low-rank tier: equilibration, not the solve, was the cost

The Woodbury apply was fast from the first measurement — 275× per iteration at `n = 2000,
k = 1`. Setup was not: 40 ms at `n = 1000`, against 1.6 ms once fixed. Both regressions were
in the same place, `equilibrate!`'s column walk, and neither was in the linear algebra.

- **The generic `structural_rows` is `O(mn)`.** Without a method, equilibration visits every
  row of every column — `10⁶` entries for a matrix holding `kn + m₀`. Adding the method took
  setup from 40,216 to 1,589 µs, a factor of 25.

- **`Iterators.flatten` over a heterogeneous tuple costs a dynamic dispatch.** The first
  method returned `Iterators.flatten((1:k, view(ord, …)))`, whose iteration state is a union
  across the two halves. That is a dynamic call on the equilibration path: `--trim` rejected it
  outright (three verifier errors, all rooted at `column_norms!`), and replacing it with a
  concrete `CoupledRows` iterator took setup from 1,589 to 154 µs — another factor of 10, on
  top of correctness. The trim gate found a performance bug that no timing had flagged.

## Portfolio in the low-rank type: the honest number is 1.16×, not 900×

Portfolio is the low-rank shape exactly — a budget row and five factor rows above one bound
row per asset, over a diagonal `P` — so it is the validation the synthetic sweep cannot be.
Re-expressed as `Diagonal` + `RowCoupled` it selects `lowrank`, and the objective agrees with
the `SparseMatrixCSC` form to 1.6e-12.

Two regimes, and they disagree. Both forms reach `SOLVED` in both:

| regime | sparse (`ldl_kkt`) | `RowCoupled` (`lowrank`) | total | per iteration |
|---|---|---|---|---|
| fixed ρ, 1e-6 | 14.61 ms, 2750 iter | 12.37 ms, 2750 iter | 1.18× | 1.17× |
| adaptive ρ, 1e-9 | 5.07 ms, 900 iter | 8.71 ms, 1900 iter | **0.58×** | 1.18× |

Setup is 1.87× either way. At a fixed `ρ` the two forms take the same iterates, so 1.18× is
the backends' ratio and nothing else. Under adaptive `ρ` they do not: the two solves differ in
the last digits — the equilibration is bit-identical, so this is the solve alone — and `ρ`
adaptation amplifies that into different update points, one update against two, with final
estimates of 0.383 against 0.015. The low-rank form is faster every iteration and takes 2.1×
as many.

**Storage goes the wrong way here.** The Woodbury form holds `n + kn + k(k+1)/2` = 3556 words
against the KKT factor's 2322: six coupling rows over 505 columns is a denser object than the
sparse factor of a matrix whose only dense rows are those same six. The tier saves storage
against a *dense* backend, not against every sparse one, and Portfolio is a case where it
loses.

Two more things follow. The sweep's 20–920× is against the **dense** path; against a **sparse
KKT** path on a real problem the margin is 1.18×, and the win is in setup rather than in a
large per-iteration gain at this size. And a backend comparison run under adaptive `ρ`
measures the `ρ` trajectory as much as the backend, so it needs the fixed-`ρ` number beside
it — with both runs converged, since two runs that stop at `max_iter` agree on the iteration
count for no reason at all.

## The low-rank gate, measured

The flop count says the correction wins until `k = n/2`: two `gemv`s against a `k×n` block
where the rung below does one `symv` against `n×n`. Measured, the crossing is far earlier and
depends on the caller's thread count, because `symv` is multithreaded and both `gemv`s here
are narrow — `k/n ≈ 0.225` at eight BLAS threads, `≈ 0.445` at one
(`bench/results/gate_crossover_lowrank.json`).

`src/` and `ext/` pin no threads, so a limit derived at one thread promises something the
deployed solver does not deliver. The limit is `10k <= n`, below the threaded crossing: at
`k = n/10` the solve is 1.78–2.27× ahead and setup 3.0–6.8× at every size measured.

This is the same mistake the band gate made twice, in a third form: reasoning about a
single-threaded kernel while the competitor it is measured against is not one.

## What the ρ-only channel is worth

`V = E ⊙ C ⊙ D` is the only part of `DiagonalLowRank`'s factorization that `ρ` does not enter,
so a ρ move rebuilds the core, `Y = V C⁻¹` and the `k×k` capacitance but keeps `V`. Measured
against a full rebuild (`bench/results/rho_update.json`):

| case | 500 | 1000 | 2000 |
|---|---|---|---|
| `lowrank k=1` | 1.16× | 1.16× | 1.16× |
| `lowrank k=4` | 1.41× | 1.44× | 1.43× |
| `lowrank k=n/10` | 1.23× | 1.14× | 1.15× |
| `tridiagonal`, `cholesky` | 1.00× | 1.00× | 1.00× |

Modest, and it peaks in the middle rather than at small `k`: `V`'s rebuild is one of `2 + k`
passes over `k×n`, so its share shrinks as `k` grows, while at `k = 1` there is too little
work either way for the saving to show. The rows at 1.00× are the backends that take the
default, and they are in the table because a channel that is only reported where it wins is
not a measurement.

## The operator route: a wrapper, not a relaxed bound

`Workspace`'s `MP <: AbstractMatrix` / `MA <: AbstractMatrix` bound was never what excluded a
`LinearMap`. It participates in exactly one method signature across `src/` and `ext/` —
`mul_At!` for a `Tridiagonal` `A` — and `RowCoupled` had already shipped as a products-only
operator declared `<: AbstractMatrix` that clears equilibration, `--trim` and the StrictMode
audit. What excluded a `LinearMap` was that it is a separate hierarchy, which a wrapper fixes
and a relaxed bound does not: both routes need the same override set, and only the relaxation
also needs signature changes and a dependency.

So `ProductOperator` wraps, and the bound stands. Three things the wrapper had to answer that
products do not give:

- **Symmetry and positive-definiteness** are declared by the author. Verifying either from
  products costs more than the solve. For a `LinearMap` the extension reads them from the
  map's own `issymmetric`/`isposdef`, which are properties it already carries.
- **The reduced diagonal** is `P[j,j]` and the ρ-weighted column norms of `A`, neither a
  product. A `ProductOperator` runs unpreconditioned instead: conjugate gradients converges
  without a preconditioner, so this costs iterations, not correctness.
- **Equilibration** cannot be answered at all, and `getindex` throws a message naming
  `scaling = 0` and both seam levels. That closes the one gap the protocol had left open —
  a products-only operator at the default scaling previously died with a `CanonicalIndexError`
  from inside `weighted_colmax`.

**The audit needed the matrix-free exemption extended, not the code fixed.** The first
StrictMode run reported `admm_step!` and `solve_system!` allocating at 50 sites on this path.
Every site is inside Krylov's `cg!` — `time_ns`, `allocate_if`, `ktimer` — statically visible
and never taken, which is why `:indirect` already used the measured tier rather than the
static check. Measured, the path allocates 0 bytes.

## The plan's own reorderings

- **S5 moved from before S3 to after it.** The phasing table put the update-path split ahead of
  the Woodbury backend while the decision list said S5 lands with S3. Those two cannot both
  hold, and S3 shipped alone. The order that survives is S3 first: the
  split is interface work, and `DiagonalLowRank.factorize!` — which rebuilds `V`, `Y` and the
  capacitance for a ρ move that changes only the core and the `k×k` system — is the concrete
  shape it has to serve.

## Corrections this work made to itself

- **The band limit moved twice before it was right**: `n/2` as inherited, then `n/7`, then
  `n/4`. The middle value came from probes taken while BLAS had eight threads. `symv`
  parallelizes and a banded triangular solve does not, so the band appeared to lose from
  `b = 0.05n` upward when at one thread it was ahead to `b ≈ 0.3n`. Three sweeps disagreed with
  each other before the cause was found. Pin the thread count in anything that times a kernel.

- **No faster banded solve is available.** Five alternatives were built and measured against
  `ldiv!`'s 24.1 µs at `n = 1000, b = 100`: a block-inverse BLAS chain (29.6 µs), a `trmv`
  variant (28.5 µs), hand-fused SIMD kernels (32.7 µs), fatter `k = 2b` blocks (38–45 µs), and
  an `f32` factor (19.1 µs at a relative error of 3e-7, which iterative refinement would spend
  again). None win. The substitution already runs at 16.7 GFLOP/s against `dgemv`'s 18.4 here,
  and streaming the factor twice floors the solve near 16 µs — under 1.5× of headroom. SPIKE
  and cyclic reduction buy parallelism this package does not use, a semiseparable inverse lands
  in the same traffic class, and a banded approximate inverse needs a wider band than the exact
  solve.

- **Portfolio is rank six, not rank one.** Its `A` carries five factor rows of roughly 250
  nonzeros beside the budget row, so a rank-1 correction leaves a 76%-dense remainder. Nor is
  the budget row why the class routes to `ldl_kkt`: one factor row trips the densest-row gate
  by itself.

- **`setup` densifies `P` before any backend is chosen**, including under `linsys = :indirect`,
  because `is_convex`'s generic method runs unconditionally.

- **A capability query cannot replace the gates.** Two of them factor the real equilibrated
  matrix and hand that factorization to `setup`; a query answering only "does this fit" would
  either discard it or cache it and be the gate again under another name.

- **S2 was deferred until a structural term got a second consumer, and the condition was never
  met.** S3 added a fifth private term rather than reusing one: `coupling_rank`
  (`src/rowcoupled.jl:87`) has a single structural consumer, `src/lowrank.jl:77`. Every one of
  the terms a shared vocabulary would have unified is still read only inside the extension or
  the type that defines it. The description that is genuinely shared is `structural_rows`, which
  already existed, is per column rather than per matrix, and now serves the banded types, the
  sparse ones through their traversal overrides, `RowCoupled` and the matrix-free
  preconditioner.

- **The `AbstractMatrix` bound never had to be relaxed.** An `AbstractMatrix` subtype that
  defines no `getindex` at all is holdable and solvable as it stands: `bench/lazy_operator.jl`'s
  `LazyPSD` supplies `size`, `mul!` in both directions and the predicates, stores no `n×n`
  object, and reaches `IndirectCG` through the ladder. What was missing was not the ability to
  hold such a type but a way for it to say it holds no entries, which is `is_materializable`.
  `RowCoupled` is not that proof — it does define `getindex`, and uses it only outside the hot
  path.

- **The per-sweep equilibration seam already existed when S6 was written as unbuilt work.**
  `column_norms!` and `cost_norms!` are override points, and `ext/PureOSQPGPUArraysCoreExt.jl`
  replaces both with whole-matrix reductions for a matrix that must not be indexed at all —
  exercised by `test/gpu_tests.jl` under `JLArrays.allowscalar(false)`, which fails the run on
  any scalar access. An operator supplying its own norms uses that same seam, so the option S6
  listed as "needs a protocol addition" needs none.

- **S6's probe arithmetic was wrong, and the cache that would fix it is the dense backend.**
  Basis probing does not cost `n + m` products paid once. `equilibrate!` calls `cost_norms!`
  once to seed (`src/scaling.jl:164`) and again at the end of every sweep (`:175`), and
  `column_norms!` once per sweep (`:166`), so the probe count is `(2·sweeps + 1)·n` — **21n** at
  the default `scaling = 10`. No `Aᵀ` probes enter it: `weighted_colmax_rowmax!` accumulates the
  row norms inside the column pass. Caching the columns to avoid re-probing costs `n(m + n)`
  words, which is `ReducedCholesky`'s `W` plus `Rinv` exactly.

- **Probing is exact only where it is unnecessary.** `A·eⱼ` reproduces column `j` bit for bit
  when the operator's `mul!` selects stored entries — the case in which the entries could simply
  have been walked. An operator applying a factored or composed form recomputes the entry, so
  the agreement is to that operator's rounding, not to the entries. The claim that probing is
  "bit-identical to walking the entries" holds in the wrong half of the domain.

- **Unwrapping `Symmetric{<:Any, <:SparseMatrixCSC}` in `setup` would be unsound.** `validate`
  tests the wrapper, whose `issymmetric` is `true` however the parent is filled, so unwrapping
  `Symmetric(sparse(triu(M)))` substitutes the stored triangle for the full matrix. The
  admission predicate was narrowed instead — see the section on that wrapper below.

- **A direct backend's `factorize!` cannot absorb the convexity test.** It factors
  `c D P D + σI + Ãᵀ diag(ρ) Ã`, which a large enough `ρ` makes positive definite over an
  indefinite `P`; the counterexample is an assertion in `test/linsys_tests.jl`. `is_convex` stays
  unconditional and before the backend choice.

## The banded column walk was quadratic

Reproduced before it was fixed, since the claim arrived from a review rather than from a
measurement here. `bench/banded_structural_rows.jl` times `setup` for a `BandedMatrix` `P` of
bandwidth 8 with a `Diagonal` `A`, once at the default `scaling = 10` and once at `scaling = 0`,
so the equilibration share separates by subtraction. Samples in
`bench/results/banded_structural_rows.json`.

The defect is real and its growth is exactly quadratic. Median `setup`, one BLAS thread:

| n | no method | with method | equilibration, no method | with method |
|---|---|---|---|---|
| 250 | 2.62 ms | 0.20 ms | 2.54 ms | 0.126 ms |
| 500 | 10.54 ms | 0.45 ms | 10.34 ms | 0.253 ms |
| 1000 | 42.31 ms | 1.05 ms | 41.77 ms | 0.509 ms |
| 2000 | 169.63 ms | 2.74 ms | 167.86 ms | 1.022 ms |

Growth exponent per doubling of `n`: equilibration goes from **2.01** to **1.01** — quadratic
to linear, which is the shape the fix predicts, since a bandwidth-derived range makes the walk
`O(nb)` for a fixed `b`. End to end `setup` is 12.9× faster at `n = 250` and **61.8×** at
`n = 2000`, and the ratio keeps growing with `n` because the two curves have different orders.

Equilibration is 97% of `setup` here without the method and 37% with it, so this was not a
share of setup — it *was* setup.

**The thread count does not enter.** Every figure above repeats to within 0.5% at eight BLAS
threads. The walk is scalar indexing, and at these sizes nothing in the banded path is
BLAS-bound, so this is the one gate measurement in this campaign where the thread trap that
caught the band limit twice has no purchase. Both thread counts are recorded anyway, because
that is only knowable after measuring.

`Symmetric{<:Any,<:BandedMatrix}` needs `banded_bandwidth`, the wider of the parent's two
bandwidths, rather than the parent's own asymmetric pair: the wrapper mirrors one triangle
across the diagonal, so a range built from the stored triangle's bandwidths truncates the
mirrored half.

**A review aside, refuted.** `P[band(k)] .= x` was reported to throw
`MethodError: no method matching size(::BandedMatrix)`. It does not, on BandedMatrices as
resolved here; the benchmark and the test both build their bands that way.

## The two spellings of one tridiagonal band

`TridiagonalReduced.factorize!` reads `P` through `P[j, j]` and `P[j, j+1]` only, and
`validate` has established that `P` is symmetric before any of it runs, so a `Tridiagonal` `P`
was always servable — it simply was not enumerated. `choose_backend`'s two `SymTridiagonal`
methods now take `Tridiagonal` as well, and the banded extension's `A::NarrowBand` method
gives up the `Tridiagonal` `P` position it no longer needs, which is what keeps the two sets
disjoint. No new backend code.

**The review's suggestion to drop `Tridiagonal` from `WideBand` was refused, and the reason is
in the code.** `WideBand` names both argument positions, and the `A` position is load-bearing:
`(SymTridiagonal P, Tridiagonal A)` has a reduced bandwidth of 2 and is the pair that
`bench/strictmode_audit.jl` runs through the banded backend. Removing the type would send it to
the dense terminal. Only the second method's `P` position was narrowed.

Measured against `ReducedCholesky` on the same problem, `Tridiagonal` `P` with `Diagonal` `A`
(`bench/results/tridiagonal_rung.json`). Setup is `:auto` against `linsys = :dense`; the
per-iteration figures build both backends over one workspace, so they read the same
equilibrated data.

| n | setup ×, 1 thread | setup ×, 8 threads | solve ×, 1 thread | solve ×, 8 threads |
|---|---|---|---|---|
| 250 | 4.60 | 4.01 | 3.95 | 4.10 |
| 500 | 5.20 | 3.59 | 6.74 | 2.18 |
| 1000 | 4.79 | 4.09 | 12.96 | 3.02 |
| 2000 | 5.61 | 4.37 | 59.08 | 8.78 |

The tridiagonal backend wins at every size and at both thread counts, so the widening
introduces no gate. The thread count matters exactly where the log predicts: the dense
backend's per-iteration cost is a threaded `symv` and a tridiagonal `ldiv!` is not threaded,
which cuts the per-iteration margin from 59× to 8.8× at `n = 2000`. It does not reverse it,
and both counts had to be run to know that.

No trim entry point took a `Tridiagonal` `P` before this, so the widening's effect on `--trim`
could not have been read off the existing gate: it produces a `Workspace` type nothing was
checking. `solve_tridiagonal_unsym` is that signature, and it passes.

**The setup margin is capped by something else, and it is worth naming.** At one thread the
tridiagonal setup is 61.4 ms at `n = 2000` while the same band spelled `SymTridiagonal` sets up
in 0.31 ms. Nearly all of the difference is `is_convex`: 64.8 ms for the `Tridiagonal` against
14.3 µs for the `SymTridiagonal`, the generic method densifying `P` because only
`SymTridiagonal` has an override. That is `is-convex-coverage`'s subject, not this one's; it is
recorded here because it is what the 4–5× setup ratio above is measuring against, and closing
it moves this row rather than that one.

## A `Symmetric` wrapper was admitted by one predicate and rejected by every consumer

Reproduced before anything was changed, since the claim arrived from a review. `n = m = 100`,
`P = Symmetric(sparse(Diagonal(...)))`, `A` one dense row over an identity block — 199 stored
entries against the density gate's 1000, so that rung declines, and a densest row of 100
against the fill gate's 500, so the KKT rung proceeds:

```
MethodError: no method matching kkt_gram(::Type{Float64},
    ::Symmetric{Float64, SparseMatrixCSC{Float64, Int64}}, ::SparseMatrixCSC{Float64, Int64},
    ::Int64, ::Int64)
```

Both sparse rungs admitted `P` on `issparse`, which unwraps through `parent` and so answers
`true` for the wrapper, while `kkt_gram`, `refill_kkt!`, `reduced_gram` and `refill!` are all
written against `SparseMatrixCSC`'s stored columns. The predicate now names that concrete
type, which is what the consumers require, and the wrapper descends to `formed_rung` instead —
a backend that reaches `P` only through `scaled_col!` and `add_scaled_col!`, so the generic
entrywise traversals serve it correctly. The same problem then solves through the wrapper,
through a `Symmetric` over a dense parent, and through a `Symmetric` over a stored triangle,
all three bit-identically.

**Unwrapping the parent in `setup` would be unsound, and the reason is in `validate`.** The
review proposed unwrapping `Symmetric{<:Any, <:SparseMatrixCSC}` on the grounds that symmetry
is established beforehand. It is not: `validate` calls `issymmetric` on the *wrapper*, which
is `true` by construction whatever the parent holds. `Symmetric(sparse(triu(M)))` is a
documented input form, and unwrapping it would silently substitute the stored triangle for the
full matrix — a wrong answer where there is currently a correct one. The wrapper stays, and
its cost is stated in `setup`'s docstring instead: it does not reach the sparse-factorization
backends, and it equilibrates entrywise rather than by stored column.

## What the duplicate density scan costs

`densest_row(A)` runs once in `sparse_kkt_backend` and again in `cholmod_backend`, so a sparse
`setup` that descends past the first sparse rung scans `A`'s row indices twice. Measured
against the setup it gates (`bench/gate_scan_cost.jl`, samples in
`bench/results/gate_scan_cost.json`), on the OSQP suite and two larger sparse problems:

| case | nnz(A) | one scan | setup | two scans as a share |
|---|---|---|---|---|
| Random QP | 3 782 | 1.22 µs | 0.29 ms | 0.84% |
| Portfolio | 2 294 | 1.06 µs | 0.30 ms | 0.70% |
| Control | 6 540 | 1.95 µs | 1.45 ms | 0.27% |
| Sparse 800 | 12 894 | 3.68 µs | 19.6 ms | 0.038% |
| Sparse 1600 | 50 925 | 12.8 µs | 121.8 ms | 0.021% |

One BLAS thread; at eight the shares move by less than a factor of two in either direction,
because the scan is scalar and setup's factorizations are not. The scan grows with `nnz(A)`
and setup grows faster, so the share falls as problems get larger — 0.84% is the worst case
here and it is the smallest problem. **Memoizing it would buy under a percent at the top of
the range and nothing at all where setup is actually expensive**, so the duplication stays.

This measures `densest_row` alone, not the whole fill gate, which also runs `reduced_nnz` over
the reduced matrix's pattern. No sample here covers the gate as a whole, which is why the design
document quotes no figure for it.

## The convexity test, and what else `setup` scans

Two types the package ships a backend for reached `is_convex`'s densifying method:
`Tridiagonal`, which `tridiagonal-rung` had just made selectable, and
`Symmetric{<:Any,<:BandedMatrix}`, which `banded_bandwidth` and `structural_rows` already
served everywhere else. Both now answer through the band. Samples in
`bench/results/is_convex_coverage.json` (`bench/is_convex_coverage.jl`); the generic figures
are the generic method itself, reached through `invoke`, not a re-derivation of it.

`is_convex` medians, structured against generic:

| n | `Tridiagonal`, 1 thread | 8 threads | `Symmetric{BandedMatrix}`, 1 thread | 8 threads |
|---|---|---|---|---|
| 500 | 4.6 µs / 1.30 ms | 4.7 µs / 0.66 ms | 36.3 µs / 1.31 ms | 39.2 µs / 0.78 ms |
| 1000 | 8.7 µs / 10.1 ms | 9.0 µs / 3.38 ms | 75.7 µs / 8.73 ms | 78.1 µs / 3.93 ms |
| 2000 | 16.8 µs / 55.3 ms | 16.9 µs / 19.6 ms | 145 µs / 59.0 ms | 157 µs / 22.7 ms |

The structured tests are `O(n)` and `O(n b²)`; the generic one is a threaded `potrf` over a
densification, so it is the only column the thread count moves. It moves it by under 3×
against ratios of 20× to 3300×, so the widening is not thread-sensitive.

**`issymmetric` was the largest single term in a banded `setup`, and now is not.**
`validate`'s symmetry test had no structure-aware method, so a `BandedMatrix` `P` was compared
at all `n²` positions while everything else in that setup had been made `O(n b)`. At
`n = 2000`, bandwidth 8, one thread: 1.314 ms of scan against a 1.464 ms `setup` — larger than
equilibration (1.02 ms) and an order of magnitude above `is_convex` (0.134 ms). Comparing only
the band takes it to 0.030 ms, a factor of 44.

| n | generic scan | band comparison | `setup` with the generic predicate | with the band comparison |
|---|---|---|---|---|
| 500 | 0.087 ms | 0.008 ms | 0.451 ms | 0.372 ms |
| 1000 | 0.334 ms | 0.015 ms | 1.054 ms | 0.735 ms |
| 2000 | 1.314 ms | 0.030 ms | 2.748 ms | 1.464 ms |

The two `setup` columns come from one run: the measured `setup` and that same figure with the
band comparison exchanged for the entrywise scan, which is what the results file's
`setup_generic_predicate_s` holds. A separate sweep taken before the override existed measured
0.450, 1.055 and 2.751 ms directly, so the substitution is worth within 0.3% of the real thing.

Both thread counts agree to within 1%: the scan is scalar indexing, as the banded column walk
was.

**The override is `PureOSQP.is_symmetric`, not a method on `issymmetric`.** BandedMatrices
defines no `issymmetric` method, so adding one from PureOSQP's extension would be type piracy
that silently changes `issymmetric` for every other user of BandedMatrices in the session.
`validate` and `update!` call a package-owned predicate instead, defaulting to `issymmetric`,
which is the same override-point shape `is_convex`, `structural_rows` and `reduced_diagonal!`
already have — and the shape S4 needs anyway, since a lazy operator has no `issymmetric` at
all.

**The claim that a direct backend's `factorize!` makes `is_convex` redundant is false, and the
counterexample is in the suite.** `factorize!` tests `c D P D + σI + Ãᵀ diag(ρ) Ã`, not
`P + σI`. With `A = I`, `scaling = 0` and `ρ = 50`, an indefinite `SymTridiagonal` `P` whose
diagonal sits 5 below its positive-definite form gives `is_convex(T, P, σ) == false` and
`TridiagonalReduced.factorize!(ls, ws) == true` on the same workspace. So `is_convex` stays
where it is, unconditional and before the backend choice; nothing about its placement changed.

**The GPU path keeps the host densification deliberately.** `ext/PureOSQPGPUArraysCoreExt.jl`
overrides `choose_backend`, `column_norms!`, `cost_norms!` and `reduced_diagonal!` but not
`is_convex`, so `Matrix{T}(P)` copies `P` back to the host and factors it there. A device
matrix declares no algebraic form, so there is nothing cheaper to dispatch on, and the measured
share does not argue for inventing one: on a `JLArray` setup it is 0.027 ms of 59.2 ms at
`n = 100` and 0.133 ms of 281 ms at `n = 200` — under 0.05% either way, because the matrix-free
setup's own whole-matrix reductions cost far more. A comment beside the other overrides records
that.

## The matrix-free preconditioner summed every row, at every rho adaptation

`reduced_diagonal!` ran its inner sum over `eachindex(rho, E)` — every one of the `m` rows,
for every column — while the four column traversals beside it had followed `structural_rows`
since they were written. `IndirectCG.factorize!` is its only in-tree caller and
`refactor_rho!` defaults to `factorize!`, so a `Diagonal`, `Bidiagonal` or `RowCoupled` `A`
paid an `O(mn)` walk at setup and again at every rho adaptation.

Following `structural_rows(A, j)` is exact rather than approximate: an omitted row contributes
`rho[i] * (E[i] * 0 * D[j])^2`, which is zero, and the running sum starts from
`c·D[j]²·P[j,j] + σ` rather than from the first row. The kernel's output is bit-identical, and
both benchmark states agree on every iteration count below, so this changes what is visited
and nothing else.

Samples in `bench/results/reduced_diagonal_structural.json` (`bench/reduced_diagonal_structural.jl`).
The kernel sweep carries both forms in a single run — the every-row form is a local function
asserted equal to the shipped one before anything is timed — and each row is labelled by the
row set the shipped function actually walks, read off it with a counting matrix wrapper, so
running the file against both states of `src/scaling.jl` leaves both in one document.

`reduced_diagonal!` medians, one BLAS thread:

| structure | n | m | every row | structural rows | |
|---|---|---|---|---|---|
| `Diagonal` | 500 | 500 | 155.3 µs | 1.15 µs | 135× |
| `Diagonal` | 1000 | 1000 | 643.6 µs | 2.19 µs | 294× |
| `Diagonal` | 2000 | 2000 | 2531.0 µs | 4.23 µs | **599×** |
| `Bidiagonal` | 2000 | 2000 | 2527.5 µs | 4.75 µs | 532× |
| `RowCoupled` | 2000 | 2000 | 2583.7 µs | 10.18 µs | 254× |
| `RowCoupled` | 2000 | 4000 | 5143.3 µs | 12.50 µs | 411× |

The every-row column is flat across the three structures at a given `m·n` — 2.53, 2.53 and
2.58 ms — which is the shape of a walk that does not depend on what the matrix holds.

End to end, `linsys = :indirect` at the default rho adaptation, one BLAS thread:

| structure | n | m | iterations | rebuilds | every row | structural rows | |
|---|---|---|---|---|---|---|---|
| `Diagonal` | 500 | 500 | 50 | 2 | 0.315 ms | 0.156 ms | 2.02× |
| `Diagonal` | 1000 | 1000 | 75 | 2 | 1.628 ms | 0.378 ms | 4.31× |
| `Diagonal` | 2000 | 2000 | 50 | 2 | 3.057 ms | 0.559 ms | **5.47×** |
| `Bidiagonal` | 2000 | 2000 | 125 | 3 | 10.921 ms | 3.263 ms | 3.35× |
| `RowCoupled` | 1000 | 2000 | 450 | 10 | 27.752 ms | 21.322 ms | 1.30× |
| `RowCoupled` | 2000 | 4000 | 575 | 12 | 82.519 ms | 56.186 ms | 1.47× |

`rebuilds` counts the setup rebuild plus one per rho adaptation, which at the default
`adaptive_rho_interval = 50` is what turns a setup cost into a repeated one. The margin is
largest where the ADMM iteration is cheapest — a `Diagonal` `A` at `n = 2000` spends most of
its solve in the preconditioner rebuild, a `RowCoupled` one at 575 iterations does not.

**The thread count does not enter.** Every figure repeats to within 1.5% at eight BLAS
threads; the walk is scalar indexing and the matrix-free iteration at these sizes is not
BLAS-bound. Both counts are recorded anyway.

**`SparseMatrixCSC` is deliberately unaffected.** The sparse extension overrides the four
whole traversals rather than `structural_rows`, so a sparse `A` still gets `axes(A, 1)` here
and its timings do not move. A `structural_rows` method for it would have exactly one
consumer.

**Two spec premises were wrong on contact.** `m = 2n` was asked for on all three structures,
but `Diagonal` and `Bidiagonal` are square, so only `RowCoupled` carries both row shapes. And
a structured `A` does not reach a bit-identical *iterate* under `linsys = :indirect`, only a
bit-identical preconditioner and an identical iteration count: `mul!` against a `Bidiagonal`
or a `RowCoupled` sums a row in a different order than the dense `gemv`, which moves the last
bit of `x` by around 1 ulp. Measured on the equilibration factors, the preconditioner and each
matvec separately — `D`, `E`, `c` and `prec` agree exactly for all three structures, and the
divergence is `mul_A!` for `Bidiagonal` (1.1e-16) and `mul_At!` for `RowCoupled` (4.4e-16).
That predates this work and is a property of those types' `mul!`, not of the row set.

## Representation independence, asserted

The campaign's governing invariant — the same problem expressed two ways equilibrates and
solves alike — now has a standing test in `test/scaling_tests.jl`. `Diagonal` `P` with a
`RowCoupled` `A` against the same entries as `SparseMatrixCSC`, `n = 60`, `k = 4`, at the
default `scaling = 10`.

The two forms reach different backends — `lowrank` against `sparse_formed` — and still agree
exactly on the equilibration: `D`, `E` and `c` are bit-identical, no tolerance. The solve
agrees to the last few bits rather than exactly, since the two backends solve the same reduced
system in a different order: at a fixed rho both take 150 iterations, the objectives differ by
1.2e-15 relative and `x` by 1.8e-15 absolute.

`c` survives bit-identically despite being the one accumulated quantity in the sweep, because
`cost_norms!` sums over columns in the same order for both forms and everything below it is a
maximum, which does not care what order it sees the entries in.

**The fixed rho is load-bearing, not a convenience.** Under adaptive rho the last-digit
difference between the two backends lands on different update points, and the two forms take
different iteration counts for a reason that is not equilibration — the effect measured on
Portfolio above, 900 iterations against 1900. An adaptive-rho form of this assertion would
fail while the invariant it names still holds.

## The operator protocol is a trait, not a relaxed bound

`Workspace`'s `MP <: AbstractMatrix` / `MA <: AbstractMatrix` bounds are unchanged and no
operator library was added. What was missing was not the ability to hold an operator — an
`AbstractMatrix` subtype that defines no `getindex` was always holdable — but a way for one
to *say* it holds no entries, so the paths that read entries could decline it instead of
failing inside a factorization. `is_materializable` is that statement.

The method body is the literal `true`, so a call against a type with no override folds away.
Measured on inference rather than assumed: `dense_rung(Matrix, Matrix, …)` still infers
`Tuple{ReducedCholesky{Float64, Matrix{Float64}}, Bool}` and `select_backend` on the same
pair infers the same concrete tuple, so adding the trait to rung 6 did not turn the ladder
into a union.

**Two rungs consult it, not one.** `dense_rung` forms the reduced matrix with a product and
`formed_rung` accumulates it over stored entries; both read entries, so both decline. The
guard lands in `formed_rung`'s **sparse** method rather than in the generic one in `src/`:
the generic already returns `nothing` for everything, so a guard there would be dead, while
the sparse method takes `P` untyped and would otherwise accept an operator `P` beside a
`SparseMatrixCSC` `A`. `sparse_kkt_backend` and `cholmod_backend` already required
`P isa SparseMatrixCSC`, so they needed nothing.

**The untyped `dense_rung` fallback was reachable after all — from a test.**
`test/selection_tests.jl` proved the terminal's decline with a bare `struct Opaque end`,
which is not an `AbstractMatrix` and so could never reach `setup`. Deleting the fallback
broke that assertion. The test now declares `Opaque <: AbstractMatrix{Float64}` with
`is_materializable` false, so what it checks is a decline a caller can actually provoke.

## The hot-path guarantees are the caller's to keep

`bench/strictmode_audit.jl` gained an `:operator` row: a `P` that applies
`Diagonal(d) + α v vᵀ` through a closure, stores no matrix, and declares
`is_materializable` false (`bench/lazy_operator.jl`). It reaches `IndirectCG` through the
ladder rather than through `linsys = :indirect`.

The first run of that row failed three guarantees, and both causes were in code this package
does not own:

- **A broadcast in the operator's own `mul!` costs `noalloc`.** `@. y = d * x + s * v` made
  `mul!` on `ReducedOperator` report six allocation sites. AllocCheck cannot rule out
  `Base.Broadcast`'s aliasing branch — the same reason `src/elementwise.jl` writes explicit
  loops — so an operator written the obvious way loses the guarantee for its caller. A loop
  over `eachindex(y)` restores it. `eachindex(y, d, x, v)` would not: its `DimensionMismatch`
  message is built through code that costs the same proof.

- **Interpolating a type into an error message costs `typestable`.** `require_entries`'s
  refusal first read `"...holds $(typeof(P)) and $(typeof(A))..."`. For a materializable pair
  the condition folds to `true` and that branch is dead, which is why `require_host` has done
  the same thing for a long time without consequence. For an operator that declines, the
  condition folds to `false` and the branch is *live* code, and `show(::IO, ::Type)` is a
  runtime dispatch — JET traced 23 findings from that one interpolation. The message names no
  type now.

Both were found by running the audit, not by reading the code. With them fixed the row is
green at the full tier: `admm_step!`, `update_residuals!` and `solve_system!` carry
`typestable` and `noalloc`, and `ReducedOperator`'s `mul!` carries both statically.

The conclusion is a condition on the invariant rather than a defect: for a caller-supplied
operator the hot-path guarantees are inherited from that operator's `mul!`. This package can
show that its own machinery does not take them away; it cannot establish them.

## What the matrix-free route costs an operator

`bench/results/operator_protocol.json`, 8 rows (2 sizes × 2 thread counts). `P` is
`Diagonal(d) + α v vᵀ` and `A` is dense; given as the operator it reaches `indirect`, given
as the `n×n` matrix it names it reaches `cholesky`. `scaling = 0`, BLAS pinned per sweep,
warmed once, Chairmarks `@be` medians.

| n | threads | lazy setup | dense setup | × | lazy step | dense step | × |
|---|---|---|---|---|---|---|---|
| 500 | 1 | 0.167 ms | 8.992 ms | 53.99 | 302.9 µs | 44.9 µs | 0.15 |
| 1000 | 1 | 0.677 ms | 56.080 ms | 82.88 | 1082.3 µs | 258.7 µs | 0.24 |
| 500 | 8 | 0.179 ms | 5.543 ms | 30.90 | 309.9 µs | 42.2 µs | 0.14 |
| 1000 | 8 | 0.682 ms | 25.091 ms | 36.79 | 567.5 µs | 94.6 µs | 0.17 |

The trade is the expected one and both halves of it are large: setup is 31–83× cheaper
because nothing is factored, and each iteration is 4–7× dearer because the inner solve is a
conjugate-gradient run against products. Which side wins depends on the iteration count, and
no threshold is introduced from these numbers — `is_materializable` is a declaration by the
operator's author, so there is no gate here to place.

The thread count moves the dense column by about 2× (a threaded `potrf` and `symv`) and the
lazy column by up to 1.9× at `n = 1000`, where the conjugate-gradient inner solve is
dominated by `gemv` against the dense `A`. It narrows the setup margin and widens the
per-iteration one; it reverses neither.
