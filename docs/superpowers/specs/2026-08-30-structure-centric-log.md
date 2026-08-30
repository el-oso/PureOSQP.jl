# Structure-centric work — log

Findings, measurements and reversals. The requirements and their status are in
`2026-08-30-structure-centric-design.md`, which does not depend on this file.

## The two gate limits, measured

**Measured 2026-08-30. Both thresholds are too conservative, and the cost is large.**
Reproduce with `bench/gate_crossover_fill.jl` and `bench/gate_crossover_band.jl`; samples in
`bench/results/`.

| gate | current | measured | cost of the current setting |
|---|---|---|---|
| `nnz(L) < DENSE_FACTOR_FILL·n²` | `0.05` | parity near **0.23** | at `n = 1000`, `:auto` is **1.5–2× slower** than the sparse backend it declines — fill 0.086: 60 ms vs 31 ms; 0.121: 88 vs 57; 0.166: 85 vs 59 |
| banded declines at `b >= n ÷ 2` | `0.5·n` | **no crossover found** | banded still wins **1.6×** at `b/n = 0.8`, well past the cutoff |

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

The banded gate was not measured, and its comment — "at half the matrix or wider, the dense
path wins outright" — is false. **Changed** to two conditions:

- storage, `2b + 1 <= m + n`, the point where the band stops being smaller than the dense
  backend's `W` and `Rinv` together;
- and `5b <= 4n`, because storage alone accepts too much. Measured against the reduced matrix
  itself the band stops being a compression at `2b + 1 = n`, and beyond that the banded
  factorization's worse constants decide: the sweep has it ahead from `b = 2` through
  `b = 0.8n` and **behind near `b = n - 1`, at 0.80× for `n = 200`**. The second condition is
  where the sweep stops, so the accepted range is exactly the measured one.

The bandwidth is also clamped to `n - 1` first: a wider one describes the same full matrix and

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
