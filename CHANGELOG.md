# Changelog

Notable changes, and the measurements that drove them. The reference documentation states
what is true now; this file is where the history lives.

## Unreleased

### Added

- **`SparseLDL` and `LDLKKT`**, the reduced and full-KKT backends factored by
  LDLFactorizations.jl, a pure-Julia `LDLᵀ`. A weak dependency: loading it changes which
  engine factors, nothing a caller can observe but speed, and without it the CHOLMOD
  backends serve. Its numeric factorization is 2.3–3.1× faster than CHOLMOD's on these
  problems, allocates nothing, produces identical fill, and hands back `L` and `D` as Julia
  arrays so nothing has to be extracted from a foreign factor on every refactorization.
  Only the factorization is delegated; the substitutions and the diagonal scaling stay here,
  being as fast or faster than the library's own.
- **`check_factor`**, which establishes once per factorization that every index the
  substitutions will use is in range, so those loops can run unchecked.
- **`ReducedGram`**, a stored map from each contribution to its slot in the reduced matrix,
  so a refactorization refills values in one allocation-free pass instead of rebuilding
  through four sparse products.
- **`SparseKKT`**, a sparse `LDLᵀ` factorization of the full `(n+m)×(n+m)` quasi-definite
  system, selected when a dense row in `A` would densify the reduced matrix.
- **`SparseCholmod`** and **`SparseFormedInverse`**, the reduced backends for a sparse `A`:
  one forms the reduced matrix from stored entries and factors it with CHOLMOD, the other
  forms it sparsely and factors it densely.
- **`IndirectCG`**, a matrix-free preconditioned-CG backend over Krylov.jl.
- **GPU arrays**, through the matrix-free backend only, with JLArrays as the correctness
  gate.
- **Solution derivatives**, by implicit differentiation of the KKT conditions.
- **A MathOptInterface wrapper**, as a package extension.
- **`linsys = :dense`**, to overrule the representation gates.
- **`bench/osqp_suite.jl`**, the OSQP benchmark suite's seven problem classes.

### Measurements worth keeping

**The substitutions were paying a bounds check per nonzero.** `x[rows[p]] -= vals[p] * xj`
indexes by a row index read out of the factor, which no compiler can prove is in range;
QDLDL's identical C loop checks nothing. On factors holding two to three nonzeros per column
that check is a large share of the per-entry work — removing it measured 1.18× (Lasso),
1.23× (Huber) and 1.32× (Portfolio) on the substitution pair, bit-identical results. This
was the single largest item in closing the suite: Portfolio went 0.88× to 1.00×. The
property is now a checked precondition rather than an assumption, which is what makes the
unchecked loops defensible.

**Deriving the slot map from the formed matrix is unsound, and the guard caught it.** It
looked like free work — `cholmod_backend` forms the reduced matrix through sparse products
to measure its fill, so the pattern already exists when the map is wanted. But a sparse
product drops an entry that cancels to exactly zero, and whether one cancels depends on `ρ`:
Lasso's `[I 0 −I; I 0 I]` blocks give `−w_k + w_k` at one entry. A map built from that
pattern would have no slot for it, and the moment `adapt_rho!` moved `ρ` the term would be
silently dropped. The duplication that remains is the price of a pattern that does not move.

**The Portfolio class exposed a limit of the reduced form.** Eliminating to the reduced
system squares `A`, so a single dense row makes the result dense however sparse the rest of
it is. On the OSQP suite's Portfolio class — whose budget constraint `1ᵀx = 1` touches every
column — `A` is 0.9% dense and the reduced matrix is 99% dense. PureOSQP ran at 0.18× of
libosqp there, and no backend choice helped, because the choice happens after the reduced
matrix is formed. `SparseKKT` took it to 0.56×, a 2.9× improvement, by never forming that
matrix.

That result also overturned an earlier verdict recorded below: a sparse `LDLᵀ` of the full
KKT had been measured and rejected. The rejection was correct for the family it was measured
on, whose `A` has no dense row, and it did not generalize. The suite is what showed that,
which is the argument for benchmarking against structured problems rather than synthetic
ones.

**A nearly-dense matrix stored sparsely costs an order of magnitude in equilibration.** The
suite's Eq QP class has `nnz(P)/n² = 0.99`. Handed that as a `SparseMatrixCSC`, `scale!`
takes 2.98 ms; handed the same matrix as a `Matrix`, 0.304 ms, and the whole solve goes from
4.40 ms to 1.10 ms — from 0.76× of libosqp to 3.1×. The guidance to densify above about 10%
density is not a rounding-error concern.

### How the sparsest case was closed


The 1% case, `n = 200`, `m = 400`, used to be the one cell where OSQP was ahead, by 1.7×.
Splitting it showed setup was not the story — what remained was the iteration loop, and
within it the triangular solve. The reduced system is `n×n` and **dense whatever `A` looked
like**, so `ldiv!` cost ~12 µs per iteration, which at 475 iterations was most of the time.

The obvious reading was that this needed a sparse factorization backend, mirroring
upstream's own design. Measurement says otherwise, twice over.

First, a sparse factorization does not pay here. A sparse LDLᵀ of the full KKT matrix does
beat the dense solve — 6.31 µs against 11.37 µs at 1% — but loses the factorization, 141 µs
against 110 µs, and by 5% density it is 6.6× behind on the factorization and behind on the
solve as well. The fill-in table under [Algorithm](@ref "The linear system") explains why:
the factor is 48–83% dense even when the matrix is not.

Second, and decisively, the dense solve was simply running well below what the hardware
allows. The kernel was two triangular solves, whose entries must be computed in sequence.
Replacing it with a `symv` against the inverted factor — identical flop count, no
dependency chain — took the per-iteration solve from 11.4 µs to 1.6 µs, and the cell from
8.08 ms to 3.54 ms against OSQP's 4.80 ms.

The lesson generalises past this cell: the same change sped up every other case on this
page, because the dense solve was on all of their hot paths too.

The *first* finding does not generalise, and
[The OSQP benchmark suite](@ref "The OSQP benchmark suite") is where it breaks. Sparse
full-KKT was rejected on a family whose `A` has no dense row, where the reduced form is a
good trade. On the Portfolio class one row of `A` is dense, the reduced matrix comes out 99%
dense from a 0.9% dense `A`, and the full-KKT form is the only one that avoids squaring that
row. Read the rejection as scoped to the corpus it was measured on.

### What an earlier PureBLAS measurement got wrong

A previous version of this page reported PureBLAS at 0.49–0.96×, with transposed `gemv`
**10× slower** than OpenBLAS. That was real, and it was a genuine bug rather than a tuning
problem: PureBLAS's BLAS-2 SIMD path was unreachable *through `activate()`* specifically,
so every measurement taken via the libblastrampoline reroute — which is how this benchmark
runs — fell back to a scalar path. The same kernels called directly were fine. It is fixed
upstream; transposed `gemv` went from 41.6 µs to 2.51 µs.

The mistaken diagnosis is worth recording too. The obvious suspicion was per-machine
tuning: PureBLAS autotunes, and two of its seven knobs (`gemvt_percol_window`, `gemvt_pf`)
govern exactly this path. That hypothesis was wrong. Running `PureBLAS.tune!(unlocked=true)`
here — three independent calibration runs on an idle machine — pins **nothing**:
`sytrf_cmult` disagreed across runs (`[1, 2, 2]`) and every other knob tied, so the report
is "the in-code defaults are adequate here". Tuning was never what separated the two.

