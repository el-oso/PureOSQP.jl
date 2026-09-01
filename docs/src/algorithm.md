# Algorithm

This page is how the solver works inside. You do not need it to use the package — start at
[Examples](@ref) for that — but it is what the tuning knobs, the backend names and the
benchmark tables all refer back to.

## The idea in one paragraph

The problem is hard because two demands fight: minimize the objective, and stay inside the
bounds. Either alone is easy. Minimizing a quadratic with no constraints is one linear solve;
clipping a vector into a box is one `max` and one `min`. ADMM — *alternating direction method
of multipliers* — exploits that by keeping **two copies** of the answer, letting each satisfy
one demand, and adding a penalty that drags them together. Each iteration is: solve the
unconstrained problem for `x`, clip the other copy `z` into the box, then update a multiplier
`y` by however far apart they still are. When they agree, both demands hold at once and you
have the solution.

That gives the shape of everything below. The linear solve is the expensive half and gets most
of this page. `ρ` is the weight on the disagreement penalty — too small and the copies drift,
too large and progress stalls, which is why it is retuned as the solve proceeds. `σ` keeps the
linear system solvable when `P` alone is not. `α` over-relaxes the update, a standard trick
worth a modest constant factor.

The one property that matters for performance: **the matrix in that linear solve does not
change between iterations** unless `ρ` does. So it is factored once and reused hundreds of
times, and nearly all the engineering here is about making that factorization and its reuse as
cheap as the problem's structure allows.

## The ADMM iteration

Each iteration solves one linear system and then does a projection:

```math
\begin{aligned}
\begin{bmatrix} P + \sigma I & A^\top \\ A & -\mathrm{diag}(\rho)^{-1} \end{bmatrix}
\begin{bmatrix} \tilde x^{k+1} \\ \nu^{k+1} \end{bmatrix}
&= \begin{bmatrix} \sigma x^k - q \\ z^k - \rho^{-1} \odot y^k \end{bmatrix} \\
\tilde z^{k+1} &= z^k + \rho^{-1} \odot (\nu^{k+1} - y^k) \\
x^{k+1} &= \alpha \tilde x^{k+1} + (1-\alpha) x^k \\
z^{k+1} &= \Pi_{[l,u]}\!\left(\alpha \tilde z^{k+1} + (1-\alpha) z^k + \rho^{-1} \odot y^k\right) \\
y^{k+1} &= y^k + \rho \odot \left(\alpha \tilde z^{k+1} + (1-\alpha) z^k - z^{k+1}\right)
\end{aligned}
```

`σ` regularizes the `(1,1)` block, `α` is the over-relaxation parameter, and `ρ` is a
vector with a larger value on equality rows.

## The linear system

The reference implementation forms the `(n+m)×(n+m)` quasi-definite matrix above and
factors it with a sparse pivot-free LDLᵀ. Eliminating `ν` gives an equivalent system that
is `n×n` and symmetric positive definite:

```math
(P + \sigma I + A^\top \mathrm{diag}(\rho) A)\, \tilde x = \mathrm{rhs}_x + A^\top(\rho \odot \mathrm{rhs}_z),
\qquad \tilde z = A \tilde x
```

PureOSQP factors that with `cholesky!` — *shə-LES-kee*, after a French officer, and
[not with a hard `k`](@ref "The name Cholesky"). The reason upstream avoids it is fill-in: `AᵀA` is
dense even when `A` is sparse. With sparsity off the table that objection disappears, and
measurement shows the reduced form is faster in every dense regime — from 1.45× when
`m < n` up to 43× when `m ≫ n`.

`Ãᵀ diag(ρ) Ã` is what fills in, and it fills in by a fixed amount: diagonal scaling
preserves a bandwidth and squaring `A` doubles it, so

```math
\mathrm{bandwidth}(R) = \max\bigl(\mathrm{bandwidth}(P),\; 2\,\mathrm{bandwidth}(A)\bigr)
```

`choose_backend` dispatches on the pair of types, so a problem whose `R` stays narrow is
solved as such without a setting:

| `P` | `A` | bandwidth of `R` | backend | solve |
|---|---|---|---|---|
| `Diagonal` | `Diagonal` | 0 | [`DiagonalReduced`](@ref PureOSQP.DiagonalReduced) | `n` divisions |
| `SymTridiagonal` or `Tridiagonal` | `Diagonal` | 1 | [`TridiagonalReduced`](@ref PureOSQP.TridiagonalReduced) | `ldlt`, `O(n)` |
| `Diagonal` | `Bidiagonal` | 1 | [`TridiagonalReduced`](@ref PureOSQP.TridiagonalReduced) | `ldlt`, `O(n)` |
| `SymTridiagonal` or `Tridiagonal` | `Bidiagonal` | 1 | [`TridiagonalReduced`](@ref PureOSQP.TridiagonalReduced) | `ldlt`, `O(n)` |
| banded | banded | `2 ≤ b ≤ n/4` | `BandedReduced`, with BandedMatrices.jl loaded | banded `cholesky`, `O(n b²)` |

Against the dense path those are 1710×, 1270× and 699× on setup at `n = 2000`, and 1216×,
725× and 252× end to end, on the same iterates — see
[Structured backends](@ref "Structured backends"). A separable objective under box
constraints is the first row; a tridiagonal one — smoothing, trend filtering — the second;
differencing constraints, where a `Tridiagonal` `A` squares to bandwidth 2, the last.

The same band spelled `Tridiagonal` selects the same backend as `SymTridiagonal`. Both name
bandwidth 1, `setup` has already established that `P` is symmetric, and the backend reads
only the diagonal and one superdiagonal — so the choice of spelling is the user's and does
not change what solves the problem.

The banded backend is a package extension, so it exists only once BandedMatrices.jl is
loaded; without it those problems take the dense path, correctly but densely. It declines
two ways: below bandwidth 2 the LinearAlgebra backends above are cheaper, and above a quarter
of the matrix the dense path wins per iteration.

Two things the arithmetic will not do for you here, both of which is why the bands are
computed entry by entry rather than by forming the product. `D P D` on a `SymTridiagonal`
returns a `Tridiagonal`, which `cholesky` rejects as not Hermitian though it is symmetric to
`1e-17`; and a `Diagonal` `P` with a `Bidiagonal` `A` returns a dense `Array` despite having
bandwidth 1. An `ldlt` also reports neither indefiniteness nor a zero pivot the way a
Cholesky does — it returns a negative pivot for the first and throws for the second — so
both are tested explicitly.

A `Diagonal` `P` with a general `A` gets no treatment at all, and correctly: its reduced
matrix is dense whatever `P` looked like.

Fill-in is worth quantifying, because it also settles whether a sparse factorization would
be worth adding. On random sparse `A`, the reduced matrix `R` is much sparser than `A`
suggests it should be, but its Cholesky factor is not:

| n | m | density(A) | density(R) | density of chol(R) | fill |
|---|---|---|---|---|---|
| 200 | 400 | 1% | 6.3% | 47.8% | 7.6× |
| 200 | 400 | 5% | 78.9% | 100% | 1.3× |
| 400 | 800 | 1% | 11.2% | 83.3% | 7.4× |
| 400 | 800 | 5% | 95.4% | 100% | 1.1× |

A uniformly random sparsity pattern is an expander: it has no small vertex separators, so
elimination fills aggressively no matter how the matrix is ordered. By 5% density `R` is
effectively dense. Structured problems — grids, chains, banded couplings — behave far
better, but they are also the large problems where the dense path already wins.

### Solving with the inverse

Once `R` is factored it is inverted in place, and each iteration solves by one `symv`
against the stored inverse rather than by a Cholesky `ldiv!`. Both do `2n²` flops. The
difference is that a triangular solve computes its entries in sequence, one depending on
the last, while a symmetric matrix-vector product has no such chain:

| kernel | solve, n = 200 | relative error |
|---|---|---|
| `potrs` (two triangular solves) | 11.43 µs | 5.2e-16 |
| `symv` against the inverse | 1.61 µs | 6.8e-16 |
| two `trmv` against `L⁻¹` | 3.82 µs | 5.7e-16 |

At these sizes the factor is a few hundred kilobytes and stays in cache, so the sequential
dependency — not memory bandwidth — is what bounds the triangular solve. Inverting costs
roughly two Cholesky factorizations (`potri` on top of `potrf`), paid once per `ρ` update
and repaid over the hundreds of iterations between updates.

Forming an explicit inverse is normally poor practice, and it is safe here for a specific
reason: the reduced matrix carries the `σI` regularization, so its conditioning is bounded
by construction rather than inherited from the data. That is what the error column above
shows — the inverse is no less accurate than the triangular solve.

The counter-pressure is conditioning: forming `AᵀρA` squares `cond(A)`. Measured relative
error of the inner solve against an extended-precision reference, `n = 60`, `m = 200`,
`P = 0`, on `A` built with geometrically spread singular values:

| cond(A) | reduced, unscaled | reduced, equilibrated | full KKT |
|---|---|---|---|
| 1e4 | 8.2e-10 | 4.3e-10 | 1.5e-13 |
| 1e6 | 1.1e-05 | 4.5e-09 | 8.8e-12 |
| 1e8 | 9.5e-02 | 8.0e-09 | 3.1e-10 |
| 1e10 | fails | 4.8e-09 | 3.2e-08 |
| 1e14 | fails | 1.3e-08 | 2.0e-04 |
| 1e16 | fails | 1.2e-08 | 3.4e-02 |

Equilibration is what makes the reduced form usable: without it the Cholesky loses all
accuracy by `cond(A) = 1e8` and fails outright past that. With it — the default — the
reduced solve holds around `1e-8` across the whole range, and past `cond(A) = 1e10` it is
*more* accurate than the full KKT factorization, because equilibration bounds what the
reduced matrix inherits while the quasi-definite matrix keeps the raw conditioning.

That table is built from a family of `A` whose ill-conditioning is a spread of singular
values, which is close to what equilibration is designed to fix. The harder case is
ill-conditioning no diagonal scaling can remove — `A` with unit-norm columns and nearly
parallel rows:

| row spread | cond(A) | reduced, equilibrated | Cholesky succeeded | full KKT |
|---|---|---|---|---|
| 1e-2 | 2.7e2 | 1.0e-12 | yes | 4.6e-14 |
| 1e-4 | 3.4e4 | 5.4e-09 | yes | 7.0e-11 |
| 1e-6 | 3.4e6 | 2.5e-08 | yes | 4.3e-10 |
| 1e-8 | 3.4e8 | 2.7e-08 | yes | 3.4e-10 |
| 1e-10 | 3.4e10 | 2.7e-08 | yes | 2.4e-10 |

Here the Cholesky is around a hundred times less accurate than the full KKT factorization,
but it still succeeds and still plateaus near `1e-8`; it degrades gradually rather than
silently collapsing. Across every case measured, with equilibration enabled, the Cholesky
was never observed to fail.

Both tables are reproduced by `bench/kkt_backend.jl`.

### Choosing a backend

`linsys = :auto` (the default) descends the selection ladder and takes the first rung that
serves the given `P` and `A`; the reduced Cholesky is its terminal rung, reached by any pair
that can be materialized and nothing cheaper fits. That Cholesky falls back to a
`bunchkaufman!` factorization of the full quasi-definite system if it reports that the matrix
is not positive definite. On the measurements above that fallback never
triggers with equilibration on, so treat it as a safety net rather than as the mechanism
that handles ill-conditioning.

`linsys = :kkt` forces the full quasi-definite factorization for every solve. It is
slower — that is the whole point of the reduced form — but it is the more accurate
factorization at moderate conditioning, and it is the closest match to what the reference
implementation does, which makes it useful when a result is in question. The entire test
corpus runs through both backends.
## Equilibration

Modified Ruiz equilibration on `[P Aᵀ; A 0]`, ten sweeps, each followed by a cost
normalization step. It is stored as factors rather than applied to the matrices:

```math
\tilde P = c\,D P D, \qquad \tilde A = E A D, \qquad \tilde q = c\,D q, \qquad
\tilde l = E l, \qquad \tilde u = E u
```

Every per-iteration product runs `mul!` on the caller's original matrix with the factors
applied around it, so a structured or lazy `A` keeps its fast product and nothing is
copied. Which backend materializes anything depends on the rung: the dense terminal holds an
`m×n` scaled copy of `A` and the `n×n` reduced matrix, rebuilt whenever `ρ` changes, while the
structured backends store only what their structure needs — `n` reciprocals, two bands,
`O(nb)`, or a `k×n` correction — and the matrix-free backend stores none of it.

## Adaptive ρ

`ρ` is re-estimated from the ratio of normalized primal and dual residuals,

```math
\rho_{\text{new}} = \rho \sqrt{
  \frac{r_{\text{prim}} / \max(\|z\|_\infty, \|Ax\|_\infty)}
       {r_{\text{dual}} / \max(\|q\|_\infty, \|A^\top y\|_\infty, \|Px\|_\infty)}}
```

and adopted only when it moves by more than a factor of `adaptive_rho_tolerance`, since
adopting it forces a refactorization.

The reference implementation triggers this on wall-clock time by default — once 0.4 × the
setup time has elapsed. PureOSQP uses a fixed iteration interval (default 50) instead, so
that iteration counts do not depend on how fast the machine is.

## Infeasibility

Both certificates come from the iterate differences `δx`, `δy`.

The problem is primal infeasible when, after projecting `δy` onto the polar of the
recession cone of `[l,u]`,

```math
u^\top \max(\delta y, 0) + l^\top \min(\delta y, 0) < \varepsilon \|\delta y\|,
\qquad \|A^\top \delta y\|_\infty < \varepsilon \|\delta y\|_\infty
```

It is dual infeasible when

```math
q^\top \delta x < 0, \qquad \|P \delta x\|_\infty < \varepsilon\|\delta x\|_\infty,
\qquad A \delta x \in \text{recession cone of } [l,u]
```

## Polishing

The active set is guessed from the ADMM iterates: row `i` is lower-active when
`z_i - l_i < -y_i` or `l_i == u_i`, and upper-active when `u_i - z_i < y_i`. The
resulting equality-constrained QP

```math
\begin{bmatrix} P + \delta I & A_{\text{red}}^\top \\ A_{\text{red}} & -\delta I \end{bmatrix}
\begin{bmatrix} x \\ y_{\text{red}} \end{bmatrix}
= \begin{bmatrix} -q \\ b_{\text{red}} \end{bmatrix}
```

is factored with `bunchkaufman!` and corrected by three steps of iterative refinement
against the unregularized operator, which removes the error introduced by `δ`. The polished
point replaces the ADMM answer only if both residuals improve.

## The name Cholesky

The name appears throughout this manual, and the pronunciation most often heard in English is
the one variant with no support from any of the languages involved. So, briefly.

**André-Louis Cholesky** (born 15 October 1875 in Montguyon; died 31 August 1918 of wounds
received in northern France) was a French army officer and geodesist who ended as head of the
Topographical Service of Tunisia. He did not publish the method himself. It appeared
posthumously in 1924, when a fellow officer, Commandant Benoît, wrote it up in the
*Bulletin géodésique* as *"Note sur une méthode de résolution des équations normales…
(Procédé du Commandant Cholesky)"*.

**Say it `/ʃəˈlɛski/` — *shə-LES-kee*.** The first sound is the *sh* of *shoe*.

The reason is that he was French, and French ⟨ch⟩ is /ʃ/. There is a second defensible reading
from the family's origins: his paternal line descended from the **Cholewski** family, which
left Poland during the Great Emigration, and Polish ⟨ch⟩ is /x/ — the fricative in *Bach*, in
Greek χ, in Russian х, in Spanish *j*. That gives *kho-LES-kee*, and it has been argued for in
the field's own literature: a 1990 NA Digest exchange set out three candidates and concluded
that "all three current pronunciations seem acceptable" pending evidence of the name's origin,
noting that a Polish origin would make *Kholesky* correct.

**What has no basis is a hard English /k/ — "koh-LES-kee", the *k* of *kiosk*.** It is neither
the French /ʃ/ nor the Polish /x/. The two are distinct sounds: /x/ is a fricative, air still
flowing; /k/ is a plosive, stopped and released. The /k/ reading most likely comes from the
English habit of pronouncing ⟨ch⟩ as /k/ in words taken from Greek — *chorus*, *chaos*,
*character* — and this name is not Greek.

References: the pronunciation `/ʃəˈlɛski/` is given by
[Wikipedia's article on the decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition);
the Cholewski descent by
[its biography of Cholesky](https://en.wikipedia.org/wiki/Andr%C3%A9-Louis_Cholesky); the
dates, rank and the Benoît publication by the
[MacTutor biography](https://mathshistory.st-andrews.ac.uk/Biographies/Cholesky/); and the
three-way discussion by [NA Digest, Volume 90 Issue 11 (18 March 1990)](https://www.netlib.org/na-digest-html/90/v90n11.html).
