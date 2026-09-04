# Algorithm

This page explains the internal working of the solver.

## The idea in one paragraph

The problem involves minimizing an objective while staying within bounds. ADMM (*alternating direction method of multipliers*) solves this by maintaining two copies of the answer: one that minimizes the objective and one that satisfies the bounds. A penalty term is added to pull these two copies together. Each iteration consists of solving an unconstrained problem for `x`, clipping `z` into the box, and updating a multiplier `y`. When they agree, you have the solution.

The linear solve is the most expensive part. The matrix in the linear solve remains unchanged unless `ρ` changes. The solver's efficiency comes from factoring this matrix once and reusing it.

## The ADMM iteration

Each iteration solves one linear system and then performs a projection:

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

`σ` regularizes the `(1,1)` block, `α` is the over-relaxation parameter, and `ρ` is a vector with larger values on equality rows.

## The linear system

The reference implementation factors the `(n+m)×(n+m)` quasi-definite matrix with a sparse pivot-free LDLᵀ. An equivalent `n×n` symmetric positive definite system can be used:

```math
(P + \sigma I + A^\top \mathrm{diag}(\rho) A)\, \tilde x = \mathrm{rhs}_x + A^\top(\rho \odot \mathrm{rhs}_z),
\qquad \tilde z = A \tilde x
```

PureOSQP factors this with `cholesky!`. In dense regimes, this reduced form is faster.

```math
\mathrm{bandwidth}(R) = \max\bigl(\mathrm{bandwidth}(P),\; 2\,\mathrm{bandwidth}(A)\bigr)
```

The `choose_backend` function selects a solver based on the problem's structure:

| `P` | `A` | bandwidth of `R` | backend | solve |
|---|---|---|---|---|
| `Diagonal` | `Diagonal` | 0 | [`DiagonalReduced`](@ref PureOSQP.DiagonalReduced) | `n` divisions |
| `SymTridiagonal` or `Tridiagonal` | `Diagonal` | 1 | [`TridiagonalReduced`](@ref PureOSQP.TridiagonalReduced) | `ldlt`, `O(n)` |
| `Diagonal` | `Bidiagonal` | 1 | [`TridiagonalReduced`](@ref PureOSQP.TridiagonalReduced) | `ldlt`, `O(n)` |
| `SymTridiagonal` or `Tridiagonal` | `Bidiagonal` | 1 | [`TridiagonalReduced`](@ref PureOSQP.TridiagonalReduced) | `ldlt`, `O(n)` |
| banded | banded | ``2 \leq b \leq n/4`` | `BandedReduced` | banded `cholesky`, `O(nb²)` |

The banded backend is a package extension requiring `BandedMatrices.jl`.

A `Diagonal` `P` with a general `A` does not receive special treatment, and its reduced matrix is dense.

Fill-in is a factor. On random sparse `A`, the reduced matrix `R` is much sparser than `A` suggests, but its Cholesky factor is not.

| n | m | density(A) | density(R) | density of chol(R) | fill |
|---|---|---|---|---|---|
| 200 | 400 | 1% | 6.3% | 47.8% | 7.6× |
| 200 | 400 | 5% | 78.9% | 100% | 1.3× |
| 400 | 800 | 1% | 11.2% | 83.3% | 7.4× |
| 400 | 800 | 5% | 95.4% | 100% | 1.1× |

## Solving with the inverse

Once `R` is factored, it is inverted in place. This is faster than a Cholesky `ldiv!` because it uses a symmetric matrix-vector product (`symv`) instead of two triangular solves.

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

Modified Ruiz equilibration is applied to the system. It is stored as factors rather than applied to the matrices:

```math
\tilde P = c\,D P D, \qquad \tilde A = E A D, \qquad \tilde q = c\,D q, \qquad
\tilde l = E l, \qquad \tilde u = E u
```

Every per-iteration product runs `mul!` on the caller's original matrix with the factors applied around it, so a structured or lazy `A` keeps its fast product and nothing is copied.

## Convergence

The iteration stops when primal and dual residuals satisfy the given tolerances.

```math
r_{\rm prim} = \|Ax - z\|_\infty,
\qquad
r_{\rm dual} = \|Px + q + A^\top y\|_\infty
```

Both residuals are reported in **problem space**.

Tolerances are absolute plus relative:

```math
\epsilon_{\rm prim} = \epsilon_{\rm abs} + \epsilon_{\rm rel}\max\bigl(\|Ax\|_\infty,\|z\|_\infty\bigr),
\qquad
\epsilon_{\rm dual} = \epsilon_{\rm abs} + \epsilon_{\rm rel}\max\bigl(\|Px\|_\infty,\|q\|_\infty,\|A^\top y\|_\infty\bigr)
```

**The duality gap is a third test, and it can only delay convergence.** With
`check_dualgap` — on by default, following libosqp 1.x — a point must additionally satisfy
`|gap| < ε_gap` before it is called `SOLVED`. It is checked *after* the residuals pass, never
instead of them, because a point with a small gap and a large residual is not a solution.

Three more things decide what the caller sees:

- **Checking is periodic.** The test runs every `check_termination` iterations, 25 by default,
  so a reported iteration count is quantized to that. It is not a search for the first
  iteration that would have passed.
- **Failing the tolerances triggers the infeasibility certificates**, not just another
  iteration — a large primal residual is what prompts the primal-infeasibility test below.
- **A relaxed pass is reported as such.** When the loop ends without converging, the whole
  test is repeated at ten times the tolerances; passing that gives `SOLVED_INACCURATE`, which
  is a different answer from `SOLVED` and never presented as one. A residual above `INFTY`
  gives `NON_CONVEX`: a convex problem's residuals cannot diverge.

`scaled_termination` tests the equilibrated residuals instead of the unscaled ones. It is off
by default, since the natural question is about your problem rather than the solver's internal
one.
## Adaptive ρ

`ρ` is re-estimated based on the ratio of primal and dual residuals:

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

Infeasibility is detected using the differences in iterates `δx` and `δy`.

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
