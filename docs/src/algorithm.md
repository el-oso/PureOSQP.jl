# Algorithm

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

PureOSQP factors that with `cholesky!`. The reason upstream avoids it is fill-in: `AᵀA` is
dense even when `A` is sparse. With sparsity off the table that objection disappears, and
measurement shows the reduced form is faster in every dense regime — from 1.45× when
`m < n` up to 43× when `m ≫ n`.

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

`linsys = :auto` (the default) uses the reduced Cholesky and falls back to a
`bunchkaufman!` factorization of the full quasi-definite system if the Cholesky reports
that the matrix is not positive definite. On the measurements above that fallback never
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
copied. Only the direct solve materializes dense buffers: an `m×n` scaled copy of `A` and
the `n×n` reduced matrix, rebuilt whenever `ρ` changes.

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
