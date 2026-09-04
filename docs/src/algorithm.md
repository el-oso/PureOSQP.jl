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

The two systems for a small sparse pair, with each entry colored by where it comes from. The
reduced matrix is a fraction of the size, and `AᵀρA` fills it in wherever two rows of `A`
share a column.

::: details Code that draws the figure

```@example alg_kkt_figure
using CairoMakie, LinearAlgebra, SparseArrays, Random
Random.seed!(1)

n, m = 10, 16
A = Matrix(sprand(Bool, m, n, 0.15) .| sparse(1:m, mod1.(1:m, n), true, m, n))  # no empty row
Pn = [abs(i - j) <= 1 for i in 1:n, j in 1:n]                                  # tridiagonal P

# Entry codes: 0 empty, 1 from P, 2 from A, 3 both (or the −diag(ρ)⁻¹ block).
kkt = zeros(Int, n + m, n + m)
kkt[1:n, 1:n] .= Pn
kkt[1:n, (n + 1):end] .= 2 .* A'
kkt[(n + 1):end, 1:n] .= 2 .* A
for i in 1:m
    kkt[n + i, n + i] = 3
end
red = Pn .+ 2 .* ((A' * A) .> 0)

wong = cgrad([:white, "#0072B2", "#E69F00", "#CC79A7"]; categorical = true)
pattern!(ax, codes) = heatmap!(ax, 1:size(codes, 2), 1:size(codes, 1), permutedims(codes);
                               colormap = wong, colorrange = (-0.5, 3.5))

fig = Figure(size = (820, 420))
ax1 = Axis(fig[1, 1]; aspect = DataAspect(), yreversed = true,
           title = "full KKT, (n+m)×(n+m) = $(n + m)×$(n + m), LDLᵀ")
pattern!(ax1, kkt)
lines!(ax1, [n + 0.5, n + 0.5], [0.5, n + m + 0.5]; color = :black, linewidth = 1)
lines!(ax1, [0.5, n + m + 0.5], [n + 0.5, n + 0.5]; color = :black, linewidth = 1)
text!(ax1, n / 2 + 0.5, -0.6; text = "P + σI", align = (:center, :bottom), color = "#0072B2", fontsize = 14)
text!(ax1, n + m / 2 + 0.5, -0.6; text = "Aᵀ", align = (:center, :bottom), color = "#E69F00", fontsize = 14)
text!(ax1, -0.5, n / 2 + 0.5; text = "P + σI", align = (:right, :center), color = "#0072B2", fontsize = 14)
text!(ax1, -0.5, n + m / 2 + 0.5; text = "A", align = (:right, :center), color = "#E69F00", fontsize = 14)
text!(ax1, n + m + 1, n + m / 2 + 0.5; text = "−diag(ρ)⁻¹", align = (:left, :center), color = "#CC79A7", fontsize = 14)
ax2 = Axis(fig[1, 2]; aspect = DataAspect(), yreversed = true,
           title = "reduced, n×n = $n×$n, Cholesky")
pattern!(ax2, red)
text!(ax2, n / 2 + 0.5, -0.6; text = "P + σI + Aᵀdiag(ρ)A", align = (:center, :bottom), fontsize = 14)
Legend(fig[2, 1:2],
    [PolyElement(color = c) for c in ("#0072B2", "#E69F00", "#CC79A7")],
    ["from P", "from A", "both, or −diag(ρ)⁻¹"]; orientation = :horizontal, framevisible = false)
foreach(ax -> (hidedecorations!(ax); hidespines!(ax)), (ax1, ax2))
colsize!(fig.layout, 1, Relative(0.62))
nothing # hide
```

:::

```@example alg_kkt_figure
fig # hide
```

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

The first row, drawn. This is a fresh random pattern at the same `n`, `m` and density, so
its percentages differ a little from the table's; the shape of the result does not. `R`
keeps most of `A`'s sparsity, and its Cholesky factor keeps almost none of it.

::: details Code that draws the figure

```@example alg_fill_figure
using CairoMakie, LinearAlgebra, SparseArrays, Random
Random.seed!(1)

n, m = 200, 400
A = sprand(m, n, 0.01)
R = sparse(1.0I, n, n) + A' * A
L = cholesky(Matrix(Symmetric(R))).L
density(M) = round(100 * count(!iszero, M) / length(M); digits = 1)

fig = Figure(size = (900, 380))
for (i, (name, M)) in enumerate(("A" => Matrix(A), "R = σI + AᵀρA" => Matrix(R), "chol(R)" => Matrix(L)))
    axf = Axis(fig[1, i]; aspect = DataAspect(), yreversed = true,
               title = "$name   $(size(M, 1))×$(size(M, 2)),  $(density(M))% nonzero")
    heatmap!(axf, 1:size(M, 2), 1:size(M, 1), permutedims(M .!= 0);
             colormap = [:white, "#0072B2"], colorrange = (0, 1))
    hidedecorations!(axf); hidespines!(axf)
end
nothing # hide
```

:::

```@example alg_fill_figure
fig # hide
```

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

The whole selection, top to bottom. A pair whose types name a backend outright takes it
without descending; every other pair starts at rung 1 and stops at the first rung that
accepts it. The two sparse factorization rungs decide by factoring and keep the factor they
produce, so a `yes` there costs nothing extra.

::: details Code that draws the figure

```@example alg_ladder_figure
using CairoMakie

steps = [
    ("Diagonal P and Diagonal A", ":diagonal — n divisions"),
    ("SymTridiagonal/Tridiagonal P with Diagonal A,\nor Diagonal/tridiagonal P with Bidiagonal A", ":tridiagonal — ldlt, O(n)"),
    ("banded P and A, 2 ≤ bandwidth(R) ≤ n/4\n(BandedMatrices loaded)", ":banded — banded Cholesky, O(nb²)"),
    ("sparse A with more than 10% nonzeros", ":cholesky — dense reduced matrix"),
    ("sparse A: factor the full KKT matrix;\nfactor under 5% of n² nonzeros?", ":sparse_kkt"),
    ("sparse A: factor R sparsely;\nfactor under 5% of n² nonzeros?", ":cholmod"),
    ("KroneckerOperator A, P = μI,\none ρ for all rows, scaling = 0", ":kronecker — eigenbases of the factors"),
    ("BlockDiagonal P and A, same partition", ":block — one solve per block"),
    ("Diagonal P and RowCoupled A, 10k ≤ n", ":lowrank — Woodbury, O(nk)"),
    ("sparse A: form R from stored entries", ":sparse_formed — dense inverse"),
    ("P and A can be materialized", ":cholesky — dense reduced matrix"),
    ("neither can be (a LinearMap)", ":indirect — conjugate gradients"),
]

fig = Figure(size = (860, 760))
ax = Axis(fig[1, 1])
hidedecorations!(ax); hidespines!(ax)
box!(x, y, w, h, txt; color, textcolor = :black) = begin
    poly!(ax, Rect2f(x, y, w, h); color, strokecolor = :black, strokewidth = 1)
    text!(ax, x + w / 2, y + h / 2; text = txt, align = (:center, :center), fontsize = 12, color = textcolor)
end
down!(x, y0, y1) = begin
    lines!(ax, [x, x], [y0, y1]; color = :gray40)
    scatter!(ax, [x], [y1]; marker = :dtriangle, color = :gray40, markersize = 10)
end
H = length(steps)
for (i, (question, answer)) in enumerate(steps)
    y = H - i
    box!(0, y + 0.1, 5.2, 0.8, question; color = (:gray, 0.12))
    box!(6.4, y + 0.1, 4.6, 0.8, answer; color = ("#0072B2", 0.15), textcolor = "#0072B2")
    lines!(ax, [5.2, 6.4], [y + 0.5, y + 0.5]; color = :gray40)
    scatter!(ax, [6.4], [y + 0.5]; marker = :rtriangle, color = :gray40, markersize = 10)
    text!(ax, 5.8, y + 0.55; text = "yes", align = (:center, :bottom), fontsize = 10, color = :gray40)
    i < H && down!(2.6, y + 0.1, y - 0.1)
end
text!(ax, 2.6, H + 0.25; text = "P, A   (linsys = :auto)", align = (:center, :bottom), fontsize = 13)
down!(2.6, H + 0.2, H - 0.1)
lines!(ax, [-0.25, -0.25], [H - 3 + 0.1, H - 0.1]; color = :gray50)
text!(ax, -0.4, H - 1.5; text = "by type", align = (:center, :bottom), fontsize = 11, color = :gray50, rotation = pi / 2)
lines!(ax, [-0.25, -0.25], [0.1, H - 3 - 0.1]; color = :gray50)
text!(ax, -0.4, (H - 3) / 2; text = "the ladder, rungs 1–8", align = (:center, :bottom), fontsize = 11, color = :gray50, rotation = pi / 2)
text!(ax, 5.0, -0.5; text = "a :cholesky that finds R not positive definite is rebuilt as :bunchkaufman (full KKT)",
      align = (:center, :top), fontsize = 11, color = :gray30)
limits!(ax, -1.2, 11.2, -1.2, H + 1)
nothing # hide
```

:::

```@example alg_ladder_figure
fig # hide
```

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

::: details Code that draws the figure

```@example alg_equil_figure
using CairoMakie

fig = Figure(size = (760, 330))
ax = Axis(fig[1, 1]; aspect = DataAspect())
hidedecorations!(ax); hidespines!(ax)
# A diagonal factor: an outlined square with its diagonal drawn.
diagbox!(x, y, s, label, color) = begin
    lines!(ax, Rect2f(x, y, s, s); color = :black, linewidth = 1)
    lines!(ax, [x, x + s], [y + s, y]; color, linewidth = 3)
    text!(ax, x + s / 2, y - 0.25; text = label, align = (:center, :top), fontsize = 14, color)
end
# The caller's matrix, held by reference.
refbox!(x, y, w, h, label) = begin
    poly!(ax, Rect2f(x, y, w, h); color = (:gray, 0.25), strokecolor = :black, strokewidth = 1)
    text!(ax, x + w / 2, y + h / 2; text = label, align = (:center, :center), fontsize = 15)
    text!(ax, x + w / 2, y - 0.3; text = "the caller's array, by reference",
          align = (:center, :top), fontsize = 12, color = :gray30)
end
n, m = 3.0, 4.5
y0, y1 = 6.5, 0.5
text!(ax, 0, y0 + n / 2; text = L"\tilde{P} = c\,\cdot", align = (:right, :center), fontsize = 16)
diagbox!(0.5, y0, n, "D", "#E69F00")
refbox!(4.0, y0, n, n, "P")
diagbox!(7.5, y0, n, "D", "#E69F00")
text!(ax, 0, y1 + m / 2; text = L"\tilde{A} =", align = (:right, :center), fontsize = 16)
diagbox!(0.5, y1, m, "E", "#CC79A7")
refbox!(5.5, y1, n, m, "A")
diagbox!(9.0, y1 + (m - n), n, "D", "#E69F00")
text!(ax, 13.5, y0 + n / 2; text = "stored: c and the diagonal of D\n(a scalar and an n-vector)",
      align = (:left, :center), fontsize = 13, color = :gray30)
text!(ax, 13.5, y1 + m / 2; text = "stored: the diagonals of E and D\n(an m-vector, the same n-vector)",
      align = (:left, :center), fontsize = 13, color = :gray30)
limits!(ax, -2.5, 22, -1.2, 10.2)
nothing # hide
```

:::

```@example alg_equil_figure
fig # hide
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
