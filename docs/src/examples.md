# Examples

The first seven sections are the applications from the
[OSQP documentation](https://osqp.org/docs/examples/), written against PureOSQP; the rest
work through the solver's own interface. Every block is executed when these docs are built,
so the numbers below are what the code actually produced.

Two differences from the upstream versions are worth knowing before you read the
applications. PureOSQP takes the **full symmetric** `P`, not an upper triangle. And every
matrix in them is built dense — these particular problems are highly structured and sparse,
which is where the reference implementation is strongest, so treat them as a guide to
*formulating* problems rather than as a claim about which solver to use on them.
[Matrix types](matrices.md) is the page on the other storage formats, and which to reach for.

## Basic usage

```@example demo
using PureOSQP

P = [4.0 1.0; 1.0 2.0]
q = [1.0, 1.0]
A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
l = [1.0, 0.0, 0.0]
u = [1.0, 0.7, 0.7]

sol = solve(P, q, A, l, u)
(sol.status, sol.x, sol.obj_val)
```

The default tolerances are `1e-3`, matching upstream. For a sharper answer, tighten them or
turn on polishing:

```@example demo
sharp = solve(P, q, A, l, u; eps_abs = 1e-9, eps_rel = 1e-9, polish = true)
(sharp.x, sharp.obj_val)
```

## Least-squares

Fit `Aₐx ≈ b` as closely as possible, but with bounds on `x` that a plain `\` cannot express.
Unconstrained least-squares has a closed form and does not need a solver; the moment you add
`0 ≤ x ≤ 1`, it does. The formulation below introduces `y = Aₐx - b` as its own variable, which
keeps the objective diagonal and the constraint matrix sparse instead of forming `AₐᵀAₐ`.

```math
\begin{array}{ll}
  \mbox{minimize}   & \tfrac12 \|A_d x - b\|_2^2 \\
  \mbox{subject to} & 0 \le x \le 1
\end{array}
```

Introducing `y = A_d x - b` turns this into a QP in `(x, y)`. The residual carries the whole
objective, and the constraint matrix has two row groups: `m` rows that define `y`, and `n`
rows that carry the box on `x`.

::: details Code that draws the figure

```@example ex_blocks
using CairoMakie

# One block-outline drawing of a constraint matrix. `cols` and `rows` are `name => size`
# pairs, one per variable group and per constraint-row group; a block is
# `(rows, cols, label)` with ranges of group indices. Sizes are drawn in proportion, except
# that a group thinner than `minsize` is widened to it so its blocks can hold a label;
# `margin` is the room left for the row labels, as a fraction of the drawn width. A
# label `I` or `−I` is drawn as the identity's diagonal, `0` as an empty cell, anything else
# as a filled block.
function blockfigure(cols, rows, blocks; minsize = 0, margin = 0.45, size = (640, 400), fontsize = 14)
    w = [max(Float64(s), minsize) for (_, s) in cols]
    h = [max(Float64(s), minsize) for (_, s) in rows]
    x, y = [0.0; cumsum(w)], [0.0; cumsum(h)]
    W, H = x[end], y[end]
    fig = Figure(; size)
    ax = Axis(fig[1, 1]; aspect = DataAspect())
    # y runs downward so that row group 1 is on top, as in the matrix literal.
    for (r, c, label) in blocks
        x0, x1 = x[first(c)], x[last(c) + 1]
        y0, y1 = y[first(r)], y[last(r) + 1]
        if occursin(r"^[-−]?I$", label)
            lines!(ax, [x0, x1], [-y0, -y1]; color = "#E69F00", linewidth = 3)
            text!(ax, x0 + 0.72 * (x1 - x0), -(y0 + 0.3 * (y1 - y0)); text = label,
                  align = (:center, :center), fontsize, color = "#E69F00")
        elseif label == "0"
            text!(ax, (x0 + x1) / 2, -(y0 + y1) / 2; text = "0",
                  align = (:center, :center), fontsize, color = :gray60)
        else
            poly!(ax, Rect2f(x0, -y1, x1 - x0, y1 - y0); color = ("#0072B2", 0.25))
            text!(ax, (x0 + x1) / 2, -(y0 + y1) / 2; text = label,
                  align = (:center, :center), fontsize, color = "#0072B2")
        end
    end
    for xi in x[2:(end - 1)]
        lines!(ax, [xi, xi], [0, -H]; color = :gray70, linestyle = :dash, linewidth = 1)
    end
    for yi in y[2:(end - 1)]
        lines!(ax, [0, W], [-yi, -yi]; color = :gray70, linestyle = :dash, linewidth = 1)
    end
    lines!(ax, Rect2f(0, -H, W, H); color = :black, linewidth = 1.5)
    for (i, (name, _)) in enumerate(cols)
        text!(ax, (x[i] + x[i + 1]) / 2, 0.015H; text = name,
              align = (:center, :bottom), fontsize = fontsize - 2)
    end
    for (j, (name, _)) in enumerate(rows)
        text!(ax, -0.015W, -(y[j] + y[j + 1]) / 2; text = name,
              align = (:right, :center), fontsize = fontsize - 2)
    end
    hidedecorations!(ax); hidespines!(ax)
    limits!(ax, -margin * W, 1.02W, -1.02H, 0.12H)
    return fig
end

fig = blockfigure(["x  (n = 20)" => 20, "y  (m = 30)" => 30],
                  ["y = Ad x − b" => 30, "0 ≤ x ≤ 1" => 20],
                  [(1, 1, "Ad"), (1, 2, "−I"), (2, 1, "I"), (2, 2, "0")]; size = (520, 400))
nothing # hide
```

:::

```@example ex_blocks
fig # hide
```

```@example lsq
using PureOSQP, LinearAlgebra, Random

Random.seed!(1)
m, n = 30, 20
Ad = randn(m, n)
b = randn(m)

# variables (x, y) with y = Ad*x - b
P = [zeros(n, n) zeros(n, m); zeros(m, n) Matrix(1.0I, m, m)]
q = zeros(n + m)
A = [Ad              -Matrix(1.0I, m, m);
     Matrix(1.0I, n, n)  zeros(n, m)]
l = [b; zeros(n)]
u = [b; ones(n)]

sol = solve(P, q, A, l, u; eps_abs = 1e-9, eps_rel = 1e-9, polish = true, max_iter = 100_000)
x = sol.x[1:n]
(sol.status, residual = norm(Ad * x - b), in_box = all(-1e-7 .<= x .<= 1 + 1e-7))
```

## Lasso

Least-squares with a preference for *simple* answers. The `‖x‖₁` term penalizes the total size
of the coefficients in a way that drives most of them to exactly zero, so the fit selects a
handful of predictors rather than using all of them a little. `γ` sets how aggressive that
selection is. It becomes a QP by splitting each coefficient into positive and negative parts —
that is what the extra variables below are doing.

```math
\begin{array}{ll}
  \mbox{minimize} & \tfrac12 \|A_d x - b\|_2^2 + \gamma \|x\|_1
\end{array}
```

which becomes, in `(x, y, t)`,

```math
\begin{array}{ll}
  \mbox{minimize}   & \tfrac12 y^T y + \gamma \mathbf{1}^T t \\
  \mbox{subject to} & y = A_d x - b \\
                    & -t \le x \le t
\end{array}
```

`γ` enters only through `q`, which is the case [`update!`](@ref) exists for: the whole
regularization path reuses one workspace, and each solve warm starts from the last.

Three variable groups and three row groups: the residual definition, then the two halves of
`−t ≤ x ≤ t`.

::: details Code that draws the figure

```@example ex_blocks
fig = blockfigure(["x  (n = 10)" => 10, "y  (m = 200)" => 200, "t  (n)" => 10],
                  ["y = Ad x − b" => 200, "x − t ≤ 0" => 10, "x + t ≥ 0" => 10],
                  [(1, 1, "Ad"), (1, 2, "−I"), (1, 3, "0"), (2, 1, "I"), (2, 2, "0"), (2, 3, "−I"),
                   (3, 1, "I"), (3, 2, "0"), (3, 3, "I")]; minsize = 40, size = (560, 480))
nothing # hide
```

:::

```@example ex_blocks
fig # hide
```

```@example lasso
using PureOSQP, LinearAlgebra, Random

Random.seed!(1)
n, m = 10, 200
Ad = randn(m, n)
x_true = (rand(n) .> 0.5) .* randn(n) ./ sqrt(n)
b = Ad * x_true .+ 0.5 .* randn(m)

nv = 2n + m
P = zeros(nv, nv)
P[n+1:n+m, n+1:n+m] = Matrix(1.0I, m, m)
In, Im = Matrix(1.0I, n, n), Matrix(1.0I, m, m)
A = [Ad  -Im            zeros(m, n);
     In   zeros(n, m)  -In;
     In   zeros(n, m)   In]
l = [b; fill(-Inf, n); zeros(n)]
u = [b; zeros(n); fill(Inf, n)]

ws = setup(P, zeros(nv), A, l, u; eps_abs = 1e-8, eps_rel = 1e-8, max_iter = 100_000)
for γ in (1.0, 3.0, 10.0)
    update!(ws; q = [zeros(n + m); fill(γ, n)])
    sol = solve!(ws)
    nnz = count(>(1e-4), abs.(sol.x[1:n]))
    println("γ = $γ:  $(nnz) nonzeros, ‖Ax−b‖ = $(round(norm(Ad * sol.x[1:n] - b), digits = 3))")
end
```

Sparsity increases with `γ`, as it should.

!!! note "Reading `refactor_count`"
    `update!` with only `q` never refactorizes. `ws.refactor_count` will still grow across
    this loop, because **adaptive ρ** refactorizes too — it is a total over the workspace's
    life, not a count of what `update!` did. To see whether a particular `update!`
    refactorized, read `refactor_count` immediately before and after that call.

## Huber fitting

Robust regression, replacing the squared loss with the Huber penalty so that outliers do
not dominate:

```math
\phi_{\rm hub}(t) = \begin{cases} t^2 & |t| \le 1 \\ 2|t| - 1 & |t| > 1 \end{cases}
```

The equivalent QP, in `(x, u, r, s)`:

```math
\begin{array}{ll}
  \mbox{minimize}   & u^T u + 2\,\mathbf{1}^T (r+s) \\
  \mbox{subject to} & A_d x - b - u = r - s \\
                    & r \ge 0,\quad s \ge 0
\end{array}
```

`u` carries the quadratic part of the loss and `r − s` the linear part, so the first row
group is the residual `Ad x − u − r + s = b`; the second is the identity over `(r, s)`,
carrying their nonnegativity.

::: details Code that draws the figure

```@example ex_blocks
fig = blockfigure(["x  (n = 10)" => 10, "u  (m = 100)" => 100, "r  (m)" => 100, "s  (m)" => 100],
                  ["Ad x − u − r + s = b" => 100, "r, s ≥ 0" => 200],
                  [(1, 1, "Ad"), (1, 2, "−I"), (1, 3, "−I"), (1, 4, "I"), (2, 1:2, "0"), (2, 3:4, "I")];
                  minsize = 30, size = (620, 520))
nothing # hide
```

:::

```@example ex_blocks
fig # hide
```

```@example huber
using PureOSQP, LinearAlgebra, Random

Random.seed!(1)
n, m = 10, 100
Ad = randn(m, n)
x_true = randn(n) ./ sqrt(n)
clean = rand(m) .>= 0.1                      # 10% of the measurements are outliers
b = Ad * x_true .+ 0.5 .* randn(m) .* clean .+ 10.0 .* randn(m) .* .!clean

nv = n + 3m
P = zeros(nv, nv)
P[n+1:n+m, n+1:n+m] = 2 * Matrix(1.0I, m, m)
q = [zeros(n + m); fill(2.0, 2m)]
Im = Matrix(1.0I, m, m)
A = [Ad                -Im  -Im  Im;
     zeros(2m, n + m)   Matrix(1.0I, 2m, 2m)]

sol = solve(P, q, A, [b; zeros(2m)], [b; fill(Inf, 2m)];
            eps_abs = 1e-8, eps_rel = 1e-8, polish = true, max_iter = 200_000)

x_huber = sol.x[1:n]
x_lsq = Ad \ b
(huber_error = norm(x_huber - x_true), least_squares_error = norm(x_lsq - x_true))
```

The Huber fit recovers `x_true` several times more accurately than least-squares, which is
the point of the formulation. Over twelve seeds at this outlier rate, the Huber estimate had
the lower recovery error on all twelve, with median error 0.19 against 0.95.

## Support vector machine

Draw the dividing line between two labelled classes, as far from both as you can. The `xᵀx`
term prefers a wide margin; the `max(0, ·)` hinge charges for every point on the wrong side of
it, with `γ` setting how much a misclassification costs relative to margin width. The hinge is
not quadratic, so it enters as a slack variable per data point — one extra variable and one
extra row each.

```math
\begin{array}{ll}
  \mbox{minimize} & \tfrac12 x^T x + \gamma \sum_{i=1}^m \max(0,\; b_i a_i^T x + 1)
\end{array}
```

with the hinge losses lifted into variables `t`:

```math
\begin{array}{ll}
  \mbox{minimize}   & \tfrac12 x^T x + \gamma \mathbf{1}^T t \\
  \mbox{subject to} & t \ge \mathrm{diag}(b) A_d x + 1,\quad t \ge 0
\end{array}
```

One row group per inequality: the hinge, and the identity that keeps `t ≥ 0`.

::: details Code that draws the figure

```@example ex_blocks
fig = blockfigure(["x  (n = 10)" => 10, "t  (m = 200)" => 200],
                  ["diag(b) Ad x − t ≤ −1" => 200, "t ≥ 0" => 200],
                  [(1, 1, "diag(b) Ad"), (1, 2, "−I"), (2, 1, "0"), (2, 2, "I")];
                  minsize = 80, margin = 0.9, size = (600, 420), fontsize = 13)
nothing # hide
```

:::

```@example ex_blocks
fig # hide
```

```@example svm
using PureOSQP, LinearAlgebra, Random

Random.seed!(1)
n, m = 10, 200
half = m ÷ 2
b = [ones(half); -ones(half)]
Ad = [randn(half, n) ./ sqrt(n) .+ 1 / n;
      randn(half, n) ./ sqrt(n) .- 1 / n]
γ = 1.0

P = [Matrix(1.0I, n, n) zeros(n, m); zeros(m, n) zeros(m, m)]
q = [zeros(n); fill(γ, m)]
Im = Matrix(1.0I, m, m)
A = [Diagonal(b) * Ad  -Im;
     zeros(m, n)        Im]
l = [fill(-Inf, m); zeros(m)]
u = [fill(-1.0, m); fill(Inf, m)]

sol = solve(P, q, A, l, u; eps_abs = 1e-8, eps_rel = 1e-8, polish = true, max_iter = 200_000)
w = sol.x[1:n]
accuracy = count(i -> sign((Ad*w)[i]) == -b[i], 1:m) / m
(sol.status, weight_norm = norm(w), training_accuracy = accuracy)
```

## Portfolio optimization

Split a budget across assets to earn as much as possible without taking on more risk than you
are willing to. `μ` is the expected return of each asset and `Σ` how they move together, so
`xᵀΣx` is the variance of the whole portfolio and `γ` is how much return you demand per unit of
risk. This is the textbook Markowitz problem, and it is a QP as written — no reformulation
needed. The one below is the factor-model form, which keeps `Σ` as a small factor matrix plus a
diagonal rather than a full covariance.

```math
\begin{array}{ll}
  \mbox{maximize}   & \mu^T x - \gamma\, x^T \Sigma x \\
  \mbox{subject to} & \mathbf{1}^T x = 1,\quad x \ge 0
\end{array}
```

with a factor risk model `Σ = F Fᵀ + D`. Introducing `y = Fᵀ x` keeps the quadratic term
diagonal. The constraint matrix stacks the definition of `y`, the single budget row, and a
bound per asset.

::: details Code that draws the figure

```@example ex_blocks
fig = blockfigure(["x  (n = 100)" => 100, "y  (k = 10)" => 10],
                  ["y = Fᵀ x" => 10, "1ᵀ x = 1" => 1, "0 ≤ x ≤ 1" => 100],
                  [(1, 1, "Fᵀ"), (1, 2, "−I"), (2, 1, "1ᵀ"), (2, 2, "0"), (3, 1, "I"), (3, 2, "0")];
                  minsize = 14, size = (560, 560))
nothing # hide
```

:::

```@example ex_blocks
fig # hide
```

```@example portfolio
using PureOSQP, LinearAlgebra, Random

Random.seed!(1)
n, k = 100, 10
F = randn(n, k) .* (rand(n, k) .< 0.7)
D = Diagonal(rand(n) .* sqrt(k))
μ = randn(n)
γ = 1.0

P = [Matrix(D) zeros(n, k); zeros(k, n) Matrix(1.0I, k, k)]
q = [-μ ./ (2γ); zeros(k)]
A = [F'                  -Matrix(1.0I, k, k);
     ones(1, n)           zeros(1, k);
     Matrix(1.0I, n, n)   zeros(n, k)]
l = [zeros(k); 1.0; zeros(n)]
u = [zeros(k); 1.0; ones(n)]

sol = solve(P, q, A, l, u; eps_abs = 1e-9, eps_rel = 1e-9, polish = true, max_iter = 100_000)
x = sol.x[1:n]
(budget = sum(x), smallest_weight = minimum(x),
 expected_return = dot(μ, x), risk = dot(x, (F * F' + D) * x))
```

The budget constraint holds exactly and the most negative weight is on the order of
`1e-20` — zero to machine precision, which is what a first-order method gives you on an
active bound. If you need weights that are non-negative as a hard postcondition rather than
to solver tolerance, clamp them.

## Model predictive control

The problem [`update!`](@ref) is built for. A quadcopter is driven to a reference height by
re-solving a finite-horizon optimal control problem at every step; only the initial-state
rows of `l` and `u` change, so the factorization is computed once and reused.

```math
\begin{array}{ll}
  \mbox{minimize}   & (x_N - x_r)^T Q_N (x_N - x_r) + \sum_{k=0}^{N-1} (x_k - x_r)^T Q (x_k - x_r) + u_k^T R u_k \\
  \mbox{subject to} & x_{k+1} = A x_k + B u_k \\
                    & x_{\min} \le x_k \le x_{\max},\quad u_{\min} \le u_k \le u_{\max} \\
                    & x_0 = \bar x
\end{array}
```

Stacked over the horizon as `z = (x₀, …, x_N, u₀, …, u_{N−1})`, the dynamics rows are
block-bidiagonal: row group `k` holds `Ad` under `x_{k−1}`, `−I` under `x_k`, and `Bd` under
`u_{k−1}`, so the input columns sit one stage behind the state they produce. The first row
group pins `x₀` to the measured state, and is the only part that changes between solves.
Below these rows `A` stacks the identity, one bound per variable; it is not drawn. The `u`
columns are drawn wider than their true four so the blocks can be labeled.

::: details Code that draws the figure

```@example ex_blocks
nx, nu, N = 12, 4, 10
sub(k) = join(Char(0x2080 + d - '0') for d in string(k))
cols = [["x$(sub(k))" => nx for k in 0:N]; ["u$(sub(k))" => nu for k in 0:N-1]]
rows = [["x₀ = x̄" => nx]; ["k = $k" => nx for k in 1:N]]
blocks = [[(k + 1, k + 1, "−I") for k in 0:N];
          [(k + 1, k, "Ad") for k in 1:N];
          [(k + 1, N + 1 + k, "Bd") for k in 1:N]]
fig = blockfigure(cols, rows, blocks; minsize = 8, margin = 0.3, size = (720, 500), fontsize = 10)
nothing # hide
```

:::

```@example ex_blocks
fig # hide
```

```@example mpc
using PureOSQP, LinearAlgebra, Printf

Ad = [1.0 0 0 0 0 0 0.1 0 0 0 0 0
      0 1.0 0 0 0 0 0 0.1 0 0 0 0
      0 0 1.0 0 0 0 0 0 0.1 0 0 0
      0.0488 0 0 1.0 0 0 0.0016 0 0 0.0992 0 0
      0 -0.0488 0 0 1.0 0 0 -0.0016 0 0 0.0992 0
      0 0 0 0 0 1.0 0 0 0 0 0 0.0992
      0 0 0 0 0 0 1.0 0 0 0 0 0
      0 0 0 0 0 0 0 1.0 0 0 0 0
      0 0 0 0 0 0 0 0 1.0 0 0 0
      0.9734 0 0 0 0 0 0.0488 0 0 0.9846 0 0
      0 -0.9734 0 0 0 0 0 -0.0488 0 0 0.9846 0
      0 0 0 0 0 0 0 0 0 0 0 0.9846]
Bd = [0 -0.0726 0 0.0726
      -0.0726 0 0.0726 0
      -0.0152 0.0152 -0.0152 0.0152
      0 -0.0006 0 0.0006
      0.0006 0 -0.0006 0
      0.0106 0.0106 0.0106 0.0106
      0 -1.4512 0 1.4512
      -1.4512 0 1.4512 0
      -0.3049 0.3049 -0.3049 0.3049
      0 -0.0236 0 0.0236
      0.0236 0 -0.0236 0
      0.2107 0.2107 0.2107 0.2107]
nx, nu = size(Bd)

u_hover = 10.5916
umin = fill(9.6, nu) .- u_hover
umax = fill(13.0, nu) .- u_hover
xmin = [-pi/6, -pi/6, -Inf, -Inf, -Inf, -1.0, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf]
xmax = [pi/6, pi/6, Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf]

Q = Diagonal([0, 0, 10.0, 10, 10, 10, 0, 0, 0, 5, 5, 5])
QN = Q
R = 0.1 * Matrix(1.0I, nu, nu)
x0 = zeros(nx)
xr = [0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # hover one metre up
N = 10

# Stack the horizon into one QP over z = (x_0, …, x_N, u_0, …, u_{N-1}).
nv = (N + 1) * nx + N * nu
P = zeros(nv, nv)
for k in 0:N
    P[k*nx+1:(k+1)*nx, k*nx+1:(k+1)*nx] = k == N ? QN : Q
end
for k in 0:N-1
    o = (N + 1) * nx + k * nu
    P[o+1:o+nu, o+1:o+nu] = R
end
q = [repeat(-Q * xr, N); -QN * xr; zeros(N * nu)]

Ax = zeros((N + 1) * nx, (N + 1) * nx)
for k in 0:N
    Ax[k*nx+1:(k+1)*nx, k*nx+1:(k+1)*nx] = -Matrix(1.0I, nx, nx)
end
for k in 1:N
    Ax[k*nx+1:(k+1)*nx, (k-1)*nx+1:k*nx] = Ad
end
Bu = zeros((N + 1) * nx, N * nu)
for k in 1:N
    Bu[k*nx+1:(k+1)*nx, (k-1)*nu+1:k*nu] = Bd
end
A = [[Ax Bu]; Matrix(1.0I, nv, nv)]
l = [-x0; zeros(N * nx); repeat(xmin, N + 1); repeat(umin, N)]
u = [-x0; zeros(N * nx); repeat(xmax, N + 1); repeat(umax, N)]

ws = setup(P, q, A, l, u; eps_abs = 1e-6, eps_rel = 1e-6, max_iter = 20_000)

x = copy(x0)
for step in 1:15
    sol = solve!(ws)
    sol.status == PureOSQP.SOLVED || error("step $step: $(sol.status)")
    control = sol.x[(N+1)*nx+1:(N+1)*nx+nu]
    global x = Ad * x + Bd * control
    # Only the initial-state rows change, so no refactorization is needed.
    l[1:nx] .= -x
    u[1:nx] .= -x
    update!(ws; l = l, u = u)
    step % 5 == 0 && @printf("step %2d: height = %.4f, ‖x − xr‖ = %.4f\n", step, x[3], norm(x - xr))
end

(final_height = x[3], factorizations = ws.refactor_count)
```

Fifteen closed-loop solves, **one factorization**. That is the whole reason to reach for
`update!` rather than rebuilding the workspace: the initial-state bounds move every step,
but no row changes constraint class, so `ρ` is untouched and the factorization stays valid.

## Building a workspace once

`solve` builds a workspace, solves, and throws the workspace away. [`setup`](@ref) hands it
back instead, so the equilibration factors, the buffers and the factorization survive to the
next [`solve!`](@ref) — and so do the iterates, which is what makes the second solve short.

```@example workspace
using PureOSQP, LinearAlgebra

P = [4.0 1.0; 1.0 2.0]
q = [1.0, 1.0]
A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
l = [1.0, 0.0, 0.0]
u = [1.0, 0.7, 0.7]

ws = setup(P, q, A, l, u; eps_abs = 1e-9, eps_rel = 1e-9)
first_solve = solve!(ws)
warm_solve = solve!(ws)
(dimensions(ws), first_solve.iter, warm_solve.iter)
```

[`cold_start!`](@ref) throws the iterates away without touching anything else, and
[`warm_start!`](@ref) seeds them from a point you already have. Both leave the factorization
alone, so neither costs a refactorization.

```@example workspace
cold_start!(ws)
cold = solve!(ws)

cold_start!(ws)
warm_start!(ws; x = first_solve.x, y = first_solve.y)
seeded = solve!(ws)

(cold.iter, seeded.iter)
```

The Lasso section updates `q` and the MPC section updates `l` and `u`; `P` and `A` are the
two that always refactorize.

```@example workspace
before = ws.refactor_count
update!(ws; P = [6.0 1.0; 1.0 3.0], A = [1.0 1.0; 1.0 0.0; 0.0 2.0])
after = ws.refactor_count
resolved = solve!(ws)
(refactorizations = after - before, x = resolved.x)
```

Settings can be replaced afterwards. `rho`, `sigma` and `rho_is_vec` are built into the
factorization, so changing one of those refactorizes and the rest are free.

```@example workspace
update_settings!(ws; eps_abs = 1e-6, polish = true)
update_rho!(ws, 1.0)
(ws.settings.eps_abs, ws.settings.polish, live_rho = ws.rho, setting_rho = ws.settings.rho)
```

`update_rho!` sets the value the solver is running with; `ws.settings.rho` keeps the one
`setup` was given. Two keywords are refused rather than honored, because the workspace
cannot act on them:

```@example workspace
try
    update_settings!(ws; linsys = :kkt)
catch err
    println(sprint(showerror, err))
end
```

[`capabilities`](@ref) reports what this build supports, for the packages currently loaded:

```@example workspace
capabilities()
```

## What a solve reports

The default tolerances leave a residual around `1e-3`. Polishing guesses the active set at
the ADMM solution and solves the resulting equality-constrained QP exactly, which usually
takes the KKT error to machine precision for the price of one factorization.

```@example report
using PureOSQP

P = [4.0 1.0; 1.0 2.0]
q = [1.0, 1.0]
A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
l = [1.0, 0.0, 0.0]
u = [1.0, 0.7, 0.7]

plain = solve(P, q, A, l, u)
polished = solve(P, q, A, l, u; polish = true)
(plain.rel_kkt_error, plain.status_polish, polished.rel_kkt_error, polished.status_polish)
```

`status_polish` distinguishes the five outcomes; `polished` is the narrower question of
whether it was `POLISH_SUCCESS`. Polishing is accepted only when it improves both residuals,
so `POLISH_FAILED` means the answer is the unpolished one, not that anything went wrong.

Everything else a [`Solution`](@ref) carries:

```@example report
(status = polished.status, iter = polished.iter,
 obj_val = polished.obj_val, dual_obj_val = polished.dual_obj_val,
 duality_gap = polished.duality_gap, rel_kkt_error = polished.rel_kkt_error,
 rho_estimate = polished.rho_estimate, rho_updates = polished.rho_updates)
```

The timings are in seconds, on whichever machine built these docs. `run_time` charges
`setup_time` to the first solve only, so a re-solve reports what that re-solve actually
cost:

```@example report
map(t -> round(t; digits = 6),
    (; polished.setup_time, polished.solve_time, polished.polish_time, polished.run_time))
```

## Measuring how fast a solve converges

`sol.iter` tells you how many iterations a solve took, but not *how* it got there. Two runs
can take the same number of iterations while one spends most of them near the answer and the
other only arrives at the end. `profile_primdual = true` measures that difference:

```@example primdual
using PureOSQP, LinearAlgebra

n = 30
M = randn(n, n)
P = Matrix(Symmetric(M'M)) + n * I
A = Matrix(1.0I, n, n)
q = randn(n)
l, u = fill(-0.5, n), fill(0.5, n)

sol = solve(P, q, A, l, u; eps_abs = 1e-9, eps_rel = 1e-9, profile_primdual = true)
(iter = sol.iter, trapezoid = sol.primdual_int, logmean = sol.primdual_int_log)
```

**What the number is.** The area under the duality-gap curve over the solve, in
**gap × seconds**. Smaller means the gap shrank sooner. It is a *relative* measure: useful
comparing two runs of the same problem, meaningless on its own, and not comparable across
machines — it integrates against wall-clock time, so a different CPU gives a different number
for the same solve.

**Why there are two.** They are two estimates of one quantity. The solver knows the gap only
where it refreshes residuals — every `check_termination` iterations, 25 by default — so a
rule has to assume what the gap did in between. `primdual_int` assumes a straight line
between samples; `primdual_int_log` assumes the exponential decay a converging gap actually
follows. A straight line drawn over a decaying curve sits above it, so the trapezoid reads
high and the truth lies between the two.

**Which to use.** Take `primdual_int_log`, and read the ratio between them as its error bar.
When the two are close the samples resolve the curve and either number is sound; when they
are far apart they do not. At the default interval the trapezoid runs about 3.4× high and the
log-mean about 16% low; sampling every iteration brings the ratio to 0.93
([Benchmarks](@ref "The primal-dual integral")).

```@example primdual
dense = solve(
    P, q, A, l, u; eps_abs = 1e-9, eps_rel = 1e-9,
    profile_primdual = true, check_termination = 1,
)
(ratio_default = sol.primdual_int_log / sol.primdual_int,
 ratio_dense = dense.primdual_int_log / dense.primdual_int)
```

Lower `check_termination` to sample more densely, at the cost of testing termination more
often. Profiling itself costs under 1% and does not change the answer or the iteration count.

## Choosing the linear system

`linsys = :auto` picks a backend from the representation of `P` and `A`, as above.
`linsys = :kkt` overrides that and factors the full `(n+m)×(n+m)` quasi-definite system with
Bunch-Kaufman, which is what the reference implementation does: slower, but it does not
square the conditioning of `A`, so it is the one to reach for when a result is in question.

```@example report
kkt = setup(P, q, A, l, u; linsys = :kkt, eps_abs = 1e-9, eps_rel = 1e-9)
auto = setup(P, q, A, l, u; eps_abs = 1e-9, eps_rel = 1e-9)
(PureOSQP.backend_name(kkt.linsys), PureOSQP.backend_name(auto.linsys),
 solve!(kkt).x, solve!(auto).x)
```

`linsys = :indirect` is the third: preconditioned conjugate gradients on the reduced system,
which is never formed. It is for a matrix that can only supply products, or one large and
sparse enough that forming an `n×n` inverse is the dominant cost. It lives in a package
extension over Krylov.jl, so it exists only once Krylov is loaded — without it, `setup` says
so rather than falling back.

```julia
using PureOSQP, Krylov          # Krylov.jl is a weak dependency; add it yourself

capabilities().indirect_solver  # true only with Krylov loaded
ws = setup(P, q, A, l, u; linsys = :indirect, cg_max_iter = 20)
solve!(ws)
```

The inner solve is inexact — its tolerance follows the ADMM residuals — so the iterates
differ from the direct backends in the last digits even though both converge to the same
point.

## Solution derivatives

[`adjoint_derivative`](@ref) differentiates the KKT conditions at the solution the workspace
holds, so it gives the gradients of a scalar loss with respect to all five pieces of problem
data from one factorization. Given `∂L/∂x` and `∂L/∂y`, it returns `∂L/∂P`, `∂L/∂q`,
`∂L/∂A`, `∂L/∂l` and `∂L/∂u`.

Here `L = x₁`, with the budget row `x₁ + x₂ = 1` the only active constraint:

```@example deriv
using PureOSQP

P = [4.0 1.0; 1.0 2.0]
q = [1.0, 1.0]
A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
l = [1.0, 0.0, 0.0]
u = [1.0, 1.0, 1.0]

ws = setup(P, q, A, l, u; eps_abs = 1e-10, eps_rel = 1e-10, polish = true)
sol = solve!(ws)
grad = adjoint_derivative(ws, [1.0, 0.0], zeros(3))
(sol.x, grad.dq, grad.dl)
```

Against central differences on `q`:

```@example deriv
h = 1e-6
loss(qq) = solve(P, qq, A, l, u; eps_abs = 1e-10, eps_rel = 1e-10, polish = true).x[1]
fd = [(loss(q .+ h .* e) - loss(q .- h .* e)) / 2h for e in ([1.0, 0.0], [0.0, 1.0])]
(grad.dq, fd)
```

[`forward_derivative`](@ref) goes the other way, giving the derivative of the solution along
a perturbation of the data. Widening the budget from `1` to `1 + t` moves both variables,
and the two moves sum to the extra budget:

```@example deriv
dx, dy = forward_derivative(ws; dl = [1.0, 0.0, 0.0], du = [1.0, 0.0, 0.0])
(dx, sum(dx))
```

The derivative exists only where the active set is stable. A row resting on its bound with a
multiplier of zero, or an active-set KKT matrix that is singular or nearly so, makes the
solution map non-differentiable, and both throw rather than return a regularized number that
would look like an answer.

### A QP as a differentiable layer

The derivatives above are called by hand. Loading
[ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl) instead makes
[`solve`](@ref) differentiable to any AD package that consumes ChainRules — Zygote among them
— so a QP can sit inside a loss and be trained through with no gradient plumbing of your own.

Here a QP is fitted to a target: `q` is the parameter, and the loss is how far the solution
lands from where we want it.

```@example layer
using PureOSQP, ChainRulesCore, Zygote, LinearAlgebra

P = [4.0 1.0; 1.0 2.0]
A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
l = [-1.0, -0.6, -0.6]
u = [1.0, 0.6, 0.6]
target = [0.3, -0.2]

# `q` is what we are fitting. Everything inside the loss is an ordinary solve.
loss(q) = sum(abs2, solve(P, q, A, l, u; eps_abs = 1e-10, eps_rel = 1e-10).x .- target)

q = foldl(1:60; init = [0.5, 0.5]) do qk, _
    qk .- 0.5 .* only(Zygote.gradient(loss, qk))
end
(fitted_q = round.(q; digits = 5),
 x = round.(solve(P, q, A, l, u; eps_abs = 1e-10, eps_rel = 1e-10).x; digits = 5),
 target, loss = round(loss(q); digits = 12))
```

Gradient descent drives the solution onto the target, differentiating through the solver at
every step.

**This differentiates the solution, not the iteration.** The rules call
[`adjoint_derivative`](@ref) and [`forward_derivative`](@ref), which differentiate the KKT
conditions at the active set: one linear solve, reusing a factorization the solve already
produced, independent of how many iterations it took. Taping the ADMM loop instead would cost
memory in proportion to the iteration count and return the derivative of the iterate rather
than of the solution.

Two things follow, and both are refusals rather than approximations:

- **The solve must converge.** The KKT conditions hold at the solution and nowhere else, so a
  run that stopped at `max_iter` raises rather than returning the gradient of a point that is
  not the answer.
- **The active set must be non-degenerate**, which `adjoint_derivative` already requires. A
  least-squares answer there would have the right shape and units while being a different
  quantity, and nothing downstream could tell.

`polish = true` is set for you unless you ask otherwise: the derivative is taken at the active
set, and polishing is what identifies it exactly.

One reach limit worth knowing: `rrule` and `frule` cover every AD backend that consumes
ChainRules, which includes Zygote. Mooncake needs an explicit `Mooncake.@from_rrule`, and
Enzyme its own `EnzymeRules` shim.

## Infeasible problems

A problem with no feasible point stops with `PRIMAL_INFEASIBLE` and a certificate `v`
satisfying `Aᵀv = 0` with a negative support value, which proves it. Here the two rows ask
for `x ≥ 1` and `x ≤ 0`:

```@example infeasible
using PureOSQP, LinearAlgebra

A = [1.0; 1.0;;]
sol = solve([1.0;;], [0.0], A, [1.0, -Inf], [Inf, 0.0])
(sol.status, sol.prim_inf_cert, residual = norm(A' * sol.prim_inf_cert, Inf), sol.x)
```

An unbounded problem stops with `DUAL_INFEASIBLE` and a certificate `d` — a direction along
which the objective falls without limit. Minimizing `-x` over the whole line:

```@example infeasible
unbounded = solve(zeros(1, 1), [-1.0], [1.0;;], [-Inf], [Inf])
(unbounded.status, unbounded.dual_inf_cert, unbounded.obj_val, unbounded.x)
```

Neither carries a primal-dual point, so `x` and `y` come back as `NaN` rather than as the
last iterate, which is on a diverging ray. `obj_val` is `Inf` for a primal infeasibility and
`-Inf` for a dual one. The certificate that does not apply is an empty vector:

```@example infeasible
(length(sol.dual_inf_cert), length(unbounded.prim_inf_cert))
```

## JuMP and MathOptInterface

The MathOptInterface wrapper is a package extension, loaded when MathOptInterface is. Every
field of [`Settings`](@ref) is a raw optimizer attribute of the same name. This block is not
run here, since these docs do not depend on JuMP:

```julia
using JuMP, PureOSQP

model = Model(PureOSQP.Optimizer)
set_optimizer_attribute(model, "polish", true)
set_optimizer_attribute(model, "eps_abs", 1e-9)

@variable(model, 0 <= x[1:2] <= 0.7)
@constraint(model, sum(x) == 1)
@objective(model, Min, 2x[1]^2 + x[1] * x[2] + x[2]^2 + x[1] + x[2])

optimize!(model)
value.(x)        # [0.3, 0.7]
```

`PureOSQP.Optimizer` is the only name the core package owns; the wrapper itself lives in the
extension, so a caller who does not use MathOptInterface pays nothing for it.
