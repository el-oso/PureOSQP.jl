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
[Matrix representations](@ref) below covers the other storage formats.

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

Introducing `y = A_d x - b` turns this into a QP in `(x, y)`, with the residual carrying the
whole objective:

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
diagonal:

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

## Matrix representations

Everything above built `P` and `A` as ordinary dense matrices. That always works, and if your
problems are small it is all you need — you can stop reading here and come back when one gets
slow.

The rest of this page is about what to do when they do get slow. The short version: **how you
store `P` and `A` changes how much work the solver has to do**, sometimes by a factor of
hundreds, and you get that by passing a different matrix type rather than by changing any
setting.

Two words are used throughout, so here they are once:

- The **reduced matrix** is the `n×n` matrix the solver has to solve against on every
  iteration. You never see it, but it is where nearly all the time goes. Its size and shape
  come from `P` and `A`, and the whole point of the sections below is to keep it small or
  cheap. (It is `R = cDPD + σI + Ãᵀdiag(ρ)Ã`, if you want the formula; you do not need it to
  use any of this.)
- A **backend** is the code that solves against that matrix. There are ten or so. You do not
  choose one — `setup` looks at the types of `P` and `A` and picks. `PureOSQP.backend_name(ws.linsys)`
  tells you which it picked, and every example below uses that to show the choice being made.

So the workflow is always the same: pass a matrix type that describes your problem, then check
which backend you got. If it is the one you expected, the structure was used.

### Which representation, and why

There are four answers, and which is right is a property of your problem rather than a
preference. The short version:

| your problem | use | because |
|---|---|---|
| small enough to sit in cache | **dense** | nothing beats a contiguous array the CPU can keep close. Structure costs indirection that buys nothing at this size. |
| large, and mostly zeros | **sparse** | you pay for the nonzeros instead of `n²`. This is the familiar case and `SparseMatrixCSC` handles it. |
| you know more about it than "where the zeros are" | **a structured type** | block-diagonal, low-rank, Kronecker. The solver can then skip work that no sparsity pattern reveals — a `BlockDiagonal` is solved as `K` small systems, never as one big one. |
| few zeros, but a fast way to apply it, and too big for cache | **unmaterialized** | past cache the dense product is limited by memory bandwidth, not arithmetic. An operator that computes its product from `O(n)` stored numbers moves almost nothing and can win outright. |

That last row is the one that is easy to miss, so it is worth being concrete about. The tables
below come from `bench/representation_choice.jl`, single-threaded, statuses asserted — a run
that stopped at `max_iter` is not a faster answer to the same question.

**When being matrix-free does not pay.** An operator whose product costs what the dense product
costs saves a factorization once and pays for it every iteration:

| n | iterations | operator | dense | |
|---|---|---|---|---|
| 200 | 125/125 | 6.0 ms | 2.2 ms | 0.37× |
| 500 | 150/125 | 56.1 ms | 20.8 ms | 0.37× |
| 1000 | 175/175 | 220 ms | 131 ms | 0.60× |

**When it does.** The same comparison, for an operator applied in `O(n)` whose dense form is
`O(n²)`, at about a tenth of the entries nonzero — too dense for a sparse format to be the
obvious answer:

| n | fill | iterations | operator | dense | | dense `A` |
|---|---|---|---|---|---|---|
| 500 | 9.9% | 100/75 | 10.2 ms | 15.8 ms | **1.55×** | 1.9 MiB |
| 1000 | 9.8% | 100/100 | 40.5 ms | 104 ms | **2.57×** | 7.6 MiB |
| 2000 | 9.8% | 100/100 | 348 ms | 730 ms | **2.10×** | 30.5 MiB |
| 4000 | 9.8% | 100/75 | 1761 ms | 4806 ms | **2.73×** | 122 MiB |

Same solver, same tolerances, both converged. The difference between the two tables is not the
size and not the sparsity — it is whether **applying** the operator is asymptotically cheaper
than the dense product. If it is, the operator wins and wins by more as the matrix leaves
cache. If it is not, no amount of size will save it.

### Unmaterialized does not mean solved by CG

One more distinction, because it decides what happens on badly conditioned problems.

An operator with no exploitable structure can only be served by *conjugate gradients* —
multiply, repeat — and CG is sensitive to conditioning. But an operator that carries its own
**direct** backend is solved by factoring, and conditioning is then no worse than the structure
implies.

The Kronecker type is the clean example. `κ(A₁ ⊗ A₂) = κ(A₁)·κ(A₂)`, so an operator at
`κ = 1e12` is built from two factors at `1e6` — and the backend eigendecomposes the *factors*,
never forming or factoring the product:

| n | κ(A) | iterations | Kronecker | dense | | conjugate gradients, same problem |
|---|---|---|---|---|---|---|
| 400 | 1e12 | 625/625 | 2.4 ms | 29.6 ms | **12×** | converges, but in 1025 iterations |
| 1600 | 1e12 | 1100/1100 | 22.4 ms | 1560 ms | **70×** | `MAX_ITER_REACHED` at 20 000 |

Both routes agree on the objective to six figures. CG on the same problem manages it at
`n = 400` and fails outright at `n = 1600` — so on an ill-conditioned problem the useful move
is a structured operator with a direct backend, not a generic one served iteratively.

The problems below are all the same QP, written five ways.

```@example storage
using PureOSQP, LinearAlgebra, SparseArrays

n = 6
Pdiag = Diagonal(2.0 .+ (1:n) ./ n)
Aband = Bidiagonal(fill(1.0, n), fill(-1.0, n - 1), :U)
q = collect(range(-1.0, 1.0; length = n))
l = fill(-0.5, n)
u = fill(0.5, n)

Pd, Ad = Matrix(Pdiag), Matrix(Aband)
reps = [
    "Matrix"                      => (Pd, Ad),
    "Diagonal, Bidiagonal"        => (Pdiag, Aband),
    "SymTridiagonal, Tridiagonal" => (SymTridiagonal(diag(Pd), zeros(n - 1)), Tridiagonal(Ad)),
    "Symmetric, SubArray"         => (Symmetric(Pd), view(Ad, :, :)),
    "SparseMatrixCSC"             => (sparse(Pd), sparse(Ad)),
]

P_before, A_before = copy(Pd), copy(Ad)
reference = setup(Pd, q, Ad, l, u; eps_abs = 1e-9, eps_rel = 1e-9)
ref = solve!(reference)
ref_backend = PureOSQP.backend_name(reference.linsys)
for (name, (Pr, Ar)) in reps
    ws = setup(Pr, q, Ar, l, u; eps_abs = 1e-9, eps_rel = 1e-9)
    @assert ws.P === Pr && ws.A === Ar          # held by reference, not copied
    sol = solve!(ws)
    backend = PureOSQP.backend_name(ws.linsys)
    @assert sol.iter == ref.iter                       # same trajectory
    @assert isapprox(sol.x, ref.x; rtol = 1e-8)        # same answer
    # Bit-exact only where the same factorization ran; see below.
    backend == ref_backend && @assert sol.x == ref.x && sol.y == ref.y
    println(rpad(name, 30), "iter = ", sol.iter, ",  backend = ", backend)
end
@assert Pd == P_before && Ad == A_before        # the caller's arrays are never written to
```

Two different things are on display here, and it is worth keeping them apart.

**A representation that only changes how entries are reached gives bit-identical answers.**
`Symmetric`, a `SubArray` and a `SparseMatrixCSC` all feed the same numbers into the same
arithmetic, so `==` holds exactly against the dense reference.

**A representation that changes which backend is chosen changes the arithmetic.** A
`Diagonal` `P` with a `Bidiagonal` `A` makes the reduced matrix tridiagonal, and that is
solved by an `ldlt` on two bands rather than by a dense inverse and a `symv` — a different
factorization, agreeing to about `1e-16` rather than to the bit. The iteration count and the
answer are the same; the last digits are not. See
[Which backend a structured matrix gets](@ref "Which backend a structured matrix gets").

`P` has to be symmetric *as stored* — a lower triangle with the upper one left at zero is
rejected rather than mirrored, since that matrix is a different, non-symmetric problem. Wrap
it in `Symmetric` to say which triangle is the real one.

### Which backend a sparse matrix gets

Eliminating `ν` from the ADMM subproblem gives an `n×n` reduced matrix, and whether that
matrix is worth keeping sparse depends on its pattern rather than on the input's density.
`linsys = :auto` decides by asking CHOLMOD to factor the pattern and measuring the fill.

```@example storage
band = 200
banded = setup(
    sparse(SymTridiagonal(fill(2.0, band), fill(0.3, band - 1))),
    collect(range(-1.0, 1.0; length = band)),
    sparse(Bidiagonal(fill(1.0, band), fill(-1.0, band - 1), :U)),
    fill(-1.0, band), fill(1.0, band),
)
PureOSQP.backend_name(banded.linsys)
```

A banded `A` gives a banded reduced matrix, so it is formed and factored sparsely. A
scattered pattern fills in, and then the sparse factor is no cheaper than the dense inverse;
that case still forms the reduced matrix over the stored entries, but factors it densely.

```@example storage
using Random
Random.seed!(1)
ns, ms = 80, 160
scattered = setup(
    sparse(1.0I, ns, ns), collect(range(-1.0, 1.0; length = ns)),
    sprandn(ms, ns, 0.05), fill(-1.0, ms), fill(1.0, ms),
)
PureOSQP.backend_name(scattered.linsys)
```

Both are reported by `PureOSQP.backend_name(ws.linsys)`, which names whichever backend the
workspace ended up with. The dense default is `:cholesky`, and the full quasi-definite
factorization is `:bunchkaufman`.

### Which backend a structured matrix gets

A structured `P` and `A` are not just read more cheaply — they can make the reduced matrix
itself narrow, and then there is far less to factor. Eliminating `ν` gives

```math
R = c D P D + \sigma I + \tilde A^\top \mathrm{diag}(\rho) \tilde A
```

Diagonal scaling preserves a bandwidth and `ÃᵀρÃ` doubles `A`'s, so
`bandwidth(R) = max(bandwidth(P), 2 bandwidth(A))`. `linsys = :auto` dispatches on the pair
of types, with no setting and no density gate involved.

```@example structured
using PureOSQP, LinearAlgebra
n = 200
q, l, u = collect(range(-1.0, 1.0; length = n)), fill(-1.0, n), fill(1.0, n)

# A separable objective under box constraints: R is diagonal, so nothing is factored.
box = setup(Diagonal(fill(2.0, n)), q, Diagonal(ones(n)), l, u)

# A tridiagonal objective under box constraints: R stays tridiagonal.
smooth = setup(SymTridiagonal(fill(2.0, n), fill(0.3, n - 1)), q, Diagonal(ones(n)), l, u)

(PureOSQP.backend_name(box.linsys), PureOSQP.backend_name(smooth.linsys))
```

The first has nothing to factor at all — a solve is `n` divisions — and the second is an
`ldlt` that costs `O(n)`. Against the dense path the same problems would otherwise take,
that is worth a great deal at any size worth caring about; see
[Structured backends](@ref "Structured backends") for the measurements.

Widening `A` widens `R` faster than widening `P` does, which is the practical consequence of
the rule above. A `Tridiagonal` `A` squares to bandwidth 2, past what `SymTridiagonal`
stores, and is served by a banded Cholesky once BandedMatrices.jl is loaded:

```@example structured
using BandedMatrices
diff = setup(
    SymTridiagonal(fill(4.0, n), fill(0.3, n - 1)), q,
    Tridiagonal(fill(-0.25, n - 1), ones(n), fill(-0.25, n - 1)), l, u,
)
(PureOSQP.backend_name(diff.linsys), diff.linsys.bw)
```

Without BandedMatrices loaded that problem takes the dense path instead — correctly, just
not cheaply. Structure in `P` alone never survives: `ÃᵀρÃ` is dense for a general `A`
whatever `P` looked like, so a `Diagonal` `P` with a dense `A` is a dense reduced matrix and
gets the dense backend.

```@example structured
using Random
Random.seed!(2)
PureOSQP.backend_name(setup(Diagonal(fill(2.0, 40)), q[1:40], randn(60, 40),
                            fill(-1.0, 60), fill(1.0, 60)).linsys)
```

### Supplying a matrix type of your own

`P` and `A` are held by reference and reached through a small set of functions, so a
representation the package has never heard of works by declaring itself
`<: AbstractMatrix{T}` and supplying `size`, `mul!`, and `mul!` against its adjoint. That
much is enough to solve. Everything below is optional and each override replaces one generic
walk over entries with whatever the representation can answer more cheaply.

There are two seam levels, and which one a representation wants depends on whether it can
enumerate a column.

**Per column.** [`PureOSQP.structural_rows`](@ref)`(M, j)` names the rows column `j` can hold
a nonzero in; the four traversals in `src/scaling.jl` — `weighted_colmax`,
`weighted_colmax_rowmax!`, `scaled_col!` and `add_scaled_col!` — follow it, so a single
`structural_rows` method makes equilibration and the dense formation cost the column's own
entries rather than all `m` of them. A representation whose columns are cheaper to walk than
to index overrides the four traversals directly instead; the sparse extension does that,
because `M[i, j]` on a `SparseMatrixCSC` is a binary search.

**Per sweep.** `column_norms!` and `cost_norms!` are the whole of what equilibration asks
per sweep, so a representation that answers in whole-matrix or closed form overrides those
two and never sees a column index. The GPU extension is the shipped example: it replaces
both with array reductions, which is what lets a device array equilibrate under
`allowscalar(false)`.

Beyond equilibration there are three more override points, all optional:
`PureOSQP.reduced_diagonal!` for the matrix-free preconditioner,
[`PureOSQP.is_convex`](@ref) for the convexity test `setup` runs before choosing a backend,
and [`PureOSQP.is_symmetric`](@ref) for the symmetry check — the last two both densify or
scan `n²` positions otherwise.

[`PureOSQP.RowCoupled`](@ref) is the worked example in the package itself: a few dense rows
above a block holding one entry per row. It defines `size`, `getindex` and `mul!`, and adds
one `structural_rows` method; that is all it takes for a `Diagonal` `P` with a `RowCoupled`
`A` to reach the low-rank backend and to equilibrate at the cost of its own entries.

An operator that supplies **only** products — nothing to index at all — says so with
[`PureOSQP.is_materializable`](@ref):

```julia
PureOSQP.is_materializable(::MyOperator) = false
```

`linsys = :auto` then declines the dense terminal and lands on the matrix-free backend, which
needs Krylov.jl loaded. `polish!` and the two derivative entry points build a dense matrix
out of `P` and `A` entry by entry, so they refuse such an operator by name rather than
failing inside a factorization: pass `polish = false`, and differentiate a materialized form
of the problem. Equilibration also walks columns, so an operator that overrides neither seam
level needs `scaling = 0`.

The hot-path guarantees carry a condition here that they do not carry elsewhere. `admm_step!`
allocates nothing and is type-stable for a caller-supplied operator only as far as that
operator's own `mul!` is: a broadcast in it, or a `DimensionMismatch` message built from a
type, is enough to lose both. `bench/lazy_operator.jl` is written to hold them, and
`bench/strictmode_audit.jl` checks it.

### An operator from LinearMaps.jl

**Use this when your constraint is something you can *do* but would never want to *store*.**

The situation is common in signal and image work. "Take a running total." "Blur this." "Take a
Fourier transform, keep the low frequencies." Each of those is a perfectly good linear
constraint, and each has a matrix — but for a million-pixel image that matrix has `10¹²`
entries and cannot exist. What you have instead is a function that applies it.

[LinearMaps.jl](https://github.com/JuliaLinearAlgebra/LinearMaps.jl) is the standard Julia
package for exactly that: an object you can multiply by, built from a function. Load it and
this solver accepts one anywhere it accepts a matrix. Nothing else is needed — matrices and
maps can even be mixed in the same call.

#### When this is the right tool

Four situations, in rough order of how often they come up:

1. **The matrix will not fit.** Deblurring a 1000×1000 image is a million variables, so `A` is
   a million by a million: `8` terabytes dense. There is no trade to weigh here — an operator
   is the only way the problem exists at all. Same story for 3-D grids, large PDE-constrained
   problems, and anything where `n` runs past `10⁵`.
2. **Applying it is much cheaper than its size suggests.** A convolution or blur is a *dense*
   matrix — every output touches every input — but applying it through an FFT costs
   `O(n log n)` instead of `O(n²)`. Storing it throws that away. The same holds for any
   transform with a fast algorithm: DCT, wavelets, a fast multipole method.
3. **You already have the code, not the entries.** The operator is a simulator, an existing
   forward model, a PDE solve, a linearization somebody else wrote. You can call it; nobody
   ever assembled it, and assembling it would mean `n` separate calls.
4. **Memory is the binding constraint, not time.** The matrix-free path stores vectors where
   the direct path stores an `n×n` inverse — [33× less at `n = 4000`](@ref "The matrix-free
   backend"). If the problem does not fit in RAM, being slower is not the issue.

#### When it is the wrong tool

**If applying your operator costs about what the dense product costs, use the matrix.** Being
matrix-free is not free: it trades a factorization you pay once for an iterative solve you pay
every iteration. When the product itself is no cheaper, that trade only loses:

| n | iterations | operator | matrix | |
|---|---|---|---|---|
| 200 | 125/125 | 6.0 ms | 2.2 ms | **2.7× slower** |
| 500 | 150/125 | 56.1 ms | 20.8 ms | **2.7× slower** |
| 1000 | 175/175 | 220 ms | 131 ms | **1.7× slower** |

That is an operator built from `O(n)` stored numbers — but sitting beside a dense `A` that both
routes must multiply by, so the cheap part was never where the cost was.

Contrast it with the table in [Which representation, and why](@ref), where an operator applied
in `O(n)` against an `O(n²)` dense form wins by 1.55–2.73× at the same sizes. **Size alone does
not decide this, and neither does whether the matrix fits.** The question is whether applying
your operator is asymptotically cheaper than multiplying by its dense form. If it is not, the
matrix wins at every size.

The second way to get this wrong is conditioning. A bare `LinearMap` has no structure the
solver can exploit, so it is served by conjugate gradients, which struggles as conditioning
worsens — and this is not a small effect: on the badly conditioned sweep in
[Benchmarks](@ref "Conditioning") the matrix-free backend fails to converge at *every* κ tested,
including mild ones. If your problem is ill-conditioned, a bare map is the wrong shape; give
the solver a structured type with a direct backend instead
([Unmaterialized does not mean solved by CG](@ref)).

Building one takes two functions: how to apply it, and how to apply its transpose. The
transpose is not optional; the solver needs both directions.

```@example linearmaps
using PureOSQP, LinearMaps, LinearAlgebra, Krylov, Random
Random.seed!(4)

n, m = 60, 40
# The constraint: scale each entry by w, take a running total, keep the first m.
# `forward` applies it; `adjoint_` applies its transpose. No m×n array is ever built.
w = 0.5 .+ rand(n)
forward(y, x) = (y .= cumsum(w .* x)[1:m])
function adjoint_(x, y)
    fill!(x, 0.0)
    x[1:m] .= y
    reverse!(x); cumsum!(x, x); reverse!(x)
    x .*= w
    return x
end
A = LinearMap{Float64}((y, x) -> forward(y, x), (x, y) -> adjoint_(x, y), m, n)

# The traits are declared, not inferred -- see below.
P = LinearMap(Diagonal(fill(2.0, n)); issymmetric = true, isposdef = true)
q = randn(n)
l, u = fill(-1.0, m), fill(1.0, m)

sol = PureOSQP.solve(P, q, A, l, u; scaling = 0, linsys = :indirect)
(sol.status, sol.iter, round(sol.obj_val; digits = 6))
```

That call needed three things beyond the map itself. Each will bite you if you skip it, so
here they are with what goes wrong.

**1. `using Krylov`.** An operator has no entries, so none of the usual backends can factor
anything. The only one that works is the matrix-free one, which multiplies instead of
factoring — and it lives in Krylov.jl. Without it loaded you get an error naming the remedy.
You do not have to pass `linsys = :indirect`; the solver finds it on its own. It is written
above only to make the requirement visible.

**2. `scaling = 0`.** By default the solver rescales your problem for numerical health, which
means reading down each column of `A` to find its largest entry. A map has no columns to read.
Passing `scaling = 0` turns that step off. If you forget, `setup` throws and says so — it does
not silently skip the rescaling.

**3. Declaring `issymmetric` and `isposdef` on `P`.** This is the one that surprises people.
Write `LinearMap(Diagonal(fill(2.0, n)))` — obviously a positive-definite matrix — and ask it,
and it says `isposdef == false`. LinearMaps does not inspect what you gave it; it reports only
what you *told* it. So the solver sees an objective not claiming to be convex and refuses it.

That refusal is correct, not a bug: the solver cannot factor an operator to check, so an
unclaimed property is an unknown one. Declare them at construction, as in the example. (Or
build the wrapper yourself with `ProductOperator{T}(map; symmetric, posdef)` if you want to
override what a map claims.)

That third point is also what LinearMaps buys you over writing an operator by hand: those two
declarations travel with the map, so [`PureOSQP.is_convex`](@ref) is answered by reading a flag
instead of factoring a matrix.

**One thing to expect: a map runs without a preconditioner.** A preconditioner is a cheap
approximation of the problem that makes the iteration converge faster, and the one used here is
built from the diagonal of the reduced matrix. A map has no entries, so there is no diagonal to
read, and the solver proceeds without one. Concretely: setup gets *cheaper* (nothing to build)
and each iteration gets **1.36–1.51× dearer**, measured on the same operator written both ways
([Benchmarks](@ref "An operator that is never materialized")).

Usually you just accept that. If the iteration count matters, give your map's type a
`PureOSQP.structural_rows` method — one method, described under
[Structured operators](@ref "2. `structural_rows` — setup stops paying for the zeros"), which
recovers the preconditioner *and* lets you drop `scaling = 0`. Setting `probe = true` is not a
substitute; probing answers the rescaling question, not this one.

## Structured operators the package ships

`Diagonal` and `Bidiagonal` above are LinearAlgebra's. This package ships three more matrix
types of its own, for three shapes that come up constantly and that LinearAlgebra has no type
for.

**Start here: which one, if any, is yours?**

| if your problem is… | use | typical source |
|---|---|---|
| many small independent sub-problems, side by side | [`PureOSQP.BlockDiagonal`](@ref) | one QP per time step, per asset, per scenario — anything that would be separate problems if they did not share a solve |
| mostly independent, but with a *few* rows tying everything together | [`PureOSQP.RowCoupled`](@ref) | box constraints plus a handful of budget or total-mass rows |
| a constraint applied across two dimensions at once | [`PureOSQP.KroneckerOperator`](@ref) | a 2-D grid, an image, space × time — where the constraint is "this in one direction, that in the other" |
| none of these | nothing to do | pass ordinary matrices; the solver is still fast |

If none of the rows fit, you have lost nothing by reading — these are optimizations, not
requirements, and the dense path gives the same answers.

Each one is used the same way: build it, pass it to [`setup`](@ref) exactly where you would
have passed a matrix, and check `backend_name` to confirm it was picked up. You never
configure anything.

### Block-diagonal

**Use it when your problem is really several smaller problems side by side.** Four machines
scheduled independently, twelve months priced independently, a hundred scenarios — anything
where variable 3 never appears in a constraint with variable 40.

The payoff is large and worth understanding, because it is why this type exists. Solving one
`n×n` system costs about `n³`. Solving `K` systems of size `n/K` costs `K(n/K)³ = n³/K²`. At
`K = 10` that is a hundred times less work, and a tenth of the memory. The solver gets that
automatically once it can *see* the blocks — which is what [`PureOSQP.BlockDiagonal`](@ref) is
for. Handed the same numbers as one big dense matrix, it cannot see them, and pays the `n³`.

Store it as a vector of the blocks. `P` and `A` must split at the same places, since a block
of the problem is only independent if both halves agree it is.

```@example blocks
using PureOSQP, LinearAlgebra

blocks_P = [Matrix(Symmetric([2.0 0.3; 0.3 2.0])) for _ in 1:4]
blocks_A = [[1.0 -1.0; 0.5 1.0] for _ in 1:4]
P = PureOSQP.BlockDiagonal(blocks_P)
A = PureOSQP.BlockDiagonal(blocks_A)

n, m = size(P, 1), size(A, 1)
q = collect(range(-1.0, 1.0; length = n))
ws = setup(P, q, A, fill(-1.0, m), fill(1.0, m))
PureOSQP.backend_name(ws.linsys)
```

`backend_name` returned `:block`, so the blocks were found. Here is what that saved, counted in
numbers stored:

```@example blocks
(blocks = PureOSQP.backend_info(ws.linsys).factor_nnz, dense = n * (n + 1) ÷ 2)
```

Four blocks of two variables each store 12 numbers where the dense route stores 36. The gap
widens fast: at 20 blocks of 12 it is 1 560 against 28 920, and the timings are in
[Benchmarks](@ref "Block-diagonal structure").

**If `backend_name` comes back `:cholesky` instead**, the blocks were not used. The usual
reason is that `P` and `A` split at different places, so the problem does not actually
decouple; the answer is still correct, just computed the slow way.

### Kronecker

**Use it when a constraint acts on two dimensions at once.** The clearest case is a 2-D grid:
you want something smoothed along rows *and* along columns. Written out, that constraint matrix
is enormous and almost all zeros. Written as `A₁ ⊗ A₂` — "`A₁` across, `A₂` down" — it is two
small matrices.

The saving in storage is immediate: a `6×6` constraint below is stored as `4 + 9 = 13` numbers
instead of 36, and that ratio grows as the square. The saving in time is larger, because the
backend never builds the big matrix at all.

**This one has conditions, and they are strict.** It is the fussiest type in the package, so
check them before reaching for it. All three are properties of your problem, not settings you
can turn on:

| condition | in plain terms | how to check |
|---|---|---|
| `P` must be `μI` — a single number times the identity | your objective weights every variable equally, or there is no objective at all | `P isa Diagonal && allequal(P.diag)` |
| `ρ` must be one number | every constraint is an inequality — no equalities | you passed no row with `l[i] == u[i]` |
| `scaling = 0` | you turn equilibration off explicitly | pass `scaling = 0` to `setup` |

The second is the one that catches people: **a single equality row disables this backend.** And
a Kronecker *`P`* does not qualify for the first — it must be a multiple of the identity.

If any condition fails the solver quietly uses the dense route instead, so you get the right
answer either way. That is why every example here checks `backend_name`: it is the only way to
tell whether you got what you asked for.

```@example kron
using PureOSQP, LinearAlgebra

A1 = [1.0 0.5; -0.5 1.0]
A2 = [2.0 0.0 1.0; 0.0 1.5 0.0; 1.0 0.0 2.0]
A = PureOSQP.KroneckerOperator(A1, A2)      # 6×6, stored as 4 + 9 entries

n = size(A, 2)
P = Diagonal(fill(2.0, n))                  # μI, as the tier requires
q = collect(range(-1.0, 1.0; length = n))
ws = setup(P, q, A, fill(-1.0, n), fill(1.0, n); scaling = 0)
PureOSQP.backend_name(ws.linsys)
```

Break any one condition and the rung declines — the problem is solved by the dense terminal
instead, more slowly and just as correctly:

```@example kron
equilibrated = setup(P, q, A, fill(-1.0, n), fill(1.0, n))   # scaling left at its default
nonscalar = setup(Diagonal(1.0:n), q, A, fill(-1.0, n), fill(1.0, n); scaling = 0)
(equilibrated = PureOSQP.backend_name(equilibrated.linsys),
 nonscalar = PureOSQP.backend_name(nonscalar.linsys))
```

#### Ill-conditioned Kronecker problems

Giving up equilibration is the price of this tier, and an ill-conditioned problem is exactly
where equilibration earns its keep — so that is where the trade has to be judged. Note that
`κ(A₁ ⊗ A₂) = κ(A₁)·κ(A₂)`, so each factor carries the square root of the figure below.

**The backend stays sound.** Against a dense path given the same `scaling = 0`, so the
comparison is the backends' and nothing else, it matches iteration for iteration and agrees on
the solution up to `κ(A) ≈ 10¹⁶` (`bench/results/kronecker_conditioning.json`):

| κ(A) | kronecker | dense, also unscaled | solutions agree |
|---|---|---|---|
| 1e2 | SOLVED, 175 | SOLVED, 175 | yes |
| 1e8 | SOLVED, 450 | SOLVED, 450 | yes |
| 1e12 | SOLVED, 1100 | SOLVED, 1100 | yes |
| 1e16 | SOLVED, 2525 | SOLVED, 2525 | yes |

Iterations climb steeply with conditioning — 175 to 2525 — because nothing is preconditioning
the problem. That is the cost, and it is not hidden by the structure.

**Whether it still wins depends on size**, because the tier buys `O(n₁n₂(n₁+n₂))` per iteration
against a dense `O(n₁²n₂²)`, and that has to cover the extra iterations. Against a dense path
allowed its equilibration — the choice a caller actually faces — at `κ(A) = 10¹²`:

| n | kronecker (`scaling = 0`) | dense, equilibrated | speedup |
|---|---|---|---|
| 30 | 175 iter, 0.11 ms | 300 iter, 0.19 ms | 1.7× |
| 168 | 350 iter, 0.61 ms | 575 iter, 3.27 ms | 5.4× |
| 480 | 875 iter, 3.98 ms | 500 iter, 27.25 ms | 6.9× |

At `n = 480` the tier takes 1.75× the iterations and still finishes 6.9× sooner. The iteration
counts are noisy in both directions — ADMM's trajectory is sensitive to scaling — so read the
times rather than the ratio of counts.

If your `P` is zero rather than `μI`, equilibration and the structure are compatible in
principle: a Kronecker product's row and column ∞-norms are exactly the Kronecker products of
the factors' norms, so equilibrating each factor would preserve the diagonalization. That
route is not built.

### Low-rank coupling

**Use it when almost every constraint touches one variable, and only a handful touch many.**
This is extremely common and easy to miss. A portfolio with a bound on each holding plus one
row saying "the weights sum to 1". A schedule with a limit per machine plus two rows for total
capacity. A design with a box on each parameter plus a budget.

Written as an ordinary matrix, those few dense rows make the whole thing look dense, and the
solver pays as if every constraint coupled everything. [`PureOSQP.RowCoupled`](@ref) separates
the two kinds so it can charge you only for the coupling rows you actually have.

It takes three arguments, in this order:

1. `coupling` — the few dense rows, as a `k×n` matrix. These are the rows that touch many
   variables.
2. `weights` — one number per single-entry row.
3. `cols` — which variable each of those rows refers to.

So `RowCoupled(C, ones(n), 1:n)` means "these `k` dense rows, then a plain bound on each of the
`n` variables".

```@example rowcoupled
using PureOSQP, LinearAlgebra

n = 24
coupling = reshape(collect(range(0.1, 0.8; length = 2n)), 2, n)   # two dense rows
A = PureOSQP.RowCoupled(coupling, ones(n), collect(1:n))          # then a bound per variable
P = Diagonal(fill(1.5, n))
q = collect(range(-1.0, 1.0; length = n))
m = size(A, 1)
ws = setup(P, q, A, fill(-1.0, m), fill(1.0, m))
PureOSQP.backend_name(ws.linsys)
```

`:lowrank` means it worked. The cost is `O(nk)` instead of `O(n²)`, so it wins by more the
fewer coupling rows you have: at one coupling row in 2000 variables it is
[**923× faster**](@ref "Low-rank structure") than the dense route.

**One condition:** the coupling rows have to be a small fraction of the variables — the backend
declines once `10k > n`. Two coupling rows therefore need at least 20 variables, which is why
`n = 24` above. Below that threshold the correction costs more than the dense solve it would
replace, so declining is the right answer. `P` must also be `Diagonal`.

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
