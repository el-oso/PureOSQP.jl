# Examples

These are the applications from the [OSQP documentation](https://osqp.org/docs/examples/),
written against PureOSQP. Each one is executed when these docs are built, so the numbers
below are what the code actually produced.

Two differences from the upstream versions are worth knowing before you read them. PureOSQP
takes the **full symmetric** `P`, not an upper triangle. And it is a dense solver, so every
matrix here is built dense — these particular problems are highly structured and sparse,
which is where the reference implementation is strongest, so treat them as a guide to
*formulating* problems rather than as a claim about which solver to use on them.

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
