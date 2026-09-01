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

## Matrix representations

The applications above build every matrix dense. The solver does not require that: `P` and
`A` are held by reference and reached through `mul!` and four column traversals, so any
`AbstractMatrix` will do. The problems below are all the same QP, written five ways.

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

Implementing the [operator protocol](@ref "What to implement, in order") by hand is only worth
it for an operator you own. For one already expressed as a `LinearMaps.LinearMap`, loading
LinearMaps is enough: `setup` and `solve` accept a map wherever they accept a matrix, wrapping
it in a [`PureOSQP.ProductOperator`](@ref) for you.

A `LinearMap` is not an `AbstractMatrix`, so without the extension loaded it reaches neither
entry point. With it, matrices and maps mix freely in one call.

```@example linearmaps
using PureOSQP, LinearMaps, LinearAlgebra, Krylov, Random
Random.seed!(4)

n, m = 60, 40
# A constraint that is a product rather than a table: scale, take a running sum, keep the
# first m entries. Nothing here is ever assembled into an m×n array.
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

Two things that call needs, and both are properties of operators rather than of LinearMaps.
`scaling = 0`, because equilibration reads columns and a map has none to read — override
`PureOSQP.structural_rows` for the map's type if you want equilibration back. And Krylov.jl
loaded, since the matrix-free backend is the only one that can serve an operator; `:auto`
reaches it on its own, and naming it makes the requirement explicit.

What loading LinearMaps buys over hand-wrapping is the pair of declarations the wrapper cannot
compute: `issymmetric` and `isposdef` travel with the map, and
[`PureOSQP.is_convex`](@ref) is then answered by reading them rather than by factoring.

**They are declarations, not deductions.** `LinearMap(Diagonal(fill(2.0, n)))` reports
`isposdef == false` — LinearMaps records what its author asserted and infers nothing from the
wrapped matrix — and [`setup`](@ref) refuses it as a non-convex objective. That refusal is
correct: an operator that has not claimed positive-definiteness has not established it. State
the traits at construction as above, or override the map with
`ProductOperator{T}(map; symmetric, posdef)`.

**A map runs unpreconditioned.** The matrix-free backend preconditions with the reduced
diagonal, and a `LinearMap` has no entries to supply it, so `prec` stays at ones. Measured on
the same operator written both ways, that makes setup cheaper — there is no preconditioner to
build — and each iteration 1.36–1.52× dearer
([Benchmarks](@ref "An operator that is never materialized")). `probe = true` does not change
it: probing answers equilibration's column norms, not this seam. Give the map's type a
`PureOSQP.structural_rows` method to get a preconditioner, which is the same override that
restores equilibration and lets you drop `scaling = 0`.

## Structured operators the package ships

Three representations come with a backend that never forms the reduced matrix. Each is an
ordinary argument to [`setup`](@ref) — the structure is declared by the type, and selection
finds it by dispatch.

### Block-diagonal

[`PureOSQP.BlockDiagonal`](@ref) is a diagonal run of blocks, stored as the blocks. When `P`
and `A` split their columns at the same places, the reduced matrix decouples into independent
systems that are factored one at a time.

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

The storage that buys, against forming the `n×n` reduced matrix:

```@example blocks
(blocks = PureOSQP.backend_info(ws.linsys).factor_nnz, dense = n * (n + 1) ÷ 2)
```

### Kronecker

[`PureOSQP.KroneckerOperator`](@ref) is `A₁ ⊗ A₂` held as its two factors, and the backend
solves through their eigenbases. It has the narrowest acceptance region in the package, and
all three conditions are structural rather than tunable:

| condition | why |
|---|---|
| `P` is a *scalar* multiple of `I` | `c·D·P·D` and `ρ·Ãᵀ diag(ρ) Ã` are simultaneously diagonalizable only then. A Kronecker `P` does **not** qualify. |
| `ρ` is one number | true automatically when every constraint is an inequality; one equality row gives `ρ` a second value |
| `scaling = 0` | `c·μ·D²` is diagonal but not scalar, so any equilibration breaks the diagonalization |

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

[`PureOSQP.RowCoupled`](@ref) is a few dense rows above rows holding one entry each, which
with a `Diagonal` `P` makes the reduced matrix a diagonal plus a rank-`k` correction, solved
by Woodbury.

```@example rowcoupled
using PureOSQP, LinearAlgebra

# The rung accepts while `10k <= n`, so two coupling rows need at least twenty variables:
# below that the correction costs more than the dense solve it replaces.
n = 24
coupling = reshape(collect(range(0.1, 0.8; length = 2n)), 2, n)   # two dense rows
A = PureOSQP.RowCoupled(coupling, ones(n), collect(1:n))          # then a bound per variable
P = Diagonal(fill(1.5, n))
q = collect(range(-1.0, 1.0; length = n))
m = size(A, 1)
ws = setup(P, q, A, fill(-1.0, m), fill(1.0, m))
PureOSQP.backend_name(ws.linsys)
```

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
