# Operators from functions

A [`LinearMap`](https://github.com/JuliaLinearAlgebra/LinearMaps.jl) is a matrix you never
store. Instead of entries you supply two functions — how to multiply by it, and how to
multiply by its transpose — and the solver accepts it anywhere it accepts a matrix.

Three worked problems follow. Each one has a constraint matrix that would be large, dense, or
simply never assembled, and each is solved without building it.

All three need the same three things, covered in
[An operator from LinearMaps.jl](@ref): load `Krylov`, pass `scaling = 0`, and declare
`issymmetric` and `isposdef` on `P`. Each example states the problem, draws the operator, solves
with the map, and then solves the same problem with every matrix written out and prints the
difference between the two answers.

## 1. Fitting to sensor readings

You have a signal of 240 points but only measured every fifth one. You want the closest signal
to a target that still agrees with all 48 readings to within 0.05.

```math
\begin{array}{ll}
  \mbox{minimize}   & \|x\|_2^2 + q^T x \\
  \mbox{subject to} & b - 0.05 \le A x \le b + 0.05
\end{array}
```

Here `q = -2t + noise` for the target `t`, so the objective is `‖x - x₀‖²` up to a constant,
with `x₀ = -q/2` the noisy target; `b` holds the 48 readings. The constraint matrix picks out
the measured entries, and its transpose puts them back where they came from:

```math
(Ax)_j = x_{5j-4},
\qquad
(A^{\top}y)_i = \begin{cases} y_j & i = 5j-4 \\ 0 & \text{otherwise} \end{cases}
```

Written down `A` is a 48×240 array with a single 1 in each row. As a program it is one
indexing step in each direction, and the index range is all that is stored:

```math
x \in \mathbb{R}^{240}
\;\xrightarrow{\;\;[\,1{:}5{:}240\,]\;\;}\;
Ax \in \mathbb{R}^{48}
\qquad\qquad
y \in \mathbb{R}^{48}
\;\xrightarrow{\;\;\text{scatter to } 1{:}5{:}240\;\;}\;
A^{\top}y \in \mathbb{R}^{240}
```

```math
A \;=\; \underbrace{I_{240}[\,1{:}5{:}240,\;:\,]}_{48 \times 240,\ \text{48 ones}}
\qquad \text{stored: the range } 1{:}5{:}240 \text{, and nothing else}
```

```@example lm_sensors
using PureOSQP, LinearMaps, LinearAlgebra, Krylov, Random
Random.seed!(1)

n, idx = 240, 1:5:240
m = length(idx)

A = LinearMap{Float64}(
    (y, x) -> (@views y .= x[idx]),                      # keep the measured entries
    (x, y) -> (fill!(x, 0.0); @views x[idx] .= y; x),    # scatter them back
    m, n,
)

truth = [sin(2pi * k / n) for k in 1:n]
b = truth[idx]                                           # what the sensors read

P = LinearMap(Diagonal(fill(2.0, n)); issymmetric = true, isposdef = true)
q = -2 .* truth .+ 0.05 .* randn(n)

sol = PureOSQP.solve(P, q, A, b .- 0.05, b .+ 0.05; scaling = 0, eps_abs = 1e-9, eps_rel = 1e-9)
(sol.status, sol.iter, round(sol.obj_val; digits = 8))
```

The same problem with `A` written out as its 48×240 array, and `P` as a `Diagonal`:

```@example lm_sensors
Ad = Matrix(1.0I, n, n)[idx, :]
dense = PureOSQP.solve(Diagonal(fill(2.0, n)), q, Ad, b .- 0.05, b .+ 0.05; eps_abs = 1e-9, eps_rel = 1e-9)
(dense.status, maximum(abs, sol.x .- dense.x))
```

This is the easiest case to reach for: a measurement operator that selects, masks or reorders
is pure bookkeeping, and storing it as a matrix buys nothing.

## 2. A constraint on a 2-D grid

A 14×14 image, with a limit on how fast it may change from one pixel to the next — in both
directions at once. The image is 196 variables, so a constraint matrix would be 392×196.

With `X` the image and `x = vec(X)` its columns stacked into one vector, the problem is to
stay close to a target image `X₀` while keeping every neighbor difference within 0.25:

```math
\begin{array}{ll}
  \mbox{minimize}   & \|x - \mathrm{vec}(X_0)\|_2^2 \\
  \mbox{subject to} & -0.25 \le (I \otimes D)\,x \le 0.25 \\
                    & -0.25 \le (D \otimes I)\,x \le 0.25
\end{array}
```

`D` is the 14×14 difference operator along one axis, lower bidiagonal with 1 on the diagonal
and −1 below it:

```math
(Dv)_1 = v_1, \qquad (Dv)_i = v_i - v_{i-1} \quad (i = 2, \dots, 14)
```

The trick is that a 2-D operation built from a 1-D one along each axis is a **Kronecker
product**: `I ⊗ D` applies `D` down every column of `X`, and `D ⊗ I` applies it along every
row.

```math
X \in \mathbb{R}^{14 \times 14}
\;\xrightarrow{\;\;D \text{ down each column}\;\;}\;
DX = \mathrm{unvec}\big((I \otimes D)\,x\big)
\qquad\qquad
X
\;\xrightarrow{\;\;D \text{ along each row}\;\;}\;
XD^{\top} = \mathrm{unvec}\big((D \otimes I)\,x\big)
```

`kron` composes maps without forming anything, so the constraint matrix is the two blocks
stacked, and only the 14×14 factors exist:

```math
A \;=\; \begin{pmatrix} I_{14} \otimes D \\ D \otimes I_{14} \end{pmatrix}
\qquad 392 \times 196,
\qquad \text{stored: } D \text{ and } I_{14} \text{, both } 14 \times 14
```

One small difference operator serves both axes:

```@example lm_grid
using PureOSQP, LinearMaps, LinearAlgebra, Krylov, Random
Random.seed!(2)

k = 14
n = k * k

# Difference between neighbours, along one axis. Both directions supplied.
D = LinearMap{Float64}(
    (y, x) -> (y[1] = x[1]; @views y[2:k] .= x[2:k] .- x[1:(k - 1)]; y),
    (x, y) -> (@views x[1:(k - 1)] .= y[1:(k - 1)] .- y[2:k]; x[k] = y[k]; x),
    k, k,
)
Ik = LinearMap(Matrix(1.0I, k, k))

A = [kron(Ik, D); kron(D, Ik)]     # down the columns, then across the rows

img = [exp(-((i - 7)^2 + (j - 7)^2) / 18) for i in 1:k, j in 1:k]
P = LinearMap(Diagonal(fill(2.0, n)); issymmetric = true, isposdef = true)
q = -2 .* vec(img)

sol = PureOSQP.solve(
    P, q, A, fill(-0.25, 2n), fill(0.25, 2n);
    scaling = 0, eps_abs = 1e-9, eps_rel = 1e-9,
)
(size(A), sol.status, sol.iter, round(sol.obj_val; digits = 8))
```

The same problem with `D` written out and both Kronecker products formed as 196-column arrays:

```@example lm_grid
Dd = Bidiagonal(ones(k), -ones(k - 1), :L)
Ad = [kron(I(k), Dd); kron(Dd, I(k))]
dense = PureOSQP.solve(
    Diagonal(fill(2.0, n)), q, Ad, fill(-0.25, 2n), fill(0.25, 2n);
    eps_abs = 1e-9, eps_rel = 1e-9,
)
(size(Ad), dense.status, maximum(abs, sol.x .- dense.x))
```

Only the 14×14 factor is ever stored. The saving grows fast: the same construction on a
256×256 image gives a constraint matrix with `2.1e10` entries, built from a 256×256 one.

## 3. Reusing a model you already have

Sometimes the operator is code somebody already wrote — a filter, a simulator, a forward
model. You can call it, but nobody ever assembled it, and doing so would mean one call per
column.

Here it is a first-order recursion, `yₖ = 0.6·yₖ₋₁ + xₖ`, the discrete form of a system that
carries part of its state forward. The problem is to stay close to a random target `x₀` while
keeping the system's output within ±1.5 at every step:

```math
\begin{array}{ll}
  \mbox{minimize}   & \|x - x_0\|_2^2 \\
  \mbox{subject to} & -1.5 \le A x \le 1.5
\end{array}
```

Unrolling the recursion from `y₀ = 0` gives every output as a decaying sum of the inputs before
it, which is what `A` is as a matrix — lower triangular, with `a = 0.6`:

```math
y_k = \sum_{j \le k} a^{\,k-j} x_j
\qquad\Longrightarrow\qquad
A = \begin{pmatrix} 1 & & & \\ a & 1 & & \\ a^2 & a & 1 & \\ \vdots & \ddots & \ddots & \ddots \end{pmatrix}
\qquad \text{stored: } a \text{, and nothing else}
```

Its transpose is the same recursion run backwards, so both directions are one loop:

```math
x
\;\xrightarrow{\;\;y_k = a\,y_{k-1} + x_k,\ \ k = 1, \dots, n\;\;}\;
Ax
\qquad\qquad
y
\;\xrightarrow{\;\;z_k = a\,z_{k+1} + y_k,\ \ k = n, \dots, 1\;\;}\;
A^{\top}y
```

```@example lm_model
using PureOSQP, LinearMaps, LinearAlgebra, Krylov, Random
Random.seed!(3)

n, a = 150, 0.6

function forward!(y, x)
    acc = 0.0
    for i in eachindex(x)
        acc = a * acc + x[i]
        y[i] = acc
    end
    return y
end

function adjoint!(x, y)                 # the same recursion, backwards
    acc = 0.0
    for i in reverse(eachindex(y))
        acc = a * acc + y[i]
        x[i] = acc
    end
    return x
end

A = LinearMap{Float64}(forward!, adjoint!, n, n)

P = LinearMap(Diagonal(fill(2.0, n)); issymmetric = true, isposdef = true)
q = -2 .* randn(n)

sol = PureOSQP.solve(
    P, q, A, fill(-1.5, n), fill(1.5, n);
    scaling = 0, eps_abs = 1e-9, eps_rel = 1e-9,
)
(sol.status, sol.iter, round(sol.obj_val; digits = 8))
```

The same problem with `A` written out from its closed form, `Aᵢⱼ = aⁱ⁻ʲ` for `i ≥ j`:

```@example lm_model
Ad = [i >= j ? a^(i - j) : 0.0 for i in 1:n, j in 1:n]
dense = PureOSQP.solve(
    Diagonal(fill(2.0, n)), q, Ad, fill(-1.5, n), fill(1.5, n);
    eps_abs = 1e-9, eps_rel = 1e-9,
)
(dense.status, maximum(abs, sol.x .- dense.x))
```

As a matrix this is dense and lower triangular — 11 325 nonzeros here, and `n(n+1)/2` in
general. As a map it is two loops over `n` values.

## Choosing between a map and a matrix

A map wins when applying it is **asymptotically cheaper** than multiplying by its dense form.
All three examples above qualify: `O(n)` work against an `O(n²)` matrix.

It is the wrong choice when the product costs the same either way — a map has no entries to
factor, so it is solved by conjugate gradients, which pays every iteration for a factorization
a matrix pays for once. The measured comparison is in
[When it is the wrong tool](@ref), and the conditioning limit — where a bare map does not
converge at all — is in [Unmaterialized does not mean solved by CG](@ref).
