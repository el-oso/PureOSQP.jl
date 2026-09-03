# Operators from functions

A [`LinearMap`](https://github.com/JuliaLinearAlgebra/LinearMaps.jl) is a matrix you never
store. Instead of entries you supply two functions — how to multiply by it, and how to
multiply by its transpose — and the solver accepts it anywhere it accepts a matrix.

Three worked problems follow. Each one has a constraint matrix that would be large, dense, or
simply never assembled, and each is solved without building it.

All three need the same three things, covered in
[An operator from LinearMaps.jl](@ref): load `Krylov`, pass `scaling = 0`, and declare
`issymmetric` and `isposdef` on `P`. Every example below is checked against the same problem
with every matrix materialized; the agreement is reported with each one.

## 1. Fitting to sensor readings

You have a signal of 240 points but only measured every fifth one. You want the closest signal
to a target that still agrees with all 48 readings to within 0.05.

The constraint matrix picks out the measured entries. Written down it is a 48×240 array of
almost entirely zeros. As a map it is a single indexing operation, and its transpose scatters
the values back:

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

Agrees with the materialized problem to `1.8e-10`.

This is the easiest case to reach for: a measurement operator that selects, masks or reorders
is pure bookkeeping, and storing it as a matrix buys nothing.

## 2. A constraint on a 2-D grid

A 14×14 image, with a limit on how fast it may change from one pixel to the next — in both
directions at once. The image is 196 variables, so a constraint matrix would be 392×196.

The trick is that a 2-D operation built from a 1-D one along each axis is a **Kronecker
product**, and `kron` composes maps without forming anything. One small 14×14 difference
operator serves both axes:

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

Agrees with the materialized problem to `3.5e-12`.

Only the 14×14 factor is ever stored. The saving grows fast: the same construction on a
256×256 image gives a constraint matrix with `2.1e10` entries, built from a 256×256 one.

## 3. Reusing a model you already have

Sometimes the operator is code somebody already wrote — a filter, a simulator, a forward
model. You can call it, but nobody ever assembled it, and doing so would mean one call per
column.

Here it is a first-order recursion, `yₖ = 0.6·yₖ₋₁ + xₖ`, the discrete form of a system that
carries part of its state forward. Its transpose is the same recursion run backwards:

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

Agrees with the materialized problem to `1.2e-9`.

As a matrix this is dense and lower triangular — 11 250 nonzeros here, and `n²/2` in general.
As a map it is two loops over `n` values.

## Choosing between a map and a matrix

A map wins when applying it is **asymptotically cheaper** than multiplying by its dense form.
All three examples above qualify: `O(n)` work against an `O(n²)` matrix.

It is the wrong choice when the product costs the same either way — a map has no entries to
factor, so it is solved by conjugate gradients, which pays every iteration for a factorization
a matrix pays for once. The measured comparison is in
[When it is the wrong tool](@ref), and the conditioning limit — where a bare map does not
converge at all — is in [Unmaterialized does not mean solved by CG](@ref).
