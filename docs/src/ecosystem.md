# Working with other packages

PureOSQP has **no required dependencies**. Capabilities appear as you load relevant packages via Julia's extension mechanism.

Two mechanisms are at work:

**Extensions** are code that loads only when you load a trigger package. There are seven, listed below.

**Genericity** allows the solver to handle any numeric or matrix type that behaves correctly. Precision types are handled this way, without needing extensions.

## Number types

The element type of your arrays determines the solver's precision. All types below were tested and show identical convergence behavior.

| type | package | digits | cost vs `Float64` | note |
|---|---|---|---|---|
| `Float32` | — | ~7 | — | works; halves memory |
| `Float64` | — | ~16 | 1× | the default |
| `Double64` | [DoubleFloats.jl](https://github.com/JuliaMath/DoubleFloats.jl) | ~32 | ~22× | best for extra precision |
| `Float128` | [Quadmath.jl](https://github.com/JuliaMath/Quadmath.jl) | ~34 | ~85× | software-emulated |
| `BigFloat` | stdlib | tunable | ~264× | arbitrary precision |

`Double64` is roughly four times faster than `Float128` at the same precision.

```julia
using PureOSQP, DoubleFloats
sol = solve(Double64.(P), Double64.(q), Double64.(A), Double64.(l), Double64.(u))
```

**`Rational` does not work**, and the reason is inherent rather than a gap here: exact rational
arithmetic grows denominators without bound, and `Rational{Int}` overflows within a few
iterations. `Rational{BigInt}` avoids the overflow at a cost that makes it impractical.

**More precision does not make an ill-conditioned problem converge faster.** This is worth
saying plainly because it is a natural thing to reach for. The iteration count is a property of
the problem's geometry, not of the arithmetic: at every precision above, the same problem took
the same 250 iterations. Extra precision lowers the floor on achievable accuracy; it does not
speed the descent toward it. If a badly conditioned problem is stopping at `max_iter`, see
[Conditioning](@ref) — the lever there is structure, not bits.

## Matrix and operator types

| package | what it buys | how |
|---|---|---|
| `LinearAlgebra` (stdlib) | `Diagonal`, `Bidiagonal`, `SymTridiagonal`, `Tridiagonal`, `Symmetric`, views | genericity, plus dedicated backends |
| [SparseArrays](https://github.com/JuliaSparse/SparseArrays.jl) (stdlib) | sparse `P` and `A`, walked by stored entry | extension |
| [BandedMatrices.jl](https://github.com/JuliaLinearAlgebra/BandedMatrices.jl) | bandwidth ≥ 2 | extension |
| [LinearMaps.jl](https://github.com/JuliaLinearAlgebra/LinearMaps.jl) | an operator you can apply but not store | extension |
| [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl) | the matrix-free backend, needed for any operator | extension |
| [LDLFactorizations.jl](https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl) | a pure-Julia sparse `LDLᵀ` instead of SuiteSparse | extension |
| [GPUArraysCore.jl](https://github.com/JuliaGPU/GPUArrays.jl) | GPU arrays, matrix-free only | extension |

[Which representation, and why](@ref) helps you choose.

Two key extensions:
- **LDLFactorizations**: Provides a pure-Julia sparse `LDLᵀ`, making the package compatible with `juliac --trim`.
- **Krylov**: Required for any operator that cannot be materialized.

## Modelling and interfaces

[MathOptInterface.jl](https://github.com/jump-dev/MathOptInterface.jl) has an extension, so
PureOSQP is usable as a JuMP solver:

```julia
using JuMP, PureOSQP, MathOptInterface
model = Model(PureOSQP.Optimizer)
```

The wrapper passes `MOI.Test`, which is a far more thorough conformance suite than anything
hand-written.

## Differentiating a solve

There are two ways to differentiate:

1. **Built-in:** [`adjoint_derivative`](@ref) and [`forward_derivative`](@ref) differentiate the *solution* by implicitly differentiating the KKT conditions. This is efficient and works in one solve.
2. **Through iterations:** Using `ForwardDiff.Dual` numbers differentiates the entire optimization loop. This is better for sensitivity analysis but is much more expensive.


```julia
using PureOSQP, ForwardDiff
ForwardDiff.derivative(t -> solve(P, q .* t, A, l, u).obj_val, 1.3)
```

Checked against central differences on a small QP: agreement to `5e-11`.

**Prefer the built-in one for anything real.** Forward mode costs one solve per parameter, and
`P` alone has `n²` of them; and it differentiates `x_N(θ)`, the iterate you stopped at, rather
than `x*(θ)`, the solution. They converge to the same thing, but the implicit form gets there
in one solve and does not depend on the path taken. ForwardDiff's value here is as an
independent check on the implicit derivative — exact, with no finite-difference step to choose
— which is what the package's own tests use it for.
