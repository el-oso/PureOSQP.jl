# Working with other packages

A solver package needs only `LinearAlgebra`,
[TypeContracts.jl](https://github.com/el-oso/TypeContracts.jl) and PureQPBase.jl, and
PureQPBase.jl needs the first two and nothing else. **No numerical library is required.** You
gain capabilities as you load packages, through Julia's extension mechanism.

Two things are at work here:

**Extensions** are code that loads only when you load a trigger package. The ones below belong
to PureQPBase, because they extend the backends rather than any algorithm. PureOSQP adds two
of its own, for COSMOAccelerators and for its MathOptInterface optimizer, and PureIPM adds one,
for its optimizer. PureDAQP adds none: it has no backend to extend, and no optimizer yet.

**Generic code** lets the solver take any numeric or matrix type that behaves correctly.
Precision types work this way, with no extension.

## Number types

The element type of your arrays sets the solver's precision. We tested every type below, and
they all converge the same way.

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

**`Rational` does not work**, and that is inherent, not a gap here. Exact rational arithmetic
grows denominators without bound, so `Rational{Int}` overflows within a few iterations.
`Rational{BigInt}` avoids the overflow, at a cost that makes it impractical.

**More precision does not make an ill-conditioned problem converge faster.** It is worth saying
plainly, because reaching for more precision is a natural move. The iteration count comes from
the problem's geometry, not from the arithmetic. At every precision above, the same problem took
the same 250 iterations. Extra precision lowers the floor on the accuracy you can reach. It does
not get you there sooner. If a badly conditioned problem stops at `max_iter`, see
[Conditioning](@ref). What helps there is structure, not bits.

## Matrix and operator types

| package | what it buys | how |
|---|---|---|
| `LinearAlgebra` (stdlib) | `Diagonal`, `Bidiagonal`, `SymTridiagonal`, `Tridiagonal`, `Symmetric`, views | genericity, plus dedicated backends |
| [SparseArrays](https://github.com/JuliaSparse/SparseArrays.jl) (stdlib) | sparse `P` and `A`, walked by stored entry | extension |
| [BandedMatrices.jl](https://github.com/JuliaLinearAlgebra/BandedMatrices.jl) | bandwidth ≥ 2 | extension |
| [LinearMaps.jl](https://github.com/JuliaLinearAlgebra/LinearMaps.jl) | an operator you can apply but not store | extension |
| [SciMLOperators.jl](https://github.com/SciML/SciMLOperators.jl) | the same, and composed operators that apply without allocating | extension |
| [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl) | the matrix-free backend, needed for any operator | extension |
| [LDLFactorizations.jl](https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl) | a pure-Julia sparse `LDLᵀ` instead of SuiteSparse | extension |

[Which type to use](@ref) helps you choose.

Two extensions matter most:
- **LDLFactorizations** gives a pure-Julia sparse `LDLᵀ`, which keeps the package compatible
  with `juliac --trim`.
- **Krylov** is required for any operator the solver cannot materialize.

## Modelling and interfaces

[MathOptInterface.jl](https://github.com/jump-dev/MathOptInterface.jl) has an extension, so you
can use either solver from JuMP. Each package supplies the optimizer for its own algorithm:

```julia
using JuMP, PureOSQP, MathOptInterface
model = Model(PureOSQP.Optimizer)       # OperatorSplitting

using PureIPM
model = Model(PureOSQP.Optimizer)        # InteriorPoint
```

One wrapper serves both. It lives in PureQPBase and carries the algorithm, so the two behave the
same apart from the method they run and the parameters they take.

The wrapper passes `MOI.Test` under both optimizers. That suite is far more thorough than
anything we could write by hand, and `PureOSQP/test/moi_tests.jl` runs it against both. Three
attributes are left out: `ConstraintBasisStatus`, `VariableBasisStatus` and `ObjectiveBound`.
Neither algorithm produces a basis or a bound.

You pass settings by name, as in `set_attribute(model, "linsys", :kkt)`. A name is either an
[`Options`](@ref) field or a parameter of that optimizer's algorithm. A parameter of the other
algorithm is not accepted. A setting that takes a symbol also takes its name as a string. A bad
value throws when you set it. Read a setting you have not set and you get its default for that
algorithm. `MOI.TimeLimitSec` sets `time_limit`, which limits the iterations of either
algorithm. It does not count setup or polishing.

## Differentiating a solve

There are two ways to differentiate:

1. **Built in.** [`adjoint_derivative`](@ref) and [`forward_derivative`](@ref) differentiate the
   *solution*, by differentiating the KKT conditions implicitly. This is cheap: one solve.
2. **Through the iterations.** `ForwardDiff.Dual` numbers differentiate the whole optimization
   loop. This suits sensitivity analysis, and it costs much more.


```julia
using PureOSQP, ForwardDiff
ForwardDiff.derivative(t -> solve(P, q .* t, A, l, u).obj_val, 1.3)
```

Checked against central differences on a small QP, the two agree to `5e-11`.

**Use the built-in one for real work.** Forward mode costs one solve per parameter, and `P`
alone has `n²` of them. It also differentiates `x_N(θ)`, the iterate you stopped at, rather than
`x*(θ)`, the solution. The two converge to the same thing, but the implicit form gets there in
one solve and does not depend on the path. What ForwardDiff is worth here is an independent
check on the implicit derivative: exact, with no finite-difference step to choose. That is what
the package's own tests use it for.
