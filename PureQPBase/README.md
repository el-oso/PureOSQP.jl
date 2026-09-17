# PureQPBase.jl

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://el-oso.github.io/PureQP.jl/dev/)

The algorithm-independent core of [PureQP.jl](https://github.com/el-oso/PureQP.jl): the
problem representation, every linear-system backend and the ladder that picks one,
equilibration, the termination, polishing and derivative kernels, and the contracts a solver
implements.

**It defines no algorithm and solves nothing on its own.** `setup` and `solve` take one as
their sixth positional argument, and this package supplies no instance to pass:
[PureOSQP.jl](https://github.com/el-oso/PureQP.jl/tree/main/PureOSQP) supplies
`OperatorSplitting`, [PureIPM.jl](https://github.com/el-oso/PureQP.jl/tree/main/PureIPM)
supplies `InteriorPoint`. Install this package directly only to write a third.

```julia
using PureQPBase, PureOSQP     # or PureIPM; either re-exports this package
```

## What it holds

**The problem.** `Problem` keeps `P` and `A` by reference and never mutates them.
Equilibration is stored as factors and applied lazily, so every product calls `mul!` on the
matrix you passed, whatever its type.

**The backends.** A reduced Cholesky, a full quasi-definite KKT factorization, and — through
package extensions — sparse factorizations (CHOLMOD and LDLFactorizations), banded, block,
Kronecker, diagonal-plus-low-rank, and a matrix-free conjugate-gradient solve over
[Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl).

**The selection ladder.** Which backend serves a problem is decided once, from the declared
types of `P` and `A` and from the sparsity pattern, never from the values — so a problem's
backend does not change when its numbers do, and `setup` can factor once with the values the
solve will use. Each algorithm descends its own ladder over shared rungs;
[Backend selection](https://el-oso.github.io/PureQP.jl/dev/selection) is the map.

**The contracts.** `LinearSystem`, `Preconditioner`, `QPAlgorithm` and `QPWorkspace` are
declared with [TypeContracts.jl](https://github.com/el-oso/TypeContracts.jl), so a type that
claims one and does not implement it is rejected rather than failing at a call site.

`PureQPBase.conforms(alg; eps, slow_iters)` goes further, and is what a third algorithm
should run: a contract says which methods exist, not what their values mean, and this asserts
the guarantees `Solution` and `Status` carry — that stopping early is never reported as
solved, that a run with no point fills `x` and `y` with `NaN`, that the objective is the
objective, that the timings add up. It lives in a `Test` extension, so it costs a caller
nothing.

## Dependencies

`LinearAlgebra` and `TypeContracts`. Everything else — SparseArrays, BandedMatrices,
LDLFactorizations, Krylov, LinearMaps, SciMLOperators, GPUArraysCore, ChainRulesCore,
MathOptInterface — is a package extension, loaded only if you load it.

## License

**MIT.** This package is not a derivative of OSQP. The equilibration follows Ruiz 2001 and
the infeasibility certificates follow Banjac et al. 2019, both written from the papers. See
[Attribution](https://el-oso.github.io/PureQP.jl/dev/attribution).
