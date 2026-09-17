# PureQPBase.jl

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://el-oso.github.io/PureQP.jl/dev/)

The shared core of [PureQP.jl](https://github.com/el-oso/PureQP.jl). It holds the problem
type, every linear-system backend, the code that picks one, equilibration, the termination
tests, the polishing and derivative kernels, and the contracts a solver implements.

**It has no algorithm, so it solves nothing on its own.** `setup` and `solve` take an
algorithm as their sixth argument, and this package has none to give you.
[PureOSQP.jl](https://github.com/el-oso/PureQP.jl/tree/main/PureOSQP) supplies
`OperatorSplitting`.
[PureIPM.jl](https://github.com/el-oso/PureQP.jl/tree/main/PureIPM) supplies `InteriorPoint`.
Install either one and you get this package with it:

```julia
using PureOSQP     # or PureIPM; both re-export this package
```

Install this package on its own only if you want to write a third algorithm.

## What it holds

**The problem.** `Problem` holds `P` and `A` by reference and never changes them. It keeps the
equilibration as factors and applies them as it goes, so every product calls `mul!` on the
matrix you passed, whatever type that is.

**The backends.** A reduced Cholesky and a full KKT factorization. Package extensions add
more: sparse factorizations through CHOLMOD and LDLFactorizations, banded, block, Kronecker,
diagonal plus low rank, and a matrix-free solve with conjugate gradients over
[Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl).

**The choice between them.** The package picks a backend once, in `setup`. It reads the types
of `P` and `A` and the sparsity pattern. It never reads the values, so the backend does not
change when your numbers do, and `setup` factors once with the values the solve uses. Each
algorithm tries the candidates in its own order — see
[How a backend is chosen](https://el-oso.github.io/PureQP.jl/dev/selection).

**The contracts.** `LinearSystem`, `Preconditioner`, `QPAlgorithm` and `QPWorkspace` use
[TypeContracts.jl](https://github.com/el-oso/TypeContracts.jl). Claim one and leave a method
out, and the package refuses to precompile and names what is missing.

`PureQPBase.conforms(alg; eps, slow_iters)` goes further, and a third algorithm should run it.
A contract says which methods exist. It cannot say what their values mean. `conforms` tests
that: a run that stops early never reports `SOLVED`, a run with no answer fills `x` and `y`
with `NaN`, the objective it reports is the objective, and the times it reports are real. It
lives in a `Test` extension, so it costs a caller nothing.

## Dependencies

`LinearAlgebra` and `TypeContracts`, and nothing else. SparseArrays, BandedMatrices,
LDLFactorizations, Krylov, LinearMaps, SciMLOperators, GPUArraysCore, ChainRulesCore and
MathOptInterface are all package extensions. Each loads only if you load it.

## License

**MIT.** This package is not derived from OSQP. The equilibration follows Ruiz 2001 and the
infeasibility certificates follow Banjac et al. 2019. We wrote both from the papers. See
[Attribution](https://el-oso.github.io/PureQP.jl/dev/attribution).
