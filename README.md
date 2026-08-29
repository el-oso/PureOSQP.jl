# PureOSQP.jl

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://el-oso.github.io/PureOSQP.jl/dev/)
[![Build Status](https://github.com/el-oso/PureOSQP.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/el-oso/PureOSQP.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://coveralls.io/repos/github/el-oso/PureOSQP.jl/badge.svg?branch=main)](https://coveralls.io/github/el-oso/PureOSQP.jl?branch=main)

A pure-Julia implementation of the [OSQP](https://osqp.org) operator-splitting solver for
convex quadratic programs:

```
minimize    ½ xᵀPx + qᵀx
subject to  l ≤ Ax ≤ u
```

**The goal is to support every matrix representation** — dense, sparse, structured, lazy,
and anything else satisfying the `AbstractMatrix` interface — over any `Real` element type.
That is the design aim, not a caveat attached to one favoured format. `P` and `A` are held
by reference and never copied or modified, and every per-iteration product runs `mul!` on
the matrix you passed, so a `Diagonal`, a `SubArray` or a `SparseMatrixCSC` keeps its own
product rather than being flattened into a dense copy.

The numerics
use `LinearAlgebra` alone; the only other dependency is
[TypeContracts.jl](https://github.com/el-oso/TypeContracts.jl), which declares the
linear-system backend interface and checks it at precompilation. Sparse `P` and `A` are
accepted as they are: a `SparseArrays` **weak** dependency lets equilibration walk only
their stored entries, the per-iteration products use their own `mul!`, the reduced matrix is
accumulated over the stored entries rather than through a dense copy of `A`, and it is
factored by CHOLMOD when its factor stays sparse. Eliminating to the reduced system does fill
in whatever sparsity `A` had, so on a pattern that fills in the dense factorization is still
the faster one, and is still what gets chosen.

The hot path is proven allocation-free and type-stable, and every public entry point
compiles under `juliac --trim`. See [Guarantees](https://el-oso.github.io/PureOSQP.jl/dev/guarantees).

```julia
using PureOSQP

P = [4.0 1.0; 1.0 2.0]
q = [1.0, 1.0]
A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
l = [1.0, 0.0, 0.0]
u = [1.0, 0.7, 0.7]

sol = solve(P, q, A, l, u)
sol.status   # SOLVED
sol.x        # [0.3, 0.7]
```

## What it implements

Modified Ruiz equilibration, vector-valued adaptive ρ with an equality/inequality split,
primal and dual infeasibility certificates, warm starting, and active-set solution
polishing — that is, the parts of OSQP that determine how it actually behaves, not just the
ADMM recurrence.

Duality-gap termination is implemented and on by default, following libosqp 1.x.

A MathOptInterface wrapper ships as a package extension: `Model(PureOSQP.Optimizer)` works
from JuMP, and the whole of `MOI.Test` passes.

Solution derivatives ship too, by implicit differentiation of the KKT conditions rather
than through the ADMM loop, checked against `ForwardDiff` to `1e-13`. Dual numbers run the
solver directly, since the element type is `Real`.

A matrix-free backend ships as a second extension, over
[Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl): `linsys = :indirect` runs
preconditioned conjugate gradients on the reduced system and never forms it, so a matrix
that only supplies products is solvable.
Which backend is faster depends on the problem and the answer turns over: on dense QPs the
factorization wins by 15–49×, and on sparse ones it depends on size — the matrix-free
backend is 0.09× at `n = 200` and 1.90× at `n = 4000, m = 8000`, on 33× less memory, because
the direct backend still factors a dense `n×n` inverse however sparse the input.

The backend is chosen by representation, and for a sparse `A` there are two. One accumulates
the reduced matrix over the stored entries instead of writing `A` into an `m×n` buffer for a
dense product — at `n = 2000, m = 4000` and 0.25% density that takes a refactorization from
400 ms to 148 ms, a whole solve from 1192 ms to 777 ms, and the workspace from 93 MiB to
32 MiB. The other also *factors* sparsely, by CHOLMOD through SparseArrays, whenever the
Cholesky factor stays sparse enough to pay: on banded problems that is worth 20× at
`n = 1000` and 37× at `n = 2000` against the dense factorization, on identical iterates.
Both gates are measured rather than guessed — the second asks CHOLMOD what the fill actually
is — and both choices happen once, at `setup`.

A structured `P` is read through traversals that visit only the rows a band type can hold a
nonzero in, so a `Diagonal` or `SymTridiagonal` equilibrates about 2.2× faster than the same
matrix stored densely rather than, as before, slower.

See the [Roadmap](https://el-oso.github.io/PureOSQP.jl/dev/roadmap) for what remains.
