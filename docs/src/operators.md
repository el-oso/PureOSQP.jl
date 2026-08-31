# Structured operators

The solver holds `P` and `A` by reference and never copies them, so a matrix that knows its
own structure keeps that structure all the way to the point where a backend is chosen. This
page is how to bring one: a representation that stores less than `m×n`, computes its products
from what it stores, and is solved by a backend that never forms an `n×n` object.

Two examples ship, and both are worth reading as templates:
[`PureOSQP.BlockDiagonal`](@ref) with [`PureOSQP.BlockReduced`](@ref), and
[`PureOSQP.RowCoupled`](@ref) with [`PureOSQP.DiagonalLowRank`](@ref).

## What to implement, in order

Each step buys something on its own, so a representation is usable before it is finished.

### 1. `size` and `mul!` — the operator works

Declare the type `<: AbstractMatrix{T}` and give it `size`, `mul!` against a vector, and
`mul!` against a vector for its adjoint. Products run off whatever the type stores.

```julia
struct Blocks{T} <: AbstractMatrix{T}
    blocks::Vector{Matrix{T}}
end

Base.size(A::Blocks) = (sum(b -> size(b, 1), A.blocks), sum(b -> size(b, 2), A.blocks))

function LinearAlgebra.mul!(y::AbstractVector, A::Blocks, x::AbstractVector)
    r = c = 1
    for b in A.blocks
        rows, cols = size(b)
        mul!(view(y, r:(r + rows - 1)), b, view(x, c:(c + cols - 1)))
        r += rows
        c += cols
    end
    return y
end
```

At this point the solver runs, and the per-iteration products cost what the structure costs.
Selection still reaches the dense terminal, so an `n×n` reduced matrix is still formed.

An operator from a hierarchy that is not `AbstractMatrix` — `LinearMaps.LinearMap`,
`SciMLOperators.AbstractSciMLOperator` — is wrapped rather than rewritten:
[`PureOSQP.ProductOperator`](@ref) presents one as an `AbstractMatrix`, and loading LinearMaps
lets `setup` and `solve` take a `LinearMap` directly.

### 2. `structural_rows` — setup stops paying for the zeros

[`PureOSQP.structural_rows`](@ref)`(M, j)` answers which rows column `j` can hold a nonzero
in. Equilibration and the dense formation both walk columns through it, and its generic answer
is *every* row — so without a method, setup costs `O(mn)` however little the type stores.

```julia
PureOSQP.structural_rows(A::Blocks, j::Integer) = rowrange_of_the_block_holding(A, j)
```

This is the highest-value method on the page. Adding it for `BandedMatrix` took a banded
`setup` at `n = 2000` from 170 ms to 2.7 ms, because equilibration had been visiting every
row of every column of a matrix holding `O(nb)` entries.

An operator with no columns to enumerate — one that only multiplies — has two other routes:
build it with `probe = true`, which recovers column `j` as `op * eⱼ`, or pass `scaling = 0`
and skip equilibration. Both are described under [`PureOSQP.ProductOperator`](@ref).

### 3. A `LinearSystem` — the reduced matrix is never formed

Subtype [`PureOSQP.LinearSystem`](@ref) and implement `factorize!`, `solve_system!` and
`backend_info`; the contract is enforced at precompilation, and
[`PureOSQP.refactor_rho!`](@ref) is optional. The backend decides what "solve the reduced
system" means for this structure, and it is where the `O(n²)` object stops existing.

```julia
mutable struct BlockSolve{T} <: PureOSQP.LinearSystem
    inv::Vector{Matrix{T}}      # one inverse per block; no n×n anything
end
```

`factorize!` builds whatever the structure implies — for blocks, one small factorization each;
for a low-rank correction, a `k×k` capacitance. `solve_system!` calls
[`PureOSQP.reduced_rhs!`](@ref) first, writes `ws.xtilde`, then `mul_A!` into `ws.ztilde`.

Two things worth copying from the shipped backends rather than rediscovering. Invert each
block and use `symv` instead of keeping a factor and calling `ldiv!`: both cost `2nᵢ²` flops,
but a triangular solve computes its entries in sequence and `symv` does not, and the `σI` in
the reduced matrix bounds the conditioning that would otherwise make inverting unwise. And
assemble each block with one `mul!` rather than a scalar loop — the difference was 1.7× on the
block backend.

### 4. `choose_backend` — selection finds it

Selection is multiple dispatch. A method on `choose_backend` for your pair of types wins over
the ladder by being more specific, which is the whole mechanism:

```julia
PureOSQP.choose_backend(P::Blocks, A::Blocks, proto, n, m) = (BlockSolve(...), false)
```

Return `(backend, false)` when the backend arrives unfactored, `(backend, true)` when it
already carries a factorization of the current data. A rung inside the ladder instead returns
`nothing` to decline; see [`PureOSQP.select_backend`](@ref) for the order and what each rung
serves.

## What each omission costs

| stopped after | products | setup | reduced matrix | selection |
|---|---|---|---|---|
| 1 — `size`, `mul!` | structured | `O(mn)` | formed, `n×n` | dense terminal |
| 2 — `structural_rows` | structured | `O(nnz)` | formed, `n×n` | dense terminal |
| 3 — a `LinearSystem` | structured | `O(nnz)` | never formed | needs step 4 |
| 4 — `choose_backend` | structured | `O(nnz)` | never formed | reaches your backend |

## Paths that need entries, and how they decline

Polishing and the solution derivatives copy `P` and `A` into a dense factorization one entry
at a time. An operator that cannot answer that declares
[`PureOSQP.is_materializable`](@ref) `false`, and those paths then throw a message naming the
remedy rather than a `MethodError` from inside the copy. The rungs that would form a matrix
decline it too, so `linsys = :auto` reaches the matrix-free backend instead of failing inside
a factorization.

Conjugate gradients is the fallback for an operator with no structure to exploit, and it is a
real fallback rather than a good one: on an ill-conditioned problem it can return an answer
that is wrong rather than merely slow. A structured direct backend is the reason this page
exists.
