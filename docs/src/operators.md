# Structured operators

This page is for adding a matrix type the package does not ship. To *use* the shipped
structured types, see [Examples](@ref "Structured operators the package ships").

Integration is incremental across four steps; you can stop after any of them. Step 1 makes
your type work. Step 2 makes setup cheaper. Step 3 is where the large wins are. Step 4 makes
the solver find it automatically. A type that stops after step 1 is usable but leaves speed
on the table.

The solver holds `P` and `A` by reference and never copies them, so a matrix that knows its
own structure keeps it until a backend is chosen. What you build is a representation that
stores less than `m×n`, computes its products from what it stores, and — by step 3 — is solved
by a backend that never forms an `n×n` object.

Three shipped examples serve as templates:

| representation | backend | what the structure buys |
|---|---|---|
| [`PureOSQP.BlockDiagonal`](@ref) | [`PureOSQP.BlockReduced`](@ref) | `R` decouples into `K` independent systems |
| [`PureOSQP.RowCoupled`](@ref) | [`PureOSQP.DiagonalLowRank`](@ref) | `R` is a diagonal plus a rank-`k` correction, solved by Woodbury |
| [`PureOSQP.KroneckerOperator`](@ref) | [`PureOSQP.KroneckerReduced`](@ref) | `R` is diagonal in the factors' eigenbasis |

`docs/src/examples.md` runs all three. The Kronecker backend is the cautionary case: it
applies only when `P` is a scalar multiple of `I`, `ρ` is one number, and `scaling = 0`, so
most of its design is the check that declines. A backend that would answer wrongly outside
its conditions must check them.

## What to implement, in order

The methods below are the **operator protocol** — the set a type implements to be usable as
a `P` or an `A`. The term is used throughout this manual and in the benchmarks and means
exactly this list. A type that implements it is an operator whether or not it stores entries;
nothing else is required, and no method here is inherited from `AbstractMatrix`.

Each step is useful on its own, so a representation works before it is finished.

### 1. `size`, `mul!` and `getindex` — the operator works

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

`getindex` is needed too: `setup` tests `P` for symmetry and for `P + σI` being positive
definite *before* a backend is chosen, and the generic methods for both read entries. An
operator with only `size` and `mul!` fails there with a `CanonicalIndexError`, not at solve
time. Two fixes — give the type a `getindex` (a cheap lookup for most structured types), or
override [`PureOSQP.is_symmetric`](@ref) and [`PureOSQP.is_convex`](@ref) block-wise and
declare [`PureOSQP.is_materializable`](@ref) `false`.

With that, the solver runs and the per-iteration products cost what the structure costs.
Selection still reaches the dense terminal, so an `n×n` reduced matrix is still formed.

An operator from a hierarchy that is not `AbstractMatrix` — `LinearMaps.LinearMap`,
`SciMLOperators.AbstractSciMLOperator` — is wrapped rather than rewritten:
[`PureOSQP.ProductOperator`](@ref) presents one as an `AbstractMatrix`, and loading LinearMaps
lets `setup` and `solve` take a `LinearMap` directly.

### 2. `structural_rows` — setup stops paying for the zeros

[`PureOSQP.structural_rows`](@ref)`(M, j)` answers which rows column `j` can hold a nonzero
in. Equilibration and the dense formation both walk columns through it, and its generic
answer is *every* row — so without a method, setup costs `O(mn)` however little the type
stores.

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

`factorize!` builds whatever the structure implies — for blocks, one small factorization
each; for a low-rank correction, a `k×k` capacitance. `solve_system!` calls
[`PureOSQP.reduced_rhs!`](@ref) first, writes `ws.xtilde`, then `mul_A!` into `ws.ztilde`.

Two things the shipped backends do well. Invert each block and use `symv` instead of keeping
a factor and calling `ldiv!`: both cost `2nᵢ²` flops, but a triangular solve computes its
entries in sequence while `symv` does not, and the `σI` in the reduced matrix bounds the
conditioning that would otherwise make inverting unwise. And assemble each block with one
`mul!` rather than a scalar loop — the difference was 1.7× on the block backend.

### 4. `choose_backend` — selection finds it

Selection is multiple dispatch. A method on `choose_backend` for your pair of types wins over
the ladder by being more specific, which is the whole mechanism:

```julia
function PureOSQP.choose_backend(
        P::Blocks, A::Blocks, proto::AbstractVector, n::Integer, m::Integer,
        D, E, c, rho_vec, sigma
    )
    return (BlockSolve(...), false)
end
```

The ten arguments are the seam's signature: `D`, `E`, `c` and `rho_vec` are there because
two rungs decide by factoring the *equilibrated* matrix and handing that factorization on.
Define a method with fewer arguments and it never dispatches — the ladder proceeds to the
dense terminal and nothing reports a problem.

Return `(backend, false)` when the backend arrives unfactored, `(backend, true)` when it
already carries a factorization of the current data. A rung inside the ladder instead returns
`nothing` to decline; see [`PureOSQP.select_backend`](@ref) for the order and what each rung
serves.

## What each omission costs

| stopped after | products | equilibration | reduced matrix | selection |
|---|---|---|---|---|
| 1 — `size`, `mul!`, `getindex` | structured | `O(mn)` | formed, `n×n` | dense terminal |
| 2 — `structural_rows` | structured | `O(nnz)` | formed, `n×n` | dense terminal |
| 3 — a `LinearSystem` | structured | `O(nnz)` | never formed | needs step 4 |
| 4 — `choose_backend` | structured | `O(nnz)` | never formed | reaches your backend |

The equilibration column is not the whole of `setup`. `is_symmetric` and `is_convex` run
before a backend is chosen and their generic methods are `O(n²)` and `O(n³)` — a type that
does not override them pays that on every `setup`. [`PureOSQP.BlockDiagonal`](@ref) overrides
both block-wise and is worth reading for the shape.

## Paths that need entries, and how they decline

Polishing and the solution derivatives copy `P` and `A` into a dense factorization one entry
at a time. An operator that cannot answer that declares
[`PureOSQP.is_materializable`](@ref) `false`, and those paths then throw a message naming the
remedy rather than a `MethodError` from inside the copy. The rungs that would form a matrix
decline it too, so `linsys = :auto` reaches the matrix-free backend instead of failing inside
a factorization.

Conjugate gradients is the fallback for an operator with no structure to exploit, and it is
a real fallback rather than a good one: on an ill-conditioned problem it can return a wrong
answer rather than a merely slow one. A structured direct backend is the point of this page.
