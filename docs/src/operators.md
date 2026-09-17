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
| [`PureQPBase.BlockDiagonal`](@ref) | [`PureQPBase.BlockReduced`](@ref) | `R` decouples into `K` independent systems |
| [`PureQPBase.RowCoupled`](@ref) | [`PureQPBase.DiagonalLowRank`](@ref) | `R` is a diagonal plus a rank-`k` correction, solved by Woodbury |
| [`PureQPBase.KroneckerOperator`](@ref) | [`PureQPBase.KroneckerReduced`](@ref) | `R` is diagonal in the factors' eigenbasis |

`docs/src/examples.md` runs all three. The Kronecker backend is the cautionary case: it
applies only when `P` is a scalar multiple of `I`, `ρ` is one number, and `scaling = 0`, so
most of its design is the check that decides whether it can be used at all. A backend that
would answer wrongly outside its conditions must check them.

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
override [`PureQPBase.is_symmetric`](@ref) and [`PureQPBase.is_convex`](@ref) block-wise and
declare [`PureQPBase.is_materializable`](@ref) `false`.

With that, the solver runs and the per-iteration products cost what the structure costs.
Selection still reaches the dense terminal, so an `n×n` reduced matrix is still formed.

An operator from a hierarchy that is not `AbstractMatrix` — `LinearMaps.LinearMap`,
`SciMLOperators.AbstractSciMLOperator` — is wrapped rather than rewritten:
[`PureQPBase.ProductOperator`](@ref) presents one as an `AbstractMatrix`. Loading either package
lets `setup` and `solve` take its operators directly; see
[Operators from functions](@ref) for which of the two to reach for.

### 2. `structural_rows` — setup stops paying for the zeros

[`PureQPBase.structural_rows`](@ref)`(M, j)` answers which rows column `j` can hold a nonzero
in. Equilibration and the dense formation both walk columns through it, and its generic
answer is *every* row — so without a method, setup costs `O(mn)` however little the type
stores.

```julia
PureQPBase.structural_rows(A::Blocks, j::Integer) = rowrange_of_the_block_holding(A, j)
```

This is the highest-value method on the page. Adding it for `BandedMatrix` took a banded
`setup` at `n = 2000` from 170 ms to 2.7 ms, because equilibration had been visiting every
row of every column of a matrix holding `O(nb)` entries.

An operator with no columns to enumerate — one that only multiplies — has two other routes:
build it with `probe = true`, which recovers column `j` as `op * eⱼ`, or pass `scaling = 0`
and skip equilibration. Both are described under [`PureQPBase.ProductOperator`](@ref).

### 3. A `LinearSystem` — the reduced matrix is never formed

Subtype [`PureQPBase.LinearSystem`](@ref) and implement `factorize!(ls, prob, wt)`,
`solve_system!(ls, prob, wt, rhs_x, rhs_z, x, z)` and `backend_info`; the contract is enforced
at precompilation, and [`PureQPBase.refactor_weights!`](@ref) is optional. `prob` is the
[`PureQPBase.Problem`](@ref) and `wt` the [`PureQPBase.SystemWeights`](@ref). The backend decides
what "solve the reduced system" means for this structure, and it is where the `O(n²)` object
stops existing.

```julia
mutable struct BlockSolve{T} <: PureQPBase.LinearSystem
    inv::Vector{Matrix{T}}      # one inverse per block; no n×n anything
end
```

`factorize!` builds whatever the structure implies — for blocks, one small factorization
each; for a low-rank correction, a `k×k` capacitance. `solve_system!` calls
[`PureQPBase.reduced_rhs!`](@ref) first, writes `x`, then `mul_A!` into `z`.

Two things the shipped backends do well. Invert each block and use `symv` instead of keeping
a factor and calling `ldiv!`: both cost `2nᵢ²` flops, but a triangular solve computes its
entries in sequence while `symv` does not, and the `σI` in the reduced matrix bounds the
conditioning that would otherwise make inverting unwise. And assemble each block with one
`mul!` rather than a scalar loop — the difference was 1.7× on the block backend.

### 4. `choose_backend` — selection finds it

Selection is multiple dispatch. A method on `choose_backend` for your pair of types wins over
the ladder by being more specific, which is the whole mechanism:

```julia
function PureQPBase.choose_backend(
        P::Blocks, A::Blocks, prob::PureQPBase.Problem, wt::PureQPBase.SystemWeights,
        sel::PureQPBase.SelectionFor
    )
    return (BlockSolve(...), false)
end
```

`prob` is the [`PureQPBase.Problem`](@ref) holding `P`, `A` and the equilibration factors; `wt`
is the [`PureQPBase.SystemWeights`](@ref) holding `ρ` and `σ`; `sel` is
[`PureQPBase.ADMMSelection`](@ref) or [`PureQPBase.IPMSelection`](@ref), which lets a method serve
one algorithm's ladder, the other's, or both by leaving it untyped as `SelectionFor` above.
Define a method that matches neither `P` nor `A` specifically and it never dispatches — the
ladder proceeds to the dense terminal and nothing reports a problem.

Return `(backend, false)` when the backend arrives unfactored, `(backend, true)` when it
already carries a factorization of the current data. A rung inside the ladder instead returns
`nothing` to leave the pair to the next rung; see [`PureQPBase.select_backend`](@ref) for the
order and what each rung serves.

## What each omission costs

| stopped after | products | equilibration | reduced matrix | selection |
|---|---|---|---|---|
| 1 — `size`, `mul!`, `getindex` | structured | `O(mn)` | formed, `n×n` | dense terminal |
| 2 — `structural_rows` | structured | `O(nnz)` | formed, `n×n` | dense terminal |
| 3 — a `LinearSystem` | structured | `O(nnz)` | never formed | needs step 4 |
| 4 — `choose_backend` | structured | `O(nnz)` | never formed | reaches your backend |

The equilibration column is not the whole of `setup`. `is_symmetric` and `is_convex` run
before a backend is chosen and their generic methods are `O(n²)` and `O(n³)` — a type that
does not override them pays that on every `setup`. [`PureQPBase.BlockDiagonal`](@ref) overrides
both block-wise and is worth reading for the shape.

## What happens on paths that need entries

Polishing and the solution derivatives copy `P` and `A` into a dense factorization one entry
at a time. An operator that cannot answer that declares
[`PureQPBase.is_materializable`](@ref) `false`, and those paths then throw a message naming the
remedy rather than a `MethodError` from inside the copy. The rungs that would form a matrix
are skipped too, so `linsys = :auto` reaches the matrix-free backend instead of failing inside
a factorization.

Conjugate gradients is the fallback for an operator with no structure to exploit, and it is
a real fallback rather than a good one: on an ill-conditioned problem it can return a wrong
answer rather than a merely slow one. A structured direct backend is the point of this page.

## Operators under the interior-point method

[`InteriorPoint`](@ref) solves an operator pair only with `linsys = :indirect`, a
`preconditioner` the caller supplies, and `scaling = 0`; `linsys = :auto` never chooses it, and
the same three requirements hold for matrices. Without a preconditioner, with the built-in
[`PureQPBase.JacobiPreconditioner`](@ref) or [`PureQPBase.IdentityPreconditioner`](@ref), with
equilibration on, or with an operator built with `probe = true`, `setup` throws an
`ArgumentError` naming the requirement.

The preconditioner approximates `P + σI + Aᵀ diag(w) A` for the `P` and `A` passed to `setup`.
The interior-point weights `w` change every outer iteration and reach `1/reg_dual` on equality
and active rows, so a product-only preconditioner does not keep conjugate gradients within
budget; the caller supplies one built from what they know about the operators. It is refreshed
through [`PureQPBase.update_preconditioner!`](@ref), which receives `k = -1` for the starting
point and the outer iteration `k = 0, 1, 2, …` afterwards, and applied through
`LinearAlgebra.ldiv!`. A factor of the reduced matrix built from dense copies of `P` and `A`,
refreshed every third outer iteration:

```julia
mutable struct LaggedCholesky{T}
    const P::Matrix{T}
    const A::Matrix{T}
    const every::Int
    F::Cholesky{T, Matrix{T}}
    sigma::T
end
LaggedCholesky(P, A; every = 3) =
    LaggedCholesky(Matrix(P), Matrix(A), every, cholesky(Matrix(1.0I, size(P)...)), NaN)

function PureQPBase.update_preconditioner!(M::LaggedCholesky, prob, wt, k::Int)
    (k < 0 || iszero(k % M.every) || wt.sigma != M.sigma) || return M
    M.F = cholesky(Symmetric(M.P + wt.sigma * I + M.A' * Diagonal(wt.w) * M.A))
    M.sigma = wt.sigma
    return M
end
LinearAlgebra.ldiv!(y::AbstractVector, M::LaggedCholesky, x::AbstractVector) = ldiv!(y, M.F, x)

Pop = LinearMap(P; issymmetric = true, isposdef = true)
sol = solve(Pop, q, LinearMap(A), l, u, InteriorPoint(); linsys = :indirect,
            preconditioner = LaggedCholesky(P, A), scaling = 0)
```

Each Newton solve starts conjugate gradients from zero and stops once the two-norm of its
recursively updated residual is below `cg_tol_fraction · min(μ, ‖r‖∞)`. A solve that spends
`cg_max_iter` iterations, or that conjugate gradients abandons because the preconditioner is
not symmetric positive definite, is missed; `cg_fail_limit` missed solves in a row end the run
`NUMERICAL_ERROR`. There is no refinement step (`InteriorPoint`'s `refine_iter` defaults to `0`
here), and
`Solution.cg_iters` reports the conjugate-gradient iterations of the solve.

### Measured

`PureIPM/bench/ipm_matrixfree.jl` writes `PureIPM/bench/results/ipm_matrixfree.json`. It runs dense instances
with a planted solution as `LinearMap`s through `LaggedCholesky` (`every = 3`) at
`n = m ∈ {500, 1000, 2000}`, `κ(A) ∈ {1, 1e6}`, active fractions `{0.1, 0.9}`, every row
two-sided and with a mix of 20% equality, 20% lower-only, 20% upper-only and 10% free rows:
24 instances, at `eps_abs = eps_rel = 1e-6`, `reg_primal = reg_dual = 1e-8`,
`cg_max_iter = 500`, BLAS single-threaded, four instances at a time. Two criteria decide
whether the path is supported:

- **G1**: the solve ends `SOLVED` with the largest optimality residual, computed from the data,
  at most `1e-5`.
- **G2**: with `f` the median inner iterations per solve over the first three outer iterations
  and `t` over the last three, `t ≤ max(10f, min(100, n/10))`, no solve reaches `cg_max_iter`,
  and no solve takes `n` inner iterations or more.

| `n` | G1 | G2 | outer iterations | largest residual | largest `t` / G2 bound | largest inner count | wall clock, all 8 (dense KKT) |
|---|---|---|---|---|---|---|---|
| 500 | 8/8 | 8/8 | 6–9 | `7.9e-7` | 44.5 / 90 | 107 | 1.4 s (2.0 s) |
| 1000 | 8/8 | 8/8 | 6–9 | `5.6e-7` | 46 / 120 | 138 | 8.7 s (7.7 s) |
| 2000 | 8/8 | 8/8 | 6–9 | `6.5e-7` | 44.5 / 115 | 138 | 29.5 s (33.5 s) |

Both criteria pass on all 24, so the path is supported on this family. Every instance takes
the same number of outer iterations as the dense full KKT factorization (`linsys = :kkt`) on
the same matrices. Wall clock includes every preconditioner build; at this density the
lagged Cholesky costs about what the full KKT factorization does, so the timing says the
path is not slower, not that it is faster. ADMM on the same operators (`linsys = :indirect`,
`n = 500`, `eps = 1e-6`, 10 s limit) solves seven of the eight in 0.02–7.0 s, the four with
`κ = 1` under 0.1 s and two with `κ = 1e6` above 6 s, and reaches the limit on `κ = 1e6`,
`0.9` active, mixed rows; the interior point takes 0.04–0.66 s on the same eight. These
timings are indicative: they ran on a workstation with an unpinned clock.

A primal-infeasible operator instance (`n = 100`, two copies of one row with disjoint
intervals) ends `PRIMAL_INFEASIBLE`, and a `Diagonal` preconditioner holding a negative entry
ends `NUMERICAL_ERROR`.

The measurement covers this dense family and this preconditioner only. On one sparse instance
(`n = 1000`, ten random nonzeros per row of `A` plus the identity) a limited-memory incomplete
`LDLᵀ` refreshed every outer iteration fails both criteria: its inner counts grow to the
`500` cap by the eighth outer iteration and the run ends `NUMERICAL_ERROR`. A preconditioner
has to keep the inner count bounded as the weights spread; that is the property to check
before relying on this path.
