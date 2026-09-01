# Matrix types

How you store `P` and `A` decides how much work the solver does — sometimes by a factor of
hundreds — and you choose it by passing a different matrix type, not by changing a setting.

This page is the catalogue and the reasoning: which representation suits which problem, what
each type looks like, what backend it reaches, and how to bring a type of your own. For worked
problems see [Examples](@ref); for the measurements behind the advice, [Benchmarks](@ref).

## Matrix representations

Everything above built `P` and `A` as ordinary dense matrices. That always works, and if your
problems are small it is all you need — you can stop reading here and come back when one gets
slow.

The rest of this page is about what to do when they do get slow. The short version: **how you
store `P` and `A` changes how much work the solver has to do**, sometimes by a factor of
hundreds, and you get that by passing a different matrix type rather than by changing any
setting.

Two words are used throughout, so here they are once:

- The **reduced matrix** is the `n×n` matrix the solver has to solve against on every
  iteration. You never see it, but it is where nearly all the time goes. Its size and shape
  come from `P` and `A`, and the whole point of the sections below is to keep it small or
  cheap. (It is `R = cDPD + σI + Ãᵀdiag(ρ)Ã`, if you want the formula; you do not need it to
  use any of this.)
- A **backend** is the code that solves against that matrix. There are ten or so. You do not
  choose one — `setup` looks at the types of `P` and `A` and picks. `PureOSQP.backend_name(ws.linsys)`
  tells you which it picked, and every example below uses that to show the choice being made.

So the workflow is always the same: pass a matrix type that describes your problem, then check
which backend you got. If it is the one you expected, the structure was used.

### Which representation, and why

There are four answers, and which is right is a property of your problem rather than a
preference. The short version:

| your problem | use | because |
|---|---|---|
| small enough to sit in cache | **dense** | nothing beats a contiguous array the CPU can keep close. Structure costs indirection that buys nothing at this size. |
| large, and mostly zeros | **sparse** | you pay for the nonzeros instead of `n²`. This is the familiar case and `SparseMatrixCSC` handles it. |
| you know more about it than "where the zeros are" | **a structured type** | block-diagonal, low-rank, Kronecker. The solver can then skip work that no sparsity pattern reveals — a `BlockDiagonal` is solved as `K` small systems, never as one big one. |
| few zeros, but a fast way to apply it, and too big for cache | **unmaterialized** | past cache the dense product is limited by memory bandwidth, not arithmetic. An operator that computes its product from `O(n)` stored numbers moves almost nothing and can win outright. |

That last row is the one that is easy to miss, so it is worth being concrete about. The tables
below come from `bench/representation_choice.jl`, single-threaded, statuses asserted — a run
that stopped at `max_iter` is not a faster answer to the same question.

**When being matrix-free does not pay.** An operator whose product costs what the dense product
costs saves a factorization once and pays for it every iteration:

| n | iterations | operator | dense | |
|---|---|---|---|---|
| 200 | 125/125 | 6.0 ms | 2.2 ms | 0.37× |
| 500 | 150/125 | 56.1 ms | 20.8 ms | 0.37× |
| 1000 | 175/175 | 220 ms | 131 ms | 0.60× |

**When it does.** The same comparison, for an operator applied in `O(n)` whose dense form is
`O(n²)`, at about a tenth of the entries nonzero — too dense for a sparse format to be the
obvious answer:

| n | fill | iterations | operator | dense | | dense `A` |
|---|---|---|---|---|---|---|
| 500 | 9.9% | 100/75 | 10.2 ms | 15.8 ms | **1.55×** | 1.9 MiB |
| 1000 | 9.8% | 100/100 | 40.5 ms | 104 ms | **2.57×** | 7.6 MiB |
| 2000 | 9.8% | 100/100 | 348 ms | 730 ms | **2.10×** | 30.5 MiB |
| 4000 | 9.8% | 100/75 | 1761 ms | 4806 ms | **2.73×** | 122 MiB |

Same solver, same tolerances, both converged. The difference between the two tables is not the
size and not the sparsity — it is whether **applying** the operator is asymptotically cheaper
than the dense product. If it is, the operator wins and wins by more as the matrix leaves
cache. If it is not, no amount of size will save it.

### Every type the solver takes, and the shape it means

The decision above is about categories. This is the catalogue: what each type looks like, what
to call it, and what it buys. `•` marks a stored entry, blank is a structural zero.

**Dense `Matrix`.** Every entry stored. The baseline, and the right answer whenever the
problem is small or has no structure to declare.

```math
\begin{pmatrix} • & • & • & • \\ • & • & • & • \\ • & • & • & • \\ • & • & • & • \end{pmatrix}
```

**`Diagonal`** (LinearAlgebra). One entry per row. With a `Diagonal` `A` too, the reduced
matrix is diagonal and a solve is `n` divisions — no factorization at all.

```math
\begin{pmatrix} • & & & \\ & • & & \\ & & • & \\ & & & • \end{pmatrix}
```

**`Bidiagonal`**, and **`SymTridiagonal`** / **`Tridiagonal`** (LinearAlgebra). Bandwidth 1.
Smoothing, trend filtering and differencing constraints land here; the backend is an `ldlt` in
`O(n)`.

```math
\begin{pmatrix} • & • & & \\ • & • & • & \\ & • & • & • \\ & & • & • \end{pmatrix}
```

**`BandedMatrix`** ([BandedMatrices.jl](https://github.com/JuliaLinearAlgebra/BandedMatrices.jl)).
Bandwidth 2 and up, which LinearAlgebra has no symmetric type for. Factored as a banded
Cholesky in `O(n b²)`.

```math
\begin{pmatrix} • & • & • & & \\ • & • & • & • & \\ • & • & • & • & • \\ & • & • & • & • \\ & & • & • & • \end{pmatrix}
```

**`SparseMatrixCSC`** (SparseArrays). Entries wherever you put them, stored by column. The
right answer when the pattern is irregular and mostly empty.

```math
\begin{pmatrix} • & & • & \\ & • & & \\ • & & & • \\ & & • & • \end{pmatrix}
```

**[`PureOSQP.BlockDiagonal`](@ref).** A run of independent blocks, stored as the blocks. `K`
systems of size `n/K` instead of one of size `n`: `n³/K²` work and `1/K` the memory.

```math
\begin{pmatrix} • & • & & & & \\ • & • & & & & \\ & & • & • & & \\ & & • & • & & \\ & & & & • & • \\ & & & & • & • \end{pmatrix}
```

**[`PureOSQP.RowCoupled`](@ref).** A few dense rows above rows holding one entry each — a bound
per variable plus a budget or total. Solved by Woodbury in `O(nk)`, never forming the `n×n`.

```math
\begin{pmatrix} • & • & • & • \\ • & • & • & • \\ • & & & \\ & • & & \\ & & • & \\ & & & • \end{pmatrix}
```

**[`PureOSQP.KroneckerOperator`](@ref).** `A₁ ⊗ A₂`, held as its two factors. A constraint
acting across two dimensions at once; the `6×6` below is stored as `4 + 9` numbers.

```math
A_1 \otimes A_2 = \begin{pmatrix} a_{11}A_2 & a_{12}A_2 \\ a_{21}A_2 & a_{22}A_2 \end{pmatrix}
```

**[`PureOSQP.ProductOperator`](@ref), and `LinearMaps.LinearMap`.** No entries at all — a
function that applies the matrix. There is no picture to draw, which is the point:

```math
x \;\longmapsto\; Ax
```

Views (`SubArray`), `Symmetric` wrappers and GPU arrays are all accepted too; they are storage
decisions rather than shapes, and carry no backend of their own.

| type | reduced matrix | backend |
|---|---|---|
| `Matrix` | dense | `cholesky` |
| `Diagonal` with `Diagonal` | diagonal | `diagonal` |
| tri/bidiagonal with `Diagonal` | bandwidth 1 | `tridiagonal` |
| banded with banded | bandwidth `2 ≤ b ≤ n/4` | `banded` |
| `SparseMatrixCSC` | sparse, or dense once it fills | `cholmod`, `ldlfactorizations`, `sparse_formed` |
| `BlockDiagonal` pair | `K` blocks | `block` |
| `Diagonal` with `RowCoupled` | diagonal plus rank `k` | `lowrank` |
| `μI` with `KroneckerOperator` | diagonal in the factors' eigenbasis | `kronecker` |
| an operator | never formed | `indirect` |

`PureOSQP.backend_name(ws.linsys)` reports which one you got.

### Unmaterialized does not mean solved by CG

One more distinction, because it decides what happens on badly conditioned problems.

An operator with no exploitable structure can only be served by *conjugate gradients* —
multiply, repeat — and CG is sensitive to conditioning. But an operator that carries its own
**direct** backend is solved by factoring, and conditioning is then no worse than the structure
implies.

The Kronecker type is the clean example. `κ(A₁ ⊗ A₂) = κ(A₁)·κ(A₂)`, so an operator at
`κ = 1e12` is built from two factors at `1e6` — and the backend eigendecomposes the *factors*,
never forming or factoring the product:

| n | κ(A) | iterations | Kronecker | dense | | conjugate gradients, same problem |
|---|---|---|---|---|---|---|
| 400 | 1e12 | 625/625 | 2.4 ms | 29.6 ms | **12×** | converges, but in 1025 iterations |
| 1600 | 1e12 | 1100/1100 | 22.4 ms | 1560 ms | **70×** | `MAX_ITER_REACHED` at 20 000 |

Both routes agree on the objective to six figures. CG on the same problem manages it at
`n = 400` and fails outright at `n = 1600` — so on an ill-conditioned problem the useful move
is a structured operator with a direct backend, not a generic one served iteratively.

The problems below are all the same QP, written five ways.

```@example storage
using PureOSQP, LinearAlgebra, SparseArrays

n = 6
Pdiag = Diagonal(2.0 .+ (1:n) ./ n)
Aband = Bidiagonal(fill(1.0, n), fill(-1.0, n - 1), :U)
q = collect(range(-1.0, 1.0; length = n))
l = fill(-0.5, n)
u = fill(0.5, n)

Pd, Ad = Matrix(Pdiag), Matrix(Aband)
reps = [
    "Matrix"                      => (Pd, Ad),
    "Diagonal, Bidiagonal"        => (Pdiag, Aband),
    "SymTridiagonal, Tridiagonal" => (SymTridiagonal(diag(Pd), zeros(n - 1)), Tridiagonal(Ad)),
    "Symmetric, SubArray"         => (Symmetric(Pd), view(Ad, :, :)),
    "SparseMatrixCSC"             => (sparse(Pd), sparse(Ad)),
]

P_before, A_before = copy(Pd), copy(Ad)
reference = setup(Pd, q, Ad, l, u; eps_abs = 1e-9, eps_rel = 1e-9)
ref = solve!(reference)
ref_backend = PureOSQP.backend_name(reference.linsys)
for (name, (Pr, Ar)) in reps
    ws = setup(Pr, q, Ar, l, u; eps_abs = 1e-9, eps_rel = 1e-9)
    @assert ws.P === Pr && ws.A === Ar          # held by reference, not copied
    sol = solve!(ws)
    backend = PureOSQP.backend_name(ws.linsys)
    @assert sol.iter == ref.iter                       # same trajectory
    @assert isapprox(sol.x, ref.x; rtol = 1e-8)        # same answer
    # Bit-exact only where the same factorization ran; see below.
    backend == ref_backend && @assert sol.x == ref.x && sol.y == ref.y
    println(rpad(name, 30), "iter = ", sol.iter, ",  backend = ", backend)
end
@assert Pd == P_before && Ad == A_before        # the caller's arrays are never written to
```

Two different things are on display here, and it is worth keeping them apart.

**A representation that only changes how entries are reached gives bit-identical answers.**
`Symmetric`, a `SubArray` and a `SparseMatrixCSC` all feed the same numbers into the same
arithmetic, so `==` holds exactly against the dense reference.

**A representation that changes which backend is chosen changes the arithmetic.** A
`Diagonal` `P` with a `Bidiagonal` `A` makes the reduced matrix tridiagonal, and that is
solved by an `ldlt` on two bands rather than by a dense inverse and a `symv` — a different
factorization, agreeing to about `1e-16` rather than to the bit. The iteration count and the
answer are the same; the last digits are not. See
[Which backend a structured matrix gets](@ref "Which backend a structured matrix gets").

`P` has to be symmetric *as stored* — a lower triangle with the upper one left at zero is
rejected rather than mirrored, since that matrix is a different, non-symmetric problem. Wrap
it in `Symmetric` to say which triangle is the real one.

### Which backend a sparse matrix gets

Eliminating `ν` from the ADMM subproblem gives an `n×n` reduced matrix, and whether that
matrix is worth keeping sparse depends on its pattern rather than on the input's density.
`linsys = :auto` decides by asking CHOLMOD to factor the pattern and measuring the fill.

```@example storage
band = 200
banded = setup(
    sparse(SymTridiagonal(fill(2.0, band), fill(0.3, band - 1))),
    collect(range(-1.0, 1.0; length = band)),
    sparse(Bidiagonal(fill(1.0, band), fill(-1.0, band - 1), :U)),
    fill(-1.0, band), fill(1.0, band),
)
PureOSQP.backend_name(banded.linsys)
```

A banded `A` gives a banded reduced matrix, so it is formed and factored sparsely. A
scattered pattern fills in, and then the sparse factor is no cheaper than the dense inverse;
that case still forms the reduced matrix over the stored entries, but factors it densely.

```@example storage
using Random
Random.seed!(1)
ns, ms = 80, 160
scattered = setup(
    sparse(1.0I, ns, ns), collect(range(-1.0, 1.0; length = ns)),
    sprandn(ms, ns, 0.05), fill(-1.0, ms), fill(1.0, ms),
)
PureOSQP.backend_name(scattered.linsys)
```

Both are reported by `PureOSQP.backend_name(ws.linsys)`, which names whichever backend the
workspace ended up with. The dense default is `:cholesky`, and the full quasi-definite
factorization is `:bunchkaufman`.

### Which backend a structured matrix gets

A structured `P` and `A` are not just read more cheaply — they can make the reduced matrix
itself narrow, and then there is far less to factor. Eliminating `ν` gives

```math
R = c D P D + \sigma I + \tilde A^\top \mathrm{diag}(\rho) \tilde A
```

Diagonal scaling preserves a bandwidth and `ÃᵀρÃ` doubles `A`'s, so
`bandwidth(R) = max(bandwidth(P), 2 bandwidth(A))`. `linsys = :auto` dispatches on the pair
of types, with no setting and no density gate involved.

```@example structured
using PureOSQP, LinearAlgebra
n = 200
q, l, u = collect(range(-1.0, 1.0; length = n)), fill(-1.0, n), fill(1.0, n)

# A separable objective under box constraints: R is diagonal, so nothing is factored.
box = setup(Diagonal(fill(2.0, n)), q, Diagonal(ones(n)), l, u)

# A tridiagonal objective under box constraints: R stays tridiagonal.
smooth = setup(SymTridiagonal(fill(2.0, n), fill(0.3, n - 1)), q, Diagonal(ones(n)), l, u)

(PureOSQP.backend_name(box.linsys), PureOSQP.backend_name(smooth.linsys))
```

The first has nothing to factor at all — a solve is `n` divisions — and the second is an
`ldlt` that costs `O(n)`. Against the dense path the same problems would otherwise take,
that is worth a great deal at any size worth caring about; see
[Structured backends](@ref "Structured backends") for the measurements.

Widening `A` widens `R` faster than widening `P` does, which is the practical consequence of
the rule above. A `Tridiagonal` `A` squares to bandwidth 2, past what `SymTridiagonal`
stores, and is served by a banded Cholesky once BandedMatrices.jl is loaded:

```@example structured
using BandedMatrices
diff = setup(
    SymTridiagonal(fill(4.0, n), fill(0.3, n - 1)), q,
    Tridiagonal(fill(-0.25, n - 1), ones(n), fill(-0.25, n - 1)), l, u,
)
(PureOSQP.backend_name(diff.linsys), diff.linsys.bw)
```

Without BandedMatrices loaded that problem takes the dense path instead — correctly, just
not cheaply. Structure in `P` alone never survives: `ÃᵀρÃ` is dense for a general `A`
whatever `P` looked like, so a `Diagonal` `P` with a dense `A` is a dense reduced matrix and
gets the dense backend.

```@example structured
using Random
Random.seed!(2)
PureOSQP.backend_name(setup(Diagonal(fill(2.0, 40)), q[1:40], randn(60, 40),
                            fill(-1.0, 60), fill(1.0, 60)).linsys)
```

### Supplying a matrix type of your own

`P` and `A` are held by reference and reached through a small set of functions, so a
representation the package has never heard of works by declaring itself
`<: AbstractMatrix{T}` and supplying `size`, `mul!`, and `mul!` against its adjoint. That
much is enough to solve. Everything below is optional and each override replaces one generic
walk over entries with whatever the representation can answer more cheaply.

There are two seam levels, and which one a representation wants depends on whether it can
enumerate a column.

**Per column.** [`PureOSQP.structural_rows`](@ref)`(M, j)` names the rows column `j` can hold
a nonzero in; the four traversals in `src/scaling.jl` — `weighted_colmax`,
`weighted_colmax_rowmax!`, `scaled_col!` and `add_scaled_col!` — follow it, so a single
`structural_rows` method makes equilibration and the dense formation cost the column's own
entries rather than all `m` of them. A representation whose columns are cheaper to walk than
to index overrides the four traversals directly instead; the sparse extension does that,
because `M[i, j]` on a `SparseMatrixCSC` is a binary search.

**Per sweep.** `column_norms!` and `cost_norms!` are the whole of what equilibration asks
per sweep, so a representation that answers in whole-matrix or closed form overrides those
two and never sees a column index. The GPU extension is the shipped example: it replaces
both with array reductions, which is what lets a device array equilibrate under
`allowscalar(false)`.

Beyond equilibration there are three more override points, all optional:
`PureOSQP.reduced_diagonal!` for the matrix-free preconditioner,
[`PureOSQP.is_convex`](@ref) for the convexity test `setup` runs before choosing a backend,
and [`PureOSQP.is_symmetric`](@ref) for the symmetry check — the last two both densify or
scan `n²` positions otherwise.

[`PureOSQP.RowCoupled`](@ref) is the worked example in the package itself: a few dense rows
above a block holding one entry per row. It defines `size`, `getindex` and `mul!`, and adds
one `structural_rows` method; that is all it takes for a `Diagonal` `P` with a `RowCoupled`
`A` to reach the low-rank backend and to equilibrate at the cost of its own entries.

An operator that supplies **only** products — nothing to index at all — says so with
[`PureOSQP.is_materializable`](@ref):

```julia
PureOSQP.is_materializable(::MyOperator) = false
```

`linsys = :auto` then declines the dense terminal and lands on the matrix-free backend, which
needs Krylov.jl loaded. `polish!` and the two derivative entry points build a dense matrix
out of `P` and `A` entry by entry, so they refuse such an operator by name rather than
failing inside a factorization: pass `polish = false`, and differentiate a materialized form
of the problem. Equilibration also walks columns, so an operator that overrides neither seam
level needs `scaling = 0`.

The hot-path guarantees carry a condition here that they do not carry elsewhere. `admm_step!`
allocates nothing and is type-stable for a caller-supplied operator only as far as that
operator's own `mul!` is: a broadcast in it, or a `DimensionMismatch` message built from a
type, is enough to lose both. `bench/lazy_operator.jl` is written to hold them, and
`bench/strictmode_audit.jl` checks it.

### An operator from LinearMaps.jl

**Use this when your constraint is something you can *do* but would never want to *store*.**

The situation is common in signal and image work. "Take a running total." "Blur this." "Take a
Fourier transform, keep the low frequencies." Each of those is a perfectly good linear
constraint, and each has a matrix — but for a million-pixel image that matrix has `10¹²`
entries and cannot exist. What you have instead is a function that applies it.

[LinearMaps.jl](https://github.com/JuliaLinearAlgebra/LinearMaps.jl) is the standard Julia
package for exactly that: an object you can multiply by, built from a function. Load it and
this solver accepts one anywhere it accepts a matrix. Nothing else is needed — matrices and
maps can even be mixed in the same call.

#### When this is the right tool

Four situations, in rough order of how often they come up:

1. **The matrix will not fit.** Deblurring a 1000×1000 image is a million variables, so `A` is
   a million by a million: `8` terabytes dense. There is no trade to weigh here — an operator
   is the only way the problem exists at all. Same story for 3-D grids, large PDE-constrained
   problems, and anything where `n` runs past `10⁵`.
2. **Applying it is much cheaper than its size suggests.** A convolution or blur is a *dense*
   matrix — every output touches every input — but applying it through an FFT costs
   `O(n log n)` instead of `O(n²)`. Storing it throws that away. The same holds for any
   transform with a fast algorithm: DCT, wavelets, a fast multipole method.
3. **You already have the code, not the entries.** The operator is a simulator, an existing
   forward model, a PDE solve, a linearization somebody else wrote. You can call it; nobody
   ever assembled it, and assembling it would mean `n` separate calls.
4. **Memory is the binding constraint, not time.** The matrix-free path stores vectors where
   the direct path stores an `n×n` inverse — [33× less at `n = 4000`](@ref "The matrix-free
   backend"). If the problem does not fit in RAM, being slower is not the issue.

#### When it is the wrong tool

**If applying your operator costs about what the dense product costs, use the matrix.** Being
matrix-free is not free: it trades a factorization you pay once for an iterative solve you pay
every iteration. When the product itself is no cheaper, that trade only loses:

| n | iterations | operator | matrix | |
|---|---|---|---|---|
| 200 | 125/125 | 6.0 ms | 2.2 ms | **2.7× slower** |
| 500 | 150/125 | 56.1 ms | 20.8 ms | **2.7× slower** |
| 1000 | 175/175 | 220 ms | 131 ms | **1.7× slower** |

That is an operator built from `O(n)` stored numbers — but sitting beside a dense `A` that both
routes must multiply by, so the cheap part was never where the cost was.

Contrast it with the table in [Which representation, and why](@ref), where an operator applied
in `O(n)` against an `O(n²)` dense form wins by 1.55–2.73× at the same sizes. **Size alone does
not decide this, and neither does whether the matrix fits.** The question is whether applying
your operator is asymptotically cheaper than multiplying by its dense form. If it is not, the
matrix wins at every size.

The second way to get this wrong is conditioning. A bare `LinearMap` has no structure the
solver can exploit, so it is served by conjugate gradients, which struggles as conditioning
worsens — and this is not a small effect: on the badly conditioned sweep in
[Benchmarks](@ref "Conditioning") the matrix-free backend fails to converge at *every* κ tested,
including mild ones. If your problem is ill-conditioned, a bare map is the wrong shape; give
the solver a structured type with a direct backend instead
([Unmaterialized does not mean solved by CG](@ref)).

Building one takes two functions: how to apply it, and how to apply its transpose. The
transpose is not optional; the solver needs both directions.

```@example linearmaps
using PureOSQP, LinearMaps, LinearAlgebra, Krylov, Random
Random.seed!(4)

n, m = 60, 40
# The constraint: scale each entry by w, take a running total, keep the first m.
# `forward` applies it; `adjoint_` applies its transpose. No m×n array is ever built.
w = 0.5 .+ rand(n)
forward(y, x) = (y .= cumsum(w .* x)[1:m])
function adjoint_(x, y)
    fill!(x, 0.0)
    x[1:m] .= y
    reverse!(x); cumsum!(x, x); reverse!(x)
    x .*= w
    return x
end
A = LinearMap{Float64}((y, x) -> forward(y, x), (x, y) -> adjoint_(x, y), m, n)

# The traits are declared, not inferred -- see below.
P = LinearMap(Diagonal(fill(2.0, n)); issymmetric = true, isposdef = true)
q = randn(n)
l, u = fill(-1.0, m), fill(1.0, m)

sol = PureOSQP.solve(P, q, A, l, u; scaling = 0, linsys = :indirect)
(sol.status, sol.iter, round(sol.obj_val; digits = 6))
```

That call needed three things beyond the map itself. Each will bite you if you skip it, so
here they are with what goes wrong.

**1. `using Krylov`.** An operator has no entries, so none of the usual backends can factor
anything. The only one that works is the matrix-free one, which multiplies instead of
factoring — and it lives in Krylov.jl. Without it loaded you get an error naming the remedy.
You do not have to pass `linsys = :indirect`; the solver finds it on its own. It is written
above only to make the requirement visible.

**2. `scaling = 0`.** By default the solver rescales your problem for numerical health, which
means reading down each column of `A` to find its largest entry. A map has no columns to read.
Passing `scaling = 0` turns that step off. If you forget, `setup` throws and says so — it does
not silently skip the rescaling.

**3. Declaring `issymmetric` and `isposdef` on `P`.** This is the one that surprises people.
Write `LinearMap(Diagonal(fill(2.0, n)))` — obviously a positive-definite matrix — and ask it,
and it says `isposdef == false`. LinearMaps does not inspect what you gave it; it reports only
what you *told* it. So the solver sees an objective not claiming to be convex and refuses it.

That refusal is correct, not a bug: the solver cannot factor an operator to check, so an
unclaimed property is an unknown one. Declare them at construction, as in the example. (Or
build the wrapper yourself with `ProductOperator{T}(map; symmetric, posdef)` if you want to
override what a map claims.)

That third point is also what LinearMaps buys you over writing an operator by hand: those two
declarations travel with the map, so [`PureOSQP.is_convex`](@ref) is answered by reading a flag
instead of factoring a matrix.

**One thing to expect: a map runs without a preconditioner.** A preconditioner is a cheap
approximation of the problem that makes the iteration converge faster, and the one used here is
built from the diagonal of the reduced matrix. A map has no entries, so there is no diagonal to
read, and the solver proceeds without one. Concretely: setup gets *cheaper* (nothing to build)
and each iteration gets **1.36–1.51× dearer**, measured on the same operator written both ways
([Benchmarks](@ref "An operator that is never materialized")).

Usually you just accept that. If the iteration count matters, give your map's type a
`PureOSQP.structural_rows` method — one method, described under
[Structured operators](@ref "2. `structural_rows` — setup stops paying for the zeros"), which
recovers the preconditioner *and* lets you drop `scaling = 0`. Setting `probe = true` is not a
substitute; probing answers the rescaling question, not this one.

## Structured operators the package ships

`Diagonal` and `Bidiagonal` above are LinearAlgebra's. This package ships three more matrix
types of its own, for three shapes that come up constantly and that LinearAlgebra has no type
for.

**Start here: which one, if any, is yours?**

| if your problem is… | use | typical source |
|---|---|---|
| many small independent sub-problems, side by side | [`PureOSQP.BlockDiagonal`](@ref) | one QP per time step, per asset, per scenario — anything that would be separate problems if they did not share a solve |
| mostly independent, but with a *few* rows tying everything together | [`PureOSQP.RowCoupled`](@ref) | box constraints plus a handful of budget or total-mass rows |
| a constraint applied across two dimensions at once | [`PureOSQP.KroneckerOperator`](@ref) | a 2-D grid, an image, space × time — where the constraint is "this in one direction, that in the other" |
| none of these | nothing to do | pass ordinary matrices; the solver is still fast |

If none of the rows fit, you have lost nothing by reading — these are optimizations, not
requirements, and the dense path gives the same answers.

Each one is used the same way: build it, pass it to [`setup`](@ref) exactly where you would
have passed a matrix, and check `backend_name` to confirm it was picked up. You never
configure anything.

### Block-diagonal

**Use it when your problem is really several smaller problems side by side.** Four machines
scheduled independently, twelve months priced independently, a hundred scenarios — anything
where variable 3 never appears in a constraint with variable 40.

The payoff is large and worth understanding, because it is why this type exists. Solving one
`n×n` system costs about `n³`. Solving `K` systems of size `n/K` costs `K(n/K)³ = n³/K²`. At
`K = 10` that is a hundred times less work, and a tenth of the memory. The solver gets that
automatically once it can *see* the blocks — which is what [`PureOSQP.BlockDiagonal`](@ref) is
for. Handed the same numbers as one big dense matrix, it cannot see them, and pays the `n³`.

Store it as a vector of the blocks. `P` and `A` must split at the same places, since a block
of the problem is only independent if both halves agree it is.

```@example blocks
using PureOSQP, LinearAlgebra

blocks_P = [Matrix(Symmetric([2.0 0.3; 0.3 2.0])) for _ in 1:4]
blocks_A = [[1.0 -1.0; 0.5 1.0] for _ in 1:4]
P = PureOSQP.BlockDiagonal(blocks_P)
A = PureOSQP.BlockDiagonal(blocks_A)

n, m = size(P, 1), size(A, 1)
q = collect(range(-1.0, 1.0; length = n))
ws = setup(P, q, A, fill(-1.0, m), fill(1.0, m))
PureOSQP.backend_name(ws.linsys)
```

`backend_name` returned `:block`, so the blocks were found. Here is what that saved, counted in
numbers stored:

```@example blocks
(blocks = PureOSQP.backend_info(ws.linsys).factor_nnz, dense = n * (n + 1) ÷ 2)
```

Four blocks of two variables each store 12 numbers where the dense route stores 36. The gap
widens fast: at 20 blocks of 12 it is 1 560 against 28 920, and the timings are in
[Benchmarks](@ref "Block-diagonal structure").

**If `backend_name` comes back `:cholesky` instead**, the blocks were not used. The usual
reason is that `P` and `A` split at different places, so the problem does not actually
decouple; the answer is still correct, just computed the slow way.

### Kronecker

**Use it when a constraint acts on two dimensions at once.** The clearest case is a 2-D grid:
you want something smoothed along rows *and* along columns. Written out, that constraint matrix
is enormous and almost all zeros. Written as `A₁ ⊗ A₂` — "`A₁` across, `A₂` down" — it is two
small matrices.

The saving in storage is immediate: a `6×6` constraint below is stored as `4 + 9 = 13` numbers
instead of 36, and that ratio grows as the square. The saving in time is larger, because the
backend never builds the big matrix at all.

**This one has conditions, and they are strict.** It is the fussiest type in the package, so
check them before reaching for it. All three are properties of your problem, not settings you
can turn on:

| condition | in plain terms | how to check |
|---|---|---|
| `P` must be `μI` — a single number times the identity | your objective weights every variable equally, or there is no objective at all | `P isa Diagonal && allequal(P.diag)` |
| `ρ` must be one number | every constraint is an inequality — no equalities | you passed no row with `l[i] == u[i]` |
| `scaling = 0` | you turn equilibration off explicitly | pass `scaling = 0` to `setup` |

The second is the one that catches people: **a single equality row disables this backend.** And
a Kronecker *`P`* does not qualify for the first — it must be a multiple of the identity.

If any condition fails the solver quietly uses the dense route instead, so you get the right
answer either way. That is why every example here checks `backend_name`: it is the only way to
tell whether you got what you asked for.

```@example kron
using PureOSQP, LinearAlgebra

A1 = [1.0 0.5; -0.5 1.0]
A2 = [2.0 0.0 1.0; 0.0 1.5 0.0; 1.0 0.0 2.0]
A = PureOSQP.KroneckerOperator(A1, A2)      # 6×6, stored as 4 + 9 entries

n = size(A, 2)
P = Diagonal(fill(2.0, n))                  # μI, as the tier requires
q = collect(range(-1.0, 1.0; length = n))
ws = setup(P, q, A, fill(-1.0, n), fill(1.0, n); scaling = 0)
PureOSQP.backend_name(ws.linsys)
```

Break any one condition and the rung declines — the problem is solved by the dense terminal
instead, more slowly and just as correctly:

```@example kron
equilibrated = setup(P, q, A, fill(-1.0, n), fill(1.0, n))   # scaling left at its default
nonscalar = setup(Diagonal(1.0:n), q, A, fill(-1.0, n), fill(1.0, n); scaling = 0)
(equilibrated = PureOSQP.backend_name(equilibrated.linsys),
 nonscalar = PureOSQP.backend_name(nonscalar.linsys))
```

#### Ill-conditioned Kronecker problems

Giving up equilibration is the price of this tier, and an ill-conditioned problem is exactly
where equilibration earns its keep — so that is where the trade has to be judged. Note that
`κ(A₁ ⊗ A₂) = κ(A₁)·κ(A₂)`, so each factor carries the square root of the figure below.

**The backend stays sound.** Against a dense path given the same `scaling = 0`, so the
comparison is the backends' and nothing else, it matches iteration for iteration and agrees on
the solution up to `κ(A) ≈ 10¹⁶` (`bench/results/kronecker_conditioning.json`):

| κ(A) | kronecker | dense, also unscaled | solutions agree |
|---|---|---|---|
| 1e2 | SOLVED, 175 | SOLVED, 175 | yes |
| 1e8 | SOLVED, 450 | SOLVED, 450 | yes |
| 1e12 | SOLVED, 1100 | SOLVED, 1100 | yes |
| 1e16 | SOLVED, 2525 | SOLVED, 2525 | yes |

Iterations climb steeply with conditioning — 175 to 2525 — because nothing is preconditioning
the problem. That is the cost, and it is not hidden by the structure.

**Whether it still wins depends on size**, because the tier buys `O(n₁n₂(n₁+n₂))` per iteration
against a dense `O(n₁²n₂²)`, and that has to cover the extra iterations. Against a dense path
allowed its equilibration — the choice a caller actually faces — at `κ(A) = 10¹²`:

| n | kronecker (`scaling = 0`) | dense, equilibrated | speedup |
|---|---|---|---|
| 30 | 175 iter, 0.11 ms | 300 iter, 0.19 ms | 1.7× |
| 168 | 350 iter, 0.61 ms | 575 iter, 3.27 ms | 5.4× |
| 480 | 875 iter, 3.98 ms | 500 iter, 27.25 ms | 6.9× |

At `n = 480` the tier takes 1.75× the iterations and still finishes 6.9× sooner. The iteration
counts are noisy in both directions — ADMM's trajectory is sensitive to scaling — so read the
times rather than the ratio of counts.

If your `P` is zero rather than `μI`, equilibration and the structure are compatible in
principle: a Kronecker product's row and column ∞-norms are exactly the Kronecker products of
the factors' norms, so equilibrating each factor would preserve the diagonalization. That
route is not built.

### Low-rank coupling

**Use it when almost every constraint touches one variable, and only a handful touch many.**
This is extremely common and easy to miss. A portfolio with a bound on each holding plus one
row saying "the weights sum to 1". A schedule with a limit per machine plus two rows for total
capacity. A design with a box on each parameter plus a budget.

Written as an ordinary matrix, those few dense rows make the whole thing look dense, and the
solver pays as if every constraint coupled everything. [`PureOSQP.RowCoupled`](@ref) separates
the two kinds so it can charge you only for the coupling rows you actually have.

It takes three arguments, in this order:

1. `coupling` — the few dense rows, as a `k×n` matrix. These are the rows that touch many
   variables.
2. `weights` — one number per single-entry row.
3. `cols` — which variable each of those rows refers to.

So `RowCoupled(C, ones(n), 1:n)` means "these `k` dense rows, then a plain bound on each of the
`n` variables".

```@example rowcoupled
using PureOSQP, LinearAlgebra

n = 24
coupling = reshape(collect(range(0.1, 0.8; length = 2n)), 2, n)   # two dense rows
A = PureOSQP.RowCoupled(coupling, ones(n), collect(1:n))          # then a bound per variable
P = Diagonal(fill(1.5, n))
q = collect(range(-1.0, 1.0; length = n))
m = size(A, 1)
ws = setup(P, q, A, fill(-1.0, m), fill(1.0, m))
PureOSQP.backend_name(ws.linsys)
```

`:lowrank` means it worked. The cost is `O(nk)` instead of `O(n²)`, so it wins by more the
fewer coupling rows you have: at one coupling row in 2000 variables it is
[**923× faster**](@ref "Low-rank structure") than the dense route.

**One condition:** the coupling rows have to be a small fraction of the variables — the backend
declines once `10k > n`. Two coupling rows therefore need at least 20 variables, which is why
`n = 24` above. Below that threshold the correction costs more than the dense solve it would
replace, so declining is the right answer. `P` must also be `Diagonal`.

