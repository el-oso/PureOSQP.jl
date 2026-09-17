# How a backend is chosen

Every solve factors a linear system, and which one it factors is decided once, in `setup`.
The choice depends on three things: what the caller asked for, what types `P` and `A` are,
and which algorithm is running. This page is the map.

## The two systems

Both algorithms need the same solve, `(x, z)` from `(b_x, b_z)`, and there are two forms of
it.

The **augmented** system keeps the constraints as rows:

```math
\begin{bmatrix} \tilde P + \sigma I & \tilde A^\top \\ \tilde A & -\operatorname{diag}(w)^{-1} \end{bmatrix}
\begin{bmatrix} x \\ \nu \end{bmatrix} =
\begin{bmatrix} b_x \\ b_z \end{bmatrix}
```

The **reduced** system eliminates them:

```math
\left( \tilde P + \sigma I + \tilde A^\top \operatorname{diag}(w) \tilde A \right) x = b_x + \tilde A^\top (w \odot b_z)
```

The reduced form is `n×n` rather than `(n+m)×(n+m)` and is what every structured backend
exploits, because structure in `P` and `A` survives into it. It has one cost: forming
`Ãᵀ diag(w) Ã` mixes the weights into the matrix. That is harmless while the weights stay in
a narrow band, and it is not harmless when they do not — which is the difference between the
two algorithms below.

## The decision

```text
setup(P, q, A, l, u, algorithm; linsys)
│
├── linsys names a backend ───► named_backend
│                               ├── the pair admits it ──────► build it
│                               └── it does not ─────────────► throw, naming the condition
│
└── linsys = :auto ───────────► choose_backend(P, A, prob, wt, selection)
                                ├── a method for this (P, A) ► that backend
                                │   pair exists                (Kronecker, banded, block, …)
                                └── none exists ─────────────► select_backend: descend the
                                                               ladder for this algorithm,
                                                               taking the first rung that
                                                               serves the pair
                        ▼
                    factorize!
                        ├── succeeds ───► this is the workspace's backend, and its type
                        │                 from here on
                        └── fails ──────► rebuild on FullKKT, or throw if that is what
                                          already failed
```

`linsys` is an instruction rather than a hint: a named backend that the pair does not admit
is refused with the condition it failed, not silently replaced. The one exception is the last
step — a factorization that fails is rebuilt on the full KKT system, which is the only
selection decision made after `setup` has already chosen.

## The ladders

The ladder is the fallback for a pair with no `choose_backend` method of its own. Each rung
is a function that serves the pair or declines, and the order is fixed in one place. The two
algorithms share the rungs and differ in which they descend.

```text
   rung            what it builds              OperatorSplitting   InteriorPoint
   ────────────────────────────────────────────────────────────────────────────────
   kkt_rung        sparse, augmented                   1                 1
   reduced_rung    sparse, reduced                     2                 2
   kronecker_rung  two eigenbases and a diagonal       3                 ·
   block_rung      one factor per block                4                 3
   lowrank_rung    diagonal plus rank k                5             declines
   formed_rung     the reduced inverse, reused         6                 ·
   dense_rung      the terminal                        7                 4
                                              ReducedCholesky        FullKKT
   indirect_rung   conjugate gradients                 8                 5
```

A `·` is a rung the interior-point ladder does not descend at all. Three entries differ, and
each for a reason that comes from the weights.

**The Kronecker rung is absent under the interior-point method.** Diagonalizing `A₁ ⊗ A₂`
needs one weight for every row, and the interior-point weights differ from row to row from
the first iteration.

**`formed_rung` is absent.** It inverts the reduced matrix once and applies the inverse
thereafter, which pays over the hundreds of solves an ADMM run takes. An interior-point run
takes a handful of solves per factorization, so the inverse would be rebuilt almost as often
as it is used.

**`lowrank_rung` declines.** This is the reduced form's cost, arriving. An active row's weight
reaches `1/δ_d`, so the rank-`k` correction sits orders of magnitude above the diagonal core
it corrects and the small directions are rounded away as the matrix is formed. Measured
against a reference in extended precision, the reduced matrix reaches about `1e-8` where the
augmented factorization reaches `1e-15`; solving the reduced matrix through the Woodbury
identity rather than densely does not change that, because the loss happens when the matrix is
built (`PureIPM/bench/ipm_lowrank_terminal.jl`).

That is also why the terminal differs: `ReducedCholesky` for operator splitting,
[`FullKKT`](@ref) for the interior-point method. The first reduces and the second does not.

## What the sparse rungs ask

The two sparse rungs do not decide for themselves. Both put one question to the sparsity
pattern — `sparse_form(P, A, n, m, selection)` — and act on its single answer:

```text
   the pattern of P and A ──► sparse_form ──┬── :kkt ─────► factor the augmented system
                                            │               sparsely
                                            ├── :reduced ─► factor the reduced system
                                            │               sparsely
                                            └── :none ────► decline, and the ladder
                                                            continues below
```

The rule reads the pattern and nothing else — how many nonzeros the densest row of `A` holds,
and the sum of the squared row counts, which is what the reduced matrix's fill costs. Values
never enter it, so a problem's backend does not change when its numbers do, and `setup` can
factor once with the values the solve will use.

A row spanning every variable fills the reduced matrix by itself, so a pattern holding one
sends both algorithms to the augmented form. That is the shape of a budget constraint, and it
is why the OSQP suite's Portfolio class factors the `(n+m)` system.

`sparse_form` answers for `SparseMatrixCSC` pairs. A structured type that is not stored as CSC
never reaches it, and descends past both sparse rungs to the rungs that dispatch on its own
type.

## Asking what was chosen

```julia
ws = setup(P, q, A, l, u, OperatorSplitting())
backend_name(ws.linsys)      # :cholesky, :sparse_formed, :banded, ...
backend_info(ws.linsys)      # the form, its dimension, and what the factor stores
factor_fill(ws)              # that store, against n²
```

[`recommend_linsys`](@ref) goes further and measures: it builds every backend the pair admits,
runs a bounded number of iterations on each, and ranks them by the cost of a whole solve. The
ladder is fitted to a benchmark suite, and `recommend_linsys` is how a problem it misjudges
gets a second opinion.
