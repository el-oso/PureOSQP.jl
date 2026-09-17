using LinearAlgebra, SparseArrays

"""
Backends that form the reduced matrix sparsely *and* factor it sparsely.

Which one the ladder picks depends on what is loaded: LDLFactorizations supplies the faster
factorization when it is available, and SparseArrays' CHOLMOD serves otherwise. A test that
cares the reduced matrix was factored sparsely means either, so it asks for membership here
rather than naming one; the answer itself is checked against a dense reference, which pins
whichever engine actually ran.
"""
const SPARSE_FACTOR_BACKENDS = (:cholmod, :ldlfactorizations)

"The same pair for the full quasi-definite KKT system, which has its own two engines."
const SPARSE_KKT_BACKENDS = (:sparse_kkt, :ldl_kkt)

"""
    backend_for(P, q, A, l, u; rho = 0.1, sigma = 1e-6, scaling = 10, linsys = :auto,
                selection = ADMMSelection(), preconditioner = nothing)
        -> (prob, weights, linsys)

The problem, the weights and the backend a solver would hold for this data, built without
one. A named `linsys` is honored the way a caller naming it is; `:auto` descends the ladder.
The backend comes back factorized, so a test can solve through it immediately.

`rho` is uniform. An algorithm assigns it per row — a larger weight on the rows it treats as
equalities — which changes the numbers in the reduced matrix but not which backend forms it
or how it is factored, and those are what this package answers for.
"""
function backend_for(
        P, q, A, l, u; rho = 0.1, sigma = 1.0e-6, scaling = 10, linsys::Symbol = :auto,
        selection = PureQPBase.ADMMSelection(), preconditioner = nothing
    )
    n, m = PureQPBase.validate(P, q, A, l, u)
    prob = PureQPBase.validated_problem(Float64, n, m, P, q, A, l, u, scaling)
    wt = raw_weights(fill(rho, m), sigma)
    named = PureQPBase.named_backend(Val(linsys), P, A, prob, wt, selection, preconditioner)
    ls, factored = if isnothing(named)
        PureQPBase.choose_backend(P, A, prob, wt, selection)
    else
        named
    end
    factored || PureQPBase.factorize!(ls, prob, wt)
    return (prob, wt, ls)
end

"""
    reduced_matrix(P, A, wt) -> Matrix

`P + σI + Aᵀ diag(w) A`, densely, from the matrices as given. This is what every backend
solves against, however it chooses to represent it.
"""
reduced_matrix(P, A, wt) =
    Matrix(P) + wt.sigma * I + Matrix(A)' * Diagonal(wt.w) * Matrix(A)

"""
    kkt_matrix(P, A, wt) -> Matrix

The full quasi-definite system `[P + σI  Aᵀ; A  -diag(1/w)]`, densely. Solving it against
`[bx; bz]` gives the `(x, z)` a backend's [`solve_system!`](@ref) must reproduce.
"""
kkt_matrix(P, A, wt) = [
    Matrix(P) + wt.sigma * I Matrix(A)'
    Matrix(A) -Diagonal(one.(wt.w) ./ wt.w)
]

"""
    raw_problem(P, A, n, m; D = ones(n), E = ones(m), c = 1.0) -> PureQPBase.Problem

A `Problem` built directly from these fields, without the validation or equilibration
[`validated_problem`](@ref) performs — for exercising one selection-ladder rung against
matrices that do not support those operations.
"""
function raw_problem(P, A, n::Integer, m::Integer; D = ones(n), E = ones(m), c = 1.0)
    zn, zm = zeros(n), zeros(m)
    return PureQPBase.Problem(
        P, A, n, m, zn, zm, zm, copy(zn), copy(zm), copy(zm),
        collect(D), collect(E), c, 0, copy(zn), copy(zm), copy(zn), copy(zm),
    )
end

"A `SystemWeights` built directly from `rho` and `sigma`, for the same purpose."
raw_weights(rho, sigma) = PureQPBase.SystemWeights(rho, inv.(rho), sigma)

"""
    banded_qp(n, m; band = 3, seed = 0) -> (P, q, A, l, u)

A convex QP whose matrices are banded, as in model-predictive control: each constraint row
couples a contiguous run of variables. The reduced matrix inherits the structure, which is
what a sparse factorization needs in order to pay.
"""
function banded_qp(n, m; band = 3, seed = 0)
    Random.seed!(n + m + band + seed)
    rows, cols, vals = Int[], Int[], Float64[]
    for i in 1:m, j in max(1, div(i * n, m) - band):min(n, div(i * n, m) + band)
        push!(rows, i)
        push!(cols, j)
        push!(vals, randn())
    end
    A = sparse(rows, cols, vals, m, n)
    S = spdiagm(-1 => randn(n - 1), 0 => randn(n), 1 => randn(n - 1))
    P = sparse(Symmetric(S'S)) + 3.0I
    b = A * randn(n)
    return (P, randn(n), A, b .- rand(m), b .+ rand(m))
end

"""
    random_qp(n, m; seed = 0, colscale = 0) -> (P, q, A, l, u)

A random convex QP. `colscale` spreads the column norms of `A` over `10^±colscale`, which is
what equilibration is asked to undo.
"""
function random_qp(n, m; seed = 0, colscale = 0)
    Random.seed!(seed)
    X = randn(n, n)
    P = X'X / n + I
    q = randn(n)
    A = randn(m, n)
    iszero(colscale) || (A = A * Diagonal(exp10.(range(-colscale, colscale; length = n))))
    x0 = randn(n)
    Ax = A * x0
    return (P, q, A, Ax .- rand(m), Ax .+ rand(m))
end
