using LinearAlgebra, SparseArrays, OSQP

"""
Backends that form the reduced matrix sparsely *and* factor it sparsely.

Which one `setup` picks depends on what is loaded: LDLFactorizations supplies the faster
factorization when it is available, and SparseArrays' CHOLMOD serves otherwise. A test that
cares the reduced matrix was factored sparsely means either, so it asks for membership here
rather than naming one; the answer itself is checked against the dense backend, which pins
whichever engine actually ran.
"""
const SPARSE_FACTOR_BACKENDS = (:cholmod, :ldlfactorizations)

"The same pair for the full quasi-definite KKT system, which has its own two engines."
const SPARSE_KKT_BACKENDS = (:sparse_kkt, :ldl_kkt)

"""
    kkt_residuals(P, q, A, l, u, x, y) -> (r_prim, r_dual, r_opt)

Optimality residuals of a candidate point, computed from the original problem data alone.
Nothing here touches solver internals, so it cannot inherit a scaling bug from the solver
it is judging.

`r_opt` is the relative duality gap, plus a sign check on the rows it cannot cover.
Complementarity is deliberately *not* measured as "`y_i` nonzero implies the corresponding
bound is active": at a first-order solution `y_i` is a tiny nonzero float on inactive rows,
which that test reads as a full-magnitude violation.

    gap = xᵀPx + qᵀx + SC(y),   SC(y) = uᵀ max(y, 0) + lᵀ min(y, 0)

A row whose relevant bound is infinite contributes no finite term to `SC(y)`; for such a
row the KKT condition is a sign condition on `y`, scored separately as a relative sign
violation. Folding it into `SC(y)` would require a `0 * Inf` convention that only holds
while `y` stays under whatever deadzone is chosen.
"""
function kkt_residuals(P, q, A, l, u, x, y)
    Ax = A * x
    z = clamp.(Ax, l, u)
    r_prim = isempty(Ax) ? zero(eltype(x)) : maximum(abs, Ax .- z)
    r_dual = maximum(abs, P * x .+ q .+ A' * y)
    quad = dot(x, P * x)
    lin = dot(q, x)
    sup = zero(eltype(x))
    sign_viol = zero(eltype(x))
    ny = maximum(abs, y; init = zero(eltype(y)))
    for i in eachindex(y)
        yi = y[i]
        iszero(yi) && continue
        bound = yi > 0 ? u[i] : l[i]
        if isfinite(bound)
            sup += bound * yi
        else
            sign_viol = max(sign_viol, abs(yi))
        end
    end
    gap = abs(quad + lin + sup)
    r_gap = gap / max(one(gap), abs(quad), abs(lin), abs(sup))
    r_sign = ny > 0 ? sign_viol / ny : zero(sign_viol)
    return (r_prim, r_dual, max(r_gap, r_sign))
end

"""
    osqp_ref(P, q, A, l, u; kwargs...)

Run the reference C implementation on the same problem. `adaptive_rho_interval` is always
pinned: libosqp 0.6.2 otherwise adapts on wall-clock time, which makes iteration counts
machine-dependent.
"""
function osqp_ref(P, q, A, l, u; kwargs...)
    model = OSQP.Model()
    OSQP.setup!(
        model; P = sparse(Symmetric(Matrix(P))), q = collect(q),
        A = sparse(Matrix(A)), l = collect(l), u = collect(u), verbose = false,
        adaptive_rho_interval = 50, check_termination = 25, kwargs...
    )
    return OSQP.solve!(model)
end

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

"""
    primal_infeasible_qp(n, m, seed) -> (P, q, A, l, u)

A random primal-infeasible QP with `m > n`: `y` spans a direction of the null space of `Aᵀ`,
and two-sided bounds around a point `A x₀` are shifted by `−c·y`, which lowers
`uᵀmax(y,0) + lᵀmin(y,0)` by `c‖y‖²` to below zero while `yᵀA x = 0` for every `x`.
"""
function primal_infeasible_qp(n, m, seed)
    rng = Xoshiro(seed)
    X = randn(rng, n, n)
    P = X'X / n + I
    A = randn(rng, m, n)
    y = nullspace(Matrix(A'))[:, 1]
    b = A * randn(rng, n)
    r = rand(rng, m) .+ 0.1
    c = 2 * sum(abs.(y) .* r) / dot(y, y)
    return (P, randn(rng, n), A, b .- r .- c .* y, b .+ r .- c .* y)
end

"""
    dual_infeasible_qp(n, m, seed) -> (P, q, A, l, u)

A random dual-infeasible (unbounded) QP: `P` has rank `n − 1` with null vector `v`,
`qᵀv = −1`, the first `m ÷ 4` rows of `A` are orthogonal to `v` and two-sided, and every
other row keeps only the bound that `v` moves away from.
"""
function dual_infeasible_qp(n, m, seed)
    rng = Xoshiro(seed)
    X = randn(rng, n, n - 1)
    P = Matrix(Symmetric(X * X' / n))
    v = nullspace(P)[:, 1]
    A = randn(rng, m, n)
    for i in 1:(m ÷ 4)
        A[i, :] .-= dot(A[i, :], v) .* v
    end
    b = A * randn(rng, n)
    l, u = b .- rand(rng, m) .- 0.1, b .+ rand(rng, m) .+ 0.1
    Av = A * v
    for i in (m ÷ 4 + 1):m
        Av[i] > 0 ? (u[i] = Inf) : (l[i] = -Inf)
    end
    q = randn(rng, n)
    q .-= (dot(q, v) + 1) .* v
    return (P, q, A, l, u)
end

"""
    is_primal_certificate(A, l, u, y; tol = 1e-6) -> Bool

Whether `y` proves `l ≤ Ax ≤ u` infeasible, from the data alone:
`uᵀmax(y,0) + lᵀmin(y,0) < 0` with every infinite bound meeting a zero entry, and
`‖Aᵀy‖∞ < tol‖y‖∞`.
"""
function is_primal_certificate(A, l, u, y; tol = 1.0e-6)
    s = 0.0
    for i in eachindex(y)
        y[i] > 0 && (s += u[i] * y[i])
        y[i] < 0 && (s += l[i] * y[i])
    end
    return isfinite(s) && s < 0 && norm(A' * y, Inf) < tol * norm(y, Inf)
end

"""
    is_dual_certificate(P, q, A, l, u, x; tol = 1e-6) -> Bool

Whether `x` proves the QP unbounded below, from the data alone: `qᵀx < 0`,
`‖Px‖∞ < tol‖x‖∞`, and `Ax` in the recession cone of `[l, u]` to within `tol‖x‖∞`.
"""
function is_dual_certificate(P, q, A, l, u, x; tol = 1.0e-6)
    nx = norm(x, Inf)
    Ax = A * x
    cone = all(eachindex(Ax)) do i
        (!isfinite(u[i]) || Ax[i] <= tol * nx) && (!isfinite(l[i]) || Ax[i] >= -tol * nx)
    end
    return nx > 0 && dot(q, x) < 0 && norm(P * x, Inf) < tol * nx && cone
end

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
