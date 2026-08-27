using LinearAlgebra, SparseArrays, OSQP

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
