@enum LDPStatus LDP_OPTIMAL LDP_INFEASIBLE LDP_ITERATION_LIMIT LDP_CYCLED

"""
State for one least-distance problem `min ‖u‖² s.t. Mu ≤ d`, with rows flagged
equality held permanently in the working set.

`active` lists the working-set rows in the order they entered, which is the
order the `LDLᵀ` of `Mₐ Mₐᵀ` was built in; `where_in_active[i]` is where row `i`
sits in it, or 0. Every buffer is sized once.
"""
mutable struct LDPWorkspace{T <: Real}
    M::Matrix{T}
    d::Vector{T}
    iseq::Vector{Bool}             # rows that must stay in the working set
    active::Vector{Int}
    where_in_active::Vector{Int}
    lambda::Vector{T}              # multipliers of the active rows, in `active` order
    lambda_star::Vector{T}
    p::Vector{T}                   # search direction in multiplier space
    mu::Vector{T}                  # reduced costs of the inactive rows
    u::Vector{T}                   # primal point of the LDP, Mₐᵀλ
    g::Vector{T}                   # scratch: products against the active rows
    F::GramLDL{T}
end

function LDPWorkspace(M::Matrix{T}, d::Vector{T}, iseq::AbstractVector{Bool}) where {T <: Real}
    m, n = size(M)
    kmax = min(m, n) + 1
    return LDPWorkspace{T}(
        M, d, collect(Bool, iseq), Int[], zeros(Int, m), T[], zeros(T, kmax), zeros(T, kmax),
        zeros(T, m), zeros(T, n), zeros(T, kmax), GramLDL{T}(kmax)
    )
end

"Put row `r` into the working set, extending the factorization."
function activate!(ws::LDPWorkspace{T}, r::Integer) where {T}
    k = ws.F.k
    m_r = view(ws.M, r, :)
    g = view(ws.g, 1:k)
    for (j, a) in enumerate(ws.active)
        g[j] = dot(view(ws.M, a, :), m_r)
    end
    add_row!(ws.F, g, dot(m_r, m_r))
    push!(ws.active, r)
    push!(ws.lambda, zero(T))
    ws.where_in_active[r] = length(ws.active)
    return ws
end

"Take the `i`-th working-set row back out."
function deactivate!(ws::LDPWorkspace, i::Integer)
    r = ws.active[i]
    remove_row!(ws.F, i)
    deleteat!(ws.active, i)
    deleteat!(ws.lambda, i)
    ws.where_in_active[r] = 0
    for j in i:length(ws.active)
        ws.where_in_active[ws.active[j]] = j
    end
    return ws
end

"""
    step_and_drop!(ws, p, k) -> Bool

Move `λ` along `p` until the first inequality multiplier in the working set hits
zero, then drop that row. `false` when nothing blocks, which for the singular
branch means the problem is infeasible.

Only inequality rows block: an equality multiplier is free in sign, so it never
stops the step.
"""
function step_and_drop!(ws::LDPWorkspace{T}, p::AbstractVector{T}, zero_tol::T) where {T}
    alpha = typemax(T)
    blocking = 0
    for i in eachindex(ws.active)
        (ws.iseq[ws.active[i]] || p[i] >= -zero_tol) && continue
        cand = -ws.lambda[i] / p[i]
        if cand < alpha
            alpha = cand
            blocking = i
        end
    end
    iszero(blocking) && return false
    for i in eachindex(ws.lambda)
        ws.lambda[i] += alpha * p[i]
    end
    deactivate!(ws, blocking)
    return true
end

"""
    solve_ldp!(ws; max_iter, zero_tol, primal_tol) -> (status, iterations)

Algorithm 1 of Arnström, Bemporad & Axehill, *A dual active-set solver for
embedded quadratic programming using recursive LDLᵀ updates*, IEEE TAC 2022:
a dual active-set method for `min ‖u‖² s.t. Mu ≤ d`.

Each iteration solves `Mₐ Mₐᵀ λ = -dₐ` through the maintained factorization. A
`λ` that is nonnegative on the inequality rows is dual feasible, and the run
stops once no inactive row is violated; otherwise the most violated row joins
the working set. A `λ` with a negative inequality entry means the step is
blocked, so the run walks toward it and drops the first row that reaches zero.
A singular Gram matrix is handled by walking along its null direction instead.
"""
function solve_ldp!(
        ws::LDPWorkspace{T}; max_iter::Int = 1000,
        zero_tol::T = sqrt(eps(T)), primal_tol::T = sqrt(eps(T))
    ) where {T}
    m = size(ws.M, 1)
    for iter in 1:max_iter
        k = ws.F.k
        singular = 0
        for i in 1:k
            if ws.F.D[i] <= zero_tol
                singular = i
                break
            end
        end

        if !iszero(singular)
            p = view(ws.p, 1:k)
            singular_direction!(p, ws.F, singular)
            step_and_drop!(ws, p, zero_tol) || return (LDP_INFEASIBLE, iter)
            continue
        end

        lam = view(ws.lambda_star, 1:k)
        for i in 1:k
            lam[i] = -ws.d[ws.active[i]]
        end
        solve_gram!(ws.F, lam)

        dual_feasible = true
        for i in 1:k
            if !ws.iseq[ws.active[i]] && lam[i] < 0
                dual_feasible = false
                break
            end
        end

        if !dual_feasible
            p = view(ws.p, 1:k)
            for i in 1:k
                p[i] = lam[i] - ws.lambda[i]
            end
            step_and_drop!(ws, p, zero_tol) || return (LDP_CYCLED, iter)
            continue
        end

        # Dual feasible: accept it, then price the inactive rows.
        copyto!(view(ws.lambda, 1:k), lam)
        fill!(ws.u, zero(T))
        for i in 1:k
            axpy!(ws.lambda[i], view(ws.M, ws.active[i], :), ws.u)
        end

        worst = zero(T)
        entering = 0
        for r in 1:m
            iszero(ws.where_in_active[r]) || continue
            mu_r = dot(view(ws.M, r, :), ws.u) + ws.d[r]
            ws.mu[r] = mu_r
            if mu_r < worst
                worst = mu_r
                entering = r
            end
        end

        (iszero(entering) || worst >= -primal_tol) && return (LDP_OPTIMAL, iter)
        activate!(ws, entering)
    end
    return (LDP_ITERATION_LIMIT, max_iter)
end

"""
The once-computed part of the reduction from a QP to a least-distance problem:
the Cholesky factor `R` of `H + εI`, the transformed and row-normalized
constraint matrix `M = A R⁻¹` with its right-hand side `b`, and the working
state. Only `d` changes between proximal-point iterations, so `R`, `M` and the
`LDLᵀ` of the working set are all reused.
"""
struct DAQPReduction{T <: Real, F <: Cholesky}
    R::F
    b::Vector{T}
    eqflag::Vector{Bool}
    eps_prox::T
    ws::LDPWorkspace{T}
    # How each least-distance row came from a problem row: `rows[j]` is the problem row,
    # `signs[j]` is +1 for its upper side and -1 for its lower, and `scale[j]` is the row
    # norm it was divided by. Mapping a multiplier back undoes both.
    rows::Vector{Int}
    signs::Vector{T}
    scale::Vector{T}
end

"""
    reduce_qp(H, f, A, bupper, blower, iseq; eps_prox) -> DAQPReduction

Factor `H + εI` and transform the constraints. Each two-sided row becomes two
one-sided rows, since the working set tracks which side is active.

With `eps_prox = 0` this needs `H ≻ 0`. Any `ε > 0` makes `H + εI` positive
definite for `H ⪰ 0`, which is what lets the proximal-point loop take a singular
`H` — an LP being the extreme case.
"""
function reduce_qp(
        H::AbstractMatrix{T}, f::AbstractVector{T}, A::AbstractMatrix{T},
        bupper::AbstractVector{T}, blower::AbstractVector{T},
        iseq::AbstractVector{Bool}; eps_prox::T = zero(T)
    ) where {T <: Real}
    Hs = Matrix(Symmetric((H + H') / 2))
    if !iszero(eps_prox)
        for i in axes(Hs, 1)
            Hs[i, i] += eps_prox
        end
    end
    R = cholesky(Symmetric(Hs))
    Au = A / R.U

    # One row per finite bound: `Ax ≤ u` directly, `-Ax ≤ -l` for the lower side.
    rows = Vector{Int}()
    signs = Vector{T}()
    for i in axes(A, 1)
        if iseq[i] || isfinite(bupper[i])
            push!(rows, i)
            push!(signs, one(T))
        end
        if !iseq[i] && isfinite(blower[i])
            push!(rows, i)
            push!(signs, -one(T))
        end
    end
    mm = length(rows)
    M = Matrix{T}(undef, mm, size(Au, 2))
    b = Vector{T}(undef, mm)
    eqflag = falses(mm)
    for (j, (i, s)) in enumerate(zip(rows, signs))
        @views M[j, :] .= s .* Au[i, :]
        b[j] = s > 0 ? bupper[i] : -blower[i]
        eqflag[j] = iseq[i]
    end
    # Row normalization, as the reference does: it makes `primal_tol` mean the
    # same thing on every row regardless of how that row was scaled. Scaling `M`
    # and `b` together keeps `d = b + M v` correct at every later iteration.
    scale = ones(T, mm)
    for j in axes(M, 1)
        nrm = norm(view(M, j, :))
        if nrm > 0
            @views M[j, :] ./= nrm
            b[j] /= nrm
            scale[j] = nrm
        end
    end

    ws = LDPWorkspace(M, similar(b), eqflag)
    for j in axes(M, 1)
        eqflag[j] && activate!(ws, j)
    end
    return DAQPReduction{T, typeof(R)}(
        R, b, collect(Bool, eqflag), eps_prox, ws, rows, signs, scale
    )
end

"""
    rebuild_rhs!(red, bupper, blower) -> red

Refresh the least-distance right-hand side after the caller changed `l` or `u`, keeping the
factorization and the transformed constraint matrix.
"""
function rebuild_rhs!(red::DAQPReduction{T}, bupper::AbstractVector{T}, blower::AbstractVector{T}) where {T}
    for j in eachindex(red.rows)
        i, s = red.rows[j], red.signs[j]
        red.b[j] = (s > 0 ? bupper[i] : -blower[i]) / red.scale[j]
    end
    return red
end

"""
    reset_working_set!(red) -> red

Drop every inequality row from the working set, keeping the equality rows, which must always
be in it. The next solve then starts from the same state a fresh setup would.
"""
function reset_working_set!(red::DAQPReduction)
    ws = red.ws
    while !isempty(ws.active)
        deactivate!(ws, length(ws.active))
    end
    for j in eachindex(red.eqflag)
        red.eqflag[j] && activate!(ws, j)
    end
    return red
end

"""
    multipliers!(y, red, lambda) -> y

Map the least-distance multipliers back to one multiplier per problem row. A row split into
two sides contributes the difference of the two, and each is divided by the norm its row was
scaled by, so `y` is the multiplier of the caller's own constraint.
"""
function multipliers!(y::AbstractVector{T}, red::DAQPReduction{T}, lambda::AbstractVector{T}) where {T}
    fill!(y, zero(T))
    for (slot, j) in enumerate(red.ws.active)
        y[red.rows[j]] += red.signs[j] * lambda[slot] / red.scale[j]
    end
    return y
end

"""
    run_daqp!(red, f; ...) -> (x, lambda, active, status, iters)

Algorithm 1, or Algorithm 2 wrapped around it when `eps_prox > 0`, on a reduction that
already exists. The working set is whatever the reduction currently holds, so a repeated
call warm-starts from the previous answer.
"""
function run_daqp!(
        red::DAQPReduction{T}, f::AbstractVector{T};
        max_iter::Int, zero_tol::T, primal_tol::T,
        eps_prox::T, eta_prox::T, max_prox::Int
    ) where {T}
    ws = red.ws
    n = size(ws.M, 2)
    x = zeros(T, n)
    if iszero(eps_prox)
        status, iters, v = inner_solve!(red, f, x; max_iter, zero_tol, primal_tol)
        status == LDP_OPTIMAL || return (T[], T[], ws.active, status, iters)
        return (red.R.U \ (-ws.u - v), copy(ws.lambda), copy(ws.active), status, iters)
    end
    xold = similar(x)
    total = 0
    for _ in 1:max_prox
        copyto!(xold, x)
        status, iters, v = inner_solve!(red, f, x; max_iter, zero_tol, primal_tol)
        total += iters
        status == LDP_OPTIMAL || return (T[], T[], ws.active, status, total)
        x = red.R.U \ (-ws.u - v)
        norm(x - xold, Inf) < eta_prox &&
            return (x, copy(ws.lambda), copy(ws.active), status, total)
    end
    return (x, copy(ws.lambda), copy(ws.active), LDP_ITERATION_LIMIT, total)
end

"""
    inner_solve!(red, f, x) -> (status, iters, v)

One pass of Algorithm 1 at the current proximal centre `x`: rebuild `d` from
`v = R⁻ᵀ(f − εx)`, then run the dual active-set loop warm-started from whatever
working set the previous pass left behind.
"""
function inner_solve!(
        red::DAQPReduction{T}, f::AbstractVector{T}, x::AbstractVector{T};
        max_iter::Int, zero_tol::T, primal_tol::T
    ) where {T}
    ws = red.ws
    rhs = iszero(red.eps_prox) ? f : f .- red.eps_prox .* x
    v = red.R.L \ rhs
    mul!(ws.d, ws.M, v)
    ws.d .+= red.b
    status, iters = solve_ldp!(ws; max_iter, zero_tol, primal_tol)
    return status, iters, v
end

"""
    solve_qp_daqp(H, f, A, bupper, blower, iseq; ...) -> (x, lambda, active, status, iters)

Solve `min ½xᵀHx + fᵀx s.t. blower ≤ Ax ≤ bupper`.

With `eps_prox = 0` this is Algorithm 1 alone and needs `H ≻ 0`. With
`eps_prox > 0` it runs Algorithm 2 of Arnström, Bemporad & Axehill: outer
proximal-point iterations solving
`min ½xᵀ(H+εI)x + (f−εxₖ)ᵀx` and stopping once `‖x − xₖ‖ < eta_prox`. That
accepts `H ⪰ 0`, and improves conditioning generally. The factorization and the
working set carry across outer iterations, so each one costs an `O(mn)` update
of `d` plus a warm-started inner solve.
"""
function solve_qp_daqp(
        H::AbstractMatrix{T}, f::AbstractVector{T}, A::AbstractMatrix{T},
        bupper::AbstractVector{T},
        blower::AbstractVector{T} = fill(-T(Inf), size(A, 1)),
        iseq::AbstractVector{Bool} = falses(size(A, 1));
        max_iter::Int = 1000, zero_tol::T = sqrt(eps(T)), primal_tol::T = sqrt(eps(T)),
        eps_prox::T = zero(T), eta_prox::T = sqrt(eps(T)), max_prox::Int = 100
    ) where {T <: Real}
    red = reduce_qp(H, f, A, bupper, blower, iseq; eps_prox)
    ws = red.ws
    n = size(H, 1)
    x = zeros(T, n)
    total = 0

    if iszero(eps_prox)
        status, iters, v = inner_solve!(red, f, x; max_iter, zero_tol, primal_tol)
        status == LDP_OPTIMAL || return (T[], T[], ws.active, status, iters)
        return (red.R.U \ (-ws.u - v), copy(ws.lambda), copy(ws.active), status, iters)
    end

    xold = similar(x)
    for _ in 1:max_prox
        copyto!(xold, x)
        status, iters, v = inner_solve!(red, f, x; max_iter, zero_tol, primal_tol)
        total += iters
        status == LDP_OPTIMAL || return (T[], T[], ws.active, status, total)
        x = red.R.U \ (-ws.u - v)
        norm(x - xold, Inf) < eta_prox && return (x, copy(ws.lambda), copy(ws.active), status, total)
    end
    return (x, copy(ws.lambda), copy(ws.active), LDP_ITERATION_LIMIT, total)
end
