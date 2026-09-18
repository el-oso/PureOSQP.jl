@enum LDPStatus LDP_OPTIMAL LDP_INFEASIBLE LDP_ITERATION_LIMIT LDP_CYCLED

const SIDE_INACTIVE = Int8(0)
const SIDE_UPPER = Int8(1)
const SIDE_LOWER = Int8(-1)

"""
State for one two-sided least-distance problem, `lo ≤ Mu ≤ hi`.

A row is in the working set at one of its two bounds, and `side` records which. Its
multiplier is then signed: non-negative at the upper bound, non-positive at the lower one.
Working in those signed multipliers means the factored Gram matrix is over the *unsigned*
rows, so a row switching sides costs nothing — the factorization does not change.

That is also why there is one row here per problem row. Splitting each two-sided row into
two one-sided rows would double the length of the pricing loop, which is the dominant cost
of an iteration.

`active` lists the working set in the order it was built, matching the order of the `LDLᵀ`;
`slot[j]` is where row `j` sits in it, or 0.
"""
mutable struct LDPWorkspace{T <: Real}
    # Stored transposed, `n × m`: every kernel here reads one *row* of `M`, which as a
    # column of `Mt` is contiguous. Held the other way round each read is strided, BLAS
    # drops to its scalar path, and every element costs a cache line.
    Mt::Matrix{T}
    hi::Vector{T}          # upper target, recomputed whenever `v` changes
    lo::Vector{T}
    iseq::Vector{Bool}
    side::Vector{Int8}
    active::Vector{Int}
    slot::Vector{Int}
    mu::Vector{T}          # signed multipliers of the active rows, in `active` order
    mu_star::Vector{T}
    p::Vector{T}
    u::Vector{T}           # primal point of the least-distance problem, Mₐᵀμ
    g::Vector{T}           # scratch: products against the active rows
    Mv::Vector{T}          # scratch: M * v
    F::GramLDL{T}
end

function LDPWorkspace(Mt::Matrix{T}, iseq::AbstractVector{Bool}) where {T <: Real}
    n, m = size(Mt)
    kmax = min(m, n) + 1
    return LDPWorkspace{T}(
        Mt, zeros(T, m), zeros(T, m), collect(Bool, iseq), zeros(Int8, m),
        Int[], zeros(Int, m), T[], zeros(T, kmax), zeros(T, kmax),
        zeros(T, n), zeros(T, kmax), zeros(T, m), GramLDL{T}(kmax)
    )
end

"Row `r` of the constraint matrix, contiguous because the matrix is stored transposed."
@inline row(ws::LDPWorkspace, r::Integer) = view(ws.Mt, :, r)

"The bound row `j` is held at, given the side it entered on."
@inline target(ws::LDPWorkspace, j::Integer) = ws.side[j] == SIDE_LOWER ? ws.lo[j] : ws.hi[j]

"""
The sign row `j`'s multiplier must carry to be dual feasible.

`u = Mₐᵀμ`, and a row held at its *upper* target pushes `u` in the negative direction of
that row, so its multiplier is non-positive; one held at its lower target is non-negative.
Multiplying by this makes both cases read `musign * μ ≥ 0`.
"""
@inline musign(ws::LDPWorkspace{T}, j::Integer) where {T} =
    ws.side[j] == SIDE_UPPER ? -one(T) : one(T)

"Put row `r` into the working set at `side`, extending the factorization."
function activate!(ws::LDPWorkspace{T}, r::Integer, side::Int8) where {T}
    k = ws.F.k
    m_r = row(ws, r)
    g = view(ws.g, 1:k)
    for (j, a) in enumerate(ws.active)
        g[j] = dot(row(ws, a), m_r)
    end
    add_row!(ws.F, g, dot(m_r, m_r))
    push!(ws.active, r)
    push!(ws.mu, zero(T))
    ws.side[r] = side
    ws.slot[r] = length(ws.active)
    return ws
end

"Take the `i`-th working-set row back out."
function deactivate!(ws::LDPWorkspace, i::Integer)
    r = ws.active[i]
    remove_row!(ws.F, i)
    deleteat!(ws.active, i)
    deleteat!(ws.mu, i)
    ws.slot[r] = 0
    ws.side[r] = SIDE_INACTIVE
    for j in i:length(ws.active)
        ws.slot[ws.active[j]] = j
    end
    return ws
end

"""
    step_and_drop!(ws, p, zero_tol) -> Bool

Move the multipliers along `p` until the first one reaches zero from its feasible side, then
drop that row. `false` when nothing blocks, which on the singular branch means the problem
is infeasible.

An equality row never blocks: its multiplier is free in sign.
"""
function step_and_drop!(ws::LDPWorkspace{T}, p::AbstractVector{T}, zero_tol::T) where {T}
    alpha = typemax(T)
    blocking = 0
    for i in eachindex(ws.active)
        r = ws.active[i]
        ws.iseq[r] && continue
        # Feasibility is `musign * mu >= 0`, so the step blocks when `musign * p < 0`.
        musign(ws, r) * p[i] < -zero_tol || continue
        cand = -ws.mu[i] / p[i]
        if cand < alpha
            alpha = cand
            blocking = i
        end
    end
    iszero(blocking) && return false
    for i in eachindex(ws.mu)
        ws.mu[i] += alpha * p[i]
    end
    deactivate!(ws, blocking)
    return true
end

"""
    solve_ldp!(ws; max_iter, zero_tol, primal_tol) -> (status, iterations)

Algorithm 1 of Arnström, Bemporad & Axehill, *A dual active-set solver for embedded
quadratic programming using recursive LDLᵀ updates*, IEEE TAC 2022, on the two-sided
problem `lo ≤ Mu ≤ hi`.

Each iteration solves `Mₐ Mₐᵀ μ = tₐ` for the bounds the working set is held at. Multipliers
that are feasible in sign make the point dual feasible, and the run stops once no inactive
row is outside its bounds; otherwise the worst violator enters at the side it violates.
Multipliers that are not lead to a step toward them, dropping the first row whose multiplier
reaches zero. A singular Gram matrix is handled by walking along its null direction.
"""
function solve_ldp!(
        ws::LDPWorkspace{T}; max_iter::Int = 1000,
        zero_tol::T = sqrt(eps(T)), primal_tol::T = sqrt(eps(T))
    ) where {T}
    m = size(ws.Mt, 2)
    bland_after = 4 * (m + 1)
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
            # `singular_direction!` puts `+1` at the dependent row. Signed multipliers make
            # the two sides asymmetric, so that `+1` is a blocking candidate for a row held
            # at its upper bound — and blocking on it would drop the row that just entered,
            # which pricing would then choose again, forever. Orienting the direction so the
            # dependent row moves the feasible way restores the invariant the one-sided form
            # gets for free, and leaves the dependency to be resolved by one of the rows it
            # depends on.
            if musign(ws, ws.active[singular]) < 0
                for i in eachindex(p)
                    p[i] = -p[i]
                end
            end
            step_and_drop!(ws, p, zero_tol) || return (LDP_INFEASIBLE, iter)
            continue
        end

        mus = view(ws.mu_star, 1:k)
        for i in 1:k
            mus[i] = target(ws, ws.active[i])
        end
        solve_gram!(ws.F, mus)

        dual_feasible = true
        for i in 1:k
            r = ws.active[i]
            if !ws.iseq[r] && musign(ws, r) * mus[i] < 0
                dual_feasible = false
                break
            end
        end

        if !dual_feasible
            p = view(ws.p, 1:k)
            for i in 1:k
                p[i] = mus[i] - ws.mu[i]
            end
            step_and_drop!(ws, p, zero_tol) || return (LDP_CYCLED, iter)
            continue
        end

        copyto!(view(ws.mu, 1:k), mus)
        fill!(ws.u, zero(T))
        for i in 1:k
            axpy!(ws.mu[i], row(ws, ws.active[i]), ws.u)
        end

        # Price the inactive rows. Dantzig's rule takes the worst violation, which is the
        # fast choice but can cycle: a row that keeps swapping sides re-enters forever.
        # Past `bland_after` iterations the run switches to the lowest violated index,
        # Bland's rule, which terminates finitely at the cost of taking more steps.
        bland = iter > bland_after
        worst = primal_tol
        entering = 0
        entering_side = SIDE_UPPER
        for r in 1:m
            iszero(ws.slot[r]) || continue
            rr = dot(row(ws, r), ws.u)
            over = rr - ws.hi[r]
            under = ws.lo[r] - rr
            if over > worst
                worst = bland ? primal_tol : over
                entering = r
                entering_side = SIDE_UPPER
                bland && break
            end
            if under > worst
                worst = bland ? primal_tol : under
                entering = r
                entering_side = SIDE_LOWER
                bland && break
            end
        end

        iszero(entering) && return (LDP_OPTIMAL, iter)
        activate!(ws, entering, entering_side)
    end
    return (LDP_ITERATION_LIMIT, max_iter)
end

"""
The once-computed part of the reduction from a QP to a least-distance problem: the Cholesky
factor `R` of `P + εI`, the transformed and row-normalized constraint matrix `M = A R⁻¹`
with the caller's bounds in the same scaling, and the working state.

Only the targets change between proximal-point iterations, so `R`, `M` and the `LDLᵀ` of the
working set are all reused.
"""
struct DAQPReduction{T <: Real, F <: Cholesky}
    R::F
    bu::Vector{T}     # the caller's bounds, divided by the row norms
    bl::Vector{T}
    eps_prox::T
    ws::LDPWorkspace{T}
    scale::Vector{T}  # the row norm each row was divided by
end

"""
    reduce_qp(H, f, A, bupper, blower, iseq; eps_prox) -> DAQPReduction

Factor `H + εI` and transform the constraints into `lo ≤ Mu ≤ hi` with `M = A R⁻¹`.

With `eps_prox = 0` this needs `H ≻ 0`. Any `ε > 0` makes `H + εI` positive definite for
`H ⪰ 0`, which is what lets the proximal-point loop take a singular `H`, an LP being the
extreme case.
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
    # `Mᵀ = (A R⁻¹)ᵀ = R⁻ᵀ Aᵀ`, built transposed from the start rather than transposing a
    # built `M`: one triangular solve either way, and the result is the layout the loop
    # wants.
    Mt = Matrix{T}(transpose(R.U) \ Matrix{T}(transpose(A)))

    # Row normalization, as the reference does: it makes `primal_tol` mean the same thing on
    # every row however that row happened to be scaled.
    m = size(Mt, 2)
    scale = ones(T, m)
    bu = Vector{T}(undef, m)
    bl = Vector{T}(undef, m)
    for j in 1:m
        col = view(Mt, :, j)
        nrm = norm(col)
        if nrm > 0
            col ./= nrm
            scale[j] = nrm
        end
        bu[j] = bupper[j] / scale[j]
        bl[j] = blower[j] / scale[j]
    end

    ws = LDPWorkspace(Mt, iseq)
    return DAQPReduction{T, typeof(R)}(R, bu, bl, eps_prox, ws, scale)
end

"""
    set_targets!(red, v) -> red

Rebuild the least-distance bounds for the current `v`.

With `x = R⁻¹(−u − v)`, the caller's `bl ≤ Ax ≤ bu` becomes
`−(bu + Mv) ≤ Mu ≤ −(bl + Mv)`: the two bounds swap, because `Ax` runs against `Mu`.
"""
function set_targets!(red::DAQPReduction{T}, v::AbstractVector{T}) where {T}
    ws = red.ws
    mul!(ws.Mv, transpose(ws.Mt), v)
    for j in eachindex(ws.hi)
        ws.hi[j] = -(red.bl[j] + ws.Mv[j])
        ws.lo[j] = -(red.bu[j] + ws.Mv[j])
    end
    return red
end

"""
    rebuild_bounds!(red, bupper, blower) -> red

Adopt new caller bounds, keeping the factorization and the transformed constraint matrix.
"""
function rebuild_bounds!(red::DAQPReduction{T}, bupper::AbstractVector{T}, blower::AbstractVector{T}) where {T}
    for j in eachindex(red.bu)
        red.bu[j] = bupper[j] / red.scale[j]
        red.bl[j] = blower[j] / red.scale[j]
    end
    return red
end

"""
    reset_working_set!(red) -> red

Empty the working set apart from the equality rows, which must always be in it. The next
solve then starts where a fresh setup would.
"""
function reset_working_set!(red::DAQPReduction)
    ws = red.ws
    while !isempty(ws.active)
        deactivate!(ws, length(ws.active))
    end
    for j in eachindex(ws.iseq)
        ws.iseq[j] && activate!(ws, j, SIDE_UPPER)
    end
    return red
end

"""
    multipliers!(y, red, red_ws) -> y

Map the least-distance multipliers back to one per problem row, undoing the row scaling. The
sign already distinguishes the two bounds, so nothing else is needed.
"""
function multipliers!(y::AbstractVector{T}, red::DAQPReduction{T}) where {T}
    fill!(y, zero(T))
    ws = red.ws
    for (slot, j) in enumerate(ws.active)
        # The signed multiplier already distinguishes the two bounds: non-negative for a row
        # held at the caller's upper bound, non-positive at the lower one.
        y[j] = ws.mu[slot] / red.scale[j]
    end
    return y
end

"""
    inner_solve!(red, f, x) -> (status, iters, v)

One pass of Algorithm 1 at the current proximal centre `x`: rebuild the targets from
`v = R⁻ᵀ(f − εx)`, then run the dual active-set loop, warm-started from whatever working set
is in place.
"""
function inner_solve!(
        red::DAQPReduction{T}, f::AbstractVector{T}, x::AbstractVector{T};
        max_iter::Int, zero_tol::T, primal_tol::T
    ) where {T}
    rhs = iszero(red.eps_prox) ? f : f .- red.eps_prox .* x
    v = red.R.L \ rhs
    set_targets!(red, v)
    status, iters = solve_ldp!(red.ws; max_iter, zero_tol, primal_tol)
    return status, iters, v
end

"""
    run_daqp!(red, f; ...) -> (x, status, iters)

Algorithm 1, or Algorithm 2 wrapped around it when `eps_prox > 0`, on a reduction that
already exists. The working set is whatever the reduction holds, so a repeated call warm
starts from the previous answer.
"""
function run_daqp!(
        red::DAQPReduction{T}, f::AbstractVector{T};
        max_iter::Int, zero_tol::T, primal_tol::T,
        eps_prox::T, eta_prox::T, max_prox::Int
    ) where {T}
    ws = red.ws
    x = zeros(T, size(ws.Mt, 1))
    if iszero(eps_prox)
        status, iters, v = inner_solve!(red, f, x; max_iter, zero_tol, primal_tol)
        status == LDP_OPTIMAL || return (T[], status, iters)
        return (red.R.U \ (-ws.u - v), status, iters)
    end
    xold = similar(x)
    total = 0
    for _ in 1:max_prox
        copyto!(xold, x)
        status, iters, v = inner_solve!(red, f, x; max_iter, zero_tol, primal_tol)
        total += iters
        status == LDP_OPTIMAL || return (T[], status, total)
        x = red.R.U \ (-ws.u - v)
        norm(x - xold, Inf) < eta_prox && return (x, status, total)
    end
    return (x, LDP_ITERATION_LIMIT, total)
end
