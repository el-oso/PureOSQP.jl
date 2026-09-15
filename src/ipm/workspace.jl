"Row classes of the interior-point method: a free row has no bound, an equality row `l == u`."
const ROW_FREE = Int8(-1)
const ROW_INEQUALITY = Int8(0)
const ROW_EQUALITY = Int8(1)

"""
    classify_rows!(rclass, has_l, has_u, prob) -> n_sides

Set every row's class and side masks from `prob`'s current bounds, and return the number of
inequality sides across all rows. Used at workspace construction and by [`update!`](@ref)
whenever `l` or `u` changes.
"""
function classify_rows!(rclass::AbstractVector{Int8}, has_l::AbstractVector{Bool}, has_u::AbstractVector{Bool}, prob::Problem{T}) where {T}
    loose = INFTY(T) * MIN_SCALING(T)
    sides = 0
    for i in eachindex(rclass)
        if prob.l0[i] == prob.u0[i]
            rclass[i] = ROW_EQUALITY
            has_l[i] = false
            has_u[i] = false
        elseif prob.l[i] < -loose && prob.u[i] > loose
            rclass[i] = ROW_FREE
            has_l[i] = false
            has_u[i] = false
        else
            rclass[i] = ROW_INEQUALITY
            has_l[i] = prob.l[i] > -loose
            has_u[i] = prob.u[i] < loose
            sides += has_l[i] + has_u[i]
        end
    end
    return sides
end

"""
    IPMWorkspace{T,MP,MA,V,VI,VB,LS}

Solver state of the interior-point method, built by [`setup`](@ref) with `algorithm = :ipm`.
The problem is the equilibrated one, and every iterate is in scaled space.

An inequality row `i` carries a slack and a multiplier for each finite side:
`s_l = Ãx − l̃` and `z_l` when `has_l[i]`, `s_u = ũ − Ãx` and `z_u` when `has_u[i]`, and its
multiplier is `y = z_u − z_l`. An absent side holds `s = 1`, `z = 0`, so the elementwise
loops need no branch beyond the mask. An equality row carries a free multiplier `y`; a free
row has `y = 0`.

`seeded` says whether `x` and `y` are a starting point: a solve that ends with a point
([`has_solution`](@ref)) and [`warm_start!`](@ref) set it; a solve that ends without one,
[`cold_start!`](@ref) and `warm_starting = false` clear it. After a solve without a point, `x`
and `y` still hold the last iterate.
"""
mutable struct IPMWorkspace{
        T <: Real, MP <: AbstractMatrix, MA <: AbstractMatrix, V <: AbstractVector{T},
        VI <: AbstractVector{Int8}, VB <: AbstractVector{Bool}, LS <: LinearSystem,
    }
    const prob::Problem{T, MP, MA, V}
    const linsys::LS
    # `sigma` is the current `reg_primal`, so a regularization bump replaces the object; `w`
    # and `w_inv` are rewritten in place every outer iteration.
    weights::SystemWeights{T, V}
    const rclass::VI
    const has_l::VB
    const has_u::VB
    # Inequality sides across every row; recomputed by `update!` whenever `l` or `u` changes
    # a row's class.
    n_sides::Int
    const x::V
    const y::V
    const s_l::V
    const s_u::V
    const z_l::V
    const z_u::V
    const Ax::V
    const Px::V
    const Aty::V
    # `clamp(Ãx, l̃, ũ)`, the feasible point the primal residual is measured against.
    const z::V
    # Newton residuals: `r_d = P̃x + q̃ + Ãᵀy`; `r_l = Ãx − l̃ − s_l` on a lower side and
    # `Ãx − l̃` on an equality row; `r_u = ũ − Ãx − s_u` on an upper side; zero elsewhere.
    const r_d::V
    const r_l::V
    const r_u::V
    # The complementarity terms of the right-hand side, zero on absent sides.
    const rc_l::V
    const rc_u::V
    const rhs_x::V
    const rhs_z::V
    const dx::V
    const dy::V
    const ds_l::V
    const ds_u::V
    const dz_l::V
    const dz_u::V
    const Adx::V
    # Refinement residual and correction.
    const res_x::V
    const res_z::V
    const corr_x::V
    const corr_y::V
    # Infeasibility certificate candidates. The tests project them in place, and the one that
    # passes is what the solution reports.
    const cert_x::V
    const cert_y::V
    # The regularization in force: the settings' values times ten per bump in this solve.
    reg_primal::T
    reg_dual::T
    reg_bumps::Int
    # The backend's factorization was built with a `sigma` other than the current one, so the
    # next factorization must be a full one.
    sigma_changed::Bool
    # Guards: consecutive steps shorter than `STALL_STEP`, consecutive iterations whose merit
    # (`mu`, or `rnorm` without an inequality side) did not fall, the previous merit, and
    # whether the certificate tests now run every iteration.
    short_steps::Int
    flat_merit::Int
    last_merit::T
    alert::Bool
    mu::T
    rnorm::T
    alpha::T
    prim_res::T
    dual_res::T
    scaled_prim_res::T
    scaled_dual_res::T
    obj_val::T
    dual_obj_val::T
    duality_gap::T
    scaled_duality_gap::T
    xtPx::T
    qtx::T
    SCy::T
    rel_kkt_error::T
    cg_iters::Int
    # Consecutive Newton solves the backend reported as missed (see `last_solve_converged`).
    cg_misses::Int
    iter::Int
    status::Status
    seeded::Bool
    first_run::Bool
    polished::Bool
    status_polish::PolishStatus
    setup_time::Float64
    # Accumulated across the `update!` calls made since the previous solve, and charged to
    # the next one; reset once reported, as in `Workspace`.
    update_time::Float64
    solve_time::Float64
    polish_time::Float64
    settings::IPMSettings{T}
end

function Base.show(io::IO, ws::IPMWorkspace)
    print(
        io, "PureOSQP IPMWorkspace: ", ws.prob.n, "×", ws.prob.m,
        ", backend ", backend_name(ws.linsys),
        ", status ", status_name(ws.status),
    )
    return nothing
end

"""
    ipm_workspace(ls, prob, wt, settings) -> IPMWorkspace

Classify the rows of `prob` and allocate the interior-point state around the backend `ls`,
which solves through the weights object `wt`.
"""
function ipm_workspace(ls::LinearSystem, prob::Problem{T}, wt::SystemWeights{T}, settings::IPMSettings{T}) where {T}
    n, m, q0 = prob.n, prob.m, prob.q0
    buf(k) = fill!(similar(q0, T, k), zero(T))
    rclass = fill!(similar(q0, Int8, m), ROW_INEQUALITY)
    has_l = fill!(similar(q0, Bool, m), false)
    has_u = fill!(similar(q0, Bool, m), false)
    sides = classify_rows!(rclass, has_l, has_u, prob)
    ws = IPMWorkspace{T, typeof(prob.P), typeof(prob.A), typeof(q0), typeof(rclass), typeof(has_l), typeof(ls)}(
        prob, ls, wt, rclass, has_l, has_u, sides,
        buf(n), buf(m), buf(m), buf(m), buf(m), buf(m),
        buf(m), buf(n), buf(n), buf(m),
        buf(n), buf(m), buf(m), buf(m), buf(m),
        buf(n), buf(m), buf(n), buf(m), buf(m), buf(m), buf(m), buf(m), buf(m),
        buf(n), buf(m), buf(n), buf(m),
        buf(n), buf(m),
        settings.reg_primal, settings.reg_dual, 0, false,
        0, 0, zero(T), false,
        zero(T), zero(T), zero(T),
        zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T),
        zero(T), zero(T), zero(T), zero(T),
        0, 0, 0, UNSOLVED, false, true, false, POLISH_NOT_PERFORMED, 0.0, 0.0, 0.0, 0.0,
        settings,
    )
    return ws
end

function refuse_ipm_operators()
    throw(
        ArgumentError(
            "algorithm = :ipm factors a matrix built from the entries of P and A, and one of " *
                "them declares `PureOSQP.is_materializable` false: it supplies products only. " *
                "Pass matrices, pass linsys = :indirect with a caller-supplied preconditioner " *
                "and scaling = 0, or use algorithm = :admm."
        )
    )
end

"Whether `M` is a preconditioner the caller built, rather than `nothing` or a built-in one."
caller_preconditioner(M) = !(M isa Union{Nothing, IdentityPreconditioner, JacobiPreconditioner})

probing(M) = M isa ProductOperator && M.probe

function setup_backend(
        ::Val{:ipm}, ::Val{LS}, ::Type{T}, P::AbstractMatrix, q::AbstractVector,
        A::AbstractMatrix, l::AbstractVector, u::AbstractVector; preconditioner = nothing, kwargs...
    ) where {LS, T <: Real}
    t0 = time_ns()
    nv, mv = validate(P, q, A, l, u)
    settings = IPMSettings{T}(; linsys = LS, kwargs...)
    isnothing(preconditioner) || LS === :indirect || throw(
        ArgumentError(
            "preconditioner is used only by the matrix-free backend: pass linsys = :indirect with it."
        )
    )
    LS === :indirect && !caller_preconditioner(preconditioner) && throw(
        ArgumentError(
            "the interior-point method uses conjugate gradients only with a caller-supplied " *
                "preconditioner; measured without one, or with the Jacobi diagonal, it does not " *
                "reach the tolerance on most problems. Pass one, choose a direct linsys, or use " *
                "algorithm = :admm."
        )
    )
    LS === :indirect || (is_materializable(P) && is_materializable(A)) || refuse_ipm_operators()
    if LS === :indirect
        iszero(settings.scaling) || throw(
            ArgumentError(
                "a caller-supplied preconditioner approximates P + reg_primal*I + A' * Diagonal(w) * A " *
                    "for the P and A passed to setup, which equilibration would change: pass scaling = 0 with it."
            )
        )
        (probing(P) || probing(A)) && throw(
            ArgumentError(
                "a caller-supplied preconditioner approximates the operators passed to setup, " *
                    "and probe = true exists only to equilibrate them: build them without probe."
            )
        )
    end
    LS === :kronecker && throw(
        ArgumentError(
            "linsys = :kronecker is not available with algorithm = :ipm: the Kronecker " *
                "backend needs the same weight on every row, and the interior-point weights " *
                "differ from row to row. Choose another linsys, or use algorithm = :admm."
        )
    )
    LS === :lowrank && throw(
        ArgumentError(
            "linsys = :lowrank is not available with algorithm = :ipm: on linear programs " *
                "its Woodbury solve does not reach the interior-point tolerances. Leave " *
                "linsys = :auto, which serves the pair with linsys = :kkt, or use algorithm = :admm."
        )
    )
    is_convex(T, P, settings.reg_primal) || throw(
        ArgumentError(
            "P + reg_primal*I is not positive definite: P is indefinite, so the problem is not convex."
        )
    )
    prob = validated_problem(T, nv, mv, P, q, A, l, u, settings.scaling)
    n, m, q0 = prob.n, prob.m, prob.q0
    # Unit weights are the starting-point system, so a rung that decides by factoring leaves
    # the first factorization of a solve in place.
    wt = SystemWeights(fill!(similar(q0, T, m), one(T)), fill!(similar(q0, T, m), one(T)), settings.reg_primal)
    sel = IPMSelection()
    if LS === :kkt
        ws = ipm_workspace(FullKKT(q0, n, m), prob, wt, settings)
    elseif LS === :dense
        ws = ipm_workspace(ReducedCholesky(q0, n, m), prob, wt, settings)
    elseif LS === :indirect
        ws = ipm_workspace(indirect_backend(q0, n, m, preconditioner), prob, wt, settings)
        adopt_settings!(ws.linsys, settings)
        use_residual_stop!(ws.linsys, true)
    elseif LS === :sparse
        rung = kkt_rung(P, A, prob, wt, sel)
        isnothing(rung) && (rung = reduced_rung(P, A, prob, wt, sel))
        isnothing(rung) && throw(
            ArgumentError(
                "linsys = :sparse factors the reduced or KKT matrix sparsely and could not " *
                    "serve this pair: it needs a SparseMatrixCSC A whose factor stays sparse " *
                    "enough, and SparseArrays.jl loaded. Retry with linsys = :auto."
            )
        )
        ws = ipm_workspace(first(rung), prob, wt, settings)
    elseif LS === :diagonal
        (P isa Diagonal && A isa Diagonal) || throw(
            ArgumentError("linsys = :diagonal needs P and A both diagonal")
        )
        ws = ipm_workspace(first(choose_backend(P, A, prob, wt, sel)), prob, wt, settings)
    elseif LS === :tridiagonal
        tridiag_pair =
            (P isa Union{SymTridiagonal, Tridiagonal} && A isa Diagonal) ||
            (P isa Union{Diagonal, SymTridiagonal, Tridiagonal} && A isa Bidiagonal)
        tridiag_pair || throw(
            ArgumentError(
                "linsys = :tridiagonal needs a diagonal, symmetric-tridiagonal or tridiagonal " *
                    "P with a diagonal A, or any of those P with a bidiagonal A"
            )
        )
        ws = ipm_workspace(first(choose_backend(P, A, prob, wt, sel)), prob, wt, settings)
    elseif LS === :block
        rung = block_rung(P, A, prob, wt, sel)
        isnothing(rung) && throw(
            ArgumentError(
                "linsys = :block needs P and A both block diagonal over the same column " *
                    "partition, with more than one block, and declines this pair"
            )
        )
        ws = ipm_workspace(first(rung), prob, wt, settings)
    else
        ws = ipm_workspace(first(choose_backend(P, A, prob, wt, sel)), prob, wt, settings)
    end
    ws.setup_time = (time_ns() - t0) / 1.0e9
    return ws
end

"""
    select_backend(P, A, prob, wt, sel::IPMSelection) -> (LinearSystem, Bool)

The interior-point ladder: the sparse KKT factorization first, then the sparse reduced one
and the structured reduced backends, and [`FullKKT`](@ref) as the terminal for any
materializable pair. The Kronecker rung is absent, since it needs uniform weights; the
low-rank rung declines (see its [`IPMSelection`](@ref) method); and
[`formed_rung`](@ref) is absent, since its inverse would be rebuilt every outer iteration for
a handful of solves.
"""
function select_backend(P, A, prob, wt, sel::IPMSelection)
    rung = density_gate_rung(P, A, prob, sel)
    isnothing(rung) || return rung
    rung = kkt_rung(P, A, prob, wt, sel)
    isnothing(rung) || return rung
    rung = reduced_rung(P, A, prob, wt, sel)
    isnothing(rung) || return rung
    rung = block_rung(P, A, prob, wt, sel)
    isnothing(rung) || return rung
    rung = lowrank_rung(P, A, prob, wt, sel)
    isnothing(rung) || return rung
    rung = dense_rung(P, A, prob, sel)
    isnothing(rung) || return rung
    return indirect_rung(P, A, prob, sel)
end

"""
    dense_rung(P, A, prob, sel::IPMSelection) -> (LinearSystem, Bool) or nothing

The interior-point terminal: [`FullKKT`](@ref) when LAPACK serves the element type. The
reduced matrix it avoids is inverted once per outer iteration for a handful of solves, and
its weights reach `1/δ_d`. Other element types get [`ReducedCholesky`](@ref), since
`bunchkaufman!` has no generic method.
"""
function dense_rung(P::AbstractMatrix, A::AbstractMatrix, prob::Problem{T}, sel::IPMSelection) where {T}
    (is_materializable(P) && is_materializable(A)) || return nothing
    T <: LinearAlgebra.BlasFloat && return (FullKKT(prob.q0, prob.n, prob.m), false)
    return (ReducedCholesky(prob.q0, prob.n, prob.m), false)
end

dense_rung(P, A, prob, sel::IPMSelection) = nothing

"""
    lowrank_rung(P::Diagonal, A::RowCoupled, prob, wt, sel::IPMSelection) -> nothing

Declines, so the pair reaches [`FullKKT`](@ref). A variable that only the coupling rows reach
has `δ_p` alone in the diagonal core wherever `P` is zero, which puts `1/δ_p` in the core's
inverse; on linear programs over these pairs the Woodbury solve through it ends without a
solution (`bench/ipm_backends.jl`).
"""
lowrank_rung(P::Diagonal, A::RowCoupled, prob, wt, sel::IPMSelection) = nothing

"""
    indirect_rung(P, A, prob, sel::IPMSelection)

Refuses: the interior-point method runs the matrix-free backend only when it is named, with
a caller-supplied preconditioner.
"""
indirect_rung(P, A, prob, sel::IPMSelection) = refuse_ipm_operators()

"""
    warm_start!(ws::IPMWorkspace; x = nothing, y = nothing)

Seed the next solve's starting point in problem space. Slacks and multipliers are rebuilt
from `x` and `y` when the solve starts.
"""
function warm_start!(ws::IPMWorkspace{T}; x = nothing, y = nothing) where {T}
    prob = ws.prob
    if !isnothing(x)
        length(x) == prob.n || throw(ArgumentError("length(x) must be $(prob.n)"))
        all(isfinite, x) || throw(ArgumentError("x must be finite, found NaN or Inf"))
        ws.x .= T.(x) ./ prob.D
    end
    if !isnothing(y)
        length(y) == prob.m || throw(ArgumentError("length(y) must be $(prob.m)"))
        all(isfinite, y) || throw(ArgumentError("y must be finite, found NaN or Inf"))
        ws.y .= prob.c .* T.(y) ./ prob.E
    end
    ws.seeded = true
    return ws
end

"""
    cold_start!(ws::IPMWorkspace) -> ws

Zero `x` and `y` and clear `seeded`, so the next solve computes its own starting point.
"""
function cold_start!(ws::IPMWorkspace{T}) where {T}
    fill!(ws.x, zero(T))
    fill!(ws.y, zero(T))
    ws.seeded = false
    return ws
end

"""
    update!(ws::IPMWorkspace; q, l, u, P, A) -> ws

Replace problem data in an existing interior-point workspace, as [`update!`](@ref) does for
[`Workspace`](@ref): the same validation and adoption, checked against convexity at
`reg_primal`. A row whose bounds move into or out of equality or freeness is reclassified;
`s_l`, `s_u`, `z_l` and `z_u` are left as they are, since `starting_point!` rebuilds
them from `x`, `y` and the current classes at the next solve.

No factorization happens here: every outer iteration factorizes the Newton system at its own
weights regardless, and a solve resets the regularization from `settings` before its first
one, so a later solve picks up the new data whether or not `P` or `A` changed.
"""
function update!(
        ws::IPMWorkspace{T}; q = nothing, l = nothing, u = nothing, P = nothing, A = nothing
    ) where {T}
    t0 = time_ns()
    prob = ws.prob
    validate_update!(prob, ws.linsys; P, A, q, l, u)
    !isnothing(P) && !is_convex(T, P, ws.settings.reg_primal) &&
        throw(
        ArgumentError(
            "P + reg_primal*I is not positive definite: P is indefinite, so the problem is not convex."
        )
    )
    adopt_update!(prob; P, A, q, l, u)
    if !isnothing(l) || !isnothing(u)
        ws.n_sides = classify_rows!(ws.rclass, ws.has_l, ws.has_u, prob)
    end
    ws.update_time += (time_ns() - t0) / 1.0e9
    return ws
end

"""
    update_settings!(ws::IPMWorkspace; kwargs...) -> ws

Replace the workspace's settings, keeping every field not named in `kwargs`, exactly as
[`update_settings!`](@ref) does for [`Workspace`](@ref): the keywords are those of
[`IPMSettings`](@ref) and are validated the same way, `linsys` and `scaling` are rejected
because the backend and the equilibration are fixed once the workspace is built, and a
`:indirect` backend's own copy of the conjugate-gradient settings is refreshed at once.

Every other field, `reg_primal` and `reg_dual` included, is free: a solve resets the
regularization from `settings` before its first iteration and refactorizes every iteration
after, so nothing here needs to trigger a refactorization the way a `ρ` or `σ` change does
for ADMM.
"""
function update_settings!(ws::IPMWorkspace{T}; kwargs...) where {T}
    old = ws.settings
    new = IPMSettings{T}(; settings_tuple(old)..., kwargs...)
    new.linsys === old.linsys || throw(
        ArgumentError(
            "linsys is fixed once the workspace is built, because the backend is part of " *
                "its type. Call setup again to change it."
        )
    )
    new.scaling == old.scaling || throw(
        ArgumentError(
            "scaling is fixed once the workspace is built, because the equilibration " *
                "factors come from the data setup saw. Call setup again to change it."
        )
    )
    ws.settings = new
    adopt_settings!(ws.linsys, new)
    return ws
end
