"""
    Status

How a solve ended, reported as `Solution.status`. Eleven values, in three groups.

**Converged.** [`has_solution`](@ref) is true and `x`, `y` are the answer.

| status | meaning |
|---|---|
| `SOLVED` | primal and dual residuals are under the requested tolerances, and the duality gap too when `check_dualgap` is set |
| `SOLVED_INACCURATE` | the run stopped without meeting those, but the residuals clear ten times them |

**Stopped early.** [`has_solution`](@ref) is true: the iterates reached are a valid point,
simply not a converged one, and are returned rather than discarded.

| status | meaning |
|---|---|
| `MAX_ITER_REACHED` | `max_iter` was spent and even the ten-times check failed |
| `TIME_LIMIT_REACHED` | `time_limit` was spent first; the budget is checked every iteration and the status returned without re-testing the tolerances, so a run can stop at a point that would have passed |
| `INTERRUPTED` | a `Ctrl-C` landed inside the loop |

**No point to return.** [`has_solution`](@ref) is false, and `x` and `y` are filled with
`NaN` rather than a plausible-looking number.

| status | meaning |
|---|---|
| `PRIMAL_INFEASIBLE` | a certificate was found; it is in `Solution.prim_inf_cert` |
| `DUAL_INFEASIBLE` | a certificate was found; it is in `Solution.dual_inf_cert` |
| `PRIMAL_INFEASIBLE_INACCURATE`, `DUAL_INFEASIBLE_INACCURATE` | the same certificates, established only at ten times the requested tolerances |
| `NON_CONVEX` | the residuals diverged, which a convex problem's cannot |
| `UNSOLVED` | the loop has not run. It is a workspace's state before its first [`solve!`](@ref) and never the status of a completed solve |

An unconverged result is never reported as `SOLVED`.

Failures that are not outcomes of the algorithm — a non-symmetric `P`, a bad setting, a
dimension mismatch — raise instead of returning a status, so there is no error code to
inspect and no way to miss one by not looking.
"""
@enum Status begin
    UNSOLVED
    SOLVED
    SOLVED_INACCURATE
    PRIMAL_INFEASIBLE
    PRIMAL_INFEASIBLE_INACCURATE
    DUAL_INFEASIBLE
    DUAL_INFEASIBLE_INACCURATE
    MAX_ITER_REACHED
    TIME_LIMIT_REACHED
    INTERRUPTED
    NON_CONVEX
end

"""
    PolishStatus

Outcome of the polishing step, reported as `Solution.status_polish`. Polishing guesses the
active set, so it can decline for reasons that are not failures.

| status | value | meaning |
|---|---|---|
| `POLISH_LINSYS_ERROR` | -2 | the active-set KKT matrix could not be factored |
| `POLISH_FAILED` | -1 | the polished point did not improve both residuals, so it was discarded |
| `POLISH_NOT_PERFORMED` | 0 | `polish` was off, which is the default |
| `POLISH_SUCCESS` | 1 | the polished point was accepted and is what `Solution` carries |
| `POLISH_NO_ACTIVE_SET_FOUND` | 2 | no constraint was active, so there was nothing to polish |

Only `POLISH_SUCCESS` changes the answer. `Solution.polished` is the narrower question of
whether that happened; the other four all leave the ADMM point in place, and only the first
two describe something going wrong.
"""
@enum PolishStatus begin
    POLISH_LINSYS_ERROR = -2
    POLISH_FAILED = -1
    POLISH_NOT_PERFORMED = 0
    POLISH_SUCCESS = 1
    POLISH_NO_ACTIVE_SET_FOUND = 2
end

"Statuses that carry a meaningful primal-dual point."
@inline has_solution(s::Status) =
    s === SOLVED || s === SOLVED_INACCURATE || s === MAX_ITER_REACHED ||
    s === TIME_LIMIT_REACHED || s === INTERRUPTED

@inline INFTY(::Type{T}) where {T} = min(T(1.0e30), prevfloat(typemax(T)))
@inline MIN_SCALING(::Type{T}) where {T} = T(1.0e-4)
@inline MAX_SCALING(::Type{T}) where {T} = T(1.0e4)
@inline RHO_MIN(::Type{T}) where {T} = T(1.0e-6)
@inline RHO_MAX(::Type{T}) where {T} = T(1.0e6)
@inline RHO_TOL(::Type{T}) where {T} = T(1.0e-4)
@inline RHO_EQ_OVER_INEQ(::Type{T}) where {T} = T(1.0e3)
@inline DIVISION_TOL(::Type{T}) where {T} = one(T) / INFTY(T)
"Multipliers below this are treated as zero in the duality gap, where a 1e-20 `y` against a
large bound would otherwise dominate the sum."
@inline ZERO_DEADZONE(::Type{T}) where {T} = T(1.0e-10)

"The backends `linsys` may name. [`setup`](@ref) rejects anything else before turning the
choice into a type parameter, so an unusable name costs an error and not a specialization."
const LINSYS_OPTIONS = (:auto, :dense, :kkt, :indirect)

"""
    Settings{T}

Algorithm parameters. Defaults follow libosqp 0.6.2 with two exceptions.
`adaptive_rho_interval` is a fixed iteration count rather than a wall-clock fraction of
the setup time, so iteration counts are reproducible across machines. And `check_dualgap`
defaults on, following libosqp 1.x, so a run stops on the duality gap as well as on the
two residuals.

`profile_primdual` is off by default and is the only setting that makes the solve read a
clock it would not otherwise read. Turning it on fills [`Solution`](@ref)'s `primdual_int`
and `primdual_int_log`, at a measured cost under 1%.
"""
struct Settings{T <: Real}
    rho::T
    sigma::T
    alpha::T
    max_iter::Int
    time_limit::T
    eps_abs::T
    eps_rel::T
    eps_prim_inf::T
    eps_dual_inf::T
    scaling::Int
    adaptive_rho::Symbol
    adaptive_rho_interval::Int
    adaptive_rho_fraction::T
    adaptive_rho_tolerance::T
    check_termination::Int
    cg_max_iter::Int
    cg_tol_fraction::T
    cg_tol_reduction::Int
    check_dualgap::Bool
    profile_primdual::Bool
    scaled_termination::Bool
    rho_is_vec::Bool
    polish::Bool
    polish_refine_iter::Int
    delta::T
    warm_starting::Bool
    verbose::Bool
    linsys::Symbol
end

function Settings{T}(;
        rho = 0.1, sigma = 1.0e-6, alpha = 1.6, max_iter = 4000, time_limit = Inf,
        eps_abs = 1.0e-3, eps_rel = 1.0e-3, eps_prim_inf = 1.0e-4, eps_dual_inf = 1.0e-4,
        scaling = 10, adaptive_rho = true, adaptive_rho_interval = 50,
        adaptive_rho_fraction = 0.4, adaptive_rho_tolerance = 5.0, check_termination = 25,
        cg_max_iter = 20, cg_tol_fraction = 0.15, cg_tol_reduction = 10,
        check_dualgap = true, profile_primdual = false,
        scaled_termination = false, rho_is_vec = true,
        polish = false, polish_refine_iter = 3, delta = 1.0e-6,
        warm_starting = true, verbose = false, linsys = :auto,
    ) where {T <: Real}
    # `adaptive_rho` names a mode. A `Bool` is also accepted: `true` is `:iterations`.
    rho_mode = adaptive_rho isa Bool ? (adaptive_rho ? :iterations : :disabled) :
        Symbol(adaptive_rho)
    rho_mode in (:disabled, :iterations, :kkt_error) || throw(
        ArgumentError(
            "adaptive_rho must be :disabled, :iterations, :kkt_error or a Bool, got :$rho_mode"
        )
    )
    0 < adaptive_rho_fraction <= 1 || throw(
        ArgumentError("adaptive_rho_fraction must lie in (0, 1], got $adaptive_rho_fraction")
    )
    linsys in LINSYS_OPTIONS || throw(
        ArgumentError("linsys must be :auto, :dense, :kkt or :indirect, got :$linsys")
    )
    cg_max_iter > 0 || throw(ArgumentError("cg_max_iter must be positive, got $cg_max_iter"))
    0 < cg_tol_fraction <= 1 || throw(
        ArgumentError("cg_tol_fraction must lie in (0, 1], got $cg_tol_fraction")
    )
    cg_tol_reduction > 0 || throw(ArgumentError("cg_tol_reduction must be positive"))
    sigma > 0 || throw(ArgumentError("sigma must be positive, got $sigma"))
    rho > 0 || throw(ArgumentError("rho must be positive, got $rho"))
    0 < alpha < 2 || throw(ArgumentError("alpha must lie in (0, 2), got $alpha"))
    max_iter > 0 || throw(ArgumentError("max_iter must be positive, got $max_iter"))
    time_limit > 0 || throw(ArgumentError("time_limit must be positive (Inf disables it), got $time_limit"))
    eps_abs >= 0 && eps_rel >= 0 || throw(ArgumentError("eps_abs and eps_rel must be non-negative"))
    eps_abs > 0 || eps_rel > 0 || throw(ArgumentError("at least one of eps_abs, eps_rel must be positive"))
    eps_prim_inf > 0 && eps_dual_inf > 0 || throw(ArgumentError("eps_prim_inf and eps_dual_inf must be positive"))
    scaling >= 0 || throw(ArgumentError("scaling must be non-negative, got $scaling"))
    check_termination >= 0 || throw(ArgumentError("check_termination must be non-negative, got $check_termination"))
    adaptive_rho_interval >= 0 || throw(ArgumentError("adaptive_rho_interval must be non-negative"))
    adaptive_rho_tolerance >= 1 || throw(ArgumentError("adaptive_rho_tolerance must be at least 1"))
    polish_refine_iter >= 0 || throw(ArgumentError("polish_refine_iter must be non-negative"))
    delta > 0 || throw(ArgumentError("delta must be positive, got $delta"))
    return Settings{T}(
        T(rho), T(sigma), T(alpha), Int(max_iter), T(time_limit),
        T(eps_abs), T(eps_rel), T(eps_prim_inf), T(eps_dual_inf),
        Int(scaling), rho_mode, Int(adaptive_rho_interval), T(adaptive_rho_fraction),
        T(adaptive_rho_tolerance), Int(check_termination),
        Int(cg_max_iter), T(cg_tol_fraction), Int(cg_tol_reduction),
        Bool(check_dualgap), Bool(profile_primdual),
        Bool(scaled_termination), Bool(rho_is_vec),
        Bool(polish), Int(polish_refine_iter), T(delta),
        Bool(warm_starting), Bool(verbose), Symbol(linsys),
    )
end

"""
    Solution{T}

Result of a solve. `x` and `y` are in the caller's problem space. When the status is an
infeasibility, the corresponding certificate is populated and `x`/`y` are filled with
`NaN`; otherwise the certificates are empty.

`duality_gap` is `xᵀPx + qᵀx + SC(y)`, where `SC` is the support function of `[l, u]`; it
is zero at an exact solution and is reported unscaled. `rel_kkt_error` is the largest of
the two residuals and the gap, so one number bounds how far the point is from optimal.
`rho_updates` counts adaptive-`ρ` changes only, unlike `Workspace.refactor_count`, which
also counts refactorizations forced by new data.

`primdual_int` and `primdual_int_log` are the primal-dual integral, `∫|gap| dt` over the
solve, and are zero unless `profile_primdual` was set. They differ only in how the gap is
interpolated between the iterations that sampled it: the first joins samples with a straight
line, the second with an exponential, which is what a geometrically decaying gap does. The
logarithmic mean never exceeds the arithmetic one, so the second is the lower of the two and
the pair brackets the integral; measured, they differ by 1.3× to 4.2×
(`bench/primdual_integral.jl`).

**Both integrate against wall-clock time and neither is reproducible.** They cannot be
compared across machines, or between runs on a machine whose clock is not pinned, and no
test asserts a value for either — only that they are positive, ordered, and unaffected by
the measuring. They exist to compare convergence *profiles*, not to be an output of the
solve.

Times are in seconds. `setup_time` belongs to the workspace and is reported by every solve
that uses it, but `run_time` counts it only for the first solve — a re-solve did not pay
it again, so adding it in would overstate the cost of the loop that `update!` exists to
make cheap.

`update_time` is the time spent in [`update!`](@ref) since the previous solve, accumulated
across however many calls were made, and it *is* counted in `run_time`: in the loop
`update!` exists for, a cycle is an update followed by a solve, and that pair is what the
caller pays. It resets once reported, so each solve accounts for its own updates and no
others.
"""
struct Solution{T <: Real}
    x::Vector{T}
    y::Vector{T}
    status::Status
    obj_val::T
    dual_obj_val::T
    duality_gap::T
    prim_res::T
    dual_res::T
    rel_kkt_error::T
    iter::Int
    # Zero unless `profile_primdual` was set. Wall-clock quantities: not reproducible across
    # machines, and not comparable between runs on a machine whose clock is not pinned.
    primdual_int::Float64
    primdual_int_log::Float64
    rho_estimate::T
    rho_updates::Int
    polished::Bool
    status_polish::PolishStatus
    setup_time::Float64
    update_time::Float64
    solve_time::Float64
    polish_time::Float64
    run_time::Float64
    prim_inf_cert::Vector{T}
    dual_inf_cert::Vector{T}
end

"""
    Workspace{T,MP,MA,V,VI,LS}

Solver state. The caller's `P` and `A` are held by reference and never mutated: Ruiz
equilibration lives in the factors `D`, `E`, `c` and is applied lazily on every product.

The buffers are `similar` to the `q` that built the workspace, so they follow the array
type of the caller's data rather than always being `Vector`.
"""
mutable struct Workspace{
        T <: Real, MP <: AbstractMatrix, MA <: AbstractMatrix,
        V <: AbstractVector{T}, VI <: AbstractVector{Int8}, LS <: LinearSystem,
    }
    P::MP
    A::MA
    n::Int
    m::Int
    q0::V
    l0::V
    u0::V
    q::V
    l::V
    u::V
    D::V
    E::V
    c::T
    x::V
    y::V
    z::V
    x_prev::V
    z_prev::V
    xtilde::V
    ztilde::V
    delta_x::V
    delta_y::V
    Ax::V
    Px::V
    Aty::V
    rhs_x::V
    rhs_z::V
    tmp_n::V
    tmp_m::V
    work_n::V
    work_m::V
    rho::T
    rho_vec::V
    rho_inv_vec::V
    constr_type::VI
    linsys::LS
    refactor_count::Int
    prim_res::T
    dual_res::T
    scaled_prim_res::T
    scaled_dual_res::T
    obj_val::T
    dual_obj_val::T
    duality_gap::T
    scaled_duality_gap::T
    # The three terms of the gap, kept for its termination tolerance.
    xtPx::T
    qtx::T
    SCy::T
    rel_kkt_error::T
    last_rel_kkt::T
    # The primal-dual integral, accumulated only when `profile_primdual` is set. Two rules
    # over the same samples: `primdual_int` interpolates the gap linearly between them, as a
    # trapezoid; `primdual_int_log` interpolates it exponentially, which is what a
    # geometrically decaying gap actually does between samples. `last_gap_time` and
    # `last_gap` are the previous sample. Times are seconds since the loop started.
    primdual_int::Float64
    primdual_int_log::Float64
    last_gap_time::Float64
    last_gap::T
    # `solve!` stamps this before the loop, so the integral's clock starts where the loop
    # does rather than where the workspace was built.
    loop_start::UInt64
    rho_estimate::T
    rho_updates::Int
    iter::Int
    status::Status
    polished::Bool
    status_polish::PolishStatus
    setup_time::Float64
    update_time::Float64
    first_run::Bool
    solve_time::Float64
    polish_time::Float64
    settings::Settings{T}
end

"""
    check_bounds(l, u)

Throw unless `l ≤ u` elementwise, naming the first index that violates it.
"""
function check_bounds(l::Vector, u::Vector)
    for i in eachindex(l)
        l[i] <= u[i] || throw(ArgumentError("l must be elementwise ≤ u, violated at index $i: $(l[i]) > $(u[i])"))
    end
    return nothing
end

function check_bounds(l, u)
    # A whole-array reduction rather than an indexed loop, so an array that forbids scalar
    # indexing still validates. Naming the offending row needs indexing, so that runs on a
    # host copy and is paid only when the throw is happening anyway.
    all(l .<= u) && return nothing
    return check_bounds(Array(l), Array(u))
end

"""
    is_symmetric(M) -> Bool

Whether `M` equals its transpose, which [`setup`](@ref) requires of `P`.

The generic method is `issymmetric`, an entrywise scan over all `n²` positions. A
representation whose entries are structurally zero outside a known set overrides this and
compares only that set — `ext/PureOSQPBandedMatricesExt.jl` does, where the generic scan is
the largest single term in a banded `setup`. It is an override point for the same reason
[`is_convex`](@ref) is: the cost is a property of the representation, not of the problem.
"""
is_symmetric(M) = issymmetric(M)

function validate(P, q, A, l, u)
    n = size(P, 1)
    size(P, 2) == n || throw(ArgumentError("P must be square, got size $(size(P))"))
    size(A, 2) == n || throw(ArgumentError("size(A, 2) = $(size(A, 2)) must equal size(P, 1) = $n"))
    m = size(A, 1)
    length(q) == n || throw(ArgumentError("length(q) = $(length(q)) must equal size(P, 1) = $n"))
    length(l) == m || throw(ArgumentError("length(l) = $(length(l)) must equal size(A, 1) = $m"))
    length(u) == m || throw(ArgumentError("length(u) = $(length(u)) must equal size(A, 1) = $m"))
    is_symmetric(P) || throw(ArgumentError("P must be symmetric. Pass the full matrix or a Symmetric wrapper, not a stored triangle."))
    all(isfinite, q) || throw(ArgumentError("q must be finite, found NaN or Inf"))
    any(isnan, l) && throw(ArgumentError("l contains NaN"))
    any(isnan, u) && throw(ArgumentError("u contains NaN"))
    any(==(Inf), l) && throw(ArgumentError("l may not be +Inf"))
    any(==(-Inf), u) && throw(ArgumentError("u may not be -Inf"))
    check_bounds(l, u)
    check_storage(P, n, n)
    check_storage(A, m, n)
    return (n, m)
end

"""
    check_storage(M, rows, cols)

Establish that traversing `M`'s stored entries stays in range, or throw.

A representation whose column traversals index a weight vector by an index read out of the
matrix overrides this, so that those traversals can drop their per-entry bounds check. That
is worth 7.7× on the equilibration sweeps of a `P` holding 39 638 entries, where the check is
most of the per-entry work.

The generic method has nothing to check: [`structural_rows`](@ref) answers with `axes(M, 1)`,
which the compiler can already prove.
"""
check_storage(M, rows::Integer, cols::Integer) = nothing

"""
    setup(P, q, A, l, u; kwargs...) -> Workspace

Build a workspace for `min ½xᵀPx + qᵀx  s.t.  l ≤ Ax ≤ u`.

`P` must be a full symmetric matrix (or a `Symmetric` wrapper), not a stored triangle.
`P` and `A` may be any `AbstractMatrix` and are not copied or modified. Keyword arguments
are the fields of [`Settings`](@ref).

A `Symmetric` wrapper is accepted over any parent, but it costs something over a
`SparseMatrixCSC`: the sparse-factorization backends are keyed on that concrete type, so a
wrapped one descends past them, and equilibration walks the wrapper entrywise rather than by
stored column. Pass the full `SparseMatrixCSC` to reach those backends.
"""
function setup(
        P::AbstractMatrix, q::AbstractVector, A::AbstractMatrix,
        l::AbstractVector, u::AbstractVector; kwargs...
    )
    T = float(promote_type(eltype(P), eltype(q), eltype(A), eltype(l), eltype(u)))
    return setup(T, P, q, A, l, u; kwargs...)
end

# `@constprop :aggressive` because the settings have to reach inference as constants. A
# non-default keyword — `scaling = 0`, `linsys = :kkt` — otherwise arrives as a non-singleton
# `Pairs`, the compiler's size heuristic refuses to propagate it into a method this large, and
# `settings.scaling` stays unknown, leaving every branch below live. The return then merges one
# `Workspace` per reachable backend: `ReducedCholesky`, `FullKKT`, `IndirectCG` once Krylov is
# loaded, and whichever the ladder chose. That is three for a pair the ladder sends to
# `ReducedCholesky` anyway, and four for a pair with a backend of its own.
# `Base.Compiler.MAX_TYPEUNION_LENGTH` is 3, so the fourth widens the union to
# `Workspace{…} where LS` and every later `solve!` is a dynamic dispatch, which `--trim`
# rejects. An absent keyword leaves the empty `Pairs`, a singleton that folds without help.
#
# `linsys` is lifted out of the keywords and into a `Val` because constant propagation is not
# enough for it: naming a backend has to make the other branches *unreachable*, not merely
# narrow the merged return type, since the branch this eliminates is the one that reaches
# `choose_backend` and, through the sparse ladder, CHOLMOD's bindings. Propagation does not
# enter a method this large — the constant arrives as `linsys::Symbol` — so the choice is
# carried as a type parameter and the dead branches are gone by specialization instead.
Base.@constprop :aggressive function setup(
        ::Type{T}, P::AbstractMatrix, q::AbstractVector, A::AbstractMatrix,
        l::AbstractVector, u::AbstractVector; linsys::Symbol = :auto, kwargs...
    ) where {T <: Real}
    # Rejected here rather than left to `Settings`: past this point the name becomes a type
    # parameter, and an unusable one would specialize the whole of `setup_backend` before the
    # settings it cannot satisfy are ever built.
    linsys in LINSYS_OPTIONS || throw(
        ArgumentError("linsys must be :auto, :dense, :kkt or :indirect, got :$linsys")
    )
    return setup_backend(Val(linsys), T, P, q, A, l, u; kwargs...)
end

function setup_backend(
        ::Val{LS}, ::Type{T}, P::AbstractMatrix, q::AbstractVector, A::AbstractMatrix,
        l::AbstractVector, u::AbstractVector; kwargs...
    ) where {LS, T <: Real}
    t0 = time_ns()
    n, m = validate(P, q, A, l, u)
    settings = Settings{T}(; linsys = LS, kwargs...)
    if !is_convex(T, P, settings.sigma)
        throw(ArgumentError("P + sigma*I is not positive definite: P is indefinite, so the problem is not convex. Increase sigma if P + sigma*I can be made positive definite."))
    end
    inf = INFTY(T)
    # `similar`, not `collect`: the buffers inherit the caller's array type, and every
    # later buffer is `similar` to `q0` in turn.
    q0 = copyto!(similar(q, T, n), q)
    l0 = max.(copyto!(similar(l, T, m), l), -inf)
    u0 = min.(copyto!(similar(u, T, m), u), inf)
    # A single definition, and no default argument: a local function assigned more than
    # once is boxed, which turns every call through it into a dynamic dispatch and makes
    # the entry points fail `--trim`.
    buf(k, v) = fill!(similar(q0, T, k), v)
    z = zero(T)
    o = one(T)
    ctype = fill!(similar(q0, Int8, m), zero(Int8))
    # Equilibration and the ρ split run before the backend exists, because choosing a backend
    # well means building the reduced matrix and factoring it, and doing that with the values
    # the solver will actually use makes that factorization the setup factorization. The
    # buffers they fill are the ones the workspace then adopts.
    q, l, u = copy(q0), copy(l0), copy(u0)
    D, E = buf(n, o), buf(m, o)
    tmp_n, tmp_m, work_n = buf(n, z), buf(m, z), buf(n, z)
    c = equilibrate!(T, P, A, q0, l0, u0, q, l, u, D, E, tmp_n, tmp_m, work_n, n, settings.scaling)
    rho = clamp(settings.rho, RHO_MIN(T), RHO_MAX(T))
    rho_vec, rho_inv_vec = buf(m, o), buf(m, o)
    classify_rho!(
        ctype, rho_vec, rho_inv_vec, l, u, rho,
        INFTY(T) * MIN_SCALING(T), settings.rho_is_vec
    )
    make(ls) = Workspace{T, typeof(P), typeof(A), typeof(q0), typeof(ctype), typeof(ls)}(
        P, A, n, m,
        q0, l0, u0,
        q, l, u,
        D, E, c,
        buf(n, z), buf(m, z), buf(m, z), buf(n, z), buf(m, z), buf(n, z), buf(m, z), buf(n, z), buf(m, z),
        buf(m, z), buf(n, z), buf(n, z),
        buf(n, z), buf(m, z), tmp_n, tmp_m, work_n, buf(m, z),
        rho, rho_vec, rho_inv_vec, ctype,
        ls, 0,
        zero(T), zero(T), zero(T), zero(T), zero(T),
        zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), INFTY(T),
        0.0, 0.0, 0.0, zero(T), zero(UInt64),
        settings.rho, 0, 0, UNSOLVED, false, POLISH_NOT_PERFORMED,
        0.0, 0.0, true, 0.0, 0.0,
        settings,
    )
    # `LS` is a type parameter, so a named backend leaves exactly one of these branches live
    # and the rest are gone before the trimmer sees them. `settings` still holds and validates
    # the same value; reading it back here instead would put the choice beyond inference's
    # reach and make every branch reachable again.
    if LS === :kkt
        ws = make(FullKKT(q0, n, m))
        refactor!(ws)
        return finish_setup!(ws, t0)
    elseif LS === :dense
        # Past `choose_backend` entirely. Its two gates for a sparse `A` are measured
        # thresholds, and this is the way to overrule one that misjudges a problem.
        ws = make(ReducedCholesky(q0, n, m))
        refactor!(ws)
        return finish_setup!(ws, t0)
    elseif LS === :indirect
        ws = make(indirect_backend(q0, n, m))
        refactor!(ws)
        return finish_setup!(ws, t0)
    end
    # `choose_backend` picks by representation; the choice is settled here, once. The backend
    # is then part of the workspace's type, so the per-iteration solve dispatches statically.
    ls, factored = choose_backend(P, A, q0, n, m, D, E, c, rho_vec, settings.sigma)
    ws = make(ls)
    if factored || factorize!(ws.linsys, ws)
        ws.refactor_count += 1
        return finish_setup!(ws, t0)
    end
    kkt = make(FullKKT(q0, n, m))
    refactor!(kkt)
    return finish_setup!(kkt, t0)
end

"Record how long `setup` took. Called on each of its return paths."
function finish_setup!(ws::Workspace, t0::UInt64)
    ws.setup_time = (time_ns() - t0) / 1.0e9
    return ws
end

"""
    warm_start!(ws; x = nothing, y = nothing)

Seed the iterates in problem space. `z` is set to the scaled `Ax`.
"""
function warm_start!(ws::Workspace{T}; x = nothing, y = nothing) where {T}
    if !isnothing(x)
        length(x) == ws.n || throw(ArgumentError("length(x) must be $(ws.n)"))
        ws.x .= T.(x) ./ ws.D
    end
    if !isnothing(y)
        length(y) == ws.m || throw(ArgumentError("length(y) must be $(ws.m)"))
        ws.y .= ws.c .* T.(y) ./ ws.E
    end
    ws.m > 0 && mul_A!(ws.z, ws, ws.x)
    return ws
end

"""
    cold_start!(ws) -> ws

Zero the iterates `x`, `y` and `z`, discarding whatever warm-start state the workspace
held. The equilibration factors, the factorization and the problem data are untouched, so
the next [`solve!`](@ref) restarts the ADMM iteration without rebuilding anything.

The solver already does this where it must: at the start of a solve when
`warm_starting = false`, and after a run that ended without a meaningful primal-dual point,
since those iterates lie on a diverging ray. Call it directly to discard a warm start you
no longer want — after a large change in the data, say, when the previous solution is a
worse starting point than the origin.
"""
function cold_start!(ws::Workspace{T}) where {T}
    fill!(ws.x, zero(T))
    fill!(ws.y, zero(T))
    fill!(ws.z, zero(T))
    return ws
end
