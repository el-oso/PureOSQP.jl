"""
    IPMSettings{T}

Parameters of the interior-point method, selected by `algorithm = :ipm` in [`setup`](@ref)
and [`solve`](@ref). The keyword arguments of those two functions are then these fields.

- `max_iter = 100` — outer iterations.
- `time_limit = Inf` — seconds of iteration before the run stops with `TIME_LIMIT_REACHED`,
  measured and reported as in [`Settings`](@ref).
- `eps_abs = 1e-8`, `eps_rel = 1e-8` — termination tolerances, with the same meaning as in
  [`Settings`](@ref); only the defaults differ.
- `eps_prim_inf = 1e-8`, `eps_dual_inf = 1e-8` — tolerances of the primal and dual
  infeasibility certificate tests, as in [`Settings`](@ref).
- `scaling = 10` — Ruiz equilibration sweeps, as in [`Settings`](@ref).
- `check_termination = 1` — test for termination every this many iterations; `0` tests only
  at `max_iter`.
- `check_dualgap = true`, `scaled_termination = false` — as in [`Settings`](@ref).
- `reg_primal = 1e-8`, `reg_dual = 1e-8` — the proximal regularization `δ_p` on the primal
  block and `δ_d` on each slack row, in scaled space. They are never refined away, and
  convexity is checked on `P + reg_primal*I`.
- `max_reg_bumps = 5` — when the Newton system cannot be factorized, both regularizations
  are multiplied by ten and the factorization retried, at most this many times in one solve;
  past that the run ends `NUMERICAL_ERROR`. Every solve starts from `reg_primal` and
  `reg_dual`.
- `refine_iter = 1` — refinement steps per Newton solve against the regularized system,
  which correct rounding in the factorization.
- `step_fraction = 0.99` — the fraction of the step to the boundary that is taken.
- `warm_starting = true` — start a re-solve from the previous point.
- `linsys = :auto` — the backend, as in [`Settings`](@ref). `:kronecker` and `:lowrank` are
  refused; `:indirect` runs only with a caller-supplied `preconditioner` (see
  [`setup`](@ref)), and then refines nothing: `refine_iter` defaults to `0` there.
- `cg_max_iter = 500` — conjugate-gradient iterations per Newton solve with
  `linsys = :indirect`. A solve that spends them all, or that conjugate gradients abandons
  because the operator or the preconditioner is not positive definite, is a missed solve.
- `cg_tol_fraction = 0.1` — each solve stops once the two-norm of its recursively updated
  residual is below this fraction of `min(μ, ‖r‖∞)`, the barrier parameter and the largest
  Newton residual, floored at `eps(T)` relative to the right-hand side.
- `cg_fail_limit = 3` — this many missed solves in a row end the run `NUMERICAL_ERROR`.
- `polishing = false` — refine the solution by [`polish_kernel!`](@ref) once the run ends
  `SOLVED` or `SOLVED_INACCURATE`, as in [`Settings`](@ref). Recommended before taking a
  derivative: an interior-point solution carries inactive-row multipliers of size
  `μ_final`, which is small but not the near-zero `active_kkt` requires.
- `polish_refine_iter = 3`, `delta = 1e-6` — iterative-refinement steps and the
  regularization of the polishing solve, as in [`Settings`](@ref).
- `verbose = false` — reserved for a per-iteration report; nothing is printed yet.

A reduced backend solves `P̃ + δ_p I + Ãᵀ diag(w) Ã`, whose weights reach `1/δ_d` on
equality rows and on active inequality rows, so its conditioning is bounded by
`(λ_max(P̃) + ‖Ã‖²/δ_d) / (λ_min(P̃) + δ_p)`. That bound is not checked.

Any `T <: Real` is accepted. The `1e-8` defaults of the four tolerances and the two
regularizations hold for `Float64` and finer arithmetic (`BigFloat`); in coarser arithmetic
(`Float32`) each defaults to `sqrt(eps(T))`. A number type that wraps another, such
as `ForwardDiff.Dual`, takes the precision of `float(T)`. A value passed explicitly is used as
given.
"""
struct IPMSettings{T <: Real}
    max_iter::Int
    time_limit::T
    eps_abs::T
    eps_rel::T
    eps_prim_inf::T
    eps_dual_inf::T
    scaling::Int
    check_termination::Int
    check_dualgap::Bool
    scaled_termination::Bool
    reg_primal::T
    reg_dual::T
    max_reg_bumps::Int
    refine_iter::Int
    step_fraction::T
    warm_starting::Bool
    linsys::Symbol
    cg_max_iter::Int
    cg_tol_fraction::T
    cg_fail_limit::Int
    polishing::Bool
    polish_refine_iter::Int
    delta::T
    verbose::Bool
end

"""
    precision_eps(T) -> eps

The spacing of `T`'s arithmetic, `eps(float(T))`: the value type's `eps` for a number type
that wraps one and defines `eps` through it (as `ForwardDiff.Dual` does), `Float64`'s for
integers and rationals.
"""
@inline precision_eps(::Type{T}) where {T <: Real} = eps(float(T))

"""
    ipm_floor(T) -> T

The default of the interior-point regularizations, tolerances and short-step threshold:
`1e-8` in `Float64` and finer arithmetic, `sqrt(eps)` in arithmetic coarser than `Float64`.
"""
@inline function ipm_floor(::Type{T}) where {T <: Real}
    e = precision_eps(T)
    return e > eps(Float64) ? T(sqrt(e)) : T(1.0e-8)
end

function IPMSettings{T}(;
        max_iter = 100, time_limit = Inf, eps_abs = ipm_floor(T), eps_rel = ipm_floor(T),
        eps_prim_inf = ipm_floor(T), eps_dual_inf = ipm_floor(T), scaling = 10,
        check_termination = 1, check_dualgap = true, scaled_termination = false,
        linsys = :auto, reg_primal = ipm_floor(T), reg_dual = ipm_floor(T), max_reg_bumps = 5,
        refine_iter = linsys === :indirect ? 0 : 1, step_fraction = 0.99, warm_starting = true,
        cg_max_iter = 500, cg_tol_fraction = 0.1, cg_fail_limit = 3,
        polishing = false, polish_refine_iter = 3, delta = 1.0e-6, verbose = false,
    ) where {T <: Real}
    linsys in LINSYS_OPTIONS || throw(
        ArgumentError("linsys must be one of $(join(LINSYS_OPTIONS, ", ")), got :$linsys")
    )
    max_iter > 0 || throw(ArgumentError("max_iter must be positive, got $max_iter"))
    time_limit > 0 || throw(ArgumentError("time_limit must be positive (Inf disables it), got $time_limit"))
    eps_abs >= 0 && eps_rel >= 0 || throw(ArgumentError("eps_abs and eps_rel must be non-negative"))
    eps_abs > 0 || eps_rel > 0 || throw(ArgumentError("at least one of eps_abs, eps_rel must be positive"))
    eps_prim_inf > 0 && eps_dual_inf > 0 || throw(ArgumentError("eps_prim_inf and eps_dual_inf must be positive"))
    max_reg_bumps >= 0 || throw(ArgumentError("max_reg_bumps must be non-negative, got $max_reg_bumps"))
    scaling >= 0 || throw(ArgumentError("scaling must be non-negative, got $scaling"))
    check_termination >= 0 || throw(ArgumentError("check_termination must be non-negative, got $check_termination"))
    reg_primal > 0 || throw(ArgumentError("reg_primal must be positive, got $reg_primal"))
    reg_dual > 0 || throw(ArgumentError("reg_dual must be positive, got $reg_dual"))
    refine_iter >= 0 || throw(ArgumentError("refine_iter must be non-negative, got $refine_iter"))
    0 < step_fraction < 1 || throw(ArgumentError("step_fraction must lie in (0, 1), got $step_fraction"))
    cg_max_iter > 0 || throw(ArgumentError("cg_max_iter must be positive, got $cg_max_iter"))
    cg_tol_fraction > 0 || throw(ArgumentError("cg_tol_fraction must be positive, got $cg_tol_fraction"))
    cg_fail_limit > 0 || throw(ArgumentError("cg_fail_limit must be positive, got $cg_fail_limit"))
    polish_refine_iter >= 0 || throw(ArgumentError("polish_refine_iter must be non-negative"))
    delta > 0 || throw(ArgumentError("delta must be positive, got $delta"))
    return IPMSettings{T}(
        Int(max_iter), T(time_limit), T(eps_abs), T(eps_rel), T(eps_prim_inf),
        T(eps_dual_inf), Int(scaling), Int(check_termination),
        Bool(check_dualgap), Bool(scaled_termination), T(reg_primal), T(reg_dual),
        Int(max_reg_bumps), Int(refine_iter), T(step_fraction), Bool(warm_starting), Symbol(linsys),
        Int(cg_max_iter), T(cg_tol_fraction), Int(cg_fail_limit),
        Bool(polishing), Int(polish_refine_iter), T(delta), Bool(verbose),
    )
end
