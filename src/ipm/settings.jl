"""
    IPMSettings{T}

Parameters of the interior-point method, selected by `algorithm = :ipm` in [`setup`](@ref)
and [`solve`](@ref). The keyword arguments of those two functions are then these fields.

- `max_iter = 100` — outer iterations.
- `eps_abs = 1e-8`, `eps_rel = 1e-8` — termination tolerances, with the same meaning as in
  [`Settings`](@ref); only the defaults differ.
- `scaling = 10` — Ruiz equilibration sweeps, as in [`Settings`](@ref).
- `check_termination = 1` — test for termination every this many iterations; `0` tests only
  at `max_iter`.
- `check_dualgap = true`, `scaled_termination = false` — as in [`Settings`](@ref).
- `reg_primal = 1e-8`, `reg_dual = 1e-8` — the proximal regularization `δ_p` on the primal
  block and `δ_d` on each slack row, in scaled space. They are never refined away, and
  convexity is checked on `P + reg_primal*I`.
- `refine_iter = 1` — refinement steps per Newton solve against the regularized system,
  which correct rounding in the factorization.
- `step_fraction = 0.99` — the fraction of the step to the boundary that is taken.
- `warm_starting = true` — start a re-solve from the previous point.
- `linsys = :auto` — the backend, as in [`Settings`](@ref). `:indirect` and `:kronecker` are
  refused.

A reduced backend solves `P̃ + δ_p I + Ãᵀ diag(w) Ã`, whose weights reach `1/δ_d` on
equality rows and on active inequality rows, so its conditioning is bounded by
`(λ_max(P̃) + ‖Ã‖²/δ_d) / (λ_min(P̃) + δ_p)`. That bound is not checked.
"""
struct IPMSettings{T <: Real}
    max_iter::Int
    eps_abs::T
    eps_rel::T
    scaling::Int
    check_termination::Int
    check_dualgap::Bool
    scaled_termination::Bool
    reg_primal::T
    reg_dual::T
    refine_iter::Int
    step_fraction::T
    warm_starting::Bool
    linsys::Symbol
end

function IPMSettings{T}(;
        max_iter = 100, eps_abs = 1.0e-8, eps_rel = 1.0e-8, scaling = 10,
        check_termination = 1, check_dualgap = true, scaled_termination = false,
        reg_primal = 1.0e-8, reg_dual = 1.0e-8, refine_iter = 1, step_fraction = 0.99,
        warm_starting = true, linsys = :auto,
    ) where {T <: Real}
    linsys in LINSYS_OPTIONS || throw(
        ArgumentError("linsys must be one of $(join(LINSYS_OPTIONS, ", ")), got :$linsys")
    )
    max_iter > 0 || throw(ArgumentError("max_iter must be positive, got $max_iter"))
    eps_abs >= 0 && eps_rel >= 0 || throw(ArgumentError("eps_abs and eps_rel must be non-negative"))
    eps_abs > 0 || eps_rel > 0 || throw(ArgumentError("at least one of eps_abs, eps_rel must be positive"))
    scaling >= 0 || throw(ArgumentError("scaling must be non-negative, got $scaling"))
    check_termination >= 0 || throw(ArgumentError("check_termination must be non-negative, got $check_termination"))
    reg_primal > 0 || throw(ArgumentError("reg_primal must be positive, got $reg_primal"))
    reg_dual > 0 || throw(ArgumentError("reg_dual must be positive, got $reg_dual"))
    refine_iter >= 0 || throw(ArgumentError("refine_iter must be non-negative, got $refine_iter"))
    0 < step_fraction < 1 || throw(ArgumentError("step_fraction must lie in (0, 1), got $step_fraction"))
    return IPMSettings{T}(
        Int(max_iter), T(eps_abs), T(eps_rel), Int(scaling), Int(check_termination),
        Bool(check_dualgap), Bool(scaled_termination), T(reg_primal), T(reg_dual),
        Int(refine_iter), T(step_fraction), Bool(warm_starting), Symbol(linsys),
    )
end
