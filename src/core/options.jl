"""
    QPAlgorithm

The method a solve runs, passed as the sixth positional argument of [`setup`](@ref) and
[`solve`](@ref): [`OperatorSplitting`](@ref) (the default) or [`InteriorPoint`](@ref). An
algorithm object holds the parameters only that method reads; everything both methods read
is an [`Options`](@ref) field, passed as a keyword argument.

A subtype implements the methods `TypeContracts.describe(QPAlgorithm)` lists, checked at
precompilation; see [Interfaces](@ref).
"""
abstract type QPAlgorithm end

"""
    QPWorkspace{T}

Solver state built by [`setup`](@ref): an [`OperatorSplittingWorkspace`](@ref) or an
[`InteriorPointWorkspace`](@ref). Both hold `algorithm`, the element-typed algorithm
parameters, and `options`, the [`Options`](@ref) in force.

A subtype implements the methods `TypeContracts.describe(QPWorkspace)` lists, checked at
precompilation, and holds the fields the methods written for every workspace read: `prob`,
`linsys`, `algorithm`, `options`, `x`, `y`, `z`, `status` and `polished`. See [Interfaces](@ref).
"""
abstract type QPWorkspace{T <: Real} end

"""
    Options{T}

The settings both algorithms read, passed as keyword arguments to [`setup`](@ref),
[`solve`](@ref) and [`update_settings!`](@ref). A keyword that is not given takes the default
of the algorithm the solve runs, [`default_options`](@ref); a keyword that is given is used as
given.

| option | `OperatorSplitting` | `InteriorPoint` | meaning |
|---|---|---|---|
| `max_iter` | `4000` | `100` | iterations of the method |
| `time_limit` | `Inf` | `Inf` | seconds of iteration before the run stops with `TIME_LIMIT_REACHED` |
| `eps_abs`, `eps_rel` | `1e-3` | `ipm_floor(T)` | absolute and relative termination tolerances |
| `eps_prim_inf`, `eps_dual_inf` | `1e-4` | `ipm_floor(T)` | tolerances of the infeasibility certificate tests |
| `scaling` | `10` | `10` | Ruiz equilibration sweeps; `0` turns equilibration off |
| `check_termination` | `25` | `1` | test for termination every this many iterations; `0` tests only at `max_iter` |
| `check_dualgap` | `true` | `true` | require the duality gap to pass its tolerance too |
| `scaled_termination` | `false` | `false` | judge the residuals in the equilibrated space |
| `warm_starting` | `true` | `true` | start a re-solve from the previous point |
| `linsys` | `:auto` | `:auto` | the backend (see `PureOSQP.LINSYS_OPTIONS`) |
| `polishing` | `false` | `false` | refine a converged point by an active-set solve |
| `polish_refine_iter` | `3` | `3` | iterative-refinement steps of the polishing solve |
| `delta` | `1e-6` | `1e-6` | regularization of the polishing solve |
| `cg_max_iter` | `20` | `500` | conjugate-gradient iterations per linear solve with `linsys = :indirect` |
| `cg_tol_fraction` | `0.15` | `0.1` | each conjugate-gradient solve stops below this fraction of the level its algorithm sets |

`ipm_floor(T)` is `1e-8` in `Float64` and finer arithmetic and `sqrt(eps(T))` in coarser
arithmetic (`Float32`).
"""
struct Options{T <: Real}
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
    warm_starting::Bool
    linsys::Symbol
    polishing::Bool
    polish_refine_iter::Int
    delta::T
    cg_max_iter::Int
    cg_tol_fraction::T
end

"The backends `linsys` may name. [`setup`](@ref) rejects anything else before turning the
choice into a type parameter, so an unusable name costs an error and not a specialization.

`:auto` descends the selection ladder. `:dense`, `:kkt` and `:indirect` name a backend
outright. The rest name a *kind* the pair must admit — `:sparse` factors the reduced or
KKT matrix sparsely, `:diagonal`, `:tridiagonal`, `:block`, `:kronecker` and `:lowrank`
name their structured backends — and are refused, with the condition stated, when the pair
does not admit one. They are the escape hatch for a pair the ladder's measured thresholds
misjudge: the same backend the ladder would have chosen, chosen by the caller instead."
const LINSYS_OPTIONS = (
    :auto, :dense, :kkt, :indirect,
    :sparse, :diagonal, :tridiagonal, :block, :kronecker, :lowrank,
)

# The options whose defaults differ by algorithm have no keyword default here: they come from
# `algorithm_defaults`, merged in ahead of the caller's keywords.
function Options{T}(;
        max_iter, time_limit = Inf, eps_abs, eps_rel, eps_prim_inf, eps_dual_inf, scaling = 10,
        check_termination, check_dualgap = true, scaled_termination = false,
        warm_starting = true, linsys = :auto, polishing = false, polish_refine_iter = 3,
        delta = 1.0e-6, cg_max_iter, cg_tol_fraction,
    ) where {T <: Real}
    linsys in LINSYS_OPTIONS || throw(
        ArgumentError("linsys must be one of $(join(LINSYS_OPTIONS, ", ")), got :$linsys")
    )
    max_iter > 0 || throw(ArgumentError("max_iter must be positive, got $max_iter"))
    time_limit > 0 || throw(ArgumentError("time_limit must be positive (Inf disables it), got $time_limit"))
    eps_abs >= 0 && eps_rel >= 0 || throw(ArgumentError("eps_abs and eps_rel must be non-negative"))
    eps_abs > 0 || eps_rel > 0 || throw(ArgumentError("at least one of eps_abs, eps_rel must be positive"))
    eps_prim_inf > 0 && eps_dual_inf > 0 || throw(ArgumentError("eps_prim_inf and eps_dual_inf must be positive"))
    scaling >= 0 || throw(ArgumentError("scaling must be non-negative, got $scaling"))
    check_termination >= 0 || throw(ArgumentError("check_termination must be non-negative, got $check_termination"))
    polish_refine_iter >= 0 || throw(ArgumentError("polish_refine_iter must be non-negative"))
    delta > 0 || throw(ArgumentError("delta must be positive, got $delta"))
    cg_max_iter > 0 || throw(ArgumentError("cg_max_iter must be positive, got $cg_max_iter"))
    0 < cg_tol_fraction <= 1 || throw(
        ArgumentError("cg_tol_fraction must lie in (0, 1], got $cg_tol_fraction")
    )
    return Options{T}(
        Int(max_iter), T(time_limit), T(eps_abs), T(eps_rel), T(eps_prim_inf), T(eps_dual_inf),
        Int(scaling), Int(check_termination), Bool(check_dualgap), Bool(scaled_termination),
        Bool(warm_starting), Symbol(linsys), Bool(polishing), Int(polish_refine_iter), T(delta),
        Int(cg_max_iter), T(cg_tol_fraction),
    )
end

"""
    default_options(alg, T) -> Options{T}

The options a solve with algorithm `alg` in element type `T` runs with when no keyword is
passed. The two algorithms differ in `max_iter`, the four tolerances, `check_termination`,
`cg_max_iter` and `cg_tol_fraction` (see [`Options`](@ref)).
"""
default_options(alg::QPAlgorithm, ::Type{T}) where {T <: Real} = Options{T}(; algorithm_defaults(alg, T)...)

"""
    settings_tuple(s) -> NamedTuple

`s` as a keyword-ready `NamedTuple`. `Val(fieldcount(...))` keeps the loop unrolled, so the
field accesses resolve statically.
"""
function settings_tuple(s::S) where {S}
    return NamedTuple{fieldnames(S)}(ntuple(i -> getfield(s, i), Val(fieldcount(S))))
end

"""
    check_option_names(kwargs)

Throw, naming the algorithm it belongs to, for a keyword that is a parameter of an algorithm
rather than an option. Any other name that is not an option is left for the
[`Options`](@ref) constructor to refuse.
"""
function check_option_names(kwargs)
    for name in keys(kwargs)
        name in OPTION_NAMES && continue
        owner = name in OPERATOR_SPLITTING_NAMES ? "OperatorSplitting" :
            name in INTERIOR_POINT_NAMES ? "InteriorPoint" : ""
        isempty(owner) || throw(
            ArgumentError(
                lazy"$name is a parameter of $owner, not an option: pass it as $owner($name = ...)."
            )
        )
    end
    return nothing
end

const OPTION_NAMES = fieldnames(Options{Float64})
