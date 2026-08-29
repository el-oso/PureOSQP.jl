module PureOSQPMathOptInterfaceExt

"""
    PureOSQPMathOptInterfaceExt

MathOptInterface wrapper, loaded when MathOptInterface is. Constraints arrive as a
`MatrixOfConstraints` with a `Hyperrectangle`, which is already the `l <= Ax <= u` form the
solver wants, so the only assembly left is appending a row per bounded variable.
"""

import MathOptInterface as MOI
import PureOSQP
using SparseArrays
using LinearAlgebra: dot

const _SETS{T} = Union{MOI.EqualTo{T}, MOI.GreaterThan{T}, MOI.LessThan{T}, MOI.Interval{T}}

MOI.Utilities.@product_of_sets(
    _RowSets, MOI.EqualTo{T}, MOI.GreaterThan{T}, MOI.LessThan{T}, MOI.Interval{T},
)

const _Cache{T} = MOI.Utilities.GenericModel{
    T,
    MOI.Utilities.ObjectiveContainer{T},
    MOI.Utilities.VariablesContainer{T},
    MOI.Utilities.MatrixOfConstraints{
        T,
        MOI.Utilities.MutableSparseMatrixCSC{T, Int, MOI.Utilities.OneBasedIndexing},
        MOI.Utilities.Hyperrectangle{T},
        _RowSets{T},
    },
}

mutable struct Optimizer{T} <: MOI.AbstractOptimizer   # reached as PureOSQP.Optimizer()
    settings::Dict{Symbol, Any}
    silent::Bool
    sense::MOI.OptimizationSense
    obj_constant::T
    sets::_RowSets{T}
    bound_row::Vector{Int}
    n_affine::Int
    n::Int
    P::SparseMatrixCSC{T, Int}
    q::Vector{T}
    A::SparseMatrixCSC{T, Int}
    l::Vector{T}
    u::Vector{T}
    sol::Union{Nothing, PureOSQP.Solution{T}}
    Ax::Vector{T}
end

function Optimizer{T}(; kwargs...) where {T}
    o = Optimizer{T}(
        Dict{Symbol, Any}(), false, MOI.MIN_SENSE, zero(T),
        _RowSets{T}(), Int[], 0, 0,
        spzeros(T, 0, 0), T[], spzeros(T, 0, 0), T[], T[], nothing, T[]
    )
    for (k, v) in kwargs
        MOI.set(o, MOI.RawOptimizerAttribute(String(k)), v)
    end
    return o
end
# The core owns the name `PureOSQP.Optimizer` as a function; this is its only method, so
# `Model(PureOSQP.Optimizer)` and `MOI.instantiate` both reach the struct above.
PureOSQP.Optimizer(; kwargs...) = Optimizer{Float64}(; kwargs...)

MOI.get(::Optimizer, ::MOI.SolverName) = "PureOSQP"
MOI.get(::Optimizer, ::MOI.SolverVersion) = string(pkgversion(PureOSQP))

MOI.is_empty(o::Optimizer) = o.n == 0 && o.n_affine == 0 && o.sol === nothing

function MOI.empty!(o::Optimizer{T}) where {T}
    o.sense = MOI.MIN_SENSE
    o.obj_constant = zero(T)
    o.sets = _RowSets{T}()
    empty!(o.bound_row); o.n_affine = 0; o.n = 0
    o.sol = nothing; empty!(o.Ax)
    return
end

MOI.supports_incremental_interface(::Optimizer) = false

MOI.supports_constraint(::Optimizer{T}, ::Type{MOI.VariableIndex}, ::Type{<:_SETS{T}}) where {T} = true
MOI.supports_constraint(::Optimizer{T}, ::Type{MOI.ScalarAffineFunction{T}}, ::Type{<:_SETS{T}}) where {T} = true
MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true
MOI.supports(::Optimizer{T}, ::MOI.ObjectiveFunction{<:Union{MOI.VariableIndex, MOI.ScalarAffineFunction{T}, MOI.ScalarQuadraticFunction{T}}}) where {T} = true

MOI.supports(::Optimizer, ::MOI.Silent) = true
MOI.set(o::Optimizer, ::MOI.Silent, v::Bool) = (o.silent = v; nothing)
MOI.get(o::Optimizer, ::MOI.Silent) = o.silent

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true
MOI.set(o::Optimizer, ::MOI.TimeLimitSec, v::Real) = (o.settings[:time_limit] = Float64(v); nothing)
MOI.set(o::Optimizer, ::MOI.TimeLimitSec, ::Nothing) = (delete!(o.settings, :time_limit); nothing)
MOI.get(o::Optimizer, ::MOI.TimeLimitSec) = get(o.settings, :time_limit, nothing)

MOI.supports(::Optimizer, a::MOI.RawOptimizerAttribute) = Symbol(a.name) in fieldnames(PureOSQP.Settings)
function MOI.set(o::Optimizer, a::MOI.RawOptimizerAttribute, v)
    MOI.supports(o, a) || throw(MOI.UnsupportedAttribute(a))
    o.settings[Symbol(a.name)] = v
    return
end
function MOI.get(o::Optimizer, a::MOI.RawOptimizerAttribute)
    haskey(o.settings, Symbol(a.name)) || throw(MOI.GetAttributeNotAllowed(a))
    return o.settings[Symbol(a.name)]
end

_csc(A::MOI.Utilities.MutableSparseMatrixCSC{T, Int, MOI.Utilities.OneBasedIndexing}) where {T} =
    SparseMatrixCSC{T, Int}(A.m, A.n, A.colptr, A.rowval, A.nzval)

function MOI.copy_to(dest::Optimizer{T}, src::MOI.ModelLike) where {T}
    MOI.empty!(dest)
    cache = _Cache{T}()
    index_map = MOI.copy_to(cache, src)
    A_aff = _csc(cache.constraints.coefficients)
    n = size(A_aff, 2)
    dest.n, dest.n_affine = n, size(A_aff, 1)
    dest.sets = cache.constraints.sets
    l = copy(cache.constraints.constants.lower)
    u = copy(cache.constraints.constants.upper)
    lo, up = cache.variables.lower, cache.variables.upper
    dest.bound_row = zeros(Int, n)
    cols = Int[]
    for j in 1:n
        if lo[j] > typemin(T) || up[j] < typemax(T)
            push!(cols, j); dest.bound_row[j] = length(cols)
            push!(l, lo[j]); push!(u, up[j])
        end
    end
    nb = length(cols)
    dest.A = vcat(A_aff, sparse(1:nb, cols, ones(T, nb), nb, n))
    dest.l, dest.u = l, u
    P, q, c = _objective(cache, T, n)
    dest.sense = MOI.get(cache, MOI.ObjectiveSense())
    if dest.sense == MOI.MAX_SENSE
        P, q, c = -P, -q, -c
    end
    dest.P, dest.q, dest.obj_constant = P, q, c
    return index_map
end

function _objective(cache, ::Type{T}, n) where {T}
    if MOI.get(cache, MOI.ObjectiveSense()) == MOI.FEASIBILITY_SENSE
        return spzeros(T, n, n), zeros(T, n), zero(T)
    end
    F = MOI.get(cache, MOI.ObjectiveFunctionType())
    g = convert(MOI.ScalarQuadraticFunction{T}, MOI.get(cache, MOI.ObjectiveFunction{F}()))
    I, J, V = Int[], Int[], T[]
    for t in g.quadratic_terms
        i, j = t.variable_1.value, t.variable_2.value
        push!(I, i); push!(J, j); push!(V, t.coefficient)
        i == j || (push!(I, j); push!(J, i); push!(V, t.coefficient))
    end
    q = zeros(T, n)
    for t in g.affine_terms
        q[t.variable.value] += t.coefficient
    end
    return sparse(I, J, V, n, n), q, g.constant
end

function MOI.optimize!(o::Optimizer{T}) where {T}
    settings = copy(o.settings)
    o.silent && (settings[:verbose] = false)
    ws = PureOSQP.setup(T, o.P, o.q, o.A, o.l, o.u; settings...)
    o.sol = PureOSQP.solve!(ws)
    xr = _is_cert(o.sol.status) ? o.sol.dual_inf_cert : o.sol.x
    o.Ax = isempty(xr) ? fill(T(NaN), length(o.l)) : o.A * xr
    return
end

### results

const _TERMINATION = Dict(
    PureOSQP.UNSOLVED => MOI.OTHER_ERROR,
    PureOSQP.SOLVED => MOI.OPTIMAL,
    PureOSQP.SOLVED_INACCURATE => MOI.ALMOST_OPTIMAL,
    PureOSQP.PRIMAL_INFEASIBLE => MOI.INFEASIBLE,
    PureOSQP.PRIMAL_INFEASIBLE_INACCURATE => MOI.ALMOST_INFEASIBLE,
    PureOSQP.DUAL_INFEASIBLE => MOI.DUAL_INFEASIBLE,
    PureOSQP.DUAL_INFEASIBLE_INACCURATE => MOI.ALMOST_DUAL_INFEASIBLE,
    PureOSQP.MAX_ITER_REACHED => MOI.ITERATION_LIMIT,
    PureOSQP.TIME_LIMIT_REACHED => MOI.TIME_LIMIT,
    PureOSQP.INTERRUPTED => MOI.INTERRUPTED,
    PureOSQP.NON_CONVEX => MOI.INVALID_MODEL,
)

function MOI.get(o::Optimizer, ::MOI.TerminationStatus)
    o.sol === nothing && return MOI.OPTIMIZE_NOT_CALLED
    return _TERMINATION[o.sol.status]
end

MOI.get(o::Optimizer, ::MOI.RawStatusString) =
    o.sol === nothing ? "optimize! not called" : string(o.sol.status)

function MOI.get(o::Optimizer, ::MOI.ResultCount)
    o.sol === nothing && return 0
    return o.sol.status == PureOSQP.NON_CONVEX ? 0 : 1
end

function MOI.get(o::Optimizer, attr::MOI.PrimalStatus)
    (o.sol === nothing || attr.result_index != 1) && return MOI.NO_SOLUTION
    s = o.sol.status
    s == PureOSQP.SOLVED && return MOI.FEASIBLE_POINT
    s == PureOSQP.SOLVED_INACCURATE && return MOI.NEARLY_FEASIBLE_POINT
    s == PureOSQP.DUAL_INFEASIBLE && return MOI.INFEASIBILITY_CERTIFICATE
    s == PureOSQP.DUAL_INFEASIBLE_INACCURATE && return MOI.NEARLY_INFEASIBILITY_CERTIFICATE
    _stopped_early(s) && return MOI.UNKNOWN_RESULT_STATUS
    return MOI.NO_SOLUTION
end

function MOI.get(o::Optimizer, attr::MOI.DualStatus)
    (o.sol === nothing || attr.result_index != 1) && return MOI.NO_SOLUTION
    s = o.sol.status
    s == PureOSQP.SOLVED && return MOI.FEASIBLE_POINT
    s == PureOSQP.SOLVED_INACCURATE && return MOI.NEARLY_FEASIBLE_POINT
    s == PureOSQP.PRIMAL_INFEASIBLE && return MOI.INFEASIBILITY_CERTIFICATE
    s == PureOSQP.PRIMAL_INFEASIBLE_INACCURATE && return MOI.NEARLY_INFEASIBILITY_CERTIFICATE
    _stopped_early(s) && return MOI.UNKNOWN_RESULT_STATUS
    return MOI.NO_SOLUTION
end

_flip(o::Optimizer, v) = o.sense == MOI.MAX_SENSE ? -v : v

_is_cert(s) = s == PureOSQP.DUAL_INFEASIBLE || s == PureOSQP.DUAL_INFEASIBLE_INACCURATE

"Statuses that stopped on a budget rather than on the residuals, so the point is real but
its status as a solution is unknown."
_stopped_early(s) = s == PureOSQP.MAX_ITER_REACHED || s == PureOSQP.TIME_LIMIT_REACHED ||
    s == PureOSQP.INTERRUPTED
_is_pinf(s) = s == PureOSQP.PRIMAL_INFEASIBLE || s == PureOSQP.PRIMAL_INFEASIBLE_INACCURATE

function MOI.get(o::Optimizer{T}, attr::MOI.ObjectiveValue) where {T}
    MOI.check_result_index_bounds(o, attr)
    if _is_cert(o.sol.status)
        return _flip(o, dot(o.q, o.sol.dual_inf_cert))
    end
    return _flip(o, o.sol.obj_val + o.obj_constant)
end

function _support(o::Optimizer{T}, y) where {T}
    s = zero(T)
    for i in eachindex(y)
        yi = y[i]
        abs(yi) <= 1.0e-10 && continue
        s -= yi > 0 ? o.u[i] * yi : o.l[i] * yi
    end
    return s
end

function MOI.get(o::Optimizer{T}, attr::MOI.DualObjectiveValue) where {T}
    MOI.check_result_index_bounds(o, attr)
    _is_pinf(o.sol.status) && return _flip(o, _support(o, o.sol.prim_inf_cert))
    return _flip(o, o.sol.dual_obj_val + o.obj_constant)
end

MOI.get(o::Optimizer, ::MOI.SolveTimeSec) = o.sol.run_time
MOI.get(o::Optimizer, ::MOI.BarrierIterations) = o.sol.iter

function MOI.get(o::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(o, attr)
    _is_cert(o.sol.status) && return o.sol.dual_inf_cert[vi.value]
    return o.sol.x[vi.value]
end

_row(o::Optimizer, ci::MOI.ConstraintIndex{MOI.VariableIndex}) = o.n_affine + o.bound_row[ci.value]
_row(o::Optimizer, ci::MOI.ConstraintIndex{<:MOI.ScalarAffineFunction}) = MOI.Utilities.rows(o.sets, ci)

function MOI.get(o::Optimizer, attr::MOI.ConstraintPrimal, ci::MOI.ConstraintIndex)
    MOI.check_result_index_bounds(o, attr)
    return o.Ax[_row(o, ci)]
end

_split(::Type{<:MOI.GreaterThan}, d) = max(d, zero(d))
_split(::Type{<:MOI.LessThan}, d) = min(d, zero(d))
_split(::Type{<:Union{MOI.EqualTo, MOI.Interval}}, d) = d

function MOI.get(o::Optimizer, attr::MOI.ConstraintDual, ci::MOI.ConstraintIndex{F, S}) where {F, S}
    MOI.check_result_index_bounds(o, attr)
    y = _is_pinf(o.sol.status) ? o.sol.prim_inf_cert : o.sol.y
    d = -y[_row(o, ci)]
    return F === MOI.VariableIndex ? _split(S, d) : d
end

MOI.get(o::Optimizer, ::MOI.NumberOfVariables) = o.n
MOI.get(o::Optimizer, ::MOI.ListOfVariableIndices) = MOI.VariableIndex.(1:o.n)

end
