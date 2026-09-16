"""
Accepts a `LinearMaps.LinearMap` wherever [`PureQPBase.setup`](@ref) takes a matrix.

A `LinearMap` is not an `AbstractMatrix`, so it reaches the solver through
[`PureQPBase.ProductOperator`](@ref). Wrapping is all this extension does; the protocol the
wrapper implements lives in `src/core/operator.jl` and needs no dependency.

What loading LinearMaps buys over wrapping by hand is the two declarations the wrapper cannot
compute: LinearMaps tracks `issymmetric` and `isposdef` on its maps, so a map built from a
symmetric positive-definite factor arrives already saying so.
"""
module PureQPBaseLinearMapsExt

using PureQPBase
using LinearMaps
using LinearAlgebra

"""
    PureQPBase.ProductOperator{T}(map::LinearMap; symmetric, posdef)

Wrap a `LinearMap`, taking `symmetric` and `posdef` from the map's own traits unless the
caller states otherwise.

`issymmetric` and `isposdef` are properties a `LinearMap` carries rather than computes, so
reading them costs nothing and is what the map's author already declared.
"""
function PureQPBase.ProductOperator{T}(
        map::LinearMap;
        symmetric::Bool = issymmetric(map), posdef::Bool = isposdef(map),
        probe::Bool = false
    ) where {T <: Real}
    rows, cols = size(map)
    basis = zeros(T, probe ? cols : 0)
    column = zeros(T, probe ? rows : 0)
    mapt = adjoint(map)
    return PureQPBase.ProductOperator{T, typeof(map), typeof(mapt), typeof(basis)}(
        map, mapt, rows, cols, symmetric, posdef, probe, basis, column
    )
end

"""
    setup(P::LinearMap, q, A, l, u, alg = OperatorSplitting(); kwargs...)

Solve with an operator cost, an operator constraint, or both.

Each `LinearMap` is wrapped in a [`PureQPBase.ProductOperator`](@ref); a matrix argument is
passed through untouched, so mixing the two is ordinary. The element type is taken from `q`,
which is the vector the solve is carried out in.

Equilibration cannot read an operator's entries, so `scaling = 0` is required unless the
wrapped map has a `PureQPBase.structural_rows` method; without it, `setup` throws and names
both remedies.
"""
function PureQPBase.setup(
        P::Union{LinearMap, AbstractMatrix}, q::AbstractVector, A::Union{LinearMap, AbstractMatrix},
        l::AbstractVector, u::AbstractVector, alg::PureQPBase.QPAlgorithm...; kwargs...
    )
    T = float(eltype(q))
    return PureQPBase.setup(as_operator(T, P), q, as_operator(T, A), l, u, alg...; kwargs...)
end

"""
    solve(P::LinearMap, q, A, l, u, alg = OperatorSplitting(); kwargs...)

Set up and solve in one call, wrapping each `LinearMap` as [`setup`](@ref) does.

`solve` takes `AbstractMatrix` arguments, so a `LinearMap` reaches neither it nor the
`warm_start!` it forwards to without this.
"""
function PureQPBase.solve(
        P::Union{LinearMap, AbstractMatrix}, q::AbstractVector, A::Union{LinearMap, AbstractMatrix},
        l::AbstractVector, u::AbstractVector, alg::PureQPBase.QPAlgorithm...; kwargs...
    )
    T = float(eltype(q))
    return PureQPBase.solve(as_operator(T, P), q, as_operator(T, A), l, u, alg...; kwargs...)
end

"A `LinearMap` becomes a [`PureQPBase.ProductOperator`](@ref); anything else is already one."
as_operator(::Type{T}, M::LinearMap) where {T} = PureQPBase.ProductOperator{T}(M)
as_operator(::Type{T}, M::AbstractMatrix) where {T} = M

end
