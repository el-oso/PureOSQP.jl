"""
Accepts a `LinearMaps.LinearMap` wherever [`PureOSQP.setup`](@ref) takes a matrix.

A `LinearMap` is not an `AbstractMatrix`, so it reaches the solver through
[`PureOSQP.ProductOperator`](@ref). Wrapping is all this extension does; the protocol the
wrapper implements lives in `src/operator.jl` and needs no dependency.

What loading LinearMaps buys over wrapping by hand is the two declarations the wrapper cannot
compute: LinearMaps tracks `issymmetric` and `isposdef` on its maps, so a map built from a
symmetric positive-definite factor arrives already saying so.
"""
module PureOSQPLinearMapsExt

using PureOSQP
using LinearMaps
using LinearAlgebra

"""
    PureOSQP.ProductOperator{T}(map::LinearMap; symmetric, posdef)

Wrap a `LinearMap`, taking `symmetric` and `posdef` from the map's own traits unless the
caller states otherwise.

`issymmetric` and `isposdef` are properties a `LinearMap` carries rather than computes, so
reading them costs nothing and is what the map's author already declared.
"""
function PureOSQP.ProductOperator{T}(
        map::LinearMap;
        symmetric::Bool = issymmetric(map), posdef::Bool = isposdef(map),
        probe::Bool = false
    ) where {T <: Real}
    rows, cols = size(map)
    basis = zeros(T, probe ? cols : 0)
    column = zeros(T, probe ? rows : 0)
    mapt = adjoint(map)
    return PureOSQP.ProductOperator{T, typeof(map), typeof(mapt), typeof(basis)}(
        map, mapt, rows, cols, symmetric, posdef, probe, basis, column
    )
end

"""
    setup(P::LinearMap, q, A, l, u; kwargs...)

Solve with an operator cost, an operator constraint, or both.

Each `LinearMap` is wrapped in a [`PureOSQP.ProductOperator`](@ref); a matrix argument is
passed through untouched, so mixing the two is ordinary. The element type is taken from `q`,
which is the vector the solve is carried out in.

Equilibration cannot read an operator's entries, so `scaling = 0` is required unless the
wrapped map has a `PureOSQP.structural_rows` method; without it, `setup` throws and names
both remedies.
"""
function PureOSQP.setup(
        P::Union{LinearMap, AbstractMatrix}, q::AbstractVector, A::Union{LinearMap, AbstractMatrix},
        l::AbstractVector, u::AbstractVector; kwargs...
    )
    T = float(eltype(q))
    return PureOSQP.setup(as_operator(T, P), q, as_operator(T, A), l, u; kwargs...)
end

"""
    solve(P::LinearMap, q, A, l, u; kwargs...)

Set up and solve in one call, wrapping each `LinearMap` as [`setup`](@ref) does.

`solve` takes `AbstractMatrix` arguments, so a `LinearMap` reaches neither it nor the
`warm_start!` it forwards to without this.
"""
function PureOSQP.solve(
        P::Union{LinearMap, AbstractMatrix}, q::AbstractVector, A::Union{LinearMap, AbstractMatrix},
        l::AbstractVector, u::AbstractVector; kwargs...
    )
    T = float(eltype(q))
    return PureOSQP.solve(as_operator(T, P), q, as_operator(T, A), l, u; kwargs...)
end

"A `LinearMap` becomes a [`PureOSQP.ProductOperator`](@ref); anything else is already one."
as_operator(::Type{T}, M::LinearMap) where {T} = PureOSQP.ProductOperator{T}(M)
as_operator(::Type{T}, M::AbstractMatrix) where {T} = M

end
