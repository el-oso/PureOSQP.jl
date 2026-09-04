"""
Accepts a `SciMLOperators.AbstractSciMLOperator` wherever [`PureOSQP.setup`](@ref) takes a
matrix.

An `AbstractSciMLOperator` is not an `AbstractMatrix`, so it reaches the solver through
[`PureOSQP.ProductOperator`](@ref), as a `LinearMaps.LinearMap` does. Wrapping is all this
extension does; the protocol the wrapper implements lives in `src/operator.jl` and needs no
dependency.

An operator built with `*` — and with `kron` or `inv` — needs scratch for its intermediates
before it can multiply, and reports `iscached` as `false` until `cache_operator` has given it
some. `setup` refuses such an operator, so the failure lands at setup with a remedy rather
than partway into the first iteration. Sums and scalings need nothing, and report `iscached`
as `true` from the start.

`Aᵀ` is applied every iteration, so `setup` also refuses an operator that cannot supply one.
A `FunctionOperator` built without `op_adjoint` and not declared symmetric reports
`has_adjoint` as `false`, and its adjoint falls back to converting itself to a matrix, which
an operator has no entries for.

An operator that holds entries — `MatrixOperator`, and the `DiagonalOperator` built from one —
is unwrapped to the matrix underneath instead. Wrapping it would hide entries the solver can
use, forcing `scaling = 0` and a matrix-free solve on a problem that needs neither.

The wrapper holds the operator and its adjoint, both taken once at `setup`. Updating an
operator in place afterwards, through `update_coefficients!` or a `MatrixOperator`'s
`update_func!`, is applied to `A` but not to `Aᵀ`: build a new workspace instead.
"""
module PureOSQPSciMLOperatorsExt

using PureOSQP
using SciMLOperators
using SciMLOperators: AbstractSciMLOperator
using LinearAlgebra

"""
    uncached()

Throw the refusal a composed operator owes its caller when it has no scratch for its
intermediates.

The message interpolates nothing: `show(::IO, ::Type)` is a runtime dispatch `--trim` cannot
resolve, and this branch is live code for every operator the caller has not cached.
"""
uncached() = throw(
    ArgumentError(
        "this operator has no cache, so it cannot multiply. A SciMLOperator built with `*`, " *
            "`kron` or `inv` needs scratch for its intermediates: build it with " *
            "`op = cache_operator(op, x)` for an `x` the length of the operator's input, then " *
            "pass the result."
    )
)

"""
    no_adjoint()

Throw the refusal an operator owes its caller when it cannot supply its own transpose.

`Aᵀ` runs every iteration, and an operator without one falls back to converting itself to a
matrix, which is the thing an operator has no entries for. The message interpolates nothing,
for the reason [`uncached`](@ref) does not.
"""
no_adjoint() = throw(
    ArgumentError(
        "this operator cannot supply its transpose, which the solver applies every " *
            "iteration. Give the operator an `op_adjoint`, or declare it symmetric with " *
            "`issymmetric = true` if it is."
    )
)

"""
    PureOSQP.ProductOperator{T}(op::AbstractSciMLOperator; symmetric, posdef)

Wrap a SciMLOperator, taking `symmetric` and `posdef` from its own traits unless the caller
states otherwise.

`issymmetric` and `isposdef` are properties a SciMLOperator carries rather than computes, so
reading them costs nothing and is what the operator's author already declared.

An operator that reports `iscached` as `false`, or that cannot supply its transpose, is
refused here rather than at its first product, which is several iterations into a solve and
reports a failed assertion from inside the operator instead of a remedy.
"""
function PureOSQP.ProductOperator{T}(
        op::AbstractSciMLOperator;
        symmetric::Bool = issymmetric(op), posdef::Bool = isposdef(op),
        probe::Bool = false
    ) where {T <: Real}
    iscached(op) || uncached()
    has_adjoint(op) || no_adjoint()
    rows, cols = size(op)
    basis = zeros(T, probe ? cols : 0)
    column = zeros(T, probe ? rows : 0)
    opt = adjoint(op)
    return PureOSQP.ProductOperator{T, typeof(op), typeof(opt), typeof(basis)}(
        op, opt, rows, cols, symmetric, posdef, probe, basis, column
    )
end

"""
    setup(P::AbstractSciMLOperator, q, A, l, u; kwargs...)

Solve with an operator cost, an operator constraint, or both.

Each operator is wrapped in a [`PureOSQP.ProductOperator`](@ref); a matrix argument is passed
through untouched, so mixing the two is ordinary. The element type is taken from `q`, which is
the vector the solve is carried out in.

Equilibration cannot read an operator's entries, so `scaling = 0` is required unless the
wrapped operator has a `PureOSQP.structural_rows` method; without it, `setup` throws and names
both remedies.
"""
function PureOSQP.setup(
        P::Union{AbstractSciMLOperator, AbstractMatrix}, q::AbstractVector,
        A::Union{AbstractSciMLOperator, AbstractMatrix},
        l::AbstractVector, u::AbstractVector; kwargs...
    )
    T = float(eltype(q))
    return PureOSQP.setup(as_operator(T, P), q, as_operator(T, A), l, u; kwargs...)
end

"""
    solve(P::AbstractSciMLOperator, q, A, l, u; kwargs...)

Set up and solve in one call, wrapping each operator as [`setup`](@ref) does.

`solve` takes `AbstractMatrix` arguments, so an operator reaches neither it nor the
`warm_start!` it forwards to without this.
"""
function PureOSQP.solve(
        P::Union{AbstractSciMLOperator, AbstractMatrix}, q::AbstractVector,
        A::Union{AbstractSciMLOperator, AbstractMatrix},
        l::AbstractVector, u::AbstractVector; kwargs...
    )
    T = float(eltype(q))
    return PureOSQP.solve(as_operator(T, P), q, as_operator(T, A), l, u; kwargs...)
end

"""
    as_operator(T, M)

Present `M` as something [`setup`](@ref) accepts.

An operator holding a matrix gives the matrix back: its entries reach the equilibration and
the factoring backends, which a wrapper would hide. Any other SciMLOperator is wrapped. A
matrix is already what the solver wants.
"""
as_operator(::Type{T}, M::MatrixOperator) where {T} = convert(AbstractMatrix, M)
as_operator(::Type{T}, M::AbstractSciMLOperator) where {T} = PureOSQP.ProductOperator{T}(M)
as_operator(::Type{T}, M::AbstractMatrix) where {T} = M

end
