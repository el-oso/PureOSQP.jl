"""
    ProductOperator{T,L} <: AbstractMatrix{T}

An operator that supplies products but no entries, presented as an `AbstractMatrix` so the
solver's own seams reach it.

`LinearMaps.LinearMap`, `SciMLOperators.AbstractSciMLOperator` and
`AbstractOperators.AbstractOperator` are each their own type hierarchy, none of them below
`AbstractMatrix`, so none can be handed to [`setup`](@ref) directly. This wraps one. The
solver never needed the entries of such an operator — [`is_materializable`](@ref) is `false`
here, so the rungs that would form a matrix decline and `linsys = :auto` reaches the
matrix-free backend — but it does need three answers no product can give:

  - `size`, taken from the wrapped operator;
  - whether the operator is symmetric, which [`setup`](@ref) requires of `P`;
  - whether `P + σI` is positive definite, which [`is_convex`](@ref) answers.

The last two are declared by the author rather than computed: verifying either from products
alone costs more than the solve. `symmetric` and `posdef` are ignored for a `ProductOperator`
standing in for `A`, which is neither.

Equilibration is the one seam this does not answer. Ruiz needs column and row ∞-norms, which
products do not give; `getindex` therefore throws a message naming the two ways out rather
than letting a `CanonicalIndexError` escape from inside a column walk. Either pass
`scaling = 0`, or give the wrapped type a [`structural_rows`](@ref) method — or, for a
representation with no columns to speak of, `PureOSQP.column_norms!` and
`PureOSQP.cost_norms!`.
"""
struct ProductOperator{T <: Real, L} <: AbstractMatrix{T}
    op::L
    rows::Int
    cols::Int
    symmetric::Bool
    posdef::Bool
end

"""
    ProductOperator{T}(op; symmetric = false, posdef = false)

Wrap `op`, which must answer `size`, `mul!` against a vector, and `mul!` against a vector for
`adjoint(op)` — so `op` has to be adjoint-able. `LinearMap`, `AbstractSciMLOperator` and
`AbstractOperator` all are.

`symmetric` and `posdef` are the author's declaration about the operator, checked by nothing.
They are required of a `P` and irrelevant to an `A`.
"""
function ProductOperator{T}(op; symmetric::Bool = false, posdef::Bool = false) where {T <: Real}
    rows, cols = size(op)
    return ProductOperator{T, typeof(op)}(op, rows, cols, symmetric, posdef)
end

Base.size(M::ProductOperator) = (M.rows, M.cols)

# The wrapped operator is the thing that knows how to multiply; both directions forward
# straight to it so no copy of the operand is made.
LinearAlgebra.mul!(y::AbstractVector, M::ProductOperator, x::AbstractVector) = mul!(y, M.op, x)

function LinearAlgebra.mul!(
        y::AbstractVector, Mt::Adjoint{<:Any, <:ProductOperator}, x::AbstractVector
    )
    return mul!(y, parent(Mt).op', x)
end

# The message names no type: interpolating one goes through `show(::IO, ::Type)`, a runtime
# dispatch `--trim` cannot resolve, and this branch is live code for an operator that has no
# entries to give.
function Base.getindex(M::ProductOperator, ::Integer, ::Integer)
    throw(
        ArgumentError(
            "this operator supplies products only and has no entries to read. Equilibration " *
                "needs column and row norms: pass `scaling = 0` to skip it, or give the " *
                "wrapped type a `PureOSQP.structural_rows` method, or override " *
                "`PureOSQP.column_norms!` and `PureOSQP.cost_norms!` for it."
        )
    )
end

is_materializable(::ProductOperator) = false
is_symmetric(M::ProductOperator) = M.symmetric
is_convex(::Type{T}, P::ProductOperator, sigma) where {T} = P.posdef

"""
    unpreconditioned!(dest)

Fill `dest` with the identity preconditioner, which is what [`reduced_diagonal!`](@ref)
returns for an operator whose diagonal is unavailable.

Jacobi preconditioning needs `P[j,j]` and the `ρ`-weighted column norms of `A`; neither is a
product, so an operator that supplies only products has nothing to build it from. Conjugate
gradients converges without a preconditioner, so this costs iterations rather than the
answer. An operator that can produce the reduced diagonal gives its wrapped type a
[`reduced_diagonal!`](@ref) method, which is more specific than the three below.
"""
unpreconditioned!(dest) = fill!(dest, one(eltype(dest)))

reduced_diagonal!(dest, ::Type{T}, P::ProductOperator, A, rho, E, D, sigma, c) where {T} =
    unpreconditioned!(dest)
reduced_diagonal!(dest, ::Type{T}, P, A::ProductOperator, rho, E, D, sigma, c) where {T} =
    unpreconditioned!(dest)
reduced_diagonal!(
    dest, ::Type{T}, P::ProductOperator, A::ProductOperator, rho, E, D, sigma, c
) where {T} = unpreconditioned!(dest)
