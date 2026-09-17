# A products-only operator, shared by `operator_protocol.jl` and `strictmode_audit.jl`.
#
# `P = Diagonal(d) + α v vᵀ` with `α ≥ 0`, applied through a closure. Nothing here stores an
# `n×n` object, so the entries this package's dense and polishing paths read do not exist;
# `PureOSQP.is_materializable` is how the operator says so, and `setup` then descends past
# the dense terminal to the matrix-free rung.
#
# The overrides below are the whole protocol: `size`, `mul!` in both directions, the two
# predicates `setup` asks before choosing a backend, and the per-column seam the matrix-free
# preconditioner needs.
using PureOSQP, LinearAlgebra

struct LazyPSD{T, F} <: AbstractMatrix{T}
    apply!::F
    d::Vector{T}
    v::Vector{T}
    alpha::T
end

"""
    LazyPSD(d, v, alpha) -> LazyPSD

`Diagonal(d) + alpha * v * v'` as an operator. `alpha` must be nonnegative, which is what
makes the convexity test below a scan of `d`.
"""
function LazyPSD(d::Vector{T}, v::Vector{T}, alpha::T) where {T <: Real}
    alpha >= zero(T) || throw(ArgumentError("alpha must be nonnegative, got $alpha"))
    length(v) == length(d) || throw(DimensionMismatch("v and d must have equal length"))
    # An explicit loop over the destination's indices, not a broadcast: AllocCheck reports a
    # broadcast as allocating because of an aliasing branch it cannot rule out, and
    # `eachindex(y, d, x, v)` builds its DimensionMismatch message through code that costs
    # the same proof. This is the shape a caller's `mul!` needs for the hot-path guarantees
    # to survive it.
    apply! = function (y, x)
        s = alpha * dot(v, x)
        for i in eachindex(y)
            y[i] = d[i] * x[i] + s * v[i]
        end
        return y
    end
    return LazyPSD{T, typeof(apply!)}(apply!, d, v, alpha)
end

Base.size(P::LazyPSD) = (length(P.d), length(P.d))

LinearAlgebra.mul!(y::AbstractVector, P::LazyPSD, x::AbstractVector) = P.apply!(y, x)
LinearAlgebra.mul!(
    y::AbstractVector, P::Adjoint{<:Any, <:LazyPSD}, x::AbstractVector
) = parent(P).apply!(y, x)

PureOSQP.is_materializable(::LazyPSD) = false

# `Diagonal(d) + α v vᵀ` is symmetric by construction, and with `α ≥ 0` the rank-one term is
# positive semidefinite, so `P + σI` is positive definite exactly when `d + σ` is.
LinearAlgebra.issymmetric(::LazyPSD) = true
PureOSQP.is_convex(::Type{T}, P::LazyPSD, sigma) where {T} =
    all(dj -> dj + sigma > zero(T), P.d)

# The per-column seam. `P[j, j] = d[j] + α v[j]²` in closed form; `A` is an ordinary matrix
# here, so its part of the row sum still follows `structural_rows`.
function PureOSQP.reduced_diagonal!(
        dest, ::Type{T}, P::LazyPSD, A, rho, E, D, sigma, c
    ) where {T}
    for j in eachindex(dest)
        dj = D[j]
        pjj = P.d[j] + P.alpha * P.v[j]^2
        acc = c * dj * pjj * dj + sigma
        for i in PureOSQP.structural_rows(A, j)
            a = E[i] * T(A[i, j]) * dj
            acc += rho[i] * a * a
        end
        dest[j] = inv(max(acc, sqrt(eps(T))))
    end
    return dest
end

"The same operator as a matrix, for comparison against the paths that read entries."
materialize(P::LazyPSD) = Matrix(Diagonal(P.d)) .+ P.alpha .* (P.v * P.v')
