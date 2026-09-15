"""
    Problem{T,MP,MA,V}

The problem data an algorithm solves: the caller's `P` and `A`, the equilibration factors,
and the scratch the products and the backends use. Shared by every [`LinearSystem`](@ref)
backend and, in time, by more than one algorithm.

`P` and `A` are mutable fields: [`update!`](@ref) replaces them when the caller supplies new
matrices, and every other field stays as it was.
"""
mutable struct Problem{T <: Real, MP <: AbstractMatrix, MA <: AbstractMatrix, V <: AbstractVector{T}}
    P::MP
    A::MA
    n::Int
    m::Int
    q0::V   # caller's data, clamped to ±INFTY
    l0::V
    u0::V
    q::V    # equilibrated
    l::V
    u::V
    D::V    # Ruiz factors
    E::V
    c::T
    scaling::Int   # sweeps requested at setup; 0 means D, E and c are identity
    tmp_n::V       # scratch of mul_A!/mul_At!/mul_P!
    tmp_m::V
    work_n::V      # scratch of reduced_rhs! and the backends
    work_m::V
end

"""
    Problem(T, P, q, A, l, u; scaling) -> Problem

Validate `P`, `q`, `A`, `l`, `u`, allocate the buffers `similar` to `q` follows, and run
`scaling` sweeps of Ruiz equilibration into them.

Does not check convexity: whether `P + shift*I` is positive definite is a question for the
algorithm that uses this `Problem` — `σ` for ADMM — not for the data on its own, so the
caller runs [`is_convex`](@ref) itself, with whichever shift it needs.
"""
function Problem(
        ::Type{T}, P::AbstractMatrix, q::AbstractVector, A::AbstractMatrix,
        l::AbstractVector, u::AbstractVector; scaling::Integer
    ) where {T <: Real}
    n, m = validate(P, q, A, l, u)
    return validated_problem(T, n, m, P, q, A, l, u, scaling)
end

"""
    validated_problem(T, n, m, P, q, A, l, u, scaling) -> Problem

`Problem` for data `validate` has already accepted, for a caller that has to
run other checks between validation and equilibration and should not pay for validation twice.
"""
function validated_problem(
        ::Type{T}, n::Integer, m::Integer, P::AbstractMatrix, q::AbstractVector,
        A::AbstractMatrix, l::AbstractVector, u::AbstractVector, scaling::Integer
    ) where {T <: Real}
    inf = INFTY(T)
    q0 = copyto!(similar(q, T, n), q)
    l0 = max.(copyto!(similar(l, T, m), l), -inf)
    u0 = min.(copyto!(similar(u, T, m), u), inf)
    # A single definition, and no default argument: a local function assigned more than once
    # is boxed, which turns every call through it into a dynamic dispatch.
    buf(k, v) = fill!(similar(q0, T, k), v)
    z = zero(T)
    o = one(T)
    q, l, u = copy(q0), copy(l0), copy(u0)
    D, E = buf(n, o), buf(m, o)
    tmp_n, tmp_m, work_n, work_m = buf(n, z), buf(m, z), buf(n, z), buf(m, z)
    c = equilibrate!(T, P, A, q0, l0, u0, q, l, u, D, E, tmp_n, tmp_m, work_n, n, scaling)
    return Problem{T, typeof(P), typeof(A), typeof(q0)}(
        P, A, n, m, q0, l0, u0, q, l, u, D, E, c, Int(scaling), tmp_n, tmp_m, work_n, work_m,
    )
end
