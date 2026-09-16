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

"""
    validate_update!(prob, ls; P, A, q, l, u) -> Nothing

Throw unless the proposed data can replace what `prob` holds: dimensions, representation,
symmetry, finiteness and bound ordering, plus whatever the backend `ls` needs through
[`check_update`](@ref). Reads the arguments and `prob`, and writes nothing.

Convexity is not checked here, because the shift it is checked at belongs to the algorithm.
"""
function validate_update!(
        prob::Problem{T, MP, MA}, ls; P = nothing, A = nothing, q = nothing, l = nothing,
        u = nothing
    ) where {T, MP, MA}
    n, m = prob.n, prob.m
    if !isnothing(P)
        size(P) == (n, n) || throw(ArgumentError("P must stay $(n)×$(n), got $(size(P))"))
        P isa MP || throw(
            ArgumentError(
                "P must keep the representation the workspace was built with: its linear-" *
                    "system backend is built for that representation and would read only the " *
                    "structure it implies. Rebuild the workspace with setup to change it."
            )
        )
        is_symmetric(P) || throw(ArgumentError("P must be symmetric"))
        is_materializable(P) || check_symmetric_products(P, prob.q0)
        is_materializable(P) && check_finite(P, n, n, "P")
        check_storage(P, n, n)
    end
    if !isnothing(A)
        size(A) == (m, n) || throw(ArgumentError("A must stay $(m)×$(n), got $(size(A))"))
        A isa MA || throw(
            ArgumentError(
                "A must keep the representation the workspace was built with: its linear-" *
                    "system backend is built for that representation and would read only the " *
                    "structure it implies. Rebuild the workspace with setup to change it."
            )
        )
        is_materializable(A) && check_finite(A, m, n, "A")
        check_storage(A, m, n)
    end
    if !isnothing(P) || !isnothing(A)
        check_update(ls, isnothing(P) ? prob.P : P, isnothing(A) ? prob.A : A)
    end
    if !isnothing(q)
        length(q) == n || throw(ArgumentError("length(q) must be $n, got $(length(q))"))
        all(isfinite, q) || throw(ArgumentError("q must be finite, found NaN or Inf"))
    end
    if !isnothing(l) || !isnothing(u)
        # Lengths first: the walks below index every row of both proposals, and a short one
        # would reach the end of a vector rather than this message.
        isnothing(l) || length(l) == m ||
            throw(ArgumentError("length(l) must be $m, got $(length(l))"))
        isnothing(u) || length(u) == m ||
            throw(ArgumentError("length(u) must be $m, got $(length(u))"))
        inf = INFTY(T)
        if !isnothing(l)
            any(isnan, l) && throw(ArgumentError("l contains NaN"))
            any(li -> li == Inf, l) && throw(ArgumentError("l may not be +Inf"))
        end
        if !isnothing(u)
            any(isnan, u) && throw(ArgumentError("u contains NaN"))
            any(ui -> ui == -Inf, u) && throw(ArgumentError("u may not be -Inf"))
        end
        # The ordering test runs on the clamped proposals, not on what the workspace holds,
        # so a pair that fails it leaves the old bounds in place.
        for i in 1:m
            li = isnothing(l) ? prob.l0[i] : max(T(l[i]), -inf)
            ui = isnothing(u) ? prob.u0[i] : min(T(u[i]), inf)
            li <= ui ||
                throw(ArgumentError("l must be elementwise ≤ u, violated at index $i: $li > $ui"))
        end
    end
    return nothing
end

"""
    adopt_update!(prob; P, A, q, l, u) -> prob

Replace what `prob` holds with the data given, and reapply the existing equilibration to it.
Runs only after [`validate_update!`](@ref) and every algorithm-specific check have passed.
"""
function adopt_update!(
        prob::Problem{T}; P = nothing, A = nothing, q = nothing, l = nothing, u = nothing
    ) where {T}
    # Every check reads the arguments, never the problem's own fields, so the matrices are
    # adopted only once none of them can refuse. A refusal that had already replaced `P` or
    # `A` would leave the workspace holding a matrix its factorization and its buffers were
    # not built for, and the next solve reads out of range.
    isnothing(P) || (prob.P = P)
    isnothing(A) || (prob.A = A)
    isnothing(q) || (prob.q0 .= q)
    isnothing(l) || (prob.l0 .= max.(T.(l), -INFTY(T)))
    isnothing(u) || (prob.u0 .= min.(T.(u), INFTY(T)))

    # Reapply the existing equilibration to whatever changed.
    isnothing(q) || (prob.q .= prob.c .* prob.D .* prob.q0)
    if !isnothing(l) || !isnothing(u)
        prob.l .= prob.E .* prob.l0
        prob.u .= prob.E .* prob.u0
    end
    return prob
end
