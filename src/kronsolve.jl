"""
    KroneckerReduced{T,M,V} <: LinearSystem

The reduced system when `A` is `A₁ ⊗ A₂` and `P` is a multiple of the identity, which makes

    R = (cμ + σ)I + ρ (G₁ ⊗ G₂),   Gᵢ = AᵢᵀAᵢ

diagonal in the eigenbasis of the two factors. With `Gᵢ = QᵢΛᵢQᵢᵀ`,

    R⁻¹ = (Q₁ ⊗ Q₂) diag(1 / (cμ + σ + ρ λ₁ₐ λ₂ᵦ)) (Q₁ ⊗ Q₂)ᵀ

so a solve is four small matrix multiplications and a scale: `O(n₁n₂(n₁ + n₂))` against the
`O(n₁²n₂²)` of a dense `symv`, in `O(n₁² + n₂²)` storage against `O(n₁²n₂²)`. Factorizing is
two eigendecompositions of the factors, `O(n₁³ + n₂³)`.

`Q1` and `Q2` hold the eigenvectors and `dinv` the reciprocal of the diagonal, laid out so
that `dinv[b, a]` pairs `λ₂ᵦ` with `λ₁ₐ` — the second factor running fastest, matching `vec`.

**Every condition in the first sentence is load-bearing**, and the log records the measurement
for each: a Kronecker `P` that is not a scalar multiple of `I` breaks the diagonalization, a
non-uniform `ρ` breaks it, and so does equilibration, because `c·μ·D²` is diagonal but not
scalar. [`kronecker_rung`](@ref) declines all three rather than returning a wrong answer.
"""
mutable struct KroneckerReduced{
        T <: Real, M <: AbstractMatrix{T}, V <: AbstractVector{T},
    } <: LinearSystem
    Q1::M
    Q2::M
    lambda1::V
    lambda2::V
    dinv::M          # `n₂×n₁`, the reciprocal diagonal in the eigenbasis
    X::M             # `n₂×n₁` scratch, the right-hand side reshaped
    Z::M             # `n₂×n₁` scratch, one product in
    mu::T            # the `μ` of `P = μI`, settled by the rung
end

"""
    KroneckerReduced(proto::AbstractVector, n1, n2)

Build the backend's storage as `similar(proto, ...)`, following the array type of the data it
was given. See [`ReducedCholesky`](@ref) on why `proto` is a vector.
"""
function KroneckerReduced(
        proto::AbstractVector{T}, n1::Integer, n2::Integer, mu::T
    ) where {T <: Real}
    return KroneckerReduced{T, Matrix{T}, Vector{T}}(
        similar(proto, T, n1, n1), similar(proto, T, n2, n2),
        similar(proto, T, n1), similar(proto, T, n2),
        similar(proto, T, n2, n1), similar(proto, T, n2, n1), similar(proto, T, n2, n1),
        mu,
    )
end

backend_name(::KroneckerReduced) = :kronecker

# Two eigenbases and a diagonal is everything stored; neither factor's Gram is kept.
function backend_info(ls::KroneckerReduced)
    n1, n2 = size(ls.Q1, 1), size(ls.Q2, 1)
    stored = n1 * n1 + n2 * n2 + n1 * n2
    return BackendInfo(backend_name(ls), true, :reduced, n1 * n2, stored)
end

"""
    kronecker_rung(P, A, proto, n, m, D, E, c, rho_vec, sigma) -> (LinearSystem, Bool) or nothing

Ladder rung for `A = A₁ ⊗ A₂` with a scalar `P`. Declines unless every condition the
diagonalization needs holds: `P` a multiple of the identity, `ρ` uniform, and no equilibration
scaling in force.
"""
kronecker_rung(P, A, proto::AbstractVector, n::Integer, m::Integer, D, E, c, rho_vec, sigma) =
    nothing

function kronecker_rung(
        P, A::KroneckerOperator, proto::AbstractVector{T}, n::Integer, m::Integer,
        D, E, c, rho_vec, sigma
    ) where {T <: Real}
    # `μ` is read here, where a `nothing` is a branch rather than a value flowing on, and the
    # backend then carries it as a `T`. Leaving `Union{Nothing,T}` for `factorize!` to narrow
    # is what `--trim` refuses to resolve.
    mu = scalar_multiple(P)
    isnothing(mu) && return nothing
    # `ρ` enters as `ρ (G₁ ⊗ G₂)` only when it is one number. A single equality row or a free
    # row gives it two values and the eigenbasis stops diagonalizing.
    isempty(rho_vec) && return nothing
    all(==(first(rho_vec)), rho_vec) || return nothing
    # Equilibration puts `c·μ·D²` in the reduced matrix, which is diagonal but not scalar, so
    # only unscaled data keeps the structure. `scaling = 0` is what produces this.
    (all(isone, D) && all(isone, E) && isone(c)) || return nothing
    n1, n2 = size(A.A1, 2), size(A.A2, 2)
    return (KroneckerReduced(proto, n1, n2, T(mu)), false)
end

function factorize!(ls::KroneckerReduced{T}, ws)::Bool where {T}
    A = ws.A
    rho = first(ws.rho_vec)
    shift = ws.c * ls.mu + ws.settings.sigma
    # `Gᵢ = AᵢᵀAᵢ` is formed at factor size and thrown away; only its eigenbasis is kept.
    F1 = eigen(Symmetric(A.A1' * A.A1))
    F2 = eigen(Symmetric(A.A2' * A.A2))
    copyto!(ls.Q1, F1.vectors)
    copyto!(ls.Q2, F2.vectors)
    copyto!(ls.lambda1, F1.values)
    copyto!(ls.lambda2, F2.values)
    for a in axes(ls.dinv, 2), b in axes(ls.dinv, 1)
        d = shift + rho * ls.lambda1[a] * ls.lambda2[b]
        d > zero(T) || return false
        ls.dinv[b, a] = inv(d)
    end
    return true
end

function solve_system!(ls::KroneckerReduced, ws, rhs_x, rhs_z)::Nothing
    reduced_rhs!(ws, rhs_x, rhs_z)
    # Copied into the backend's own `n₂×n₁` scratch rather than reshaped in place: `reshape`
    # of a vector allocates an array header, and this runs every iteration.
    copyto!(ls.X, ws.work_n)
    mul!(ls.Z, ls.Q2', ls.X)          # Q₂ᵀ X
    mul!(ls.X, ls.Z, ls.Q1)           # Q₂ᵀ X Q₁
    # A loop, not `ls.X .*= ls.dinv`: an in-place broadcast has `X` on both sides, which
    # leaves an `unaliascopy` branch AllocCheck reports as an allocation. See
    # `src/elementwise.jl`, which does the same for the vector cases.
    for i in eachindex(ls.X, ls.dinv)
        ls.X[i] *= ls.dinv[i]
    end
    mul!(ls.Z, ls.Q2, ls.X)           # Q₂ (…)
    mul!(ls.X, ls.Z, ls.Q1')          # Q₂ (…) Q₁ᵀ
    copyto!(ws.xtilde, ls.X)
    ws.m > 0 && mul_A!(ws.ztilde, ws, ws.xtilde)
    return nothing
end

@verify KroneckerReduced trim_compat = true
