"""
The backend for a reduced matrix that is banded but wider than tridiagonal.

`bandwidth(R) = max(bandwidth(P), 2 bandwidth(A))`, so a `Tridiagonal` `A` already squares
past what `SymTridiagonal` can hold. BandedMatrices.jl stores any bandwidth, and its
`cholesky` is LAPACK's banded factorization: `O(n b²)` to factor, `O(n b)` to solve, against
the `O(n³)` and `O(n²)` the dense backend would spend on the same problem.

Loading BandedMatrices is what makes this reachable. Without it a banded problem is served by
[`PureOSQP.ReducedCholesky`](@ref) as before, correctly but densely.
"""
module PureOSQPBandedMatricesExt

using PureOSQP
using BandedMatrices
using LinearAlgebra

using PureOSQP: LinearSystem, reduced_rhs!, mul_A!

"""
    BandedReduced{T,M,F} <: LinearSystem

The reduced system stored as a `Symmetric` `BandedMatrix` and factored by a banded Cholesky.

`R` is the storage the reduced matrix is assembled into, and `cholesky!` then overwrites it
with the factor — so between factorizations `R` holds the factor, not the matrix. Every
refactorization refills the bands from scratch before factoring, which is why that is safe.

Unlike the tridiagonal backend's `ldlt`, a banded Cholesky reports indefiniteness through
`issuccess`, so [`PureOSQP.factorize!`](@ref) needs no separate test of the pivots.
"""
mutable struct BandedReduced{T <: Real, M, F} <: LinearSystem
    R::M
    fact::F
    bw::Int
end

"""
    banded_bandwidth(M) -> Int

The number of off-diagonals `M` can hold a nonzero in, for the representations this backend
accepts. It is a property of the type, not of the values, so a numerically zero band is
still stored — refusing to shrink it keeps the answer independent of the data.
"""
banded_bandwidth(::Diagonal) = 0
banded_bandwidth(::Bidiagonal) = 1
banded_bandwidth(::Tridiagonal) = 1
banded_bandwidth(::SymTridiagonal) = 1
banded_bandwidth(M::BandedMatrix) = max(bandwidth(M, 1), bandwidth(M, 2))
banded_bandwidth(M::Symmetric{<:Any, <:BandedMatrix}) = banded_bandwidth(parent(M))

"""
    reduced_bandwidth(P, A) -> Int

`max(bandwidth(P), 2 bandwidth(A))` — the bandwidth of `c D P D + σI + Ãᵀ diag(ρ) Ã`.
Diagonal scaling preserves a bandwidth and squaring `A` doubles it.
"""
reduced_bandwidth(P, A) = max(banded_bandwidth(P), 2 * banded_bandwidth(A))

const BandedLike = Union{
    Diagonal, Bidiagonal, Tridiagonal, SymTridiagonal,
    BandedMatrix, Symmetric{<:Any, <:BandedMatrix},
}

# The types PureOSQP itself has no backend for. The two methods below split on `A` so that
# neither overlaps the other, nor the `(Diagonal, Diagonal)`, `(SymTridiagonal, Diagonal)`
# and `(Diagonal|SymTridiagonal, Bidiagonal)` methods in `src/linsys.jl`, whose reduced
# matrices are narrow enough for the LinearAlgebra backends.
const WideBand = Union{Tridiagonal, BandedMatrix, Symmetric{<:Any, <:BandedMatrix}}
const NarrowBand = Union{Diagonal, Bidiagonal}

PureOSQP.choose_backend(
    P::BandedLike, A::WideBand, proto::AbstractVector{T}, n::Integer, m::Integer,
    D, E, c, rho_vec, sigma
) where {T <: Real} = banded_backend(P, A, proto, n, m)

PureOSQP.choose_backend(
    P::WideBand, A::NarrowBand, proto::AbstractVector{T}, n::Integer, m::Integer,
    D, E, c, rho_vec, sigma
) where {T <: Real} = banded_backend(P, A, proto, n, m)

function banded_backend(P, A, proto::AbstractVector{T}, n::Integer, m::Integer) where {T <: Real}
    b = reduced_bandwidth(P, A)
    # Below bandwidth 2 the LinearAlgebra backends already apply and are cheaper than a
    # banded factorization; at half the matrix or wider, the dense path wins outright.
    (b < 2 || b >= n ÷ 2) && return PureOSQP.dense_rung(P, A, proto, n, m)
    R = BandedMatrix{T}(undef, (n, n), (b, b))
    fill!(R.data, zero(T))
    for i in 1:n
        R[i, i] = one(T)
    end
    fact = cholesky(Symmetric(R))
    return (BandedReduced{T, typeof(R), typeof(fact)}(R, fact, b), false)
end

PureOSQP.backend_name(::BandedReduced) = :banded

# The factor keeps `R`'s lower bandwidth, so one triangle of an `n×n` matrix of bandwidth
# `bw` is `n(bw+1)` entries less the `bw(bw+1)/2` that run off the top-left corner.
function PureOSQP.backend_info(ls::BandedReduced)
    dim, bw = size(ls.R, 1), ls.bw
    return PureOSQP.BackendInfo(
        PureOSQP.backend_name(ls), true, :reduced, dim, dim * (bw + 1) - bw * (bw + 1) ÷ 2
    )
end

function PureOSQP.factorize!(ls::BandedReduced{T}, ws)::Bool where {T}
    n, m, b = ws.n, ws.m, ls.bw
    P, A, D, E = ws.P, ws.A, ws.D, ws.E
    c, rho, sigma = ws.c, ws.rho_vec, ws.settings.sigma
    R = ls.R
    fill!(R.data, zero(T))
    # `c D P D + σI`, over the entries the band holds.
    for j in 1:n, i in max(1, j - b):j
        v = c * D[i] * P[i, j] * D[j]
        i == j && (v += sigma)
        iszero(v) || (R[i, j] = v; R[j, i] = v)
    end
    # `Ãᵀ diag(ρ) Ã`, row by row: row `k` of `A` reaches only its own band, so each row
    # contributes to a fixed number of entries however large `n` is.
    ba = banded_bandwidth(A)
    for k in 1:m
        w = rho[k] * E[k] * E[k]
        lo, hi = max(1, k - ba), min(n, k + ba)
        for j in lo:hi
            akj = A[k, j] * D[j]
            iszero(akj) && continue
            for i in lo:j
                aki = A[k, i] * D[i]
                iszero(aki) && continue
                v = R[i, j] + w * aki * akj
                R[i, j] = v
                i == j || (R[j, i] = v)
            end
        end
    end
    ls.fact = cholesky!(Symmetric(R); check = false)
    return issuccess(ls.fact)
end

function PureOSQP.solve_system!(ls::BandedReduced, ws, rhs_x, rhs_z)::Nothing
    reduced_rhs!(ws, rhs_x, rhs_z)
    copyto!(ws.xtilde, ws.work_n)
    ldiv!(ls.fact, ws.xtilde)
    ws.m > 0 && mul_A!(ws.ztilde, ws, ws.xtilde)
    return nothing
end

# A banded matrix is factored as one, rather than densified into an `n×n` Cholesky.
function PureOSQP.is_convex(::Type{T}, P::BandedMatrix, sigma) where {T}
    isempty(P) && return true
    b = banded_bandwidth(P)
    S = BandedMatrix{T}(undef, size(P), (b, b))
    fill!(S.data, zero(T))
    for j in axes(P, 2), i in max(1, j - b):min(size(P, 1), j + b)
        S[i, j] = P[i, j]
    end
    for i in axes(P, 1)
        S[i, i] += sigma
    end
    return issuccess(cholesky!(Symmetric(S); check = false))
end

end
