"""
In-place `LDLᵀ` of the Gram matrix of an active row set, maintained under row
insertion and deletion.

`L` is unit lower triangular and `D` holds its diagonal. Both live in full-size
buffers with `k` marking how much is live, so a solve that adds and drops
thousands of constraints allocates only at setup.

The factored matrix is `Mₐ Mₐᵀ`, where `Mₐ` holds the rows of `M` currently
active, in the order they were added.
"""
# `const` on every field but `k`: only the live size is rebound, the buffers are written
# through. A fully immutable struct would be the wrong trade -- inlined into the workspace it
# loses the nonnull and alignment facts the vectorizer needs, and these loops go scalar.
mutable struct GramLDL{T <: Real}
    const L::Matrix{T}
    const D::Vector{T}
    k::Int
    const w::Vector{T}   # scratch for the rank-one sweep
    const b::Vector{T}   # scratch for the forward solve
end

function GramLDL{T}(kmax::Integer) where {T <: Real}
    return GramLDL{T}(zeros(T, kmax, kmax), zeros(T, kmax), 0, zeros(T, kmax), zeros(T, kmax))
end

Base.size(F::GramLDL) = F.k

"""
Working sets up to this size solve through plain loops rather than LAPACK.

`ldiv!` on a triangular view routes to `trtrs!`, whose call overhead dominates at this scale:
58 ns against 14 ns for a loop at `k = 8`. These solves run twice per iteration, so on a
small problem they are a third of it. LAPACK wins again once the triangle is big enough for
its blocking to pay, which is why this is a threshold and not a replacement.
"""
const SMALL_TRIANGLE = 32

"Solve `L y = b` in place over the leading `length(b)` block."
function forward_L!(F::GramLDL{T}, b::AbstractVector{T}) where {T}
    isempty(b) && return b
    k = length(b)
    if k > SMALL_TRIANGLE
        ldiv!(UnitLowerTriangular(view(F.L, 1:k, 1:k)), b)
        return b
    end
    # Column-oriented: each column of `L` is contiguous, and the update walks the tail of
    # `b`. `k <= F.k <= size(L, 1)` by construction, so the indices are in range.
    L = F.L
    @inbounds for j in 1:k
        bj = b[j]
        @simd for i in (j + 1):k
            b[i] -= L[i, j] * bj
        end
    end
    return b
end

"Solve `Lᵀ y = b` in place over the leading `length(b)` block."
function backward_L!(F::GramLDL{T}, b::AbstractVector{T}) where {T}
    isempty(b) && return b
    k = length(b)
    if k > SMALL_TRIANGLE
        ldiv!(UnitLowerTriangular(view(F.L, 1:k, 1:k))', b)
        return b
    end
    # `Lᵀ` is upper triangular, so this runs backwards, reading down column `i` of `L`. No
    # `@simd`: this is a reduction, and reassociating it moves the answer at the last digit,
    # which is where agreement with the reference solver is measured.
    L = F.L
    @inbounds for i in k:-1:1
        acc = b[i]
        for j in (i + 1):k
            acc -= L[j, i] * b[j]
        end
        b[i] = acc
    end
    return b
end

"""
    solve_gram!(F, rhs) -> rhs

Solve `Mₐ Mₐᵀ x = rhs` in place through the stored factors.
"""
function solve_gram!(F::GramLDL, rhs::AbstractVector)
    forward_L!(F, rhs)
    @inbounds @simd for i in eachindex(rhs)
        rhs[i] /= F.D[i]
    end
    backward_L!(F, rhs)
    return rhs
end

"""
    add_row!(F, g, beta; zero_tol) -> F

Extend the factorization by one active row `a`, given `g = Mₐ a` against the
rows already active and `beta = aᵀa`.

A new diagonal at or below `zero_tol` means the row is linearly dependent on
those already active. It is stored as an exact zero: that is what makes the
dependency detectable, since the caller scans `D` for a zero and takes the
singular branch instead of dividing by it.
"""
function add_row!(F::GramLDL{T}, g::AbstractVector{T}, beta::T, zero_tol::T = sqrt(eps(T))) where {T}
    k = F.k
    if iszero(k)
        F.L[1, 1] = one(T)
        F.D[1] = beta
        F.k = 1
        return F
    end
    b = view(F.b, 1:k)
    # An explicit loop rather than `copyto!`: that checks lengths and throws, and building
    # the exception is an allocation site the hot-path guarantee sees whether or not the
    # branch can be reached. The caller always passes `k` entries.
    @inbounds @simd for i in 1:k
        b[i] = g[i]
    end
    forward_L!(F, b)

    d = beta
    # The `D[i] > zero_tol` test cannot leave this loop: a zero pivot must not be divided by,
    # and which pivots are zero is a property of the data. It predicts well, being false only
    # on the dependent rows, which are rare.
    @inbounds for i in 1:k
        lki = F.D[i] > zero_tol ? b[i] / F.D[i] : zero(T)
        F.L[k + 1, i] = lki
        d -= F.D[i] * lki^2
    end
    F.L[k + 1, k + 1] = one(T)
    F.D[k + 1] = d > zero_tol ? d : zero(T)
    F.k = k + 1
    return F
end

"""
    rank_one!(F, off, l, delta, n)

`L D Lᵀ ← L D Lᵀ + delta·l lᵀ` on the `n×n` block whose first index is `off+1`.
Algorithm C1 of Gill, Golub, Murray and Saunders, *Methods for modifying matrix
factorizations*, Math. Comp. 28(126):505-535, 1974. `l` is overwritten.
"""
function rank_one!(F::GramLDL{T}, off::Int, l::AbstractVector{T}, delta::T, n::Int) where {T}
    a = delta
    # `off + n <= F.k <= size(L, 1)`, so every index is in range. The inner loop is not
    # `@simd`: `l[r]` is written then read in the same iteration, a dependence that
    # reassociating would break.
    @inbounds for j in 1:n
        p = l[j]
        dold = F.D[off + j]
        dnew = dold + a * p^2
        F.D[off + j] = dnew > 0 ? dnew : zero(T)
        # A pivot reaching zero ends the sweep's contribution rather than dividing by it.
        if !(dnew > 0)
            a = zero(T)
            continue
        end
        b = p * a / dnew
        a = dold * a / dnew
        for r in (j + 1):n
            lrj = F.L[off + r, off + j]
            l[r] -= p * lrj
            F.L[off + r, off + j] = lrj + b * l[r]
        end
    end
    return F
end

"""
    remove_row!(F, i) -> F

Drop the `i`-th active row. The trailing block is repaired with a rank-one
update first, then everything below and right of `i` shifts up and left by one.
The repair is what a deletion costs.
"""
function remove_row!(F::GramLDL{T}, i::Integer) where {T}
    k = F.k
    ntail = k - i
    if ntail > 0
        l = view(F.w, 1:ntail)
        for r in 1:ntail
            l[r] = F.L[i + r, i]
        end
        rank_one!(F, i, l, F.D[i], ntail)

        # Shift the repaired trailing block, and the columns left of i, up-left.
        for c in 1:(i - 1), r in 1:ntail
            F.L[i + r - 1, c] = F.L[i + r, c]
        end
        for j in 1:ntail, r in (j + 1):ntail
            F.L[i + r - 1, i + j - 1] = F.L[i + r, i + j]
        end
        for r in 1:ntail
            F.D[i + r - 1] = F.D[i + r]
            F.L[i + r - 1, i + r - 1] = one(T)
        end
    end
    F.k = k - 1
    return F
end

"""
    singular_direction!(p, F, i) -> p

A direction in the space of active multipliers along which the Gram matrix is
singular, for the dependency marked by a zero at `D[i]`. It satisfies
`Mₐᵀ p = 0`, so moving along it changes the multipliers and not the primal point.
"""
function singular_direction!(p::AbstractVector{T}, F::GramLDL{T}, i::Integer) where {T}
    fill!(p, zero(T))
    if i > 1
        for j in 1:(i - 1)
            p[j] = -F.L[i, j]
        end
        backward_L!(F, view(p, 1:(i - 1)))
    end
    p[i] = one(T)
    return p
end
