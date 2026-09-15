# Reference preconditioners for `algorithm = :ipm, linsys = :indirect`. They are examples of
# the `update_preconditioner!` interface, not part of the package.
using PureOSQP, LinearAlgebra, SparseArrays, LimitedLDLFactorizations

"""
    LaggedCholesky(P, A; every = 3)

The Cholesky factor of `P + σI + Aᵀ diag(w) A`, formed from dense copies of `P` and `A` and
rebuilt at the starting point, every `every` outer iterations, and whenever `σ` changes.
"""
mutable struct LaggedCholesky{T}
    const P::Matrix{T}
    const A::Matrix{T}
    const WA::Matrix{T}
    const K::Matrix{T}
    const every::Int
    F::Cholesky{T, Matrix{T}}
    sigma::T
    builds::Int
end

function LaggedCholesky(P, A; every = 3)
    T, (m, n) = float(eltype(A)), size(A)
    F = cholesky(Matrix{T}(I, n, n))
    return LaggedCholesky{T}(Matrix{T}(P), Matrix{T}(A), zeros(T, m, n), zeros(T, n, n), every, F, T(NaN), 0)
end

function PureOSQP.update_preconditioner!(M::LaggedCholesky, prob, wt, k::Int)
    (k < 0 || iszero(k % M.every) || wt.sigma != M.sigma) || return M
    M.WA .= wt.w .* M.A
    mul!(M.K, M.A', M.WA)
    M.K .+= M.P
    M.K[diagind(M.K)] .+= wt.sigma
    M.F = cholesky!(Hermitian(M.K, :U))
    M.sigma = wt.sigma
    M.builds += 1
    return M
end

LinearAlgebra.ldiv!(y::AbstractVector, M::LaggedCholesky, x::AbstractVector) = ldiv!(y, M.F, x)

# Shifts tried in turn: `lldl` raises its shift only when a pivot vanishes, and on reduced
# matrices it returns negative pivots at no shift.
const LLDL_SHIFTS = (0.0, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 1.0e1, 1.0e2)

"""
    IncompleteLDL(P, A; memory = 10)

A limited-memory incomplete `LDLᵀ` of the sparse reduced matrix, rebuilt every outer iteration
with the smallest diagonal shift that leaves every pivot positive.
"""
mutable struct IncompleteLDL{T, F}
    const P::SparseMatrixCSC{T, Int}
    const A::SparseMatrixCSC{T, Int}
    const memory::Int
    F::F
    builds::Int
end

function IncompleteLDL(P, A; memory = 10)
    n = size(A, 2)
    F = lldl(sparse(1.0I, n, n); memory)
    return IncompleteLDL(sparse(P), sparse(A), memory, F, 0)
end

function PureOSQP.update_preconditioner!(M::IncompleteLDL, prob, wt, k::Int)
    K = tril(M.P + M.A' * Diagonal(wt.w) * M.A + wt.sigma * I)
    for α in LLDL_SHIFTS
        F = lldl(K; memory = M.memory, α)
        if all(>(0), F.D)
            M.F = F
            M.builds += 1
            return M
        end
    end
    return error("no positive-definite lldl factor up to a shift of $(last(LLDL_SHIFTS))")
end

LinearAlgebra.ldiv!(y::AbstractVector, M::IncompleteLDL, x::AbstractVector) = ldiv!(y, M.F, x)
