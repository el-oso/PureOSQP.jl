"""
    SystemWeights{T,V}

The diagonal weights and the primal regularization a [`LinearSystem`](@ref) is built from:

    reduced   P̃ + σI + Ãᵀ diag(w) Ã
    KKT       [P̃ + σI   Ãᵀ  ;  Ã   −diag(w_inv)]

Invariants, maintained by the owner: `w[i] > 0`, `w_inv[i] == inv(w[i])` as the owner
computed it, `sigma > 0`, `length(w) == m`. The vectors are read in place, so a change to
their contents reaches the next [`refactor_weights!`](@ref) without a new object; a change to
`sigma` needs a new object.

ADMM holds `w = ρ`, `w_inv = ρ⁻¹` and `sigma = σ`.
"""
struct SystemWeights{T <: Real, V <: AbstractVector{T}}
    w::V
    w_inv::V
    sigma::T
end
