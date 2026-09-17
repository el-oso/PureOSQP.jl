# Matrices of a prescribed condition number, shared by the conditioning sweeps.
using LinearAlgebra, Random

"A symmetric positive definite matrix of side `k` with condition number `kappa`."
function spd(rng, k, kappa)
    Q = qr(randn(rng, k, k)).Q
    d = exp10.(range(0, log10(kappa); length = k))
    return Matrix(Symmetric(Matrix(Q * Diagonal(d) * Q')))
end

"A `k×k` matrix with condition number `kappa`."
function illconditioned(rng, k, kappa)
    U = qr(randn(rng, k, k)).Q
    V = qr(randn(rng, k, k)).Q
    s = exp10.(range(0, -log10(kappa); length = k))
    return Matrix(U * Diagonal(s) * V')
end
