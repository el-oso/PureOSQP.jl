# The OSQP benchmark suite's problem classes, ported from its own problem definitions
# (github.com/osqp/osqp_benchmarks, `problem_classes/`).
#
# Every other sparse benchmark here generates uniformly random sparsity, which is the worst
# case for a sparse factorization: a random graph has no separator, so the Cholesky factor
# fills in whatever the matrix looked like. These seven carry the block and band structure
# real problems have, so they are what decides which backend gets chosen in practice.
#
# Definitions only, so that a benchmark measuring one part of the solver can `include` them
# without also running the head-to-head table.
using LinearAlgebra, SparseArrays, MatrixEquations, Random

sprandn_like(rng, m, n, d) = sprandn(rng, m, n, d)
zeromat(m, n) = spzeros(Float64, m, n)

"Random QP: `m = 10n` one-sided inequalities on a dense-ish sparse `P`."
function random_qp(n; seed = 1)
    rng = MersenneTwister(seed)
    m = 10n
    Pr = sprandn_like(rng, n, n, 0.15)
    P = sparse(Symmetric(Pr * Pr')) + 1.0e-2 * I
    q = randn(rng, n)
    A = sprandn_like(rng, m, n, 0.15)
    v = randn(rng, n)
    u = A * v .+ rand(rng, m)
    l = fill(-Inf, m)
    return (P, q, A, l, u)
end

"Equality-constrained QP: `m = n/2` rows with `l == u`."
function eq_qp(n; seed = 1)
    rng = MersenneTwister(seed)
    m = n ÷ 2
    Pr = sprandn_like(rng, n, n, 0.15)
    P = sparse(Symmetric(Pr * Pr')) + 1.0e-2 * I
    q = randn(rng, n)
    A = sprandn_like(rng, m, n, 0.15)
    b = A * randn(rng, n)
    return (P, q, A, copy(b), b)
end

"""
Portfolio: `k` factors over `n = 100k` assets, in the factor-model form that makes the
objective diagonal and pushes the covariance into constraints.
"""
function portfolio(k; seed = 1)
    rng = MersenneTwister(seed)
    n = 100k
    F = sprandn_like(rng, n, k, 0.5)
    D = spdiagm(0 => rand(rng, n) .* sqrt(k))
    mu = randn(rng, n)
    gamma = 1.0
    P = blockdiag(2 * D, sparse(2.0I, k, k))
    q = vcat(-mu ./ gamma, zeros(k))
    A = vcat(
        hcat(sparse(ones(1, n)), zeromat(1, k)),
        hcat(sparse(F'), sparse(-1.0I, k, k)),
        hcat(sparse(1.0I, n, n), zeromat(n, k)),
    )
    l = vcat(1.0, zeros(k), zeros(n))
    u = vcat(1.0, zeros(k), ones(n))
    return (P, q, A, l, u)
end

"Lasso: `m = 100n` data points, in the split form with a residual and an epigraph variable."
function lasso(n; seed = 1)
    rng = MersenneTwister(seed)
    m = 100n
    Ad = sprandn_like(rng, m, n, 0.15)
    x_true = (rand(rng, n) .> 0.5) .* randn(rng, n) ./ sqrt(n)
    bd = Ad * x_true .+ randn(rng, m)
    lambda = norm(Ad' * bd, Inf) / 5
    In = sparse(1.0I, n, n)
    P = blockdiag(zeromat(n, n), sparse(2.0I, m, m), zeromat(n, n))
    q = vcat(zeros(n + m), fill(lambda, n))
    A = vcat(
        hcat(Ad, sparse(-1.0I, m, m), zeromat(m, n)),
        hcat(In, zeromat(n, m), -In),
        hcat(In, zeromat(n, m), In),
    )
    l = vcat(bd, fill(-Inf, n), zeros(n))
    u = vcat(bd, zeros(n), fill(Inf, n))
    return (P, q, A, l, u)
end

"Support vector machine: hinge loss on `m = 100n` points, half of each label."
function svm(n; seed = 1)
    rng = MersenneTwister(seed)
    m = 100n
    N = m ÷ 2
    b = vcat(ones(N), -ones(N))
    upp = sprandn_like(rng, N, n, 0.15)
    low = sprandn_like(rng, N, n, 0.15)
    Asvm = vcat(
        upp ./ sqrt(n) .+ (upp .!= 0) ./ n,
        low ./ sqrt(n) .- (low .!= 0) ./ n,
    )
    P = blockdiag(sparse(1.0I, n, n), zeromat(m, m))
    q = vcat(zeros(n), fill(0.5, m))
    A = vcat(
        hcat(spdiagm(0 => b) * Asvm, sparse(-1.0I, m, m)),
        hcat(zeromat(m, n), sparse(1.0I, m, m)),
    )
    l = vcat(fill(-Inf, m), zeros(m))
    u = vcat(fill(-1.0, m), fill(Inf, m))
    return (P, q, A, l, u)
end

"Huber fitting: `m = 100n` points with 5% outliers, in the two-slack epigraph form."
function huber(n; seed = 1)
    rng = MersenneTwister(seed)
    m = 100n
    Ad = sprandn_like(rng, m, n, 0.15)
    x_true = randn(rng, n) ./ sqrt(n)
    inlier = rand(rng, m) .< 0.95
    bd = Ad * x_true .+ (0.5 .* randn(rng, m)) .* inlier .+ (10 .* rand(rng, m)) .* .!inlier
    Im = sparse(1.0I, m, m)
    P = blockdiag(zeromat(n, n), Im, zeromat(2m, 2m))
    q = vcat(zeros(n + m), ones(2m))
    A = vcat(
        hcat(Ad, -Im, -Im, Im),
        hcat(zeromat(m, n), zeromat(m, m), Im, zeromat(m, m)),
        hcat(zeromat(m, n), zeromat(m, m), zeromat(m, m), Im),
    )
    l = vcat(bd, zeros(2m))
    u = vcat(bd, fill(Inf, 2m))
    return (P, q, A, l, u)
end

"""
Control: an `nx`-state, `nu`-input MPC problem over a horizon of 10, with the terminal cost
from the discrete algebraic Riccati equation. This is the class with a band, and the one a
sparse factorization exists for.
"""
function control(nx; seed = 1, horizon = 10)
    rng = MersenneTwister(seed)
    nu = nx ÷ 2
    Ad = Matrix(1.0I, nx, nx) + 0.1 * randn(rng, nx, nx)
    # Pull any unstable eigenvalue back inside the unit circle, so the horizon is meaningful.
    vals, V = eigen(Ad)
    vals = [abs(λ) < 1 - 1.0e-2 ? λ : λ / (abs(λ) + 1.0e-2) for λ in vals]
    Ad = real.(V * Diagonal(vals) * inv(V))
    Bd = randn(rng, nx, nu)
    R = 0.1 * Matrix(1.0I, nu, nu)
    Q = Diagonal(rand(rng, nx) .* (rand(rng, nx) .< 0.7))
    X, = ared(Ad, Bd, R, Matrix(Q))
    QN = X * X'
    umin = -rand(rng, nu)
    xmin = -1.0 .- rand(rng, nx)
    x0 = 0.5 .* xmin .+ rand(rng, nx) .* (0.5 .* (-xmin) .- 0.5 .* xmin)

    T = horizon
    P = 2 * blockdiag(
        kron(sparse(1.0I, T, T), sparse(Q)), sparse(QN),
        kron(sparse(1.0I, T, T), sparse(R)),
    )
    q = zeros((T + 1) * nx + T * nu)
    Ax = kron(sparse(1.0I, T + 1, T + 1), sparse(-1.0I, nx, nx)) +
        kron(spdiagm(-1 => ones(T)), sparse(Ad))
    Au = kron(vcat(zeromat(1, T), sparse(1.0I, T, T)), sparse(Bd))
    Aeq = hcat(Ax, Au)
    beq = vcat(-x0, zeros(T * nx))
    Abound = sparse(1.0I, size(Aeq, 2), size(Aeq, 2))
    A = vcat(Aeq, Abound)
    lb = vcat(repeat(xmin, T + 1), repeat(umin, T))
    ub = -lb
    return (P, q, A, vcat(beq, lb), vcat(beq, ub))
end

const CASES = [
    ("Random QP", () -> random_qp(50)),
    ("Eq QP", () -> eq_qp(200)),
    ("Portfolio", () -> portfolio(5)),
    ("Lasso", () -> lasso(8)),
    ("SVM", () -> svm(8)),
    ("Huber", () -> huber(6)),
    ("Control", () -> control(20)),
]
