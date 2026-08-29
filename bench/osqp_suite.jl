# The OSQP benchmark suite's problem classes, against libosqp.
#
# Every other sparse benchmark here generates uniformly random sparsity, which is the worst
# case for a sparse factorization: a random graph has no separator, so the Cholesky factor
# fills in whatever the matrix looked like. That is why `SparseCholmod`'s fill gate refuses
# those problems, and it is also why they cannot say whether the gate is right — a corpus
# that never has exploitable structure cannot test a decision about exploitable structure.
#
# These seven classes are the ones OSQP's own benchmark suite uses, ported from its problem
# definitions (github.com/osqp/osqp_benchmarks, `problem_classes/`). They carry the block
# and band structure real problems have, so they are what decides which backend gets chosen
# in practice.
using PureOSQP, OSQP, LinearAlgebra, SparseArrays, MatrixEquations, BenchmarkTools
using Random, Printf, JSON

BLAS.set_num_threads(1)

const OPTS = (eps_abs = 1.0e-5, eps_rel = 1.0e-5, max_iter = 20_000)

# libosqp 0.6.2 has no duality-gap test and would reject the setting; with it on the two
# solvers stop on different criteria and the iteration counts would compare the criteria.
const PURE_ONLY = (check_dualgap = false,)

solve_pure(P, q, A, l, u) = PureOSQP.solve(P, q, A, l, u; OPTS..., PURE_ONLY...)

function solve_osqp(P, q, A, l, u)
    model = OSQP.Model()
    OSQP.setup!(
        model; P = sparse(Symmetric(P)), q = collect(q), A = sparse(A),
        l = collect(l), u = collect(u), verbose = false,
        adaptive_rho_interval = 50, check_termination = 25, OPTS...
    )
    return OSQP.solve!(model)
end

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

"""
    compare(name, prob) -> NamedTuple

Time both solvers on one problem and record which backend PureOSQP chose.

The two objectives must agree; a row whose answers differ is not a timing comparison and is
refused rather than reported.
"""
function compare(name, prob)
    P, q, A, l, u = prob
    sp = solve_pure(P, q, A, l, u)
    so = solve_osqp(P, q, A, l, u)
    ok = sp.status == PureOSQP.SOLVED && so.info.status == :Solved
    ok || return (; name, n = size(A, 2), m = size(A, 1), skipped = "$(sp.status)/$(so.info.status)")
    gap = abs(sp.obj_val - so.info.obj_val) / max(1, abs(so.info.obj_val))
    gap < 1.0e-4 || error("$name: objectives disagree by $gap")
    ws = PureOSQP.setup(P, q, A, l, u; OPTS..., PURE_ONLY...)
    tp = minimum(@benchmark(solve_pure($P, $q, $A, $l, $u), samples = 3, evals = 1, seconds = 90)).time / 1.0e9
    to = minimum(@benchmark(solve_osqp($P, $q, $A, $l, $u), samples = 3, evals = 1, seconds = 90)).time / 1.0e9
    return (;
        name, n = size(A, 2), m = size(A, 1), nnz_A = nnz(sparse(A)), nnz_P = nnz(sparse(P)),
        backend = PureOSQP.backend_name(ws.linsys), t_pure = tp, t_osqp = to, ratio = to / tp,
        iter_pure = sp.iter, iter_osqp = so.info.iter, obj_gap = gap,
    )
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

println("\nThe OSQP benchmark suite's problem classes. Both solvers hold SparseMatrixCSC.\n")
@printf(
    "%-12s %6s %6s %8s %-16s | %11s %11s %8s | %6s %6s\n",
    "class", "n", "m", "nnz(A)", "PureOSQP backend", "PureOSQP", "OSQP", "vs OSQP", "it pu", "it os"
)
println("-"^108)
rows = NamedTuple[]
for (name, build) in CASES
    r = compare(name, build())
    push!(rows, r)
    if haskey(r, :skipped)
        @printf("%-12s %6d %6d %8s %-16s | %s\n", r.name, r.n, r.m, "-", "-", "skipped: " * r.skipped)
    else
        @printf(
            "%-12s %6d %6d %8d %-16s | %8.2f ms %8.2f ms %7.2fx | %6d %6d\n",
            r.name, r.n, r.m, r.nnz_A, r.backend,
            1.0e3r.t_pure, 1.0e3r.t_osqp, r.ratio, r.iter_pure, r.iter_osqp
        )
    end
    flush(stdout)
end

open(joinpath(@__DIR__, "results", "osqp_suite.json"), "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "blas_threads" => BLAS.get_num_threads(),
            "cases" => [Dict(string(k) => string(v) for (k, v) in pairs(r)) for r in rows],
        ), 2
    )
end
println("\nsaved bench/results/osqp_suite.json")
