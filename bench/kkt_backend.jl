# Reduced-Cholesky vs full-KKT Bunch-Kaufman: the measurement behind PureOSQP's choice of
# linear-system backend. Reproduces both tables in docs/src/algorithm.md.
using LinearAlgebra, BenchmarkTools, Printf, Random, JSON

BLAS.set_num_threads(1)

"Time one factorization of each backend for a problem of size (n, m)."
function factorization_cost(n, m; sigma = 1.0e-6, rho = 0.1, seed = 0)
    Random.seed!(seed)
    X = randn(n, n)
    P = Matrix(Symmetric(X'X / n))
    A = randn(m, n)
    K = Matrix{Float64}(undef, n + m, n + m)
    K[1:n, 1:n] .= P
    for i in 1:n
        K[i, i] += sigma
    end
    K[(n + 1):end, 1:n] .= A
    K[1:n, (n + 1):end] .= A'
    K[(n + 1):end, (n + 1):end] .= 0
    for i in 1:m
        K[n + i, n + i] = -1 / rho
    end
    t_bk = @belapsed bunchkaufman!(Symmetric(Kc, :L)) setup = (Kc = copy($K)) evals = 1
    R = Matrix{Float64}(undef, n, n)
    t_chol = @belapsed begin
        mul!($R, $A', $A, $rho, 0.0)
        $R .+= $P
        for i in 1:($n)
            $R[i, i] += $sigma
        end
        cholesky!(Symmetric($R))
    end evals = 1
    return (; n, m, t_bk, t_chol, ratio = t_bk / t_chol)
end

"Modified Ruiz equilibration of [P Aᵀ; A 0], as in PureOSQP."
function ruiz(P, A, q; iters = 10, minsc = 1.0e-4, maxsc = 1.0e4)
    n, m = size(P, 1), size(A, 1)
    P, A, q = copy(P), copy(A), copy(q)
    D, E, c = ones(n), ones(m), 1.0
    lim(v) = clamp(v < minsc ? 1.0 : v, minsc, maxsc)
    for _ in 1:iters
        d = [lim(max(maximum(abs, view(P, :, j)), maximum(abs, view(A, :, j)))) for j in 1:n]
        e = [lim(maximum(abs, view(A, i, :))) for i in 1:m]
        d .= inv.(sqrt.(d))
        e .= inv.(sqrt.(e))
        P .= Diagonal(d) * P * Diagonal(d)
        A .= Diagonal(e) * A * Diagonal(d)
        q .*= d
        D .*= d
        E .*= e
        ct = max(sum(j -> maximum(abs, view(P, :, j)), 1:n) / n, lim(maximum(abs, q)))
        ct = 1 / lim(ct)
        P .*= ct
        q .*= ct
        c *= ct
    end
    return (P, A, q, D, E, c)
end

"""
Relative error of the inner solve, both backends, before and after equilibration, against
an extended-precision reference. `P = 0` is the worst case for the conditioning of `AᵀA`.
"""
function inner_solve_accuracy(n, m, condA; sigma = 1.0e-6, rho = 0.1, seed = 0)
    Random.seed!(seed)
    U = Matrix(qr(randn(m, n)).Q)[:, 1:n]
    V = Matrix(qr(randn(n, n)).Q)
    A0 = U * Diagonal(exp10.(range(0, log10(condA); length = n))) * V'
    _, A1, _, _, _, _ = ruiz(zeros(n, n), A0, zeros(n))
    out = map((A0, A1)) do A
        b1, b2 = randn(n), randn(m)
        K = [sigma * I(n)  A'; A  -(1 / rho) * I(m)]
        ref = Float64.(big.(K) \ big.([b1; b2]))[1:n]
        bk = (bunchkaufman!(Symmetric(copy(K), :L)) \ [b1; b2])[1:n]
        F = cholesky!(Symmetric(sigma * I(n) + rho * (A'A)); check = false)
        ch = issuccess(F) ? F \ (b1 + A' * (rho * b2)) : fill(NaN, n)
        (
            cond = cond(A),
            err_bk = norm(bk .- ref, Inf) / norm(ref, Inf),
            err_chol = norm(ch .- ref, Inf) / norm(ref, Inf),
        )
    end
    return (; condA, raw = out[1], equilibrated = out[2])
end

const SIZES = [(50, 50), (50, 500), (200, 200), (200, 2000), (500, 500), (500, 100)]
const CONDS = [1.0e2, 1.0e4, 1.0e6, 1.0e8, 1.0e10, 1.0e12, 1.0e14, 1.0e16]

println("--- factorization cost (s), single-threaded ---")
@printf("%6s %6s %13s %13s %8s\n", "n", "m", "full-KKT BK", "reduced Chol", "ratio")
costs = [factorization_cost(n, m) for (n, m) in SIZES]
for r in costs
    @printf("%6d %6d %13.3e %13.3e %8.2f\n", r.n, r.m, r.t_bk, r.t_chol, r.ratio)
    flush(stdout)
end

println("\n--- inner-solve relative error, n = 60, m = 200, P = 0 ---")
@printf("%10s | %11s %11s | %11s %11s\n", "cond(A)", "Chol raw", "BK raw", "Chol ruiz", "BK ruiz")
accs = [inner_solve_accuracy(60, 200, c) for c in CONDS]
for r in accs
    @printf(
        "%10.0e | %11.2e %11.2e | %11.2e %11.2e\n", r.condA,
        r.raw.err_chol, r.raw.err_bk, r.equilibrated.err_chol, r.equilibrated.err_bk
    )
    flush(stdout)
end

# JSON has no NaN, and a failed Cholesky is recorded as one.
nt2d(nt) = Dict(string(k) => (v isa AbstractFloat && isnan(v) ? nothing : v) for (k, v) in pairs(nt))

open(joinpath(@__DIR__, "results", "kkt_backend.json"), "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "costs" => map(nt2d, costs),
            "accuracy" => [
                Dict(
                        "condA" => r.condA, "raw" => nt2d(r.raw),
                        "equilibrated" => nt2d(r.equilibrated)
                    ) for r in accs
            ]
        ), 2
    )
end
println("\nsaved bench/results/kkt_backend.json")

"""
Ill-conditioning that equilibration cannot remove: `A` has unit-norm columns and rows that
are nearly parallel, so there is no diagonal scaling that improves it. This is the case
that decides whether the Cholesky fallback is reachable in practice.
"""
function near_parallel_accuracy(n, m, eps_par; sigma = 1.0e-6, rho = 0.1, seed = 0)
    Random.seed!(seed)
    base = randn(n)
    base ./= norm(base)
    A0 = Matrix{Float64}(undef, m, n)
    for i in 1:m
        r = base .+ eps_par .* randn(n)
        A0[i, :] .= r ./ norm(r)
    end
    _, A1, _, _, _, _ = ruiz(zeros(n, n), A0, zeros(n))
    out = map((A0, A1)) do A
        b1, b2 = randn(n), randn(m)
        K = [sigma * I(n)  A'; A  -(1 / rho) * I(m)]
        ref = Float64.(big.(K) \ big.([b1; b2]))[1:n]
        F = cholesky!(Symmetric(sigma * I(n) + rho * (A'A)); check = false)
        ch = issuccess(F) ? F \ (b1 + A' * (rho * b2)) : fill(NaN, n)
        bk = (bunchkaufman!(Symmetric(copy(K), :L)) \ [b1; b2])[1:n]
        (
            cond = cond(A), chol_ok = issuccess(F),
            err_chol = norm(ch .- ref, Inf) / norm(ref, Inf),
            err_bk = norm(bk .- ref, Inf) / norm(ref, Inf),
        )
    end
    return (; eps_par, raw = out[1], equilibrated = out[2])
end

println("\n--- near-parallel rows: conditioning equilibration cannot remove ---")
@printf(
    "%9s %10s | %11s %7s | %11s %7s\n", "row spread", "cond(A)",
    "Chol ruiz", "ok?", "BK ruiz", ""
)
for e in (1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8, 1.0e-10)
    r = near_parallel_accuracy(40, 120, e)
    @printf(
        "%9.0e %10.2e | %11.2e %7s | %11.2e\n", r.eps_par, r.equilibrated.cond,
        r.equilibrated.err_chol, r.equilibrated.chol_ok, r.equilibrated.err_bk
    )
    flush(stdout)
end
