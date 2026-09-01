# How the backends behave as conditioning worsens, at a size taken from a real case:
# `n = 300`, with `κ(P) = κ(A)` swept up to the `1e12` that case carries.
#
# Read the statuses before the timings, because the sweep spans two regimes. Below the wall
# every backend converges and the columns compare four answers: iteration counts and objectives
# are directly comparable and the timings are times to a solution. Above it none of them
# converges within `max_iter`, so the columns compare where four backends are after the same
# number of iterations, the objective becomes a consistency check between backends rather than
# evidence any is near the optimum, and the time is a per-iteration cost.
#
# The comparison that survives into the second regime: the direct backends track each other and
# the matrix-free one does not track them at all. Conjugate gradients on a reduced matrix whose
# conditioning is `κ(A)²` is not solving the same problem to a worse tolerance — its iterates
# are somewhere else entirely.
using PureOSQP, Krylov, LDLFactorizations, LinearAlgebra, Printf, JSON, Random, SparseArrays
using Chairmarks, Statistics

BLAS.set_num_threads(1)

const N = 300
# Swept rather than fixed, because the interesting quantity is where ADMM stops converging.
# `1e12` is the real case; the values below it are what make the comparison at `1e12` readable,
# since a backend column only means something once there is a regime where the runs agree.
const KAPPAS = (1.0e4, 1.0e8, 1.0e10, 1.0e12)
# `max_iter` well above the default: the converged regime runs to five figures of iterations
# near the wall, and cutting it there would report a tolerance choice as a failure to converge.
const OPTS = (eps_abs = 1.0e-6, eps_rel = 1.0e-6, max_iter = 20_000)

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

"""
    problems() -> Vector

The same size and conditioning in two shapes: one dense pair, and one split into `K` blocks so
the block backend is reachable. The block problem's `κ` is its blocks', which is the whole
matrix's since the blocks are independent.
"""
function problems(kappa)
    rng = MersenneTwister(78)
    dense = (spd(rng, N, kappa), illconditioned(rng, N, kappa))
    K = 6
    nb = N ÷ K
    blocked = (
        PureOSQP.BlockDiagonal([spd(rng, nb, kappa) for _ in 1:K]),
        PureOSQP.BlockDiagonal([illconditioned(rng, nb, kappa) for _ in 1:K]),
    )
    return [("dense", dense..., (:auto, :kkt, :indirect)), ("blocks of $nb", blocked..., (:auto, :dense, :kkt, :indirect))]
end

println("\nBackends against conditioning: n = $N, eps = $(OPTS.eps_abs), max_iter = $(OPTS.max_iter).\n")

rows = NamedTuple[]
for kappa in KAPPAS
    @printf("κ = %.0e\n", kappa)
    @printf(
        "%-13s %-10s %-12s %-20s %7s %11s %9s %10s\n",
        "shape", "linsys", "backend", "status", "iter", "objective", "words", "time"
    )
    println("-"^100)
    for (shape, P, A, settings) in problems(kappa)
        n = size(P, 2)
        q = randn(MersenneTwister(5), n)
        b = A * randn(MersenneTwister(6), n)
        l, u = b .- rand(MersenneTwister(7), n), b .+ rand(MersenneTwister(8), n)
        for ls in settings
            ws = PureOSQP.setup(P, q, A, l, u; linsys = ls)
            info = PureOSQP.backend_info(ws.linsys)
            # Warmed before timing: each backend is a fresh specialization, and a first call
            # measures the compiler rather than the solve.
            r = PureOSQP.solve(P, q, A, l, u; linsys = ls, OPTS...)
            t = median(s.time for s in @be(PureOSQP.solve(P, q, A, l, u; linsys = ls, OPTS...), seconds = 2).samples)
            push!(
                rows, (;
                    kappa, shape, linsys = string(ls), backend = string(info.name),
                    status = string(r.status), iter = r.iter, obj_val = r.obj_val,
                    words = info.factor_nnz, seconds = t,
                )
            )
            @printf(
                "%-13s %-10s %-12s %-20s %7d %11.5g %9d  %7.1f ms\n",
                shape, ls, info.name, r.status, r.iter, r.obj_val, info.factor_nnz, 1.0e3t
            )
            flush(stdout)
        end
    end
    println()
end

open(joinpath(@__DIR__, "results", "illconditioned.json"), "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "blas_threads" => BLAS.get_num_threads(),
            "n" => N, "kappas" => collect(KAPPAS),
            "eps" => OPTS.eps_abs, "max_iter" => OPTS.max_iter,
            "cases" => [Dict(string(k) => string(v) for (k, v) in pairs(r)) for r in rows],
        ), 2
    )
end
println("saved bench/results/illconditioned.json")
