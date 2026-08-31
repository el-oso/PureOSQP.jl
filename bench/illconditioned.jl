# How the backends behave on a badly conditioned problem, at a size and conditioning taken
# from a real case: `n = 300`, `κ(P) = κ(A) = 1e12`.
#
# Read the statuses before the timings. ADMM does not converge on these problems within
# `max_iter`, so what is compared is where four backends are after the same number of
# iterations, not four answers. That makes the objective column a **consistency** check
# between backends and not evidence that any of them is near the optimum.
#
# The comparison that does survive non-convergence: the three direct backends track each
# other, and the matrix-free one does not track them at all. Conjugate gradients on a reduced
# matrix whose conditioning is `κ(A)²` is not solving the same problem to a worse tolerance —
# its iterates are somewhere else entirely.
using PureOSQP, Krylov, LDLFactorizations, LinearAlgebra, Printf, JSON, Random, SparseArrays
using Chairmarks, Statistics

BLAS.set_num_threads(1)

const N = 300
const KAPPA = 1.0e12
const OPTS = (eps_abs = 1.0e-8, eps_rel = 1.0e-8)

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
function problems()
    rng = MersenneTwister(78)
    dense = (spd(rng, N, KAPPA), illconditioned(rng, N, KAPPA))
    K = 6
    nb = N ÷ K
    blocked = (
        PureOSQP.BlockDiagonal([spd(rng, nb, KAPPA) for _ in 1:K]),
        PureOSQP.BlockDiagonal([illconditioned(rng, nb, KAPPA) for _ in 1:K]),
    )
    return [("dense", dense..., (:auto, :kkt, :indirect)), ("blocks of $nb", blocked..., (:auto, :dense, :kkt, :indirect))]
end

println("\nBackends on a badly conditioned problem: n = $N, κ = $(KAPPA).\n")
@printf(
    "%-13s %-10s %-12s %-20s %7s %9s %9s\n",
    "shape", "linsys", "backend", "status", "iter", "objective", "words"
)
println("-"^92)

rows = NamedTuple[]
for (shape, P, A, settings) in problems()
    n = size(P, 2)
    q = randn(MersenneTwister(5), n)
    b = A * randn(MersenneTwister(6), n)
    l, u = b .- rand(MersenneTwister(7), n), b .+ rand(MersenneTwister(8), n)
    @printf("cond(P) = %.3g   cond(A) = %.3g\n", cond(Matrix(P)), cond(Matrix(A)))
    for ls in settings
        ws = PureOSQP.setup(P, q, A, l, u; linsys = ls)
        info = PureOSQP.backend_info(ws.linsys)
        # Warmed before timing: each backend is a fresh specialization, and a first call
        # measures the compiler rather than the solve.
        r = PureOSQP.solve(P, q, A, l, u; linsys = ls, OPTS...)
        t = median(s.time for s in @be(PureOSQP.solve(P, q, A, l, u; linsys = ls, OPTS...), seconds = 4).samples)
        push!(
            rows, (;
                shape, linsys = string(ls), backend = string(info.name),
                status = string(r.status), iter = r.iter, obj_val = r.obj_val,
                words = info.factor_nnz, seconds = t,
            )
        )
        @printf(
            "%-13s %-10s %-12s %-20s %7d %9.4g %9d  %7.1f ms\n",
            shape, ls, info.name, r.status, r.iter, r.obj_val, info.factor_nnz, 1.0e3t
        )
        flush(stdout)
    end
    println()
end

open(joinpath(@__DIR__, "results", "illconditioned.json"), "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "blas_threads" => BLAS.get_num_threads(),
            "n" => N, "kappa" => KAPPA,
            "cases" => [Dict(string(k) => string(v) for (k, v) in pairs(r)) for r in rows],
        ), 2
    )
end
println("saved bench/results/illconditioned.json")
