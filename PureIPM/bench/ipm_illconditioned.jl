# How the interior-point method behaves as conditioning worsens, at the size and sweep
# `bench/illconditioned.jl` puts the operator-splitting method through: `n = 300`, with
# `κ(P) = κ(A)` swept to `1e12`.
#
# Two quantities matter here that have no operator-splitting counterpart. `reg_bumps` counts
# the times a Newton system failed to factor and the regularization was raised to retry, so it
# says how much of the run was spent working around conditioning rather than converging. And
# the referee residual, computed from the problem data alone, says whether a `SOLVED` at
# `eps = 1e-8` means what it says: the method's own residuals are measured on the regularized
# system it factored, the referee's are not.
#
# `:indirect` is absent: the interior-point method runs conjugate gradients only with a
# caller-supplied preconditioner, which is `bench/ipm_matrixfree.jl`'s subject.
using PureOSQP, PureIPM, PureQPBase
using LDLFactorizations, LinearAlgebra, Printf, JSON, Random, SparseArrays
using Chairmarks, Statistics

BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "..", "..", "bench", "helpers_conditioning.jl"))

const N = 300
const KAPPAS = (1.0e4, 1.0e8, 1.0e10, 1.0e12)
# The defaults: this measures what a caller gets, and the interior-point defaults are already
# `1e-8` with a `max_iter` of 100. A run that needs more than that has not converged.
const OPTS = (eps_abs = 1.0e-8, eps_rel = 1.0e-8)

"""
    problems(kappa) -> Vector

The same size and conditioning in two shapes: one dense pair, and one split into blocks so the
block backend is reachable. The block problem's `κ` is its blocks', which is the whole matrix's
since the blocks are independent.
"""
function problems(kappa)
    rng = MersenneTwister(78)
    dense = (spd(rng, N, kappa), illconditioned(rng, N, kappa))
    K = 6
    nb = N ÷ K
    blocked = (
        PureQPBase.BlockDiagonal([spd(rng, nb, kappa) for _ in 1:K]),
        PureQPBase.BlockDiagonal([illconditioned(rng, nb, kappa) for _ in 1:K]),
    )
    return [
        ("dense", dense..., (:auto, :kkt)),
        ("blocks of $nb", blocked..., (:auto, :dense, :kkt)),
    ]
end

"The largest optimality residual of `(x, y)`, computed from the problem data alone."
function referee(P, A, q, l, u, x, y)
    Ax = A * x
    z = clamp.(Ax, l, u)
    r_prim = isempty(Ax) ? 0.0 : maximum(abs, Ax .- z)
    r_dual = maximum(abs, P * x .+ q .+ A' * y)
    return max(r_prim, r_dual)
end

println("\nInterior point against conditioning: n = $N, eps = $(OPTS.eps_abs).\n")

rows = NamedTuple[]
for kappa in KAPPAS
    @printf("κ = %.0e\n", kappa)
    @printf(
        "%-13s %-8s %-14s %-18s %5s %6s %11s %11s %9s\n",
        "shape", "linsys", "backend", "status", "iter", "bumps", "objective", "referee", "time"
    )
    println("-"^106)
    for (shape, P, A, settings) in problems(kappa)
        n = size(P, 2)
        q = randn(MersenneTwister(5), n)
        b = A * randn(MersenneTwister(6), n)
        l, u = b .- rand(MersenneTwister(7), n), b .+ rand(MersenneTwister(8), n)
        Pd, Ad = Matrix(P), Matrix(A)
        for ls in settings
            ws = setup(P, q, A, l, u, InteriorPoint(); linsys = ls, OPTS...)
            info = PureQPBase.backend_info(ws.linsys)
            # Warmed before timing: each backend is a fresh specialization, and a first call
            # measures the compiler rather than the solve.
            r = solve!(ws)
            ref = referee(Pd, Ad, q, l, u, r.x, r.y)
            t = median(
                s.time for s in
                    @be(solve(P, q, A, l, u, InteriorPoint(); linsys = ls, OPTS...), seconds = 2).samples
            )
            push!(
                rows, (;
                    kappa, shape, linsys = string(ls), backend = string(info.name),
                    status = string(r.status), iter = r.iter, reg_bumps = ws.reg_bumps,
                    obj_val = r.obj_val, referee = ref, seconds = t,
                )
            )
            @printf(
                "%-13s %-8s %-14s %-18s %5d %6d %11.5g %11.2e %6.1f ms\n",
                shape, ls, info.name, r.status, r.iter, ws.reg_bumps, r.obj_val, ref, 1.0e3t
            )
            flush(stdout)
        end
    end
    println()
end

open(joinpath(@__DIR__, "results", "ipm_illconditioned.json"), "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "blas_threads" => BLAS.get_num_threads(),
            "n" => N, "kappas" => collect(KAPPAS), "eps" => OPTS.eps_abs,
            "cases" => [Dict(string(k) => string(v) for (k, v) in pairs(r)) for r in rows],
        ), 2
    )
end
println("saved bench/results/ipm_illconditioned.json")
