# How the interior-point method scales in `n`, dense and sparse, against Clarabel and against
# the operator-splitting method.
#
# `bench/ipm_vs_clarabel.jl` compares the three on the suite classes at one size each, which
# says who wins a given problem but not how the gap moves with size. This sweeps `n` on random
# QPs of both representations, holding `m = 2n`.
#
# Tolerances follow `bench/ipm_vs_clarabel.jl`: both interior-point methods at `1e-8`, their
# own order of accuracy, and the operator-splitting method at `1e-6`, the tightest it reaches
# in a modest iteration count here. Its column is a wall clock at a looser tolerance, not a
# like-for-like comparison, and `max |Δx|` says how far apart the answers are.
using PureOSQP, PureIPM, PureQPBase
using LinearAlgebra, SparseArrays, Random, JSON, Chairmarks, Printf

include(joinpath(@__DIR__, "..", "..", "bench", "helpers_clarabel.jl"))

BLAS.set_num_threads(1)

const IPM_TOL = 1.0e-8
const ADMM_TOL = 1.0e-6
const SECONDS = 0.3
const SIZES = (50, 100, 200, 400)
# Enough nonzeros per row that the sparse pair is genuinely sparse at every size in the sweep.
const DENSITY = 0.05

"""
    problem(n, kind, rng) -> (P, q, A, l, u)

A random convex QP with `m = 2n` two-sided rows, either dense or sparse. The sparse pair is
the same problem shape with `DENSITY` of the entries kept, so the two columns differ in
representation and sparsity together — which is what a caller actually chooses between.
"""
function problem(n, kind, rng)
    m = 2n
    if kind === :dense
        X = randn(rng, n, n)
        P = Matrix(Symmetric(X'X / n + I))
        A = randn(rng, m, n)
    else
        X = sprandn(rng, n, n, DENSITY)
        P = Matrix(Symmetric(sparse(X'X / n + I)))
        P = sparse(P)
        A = sprandn(rng, m, n, DENSITY) + sparse(1.0I, m, n)
    end
    q = randn(rng, n)
    b = A * randn(rng, n)
    return P, q, A, b .- rand(rng, m), b .+ rand(rng, m)
end

rows = NamedTuple[]
for kind in (:dense, :sparse)
    println("\n", uppercase(String(kind)), " random QP, m = 2n\n")
    @printf(
        "%6s %6s | %-20s | %-20s | %-20s | %s\n",
        "n", "m", "IPM (eps 1e-8)", "Clarabel (tol 1e-8)", "ADMM (eps 1e-6)", "max |Δx|/|x|"
    )
    println("-"^108)
    for n in SIZES
        rng = MersenneTwister(4000 + n)
        P, q, A, l, u = problem(n, kind, rng)
        m = size(A, 1)

        ipm = solve(P, q, A, l, u, InteriorPoint(); eps_abs = IPM_TOL, eps_rel = IPM_TOL)
        ipm_bm = @b solve($P, $q, $A, $l, $u, InteriorPoint(); eps_abs = IPM_TOL, eps_rel = IPM_TOL) seconds = SECONDS

        admm = solve(P, q, A, l, u; eps_abs = ADMM_TOL, eps_rel = ADMM_TOL, max_iter = 20_000)
        admm_bm = @b solve($P, $q, $A, $l, $u; eps_abs = ADMM_TOL, eps_rel = ADMM_TOL, max_iter = 20_000) seconds = SECONDS

        clar = run_clarabel(P, q, A, l, u; tol = IPM_TOL)
        clar_bm = @b run_clarabel($P, $q, $A, $l, $u; tol = IPM_TOL) seconds = SECONDS

        scale = max(1.0, maximum(abs, ipm.x; init = 0.0))
        dx_clarabel = maximum(abs, ipm.x .- clar.x; init = 0.0) / scale
        dx_admm = maximum(abs, ipm.x .- admm.x; init = 0.0) / scale

        push!(
            rows, (;
                kind = String(kind), n, m,
                backend = string(PureQPBase.backend_name(setup(P, q, A, l, u, InteriorPoint()).linsys)),
                ipm_iter = ipm.iter, ipm_ms = 1.0e3ipm_bm.time, ipm_status = string(ipm.status),
                clarabel_iter = clar.iterations, clarabel_ms = 1.0e3clar_bm.time,
                admm_iter = admm.iter, admm_ms = 1.0e3admm_bm.time, admm_status = string(admm.status),
                dx_ipm_clarabel = dx_clarabel, dx_ipm_admm = dx_admm,
            )
        )
        @printf(
            "%6d %6d | %3d it %9.3f ms | %3d it %9.3f ms | %5d it %9.3f ms | %.1e / %.1e\n",
            n, m, ipm.iter, 1.0e3ipm_bm.time, clar.iterations, 1.0e3clar_bm.time,
            admm.iter, 1.0e3admm_bm.time, dx_clarabel, dx_admm
        )
        flush(stdout)
    end
end

open(joinpath(@__DIR__, "results", "ipm_scaling.json"), "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "blas_threads" => BLAS.get_num_threads(),
            "ipm_tol" => IPM_TOL, "admm_tol" => ADMM_TOL,
            "sizes" => collect(SIZES), "density" => DENSITY,
            "cases" => [Dict(string(k) => string(v) for (k, v) in pairs(r)) for r in rows],
        ), 2
    )
end
println("\nsaved bench/results/ipm_scaling.json")
