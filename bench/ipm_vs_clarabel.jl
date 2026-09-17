# The interior-point method against Clarabel (also an interior-point method) and against
# ADMM, on the OSQP benchmark suite's problem classes at their smallest representative sizes.
#
# `InteriorPoint()` runs at `eps_abs = eps_rel = 1e-8`, its own default order of accuracy;
# `Clarabel` is set to the same gap tolerance, so `x` is compared between two solvers aiming
# at the same accuracy. `OperatorSplitting()` runs separately at `eps_abs = eps_rel = 1e-6`,
# the tightest tolerance ADMM reaches in a modest iteration count on these problems, so its
# iteration count and wall clock are reported side by side with the IPM's rather than folded
# into the same referee.
#
#     julia --project=bench bench/ipm_vs_clarabel.jl    # writes bench/results/ipm_vs_clarabel.json
using PureOSQP, PureIPM, Clarabel
using LinearAlgebra, SparseArrays, Random, JSON, Chairmarks, Printf

include(joinpath(@__DIR__, "suite_problems.jl"))

BLAS.set_num_threads(1)

const RESULTS = joinpath(@__DIR__, "results", "ipm_vs_clarabel.json")
const IPM_TOL = 1.0e-8
const ADMM_TOL = 1.0e-6
const SECONDS = 0.3

"The smallest size of each suite class that still exercises its structure."
const SMALL_CASES = [
    ("Random QP", () -> random_qp(6)),
    ("Eq QP", () -> eq_qp(20)),
    ("Portfolio", () -> portfolio(1)),
    ("Lasso", () -> lasso(2)),
    ("SVM", () -> svm(2)),
    ("Huber", () -> huber(2)),
    ("Control", () -> control(4)),
]

include(joinpath(@__DIR__, "helpers_clarabel.jl"))

function run_case(name, gen)
    P, q, A, l, u = gen()
    n, m = size(A, 2), size(A, 1)

    ipm = PureOSQP.solve(P, q, A, l, u, PureIPM.InteriorPoint(); eps_abs = IPM_TOL, eps_rel = IPM_TOL)
    ipm_bm = @b PureOSQP.solve($P, $q, $A, $l, $u, PureIPM.InteriorPoint(); eps_abs = IPM_TOL, eps_rel = IPM_TOL) seconds = SECONDS

    admm = PureOSQP.solve(P, q, A, l, u; eps_abs = ADMM_TOL, eps_rel = ADMM_TOL, max_iter = 20_000)
    admm_bm = @b PureOSQP.solve($P, $q, $A, $l, $u; eps_abs = ADMM_TOL, eps_rel = ADMM_TOL, max_iter = 20_000) seconds = SECONDS

    clar = run_clarabel(P, q, A, l, u; tol = IPM_TOL)
    clar_bm = @b run_clarabel($P, $q, $A, $l, $u; tol = IPM_TOL) seconds = SECONDS

    dx_clarabel = maximum(abs, ipm.x .- clar.x; init = 0.0) / max(1.0, maximum(abs, ipm.x; init = 0.0))
    dx_admm = maximum(abs, ipm.x .- admm.x; init = 0.0) / max(1.0, maximum(abs, ipm.x; init = 0.0))

    @printf(
        "%-10s n=%-4d m=%-5d | ipm %3d it %7.3f ms | clarabel %3d it %7.3f ms | admm %5d it %7.3f ms | dx(ipm,clarabel)=%.1e dx(ipm,admm)=%.1e\n",
        name, n, m, ipm.iter, 1.0e3ipm_bm.time, clar.iterations, 1.0e3clar_bm.time,
        admm.iter, 1.0e3admm_bm.time, dx_clarabel, dx_admm,
    )
    flush(stdout)
    return (;
        name, n, m,
        ipm = (; iter = ipm.iter, status = String(Symbol(ipm.status)), time_ms = 1.0e3ipm_bm.time, obj = ipm.obj_val),
        clarabel = (; iter = clar.iterations, status = String(Symbol(clar.status)), time_ms = 1.0e3clar_bm.time, obj = clar.obj_val),
        admm = (; iter = admm.iter, status = String(Symbol(admm.status)), time_ms = 1.0e3admm_bm.time, obj = admm.obj_val),
        dx_ipm_clarabel = dx_clarabel, dx_ipm_admm = dx_admm,
    )
end

@printf(
    "%-10s %-8s %-7s | %-14s | %-18s | %-18s | %s\n",
    "class", "n", "m", "IPM (eps 1e-8)", "Clarabel (tol 1e-8)", "ADMM (eps 1e-6)", "max |Δx|/|x|"
)
println("-"^130)
results = [run_case(name, gen) for (name, gen) in SMALL_CASES]

open(RESULTS, "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "blas_threads" => BLAS.get_num_threads(),
            "ipm_tol" => IPM_TOL,
            "admm_tol" => ADMM_TOL,
            "clarabel_version" => string(pkgversion(Clarabel)),
            "results" => [
                Dict(
                    "name" => r.name, "n" => r.n, "m" => r.m,
                    "ipm" => Dict(pairs(r.ipm)), "clarabel" => Dict(pairs(r.clarabel)),
                    "admm" => Dict(pairs(r.admm)),
                    "dx_ipm_clarabel" => r.dx_ipm_clarabel, "dx_ipm_admm" => r.dx_ipm_admm,
                ) for r in results
            ],
        ), 2
    )
end
println("\nsaved bench/results/ipm_vs_clarabel.json")
