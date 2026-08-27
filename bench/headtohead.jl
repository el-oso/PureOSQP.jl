# PureOSQP vs the reference C implementation (OSQP.jl 0.8.1 / libosqp 0.6.2) on dense QPs.
# Both solvers get identical settings and a deterministic rho schedule.
using PureOSQP, OSQP, LinearAlgebra, SparseArrays, BenchmarkTools, Random, Printf, JSON

BLAS.set_num_threads(1)

const SETTINGS = (
    eps_abs = 1.0e-6, eps_rel = 1.0e-6, max_iter = 20_000,
    scaling = 10, adaptive_rho = true, polish = false,
)

function make_problem(n, m; seed = 0)
    Random.seed!(seed)
    X = randn(n, n)
    P = X'X / n + I
    q = randn(n)
    A = randn(m, n)
    x0 = randn(n)
    Ax = A * x0
    l = Ax .- rand(m)
    u = Ax .+ rand(m)
    return (P, q, A, l, u)
end

function run_osqp(P, q, A, l, u)
    model = OSQP.Model()
    OSQP.setup!(
        model; P = sparse(Symmetric(P)), q = q, A = sparse(A), l = l, u = u,
        verbose = false, adaptive_rho_interval = 50, check_termination = 25,
        SETTINGS...
    )
    return OSQP.solve!(model)
end

function bench_case(n, m; seed = 0)
    P, q, A, l, u = make_problem(n, m; seed)
    sp = PureOSQP.solve(P, q, A, l, u; SETTINGS...)
    rc = run_osqp(P, q, A, l, u)
    @assert sp.status == PureOSQP.SOLVED "PureOSQP: $(sp.status)"
    @assert rc.info.status == :Solved "C OSQP: $(rc.info.status)"
    tj = @belapsed PureOSQP.solve($P, $q, $A, $l, $u; SETTINGS...)
    tc = @belapsed run_osqp($P, $q, $A, $l, $u)
    objdiff = abs(sp.obj_val - rc.info.obj_val) / max(1, abs(rc.info.obj_val))
    return (;
        n, m, t_pure = tj, t_c = tc, speedup = tc / tj,
        iter_pure = sp.iter, iter_c = rc.info.iter, objdiff,
    )
end

const CASES = [
    (10, 20), (25, 50), (50, 100), (100, 200), (200, 400), (400, 800),
    (100, 50), (200, 100), (100, 1000), (200, 2000),
]

results = NamedTuple[]
@printf(
    "%5s %6s | %11s %11s %8s | %6s %6s | %9s\n",
    "n", "m", "PureOSQP", "C OSQP", "speedup", "it_pu", "it_C", "obj rel Δ"
)
println("-"^76)
for (n, m) in CASES
    r = bench_case(n, m)
    push!(results, r)
    @printf(
        "%5d %6d | %9.3f ms %9.3f ms %7.2fx | %6d %6d | %9.2e\n",
        r.n, r.m, 1.0e3r.t_pure, 1.0e3r.t_c, r.speedup, r.iter_pure, r.iter_c, r.objdiff
    )
    flush(stdout)
end

open(joinpath(@__DIR__, "results", "headtohead.json"), "w") do io
    JSON.print(
        io, Dict(
            "settings" => Dict(pairs(SETTINGS)),
            "julia_version" => string(VERSION),
            "results" => results
        ), 2
    )
end
println("\nsaved bench/results/headtohead.json")
