# PureOSQP vs the reference C implementation on dense QPs, against both libosqp versions:
#   * 0.6.2 through OSQP.jl, timed here with BenchmarkTools;
#   * 1.x through the subprocess oracle in bench/oracle_v1, timed *inside that process*
#     around the same setup+solve span, so interpreter startup and the JSON round trip are
#     excluded. libosqp 1.x has no Julia wrapper — see bench/oracle_v1/README.md.
# The 1.x column is skipped, not failed, when no interpreter with `osqp` is available.
# Both solvers get identical settings and a deterministic rho schedule.
using PureOSQP, OSQP, LinearAlgebra, SparseArrays, BenchmarkTools, Random, Printf, JSON

BLAS.set_num_threads(1)

const ORACLE_V1 = joinpath(@__DIR__, "oracle_v1", "osqp_v1_oracle.py")

function find_interpreter()
    cands = String[]
    haskey(ENV, "PUREOSQP_PY") && push!(cands, ENV["PUREOSQP_PY"])
    push!(cands, joinpath(@__DIR__, "oracle_v1", ".venv", "bin", "python3"))
    for c in cands
        isfile(c) || continue
        try
            success(pipeline(`$c -c "import osqp"`; stdout = devnull, stderr = devnull)) && return c
        catch
        end
    end
    return nothing
end

nullable(v) = [isfinite(x) ? x : nothing for x in v]

"Time `setup + solve` under libosqp 1.x, measured inside the oracle process."
function osqp_v1(py, P, q, A, l, u; repeat = 5)
    problem = Dict(
        "P" => [collect(Float64, view(P, i, :)) for i in axes(P, 1)],
        "q" => collect(Float64, q),
        "A" => [collect(Float64, view(A, i, :)) for i in axes(A, 1)],
        "l" => nullable(l), "u" => nullable(u), "repeat" => repeat,
        "settings" => Dict{String, Any}(
            "eps_abs" => 1.0e-6, "eps_rel" => 1.0e-6, "max_iter" => 20_000, "scaling" => 10,
            "adaptive_rho_interval" => 50, "check_termination" => 25,
        ),
    )
    out = read(pipeline(IOBuffer(JSON.json(problem) * "\n"), `$py $ORACLE_V1`), String)
    r = JSON.parse(strip(out))
    haskey(r, "error") && error("libosqp 1.x oracle failed: $(r["error"])")
    return r
end

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

function bench_case(n, m, py; seed = 0)
    P, q, A, l, u = make_problem(n, m; seed)
    sp = PureOSQP.solve(P, q, A, l, u; SETTINGS...)
    rc = run_osqp(P, q, A, l, u)
    @assert sp.status == PureOSQP.SOLVED "PureOSQP: $(sp.status)"
    @assert rc.info.status == :Solved "C OSQP: $(rc.info.status)"
    tj = @belapsed PureOSQP.solve($P, $q, $A, $l, $u; SETTINGS...)
    tc = @belapsed run_osqp($P, $q, $A, $l, $u)
    t1 = NaN
    iter1 = -1
    if !isnothing(py)
        r1 = osqp_v1(py, P, q, A, l, u)
        @assert r1["status"] == "solved" "libosqp 1.x: $(r1["status"])"
        t1 = r1["total_time"]
        iter1 = r1["iter"]
    end
    objdiff = abs(sp.obj_val - rc.info.obj_val) / max(1, abs(rc.info.obj_val))
    return (;
        n, m, t_pure = tj, t_c = tc, t_v1 = t1,
        speedup = tc / tj, speedup_v1 = t1 / tj,
        iter_pure = sp.iter, iter_c = rc.info.iter, iter_v1 = iter1, objdiff,
    )
end

const CASES = [
    (10, 20), (25, 50), (50, 100), (100, 200), (200, 400), (400, 800),
    (100, 50), (200, 100), (100, 1000), (200, 2000),
]

py = find_interpreter()
isnothing(py) && @info "No interpreter with `osqp`; the libosqp 1.x columns are skipped."

results = NamedTuple[]
@printf(
    "%5s %6s | %10s %10s %7s | %10s %7s | %6s %6s %6s | %9s\n",
    "n", "m", "PureOSQP", "0.6.2", "vs", "1.x", "vs", "it_pu", "it_062", "it_1x", "obj rel Δ"
)
println("-"^100)
for (n, m) in CASES
    r = bench_case(n, m, py)
    push!(results, r)
    @printf(
        "%5d %6d | %8.3f ms %8.3f ms %6.2fx | %8.3f ms %6.2fx | %6d %6d %6d | %9.2e\n",
        r.n, r.m, 1.0e3r.t_pure, 1.0e3r.t_c, r.speedup, 1.0e3r.t_v1, r.speedup_v1,
        r.iter_pure, r.iter_c, r.iter_v1, r.objdiff
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
