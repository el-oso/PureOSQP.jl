# PureOSQP against libosqp 1.x, reached through the subprocess oracle in bench/oracle_v1.
# OSQP.jl wraps 0.6.2 only; see bench/oracle_v1/README.md for why this route exists.
# Skipped, not failed, when no interpreter with `osqp` is available.
using PureOSQP, LinearAlgebra, Random, Printf, JSON

BLAS.set_num_threads(1)

const ORACLE = joinpath(@__DIR__, "oracle_v1", "osqp_v1_oracle.py")

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

function oracle_solve(py, P, q, A, l, u; settings = Dict{String, Any}())
    problem = Dict(
        "P" => [collect(Float64, view(P, i, :)) for i in axes(P, 1)],
        "q" => collect(Float64, q),
        "A" => [collect(Float64, view(A, i, :)) for i in axes(A, 1)],
        "l" => nullable(l), "u" => nullable(u), "settings" => settings,
    )
    out = read(pipeline(IOBuffer(JSON.json(problem) * "\n"), `$py $ORACLE`), String)
    return JSON.parse(strip(out))
end

function random_qp(n, m; seed = 0)
    Random.seed!(seed)
    X = randn(n, n)
    A = randn(m, n)
    Ax = A * randn(n)
    return (X'X / n + I, randn(n), A, Ax .- rand(m), Ax .+ rand(m))
end

py = find_interpreter()
if isnothing(py)
    @info """No interpreter with the `osqp` module found, so the libosqp 1.x comparison is
    skipped. See bench/oracle_v1/README.md to set one up."""
    exit(0)
end

const OPTS = (eps_abs = 1.0e-6, eps_rel = 1.0e-6, max_iter = 20_000, scaling = 10)
const PY_OPTS = Dict{String, Any}(
    "eps_abs" => 1.0e-6, "eps_rel" => 1.0e-6, "max_iter" => 20_000, "scaling" => 10,
    "adaptive_rho_interval" => 50, "check_termination" => 25,
)

const CASES = [(10, 20), (25, 50), (50, 100), (100, 200), (200, 400), (100, 50), (50, 500)]

# No timing column. Each oracle call pays a fresh interpreter start and a JSON round trip,
# well over a hundred milliseconds that has nothing to do with libosqp; a "speedup" built
# from that would be meaningless. bench/headtohead.jl is the timing comparison — this
# script checks agreement.
results = []
@printf(
    "%5s %6s | %-10s %-10s | %6s %6s | %10s\n",
    "n", "m", "PureOSQP", "libosqp1x", "it_pu", "it_1x", "obj rel Δ"
)
println("-"^72)
for (n, m) in CASES
    P, q, A, l, u = random_qp(n, m; seed = n + m)
    j = PureOSQP.solve(P, q, A, l, u; OPTS...)
    c = oracle_solve(py, P, q, A, l, u; settings = PY_OPTS)
    haskey(c, "error") && error("oracle failed on n=$n m=$m: $(c["error"])")
    objdiff = abs(j.obj_val - c["obj_val"]) / max(1, abs(c["obj_val"]))
    push!(
        results, (;
            n, m, status_pure = string(j.status), status_1x = c["status"],
            iter_pure = j.iter, iter_1x = c["iter"], objdiff,
        )
    )
    @printf(
        "%5d %6d | %-10s %-10s | %6d %6d | %10.2e\n",
        n, m, string(j.status), c["status"], j.iter, c["iter"], objdiff
    )
    flush(stdout)
end

matched = count(r -> r.iter_pure == r.iter_1x, results)
@printf(
    "\niteration counts identical on %d of %d cases; max objective rel Δ = %.2e\n",
    matched, length(results), maximum(r -> r.objdiff, results)
)

open(joinpath(@__DIR__, "results", "headtohead_v1.json"), "w") do io
    JSON.print(
        io, Dict(
            "oracle" => "libosqp 1.x via the osqp Python wheel, run as a subprocess",
            "julia_version" => string(VERSION),
            "note" => "agreement check only; no timings, because each oracle call pays interpreter startup and a JSON round trip",
            "results" => [Dict(string(k) => v for (k, v) in pairs(r)) for r in results],
        ), 2
    )
end
println("saved bench/results/headtohead_v1.json")
