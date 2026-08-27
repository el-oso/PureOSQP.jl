# The sequential-resolve path: P and A fixed, q/l/u changing every step, as a receding
# horizon loop does. Compares
#   (1) PureOSQP with update!      -- setup once, then update! + solve! per step
#   (2) PureOSQP without update!   -- a fresh setup + solve per step, the naive loop
#   (3) libosqp 1.x                -- setup once, then update + solve per step
# The libosqp timings are measured inside its own process (see bench/oracle_v1), so they
# exclude interpreter startup and the JSON round trip.
using PureOSQP, LinearAlgebra, Random, Printf, JSON

BLAS.set_num_threads(1)

const ORACLE = joinpath(@__DIR__, "oracle_v1", "osqp_v1_oracle.py")
const STEPS = 20

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

function make_sequence(n, m; seed = 0, steps = STEPS)
    Random.seed!(seed)
    X = randn(n, n)
    P = X'X / n + I
    A = randn(m, n)
    q0 = randn(n)
    b = A * randn(n)
    l0, u0 = b .- rand(m), b .+ rand(m)
    seq = map(1:steps) do _
        bk = A * randn(n)
        (q = randn(n), l = bk .- rand(m), u = bk .+ rand(m))
    end
    return (P, q0, A, l0, u0, seq)
end

const OPTS = (eps_abs = 1.0e-6, eps_rel = 1.0e-6, max_iter = 20_000)
const PY_OPTS = Dict{String, Any}(
    "eps_abs" => 1.0e-6, "eps_rel" => 1.0e-6, "max_iter" => 20_000,
    "adaptive_rho_interval" => 50, "check_termination" => 25,
)

function pure_with_update(P, q0, A, l0, u0, seq)
    ws = setup(P, q0, A, l0, u0; OPTS...)
    objs = Float64[]
    t = @elapsed for s in seq
        update!(ws; q = s.q, l = s.l, u = s.u)
        push!(objs, PureOSQP.solve!(ws).obj_val)
    end
    return (t, objs, ws.refactor_count)
end

function pure_without_update(P, q0, A, l0, u0, seq)
    objs = Float64[]
    t = @elapsed for s in seq
        push!(objs, PureOSQP.solve(P, s.q, A, s.l, s.u; OPTS...).obj_val)
    end
    return (t, objs)
end

function osqp_sequence(py, P, q0, A, l0, u0, seq)
    problem = Dict(
        "P" => [collect(Float64, view(P, i, :)) for i in axes(P, 1)],
        "q" => collect(Float64, q0),
        "A" => [collect(Float64, view(A, i, :)) for i in axes(A, 1)],
        "l" => nullable(l0), "u" => nullable(u0), "settings" => PY_OPTS,
        "sequence" => [
            Dict(
                    "q" => collect(Float64, s.q), "l" => nullable(s.l),
                    "u" => nullable(s.u)
                ) for s in seq
        ],
    )
    out = read(pipeline(IOBuffer(JSON.json(problem) * "\n"), `$py $ORACLE`), String)
    r = JSON.parse(strip(out))
    haskey(r, "error") && error("oracle failed: $(r["error"])")
    t = sum(s -> s["update_time"] + s["solve_time"], r["solves"])
    return (t, Float64[s["obj_val"] for s in r["solves"]])
end

py = find_interpreter()
isnothing(py) && @info "No interpreter with `osqp` found; the libosqp 1.x column is skipped."

const CASES = [(10, 20), (25, 50), (50, 100), (100, 200), (200, 400)]

results = []
@printf(
    "%5s %6s | %11s %11s %8s | %11s %8s | %5s %9s\n", "n", "m",
    "update!", "fresh setup", "saved", "libosqp1x", "ratio", "refac", "obj Δ"
)
println("-"^88)
for (n, m) in CASES
    P, q0, A, l0, u0, seq = make_sequence(n, m; seed = n + m)
    pure_with_update(P, q0, A, l0, u0, seq[1:2])          # warm up compilation
    t_upd, objs_upd, nrefac = pure_with_update(P, q0, A, l0, u0, seq)
    t_new, objs_new = pure_without_update(P, q0, A, l0, u0, seq)
    # Two runs converged to the same tolerance need only agree to that tolerance; a
    # warm-started sequence and a cold sequence stop at different points inside it.
    agree = maximum(abs, objs_upd .- objs_new) / max(1, maximum(abs, objs_new))
    @assert agree < 100 * OPTS.eps_abs "update! and fresh setup disagree by $agree"
    t_c, objs_c = isnothing(py) ? (NaN, Float64[]) : osqp_sequence(py, P, q0, A, l0, u0, seq)
    objdiff = isempty(objs_c) ? NaN :
        maximum(abs, objs_upd .- objs_c) / max(1, maximum(abs, objs_c))
    push!(
        results, (;
            n, m, steps = STEPS, t_update = t_upd, t_fresh = t_new,
            t_osqp_v1 = t_c, refactorizations = nrefac, objdiff,
        )
    )
    @printf(
        "%5d %6d | %9.2f ms %9.2f ms %7.2fx | %9.2f ms %7.2fx | %5d %9.1e\n",
        n, m, 1.0e3t_upd, 1.0e3t_new, t_new / t_upd, 1.0e3t_c, t_c / t_upd,
        nrefac, objdiff
    )
    flush(stdout)
end

println("\n$STEPS solves per case. \"refac\" counts factorizations across the whole")
println("sequence: 1 from setup plus one per step whose constraint classification changed.")

open(joinpath(@__DIR__, "results", "update_bench.json"), "w") do io
    JSON.print(
        io, Dict(
            "steps" => STEPS, "julia_version" => string(VERSION),
            "oracle" => "libosqp 1.x, timed inside its own process",
            "results" => [Dict(string(k) => v for (k, v) in pairs(r)) for r in results],
        ), 2
    )
end
println("saved bench/results/update_bench.json")
