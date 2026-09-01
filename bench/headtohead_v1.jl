# PureOSQP against libosqp 1.x, reached by `ccall` through `bench/osqp_v1.jl`.
#
# 1.x is the version whose termination rules this package follows -- `check_dualgap` defaults
# on because 1.x does -- so this is the comparison that tests what is actually implemented.
# 0.6.2 is checked separately by `bench/headtohead.jl` through OSQP.jl.
#
# `OSQP_jll` v100 conflicts with the v0.6 one that OSQP.jl pins, so this runs in its own
# project. From a fresh checkout:
#
#     julia --project=bench/osqpv1 -e 'using Pkg; Pkg.instantiate()'
#     julia --project=bench/osqpv1 bench/headtohead_v1.jl
#
# Iteration counts and objectives, not timings. Both solvers run in this process now, so a
# timing comparison would be fair -- but the point of this file is agreement, and
# `bench/headtohead.jl` already carries the timings against 0.6.2.
using PureOSQP, LinearAlgebra, Random, Printf, JSON

include(joinpath(@__DIR__, "osqp_v1.jl"))

const RESULTS = joinpath(@__DIR__, "results", "headtohead_v1.json")
# `adaptive_rho_interval` pinned on both sides so the comparison is deterministic, matching
# what `bench/headtohead.jl` does against 0.6.2.
const OPTS = (eps_abs = 1.0e-6, eps_rel = 1.0e-6, max_iter = 20_000, adaptive_rho_interval = 50)
const CASES = [(10, 20), (25, 50), (50, 100), (100, 200), (200, 400), (100, 50), (100, 1000)]

"A dense random QP with a strictly convex objective."
function problem(n, m; seed = n + m)
    rng = MersenneTwister(seed)
    X = randn(rng, n, n)
    P = Matrix(Symmetric(X'X)) + n * I
    A = randn(rng, m, n)
    b = A * randn(rng, n)
    return P, randn(rng, n), A, b .- rand(rng, m), b .+ rand(rng, m)
end

rows = NamedTuple[]
@printf(
    "%5s %6s | %8s %8s %7s | %11s %11s\n",
    "n", "m", "PureOSQP", "libosqp", "match", "obj rel Δ", "max |Δx|"
)
println("-"^68)
mismatches = 0
for (n, m) in CASES
    P, q, A, l, u = problem(n, m)
    c = solve_v1(P, q, A, l, u; verbose = 0, OPTS...)
    j = PureOSQP.solve(P, q, A, l, u; OPTS...)
    dobj = abs(c.obj_val - j.obj_val) / max(1, abs(j.obj_val))
    dx = maximum(abs, c.x .- j.x)
    same = c.iter == j.iter
    same || (global mismatches += 1)
    push!(
        rows, (;
            n, m, iter_pure = j.iter, iter_libosqp = c.iter, iters_match = same,
            obj_pure = j.obj_val, obj_libosqp = c.obj_val, obj_rel_diff = dobj, max_dx = dx,
            primdual_int_libosqp = c.primdual_int,
        )
    )
    @printf(
        "%5d %6d | %8d %8d %7s | %11.2e %11.2e\n",
        n, m, j.iter, c.iter, same ? "yes" : "NO", dobj, dx
    )
    flush(stdout)
end

println()
if iszero(mismatches)
    println("Iteration counts identical on all $(length(CASES)) cases.")
else
    println("$mismatches of $(length(CASES)) cases differ in iteration count.")
end

open(RESULTS, "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "libosqp_version" => unsafe_string(
                ccall((:osqp_version, osqp_builtin_double), Cstring, ())
            ),
            "eps" => OPTS.eps_abs, "adaptive_rho_interval" => OPTS.adaptive_rho_interval,
            "cases" => rows,
        ), 2
    )
end
println("\nwrote $RESULTS")
