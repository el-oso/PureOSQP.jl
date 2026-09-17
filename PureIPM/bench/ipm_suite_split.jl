# Setup against iteration for the interior-point method, on the OSQP suite.
#
# `bench/ipm_vs_clarabel.jl` times the whole call. That is the number that matters, but it
# cannot say whether a class is slow because the first factorization costs more or because each
# Newton iteration does. The two are priced differently here than under operator splitting: that
# method factors once and iterates cheaply, this one refactors every iteration, so what setup
# buys is a starting point rather than a factorization to reuse.
#
# `iterate_ms` is the per-iteration cost the ranking in `recommend_linsys` uses, and it is what
# to compare across classes; `setup_ms` is paid once.
using PureOSQP, PureIPM, PureQPBase
using LinearAlgebra, SparseArrays, LDLFactorizations, Chairmarks, Printf, JSON, Statistics

BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "..", "..", "bench", "suite_problems.jl"))

const OPTS = (eps_abs = 1.0e-8, eps_rel = 1.0e-8)
const BUDGET = 5

median_seconds(f) = median(s.time for s in @be(f(), seconds = BUDGET).samples)

println("\nInterior point: setup against iteration on the OSQP suite, eps = $(OPTS.eps_abs).\n")
@printf(
    "%-12s %5s %5s %-18s %5s %9s %9s %10s %11s\n",
    "class", "n", "m", "backend", "iter", "setup ms", "solve ms", "total ms", "per iter ms"
)
println("-"^96)

rows = NamedTuple[]
for (name, gen) in CASES
    P, q, A, l, u = gen()
    n, m = size(A, 2), size(A, 1)
    ws = setup(P, q, A, l, u, InteriorPoint(); OPTS...)
    backend = PureQPBase.backend_name(ws.linsys)
    iter = solve!(ws).iter

    setup_s = median_seconds(() -> setup(P, q, A, l, u, InteriorPoint(); OPTS...))
    total_s = median_seconds(() -> solve(P, q, A, l, u, InteriorPoint(); OPTS...))
    # The loop is the total less the setup that preceded it; dividing by the iterations it ran
    # gives the per-iteration cost, which is what differs between classes.
    solve_s = total_s - setup_s
    per_iter = iter > 0 ? solve_s / iter : 0.0

    push!(
        rows, (;
            class = name, n, m, backend = string(backend), iter,
            setup_seconds = setup_s, solve_seconds = solve_s, total_seconds = total_s,
            iterate_seconds = per_iter,
        )
    )
    @printf(
        "%-12s %5d %5d %-18s %5d %9.3f %9.3f %10.3f %11.4f\n",
        name, n, m, backend, iter, 1.0e3setup_s, 1.0e3solve_s, 1.0e3total_s, 1.0e3per_iter
    )
    flush(stdout)
end

open(joinpath(@__DIR__, "results", "ipm_suite_split.json"), "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "blas_threads" => BLAS.get_num_threads(),
            "eps" => OPTS.eps_abs,
            "cases" => [Dict(string(k) => string(v) for (k, v) in pairs(r)) for r in rows],
        ), 2
    )
end
println("\nsaved bench/results/ipm_suite_split.json")
