# The sequential-resolve path under the interior-point method: `P` and `A` fixed, `q`, `l` and
# `u` changing every step, as a receding-horizon loop does. Compares
#   (1) setup once, then `update!` + `solve!` per step
#   (2) a fresh `setup` + `solve` per step, the naive loop
#
# What `update!` saves is different here than under operator splitting. That method keeps one
# factorization across the whole loop, so skipping setup skips the factorization; this one
# refactors every iteration regardless, so what is saved is the validation, the equilibration
# and the allocation of the workspace — and, when the bounds move a row into or out of equality
# or freeness, the reclassification is done in place rather than rebuilt.
using PureOSQP, PureIPM, PureQPBase
using LinearAlgebra, SparseArrays, LDLFactorizations, Random, Printf, JSON, Statistics

BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "..", "..", "bench", "suite_problems.jl"))

const OPTS = (eps_abs = 1.0e-8, eps_rel = 1.0e-8)
const STEPS = 20

"""
    steps(q, l, u, rng, k) -> Vector

`k` perturbations of the vector data, each a step of the loop. `P` and `A` never change, which
is what makes the workspace reusable.
"""
function steps(q, l, u, rng, k)
    return [
        (
            q .+ 0.05 .* randn(rng, length(q)), l .- 0.05 .* rand(rng, length(l)),
            u .+ 0.05 .* rand(rng, length(u)),
        ) for _ in 1:k
    ]
end

"Run the loop with one workspace, returning the elapsed seconds and the iterations taken."
function reuse(P, q, A, l, u, seq)
    ws = setup(P, q, A, l, u, InteriorPoint(); OPTS...)
    solve!(ws)
    iters = 0
    t = @elapsed for (q2, l2, u2) in seq
        update!(ws; q = q2, l = l2, u = u2)
        iters += solve!(ws).iter
    end
    return t, iters
end

"The same loop, building a workspace per step."
function fresh(P, q, A, l, u, seq)
    iters = 0
    t = @elapsed for (q2, l2, u2) in seq
        iters += solve(P, q2, A, l2, u2, InteriorPoint(); OPTS...).iter
    end
    return t, iters
end

"""
    median_of(f; repeats = 5) -> (seconds, iterations)

The median wall time of `f` over `repeats` runs, with the iteration count it reported. Each
run is a whole loop, so a median over runs is the right summary rather than a median over the
steps inside one.
"""
function median_of(f; repeats = 5)
    out = [f() for _ in 1:repeats]
    return median(first.(out)), last(out[1])
end

println("\nInterior point: $STEPS-step resolve loop, eps = $(OPTS.eps_abs).\n")
@printf(
    "%-12s %5s %5s %-18s %11s %11s %9s %9s\n",
    "class", "n", "m", "backend", "update! ms", "fresh ms", "speedup", "iters"
)
println("-"^92)

rows = NamedTuple[]
for (name, gen) in CASES
    P, q, A, l, u = gen()
    n, m = size(A, 2), size(A, 1)
    seq = steps(q, l, u, MersenneTwister(91), STEPS)
    ws = setup(P, q, A, l, u, InteriorPoint(); OPTS...)
    backend = PureQPBase.backend_name(ws.linsys)

    reuse(P, q, A, l, u, seq)                       # warm both paths before timing
    fresh(P, q, A, l, u, seq)
    tu, iu = median_of(() -> reuse(P, q, A, l, u, seq))
    tf, if_ = median_of(() -> fresh(P, q, A, l, u, seq))

    push!(
        rows, (;
            class = name, n, m, backend = string(backend), steps = STEPS,
            update_seconds = tu, fresh_seconds = tf, update_iters = iu, fresh_iters = if_,
        )
    )
    @printf(
        "%-12s %5d %5d %-18s %11.3f %11.3f %8.2fx %9d\n",
        name, n, m, backend, 1.0e3tu, 1.0e3tf, tf / tu, iu
    )
    flush(stdout)
end

open(joinpath(@__DIR__, "results", "ipm_update.json"), "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "blas_threads" => BLAS.get_num_threads(),
            "eps" => OPTS.eps_abs, "steps" => STEPS,
            "cases" => [Dict(string(k) => string(v) for (k, v) in pairs(r)) for r in rows],
        ), 2
    )
end
println("\nsaved PureIPM/bench/results/ipm_update.json")
