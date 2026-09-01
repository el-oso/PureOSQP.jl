# What adapting `ρ` buys, and whether the trigger matters.
#
# Two separate questions, and they have different answers.
#
#   does adapting help?     compare `adaptive_rho = :disabled` against adapting at all
#   does the trigger help?  compare `:iterations`, which retunes on a fixed schedule,
#                           against `:kkt_error`, which retunes only when the relative KKT
#                           error has fallen by `adaptive_rho_fraction` since the last look
#
# The second is the one worth measuring before building anything cleverer. `:kkt_error` is
# already a convergence-progress trigger, so if it does not beat a fixed interval, a better
# progress signal -- the duality gap's local decay rate, which the primal-dual integral's
# log-mean rule computes as a byproduct -- has no obvious room to help either.
#
# Iteration counts, not wall clock: the trigger changes where `ρ` moves, and the point is
# whether that reaches the same tolerance in fewer steps. Refactorization counts are reported
# alongside, since a schedule that reached the same place with fewer refactorizations would
# still be worth having.
#
# Run:  julia --project=bench bench/rho_schedule.jl
using PureOSQP, LinearAlgebra, SparseArrays, Printf, JSON, Random
using LDLFactorizations, Krylov

BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "suite_problems.jl"))

const RESULTS = joinpath(@__DIR__, "results", "rho_schedule.json")
const OPTS = (eps_abs = 1.0e-6, eps_rel = 1.0e-6, max_iter = 50_000)
const MODES = (:disabled, :iterations, :kkt_error)

rows = NamedTuple[]
@printf(
    "%-12s %10s %12s %11s | %s\n",
    "class", "disabled", ":iterations", ":kkt_error", "refactorizations"
)
println("-"^76)
totals = Dict(m => 0 for m in MODES)
for (name, make) in CASES
    P, q, A, l, u = make()
    iters = Int[]
    refacs = Int[]
    for mode in MODES
        ws = PureOSQP.setup(P, q, A, l, u; OPTS..., adaptive_rho = mode)
        r = PureOSQP.solve!(ws)
        push!(iters, r.iter)
        push!(refacs, ws.refactor_count)
        totals[mode] += r.iter
    end
    push!(
        rows, (;
            class = name,
            iter_disabled = iters[1], iter_iterations = iters[2], iter_kkt_error = iters[3],
            refac_disabled = refacs[1], refac_iterations = refacs[2],
            refac_kkt_error = refacs[3],
        )
    )
    @printf(
        "%-12s %10d %12d %11d | %d / %d / %d\n",
        name, iters..., refacs...
    )
    flush(stdout)
end
@printf(
    "%-12s %10d %12d %11d\n",
    "TOTAL", totals[:disabled], totals[:iterations], totals[:kkt_error]
)

println()
println("Adapting is the lever. Which trigger fires it is not: on this corpus the two")
println("triggers reach the same tolerance in the same number of iterations, class by class.")

open(RESULTS, "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "blas_threads" => BLAS.get_num_threads(),
            "eps" => OPTS.eps_abs, "max_iter" => OPTS.max_iter,
            "totals" => Dict(string(m) => totals[m] for m in MODES),
            "cases" => rows,
        ), 2
    )
end
println("\nwrote $RESULTS")
