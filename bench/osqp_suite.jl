# The OSQP benchmark suite's problem classes, against libosqp.
#
# Every other sparse benchmark here generates uniformly random sparsity, which is the worst
# case for a sparse factorization: a random graph has no separator, so the Cholesky factor
# fills in whatever the matrix looked like. That is why `SparseCholmod`'s fill gate refuses
# those problems, and it is also why they cannot say whether the gate is right — a corpus
# that never has exploitable structure cannot test a decision about exploitable structure.
#
# The classes themselves are in `suite_problems.jl`, which other benchmarks include.
using PureOSQP, OSQP, LinearAlgebra, SparseArrays, Chairmarks, LDLFactorizations
using Printf, JSON, Statistics

include(joinpath(@__DIR__, "suite_problems.jl"))

BLAS.set_num_threads(1)

const OPTS = (eps_abs = 1.0e-5, eps_rel = 1.0e-5, max_iter = 20_000)

# libosqp 0.6.2 has no duality-gap test and would reject the setting; with it on the two
# solvers stop on different criteria and the iteration counts would compare the criteria.
const PURE_ONLY = (check_dualgap = false,)

# Seconds given to each of the four measurements a row makes.
const BUDGET = 5

"""
    abba(a, b) -> (ta, tb)

Time `a` and `b` in the order a, b, b, a, and return each one's median.

A row that timed `a` fully before starting `b` charges any drift over the row — the clock
ramping, the package heating — entirely to whichever ran second. Reversing the order for the
second pair puts each one both early and late, so a drift that is monotonic over the row lands
on both equally rather than becoming a ratio.

The two turns are pooled and the median taken over all of their samples together, so a turn
that ran under a transient does not get half the weight of a clean one.
"""
function abba(a, b)
    ta1, tb1 = (@be a() seconds = BUDGET), (@be b() seconds = BUDGET)
    tb2, ta2 = (@be b() seconds = BUDGET), (@be a() seconds = BUDGET)
    return (pooled_median(ta1, ta2), pooled_median(tb1, tb2))
end

pooled_median(x, y) = median(s.time for s in Iterators.flatten((x.samples, y.samples)))

solve_pure(P, q, A, l, u) = PureOSQP.solve(P, q, A, l, u; OPTS..., PURE_ONLY...)

function solve_osqp(P, q, A, l, u)
    model = OSQP.Model()
    OSQP.setup!(
        model; P = sparse(Symmetric(P)), q = collect(q), A = sparse(A),
        l = collect(l), u = collect(u), verbose = false,
        adaptive_rho_interval = 50, check_termination = 25, OPTS...
    )
    return OSQP.solve!(model)
end


"""
    compare(name, prob) -> NamedTuple

Time both solvers on one problem and record which backend PureOSQP chose.

The two objectives must agree; a row whose answers differ is not a timing comparison and is
refused rather than reported.
"""
function compare(name, prob)
    P, q, A, l, u = prob
    sp = solve_pure(P, q, A, l, u)
    so = solve_osqp(P, q, A, l, u)
    ok = sp.status == PureOSQP.SOLVED && so.info.status == :Solved
    ok || return (; name, n = size(A, 2), m = size(A, 1), skipped = "$(sp.status)/$(so.info.status)")
    gap = abs(sp.obj_val - so.info.obj_val) / max(1, abs(so.info.obj_val))
    gap < 1.0e-4 || error("$name: objectives disagree by $gap")
    ws = PureOSQP.setup(P, q, A, l, u; OPTS..., PURE_ONLY...)
    tp, to = abba(() -> solve_pure(P, q, A, l, u), () -> solve_osqp(P, q, A, l, u))
    return (;
        name, n = size(A, 2), m = size(A, 1), nnz_A = nnz(sparse(A)), nnz_P = nnz(sparse(P)),
        backend = PureOSQP.backend_name(ws.linsys), t_pure = tp, t_osqp = to, ratio = to / tp,
        iter_pure = sp.iter, iter_osqp = so.info.iter, obj_gap = gap,
    )
end


println("\nThe OSQP benchmark suite's problem classes. Both solvers hold SparseMatrixCSC.\n")
@printf(
    "%-12s %6s %6s %8s %-16s | %11s %11s %8s | %6s %6s\n",
    "class", "n", "m", "nnz(A)", "PureOSQP backend", "PureOSQP", "OSQP", "vs OSQP", "it pu", "it os"
)
println("-"^108)
rows = NamedTuple[]
for (name, build) in CASES
    r = compare(name, build())
    push!(rows, r)
    if haskey(r, :skipped)
        @printf("%-12s %6d %6d %8s %-16s | %s\n", r.name, r.n, r.m, "-", "-", "skipped: " * r.skipped)
    else
        @printf(
            "%-12s %6d %6d %8d %-16s | %8.2f ms %8.2f ms %7.2fx | %6d %6d\n",
            r.name, r.n, r.m, r.nnz_A, r.backend,
            1.0e3r.t_pure, 1.0e3r.t_osqp, r.ratio, r.iter_pure, r.iter_osqp
        )
    end
    flush(stdout)
end

open(joinpath(@__DIR__, "results", "osqp_suite.json"), "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "blas_threads" => BLAS.get_num_threads(),
            "cases" => [Dict(string(k) => string(v) for (k, v) in pairs(r)) for r in rows],
        ), 2
    )
end
println("\nsaved bench/results/osqp_suite.json")
