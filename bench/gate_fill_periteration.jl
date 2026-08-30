# What the sparse factor costs per iteration, against the dense inverse's `symv`.
#
# The fill limit is stated in `nnz(L)/n²` and set below the point where the *per-iteration*
# solve stops paying, so that the accepted region wins on the factorization and on the solve
# both. `bench/gate_crossover_fill.jl` measures whole runs instead, where one factorization is
# amortized over many iterations, and puts the crossing much further out. Neither answers what
# the per-iteration loss actually is between the two crossings, which is the number the limit
# should be priced on.
#
# Each row builds a problem whose reduced matrix has a target fill, then times one
# `solve_system!` on the sparse backend and on `linsys = :dense`, on the same workspace data.
# The solve is the whole of what an iteration spends in the linear system.
using PureOSQP, LDLFactorizations, LinearAlgebra, SparseArrays
using Chairmarks, Printf, JSON, Statistics, Random

BLAS.set_num_threads(1)

const OPTS = (eps_abs = 1.0e-9, eps_rel = 1.0e-9)
const BUDGET = parse(Float64, get(ENV, "GATE_BUDGET", "0.5"))
const SIZES = isempty(ARGS) ? (500, 1000) : Tuple(parse.(Int, ARGS))

"""
    banded_qp(n, b) -> (P, q, A, l, u)

A box-constrained QP whose reduced matrix is banded with half-bandwidth `b`, held sparse so
the sparse rungs are the ones that see it. Fill grows with `b`, which is the sweep variable.
"""
function banded_qp(n, b)
    rng = MersenneTwister(9161 + 31 * n + b)
    P = spdiagm(0 => rand(rng, n) .+ 1.0)
    rows, cols, vals = Int[], Int[], Float64[]
    for j in 1:n, i in max(1, j - b):min(n, j + b)
        push!(rows, i)
        push!(cols, j)
        push!(vals, i == j ? 1.0 : 0.05 / b)
    end
    A = sparse(rows, cols, vals, n, n)
    return P, randn(rng, n), A, -rand(rng, n), rand(rng, n)
end

"The reduced factor's fill, as the selection thresholds state it."
fill_of(ws) = PureOSQP.factor_fill(ws)

rows = NamedTuple[]
println("\nPer-iteration solve: sparse factor against the dense inverse.\n")
@printf(
    "%6s %6s %8s %-18s | %11s %11s | %8s | %6s\n",
    "n", "b", "fill", "sparse backend", "sparse µs", "dense µs", "sparse×", "iter"
)
println("-"^92)

for n in SIZES
    for b in unique(round.(Int, n .* (0.005, 0.02, 0.04, 0.06, 0.08, 0.1, 0.13, 0.16, 0.2)))
        b < 1 && continue
        P, q, A, l, u = banded_qp(n, b)
        # `linsys = :dense` is the alternative the limit chooses when it declines, so it is
        # what the sparse solve has to beat.
        sparse_ws = PureOSQP.setup(P, q, A, l, u; OPTS...)
        dense_ws = PureOSQP.setup(P, q, A, l, u; linsys = :dense, OPTS...)
        name = PureOSQP.backend_name(sparse_ws.linsys)
        # A backend that formed the matrix densely anyway is not a sparse factor, and its row
        # would compare the dense path with itself.
        name in (:ldlfactorizations, :cholmod, :ldl_kkt, :sparse_kkt) || continue
        f = fill_of(sparse_ws)
        bx, bz = randn(n), randn(n)
        ts = @be PureOSQP.solve_system!($sparse_ws.linsys, $sparse_ws, $bx, $bz) seconds = BUDGET
        td = @be PureOSQP.solve_system!($dense_ws.linsys, $dense_ws, $bx, $bz) seconds = BUDGET
        s, d = median(x.time for x in ts.samples), median(x.time for x in td.samples)
        sol = PureOSQP.solve(P, q, A, l, u; OPTS...)
        push!(
            rows, (;
                n, b, fill = f, backend = string(name),
                sparse_seconds = s, dense_seconds = d, ratio = d / s, iter = sol.iter,
                sparse_samples = [x.time for x in ts.samples],
                dense_samples = [x.time for x in td.samples],
            )
        )
        @printf(
            "%6d %6d %8.4f %-18s | %8.2f µs %8.2f µs | %6.2fx | %6d\n",
            n, b, f, name, 1.0e6s, 1.0e6d, d / s, sol.iter
        )
        flush(stdout)
    end
end

open(joinpath(@__DIR__, "results", "gate_fill_periteration.json"), "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "blas_threads" => BLAS.get_num_threads(),
            "budget_seconds" => BUDGET,
            "cases" => rows,
        ), 2
    )
end

println("\nCrossing, where the sparse solve stops beating the dense one:\n")
for n in SIZES
    rs = filter(r -> r.n == n, rows)
    isempty(rs) && continue
    lost = findfirst(r -> r.ratio < 1, rs)
    if isnothing(lost)
        @printf("  n = %5d   sparse ahead at every fill measured\n", n)
    else
        @printf(
            "  n = %5d   crosses between fill %.4f (%.2fx) and %.4f (%.2fx)\n",
            n, rs[max(lost - 1, 1)].fill, rs[max(lost - 1, 1)].ratio, rs[lost].fill, rs[lost].ratio
        )
    end
end
println("\nsaved bench/results/gate_fill_periteration.json")
