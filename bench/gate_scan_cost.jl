# What the density gate's scan of `A` costs against the setup it gates.
#
# `densest_row` runs once in `sparse_kkt_backend` and again in `cholmod_backend`, so a sparse
# `setup` that descends past the first rung pays for two passes over `A`'s stored row indices.
# Whether that duplication is worth removing is a question about its share of setup, and this
# file measures both halves on the OSQP suite plus two larger sparse problems.
#
# Swept at two BLAS thread counts: the scan is scalar and `setup`'s factorizations are not, so
# the share the scan holds depends on how many threads the caller gives the rest of setup.
using PureOSQP, LinearAlgebra, SparseArrays
using Chairmarks, Printf, JSON, Statistics, Random

include(joinpath(@__DIR__, "suite_problems.jl"))

const EXT = Base.get_extension(PureOSQP, :PureOSQPSparseArraysExt)
const THREADS = (1, min(8, Sys.CPU_THREADS))
const BUDGET = 0.3
const RESULTS = joinpath(@__DIR__, "results", "gate_scan_cost.json")

"""
    big_qp(n, m; seed) -> (P, q, A, l, u)

A sparse QP large enough to put `setup` in the hundreds of milliseconds, so the scan's share
is read over three orders of magnitude of problem size rather than one.
"""
function big_qp(n, m; seed = 3)
    rng = MersenneTwister(seed)
    Pu = sprandn(rng, n, n, 0.02)
    P = Pu + Pu' + (2n) * I
    A = sprandn(rng, m, n, 0.02)
    l = -rand(rng, m) .- 1
    return P, randn(rng, n), A, l, -copy(l)
end

cases = vcat(
    CASES,
    [("Sparse 800", () -> big_qp(800, 800)), ("Sparse 1600", () -> big_qp(1600, 1600))],
)

med(x) = median(s.time for s in x.samples)

rows = NamedTuple[]
for nthreads in THREADS
    BLAS.set_num_threads(nthreads)
    @printf("\nBLAS threads = %d\n", nthreads)
    @printf(
        "%-12s %6s %6s %9s | %10s %11s %9s\n",
        "case", "n", "m", "nnz(A)", "scan", "setup", "2 scans"
    )
    println("-"^70)
    for (name, build) in cases
        P, q, A, l, u = build()
        A isa SparseMatrixCSC || continue
        n, m = size(A, 2), size(A, 1)
        backend = String(PureOSQP.backend_name(PureOSQP.setup(P, q, A, l, u).linsys))
        EXT.densest_row(A)
        sc = @be EXT.densest_row($A) seconds = BUDGET
        st = @be PureOSQP.setup($P, $q, $A, $l, $u) seconds = BUDGET
        tsc, tst = med(sc), med(st)
        push!(
            rows, (;
                case = name, n, m, nnz_A = nnz(A), blas_threads = nthreads, backend,
                scan = tsc, setup = tst, two_scan_share = 2tsc / tst,
            )
        )
        @printf(
            "%-12s %6d %6d %9d | %7.3f µs %8.3f ms %8.4f%%\n",
            name, n, m, nnz(A), 1.0e6tsc, 1.0e3tst, 100 * 2tsc / tst
        )
        flush(stdout)
    end
end

open(RESULTS, "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "cpu_threads" => Sys.CPU_THREADS,
            "budget_seconds" => BUDGET,
            "cases" => rows,
        ), 2
    )
end
println("\nwrote $RESULTS")
