# What the matrix-free route costs a products-only operator, against the same problem
# materialized.
#
# `P` is `Diagonal(d) + α v vᵀ`. Given as a `LazyPSD` it declines the dense terminal through
# `is_materializable` and lands on `IndirectCG`; given as the `n×n` matrix it names, it stops
# at `ReducedCholesky`. Both spellings are the same problem, so the two columns are a price
# for never forming the matrix rather than a comparison of two problems.
#
# Setup and one `admm_step!` are timed separately: setup is where the dense route pays its
# `O(n³)` factorization and the lazy one pays nothing, and the step is where the lazy route
# pays for an inexact inner solve every iteration.
#
# Swept at two BLAS thread counts. `ReducedCholesky`'s step is a threaded `symv` and the
# conjugate-gradient inner solve is a chain of products, so a single-threaded sweep answers a
# different question than the one the deployed solver -- which pins no threads -- runs.
#
# Run:  cd bench && jl operator_protocol.jl
using PureOSQP, LinearAlgebra
using Krylov                   # the matrix-free backend, a weak dependency
using Chairmarks, Printf, JSON, Statistics, Random

include(joinpath(@__DIR__, "lazy_operator.jl"))

const SIZES = (500, 1000)
const THREADS = (1, min(8, Sys.CPU_THREADS))
const BUDGET = 0.3
const RESULTS = joinpath(@__DIR__, "results", "operator_protocol.json")

"""
    problem(n) -> (P, q, A, l, u)

A box-constrained QP whose `P` is a positive diagonal plus a nonnegative rank-one term, and
whose `A` is dense. `scaling = 0` throughout: equilibration walks columns, which a
products-only operator cannot answer without the traversal overrides this file does not add.
"""
function problem(n)
    rng = MersenneTwister(4231 + 7 * n)
    d = rand(rng, n) .+ 2.0
    v = randn(rng, n) ./ sqrt(n)
    P = LazyPSD(d, v, 0.5)
    A = randn(rng, n, n) ./ sqrt(n)
    b = A * randn(rng, n)
    return P, randn(rng, n), A, b .- rand(rng, n), b .+ rand(rng, n)
end

const OPTS = (scaling = 0, eps_abs = 1.0e-8, eps_rel = 1.0e-8)

med(x) = median(s.time for s in x.samples)

rows = NamedTuple[]
for nthreads in THREADS
    BLAS.set_num_threads(nthreads)
    @printf("\nBLAS threads = %d\n", nthreads)
    @printf(
        "%6s %-10s %-10s | %11s %11s %8s | %11s %11s %8s\n",
        "n", "lazy", "dense", "lazy setup", "dense setup", "×", "lazy step", "dense step", "×"
    )
    println("-"^96)
    for n in SIZES
        P, q, A, l, u = problem(n)
        Pm = materialize(P)

        lazy = PureOSQP.setup(P, q, A, l, u; OPTS...)
        dense = PureOSQP.setup(Pm, q, A, l, u; OPTS...)
        ln = String(PureOSQP.backend_name(lazy.linsys))
        dn = String(PureOSQP.backend_name(dense.linsys))
        # The times mean nothing unless the two spellings are the same problem.
        PureOSQP.admm_step!(lazy)
        PureOSQP.admm_step!(dense)
        isapprox(lazy.x, dense.x; rtol = 1.0e-6) || error("n=$n: the two spellings disagree")

        sl = @be PureOSQP.setup($P, $q, $A, $l, $u; OPTS...) seconds = BUDGET
        sd = @be PureOSQP.setup($Pm, $q, $A, $l, $u; OPTS...) seconds = BUDGET
        vl = @be PureOSQP.admm_step!($lazy) seconds = BUDGET
        vd = @be PureOSQP.admm_step!($dense) seconds = BUDGET
        tsl, tsd, tvl, tvd = med(sl), med(sd), med(vl), med(vd)
        push!(
            rows, (;
                n, blas_threads = nthreads, lazy_backend = ln, dense_backend = dn,
                lazy_setup = tsl, dense_setup = tsd, setup_ratio = tsd / tsl,
                lazy_step = tvl, dense_step = tvd, step_ratio = tvd / tvl,
            )
        )
        @printf(
            "%6d %-10s %-10s | %8.3f ms %8.3f ms %7.2fx | %8.3f µs %8.3f µs %7.2fx\n",
            n, ln, dn, 1.0e3tsl, 1.0e3tsd, tsd / tsl, 1.0e6tvl, 1.0e6tvd, tvd / tvl
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
            "sizes" => collect(SIZES),
            "cases" => rows,
        ), 2
    )
end
println("\nwrote $RESULTS")
