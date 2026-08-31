# `TridiagonalReduced` against `ReducedCholesky` for a `Tridiagonal` `P` with a `Diagonal` `A`.
#
# That pair's reduced matrix has bandwidth 1, so both backends serve it and the choice between
# them is the question this file answers. Setup is timed through `setup` itself -- `:auto`
# against `linsys = :dense` -- while the per-iteration comparison builds both backends over one
# workspace, since `factorize!` and `solve_system!` read their data from the workspace rather
# than from their own copy of the problem.
#
# Swept at two BLAS thread counts. `ReducedCholesky`'s solve is a threaded `symv` and a
# tridiagonal `ldiv!` is not threaded, so a single-threaded sweep answers a different question
# than the one the deployed solver -- which pins no threads -- runs.
using PureOSQP, LinearAlgebra
using Chairmarks, Printf, JSON, Statistics, Random

const SIZES = (250, 500, 1000, 2000)
const THREADS = (1, min(8, Sys.CPU_THREADS))
const BUDGET = 0.3
const RESULTS = joinpath(@__DIR__, "results", "tridiagonal_rung.json")

"""
    problem(n) -> (P, q, A, l, u)

A box-constrained QP whose `P` is a diagonally dominant `Tridiagonal` and whose `A` is
`Diagonal`, so the reduced matrix has bandwidth 1 at every size here.
"""
function problem(n)
    rng = MersenneTwister(9721 + 7 * n)
    ev = rand(rng, n - 1) ./ 8
    P = Tridiagonal(copy(ev), rand(rng, n) .+ 3, copy(ev))
    A = Diagonal(rand(rng, n) .+ 0.5)
    return P, randn(rng, n), A, -rand(rng, n), rand(rng, n)
end

med(x) = median(s.time for s in x.samples)

rows = NamedTuple[]
for nthreads in THREADS
    BLAS.set_num_threads(nthreads)
    @printf("\nBLAS threads = %d\n", nthreads)
    @printf(
        "%6s %-12s | %10s %10s %8s | %11s %11s %8s\n",
        "n", "backend", "tri setup", "chol setup", "setup×", "tri solve", "chol solve", "solve×"
    )
    println("-"^86)
    for n in SIZES
        P, q, A, l, u = problem(n)
        name = String(PureOSQP.backend_name(PureOSQP.setup(P, q, A, l, u).linsys))
        # One workspace carries the dense backend; the tridiagonal one is built beside it and
        # reads the same equilibrated data through the same `ws`.
        ws = PureOSQP.setup(P, q, A, l, u; linsys = :dense)
        tl = PureOSQP.TridiagonalReduced(zeros(n), n)
        PureOSQP.factorize!(tl, ws) || error("n=$n: tridiagonal factorization failed")
        bx, bz = randn(n), randn(n)
        # The times mean nothing unless the two backends solve the same system.
        PureOSQP.solve_system!(tl, ws, bx, bz)
        xt = copy(ws.xtilde)
        PureOSQP.solve_system!(ws.linsys, ws, bx, bz)
        isapprox(xt, ws.xtilde; rtol = 1.0e-8) || error("n=$n: backends disagree")

        st = @be PureOSQP.setup($P, $q, $A, $l, $u) seconds = BUDGET
        sc = @be PureOSQP.setup($P, $q, $A, $l, $u; linsys = :dense) seconds = BUDGET
        vt = @be PureOSQP.solve_system!($tl, $ws, $bx, $bz) seconds = BUDGET
        vc = @be PureOSQP.solve_system!($ws.linsys, $ws, $bx, $bz) seconds = BUDGET
        tst, tsc, tvt, tvc = med(st), med(sc), med(vt), med(vc)
        push!(
            rows, (;
                n, blas_threads = nthreads, backend = name,
                tri_setup = tst, cholesky_setup = tsc, setup_ratio = tsc / tst,
                tri_solve = tvt, cholesky_solve = tvc, solve_ratio = tvc / tvt,
            )
        )
        @printf(
            "%6d %-12s | %7.3f ms %7.3f ms %7.2fx | %8.3f µs %8.3f µs %7.2fx\n",
            n, name, 1.0e3tst, 1.0e3tsc, tsc / tst, 1.0e6tvt, 1.0e6tvc, tvc / tvt
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
