# The banded factorization against the dense one past the bandwidth the rung accepts.
#
# `bench/gate_crossover_band.jl` measures the selection a caller gets, so it stops where the
# rung stops: at `5b = 4n`. This measures the two backends themselves, across the whole range
# including the part the rung declines, which is what says whether that limit sits in the
# right place.
#
# Both backends read their data from the same workspace — `factorize!` and `solve_system!`
# take the workspace, not their own copy of the problem — so one is built by `setup` and the
# other constructed beside it. Nothing here extends `choose_backend`, so running this file
# leaves selection exactly as it ships.
using PureOSQP, BandedMatrices, LinearAlgebra
using Chairmarks, Printf, JSON, Statistics, Random

BLAS.set_num_threads(1)

const OPTS = (eps_abs = 1.0e-9, eps_rel = 1.0e-9)
const BUDGET = parse(Float64, get(ENV, "GATE_BUDGET", "0.3"))
const SIZES = isempty(ARGS) ? (200, 500, 1000) : Tuple(parse.(Int, ARGS))
const FRACTIONS = (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 0.99)

const Ext = Base.get_extension(PureOSQP, :PureOSQPBandedMatricesExt)

"""
    problem(n, b) -> (P, q, A, l, u)

A box-constrained QP whose reduced matrix has bandwidth exactly `b`: a symmetric
`BandedMatrix` `P` of half-bandwidth `b`, diagonally dominant, with a `Diagonal` `A`.
"""
function problem(n, b)
    rng = MersenneTwister(4441 + 13 * n + b)
    P = BandedMatrix{Float64}(undef, (n, n), (b, b))
    fill!(P.data, 0.0)
    for j in 1:n
        for i in max(1, j - b):(j - 1)
            v = (rand(rng) - 0.5) / (4b)
            P[i, j] = v
            P[j, i] = v
        end
        P[j, j] = 1.0 + rand(rng)
    end
    return P, randn(rng, n), Diagonal(rand(rng, n) .+ 0.5), -rand(rng, n), rand(rng, n)
end

"A `BandedReduced` for this workspace's data, built without asking the rung's permission."
function banded_for(ws, b)
    n = ws.n
    R = BandedMatrix{Float64}(undef, (n, n), (b, b))
    fill!(R.data, 0.0)
    for i in 1:n
        R[i, i] = 1.0
    end
    fact = cholesky(Symmetric(R))
    return Ext.BandedReduced{Float64, typeof(R), typeof(fact)}(R, fact, b)
end

rows = NamedTuple[]
println("\nBanded against dense past the accepted bandwidth. `|` marks where the rung declines.\n")
@printf(
    "%6s %6s %5s %s | %11s %11s %8s | %11s %11s %8s\n",
    "n", "b", "b/n", " ", "band fact", "dense fact", "fact×", "band solve", "dense solve", "solve×"
)
println("-"^108)

for n in SIZES
    for b in sort(unique(clamp.(round.(Int, collect(FRACTIONS) .* n), 2, n - 1)))
        P, q, A, l, u = problem(n, b)
        # The workspace carries the dense backend; the banded one is built beside it and reads
        # the same equilibrated data through the same `ws`.
        ws = PureOSQP.setup(P, q, A, l, u; linsys = :dense, OPTS...)
        bl = banded_for(ws, b)
        PureOSQP.factorize!(bl, ws) || error("n=$n b=$b: banded factorization failed")
        accepted = PureOSQP.backend_name(PureOSQP.setup(P, q, A, l, u; OPTS...).linsys) === :banded
        bx, bz = randn(n), randn(n)
        # Both backends must agree before their times mean anything.
        PureOSQP.solve_system!(bl, ws, bx, bz)
        xb = copy(ws.xtilde)
        PureOSQP.solve_system!(ws.linsys, ws, bx, bz)
        isapprox(xb, ws.xtilde; rtol = 1.0e-8) || error("n=$n b=$b: backends disagree")

        fb = @be PureOSQP.factorize!($bl, $ws) seconds = BUDGET
        fd = @be PureOSQP.factorize!($ws.linsys, $ws) seconds = BUDGET
        sb = @be PureOSQP.solve_system!($bl, $ws, $bx, $bz) seconds = BUDGET
        sd = @be PureOSQP.solve_system!($ws.linsys, $ws, $bx, $bz) seconds = BUDGET
        med(x) = median(s.time for s in x.samples)
        tfb, tfd, tsb, tsd = med(fb), med(fd), med(sb), med(sd)
        push!(
            rows, (;
                n, b, frac = b / n, accepted,
                band_factor = tfb, dense_factor = tfd, factor_ratio = tfd / tfb,
                band_solve = tsb, dense_solve = tsd, solve_ratio = tsd / tsb,
            )
        )
        @printf(
            "%6d %6d %5.2f %s | %8.2f ms %8.2f ms %7.2fx | %8.2f µs %8.2f µs %7.2fx\n",
            n, b, b / n, accepted ? " " : "|", 1.0e3tfb, 1.0e3tfd, tfd / tfb,
            1.0e6tsb, 1.0e6tsd, tsd / tsb
        )
        flush(stdout)
    end
end

open(joinpath(@__DIR__, "results", "gate_band_beyond.json"), "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "blas_threads" => BLAS.get_num_threads(),
            "budget_seconds" => BUDGET,
            "cases" => rows,
        ), 2
    )
end
println("\nsaved bench/results/gate_band_beyond.json")
