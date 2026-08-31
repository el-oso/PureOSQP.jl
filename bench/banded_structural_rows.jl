# What equilibration costs on a `BandedMatrix`, as a function of `n` at a fixed bandwidth.
#
# `setup` at the default `scaling = 10` walks every column of `P` and `A` ten times through
# `structural_rows`. A representation without a method for it reports every row of the
# column, so a matrix holding `O(nb)` entries is visited `O(n^2)` times per sweep. Each row
# below times `setup` twice on the same problem -- once scaled, once at `scaling = 0` -- so
# the equilibration share separates from the rest of setup by subtraction.
#
# Swept at two BLAS thread counts. `src/` and `ext/` pin nothing, so the deployed solver runs
# at whatever the caller has, and a column walk timed against a threaded competitor answers a
# different question than one timed alone.
#
# Each row is labelled by the `structural_rows` method a banded column actually dispatches to,
# so running the file before and after the extension defines one leaves both states in the
# results file and the ratio is read from a single document.
using PureOSQP, BandedMatrices, LinearAlgebra, Chairmarks, Printf, JSON, Statistics, Random

"Which `structural_rows` method a banded column reaches: the generic every-row fallback in
PureOSQP, or a bandwidth-derived one from the extension."
function dispatch_label()
    m = which(PureOSQP.structural_rows, (BandedMatrix{Float64, Matrix{Float64}}, Int))
    return parentmodule(m) === PureOSQP ? "generic" : "banded"
end

const LABEL = dispatch_label()
const SIZES = (250, 500, 1000, 2000)
const BANDWIDTH = 8
const THREADS = (1, min(8, Sys.CPU_THREADS))
const RESULTS = joinpath(@__DIR__, "results", "banded_structural_rows.json")

times(x) = [s.time for s in x.samples]

"""
    problem(n, b) -> (P, q, A, l, u)

A box-constrained QP whose `P` is a `BandedMatrix` of bandwidth `b` and whose `A` is
`Diagonal`, so `reduced_bandwidth` is `b` and the banded rung takes it for every `n` here.
"""
function problem(n, b)
    rng = MersenneTwister(6151 + 3 * n)
    P = BandedMatrix{Float64}(undef, (n, n), (b, b))
    fill!(P.data, 0.0)
    P[band(0)] .= rand(rng, n) .+ 2b
    for k in 1:b
        P[band(k)] .= rand(rng, n - k) ./ (4b)
        P[band(-k)] .= P[band(k)]
    end
    A = Diagonal(rand(rng, n) .+ 0.5)
    return P, randn(rng, n), A, -rand(rng, n), rand(rng, n)
end

"The median of `f`'s samples, warmed once so no compilation lands in the timings."
function median_time(f)
    f()
    return median(times(@be f() seconds = 1.0))
end

"`log2` of the ratio between consecutive sizes: 1 for linear growth, 2 for quadratic."
function exponents(sizes, ts)
    return [log2(ts[i + 1] / ts[i]) for i in 1:(length(ts) - 1)]
end

function main()
    rows = Dict{String, Any}[]
    for nthreads in THREADS
        BLAS.set_num_threads(nthreads)
        scaled, unscaled = Float64[], Float64[]
        for n in SIZES
            P, q, A, l, u = problem(n, BANDWIDTH)
            ws = setup(P, q, A, l, u; scaling = 10)
            name = String(PureOSQP.backend_name(ws.linsys))
            ts = median_time(() -> setup(P, q, A, l, u; scaling = 10))
            t0 = median_time(() -> setup(P, q, A, l, u; scaling = 0))
            push!(scaled, ts)
            push!(unscaled, t0)
            push!(
                rows, Dict{String, Any}(
                    "label" => LABEL, "blas_threads" => nthreads, "n" => n,
                    "bandwidth" => BANDWIDTH, "backend" => name,
                    "setup_scaled_s" => ts, "setup_unscaled_s" => t0,
                    "equilibration_s" => ts - t0,
                )
            )
            @printf(
                "%-8s threads=%d n=%5d %-10s scaled %8.3f ms  unscaled %8.3f ms  equil %8.3f ms\n",
                LABEL, nthreads, n, name, 1.0e3ts, 1.0e3t0, 1.0e3 * (ts - t0)
            )
            flush(stdout)
        end
        @printf(
            "%-8s threads=%d growth exponent per doubling: scaled %s  equilibration %s\n",
            LABEL, nthreads,
            string(round.(exponents(SIZES, scaled); digits = 2)),
            string(round.(exponents(SIZES, scaled .- unscaled); digits = 2))
        )
        flush(stdout)
    end
    old = isfile(RESULTS) ? JSON.parsefile(RESULTS)["cases"] : Any[]
    kept = filter(r -> r["label"] != LABEL, old)
    open(RESULTS, "w") do io
        JSON.print(
            io, Dict(
                "julia_version" => string(VERSION),
                "cpu_threads" => Sys.CPU_THREADS,
                "sizes" => collect(SIZES),
                "cases" => vcat(kept, rows),
            ), 2
        )
    end
    return println("wrote $RESULTS")
end

main()
