# Whether probing an operator's columns is worth what it costs.
#
# An operator that supplies only products cannot be equilibrated by walking entries. Two
# answers: skip equilibration with `scaling = 0`, or recover each column as `op * eⱼ`. The
# choice is a trade and both halves are measured here — what skipping equilibration costs in
# iterations on a badly scaled problem, and what probing costs in time to avoid it.
#
# `equilibrate!` seeds `cost_norms!` once and calls it again per sweep, and calls
# `column_norms!` once per sweep, so probing is `(2·sweeps + 1)·n` products, not `n + m` once.
# The row norms come from the same column pass, so no adjoint products are needed.
using PureOSQP, Krylov, LinearAlgebra, Chairmarks, Printf, JSON, Statistics, Random

BLAS.set_num_threads(1)

# A badly scaled problem needs a high ceiling to converge unequilibrated at all; a run that
# stops at `max_iter` reports the ceiling rather than the cost of skipping equilibration, and
# two such runs agree with each other for no reason.
const OPTS = (linsys = :indirect, eps_abs = 1.0e-6, eps_rel = 1.0e-6, max_iter = 100_000)
const BUDGET = 2

median_time(b) = median(s.time for s in b.samples)

"""
    problem(n, m, spread) -> (P, q, A, l, u)

A QP whose rows and columns span `spread` orders of magnitude, which is what equilibration
exists to fix. `spread = 0` is a well-scaled problem.
"""
function problem(n, m, spread)
    rng = MersenneTwister(2718 + n + round(Int, 10spread))
    S = randn(rng, n, n)
    P = Symmetric(S'S ./ n + 8I)
    A = randn(rng, m, n) ./ sqrt(n)
    # Geometric row and column weights, so no single entry dominates by accident.
    A .*= exp10.(range(-spread / 2, spread / 2; length = m))
    A .*= exp10.(range(-spread / 2, spread / 2; length = n))'
    b = A * randn(rng, n)
    return P, randn(rng, n), A, b .- rand(rng, m), b .+ rand(rng, m)
end

operator(M, T = Float64; kwargs...) = PureOSQP.ProductOperator{T}(M; kwargs...)

println("\nProbed equilibration against skipping it, on badly scaled problems.\n")
@printf(
    "%5s %5s %7s | %7s %-9s %10s | %7s %-9s %10s %9s | %8s\n",
    "n", "m", "spread", "iter", "unscaled", "total", "iter", "probed", "total", "of which", "iter"
)
@printf(
    "%5s %5s %7s | %7s %-9s %10s | %7s %-9s %10s %9s | %8s\n",
    "", "", "", "", "(scaling=0)", "", "", "", "", "setup", "ratio"
)
println("-"^112)

rows = NamedTuple[]
for n in (100, 200), m in (60, 120), spread in (0.0, 4.0, 8.0)
    P, q, A, l, u = problem(n, m, spread)
    Pu = operator(Matrix(P); symmetric = true, posdef = true)
    Au = operator(A)
    Pp = operator(Matrix(P); symmetric = true, posdef = true, probe = true)
    Ap = operator(A; probe = true)

    unscaled = PureOSQP.solve(Pu, q, Au, l, u; OPTS..., scaling = 0)
    probed = PureOSQP.solve(Pp, q, Ap, l, u; OPTS...)
    tu = median_time(@be(PureOSQP.solve(Pu, q, Au, l, u; OPTS..., scaling = 0), seconds = BUDGET))
    tp = median_time(@be(PureOSQP.solve(Pp, q, Ap, l, u; OPTS...), seconds = BUDGET))
    sp = median_time(@be(PureOSQP.setup(Pp, q, Ap, l, u; OPTS...), seconds = BUDGET))

    # A run that stops at the ceiling has not converged, and its iteration count reports
    # `max_iter` rather than a cost. That is the finding on a badly scaled problem, not an
    # error, so the status travels with every row and the ratio is left out when it applies.
    both_converged = unscaled.status === PureOSQP.SOLVED && probed.status === PureOSQP.SOLVED
    push!(
        rows, (;
            n, m, spread, unscaled_iter = unscaled.iter, unscaled_status = string(unscaled.status),
            probed_iter = probed.iter, probed_status = string(probed.status),
            unscaled_total = tu, probed_total = tp, probed_setup = sp,
        )
    )
    @printf(
        "%5d %5d %7.1f | %7d %-9s %8.2f ms | %7d %-9s %8.2f ms %7.2f ms | %8s\n",
        n, m, spread,
        unscaled.iter, unscaled.status === PureOSQP.SOLVED ? "solved" : "no conv", 1.0e3tu,
        probed.iter, probed.status === PureOSQP.SOLVED ? "solved" : "no conv", 1.0e3tp,
        1.0e3sp,
        both_converged ? @sprintf("%.2f", unscaled.iter / probed.iter) : "--"
    )
    flush(stdout)
end

open(joinpath(@__DIR__, "results", "probe_equilibration.json"), "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "blas_threads" => BLAS.get_num_threads(),
            "cases" => [Dict(string(k) => string(v) for (k, v) in pairs(r)) for r in rows],
        ), 2
    )
end
println("\nsaved bench/results/probe_equilibration.json")
