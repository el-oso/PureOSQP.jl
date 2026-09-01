# The primal-dual integral under two quadrature rules, and what measuring it costs.
#
# `∫|gap| dt` is sampled wherever the solver refreshes its residuals, so the nodes are
# whatever times the iterations landed on. Two rules run over those same samples:
#
#   trapezoid   joins consecutive samples with a straight line
#   log-mean    joins them with an exponential and integrates that exactly, which is the
#               interval times the logarithmic mean of the endpoints
#
# The second is the better model for a geometrically decaying gap, and the logarithmic mean
# never exceeds the arithmetic one, so the trapezoid is an upper bound and the pair brackets
# the truth. How far apart they sit is the question this file answers, because it decides
# whether the choice of rule matters when comparing two solvers.
#
# Interpolatory rules at chosen nodes -- Clenshaw-Curtis, Gauss -- do not apply here. They
# evaluate the integrand at abscissae the rule picks, and nothing can make an ADMM iteration
# land at a prescribed time. Their accuracy also assumes a smooth integrand, where this one
# has kinks wherever `ρ` retunes.
#
# Run:  julia --project=bench bench/primdual_integral.jl
using PureOSQP, LinearAlgebra, Random, Printf, JSON, Statistics

BLAS.set_num_threads(1)

const RESULTS = joinpath(@__DIR__, "results", "primdual_integral.json")
const OPTS = (eps_abs = 1.0e-9, eps_rel = 1.0e-9, max_iter = 50_000)

"A QP whose `P` has condition number `kappa`, under a box."
function problem(n, kappa)
    rng = MersenneTwister(7)
    Q = qr(randn(rng, n, n)).Q
    d = exp10.(range(0, log10(kappa); length = n))
    P = Matrix(Symmetric(Matrix(Q * Diagonal(d) * Q')))
    return P, randn(rng, n), Matrix(1.0I, n, n), fill(-0.5, n), fill(0.5, n)
end

const CASES = [
    ("well conditioned", 40, 1.0e2),
    ("moderate", 40, 1.0e6),
    ("ill conditioned", 40, 1.0e9),
    ("larger", 150, 1.0e4),
]

rows = NamedTuple[]
@printf(
    "%-20s %7s %13s %13s %9s %10s\n",
    "problem", "iters", "trapezoid", "log-mean", "log/trap", "overhead"
)
println("-"^80)
for (name, n, kappa) in CASES
    P, q, A, l, u = problem(n, kappa)
    PureOSQP.solve(P, q, A, l, u; OPTS...)
    PureOSQP.solve(P, q, A, l, u; OPTS..., profile_primdual = true)

    off = PureOSQP.solve(P, q, A, l, u; OPTS...)
    on = PureOSQP.solve(P, q, A, l, u; OPTS..., profile_primdual = true)
    # Measuring must not change what is measured.
    on.iter == off.iter || error("$name: profiling changed the iteration count")
    isapprox(on.obj_val, off.obj_val; rtol = 1.0e-12) ||
        error("$name: profiling changed the objective")
    on.primdual_int_log <= on.primdual_int ||
        error("$name: the log-mean rule exceeded the trapezoid")

    t_off = minimum(@elapsed(PureOSQP.solve(P, q, A, l, u; OPTS...)) for _ in 1:5)
    t_on = minimum(
        @elapsed(PureOSQP.solve(P, q, A, l, u; OPTS..., profile_primdual = true)) for _ in 1:5
    )
    overhead = 100 * (t_on / t_off - 1)
    push!(
        rows, (;
            case = name, n, kappa, iter = on.iter, status = string(on.status),
            trapezoid = on.primdual_int, logmean = on.primdual_int_log,
            ratio = on.primdual_int_log / on.primdual_int,
            seconds_off = t_off, seconds_on = t_on, overhead_percent = overhead,
        )
    )
    @printf(
        "%-20s %7d %13.5g %13.5g %9.4f %9.2f%%\n",
        name, on.iter, on.primdual_int, on.primdual_int_log,
        on.primdual_int_log / on.primdual_int, overhead
    )
    flush(stdout)
end

println()
println("The rules disagree by more than the overhead of computing either, so which one is")
println("used is part of what the number means. Neither is reproducible across machines:")
println("both integrate against wall-clock time.")

# What the two rules disagree about is the curve between samples, so sampling the same
# problem more densely has to bring them together. How near 1 the ratio gets is therefore a
# reading on whether the samples resolve the gap at all.
#
# `check_termination` is the only handle on the sampling interval, and it also decides where
# the solve stops, so the iteration count moves down the column too. The trend is not a
# single-variable experiment; it is reported as the trend it is.
println("\nThe same problem sampled more densely, which is what the ratio measures.\n")
@printf("%-14s %7s %13s %13s %9s\n", "sample every", "iters", "trapezoid", "log-mean", "log/trap")
println("-"^62)
sampling = NamedTuple[]
let (P, q, A, l, u) = problem(40, 1.0e2)
    for ct in (25, 10, 5, 2, 1)
        o = (OPTS..., check_termination = ct, profile_primdual = true)
        PureOSQP.solve(P, q, A, l, u; o...)
        r = PureOSQP.solve(P, q, A, l, u; o...)
        push!(
            sampling, (;
                check_termination = ct, iter = r.iter, trapezoid = r.primdual_int,
                logmean = r.primdual_int_log, ratio = r.primdual_int_log / r.primdual_int,
            )
        )
        @printf(
            "%-14d %7d %13.5g %13.5g %9.4f\n",
            ct, r.iter, r.primdual_int, r.primdual_int_log, r.primdual_int_log / r.primdual_int
        )
        flush(stdout)
    end
end

open(RESULTS, "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "blas_threads" => BLAS.get_num_threads(),
            "eps" => OPTS.eps_abs, "max_iter" => OPTS.max_iter,
            "cases" => rows, "sampling" => sampling,
        ), 2
    )
end
println("\nwrote $RESULTS")
