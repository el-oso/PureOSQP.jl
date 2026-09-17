# Setup against iteration, for both solvers, on the OSQP suite.
#
# `bench/osqp_suite.jl` times the whole call. That is the number that matters, but it cannot
# say whether a class is behind because the factorization costs more or because each ADMM
# iteration does. The two solvers take the same number of iterations on these problems, so
# splitting the total at the end of setup gives a per-iteration cost that is directly
# comparable.
using PureOSQP, LinearAlgebra, SparseArrays, Chairmarks, LDLFactorizations, Printf
using Statistics

BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "..", "..", "bench", "suite_problems.jl"))
include(joinpath(@__DIR__, "osqp_v1.jl"))

# `check_dualgap` is off on both, as in `bench/osqp_suite.jl`: it is the one termination test
# the two compute at different points, and leaving it on makes the iteration counts
# incomparable, which is what splitting the total at the end of setup relies on.
const OPTS = (eps_abs = 1.0e-5, eps_rel = 1.0e-5, max_iter = 20_000, check_dualgap = false)

# Seconds given to each measurement.
const BUDGET = 5

"""
    abba(a, b) -> (ta, tb)

Time `a` and `b` in the order a, b, b, a, and return each one's median, so that a drift that
is monotonic over the pair lands on both equally rather than becoming a ratio. `a` and `b` are
thunks returning a Chairmarks benchmark; the samples of each one's two turns are pooled before
the median is taken.
"""
function abba(a, b)
    ta1, tb1 = a(), b()
    tb2, ta2 = b(), a()
    return (pooled_median(ta1, ta2), pooled_median(tb1, tb2))
end

pooled_median(x, y) = median(s.time for s in Iterators.flatten((x.samples, y.samples)))

# libosqp's setup is timed from CSC it already holds, as a C caller would.
osqp_model(data, q, l, u) = setup_v1(
    data, q, l, u;
    verbose = false, adaptive_rho_interval = 50, check_termination = 25, OPTS...
)

println("\nSetup and iteration, split. Per-iteration figures are µs.\n")
@printf(
    "%-12s %5s | %8s %8s %8s | %8s %8s %8s | %7s %7s\n",
    "class", "iter", "pu setup", "pu loop", "pu µs/it", "os setup", "os loop", "os µs/it",
    "setup×", "loop×"
)
println("-"^104)
for (name, build) in CASES
    P, q, A, l, u = build()
    sp = PureOSQP.solve(P, q, A, l, u; OPTS...)
    iter = sp.iter
    # Both `solve!`s consume the object they are handed, so each sample gets a fresh one
    # from the setup expression, which Chairmarks runs untimed. libosqp's solver holds C
    # memory that Julia's collector cannot see, so every sample that builds one frees it in
    # the teardown slot, which Chairmarks also leaves untimed.
    # `0` is a dummy setup value: Chairmarks reads `nothing` in that slot as "no setup" and
    # then calls the benchmarked function with no arguments, leaving nothing for the
    # teardown to free.
    data = CSCData(sparse(P), sparse(A))
    build_osqp = _ -> osqp_model(data, q, l, u)
    solve_osqp = m -> (solve_v1!(m); m)
    ps, ls = abba(
        () -> @be(PureOSQP.setup(P, q, A, l, u; OPTS...), seconds = BUDGET),
        () -> @be(0, build_osqp, cleanup!, seconds = BUDGET),
    )
    po, lo = abba(
        () -> @be(
            PureOSQP.setup(P, q, A, l, u; OPTS...),
            PureOSQP.solve!(_), seconds = BUDGET
        ),
        () -> @be(osqp_model(data, q, l, u), solve_osqp, cleanup!, seconds = BUDGET),
    )
    @printf(
        "%-12s %5d | %6.2fms %6.2fms %8.2f | %6.2fms %6.2fms %8.2f | %6.2fx %6.2fx\n",
        name, iter, 1.0e3ps, 1.0e3po, 1.0e6po / iter, 1.0e3ls, 1.0e3lo, 1.0e6lo / iter,
        ls / ps, lo / po
    )
    flush(stdout)
end
