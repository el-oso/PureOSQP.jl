# Setup against iteration, for both solvers, on the OSQP suite.
#
# `bench/osqp_suite.jl` times the whole call. That is the number that matters, but it cannot
# say whether a class is behind because the factorization costs more or because each ADMM
# iteration does. The two solvers take the same number of iterations on these problems, so
# splitting the total at the end of setup gives a per-iteration cost that is directly
# comparable.
using PureOSQP, OSQP, LinearAlgebra, SparseArrays, Chairmarks, LDLFactorizations, Printf

BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "suite_problems.jl"))

const OPTS = (eps_abs = 1.0e-5, eps_rel = 1.0e-5, max_iter = 20_000)
const PURE_ONLY = (check_dualgap = false,)

function osqp_model(P, q, A, l, u)
    model = OSQP.Model()
    OSQP.setup!(
        model; P = sparse(Symmetric(P)), q = collect(q), A = sparse(A),
        l = collect(l), u = collect(u), verbose = false,
        adaptive_rho_interval = 50, check_termination = 25, OPTS...
    )
    return model
end

println("\nSetup and iteration, split. Per-iteration figures are µs.\n")
@printf(
    "%-12s %5s | %8s %8s %8s | %8s %8s %8s | %7s %7s\n",
    "class", "iter", "pu setup", "pu loop", "pu µs/it", "os setup", "os loop", "os µs/it",
    "setup×", "loop×"
)
println("-"^104)
for (name, build) in CASES
    P, q, A, l, u = build()
    sp = PureOSQP.solve(P, q, A, l, u; OPTS..., PURE_ONLY...)
    iter = sp.iter
    # Both `solve!`s consume the object they are handed, so each sample gets a fresh one
    # from the setup expression, which Chairmarks runs untimed.
    ps = (@b PureOSQP.setup(P, q, A, l, u; OPTS..., PURE_ONLY...)).time
    po = (@b PureOSQP.setup(P, q, A, l, u; OPTS..., PURE_ONLY...) PureOSQP.solve!(_)).time
    ls = (@b osqp_model(P, q, A, l, u)).time
    lo = (@b osqp_model(P, q, A, l, u) OSQP.solve!(_)).time
    @printf(
        "%-12s %5d | %6.2fms %6.2fms %8.2f | %6.2fms %6.2fms %8.2f | %6.2fx %6.2fx\n",
        name, iter, 1.0e3ps, 1.0e3po, 1.0e6po / iter, 1.0e3ls, 1.0e3lo, 1.0e6lo / iter,
        ls / ps, lo / po
    )
    flush(stdout)
end
