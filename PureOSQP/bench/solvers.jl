# PureOSQP against three other solvers on dense QPs, one per algorithm family:
#
#   libosqp   operator splitting, sparse linear algebra  (the reference implementation)
#   DAQP      dense active set                            (built for exactly this shape)
#   Clarabel  interior point
#
# DAQP is the one that matters most here. Every other comparison in this repository puts a
# sparse solver on dense data, which is its worst case; DAQP is designed for dense QPs, so
# it is the honest question "is a dense ADMM solver competitive with a dense active-set
# one" rather than "does dense beat sparse on dense data".
using PureOSQP, DAQP, Clarabel
using LinearAlgebra, SparseArrays, BenchmarkTools, Random, Printf, JSON

include(joinpath(@__DIR__, "osqp_v1.jl"))

BLAS.set_num_threads(1)

const TOL = 1.0e-6

function make_problem(n, m; seed = 0)
    Random.seed!(seed)
    X = randn(n, n)
    P = Matrix(X'X / n + I)
    q = randn(n)
    A = randn(m, n)
    Ax = A * randn(n)
    return (P, q, A, Ax .- rand(m), Ax .+ rand(m))
end

pure(P, q, A, l, u) = PureOSQP.solve(
    P, q, A, l, u;
    # `check_dualgap` is off on both operator-splitting solvers. Left on, the gap test can
    # only add iterations, and the two compute the gap at different points, which would make
    # this a comparison of stopping rules dressed up as a comparison of solvers.
    eps_abs = TOL, eps_rel = TOL, max_iter = 20_000, check_dualgap = false
).x

osqp(data, q, l, u) = solve_v1(
    data, q, l, u;
    verbose = false, eps_abs = TOL, eps_rel = TOL, max_iter = 20_000,
    adaptive_rho_interval = 50, check_termination = 25, check_dualgap = false
).x

daqp(P, q, A, l, u) = DAQP.quadprog(P, q, A, u, l, zeros(Cint, length(l)))[1]

function clarabel(P, q, A, b, m)
    settings = Clarabel.Settings(verbose = false, tol_gap_abs = TOL, tol_gap_rel = TOL)
    solver = Clarabel.Solver()
    Clarabel.setup!(solver, P, q, A, b, [Clarabel.NonnegativeConeT(2m)], settings)
    return Clarabel.solve!(solver).x
end

# Each solver is timed on the input form it reads: libosqp and Clarabel take sparse matrices, so
# the conversion from the dense problem is built once beforehand and not charged to them.
# Clarabel takes one-sided cones, so the two-sided rows are stacked as Ax ≤ u, -Ax ≤ -l.
const SOLVERS = [
    "PureOSQP" => (identity, pure),
    "OSQP" => (((P, q, A, l, u),) -> (CSCData(P, A), q, l, u), osqp),
    "DAQP" => (identity, daqp),
    "Clarabel" => (
        ((P, q, A, l, u),) -> (sparse(triu(P)), q, sparse([A; -A]), [u; -l], length(l)),
        clarabel,
    ),
]
const CASES = [(10, 20), (25, 50), (50, 100), (100, 200), (200, 400), (100, 50)]

function run_cases()
    results = NamedTuple[]
    @printf(
        "%5s %6s | %10s %10s %10s %10s | %9s\n",
        "n", "m", "PureOSQP", "OSQP", "DAQP", "Clarabel", "max |Δx|"
    )
    println("-"^74)
    for (n, m) in CASES
        P, q, A, l, u = make_problem(n, m; seed = n + m)
        inputs = Dict(name => prep((P, q, A, l, u)) for (name, (prep, _)) in SOLVERS)
        xs = Dict(name => f(inputs[name]...) for (name, (_, f)) in SOLVERS)
        ref = xs["PureOSQP"]
        dx = maximum(name -> maximum(abs, xs[name] .- ref), first.(SOLVERS))
        ts = Dict{String, Float64}()
        for (name, (_, f)) in SOLVERS
            args = inputs[name]
            ts[name] = @belapsed $f($args...)
        end
        push!(results, (; n, m, times = ts, max_dx = dx))
        @printf(
            "%5d %6d | %8.3f ms %8.3f ms %8.3f ms %8.3f ms | %9.1e\n",
            n, m, 1.0e3ts["PureOSQP"], 1.0e3ts["OSQP"], 1.0e3ts["DAQP"], 1.0e3ts["Clarabel"], dx
        )
        flush(stdout)
    end
    return results
end

results = run_cases()

open(joinpath(@__DIR__, "results", "solvers.json"), "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "blas_threads" => BLAS.get_num_threads(),
            "tol" => TOL,
            "versions" => Dict(
                "libosqp" => unsafe_string(
                    ccall((:osqp_version, osqp_builtin_double), Cstring, ())
                ),
                "DAQP" => string(pkgversion(DAQP)),
                "Clarabel" => string(pkgversion(Clarabel)),
            ),
            "results" => [
                Dict("n" => r.n, "m" => r.m, "max_dx" => r.max_dx, "times" => r.times)
                    for r in results
            ],
        ), 2
    )
end
println("\nsaved bench/results/solvers.json")
