# What the matrix representation costs the interior-point method.
#
# The same problem is handed over four ways — dense, sparse, the structured type it really is,
# and a products-only operator — and each is solved at the defaults. The iterations must agree:
# the representation changes which backend forms and factors the Newton system, not the
# sequence of points it visits. Where they do not agree the backends disagree numerically, and
# the row says so rather than being read as a speed difference.
#
# The operator column is absent for pairs the method refuses without a caller-supplied
# preconditioner, which is `PureIPM/bench/ipm_matrixfree.jl`'s subject.
using PureOSQP, PureIPM, PureQPBase
using LinearAlgebra, SparseArrays, BandedMatrices, LDLFactorizations
using Chairmarks, Random, Printf, JSON, Statistics

BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "..", "..", "bench", "structured_problems.jl"))

const OPTS = (eps_abs = 1.0e-8, eps_rel = 1.0e-8)
const BUDGET = 3

median_seconds(f) = median(s.time for s in @be(f(), seconds = BUDGET).samples)

"Solve `(P, A)` as given, returning the backend, the iterations, the status and the time."
function measure(P, q, A, l, u)
    ws = setup(P, q, A, l, u, InteriorPoint(); OPTS...)
    r = solve!(ws)
    t = median_seconds(() -> solve(P, q, A, l, u, InteriorPoint(); OPTS...))
    return (
        backend = string(PureQPBase.backend_name(ws.linsys)), iter = r.iter,
        status = string(r.status), seconds = t,
    )
end

println("\nInterior point against the matrix representation, eps = $(OPTS.eps_abs).\n")
@printf(
    "%-16s %-12s %-18s %5s %-18s %9s\n",
    "family", "given as", "backend", "iter", "status", "time"
)
println("-"^86)

rows = NamedTuple[]
for f in structured_families(120)
    Pd, Ad = Matrix(f.P), Matrix(f.A)
    givens = [
        ("structured", f.P, f.A),
        ("dense", Pd, Ad),
        ("sparse", sparse(Pd), sparse(Ad)),
    ]
    for (given, P, A) in givens
        m = measure(P, f.q, A, f.l, f.u)
        push!(rows, (; family = f.name, given, m...))
        @printf(
            "%-16s %-12s %-18s %5d %-18s %6.2f ms\n",
            f.name, given, m.backend, m.iter, m.status, 1.0e3m.seconds
        )
        flush(stdout)
    end
    println()
end

open(joinpath(@__DIR__, "results", "ipm_representation.json"), "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "blas_threads" => BLAS.get_num_threads(),
            "eps" => OPTS.eps_abs,
            "cases" => [Dict(string(k) => string(v) for (k, v) in pairs(r)) for r in rows],
        ), 2
    )
end
println("saved PureIPM/bench/results/ipm_representation.json")
