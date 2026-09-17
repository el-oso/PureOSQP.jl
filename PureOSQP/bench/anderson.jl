# Anderson acceleration against the plain iteration, through the solver's own interface.
#
# ADMM is a fixed-point iteration on `w = [x; ρ⁻¹ ⊙ y + z]`. Anderson extrapolates from a
# window of past iterates instead of stepping straight to the next one, reaching the same
# tolerance in fewer steps at the cost of a small least-squares solve per step.
#
# An extrapolated point can be worse than the plain one, so one whose residual exceeds
# `safeguard_tol` times the plain step's is discarded. `safeguard_tol = Inf` keeps every
# proposal, which is what the third column measures.
#
# Run:  julia --project=bench PureOSQP/bench/anderson.jl
using PureOSQP, COSMOAccelerators, LinearAlgebra, SparseArrays, Random, Printf, JSON
using Chairmarks, Statistics

BLAS.set_num_threads(1)

const RESULTS = joinpath(@__DIR__, "results", "anderson.json")
const OPT = (eps_abs = 1.0e-8, eps_rel = 1.0e-8, max_iter = 20_000)

med(x) = median(s.time for s in x.samples)

plain(P, q, A, l, u) = PureOSQP.solve(P, q, A, l, u; OPT...)

function accelerated(P, q, A, l, u; tol)
    n, m = length(q), length(l)
    return PureOSQP.solve(
        P, q, A, l, u;
        accelerator = PureOSQP.anderson(Float64, n + m; safeguard_tol = tol), OPT...
    )
end

cases = Any[]
problems = Pair{String, Any}[]

Random.seed!(1)
let n = 100, m = 200
    X = randn(n, n)
    A = randn(m, n)
    b = A * randn(n)
    push!(problems, "dense QP" => (Matrix(Symmetric(X'X / n + I)), randn(n), A, b .- 1, b .+ 1))
end
let n = 300, m = 600
    S = sprandn(n, n, 0.02)
    A = Matrix(sprandn(m, n, 0.02))
    b = A * randn(n)
    push!(problems, "sparse-ish" => (Matrix(Symmetric(S * S')) + 2I, randn(n), A, b .- 1, b .+ 1))
end
let n = 60, m = 120
    X = randn(n, n)
    A = randn(m, n)
    b = A * randn(n)
    push!(problems, "equality-heavy" => (Matrix(Symmetric(X'X / n + I)), randn(n), A, copy(b), b))
end
# Conditioning is where the extrapolation's least-squares solve is hardest, and where the
# safeguard is most likely to earn its place.
let n = 120, m = 240
    U, _ = qr(randn(n, n))
    P = Matrix(Symmetric(U * Diagonal(exp10.(range(0, 8; length = n))) * U'))
    A = randn(m, n)
    b = A * randn(n)
    push!(problems, "ill-conditioned" => (P, randn(n), A, b .- 1, b .+ 1))
end

println("\nSame tolerance, plain against accelerated. Setup is inside the timing.\n")
@printf(
    "%-16s %7s %7s %7s %9s %9s %7s %s\n",
    "problem", "plain", "accel", "iters", "plain ms", "accel ms", "time", "status"
)
println("-"^84)
for (name, p) in problems
    P, q, A, l, u = p
    sp = plain(P, q, A, l, u)
    sa = accelerated(P, q, A, l, u; tol = 2.0)
    tp = med(@be plain($P, $q, $A, $l, $u) seconds = 5)
    ta = med(@be accelerated($P, $q, $A, $l, $u; tol = 2.0) seconds = 5)
    st = sp.status === sa.status ? String(Symbol(sp.status)) : "$(sp.status)/$(sa.status)"
    @printf(
        "%-16s %7d %7d %6.2fx %9.2f %9.2f %6.2fx %s\n",
        name, sp.iter, sa.iter, sp.iter / sa.iter, 1.0e3tp, 1.0e3ta, tp / ta, st
    )
    push!(
        cases, (;
            case = name, plain_iters = sp.iter, accel_iters = sa.iter,
            plain_seconds = tp, accel_seconds = ta, status = st,
            agreement = maximum(abs, sp.x .- sa.x),
        )
    )
end

open(RESULTS, "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION), "blas_threads" => BLAS.get_num_threads(),
            "eps" => OPT.eps_abs, "cases" => cases,
        ), 2
    )
end
println("\nwrote $RESULTS")
