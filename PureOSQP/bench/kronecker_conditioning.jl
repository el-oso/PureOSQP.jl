# The Kronecker tier on ill-conditioned problems, which is where its one restriction bites.
#
# The rung requires `scaling = 0`: `c·μ·D²` is diagonal but not scalar, so any equilibration
# breaks the diagonalization the backend is built on. Equilibration is exactly what an
# ill-conditioned problem wants, so the tier trades it away — and the question this measures is
# what that costs and when the structure pays for it anyway.
#
# Two sweeps. The first holds the size fixed and raises `κ`, against a dense path given the
# same `scaling = 0` so the comparison is the backends'. The second compares against a dense
# path allowed its equilibration, which is what a caller actually chooses between.
#
# `κ(A₁ ⊗ A₂) = κ(A₁)·κ(A₂)`, so the factors carry the square root of the figure reported.
using PureOSQP, LinearAlgebra, Chairmarks, Printf, JSON, Statistics, Random

BLAS.set_num_threads(1)

const TOL = (eps_abs = 1.0e-8, eps_rel = 1.0e-8)
const BUDGET = 2

median_time(b) = median(s.time for s in b.samples)

"A `k×k` matrix with condition number `kappa`, from a random orthogonal pair."
function illconditioned(rng, k, kappa)
    U = qr(randn(rng, k, k)).Q
    V = qr(randn(rng, k, k)).Q
    return Matrix(U * Diagonal(exp10.(range(0, -log10(kappa); length = k))) * V')
end

"""
    problem(rng, n1, n2, kappa) -> (K, dense, P, q, l, u)

A box-constrained QP whose `A` is `A₁ ⊗ A₂` with each factor at condition number `kappa`, and
whose `P` is the identity — the scalar cost the tier requires.
"""
function problem(rng, n1, n2, kappa)
    A1 = illconditioned(rng, n1, kappa)
    A2 = illconditioned(rng, n2, kappa)
    K = PureOSQP.KroneckerOperator(A1, A2)
    dense = kron(A1, A2)
    n = n1 * n2
    b = dense * randn(rng, n)
    return K, dense, Diagonal(fill(1.0, n)), randn(rng, n), b .- rand(rng, n), b .+ rand(rng, n)
end

rows = NamedTuple[]

println("\nConditioning sweep at a fixed size, both paths unscaled: the backends' comparison.\n")
@printf("%10s %-10s | %-9s %6s | %-9s %6s | %8s\n", "κ(A)", "backend", "kron", "iter", "dense", "iter", "agree")
println("-"^74)
for kappa in (1.0e1, 1.0e2, 1.0e4, 1.0e6, 1.0e8)
    rng = MersenneTwister(90)
    K, dense, P, q, l, u = problem(rng, 6, 5, kappa)
    ws = PureOSQP.setup(P, q, K, l, u; scaling = 0, TOL...)
    kr = PureOSQP.solve(P, q, K, l, u; scaling = 0, TOL...)
    dn = PureOSQP.solve(P, q, dense, l, u; scaling = 0, TOL...)
    agree = isapprox(kr.x, dn.x; rtol = 1.0e-5)
    push!(
        rows, (;
            sweep = "conditioning", n = size(dense, 2), kappa = cond(dense),
            backend = string(PureOSQP.backend_name(ws.linsys)),
            kron_status = string(kr.status), kron_iter = kr.iter,
            dense_status = string(dn.status), dense_iter = dn.iter, agree,
        )
    )
    @printf(
        "%10.2e %-10s | %-9s %6d | %-9s %6d | %8s\n", cond(dense),
        PureOSQP.backend_name(ws.linsys), kr.status, kr.iter, dn.status, dn.iter, agree
    )
    flush(stdout)
end

println("\nAgainst a dense path allowed its equilibration, which is the real choice.\n")
@printf(
    "%6s %10s | %-9s %6s %10s | %-9s %6s %10s | %8s\n",
    "n", "κ(A)", "kron", "iter", "time", "dense", "iter", "time", "kron×"
)
println("-"^92)
for (n1, n2) in ((6, 5), (14, 12), (24, 20))
    rng = MersenneTwister(91)
    K, dense, P, q, l, u = problem(rng, n1, n2, 1.0e6)
    kr = PureOSQP.solve(P, q, K, l, u; scaling = 0, TOL...)
    de = PureOSQP.solve(P, q, dense, l, u; TOL...)
    tk = median_time(@be(PureOSQP.solve(P, q, K, l, u; scaling = 0, TOL...), seconds = BUDGET))
    td = median_time(@be(PureOSQP.solve(P, q, dense, l, u; TOL...), seconds = BUDGET))
    push!(
        rows, (;
            sweep = "against_equilibrated", n = n1 * n2, kappa = cond(dense),
            backend = "kronecker", kron_status = string(kr.status), kron_iter = kr.iter,
            dense_status = string(de.status), dense_iter = de.iter,
            kron_seconds = tk, dense_seconds = td,
        )
    )
    @printf(
        "%6d %10.2e | %-9s %6d %7.2f ms | %-9s %6d %7.2f ms | %7.2fx\n",
        n1 * n2, cond(dense), kr.status, kr.iter, 1.0e3tk, de.status, de.iter, 1.0e3td, td / tk
    )
    flush(stdout)
end

open(joinpath(@__DIR__, "results", "kronecker_conditioning.json"), "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "blas_threads" => BLAS.get_num_threads(),
            "cases" => [Dict(string(k) => string(v) for (k, v) in pairs(r)) for r in rows],
        ), 2
    )
end
println("\nsaved bench/results/kronecker_conditioning.json")
