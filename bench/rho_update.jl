# What the ρ-only update channel saves, per backend.
#
# `refactor_rho!` runs when `adapt_rho!` moves `ρ` and nothing else; `factorize!` runs when the
# data or the equilibration changed. A backend that does not override the first pays the
# second, so a ratio of 1.00× is the honest report for it, not a missing row.
using PureOSQP, LinearAlgebra, BandedMatrices, Chairmarks, Printf, JSON, Statistics, Random
using LDLFactorizations

BLAS.set_num_threads(1)

const OPTS = (eps_abs = 1.0e-9, eps_rel = 1.0e-9)
const BUDGET = 2

times(x) = [s.time for s in x.samples]

"""
    abba(a, b) -> (ta, tb)

Time `a` and `b` in the order a, b, b, a and return each one's median over the pooled samples
of its two turns, so a drift that is monotonic over the pair lands on both equally rather
than becoming a ratio.
"""
function abba(a, b)
    ta1, tb1 = a(), b()
    tb2, ta2 = b(), a()
    return (median(vcat(times(ta1), times(ta2))), median(vcat(times(tb1), times(tb2))))
end

"""
    cases(n) -> Vector

One `(name, P, A)` per backend that reaches a ρ update, at size `n`.
"""
function cases(n)
    rng = MersenneTwister(90210 + n)
    return [
        (
            "lowrank k=1", Diagonal(rand(rng, n) .+ 0.5),
            PureOSQP.RowCoupled(randn(rng, 1, n) ./ 4, ones(n - 1), collect(1:(n - 1))),
        ),
        (
            "lowrank k=4", Diagonal(rand(rng, n) .+ 0.5),
            PureOSQP.RowCoupled(randn(rng, 4, n) ./ 4, ones(n - 4), collect(1:(n - 4))),
        ),
        (
            "lowrank k=n/10", Diagonal(rand(rng, n) .+ 0.5),
            PureOSQP.RowCoupled(
                randn(rng, n ÷ 10, n) ./ 4, ones(n - n ÷ 10), collect(1:(n - n ÷ 10))
            ),
        ),
        (
            "tridiagonal", SymTridiagonal(rand(rng, n) .+ 3, rand(rng, n - 1) ./ 8),
            Diagonal(rand(rng, n) .+ 0.5),
        ),
        (
            "dense", Symmetric(Matrix(SymTridiagonal(rand(rng, n) .+ 4, rand(rng, n - 1) ./ 8))),
            Matrix(Diagonal(rand(rng, n) .+ 0.5)),
        ),
    ]
end

println("\nThe ρ-only update channel against a full rebuild.\n")
@printf(
    "%-16s %6s %-13s | %12s %12s %8s\n",
    "case", "n", "backend", "refactor_rho!", "factorize!", "saved"
)
println("-"^76)

rows = NamedTuple[]
for n in (500, 1000, 2000)
    for (name, P, A) in cases(n)
        qv, lb, ub = randn(n), -rand(n), rand(n)
        ws = PureOSQP.setup(P, qv, A, lb, ub; OPTS...)
        backend = PureOSQP.backend_name(ws.linsys)
        # Both paths must leave the same factorization behind, or the cheap one is not an
        # update of the expensive one.
        PureOSQP.refactor_rho!(ws.linsys, ws) || error("$name at n=$n: rho update failed")
        rhs = copy(ws.rhs_x)
        PureOSQP.solve_system!(ws.linsys, ws, rhs, ws.rhs_z)
        xr = copy(ws.xtilde)
        PureOSQP.factorize!(ws.linsys, ws) || error("$name at n=$n: rebuild failed")
        PureOSQP.solve_system!(ws.linsys, ws, rhs, ws.rhs_z)
        isapprox(xr, ws.xtilde; rtol = 1.0e-12) ||
            error("$name at n=$n: the two paths disagree")
        tr, tf = abba(
            () -> @be(PureOSQP.refactor_rho!(ws.linsys, ws), seconds = BUDGET),
            () -> @be(PureOSQP.factorize!(ws.linsys, ws), seconds = BUDGET),
        )
        push!(rows, (; name, n, backend = string(backend), rho = tr, full = tf))
        @printf(
            "%-16s %6d %-13s | %9.1f us %9.1f us %7.2fx\n",
            name, n, backend, 1.0e6tr, 1.0e6tf, tf / tr
        )
        flush(stdout)
    end
end

open(joinpath(@__DIR__, "results", "rho_update.json"), "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "blas_threads" => BLAS.get_num_threads(),
            "cases" => [Dict(string(k) => string(v) for (k, v) in pairs(r)) for r in rows],
        ), 2
    )
end
println("\nsaved bench/results/rho_update.json")
