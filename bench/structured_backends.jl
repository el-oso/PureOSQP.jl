# What structure in `P` and `A` is worth, against the dense path the same problem takes
# without it.
#
# `bandwidth(R) = max(bandwidth(P), 2 bandwidth(A))` for `R = c D P D + σI + Ãᵀ diag(ρ) Ã`,
# because diagonal scaling preserves a bandwidth and squaring `A` doubles it. Each row below
# is one problem solved twice — once in a structured type and once as a `Matrix` — so the
# iteration counts match exactly and the difference is the backend.
using PureOSQP, BandedMatrices, LinearAlgebra, Chairmarks, Printf, JSON, Statistics

BLAS.set_num_threads(1)

const OPTS = (eps_abs = 1.0e-9, eps_rel = 1.0e-9)
const BUDGET = 2

"""
    abba(a, b) -> (ta, tb)

Time `a` and `b` in the order a, b, b, a and return each one's median over the pooled
samples of its two turns, so a drift that is monotonic over the pair lands on both equally
rather than becoming a ratio.
"""
function abba(a, b)
    ta1, tb1 = a(), b()
    tb2, ta2 = b(), a()
    return (pooled_median(ta1, ta2), pooled_median(tb1, tb2))
end

pooled_median(x, y) = median(s.time for s in Iterators.flatten((x.samples, y.samples)))

"""
    families(n) -> Vector

One problem per reachable bandwidth of the reduced matrix, at size `n`.
"""
function families(n)
    return [
        (
            "Diagonal, Diagonal", 0,
            Diagonal(rand(n) .+ 0.5), Diagonal(rand(n) .+ 0.5),
        ),
        (
            "SymTridiagonal, Diagonal", 1,
            SymTridiagonal(rand(n) .+ 3, rand(n - 1) ./ 8), Diagonal(rand(n) .+ 0.5),
        ),
        (
            "SymTridiagonal, Tridiagonal", 2,
            SymTridiagonal(rand(n) .+ 4, rand(n - 1) ./ 8),
            Tridiagonal(rand(n - 1) ./ 4, rand(n) .+ 1, rand(n - 1) ./ 4),
        ),
    ]
end

println("\nStructured backends against the dense path, on the same problem.\n")
@printf(
    "%-28s %6s %5s %-13s | %10s %10s %8s | %10s %10s %8s | %6s\n",
    "P, A", "n", "bw(R)", "backend", "setup", "dense setup", "setup×",
    "total", "dense total", "total×", "iter"
)
println("-"^136)

rows = NamedTuple[]
for n in (100, 400, 1000, 2000)
    for (name, bw, P, A) in families(n)
        q, l, u = randn(n), -rand(n), rand(n)
        Pm, Am = Matrix(P), Matrix(A)
        ws = PureOSQP.setup(P, q, A, l, u; OPTS...)
        backend = PureOSQP.backend_name(ws.linsys)
        structured = PureOSQP.solve(P, q, A, l, u; OPTS...)
        dense = PureOSQP.solve(Pm, q, Am, l, u; OPTS...)
        # A row whose answers or iteration counts differ is not a comparison of backends.
        structured.iter == dense.iter ||
            error("$name at n=$n: $(structured.iter) vs $(dense.iter) iterations")
        isapprox(structured.x, dense.x; rtol = 1.0e-6) ||
            error("$name at n=$n: solutions differ")
        ts, tds = abba(
            () -> @be(PureOSQP.setup(P, q, A, l, u; OPTS...), seconds = BUDGET),
            () -> @be(PureOSQP.setup(Pm, q, Am, l, u; OPTS...), seconds = BUDGET),
        )
        tt, tdt = abba(
            () -> @be(PureOSQP.solve(P, q, A, l, u; OPTS...), seconds = BUDGET),
            () -> @be(PureOSQP.solve(Pm, q, Am, l, u; OPTS...), seconds = BUDGET),
        )
        push!(
            rows, (;
                name, n, bw, backend, setup = ts, dense_setup = tds,
                total = tt, dense_total = tdt, iter = structured.iter,
            )
        )
        @printf(
            "%-28s %6d %5d %-13s | %8.1f us %8.1f us %7.1fx | %8.2f ms %8.2f ms %7.1fx | %6d\n",
            name, n, bw, backend, 1.0e6ts, 1.0e6tds, tds / ts,
            1.0e3tt, 1.0e3tdt, tdt / tt, structured.iter
        )
        flush(stdout)
    end
end

open(joinpath(@__DIR__, "results", "structured_backends.json"), "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "blas_threads" => BLAS.get_num_threads(),
            "cases" => [Dict(string(k) => string(v) for (k, v) in pairs(r)) for r in rows],
        ), 2
    )
end
println("\nsaved bench/results/structured_backends.json")
