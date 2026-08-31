# What a low-rank coupling is worth, against the dense path the same problem takes without
# it.
#
# `A` is `k` dense rows above one-entry rows on the remaining variables, so with a `Diagonal`
# `P` the reduced matrix is a diagonal plus a rank-`k` correction. The Woodbury backend
# applies that in `O(nk)` and stores it in `O(nk)`; the dense path forms all `n²` of it. Each
# row is one problem solved twice — once as a `RowCoupled` and once as a `Matrix` — so the
# iteration counts match exactly and the difference is the backend.
using PureOSQP, LinearAlgebra, Chairmarks, Printf, JSON, Statistics

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

"A `Diagonal` cost with `k` coupling rows above bounds on the remaining `n - k` variables."
function problem(n, k)
    P = Diagonal(rand(n) .+ 0.5)
    A = PureOSQP.RowCoupled(randn(k, n) ./ 4, ones(n - k), collect(1:(n - k)))
    return P, randn(n), A, -rand(n), rand(n)
end

println("\nThe low-rank backend against the dense path, on the same problem.\n")
@printf(
    "%-6s %4s %-9s | %10s %10s %8s | %10s %10s %8s | %6s\n",
    "n", "k", "backend", "setup", "dense setup", "setup×",
    "total", "dense total", "total×", "iter"
)
println("-"^108)

rows = NamedTuple[]
for n in (500, 1000, 2000), k in (1, 2, 6, 16)
    P, q, A, l, u = problem(n, k)
    Pm, Am = Matrix(P), Matrix(A)
    backend = PureOSQP.backend_name(PureOSQP.setup(P, q, A, l, u; OPTS...).linsys)
    structured = PureOSQP.solve(P, q, A, l, u; OPTS...)
    dense = PureOSQP.solve(Pm, q, Am, l, u; OPTS...)
    # A row whose answers or iteration counts differ is not a comparison of backends.
    structured.iter == dense.iter ||
        error("n=$n, k=$k: $(structured.iter) vs $(dense.iter) iterations")
    isapprox(structured.x, dense.x; rtol = 1.0e-6) || error("n=$n, k=$k: solutions differ")
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
            n, k, backend, setup = ts, dense_setup = tds,
            total = tt, dense_total = tdt, iter = structured.iter,
        )
    )
    @printf(
        "%-6d %4d %-9s | %8.1f us %8.1f us %7.1fx | %8.2f ms %8.2f ms %7.1fx | %6d\n",
        n, k, backend, 1.0e6ts, 1.0e6tds, tds / ts,
        1.0e3tt, 1.0e3tdt, tdt / tt, structured.iter
    )
    flush(stdout)
end

open(joinpath(@__DIR__, "results", "lowrank_backend.json"), "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "blas_threads" => BLAS.get_num_threads(),
            "cases" => [Dict(string(k) => string(v) for (k, v) in pairs(r)) for r in rows],
        ), 2
    )
end
println("\nsaved bench/results/lowrank_backend.json")
