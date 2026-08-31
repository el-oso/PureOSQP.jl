# The Portfolio problem of the benchmark suite, re-expressed in the type that names its
# structure.
#
# `A` is a budget row above `k` factor rows above one bound row per asset, and `P` is
# diagonal — the low-rank shape exactly. As a `SparseMatrixCSC` that is invisible: the factor
# rows are half dense, so the reduced matrix's factor fills in and selection routes the
# problem to the sparse KKT backend. Handing the same numbers over as a `RowCoupled` is the
# whole difference measured here.
#
# Measured in two ρ regimes. At a fixed `ρ` both forms take the same iterates, so the ratio is
# the backends'. Under adaptive `ρ` they do not: the two solves differ in the last digits, `ρ`
# adaptation amplifies that into different update points, and the iteration counts diverge by
# more than the per-iteration gain. Both are reported because neither alone answers "is this
# faster", and the second is the default setting.
using PureOSQP, LinearAlgebra, SparseArrays, Random, Chairmarks, Printf, JSON, Statistics
using LDLFactorizations

BLAS.set_num_threads(1)

const BUDGET = 3

# Fixed `ρ` converges far more slowly than adaptive `ρ`, so it gets the looser tolerance and
# the higher iteration ceiling it needs to reach SOLVED. A run that stops at `max_iter` would
# make the matching iteration counts vacuous — both would be reporting the ceiling.
const REGIMES = (
    ("fixed ρ", (eps_abs = 1.0e-6, eps_rel = 1.0e-6, adaptive_rho = false, max_iter = 200_000)),
    ("adaptive ρ", (eps_abs = 1.0e-9, eps_rel = 1.0e-9)),
)

include(joinpath(@__DIR__, "suite_problems.jl"))

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
    as_rowcoupled(P, A, k, n) -> (Diagonal, RowCoupled)

The same `P` and `A` in the types that name their structure: `k + 1` coupling rows (the budget
row and the factor rows) above one unit row per asset.
"""
function as_rowcoupled(P, A, k, n)
    ncols = size(A, 2)
    # The bound rows are `I` on the first `n` columns, in order, so their weights are one and
    # their columns are themselves.
    A[(k + 2):end, :] == sparse(1.0I, n, ncols) || error("the bound block is not the identity")
    return Diagonal(diag(P)), PureOSQP.RowCoupled(Matrix(A[1:(k + 1), :]), ones(n), collect(1:n))
end

k = 5
P, q, A, l, u = portfolio(k)
n = 100k
Pd, Ac = as_rowcoupled(P, A, k, n)

@printf("\nPortfolio, k = %d: %d variables, %d constraints\n", k, size(A, 2), size(A, 1))

# What the DESIGN asks this tier to be judged on: the words each backend stores for the
# factorization, independent of how long anything takes.
for (form, ws) in (
        ("SparseMatrixCSC", PureOSQP.setup(P, q, A, l, u)),
        ("RowCoupled", PureOSQP.setup(Pd, q, Ac, l, u)),
    )
    info = PureOSQP.backend_info(ws.linsys)
    @printf(
        "  %-16s %-10s system %-8s dim %5d  factor %9d words\n",
        form, info.name, info.system, info.dim, info.factor_nnz
    )
end

rows = NamedTuple[]
for (regime, opts) in REGIMES
    sp = PureOSQP.solve(P, q, A, l, u; opts...)
    lr = PureOSQP.solve(Pd, q, Ac, l, u; opts...)
    # A run that stopped at the ceiling is not a converged comparison, and its iteration
    # count carries no information about the backend.
    sp.status === PureOSQP.SOLVED || error("$regime: sparse form returned $(sp.status)")
    lr.status === PureOSQP.SOLVED || error("$regime: coupled form returned $(lr.status)")
    rel = abs(lr.obj_val - sp.obj_val) / max(1, abs(sp.obj_val))
    rel < 1.0e-6 || error("$regime: objectives differ by $rel")
    ts, tc = abba(
        () -> @be(PureOSQP.solve(P, q, A, l, u; opts...), seconds = BUDGET),
        () -> @be(PureOSQP.solve(Pd, q, Ac, l, u; opts...), seconds = BUDGET),
    )
    es, ec = abba(
        () -> @be(PureOSQP.setup(P, q, A, l, u; opts...), seconds = BUDGET),
        () -> @be(PureOSQP.setup(Pd, q, Ac, l, u; opts...), seconds = BUDGET),
    )
    println("\n  $regime")
    @printf(
        "  %-16s %-10s %10s %8s %10s %14s\n",
        "form", "backend", "total", "iter", "setup", "objective"
    )
    println("  " * "-"^74)
    for (form, t, e, r) in (
            ("SparseMatrixCSC", ts, es, sp), ("RowCoupled", tc, ec, lr),
        )
        @printf(
            "  %-16s %-10s %7.2f ms %8d %7.2f ms %14.8f\n",
            form, r.status, 1.0e3t, r.iter, 1.0e3e, r.obj_val
        )
        push!(
            rows, (;
                regime, form, status = string(r.status), total = t, setup = e,
                iter = r.iter, obj_val = r.obj_val,
            )
        )
    end
    # Per-iteration excludes setup, which the totals above include and which is measured
    # separately beside them.
    @printf(
        "\n  %.2fx total, %.2fx setup, %.2fx per iteration\n",
        ts / tc, es / ec, ((ts - es) / sp.iter) / ((tc - ec) / lr.iter)
    )
end

open(joinpath(@__DIR__, "results", "lowrank_portfolio.json"), "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "blas_threads" => BLAS.get_num_threads(),
            "cases" => [Dict(string(k) => string(v) for (k, v) in pairs(r)) for r in rows],
        ), 2
    )
end
println("\nsaved bench/results/lowrank_portfolio.json")
