# PureOSQP against libosqp on sparse problems, both handed sparse matrices.
#
# `matrix_types.jl` measures sparse `A` on random patterns, which is the worst case for any
# sparse factorization -- no separators, near-maximal fill -- and where PureOSQP's answer is
# to form the reduced matrix sparsely and then factor it densely. This file adds the case
# that corpus cannot reach: structured sparsity, where the reduced matrix keeps its band and
# PureOSQP factors it with CHOLMOD, which is the regime OSQP's own sparse LDLᵀ is built for.
#
# Both solvers get `SparseMatrixCSC`. Neither is handed a dense copy of anything.
using PureOSQP, OSQP, LinearAlgebra, SparseArrays, BenchmarkTools, Random, Printf, JSON

BLAS.set_num_threads(1)

const OPTS = (eps_abs = 1.0e-6, eps_rel = 1.0e-6, max_iter = 20_000)

# libosqp 0.6.2 has no duality-gap test and would reject the setting. With it on the two
# solvers stop on different criteria, and the iteration counts below would be comparing the
# criteria rather than the algorithms.
const PURE_ONLY = (check_dualgap = false,)

solve_pure(P, q, A, l, u) = PureOSQP.solve(P, q, A, l, u; OPTS..., PURE_ONLY...)

function solve_osqp(P, q, A, l, u)
    model = OSQP.Model()
    OSQP.setup!(
        model; P = sparse(Symmetric(P)), q = collect(q), A = sparse(A),
        l = collect(l), u = collect(u), verbose = false,
        adaptive_rho_interval = 50, check_termination = 25, OPTS...
    )
    return OSQP.solve!(model)
end

"""
    banded_qp(n, m; band) -> (P, q, A, l, u)

A QP whose constraint rows each couple a contiguous run of variables, as in a
model-predictive control horizon. The reduced matrix inherits the band.
"""
function banded_qp(n, m; band = 3, seed = 0)
    Random.seed!(n + m + band + seed)
    rows, cols, vals = Int[], Int[], Float64[]
    for i in 1:m, j in max(1, div(i * n, m) - band):min(n, div(i * n, m) + band)
        push!(rows, i)
        push!(cols, j)
        push!(vals, randn())
    end
    A = sparse(rows, cols, vals, m, n)
    S = spdiagm(-1 => randn(n - 1), 0 => randn(n), 1 => randn(n - 1))
    P = sparse(Symmetric(S'S)) + 3.0I
    b = A * randn(n)
    return (P, randn(n), A, b .- rand(m), b .+ rand(m))
end

"""
    random_qp(n, m, density) -> (P, q, A, l, u)

Uniformly random sparsity, which has no structure for a sparse factorization to exploit.
Kept alongside the banded family because the two give opposite answers, and reporting only
one of them would be choosing the answer.
"""
function random_qp(n, m, density; seed = 0)
    Random.seed!(n + m + round(Int, 1.0e4density) + seed)
    A = sprandn(m, n, density)
    S = sprandn(n, n, density)
    P = sparse(Symmetric(S'S)) + (n * density + 1) * I
    b = A * randn(n)
    return (P, randn(n), A, b .- rand(m), b .+ rand(m))
end

timed(f) = minimum(@benchmark($f(), samples = 5, evals = 1, seconds = 60)).time / 1.0e9

function compare(name, P, q, A, l, u)
    sp = solve_pure(P, q, A, l, u)
    so = solve_osqp(P, q, A, l, u)
    (sp.status == PureOSQP.SOLVED && so.info.status == :Solved) ||
        error("$name: PureOSQP $(sp.status), OSQP $(so.info.status)")
    # The referee judges both from the original data, so neither solver's own scaling can
    # flatter it. A large gap here would mean the times below are comparing different
    # answers.
    obj_gap = abs(sp.obj_val - so.info.obj_val) / max(1, abs(so.info.obj_val))
    obj_gap < 1.0e-6 || error("$name: objectives disagree by $obj_gap")
    ws = PureOSQP.setup(P, q, A, l, u; OPTS..., PURE_ONLY...)
    tp = timed(() -> solve_pure(P, q, A, l, u))
    to = timed(() -> solve_osqp(P, q, A, l, u))
    return (;
        name, n = size(A, 2), m = size(A, 1), nnz_A = nnz(A), nnz_P = nnz(P),
        backend = PureOSQP.backend_name(ws.linsys), t_pure = tp, t_osqp = to,
        ratio = to / tp, iter_pure = sp.iter, iter_osqp = so.info.iter, obj_gap,
    )
end

function report(title, cases)
    println("\n", title, "\n")
    @printf(
        "%6s %6s %9s %-16s | %11s %11s %8s | %6s %6s\n",
        "n", "m", "nnz(A)", "PureOSQP backend", "PureOSQP", "OSQP", "vs OSQP", "it pu", "it os"
    )
    println("-"^100)
    rows = NamedTuple[]
    for (name, prob) in cases
        r = compare(name, prob...)
        push!(rows, r)
        @printf(
            "%6d %6d %9d %-16s | %8.2f ms %8.2f ms %7.2fx | %6d %6d\n",
            r.n, r.m, r.nnz_A, r.backend, 1.0e3r.t_pure, 1.0e3r.t_osqp, r.ratio,
            r.iter_pure, r.iter_osqp
        )
        flush(stdout)
    end
    return rows
end

banded = report(
    "Banded, as in an MPC horizon. The reduced matrix keeps the band, so PureOSQP factors it\n" *
        "with CHOLMOD -- the regime OSQP's sparse LDLᵀ is built for.",
    [("banded", banded_qp(n, 2n; band = 3)) for n in (200, 500, 1000, 2000)],
)

random = report(
    "Uniformly random sparsity, with no structure to exploit. The reduced matrix fills in, so\n" *
        "PureOSQP forms it sparsely and factors it densely.",
    [
        ("random", random_qp(n, 2n, d)) for (n, d) in
            ((200, 0.01), (200, 0.05), (500, 0.01), (1000, 0.005), (2000, 0.0025))
    ],
)

open(joinpath(@__DIR__, "results", "sparse_headtohead.json"), "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "blas_threads" => BLAS.get_num_threads(),
            "banded" => [Dict(string(k) => string(v) for (k, v) in pairs(r)) for r in banded],
            "random" => [Dict(string(k) => string(v) for (k, v) in pairs(r)) for r in random],
        ), 2
    )
end
println("\nsaved bench/results/sparse_headtohead.json")
