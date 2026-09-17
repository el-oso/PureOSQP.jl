# What the dense terminal costs a pair the low-rank rung declines.
#
# `lowrank_rung` declines for `InteriorPoint()`: on a variable that only the coupling rows
# reach, the diagonal core holds `δ_p` alone and the Woodbury solve through its inverse ends
# without a solution. Declining is right. Where the pair lands after that is a separate
# question, and this measures it: a `Diagonal`/`RowCoupled` pair reaches `FullKKT`, which
# factors an `(n+m)×(n+m)` dense matrix, while the same numbers handed over as
# `SparseMatrixCSC` reach the sparse KKT factorization.
#
# The sparse column carries the cost of building the sparse pair, so the comparison is what a
# caller holding the structured pair would actually pay to convert and solve.
#
#     julia --project=bench PureIPM/bench/ipm_lowrank_terminal.jl
using PureOSQP, PureIPM, PureQPBase
using LinearAlgebra, SparseArrays, LDLFactorizations, BandedMatrices
using Chairmarks, Random, Printf, JSON, Statistics

BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "..", "..", "bench", "structured_problems.jl"))

const OPTS = (eps_abs = 1.0e-8, eps_rel = 1.0e-8)
const BUDGET = 3
const SIZES = (120, 240, 480)

median_seconds(f) = median(s.time for s in @be(f(), seconds = BUDGET).samples)

println("\nInterior point on a low-rank pair: the dense terminal against the sparse KKT form.\n")
@printf(
    "%-16s %5s %-14s %10s %10s %10s %8s %6s\n",
    "family", "n", "structured", "dense ms", "convert ms", "sparse ms", "ratio", "iter"
)
println("-"^92)

rows = NamedTuple[]
for n in SIZES, f in structured_families(n)
    occursin("rank", f.name) || continue
    Ps, As = sparse(Matrix(f.P)), sparse(Matrix(f.A))
    structured = solve(f.P, f.q, f.A, f.l, f.u, InteriorPoint(); OPTS...)
    sparsed = solve(Ps, f.q, As, f.l, f.u, InteriorPoint(); OPTS...)
    backend = string(PureQPBase.backend_name(setup(f.P, f.q, f.A, f.l, f.u, InteriorPoint(); OPTS...).linsys))

    t_dense = median_seconds(() -> solve(f.P, f.q, f.A, f.l, f.u, InteriorPoint(); OPTS...))
    t_sparse = median_seconds(() -> solve(Ps, f.q, As, f.l, f.u, InteriorPoint(); OPTS...))
    t_convert = median_seconds(() -> (sparse(Matrix(f.P)), sparse(Matrix(f.A))))
    ratio = t_dense / (t_sparse + t_convert)
    dx = maximum(abs, structured.x .- sparsed.x; init = 0.0)

    push!(
        rows, (;
            family = f.name, n, backend,
            dense_seconds = t_dense, sparse_seconds = t_sparse, convert_seconds = t_convert,
            ratio, iter = structured.iter, sparse_iter = sparsed.iter, dx,
            status = string(structured.status),
        )
    )
    @printf(
        "%-16s %5d %-14s %10.3f %10.3f %10.3f %7.1fx %6d\n",
        f.name, n, backend, 1.0e3t_dense, 1.0e3t_convert, 1.0e3t_sparse, ratio, structured.iter
    )
    flush(stdout)
end

# The two routes factor the same system by different means, so they must agree on both the
# iteration count and the answer; a row where they do not is a numerical difference rather
# than a cost one.
for r in rows
    r.iter == r.sparse_iter || @warn "iteration counts differ" r.family r.n r.iter r.sparse_iter
end

open(joinpath(@__DIR__, "results", "ipm_lowrank_terminal.json"), "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "blas_threads" => BLAS.get_num_threads(),
            "eps" => OPTS.eps_abs,
            "sizes" => collect(SIZES),
            "cases" => [Dict(string(k) => string(v) for (k, v) in pairs(r)) for r in rows],
        ), 2
    )
end
println("\nsaved PureIPM/bench/results/ipm_lowrank_terminal.json")
