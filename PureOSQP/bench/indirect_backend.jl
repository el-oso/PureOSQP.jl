# When the matrix-free backend is the right one.
#
# The direct backends form the reduced matrix `P̃ + σI + Ãᵀ diag(ρ) Ã` and factor it. The
# factored matrix is dense whatever the input, an `n×n` inverse, and for a dense or
# dense-enough `A` an `m×n` copy is built to form it; for a sparse `A` it is accumulated
# over the stored entries instead. The matrix-free backend forms nothing at all and applies
# the same operator through the caller's own products, so its cost follows `nnz` per CG
# iteration and its storage follows `n + m`.
#
# That makes the comparison a question about the problem, not about the solver. The two
# sweeps below separate the two things that decide it: how large the problem is, and how
# sparse it is.
using PureOSQP, Krylov, LinearAlgebra, SparseArrays, BenchmarkTools, Random, Printf, JSON

BLAS.set_num_threads(1)

const OPTS = (eps_abs = 1.0e-6, eps_rel = 1.0e-6, max_iter = 20_000)

"""
    sparse_qp(n, m, density) -> (P, q, A, l, u)

A convex QP with both matrices sparse. `P` is made diagonally dominant so it stays positive
definite as the density falls, and the bounds bracket a feasible point so the problem is
solvable rather than merely well formed.
"""
function sparse_qp(n, m, density)
    Random.seed!(n + m + round(Int, 10_000density))
    A = sprandn(m, n, density)
    S = sprandn(n, n, density)
    P = sparse(Symmetric(S'S)) + (n * density + 1) * I
    b = A * randn(n)
    return (P, randn(n), A, b .- rand(m), b .+ rand(m))
end

# The big direct cases run over a second each, so the default budget would sit on one row
# for minutes. Timing is a median over a handful of samples, which is enough to separate
# ratios this large.
timed(f) = minimum(@benchmark($f(), samples = 5, evals = 1, seconds = 60)).time / 1.0e9

"""
    compare(n, m, density) -> NamedTuple

Solve one problem with both backends and measure time, iterations and workspace size.

The two backends stop at different iteration counts by design: the inner solve is inexact,
so the outer iteration follows a slightly different path. What must agree is the answer,
which is asserted rather than reported.
"""
function compare(n, m, density)
    P, q, A, l, u = sparse_qp(n, m, density)
    res = map((:auto, :indirect)) do ls
        ws = PureOSQP.setup(P, q, A, l, u; OPTS..., linsys = ls)
        s = PureOSQP.solve!(ws)
        s.status == PureOSQP.SOLVED || error("$ls did not solve $n×$m at density $density")
        (;
            t = timed(() -> PureOSQP.solve(P, q, A, l, u; OPTS..., linsys = ls)),
            iter = s.iter, bytes = Base.summarysize(ws), obj = s.obj_val,
        )
    end
    direct, indirect = res
    isapprox(direct.obj, indirect.obj; rtol = 1.0e-6) ||
        error("backends disagree: $(direct.obj) vs $(indirect.obj)")
    return (;
        n, m, density, nnz_A = nnz(A), nnz_P = nnz(P),
        t_direct = direct.t, t_indirect = indirect.t,
        iter_direct = direct.iter, iter_indirect = indirect.iter,
        bytes_direct = direct.bytes, bytes_indirect = indirect.bytes,
        speedup = direct.t / indirect.t, shrink = direct.bytes / indirect.bytes,
    )
end

function report(title, cases)
    println("\n", title, "\n")
    @printf(
        "%6s %6s %8s | %11s %11s %8s | %9s %9s %7s | %6s %6s\n",
        "n", "m", "density", "direct", "matrix-free", "speedup",
        "direct", "free", "shrink", "it dir", "it free"
    )
    println("-"^104)
    rows = NamedTuple[]
    for (n, m, d) in cases
        r = compare(n, m, d)
        push!(rows, r)
        @printf(
            "%6d %6d %7.2f%% | %8.2f ms %8.2f ms %7.2fx | %7.1f MiB %5.2f MiB %6.0fx | %6d %6d\n",
            r.n, r.m, 100r.density, 1.0e3r.t_direct, 1.0e3r.t_indirect, r.speedup,
            r.bytes_direct / 2^20, r.bytes_indirect / 2^20, r.shrink,
            r.iter_direct, r.iter_indirect
        )
        flush(stdout)
    end
    return rows
end

# Sparsity held roughly constant in absolute terms -- about 5 nonzeros per row of `A` --
# so what varies down this sweep is size alone.
size_rows = report(
    "Growing the problem at fixed sparsity per row. The direct backend factors a dense n×n\n" *
        "inverse however sparse the input, at O(n³), so this sweep is where the two cross.",
    (
        (200, 400, 0.025), (500, 1000, 0.01), (1000, 2000, 0.005), (2000, 4000, 0.0025),
        (3000, 6000, 0.0017), (4000, 8000, 0.00125),
    ),
)

# Size held fixed, so what varies is how much work each product does.
density_rows = report(
    "Growing the density at fixed size. Both costs now follow `nnz` -- the direct backend\n" *
        "forms the reduced matrix sparsely too -- but the matrix-free one climbs far faster,\n" *
        "because it pays per CG iteration where the direct backend pays per refactorization.",
    ((1000, 2000, 0.002), (1000, 2000, 0.005), (1000, 2000, 0.01), (1000, 2000, 0.02)),
)

open(joinpath(@__DIR__, "results", "indirect_backend.json"), "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "blas_threads" => BLAS.get_num_threads(),
            "by_size" => [Dict(string(k) => v for (k, v) in pairs(r)) for r in size_rows],
            "by_density" => [Dict(string(k) => v for (k, v) in pairs(r)) for r in density_rows],
        ), 2
    )
end
println("\nsaved bench/results/indirect_backend.json")
