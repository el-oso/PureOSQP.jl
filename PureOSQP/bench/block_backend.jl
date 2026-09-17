# What a block-diagonal structure is worth, against the two things a caller could otherwise do
# with it.
#
# A `BlockDiagonal` `P` and `A` split at the same columns make the reduced matrix decouple into
# `K` independent systems. The alternatives are forming the whole `n×n` reduced matrix
# (`linsys = :dense`, which reads only the blocks' entries but stores and factors all of `n²`)
# and not forming anything (`linsys = :indirect`, matrix-free conjugate gradients). Those are
# the comparisons that exist for an operator held as blocks; a sparse factorization is not one,
# because it needs the matrix assembled.
#
# `n` is held fixed while `K` grows, so each row is the same problem size carved differently.
using PureOSQP, Krylov, LinearAlgebra, Chairmarks, Printf, JSON, Statistics, Random

BLAS.set_num_threads(1)

const OPTS = (eps_abs = 1.0e-9, eps_rel = 1.0e-9)
const BUDGET = 2
const N = 240
const BLOCK_COUNTS = (2, 3, 4, 6, 8, 12, 20)

times(x) = [s.time for s in x.samples]

"""
    abba(a, b) -> (ta, tb)

Time `a` and `b` in the order a, b, b, a and return each one's median over the pooled samples
of its two turns, so a drift that is monotonic over the pair lands on both equally.
"""
function abba(a, b)
    ta1, tb1 = a(), b()
    tb2, ta2 = b(), a()
    return (
        median(vcat(times(ta1), times(ta2))),
        median(vcat(times(tb1), times(tb2))),
    )
end

"""
    problem(n, K) -> (P, A, q, l, u)

`K` equal blocks over `n` columns, each block square in `P` and two-thirds as tall in `A`.
"""
function problem(n, K)
    rng = MersenneTwister(613 + n + K)
    nb = n ÷ K
    mb = max(2, (2nb) ÷ 3)
    Pb = map(1:K) do _
        S = randn(rng, nb, nb)
        Matrix(Symmetric(S'S ./ nb + 3I))
    end
    Ab = [randn(rng, mb, nb) ./ sqrt(nb) for _ in 1:K]
    P = PureOSQP.BlockDiagonal(Pb)
    A = PureOSQP.BlockDiagonal(Ab)
    m = size(A, 1)
    b = A * randn(rng, size(A, 2))
    return P, A, randn(rng, size(A, 2)), b .- rand(rng, m), b .+ rand(rng, m)
end

println("\nThe block backend against forming the reduced matrix and against matrix-free CG.\n")
@printf(
    "%4s %5s %6s | %9s %9s %8s | %9s %8s | %8s %8s %7s\n",
    "K", "block", "iter", "block", "dense", "vs dense", "indirect", "vs cg",
    "words", "dense", "saved"
)
println("-"^100)

rows = NamedTuple[]
for K in BLOCK_COUNTS
    P, A, q, l, u = problem(N, K)
    n = size(P, 1)
    ws = PureOSQP.setup(P, q, A, l, u; OPTS...)
    PureOSQP.backend_name(ws.linsys) === :block ||
        error("K=$K did not build the block backend")
    blk = PureOSQP.solve(P, q, A, l, u; OPTS...)
    den = PureOSQP.solve(P, q, A, l, u; linsys = :dense, OPTS...)
    ind = PureOSQP.solve(P, q, A, l, u; linsys = :indirect, OPTS...)
    # The block and dense paths solve the same system exactly; CG solves it approximately, so
    # only the first pair must match iteration for iteration.
    blk.iter == den.iter || error("K=$K: $(blk.iter) vs $(den.iter) iterations")
    isapprox(blk.x, den.x; rtol = 1.0e-6) || error("K=$K: solutions differ")

    tb, td = abba(
        () -> @be(PureOSQP.solve(P, q, A, l, u; OPTS...), seconds = BUDGET),
        () -> @be(PureOSQP.solve(P, q, A, l, u; linsys = :dense, OPTS...), seconds = BUDGET),
    )
    ti = median(times(@be(PureOSQP.solve(P, q, A, l, u; linsys = :indirect, OPTS...), seconds = BUDGET)))

    words = PureOSQP.backend_info(ws.linsys).factor_nnz
    dense_words = n * (n + 1) ÷ 2
    push!(
        rows, (;
            K, n, block_size = n ÷ K, iter = blk.iter, cg_iter = ind.iter,
            block = tb, dense = td, indirect = ti, words, dense_words,
        )
    )
    @printf(
        "%4d %5d %6d | %7.2f ms %7.2f ms %7.2fx | %7.2f ms %7.2fx | %8d %8d %6.1fx\n",
        K, n ÷ K, blk.iter, 1.0e3tb, 1.0e3td, td / tb, 1.0e3ti, ti / tb,
        words, dense_words, dense_words / words
    )
    flush(stdout)
end

open(joinpath(@__DIR__, "results", "block_backend.json"), "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "blas_threads" => BLAS.get_num_threads(),
            "n" => N,
            "cases" => [Dict(string(k) => string(v) for (k, v) in pairs(r)) for r in rows],
        ), 2
    )
end
println("\nsaved bench/results/block_backend.json")
