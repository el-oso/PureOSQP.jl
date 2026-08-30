# Which linear-system backend `setup` picks for every benchmark problem.
#
# The table is the reference a change to backend selection is checked against: the backend and
# the concrete `LinearSystem` type.
#
# `refactor_count` is recorded but does not discriminate: `setup` increments it on both
# branches, so it is 1 on every row whether or not selection produced the factorization. What
# does discriminate is calling `solve_system!` on a backend straight from `choose_backend`
# without calling `factorize!` first, and comparing against a workspace's own backend.
#
# Every extension that provides a backend is loaded here. A backend whose extension is not
# loaded is simply not a candidate, so an oracle taken without them records the fallback
# rather than the choice.
using PureOSQP, LinearAlgebra, SparseArrays, Printf, JSON
using LDLFactorizations, BandedMatrices, Krylov

include(joinpath(@__DIR__, "suite_problems.jl"))

"""
    families(n) -> Vector

The structured `(P, A)` pairs at size `n`, each also in its dense form.
"""
function families(n)
    return [
        (
            "Diagonal, Diagonal",
            Diagonal(rand(n) .+ 0.5), Diagonal(rand(n) .+ 0.5),
        ),
        (
            "SymTridiagonal, Diagonal",
            SymTridiagonal(rand(n) .+ 3, rand(n - 1) ./ 8), Diagonal(rand(n) .+ 0.5),
        ),
        (
            "SymTridiagonal, Tridiagonal",
            SymTridiagonal(rand(n) .+ 4, rand(n - 1) ./ 8),
            Tridiagonal(rand(n - 1) ./ 4, rand(n) .+ 1, rand(n - 1) ./ 4),
        ),
        (
            "SymTridiagonal, Bidiagonal",
            SymTridiagonal(rand(n) .+ 4, rand(n - 1) ./ 8),
            Bidiagonal(rand(n) .+ 1, rand(n - 1) ./ 4, :L),
        ),
        (
            "Symmetric, Banded",
            Symmetric(Matrix(SymTridiagonal(rand(n) .+ 4, rand(n - 1) ./ 8))),
            BandedMatrix(0 => rand(n) .+ 1, 1 => rand(n - 1) ./ 4, -1 => rand(n - 1) ./ 4),
        ),
        (
            # Reduced bandwidth `2·(n÷10)`, a fifth of `n` — inside the range the
            # banded backend accepts, whose limit is `4b <= n`.

            "SymTridiagonal, wide Banded",
            SymTridiagonal(rand(n) .+ 4, rand(n - 1) ./ 8),
            wide_band(n, n ÷ 10),
        ),
    ]
end

"""
    wide_band(n, b) -> BandedMatrix

An `n×n` band of half-width `b`, diagonally dominant so the reduced matrix stays positive
definite however wide the band is.
"""
function wide_band(n::Integer, b::Integer)
    A = BandedMatrix{Float64}(undef, (n, n), (b, b))
    fill!(A.data, 0.0)
    for j in 1:n, i in max(1, j - b):min(n, j + b)
        A[i, j] = i == j ? 1.0 : 0.01
    end
    return A
end

"One row: set the problem up and read the choice off the workspace."
function probe(name, P, q, A, l, u)
    ws = PureOSQP.setup(P, q, A, l, u)
    return (;
        problem = name,
        n = ws.n,
        m = ws.m,
        P_type = string(typeof(P)),
        A_type = string(typeof(A)),
        backend = string(PureOSQP.backend_name(ws.linsys)),
        linsys_type = string(typeof(ws.linsys)),
        refactor_count = string(ws.refactor_count),
    )
end

rows = NamedTuple[]

for (name, make) in CASES
    P, q, A, l, u = make()
    push!(rows, probe(name, P, q, A, l, u))
end

for n in (100, 400, 1000, 2000)
    for (name, P, A) in families(n)
        q, l, u = randn(n), -rand(n), rand(n)
        push!(rows, probe("$name (n=$n)", P, q, A, l, u))
        push!(rows, probe("$name (n=$n), dense", Matrix(P), q, Matrix(A), l, u))
    end
end

@printf(
    "%-40s %6s %6s %-26s %-26s %-18s %s\n",
    "problem", "n", "m", "typeof(P)", "typeof(A)", "backend", "refactor_count"
)
println("-"^150)
for r in rows
    @printf(
        "%-40s %6d %6d %-26s %-26s %-18s %s\n",
        r.problem, r.n, r.m, r.P_type, r.A_type, r.backend, r.refactor_count
    )
end

open(joinpath(@__DIR__, "results", "selection_oracle.json"), "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "blas_threads" => BLAS.get_num_threads(),
            "cases" => [Dict(string(k) => string(v) for (k, v) in pairs(r)) for r in rows],
        ), 2
    )
end
println("\nsaved bench/results/selection_oracle.json")
