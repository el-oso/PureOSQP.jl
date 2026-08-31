# Whether the benchmark suite's problems have block-diagonal structure worth a backend.
#
# `Ãᵀ diag(ρ) Ã` decouples into independent blocks exactly when `A`'s columns partition into
# groups sharing no row, and the reduced matrix decouples on that same partition only if `P`
# respects it too. This measures the partition for each suite problem, the flop ceiling a
# block-diagonal factorization could reach against a dense one, and what the backend those
# problems actually select already stores.
#
# No timing here: the question is structural, so the numbers are counts and are exact.
using PureOSQP, SparseArrays, LinearAlgebra, Printf, JSON
using LDLFactorizations, BandedMatrices, Krylov

include(joinpath(@__DIR__, "suite_problems.jl"))

"""
    column_partition(A) -> (root, sizes)

Connected components of the graph whose vertices are `A`'s columns and whose edges join two
columns sharing a nonzero row. `root[j]` identifies `j`'s component; `sizes` are the component
sizes, largest first.
"""
function column_partition(A::AbstractMatrix)
    n = size(A, 2)
    S = sparse(A .!= 0)
    parent = collect(1:n)
    find(x) = parent[x] == x ? x : (parent[x] = find(parent[x]))
    for i in axes(S, 1)
        cols = findall(!iszero, view(S, i, :))
        isempty(cols) && continue
        for c in view(cols, 2:lastindex(cols))
            parent[find(c)] = find(cols[1])
        end
    end
    root = [find(j) for j in 1:n]
    counts = Dict{Int, Int}()
    for r in root
        counts[r] = get(counts, r, 0) + 1
    end
    return root, sort!(collect(values(counts)); rev = true)
end

"Whether `P` has no entry linking two columns in different components."
function respects_partition(P::AbstractMatrix, root::Vector{Int})
    S = sparse(Matrix(P) .!= 0)
    for j in axes(S, 2), i in findall(!iszero, view(S, :, j))
        root[i] == root[j] || return false
    end
    return true
end

println("\nBlock structure of the suite problems, and what a block backend could buy.\n")
@printf(
    "%-12s %6s %7s %8s %6s %6s | %11s %13s %8s\n",
    "problem", "n", "blocks", "largest", "frac", "P ok", "backend", "factor words",
    "vs block"
)
println("-"^104)

rows = NamedTuple[]
for (name, make) in CASES
    P, q, A, l, u = make()
    n = size(A, 2)
    root, sizes = column_partition(Matrix(A))
    ws = PureOSQP.setup(P, q, A, l, u)
    backend = PureOSQP.backend_name(ws.linsys)
    stored = PureOSQP.backend_info(ws.linsys).factor_nnz
    # What a dense factorization of each block would hold, as one triangle.
    block_words = sum(s -> s * (s + 1) ÷ 2, sizes)
    # The best a block-diagonal factorization could do against a dense one, on flops.
    ceiling = sum(s -> (s / n)^3, sizes)
    push!(
        rows, (;
            name, n, blocks = length(sizes), largest = sizes[1], frac = sizes[1] / n,
            p_respects = respects_partition(P, root), backend = string(backend),
            factor_words = stored, block_words, flop_ceiling = ceiling,
        )
    )
    @printf(
        "%-12s %6d %7d %8d %6.3f %6s | %11s %13d %8.3f\n",
        name, n, length(sizes), sizes[1], sizes[1] / n, respects_partition(P, root),
        backend, stored, stored / block_words
    )
end

open(joinpath(@__DIR__, "results", "block_structure.json"), "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "cases" => [Dict(string(k) => string(v) for (k, v) in pairs(r)) for r in rows],
        ), 2
    )
end
println("\nsaved bench/results/block_structure.json")
