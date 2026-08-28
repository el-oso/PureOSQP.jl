# What the matrix type costs.
#
# PureOSQP holds the caller's `P` and `A` and applies equilibration lazily, so every
# per-iteration product runs `mul!` on whatever was passed in. This measures what that is
# worth: a structured matrix keeps its fast product in the iteration loop, but the direct
# solve still materialises dense buffers, so the saving is bounded by how much of the run
# is iterations.
#
# The second half is the case the rest of this repository's benchmarks deliberately avoid:
# a genuinely sparse `A`, where OSQP's sparse LDLᵀ is playing to its strength and PureOSQP
# has no answer but to treat it as dense.
using PureOSQP, OSQP, LinearAlgebra, SparseArrays, BenchmarkTools, Random, Printf, JSON

BLAS.set_num_threads(1)

const OPTS = (eps_abs = 1.0e-6, eps_rel = 1.0e-6, max_iter = 20_000)

solve_pure(P, q, A, l, u) = PureOSQP.solve(P, q, A, l, u; OPTS...)

function solve_osqp(P, q, A, l, u)
    model = OSQP.Model()
    OSQP.setup!(
        model; P = sparse(Symmetric(Matrix(P))), q = collect(q), A = sparse(A),
        l = collect(l), u = collect(u), verbose = false,
        adaptive_rho_interval = 50, check_termination = 25, OPTS...
    )
    return OSQP.solve!(model)
end

# Each family is ONE problem solved twice: once with the matrix stored as a plain `Matrix`,
# once in its structured type. Same numbers, so the iteration counts must match exactly and
# the time difference is purely what the cheaper `mul!` buys. Comparing different families
# against each other would compare different QPs, which says nothing about storage.
function structured_families(n, m; seed = 1)
    Random.seed!(seed)
    X = randn(n, n)
    dense = Matrix(X'X / n + I)
    diagonal = Matrix(Diagonal(diag(dense)))
    band = Matrix(Tridiagonal(diag(dense, -1), diag(dense) .+ 2, diag(dense, 1)))
    A = randn(m, n)
    Abig = randn(m + 20, n + 10)
    Abig[1:m, 1:n] .= A
    return [
        ("dense P", dense, Symmetric(dense), A, A),
        ("diagonal P", diagonal, Diagonal(diag(diagonal)), A, A),
        ("tridiagonal P", band, Symmetric(band), A, A),
        ("A as a view", dense, dense, A, view(Abig, 1:m, 1:n)),
    ]
end

function run_structured(n, m)
    println("Storage type, n = $n, m = $m. Each row is one problem solved twice.\n")
    @printf("%-16s %11s %11s %9s %8s\n", "problem", "as Matrix", "structured", "speedup", "iters")
    println("-"^60)
    Random.seed!(11)
    q = randn(n)
    xref = randn(n)
    rows = NamedTuple[]
    for (name, Pplain, Pstruct, Aplain, Astruct) in structured_families(n, m)
        b = Aplain * xref
        l, u = b .- rand(m), b .+ rand(m)
        s1 = solve_pure(Pplain, q, Aplain, l, u)
        s2 = solve_pure(Pstruct, q, Astruct, l, u)
        @assert s1.status == PureOSQP.SOLVED "$name: $(s1.status)"
        # If these differ, the two runs are not the same problem and the timing is void.
        @assert s1.iter == s2.iter "$name: $(s1.iter) vs $(s2.iter) iterations"
        t1 = @belapsed solve_pure($Pplain, $q, $Aplain, $l, $u)
        t2 = @belapsed solve_pure($Pstruct, $q, $Astruct, $l, $u)
        push!(rows, (; problem = name, t_matrix = t1, t_structured = t2, ratio = t1 / t2, iters = s1.iter))
        @printf("%-16s %9.3f ms %9.3f ms %8.2fx %8d\n", name, 1.0e3t1, 1.0e3t2, t1 / t2, s1.iter)
        flush(stdout)
    end
    return rows
end

function run_sparse()
    println("\n\nSparse A. PureOSQP is measured both ways: handed dense copies, and handed")
    println("the sparse matrices, where the SparseArrays extension walks `nzrange`.\n")
    @printf(
        "%6s %6s %8s | %11s %11s %11s | %8s %8s | %6s\n",
        "n", "m", "density", "densified", "sparse in", "OSQP", "vs OSQP", "gain", "iters"
    )
    println("-"^92)
    rows = NamedTuple[]
    for (n, m) in ((200, 400), (400, 800)), density in (0.01, 0.05, 0.2)
        Random.seed!(n + m + round(Int, 1000density))
        A = sprandn(m, n, density)
        # P sparse too, and diagonally dominant so the problem stays convex.
        S = sprandn(n, n, density)
        Psp = sparse(Symmetric(S'S)) + (n * density + 1) * I
        b = A * randn(n)
        l, u = b .- rand(m), b .+ rand(m)
        q = randn(n)
        Pd, Ad = Matrix(Psp), Matrix(A)
        sp = solve_pure(Pd, q, Ad, l, u)
        ss = solve_pure(Psp, q, A, l, u)
        so = solve_osqp(Psp, q, A, l, u)
        (sp.status == PureOSQP.SOLVED && so.info.status == :Solved) || continue
        # Storage must not change the answer, only the speed.
        @assert ss.iter == sp.iter "sparse input changed the iteration count"
        tp = @belapsed solve_pure($Pd, $q, $Ad, $l, $u)
        ts = @belapsed solve_pure($Psp, $q, $A, $l, $u)
        to = @belapsed solve_osqp($Psp, $q, $A, $l, $u)
        push!(
            rows, (;
                n, m, density, t_densified = tp, t_sparse = ts, t_osqp = to,
                ratio = to / ts, gain = tp / ts,
                iter_pure = sp.iter, iter_osqp = so.info.iter,
            )
        )
        @printf(
            "%6d %6d %7.0f%% | %9.3f ms %9.3f ms %9.3f ms | %7.2fx %7.2fx | %6d\n",
            n, m, 100density, 1.0e3tp, 1.0e3ts, 1.0e3to, to / ts, tp / ts, sp.iter
        )
        flush(stdout)
    end
    return rows
end

struct_rows = run_structured(150, 300)
sparse_rows = run_sparse()

open(joinpath(@__DIR__, "results", "matrix_types.json"), "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "blas_threads" => BLAS.get_num_threads(),
            "structured" => [Dict(string(k) => v for (k, v) in pairs(r)) for r in struct_rows],
            "sparse" => [Dict(string(k) => v for (k, v) in pairs(r)) for r in sparse_rows],
        ), 2
    )
end
println("\nsaved bench/results/matrix_types.json")
