# Declared structure against a sparse factorization of the same matrix.
#
# OSQP takes CSC and factors sparsely, so it exploits *where the zeros are*. These problems
# carry more than that -- blocks that decouple, a band, a low-rank coupling, a Kronecker
# product -- and this measures what declaring it is worth against discovering the sparsity.
#
# Both solvers are given the same problem. PureOSQP gets the structured type; OSQP gets the
# same matrices as `SparseMatrixCSC`, which is its native input and the best form available to
# it. Settings are pinned on both sides -- `adaptive_rho_interval = 50`, and no duality-gap
# test, which 0.6.2 does not have -- so the two take the same path and the difference is the
# per-iteration solve.
#
# Every row asserts the two solvers agree before it is reported: the same status, the same
# iteration count, and objectives within `1e-9`. A speed comparison against a solver that took
# a different number of steps, or landed somewhere else, would measure nothing.
#
# The Kronecker row is the one to read first: its `A` has no zeros at all, so a sparse
# factorization has nothing to work with, while the operator is two small factors.
#
# Run:  julia --project=bench bench/structured_vs_osqp.jl
using PureOSQP, OSQP, LinearAlgebra, SparseArrays, Random, Printf, JSON, Statistics
using BandedMatrices, LDLFactorizations, Krylov, Chairmarks

BLAS.set_num_threads(1)

const RESULTS = joinpath(@__DIR__, "results", "structured_vs_osqp.json")
const OPT = (
    eps_abs = 1.0e-6, eps_rel = 1.0e-6, max_iter = 20_000,
    adaptive_rho_interval = 50, check_dualgap = false,
)

med(x) = median(s.time for s in x.samples)

"Solve with OSQP from the same matrices in CSC, its native form."
function osqp_solve(P, q, A, l, u; scaling)
    m = OSQP.Model()
    OSQP.setup!(
        m; P = sparse(triu(SparseMatrixCSC{Float64, Int}(P))), q = collect(Float64, q),
        A = sparse(SparseMatrixCSC{Float64, Int}(A)), l = collect(Float64, l),
        u = collect(Float64, u), eps_abs = OPT.eps_abs, eps_rel = OPT.eps_rel,
        max_iter = OPT.max_iter, adaptive_rho_interval = OPT.adaptive_rho_interval,
        scaling = scaling, verbose = false,
    )
    r = OSQP.solve!(m)
    return (status = string(r.info.status), iter = Int(r.info.iter), obj = r.info.obj_val)
end

"""
    case(name, Pj, Aj, q, l, u; scaling) -> NamedTuple

Time both solvers on one problem, refusing to report a row whose two sides disagree.

`Pj` and `Aj` are the structured spellings; OSQP is handed `Matrix` versions of the same
matrices, sparsified, which is the best input it can take.
"""
function case(name, Pj, Aj, q, l, u; scaling = 10)
    Pd, Ad = Matrix(Pj), Matrix(Aj)
    ws = PureOSQP.setup(Pj, q, Aj, l, u; OPT..., scaling)
    backend = String(PureOSQP.backend_name(ws.linsys))
    rj = PureOSQP.solve!(ws)
    rc = osqp_solve(Pd, q, Ad, l, u; scaling)

    rj.status === PureOSQP.SOLVED || error("$name: PureOSQP returned $(rj.status)")
    rc.status == "Solved" || error("$name: OSQP returned $(rc.status)")
    rj.iter == rc.iter ||
        error("$name: $(rj.iter) iterations against OSQP's $(rc.iter); not comparable")
    isapprox(rj.obj_val, rc.obj; rtol = 1.0e-9) ||
        error("$name: objectives differ, $(rj.obj_val) against $(rc.obj)")

    tj = med(@be PureOSQP.solve($Pj, $q, $Aj, $l, $u; OPT..., scaling = $scaling) seconds = 3)
    tc = med(@be osqp_solve($Pd, $q, $Ad, $l, $u; scaling = $scaling) seconds = 3)
    density = 100 * nnz(sparse(Ad)) / length(Ad)
    row = (;
        case = name, n = length(q), m = length(l), backend, iter = rj.iter,
        density_percent = density, pure_seconds = tj, osqp_seconds = tc, speedup = tc / tj,
    )
    @printf(
        "%-14s %-12s %6d %8.1f%% %8d %10.2f ms %10.2f ms %8.2fx\n",
        name, backend, length(q), density, rj.iter, 1.0e3tj, 1.0e3tc, tc / tj
    )
    flush(stdout)
    return row
end

println("\nDeclared structure against a sparse factorization of the same matrix.\n")
@printf(
    "%-14s %-12s %6s %9s %8s %13s %13s %9s\n",
    "structure", "backend", "n", "nnz(A)", "iters", "PureOSQP", "OSQP", "vs OSQP"
)
println("-"^92)

rows = NamedTuple[]

# Blocks that decouple: the reduced matrix splits into K independent systems.
let
    Random.seed!(4)
    K, nb = 8, 25
    n = K * nb
    P = PureOSQP.BlockDiagonal([Matrix(Symmetric(randn(nb, nb))) + nb * I for _ in 1:K])
    A = PureOSQP.BlockDiagonal([randn(nb, nb) for _ in 1:K])
    b = Matrix(A) * randn(n)
    push!(rows, case("block-diagonal", P, A, randn(n), b .- rand(n), b .+ rand(n)))
end

# A band: the reduced matrix keeps it, so the factor stays banded.
let
    Random.seed!(5)
    n = 400
    P = SymTridiagonal(4.0 .+ rand(n), 0.5 .* rand(n - 1))
    A = Diagonal(1.0 .+ rand(n))
    b = A * randn(n)
    push!(rows, case("tridiagonal", P, A, randn(n), b .- rand(n), b .+ rand(n)))
end

let
    Random.seed!(5)
    n, b = 400, 3
    P = Symmetric(
        BandedMatrix(0 => 8.0 .+ rand(n), (1:b .=> [0.3 .* rand(n - i) for i in 1:b])...)
    )
    A = BandedMatrix(0 => 1.0 .+ rand(n), 1 => 0.2 .* rand(n - 1))
    bb = Matrix(A) * randn(n)
    push!(rows, case("banded", P, A, randn(n), bb .- rand(n), bb .+ rand(n)))
end

# A few dense rows over a box: a diagonal plus a rank-k correction, solved by Woodbury.
let
    Random.seed!(6)
    n, k = 400, 2
    A = PureOSQP.RowCoupled(randn(k, n) ./ n, ones(n), collect(1:n))
    P = Diagonal(2.0 .+ rand(n))
    b = Matrix(A) * randn(n)
    m = size(A, 1)
    push!(rows, case("low-rank", P, A, randn(n), b .- rand(m), b .+ rand(m)))
end

# A₁ ⊗ A₂, which has no zeros at all: sparsity offers nothing, structure offers everything.
let
    Random.seed!(7)
    k = 20
    A1, A2 = randn(k, k) ./ sqrt(k), randn(k, k) ./ sqrt(k)
    A = PureOSQP.KroneckerOperator(A1, A2)
    n = k * k
    P = Diagonal(fill(2.0, n))
    b = kron(A1, A2) * randn(n)
    push!(rows, case("kronecker", P, A, randn(n), b .- rand(n), b .+ rand(n); scaling = 0))
end

println()
println("Same problem, same settings, same iteration count, objectives agreeing to 1e-9.")
println("OSQP sees where the zeros are; PureOSQP is told what the matrix is.")

open(RESULTS, "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "blas_threads" => BLAS.get_num_threads(),
            "eps" => OPT.eps_abs, "adaptive_rho_interval" => OPT.adaptive_rho_interval,
            "cases" => rows,
        ), 2
    )
end
println("\nwrote $RESULTS")
