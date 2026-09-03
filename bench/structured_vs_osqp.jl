# Declared structure against a sparse factorization of the same matrix.
#
# OSQP takes CSC and nothing else, so what it exploits is *where the zeros are*. These problems
# carry more than that -- blocks that decouple, a band, a low-rank coupling, a Kronecker
# product -- and this measures what declaring it is worth against discovering the sparsity.
#
# One problem is run three ways: OSQP on CSC, PureOSQP on the same CSC, and PureOSQP on the
# structured type. The middle column is what separates the two effects -- sparse against sparse
# is the implementation, structured against sparse is what the declaration buys. Settings are
# pinned on all three -- `adaptive_rho_interval = 50`, and no duality-gap test, which 0.6.2 does
# not have -- so all three take the same path and the difference is the per-iteration solve.
#
# Every row asserts the three agree before it is reported: the same status, the same iteration
# count, and objectives within `1e-9`. A speed comparison against a solver that took a
# different number of steps, or landed somewhere else, would measure nothing.
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

Time one problem three ways, refusing to report a row whose sides disagree.

`Pj` and `Aj` are the structured spellings. The same matrices sparsified go to OSQP, which
takes no other format, and to a second PureOSQP workspace, so the row separates implementation
from declared structure.
"""
function case(name, Pj, Aj, q, l, u; scaling = 10)
    Pd, Ad = Matrix(Pj), Matrix(Aj)
    # The same matrices as CSC. This is OSQP's only input format, and it is also a form
    # PureOSQP accepts, so the middle column isolates the two effects: sparse against sparse
    # is the implementation, structured against sparse is what declaring it buys.
    Ps, As = sparse(Pd), sparse(Ad)

    ws = PureOSQP.setup(Pj, q, Aj, l, u; OPT..., scaling)
    wss = PureOSQP.setup(Ps, q, As, l, u; OPT..., scaling)
    backend = String(PureOSQP.backend_name(ws.linsys))
    backend_sparse = String(PureOSQP.backend_name(wss.linsys))
    rj = PureOSQP.solve!(ws)
    rs = PureOSQP.solve!(wss)
    rc = osqp_solve(Pd, q, Ad, l, u; scaling)

    rj.status === PureOSQP.SOLVED || error("$name: structured returned $(rj.status)")
    rs.status === PureOSQP.SOLVED || error("$name: sparse returned $(rs.status)")
    rc.status == "Solved" || error("$name: OSQP returned $(rc.status)")
    rj.iter == rc.iter == rs.iter ||
        error("$name: iterations $(rj.iter)/$(rs.iter)/$(rc.iter) differ; not comparable")
    (
        isapprox(rj.obj_val, rc.obj; rtol = 1.0e-9) &&
            isapprox(rs.obj_val, rc.obj; rtol = 1.0e-9)
    ) || error("$name: objectives differ")

    tj = med(@be PureOSQP.solve($Pj, $q, $Aj, $l, $u; OPT..., scaling = $scaling) seconds = 3)
    ts = med(@be PureOSQP.solve($Ps, $q, $As, $l, $u; OPT..., scaling = $scaling) seconds = 3)
    tc = med(@be osqp_solve($Pd, $q, $Ad, $l, $u; scaling = $scaling) seconds = 3)
    density = 100 * nnz(As) / length(Ad)
    row = (;
        case = name, n = length(q), m = length(l), backend, backend_sparse, iter = rj.iter,
        density_percent = density, structured_seconds = tj, sparse_seconds = ts,
        osqp_seconds = tc, structured_vs_osqp = tc / tj, sparse_vs_osqp = tc / ts,
        structured_vs_sparse = ts / tj,
    )
    @printf(
        "%-14s %7.1f%% %6d %10.2f %10.2f %10.2f %9.2fx %9.2fx  %s\n",
        name, density, rj.iter, 1.0e3tc, 1.0e3ts, 1.0e3tj, tc / ts, ts / tj, backend
    )
    flush(stdout)
    return row
end

println("\nOne problem, three ways: OSQP sparse, PureOSQP sparse, PureOSQP structured.")
println("Times in ms.\n")
@printf(
    "%-14s %8s %6s %10s %10s %10s %10s %10s  %s\n",
    "structure", "nnz(A)", "iters", "OSQP", "Pure/sp", "Pure/str",
    "sp vs OSQP", "str vs sp", "backend"
)
println("-"^104)

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
