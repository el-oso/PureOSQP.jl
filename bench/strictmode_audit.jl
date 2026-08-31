# StrictMode gate for PureOSQP's hot path.
#
# Run:  cd bench && jl strictmode_audit.jl
# The global Stop hook runs this automatically after any turn that touches src/.
#
# `analysis = "full"` in bench/LocalPreferences.toml is deliberate: the `:fast` heuristic
# reports `admm_step!` and `update_residuals!` as allocating when AllocCheck proves they
# do not, so the cheap tier cannot be the gate for this package.
using PureOSQP
using Krylov                   # supplies the :indirect backend, a weak dependency
using LDLFactorizations        # supplies the LDLᵀ backends, likewise
using BandedMatrices           # supplies the banded backend, likewise
using StrictMode
using AllocCheck, JET          # the :full backends; StrictMode dispatches to them
using LinearAlgebra, SparseArrays, Random

include(joinpath(@__DIR__, "lazy_operator.jl"))

# A disabled audit prints exactly like a clean one. Never report a pass without this.
StrictMode.assert_enabled()

function example_workspace(backend::Symbol)
    Random.seed!(1)
    n, m = 12, 30
    X = randn(n, n)
    P = Matrix(X'X / n + I)
    q = randn(n)
    A = randn(m, n)
    b = A * randn(n)
    l, u = b .- rand(m), b .+ rand(m)
    if backend === :sparse
        # The sparse backend is chosen by representation and density, not by a setting, so
        # it is reached by handing `setup` sparse matrices sparse enough to clear the gate.
        return example_sparse_workspace(n, m)
    elseif backend === :cholmod
        # Banded, so the reduced matrix's factor stays sparse enough to be worth keeping.
        return example_banded_workspace(200, 400)
    elseif backend === :sparse_kkt
        return example_kkt_workspace(200, 100)
    elseif backend === :banded
        return example_banded_backend_workspace(200)
    elseif backend === :tridiagonal
        return example_tridiagonal_workspace(200)
    elseif backend === :lowrank
        return example_lowrank_workspace(200, 3)
    elseif backend === :block
        return example_block_workspace(200, 5)
    elseif backend === :operator
        return example_operator_workspace(200)
    elseif backend === :productoperator
        return example_product_operator_workspace(200)
    elseif backend === :diagonal
        # Chosen by representation, like the sparse backends: no setting reaches it.
        return example_diagonal_workspace(200)
    end
    ws = PureOSQP.setup(P, q, A, l, u; linsys = backend)
    PureOSQP.solve!(ws)        # compile every specialization before analysing it
    return ws
end

function example_banded_backend_workspace(n)
    Random.seed!(10)
    # A `Tridiagonal` A squares to bandwidth 2, past what `SymTridiagonal` holds.
    P = SymTridiagonal(rand(n) .+ 4, rand(n - 1) ./ 8)
    A = Tridiagonal(rand(n - 1) ./ 4, rand(n) .+ 1, rand(n - 1) ./ 4)
    ws = PureOSQP.setup(P, randn(n), A, -rand(n), rand(n))
    PureOSQP.solve!(ws)
    return ws
end

function example_tridiagonal_workspace(n)
    Random.seed!(9)
    P = SymTridiagonal(rand(n) .+ 3, rand(n - 1) ./ 8)
    A = Diagonal(rand(n) .+ 0.5)
    ws = PureOSQP.setup(P, randn(n), A, -rand(n), rand(n))
    PureOSQP.solve!(ws)
    return ws
end

"A random symmetric positive definite block of side `k`."
function spd_block(k)
    S = randn(k, k)
    return Matrix(Symmetric(S'S ./ k + 3I))
end

function example_block_workspace(n, K)
    Random.seed!(14)
    nb = n ÷ K
    P = PureOSQP.BlockDiagonal([spd_block(nb) for _ in 1:K])
    A = PureOSQP.BlockDiagonal([randn(nb, nb) ./ sqrt(nb) for _ in 1:K])
    m = size(A, 1)
    b = randn(m)
    ws = PureOSQP.setup(P, randn(size(A, 2)), A, b .- rand(m), b .+ rand(m))
    PureOSQP.solve!(ws)
    return ws
end

function example_lowrank_workspace(n, k)
    Random.seed!(11)
    P = Diagonal(rand(n) .+ 0.5)
    A = PureOSQP.RowCoupled(randn(k, n) ./ 4, ones(n - k), collect(1:(n - k)))
    ws = PureOSQP.setup(P, randn(n), A, -rand(n), rand(n))
    PureOSQP.solve!(ws)
    return ws
end

"""
A workspace over a caller-supplied operator that stores no matrix at all: `P` applies
`Diagonal(d) + α v vᵀ` through a closure and declares `is_materializable` false, so the
ladder descends past the dense terminal to the matrix-free rung.

What this row measures is narrower than the others. The hot path here runs the caller's
`mul!`, so `noalloc` and `typestable` on `admm_step!` hold only as far as that `mul!` does;
the operator in `lazy_operator.jl` is written to carry them, and this row is what shows that
the surrounding machinery does not take them away.
"""
function example_operator_workspace(n)
    Random.seed!(12)
    P = LazyPSD(rand(n) .+ 2.0, randn(n) ./ sqrt(n), 0.5)
    A = randn(n, n) ./ sqrt(n)
    b = A * randn(n)
    # `scaling = 0`: equilibration walks columns, and this operator supplies products only.
    ws = PureOSQP.setup(P, randn(n), A, b .- rand(n), b .+ rand(n); scaling = 0)
    PureOSQP.solve!(ws)
    return ws
end

# `LazyPSD` above is an operator written as an `AbstractMatrix` directly. `ProductOperator`
# is the other route -- a wrapper around a hierarchy that is not one -- and it reaches the
# solve path through a different concrete `Workspace` type, so it is analysed separately.
function example_product_operator_workspace(n)
    Random.seed!(13)
    S = randn(n, n)
    P = PureOSQP.ProductOperator{Float64}(
        Symmetric(S'S ./ n + 8I); symmetric = true, posdef = true
    )
    A = PureOSQP.ProductOperator{Float64}(randn(n, n) ./ sqrt(n))
    b = randn(n)
    ws = PureOSQP.setup(P, randn(n), A, b .- rand(n), b .+ rand(n); scaling = 0)
    PureOSQP.solve!(ws)
    return ws
end

function example_diagonal_workspace(n)
    Random.seed!(7)
    P, A = Diagonal(rand(n) .+ 0.5), Diagonal(rand(n) .+ 0.5)
    l, u = -rand(n), rand(n)
    ws = PureOSQP.setup(P, randn(n), A, l, u)
    PureOSQP.solve!(ws)
    return ws
end

function example_banded_workspace(n, m)
    Random.seed!(3)
    rows, cols, vals = Int[], Int[], Float64[]
    for i in 1:m, j in max(1, div(i * n, m) - 2):min(n, div(i * n, m) + 2)
        push!(rows, i)
        push!(cols, j)
        push!(vals, randn())
    end
    A = sparse(rows, cols, vals, m, n)
    S = spdiagm(-1 => randn(n - 1), 0 => randn(n), 1 => randn(n - 1))
    P = sparse(Symmetric(S'S)) + 3.0I
    b = A * randn(n)
    ws = PureOSQP.setup(P, randn(n), A, b .- rand(m), b .+ rand(m))
    PureOSQP.solve!(ws)
    return ws
end

"""
A problem the full quasi-definite KKT backend is chosen for: a budget row touching every
column, which squares into a dense reduced matrix while leaving `K` sparse. This is what the
OSQP suite's Portfolio class looks like.
"""
function example_kkt_workspace(n, m)
    Random.seed!(8)
    A = vcat(sprandn(m - 1, n, 0.02), sparse(ones(1, n)))
    P = sparse(1.0I, n, n)
    b = A * randn(n)
    ws = PureOSQP.setup(P, randn(n), A, b .- rand(m), b .+ rand(m))
    PureOSQP.solve!(ws)
    return ws
end

function example_sparse_workspace(n, m)
    Random.seed!(2)
    A = sprandn(m, n, 0.05)
    S = sprandn(n, n, 0.05)
    P = sparse(Symmetric(S'S)) + (n * 0.05 + 1) * I
    b = A * randn(n)
    ws = PureOSQP.setup(P, randn(n), A, b .- rand(m), b .+ rand(m))
    PureOSQP.solve!(ws)
    return ws
end

const GUARANTEES = Dict(
    :hot => (:typestable, :noalloc),
    :warm => (:typestable,),
    # The matrix-free backend calls Krylov's `cg!`, which AllocCheck cannot clear: it times
    # itself through an opaque `ccall` to `jl_hrtime`, and its verbose-reporting and
    # residual-history branches are guarded by runtime values, so their `Printf` and
    # `resize!` calls are live code as far as static analysis is concerned even though no
    # solve here takes them. Whitelisting those findings one by one would hollow the gate
    # out, so the claim is split instead: what this package owns is proved statically, and
    # what Krylov owns is measured. See `measured_noalloc` and the `ReducedOperator` check.
    :hot_measured => (:typestable,),
    # `SparseCholmod` builds the reduced matrix through SparseArrays' own sparse products,
    # whose constructors validate dimensions and format the message through code a static
    # analyzer cannot see past -- the same never-taken error path that costs any caller of
    # those constructors its inferrability. That reaches `factorize!` and, through it,
    # `solve!`. It does not reach the hot path, which keeps every guarantee: `admm_step!`,
    # `update_residuals!` and `solve_system!` are checked here exactly as for every other
    # backend. Stated rather than hidden, because the claim really is narrower here.
    :warm_sparse => (),
)

"""
    measured_noalloc(measure)

Assert a hot-path call allocates nothing, by measurement rather than by static analysis.

This is weaker evidence than AllocCheck and is used only where AllocCheck cannot be
applied. It is paired with a static `:noalloc` check on `ReducedOperator`'s `mul!`, which
is the part of the matrix-free path this package actually writes.
"""
function measured_noalloc(measure)
    measure()   # discard the first, which pays for any lingering compilation
    bytes = measure()
    iszero(bytes) || error("measured $bytes bytes at run time")
    return nothing
end

failures = String[]

for backend in (
        :auto, :kkt, :sparse, :cholmod, :diagonal, :tridiagonal, :banded, :lowrank, :indirect,
        :block, :operator, :productoperator,
    )
    ws = example_workspace(backend)
    W = typeof(ws)
    LS = typeof(ws.linsys)
    V = Vector{Float64}
    # `:operator` and `:productoperator` reach the same matrix-free backend as `:indirect`
    # and take the same exemption for Krylov's `cg!`, whose timing and `allocate_if` branches
    # are statically visible and never taken.
    matrix_free = backend in (:indirect, :operator, :productoperator)
    tier = matrix_free ? :hot_measured : :hot
    # See `:warm_sparse`: only the factorization side is affected, never the hot path.
    # `:cholmod` reaches sparse arithmetic whichever engine factors it -- the reduced matrix
    # is assembled the same way before either sees it.
    warm = backend === :cholmod ? :warm_sparse : :warm
    solve_sys() = @allocated PureOSQP.solve_system!(ws.linsys, ws, ws.rhs_x, ws.rhs_z)
    step() = @allocated PureOSQP.admm_step!(ws)
    checks = Any[
        (PureOSQP.admm_step!, (W,), tier, step),
        (PureOSQP.update_residuals!, (W,), :hot, nothing),
        (PureOSQP.solve_system!, (LS, W, V, V), tier, solve_sys),
        (PureOSQP.check_termination, (W, Bool), :warm, nothing),
        (PureOSQP.factorize!, (LS, W), warm, nothing),
        # Runs every time `ρ` moves, so it sits inside the solve loop rather than at setup.
        (PureOSQP.refactor_rho!, (LS, W), warm, nothing),
        (PureOSQP.solve!, (W,), warm, nothing),
    ]
    if matrix_free
        # The operator is this package's own code and gets the full static guarantee, with
        # no exemption: it is where a matrix-free product would allocate if one did.
        op = Base.get_extension(PureOSQP, :PureOSQPKrylovExt).ReducedOperator(ws)
        push!(checks, (LinearAlgebra.mul!, (V, typeof(op), V), :hot, nothing))
    end
    if backend === :cholmod
        # `factorize!` gets no static claim because it reaches CHOLMOD, but the part this
        # package owns -- rebuilding the reduced matrix -- is plain loops over vectors and
        # carries the full guarantee. A refactorization runs every time `ρ` moves, so an
        # allocation here would land inside the solve loop, not just at setup.
        Ext = Base.get_extension(PureOSQP, :PureOSQPSparseArraysExt)
        G = typeof(ws.linsys.gram)
        M = typeof(ws.P)
        push!(
            checks,
            (Ext.refill!, (G, M, M, V, V, V, Float64, Float64), :hot, nothing)
        )
    end
    for (f, types, tier, measure) in checks
        label = "$(nameof(f))($(join(types, ", "))) [linsys=$backend]"
        try
            StrictMode.check(f, types; guarantees = GUARANTEES[tier], mode = :full)
            extra = ""
            if tier === :hot_measured
                measured_noalloc(measure)
                extra = ", noalloc (measured: 0 bytes)"
            end
            claims = isempty(GUARANTEES[tier]) ? "no static claim (sparse arithmetic)" : join(GUARANTEES[tier], ", ")
            println("  ✓ ", label, "  ", claims, extra)
        catch e
            push!(failures, label)
            println("  ✗ ", label)
            println(sprint(showerror, e))
        end
    end
end
if isempty(failures)
    println(
        "\nStrictMode: all guarantees hold (checks_enabled=", StrictMode.checks_enabled(),
        ", mode=", StrictMode.analysis_mode(), ")."
    )
else
    println("\nStrictMode: ", length(failures), " failing guarantee(s):")
    foreach(f -> println("  - ", f), failures)
    exit(1)
end
