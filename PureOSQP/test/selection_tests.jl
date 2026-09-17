@testitem "every benchmark suite class reaches its recorded backend" begin
    using LinearAlgebra, SparseArrays
    using LDLFactorizations, BandedMatrices, Krylov
    # The benchmark problem classes are the reference selection is asserted against: they
    # carry the block and band structure real problems have, where the other generators
    # here are uniformly random and land every sparse pattern on the same rung.
    include(joinpath(@__DIR__, "..", "..", "bench", "suite_problems.jl"))

    expected = Dict(
        "Random QP" => :sparse_formed,
        "Eq QP" => :sparse_formed,
        "Portfolio" => :ldl_kkt,
        "Lasso" => :ldlfactorizations,
        "SVM" => :ldlfactorizations,
        "Huber" => :ldlfactorizations,
        "Control" => :sparse_formed,
    )
    @test sort(first.(CASES)) == sort(collect(keys(expected)))
    for (name, make) in CASES
        P, q, A, l, u = make()
        ws = setup(P, q, A, l, u)
        @test PureOSQP.backend_name(ws.linsys) === expected[name]
    end
end

@testitem "the sparse rule reads the pattern and nothing else" begin
    using PureQPBase
    using LinearAlgebra, SparseArrays, Random
    using LDLFactorizations
    Ext = Base.get_extension(PureQPBase, :PureQPBaseSparseArraysExt)
    Random.seed!(74)

    # A pattern's answer must not move when the stored values do: that is what "decides from
    # the pattern" means, and it is what lets `setup` factor once.
    n, m = 400, 200
    A = sprandn(m, n, 0.01)
    P = sparse(1.0I, n, n)
    for sel in (PureOSQP.ADMMSelection(), PureOSQP.IPMSelection())
        want = Ext.sparse_form(P, A, n, m, sel)
        scaled = SparseMatrixCSC(m, n, copy(A.colptr), copy(A.rowval), nonzeros(A) .* 1.0e6)
        @test Ext.sparse_form(P, scaled, n, m, sel) === want
    end

    # One row spanning the variables fills the reduced matrix by itself, so both algorithms
    # take the KKT form. It is the only route to that form under ADMM, so dropping the row
    # leaves the reduced one.
    budget = sparse([fill(1.0, 1, n); Matrix(sprandn(m - 1, n, 0.005))])
    for sel in (PureOSQP.ADMMSelection(), PureOSQP.IPMSelection())
        @test Ext.sparse_form(P, budget, n, m, sel) === :kkt
    end
    @test Ext.sparse_form(P, budget[2:end, :], n, m - 1, PureOSQP.ADMMSelection()) === :reduced

    # A pattern with no sparsity left to exploit is served by neither sparse form under the
    # interior-point method, whose terminal is the dense KKT factorization.
    full = sparse(randn(m, n))
    @test Ext.sparse_form(sparse(randn(n, n)), full, n, m, PureOSQP.IPMSelection()) === :none
    @test Ext.sparse_form(P, full, n, m, PureOSQP.ADMMSelection()) === :none

    # `row_pattern` is the one pass over `A` the rule needs.
    densest, sumsq = Ext.row_pattern(A)
    counts = [count(==(i), rowvals(A)) for i in 1:m]
    @test densest == maximum(counts)
    @test sumsq == sum(abs2, counts)
end

@testitem "recommend_linsys ranks what it measured" begin
    using LinearAlgebra, SparseArrays, Random
    using LDLFactorizations
    Random.seed!(75)

    # Banded, so a sparse factorization has something to win with and the ranking is not a
    # tie between candidates doing the same dense arithmetic.
    n = 150
    P = sparse(SymTridiagonal(fill(2.0, n), fill(0.3, n - 1)))
    A = sparse(Bidiagonal(fill(1.0, n), fill(-1.0, n - 1), :U))
    q, l, u = collect(range(-1.0, 1.0; length = n)), fill(-1.0, n), fill(1.0, n)

    advice = recommend_linsys(P, q, A, l, u; max_iter = 5, repeats = 2)
    @test advice isa LinsysAdvice
    @test advice.linsys in PureOSQP.LINSYS_OPTIONS
    # Ranked by the cost of a whole solve, with the dense terminal and `:auto` both reached.
    @test issorted(advice.candidates; by = c -> c.total_ms)
    @test advice.solve_iters >= 1
    @test :auto in [c.linsys for c in advice.candidates]
    @test :dense in [c.linsys for c in advice.candidates]
    # Every candidate ran the same bounded number of iterations and reports a real fill, and
    # its total is its setup plus its per-iteration cost over the solve's own iterations.
    for c in advice.candidates
        @test 0 < c.iter <= 5
        @test c.setup_ms > 0 && c.solve_ms > 0
        @test c.factor_fill >= 0
        @test c.total_ms ≈ c.setup_ms + c.iterate_ms * advice.solve_iters
    end
    # The name it reports is one `setup` accepts, and reaches the backend it was measured on.
    ws = setup(P, q, A, l, u; linsys = advice.linsys)
    @test PureOSQP.backend_name(ws.linsys) === first(advice.candidates).backend
    # A name the pair refuses is left out of the ranking rather than raising.
    @test !(:kronecker in [c.linsys for c in advice.candidates])
    @test occursin("LinsysAdvice", sprint(show, MIME"text/plain"(), advice))

    @test_throws "max_iter must be at least 1" recommend_linsys(P, q, A, l, u; max_iter = 0)
    @test_throws "repeats must be at least 1" recommend_linsys(P, q, A, l, u; repeats = 0)
end

@testitem "every structured family reaches its recorded backend" begin
    using LinearAlgebra, BandedMatrices, Random
    using LDLFactorizations, SparseArrays, Krylov
    Random.seed!(71)

    "An `n×n` band of half-width `b`, diagonally dominant so the reduced matrix is definite."
    function wide_band(n, b)
        A = BandedMatrix{Float64}(undef, (n, n), (b, b))
        fill!(A.data, 0.0)
        for j in 1:n, i in max(1, j - b):min(n, j + b)
            A[i, j] = i == j ? 1.0 : 0.01
        end
        return A
    end

    # Each pair is named with the backend it selects. Handing the same numbers over as dense
    # `Matrix`es always reaches the terminal rung, which is what the second assertion checks —
    # for the pairs whose structured form selects something else, that is the structure being
    # worth something; for the one whose structured form is already `:cholesky`, it is not a
    # change.
    families(n) = [
        (
            :diagonal,
            Diagonal(rand(n) .+ 0.5), Diagonal(rand(n) .+ 0.5),
        ),
        (
            :tridiagonal,
            SymTridiagonal(rand(n) .+ 3, rand(n - 1) ./ 8), Diagonal(rand(n) .+ 0.5),
        ),
        (
            :banded,
            SymTridiagonal(rand(n) .+ 4, rand(n - 1) ./ 8),
            Tridiagonal(rand(n - 1) ./ 4, rand(n) .+ 1, rand(n - 1) ./ 4),
        ),
        (
            :tridiagonal,
            SymTridiagonal(rand(n) .+ 4, rand(n - 1) ./ 8),
            Bidiagonal(rand(n) .+ 1, rand(n - 1) ./ 4, :L),
        ),
        # `Symmetric` over a dense parent is not one of the types the banded rung accepts, so
        # this never reaches it and the terminal takes it however narrow the band is.
        (
            :cholesky,
            Symmetric(Matrix(SymTridiagonal(rand(n) .+ 4, rand(n - 1) ./ 8))),
            BandedMatrix(0 => rand(n) .+ 1, 1 => rand(n - 1) ./ 4, -1 => rand(n - 1) ./ 4),
        ),
        # Either side of the rung's limit, which accepts while `4b <= n`. A `BandedMatrix` `A`
        # of half-width `bA` gives a reduced bandwidth of `2bA`, so `bA = 12` is inside at
        # `n = 100` and `bA = 13` is outside.
        (:banded, SymTridiagonal(rand(n) .+ 4, rand(n - 1) ./ 8), wide_band(n, 12)),
        (:cholesky, SymTridiagonal(rand(n) .+ 4, rand(n - 1) ./ 8), wide_band(n, 13)),
        # Three dense rows above bounds on the remaining variables: a diagonal core plus a
        # rank-3 correction, which is not banded at any width.
        (
            :lowrank,
            Diagonal(rand(n) .+ 0.5),
            PureOSQP.RowCoupled(randn(3, n) ./ 4, ones(n - 3), collect(1:(n - 3))),
        ),
        # Either side of the rung's limit, which accepts while `10k <= n`.
        (
            :lowrank,
            Diagonal(rand(n) .+ 0.5),
            PureOSQP.RowCoupled(randn(10, n) ./ 4, ones(n - 10), collect(1:(n - 10))),
        ),
        (
            :cholesky,
            Diagonal(rand(n) .+ 0.5),
            PureOSQP.RowCoupled(randn(11, n) ./ 4, ones(n - 11), collect(1:(n - 11))),
        ),
    ]

    n = 100
    for (backend, P, A) in families(n)
        q, l, u = randn(n), -rand(n), rand(n)
        @test PureOSQP.backend_name(setup(P, q, A, l, u).linsys) === backend
        @test PureOSQP.backend_name(setup(Matrix(P), q, Matrix(A), l, u).linsys) === :cholesky
    end
end

@testitem "the ladder's terminal and indirect rungs" begin
    using LinearAlgebra, Krylov
    include(joinpath(@__DIR__, "helpers.jl"))
    n, m = 5, 4
    P, A = Matrix(1.0I, n, n), randn(m, n)
    prob = raw_problem(P, A, n, m)
    wt = raw_weights(ones(m), 1.0e-6)
    sel = PureOSQP.ADMMSelection()

    # A materializable pair stops at the dense terminal, and the rungs above it decline.
    @test isnothing(PureOSQP.formed_rung(P, A, prob, sel))
    ls, factored = PureOSQP.dense_rung(P, A, prob, sel)
    @test ls isa PureOSQP.ReducedCholesky
    @test !factored

    # Below the terminal: an operator that supplies only products reaches the matrix-free
    # rung instead of falling out of the ladder. `is_materializable` is what declines the
    # terminal, so the decline is reachable from a type `setup` accepts.
    struct Opaque <: AbstractMatrix{Float64} end
    PureOSQP.is_materializable(::Opaque) = false
    opaque_prob = raw_problem(Opaque(), Opaque(), n, m)
    @test isnothing(PureOSQP.dense_rung(Opaque(), Opaque(), opaque_prob, sel))
    ls, factored = PureOSQP.indirect_rung(Opaque(), Opaque(), opaque_prob, sel)
    @test PureOSQP.backend_name(ls) === :indirect
    @test !factored

    # The descent itself, not just its rungs: a pair no rung above the terminal serves stops
    # at the terminal, and one no rung serves at all reaches the bottom.
    ls, factored = PureOSQP.select_backend(P, A, prob, wt, sel)
    @test ls isa PureOSQP.ReducedCholesky
    @test !factored
    ls, factored = PureOSQP.select_backend(Opaque(), Opaque(), opaque_prob, wt, sel)
    @test PureOSQP.backend_name(ls) === :indirect
    @test !factored
end

@testitem "backend_info describes each backend it is asked about" begin
    using LinearAlgebra, SparseArrays, Random
    using LDLFactorizations, BandedMatrices, Krylov
    Random.seed!(72)

    # `factor_nnz` counts one triangle of whatever the backend stores, in that
    # factorization's own convention. `factor_fill` is what normalizes it against the
    # problem's `n` so two backends' fills compare.
    function check(ws, name, direct, system, dim)
        info = PureOSQP.backend_info(ws.linsys)
        @test info isa PureOSQP.BackendInfo
        @test info.name === name === PureOSQP.backend_name(ws.linsys)
        @test info.direct == direct
        @test info.system === system
        @test info.dim == dim
        @test 0 <= info.factor_nnz <= dim * (dim + 1) ÷ 2
        return info
    end

    n, m = 30, 20
    q, l, u = randn(n), -rand(m), rand(m)
    P, A = Matrix(1.0I, n, n), randn(m, n)
    check(setup(P, q, A, l, u), :cholesky, true, :reduced, n)
    check(setup(P, q, A, l, u; linsys = :kkt), :bunchkaufman, true, :kkt, n + m)

    # `factor_fill` normalizes against `n` for both, where `dim` differs between them.
    red = setup(P, q, A, l, u)
    kkt = setup(P, q, A, l, u; linsys = :kkt)
    @test PureOSQP.factor_fill(red) ==
        PureOSQP.backend_info(red.linsys).factor_nnz / n^2
    @test PureOSQP.factor_fill(kkt) ==
        PureOSQP.backend_info(kkt.linsys).factor_nnz / n^2
    # The KKT backend's own `dim` is `n + m`, so normalizing by it would differ.
    @test PureOSQP.backend_info(kkt.linsys).dim == n + m
    check(setup(P, q, A, l, u; linsys = :indirect), :indirect, false, :reduced, n)
    @test iszero(PureOSQP.backend_info(setup(P, q, A, l, u; linsys = :indirect).linsys).factor_nnz)

    ld = Diagonal(rand(n) .+ 0.5)
    dq, dl, du = randn(n), -rand(n), rand(n)
    check(setup(ld, dq, Diagonal(rand(n) .+ 0.5), dl, du), :diagonal, true, :reduced, n)
    tri = SymTridiagonal(rand(n) .+ 3, rand(n - 1) ./ 8)
    check(setup(tri, dq, ld, dl, du), :tridiagonal, true, :reduced, n)
    band = Tridiagonal(rand(n - 1) ./ 4, rand(n) .+ 1, rand(n - 1) ./ 4)
    check(setup(tri, dq, band, dl, du), :banded, true, :reduced, n)

    # A banded factor of bandwidth `b` holds `b + 1` entries per column, less the corner.
    binfo = PureOSQP.backend_info(setup(tri, dq, band, dl, du).linsys)
    @test binfo.factor_nnz == n * 3 - 3

    # The sparse rungs: an identity `A` keeps the KKT factor sparse, a dense-enough sparse
    # `A` sends the pair to the terminal instead.
    sp = setup(sparse(1.0I, n, n), dq, sparse(1.0I, n, n), dl, du)
    info = PureOSQP.backend_info(sp.linsys)
    @test info.factor_nnz < info.dim^2
    @test info.system in (:reduced, :kkt)
    @test info.direct
end

@testitem "two spellings of one matrix select the same backend" begin
    using LinearAlgebra, SparseArrays, BandedMatrices, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(73)

    # `Tridiagonal` and `SymTridiagonal` name the same band, so a problem written either way
    # is one problem and has to reach one backend. Calling `choose_backend` directly beside
    # `setup` is what makes a method ambiguity between `src` and the banded extension fail
    # here rather than at some caller's first use.
    n = 60
    dv, ev = rand(n) .+ 3, rand(n - 1) ./ 8
    sym = SymTridiagonal(copy(dv), copy(ev))
    tri = Tridiagonal(copy(ev), copy(dv), copy(ev))
    diag_A = Diagonal(rand(n) .+ 0.5)
    bidi_A = Bidiagonal(rand(n) .+ 1, rand(n - 1) ./ 4, :L)
    tri_A = Tridiagonal(rand(n - 1) ./ 4, rand(n) .+ 1, rand(n - 1) ./ 4)

    q, l, u = randn(n), -rand(n), rand(n)
    named(P, A) = PureOSQP.backend_name(setup(P, q, A, l, u).linsys)

    @test named(tri, diag_A) === named(sym, diag_A) === :tridiagonal
    @test named(tri, bidi_A) === named(sym, bidi_A) === :tridiagonal
    @test named(sym, tri_A) === :banded

    wt = raw_weights(ones(n), 1.0e-6)
    picked(P, A) = PureOSQP.backend_name(
        first(PureOSQP.choose_backend(P, A, raw_problem(P, A, n, n), wt, PureOSQP.ADMMSelection()))
    )
    @test picked(tri, diag_A) === picked(sym, diag_A) === :tridiagonal
    @test picked(tri, bidi_A) === picked(sym, bidi_A) === :tridiagonal
    @test picked(sym, tri_A) === :banded
end
