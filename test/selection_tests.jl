@testitem "every benchmark suite class reaches its recorded backend" begin
    using LinearAlgebra, SparseArrays
    using LDLFactorizations, BandedMatrices, Krylov
    # The benchmark problem classes are the reference selection is asserted against: they
    # carry the block and band structure real problems have, where the other generators
    # here are uniformly random and land every sparse pattern on the same rung.
    include(joinpath(@__DIR__, "..", "bench", "suite_problems.jl"))

    expected = Dict(
        "Random QP" => :cholesky,
        "Eq QP" => :cholesky,
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

@testitem "every structured family reaches its recorded backend" begin
    using LinearAlgebra, BandedMatrices, Random
    using LDLFactorizations, SparseArrays, Krylov
    Random.seed!(71)

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
        # Reduced bandwidth `n ÷ 2`, where the band is still the smaller representation:
        # `2b + 1 = n + 1` against the dense backend's `m + n = 2n`.
        (
            :banded,
            SymTridiagonal(rand(n) .+ 4, rand(n - 1) ./ 8),
            let b = n ÷ 4, A = BandedMatrix{Float64}(undef, (n, n), (b, b))
                fill!(A.data, 0.0)
                for j in 1:n, i in max(1, j - b):min(n, j + b)
                    A[i, j] = i == j ? 1.0 : 0.01
                end
                A
            end,
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
    n, m = 5, 4
    proto = zeros(n)
    P, A = Matrix(1.0I, n, n), randn(m, n)

    # A materializable pair stops at the dense terminal, and the rungs above it decline.
    @test isnothing(PureOSQP.density_gate_rung(P, A, proto, n, m))
    @test isnothing(PureOSQP.formed_rung(P, A, proto, n, m))
    ls, factored = PureOSQP.dense_rung(P, A, proto, n, m)
    @test ls isa PureOSQP.ReducedCholesky
    @test !factored

    # Below the terminal: an operator the terminal cannot form a product from reaches the
    # matrix-free rung instead of falling out of the ladder.
    struct Opaque end
    @test isnothing(PureOSQP.dense_rung(Opaque(), Opaque(), proto, n, m))
    ls, factored = PureOSQP.indirect_rung(Opaque(), Opaque(), proto, n, m)
    @test PureOSQP.backend_name(ls) === :indirect
    @test !factored

    # The descent itself, not just its rungs: a pair no rung above the terminal serves stops
    # at the terminal, and one no rung serves at all reaches the bottom.
    D, E, c, rho, sig = ones(n), ones(m), 1.0, ones(m), 1.0e-6
    ls, factored = PureOSQP.select_backend(P, A, proto, n, m, D, E, c, rho, sig)
    @test ls isa PureOSQP.ReducedCholesky
    @test !factored
    ls, factored = PureOSQP.select_backend(Opaque(), Opaque(), proto, n, m, D, E, c, rho, sig)
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
