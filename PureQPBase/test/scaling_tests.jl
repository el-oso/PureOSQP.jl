@testitem "lazy scaled products match explicit scaling" begin
    using PureQPBase, LinearAlgebra, SparseArrays, Random
    Random.seed!(1)
    n, m = 7, 11
    P = (X = randn(n, n); Matrix(Symmetric(X'X)))
    A = randn(m, n) * Diagonal(exp10.(range(-3, 3; length = n)))
    q = randn(n)
    l = -rand(m)
    u = rand(m)
    prob = PureQPBase.validated_problem(Float64, n, m, P, q, A, l, u, 10)
    Pt = prob.c .* (Diagonal(prob.D) * P * Diagonal(prob.D))
    At = Diagonal(prob.E) * A * Diagonal(prob.D)
    x = randn(n)
    y = randn(m)
    @test PureQPBase.mul_A!(similar(y), prob, x) ≈ At * x
    @test PureQPBase.mul_At!(similar(x), prob, y) ≈ At' * y
    @test PureQPBase.mul_P!(similar(x), prob, x) ≈ Pt * x
    @test prob.q ≈ prob.c .* (prob.D .* q)
    @test prob.l ≈ prob.E .* l
    @test prob.u ≈ prob.E .* u
end

@testitem "equilibration reduces the column-norm spread" begin
    using PureQPBase, LinearAlgebra, SparseArrays, Random
    Random.seed!(2)
    n, m = 20, 60
    A = randn(m, n) * Diagonal(exp10.(range(-4, 4; length = n)))
    prob = PureQPBase.validated_problem(
        Float64, n, m, zeros(n, n), zeros(n), A, -ones(m), ones(m), 10
    )
    At = Diagonal(prob.E) * A * Diagonal(prob.D)
    spread(M) = (v = [maximum(abs, view(M, :, j)) for j in axes(M, 2)]; maximum(v) / minimum(v))
    @test spread(A) > 1.0e6
    @test spread(At) < 10
end

@testitem "scaling = 0 leaves the factors at identity" begin
    using PureQPBase, LinearAlgebra, Random
    Random.seed!(4)
    A = randn(5, 3)
    P = (X = randn(3, 3); Matrix(X'X))
    q = randn(3)
    prob = PureQPBase.validated_problem(Float64, 3, 5, P, q, A, -ones(5), ones(5), 0)
    @test all(isone, prob.D)
    @test all(isone, prob.E)
    @test isone(prob.c)
    @test prob.q == q
end

@testitem "storage type never changes the scaling factors" begin
    using PureQPBase, LinearAlgebra, SparseArrays, Random
    # The SparseArrays extension replaces the column traversals with `nzrange` walks. It
    # must be a pure speed change: if it ever disagreed with the generic fallback the two
    # storages would equilibrate differently and silently take different paths.
    Random.seed!(40)
    n, m = 60, 120
    S = sprandn(n, n, 0.05)
    Psp = sparse(Symmetric(S'S)) + 4I
    Asp = sprandn(m, n, 0.05)
    q = randn(n)
    b = Asp * randn(n)
    l, u = b .- rand(m), b .+ rand(m)
    built(P, A) = PureQPBase.validated_problem(Float64, n, m, P, q, A, l, u, 10)

    dense = built(Matrix(Psp), Matrix(Asp))
    sprse = built(Psp, Asp)
    @test dense.D == sprse.D
    @test dense.E == sprse.E
    @test dense.c == sprse.c
end

@testitem "equilibration honours the AbstractMatrix promise" begin
    using PureQPBase, LinearAlgebra, SparseArrays, Random
    # The column traversals iterate `axes(M, 1)`, not `1:size(M, 1)`, so a wrapper with the
    # same numbers must give the same factors. A `view` is the wrapper the corpus uses; a
    # `Symmetric` sparse P is the one that silently falls back to the slow generic path,
    # which is a speed matter and must remain a correctness non-event.
    Random.seed!(41)
    n, m = 40, 80
    X = randn(n, n)
    Pd = Matrix(X'X + I)
    Ad = randn(m, n)
    big = randn(m + 7, n + 5)
    big[1:m, 1:n] .= Ad
    q = randn(n)
    b = Ad * randn(n)
    l, u = b .- rand(m), b .+ rand(m)
    built(P, A) = PureQPBase.validated_problem(Float64, n, m, P, q, A, l, u, 10)

    plain = built(Pd, Ad)
    for w in (built(Pd, view(big, 1:m, 1:n)), built(Symmetric(Pd), Ad))
        @test w.D ≈ plain.D
        @test w.E ≈ plain.E
        @test w.c ≈ plain.c
    end

    Ssp = sparse(Symmetric((S = sprandn(n, n, 0.1); Matrix(S'S) + 3I)))
    sym_sparse = built(Symmetric(Ssp), Ad)
    plain_sparse = built(Matrix(Ssp), Ad)
    @test sym_sparse.D ≈ plain_sparse.D
    @test sym_sparse.E ≈ plain_sparse.E
end

@testitem "a band type never changes the scaling factors" begin
    using PureQPBase, LinearAlgebra, Random
    # The column traversals visit only the rows a band type can hold a nonzero in, which is
    # sound only because the callers' functions map zero to zero and the running maxima
    # start at zero. If either stopped holding, the structured and dense forms of the same
    # matrix would equilibrate differently and take different paths from there.
    Random.seed!(41)
    n, m = 50, 100
    A = randn(m, n)
    q = randn(n)
    b = A * randn(n)
    l, u = b .- rand(m), b .+ rand(m)
    built(P) = PureQPBase.validated_problem(Float64, n, m, P, q, A, l, u, 10)

    d = abs.(randn(n)) .+ 2
    e = randn(n - 1) ./ 4
    for Ps in (
            Diagonal(d),
            SymTridiagonal(d, e),
            Tridiagonal(e, d, e),
            Bidiagonal(d, e, :U),
            Bidiagonal(d, e, :L),
        )
        # `Bidiagonal` is not symmetric, so it is only admissible as `P` through the
        # symmetric part the solver actually reads; compare against exactly that.
        Pd = Matrix(Ps)
        issymmetric(Pd) || continue
        structured, dense = built(Ps), built(Pd)
        @test structured.D == dense.D
        @test structured.E == dense.E
        @test structured.c == dense.c
    end
end

@testitem "a structured problem equilibrates as the entries alone do" begin
    using PureQPBase, LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    # Equilibration is a property of the numbers, not of how they are stored: declaring a
    # problem's structure must give the same `D`, `E` and `c` as handing over the same
    # entries with no structure to declare. The two forms here reach different backends, so
    # this is the seam where a traversal override, a `structural_rows` method or a scaling
    # shortcut could silently change what a solver is given.
    Random.seed!(45)
    n, k = 60, 4
    C = randn(k, n) ./ 4
    A = PureQPBase.RowCoupled(C, rand(n) .+ 0.5, collect(1:n))
    P = Diagonal(rand(n) .+ 1)
    m = k + n
    q = randn(n)
    b = Matrix(A) * randn(n)
    l, u = b .- rand(m), b .+ rand(m)

    sprob, _, sls = backend_for(P, q, A, l, u)
    fprob, _, fls = backend_for(sparse(P), q, sparse(Matrix(A)), l, u)
    @test PureQPBase.backend_name(sls) == :lowrank
    @test PureQPBase.backend_name(fls) != :lowrank
    @test sprob.D == fprob.D
    @test sprob.E == fprob.E
    @test sprob.c == fprob.c
end

@testitem "a sparse matrix's finiteness check reads only its stored entries" begin
    using PureQPBase, SparseArrays, LinearAlgebra
    # Reading every position of a SparseMatrixCSC is one search per `(i, j)`. For this matrix
    # that is 1e10 searches; on the suite's sparse classes that pattern costs more than the
    # rest of building the problem.
    n = 100_000
    A = sparse([1, 7, n], [1, 3, n], [1.0, 2.0, 3.0], n, n)
    @test (@elapsed PureQPBase.check_finite(A, n, n, "A")) < 1.0
    nonzeros(A)[2] = NaN
    @test_throws "A is not finite at entry (7, 3)" PureQPBase.check_finite(A, n, n, "A")
end
