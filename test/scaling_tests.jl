@testitem "lazy scaled products match explicit scaling" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(1)
    n, m = 7, 11
    P = (X = randn(n, n); Matrix(Symmetric(X'X)))
    A = randn(m, n) * Diagonal(exp10.(range(-3, 3; length = n)))
    q = randn(n)
    l = -rand(m)
    u = rand(m)
    ws = setup(P, q, A, l, u; scaling = 10)
    Pt = ws.c .* (Diagonal(ws.D) * P * Diagonal(ws.D))
    At = Diagonal(ws.E) * A * Diagonal(ws.D)
    x = randn(n)
    y = randn(m)
    @test PureOSQP.mul_A!(similar(y), ws, x) ≈ At * x
    @test PureOSQP.mul_At!(similar(x), ws, y) ≈ At' * y
    @test PureOSQP.mul_P!(similar(x), ws, x) ≈ Pt * x
    @test ws.q ≈ ws.c .* (ws.D .* q)
    @test ws.l ≈ ws.E .* l
    @test ws.u ≈ ws.E .* u
end

@testitem "equilibration reduces the column-norm spread" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(2)
    n, m = 20, 60
    A = randn(m, n) * Diagonal(exp10.(range(-4, 4; length = n)))
    ws = setup(zeros(n, n), zeros(n), A, -ones(m), ones(m); scaling = 10)
    At = Diagonal(ws.E) * A * Diagonal(ws.D)
    spread(M) = (v = [maximum(abs, view(M, :, j)) for j in axes(M, 2)]; maximum(v) / minimum(v))
    @test spread(A) > 1.0e6
    @test spread(At) < 10
end

@testitem "P and A are never mutated" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(3)
    A = randn(5, 3)
    P = (X = randn(3, 3); Matrix(X'X))
    A0, P0 = copy(A), copy(P)
    PureOSQP.solve(P, zeros(3), A, -ones(5), ones(5); scaling = 10, max_iter = 50)
    @test A == A0
    @test P == P0
end

@testitem "scaling = 0 leaves the factors at identity" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(4)
    A = randn(5, 3)
    P = (X = randn(3, 3); Matrix(X'X))
    q = randn(3)
    ws = setup(P, q, A, -ones(5), ones(5); scaling = 0)
    @test all(isone, ws.D)
    @test all(isone, ws.E)
    @test isone(ws.c)
    @test ws.q == q
end

@testitem "storage type never changes the scaling factors" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
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
    opts = (eps_abs = 1.0e-9, eps_rel = 1.0e-9, max_iter = 100_000)

    dense = setup(Matrix(Psp), q, Matrix(Asp), l, u; opts...)
    sprse = setup(Psp, q, Asp, l, u; opts...)
    @test dense.D == sprse.D
    @test dense.E == sprse.E
    @test dense.c == sprse.c

    sd = PureOSQP.solve(Matrix(Psp), q, Matrix(Asp), l, u; opts...)
    ss = PureOSQP.solve(Psp, q, Asp, l, u; opts...)
    @test sd.status == SOLVED
    @test ss.iter == sd.iter
    @test ss.x ≈ sd.x rtol = 1.0e-8
    @test maximum(kkt_residuals(Matrix(Psp), q, Matrix(Asp), l, u, ss.x, ss.y)) < 1.0e-5
end

@testitem "equilibration honours the AbstractMatrix promise" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
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

    plain = setup(Pd, q, Ad, l, u)
    viewed = setup(Pd, q, view(big, 1:m, 1:n), l, u)
    symwrapped = setup(Symmetric(Pd), q, Ad, l, u)
    for w in (viewed, symwrapped)
        @test w.D ≈ plain.D
        @test w.E ≈ plain.E
        @test w.c ≈ plain.c
    end

    Ssp = sparse(Symmetric((S = sprandn(n, n, 0.1); Matrix(S'S) + 3I)))
    sym_sparse = setup(Symmetric(Ssp), q, Ad, l, u)
    plain_sparse = setup(Matrix(Ssp), q, Ad, l, u)
    @test sym_sparse.D ≈ plain_sparse.D
    @test sym_sparse.E ≈ plain_sparse.E
end
