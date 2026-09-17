@testitem "P and A are never mutated" begin
    using LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(3)
    A = randn(5, 3)
    P = (X = randn(3, 3); Matrix(X'X))
    A0, P0 = copy(A), copy(P)
    PureOSQP.solve(P, zeros(3), A, -ones(5), ones(5); scaling = 10, max_iter = 50)
    @test A == A0
    @test P == P0
end

@testitem "storage type never changes the solution" begin
    using LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(40)
    n, m = 60, 120
    S = sprandn(n, n, 0.05)
    Psp = sparse(Symmetric(S'S)) + 4I
    Asp = sprandn(m, n, 0.05)
    q = randn(n)
    b = Asp * randn(n)
    l, u = b .- rand(m), b .+ rand(m)
    opts = (eps_abs = 1.0e-9, eps_rel = 1.0e-9, max_iter = 100_000)

    sd = PureOSQP.solve(Matrix(Psp), q, Matrix(Asp), l, u; opts...)
    ss = PureOSQP.solve(Psp, q, Asp, l, u; opts...)
    @test sd.status == SOLVED
    @test ss.iter == sd.iter
    @test ss.x ≈ sd.x rtol = 1.0e-8
    @test maximum(kkt_residuals(Matrix(Psp), q, Matrix(Asp), l, u, ss.x, ss.y)) < 1.0e-5
end

@testitem "a band type never changes the solution" begin
    using LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(41)
    n, m = 50, 100
    A = randn(m, n)
    q = randn(n)
    b = A * randn(n)
    l, u = b .- rand(m), b .+ rand(m)
    opts = (eps_abs = 1.0e-9, eps_rel = 1.0e-9, max_iter = 100_000)

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
        ss = PureOSQP.solve!(setup(Ps, q, A, l, u; opts...))
        sd = PureOSQP.solve!(setup(Pd, q, A, l, u; opts...))
        @test ss.status == SOLVED
        @test ss.iter == sd.iter
        @test ss.x == sd.x
    end
end

@testitem "two spellings of one problem solve alike" begin
    using LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    # The two forms reach different backends, so this is where a structured backend that
    # disagreed with the formed one in its last digits would show up.
    Random.seed!(45)
    n, k = 60, 4
    C = randn(k, n) ./ 4
    A = PureOSQP.RowCoupled(C, rand(n) .+ 0.5, collect(1:n))
    P = Diagonal(rand(n) .+ 1)
    m = k + n
    q = randn(n)
    b = Matrix(A) * randn(n)
    l, u = b .- rand(m), b .+ rand(m)
    # A fixed rho keeps the comparison on the backends: rho adaptation turns a last-digit
    # difference between two of them into a different update point, and from there the two
    # forms take different iteration counts for a reason that has nothing to do with either.
    opts = (eps_abs = 1.0e-9, eps_rel = 1.0e-9, max_iter = 20_000)
    fixed = OperatorSplitting(adaptive_rho = false)

    ss = PureOSQP.solve(P, q, A, l, u, fixed; opts...)
    sf = PureOSQP.solve(sparse(P), q, sparse(Matrix(A)), l, u, fixed; opts...)
    @test ss.status == SOLVED
    @test sf.status == SOLVED
    @test ss.iter == sf.iter
    @test ss.obj_val ≈ sf.obj_val rtol = 1.0e-8
    @test ss.x ≈ sf.x rtol = 1.0e-8
end
