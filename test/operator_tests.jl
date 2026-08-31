@testitem "a products-only operator solves what its matrix solves" begin
    using LinearAlgebra, Krylov, Random
    Random.seed!(31)

    # No `getindex` anywhere on this type or its adjoint, so anything that reaches for an
    # entry fails rather than quietly working through a fallback.
    struct MulOnly{T}
        M::Matrix{T}
    end
    struct MulOnlyAdjoint{T}
        parent::MulOnly{T}
    end
    Base.size(o::MulOnly) = size(o.M)
    Base.size(o::MulOnlyAdjoint) = reverse(size(o.parent.M))
    Base.adjoint(o::MulOnly) = MulOnlyAdjoint(o)
    LinearAlgebra.mul!(y::AbstractVector, o::MulOnly, x::AbstractVector) = mul!(y, o.M, x)
    LinearAlgebra.mul!(y::AbstractVector, o::MulOnlyAdjoint, x::AbstractVector) =
        mul!(y, o.parent.M', x)

    n, m = 50, 30
    # Well conditioned: the operator runs unpreconditioned, and an ill-conditioned reduced
    # matrix would separate the two paths by conditioning rather than by representation.
    P = let S = randn(n, n)
        Symmetric(S'S ./ n + 8I)
    end
    A = randn(m, n) ./ sqrt(n)
    q = randn(n)
    b = A * randn(n)
    l, u = b .- rand(m), b .+ rand(m)
    opts = (scaling = 0, linsys = :indirect, eps_abs = 1.0e-9, eps_rel = 1.0e-9)

    Pop = PureOSQP.ProductOperator{Float64}(MulOnly(Matrix(P)); symmetric = true, posdef = true)
    Aop = PureOSQP.ProductOperator{Float64}(MulOnly(A))

    @test size(Pop) == (n, n)
    @test size(Aop) == (m, n)
    @test !PureOSQP.is_materializable(Pop)
    @test PureOSQP.is_symmetric(Pop)
    @test PureOSQP.is_convex(Float64, Pop, 1.0e-6)

    dense = solve(P, q, A, l, u; opts...)
    operator = solve(Pop, q, Aop, l, u; opts...)
    @test operator.status === PureOSQP.SOLVED
    @test operator.x ≈ dense.x rtol = 1.0e-6
end

@testitem "an operator with no entries declines the paths that need them" begin
    using LinearAlgebra, Krylov, Random
    Random.seed!(32)

    struct ProductsOnly{T}
        M::Matrix{T}
    end
    struct ProductsOnlyAdjoint{T}
        parent::ProductsOnly{T}
    end
    Base.size(o::ProductsOnly) = size(o.M)
    Base.size(o::ProductsOnlyAdjoint) = reverse(size(o.parent.M))
    Base.adjoint(o::ProductsOnly) = ProductsOnlyAdjoint(o)
    LinearAlgebra.mul!(y::AbstractVector, o::ProductsOnly, x::AbstractVector) = mul!(y, o.M, x)
    LinearAlgebra.mul!(y::AbstractVector, o::ProductsOnlyAdjoint, x::AbstractVector) =
        mul!(y, o.parent.M', x)

    n, m = 40, 25
    P = let S = randn(n, n)
        Symmetric(S'S ./ n + 8I)
    end
    A = randn(m, n) ./ sqrt(n)
    q = randn(n)
    b = A * randn(n)
    l, u = b .- rand(m), b .+ rand(m)
    Pop = PureOSQP.ProductOperator{Float64}(ProductsOnly(Matrix(P)); symmetric = true, posdef = true)
    Aop = PureOSQP.ProductOperator{Float64}(ProductsOnly(A))
    opts = (scaling = 0, linsys = :indirect, eps_abs = 1.0e-9, eps_rel = 1.0e-9)

    # Equilibration reads columns, which products cannot answer. The message names the two
    # ways out rather than escaping as an index error from inside a column walk.
    @test_throws "supplies products only" setup(Pop, q, Aop, l, u)
    @test_throws "scaling = 0" setup(Pop, q, Aop, l, u)

    # Polishing and the derivatives copy entries into a dense factorization.
    @test_throws "is_materializable" solve(Pop, q, Aop, l, u; opts..., polish = true)

    # Every rung that would form a matrix declines, so `:auto` reaches the matrix-free one
    # instead of failing inside a factorization.
    ws = setup(Pop, q, Aop, l, u; scaling = 0)
    @test PureOSQP.backend_name(ws.linsys) === :indirect
end

@testitem "probing equilibrates an operator with no entries" begin
    using LinearAlgebra, Krylov, Random
    Random.seed!(34)

    n, m = 40, 25
    # Badly scaled on purpose: equilibration is what this test is about, so the factors must
    # be far from one.
    P = let S = randn(n, n)
        Symmetric(S'S ./ n + 8I)
    end
    A = randn(m, n) ./ sqrt(n)
    A[1, :] .*= 1.0e4
    A[:, 1] .*= 1.0e3
    q = randn(n)
    b = A * randn(n)
    l, u = b .- rand(m), b .+ rand(m)

    probed = setup(
        PureOSQP.ProductOperator{Float64}(Matrix(P); symmetric = true, posdef = true, probe = true),
        q,
        PureOSQP.ProductOperator{Float64}(A; probe = true),
        l, u; linsys = :indirect,
    )
    walked = setup(P, q, A, l, u)

    # `A * eⱼ` copies column `j` when the wrapped operator selects stored entries, so the
    # factors are the same numbers by the same arithmetic, not merely close.
    @test probed.D == walked.D
    @test probed.E == walked.E
    @test probed.c == walked.c

    # Without `probe`, the same operator refuses and names every way out.
    bare = PureOSQP.ProductOperator{Float64}(Matrix(P); symmetric = true, posdef = true)
    @test_throws "probe = true" setup(bare, q, PureOSQP.ProductOperator{Float64}(A), l, u)
end

@testitem "a LinearMap reaches the solver" begin
    using LinearAlgebra, LinearMaps, Krylov, Random
    Random.seed!(33)

    n, m = 50, 30
    P = let S = randn(n, n)
        Symmetric(S'S ./ n + 8I)
    end
    A = randn(m, n) ./ sqrt(n)
    q = randn(n)
    b = A * randn(n)
    l, u = b .- rand(m), b .+ rand(m)
    opts = (scaling = 0, linsys = :indirect, eps_abs = 1.0e-9, eps_rel = 1.0e-9)

    Pmap = LinearMap(Matrix(P); issymmetric = true, isposdef = true)
    Amap = LinearMap(A)

    # `symmetric` and `posdef` come from the map's own traits rather than from the caller.
    ws = setup(Pmap, q, Amap, l, u; opts...)
    @test ws.P isa PureOSQP.ProductOperator
    @test PureOSQP.is_symmetric(ws.P)
    @test PureOSQP.is_convex(Float64, ws.P, 1.0e-6)

    dense = solve(P, q, A, l, u; opts...)
    mapped = solve(Pmap, q, Amap, l, u; opts...)
    @test mapped.status === PureOSQP.SOLVED
    @test mapped.x ≈ dense.x rtol = 1.0e-6

    # A map and a matrix mix: only the map is wrapped.
    mixed = setup(Pmap, q, A, l, u; opts...)
    @test mixed.P isa PureOSQP.ProductOperator
    @test mixed.A isa Matrix
    @test solve(Pmap, q, A, l, u; opts...).status === PureOSQP.SOLVED
end
