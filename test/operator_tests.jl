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

@testitem "a SciMLOperator reaches the solver" begin
    using LinearAlgebra, SciMLOperators, Krylov, Random
    Random.seed!(34)

    n, m = 50, 30
    P = let S = randn(n, n)
        Symmetric(S'S ./ n + 8I)
    end
    A = randn(m, n) ./ sqrt(n)
    q = randn(n)
    b = A * randn(n)
    l, u = b .- rand(m), b .+ rand(m)
    opts = (scaling = 0, linsys = :indirect, eps_abs = 1.0e-9, eps_rel = 1.0e-9)

    # The in-place signature is `op(w, v, u, p, t)`: `w` receives the result, `v` is the
    # vector being multiplied. A four-argument function is the out-of-place form instead,
    # and writing into its first argument overwrites the caller's vector.
    Pd, Ad = Matrix(P), A
    Pop = FunctionOperator(
        (w, v, u, p, t) -> mul!(w, Pd, v), zeros(n), zeros(n);
        op_adjoint = (w, v, u, p, t) -> mul!(w, Pd', v),
        islinear = true, issymmetric = true, isposdef = true,
    )
    Aop = FunctionOperator(
        (w, v, u, p, t) -> mul!(w, Ad, v), zeros(n), zeros(m);
        op_adjoint = (w, v, u, p, t) -> mul!(w, Ad', v), islinear = true,
    )

    # `symmetric` and `posdef` come from the operator's own traits rather than the caller.
    ws = setup(Pop, q, Aop, l, u; opts...)
    @test ws.P isa PureOSQP.ProductOperator
    @test PureOSQP.is_symmetric(ws.P)
    @test PureOSQP.is_convex(Float64, ws.P, 1.0e-6)

    dense = solve(P, q, A, l, u; opts...)
    opped = solve(Pop, q, Aop, l, u; opts...)
    @test opped.status === PureOSQP.SOLVED
    @test opped.x ≈ dense.x rtol = 1.0e-6

    # An operator and a matrix mix: only the operator is wrapped.
    mixed = setup(Pop, q, A, l, u; opts...)
    @test mixed.P isa PureOSQP.ProductOperator
    @test mixed.A isa Matrix

    # An operator holding entries is unwrapped, not wrapped: its entries are what the
    # equilibration and the factoring backends need, and the matrix itself is what the
    # workspace holds.
    Pmat = Matrix(P)
    ws_mat = setup(MatrixOperator(Pmat), q, A, l, u)
    @test ws_mat.P === Pmat

    # Unwrapping keeps the type, so a diagonal operator still reaches the diagonal backend
    # rather than a dense one.
    d = 2.0 .+ rand(n)
    ws_diag = setup(DiagonalOperator(d), q, Diagonal(ones(n)), fill(-1.0, n), fill(1.0, n))
    @test ws_diag.P isa Diagonal

    # `Aᵀ` runs every iteration, so an operator that cannot supply one is refused at setup
    # rather than failing inside the first product.
    noadj = FunctionOperator(
        (w, v, u, p, t) -> mul!(w, Ad, v), zeros(n), zeros(m); islinear = true
    )
    @test_throws "op_adjoint" setup(Pop, q, noadj, l, u; opts...)

    # A composed operator carries no scratch until `cache_operator` gives it some, so it is
    # refused at setup rather than failing partway into the first iteration.
    composed = Aop * DiagonalOperator(ones(n))
    @test_throws "cache_operator" setup(Pop, q, composed, l, u; opts...)
    cached = cache_operator(composed, zeros(n))
    @test solve(Pop, q, cached, l, u; opts...).status === PureOSQP.SOLVED
end

@testitem "an operator solve does not allocate per iteration" begin
    using LinearAlgebra, SciMLOperators, Krylov, Random
    Random.seed!(35)

    # `Aᵀy` runs every iteration, and a SciMLOperator's `adjoint` builds a new object, so a
    # wrapper that rebuilt it per product would allocate in proportion to the iteration
    # count. Holding the total flat across two iteration counts is what proves it does not.
    # The matrix travels as the operator's `p` rather than as a captured variable: a
    # `@testitem` body is module top level, so a captured matrix is a non-constant global and
    # every product would allocate through a dynamic dispatch, hiding what is being measured.
    n = 40
    Ad = Matrix(0.6I, n, n) + diagm(-1 => fill(0.3, n - 1))
    Aop = FunctionOperator(
        (w, v, u, p, t) -> mul!(w, p, v), zeros(n), zeros(n);
        op_adjoint = (w, v, u, p, t) -> mul!(w, p', v), islinear = true, p = Ad,
    )
    Pop = FunctionOperator(
        (w, v, u, p, t) -> (w .= 2 .* v), zeros(n), zeros(n);
        op_adjoint = (w, v, u, p, t) -> (w .= 2 .* v),
        islinear = true, issymmetric = true, isposdef = true,
    )
    q = randn(n)
    l, u = fill(-1.0, n), fill(1.0, n)

    results = map((25, 100)) do iters
        ws = setup(
            Pop, q, Aop, l, u; scaling = 0, max_iter = iters,
            eps_abs = 1.0e-14, eps_rel = 1.0e-14, check_termination = 1,
        )
        solve!(ws)
        # Each measured run starts from the same warm start as the one that reports `iter`,
        # so the two describe the same work. A solve resumed from a converged workspace stops
        # in a handful of iterations and would measure something else entirely.
        warm_start!(ws; x = zeros(n), y = zeros(n))
        iters_taken = solve!(ws).iter
        warm_start!(ws; x = zeros(n), y = zeros(n))
        (@allocated(solve!(ws)), iters_taken)
    end
    # Both runs must stop at `max_iter`, or equal totals would say nothing about the
    # per-iteration cost.
    @test first(results)[2] == 25
    @test last(results)[2] == 100
    @test first(results)[1] == last(results)[1]
end
