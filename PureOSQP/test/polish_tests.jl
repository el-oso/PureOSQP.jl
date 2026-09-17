@testitem "polishing sharpens a loose solution" begin
    using PureIPM
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    for algorithm in (OperatorSplitting(), InteriorPoint())
        P, q, A, l, u = random_qp(12, 30; seed = 16)
        loose = PureOSQP.solve(P, q, A, l, u, algorithm; eps_abs = 1.0e-3, eps_rel = 1.0e-3, polishing = false)
        sharp = PureOSQP.solve(P, q, A, l, u, algorithm; eps_abs = 1.0e-3, eps_rel = 1.0e-3, polishing = true)
        @test sharp.polished
        r_loose = maximum(kkt_residuals(P, q, A, l, u, loose.x, loose.y))
        r_sharp = maximum(kkt_residuals(P, q, A, l, u, sharp.x, sharp.y))
        @test r_sharp < r_loose / 100
    end
end

@testitem "polishing never makes the answer worse" begin
    using PureIPM
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    for algorithm in (OperatorSplitting(), InteriorPoint())
        checked = Ref(0)
        for trial in 1:15
            Random.seed!(200 + trial)
            n, m = rand(3:12), rand(3:25)
            X = randn(n, n)
            P = iszero(trial % 3) ? Matrix(X'X) : Matrix(X'X + I)
            q = randn(n)
            A = randn(m, n)
            Ax = A * randn(n)
            l, u = Ax .- rand(m), Ax .+ rand(m)
            a = PureOSQP.solve(P, q, A, l, u, algorithm; eps_abs = 1.0e-4, eps_rel = 1.0e-4, polishing = false)
            b = PureOSQP.solve(P, q, A, l, u, algorithm; eps_abs = 1.0e-4, eps_rel = 1.0e-4, polishing = true)
            a.status == SOLVED || continue
            checked[] += 1
            @test b.status == SOLVED
            # Judged by the independent referee, not by the solver's own reported residuals,
            # which are exactly what polish!'s acceptance rule already compares.
            ra = maximum(kkt_residuals(P, q, A, l, u, a.x, a.y))
            rb = maximum(kkt_residuals(P, q, A, l, u, b.x, b.y))
            @test rb <= max(ra * (1 + 1.0e-6), 1.0e-9)
        end
        @test checked[] >= 10
    end
end

@testitem "no active set means polishing is skipped, not failed" begin
    using PureIPM
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(18)
    n = 8
    P = (X = randn(n, n); Matrix(X'X + I))
    q = randn(n)
    A = Matrix(1.0I, n, n)
    for algorithm in (OperatorSplitting(), InteriorPoint())
        s = PureOSQP.solve(
            P, q, A, fill(-1.0e3, n), fill(1.0e3, n), algorithm;
            polishing = true, eps_abs = 1.0e-8, eps_rel = 1.0e-8, max_iter = 100_000
        )
        @test s.status == SOLVED
        @test !s.polished
        @test s.x ≈ -(P \ q) rtol = 1.0e-5
    end
end

@testitem "polish and the derivatives refuse a non-materializable operator" begin
    using PureIPM
    using LinearAlgebra, Random, Krylov

    # Products and nothing else, as in `PureOSQP/test/linsys_tests.jl`. Both paths below build a
    # dense matrix out of `P` and `A` entry by entry, which this operator cannot serve.
    struct ProductsOnly{T} <: AbstractMatrix{T}
        m::Matrix{T}
    end
    Base.size(op::ProductsOnly) = size(op.m)
    LinearAlgebra.mul!(y::AbstractVector, op::ProductsOnly, x::AbstractVector) = mul!(y, op.m, x)
    LinearAlgebra.mul!(
        y::AbstractVector, op::Adjoint{<:Any, <:ProductsOnly}, x::AbstractVector
    ) = mul!(y, parent(op).m', x)
    PureOSQP.is_materializable(::ProductsOnly) = false
    LinearAlgebra.issymmetric(op::ProductsOnly) = issymmetric(op.m)
    PureOSQP.is_convex(::Type{T}, op::ProductsOnly, sigma) where {T} =
        PureOSQP.is_convex(T, op.m, sigma)
    PureOSQP.reduced_diagonal!(
        dest, ::Type{T}, P::ProductsOnly, A::ProductsOnly, rho, E, D, sigma, c
    ) where {T} = PureOSQP.reduced_diagonal!(dest, T, P.m, A.m, rho, E, D, sigma, c)

    Random.seed!(32)
    n, m = 12, 24
    X = randn(n, n)
    P, A = ProductsOnly(Matrix(X'X / n + I)), ProductsOnly(randn(m, n))
    q = randn(n)
    b = A.m * randn(n)
    l, u = b .- rand(m), b .+ rand(m)
    opts = (scaling = 0, eps_abs = 1.0e-8, eps_rel = 1.0e-8, max_iter = 100_000)

    @test_throws "Leave `polishing = false`" PureOSQP.solve(P, q, A, l, u; opts..., polishing = true)

    ws = setup(P, q, A, l, u; opts...)
    @test PureOSQP.solve!(ws).status == SOLVED
    @test_throws "supplies products only" PureOSQP.adjoint_derivative(ws, randn(n), randn(m))
    @test_throws "no matrix-free form" PureOSQP.forward_derivative(ws; dq = randn(n))

    # The same operator through `InteriorPoint()`, `linsys = :indirect`: setup accepts it
    # with a caller-supplied preconditioner, but polishing and the derivatives refuse it by
    # the same name as above, including before the workspace's `polished` field is even
    # examined.
    ipm_opts = (linsys = :indirect, scaling = 0, eps_abs = 1.0e-8, eps_rel = 1.0e-8)
    @test_throws "Leave `polishing = false`" PureOSQP.solve(
        P, q, A, l, u, InteriorPoint(); ipm_opts..., preconditioner = Diagonal(ones(n)), polishing = true
    )
    ws_ipm = setup(P, q, A, l, u, InteriorPoint(); ipm_opts..., preconditioner = Diagonal(ones(n)))
    @test PureOSQP.solve!(ws_ipm).status == SOLVED
    @test_throws "supplies products only" PureOSQP.adjoint_derivative(ws_ipm, randn(n), randn(m))
    @test_throws "no matrix-free form" PureOSQP.forward_derivative(ws_ipm; dq = randn(n))
end
