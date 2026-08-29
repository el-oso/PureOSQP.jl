@testitem "the adjoint derivative matches finite differences in every parameter" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(11)
    n, m = 4, 6
    X = randn(n, n)
    P = Matrix(X'X + I)
    q = randn(n)
    A = randn(m, n)
    b = A * randn(n)
    l, u = b .- rand(m), b .+ rand(m)
    opts = (eps_abs = 1.0e-12, eps_rel = 1.0e-12, max_iter = 200_000, polish = true)

    ws = setup(P, q, A, l, u; opts...)
    @test PureOSQP.solve!(ws).status == SOLVED

    gx, gy = randn(n), randn(m)
    d = adjoint_derivative(ws, gx, gy)

    # The loss whose gradient `d` claims to be, differenced centrally through whole solves.
    L(P, q, A, l, u) = (w = PureOSQP.solve(P, q, A, l, u; opts...); dot(gx, w.x) + dot(gy, w.y))
    h = 1.0e-6
    fd(f) = (f(h) - f(-h)) / 2h

    dP = (Y = randn(n, n); Matrix(Y + Y') / 2)
    dq, dA = randn(n), randn(m, n)
    dl, du = randn(m), randn(m)
    @test dot(d.dP, dP) ≈ fd(t -> L(P + t * dP, q, A, l, u)) rtol = 1.0e-5
    @test dot(d.dq, dq) ≈ fd(t -> L(P, q + t * dq, A, l, u)) rtol = 1.0e-5
    @test dot(d.dl, dl) ≈ fd(t -> L(P, q, A, l + t * dl, u)) rtol = 1.0e-5
    @test dot(d.du, du) ≈ fd(t -> L(P, q, A, l, u + t * du)) rtol = 1.0e-5
    # `A` enters both blocks of the KKT residual, so its gradient has two terms. Dropping
    # the cross term is the usual error and this is what catches it.
    @test dot(d.dA, dA) ≈ fd(t -> L(P, q, A + t * dA, l, u)) rtol = 1.0e-5

    # Shapes and supports.
    @test size(d.dP) == (n, n)
    @test d.dP ≈ d.dP'                    # P is symmetric, so its gradient is
    @test length(d.dq) == n
    @test size(d.dA) == (m, n)
    @test length(d.dl) == m && length(d.du) == m
    # A row cannot be active at both of its bounds.
    @test all(iszero, d.dl .* d.du)

    @test_throws "must equal n" adjoint_derivative(ws, randn(n + 1), gy)
    @test_throws "must equal m" adjoint_derivative(ws, gx, randn(m + 1))
end

@testitem "the forward derivative matches differencing the solution itself" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(11)
    n, m = 4, 6
    X = randn(n, n)
    P = Matrix(X'X + I)
    q = randn(n)
    A = randn(m, n)
    b = A * randn(n)
    l, u = b .- rand(m), b .+ rand(m)
    opts = (eps_abs = 1.0e-12, eps_rel = 1.0e-12, max_iter = 200_000, polish = true)
    ws = setup(P, q, A, l, u; opts...)
    PureOSQP.solve!(ws)

    h = 1.0e-6
    dq = randn(n)
    fx, fy = forward_derivative(ws; dq)
    wp = PureOSQP.solve(P, q + h * dq, A, l, u; opts...)
    wm = PureOSQP.solve(P, q - h * dq, A, l, u; opts...)
    @test fx ≈ (wp.x - wm.x) / 2h atol = 1.0e-6
    @test fy ≈ (wp.y - wm.y) / 2h atol = 1.0e-5

    # Forward and adjoint solve the same system, so they must agree on every directional
    # derivative: <g, J dθ> = <J' g, dθ>.
    gx, gy = randn(n), randn(m)
    d = adjoint_derivative(ws, gx, gy)
    @test dot(gx, fx) + dot(gy, fy) ≈ dot(d.dq, dq) rtol = 1.0e-8

    # Omitted perturbations are zero, so no perturbation means no derivative.
    zx, zy = forward_derivative(ws)
    @test all(iszero, zx)
    @test all(iszero, zy)
end

@testitem "a derivative that does not exist is refused, not approximated" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(11)
    n, m = 4, 6
    X = randn(n, n)
    P = Matrix(X'X + I)
    q = randn(n)
    A = randn(m, n)
    b = A * randn(n)
    l, u = b .- rand(m), b .+ rand(m)
    opts = (eps_abs = 1.0e-12, eps_rel = 1.0e-12, max_iter = 200_000, polish = true)

    # Two identical equality rows: both active by construction, so the active constraint
    # gradients are exactly dependent and no derivative exists. A regularized solve would
    # return a finite, plausible, wrong number instead.
    A2 = vcat(A, A[1:1, :])
    l2 = vcat(l, [b[1]])
    u2 = vcat(u, [b[1]])
    l2[1] = b[1]
    u2[1] = b[1]
    ws2 = setup(P, q, A2, l2, u2; opts...)
    PureOSQP.solve!(ws2)
    @test_throws "cannot be independent" adjoint_derivative(ws2, randn(n), randn(m + 1))

    # More equality rows than variables: the same refusal, reached by counting.
    A3 = A[1:5, :]
    b3 = A3 * randn(n)
    ws3 = setup(P, q, A3, b3, b3; opts...)
    PureOSQP.solve!(ws3)
    @test_throws "cannot be independent" adjoint_derivative(ws3, randn(n), randn(5))
    @test_throws "cannot be independent" forward_derivative(ws3; dq = randn(n))
end
