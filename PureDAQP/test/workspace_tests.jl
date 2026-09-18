@testitem "ActiveSet honours the Solution and Status contract" begin
    using PureQPBase, PureDAQP
    # The assertions live in PureQPBase, which owns `Solution` and `Status`, so every
    # algorithm is held to one statement of what their values mean rather than to whatever
    # its own suite happens to check.
    PureQPBase.conforms(ActiveSet(); eps = 1.0e-8, slow_iters = 1)
end

@testitem "setup then solve! matches a one-shot solve" begin
    using PureDAQP, LinearAlgebra, Random

    rng = MersenneTwister(23)
    n, m = 6, 10
    P = Matrix(1.0I, n, n)
    q = randn(rng, n)
    A = randn(rng, m, n)
    bu = 0.5 .* abs.(A * randn(rng, n)) .+ 0.05
    ws = setup(P, q, A, -bu, bu, ActiveSet())
    @test dimensions(ws) == (n, m)
    a = solve!(ws)
    b = solve(P, q, A, -bu, bu, ActiveSet())
    @test a.status == SOLVED
    @test a.x ≈ b.x atol = 1.0e-10
end

@testitem "update! of q, l and u keeps the reduction" begin
    using PureDAQP, LinearAlgebra, Random

    rng = MersenneTwister(24)
    n, m = 5, 9
    P = Matrix(1.0I, n, n)
    q = randn(rng, n)
    A = randn(rng, m, n)
    bu = 0.5 .* abs.(A * randn(rng, n)) .+ 0.05
    ws = setup(P, q, A, -bu, bu, ActiveSet())
    solve!(ws)
    red = ws.red

    q2 = randn(rng, n)
    bu2 = bu .* 1.3
    update!(ws; q = q2, l = -bu2, u = bu2)
    # Changing only the vectors must not rebuild the factorization: that is the whole
    # reason a re-solve is cheap here.
    @test ws.red === red
    after = solve!(ws)
    fresh = solve(P, q2, A, -bu2, bu2, ActiveSet())
    @test after.status == SOLVED
    @test after.x ≈ fresh.x atol = 1.0e-9
end

@testitem "update! of P or A rebuilds the reduction" begin
    using PureDAQP, LinearAlgebra, Random

    rng = MersenneTwister(25)
    n, m = 5, 8
    P = Matrix(1.0I, n, n)
    q = randn(rng, n)
    A = randn(rng, m, n)
    bu = 0.5 .* abs.(A * randn(rng, n)) .+ 0.05
    ws = setup(P, q, A, -bu, bu, ActiveSet())
    solve!(ws)
    red = ws.red

    A2 = randn(rng, m, n)
    update!(ws; A = A2)
    @test ws.red !== red
    after = solve!(ws)
    @test after.status == SOLVED
    @test after.x ≈ solve(P, q, A2, -bu, bu, ActiveSet()).x atol = 1.0e-9
end

@testitem "cold_start! drops the working set, and the answer is unchanged" begin
    using PureDAQP, LinearAlgebra, Random

    rng = MersenneTwister(26)
    n, m = 6, 12
    P = Matrix(1.0I, n, n)
    q = randn(rng, n)
    A = randn(rng, m, n)
    bu = 0.5 .* abs.(A * randn(rng, n)) .+ 0.05
    ws = setup(P, q, A, -bu, bu, ActiveSet())
    warm = solve!(ws)
    cold_start!(ws)
    cold = solve!(ws)
    @test cold.status == SOLVED
    @test cold.x ≈ warm.x atol = 1.0e-10
end

@testitem "what ActiveSet refuses, and why" begin
    using PureDAQP, LinearAlgebra

    P = Matrix(1.0I, 2, 2)
    q = [1.0, 1.0]
    A = [1.0 0.0; 0.0 1.0]
    l = [-1.0, -1.0]
    u = [1.0, 1.0]

    # No backend to choose: the method maintains its own factorization.
    @test_throws ArgumentError solve(P, q, A, l, u, ActiveSet(); linsys = :kkt)
    # Equilibration would rescale the rows the working set is priced against.
    @test_throws ArgumentError solve(P, q, A, l, u, ActiveSet(); scaling = 10)
    # Parameters belonging to another algorithm.
    @test_throws ArgumentError solve(P, q, A, l, u, ActiveSet(); rho = 0.1)
    # An indefinite P is not a convex problem at all.
    @test_throws ArgumentError solve([1.0 0.0; 0.0 -1.0], q, A, l, u, ActiveSet())
end

@testitem "eps_prox cannot be changed on an existing workspace" begin
    using PureDAQP, LinearAlgebra

    P = Matrix(1.0I, 3, 3)
    q = [1.0, 2.0, 3.0]
    A = Matrix(1.0I, 3, 3)
    ws = setup(P, q, A, fill(-1.0, 3), fill(1.0, 3), ActiveSet())
    # It is built into the Cholesky factor of `P + eps_prox*I`, so changing it would leave
    # the workspace's factorization describing a different problem.
    @test_throws ArgumentError update_settings!(ws, ActiveSet(; eps_prox = 1.0e-4))
    update_settings!(ws, ActiveSet())
    @test solve!(ws).status == SOLVED
end
