@testitem "the documented example gives the documented answer" begin
    using PureDAQP
    P = [4.0 1.0; 1.0 2.0]
    q = [1.0, 1.0]
    A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
    l = [1.0, 0.0, 0.0]
    u = [1.0, 0.7, 0.7]
    sol = solve(P, q, A, l, u, ActiveSet())
    @test sol.status == SOLVED
    @test sol.x ≈ [0.3, 0.7] atol = 1.0e-9
    @test sol.obj_val ≈ 1.88 atol = 1.0e-9
    # An active-set method stops at a vertex of its working set rather than at a tolerance,
    # so the residuals are at rounding level rather than at `eps_abs`. Not exactly zero: the
    # working-set equations hold exactly in exact arithmetic, but these residuals are
    # recomputed in floating point from the original data, and how they round depends on the
    # BLAS underneath.
    @test sol.prim_res < 1.0e-14
    @test sol.dual_res < 1.0e-12
    # Row 2 is inactive. This one *is* exactly zero: `y` is zeroed and only the working set
    # is written into it, so an inactive row is never assigned at all.
    @test iszero(sol.y[2])
end

@testitem "matches libdaqp on random strictly convex problems" begin
    using PureDAQP, LinearAlgebra, Random
    import DAQP

    rng = MersenneTwister(91)
    for _ in 1:150
        n, m = rand(rng, 2:10), rand(rng, 2:14)
        Q = qr(randn(rng, n, n)).Q
        H = Matrix(Symmetric(Q * Diagonal(exp10.(range(0, 2; length = n))) * Q'))
        f = randn(rng, n)
        A = randn(rng, m, n)
        bu = 0.5 .* abs.(A * randn(rng, n)) .+ 0.05
        sol = solve(H, f, A, -bu, bu, ActiveSet())
        @test sol.status == SOLVED
        xr = DAQP.quadprog(H, f, A, bu, -bu)[1]
        @test norm(sol.x - xr, Inf) / (1 + norm(xr, Inf)) < 1.0e-9
        @test sol.prim_res < 1.0e-10
        @test sol.dual_res < 1.0e-9
    end
end

@testitem "matches libdaqp once the working set is gathered" begin
    using PureDAQP, LinearAlgebra, Random
    import DAQP

    # Sized past `PACKED_KMIN`, so the working set is held as a packed block and the loop
    # takes its products against that rather than one row at a time. The smaller random
    # problems above stay under that capacity and never reach this path.
    rng = MersenneTwister(77)
    for _ in 1:25
        n, m = rand(rng, 18:40), rand(rng, 30:70)
        @test min(n, m) + 1 >= PureDAQP.PACKED_KMIN
        Q = qr(randn(rng, n, n)).Q
        H = Matrix(Symmetric(Q * Diagonal(exp10.(range(0, 2; length = n))) * Q'))
        f = randn(rng, n)
        A = randn(rng, m, n)
        bu = 0.5 .* abs.(A * randn(rng, n)) .+ 0.05
        sol = solve(H, f, A, -bu, bu, ActiveSet())
        @test sol.status == SOLVED
        xr = DAQP.quadprog(H, f, A, bu, -bu)[1]
        @test norm(sol.x - xr, Inf) / (1 + norm(xr, Inf)) < 1.0e-9
        @test sol.prim_res < 1.0e-10
        @test sol.dual_res < 1.0e-9
    end
end

@testitem "equality rows leave the gathered working set in order" begin
    using PureDAQP, LinearAlgebra, Random

    # Rows dropped from the middle of a packed working set shift the block that follows them.
    # Equalities never leave, so a run that drops inequalities around them checks that the
    # gathered rows stay paired with the factorization they belong to.
    rng = MersenneTwister(78)
    for _ in 1:20
        n = rand(rng, 20:30)
        P = Matrix(1.0I, n, n)
        q = randn(rng, n)
        A = randn(rng, n + 12, n)
        b = A * randn(rng, n)
        l = copy(b)
        u = copy(b)
        l[4:end] .-= 0.4 .* abs.(randn(rng, n + 9))
        u[4:end] .+= 0.4 .* abs.(randn(rng, n + 9))
        sol = solve(P, q, A, l, u, ActiveSet())
        @test sol.status == SOLVED
        for i in 1:3
            @test abs(dot(A[i, :], sol.x) - b[i]) < 1.0e-9
        end
        @test sol.prim_res < 1.0e-9
    end
end

@testitem "proximal-point iterations accept a singular P" begin
    using PureDAQP, LinearAlgebra, Random

    rng = MersenneTwister(92)
    for _ in 1:30
        n, m = rand(rng, 4:9), rand(rng, 6:16)
        Q = qr(randn(rng, n, n)).Q
        # Two zero eigenvalues: `P` is positive semidefinite but not invertible, which the
        # reduction cannot factor without regularization.
        lam = vcat(ones(n - 2), zeros(2))
        P = Matrix(Symmetric(Q * Diagonal(lam) * Q'))
        q = randn(rng, n)
        A = randn(rng, m, n)
        bu = 0.8 .* abs.(A * randn(rng, n)) .+ 0.1
        sol = solve(P, q, A, -bu, bu, ActiveSet(; eps_prox = 1.0e-4, eta_prox = 1.0e-10))
        @test sol.status == SOLVED
        @test sol.prim_res < 1.0e-8
    end
end

@testitem "an exactly singular P is refused without eps_prox" begin
    using PureDAQP, LinearAlgebra

    # Built to be singular exactly rather than by construction and rounding: a random
    # rank-deficient `P` often comes back from `Q Λ Qᵀ` with eigenvalues near ±1e-16, and
    # Cholesky then succeeds or fails depending on which side of zero they land.
    P = [1.0 0.0; 0.0 0.0]
    q = [1.0, -1.0]
    A = [1.0 0.0; 0.0 1.0]
    l = [-1.0, -1.0]
    u = [1.0, 1.0]
    @test_throws ArgumentError solve(P, q, A, l, u, ActiveSet())
    sol = solve(P, q, A, l, u, ActiveSet(; eps_prox = 1.0e-4, eta_prox = 1.0e-12))
    @test sol.status == SOLVED
    # Minimizing ½x₁² + x₁ − x₂ over the box puts x₁ at −1 and x₂ at its upper bound.
    @test sol.x ≈ [-1.0, 1.0] atol = 1.0e-6
end

@testitem "a linear program is the extreme case of a singular P" begin
    using PureDAQP, LinearAlgebra, Random

    rng = MersenneTwister(44)
    for _ in 1:15
        n, m = rand(rng, 3:7), rand(rng, 8:18)
        P = zeros(n, n)
        q = randn(rng, n)
        A = randn(rng, m, n)
        # The origin is strictly feasible, so the program is bounded.
        bu = abs.(randn(rng, m)) .+ 0.5
        bl = -abs.(randn(rng, m)) .- 0.5
        sol = solve(P, q, A, bl, bu, ActiveSet(; eps_prox = 1.0e-4, eta_prox = 1.0e-10, max_prox = 400))
        @test sol.status == SOLVED
        @test sol.prim_res < 1.0e-7
    end
end

@testitem "equality rows stay in the working set" begin
    using PureDAQP, LinearAlgebra, Random

    rng = MersenneTwister(13)
    for _ in 1:40
        n = rand(rng, 3:8)
        P = Matrix(1.0I, n, n)
        q = randn(rng, n)
        A = randn(rng, n + 2, n)
        b = A * randn(rng, n)
        l = copy(b)
        u = copy(b)
        # The first two rows are equalities; the rest are a loose box that cannot bind.
        l[3:end] .-= 10.0
        u[3:end] .+= 10.0
        sol = solve(P, q, A, l, u, ActiveSet())
        @test sol.status == SOLVED
        @test abs(dot(A[1, :], sol.x) - b[1]) < 1.0e-9
        @test abs(dot(A[2, :], sol.x) - b[2]) < 1.0e-9
    end
end
