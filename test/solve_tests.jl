@testitem "solves the reference QP" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P = [4.0 1.0; 1.0 2.0]
    q = [1.0, 1.0]
    A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
    l = [1.0, 0.0, 0.0]
    u = [1.0, 0.7, 0.7]
    s = PureOSQP.solve(P, q, A, l, u; eps_abs = 1.0e-9, eps_rel = 1.0e-9, max_iter = 20_000)
    @test s.status == SOLVED
    @test maximum(kkt_residuals(P, q, A, l, u, s.x, s.y)) < 1.0e-6
    @test s.obj_val ≈ 1.88 atol = 1.0e-6
end

@testitem "the referee accepts every solved random instance" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    solved = Ref(0)
    for trial in 1:30
        Random.seed!(100 + trial)
        n, m = rand(3:15), rand(1:30)
        P = (X = randn(n, n); Matrix(X'X))
        A = randn(m, n) * Diagonal(exp10.(rand(-3:3, n)))
        q = randn(n)
        l = -rand(m) .- 0.1
        u = rand(m) .+ 0.1
        s = PureOSQP.solve(P, q, A, l, u; eps_abs = 1.0e-9, eps_rel = 1.0e-9, max_iter = 100_000)
        s.status == SOLVED || continue
        solved[] += 1
        @test maximum(kkt_residuals(P, q, A, l, u, s.x, s.y)) < 1.0e-5
    end
    @test solved[] >= 25
end

@testitem "equilibration is what makes a badly scaled problem tractable" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(10, 25; seed = 12, colscale = 4)
    off = PureOSQP.solve(
        P, q, A, l, u; scaling = 0, eps_abs = 1.0e-10, eps_rel = 1.0e-10,
        max_iter = 200_000
    )
    on = PureOSQP.solve(
        P, q, A, l, u; scaling = 10, eps_abs = 1.0e-10, eps_rel = 1.0e-10,
        max_iter = 200_000
    )
    @test on.status == SOLVED
    @test maximum(kkt_residuals(P, q, A, l, u, on.x, on.y)) < 1.0e-6
    # Without equilibration the same problem does not converge in 200_000 iterations.
    @test off.status == MAX_ITER_REACHED
end

@testitem "hitting max_iter is never reported as solved" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(30, 60; seed = 13)
    s = PureOSQP.solve(P, q, A, l, u; max_iter = 2, eps_abs = 1.0e-14, eps_rel = 1.0e-14)
    @test s.status == MAX_ITER_REACHED
end

@testitem "adaptive rho reduces the iteration count from a bad rho" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(25, 60; seed = 14)
    fixed = PureOSQP.solve(
        P, q, A, l, u; adaptive_rho = false, rho = 1.0e-4,
        eps_abs = 1.0e-8, eps_rel = 1.0e-8, max_iter = 100_000
    )
    adapt = PureOSQP.solve(
        P, q, A, l, u; adaptive_rho = true, rho = 1.0e-4,
        eps_abs = 1.0e-8, eps_rel = 1.0e-8, max_iter = 100_000
    )
    @test adapt.status == SOLVED
    @test adapt.iter < fixed.iter
end

@testitem "rho vector classifies equalities, inequalities and free rows" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    A = [1.0 0.0; 0.0 1.0; 1.0 1.0]
    l = [1.0, -Inf, 0.0]
    u = [1.0, Inf, 1.0]
    ws = setup(zeros(2, 2), zeros(2), A, l, u; scaling = 0, rho = 0.1)
    @test ws.rho_vec[1] ≈ 1.0e3 * 0.1
    @test ws.rho_vec[2] ≈ 1.0e-6
    @test ws.rho_vec[3] ≈ 0.1
    @test ws.rho_inv_vec ≈ 1 ./ ws.rho_vec
    @test ws.constr_type == Int8[1, -1, 0]
end

@testitem "warm starting from the solution cuts the iteration count" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(12, 30; seed = 15)
    cold = PureOSQP.solve(P, q, A, l, u; eps_abs = 1.0e-8, eps_rel = 1.0e-8, max_iter = 100_000)
    ws = setup(P, q, A, l, u; eps_abs = 1.0e-8, eps_rel = 1.0e-8, max_iter = 100_000)
    warm_start!(ws; x = cold.x, y = cold.y)
    warm = PureOSQP.solve!(ws)
    @test warm.status == SOLVED
    @test warm.iter < cold.iter
    @test warm.x ≈ cold.x rtol = 1.0e-5
end

@testitem "verbose prints a progress report, and is silent when off" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(12, 30; seed = 21)

    # `Core.stdout` writes to the file descriptor, so capture at that level rather than by
    # rebinding `Base.stdout`.
    function capture(f)
        (path, io) = mktemp()
        try
            redirect_stdout(f, io)
            close(io)
            return read(path, String)
        finally
            rm(path; force = true)
        end
    end

    loud = capture() do
        PureOSQP.solve(P, q, A, l, u; verbose = true, check_termination = 25)
    end
    quiet = capture() do
        PureOSQP.solve(P, q, A, l, u; verbose = false, check_termination = 25)
    end

    # The defect this guards against is a setting that is accepted and then ignored.
    @test isempty(quiet)
    @test !isempty(loud)
    @test occursin("PureOSQP", loud)
    @test occursin("iter", loud)
    @test occursin("status:", loud)
    @test occursin("solved", loud)
    @test occursin("number of iterations:", loud)
    # One row per termination check, plus the header and footer blocks.
    sol = PureOSQP.solve(P, q, A, l, u; check_termination = 25)
    @test count(==('\n'), loud) >= sol.iter ÷ 25

    # Polishing reports its own outcome, and only when it was asked for.
    polished = capture() do
        PureOSQP.solve(P, q, A, l, u; verbose = true, polish = true)
    end
    @test occursin("polish:", polished)
    @test !occursin("polish:", loud)
end

@testitem "cold_start! discards the warm start without touching the problem" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(12, 30; seed = 33)
    opts = (eps_abs = 1.0e-8, eps_rel = 1.0e-8, max_iter = 100_000)

    ws = setup(P, q, A, l, u; opts...)
    first = PureOSQP.solve!(ws)
    @test first.status == SOLVED

    warm = PureOSQP.solve!(ws)               # warm started from `first`
    @test warm.iter < first.iter

    # Only the iterates are discarded. `ρ`, the equilibration factors and the
    # factorization survive, as they do upstream, so cold starting costs no refactorization.
    refac = ws.refactor_count
    rho = copy(ws.rho_vec)
    @test cold_start!(ws) === ws
    @test all(iszero, ws.x)
    @test all(iszero, ws.y)
    @test all(iszero, ws.z)
    @test ws.refactor_count == refac
    @test ws.rho_vec == rho

    cold = PureOSQP.solve!(ws)
    @test cold.status == SOLVED
    # The warm start really was thrown away: restarting from the origin takes more
    # iterations than continuing from the previous solution did.
    @test cold.iter > warm.iter
    # It converges to the same point regardless. Note the iteration count need not match
    # the very first solve: `ρ` has been adapted since, and cold starting does not undo that.
    @test cold.x ≈ first.x rtol = 1.0e-6
    @test cold.obj_val ≈ first.obj_val rtol = 1.0e-8

    @test :cold_start! in names(PureOSQP)
end

@testitem "time_limit stops the loop and reports it" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    # Tight tolerances on a problem that needs many iterations, so the limit binds well
    # before convergence rather than racing it.
    P, q, A, l, u = random_qp(60, 150; seed = 44)
    opts = (eps_abs = 1.0e-12, eps_rel = 1.0e-12, max_iter = 200_000)

    unlimited = PureOSQP.solve(P, q, A, l, u; opts...)
    limited = PureOSQP.solve(P, q, A, l, u; opts..., time_limit = 0.01)

    @test limited.status == TIME_LIMIT_REACHED
    @test limited.iter < unlimited.iter
    # The point is unconverged but meaningful: it is reported, not replaced with NaN.
    @test PureOSQP.has_solution(limited.status)
    @test all(isfinite, limited.x)
    @test all(isfinite, limited.y)
    @test isfinite(limited.obj_val)
    @test limited.prim_res >= 0

    # Wall clock is only bounded loosely from above -- one iteration can overshoot -- but a
    # 10 ms budget must not take anything like the unlimited run.
    t = @elapsed PureOSQP.solve(P, q, A, l, u; opts..., time_limit = 0.01)
    @test t < 1.0

    # A limit that cannot bind leaves the result identical to no limit at all.
    generous = PureOSQP.solve(P, q, A, l, u; opts..., time_limit = 1000.0)
    @test generous.status == unlimited.status
    @test generous.iter == unlimited.iter

    @test_throws "time_limit must be positive" PureOSQP.solve(P, q, A, l, u; time_limit = 0)
    @test_throws "time_limit must be positive" PureOSQP.solve(P, q, A, l, u; time_limit = -1)
end

@testitem "the reported duality gap and objectives are consistent and real" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(20, 45; seed = 55)
    opts = (eps_abs = 1.0e-10, eps_rel = 1.0e-10, max_iter = 100_000)
    s = PureOSQP.solve(P, q, A, l, u; opts...)
    @test s.status == SOLVED

    # gap == primal - dual is an identity of the definitions, so it must hold tightly.
    @test s.duality_gap ≈ s.obj_val - s.dual_obj_val atol = 1.0e-9

    # The primal objective, recomputed from the original data rather than the workspace's
    # scaled copies. This is what catches an unscaling error.
    @test s.obj_val ≈ 0.5 * dot(s.x, P * s.x) + dot(q, s.x) rtol = 1.0e-6

    # Strong duality at a converged point.
    @test abs(s.duality_gap) < 1.0e-5
    @test s.obj_val ≈ s.dual_obj_val atol = 1.0e-5

    @test s.rel_kkt_error ≈ max(s.prim_res, s.dual_res, abs(s.duality_gap))

    # And it is measuring something: stop far short of convergence and the gap is wide.
    early = PureOSQP.solve(P, q, A, l, u; eps_abs = 1.0e-10, eps_rel = 1.0e-10, max_iter = 5)
    @test early.status == MAX_ITER_REACHED
    @test abs(early.duality_gap) > abs(s.duality_gap)
    @test early.rel_kkt_error > s.rel_kkt_error
end

@testitem "polish reports which outcome it had, not just success" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(12, 30; seed = 15)

    # Not asked for.
    off = PureOSQP.solve(P, q, A, l, u; eps_abs = 1.0e-9, eps_rel = 1.0e-9)
    @test off.status_polish == POLISH_NOT_PERFORMED
    @test !off.polished

    on = PureOSQP.solve(P, q, A, l, u; eps_abs = 1.0e-9, eps_rel = 1.0e-9, polish = true)
    @test on.status_polish == POLISH_SUCCESS
    @test on.polished

    # An unconstrained problem has no active set, which is a decline rather than a failure.
    n = 5
    X = randn(MersenneTwister(3), n, n)
    Pu = Matrix(X'X + I)
    free = PureOSQP.solve(
        Pu, randn(MersenneTwister(4), n), zeros(0, n), Float64[], Float64[];
        polish = true, eps_abs = 1.0e-9, eps_rel = 1.0e-9
    )
    @test free.status == SOLVED
    @test free.status_polish == POLISH_NO_ACTIVE_SET_FOUND
    @test !free.polished
    # `polished` stays the narrow question "were the iterates replaced".
    @test free.polished == (free.status_polish == POLISH_SUCCESS)
end

@testitem "rho_updates counts adaptations, separately from refactorizations" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(25, 60; seed = 14)
    opts = (eps_abs = 1.0e-8, eps_rel = 1.0e-8, max_iter = 100_000)

    fixed = PureOSQP.solve(P, q, A, l, u; adaptive_rho = false, rho = 1.0e-4, opts...)
    @test fixed.rho_updates == 0

    adapt = PureOSQP.solve(P, q, A, l, u; adaptive_rho = true, rho = 1.0e-4, opts...)
    @test adapt.rho_updates > 0
    @test adapt.rho_estimate > 0

    # The counter is per-run; the workspace's refactor_count is cumulative and also counts
    # the factorization done at setup, so the two must not be confused.
    ws = setup(P, q, A, l, u; adaptive_rho = true, rho = 1.0e-4, opts...)
    first = PureOSQP.solve!(ws)
    second = PureOSQP.solve!(ws)
    @test second.rho_updates <= first.rho_updates
    @test ws.refactor_count >= first.rho_updates + second.rho_updates
end

@testitem "timings are reported and add up" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(30, 70; seed = 66)
    s = PureOSQP.solve(P, q, A, l, u; eps_abs = 1.0e-9, eps_rel = 1.0e-9, polish = true)
    @test s.setup_time > 0
    @test s.solve_time > 0
    @test s.polish_time > 0
    @test s.run_time ≈ s.setup_time + s.solve_time + s.polish_time

    # `setup_time` belongs to the workspace, so every solve reports it, but only the first
    # charges it to `run_time` -- a re-solve did not pay it again.
    ws = setup(P, q, A, l, u; eps_abs = 1.0e-9, eps_rel = 1.0e-9)
    first = PureOSQP.solve!(ws)
    again = PureOSQP.solve!(ws)
    @test first.setup_time > 0
    @test again.setup_time == first.setup_time
    @test first.run_time ≈ first.setup_time + first.solve_time + first.polish_time
    @test again.run_time ≈ again.solve_time + again.polish_time
    @test again.solve_time > 0
    @test iszero(again.polish_time)    # polishing was not requested
end

@testitem "the duality-gap test is a real extra condition" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    opts = (eps_abs = 1.0e-6, eps_rel = 1.0e-6, max_iter = 100_000)

    # The gap check must bind somewhere: if it never changed an outcome, defaulting it on
    # would be cost without effect. It binds rarely on well-scaled problems -- the
    # residuals usually imply the gap -- so this sweeps badly scaled objectives, where the
    # gap tolerance and the residual tolerances are set by different quantities.
    bound = Ref(0)
    for trial in 1:25
        Random.seed!(900 + trial)
        n, m = rand(3:12), rand(4:20)
        X = randn(n, n)
        P = Matrix(X'X) * 10.0^rand(-3:3)
        A = randn(m, n) * 10.0^rand(-2:2)
        q = randn(n) * 10.0^rand(-3:3)
        b = A * randn(n)
        l, u = b .- rand(m), b .+ rand(m)
        with = PureOSQP.solve(P, q, A, l, u; opts..., check_dualgap = true)
        without = PureOSQP.solve(P, q, A, l, u; opts..., check_dualgap = false)
        (with.status == SOLVED && without.status == SOLVED) || continue

        # It can only delay convergence, never declare it early.
        @test with.iter >= without.iter
        with.iter > without.iter && (bound[] += 1)

        # Both are genuine solutions; the gap-checked one is no worse. The tolerance is
        # loose because these objectives are deliberately scaled by up to 1e3, and the
        # referee's measure is relative to the problem, not to `eps_abs`. Tight-tolerance
        # refereeing on well-scaled data is covered by its own test.
        @test maximum(kkt_residuals(P, q, A, l, u, with.x, with.y)) < 1.0e-2
        @test abs(with.duality_gap) <= max(abs(without.duality_gap), 1.0e-8)
    end
    @test bound[] > 0

    # And the setting is honoured rather than ignored: with the residual tolerances wide
    # open and the gap tolerance doing the work, the two disagree on when to stop.
    P, q, A, l, u = random_qp(15, 40; seed = 731)
    loose = (eps_abs = 1.0e-3, eps_rel = 1.0e-3, max_iter = 100_000)
    a = PureOSQP.solve(P, q, A, l, u; loose..., check_dualgap = true)
    b = PureOSQP.solve(P, q, A, l, u; loose..., check_dualgap = false)
    @test a.iter >= b.iter
    @test abs(a.duality_gap) <= abs(b.duality_gap) + 1.0e-8
end

@testitem "scaled_termination and rho_is_vec change what they claim to" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(12, 30; seed = 77, colscale = 3)
    opts = (eps_abs = 1.0e-8, eps_rel = 1.0e-8, max_iter = 100_000)

    # Judging termination in scaled space is a different test, so it may stop elsewhere;
    # either way the answer must still be a solution of the original problem.
    unscaled = PureOSQP.solve(P, q, A, l, u; opts..., scaled_termination = false)
    scaled = PureOSQP.solve(P, q, A, l, u; opts..., scaled_termination = true)
    @test unscaled.status == SOLVED
    @test scaled.status == SOLVED
    @test maximum(kkt_residuals(P, q, A, l, u, scaled.x, scaled.y)) < 1.0e-4
    @test scaled.x ≈ unscaled.x rtol = 1.0e-3

    # `rho_is_vec = false` drops the equality/inequality split, so every row shares one ρ.
    A2 = [1.0 0.0; 0.0 1.0; 1.0 1.0]
    l2 = [1.0, -Inf, 0.0]
    u2 = [1.0, Inf, 1.0]
    split = setup(zeros(2, 2), zeros(2), A2, l2, u2; scaling = 0, rho = 0.1)
    flat = setup(
        zeros(2, 2), zeros(2), A2, l2, u2; scaling = 0, rho = 0.1, rho_is_vec = false
    )
    @test length(unique(split.rho_vec)) == 3     # equality, free, inequality
    @test all(≈(0.1), flat.rho_vec)
    @test all(iszero, flat.constr_type)
end

@testitem "an interrupt returns the point reached, other exceptions propagate" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    # A matrix that throws once it has been read a set number of times. That is how a
    # Ctrl-C in the middle of the ADMM loop reaches `solve!`, without needing a signal --
    # `mul_A!` reads `A` every iteration, so the throw lands inside the loop.
    mutable struct Fuse{T} <: AbstractMatrix{T}
        A::Matrix{T}
        n::Int
        interrupt::Bool
    end
    Base.size(F::Fuse) = size(F.A)
    function Base.getindex(F::Fuse, i::Int, j::Int)
        F.n -= 1
        iszero(F.n) && throw(F.interrupt ? InterruptException() : ErrorException("boom"))
        return F.A[i, j]
    end

    P, q, A, l, u = random_qp(8, 16; seed = 3)

    F = Fuse(A, typemax(Int), true)
    ws = setup(P, q, F, l, u; max_iter = 100_000, check_termination = 0)
    F.n = 20 * length(A)                  # arm it a few iterations into the loop
    sol = PureOSQP.solve!(ws)
    @test sol.status == INTERRUPTED
    @test PureOSQP.has_solution(sol.status)
    @test 0 < sol.iter < 100_000
    # The point is unconverged but real: reported, not replaced with NaN.
    @test all(isfinite, sol.x)
    @test all(isfinite, sol.y)
    @test isfinite(sol.prim_res)
    @test isfinite(sol.obj_val)

    # Anything that is not an interrupt is a bug, and must not be swallowed as a status.
    G = Fuse(A, typemax(Int), false)
    ws2 = setup(P, q, G, l, u; max_iter = 100_000, check_termination = 0)
    G.n = 20 * length(A)
    @test_throws "boom" PureOSQP.solve!(ws2)
end

@testitem "adaptive_rho takes a mode, and a Bool still means what it names" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(20, 60; seed = 7)
    opts = (eps_abs = 1.0e-8, eps_rel = 1.0e-8, max_iter = 100_000)

    off = PureOSQP.solve(P, q, A, l, u; opts..., adaptive_rho = false)
    dis = PureOSQP.solve(P, q, A, l, u; opts..., adaptive_rho = :disabled)
    on = PureOSQP.solve(P, q, A, l, u; opts..., adaptive_rho = true)
    its = PureOSQP.solve(P, q, A, l, u; opts..., adaptive_rho = :iterations)
    kkt = PureOSQP.solve(P, q, A, l, u; opts..., adaptive_rho = :kkt_error)

    # The Bool is the old spelling of two of the modes, so they must agree exactly.
    @test off.iter == dis.iter
    @test iszero(dis.rho_updates)
    @test on.iter == its.iter
    @test on.rho_updates > 0
    @test kkt.status == SOLVED

    # A fraction that can never be met means the gate never opens, so rho never moves --
    # which proves the gate is consulted rather than ignored.
    never = PureOSQP.solve(
        P, q, A, l, u; opts..., adaptive_rho = :kkt_error, adaptive_rho_fraction = 1.0e-300
    )
    @test iszero(never.rho_updates)
    @test never.iter == off.iter        # identical to not adapting at all

    @test_throws "adaptive_rho must be" PureOSQP.solve(P, q, A, l, u; adaptive_rho = :nope)
    @test_throws "adaptive_rho_fraction must lie in (0, 1]" PureOSQP.solve(
        P, q, A, l, u; adaptive_rho_fraction = 0.0
    )
end

@testitem "update_settings! refactorizes only for what the factorization contains" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(15, 40; seed = 91)
    ws = setup(P, q, A, l, u; eps_abs = 1.0e-6, eps_rel = 1.0e-6)

    # A tolerance is not in the matrix, so changing it is free.
    before = ws.refactor_count
    update_settings!(ws; eps_abs = 1.0e-9, max_iter = 50_000)
    @test ws.settings.eps_abs == 1.0e-9
    @test ws.settings.max_iter == 50_000
    @test ws.settings.eps_rel == 1.0e-6          # untouched fields survive
    @test ws.refactor_count == before

    # `rho` and `sigma` are in it, so changing either must refactorize.
    update_settings!(ws; rho = 0.5)
    @test ws.refactor_count > before
    @test ws.rho ≈ 0.5
    mid = ws.refactor_count
    update_settings!(ws; sigma = 1.0e-5)
    @test ws.refactor_count > mid

    # Rejected, not silently ignored: the backend is part of the workspace's type, and the
    # equilibration factors were computed once from the data setup saw.
    @test_throws "linsys is fixed" update_settings!(ws; linsys = :kkt)
    @test_throws "scaling is fixed" update_settings!(ws; scaling = 0)
    # A rejected call leaves the workspace alone.
    @test ws.settings.linsys === :auto
    @test_throws "eps_abs and eps_rel must be non-negative" update_settings!(ws; eps_abs = -1)

    @test PureOSQP.solve!(ws).status == SOLVED
end

@testitem "update_rho! is the solver's own rho change, exposed" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    A = [1.0 0.0; 0.0 1.0; 1.0 1.0]
    l = [1.0, -Inf, 0.0]
    u = [1.0, Inf, 1.0]
    ws = setup(zeros(2, 2), zeros(2), A, l, u; scaling = 0, rho = 0.1)
    before = ws.refactor_count

    @test update_rho!(ws, 0.7) === ws
    @test ws.refactor_count > before
    @test ws.rho ≈ 0.7
    # Split across the constraint classes exactly as adaptive rho does.
    @test ws.rho_vec[1] ≈ 1.0e3 * 0.7      # equality
    @test ws.rho_vec[2] ≈ 1.0e-6           # free row
    @test ws.rho_vec[3] ≈ 0.7              # inequality

    # Clamped to the solver's band rather than accepted as given.
    update_rho!(ws, 1.0e12)
    @test ws.rho ≈ 1.0e6
    @test_throws "rho must be positive" update_rho!(ws, 0.0)
end

@testitem "the introspection surface reports what this build does" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(6, 14; seed = 5)
    ws = setup(P, q, A, l, u)
    @test dimensions(ws) == (6, 14)

    c = capabilities()
    @test c.direct_solver
    @test c.update_matrices
    @test c.derivatives
    # Claimed only where true: the matrix-free backend and C code generation are the
    # roadmap items still open.
    @test !c.indirect_solver
    @test !c.codegen
end
