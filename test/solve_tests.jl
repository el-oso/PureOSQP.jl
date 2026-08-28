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
