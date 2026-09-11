@testitem "update! matches a fresh setup" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(10, 24; seed = 60)
    opts = (eps_abs = 1.0e-9, eps_rel = 1.0e-9, max_iter = 100_000)
    ws = setup(P, q, A, l, u; opts...)
    PureOSQP.solve!(ws)
    Random.seed!(61)
    for _ in 1:5
        q2 = randn(10)
        b = A * randn(10)
        l2, u2 = b .- rand(24), b .+ rand(24)
        update!(ws; q = q2, l = l2, u = u2)
        got = PureOSQP.solve!(ws)
        want = PureOSQP.solve(P, q2, A, l2, u2; opts...)
        @test got.status == SOLVED
        @test want.status == SOLVED
        @test got.x ≈ want.x rtol = 1.0e-5
        @test abs(got.obj_val - want.obj_val) <= 1.0e-6 * max(1, abs(want.obj_val))
        @test maximum(kkt_residuals(P, q2, A, l2, u2, got.x, got.y)) < 1.0e-5
    end
end

@testitem "update! refactorizes only when it must" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P = [4.0 1.0; 1.0 2.0]
    A = [1.0 1.0; 1.0 0.0]
    ws = setup(P, [1.0, 1.0], A, [0.0, 0.0], [1.0, 1.0])
    @test ws.refactor_count == 1
    # A linear-cost change touches no factorization.
    update!(ws; q = [2.0, 3.0])
    @test ws.refactor_count == 1
    # Bounds that keep every row an inequality do not either.
    update!(ws; l = [0.0, 0.1], u = [1.0, 0.9])
    @test ws.refactor_count == 1
    # Turning a row into an equality changes its rho, which is baked into the factorization.
    update!(ws; l = [0.5, 0.1], u = [0.5, 0.9])
    @test ws.refactor_count == 2
    @test ws.constr_type[1] == Int8(1)
    # Turning it back changes it again.
    update!(ws; l = [0.0, 0.1], u = [1.0, 0.9])
    @test ws.refactor_count == 3
    # Matrix updates always do.
    update!(ws; P = [5.0 1.0; 1.0 3.0])
    @test ws.refactor_count == 4
    update!(ws; A = [1.0 2.0; 1.0 0.0])
    @test ws.refactor_count == 5
end

@testitem "update! of P and A gives the same answer as a fresh setup" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(8, 20; seed = 62)
    P2, _, A2, _, _ = random_qp(8, 20; seed = 63)
    opts = (eps_abs = 1.0e-9, eps_rel = 1.0e-9, max_iter = 100_000)
    ws = setup(P, q, A, l, u; opts...)
    PureOSQP.solve!(ws)
    b = A2 * randn(8)
    l2, u2 = b .- rand(20), b .+ rand(20)
    update!(ws; P = P2, A = A2, l = l2, u = u2)
    got = PureOSQP.solve!(ws)
    want = PureOSQP.solve(P2, q, A2, l2, u2; opts...)
    @test got.status == SOLVED
    @test got.x ≈ want.x rtol = 1.0e-5
    @test maximum(kkt_residuals(P2, q, A2, l2, u2, got.x, got.y)) < 1.0e-5
end

@testitem "update! warm starts the next solve" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(12, 30; seed = 64)
    opts = (eps_abs = 1.0e-8, eps_rel = 1.0e-8, max_iter = 100_000)
    ws = setup(P, q, A, l, u; opts...)
    first = PureOSQP.solve!(ws)
    # A small perturbation should be cheap from the previous solution.
    q2 = q .+ 1.0e-3 .* randn(12)
    update!(ws; q = q2)
    second = PureOSQP.solve!(ws)
    cold = PureOSQP.solve(P, q2, A, l, u; opts...)
    @test second.status == SOLVED
    @test second.iter < cold.iter
    @test second.x ≈ cold.x rtol = 1.0e-4
end

@testitem "update! validates its arguments" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P = [4.0 1.0; 1.0 2.0]
    A = [1.0 1.0; 1.0 0.0]
    ws = setup(P, [1.0, 1.0], A, [0.0, 0.0], [1.0, 1.0])
    @test_throws "length(q) must be 2" update!(ws; q = [1.0, 2.0, 3.0])
    @test_throws "q must be finite" update!(ws; q = [NaN, 1.0])
    @test_throws "length(l) must be 2" update!(ws; l = [0.0])
    @test_throws "l must be elementwise ≤ u" update!(ws; l = [2.0, 0.0])
    @test_throws "l may not be +Inf" update!(ws; l = [Inf, 0.0])
    @test_throws "P must stay 2×2" update!(ws; P = Matrix(1.0I, 3, 3))
    @test_throws "P must be symmetric" update!(ws; P = [1.0 2.0; 0.0 1.0])
    @test_throws "not positive definite" update!(ws; P = [2.0 5.0; 5.0 1.0])
    @test_throws "A must stay 2×2" update!(ws; A = ones(3, 2))
end

@testitem "update! keeps the representation the backend was built for" begin
    using LinearAlgebra, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(70)
    # A dense workspace: a structured replacement would be solved through the structure the
    # dense backend reads, so it is refused rather than answered wrongly.
    P, q, A, l, u = random_qp(6, 10; seed = 70)
    # Tight tolerances: the answer is compared against a fresh solve below, so both sides
    # have to be converged to the same place for the comparison to mean anything.
    tight = (eps_abs = 1.0e-9, eps_rel = 1.0e-9, max_iter = 100_000)
    ws = setup(P, q, A, l, u; tight...)
    @test_throws "keep the representation" update!(ws; P = Diagonal(diag(P)))
    @test_throws "keep the representation" update!(ws; A = PureOSQP.RowCoupled(randn(4, 6), 6))
    # A same-type replacement passes and still solves what a fresh setup solves.
    P2 = P + 1.0e-3 * P
    update!(ws; P = P2)
    got = solve!(ws)
    want = solve(P2, q, A, l, u; eps_abs = 1.0e-9, eps_rel = 1.0e-9, max_iter = 100_000)
    @test got.status == SOLVED
    @test got.x ≈ want.x rtol = 1.0e-5

    # The structured backends: their storage reads only the structure they hold, so a
    # denser replacement is exactly the silent wrong answer the check exists for.
    n = 20
    Pd = Diagonal(rand(n) .+ 1)
    Ad = Diagonal(rand(n))
    qd = randn(n)
    ld, ud = -rand(n), rand(n)
    ws = setup(Pd, qd, Ad, ld, ud)
    @test PureOSQP.backend_name(ws.linsys) === :diagonal
    @test_throws "keep the representation" update!(ws; A = randn(n, n))
    X = randn(n, n)
    @test_throws "keep the representation" update!(ws; P = X'X ./ n + 2I)
end

@testitem "update! re-checks the invariants the structured backends carry" begin
    using LinearAlgebra, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    # Kronecker: a new scalar P refreshes μ and still matches a fresh setup; a general P,
    # and bounds that split the ρ classes, are refused.
    Random.seed!(71)
    n1, n2 = 8, 6
    A1, A2 = randn(n1, n1), randn(n2, n2)
    K = PureOSQP.KroneckerOperator(A1, A2)
    n = n1 * n2
    P = Diagonal(fill(2.0, n))
    q = randn(n)
    b = kron(A1, A2) * randn(n)
    l, u = b .- rand(n), b .+ rand(n)
    opts = (scaling = 0, eps_abs = 1.0e-9, eps_rel = 1.0e-9, max_iter = 100_000)
    ws = setup(P, q, K, l, u; opts...)
    @test PureOSQP.backend_name(ws.linsys) === :kronecker
    P2 = Diagonal(fill(5.0, n))
    update!(ws; P = P2)
    got = solve!(ws)
    want = solve(P2, q, K, l, u; opts...)
    @test got.status == SOLVED
    @test got.x ≈ want.x rtol = 1.0e-6
    # Still a `Diagonal`, so the representation check passes and the rung's own invariant is
    # what refuses it: the Kronecker solve needs `P` to be `μI`, not merely diagonal.
    @test_throws "scalar multiple of the identity" update!(ws; P = Diagonal(2.0 .+ rand(n)))
    l2 = copy(l)
    u2 = copy(u)
    l2[1:5] .= u2[1:5]
    @test_throws "uniform ρ" update!(ws; l = l2, u = u2)
    # A refusal leaves the workspace exactly as it was: the next solve still solves the
    # problem it held.
    again = solve!(ws)
    @test again.status == SOLVED
    @test again.x ≈ want.x rtol = 1.0e-6

    # Block: the partition and the block sizes are the invariant, not the type.
    Random.seed!(72)
    Kc, nb, mb = 4, 10, 6
    Pb = PureOSQP.BlockDiagonal(
        [
            let S = randn(nb, nb)
                Matrix(Symmetric(S'S ./ nb + 2I))
            end for _ in 1:Kc
        ]
    )
    Ab = PureOSQP.BlockDiagonal([randn(mb, nb) ./ sqrt(nb) for _ in 1:Kc])
    qb = randn(Kc * nb)
    bb = Ab * randn(Kc * nb)
    lb, ub = bb .- rand(Kc * mb), bb .+ rand(Kc * mb)
    ws = setup(Pb, qb, Ab, lb, ub)
    @test PureOSQP.backend_name(ws.linsys) === :block
    # Same type, different partition: the block runs do not line up with the storage.
    Pc = PureOSQP.BlockDiagonal(
        [
            let S = randn(2 * nb, 2 * nb)
                Matrix(Symmetric(S'S ./ (2 * nb) + 2I))
            end for _ in 1:(Kc ÷ 2)
        ]
    )
    @test_throws "block partition" update!(ws; P = Pc)

    # Low rank: the coupling rank is the invariant.
    Random.seed!(73)
    nn, kk = 40, 3
    Pn = Diagonal(rand(nn) .+ 1)
    An = PureOSQP.RowCoupled(randn(kk, nn), nn - kk)
    qn = randn(nn)
    bn = An * randn(nn)
    ln, un = bn .- rand(size(An, 1)), bn .+ rand(size(An, 1))
    ws = setup(Pn, qn, An, ln, un)
    @test PureOSQP.backend_name(ws.linsys) === :lowrank
    An2 = PureOSQP.RowCoupled(randn(kk + 2, nn), nn - kk - 2)
    @test_throws "coupling rows" update!(ws; A = An2)
end

@testitem "update! is timed and the time is charged to the next solve" begin
    using LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(20, 40; seed = 21)
    opts = (eps_abs = 1.0e-7, eps_rel = 1.0e-7, max_iter = 20_000)

    ws = setup(P, q, A, l, u; opts...)
    first = solve!(ws)
    # Nothing was updated before the first solve, and `run_time` charges setup once.
    @test first.update_time == 0.0
    @test first.run_time ≈ first.setup_time + first.update_time + first.solve_time + first.polish_time

    # Several calls before one solve all belong to it, so the time accumulates.
    update!(ws; q = q .* 1.01)
    update!(ws; l = l .- 0.01, u = u .+ 0.01)
    second = solve!(ws)
    @test second.update_time > 0.0
    # A re-solve did not pay setup again, but it did pay for the updates.
    @test second.run_time ≈ second.update_time + second.solve_time + second.polish_time
    @test second.run_time > second.solve_time

    # And the count resets, so a solve never reports another solve's updates.
    third = solve!(ws)
    @test third.update_time == 0.0
end
