@testitem "the reduced solve reproduces the full KKT system" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(5)
    n, m = 9, 14
    P = (X = randn(n, n); Matrix(X'X))
    A = randn(m, n)
    ws = setup(P, randn(n), A, -rand(m), rand(m); scaling = 0, sigma = 1.0e-6, rho = 0.1)
    @test ws.linsys isa PureOSQP.ReducedCholesky
    K = [P + ws.settings.sigma * I  A'; A  -Diagonal(1 ./ ws.rho_vec)]
    bx, bz = randn(n), randn(m)
    PureOSQP.solve_system!(ws.linsys, ws, bx, bz)
    ref = K \ [bx; bz]
    @test ws.xtilde ≈ ref[1:n] rtol = 1.0e-9
    @test ws.ztilde ≈ A * ws.xtilde rtol = 1.0e-9
end

@testitem "an ill-conditioned A falls back to the full KKT backend at setup" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(6)
    n, m = 40, 120
    U = Matrix(qr(randn(m, n)).Q)[:, 1:n]
    V = Matrix(qr(randn(n, n)).Q)
    A = U * Diagonal(exp10.(range(0, 11; length = n))) * V'
    P = zeros(n, n)
    ws = setup(P, randn(n), A, -ones(m), ones(m); scaling = 0)
    @test ws.linsys isa PureOSQP.FullKKT
    bx, bz = randn(n), randn(m)
    PureOSQP.solve_system!(ws.linsys, ws, bx, bz)
    # At this conditioning a Float64 `K \ b` is no more trustworthy than the solver, so
    # the reference is computed in extended precision.
    K = [P + ws.settings.sigma * I  A'; A  -Diagonal(1 ./ ws.rho_vec)]
    ref = Float64.(big.(K) \ big.([bx; bz]))[1:n]
    @test norm(ws.xtilde .- ref, Inf) < 1.0e-4 * norm(ref, Inf)
end

@testitem "both backends reach the same solution" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(15, 40; seed = 7)
    opts = (eps_abs = 1.0e-9, eps_rel = 1.0e-9, max_iter = 100_000)
    chol = PureOSQP.solve(P, q, A, l, u; linsys = :auto, opts...)
    kkt = PureOSQP.solve(P, q, A, l, u; linsys = :kkt, opts...)
    @test chol.status == SOLVED
    @test kkt.status == SOLVED
    @test chol.x ≈ kkt.x rtol = 1.0e-5
    @test abs(chol.obj_val - kkt.obj_val) <= 1.0e-6 * max(1, abs(kkt.obj_val))
end

@testitem "linsys rejects an unknown backend" begin
    @test_throws "linsys must be :auto, :dense, :kkt or :indirect" setup([1.0;;], [0.0], [1.0;;], [0.0], [1.0]; linsys = :magic)
end

@testitem "the LinearSystem contract is enforced, not decorative" begin
    using LinearAlgebra, SparseArrays, OSQP, Random, TypeContracts
    include(joinpath(@__DIR__, "helpers.jl"))
    LS = PureOSQP.LinearSystem
    spec = TypeContracts.list_contract(LS)
    @test length(spec) == 2

    # Both shipped backends satisfy it.
    for B in (
            PureOSQP.ReducedCholesky{Float64, Matrix{Float64}},
            PureOSQP.FullKKT{
                Float64, Matrix{Float64}, Vector{Float64},
                BunchKaufman{Float64, Matrix{Float64}, Vector{Int}},
            },
        )
        @test TypeContracts.satisfies(B, LS).satisfied
    end

    # And a type that declares the supertype without implementing it is rejected. Without
    # this the contract could be satisfied vacuously and nobody would notice.
    @eval struct IncompleteBackend <: PureOSQP.LinearSystem end
    @test !TypeContracts.satisfies(IncompleteBackend, LS).satisfied
    @test length(TypeContracts.satisfies(IncompleteBackend, LS).missing_methods) == 2
    @test_throws TypeContracts.InterfaceError TypeContracts.check_contract(IncompleteBackend, LS)
end

@testitem "a sparse A forms the reduced matrix without a dense buffer" begin
    using LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(4)
    n, m = 60, 120
    A = sprandn(m, n, 0.05)
    S = sprandn(n, n, 0.05)
    P = sparse(Symmetric(S'S)) + (n * 0.05 + 1) * I
    b = A * randn(n)
    q, l, u = randn(n), b .- rand(m), b .+ rand(m)
    opts = (eps_abs = 1.0e-9, eps_rel = 1.0e-9, max_iter = 50_000)

    ws = setup(P, q, A, l, u; opts...)
    @test PureOSQP.backend_name(ws.linsys) == :sparse_formed
    # The point of the backend: no m×n buffer exists to hold a densified A.
    @test !hasproperty(ws.linsys, :W)

    # It must form the same matrix the dense product forms, so the whole run must agree
    # with the dense backend step for step -- not merely land on the same answer.
    sp = PureOSQP.solve!(ws)
    dn = solve(Matrix(P), q, Matrix(A), l, u; opts...)
    @test sp.status == SOLVED
    @test sp.iter == dn.iter
    @test sp.x ≈ dn.x atol = 1.0e-10
    @test sp.obj_val ≈ dn.obj_val atol = 1.0e-10
    @test maximum(kkt_residuals(P, q, A, l, u, sp.x, sp.y)) < 1.0e-7
end

@testitem "a dense-enough sparse A keeps the dense product" begin
    using LinearAlgebra, SparseArrays, Random
    # Accumulating over stored entries loses to `syrk` once there are enough of them, so
    # above the density threshold a SparseMatrixCSC is served by the dense-forming backend.
    Random.seed!(5)
    n, m = 40, 80
    A = sprandn(m, n, 0.6)
    P = sparse(1.0I, n, n)
    b = A * randn(n)
    q, l, u = randn(n), b .- rand(m), b .+ rand(m)
    ws = setup(P, q, A, l, u)
    @test PureOSQP.backend_name(ws.linsys) == :cholesky
    @test solve!(ws).status == SOLVED
end

@testitem "the sparse backend survives a data update" begin
    using LinearAlgebra, SparseArrays, Random
    # The transpose the accumulation walks is rebuilt per factorization rather than cached,
    # because `update!` may replace A with a different matrix entirely.
    Random.seed!(6)
    n, m = 30, 60
    A = sprandn(m, n, 0.08)
    P = sparse(2.0I, n, n)
    b = A * randn(n)
    q, l, u = randn(n), b .- rand(m), b .+ rand(m)
    opts = (eps_abs = 1.0e-9, eps_rel = 1.0e-9, max_iter = 50_000)
    A2 = sprandn(m, n, 0.08)
    b2 = A2 * randn(n)
    l2, u2 = b2 .- rand(m), b2 .+ rand(m)

    # Both workspaces run the identical sequence, because `update!` keeps the iterate and a
    # warm-started solve takes a different path from a cold one. Comparing against a fresh
    # solve would compare warm against cold rather than sparse against dense.
    sparse_ws = setup(P, q, A, l, u; opts...)
    dense_ws = setup(Matrix(P), q, Matrix(A), l, u; opts...)
    @test PureOSQP.backend_name(sparse_ws.linsys) == :sparse_formed
    for w in (sparse_ws, dense_ws)
        solve!(w)
    end
    update!(sparse_ws; A = A2, l = l2, u = u2)
    update!(dense_ws; A = Matrix(A2), l = l2, u = u2)
    s = solve!(sparse_ws)
    ref = solve!(dense_ws)

    @test s.status == SOLVED
    @test s.iter == ref.iter
    @test s.x ≈ ref.x atol = 1.0e-10
end

@testitem "a banded problem is factored sparsely" begin
    using LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    # Banded is what MPC and other structured QPs look like, and is where a sparse
    # factorization earns its place: the reduced matrix stays banded, so its Cholesky
    # factor does too. The dense backend would invert an n×n matrix instead.
    P, q, A, l, u = banded_qp(200, 400; band = 3)
    opts = (eps_abs = 1.0e-9, eps_rel = 1.0e-9, max_iter = 50_000)

    ws = setup(P, q, A, l, u; opts...)
    @test PureOSQP.backend_name(ws.linsys) == :cholmod

    sp = PureOSQP.solve!(ws)
    dn = solve(Matrix(P), q, Matrix(A), l, u; opts...)
    @test sp.status == SOLVED
    # A different factorization of the same matrix, so the path must be identical, not
    # merely the answer.
    @test sp.iter == dn.iter
    @test sp.x ≈ dn.x atol = 1.0e-10
    @test sp.obj_val ≈ dn.obj_val atol = 1.0e-10
    @test maximum(kkt_residuals(P, q, A, l, u, sp.x, sp.y)) < 1.0e-7
end

@testitem "the sparse factorization solve allocates nothing" begin
    using LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    # CHOLMOD's own `ldiv!` allocates a result and workspace on every call, which the hot
    # path may not do, so the backend applies the permutation and the two triangular solves
    # itself over buffers it owns.
    P, q, A, l, u = banded_qp(200, 400; band = 2)
    ws = setup(P, q, A, l, u; eps_abs = 1.0e-9, eps_rel = 1.0e-9)
    @test PureOSQP.backend_name(ws.linsys) == :cholmod
    PureOSQP.solve!(ws)
    PureOSQP.solve_system!(ws.linsys, ws, ws.rhs_x, ws.rhs_z)      # warm up
    allocs = [(@allocated PureOSQP.solve_system!(ws.linsys, ws, ws.rhs_x, ws.rhs_z)) for _ in 1:4]
    @test all(iszero, allocs)
end

@testitem "a factor that fills in is left to the dense backend" begin
    using LinearAlgebra, SparseArrays, Random
    # Random sparsity has no separators, so the Cholesky factor fills in almost completely
    # and the dense inverse wins the per-iteration solve. The backend is chosen by asking
    # CHOLMOD what the fill actually is, so this is decided rather than assumed.
    Random.seed!(11)
    n, m = 150, 300
    A = sprandn(m, n, 0.05)
    S = sprandn(n, n, 0.05)
    P = sparse(Symmetric(S'S)) + (n * 0.05 + 1) * I
    b = A * randn(n)
    ws = setup(P, randn(n), A, b .- rand(m), b .+ rand(m))
    @test PureOSQP.backend_name(ws.linsys) == :sparse_formed
    @test solve!(ws).status == SOLVED
end

@testitem "the sparse factorization survives a change of sparsity pattern" begin
    using LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    # Refactorization reuses the symbolic analysis, which is only valid while the pattern
    # holds. `update!` may replace A with one shaped differently, and then it must not.
    P, q, A, l, u = banded_qp(200, 400; band = 2)
    P2, _, A2, l2, u2 = banded_qp(200, 400; band = 3, seed = 99)
    opts = (eps_abs = 1.0e-9, eps_rel = 1.0e-9, max_iter = 50_000)

    sparse_ws = setup(P, q, A, l, u; opts...)
    dense_ws = setup(Matrix(P), q, Matrix(A), l, u; opts...)
    @test PureOSQP.backend_name(sparse_ws.linsys) == :cholmod
    for w in (sparse_ws, dense_ws)
        solve!(w)
    end
    update!(sparse_ws; A = A2, l = l2, u = u2)
    update!(dense_ws; A = Matrix(A2), l = l2, u = u2)
    s = solve!(sparse_ws)
    ref = solve!(dense_ws)

    @test s.status == SOLVED
    @test s.iter == ref.iter
    @test s.x ≈ ref.x atol = 1.0e-10
end

@testitem "linsys = :dense overrules the representation gates" begin
    using LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    # Both gates for a sparse A are measured thresholds, so there has to be a way to
    # overrule one that misjudges a problem. `:dense` goes past `choose_backend` entirely.
    P, q, A, l, u = banded_qp(200, 400; band = 3)
    opts = (eps_abs = 1.0e-9, eps_rel = 1.0e-9, max_iter = 50_000)

    @test PureOSQP.backend_name(setup(P, q, A, l, u; opts...).linsys) == :cholmod
    forced = setup(P, q, A, l, u; opts..., linsys = :dense)
    @test PureOSQP.backend_name(forced.linsys) == :cholesky

    # Overruling the choice must change only the route, not the answer.
    s = solve!(forced)
    ref = solve(P, q, A, l, u; opts...)
    @test s.status == SOLVED
    @test s.iter == ref.iter
    @test s.x ≈ ref.x atol = 1.0e-9
end

@testitem "a dense row in A routes to the full KKT" begin
    using LinearAlgebra, SparseArrays, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    # Eliminating to the reduced system squares A, so one dense row makes the reduced matrix
    # dense however sparse the rest of it is. The full KKT keeps that row as one sparse row.
    # This is the OSQP suite's Portfolio shape: a budget constraint over every variable.
    Random.seed!(12)
    n, k = 150, 3
    F = sprandn(n, k, 0.5)
    P = blockdiag(spdiagm(0 => rand(n) .+ 1), sparse(2.0I, k, k))
    q = vcat(randn(n), zeros(k))
    A = vcat(
        hcat(sparse(ones(1, n)), spzeros(1, k)),
        hcat(sparse(F'), sparse(-1.0I, k, k)),
        hcat(sparse(1.0I, n, n), spzeros(n, k)),
    )
    l = vcat(1.0, zeros(k), zeros(n))
    u = vcat(1.0, zeros(k), ones(n))
    opts = (eps_abs = 1.0e-8, eps_rel = 1.0e-8, max_iter = 50_000)

    ws = setup(P, q, A, l, u; opts...)
    @test PureOSQP.backend_name(ws.linsys) == :sparse_kkt
    s = PureOSQP.solve!(ws)
    @test s.status == SOLVED
    @test maximum(kkt_residuals(P, q, A, l, u, s.x, s.y)) < 1.0e-6

    # A different factorization of the same system, so it must agree with the dense one.
    ref = solve(P, q, A, l, u; opts..., linsys = :dense)
    @test ref.status == SOLVED
    @test s.obj_val ≈ ref.obj_val rtol = 1.0e-7
    @test s.x ≈ ref.x atol = 1.0e-6
end

@testitem "the full-KKT backend survives a pattern change" begin
    using LinearAlgebra, SparseArrays, Random
    # Refactorization reuses the symbolic analysis, which holds only while the pattern does.
    # `n` is large enough that the full KKT is the better form: below about n = 100 a dense
    # `symv` on the reduced system beats a sparse triangular solve on one of size n + m,
    # and the fill gate says so rather than taking the sparse route regardless.
    Random.seed!(13)
    n = 250
    P = spdiagm(0 => rand(n) .+ 1)
    q = randn(n)
    dense_row = sparse(ones(1, n))
    A = vcat(dense_row, sparse(1.0I, n, n))
    l = vcat(1.0, zeros(n))
    u = vcat(1.0, ones(n))
    opts = (eps_abs = 1.0e-8, eps_rel = 1.0e-8, max_iter = 50_000)

    ws = setup(P, q, A, l, u; opts...)
    @test PureOSQP.backend_name(ws.linsys) == :sparse_kkt
    @test solve!(ws).status == SOLVED

    A2 = vcat(dense_row, sparse(2.0I, n, n))
    update!(ws; A = A2)
    s = solve!(ws)
    @test s.status == SOLVED
    ref = solve(P, q, A2, l, u; opts..., linsys = :dense)
    @test s.obj_val ≈ ref.obj_val rtol = 1.0e-7
end
