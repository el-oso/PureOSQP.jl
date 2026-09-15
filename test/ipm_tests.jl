@testitem "interior point: the structural corpus passes the referee on both backends" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(30)
    n = 8
    Xn = randn(n, n)
    Pfull = Matrix(Xn'Xn + I)
    Prank = (Y = randn(n, 3); Matrix(Y * Y'))
    Abase = randn(12, n)
    xf = randn(n)
    b = Abase * xf
    cases = Any[
        ("dense PSD", Pfull, randn(n), Abase, b .- rand(12), b .+ rand(12)),
        ("rank-deficient P", Prank, randn(n), Abase, b .- rand(12), b .+ rand(12)),
        ("LP, P = 0", zeros(n, n), randn(n), Abase, b .- rand(12), b .+ rand(12)),
        ("equalities", Pfull, randn(n), Abase[1:4, :], Abase[1:4, :] * xf, Abase[1:4, :] * xf),
        (
            "fixed variable", Pfull, randn(n), Matrix(1.0I, n, n),
            [xf[1]; fill(-5.0, n - 1)], [xf[1]; fill(5.0, n - 1)],
        ),
        ("upper bounds only", Pfull, randn(n), Abase, fill(-Inf, 12), b .+ rand(12)),
        ("lower bounds only", Pfull, randn(n), Abase, b .- rand(12), fill(Inf, 12)),
        ("free rows", Pfull, randn(n), Abase, fill(-Inf, 12), fill(Inf, 12)),
        (
            "mixed infinite bounds", Pfull, randn(n), Abase,
            [iszero(i % 2) ? -Inf : b[i] - rand() for i in 1:12],
            [iszero(i % 3) ? Inf : b[i] + rand() for i in 1:12],
        ),
        (
            "m < n", Pfull, randn(n), Abase[1:3, :],
            Abase[1:3, :] * xf .- 0.5, Abase[1:3, :] * xf .+ 0.5,
        ),
        ("m = 0", Pfull, randn(n), zeros(0, n), Float64[], Float64[]),
        ("n = 1", reshape([2.0], 1, 1), [1.0], reshape([1.0], 1, 1), [-1.0], [1.0]),
        ("diagonal P", Matrix(Diagonal(rand(n) .+ 1)), randn(n), Abase, b .- rand(12), b .+ rand(12)),
        ("Symmetric P", Symmetric(Pfull), randn(n), Abase, b .- rand(12), b .+ rand(12)),
        (
            "A::SubArray", Pfull, randn(n), view(Abase, 1:6, 1:n),
            b[1:6] .- rand(6), b[1:6] .+ rand(6),
        ),
        ("A::SparseMatrixCSC", Pfull, randn(n), sparse(Abase), b .- rand(12), b .+ rand(12)),
    ]
    for backend in (:auto, :kkt)
        for (name, P, q, A, l, u) in cases
            ws = setup(P, q, A, l, u; algorithm = :ipm, linsys = backend)
            @test ws isa IPMWorkspace
            # Every pair here is dense or has a dense P, which the interior-point ladder
            # serves with the full KKT factorization.
            @test PureOSQP.backend_name(ws.linsys) === :bunchkaufman
            s = solve!(ws)
            @test s.status == SOLVED
            @test s.iter <= 30
            r = maximum(kkt_residuals(Matrix(P), q, Matrix(A), l, u, s.x, s.y))
            @test r < 1.0e-5
            ref = PureOSQP.solve(
                P, q, A, l, u; eps_abs = 1.0e-9, eps_rel = 1.0e-9,
                max_iter = 200_000, polishing = true
            )
            @test abs(s.obj_val - ref.obj_val) <= 1.0e-6 * max(1, abs(ref.obj_val))
        end
    end
end

@testitem "interior point: objective agrees with the C library" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    for (n, m, seed) in ((8, 12, 41), (12, 5, 42), (6, 40, 43), (20, 20, 44), (40, 15, 55), (100, 150, 250))
        P, q, A, l, u = random_qp(n, m; seed)
        c = osqp_ref(
            P, q, A, l, u; eps_abs = 1.0e-9, eps_rel = 1.0e-9,
            max_iter = 100_000, polish = true
        )
        s = PureOSQP.solve(P, q, A, l, u; algorithm = :ipm)
        @test s.status == SOLVED
        @test s.iter <= 30
        @test maximum(kkt_residuals(P, q, A, l, u, s.x, s.y)) < 1.0e-5
        @test abs(s.obj_val - c.info.obj_val) <= 1.0e-6 * max(1, abs(c.info.obj_val))
    end
end

@testitem "interior point: sparse pairs try the KKT factorization first" begin
    using LinearAlgebra, SparseArrays, OSQP, Random, LDLFactorizations
    include(joinpath(@__DIR__, "helpers.jl"))
    n = 400
    P, q, A0, _, _ = banded_qp(n, n ÷ 2; band = 2)
    # Box rows on every variable keep the LPs bounded, and the rows of `A0` are built around a
    # point inside the box so the problem stays feasible.
    Random.seed!(82)
    b0 = A0 * (rand(n) .- 0.5)
    A = [A0; sparse(1.0I, n, n)]
    l, u = [b0 .- rand(n ÷ 2); fill(-1.0, n)], [b0 .+ rand(n ÷ 2); fill(1.0, n)]
    cases = [
        # An LP on sparse data: the sparse KKT factor clears the fill gate.
        ("sparse LP", spzeros(n, n), A, SPARSE_KKT_BACKENDS),
        # The sparse rungs need a sparse P, so these reach the dense terminal.
        ("dense P, sparse A", Matrix(P), A, (:bunchkaufman,)),
        ("dense LP", zeros(n, n), Matrix(A), (:bunchkaufman,)),
    ]
    for (name, Pc, Ac, backends) in cases
        ws = setup(Pc, q, Ac, l, u; algorithm = :ipm)
        @test PureOSQP.backend_name(ws.linsys) in backends
        s = solve!(ws)
        @test s.status == SOLVED
        @test s.iter <= 30
        @test maximum(kkt_residuals(Matrix(Pc), q, Matrix(Ac), l, u, s.x, s.y)) < 1.0e-5
    end
end

@testitem "interior point: equality, one-sided and free rows" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(81)
    n, m = 40, 60
    X = randn(n, n)
    P = X'X / n + I
    A = randn(m, n)
    b = A * randn(n)
    l, u = b .- rand(m), b .+ rand(m)
    l[1:12] .= b[1:12]
    u[1:12] .= b[1:12]
    l[13:24] .= -Inf
    u[25:36] .= Inf
    l[37:42] .= -Inf
    u[37:42] .= Inf
    q = randn(n)
    for sc in (0, 10)
        s = PureOSQP.solve(P, q, A, l, u; algorithm = :ipm, scaling = sc)
        @test s.status == SOLVED
        @test s.iter <= 30
        @test maximum(kkt_residuals(P, q, A, l, u, s.x, s.y)) < 1.0e-5
        # A free row carries no multiplier.
        @test all(iszero, s.y[37:42])
    end

    # Equality rows only: no complementarity, one solve and the full step per iteration,
    # which converges within two outer iterations.
    Ae = A[1:20, :]
    be = Ae * randn(n)
    for sc in (0, 10)
        s = PureOSQP.solve(P, q, Ae, be, be; algorithm = :ipm, scaling = sc)
        @test s.status == SOLVED
        @test s.iter <= 2
        @test maximum(kkt_residuals(P, q, Ae, be, be, s.x, s.y)) < 1.0e-5
    end
end

@testitem "interior point: outer iterations of the reference prototype" begin
    using LinearAlgebra, SparseArrays, Random
    # The dense generator of `bench/ipm_matrixfree_spike.jl` (`make_instance`), every row
    # two-sided, and with the row mix of `bench/ipm_rowtypes_spike.jl` (`mixed`: 20% equality,
    # 20% lower-only, 20% upper-only, 10% free). Seeds are explicit so the instances do not
    # depend on hashing.
    function spike_problem(n, κ, frac, seed; mixed = false)
        rng = Xoshiro(seed)
        m = n
        U = Matrix(qr(randn(rng, m, m)).Q)
        V = Matrix(qr(randn(rng, n, n)).Q)
        Q = Matrix(qr(randn(rng, n, n)).Q)
        A = U * Diagonal(exp10.(range(0, -log10(κ); length = n))) * V'
        P = Q * Diagonal(exp10.(range(0, -2; length = n))) * Q'
        P = (P + P') / 2
        xstar = randn(rng, n)
        a = A * xstar
        l = a .- (0.5 .+ rand(rng, m))
        u = a .+ (0.5 .+ rand(rng, m))
        ystar = zeros(m)
        if !mixed
            for i in randperm(rng, m)[1:round(Int, frac * m)]
                mag = 0.5 + rand(rng)
                if rand(rng, Bool)
                    ystar[i] = mag
                    u[i] = a[i]
                else
                    ystar[i] = -mag
                    l[i] = a[i]
                end
            end
            return P, -(P * xstar + A' * ystar), A, l, u
        end
        kind = fill(:two, m)
        p = randperm(rng, m)
        j = 0
        for (k, f) in ((:eq, 0.2), (:lo, 0.2), (:up, 0.2), (:free, 0.1)), _ in 1:round(Int, f * m)
            kind[p[j += 1]] = k
        end
        fill!(l, -Inf)
        fill!(u, Inf)
        for i in 1:m
            gap = 0.5 + rand(rng)
            mag = 0.5 + rand(rng)
            active = rand(rng) < frac
            k = kind[i]
            if k === :eq
                l[i] = u[i] = a[i]
                ystar[i] = rand(rng, Bool) ? mag : -mag
            elseif k === :lo
                active ? (l[i] = a[i]; ystar[i] = -mag) : (l[i] = a[i] - gap)
            elseif k === :up
                active ? (u[i] = a[i]; ystar[i] = mag) : (u[i] = a[i] + gap)
            elseif k === :two
                l[i] = a[i] - gap
                u[i] = a[i] + 0.5 + rand(rng)
                if active
                    rand(rng, Bool) ? (u[i] = a[i]; ystar[i] = mag) : (l[i] = a[i]; ystar[i] = -mag)
                end
            end
        end
        return P, -(P * xstar + A' * ystar), A, l, u
    end

    # The sparse KKT backend on dense data, which its fill gate would never select.
    Ext = Base.get_extension(PureOSQP, :PureOSQPSparseArraysExt)
    function sparse_kkt_workspace(P, q, A, l, u; kwargs...)
        Ps, As = sparse(P), sparse(A)
        m, n = size(A)
        prob = PureOSQP.Problem(Float64, Ps, q, As, l, u; scaling = 0)
        wt = PureOSQP.SystemWeights(ones(m), ones(m), 1.0e-8)
        gram = Ext.kkt_gram(Float64, Ps, As, n, m)
        K = Ext.refill_kkt!(gram, Ps, As, wt.w_inv, prob.E, prob.D, prob.c, wt.sigma)
        F = ldlt(Symmetric(K, :U))
        LD = sparse(F.LD)
        ls = Ext.SparseKKT{Float64, Vector{Float64}, typeof(F)}(
            gram, F, LD, inv.(diag(LD)), F.p, zeros(n + m), zeros(n + m)
        )
        return PureOSQP.ipm_workspace(ls, prob, wt, IPMSettings{Float64}(; scaling = 0, kwargs...))
    end

    # Outer iterations to `eps = 1e-8` of the prototype `ipm3` in `bench/ipm_rowtypes_spike.jl`
    # on these instances at `δ = 1e-8`: `:exact` (Bunch–Kaufman, no refinement) and
    # `:exact_cholmod_ref1` (CHOLMOD `ldlt`, one refinement step) agree on every one.
    # Keyed by (κ, active fraction), then (two-sided, mixed).
    expected = Dict(
        (1.0, 0.1) => (7, 7), (1.0, 0.5) => (7, 7), (1.0, 0.9) => (7, 7),
        (1.0e3, 0.1) => (6, 6), (1.0e3, 0.5) => (8, 8), (1.0e3, 0.9) => (9, 8),
        (1.0e6, 0.1) => (7, 7), (1.0e6, 0.5) => (9, 9), (1.0e6, 0.9) => (11, 10),
    )
    grid = [(κ, frac) for κ in (1.0, 1.0e3, 1.0e6) for frac in (0.1, 0.5, 0.9)]
    for (index, (κ, frac)) in enumerate(grid)
        seed = 700 + index
        for (k, mixed) in enumerate((false, true))
            data = spike_problem(200, κ, frac, seed; mixed)
            ref = expected[(κ, frac)][k]
            dense = PureOSQP.solve(data...; algorithm = :ipm, linsys = :kkt, scaling = 0, refine_iter = 0)
            @test dense.status == SOLVED
            @test abs(dense.iter - ref) <= 2
            sparse_ws = sparse_kkt_workspace(data...; refine_iter = 1)
            @test PureOSQP.backend_name(sparse_ws.linsys) === :sparse_kkt
            s = solve!(sparse_ws)
            @test s.status == SOLVED
            @test abs(s.iter - ref) <= 2
        end
    end
end

@testitem "interior point: unsupported inputs are refused by name" begin
    using LinearAlgebra, SparseArrays, Random, Krylov
    Random.seed!(83)
    n, m = 6, 9
    X = randn(n, n)
    P = Matrix(X'X / n + I)
    A = randn(m, n)
    q, l, u = randn(n), -rand(m), rand(m)

    @test_throws "needs Float64 or a finer element type" setup(
        Float32.(P), Float32.(q), Float32.(A), Float32.(l), Float32.(u); algorithm = :ipm
    )
    Pop = PureOSQP.ProductOperator{Float64}(P; symmetric = true, posdef = true)
    Aop = PureOSQP.ProductOperator{Float64}(A)
    @test_throws "supplies products only" setup(Pop, q, Aop, l, u; algorithm = :ipm)
    @test_throws "linsys = :indirect is not available with algorithm = :ipm" setup(
        P, q, A, l, u; algorithm = :ipm, linsys = :indirect
    )
    @test_throws "algorithm must be :admm or :ipm" setup(P, q, A, l, u; algorithm = :newton)

    n1, n2 = 3, 4
    K = PureOSQP.KroneckerOperator(randn(n1, n1), randn(n2, n2))
    Pk = Diagonal(fill(2.0, n1 * n2))
    qk, lk, uk = randn(n1 * n2), -rand(n1 * n2), rand(n1 * n2)
    @test_throws "linsys = :kronecker is not available with algorithm = :ipm" setup(
        Pk, qk, K, lk, uk; algorithm = :ipm, linsys = :kronecker, scaling = 0
    )
    # The same pair on `:auto` is declined by the Kronecker rung and still solves.
    ws = setup(Pk, qk, K, lk, uk; algorithm = :ipm, scaling = 0)
    @test PureOSQP.backend_name(ws.linsys) !== :kronecker
    @test solve!(ws).status == SOLVED
end

@testitem "interior point: a GPU array is refused by name" tags = [:gpu] begin
    using LinearAlgebra, Random, JLArrays, GPUArraysCore
    JLArrays.allowscalar(false)
    Random.seed!(84)
    n, m = 8, 16
    X = randn(n, n)
    P = jl(Matrix(X'X / n + I))
    A = jl(randn(m, n))
    q, l, u = jl(randn(n)), jl(-ones(m)), jl(ones(m))
    @test_throws "algorithm = :ipm runs on the host" setup(P, q, A, l, u; algorithm = :ipm)
end

@testitem "interior point: ADMM stays the default, and the IPM infers concretely" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P, q, A, l, u = random_qp(10, 15; seed = 5)
    @test setup(P, q, A, l, u) isa Workspace
    @test setup(P, q, A, l, u; algorithm = :admm) isa Workspace

    ws = setup(P, q, A, l, u; algorithm = :ipm)
    W = typeof(ws)
    @test isconcretetype(W)
    @test only(Base.return_types(solve!, (W,))) === Solution{Float64}
    @test only(Base.return_types(PureOSQP.ipm_step!, (W,))) === W

    # A re-solve starts from the previous point and a cold start forgets it.
    s1 = solve!(ws)
    @test s1.status == SOLVED
    @test ws.seeded
    s2 = solve!(ws)
    @test s2.status == SOLVED
    @test s2.x ≈ s1.x atol = 1.0e-6
    cold_start!(ws)
    @test !ws.seeded
    s3 = solve!(ws)
    @test s3.iter == s1.iter
    @test s3.x == s1.x
end
