@testitem "structural corpus: every instance passes the referee, on both backends" begin
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
            [i % 2 == 0 ? -Inf : b[i] - rand() for i in 1:12],
            [i % 3 == 0 ? Inf : b[i] + rand() for i in 1:12],
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
            s = PureOSQP.solve(
                P, q, A, l, u; eps_abs = 1.0e-9, eps_rel = 1.0e-9,
                max_iter = 200_000, polish = true, linsys = backend
            )
            @test s.status == SOLVED
            s.status == SOLVED || continue
            r = maximum(kkt_residuals(Matrix(P), q, Matrix(A), l, u, s.x, s.y))
            @test r < 1.0e-5
        end
    end
end

@testitem "structural corpus: objective agrees with the C library" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    for (n, m, seed) in ((8, 12, 41), (12, 5, 42), (6, 40, 43), (20, 20, 44))
        P, q, A, l, u = random_qp(n, m; seed)
        c = osqp_ref(
            P, q, A, l, u; eps_abs = 1.0e-9, eps_rel = 1.0e-9,
            max_iter = 100_000, polish = true
        )
        j = PureOSQP.solve(
            P, q, A, l, u; eps_abs = 1.0e-9, eps_rel = 1.0e-9,
            max_iter = 100_000, polish = true
        )
        @test j.status == SOLVED
        @test abs(j.obj_val - c.info.obj_val) <= 1.0e-6 * max(1, abs(c.info.obj_val))
    end
end

@testitem "the C library's own solution passes the referee" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    # An oracle is untrusted until it has judged a known-good answer. This is the check
    # that caught the referee measuring complementarity as a sign predicate: the C
    # library failed it on 23 of 30 problems, which is a fact about the referee.
    for trial in 1:12
        Random.seed!(100 + trial)
        n, m = rand(3:15), rand(1:30)
        P = (X = randn(n, n); Matrix(X'X))
        A = randn(m, n) * Diagonal(exp10.(rand(-3:3, n)))
        q = randn(n)
        l = -rand(m) .- 0.1
        u = rand(m) .+ 0.1
        c = osqp_ref(P, q, A, l, u; eps_abs = 1.0e-9, eps_rel = 1.0e-9, max_iter = 100_000)
        c.info.status == :Solved || continue
        @test maximum(kkt_residuals(P, q, A, l, u, c.x, c.y)) < 1.0e-5
    end
end

@testitem "Float32 solves the reference QP" begin
    using LinearAlgebra, SparseArrays, OSQP, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    P = Float32[4 1; 1 2]
    q = Float32[1, 1]
    A = Float32[1 1; 1 0; 0 1]
    s = PureOSQP.solve(
        P, q, A, Float32[1, 0, 0], Float32[1, 0.7, 0.7];
        eps_abs = 1.0f-5, eps_rel = 1.0f-5
    )
    @test s isa Solution{Float32}
    @test s.status == SOLVED
    @test s.obj_val ≈ 1.88f0 atol = 1.0f-3
end
