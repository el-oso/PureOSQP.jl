@testitem "a GPU array solves through the matrix-free backend" begin
    using LinearAlgebra, Random, Krylov, JLArrays, GPUArraysCore
    # JLArrays is CPU-hosted and slow, so this is a correctness gate, not a performance one:
    # with `allowscalar(false)` it enforces exactly the discipline CUDA.jl enforces, which is
    # what the whole GPU path is about. It says nothing about whether CUDA works -- see the
    # LAPACK item below for why a green run here is not that claim.
    JLArrays.allowscalar(false)
    Random.seed!(3)
    n, m = 20, 40
    X = randn(n, n)
    P = Matrix(X'X / n + I)
    q = randn(n)
    A = randn(m, n)
    b = A * randn(n)
    l, u = b .- rand(m), b .+ rand(m)
    opts = (eps_abs = 1.0e-8, eps_rel = 1.0e-8, max_iter = 20_000, linsys = :indirect)

    host = PureOSQP.solve(P, q, A, l, u; opts...)
    device = PureOSQP.solve(jl(P), jl(q), jl(A), jl(l), jl(u); opts...)

    @test host.status == SOLVED
    @test device.status == SOLVED
    # The inner solve is inexact and the two run different reduction orders, so the iterates
    # are close rather than identical. Both must solve the original problem.
    @test device.x ≈ host.x atol = 1.0e-5
    @test device.obj_val ≈ host.obj_val atol = 1.0e-5
    # The result is host memory whatever the workspace was built from.
    @test device.x isa Vector{Float64}
    @test device.y isa Vector{Float64}
end

@testitem "a GPU array is refused by the direct backends" begin
    using LinearAlgebra, Random, JLArrays, GPUArraysCore
    # `JLArray <: StridedArray`, so without an explicit refusal the dense backend dispatches
    # to CPU LAPACK and quietly succeeds here while failing inside `factorize!` on a CuArray.
    # The refusal has to happen at setup, where the caller can act on it.
    JLArrays.allowscalar(false)
    @test JLArray{Float64, 2} <: StridedArray
    Random.seed!(4)
    n, m = 8, 16
    X = randn(n, n)
    P = jl(Matrix(X'X / n + I))
    A = jl(randn(m, n))
    q, l, u = jl(randn(n)), jl(-ones(m)), jl(ones(m))
    @test_throws "only the matrix-free backend has a GPU counterpart" setup(P, q, A, l, u)
    @test_throws "only the matrix-free backend has a GPU counterpart" setup(P, q, A, l, u; linsys = :auto)
end

@testitem "a GPU array survives update! and warm starting" begin
    using LinearAlgebra, Random, Krylov, JLArrays, GPUArraysCore
    JLArrays.allowscalar(false)
    Random.seed!(5)
    n, m = 15, 30
    X = randn(n, n)
    P = Matrix(X'X / n + I)
    A = randn(m, n)
    b = A * randn(n)
    q, l, u = randn(n), b .- rand(m), b .+ rand(m)
    opts = (eps_abs = 1.0e-8, eps_rel = 1.0e-8, max_iter = 20_000, linsys = :indirect)

    ws = setup(jl(P), jl(q), jl(A), jl(l), jl(u); opts...)
    @test PureOSQP.backend_name(ws.linsys) == :indirect
    @test solve!(ws).status == SOLVED

    q2 = randn(n)
    update!(ws; q = jl(q2))
    s = solve!(ws)
    @test s.status == SOLVED
    ref = PureOSQP.solve(P, q2, A, l, u; opts...)
    @test s.x ≈ ref.x atol = 1.0e-5

    cold_start!(ws)
    @test solve!(ws).status == SOLVED
end
