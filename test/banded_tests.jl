@testitem "a banded reduced system solves through the banded backend" begin
    using LinearAlgebra, BandedMatrices, Random
    Random.seed!(31)
    n = 24
    # A tridiagonal A squares to bandwidth 2, which no symmetric LinearAlgebra type stores.
    P = SymTridiagonal(rand(n) .+ 4, rand(n - 1) ./ 8)
    A = Tridiagonal(rand(n - 1) ./ 4, rand(n) .+ 1, rand(n - 1) ./ 4)
    ws = setup(P, randn(n), A, -rand(n), rand(n); scaling = 0, sigma = 1.0e-6, rho = 0.1)
    @test PureOSQP.backend_name(ws.linsys) == :banded

    # The bandwidth rule the backend is built on: max(bw(P), 2 bw(A)).
    @test ws.linsys.bw == 2
    # `cholesky!` overwrites the assembled matrix, so what `R` holds now is the factor.
    # The reduced matrix is checked through the system it solves, below.
    R = Matrix(P) + ws.settings.sigma * I + Matrix(A)' * Diagonal(ws.rho_vec) * Matrix(A)
    @test Matrix(ws.linsys.fact.U)' * Matrix(ws.linsys.fact.U) ≈ R rtol = 1.0e-9

    bx, bz = randn(n), randn(n)
    PureOSQP.solve_system!(ws.linsys, ws, bx, bz)
    K = [Matrix(P) + ws.settings.sigma * I  Matrix(A)'; Matrix(A)  -Diagonal(1 ./ ws.rho_vec)]
    ref = K \ [bx; bz]
    @test ws.xtilde ≈ ref[1:n] rtol = 1.0e-9
    @test ws.ztilde ≈ A * ws.xtilde rtol = 1.0e-9
end

@testitem "the banded backend agrees with the dense one end to end" begin
    using LinearAlgebra, BandedMatrices, Random
    Random.seed!(32)
    n = 40
    P = SymTridiagonal(rand(n) .+ 4, rand(n - 1) ./ 8)
    A = Tridiagonal(rand(n - 1) ./ 4, rand(n) .+ 1, rand(n - 1) ./ 4)
    q, l, u = randn(n), -rand(n), rand(n)
    opts = (eps_abs = 1.0e-9, eps_rel = 1.0e-9)
    banded = solve(P, q, A, l, u; opts...)
    dense = solve(Matrix(P), q, Matrix(A), l, u; opts...)
    @test banded.status == PureOSQP.SOLVED
    @test banded.x ≈ dense.x rtol = 1.0e-6
    @test banded.iter == dense.iter
end

@testitem "the narrow and wide cases stay with their own backends" begin
    using LinearAlgebra, BandedMatrices, Random
    Random.seed!(33)
    n = 20
    q, l, u = randn(n), -rand(n), rand(n)
    # Bandwidth 0 and 1 keep the LinearAlgebra backends even with BandedMatrices loaded.
    diag_ws = setup(Diagonal(rand(n) .+ 1), q, Diagonal(rand(n) .+ 1), l, u)
    @test PureOSQP.backend_name(diag_ws.linsys) == :diagonal
    tri_ws = setup(SymTridiagonal(rand(n) .+ 3, rand(n - 1) ./ 8), q, Diagonal(rand(n) .+ 1), l, u)
    @test PureOSQP.backend_name(tri_ws.linsys) == :tridiagonal
    # A band wide enough to stop being the smaller representation: this `A` has `m = n`, so
    # the rung declines once `2b + 1` exceeds `m + n`, which `b = n ÷ 2` does by one.
    wide = BandedMatrix{Float64}(undef, (n, n), (n ÷ 2, n ÷ 2))
    fill!(wide.data, 0.0)
    for j in 1:n, i in max(1, j - n ÷ 2):min(n, j + n ÷ 2)
        wide[i, j] = i == j ? 2.0 : 0.01
    end
    wide_ws = setup(Diagonal(rand(n) .+ 1), q, wide, l, u)
    @test PureOSQP.backend_name(wide_ws.linsys) == :cholesky
end
