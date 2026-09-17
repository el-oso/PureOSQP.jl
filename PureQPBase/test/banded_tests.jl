@testitem "a banded reduced system solves through the banded backend" begin
    using PureQPBase, LinearAlgebra, BandedMatrices, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(31)
    n = 24
    # A tridiagonal A squares to bandwidth 2, which no symmetric LinearAlgebra type stores.
    P = SymTridiagonal(rand(n) .+ 4, rand(n - 1) ./ 8)
    A = Tridiagonal(rand(n - 1) ./ 4, rand(n) .+ 1, rand(n - 1) ./ 4)
    prob, wt, ls = backend_for(P, randn(n), A, -rand(n), rand(n); scaling = 0)
    @test PureQPBase.backend_name(ls) == :banded

    # The bandwidth rule the backend is built on: max(bw(P), 2 bw(A)).
    @test ls.bw == 2
    # `cholesky!` overwrites the assembled matrix, so what `R` holds now is the factor.
    # The reduced matrix is checked through the system it solves, below.
    @test Matrix(ls.fact.U)' * Matrix(ls.fact.U) ≈ reduced_matrix(P, A, wt) rtol = 1.0e-9

    bx, bz = randn(n), randn(n)
    x, z = zeros(n), zeros(n)
    PureQPBase.solve_system!(ls, prob, wt, bx, bz, x, z)
    ref = kkt_matrix(P, A, wt) \ [bx; bz]
    @test x ≈ ref[1:n] rtol = 1.0e-9
    @test z ≈ A * x rtol = 1.0e-9
end

@testitem "the narrow and wide cases stay with their own backends" begin
    using PureQPBase, LinearAlgebra, BandedMatrices, Random
    include(joinpath(@__DIR__, "helpers.jl"))
    Random.seed!(33)
    n = 20
    q, l, u = randn(n), -rand(n), rand(n)
    name(P, A) = PureQPBase.backend_name(last(backend_for(P, q, A, l, u)))
    # Bandwidth 0 and 1 keep the LinearAlgebra backends even with BandedMatrices loaded.
    @test name(Diagonal(rand(n) .+ 1), Diagonal(rand(n) .+ 1)) == :diagonal
    @test name(SymTridiagonal(rand(n) .+ 3, rand(n - 1) ./ 8), Diagonal(rand(n) .+ 1)) == :tridiagonal
    # A band wide enough to stop being the smaller representation: this `A` has `m = n`, so
    # the rung declines once `2b + 1` exceeds `m + n`, which `b = n ÷ 2` does by one.
    wide = BandedMatrix{Float64}(undef, (n, n), (n ÷ 2, n ÷ 2))
    fill!(wide.data, 0.0)
    for j in 1:n, i in max(1, j - n ÷ 2):min(n, j + n ÷ 2)
        wide[i, j] = i == j ? 2.0 : 0.01
    end
    @test name(Diagonal(rand(n) .+ 1), wide) == :cholesky
end

@testitem "equilibration of a banded P matches the dense form" begin
    using PureQPBase, LinearAlgebra, BandedMatrices, Random
    Random.seed!(34)
    n, b = 40, 3
    P = BandedMatrix{Float64}(undef, (n, n), (b, b))
    fill!(P.data, 0.0)
    P[band(0)] .= rand(n) .+ 2b
    for k in 1:b
        P[band(k)] .= rand(n - k) ./ (4b)
        P[band(-k)] .= P[band(k)]
    end
    A = Diagonal(rand(n) .+ 0.5)
    q, l, u = randn(n), -rand(n), rand(n)

    banded = PureQPBase.validated_problem(Float64, n, n, P, q, A, l, u, 10)
    dense = PureQPBase.validated_problem(Float64, n, n, Matrix(P), q, Matrix(A), l, u, 10)

    # The band-restricted column walk skips only entries that are structurally zero, and a
    # zero moves neither a running maximum nor a scaling factor. So the two forms must agree
    # exactly, not approximately -- equilibration is what the libosqp iteration-count
    # comparison rests on.
    @test banded.D == dense.D
    @test banded.E == dense.E
    @test banded.c == dense.c
end
