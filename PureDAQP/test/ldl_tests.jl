@testitem "GramLDL reproduces the Gram matrix as rows are added" begin
    using LinearAlgebra, Random
    using PureDAQP: GramLDL, add_row!

    "The matrix the factors stand for."
    function reconstruct(F)
        k = F.k
        L = Matrix(UnitLowerTriangular(view(F.L, 1:k, 1:k)))
        return L * Diagonal(view(F.D, 1:k)) * L'
    end

    rng = MersenneTwister(5)
    for _ in 1:60
        m, n = rand(rng, 3:14), rand(rng, 2:8)
        M = randn(rng, m, n)
        F = GramLDL{Float64}(m)
        active = Int[]
        for r in randperm(rng, m)[1:rand(rng, 1:min(m, n))]
            g = isempty(active) ? Float64[] : M[active, :] * M[r, :]
            add_row!(F, g, dot(M[r, :], M[r, :]))
            push!(active, r)
            G = M[active, :] * M[active, :]'
            @test opnorm(reconstruct(F) - G) / max(1.0, opnorm(G)) < 1.0e-9
        end
    end
end

@testitem "GramLDL repairs itself as rows are removed" begin
    using LinearAlgebra, Random
    using PureDAQP: GramLDL, add_row!, remove_row!

    function reconstruct(F)
        k = F.k
        iszero(k) && return zeros(0, 0)
        L = Matrix(UnitLowerTriangular(view(F.L, 1:k, 1:k)))
        return L * Diagonal(view(F.D, 1:k)) * L'
    end

    rng = MersenneTwister(7)
    for _ in 1:60
        m, n = rand(rng, 4:14), rand(rng, 3:8)
        M = randn(rng, m, n)
        F = GramLDL{Float64}(m)
        active = Int[]
        for r in randperm(rng, m)[1:min(m, n)]
            g = isempty(active) ? Float64[] : M[active, :] * M[r, :]
            add_row!(F, g, dot(M[r, :], M[r, :]))
            push!(active, r)
        end
        # Drop in a random order, checking the factors after every single removal: a
        # rank-one repair that is subtly wrong shows up here and nowhere else.
        while F.k > 1
            i = rand(rng, 1:F.k)
            remove_row!(F, i)
            deleteat!(active, i)
            G = M[active, :] * M[active, :]'
            @test opnorm(reconstruct(F) - G) / max(1.0, opnorm(G)) < 1.0e-8
        end
    end
end

@testitem "solve_gram! matches a dense solve" begin
    using LinearAlgebra, Random
    using PureDAQP: GramLDL, add_row!, solve_gram!

    rng = MersenneTwister(11)
    for _ in 1:40
        M = randn(rng, 9, 6)
        F = GramLDL{Float64}(9)
        active = Int[]
        for r in randperm(rng, 9)[1:5]
            g = isempty(active) ? Float64[] : M[active, :] * M[r, :]
            add_row!(F, g, dot(M[r, :], M[r, :]))
            push!(active, r)
        end
        G = M[active, :] * M[active, :]'
        rhs = randn(rng, length(active))
        x = copy(rhs)
        solve_gram!(F, x)
        @test norm(G * x - rhs) / max(1.0, norm(rhs)) < 1.0e-7
    end
end

@testitem "a dependent row is marked by a zero pivot, not a division" begin
    using LinearAlgebra
    using PureDAQP: GramLDL, add_row!

    # The third row is the sum of the first two, so the Gram matrix is singular. The
    # factorization must record that as an exact zero: the solver reads `D` for it and
    # takes its singular branch rather than dividing by a tiny pivot.
    M = [1.0 0.0; 0.0 1.0; 1.0 1.0]
    F = GramLDL{Float64}(3)
    add_row!(F, Float64[], dot(M[1, :], M[1, :]))
    add_row!(F, [dot(M[1, :], M[2, :])], dot(M[2, :], M[2, :]))
    add_row!(F, [dot(M[1, :], M[3, :]), dot(M[2, :], M[3, :])], dot(M[3, :], M[3, :]))
    @test F.k == 3
    @test iszero(F.D[3])
    @test all(isfinite, view(F.D, 1:3))
end
