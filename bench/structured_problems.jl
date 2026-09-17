# The structured `(P, A)` pairs `PureQPBase/test/selection_tests.jl` checks backend selection against,
# and the block-diagonal reference case, shared by `snapshot.jl` and `ipm_backends.jl`.
# Needs `PureOSQP`, `LinearAlgebra`, `BandedMatrices` and `Random` loaded.

"An `n×n` band of half-width `b`, diagonally dominant so the reduced matrix stays definite."
function wide_band(n, b)
    band = BandedMatrix{Float64}(undef, (n, n), (b, b))
    fill!(band.data, 0.0)
    for j in 1:n, i in max(1, j - b):min(n, j + b)
        band[i, j] = i == j ? 1.0 : 0.01
    end
    return band
end

"""
    structured_families(n) -> Vector{NamedTuple}

The structured `(P, A)` pairs `PureQPBase/test/selection_tests.jl` checks backend selection against, each
under its own seed so adding or reordering a family never changes another one's numbers.
"""
function structured_families(n)
    specs = (
        (
            seed = 9101, name = "diagonal",
            build = () -> (Diagonal(rand(n) .+ 0.5), Diagonal(rand(n) .+ 0.5)),
        ),
        (
            seed = 9102, name = "tridiagonal_diag",
            build = () -> (
                SymTridiagonal(rand(n) .+ 3, rand(n - 1) ./ 8), Diagonal(rand(n) .+ 0.5),
            ),
        ),
        (
            seed = 9103, name = "banded_tridiag",
            build = () -> (
                SymTridiagonal(rand(n) .+ 4, rand(n - 1) ./ 8),
                Tridiagonal(rand(n - 1) ./ 4, rand(n) .+ 1, rand(n - 1) ./ 4),
            ),
        ),
        (
            seed = 9104, name = "tridiagonal_bidiag",
            build = () -> (
                SymTridiagonal(rand(n) .+ 4, rand(n - 1) ./ 8),
                Bidiagonal(rand(n) .+ 1, rand(n - 1) ./ 4, :L),
            ),
        ),
        (
            seed = 9105, name = "cholesky_symmetric_banded",
            build = () -> (
                Symmetric(Matrix(SymTridiagonal(rand(n) .+ 4, rand(n - 1) ./ 8))),
                BandedMatrix(0 => rand(n) .+ 1, 1 => rand(n - 1) ./ 4, -1 => rand(n - 1) ./ 4),
            ),
        ),
        (
            seed = 9106, name = "banded_wide_inside",
            build = () -> (SymTridiagonal(rand(n) .+ 4, rand(n - 1) ./ 8), wide_band(n, 12)),
        ),
        (
            seed = 9107, name = "cholesky_wide_outside",
            build = () -> (SymTridiagonal(rand(n) .+ 4, rand(n - 1) ./ 8), wide_band(n, 13)),
        ),
        (
            seed = 9108, name = "lowrank_rank3",
            build = () -> (
                Diagonal(rand(n) .+ 0.5),
                PureOSQP.RowCoupled(randn(3, n) ./ 4, ones(n - 3), collect(1:(n - 3))),
            ),
        ),
        (
            seed = 9109, name = "lowrank_rank10",
            build = () -> (
                Diagonal(rand(n) .+ 0.5),
                PureOSQP.RowCoupled(randn(10, n) ./ 4, ones(n - 10), collect(1:(n - 10))),
            ),
        ),
        (
            seed = 9110, name = "cholesky_rank11",
            build = () -> (
                Diagonal(rand(n) .+ 0.5),
                PureOSQP.RowCoupled(randn(11, n) ./ 4, ones(n - 11), collect(1:(n - 11))),
            ),
        ),
    )
    fam = NamedTuple[]
    for s in specs
        Random.seed!(s.seed)
        P, A = s.build()
        q, l, u = randn(n), -rand(n), rand(n)
        push!(fam, (name = s.name, P = P, A = A, q = q, l = l, u = u))
    end
    return fam
end

"""
`BlockDiagonal` `P` and `A` with matching block partitions, `PureQPBase/test/block_tests.jl`'s reference
case for the block backend.
"""
function block_problem()
    Random.seed!(9301)
    K, nb, mb = 5, 12, 8
    spd(k) = Matrix(
        Symmetric(
            let S = randn(k, k)
                S'S ./ k + 3I
            end
        )
    )
    P = PureOSQP.BlockDiagonal([spd(nb) for _ in 1:K])
    A = PureOSQP.BlockDiagonal([randn(mb, nb) ./ sqrt(nb) for _ in 1:K])
    n, m = size(P, 1), size(A, 1)
    q = randn(n)
    b = A * randn(n)
    return (P, q, A, b .- rand(m), b .+ rand(m))
end
