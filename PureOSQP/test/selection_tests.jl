@testitem "recommend_linsys ranks what it measured" begin
    using LinearAlgebra, SparseArrays, Random
    using LDLFactorizations
    Random.seed!(75)

    # Banded, so a sparse factorization has something to win with and the ranking is not a
    # tie between candidates doing the same dense arithmetic.
    n = 150
    P = sparse(SymTridiagonal(fill(2.0, n), fill(0.3, n - 1)))
    A = sparse(Bidiagonal(fill(1.0, n), fill(-1.0, n - 1), :U))
    q, l, u = collect(range(-1.0, 1.0; length = n)), fill(-1.0, n), fill(1.0, n)

    advice = recommend_linsys(P, q, A, l, u; max_iter = 5, repeats = 2)
    @test advice isa LinsysAdvice
    @test advice.linsys in PureOSQP.LINSYS_OPTIONS
    # Ranked by the cost of a whole solve, with the dense terminal and `:auto` both reached.
    @test issorted(advice.candidates; by = c -> c.total_ms)
    @test advice.solve_iters >= 1
    @test :auto in [c.linsys for c in advice.candidates]
    @test :dense in [c.linsys for c in advice.candidates]
    # Every candidate ran the same bounded number of iterations and reports a real fill, and
    # its total is its setup plus its per-iteration cost over the solve's own iterations.
    for c in advice.candidates
        @test 0 < c.iter <= 5
        @test c.setup_ms > 0 && c.solve_ms > 0
        @test c.factor_fill >= 0
        @test c.total_ms ≈ c.setup_ms + c.iterate_ms * advice.solve_iters
    end
    # The name it reports is one `setup` accepts, and reaches the backend it was measured on.
    ws = setup(P, q, A, l, u; linsys = advice.linsys)
    @test PureOSQP.backend_name(ws.linsys) === first(advice.candidates).backend
    # A name the pair refuses is left out of the ranking rather than raising.
    @test !(:kronecker in [c.linsys for c in advice.candidates])
    @test occursin("LinsysAdvice", sprint(show, MIME"text/plain"(), advice))

    @test_throws "max_iter must be at least 1" recommend_linsys(P, q, A, l, u; max_iter = 0)
    @test_throws "repeats must be at least 1" recommend_linsys(P, q, A, l, u; repeats = 0)
end

@testitem "factor_fill normalizes a workspace's factor against n" begin
    using LinearAlgebra, Random
    Random.seed!(72)
    n, m = 30, 20
    q, l, u = randn(n), -rand(m), rand(m)
    P, A = Matrix(1.0I, n, n), randn(m, n)
    red = setup(P, q, A, l, u)
    kkt = setup(P, q, A, l, u; linsys = :kkt)

    @test PureOSQP.factor_fill(red) == PureOSQP.backend_info(red.linsys).factor_nnz / n^2
    @test PureOSQP.factor_fill(kkt) == PureOSQP.backend_info(kkt.linsys).factor_nnz / n^2
    # The KKT backend's own `dim` is `n + m`, so normalizing by it would differ.
    @test PureOSQP.backend_info(kkt.linsys).dim == n + m
end
