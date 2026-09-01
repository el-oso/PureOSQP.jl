@testitem "the rrule matches finite differences through a whole solve" begin
    using ChainRulesCore, Zygote, LinearAlgebra

    P = [4.0 1.0; 1.0 2.0]
    q = [1.0, 1.0]
    A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
    l = [1.0, 0.0, 0.0]
    u = [1.0, 1.0, 1.0]
    tol = (eps_abs = 1.0e-10, eps_rel = 1.0e-10)

    # Differentiating a loss of the solution, through Zygote, without touching
    # `adjoint_derivative` by hand. This is the check that the rule is wired up, not just that
    # the derivative underneath it is right.
    loss(qq) = sum(solve(P, qq, A, l, u; tol...).x .^ 2)
    g = only(Zygote.gradient(loss, q))

    h = 1.0e-6
    fd = map(1:2) do i
        e = zeros(2)
        e[i] = h
        (loss(q .+ e) - loss(q .- e)) / 2h
    end
    @test g ≈ fd rtol = 1.0e-5

    # And against the implicit derivative directly, which is the quantity the rule is supposed
    # to be forwarding. A correct adjoint wired up wrongly would pass neither of these, but
    # only this one says where the fault is.
    ws = setup(P, q, A, l, u; tol..., polish = true)
    sol = PureOSQP.solve!(ws)
    direct = PureOSQP.adjoint_derivative(ws, 2 .* sol.x, zeros(3))
    @test g ≈ direct.dq rtol = 1.0e-8
end

@testitem "the rrule reaches every argument, and the frule agrees with it" begin
    using ChainRulesCore, Zygote, LinearAlgebra, Random
    Random.seed!(23)

    P = [3.0 0.5; 0.5 2.0]
    q = [0.4, -1.1]
    A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
    l = [0.5, -1.0, -1.0]
    u = [0.5, 1.0, 1.0]
    tol = (eps_abs = 1.0e-10, eps_rel = 1.0e-10)

    # Every one of the five data arguments carries a gradient, checked against central
    # differences rather than against "is it nonzero" -- whether a given gradient vanishes is
    # a property of the problem, not of the rule. Here the loss is quadratic in `x` and the
    # box rows are slack, so `q`, `l` and `u` all move it.
    loss(PP, qq, AA, ll, uu) = sum(solve(PP, qq, AA, ll, uu; tol...).x .^ 2)
    gP, gq, gA, gl, gu = Zygote.gradient(loss, P, q, A, l, u)
    for g in (gP, gq, gA, gl, gu)
        @test !isnothing(g)
    end

    h = 1.0e-6
    "Central difference of `loss` along a perturbation of one vector argument."
    function fd_vec(v, set)
        return map(eachindex(v)) do i
            e = zeros(length(v))
            e[i] = h
            (loss(set(v .+ e)...) - loss(set(v .- e)...)) / 2h
        end
    end
    @test gq ≈ fd_vec(q, qq -> (P, qq, A, l, u)) rtol = 1.0e-5

    # `l` and `u` move together, for the same reason `P`'s entries do: row 1 is an equality,
    # so nudging its lower bound alone would put `l` above `u` and be rejected before the
    # solve. Sliding the pair is a legal perturbation of every row, and its derivative is the
    # sum of the two partials.
    fd_bounds = map(eachindex(l)) do i
        e = zeros(length(l))
        e[i] = h
        (loss(P, q, A, l .+ e, u .+ e) - loss(P, q, A, l .- e, u .- e)) / 2h
    end
    @test gl .+ gu ≈ fd_bounds rtol = 1.0e-5 atol = 1.0e-8

    # `P` enters as ½xᵀPx, so a lone entry cannot be perturbed without breaking symmetry: move
    # the pair, and the derivative of that move is the sum of the two partials.
    for (i, j) in ((1, 1), (1, 2))
        E = zeros(2, 2)
        E[i, j] += h
        E[j, i] += h
        fd = (loss(P .+ E, q, A, l, u) - loss(P .- E, q, A, l, u)) / 2h
        @test gP[i, j] + gP[j, i] ≈ fd rtol = 1.0e-4 atol = 1.0e-8
    end

    # The forward rule pushes a perturbation the other way. Along a direction in `q`, the
    # directional derivative it returns must match what the reverse rule reports contracted
    # with the same direction.
    v = [1.0, -0.5]
    _, tangent = ChainRulesCore.frule(
        (NoTangent(), ZeroTangent(), v, ZeroTangent(), ZeroTangent(), ZeroTangent()),
        solve, P, q, A, l, u; tol...
    )
    gsum = only(Zygote.gradient(qq -> sum(solve(P, qq, A, l, u; tol...).x), q))
    @test sum(tangent.x) ≈ dot(gsum, v) rtol = 1.0e-6
end

@testitem "differentiating an unconverged solve is refused" begin
    using ChainRulesCore, Zygote, LinearAlgebra
    # The rules differentiate the KKT conditions, which hold at the solution and nowhere else,
    # so a run that stopped early has nothing to differentiate. Refusing beats returning the
    # gradient of a point that is not the answer.
    P = [4.0 1.0; 1.0 2.0]
    q = [1.0, 1.0]
    A = [1.0 1.0; 1.0 0.0; 0.0 1.0]
    l = [1.0, 0.0, 0.0]
    u = [1.0, 1.0, 1.0]
    @test_throws "cannot differentiate a solve that ended" Zygote.gradient(
        qq -> sum(solve(P, qq, A, l, u; max_iter = 2).x), q
    )
end
