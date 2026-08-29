@testitem "the public entry points are --trim compatible" begin
    using TrimCheck
    # `juliac --trim` needs every call resolved statically. This is what stops the solver
    # from silently acquiring a dynamic dispatch or a reflective call.
    #
    # `not_trimmable` is a negative control: it calls `Base.return_types`, which the
    # trimmer cannot resolve. If the checker passed that one it would not be
    # discriminating, and its verdict on the real entry points would mean nothing.
    entry = joinpath(@__DIR__, "trim", "entrypoints.jl")
    names = [
        :solve_default, :solve_polish, :solve_kkt, :solve_unscaled, :solve_verbose,
        :solve_interruptible,
        :solve_time_limited,
        :setup_solve_update, :warm_started, :not_trimmable,
    ]
    sigs = [
        :(TrimEntry.$f(Matrix{Float64}, Vector{Float64}, Matrix{Float64}, Vector{Float64}, Vector{Float64}))
            for f in names
    ]
    results = TrimCheck.validate(sigs...; init = :(include($entry); using .TrimEntry), progressbar = false)
    ok = Dict(
        String(f) => occursin("is trim compatible", sprint(show, r))
            for (f, r) in zip(names, results)
    )
    for f in names[1:(end - 1)]
        @test ok[String(f)]
    end
    @test !ok["not_trimmable"]
end
