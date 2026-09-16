"""
    LinsysAdvice

What [`recommend_linsys`](@ref) measured: `linsys`, the name that was fastest, and
`candidates`, every backend the pair admitted, fastest first.

Each candidate carries the `linsys` name a caller would pass, the [`backend_name`](@ref) that
name reached, the milliseconds [`setup`](@ref) and the bounded solve took, the
[`factor_fill`](@ref) of the factorization it holds, and the status and iteration count the
bounded solve stopped at. `iterate_ms` is the solve divided by its iterations, which is the
figure to compare across candidates: they all take the same iterates, so they differ in
per-iteration cost and not in path.
"""
const LinsysMeasurement = @NamedTuple{
    linsys::Symbol, backend::Symbol, setup_ms::Float64, solve_ms::Float64,
    iterate_ms::Float64, factor_fill::Float64, status::Status, iter::Int,
}

struct LinsysAdvice
    linsys::Symbol
    candidates::Vector{LinsysMeasurement}
end

function Base.show(io::IO, ::MIME"text/plain", a::LinsysAdvice)
    println(io, "LinsysAdvice: linsys = :", a.linsys)
    println(
        io, "  ", rpad("linsys", 13), rpad("backend", 20), lpad("setup ms", 10),
        lpad("solve ms", 10), lpad("ms/iter", 10), lpad("fill", 10), lpad("iter", 7), "  status"
    )
    for c in a.candidates
        println(
            io, "  ", rpad(":$(c.linsys)", 13), rpad(c.backend, 20),
            lpad(round(c.setup_ms; digits = 3), 10), lpad(round(c.solve_ms; digits = 3), 10),
            lpad(round(c.iterate_ms; digits = 4), 10), lpad(round(c.factor_fill; digits = 5), 10),
            lpad(c.iter, 7), "  ", status_name(c.status)
        )
    end
    return nothing
end

"""
    recommend_linsys(P, q, A, l, u, alg = OperatorSplitting(); max_iter = 25, repeats = 3, kwargs...)

Measure every backend this problem admits and rank them, fastest per iteration first.

[`setup`](@ref) chooses a backend from the types of `P` and `A` and, for a `SparseMatrixCSC`
pair, a property of their sparsity pattern. That is a rule fitted to a benchmark suite, so it
is right about a class of problems and not about any particular one. This runs the experiment
instead: for each of `PureOSQP.LINSYS_OPTIONS` the pair accepts it builds the workspace, runs
`max_iter` iterations, and reports the times and the factor fill. Pass the winner's `linsys`
to `setup` to pin it.

Nothing on the solve path calls this, and it adds no dependency: the clock is `time_ns()` and
the repetitions are its own. A name the pair refuses is left out of the ranking rather than
raising. `kwargs` go to every `setup`, so the measurement runs at the settings the real solve
will use — `scaling` in particular changes the matrices every backend factors.

`max_iter` bounds the work, so the candidates are compared on cost per iteration and not on
whether they converge. `repeats` runs each candidate that many times after a warm-up and
keeps the fastest, which is what takes Julia's compilation out of the numbers.

```julia
julia> using PureOSQP, SparseArrays

julia> advice = recommend_linsys(P, q, A, l, u, InteriorPoint());

julia> advice.linsys
:sparse

julia> ws = setup(P, q, A, l, u, InteriorPoint(); linsys = advice.linsys);
```
"""
function recommend_linsys(
        P::AbstractMatrix, q::AbstractVector, A::AbstractMatrix, l::AbstractVector,
        u::AbstractVector, alg::QPAlgorithm = OperatorSplitting();
        max_iter::Integer = 25, repeats::Integer = 3, kwargs...
    )
    max_iter >= 1 || throw(ArgumentError("max_iter must be at least 1, got $max_iter"))
    repeats >= 1 || throw(ArgumentError("repeats must be at least 1, got $repeats"))
    rows = LinsysMeasurement[]
    for name in LINSYS_OPTIONS
        row = measure_linsys(P, q, A, l, u, alg, name, Int(max_iter), Int(repeats); kwargs...)
        isnothing(row) || push!(rows, row)
    end
    isempty(rows) && throw(
        ArgumentError(
            "no backend served this problem, not even linsys = :auto, so there is nothing " *
                "to recommend: call setup directly to see why it refuses."
        )
    )
    sort!(rows; by = r -> r.iterate_ms)
    # `:auto` reaches one of the named backends and measures as a duplicate of it, so the
    # name to report is the specific one a caller would pin.
    best = first(rows)
    named = findfirst(r -> r.linsys !== :auto && r.backend === best.backend, rows)
    return LinsysAdvice(isnothing(named) ? best.linsys : rows[named].linsys, rows)
end

"""
    measure_linsys(P, q, A, l, u, alg, name, max_iter, repeats; kwargs...) -> NamedTuple or nothing

Time `setup` and a bounded `solve!` on `linsys = name`, or `nothing` where that name refuses
the pair.

An `ArgumentError` is how a `linsys` name declines a problem it cannot serve, and a
`MethodError` is how a representation reaches a backend with no method for it; both mean the
candidate is out of the ranking, which is not the same as it being slow. Anything else is the
caller's problem and is rethrown.
"""
function measure_linsys(
        P, q, A, l, u, alg::QPAlgorithm, name::Symbol, max_iter::Int, repeats::Int; kwargs...
    )
    build() = setup(P, q, A, l, u, alg; linsys = name, max_iter, kwargs...)
    ws = try
        build()
    catch err
        err isa Union{ArgumentError, MethodError} && return nothing
        rethrow()
    end
    sol = solve!(ws)            # warm-up: compile this workspace type before timing it
    setup_ms, solve_ms = Inf, Inf
    for _ in 1:repeats
        t0 = time_ns()
        ws = build()
        t1 = time_ns()
        sol = solve!(ws)
        t2 = time_ns()
        setup_ms = min(setup_ms, (t1 - t0) / 1.0e6)
        solve_ms = min(solve_ms, (t2 - t1) / 1.0e6)
    end
    return (;
        linsys = name, backend = backend_name(ws.linsys), setup_ms, solve_ms,
        iterate_ms = solve_ms / max(sol.iter, 1), factor_fill = factor_fill(ws),
        status = sol.status, iter = sol.iter,
    )
end
