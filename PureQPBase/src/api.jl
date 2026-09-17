# `@constprop :aggressive` for the same reason `setup` carries it, at the second of two
# barriers: without it the keyword values arrive at `setup`'s call site as runtime values, so
# `setup`'s own annotation has no constants to propagate. Both are needed; either alone leaves
# the widening in place.
"""
    solve(P, q, A, l, u, alg; x0 = nothing, y0 = nothing, kwargs...) -> Solution

Solve `min ½xᵀPx + qᵀx  s.t.  l ≤ Ax ≤ u` in one call: build the workspace, warm-start
from `x0` and `y0` when given, and run the loop.

`P` must be symmetric and the problem convex, and the five inputs are validated exactly as in
[`setup`](@ref): `q` finite, `l ≤ u` elementwise, `l` free of `+Inf` and `u` free of `-Inf`
(which spell an unbounded row), and every stored entry of `P` and `A` finite. `P` and `A` may
be any `AbstractMatrix` and are never modified; the solve runs in the promotion of the five
inputs' element types.

`alg` is the algorithm, with its parameters. The keyword arguments are the fields of
[`Options`](@ref) — `max_iter`, `time_limit`, the tolerances, `scaling`, `check_dualgap`,
`polishing`, `warm_starting`, `linsys` and the rest — plus `preconditioner` and, for an
algorithm that accepts one, `accelerator`; [`setup`](@ref) describes them.

`x0` and `y0` seed the iteration in problem space. With `warm_starting = true` (the
default), a later [`solve!`](@ref) on the same workspace starts from its last point instead.

Returns a [`Solution`](@ref). Its `status` says how the run ended and
[`has_solution`](@ref) says whether its `x` and `y` are a meaningful point; failures that
are not outcomes of the algorithm — a non-symmetric `P`, a dimension mismatch, a bad
setting — raise rather than returning a status.
"""
Base.@constprop :aggressive function solve(
        P::AbstractMatrix, q::AbstractVector, A::AbstractMatrix,
        l::AbstractVector, u::AbstractVector, alg::QPAlgorithm;
        x0 = nothing, y0 = nothing, linsys::Symbol = :auto, preconditioner = nothing,
        accelerator = nothing, kwargs...
    )
    # The backend name is lifted into its `Val` here, in the frame the caller's keyword
    # reaches, rather than deeper through `setup` (see [`check_linsys`](@ref)).
    T = float(promote_type(eltype(P), eltype(q), eltype(A), eltype(l), eltype(u)))
    ws = build_workspace(
        T, alg, check_linsys(linsys), P, q, A, l, u, preconditioner, accelerator; kwargs...
    )
    if !isnothing(x0) || !isnothing(y0)
        # `solve!` cold starts when `warm_starting` is off, so a seed given alongside it
        # would be written and then discarded before the first step.
        ws.options.warm_starting || throw(
            ArgumentError(
                "x0 and y0 seed the iteration, which warm_starting = false then discards " *
                    "before the first step. Pass one or the other."
            )
        )
        warm_start!(ws; x = x0, y = y0)
    end
    return solve!(ws)
end

"""
    dimensions(ws) -> (n, m)

Number of variables and of constraint rows.
"""
dimensions(ws::QPWorkspace) = (ws.prob.n, ws.prob.m)

"""
    capabilities() -> NamedTuple

What this build of the solver supports, reported for the packages currently loaded rather
than for the package alone: `indirect_solver` is true once Krylov.jl is loaded and the
matrix-free extension exists.

There is no `error_message` counterpart: this package throws exceptions carrying their own
messages rather than returning codes to be looked up.
"""
capabilities() = (
    direct_solver = true,
    indirect_solver = !isnothing(Base.get_extension(@__MODULE__, :PureQPBaseKrylovExt)),
    codegen = false,
    update_matrices = true,
    derivatives = true,
)
