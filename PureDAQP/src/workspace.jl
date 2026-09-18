"""
    ActiveSetWorkspace

The state a dual active-set solve runs on: the [`Problem`](@ref), the resolved
[`ActiveSet`](@ref), the [`Options`](@ref), the reduction to a least-distance problem, and
the iterates in problem space.

The reduction holds the Cholesky factor of `P` (or `P + εI`) and the transformed constraint
matrix, both built once at [`setup`](@ref). A re-solve through [`update!`](@ref) keeps them
whenever `P` and `A` are unchanged, and keeps the working set too, which is what makes a
warm start cheap here.
"""
mutable struct ActiveSetWorkspace{
        T <: Real, MP <: AbstractMatrix, MA <: AbstractMatrix, V <: AbstractVector{T},
        RD <: DAQPReduction{T},
    } <: QPWorkspace{T}
    const prob::Problem{T, MP, MA, V}
    algorithm::ActiveSet{T, T, T, T}
    const options::Options{T}
    # Concretely typed: `DAQPReduction{T}` alone leaves the factorization parameter abstract,
    # which costs a dynamic dispatch on every solve. Rebound by `update!` when `P` or `A`
    # changes, which is the one thing that forces a fresh reduction.
    red::RD
    const x::V
    const y::V
    const z::V
    status::Status
    const polished::Bool
    const status_polish::PolishStatus
    iter::Int
    warm::Bool          # carry the working set into the next solve
    const setup_time::Float64
    update_time::Float64
    solve_time::Float64
end

"""
The workspace `solve(P, q, A, l, u, ActiveSet())` builds for dense data.

The guarantees on the entry points are stated against this instantiation: a `where` clause
binds its type variables to the method rather than the module, so a parametric declaration
has no concrete signature to check and needs the instantiations named.
"""
const DenseWorkspace{T} = ActiveSetWorkspace{
    T, Matrix{T}, Matrix{T}, Vector{T}, DAQPReduction{T, Cholesky{T, Matrix{T}}},
}

function Base.show(io::IO, ws::ActiveSetWorkspace{T}) where {T}
    n, m = dimensions(ws)
    print(io, "ActiveSetWorkspace{", T, "}: ", n, " variables, ", m, " rows, ")
    print(io, ws.red.ws.F.k, " rows in the working set")
    return nothing
end

"""
Stands in for the linear-system backend this method does not have.

`validate_update!` asks the backend to invalidate whatever it caches about `P` and `A`.
There is nothing to invalidate here, because [`update!`](@ref) rebuilds the whole reduction
when either changes. Passing a type of this package's own keeps that a method we are
entitled to define, rather than one attached to `Nothing`.
"""
struct NoBackend end

check_update(::NoBackend, P, A) = nothing

"Refuse what the reduction cannot represent, naming the way out."
function refuse_activeset(LS::Symbol, options::Options)
    LS in (:auto, :dense) || throw(
        ArgumentError(
            "linsys = :$LS is not available with ActiveSet(): the method reduces the problem " *
                "to a least-distance problem and maintains its own factorization, so it has no " *
                "backend to choose. Pass linsys = :auto."
        )
    )
    iszero(options.scaling) || throw(
        ArgumentError(
            "ActiveSet() needs scaling = 0: the reduction normalizes the rows of the " *
                "transformed constraint matrix itself, and equilibration on top of that would " *
                "rescale the rows the working set is priced against."
        )
    )
    return nothing
end

function setup_backend(
        alg::ActiveSet, ::Val{LS}, ::Type{T}, P::AbstractMatrix, q::AbstractVector,
        A::AbstractMatrix, l::AbstractVector, u::AbstractVector, options::Options,
        preconditioner, accelerator
    ) where {LS, T <: Real}
    t0 = time_ns()
    isnothing(accelerator) || throw(
        ArgumentError("accelerator is used only by OperatorSplitting: this method has no fixed-point iteration to accelerate.")
    )
    isnothing(preconditioner) || throw(
        ArgumentError("preconditioner is used only by the matrix-free backend, which ActiveSet() does not have.")
    )
    refuse_activeset(LS, options)

    n, m = validate(P, q, A, l, u)
    resolved = element_typed(alg, T, options)
    is_materializable(P) && is_materializable(A) || throw(
        ArgumentError(
            "ActiveSet() needs P and A it can read entry by entry: the reduction forms " *
                "A / R for the Cholesky factor R of P, which an operator cannot supply."
        )
    )
    # Convexity is not checked here: `reduce_qp` factors `P + eps_prox*I` and reports a
    # failure, which is the same question asked once instead of twice.
    prob = validated_problem(T, n, m, P, q, A, l, u, options.scaling)
    # `convert` rather than `Matrix{T}`/`Vector{T}`: those copy even when the argument
    # already has the type asked for, and the reduction only reads this data.
    iseq = [prob.l0[i] == prob.u0[i] for i in 1:m]
    red = reduce_qp(
        convert(Matrix{T}, P), convert(Matrix{T}, A),
        convert(Vector{T}, prob.u0), convert(Vector{T}, prob.l0),
        iseq; eps_prox = resolved.eps_prox
    )

    ws = ActiveSetWorkspace{T, typeof(prob.P), typeof(prob.A), typeof(prob.q), typeof(red)}(
        prob, resolved, options, red,
        zeros(T, n), zeros(T, m), zeros(T, m),
        UNSOLVED, false, POLISH_NOT_PERFORMED, 0, false,
        (time_ns() - t0) / 1.0e9, 0.0, 0.0,
    )
    return ws
end

"""
    solve!(ws) -> Solution

Run the dual active-set method from the workspace's state.

The working set carries over from the previous solve when one has run and nothing has
invalidated it, so a re-solve after [`update!`](@ref) starts from the previous answer's
active rows. [`cold_start!`](@ref) drops it.
"""
function solve!(ws::ActiveSetWorkspace{T}) where {T}
    t0 = time_ns()
    prob, alg = ws.prob, ws.algorithm
    ws.warm || reset_working_set!(ws.red)

    x, status, iters = run_daqp!(
        ws.red, prob.q0;
        max_iter = ws.options.max_iter, zero_tol = alg.zero_tol,
        primal_tol = alg.primal_tol, eps_prox = alg.eps_prox,
        eta_prox = alg.eta_prox, max_prox = alg.max_prox,
    )
    ws.iter = iters
    ws.warm = true

    if status == LDP_OPTIMAL
        copyto!(ws.x, x)
        multipliers!(ws.y, ws.red)
        mul!(ws.z, prob.A, ws.x)
        ws.status = SOLVED
    elseif status == LDP_INFEASIBLE
        fill!(ws.x, T(NaN))
        fill!(ws.y, T(NaN))
        fill!(ws.z, T(NaN))
        ws.status = PRIMAL_INFEASIBLE
    elseif status == LDP_ITERATION_LIMIT
        ws.status = MAX_ITER_REACHED
    else
        ws.status = NUMERICAL_ERROR
    end
    ws.solve_time = (time_ns() - t0) / 1.0e9
    return build_solution(ws)
end

@strict_function signatures = [(DenseWorkspace{Float64},)] function warm_start!(ws::ActiveSetWorkspace{T}; x = nothing, y = nothing) where {T}
    prob = ws.prob
    if !isnothing(x)
        length(x) == prob.n || throw(ArgumentError("length(x) must be $(prob.n)"))
        all(isfinite, x) || throw(ArgumentError("x must be finite, found NaN or Inf"))
        ws.x .= T.(x)
    end
    if !isnothing(y)
        length(y) == prob.m || throw(ArgumentError("length(y) must be $(prob.m)"))
        all(isfinite, y) || throw(ArgumentError("y must be finite, found NaN or Inf"))
        ws.y .= T.(y)
    end
    # The working set, not the point, is what this method restarts from, and the caller has
    # given a point. Keep whatever working set is there: it is the best guess available.
    ws.warm = true
    return ws
end

@strict_function signatures = [(DenseWorkspace{Float64},)] function cold_start!(ws::ActiveSetWorkspace{T}) where {T}
    fill!(ws.x, zero(T))
    fill!(ws.y, zero(T))
    fill!(ws.z, zero(T))
    ws.warm = false
    ws.status = UNSOLVED
    return ws
end

"""
    update!(ws; q, l, u, P, A) -> ws

Replace problem data. Changing `q`, `l` or `u` keeps the Cholesky factor and the transformed
constraint matrix, so only the right-hand side is rebuilt. Changing `P` or `A` rebuilds the
reduction, which is the expensive path.
"""
function update!(
        ws::ActiveSetWorkspace{T}; q = nothing, l = nothing, u = nothing, P = nothing, A = nothing
    ) where {T}
    t0 = time_ns()
    prob = ws.prob
    validate_update!(prob, NoBackend(); P, A, q, l, u)
    adopt_update!(prob; P, A, q, l, u)
    if !isnothing(P) || !isnothing(A)
        m = prob.m
        iseq = [prob.l0[i] == prob.u0[i] for i in 1:m]
        ws.red = reduce_qp(
            convert(Matrix{T}, prob.P), convert(Matrix{T}, prob.A),
            convert(Vector{T}, prob.u0), convert(Vector{T}, prob.l0),
            iseq; eps_prox = ws.algorithm.eps_prox
        )
        ws.warm = false
    elseif !isnothing(l) || !isnothing(u)
        rebuild_bounds!(ws.red, Vector{T}(prob.u0), Vector{T}(prob.l0))
    end
    ws.update_time += (time_ns() - t0) / 1.0e9
    return ws
end

function update_settings!(ws::ActiveSetWorkspace{T}, alg::ActiveSet) where {T}
    new = element_typed(alg, T, ws.options)
    new.eps_prox == ws.algorithm.eps_prox || throw(
        ArgumentError(
            "eps_prox is built into the factorization of P + eps_prox*I, so it cannot be " *
                "changed on an existing workspace. Build a new one with setup."
        )
    )
    ws.algorithm = new
    return ws
end

# An active-set solution puts every inactive row's multiplier at exactly zero, which is what
# the active-set test reads, so no polishing is needed before a derivative.
derivative_ready(::ActiveSetWorkspace) = nothing
