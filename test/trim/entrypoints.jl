# Entry points for the `--trim` compatibility check. Each wraps one public path with a
# concrete signature, because `juliac --trim` and TrimCheck both analyse from a call, not
# from a module.
module TrimEntry

using PureOSQP

const M = Matrix{Float64}
const V = Vector{Float64}

solve_default(P::M, q::V, A::M, l::V, u::V) = PureOSQP.solve(P, q, A, l, u)
solve_polish(P::M, q::V, A::M, l::V, u::V) = PureOSQP.solve(P, q, A, l, u; polish = true)
solve_kkt(P::M, q::V, A::M, l::V, u::V) = PureOSQP.solve(P, q, A, l, u; linsys = :kkt)
solve_unscaled(P::M, q::V, A::M, l::V, u::V) = PureOSQP.solve(P, q, A, l, u; scaling = 0)

# `verbose` prints through hand-written formatting precisely because `--trim` rejects
# Printf and `Base.stdout`. Pinned as its own entry point so that stays checked.
solve_verbose(P::M, q::V, A::M, l::V, u::V) = PureOSQP.solve(P, q, A, l, u; verbose = true)

# `time_limit` puts `time_ns` and the UInt64 budget arithmetic on the solve path.
solve_time_limited(P::M, q::V, A::M, l::V, u::V) = PureOSQP.solve(P, q, A, l, u; time_limit = 10.0)

function setup_solve_update(P::M, q::V, A::M, l::V, u::V)
    ws = PureOSQP.setup(P, q, A, l, u)
    PureOSQP.solve!(ws)
    PureOSQP.update!(ws; q = q, l = l, u = u)
    return PureOSQP.solve!(ws)
end

function warm_started(P::M, q::V, A::M, l::V, u::V)
    ws = PureOSQP.setup(P, q, A, l, u)
    s = PureOSQP.solve!(ws)
    PureOSQP.warm_start!(ws; x = s.x, y = s.y)
    return PureOSQP.solve!(ws)
end

# Negative control: reflection is exactly what `--trim` cannot resolve. If the checker
# calls this one compatible it is not discriminating and its verdict on the real entry
# points means nothing.
not_trimmable(P::M, q::V, A::M, l::V, u::V) = length(Base.return_types(PureOSQP.solve, (M, V, M, V, V)))

end # module TrimEntry
