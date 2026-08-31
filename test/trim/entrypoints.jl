# Entry points for the `--trim` compatibility check. Each wraps one public path with a
# concrete signature, because `juliac --trim` and TrimCheck both analyse from a call, not
# from a module.
module TrimEntry

using PureOSQP
using Krylov
using LinearAlgebra
using BandedMatrices

const M = Matrix{Float64}
const V = Vector{Float64}
const DM = Diagonal{Float64, Vector{Float64}}
const STM = SymTridiagonal{Float64, Vector{Float64}}
const TM = Tridiagonal{Float64, Vector{Float64}}

solve_default(P::M, q::V, A::M, l::V, u::V) = PureOSQP.solve(P, q, A, l, u)
solve_polish(P::M, q::V, A::M, l::V, u::V) = PureOSQP.solve(P, q, A, l, u; polish = true)
solve_kkt(P::M, q::V, A::M, l::V, u::V) = PureOSQP.solve(P, q, A, l, u; linsys = :kkt)
solve_unscaled(P::M, q::V, A::M, l::V, u::V) = PureOSQP.solve(P, q, A, l, u; scaling = 0)

# The diagonal backend is reached by representation rather than by a setting, so it needs a
# signature of its own to be analysed at all.
solve_diagonal(P::DM, q::V, A::DM, l::V, u::V) = PureOSQP.solve(P, q, A, l, u)
solve_tridiagonal(P::STM, q::V, A::DM, l::V, u::V) = PureOSQP.solve(P, q, A, l, u)

# The same band spelled `Tridiagonal` reaches the same backend over a different `Workspace`
# type, which is a specialization of every solve method in its own right.
solve_tridiagonal_unsym(P::TM, q::V, A::DM, l::V, u::V) = PureOSQP.solve(P, q, A, l, u)

# A Tridiagonal A squares past SymTridiagonal, so this is the BandedMatrices extension.
solve_banded(P::STM, q::V, A::TM, l::V, u::V) = PureOSQP.solve(P, q, A, l, u)

# The Woodbury backend, whose `A` is a matrix type of this package's own rather than one of
# LinearAlgebra's.
const RC = PureOSQP.RowCoupled{Float64, Matrix{Float64}, Vector{Float64}}
solve_lowrank(P::DM, q::V, A::RC, l::V, u::V) = PureOSQP.solve(P, q, A, l, u)

# The matrix-free backend exists only once Krylov is loaded, so this entry point is what
# checks that a weak dependency on the solve path does not cost the trim guarantee.
solve_indirect(P::M, q::V, A::M, l::V, u::V) = PureOSQP.solve(P, q, A, l, u; linsys = :indirect)

# An operator that supplies only products. Its `getindex` throws on a branch the trimmer
# analyses whether or not it runs, which is what keeps that message free of the string
# formatting `--trim` rejects.
# The block backend, whose `P` and `A` are both this package's own matrix type.
const BD = PureOSQP.BlockDiagonal{Float64, Matrix{Float64}}
solve_block(P::BD, q::V, A::BD, l::V, u::V) = PureOSQP.solve(P, q, A, l, u)

# The Kronecker backend. Its rung accepts only unscaled data with a scalar `P`, so the entry
# point carries `scaling = 0` the way the caller would have to.
# No Kronecker entry point yet: `solve!` on a workspace carrying a `KroneckerReduced` is an
# unresolved call to the verifier, and the cause is not isolated. The bisection that narrows
# it: `setup` alone on a `KroneckerOperator` passes, and so does a `solve` whose rung declines
# (default scaling, which lands on the dense terminal) — only a `solve` whose rung *accepts*
# fails. Ruled out along the way: the `scaling = 0` keyword (it constant-folds), the three-way
# backend union (identical to `RowCoupled`'s and `BlockDiagonal`'s, which both pass), the
# operator's type-parameter count, a `Union{Nothing,T}` in the rung (now split into a
# predicate and a value), and the operator's own `mul!`, adjoint `mul!` and `getindex`, which
# pass trim on their own. `KroneckerReduced`'s methods are `typestable, noalloc` under
# StrictMode, so this is a `--trim` resolution gap rather than a type-stability one.
const KO = PureOSQP.KroneckerOperator{Float64, Matrix{Float64}}

const PO = PureOSQP.ProductOperator{Float64, Matrix{Float64}, Vector{Float64}}
solve_operator(P::PO, q::V, A::PO, l::V, u::V) =
    PureOSQP.solve(P, q, A, l, u; scaling = 0, linsys = :indirect)

# `verbose` prints through hand-written formatting precisely because `--trim` rejects
# Printf and `Base.stdout`. Pinned as its own entry point so that stays checked.
solve_verbose(P::M, q::V, A::M, l::V, u::V) = PureOSQP.solve(P, q, A, l, u; verbose = true)

# The `try`/`catch` guarding the ADMM loop against an interrupt is on every solve path,
# so the trimmer sees it whether or not one is ever raised.
solve_interruptible(P::M, q::V, A::M, l::V, u::V) = PureOSQP.solve(P, q, A, l, u; max_iter = 50)

# `time_limit` puts `time_ns` and the UInt64 budget arithmetic on the solve path.
solve_time_limited(P::M, q::V, A::M, l::V, u::V) = PureOSQP.solve(P, q, A, l, u; time_limit = 10.0)

# The rest of the exported surface. `update_settings!` in particular compares settings
# field by field rather than looping over a tuple of symbols, because `getfield` with a
# symbol the compiler cannot see is a dynamic call -- this is what checks that reasoning.
function settings_and_rho(P::M, q::V, A::M, l::V, u::V)
    ws = PureOSQP.setup(P, q, A, l, u)
    PureOSQP.update_settings!(ws; eps_abs = 1.0e-9, rho = 0.5)
    PureOSQP.update_rho!(ws, 0.25)
    PureOSQP.cold_start!(ws)
    n, m = PureOSQP.dimensions(ws)
    c = PureOSQP.capabilities()
    return n + m + (c.direct_solver ? 1 : 0)
end

function derivatives(P::M, q::V, A::M, l::V, u::V)
    ws = PureOSQP.setup(P, q, A, l, u)
    PureOSQP.solve!(ws)
    d = PureOSQP.adjoint_derivative(ws, q, l)
    fx, fy = PureOSQP.forward_derivative(ws; dq = q)
    return d.dq[1] + fx[1] + fy[1]
end

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
