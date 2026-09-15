"""
    update_preconditioner!(M, prob, wt, k::Int) -> M

Refresh the preconditioner of `P + wt.sigma*I + A' * Diagonal(wt.w) * A`, where `P` and `A`
are the matrices or operators passed to [`setup`](@ref) (a caller-supplied preconditioner
requires `scaling = 0`, so no equilibration intervenes). `wt.w` and `wt.sigma` are the only
documented reads of `wt`; `prob` is passed for dispatch and is not part of the documented
interface.

`k` is the refresh index the algorithm sets before calling: under ADMM, the number of
refactorizations so far.

Called from the matrix-free backend's `factorize!` and `refactor_weights!`, before the solves
that use it. The returned object replaces `M` and must have the same type; another type throws
an `ArgumentError`. Refresh lazily by returning `M` unchanged.

`M` is applied through `LinearAlgebra.ldiv!(y, M, x)`, so a `Cholesky`, `LDLt` or any
factorization object is usable as it is, and `M` must be symmetric positive definite. The
default method never refreshes.
"""
update_preconditioner!(M, prob, wt, k::Int) = M

"""
    IdentityPreconditioner()

No preconditioning: conjugate gradients on the reduced system as it stands. Needs nothing
from `P` or `A`, so it runs with equilibration on.
"""
struct IdentityPreconditioner end

"""
    JacobiPreconditioner(dinv)

The inverted diagonal of the reduced matrix, which [`reduced_diagonal!`](@ref) fills at every
[`update_preconditioner!`](@ref) from the current weights. It is the matrix-free backend's
default, and it runs with equilibration on.

`ldiv!(y, J, x)` computes `y = dinv .* x`. It multiplies by the stored reciprocal rather than
dividing by the diagonal, and the two differ in the last bit on about a quarter of entries,
so a `Diagonal` of either vector is not a substitute.
"""
mutable struct JacobiPreconditioner{V <: AbstractVector}
    const dinv::V
end

LinearAlgebra.ldiv!(y::AbstractVector, J::JacobiPreconditioner, x::AbstractVector) =
    multiply!(y, J.dinv, x)

function update_preconditioner!(J::JacobiPreconditioner, prob, wt, k::Int)
    reduced_diagonal!(
        J.dinv, eltype(J.dinv), prob.P, prob.A, wt.w, prob.E, prob.D, wt.sigma, prob.c
    )
    return J
end

"""
    set_refresh_index!(ls, k) -> Nothing

Hand the backend the refresh index its next [`update_preconditioner!`](@ref) passes on. A
backend without a preconditioner ignores it, which is the default.
"""
set_refresh_index!(ls::LinearSystem, k::Int) = nothing

"""
    use_residual_stop!(ls, on::Bool) -> Nothing

Choose the inner stopping rule of an iterative backend. Off (the default), conjugate gradients
stops at the absolute tolerance on its preconditioned residual norm. On, it stops once the
two-norm of its recursively updated, unpreconditioned residual reaches the tolerance, or at
its own machine-precision stop. A direct backend ignores it.
"""
use_residual_stop!(ls::LinearSystem, on::Bool) = nothing

"""
    last_solve_converged(ls) -> Bool

Whether the backend's most recent solve met its stopping test. An iterative solve that spent
its whole iteration budget, or broke down, did not. A direct backend always does.
"""
last_solve_converged(ls::LinearSystem) = true

"""
    inner_iterations(ls) -> Int

The inner iterations an iterative backend has spent over its life; zero for a direct one.
`Solution.cg_iters` is its difference across one solve.
"""
inner_iterations(ls::LinearSystem) = 0
