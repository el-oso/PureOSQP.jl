# The reference C implementation, kept apart from `helpers.jl` so that only the items that
# compare against it load OSQP. Loading a package costs the same whether or not the item calls
# it, and compilation is most of this suite's wall time.
using LinearAlgebra, SparseArrays, OSQP

"""
    osqp_ref(P, q, A, l, u; kwargs...)

Run the reference C implementation on the same problem. `adaptive_rho_interval` is always
pinned: libosqp 0.6.2 otherwise adapts on wall-clock time, which makes iteration counts
machine-dependent.
"""
function osqp_ref(P, q, A, l, u; kwargs...)
    model = OSQP.Model()
    OSQP.setup!(
        model; P = sparse(Symmetric(Matrix(P))), q = collect(q),
        A = sparse(Matrix(A)), l = collect(l), u = collect(u), verbose = false,
        adaptive_rho_interval = 50, check_termination = 25, kwargs...
    )
    return OSQP.solve!(model)
end
