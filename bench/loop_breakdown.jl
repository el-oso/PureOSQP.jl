# What one pass through the ADMM loop costs, by part.
#
# `bench/suite_split.jl` gives a per-iteration cost for each solver. This says where ours
# goes: the ADMM step itself, the residual updates that run only on a check, and the
# refactorizations `adapt_rho!` triggers — which are charged to the loop, not to setup, and
# on a short run are a large share of it.
using PureOSQP, LinearAlgebra, SparseArrays, Chairmarks, Printf

BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "suite_problems.jl"))

const OPTS = (eps_abs = 1.0e-5, eps_rel = 1.0e-5, max_iter = 20_000, check_dualgap = false)

"""
    steps(P, q, A, l, u, k) -> seconds

Cost of `k` ADMM steps, timed over a workspace the setup expression makes fresh for each
sample.

Stepping one workspace for as long as a benchmark wants to sample runs it far past where the
solver stops, and the iterates then reach denormals — the measurement ends up about those
rather than about the step. `k` is the iteration count the real solve takes, so each sample
walks the real trajectory.
"""
function steps(P, q, A, l, u, k)
    return (
        @b PureOSQP.setup(P, q, A, l, u; OPTS...) for _ in 1:k
            PureOSQP.admm_step!(_)
        end
    ).time
end

println("\nPer-iteration cost by part, against the refactorizations charged to the loop.\n")
@printf(
    "%-12s %5s %5s %6s | %8s %8s %8s | %9s %9s\n",
    "class", "iter", "refac", "checks", "step µs", "resid µs", "factor µs",
    "step tot", "refac tot"
)
println("-"^96)
for (name, build) in CASES
    P, q, A, l, u = build()
    sol = PureOSQP.solve(P, q, A, l, u; OPTS...)
    ws = PureOSQP.setup(P, q, A, l, u; OPTS...)
    PureOSQP.solve!(ws)
    # `setup` counts one, so the rest are the loop's.
    refac = ws.refactor_count - 1
    checks = ws.iter ÷ ws.settings.check_termination
    t_all = steps(P, q, A, l, u, sol.iter)
    t_step = t_all / sol.iter
    fresh = PureOSQP.setup(P, q, A, l, u; OPTS...)
    t_res = (@b PureOSQP.update_residuals!(fresh)).time
    t_fac = (@b PureOSQP.factorize!(fresh.linsys, fresh)).time
    @printf(
        "%-12s %5d %5d %6d | %8.2f %8.2f %8.2f | %7.2fms %7.2fms\n",
        name, sol.iter, refac, checks,
        1.0e6t_step, 1.0e6t_res, 1.0e6t_fac,
        1.0e3t_all, 1.0e3refac * t_fac
    )
    flush(stdout)
end
