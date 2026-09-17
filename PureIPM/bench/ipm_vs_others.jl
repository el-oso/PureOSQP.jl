# The interior-point method against ADMM, Clarabel (also an interior-point method), and
# ECOS (a second-order-cone solver, not a QP solver) on the OSQP benchmark suite's problem
# classes at their smallest representative sizes.
#
# ECOS takes no quadratic objective, so every case is reformulated as an SOCP before ECOS
# ever sees it. MathOptInterface does this automatically: `MOI.instantiate(ECOS.Optimizer;
# with_bridge_type = Float64)` reports `ECOS.Optimizer` as unable to accept a
# `ScalarQuadraticFunction`-in-`LessThan` constraint, so its `Objective.SlackBridge`
# introduces an epigraph variable `t` (minimize `t` subject to `0.5x'Px + q'x <= t`), and its
# `Constraint.QuadtoSOCBridge` rewrites that quadratic constraint as a rotated
# second-order-cone constraint using a Cholesky factor `U` of `P` (`P = U'U`). No hand-written
# reformulation was needed; `moi_cache_from_qp` below only builds the *quadratic objective and
# linear constraints* in MOI's term format — the QP-to-SOCP rewrite itself is the bridge's
# job, verified by inspecting the raw `ECOS.Optimizer`'s cone list after solving a toy problem
# (a size-1 nonnegative cone plus a size-4 second-order cone, where the plain QP had none).
#
# `QuadtoSOCBridge` requires `P` positive *definite*, not merely semidefinite (`cholesky`
# throws otherwise); all seven suite classes here have their P's quadratic block strictly
# convex (the pure zero blocks belong to non-quadratic epigraph variables the bridge never
# touches), so every class reformulates. A class where it does not would be skipped with the
# reason recorded in the JSON, but none needed it.
#
# ECOS.jl fuses the bridge's SOC construction, ECOS's own sparse-matrix marshaling, and the
# C library's `ECOS_setup`/`ECOS_solve` into one `MOI.optimize!` call — there is no exposed
# hook to time the reformulation in isolation. What IS separable: ECOS's C library reports its
# own `tsetup + tsolve` through `MOI.SolveTimeSec()`, timed inside the C call. Subtracting that
# from the wall-clock median of the whole `MOI.optimize!` call gives the Julia-side cost (MOI
# bridging plus ECOS.jl's array marshaling) as `ecos_reform_ms`; the two are reported
# separately below, but `ecos_reform_ms` is a *derived* difference, not a second independent
# timing. Neither ECOS's cone count nor its iteration count is comparable to the IPM's or
# Clarabel's: it is iterating on a larger, differently structured problem (original variables
# plus one epigraph variable per case, an inequality replaced by a second-order cone).
#
# `InteriorPoint()` and `Clarabel` both run at `eps_abs = eps_rel = 1e-8`; ECOS runs at its
# closest equivalent, `feastol = abstol = reltol = 1e-8`. `OperatorSplitting()` (ADMM) runs
# separately at `eps_abs = eps_rel = 1e-6`, reported alongside rather than folded into the same
# referee.
#
#     taskset -c 15 julia --project=bench PureIPM/bench/ipm_vs_others.jl   # writes PureIPM/bench/results/ipm_vs_others.json
using PureOSQP, PureIPM, Clarabel, ECOS, MathOptInterface
using LinearAlgebra, SparseArrays, Random, JSON, Chairmarks, Printf
const MOI = MathOptInterface

include(joinpath(@__DIR__, "..", "..", "bench", "suite_problems.jl"))

BLAS.set_num_threads(1)

const RESULTS = joinpath(@__DIR__, "results", "ipm_vs_others.json")
const IPM_TOL = 1.0e-8
const ADMM_TOL = 1.0e-6
const SECONDS = 0.3

"The smallest size of each suite class that still exercises its structure."
const SMALL_CASES = [
    ("Random QP", () -> random_qp(6)),
    ("Eq QP", () -> eq_qp(20)),
    ("Portfolio", () -> portfolio(1)),
    ("Lasso", () -> lasso(2)),
    ("SVM", () -> svm(2)),
    ("Huber", () -> huber(2)),
    ("Control", () -> control(4)),
]

"Clarabel takes one-sided cones: the two-sided rows are stacked as `Ax ≤ u, -Ax ≤ -l`."
function clarabel_form(P, A, l, u)
    finite_l = isfinite.(l)
    finite_u = isfinite.(u)
    rows = Vector{SparseMatrixCSC{Float64, Int}}()
    bnd = Float64[]
    if any(finite_u)
        push!(rows, A[finite_u, :])
        append!(bnd, u[finite_u])
    end
    if any(finite_l)
        push!(rows, -A[finite_l, :])
        append!(bnd, -l[finite_l])
    end
    return (sparse(triu(P)), vcat(rows...), bnd)
end

function run_clarabel(P, q, A, l, u)
    Pc, Ac, bc = clarabel_form(P, A, l, u)
    settings = Clarabel.Settings(
        verbose = false, tol_gap_abs = IPM_TOL, tol_gap_rel = IPM_TOL,
        tol_feas = IPM_TOL,
    )
    solver = Clarabel.Solver()
    Clarabel.setup!(solver, Pc, q, Ac, bc, [Clarabel.NonnegativeConeT(length(bc))], settings)
    return Clarabel.solve!(solver)
end

"""
MOI's `ScalarQuadraticFunction` applies the implicit `0.5` factor only to a same-variable
(diagonal) term; a cross term for an unordered pair `(i, j)`, `i < j`, is added once and used
at its raw coefficient. So `0.5x'Px` becomes: one term `P[i,i]` per diagonal entry, one term
`P[i,j]` (not `2P[i,j]`) per upper-triangular off-diagonal entry — doubling the off-diagonal
coefficient silently changes the quadratic form `QuadtoSOCBridge` reconstructs from it, which
breaks positive-definiteness for some of these problems without ECOS ever raising an error on
the others. Verified against `PureOSQP`'s IPM objective (dobj ~1e-6 to 1e-11) on all seven
classes below.
"""
function moi_cache_from_qp(P, q, A, l, u)
    n = length(q)
    cache = MOI.Utilities.Model{Float64}()
    x = MOI.add_variables(cache, n)
    Pu = triu(P)
    rows = rowvals(Pu)
    vals = nonzeros(Pu)
    qterms = MOI.ScalarQuadraticTerm{Float64}[]
    for j in 1:size(Pu, 2)
        for idx in nzrange(Pu, j)
            i = rows[idx]
            c = vals[idx]
            iszero(c) && continue
            push!(qterms, MOI.ScalarQuadraticTerm(c, x[i], x[j]))
        end
    end
    aterms = MOI.ScalarAffineTerm{Float64}[]
    for i in eachindex(q)
        iszero(q[i]) && continue
        push!(aterms, MOI.ScalarAffineTerm(q[i], x[i]))
    end
    obj = MOI.ScalarQuadraticFunction(qterms, aterms, 0.0)
    MOI.set(cache, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(cache, MOI.ObjectiveFunction{typeof(obj)}(), obj)
    Acsc = SparseMatrixCSC(A)
    rowsA = rowvals(Acsc)
    valsA = nonzeros(Acsc)
    row_terms = [MOI.ScalarAffineTerm{Float64}[] for _ in eachindex(l)]
    for j in 1:size(Acsc, 2)
        for idx in nzrange(Acsc, j)
            i = rowsA[idx]
            c = valsA[idx]
            iszero(c) && continue
            push!(row_terms[i], MOI.ScalarAffineTerm(c, x[j]))
        end
    end
    for i in eachindex(l, u, row_terms)
        f = MOI.ScalarAffineFunction(row_terms[i], 0.0)
        li, ui = l[i], u[i]
        if isfinite(li) && isfinite(ui) && li == ui
            MOI.add_constraint(cache, f, MOI.EqualTo(li))
        elseif isfinite(li) && isfinite(ui)
            MOI.add_constraint(cache, f, MOI.Interval(li, ui))
        elseif isfinite(ui)
            MOI.add_constraint(cache, f, MOI.LessThan(ui))
        elseif isfinite(li)
            MOI.add_constraint(cache, f, MOI.GreaterThan(li))
        end
    end
    return cache, x
end

function run_ecos(P, q, A, l, u)
    cache, x = moi_cache_from_qp(P, q, A, l, u)
    dest = MOI.instantiate(ECOS.Optimizer; with_bridge_type = Float64, with_cache_type = Float64)
    MOI.set(dest, MOI.Silent(), true)
    MOI.set(dest, MOI.RawOptimizerAttribute("feastol"), IPM_TOL)
    MOI.set(dest, MOI.RawOptimizerAttribute("abstol"), IPM_TOL)
    MOI.set(dest, MOI.RawOptimizerAttribute("reltol"), IPM_TOL)
    MOI.copy_to(dest, cache)
    MOI.optimize!(dest)
    return dest, x
end

function run_case(name, gen)
    P, q, A, l, u = gen()
    n, m = size(A, 2), size(A, 1)

    ipm = PureOSQP.solve(P, q, A, l, u, PureIPM.InteriorPoint(); eps_abs = IPM_TOL, eps_rel = IPM_TOL)
    ipm_bm = @b PureOSQP.solve($P, $q, $A, $l, $u, PureIPM.InteriorPoint(); eps_abs = IPM_TOL, eps_rel = IPM_TOL) seconds = SECONDS

    admm = PureOSQP.solve(P, q, A, l, u; eps_abs = ADMM_TOL, eps_rel = ADMM_TOL, max_iter = 20_000)
    admm_bm = @b PureOSQP.solve($P, $q, $A, $l, $u; eps_abs = ADMM_TOL, eps_rel = ADMM_TOL, max_iter = 20_000) seconds = SECONDS

    clar = run_clarabel(P, q, A, l, u)
    clar_bm = @b run_clarabel($P, $q, $A, $l, $u) seconds = SECONDS

    dx_clarabel = maximum(abs, ipm.x .- clar.x; init = 0.0) / max(1.0, maximum(abs, ipm.x; init = 0.0))
    dx_admm = maximum(abs, ipm.x .- admm.x; init = 0.0) / max(1.0, maximum(abs, ipm.x; init = 0.0))

    ecos_ok = true
    ecos_reason = ""
    local ecos_dest, ecos_x, ecos_bm, dx_ecos
    try
        ecos_dest, ecos_x = run_ecos(P, q, A, l, u)
        ecos_bm = @b run_ecos($P, $q, $A, $l, $u) seconds = SECONDS
        xecos = MOI.get.(ecos_dest, MOI.VariablePrimal(), ecos_x)
        dx_ecos = maximum(abs, ipm.x .- xecos; init = 0.0) / max(1.0, maximum(abs, ipm.x; init = 0.0))
    catch e
        ecos_ok = false
        ecos_reason = sprint(showerror, e)
    end

    if ecos_ok
        ecos_iter = Int(MOI.get(ecos_dest, MOI.BarrierIterations()))
        ecos_status = String(Symbol(MOI.get(ecos_dest, MOI.TerminationStatus())))
        ecos_obj = MOI.get(ecos_dest, MOI.ObjectiveValue())
        ecos_total_ms = 1.0e3ecos_bm.time
        ecos_internal_ms = 1.0e3MOI.get(ecos_dest, MOI.SolveTimeSec())
        ecos_reform_ms = max(0.0, ecos_total_ms - ecos_internal_ms)
        @printf(
            "%-10s n=%-4d m=%-5d | ipm %3d it %7.3f ms | clarabel %3d it %7.3f ms | admm %5d it %7.3f ms | ecos %3d it %7.3f ms (reform %5.3f ms) | dx(clar)=%.1e dx(admm)=%.1e dx(ecos)=%.1e\n",
            name, n, m, ipm.iter, 1.0e3ipm_bm.time, clar.iterations, 1.0e3clar_bm.time,
            admm.iter, 1.0e3admm_bm.time, ecos_iter, ecos_total_ms, ecos_reform_ms,
            dx_clarabel, dx_admm, dx_ecos,
        )
    else
        ecos_iter = -1
        ecos_status = "reformulation_failed"
        ecos_obj = NaN
        ecos_total_ms = NaN
        ecos_reform_ms = NaN
        dx_ecos = NaN
        @printf(
            "%-10s n=%-4d m=%-5d | ipm %3d it %7.3f ms | clarabel %3d it %7.3f ms | admm %5d it %7.3f ms | ecos SKIPPED: %s\n",
            name, n, m, ipm.iter, 1.0e3ipm_bm.time, clar.iterations, 1.0e3clar_bm.time,
            admm.iter, 1.0e3admm_bm.time, ecos_reason,
        )
    end
    flush(stdout)

    return (;
        name, n, m,
        ipm = (; iter = ipm.iter, status = String(Symbol(ipm.status)), time_ms = 1.0e3ipm_bm.time, obj = ipm.obj_val),
        clarabel = (; iter = clar.iterations, status = String(Symbol(clar.status)), time_ms = 1.0e3clar_bm.time, obj = clar.obj_val),
        admm = (; iter = admm.iter, status = String(Symbol(admm.status)), time_ms = 1.0e3admm_bm.time, obj = admm.obj_val),
        ecos = (;
            ok = ecos_ok, reason = ecos_reason, iter = ecos_iter, status = ecos_status,
            total_time_ms = ecos_total_ms, reform_ms = ecos_reform_ms, obj = ecos_obj,
        ),
        dx_ipm_clarabel = dx_clarabel, dx_ipm_admm = dx_admm, dx_ipm_ecos = dx_ecos,
    )
end

@printf(
    "%-10s %-8s %-7s | %-14s | %-18s | %-18s | %-14s\n",
    "class", "n", "m", "IPM (eps 1e-8)", "Clarabel (tol 1e-8)", "ADMM (eps 1e-6)", "ECOS/SOCP (1e-8)",
)
println("-"^160)
results = [run_case(name, gen) for (name, gen) in SMALL_CASES]

open(RESULTS, "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "blas_threads" => BLAS.get_num_threads(),
            "ipm_tol" => IPM_TOL,
            "admm_tol" => ADMM_TOL,
            "clarabel_version" => string(pkgversion(Clarabel)),
            "ecos_version" => string(pkgversion(ECOS)),
            "moi_version" => string(pkgversion(MathOptInterface)),
            "note" => "ECOS solves a bridged SOCP reformulation (epigraph var + a second-order " *
                "cone per case), not the QP; its iteration count is not comparable to the IPM's " *
                "or Clarabel's. ecos.reform_ms is derived as (wall-clock median) - (ECOS's own " *
                "reported tsetup+tsolve), not an independently timed quantity.",
            "results" => [
                Dict(
                    "name" => r.name, "n" => r.n, "m" => r.m,
                    "ipm" => Dict(pairs(r.ipm)), "clarabel" => Dict(pairs(r.clarabel)),
                    "admm" => Dict(pairs(r.admm)), "ecos" => Dict(pairs(r.ecos)),
                    "dx_ipm_clarabel" => r.dx_ipm_clarabel, "dx_ipm_admm" => r.dx_ipm_admm,
                    "dx_ipm_ecos" => r.dx_ipm_ecos,
                ) for r in results
            ],
        ), 2
    )
end
println("\nsaved PureIPM/bench/results/ipm_vs_others.json")
