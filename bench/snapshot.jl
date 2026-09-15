# A fixed-settings snapshot of ADMM's behavior, checked against `design/modular-ipm.md`: a
# change to the linear-system code must leave every case's iterates identical.
#
# For every OSQP benchmark suite class (`suite_problems.jl`) and every structured family
# `test/selection_tests.jl` asserts backend selection against, plus a sample of the ladder's
# explicitly named backends, the block and Kronecker backends, the two matrix-free operator
# routes, an Anderson-accelerated solve, an `update!`-and-resolve cycle and the KKT-error
# adaptive-ρ schedule, it solves at fixed settings and records `(backend_name, iter,
# refactor_count, status, round(obj_val, 10), accel_declined)`.
#
# `bench/results/snapshot_s0.json` is a floating-point artifact: the objective is rounded to
# ten digits, which still leaves it sensitive to summation order, so a different BLAS build or
# a different core count is not guaranteed to reproduce it bit-for-bit. It is compared only on
# the machine that generated it.
#
# Not a test item: `julia --project=bench bench/snapshot.jl` regenerates it, and
# `julia --project=bench bench/snapshot.jl --check` recomputes on the current tree and reports
# every mismatch against the saved copy, exiting nonzero if there is one.
#
# Every extension is loaded for the whole run, so a suite class's `:auto` choice is made with
# every backend available at once — `test/selection_tests.jl`'s own reference does the same,
# which is why `Lasso`, `SVM` and `Huber` land on `:ldlfactorizations` and `Portfolio` on
# `:ldl_kkt` below rather than on a `SparseArrays`-only backend.
using PureOSQP, LinearAlgebra, SparseArrays, Random, JSON
using LDLFactorizations, BandedMatrices, Krylov, LinearMaps, COSMOAccelerators

BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "suite_problems.jl"))
include(joinpath(@__DIR__, "structured_problems.jl"))

const RESULTS = joinpath(@__DIR__, "results", "snapshot_s0.json")
const OPTS = (eps_abs = 1.0e-6, eps_rel = 1.0e-6, max_iter = 20_000)

"""
`ProductOperator` wraps a dense objective and constraint matrix that carry no factorizable
entries, so the ladder declines the dense terminal and lands on the matrix-free rung — the
package's own products-only protocol, as opposed to the `LinearMaps` route below.
"""
function productoperator_problem(n)
    Random.seed!(9201)
    S = randn(n, n)
    P = PureOSQP.ProductOperator{Float64}(Symmetric(S'S ./ n + 8I); symmetric = true, posdef = true)
    A = PureOSQP.ProductOperator{Float64}(randn(n, n) ./ sqrt(n))
    b = randn(n)
    return (P, randn(n), A, b .- rand(n), b .+ rand(n))
end

"""
The other matrix-free route: a `LinearMaps.LinearMap` handed to `setup` directly, which the
LinearMaps extension turns into a `ProductOperator` rather than the caller doing it.
"""
function linearmap_problem(n)
    Random.seed!(9202)
    d = rand(n) .+ 2.0
    v = randn(n) ./ sqrt(n)
    alpha = 0.5
    apply! = (y, x) -> (y .= d .* x .+ (alpha * dot(v, x)) .* v)
    P = LinearMap{Float64}(apply!, n; ismutating = true, issymmetric = true, isposdef = true)
    A = randn(n, n) ./ sqrt(n)
    b = A * randn(n)
    return (P, randn(n), A, b .- rand(n), b .+ rand(n))
end

"""
A `KroneckerOperator` `A` against a scalar multiple of the identity `P`, `test/kronecker_tests.jl`'s
reference case for the Kronecker backend. `kronecker_rung` requires uniform `ρ` and identity
scaling, hence `scaling = 0`.
"""
function kronecker_problem()
    Random.seed!(9302)
    n1, n2 = 12, 10
    A1, A2 = randn(n1, n1), randn(n2, n2)
    K = PureOSQP.KroneckerOperator(A1, A2)
    n = n1 * n2
    P = Diagonal(fill(2.0, n))
    q = randn(n)
    b = kron(A1, A2) * randn(n)
    return (P, q, K, b .- rand(n), b .+ rand(n))
end

"A dense QP, the same shape `test/solve_tests.jl` builds its Anderson-accelerated cases from."
function anderson_problem()
    Random.seed!(9303)
    n, m = 60, 120
    X = randn(n, n)
    A = randn(m, n)
    q = randn(n)
    b = A * randn(n)
    return (Matrix(Symmetric(X'X / n + I)), q, A, b .- 1, b .+ 1)
end

"""
    run_update_case() -> NamedTuple

Set a problem up, solve it, `update!` `q`, `l` and `u`, and solve again. The recorded `iter`
is the second solve's; `refactor_count` accumulates across both, since it lives on the
workspace rather than being reset per solve.
"""
function run_update_case()
    Random.seed!(9304)
    n, m = 60, 120
    X = randn(n, n)
    A = randn(m, n)
    P = Matrix(Symmetric(X'X / n + I))
    q = randn(n)
    b = A * randn(n)
    ws = PureOSQP.setup(P, q, A, b .- 1, b .+ 1; OPTS..., linsys = :auto, scaling = 10)
    PureOSQP.solve!(ws)
    b2 = A * randn(n)
    PureOSQP.update!(ws; q = randn(n), l = b2 .- 1, u = b2 .+ 1)
    PureOSQP.solve!(ws)
    PureOSQP.has_solution(ws.status) ||
        @warn "update/resolve/auto did not reach a solved status" status = PureOSQP.status_name(ws.status)
    return (
        backend_name = String(PureOSQP.backend_name(ws.linsys)),
        iter = ws.iter,
        refactor_count = ws.refactor_count,
        status = PureOSQP.status_name(ws.status),
        obj_val = round(ws.obj_val; digits = 10),
        accel_declined = ws.accel_declined,
    )
end

suite_build = Dict(CASES)
families = structured_families(100)
family_build = Dict(f.name => (() -> (f.P, f.q, f.A, f.l, f.u)) for f in families)

"""
    mkcase(name, build, linsys, scaling; accelerator, extra_opts, expect) -> NamedTuple

One case: `build()` returns `(P, q, A, l, u)`. `accelerator`, when given, is `(n, m) ->
accelerator` (the accelerator itself needs the problem's size). `extra_opts` are merged into
the fixed `OPTS` on top of `linsys` and `scaling`. `expect`, when given, is the backend name
`run_case` asserts against before recording anything.
"""
mkcase(
    name, build, linsys, scaling; accelerator = nothing, extra_opts = NamedTuple(),
    expect = nothing,
) = (; name, build, linsys, scaling, accelerator, extra_opts, expect)

cases = NamedTuple[]

# Every suite class at `:auto`, the reference `test/selection_tests.jl` checks selection
# against. `Lasso`, `SVM` and `Huber` land on the `LDLFactorizations` extension's
# `:ldlfactorizations`, `Portfolio` on its `:ldl_kkt` — both asserted, since neither has its
# own dedicated construction here.
suite_expect = Dict(
    "Lasso" => :ldlfactorizations, "SVM" => :ldlfactorizations, "Huber" => :ldlfactorizations,
    "Portfolio" => :ldl_kkt,
)
for (name, build) in CASES
    push!(
        cases,
        mkcase(
            "suite/$name/auto", build, :auto, 10;
            expect = get(suite_expect, name, nothing)
        )
    )
end

# A sample of suite classes forced onto a named backend, so a later change to the ladder is
# caught even where `:auto` still lands on the same rung.
for (name, ls) in (
        ("Random QP", :dense), ("Random QP", :indirect), ("Random QP", :sparse),
        ("Eq QP", :dense), ("Portfolio", :kkt),
        ("Lasso", :sparse), ("SVM", :sparse), ("Huber", :sparse), ("Control", :sparse),
    )
    push!(cases, mkcase("suite/$name/$ls", suite_build[name], ls, 10))
end

# Every structured family at `:auto`. `banded_tridiag` and `banded_wide_inside` are the
# `BandedMatrices` extension's `:banded` backend, asserted here since `:banded` names no kind
# `linsys` can request outright.
family_expect = Dict("banded_tridiag" => :banded, "banded_wide_inside" => :banded)
for f in families
    push!(
        cases,
        mkcase(
            "family/$(f.name)/auto", family_build[f.name], :auto, 10;
            expect = get(family_expect, f.name, nothing)
        )
    )
end

# Families forced onto the named backend their structure admits.
for (name, ls) in (
        ("diagonal", :diagonal), ("diagonal", :indirect),
        ("tridiagonal_diag", :tridiagonal), ("tridiagonal_bidiag", :tridiagonal),
        ("lowrank_rank3", :lowrank), ("lowrank_rank10", :lowrank),
    )
    push!(cases, mkcase("family/$name/$ls", family_build[name], ls, 10))
end

# The two matrix-free operator routes, both `:auto`: neither is materializable, so the ladder
# reaches `:indirect` on its own. Equilibration walks columns, which a products-only operator
# cannot answer, hence `scaling = 0`.
push!(
    cases,
    mkcase("operator/productoperator/auto", () -> productoperator_problem(200), :auto, 0)
)
push!(cases, mkcase("operator/linearmap/auto", () -> linearmap_problem(200), :auto, 0))

# The block and Kronecker backends, both at `:auto` and forced by name.
push!(cases, mkcase("block/auto", block_problem, :auto, 10; expect = :block))
push!(cases, mkcase("block/block", block_problem, :block, 10; expect = :block))
push!(cases, mkcase("kronecker/auto", kronecker_problem, :auto, 0; expect = :kronecker))
push!(cases, mkcase("kronecker/kronecker", kronecker_problem, :kronecker, 0; expect = :kronecker))

# An Anderson-accelerated solve (the COSMOAccelerators extension) over a plain dense QP.
push!(
    cases,
    mkcase(
        "anderson/auto", anderson_problem, :auto, 10;
        accelerator = (n, m) -> PureOSQP.anderson(Float64, n + m)
    )
)

# `Random QP` again, but with the KKT-error adaptive-ρ schedule instead of the default fixed
# interval.
push!(
    cases,
    mkcase(
        "adaptive_rho/kkt_error/auto", suite_build["Random QP"], :auto, 10;
        extra_opts = (adaptive_rho = :kkt_error,)
    )
)

"""
    run_case(c) -> NamedTuple

Set the problem `c.build()` returns up under `c.linsys`, `c.scaling`, `c.extra_opts` and
`c.accelerator`, solve it, and read off the recorded fields plus the backend actually reached.
Asserts `c.expect` against the reached backend when it is given.
"""
function run_case(c)
    P, q, A, l, u = c.build()
    n, m = length(q), length(l)
    accel_kwargs = isnothing(c.accelerator) ? (;) : (accelerator = c.accelerator(n, m),)
    ws = PureOSQP.setup(
        P, q, A, l, u; OPTS..., c.extra_opts..., linsys = c.linsys, scaling = c.scaling,
        accel_kwargs...
    )
    PureOSQP.solve!(ws)
    PureOSQP.has_solution(ws.status) ||
        @warn "case did not reach a solved status" case = c.name status = PureOSQP.status_name(ws.status)
    backend = PureOSQP.backend_name(ws.linsys)
    isnothing(c.expect) || backend === c.expect ||
        error("case $(c.name): expected backend $(c.expect), reached $backend")
    return (
        backend_name = String(backend),
        iter = ws.iter,
        refactor_count = ws.refactor_count,
        status = PureOSQP.status_name(ws.status),
        obj_val = round(ws.obj_val; digits = 10),
        accel_declined = ws.accel_declined,
    )
end

function run_all(cases)
    results = Dict{String, Any}()
    for c in cases
        results[c.name] = run_case(c)
    end
    results["update/resolve/auto"] = run_update_case()
    return results
end

function write_snapshot(results)
    mkpath(dirname(RESULTS))
    open(RESULTS, "w") do io
        JSON.print(
            io, Dict(
                "julia_version" => string(VERSION),
                "blas_threads" => BLAS.get_num_threads(),
                "cases" => results,
            ), 2
        )
    end
    return println("wrote $RESULTS ($(length(results)) cases)")
end

"Compare `results` against the saved snapshot field by field; return whether every case matched."
function check_snapshot(results)
    saved_cases = JSON.parsefile(RESULTS)["cases"]
    mismatches = String[]
    for (name, r) in results
        if !haskey(saved_cases, name)
            push!(mismatches, "$name: not in saved snapshot")
            continue
        end
        saved = saved_cases[name]
        current = Dict(
            "backend_name" => r.backend_name, "iter" => r.iter,
            "refactor_count" => r.refactor_count, "status" => r.status, "obj_val" => r.obj_val,
            "accel_declined" => r.accel_declined,
        )
        for k in ("backend_name", "iter", "refactor_count", "status", "obj_val", "accel_declined")
            current[k] == saved[k] ||
                push!(mismatches, "$name.$k: saved $(saved[k]), now $(current[k])")
        end
    end
    for name in keys(saved_cases)
        haskey(results, name) || push!(mismatches, "$name: in saved snapshot, missing now")
    end
    if isempty(mismatches)
        println("snapshot check passed: $(length(results)) cases match $RESULTS")
        return true
    end
    println("snapshot check FAILED: $(length(mismatches)) mismatch(es)")
    foreach(m -> println("  ", m), mismatches)
    return false
end

results = run_all(cases)
if "--check" in ARGS
    check_snapshot(results) || exit(1)
else
    write_snapshot(results)
end
