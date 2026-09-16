# Which backend is fastest for a given `(P, A)` pair, measured on both algorithms.
#
# The two algorithms spend a factorization very differently. ADMM factors once and solves
# against that factorization hundreds of times, so it can afford an expensive factorization
# whose solves are cheap. The interior-point method refactorizes the Newton system every
# outer iteration and solves against it two or three times, so it cannot. The terminal each
# falls back to differs too: ADMM's is `ReducedCholesky`, which inverts the `n×n` reduced
# matrix, while the interior-point method's is `FullKKT`, which factors the dense
# `(n+m)×(n+m)` quasi-definite matrix.
#
# For every problem this records one solve on each backend the pair admits, under
# `InteriorPoint()` and under `OperatorSplitting()`, with the fill each factor carries. The
# fill is reported three ways — against `(n+m)^2`, against `n^2`, and as the reduced factor's
# own fraction of `n^2` — because which denominator a backend should be judged by is exactly
# the open question: `(n+m)^2` is the size of the dense system the sparse KKT factor replaces
# under the interior-point method, and `n^2` the size of the one it replaces under ADMM.
#
# Under `InteriorPoint()` the two sparse forms are built directly, so both are timed whatever
# the rungs would have chosen. Under `OperatorSplitting()` they cannot be: a workspace is
# assembled inside `setup` and there is no entry point that takes a prebuilt backend, so ADMM
# is measured through the `linsys` names a caller can actually pass, and `linsys = :sparse`
# reports which of the two sparse forms its ladder reached. The fills of both forms are
# recorded regardless — they depend on the patterns of `P` and `A` alone, not on the weights,
# so they are the same numbers under either algorithm.
#
# The problems are the OSQP suite's seven classes (`suite_problems.jl`), the structured pairs
# of `test/selection_tests.jl` (`structured_problems.jl`), and three synthetic sparse families
# swept over `n`, `m/n` and the density or bandwidth of `A`, chosen so the KKT factor's fill
# crosses 0.001 to 0.35 of `(n+m)^2`.
#
# A backend whose measured factorization alone would put a solve an order of magnitude past
# the fastest one keeps that factorization time and is not solved: the skip is a measurement,
# not an assumption, and the recorded `factorize_ms` is the evidence.
#
#     julia --project=bench bench/ipm_selection.jl   # writes bench/results/ipm_selection.json
using PureOSQP, PureQPBase, LinearAlgebra, SparseArrays, Random, JSON, Chairmarks, Statistics
using LDLFactorizations, BandedMatrices

BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "suite_problems.jl"))
include(joinpath(@__DIR__, "structured_problems.jl"))

const RESULTS = joinpath(@__DIR__, "results", "ipm_selection.json")
const SExt = Base.get_extension(PureQPBase, :PureQPBaseSparseArraysExt)
const EPS = 1.0e-8
# A solve predicted past this from one measured factorization is recorded and skipped.
const SKIP_MS = 2500.0
const BUDGET = 0.15

"The problem and unit weights an interior-point `setup` starts from."
function ipm_parts(P, q, A, l, u)
    T = Float64
    options = Options{T}(;
        PureOSQP.algorithm_defaults(InteriorPoint(), T)..., eps_abs = EPS, eps_rel = EPS,
    )
    algorithm = InteriorPoint{T}(InteriorPoint(), options.linsys)
    m = size(A, 1)
    prob = PureOSQP.Problem(T, P, q, A, l, u; scaling = options.scaling)
    wt = PureOSQP.SystemWeights(ones(m), ones(m), algorithm.reg_primal)
    return (prob, wt, algorithm, options)
end

"""
    ipm_ws(kind, data) -> InteriorPointWorkspace or nothing

A workspace on the named backend, built directly so that no rung's fill gate intervenes —
which is the point: the gates are what this benchmark is measuring against. `nothing` means
the pair does not admit the backend at all.
"""
function ipm_ws(kind, data)
    P, q, A, l, u = data
    prob, wt, algorithm, options = ipm_parts(data...)
    n, m = prob.n, prob.m
    ls = if kind === :dense
        PureOSQP.FullKKT(prob.q0, n, m)
    elseif kind === :sparse_kkt
        (P isa SparseMatrixCSC && A isa SparseMatrixCSC) || return nothing
        SExt.factored_kkt_backend(P, A, prob, wt)
    elseif kind === :sparse_reduced
        (P isa SparseMatrixCSC && A isa SparseMatrixCSC) || return nothing
        SExt.cholmod_backend(P, A, prob, wt)
    elseif kind === :auto
        first(PureOSQP.choose_backend(P, A, prob, wt, PureOSQP.IPMSelection()))
    else
        error("unknown backend kind $kind")
    end
    isnothing(ls) && return nothing
    return PureOSQP.ipm_workspace(ls, prob, wt, algorithm, options)
end

"""
    run_of(build, kind, algorithm, floor_ms, floor_iter) -> NamedTuple or nothing

One solve on the workspace `build()` returns: the backend's name and factor fill, the status
and iterations, and the median of `solve!` over a short budget.

`floor_ms` is the fastest solve seen for this problem so far. A backend whose one measured
`factorize!` already puts its solve ten times past that is left unsolved, with `factorize_ms`
recorded and `solve_ms` missing — which is the measurement that justifies skipping it.
"""
function run_of(build, kind, algorithm, floor_ms, floor_iter)
    ws = build()
    isnothing(ws) && return nothing
    ls, prob, wt = ws.linsys, ws.prob, ws.weights
    name = String(PureOSQP.backend_name(ls))
    fill = PureOSQP.backend_info(ls).factor_nnz
    factor = @b PureOSQP.factorize!($ls, $prob, $wt) seconds = BUDGET
    factor_ms = 1.0e3 * factor.time
    predicted = factor_ms * max(floor_iter, 1)
    if predicted > 10 * floor_ms && predicted > SKIP_MS
        return (
            algorithm, kind = String(kind), backend = name, fill, factorize_ms = factor_ms,
            solve_ms = NaN, status = "not solved", iter = -1, skipped = true,
        )
    end
    s = solve!(build())
    t = @b build() solve!(_) seconds = BUDGET
    return (
        algorithm, kind = String(kind), backend = name, fill, factorize_ms = factor_ms,
        solve_ms = 1.0e3 * t.time, status = String(PureOSQP.status_name(s.status)),
        iter = s.iter, skipped = false,
    )
end

"An ADMM workspace on a `linsys` name, or `nothing` where that name refuses the pair."
function admm_ws(name, data)
    P, q, A, l, u = data
    return try
        setup(P, q, A, l, u, OperatorSplitting(); linsys = name, eps_abs = EPS, eps_rel = EPS)
    catch err
        err isa ArgumentError || rethrow()
        nothing
    end
end

function report(x)
    println(
        rpad(x.name, 24), " n=", rpad(x.n, 6), " m=", rpad(x.m, 6),
        " kktfill=", rpad(round(x.kkt_fill; digits = 5), 9),
        " redfill=", rpad(round(x.reduced_fill; digits = 5), 9),
        " ipm_best=", rpad(x.ipm_best, 20), " admm_best=", x.admm_best,
    )
    for run in x.runs
        tail = if run.skipped
            "  factorize $(round(run.factorize_ms; digits = 2)) ms, solve skipped"
        else
            "  $(round(run.solve_ms; digits = 3)) ms  $(run.status)  $(run.iter) iter"
        end
        println("    ", rpad(run.algorithm, 5), rpad(run.kind, 16), rpad(run.backend, 20), tail)
    end
    flush(stdout)
    return nothing
end

"""
    fastest(runs, algorithm) -> (backend, solve_ms)

The fastest run among `runs`, by backend name, or `("none", NaN)` when every candidate was
skipped.

A run that stopped at `max_iter` counts. Which backend a workspace holds does not change the
iterates — every backend solves the same linear system to the same tolerance — so the
candidates take the same path and the same number of steps, and their times compare as
per-iteration cost whether or not that path reached the tolerance. ADMM does not reach `1e-8`
within `max_iter` on the random-sparsity families here, and dropping those rows would throw
away the whole ADMM sweep rather than the part of it that is uninformative.
"""
function fastest(runs, algorithm)
    ran = [r for r in runs if r.algorithm == algorithm && !r.skipped]
    isempty(ran) && return ("none", NaN)
    best = ran[argmin(r.solve_ms for r in ran)]
    return (best.backend, best.solve_ms)
end

"""
    case(family, name, data) -> NamedTuple

Every backend the pair admits under either algorithm, and the fill each factor carries.
"""
function case(family, name, data)
    P, q, A, l, u = data
    m, n = size(A)
    dens = A isa SparseMatrixCSC ? nnz(A) / (m * n) : count(!iszero, A) / (m * n)
    runs = NamedTuple[]
    floor_ms, floor_iter = Inf, 0
    # Cheapest first, so the skip rule has a floor to compare against before the dense
    # terminal is reached. `:auto` comes last and is measured like the rest: for a structured
    # pair it is the only candidate that reaches the structured backend at all, and for a
    # sparse one it repeats whichever candidate the rungs chose.
    for kind in (:sparse_kkt, :sparse_reduced, :dense, :auto)
        r = run_of(() -> ipm_ws(kind, data), kind, "ipm", floor_ms, floor_iter)
        isnothing(r) && continue
        push!(runs, r)
        if !r.skipped && r.solve_ms < floor_ms
            floor_ms, floor_iter = r.solve_ms, r.iter
        end
    end
    floor_ms, floor_iter = Inf, 0
    for kind in (:sparse, :dense, :kkt, :auto)
        r = run_of(() -> admm_ws(kind, data), kind, "admm", floor_ms, floor_iter)
        isnothing(r) && continue
        push!(runs, r)
        if !r.skipped && r.solve_ms < floor_ms
            floor_ms, floor_iter = r.solve_ms, r.iter
        end
    end
    fill_of(k) = (i = findfirst(r -> r.algorithm == "ipm" && r.kind == k, runs); isnothing(i) ? -1 : runs[i].fill)
    kkt_nnz, red_nnz = fill_of("sparse_kkt"), fill_of("sparse_reduced")
    ipm_best, ipm_best_ms = fastest(runs, "ipm")
    admm_best, admm_best_ms = fastest(runs, "admm")
    auto_of(a) = (i = findfirst(r -> r.algorithm == a && r.kind == "auto", runs); isnothing(i) ? ("none", NaN) : (runs[i].backend, runs[i].solve_ms))
    ipm_auto, ipm_auto_ms = auto_of("ipm")
    admm_auto, admm_auto_ms = auto_of("admm")
    r = (
        family, name, n, m, density = dens, kkt_nnz, reduced_nnz = red_nnz,
        kkt_fill = kkt_nnz < 0 ? NaN : kkt_nnz / (n + m)^2,
        kkt_fill_n2 = kkt_nnz < 0 ? NaN : kkt_nnz / n^2,
        reduced_fill = red_nnz < 0 ? NaN : red_nnz / n^2,
        ipm_best, ipm_best_ms, ipm_auto, ipm_auto_ms,
        admm_best, admm_best_ms, admm_auto, admm_auto_ms, runs,
    )
    report(r)
    return r
end

"A sparse QP with `m` two-sided rows at density `d`; `pdens = 0` leaves `P` diagonal."
function sprand_qp(n, m, d; seed = 11, pdens = 0.15)
    rng = MersenneTwister(seed)
    Pr = sprandn(rng, n, n, pdens)
    P = sparse(Symmetric(Pr * Pr')) + 1.0e-2 * I
    A = sprandn(rng, m, n, d)
    b = A * randn(rng, n)
    return (P, randn(rng, n), A, b .- rand(rng, m), b .+ rand(rng, m))
end

"Constraint rows over contiguous runs of `2band + 1` variables, so the factor stays sparse."
function banded_qp(n, m; band = 3, seed = 0)
    rng = MersenneTwister(n + m + band + seed)
    rows, cols, vals = Int[], Int[], Float64[]
    for i in 1:m, j in max(1, div(i * n, m) - band):min(n, div(i * n, m) + band)
        push!(rows, i)
        push!(cols, j)
        push!(vals, randn(rng))
    end
    A = sparse(rows, cols, vals, m, n)
    S = spdiagm(-1 => randn(rng, n - 1), 0 => randn(rng, n), 1 => randn(rng, n - 1))
    P = sparse(Symmetric(S'S)) + 3.0I
    b = A * randn(rng, n)
    return (P, randn(rng, n), A, b .- rand(rng, m), b .+ rand(rng, m))
end

rows = NamedTuple[]
t0 = time()

for (name, make) in CASES
    push!(rows, case("suite", name, make()))
end

for f in structured_families(200)
    push!(rows, case("structured", f.name, (f.P, f.q, f.A, f.l, f.u)))
end

for n in (100, 300), ratio in (0.5, 1.0, 4.0), d in (0.005, 0.05, 0.4)
    m = round(Int, ratio * n)
    push!(rows, case("sprand", "sprand($n,$m,d=$d)", sprand_qp(n, m, d)))
end

for n in (200, 600), ratio in (0.5, 1.0, 2.0), band in (1, 16, 64)
    m = round(Int, ratio * n)
    push!(rows, case("banded", "banded($n,$m,b=$band)", banded_qp(n, m; band)))
end

for n in (200, 600), ratio in (0.5, 1.0, 4.0), d in (0.005, 0.02, 0.08)
    m = round(Int, ratio * n)
    push!(rows, case("diagP", "diagP($n,$m,d=$d)", sprand_qp(n, m, d; pdens = 0.0)))
end

elapsed = time() - t0

"""
    crossover(rows, field, algorithm, sparse_kinds, dense_kind) -> NamedTuple

The bracket the data leaves for a threshold on `field`: the largest value at which the
fastest backend among `sparse_kinds` still beats `dense_kind`, and the smallest at which it
does not. A threshold inside the bracket routes every measured pair to whichever side was
faster.

A field that separates the two leaves `won_up_to < lost_from`; one that does not leaves them
crossed, and is not a threshold this benchmark supports. `sparse_kinds` is the rung's own
choice: a gate that decides between the sparse KKT factorization and falling through is read
off the `("sparse_kkt",)` row.
"""
function crossover(rows, field, algorithm, sparse_kinds, dense_kind)
    won, lost = Float64[], Float64[]
    for r in rows
        v = getfield(r, field)
        isnan(v) && continue
        i = findfirst(x -> x.algorithm == algorithm && x.kind == dense_kind, r.runs)
        (isnothing(i) || r.runs[i].skipped) && continue
        dense_ms = r.runs[i].solve_ms
        sparse_ms = minimum(
            (
                x.solve_ms for x in r.runs
                    if x.algorithm == algorithm && x.kind in sparse_kinds && !x.skipped
            ); init = Inf
        )
        isinf(sparse_ms) && continue
        push!(sparse_ms < dense_ms ? won : lost, v)
    end
    return (
        field = String(field), algorithm, against = join(sparse_kinds, "+"),
        won_up_to = isempty(won) ? NaN : maximum(won),
        lost_from = isempty(lost) ? NaN : minimum(lost),
        n_won = length(won), n_lost = length(lost),
    )
end

sparse_rows = [r for r in rows if !isnan(r.kkt_fill)]
crossings = vcat(
    [crossover(sparse_rows, f, "ipm", ("sparse_kkt",), "dense") for f in (:kkt_fill, :kkt_fill_n2, :density)],
    [crossover(sparse_rows, f, "ipm", ("sparse_kkt", "sparse_reduced"), "dense") for f in (:kkt_fill, :reduced_fill)],
    [crossover(sparse_rows, f, "admm", ("sparse",), "dense") for f in (:kkt_fill, :kkt_fill_n2, :reduced_fill, :density)],
)

println("\nwhere a sparse factorization stops beating the algorithm's dense terminal")
for c in crossings
    println(
        rpad(c.algorithm, 6), rpad(c.field, 14), rpad(c.against, 28),
        " wins up to ", rpad(round(c.won_up_to; digits = 5), 9),
        ", loses from ", rpad(round(c.lost_from; digits = 5), 9),
        "  (", c.n_won, "/", c.n_lost, ")",
        c.won_up_to < c.lost_from ? "  separates" : "  does not separate",
    )
end

"""
    predictors(rows, algorithm, kind_a, kind_b) -> Vector{NamedTuple}

How well each quantity a selection rule could be written on predicts which of two backends
is faster, as the correlation between `log(t_a / t_b)` and the log of that quantity over
every pair that ran both.

A rule written on a quantity that correlates weakly is a rule the measurements do not
support, whatever threshold it is given; one that correlates strongly names the comparison
the rule should be making. Logs on both sides because every quantity here spans orders of
magnitude and the costs are products of them.
"""
function predictors(rows, algorithm, kind_a, kind_b)
    ms(r, k) = (
        i = findfirst(x -> x.algorithm == algorithm && x.kind == k, r.runs);
        isnothing(i) || r.runs[i].skipped ? NaN : r.runs[i].solve_ms
    )
    usable = [r for r in rows if isfinite(ms(r, kind_a)) && isfinite(ms(r, kind_b))]
    y = [log(ms(r, kind_a) / ms(r, kind_b)) for r in usable]
    quantities = (
        "nnzL_kkt/nnzL_reduced" => (r -> r.kkt_nnz / r.reduced_nnz),
        "kkt_fill = nnzL/(n+m)^2" => (r -> r.kkt_fill),
        "kkt_fill against n^2" => (r -> r.kkt_fill_n2),
        "reduced_fill = nnzL/n^2" => (r -> r.reduced_fill),
        "m/n" => (r -> r.m / r.n),
        "density of A" => (r -> r.density),
        "n" => (r -> Float64(r.n)),
    )
    return [
        (
            algorithm, comparison = "$kind_a vs $kind_b", quantity = name,
            n_pairs = length(usable),
            correlation = length(usable) < 3 ? NaN : cor([log(f(r)) for r in usable], y),
        ) for (name, f) in quantities
    ]
end

predicted = vcat(
    predictors(sparse_rows, "ipm", "sparse_kkt", "dense"),
    predictors(sparse_rows, "ipm", "sparse_kkt", "sparse_reduced"),
    predictors(sparse_rows, "admm", "sparse", "dense"),
)
println("\nhow well each quantity predicts which backend is faster (log correlation)")
for p in predicted
    println(
        rpad(p.algorithm, 6), rpad(p.comparison, 30), rpad(p.quantity, 26),
        rpad(round(p.correlation; digits = 3), 8), "(", p.n_pairs, " pairs)",
    )
end

for alg in ("ipm", "admm")
    best = alg == "ipm" ? (r -> (r.ipm_auto, r.ipm_auto_ms, r.ipm_best, r.ipm_best_ms)) :
        (r -> (r.admm_auto, r.admm_auto_ms, r.admm_best, r.admm_best_ms))
    off = [r for r in rows if first(best(r)) != best(r)[3]]
    println("\n:auto differs from the fastest measured backend on ", length(off), " of ", length(rows), " problems (", alg, ")")
    for r in off
        a, ams, b, bms = best(r)
        # `:auto` reaching a backend whose solve was skipped has no time to show, and the
        # measured `factorize_ms` in the JSON is what says how far behind it is.
        ratio = isnan(ams) ? "solve skipped" : "$(round(ams / bms; digits = 2))x"
        println(
            "  ", rpad(r.name, 24), rpad(a, 20), rpad(round(ams; digits = 2), 10),
            " vs ", rpad(b, 20), rpad(round(bms; digits = 2), 10), ratio,
        )
    end
end
println("\nelapsed ", round(elapsed; digits = 1), " s")

mkpath(dirname(RESULTS))
open(RESULTS, "w") do io
    JSON.json(
        io, Dict(
            "julia" => string(VERSION), "blas_threads" => BLAS.get_num_threads(),
            "eps" => EPS, "budget_seconds" => BUDGET, "skip_ms" => SKIP_MS,
            "elapsed_seconds" => elapsed, "crossings" => crossings,
            "predictors" => predicted, "cases" => rows,
        ); allownan = true
    )
end
println("wrote $RESULTS")
