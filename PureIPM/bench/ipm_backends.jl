# Every structured backend under `InteriorPoint()`, against the dense full KKT factorization on
# the same problem, plus the cost of that factorization's entry-by-entry fill and one `Float32`
# run of the dense spike generator.
#
# For each problem it records the backend's status, outer iterations and the referee
# (`kkt_residuals`, computed from the original data), the same three for `linsys = :kkt`, and
# the backend `linsys = :auto` reaches. A backend is `routed` when it fails the referee
# (`1e-5`) or takes more than twice the full KKT factorization's iterations on any of its
# problems; `:auto` under the interior-point method then serves those pairs with `:kkt`.
#
# The structured families are `test/selection_tests.jl`'s at `n = 100` (`structured_problems.jl`),
# each as given (`qp`), with `P` zeroed (`lp`), with every fourth row an equality at zero
# (`eq`), and both (`lp_eq`); `x = 0` is feasible for all of them. The sparse engines are built
# directly, since which one `:auto` reaches depends on whether LDLFactorizations is loaded.
# `SparseFormedInverse` is not listed: the interior-point ladder has no formed rung and no
# `linsys` names it.
#
#     julia --project=bench bench/ipm_backends.jl    # writes bench/results/ipm_backends.json
using PureOSQP, PureIPM, PureQPBase, LinearAlgebra, SparseArrays, Random, JSON, Chairmarks
using LDLFactorizations, BandedMatrices

BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "..", "..", "bench", "suite_problems.jl"))
include(joinpath(@__DIR__, "..", "..", "bench", "structured_problems.jl"))

const RESULTS = joinpath(@__DIR__, "results", "ipm_backends.json")
const REFEREE = 1.0e-5
const SExt = Base.get_extension(PureQPBase, :PureQPBaseSparseArraysExt)

"The referee of `test/helpers.jl`: primal, dual and gap residuals from the original data."
function kkt_residuals(P, q, A, l, u, x, y)
    Ax = A * x
    z = clamp.(Ax, l, u)
    r_prim = isempty(Ax) ? zero(eltype(x)) : maximum(abs, Ax .- z)
    r_dual = maximum(abs, P * x .+ q .+ A' * y)
    quad = dot(x, P * x)
    lin = dot(q, x)
    sup = zero(eltype(x))
    sign_viol = zero(eltype(x))
    ny = maximum(abs, y; init = zero(eltype(y)))
    for i in eachindex(y)
        yi = y[i]
        iszero(yi) && continue
        bound = yi > 0 ? u[i] : l[i]
        if isfinite(bound)
            sup += bound * yi
        else
            sign_viol = max(sign_viol, abs(yi))
        end
    end
    gap = abs(quad + lin + sup)
    r_gap = gap / max(one(gap), abs(quad), abs(lin), abs(sup))
    r_sign = ny > 0 ? sign_viol / ny : zero(sign_viol)
    return (r_prim, r_dual, max(r_gap, r_sign))
end

"A `banded_qp` of `test/helpers.jl`: constraint rows over contiguous runs of variables."
function banded_qp(n, m; band = 3, seed = 0)
    Random.seed!(n + m + band + seed)
    rows, cols, vals = Int[], Int[], Float64[]
    for i in 1:m, j in max(1, div(i * n, m) - band):min(n, div(i * n, m) + band)
        push!(rows, i)
        push!(cols, j)
        push!(vals, randn())
    end
    A = sparse(rows, cols, vals, m, n)
    S = spdiagm(-1 => randn(n - 1), 0 => randn(n), 1 => randn(n - 1))
    P = sparse(Symmetric(S'S)) + 3.0I
    b = A * randn(n)
    return (P, randn(n), A, b .- rand(m), b .+ rand(m))
end

zero_like(P::Diagonal) = Diagonal(zeros(size(P, 1)))
zero_like(P::SymTridiagonal) = SymTridiagonal(zeros(size(P, 1)), zeros(size(P, 1) - 1))
zero_like(P::Symmetric) = Symmetric(zeros(size(P)))
zero_like(P::PureOSQP.BlockDiagonal) = PureOSQP.BlockDiagonal([zeros(size(b)) for b in P.blocks])
zero_like(P::SparseMatrixCSC) = spzeros(size(P)...)
zero_like(P::Matrix) = zeros(size(P))

"Square blocks, so the LP over them is bounded."
function block_lp()
    Random.seed!(9311)
    K, nb = 5, 12
    A = PureOSQP.BlockDiagonal([randn(nb, nb) ./ sqrt(nb) + 2I for _ in 1:K])
    P = PureOSQP.BlockDiagonal([zeros(nb, nb) for _ in 1:K])
    return (P, randn(K * nb), A, -rand(K * nb), rand(K * nb))
end

"A dense QP with `m = 2n` two-sided rows around a feasible point."
function dense_qp(n; seed = 6)
    Random.seed!(seed)
    X = randn(n, n)
    A = randn(2n, n)
    b = A * randn(n)
    return (X'X / n + I, randn(n), A, b .- rand(2n), b .+ rand(2n))
end

"""
    workspace(kind, P, q, A, l, u) -> InteriorPointWorkspace

An interior-point workspace on the backend `kind`: `:auto`, a `linsys` value, or one of the
backends built directly (`:lowrank`, `:cholmod`, `:sparse_kkt`).
"""
function workspace(kind, P, q, A, l, u)
    kind in (:lowrank, :cholmod, :sparse_kkt) ||
        return setup(P, q, A, l, u, InteriorPoint(); linsys = kind)
    T = Float64
    options = default_options(InteriorPoint(), T)
    algorithm = InteriorPoint{T}(InteriorPoint(), options.linsys)
    m, n = size(A)
    prob = PureOSQP.Problem(T, P, q, A, l, u; scaling = options.scaling)
    wt = PureOSQP.SystemWeights(ones(m), ones(m), algorithm.reg_primal)
    if kind === :lowrank
        ls = PureOSQP.DiagonalLowRank(prob.q0, n, PureQPBase.coupling_rank(A))
    elseif kind === :cholmod
        gram = SExt.reduced_gram(T, P, A, n)
        R = SExt.refill!(gram, P, A, wt.w, prob.E, prob.D, prob.c, wt.sigma)
        F = cholesky(Symmetric(R, :U))
        L = sparse(F.L)
        ls = SExt.SparseCholmod{T, Vector{T}, typeof(F)}(
            gram, F, L, SparseMatrixCSC(transpose(L)), F.p, zeros(n)
        )
    else
        gram = SExt.kkt_gram(T, P, A, n, m)
        K = SExt.refill_kkt!(gram, P, A, wt.w_inv, prob.E, prob.D, prob.c, wt.sigma)
        F = ldlt(Symmetric(K, :U))
        LD = sparse(F.LD)
        ls = SExt.SparseKKT{T, Vector{T}, typeof(F)}(
            gram, F, LD, inv.(diag(LD)), F.p, zeros(n + m), zeros(n + m)
        )
    end
    # A backend built unfactored is factored by the solve's starting point.
    return PureIPM.ipm_workspace(ls, prob, wt, algorithm, options)
end

function measure(kind, data)
    ws = workspace(kind, data...)
    s = solve!(ws)
    r = has_solution(s.status) ? maximum(kkt_residuals(data..., s.x, s.y)) : NaN
    return (
        backend = String(PureOSQP.backend_name(ws.linsys)),
        status = PureOSQP.status_name(s.status), iter = s.iter, referee = r,
    )
end

function row(problem, variant, kind, data)
    b = measure(kind, data)
    k = measure(:kkt, data)
    auto = String(PureOSQP.backend_name(workspace(:auto, data...).linsys))
    ok = b.status == "solved" && b.referee < REFEREE && b.iter <= 2k.iter
    r = (;
        problem, variant, n = length(data[2]), m = length(data[4]),
        b.backend, b.status, b.iter, b.referee,
        kkt_status = k.status, kkt_iter = k.iter, kkt_referee = k.referee,
        auto_backend = auto, within_rule = ok,
    )
    println(r)
    flush(stdout)
    return r
end

variants(P, q, A, l, u) = begin
    le, ue = copy(l), copy(u)
    le[1:4:end] .= 0
    ue[1:4:end] .= 0
    [
        ("qp", (P, q, A, l, u)), ("lp", (zero_like(P), q, A, l, u)),
        ("eq", (P, q, A, le, ue)), ("lp_eq", (zero_like(P), q, A, le, ue)),
    ]
end

family_kind = Dict(
    "diagonal" => :auto, "tridiagonal_diag" => :auto, "banded_tridiag" => :auto,
    "tridiagonal_bidiag" => :auto, "banded_wide_inside" => :auto,
    "lowrank_rank3" => :lowrank, "lowrank_rank10" => :lowrank,
    "cholesky_symmetric_banded" => :dense, "cholesky_wide_outside" => :dense,
    "cholesky_rank11" => :dense,
)

rows = NamedTuple[]
for f in structured_families(100), (variant, data) in variants(f.P, f.q, f.A, f.l, f.u)
    push!(rows, row(f.name, variant, family_kind[f.name], data))
end
push!(rows, row("block", "qp", :auto, block_problem()))
push!(rows, row("block_square", "lp", :auto, block_lp()))
for (variant, data) in variants(dense_qp(60)...)[1:2]
    push!(rows, row("dense_qp(60)", variant, :dense, data))
end
banded = banded_qp(200, 300)
for (variant, data) in (("qp", banded), ("lp", (zero_like(banded[1]), banded[2:end]...)))
    for kind in (:auto, :cholmod, :sparse_kkt)
        push!(rows, row("banded_qp(200,300)", variant, kind, data))
    end
end
for (name, make) in CASES
    data = make()
    push!(rows, row(name, "suite", :auto, data))
    name in ("Portfolio", "Lasso", "SVM", "Huber") &&
        push!(rows, row(name, "suite", :sparse_kkt, data))
end

"""
    fill_cost(data) -> NamedTuple

Per outer iteration of `linsys = :kkt`, against one factorization and the `bunchkaufman!`
inside it; their difference is the entry-by-entry fill of `K`.
"""
function fill_cost(name, data)
    P, q, A, l, u = data
    ws = setup(P, q, A, l, u, InteriorPoint(); linsys = :kkt)
    solve!(ws)
    s = solve!(cold_start!(ws))
    ls, prob, wt = ws.linsys, ws.prob, ws.weights
    factor = @b PureOSQP.factorize!(ls, prob, wt) seconds = 1
    Ad = Diagonal(prob.E) * Matrix(A) * Diagonal(prob.D)
    K = [
        prob.c .* (Diagonal(prob.D) * Matrix(P) * Diagonal(prob.D)) + wt.sigma * I Ad'
        Ad -Diagonal(wt.w_inv)
    ]
    bk = @b copy(K) bunchkaufman!(Symmetric(_, :L); check = false) seconds = 1
    r = (
        problem = name, n = prob.n, m = prob.m, iter = s.iter,
        iteration_ms = 1.0e3 * s.solve_time / s.iter, factorize_ms = 1.0e3 * factor.time,
        bunchkaufman_ms = 1.0e3 * bk.time, fill_ms = 1.0e3 * (factor.time - bk.time),
    )
    println(r)
    flush(stdout)
    return r
end

fills = [
    fill_cost("dense_qp(200)", dense_qp(200)),
    fill_cost("banded_qp(200,300)", banded),
    fill_cost("Lasso", CASES[findfirst(c -> first(c) == "Lasso", CASES)][2]()),
]

"""
The dense generator of `bench/ipm_matrixfree_spike.jl`, two-sided or with the row mix of
`bench/ipm_rowtypes_spike.jl`, as `test/ipm_tests.jl` reproduces it.
"""
function spike_problem(n, κ, frac, seed; mixed = false)
    rng = Xoshiro(seed)
    m = n
    U = Matrix(qr(randn(rng, m, m)).Q)
    V = Matrix(qr(randn(rng, n, n)).Q)
    Q = Matrix(qr(randn(rng, n, n)).Q)
    A = U * Diagonal(exp10.(range(0, -log10(κ); length = n))) * V'
    P = Q * Diagonal(exp10.(range(0, -2; length = n))) * Q'
    P = (P + P') / 2
    xstar = randn(rng, n)
    a = A * xstar
    l = a .- (0.5 .+ rand(rng, m))
    u = a .+ (0.5 .+ rand(rng, m))
    ystar = zeros(m)
    if !mixed
        for i in randperm(rng, m)[1:round(Int, frac * m)]
            mag = 0.5 + rand(rng)
            if rand(rng, Bool)
                ystar[i] = mag
                u[i] = a[i]
            else
                ystar[i] = -mag
                l[i] = a[i]
            end
        end
        return P, -(P * xstar + A' * ystar), A, l, u
    end
    kind = fill(:two, m)
    p = randperm(rng, m)
    j = 0
    for (k, f) in ((:eq, 0.2), (:lo, 0.2), (:up, 0.2), (:free, 0.1)), _ in 1:round(Int, f * m)
        kind[p[j += 1]] = k
    end
    fill!(l, -Inf)
    fill!(u, Inf)
    for i in 1:m
        gap = 0.5 + rand(rng)
        mag = 0.5 + rand(rng)
        active = rand(rng) < frac
        k = kind[i]
        if k === :eq
            l[i] = u[i] = a[i]
            ystar[i] = rand(rng, Bool) ? mag : -mag
        elseif k === :lo
            active ? (l[i] = a[i]; ystar[i] = -mag) : (l[i] = a[i] - gap)
        elseif k === :up
            active ? (u[i] = a[i]; ystar[i] = mag) : (u[i] = a[i] + gap)
        elseif k === :two
            l[i] = a[i] - gap
            u[i] = a[i] + 0.5 + rand(rng)
            if active
                rand(rng, Bool) ? (u[i] = a[i]; ystar[i] = mag) : (l[i] = a[i]; ystar[i] = -mag)
            end
        end
    end
    return P, -(P * xstar + A' * ystar), A, l, u
end

"""
    float32_run(data) -> NamedTuple

`FullKKT` in `Float32` at `eps = 1e-4` and `δ = sqrt(eps(Float32))`, without equilibration,
built directly; the referee runs in `Float64` on the original data. The `Float64`
run at the same tolerance is alongside.
"""
function float32_run(data)
    T = Float32
    δ = sqrt(eps(T))
    tol = 1.0e-4
    options = Options{T}(;
        PureOSQP.algorithm_defaults(InteriorPoint(), T)...,
        eps_abs = tol, eps_rel = tol, eps_prim_inf = tol, eps_dual_inf = tol, scaling = 0,
    )
    algorithm = InteriorPoint{T}(InteriorPoint(reg_primal = δ, reg_dual = δ), options.linsys)
    P, q, A, l, u = data
    m, n = size(A)
    prob = PureOSQP.Problem(T, T.(P), T.(q), T.(A), T.(l), T.(u); scaling = 0)
    wt = PureOSQP.SystemWeights(ones(T, m), ones(T, m), δ)
    ws = PureIPM.ipm_workspace(PureOSQP.FullKKT(prob.q0, n, m), prob, wt, algorithm, options)
    s = solve!(ws)
    r = has_solution(s.status) ? maximum(kkt_residuals(data..., Float64.(s.x), Float64.(s.y))) : NaN
    d = PureOSQP.solve(data..., InteriorPoint(); linsys = :kkt, scaling = 0, eps_abs = tol, eps_rel = tol)
    return (
        status = PureOSQP.status_name(s.status), iter = s.iter, referee = r, reg_bumps = ws.reg_bumps,
        float64_iter = d.iter, float64_referee = maximum(kkt_residuals(data..., d.x, d.y)),
    )
end

float32 = NamedTuple[]
grid = [(κ, frac) for κ in (1.0, 1.0e3, 1.0e6) for frac in (0.1, 0.5, 0.9)]
for (index, (κ, frac)) in enumerate(grid), mixed in (false, true)
    r = (; kappa = κ, frac, mixed, float32_run(spike_problem(200, κ, frac, 700 + index; mixed))...)
    println(r)
    flush(stdout)
    push!(float32, r)
end

routed = sort(unique(r.backend for r in rows if !r.within_rule))
mkpath(dirname(RESULTS))
open(RESULTS, "w") do io
    JSON.json(
        io, Dict(
            "julia" => string(VERSION), "blas_threads" => BLAS.get_num_threads(),
            "referee" => REFEREE,
            "rule" => "routed to :kkt when a backend fails the referee or takes more than 2x the :kkt iterations on any problem",
            "routed" => routed, "backends" => rows, "fullkkt_fill" => fills, "float32" => float32,
        ); allownan = true
    )
end
println("routed: ", routed)
println("wrote $RESULTS")
