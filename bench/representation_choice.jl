# Which representation to hand the solver, measured rather than argued.
#
# Storing `P` and `A` densely, sparsely, in a structured type, or not at all are four answers
# to one question, and which is right is a property of the problem. This file measures three
# cases that separate them, each chosen so the answer is not the one a slogan would give.
#
#   nothing gained    an operator whose product costs what the dense product costs. Being
#                     matrix-free buys a cheaper setup and pays for it every iteration, so
#                     the dense form wins and keeps winning as `n` grows.
#   asymptotic win    an operator applied in `O(n)` whose dense form is `O(n²)`, at a fill
#                     too high for a sparse format to help. The operator wins at every size
#                     measured, by more as `n` leaves cache.
#   direct, not CG    an operator carrying its own direct backend, on a problem too badly
#                     conditioned for conjugate gradients. `κ(A₁ ⊗ A₂) = κ(A₁)·κ(A₂)`, so a
#                     Kronecker operator factors two well-conditioned small matrices where the
#                     dense route factors one badly-conditioned large one.
#
# The third is the case that shows "unmaterialized" and "solved iteratively" are different
# choices. A matrix-free backend is one way to serve an operator; a structured direct backend
# is another, and it is the one that survives ill-conditioning.
#
# Every row asserts its status, because a backend that ran out of iterations is not a faster
# answer to the same question.
#
# Run:  julia --project=bench bench/representation_choice.jl
using PureOSQP, LinearAlgebra, LinearMaps, Krylov
using Printf, JSON, Random, Statistics

include(joinpath(@__DIR__, "lazy_operator.jl"))

BLAS.set_num_threads(1)

const OPTS = (scaling = 0, eps_abs = 1.0e-6, eps_rel = 1.0e-6, max_iter = 20_000)
const RESULTS = joinpath(@__DIR__, "results", "representation_choice.json")

best(f, reps) = minimum(
    begin
            f()
        end for _ in 1:reps
)

"""
    solved!(r, what)

Return `r`, having refused to report a run that did not converge. A comparison between a
converged run and one that stopped at `max_iter` measures the iteration cap.
"""
function solved!(r, what)
    r.status === PureOSQP.SOLVED || error("$what did not converge: $(r.status) at $(r.iter)")
    return r
end

rows = NamedTuple[]

# ---------------------------------------------------------------- nothing gained
#
# `P = Diagonal(d) + α v vᵀ` as an operator, against the `n×n` matrix it names, with a dense
# `A` in both arms. The operator's product is `O(n)`, but `A`'s is `O(n²)` either way and the
# matrix-free backend applies it several times per iteration, so nothing is saved where the
# cost actually is.
println("\nAn operator with no asymptotic advantage over its dense form")
@printf("%6s | %-16s %9s %9s %9s\n", "n", "iters op/dense", "operator", "dense", "speedup")
println("-"^62)
for n in (200, 500, 1000)
    rng = MersenneTwister(4231 + 7n)
    Pop = LazyPSD(rand(rng, n) .+ 2.0, randn(rng, n) ./ sqrt(n), 0.5)
    Pm = materialize(Pop)
    A = randn(rng, n, n) ./ sqrt(n)
    b = A * randn(rng, n)
    q, l, u = randn(rng, n), b .- rand(rng, n), b .+ rand(rng, n)

    ro = solved!(PureOSQP.solve(Pop, q, A, l, u; OPTS...), "operator n=$n")
    rd = solved!(PureOSQP.solve(Pm, q, A, l, u; OPTS...), "dense n=$n")
    to = best(() -> @elapsed(PureOSQP.solve(Pop, q, A, l, u; OPTS...)), 3)
    td = best(() -> @elapsed(PureOSQP.solve(Pm, q, A, l, u; OPTS...)), 3)
    push!(
        rows, (;
            case = "no_advantage", n, operator_iter = ro.iter, dense_iter = rd.iter,
            operator_seconds = to, dense_seconds = td, speedup = td / to,
        )
    )
    @printf(
        "%6d | %-16s %7.1f ms %7.1f ms %8.2fx\n",
        n, string(ro.iter, "/", rd.iter), 1.0e3to, 1.0e3td, td / to
    )
    flush(stdout)
end

# ---------------------------------------------------------------- asymptotic win
#
# `A = I + α·(windowed average over ±b)`, with `b = n/20`, so about a tenth of the entries are
# nonzero -- too dense for a sparse format to be the obvious answer, and structured enough that
# a running sum applies it in `O(n)` where the dense product is `O(n²)`. Diagonally dominant,
# so it is well conditioned and the inner solve is not what is being measured.
"""
    window_problem(n; frac, alpha) -> (P, q, Aop, Adense, l, u, b)

The windowed-average operator and the matrix it names.
"""
function window_problem(n; frac = 0.05, alpha = 0.4)
    b = max(1, round(Int, frac * n))
    rng = MersenneTwister(3 + n)
    function forward!(y, x)
        c = cumsum(vcat(zero(eltype(x)), x))
        for i in eachindex(y)
            lo, hi = max(1, i - b), min(n, i + b)
            y[i] = x[i] + alpha * (c[hi + 1] - c[lo]) / (hi - lo + 1)
        end
        return y
    end
    Ad = zeros(n, n)
    for i in 1:n
        lo, hi = max(1, i - b), min(n, i + b)
        Ad[i, i] += 1.0
        for j in lo:hi
            Ad[i, j] += alpha / (hi - lo + 1)
        end
    end
    Aop = LinearMap{Float64}(
        (y, x) -> forward!(y, x), (x, y) -> mul!(x, transpose(Ad), y), n, n
    )
    P = Diagonal(fill(2.0, n))
    bb = Ad * randn(rng, n)
    return P, randn(rng, n), Aop, Ad, bb .- rand(rng, n), bb .+ rand(rng, n), b
end

println("\nAn operator applied in O(n) whose dense form is O(n²), at ~10% fill")
@printf(
    "%6s %6s %7s | %-16s %9s %9s %9s %9s\n",
    "n", "band", "fill", "iters op/dense", "operator", "dense", "speedup", "dense P"
)
println("-"^92)
for n in (500, 1000, 2000, 4000)
    P, q, Aop, Ad, l, u, b = window_problem(n)
    ro = solved!(PureOSQP.solve(P, q, Aop, l, u; OPTS...), "window operator n=$n")
    rd = solved!(PureOSQP.solve(P, q, Ad, l, u; OPTS...), "window dense n=$n")
    reps = n >= 4000 ? 2 : 3
    to = best(() -> @elapsed(PureOSQP.solve(P, q, Aop, l, u; OPTS...)), reps)
    td = best(() -> @elapsed(PureOSQP.solve(P, q, Ad, l, u; OPTS...)), reps)
    fill_pct = 100 * count(!iszero, Ad) / n^2
    mib = n^2 * 8 / 2^20
    push!(
        rows, (;
            case = "asymptotic_win", n, bandwidth = b, fill_percent = fill_pct,
            operator_iter = ro.iter, dense_iter = rd.iter,
            operator_seconds = to, dense_seconds = td, speedup = td / to, dense_mib = mib,
        )
    )
    @printf(
        "%6d %6d %6.1f%% | %-16s %7.1f ms %7.1f ms %8.2fx %7.1f MiB\n",
        n, b, fill_pct, string(ro.iter, "/", rd.iter), 1.0e3to, 1.0e3td, td / to, mib
    )
    flush(stdout)
end

# ---------------------------------------------------------------- direct, not CG
#
# A Kronecker operator at `κ(A) = 1e12`, where the matrix-free backend cannot converge at all.
# Its backend is direct: it eigendecomposes the two factors, each of which carries the square
# root of the conditioning, and never forms or factors the product.
println("\nAn ill-conditioned operator with a direct backend, where CG cannot go")
@printf(
    "%6s %10s %10s | %-14s %9s %9s %9s\n",
    "n", "κ(factor)", "κ(A)", "iters kr/dn", "kronecker", "dense", "speedup"
)
println("-"^80)
for k in (20, 40)
    rng = MersenneTwister(21)
    mk(sz) = (
        U = qr(randn(rng, sz, sz)).Q; V = qr(randn(rng, sz, sz)).Q;
        Matrix(U * Diagonal(exp10.(range(0, -6.0; length = sz))) * V')
    )
    A1, A2 = mk(k), mk(k)
    Aop = PureOSQP.KroneckerOperator(A1, A2)
    Ad = kron(A1, A2)
    n = k * k
    P = Diagonal(fill(2.0, n))
    bb = Ad * randn(rng, n)
    q, l, u = randn(rng, n), bb .- rand(rng, n), bb .+ rand(rng, n)

    ws = PureOSQP.setup(P, q, Aop, l, u; OPTS...)
    backend = String(PureOSQP.backend_name(ws.linsys))
    backend == "kronecker" || error("n=$n reached $backend, not the Kronecker backend")
    rk = solved!(PureOSQP.solve(P, q, Aop, l, u; OPTS...), "kronecker n=$n")
    rd = solved!(PureOSQP.solve(P, q, Ad, l, u; OPTS...), "dense n=$n")
    isapprox(rk.obj_val, rd.obj_val; rtol = 1.0e-6) ||
        error("n=$n: the two routes disagree on the objective")
    tk = best(() -> @elapsed(PureOSQP.solve(P, q, Aop, l, u; OPTS...)), 2)
    td = best(() -> @elapsed(PureOSQP.solve(P, q, Ad, l, u; OPTS...)), 2)
    # What conjugate gradients does with the same problem, for contrast. It is not expected to
    # converge, so it is reported rather than asserted.
    rcg = PureOSQP.solve(P, q, Ad, l, u; OPTS..., linsys = :indirect)
    push!(
        rows, (;
            case = "direct_not_cg", n, cond_factor = cond(A1), cond_A = cond(Ad),
            kronecker_iter = rk.iter, dense_iter = rd.iter,
            kronecker_seconds = tk, dense_seconds = td, speedup = td / tk,
            cg_status = string(rcg.status), cg_iter = rcg.iter,
            cg_obj = rcg.obj_val, direct_obj = rd.obj_val,
        )
    )
    @printf(
        "%6d %10.1e %10.1e | %-14s %7.1f ms %7.1f ms %8.1fx\n",
        n, cond(A1), cond(Ad), string(rk.iter, "/", rd.iter), 1.0e3tk, 1.0e3td, td / tk
    )
    @printf(
        "       conjugate gradients on the same problem: %s at %d iterations, objective %.4g against %.4g\n",
        rcg.status, rcg.iter, rcg.obj_val, rd.obj_val
    )
    flush(stdout)
end

open(RESULTS, "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "blas_threads" => BLAS.get_num_threads(),
            "eps" => OPTS.eps_abs, "max_iter" => OPTS.max_iter,
            "cases" => rows,
        ), 2
    )
end
println("\nwrote $RESULTS")
