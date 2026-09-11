# Anderson acceleration of the ADMM fixed point, measured against the plain iteration.
#
# ADMM is a fixed-point map on `w = [x; ρ⁻¹ ⊙ y + z]`. Anderson extrapolates from a window
# of past residuals `f = w - g(w)` instead of taking `w = g(w)` directly, which costs a
# small least-squares solve per iteration and can cut the number of iterations.
#
# An extrapolated point is not guaranteed to be better, so it is kept only when its residual
# is within `tau` of the plain step's; otherwise the plain step stands. Without that guard
# the iteration can diverge.
#
# Run:  julia --project=bench bench/anderson_prototype.jl
using PureOSQP, COSMOAccelerators, LinearAlgebra, SparseArrays, Random, Printf, JSON
using Chairmarks, Statistics
const CA = COSMOAccelerators

med(x) = median(s.time for s in x.samples)

BLAS.set_num_threads(1)

const RESULTS = joinpath(@__DIR__, "results", "anderson_prototype.json")
const TOL = 1.0e-8

"Pack the ADMM iterates into the fixed-point vector the accelerator works on."
function pack!(w, ws)
    n = ws.n
    @views w[1:n] .= ws.x
    @views w[(n + 1):end] .= ws.rho_inv_vec .* ws.y .+ ws.z
    return w
end

"""
    unpack!(ws, w)

Split an accelerated `w` back into iterates.

`w`'s second block is `ρ⁻¹ ⊙ y + z`, and the projection separates it: `z` is the part inside
the bounds, `y` is what is left over scaled back by `ρ`. Carrying the previous `y` across
instead would leave the two inconsistent, which the iteration does not recover from.
"""
function unpack!(ws, w)
    n = ws.n
    @views ws.x .= w[1:n]
    v = @view w[(n + 1):end]
    @. ws.z = clamp(v, ws.l, ws.u)
    @. ws.y = ws.rho_vec * (v - ws.z)
    return ws
end

"""
    run_loop(P, q, A, l, u; accelerate, mem, tau, max_iter) -> (status, iters)

Drive the solver's own ADMM step, optionally extrapolating between steps.
"""
function run_loop(
        P, q, A, l, u; accelerate::Bool, safeguard::Bool = true,
        mem::Int = 10, tau = 2.0, max_iter = 20_000
    )
    ws = PureOSQP.setup(P, q, A, l, u; eps_abs = TOL, eps_rel = TOL, max_iter)
    n, m = ws.n, ws.m
    aa = CA.AndersonAccelerator{Float64}(n + m; mem)
    w, w_prev = zeros(n + m), zeros(n + m)
    status = PureOSQP.UNSOLVED
    iters = max_iter
    declined = 0
    for iter in 1:max_iter
        # One step per iteration. `admm_step!` swaps the previous-iterate buffers and
        # advances `y` in place, so it cannot be run speculatively and undone.
        guarding = false
        nrm_plain = zero(Float64)
        if accelerate
            pack!(w, ws)
            if iter > 1
                CA.update!(aa, w, w_prev, iter)
                CA.accelerate!(w, w_prev, aa, iter)
                if CA.was_successful(aa)
                    # `aa.f` holds `x - g`, the residual the plain step would have had, so
                    # the bound the accelerated point must meet costs nothing to compute.
                    guarding = safeguard
                    nrm_plain = guarding ? norm(aa.f, 2) : zero(Float64)
                    unpack!(ws, w)
                    pack!(w, ws)
                end
            end
            copyto!(w_prev, w)
        end
        PureOSQP.admm_step!(ws)
        if guarding
            pack!(w, ws)
            # ADMM alone never lets this residual grow; extrapolation can. A candidate
            # whose residual exceeds `tau` times the plain step's is discarded, and the
            # step is retaken from the last point the accelerator did not touch.
            if norm(w_prev .- w, 2) > tau * nrm_plain
                declined += 1
                copyto!(w, aa.g_last)
                unpack!(ws, w)
                copyto!(w_prev, w)
                PureOSQP.admm_step!(ws)
            end
        end
        PureOSQP.update_residuals!(ws)
        st = PureOSQP.check_termination(ws)
        if st !== PureOSQP.UNSOLVED
            status = st
            iters = iter
            break
        end
    end
    return (status, iters, declined)
end

cases = Any[]
Random.seed!(1)
problems = Dict{String, Any}()
let n = 100, m = 200
    X = randn(n, n)
    A = randn(m, n)
    b = A * randn(n)
    problems["dense QP"] = (Matrix(Symmetric(X'X / n + I)), randn(n), A, b .- 1, b .+ 1)
end
let n = 300, m = 600
    S = sprandn(n, n, 0.02)
    A = Matrix(sprandn(m, n, 0.02))
    b = A * randn(n)
    problems["sparse-ish"] = (Matrix(Symmetric(S * S')) + 2I, randn(n), A, b .- 1, b .+ 1)
end
let n = 60, m = 120
    X = randn(n, n)
    A = randn(m, n)
    b = A * randn(n)
    problems["equality-heavy"] = (Matrix(Symmetric(X'X / n + I)), randn(n), A, copy(b), b)
end

println("\nTo the same tolerance: plain, accelerated with the safeguard, and without it.")
println("Setup is inside the timing, as it is for any single solve.")
println("`declined` counts the extrapolations the safeguard threw away.\n")
@printf(
    "%-16s %8s %8s %8s %9s %9s %9s %9s %s\n",
    "problem", "plain", "guarded", "unguarded", "plain ms", "guard ms", "unguard ms",
    "declined", "status"
)
println("-"^96)
for (name, p) in sort(collect(problems); by = first)
    P, q, A, l, u = p
    sp, ip, _ = run_loop(P, q, A, l, u; accelerate = false)
    sg, ig, dg = run_loop(P, q, A, l, u; accelerate = true, safeguard = true)
    sn, iN, _ = run_loop(P, q, A, l, u; accelerate = true, safeguard = false)
    tp = med(@be run_loop($P, $q, $A, $l, $u; accelerate = false) seconds = 5)
    tg = med(
        @be run_loop($P, $q, $A, $l, $u; accelerate = true, safeguard = true) seconds = 5
    )
    tn = med(
        @be run_loop($P, $q, $A, $l, $u; accelerate = true, safeguard = false) seconds = 5
    )
    st = (sp === sg === sn) ? String(Symbol(sp)) : "$(sp)/$(sg)/$(sn)"
    @printf(
        "%-16s %8d %8d %9d %9.2f %9.2f %10.2f %9d %s\n",
        name, ip, ig, iN, 1.0e3tp, 1.0e3tg, 1.0e3tn, dg, st
    )
    push!(
        cases, (;
            case = name, plain_iters = ip, guarded_iters = ig, unguarded_iters = iN,
            declined = dg, plain_seconds = tp, guarded_seconds = tg,
            unguarded_seconds = tn, status = st,
        )
    )
end

open(RESULTS, "w") do io
    JSON.print(io, Dict("julia_version" => string(VERSION), "eps" => TOL, "cases" => cases), 2)
end
println("\nwrote $RESULTS")
