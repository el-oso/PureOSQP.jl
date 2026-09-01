# The sequential-resolve path: P and A fixed, q/l/u changing every step, as a receding
# horizon loop does. Compares
#   (1) PureOSQP with update!      -- setup once, then update! + solve! per step
#   (2) PureOSQP without update!   -- a fresh setup + solve per step, the naive loop
#
# There is no libosqp 1.x column: it has no Julia wrapper, and reaching it needs a `ccall`
# against the library `OSQP_jll` ships. See `docs/src/roadmap.md`.
#
# When PureBLAS is present, (1) is measured a second time with it activated. This loop is
# where the BLAS choice is least predictable from the single-solve numbers: a re-solve
# sequence refactorizes far more often per iteration than one long solve does, so it
# weights `potrf` and `potri` — where the two libraries differ most, and in opposite
# directions — much more heavily. Run it with `--project=bench/pureblas`; the column is
# skipped otherwise.
using PureOSQP, LinearAlgebra, Random, Printf, JSON

const HAVE_PUREBLAS = try
    @eval using PureBLAS
    true
catch
    false
end

BLAS.set_num_threads(1)

const STEPS = 20


function make_sequence(n, m; seed = 0, steps = STEPS)
    Random.seed!(seed)
    X = randn(n, n)
    P = X'X / n + I
    A = randn(m, n)
    q0 = randn(n)
    b = A * randn(n)
    l0, u0 = b .- rand(m), b .+ rand(m)
    seq = map(1:steps) do _
        bk = A * randn(n)
        (q = randn(n), l = bk .- rand(m), u = bk .+ rand(m))
    end
    return (P, q0, A, l0, u0, seq)
end

const OPTS = (eps_abs = 1.0e-6, eps_rel = 1.0e-6, max_iter = 20_000)
function pure_with_update(P, q0, A, l0, u0, seq)
    ws = setup(P, q0, A, l0, u0; OPTS...)
    objs = Float64[]
    t = @elapsed for s in seq
        update!(ws; q = s.q, l = s.l, u = s.u)
        push!(objs, PureOSQP.solve!(ws).obj_val)
    end
    return (t, objs, ws.refactor_count)
end

"""
The same `update!` sequence with PureBLAS rerouting LinearAlgebra. Activation is asserted
at the measurement rather than assumed: `BLAS.get_config()` cannot see the libblastrampoline
forwards, so a reroute that silently failed would look like a clean tie.
"""
function pureblas_with_update(P, q0, A, l0, u0, seq)
    PureBLAS.is_active() && PureBLAS.deactivate()
    pure_with_update(P, q0, A, l0, u0, seq[1:2])
    PureBLAS.activate()
    @assert PureBLAS.is_active()
    pure_with_update(P, q0, A, l0, u0, seq[1:2])       # compile on this backend too
    t, objs, nrefac = pure_with_update(P, q0, A, l0, u0, seq)
    @assert PureBLAS.is_active()
    PureBLAS.deactivate()
    return (t, objs, nrefac)
end

function pure_without_update(P, q0, A, l0, u0, seq)
    objs = Float64[]
    t = @elapsed for s in seq
        push!(objs, PureOSQP.solve(P, s.q, A, s.l, s.u; OPTS...).obj_val)
    end
    return (t, objs)
end

HAVE_PUREBLAS || @info "PureBLAS not available; that column is skipped."

const CASES = [(10, 20), (25, 50), (50, 100), (100, 200), (200, 400)]

results = []
@printf(
    "%5s %6s | %11s %11s %8s | %11s %8s | %5s\n", "n", "m",
    "update!", "fresh setup", "saved", "+PureBLAS", "vs(pb)", "refac"
)
println("-"^80)
for (n, m) in CASES
    P, q0, A, l0, u0, seq = make_sequence(n, m; seed = n + m)
    pure_with_update(P, q0, A, l0, u0, seq[1:2])          # warm up compilation
    t_upd, objs_upd, nrefac = pure_with_update(P, q0, A, l0, u0, seq)
    t_new, objs_new = pure_without_update(P, q0, A, l0, u0, seq)
    # Two runs converged to the same tolerance need only agree to that tolerance; a
    # warm-started sequence and a cold sequence stop at different points inside it.
    agree = maximum(abs, objs_upd .- objs_new) / max(1, maximum(abs, objs_new))
    @assert agree < 100 * OPTS.eps_abs "update! and fresh setup disagree by $agree"
    t_pb, dx_pb = if HAVE_PUREBLAS
        tp, objs_pb, refac_pb = pureblas_with_update(P, q0, A, l0, u0, seq)
        # The BLAS must not change the path, only the speed.
        @assert refac_pb == nrefac "PureBLAS changed the factorization count"
        (tp, maximum(abs, objs_upd .- objs_pb) / max(1, maximum(abs, objs_upd)))
    else
        (NaN, NaN)
    end
    push!(
        results, (;
            n, m, steps = STEPS, t_update = t_upd, t_fresh = t_new,
            t_pureblas = t_pb, dobj_pureblas = dx_pb, refactorizations = nrefac,
        )
    )
    @printf(
        "%5d %6d | %9.2f ms %9.2f ms %7.2fx | %9.2f ms %7.2fx | %5d\n",
        n, m, 1.0e3t_upd, 1.0e3t_new, t_new / t_upd, 1.0e3t_pb, t_upd / t_pb, nrefac
    )
    flush(stdout)
end

println("\n$STEPS solves per case. \"refac\" counts factorizations across the whole")
println("sequence: 1 from setup plus one per step whose constraint classification changed.")

open(joinpath(@__DIR__, "results", "update_bench.json"), "w") do io
    JSON.print(
        io, Dict(
            "steps" => STEPS, "julia_version" => string(VERSION),
            "pureblas_commit" => !HAVE_PUREBLAS ? "absent" : try
                    strip(read(`git -C $(pkgdir(PureBLAS)) rev-parse --short HEAD`, String))
            catch
                    "unknown"
            end,
            # A skipped column is `NaN` in the table and `null` here: JSON has no NaN, and
            # `allownan` would write one this file's own readers could not parse back.
            "results" => [
                Dict(
                        string(k) => (v isa AbstractFloat && isnan(v) ? nothing : v)
                        for (k, v) in pairs(r)
                    ) for r in results
            ],
        ), 2
    )
end
println("saved bench/results/update_bench.json")
