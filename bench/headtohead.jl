# PureOSQP vs OSQP.jl (libosqp 0.6.2) on dense QPs, with PureOSQP measured on both
# OpenBLAS and PureBLAS.
#
# The PureBLAS column is optional: PureBLAS is unregistered, so it is loaded if present and
# the column is skipped otherwise. `PureBLAS.activate()` reroutes LinearAlgebra through
# libblastrampoline, which `BLAS.get_config()` cannot report — `PureBLAS.is_active()` is
# the check, and it is asserted at every measurement.
#
# Agreement with libosqp 1.x is checked separately by bench/headtohead_v1.jl; it has no
# Julia wrapper, so timing it here would compare against a subprocess rather than a library.
using PureOSQP, OSQP, LinearAlgebra, SparseArrays, BenchmarkTools, Random, Printf, JSON

const HAVE_PUREBLAS = try
    @eval using PureBLAS
    true
catch
    false
end

BLAS.set_num_threads(1)

const SETTINGS = (
    eps_abs = 1.0e-6, eps_rel = 1.0e-6, max_iter = 20_000,
    scaling = 10, adaptive_rho = true, polish = false,
)

function make_problem(n, m; seed = 0)
    Random.seed!(seed)
    X = randn(n, n)
    P = X'X / n + I
    q = randn(n)
    A = randn(m, n)
    Ax = A * randn(n)
    return (P, q, A, Ax .- rand(m), Ax .+ rand(m))
end

function run_osqp(P, q, A, l, u)
    model = OSQP.Model()
    OSQP.setup!(
        model; P = sparse(Symmetric(P)), q = q, A = sparse(A), l = l, u = u,
        verbose = false, adaptive_rho_interval = 50, check_termination = 25,
        SETTINGS...
    )
    return OSQP.solve!(model)
end

function bench_case(n, m; seed = 0)
    P, q, A, l, u = make_problem(n, m; seed)
    sp = PureOSQP.solve(P, q, A, l, u; SETTINGS...)
    rc = run_osqp(P, q, A, l, u)
    @assert sp.status == PureOSQP.SOLVED "PureOSQP: $(sp.status)"
    @assert rc.info.status == :Solved "OSQP.jl: $(rc.info.status)"

    HAVE_PUREBLAS && PureBLAS.is_active() && PureBLAS.deactivate()
    t_pure = @belapsed PureOSQP.solve($P, $q, $A, $l, $u; SETTINGS...)
    t_osqp = @belapsed run_osqp($P, $q, $A, $l, $u)

    t_pb, dx = NaN, NaN
    if HAVE_PUREBLAS
        PureBLAS.activate()
        @assert PureBLAS.is_active()
        sb = PureOSQP.solve(P, q, A, l, u; SETTINGS...)
        t_pb = @belapsed PureOSQP.solve($P, $q, $A, $l, $u; SETTINGS...)
        PureBLAS.deactivate()
        @assert sb.iter == sp.iter "PureBLAS changed the iteration count"
        dx = maximum(abs, sb.x .- sp.x)
    end

    objdiff = abs(sp.obj_val - rc.info.obj_val) / max(1, abs(rc.info.obj_val))
    return (;
        n, m, t_openblas = t_pure, t_pureblas = t_pb, t_osqp,
        speedup = t_osqp / t_pure, speedup_pureblas = t_osqp / t_pb,
        iter_pure = sp.iter, iter_osqp = rc.info.iter, objdiff, dx_pureblas = dx,
    )
end

const CASES = [
    (10, 20), (25, 50), (50, 100), (100, 200), (200, 400), (400, 800),
    (100, 50), (200, 100), (100, 1000), (200, 2000),
]

HAVE_PUREBLAS || @info "PureBLAS not available; that column is skipped."

results = NamedTuple[]
@printf(
    "%5s %6s | %10s %10s %10s | %7s %7s | %6s %6s | %9s\n",
    "n", "m", "PureOSQP", "+PureBLAS", "OSQP.jl", "vs", "vs(pb)", "it_pu", "it_osqp", "obj rel Δ"
)
println("-"^102)
for (n, m) in CASES
    r = bench_case(n, m)
    push!(results, r)
    @printf(
        "%5d %6d | %8.3f ms %8.3f ms %8.3f ms | %6.2fx %6.2fx | %6d %6d | %9.2e\n",
        r.n, r.m, 1.0e3r.t_openblas, 1.0e3r.t_pureblas, 1.0e3r.t_osqp,
        r.speedup, r.speedup_pureblas, r.iter_pure, r.iter_osqp, r.objdiff
    )
    flush(stdout)
end

open(joinpath(@__DIR__, "results", "headtohead.json"), "w") do io
    JSON.print(
        io, Dict(
            "reference" => "OSQP.jl / libosqp 0.6.2",
            "julia_version" => string(VERSION),
            "blas_threads" => BLAS.get_num_threads(),
            "julia_threads" => Threads.nthreads(),
            "pureblas" => HAVE_PUREBLAS ? Dict(
                    "version" => string(pkgversion(PureBLAS)),
                    "commit" => try
                        strip(read(`git -C $(pkgdir(PureBLAS)) rev-parse --short HEAD`, String))
                catch
                        "unknown"
                end,
                ) : nothing,
            # JSON has no NaN, and a skipped PureBLAS column is recorded as one.
            "results" => [
                Dict(
                        string(k) => (v isa AbstractFloat && isnan(v) ? nothing : v)
                        for (k, v) in pairs(r)
                    ) for r in results
            ],
        ), 2
    )
end
println("\nsaved bench/results/headtohead.json")
