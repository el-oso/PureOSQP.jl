# PureOSQP on PureBLAS instead of OpenBLAS.
#
# PureBLAS.activate() overlays per-symbol forwards onto libblastrampoline, so PureOSQP
# needs no code changes: the same `mul!`, `cholesky!` and `ldiv!` calls are rerouted in
# process. Note `BLAS.get_config()` does NOT show this — OpenBLAS stays loaded and the
# forwards sit on top of it — so activation is asserted with `PureBLAS.is_active()`.
#
# PureBLAS is not registered. Set it up once with:
#
#     julia --project=bench/pureblas -e 'using Pkg; \
#       Pkg.develop([PackageSpec(path="/path/to/PureBLAS.jl"), PackageSpec(path=".")]); \
#       Pkg.add(["BenchmarkTools", "JSON"])'
#
# then run:  julia --project=bench/pureblas bench/pureblas_backend.jl
# The script exits cleanly when PureBLAS is unavailable.
using PureOSQP, LinearAlgebra, BenchmarkTools, Random, Printf, JSON

const HAVE_PUREBLAS = try
    @eval using PureBLAS
    true
catch
    false
end

if !HAVE_PUREBLAS
    @info "PureBLAS not available in this environment; skipping. See the header for setup."
    exit(0)
end

BLAS.set_num_threads(1)

const OPTS = (eps_abs = 1.0e-6, eps_rel = 1.0e-6, max_iter = 20_000, scaling = 10)

function make_problem(n, m; seed = 0)
    Random.seed!(seed)
    X = randn(n, n)
    P = Matrix(X'X / n + I)
    q = randn(n)
    A = randn(m, n)
    b = A * randn(n)
    return (P, q, A, b .- rand(m), b .+ rand(m))
end

"Solve the same problem under each backend, asserting which one is live at solve time."
function bench_pair(n, m; seed = n + m)
    P, q, A, l, u = make_problem(n, m; seed)
    PureBLAS.is_active() && PureBLAS.deactivate()
    s_open = PureOSQP.solve(P, q, A, l, u; OPTS...)
    @assert !PureBLAS.is_active()
    t_open = @belapsed PureOSQP.solve($P, $q, $A, $l, $u; OPTS...)
    PureBLAS.activate()
    @assert PureBLAS.is_active()
    s_pure = PureOSQP.solve(P, q, A, l, u; OPTS...)
    t_pure = @belapsed PureOSQP.solve($P, $q, $A, $l, $u; OPTS...)
    PureBLAS.deactivate()
    return (;
        n, m, t_openblas = t_open, t_pureblas = t_pure, ratio = t_open / t_pure,
        iter_openblas = s_open.iter, iter_pureblas = s_pure.iter,
        dx = maximum(abs, s_open.x .- s_pure.x),
        dobj = abs(s_open.obj_val - s_pure.obj_val),
    )
end

"Time one BLAS call under each backend."
function op_pair(f)
    PureBLAS.is_active() && PureBLAS.deactivate()
    f()
    to = @belapsed $f()
    PureBLAS.activate()
    f()
    tp = @belapsed $f()
    PureBLAS.deactivate()
    return (to, tp)
end

"""
The individual BLAS calls PureOSQP makes, so a whole-solve ratio can be attributed rather
than guessed at. `gemv` and `trsv` run every iteration; `syrk` and `potrf` only on a rho
update.
"""
function op_breakdown(n, m)
    Random.seed!(1)
    A = randn(m, n)
    W = randn(m, n)
    x = randn(n)
    y = randn(m)
    ox, oy = similar(x), similar(y)
    R = Matrix{Float64}(undef, n, n)
    Xn = randn(n, n)
    S = Matrix(Symmetric(Xn'Xn + n * I))
    F = cholesky(Symmetric(copy(S)))
    b = randn(n)
    At = Matrix(A')
    return [
        "gemv  A*x" => op_pair(() -> mul!(oy, A, x)),
        "gemv  A'y (trans)" => op_pair(() -> mul!(ox, A', y)),
        "gemv  At*y (dense)" => op_pair(() -> mul!(ox, At, y)),
        "syrk  W'W" => op_pair(() -> mul!(R, W', W)),
        "potrf" => op_pair(() -> cholesky!(Symmetric(copy(S)); check = false)),
        "trsv  F\\b" => op_pair(() -> ldiv!(F, copy(b))),
    ]
end

const CASES = [(25, 50), (50, 100), (100, 200), (200, 400), (100, 50)]

# Record the thread configuration of both sides. These are not symmetric: OpenBLAS is
# pinned with BLAS.set_num_threads, while PureBLAS is plain Julia and is bounded by
# Threads.nthreads(), which BLAS.set_num_threads does not touch. Print both so the
# comparison can be read for what it is.
@printf(
    "PureOSQP: OpenBLAS vs PureBLAS %s\n  BLAS threads = %d, Julia threads = %d, CPU threads = %d\n\n",
    pkgversion(PureBLAS), BLAS.get_num_threads(), Threads.nthreads(), Sys.CPU_THREADS
)
@printf(
    "%5s %6s | %11s %11s %8s | %6s %6s | %9s\n",
    "n", "m", "OpenBLAS", "PureBLAS", "ratio", "it_ob", "it_pb", "|Δx|"
)
println("-"^76)
results = NamedTuple[]
for (n, m) in CASES
    r = bench_pair(n, m)
    push!(results, r)
    @printf(
        "%5d %6d | %9.3f ms %9.3f ms %7.2fx | %6d %6d | %9.2e\n",
        r.n, r.m, 1.0e3r.t_openblas, 1.0e3r.t_pureblas, r.ratio,
        r.iter_openblas, r.iter_pureblas, r.dx
    )
    flush(stdout)
end

println("\nPer-operation breakdown at n=200, m=400:")
ops = op_breakdown(200, 400)
for (name, (to, tp)) in ops
    @printf("  %-20s OpenBLAS %8.2f µs   PureBLAS %8.2f µs   %5.2fx\n", name, 1.0e6to, 1.0e6tp, to / tp)
end

open(joinpath(@__DIR__, "results", "pureblas_backend.json"), "w") do io
    JSON.print(
        io, Dict(
            "pureblas_version" => string(pkgversion(PureBLAS)),
            # The version string does not move between branches; record the commit, which does.
            "pureblas_commit" => try
                strip(read(`git -C $(pkgdir(PureBLAS)) rev-parse --short HEAD`, String))
            catch
                "unknown"
            end,
            "julia_version" => string(VERSION),
            "blas_threads" => BLAS.get_num_threads(),
            "julia_threads" => Threads.nthreads(),
            "cpu_threads" => Sys.CPU_THREADS,
            "solve" => [Dict(string(k) => v for (k, v) in pairs(r)) for r in results],
            "ops" => [Dict("op" => nm, "t_openblas" => t[1], "t_pureblas" => t[2]) for (nm, t) in ops],
        ), 2
    )
end
println("\nsaved bench/results/pureblas_backend.json")
