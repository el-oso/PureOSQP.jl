# Curiosity comparison, not a gate: how does Clarabel.rs (the Rust implementation, built
# with its `faer-sparse` feature so it factors through faer rather than its bundled QDLDL)
# compare with PureOSQP's interior-point method and with Clarabel.jl, on the same seven
# OSQP-suite classes bench/ipm_vs_clarabel.jl already uses, at the same small sizes and
# the same 1e-8 tolerance?
#
# Clarabel.rs is not a PureOSQP.jl dependency: this script shells out to a prebuilt binary
# and degrades to a Julia-only table (with a note) when `cargo` or the crate build is
# unavailable, so nothing here can break a normal `Pkg.test()` or CI run.
#
# Rerun:
#     julia --project=bench bench/clarabel_rs_compare.jl
#
# The first run builds bench/clarabel_rs (Cargo.toml + src/main.rs) out-of-tree, with a
# target directory under `tempdir()` rather than inside the repository:
#     cargo build --release --manifest-path bench/clarabel_rs/Cargo.toml --target-dir <tmp>
# Rebuild explicitly (e.g. after editing src/main.rs) with:
#     rm -rf <tmp>   # printed by this script on first run
#
# Methodology: for each class, PureOSQP's `InteriorPoint()` and Clarabel.jl are benchmarked
# with Chairmarks (`@b`, which reports the field-wise *minimum* over samples collected in a
# `SECONDS`-long window). Clarabel.rs's own `main.rs` times `DefaultSolver::new` + `solve()`
# with its own `std::time::Instant` clock in the same way (minimum over repeats in the same
# window) and prints that as `solve_time_self_s`; this script separately times its own
# `run(::Cmd)` wall clock, which additionally includes process start and JSON on stdout, so
# the two numbers are reported side by side rather than conflated. Problem files (CSC arrays
# plus one-sided bounds, `Ax <= b`) are written to a scratch directory excluded from both
# timings. Both processes are pinned to core 15 and run single-threaded.
using PureOSQP, Clarabel
using LinearAlgebra, SparseArrays, Random, JSON, Chairmarks, Printf

include(joinpath(@__DIR__, "suite_problems.jl"))

BLAS.set_num_threads(1)

const RESULTS = joinpath(@__DIR__, "results", "clarabel_rs_compare.json")
const TOL = 1.0e-8
const SECONDS = 0.3
const CORE = 15

"The smallest size of each suite class that still exercises its structure (matches bench/ipm_vs_clarabel.jl)."
const SMALL_CASES = [
    ("Random QP", () -> random_qp(6)),
    ("Eq QP", () -> eq_qp(20)),
    ("Portfolio", () -> portfolio(1)),
    ("Lasso", () -> lasso(2)),
    ("SVM", () -> svm(2)),
    ("Huber", () -> huber(2)),
    ("Control", () -> control(4)),
]

"Clarabel takes one-sided cones: the two-sided rows are stacked as `Ax <= u, -Ax <= -l`."
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
        verbose = false, tol_gap_abs = TOL, tol_gap_rel = TOL,
        tol_feas = TOL,
    )
    solver = Clarabel.Solver()
    Clarabel.setup!(solver, Pc, q, Ac, bc, [Clarabel.NonnegativeConeT(length(bc))], settings)
    return Clarabel.solve!(solver)
end

"Write `Pc` (triu), `q`, `Ac`, `bc` as 0-based CSC text, in the format bench/clarabel_rs/src/main.rs parses."
function write_problem(path, Pc, q, Ac, bc)
    return open(path, "w") do io
        println(io, size(Pc, 2), " ", size(Ac, 1), " ", nnz(Pc), " ", nnz(Ac))
        println(io, join(Pc.colptr .- 1, ' '))
        println(io, join(Pc.rowval .- 1, ' '))
        println(io, join(Pc.nzval, ' '))
        println(io, join(q, ' '))
        println(io, join(Ac.colptr .- 1, ' '))
        println(io, join(Ac.rowval .- 1, ' '))
        println(io, join(Ac.nzval, ' '))
        println(io, join(bc, ' '))
    end
end

"Build bench/clarabel_rs out-of-tree; returns the binary path, or `nothing` if unavailable."
function ensure_clarabel_rs_binary()
    if isnothing(Sys.which("cargo"))
        @warn "cargo not found on PATH: skipping the Clarabel.rs comparison"
        return nothing
    end
    manifest = joinpath(@__DIR__, "clarabel_rs", "Cargo.toml")
    target_dir = joinpath(tempdir(), "pureosqp_clarabel_rs_target")
    bin = joinpath(target_dir, "release", "clarabel_rs_bench")
    if !isfile(bin)
        println("Building Clarabel.rs driver out-of-tree at $target_dir ...")
        cmd = `cargo build --release --manifest-path $manifest --target-dir $target_dir`
        try
            run(cmd)
        catch e
            @warn "cargo build failed: skipping the Clarabel.rs comparison" exception = e
            return nothing
        end
    end
    return isfile(bin) ? bin : nothing
end

"""
Run the Rust driver over one data directory; returns its per-case results in the same
order as `SMALL_CASES` (the driver sorts the `%02d_<slug>.txt` files it reads, and
`write_problem` numbers them in that same order), plus the subprocess wall clock.
"""
function run_clarabel_rs(bin, data_dir)
    cmd = `taskset -c $CORE $bin $data_dir $SECONDS $TOL`
    cmd = setenv(cmd, "RAYON_NUM_THREADS" => "1"; dir = pwd())
    out = IOBuffer()
    wall = @elapsed run(pipeline(cmd, stdout = out, stderr = stderr))
    parsed = JSON.parse(String(take!(out)))
    return parsed, wall
end

function run_case(name, gen, data_dir, idx)
    P, q, A, l, u = gen()
    n, m = size(A, 2), size(A, 1)

    ipm = PureOSQP.solve(P, q, A, l, u, PureOSQP.InteriorPoint(); eps_abs = TOL, eps_rel = TOL)
    ipm_bm = @b PureOSQP.solve($P, $q, $A, $l, $u, PureOSQP.InteriorPoint(); eps_abs = TOL, eps_rel = TOL) seconds = SECONDS

    clar = run_clarabel(P, q, A, l, u)
    clar_bm = @b run_clarabel($P, $q, $A, $l, $u) seconds = SECONDS

    Pc, Ac, bc = clarabel_form(P, A, l, u)
    slug = replace(lowercase(name), ' ' => '_')
    write_problem(joinpath(data_dir, @sprintf("%02d_%s.txt", idx, slug)), Pc, q, Ac, bc)

    dx_clarabel = maximum(abs, ipm.x .- clar.x; init = 0.0) / max(1.0, maximum(abs, ipm.x; init = 0.0))

    return (; name, n, m, ipm, ipm_bm, clar, clar_bm, dx_clarabel)
end

mkpath(dirname(RESULTS))
data_dir = mktempdir()
cases = [run_case(name, gen, data_dir, i) for (i, (name, gen)) in enumerate(SMALL_CASES)]

bin = ensure_clarabel_rs_binary()
clarabel_rs, rs_wall = isnothing(bin) ? (nothing, NaN) : run_clarabel_rs(bin, data_dir)

@printf(
    "%-10s %-8s %-7s | %-16s | %-16s | %-16s | %s\n",
    "class", "n", "m", "IPM", "Clarabel.jl", "Clarabel.rs", "max |Δx| (IPM vs Clarabel.jl / .rs)"
)
println("-"^140)

results = map(enumerate(cases)) do (i, c)
    rs = isnothing(clarabel_rs) ? nothing : clarabel_rs[i]
    rs_ms = isnothing(rs) ? NaN : 1.0e3 * rs["solve_time_self_s"]
    rs_iter = isnothing(rs) ? -1 : rs["iterations"]
    dx_rs = isnothing(rs) ? NaN : maximum(abs, c.ipm.x .- rs["x"]; init = 0.0) / max(1.0, maximum(abs, c.ipm.x; init = 0.0))

    @printf(
        "%-10s n=%-4d m=%-5d | %3d it %7.3f ms | %3d it %7.3f ms | %3d it %7.3f ms | %.1e / %.1e\n",
        c.name, c.n, c.m, c.ipm.iter, 1.0e3c.ipm_bm.time,
        c.clar.iterations, 1.0e3c.clar_bm.time, rs_iter, rs_ms,
        c.dx_clarabel, dx_rs,
    )
    flush(stdout)

    Dict(
        "name" => c.name, "n" => c.n, "m" => c.m,
        "ipm" => Dict("iter" => c.ipm.iter, "status" => String(Symbol(c.ipm.status)), "time_ms" => 1.0e3c.ipm_bm.time, "obj" => c.ipm.obj_val),
        "clarabel_jl" => Dict("iter" => c.clar.iterations, "status" => String(Symbol(c.clar.status)), "time_ms" => 1.0e3c.clar_bm.time, "obj" => c.clar.obj_val),
        "clarabel_rs" => isnothing(rs) ? nothing : Dict(
                "iter" => rs["iterations"], "status" => rs["status"], "time_ms" => rs_ms,
                "obj" => rs["obj_val"], "reps" => rs["reps"], "linsolver" => rs["linsolver"],
            ),
        "dx_ipm_clarabel_jl" => c.dx_clarabel, "dx_ipm_clarabel_rs" => dx_rs,
    )
end

open(RESULTS, "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "blas_threads" => BLAS.get_num_threads(),
            "tol" => TOL,
            "seconds_per_benchmark" => SECONDS,
            "taskset_core" => CORE,
            "clarabel_jl_version" => string(pkgversion(Clarabel)),
            "clarabel_rs_available" => !isnothing(bin),
            "clarabel_rs_subprocess_wall_s" => rs_wall,
            "clarabel_rs_features" => "faer-sparse, serde (default-features = false)",
            "clarabel_rs_direct_solve_method" => "faer (forced; \"auto\" falls back to QDLDL below faer's flops/nnz(L) > 40 switch threshold, which these small problems don't cross)",
            "results" => results,
        ), 2
    )
end
println("\nsaved $RESULTS")
if isnothing(bin)
    println("Clarabel.rs comparison skipped (cargo or the build was unavailable).")
else
    @printf("Clarabel.rs subprocess wall clock (all %d cases, includes process start + JSON I/O): %.3f s\n", length(cases), rs_wall)
end
