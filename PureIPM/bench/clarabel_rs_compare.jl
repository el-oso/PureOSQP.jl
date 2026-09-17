# Curiosity comparison, not a gate: how does Clarabel.rs (the Rust implementation, built
# with its `faer-sparse` feature so it factors through faer rather than its bundled QDLDL)
# compare with PureOSQP's interior-point method and with Clarabel.jl, on the same seven
# OSQP-suite classes PureIPM/bench/ipm_vs_clarabel.jl already uses, at the same small sizes and
# the same 1e-8 tolerance?
#
# Clarabel.rs is not a PureOSQP.jl dependency: this script shells out to a prebuilt binary
# and degrades to a Julia-only table (with a note) when `cargo` or the crate build is
# unavailable, so nothing here can break a normal `Pkg.test()` or CI run.
#
# Rerun:
#     julia --project=bench PureIPM/bench/clarabel_rs_compare.jl [run_label]
# `run_label`, if given, is appended to the results filename (`clarabel_rs_compare_<label>.json`)
# so that two independent invocations can be diffed against each other instead of one
# overwriting the other — the run-to-run spread that decides whether a measured gap between
# solvers is real or still inside the noise (see the protocol notes below).
#
# The first run builds bench/clarabel_rs (Cargo.toml + src/main.rs) out-of-tree, with a
# target directory under `tempdir()` rather than inside the repository:
#     cargo build --release --manifest-path bench/clarabel_rs/Cargo.toml --target-dir <tmp>
# Rebuild explicitly (e.g. after editing src/main.rs) with:
#     rm -rf <tmp>   # printed by this script on first run
#
# Protocol (tight enough to tell a real difference from measurement noise):
#   - The two Julia solvers (`InteriorPoint()` and `Clarabel.jl`) are measured in this one
#     warm process, interleaved per class as A-B-B-A (`IPM, Clarabel.jl, Clarabel.jl, IPM`)
#     rather than one solver's whole run followed by the other's — a slow drift over the
#     run (thermal ramp, a scheduler hiccup) then lands on both solvers instead of biasing
#     whichever ran second. Each block is its own `@be` pass (Chairmarks); the two `IPM`
#     blocks' samples are pooled, and likewise the two `Clarabel.jl` blocks', before taking
#     the field-wise minimum and median.
#   - The one-shot `ipm`/`clar` solves above the benchmarked blocks double as the warm-up
#     call each solver needs before its first timed sample (compilation is otherwise paid
#     inside the first `@be` sample, not before it).
#   - Both solvers see the same `P, q, A, l, u`, generated once per class and reused for
#     every block; both run at `eps_abs = eps_rel = 1e-8`; `BLAS.set_num_threads(1)`; the
#     whole process is pinned to `CORE` via `sched_setaffinity` (see `pin_to_cpu!` below) —
#     `taskset -c $CORE` from the shell would only pin the shell's child, which is this same
#     process, but pinning from inside means the pin holds regardless of how the script is
#     launched.
#   - Chairmarks reports `gc_fraction` per sample. Each pooled result records whether the
#     *fastest* sample had any GC in its window (`min_gc`, which would mean the reported
#     minimum is itself contaminated) and whether *any* pooled sample did (`any_gc`/`n_gc`),
#     without excluding those samples — the field-wise minimum already discounts them unless
#     GC ran in literally the fastest one.
#   - Clarabel.rs cannot join that one process: it is measured the way its own binary can be
#     measured, which is a *different* process. `main.rs` times `DefaultSolver::new` + `solve()`
#     with its own `std::time::Instant` clock, taking the minimum over repeats inside the same
#     `SECONDS` budget (its own analogue of Chairmarks' field-wise minimum), and reports that as
#     `solve_time_self_s`. This script separately times its own `run(::Cmd)` wall clock, which
#     additionally includes process start and JSON on stdout — the two numbers are reported side
#     by side rather than conflated. This means Clarabel.rs's number and the two Julia solvers'
#     numbers are NOT measurements of the same process state (page cache, branch predictor,
#     core residency going in) even though all three are pinned to the same core in turn; a
#     cross-language gap here carries that caveat, a within-Julia gap does not.
#   - Problem files (CSC arrays plus one-sided bounds, `Ax <= b`) are written to a scratch
#     directory excluded from both timings.
using PureOSQP, PureIPM, Clarabel
using LinearAlgebra, SparseArrays, Random, JSON, Chairmarks, Printf, Statistics

include(joinpath(@__DIR__, "..", "..", "bench", "suite_problems.jl"))

BLAS.set_num_threads(1)

const RUN_LABEL = isempty(ARGS) ? "" : "_" * ARGS[1]
const RESULTS = joinpath(@__DIR__, "results", "clarabel_rs_compare$(RUN_LABEL).json")
const TOL = 1.0e-8
const SECONDS = 0.3
const CORE = 15

"""
Pin this process to `cpu` via `sched_setaffinity` (glibc/Linux). `cpu_set_t` is a 1024-bit
mask (16 `UInt64` words) in the glibc ABI on x86_64. Pinning inside the process, rather than
relying on a `taskset` wrapper around it, holds regardless of how the script is launched.
"""
function pin_to_cpu!(cpu::Integer)
    mask = zeros(UInt64, 16)
    mask[cpu ÷ 64 + 1] |= UInt64(1) << (cpu % 64)
    ret = ccall(:sched_setaffinity, Cint, (Cint, Csize_t, Ptr{UInt64}), 0, sizeof(mask), mask)
    ok = iszero(ret)
    ok || @warn "sched_setaffinity($cpu) failed" errno = Base.Libc.errno()
    return ok
end

"The other logical CPU sharing `cpu`'s physical core (its SMT sibling), or `nothing`."
function smt_sibling(cpu::Integer)
    path = "/sys/devices/system/cpu/cpu$cpu/topology/thread_siblings_list"
    isfile(path) || return nothing
    ids = parse.(Int, split(strip(read(path, String)), ','))
    others = filter(!=(cpu), ids)
    return isempty(others) ? nothing : first(others)
end

"`scaling_governor` for cpu0, standing in for the machine-wide policy."
function cpu_governor()
    path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
    return isfile(path) ? strip(read(path, String)) : "unknown"
end

pinned = pin_to_cpu!(CORE)
sibling = smt_sibling(CORE)
governor = cpu_governor()
println("taskset core=$CORE (pinned=$pinned, SMT sibling=$sibling), scaling_governor=$governor")
if governor != "powersave"
    @warn "expected scaling_governor=powersave on this machine; got $governor"
end
flush(stdout)

"The smallest size of each suite class that still exercises its structure (matches PureIPM/bench/ipm_vs_clarabel.jl)."
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

"""
Field-wise minimum and median over a pooled set of Chairmarks `Sample`s (already the
combined samples of both `A` or both `B` blocks of the interleave). `min_gc` flags whether
the single fastest sample also had `gc_fraction > 0` — the one case where the reported
minimum is itself contaminated by garbage collection rather than merely coexisting with GC
elsewhere in the pool. `any_gc`/`n_gc` report the wider contamination without excluding it:
at these sub-millisecond scales, a benchmark that ran long enough to see zero GC-affected
samples out of thousands would prove nothing about steady-state behavior.
"""
function summarize(samples)
    times = getfield.(samples, :time)
    gcs = getfield.(samples, :gc_fraction)
    i = argmin(times)
    return (;
        n = length(samples),
        min_s = times[i],
        min_gc = gcs[i] > 0,
        median_s = median(times),
        any_gc = any(>(0), gcs),
        n_gc = count(>(0), gcs),
    )
end

function run_case(name, gen, data_dir, idx)
    P, q, A, l, u = gen()
    n, m = size(A, 2), size(A, 1)

    # Warm-up (compiles both call paths) and the correctness check `dx_clarabel` needs.
    ipm = PureOSQP.solve(P, q, A, l, u, PureIPM.InteriorPoint(); eps_abs = TOL, eps_rel = TOL)
    clar = run_clarabel(P, q, A, l, u)

    # Interleaved A-B-B-A: two `@be` passes per solver, pooled below.
    ipm_a1 = @be PureOSQP.solve($P, $q, $A, $l, $u, PureIPM.InteriorPoint(); eps_abs = TOL, eps_rel = TOL) seconds = SECONDS
    clar_b1 = @be run_clarabel($P, $q, $A, $l, $u) seconds = SECONDS
    clar_b2 = @be run_clarabel($P, $q, $A, $l, $u) seconds = SECONDS
    ipm_a2 = @be PureOSQP.solve($P, $q, $A, $l, $u, PureIPM.InteriorPoint(); eps_abs = TOL, eps_rel = TOL) seconds = SECONDS

    ipm_stats = summarize(vcat(ipm_a1.samples, ipm_a2.samples))
    clar_stats = summarize(vcat(clar_b1.samples, clar_b2.samples))

    Pc, Ac, bc = clarabel_form(P, A, l, u)
    slug = replace(lowercase(name), ' ' => '_')
    write_problem(joinpath(data_dir, @sprintf("%02d_%s.txt", idx, slug)), Pc, q, Ac, bc)

    dx_clarabel = maximum(abs, ipm.x .- clar.x; init = 0.0) / max(1.0, maximum(abs, ipm.x; init = 0.0))

    return (; name, n, m, ipm, ipm_stats, clar, clar_stats, dx_clarabel)
end

mkpath(dirname(RESULTS))
data_dir = mktempdir()
cases = [run_case(name, gen, data_dir, i) for (i, (name, gen)) in enumerate(SMALL_CASES)]

bin = ensure_clarabel_rs_binary()
clarabel_rs, rs_wall = isnothing(bin) ? (nothing, NaN) : run_clarabel_rs(bin, data_dir)

@printf(
    "%-10s %-8s %-7s | %-22s | %-22s | %-16s | %s\n",
    "class", "n", "m", "IPM (min/med, µs)", "Clarabel.jl (min/med, µs)", "Clarabel.rs (µs)", "max |Δx| (IPM vs .jl / .rs)"
)
println("-"^150)

results = map(enumerate(cases)) do (i, c)
    rs = isnothing(clarabel_rs) ? nothing : clarabel_rs[i]
    rs_us = isnothing(rs) ? NaN : 1.0e6 * rs["solve_time_self_s"]
    rs_iter = isnothing(rs) ? -1 : rs["iterations"]
    dx_rs = isnothing(rs) ? NaN : maximum(abs, c.ipm.x .- rs["x"]; init = 0.0) / max(1.0, maximum(abs, c.ipm.x; init = 0.0))

    gc_flag(s) = s.min_gc ? "*" : (s.any_gc ? "+" : " ")

    @printf(
        "%-10s n=%-4d m=%-5d | %3d it %8.3f/%7.3f%s | %3d it %8.3f/%7.3f%s | %3d it %8.3f | %.1e / %.1e\n",
        c.name, c.n, c.m,
        c.ipm.iter, 1.0e6c.ipm_stats.min_s, 1.0e6c.ipm_stats.median_s, gc_flag(c.ipm_stats),
        c.clar.iterations, 1.0e6c.clar_stats.min_s, 1.0e6c.clar_stats.median_s, gc_flag(c.clar_stats),
        rs_iter, rs_us,
        c.dx_clarabel, dx_rs,
    )
    flush(stdout)

    Dict(
        "name" => c.name, "n" => c.n, "m" => c.m,
        "ipm" => Dict(
            "iter" => c.ipm.iter, "status" => String(Symbol(c.ipm.status)), "obj" => c.ipm.obj_val,
            "min_us" => 1.0e6c.ipm_stats.min_s, "median_us" => 1.0e6c.ipm_stats.median_s,
            "n_samples" => c.ipm_stats.n, "min_sample_gc" => c.ipm_stats.min_gc,
            "any_sample_gc" => c.ipm_stats.any_gc, "n_gc_samples" => c.ipm_stats.n_gc,
        ),
        "clarabel_jl" => Dict(
            "iter" => c.clar.iterations, "status" => String(Symbol(c.clar.status)), "obj" => c.clar.obj_val,
            "min_us" => 1.0e6c.clar_stats.min_s, "median_us" => 1.0e6c.clar_stats.median_s,
            "n_samples" => c.clar_stats.n, "min_sample_gc" => c.clar_stats.min_gc,
            "any_sample_gc" => c.clar_stats.any_gc, "n_gc_samples" => c.clar_stats.n_gc,
        ),
        "clarabel_rs" => isnothing(rs) ? nothing : Dict(
                "iter" => rs["iterations"], "status" => rs["status"], "min_us" => rs_us,
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
            "taskset_pinned" => pinned,
            "smt_sibling_core" => sibling,
            "cpu_scaling_governor" => governor,
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
println("(* = fastest sample had GC running in its window; + = some pooled sample did, fastest did not)")
