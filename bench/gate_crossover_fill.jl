# How the sparse-factored backends compare with the dense one as the factor fills in.
#
# `PureOSQPSparseArraysExt` takes a sparse factorization only while `nnz(L) < 0.05 n^2`, and
# rejects the reduced matrix even earlier if `nnz(R)` alone passes that same limit. This file
# measures what those limits are worth: it sweeps the fill a factor carries from about 1% of
# `n^2` up past 40%, and at each point times a whole `solve` -- setup, factorization and
# every iteration -- once per backend on one problem, so the only thing that differs between
# the timings is which linear system solver ran.
#
# The `choose_backend` method below is what makes the sweep possible: past the fill limit
# `setup` will not build a sparse backend at all, so the comparison needs a way to ask for
# one anyway. Extending a function in another module registers the method globally, whatever
# module the statement sits in, so running this file leaves that method in place for the rest
# of the session: run it in a process of its own, not in a shared daemon alongside anything
# whose selection matters.
#
# The module keeps this file's names to itself: benchmarks here are run through a persistent
# Julia session, where two scripts sharing `Main` would silently share their constants.
module GateCrossoverFill

using PureOSQP, LinearAlgebra, SparseArrays, Random, Printf, JSON, Chairmarks, Statistics
using LDLFactorizations

BLAS.set_num_threads(1)

const Ext = Base.get_extension(PureOSQP, :PureOSQPSparseArraysExt)

# Which backend `setup` builds for a sparse pair: `:auto` leaves the shipped selection alone,
# `:ldl` and `:cholmod` take that sparse backend at any fill.
const MODE = Ref(:auto)

function PureOSQP.choose_backend(
        P::SparseMatrixCSC, A::SparseMatrixCSC, proto::AbstractVector{T},
        n::Integer, m::Integer, D, E, c, rho_vec, sigma
    ) where {T <: Real}
    shipped() = invoke(
        PureOSQP.choose_backend,
        Tuple{Any, SparseMatrixCSC, AbstractVector{T}, Integer, Integer, Any, Any, Any, Any, Any},
        P, A, proto, n, m, D, E, c, rho_vec, sigma
    )
    MODE[] === :auto && return shipped()
    gram = Ext.reduced_gram(T, P, A, n)
    R = Ext.refill!(gram, P, A, rho_vec, E, D, c, sigma)
    if MODE[] === :ldl
        ls = PureOSQP.ldl_backend(gram, proto, n, Inf)
        return isnothing(ls) ? shipped() : (ls, true)
    end
    F = cholesky(Symmetric(R, :U); check = false)
    issuccess(F) || return shipped()
    L = sparse(F.L)
    Lt = SparseMatrixCSC(transpose(L))
    Ext.check_factor(L, n)
    Ext.check_factor(Lt, n)
    ls = Ext.SparseCholmod{T, typeof(proto), typeof(F)}(gram, F, L, Lt, F.p, similar(proto, T, n))
    return (ls, true)
end

const OPTS = (eps_abs = 1.0e-6, eps_rel = 1.0e-6, max_iter = 10_000)
const MODES = (:ldl, :cholmod, :dense, :auto)

"""
    banded_qp(n, band) -> (P, q, A, l, u)

A QP whose objective couples each variable with the `band` on either side of it. The reduced
matrix keeps that band and its factor carries no fill beyond it, so `band / n` sets the fill
directly and the sweep runs over the case most favourable to a sparse factorization.
"""
function banded_qp(n, band; seed = 0)
    Random.seed!(n + band + seed)
    rows, cols, vals = Int[], Int[], Float64[]
    for j in 1:n, i in max(1, j - band):(j - 1)
        push!(rows, i)
        push!(cols, j)
        push!(vals, randn())
    end
    U = sparse(rows, cols, vals, n, n)
    # Diagonally dominant, so `P` is positive definite whatever the draw.
    P = sparse(Symmetric(U + transpose(U))) + (2band + 3) * I
    A = spdiagm(n, n, -1 => randn(n - 1), 0 => randn(n) .+ 2, 1 => randn(n - 1))
    b = A * randn(n)
    return (P, randn(n), A, b .- rand(n), b .+ rand(n))
end

"""
    random_qp(n, density) -> (P, q, A, l, u)

Uniformly random sparsity. The reduced matrix stays far sparser than its factor, so this is
the family where the fill limit on `nnz(L)` is the gate that fires rather than the one on
`nnz(R)`.
"""
function random_qp(n, density; seed = 0)
    Random.seed!(n + round(Int, 1.0e5density) + seed)
    A = sprandn(n, n, density)
    S = sprandn(n, n, density)
    P = sparse(Symmetric(transpose(S) * S)) + (n * density + 1) * I
    b = A * randn(n)
    return (P, randn(n), A, b .- rand(n), b .+ rand(n))
end

"Solve `prob` end to end with the backend `mode` asks for."
function run_case(prob, mode)
    MODE[] = mode === :dense ? :auto : mode
    P, q, A, l, u = prob
    return mode === :dense ?
        PureOSQP.solve(P, q, A, l, u; linsys = :dense, OPTS...) :
        PureOSQP.solve(P, q, A, l, u; OPTS...)
end

"The workspace `mode` builds, for reading the backend and the fill it settled on."
function setup_case(prob, mode)
    MODE[] = mode === :dense ? :auto : mode
    P, q, A, l, u = prob
    return mode === :dense ?
        PureOSQP.setup(P, q, A, l, u; linsys = :dense, OPTS...) :
        PureOSQP.setup(P, q, A, l, u; OPTS...)
end

factor_nnz(ls) = hasproperty(ls, :L) ? nnz(ls.L) : 0
reduced_nnz(ls) = hasproperty(ls, :gram) ? nnz(ls.gram.R) : 0

"""
    measure(name, prob, budget) -> NamedTuple

Time one problem once per backend. Every mode solves the same data with the same settings,
so a difference in iteration count would mean the timings are not comparable; the count is
carried out with them and checked.
"""
function measure(name, prob, budget)
    n = length(prob[2])
    times, samples, iters, backends, statuses = Dict(), Dict(), Dict(), Dict(), Dict()
    nnz_L, nnz_R = 0, 0
    for mode in MODES
        ws = setup_case(prob, mode)
        sol = PureOSQP.solve!(ws)
        backends[mode] = string(PureOSQP.backend_name(ws.linsys))
        iters[mode] = sol.iter
        statuses[mode] = string(sol.status)
        if mode === :ldl
            nnz_L, nnz_R = factor_nnz(ws.linsys), reduced_nnz(ws.linsys)
        end
        b = @be run_case(prob, mode) seconds = budget
        samples[mode] = [s.time for s in b.samples]
        times[mode] = median(samples[mode])
    end
    iter_ok = allequal(iters[m] for m in MODES)
    sparse_best = min(times[:ldl], times[:cholmod])
    return (;
        name, n,
        nnz_L, nnz_R,
        fill_L = nnz_L / n^2, fill_R = nnz_R / n^2,
        iters, iter_ok, statuses, backends, times, samples,
        speedup_ldl = times[:dense] / times[:ldl],
        speedup_cholmod = times[:dense] / times[:cholmod],
        speedup_best = times[:dense] / sparse_best,
        speedup_auto = times[:dense] / times[:auto],
        # Against what the shipped selection delivers, which past the fill limit is a dense
        # factorization of a reduced matrix it may still have formed sparsely.
        ldl_vs_auto = times[:auto] / times[:ldl],
        cholmod_vs_auto = times[:auto] / times[:cholmod],
    )
end

function header()
    @printf(
        "%-22s %5s %7s %7s %6s | %9s %9s %9s %9s | %6s %6s %6s %6s\n",
        "problem", "n", "fill(R)", "fill(L)", "iter",
        "ldl", "cholmod", "dense", "auto(ship)",
        "ldl/d", "chol/d", "ldl/a", "chol/a"
    )
    println("-"^131)
    return nothing
end

function line(r)
    @printf(
        "%-22s %5d %7.3f %7.3f %6d | %6.2f ms %6.2f ms %6.2f ms %6.2f ms | %5.2fx %5.2fx %5.2fx %5.2fx%s\n",
        r.name, r.n, r.fill_R, r.fill_L, r.iters[:ldl],
        1.0e3r.times[:ldl], 1.0e3r.times[:cholmod], 1.0e3r.times[:dense], 1.0e3r.times[:auto],
        r.speedup_ldl, r.speedup_cholmod, r.ldl_vs_auto, r.cholmod_vs_auto,
        r.iter_ok ? "" : "  ITERATION COUNTS DIFFER"
    )
    return flush(stdout)
end

const SIZES = (200, 500, 1000)
# Fractions of `n` for the half-bandwidth, which is about half the fill the factor ends with.
const BANDS = (0.005, 0.025, 0.05, 0.09, 0.13, 0.17, 0.21, 0.26, 0.32)
const DENSITIES = (0.002, 0.004, 0.006, 0.01)

budget_for(n) = n >= 1000 ? 3.0 : 1.5

println("\nBanded: the reduced matrix keeps its band, so the factor carries no fill beyond it.\n")
header()
banded_rows = NamedTuple[]
for n in SIZES, f in BANDS
    band = max(1, round(Int, f * n))
    r = measure("banded b=$band", banded_qp(n, band), budget_for(n))
    push!(banded_rows, r)
    line(r)
end

println("\nRandom: the factor fills in far past the reduced matrix.\n")
header()
random_rows = NamedTuple[]
for n in SIZES, d in DENSITIES
    r = measure("random d=$d", random_qp(n, d), budget_for(n))
    push!(random_rows, r)
    line(r)
end

mismatched = count(!(r.iter_ok) for r in vcat(banded_rows, random_rows))
iszero(mismatched) ||
    println("\n$mismatched of $(length(banded_rows) + length(random_rows)) problems had backends disagree on the iteration count; their rows compare different runs.")

serialize(r) = Dict(
    "name" => r.name, "n" => r.n,
    "nnz_L" => r.nnz_L, "nnz_R" => r.nnz_R,
    "fill_L" => r.fill_L, "fill_R" => r.fill_R,
    "iter_ok" => r.iter_ok,
    "iters" => Dict(string(k) => v for (k, v) in r.iters),
    "statuses" => Dict(string(k) => v for (k, v) in r.statuses),
    "backends" => Dict(string(k) => v for (k, v) in r.backends),
    "median_seconds" => Dict(string(k) => v for (k, v) in r.times),
    "samples_seconds" => Dict(string(k) => v for (k, v) in r.samples),
)

open(joinpath(@__DIR__, "results", "gate_crossover_fill.json"), "w") do io
    JSON.print(
        io, Dict(
            "julia_version" => string(VERSION),
            "blas_threads" => BLAS.get_num_threads(),
            "settings" => Dict(string(k) => v for (k, v) in pairs(OPTS)),
            "dense_factor_fill" => Ext.DENSE_FACTOR_FILL,
            "banded" => serialize.(banded_rows),
            "random" => serialize.(random_rows),
        ), 2
    )
end
println("\nsaved bench/results/gate_crossover_fill.json")

end
