# What the reduced diagonal costs when `A` declares its structure.
#
# `reduced_diagonal!` builds the matrix-free backend's Jacobi preconditioner column by
# column, and the inner sum runs over the rows `A` can hold a nonzero in. Summing every row
# instead makes a `Diagonal`, `Bidiagonal` or `RowCoupled` `A` cost `O(mn)` where the matrix
# holds `O(n)` or `O(kn)` entries. `refactor_rho!` defaults to `factorize!`, so the cost is
# paid again at every rho adaptation rather than once at setup, which is why the end-to-end
# sweep below leaves adaptation at its default.
#
# The kernel sweep carries both states in a single run: `full_walk!` is the every-row form,
# asserted to agree bit for bit with the shipped function before anything is timed, so the
# two differ only in which rows they visit. The end-to-end sweep cannot do that -- it goes
# through `solve` -- so each row is labelled by the row set the shipped function actually
# walks, and running the file against both states of `src/scaling.jl` leaves both in the
# results file.
#
# `Diagonal` and `Bidiagonal` are square, so `m = 2n` exists only for `RowCoupled`.
#
# Swept at two BLAS thread counts: `src/` and `ext/` pin none, so the deployed solver runs at
# whatever the caller has.
using PureOSQP, Krylov, LinearAlgebra, Chairmarks, Printf, JSON, Statistics, Random

const SIZES = (500, 1000, 2000)
const THREADS = (1, min(8, Sys.CPU_THREADS))
const RESULTS = joinpath(@__DIR__, "results", "reduced_diagonal_structural.json")

"The reduced diagonal with the inner sum over every row of `A`, whatever `A` can hold."
function full_walk!(dest, ::Type{T}, P, A, rho, E, D, sigma, c) where {T}
    for j in eachindex(dest)
        dj = D[j]
        d = c * dj * T(P[j, j]) * dj + sigma
        for i in eachindex(rho, E)
            a = E[i] * T(A[i, j]) * dj
            d += rho[i] * a * a
        end
        dest[j] = inv(max(d, sqrt(eps(T))))
    end
    return dest
end

"A matrix that counts entry reads and forwards its structure to what it wraps."
struct Probe{T, M <: AbstractMatrix{T}} <: AbstractMatrix{T}
    parent::M
    reads::Base.RefValue{Int}
end
Probe(m::AbstractMatrix) = Probe(m, Ref(0))
Base.size(p::Probe) = size(p.parent)
Base.getindex(p::Probe, i::Int, j::Int) = (p.reads[] += 1; p.parent[i, j])
PureOSQP.structural_rows(p::Probe, j::Integer) = PureOSQP.structural_rows(p.parent, j)

"""
    dispatch_label() -> String

Which row set `reduced_diagonal!` walks, read off the shipped function rather than assumed:
`n` reads over a `Diagonal` `A` of order `n` means the structural rows, `n^2` means all of
them.
"""
function dispatch_label()
    n = 8
    A = Probe(Diagonal(ones(n)))
    PureOSQP.reduced_diagonal!(zeros(n), Float64, Diagonal(ones(n)), A, ones(n), ones(n), ones(n), 1.0e-6, 1.0)
    return A.reads[] == n ? "structural" : "full"
end

const LABEL = dispatch_label()

times(x) = [s.time for s in x.samples]

"The median of `f`'s samples, warmed once so no compilation lands in the timings."
function median_time(f, seconds = 1.0)
    f()
    return median(times(@be f() seconds = seconds))
end

"""
    constraints(kind, n, rows) -> A

The constraint matrix of the named structure with `rows` rows over `n` columns. `RowCoupled`
carries three dense coupling rows above single-entry ones, whose columns repeat when there
are more rows than columns.
"""
function constraints(kind, n, rows)
    rng = MersenneTwister(9377 + n)
    kind === :diagonal && return Diagonal(rand(rng, n) .+ 0.5)
    kind === :bidiagonal && return Bidiagonal(rand(rng, n) .+ 0.5, rand(rng, n - 1) ./ 4, :U)
    k = 3
    cols = [1 + (r - 1) % n for r in 1:(rows - k)]
    return PureOSQP.RowCoupled(randn(rng, k, n) ./ (4n), ones(rows - k), cols)
end

"The problem the end-to-end sweep solves: a diagonal cost under `A`'s constraints."
function problem(kind, n, rows)
    rng = MersenneTwister(4441 + n)
    A = constraints(kind, n, rows)
    return Diagonal(rand(rng, n) .+ 0.5), randn(rng, n), A, -rand(rng, rows) .- 0.5, rand(rng, rows) .+ 0.5
end

"Row shapes each structure admits: the square ones cannot be given more rows than columns."
shapes(kind, n) = kind === :rowcoupled ? (n, 2n) : (n,)

function kernel_rows(nthreads)
    rows = Dict{String, Any}[]
    for kind in (:diagonal, :bidiagonal, :rowcoupled), n in SIZES, m in shapes(kind, n)
        rng = MersenneTwister(1201 + n)
        P, A = Diagonal(rand(rng, n) .+ 0.5), constraints(kind, n, m)
        rho, E, D = rand(rng, m) .+ 0.1, rand(rng, m) .+ 0.5, rand(rng, n) .+ 0.5
        args = (Float64, P, A, rho, E, D, 1.0e-6, 1.3)
        # The two forms must agree exactly, or the comparison below is between two different
        # computations rather than two ways of reaching the same one.
        @assert PureOSQP.reduced_diagonal!(zeros(n), args...) == full_walk!(zeros(n), args...)

        dest = zeros(n)
        after = median_time(() -> PureOSQP.reduced_diagonal!(dest, args...))
        before = median_time(() -> full_walk!(dest, args...))
        push!(
            rows, Dict{String, Any}(
                "sweep" => "kernel", "structure" => String(kind), "blas_threads" => nthreads,
                "n" => n, "m" => m, "label" => LABEL,
                "full_walk_s" => before, "structural_s" => after,
            )
        )
        @printf(
            "kernel  threads=%d %-11s n=%5d m=%5d  full %9.3f us  structural %9.3f us  %6.1fx\n",
            nthreads, kind, n, m, 1.0e6before, 1.0e6after, before / after
        )
        flush(stdout)
    end
    return rows
end

function solve_rows(nthreads)
    rows = Dict{String, Any}[]
    for kind in (:diagonal, :bidiagonal, :rowcoupled), n in SIZES, m in shapes(kind, n)
        P, q, A, l, u = problem(kind, n, m)
        opts = (linsys = :indirect, eps_abs = 1.0e-8, eps_rel = 1.0e-8)
        res = PureOSQP.solve(P, q, A, l, u; opts...)
        # One rebuild at setup, and one more at every rho adaptation the run took.
        rebuilds = 1 + res.iter ÷ 50
        t = median_time(() -> PureOSQP.solve(P, q, A, l, u; opts...), 2.0)
        push!(
            rows, Dict{String, Any}(
                "sweep" => "solve", "structure" => String(kind), "blas_threads" => nthreads,
                "n" => n, "m" => m, "label" => LABEL, "solve_s" => t,
                "iterations" => res.iter, "rebuilds" => rebuilds,
                "status" => String(Symbol(res.status)),
            )
        )
        @printf(
            "solve   threads=%d %-11s n=%5d m=%5d  %-10s %9.3f ms  iter %5d  rebuilds %3d  %s\n",
            nthreads, kind, n, m, LABEL, 1.0e3t, res.iter, rebuilds, res.status
        )
        flush(stdout)
    end
    return rows
end

function main()
    rows = Dict{String, Any}[]
    for nthreads in THREADS
        BLAS.set_num_threads(nthreads)
        append!(rows, kernel_rows(nthreads))
        append!(rows, solve_rows(nthreads))
    end
    old = isfile(RESULTS) ? JSON.parsefile(RESULTS)["cases"] : Any[]
    kept = filter(r -> get(r, "label", "") != LABEL, old)
    open(RESULTS, "w") do io
        JSON.print(
            io, Dict(
                "julia_version" => string(VERSION),
                "cpu_threads" => Sys.CPU_THREADS,
                "sizes" => collect(SIZES),
                "cases" => vcat(kept, rows),
            ), 2
        )
    end
    return println("wrote $RESULTS")
end

main()
