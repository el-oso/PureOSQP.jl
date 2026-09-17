# One index over the benchmark caches. Each package's benchmarks write their samples into
# that package's own `bench/results`; this reads all of them and writes a single
# `bench/results/index.json`, which `docs/src/benchmarks.md` renders.
#
#     julia --project=bench bench/consolidate.jl
#
# It reads the caches and never runs a benchmark, so it is cheap and its answer does not
# depend on the machine it runs on.
using JSON, Printf

const ROOT = dirname(@__DIR__)
const INDEX = joinpath(@__DIR__, "results", "index.json")

"""
Where the benchmarks and their caches live. `shared` holds what neither package owns alone:
the problem generators, the snapshot gate and the StrictMode audit.
"""
const SOURCES = (
    (package = "PureOSQP", dir = joinpath(ROOT, "PureOSQP", "bench")),
    (package = "PureIPM", dir = joinpath(ROOT, "PureIPM", "bench")),
    (package = "shared", dir = @__DIR__),
)

"The value the file records under `key`, and `nothing` when it records none."
meta(data, key) = data isa AbstractDict && haskey(data, key) ? data[key] : nothing

"""
    describe(package, dir, file) -> Dict

One record: which package measured it, which script writes it, and what the run was. The
script name is the cache name with a `.jl` extension, which is the convention every
benchmark here follows; a cache with no such script is reported rather than assumed stale,
since a script may write more than one file.
"""
function describe(package, dir, file)
    name = first(splitext(file))
    # A measured `Inf` or `NaN` — an unbounded objective, a solver that diverged — is written
    # through as a bare literal, which strict JSON has no spelling for.
    data = JSON.parsefile(joinpath(dir, "results", file); allownan = true)
    script = name * ".jl"
    return Dict(
        "package" => package,
        "name" => name,
        "script" => isfile(joinpath(dir, script)) ? script : nothing,
        "results" => relpath(joinpath(dir, "results", file), ROOT),
        "julia_version" => meta(data, "julia_version"),
        "blas_threads" => meta(data, "blas_threads"),
        "keys" => data isa AbstractDict ? sort(string.(collect(keys(data)))) : String[],
        "bytes" => filesize(joinpath(dir, "results", file)),
    )
end

entries = Dict{String, Any}[]
unrun = Dict{String, Any}[]
for (package, dir) in SOURCES
    resdir = joinpath(dir, "results")
    isdir(resdir) || continue
    for file in sort(filter(f -> endswith(f, ".json") && f != "index.json", readdir(resdir)))
        push!(entries, describe(package, dir, file))
    end
    # A benchmark that has never been run leaves no cache, so the docs have nothing to show
    # for it. Naming it here is the only way that stays visible. A script that saves no
    # samples is a problem generator or a gate rather than a benchmark, and is not expected
    # to leave one.
    for script in sort(filter(f -> endswith(f, ".jl"), readdir(dir)))
        script == basename(@__FILE__) && continue
        base = first(splitext(script))
        occursin("\"results\"", read(joinpath(dir, script), String)) || continue
        any(e -> e["package"] == package && startswith(e["name"], base), entries) && continue
        push!(unrun, Dict("package" => package, "script" => script))
    end
end

orphans = [e for e in entries if isnothing(e["script"])]

open(INDEX, "w") do io
    JSON.print(
        io, Dict(
            "entries" => entries,
            "unrun" => unrun,
            "packages" => [s.package for s in SOURCES],
        ), 2
    )
end

@printf("%-10s %-34s %-10s %8s\n", "package", "cache", "julia", "bytes")
println("-"^66)
for e in entries
    @printf(
        "%-10s %-34s %-10s %8d\n",
        e["package"], e["name"], something(e["julia_version"], "-"), e["bytes"]
    )
end
@printf("\n%d caches over %d packages\n", length(entries), length(SOURCES))
isempty(orphans) ||
    println("no script writes: ", join((e["name"] for e in orphans), ", "))
isempty(unrun) ||
    println("never run: ", join((string(e["package"], "/", e["script"]) for e in unrun), ", "))
println("\nwrote ", relpath(INDEX, ROOT))
