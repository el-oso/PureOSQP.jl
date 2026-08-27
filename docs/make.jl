using Documenter, DocumenterVitepress, PureOSQP
using Documenter: Remotes

makedocs(;
    modules = [PureOSQP],
    authors = "el-oso",
    sitename = "PureOSQP.jl",
    # Stated rather than inferred from git, so the docs build in a bare checkout too
    # while keeping per-source links working.
    repo = Remotes.GitHub("el-oso", "PureOSQP.jl"),
    remotes = Dict(dirname(@__DIR__) => Remotes.GitHub("el-oso", "PureOSQP.jl")),
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/el-oso/PureOSQP.jl",
        devbranch = "main",
        devurl = "dev",
    ),
    pages = [
        "Home" => "index.md",
        "Examples" => "examples.md",
        "Algorithm" => "algorithm.md",
        "Benchmarks" => "benchmarks.md",
        "API" => "api.md",
    ],
    # Not a blanket `true`: a failing @example block must fail the build, since the
    # examples page is the only thing checking that the documented code still runs.
    warnonly = [:missing_docs, :cross_references, :docs_block],
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/el-oso/PureOSQP.jl",
    target = joinpath(@__DIR__, "build"),
    branch = "gh-pages",
    devbranch = "main",
    push_preview = true,
)
