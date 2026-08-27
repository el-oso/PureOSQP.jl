using Documenter, DocumenterVitepress, PureOSQP

makedocs(;
    modules = [PureOSQP],
    authors = "el-oso",
    sitename = "PureOSQP.jl",
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
