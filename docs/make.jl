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
        "Structured operators" => "operators.md",
        "Other packages" => "ecosystem.md",
        "Guarantees" => "guarantees.md",
        "Benchmarks" => "benchmarks.md",
        "API" => "api.md",
        "Roadmap" => "roadmap.md",
        "Attribution" => "attribution.md",
    ],
    # Not a blanket `true`: a failing @example block must fail the build, since the
    # examples page is the only thing checking that the documented code still runs.
    #
    # `:cross_references` is deliberately absent. A dead `@ref` is caught here, where the
    # error names the offending link; downgraded to a warning it survives to the Vitepress
    # stage, which reports only "1 dead link(s) found" and cannot run in every environment.
    warnonly = [:missing_docs, :docs_block],
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/el-oso/PureOSQP.jl",
    target = joinpath(@__DIR__, "build"),
    branch = "gh-pages",
    devbranch = "main",
    push_preview = true,
)
