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
    # Grouped rather than flat: every top-level entry becomes a navbar item, and ten of them
    # overflow the bar into the sidebar. Four groups keep the bar short and put each page
    # under the question it answers.
    pages = [
        "Home" => "index.md",
        "Guide" => [
            "Examples" => "examples.md",
            "Matrix types" => "matrices.md",
            "Structured operators" => "operators.md",
            "Operators from functions" => "linearmaps.md",
            "Other packages" => "ecosystem.md",
        ],
        "Reference" => [
            "API" => "api.md",
            "Algorithm" => "algorithm.md",
            "Benchmarks" => "benchmarks.md",
            "Guarantees" => "guarantees.md",
        ],
        "Project" => [
            "Roadmap" => "roadmap.md",
            "Attribution" => "attribution.md",
        ],
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
