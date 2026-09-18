using Documenter, DocumenterVitepress, PureOSQP, PureIPM, PureDAQP, PureQPBase

"""
    Mermaid()

Render ```` ```mermaid ```` blocks as diagrams.

Vitepress highlights that language but draws nothing without a renderer, so a diagram written
without this ships as a syntax-coloured code block and no build step complains. This adds the
renderer through DocumenterVitepress's plugin hooks rather than by committing a
`.vitepress/config.mts`: that file is generated on every build and carries the navigation
derived from `pages` below, so owning a copy would freeze the navigation and have to track
the template's changes by hand.
"""
struct Mermaid <: Documenter.Plugin end

DocumenterVitepress.vitepress_dependencies(::Mermaid) = Dict(
    "mermaid" => "^11.4.1",
    "vitepress-plugin-mermaid" => "^2.0.17",
)

# Keyed off the import and the export rather than surrounding whitespace, so a change to the
# template leaves this working or fails loudly rather than silently matching nothing.
function DocumenterVitepress.vitepress_config_transform(::Mermaid, config::String)
    occursin("withMermaid", config) && return config
    marker = "export default defineConfig({"
    occursin(marker, config) ||
        error("Mermaid: the Vitepress config no longer has the `$marker` this keys off")
    # Naming the config and re-exporting it wraps the call without having to balance the
    # parentheses of a block this does not otherwise read.
    out = replace(
        config,
        "import { defineConfig } from 'vitepress'" =>
            "import { defineConfig } from 'vitepress'\nimport { withMermaid } from 'vitepress-plugin-mermaid'",
    )
    out = replace(out, marker => "const documenterConfig = defineConfig({")
    return out * "\nexport default withMermaid(documenterConfig)\n"
end

makedocs(;
    plugins = [Mermaid()],
    modules = [PureOSQP, PureIPM, PureDAQP, PureQPBase],
    authors = "el-oso",
    sitename = "PureQP.jl",
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/el-oso/PureQP.jl",
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
            "Choosing an algorithm" => "algorithms.md",
            "Matrix types" => "matrices.md",
            "Structured operators" => "operators.md",
            "Operators from functions" => "linearmaps.md",
            "Other packages" => "ecosystem.md",
        ],
        "Reference" => [
            "API" => "api.md",
            "Algorithm" => "algorithm.md",
            "Backend selection" => "selection.md",
            "Interfaces" => "interfaces.md",
            "Adding an algorithm" => "newalgorithm.md",
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
    repo = "github.com/el-oso/PureQP.jl",
    target = joinpath(@__DIR__, "build"),
    branch = "gh-pages",
    devbranch = "main",
    push_preview = true,
)
