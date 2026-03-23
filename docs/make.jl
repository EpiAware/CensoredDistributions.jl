using Pkg: Pkg
Pkg.instantiate()

using DocumenterVitepress
using Documenter
using DocumenterCitations
using CensoredDistributions

# Check for skip notebooks option
skip_notebooks = "--skip-notebooks" in ARGS ||
                 get(ENV, "SKIP_NOTEBOOKS", "false") == "true"

include("pages.jl")

if !skip_notebooks
    using Literate

    tutorials_dir = joinpath(
        @__DIR__, "src", "getting-started", "tutorials"
    )
    tutorial_files = [
        "analytical-primarycensored-cdfs.jl",
        "exponentially-tilted-primary-events.jl",
        "fitting-with-turing.jl"
    ]

    println(
        "Building Literate tutorials " *
        "(this may take several minutes)..."
    )
    for file in tutorial_files
        Literate.markdown(
            joinpath(tutorials_dir, file),
            tutorials_dir;
            flavor = Literate.DocumenterFlavor(),
            mdstrings = true,
            credit = false
        )
    end
    println("Literate tutorial processing complete")
else
    println(
        "Skipping Literate tutorial processing " *
        "(--skip-notebooks or SKIP_NOTEBOOKS=true)"
    )
end

# Generate index.md from README.md
open(joinpath(joinpath(@__DIR__, "src"), "index.md"), "w") do io
    println(io, "```@meta")
    println(io,
        "EditURL = " *
        "\"https://github.com/EpiAware/" *
        "CensoredDistributions.jl/blob/main/" *
        "README.md\"")
    println(io, "```")

    for line in eachline(
        joinpath(dirname(@__DIR__), "README.md")
    )
        # Replace ```julia with ```@example readme
        if startswith(line, "```julia")
            println(io, "```@example readme")
            # Remove logo from title line for docs
        elseif contains(line, "docs/src/assets/logo.svg")
            println(io, replace(line,
                r"\s*<img[^>]*docs/src/assets/logo\.svg[^>]*>" => ""))
            # Skip badge table and Websites line
        elseif startswith(line, "|")  # Table rows
            continue
        elseif startswith(line, "**Websites**")
            continue
        else
            # Convert absolute doc URLs to @ref links
            # so links stay within the current version
            line = replace(line,
                "[Getting Started documentation](https://censoreddistributions.epiaware.org/stable/getting-started/)" => "[Getting Started documentation](@ref getting-started)",
                "[Getting Started Tutorials](https://censoreddistributions.epiaware.org/stable/getting-started/)" => "[Getting Started Tutorials](@ref getting-started)",
                "[API Reference](https://censoreddistributions.epiaware.org/stable/lib/public/)" => "[API Reference](@ref public-api)",
                "[Developer Documentation](https://censoreddistributions.epiaware.org/stable/developer/)" => "[Developer Documentation](@ref developer)",
                "[developer documentation](https://censoreddistributions.epiaware.org/stable/developer/)" => "[developer documentation](@ref developer)")
            println(io, line)
        end
    end
end

# Generate release-notes.md by combining header with NEWS.md
include("release_notes_header.jl")

news_src = joinpath(dirname(@__DIR__), "NEWS.md")
release_notes_dest = joinpath(
    joinpath(@__DIR__, "src"), "release-notes.md"
)

if isfile(news_src)
    open(release_notes_dest, "w") do io
        # Write the header content
        print(io, RELEASE_NOTES_HEADER)

        # Append the NEWS.md content
        for line in eachline(news_src)
            println(io, line)
        end
    end
    println("Generated release-notes.md from header + NEWS.md")
else
    println("NEWS.md not found in project root")
end

DocMeta.setdocmeta!(CensoredDistributions, :DocTestSetup,
    :(using CensoredDistributions); recursive = true)

# Set up citations
bib = CitationBibliography(
    joinpath(@__DIR__, "src", "refs.bib");
    style = :numeric
)

makedocs(; sitename = "CensoredDistributions.jl",
    authors = "Sam Abbott, and contributors",
    clean = true, doctest = false, linkcheck = true,
    warnonly = [
        :docs_block, :missing_docs,
        :linkcheck, :autodocs_block
    ],
    modules = [CensoredDistributions],
    pages = pages,
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/EpiAware/" *
               "CensoredDistributions.jl",
        devbranch = "main",
        devurl = "dev",
        deploy_url = "censoreddistributions.epiaware.org"
    ),
    plugins = [bib]
)

DocumenterVitepress.deploydocs(
    repo = "github.com/EpiAware/CensoredDistributions.jl",
    target = "build",
    branch = "gh-pages",
    devbranch = "main",
    push_preview = true
)
