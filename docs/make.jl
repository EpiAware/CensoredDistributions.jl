using Pkg: Pkg
Pkg.instantiate()

using Documenter
using CensoredDistributions

# Check for skip notebooks option
skip_notebooks = "--skip-notebooks" in ARGS ||
                 get(ENV, "SKIP_NOTEBOOKS", "false") == "true"

if !skip_notebooks
    using Pluto: Configuration.CompilerOptions
    using PlutoStaticHTML

    include("pages.jl")
    include("build.jl")

    println("Building Pluto notebooks (this may take several minutes)...")
    build("getting-started")
    build("getting-started/tutorials")
    println("✓ Notebook processing complete")
else
    println("⚠ Skipping Pluto notebook processing (--skip-notebooks or SKIP_NOTEBOOKS=true)")
    include("pages.jl")
end

# Generate index.md from README.md
open(joinpath(joinpath(@__DIR__, "src"), "index.md"), "w") do io
    println(io, "```@meta")
    println(io,
        "EditURL = \"https://github.com/EpiAware/CensoredDistributions.jl/blob/main/README.md\"")
    println(io, "```")

    for line in eachline(joinpath(dirname(@__DIR__), "README.md"))
        # Replace ```julia with ```@example readme
        if startswith(line, "```julia")
            println(io, "```@example readme")
        else
            println(io, line)
        end
    end
end

# Generate release-notes.md by combining header with NEWS.md
include("release_notes_header.jl")

news_src = joinpath(dirname(@__DIR__), "NEWS.md")
release_notes_dest = joinpath(joinpath(@__DIR__, "src"), "release-notes.md")

if isfile(news_src)
    open(release_notes_dest, "w") do io
        # Write the header content
        print(io, RELEASE_NOTES_HEADER)

        # Append the NEWS.md content
        for line in eachline(news_src)
            println(io, line)
        end
    end
    println("✓ Generated release-notes.md from header + NEWS.md")
else
    println("⚠ NEWS.md not found in project root")
end

DocMeta.setdocmeta!(
    CensoredDistributions, :DocTestSetup, :(using CensoredDistributions); recursive = true)

makedocs(; sitename = "CensoredDistributions.jl",
    authors = "Sam Abbott, and contributors",
    clean = true, doctest = false, linkcheck = true,
    warnonly = [:docs_block, :missing_docs, :linkcheck, :autodocs_block],
    modules = [CensoredDistributions],
    pages = pages,
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        mathengine = Documenter.MathJax3(),
        size_threshold = 6000 * 2^10,
        size_threshold_warn = 2000 * 2^10
    )
)

deploydocs(
    repo = "github.com/EpiAware/CensoredDistributions.jl.git",
    target = "build",
    push_preview = true
)
