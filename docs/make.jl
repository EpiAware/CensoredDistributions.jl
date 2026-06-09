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

    # Light tutorials: Literate emits `@example` blocks that Documenter runs
    # in-process. They are cheap and accumulate no native/memory state.
    light_tutorials = [
        "analytical-primarycensored-cdfs.jl",
        "exponentially-tilted-primary-events.jl",
        "composer-toolkit.jl",
        "fit-marginal-sample-event-based.jl"
    ]

    # Heavy tutorials: live MCMC fits or a multi-backend AD benchmark, plus
    # CairoMakie / AlgebraOfGraphics / PairPlots. Run each in its own subprocess
    # with `execute = true` so the captured outputs become static ````julia````
    # blocks; Documenter then renders without re-executing, and no native or
    # memory state accumulates across tutorials in the long-lived Documenter
    # process. `ad-backends` runs AD over many backends and Makie plots, so it
    # is isolated here too.
    heavy_tutorials = [
        "ad-backends.jl",
        "fitting-with-turing.jl",
        "bdbv-linelist-analysis.jl",
        "andv-linelist-analysis.jl"
    ]

    println(
        "Building light Literate tutorials " *
        "(this may take several minutes)..."
    )
    for file in light_tutorials
        Literate.markdown(
            joinpath(tutorials_dir, file),
            tutorials_dir;
            flavor = Literate.DocumenterFlavor(),
            mdstrings = true,
            credit = false
        )
    end

    println(
        "Executing heavy Literate tutorials, one per subprocess..."
    )
    runner = joinpath(@__DIR__, "run_literate_tutorial.jl")
    for file in heavy_tutorials
        input = joinpath(tutorials_dir, file)
        println("  executing $file in a fresh subprocess...")
        run(`$(Base.julia_cmd()) --project=$(@__DIR__) $runner $input $tutorials_dir`)
    end
    println("Literate tutorial processing complete")
else
    println(
        "Skipping Literate tutorial processing " *
        "(--skip-notebooks or SKIP_NOTEBOOKS=true)"
    )
    # A fast build skips the heavy Literate + `@example` execution, but the
    # tutorial pages are still referenced by the nav and linked from other pages.
    # Write a lightweight stub `.md` for each so the nav resolves and the rest of
    # the site builds; a full build overwrites these with the rendered tutorials.
    tutorials_dir = joinpath(
        @__DIR__, "src", "getting-started", "tutorials"
    )
    # Each stub heading preserves the cross-reference `@id` the full tutorial
    # defines, so `@ref`s from other pages (e.g. the FAQ's `@ref ad-backends`)
    # still resolve in a fast build.
    tutorial_stubs = [
        "analytical-primarycensored-cdfs.md" => "# Analytical CDF solutions",
        "exponentially-tilted-primary-events.md" => "# Exponentially tilted primary events",
        "ad-backends.md" => "# [Automatic differentiation backends](@id ad-backends)",
        "fitting-with-turing.md" => "# Fitting with Turing.jl",
        "composer-toolkit.md" => "# [The composer toolkit](@id composer-toolkit)",
        "fit-marginal-sample-event-based.md" => "# Fit marginal, sample event based",
        "bdbv-linelist-analysis.md" => "# Bundibugyo Ebola delays from the 2012 Isiro line list",
        "andv-linelist-analysis.md" => "# Real-time Andes virus delays from the Epuyén line list"
    ]
    for (file, heading) in tutorial_stubs
        open(joinpath(tutorials_dir, file), "w") do io
            println(io, heading)
            println(io)
            println(io,
                "_This tutorial is omitted from the fast documentation " *
                "build. Build the full documentation (`task docs`) to render " *
                "it._")
        end
    end
    println("Wrote fast-build tutorial stubs")
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
                "[API Reference](https://censoreddistributions.epiaware.org/stable/lib/public)" => "[API Reference](@ref public-api)",
                "[Developer Documentation](https://censoreddistributions.epiaware.org/stable/developer/)" => "[Developer Documentation](@ref developer)",
                "[developer documentation](https://censoreddistributions.epiaware.org/stable/developer/)" => "[developer documentation](@ref developer)",
                "[Automatic differentiation backends](https://censoreddistributions.epiaware.org/stable/getting-started/tutorials/ad-backends/)" => "[Automatic differentiation backends](@ref ad-backends)")
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
    # A fast build skips the network linkcheck (rate-limited, irrelevant to a
    # local content build); a full build keeps it strict.
    clean = true, doctest = false, linkcheck = !skip_notebooks,
    warnonly = [
        :docs_block, :missing_docs,
        :autodocs_block
    ],
    modules = [CensoredDistributions],
    pages = pages,
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/EpiAware/" *
               "CensoredDistributions.jl",
        devbranch = "main",
        devurl = "dev",
        deploy_url = "censoreddistributions.epiaware.org",
        keep = :patch
    ),
    plugins = [bib]
)

# Copy every tutorial data directory into the matching build output dir so the
# bundled data ships with the rendered site (and `@example` blocks that read it
# resolve at view time). Runs after `makedocs` so `clean = true` does not wipe it;
# generic over any tutorial that carries a `data` or `<name>-data` dir.
let src_root = joinpath(@__DIR__, "src"), build_root = joinpath(@__DIR__, "build")
    for (root, dirs, _) in walkdir(src_root)
        for d in dirs
            (d == "data" || endswith(d, "-data")) || continue
            src_data = joinpath(root, d)
            rel = relpath(src_data, src_root)
            dest_data = joinpath(build_root, rel)
            mkpath(dirname(dest_data))
            cp(src_data, dest_data; force = true)
            println("Copied tutorial data: $rel")
        end
    end
end

DocumenterVitepress.deploydocs(
    repo = "github.com/EpiAware/CensoredDistributions.jl",
    target = "build",
    branch = "gh-pages",
    devbranch = "main",
    push_preview = true
)
