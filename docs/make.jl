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
        "survival-delay-families.jl",
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
        "andv-linelist-analysis.jl",
        "ebola-stratified-delays.jl",
        "rt-renewal-convolution.jl",
        "epinowcast-nowcasting.jl",
        "linear-chain-sir.jl",
        "branching-one_of.jl",
        "pairwise-survival-transmission.jl"
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

    # Heavy tutorials sample several MCMC chains with `MCMCThreads()`, so each
    # subprocess needs more than one thread to run them in parallel. The parent
    # docs process is usually single-threaded (Documenter sets no thread count),
    # and `Base.julia_cmd()` would propagate that single thread to the child, so
    # the chains would run serially. Read the requested count from
    # `JULIA_NUM_THREADS` (default 4) and pass it explicitly to each subprocess.
    tutorial_threads = get(ENV, "JULIA_NUM_THREADS", "4")
    println(
        "Executing heavy Literate tutorials, one per subprocess " *
        "($(tutorial_threads) threads each)..."
    )
    runner = joinpath(@__DIR__, "run_literate_tutorial.jl")
    for file in heavy_tutorials
        input = joinpath(tutorials_dir, file)
        println("  executing $file in a fresh subprocess...")
        run(`$(Base.julia_cmd()) --threads=$(tutorial_threads) --project=$(@__DIR__) $runner $input $tutorials_dir`)
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
        "composer-toolkit.md" => "# [Composing censored distributions](@id composer-toolkit)",
        "survival-delay-families.md" => "# [Delay families from SurvivalDistributions.jl](@id survival-delay-families)",
        "fit-marginal-sample-event-based.md" => "# [Fit marginal, sample event based](@id fit-marginal-sample-event-based)",
        "bdbv-linelist-analysis.md" => "# [Bundibugyo Ebola delays from the 2012 Isiro line list](@id bdbv-linelist-analysis)",
        "andv-linelist-analysis.md" => "# [Real-time Andes virus delays from the Epuyén line list](@id andv-linelist-analysis)",
        "ebola-stratified-delays.md" => "# [Stratified onset-to-test delays in the 2014-2016 Sierra Leone Ebola outbreak](@id ebola-stratified-delays)",
        "rt-renewal-convolution.md" => "# [An Rt renewal model with delay convolution](@id rt-renewal-convolution)",
        "linear-chain-sir.md" => "# [A composed delay as ODE compartments: the linear chain trick](@id linear-chain-sir)",
        "epinowcast-nowcasting.md" => "# [An epinowcast-style hazard nowcasting model](@id epinowcast-nowcasting)",
        "branching-one_of.md" => "# [A branching-process-like natural history with one_of outcomes](@id branching-one_of)",
        "pairwise-survival-transmission.md" => "# [Pairwise survival analysis of transmission (Kenah)](@id pairwise-survival-transmission)"
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

# Generate the API reference pages (lib/public.md, lib/internals.md) from the
# module's documented bindings. `@autodocs` splices ONE docstring block per
# documented METHOD SIGNATURE, so a function with several `@doc`-annotated
# methods (e.g. `primary_censored`) appears many times in both the rendered API
# and the `@index` (issue #184). Instead, each binding is listed ONCE in a
# `@docs` block: Documenter then combines all of a binding's method docstrings
# under a single heading with a single `@index` entry, while still showing every
# docstring. The binding list is derived from the module at build time, so it
# composes with whatever names happen to be exported.

# Whether `sym` is part of `mod`'s public API, matching how Documenter's
# `@autodocs` partitions `Public`/`Private` (`Base.ispublic` on >= 1.11, else
# exported). Note the check is against `mod`, not the binding's defining module,
# so a docstring CensoredDistributions attaches to an extended foreign function
# (e.g. `Base.show`, `Distributions.logpdf`) is internal unless re-exported.
function _is_public(mod::Module, sym::Symbol)
    return @static if isdefined(Base, :ispublic)
        Base.ispublic(mod, sym)
    else
        Base.isexported(mod, sym)
    end
end

function api_bindings(mod::Module)
    # The keys of `Docs.meta(mod)` are every binding `mod` attaches a docstring
    # to, including extended functions owned by other modules. This is the same
    # set `@autodocs Modules = [mod]` walks, so listing each binding once here
    # (rather than once per method signature) keeps every docstring while
    # collapsing the index to one entry per function. Derived from the module at
    # build time, so it composes with whatever names are exported.
    meta = Base.Docs.meta(mod)
    vars = sort!([b.var for b in keys(meta)]; by = string)
    public = Symbol[]
    private = Symbol[]
    for v in vars
        v === nameof(mod) && continue  # skip the module's own docstring
        push!(_is_public(mod, v) ? public : private, v)
    end
    return public, private
end

function write_api_page(path, title, anchor, page, intro, api_heading, mod, names)
    # `docs/src/lib/` holds only generated pages (gitignored), so it is absent
    # on a fresh checkout (CI) — create it before writing.
    mkpath(dirname(path))
    open(path, "w") do io
        if anchor === nothing
            println(io, "# $title")
        else
            println(io, "# [$title](@id $anchor)")
        end
        println(io)
        println(io, intro)
        println(io)
        println(io, "## Contents")
        println(io)
        println(io, "```@contents")
        println(io, "Pages = [\"$page\"]")
        println(io, "Depth = 2:2")
        println(io, "```")
        println(io)
        println(io, "## Index")
        println(io)
        println(io, "```@index")
        println(io, "Pages = [\"$page\"]")
        println(io, "```")
        println(io)
        # Section heading is a stable `@ref` target for other pages, so keep
        # the original "Public API" / "Internal API" titles.
        println(io, "## $api_heading")
        println(io)
        println(io, "```@docs")
        for name in names
            # Qualify through the package, not the binding's defining module:
            # extended foreign functions (`logpdf`, `params`, `show`, ...) are
            # imported into `mod`, so `mod.name` resolves on the docs page while
            # `Distributions.logpdf` etc. would not (those modules are not in
            # the page's scope). Documenter still splices every method docstring
            # registered for the binding.
            println(io, string(mod, ".", name))
        end
        println(io, "```")
    end
end

let (public, private) = api_bindings(CensoredDistributions)
    lib_dir = joinpath(@__DIR__, "src", "lib")
    write_api_page(
        joinpath(lib_dir, "public.md"),
        "Public Documentation", "public-api", "public.md",
        "Documentation for `CensoredDistributions.jl`'s public interface.\n\n" *
        "See the Internals section of the manual for internal package docs " *
        "covering all submodules.",
        "Public API", CensoredDistributions, public
    )
    write_api_page(
        joinpath(lib_dir, "internals.md"),
        "Internal Documentation", nothing, "internals.md",
        "Documentation for `CensoredDistributions.jl`'s internal interface.",
        "Internal API", CensoredDistributions, private
    )
    println(
        "Generated API pages: $(length(public)) public, " *
        "$(length(private)) internal bindings"
    )
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
    # The benchmark-history page is published by a separate workflow and
    # only resolves once the maintainer enables Pages for the benchmarks
    # branch, so it is intentionally excluded from linkcheck until live.
    linkcheck_ignore = [
        r"EpiAware\.github\.io/CensoredDistributions\.jl/history"
    ],
    warnonly = [
        :docs_block, :missing_docs,
        :autodocs_block,
        # Internal "see also" @refs to undocumented helper functions render as
        # plain code rather than failing the build.
        :cross_references
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
