# MANAGED by EpiAwarePackageTools.scaffold — do not edit by hand.
#
# The standard EpiAware documentation build: Documenter + DocumenterVitepress
# (the org docs standard, reproducing CensoredDistributions.jl generically). It
#
#   - runs the Literate.jl tutorial pipeline (light tutorials rendered in
#     process, heavy tutorials each executed in a fresh subprocess) driven by
#     the package-owned `docs_config.jl`, with fast-build stubs on
#     `--skip-notebooks`,
#   - generates `src/index.md` from the package README (badge block + any raw
#     badge table stripped, ```julia blocks turned into `@example readme`),
#   - generates `src/release-notes.md` from a project-root `NEWS.md` when one
#     exists, prefixed with the package-owned release-notes header,
#   - generates `src/benchmarks.md` (a managed structure linking the published
#     performance history) splicing in the package-owned `benchmarks.md` prose,
#   - generates the API reference pages (`lib/public.md`, `lib/internals.md`)
#     from the module's documented bindings (one `@docs` entry per binding so
#     the index has one entry per function, not one per method signature), and
#   - renders the site with `DocumenterVitepress.MarkdownVitepress` (adding
#     DocumenterCitations when a `src/refs.bib` exists) and deploys it with
#     `DocumenterVitepress.deploydocs`.
#
# All package-specific data (tutorial lists, link rewrites, linkcheck ignores)
# lives in the package-owned `docs_config.jl`; this managed file carries none,
# so it can be re-applied on every `update` without losing package content. An
# empty config builds a site with no tutorials and degrades gracefully when a
# package has no `NEWS.md` or `refs.bib`.
#
# Build it with `task docs` (or `julia --project=docs docs/make.jl`).

using Pkg: Pkg
Pkg.instantiate()

using DocumenterVitepress
using Documenter
using DocumenterCitations
using CensoredDistributions

# Check for skip notebooks option
skip_notebooks = "--skip-notebooks" in ARGS ||
                 get(ENV, "SKIP_NOTEBOOKS", "false") == "true"

# The docs navigation tree (package-owned).
include("pages.jl")
# Package-specific build config: tutorial lists, README/index link rewrites,
# and linkcheck ignores. Package-owned and never overwritten, so an empty
# config builds a site with no tutorials.
include("docs_config.jl")

# --- Literate tutorial pipeline -------------------------------------------
tutorials_dir = joinpath(@__DIR__, "src", TUTORIALS_SUBDIR)
has_tutorials = !isempty(LIGHT_TUTORIALS) || !isempty(HEAVY_TUTORIALS)

if !skip_notebooks
    if has_tutorials
        using Literate

        # Light tutorials: Literate emits `@example` blocks that Documenter
        # runs in-process. They are cheap and accumulate no native/memory
        # state.
        if !isempty(LIGHT_TUTORIALS)
            println(
                "Building light Literate tutorials " *
                "(this may take several minutes)..."
            )
            for file in LIGHT_TUTORIALS
                Literate.markdown(
                    joinpath(tutorials_dir, file),
                    tutorials_dir;
                    flavor = Literate.DocumenterFlavor(),
                    mdstrings = true,
                    credit = false
                )
            end
        end

        # Heavy tutorials: live MCMC fits, multi-backend AD benchmarks, or
        # plotting. Run each in its own subprocess with `execute = true` so the
        # captured outputs become static code blocks; Documenter then renders
        # without re-executing, and no native or memory state accumulates
        # across tutorials in the long-lived Documenter process.
        if !isempty(HEAVY_TUTORIALS)
            # Each heavy subprocess needs more than one thread to sample MCMC
            # chains with `MCMCThreads()` in parallel. The parent docs process
            # is usually single-threaded and `Base.julia_cmd()` would
            # propagate that, so read the requested count from
            # `JULIA_NUM_THREADS` (default 4) and pass it explicitly.
            tutorial_threads = get(ENV, "JULIA_NUM_THREADS", "4")
            println(
                "Executing heavy Literate tutorials, one per subprocess " *
                "($(tutorial_threads) threads each)..."
            )
            runner = joinpath(@__DIR__, "run_literate_tutorial.jl")
            jl = Base.julia_cmd()
            for file in HEAVY_TUTORIALS
                input = joinpath(tutorials_dir, file)
                println("  executing $file in a fresh subprocess...")
                opts = `--threads=$(tutorial_threads) --project=$(@__DIR__)`
                run(`$jl $opts $runner $input $tutorials_dir`)
            end
        end
        println("Literate tutorial processing complete")
    end
else
    println(
        "Skipping Literate tutorial processing " *
        "(--skip-notebooks or SKIP_NOTEBOOKS=true)"
    )
    # A fast build skips the heavy Literate + `@example` execution, but the
    # tutorial pages are still referenced by the nav and linked from other
    # pages. Write a lightweight stub `.md` for each so the nav resolves and
    # the rest of the site builds; a full build overwrites these with the
    # rendered tutorials. Each stub heading preserves the cross-reference `@id`
    # the full tutorial defines, so `@ref`s from other pages still resolve.
    if !isempty(TUTORIAL_STUBS)
        mkpath(tutorials_dir)
        for (file, heading) in TUTORIAL_STUBS
            open(joinpath(tutorials_dir, file), "w") do io
                println(io, heading)
                println(io)
                println(io,
                    "_This tutorial is omitted from the fast documentation " *
                    "build. Build the full documentation (`task docs`) to " *
                    "render it._")
            end
        end
        println("Wrote fast-build tutorial stubs")
    end
end

# --- README -> index.md ----------------------------------------------------
# Strip the managed badge block (the markers and everything between them), drop
# an inline logo from the title, and apply package-specific link rewrites from
# `INDEX_REWRITES` (e.g. absolute doc URLs to in-site `@ref`s) so links stay
# within the built version. Two behaviours are config-gated:
#
#   - `README_EXECUTE` (default `true`): turn ```julia blocks into runnable
#     `@example readme` blocks. Set `false` when the README's code is
#     illustrative (placeholder names) and must not execute on the home page.
#   - `README_STRIP_TABLES` (default `false`): drop raw markdown table rows and
#     a leading `**Websites**` line. Needed only for a README whose badges are a
#     raw table NOT wrapped in the managed markers; leave `false` to keep
#     content tables (the marker strip already removes the badges).
let readme = joinpath(dirname(@__DIR__), "README.md"),
    index = joinpath(@__DIR__, "src", "index.md")

    execute_readme = @isdefined(README_EXECUTE) ? README_EXECUTE : true
    strip_tables = @isdefined(README_STRIP_TABLES) ? README_STRIP_TABLES : false
    mkpath(dirname(index))
    open(index, "w") do io
        println(io, "```@meta")
        println(io,
            "EditURL = \"https://github.com/EpiAware/CensoredDistributions.jl/blob/main/README.md\"")
        println(io, "```")
        println(io)
        in_badges = false
        for line in eachline(readme)
            if occursin("<!-- badges:start -->", line)
                in_badges = true
                continue
            elseif occursin("<!-- badges:end -->", line)
                in_badges = false
                continue
            end
            in_badges && continue
            if execute_readme && startswith(line, "```julia")
                println(io, "```@example readme")
            elseif occursin("docs/src/assets/logo.svg", line)
                println(io, replace(line,
                    r"\s*<img[^>]*docs/src/assets/logo\.svg[^>]*>" => ""))
            elseif strip_tables && startswith(line, "|")
                continue
            elseif strip_tables && startswith(line, "**Websites**")
                continue
            else
                for (from, to) in INDEX_REWRITES
                    line = replace(line, from => to)
                end
                println(io, line)
            end
        end
    end
    println("Generated index.md from README.md")
end

# --- release-notes.md ------------------------------------------------------
# Combine the package-owned release-notes header with a project-root NEWS.md.
# Both are optional: a package without a NEWS.md (or the header file) simply
# gets no release-notes page.
news_src = joinpath(dirname(@__DIR__), "NEWS.md")
header_src = joinpath(@__DIR__, "release_notes_header.jl")
if isfile(news_src) && isfile(header_src)
    include(header_src)
    release_notes_dest = joinpath(@__DIR__, "src", "release-notes.md")
    open(release_notes_dest, "w") do io
        print(io, RELEASE_NOTES_HEADER)
        for line in eachline(news_src)
            println(io, line)
        end
    end
    println("Generated release-notes.md from header + NEWS.md")
else
    println("No NEWS.md / release-notes header found; skipping release notes")
end

# --- benchmark history page ------------------------------------------------
# A managed docs page wiring the package's benchmark suite + published
# performance timeline into the nav. The kit manages the page STRUCTURE (how to
# run the suite, and a link to the published history); the package supplies its
# own narrative via a package-owned `docs/benchmarks.md` prose hook, spliced in
# verbatim when present. Generation is gated on a `BENCHMARK_PAGE` config flag
# (default `true`) so a package can opt out. The history itself is published to
# the repo's `benchmarks` branch by `benchmark-history.yaml` and served at the
# GitHub Pages default host; that URL only resolves once Pages is enabled for
# that branch, so it is added to the linkcheck ignore list.
benchmark_linkcheck = Regex[]
if (@isdefined(BENCHMARK_PAGE) ? BENCHMARK_PAGE : true)
    repo_parts = split("EpiAware/CensoredDistributions.jl", "/")
    history_url = length(repo_parts) == 2 ?
                  "$(repo_parts[1]).github.io/$(repo_parts[2])/history" : ""
    intro_src = joinpath(@__DIR__, "benchmarks.md")
    prose = isfile(intro_src) ? read(intro_src, String) :
            "Performance benchmarks for `CensoredDistributions`."
    open(joinpath(@__DIR__, "src", "benchmarks.md"), "w") do io
        println(io, "# [Benchmarks](@id benchmarks)")
        println(io)
        println(io, prose)
        println(io)
        println(io, "## Running benchmarks")
        println(io)
        println(io, "Run the suite locally with `task benchmark`, and compare")
        println(io, "against `main` with `task benchmark-compare`. The suite")
        println(io, "is defined in `benchmark/benchmarks.jl`.")
        println(io)
        println(io, "## Performance history")
        println(io)
        if isempty(history_url)
            println(io, "A performance timeline is published on each release.")
        else
            println(io,
                "A continuously updated performance timeline (per-benchmark " *
                "plots and a ratio table) is published at")
            println(io, "[the benchmark history](https://$history_url).")
            push!(benchmark_linkcheck, Regex(replace(history_url, "." => "\\.")))
        end
    end
    println("Generated benchmarks.md (benchmark history page)")
else
    println("BENCHMARK_PAGE = false; skipping benchmark history page")
end

# --- API reference pages ---------------------------------------------------
# Generate the API reference pages (lib/public.md, lib/internals.md) from the
# module's documented bindings. `@autodocs` splices ONE docstring block per
# documented method SIGNATURE, so a function with several `@doc`-annotated
# methods appears many times in both the rendered API and the `@index`.
# Instead, each binding is listed ONCE in a `@docs` block: Documenter then
# combines all of a binding's method docstrings under a single heading with a
# single `@index` entry, while still showing every docstring. The binding list
# is derived from the module at build time, so it composes with whatever names
# happen to be exported.

# Whether `sym` is part of `mod`'s public API, matching how Documenter's
# `@autodocs` partitions `Public`/`Private` (`Base.ispublic` on >= 1.11, else
# exported).
function _is_public(mod::Module, sym::Symbol)
    return @static if isdefined(Base, :ispublic)
        Base.ispublic(mod, sym)
    else
        Base.isexported(mod, sym)
    end
end

# The bindings `mod` documents, split into public and private. Listing each
# binding ONCE (rather than once per method signature, as `@autodocs` would)
# keeps every method docstring while collapsing the `@index` to one entry per
# function.
function api_bindings(mod::Module)
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

function write_api_page(path, title, anchor, page, intro, api_heading,
        mod, names)
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
        println(io, "## $api_heading")
        println(io)
        println(io, "```@docs")
        for name in names
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
        "Documentation for `CensoredDistributions`'s public interface.",
        "Public API", CensoredDistributions, public
    )
    write_api_page(
        joinpath(lib_dir, "internals.md"),
        "Internal Documentation", nothing, "internals.md",
        "Documentation for `CensoredDistributions`'s internal interface.",
        "Internal API", CensoredDistributions, private
    )
    println(
        "Generated API pages: $(length(public)) public, " *
        "$(length(private)) internal bindings")
end

DocMeta.setdocmeta!(CensoredDistributions, :DocTestSetup,
    :(using CensoredDistributions); recursive = true)

# --- citations -------------------------------------------------------------
# Wire DocumenterCitations only when the package ships a bibliography, so a
# package with no `src/refs.bib` builds without citations.
bib_path = joinpath(@__DIR__, "src", "refs.bib")
plugins = if isfile(bib_path)
    [CitationBibliography(bib_path; style = :numeric)]
else
    Documenter.Plugin[]
end

makedocs(; sitename = "CensoredDistributions.jl",
    authors = "Sam Abbott, and contributors",
    # A fast build skips the network linkcheck (rate-limited, irrelevant to a
    # local content build); a full build keeps it strict.
    clean = true, doctest = false, linkcheck = !skip_notebooks,
    linkcheck_ignore = vcat(LINKCHECK_IGNORE, benchmark_linkcheck),
    warnonly = [
        :docs_block, :missing_docs, :autodocs_block, :cross_references
    ],
    modules = [CensoredDistributions],
    pages = pages,
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/EpiAware/CensoredDistributions.jl",
        devbranch = "main",
        devurl = "dev",
        # `deploy_url` controls the VitePress base path. The DEFAULT is
        # `nothing`: DocumenterVitepress then derives the base from the repo
        # name, so the site renders at the GitHub project-pages URL
        # (`epiaware.org/<Repo>.jl/`) with NO DNS to wire. A bare-domain
        # `deploy_url` would set base `/`, which only renders correctly when the
        # site is served at that domain's root (a wired custom subdomain). To
        # opt into `<pkg>.epiaware.org`, scaffold/update with
        # `docs_subdomain = true` (or a host string) AND set the repo's GitHub
        # Pages custom domain + a DNS record for that host.
        deploy_url = "censoreddistributions.epiaware.org",
        keep = :patch
    ),
    plugins = plugins
)

# Copy every tutorial data directory into the matching build output dir so the
# bundled data ships with the rendered site (and `@example` blocks that read it
# resolve at view time). Runs after `makedocs` so `clean = true` does not wipe
# it; generic over any tutorial that carries a `data` or `<name>-data` dir.
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
