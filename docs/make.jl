# MANAGED by EpiAwarePackageTools.scaffold — do not edit by hand.
#
# Thin entry point for the standard EpiAware documentation build. All build
# logic lives in `EpiAwarePackageTools.DocsBuild.build_docs` (versioned +
# tested in the kit); this file only wires the package-owned `pages.jl` +
# `docs_config.jl` into that call, so it can be re-applied on every `update`
# without losing package content.
#
# `build_docs`:
#   - runs the Literate tutorial pipeline (light in-process, heavy one per
#     subprocess) driven by `docs_config.jl`, with fast-build stubs on
#     `--skip-notebooks`,
#   - generates `src/index.md` from the README (badges stripped, any
#     `INDEX_STRIP_SECTIONS` removed, link rewrites applied),
#   - generates `src/release-notes.md` from a project-root `NEWS.md`,
#   - generates `src/benchmarks.md` (a tight skeleton + the package-owned
#     `docs/benchmarks.md` prose hook + the rendered performance history),
#   - generates the API pages from the module's documented bindings, and
#   - renders + deploys with DocumenterVitepress.
#
# Build it with `task docs` (or `julia --project=docs docs/make.jl`).

using Pkg: Pkg
Pkg.instantiate()

using EpiAwarePackageTools
using CensoredDistributions

# The docs navigation tree (package-owned).
include("pages.jl")
# Package-specific build config: tutorial lists, README/index link rewrites,
# named-section strips, and linkcheck ignores. Package-owned and never
# overwritten, so an empty config builds a site with no tutorials.
include("docs_config.jl")

# Read a package-owned config const, defaulting when an older `docs_config.jl`
# (package-owned, not re-applied by `update`) predates it.
_cfg(sym, default) = isdefined(@__MODULE__, sym) ?
                     getfield(@__MODULE__, sym) : default

build_docs(
    CensoredDistributions;
    repo = "EpiAware/CensoredDistributions.jl",
    authors = "Sam Abbott, and contributors",
    deploy_url = "censoreddistributions.epiaware.org",
    pages = pages,
    skip_notebooks = "--skip-notebooks" in ARGS ||
                     get(ENV, "SKIP_NOTEBOOKS", "false") == "true",
    tutorials_subdir = _cfg(:TUTORIALS_SUBDIR,
        joinpath("getting-started", "tutorials")),
    light_tutorials = _cfg(:LIGHT_TUTORIALS, String[]),
    heavy_tutorials = _cfg(:HEAVY_TUTORIALS, String[]),
    tutorial_stubs = _cfg(:TUTORIAL_STUBS, Pair{String, String}[]),
    linkcheck_ignore = _cfg(:LINKCHECK_IGNORE, Regex[]),
    index_rewrites = _cfg(:INDEX_REWRITES, Pair{String, String}[]),
    readme_execute = _cfg(:README_EXECUTE, true),
    index_strip_sections = _cfg(:INDEX_STRIP_SECTIONS, String[]),
    benchmark_page = _cfg(:BENCHMARK_PAGE, true)
)
