# MANAGED by EpiAwarePackageTools.scaffold — do not edit by hand.
#
# Thin entry point for the standard EpiAware documentation build. All build
# logic lives in `EpiAwarePackageTools.DocsBuild.build_docs` (versioned +
# tested in the kit); this file only wires the package-owned `pages.jl` +
# `docs_config.jl` into that call, so it can be re-applied on every `scaffold_update`
# without losing package content.
#
# `build_docs`:
#   - runs the Literate tutorial pipeline (light in-process, heavy one per
#     subprocess) driven by `docs_config.jl`; under `--skip-notebooks` the
#     light tutorials still render in-process (cheap) and only the heavy ones
#     fall back to fast-build heading stubs; independent of that flag, any
#     `FORCE_STUB_TUTORIALS` entry always renders from its heading stub
#     without running, while its heavy siblings still execute normally,
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

# The docs navigation tree (`pages.jl`) and package-specific build config
# (`docs_config.jl`: tutorial lists, README/index link rewrites, named-section
# strips, linkcheck ignores). Both are package-owned — written on `scaffold`,
# never re-applied by `scaffold_update` — so an adopter predating either file has none.
# Guard the include so a re-applied managed `make.jl` still loads and falls
# back to defaults (#163) rather than erroring on a missing file; `_cfg` then
# defaults every key a missing or older config predates. The fallback warns
# because a build that silently defaults `pages` publishes a Home-only
# navigation, which a green docs run would otherwise hide (#188).
for _f in ("pages.jl", "docs_config.jl")
    if isfile(joinpath(@__DIR__, _f))
        include(joinpath(@__DIR__, _f))
    else
        @warn "docs/$(_f) not found; building with defaults " *
              "(a missing pages.jl leaves the site with a Home-only nav). " *
              "Write it if this package should own one."
    end
end

# Read a package-owned config const, defaulting when a missing or older
# `docs_config.jl`/`pages.jl` (package-owned, not re-applied by `scaffold_update`)
# predates it.
_cfg(sym, default) = isdefined(@__MODULE__, sym) ?
                     getfield(@__MODULE__, sym) : default

build_docs(
    CensoredDistributions;
    repo = "EpiAware/CensoredDistributions.jl",
    authors = "Sam Abbott, Damon Bayer, Sam Brand, Michael DeWitt, Joseph Lemaitre",
    deploy_url = "censoreddistributions.epiaware.org",
    pages = _cfg(:pages, ["Home" => "index.md"]),
    skip_notebooks = "--skip-notebooks" in ARGS ||
                     get(ENV, "SKIP_NOTEBOOKS", "false") == "true",
    tutorials_subdir = _cfg(:TUTORIALS_SUBDIR,
        joinpath("getting-started", "tutorials")),
    light_tutorials = _cfg(:LIGHT_TUTORIALS, String[]),
    heavy_tutorials = _cfg(:HEAVY_TUTORIALS, String[]),
    tutorial_stubs = _cfg(:TUTORIAL_STUBS, Pair{String, String}[]),
    force_stub_tutorials = _cfg(:FORCE_STUB_TUTORIALS, String[]),
    linkcheck_ignore = _cfg(:LINKCHECK_IGNORE, Regex[]),
    index_rewrites = _cfg(:INDEX_REWRITES, Pair{String, String}[]),
    readme_execute = _cfg(:README_EXECUTE, true),
    index_strip_sections = _cfg(:INDEX_STRIP_SECTIONS, String[]),
    benchmark_page = _cfg(:BENCHMARK_PAGE, false),
    # Performance-history rendering (#193): restrict to headline suites and cap
    # the overall summary/detail to the most-recent revisions. Both default to
    # the whole timeline when a package predates these config keys.
    history_suites = _cfg(:HISTORY_SUITES, String[]),
    history_commits = _cfg(:HISTORY_COMMITS, 5),
    # Overall-summary regression cutoff: the ratio (against the oldest shown
    # revision) at or above which a suite's `Status` flags "⚠ reg". Defaults
    # when a package predates this config key.
    history_regression_threshold = _cfg(:HISTORY_REGRESSION_THRESHOLD, 1.1),
    # Extra docstring-owning modules for a re-export the alias walk cannot
    # reach (e.g. one referenced only from prose); owners of re-exported API
    # bindings are auto-discovered, so most packages leave this empty (#175).
    extra_modules = _cfg(:EXTRA_MODULES, Module[])
)
