# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# Package-specific configuration read by the managed `make.jl`. It drives the
# Literate.jl tutorial pipeline and the README/index link rewrites, and lists
# the linkcheck URLs to ignore. These values reproduce CensoredDistributions.jl's
# documentation build; edit them as the docs grow.

# Light tutorials: Literate emits `@example` blocks that Documenter runs
# in-process. They are cheap and accumulate no native/memory state.
const LIGHT_TUTORIALS = String[
    "analytical-primarycensored-cdfs.jl",
    "exponentially-tilted-primary-events.jl"
]

# Heavy tutorials: live MCMC fits or a multi-backend AD benchmark, plus
# CairoMakie / PairPlots. Each runs once in a fresh subprocess so native/memory
# state cannot accumulate across the long-lived Documenter process.
const HEAVY_TUTORIALS = String[
    "ad-backends.jl",
    "fitting-with-turing.jl"
]

# Where the tutorial `.jl` sources and rendered `.md` pages live, relative to
# `docs/src`.
const TUTORIALS_SUBDIR = joinpath("getting-started", "tutorials")

# Fast-build stubs (`--skip-notebooks`): each heading preserves the tutorial's
# `@id` so cross-references from other pages still resolve in a fast build.
const TUTORIAL_STUBS = Pair{String, String}[
    "analytical-primarycensored-cdfs.md" => "# Analytical CDF solutions",
    "exponentially-tilted-primary-events.md" => "# Exponentially tilted primary events",
    "ad-backends.md" => "# [Automatic differentiation backends](@id ad-backends)",
    "fitting-with-turing.md" => "# Fitting with Turing.jl"
]

# The benchmark history page is published by a separate workflow and only
# resolves once the maintainer enables Pages for the benchmarks branch, so the
# managed `make.jl` already adds it to the linkcheck ignore list. Nothing
# further to ignore here.
const LINKCHECK_IGNORE = Regex[]

# README -> index.md link rewrites: absolute docs URLs become in-site `@ref`s so
# links stay within the built version.
const INDEX_REWRITES = Pair{String, String}[
    "[Getting Started documentation](https://censoreddistributions.epiaware.org/stable/getting-started/)" => "[Getting Started documentation](@ref getting-started)",
    "[Getting Started Tutorials](https://censoreddistributions.epiaware.org/stable/getting-started/)" => "[Getting Started Tutorials](@ref getting-started)",
    "[API Reference](https://censoreddistributions.epiaware.org/stable/lib/public)" => "[API Reference](@ref public-api)",
    "[Developer Documentation](https://censoreddistributions.epiaware.org/stable/developer/)" => "[Developer Documentation](@ref developer)",
    "[developer documentation](https://censoreddistributions.epiaware.org/stable/developer/)" => "[developer documentation](@ref developer)",
    "[Automatic differentiation backends](https://censoreddistributions.epiaware.org/stable/getting-started/tutorials/ad-backends/)" => "[Automatic differentiation backends](@ref ad-backends)"
]

# README ```julia blocks become runnable `@example readme` blocks on the home
# page (the README's examples are real, runnable code).
const README_EXECUTE = true

# No README section is dropped from the home page. (The badge block is always
# stripped via its `<!-- badges:start/end -->` markers.) In particular the
# "Relationship to Distributions.jl" comparison table is KEPT on the home page.
const INDEX_STRIP_SECTIONS = String[]

# Generate the benchmark page (`src/benchmarks.md`): the package-owned
# `docs/benchmarks.md` prose hook plus the rendered performance history.
const BENCHMARK_PAGE = true
