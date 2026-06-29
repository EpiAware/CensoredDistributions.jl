# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# Package-specific configuration read by the managed `make.jl`. It drives the
# Literate.jl tutorial pipeline and the README/index link rewrites, and lists
# the linkcheck URLs to ignore. These values reproduce CensoredDistributions.jl's
# documentation build exactly; edit them as the docs grow.

# Light tutorials: Literate emits `@example` blocks that Documenter runs
# in-process. They are cheap and accumulate no native/memory state.
const LIGHT_TUTORIALS = String[
    "analytical-primarycensored-cdfs.jl",
    "exponentially-tilted-primary-events.jl",
    "composer-toolkit.jl",
    "survival-delay-families.jl",
    "fit-marginal-sample-event-based.jl"
]

# Heavy tutorials: live MCMC fits or a multi-backend AD benchmark, plus
# CairoMakie / AlgebraOfGraphics / PairPlots. Each runs once in a fresh
# subprocess so native/memory state cannot accumulate across the long-lived
# Documenter process.
const HEAVY_TUTORIALS = String[
    "ad-backends.jl",
    "fitting-with-turing.jl",
    "bdbv-linelist-analysis.jl",
    "andv-linelist-analysis.jl",
    "ebola-stratified-delays.jl",
    "rt-renewal-convolution.jl",
    "renewal-susceptibility.jl",
    "epinowcast-nowcasting.jl",
    "linear-chain-sir.jl",
    "branching-competing.jl",
    "pairwise-survival-transmission.jl",
    "recurrent-multistate.jl"
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
    "fitting-with-turing.md" => "# Fitting with Turing.jl",
    "composer-toolkit.md" => "# [Composing censored distributions](@id composer-toolkit)",
    "survival-delay-families.md" => "# [Delay families from SurvivalDistributions.jl](@id survival-delay-families)",
    "fit-marginal-sample-event-based.md" => "# [Fit marginal, sample event based](@id fit-marginal-sample-event-based)",
    "bdbv-linelist-analysis.md" => "# [Bundibugyo Ebola delays from the 2012 Isiro line list](@id bdbv-linelist-analysis)",
    "andv-linelist-analysis.md" => "# [Real-time Andes virus delays from the Epuyén line list](@id andv-linelist-analysis)",
    "ebola-stratified-delays.md" => "# [Stratified onset-to-test delays in the 2014-2016 Sierra Leone Ebola outbreak](@id ebola-stratified-delays)",
    "rt-renewal-convolution.md" => "# [An Rt renewal model with delay convolution](@id rt-renewal-convolution)",
    "renewal-susceptibility.md" => "# [A susceptibility-depleting renewal model](@id renewal-susceptibility)",
    "linear-chain-sir.md" => "# [A composed delay as ODE compartments: the linear chain trick](@id linear-chain-sir)",
    "epinowcast-nowcasting.md" => "# [An epinowcast-style hazard nowcasting model](@id epinowcast-nowcasting)",
    "branching-competing.md" => "# [A branching-process-like natural history with competing outcomes](@id branching-competing)",
    "recurrent-multistate.md" => "# [Recurrent multi-state transitions: waning and reinfection](@id recurrent-multistate)",
    "pairwise-survival-transmission.md" => "# [Pairwise survival analysis of transmission (Kenah)](@id pairwise-survival-transmission)"
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

# Drop raw markdown table rows when generating the home page. CD's README
# carries a "Relationship to Distributions.jl" comparison table that the home
# page omits (the same behaviour as the previous bespoke `make.jl`).
const README_STRIP_TABLES = true

# Generate the benchmark history page (`src/benchmarks.md`), splicing in the
# package-owned `docs/benchmarks.md` prose.
const BENCHMARK_PAGE = true
