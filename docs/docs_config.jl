# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# Package-specific configuration read by the managed `make.jl`. It drives the
# Literate.jl tutorial pipeline and the README/index link rewrites, and lists
# the linkcheck URLs to ignore. These values reproduce CensoredDistributions.jl's
# documentation build exactly; edit them as the docs grow.

# Restored (#827) from the #363 whole-build SKIP_NOTEBOOKS stop-gap: the
# heavy MCMC tutorials no longer collectively need to be stubbed. andv/bdbv
# linelist switched from AutoMooncake to AutoForwardDiff (compile-cost only,
# not an AD-demo change) and now render in well under the 6h CI limit;
# linear-chain-sir, branching-competing, pairwise-survival-transmission and
# recurrent-multistate were never actually slow (never measured before #363's
# stop-gap); ad-backends is untouched (a genuine multi-backend AD demo,
# ~30-45m, fits comfortably). `epinowcast-nowcasting` alone is
# FORCE_STUB_TUTORIALS below, not SKIP_NOTEBOOKS — see that comment.

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

# `epinowcast-nowcasting` alone never renders, independent of
# `--skip-notebooks`/`SKIP_NOTEBOOKS` (kit `force_stub_tutorials`, #111): even
# right-sized (`max_delay = 10`, `n_days = 28`) with a `NUTS(...; max_depth =
# 8)` backstop, it did not complete a single 60-draw/2-chain serial fit in
# 2.5h locally (#827) — a genuine weak-identifiability signal (the
# expectation random walk and the reference-date hazard random walk trade
# off), not a toy-data or compile-cost artefact like its siblings above. This
# is a maintainer modelling decision (non-centred parameterisation, tighter
# RW-scale priors, or fewer effect families), not something to paper over
# here; see #827 for the investigation.
const FORCE_STUB_TUTORIALS = String["epinowcast-nowcasting.jl"]

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
    "epinowcast-nowcasting.md" => """
    # [An epinowcast-style hazard nowcasting model](@id epinowcast-nowcasting)

    !!! note "Stubbed: known weak identifiability, not a build-speed issue"
        Right-sizing the toy data alone does not make this fit terminate
        in reasonable time — a genuine identifiability signal under
        investigation, not a toy-data or compile-cost artefact. See
        [issue #827](https://github.com/EpiAware/CensoredDistributions.jl/issues/827).
    """,
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

# No README section is dropped from the home page. (The badge block is always
# stripped via its `<!-- badges:start/end -->` markers.) In particular the
# "Relationship to Distributions.jl" comparison table is KEPT on the home page;
# the previous bespoke `make.jl` dropped it, which was a mistake.
const INDEX_STRIP_SECTIONS = String[]

# Generate the benchmark page (`src/benchmarks.md`): the package-owned
# `docs/benchmarks.md` prose hook plus the rendered performance history.
const BENCHMARK_PAGE = true
