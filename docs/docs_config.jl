# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# Package-specific configuration read by the managed `make.jl`. It drives the
# Literate.jl tutorial pipeline and the README/index link rewrites, and lists
# the linkcheck URLs to ignore.

using Distributions: Distributions
using Statistics: Statistics

# Tutorial source `.jl` files (Literate scripts) under `TUTORIALS_SUBDIR`.
#
# Light tutorials emit `@example` blocks that Documenter runs in-process; keep
# cheap tutorials here.
#
# Every current tutorial loads CairoMakie/AlgebraOfGraphics for plotting, and
# some load multiple AD backends or Turing on top, so none is cheap enough to
# run in-process; all four are `HEAVY_TUTORIALS` below.
const LIGHT_TUTORIALS = String[]

# Heavy tutorials (live MCMC fits, multi-backend AD, plotting) are each
# executed once in a fresh subprocess so native/memory state cannot accumulate.
const HEAVY_TUTORIALS = [
    "analytical-primarycensored-cdfs.jl",
    "exponentially-tilted-primary-events.jl",
    "ad-backends.jl",
    "fitting-with-turing.jl"
]

# Where the tutorial `.jl` sources and rendered `.md` pages live, relative to
# `docs/src`.
const TUTORIALS_SUBDIR = joinpath("getting-started", "tutorials")

# Fast-build stubs (`--skip-notebooks`): `"file.md" => "# Heading"` pairs. The
# heading should preserve the tutorial's `@id` (e.g.
# `"# [Title](@id my-anchor)"`) so cross-references from other pages still
# resolve in a fast build. Three of the four tutorials carry no explicit `@id`
# in their source heading (Documenter auto-slugs the heading text instead), so
# their stub headings match the source heading verbatim rather than inventing
# a new anchor.
const TUTORIAL_STUBS = Pair{String, String}[
    "analytical-primarycensored-cdfs.md" => "# Analytical CDF solutions for primary censored distributions",
    "exponentially-tilted-primary-events.md" => "# How exponentially tilted primary events affect observed delay distributions",
    "ad-backends.md" => "# [Automatic differentiation backends](@id ad-backends)",
    "fitting-with-turing.md" => "# Fitting CensoredDistributions.jl modified distributions with Turing.jl"
]

# Heavy tutorials that always render from their `TUTORIAL_STUBS` heading and
# never execute, independent of `--skip-notebooks` — the escape hatch for a
# heavy tutorial with a problem of its own (e.g. a model that does not
# terminate in reasonable time), so it need not block its siblings from
# running for real. Leave empty; every heavy tutorial with no such problem
# should execute.
const FORCE_STUB_TUTORIALS = String[]

# Whether this package advertises itself as part of the EpiAware ecosystem: a
# "Part of the EpiAware ecosystem" section in the managed README block, and the
# EpiAware logo + org links in the docs footer. Left off (matching this
# repo's current README, which carries no such section) rather than opted in
# as a side effect of this migration — a maintainer call, not a build-mechanics
# one.
const ORG_BRANDING = false

# Regexes for URLs to skip during the (full-build) linkcheck, e.g. a page
# published by a separate workflow that is not yet live. This repo already has
# a live docs site and GitHub Discussions enabled, so none of the usual
# brand-new-package exclusions apply; left empty.
const LINKCHECK_IGNORE = Regex[]

# README -> index.md link rewrites: `from => to` pairs applied line by line,
# e.g. rewriting an absolute docs URL to an in-site `@ref` so links stay within
# the built version. Ported from the bespoke rewrite list the old `make.jl`
# applied by hand; two entries in that list ("Getting Started Tutorials",
# "Automatic differentiation backends" as standalone link text) no longer
# match anything in the current README and are dropped as dead.
const INDEX_REWRITES = [
    "[Getting Started documentation](https://censoreddistributions.epiaware.org/stable/getting-started/)" => "[Getting Started documentation](@ref getting-started)",
    "[API Reference](https://censoreddistributions.epiaware.org/stable/lib/public)" => "[API Reference](@ref public-api)",
    "[Developer Documentation](https://censoreddistributions.epiaware.org/stable/developer/)" => "[Developer Documentation](@ref developer)",
    "[developer documentation](https://censoreddistributions.epiaware.org/stable/developer/)" => "[developer documentation](@ref developer)"
]

# Whether README ```julia blocks become runnable `@example readme` blocks on the
# generated home page. Keep `true` when the README's examples are real, runnable
# code; set `false` when they are illustrative (placeholder names) and must not
# execute. The old `make.jl` rewrote every ```julia fence unconditionally, so
# this matches existing behaviour.
const README_EXECUTE = true

# README headings whose whole section (heading + body, up to the next heading
# of the same or a higher level) is dropped when generating the home page. The
# managed badge block is always stripped via its `<!-- badges:start/end -->`
# markers (newly added to README.md by this same change; the old `make.jl`
# instead stripped every line starting with `|`, which would have also
# stripped the "Relationship to Distributions.jl" comparison table further
# down the README — a real bug in the old build this migration fixes). No
# further named section needs stripping.
const INDEX_STRIP_SECTIONS = String[]

# Whether the build generates the benchmark page (`src/benchmarks.md`). This
# repo has a `benchmark/` suite but has never published a benchmarks nav page
# (absent from the current `pages.jl`); left `false` to preserve the existing
# page set rather than adding a new page as a side effect of this migration.
const BENCHMARK_PAGE = false

# Extra docstring-owning modules for a re-export the auto-discovery in
# `build_docs` cannot reach. `docs/src/lib/internals.md` lists bare
# `CensoredDistributions.mean`/`.median`/`.quantile`/`.std`/`.var` in its
# `@docs` block: this package extends these Statistics/Distributions generic
# functions with new methods but adds no docstring of its own to them, so
# Documenter displays the UPSTREAM docstring verbatim — and that upstream
# docstring's own `[`skipmissing`](@ref)`/`[`gradlogpdf`](@ref)`/etc.
# cross-references need `Statistics`/`Distributions` in the resolution scope
# to resolve. Auto-discovery only widens for genuinely re-exported (exported)
# bindings; these are internal (unexported), so it does not reach them.
# Without this, the full (non-`--skip-notebooks`) docs build fails at the
# final VitePress dead-link check on 14 literal unresolved `@ref`s in the
# generated `lib/internals.md` — discovered while migrating this build, not
# introduced by it (the docstrings themselves are untouched).
const EXTRA_MODULES = Module[Distributions, Statistics]
