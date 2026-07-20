# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# QA configuration values the managed `quality.jl` testset reads. Fill in the
# package-specific inputs the shared helpers need; the standard testset logic
# stays in `quality.jl` (managed). Edit freely.

using CensoredDistributions

const QA_CONFIG = (
    # The module under test.
    mod = CensoredDistributions,

    # Path to the isolated JET environment (see test/jet/Project.toml).
    jet_env = joinpath(@__DIR__, "..", "jet"),

    # Per-check Aqua relaxations, e.g. (; ambiguities = false). Empty = all on.
    aqua = (;),

    # ExplicitImports `ignore`: symbols the main module legitimately imports
    # non-publicly.
    # - `Censored`: Distributions.jl compatibility type used in get_dist.jl.
    # - `_in_closed_interval`: internal Distributions utility used for bounds
    #   checking.
    # - `_gamma_cdf`, `_grad_p_a_series`, `_gamma_cdf_value_and_partials`:
    #   internal AD-safe gamma CDF helpers used by the ChainRulesCore and
    #   ForwardDiff extensions.
    # - `Dual`, `value`, `partials`: ForwardDiff internals used by the
    #   ForwardDiff extension to construct Dual return values; no public
    #   alternative for the Dual reconstruction pattern.
    # - `@grad_from_chainrules`, `TrackedReal`: ReverseDiff internals used by
    #   the ReverseDiff extension; `@grad_from_chainrules` is the standard
    #   ChainRules-to-ReverseDiff bridge macro and `TrackedReal` is the
    #   public-by-convention tape value type, neither marked `public`.
    # - `_collect_unique_boundaries`: internal boundary builder the AD
    #   extensions import to mark non-differentiable (zero-tangent rule).
    ei_ignore = (
        :Censored, :_in_closed_interval, :_gamma_cdf, :_grad_p_a_series,
        :_gamma_cdf_value_and_partials,
        :Dual, :value, :partials,
        Symbol("@grad_from_chainrules"), :TrackedReal,
        :_collect_unique_boundaries
    ),

    # Docstring `crossref_ignore`: upstream names docstrings link to via
    # `[`name`](@ref)`.
    crossref_ignore = (:pdf, :cdf, :logpdf, :logcdf, :rand, :quantile),

    # Extra docstring-format options, e.g.
    # (; exported_only_examples = true, require_field_docs = true).
    docstring = (;),

    # README section-structure check. `path` is the package root (its
    # README.md). Override `required`/`order` to extend or relax the standard
    # section set, e.g.
    #   (; required = vcat(STANDARD_README_SECTIONS, [("Benchmarks",)]))
    # Empty `(;)` uses the standard structure in standard order.
    #
    # `order = false`: this repo's own README has "Supporting and citing"
    # before "Contributing", the reverse of `STANDARD_README_SECTIONS`'
    # order — even though that constant's docstring says it was distilled
    # from this repo as the org's gold standard. All five required sections
    # are present; only their relative order disagrees with the standard.
    # Not resolved here (a README content/structure question, out of scope
    # for this build-tooling migration) — flagged in the PR for a maintainer
    # call: reorder this README to match, or fix the stale kit constant.
    readme = (; path = joinpath(@__DIR__, "..", ".."), order = false),

    # Package extensions to ambiguity-check. Each entry:
    #   (; name = :MyPkgSomeTriggerExt,
    #      triggers = ("SomeTrigger",),       # packages to load first
    #      prefixes = ("MyPkg", "SomeTrigger"),
    #      expect_phantoms = false,    # true if a third party adds phantoms
    #      broken = false)             # true to quarantine a known ambiguity
    #
    # Only `CensoredDistributionsIntegralsExt` is listed here.
    # `CensoredDistributionsChainRulesCoreExt`, `...EnzymeExt`,
    # `...ForwardDiffExt`, `...MooncakeExt`, and `...ReverseDiffExt` are
    # deliberately NOT listed: their trigger packages live in the isolated
    # `test/ad` environment, not the main test env this check runs in (adding
    # them here would pull five heavy AD backends into the main env, the
    # exact duplication the isolation exists to avoid — mirrors
    # LoweredDistributions.jl's precedent for its AlgebraicPetri extension).
    # Their extensions are already exercised, more thoroughly than a bare
    # ambiguity scan, by the dedicated AD gradient test harness in `test/ad/`.
    extensions = (
        (; name = :CensoredDistributionsIntegralsExt,
        triggers = ("Integrals",),
        prefixes = ("CensoredDistributions", "Integrals")),
    )
)
