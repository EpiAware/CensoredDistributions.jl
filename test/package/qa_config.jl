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

    # Per-check Aqua relaxations. Empty = every Aqua check on.
    aqua = (;),

    # ExplicitImports `ignore`: internal/non-public names the package and its
    # extensions legitimately import. Carried over from the previous bespoke
    # ExplicitImports check:
    #   - Censored: used in get_dist.jl for Distributions.jl compatibility.
    #   - _in_closed_interval: internal Distributions bounds helper.
    #   - _gamma_cdf, _grad_p_a_series, _gamma_cdf_value_and_partials: internal
    #     AD-safe gamma CDF helpers used by the ChainRulesCore/ForwardDiff exts.
    #   - Dual, value, partials: ForwardDiff internals the ForwardDiff ext uses
    #     to reconstruct Dual return values (no public alternative).
    #   - @grad_from_chainrules, TrackedReal: ReverseDiff internals (the standard
    #     ChainRules->ReverseDiff bridge macro and tape value type).
    ei_ignore = (
        :Censored, :_in_closed_interval, :_gamma_cdf, :_grad_p_a_series,
        :_gamma_cdf_value_and_partials,
        :Dual, :value, :partials,
        Symbol("@grad_from_chainrules"), :TrackedReal
    ),

    # Docstring `crossref_ignore`: upstream names docstrings link to via
    # `[`name`](@ref)` that are not part of this package's own surface.
    crossref_ignore = (:pdf, :cdf, :logpdf, :logcdf, :rand, :quantile),

    # Extra docstring-format options. Empty = the standard strict checks.
    docstring = (;),

    # README section-structure check. CD lists "Supporting and citing" before
    # "Contributing" (the reverse of the standard order), so the required set is
    # given explicitly in CD's order rather than relying on the standard one.
    readme = (;
        path = joinpath(@__DIR__, "..", ".."),
        required = [
            ("Why", "Overview", "Features", "About"),
            ("Getting started", "Usage", "Quickstart", "Quick start"),
            ("Documentation", "Where to learn more", "Learn more"),
            ("Citing", "Citation", "License", "Supporting"),
            ("Contributing",)
        ]
    ),

    # Package extensions to ambiguity-check. Each loads its trigger package(s),
    # then asserts the extension introduces no method ambiguity on the package's
    # own surface. `broken = true` quarantines a known, issue-tracked ambiguity.
    # Only extensions whose triggers are test-env deps are listed; the
    # ReverseDiff and Enzyme extensions are exercised by the dedicated AD harness
    # (test/ad), which proves gradient correctness directly.
    extensions = (
        (; name = :CensoredDistributionsIntegralsExt,
            triggers = ("Integrals",),
            prefixes = ("CensoredDistributions", "Integrals")),
        (; name = :CensoredDistributionsChainRulesCoreExt,
            triggers = ("Mooncake",),
            prefixes = ("CensoredDistributions",)),
        (; name = :CensoredDistributionsMooncakeExt,
            triggers = ("Mooncake",),
            prefixes = ("CensoredDistributions",)),
        # Quarantined (issue #672): the `_gamma_cdf` Dual overload set in
        # ext/CensoredDistributionsForwardDiffExt.jl has 6 method ambiguities in
        # its partial-Dual signatures. Tracked as @test_broken; the fix (collapse
        # the overloads so the "at least one Dual" space is partitioned
        # disjointly) flips this green, signalling the bug is gone.
        (; name = :CensoredDistributionsForwardDiffExt,
            triggers = ("ForwardDiff",),
            prefixes = ("CensoredDistributions", "ForwardDiff"),
            broken = true)
    )
)
