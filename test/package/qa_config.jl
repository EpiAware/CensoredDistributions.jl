# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# QA configuration values the managed `quality.jl` testset reads. Fill in the
# package-specific inputs the shared helpers need; the standard testset logic
# stays in `quality.jl` (managed). Edit freely.

using CensoredDistributions

const QA_CONFIG = (
    # The module under test.
    mod = CensoredDistributions,

    # Path to the isolated JET environment (see test/jet/Project.toml). The
    # kit's `test_linting` runs that env's `runtests.jl` in a subprocess.
    jet_env = joinpath(@__DIR__, "..", "jet"),

    # Per-check Aqua relaxations. Empty = every Aqua check on.
    aqua = (;),

    # ExplicitImports `ignore`: internal/non-public names the package and its
    # extensions legitimately import. Carried over verbatim from the previous
    # bespoke ExplicitImports check:
    #   - Censored: used in get_dist.jl for Distributions.jl compatibility.
    #   - _in_closed_interval: internal Distributions bounds helper.
    #   - _gamma_cdf, _grad_p_a_series, _gamma_cdf_value_and_partials: internal
    #     AD-safe gamma CDF helpers used by the ChainRulesCore/ForwardDiff exts.
    #   - Dual, value, partials: ForwardDiff internals the ForwardDiff ext uses
    #     to reconstruct Dual return values (no public alternative).
    #   - @grad_from_chainrules, TrackedReal: ReverseDiff internals (the standard
    #     ChainRules->ReverseDiff bridge macro and tape value type).
    #   - _primal, _window_quantile: internal Convolved quadrature helpers the
    #     ChainRulesCore ext marks `@non_differentiable`.
    #   - _premodified_rate_primal: internal Modified knot-scan rate the
    #     ChainRulesCore and Mooncake exts mark non-differentiable/zero-derivative.
    #   - _cdf_ad_safe, _ccdf_ad_safe, _logcdf_ad_safe, _logccdf_ad_safe:
    #     internal AD-safe CDF/CCDF hooks the SurvivalDistributions ext overloads.
    #   - _split_edge_name, _is_positional_edge_name, _next_event_name,
    #     _all_positional_event_names, _split_edge: internal composer
    #     edge/event-name helpers the Mooncake ext registers `@zero_adjoint` on.
    #   - _ctor_has_check_args: internal leaf-reconstruction reflection helper the
    #     Mooncake ext registers a `@zero_adjoint` on (LTS foreigncall guard).
    #   - _collect_unique_boundaries: internal boundary builder the AD exts mark
    #     non-differentiable (zero-tangent rule).
    ei_ignore = (
        :Censored, :_in_closed_interval, :_gamma_cdf, :_grad_p_a_series,
        :_gamma_cdf_value_and_partials,
        :Dual, :value, :partials,
        Symbol("@grad_from_chainrules"), :TrackedReal,
        :_primal, :_window_quantile, :_premodified_rate_primal,
        :_cdf_ad_safe, :_ccdf_ad_safe, :_logcdf_ad_safe, :_logccdf_ad_safe,
        :_split_edge_name, :_is_positional_edge_name, :_next_event_name,
        :_all_positional_event_names, :_split_edge, :_ctor_has_check_args,
        :_collect_unique_boundaries
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
    # own surface (`prefixes`). `expect_phantoms = true` first asserts a
    # third-party phantom IS present (so the on-surface filter is doing real
    # work); `broken = true` quarantines a known, issue-tracked ambiguity.
    # The ReverseDiff and Enzyme extensions are exercised by the dedicated AD
    # harness (test/ad), which proves gradient correctness directly.
    extensions = (
        (; name = :CensoredDistributionsIntegralsExt,
            triggers = ("Integrals",),
            prefixes = ("CensoredDistributions", "Integrals")),
        # The SurvivalDistributions ext adds the AD-safe GeneralizedGamma
        # `logcdf` routing; clean on the package + SurvivalDistributions +
        # Distributions surface (originally the #653 check).
        (; name = :CensoredDistributionsSurvivalDistributionsExt,
            triggers = ("SurvivalDistributions",),
            prefixes = ("CensoredDistributions", "SurvivalDistributions",
                "Distributions")),
        # The FlexiChains ext adds the vectorised chain readers; loads DynamicPPL
        # then FlexiChains, clean on the package surface.
        (; name = :CensoredDistributionsFlexiChainsExt,
            triggers = ("DynamicPPL", "FlexiChains"),
            prefixes = ("CensoredDistributions",)),
        # The reverse-mode AD extensions (the ChainRules rule + the Mooncake
        # wrapper) are clean on the package surface; Mooncake pulls in both.
        (; name = :CensoredDistributionsChainRulesCoreExt,
            triggers = ("Mooncake",),
            prefixes = ("CensoredDistributions",)),
        (; name = :CensoredDistributionsMooncakeExt,
            triggers = ("Mooncake",),
            prefixes = ("CensoredDistributions",)),
        # Catalyst transitively loads Symbolics and its `SymbolicsDistributionsExt`,
        # whose `logpdf/cdf/...(::Distribution, ::Num)` methods are ambiguous with
        # every concrete `f(::SomeDist, ::Real)` (a `Num`/`Real` overlap) — a
        # pre-existing phantom of the Symbolics integration, not introduced here.
        # `expect_phantoms` asserts those phantoms ARE present; the on-surface
        # filter then proves the Catalyst bridge adds no ambiguity of its own.
        (; name = :CensoredDistributionsCatalystExt,
            triggers = ("Catalyst",),
            prefixes = ("CensoredDistributions",),
            expect_phantoms = true),
        # FIXED (issue #672): the `_gamma_cdf` Dual overload set in
        # ext/CensoredDistributionsForwardDiffExt.jl previously had 6 method
        # ambiguities in its partial-Dual signatures; collapsing the overloads so
        # the "at least one Dual" space is partitioned disjointly removed them.
        (; name = :CensoredDistributionsForwardDiffExt,
            triggers = ("ForwardDiff",),
            prefixes = ("CensoredDistributions", "ForwardDiff")),
        # FIXED (issue #673): `composed_distribution_model` over a
        # `latent(primary_censored(...))` row vector previously matched both the
        # single-record and batch methods ambiguously; the disambiguating batch
        # method for `Latent{<:PrimaryCensored}` over `::AbstractVector` removes it.
        (; name = :CensoredDistributionsDynamicPPLExt,
            triggers = ("DynamicPPL",),
            prefixes = ("CensoredDistributions",))
    )
)
