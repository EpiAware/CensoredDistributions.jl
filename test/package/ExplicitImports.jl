@testitem "ExplicitImports analysis" tags=[:quality] begin
    using EpiAwarePackageTools: test_explicit_imports

    # The four ExplicitImports conformance checks (no stale explicit imports,
    # no implicit imports, all explicit imports public, all via owners) run via
    # `EpiAwarePackageTools.test_explicit_imports`. The only CD-specific input is
    # the `ignore` tuple of internal/non-public names CD legitimately imports;
    # it stays here as a package fixture.
    #
    # Skip the public-import check for internal/non-public functions we need:
    # - Censored: Used in get_dist.jl for Distributions.jl compatibility
    # - _in_closed_interval: Internal Distributions utility used for bounds checking
    # - _gamma_cdf, _grad_p_a_series, _gamma_cdf_value_and_partials:
    #   Internal AD-safe gamma CDF helpers used by the ChainRulesCore
    #   and ForwardDiff extensions
    # - Dual, value, partials: ForwardDiff internals used by the
    #   ForwardDiff extension to construct Dual return values; no public
    #   alternative for the Dual reconstruction pattern.
    # - @grad_from_chainrules, TrackedReal: ReverseDiff internals used by
    #   the ReverseDiff extension; @grad_from_chainrules is the standard
    #   ChainRules-to-ReverseDiff bridge macro and TrackedReal is the
    #   public-by-convention tape value type, neither marked `public`.
    # - _primal, _window_quantile: internal Convolved quadrature helpers
    #   marked `@non_differentiable` by the ChainRulesCore extension.
    # - _premodified_rate_primal: internal `Modified` knot-scan rate the
    #   ChainRulesCore and Mooncake extensions import to mark
    #   `@non_differentiable`/`@zero_derivative` (the knots carry no gradient;
    #   the same internal-import-to-add-a-rule pattern as _window_quantile).
    # - _cdf_ad_safe, _ccdf_ad_safe, _logcdf_ad_safe, _logccdf_ad_safe:
    #   internal AD-safe CDF/CCDF hooks the SurvivalDistributions
    #   extension overloads for its leaf families; the ext
    #   imports them to add methods, the standard internal-import pattern.
    # - _split_edge_name, _is_positional_edge_name, _next_event_name,
    #   _all_positional_event_names, _split_edge: internal composer
    #   edge/event-name helpers the Mooncake extension imports to register
    #   `@zero_adjoint` rules (the pure-string name derivation Mooncake
    #   reverse cannot trace); the same internal-import-to-add-a-rule pattern.
    # - _ctor_has_check_args: internal leaf-reconstruction reflection helper
    #   the Mooncake extension imports to register a `@zero_adjoint` so its
    #   `hasmethod` foreigncall is not traced on Julia LTS.
    # - _collect_unique_boundaries: internal boundary builder the AD
    #   extensions import to mark non-differentiable (zero-tangent rule).
    test_explicit_imports(CensoredDistributions;
        ignore = (
            :Censored, :_in_closed_interval, :_gamma_cdf, :_grad_p_a_series,
            :_gamma_cdf_value_and_partials,
            :Dual, :value, :partials,
            Symbol("@grad_from_chainrules"), :TrackedReal,
            :_primal, :_window_quantile, :_premodified_rate_primal,
            :_cdf_ad_safe, :_ccdf_ad_safe, :_logcdf_ad_safe, :_logccdf_ad_safe,
            :_split_edge_name, :_is_positional_edge_name, :_next_event_name,
            :_all_positional_event_names, :_split_edge, :_ctor_has_check_args,
            :_collect_unique_boundaries
        ))
end
