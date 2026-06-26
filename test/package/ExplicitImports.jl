@testitem "ExplicitImports analysis" tags=[:quality] begin
    using EpiAwarePackageTools: test_explicit_imports
    using CensoredDistributions

    # The four ExplicitImports conformance checks (no stale explicit imports,
    # no implicit imports, all explicit imports public in their source module,
    # all explicit imports via their owning module) run through the shared kit
    # wrapper. CD's non-public-import `ignore` tuple stays here as a package
    # fixture, with its per-symbol justifications:
    #
    # - Censored: used in get_dist.jl for Distributions.jl compatibility.
    # - _in_closed_interval: internal Distributions utility used for bounds
    #   checking.
    # - _gamma_cdf, _grad_p_a_series, _gamma_cdf_value_and_partials: internal
    #   AD-safe gamma CDF helpers used by the ChainRulesCore and ForwardDiff
    #   extensions.
    # - Dual, value, partials: ForwardDiff internals used by the ForwardDiff
    #   extension to construct Dual return values; no public alternative for
    #   the Dual reconstruction pattern.
    # - @grad_from_chainrules, TrackedReal: ReverseDiff internals used by the
    #   ReverseDiff extension; @grad_from_chainrules is the standard
    #   ChainRules-to-ReverseDiff bridge macro and TrackedReal is the
    #   public-by-convention tape value type, neither marked `public`.
    test_explicit_imports(CensoredDistributions;
        ignore = (
            :Censored, :_in_closed_interval, :_gamma_cdf, :_grad_p_a_series,
            :_gamma_cdf_value_and_partials,
            :Dual, :value, :partials,
            Symbol("@grad_from_chainrules"), :TrackedReal
        ))
end
