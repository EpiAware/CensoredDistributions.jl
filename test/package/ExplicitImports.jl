@testitem "ExplicitImports analysis" tags=[:quality] begin
    using ExplicitImports
    using CensoredDistributions

    # Use the testing-specific functions from ExplicitImports.jl
    # These are designed to be used in test suites and give clear pass/fail results

    # Test that there are no stale explicit imports
    @test check_no_stale_explicit_imports(CensoredDistributions) === nothing

    # Test that there are no implicit imports
    @test check_no_implicit_imports(CensoredDistributions) === nothing

    # Test that all explicit imports are public in their source modules
    # Allow non-public imports for internal functions we need to use
    @test check_all_explicit_imports_are_public(CensoredDistributions;
        # Skip check for internal/non-public functions that we need to use:
        # - Censored: Used in get_dist.jl for Distributions.jl compatibility
        # - _in_closed_interval: Internal Distributions utility used for bounds checking
        # - _gamma_cdf, _grad_p_a_series: Internal AD-safe gamma CDF used by the
        #   ChainRulesCore and ForwardDiff extensions
        # - Dual, value, partials: ForwardDiff internals used by the
        #   ForwardDiff extension to construct Dual return values; no public
        #   alternative for the Dual reconstruction pattern.
        # - @grad_from_chainrules, TrackedReal: ReverseDiff internals used by
        #   the ReverseDiff extension; @grad_from_chainrules is the standard
        #   ChainRules-to-ReverseDiff bridge macro and TrackedReal is the
        #   public-by-convention tape value type, neither marked `public`.
        ignore = (
            :Censored, :_in_closed_interval, :_gamma_cdf, :_grad_p_a_series,
            :Dual, :value, :partials,
            Symbol("@grad_from_chainrules"), :TrackedReal
        )
    ) === nothing

    # Test that all explicit imports come from the owning module
    # This should pass now that minimum/maximum are imported from Base
    @test check_all_explicit_imports_via_owners(CensoredDistributions) === nothing
end
