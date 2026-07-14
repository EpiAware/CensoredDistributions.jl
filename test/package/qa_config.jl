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
    # extensions legitimately import, carried over verbatim from the previous
    # bespoke ExplicitImports check:
    #   - Censored: used in get_dist.jl for Distributions.jl compatibility.
    #   - _in_closed_interval: internal Distributions bounds helper.
    #   - _gamma_cdf, _grad_p_a_series, _gamma_cdf_value_and_partials: internal
    #     AD-safe gamma CDF helpers used by the ChainRulesCore/ForwardDiff exts.
    #   - Dual, value, partials: ForwardDiff internals the ForwardDiff ext uses
    #     to reconstruct Dual return values (no public alternative).
    #   - @grad_from_chainrules, TrackedReal: ReverseDiff internals (the standard
    #     ChainRules->ReverseDiff bridge macro and tape value type).
    #   - _collect_unique_boundaries: internal boundary builder the AD exts mark
    #     non-differentiable (zero-tangent rule).
    ei_ignore = (
        :Censored, :_in_closed_interval, :_gamma_cdf, :_grad_p_a_series,
        :_gamma_cdf_value_and_partials,
        :Dual, :value, :partials,
        Symbol("@grad_from_chainrules"), :TrackedReal,
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

    # Per-extension ambiguity checks. Package-level method ambiguities are already
    # covered by Aqua above; the kit's per-extension surface check is left empty
    # here because the extension-specific fixes it asserts (the ForwardDiff #672
    # and DynamicPPL #673 collapses) live on the feature stack, not on `main`.
    # Populate this as those land on main (tracked as a docs/QA follow-up).
    extensions = ()
)
