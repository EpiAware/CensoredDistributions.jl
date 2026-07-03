# Public API declarations for Julia 1.11+

# Core distribution types (public but not exported)
public PrimaryCensored
public IntervalCensored
public Weighted
public Convolved

# `PrimaryConditional` and `PrimaryEvent` are exported from the main module (the
# latent-form observation distributions). `_SecondaryConditional`, the internal
# per-record kernel underneath them, stays private.

# The primary-censored family supertype (public but not exported).
public AbstractPrimaryCensored

# Primary censoring solver supertype (public but not exported).
# `AnalyticalSolver` and `NumericSolver` are exported in the main module.
public AbstractSolverMethod

# Extension helper for user-defined analytical CDF pairs (public but not exported)
public primarycensored_uniform_cdf_formula

# Pluggable integration: the default solver, the entry point, and the
# quadrature helper (public but not exported). `GaussLegendre` stays
# unexported to avoid clashing with `Integrals.GaussLegendre` when both
# are loaded; the Integrals.jl extension adds an `integrate` method.
public GaussLegendre, integrate, gl_integrate
