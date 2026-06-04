# Public API declarations for Julia 1.11+

# Core distribution types (public but not exported)
public PrimaryCensored
public IntervalCensored
public Weighted
public Convolved

# Internal bounded-primary prior (public but not exported); folded former
# WithinWindowPrimary used by the coupled Latent path (#299)
public BoundedPrimary

# Primary censoring utilities and solver methods (public but not exported)
public AbstractSolverMethod
public AnalyticalSolver
public NumericSolver

# Formulation method types (exported as Marginal/Latent; abstract type public)
public AbstractFormulation

# Extension helper for user-defined analytical CDF pairs (public but not exported)
public primarycensored_uniform_cdf_formula
