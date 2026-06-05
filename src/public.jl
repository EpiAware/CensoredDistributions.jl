# Public API declarations for Julia 1.11+

# Core distribution types (public but not exported)
public PrimaryCensored
public IntervalCensored
public Weighted
public Convolved
public SequentialDistribution

# Latent-formulation internals (public but not exported)
public BoundedPrimary
public LatentIntervalCensored

# Primary censoring utilities and solver methods (public but not exported)
public AbstractSolverMethod
public AnalyticalSolver
public NumericSolver

# Formulation method types (Marginal/Latent exported; Auto default and the
# abstract type public but not exported)
public AbstractPCMethod
public Auto

# Extension helper for user-defined analytical CDF pairs (public but not exported)
public primarycensored_uniform_cdf_formula
