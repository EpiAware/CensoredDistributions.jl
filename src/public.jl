# Public API declarations for Julia 1.11+

# Core distribution types (public but not exported)
public PrimaryCensored
public ParallelPrimaryCensored
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

# Formulation method types (Latent is exported; the default marginal tag and the
# abstract type are public but not exported)
public AbstractPCMethod

# Extension helper for user-defined analytical CDF pairs (public but not exported)
public primarycensored_uniform_cdf_formula
