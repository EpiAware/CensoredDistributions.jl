# Public API declarations for Julia 1.11+

# Core distribution types (public but not exported)
public PrimaryCensored
public ParallelPrimaryCensored
public IntervalCensored
public Weighted
public Convolved
public SequentialDistribution
public EventTree
public EventEdge

# Competing-outcome (disjunctive) node MixtureModel lowering. `Competing` is
# exported; `as_mixture` is public but not exported.
public as_mixture

# Marginal/Latent primary-censored formulation types (public but not exported).
# The resolved `mode` selects between these when building a `primary_censored`
# delay; the DynamicPPL extension dispatches its submodels on them.
public MarginalPrimaryCensored
public LatentPrimaryCensored

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
