# Public API declarations for Julia 1.11+

# Core distribution types (public but not exported)
public PrimaryCensored
public IntervalCensored
public Weighted
public Convolved

# Solver supertype is ConvolvedDistributions.AbstractSolverMethod, re-exported
# (not defined locally); `AnalyticalSolver`/`NumericSolver` are exported from
# the main module with their ConvolvedDistributions definitions.
