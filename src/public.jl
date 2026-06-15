# Public API declarations for Julia 1.11+

# Core distribution types (public but not exported)
public PrimaryCensored
public Latent
public IntervalCensored
public Weighted
public Convolved

# Linear chain trick per-step record (public but not exported): the (rate,
# stages) Erlang-stage struct `linear_chain_stages` returns.
public ChainStage

# Build-once delay PMF for vector evaluation (public but not exported): the
# explicit precomputed-PMF object the caller builds once with `discretise_pmf`
# and reuses across a vector of reference dates / records (the nowcasting path).
public DelayPMF, discretise_pmf

# Primary censoring utilities and solver methods (public but not exported)
public AbstractSolverMethod
public AnalyticalSolver
public NumericSolver

# Extension helper for user-defined analytical CDF pairs (public but not exported)
public primarycensored_uniform_cdf_formula

# Composer step/branch/outcome names (public but not exported): used by the
# DynamicPPL extension to key parameter priors by child name.
public component_names

# Horizon-aware event-vector log density (public but not exported): the per-record
# right-truncation entry point used by the composed record model.
public event_logpdf

# Sample a Competing outcome and its time as `(name, time)` (public but not
# exported): the outcome-retaining draw used by full-path tree simulation.
public rand_outcome

# Per-record composed distributions for vectorised scoring + sampling (public but
# not exported): the assembly entry that bakes per-record metadata and shares the
# segment construction across records.
public record_distributions, EventRecord

# Grouped per-stratum scoring (public but not exported): the varying-parameter
# primitive scoring records grouped by an integer stratum id, each stratum's
# records built from its own (possibly partially-pooled) composed distribution.
public batched_event_logpdf

# Vectorised LATENT scoring (public but not exported): the stacked primary
# priors and the vectorised observed conditional that express the latent table
# as a `primaries ~ product_distribution(...)` plus `@addlogprob! ...` pair.
public latent_primary_priors, latent_observed_logpdf

# Pluggable integration: the default solver, the entry point, and the
# quadrature helper (public but not exported). `GaussLegendre` stays
# unexported to avoid clashing with `Integrals.GaussLegendre` when both
# are loaded; the Integrals.jl extension adds an `integrate` method.
public GaussLegendre, integrate, gl_integrate

# Public interface-conformance harness submodule (public but not exported):
# `TestUtils.test_interface(d)` lets a downstream author verify a new leaf /
# composer against the package's interface checklist.
public TestUtils

# The Tables.jl column table `params_table` returns (public but not exported): a
# Tables.jl source that prints as a padded table and forwards column access.
public ParamsTable
