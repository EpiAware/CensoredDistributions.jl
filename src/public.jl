# Public API declarations for Julia 1.11+

# Abstract type hierarchy (public but not exported): the family supertypes
# downstream authors subtype, plus the positional-multi-child intermediate.
# `AbstractOneOf` re-roots under `AbstractComposedDistribution`;
# `AbstractPrimaryCensored` is the core primary-censored family.
public AbstractComposedDistribution, AbstractMultiChild,
       AbstractModifiedDistribution, AbstractPrimaryCensored

# Core distribution types (public but not exported)
public PrimaryCensored
public Latent
public IntervalCensored
public Weighted
public Convolved

# Monotone operational-time warp leaf and its constructor (public but not
# exported): the continuous generalisation of the exported `affine`, giving a
# calendar-time-varying intensity. `TimeChange` is the type, `timechange` the
# constructor. Kept off `export` per the sparse-surface precedent,
# as the exported `affine` covers the common linear case.
public TimeChange, timechange

# Linear chain trick per-step record (public but not exported): the (rate,
# stages) Erlang-stage struct `compartment_stages` returns.
public ChainStage

# Build-once delay PMF for vector evaluation (public but not exported): the
# explicit precomputed-PMF object the caller builds once with `discretise_pmf`
# and reuses across a vector of reference dates / records (the nowcasting path).
public DelayPMF, discretise_pmf

# Primary censoring solver supertype (public but not exported).
# `AnalyticalSolver` and `NumericSolver` are exported in the main module.
public AbstractSolverMethod

# Extension helper for user-defined analytical CDF pairs (public but not exported)
public primarycensored_uniform_cdf_formula

# Composer step/branch/outcome names (public but not exported): used by the
# DynamicPPL extension to key parameter priors by child name.
public component_names

# Hazard accessors from a composed tree (public but not exported): `hazard`,
# `loghazard`, `cumhazard` and `survival` read the hazard surface of any composed
# delay through the verbs. Unexported so they do not clash with
# `SurvivalDistributions.hazard` / `cumhazard` / `loghazard` when both packages
# are loaded; the SurvivalDistributions extension makes the two coincide.
public hazard, loghazard, cumhazard, survival

# Hazard link type (public but not exported): the `(g, g⁻¹)` link pair that
# `hazard_link` builds and the exported `LogLink` / `IdentityLink` / `LogitLink`
# constants are instances of. Marked `public` so the `[`HazardLink`](@ref)`
# cross-references in the `Modified` / `hazard_link` / `modify` docstrings resolve.
public HazardLink

# Composer-node extension contract (public but not exported): the three methods
# a new composer node implements to walk the flat event vector. Reached by the
# qualified name (`CensoredDistributions.child_nleaves` etc.), as the leaf hooks
# `free_leaf` / `rewrap_leaf` are. Documented in
# `docs/src/developer/extending.md`.
public child_nleaves, child_logpdf, child_rand!

# `event_logpdf` is the internal horizon-aware event-vector log density (the
# per-record right-truncation scorer the composed record model and the
# `logpdf(d, rows)` front-door delegate to). It is not part of the public surface:
# `logpdf(d, events)` is the public single-record entry and `logpdf(d, rows)` the
# public table entry. The horizon-kwarg variant stays internal (reached by the
# qualified name); the user-facing right-truncation verb is `truncated`.

# Sample a Resolve outcome and its time as `(name, time)` (public but not
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

# Vectorised latent scoring (public but not exported): the stacked primary
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

# Log-space completeness thinning (public but not exported): the AD-stable
# `logcdf`-based completeness and log-rate thinning helpers.
# Useful for joint offspring scoring but kept off the top-level namespace,
# reached qualified (`CensoredDistributions.log_thin_by_completeness`).
public log_completeness_probability, log_thin_by_completeness

# PPL-neutral LogDensityProblems layer (public but not exported): the
# flat-vector <-> nested-NamedTuple codec (`flatten` / `unflatten` /
# `flat_dimension`, ordered by the `params_table` row walk), the `as_logdensity`
# assembler and the `ComposedLogDensity` spec, plus the seam the weakdep
# extensions extend — the constrained-scale `logdensity` evaluation and the
# prior-driven `to_constrained` transform (its method lives in `BijectorsExt`).
# These are public but not exported: the generic `flatten` / `unflatten` names
# would clash with the `Iterators.flatten` mental model and crowd the top-level
# namespace, so the whole layer is reached by the qualified name
# (`CensoredDistributions.flatten`, `CensoredDistributions.as_logdensity`, ...).
# Marked `public` so they are documented, their docstring `@ref`s resolve, and
# the weakdep extensions import them from a public name.
public flatten, unflatten, flat_dimension, as_logdensity, ComposedLogDensity
public logdensity, to_constrained, free_dimension

# Recurrent / cyclic multi-state accessors (public but not exported). The state-
# graph readers (`transient_states`, `absorbing_states`, `is_absorbing`) and the
# path reader (`visited_states`) inspect a `RecurrentStates` model and its
# `StatePath` realisations; `transition_probability` is the CTMC `exp(Q t)`
# kernel. Panel-data (state-at-visit) and jump-chain scoring both go through the
# exported `logpdf` front door, which dispatches on the observation shape, so
# there is no bespoke panel-scoring name on the public surface.
public transient_states, absorbing_states, is_absorbing, visited_states,
       transition_probability
