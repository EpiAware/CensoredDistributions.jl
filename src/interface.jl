# ============================================================================
# Abstract type: the primary-censored family the package dispatches on
# ============================================================================

"""
    AbstractPrimaryCensored

Supertype of the primary-censored family the package dispatches on:
`PrimaryCensored` (the primary-event-censored delay) and the latent
`PrimaryConditional` (the secondary conditioned on a realised primary). Core CD,
distinct from interval censoring, and it stays in the package. Univariate and
continuous, so non-parametric.

Required of a concrete subtype:

- `get_dist(d)` — the underlying delay distribution;
- `params(d)`;
- `logpdf(d, x)` finite on its support;
- `Base.show(io, d)`.

`double_interval_censored` is a constructor function, not a type: it returns a
pipeline (`interval_censored(truncated(primary_censored(...)))`) whose object
type is the outer wrapper, so there is no `DoubleIntervalCensored` type to place
under this supertype.
"""
abstract type AbstractPrimaryCensored <: UnivariateDistribution{Continuous} end
