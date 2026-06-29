# ============================================================================
# Abstract type hierarchy: the primary-censored family
# ============================================================================
#
# `AbstractPrimaryCensored` is the supertype the package dispatches on for the
# primary-censored family — the primary-event-censored delay `PrimaryCensored`
# and the latent `PrimaryConditional` (the secondary conditioned on a realised
# primary). Both are univariate and continuous, so the supertype is non-
# parametric (mirroring the `AbstractOneOf` precedent on the integration line).
# Concrete types subtype it; shared interface behaviour and the documented
# contract hang off the abstract.

"""
    AbstractPrimaryCensored

Supertype of the primary-censored family the package dispatches on:
`PrimaryCensored` (the primary-event-censored delay) and the latent
`PrimaryConditional` (the secondary conditioned on a realised primary).
Univariate and continuous, so non-parametric.

Required of a concrete subtype:

- `get_dist(d)` — the underlying delay distribution;
- `params(d)`;
- `logpdf(d, x)` finite on its support;
- `Base.show(io, d)`.

Verify a subtype with
`CensoredDistributions.TestUtils.test_primary_censored_interface`.
"""
abstract type AbstractPrimaryCensored <: UnivariateDistribution{Continuous} end
