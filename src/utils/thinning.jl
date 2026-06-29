# ============================================================================
# Completeness / ascertainment thinning helpers (log space)
# ============================================================================
#
# Turing-free, distributions-led helpers for thinning a reproduction number (or
# any rate) by the probability that an event has been observed by a horizon. The
# completeness probability is the CDF of the reporting delay at the observation
# window, `cdf(delay, window)`: the fraction of events whose delay elapsed by
# then. `delay` may be any `UnivariateDistribution`, including a `Convolved`
# chain, so a multi-stage delay thins by `cdf(convolved(...), window)` with no
# special casing. Linear-space completeness is `cdf(delay, window)` and linear-
# space thinning `R * cdf(delay, window)`; the log-space helpers below are the
# AD-stable forms kept public.

@doc raw"

Log completeness probability at a horizon.

`log_completeness_probability(delay, window)` is the log cumulative distribution
function of the reporting `delay` at `window`,
``\log F_{\text{delay}}(\text{window})``: the log of the completeness
probability `cdf(delay, window)`. Working in log space keeps a thinned rate
strictly positive when the completeness underflows, which is what the
[`log_thin_by_completeness`](@ref) joint scoring relies on: a completeness near
zero gives a large-magnitude negative log, not a rate of exactly zero that would
collapse a negative-binomial offspring mean to a degenerate point mass (and a
`NaN` gradient under reverse-mode AD).

# Arguments
- `delay`: the reporting-delay distribution (a `UnivariateDistribution`, e.g. a
  [`Convolved`](@ref) chain).
- `window`: the observation horizon at which to evaluate completeness.

# Examples
```@example
using CensoredDistributions, Distributions

# Not exported; reach it by its qualified name.
import CensoredDistributions: log_completeness_probability

log_completeness_probability(LogNormal(1.5, 0.5), 7.0)

chain = convolved(Gamma(2.0, 1.0), LogNormal(1.0, 0.4))
log_completeness_probability(chain, 14.0)
```

# See also
- [`log_thin_by_completeness`](@ref): log-space thinning of a log rate
"
log_completeness_probability(delay, window) = logcdf(delay, window)

@doc raw"

Thin a log rate by the log completeness probability at a horizon.

`log_thin_by_completeness(log_R, delay, window)` returns
``\log R + \log F_{\text{delay}}(\text{window})``, the log of the thinned rate
``R \cdot F_{\text{delay}}(\text{window})`` computed entirely in log space. This
is the AD-stable form for a joint offspring model: a negative-binomial mean set
to `exp(log_thin_by_completeness(...))` stays strictly positive even when the
completeness underflows, so the success probability never reaches the `0`/`1`
boundary and the reverse-mode gradient stays finite. Multiplying in linear space
(`R * cdf(delay, window)`) can drive the rate to exactly zero, which collapses
the offspring mean to a degenerate point mass.

# Arguments
- `log_R`: the log of the rate to thin (e.g. a log reproduction number).
- `delay`: the reporting-delay distribution (a `UnivariateDistribution`, e.g. a
  [`Convolved`](@ref) chain).
- `window`: the observation horizon at which to evaluate completeness.

# Examples
```@example
using CensoredDistributions, Distributions

# Not exported; reach it by its qualified name.
import CensoredDistributions: log_thin_by_completeness

log_thin_by_completeness(log(1.5), LogNormal(1.5, 0.5), 7.0)

chain = convolved(Gamma(2.0, 1.0), LogNormal(1.0, 0.4))
log_thin_by_completeness(log(1.5), chain, 14.0)
```

# See also
- [`log_completeness_probability`](@ref): the log completeness probability
"
function log_thin_by_completeness(log_R, delay, window)
    return log_R + log_completeness_probability(delay, window)
end
