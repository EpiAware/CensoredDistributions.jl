# ============================================================================
# Completeness / ascertainment thinning helpers
# ============================================================================
#
# Turing-free, distributions-led helpers for thinning a reproduction number (or
# any rate) by the probability that an event has been observed by a horizon. The
# completeness probability is the CDF of the reporting delay evaluated at the
# observation window: the fraction of events whose delay has elapsed by then.
# Both helpers accept any `UnivariateDistribution`, including a `Convolved`
# chain, so a multi-stage delay thins by `cdf(convolved(...),
# window)` with no special casing.

@doc raw"

Probability that an event is complete (observed) by a horizon.

`completeness_probability(delay, window)` is the cumulative distribution function
of the reporting `delay` evaluated at `window`, ``F_{\text{delay}}(\text{window})``:
the fraction of events whose delay has elapsed by the observation window, i.e.
the ascertainment/completeness probability. `delay` may be any univariate
distribution, including a [`Convolved`](@ref) chain of stages.

# Arguments
- `delay`: the reporting-delay distribution (a `UnivariateDistribution`, e.g. a
  [`Convolved`](@ref) chain).
- `window`: the observation horizon at which to evaluate completeness.

# Examples
```@example
using CensoredDistributions, Distributions

# Single-stage delay
completeness_probability(LogNormal(1.5, 0.5), 7.0)

# Multi-stage (convolved) delay chain
chain = convolved(Gamma(2.0, 1.0), LogNormal(1.0, 0.4))
completeness_probability(chain, 14.0)
```

# See also
- [`thin_by_completeness`](@ref): thin a rate by this probability
"
completeness_probability(delay, window) = cdf(delay, window)

@doc raw"

Log completeness probability at a horizon.

`log_completeness_probability(delay, window)` is the log cumulative distribution
function of the reporting `delay` at `window`,
``\log F_{\text{delay}}(\text{window})``: the log of
[`completeness_probability`](@ref). Working in log space keeps a thinned rate
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
- [`completeness_probability`](@ref): the probability itself
- [`log_thin_by_completeness`](@ref): log-space thinning of a log rate
"
log_completeness_probability(delay, window) = logcdf(delay, window)

@doc raw"

Thin a rate by the completeness probability at a horizon.

`thin_by_completeness(R, delay, window)` scales `R` by the
[`completeness_probability`](@ref) of `delay` at `window`,
``R \cdot F_{\text{delay}}(\text{window})``: the expected observed rate once the
unobserved (not-yet-complete) fraction is removed. Works on a [`Convolved`](@ref)
chain, so a multi-stage delay thins by `R * cdf(convolved(...),
window)`.

# Arguments
- `R`: the rate to thin (e.g. a reproduction number).
- `delay`: the reporting-delay distribution (a `UnivariateDistribution`, e.g. a
  [`Convolved`](@ref) chain).
- `window`: the observation horizon at which to evaluate completeness.

# Examples
```@example
using CensoredDistributions, Distributions

# Thin a reproduction number by single-stage reporting completeness
thin_by_completeness(1.5, LogNormal(1.5, 0.5), 7.0)

# Thin by a convolved delay chain
chain = convolved(Gamma(2.0, 1.0), LogNormal(1.0, 0.4))
thin_by_completeness(1.5, chain, 14.0)
```

# See also
- [`completeness_probability`](@ref): the underlying completeness probability
"
function thin_by_completeness(R, delay, window)
    return R * completeness_probability(delay, window)
end

@doc raw"

Thin a log rate by the log completeness probability at a horizon.

`log_thin_by_completeness(log_R, delay, window)` returns
``\log R + \log F_{\text{delay}}(\text{window})``, the log of the thinned rate
``R \cdot F_{\text{delay}}(\text{window})`` computed entirely in log space. This
is the AD-stable form for a joint offspring model: a negative-binomial mean set
to `exp(log_thin_by_completeness(...))` stays strictly positive even when the
completeness underflows, so the success probability never reaches the `0`/`1`
boundary and the reverse-mode gradient stays finite. The plain
[`thin_by_completeness`](@ref) multiplies in linear space and can drive the rate
to exactly zero, which collapses the offspring mean to a degenerate point mass.

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
- [`thin_by_completeness`](@ref): linear-space thinning of a rate
- [`log_completeness_probability`](@ref): the log completeness probability
"
function log_thin_by_completeness(log_R, delay, window)
    return log_R + log_completeness_probability(delay, window)
end
