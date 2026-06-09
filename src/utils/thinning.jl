# ============================================================================
# Completeness / ascertainment thinning helpers
# ============================================================================
#
# Turing-free, distributions-led helpers for thinning a reproduction number (or
# any rate) by the probability that an event has been observed by a horizon. The
# completeness probability is the CDF of the reporting delay evaluated at the
# observation window: the fraction of events whose delay has elapsed by then.
# Both helpers accept any `UnivariateDistribution`, including a `Convolved`
# chain, so a multi-stage delay thins by `cdf(convolve_distributions(...),
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
chain = convolve_distributions(Gamma(2.0, 1.0), LogNormal(1.0, 0.4))
completeness_probability(chain, 14.0)
```

# See also
- [`thin_by_completeness`](@ref): thin a rate by this probability
"
completeness_probability(delay, window) = cdf(delay, window)

@doc raw"

Thin a rate by the completeness probability at a horizon.

`thin_by_completeness(R, delay, window)` scales `R` by the
[`completeness_probability`](@ref) of `delay` at `window`,
``R \cdot F_{\text{delay}}(\text{window})``: the expected observed rate once the
unobserved (not-yet-complete) fraction is removed. Works on a [`Convolved`](@ref)
chain, so a multi-stage delay thins by `R * cdf(convolve_distributions(...),
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
chain = convolve_distributions(Gamma(2.0, 1.0), LogNormal(1.0, 0.4))
thin_by_completeness(1.5, chain, 14.0)
```

# See also
- [`completeness_probability`](@ref): the underlying completeness probability
"
function thin_by_completeness(R, delay, window)
    return R * completeness_probability(delay, window)
end
