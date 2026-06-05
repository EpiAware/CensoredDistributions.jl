# ============================================================================
# Path ascertainment / completeness thinning (#300).
#
# A delay chain is only ascertained (observed) once every event in it has
# happened by the observation horizon. The probability of that is the CDF of the
# total delay across the chain evaluated at the remaining window, i.e. the
# probability the chain completes in time. The andv completeness model uses this
# to thin offspring counts: an offspring's effective reproduction number is
# `R_eff = R * p` where `p` is the probability the offspring's whole infection
# chain has completed (and so been counted) by the horizon.
#
# This is a multiplicative thinning of a RATE or COUNT, not a re-weighting of a
# log density, so it is kept distinct from `weight` (which scales a log density
# by an observation count). The helpers here return the completeness probability
# `p` and the thinned quantity `R * p`, composing `convolve_distributions` for
# the chain delay and `cdf` for the completion probability, with nothing
# PPL-specific.
# ============================================================================

@doc """

Probability that a delay chain completes by an observation window.

The completeness probability is `cdf(delay, window)`: the probability that the
total delay across the chain is at most `window`, so every event in the chain
has happened (and the chain has been ascertained) by the observation horizon.
Pass a single delay for a one-segment chain, or a [`Convolved`](@ref) (from
[`convolve_distributions`](@ref)) for the total delay across several segments
whose intermediate events are not separately observed.

This is the andv completeness factor `p`: with `window = horizon - origin` the
returned probability is the chance an event chain starting at `origin` has
completed by `horizon`, the quantity offspring counts are thinned by in
[`thin_by_completeness`](@ref).

# Arguments
- `delay`: the chain delay distribution. A single delay gives a one-segment
  chain; a [`Convolved`](@ref) gives the total delay across several segments.
- `window`: the observation window `horizon - origin`. A non-positive window
  returns probability zero (the horizon has not passed the chain origin).

# Examples
```@example
using CensoredDistributions, Distributions

# One-segment chain: probability the delay is at most the window.
incubation = LogNormal(1.5, 0.5)
p_single = completeness_probability(incubation, 6.0)

# Two-segment chain whose intermediate event is not observed: probability the
# total delay across both segments completes by the window.
generation = Gamma(2.0, 1.0)
chain = convolve_distributions(incubation, generation)
p_chain = completeness_probability(chain, 6.0)
```

# See also
- [`thin_by_completeness`](@ref): thin a count or rate by this probability
- [`convolve_distributions`](@ref): builds the chain delay
- [`truncate_to_horizon`](@ref): the right-truncation companion for delays
"""
function completeness_probability(delay::UnivariateDistribution, window::Real)
    window <= zero(window) && return zero(float(window))
    return cdf(delay, window)
end

@doc """

Thin a count or rate by the probability its delay chain has completed.

Returns `quantity * completeness_probability(delay, window)`: the andv
completeness thinning `R_eff = R * p`, where `p` is the probability the delay
chain has completed (and so been ascertained) by the observation window. Use it
to map a latent reproduction number, offspring count, or rate to the value
consistent with what has been observed by the horizon, so an incompletely
observed chain is down-weighted by exactly its chance of having completed.

The thinning is multiplicative on the quantity itself, distinct from
[`weight`](@ref), which scales a log density by an observation count. Both the
quantity and the delay parameters flow through, so the thinned value is safe to
differentiate.

# Arguments
- `quantity`: the count or rate to thin (for example a reproduction number `R`).
- `delay`: the chain delay distribution, as in [`completeness_probability`](@ref).
- `window`: the observation window `horizon - origin`.

# Examples
```@example
using CensoredDistributions, Distributions

R = 2.5
generation = convolve_distributions(LogNormal(1.5, 0.5), Gamma(2.0, 1.0))

# Effective R after thinning by the chain-completeness probability.
R_eff = thin_by_completeness(R, generation, 6.0)
```

# See also
- [`completeness_probability`](@ref): the completeness factor `p`
- [`convolve_distributions`](@ref): builds the chain delay
"""
function thin_by_completeness(
        quantity::Real, delay::UnivariateDistribution, window::Real)
    return quantity * completeness_probability(delay, window)
end
