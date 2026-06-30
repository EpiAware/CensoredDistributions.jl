module CensoredDistributionsBijectorsExt

# Prior-driven constrained<->unconstrained transform for the LogDensityProblems
# layer. The per-parameter constraint is carried by the prior distribution
# itself, so `bijector(prior)` per row gives the flat transform with no bespoke
# domain table, not from the table's `support` column (the edge's variate
# support, not the parameter's domain). Loaded only when Bijectors is available.

using CensoredDistributions: CensoredDistributions, ComposedLogDensity
using Bijectors: Bijectors, bijector, inverse, with_logabsdet_jacobian

# The per-row inverse bijectors (unconstrained -> constrained), one per flat
# parameter, in table-row order. `bijector(prior)` is the constrained ->
# unconstrained map for that row's prior; its `inverse` is the direction a
# sampler's unconstrained draw is pushed through.
function _inverse_bijectors(prob::ComposedLogDensity)
    priors = CensoredDistributions.flat_priors(prob)
    return map(p -> inverse(bijector(p)), priors)
end

# `to_constrained(prob, z)`: push each unconstrained coordinate through its
# row's inverse bijector, accumulating the log-Jacobian. The transforms are
# univariate (one scalar prior per row), so the flat dimension is unchanged and
# the map is element-wise; the total log-Jacobian is the sum of the per-row
# terms.
function CensoredDistributions.to_constrained(
        prob::ComposedLogDensity, z::AbstractVector)
    binvs = _inverse_bijectors(prob)
    length(z) == length(binvs) || throw(DimensionMismatch(
        "unconstrained vector has length $(length(z)) but $(prob.dist) has " *
        "$(length(binvs)) free parameters"))
    xs_and_logj = map((b, zi) -> with_logabsdet_jacobian(b, zi), binvs, z)
    x = map(first, xs_and_logj)
    logjac = sum(last, xs_and_logj)
    return collect(x), logjac
end

end
