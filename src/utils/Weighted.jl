@doc "

A distribution wrapper that applies a weight to the log-probability of an
underlying distribution.
This is primarily used where observations have associated counts or weights.

Only the `logpdf` method is affected by the weight - all other methods (pdf,
cdf, sampling, etc.) delegate directly to the underlying distribution.

# Weight Types Supported

The `Weighted` struct supports three different weight scenarios:

1. **Real weights**: Constructor weight is a specific value (e.g., `2.5`)
2. **Missing weights**: Constructor weight is `missing`, allowing weights to be
   provided at observation time via joint observations `(value = x, weight = w)`
3. **Zero weights**: Handled specially to return `-Inf` and avoid NaN from
   `0 * -Inf`

# Examples
```@example
using CensoredDistributions, Distributions

# Single weighted observation
d = LogNormal(1.5, 0.5)
wd = weight(d, 10.0)  # Observation with weight/count of 10

# Weighted log-probability calculation
observed_value = 2.0
weighted_logpdf = logpdf(wd, observed_value)

# Compare with manual calculation
manual_logpdf = 10.0 * logpdf(d, observed_value)
# weighted_logpdf ≈ manual_logpdf
```
"
struct Weighted{D <: UnivariateDistribution, T <: Union{Real, Missing}} <:
       AbstractModifiedDistribution{Univariate, ValueSupport}
    "The underlying distribution being weighted."
    dist::D
    "The weight to apply to log-probabilities."
    weight::T

    function Weighted(
            dist::D, weight::T) where {
            D <: UnivariateDistribution, T <: Union{Real, Missing}}
        if !ismissing(weight) && weight < 0
            throw(ArgumentError("Weight must be non-negative"))
        end
        new{D, T}(dist, weight)
    end
end

# ============================================================================
# Deprecation: the weighting surface moves to ModifiedDistributions.jl
# ============================================================================

# `weight` / `Weighted` will move to the standalone ModifiedDistributions.jl
# package; the constructors emit a deprecation warning under `--depwarn=yes` so
# callers can migrate ahead of removal, and the `Weighted` type stays reachable
# (it is `public`) meanwhile.
#
# The warning fires at most once per session, gated on a module-level flag,
# because the `weight` constructor runs inside differentiated/sampled closures:
# `Base.depwarn` walks a `backtrace()` and reads the world counter on every
# call, which under `--depwarn=yes` (forced by `Pkg.test`) stalls a NUTS fit.
# The cached flag makes every call after the first a single `Bool` read, and the
# first call is shielded from Enzyme by an `EnzymeRules.inactive` rule in the
# Enzyme extension.
const _WEIGHT_DEPRECATION_WARNED = Ref(false)

function _weight_deprecation()
    _WEIGHT_DEPRECATION_WARNED[] && return nothing
    _WEIGHT_DEPRECATION_WARNED[] = true
    Base.depwarn(
        "`weight` is deprecated and will move to the standalone " *
        "ModifiedDistributions.jl package in a future breaking release; it " *
        "still works for now. Track the migration in CensoredDistributions.jl " *
        "issue #128.",
        :weight)
    return nothing
end

# ============================================================================
# Constructor Functions
# ============================================================================

@doc "

Create a weighted distribution where the log-probability is scaled by `w`.

!!! warning \"Deprecated\"
    `weight` is deprecated and will move to the standalone
    ModifiedDistributions.jl package in a future breaking release. It still
    works for now and emits a deprecation warning under `--depwarn=yes`.

A `Weighted` distribution will contribute `w * logpdf(dist, x)` to the
log-probability when evaluating `logpdf(weighted_dist, x)`.

# Examples
```@example
# For aggregated count data
y_obs = 3.5  # Observed value
n_count = 25  # Number of times this value was observed

d = Normal(2.0, 1.0)
weighted_d = weight(d, n_count)

# Weighted log-probability calculation
weighted_logpdf = logpdf(weighted_d, y_obs)
# equivalent to: n_count * logpdf(d, y_obs)
```
"
function weight(dist::UnivariateDistribution, w::Real)
    _weight_deprecation()
    return Weighted(dist, w)
end

@doc "

Return the distribution unweighted when the weight is `nothing`.

This lets callers thread an optional weight through `weight(dist, w)` and pass
`nothing` to mean \"no weight\": the distribution is returned unchanged so a
`~ weight(dist, nothing)` statement scores the plain `logpdf`.

# Examples
```@example
using CensoredDistributions, Distributions

d = Normal(2.0, 1.0)
weight(d, nothing) === d
```
"
function weight(dist::UnivariateDistribution, ::Nothing)
    _weight_deprecation()
    return dist
end

@doc "

Create a product distribution of weighted distributions, each with a different
weight.

A `Product` distribution of `Weighted` distributions suitable for vectorized
observations.

# Arguments
- `dist`: The univariate distribution to be replicated and weighted for each observation
- `weights`: Vector of weights to apply to each copy of the distribution

# Examples
```@example
y_obs = [3.5, 4.2, 3.8]  # Observed values
n_counts = [25, 10, 15]  # Counts for each observation

d = Normal(2.0, 1.0)
weighted_dists = weight(d, n_counts)

# Weighted log-probability calculation
weighted_logpdf = logpdf(weighted_dists, y_obs)
# equivalent to: sum(n_counts .* logpdf.(d, y_obs))
```

# See also
- [`Weighted`](@ref): The underlying weighted distribution type
"
function weight(dist::UnivariateDistribution, weights::AbstractVector{<:Real})
    _weight_deprecation()
    return product_distribution([Weighted(dist, w) for w in weights])
end

@doc "

Create a product distribution of weighted distributions, where each
distribution has its own weight.

A `Product` distribution of `Weighted` distributions suitable for vectorized
observations with different distributions.

# Examples
```@example
using CensoredDistributions, Distributions

y_obs = [3.5, 4.2, 3.8]  # Observed values
dists = [Normal(2.0, 0.5), Normal(2.5, 0.8), Normal(1.8, 0.6)]
n_counts = [25, 10, 15]  # Counts for each observation
weighted_dists = weight(dists, n_counts)
```
"
function weight(
        dists::AbstractVector{<:UnivariateDistribution},
        weights::AbstractVector{<:Real})
    _weight_deprecation()
    length(dists) == length(weights) ||
        throw(
            ArgumentError(
            "Number of distributions must equal number of weights"
        )
        )
    return product_distribution(
        [Weighted(d, w) for (d, w) in zip(dists, weights)]
    )
end

@doc "

Create a weighted distribution with missing constructor weight.

Useful for creating distributions where weights will be provided at observation
time. Uses `missing` as constructor weight, enabling observation weight to be
used directly.

# Examples
```@example
using CensoredDistributions, Distributions

d = Normal(2.0, 0.5)

# Create weighted distribution with missing constructor weight
weighted_dist = weight(d)

# Weight provided at observation time via joint observations
logpdf(weighted_dist, (value = 3.5, weight = 25))
```
"
function weight(dist::UnivariateDistribution)
    _weight_deprecation()
    return Weighted(dist, missing)
end

@doc "

Create a product distribution of weighted distributions with missing constructor
weights.

Useful for creating distributions where weights will be provided at observation
time. Each distribution uses `missing` as constructor weight, enabling
observation weight to be used directly.

# Examples
```@example
using CensoredDistributions, Distributions

y_obs = [3.5, 4.2, 3.8]  # Observed values
dists = [Normal(2.0, 0.5), Normal(2.5, 0.8), Normal(1.8, 0.6)]

# Create weighted distributions with missing constructor weights
weighted_dists = weight(dists)

# Weights provided at observation time via joint observations
logpdf(weighted_dists, (values = y_obs, weights = [25, 10, 15]))
```
"
function weight(dists::AbstractVector{<:UnivariateDistribution})
    _weight_deprecation()
    return product_distribution(
        [Weighted(d, missing) for d in dists]
    )
end

# ============================================================================
# Distributions.jl Interface - Basic Properties
# ============================================================================

# Weight doesn't affect element type
Base.eltype(::Type{<:Weighted{D, T}}) where {D, T} = eltype(D)
minimum(d::Weighted) = minimum(get_dist(d))
maximum(d::Weighted) = maximum(get_dist(d))
insupport(d::Weighted, x::Real) = insupport(get_dist(d), x)
params(d::Weighted) = (params(get_dist(d))..., d.weight)

# ============================================================================
# Probability Density Functions
# ============================================================================

@doc "

Return the probability density from the underlying distribution (unweighted).

See also: [`logpdf`](@ref)
"
function pdf(d::Weighted, x::Real)
    return pdf(get_dist(d), x)
end

# Helper function for weighted logpdf computation with proper validation
function _logpdf(dist, value, weight)
    # Handle missing or zero weights to avoid NaN from 0 * -Inf operations.
    # Seed the sentinel from the promoted weight/distribution type so
    # ForwardDiff Duals survive the zero-weight branch (without evaluating
    # the underlying logpdf on the short-circuit path).
    if ismissing(weight) || weight == 0
        wtype = ismissing(weight) ? Bool : typeof(weight)
        T = float(promote_type(wtype, eltype(dist)))
        return oftype(zero(T), -Inf)
    end
    return weight * logpdf(dist, value)
end

@doc "

Return the weighted log-probability for scalar observations.

See also: [`pdf`](@ref)
"
function logpdf(d::Weighted, x::Real)
    return _logpdf(get_dist(d), x, d.weight)
end

@doc "

Return the weighted log-probability for joint observations as NamedTuple.

Combines constructor weight with observation weight via multiplication.
Expected format: `(value = x, weight = w)`.

See also: [`pdf`](@ref)
"
function logpdf(d::Weighted, obs::NamedTuple{(:value, :weight)})
    final_weight = combine_weights(d.weight, obs.weight)
    return _logpdf(get_dist(d), obs.value, final_weight)
end

# ============================================================================
# Product Distribution logpdf Methods
# ============================================================================

# Helper function for Product logpdf computation
function _logpdf_product(
        d::Product{<:ValueSupport, <:Weighted, <:AbstractVector{<:Weighted}},
        values, obs_weights)
    # Base (unweighted) logpdfs, one per component. When every component shares
    # the same underlying distribution and that distribution has a cached-CDF
    # batched `logpdf` (the `weight(dist, weights)` aggregation pattern over an
    # `IntervalCensored`: many duplicate observation/window combinations against
    # one distribution), these are scored in a single vectorised `logpdf(dist, x)`
    # call rather than a per-observation loop, reusing the batched PDF
    # methods. The vectorised call returns the same values as the per-element
    # loop (it just caches shared CDF evaluations), so the weighted sum is
    # numerically identical; the loop fallback (`_weighted_base_logpdfs`) keeps
    # every other case correct.
    logpdfs = _weighted_base_logpdfs(d.v, values)
    constructor_weights = [wd.weight for wd in d.v]

    # Combine weights and compute final result
    final_weights = combine_weights(constructor_weights, obs_weights)
    # Handle missing or zero weights - return -Inf if any found
    if any(ismissing, final_weights) || any(w -> w == 0, final_weights)
        return -Inf
    end
    return sum(final_weights .* logpdfs)
end

# The unweighted base logpdfs of a vector of `Weighted` components at `values`.
# The vectorised batched `logpdf(dist, values)` call is taken only when every
# component wraps the same distribution and that distribution has a specialised
# batched `logpdf` over a vector of scalar observations (the cached-CDF
# `IntervalCensored` path, `_has_batched_logpdf`); otherwise the per-component
# loop is used. Both conditions are on the data (the wrapped distribution's type
# and object identity), never on a sampled parameter value, so the branch is
# AD-safe; and gating on `_has_batched_logpdf` keeps a plain distribution (no
# batched method, no CDF-cache win) on the loop, which avoids routing it through
# the deprecated/AD-hostile generic `logpdf(d, ::AbstractVector)`.
function _weighted_base_logpdfs(components, values)
    shared = _shared_weighted_dist(components)
    (shared === nothing || !_has_batched_logpdf(shared)) &&
        return [logpdf(wd.dist, v) for (wd, v) in zip(components, values)]
    return logpdf(shared, collect(values))
end

# Whether a distribution provides a specialised, value-identical batched
# `logpdf(d, ::AbstractVector{<:Real})` over a vector of scalar observations that
# is worth a single vectorised call. Only `IntervalCensored` does (it caches the
# shared interval CDFs); everything else falls back to the per-observation
# loop. Composer `logpdf(::AbstractVector)` methods score a single multivariate
# event, not a batch, so they are deliberately excluded.
_has_batched_logpdf(::UnivariateDistribution) = false
_has_batched_logpdf(::IntervalCensored) = true

# The single underlying distribution shared by every `Weighted` component, or
# `nothing` when they are not all the same object. Identity (`===`) keeps the
# branch data-driven: it depends only on which distribution object each
# component wraps, never on a (possibly `Dual`/sampled) parameter value, so it is
# AD-safe and matches the `weight(dist, weights)` aggregation pattern (one shared
# `dist`, many weights). An empty vector has no shared distribution.
function _shared_weighted_dist(components)
    isempty(components) && return nothing
    first_dist = first(components).dist
    for wd in components
        wd.dist === first_dist || return nothing
    end
    return first_dist
end

@doc "

Efficient vectorised log-probability computation for
Product{<:ValueSupport, <:Weighted} with joint observations.

Handles joint observations and weight stacking.
Expected format: `(values = [...], weights = [...])`.

See also: [`logpdf`](@ref)
"
function logpdf(
        d::Product{<:ValueSupport, <:Weighted, <:AbstractVector{<:Weighted}},
        obs::NamedTuple{(:values, :weights)})
    return _logpdf_product(d, obs.values, obs.weights)
end

@doc "

Efficient vectorised log-probability computation for
Product{<:ValueSupport, <:Weighted} with vector observations.

See also: [`logpdf`](@ref)
"
function logpdf(
        d::Product{<:ValueSupport, <:Weighted, <:AbstractVector{<:Weighted}},
        x::AbstractVector{<:Real})
    return _logpdf_product(d, x, missing)  # No observation weights
end

# ============================================================================
# Log-likelihood Methods for Weighted Distributions
# ============================================================================

@doc "

Compute log-likelihood for single Weighted distribution with joint observations.

Handles joint observations as NamedTuple: `(value = x, weight = w)`.

See also: [`logpdf`](@ref)
"
function loglikelihood(d::Weighted, obs::NamedTuple{(:value, :weight)})
    return logpdf(d, obs)
end

@doc "

Compute log-likelihood for single Weighted distribution with vectorized joint
observations.

Handles joint observations as NamedTuple: `(values = [...], weights = [...])`.
This is useful when a single weighted distribution is used with multiple
observations.

See also: [`logpdf`](@ref)
"
function loglikelihood(d::Weighted, obs::NamedTuple{(:values, :weights)})
    # For single distribution with multiple observations, sum logpdf results
    return sum(logpdf(d, (value = v, weight = w))
    for (v, w) in zip(obs.values, obs.weights))
end

@doc "

Compute log-likelihood for Product{<:ValueSupport, <:Weighted} with joint
observations.

Handles joint observations as NamedTuple: `(values = [...], weights = [...])`.

See also: [`logpdf`](@ref)
"
function loglikelihood(
        d::Product{<:ValueSupport, <:Weighted, <:AbstractVector{<:Weighted}},
        obs::NamedTuple{(:values, :weights)})
    return logpdf(d, obs)
end

# ============================================================================
# Other Distribution Interface Methods (delegate to underlying distribution)
# ============================================================================

@doc "

Compute the cumulative distribution function (delegates to underlying
distribution).

See also: [`logcdf`](@ref)
"
function cdf(d::Weighted, x::Real)
    return cdf(get_dist(d), x)
end

@doc "

Compute the log cumulative distribution function (delegates to underlying
distribution).

See also: [`cdf`](@ref)
"
function logcdf(d::Weighted, x::Real)
    return logcdf(get_dist(d), x)
end

@doc "

Compute the complementary cumulative distribution function (delegates to
underlying distribution).

See also: [`cdf`](@ref)
"
function ccdf(d::Weighted, x::Real)
    return ccdf(get_dist(d), x)
end

@doc "

Compute the log complementary cumulative distribution function (delegates to
underlying distribution).

See also: [`logcdf`](@ref)
"
function logccdf(d::Weighted, x::Real)
    return logccdf(get_dist(d), x)
end

@doc "

Compute the quantile function (delegates to underlying distribution).

See also: [`cdf`](@ref)
"
function quantile(d::Weighted, p::Real)
    return quantile(get_dist(d), p)
end

@doc "

Generate a random sample (delegates to underlying distribution).

See also: [`quantile`](@ref)
"
Base.rand(rng::AbstractRNG, d::Weighted) = rand(rng, get_dist(d))

@doc "

Create a sampler for efficient sampling (delegates to underlying distribution).

The weight only scales the log-density, so it plays no part in sampling and is
dropped here. Rewrapping would also be invalid: `sampler` may return a sampler
object (e.g. `GammaMTSampler`) rather than a `UnivariateDistribution`.

See also: [`rand`](@ref)
"
sampler(d::Weighted) = sampler(get_dist(d))

# ============================================================================
# Helper Functions for Observation and Weight Processing
# ============================================================================

@doc "

Combine constructor weight with observation weight using dispatch-based rules.

Weight combination rules:
- `missing, missing → missing` (both missing means no weight)
- `w1, missing → w1` (use constructor weight)
- `missing, w2 → w2` (use observation weight)
- `w1, w2 → w1 * w2` (multiply weights)

# Vector Extensions

For Product distributions, additional methods handle vectorised weight
combinations:
- `Vector, Vector → combine_weights.(vector1, vector2)` (element-wise
  combination)
- `Vector, missing → Vector` (keep constructor weights)
- `Vector, scalar → [combine_weights(w, scalar) for w in Vector]` (broadcast
  scalar)

"
function combine_weights(::Missing, ::Missing)
    return missing
end

function combine_weights(w1, ::Missing)
    return w1
end

function combine_weights(::Missing, w2)
    return w2
end

function combine_weights(w1, w2)
    # Handle zero weights to avoid NaN from 0 * Inf. Seed the zero from the
    # promoted product type so all branches agree (and ForwardDiff Duals
    # are not stripped on the zero-weight branch).
    T = promote_type(typeof(w1), typeof(w2))
    (w1 == 0 || w2 == 0) && return zero(T)
    return w1 * w2
end

# Vector extensions for Product distributions
function combine_weights(
        constructor_weights::AbstractVector, obs_weights::AbstractVector)
    return combine_weights.(constructor_weights, obs_weights)
end

function combine_weights(constructor_weights::AbstractVector, ::Missing)
    return constructor_weights
end

function combine_weights(constructor_weights::AbstractVector, obs_weight::Real)
    return [combine_weights(w, obs_weight) for w in constructor_weights]
end
