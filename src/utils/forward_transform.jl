# ============================================================================
# Forward-transform leaves: thin / cumulative
# ============================================================================
#
# Deterministic operations applied on the FORWARD path only: the count series
# that `convolve_distributions(stack, series)` produces. They are the forward
# dual of `Weighted` (which touches `logpdf` only): a forward transform is
# TRANSPARENT to `logpdf` and to introspection, and materialises only when a
# stack is convolved through a timeseries.
#
# Each is a leaf wrapper, so it nests in `compose`/`Sequential`/`Parallel`/
# `Select`/`Competing` like any `UnivariateDistribution`. `free_leaf` peels the
# wrapper to its inner delay (so `params_table`/`update`/the tree scorer see the
# inner delay), `rewrap_leaf` rebuilds it, and `_forward_apply` is the op the
# convolve layer applies to a branch's series.
#
# The boundary (per the in/out test for generic node ops): a forward transform
# is a DETERMINISTIC function of the series (a fixed factor, a cumulative sum).
# A scale that is itself a sampled latent process stays user/model-side.

# --- Scaled: multiply a branch's forward count series by a fixed factor ------

@doc "

A delay carrying a fixed forward count factor, applied by
[`convolve_distributions`](@ref) to the branch's output series. Transparent to
`logpdf` (an individual delay use ignores the factor) and to introspection
(`free_leaf` peels to the inner delay). Construct with [`thin`](@ref) (a
probability in ``[0, 1]``).

# See also
- [`thin`](@ref): the friendly constructor
- [`Cumulative`](@ref): the cumulative-sum forward transform
"
struct Scaled{D <: UnivariateDistribution, T <: Real} <:
       UnivariateDistribution{ValueSupport}
    "The inner delay distribution."
    dist::D
    "The forward count factor applied to the convolved series."
    factor::T

    function Scaled(dist::D, factor::T) where {
            D <: UnivariateDistribution, T <: Real}
        factor >= zero(factor) ||
            throw(ArgumentError("scale factor must be non-negative"))
        new{D, T}(dist, factor)
    end
end

@doc "

Thin a delay's forward count by a probability `p`.

`thin(d, p)` carries a forward factor `p ∈ [0, 1]` that
[`convolve_distributions`](@ref) multiplies into the branch's count series (e.g.
ascertainment of cases, the infection fatality ratio for deaths). It is
transparent to `logpdf`, so an individual delay use scores the inner delay
unchanged; the factor materialises only under convolution.

`thin(d, nothing)` returns `d` unchanged, so an optional factor can be threaded
through.

# Arguments
- `d`: the inner delay distribution.
- `p`: the thinning probability in ``[0, 1]``.

# Examples
```@example
using CensoredDistributions, Distributions

d = thin(LogNormal(1.5, 0.5), 0.3)
logpdf(d, 2.0) == logpdf(LogNormal(1.5, 0.5), 2.0)
```

# See also
- [`cumulative`](@ref): cumulative-sum a branch's series
- [`convolve_distributions`](@ref): applies the forward factor
"
function thin(dist::UnivariateDistribution, p::Real)
    (zero(p) <= p <= one(p)) ||
        throw(ArgumentError("thin probability must be in [0, 1]"))
    return Scaled(dist, p)
end

thin(dist::UnivariateDistribution, ::Nothing) = dist

# --- Cumulative: cumulative-sum a branch's forward count series --------------

@doc "

A delay whose forward count series is accumulated by
[`convolve_distributions`](@ref) (a running `cumsum`, e.g. cumulative incidence
or cumulative deaths). Transparent to `logpdf` and to introspection. Construct
with [`cumulative`](@ref).

# See also
- [`cumulative`](@ref): the friendly constructor
- [`Scaled`](@ref): the fixed-factor forward transform
"
struct Cumulative{D <: UnivariateDistribution} <:
       UnivariateDistribution{ValueSupport}
    "The inner delay distribution."
    dist::D
end

@doc "

Accumulate a delay's forward count series.

`cumulative(d)` carries a forward operation that [`convolve_distributions`](@ref)
applies as a running sum of the branch's count series, giving cumulative counts
(cumulative incidence, cumulative deaths). It is transparent to `logpdf`, so an
individual delay use scores the inner delay unchanged.

# Arguments
- `d`: the inner delay distribution.

# Examples
```@example
using CensoredDistributions, Distributions

d = cumulative(Gamma(2.0, 1.0))
logpdf(d, 2.0) == logpdf(Gamma(2.0, 1.0), 2.0)
```

# See also
- [`thin`](@ref): a fixed forward factor
- [`convolve_distributions`](@ref): applies the cumulative sum
"
cumulative(dist::UnivariateDistribution) = Cumulative(dist)

# --- transparency: delegate every distribution method to the inner delay -----

const _ForwardTransform = Union{Scaled, Cumulative}

# The inner delay a forward transform wraps.
get_dist(d::_ForwardTransform) = d.dist

for f in (:minimum, :maximum, :insupport, :mean, :var, :std, :mode, :median,
    :skewness, :kurtosis)
    @eval Distributions.$f(d::_ForwardTransform) = Distributions.$f(d.dist)
end

for f in (:pdf, :logpdf, :cdf, :logcdf, :ccdf, :logccdf, :quantile)
    @eval Distributions.$f(d::_ForwardTransform, x::Real) = Distributions.$f(d.dist, x)
end

Distributions.insupport(d::_ForwardTransform, x::Real) = insupport(d.dist, x)

Base.rand(rng::AbstractRNG, d::_ForwardTransform) = rand(rng, d.dist)

Distributions.params(d::Scaled) = (params(d.dist)..., d.factor)
Distributions.params(d::Cumulative) = params(d.dist)

# --- introspection transparency (free_leaf / rewrap_leaf) --------------------

# Peel to the inner free delay so introspection and the tree scorer see the
# delay, not the forward op. `rewrap_leaf` rebuilds the same forward op around a
# new inner delay.
free_leaf(d::Scaled) = free_leaf(d.dist)
rewrap_leaf(d::Scaled, inner) = Scaled(rewrap_leaf(d.dist, inner), d.factor)

free_leaf(d::Cumulative) = free_leaf(d.dist)
rewrap_leaf(d::Cumulative, inner) = Cumulative(rewrap_leaf(d.dist, inner))

# --- the forward op the convolve layer applies -------------------------------

# Peel forward-transform wrappers to the underlying (possibly censored) delay,
# returning `(delay, ops)` where `ops` is the ordered tuple of forward
# transforms to apply to the branch's convolved series. A non-transform leaf has
# no ops.
_peel_forward(d::UnivariateDistribution) = (d, ())
function _peel_forward(d::Scaled)
    inner, ops = _peel_forward(d.dist)
    return inner, (ops..., d)
end
function _peel_forward(d::Cumulative)
    inner, ops = _peel_forward(d.dist)
    return inner, (ops..., d)
end

# Apply one forward op to a series.
_forward_apply(d::Scaled, series) = d.factor .* series
_forward_apply(d::Cumulative, series) = cumsum(series)

# Apply a tuple of forward ops to a series, in order.
function _apply_forward_ops(series, ops)
    foldl((s, op) -> _forward_apply(op, s), ops; init = series)
end
