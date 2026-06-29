# ============================================================================
# Forward-transform leaves: a generic transform with thin / cumulative
# ============================================================================
#
# A forward transform carries an op that `convolved(stack, series)` applies to
# the branch's output count series. `cumulative` and the generic `transform`
# are transparent to `logpdf`; `thin(d, p)` is not, carrying the probabilistic
# one-of `resolve(:event => (d, p), :none => (NoEvent(), 1 - p))` semantics into
# `logpdf` / `rand` (each event reported with probability `p`).

# --- the ops --------------------------------------------------------------

# Multiply the series by a fixed factor (thinning / rescaling of expected
# counts). `cumulative` accumulates the series. Typed so the factor is readable.
struct ThinOp{T <: Real}
    factor::T
end
struct CumulativeOp end

# Apply one op to a series. A bare callable is its own op (the escape hatch).
_apply_op(op::ThinOp, series) = op.factor .* series
_apply_op(::CumulativeOp, series) = cumsum(series)
_apply_op(op, series) = op(series)

# The op's display parameters (only a numeric factor surfaces; others are pure).
_op_params(op::ThinOp) = (op.factor,)
_op_params(op) = ()

# --- the generic node -----------------------------------------------------

@doc "

A delay `dist` carrying a forward-transform `op`, applied by
[`convolved`](@ref) to the branch's output count series, and
transparent to introspection (`free_leaf` peels to the inner delay). Construct
with the generic [`transform`](@ref) or the specialised [`thin`](@ref) /
[`cumulative`](@ref).

The generic [`transform`](@ref) and [`cumulative`](@ref) ops are transparent to
`logpdf` (an individual delay use ignores the op). [`thin`](@ref) is the
exception: it carries the probabilistic one-of `resolve` + `NoEvent` semantics
into `logpdf` / `rand` (each event reported with probability `p`), so it is not
logpdf-transparent.

# See also
- [`transform`](@ref), [`thin`](@ref), [`cumulative`](@ref): constructors
"
struct Transformed{D <: UnivariateDistribution, Op} <:
       AbstractModifiedDistribution{Univariate, ValueSupport}
    "The inner delay distribution."
    dist::D
    "The forward op applied to the convolved series."
    op::Op
end

@doc "

Apply a forward-transform op to a delay's convolved count series.

`transform(d, op)` carries `op` (a [`thin`](@ref)/[`cumulative`](@ref) op or any
callable `series -> series`) that [`convolved`](@ref) applies to the
branch's output series. Transparent to `logpdf`. Prefer [`thin`](@ref) /
[`cumulative`](@ref) for the common cases; use `transform` for an arbitrary
deterministic series map.

# Arguments
- `d`: the inner delay distribution.
- `op`: a forward op or a callable `series -> series`.

# Examples
```@example
using CensoredDistributions, Distributions

d = transform(Gamma(2.0, 1.0), s -> 0.5 .* s)
logpdf(d, 2.0) == logpdf(Gamma(2.0, 1.0), 2.0)
```

# See also
- [`thin`](@ref), [`cumulative`](@ref): specialised forward transforms
"
transform(d::UnivariateDistribution, op) = Transformed(d, op)

@doc raw"

Thin a delay by a reporting probability `p`: each event is reported with
probability `p`, else nothing happens.

`thin(d, p)` produces a `Transformed(d, ThinOp(p))` that carries the same
semantics as the probabilistic one-of
`resolve(:event => (d, p), :none => (NoEvent(), 1 - p))` (see [`resolve`](@ref) /
[`NoEvent`](@ref)); it does not construct a `resolve` node. It is a first-class
composable op, not a convolve-only forward scaler:

- Under `logpdf` / `rand` it is the honest generative model. An event is reported
  with probability `p` (its time then drawn from `d`), else `missing`:
  ``\text{logpdf} = \log p + \log f_d(x)`` (the defective density of an event
  observed at `x`, integrating to `p`), and `rand` returns a time with
  probability `p`, else `missing`. So thinning enters the per-record likelihood;
  it is not logpdf-transparent.
- Under [`convolved`](@ref)`(stack, series)` it still scales the
  branch's expected-count series by `p` (e.g. ascertainment of cases, the
  infection fatality ratio for deaths): the `(1 - p)` no-event mass leaves the
  observed stream. So the aggregate-count convolution is unchanged from the
  forward-scaler behaviour — the convolution-marginal equivalence.

`thin(d, nothing)` returns `d` unchanged.

# Arguments
- `d`: the inner delay distribution.
- `p`: the reporting probability in ``[0, 1]``.

# Examples
```@example
using CensoredDistributions, Distributions

d = thin(LogNormal(1.5, 0.5), 0.3)
# Reporting enters the per-record likelihood (not logpdf-transparent).
logpdf(d, 2.0) == log(0.3) + logpdf(LogNormal(1.5, 0.5), 2.0)
```

# See also
- [`resolve`](@ref), [`NoEvent`](@ref): the one-of whose semantics `thin` carries
- [`transform`](@ref): the generic forward transform
- [`cumulative`](@ref): cumulative-sum a branch's series
"
function thin(dist::UnivariateDistribution, p::Real)
    (zero(p) <= p <= one(p)) ||
        throw(ArgumentError("thin probability must be in [0, 1]"))
    return Transformed(dist, ThinOp(p))
end

thin(dist::UnivariateDistribution, ::Nothing) = dist

@doc "

Accumulate a delay's forward count series.

`cumulative(d)` is [`transform`](@ref) with a running-sum op that
[`convolved`](@ref) applies to the branch's count series, giving
cumulative counts (cumulative incidence, cumulative deaths). Transparent to
`logpdf`.

# Arguments
- `d`: the inner delay distribution.

# Examples
```@example
using CensoredDistributions, Distributions

d = cumulative(Gamma(2.0, 1.0))
logpdf(d, 2.0) == logpdf(Gamma(2.0, 1.0), 2.0)
```

# See also
- [`transform`](@ref): the generic forward transform
- [`thin`](@ref): a fixed forward factor
"
cumulative(dist::UnivariateDistribution) = Transformed(dist, CumulativeOp())

# --- transparency: delegate every distribution method to the inner delay -----

# The inner delay a forward transform wraps.
get_dist(d::Transformed) = d.dist

for f in (:minimum, :maximum, :mean, :var, :std, :mode, :median,
    :skewness, :kurtosis)
    @eval Distributions.$f(d::Transformed) = Distributions.$f(d.dist)
end

for f in (:pdf, :logpdf, :cdf, :logcdf, :ccdf, :logccdf, :quantile)
    @eval Distributions.$f(d::Transformed, x::Real) = Distributions.$f(d.dist, x)
end

Distributions.insupport(d::Transformed, x::Real) = insupport(d.dist, x)

Base.rand(rng::AbstractRNG, d::Transformed) = rand(rng, d.dist)

Distributions.params(d::Transformed) = (params(d.dist)..., _op_params(d.op)...)

# --- thin: the honest one-of (resolve + NoEvent) scalar semantics ------------
#
# `thin(d, p)` is not logpdf-transparent (unlike `cumulative` / the generic
# `transform`): it carries the `resolve(:event => (d, p), :none => (NoEvent(),
# 1 - p))` semantics into `logpdf` / `rand`. Dispatch is on the `ThinOp` op so
# only `thin` overrides the transparent block above; `cumulative` and the generic
# transform keep delegating to the inner delay.
#
# An event is reported with probability `p`, else nothing happens (`missing`), so
# the scalar density is defective (it integrates to `p`, not 1): the joint density
# of "an event occurred and was at `x`" is `p · f_d(x)`. This is exactly the
# `:event` branch of the resolve + NoEvent node (`_one_of_condition_logpdf`'s
# `log p + logpdf(d, x)`); the `none` branch carries the residual `1 - p` survival
# mass. The convolve layer's `ThinOp(p)` series-scaling is the aggregate-count
# image of the same model (the convolution-marginal equivalence).
const _Thinned = Transformed{D, <:ThinOp} where {D <: UnivariateDistribution}

# The reporting probability `p` the `:event` branch carries.
_thin_prob(d::_Thinned) = d.op.factor

# Defective density: `log p + logpdf(d, x)` (mass `p`), the `:event`-branch
# conditioned log-density of the resolve + NoEvent one-of.
function Distributions.logpdf(d::_Thinned, x::Real)
    return log(_thin_prob(d)) + logpdf(d.dist, x)
end
Distributions.pdf(d::_Thinned, x::Real) = _thin_prob(d) * pdf(d.dist, x)

# Defective cdf: `p · F_d(x)` (it tends to `p`, not 1). `ccdf` carries the
# `1 - p` no-event survival mass on top of the unreported-event tail.
Distributions.cdf(d::_Thinned, x::Real) = _thin_prob(d) * cdf(d.dist, x)
Distributions.logcdf(d::_Thinned, x::Real) = log(_thin_prob(d)) + logcdf(d.dist, x)
Distributions.ccdf(d::_Thinned, x::Real) = one(_thin_prob(d)) - cdf(d, x)
Distributions.logccdf(d::_Thinned, x::Real) = log(ccdf(d, x))

# `quantile` / `mean` / `var` are the conditional-on-report time moments (the
# `:event` branch's own delay `d`): the defective node has no proper marginal,
# so these describe the reported-event time, matching `rand`'s non-`missing` draw.
Distributions.quantile(d::_Thinned, q::Real) = quantile(d.dist, q)
for f in (:mean, :var, :std, :mode, :median, :skewness, :kurtosis)
    @eval Distributions.$f(d::_Thinned) = Distributions.$f(d.dist)
end

# Sample the one-of: report with probability `p` (time drawn from `d`), else
# `missing` (no event), mirroring `rand_outcome` on the resolve + NoEvent node.
function Base.rand(rng::AbstractRNG, d::_Thinned)
    return rand(rng) < _thin_prob(d) ? rand(rng, d.dist) : missing
end

# --- introspection transparency (free_leaf / rewrap_leaf) --------------------

# Peel to the inner free delay so introspection and the tree scorer see the
# delay, not the forward op. `rewrap_leaf` rebuilds the same op around a new
# inner delay.
free_leaf(d::Transformed) = free_leaf(d.dist)
rewrap_leaf(d::Transformed, inner) = Transformed(rewrap_leaf(d.dist, inner), d.op)

# --- thin weight as a surfaced free parameter --------------------------------
#
# Unlike the fixed censoring bounds, a `thin(d, p)` reporting probability `p` is
# a free parameter (it enters the per-record likelihood as the defective density
# `log p + logpdf(d, x)`), so it must be inventoried by `params_table`,
# defaulted by `build_priors`, and round-tripped by `update` /
# `composed_parameters_model` like any leaf parameter. These helpers expose the
# outermost `ThinOp` factor (`_thin_factor`) and rebuild a leaf with a new factor
# (`_set_thin_factor`); a leaf with no thin op reports `nothing` and is left
# unchanged. The introspection layer appends one `:thin` row per thinned leaf
# (support `[0, 1]`), and reconstruction splits the trailing thin value off the
# value tuple and routes it back into the op.

# The outermost `ThinOp` factor of a (possibly censored) leaf, or `nothing` when
# the leaf carries no thin op. Peels the same wrappers as `free_leaf` so a
# thinned censored delay still reports its factor, but stops at the first
# `ThinOp` (a single surfaced weight per leaf).
_thin_factor(leaf) = nothing
function _thin_factor(d::Transformed)
    return d.op isa ThinOp ? d.op.factor : _thin_factor(d.dist)
end
_thin_factor(d::PrimaryCensored) = _thin_factor(d.dist)
_thin_factor(d::IntervalCensored) = _thin_factor(d.dist)
_thin_factor(d::Truncated) = _thin_factor(d.untruncated)

# Rebuild a leaf with its outermost `ThinOp` factor replaced by `p`, keeping the
# inner delay and any censoring untouched. Mirrors `_thin_factor`'s peel; only
# the `ThinOp` node is swapped (the censoring wrappers are returned unchanged, so
# nothing re-validates the inner delay on the AD reconstruction path). A
# non-thinned leaf is returned unchanged (it has no factor to set).
_set_thin_factor(leaf, p) = leaf
function _set_thin_factor(d::Transformed, p)
    return d.op isa ThinOp ? Transformed(d.dist, ThinOp(p)) :
           Transformed(_set_thin_factor(d.dist, p), d.op)
end
function _set_thin_factor(d::PrimaryCensored, p)
    return PrimaryCensored(_set_thin_factor(d.dist, p), d.primary_event, d.method)
end
function _set_thin_factor(d::IntervalCensored, p)
    return IntervalCensored(_set_thin_factor(d.dist, p), d.boundaries)
end
function _set_thin_factor(d::Truncated, p)
    return truncated(_set_thin_factor(d.untruncated, p); lower = d.lower,
        upper = d.upper)
end

# --- the forward op the convolve layer applies -------------------------------

# Peel forward-transform wrappers to the underlying (possibly censored) delay,
# returning `(delay, ops)` where `ops` is the ordered tuple of ops to apply to
# the branch's convolved series. A non-transform leaf has no ops.
_peel_forward(d) = (d, ())
function _peel_forward(d::Transformed)
    inner, ops = _peel_forward(d.dist)
    return inner, (ops..., d.op)
end

# Apply a tuple of forward ops to a series, in order.
function _apply_forward_ops(series, ops)
    foldl((s, op) -> _apply_op(op, s), ops; init = series)
end
