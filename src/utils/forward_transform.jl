# ============================================================================
# Forward-transform leaves: a generic transform with thin / cumulative
# ============================================================================
#
# A forward transform carries an OP that `convolve_distributions(stack, series)`
# applies to the branch's output count series, materialising only when a stack is
# convolved through a timeseries.
#
# `cumulative` and the generic `transform(d, f)` are DETERMINISTIC functions of
# the series and stay TRANSPARENT to `logpdf` (an individual delay use ignores
# the op): the forward dual of `Weighted` (which touches `logpdf` only).
#
# `thin` is the exception. `thin(d, p)` is the friendly constructor over the
# probabilistic one-of `resolve(:event => (d, p), :none => (NoEvent(), 1 - p))`:
# each event is reported with probability `p`, else nothing happens. So thinning
# is NOT logpdf-transparent — it enters the per-record likelihood (the honest
# generative model): `logpdf(thin(d, p), x) == log(p) + logpdf(d, x)` (the
# defective density of an event observed at `x`), and `rand` returns a time with
# probability `p`, else `missing`. Under `convolve_distributions(stack, series)`
# this STILL reduces to the same `p`-scaling of the branch's count series (the
# `(1 - p)` no-event mass leaves the observed stream), so the aggregate-count
# convolution is unchanged: the convolution-marginal equivalence. `thin` carries
# the `ThinOp(p)` forward op so the convolve layer scales the series by `p`
# through the same path a `Resolve` branch probability uses.
#
# One node type, `Transformed`, carries an OP. `thin` and `cumulative` are
# specialised constructors over it with TYPED ops (so the factor stays
# introspectable and validated); the generic `transform(d, f)` accepts any
# callable `series -> series` as an escape hatch.
#
# The boundary (per the in/out test for generic node ops): a forward transform
# is a function of the series. A scale that is itself a sampled latent process
# stays user/model-side.

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

A delay carrying a forward-transform op, applied by
[`convolve_distributions`](@ref) to the branch's output count series, and
transparent to introspection (`free_leaf` peels to the inner delay). Construct
with the generic [`transform`](@ref) or the specialised [`thin`](@ref) /
[`cumulative`](@ref).

The generic [`transform`](@ref) and [`cumulative`](@ref) ops are transparent to
`logpdf` (an individual delay use ignores the op). [`thin`](@ref) is the
exception: it carries the probabilistic one-of `resolve` + `NoEvent` semantics
into `logpdf` / `rand` (each event reported with probability `p`), so it is NOT
logpdf-transparent.

# See also
- [`transform`](@ref), [`thin`](@ref), [`cumulative`](@ref): constructors
"
struct Transformed{D <: UnivariateDistribution, Op} <:
       UnivariateDistribution{ValueSupport}
    "The inner delay distribution."
    dist::D
    "The forward op applied to the convolved series."
    op::Op
end

@doc "

Apply a forward-transform op to a delay's convolved count series.

`transform(d, op)` carries `op` (a [`thin`](@ref)/[`cumulative`](@ref) op or any
callable `series -> series`) that [`convolve_distributions`](@ref) applies to the
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

`thin(d, p)` is the friendly constructor over the probabilistic one-of
`resolve(:event => (d, p), :none => (NoEvent(), 1 - p))` (see [`resolve`](@ref) /
[`NoEvent`](@ref)). It is a first-class composable op, not a convolve-only forward
scaler:

- Under `logpdf` / `rand` it is the honest generative model. An event is reported
  with probability `p` (its time then drawn from `d`), else `missing`:
  ``\text{logpdf} = \log p + \log f_d(x)`` (the defective density of an event
  observed at `x`, integrating to `p`), and `rand` returns a time with
  probability `p`, else `missing`. So thinning ENTERS the per-record likelihood;
  it is NOT logpdf-transparent.
- Under [`convolve_distributions`](@ref)`(stack, series)` it STILL scales the
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
- [`resolve`](@ref), [`NoEvent`](@ref): the one-of `thin` builds under the hood
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
[`convolve_distributions`](@ref) applies to the branch's count series, giving
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
# `thin(d, p)` is NOT logpdf-transparent (unlike `cumulative` / the generic
# `transform`): it carries the `resolve(:event => (d, p), :none => (NoEvent(),
# 1 - p))` semantics into `logpdf` / `rand`. Dispatch is on the `ThinOp` op so
# only `thin` overrides the transparent block above; `cumulative` and the generic
# transform keep delegating to the inner delay.
#
# An event is reported with probability `p`, else nothing happens (`missing`), so
# the scalar density is DEFECTIVE (it integrates to `p`, not 1): the joint density
# of "an event occurred AND was at `x`" is `p · f_d(x)`. This is exactly the
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

# `quantile` / `mean` / `var` are the CONDITIONAL-on-report time moments (the
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
