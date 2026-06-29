@doc raw"""

A link for the hazard modification carried by a [`Modified`](@ref) distribution.

The modification acts on the hazard through the link `g`,

```math
h^{*}(t) = g^{-1}\!\big(g(h(t)) + \text{effect}\big),
```

so `log` gives proportional hazards, `identity` gives additive hazards and
`logit` gives a discrete-time reporting hazard. A `HazardLink` pairs the link
`g` with its inverse `g⁻¹`; the three named links ([`LogLink`](@ref),
[`IdentityLink`](@ref), [`LogitLink`](@ref)) are built-in, and any invertible
callable can be wrapped with [`hazard_link`](@ref).

# Fields
- `g`: the link `g` mapping a hazard onto the modification scale.
- `invlink`: the inverse link `g⁻¹` mapping back to a hazard.

# See also
- [`modify`](@ref): the verb that builds a [`Modified`](@ref) distribution.
"""
struct HazardLink{G, GI}
    "The link `g` mapping a hazard onto the modification scale."
    g::G
    "The inverse link `g⁻¹` mapping back to a hazard."
    invlink::GI
end

@doc "The log link (proportional hazards): `g = log`, `g⁻¹ = exp`."
const LogLink = HazardLink(log, exp)

@doc "The identity link (additive hazards): `g = g⁻¹ = identity`."
const IdentityLink = HazardLink(identity, identity)

@doc "The logit link (discrete-time reporting hazard): `g = logit`."
const LogitLink = HazardLink(_logit, _logistic)

# Show the named links compactly rather than dumping the wrapped closures.
function Base.show(io::IO, l::HazardLink)
    name = l === LogLink ? "LogLink" :
           l === IdentityLink ? "IdentityLink" :
           l === LogitLink ? "LogitLink" : "HazardLink($(l.g), $(l.invlink))"
    print(io, name)
    return nothing
end

@doc raw"""

Wrap a link and its inverse as a [`HazardLink`](@ref).

The link `g` maps a hazard onto the scale the additive `effect` acts on, and
`invlink` maps back. Use the built-in [`LogLink`](@ref), [`IdentityLink`](@ref)
or [`LogitLink`](@ref) for the standard choices; this constructor is for a
user-supplied invertible callable.

# Arguments
- `g`: the link function `g`.
- `invlink`: the inverse link `g⁻¹`.

# Examples
```@example
using CensoredDistributions

# A complementary-log-log link.
cloglog = CensoredDistributions.hazard_link(
    h -> log(-log1p(-h)), η -> -expm1(-exp(η)))
```

# See also
- [`modify`](@ref): the verb that consumes a link.
"""
hazard_link(g, invlink) = HazardLink(g, invlink)

# Normalise a user-facing `link` argument to a `HazardLink`. The bare functions
# `log`/`identity` and the symbols `:log`/`:identity`/`:logit` map onto the
# named links, so the verb accepts either idiom; a `HazardLink` flows through.
_as_hazard_link(l::HazardLink) = l
_as_hazard_link(::typeof(log)) = LogLink
_as_hazard_link(::typeof(identity)) = IdentityLink
function _as_hazard_link(s::Symbol)
    s === :log && return LogLink
    s === :identity && return IdentityLink
    s === :logit && return LogitLink
    throw(ArgumentError("unknown link $(s); use :log, :identity or :logit"))
end

@doc raw"""

A delay distribution whose hazard is modified through a link.

`Modified` carries a base delay `dist`, a hazard `effect` and a
[`HazardLink`](@ref) `link`, and lazily instantiates the modified hazard

```math
h^{*}(t) = g^{-1}\!\big(g(h(t)) + \text{effect}\big)
```

in `logpdf`/`cdf`/`ccdf`/`rand`, where `g` is the link. The modification is
never materialised eagerly, so a `Modified` nests as a leaf in [`compose`](@ref),
the censoring wrappers and the truncation helpers, and composes with everything
that consumes a `UnivariateDistribution`.

The instantiation path is chosen by dispatch on the base `dist` and the link,
mirroring the [`primary_censored`](@ref) solver architecture:

- continuous base, [`LogLink`](@ref): analytic proportional hazards,
  ``S^{*} = S^{e^{\beta}}``;
- continuous base, [`IdentityLink`](@ref): analytic additive hazards,
  ``S^{*}(t) = S(t)\,e^{-\beta t}``;
- continuous base, any other link: numeric, integrating the modified
  cumulative hazard through the same Gauss-Legendre solver
  [`primary_censored`](@ref) uses;
- discrete base (an [`interval_censored`](@ref) PMF): exact per-bin
  reconstruction through [`apply_hazard_effects`](@ref).

# Fields
- `dist`: the base delay distribution whose hazard is modified.
- `effect`: the hazard modification (a scalar, a callable `effect(t)`, or a
  per-bin vector for a discrete base).
- `link`: the hazard link `g` and its inverse.
- `method`: the quadrature solver used on the continuous numeric path.

# See also
- [`modify`](@ref): the constructor verb.
"""
struct Modified{D <: UnivariateDistribution, E, L <: HazardLink, M} <:
       AbstractModifiedDistribution{Univariate, Continuous}
    "The base delay distribution whose hazard is modified."
    dist::D
    "The hazard modification effect (a scalar, a callable `effect(t)`, or a
    per-bin vector for a discrete base)."
    effect::E
    "The hazard link `g` and its inverse."
    link::L
    "The quadrature solver used on the continuous numeric path."
    method::M

    function Modified(dist::D, effect::E, link::L,
            method::M) where {
            D <: UnivariateDistribution, E, L <: HazardLink, M}
        new{D, E, L, M}(dist, effect, link, method)
    end
end

@doc raw"""

Modify the hazard of a delay distribution through a link.

`modify(d, effect; link = log)` returns a [`Modified`](@ref) distribution whose
hazard is `h^{*}(t) = g^{-1}(g(h(t)) + effect)`, where `g` is the `link`
(`log` for proportional hazards, `identity` for additive hazards, `:logit` for a
discrete-time reporting hazard, or any invertible callable via
[`hazard_link`](@ref)). The modification is instantiated lazily, so the result
composes everywhere a `UnivariateDistribution` does.

For a continuous base the `log` and `identity` links use closed forms; any other
link integrates the modified cumulative hazard numerically through the `method`
solver. For a discrete (interval-censored) base, `effect` is a per-bin vector and
the modified PMF is reconstructed exactly through [`apply_hazard_effects`](@ref),
for any link including a user callable.

# Arguments
- `d`: the base delay distribution.
- `effect`: the hazard modification. A scalar or a callable `effect(t)` on a
  continuous base; a per-bin vector on a discrete base.

# Keyword Arguments
- `link`: the hazard link. The functions `log` (default) and `identity`, the
  symbols `:log`/`:identity`/`:logit`, or a [`HazardLink`](@ref) such as
  [`LogitLink`](@ref) or one from [`hazard_link`](@ref).
- `method`: the quadrature solver for the continuous numeric path (default
  `GaussLegendre(; n = 64)`).

# Examples
```@example
using CensoredDistributions, Distributions

# Proportional hazards: halve the hazard of a LogNormal delay.
d = modify(LogNormal(1.5, 0.5), -log(2.0); link = log)
ccdf(d, 2.0)

# Discrete reporting hazard on a daily interval-censored delay.
ic = interval_censored(LogNormal(1.5, 0.5), 1.0)
effects = fill(0.2, 11)
m = modify(ic, effects; link = :logit)
```

# See also
- [`Modified`](@ref): the wrapped type.
- [`apply_hazard_effects`](@ref): the discrete per-bin reconstruction.
"""
function modify(d::UnivariateDistribution, effect; link = log,
        method = GaussLegendre(; n = 64))
    return Modified(d, effect, _as_hazard_link(link), method)
end

# Parameter extraction: the base distribution's parameters followed by the
# effect. A callable effect carries no numeric parameters, so only the base
# params surface in that case.
function params(d::Modified)
    base = params(d.dist)
    return d.effect isa Function ? base : (base..., d.effect)
end

Base.eltype(::Type{<:Modified{D}}) where {D} = eltype(D)
minimum(d::Modified) = minimum(d.dist)
maximum(d::Modified) = maximum(d.dist)
insupport(d::Modified, x::Real) = insupport(d.dist, x)

# A Modified is transparent to its inner free delay for composer introspection,
# mirroring the censoring wrappers and Affine: the effect/link are fixed
# structure, the inner delay's parameters are the free ones.
free_leaf(d::Modified) = free_leaf(d.dist)
function rewrap_leaf(d::Modified, inner)
    rebuilt = rewrap_leaf(d.dist, inner)
    return Modified(rebuilt, d.effect, d.link, d.method)
end

get_dist(d::Modified) = d.dist

# Show a Modified compactly: its base and its link, never the solver arrays.
function Base.show(io::IO, d::Modified)
    print(io, "Modified(", d.dist, "; link=", d.link, ")")
    return nothing
end

# ============================================================================
# Continuous, log link: analytic proportional hazards
# ============================================================================
#
# With β the (scalar) effect and θ = exp(β), the modified survival is S^θ:
# logS* = θ logS, H* = θ H, h* = θ h, so
#   logpdf* = log h* + logS* = log θ + log h + θ logS
#           = β + (logpdf - logccdf) + θ logccdf
#           = β + logpdf + (θ - 1) logccdf.
# rand inverts S*(t) = U: S(t) = U^{1/θ}, t = quantile(d, 1 - U^{1/θ}).

@doc "

Compute the log survival function on the proportional-hazards path.

See also: [`ccdf`](@ref), [`logpdf`](@ref)
"
function logccdf(d::Modified{<:UnivariateDistribution{Continuous}, <:Real, typeof(LogLink)}, x::Real)
    θ = exp(d.effect)
    return θ * _logccdf_ad_safe(d.dist, x)
end

@doc "

Compute the log probability density on the proportional-hazards path.

See also: [`pdf`](@ref), [`ccdf`](@ref)
"
function logpdf(d::Modified{<:UnivariateDistribution{Continuous}, <:Real, typeof(LogLink)}, x::Real)
    insupport(d, x) || return oftype(float(x), -Inf)
    β = d.effect
    θ = exp(β)
    return β + logpdf(d.dist, x) + (θ - one(θ)) * _logccdf_ad_safe(d.dist, x)
end

function Base.rand(rng::AbstractRNG,
        d::Modified{<:UnivariateDistribution{Continuous}, <:Real, typeof(LogLink)})
    θ = exp(d.effect)
    u = rand(rng)
    # S*(t) = S(t)^θ = u  =>  S(t) = u^{1/θ}  =>  F(t) = 1 - u^{1/θ}.
    return quantile(d.dist, 1 - u^(1 / θ))
end

# ============================================================================
# Continuous, identity link: analytic additive hazards
# ============================================================================
#
# With β the (scalar) effect, the additive hazard is h*(t) = h(t) + β, so
#   H*(t) = H(t) + β t,  S*(t) = S(t) exp(-β t),  logS* = logS - β t,
#   logpdf* = log h*(t) + logS*(t) = log(h(t) + β) - H(t) - β t,
# where log h(t) = logpdf - logccdf, so h(t) = exp(logpdf - logccdf).
#
# The closed form is only the survival of a valid distribution while the
# additive hazard h(t) + β stays non-negative for all t in support. For β >= 0
# this holds (h >= 0), so the closed form is used directly. For β < 0 the
# hazard goes negative wherever h(t) < |β| — near the origin for any base with
# h(0) = 0 (LogNormal, Gamma/Weibull shape > 1, ...) — and the closed-form
# S* = S exp(-β t) would exceed 1, giving a negative, non-monotone cdf
# (issue #670). The model is then the clamped additive hazard
# h*(t) = max(h(t) + β, 0): we route logccdf/logpdf through the numeric
# clamped cumulative-hazard integration so survival, cdf and pdf stay mutually
# consistent and the cdf stays monotone in [0, 1]. (A large negative effect
# therefore truncates the early hazard rather than producing an invalid law.)

@doc "

Compute the log survival function on the additive-hazards path.

See also: [`ccdf`](@ref), [`logpdf`](@ref)
"
function logccdf(
        d::Modified{<:UnivariateDistribution{Continuous}, <:Real, typeof(IdentityLink)},
        x::Real)
    β = d.effect
    # Negative effect: the hazard can dip below zero, so integrate the clamped
    # hazard numerically (see the section comment above).
    if β < zero(β)
        x <= minimum(d.dist) && return zero(float(typeof(x)))
        return -_modified_cumhazard(d, x)
    end
    return _logccdf_ad_safe(d.dist, x) - β * x
end

@doc "

Compute the log probability density on the additive-hazards path.

See also: [`pdf`](@ref), [`ccdf`](@ref)
"
function logpdf(
        d::Modified{<:UnivariateDistribution{Continuous}, <:Real, typeof(IdentityLink)},
        x::Real)
    insupport(d, x) || return oftype(float(x), -Inf)
    β = d.effect
    # Negative effect: use the clamped hazard and its numeric cumulative hazard
    # so the density matches the (clamped) survival above.
    β < zero(β) && return _numeric_logpdf(d, x)
    logS = _logccdf_ad_safe(d.dist, x)
    # h(t) = exp(logpdf - logS); modified hazard h(t) + β stays positive here.
    h = exp(logpdf(d.dist, x) - logS)
    hstar = h + β
    hstar <= zero(hstar) && return oftype(float(x), -Inf)
    return log(hstar) + logS - β * x
end

function Base.rand(rng::AbstractRNG,
        d::Modified{<:UnivariateDistribution{Continuous}, <:Real, typeof(IdentityLink)})
    return quantile(d, rand(rng))
end

# ============================================================================
# Continuous, general link: numeric modified cumulative hazard
# ============================================================================
#
# H*(t) = ∫₀ᵗ g⁻¹(g(h(u)) + effect(u)) du with h(u) = f(u)/S(u) the base
# hazard, integrated through the same Gauss-Legendre solver primary-censoring
# uses. S*(t) = exp(-H*(t)). The `effect` may be a scalar or a callable.

# The base hazard h(u) = f(u) / S(u) = exp(logpdf - logccdf), AD-safe and
# clamped to a tiny positive floor so the link's `log`/`logit` stays finite at
# the support edge where the survival has numerically exhausted.
function _base_hazard(dist::UnivariateDistribution, u::Real)
    logS = _logccdf_ad_safe(dist, u)
    h = exp(logpdf(dist, u) - logS)
    return max(h, eps(float(typeof(h))))
end

# The effect evaluated at `u`: a callable is applied, a scalar is constant.
_effect_at(effect, u) = effect isa Function ? effect(u) : effect

# The pre-clamp modified rate g⁻¹(g(h(u)) + effect(u)): the modified hazard
# before the non-negativity clamp. Its zero-crossings are the knots where the
# clamp engages, so `max(rate, 0)` has a kink. Used both to clamp (below) and to
# locate the knots that split the cumulative-hazard quadrature.
function _premodified_rate(d::Modified, u::Real)
    h = _base_hazard(d.dist, u)
    return d.link.invlink(d.link.g(h) + _effect_at(d.effect, u))
end

# The modified instantaneous hazard h*(u) = max(g⁻¹(g(h(u)) + effect(u)), 0).
# The clamp at zero keeps the modified hazard a valid (non-negative) hazard for
# links whose inverse can return a negative rate. The additive (identity) link
# does this when the effect is negative and the base hazard is smaller than the
# effect magnitude (h(u) + β < 0); clamping there makes the cumulative-hazard
# integral H* = ∫ max(h + β, 0) consistent with the logpdf clamp below, so
# survival, cdf and pdf all use the same clamped hazard. For links with a
# non-negative inverse (log → exp, logit → logistic) the clamp is a no-op.
function _modified_hazard(d::Modified, u::Real)
    hstar = _premodified_rate(d, u)
    return max(hstar, zero(hstar))
end

# The pre-clamp modified rate for the knot scan ONLY. Same value as
# `_premodified_rate`, but computed on AD-stripped (primal) parameters so the
# rate — and the `logpdf`/`logccdf` it calls — never runs under an AD trace. The
# knots only locate where the additive-hazard clamp engages; they split the
# cumulative-hazard quadrature at a continuous kink and carry no gradient.
#
# Stripping is necessary because a traced rate evaluates `logpdf(base, lo)` at
# the support edge, which is `-Inf` for a base whose density vanishes there
# (Gamma shape > 1, LogNormal, ...). Reverse-mode AD then forms the discarded
# `0 * (-Inf)` adjoint as a `NaN` that poisons the base distribution's first
# parameter (#680). The element-wise `_primal` strips ForwardDiff `Dual`s and
# ReverseDiff `TrackedReal`s (each has a `_primal` method) to plain `Float64`
# before any density is evaluated. Mooncake leaves `_primal` as identity, so it
# additionally gets the explicit `@zero_derivative` rule in its extension, and
# ChainRules-based reverse modes get the `@non_differentiable` mark; both cut the
# trace at the scan while the quadrature integrand (which DOES carry the
# gradient) stays untouched.
function _premodified_rate_primal(d::Modified, u::Real)
    base = _primal_distribution(d.dist)
    effect = d.effect isa Function ? d.effect : _primal(d.effect)
    up = _primal(u)
    h = _base_hazard(base, up)
    return d.link.invlink(d.link.g(h) + _effect_at(effect, up))
end

# Locate the clamp knots in `(lo, t)`: the points where the pre-clamp modified
# rate crosses zero, so the clamped integrand `max(rate, 0)` has a kink there.
# A coarse scan brackets each sign change, then bisection refines it. The scan
# runs on AD-stripped primals (`_premodified_rate_primal`), so the knots are
# plain `Float64` even under AD; the quadrature integrand still carries the
# `Dual`s, which is what differentiates correctly. The knots are where
# `max(rate, 0)` is continuous, so they introduce no boundary terms under
# differentiation. Integrating each smooth panel between knots (rather than one
# fixed rule over a kinked integrand) keeps the cumulative hazard, and so the
# cdf, monotone to machine precision (#670).
const _MODIFIED_KNOT_SCAN = 64

function _modified_knots(d::Modified, lo::Real, t::Real)
    rate(u) = _primal(_premodified_rate_primal(d, u))
    knots = Float64[]
    a = _primal(float(lo))
    b = _primal(float(t))
    step = (b - a) / _MODIFIED_KNOT_SCAN
    step > zero(step) || return knots
    prev_u = a
    prev_r = rate(a)
    for i in 1:_MODIFIED_KNOT_SCAN
        cur_u = i == _MODIFIED_KNOT_SCAN ? b : a + i * step
        cur_r = rate(cur_u)
        if (prev_r < 0) != (cur_r < 0)
            lo2, hi2 = prev_u, cur_u
            for _ in 1:60
                mid = (lo2 + hi2) / 2
                (rate(lo2) < 0) == (rate(mid) < 0) ? (lo2 = mid) : (hi2 = mid)
            end
            push!(knots, (lo2 + hi2) / 2)
        end
        prev_u = cur_u
        prev_r = cur_r
    end
    return knots
end

# The modified cumulative hazard H*(t) = ∫₀ᵗ h*(u) du via the solver. The lower
# bound is the base support minimum (≥ 0 for a delay); a non-positive `t`
# carries no hazard. The clamped integrand `max(rate, 0)` is kinked wherever the
# clamp engages, so a single fixed-node rule over the whole span loses
# monotonicity in the tail (#670); split the integral at the clamp knots and
# integrate each smooth panel, summing the pieces.
function _modified_cumhazard(d::Modified, t::Real)
    lo = max(minimum(d.dist), zero(t))
    t <= lo && return zero(float(promote_type(typeof(t), eltype(d.dist))))
    integrand = u -> _modified_hazard(d, u)
    knots = _modified_knots(d, lo, t)
    isempty(knots) && return integrate(d.method, integrand, lo, t)
    acc = integrate(d.method, integrand, lo, oftype(t, knots[1]))
    for k in 2:length(knots)
        acc += integrate(d.method, integrand,
            oftype(t, knots[k - 1]), oftype(t, knots[k]))
    end
    acc += integrate(d.method, integrand, oftype(t, knots[end]), t)
    return acc
end

# `logpdf* = log h*(t) - H*(t)` from the clamped modified hazard and its
# numeric cumulative hazard. The density is zero where the hazard is clamped to
# zero (h* <= 0) and in the deep tail where the survival has numerically
# exhausted (H* or h* non-finite), keeping log h* - H* from evaluating to NaN
# (Inf - Inf) where S* ≈ 0.
function _numeric_logpdf(d::Modified, x::Real)
    H = _modified_cumhazard(d, x)
    hstar = _modified_hazard(d, x)
    (isfinite(H) && isfinite(hstar)) || return oftype(float(x), -Inf)
    hstar <= zero(hstar) && return oftype(float(x), -Inf)
    return log(hstar) - H
end

# A general-link Modified on a continuous base: the union of all links that are
# NOT the analytic log/identity specialisations. Dispatch picks this only when
# neither analytic method applies.
const _NumericModified = Modified{<:UnivariateDistribution, E, L} where {
    E, L <: HazardLink}

@doc "

Compute the log survival function by numeric cumulative-hazard integration.

See also: [`ccdf`](@ref), [`logpdf`](@ref)
"
function logccdf(d::_NumericModified, x::Real)
    x <= minimum(d.dist) && return zero(float(typeof(x)))
    return -_modified_cumhazard(d, x)
end

@doc "

Compute the log probability density on the numeric path.

`logpdf* = log h*(t) - H*(t)`, the modified hazard times the modified survival.

See also: [`pdf`](@ref), [`ccdf`](@ref)
"
function logpdf(d::_NumericModified, x::Real)
    insupport(d, x) || return oftype(float(x), -Inf)
    return _numeric_logpdf(d, x)
end

function Base.rand(rng::AbstractRNG, d::_NumericModified)
    return quantile(d, rand(rng))
end

# ============================================================================
# Shared continuous interface: derive the rest from logccdf / logpdf
# ============================================================================

const _ContinuousModified = Modified{<:UnivariateDistribution,
    <:Union{Real, Function}}

pdf(d::_ContinuousModified, x::Real) = exp(logpdf(d, x))

function ccdf(d::_ContinuousModified, x::Real)
    return exp(logccdf(d, x))
end

function cdf(d::_ContinuousModified, x::Real)
    return -expm1(logccdf(d, x))
end

function logcdf(d::_ContinuousModified, x::Real)
    return log1mexp(logccdf(d, x))
end

@doc "

Compute the quantile (inverse CDF) by numeric inversion of the modified CDF.

See also: [`cdf`](@ref)
"
function quantile(d::_ContinuousModified, p::Real)
    initial_guess_fn = (dd, pp) -> [quantile(dd.dist, pp)]
    return _quantile_optimization(d, p; initial_guess_fn = initial_guess_fn)
end

sampler(d::_ContinuousModified) = d

# ============================================================================
# Discrete base (interval-censored PMF): exact per-bin reconstruction
# ============================================================================
#
# The discrete path lifts `apply_hazard_effects` onto a distribution: the base
# PMF over the interval grid is reshaped per bin through the link, then the
# interval masses come straight from the reconstructed PMF. The `effect` is a
# per-bin vector. Works for ANY link, including a user callable, since each bin
# is the cheap scalar map g⁻¹(g(h_d) + effect_d).
#
# A per-bin vector effect (not a scalar or callable) is what selects this path,
# so the discrete and continuous (`_ContinuousModified`, a `Real`/`Function`
# effect) method sets are disjoint even though an `IntervalCensored` base is
# itself a `UnivariateDistribution`: no `pdf`/`cdf`/... ambiguity between them.

const _DiscreteModified = Modified{<:IntervalCensored, <:AbstractVector}

# The interval grid `[0, w), [w, 2w), ...` of a regular interval-censored base,
# one grid point per effect entry. The number of bins is the effect length.
function _discrete_grid(d::_DiscreteModified)
    ic = d.dist
    is_regular_intervals(ic) ||
        throw(ArgumentError(
            "discrete modify requires a regular interval-censored base"))
    w = interval_width(ic)
    n = length(d.effect)
    # Explicit comprehension, not `(0:(n-1)) .* w`: a float-stepped range
    # builds a `StepRangeLen`/`TwicePrecision` whose `floatrange`
    # bit-twiddling Enzyme reverse cannot differentiate (#728).
    grid = [i * w for i in 0:(n - 1)]
    return grid, w
end

# The baseline PMF over the grid, then the link-modified PMF: the base hazard is
# the per-bin discrete-time hazard of the baseline PMF, modified by the effect
# through the link, then reconstructed. Built once per evaluation.
function _modified_pmf(d::_DiscreteModified)
    grid, _ = _discrete_grid(d)
    ic = d.dist
    base = map(g -> pdf(ic, g), grid)
    return _apply_hazard_link(base, d.effect, d.link)
end

# Index of the grid bin containing `x` (1-based), or 0 if outside the grid.
function _discrete_bin(d::_DiscreteModified, x::Real)
    _, w = _discrete_grid(d)
    n = length(d.effect)
    b = floor(Int, x / w) + 1
    return (x < 0 || b < 1 || b > n) ? 0 : b
end

@doc "

Compute the probability mass for the interval containing `x` on the discrete
per-bin hazard path.

See also: [`logpdf`](@ref), [`cdf`](@ref)
"
function pdf(d::_DiscreteModified, x::Real)
    b = _discrete_bin(d, x)
    p = _modified_pmf(d)
    return b == 0 ? zero(eltype(p)) : p[b]
end

@doc "

Compute the log probability mass on the discrete per-bin hazard path.

See also: [`pdf`](@ref)
"
function logpdf(d::_DiscreteModified, x::Real)
    return log(pdf(d, x))
end

@doc "

Compute the cumulative distribution function on the discrete per-bin path.

See also: [`logcdf`](@ref)
"
function cdf(d::_DiscreteModified, x::Real)
    p = _modified_pmf(d)
    b = _discrete_bin(d, x)
    if x < 0
        return zero(eltype(p))
    elseif b == 0
        return one(eltype(p))
    end
    return sum(@view p[1:b])
end

function ccdf(d::_DiscreteModified, x::Real)
    return one(eltype(_modified_pmf(d))) - cdf(d, x)
end

function logcdf(d::_DiscreteModified, x::Real)
    return log(cdf(d, x))
end

function logccdf(d::_DiscreteModified, x::Real)
    return log(ccdf(d, x))
end

function Base.rand(rng::AbstractRNG, d::_DiscreteModified)
    p = _modified_pmf(d)
    _, w = _discrete_grid(d)
    u = rand(rng)
    acc = zero(eltype(p))
    @inbounds for b in eachindex(p)
        acc += p[b]
        u <= acc && return (b - 1) * w
    end
    return (length(p) - 1) * w
end

sampler(d::_DiscreteModified) = d
