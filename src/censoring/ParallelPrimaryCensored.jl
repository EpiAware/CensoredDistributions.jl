# ============================================================================
# Parallel (shared-origin) primary-censored distribution
#
# `primary_censored([delay_1, ..., delay_n], primary_event)` extends
# `primary_censored` to a VECTOR of delays. The result is a multivariate
# distribution over the event-time vector `[primary, observed_1, ...,
# observed_n]`: one shared primary (latent origin) draw `O` plus an
# independent branch delay per observation, `observed_i = O + delay_i`.
# Because every branch shares the same origin draw, the observations are
# dependent through that common additive offset.
#
# This is the design-B counterpart of the single-delay `primary_censored`:
# `logpdf` dispatches on the missingness of the event-time vector. The shared
# primary may be missing (marginalised by a single one-dimensional origin
# integral) or concrete (conditioned); each branch observation may be missing
# (marginalised) or present (conditioned). It supersedes the standalone
# `ParallelDistribution` from #312, reusing the same shared-origin maths: one
# 1-D origin integral regardless of `n`, and the origin-independent closed-form
# normaliser `Z = ∏ᵢ (F_{delay_i}(b_i) − F_{delay_i}(a_i))` for static child
# bounds.
# ============================================================================

@doc raw"""

Joint distribution of several delays that branch from one shared primary event.

`ParallelPrimaryCensored` models a single latent origin (primary event) time
``O \sim \mathrm{primary\_event}`` and ``n`` independent branch delays
``D_i``, with the observed times ``Y_i = O + D_i``. The event-time vector
``[O, Y_1, \dots, Y_n]`` is a multivariate realisation. Because every branch
shares the *same* origin draw, the branches are dependent: the origin is a
common additive offset. This is the "parallel" (shared-origin) construction,
distinct from [`primary_censored`](@ref) with a single delay (one branch).

Like the single-delay [`Auto`](@ref) formulation, `logpdf` dispatches on the
missingness of the event-time vector:

- a missing primary is marginalised by a single one-dimensional origin integral,
- a concrete primary conditions on that origin,
- a missing branch observation is marginalised (it drops out of the joint),
- a present branch observation is conditioned on.

# Joint density (marginalised primary)

With every branch present and the primary marginalised,

```math
f(y_1, \dots, y_n)
  = \int f_O(o) \prod_{i=1}^{n} f_{D_i}(y_i - o)
      \prod_{i=1}^{n} \mathbf{1}[a_i \le y_i - o \le b_i]\; \mathrm{d}o,
```

where ``(a_i, b_i)`` are the per-branch [`child_bounds`](@ref). The indicators
restrict the origin to the single interval
``[\,\max_i (y_i - b_i),\ \min_i (y_i - a_i)\,]``, intersected with
``\mathrm{support}(O)``, giving one finite integration window. Missing branches
contribute no factor and do not narrow the window.

# Conditional density (concrete primary)

With a concrete primary ``p`` the joint factorises into the primary prior plus
the present-branch delay densities at the implied delays ``y_i - p``,

```math
\log f(p, y_1, \dots, y_n)
  = \log f_O(p) + \sum_{i : y_i \text{ present}} \log f_{D_i}(y_i - p).
```

# Bounded-branch normalisation

Substituting ``u = y_i - o`` in the marginal integral shows the per-branch
window mass ``P(a_i \le D_i \le b_i) = F_{D_i}(b_i) - F_{D_i}(a_i)`` does not
depend on the origin, so the origin density integrates out and the normalising
constant is the closed product

```math
Z = \prod_{i=1}^{n} \bigl(F_{D_i}(b_i) - F_{D_i}(a_i)\bigr),
```

precomputed at construction and subtracted in log space inside `logpdf`. With
every bound ``(-\infty, +\infty)`` we have ``Z = 1`` and nothing is subtracted.

# Relationship to single-delay `primary_censored`

For ``n = 1`` and infinite bounds the branch marginal equals
`primary_censored(delays[1], primary_event)`: the `[missing, y]` `logpdf` and
the joint `cdf` of the one-element vector match the scalar primary-censored
numerics.

# See also
- [`primary_censored`](@ref): constructor (vector of delays selects this type)
- [`primary_prior`](@ref): the prior over the shared primary event time
"""
struct ParallelPrimaryCensored{
    P <: UnivariateDistribution, C <: Tuple, B <: Tuple, S, L <: Real} <:
       Distribution{Multivariate, Continuous}
    "The shared latent origin (primary event) distribution."
    primary_event::P
    "Tuple of independent branch delay distributions, one per observation."
    delays::C
    "Per-branch `(lower, upper)` bounds for the branch delay; `(-Inf, Inf)`
    means the branch is untruncated."
    child_bounds::B
    "Quadrature solver for the one-dimensional origin marginalisation."
    solver::S
    "Precomputed log-normalisation constant `log Z`; `0.0` when every branch
    bound is infinite."
    log_norm::L

    function ParallelPrimaryCensored(
            primary_event::P, delays::C, child_bounds::B, solver::S,
            log_norm::L) where {
            P <: UnivariateDistribution, C <: Tuple, B <: Tuple, S, L <: Real}
        length(delays) >= 1 ||
            throw(ArgumentError(
                "ParallelPrimaryCensored needs at least one delay"))
        all(d -> d isa UnivariateDistribution, delays) ||
            throw(ArgumentError(
                "all delays must be UnivariateDistributions"))
        length(child_bounds) == length(delays) ||
            throw(ArgumentError(
                "child_bounds must have one (lower, upper) pair per delay"))
        all(b -> length(b) == 2 && b[1] <= b[2], child_bounds) ||
            throw(ArgumentError(
                "each child bound must be (lower, upper) with lower <= upper"))
        new{P, C, B, S, L}(
            primary_event, delays, child_bounds, solver, log_norm)
    end
end

# Normalise a child_bounds specification to a tuple of float pairs.
function _parallel_normalise_bounds(child_bounds)
    return Tuple(
        (float(b[1]), float(b[2])) for b in child_bounds)
end

# Truncate a branch delay to its bounds when either side is finite, leaving an
# untruncated branch as-is. Used by sampling so each branch draw respects its
# static window.
function _parallel_maybe_truncate(d::UnivariateDistribution, bound)
    lo, hi = bound
    (isfinite(lo) || isfinite(hi)) || return d
    return truncated(d, lo, hi)
end

# ----------------------------------------------------------------------------
# Constructor entry point: vector of delays -> ParallelPrimaryCensored
# ----------------------------------------------------------------------------

@doc "

Create a parallel (shared-origin) primary event censored distribution from a
vector of delays.

Each observation branches from one shared primary event (latent origin); branch
`i` observes the origin plus an independent `delays[i]`. The result is a
multivariate distribution over the event-time vector `[primary, observed_1, ...,
observed_n]`. With `n = 1` this reduces to the single-delay
[`primary_censored`](@ref) numerics. See [`ParallelPrimaryCensored`](@ref) for
the joint density and the bounded-branch normalisation.

# Arguments
- `delays`: a vector of branch delay `UnivariateDistribution`s (length
  `n >= 1`). Each may be a plain delay, a `truncated(delay, ...)`, or any
  continuous-`pdf`/`cdf` univariate distribution.
- `primary_event`: the shared primary event `UnivariateDistribution`.

# Keyword Arguments
- `child_bounds`: per-branch `(lower, upper)` static truncation bounds for the
  branch delay, as a vector of pairs (one per delay). Defaults to `(-Inf, Inf)`
  for every branch; any finite bound triggers the closed-form normalisation.
- `solver`: quadrature solver for the origin marginalisation (default:
  `GaussLegendre(; n = 64)`, AD-friendly).

# Examples
```@example
using CensoredDistributions, Distributions

# Two observations sharing one daily infection window
d = primary_censored([Gamma(2.0, 1.0), LogNormal(1.0, 0.5)], Uniform(0.0, 1.0))

# Joint density with the primary marginalised
lp = logpdf(d, [missing, 2.0, 3.0])

# Conditioning on a concrete primary; a missing branch is marginalised
lp_cond = logpdf(d, [0.3, 2.0, missing])

# Draw the event-time vector [primary, observed_1, observed_2]
sample = rand(d)
```

# See also
- [`ParallelPrimaryCensored`](@ref): the distribution type
- [`primary_censored`](@ref): the single-delay scalar counterpart
"
function primary_censored(
        delays::AbstractVector,
        primary_event::UnivariateDistribution;
        child_bounds = nothing, solver = GaussLegendre(; n = 64))
    length(delays) >= 1 ||
        throw(ArgumentError("primary_censored needs at least one delay"))
    all(d -> d isa UnivariateDistribution, delays) ||
        throw(ArgumentError("all delays must be UnivariateDistributions"))
    delays_t = Tuple(delays)
    bounds_t = child_bounds === nothing ?
               map(_ -> (-Inf, Inf), delays_t) :
               _parallel_normalise_bounds(child_bounds)
    log_norm = _parallel_log_norm(delays_t, bounds_t)
    return ParallelPrimaryCensored(
        primary_event, delays_t, bounds_t, solver, log_norm)
end

# ----------------------------------------------------------------------------
# Interface: length / eltype / params / prior / sampling
# ----------------------------------------------------------------------------

# The event-time vector is [primary, observed_1, ..., observed_n].
Base.length(d::ParallelPrimaryCensored) = length(d.delays) + 1

function Base.eltype(::Type{<:ParallelPrimaryCensored{P, C}}) where {
        P, C <: Tuple}
    return mapreduce(eltype, promote_type, fieldtypes(C); init = eltype(P))
end

# Numeric element type carrying any AD tangent in the distribution parameters.
# `eltype` on a distribution *type* returns the sample type, losing a Dual; the
# parameter type is recovered from `params`, recursing into nested param tuples.
_parallel_leaf_eltype(x::Real) = float(typeof(x))
function _parallel_leaf_eltype(t::Tuple)
    return isempty(t) ? Float64 :
           mapreduce(_parallel_leaf_eltype, promote_type, t)
end

function _parallel_param_eltype(dist::UnivariateDistribution)
    p = params(dist)
    return isempty(p) ? Float64 :
           mapreduce(_parallel_leaf_eltype, promote_type, p)
end

function _parallel_eltype(d::ParallelPrimaryCensored)
    T = _parallel_param_eltype(d.primary_event)
    for c in d.delays
        T = promote_type(T, _parallel_param_eltype(c))
    end
    return T
end

function params(d::ParallelPrimaryCensored)
    return (params(d.primary_event), map(params, d.delays)...)
end

@doc "

Return the shared-origin prior of a [`ParallelPrimaryCensored`](@ref).

The shared origin is the primary event distribution. Mirrors
[`primary_prior`](@ref) for the single-delay `PrimaryCensored`.

See also: [`primary_prior`](@ref)
"
primary_prior(d::ParallelPrimaryCensored) = d.primary_event

@doc "

Test whether the event-time vector `[primary, observed_1, ..., observed_n]` is
in the joint support of a [`ParallelPrimaryCensored`](@ref). A missing primary
or a missing branch is admissible.

See also: [`logpdf`](@ref)
"
function insupport(d::ParallelPrimaryCensored, x::AbstractVector)
    length(x) == length(d) || return false
    p = x[1]
    if p === missing
        # Reachable for some origin in the window.
        return true
    end
    insupport(d.primary_event, p) || return false
    @inbounds for i in eachindex(d.delays)
        y = x[i + 1]
        y === missing && continue
        insupport(d.delays[i], y - p) || return false
    end
    return true
end

@doc "

Draw an event-time vector `[primary, observed_1, ..., observed_n]`: one shared
primary draw, then each branch observation as that primary plus an independent
(bound-truncated) branch delay draw.

See also: [`logpdf`](@ref)
"
function Distributions._rand!(
        rng::AbstractRNG, d::ParallelPrimaryCensored,
        x::AbstractVector{<:Real})
    o = rand(rng, d.primary_event)
    x[1] = o
    @inbounds for i in eachindex(d.delays)
        x[i + 1] = o +
                   rand(rng,
            _parallel_maybe_truncate(d.delays[i], d.child_bounds[i]))
    end
    return x
end

# ----------------------------------------------------------------------------
# Normalisation constant (origin-independent closed product)
# ----------------------------------------------------------------------------

# log Z = log P(all branches fall in their windows). Substituting u = y_i - o
# in the marginal integral shows the per-branch window mass is
# P(a_i <= D_i <= b_i) = F_{D_i}(b_i) - F_{D_i}(a_i), independent of the origin,
# so Z = Π_i (F_{D_i}(b_i) - F_{D_i}(a_i)). No quadrature is needed; returns
# zero(T) (log 1) when every bound is infinite.
function _parallel_log_norm(delays, bounds)
    T = float(mapreduce(_parallel_param_eltype, promote_type, delays))

    any(b -> isfinite(b[1]) || isfinite(b[2]), bounds) || return zero(T)

    logZ = zero(T)
    for (c, b) in zip(delays, bounds)
        hi = isfinite(b[2]) ? T(_cdf_ad_safe(c, b[2])) : one(T)
        lo = isfinite(b[1]) ? T(_cdf_ad_safe(c, b[1])) : zero(T)
        mass = hi - lo
        mass <= 0 && return convert(T, -Inf)
        logZ += log(mass)
    end
    return logZ
end

# Per-branch log window mass log(F(b_i) - F(a_i)); zero when the bound is
# infinite. Used to subtract only the *present* branches' contributions from
# the precomputed full `log_norm` when some branches are missing.
function _parallel_branch_log_mass(c::UnivariateDistribution, b, ::Type{T}) where {T}
    (isfinite(b[1]) || isfinite(b[2])) || return zero(T)
    hi = isfinite(b[2]) ? T(_cdf_ad_safe(c, b[2])) : one(T)
    lo = isfinite(b[1]) ? T(_cdf_ad_safe(c, b[1])) : zero(T)
    mass = hi - lo
    mass <= 0 && return convert(T, -Inf)
    return log(mass)
end

# ----------------------------------------------------------------------------
# Origin integration window (present branches only)
# ----------------------------------------------------------------------------

# The single origin interval [lo, hi] on which the present-branch product kernel
# is non-zero: intersection of support(primary_event) with each present branch's
# [y_i - b_i, y_i - a_i]. Missing branches do not constrain the window.
function _parallel_origin_window(
        d::ParallelPrimaryCensored, present, yvals)
    lo = float(minimum(d.primary_event))
    hi = float(maximum(d.primary_event))
    @inbounds for k in eachindex(present)
        i = present[k]
        a = d.child_bounds[i][1]
        b = d.child_bounds[i][2]
        y = yvals[k]
        # y_i - o ∈ [a, b]  ⇔  o ∈ [y_i - b, y_i - a].
        lo = max(lo, y - b)
        hi = min(hi, y - a)
    end
    return lo, hi
end

# Product of present-branch delay densities at (y_i - o), with the bound
# indicator. pdf is differentiable for the supported branch kinds.
function _parallel_pdf_product(d, present, yvals, o)
    p = one(promote_type(typeof(o), eltype(yvals)))
    @inbounds for k in eachindex(present)
        i = present[k]
        u = yvals[k] - o
        b = d.child_bounds[i]
        (b[1] <= u <= b[2]) || return zero(p)
        p *= pdf(d.delays[i], u)
    end
    return p
end

# Product-of-densities quadrature over the origin:
#   ∫ f_O(o) · kernel(o) do   over o ∈ [lower, upper],
# mapped to the fixed reference domain (-1, 1) with the bounds carried as
# `params` and the change of variable o = m + h·s applied inside the integrand,
# so the Integrals.jl reverse rule stays on the parameter tangent (AD-safe).
function _parallel_quadrature(
        primary_event, kernel::F, lower, upper, solver) where {F}
    m = (lower + upper) / 2
    h = (upper - lower) / 2

    function integrand(s, par)
        m_, h_ = par
        o = m_ + h_ * s
        return h_ * pdf(primary_event, o) * kernel(o)
    end

    prob = IntegralProblem(integrand, (-one(m), one(m)), (m, h))
    return solve(prob, solver)[1]
end

# ----------------------------------------------------------------------------
# logpdf / pdf (missingness dispatch)
# ----------------------------------------------------------------------------

@doc "

Log density of the event-time vector `[primary, observed_1, ..., observed_n]`
under [`ParallelPrimaryCensored`](@ref).

A missing primary marginalises the shared origin by a single one-dimensional
quadrature over the present branches; a concrete primary conditions on it
(`logpdf(primary_event, p)` plus the present-branch delay densities at
`observed_i - p`). A missing branch observation is marginalised (it drops from
the joint). With finite child bounds the present branches' log window mass is
subtracted so the density integrates to one.

Missingness is inspected only through control flow; concrete values alone enter
the differentiated arithmetic, so the log density differentiates on every
supported backend.

See also: [`pdf`](@ref), [`primary_prior`](@ref)
"
function logpdf(d::ParallelPrimaryCensored, x::AbstractVector)
    length(x) == length(d) ||
        throw(DimensionMismatch(
            "event-time vector length must be 1 + number of delays"))
    p = x[1]
    if p === missing
        return _parallel_marginal_logpdf(d, x)
    end
    return _parallel_conditional_logpdf(d, x)
end

# Concrete primary: log f_O(p) + Σ_present log f_{D_i}(y_i - p) - Σ_present logZ_i.
# Missing branches drop out. Pulls concrete values out before arithmetic so no
# Union{Missing} reaches the differentiated path.
function _parallel_conditional_logpdf(d::ParallelPrimaryCensored, x)
    T = promote_type(eltype(x) <: Real ? eltype(x) : Float64,
        _parallel_eltype(d))
    p = x[1]
    p isa Real || return convert(T, -Inf)
    isnan(p) && return convert(T, NaN)
    insupport(d.primary_event, p) || return convert(T, -Inf)

    lp = convert(T, logpdf(d.primary_event, p))
    @inbounds for i in eachindex(d.delays)
        y = x[i + 1]
        y === missing && continue
        isnan(y) && return convert(T, NaN)
        u = y - p
        b = d.child_bounds[i]
        (b[1] <= u <= b[2]) || return convert(T, -Inf)
        lp += convert(T, logpdf(d.delays[i], u))
        lp -= _parallel_branch_log_mass(d.delays[i], b, T)
    end
    return lp
end

# Missing primary: marginalise the shared origin over the present branches.
# Missing branches contribute no factor and do not narrow the origin window.
function _parallel_marginal_logpdf(d::ParallelPrimaryCensored, x)
    T = promote_type(eltype(x) <: Real ? eltype(x) : Float64,
        _parallel_eltype(d))

    # Collect present branch indices and their concrete observed values.
    present = Int[]
    yvals = T[]
    @inbounds for i in eachindex(d.delays)
        y = x[i + 1]
        y === missing && continue
        isnan(y) && return convert(T, NaN)
        push!(present, i)
        push!(yvals, convert(T, y))
    end

    # No present branch: nothing to condition on, the joint over an all-missing
    # observation marginalises to 1 (log 0).
    isempty(present) && return zero(T)

    lower, upper = _parallel_origin_window(d, present, yvals)
    upper > lower || return convert(T, -Inf)

    kernel(o) = _parallel_pdf_product(d, present, yvals, o)
    val = _parallel_quadrature(
        d.primary_event, kernel, lower, upper, d.solver)
    val <= 0 && return convert(T, -Inf)

    # Subtract only the present branches' window mass.
    logZ = zero(T)
    @inbounds for i in present
        logZ += _parallel_branch_log_mass(d.delays[i], d.child_bounds[i], T)
    end
    return convert(T, log(val) - logZ)
end

@doc "

Density of the event-time vector `[primary, observed_1, ..., observed_n]` under
[`ParallelPrimaryCensored`](@ref).

See also: [`logpdf`](@ref)
"
function pdf(d::ParallelPrimaryCensored, x::AbstractVector)
    return exp(logpdf(d, x))
end

# Distributions.jl multivariate hook: `_logpdf` for concrete-Real vectors so the
# generic `logpdf(::Multivariate, ::AbstractVector)` machinery (and the
# all-present marginal path used by the cdf tests) dispatches here too.
function Distributions._logpdf(
        d::ParallelPrimaryCensored, x::AbstractVector{<:Real})
    # A fully concrete observation conditions on the given primary.
    return _parallel_conditional_logpdf(d, x)
end

# ----------------------------------------------------------------------------
# Joint CDF over the observations (shared-origin integral)
# ----------------------------------------------------------------------------

# Product of branch delay CDFs at (x_i - o) over the present branches.
function _parallel_cdf_product(d, present, xvals, o)
    p = one(promote_type(typeof(o), eltype(xvals)))
    @inbounds for k in eachindex(present)
        i = present[k]
        p *= _cdf_ad_safe(d.delays[i], xvals[k] - o)
    end
    return p
end

@doc "

Joint cumulative distribution function over the branch observations,
`P(Y_i ≤ x_i for all present i)`, with the shared primary marginalised.

The observation vector is `[primary, observed_1, ..., observed_n]`; the primary
slot is ignored (the origin is marginalised) and a missing branch is dropped
from the joint. Bounded branches are not renormalised here (this is the joint
sub-distribution mass); use `logpdf` for the normalised density.

See also: [`logcdf`](@ref)
"
function cdf(d::ParallelPrimaryCensored, x::AbstractVector)
    length(x) == length(d) ||
        throw(DimensionMismatch(
            "observation vector length must be 1 + number of delays"))
    T = promote_type(eltype(x) <: Real ? eltype(x) : Float64,
        _parallel_eltype(d))

    present = Int[]
    xvals = T[]
    @inbounds for i in eachindex(d.delays)
        xi = x[i + 1]
        xi === missing && continue
        isnan(xi) && return convert(T, NaN)
        push!(present, i)
        push!(xvals, convert(T, xi))
    end
    isempty(present) && return one(T)

    lower = float(minimum(d.primary_event))
    upper = float(maximum(d.primary_event))
    # Above x_i - min(D_i) every branch CDF is saturated; clip the window.
    @inbounds for k in eachindex(present)
        i = present[k]
        cmin = max(minimum(d.delays[i]), d.child_bounds[i][1])
        upper = min(upper, xvals[k] - cmin)
    end
    upper > lower || return zero(T)

    kernel(o) = _parallel_cdf_product(d, present, xvals, o)
    val = _parallel_quadrature(
        d.primary_event, kernel, lower, upper, d.solver)
    return clamp(convert(T, val), zero(T), one(T))
end

@doc "

Joint log CDF over the branch observations.

See also: [`cdf`](@ref)
"
function logcdf(d::ParallelPrimaryCensored, x::AbstractVector)
    c = cdf(d, x)
    return c <= 0 ? oftype(float(c), -Inf) : log(c)
end

# ----------------------------------------------------------------------------
# Specialised real-time truncation
# ----------------------------------------------------------------------------

@doc raw"""

Truncate a [`ParallelPrimaryCensored`](@ref) to a real-time observation
horizon.

`truncated(d, horizon)` represents a per-record real-time observation: branches
whose observed time has been seen by `horizon` are conditioned on, while
not-yet-seen branches (passed as `missing` in the observation) enter the
joint-over-origin truncation denominator. This is the multivariate, shared-origin
analogue of single-delay real-time truncation. The truncated density is

```math
f_T([p, y_1, \dots, y_n] \mid \text{horizon})
  = \frac{f([p, y_1, \dots, y_n])}{P(\text{all branches} \le \text{horizon})},
```

where the denominator is the joint CDF [`cdf`](@ref) evaluated at `horizon` on
every branch (the shared-origin integral). A missing branch in the observation
marginalises in the numerator as usual.

# See also
- [`ParallelPrimaryCensored`](@ref): the untruncated distribution
- [`cdf`](@ref): the joint denominator
"""
function truncated(d::ParallelPrimaryCensored, horizon::Real)
    return TruncatedParallelPrimaryCensored(d, float(horizon))
end

@doc "

Real-time-truncated [`ParallelPrimaryCensored`](@ref): the joint shared-origin
distribution conditioned on every branch observation falling at or before a
real-time `horizon`.

# See also
- [`truncated`](@ref): constructor
- [`ParallelPrimaryCensored`](@ref): the untruncated distribution
"
struct TruncatedParallelPrimaryCensored{D <: ParallelPrimaryCensored, T <: Real} <:
       Distribution{Multivariate, Continuous}
    "The untruncated shared-origin distribution."
    untruncated::D
    "The real-time observation horizon; all branches are conditioned on
    being observed at or before this time."
    horizon::T
end

Base.length(d::TruncatedParallelPrimaryCensored) = length(d.untruncated)

function Base.eltype(::Type{<:TruncatedParallelPrimaryCensored{D}}) where {D}
    return eltype(D)
end

primary_prior(d::TruncatedParallelPrimaryCensored) = primary_prior(d.untruncated)

# Log denominator: log P(all branches <= horizon) via the joint CDF, evaluated
# at `horizon` on every branch (primary slot ignored). Computed once per logpdf.
function _parallel_truncation_logdenominator(
        d::TruncatedParallelPrimaryCensored, ::Type{T}) where {T}
    inner = d.untruncated
    n = length(inner.delays)
    xfull = Vector{Any}(undef, n + 1)
    xfull[1] = missing
    @inbounds for i in 1:n
        xfull[i + 1] = d.horizon
    end
    c = cdf(inner, xfull)
    return c <= 0 ? convert(T, -Inf) : convert(T, log(c))
end

@doc "

Log density of the event-time vector under a real-time-truncated
[`ParallelPrimaryCensored`](@ref): the untruncated joint log density minus the
log joint probability that all branches are observed by the horizon.

See also: [`truncated`](@ref), [`cdf`](@ref)
"
function logpdf(d::TruncatedParallelPrimaryCensored, x::AbstractVector)
    num = logpdf(d.untruncated, x)
    T = typeof(float(num))
    den = _parallel_truncation_logdenominator(d, T)
    return num - den
end

function pdf(d::TruncatedParallelPrimaryCensored, x::AbstractVector)
    return exp(logpdf(d, x))
end

# Rejection sampling against the horizon: redraw the shared-origin event-time
# vector until every branch observation falls at or before the horizon.
function Distributions._rand!(
        rng::AbstractRNG, d::TruncatedParallelPrimaryCensored,
        x::AbstractVector{<:Real})
    inner = d.untruncated
    while true
        Distributions._rand!(rng, inner, x)
        @inbounds if all(i -> x[i] <= d.horizon, 2:length(x))
            return x
        end
    end
end
