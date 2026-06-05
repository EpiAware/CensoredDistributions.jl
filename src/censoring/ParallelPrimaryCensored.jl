# ============================================================================
# Parallel (shared-origin) primary-censored distribution
# ============================================================================
#
# `primary_censored([delay_1, ..., delay_n], primary_event)` extends
# `primary_censored` to a vector of delays. The result is a multivariate
# distribution over the event-time vector `[primary, observed_1, ...,
# observed_n]`: one shared primary (latent origin) draw `O` plus an independent
# branch delay per observation, `observed_i = O + delay_i`. Because every branch
# shares the same origin draw, the observations are dependent through that common
# additive offset (`Cov(observed_i, observed_j) = Var(O) > 0`).
#
# This is the design-B counterpart of the single-delay `primary_censored`:
# `logpdf` dispatches on the missingness of the event-time vector. The shared
# primary may be missing (marginalised by a single one-dimensional origin
# integral) or concrete (conditioned); each branch observation may be missing
# (marginalised) or present (conditioned). It supersedes the standalone
# `ParallelDistribution` from #312, reusing the same shared-origin maths: one
# 1-D origin integral regardless of `n`.
#
# Per-branch bounds come from each delay's own support. A user who wants a
# bounded branch passes a `truncated(delay, lower, upper)`: its `minimum` and
# `maximum` give the integration window and its `pdf`/`cdf` already self-
# normalise, so the joint density integrates to one with no separate
# normalisation constant.

@doc """

Joint distribution of several delays that branch from one shared primary event.

`ParallelPrimaryCensored` models a single latent origin (primary event) time
``O \\sim \\mathrm{primary\\_event}`` and ``n`` independent branch delays
``D_i``, with the observed times ``Y_i = O + D_i``. The event-time vector
``[O, Y_1, \\dots, Y_n]`` is a multivariate realisation. Because every branch
shares the *same* origin draw, the branches are dependent: the origin is a
common additive offset, so ``\\mathrm{Cov}(Y_i, Y_j) = \\mathrm{Var}(O) > 0``
for ``i \\ne j``. The joint density is therefore not the product of the branch
marginals (see the coupling note below). This is the "parallel" (shared-origin)
construction, distinct from [`primary_censored`](@ref) with a single delay (one
branch).

Like the single-delay [`Auto`](@ref) formulation, `logpdf` dispatches on the
missingness of the event-time vector:

- a missing primary is marginalised by a single one-dimensional origin integral,
- a concrete primary conditions on that origin,
- a missing branch observation is marginalised (it drops out of the joint),
- a present branch observation is conditioned on.

# Joint density (marginalised primary)

With every branch present and the primary marginalised,

```math
f(y_1, \\dots, y_n)
  = \\int_{-\\infty}^{\\infty} f_O(o) \\prod_{i=1}^{n} f_{D_i}(y_i - o)\\;
    \\mathrm{d}o,
```

where each ``f_{D_i}`` is the branch delay density (already normalised over its
own support, including any truncation). The supports restrict the origin to the
single interval
``[\\,\\max_i (y_i - \\max D_i),\\ \\min_i (y_i - \\min D_i)\\,]``, intersected
with ``\\mathrm{support}(O)``, giving one finite integration window. Missing
branches contribute no factor and do not narrow the window.

# Conditional density (concrete primary)

With a concrete primary ``p`` the joint factorises into the primary prior plus
the present-branch delay densities at the implied delays ``y_i - p``,

```math
\\log f(p, y_1, \\dots, y_n)
  = \\log f_O(p) + \\sum_{i : y_i \\text{ present}} \\log f_{D_i}(y_i - p).
```

# Bounded branches

A bounded branch is specified by passing a `truncated(delay, lower, upper)` as
that delay. Its support ``[\\,\\mathrm{lower}, \\mathrm{upper}\\,]`` bounds the
origin window and its density already integrates to one over that support, so

```math
\\int \\cdots \\int f(y_1, \\dots, y_n)\\, \\mathrm{d}y_1 \\cdots \\mathrm{d}y_n
  = \\int f_O(o) \\prod_{i=1}^{n}
      \\Bigl(\\int f_{D_i}(y_i - o)\\, \\mathrm{d}y_i\\Bigr) \\mathrm{d}o
  = \\int f_O(o)\\, \\mathrm{d}o = 1.
```

No separate normalisation constant is needed: the per-branch mass is one by
construction. The `child_bounds` keyword is retained for back-compatibility and
is sugar for wrapping each delay in `truncated(delay, lower, upper)`.

# Coupling: the branches are not independent

The shared origin couples the branches. From ``Y_i = O + D_i`` with independent
``D_i`` and a shared ``O``,

```math
\\mathrm{Cov}(Y_i, Y_j) = \\mathrm{Var}(O) > 0 \\quad (i \\ne j),
```

so the joint density differs from the product of the branch marginals,
``f(y_1, \\dots, y_n) \\ne \\prod_i f_{Y_i}(y_i)``. Treating the branches as
independent (summing the marginal `logpdf`s) discards this covariance; the joint
`logpdf` here integrates over the *single* shared origin and so retains it.

# Relationship to single-delay `primary_censored`

For ``n = 1`` and an untruncated delay the branch marginal equals
`primary_censored(delays[1], primary_event)`: the `[missing, y]` `logpdf` and
the joint `cdf` of the one-element vector match the scalar primary-censored
numerics.

# Fields

The `primary_event` field holds the shared latent origin distribution and
`delays` the tuple of branch delay distributions (each already carrying any
truncation), `solver` the origin quadrature rule.

# See also
- [`primary_censored`](@ref): constructor (vector of delays selects this type)
- [`primary_prior`](@ref): the prior over the shared primary event time
"""
struct ParallelPrimaryCensored{
    P <: UnivariateDistribution, C <: Tuple, S} <:
       Distribution{Multivariate, Continuous}
    "The shared latent origin (primary event) distribution."
    primary_event::P
    "Tuple of independent branch delay distributions, one per observation.
    Each carries its own (possibly truncated) support and normalisation."
    delays::C
    "Quadrature solver for the one-dimensional origin marginalisation."
    solver::S

    function ParallelPrimaryCensored(
            primary_event::P, delays::C, solver::S) where {
            P <: UnivariateDistribution, C <: Tuple, S}
        length(delays) >= 1 ||
            throw(ArgumentError(
                "ParallelPrimaryCensored needs at least one delay"))
        all(d -> d isa UnivariateDistribution, delays) ||
            throw(ArgumentError(
                "all delays must be UnivariateDistributions"))
        new{P, C, S}(primary_event, delays, solver)
    end
end

# ----------------------------------------------------------------------------
# Constructor entry point: vector of delays -> ParallelPrimaryCensored
# ----------------------------------------------------------------------------

@doc """

Create a parallel (shared-origin) primary event censored distribution from a
vector of delays.

Each observation branches from one shared primary event (latent origin); branch
`i` observes the origin plus an independent `delays[i]`. The result is a
multivariate distribution over the event-time vector `[primary, observed_1, ...,
observed_n]`. With `n = 1` and an untruncated delay this reduces to the
single-delay [`primary_censored`](@ref) numerics. See
[`ParallelPrimaryCensored`](@ref) for the joint density and the bounded-branch
treatment.

Bound a branch by passing a `truncated(delay, lower, upper)` as that delay: its
own support bounds the origin window and its density self-normalises, so the
joint density integrates to one with no separate normalisation constant.

# Arguments
- `delays`: a vector of branch delay `UnivariateDistribution`s (length
  `n >= 1`). Each may be a plain delay, a `truncated(delay, ...)`, or any
  continuous-`pdf`/`cdf` univariate distribution.
- `primary_event`: the shared primary event `UnivariateDistribution`.

# Keyword Arguments
- `child_bounds`: deprecated sugar for bounding the branches. A vector of
  `(lower, upper)` pairs (one per delay) wraps each delay in
  `truncated(delay, lower, upper)`. Prefer passing `truncated` delays directly.
- `solver`: quadrature solver for the origin marginalisation (default:
  `GaussLegendre(; n = 64)`, AD-friendly).

# Examples
```@example
using CensoredDistributions, Distributions

# Two observations sharing one daily infection window
d = primary_censored([Gamma(2.0, 1.0), LogNormal(1.0, 0.5)], Uniform(0.0, 1.0))

# Marginalise the shared primary (missing) over both observed branches
lp_joint = logpdf(d, [missing, 2.0, 3.0])

# Condition on a concrete primary p = 0.3: the joint factorises into the
# primary prior plus each branch delay density at the implied delay y_i - p
lp_cond = logpdf(d, [0.3, 2.0, 3.0])

# A missing branch (the second observation) is marginalised out of the joint
lp_drop = logpdf(d, [0.3, 2.0, missing])

# Draw the event-time vector [primary, observed_1, observed_2]
sample = rand(d)

# A bounded branch: pass a truncated delay (here branch 1 is capped at 4 days)
d_bounded = primary_censored(
    [truncated(Gamma(2.0, 1.0), 0.0, 4.0), LogNormal(1.0, 0.5)],
    Uniform(0.0, 1.0))
```

# See also
- [`ParallelPrimaryCensored`](@ref): the distribution type
- [`primary_censored`](@ref): the single-delay scalar counterpart
"""
function primary_censored(
        delays::AbstractVector,
        primary_event::UnivariateDistribution;
        child_bounds = nothing, solver = GaussLegendre(; n = 64))
    length(delays) >= 1 ||
        throw(ArgumentError("primary_censored needs at least one delay"))
    all(d -> d isa UnivariateDistribution, delays) ||
        throw(ArgumentError("all delays must be UnivariateDistributions"))
    bounded = child_bounds === nothing ? delays :
              _apply_child_bounds(delays, child_bounds)
    delays_t = Tuple(bounded)
    return ParallelPrimaryCensored(primary_event, delays_t, solver)
end

# Back-compat: `child_bounds` wraps each delay in a `truncated` with the given
# (lower, upper). A truncated delay self-normalises, so this is exactly
# equivalent to the caller passing the truncated delays directly. An infinite
# pair leaves the branch untruncated.
function _apply_child_bounds(delays, child_bounds)
    length(child_bounds) == length(delays) ||
        throw(ArgumentError(
            "child_bounds must have one (lower, upper) pair per delay"))
    return map(delays, child_bounds) do d, b
        length(b) == 2 && b[1] <= b[2] ||
            throw(ArgumentError(
                "each child bound must be (lower, upper) with lower <= upper"))
        lo, hi = float(b[1]), float(b[2])
        (isfinite(lo) || isfinite(hi)) ? truncated(d, lo, hi) : d
    end
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
# Non-numeric leaves (e.g. a `nothing` bound in a `Truncated`'s params) carry no
# tangent and contribute `Float64` to the promotion.
_parallel_leaf_eltype(x::Real) = float(typeof(x))
_parallel_leaf_eltype(::Any) = Float64
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
branch delay draw. The shared primary couples the branches.

See also: [`logpdf`](@ref)
"
function Distributions._rand!(
        rng::AbstractRNG, d::ParallelPrimaryCensored,
        x::AbstractVector{<:Real})
    o = rand(rng, d.primary_event)
    x[1] = o
    @inbounds for i in eachindex(d.delays)
        x[i + 1] = o + rand(rng, d.delays[i])
    end
    return x
end

# ----------------------------------------------------------------------------
# Origin integration window (present branches only)
# ----------------------------------------------------------------------------

# The single origin interval [lo, hi] on which the present-branch product kernel
# is non-zero: intersection of support(primary_event) with each present branch's
# [y_i - max(D_i), y_i - min(D_i)]. Each delay's own support (including any
# truncation) supplies the bound, so missing branches do not constrain the
# window.
function _parallel_origin_window(
        d::ParallelPrimaryCensored, present, yvals)
    lo = float(minimum(d.primary_event))
    hi = float(maximum(d.primary_event))
    @inbounds for k in eachindex(present)
        i = present[k]
        a = float(minimum(d.delays[i]))
        b = float(maximum(d.delays[i]))
        y = yvals[k]
        # y_i - o ∈ support(D_i) = [a, b]  ⇔  o ∈ [y_i - b, y_i - a].
        lo = max(lo, y - b)
        hi = min(hi, y - a)
    end
    return lo, hi
end

# Product of present-branch delay densities at (y_i - o). pdf is differentiable
# for the supported branch kinds and is zero outside each branch's support.
function _parallel_pdf_product(d, present, yvals, o)
    p = one(promote_type(typeof(o), eltype(yvals)))
    @inbounds for k in eachindex(present)
        i = present[k]
        p *= pdf(d.delays[i], yvals[k] - o)
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

# Collect the present (non-missing) branch indices and their concrete observed
# values from the event-time vector `x` (whose first slot is the primary).
# Shared by the marginal logpdf and the joint cdf. Returns `nothing` for the
# value vector when any present observation is NaN, so the caller propagates it.
function _parallel_present(d::ParallelPrimaryCensored, x, ::Type{T}) where {T}
    present = Int[]
    vals = T[]
    @inbounds for i in eachindex(d.delays)
        xi = x[i + 1]
        xi === missing && continue
        isnan(xi) && return present, nothing
        push!(present, i)
        push!(vals, convert(T, xi))
    end
    return present, vals
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
the joint). Each branch density is already normalised over its (possibly
truncated) support, so the joint density integrates to one with no separate
normalisation constant.

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

# Concrete primary: log f_O(p) + Σ_present log f_{D_i}(y_i - p). Missing branches
# drop out. Pulls concrete values out before arithmetic so no Union{Missing}
# reaches the differentiated path.
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
        insupport(d.delays[i], u) || return convert(T, -Inf)
        lp += convert(T, logpdf(d.delays[i], u))
    end
    return lp
end

# Missing primary: marginalise the shared origin over the present branches.
# Missing branches contribute no factor and do not narrow the origin window.
function _parallel_marginal_logpdf(d::ParallelPrimaryCensored, x)
    T = promote_type(eltype(x) <: Real ? eltype(x) : Float64,
        _parallel_eltype(d))

    present, yvals = _parallel_present(d, x, T)
    yvals === nothing && return convert(T, NaN)

    # No present branch: nothing to condition on, the joint over an all-missing
    # observation marginalises to 1 (log 0).
    isempty(present) && return zero(T)

    lower, upper = _parallel_origin_window(d, present, yvals)
    upper > lower || return convert(T, -Inf)

    kernel(o) = _parallel_pdf_product(d, present, yvals, o)
    val = _parallel_quadrature(
        d.primary_event, kernel, lower, upper, d.solver)
    val <= 0 && return convert(T, -Inf)
    return convert(T, log(val))
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
from the joint.

See also: [`logcdf`](@ref)
"
function cdf(d::ParallelPrimaryCensored, x::AbstractVector)
    length(x) == length(d) ||
        throw(DimensionMismatch(
            "observation vector length must be 1 + number of delays"))
    T = promote_type(eltype(x) <: Real ? eltype(x) : Float64,
        _parallel_eltype(d))

    present, xvals = _parallel_present(d, x, T)
    xvals === nothing && return convert(T, NaN)
    isempty(present) && return one(T)

    lower = float(minimum(d.primary_event))
    upper = float(maximum(d.primary_event))
    # Above x_i - min(D_i) every branch CDF is saturated; clip the window.
    @inbounds for k in eachindex(present)
        i = present[k]
        upper = min(upper, xvals[k] - float(minimum(d.delays[i])))
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

@doc """

Truncate a [`ParallelPrimaryCensored`](@ref) to a real-time observation
horizon.

`truncated(d, horizon)` represents a per-record real-time observation: branches
whose observed time has been seen by `horizon` are conditioned on, while
not-yet-seen branches (passed as `missing` in the observation) enter the
joint-over-origin truncation denominator. This is the multivariate,
shared-origin analogue of single-delay real-time truncation. The truncated
density is

```math
f_T([p, y_1, \\dots, y_n] \\mid \\text{horizon})
  = \\frac{f([p, y_1, \\dots, y_n])}{
      P(\\text{all branches} \\le \\text{horizon})},
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
struct TruncatedParallelPrimaryCensored{
    D <: ParallelPrimaryCensored, T <: Real} <:
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

function primary_prior(d::TruncatedParallelPrimaryCensored)
    return primary_prior(d.untruncated)
end

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
