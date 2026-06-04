@doc raw"""

Joint distribution of several delays that branch from one shared origin.

`ParallelDistribution` models a single latent origin time ``O \sim
\mathrm{primary\_event}`` and ``n`` independent child (branch) delays
``D_i``, with the observed children ``Y_i = O + D_i``. The observation
``(y_1, \dots, y_n)`` is a multivariate realisation. Because every child
shares the *same* origin draw, the children are dependent: the origin is a
common additive offset. This is the "parallel" construction, distinct from
[`Convolved`](@ref) (a *sum* of delays giving one scalar) and from
[`primary_censored`](@ref) (one delay, one observation).

The latent origin is marginalised by a single one-dimensional integral,
regardless of ``n``. No PPL, sampling, or latent-variable augmentation is
used: the marginal density and CDF are pure Distributions.jl quadrature.

# Joint density

```math
f(y_1, \dots, y_n)
  = \int f_O(o) \prod_{i=1}^{n} f_{D_i}(y_i - o)
      \prod_{i=1}^{n} \mathbf{1}[a_i \le y_i - o \le b_i]\; \mathrm{d}o,
```

where ``(a_i, b_i)`` are per-child bounds (truncation of the child delay).
The indicator restricts the origin to the single interval
``[\,\max_i (y_i - b_i),\ \min_i (y_i - a_i)\,]``, further intersected
with ``\mathrm{support}(O)``, giving one finite integration window
``[\mathrm{lo}, \mathrm{hi}]``.

# Joint CDF

```math
F(x_1, \dots, x_n)
  = \int f_O(o) \prod_{i=1}^{n} F_{D_i}(x_i - o)\; \mathrm{d}o.
```

# Bounded-child normalisation

With any finite bound the density above is the *unnormalised* joint mass
of the event "all children fall in their windows". Substituting
``u = y_i - o`` shows the per-child window mass
``P(a_i \le D_i \le b_i) = F_{D_i}(b_i) - F_{D_i}(a_i)`` does not depend on
the origin, so the origin density integrates out and the normalising
constant is the closed product

```math
Z = \prod_{i=1}^{n} \bigl(F_{D_i}(b_i) - F_{D_i}(a_i)\bigr),
```

precomputed at construction and subtracted in log space inside `logpdf`,
so the bounded `logpdf` integrates to one over the children. With every
bound ``(-\infty, +\infty)`` we have ``Z = 1`` and nothing is subtracted.

# Component distributions

Each child component is any continuous `UnivariateDistribution`: a plain
delay, a `truncated(delay, ...)`, or a [`primary_censored`](@ref) delay,
since these all expose a continuous `pdf`/`cdf` that the origin integrand
consumes directly. Interval-censored components
([`interval_censored`](@ref), [`double_interval_censored`](@ref)) carry
*discrete* interval mass rather than a density, so they are **not**
supported as components: dropping their PMF into the continuous origin
integrand would double-count the interval mass. Apply interval censoring
to the *observations* (e.g. per-branch bins) outside this distribution
instead.

# Relationship to `primary_censored`

For ``n = 1`` and infinite bounds the marginal of the single child equals
`primary_censored(delays[1], primary_event)`: `cdf`/`logpdf` of the
one-element vector match the scalar primary-censored numerics.

# Fields

The `primary_event` field holds the latent origin distribution, `delays`
the tuple of child branch distributions, `child_bounds` the per-child
``(a_i, b_i)`` truncation pairs, `solver` the origin quadrature rule, and
`log_norm` the precomputed ``\log Z`` normalisation constant.

# See also
- [`parallel_distribution`](@ref): Constructor function
- [`primary_censored`](@ref): the one-child scalar special case
- [`Convolved`](@ref): a *sum* of delays (scalar), not a shared origin
- [`primary_prior`](@ref): the prior over the latent origin
"""
struct ParallelDistribution{
    P <: UnivariateDistribution, C <: Tuple, B <: Tuple, S, L <: Real} <:
       Distribution{Multivariate, Continuous}
    "The latent origin (primary event) distribution."
    primary_event::P
    "Tuple of independent child (branch) delay distributions, one per
    observation."
    delays::C
    "Per-child `(lower, upper)` bounds for the child delay; `(-Inf, Inf)`
    means the child is untruncated."
    child_bounds::B
    "Quadrature solver for the one-dimensional origin marginalisation."
    solver::S
    "Precomputed log-normalisation constant `log Z`; `0.0` when every
    child bound is infinite."
    log_norm::L

    function ParallelDistribution(
            primary_event::P, delays::C, child_bounds::B, solver::S,
            log_norm::L) where {
            P <: UnivariateDistribution, C <: Tuple, B <: Tuple, S, L <: Real}
        length(delays) >= 1 ||
            throw(ArgumentError("ParallelDistribution needs ≥ 1 delay"))
        all(d -> d isa UnivariateDistribution, delays) ||
            throw(ArgumentError("all delays must be UnivariateDistributions"))
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

@doc "

Construct a [`ParallelDistribution`](@ref) joint delay distribution.

One latent origin event is drawn from `primary_event`; each child
observation is that origin plus an independent branch delay. The latent
origin is marginalised by a single one-dimensional quadrature, so the
result is a pure Distributions.jl multivariate distribution with no PPL
dependency.

# Arguments
- `primary_event`: the latent origin (primary event) `UnivariateDistribution`.
- `delays...`: one or more child delay `UnivariateDistribution`s. Each may
  be a plain delay, a `truncated(delay, ...)`, or a
  [`primary_censored`](@ref) delay (anything with a continuous
  `pdf`/`cdf`). The number of delays sets `length(d)`. Interval-censored
  components are not supported (see [`ParallelDistribution`](@ref)).

# Keyword Arguments
- `child_bounds`: per-child `(lower, upper)` truncation bounds for the
  child delay, as a vector/tuple of pairs (one per delay). Defaults to
  `(-Inf, Inf)` for every child. Any finite bound triggers precomputation
  of the log-normalisation constant.
- `solver`: quadrature solver for the origin marginalisation. Defaults to
  `GaussLegendre(; n = 64)` (AD-safe fixed-node).

# Examples
```@example
using CensoredDistributions, Distributions

# Two children sharing a uniform origin window
d = parallel_distribution(
    Uniform(0.0, 1.0), Gamma(2.0, 1.0), LogNormal(1.0, 0.5))
lp = logpdf(d, [2.0, 3.0])
cp = cdf(d, [2.0, 3.0])

# A truncated second child
db = parallel_distribution(
    Uniform(0.0, 1.0), Gamma(2.0, 1.0), LogNormal(1.0, 0.5);
    child_bounds = [(-Inf, Inf), (0.0, 5.0)])
lpb = logpdf(db, [2.0, 3.0])
```

# See also
- [`ParallelDistribution`](@ref): The distribution type
"
function parallel_distribution(
        primary_event::UnivariateDistribution,
        delays::UnivariateDistribution...;
        child_bounds = nothing, solver = GaussLegendre(; n = 64))
    length(delays) >= 1 ||
        throw(ArgumentError("parallel_distribution needs ≥ 1 delay"))
    delays_t = delays
    bounds_t = child_bounds === nothing ?
               map(_ -> (-Inf, Inf), delays_t) :
               _normalise_bounds(Tuple(child_bounds))
    log_norm = _parallel_distribution_log_norm(delays_t, bounds_t)
    return ParallelDistribution(
        primary_event, delays_t, bounds_t, solver, log_norm)
end

# ---------------------------------------------------------------------------
# Interface: length / eltype / params / sampling
# ---------------------------------------------------------------------------

Base.length(d::ParallelDistribution) = length(d.delays)

function Base.eltype(::Type{<:ParallelDistribution{P, C}}) where {
        P, C <: Tuple}
    return mapreduce(eltype, promote_type, fieldtypes(C); init = eltype(P))
end

# Numeric element type carrying any AD tangent in the distribution
# parameters. `eltype` on a distribution *type* returns the sample type
# (e.g. `Float64` for `Gamma{Dual}`), which loses the Dual under
# ForwardDiff; the actual parameter type is recovered from `params`,
# recursing into the nested param tuples that composed branches (e.g.
# `Convolved`) expose so the leaf numeric type is found.
_leaf_eltype(x::Real) = float(typeof(x))
function _leaf_eltype(t::Tuple)
    return isempty(t) ? Float64 :
           mapreduce(_leaf_eltype, promote_type, t)
end

function _param_eltype(dist::UnivariateDistribution)
    p = params(dist)
    return isempty(p) ? Float64 : mapreduce(_leaf_eltype, promote_type, p)
end

function _parallel_distribution_eltype(d::ParallelDistribution)
    T = _param_eltype(d.primary_event)
    for c in d.delays
        T = promote_type(T, _param_eltype(c))
    end
    return T
end

function params(d::ParallelDistribution)
    return (params(d.primary_event), map(params, d.delays)...)
end

@doc "

Return the latent-origin prior of a [`ParallelDistribution`](@ref).

The shared origin is the primary event distribution. Mirrors
[`primary_prior`](@ref) for `PrimaryCensored`, so downstream code can read
the prior over the marginalised origin.

See also: [`primary_prior`](@ref)
"
primary_prior(d::ParallelDistribution) = d.primary_event

# Whether any child bound is finite (triggers normalisation).
function _has_finite_child_bounds(d::ParallelDistribution)
    return any(b -> isfinite(b[1]) || isfinite(b[2]), d.child_bounds)
end

# Sample the shared origin once, then each child delay (truncated to its
# bounds when finite) added to that origin.
function Distributions._rand!(
        rng::AbstractRNG, d::ParallelDistribution, x::AbstractVector{<:Real})
    o = rand(rng, d.primary_event)
    @inbounds for i in eachindex(d.delays)
        x[i] = o + rand(rng, _maybe_truncate(d.delays[i], d.child_bounds[i]))
    end
    return x
end

# ---------------------------------------------------------------------------
# Origin integration window
# ---------------------------------------------------------------------------

# The single origin interval [lo, hi] on which the product kernel is
# non-zero for observation `y`: intersection of support(primary_event)
# with each child's [y_i - b_i, y_i - a_i]. Returns a (lo, hi) pair; the
# window is empty when hi <= lo.
function _origin_window(d::ParallelDistribution, y::AbstractVector{<:Real})
    lo = float(minimum(d.primary_event))
    hi = float(maximum(d.primary_event))
    @inbounds for i in eachindex(d.delays)
        a = d.child_bounds[i][1]
        b = d.child_bounds[i][2]
        # y_i - o ∈ [a, b]  ⇔  o ∈ [y_i - b, y_i - a].
        lo = _max2(lo, y[i] - b)
        hi = _min2(hi, y[i] - a)
    end
    return lo, hi
end

# Effective child lower support after intersecting the delay with its
# bound; used to clip the origin window in `cdf` (above x_i - this edge
# every child CDF is saturated).
_child_min(c::UnivariateDistribution, b) = _max2(minimum(c), b[1])

# ---------------------------------------------------------------------------
# Shared fixed-domain Gauss-Legendre quadrature over the origin
# ---------------------------------------------------------------------------

# Product-of-densities / product-of-CDFs quadrature over the origin.
#   ∫ f_O(o) · kernel(o) do   over o ∈ [lower, upper],
# mapped to the fixed reference domain (-1, 1) with the bounds carried as
# `params` and the change of variable o = m + h·s applied inside the
# integrand, so the Integrals.jl reverse rule stays on the parameter
# tangent (AD-safe). Mirrors `_convolved_quadrature` in Convolved.jl.
function _parallel_distribution_quadrature(
        primary_event, kernel::F, lower, upper, solver) where {F}
    m = (lower + upper) / 2
    h = (upper - lower) / 2

    function integrand(s, p)
        m_, h_ = p
        o = m_ + h_ * s
        return h_ * pdf(primary_event, o) * kernel(o)
    end

    prob = IntegralProblem(integrand, (-one(m), one(m)), (m, h))
    return solve(prob, solver)[1]
end

# Product of child delay densities at (y_i - o), with the bound indicator.
# pdf(component, ·) is differentiable for the supported component kinds
# (plain / truncated / primary-censored); the AD-fragile path is the *CDF*
# via gamma_inc, handled through `_cdf_ad_safe` in the CDF kernel.
function _child_pdf_product(delays, bounds, o, y)
    p = one(promote_type(typeof(o), eltype(y)))
    @inbounds for i in eachindex(delays)
        u = y[i] - o
        (bounds[i][1] <= u <= bounds[i][2]) || return zero(p)
        p *= pdf(delays[i], u)
    end
    return p
end

# Product of child delay CDFs at (x_i - o).
function _child_cdf_product(delays, o, x)
    p = one(promote_type(typeof(o), eltype(x)))
    @inbounds for i in eachindex(delays)
        p *= _cdf_ad_safe(delays[i], x[i] - o)
    end
    return p
end

# ---------------------------------------------------------------------------
# Normalisation constant
# ---------------------------------------------------------------------------

# log Z = log P(all children fall in their windows). Substituting
# u = y_i - o in the joint-density integral shows the per-child window mass
# is P(a_i ≤ D_i ≤ b_i) = F_{D_i}(b_i) - F_{D_i}(a_i), independent of the
# origin o, so the origin density integrates out to 1 and
#   Z = Π_i (F_{D_i}(b_i) - F_{D_i}(a_i)).
# No quadrature is needed. Returns zero(T) (log 1) when every bound is
# infinite (each factor is then 1).
function _parallel_distribution_log_norm(delays, bounds)
    # Element type carries any AD tangent in the delay parameters so a
    # bounded `logpdf` differentiates through the normalisation too.
    T = float(mapreduce(_param_eltype, promote_type, delays))

    any(b -> isfinite(b[1]) || isfinite(b[2]), bounds) ||
        return zero(T)

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

# ---------------------------------------------------------------------------
# logpdf / pdf
# ---------------------------------------------------------------------------

@doc "

Compute the joint log probability density of a parallel observation.

Marginalises the latent origin by a single one-dimensional quadrature
(AD-safe Gauss-Legendre). With finite child bounds the precomputed log
normalisation constant is subtracted so the density integrates to one over
the children.

See also: [`pdf`](@ref), [`ParallelDistribution`](@ref)
"
function Distributions._logpdf(
        d::ParallelDistribution, y::AbstractVector{<:Real})
    T = promote_type(eltype(y), _parallel_distribution_eltype(d))
    any(isnan, y) && return convert(T, NaN)

    lower, upper = _origin_window(d, y)
    upper > lower || return convert(T, -Inf)

    delays = d.delays
    bounds = d.child_bounds
    kernel(o) = _child_pdf_product(delays, bounds, o, y)
    val = _parallel_distribution_quadrature(
        d.primary_event, kernel, lower, upper, d.solver)

    val <= 0 && return convert(T, -Inf)
    return convert(T, log(val) - d.log_norm)
end

# ---------------------------------------------------------------------------
# cdf
# ---------------------------------------------------------------------------

@doc "

Compute the joint cumulative distribution function `P(Y_i ≤ x_i ∀ i)`.

Integrates the product of child delay CDFs at `x_i - o` against the origin
density over a single one-dimensional window. Bounded children are not
renormalised here (the CDF is the joint sub-distribution mass); use
`logpdf` for the normalised density.

See also: [`logcdf`](@ref), [`ParallelDistribution`](@ref)
"
function cdf(d::ParallelDistribution, x::AbstractVector{<:Real})
    T = promote_type(eltype(x), _parallel_distribution_eltype(d))
    any(isnan, x) && return convert(T, NaN)

    lower = float(minimum(d.primary_event))
    upper = float(maximum(d.primary_event))

    # Restrict to the origin range that contributes: above x_i - min(D_i)
    # every child CDF is fixed; below x_i - support the product is zero.
    @inbounds for i in eachindex(d.delays)
        upper = _min2(
            upper, x[i] - _child_min(d.delays[i], d.child_bounds[i]))
    end

    upper > lower || return zero(T)

    delays = d.delays
    kernel(o) = _child_cdf_product(delays, o, x)
    val = _parallel_distribution_quadrature(
        d.primary_event, kernel, lower, upper, d.solver)
    return clamp(convert(T, val), zero(T), one(T))
end

@doc "

Compute the joint log CDF of a parallel observation.

See also: [`cdf`](@ref)
"
function logcdf(d::ParallelDistribution, x::AbstractVector{<:Real})
    c = cdf(d, x)
    return c <= 0 ? oftype(float(c), -Inf) : log(c)
end
