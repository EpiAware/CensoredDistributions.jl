@doc "

Latent-variable wrapper turning a primary-censored delay into its latent
representation, sampling the primary event time rather than integrating it out.

A latent primary-censored delay is multivariate over the event times
`[primary, observed]`: the primary event time is a sampled latent variable
rather than integrated out. `rand` produces `[primary, observed]` and
`logpdf([primary, observed])` is the primary prior plus the conditional of the
observed time given the primary.

Marginal versus latent is dispatch on the type: the plain
[`PrimaryCensored`](@ref) is the univariate marginal default, and wrapping it in
`Latent` selects the latent representation. Construct via [`latent`](@ref).

The `dist` field holds the wrapped primary-censored node.

# See also
- [`latent`](@ref): constructor
- [`PrimaryConditional`](@ref): the conditional scored and sampled here
- [`get_primary_event`](@ref): the primary prior sampled here
"
struct Latent{D} <: Distribution{Multivariate, Continuous}
    "The wrapped primary-censored node."
    dist::D
end

@doc "

Turn a primary-censored node into its latent representation.

Returns a [`Latent`](@ref) multivariate distribution over `[primary, observed]`.
A draw is a labelled `NamedTuple` `(primary = ..., observed = ...)`.

# Arguments
- `d`: A primary-censored node (for example from [`primary_censored`](@ref)).

# Examples
```@example
using CensoredDistributions, Distributions

d = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))
ld = latent(d)
rand(ld)
```

# See also
- [`marginal`](@ref): the inverse, recovering the wrapped marginal node.
"
latent(d) = Latent(d)

@doc "

Recover the marginal node a [`latent`](@ref) wraps (the inverse of `latent`).

`marginal(d)` is the inverse of [`latent`](@ref): it unwraps a [`Latent`](@ref)
back to the marginal node it carries, so `marginal(latent(d)) == d`. It is
IDEMPOTENT — a node that is not a `Latent` is returned unchanged, so
`marginal(d) == d` and `marginal(marginal(x)) == marginal(x)`. Use it to move
from the per-event latent view back to the collapsed marginal/observed view.

# Arguments
- `d`: A [`Latent`](@ref) to unwrap, or any node (returned unchanged).

# Examples
```@example
using CensoredDistributions, Distributions

d = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))
marginal(latent(d)) == d
```

# See also
- [`latent`](@ref): the forward direction this inverts.
"
marginal(d::Latent) = d.dist
marginal(d) = d

Base.length(::Latent) = 2
Base.eltype(::Type{<:Latent{D}}) where {D} = eltype(D)
params(d::Latent) = params(d.dist)

# Support of the observed-delay marginal: primary window plus the bare delay.
minimum(d::Latent) = minimum(get_primary_event(d)) + minimum(get_dist(d))
maximum(d::Latent) = maximum(get_primary_event(d)) + maximum(get_dist(d))
insupport(d::Latent, x::Real) = minimum(d) <= x <= maximum(d)

@doc "

Draw a labelled latent event record `(primary = ..., observed = ...)`: a primary
event time from the primary prior, then the observed time from
[`PrimaryConditional`](@ref) given that primary. The draw is a `NamedTuple`
(self-labelling; the underlying scored representation is the vector
`[primary, observed]`).

See also: [`logpdf`](@ref)
"
function Base.rand(rng::AbstractRNG, d::Latent)
    p = rand(rng, get_primary_event(d))
    y = rand(rng, PrimaryConditional(d, p))
    return (primary = p, observed = y)
end

# Batch the count form into `n` independent labelled records (the north-star:
# `rand(d, n)` simulates `n` draws from the OBJECT). A `Latent` is multivariate
# but `rand(rng, d)` returns a labelled `NamedTuple`, not a numeric vector that
# can fill a matrix column, so the generic `rand(::Multivariate, ::Int)` matrix
# fallback recurses (StackOverflow). This terminating method returns one full
# record per draw, matching the record-aware `rand(d, rows)` batch shape. The
# `n::Integer` count form is disambiguated from `rand(d, rows::AbstractVector)`.
function Base.rand(rng::AbstractRNG, d::Latent, n::Integer)
    return [rand(rng, d) for _ in 1:n]
end

# Disambiguate against Distributions' `rand(::Sampleable{Multivariate,
# Continuous}, ::Int)`: `Latent` is more specific in the distribution argument
# but `Int` ties on the count, so spell out the `Int` method explicitly.
function Base.rand(rng::AbstractRNG, d::Latent, n::Int)
    invoke(
        rand, Tuple{AbstractRNG, Latent, Integer}, rng, d, n)
end

Base.rand(d::Latent, n::Integer) = rand(default_rng(), d, n)

# Disambiguate the no-rng count form against `rand(::Sampleable, ::Int...)`.
Base.rand(d::Latent, n::Int) = rand(default_rng(), d, n)

@doc "

Joint log density of the latent event record: the primary prior density plus
the [`PrimaryConditional`](@ref) of the observed time given the primary,
`logpdf(get_primary_event(d), p) + logpdf(PrimaryConditional(d, p), y)`.

Accepts either the scored vector `[primary, observed]` or the labelled
`NamedTuple` `(primary = ..., observed = ...)` (converted internally to the
scored vector). For the single-value observed marginal, call `logpdf(d, x)`
with a scalar `x`, which integrates this joint over the primary.

See also: [`PrimaryConditional`](@ref), [`rand`](@ref)
"
function logpdf(d::Latent, x::AbstractVector)
    p = x[1]
    y = x[2]
    return logpdf(get_primary_event(d), p) + logpdf(PrimaryConditional(d, p), y)
end

# Accept the labelled NamedTuple draw, converting to the scored `[primary,
# observed]` vector internally. Keying is BY NAME, so field order does not
# matter and an unexpected field errors.
function logpdf(d::Latent, x::NamedTuple)
    return logpdf(d, _latent_record_vector(x))
end

function _latent_record_vector(x::NamedTuple)
    Set(keys(x)) == Set((:primary, :observed)) || throw(ArgumentError(
        "a latent event record needs fields (:primary, :observed); got " *
        "$(collect(keys(x)))"))
    return [x.primary, x.observed]
end

# Observed-delay interface via the augmented-data integral
# --------------------------------------------------------
# The observed-delay density and distribution function are computed from the
# LATENT formulation: the primary event time is the augmented variable, and the
# observed marginal is the joint integrated over it. This is genuinely distinct
# from the analytic `primary_censored` marginal (the target), so comparing the
# two is a real validation rather than a tautology.
#
# With prior `p ~ get_primary_event(d)` and the bare continuous delay
# `get_dist(d)`, the conditional of the observed time is `delay` shifted by `p`
# (the sampled-origin rule the joint `logpdf(d, [p, y])` already uses), so
#   f_Y(y) = ∫ pdf(prior, p) · pdf(delay, y - p) dp     (= ∫ exp(logpdf(d,[p,y])) dp)
#   F_Y(x) = ∫ pdf(prior, p) · cdf(delay, x - p) dp
# over the primary window `[minimum(prior), maximum(prior)]`. The integral uses
# the SAME Gauss-Legendre solver the primary-censored numeric path uses.

# Resolve the quadrature solver: reuse the one carried by the wrapped
# primary-censored node so the latent integral matches the package's numeric
# path, falling back to the default 64-node rule for a bare delay.
function _latent_solver(d::Latent)
    node = _primary_censored_node(d)
    return node === nothing ? GaussLegendre(; n = _PRIMARY_NODES) :
           node.method.solver
end

# Primary-window integration bounds for an observed `x`. The conditional delay
# `delay` at the implied gap `x - p` only has mass where `x - p > minimum(delay)`,
# i.e. `p < x - minimum(delay)`. Clamping the upper bound to that point (rather
# than integrating the full window and relying on `pdf(delay, ≤ 0) = 0`) keeps
# every quadrature node strictly inside the delay support, away from the
# `logpdf(delay, 0⁺) → -∞` boundary singularity. That stabilises the integrand
# and, critically, its parameter gradient, mirroring how the analytic numeric
# `primary_censored` path clamps its own integration bounds. Returns `lo == hi`
# when `x` is below the support, signalling zero mass.
function _latent_bounds(d::Latent, x::Real)
    prior = get_primary_event(d)
    delay = get_dist(d)
    lo = minimum(prior)
    hi = min(maximum(prior), x - minimum(delay))
    return lo, max(lo, hi)
end

# Integrate `pdf(prior, p) * kernel(x - p)` over the (clamped) primary window.
# `kernel` is `pdf(delay, ·)` for the density and `cdf(delay, ·)` for the
# distribution function. AD-safe: a fixed-node weighted sum whose accumulator
# type is taken from the integrand.
function _latent_integral(d::Latent, x::Real, kernel::K) where {K}
    prior = get_primary_event(d)
    lo, hi = _latent_bounds(d, x)
    integrand(p) = pdf(prior, p) * kernel(x - p)
    return integrate(_latent_solver(d), integrand, lo, hi)
end

# Log of the density integral, computed in LOG SPACE over the Gauss-Legendre
# nodes so a tiny integrand (extreme parameters, far-tail observations) neither
# underflows to `log(0) = -Inf` nor poisons the gradient with `NaN`. This is the
# numerically robust analogue of `log(pdf(d, x))`: the linear integral takes the
# log of a sum that can round to zero, while this sums log-densities with
# `logsumexp` and never forms the tiny linear value. Used by the single-value
# observed `logpdf` so the latent model is as init-robust as the analytic
# marginal under Turing.
#
# The map `[-1, 1] -> [lo, hi]` contributes `log(h)`; each node contributes
# `log(w_i) + log(pdf(prior, p_i)) + logpdf(delay, x - p_i)`. Falls back to
# `log(pdf(d, x))` for a non-Gauss-Legendre solver (e.g. an Integrals.jl
# algorithm), which does its own integration.
function _latent_logpdf(d::Latent, x::Real)
    solver = _latent_solver(d)
    solver isa GaussLegendre || return log(pdf(d, x))
    prior = get_primary_event(d)
    delay = get_dist(d)
    lo, hi = _latent_bounds(d, x)
    hi <= lo && return oftype(float(x), -Inf)
    h = (hi - lo) / 2
    m = (lo + hi) / 2
    nodes, weights = solver.rule.nodes, solver.rule.weights
    log_terms = map(eachindex(nodes)) do i
        p = m + h * nodes[i]
        return log(weights[i]) + logpdf(prior, p) + logpdf(delay, x - p)
    end
    return log(h) + logsumexp(log_terms)
end

@doc "

Observed-delay density of `x` under the latent model, the joint integrated over
the primary, `∫ pdf(prior, p) · pdf(delay, x - p) dp`. Computed from the latent
(augmented-data) formulation by Gauss-Legendre quadrature over the primary
window, NOT the analytic `primary_censored` marginal. Distinct from the joint
`pdf(d, [p, y])`.

See also: [`cdf`](@ref), [`logpdf`](@ref)
"
pdf(d::Latent, x::Real) = _latent_integral(d, x, u -> pdf(get_dist(d), u))

@doc "

Observed-delay log density of `x` under the latent model, the single-value
observed marginal. Computed from the latent (augmented-data) formulation,
integrating the joint over the primary in LOG SPACE (a `logsumexp` over the
Gauss-Legendre nodes) so it stays finite and differentiable where the linear
`log(pdf(d, x))` would underflow. Distinct from the joint `logpdf(d, [p, y])`
(the primary-explicit score over the scored vector).

See also: [`pdf`](@ref), [`cdf`](@ref)
"
logpdf(d::Latent, x::Real) = _latent_logpdf(d, x)

@doc "

Observed-delay cumulative distribution function of the latent model, the joint
integrated over the primary, `∫ pdf(prior, p) · cdf(delay, x - p) dp`. Computed
from the latent (augmented-data) formulation by Gauss-Legendre quadrature over
the primary window, NOT the analytic `primary_censored` marginal. It agrees
with that analytic marginal to quadrature tolerance.

See also: [`logcdf`](@ref), [`ccdf`](@ref), [`pdf`](@ref)
"
function cdf(d::Latent, x::Real)
    result = _latent_integral(d, x, u -> _cdf_ad_safe(get_dist(d), u))
    return clamp(result, zero(result), one(result))
end

@doc "

Observed-delay log cumulative distribution function of the latent model, the
log of the augmented-data [`cdf`](@ref) integral.

See also: [`cdf`](@ref)
"
logcdf(d::Latent, x::Real) = log(cdf(d, x))

@doc "

Observed-delay complementary cumulative distribution function (survival) of the
latent model, `1 - cdf(d, x)` from the augmented-data integral.

See also: [`logccdf`](@ref), [`cdf`](@ref)
"
ccdf(d::Latent, x::Real) = 1 - cdf(d, x)

@doc "

Observed-delay log complementary cumulative distribution function of the latent
model, `log(ccdf(d, x))` from the augmented-data integral.

See also: [`ccdf`](@ref)
"
logccdf(d::Latent, x::Real) = log(ccdf(d, x))

@doc "

Observed-delay quantile (inverse CDF) of the latent model, found by root-finding
on the latent [`cdf`](@ref). The cdf is the augmented-data integral, so the
quantile inverts the latent formulation rather than the analytic marginal.

See also: [`cdf`](@ref)
"
function quantile(d::Latent, q::Real)
    initial_guess_fn = function (d, q)
        return [quantile(get_dist(d), q) + mean(get_primary_event(d))]
    end
    return _quantile_optimization(d, q; initial_guess_fn = initial_guess_fn)
end
