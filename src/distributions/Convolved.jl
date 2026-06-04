@doc """

Distribution of a sum of independent random variables (a convolution).

`Convolved` represents ``X = X_1 + X_2 + \\dots + X_n`` where the
``X_i`` are independent univariate distributions. It is the keystone
primitive for multi-delay epidemiological models, where an observed
delay is the sum of several independent stages (e.g.
onset-to-death = onset-to-admission ``\\oplus`` admission-to-death).

Components may have negative support (for example a `Normal` capturing
pre-symptomatic transmission timing); `minimum` and `maximum` are the
sums of the component supports, taking the value ``\\pm\\infty`` where a
component is unbounded.

# CDF computation

The CDF is computed by integrating one component out against the CDF of
the others:

```math
F_X(x) = \\int F_{R}(x - t)\\, f_{C}(t)\\, \\mathrm{d}t
```

where ``C`` is the last component (the integration variable) and ``R``
is the convolution of the remaining components. For more than two
components the remaining convolution is folded recursively.

Where an analytical convolution is available (`Distributions.convolve`
applies, e.g. `Normal`+`Normal`, equal-scale `Gamma`, equal-rate
`Exponential`) the two-component result is taken directly from the
convolved distribution unless `force_numeric` is set. All other cases use
AD-safe fixed-node Gauss-Legendre quadrature: the integral is mapped from
the fixed reference domain ``(-1, 1)`` onto the real bounds inside the
integrand and reduced as a bare weighted dot product (`gl_integrate`),
which lets every AD backend specialise on the integrand's own type so
component `Dual`s and tangents propagate.

The `force_numeric` field forces the numeric quadrature path even when an
analytic convolution exists, mirroring `primary_censored`; it is useful
for validation and debugging.

# Component-wise inner truncation

Each component carries a pair of bounds ``(a_i, b_i)`` (default
``(-\\infty, +\\infty)``). With finite bounds the distribution represents
the joint event

```math
P\\Big(\\sum_i X_i \\le x \\ \\wedge\\ a_i \\le X_i \\le b_i\\ \\forall i\\Big),
```

i.e. each component's contribution to the convolution integral is
restricted to ``[a_i, b_i]``. This caps a delay *inside* the convolution,
which `truncated(Convolved(...))` cannot express because the cap must
live inside the integral. The bounds clip the Gauss-Legendre integration
limits passed to `gl_integrate`, so they ride the same integrand the AD
backends already specialise on and stay AD-safe. The analytic fast path
is used only when every bound is ``\\pm\\infty``; any finite bound forces
the numeric path.

With finite bounds `cdf` returns the **unnormalised joint mass**
``P(\\sum_i X_i \\le x \\wedge a_i \\le X_i \\le b_i\\ \\forall i)``, not a
conditional (normalised) CDF: it saturates at the total truncation mass
``P(a_i \\le X_i \\le b_i\\ \\forall i) < 1`` rather than 1, so
`cdf(d, maximum(d)) < 1`. Likewise `pdf` is the corresponding
unnormalised joint density. This is intended for use as a per-record
likelihood term; divide by the saturated mass if a normalised
conditional distribution is wanted. A bound whose intersection with its
component's support is empty is rejected at construction.

# See also
- [`convolve_distributions`](@ref): Constructor function
"""
struct Convolved{C <: Tuple, B <: Tuple} <:
       UnivariateDistribution{Continuous}
    "Tuple of independent component distributions to be summed."
    components::C
    "Per-component `(lower, upper)` inner-convolution truncation bounds."
    bounds::B
    "Force numeric quadrature even when an analytic convolution exists."
    force_numeric::Bool

    function Convolved(components::C, bounds::B;
            force_numeric::Bool = false) where {C <: Tuple, B <: Tuple}
        length(components) >= 1 ||
            throw(ArgumentError("Convolved needs at least one component"))
        all(c -> c isa UnivariateDistribution, components) ||
            throw(ArgumentError(
                "All components must be UnivariateDistributions"))
        length(bounds) == length(components) ||
            throw(ArgumentError(
                "bounds must have one (lower, upper) pair per component"))
        all(b -> length(b) == 2 && b[1] <= b[2], bounds) ||
            throw(ArgumentError(
                "each bound must be a (lower, upper) pair with lower <= upper"))
        # Reject bounds whose intersection with a component's support is
        # empty (e.g. an upper bound below the component minimum). Such a
        # degenerate window has no mass and would otherwise feed an
        # inverted [lower, upper] into the quadrature, producing silent
        # `NaN` for nested components. Erroring at construction keeps the
        # failure loud and local.
        all(((c, b),) -> _bound_overlaps_support(c, b),
            zip(components, bounds)) || throw(ArgumentError(
            "a bound has empty intersection with its component's support"))
        new{C, B}(components, bounds, force_numeric)
    end
end

# Default: unbounded components.
function Convolved(components::Tuple; force_numeric::Bool = false)
    bounds = map(_ -> (-Inf, Inf), components)
    return Convolved(components, bounds; force_numeric = force_numeric)
end

# Whether any component bound is finite (forces the numeric path).
function _has_finite_bounds(d::Convolved)
    return any(b -> isfinite(b[1]) || isfinite(b[2]), d.bounds)
end

# Whether bound `b = (a, b₂)` overlaps the component's support, i.e. the
# effective support `[max(min(c), a), min(max(c), b₂)]` is non-empty.
function _bound_overlaps_support(c::UnivariateDistribution, b)
    return _max2(minimum(c), b[1]) <= _min2(maximum(c), b[2])
end

@doc "

Create the distribution of a sum of independent delays (a convolution).

Accepts either two or more positional component distributions, or a
single vector/tuple of components. Returns a [`Convolved`](@ref)
distribution.

# Arguments
- `components`: Two or more `UnivariateDistribution`s, or a vector/tuple
  of them.

# Keyword Arguments
- `bounds`: Per-component `(lower, upper)` truncation bounds for the
  inner convolution integral, as a vector/tuple of pairs (one per
  component). Defaults to `(-Inf, Inf)` for every component. A finite
  bound restricts that component's contribution to the convolution,
  giving `P(sum ≤ x ∧ aᵢ ≤ componentᵢ ≤ bᵢ)`, and forces the numeric
  path.
- `force_numeric`: Force numeric quadrature even when an analytic
  convolution is available (default: `false`), mirroring
  `primary_censored`.

# Examples
```@example
using CensoredDistributions, Distributions

# Sum of two delays
d = convolve_distributions(Gamma(2.0, 1.0), LogNormal(1.5, 0.5))
cdf_at_5 = cdf(d, 5.0)

# Sum of three delays from a vector
d3 = convolve_distributions([Gamma(2.0, 1.0), Gamma(1.0, 1.0), Normal(0.0, 1.0)])
mean_sample = rand(d3)

# Force numeric quadrature even for an analytic pair
dn = convolve_distributions(Normal(0.0, 1.0), Normal(1.0, 2.0); force_numeric=true)
cdf_numeric = cdf(dn, 2.0)

# Cap the second component inside the convolution
dt = convolve_distributions(LogNormal(1.5, 0.5), Gamma(2.0, 1.0);
    bounds = [(-Inf, Inf), (0.0, 3.0)])
joint_prob = cdf(dt, 5.0)
```

# See also
- [`Convolved`](@ref): The distribution type
"
function convolve_distributions(
        components::AbstractVector{<:UnivariateDistribution};
        bounds = nothing, force_numeric::Bool = false)
    length(components) >= 2 ||
        throw(ArgumentError("convolve_distributions needs at least two components"))
    return _convolved_with_bounds(
        Tuple(components), bounds, force_numeric)
end

function convolve_distributions(
        c1::UnivariateDistribution, c2::UnivariateDistribution,
        rest::UnivariateDistribution...;
        bounds = nothing, force_numeric::Bool = false)
    return _convolved_with_bounds((c1, c2, rest...), bounds, force_numeric)
end

function convolve_distributions(components::Tuple;
        bounds = nothing, force_numeric::Bool = false)
    length(components) >= 2 ||
        throw(ArgumentError("convolve_distributions needs at least two components"))
    return _convolved_with_bounds(components, bounds, force_numeric)
end

# Build a `Convolved` with optional `bounds`; `nothing` means unbounded.
function _convolved_with_bounds(components::Tuple, ::Nothing, force_numeric)
    return Convolved(components; force_numeric = force_numeric)
end

function _convolved_with_bounds(components::Tuple, bounds, force_numeric)
    return Convolved(components, _normalise_bounds(Tuple(bounds));
        force_numeric = force_numeric)
end

# Coerce each bound to a homogeneous `(lower, upper)` Float pair so the
# tuple is type stable and the limits promote cleanly under AD.
function _normalise_bounds(bounds::Tuple)
    return map(b -> (float(b[1]), float(b[2])), bounds)
end

# ---------------------------------------------------------------------------
# Interface: params / support / sampling
# ---------------------------------------------------------------------------

params(d::Convolved) = map(params, d.components)

function Base.eltype(::Type{<:Convolved{C}}) where {C <: Tuple}
    return mapreduce(eltype, promote_type, fieldtypes(C))
end

# Effective support of a component after intersecting with its bounds.
_comp_min(c::UnivariateDistribution, b) = _max2(minimum(c), b[1])
_comp_max(c::UnivariateDistribution, b) = _min2(maximum(c), b[2])

function minimum(d::Convolved)
    return sum(((c, b),) -> _comp_min(c, b), zip(d.components, d.bounds))
end
function maximum(d::Convolved)
    return sum(((c, b),) -> _comp_max(c, b), zip(d.components, d.bounds))
end

function insupport(d::Convolved, x::Real)
    return minimum(d) <= x <= maximum(d)
end

# Sample each component (truncated to its bounds when finite) and sum.
# With default bounds this is the ordinary convolution sample.
function Base.rand(rng::AbstractRNG, d::Convolved)
    return sum(zip(d.components, d.bounds)) do (c, b)
        rand(rng, _maybe_truncate(c, b))
    end
end

# Truncate a component to its bounds only when at least one side is
# finite; unbounded components are returned unchanged.
function _maybe_truncate(c::UnivariateDistribution, b)
    (isfinite(b[1]) || isfinite(b[2])) || return c
    return truncated(c, b[1], b[2])
end

sampler(d::Convolved) = d

@doc "

Compute the quantile (inverse CDF) of the convolution.

No closed form exists for a generic convolution, so the quantile is found
by numerically inverting [`cdf`](@ref). The initial guess is the sum of the
component quantiles, which is exact when the components are degenerate and a
good starting point otherwise. Providing this method lets a `Convolved`
compose under `truncated`, where `Distributions` derives the truncated
quantile and inverse-CDF sampler from the base `quantile`.

See also: [`cdf`](@ref)
"
function quantile(d::Convolved, p::Real)
    return _quantile_optimization(
        d, p; initial_guess_fn = _convolved_quantile_guess)
end

# Sum of component quantiles as the inversion starting point.
function _convolved_quantile_guess(d::Convolved, p::Real)
    guess = sum(c -> float(quantile(c, p)), d.components)
    return [guess]
end

# ---------------------------------------------------------------------------
# Analytical fast path via Distributions.convolve
# ---------------------------------------------------------------------------

# `_try_convolve` returns the analytically convolved distribution when a
# closed form exists for the pair, otherwise `nothing`. Dispatch (rather
# than `try`/`catch`) selects the analytic pairs so the path stays
# differentiable under every AD backend — Mooncake reverse cannot
# differentiate through `try`/`catch`. Only continuous families whose
# `Distributions.convolve` always succeeds for the given parameters are
# enabled; `Gamma` additionally needs equal scale (else the runtime
# `convolve` would throw), so a parameter check guards it.
_try_convolve(a::UnivariateDistribution, b::UnivariateDistribution) = nothing

function _try_convolve(a::Normal, b::Normal)
    return Distributions.convolve(a, b)
end

function _try_convolve(a::Exponential, b::Exponential)
    # convolve(::Exponential, ::Exponential) asserts equal rate and throws
    # otherwise, so guard the parameters and fall back to numeric.
    return scale(a) ≈ scale(b) ? Distributions.convolve(a, b) : nothing
end

function _try_convolve(a::Gamma, b::Gamma)
    return scale(a) ≈ scale(b) ? Distributions.convolve(a, b) : nothing
end

# Reduce the component tuple to a single distribution, folding pairwise
# with `Distributions.convolve` where it applies. Returns the fully
# convolved distribution when every pair convolves analytically, else
# `nothing` so the caller falls back to numeric quadrature.
function _analytic_convolution(components::Tuple)
    acc = components[1]
    for i in 2:length(components)
        acc = _try_convolve(acc, components[i])
        acc === nothing && return nothing
    end
    return acc
end

# The analytic convolution to use for `d`, or `nothing` when none exists,
# when `d.force_numeric` requests the numeric quadrature path, or when any
# component bound is finite (which caps the inner integral and so requires
# the numeric path).
function _maybe_analytic(d::Convolved)
    (d.force_numeric || _has_finite_bounds(d)) && return nothing
    return _analytic_convolution(d.components)
end

# Fraction of probability trimmed from each tail of an unbounded
# integration component when clamping an infinite quadrature window.
const _CONVOLVED_TAIL = 1e-8

# Clamp an integration window to a finite range. Both the integrand's
# `f_C(t)` factor and (for the CDF) the transition of `F_R(x - t)` are
# negligible outside the integration component's effective support, so an
# infinite endpoint is replaced by an extreme quantile of `last_comp`.
# This lets the numeric path handle components unbounded on either side
# (e.g. Normal+Normal under `force_numeric`).
function _finite_window(last_comp, lower::Real, upper::Real)
    lo = isfinite(lower) ? lower : quantile(last_comp, _CONVOLVED_TAIL)
    hi = isfinite(upper) ? upper : quantile(last_comp, 1 - _CONVOLVED_TAIL)
    return lo, hi
end

# ---------------------------------------------------------------------------
# Numeric convolution (AD-safe Gauss-Legendre dot product)
# ---------------------------------------------------------------------------

# Scalar min/max helpers - keep the bound arithmetic below type stable
# when one side is ±Inf.
_min2(a, b) = a < b ? a : b
_max2(a, b) = a > b ? a : b

# The "rest" of the convolution (every component but the integration one)
# together with its per-component bounds. A single remaining component is
# wrapped in `_BoundedComponent` so its bound is applied in the kernel; two
# or more are wrapped in a nested `Convolved` carrying their bounds, so the
# recursion folds them. Splitting the components from the bounds keeps the
# integration window arithmetic (which needs the bounded support) in one
# place while reusing the shared quadrature scaffold unchanged.
struct _BoundedComponent{D <: UnivariateDistribution, B}
    dist::D
    bound::B
end

function _rest_distribution(c::Tuple{<:UnivariateDistribution}, b::Tuple)
    return _BoundedComponent(c[1], b[1])
end
_rest_distribution(c::Tuple, b::Tuple) = Convolved(c, b)

# Bounded support of the rest distribution (used for the integration
# window): a single component intersects its own bound; a nested
# `Convolved` already accounts for bounds in its `minimum`/`maximum`.
_rest_min(r::_BoundedComponent) = _comp_min(r.dist, r.bound)
_rest_max(r::_BoundedComponent) = _comp_max(r.dist, r.bound)
_rest_min(r::Convolved) = minimum(r)
_rest_max(r::Convolved) = maximum(r)

# Total bounded mass of the rest, `P(rest within its bounds)`. This is the
# value the bounded F_R saturates to once `x - t` exceeds the rest's upper
# edge — it is 1 only when the rest is unbounded. Evaluating the bounded
# CDF at the rest's upper support gives this mass for both kinds of rest.
_rest_total_mass(r) = _convolution_cdf(r, _rest_max(r))

# Recursion bases / steps for the two kernels. For a single (bounded)
# component the kernel applies the bound directly; for a nested `Convolved`
# it recurses through the numeric routines, which thread the bounds.
function _convolution_cdf(r::_BoundedComponent, y::Real)
    a = r.bound[1]
    b = r.bound[2]
    y <= a && return zero(_cdf_ad_safe(r.dist, y))
    upper = _min2(y, b)
    lo = isfinite(a) ? _cdf_ad_safe(r.dist, a) : zero(upper)
    return _cdf_ad_safe(r.dist, upper) - lo
end
_convolution_cdf(d::Convolved, x::Real) = _convolved_numeric_cdf(d, x)

function _convolution_pdf(r::_BoundedComponent, y::Real)
    (r.bound[1] <= y <= r.bound[2]) || return zero(pdf(r.dist, y))
    return pdf(r.dist, y)
end
_convolution_pdf(d::Convolved, x::Real) = _convolved_numeric_pdf(d, x)

# Number of Gauss-Legendre nodes for the convolution quadrature. The
# batched path integrates every point over one shared window, so a small
# point whose natural window is much tighter than the shared one is
# resolved by only the nodes that fall in its sub-range; a peaked
# component density (e.g. LogNormal) makes this the accuracy-limiting
# case. n = 192 brings the batched-vs-scalar gap on a typical batch to
# ~5e-4 (n = 64 left it at ~4e-3) and shrinks it ~15x on a deliberately
# wide batch. Cost is roughly linear in the node count and still small
# for these smooth, density-weighted integrands. The scalar path stays
# the accurate reference; raise this if a batch spans an extreme range.
const _CONVOLVED_NODES = 192

# Fixed Gauss-Legendre rule carrying its own reference nodes/weights on
# `[-1, 1]`, built once at load. Holding the nodes/weights directly (and
# calling the integrand inline in `_gl_reduce`) rather than going through
# an Integrals.jl `IntegralProblem`/`solve` boundary lets Julia specialise
# on the integrand's concrete return type, so `Dual`s and AD tangents
# propagate and the result type is inferred. This is the AD-safe pattern
# from epiforecasts/BVDOutbreakSize `src/integrate.jl`.
struct _GL{N, W}
    nodes::N
    weights::W
end

_GL(n::Int) = _GL(FastGaussQuadrature.gausslegendre(n)...)

const _CONVOLVED_GL = _GL(_CONVOLVED_NODES)

# Reduce an integrand `g` over the reference domain `[-1, 1]` against the
# `alg` rule. Seeding `acc` with `weights[1] * g(nodes[1])` fixes the
# accumulator's element type from the integrand itself, so a component
# `Dual` flows into the result rather than being forced to `Float64`.
@inline function _gl_reduce(g::G, alg::_GL) where {G}
    n, w = alg.nodes, alg.weights
    @inbounds acc = w[1] * g(n[1])
    @inbounds for i in 2:length(n)
        acc += w[i] * g(n[i])
    end
    return acc
end

# Integrate a scalar function `f` over `[lo, hi]` by Gauss-Legendre
# quadrature, mapping the reference domain `[-1, 1]` onto `[lo, hi]` inside
# the integrand. Returns a typed zero when `hi <= lo`.
function gl_integrate(f::F, lo, hi, alg::_GL = _CONVOLVED_GL) where {F}
    hi <= lo && return zero(f(lo))
    h = (hi - lo) / 2
    m = (lo + hi) / 2
    return h * _gl_reduce(s -> f(m + h * s), alg)
end

# Scalar convolution quadrature:
#   ∫ kernel(rest, x - t) · f_C(t) dt   over t ∈ [lower, upper],
# where C is the last component and `rest` the convolution of the others.
# `kernel` is `_convolution_cdf` (CDF) or `_convolution_pdf` (PDF).
# Returns the bare integral; callers add any saturated constant and clamp.
function _convolved_quadrature(
        last_comp, rest, kernel::F, x::Real, lower, upper) where {F}
    return gl_integrate(
        t -> kernel(rest, x - t) * pdf(last_comp, t), lower, upper)
end

# Vector-valued companion: one quadrature pass for a batch of points. Each
# node's `f_C(t)` factor is evaluated once and reused across every point,
# and the accumulator vector's element type is seeded from the first
# node's contribution so `Dual`s propagate (mirroring `_gl_reduce`).
function _convolved_quadrature_batched(
        last_comp, rest, kernel::F,
        x::AbstractVector{<:Real}, lower, upper) where {F}
    if upper <= lower
        z = zero(kernel(rest, first(x) - lower) * pdf(last_comp, lower))
        return fill(z, length(x))
    end
    h = (upper - lower) / 2
    m = (lower + upper) / 2
    n, w = _CONVOLVED_GL.nodes, _CONVOLVED_GL.weights

    @inbounds begin
        t1 = m + h * n[1]
        ft1 = pdf(last_comp, t1)
        acc = [w[1] * kernel(rest, xi - t1) * ft1 for xi in x]
        for i in 2:length(n)
            ti = m + h * n[i]
            fti = pdf(last_comp, ti)
            for j in eachindex(x)
                acc[j] += w[i] * kernel(rest, x[j] - ti) * fti
            end
        end
    end
    return h .* acc
end

# Mass of the bounded integration component below `cut`:
#   P(a_C ≤ C ≤ min(cut, b_C)).
# `cut` is the upper edge of the region where F_R(x - t) saturates, and the
# integration component contributes its truncated mass there. The result
# keeps the natural (AD-tracked) element type of `_cdf_ad_safe`; `z` is a
# typed zero only for the empty-window short-circuit.
function _saturated_mass(c::UnivariateDistribution, b, cut, z)
    a = b[1]
    top = _min2(cut, b[2])
    top <= a && return z
    lo = isfinite(a) ? _cdf_ad_safe(c, a) : z
    return _cdf_ad_safe(c, top) - lo
end

# Numeric convolution CDF.
#
# X = C + R with C the integration component (`last_comp`) and R the
# convolution of the remaining components (`rest`):
#   F_X(x) = ∫ F_R(x - t) f_C(t) dt   over t ∈ support(C).
# F_R(x - t) saturates to the rest's total bounded mass M_R for
# t ≤ x - max(R) and is 0 for t ≥ x - min(R), so the integral splits into
# a saturated constant term plus a transition integral over [lower, upper]:
#   F_X(x) = M_R · P(a_C ≤ C ≤ cut) + ∫_{lower}^{upper} F_R(x - t) f_C(t) dt
# Dropping the saturated term loses the mass of C in the region where R is
# already certain to be below x — wrong for bounded components (Uniform).
# Without bounds M_R = 1 and this reduces to the ordinary decomposition.
#
# With component bounds the integration window and the saturated mass are
# both intersected with the integration component's bound, F_R is the
# bounded CDF of the rest, and M_R is the rest's total bounded mass (< 1
# when bounded). The early `x >= maximum(d)` shortcut cannot return 1 when
# any bound is finite, since the joint truncation mass is then below 1.
function _convolved_numeric_cdf(d::Convolved, x::Real)
    z = zero(float(typeof(x)))
    isnan(x) && return convert(float(typeof(x)), NaN)
    x <= minimum(d) && return z
    if x >= maximum(d) && !_has_finite_bounds(d)
        return one(float(typeof(x)))
    end

    last_comp = d.components[end]
    cbound = d.bounds[end]
    rest = _rest_distribution(d.components[1:(end - 1)], d.bounds[1:(end - 1)])

    clo = _comp_min(last_comp, cbound)
    chi = _comp_max(last_comp, cbound)
    cut = x - _rest_max(rest)            # t below this: F_R saturates
    lower = _max2(clo, cut)
    upper = _min2(chi, x - _rest_min(rest))

    saturated = cut > clo ?
                _rest_total_mass(rest) *
                _saturated_mass(last_comp, cbound, cut, z) : z

    upper <= lower && return clamp(saturated, zero(saturated), one(saturated))

    lower, upper = _finite_window(last_comp, lower, upper)
    result = saturated +
             _convolved_quadrature(
        last_comp, rest, _convolution_cdf, x, lower, upper)
    return clamp(result, zero(result), one(result))
end

# Numeric convolution PDF.
#
# f_X(x) = ∫ f_R(x - t) f_C(t) dt   over t ∈ support(C).
# f_R(x - t) is 0 outside R's (bounded) support, so there is no saturated
# constant: the window is the bounded component support intersected with
# the range where x - t lands in R's bounded support.
function _convolved_numeric_pdf(d::Convolved, x::Real)
    z = zero(float(typeof(x)))
    isnan(x) && return convert(float(typeof(x)), NaN)
    (x <= minimum(d) || x >= maximum(d)) && return z

    last_comp = d.components[end]
    cbound = d.bounds[end]
    rest = _rest_distribution(d.components[1:(end - 1)], d.bounds[1:(end - 1)])

    lower = _max2(_comp_min(last_comp, cbound), x - _rest_max(rest))
    upper = _min2(_comp_max(last_comp, cbound), x - _rest_min(rest))

    upper <= lower && return z

    lower, upper = _finite_window(last_comp, lower, upper)
    result = _convolved_quadrature(
        last_comp, rest, _convolution_pdf, x, lower, upper)
    return max(result, zero(result))
end

# ---------------------------------------------------------------------------
# CDF / logcdf / pdf / logpdf
# ---------------------------------------------------------------------------

@doc "

Compute the cumulative distribution function.

Uses an analytical convolution when `Distributions.convolve` applies to
all component pairs, otherwise AD-safe numeric quadrature.

See also: [`logcdf`](@ref)
"
function cdf(d::Convolved, x::Real)
    analytic = _maybe_analytic(d)
    if analytic !== nothing
        return cdf(analytic, x)
    end
    return _convolved_numeric_cdf(d, x)
end

@doc "

Compute the log cumulative distribution function.

See also: [`cdf`](@ref)
"
function logcdf(d::Convolved, x::Real)
    analytic = _maybe_analytic(d)
    if analytic !== nothing
        return logcdf(analytic, x)
    end
    c = _convolved_numeric_cdf(d, x)
    return c <= 0 ? oftype(float(c), -Inf) : log(c)
end

function ccdf(d::Convolved, x::Real)
    return 1 - cdf(d, x)
end

function logccdf(d::Convolved, x::Real)
    logcdf_val = logcdf(d, x)
    if logcdf_val == -Inf
        return zero(logcdf_val)
    elseif logcdf_val >= 0
        return oftype(logcdf_val, -Inf)
    end
    return log1mexp(logcdf_val)
end

@doc "

Compute the probability density function.

Uses the exact analytical convolved density where `Distributions.convolve`
applies to all component pairs, otherwise the AD-safe numeric density
convolution ``f_X(x) = \\int f_R(x - t) f_C(t) \\, dt``.

See also: [`logpdf`](@ref)
"
function pdf(d::Convolved, x::Real)
    analytic = _maybe_analytic(d)
    if analytic !== nothing
        return pdf(analytic, x)
    end
    return _convolved_numeric_pdf(d, x)
end

@doc "

Compute the log probability density function.

See also: [`pdf`](@ref), [`logcdf`](@ref)
"
function logpdf(d::Convolved, x::Real)
    analytic = _maybe_analytic(d)
    if analytic !== nothing
        return logpdf(analytic, x)
    end
    if !insupport(d, x)
        return oftype(float(x), -Inf)
    end
    p = _convolved_numeric_pdf(d, x)
    return p <= 0 ? oftype(float(x), -Inf) : log(p)
end

# ---------------------------------------------------------------------------
# Batched cdf / logpdf: a single quadrature solve for a vector of points
# ---------------------------------------------------------------------------

@doc "

Compute the CDF for a vector of evaluation points using a single
quadrature solve (the integrand returns a vector).

See also: [`cdf`](@ref)
"
function cdf(d::Convolved, x::AbstractVector{<:Real})
    analytic = _maybe_analytic(d)
    if analytic !== nothing
        return map(xi -> cdf(analytic, xi), x)
    end
    return _convolved_numeric_cdf_batched(d, x)
end

# Batched numeric CDF: one Gauss-Legendre solve for all points via the
# shared `_convolved_quadrature_batched` scaffold.
#
# Same decomposition as the scalar path:
#   F_X(x_i) = F_C(lower) + ∫_{lower}^{upper} F_R(x_i - t) f_C(t) dt
# where `lower` is the shared window start. F_C(lower) is the mass of the
# integration component C below the window (where F_R(x_i - t) = 1 for
# every point, since x_i ≥ min(x)); the integral then picks up each
# point's transition region, with F_R returning 1 between `lower` and
# x_i - max(R). The saturated constant is shared across points.
function _convolved_numeric_cdf_batched(d::Convolved, x::AbstractVector{<:Real})
    T = promote_type(eltype(x), float(eltype(d)))

    # The bounded saturated-mass / window decomposition differs per point
    # under finite bounds; fall back to the scalar bounded path rather than
    # re-deriving it, keeping the shared single-pass fast path for the
    # common unbounded case.
    _has_finite_bounds(d) &&
        return map(xi -> _convolved_numeric_cdf(d, T(xi)), x)

    last_comp = d.components[end]
    rest = _rest_distribution(
        d.components[1:(end - 1)], d.bounds[1:(end - 1)])

    cmin = minimum(last_comp)
    dmin = minimum(d)
    dmax = maximum(d)

    # Shared integration window: component support intersected with the
    # reachable range across all points.
    lower = max(cmin, minimum(x) - _rest_max(rest))
    upper = min(maximum(last_comp), maximum(x) - _rest_min(rest))

    if !(upper > lower) || !isfinite(lower) || !isfinite(upper)
        # Degenerate shared window: fall back to per-point scalar solves.
        return map(xi -> _convolved_numeric_cdf(d, T(xi)), x)
    end

    saturated = lower > cmin ? T(_cdf_ad_safe(last_comp, lower)) : zero(T)
    raw = _convolved_quadrature_batched(
        last_comp, rest, _convolution_cdf, x, lower, upper)

    return map(zip(x, raw)) do (xi, ri)
        if xi <= dmin
            zero(T)
        elseif xi >= dmax
            one(T)
        else
            clamp(saturated + T(ri), zero(T), one(T))
        end
    end
end

# Batched numeric PDF: one Gauss-Legendre solve for all points via the
# shared scaffold. No saturated constant (f_R vanishes outside support).
function _convolved_numeric_pdf_batched(d::Convolved, x::AbstractVector{<:Real})
    T = promote_type(eltype(x), float(eltype(d)))

    # See `_convolved_numeric_cdf_batched`: bounded points need the scalar
    # bounded window, so fall back per point when any bound is finite.
    _has_finite_bounds(d) &&
        return map(xi -> _convolved_numeric_pdf(d, T(xi)), x)

    last_comp = d.components[end]
    rest = _rest_distribution(
        d.components[1:(end - 1)], d.bounds[1:(end - 1)])

    dmin = minimum(d)
    dmax = maximum(d)

    # Shared integration window covering every point's transition region.
    lower = max(minimum(last_comp), minimum(x) - _rest_max(rest))
    upper = min(maximum(last_comp), maximum(x) - _rest_min(rest))

    if !(upper > lower) || !isfinite(lower) || !isfinite(upper)
        # Degenerate shared window: fall back to per-point scalar solves.
        return map(xi -> _convolved_numeric_pdf(d, T(xi)), x)
    end

    raw = _convolved_quadrature_batched(
        last_comp, rest, _convolution_pdf, x, lower, upper)

    return map(zip(x, raw)) do (xi, ri)
        (xi <= dmin || xi >= dmax) ? zero(T) : max(T(ri), zero(T))
    end
end

@doc "

Compute densities for a vector of points using a single quadrature solve
(the integrand returns a vector).

See also: [`pdf`](@ref)
"
function pdf(d::Convolved, x::AbstractVector{<:Real})
    analytic = _maybe_analytic(d)
    if analytic !== nothing
        return map(xi -> pdf(analytic, xi), x)
    end
    return _convolved_numeric_pdf_batched(d, x)
end

@doc "

Compute log densities for a vector of points, reusing the batched PDF
solve for the numeric path.

See also: [`logpdf`](@ref), [`pdf`](@ref)
"
function logpdf(d::Convolved, x::AbstractVector{<:Real})
    analytic = _maybe_analytic(d)
    if analytic !== nothing
        return map(xi -> logpdf(analytic, xi), x)
    end

    T = promote_type(eltype(x), float(eltype(d)))
    pdfs = _convolved_numeric_pdf_batched(d, x)

    return map(zip(x, pdfs)) do (xi, p)
        if !insupport(d, xi)
            T(-Inf)
        else
            p <= 0 ? T(-Inf) : T(log(p))
        end
    end
end
