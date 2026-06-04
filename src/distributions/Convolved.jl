@doc raw"""

Distribution of a sum of independent random variables (a convolution).

`Convolved` represents ``X = X_1 + X_2 + \dots + X_n`` where the
``X_i`` are independent univariate distributions. It is the keystone
primitive for multi-delay epidemiological models, where an observed
delay is the sum of several independent stages (e.g.
onset-to-death = onset-to-admission ``\oplus`` admission-to-death).

Components may have negative support (for example a `Normal` capturing
pre-symptomatic transmission timing); `minimum` and `maximum` are the
sums of the component supports, taking the value ``\pm\infty`` where a
component is unbounded.

# CDF computation

The CDF is computed by integrating one component out against the CDF of
the others:

```math
F_X(x) = \int F_{R}(x - t)\, f_{C}(t)\, \mathrm{d}t
```

where ``C`` is the last component (the integration variable) and ``R``
is the convolution of the remaining components. For more than two
components the remaining convolution is folded recursively.

Where an analytical convolution is available (`Distributions.convolve`
applies, e.g. `Normal`+`Normal`, equal-rate `Gamma`, `Poisson`) the
two-component CDF is taken directly from the convolved distribution. All
other cases use AD-safe fixed-node Gauss-Legendre quadrature mirroring
`primarycensored_cdf`: the integral is mapped onto the fixed reference
domain ``(-1, 1)`` with the real bounds carried as `params` and the
change of variable applied inside the integrand, keeping the
Integrals.jl reverse rule on the AD path.

# Component-wise inner truncation

Each component carries a pair of bounds ``(a_i, b_i)`` (default
``(-\infty, +\infty)``). With finite bounds the distribution represents
the joint event

```math
P\!\left(\sum_i X_i \le x \;\wedge\; a_i \le X_i \le b_i \;\forall i\right),
```

i.e. each component's contribution to the convolution integral is
restricted to ``[a_i, b_i]``. This caps a delay *inside* the convolution,
which `truncated(Convolved(...))` cannot express because the cap must
live inside the integral. The bounds clip the Gauss-Legendre integration
limits, so they travel on the same `params` tangent as the integration
window and stay AD-safe. The analytic fast path is used only when every
bound is ``\pm\infty``; any finite bound forces the numeric path.

# See also
- [`generic_convolve`](@ref): Constructor function
"""
struct Convolved{C <: Tuple, B <: Tuple} <: UnivariateDistribution{Continuous}
    "Tuple of independent component distributions to be summed."
    components::C
    "Per-component `(lower, upper)` inner-convolution truncation bounds."
    bounds::B

    function Convolved(components::C, bounds::B) where {C <: Tuple, B <: Tuple}
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
        new{C, B}(components, bounds)
    end
end

# Default: unbounded components.
function Convolved(components::Tuple)
    bounds = map(_ -> (-Inf, Inf), components)
    return Convolved(components, bounds)
end

# Whether any component bound is finite (forces the numeric path).
function _has_finite_bounds(d::Convolved)
    return any(b -> isfinite(b[1]) || isfinite(b[2]), d.bounds)
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

# Examples
```@example
using CensoredDistributions, Distributions

# Sum of two delays
d = generic_convolve(Gamma(2.0, 1.0), LogNormal(1.5, 0.5))
cdf_at_5 = cdf(d, 5.0)

# Sum of three delays from a vector
d3 = generic_convolve([Gamma(2.0, 1.0), Gamma(1.0, 1.0), Normal(0.0, 1.0)])
mean_sample = rand(d3)

# Cap the second component inside the convolution
dt = generic_convolve(LogNormal(1.5, 0.5), Gamma(2.0, 1.0);
    bounds = [(-Inf, Inf), (0.0, 3.0)])
joint_prob = cdf(dt, 5.0)
```

# See also
- [`Convolved`](@ref): The distribution type
"
function generic_convolve(
        components::AbstractVector{<:UnivariateDistribution};
        bounds = nothing)
    length(components) >= 2 ||
        throw(ArgumentError("generic_convolve needs at least two components"))
    return _convolved_with_bounds(Tuple(components), bounds)
end

function generic_convolve(
        c1::UnivariateDistribution, c2::UnivariateDistribution,
        rest::UnivariateDistribution...; bounds = nothing)
    return _convolved_with_bounds((c1, c2, rest...), bounds)
end

function generic_convolve(components::Tuple; bounds = nothing)
    length(components) >= 2 ||
        throw(ArgumentError("generic_convolve needs at least two components"))
    return _convolved_with_bounds(components, bounds)
end

# Build a `Convolved` with optional `bounds`; `nothing` means unbounded.
_convolved_with_bounds(components::Tuple, ::Nothing) = Convolved(components)

function _convolved_with_bounds(components::Tuple, bounds)
    return Convolved(components, _normalise_bounds(Tuple(bounds)))
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

# ---------------------------------------------------------------------------
# Numeric convolution (AD-safe fixed-domain Gauss-Legendre)
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
_rest_total_mass(r, ::Type{T}) where {T} = T(_convolution_cdf(r, _rest_max(r)))

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

# Shared fixed-domain quadrature scaffold for the convolution
#   ∫ kernel(rest, x - t) · f_C(t) dt   over t ∈ support(C),
# where C is the last component and `rest` the convolution of the others.
# The integral is mapped onto the fixed reference domain (-1, 1) with the
# real bounds carried as `params` and the change of variable t = m + h·s
# applied inside the integrand, so the Integrals.jl reverse rule stays on
# the parameter-tangent AD path. `kernel` is `_convolution_cdf` (CDF) or
# `_convolution_pdf` (PDF). Returns the bare integral; callers add any
# saturated constant and clamp.
function _convolved_quadrature(
        last_comp, rest, kernel::F, x::Real, lower, upper) where {F}
    m = (lower + upper) / 2
    h = (upper - lower) / 2

    function integrand(s, p)
        m_, h_ = p
        t = m_ + h_ * s
        return h_ * kernel(rest, x - t) * pdf(last_comp, t)
    end

    prob = IntegralProblem(integrand, (-one(m), one(m)), (m, h))
    return solve(prob, GaussLegendre(; n = 64))[1]
end

# Vector-valued companion: one solve for a batch of points. The integrand
# returns a vector (one entry per point), so a single Gauss-Legendre call
# serves the whole batch over the shared window [lower, upper].
function _convolved_quadrature_batched(
        last_comp, rest, kernel::F,
        x::AbstractVector{<:Real}, lower, upper) where {F}
    m = (lower + upper) / 2
    h = (upper - lower) / 2

    function integrand(s, p)
        m_, h_ = p
        t = m_ + h_ * s
        ft = pdf(last_comp, t)
        return [h_ * kernel(rest, xi - t) * ft for xi in x]
    end

    prob = IntegralProblem(integrand, (-one(m), one(m)), (m, h))
    return solve(prob, GaussLegendre(; n = 64)).u
end

# Mass of the bounded integration component below `cut`:
#   P(a_C ≤ C ≤ min(cut, b_C)).
# `cut` is the upper edge of the region where F_R(x - t) = 1, and the
# integration component contributes its truncated mass there.
function _saturated_mass(c::UnivariateDistribution, b, cut, ::Type{T}) where {T}
    a = b[1]
    top = _min2(cut, b[2])
    top <= a && return zero(T)
    lo = isfinite(a) ? T(_cdf_ad_safe(c, a)) : zero(T)
    return T(_cdf_ad_safe(c, top)) - lo
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
# bounded CDF of the rest (each remaining component restricted to its own
# bound), and M_R is the rest's total bounded mass (< 1 when bounded). The
# early `x >= maximum(d)` shortcut cannot return 1 when any bound is
# finite, since the joint truncation mass is then below 1.
function _convolved_numeric_cdf(d::Convolved, x::Real)
    T = float(typeof(x))
    isnan(x) && return convert(T, NaN)
    x <= minimum(d) && return zero(T)
    if x >= maximum(d) && !_has_finite_bounds(d)
        return one(T)
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
                _rest_total_mass(rest, T) *
                _saturated_mass(last_comp, cbound, cut, T) : zero(T)

    upper <= lower && return clamp(saturated, zero(T), one(T))

    result = saturated +
             _convolved_quadrature(
        last_comp, rest, _convolution_cdf, x, lower, upper)
    return clamp(result, zero(T), one(T))
end

# Numeric convolution PDF.
#
# f_X(x) = ∫ f_R(x - t) f_C(t) dt   over t ∈ support(C).
# f_R(x - t) is 0 outside R's (bounded) support, so there is no saturated
# constant: the window is the bounded component support intersected with
# the range where x - t lands in R's bounded support.
function _convolved_numeric_pdf(d::Convolved, x::Real)
    T = float(typeof(x))
    isnan(x) && return convert(T, NaN)
    (x <= minimum(d) || x >= maximum(d)) && return zero(T)

    last_comp = d.components[end]
    cbound = d.bounds[end]
    rest = _rest_distribution(d.components[1:(end - 1)], d.bounds[1:(end - 1)])

    lower = _max2(_comp_min(last_comp, cbound), x - _rest_max(rest))
    upper = _min2(_comp_max(last_comp, cbound), x - _rest_min(rest))

    upper <= lower && return zero(T)

    result = _convolved_quadrature(
        last_comp, rest, _convolution_pdf, x, lower, upper)
    return max(result, zero(T))
end

# ---------------------------------------------------------------------------
# CDF / logcdf / pdf / logpdf
# ---------------------------------------------------------------------------

# Analytic fast path only when every component bound is ±Inf; any finite
# bound caps the inner integral and forces the numeric quadrature path.
function _analytic_or_nothing(d::Convolved)
    _has_finite_bounds(d) && return nothing
    return _analytic_convolution(d.components)
end

@doc "

Compute the cumulative distribution function.

Uses an analytical convolution when `Distributions.convolve` applies to
all component pairs and every component bound is `±Inf`, otherwise AD-safe
numeric quadrature.

See also: [`logcdf`](@ref)
"
function cdf(d::Convolved, x::Real)
    analytic = _analytic_or_nothing(d)
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
    analytic = _analytic_or_nothing(d)
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
    analytic = _analytic_or_nothing(d)
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
    analytic = _analytic_or_nothing(d)
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
    analytic = _analytic_or_nothing(d)
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
    # re-deriving it, keeping the shared single-solve fast path for the
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
    analytic = _analytic_or_nothing(d)
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
    analytic = _analytic_or_nothing(d)
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
