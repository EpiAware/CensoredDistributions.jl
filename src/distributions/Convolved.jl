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

# See also
- [`generic_convolve`](@ref): Constructor function
"""
struct Convolved{C <: Tuple} <: UnivariateDistribution{Continuous}
    "Tuple of independent component distributions to be summed."
    components::C

    function Convolved(components::C) where {C <: Tuple}
        length(components) >= 1 ||
            throw(ArgumentError("Convolved needs at least one component"))
        all(c -> c isa UnivariateDistribution, components) ||
            throw(ArgumentError(
                "All components must be UnivariateDistributions"))
        new{C}(components)
    end
end

@doc "

Create the distribution of a sum of independent delays (a convolution).

Accepts either two or more positional component distributions, or a
single vector/tuple of components. Returns a [`Convolved`](@ref)
distribution.

# Arguments
- `components`: Two or more `UnivariateDistribution`s, or a vector/tuple
  of them.

# Examples
```@example
using CensoredDistributions, Distributions

# Sum of two delays
d = generic_convolve(Gamma(2.0, 1.0), LogNormal(1.5, 0.5))
cdf_at_5 = cdf(d, 5.0)

# Sum of three delays from a vector
d3 = generic_convolve([Gamma(2.0, 1.0), Gamma(1.0, 1.0), Normal(0.0, 1.0)])
mean_sample = rand(d3)
```

# See also
- [`Convolved`](@ref): The distribution type
"
function generic_convolve(components::AbstractVector{<:UnivariateDistribution})
    length(components) >= 2 ||
        throw(ArgumentError("generic_convolve needs at least two components"))
    return Convolved(Tuple(components))
end

function generic_convolve(
        c1::UnivariateDistribution, c2::UnivariateDistribution,
        rest::UnivariateDistribution...)
    return Convolved((c1, c2, rest...))
end

function generic_convolve(components::Tuple)
    length(components) >= 2 ||
        throw(ArgumentError("generic_convolve needs at least two components"))
    return Convolved(components)
end

# ---------------------------------------------------------------------------
# Interface: params / support / sampling
# ---------------------------------------------------------------------------

params(d::Convolved) = map(params, d.components)

function Base.eltype(::Type{<:Convolved{C}}) where {C}
    return mapreduce(eltype, promote_type, fieldtypes(C))
end

minimum(d::Convolved) = sum(minimum, d.components)
maximum(d::Convolved) = sum(maximum, d.components)

function insupport(d::Convolved, x::Real)
    return minimum(d) <= x <= maximum(d)
end

function Base.rand(rng::AbstractRNG, d::Convolved)
    return sum(c -> rand(rng, c), d.components)
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

# Integrate the last component out against a function `kernel` of the
# remaining convolution `rest`. The "rest" distribution is itself a
# `Convolved` (or a single distribution) so both the CDF and PDF kernels
# fold recursively for more than two components.
_rest_distribution(c::Tuple{<:UnivariateDistribution}) = c[1]
_rest_distribution(c::Tuple) = Convolved(c)

# Scalar min/max helpers - keep the bound arithmetic below type stable
# when one side is ±Inf.
_min2(a, b) = a < b ? a : b
_max2(a, b) = a > b ? a : b

# Recursion bases / steps for the two kernels. For a single (degenerate)
# component the kernel is just that component's CDF/PDF; for a nested
# `Convolved` it recurses through the numeric routines.
_convolution_cdf(d::UnivariateDistribution, x::Real) = _cdf_ad_safe(d, x)
_convolution_cdf(d::Convolved, x::Real) = _convolved_numeric_cdf(d, x)

_convolution_pdf(d::UnivariateDistribution, x::Real) = pdf(d, x)
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

# Numeric convolution CDF.
#
# X = C + R with C the integration component (`last_comp`) and R the
# convolution of the remaining components (`rest`):
#   F_X(x) = ∫ F_R(x - t) f_C(t) dt   over t ∈ support(C).
# F_R(x - t) is 1 for t ≤ x - max(R) and 0 for t ≥ x - min(R), so the
# integral splits into a saturated constant term plus a transition
# integral over [lower, upper]:
#   F_X(x) = F_C(x - max(R)) + ∫_{lower}^{upper} F_R(x - t) f_C(t) dt
# Dropping the F_C term loses the mass of C in the region where R is
# already certain to be below x — wrong for bounded components (Uniform).
function _convolved_numeric_cdf(d::Convolved, x::Real)
    isnan(x) && return convert(float(typeof(x)), NaN)
    x <= minimum(d) && return zero(float(typeof(x)))
    x >= maximum(d) && return one(float(typeof(x)))

    last_comp = d.components[end]
    rest = _rest_distribution(d.components[1:(end - 1)])

    cmin = minimum(last_comp)
    cut = x - maximum(rest)              # t below this: F_R(x - t) = 1
    lower = _max2(cmin, cut)
    upper = _min2(maximum(last_comp), x - minimum(rest))

    # Mass of C below the saturated cut (where F_R = 1). Guard the
    # support boundary, where cdf at minimum is 0 by construction.
    saturated = cut > cmin ? _cdf_ad_safe(last_comp, cut) :
                zero(float(typeof(x)))

    upper <= lower && return clamp(saturated, zero(saturated), one(saturated))

    result = saturated +
             _convolved_quadrature(
        last_comp, rest, _convolution_cdf, x, lower, upper)
    return clamp(result, zero(result), one(result))
end

# Numeric convolution PDF.
#
# f_X(x) = ∫ f_R(x - t) f_C(t) dt   over t ∈ support(C).
# f_R(x - t) is 0 outside R's support, so there is no saturated constant:
# the natural window is the component support intersected with the range
# where x - t lands in R's support.
function _convolved_numeric_pdf(d::Convolved, x::Real)
    isnan(x) && return convert(float(typeof(x)), NaN)
    (x <= minimum(d) || x >= maximum(d)) && return zero(float(typeof(x)))

    last_comp = d.components[end]
    rest = _rest_distribution(d.components[1:(end - 1)])

    lower = _max2(minimum(last_comp), x - maximum(rest))
    upper = _min2(maximum(last_comp), x - minimum(rest))

    upper <= lower && return zero(float(typeof(x)))

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
    analytic = _analytic_convolution(d.components)
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
    analytic = _analytic_convolution(d.components)
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
    analytic = _analytic_convolution(d.components)
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
    analytic = _analytic_convolution(d.components)
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
    analytic = _analytic_convolution(d.components)
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
    last_comp = d.components[end]
    rest = _rest_distribution(d.components[1:(end - 1)])

    cmin = minimum(last_comp)
    dmin = minimum(d)
    dmax = maximum(d)

    # Shared integration window: component support intersected with the
    # reachable range across all points.
    lower = max(cmin, minimum(x) - maximum(rest))
    upper = min(maximum(last_comp), maximum(x) - minimum(rest))

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
    last_comp = d.components[end]
    rest = _rest_distribution(d.components[1:(end - 1)])

    dmin = minimum(d)
    dmax = maximum(d)

    # Shared integration window covering every point's transition region.
    lower = max(minimum(last_comp), minimum(x) - maximum(rest))
    upper = min(maximum(last_comp), maximum(x) - minimum(rest))

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
    analytic = _analytic_convolution(d.components)
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
    analytic = _analytic_convolution(d.components)
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
