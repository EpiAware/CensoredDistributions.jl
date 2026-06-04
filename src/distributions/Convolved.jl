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
- [`convolved`](@ref): Constructor function
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
d = convolved(Gamma(2.0, 1.0), LogNormal(1.5, 0.5))
cdf_at_5 = cdf(d, 5.0)

# Sum of three delays from a vector
d3 = convolved([Gamma(2.0, 1.0), Gamma(1.0, 1.0), Normal(0.0, 1.0)])
mean_sample = rand(d3)
```

# See also
- [`Convolved`](@ref): The distribution type
"
function convolved(components::AbstractVector{<:UnivariateDistribution})
    length(components) >= 2 ||
        throw(ArgumentError("convolved needs at least two components"))
    return Convolved(Tuple(components))
end

function convolved(c1::UnivariateDistribution, c2::UnivariateDistribution,
        rest::UnivariateDistribution...)
    return Convolved((c1, c2, rest...))
end

function convolved(components::Tuple)
    length(components) >= 2 ||
        throw(ArgumentError("convolved needs at least two components"))
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
# Numeric convolution CDF (AD-safe fixed-domain Gauss-Legendre)
# ---------------------------------------------------------------------------

# CDF of the convolution of the first `length - 1` components evaluated
# at `x - t`, where the last component is integrated out. The "rest"
# distribution is itself a `Convolved` (or single distribution) so the
# computation folds recursively for more than two components.
_rest_distribution(c::Tuple{<:UnivariateDistribution}) = c[1]
_rest_distribution(c::Tuple) = Convolved(c)

# Scalar min/max helpers - keep the bound arithmetic below type stable
# when one side is ±Inf.
_min2(a, b) = a < b ? a : b
_max2(a, b) = a > b ? a : b

# Numeric CDF for a single (degenerate) component: just the component
# CDF. Used as the recursion base for `_convolution_cdf`.
function _convolution_cdf(d::UnivariateDistribution, x::Real)
    return _cdf_ad_safe(d, x)
end

function _convolution_cdf(d::Convolved, x::Real)
    return _convolved_numeric_cdf(d, x)
end

# Numeric convolution CDF over the fixed reference domain (-1, 1).
#
# Bounds are passed through `params` to the integrand so the
# Integrals.jl reverse rule stays on the parameter-tangent AD path; the
# change of variable t = m + h·s (s ∈ (-1, 1)) is applied inside the
# integrand. The integrand is
#   F_rest(x - t) · f_last(t)
# with f_last the pdf of the integration component and F_rest the CDF of
# the remaining convolution.
function _convolved_numeric_cdf(d::Convolved, x::Real)
    isnan(x) && return convert(float(typeof(x)), NaN)
    x <= minimum(d) && return zero(float(typeof(x)))
    x >= maximum(d) && return one(float(typeof(x)))

    last_comp = d.components[end]
    rest = _rest_distribution(d.components[1:(end - 1)])

    lower = _max2(minimum(last_comp), x - maximum(rest))
    upper = _min2(maximum(last_comp), x - minimum(rest))

    upper <= lower && return zero(float(typeof(x)))

    m = (lower + upper) / 2
    h = (upper - lower) / 2

    function integrand(s, p)
        m_, h_ = p
        t = m_ + h_ * s
        return h_ * _convolution_cdf(rest, x - t) * pdf(last_comp, t)
    end

    prob = IntegralProblem(integrand, (-one(m), one(m)), (m, h))
    result = solve(prob, GaussLegendre(; n = 64))[1]
    return clamp(result, zero(result), one(result))
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

Uses the analytical convolved density where available, otherwise
numerical differentiation of the CDF.

See also: [`logpdf`](@ref)
"
function pdf(d::Convolved, x::Real)
    analytic = _analytic_convolution(d.components)
    if analytic !== nothing
        return pdf(analytic, x)
    end
    return exp(logpdf(d, x))
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

    # Central difference on the numeric CDF, matching the PrimaryCensored
    # logpdf strategy. Bounds-clamped so we never evaluate outside support.
    h = 1e-6
    lo = max(x - h / 2, minimum(d))
    hi = min(x + h / 2, maximum(d))
    cdf_hi = _convolved_numeric_cdf(d, hi)
    cdf_lo = _convolved_numeric_cdf(d, lo)
    diff = cdf_hi - cdf_lo
    diff <= 0 && return oftype(float(x), -Inf)
    return log(diff) - log(hi - lo)
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

# Batched numeric CDF: integrate over the union bounds spanning all
# evaluation points in one solve. The integrand returns a vector, one
# entry per point, so a single Gauss-Legendre call serves the whole
# batch. Points outside support are handled by clamping after the solve.
function _convolved_numeric_cdf_batched(d::Convolved, x::AbstractVector{<:Real})
    T = promote_type(eltype(x), float(eltype(d)))
    last_comp = d.components[end]
    rest = _rest_distribution(d.components[1:(end - 1)])

    dmin = minimum(d)
    dmax = maximum(d)

    # Shared integration window covering the integration component
    # support intersected with the reachable range across all points.
    lower = max(minimum(last_comp), minimum(x) - maximum(rest))
    upper = min(maximum(last_comp), maximum(x) - minimum(rest))

    if !(upper > lower) || !isfinite(lower) || !isfinite(upper)
        # Degenerate shared window: fall back to per-point scalar solves.
        return map(xi -> _convolved_numeric_cdf(d, T(xi)), x)
    end

    m = (lower + upper) / 2
    h = (upper - lower) / 2

    function integrand(s, p)
        m_, h_ = p
        t = m_ + h_ * s
        ft = pdf(last_comp, t)
        return [h_ * _convolution_cdf(rest, xi - t) * ft for xi in x]
    end

    prob = IntegralProblem(integrand, (-one(m), one(m)), (m, h))
    raw = solve(prob, GaussLegendre(; n = 64)).u

    return map(zip(x, raw)) do (xi, ri)
        if xi <= dmin
            zero(T)
        elseif xi >= dmax
            one(T)
        else
            clamp(T(ri), zero(T), one(T))
        end
    end
end

@doc "

Compute log densities for a vector of points. Falls back to the scalar
`logpdf` per point, reusing the batched CDF solve for the numeric path.

See also: [`logpdf`](@ref)
"
function logpdf(d::Convolved, x::AbstractVector{<:Real})
    analytic = _analytic_convolution(d.components)
    if analytic !== nothing
        return map(xi -> logpdf(analytic, xi), x)
    end

    T = promote_type(eltype(x), float(eltype(d)))
    h = 1e-6
    dmin = minimum(d)
    dmax = maximum(d)

    los = [max(xi - h / 2, dmin) for xi in x]
    his = [min(xi + h / 2, dmax) for xi in x]
    pts = vcat(los, his)
    cdfs = _convolved_numeric_cdf_batched(d, pts)
    n = length(x)
    cdf_lo = cdfs[1:n]
    cdf_hi = cdfs[(n + 1):end]

    return map(1:n) do i
        if !insupport(d, x[i])
            T(-Inf)
        else
            diff = cdf_hi[i] - cdf_lo[i]
            diff <= 0 ? T(-Inf) : T(log(diff) - log(his[i] - los[i]))
        end
    end
end
