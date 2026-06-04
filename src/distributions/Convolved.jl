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
AD-safe fixed-node Gauss-Legendre quadrature mirroring
`primarycensored_cdf`: the integral is mapped onto the fixed reference
domain ``(-1, 1)`` with the real bounds carried as `params` and the
change of variable applied inside the integrand, keeping the
Integrals.jl reverse rule on the AD path.

The `force_numeric` field forces the numeric quadrature path even when an
analytic convolution exists, mirroring `primary_censored`; it is useful
for validation and debugging.

# See also
- [`generic_convolve`](@ref): Constructor function
"""
struct Convolved{C <: Tuple} <: UnivariateDistribution{Continuous}
    "Tuple of independent component distributions to be summed."
    components::C
    "Force numeric quadrature even when an analytic convolution exists."
    force_numeric::Bool

    function Convolved(components::C; force_numeric::Bool = false) where {
            C <: Tuple}
        length(components) >= 1 ||
            throw(ArgumentError("Convolved needs at least one component"))
        all(c -> c isa UnivariateDistribution, components) ||
            throw(ArgumentError(
                "All components must be UnivariateDistributions"))
        new{C}(components, force_numeric)
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

# Keyword Arguments
- `force_numeric`: Force numeric quadrature even when an analytic
  convolution is available (default: `false`), mirroring
  `primary_censored`.

# Examples
```@example
using CensoredDistributions, Distributions

# Sum of two delays
d = generic_convolve(Gamma(2.0, 1.0), LogNormal(1.5, 0.5))
cdf_at_5 = cdf(d, 5.0)

# Sum of three delays from a vector
d3 = generic_convolve([Gamma(2.0, 1.0), Gamma(1.0, 1.0), Normal(0.0, 1.0)])
mean_sample = rand(d3)

# Force numeric quadrature even for an analytic pair
dn = generic_convolve(Normal(0.0, 1.0), Normal(1.0, 2.0); force_numeric=true)
cdf_numeric = cdf(dn, 2.0)
```

# See also
- [`Convolved`](@ref): The distribution type
"
function generic_convolve(
        components::AbstractVector{<:UnivariateDistribution};
        force_numeric::Bool = false)
    length(components) >= 2 ||
        throw(ArgumentError("generic_convolve needs at least two components"))
    return Convolved(Tuple(components); force_numeric = force_numeric)
end

function generic_convolve(
        c1::UnivariateDistribution, c2::UnivariateDistribution,
        rest::UnivariateDistribution...; force_numeric::Bool = false)
    return Convolved((c1, c2, rest...); force_numeric = force_numeric)
end

function generic_convolve(components::Tuple; force_numeric::Bool = false)
    length(components) >= 2 ||
        throw(ArgumentError("generic_convolve needs at least two components"))
    return Convolved(components; force_numeric = force_numeric)
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

# The analytic convolution to use for `d`, or `nothing` when none exists
# or when `d.force_numeric` requests the numeric quadrature path.
function _maybe_analytic(d::Convolved)
    d.force_numeric && return nothing
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
    # `Integrals.solve` returns an `IntegralSolution` whose element type
    # the compiler leaves as a free parameter (it cannot see through the
    # quadrature), so `[1]` infers as `Any`. Pin it to the bound type `m`
    # so the numeric path stays type stable, as the interface methods
    # expect. (Same Integrals.jl inference gap as `primarycensored_cdf`.)
    return oftype(m, solve(prob, GaussLegendre(; n = _CONVOLVED_NODES))[1])
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
    # Pin the element type (see scalar `_convolved_quadrature`); the
    # batched callers also re-convert per entry, but this keeps the helper
    # itself inferrable.
    return convert(Vector{typeof(m)},
        solve(prob, GaussLegendre(; n = _CONVOLVED_NODES)).u)
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

    lower, upper = _finite_window(last_comp, lower, upper)
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
# Concrete return type for the scalar interface methods. Pins the output
# to a single float type so the analytic/numeric branch (a `Union` from
# `_maybe_analytic`) does not leak into the inferred return type.
_convolved_T(d::Convolved, x::Real) = float(promote_type(eltype(d), typeof(x)))

function cdf(d::Convolved, x::Real)
    T = _convolved_T(d, x)
    analytic = _maybe_analytic(d)
    if analytic !== nothing
        return T(cdf(analytic, x))
    end
    return T(_convolved_numeric_cdf(d, x))
end

@doc "

Compute the log cumulative distribution function.

See also: [`cdf`](@ref)
"
function logcdf(d::Convolved, x::Real)
    T = _convolved_T(d, x)
    analytic = _maybe_analytic(d)
    if analytic !== nothing
        return T(logcdf(analytic, x))
    end
    c = _convolved_numeric_cdf(d, x)
    return c <= 0 ? T(-Inf) : T(log(c))
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
    T = _convolved_T(d, x)
    analytic = _maybe_analytic(d)
    if analytic !== nothing
        return T(pdf(analytic, x))
    end
    return T(_convolved_numeric_pdf(d, x))
end

@doc "

Compute the log probability density function.

See also: [`pdf`](@ref), [`logcdf`](@ref)
"
function logpdf(d::Convolved, x::Real)
    T = _convolved_T(d, x)
    analytic = _maybe_analytic(d)
    if analytic !== nothing
        return T(logpdf(analytic, x))
    end
    if !insupport(d, x)
        return T(-Inf)
    end
    p = _convolved_numeric_pdf(d, x)
    return p <= 0 ? T(-Inf) : T(log(p))
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
