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
convolved distribution unless a [`NumericSolver`](@ref) method is set. All
other cases use AD-safe fixed-node Gauss-Legendre quadrature: the integral
is mapped from the fixed reference domain ``(-1, 1)`` onto the real bounds
inside the integrand and reduced as a bare weighted dot product
(`gl_integrate`), which lets every AD backend specialise on the integrand's
own type so component `Dual`s and tangents propagate.

The `method` field selects the CDF/PDF backend, mirroring
`primary_censored`: an [`AnalyticalSolver`](@ref) (the default) uses the
analytic convolution when one exists and falls back to quadrature
otherwise, while a [`NumericSolver`](@ref) forces the numeric quadrature
path even when an analytic convolution exists; the latter is useful for
validation and debugging.

# See also
- [`convolved`](@ref): Constructor function
"""
struct Convolved{C <: Tuple, M <: AbstractSolverMethod} <:
       UnivariateDistribution{Continuous}
    "Tuple of independent component distributions to be summed."
    components::C
    "Solver method choose the analytic vs numeric quadrature backend."
    method::M

    function Convolved(components::C;
            method::AbstractSolverMethod = AnalyticalSolver()) where {
            C <: Tuple}
        # The type allows a single-component (degenerate) convolution so the
        # recursive moment fold and rebuild paths can wrap one component; the
        # user-facing `convolved` requires two or more, since convolving
        # fewer is a no-op.
        length(components) >= 1 ||
            throw(ArgumentError("Convolved needs at least one component"))
        all(c -> c isa UnivariateDistribution, components) ||
            throw(ArgumentError(
                "All components must be UnivariateDistributions"))
        new{C, typeof(method)}(components, method)
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
- `method`: The solver method, an [`AnalyticalSolver`](@ref) (the default)
  or [`NumericSolver`](@ref). `NumericSolver` forces numeric quadrature
  even when an analytic convolution is available, mirroring
  `primary_censored`.

# Examples
```@example
using CensoredDistributions, Distributions

# Sum of two delays
d = convolved(Gamma(2.0, 1.0), LogNormal(1.5, 0.5))
cdf_at_5 = cdf(d, 5.0)

# Sum of three delays from a vector
d3 = convolved([Gamma(2.0, 1.0), Gamma(1.0, 1.0), Normal(0.0, 1.0)])
mean_sample = rand(d3)

# Force numeric quadrature even for an analytic pair
dn = convolved(Normal(0.0, 1.0), Normal(1.0, 2.0);
    method = NumericSolver())
cdf_numeric = cdf(dn, 2.0)
```

# See also
- [`Convolved`](@ref): The distribution type
"
function convolved(
        components::AbstractVector{<:UnivariateDistribution};
        method::AbstractSolverMethod = AnalyticalSolver())
    length(components) >= 2 ||
        throw(ArgumentError("convolved needs at least two components"))
    return Convolved(Tuple(components); method = method)
end

function convolved(
        c1::UnivariateDistribution, c2::UnivariateDistribution,
        rest::UnivariateDistribution...;
        method::AbstractSolverMethod = AnalyticalSolver())
    return Convolved((c1, c2, rest...); method = method)
end

function convolved(components::Tuple;
        method::AbstractSolverMethod = AnalyticalSolver())
    length(components) >= 2 ||
        throw(ArgumentError("convolved needs at least two components"))
    return Convolved(components; method = method)
end

# ---------------------------------------------------------------------------
# Interface: params / support / sampling
# ---------------------------------------------------------------------------

params(d::Convolved) = map(params, d.components)

function Base.eltype(::Type{<:Convolved{C}}) where {C <: Tuple}
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
# Moments: exact analytic sum of independent components
# ---------------------------------------------------------------------------
#
# A `Convolved` is a sum of independent components, so the mean and variance
# are EXACT and additive: `mean = sum(mean.(components))` and
# `var = sum(var.(components))`. No sampling, no discretisation. Each component
# must provide an analytic `mean`/`var`; a component without one errors from
# its own `mean`/`var` (no fallback). The moments flow through the component
# parameters, so the path is AD-safe.

# Per-component moments via the component's analytic `mean`/`var`. A nested
# `Convolved` is itself a `UnivariateDistribution`, so it recurses through this
# same additive sum via its own `mean`/`var`.
_component_mean(c::UnivariateDistribution) = mean(c)
_component_var(c::UnivariateDistribution) = var(c)

@doc "

Mean of the convolution: the exact sum of the component means.

A [`Convolved`](@ref) is a sum of independent components, so the mean is
``\\sum_i \\mathbb{E}[X_i]``. Each component must provide an analytic `mean`;
a component without one errors (there is no numeric fallback).

See also: [`var`](@ref), [`std`](@ref)
"
mean(d::Convolved) = sum(_component_mean, d.components)

@doc "

Variance of the convolution: the exact sum of the component variances.

Independence makes the variance additive, ``\\sum_i \\mathrm{Var}[X_i]``. As
for [`mean`](@ref), each component must provide an analytic `var` or the call
errors.

See also: [`mean`](@ref), [`std`](@ref)
"
var(d::Convolved) = sum(_component_var, d.components)

@doc "

Standard deviation of the convolution, ``\\sqrt{\\mathrm{Var}[X]}``.

See also: [`var`](@ref), [`mean`](@ref)
"
std(d::Convolved) = sqrt(var(d))

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
# or when `d.method` is a `NumericSolver` requesting the numeric quadrature
# path.
function _maybe_analytic(d::Convolved)
    d.method isa NumericSolver && return nothing
    return _analytic_convolution(d.components)
end

# Fraction of probability trimmed from each tail of an unbounded
# integration component when clamping an infinite quadrature window.
const _CONVOLVED_TAIL = 1e-8

# Strip any AD wrapper (ForwardDiff `Dual`, ReverseDiff `TrackedReal`,
# Enzyme/Mooncake duals) from a scalar, returning its underlying primal
# value. The generic method is the identity on a plain real (so a
# non-AD call keeps the component's own float type, e.g. `Float32`); the
# per-backend extensions (`...ForwardDiffExt`, `...ReverseDiffExt`) add
# unwrapping methods, and `...ChainRulesCoreExt` / `...EnzymeExt` register
# it as non-differentiable so reverse and Enzyme modes do not trace it.
# This is what keeps the quadrature window (a non-differentiable
# hyperparameter — just *where* to integrate) off `quantile`'s AD path.
_primal(x::Real) = x

# Quantile used only to pick a finite quadrature endpoint. Reconstructing
# the component from primal (AD-stripped) params means `quantile` — and so
# `gamma_inc_inv` for a `Gamma` integration component — only ever sees
# plain `Float64`s. No AD backend differentiates through it, which is
# correct: the window choice carries no gradient (it is a fixed
# hyperparameter of the quadrature, like the node count). Previously the
# live `Dual`/`TrackedReal` params flowed into `quantile`, and Enzyme
# cannot push duals through `SpecialFunctions.gamma_inc_inv_qsmall`
# (`IllegalTypeAnalysisException`).
# `@noinline` so the call survives as a call site for the per-backend AD
# rules (the Enzyme `EnzymeRules` rule and the ChainRules
# `@non_differentiable` mark) to attach to; if it inlined, Enzyme would
# type-analyse `quantile`/`gamma_inc_inv` directly and abort.
@noinline function _window_quantile(comp::UnivariateDistribution, p::Real)
    primal = _primal_distribution(comp)
    return quantile(primal, p)
end

# Rebuild a distribution with its parameters stripped to primal `Float64`s
# via the type's positional constructor (`params` round-trips through the
# constructor for the Distributions.jl families used here). The
# `check_args = false` keyword is intentionally NOT passed: the original
# distribution already validated its parameters, and the primal copy uses
# the identical (now plain-`Float64`) values.
function _primal_distribution(d::UnivariateDistribution)
    D = Base.typename(typeof(d)).wrapper
    return D(map(_primal, params(d))...)
end

# Clamp an integration window to a finite range. Both the integrand's
# `f_C(t)` factor and (for the CDF) the transition of `F_R(x - t)` are
# negligible outside the integration component's effective support, so an
# infinite endpoint is replaced by an extreme quantile of `last_comp`.
# This lets the numeric path handle components unbounded on either side
# (e.g. Normal+Normal under a `NumericSolver`). The endpoint quantile is
# computed on AD-stripped params (`_window_quantile`) so the window stays
# a non-differentiated constant across every AD backend.
function _finite_window(last_comp, lower::Real, upper::Real)
    lo = isfinite(lower) ? lower : _window_quantile(last_comp, _CONVOLVED_TAIL)
    hi = isfinite(upper) ? upper :
         _window_quantile(last_comp, 1 - _CONVOLVED_TAIL)
    return lo, hi
end

# ---------------------------------------------------------------------------
# Numeric convolution (AD-safe Gauss-Legendre dot product)
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

# The convolution quadrature uses the shared Gauss-Legendre machinery in
# `src/integration/integration.jl`: the `_CONVOLVED_GL` rule (192 nodes,
# see there for the node-count rationale) and `gl_integrate`. Holding the
# rule directly and reducing inline keeps the path AD-safe (the
# accumulator type is seeded from the integrand). The batched companion
# below reuses the same nodes/weights for a one-pass vector solve.

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
