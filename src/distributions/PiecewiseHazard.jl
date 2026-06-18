@doc """

A nonparametric delay distribution on `[0, ∞)` defined by a piecewise-constant
hazard on a grid.

Given interior breakpoints ``0 < b_1 < b_2 < \\dots < b_{K-1}`` and per-interval
hazard values ``h_1, \\dots, h_K > 0``, the hazard is the step function

```math
h(t) = h_k \\quad \\text{for } t \\in [b_{k-1}, b_k),
```

with ``b_0 = 0`` and ``b_K = ∞``. The cumulative hazard is the running integral
of the steps,

```math
H(t) = \\int_0^t h(u)\\,\\mathrm{d}u,
```

which is continuous and piecewise-linear, and the survival, density and
distribution functions follow as

```math
S(t) = \\exp(-H(t)), \\qquad
f(t) = h(t)\\,S(t), \\qquad
F(t) = 1 - S(t).
```

The final hazard ``h_K`` governs the tail on ``[b_{K-1}, ∞)``; as long as it is
positive the distribution is proper (``S(t) \\to 0``). The hazard values are the
differentiable parameters, so the leaf fits cleanly with ForwardDiff and
Mooncake.

Because it is a `UnivariateDistribution`, a `PiecewiseHazard` nests as a leaf in
[`compose`](@ref), the censoring wrappers, and the truncation helpers. It is a
natural fit for the racing-hazard [`compete`](@ref) node, which multiplies branch
survivals, so a flexible cause-specific hazard drops straight in.

# Fields
- `breaks`: interior breakpoints ``b_1 < \\dots < b_{K-1}`` (length `K - 1`).
- `hazards`: per-interval constant hazard values ``h_1, \\dots, h_K > 0``
  (length `K`); the differentiable parameters.
- `cumhaz`: cumulative hazard cached at each interior breakpoint,
  ``H(b_1), \\dots, H(b_{K-1})``, so the survival and distribution functions
  need no running sum.

# See also
- [`piecewise_hazard`](@ref): constructor function
"""
struct PiecewiseHazard{T <: Real} <: UnivariateDistribution{Continuous}
    "Interior breakpoints ``0 < b_1 < \\dots < b_{K-1}`` (length `K - 1`)."
    breaks::Vector{T}
    "Per-interval constant hazard values ``h_1, \\dots, h_K > 0`` (length `K`)."
    hazards::Vector{T}
    "Cumulative hazard at each interior breakpoint, ``H(b_1), \\dots, H(b_{K-1})``
    (length `K - 1`); cached so the CDF / survival need no running sum."
    cumhaz::Vector{T}

    function PiecewiseHazard{T}(
            breaks::Vector{T}, hazards::Vector{T}) where {T <: Real}
        length(hazards) == length(breaks) + 1 ||
            throw(ArgumentError(
                "hazards must have one more element than breaks"))
        all(h -> h > zero(h), hazards) ||
            throw(ArgumentError("all hazards must be positive"))
        all(isfinite, hazards) ||
            throw(ArgumentError("all hazards must be finite"))
        all(b -> b > zero(b), breaks) ||
            throw(ArgumentError("all breakpoints must be positive"))
        issorted(breaks; lt = <=) ||
            throw(ArgumentError("breaks must be strictly increasing"))
        # Cumulative hazard at the interior breakpoints: H(b_j) accumulates the
        # rectangle widths (b_k - b_{k-1}) times the per-interval hazard h_k.
        cumhaz = similar(breaks)
        acc = zero(T)
        prev = zero(T)
        for j in eachindex(breaks)
            acc += hazards[j] * (breaks[j] - prev)
            cumhaz[j] = acc
            prev = breaks[j]
        end
        new{T}(breaks, hazards, cumhaz)
    end
end

@doc "

Create a piecewise-constant hazard distribution on `[0, ∞)`.

The hazard takes the value `hazards[k]` on the `k`th interval. The intervals are
`[0, breaks[1])`, `[breaks[1], breaks[2])`, …, `[breaks[end], ∞)`, so `hazards`
holds one more value than `breaks`.

# Arguments
- `breaks`: interior breakpoints, strictly increasing and positive
  (length `K - 1`).
- `hazards`: per-interval positive hazard values (length `K`).

# Examples
```@example
using CensoredDistributions, Distributions

# A three-piece hazard: rising then falling
d = piecewise_hazard([1.0, 3.0], [0.2, 0.8, 0.3])
pdf(d, 2.0)
cdf(d, 2.0)
mean(d)

# A constant hazard reproduces an Exponential
e = piecewise_hazard(Float64[], [0.5])
logpdf(e, 2.0) ≈ logpdf(Exponential(2.0), 2.0)
```

# See also
- [`PiecewiseHazard`](@ref): the wrapped type
"
function piecewise_hazard(breaks::AbstractVector, hazards::AbstractVector)
    T = float(promote_type(eltype(breaks), eltype(hazards)))
    b = collect(T, breaks)
    h = collect(T, hazards)
    return PiecewiseHazard{T}(b, h)
end

# Parameter extraction: the breakpoints and the hazard values. The hazards are
# the differentiable parameters; the breakpoints are the fixed grid.
params(d::PiecewiseHazard) = (d.breaks, d.hazards)

Base.eltype(::Type{<:PiecewiseHazard{T}}) where {T} = T

minimum(d::PiecewiseHazard) = zero(eltype(d))
maximum(d::PiecewiseHazard{T}) where {T} = T(Inf)
insupport(d::PiecewiseHazard, x::Real) = x >= zero(x) && isfinite(x)

# The index of the interval containing `t >= 0`: the count of interior
# breakpoints strictly below `t`, plus one. Returns `k ∈ 1:length(hazards)`.
function _interval_index(d::PiecewiseHazard, t::Real)
    k = 1
    @inbounds for b in d.breaks
        t < b && break
        k += 1
    end
    return k
end

# The hazard `h(t)` at `t >= 0`.
function _hazard_at(d::PiecewiseHazard, t::Real)
    return d.hazards[_interval_index(d, t)]
end

# The cumulative hazard `H(t) = ∫_0^t h(u) du` at `t >= 0`. The cached `cumhaz`
# carries the value at each interior breakpoint, so `H(t)` is the value at the
# breakpoint below `t` plus the current interval's hazard times the remaining
# distance into that interval. AD flows through `hazards` (and the cache, which
# is built from them) on the differentiated type.
function _cumulative_hazard(d::PiecewiseHazard, t::Real)
    T = float(promote_type(eltype(d), typeof(t)))
    t <= zero(t) && return zero(T)
    k = _interval_index(d, t)
    base = k == 1 ? zero(T) : T(d.cumhaz[k - 1])
    lower = k == 1 ? zero(T) : T(d.breaks[k - 1])
    return base + T(d.hazards[k]) * (t - lower)
end

@doc "

Compute the probability density function.

See also: [`logpdf`](@ref)
"
pdf(d::PiecewiseHazard, x::Real) = exp(logpdf(d, x))

@doc "

Compute the log probability density function `log h(t) - H(t)`.

See also: [`pdf`](@ref), [`cdf`](@ref)
"
function logpdf(d::PiecewiseHazard, x::Real)
    T = float(promote_type(eltype(d), typeof(x)))
    insupport(d, x) || return oftype(zero(T), -Inf)
    h = T(_hazard_at(d, x))
    return log(h) - _cumulative_hazard(d, x)
end

@doc "

Compute the survival function `S(t) = exp(-H(t))`.

See also: [`cdf`](@ref), [`logccdf`](@ref)
"
function ccdf(d::PiecewiseHazard, x::Real)
    T = float(promote_type(eltype(d), typeof(x)))
    x <= zero(x) && return one(T)
    return exp(-_cumulative_hazard(d, x))
end

@doc "

Compute the log survival function `-H(t)`.

See also: [`ccdf`](@ref), [`logcdf`](@ref)
"
function logccdf(d::PiecewiseHazard, x::Real)
    T = float(promote_type(eltype(d), typeof(x)))
    x <= zero(x) && return zero(T)
    return -_cumulative_hazard(d, x)
end

@doc "

Compute the cumulative distribution function `F(t) = 1 - S(t)`.

See also: [`logcdf`](@ref), [`quantile`](@ref)
"
function cdf(d::PiecewiseHazard, x::Real)
    T = float(promote_type(eltype(d), typeof(x)))
    x <= zero(x) && return zero(T)
    return -expm1(-_cumulative_hazard(d, x))
end

@doc "

Compute the log cumulative distribution function `log(1 - S(t))`.

See also: [`cdf`](@ref)
"
function logcdf(d::PiecewiseHazard, x::Real)
    T = float(promote_type(eltype(d), typeof(x)))
    x <= zero(x) && return oftype(zero(T), -Inf)
    return log1mexp(-_cumulative_hazard(d, x))
end

@doc "

Compute the quantile function (inverse CDF) by inverting the cumulative hazard.

The target cumulative hazard is `H* = -log(1 - p)`; the interval holding it is
found from the cached breakpoint cumulative hazards, then inverted linearly
within that interval's constant hazard.

See also: [`cdf`](@ref)
"
function quantile(d::PiecewiseHazard, p::Real)
    (p < 0 || p > 1) && throw(ArgumentError("p must be in [0, 1]"))
    T = float(promote_type(eltype(d), typeof(p)))
    p <= zero(p) && return zero(T)
    p >= one(p) && return T(Inf)
    # Target cumulative hazard: solve H(t) = -log(1 - p) for t.
    target = -log1p(-p)
    # Walk the cached breakpoint cumulative hazards to find the interval.
    k = 1
    lower = zero(T)
    base = zero(T)
    @inbounds for j in eachindex(d.breaks)
        T(d.cumhaz[j]) >= target && break
        base = T(d.cumhaz[j])
        lower = T(d.breaks[j])
        k += 1
    end
    return lower + (target - base) / T(d.hazards[k])
end

@doc "

Generate a random sample by inverting the cumulative hazard at an exponential
draw: with `E ~ Exponential(1)`, `T = H⁻¹(E)` has survival `exp(-H(t))`.

See also: [`quantile`](@ref)
"
function Base.rand(rng::AbstractRNG, d::PiecewiseHazard)
    # Exponential-of-cumulative-hazard sampling: draw a unit-rate exponential
    # E = -log(U) and invert H, equivalent to inverse-CDF sampling.
    return quantile(d, rand(rng))
end

@doc "

Compute the mean `E[T] = ∫_0^∞ S(t) dt`.

The survival is piecewise log-linear, so the integral is the closed-form sum of
the per-interval exponential contributions plus the final unbounded tail.

See also: [`var`](@ref)
"
function mean(d::PiecewiseHazard)
    T = float(eltype(d))
    # E[T] = ∫_0^∞ S(t) dt. On interval [b_{k-1}, b_k) the survival is
    # S(b_{k-1}) exp(-h_k (t - b_{k-1})), whose integral over the interval is
    # S(b_{k-1}) (1 - exp(-h_k Δ)) / h_k; the final interval is unbounded so its
    # contribution is S(b_{K-1}) / h_K.
    total = zero(T)
    surv = one(T)
    prev = zero(T)
    @inbounds for k in eachindex(d.breaks)
        h = T(d.hazards[k])
        width = T(d.breaks[k]) - prev
        total += surv * (-expm1(-h * width)) / h
        surv *= exp(-h * width)
        prev = T(d.breaks[k])
    end
    total += surv / T(d.hazards[end])
    return total
end

@doc "

Compute the second moment `E[T²] = ∫_0^∞ 2 t S(t) dt`.

See also: [`mean`](@ref), [`var`](@ref)
"
function _second_moment(d::PiecewiseHazard)
    T = float(eltype(d))
    # E[T²] = ∫_0^∞ 2 t S(t) dt. On [b_{k-1}, b_k) with S(t) = S_{k-1}
    # exp(-h (t - b_{k-1})) the contribution of 2 t S(t) integrates in closed
    # form; the unbounded tail uses ∫_a^∞ 2 t e^{-h (t-a)} dt = 2 e^{...}
    # (a / h + 1 / h²) evaluated at the tail anchor.
    total = zero(T)
    surv = one(T)
    prev = zero(T)
    @inbounds for k in eachindex(d.breaks)
        h = T(d.hazards[k])
        a = prev
        b = T(d.breaks[k])
        e = exp(-h * (b - a))
        # ∫_a^b 2 t e^{-h (t-a)} dt, scaled by S(a) = surv.
        term = 2 * (a / h + 1 / h^2) -
               2 * e * (b / h + 1 / h^2)
        total += surv * term
        surv *= e
        prev = b
    end
    h = T(d.hazards[end])
    a = prev
    # ∫_a^∞ 2 t e^{-h (t-a)} dt = 2 (a / h + 1 / h²), scaled by S(a).
    total += surv * 2 * (a / h + 1 / h^2)
    return total
end

@doc "

Compute the variance `E[T²] - E[T]²`.

See also: [`mean`](@ref), [`std`](@ref)
"
function var(d::PiecewiseHazard)
    m = mean(d)
    return _second_moment(d) - m^2
end

@doc "

Compute the standard deviation.

See also: [`var`](@ref), [`mean`](@ref)
"
std(d::PiecewiseHazard) = sqrt(var(d))

@doc "

Compute the median.

See also: [`quantile`](@ref)
"
median(d::PiecewiseHazard) = quantile(d, 0.5)
