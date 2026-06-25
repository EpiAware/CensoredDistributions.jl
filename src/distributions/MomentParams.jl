@doc """

A distribution parameterised by an alternative (moment) parameter set.

`MomentParams` holds a base `Distributions.jl` family under a named alternative
parameterisation (its moments, or any other reparameterisation) and delegates
every density / cdf / quantile / rand / mean / var computation to the equivalent
native distribution.
Only its parameterisation differs from the native family, so [`params`](@ref)
returns the alternative values in the registered order and the introspection /
prior front-door ([`params_table`](@ref), [`build_priors`](@ref),
[`update`](@ref)) lists and updates those alternative names rather than the
native ones.

This couples priors that act on a derived quantity.
A prior on a Gamma delay's MEAN, for example, cannot be expressed through the
native `Gamma(shape, scale)` leaf (an independent prior on `scale` cannot couple
to a prior on the mean), but `from_moments(Gamma; mean = …, shape = …)` makes
the mean a free parameter with the scale `mean / shape` derived.

The type parameters carry the native family `D` and the alternative parameter
names `names`, so the names are recoverable from the type alone.
Adding a family is a one-line [`register_moment_params`](@ref) declaration of how
its alternative parameters map to the native constructor.

Because it is a `UnivariateDistribution`, a `MomentParams` nests as a leaf in
[`compose`](@ref), [`choose`](@ref), and [`Compete`](@ref) automatically.
The wrapper and its registrations depend only on `Distributions.jl`, so they are
portable to a standalone reparameterisation package unchanged.

# Examples
```@example
using CensoredDistributions, Distributions

d = from_moments(Gamma; mean = 5.0, shape = 2.0)
mean(d)            # 5.0
logpdf(d, 4.0) == logpdf(Gamma(2.0, 2.5), 4.0)
```

# Fields
- `vals`: the alternative (moment) parameter values, in registered `names`
  order.

# See also
- [`from_moments`](@ref): the friendly front-end constructor
- [`register_moment_params`](@ref): register a family's moment map
"""
struct MomentParams{D, names, N, T <: Real} <:
       UnivariateDistribution{Continuous}
    "The alternative (moment) parameter values, in registered `names` order."
    vals::NTuple{N, T}
end

@doc """

Register how a family's alternative parameters map to its native constructor.

`register_moment_params(D, names) do vals ... end` declares, once per
`(family, names)` pair, the function turning a tuple of alternative parameter
values into the equivalent native distribution.
The map should build it with `check_args = false` (so an out-of-support proposal
yields `-Inf`, not a throw) and be plain arithmetic on `vals` (so a gradient
flows through the derived native parameters).

# Arguments
- `f`: a function `vals::NTuple -> ::UnivariateDistribution`.
- `D`: the native family type, e.g. `Gamma`.
- `names`: the alternative parameter names, e.g. `(:mean, :shape)`.

# Examples
```@example
using CensoredDistributions, Distributions

register_moment_params(Gamma, (:mean, :shape)) do (mean, shape)
    Gamma(shape, mean / shape; check_args = false)
end
```

# See also
- [`from_moments`](@ref): construct a leaf for a registered family
"""
function register_moment_params(f, ::Type{D}, names::Tuple) where {D}
    quoted = QuoteNode(names)
    return @eval function _moment_native(::Type{$D}, ::Val{$quoted}, vals)
        return $f(vals)
    end
end

# Moment map for a (family, names) pair; an unregistered pair errors clearly.
function _moment_native(::Type{D}, ::Val{names}, vals) where {D, names}
    throw(ArgumentError(
        "no moment parameterisation registered for $D with names " *
        "$(names); call register_moment_params($D, $(names)) to add one"))
end

# The derived native distribution every density method delegates to.
function _native(d::MomentParams{D, names}) where {D, names}
    return _moment_native(D, Val(names), d.vals)
end

_moment_names(::MomentParams{D, names}) where {D, names} = names
_moment_names(::Type{<:MomentParams{D, names}}) where {D, names} = names

@doc """

Construct a distribution under an alternative (moment) parameterisation.

`from_moments(D; alt_params...)` returns a [`MomentParams`](@ref) leaf for the
native family `D` parameterised by the registered alternative parameters.
The keyword names must match a [`register_moment_params`](@ref) declaration for
`D`; their values are promoted to a common real type and held in registered
order, with the native distribution derived on demand.

The call site reads as the parameterisation it expresses, e.g.
`from_moments(Gamma; mean = 5.0, shape = 2.0)`, and the leaf surfaces those
alternative names through the prior front-door so a prior on a derived quantity
couples correctly.
`check_args` (default `true`) toggles validation; the scoring path reconstructs
with `check_args = false` so an out-of-support proposal yields `-Inf` rather
than throwing mid-gradient.

# Arguments
- `D`: the native family type, e.g. `Gamma`.
- `alt_params`: the registered alternative parameters as keywords.

# Examples
```@example
using CensoredDistributions, Distributions

d = from_moments(Gamma; mean = 5.0, shape = 2.0)
mean(d)   # 5.0; the scale `mean / shape = 2.5` is derived

tree = compose((onset_admit = from_moments(Gamma; mean = 5.0, shape = 2.0),))
params_table(tree)
```

# See also
- [`MomentParams`](@ref): the wrapped type
- [`register_moment_params`](@ref): register a family's moment map
"""
function from_moments(::Type{D}; check_args::Bool = true,
        alt_params...) where {D}
    nt = values(alt_params)
    vals = promote(map(float, Tuple(nt))...)
    return _moment_params(D, keys(nt), vals; check_args = check_args)
end

# Build a `MomentParams{D, names}` from a value tuple, validating the derived
# native distribution (the family's own support rules) when `check_args` is on.
function _moment_params(::Type{D}, names::Tuple, vals::Tuple;
        check_args::Bool = true) where {D}
    d = MomentParams{D, names, length(vals), eltype(vals)}(vals)
    check_args && _check_native(d)
    return d
end

# Reconstruct the native distribution checked from its own params, so an invalid
# derived parameter is rejected up front by the family. Uses the base (UnionAll)
# constructor, which validates, not the parametric inner one.
function _check_native(d::MomentParams)
    native = _native(d)
    Base.typename(typeof(native)).wrapper(Distributions.params(native)...)
    return nothing
end

params(d::MomentParams) = d.vals

Base.eltype(::Type{<:MomentParams{D, names, N, T}}) where {D, names, N, T} = T

minimum(d::MomentParams) = minimum(_native(d))
maximum(d::MomentParams) = maximum(_native(d))
insupport(d::MomentParams, x::Real) = insupport(_native(d), x)

@doc "

Compute the probability density function.

See also: [`logpdf`](@ref)
"
pdf(d::MomentParams, x::Real) = pdf(_native(d), x)

@doc "

Compute the log probability density function.

See also: [`pdf`](@ref), [`cdf`](@ref)
"
logpdf(d::MomentParams, x::Real) = logpdf(_native(d), x)

@doc "

Compute the cumulative distribution function.

See also: [`logcdf`](@ref), [`quantile`](@ref)
"
cdf(d::MomentParams, x::Real) = cdf(_native(d), x)

@doc "

Compute the log cumulative distribution function.

See also: [`cdf`](@ref)
"
logcdf(d::MomentParams, x::Real) = logcdf(_native(d), x)

@doc "

Compute the complementary cumulative distribution function (survival).

See also: [`cdf`](@ref)
"
ccdf(d::MomentParams, x::Real) = ccdf(_native(d), x)

@doc "

Compute the log complementary cumulative distribution function.

See also: [`ccdf`](@ref)
"
logccdf(d::MomentParams, x::Real) = logccdf(_native(d), x)

@doc "

Compute the quantile function (inverse CDF).

See also: [`cdf`](@ref)
"
quantile(d::MomentParams, p::Real) = quantile(_native(d), p)

@doc "

Generate a random sample.

See also: [`quantile`](@ref)
"
Base.rand(rng::AbstractRNG, d::MomentParams) = rand(rng, _native(d))

@doc "

Return the mean, delegated to the derived native distribution.

See also: [`var`](@ref)
"
mean(d::MomentParams) = mean(_native(d))

@doc "

Compute the variance, delegated to the derived native distribution.

See also: [`mean`](@ref)
"
var(d::MomentParams) = var(_native(d))

# --- family registrations ---------------------------------------------------

# Gamma by (mean, shape): scale = mean / shape (the #710 / bdbv use-case).
register_moment_params(Gamma, (:mean, :shape)) do (mean, shape)
    Gamma(shape, mean / shape; check_args = false)
end

# LogNormal by (mean, sd): natural-scale moments mapped to native (mu, sigma).
register_moment_params(LogNormal, (:mean, :sd)) do (mean, sd)
    s2 = log1p((sd / mean)^2)
    LogNormal(log(mean) - s2 / 2, sqrt(s2); check_args = false)
end
