@doc """

A Gamma delay reparameterised by its mean and shape.

`MeanGamma(mean, shape)` is the Gamma distribution with the given `mean > 0` and
`shape > 0`, holding the scale `θ = mean / shape` as a derived quantity. It is
density-identical to `Gamma(shape, mean / shape)`; only its parameterisation
differs, so [`params`](@ref) returns `(mean, shape)` and the introspection /
prior front-door ([`params_table`](@ref), [`build_priors`](@ref),
[`update`](@ref)) lists and updates `mean` and `shape` rather than the native
`(shape, scale)`.

This matches upstream delay models that place priors directly on a delay's
MEAN (with a coupled scale), which the native `Gamma(shape, scale)` leaf cannot
express: an independent prior on `scale` cannot couple to a prior on the mean.
With `MeanGamma`, a prior on the mean is a prior on a free parameter and the
scale follows.

Because it is a `UnivariateDistribution`, a `MeanGamma` nests as a leaf in
[`compose`](@ref), [`choose`](@ref), and [`Compete`](@ref) automatically.

# Examples
```@example
using CensoredDistributions, Distributions

d = MeanGamma(5.0, 2.0)
mean(d)            # 5.0
logpdf(d, 4.0) == logpdf(Gamma(2.0, 2.5), 4.0)
```

# See also
- [`mean_gamma`](@ref): the friendly lower-case constructor
"""
struct MeanGamma{T <: Real} <: UnivariateDistribution{Continuous}
    "The mean of the distribution (positive)."
    mean::T
    "The shape parameter (positive)."
    shape::T

    function MeanGamma{T}(mean::T, shape::T; check_args::Bool = true) where {
            T <: Real}
        if check_args
            mean > zero(mean) ||
                throw(ArgumentError("mean must be positive"))
            shape > zero(shape) ||
                throw(ArgumentError("shape must be positive"))
        end
        return new{T}(mean, shape)
    end
end

@doc """

Construct a mean/shape-parameterised Gamma `MeanGamma(mean, shape)`.

The two positional arguments are promoted to a common real type. `check_args`
(default `true`) toggles the positivity checks; the scoring path reconstructs
leaves with `check_args = false` so an out-of-support sampler proposal yields
`-Inf` rather than throwing mid-gradient.

# Arguments
- `mean`: the positive mean of the delay.
- `shape`: the positive Gamma shape.

# See also
- [`MeanGamma`](@ref): the wrapped type
"""
function MeanGamma(mean::Real, shape::Real; check_args::Bool = true)
    m, s = promote(float(mean), float(shape))
    return MeanGamma{typeof(m)}(m, s; check_args = check_args)
end

@doc """

Construct a mean/shape-parameterised Gamma delay.

`mean_gamma(mean, shape)` is the friendly lower-case constructor for
[`MeanGamma`](@ref), mirroring [`affine`](@ref). It returns the Gamma with the
given `mean` and `shape` and the derived scale `mean / shape`.

# Arguments
- `mean`: the positive mean of the delay.
- `shape`: the positive Gamma shape.

# Examples
```@example
using CensoredDistributions, Distributions

d = mean_gamma(5.0, 2.0)
mean(d)   # 5.0; the scale `mean / shape = 2.5` is derived

# A prior on the MEAN now couples correctly through the front-door.
tree = compose((onset_admit = mean_gamma(5.0, 2.0),))
params_table(tree)
```

# See also
- [`MeanGamma`](@ref): the wrapped type
"""
mean_gamma(mean::Real, shape::Real) = MeanGamma(mean, shape)

# The equivalent native Gamma, the engine every density method delegates to.
# Built with `check_args = false`: a `MeanGamma` validates on construction (or
# is deliberately unchecked in the scoring path), so the derived Gamma needs no
# second check, and an unchecked `MeanGamma` stays consistent.
_as_gamma(d::MeanGamma) = Gamma(d.shape, d.mean / d.shape; check_args = false)

# Parameter extraction: the natural (mean, shape) pair. This is what
# `_param_names(::MeanGamma)` labels and what `update` / the scoring path
# reconstruct from, so the front-door lists `mean` and `shape`.
params(d::MeanGamma) = (d.mean, d.shape)

Base.eltype(::Type{<:MeanGamma{T}}) where {T} = T

minimum(d::MeanGamma) = minimum(_as_gamma(d))
maximum(d::MeanGamma) = maximum(_as_gamma(d))
insupport(d::MeanGamma, x::Real) = insupport(_as_gamma(d), x)

@doc "

Compute the probability density function.

See also: [`logpdf`](@ref)
"
pdf(d::MeanGamma, x::Real) = pdf(_as_gamma(d), x)

@doc "

Compute the log probability density function.

See also: [`pdf`](@ref), [`cdf`](@ref)
"
logpdf(d::MeanGamma, x::Real) = logpdf(_as_gamma(d), x)

@doc "

Compute the cumulative distribution function.

See also: [`logcdf`](@ref), [`quantile`](@ref)
"
cdf(d::MeanGamma, x::Real) = cdf(_as_gamma(d), x)

@doc "

Compute the log cumulative distribution function.

See also: [`cdf`](@ref)
"
logcdf(d::MeanGamma, x::Real) = logcdf(_as_gamma(d), x)

@doc "

Compute the complementary cumulative distribution function (survival).

See also: [`cdf`](@ref)
"
ccdf(d::MeanGamma, x::Real) = ccdf(_as_gamma(d), x)

@doc "

Compute the log complementary cumulative distribution function.

See also: [`ccdf`](@ref)
"
logccdf(d::MeanGamma, x::Real) = logccdf(_as_gamma(d), x)

@doc "

Compute the quantile function (inverse CDF).

See also: [`cdf`](@ref)
"
quantile(d::MeanGamma, p::Real) = quantile(_as_gamma(d), p)

@doc "

Generate a random sample.

See also: [`quantile`](@ref)
"
Base.rand(rng::AbstractRNG, d::MeanGamma) = rand(rng, _as_gamma(d))

@doc "

Return the mean, the first natural parameter.

See also: [`var`](@ref)
"
mean(d::MeanGamma) = d.mean

@doc "

Compute the variance, `mean^2 / shape`.

See also: [`mean`](@ref)
"
var(d::MeanGamma) = d.mean^2 / d.shape
