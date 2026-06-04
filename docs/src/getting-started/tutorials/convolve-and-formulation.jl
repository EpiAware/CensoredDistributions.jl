md"""
# [Composing delays and choosing a formulation](@id convolve-and-formulation)

This how-to covers two pieces of `CensoredDistributions.jl`.
The first is composing several delays into one with [`generic_convolve`](@ref).
The second is choosing how a primary-censored delay treats the primary event,
using the [`Marginal`](@ref) and [`Latent`](@ref) formulations.

It assumes you have read
[Getting Started with CensoredDistributions.jl](@ref getting-started).
For the mathematical detail and the full argument list, follow the links to
the API reference rather than relying on this page.
"""

md"""
### Packages used
"""

using CensoredDistributions
using Distributions

md"""
## Composing delays with `generic_convolve`

An observed delay is often the sum of several independent stages.
Onset-to-death, for example, is onset-to-admission plus admission-to-death.
[`generic_convolve`](@ref) builds the distribution of that sum from its
component distributions.
"""

onset_to_admission = Gamma(2.0, 1.0)
admission_to_death = LogNormal(1.5, 0.5)

onset_to_death = generic_convolve(onset_to_admission, admission_to_death)
mean(rand(onset_to_death, 10_000))

md"""
The result is an ordinary `UnivariateDistribution`, so `pdf`, `cdf`,
`logpdf`, `quantile` and `rand` all work.
"""

cdf(onset_to_death, 5.0)

md"""
You can pass any number of components, either positionally or as a vector.
"""

three_stage = generic_convolve([Gamma(2.0, 1.0), Gamma(1.0, 1.0),
    LogNormal(0.5, 0.5)])
cdf(three_stage, 5.0)

md"""
### Analytic and numeric paths

When a closed-form convolution exists for every component pair
(`Distributions.convolve` applies, for example `Normal` with `Normal` or
equal-scale `Gamma`) the CDF is taken from the analytic convolution.
"""

analytic = generic_convolve(Gamma(2.0, 1.5), Gamma(3.0, 1.5))
cdf(analytic, 8.0)

md"""
Otherwise the CDF and density use AD-safe Gauss-Legendre quadrature, so
mismatched families still compose.
"""

numeric = generic_convolve(Gamma(2.0, 1.0), LogNormal(1.5, 0.5))
cdf(numeric, 8.0)

md"""
### Capping a component inside the convolution

The `bounds` keyword caps a component's contribution from inside the integral.
Pass one `(lower, upper)` pair per component.
This differs from truncating the whole sum, which cannot reach inside the
convolution.
"""

capped = generic_convolve(LogNormal(1.5, 0.5), Gamma(2.0, 1.0);
    bounds = [(-Inf, Inf), (0.0, 3.0)])
cdf(capped, 5.0)

md"""
With a finite bound the CDF returns the unnormalised joint mass
``P(\sum_i X_i \le x \wedge a_i \le X_i \le b_i)``, so it saturates below 1.
Divide by the saturated mass if you need a normalised conditional
distribution.
See [`generic_convolve`](@ref) and [`Convolved`](@ref) for the full account.
"""

md"""
## Choosing a formulation for primary censoring

A primary-censored delay can be written in two equivalent ways.
The `formulation` keyword of [`primary_censored`](@ref) selects between them.
"""

d = primary_censored(LogNormal(1.5, 0.75), Uniform(0.0, 1.0))

md"""
### Marginalise over the primary event

[`Marginal`](@ref) is the default.
It integrates the latent primary event time out, analytically where a closed
form exists and by quadrature otherwise.
Use it when you only need the delay distribution itself.
"""

marginal = primary_censored(LogNormal(1.5, 0.75), Uniform(0.0, 1.0);
    formulation = Marginal())
logpdf(marginal, 2.0)

md"""
### Condition on a known primary event

[`Latent`](@ref)`(p)` conditions on a concrete primary event time `p`.
The conditional CDF collapses to ``F_\mathrm{delay}(x - p)`` exactly, with no
quadrature.
This is data augmentation: in a probabilistic model you sample `p` from the
prior and condition the delay on it.

[`primary_prior`](@ref) returns the prior over `p` for a fitted distribution.
"""

prior = primary_prior(d)
p = rand(prior)
latent = primary_censored(LogNormal(1.5, 0.75), Uniform(0.0, 1.0);
    formulation = Latent(p))
logpdf(latent, 2.0)

md"""
In your own PPL the pattern is

```julia
p ~ primary_prior(d)
y ~ primary_censored(delay, primary; formulation = Latent(p))
```

Sampling `p` from the prior and conditioning recovers the marginal
formulation, so the two agree in expectation.
The latent path is cheaper per evaluation and avoids quadrature, at the cost of
an extra sampled parameter per record.

See [`Marginal`](@ref), [`Latent`](@ref) and [`primary_prior`](@ref) for
details, including the coupled prior used with a secondary censoring window.
"""
