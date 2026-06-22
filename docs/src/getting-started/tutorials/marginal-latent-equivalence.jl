md"""
# [Marginal and latent forms of a censored delay](@id marginal-latent-equivalence)

## Introduction

### What are we going to do in this tutorial

A primary-censored delay has two equivalent forms that share the same
parameters.

1. The **marginal** form is the plain `primary_censored` (or
   `double_interval_censored`) distribution. The primary event time is
   integrated out inside `logpdf` and `cdf`, so the distribution is univariate
   over the observed time alone.
2. The **latent** form, `latent(d)`, keeps the primary event time as an explicit
   sampled dimension. A draw is a labelled `(primary, observed)` record, the
   joint `logpdf([primary, observed])` scores the primary prior plus the
   conditional of the observed time given that primary, and the OBSERVED
   marginal (`cdf`, `pdf`, single-value `logpdf`) delegates straight back to the
   marginal node.

These are one model scored two ways. The latent form is just a wrapper around
the marginal node, so its observed-delay `cdf` and `pdf` agree with the marginal
form analytically. We demonstrate that here, show it for a
`double_interval_censored` delay too, and forward-simulate event paths with
`rand(latent(d))`, all without Turing.

### What might I need to know before starting

This tutorial builds on
[Getting Started with CensoredDistributions.jl](@ref getting-started).
It uses the [`latent`](@ref) wrapper, its inverse [`marginal`](@ref), and the
Distributions interface (`cdf`, `pdf`, `logpdf`) that both forms share.

## Packages used

We use Distributions for the delay distribution and Random for reproducible
draws.
"""

using CensoredDistributions
using Distributions
using Random

md"""
## Build the marginal and latent forms

We start from a primary-censored log-normal delay with a uniform primary event
window `[0, 1]`. The plain node is the marginal form; wrapping it in `latent`
selects the latent form. `marginal` is the inverse, recovering the wrapped node.
"""

delay = LogNormal(1.4, 0.5)

marginal_delay = primary_censored(delay, Uniform(0, 1))

latent_delay = latent(marginal_delay)

md"""
The latent form wraps the marginal node, and `marginal` recovers it unchanged.
"""

marginal(latent_delay) === marginal_delay

md"""
## The observed marginal of the latent form is the marginal node

The OBSERVED-delay marginal under the latent model is exactly the marginal
distribution it wraps. The package provides the full Distributions interface on
the latent object (`cdf`, `pdf`, single-value `logpdf`, and the rest), each
delegating analytically to `marginal(d)`. So the latent `cdf` IS the marginal
(target) `cdf`: no quadrature and no Monte Carlo are needed.

We compare the two forms across a set of evaluation points.
"""

eval_points = [1.0, 3.0, 6.0, 10.0]

cdf_comparison = [(x = x,
                      marginal = cdf(marginal_delay, x),
                      latent = cdf(latent_delay, x))
                  for x in eval_points]

md"""
The two `cdf` columns coincide exactly because the latent `cdf` delegates to the
marginal node. The largest absolute gap across the points is the difference
between a number and itself.
"""

cdf_gap = maximum(abs(r.marginal - r.latent) for r in cdf_comparison)

md"""
The density agrees the same way. The single-value observed `logpdf` of the
latent form delegates to the marginal node, so the two pdfs match.
"""

pdf_comparison = [(x = x,
                      marginal = pdf(marginal_delay, x),
                      latent = pdf(latent_delay, x))
                  for x in eval_points]

md"""
## The same equivalence holds for a double-interval-censored delay

The censoring wrappers work for both forms. We build a
`double_interval_censored` delay, the marginal form, and wrap it in `latent`.
Scoring a single observed value through the latent form delegates to the
interval-censored marginal node, so the density and `cdf` match the marginal
form.
"""

marginal_dic = double_interval_censored(
    delay; primary_event = Uniform(0, 1), upper = 10, interval = 1)

latent_dic = latent(marginal_dic)

dic_comparison = [(x = x,
                      marginal_cdf = cdf(marginal_dic, x),
                      latent_cdf = cdf(latent_dic, x),
                      marginal_logpdf = logpdf(marginal_dic, x),
                      latent_logpdf = logpdf(latent_dic, x))
                  for x in [1.0, 3.0, 6.0, 9.0]]

md"""
The marginal and latent `cdf` and `logpdf` columns coincide, so scoring the
double-interval-censored delay in either form gives the same answer.
"""

dic_cdf_gap = maximum(abs(r.marginal_cdf - r.latent_cdf) for r in dic_comparison)

dic_logpdf_gap = maximum(
    abs(r.marginal_logpdf - r.latent_logpdf) for r in dic_comparison)

md"""
## Forward-simulate event paths from the latent form

`rand(latent(d))` draws a full event path as a labelled
`(primary = ..., observed = ...)` record. The primary lies in the window and the
observed time exceeds it by a positive delay. No model or conditioning is
needed.
"""

one_path = rand(MersenneTwister(1), latent_delay)

md"""
A comprehension batches many draws from the latent form. Keeping the observed
component recovers the observed-delay marginal, which is why the latent `cdf`
and the marginal node's `cdf` are the same function.
"""

rng = MersenneTwister(2)

many_paths = [rand(rng, latent_delay) for _ in 1:500]

observed_times = [path.observed for path in many_paths]

md"""
## When to prefer the marginal or the latent form

Prefer the marginal form by default. It carries no per-record latents, so it is
cheaper and lower-dimensional at scale.
Prefer the latent form when you need the event times themselves, for example to
forward-simulate full event paths with `rand`, or for small-count or heavily
censored problems where the extra latent dimensions cost little.

## Summary

- The marginal `primary_censored` form and its `latent` wrapper are one model
  that share parameters.
- The latent form carries the full Distributions interface; its observed `cdf`,
  `pdf` and single-value `logpdf` delegate to the marginal node, so the two
  forms agree analytically.
- The same holds for a `double_interval_censored` delay, so the censoring
  wrappers work in both forms.
- `rand(latent(d))` forward-simulates full `(primary, observed)` event paths,
  Turing-free.
"""
