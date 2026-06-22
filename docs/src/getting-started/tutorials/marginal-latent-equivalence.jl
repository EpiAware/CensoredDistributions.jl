md"""
# [Marginal and latent forms of a censored delay](@id marginal-latent-equivalence)

## Introduction

### What are we going to do in this tutorial

A primary-censored delay has two forms that share the same parameters and the
same observed-delay distribution, computed two genuinely different ways.

1. The **marginal** form is the plain `primary_censored` distribution. The
   primary event time is integrated out by the analytic closed form inside
   `logpdf` and `cdf`, so the distribution is univariate over the observed time
   alone. This is the target.
2. The **latent** form, `latent(d)`, keeps the primary event time as an explicit
   augmented variable. A draw is a labelled `(primary, observed)` record, and
   the joint `logpdf([primary, observed])` scores the primary prior plus the
   conditional of the observed time given that primary. Its observed `cdf` and
   `pdf` are computed by NUMERICALLY INTEGRATING that joint over the primary,
   the augmented-data integral, using the same Gauss-Legendre quadrature the
   package uses elsewhere.

These two computations are different: one is an analytic closed form, the other
a numeric integral over the augmented primary. They agree to quadrature
tolerance, which validates the latent formulation. We demonstrate that here,
show it for a `double_interval_censored` delay too, and forward-simulate event
paths with `rand(latent(d))`, all without Turing.

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
## Latent integration agrees with the analytic marginal

The analytic marginal `cdf` integrates the primary out in closed form. The
latent `cdf` instead evaluates the augmented-data integral
`∫ pdf(prior, p) · cdf(delay, x - p) dp` by Gauss-Legendre quadrature over the
primary window. These are two different computations of the same observed-delay
distribution, so they should agree to quadrature tolerance, NOT to machine
precision.

We compare them across a set of evaluation points.
"""

eval_points = [1.0, 3.0, 6.0, 10.0]

cdf_comparison = [(x = x,
                      marginal = cdf(marginal_delay, x),
                      latent = cdf(latent_delay, x))
                  for x in eval_points]

md"""
The analytic `marginal` column and the numeric `latent` column agree to about
quadrature tolerance. The largest gap across the points is small but non-zero,
because one is a closed form and the other a numeric integral.
"""

cdf_gap = maximum(abs(r.marginal - r.latent) for r in cdf_comparison)

md"""
The density agrees the same way. The latent `pdf` is the augmented-data integral
`∫ pdf(prior, p) · pdf(delay, x - p) dp`, again matching the analytic marginal
`pdf` to quadrature tolerance.
"""

pdf_comparison = [(x = x,
                      marginal = pdf(marginal_delay, x),
                      latent = pdf(latent_delay, x))
                  for x in eval_points]

pdf_gap = maximum(abs(r.marginal - r.latent) for r in pdf_comparison)

md"""
## The same validation holds for a double-interval-censored delay

The censoring wrappers work for both forms. We build a
`double_interval_censored` delay and wrap it in `latent`. The latent form
samples the primary, so its conditional scores the bare continuous delay, and
integrating the augmented joint over the primary reproduces the analytic
CONTINUOUS primary-censored marginal. The analytic target for the latent
double-interval-censored leaf is therefore `primary_censored(delay, pe)`.
"""

pe = Uniform(0, 1)

target = primary_censored(delay, pe)

marginal_dic = double_interval_censored(
    delay; primary_event = pe, upper = 10, interval = 1)

latent_dic = latent(marginal_dic)

dic_comparison = [(x = x,
                      analytic_target = cdf(target, x),
                      latent = cdf(latent_dic, x))
                  for x in [1.0, 3.0, 6.0, 9.0]]

md"""
The numeric latent `cdf` of the double-interval-censored leaf matches the
analytic continuous primary-censored target to quadrature tolerance.
"""

dic_cdf_gap = maximum(
    abs(r.analytic_target - r.latent) for r in dic_comparison)

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
component recovers the observed-delay marginal, the same distribution the latent
`cdf` integral computes.
"""

rng = MersenneTwister(2)

many_paths = [rand(rng, latent_delay) for _ in 1:500]

observed_times = [path.observed for path in many_paths]

md"""
## When to prefer the marginal or the latent form

Prefer the marginal form by default. It carries no per-record latents and uses
the analytic closed form, so it is cheaper and lower-dimensional at scale.
Prefer the latent form when you need the event times themselves, for example to
forward-simulate full event paths with `rand`, or for small-count or heavily
censored problems where the extra latent dimensions cost little.

## Summary

- The marginal `primary_censored` form and its `latent` wrapper are one model
  that share parameters and the same observed-delay distribution.
- The marginal `cdf`/`pdf` use the analytic closed form; the latent `cdf`/`pdf`
  numerically integrate the augmented-data joint over the primary. They agree to
  quadrature tolerance, which validates the latent formulation.
- The same holds for a `double_interval_censored` delay against the continuous
  primary-censored target, so the censoring wrappers work in both forms.
- `rand(latent(d))` forward-simulates full `(primary, observed)` event paths,
  Turing-free.
"""
