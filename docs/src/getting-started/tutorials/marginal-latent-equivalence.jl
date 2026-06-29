md"""
# [Marginal and latent forms of a censored delay](@id marginal-latent-equivalence)

## Introduction

### What are we going to do in this tutorial

A primary-censored delay has two forms that share the same parameters and the
same observed-delay distribution.

1. The **marginal** form is the plain `primary_censored` distribution. The
   primary event time is integrated out by the analytic closed form inside
   `logpdf` and `cdf`, so the distribution is univariate over the observed time
   alone. This is the target.
2. The **latent** form, `latent(d)`, keeps the primary event time as an explicit
   sampled variable. A draw is a labelled `(primary, observed)` record, and the
   joint `logpdf([primary, observed])` scores the primary prior plus the
   conditional of the observed time given that primary. The latent form has NO
   scalar observed density of its own: the observed marginal is the marginal
   form's job, recovered with `marginal(d)`.

The two forms are equivalent **in expectation**: averaging (integrating) the
latent joint over the primary recovers the analytic marginal. We demonstrate
that by forward simulation here, show it for a `double_interval_censored` delay
too, and forward-simulate event paths with `rand(latent(d))`, all without
Turing.

### What might I need to know before starting

This tutorial builds on
[Getting Started with CensoredDistributions.jl](@ref getting-started).
It uses the [`latent`](@ref) wrapper and its inverse [`marginal`](@ref).

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
## The latent form has no scalar observed density

The latent form is a joint over `[primary, observed]`. Scoring a scalar observed
value would just re-integrate the primary and reproduce the marginal default, so
the scalar density methods are deliberately undefined and throw. Recover the
observed marginal with `marginal(d)`.
"""

scalar_density_errors = try
    logpdf(latent_delay, 2.5)
    false
catch err
    err isa ArgumentError
end

md"""
The joint density over the scored `[primary, observed]` vector is what the latent
form provides, the primary prior plus the conditional of the observed given the
primary.
"""

joint_logpdf = logpdf(latent_delay, [0.3, 2.5])

md"""
## Equivalence in expectation with the analytic marginal

The two forms are equivalent in expectation: drawing many latent records and
keeping the observed component recovers the analytic marginal observed-delay
distribution. We compare the empirical observed `cdf` from the latent form
against the analytic `cdf` of the marginal form.
"""

rng = MersenneTwister(20240610)

observed_draws = [rand(rng, latent_delay).observed for _ in 1:200_000]

eval_points = [1.0, 3.0, 6.0, 10.0]

cdf_comparison = [(x = x,
                      marginal = cdf(marginal_delay, x),
                      empirical = count(<=(x), observed_draws) /
                                  length(observed_draws))
                  for x in eval_points]

md"""
The analytic `marginal` column and the `empirical` latent column agree to Monte
Carlo error. The largest gap across the points is small.
"""

cdf_gap = maximum(abs(r.marginal - r.empirical) for r in cdf_comparison)

md"""
The same equivalence holds for the density. An explicit trapezoidal integration
of the latent joint over the primary window reproduces the analytic marginal
`pdf`, the equivalence written as an integral rather than a sample average.
"""

function integrate_joint_pdf(ld, y; n = 200_000)
    lo, hi = minimum(get_primary_event(ld)), maximum(get_primary_event(ld))
    ps = range(lo, hi; length = n)
    vals = map(p -> exp(logpdf(ld, [p, y])), ps)
    return sum((vals[1:(end - 1)] .+ vals[2:end]) ./ 2) * step(ps)
end

pdf_comparison = [(x = x,
                      marginal = pdf(marginal_delay, x),
                      integrated = integrate_joint_pdf(latent_delay, x))
                  for x in eval_points]

pdf_gap = maximum(abs(r.marginal - r.integrated) for r in pdf_comparison)

md"""
## The same equivalence holds for a double-interval-censored delay

The censoring wrappers work for both forms. We build a
`double_interval_censored` delay and wrap it in `latent`. The latent form samples
the primary, so its conditional scores the bare continuous delay, and integrating
the joint over the primary reproduces the analytic CONTINUOUS primary-censored
marginal. The analytic target for the latent double-interval-censored leaf is
therefore `primary_censored(delay, pe)`.
"""

pe = Uniform(0, 1)

target = primary_censored(delay, pe)

marginal_dic = double_interval_censored(
    delay; primary_event = pe, upper = 10, interval = 1)

latent_dic = latent(marginal_dic)

dic_comparison = [(x = x,
                      analytic_target = pdf(target, x),
                      integrated = integrate_joint_pdf(latent_dic, x))
                  for x in [1.0, 3.0, 6.0, 9.0]]

md"""
The integrated latent joint of the double-interval-censored leaf matches the
analytic continuous primary-censored target.
"""

dic_pdf_gap = maximum(
    abs(r.analytic_target - r.integrated) for r in dic_comparison)

md"""
## Forward-simulate event paths from the latent form

`rand(latent(d))` draws a full event path as a labelled
`(primary = ..., observed = ...)` record. The primary lies in the window and the
observed time exceeds it by a positive delay. No model or conditioning is needed.
"""

one_path = rand(MersenneTwister(1), latent_delay)

md"""
A comprehension batches many draws from the latent form. Keeping the observed
component recovers the observed-delay marginal, the same distribution the
analytic marginal `cdf` computes.
"""

many_paths = [rand(MersenneTwister(2), latent_delay) for _ in 1:500]

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
- The marginal form carries the analytic scalar `cdf`/`pdf`; the latent form has
  no scalar density, only the joint over `[primary, observed]`. Averaging or
  integrating that joint over the primary recovers the marginal: the two forms
  are equivalent in expectation.
- The same holds for a `double_interval_censored` delay against the continuous
  primary-censored target, so the censoring wrappers work in both forms.
- `rand(latent(d))` forward-simulates full `(primary, observed)` event paths,
  Turing-free.
"""
