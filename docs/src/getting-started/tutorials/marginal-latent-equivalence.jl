md"""
# [Marginal and latent forms of a censored delay](@id marginal-latent-equivalence)

## Introduction

### What are we going to do in this tutorial

A primary-censored delay has two equivalent forms that share the same
parameters.

1. The **marginal** form is the plain `primary_censored` distribution. The
   primary event time is integrated out inside `logpdf` and `cdf`, so the
   distribution is univariate over the observed time alone.
2. The **latent** form, `latent(d)`, keeps the primary event time as an explicit
   sampled dimension. A draw is a labelled `(primary, observed)` record, and
   `logpdf` scores the primary prior plus the conditional of the observed time
   given that primary.

These are one model scored two ways. Integrating the latent joint over the
primary window reproduces the marginal density exactly. We demonstrate that
equivalence here and show how `rand(latent(d))` forward-simulates full event
paths, all without Turing.

### What might I need to know before starting

This tutorial builds on
[Getting Started with CensoredDistributions.jl](@ref getting-started).
It uses the [`latent`](@ref) wrapper, its inverse [`marginal`](@ref), and
[`get_dist`](@ref) to reach the underlying delay.

## Packages used

We use Distributions for the delay distribution, Integrals for the quadrature in
the equivalence check, Random for reproducibility, and Statistics for the Monte
Carlo average.
"""

using CensoredDistributions
using Distributions
using Integrals
using Random
using Statistics

md"""
## Build the marginal and latent forms

We start from a primary-censored log-normal delay with a uniform primary event
window `[0, 1]`. The plain node is the marginal form; wrapping it in `latent`
selects the latent form. `marginal` is the inverse, recovering the wrapped node.
"""

delay = LogNormal(1.4, 0.5)

marginal_leaf = primary_censored(delay, Uniform(0, 1))

latent_leaf = latent(marginal_leaf)

@assert marginal(latent_leaf) === marginal_leaf

md"""
## Demonstrate the marginal and latent equivalence

We compare the cumulative distribution function three ways.

- The **target** is the distribution the censoring represents: a primary `p`
  drawn from `Uniform(0, 1)` plus the `LogNormal` delay. Its `cdf` is the
  integral `∫₀¹ cdf(delay, x - p) dp`, evaluated with adaptive quadrature.
- The **marginal** form scores the `primary_censored` leaf, integrating the
  primary out inside `cdf`.
- The **latent** form carries the primary as an explicit dimension. Its `cdf` is
  a Monte Carlo average of `cdf(delay, x - p)` over primaries drawn from the
  latent representation with `rand`.

All three are the same distribution, so their `cdf` curves coincide.
"""

function target_cdf(x)
    problem = IntegralProblem((p, _) -> cdf(delay, x - p), (0.0, 1.0))
    return solve(problem, QuadGKJL(); reltol = 1e-12, abstol = 1e-12).u
end

function latent_cdf(x; rng = MersenneTwister(7), draws = 200_000)
    paths = (rand(rng, latent_leaf) for _ in 1:draws)
    return mean(cdf(delay, x - path.primary) for path in paths)
end

eval_points = [1.0, 3.0, 6.0, 10.0];

three_way = [(x = x, target = target_cdf(x),
                 marginal = cdf(marginal_leaf, x), latent = latent_cdf(x))
             for x in eval_points];

md"""
The target, marginal and latent `cdf` agree across the evaluation points. The
target-versus-marginal gap is at numerical precision (both integrate the primary
out exactly); the latent column carries only Monte Carlo noise from its finite
draws.
"""

@assert maximum(abs(r.target - r.marginal) for r in three_way) < 1e-10

@assert maximum(abs(r.target - r.latent) for r in three_way) < 5e-3

three_way

md"""
The same equivalence holds for the density. Integrating the latent joint
`logpdf` over the primary window with a trapezoidal rule reproduces the marginal
`pdf`.
"""

function integrate_primary(y; n = 200_000)
    ps = range(0.0, 1.0; length = n)
    vals = map(p -> exp(logpdf(latent_leaf, [p, y])), ps)
    return sum((vals[1:(end - 1)] .+ vals[2:end]) ./ 2) * step(ps)
end

for y in [1.0, 2.5, 4.0]
    @assert isapprox(integrate_primary(y), pdf(marginal_leaf, y); rtol = 1e-3)
end

md"""
## Forward-simulate event paths from the latent form

`rand(latent(d))` draws a full event path as a labelled
`(primary = ..., observed = ...)` record. The primary lies in the window and the
observed time exceeds it by a positive delay. No model or conditioning is
needed.
"""

one_path = rand(MersenneTwister(1), latent_leaf)

@assert 0 <= one_path.primary <= 1

@assert one_path.observed > one_path.primary

md"""
A comprehension batches many draws from the latent leaf.
"""

rng = MersenneTwister(2);

many_paths = [rand(rng, latent_leaf) for _ in 1:500];

@assert all(path -> path.observed > path.primary, many_paths)

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
- Integrating the latent joint over the primary window reproduces the marginal
  density and distribution function exactly.
- `rand(latent(d))` forward-simulates full `(primary, observed)` event paths,
  Turing-free.
"""
