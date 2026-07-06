# [The LogDensityProblems inference layer](@id logdensity-layer)

`LogDensityProblems` is the primary inference interface for a composed
distribution.
This page explains that substrate and how the Turing bridge consumes it.

## The canonical substrate

A composed distribution carries everything a fit needs, none of it tied to a
particular probabilistic programming language.
[`params_table`](@ref) inventories the free parameters, [`build_priors`](@ref)
supplies support-derived default priors, [`update`](@ref) rebuilds the
distribution from a parameter `NamedTuple`, and [`logpdf`](@ref CensoredDistributions.logpdf) scores a record.

The **LogDensityProblems route** is the canonical, PPL-neutral front-end onto
that core.
[`as_logdensity`](@ref CensoredDistributions.as_logdensity)`(dist, priors, data)` assembles a
[`ComposedLogDensity`](@ref CensoredDistributions.ComposedLogDensity) spec, and the `LogDensityProblemsExt` extension
turns it into a standard `LogDensityProblems` problem over the unconstrained
parameter vector.
That problem samples directly with AdvancedHMC, DynamicHMC, or Pathfinder, and
differentiates through the existing AD backends via `LogDensityProblemsAD`, all
without loading Turing.
The same spec is also generative.
`rand(prob)` draws the parameters from their priors and forward-simulates one
record, the prior predictive counterpart of scoring the data.

## Turing is a consumer, not a second route

The DynamicPPL/Turing extension is one consumer of the same spec, kept for the
one property a raw `LogDensityProblems` problem loses when handed to Turing.
That property is a chain where **every parameter is a named site**, not one
opaque parameter block.
[`as_turing`](@ref CensoredDistributions.as_turing)`(prob)` is the bridge: it wraps
the spec as a DynamicPPL model whose log-joint equals
[`logdensity`](@ref CensoredDistributions.logdensity)`(prob, x)` on the constrained
scale, so the two routes score the same target and a draw from either maps back
to the same named, constrained parameters through the [`params_table`](@ref) row
order.
Under the hood it samples the free parameters through
[`composed_parameters_model`](@ref)`(template, priors)`, the named-mode submodel
that places each prior as its own `~` site, and scores the data with the spec's
own `loglik` (which defaults to summing `logpdf(d, record)`, the contribution the
LogDensityProblems path also sums).

Reach for the LogDensityProblems route by default, or to hand the log-density to
any tool that speaks the `LogDensityProblems` interface.
Reach for [`as_turing`](@ref CensoredDistributions.as_turing) when a fit belongs
inside a larger Turing program or wants named chain sites and the chain
ecosystem.

## What the layer adds

Three of the four `LogDensityProblems` ingredients already exist Turing-free:
the parameter `dimension` ([`flat_dimension`](@ref CensoredDistributions.flat_dimension)), distribution
reconstruction ([`update`](@ref)), and the prior and data log-density.
The one genuinely new core piece is the flat-vector codec
([`flatten`](@ref CensoredDistributions.flatten) /
[`unflatten`](@ref CensoredDistributions.unflatten)), which bridges the sampler's flat
vector and the nested parameter `NamedTuple` the rest of the stack consumes.
The codec and the [`ComposedLogDensity`](@ref CensoredDistributions.ComposedLogDensity) spec are core (in
`src/composers/logdensity.jl`); the library-specific glue stays in weakdep
extensions:

- `LogDensityProblemsExt`: the unconstrained `LogDensityProblems` problem;
- `BijectorsExt`: the prior-driven constrained ↔ unconstrained transform;
- `DensityInterfaceExt`: marks the spec a `DensityInterface` density.

## A Turing-free fit

```julia
using CensoredDistributions, Distributions
using LogDensityProblems, LogDensityProblemsAD, Bijectors
using ADTypes: AutoForwardDiff

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
data = [[0.5, 2.0], [1.0, 3.0]]

prob = CensoredDistributions.as_logdensity(tree, build_priors(tree), data)
LogDensityProblems.dimension(prob)           # the flat parameter count
∇prob = ADgradient(AutoForwardDiff(), prob)  # ready for AdvancedHMC etc.
```

The public-but-not-exported layer is reached by its qualified name
(`CensoredDistributions.as_logdensity`, ...).
See [`as_logdensity`](@ref CensoredDistributions.as_logdensity), [`logdensity`](@ref CensoredDistributions.logdensity), and
[`ComposedLogDensity`](@ref CensoredDistributions.ComposedLogDensity) for the per-function reference.

## Driving the Turing route from the spec: `as_turing`

The same [`ComposedLogDensity`](@ref CensoredDistributions.ComposedLogDensity)
spec can also be turned into a Turing model, so one object drives both routes.
[`as_turing`](@ref CensoredDistributions.as_turing)`(prob)` wraps the spec as a
DynamicPPL model: it samples the free parameters through
[`composed_parameters_model`](@ref) (under the `d` prefix, so the chain records
`d.<edge>.<param>` and the default [`chain_to_params`](@ref) /
[`param_draws`](@ref) read applies) and scores the data with the spec's own
`loglik`.

```julia
using CensoredDistributions, Distributions, DynamicPPL, Turing

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
data = [[0.5, 2.0], [1.0, 3.0]]
prob = CensoredDistributions.as_logdensity(tree, build_priors(tree), data)

chain = sample(CensoredDistributions.as_turing(prob), NUTS(), 1000)
param_draws(tree, chain)   # every parameter is a named, recorded site
```

Because the model reuses the spec's `dist` / `priors` / `data` / `loglik`, its
log-joint equals [`logdensity`](@ref CensoredDistributions.logdensity)`(prob, x)`
on the constrained scale — the two routes score the same target. `as_turing` is
the adaptor that keeps every parameter a **named** chain site, which a raw
`LogDensityProblems` problem handed to Turing (via `externalsampler`) does not;
a parameter fixed in `priors` is substituted and never sampled, matching the
`LogDensityProblems` path. The convenience forms `as_turing(dist, priors, data)`
and `as_turing(dist, data)` mirror [`as_logdensity`](@ref CensoredDistributions.as_logdensity)'s
signatures.
