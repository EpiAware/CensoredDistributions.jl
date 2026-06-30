# [The LogDensityProblems inference layer](@id logdensity-layer)

A composed distribution can be fitted two ways.
This page explains the second one and how it relates to the Turing path.

## Two routes to the same posterior

A composed distribution carries everything a fit needs, none of it tied to a
particular probabilistic programming language.
[`params_table`](@ref) inventories the free parameters, [`build_priors`](@ref)
supplies support-derived default priors, [`update`](@ref) rebuilds the
distribution from a parameter `NamedTuple`, and [`logpdf`](@ref CensoredDistributions.logpdf) scores a record.
Both inference routes are thin front-ends onto that shared core.

The **Turing route** wraps the core in a DynamicPPL submodel.
[`composed_parameters_model`](@ref)`(template, priors)` samples each prior,
rebuilds the structure with [`compose`](@ref), and is `to_submodel`-able, so a
model scores records with `@addlogprob! logpdf(d, y)`.
This route lives in the `DynamicPPLExt` extension and gives the full Turing
machinery: sampler choice, chain objects, and named, groupable parameters.

The **LogDensityProblems route** wraps the same core in a flat-vector
log-density with no DynamicPPL dependency.
[`as_logdensity`](@ref CensoredDistributions.as_logdensity)`(dist, priors, data)` assembles a
[`ComposedLogDensity`](@ref CensoredDistributions.ComposedLogDensity) spec, and the `LogDensityProblemsExt` extension
turns it into a standard `LogDensityProblems` problem over the unconstrained
parameter vector.
That problem samples directly with AdvancedHMC, DynamicHMC, or Pathfinder, and
differentiates through the existing AD backends via `LogDensityProblemsAD`, all
without loading Turing.

## Role: orthogonal, not a replacement

The LogDensityProblems layer sits **orthogonal** to the DynamicPPL model.
It does not call into, wrap, or replace [`composed_parameters_model`](@ref); the
two never interact at runtime.
They are alternative entry points onto one scoring core, so they evaluate the
same log-posterior and a draw from either maps back to the same named,
constrained parameters through the [`params_table`](@ref) row order.
The default likelihood sums `logpdf(d, record)` over the data, the same
contribution the DynamicPPL path adds per record with `@addlogprob!`.

Use the Turing route when a fit belongs inside a larger Turing model or wants the
chain ecosystem.
Use the LogDensityProblems route to sample a composed model on its own with a
non-Turing sampler, or to hand the log-density to any tool that speaks the
`LogDensityProblems` interface.

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
