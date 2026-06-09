@doc "

Simulate event-based draws (the full internal event times) from a distribution.

Two dispatch paths share one name:

- **raw distribution (Turing-free, this method):** `predict_events(d, ...)`
  forward-simulates complete event paths directly from a latent or composed
  distribution `d` via `rand`, with no `@model` needed. This is the
  forward-simulation / new-record posterior-predictive path. A single draw,
  `n` draws, or one draw per supplied parameter set (rebuilding the distribution
  per draw) are all supported. Lives in core because it is a structured wrapper
  over `rand` on the latent form.
- **fitted Turing model (extension):** `predict_events(chain, model)` recovers the
  observed records' integrated-out latent event times by running the latent form
  of a marginal-fit model over the posterior `chain`. That method needs
  `DynamicPPL`/`Turing` and lives in the package extension.

`d` should be in its latent representation so that `rand(d)` returns the full
event-time path rather than a single marginal delay — for a leaf, wrap with
[`latent`](@ref); `rand(latent(d))` returns `[primary, observed]`. For a composed
tree (nested [`Sequential`](@ref)/[`Parallel`](@ref), a [`Competing`](@ref)
outcome, or a [`Select`](@ref) top) `rand` walks the tree sharing the latent
origin, samples each Competing outcome and the selected branch, and returns a
NAMED event record keyed by [`tree_event_names`](@ref) (an unsampled Competing
outcome is `missing`), so a whole case-study path is one `predict_events` call.

# Arguments
- `d`: A distribution whose `rand` yields a full event-time path (for example a
  [`latent`](@ref)-wrapped node).
- `n` (optional): Number of independent draws. Without it, one draw is returned;
  with it, a `Vector` of `n` draws is returned.

# Keyword Arguments
- `rng`: Random number generator (defaults to the global RNG).

# Examples
```@example
using CensoredDistributions, Distributions, Random

ld = latent(primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1)))

# One event path `[primary, observed]`.
path = predict_events(ld; rng = MersenneTwister(1))

# Many event paths.
paths = predict_events(ld, 100; rng = MersenneTwister(1))
```

# See also
- [`latent`](@ref): the latent representation simulated here.
- [`predict_events`](@ref): the `(chain, model)` method recovers fitted records'
  latent times (in the `DynamicPPL` extension).
"
function predict_events(d::Distribution; rng::AbstractRNG = default_rng())
    return rand(rng, d)
end

function predict_events(
        d::Distribution, n::Integer; rng::AbstractRNG = default_rng())
    return [rand(rng, d) for _ in 1:n]
end

@doc "

Simulate event-based draws from a [`Select`](@ref) disjunction.

`kind` names the active alternative (the selector value); the draw is that
alternative's own event path. Routing the selection through the public
`rand(::Select; kind)` keeps simulation off the internal selection helper. With
`kind = nothing` an alternative is sampled uniformly (the no-data
forward-simulation path), matching `rand`.

# Examples
```@example
using CensoredDistributions, Distributions, Random

d = select(:index => primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)),
    :sourced => primary_censored(Gamma(4.0, 1.5), Uniform(0, 1)))
path = predict_events(d; kind = :index, rng = MersenneTwister(1))
```

See also: [`Select`](@ref), [`predict_events`](@ref).
"
function predict_events(d::Select; kind::Union{Symbol, Nothing} = nothing,
        rng::AbstractRNG = default_rng())
    return rand(rng, d; kind = kind)
end

function predict_events(d::Select, n::Integer;
        kind::Union{Symbol, Nothing} = nothing,
        rng::AbstractRNG = default_rng())
    return [rand(rng, d; kind = kind) for _ in 1:n]
end

@doc "

Simulate one event-based draw per supplied parameter set.

`build` maps a parameter set to a latent/composed distribution (for example one
posterior draw `(; mu, sigma)` to `latent(primary_censored(LogNormal(mu, sigma),
Uniform(0, 1)))`); `params` is any iterable of such parameter sets. The
distribution is rebuilt for each parameter set, and one event path is drawn from
each, returning a `Vector` of draws. This is the Turing-free way to push a set of
posterior parameter draws through the latent form to get forward-simulated event
paths.

# Arguments
- `build`: A function mapping one parameter set to a distribution.
- `params`: An iterable of parameter sets.

# Keyword Arguments
- `rng`: Random number generator (defaults to the global RNG).

# Examples
```@example
using CensoredDistributions, Distributions, Random

build(p) = latent(primary_censored(LogNormal(p.mu, p.sigma), Uniform(0, 1)))
draws = [(mu = 1.4, sigma = 0.5), (mu = 1.6, sigma = 0.4)]
paths = predict_events(build, draws; rng = MersenneTwister(1))
```

# See also
- [`predict_events`](@ref): the single-distribution and `(chain, model)`
  methods.
"
function predict_events(
        build::Base.Callable, params; rng::AbstractRNG = default_rng())
    return [rand(rng, build(p)) for p in params]
end
