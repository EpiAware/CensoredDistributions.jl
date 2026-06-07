@doc "

Build a DynamicPPL submodel for a primary event censored distribution.

The submodel is `to_submodel`-able and dispatches on the type of `d`:

- a marginal [`primary_censored`](@ref) `d` scores the marginal log-density via
  `~`, with the primary event time integrated out inside `logpdf` (optionally
  scaled by a multiplicity `weight`);
- a [`latent`](@ref)-wrapped `d` (a [`Latent`](@ref) node) declares the primary
  event time as a latent variable inside the model (`p ~ get_primary_event(d)`)
  and scores the conditional of the observed time given that sampled `p` with
  [`primary_conditional_logpdf`](@ref). The user never passes `p` in; it is
  sampled inside the submodel.

For a coupled latent origin shared across records (for example a source's onset
feeding an offspring's infection), the caller samples the shared primary in their
own model and scores each record with `primary_conditional_logpdf(d, p_shared, y)`
directly, rather than wrapping with `latent`.

This function has no methods until `DynamicPPL` (or `Turing`) is loaded; the
methods live in the package extension so the core stays free of `DynamicPPL`.

# Arguments
- `d`: A [`primary_censored`](@ref) distribution, or a [`latent`](@ref)-wrapped
  one for the latent flow.
- `y`: The observed delay.

# Keyword Arguments
- `weight`: Multiplicity weight applied to the marginal `logpdf(d, y)`. `nothing`
  (the default) leaves the contribution unweighted.

# Examples
```@example
using CensoredDistributions, Distributions, DynamicPPL

d = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))

@model demo(d, y) = obs ~ to_submodel(primary_censored_model(d, y))

# The marginal submodel scores the same log-density as the distribution.
only(logjoint(demo(d, 2.0), (;))), logpdf(d, 2.0)
```

# See also
- [`latent`](@ref), [`primary_conditional_logpdf`](@ref), [`get_primary_event`](@ref)
- [`interval_censored_model`](@ref), [`double_interval_censored_model`](@ref)
"
function primary_censored_model end

@doc "

Build a DynamicPPL submodel for an interval censored distribution.

The submodel is `to_submodel`-able and scores `y` against the interval censored
distribution `d`, contributing `logpdf(d, y)` (optionally scaled by a
multiplicity `weight`). Interval censoring is always marginal, so there is no
latent path.

This function has no methods until `DynamicPPL` (or `Turing`) is loaded; the
methods live in the package extension so the core stays free of `DynamicPPL`.

# Arguments
- `d`: An [`interval_censored`](@ref) distribution.
- `y`: The observed (interval) value.

# Keyword Arguments
- `weight`: Multiplicity weight applied to `logpdf(d, y)`. `nothing` (the
  default) leaves the contribution unweighted.

# Examples
```@example
using CensoredDistributions, Distributions, DynamicPPL

d = interval_censored(LogNormal(1.5, 0.75), 1.0)

@model demo(d, y) = obs ~ to_submodel(interval_censored_model(d, y))

only(logjoint(demo(d, 2.0), (;))), logpdf(d, 2.0)
```

# See also
- [`primary_censored_model`](@ref), [`double_interval_censored_model`](@ref)
"
function interval_censored_model end

@doc "

Build a DynamicPPL submodel for a double interval censored distribution.

The submodel is `to_submodel`-able and scores `y` against the composed
distribution returned by [`double_interval_censored`](@ref) (primary censoring,
optional right truncation, and optional secondary interval censoring),
contributing `logpdf(d, y)` (optionally scaled by a multiplicity `weight`). The
whole pipeline is marginal, so there is no latent path.

This function has no methods until `DynamicPPL` (or `Turing`) is loaded; the
methods live in the package extension so the core stays free of `DynamicPPL`.

# Arguments
- `d`: A [`double_interval_censored`](@ref) distribution.
- `y`: The observed value.

# Keyword Arguments
- `weight`: Multiplicity weight applied to `logpdf(d, y)`. `nothing` (the
  default) leaves the contribution unweighted.

# Examples
```@example
using CensoredDistributions, Distributions, DynamicPPL

d = double_interval_censored(LogNormal(1.5, 0.75); upper = 10, interval = 1)

@model demo(d, y) = obs ~ to_submodel(double_interval_censored_model(d, y))

only(logjoint(demo(d, 3.0), (;))), logpdf(d, 3.0)
```

# See also
- [`primary_censored_model`](@ref), [`interval_censored_model`](@ref)
"
function double_interval_censored_model end

@doc raw"

Sample event-based draws from a marginal-fit posterior.

Fit a model in its efficient MARGINAL form (the primary event integrated out, no
extra latent dimensions), then call `predict_events(chain, model)` to obtain
event-based draws — the full internal event times — by running the LATENT form of
the same model over the fitted posterior. This works because the marginal and
latent forms are one family sharing the same parameters together with the
marginal-equals-latent equivalence (#301), so the marginal-fit posterior drops
straight into the latent form. Internally this is a thin convenience over
`Turing.predict`: the latent `model` is executed conditioned on each parameter
draw in `chain`, sampling the event variables the marginal chain does not carry.

`model` is the LATENT form of the fitted model, built with the same parameter
names as the marginal model that produced `chain` and with each event declared as
a `~` latent (for example via a [`latent`](@ref)-wrapped node and
[`primary_censored_model`](@ref), or with [`PrimaryConditional`](@ref) directly).
The two flavours from #350 are selected by the data the latent `model` carries,
dispatched by what you pass rather than by a mode flag:

- new-record posterior predictive: build `model` with `missing` event
  observations, so `predict_events` samples the complete internal event path per
  posterior draw (`[primary, observed, ...]`).
- observed-record latent recovery: build `model` with the observed event times
  supplied (the observed-but-censored events fixed), so `predict_events`
  re-samples only the integrated-out latents (the primary event time, and any
  unobserved intermediate events) conditioned on the observed data and the
  posterior parameters.

This function has no methods until `Turing` (or `DynamicPPL`) is loaded; the
method lives in the package extension so the core stays free of `DynamicPPL`.

# Arguments
- `chain`: An `MCMCChains.Chains` from fitting the MARGINAL model.
- `model`: The LATENT form of the same model (see above), carrying either
  `missing` events (new-record prediction) or the observed events (latent
  recovery).

# Keyword Arguments
- `rng`: Random number generator for the predictive sampling.
- `include_all`: Passed through to `Turing.predict`. `false` (the default)
  returns only the newly sampled event variables; `true` also keeps the
  parameters from `chain`.

# Examples
```julia
using CensoredDistributions, Distributions, Turing

# Fit the efficient marginal form.
@model function fit(y)
    mu ~ Normal(1.5, 1.0)
    d = primary_censored(LogNormal(mu, 0.5), Uniform(0, 1))
    for i in eachindex(y)
        y[i] ~ to_submodel(prefix(primary_censored_model(d, y[i]),
            Symbol(:rec, i)), false)
    end
end
chain = sample(fit(data), NUTS(), 100)

# Recover the integrated-out primary event times for the observed records.
@model function latent_recovery(y)
    mu ~ Normal(1.5, 1.0)
    d = latent(primary_censored(LogNormal(mu, 0.5), Uniform(0, 1)))
    for i in eachindex(y)
        x ~ to_submodel(prefix(primary_censored_model(d, y[i]),
            Symbol(:rec, i)), false)
    end
end
events = predict_events(chain, latent_recovery(data))
```

# See also
- [`latent`](@ref), [`primary_censored_model`](@ref),
  [`PrimaryConditional`](@ref)
"
function predict_events end
