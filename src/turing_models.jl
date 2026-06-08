@doc "

Build a DynamicPPL submodel for a primary event censored LEAF distribution.

This is a leaf building block. For a whole record (leaf or composed) call the
generic [`composed_distribution_model`](@ref) instead, which dispatches to the
right leaf model or recurses through a composer.

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

@doc "

Build a DynamicPPL submodel for ANY record distribution, leaf or composed.

This is the single generic entry point: it dispatches on the type of `d` and
either delegates to the matching leaf model or recurses through a composed
distribution.

- A leaf/univariate distribution routes to its leaf model:
  [`primary_censored`](@ref) (marginal or [`latent`](@ref)) to
  [`primary_censored_model`](@ref), an [`interval_censored`](@ref) to
  [`interval_censored_model`](@ref), and any other univariate (a
  [`double_interval_censored`](@ref) pipeline, a `Truncated`, a
  [`convolve_distributions`](@ref) result) to
  [`double_interval_censored_model`](@ref).
- A composed distribution ([`Sequential`](@ref), [`Parallel`](@ref),
  [`Competing`](@ref), or their [`latent`](@ref) wrappers) recurses through the
  composer structure, marginalising or conditioning per record per the row's
  missingness pattern, and turning the origin/shared primary on in the latent
  case.
- A [`Select`](@ref) data-selected disjunction reads the row's selector field
  (`row[d.selector]`, default `:kind`), whose value names the active
  alternative, and delegates to that selected alternative's own model. The
  selector field is stripped before delegating, so the alternative sees only its
  own events. This is the data-driven index-vs-sourced split (#356).

Unlike the leaf models, this entry never misnames the distribution: a
`Sequential` of double-censored edges is composed, not 'primary censored'. The
leaf models stay available as the correctly named building blocks and may be
called directly.

The observations come in as a `NamedTuple` `row` keyed by event name; a leaf
record may also be passed a bare observed value. `missing` fields drive the
per-record marginalise-vs-condition dispatch, and a reserved `weight`/`count`
field (or the `weight =` keyword) scales the likelihood.

This function has no methods until `DynamicPPL` (or `Turing`) is loaded; the
methods live in the package extension so the core stays free of `DynamicPPL`.

# Arguments
- `d`: Any record distribution: a leaf (univariate) or a composer.
- `row`: The record observations, a `NamedTuple` keyed by event name (or a bare
  value for a leaf).

# Keyword Arguments
- `weight`: Multiplicity weight scaling the record's likelihood. `nothing` (the
  default) leaves it unweighted; a reserved `weight`/`count` row field also
  applies.

# Examples
```@example
using CensoredDistributions, Distributions, DynamicPPL

seq = Sequential(
    primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
    primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))

@model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

row = (onset = 0.0, admit = 2.0, death = 5.0)
ev = Vector{Union{Missing, Float64}}([0.0, 2.0, 5.0])
only(logjoint(demo(seq, row), (;))), logpdf(seq, ev)
```

# See also
- [`primary_censored_model`](@ref), [`interval_censored_model`](@ref),
  [`double_interval_censored_model`](@ref): the leaf building blocks.
- [`latent`](@ref), [`predict_events`](@ref).
"
function composed_distribution_model end

@doc "

Build a DynamicPPL submodel that samples a composed distribution's parameters
from user priors and returns the reconstructed distribution (#353).

A `template` composed distribution (from [`compose`](@ref)) defines the parameter
inventory; [`params_table`](@ref) lists it. The user supplies `priors`, a nested
`NamedTuple` keyed exactly like [`params`](@ref)`(template)`: leaves keyed by
their parameter names (e.g. `:shape`/`:scale`, `:mu`/`:sigma`), composer nodes
keyed by their edge/event names, and a [`Competing`](@ref) node optionally
carrying a `branch_probs` entry. The returned submodel samples each prior and
rebuilds the SAME composer structure and names via [`compose`](@ref), so the
reconstructed distribution drops straight into the matching record submodel (for
example [`primary_censored_model`](@ref)) for the likelihood.

Sampled parameters are namespaced by their edge path through nested submodel
prefixing, so a multi-edge chain yields readable, groupable Turing chain names
like `onset_admit.shape` and `resolution.death.scale`.

This function has no methods until `DynamicPPL` (or `Turing`) is loaded; the
methods live in the package extension so the core stays free of `DynamicPPL`.

# Arguments
- `template`: a composed distribution from [`compose`](@ref) (or a bare leaf)
  that defines the parameter inventory and the structure to rebuild.
- `priors`: a nested `NamedTuple` of priors keyed like [`params`](@ref)`(template)`
  (see [`params_table`](@ref)). Every parameter in the inventory must have a prior
  and no extra keys are allowed; a [`Competing`](@ref) node's `branch_probs` may
  be omitted (the template's fixed probabilities are kept) or supplied as priors.

# Examples
```@example
using CensoredDistributions, Distributions, DynamicPPL

template = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
priors = (
    onset_admit = (shape = truncated(Normal(2, 0.5); lower = 0),
        scale = truncated(Normal(1, 0.3); lower = 0)),
    admit_death = (mu = Normal(0.5, 0.2),
        sigma = truncated(Normal(0.4, 0.1); lower = 0)))

@model function demo(template, priors)
    d ~ to_submodel(composed_parameters_model(template, priors))
    return d
end

reconstructed = demo(template, priors)()
event_names(reconstructed)
```

# See also
- [`primary_censored_model`](@ref): scores a record under the reconstructed
  distribution.
- [`params_table`](@ref), [`params`](@ref): the parameter inventory the priors are
  keyed against.
"
function composed_parameters_model end

@doc "

Read a fitted Turing chain into the nested NamedTuple that [`update`](@ref)
consumes.

`chain_to_params(template, chain)` walks the `template` composed distribution and
pulls each free parameter's value out of `chain` (sampled through
[`composed_parameters_model`](@ref)), returning a nested `NamedTuple` keyed like
[`params`](@ref)`(template)`. By default it returns posterior means; pass
`draw=i` for a single draw. The `prefix` keyword names the submodel variable the
parameters were sampled under (the `~`-bound name, default `:d`), matching the
edge-path-prefixed chain names like `d.onset_admit.shape`.

`update(template, chain_to_params(template, chain))` returns a ready-to-`rand`/
inspect distribution, replacing manual `chain[Prefixed(@varname(...))]`
reconstruction.

This function has no methods until `DynamicPPL` (or `Turing`) is loaded; the
method lives in the package extension so the core stays free of `DynamicPPL`.

# Arguments
- `template`: the composed distribution (from [`compose`](@ref)) that was the
  `composed_parameters_model` template.
- `chain`: the fitted chain to read parameter values from.

# Keyword Arguments
- `prefix`: the submodel variable name the parameters were sampled under
  (default `:d`).
- `draw`: a single iteration index to read, or `nothing` for posterior means
  (default `nothing`).

# See also
- [`update`](@ref): rebuild the distribution from the NamedTuple.
- [`composed_parameters_model`](@ref): the submodel that produced the chain.
"
function chain_to_params end
