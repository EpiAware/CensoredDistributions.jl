# ============================================================================
# Wrapping a composer with an external censoring wrapper
# ============================================================================
#
# This layer defines what it MEANS to apply a censoring wrapper
# (`primary_censored` / `interval_censored` / `double_interval_censored`) ON
# TOP of a composer. It is the EXTERNAL direction: combine first, then censor,
# the dual of the internal direction, which specialises a composer whose
# internal nodes are already censored.
#
# A censoring wrapper observes one scalar quantity. Each composer exposes its
# observed scalar through `observed_distribution`:
#   - `Convolved`  -> already the observed sum (itself);
#   - `Competing`  -> already the marginal time-to-resolution (itself);
#   - `Sequential` -> the total elapsed time origin -> terminal event, i.e. the
#     convolution of the chain steps.
# A `Parallel` has several independent endpoints and so NO single observed
# scalar; censoring a `Parallel` DISTRIBUTES the wrapper into every branch,
# returning a `Parallel` of censored branches (each branch endpoint censored on
# its own).
#
# `Convolved` and `Competing` are univariate, so the existing
# `UnivariateDistribution` wrapper methods already accept them; this file adds
# only the `Sequential` (collapse-then-wrap) and `Parallel` (distribute) cases.
# Dispatch on the composer type — no runtime predicate, no new hierarchy.

@doc "

The univariate scalar a censoring wrapper observes for a composer.

A censoring wrapper observes one quantity, so wrapping a composer first lowers
it to that quantity:

- a [`Convolved`](@ref) or [`Competing`](@ref) is already univariate (the
  observed sum, resp. the marginal time-to-resolution) and is returned
  unchanged;
- a [`Sequential`](@ref) chain's observed quantity is the total elapsed time
  from origin to the terminal event, the convolution of its steps, returned as
  a [`Convolved`](@ref).

A [`Parallel`](@ref) has several independent endpoints and so no single observed
scalar; it is not lowered here (censoring a `Parallel` distributes the wrapper
into each branch instead).

# Examples
```@example
using CensoredDistributions, Distributions

seq = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
observed_distribution(seq)
```

# See also
- [`convolve_distributions`](@ref): the chain-step convolution
"
observed_distribution(d::UnivariateDistribution) = d

function observed_distribution(d::Sequential)
    leaves = _observed_leaves(d.components)
    return length(leaves) == 1 ? only(leaves) :
           convolve_distributions(leaves)
end

# Flatten a composer's components to the univariate leaves whose sum is the
# chain's terminal time. A nested `Sequential` contributes its own steps; a
# nested `Parallel` has no single terminal time, so a chain step that is itself
# a `Parallel` cannot be collapsed and is rejected with a clear message.
function _observed_leaves(components::Tuple)
    leaves = UnivariateDistribution[]
    for c in components
        _append_observed_leaves!(leaves, c)
    end
    return leaves
end

_append_observed_leaves!(leaves, c::UnivariateDistribution) = push!(leaves, c)
function _append_observed_leaves!(leaves, c::Sequential)
    for child in c.components
        _append_observed_leaves!(leaves, child)
    end
    return leaves
end
function _append_observed_leaves!(::Any, ::Parallel)
    throw(ArgumentError(
        "cannot collapse a Sequential chain whose step is a Parallel to a " *
        "single observed time; censor the Parallel's branches instead"))
end

# ---------------------------------------------------------------------------
# Sequential: collapse to the observed total, then censor.
# ---------------------------------------------------------------------------

@doc "

Primary-event-censor the total elapsed time of a [`Sequential`](@ref) chain.

The chain is first collapsed to its observed quantity (the convolution of the
steps via [`observed_distribution`](@ref)), then primary-censored. This is the
combine-first-then-censor direction.

See also: [`primary_censored`](@ref), [`observed_distribution`](@ref)
"
function primary_censored(d::Sequential, primary_event::UnivariateDistribution;
        kwargs...)
    return primary_censored(observed_distribution(d), primary_event; kwargs...)
end

function primary_censored(d::Sequential; kwargs...)
    return primary_censored(observed_distribution(d); kwargs...)
end

@doc "

Interval-censor (day-discretise) the total elapsed time of a
[`Sequential`](@ref) chain.

The chain is collapsed to its observed total (see
[`observed_distribution`](@ref)) before the interval boundaries are applied.

See also: [`interval_censored`](@ref)
"
function interval_censored(d::Sequential, interval)
    return interval_censored(observed_distribution(d), interval)
end

@doc "

Apply the full primary/truncation/interval pipeline to the total elapsed time
of a [`Sequential`](@ref) chain.

The chain is collapsed to its observed total (see
[`observed_distribution`](@ref)) before the pipeline runs.

See also: [`double_interval_censored`](@ref)
"
# Explicit keyword signature (NOT `; kwargs...`): forwarding a `kwargs...`
# splat lowers to a dynamic `Core.kwcall`, which Enzyme cannot specialise. It
# falls to `runtime_generic_augfwd` and then fails to allocate a shadow for the
# freshly-built `Convolved` observed total (`EnzymeNoShadowError`, #444). Naming
# the keywords keeps the inner call statically dispatched, so Enzyme (and every
# other backend) differentiates the `double_interval_censored(Sequential)` path.
# `method` selects the primary-censoring solver (the migrated replacement for
# the deprecated `force_numeric` flag); `force_numeric` is still forwarded so the
# deprecation path keeps working through the composer wrapper.
function double_interval_censored(
        d::Sequential;
        primary_event::UnivariateDistribution = Uniform(0, 1),
        lower::Union{Real, Nothing} = nothing,
        upper::Union{Real, Nothing} = nothing,
        interval::Union{Real, Nothing} = nothing,
        method::Union{AbstractSolverMethod, Nothing} = nothing,
        force_numeric = nothing)
    return double_interval_censored(
        observed_distribution(d);
        primary_event = primary_event, lower = lower, upper = upper,
        interval = interval, method = method, force_numeric = force_numeric)
end

# ---------------------------------------------------------------------------
# Parallel: distribute the wrapper into every branch.
# ---------------------------------------------------------------------------

@doc "

Primary-event-censor every branch of a [`Parallel`](@ref) independently.

A `Parallel` has several independent endpoints, so censoring distributes into
each branch, returning a `Parallel` of primary-censored branches.

See also: [`primary_censored`](@ref)
"
function primary_censored(d::Parallel, primary_event::UnivariateDistribution;
        kwargs...)
    return Parallel(
        map(b -> primary_censored(b, primary_event; kwargs...), d.components),
        d.names)
end

function primary_censored(d::Parallel; kwargs...)
    return Parallel(
        map(b -> primary_censored(b; kwargs...), d.components), d.names)
end

@doc "

Interval-censor every branch of a [`Parallel`](@ref) independently.

Distributes the interval boundaries into each branch, returning a `Parallel` of
interval-censored branches.

See also: [`interval_censored`](@ref)
"
function interval_censored(d::Parallel, interval)
    return Parallel(
        map(b -> interval_censored(b, interval), d.components), d.names)
end

@doc "

Apply the full primary/truncation/interval pipeline to every branch of a
[`Parallel`](@ref) independently.

Distributes the pipeline into each branch, returning a `Parallel` of censored
branches.

See also: [`double_interval_censored`](@ref)
"
function double_interval_censored(d::Parallel; kwargs...)
    return Parallel(
        map(b -> double_interval_censored(b; kwargs...), d.components), d.names)
end
