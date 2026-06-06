@doc "

Build a DynamicPPL submodel for a primary event censored distribution.

The submodel is `to_submodel`-able and scores `y` against the marginal
distribution `d`, with the primary event time integrated out inside `logpdf`.
It dispatches on the type of `d`.

By default (`origin = nothing`) the primary event is marginalised: the submodel
contributes `logpdf(d, y)`, optionally scaled by a multiplicity `weight`. Supply
`origin` to use a caller-owned primary event time instead (the coupled case):
the submodel then scores the conditional delay `logpdf(get_dist(d), y - origin)`
and the caller declares the prior over the origin in their own model.

This function has no methods until `DynamicPPL` (or `Turing`) is loaded; the
methods live in the package extension so the core stays free of `DynamicPPL`.

# Arguments
- `d`: A [`primary_censored`](@ref) distribution.
- `y`: The observed delay.

# Keyword Arguments
- `weight`: Multiplicity weight applied to `logpdf(d, y)`. `nothing` (the
  default) leaves the contribution unweighted.
- `origin`: A caller-supplied primary event time for the coupled case. `nothing`
  (the default) marginalises the primary event.

# Examples
```jldoctest
julia> using CensoredDistributions, Distributions, DynamicPPL

julia> d = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1));

julia> @model demo(d, y) = obs ~ to_submodel(primary_censored_model(d, y));

julia> only(logjoint(demo(d, 2.0), (;))) ≈ logpdf(d, 2.0)
true
```

# See also
- [`interval_censored_model`](@ref), [`double_interval_censored_model`](@ref)
- [`get_primary_event`](@ref), [`get_dist`](@ref)
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
```jldoctest
julia> using CensoredDistributions, Distributions, DynamicPPL

julia> d = interval_censored(LogNormal(1.5, 0.75), 1.0);

julia> @model demo(d, y) = obs ~ to_submodel(interval_censored_model(d, y));

julia> only(logjoint(demo(d, 2.0), (;))) ≈ logpdf(d, 2.0)
true
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
```jldoctest
julia> using CensoredDistributions, Distributions, DynamicPPL

julia> d = double_interval_censored(LogNormal(1.5, 0.75); upper = 10, interval = 1);

julia> @model demo(d, y) = obs ~ to_submodel(double_interval_censored_model(d, y));

julia> only(logjoint(demo(d, 3.0), (;))) ≈ logpdf(d, 3.0)
true
```

# See also
- [`primary_censored_model`](@ref), [`interval_censored_model`](@ref)
"
function double_interval_censored_model end
