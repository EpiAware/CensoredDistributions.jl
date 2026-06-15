@doc """


Create a distribution that combines primary interval censoring, optional truncation, and optional secondary interval censoring in the correct order.

This is a convenience function that applies the following transformations in sequence:
1. Primary event censoring using `primary_censored(dist, primary_event)`
2. Truncation using `truncated(dist; lower=lower, upper=upper)` (if bounds are provided)
3. Secondary interval censoring using `interval_censored(dist, interval)` (if interval is provided)

The order of operations ensures mathematical correctness, particularly that truncation occurs before secondary interval censoring.

# Arguments
- `dist`: The delay distribution from primary event to observation

# Keyword Arguments
- `primary_event`: The primary event time distribution. Defaults to uniform distribution over [0, 1].
- `lower`: Lower truncation bound. If `nothing`, no lower truncation is applied.
- `upper`: Upper truncation bound (e.g., observation time `D`). If `nothing`, no upper truncation is applied.
- `interval`: Secondary censoring interval width (e.g., daily reporting). If `nothing`, no interval censoring is applied.
- `method`: Primary-censoring solver method, an [`AnalyticalSolver`](@ref) or [`NumericSolver`](@ref). Defaults to `AnalyticalSolver()`. Passing a concrete method keeps the return type concrete when the delay parameters are runtime values.
- `force_numeric`: Deprecated. Pass `method = NumericSolver()` instead.

# Returns
A composed distribution that can be used with all standard `Distributions.jl` methods (`rand`, `pdf`, `cdf`, etc.).

# Examples
```@example
using CensoredDistributions, Distributions

# Basic primary censoring only (uses default Uniform(0, 1))
dist1 = double_interval_censored(LogNormal(1.5, 0.75))

# Primary censoring + truncation
dist2 = double_interval_censored(LogNormal(1.5, 0.75); upper=10)

# Primary censoring + secondary interval censoring
dist3 = double_interval_censored(LogNormal(1.5, 0.75); interval=1)

# Full double interval censoring with truncation
dist4 = double_interval_censored(LogNormal(1.5, 0.75); upper=10, interval=1)

# Evaluate distribution functions and compute statistics
pdf_at_3 = pdf(dist4, 3.0)        # probability density at 3 days
cdf_at_7 = cdf(dist4, 7.0)        # P(X ≤ 7) with full censoring pipeline
ccdf_at_5 = ccdf(dist4, 5.0)      # survival function
q25 = quantile(dist4, 0.25)       # 25th percentile
median_delay = quantile(dist4, 0.5)  # median
q95 = quantile(dist4, 0.95)       # 95th percentile
samples = rand(dist4, 100)        # random samples

# Custom primary event distribution
dist5 = double_interval_censored(LogNormal(1.5, 0.75); primary_event=Uniform(0, 2))
```

# Mathematical Background
This function implements the complete workflow for handling censored delay distributions as described in [park2024estimating](@cite) and [charniga2024best](@cite):

1. **Primary censoring**: Accounts for uncertainty in the primary event timing
2. **Truncation**: Handles observation windows and finite study periods
3. **Secondary censoring**: Models interval censoring effects (e.g., daily reporting)

# See also
- [`double_censored`](@ref): a shorter alias for this function.
"""
function double_interval_censored(
        dist::UnivariateDistribution;
        primary_event::UnivariateDistribution = Uniform(0, 1),
        lower::Union{Real, Nothing} = nothing,
        upper::Union{Real, Nothing} = nothing,
        interval::Union{Real, Nothing} = nothing,
        method::Union{AbstractSolverMethod, Nothing} = nothing,
        force_numeric = nothing
)
    # Start with primary censoring (always applied)
    result = primary_censored(dist, primary_event; method = method,
        force_numeric = force_numeric)

    # Apply truncation if specified
    if !isnothing(lower) || !isnothing(upper)
        if !isnothing(lower) && !isnothing(upper)
            result = truncated(result, lower, upper)
        elseif !isnothing(lower)
            result = truncated(result; lower = lower)
        elseif !isnothing(upper)
            result = truncated(result; upper = upper)
        end
    end

    # Apply interval censoring if specified (must come after truncation)
    if !isnothing(interval)
        result = interval_censored(result, interval)
    end

    return result
end

@doc """

Short, clear alias for [`double_interval_censored`](@ref).

`double_censored(dist; ...)` is identical to `double_interval_censored(dist; ...)`,
provided as a shorter name for the common primary + truncation + secondary
interval pipeline. (Deliberately NOT `dic`, which clashes with the Deviance
Information Criterion.)

# Examples
```@example
using CensoredDistributions, Distributions

double_censored(LogNormal(1.5, 0.75); upper = 10, interval = 1)
```

# See also
- [`double_interval_censored`](@ref): the full constructor this aliases.
"""
const double_censored = double_interval_censored
