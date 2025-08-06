@doc raw"""
    double_interval_censored(dist::UnivariateDistribution;
                          primary_event::UnivariateDistribution = Uniform(0, 1),
                          lower::Union{Real, Nothing} = nothing,
                          upper::Union{Real, Nothing} = nothing,
                          interval::Union{Real, Nothing} = nothing,
                          force_numeric::Bool = false)

Create a distribution that combines primary interval censoring, optional truncation, and optional secondary interval censoring in the correct order.

This is a convenience function that applies the following transformations in sequence:
1. Primary event censoring using `primary_censored(dist, primary_event)`
2. Truncation using `truncated(dist; lower=lower, upper=upper)` (if bounds are provided)
3. Secondary interval censoring using `interval_censored(dist, interval)` (if interval is provided)

The order of operations ensures mathematical correctness, particularly that truncation occurs before secondary interval censoring.

# Arguments
- `dist::UnivariateDistribution`: The delay distribution from primary event to observation

# Keyword Arguments
- `primary_event::UnivariateDistribution = Uniform(0, 1)`: The primary event time distribution. Defaults to uniform distribution over [0, 1].
- `lower::Union{Real, Nothing} = nothing`: Lower truncation bound. If `nothing`, no lower truncation is applied.
- `upper::Union{Real, Nothing} = nothing`: Upper truncation bound (e.g., observation time `D`). If `nothing`, no upper truncation is applied.
- `interval::Union{Real, Nothing} = nothing`: Secondary censoring interval width (e.g., daily reporting). If `nothing`, no interval censoring is applied.
- `force_numeric::Bool = false`: Force numerical integration for primary censoring even when analytical solutions exist.

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
This function implements the complete workflow for handling censored delay distributions as described in Park et al. (2024) and Charniga et al. (2024):

1. **Primary censoring**: Accounts for uncertainty in the primary event timing
2. **Truncation**: Handles observation windows and finite study periods
3. **Secondary censoring**: Models interval censoring effects (e.g., daily reporting)

# References
- Park et al. (2024): "Estimating epidemiological delay distributions for infectious diseases"
- Charniga et al. (2024): "Best practices for estimating and reporting epidemiological delay distributions"
"""
function double_interval_censored(
        dist::UnivariateDistribution;
        primary_event::UnivariateDistribution = Uniform(0, 1),
        lower::Union{Real, Nothing} = nothing,
        upper::Union{Real, Nothing} = nothing,
        interval::Union{Real, Nothing} = nothing,
        force_numeric::Bool = false
)
    # Start with primary censoring (always applied)
    result = primary_censored(dist, primary_event; force_numeric = force_numeric)

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
