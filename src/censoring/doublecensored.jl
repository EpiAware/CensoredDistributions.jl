@doc raw"""
    doublecensored(uncensored::UnivariateDistribution, censoring::UnivariateDistribution; 
                   lower::Union{Real, Nothing} = nothing, 
                   upper::Union{Real, Nothing} = nothing, 
                   interval::Union{Real, Nothing} = nothing)

Create a distribution that combines primary censoring, optional truncation, and optional secondary event censoring in the correct order.

This is a convenience function that applies the following transformations in sequence:
1. Primary event censoring using `primarycensored(uncensored, censoring)`
2. Truncation using `truncated(dist; lower=lower, upper=upper)` (if bounds are provided)
3. Secondary event censoring using `discretise(dist, interval)` (if interval is provided)
iww
The order of operations ensures mathematical correctness, particularly that truncation occurs before secondary event censoring.

# Arguments
- `uncensored::UnivariateDistribution`: The base uncensored delay distribution
- `censoring::UnivariateDistribution`: The primary event censoring distribution

# Keyword Arguments
- `lower::Union{Real, Nothing} = nothing`: Lower truncation bound. If `nothing`, no lower truncation is applied.
- `upper::Union{Real, Nothing} = nothing`: Upper truncation bound (e.g., observation time `D`). If `nothing`, no upper truncation is applied.
- `interval::Union{Real, Nothing} = nothing`: Secondary censoring window width (e.g., daily reporting). If `nothing`, no discretisation is applied.

# Returns
A composed distribution that can be used with all standard `Distributions.jl` methods (`rand`, `pdf`, `cdf`, etc.).

# Examples
```julia
using Distributions

# Basic primary censoring only
dist1 = doublecensored(LogNormal(1.5, 0.75), Uniform(0, 1))

# Primary censoring + truncation
dist2 = doublecensored(LogNormal(1.5, 0.75), Uniform(0, 1); upper=10)

# Primary censoring + secondary censoring (discretisation)
dist3 = doublecensored(LogNormal(1.5, 0.75), Uniform(0, 1); interval=1)

# Full double censoring with truncation
dist4 = doublecensored(LogNormal(1.5, 0.75), Uniform(0, 1); upper=10, interval=1)

# Sample from any of these distributions
samples = rand(dist4, 1000)
```

# Mathematical Background
This function implements the complete workflow for handling censored delay distributions as described in Park et al. (2024) and Charniga et al. (2024):

1. **Primary censoring**: Accounts for uncertainty in the primary event timing
2. **Truncation**: Handles observation windows and finite study periods  
3. **Secondary censoring**: Models discretisation effects (e.g., daily reporting)

# References
- Park et al. (2024): "Estimating epidemiological delay distributions for infectious diseases"
- Charniga et al. (2024): "Best practices for estimating and reporting epidemiological delay distributions"
"""
function doublecensored(
    uncensored::UnivariateDistribution,
    censoring::UnivariateDistribution;
    lower::Union{Real, Nothing} = nothing,
    upper::Union{Real, Nothing} = nothing,
    interval::Union{Real, Nothing} = nothing
)
    # Start with primary censoring (always applied)
    dist = primarycensored(uncensored, censoring)
    
    # Apply truncation if specified
    if !isnothing(lower) || !isnothing(upper)
        if !isnothing(lower) && !isnothing(upper)
            dist = truncated(dist, lower, upper)
        elseif !isnothing(lower)
            dist = truncated(dist; lower=lower)
        elseif !isnothing(upper)
            dist = truncated(dist; upper=upper)
        end
    end
    
    # Apply discretisation if specified (must come after truncation)
    if !isnothing(interval)
        dist = discretise(dist, interval)
    end
    
    return dist
end