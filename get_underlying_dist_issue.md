# Issue: Add `get_underlying_dist` function for extracting core distributions

## Summary
Add a utility function `get_underlying_dist` (or similar name) that extracts the underlying "true" distribution from wrapped/censored distributions using multiple dispatch.

## Motivation
After fitting censored distributions, users often need to access the underlying distribution for:
- Parameter analysis and comparison
- Plotting the core distribution
- Debugging and inspection
- Statistical analysis of the fitted parameters

Currently, users must manually navigate the distribution structure (e.g., `dist.dist`, `dist.dist.dist`, etc.)

## Proposed API

```julia
# Extract underlying distribution
get_underlying_dist(d::IntervalCensored) = d.dist
get_underlying_dist(d::PrimaryCensored) = d.dist  # delay distribution
get_underlying_dist(d::DoubleIntervalCensored) = d.dist.dist  # delay from primary censored
get_underlying_dist(d::Truncated{<:CensoredDistribution}) = get_underlying_dist(d.untruncated)

# For product distributions - return unique underlying distributions
get_underlying_dist(d::Product) = unique([get_underlying_dist(di) for di in components(d)])

# For already-underlying distributions, return as-is
get_underlying_dist(d::ContinuousUnivariateDistribution) = d
```

## Examples

```julia
# After fitting
fitted = fit_mle(interval_censored(Normal(0, 1), 1.0), data)
underlying = get_underlying_dist(fitted)  # Returns the fitted Normal distribution

# For complex nested cases
double_fitted = fit_mle(double_interval_censored(...), data)
underlying = get_underlying_dist(double_fitted)  # Returns the delay distribution

# For product distributions
product_fitted = fit_mle(template, heterogeneous_data)
uniques = get_underlying_dist(product_fitted)  # Returns vector of unique distributions
```

## Implementation Notes
- Use multiple dispatch for clean extension to new distribution types
- Handle nested/wrapped distributions recursively
- For product distributions, avoid duplicates by returning unique distributions
- Consider name alternatives: `underlying_distribution`, `core_dist`, `base_dist`

## Benefits
- Consistent API across all censored distribution types
- Easier parameter extraction and analysis
- Better user experience for post-fitting analysis
- Enables easier plotting and visualization of fitted results
