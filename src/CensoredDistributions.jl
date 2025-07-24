module CensoredDistributions

# Non-submodule imports
using DocStringExtensions
using Distributions
using Random
using Integrals
using LogExpFunctions
using SpecialFunctions
using Optimization
using OptimizationOptimJL

# Exported censoring functions
export primary_censored, interval_censored, double_interval_censored

# Export underlying primarycensored_cdf method for user extension
export primarycensored_cdf

# Exported distribution types (needed for fitting)
export IntervalCensored, PrimaryCensored, Weighted

# Exported utilities
export weight

# Exported fitting functions
export fit_double_interval_censored

include("docstrings.jl")

include("censoring/primarycensored_cdf.jl")
include("censoring/PrimaryCensored.jl")
include("censoring/IntervalCensored.jl")
include("censoring/DoubleIntervalCensored.jl")

include("utils/Weighted.jl")

include("fitting.jl")

# Note: fit and fit_mle are extended from Distributions.jl, not exported here
end
