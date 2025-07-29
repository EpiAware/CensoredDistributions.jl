module CensoredDistributions

# Non-submodule imports
using DocStringExtensions
using Distributions
using Random
using Integrals
using LogExpFunctions
using SpecialFunctions
using HypergeometricFunctions

# Exported censoring functions
export primary_censored, interval_censored, double_interval_censored

# Export underlying primarycensored_cdf method for user extension
export primarycensored_cdf

# Exported utilities
export weight

include("docstrings.jl")

include("censoring/primarycensored_cdf.jl")
include("censoring/PrimaryCensored.jl")
include("censoring/IntervalCensored.jl")
include("censoring/DoubleIntervalCensored.jl")

include("utils/Weighted.jl")

end
