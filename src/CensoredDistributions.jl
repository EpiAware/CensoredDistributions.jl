module CensoredDistributions

# Non-submodule imports
using DocStringExtensions
using Distributions
using Random
using Integrals

# Exported censoring functions
export primary_censored, interval_censored, double_interval_censored

# Exported utilities
export weight

include("docstrings.jl")

include("censoring/PrimaryCensored.jl")
include("censoring/IntervalCensored.jl")
include("censoring/DoubleIntervalCensored.jl")

include("utils/Weighted.jl")
end
