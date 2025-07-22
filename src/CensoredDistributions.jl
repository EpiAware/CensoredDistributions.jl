module CensoredDistributions

# Non-submodule imports
using DocStringExtensions
using Distributions
using Random
using Integrals

# Exported censoring functions
export primarycensored, discretise, discretize, within_interval_censored, doublecensored

# Exported utilities
export weight

include("docstrings.jl")

include("censoring/PrimaryCensored.jl")
include("censoring/Discretised.jl")
include("censoring/WithinIntervalCensored.jl")
include("censoring/doublecensored.jl")

include("utils/Weighted.jl")
end
