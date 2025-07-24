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

# Import and re-export fitting methods from Distributions.jl
import Distributions: fit, fit_mle
export fit, fit_mle

include("docstrings.jl")

include("censoring/primarycensored_cdf.jl")
include("censoring/PrimaryCensored.jl")
include("censoring/IntervalCensored.jl")
include("censoring/DoubleIntervalCensored.jl")

include("utils/Weighted.jl")

include("censoring/fit.jl")
end
