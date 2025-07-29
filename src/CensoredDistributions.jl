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

# Exported distribution types (needed for fitting)
export IntervalCensored, PrimaryCensored, Weighted

# Note: Fitting functionality (fit, fit_mle) is provided by the OptimizationExt
# extension when Optimization.jl is loaded. This uses Bijectors.jl for
# mathematically correct parameter transformations.

# Internal stub functions that will be defined by extensions
# These are needed for testing internal extension functionality
function _get_bijector end

# Exported utilities
export weight

include("docstrings.jl")

include("censoring/primarycensored_cdf.jl")
include("censoring/PrimaryCensored.jl")
include("censoring/IntervalCensored.jl")
include("censoring/DoubleIntervalCensored.jl")

include("utils/Weighted.jl")

end
