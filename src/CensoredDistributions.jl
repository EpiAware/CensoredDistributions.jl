module CensoredDistributions

# Non-submodule imports
using DocStringExtensions: @template, DOCSTRING, EXPORTS, IMPORTS, TYPEDEF, TYPEDFIELDS,
                           TYPEDSIGNATURES
using Random: AbstractRNG

# Explicit imports approach for issue #121
# Import functions that we extend (for method extension)
import Distributions: params, insupport, pdf, logpdf, cdf, logcdf,
                      ccdf, logccdf, quantile, mean, var, std, median, sampler,
                      loglikelihood
# Import from Base for functions we extend that are re-exported by Distributions
import Base: minimum, maximum
# Use explicit using for types, constructors, and utility functions (no method extension)
using Distributions: UnivariateDistribution, Continuous, ValueSupport,
                     Truncated, Product, Censored, truncated,
                     product_distribution, Gamma, LogNormal, Uniform,
                     Weibull, shape, scale, meanlogx, stdlogx,
                     _in_closed_interval

using LogExpFunctions: logsubexp, logaddexp, log1mexp

using SpecialFunctions: gamma

using HypergeometricFunctions: M

using Integrals: IntegralProblem, solve, QuadGKJL

using Optimization: OptimizationFunction, OptimizationProblem, solve, ReturnCode

using OptimizationOptimJL: NelderMead

# Exported censoring functions
export primary_censored, interval_censored, double_interval_censored

# Export underlying methods for user extension
export primarycensored_cdf, primarycensored_logcdf

# Exported distributions
export ExponentiallyTilted

# Exported utilities
export weight, get_dist, get_dist_recursive

include("docstrings.jl")

include("censoring/primarycensored_cdf.jl")
include("censoring/PrimaryCensored.jl")
include("censoring/IntervalCensored.jl")
include("censoring/double_interval_censored.jl")

include("distributions/ExponentiallyTilted.jl")

include("utils/Weighted.jl")
include("utils/get_dist.jl")
include("utils/quantile_optimization.jl")

# Public API - functions that are part of public interface but not exported
@static if VERSION >= v"1.11"
    include("public.jl")
else
    # Julia 1.10 compatibility - no public keyword, but structs are accessible
end

end
