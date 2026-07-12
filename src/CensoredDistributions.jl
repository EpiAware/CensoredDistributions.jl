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
using Distributions: Distributions, UnivariateDistribution, Continuous,
                     ValueSupport, Truncated, Product, Censored, truncated,
                     product_distribution, Exponential, Gamma, LogNormal, Uniform,
                     Weibull, Normal, shape, scale, meanlogx, stdlogx,
                     _in_closed_interval

using PrecompileTools: @setup_workload, @compile_workload

using LogExpFunctions: logsubexp, log1mexp

using SpecialFunctions: gamma, gamma_inc, loggamma, digamma

import FastGaussQuadrature  # provides Gauss-Legendre nodes for the default solver

using Optimization: OptimizationFunction, OptimizationProblem, solve, ReturnCode

using OptimizationOptimJL: NelderMead

# The composer-leaf hooks CD extends so a censored delay behaves like any other
# leaf of a ComposedDistributions tree (see `censoring/composed_leaves.jl`).
# `_shared_tag` and `_uncertain_specs` are ComposedDistributions internals that
# a leaf-wrapper package must extend: without them a tied delay is estimated
# twice and an attached prior is silently dropped. Tracked upstream for
# promotion to the public surface.
import ComposedDistributions: free_leaf, rewrap_leaf, _shared_tag,
                              _uncertain_specs

# Exported censoring functions
export primary_censored, interval_censored, double_interval_censored

# Export underlying methods for user extension
export primarycensored_cdf, primarycensored_logcdf

# Exported solver methods for selecting the primary-censoring CDF backend
export AnalyticalSolver, NumericSolver

# Exported distributions
export ExponentiallyTilted

# Exported convolution constructor
export convolve_distributions

# Exported utilities
export weight, get_dist, get_dist_recursive

include("docstrings.jl")

include("utils/gamma_ad.jl")

include("integration/integration.jl")

include("censoring/primarycensored_cdf.jl")
include("censoring/PrimaryCensored.jl")
include("censoring/IntervalCensored.jl")
include("censoring/double_interval_censored.jl")
include("censoring/composed_leaves.jl")

include("distributions/ExponentiallyTilted.jl")
include("distributions/Convolved.jl")

include("utils/Weighted.jl")
include("utils/get_dist.jl")
include("utils/quantile_optimization.jl")

# Public API - functions that are part of public interface but not exported
@static if VERSION >= v"1.11"
    include("public.jl")
else
    # Julia 1.10 compatibility - no public keyword, but structs are accessible
end

# Precompile workload covering the double_interval_censored pipeline for
# representative delay distributions, toggling the solver method to hit both
# the analytical and numeric primary-censored CDF paths in a single entry
# point. See https://github.com/EpiAware/CensoredDistributions.jl/issues/212.
@setup_workload begin
    delays = (
        Gamma(2.0, 1.5),
        LogNormal(1.5, 0.75),
        Weibull(2.0, 1.5),
        Exponential(1.5)
    )
    primary = Uniform(0.0, 1.0)
    x = 2.5

    @compile_workload begin
        for d in delays
            for method in (AnalyticalSolver(), NumericSolver())
                dic = double_interval_censored(
                    d; primary_event = primary, upper = 10.0,
                    interval = 1.0, method = method)
                cdf(dic, x)
                logcdf(dic, x)
                pdf(dic, x)
                logpdf(dic, x)
            end
        end
    end
end

end
