module CensoredDistributions

# Non-submodule imports
using DocStringExtensions: @template, DOCSTRING, EXPORTS, IMPORTS, TYPEDEF, TYPEDFIELDS,
                           TYPEDSIGNATURES
using Random: AbstractRNG, default_rng

# Explicit imports approach for issue #121
# Import functions that we extend (for method extension)
import Distributions: params, insupport, pdf, logpdf, cdf, logcdf,
                      ccdf, logccdf, quantile, mean, var, std, median, sampler,
                      loglikelihood
# Import from Base for functions we extend that are re-exported by Distributions
import Base: minimum, maximum
# Use explicit using for types, constructors, and utility functions (no method extension)
using Distributions: Distributions, UnivariateDistribution, Distribution,
                     Continuous, Multivariate, MixtureModel,
                     ValueSupport, Truncated, Product, Censored, truncated,
                     product_distribution, Exponential, Gamma, LogNormal, Uniform,
                     Weibull, Normal, shape, scale, meanlogx, stdlogx,
                     _in_closed_interval

using PrecompileTools: @setup_workload, @compile_workload

using LogExpFunctions: logsubexp, log1mexp

using SpecialFunctions: gamma, gamma_inc, loggamma, digamma

import Tables

import FastGaussQuadrature  # provides Gauss-Legendre nodes for the default solver

using Optimization: OptimizationFunction, OptimizationProblem, solve, ReturnCode

using OptimizationOptimJL: NelderMead

# Exported censoring functions
export primary_censored, interval_censored, double_interval_censored

# Exported latent representation
export latent, PrimaryConditional, primary_conditional_logpdf

# Export underlying methods for user extension
export primarycensored_cdf, primarycensored_logcdf

# Exported distributions
export ExponentiallyTilted

# Exported convolution constructor
export convolve_distributions

# Exported generic composers and front-end constructor
export Sequential, Parallel, Competing, compose, as_mixture

# Exported composer-observed lowering used by the external censoring wrappers
export observed_distribution

# Exported utilities
export weight, get_dist, get_dist_recursive, get_primary_event

# Exported DynamicPPL submodel constructors. These have no methods until
# DynamicPPL (or Turing) is loaded; the methods live in the package extension so
# the core stays Turing-free.
export primary_censored_model, interval_censored_model,
       double_interval_censored_model

include("docstrings.jl")

include("utils/gamma_ad.jl")

include("integration/integration.jl")

include("censoring/primarycensored_cdf.jl")
include("censoring/PrimaryCensored.jl")
include("censoring/PrimaryConditional.jl")
include("censoring/Latent.jl")
include("censoring/IntervalCensored.jl")
include("censoring/double_interval_censored.jl")

include("distributions/ExponentiallyTilted.jl")
include("distributions/Convolved.jl")

include("composers/Sequential.jl")
include("composers/Parallel.jl")
include("composers/Competing.jl")
include("composers/nesting.jl")
include("composers/equality.jl")
include("composers/compose.jl")
include("composers/wrap.jl")

include("utils/Weighted.jl")
include("utils/get_dist.jl")
include("utils/quantile_optimization.jl")

# Censored specialisations of the generic composers (#329, PR3b): included last
# as they depend on the composers, the censored types, `get_dist_recursive`
# (utils/get_dist.jl) and the integration helpers.
include("composers/censored_specialisations.jl")

# Turing-free `primary_censored_model` function stub (#88, PR2). Has no methods
# until DynamicPPL is loaded; the methods live in the package extension.
include("turing_models.jl")

# Public API - functions that are part of public interface but not exported
@static if VERSION >= v"1.11"
    include("public.jl")
else
    # Julia 1.10 compatibility - no public keyword, but structs are accessible
end

# Precompile workload covering the double_interval_censored pipeline for
# representative delay distributions, toggling force_numeric to hit both the
# analytical and numeric primary-censored CDF paths in a single entry point.
# See https://github.com/EpiAware/CensoredDistributions.jl/issues/212.
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
            for force_numeric in (false, true)
                dic = double_interval_censored(
                    d; primary_event = primary, upper = 10.0,
                    interval = 1.0, force_numeric = force_numeric)
                cdf(dic, x)
                logcdf(dic, x)
                pdf(dic, x)
                logpdf(dic, x)
            end
        end
    end
end

end
