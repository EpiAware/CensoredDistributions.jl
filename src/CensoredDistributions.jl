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
using Distributions: Distributions, UnivariateDistribution,
                     ContinuousUnivariateDistribution, Continuous,
                     ValueSupport, Truncated, Product, Censored, truncated,
                     product_distribution, Exponential, Gamma, LogNormal, Uniform,
                     Weibull, _in_closed_interval

import ConvolvedDistributions: convolve_series, delay_masses, quantile_by_optimization
import ConvolvedDistributions: Convolved, AnalyticalSolver, NumericSolver,
                               AbstractSolverMethod

# The `quantile_by_optimization` method is added by ConvolvedDistributions'
# `ConvolvedDistributionsOptimizationExt` extension, which only loads once its
# trigger weakdeps -- Optimization and OptimizationOptimJL -- are loaded. Both
# are hard deps of CensoredDistributions (the quantile for PrimaryCensored /
# IntervalCensored lives behind that extension), so import them at module scope
# to guarantee the extension, and hence inverse-CDF support, is available
# whenever CensoredDistributions is loaded. They are imported for this
# load-order side effect, not for any symbol (hence the stale-import ignore in
# test/package/ExplicitImports.jl).
import Optimization
import OptimizationOptimJL

using PrecompileTools: @setup_workload, @compile_workload

using LogExpFunctions: log1mexp

# Shared AD-safety machinery (EpiAware/CensoredDistributions.jl#850): the
# AD-safe cdf/logcdf hooks, used by the discretisation (IntervalCensored etc.).
using EpiAwareADTools: cdf_ad_safe, logcdf_ad_safe

# Exported censoring functions
export primary_censored, interval_censored, double_interval_censored

# Exported solver methods for selecting the primary-censoring CDF backend.
# These are re-exported from ConvolvedDistributions (which owns the types, so
# `PrimaryCensored.method` holds a ConvolvedDistributions solver and there is
# a single source of truth for `AnalyticalSolver`/`NumericSolver`).
export AnalyticalSolver, NumericSolver

# Exported distributions
export ExponentiallyTilted

# Exported convolution constructor from ConvolvedDistributions (the old
# standalone `CensoredDistributions.Convolved`/`convolve_distributions` was
# removed; `convolved`/`Convolved` now come from ConvolvedDistributions).

# Exported utilities
export weight, get_dist, get_dist_recursive

include("docstrings.jl")

include("censoring/PrimaryCensored.jl")
include("censoring/IntervalCensored.jl")
include("censoring/double_interval_censored.jl")

include("distributions/ExponentiallyTilted.jl")

include("utils/Weighted.jl")
include("utils/get_dist.jl")
include("convolve_series.jl")

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
