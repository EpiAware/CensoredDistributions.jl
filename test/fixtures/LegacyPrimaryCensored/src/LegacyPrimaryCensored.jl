# LegacyPrimaryCensored: a self-contained, verbatim snapshot of the
# pre-ConvolvedDistributions primary-censoring implementation, used as an
# oracle by the delegation harness (`test/censoring/PrimaryCensored_delegation.jl`)
# during the refactor that reimplemented `CensoredDistributions.PrimaryCensored`
# as a thin wrapper over `ConvolvedDistributions.Convolved`.
#
# The module deliberately exports NOTHING and keeps every name internal
# (non-colliding) so a harness that does `using CensoredDistributions,
# LegacyPrimaryCensored` never hits a name ambiguity with the production
# package's `primary_censored`/`PrimaryCensored`/`AnalyticalSolver`/
# `NumericSolver`/`GaussLegendre`. The harness reaches them qualified, and
# the one public-ish alias the harness calls (`legacy_primary_censored`) is
# deliberately kept out of `export`.
module LegacyPrimaryCensored

import ConvolvedDistributions: quantile_by_optimization

# Functions extended with new methods.
import Distributions: params, insupport, pdf, logpdf, cdf, logcdf,
                      ccdf, logccdf, quantile, mean, sampler
import Base: minimum, maximum, rand

# Types, constructors, and helpers used without method extension.
using Distributions: Distributions, UnivariateDistribution,
                     ContinuousUnivariateDistribution, Continuous,
                     Truncated, truncated, Exponential, Gamma, LogNormal,
                     Uniform, Weibull, Normal, shape, scale, meanlogx,
                     stdlogx, _in_closed_interval

using Random: AbstractRNG
using LogExpFunctions: logsubexp, log1mexp
using SpecialFunctions: gamma
# Shared AD-safety machinery (mirrors the production module head): the
# gamma-CDF helper carrying the shape-parameter derivative and the AD-safe
# cdf/logcdf hooks.
using EpiAwareADTools: _gamma_cdf, cdf_ad_safe, logcdf_ad_safe
import FastGaussQuadrature

# Verbatim copies of the old implementation (see the class comment above).
include("integration.jl")
include("primarycensored_cdf.jl")
include("PrimaryCensored.jl")

# Minimal `get_dist` mirror: the verbatim old `PrimaryCensored` methods
# delegate the delay distribution to `get_dist`, which the production
# package defines in `src/utils/get_dist.jl` (not included here).
function get_dist(d::PrimaryCensored)
    return d.dist
end

# Alias the old constructor so the delegation harness can call it
# unambiguously. Deliberately not exported.
const legacy_primary_censored = primary_censored

end # module LegacyPrimaryCensored
