module CensoredDistributionsSurvivalDistributionsExt

using CensoredDistributions: _gamma_cdf
import CensoredDistributions: _cdf_ad_safe, _ccdf_ad_safe,
                              _logcdf_ad_safe, _logccdf_ad_safe
using Distributions: shape, scale
import SurvivalDistributions as SD

# AD-safe CDF family for `SurvivalDistributions.GeneralizedGamma`.
#
# GeneralizedGamma carries an inner `Gamma(nu/gamma, sigma^gamma)` and defines
# `logccdf(d, t) = logccdf(d.G, t^gamma)`. The stock `logccdf(::Gamma)` routes
# through `StatsFuns._gammalogccdf`, which has no `ForwardDiff.Dual` /
# `ReverseDiff.TrackedReal` / Mooncake method, so the censored pipelines
# (`primary_censored` / `interval_censored` / `double_interval_censored` /
# `Convolved`) that query `cdf` / `logccdf` of a GeneralizedGamma leaf error
# under every AD backend.
#
# The package already differentiates the regularised lower incomplete gamma via
# the `_gamma_cdf` helper (`src/utils/gamma_ad.jl`), whose per-backend rules live
# in the AD extensions. Routing the inner Gamma through `_gamma_cdf` at the
# transformed point `t^gamma` makes the GeneralizedGamma CDF family
# differentiate everywhere the plain `Gamma` path does, mirroring the
# `_*_ad_safe(::Gamma)` methods. The `t^gamma` transform and the inner
# `shape`/`scale` (functions of `nu`, `gamma`, `sigma`) are elementary, so the
# gradient flows through all three parameters.
#
# `GeneralizedGamma`'s constructor promotes its parameters into the inner
# `Gamma{T}`, so a `Dual`/`Tracked` parameter is preserved in `shape(d.G)` /
# `scale(d.G)`; the helpers below pull those out and the `_gamma_cdf` rules do
# the rest.
#
# `SurvivalDistributions.LogLogistic` needs NO special AD routing here: its
# `logccdf` is built from elementary operations (`log1p`/`exp`), so it
# differentiates through the generic elementary `logccdf` fallback under every
# backend without a `_*_ad_safe` method (#487).

function _gg_cdf(d::SD.GeneralizedGamma, u::Real)
    return _gamma_cdf(shape(d.G), scale(d.G), u^d.gamma)
end

function _cdf_ad_safe(d::SD.GeneralizedGamma, u::Real)
    u <= 0 && return zero(float(u))
    return _gg_cdf(d, u)
end

function _ccdf_ad_safe(d::SD.GeneralizedGamma, u::Real)
    u <= 0 && return one(float(u))
    return 1 - _gg_cdf(d, u)
end

function _logcdf_ad_safe(d::SD.GeneralizedGamma, u::Real)
    u <= 0 && return oftype(float(u), -Inf)
    return log(_gg_cdf(d, u))
end

function _logccdf_ad_safe(d::SD.GeneralizedGamma, u::Real)
    u <= 0 && return zero(float(u))
    return log1p(-_gg_cdf(d, u))
end

end # module
