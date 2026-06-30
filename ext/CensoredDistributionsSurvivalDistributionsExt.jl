module CensoredDistributionsSurvivalDistributionsExt

using CensoredDistributions: CensoredDistributions, _gamma_cdf
import CensoredDistributions: _cdf_ad_safe, _ccdf_ad_safe,
                              _logcdf_ad_safe, _logccdf_ad_safe
using CensoredDistributions: Sequential, Parallel
using Distributions: shape, scale
import Distributions: logcdf
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
# `SurvivalDistributions.LogLogistic` needs no special AD routing here: its
# `logccdf` is built from elementary operations (`log1p`/`exp`), so it
# differentiates through the generic elementary `logccdf` fallback under every
# backend without a `_*_ad_safe` method.

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

# The public `logcdf(::GeneralizedGamma, t)` must be AD-safe too, not just the
# package-internal `_*_ad_safe` helpers above. `SurvivalDistributions` defines
# `logccdf(GG, t) = logccdf(d.G, t^gamma)` but no `logcdf`, so a direct
# `logcdf(GeneralizedGamma(θ...), t)` falls through to the generic
# `Distributions.logcdf`, which evaluates the inner Gamma's `logcdf` →
# `StatsFuns._gammalogcdf`. That has no `ForwardDiff.Dual` /
# `ReverseDiff.TrackedReal` / Mooncake method, so under any AD backend it strips
# the `Dual` and throws (`no method matching _gammalogccdf(::Dual, ...)`).
# Routing `logcdf` through the `_gamma_cdf`-backed helper makes a bare `logcdf`
# differentiate everywhere the censored pipelines already do, closing the gap a
# user hits scoring a GeneralizedGamma leaf directly. `cdf`/`ccdf`/`logccdf` are
# owned by `SurvivalDistributions` (redefining them here is method-overwriting
# piracy and breaks precompilation), so they are left to the package's
# `_cdf_ad_safe` / `_ccdf_ad_safe` / `_logccdf_ad_safe` helpers, which the
# censoring pipelines already use and which the AD-parity testitem locks in.
# Only `logcdf` is unclaimed and so safely AD-routed at the public method.
logcdf(d::SD.GeneralizedGamma, t::Real) = _logcdf_ad_safe(d, t)

# ============================================================================
# Hazard accessor interop with SurvivalDistributions
# ============================================================================
#
# CensoredDistributions defines its own unexported `hazard` / `loghazard` /
# `cumhazard` / `survival` (see `src/utils/hazards.jl`) so they do not clash
# with SurvivalDistributions' exported `hazard`/`loghazard`/`cumhazard` when
# both packages are loaded. The definitions are identical (the standard survival
# identities `h = f/S`, `H = -log S`, `log h = log f - log S`), so the two agree
# on every univariate delay automatically: `SD.hazard` already reads
# `pdf`/`ccdf`, which every composed univariate delay defines, and
# `CensoredDistributions.hazard` reads the same surface, so a tree of SD leaves
# and an SD leaf in a tree both work under either entry.
#
# The only gap is the multivariate verb composer (`Sequential`): SD's generic
# `hazard(::UnivariateDistribution, t)` does not match a `Sequential` (which is
# `Multivariate`), so `SD.hazard(tree, t)` would error where
# `CensoredDistributions.hazard(tree, t)` reduces the chain to its marginal
# time-to-event convolution. Forward SD's verbs to the package accessors for
# the verb composers so the same tree-level hazard is reachable through either
# package's function name. A `Parallel` raises the same ambiguity error the
# package accessor does (its `_hazard_marginal(::Parallel)` throws).
for V in (Sequential, Parallel)
    SD.hazard(d::V, t::Real) = CensoredDistributions.hazard(d, t)
    SD.loghazard(d::V, t::Real) = CensoredDistributions.loghazard(d, t)
    SD.cumhazard(d::V, t::Real) = CensoredDistributions.cumhazard(d, t)
end

end # module
