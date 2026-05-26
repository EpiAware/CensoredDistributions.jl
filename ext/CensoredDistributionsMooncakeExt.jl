module CensoredDistributionsMooncakeExt

using CensoredDistributions: _gamma_cdf
using Mooncake: Mooncake

# Lifts the `ChainRulesCore.rrule` defined in
# `CensoredDistributionsChainRulesCoreExt` into Mooncake's rule registry
# for every scalar `Real` triple, so callers that pass mixed concrete
# types (e.g. `_gamma_cdf(k + 1, θ, t)` where `k + 1::Int`, a `Float32`
# parameter, or `BigFloat` for higher-precision testing) hit the
# explicit rule rather than falling back to Mooncake tracing
# `_gamma_p_series`'s data-dependent termination loop.
Mooncake.@from_chainrules Mooncake.DefaultCtx Tuple{typeof(_gamma_cdf), Real, Real, Real}

end
