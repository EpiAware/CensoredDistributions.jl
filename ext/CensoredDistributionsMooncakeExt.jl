module CensoredDistributionsMooncakeExt

using CensoredDistributions: _gamma_cdf
using Mooncake: Mooncake

# Lifts the `ChainRulesCore.rrule` and `ChainRulesCore.frule` defined in
# `CensoredDistributionsChainRulesCoreExt` into Mooncake's rule registry
# (default mode generates both reverse `rrule!!` and forward `frule!!`)
# for every scalar `Real` triple, so callers that pass mixed concrete
# types (e.g. `_gamma_cdf(k + 1, θ, t)` where `k + 1::Int`, a `Float32`
# parameter, or `BigFloat` for higher-precision testing) hit the
# explicit rule rather than falling back to Mooncake tracing the
# function body. The forward `frule` is what closes #270: without it the
# generated `frule!!` calls `ChainRulesCore.frule`, gets `nothing`, and
# errors with `iterate(::Nothing)`.
Mooncake.@from_chainrules Mooncake.DefaultCtx Tuple{typeof(_gamma_cdf), Real, Real, Real}

end
