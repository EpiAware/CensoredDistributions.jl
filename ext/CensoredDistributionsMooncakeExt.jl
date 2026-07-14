module CensoredDistributionsMooncakeExt

using CensoredDistributions: _gamma_cdf, _collect_unique_boundaries
using CensoredDistributions: IntervalCensored
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

# `_collect_unique_boundaries(d, x)` returns the batched-pdf boundaries:
# functions of the (constant) lags `x` and interval spec, NOT the AD
# parameters, so they carry no tangent. Without a rule Mooncake traces the
# `unique`/sort internals (a `Dict` seen-set and a `Float64`->`UInt64`
# bitcast it refuses, #699). `@zero_derivative` (both modes) runs the primal
# and returns a zero tangent; the parameter gradient flows through the CDF
# evaluation in `_compute_boundary_cdfs`, not here (#701).
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(_collect_unique_boundaries), IntervalCensored, AbstractVector}

end
