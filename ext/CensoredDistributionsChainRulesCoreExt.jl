module CensoredDistributionsChainRulesCoreExt

using CensoredDistributions: _collect_unique_boundaries
using ChainRulesCore: ChainRulesCore

# `_collect_unique_boundaries(d, x)` returns the batched-pdf boundaries:
# functions of the (constant) lags and interval spec, not the AD parameters,
# so they carry no tangent. `@non_differentiable` covers reverse-mode AD
# (ReverseDiff) without tracing the `unique`/sort internals; the parameter
# gradient flows through the CDF evaluation in `_compute_boundary_cdfs`
# (#699, #701).
#
# The gamma-CDF `rrule`/`frule` formerly defined here now live in
# `EpiAwareADToolsChainRulesCoreExt` (EpiAware/CensoredDistributions.jl#850);
# only this censoring-specific boundary rule stays local.
ChainRulesCore.@non_differentiable _collect_unique_boundaries(::Any, ::Any)

end
