module CensoredDistributionsMooncakeExt

using CensoredDistributions: _collect_unique_boundaries, IntervalCensored
using Mooncake: Mooncake

# `_collect_unique_boundaries(d, x)` returns the batched-pdf boundaries:
# functions of the (constant) lags `x` and interval spec, NOT the AD
# parameters, so they carry no tangent. Without a rule Mooncake traces the
# `unique`/sort internals (a `Dict` seen-set and a `Float64`->`UInt64`
# bitcast it refuses, #699). `@zero_derivative` (both modes) runs the primal
# and returns a zero tangent; the parameter gradient flows through the CDF
# evaluation in `_compute_boundary_cdfs`, not here (#701).
#
# The gamma-CDF `@from_chainrules` lift formerly here now lives in
# `EpiAwareADToolsMooncakeExt` (EpiAware/CensoredDistributions.jl#850); only
# this censoring-specific boundary rule stays local.
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(_collect_unique_boundaries), IntervalCensored, AbstractVector}

end
