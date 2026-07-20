module CensoredDistributionsEnzymeExt

using CensoredDistributions: _collect_unique_boundaries
using Enzyme.EnzymeRules: EnzymeRules

# `_collect_unique_boundaries(d, x)` returns the batched-pdf boundaries:
# functions of the (constant) lags and interval spec, NOT the AD parameters,
# so they carry no tangent. Enzyme's strict type analysis otherwise rejects
# the `unique`/sort `Union`-typed temporaries (`IllegalTypeAnalysisException`,
# #701). `inactive` runs the primal unchanged; the parameter gradient flows
# through the CDF evaluation in `_compute_boundary_cdfs`, not here.
#
# The `_gamma_cdf` and `SpecialFunctions.gamma` Enzyme rules formerly defined
# here now live in `EpiAwareADToolsEnzymeExt`
# (EpiAware/CensoredDistributions.jl#850); only this censoring-specific
# boundary rule stays local.
EnzymeRules.inactive(::typeof(_collect_unique_boundaries), args...) = nothing

end
