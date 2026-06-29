module CensoredDistributionsEnzymeExt

using CensoredDistributions: _gamma_cdf, _gamma_cdf_value_and_partials,
                             _window_quantile, _subevent_slice,
                             _collect_unique_boundaries, _weight_deprecation
using Distributions: UnivariateDistribution
using Enzyme: Enzyme
using Enzyme.EnzymeRules: EnzymeRules
using SpecialFunctions: gamma, digamma

# `_subevent_slice(events, o_idx, ev_idx, n)` gathers a nested tree node's
# `[origin, leaf_events...]` sub-view from the CONSTANT event vector by pure
# index bookkeeping (`src/composers/censored_specialisations.jl`). Its output is
# a freshly-allocated `Vector{eltype(events)}` of event VALUES (data); the gradient
# flows only through the leaf distribution PARAMS at each `_tree_step`, never
# through these copied event times. On a multi-edge tree the event vector has a
# non-bits `Union{Missing, Float64}` element type, and Enzyme's reverse type
# analysis cannot statically prove the layout of that `Array` allocation inside the
# differentiated recursion (`EnzymeNoTypeError` at the `Array` ctor). Marking
# the gather inactive runs it on the primal unchanged and treats the returned slice
# as `Const`, so Enzyme never type-analyses the union-array allocation while the
# leaf-param gradients still flow. This is the Enzyme analogue of the Mooncake
# `@zero_adjoint` shields on the event-name helpers. Correct because the slice
# depends only on inactive inputs (the Const event vector and `Int` indices).
EnzymeRules.inactive(::typeof(_subevent_slice), args...) = nothing

# `_window_quantile(comp, p)` returns a quadrature-window endpoint — the
# *location* at which to clamp an infinite integration limit
# (`_finite_window` in `src/distributions/Convolved.jl`). It is a
# non-differentiable hyperparameter of the quadrature (like the node
# count), so it is marked `EnzymeRules.inactive`: Enzyme runs the primal
# unchanged and treats the returned endpoint as a constant, contributing no
# tangent / no cotangent in either mode. Crucially this stops Enzyme tracing
# INTO the function at all, so it never reaches `quantile(::Gamma)` →
# `SpecialFunctions.gamma_inc_inv_qsmall`, which it cannot differentiate
# (`IllegalTypeAnalysisException`). `inactive` covers every
# activity / batch-width / mode permutation uniformly — unlike a bespoke
# `forward`/`reverse` pair, which had to enumerate `Duplicated` /
# `BatchDuplicated` / `Const` returns and still missed the `Active` scalar
# return that a `Difference` over an unbounded-above subtrahend produces under
# Enzyme reverse. Other backends get the same treatment via the ChainRules
# `@non_differentiable _primal` mark, the ForwardDiff/ReverseDiff
# primal-stripping methods and the Mooncake `@zero_derivative` rule. Marking
# the value (not just the params) inactive is correct: the endpoint is a fixed
# quadrature hyperparameter that carries no gradient.
EnzymeRules.inactive(::typeof(_window_quantile), args...) = nothing

# `_collect_unique_boundaries(d, x)` returns the batched-pdf boundaries:
# functions of the (constant) lags and interval spec, NOT the AD parameters,
# so they carry no tangent. Enzyme's strict type analysis otherwise rejects
# the `unique`/sort `Union`-typed temporaries (`IllegalTypeAnalysisException`,
# #701). `inactive` runs the primal unchanged; the parameter gradient flows
# through the CDF evaluation in `_compute_boundary_cdfs`, not here.
EnzymeRules.inactive(::typeof(_collect_unique_boundaries), args...) = nothing

# `_weight_deprecation()` emits the `weight` soft-deprecation warning (issue
# #128) via `Base.depwarn`, called from every `weight` constructor — which the
# AD fixtures invoke INSIDE the differentiated closure. `depwarn` reads
# `Base.get_world_counter()`, i.e. the `@jl_world_counter` LLVM global, and
# Enzyme cannot find a shadow for that global (`EnzymeNoShadowError`), erroring
# every `weight` path on Enzyme forward and reverse. The warning is a pure
# logging side-effect with no numeric output and no tangent, so marking it
# inactive runs it on the primal unchanged (the depwarn still fires on normal
# calls) and keeps Enzyme from tracing into the world-counter read. The other
# backends (ForwardDiff/ReverseDiff/Mooncake) tolerate the call as-is, so this
# rule is only needed for Enzyme.
EnzymeRules.inactive(::typeof(_weight_deprecation), args...) = nothing

# `EnzymeRules.@easy_rule` expands into both the reverse-mode
# (`augmented_primal` / `reverse`) and forward-mode (`forward`) rules
# for `_gamma_cdf`. The analytical (dk, dθ, dx) come from
# `_gamma_cdf_value_and_partials` in `src/utils/gamma_ad.jl`, the
# single source-of-truth helper shared with the ChainRules rrule and
# the ForwardDiff Dual path. Routing `_gamma_cdf` through this rule
# avoids Enzyme differentiating `SpecialFunctions.gamma_inc` directly
# (the previous `@import_rrule` lift returned a
# `k`-partial that was ~8% off).

EnzymeRules.@easy_rule(_gamma_cdf(k::Real, θ::Real, x::Real),
    @setup(_vp=_gamma_cdf_value_and_partials(k, θ, x),
        dk=_vp[2],
        dθ=_vp[3],
        dx=_vp[4],),
    (dk, dθ, dx))

# Rule for `SpecialFunctions.gamma`, derivative `d/dx Γ(x) = Γ(x) ψ(x)`
# (`Ω` binds to the primal `Γ(x)`; same formula as the ChainRules
# `gamma` frule/rrule that Mooncake/ReverseDiff pick up). Enzyme's own
# `EnzymeSpecialFunctionsExt` ships no `gamma` rule and instead
# mis-lowers `gamma(x)` to the `loggamma` known-op, returning `ψ(x)` —
# wrong by a factor of `Γ(x)` in both modes (upstream bug). The
# analytical Gamma and Weibull `primarycensored_cdf` paths call
# `gamma(k + 1)` / `gamma(1 + 1/k)` outside the `_gamma_cdf` rule, so
# without this the pipeline shape-partial is wrong.
EnzymeRules.@easy_rule(gamma(x::Real), (Ω * digamma(x),))

end
