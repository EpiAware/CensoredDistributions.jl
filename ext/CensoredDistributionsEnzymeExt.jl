module CensoredDistributionsEnzymeExt

using CensoredDistributions: _gamma_cdf
using ChainRulesCore: ChainRulesCore
using Enzyme: Enzyme

# Lifts the `ChainRulesCore.rrule` defined in
# `CensoredDistributionsChainRulesCoreExt` into Enzyme via its
# `@import_rrule` macro (the same idea as Mooncake's
# `@from_chainrules`). Without this, Enzyme would try to differentiate
# the `_gamma_cdf` implementation directly — that path hits
# `SpecialFunctions.gamma_inc`, whose recursive series + DomainError
# branches Enzyme cannot lower cleanly.
#
# KNOWN ISSUE: Enzyme's `@import_rrule` machinery currently returns a
# wrong `k` (shape) partial when applied to this rrule — the `θ` and `x`
# partials match the other backends, but `dk` is incorrect by ~8%. The
# extension is shipped so the rule is registered and ready when the
# upstream interaction is fixed, but Enzyme gradients on `_gamma_cdf`
# (and anything routing through it) should not be trusted today. The
# AD test suite keeps Enzyme in `ad_test_broken_backends()` to surface
# this. Tracked in
# https://github.com/EpiAware/CensoredDistributions.jl/issues/259.
Enzyme.@import_rrule(typeof(_gamma_cdf), Real, Real, Real)

end
