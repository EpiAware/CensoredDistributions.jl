module CensoredDistributionsReverseDiffExt

import CensoredDistributions: _gamma_cdf
using ReverseDiff: ReverseDiff, @grad_from_chainrules, TrackedReal

# Lift the `ChainRulesCore.rrule` defined in
# `CensoredDistributionsChainRulesCoreExt` into ReverseDiff's gradient
# table. Without this, ReverseDiff falls back to tracing through
# `_gamma_p_series`'s series loop — correct (every primitive is in
# DiffRules) but slower than calling our analytical rrule directly.
# Three independent overloads cover the mixed Tracked/untracked
# argument patterns that come up in practice (e.g. `_gamma_cdf(k, θ, t)`
# where only `k` is being differentiated through Turing).
@grad_from_chainrules _gamma_cdf(k::TrackedReal, θ::TrackedReal, x::TrackedReal)
@grad_from_chainrules _gamma_cdf(k::TrackedReal, θ::TrackedReal, x::Real)
@grad_from_chainrules _gamma_cdf(k::TrackedReal, θ::Real, x::TrackedReal)
@grad_from_chainrules _gamma_cdf(k::Real, θ::TrackedReal, x::TrackedReal)
@grad_from_chainrules _gamma_cdf(k::TrackedReal, θ::Real, x::Real)
@grad_from_chainrules _gamma_cdf(k::Real, θ::TrackedReal, x::Real)
@grad_from_chainrules _gamma_cdf(k::Real, θ::Real, x::TrackedReal)

end
