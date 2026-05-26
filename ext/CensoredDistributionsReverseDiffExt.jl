module CensoredDistributionsReverseDiffExt

import CensoredDistributions: _gamma_cdf
using ReverseDiff: ReverseDiff, @grad_from_chainrules, TrackedReal

# Lift the `ChainRulesCore.rrule` defined in
# `CensoredDistributionsChainRulesCoreExt` into ReverseDiff's gradient
# table. Without this, ReverseDiff falls back to forward-mode through
# `gamma_inc` (no `TrackedReal` method, errors) or, depending on the
# call site, traces through the function body — slower than calling
# our analytical rrule directly even when it works.
# Seven overloads cover every non-trivial Tracked/untracked subset of
# the three arguments — required because `@grad_from_chainrules` is
# signature-specific, not abstract; mixed patterns (e.g.
# `_gamma_cdf(k, θ, t)` where only `k` is being differentiated through
# Turing) need their own declaration to hit the explicit rule rather
# than the default forward-through-ReverseDiff fallback.
@grad_from_chainrules _gamma_cdf(k::TrackedReal, θ::TrackedReal, x::TrackedReal)
@grad_from_chainrules _gamma_cdf(k::TrackedReal, θ::TrackedReal, x::Real)
@grad_from_chainrules _gamma_cdf(k::TrackedReal, θ::Real, x::TrackedReal)
@grad_from_chainrules _gamma_cdf(k::Real, θ::TrackedReal, x::TrackedReal)
@grad_from_chainrules _gamma_cdf(k::TrackedReal, θ::Real, x::Real)
@grad_from_chainrules _gamma_cdf(k::Real, θ::TrackedReal, x::Real)
@grad_from_chainrules _gamma_cdf(k::Real, θ::Real, x::TrackedReal)

end
