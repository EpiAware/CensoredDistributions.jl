module CensoredDistributionsReverseDiffExt

import CensoredDistributions: _gamma_cdf, _primal
using ReverseDiff: ReverseDiff, @grad_from_chainrules, TrackedReal

# Strip a ReverseDiff `TrackedReal` to its primal value for the
# non-differentiable quadrature-window quantile (`_finite_window` in
# `src/distributions/Convolved.jl`). Reading `.value` off the
# tape entry does not record an operation, so the window endpoint stays a
# constant — exactly the intended behaviour for an integration bound.
_primal(x::TrackedReal) = _primal(ReverseDiff.value(x))

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
