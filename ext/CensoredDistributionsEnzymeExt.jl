module CensoredDistributionsEnzymeExt

using CensoredDistributions: _gamma_cdf, _gamma_cdf_value_and_partials
using Enzyme.EnzymeRules: EnzymeRules
using SpecialFunctions: gamma, digamma

# `EnzymeRules.@easy_rule` expands into both the reverse-mode
# (`augmented_primal` / `reverse`) and forward-mode (`forward`) rules
# for `_gamma_cdf`. The analytical (dk, dθ, dx) come from
# `_gamma_cdf_value_and_partials` in `src/utils/gamma_ad.jl`, the
# single source-of-truth helper shared with the ChainRules rrule and
# the ForwardDiff Dual path. Routing `_gamma_cdf` through this rule
# avoids Enzyme differentiating `SpecialFunctions.gamma_inc` directly
# (the cause of #259 — the previous `@import_rrule` lift returned a
# `k`-partial that was ~8% off).

EnzymeRules.@easy_rule(_gamma_cdf(k::Real, θ::Real, x::Real),
    @setup(_vp=_gamma_cdf_value_and_partials(k, θ, x),
        dk=_vp[2],
        dθ=_vp[3],
        dx=_vp[4],),
    (dk, dθ, dx))

# `SpecialFunctions.gamma` has no correct Enzyme rule of its own. With
# only `EnzymeSpecialFunctionsExt` loaded, Enzyme mis-lowers `gamma(x)`
# to the `loggamma` known-op and returns `digamma(x)` for the
# derivative instead of `Γ(x) ψ(x)` — silently wrong by a factor of
# `Γ(x)` in both forward and reverse mode. The analytical Gamma and
# Weibull `primarycensored_cdf` paths call `gamma(k + 1)` and
# `gamma(1 + 1/k)` outside the `_gamma_cdf` rule, so without this the
# whole-pipeline `k`-partial is wrong (the cause of the scenario-level
# Enzyme failures in #263). The derivative is the standard
# `d/dx Γ(x) = Γ(x) ψ(x)`; `Ω` binds to the primal `Γ(x)`. See the
# ChainRules `gamma` frule/rrule for the same formula.
EnzymeRules.@easy_rule(gamma(x::Real), (Ω * digamma(x),))

end
