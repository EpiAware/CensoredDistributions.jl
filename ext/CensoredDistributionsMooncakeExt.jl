module CensoredDistributionsMooncakeExt

using CensoredDistributions: _gamma_cdf, _split_edge_name,
                             _is_positional_edge_name, _next_event_name,
                             _all_positional_event_names, _split_edge,
                             _ctor_has_check_args, _window_quantile
using Distributions: UnivariateDistribution
using Mooncake: Mooncake

# Lifts the `ChainRulesCore.rrule` and `ChainRulesCore.frule` defined in
# `CensoredDistributionsChainRulesCoreExt` into Mooncake's rule registry
# (default mode generates both reverse `rrule!!` and forward `frule!!`)
# for every scalar `Real` triple, so callers that pass mixed concrete
# types (e.g. `_gamma_cdf(k + 1, θ, t)` where `k + 1::Int`, a `Float32`
# parameter, or `BigFloat` for higher-precision testing) hit the
# explicit rule rather than falling back to Mooncake tracing the
# function body. The forward `frule` is what closes the gap: without it the
# generated `frule!!` calls `ChainRulesCore.frule`, gets `nothing`, and
# errors with `iterate(::Nothing)`.
Mooncake.@from_chainrules Mooncake.DefaultCtx Tuple{typeof(_gamma_cdf), Real, Real, Real}

# Event-name derivation is zero-derivative. A composed tree derives the requested
# event names from its constant edge labels (`compose` branch names, `Sequential`
# split names, positional defaults). These helpers do pure string work — an
# underscore split, a `startswith` + digit scan, `Symbol(:event_, i)`
# construction — and return Symbols / Bool / `Tuple{Symbol, Symbol}` / `nothing`.
# The names are constant with respect to the sampled parameters; only the delay
# parameters carry gradients. They are reached from `_flat_event_names` /
# `_row_event_vector` INSIDE the differentiated record `logpdf` (the composed-tree
# / `convolve_distributions` scoring path in a Turing `@model`), and Mooncake
# reverse cannot trace the underlying string foreigncalls (`startswith` calls
# `memcmp`), so the path breaks without these rules. Declaring each as a
# zero-adjoint primitive runs the primal unchanged and returns a zero cotangent,
# letting the gradient flow through the delay parameters with no behaviour change.
#
# NOTE: `_is_positional_edge_name` / `_all_positional_event_names` no
# longer compile a `Regex`. They previously matched `r"^step_\d+$"` etc., and
# `Base.compile(::Regex)` uses a try/catch Mooncake reverse cannot differentiate;
# even shielded here, that try/catch broke Mooncake reverse wherever the helper
# was reached UN-shielded (e.g. inlined into a traced caller), forcing the bdbv
# tutorial to AutoForwardDiff. They now do a plain `startswith` + ASCII-digit
# scan, so no `Regex` is compiled on the scored path; the zero-adjoint rules
# remain for the residual `startswith`/`split` foreigncalls and the convolve path.
#
# The outputs are freshly built values that do not alias any argument field, so
# the zero-adjoint precondition holds. `_next_event_name` mutates an integer
# `Ref` counter: the generated `rrule!!` runs the primal on the forward pass, so
# the mutation happens exactly as in the primal, and the integer counter is
# non-differentiable, so a zero cotangent is correct.
Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(_split_edge_name), Symbol}
# `_split_edge` is the DOTTED parameter-path splitter (`:a.b -> (:a, :b)`), reached
# from `event(d, name)` and `build_priors`/`composed_parameters_model` INSIDE the
# differentiated reconstruction (e.g. `event(delays, :index)` in the andv Choose
# model). It does `split(string(edge), '.')` — pointer-arithmetic string search
# (`findnext`/`thisind`/`codeunit`) that Mooncake reverse cannot trace, aborting
# with the uncatchable `sub_ptr intrinsic hit`, which forced the andv tutorial to
# `AutoForwardDiff`. The split is pure constant string -> `Tuple{Symbol...}` work
# on the CONSTANT edge labels (zero derivative; only the sampled delay params carry
# gradients), so a zero-adjoint primitive runs the primal unchanged and returns a
# zero cotangent, letting the gradient flow through the delay parameters with no
# behaviour change. Mirrors the underscored `_split_edge_name` rule above.
Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(_split_edge), Symbol}
Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(_is_positional_edge_name), Symbol}
Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(_all_positional_event_names), Tuple}
Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{
    typeof(_next_event_name), Base.RefValue{Int}}

# `_ctor_has_check_args(ctor, vals)` reports (via `hasmethod`) whether a leaf
# distribution constructor accepts a `check_args` keyword, so the DynamicPPL
# extension's leaf reconstruction can skip the argument check where supported. Its
# `hasmethod` lowers to a `jl_gf_invoke_lookup` foreigncall that Mooncake reverse
# on Julia LTS has no rule for (it aborts the nested-Resolve / Choose
# reconstruction there). The result is a `Bool` constant w.r.t. the sampled params
# (only the leaf params carry gradients), so a zero-adjoint primitive runs the
# primal unchanged and returns a zero cotangent, keeping the reconstruction AD-safe
# on every Julia version. The replaced runtime `try`/`catch` was likewise
# untraceable by Mooncake LTS; this reflection-shield is the AD-safe replacement.
Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{
    typeof(_ctor_has_check_args), Any, Tuple}

# `_window_quantile(comp, p)` returns a quadrature-window endpoint: an extreme
# quantile of an integration component used only to clamp an infinite bound to a
# finite one. It is computed on AD-stripped (primal) parameters, so the window is
# a non-differentiated hyperparameter (just WHERE to integrate), not a quantity
# carrying gradient. The `ChainRulesCore.@non_differentiable` mark in
# `CensoredDistributionsChainRulesCoreExt` covers reverse-mode AD generally, but
# Mooncake does not lift that mark automatically: without an explicit rule
# Mooncake traces `quantile` (e.g. `gamma_inc_inv` for a `Gamma` component) and
# returns a `NaN` shape derivative. Both modes need shielding, so
# `@zero_derivative` (no mode argument: covers both ForwardMode and ReverseMode)
# registers the primitive and generates a zero `frule!!` and a zero `rrule!!`,
# each returning the correct ZERO tangent/rdata for its argument types (a
# hand-written `NoRData` would be wrong for the distribution argument, whose
# rdata is a `NamedTuple` of its parameters). `@zero_adjoint` would cover reverse
# only, leaving a forward `Difference` whose subtrahend is the differentiated,
# unbounded-above integration component (its upper window bound is always a
# quantile of that component) returning a `NaN` on Mooncake forward. This also
# hardens the `Convolved` numeric path.
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(_window_quantile), UnivariateDistribution, Real}

end
