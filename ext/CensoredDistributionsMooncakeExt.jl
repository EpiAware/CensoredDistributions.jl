module CensoredDistributionsMooncakeExt

using CensoredDistributions: _gamma_cdf, _split_edge_name,
                             _is_positional_edge_name, _next_event_name,
                             _all_positional_event_names
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
# NOTE (#409): `_is_positional_edge_name` / `_all_positional_event_names` no
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
Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(_is_positional_edge_name), Symbol}
Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(_all_positional_event_names), Tuple}
Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{
    typeof(_next_event_name), Base.RefValue{Int}}

end
