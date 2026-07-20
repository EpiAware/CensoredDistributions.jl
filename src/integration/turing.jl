# `as_turing`: adapt an upstream `ComposedDistributions.ComposedLogDensity`
# spec into a ready-to-sample Turing/DynamicPPL model, so one PPL-neutral spec
# drives both the LogDensityProblems route (`ComposedDistributions.logdensity`)
# and the Turing route. Declared here as a stub (no method until DynamicPPL is
# loaded); the model lives in `ext/CensoredDistributionsDynamicPPLExt.jl`, so
# this package stays Turing-free until that extension is triggered.
#
# Unlike CD's own (now-retired) internal composer, the model samples the
# estimated parameters generically off `ComposedDistributions.params_table`'s
# rows (`edge`, `param`, `prior`) rather than dispatching on composer node
# type: every estimated row becomes one named `tilde_assume!!` site
# `<prefix>.<edge>.<param>`, matching the dotted name upstream's own
# `chain_to_params`/`param_draws` read back. This is possible only because
# `params_table` already flattens the whole tree (any mix of `Sequential`,
# `Parallel`, `Resolve`, `Compete`, `Choose`, shared, pooled...) into one flat
# row set, so `as_turing` needs no per-composer-type dispatch of its own.

@doc "

Adapt an upstream `ComposedDistributions.ComposedLogDensity` spec into a
Turing/DynamicPPL model.

`as_turing(prob)` samples `prob`'s estimated flat parameters as named sites
(`<prefix>.<edge>.<param>`, one per `params_table(prob.dist)` row whose
`prior` column carries a spec), reconstructs the tree via
`ComposedDistributions.unflatten` and `ComposedDistributions.update`, and
scores `prob.data` with `prob.loglik` through `@addlogprob!`. The model's
log-joint therefore equals `ComposedDistributions.logdensity(prob, x)` at the
sampled `x`: both routes score one identical target. A fixed (non-uncertain)
parameter is never a sampled site, matching the LogDensityProblems route,
which excludes it from the flat vector.

The named sites are exactly what `ComposedDistributions.chain_to_params` /
`ComposedDistributions.param_draws` read back under the default `prefix =
:d`, so a fitted chain reads straight onto the template.

Convenience forms `as_turing(dist, priors, data)` and `as_turing(dist, data)`
mirror `ComposedDistributions.as_logdensity`: they assemble the spec first
(default priors read off the tree's `uncertain` specs), then adapt it.

This method is available only when `DynamicPPL` is loaded (the method lives in
a package extension); `ComposedDistributions` is a hard dependency of this
package (see the package `sources` note).

# Arguments
- `prob`: an assembled `ComposedDistributions.ComposedLogDensity` (from
  `as_logdensity`), or a `(dist, priors, data)` / `(dist, data)` triple to
  assemble one first.

# Keyword Arguments
- `prefix`: the submodel variable name the parameters are sampled under
  (default `:d`), matching `chain_to_params`/`param_draws`'s default.

# Examples
```@example
using CensoredDistributions, ComposedDistributions, Distributions
using DynamicPPL, Turing

tree = compose((onset_admit = uncertain(Gamma(2.0, 1.0);
    shape = LogNormal(log(2.0), 0.2)), admit_death = LogNormal(0.5, 0.4)))
data = [[0.5, 2.0], [1.0, 3.0]]
prob = ComposedDistributions.as_logdensity(tree, data)
m = as_turing(prob)
chain = sample(m, Prior(), 5; progress = false)
```

# See also
- `ComposedDistributions.as_logdensity`: assemble the spec.
- `ComposedDistributions.chain_to_params`, `ComposedDistributions.param_draws`:
  read a fitted chain back.
"
function as_turing end
