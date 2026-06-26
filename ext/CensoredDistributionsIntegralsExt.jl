module CensoredDistributionsIntegralsExt

# Optional Integrals.jl backend for the pluggable integration interface.
#
# Loading Integrals.jl adds an `integrate` method for any Integrals.jl
# algorithm, so a user can pass e.g. `QuadGKJL()` or `HCubatureJL()` as the
# `solver` to `primary_censored`/`convolved` and have that
# integral routed through an `IntegralProblem`/`solve`. Without this
# extension the package uses its lightweight default `GaussLegendre`
# solver and never touches Integrals.jl.
# See https://github.com/EpiAware/CensoredDistributions.jl/issues/208.

import CensoredDistributions: integrate
using Integrals: Integrals, IntegralProblem, solve
# `AbstractIntegralAlgorithm` is the common supertype of every Integrals.jl
# algorithm (QuadGKJL, HCubatureJL, ...). Reached through Integrals.jl's own
# SciMLBase so the extension needs no separate SciMLBase dependency.
const AbstractIntegralAlgorithm = Integrals.SciMLBase.AbstractIntegralAlgorithm

# Route a scalar integral through Integrals.jl. The integrand is wrapped to
# the `(u, p)` signature Integrals.jl expects (the package's integrands are
# already closed over their parameters, so `p` is unused). `solve(...)[1]`
# returns the scalar integral value, matching the `integrate` contract.
function integrate(
        solver::AbstractIntegralAlgorithm, f::F, lower, upper) where {F}
    prob = IntegralProblem((u, _) -> f(u), (lower, upper))
    return solve(prob, solver)[1]
end

end
