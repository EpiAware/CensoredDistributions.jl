module CensoredDistributionsLogDensityProblemsExt

# LogDensityProblems interface for the PPL-neutral spec
# (EpiAware/CensoredDistributions.jl#734): assemble the core codec + priors +
# scoring into a standard `LogDensityProblems` problem over the UNCONSTRAINED
# flat parameter vector, so a composed model is sampleable without Turing
# (AdvancedHMC / DynamicHMC / Pathfinder straight off this). The constrained
# log-density and the codec are core; the prior-driven transform is
# `BijectorsExt` (required for the unconstrained density). Gradients come from
# `LogDensityProblemsAD` / DifferentiationInterface over the existing AD
# backends. Loaded only when LogDensityProblems is available.

using CensoredDistributions: CensoredDistributions, ComposedLogDensity,
                             free_dimension, to_constrained
using LogDensityProblems: LogDensityProblems

# `dimension`: the FREE (estimated) parameter count. The transforms are
# univariate (one scalar prior per estimated row), so the unconstrained
# dimension equals the constrained one; fixed params are excluded from the
# vector, so this is the `params_table` row count only when nothing is fixed.
function LogDensityProblems.dimension(prob::ComposedLogDensity)
    return free_dimension(prob)
end

# Order-0 capability: this method evaluates the log-density only. A gradient is
# obtained by wrapping with `LogDensityProblemsAD.ADgradient(adtype, prob)`,
# which differentiates this evaluation through the existing AD backends
# (ForwardDiff / ReverseDiff / Mooncake), so no hand-written gradient is needed.
function LogDensityProblems.capabilities(::Type{<:ComposedLogDensity})
    LogDensityProblems.LogDensityOrder{0}()
end

# `logdensity(prob, z)` on the UNCONSTRAINED vector `z`: push to the constrained
# scale via the prior-driven transform (carrying the log-Jacobian), evaluate the
# core constrained log-density there, and add the Jacobian correction. This is
# the density a sampler needs (it samples on the unconstrained scale).
# `to_constrained` is provided by `BijectorsExt`; loading `LogDensityProblems`
# without `Bijectors` gives the dimension/capabilities but a transform error on
# evaluation, naming the missing dependency.
function LogDensityProblems.logdensity(
        prob::ComposedLogDensity, z::AbstractVector)
    x, logjac = to_constrained(prob, z)
    return CensoredDistributions.logdensity(prob, x) + logjac
end

end
