module CensoredDistributionsDensityInterfaceExt

# DensityInterface trait for the PPL-neutral log-density spec. Marks a
# `ComposedLogDensity` as a density object to the wider ecosystem; the
# evaluation is the core `CensoredDistributions.logdensity`. Loaded only when
# DensityInterface is available, keeping the core dependency-free.

using CensoredDistributions: CensoredDistributions, ComposedLogDensity
using DensityInterface: DensityInterface

# A `ComposedLogDensity` is a log-density function of a flat parameter vector.
function DensityInterface.DensityKind(::ComposedLogDensity)
    return DensityInterface.IsDensity()
end

# `logdensityof(prob, x)` is the assembled (unnormalised) log-posterior at the
# flat constrained vector `x`, delegating to the core scalar.
function DensityInterface.logdensityof(prob::ComposedLogDensity, x)
    return CensoredDistributions.logdensity(prob, x)
end

end
