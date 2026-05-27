# Gradient benchmarks for CensoredDistributions across AD backends.
#
# Scenarios and the backend list are sourced from the `ADFixtures`
# path package at `test/ADFixtures`, which also drives the test suite
# (`test/ad/runtests.jl`) and the docs tutorial
# (`docs/src/getting-started/tutorials/ad-backends.jl`). This keeps
# the three AD surfaces in lock-step.
#
# Each (scenario, backend) pair is first smoke-tested to make sure the
# gradient is finite before being registered as a `@benchmarkable`, so
# known-broken combinations are silently omitted and the AirspeedVelocity
# suite can run unattended.

using ADTypes
using DifferentiationInterface
using ForwardDiff
using ReverseDiff
using Enzyme
using Mooncake
using ADFixtures
import DifferentiationInterfaceTest as DIT

SUITE["AD gradients"] = BenchmarkGroup()

for scen in ADFixtures.scenarios()
    SUITE["AD gradients"][scen.name] = BenchmarkGroup()
    for entry in ADFixtures.backends()
        grad_ok = try
            g = DifferentiationInterface.gradient(
                scen.f, entry.backend, scen.x)
            g isa AbstractVector && all(isfinite, g)
        catch
            false
        end
        grad_ok || continue

        f = scen.f
        backend = entry.backend
        x = scen.x
        SUITE["AD gradients"][scen.name][entry.name] = @benchmarkable DifferentiationInterface.gradient(
            $f, $backend, $x)
    end
end
