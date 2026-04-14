# Gradient benchmarks for CensoredDistributions across AD backends.
#
# Scenarios are built using `DifferentiationInterfaceTest.Scenario{:gradient}`
# so the benchmark definitions line up one-to-one with what
# `test/ad/ad_gradients.jl` exercises for correctness. The scenario list is
# intentionally duplicated between test and benchmark environments rather
# than shared via `include`, because crossing project-env boundaries would
# require either sharing a Project.toml or duplicating dependency lists.
#
# Each (scenario, backend) pair is first smoke-tested to make sure the
# gradient is finite before being registered as a `@benchmarkable`, so
# known-broken combinations (tracked in #217 and #218) are silently omitted
# and the AirspeedVelocity suite can run unattended.

using ADTypes
using DifferentiationInterface
using ForwardDiff
using ReverseDiff
using Zygote
import DifferentiationInterfaceTest as DIT

SUITE["AD gradients"] = BenchmarkGroup()

const AD_BENCH_BACKENDS = (
    (name = "ForwardDiff", backend = AutoForwardDiff()),
    (name = "ReverseDiff", backend = AutoReverseDiff()),
    (name = "Zygote", backend = AutoZygote())
)

const AD_BENCH_SCENARIOS = let
    obs = [0.5, 1.2, 2.5, 3.8, 5.1]
    obs_int = [0.0, 1.0, 2.0, 3.0, 4.0]
    boundaries = [0.0, 1.5, 3.0, 5.0, 10.0]
    obs_double = [1.0, 2.0, 3.0, 4.0, 5.0]

    # Analytical primary-censored scenarios: loop over the distributions
    # with a closed-form CDF through Uniform primary events.
    analytical_specs = (
        (name = "PrimaryCensored Gamma+Uniform analytical",
            ctor = Gamma, θ₀ = [2.0, 1.5]),
        (name = "PrimaryCensored LogNormal+Uniform analytical",
            ctor = LogNormal, θ₀ = [1.0, 0.75]),
        (name = "PrimaryCensored Weibull+Uniform analytical",
            ctor = Weibull, θ₀ = [2.0, 1.5])
    )
    # Numerical primary-censored scenarios: same distributions, but force
    # the numerical integration path. `Gamma` is omitted for the same
    # `_gamma_inc` Dual-dispatch reason that affects IntervalCensored+Gamma.
    numerical_specs = (
        (name = "PrimaryCensored LogNormal+Uniform numerical",
            ctor = LogNormal, θ₀ = [1.0, 0.75]),
        (name = "PrimaryCensored Weibull+Uniform numerical",
            ctor = Weibull, θ₀ = [2.0, 1.5])
    )

    scenarios = DIT.Scenario[]
    for spec in analytical_specs
        f = let ctor = spec.ctor
            θ -> sum(
                x -> logpdf(
                    primary_censored(ctor(θ[1], θ[2]), Uniform(0.0, 1.0)), x),
                obs)
        end
        push!(scenarios,
            DIT.Scenario{:gradient, :out}(f, spec.θ₀; name = spec.name))
    end
    for spec in numerical_specs
        f = let ctor = spec.ctor
            θ -> sum(
                x -> logpdf(
                    primary_censored(
                        ctor(θ[1], θ[2]), Uniform(0.0, 1.0);
                        force_numeric = true),
                    x),
                obs)
        end
        push!(scenarios,
            DIT.Scenario{:gradient, :out}(f, spec.θ₀; name = spec.name))
    end

    ic_lognormal(θ) = sum(
        x -> logpdf(interval_censored(LogNormal(θ[1], θ[2]), 1.0), x),
        obs_int)
    push!(scenarios,
        DIT.Scenario{:gradient, :out}(
            ic_lognormal, [1.0, 0.75];
            name = "IntervalCensored LogNormal regular"))

    ic_gamma(θ) = sum(
        x -> logpdf(interval_censored(Gamma(θ[1], θ[2]), boundaries), x),
        [0.5, 2.0, 4.0, 7.0])
    push!(scenarios,
        DIT.Scenario{:gradient, :out}(
            ic_gamma, [2.0, 1.5];
            name = "IntervalCensored Gamma arbitrary"))

    dic_lognormal(θ) = sum(
        x -> logpdf(
            double_interval_censored(
                LogNormal(θ[1], θ[2]);
                primary_event = Uniform(0.0, 1.0),
                upper = 10.0,
                interval = 1.0),
            x),
        obs_double)
    push!(scenarios,
        DIT.Scenario{:gradient, :out}(
            dic_lognormal, [1.0, 0.75];
            name = "DoubleIntervalCensored LogNormal"))

    scenarios
end

for scen in AD_BENCH_SCENARIOS
    SUITE["AD gradients"][scen.name] = BenchmarkGroup()
    for entry in AD_BENCH_BACKENDS
        # Verify the backend produces a finite gradient for this scenario
        # before registering a benchmark; known-broken combinations
        # (see #217, #218) are silently skipped.
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
