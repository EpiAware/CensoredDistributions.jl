# Gradient benchmarks for CensoredDistributions across AD backends.
#
# Each case defines a scalar loss `loss(θ) = sum(logpdf.(construct(θ), obs))`
# and benchmarks `DifferentiationInterface.gradient` for each supported
# backend. Cases that are known to fail for a given backend are omitted so
# the benchmark suite can run unattended; the test suite in
# `test/ad/ad_gradients.jl` still reports those as `@test_broken`.

using ADTypes
using DifferentiationInterface
using ForwardDiff
using ReverseDiff
using Zygote

SUITE["AD gradients"] = BenchmarkGroup()

# Backend, eligible-case predicate. The predicate receives a case name and
# returns `true` if the backend is expected to work for that case.
const AD_BENCH_BACKENDS = (
    (name = "ForwardDiff", backend = AutoForwardDiff(),
        eligible = name -> name != "IntervalCensored Gamma arbitrary"),
    (name = "ReverseDiff", backend = AutoReverseDiff(),
        eligible = name -> true),
    (name = "Zygote", backend = AutoZygote(),
        eligible = name -> true)
)

# (name, loss, θ₀)
const AD_BENCH_CASES = let
    obs = [0.5, 1.2, 2.5, 3.8, 5.1]
    obs_int = [0.0, 1.0, 2.0, 3.0, 4.0]
    boundaries = [0.0, 1.5, 3.0, 5.0, 10.0]

    pc_ana_gamma(θ) = sum(
        x -> logpdf(primary_censored(Gamma(θ[1], θ[2]), Uniform(0.0, 1.0)), x),
        obs)
    pc_ana_lognormal(θ) = sum(
        x -> logpdf(primary_censored(LogNormal(θ[1], θ[2]), Uniform(0.0, 1.0)), x),
        obs)
    pc_num_lognormal(θ) = sum(
        x -> logpdf(
            primary_censored(
                LogNormal(θ[1], θ[2]), Uniform(0.0, 1.0); force_numeric = true),
            x),
        obs)
    ic_lognormal(θ) = sum(
        x -> logpdf(interval_censored(LogNormal(θ[1], θ[2]), 1.0), x),
        obs_int)
    ic_gamma_arb(θ) = sum(
        x -> logpdf(interval_censored(Gamma(θ[1], θ[2]), boundaries), x),
        [0.5, 2.0, 4.0, 7.0])
    dic_lognormal(θ) = sum(
        x -> logpdf(
            double_interval_censored(
                LogNormal(θ[1], θ[2]);
                primary_event = Uniform(0.0, 1.0),
                upper = 10.0,
                interval = 1.0),
            x),
        [1.0, 2.0, 3.0, 4.0, 5.0])

    (
        (name = "PrimaryCensored Gamma+Uniform analytical",
            loss = pc_ana_gamma, θ₀ = [2.0, 1.5]),
        (name = "PrimaryCensored LogNormal+Uniform analytical",
            loss = pc_ana_lognormal, θ₀ = [1.0, 0.75]),
        (name = "PrimaryCensored LogNormal+Uniform numerical",
            loss = pc_num_lognormal, θ₀ = [1.0, 0.75]),
        (name = "IntervalCensored LogNormal regular",
            loss = ic_lognormal, θ₀ = [1.0, 0.75]),
        (name = "IntervalCensored Gamma arbitrary",
            loss = ic_gamma_arb, θ₀ = [2.0, 1.5]),
        (name = "DoubleIntervalCensored LogNormal",
            loss = dic_lognormal, θ₀ = [1.0, 0.75])
    )
end

for case in AD_BENCH_CASES
    SUITE["AD gradients"][case.name] = BenchmarkGroup()
    for entry in AD_BENCH_BACKENDS
        entry.eligible(case.name) || continue
        # Verify the backend actually produces a finite gradient before
        # registering the benchmark; skip quietly otherwise.
        grad_ok = try
            g = DifferentiationInterface.gradient(
                case.loss, entry.backend, case.θ₀)
            g isa AbstractVector && all(isfinite, g)
        catch
            false
        end
        grad_ok || continue

        loss = case.loss
        backend = entry.backend
        θ₀ = case.θ₀
        SUITE["AD gradients"][case.name][entry.name] = @benchmarkable DifferentiationInterface.gradient(
            $loss, $backend, $θ₀)
    end
end
