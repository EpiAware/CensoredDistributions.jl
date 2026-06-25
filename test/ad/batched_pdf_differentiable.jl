# Guard that the batched `pdf`/`logpdf(::IntervalCensored, ::AbstractVector)`
# and `double_interval_censored` paths stay differentiable in the delay
# parameters on every backend. The boundary collection is marked
# non-differentiable (boundaries are functions of the constant lags, not the
# params), so this pins that the param gradient still flows through the cdf
# evaluations and is not silently zeroed.
#
# Cases are scored at specific lags or over a partial support so the reference
# gradient is genuinely non-zero. `sum(pdf(dic, 0:9))` with `upper=10,
# interval=1` covers the whole truncated support and is identically 1.0, whose
# gradient is the zero vector: a degenerate reference that "passes" even if AD
# were broken, so it is avoided here (#699, #701).
@testitem "batched pdf/logpdf is param-differentiable on all backends" tags=[
    :ad] begin
    using CensoredDistributions, Distributions
    using DifferentiationInterface: gradient, Constant
    using ADTypes
    using ForwardDiff, ReverseDiff, Enzyme, Mooncake

    θ = [1.0, 0.75]

    dic(θ) = double_interval_censored(LogNormal(θ[1], θ[2]);
        primary_event = Uniform(0.0, 1.0), upper = 10.0, interval = 1.0)

    # (name, f, obs). Each obs is a `Constant` context so the gradient is
    # w.r.t. the delay params only. Every case has a non-zero reference.
    cases = (
        ("IntervalCensored batched pdf (partial support)",
            (θ, o) -> sum(
                pdf(interval_censored(LogNormal(θ[1], θ[2]), 1.0), o)),
            collect(0.0:1.0:5.0)),
        ("IntervalCensored batched logpdf (scored lags)",
            (θ, o) -> sum(
                logpdf(interval_censored(LogNormal(θ[1], θ[2]), 1.0), o)),
            [1.0, 2.0, 4.0, 6.0]),
        ("DoubleIntervalCensored batched logpdf (scored lags)",
            (θ, o) -> sum(logpdf(dic(θ), o)),
            [1.0, 2.0, 4.0, 6.0]),
        ("DoubleIntervalCensored batched pdf (partial support)",
            (θ, o) -> sum(pdf(dic(θ), o)),
            collect(0.0:1.0:5.0)))

    reverse_backends = (
        ("ReverseDiff", AutoReverseDiff(compile = false)),
        ("Mooncake reverse", AutoMooncake(config = nothing)),
        ("Mooncake forward", AutoMooncakeForward()),
        ("Enzyme reverse",
            AutoEnzyme(mode = Enzyme.set_runtime_activity(Enzyme.Reverse))),
        ("Enzyme forward",
            AutoEnzyme(mode = Enzyme.set_runtime_activity(Enzyme.Forward))))

    for (cname, f, obs) in cases
        ref = gradient(f, AutoForwardDiff(), θ, Constant(obs))
        @test all(isfinite, ref)
        # Reference is non-zero: not a degenerate full-support sum.
        @test any(!iszero, ref)
        for (bname, backend) in reverse_backends
            g = gradient(f, backend, θ, Constant(obs))
            @test all(isfinite, g)
            @test any(!iszero, g)           # not silently zeroed
            @test isapprox(g, ref; rtol = 1e-6, atol = 1e-9)
        end
    end
end
