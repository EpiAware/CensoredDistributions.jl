# Automatic differentiation (AD) gradient tests for CensoredDistributions.
#
# Correctness is driven by `DifferentiationInterfaceTest.test_differentiation`
# against a finite-difference reference stored in each `Scenario`'s `res1`
# field. A fixed central-difference step (h = 1e-2) is used because
# `PrimaryCensored.logpdf` performs its own internal numerical
# differentiation with a hardcoded step, which makes the function non-smooth
# on scales smaller than ~1e-3 and defeats adaptive FD methods.
#
# The scenario list is built by looping over the distributions that have an
# analytical `primary_censored` CDF (Gamma, LogNormal, Weibull with a Uniform
# primary event), first with the analytical path and then with
# `force_numeric = true`, and is extended with the remaining censoring
# flavours (`interval_censored`, `double_interval_censored`).
#
# ForwardDiff and ReverseDiff are exercised as first-class backends. Zygote
# currently fails on every censored `logpdf` (tracked in #218) so it is run
# separately inside a `@test_broken` loop; an unexpected pass there will
# surface as a test error. The `IntervalCensored` + `Gamma` case is
# similarly broken for all backends (tracked in #217) and is moved to the
# `@test_broken` loop.

@testsnippet ADGradientScenarios begin
    using CensoredDistributions
    using Distributions
    using ADTypes
    using DifferentiationInterface
    using ForwardDiff
    using ReverseDiff
    using Zygote
    import DifferentiationInterfaceTest as DIT

    """
        simple_fd(f, θ; h = 1e-2)

    Central-difference gradient using a fixed step. A fixed step is needed
    because `PrimaryCensored.logpdf` already performs numerical
    differentiation internally with a hardcoded step, which makes the
    function non-smooth on scales smaller than ~1e-3 and defeats adaptive
    FD methods such as `FiniteDifferences.central_fdm`.
    """
    function simple_fd(f, θ::AbstractVector; h = 1e-2)
        g = similar(θ, float(eltype(θ)))
        for i in eachindex(θ)
            e = zero(θ)
            e[i] = one(eltype(θ))
            g[i] = (f(θ .+ h .* e) - f(θ .- h .* e)) / (2h)
        end
        return g
    end

    """
        make_scenario(name, f, θ₀)

    Construct a `DIT.Scenario{:gradient, :out}` with a finite-difference
    reference gradient precomputed via `simple_fd`.
    """
    function make_scenario(name::AbstractString, f, θ₀::AbstractVector)
        ref = simple_fd(f, θ₀)
        return DIT.Scenario{:gradient, :out}(f, θ₀; res1 = ref, name = name)
    end

    # Backends that are expected to work across the supported scenarios.
    # ForwardDiff is the primary backend; ReverseDiff is exercised at the
    # same level because it also succeeds on every case except
    # `IntervalCensored` + `Gamma` (see #217), which is already routed to
    # the `@test_broken` bucket.
    const WORKING_BACKENDS = [AutoForwardDiff(), AutoReverseDiff()]

    # Backends that currently fail on every censored `logpdf` and are run
    # through a `@test_broken` loop. Unexpected passes still surface as
    # errors via `Test.jl`'s broken-test machinery.
    const BROKEN_BACKENDS = [AutoZygote()]

    # Distributions with an analytical primary-censored CDF, paired with a
    # plausible parameter vector for scenario construction. Used for the
    # analytical path tests.
    const ANALYTICAL_PRIMARY_DISTS = (
        (name = "Gamma", ctor = Gamma, θ₀ = [2.0, 1.5]),
        (name = "LogNormal", ctor = LogNormal, θ₀ = [1.0, 0.75]),
        (name = "Weibull", ctor = Weibull, θ₀ = [2.0, 1.5])
    )

    # Distributions exercised through the `force_numeric = true` path.
    # `Gamma` is omitted here because the numerical integration ends up
    # calling `cdf(Gamma{Dual}, Float64)`, which hits the same
    # `SpecialFunctions._gamma_inc` Dual-dispatch gap tracked in #217.
    const NUMERICAL_PRIMARY_DISTS = (
        (name = "LogNormal", ctor = LogNormal, θ₀ = [1.0, 0.75]),
        (name = "Weibull", ctor = Weibull, θ₀ = [2.0, 1.5])
    )

    const PRIMARY_OBS = [0.5, 1.2, 2.5, 3.8, 5.1]
    const INTERVAL_OBS = [0.0, 1.0, 2.0, 3.0, 4.0]

    """
        check_broken(scenarios, backend)

    Run `@test_broken` on each (scenario, backend) pair so that known-broken
    combinations do not fail the suite, but unexpected passes still error.
    Mirrors what `DIT.test_differentiation` would do for a working scenario
    but wrapped in broken semantics.
    """
    function check_broken(scenarios, backend)
        for scen in scenarios
            ok = try
                g = DifferentiationInterface.gradient(
                    scen.f, backend, scen.x)
                g isa AbstractVector && all(isfinite, g) &&
                    isapprox(g, scen.res1; rtol = 5e-2, atol = 1e-6)
            catch
                false
            end
            @test_broken ok
        end
    end
end

@testitem "AD gradient: PrimaryCensored analytical paths" tags=[:ad] setup=[ADGradientScenarios] begin
    using CensoredDistributions
    using Distributions
    import DifferentiationInterfaceTest as DIT

    scenarios=DIT.Scenario[]
    for spec in ANALYTICAL_PRIMARY_DISTS
        f=let ctor=spec.ctor
            θ->sum(
                x->logpdf(
                    primary_censored(ctor(θ[1], θ[2]), Uniform(0.0, 1.0)), x),
                PRIMARY_OBS)
        end
        push!(scenarios,
            make_scenario("$(spec.name)+Uniform analytical", f, spec.θ₀))
    end

    DIT.test_differentiation(
        WORKING_BACKENDS, scenarios;
        correctness = true,
        type_stability = :none,
        logging = false,
        rtol = 5e-2,
        atol = 1e-6
    )
    check_broken(scenarios, AutoZygote())
end

@testitem "AD gradient: PrimaryCensored numerical paths" tags=[:ad] setup=[ADGradientScenarios] begin
    using CensoredDistributions
    using Distributions
    import DifferentiationInterfaceTest as DIT

    scenarios=DIT.Scenario[]
    for spec in NUMERICAL_PRIMARY_DISTS
        f=let ctor=spec.ctor
            θ->sum(
                x->logpdf(
                    primary_censored(
                        ctor(θ[1], θ[2]), Uniform(0.0, 1.0);
                        force_numeric = true),
                    x),
                PRIMARY_OBS)
        end
        push!(scenarios,
            make_scenario("$(spec.name)+Uniform numerical", f, spec.θ₀))
    end

    DIT.test_differentiation(
        WORKING_BACKENDS, scenarios;
        correctness = true,
        type_stability = :none,
        logging = false,
        rtol = 5e-2,
        atol = 1e-6
    )
    check_broken(scenarios, AutoZygote())
end

# PrimaryCensored Gamma with `force_numeric = true` is broken for the same
# underlying reason as IntervalCensored + Gamma: the numerical convolution
# evaluates `cdf(Gamma{Dual}, Float64)`, which hits the missing
# `SpecialFunctions._gamma_inc` Dual method. Tracked in
# https://github.com/EpiAware/CensoredDistributions.jl/issues/217.
@testitem "AD gradient: PrimaryCensored Gamma numerical (known broken)" tags=[:ad] setup=[ADGradientScenarios] begin
    using CensoredDistributions
    using Distributions
    import DifferentiationInterfaceTest as DIT

    f=θ->sum(
        x->logpdf(
            primary_censored(
                Gamma(θ[1], θ[2]), Uniform(0.0, 1.0); force_numeric = true),
            x),
        PRIMARY_OBS)
    scenarios=[make_scenario("Gamma+Uniform numerical", f, [2.0, 1.5])]

    for backend in WORKING_BACKENDS
        check_broken(scenarios, backend)
    end
    check_broken(scenarios, AutoZygote())
end

@testitem "AD gradient: IntervalCensored LogNormal regular" tags=[:ad] setup=[ADGradientScenarios] begin
    using CensoredDistributions
    using Distributions
    import DifferentiationInterfaceTest as DIT

    f=θ->sum(
        x->logpdf(interval_censored(LogNormal(θ[1], θ[2]), 1.0), x),
        INTERVAL_OBS)
    scenarios=[make_scenario("LogNormal regular", f, [1.0, 0.75])]

    DIT.test_differentiation(
        WORKING_BACKENDS, scenarios;
        correctness = true,
        type_stability = :none,
        logging = false,
        rtol = 5e-2,
        atol = 1e-6
    )
    check_broken(scenarios, AutoZygote())
end

@testitem "AD gradient: DoubleIntervalCensored LogNormal" tags=[:ad] setup=[ADGradientScenarios] begin
    using CensoredDistributions
    using Distributions
    import DifferentiationInterfaceTest as DIT

    f=θ->sum(
        x->logpdf(
            double_interval_censored(
                LogNormal(θ[1], θ[2]);
                primary_event = Uniform(0.0, 1.0),
                upper = 10.0,
                interval = 1.0),
            x),
        [1.0, 2.0, 3.0, 4.0, 5.0])
    scenarios=[make_scenario("LogNormal", f, [1.0, 0.75])]

    DIT.test_differentiation(
        WORKING_BACKENDS, scenarios;
        correctness = true,
        type_stability = :none,
        logging = false,
        rtol = 5e-2,
        atol = 1e-6
    )
    check_broken(scenarios, AutoZygote())
end

# IntervalCensored + Gamma fails through ForwardDiff and ReverseDiff because
# `cdf(Gamma{Dual}, Float64)` dispatches into
# `SpecialFunctions._gamma_inc(a, x, ind)` which lacks a method accepting a
# `Dual` first argument alongside a non-dual `x`. Tracked in
# https://github.com/EpiAware/CensoredDistributions.jl/issues/217.
@testitem "AD gradient: IntervalCensored Gamma (known broken)" tags=[:ad] setup=[ADGradientScenarios] begin
    using CensoredDistributions
    using Distributions
    import DifferentiationInterfaceTest as DIT

    boundaries=[0.0, 1.5, 3.0, 5.0, 10.0]
    obs=[0.5, 2.0, 4.0, 7.0]
    f=θ->sum(
        x->logpdf(interval_censored(Gamma(θ[1], θ[2]), boundaries), x),
        obs)
    scenarios=[make_scenario("Gamma arbitrary", f, [2.0, 1.5])]

    for backend in WORKING_BACKENDS
        check_broken(scenarios, backend)
    end
    check_broken(scenarios, AutoZygote())
end

@testitem "AD gradient: logpdf is scalar-differentiable in x" tags=[:ad] setup=[ADGradientScenarios] begin
    using CensoredDistributions
    using Distributions
    import DifferentiationInterfaceTest as DIT

    # Scalar-input sanity check: `logpdf(d, x)` should be differentiable
    # w.r.t. `x` at interior points where the density is smooth. This mirrors
    # how AD is exercised when `x` is a latent parameter inside a Turing
    # model.
    d=primary_censored(LogNormal(1.0, 0.75), Uniform(0.0, 1.0))
    f=x->logpdf(d, x[1])
    scenarios=[make_scenario("scalar-x", f, [2.5])]

    DIT.test_differentiation(
        WORKING_BACKENDS, scenarios;
        correctness = true,
        type_stability = :none,
        logging = false,
        rtol = 5e-2,
        atol = 1e-6
    )
    check_broken(scenarios, AutoZygote())
end
