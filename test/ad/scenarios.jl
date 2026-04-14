# Shared AD gradient scenarios for CensoredDistributions.
#
# This file is included from:
#   - `test/ad/runtests.jl`
#   - `benchmark/src/ad_gradients.jl`
#   - `docs/src/getting-started/tutorials/ad-backends.jl`
#
# The including file is responsible for loading these packages before the
# include:
#   CensoredDistributions, Distributions, ADTypes, DifferentiationInterface,
#   DifferentiationInterfaceTest as DIT, ForwardDiff, ReverseDiff, Enzyme,
#   Mooncake.
#
# A fixed-step central-difference reference (`simple_fd`) is used rather than
# an adaptive finite-difference scheme because `PrimaryCensored.logpdf`
# performs its own internal numerical differentiation with a hardcoded step,
# making the function non-smooth on scales smaller than ~1e-3. Adaptive FD
# methods (e.g. `FiniteDifferences.central_fdm`) are defeated by that
# internal non-smoothness, so we pin `h = 1e-2`.

"""
    simple_fd(f, θ; h = 1e-2)

Central-difference gradient with a fixed step (see file header for rationale).
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
    ad_backends()

Return a `Vector` of `(name::String, backend::ADTypes.AbstractADType)` named
tuples for every AD backend the package targets. The returned list is the
union of `ad_working_backends()` and `ad_broken_backends()`.
"""
function ad_backends()
    return vcat(ad_working_backends(), ad_broken_backends())
end

"""
    ad_working_backends()

AD backends that compute correct gradients on the non-broken scenarios. The
test suite feeds this list to `DIT.test_differentiation`.
"""
function ad_working_backends()
    return [
        (name = "ForwardDiff", backend = AutoForwardDiff()),
        (name = "ReverseDiff (tape)", backend = AutoReverseDiff(compile = false))
    ]
end

"""
    ad_broken_backends()

AD backends that currently fail for every scenario. The test suite feeds
this list to `check_broken` so known-broken state surfaces as `@test_broken`
and any unexpected pass is caught. Tracked in
https://github.com/EpiAware/CensoredDistributions.jl/issues/225.
"""
function ad_broken_backends()
    return [
        (name = "Enzyme forward", backend = AutoEnzyme(mode = Enzyme.Forward)),
        (name = "Enzyme reverse", backend = AutoEnzyme(mode = Enzyme.Reverse)),
        (name = "Mooncake", backend = AutoMooncake(config = nothing))
    ]
end

"""
    ad_scenarios(; with_reference::Bool = false)

Return a `Vector{DIT.Scenario{:gradient, :out}}`. When `with_reference = true`,
each scenario's `res1` is populated with `simple_fd`. Tests pass `true`;
benchmarks and docs pass `false` (default).
"""
function ad_scenarios(; with_reference::Bool = false)
    obs = [0.5, 1.2, 2.5, 3.8, 5.1]
    obs_int = [0.0, 1.0, 2.0, 3.0, 4.0]
    boundaries = [0.0, 1.5, 3.0, 5.0, 10.0]
    obs_int_gamma = [0.5, 2.0, 4.0, 7.0]
    obs_double = [1.0, 2.0, 3.0, 4.0, 5.0]

    analytical_specs = (
        (name = "Gamma", ctor = Gamma, θ₀ = [2.0, 1.5]),
        (name = "LogNormal", ctor = LogNormal, θ₀ = [1.0, 0.75]),
        (name = "Weibull", ctor = Weibull, θ₀ = [2.0, 1.5])
    )

    numerical_specs = (
        (name = "LogNormal", ctor = LogNormal, θ₀ = [1.0, 0.75]),
        (name = "Weibull", ctor = Weibull, θ₀ = [2.0, 1.5]),
        (name = "Gamma", ctor = Gamma, θ₀ = [2.0, 1.5])
    )

    scenarios = DIT.Scenario{:gradient, :out}[]

    function _push!(name, f, θ₀)
        res1 = with_reference ? simple_fd(f, θ₀) : nothing
        push!(scenarios,
            res1 === nothing ?
            DIT.Scenario{:gradient, :out}(f, θ₀; name = name) :
            DIT.Scenario{:gradient, :out}(f, θ₀; res1 = res1, name = name))
    end

    for spec in analytical_specs
        f = let ctor = spec.ctor
            θ -> sum(
                x -> logpdf(
                    primary_censored(ctor(θ[1], θ[2]), Uniform(0.0, 1.0)), x),
                obs)
        end
        _push!("PrimaryCensored $(spec.name)+Uniform analytical", f, spec.θ₀)
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
        _push!("PrimaryCensored $(spec.name)+Uniform numerical", f, spec.θ₀)
    end

    f_int_lognormal = θ -> sum(
        x -> logpdf(interval_censored(LogNormal(θ[1], θ[2]), 1.0), x),
        obs_int)
    _push!("IntervalCensored LogNormal regular", f_int_lognormal, [1.0, 0.75])

    f_int_gamma = θ -> sum(
        x -> logpdf(interval_censored(Gamma(θ[1], θ[2]), boundaries), x),
        obs_int_gamma)
    _push!("IntervalCensored Gamma arbitrary", f_int_gamma, [2.0, 1.5])

    f_double = θ -> sum(
        x -> logpdf(
            double_interval_censored(
                LogNormal(θ[1], θ[2]);
                primary_event = Uniform(0.0, 1.0),
                upper = 10.0,
                interval = 1.0),
            x),
        obs_double)
    _push!("DoubleIntervalCensored LogNormal", f_double, [1.0, 0.75])

    return scenarios
end

"""
    ad_broken_scenario_names()

Return a `Vector{String}` listing the `.name` of every scenario that is
expected to FAIL for every backend due to `_gamma_inc` Dual dispatch (#217).
Used by the test suite to partition working vs broken scenarios.
"""
function ad_broken_scenario_names()
    return [
        "PrimaryCensored Gamma+Uniform numerical",
        "IntervalCensored Gamma arbitrary"
    ]
end
