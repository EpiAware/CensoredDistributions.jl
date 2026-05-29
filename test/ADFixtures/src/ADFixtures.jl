"""
    ADFixtures

Shared AD gradient scenarios and backend metadata for CensoredDistributions.
Used by `test/ad/runtests.jl`, `benchmark/src/ad_gradients.jl`, and the
docs tutorial `docs/src/getting-started/tutorials/ad-backends.jl`.

The reference gradient is computed with `ForwardDiff`. `PrimaryCensored.logpdf`
internally finite-differences the CDF with a hardcoded step `h=1e-8`, which
defeats adaptive finite-difference references (e.g. `central_fdm(5, 1)`
disagrees with all AD backends by ~10% on Weibull analytical scenarios).
ForwardDiff's Dual-number propagation through that internal FD gives the
exact derivative of the package's own logpdf and matches every other working
backend (ReverseDiff, Mooncake reverse/forward) to ~1e-6.
"""
module ADFixtures

# `__precompile__(false)` skips the precompile cache so the Mooncake
# load chain (specifically `MooncakeAllocCheckExt.__init__` evaluating
# into the already-closed `AllocCheck` module) doesn't break the
# package build on CI. Negligible cost — this module is only loaded by
# the AD test, benchmark, and docs scripts, each of which already pays
# for Mooncake/Enzyme load time elsewhere.
__precompile__(false)

using CensoredDistributions
using Distributions: Distributions, Gamma, LogNormal, Weibull, Uniform, logpdf
using ADTypes: ADTypes, AutoForwardDiff, AutoReverseDiff, AutoMooncake,
               AutoMooncakeForward, AutoEnzyme
using DifferentiationInterface: DifferentiationInterface
import ForwardDiff, ReverseDiff, Mooncake, Enzyme
import DifferentiationInterfaceTest as DIT

export scenarios, backends, working_backends, broken_backends,
       broken_scenario_names, backend_broken_scenarios

_reference(f, θ) = DifferentiationInterface.gradient(f, AutoForwardDiff(), θ)

"""
    working_backends()

AD backends that compute correct gradients on at least one scenario.
"""
function working_backends()
    return [
        (name = "ForwardDiff", backend = AutoForwardDiff()),
        (name = "ReverseDiff (tape)",
            backend = AutoReverseDiff(compile = false)),
        (name = "Mooncake reverse",
            backend = AutoMooncake(config = nothing)),
        (name = "Mooncake forward", backend = AutoMooncakeForward())
    ]
end

"""
    broken_backends()

AD backends that currently fail for every scenario. Tracked in
[#225](https://github.com/EpiAware/CensoredDistributions.jl/issues/225).
"""
function broken_backends()
    return [
        (name = "Enzyme forward",
            backend = AutoEnzyme(mode = Enzyme.Forward)),
        (name = "Enzyme reverse",
            backend = AutoEnzyme(mode = Enzyme.Reverse))
    ]
end

"""
    backends()

Union of [`working_backends`](@ref) and [`broken_backends`](@ref).
"""
backends() = vcat(working_backends(), broken_backends())

"""
    broken_scenario_names()

Scenario names that fail for every backend (universal scenario-level
failures). Returns a `Vector{String}`.
"""
function broken_scenario_names()
    # No scenario fails on every backend. `IntervalCensored Gamma
    # arbitrary` previously did (#217/#257): it routed through stock
    # `Distributions.cdf(Gamma, x)` → `gamma_inc`, which no AD backend
    # covers. It now routes through the `_gamma_cdf` helper, so it works
    # everywhere except Mooncake forward (the shared #270 gap, listed in
    # `backend_broken_scenarios`).
    return String[]
end

"""
    backend_broken_scenarios()

Per-backend scenario names that fail even though the backend works on
other scenarios. Returns a `Dict{String, Set{String}}` keyed on the
backend `name` from [`working_backends`](@ref).

"""
function backend_broken_scenarios()
    return Dict{String, Set{String}}(
        "ForwardDiff" => Set{String}(),
        "ReverseDiff (tape)" => Set{String}(),
        "Mooncake reverse" => Set{String}(),
        "Mooncake forward" => Set{String}()
    )
end

"""
    scenarios(; with_reference::Bool = false)

Return a `Vector{DIT.Scenario{:gradient, :out}}`. When
`with_reference = true`, each scenario's `res1` is populated with a
ForwardDiff reference gradient (see module docstring for rationale).
"""
function scenarios(; with_reference::Bool = false)
    obs = [0.5, 1.2, 2.5, 3.8, 5.1]
    obs_int = [0.0, 1.0, 2.0, 3.0, 4.0]
    boundaries = [0.0, 1.5, 3.0, 5.0, 10.0]
    obs_int_gamma = [0.5, 2.0, 4.0, 7.0]
    obs_double = [1.0, 2.0, 3.0, 4.0, 5.0]

    primary_specs = (
        (name = "Gamma", ctor = Gamma, θ₀ = [2.0, 1.5]),
        (name = "LogNormal", ctor = LogNormal, θ₀ = [1.0, 0.75]),
        (name = "Weibull", ctor = Weibull, θ₀ = [2.0, 1.5])
    )

    out = DIT.Scenario{:gradient, :out}[]
    skip_ref = Set(broken_scenario_names())

    function _push!(name, f, θ₀)
        # Globally-broken scenarios may break the reference backend
        # itself (e.g. #217). Construct them without res1 so the test
        # runner can still mark them broken without erroring here.
        res1 = (with_reference && !(name in skip_ref)) ?
               _reference(f, θ₀) : nothing
        # Prepare at the real parameter point. DIT's default
        # `prep_args` is `zero(x)`, which builds e.g. `Gamma(0, 0)`
        # and trips the distribution's `α > 0` domain assertion.
        prep_args = (; x = θ₀, contexts = ())
        push!(out,
            res1 === nothing ?
            DIT.Scenario{:gradient, :out}(
                f, θ₀; prep_args = prep_args, name = name) :
            DIT.Scenario{:gradient, :out}(
                f, θ₀; res1 = res1, prep_args = prep_args, name = name))
    end

    for spec in primary_specs, force_numeric in (false, true)

        path = force_numeric ? "numerical" : "analytical"
        f = let ctor = spec.ctor, force_numeric = force_numeric
            θ -> sum(
                x -> logpdf(
                    primary_censored(
                        ctor(θ[1], θ[2]), Uniform(0.0, 1.0);
                        force_numeric = force_numeric),
                    x),
                obs)
        end
        _push!("PrimaryCensored $(spec.name)+Uniform $path", f, spec.θ₀)
    end

    # ExponentiallyTilted primary event — no analytical
    # `primarycensored_cdf(::Delay, ::ExponentiallyTilted, ...)` exists,
    # so the scalar `r` parameter of the prior is included in θ and the
    # whole path runs through numeric integration. Exercises gradient
    # flow through both the delay distribution params and the primary
    # event's tilt parameter.
    for spec in primary_specs
        θ₀ = vcat(spec.θ₀, 0.5)
        f = let ctor = spec.ctor
            θ -> sum(
                x -> logpdf(
                    primary_censored(
                        ctor(θ[1], θ[2]),
                        ExponentiallyTilted(0.0, 1.0, θ[3])),
                    x),
                obs)
        end
        _push!("PrimaryCensored $(spec.name)+ExponentiallyTilted numerical",
            f, θ₀)
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

    return out
end

end # module
