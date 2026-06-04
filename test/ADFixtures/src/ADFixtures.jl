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
using Distributions: Distributions, Gamma, LogNormal, Weibull, Uniform,
                     Normal, logpdf, cdf
using ADTypes: ADTypes, AutoForwardDiff, AutoReverseDiff, AutoMooncake,
               AutoMooncakeForward, AutoEnzyme
using DifferentiationInterface: DifferentiationInterface, Constant
import ForwardDiff, ReverseDiff, Mooncake, Enzyme
import DifferentiationInterfaceTest as DIT

export scenarios, backends, working_backends, broken_backends,
       broken_scenario_names, backend_broken_scenarios

# `contexts` is a tuple of `Constant`-wrapped data (the observations),
# passed positionally to DI's `gradient` and to the differentiated
# function. See `scenarios` for why data travels as a context rather
# than a closure capture.
function _reference(f, θ, contexts)
    DifferentiationInterface.gradient(f, AutoForwardDiff(), θ, contexts...)
end

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
        (name = "Mooncake forward", backend = AutoMooncakeForward()),
        # `set_runtime_activity` is the only user-facing Enzyme setting
        # needed; no `function_annotation = Duplicated`, because the
        # differentiated functions capture no data (see `scenarios`).
        (name = "Enzyme reverse",
            backend = AutoEnzyme(
                mode = Enzyme.set_runtime_activity(Enzyme.Reverse))),
        (name = "Enzyme forward",
            backend = AutoEnzyme(
                mode = Enzyme.set_runtime_activity(Enzyme.Forward)))
    ]
end

"""
    broken_backends()

AD backends that fail on at least some scenarios. `check_broken` in
`test/ad/setup.jl` runs each through plain
`DifferentiationInterface.gradient` and marks the scenarios that do
work as passing, so a partially-working backend is not forced to be
all-or-nothing. Empty today: every backend in [`backends`](@ref) is full
on all scenarios. Enzyme forward relies on `scenarios` constructing each
distribution as a literal rather than capturing a `Type` (see the comment
there and #278).
"""
function broken_backends()
    return NamedTuple{(:name, :backend)}[]
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
        "Mooncake forward" => Set{String}(),
        "Enzyme reverse" => Set{String}(),
        "Enzyme forward" => Set{String}()
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

    out = DIT.Scenario{:gradient, :out}[]
    skip_ref = Set(broken_scenario_names())

    function _push!(name, f, θ₀, contexts)
        # Globally-broken scenarios may break the reference backend
        # itself (e.g. #217). Construct them without res1 so the test
        # runner can still mark them broken without erroring here.
        res1 = (with_reference && !(name in skip_ref)) ?
               _reference(f, θ₀, contexts) : nothing
        # Prepare at the real parameter point with the real data
        # contexts. DIT's defaults `zero(x)` and `zero_contexts` would
        # build e.g. `Gamma(0, 0)` and trip the `α > 0` domain assertion.
        prep_args = (; x = θ₀, contexts = contexts)
        push!(out,
            res1 === nothing ?
            DIT.Scenario{:gradient, :out}(
                f, θ₀, contexts...; prep_args = prep_args, name = name) :
            DIT.Scenario{:gradient, :out}(
                f, θ₀, contexts...;
                res1 = res1, prep_args = prep_args, name = name))
    end

    # Observation data is passed as a `Constant` DI context, not captured
    # in a closure, so the differentiated function holds no active fields.
    # Enzyme then needs no `function_annotation = Duplicated`, the call is
    # faster, and it is more portable across backends. Each function
    # references only its `θ` argument, the passed-in data, and
    # module-level constructors.
    #
    # Delay distributions are still written as literals rather than a
    # captured `ctor::Type`. Capturing a distribution `Type` in a function
    # that also makes a keyword call (`force_numeric = ...`) trips an
    # upstream Enzyme forward-mode "mixed activity for jl_new_struct"
    # limitation (#278): the keyword-call lowering builds a struct mixing
    # the active `Type` and `Vector` fields with the inactive
    # `force_numeric` flag. Literal constructors avoid the captured-`Type`
    # field, so Enzyme forward differentiates every scenario; the
    # analytical/numerical split and math are unchanged.
    _push!("PrimaryCensored Gamma+Uniform analytical",
        (θ,
            obs) -> sum(
            x -> logpdf(
                primary_censored(Gamma(θ[1], θ[2]), Uniform(0.0, 1.0)), x),
            obs),
        [2.0, 1.5], (Constant(obs),))
    _push!("PrimaryCensored Gamma+Uniform numerical",
        (θ,
            obs) -> sum(
            x -> logpdf(
                primary_censored(Gamma(θ[1], θ[2]), Uniform(0.0, 1.0);
                    force_numeric = true), x),
            obs),
        [2.0, 1.5], (Constant(obs),))
    _push!("PrimaryCensored LogNormal+Uniform analytical",
        (θ,
            obs) -> sum(
            x -> logpdf(
                primary_censored(LogNormal(θ[1], θ[2]), Uniform(0.0, 1.0)),
                x),
            obs),
        [1.0, 0.75], (Constant(obs),))
    _push!("PrimaryCensored LogNormal+Uniform numerical",
        (θ,
            obs) -> sum(
            x -> logpdf(
                primary_censored(LogNormal(θ[1], θ[2]), Uniform(0.0, 1.0);
                    force_numeric = true), x),
            obs),
        [1.0, 0.75], (Constant(obs),))
    _push!("PrimaryCensored Weibull+Uniform analytical",
        (θ,
            obs) -> sum(
            x -> logpdf(
                primary_censored(Weibull(θ[1], θ[2]), Uniform(0.0, 1.0)), x),
            obs),
        [2.0, 1.5], (Constant(obs),))
    _push!("PrimaryCensored Weibull+Uniform numerical",
        (θ,
            obs) -> sum(
            x -> logpdf(
                primary_censored(Weibull(θ[1], θ[2]), Uniform(0.0, 1.0);
                    force_numeric = true), x),
            obs),
        [2.0, 1.5], (Constant(obs),))

    # ExponentiallyTilted primary event — no analytical
    # `primarycensored_cdf(::Delay, ::ExponentiallyTilted, ...)` exists,
    # so the scalar `r` parameter of the prior is included in θ (as θ[3])
    # and the whole path runs through numeric integration. Exercises
    # gradient flow through both the delay distribution params and the
    # primary event's tilt parameter. Written as literal constructors
    # rather than a captured `ctor::Type` loop, for the #278 reason above.
    _push!("PrimaryCensored Gamma+ExponentiallyTilted numerical",
        (θ,
            obs) -> sum(
            x -> logpdf(
                primary_censored(Gamma(θ[1], θ[2]),
                    ExponentiallyTilted(0.0, 1.0, θ[3])), x),
            obs),
        [2.0, 1.5, 0.5], (Constant(obs),))
    _push!("PrimaryCensored LogNormal+ExponentiallyTilted numerical",
        (θ,
            obs) -> sum(
            x -> logpdf(
                primary_censored(LogNormal(θ[1], θ[2]),
                    ExponentiallyTilted(0.0, 1.0, θ[3])), x),
            obs),
        [1.0, 0.75, 0.5], (Constant(obs),))
    _push!("PrimaryCensored Weibull+ExponentiallyTilted numerical",
        (θ,
            obs) -> sum(
            x -> logpdf(
                primary_censored(Weibull(θ[1], θ[2]),
                    ExponentiallyTilted(0.0, 1.0, θ[3])), x),
            obs),
        [2.0, 1.5, 0.5], (Constant(obs),))

    _push!("IntervalCensored LogNormal regular",
        (θ, obs) -> sum(
            x -> logpdf(interval_censored(LogNormal(θ[1], θ[2]), 1.0), x),
            obs),
        [1.0, 0.75], (Constant(obs_int),))

    # Two data contexts: the observations and the interval boundaries.
    _push!("IntervalCensored Gamma arbitrary",
        (θ, obs,
            bnd) -> sum(
            x -> logpdf(interval_censored(Gamma(θ[1], θ[2]), bnd), x),
            obs),
        [2.0, 1.5], (Constant(obs_int_gamma), Constant(boundaries)))

    _push!("DoubleIntervalCensored LogNormal",
        (θ,
            obs) -> sum(
            x -> logpdf(
                double_interval_censored(LogNormal(θ[1], θ[2]);
                    primary_event = Uniform(0.0, 1.0),
                    upper = 10.0, interval = 1.0), x),
            obs),
        [1.0, 0.75], (Constant(obs_double),))

    # Convolved (sum of independent delays). The analytic Normal+Normal
    # pair differentiates through `Distributions.convolve`; the
    # Gamma+LogNormal pair has no analytic convolution and exercises the
    # AD-safe numeric quadrature path (the same fixed-domain Gauss-Legendre
    # construction as PrimaryCensored). Literal constructors keep Enzyme
    # forward working (#278).
    # Guarded on `generic_convolve` existing: AirspeedVelocity benchmarks
    # the PR against the `main` baseline, building the baseline package
    # while still loading this (PR-tree) fixtures module. Referencing
    # `generic_convolve` unconditionally would throw `UndefVarError` on the
    # baseline, where it does not yet exist. The guard lets the baseline
    # skip these scenarios and the PR include them.
    if isdefined(CensoredDistributions, :generic_convolve)
        _push!("Convolved Normal+Normal analytical",
            (θ,
                obs) -> sum(
                x -> logpdf(
                    CensoredDistributions.generic_convolve(
                        Normal(θ[1], θ[2]), Normal(0.0, 1.0)), x),
                obs),
            [1.0, 2.0], (Constant(obs),))
        _push!("Convolved Gamma+LogNormal numerical",
            (θ,
                obs) -> sum(
                x -> logpdf(
                    CensoredDistributions.generic_convolve(
                        Gamma(θ[1], θ[2]), LogNormal(0.5, 0.4)), x),
                obs),
            [2.0, 1.0], (Constant(obs),))
    end

    # High-dimensional scenarios. Each observation carries its own delay
    # parameter, so the gradient is taken with respect to many inputs.
    # These give the reverse-mode backends (ReverseDiff, Enzyme reverse,
    # Mooncake reverse) a regime where they win: reverse mode costs one
    # pass regardless of the parameter count, while the forward modes
    # (ForwardDiff, Enzyme forward, Mooncake forward) pay per parameter.
    # Both an analytical and a numerical (quadrature) variant are included:
    # the numerical path does far more work per call, which is where the
    # compiled forward backends (Enzyme, Mooncake) can close the gap on
    # ForwardDiff's Dual-number propagation. Literal constructors keep
    # Enzyme forward working (#278).
    n_hd = 32
    obs_hd = collect(range(0.5, 8.0; length = n_hd))
    _push!("PrimaryCensored LogNormal+Uniform analytical $(n_hd)d",
        (θ,
            obs_hd) -> sum(
            i -> logpdf(
                primary_censored(LogNormal(θ[i], 0.5), Uniform(0.0, 1.0)),
                obs_hd[i]),
            eachindex(obs_hd)),
        fill(1.0, n_hd), (Constant(obs_hd),))
    _push!("PrimaryCensored LogNormal+Uniform numerical $(n_hd)d",
        (θ,
            obs_hd) -> sum(
            i -> logpdf(
                primary_censored(LogNormal(θ[i], 0.5), Uniform(0.0, 1.0);
                    force_numeric = true),
                obs_hd[i]),
            eachindex(obs_hd)),
        fill(1.0, n_hd), (Constant(obs_hd),))
    _push!("PrimaryCensored Gamma+Uniform analytical $(n_hd)d",
        (θ,
            obs_hd) -> sum(
            i -> logpdf(
                primary_censored(Gamma(θ[i], 1.5), Uniform(0.0, 1.0)),
                obs_hd[i]),
            eachindex(obs_hd)),
        fill(2.0, n_hd), (Constant(obs_hd),))
    _push!("PrimaryCensored Gamma+Uniform numerical $(n_hd)d",
        (θ,
            obs_hd) -> sum(
            i -> logpdf(
                primary_censored(Gamma(θ[i], 1.5), Uniform(0.0, 1.0);
                    force_numeric = true),
                obs_hd[i]),
            eachindex(obs_hd)),
        fill(2.0, n_hd), (Constant(obs_hd),))

    return out
end

end # module
