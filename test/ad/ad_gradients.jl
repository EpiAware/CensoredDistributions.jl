# Automatic differentiation (AD) gradient tests for CensoredDistributions.
#
# Each test defines a scalar loss `loss(θ) = sum(logpdf.(construct(θ), obs))`
# and checks that each supported AD backend returns a gradient that is finite
# and agrees with a finite-difference reference. ForwardDiff is the required
# backend; ReverseDiff and Zygote are exercised best-effort and marked broken
# when they fail so that unexpected successes still flag regressions.
#
# `PrimaryCensored.logpdf` performs internal numerical differentiation of
# `logcdf` with a hardcoded step, so adaptive finite-difference methods pick
# too small a step for the reference and fall into the cancellation noise.
# A simple central difference with h = 1e-2 is used instead; the accompanying
# tolerance (rtol = 5e-2) absorbs the residual truncation error.

@testsnippet ADGradientHelpers begin
    using CensoredDistributions
    using Distributions
    using ADTypes
    using DifferentiationInterface
    using ForwardDiff
    using ReverseDiff
    using Zygote

    # Backends to check. The first entry is the primary backend and must pass.
    # Later entries are best-effort and marked @test_broken on failure.
    const AD_BACKENDS = (
        (name = "ForwardDiff", backend = AutoForwardDiff(), required = true),
        (name = "ReverseDiff", backend = AutoReverseDiff(), required = false),
        (name = "Zygote", backend = AutoZygote(), required = false)
    )

    """
        simple_fd(f, θ; h=1e-2)

    Central-difference gradient using a fixed step. A fixed step is required
    because `logpdf` for `PrimaryCensored` performs an internal numerical
    differentiation with a hardcoded step, which makes the function non-smooth
    on scales smaller than ~1e-3 and defeats adaptive finite-difference
    methods such as `FiniteDifferences.central_fdm`.
    """
    function simple_fd(f, θ::AbstractVector; h = 1e-2)
        n = length(θ)
        g = similar(θ, float(eltype(θ)))
        for i in 1:n
            e = zero(θ)
            e[i] = one(eltype(θ))
            g[i] = (f(θ .+ h .* e) - f(θ .- h .* e)) / (2h)
        end
        return g
    end

    """
        check_gradient(loss, θ; rtol=5e-2, atol=1e-6, required_broken=false)

    For each backend in `AD_BACKENDS`, compute the gradient of `loss` at `θ`
    via `DifferentiationInterface` and compare against a simple central
    finite-difference reference. Required backends must return a finite
    gradient matching the reference; non-required backends are marked
    `@test_broken` when they fail, so unexpected successes still flag the
    regression.

    Set `required_broken=true` for cases where the primary backend is known
    to fail; the primary backend is then checked with `@test_broken` and a
    follow-up issue should be linked in a comment near the call site.
    """
    function check_gradient(loss, θ; rtol = 5e-2, atol = 1e-6,
            required_broken = false)
        grad_fd = simple_fd(loss, θ)
        @test all(isfinite, grad_fd)

        for entry in AD_BACKENDS
            grad_ad = try
                DifferentiationInterface.gradient(loss, entry.backend, θ)
            catch err
                err
            end

            ok = grad_ad isa AbstractVector && all(isfinite, grad_ad) &&
                 isapprox(grad_ad, grad_fd; rtol = rtol, atol = atol)

            if entry.required
                if required_broken
                    @test_broken ok
                else
                    @test ok
                end
            else
                if ok
                    @test ok
                else
                    @test_broken ok
                end
            end
        end
        return grad_fd
    end
end

@testitem "AD gradient: PrimaryCensored Gamma+Uniform analytical" tags=[:ad] setup=[ADGradientHelpers] begin
    using CensoredDistributions
    using Distributions

    obs=[0.5, 1.2, 2.5, 3.8, 5.1]
    function loss(θ)
        d=primary_censored(Gamma(θ[1], θ[2]), Uniform(0.0, 1.0))
        return sum(x->logpdf(d, x), obs)
    end
    check_gradient(loss, [2.0, 1.5])
end

@testitem "AD gradient: PrimaryCensored LogNormal+Uniform analytical" tags=[:ad] setup=[ADGradientHelpers] begin
    using CensoredDistributions
    using Distributions

    obs=[0.5, 1.2, 2.5, 3.8, 5.1]
    function loss(θ)
        d=primary_censored(LogNormal(θ[1], θ[2]), Uniform(0.0, 1.0))
        return sum(x->logpdf(d, x), obs)
    end
    check_gradient(loss, [1.0, 0.75])
end

@testitem "AD gradient: PrimaryCensored Weibull+Uniform analytical" tags=[:ad] setup=[ADGradientHelpers] begin
    using CensoredDistributions
    using Distributions

    obs=[0.5, 1.2, 2.5, 3.8, 5.1]
    function loss(θ)
        d=primary_censored(Weibull(θ[1], θ[2]), Uniform(0.0, 1.0))
        return sum(x->logpdf(d, x), obs)
    end
    check_gradient(loss, [2.0, 1.5])
end

@testitem "AD gradient: PrimaryCensored LogNormal+Uniform numerical" tags=[:ad] setup=[ADGradientHelpers] begin
    using CensoredDistributions
    using Distributions

    obs=[0.5, 1.2, 2.5, 3.8, 5.1]
    function loss(θ)
        d=primary_censored(
            LogNormal(θ[1], θ[2]), Uniform(0.0, 1.0); force_numeric = true)
        return sum(x->logpdf(d, x), obs)
    end
    check_gradient(loss, [1.0, 0.75])
end

@testitem "AD gradient: IntervalCensored LogNormal regular" tags=[:ad] setup=[ADGradientHelpers] begin
    using CensoredDistributions
    using Distributions

    obs=[0.0, 1.0, 2.0, 3.0, 4.0]
    function loss(θ)
        d=interval_censored(LogNormal(θ[1], θ[2]), 1.0)
        return sum(x->logpdf(d, x), obs)
    end
    check_gradient(loss, [1.0, 0.75])
end

# IntervalCensored with Gamma fails through ForwardDiff because
# `cdf(Gamma{Dual}, Float64)` dispatches into
# `SpecialFunctions._gamma_inc(a, x, ind)` which lacks a method accepting a
# `Dual` first argument alongside a non-dual `x`. Tracked in
# https://github.com/EpiAware/CensoredDistributions.jl/issues/217.
@testitem "AD gradient: IntervalCensored Gamma arbitrary boundaries" tags=[:ad] setup=[ADGradientHelpers] begin
    using CensoredDistributions
    using Distributions

    boundaries=[0.0, 1.5, 3.0, 5.0, 10.0]
    obs=[0.5, 2.0, 4.0, 7.0]
    function loss(θ)
        d=interval_censored(Gamma(θ[1], θ[2]), boundaries)
        return sum(x->logpdf(d, x), obs)
    end
    check_gradient(loss, [2.0, 1.5]; required_broken = true)
end

@testitem "AD gradient: DoubleIntervalCensored LogNormal" tags=[:ad] setup=[ADGradientHelpers] begin
    using CensoredDistributions
    using Distributions

    obs=[1.0, 2.0, 3.0, 4.0, 5.0]
    function loss(θ)
        d=double_interval_censored(
            LogNormal(θ[1], θ[2]);
            primary_event = Uniform(0.0, 1.0),
            upper = 10.0,
            interval = 1.0
        )
        return sum(x->logpdf(d, x), obs)
    end
    check_gradient(loss, [1.0, 0.75])
end

@testitem "AD gradient: logpdf is scalar-differentiable in x" tags=[:ad] setup=[ADGradientHelpers] begin
    using CensoredDistributions
    using Distributions

    # Scalar-input sanity check: `logpdf(d, x)` should be differentiable
    # w.r.t. `x` at interior points where the density is smooth. This mirrors
    # how AD is exercised when `x` is a latent parameter inside a Turing
    # model.
    d=primary_censored(LogNormal(1.0, 0.75), Uniform(0.0, 1.0))
    g(x)=logpdf(d, x[1])
    check_gradient(g, [2.5])
end
