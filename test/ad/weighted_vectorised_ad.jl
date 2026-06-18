# AD coverage for the vectorised weighted-observation logpdf.
#
# `weight(dist, weights)` scores a Product{Weighted} of one shared distribution
# in a SINGLE vectorised `logpdf(dist, x)` call (the cached-CDF batched PDF),
# rather than a per-observation loop. The gradient of the weighted log-density
# w.r.t. the distribution parameters must flow under every AD backend and match
# the per-observation weighted-sum loop EXACTLY.
#
# The shared `IntervalCensored(LogNormal, ...)` case is the regression guard: its
# support starts at 0, so a batched path that evaluated `cdf` AT the support edge
# (e.g. when probing the cached-CDF value type) produced NaN gradients under
# ReverseDiff. The value type is now read from `partype` without evaluating the
# CDF, keeping the batched path AD-safe.

@testitem "vectorised weighted logpdf gradient: ForwardDiff" tags=[
    :ad, :forwarddiff] begin
    using CensoredDistributions, Distributions
    using ADTypes: AutoForwardDiff
    using DifferentiationInterface: gradient
    using ForwardDiff: ForwardDiff

    x = repeat([2.0, 3.0, 4.0, 5.0], 25)        # 100 obs, 4 unique boundaries
    weights = Float64.(repeat([3.0, 2.0, 5.0, 1.0], 25))

    function f_vec(θ)
        d = interval_censored(LogNormal(θ[1], θ[2]), 1.0)
        return logpdf(weight(d, weights), x)
    end
    function f_loop(θ)
        d = interval_censored(LogNormal(θ[1], θ[2]), 1.0)
        s = zero(eltype(θ))
        for i in eachindex(x)
            s += weights[i] * logpdf(d, x[i])
        end
        return s
    end

    θ = [1.5, 0.5]
    g_vec = gradient(f_vec, AutoForwardDiff(), θ)
    g_loop = gradient(f_loop, AutoForwardDiff(), θ)

    @test all(isfinite, g_vec)
    @test any(!iszero, g_vec)
    @test isapprox(g_vec, g_loop; rtol = 1e-8, atol = 1e-10)
end

@testitem "vectorised weighted logpdf gradient: ReverseDiff" tags=[
    :ad, :reversediff] begin
    using CensoredDistributions, Distributions
    using ADTypes: AutoReverseDiff, AutoForwardDiff
    using DifferentiationInterface: gradient
    using ForwardDiff: ForwardDiff
    using ReverseDiff: ReverseDiff

    x = repeat([2.0, 3.0, 4.0, 5.0], 25)
    weights = Float64.(repeat([3.0, 2.0, 5.0, 1.0], 25))

    function f_vec(θ)
        d = interval_censored(LogNormal(θ[1], θ[2]), 1.0)
        return logpdf(weight(d, weights), x)
    end
    function f_loop(θ)
        d = interval_censored(LogNormal(θ[1], θ[2]), 1.0)
        s = zero(eltype(θ))
        for i in eachindex(x)
            s += weights[i] * logpdf(d, x[i])
        end
        return s
    end

    θ = [1.5, 0.5]
    g_vec = gradient(f_vec, AutoReverseDiff(), θ)
    g_loop = gradient(f_loop, AutoForwardDiff(), θ)

    # The LogNormal batched path must NOT regress to NaN gradients.
    @test all(isfinite, g_vec)
    @test any(!iszero, g_vec)
    @test isapprox(g_vec, g_loop; rtol = 1e-6, atol = 1e-8)
end

@testitem "vectorised weighted logpdf gradient: Mooncake reverse" tags=[
    :ad, :mooncake, :mooncake_reverse] begin
    using CensoredDistributions, Distributions
    using ADTypes: AutoMooncake, AutoForwardDiff
    using DifferentiationInterface: gradient
    using ForwardDiff: ForwardDiff
    using Mooncake: Mooncake

    x = repeat([2.0, 3.0, 4.0, 5.0], 25)
    weights = Float64.(repeat([3.0, 2.0, 5.0, 1.0], 25))

    function f_vec(θ)
        d = interval_censored(LogNormal(θ[1], θ[2]), 1.0)
        return logpdf(weight(d, weights), x)
    end
    function f_loop(θ)
        d = interval_censored(LogNormal(θ[1], θ[2]), 1.0)
        s = zero(eltype(θ))
        for i in eachindex(x)
            s += weights[i] * logpdf(d, x[i])
        end
        return s
    end

    θ = [1.5, 0.5]
    g_vec = gradient(f_vec, AutoMooncake(; config = nothing), θ)
    g_loop = gradient(f_loop, AutoForwardDiff(), θ)

    @test all(isfinite, g_vec)
    @test any(!iszero, g_vec)
    @test isapprox(g_vec, g_loop; rtol = 1e-6, atol = 1e-8)
end
