# Focused AD coverage for `double_interval_censored` over a freshly-built
# `Convolved` observed total of a `Sequential` chain. The marginal scenario
# suite runs this path for every backend; these items pin the Enzyme-reverse
# and Mooncake-reverse gradients directly so the regression cannot drift back
# in unnoticed.
#
# The earlier reverse failure came from `observed_distribution(::Sequential)`
# collapsing the chain through an abstract-eltype vector: the resulting
# `Convolved` carried an abstractly-typed component tuple, and the reverse
# pass could not build a shadow for it. Flattening the chain to a concrete
# component tuple in `wrap.jl` keeps the `Convolved` component types concrete,
# so the reverse pass differentiates the leaf params.

@testitem "double_interval_censored(Sequential) total: Enzyme reverse" tags=[
    :ad, :enzyme, :enzyme_reverse] begin
    using CensoredDistributions, Distributions
    using ADTypes: AutoEnzyme, AutoForwardDiff
    using DifferentiationInterface: gradient, Constant
    using Enzyme: Enzyme
    using ForwardDiff: ForwardDiff

    obs_int = [0.0, 1.0, 2.0, 3.0, 4.0]

    # θ = [Gamma α, Gamma θ, LogNormal μ, LogNormal σ]. The chain collapses
    # to a Convolved{Gamma, LogNormal} observed total before interval
    # censoring, so every parameter feeds the differentiated reverse pass.
    f = (θ,
        obs) -> sum(
        x -> logpdf(
            double_interval_censored(
                observed_distribution(
                    Sequential(Gamma(θ[1], θ[2]),
                    LogNormal(θ[3], θ[4])));
                primary_event = Uniform(0.0, 1.0), interval = 1.0),
            x),
        obs)

    θ = [2.0, 1.0, 0.5, 0.4]
    ctx = Constant(obs_int)
    rev = AutoEnzyme(mode = Enzyme.set_runtime_activity(Enzyme.Reverse))

    ref = gradient(f, AutoForwardDiff(), θ, ctx)
    @test ref isa AbstractVector && length(ref) == 4 && all(isfinite, ref)
    @test all(!=(0), ref)

    g = gradient(f, rev, θ, ctx)
    @test g isa AbstractVector && length(g) == 4 && all(isfinite, g)
    @test isapprox(g, ref; rtol = 5e-2, atol = 1e-6)
end

@testitem "double_interval_censored(Sequential) total: Mooncake reverse" tags=[
    :ad, :mooncake, :mooncake_reverse] begin
    using CensoredDistributions, Distributions
    using ADTypes: AutoMooncake, AutoForwardDiff
    using DifferentiationInterface: gradient, Constant
    using ForwardDiff: ForwardDiff
    using Mooncake: Mooncake

    obs_int = [0.0, 1.0, 2.0, 3.0, 4.0]

    f = (θ,
        obs) -> sum(
        x -> logpdf(
            double_interval_censored(
                observed_distribution(
                    Sequential(Gamma(θ[1], θ[2]),
                    LogNormal(θ[3], θ[4])));
                primary_event = Uniform(0.0, 1.0), interval = 1.0),
            x),
        obs)

    θ = [2.0, 1.0, 0.5, 0.4]
    ctx = Constant(obs_int)

    ref = gradient(f, AutoForwardDiff(), θ, ctx)
    @test all(!=(0), ref)

    g = gradient(f, AutoMooncake(; config = nothing), θ, ctx)
    @test g isa AbstractVector && length(g) == 4 && all(isfinite, g)
    @test isapprox(g, ref; rtol = 5e-2, atol = 1e-6)
end
