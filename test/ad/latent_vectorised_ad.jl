# AD coverage for the VECTORISED latent path (`latent_observed_logpdf`). The
# latent table scores in a vectorised conditional given the sampled primaries:
#
#   @addlogprob! latent_observed_logpdf(d, rows, primaries)
#
# This must differentiate w.r.t. the delay parameters AND the sampled primaries
# under ForwardDiff and Mooncake reverse (the project's target latent AD). The
# latent-leaf / latent-Select-leaf path reads each row's single observed value
# positionally (`_row_event_vector(row)`), so it does NOT touch the
# string-based event-name derivation (`_split_edge_name`) that Mooncake reverse
# cannot trace; the gradient flows on both backends.

@testitem "vectorised latent gradient: ForwardDiff" tags=[
    :ad, :forwarddiff] begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: latent, latent_observed_logpdf
    using ADTypes: AutoForwardDiff
    using DifferentiationInterface: gradient
    using ForwardDiff: ForwardDiff

    rows = [(delay = 3.0,), (delay = 5.0,), (delay = 2.5,)]
    primaries = [0.3, 0.6, 0.2]

    # f(θ): build a latent leaf from the gamma (shape, scale), then score the
    # whole table's observed conditional given the fixed sampled primaries.
    function f(θ)
        d = latent(primary_censored(Gamma(θ[1], θ[2]), Uniform(0, 1)))
        return latent_observed_logpdf(d, rows, primaries)
    end

    θ = [4.0, 1.5]
    g = gradient(f, AutoForwardDiff(), θ)
    @test g isa AbstractVector && length(g) == 2 && all(isfinite, g)

    # The gradient w.r.t. the sampled primaries also flows (the latent variables
    # the sampler differentiates through).
    function fp(p)
        d = latent(primary_censored(Gamma(4.0, 1.5), Uniform(0, 1)))
        return latent_observed_logpdf(d, rows, p)
    end
    gp = gradient(fp, AutoForwardDiff(), primaries)
    @test gp isa AbstractVector && length(gp) == 3 && all(isfinite, gp)
end

@testitem "vectorised latent gradient: Mooncake reverse" tags=[
    :ad, :mooncake, :mooncake_reverse] begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: latent, latent_observed_logpdf
    using ADTypes: AutoMooncake, AutoForwardDiff
    using DifferentiationInterface: gradient
    using ForwardDiff: ForwardDiff
    using Mooncake: Mooncake

    rows = [(delay = 3.0,), (delay = 5.0,), (delay = 2.5,)]
    primaries = [0.3, 0.6, 0.2]

    function f(θ)
        d = latent(primary_censored(Gamma(θ[1], θ[2]), Uniform(0, 1)))
        return latent_observed_logpdf(d, rows, primaries)
    end

    θ = [4.0, 1.5]
    ref = gradient(f, AutoForwardDiff(), θ)
    g = gradient(f, AutoMooncake(; config = nothing), θ)
    @test isapprox(g, ref; rtol = 1e-6, atol = 1e-8)

    function fp(p)
        d = latent(primary_censored(Gamma(4.0, 1.5), Uniform(0, 1)))
        return latent_observed_logpdf(d, rows, p)
    end
    refp = gradient(fp, AutoForwardDiff(), primaries)
    gp = gradient(fp, AutoMooncake(; config = nothing), primaries)
    @test isapprox(gp, refp; rtol = 1e-6, atol = 1e-8)
end

@testitem "vectorised mixed Select latent gradient: ForwardDiff" tags=[
    :ad, :forwarddiff] begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: latent, latent_observed_logpdf
    using ADTypes: AutoForwardDiff
    using DifferentiationInterface: gradient
    using ForwardDiff: ForwardDiff

    # A mixed Select table: index rows (marginal) and sourced rows (latent). Two
    # sourced rows carry latent primaries.
    rows = [(kind = :index, delay = 3.0),
        (kind = :sourced, delay = 5.0),
        (kind = :index, delay = 2.0),
        (kind = :sourced, delay = 7.0)]
    primaries = [0.4, 0.7]

    # θ = [index shape, index scale, sourced shape, sourced scale].
    function f(θ)
        d = select_branch(
            :index => primary_censored(Gamma(θ[1], θ[2]), Uniform(0, 1)),
            :sourced => latent(
                primary_censored(Gamma(θ[3], θ[4]), Uniform(0, 1))))
        return latent_observed_logpdf(d, rows, primaries)
    end

    θ = [2.0, 1.0, 4.0, 1.5]
    g = gradient(f, AutoForwardDiff(), θ)
    # Every parameter contributes: the index params through the marginal rows,
    # the sourced params through the latent conditional.
    @test g isa AbstractVector && length(g) == 4 && all(isfinite, g)
    @test all(!=(0), g)
end
