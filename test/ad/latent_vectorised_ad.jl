# AD coverage for the VECTORISED latent path (`latent_observed_logpdf`). The
# latent table scores in a vectorised conditional given the sampled primaries:
#
#   @addlogprob! latent_observed_logpdf(d, rows, primaries)
#
# This must differentiate w.r.t. the delay parameters AND the sampled primaries
# under ForwardDiff and Mooncake reverse (the project's target latent AD). The
# latent-leaf / latent-Choose-leaf path reads each row's single observed value
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

@testitem "vectorised mixed Choose latent gradient: ForwardDiff" tags=[
    :ad, :forwarddiff] begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: latent, latent_observed_logpdf
    using ADTypes: AutoForwardDiff
    using DifferentiationInterface: gradient
    using ForwardDiff: ForwardDiff

    # A mixed Choose table: index rows (marginal) and sourced rows (latent). Two
    # sourced rows carry latent primaries.
    rows = [(kind = :index, delay = 3.0),
        (kind = :sourced, delay = 5.0),
        (kind = :index, delay = 2.0),
        (kind = :sourced, delay = 7.0)]
    primaries = [0.4, 0.7]

    # θ = [index shape, index scale, sourced shape, sourced scale].
    function f(θ)
        d = choose(
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

# The latent CHAIN scores its terminal conditional through the by-name event
# vector (`_row_event_vector(chain, row)`), which DERIVES the chain's event names
# from the edge names with string ops (`_split_edge_name` in `tree_events.jl`).
# Mooncake reverse cannot trace that string-based name derivation over a row
# vector carrying `Missing` (it errors in const verification:
# "non-boolean (Missing) used in boolean context"), so the latent chain is kept
# on ForwardDiff here. A separate package-level Mooncake fix for the event-name
# derivation is in progress on another branch; once it lands the chain can move
# to the Mooncake-reverse coverage the latent LEAF already has (the leaf reads
# its observed value positionally, so it never hits the name derivation).
@testitem "vectorised latent chain gradient: ForwardDiff" tags=[
    :ad, :forwarddiff] begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: latent, latent_observed_logpdf
    using ADTypes: AutoForwardDiff
    using DifferentiationInterface: gradient
    using ForwardDiff: ForwardDiff

    # A two-edge latent chain: each row stacks the origin draw and the first gap;
    # the terminal conditions. The flat `primaries` carries a two-slot block per
    # row.
    rows = [(onset = missing, admit = missing, death = 6.0),
        (onset = missing, admit = missing, death = 9.0)]
    primaries = [0.4, 1.9, 0.6, 2.5]

    # θ = [edge1 shape, edge1 scale, edge2 shape, edge2 scale].
    function f(θ)
        e1 = primary_censored(Gamma(θ[1], θ[2]), Uniform(0, 1))
        e2 = primary_censored(Gamma(θ[3], θ[4]), Uniform(0, 1))
        d = latent(Sequential((e1, e2), (:onset_admit, :admit_death)))
        return latent_observed_logpdf(d, rows, primaries)
    end

    θ = [2.0, 1.0, 3.0, 1.5]
    g = gradient(f, AutoForwardDiff(), θ)
    @test g isa AbstractVector && length(g) == 4 && all(isfinite, g)
    # `latent_observed_logpdf` scores ONLY the terminal conditional (the chain's
    # PRIOR factors, including the intermediate-gap core, live in the separate
    # `product_distribution` statement). So only the terminal edge's params
    # (θ[3], θ[4]) appear here; the first edge's params (θ[1], θ[2]) enter via the
    # priors and so are zero in this conditional.
    @test g[3] != 0 && g[4] != 0

    # The gradient w.r.t. the stacked latents (origin draws + gaps) also flows;
    # every latent slot contributes through the reconstructed terminal gap.
    function fp(p)
        e1 = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))
        e2 = primary_censored(Gamma(3.0, 1.5), Uniform(0, 1))
        d = latent(Sequential((e1, e2), (:onset_admit, :admit_death)))
        return latent_observed_logpdf(d, rows, p)
    end
    gp = gradient(fp, AutoForwardDiff(), primaries)
    @test gp isa AbstractVector && length(gp) == 4 && all(isfinite, gp)
    @test all(!=(0), gp)
end
