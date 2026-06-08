# Tests for `composed_parameters_model` (#353): the priors -> Turing-submodel
# helper that samples a composed distribution's parameters from user priors and
# returns the reconstructed distribution, ready to score.

@testitem "composed_parameters_model: round-trip structure + names" tags=[:turing] begin
    using CensoredDistributions, Distributions, DynamicPPL, Random

    template = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    priors = (
        onset_admit = (shape = truncated(Normal(2, 0.5); lower = 0),
            scale = truncated(Normal(1, 0.3); lower = 0)),
        admit_death = (mu = Normal(0.5, 0.2),
            sigma = truncated(Normal(0.4, 0.1); lower = 0)))

    @model function pm(t, p)
        d ~ to_submodel(composed_parameters_model(t, p))
        return d
    end

    Random.seed!(101)
    m = pm(template, priors)
    d = m()

    # The reconstructed distribution has the template's structure and names.
    @test d isa CensoredDistributions.Parallel
    @test event_names(d) == event_names(template)
    @test keys(params(d)) == keys(params(template))
    @test get_event(d, :onset_admit) isa Gamma
    @test get_event(d, :admit_death) isa LogNormal

    # Sampled parameter varnames carry the edge path (Option A prefixing).
    vns = Set(string.(collect(keys(VarInfo(m)))))
    @test "d.onset_admit.shape" in vns
    @test "d.onset_admit.scale" in vns
    @test "d.admit_death.mu" in vns
    @test "d.admit_death.sigma" in vns
end

@testitem "composed_parameters_model: bare leaf template" tags=[:turing] begin
    using CensoredDistributions, Distributions, DynamicPPL, Random

    @model function pm(t, p)
        d ~ to_submodel(composed_parameters_model(t, p))
        return d
    end

    Random.seed!(102)
    m = pm(Gamma(2.0, 1.0),
        (shape = truncated(Normal(2, 0.5); lower = 0),
            scale = truncated(Normal(1, 0.3); lower = 0)))
    d = m()
    @test d isa Gamma
    vns = Set(string.(collect(keys(VarInfo(m)))))
    @test "d.shape" in vns
    @test "d.scale" in vns
end

@testitem "composed_parameters_model: nested Sequential + Competing" tags=[:turing] begin
    using CensoredDistributions, Distributions, DynamicPPL, Random

    template = compose((
        chain = [Gamma(2.0, 1.0), LogNormal(0.5, 0.4)],
        resolution = Competing(:death => (Gamma(1.5, 1.0), 0.3),
            :disch => (Gamma(2.0, 1.5), 0.7))))
    priors = (
        chain = (
            step_1 = (shape = truncated(Normal(2, 0.5); lower = 0),
                scale = truncated(Normal(1, 0.3); lower = 0)),
            step_2 = (mu = Normal(0.5, 0.2),
                sigma = truncated(Normal(0.4, 0.1); lower = 0))),
        resolution = (
            death = (shape = truncated(Normal(1.5, 0.3); lower = 0),
                scale = truncated(Normal(1, 0.2); lower = 0)),
            disch = (shape = truncated(Normal(2, 0.3); lower = 0),
                scale = truncated(Normal(1.5, 0.2); lower = 0))))
    )

    @model function pm(t, p)
        d ~ to_submodel(composed_parameters_model(t, p))
        return d
    end

    Random.seed!(103)
    m = pm(template, priors)
    d = m()

    @test event_names(d) == event_names(template)
    @test keys(params(d)) == keys(params(template))
    # Competing branch probabilities kept fixed from the template by default.
    @test get_event(d, :resolution).branch_probs == (0.3, 0.7)

    vns = Set(string.(collect(keys(VarInfo(m)))))
    @test "d.chain.step_1.shape" in vns
    @test "d.chain.step_2.sigma" in vns
    @test "d.resolution.death.scale" in vns
    @test "d.resolution.disch.shape" in vns
end

@testitem "composed_parameters_model: Competing branch_probs prior" tags=[:turing] begin
    using CensoredDistributions, Distributions, DynamicPPL, Random

    template = compose((resolution = Competing(
        :death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.7)),))
    priors = (resolution = (
        death = (shape = truncated(Normal(1.5, 0.3); lower = 0),
            scale = truncated(Normal(1, 0.2); lower = 0)),
        disch = (shape = truncated(Normal(2, 0.3); lower = 0),
            scale = truncated(Normal(1.5, 0.2); lower = 0)),
        branch_probs = (death = Uniform(0, 1), disch = Uniform(0, 1))),)

    @model function pm(t, p)
        d ~ to_submodel(composed_parameters_model(t, p))
        return d
    end

    Random.seed!(104)
    m = pm(template, priors)
    d = m()
    bp = get_event(d, :resolution).branch_probs
    @test length(bp) == 2
    @test all(0 .<= bp .<= 1)

    vns = Set(string.(collect(keys(VarInfo(m)))))
    @test "d.resolution.branch_probs.death" in vns
    @test "d.resolution.branch_probs.disch" in vns
end

@testitem "composed_parameters_model: missing/extra prior errors" tags=[:turing] begin
    using CensoredDistributions, Distributions, DynamicPPL

    template = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))

    @model function pm(t, p)
        d ~ to_submodel(composed_parameters_model(t, p))
        return d
    end

    full = (
        onset_admit = (shape = Normal(2, 0.5), scale = Normal(1, 0.3)),
        admit_death = (mu = Normal(0.5, 0.2),
            sigma = truncated(Normal(0.4, 0.1); lower = 0)))

    # Missing a leaf parameter.
    miss_param = merge(full, (; onset_admit = (shape = Normal(2, 0.5),)))
    @test_throws ArgumentError pm(template, miss_param)()

    # Extra leaf parameter.
    extra_param = merge(full,
        (; onset_admit = (shape = Normal(2, 0.5),
            scale = Normal(1, 0.3), bogus = Normal(0, 1))))
    @test_throws ArgumentError pm(template, extra_param)()

    # Missing an edge.
    miss_edge = (; onset_admit = full.onset_admit)
    @test_throws ArgumentError pm(template, miss_edge)()

    # Extra edge.
    extra_edge = merge(full, (; ghost = (mu = Normal(0, 1), sigma = Normal(1, 1))))
    @test_throws ArgumentError pm(template, extra_edge)()
end

@testitem "composed_parameters_model: AD gradient through full loop" tags=[:turing] begin
    using CensoredDistributions, Distributions, DynamicPPL, Turing, Random
    const LDP = DynamicPPL.LogDensityProblems

    # AD must flow through prior-sample -> reconstruct -> logpdf. Build the
    # log-density function with ForwardDiff and check a finite, non-zero gradient
    # over the whole sampled-and-reconstructed loop.
    template = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    priors = (
        onset_admit = (shape = truncated(Normal(2, 0.5); lower = 0),
            scale = truncated(Normal(1, 0.3); lower = 0)),
        admit_death = (mu = Normal(0.5, 0.2),
            sigma = truncated(Normal(0.4, 0.1); lower = 0)))

    @model function fit(t, p, ys)
        d ~ to_submodel(composed_parameters_model(t, p))
        for y in ys
            DynamicPPL.@addlogprob! logpdf(d, y)
        end
    end

    Random.seed!(106)
    ys = [[0.5, 2.0], [1.0, 3.0]]
    m = fit(template, priors, ys)
    # Link to the unconstrained space and start from a valid prior draw so the
    # gradient is taken at an in-support point of the full sampled-reconstructed
    # loop.
    vi = DynamicPPL.link(VarInfo(m), m)
    ldf = DynamicPPL.LogDensityFunction(
        m, DynamicPPL.getlogjoint_internal, vi; adtype = AutoForwardDiff())
    x0 = vi[:]
    lp, grad = LDP.logdensity_and_gradient(ldf, x0)

    @test isfinite(lp)
    @test length(grad) == 4
    @test all(isfinite, grad)
    @test any(!iszero, grad)
end

@testitem "composed_parameters_model: NUTS samples the full loop" tags=[:turing] begin
    using CensoredDistributions, Distributions, DynamicPPL, Turing, Random
    using FlexiChains: Prefixed, niters

    template = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    priors = (
        onset_admit = (shape = truncated(Normal(2, 0.5); lower = 0),
            scale = truncated(Normal(1, 0.3); lower = 0)),
        admit_death = (mu = Normal(0.5, 0.2),
            sigma = truncated(Normal(0.4, 0.1); lower = 0)))

    # Sample the priors, reconstruct, and score step-value vectors directly
    # through the composer logpdf (a self-contained likelihood).
    @model function fit(t, p, ys)
        d ~ to_submodel(composed_parameters_model(t, p))
        for y in ys
            DynamicPPL.@addlogprob! logpdf(d, y)
        end
        return d
    end

    Random.seed!(105)
    ys = [[0.5, 2.0], [1.0, 3.0], [0.8, 2.5]]
    chain = sample(
        fit(template, priors, ys), NUTS(), 50;
        chain_type = VNChain, progress = false)

    @test niters(chain) == 50
    # The chain carries the edge-path-prefixed parameter names.
    shape = chain[Prefixed(@varname(onset_admit.shape))]
    sigma = chain[Prefixed(@varname(admit_death.sigma))]
    @test all(isfinite.(shape))
    @test all(sigma .> 0)
end
