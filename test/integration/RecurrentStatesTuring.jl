@testitem "recurrent_states_model recovers sojourn scales" tags=[:turing] begin
    using CensoredDistributions, Distributions, Random
    using Turing: NUTS, sample, @model, to_submodel
    using DynamicPPL: DynamicPPL
    using Statistics: mean

    # A reinfection cycle with an absorbing death state.
    template = recur(
        :well => (:ill => Gamma(2.0, 5.0)),
        :ill => (:well => Gamma(2.0, 3.0), :dead => Gamma(2.0, 10.0)))

    Random.seed!(11)
    paths = [rand(template; horizon = 200.0) for _ in 1:300]

    priors = (
        well = (ill = (shape = truncated(Normal(2, 0.5); lower = 0),
            scale = truncated(Normal(5, 1.5); lower = 0)),),
        ill = (
            well = (shape = truncated(Normal(2, 0.5); lower = 0),
                scale = truncated(Normal(3, 1); lower = 0)),
            dead = (shape = truncated(Normal(2, 0.5); lower = 0),
                scale = truncated(Normal(10, 3); lower = 0))))

    @model function fit_recurrent(t, p, ps)
        m ~ to_submodel(recurrent_states_model(t, p))
        for path in ps
            DynamicPPL.@addlogprob! logpdf(m, path)
        end
    end

    Random.seed!(1)
    chain = sample(
        fit_recurrent(template, priors, paths), NUTS(), 100; progress = false)

    # The posterior means recover the data-generating sojourn scales. The
    # parameters are addressable by their state.edge.param name, which confirms
    # the chain is namespaced by state and edge.
    @test mean(chain[Symbol("m.well.ill.scale")]) ≈ 5.0 atol = 1.0
    @test mean(chain[Symbol("m.ill.well.scale")]) ≈ 3.0 atol = 0.7
    @test mean(chain[Symbol("m.ill.dead.scale")]) ≈ 10.0 atol = 2.0
    @test isfinite(mean(chain[Symbol("m.well.ill.shape")]))
end

@testitem "recurrent_states_model: registry reconstruction is ForwardDiff-stable" tags=[
    :turing] begin
    using CensoredDistributions, Distributions, DynamicPPL, Turing, Random
    using ADTypes: AutoForwardDiff, AutoMooncake
    import Mooncake
    const LDP = DynamicPPL.LogDensityProblems

    # After the registry migration each state node reconstructs through
    # `update(node, unflatten(node, x))`. Guard that the full sample ->
    # reconstruct -> score-paths loop stays ForwardDiff-differentiable (the
    # backend the recurrent CTMC/path scoring supports).
    template = recur(
        :well => (:ill => Gamma(2.0, 5.0)),
        :ill => (:well => Gamma(2.0, 3.0), :dead => Gamma(2.0, 10.0)))
    priors = (
        well = (ill = (shape = truncated(Normal(2, 0.5); lower = 0),
            scale = truncated(Normal(5, 1.5); lower = 0)),),
        ill = (
            well = (shape = truncated(Normal(2, 0.5); lower = 0),
                scale = truncated(Normal(3, 1); lower = 0)),
            dead = (shape = truncated(Normal(2, 0.5); lower = 0),
                scale = truncated(Normal(10, 3); lower = 0))))

    Random.seed!(21)
    paths = [rand(template; horizon = 200.0) for _ in 1:20]

    @model function fit_recurrent(t, p, ps)
        m ~ to_submodel(recurrent_states_model(t, p))
        for path in ps
            DynamicPPL.@addlogprob! logpdf(m, path)
        end
    end

    mdl = fit_recurrent(template, priors, paths)
    vi = DynamicPPL.link(VarInfo(mdl), mdl)
    ldf = DynamicPPL.LogDensityFunction(
        mdl, DynamicPPL.getlogjoint_internal, vi; adtype = AutoForwardDiff())
    lp, g = LDP.logdensity_and_gradient(ldf, vi[:])
    @test isfinite(lp)
    @test length(g) == 6
    @test all(isfinite, g)
    @test any(!iszero, g)

    # Reverse-mode (Mooncake) over the recurrent CTMC/path `logpdf` hits a
    # `No rrule!! available for foreigncall` — a PRE-EXISTING limitation of the
    # recurrent scoring core (the `exp(Qt)` / jump-chain path), present
    # identically on the pre-#74 head/tail-cons reconstruction, so orthogonal to
    # this registry migration. Tracked in issue #834; marked broken so a future
    # core fix flips it to an unexpected pass.
    @test_broken begin
        ldf_mc = DynamicPPL.LogDensityFunction(
            mdl, DynamicPPL.getlogjoint_internal, vi;
            adtype = AutoMooncake(config = nothing))
        LDP.logdensity_and_gradient(ldf_mc, vi[:])
        true
    end
end

@testitem "recurrent_states_model rejects bad priors" tags=[:turing] begin
    using CensoredDistributions, Distributions
    using Turing: @model, to_submodel
    using DynamicPPL: DynamicPPL

    template = recur(
        :well => (:ill => Gamma(2.0, 5.0)),
        :ill => (:well => Gamma(2.0, 3.0),))
    # Priors naming a non-existent state.
    bad = (
        well = (ill = (shape = truncated(Normal(2, 0.5); lower = 0),
            scale = truncated(Normal(5, 1.5); lower = 0)),),
        nowhere = (x = Normal(),))

    @model function fit_bad(t, p)
        m ~ to_submodel(recurrent_states_model(t, p))
    end
    @test_throws Exception DynamicPPL.VarInfo(fit_bad(template, bad))
end
