# `as_turing`: the Turing adaptor over an upstream `ComposedLogDensity` spec.
# The load-bearing checks are that the adapted model's log-joint EQUALS the
# LogDensityProblems `logdensity` on the same parameters (both routes score
# one target), that every estimated parameter is a named site in the chain
# (what a raw LogDensityProblem handed to Turing would lose), and that a
# fitted chain reads straight back onto the template via
# `ComposedDistributions.chain_to_params` / `param_draws`.

@testitem "as_turing: model log-joint == ComposedDistributions logdensity" tags=[:turing] begin
    using CensoredDistributions: as_turing
    using ComposedDistributions
    using ComposedDistributions: as_logdensity, flatten
    using Distributions, Random
    using DynamicPPL, Turing
    using FlexiChains: FlexiChains, VNChain, Extra

    tree = compose((
        onset_admit = uncertain(Gamma(2.0, 1.0);
            shape = LogNormal(log(2.0), 0.2)),
        admit_death = uncertain(LogNormal(0.5, 0.4);
            mu = Normal(0.5, 0.5))))
    data = [[0.5, 2.0], [1.0, 3.0], [0.8, 2.5]]
    prob = as_logdensity(tree, data)

    m = as_turing(prob)

    # `Prior()` is enough to exercise the wiring (no HMC tuning needed); the
    # differentiated path itself is checked separately (test/ad).
    Random.seed!(11)
    chain = sample(m, Prior(), 25; progress = false, chain_type = VNChain)
    draws = param_draws(tree, chain)
    ljs = vec(chain[Extra(:logjoint)])
    diffs = [abs(ComposedDistributions.logdensity(prob, flatten(tree, draws[i]))
                 - ljs[i]) for i in eachindex(draws)]
    @test maximum(diffs) < 1e-8
end

@testitem "as_turing: every estimated parameter is a named site" tags=[:turing] begin
    using CensoredDistributions: as_turing
    using ComposedDistributions
    using ComposedDistributions: as_logdensity
    using Distributions
    using DynamicPPL, Turing

    tree = compose((
        onset_admit = uncertain(Gamma(2.0, 1.0);
            shape = LogNormal(log(2.0), 0.2), scale = LogNormal(0.0, 0.2)),
        admit_death = uncertain(LogNormal(0.5, 0.4);
            mu = Normal(0.5, 0.5), sigma = LogNormal(log(0.4), 0.2))))
    data = [[0.5, 2.0], [1.0, 3.0]]

    m = as_turing(as_logdensity(tree, data))
    vns = Set(string.(collect(keys(VarInfo(m)))))
    @test "d.onset_admit.shape" in vns
    @test "d.onset_admit.scale" in vns
    @test "d.admit_death.mu" in vns
    @test "d.admit_death.sigma" in vns

    # The convenience forms mirror `as_logdensity`'s signatures and reach the
    # same sites.
    m2 = as_turing(tree, data)
    @test Set(string.(collect(keys(VarInfo(m2))))) == vns
    priors = param_priors(tree)
    m3 = as_turing(tree, priors, data)
    @test Set(string.(collect(keys(VarInfo(m3))))) == vns
    @test m3() isa ComposedDistributions.Parallel
end

@testitem "as_turing: a fixed parameter is not a sampled site" tags=[:turing] begin
    using CensoredDistributions: as_turing
    using ComposedDistributions
    using ComposedDistributions: as_logdensity
    using Distributions
    using DynamicPPL, Turing

    tree = compose((
        onset_admit = uncertain(Gamma(2.0, 1.0);
            shape = LogNormal(log(2.0), 0.2)),
        admit_death = LogNormal(0.5, 0.4)))
    data = [[0.5, 2.0], [1.0, 3.0]]

    # `admit_death` carries no `uncertain` spec, so its params never enter the
    # chain, matching the LogDensityProblems route (which excludes them from
    # the flat vector).
    m = as_turing(as_logdensity(tree, data))
    vns = Set(string.(collect(keys(VarInfo(m)))))
    @test !("d.admit_death.mu" in vns)
    @test !("d.admit_death.sigma" in vns)
    @test "d.onset_admit.shape" in vns

    # The reconstructed distribution holds the template's fixed values.
    @test params(event(m(), :admit_death)) == params(LogNormal(0.5, 0.4))
end

@testitem "as_turing: a centred-pooled tree is rejected, not silently mis-scored" tags=[:turing] begin
    using CensoredDistributions: as_turing
    using ComposedDistributions
    using ComposedDistributions: as_logdensity
    using Distributions
    using DynamicPPL, Turing

    # A GENERAL (non-location-scale) population takes the centred
    # parameterisation (see `ComposedDistributions.Pool`); a `Normal`/
    # `LogNormal` population would instead be reparameterised non-centred,
    # which lowers to ordinary sampleable rows `as_turing` already handles.
    population = Gamma(2.0, 1.0)
    tree = compose((
        a = uncertain(LogNormal(1.5, 0.4); mu = pool(:mu_pop, population)),
        b = uncertain(LogNormal(1.5, 0.4); mu = pool(:mu_pop, population))))
    data = [[1.0], [2.0]]

    # `as_turing` itself only builds the (lazy) `DynamicPPL.Model` spec; the
    # guard runs when the model body is actually evaluated.
    m = as_turing(as_logdensity(tree, data))
    @test_throws ArgumentError VarInfo(m)
end
