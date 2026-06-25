# Tests for the PPL-agnostic LogDensityProblems layer (#734): the core flat <->
# nested codec, the assembled ComposedLogDensity, and the weakdep extensions
# (LogDensityProblems / DensityInterface / Bijectors). The load-bearing checks
# are the codec round-trips and that the LDP log-density EQUALS the Turing /
# DynamicPPL log-joint on the same parameters (the two backends must agree), so
# a posterior is interchangeable across backends.

@testitem "codec: flatten/unflatten round-trip (basic)" begin
    using CensoredDistributions, Distributions, Tables

    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))

    @test flat_dimension(tree) ==
          length(Tables.getcolumn(params_table(tree), :edge))

    # The table's `value` column IS the flat layout; unflatten -> update-shaped
    # nested NamedTuple -> flatten reproduces it exactly.
    x = collect(Tables.getcolumn(params_table(tree), :value))
    nt = unflatten(tree, x)
    @test flatten(tree, nt) == x
    # The nested NamedTuple reconstructs the template via the core `update`.
    @test update(tree, nt) == tree
    # Round-trip through unflatten again is stable.
    @test flatten(tree, unflatten(tree, x)) == x
end

@testitem "codec: thin weight and shared tag round-trip" begin
    using CensoredDistributions, Distributions, Tables

    # A thinned leaf surfaces a `thin` row after the delay params.
    t = compose((cases = thin(LogNormal(1.5, 0.4), 0.3),))
    x = collect(Tables.getcolumn(params_table(t), :value))
    nt = unflatten(t, x)
    @test haskey(nt.cases, :thin)
    @test flatten(t, nt) == x
    @test update(t, nt) == t

    # A tag shared across two edges is inventoried once; the codec keeps one
    # flat slot for it and nests it under the tag, so `update` round-trips.
    shared_tree = compose((
        first = shared(:inc, Gamma(2.0, 1.0)),
        second = shared(:inc, Gamma(2.0, 1.0))))
    xs = collect(Tables.getcolumn(params_table(shared_tree), :value))
    nts = unflatten(shared_tree, xs)
    @test haskey(nts, :inc)
    @test flatten(shared_tree, nts) == xs
    @test update(shared_tree, nts) == shared_tree
end

@testitem "logdensity: constrained = priors + likelihood" begin
    using CensoredDistributions, Distributions, Tables

    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    priors = build_priors(tree)
    data = [[0.5, 2.0], [1.0, 3.0], [0.8, 2.5]]
    prob = as_logdensity(tree, priors, data)

    x = collect(Tables.getcolumn(params_table(tree), :value))
    ld = CensoredDistributions.logdensity(prob, x)

    flat_priors = flatten(tree, priors)
    manual_prior = sum(logpdf(flat_priors[i], x[i]) for i in eachindex(x))
    d = update(tree, unflatten(tree, x))
    manual_lik = sum(logpdf(d, y) for y in data)
    @test ld ≈ manual_prior + manual_lik

    # The data-only assembler form defaults priors to build_priors(tree).
    prob2 = as_logdensity(tree, data)
    @test CensoredDistributions.logdensity(prob2, x) ≈ ld
end

@testitem "LogDensityProblems: dimension / capabilities / transform" begin
    using CensoredDistributions, Distributions, Tables
    using Bijectors: bijector
    using LogDensityProblems

    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    prob = as_logdensity(tree, build_priors(tree),
        [[0.5, 2.0], [1.0, 3.0]])

    @test LogDensityProblems.dimension(prob) == flat_dimension(tree)
    @test LogDensityProblems.capabilities(prob) isa
          LogDensityProblems.LogDensityOrder{0}

    # The prior-driven transform round-trips: forward through bijector(prior),
    # back through to_constrained, recovering the constrained point.
    xc = collect(Tables.getcolumn(params_table(tree), :value))
    flatp = CensoredDistributions.flat_priors(prob)
    z = [bijector(flatp[i])(xc[i]) for i in eachindex(xc)]
    x2, logj = CensoredDistributions.to_constrained(prob, z)
    @test x2 ≈ xc

    # The unconstrained log-density is the constrained one + the log-Jacobian.
    ld_unc = LogDensityProblems.logdensity(prob, z)
    ld_con = CensoredDistributions.logdensity(prob, xc)
    @test ld_unc ≈ ld_con + logj
end

@testitem "DensityInterface: trait + logdensityof" begin
    using CensoredDistributions, Distributions, Tables
    using DensityInterface

    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    prob = as_logdensity(tree, build_priors(tree), [[0.5, 2.0]])

    @test DensityInterface.DensityKind(prob) isa DensityInterface.IsDensity
    x = collect(Tables.getcolumn(params_table(tree), :value))
    @test DensityInterface.logdensityof(prob, x) ≈
          CensoredDistributions.logdensity(prob, x)
end

@testitem "gradient: ADgradient over the LDP is finite" begin
    using CensoredDistributions, Distributions, Tables
    using Bijectors: bijector
    using LogDensityProblems, LogDensityProblemsAD
    using ADTypes: AutoForwardDiff
    using ForwardDiff

    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    prob = as_logdensity(tree, build_priors(tree),
        [[0.5, 2.0], [1.0, 3.0]])

    xc = collect(Tables.getcolumn(params_table(tree), :value))
    flatp = CensoredDistributions.flat_priors(prob)
    z = [bijector(flatp[i])(xc[i]) for i in eachindex(xc)]

    adprob = ADgradient(AutoForwardDiff(), prob)
    v, g = LogDensityProblems.logdensity_and_gradient(adprob, z)
    @test v ≈ LogDensityProblems.logdensity(prob, z)
    @test length(g) == flat_dimension(tree)
    @test all(isfinite, g)
end

@testitem "CONSISTENCY: LDP logdensity == Turing log-joint" tags=[:turing] begin
    using CensoredDistributions, Distributions, Random
    using DynamicPPL, Turing
    using FlexiChains: FlexiChains, VNChain, Extra

    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    priors = build_priors(tree)
    data = [[0.5, 2.0], [1.0, 3.0], [0.8, 2.5]]
    prob = as_logdensity(tree, priors, data)

    # The Turing path: sample the SAME priors via composed_parameters_model and
    # score the SAME per-record likelihood. Its stored per-draw `logjoint` is
    # the reference the LDP must reproduce.
    @model function fit(t, p, ys)
        d ~ to_submodel(composed_parameters_model(t, p))
        for y in ys
            DynamicPPL.@addlogprob! logpdf(d, y)
        end
    end
    m = fit(tree, priors, data)

    Random.seed!(11)
    chain = sample(m, NUTS(), 40; progress = false, chain_type = VNChain)
    draws = param_draws(tree, chain)
    ljs = vec(chain[Extra(:logjoint)])

    # The codec maps each Turing draw's named params to the flat layout, and the
    # LDP constrained logdensity there equals the model's stored log-joint, so a
    # DynamicPPL chain and an LDP/HMC chain are interchangeable.
    diffs = [abs(CensoredDistributions.logdensity(prob, flatten(tree, draws[i]))
                 - ljs[i]) for i in eachindex(draws)]
    @test maximum(diffs) < 1e-8
end

@testitem "RECOVERY: AdvancedHMC off the LDP (no Turing)" tags=[:turing] begin
    using CensoredDistributions, Distributions, Random, Statistics, Tables
    using Bijectors: bijector
    using LogDensityProblems, LogDensityProblemsAD
    using ADTypes: AutoForwardDiff
    using ForwardDiff
    using AdvancedHMC

    # Simulate from a known truth; recover the parameters by sampling the
    # LogDensityProblem directly with AdvancedHMC (no Turing/DynamicPPL).
    Random.seed!(2024)
    truth = compose((onset_admit = Gamma(3.0, 1.0),
        admit_death = LogNormal(0.6, 0.4)))
    data = [rand(truth) for _ in 1:300]

    template = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.5)))
    prob = as_logdensity(template, build_priors(template), data)
    adprob = ADgradient(AutoForwardDiff(), prob)
    D = LogDensityProblems.dimension(prob)

    xc0 = collect(Tables.getcolumn(params_table(template), :value))
    flatp = CensoredDistributions.flat_priors(prob)
    z0 = [bijector(flatp[i])(xc0[i]) for i in eachindex(xc0)]

    metric = DiagEuclideanMetric(D)
    ham = Hamiltonian(metric, adprob)
    eps = find_good_stepsize(ham, z0)
    integrator = Leapfrog(eps)
    kernel = HMCKernel(
        Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric),
        StepSizeAdaptor(0.8, integrator))
    samples,
    _ = sample(ham, kernel, z0, 800, adaptor, 400;
        progress = false, verbose = false)

    # Map unconstrained draws back to constrained, named parameters via the
    # codec + inverse transform, and check the posterior means recover truth.
    post = [first(CensoredDistributions.to_constrained(prob, z))
            for z in samples[401:end]]
    nt = unflatten(template, mean(post))
    @test abs(nt.onset_admit.shape - 3.0) < 1.0
    @test abs(nt.admit_death.mu - 0.6) < 0.25
    @test abs(nt.admit_death.sigma - 0.4) < 0.2
end

@testitem "param_draws labels match the codec names" tags=[:turing] begin
    using CensoredDistributions, Distributions, Random
    using DynamicPPL, Turing
    using FlexiChains: VNChain

    # A posterior read via param_draws is keyed exactly like the codec's nested
    # NamedTuple (the VarName edge paths), so the same draw flattens through the
    # codec and scores on the LDP without renaming.
    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    priors = build_priors(tree)

    @model function fit(t, p, ys)
        d ~ to_submodel(composed_parameters_model(t, p))
        for y in ys
            DynamicPPL.@addlogprob! logpdf(d, y)
        end
    end
    m = fit(tree, priors, [[0.5, 2.0], [1.0, 3.0]])
    Random.seed!(3)
    chain = sample(m, NUTS(), 20; progress = false, chain_type = VNChain)

    draw = first(param_draws(tree, chain))
    # The codec keys are the edge paths; flatten consumes the draw directly.
    @test keys(draw.onset_admit) == (:shape, :scale)
    @test keys(draw.admit_death) == (:mu, :sigma)
    @test length(flatten(tree, draw)) == flat_dimension(tree)
    # update rebuilds the template structure from a param_draws NamedTuple.
    @test update(tree, draw) isa CensoredDistributions.Parallel
end
