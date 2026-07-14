# AD coverage for a MIXED fixed + estimated composed log-density (the
# fix-vs-estimate facility, #778). A `ComposedLogDensity` whose priors pin some
# leaf parameters to constants and estimate the rest must differentiate cleanly
# w.r.t. ONLY the free (estimated) parameters: the fixed subtree is built once
# at assembly and reused (a constant, zero-derivative), while estimated leaves
# still flow a gradient. The free vector excludes the fixed params, so the
# gradient length is the estimated count, and FD / Mooncake must agree.

@testitem "fixed+estimated logdensity gradient: ForwardDiff" tags=[
    :ad, :forwarddiff] begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: as_logdensity, logdensity, free_dimension
    using ADTypes: AutoForwardDiff
    using DifferentiationInterface: gradient
    using ForwardDiff: ForwardDiff

    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    data = [[0.5, 2.0], [1.0, 3.0], [0.8, 2.5]]
    # Fix onset_admit.scale and admit_death.sigma; estimate the other two.
    priors = build_priors(tree;
        fix = (onset_admit = (scale = 1.0,), admit_death = (sigma = 0.4,)))
    prob = as_logdensity(tree, priors, data)
    @test free_dimension(prob) == 2

    # The free vector is [onset_admit.shape, admit_death.mu] in row order; the
    # fixed scale/sigma never appear in it.
    f(x) = logdensity(prob, x)
    x0 = [2.0, 0.5]
    @test isfinite(f(x0))
    g = gradient(f, AutoForwardDiff(), x0)
    @test g isa AbstractVector && length(g) == 2 && all(isfinite, g)
    @test all(!=(0), g)
end

@testitem "fixed+estimated logdensity gradient: Mooncake reverse" tags=[
    :ad, :mooncake, :mooncake_reverse] begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: as_logdensity, logdensity, free_dimension
    using ADTypes: AutoMooncake, AutoForwardDiff
    using DifferentiationInterface: gradient
    using ForwardDiff: ForwardDiff
    using Mooncake: Mooncake

    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    data = [[0.5, 2.0], [1.0, 3.0], [0.8, 2.5]]
    priors = build_priors(tree;
        fix = (onset_admit = (scale = 1.0,), admit_death = (sigma = 0.4,)))
    prob = as_logdensity(tree, priors, data)
    @test free_dimension(prob) == 2

    f(x) = logdensity(prob, x)
    x0 = [2.0, 0.5]
    ref = gradient(f, AutoForwardDiff(), x0)
    g = gradient(f, AutoMooncake(; config = nothing), x0)
    @test g isa AbstractVector && length(g) == 2 && all(isfinite, g)
    @test isapprox(g, ref; rtol = 1e-6, atol = 1e-8)
end
