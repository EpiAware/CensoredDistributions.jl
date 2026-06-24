@testitem "MeanGamma constructor and validation" begin
    using Distributions

    d = mean_gamma(5.0, 2.0)
    @test d isa CensoredDistributions.MeanGamma
    @test d.mean == 5.0
    @test d.shape == 2.0
    @test mean_gamma(5.0, 2.0) == MeanGamma(5.0, 2.0)

    # Promotion of mixed mean/shape types.
    dm = MeanGamma(5, 2.0)
    @test dm.mean isa Float64
    @test dm.shape isa Float64

    # mean and shape must be positive.
    @test_throws ArgumentError mean_gamma(0.0, 2.0)
    @test_throws ArgumentError mean_gamma(-1.0, 2.0)
    @test_throws ArgumentError mean_gamma(5.0, 0.0)
    @test_throws ArgumentError mean_gamma(5.0, -2.0)

    # check_args = false skips the positivity checks (the scoring path).
    @test MeanGamma(-1.0, 2.0; check_args = false).mean == -1.0
end

@testitem "MeanGamma is density-identical to the equivalent Gamma" begin
    using Distributions

    for (m, k) in [(5.0, 2.0), (1.5, 0.7), (10.0, 4.0)]
        d = mean_gamma(m, k)
        ref = Gamma(k, m / k)   # shape, scale = mean / shape
        @test mean(d) ≈ m
        @test var(d) ≈ m^2 / k
        @test mean(d) ≈ mean(ref)
        @test var(d) ≈ var(ref)
        @test minimum(d) == minimum(ref)
        @test maximum(d) == maximum(ref)
        for x in [0.5, 1.0, 2.5, 5.0, 9.0]
            @test logpdf(d, x) ≈ logpdf(ref, x)
            @test pdf(d, x) ≈ pdf(ref, x)
            @test cdf(d, x) ≈ cdf(ref, x)
            @test logcdf(d, x) ≈ logcdf(ref, x)
            @test ccdf(d, x) ≈ ccdf(ref, x)
            @test logccdf(d, x) ≈ logccdf(ref, x)
        end
        for p in [0.1, 0.5, 0.9]
            @test quantile(d, p) ≈ quantile(ref, p)
        end
    end
end

@testitem "MeanGamma params round-trip through the front-door" begin
    using Distributions

    d = mean_gamma(5.0, 2.0)
    @test params(d) == (5.0, 2.0)
    @test CensoredDistributions._param_names(d) == (:mean, :shape)

    # As a composed leaf, params_table lists mean and shape (not shape, scale).
    tree = compose((onset_admit = mean_gamma(5.0, 2.0),))
    tbl = params_table(tree)
    @test collect(tbl.param) == [:mean, :shape]
    @test collect(tbl.value) == [5.0, 2.0]
    # The leaf's variate support is the positive half-line.
    @test all(s -> s == (0.0, Inf), tbl.support)

    # update replaces mean and shape and reconstructs the same leaf type.
    tree2 = update(tree, (onset_admit = (mean = 8.0, shape = 3.0),))
    leaf = event(tree2, :onset_admit)
    @test leaf isa CensoredDistributions.MeanGamma
    @test mean(leaf) ≈ 8.0
    @test params(leaf) == (8.0, 3.0)
end

@testitem "MeanGamma mean-coupled prior composes and scores" begin
    using Distributions

    # A mean prior is a prior on a FREE parameter, so build_priors picks a
    # positive-truncated default for the mean (and the shape).
    tree = compose((onset_admit = mean_gamma(5.0, 2.0),))
    tbl = params_table(tree)
    priors = build_priors(tbl)
    @test haskey(priors, :onset_admit)
    @test haskey(priors.onset_admit, :mean)
    @test haskey(priors.onset_admit, :shape)
    @test minimum(priors.onset_admit.mean) == 0
    @test minimum(priors.onset_admit.shape) == 0

    # An explicit mean prior overrides only that parameter.
    mean_prior = truncated(Normal(5.0, 1.0); lower = 0)
    overridden = build_priors(tbl;
        priors = (onset_admit = (mean = mean_prior,),))
    @test overridden.onset_admit.mean === mean_prior

    # The composed leaf scores: a finite log density at an in-support point.
    leaf = event(tree, :onset_admit)
    @test isfinite(logpdf(leaf, 4.0))
    @test logpdf(leaf, 4.0) ≈ logpdf(Gamma(2.0, 2.5), 4.0)
end

@testitem "MeanGamma satisfies the leaf interface" begin
    using CensoredDistributions.TestUtils

    d = mean_gamma(5.0, 2.0)
    test_interface(d; draw = 4.0, univariate = true, overall = :scalar)
end
