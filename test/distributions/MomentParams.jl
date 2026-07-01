@testitem "from_moments constructor and validation" begin
    using Distributions

    d = from_moments(Gamma; mean = 5.0, shape = 2.0)
    @test d isa CensoredDistributions.MomentParams
    @test params(d) == (5.0, 2.0)
    @test CensoredDistributions._moment_names(d) == (:mean, :shape)

    # Promotion of mixed alternative-parameter types.
    dm = from_moments(Gamma; mean = 5, shape = 2.0)
    @test eltype(typeof(dm)) === Float64
    @test all(v -> v isa Float64, params(dm))

    # Invalid derived parameters are rejected through the native family.
    @test_throws Exception from_moments(Gamma; mean = 0.0, shape = 2.0)
    @test_throws Exception from_moments(Gamma; mean = 5.0, shape = -1.0)

    # check_args = false skips validation (the scoring path).
    bad = from_moments(Gamma; mean = -1.0, shape = 2.0, check_args = false)
    @test params(bad) == (-1.0, 2.0)

    # An unregistered family / name set errors clearly.
    @test_throws ArgumentError from_moments(Gamma; cv = 0.5, shape = 2.0)
end

@testitem "MomentParams Gamma is density-identical to native Gamma" begin
    using Distributions

    for (m, k) in [(5.0, 2.0), (1.5, 0.7), (10.0, 4.0)]
        d = from_moments(Gamma; mean = m, shape = k)
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

@testitem "MomentParams LogNormal is density-identical to native" begin
    using Distributions

    # (mean, sd) of the log-normal map to native (mu, sigma).
    for (m, s) in [(5.0, 2.0), (1.0, 0.5), (10.0, 8.0)]
        d = from_moments(LogNormal; mean = m, sd = s)
        s2 = log1p((s / m)^2)
        ref = LogNormal(log(m) - s2 / 2, sqrt(s2))
        @test mean(d) ≈ m
        @test std(d) ≈ s
        @test CensoredDistributions._moment_names(d) == (:mean, :sd)
        for x in [0.5, 1.0, 2.5, 5.0, 9.0]
            @test logpdf(d, x) ≈ logpdf(ref, x)
            @test cdf(d, x) ≈ cdf(ref, x)
        end
        for p in [0.1, 0.5, 0.9]
            @test quantile(d, p) ≈ quantile(ref, p)
        end
    end
end

@testitem "MomentParams params round-trip through the front-door" begin
    using Distributions

    d = from_moments(Gamma; mean = 5.0, shape = 2.0)
    @test CensoredDistributions._param_names(d) == (:mean, :shape)

    # As a composed leaf, params_table lists the alternative names.
    tree = compose((onset_admit = from_moments(Gamma; mean = 5.0, shape = 2.0),))
    tbl = params_table(tree)
    @test collect(tbl.param) == [:mean, :shape]
    @test collect(tbl.value) == [5.0, 2.0]
    @test all(s -> s == (0.0, Inf), tbl.support)

    # update replaces the alternative params and reconstructs the same leaf type.
    tree2 = update(tree, (onset_admit = (mean = 8.0, shape = 3.0),))
    leaf = event(tree2, :onset_admit)
    @test leaf isa CensoredDistributions.MomentParams
    @test mean(leaf) ≈ 8.0
    @test params(leaf) == (8.0, 3.0)

    # A LogNormal moment leaf round-trips its (mean, sd) names too.
    tree_ln = compose((gen = from_moments(LogNormal; mean = 4.0, sd = 1.0),))
    tbl_ln = params_table(tree_ln)
    @test collect(tbl_ln.param) == [:mean, :sd]
    upd = update(tree_ln, (gen = (mean = 6.0, sd = 2.0),))
    @test mean(event(upd, :gen)) ≈ 6.0
end

@testitem "MomentParams mean-coupled prior composes and scores" begin
    using Distributions

    # A mean prior is a prior on a FREE parameter, so build_priors picks a
    # positive-truncated default for the mean (and the shape / sd).
    tree = compose((onset_admit = from_moments(Gamma; mean = 5.0, shape = 2.0),))
    tbl = params_table(tree)
    priors = build_priors(tbl)
    @test haskey(priors, :onset_admit)
    @test haskey(priors.onset_admit, :mean)
    @test haskey(priors.onset_admit, :shape)
    @test minimum(priors.onset_admit.mean) == 0
    @test minimum(priors.onset_admit.shape) == 0

    # The LogNormal `sd` alternative parameter is positive-truncated too.
    tree_ln = compose((gen = from_moments(LogNormal; mean = 4.0, sd = 1.0),))
    priors_ln = build_priors(params_table(tree_ln))
    @test minimum(priors_ln.gen.mean) == 0
    @test minimum(priors_ln.gen.sd) == 0

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

@testitem "MomentParams unchecked reconstruction is scoring-safe" begin
    using Distributions

    # The reconstruction ctor (the scoring path) skips validation, so an
    # out-of-support proposal builds a leaf rather than throwing on construction.
    leaf = from_moments(Gamma; mean = 5.0, shape = 2.0)
    ctor = CensoredDistributions._leaf_ctor(leaf)
    @test CensoredDistributions._ctor_has_check_args(ctor, (-1.0, 2.0))
    bad = ctor(-1.0, 2.0; check_args = false)
    @test bad isa CensoredDistributions.MomentParams
    @test params(bad) == (-1.0, 2.0)

    # An out-of-support draw yields -Inf, not a throw, for a valid leaf.
    @test logpdf(leaf, -1.0) == -Inf
end

@testitem "MomentParams AD gradient is finite through the moments" begin
    using Distributions
    import ForwardDiff

    # The log density differentiates wrt the (mean, shape) pair, the gradient
    # flowing through the derived scale into the native Gamma density.
    f = θ -> logpdf(from_moments(Gamma; mean = θ[1], shape = θ[2]), 4.0)
    g = ForwardDiff.gradient(f, [5.0, 2.0])
    @test all(isfinite, g)
    @test length(g) == 2

    # A LogNormal moment leaf differentiates wrt (mean, sd) too.
    fln = θ -> logpdf(from_moments(LogNormal; mean = θ[1], sd = θ[2]), 4.0)
    gln = ForwardDiff.gradient(fln, [4.0, 1.0])
    @test all(isfinite, gln)
end

@testitem "MomentParams satisfies the leaf interface" begin
    using CensoredDistributions.TestUtils
    using Distributions

    d = from_moments(Gamma; mean = 5.0, shape = 2.0)
    test_interface(d; draw = 4.0, univariate = true, overall = :scalar)
end
