# External censoring wrappers over composers (PR3c, #334). The semantics:
# combine first, then censor (#329). A Sequential collapses to the convolution
# of its steps (its observed total) before censoring; a Parallel distributes
# the wrapper into every branch; Convolved/Competing are already univariate and
# flow through the existing wrapper methods unchanged.

@testitem "observed_distribution lowers each composer to its scalar" begin
    using Distributions

    # Sequential -> convolution of its steps.
    seq = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    obs = observed_distribution(seq)
    conv = convolve_distributions(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    @test obs isa CensoredDistributions.Convolved
    @test cdf(obs, 4.0) ≈ cdf(conv, 4.0)

    # A single-step Sequential lowers to that step itself (no convolution).
    @test observed_distribution(Sequential(Gamma(2.0, 1.0))) == Gamma(2.0, 1.0)

    # Nested Sequential flattens through to its leaves.
    nested = Sequential(Gamma(2.0, 1.0),
        Sequential(LogNormal(0.5, 0.4), Gamma(1.5, 1.0)))
    @test length(get_dist(observed_distribution(nested))) == 3

    # Univariate composers / dists are returned unchanged.
    @test observed_distribution(conv) === conv
    @test observed_distribution(Gamma(2.0, 1.0)) == Gamma(2.0, 1.0)

    # A Sequential step that is a Parallel has no single observed time.
    @test_throws ArgumentError observed_distribution(
        Sequential(Gamma(2.0, 1.0), Parallel(Gamma(1.0, 1.0), Gamma(1.0, 1.0))))
end

@testitem "Convolved/Competing flow through wrappers (already univariate)" begin
    using Distributions

    conv = convolve_distributions(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    dic = double_interval_censored(
        conv; primary_event = Uniform(0, 1), interval = 1.0)
    @test dic isa CensoredDistributions.IntervalCensored
    @test cdf(dic, 4.0) isa Real

    comp = Competing(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.7))
    ic = interval_censored(comp, 1.0)
    @test ic isa CensoredDistributions.IntervalCensored
    @test get_dist(ic) === comp
    @test pdf(ic, 2.0) ≈ cdf(comp, 3.0) - cdf(comp, 2.0)
end

@testitem "Wrapping a Sequential censors its collapsed total" begin
    using Distributions

    seq = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    conv = convolve_distributions(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))

    # Each wrapper over the chain equals the wrapper over its observed total.
    pc_seq = primary_censored(seq, Uniform(0, 1))
    pc_conv = primary_censored(conv, Uniform(0, 1))
    @test cdf(pc_seq, 3.0) ≈ cdf(pc_conv, 3.0)

    ic_seq = interval_censored(seq, 1.0)
    ic_conv = interval_censored(conv, 1.0)
    @test pdf(ic_seq, 3.0) ≈ pdf(ic_conv, 3.0)

    dic_seq = double_interval_censored(
        seq; primary_event = Uniform(0, 1), upper = 10.0, interval = 1.0)
    dic_conv = double_interval_censored(
        conv; primary_event = Uniform(0, 1), upper = 10.0, interval = 1.0)
    @test cdf(dic_seq, 4.0) ≈ cdf(dic_conv, 4.0)

    # Keyword-only primary_censored over a Sequential uses the default primary.
    @test cdf(primary_censored(seq), 3.0) ≈ cdf(primary_censored(conv), 3.0)
end

@testitem "interval_censored(Sequential) matches a Monte Carlo total" begin
    using Distributions, Random, Statistics

    Random.seed!(20240601)
    seq = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    ic = interval_censored(seq, 1.0)

    # The chain total is the sum of the two step draws; its day-discretised
    # PMF over [k, k+1) is the reference.
    N = 2_000_000
    tot = [rand(Gamma(2.0, 1.0)) + rand(LogNormal(0.5, 0.4)) for _ in 1:N]
    for k in (1.0, 2.0, 3.0, 4.0)
        mc = mean(k .<= tot .< (k + 1))
        @test isapprox(mc, pdf(ic, k); atol = 3e-3)
    end
end

@testitem "Wrapping a Parallel distributes into the branches" begin
    using Distributions

    par = Parallel(Gamma(2.0, 1.0), LogNormal(1.0, 0.5))

    ic = interval_censored(par, 1.0)
    @test ic isa CensoredDistributions.Parallel
    @test ic.components[1] isa CensoredDistributions.IntervalCensored
    @test ic.components[2] isa CensoredDistributions.IntervalCensored
    # The distributed parallel scores each branch's censored value independently.
    @test logpdf(ic, [2.0, 3.0]) ≈
          logpdf(interval_censored(Gamma(2.0, 1.0), 1.0), 2.0) +
          logpdf(interval_censored(LogNormal(1.0, 0.5), 1.0), 3.0)

    pc = primary_censored(par, Uniform(0, 1))
    @test pc isa CensoredDistributions.Parallel
    @test pc.components[1] isa CensoredDistributions.PrimaryCensored

    dic = double_interval_censored(
        par; primary_event = Uniform(0, 1), interval = 1.0)
    @test dic isa CensoredDistributions.Parallel
    @test logpdf(dic, [2.0, 3.0]) ≈
          logpdf(
        double_interval_censored(Gamma(2.0, 1.0);
            primary_event = Uniform(0, 1), interval = 1.0), 2.0) +
          logpdf(
        double_interval_censored(LogNormal(1.0, 0.5);
            primary_event = Uniform(0, 1), interval = 1.0),
        3.0)
end

@testitem "double_interval_censored(convolve_distributions(...), primary)" begin
    using Distributions

    # The canonical #329 example: combine first, then censor.
    d = double_interval_censored(
        convolve_distributions(Gamma(2.0, 1.0), LogNormal(0.5, 0.4));
        primary_event = Uniform(0, 1), upper = 10.0, interval = 1.0)
    @test d isa CensoredDistributions.IntervalCensored
    @test cdf(d, 5.0) isa Real
    @test 0.0 <= cdf(d, 5.0) <= 1.0
    @test pdf(d, 3.0) >= 0.0
end
