@testitem "truncate_to_horizon single-delay normaliser" begin
    using Distributions

    # The right-truncation log-normaliser of truncate_to_horizon must equal
    # the andv index-case term -logcdf(delay, window): logpdf of the
    # truncated object is logpdf(delay, x) - logcdf(delay, window).
    delay = LogNormal(1.5, 0.5)
    for window in (2.0, 6.0, 12.0)
        td = truncate_to_horizon(delay, window)
        for x in 0.5:0.7:min(window, 10.0)
            @test logpdf(td, x) ≈ logpdf(delay, x) - logcdf(delay, window) atol=1e-10
            @test cdf(td, x) ≈ cdf(delay, x) / cdf(delay, window) atol=1e-10
        end
        # Mass beyond the window is excluded.
        @test cdf(td, window + 1.0) ≈ 1.0 atol=1e-10
        @test pdf(td, window + 1.0) == 0.0
    end
end

@testitem "truncate_to_horizon convolved-delay normaliser" begin
    using Distributions

    # Convolved input selects the convolved-chain denominator: the
    # log-normaliser equals -logcdf(convolution, window).
    conv = convolve_distributions(LogNormal(1.5, 0.5), Gamma(2.0, 1.0))
    window = 9.0
    td = truncate_to_horizon(conv, window)
    for x in 1.0:1.5:8.5
        @test logpdf(td, x) ≈ logpdf(conv, x) - logcdf(conv, window) atol=1e-7
        @test cdf(td, x) ≈ cdf(conv, x) / cdf(conv, window) atol=1e-7
    end
end

@testitem "truncate_to_horizon empty window" begin
    using Distributions

    # A non-positive window (horizon already passed) gives an empty-support
    # truncation: no mass, matching the andv floatmin guard on a zero
    # denominator. truncated clamps upper up to lower.
    delay = Gamma(2.0, 1.0)
    td = truncate_to_horizon(delay, -1.0)
    @test minimum(td) == maximum(td)
    @test cdf(td, 5.0) == 1.0   # degenerate point mass at the lower edge
end

@testitem "truncate_chain reproduces andv index vs sourced terms" begin
    using Distributions

    # Faithful reproduction of the two andv truncation_model @addlogprob!
    # terms. Index cases: single observed delay, single-delay denominator.
    # Sourced cases: intermediate event unobserved, convolved denominator.
    inc_dist = LogNormal(1.5, 0.5)
    delta_dist = Gamma(2.0, 1.0)
    obs_time = 10.0

    # Index: anchored at infection, observed delay is inc_dist alone.
    T_inf = 4.0
    window_inc = obs_time - T_inf
    index = truncate_chain((inc_dist,), (), window_inc)
    # andv: -sum(logcdf.(inc_dist, obs_time - T_inf))
    expected_index_norm = -logcdf(inc_dist, window_inc)
    for x in 0.5:0.8:window_inc
        # logpdf has the andv normaliser added in.
        @test logpdf(index, x) ≈
              logpdf(inc_dist, x) + expected_index_norm atol=1e-10
    end

    # Sourced: anchored at onset, observed delay is the unobserved
    # convolution inc_dist ⊕ delta_dist.
    T_onset = 2.0
    window_onset = obs_time - T_onset
    sourced = truncate_chain((inc_dist, delta_dist), (false,), window_onset)
    conv = convolve_distributions(inc_dist, delta_dist)
    # andv: cdf(ConvolvedDelays(inc_dist, delta_dist), obs_time - T_onset)
    expected_sourced_norm = -logcdf(conv, window_onset)
    for x in 1.0:1.5:window_onset
        @test logpdf(sourced, x) ≈
              logpdf(conv, x) + expected_sourced_norm atol=1e-7
    end

    # The two denominators genuinely differ for the same window.
    @test logcdf(inc_dist, window_onset) != logcdf(conv, window_onset)
end

@testitem "truncate_chain collapses unobserved runs" begin
    using Distributions

    a = LogNormal(1.0, 0.5)
    b = Gamma(2.0, 1.0)
    c = Exponential(2.0)
    window = 9.0

    # An observed boundary closes the run before it; the truncation
    # distribution is the convolution of the trailing unobserved segments.

    # All unobserved -> full convolution.
    full = truncate_chain((a, b, c), (false, false), window)
    conv_full = convolve_distributions(a, b, c)
    @test logpdf(full, 4.0) ≈
          logpdf(conv_full, 4.0) - logcdf(conv_full, window) atol=1e-6

    # Middle observed -> trailing single segment c.
    trailing = truncate_chain((a, b, c), (false, true), window)
    @test logpdf(trailing, 2.0) ≈
          logpdf(truncate_to_horizon(c, window), 2.0) atol=1e-10

    # First observed -> trailing convolution of b and c.
    bc = convolve_distributions(b, c)
    tail_conv = truncate_chain((a, b, c), (true, false), window)
    @test logpdf(tail_conv, 3.0) ≈
          logpdf(bc, 3.0) - logcdf(bc, window) atol=1e-6
end

@testitem "truncate_chain validation" begin
    using Distributions

    a = Gamma(2.0, 1.0)
    b = LogNormal(1.0, 0.5)
    # observed must have one fewer entry than segments.
    @test_throws ArgumentError truncate_chain((a, b), (true, false), 5.0)
    @test_throws ArgumentError truncate_chain((a, b), (), 5.0)
    @test_throws ArgumentError truncate_chain((), (), 5.0)
end

@testitem "truncate_chain right-truncation matches Monte Carlo" begin
    using Distributions, Random, Statistics
    rng = MersenneTwister(20240617)

    # The right-truncated convolved chain should match samples from the
    # convolution discarded above the window. Compare the truncated CDF at a
    # grid against the empirical CDF of accepted Monte Carlo draws.
    inc_dist = LogNormal(1.2, 0.4)
    delta_dist = Gamma(2.0, 0.8)
    window = 8.0
    chain = truncate_chain((inc_dist, delta_dist), (false,), window)

    # Direct samples of the total delay, kept only when below the window.
    n = 400_000
    totals = [rand(rng, inc_dist) + rand(rng, delta_dist) for _ in 1:n]
    accepted = filter(t -> t <= window, totals)
    @test length(accepted) > 10_000

    for x in (2.0, 4.0, 6.0)
        mc_cdf = mean(accepted .<= x)
        @test cdf(chain, x) ≈ mc_cdf atol=5e-3
    end

    # The truncated mean matches the accepted-sample mean.
    @test mean(rand(rng, chain, 100_000)) ≈ mean(accepted) atol=0.1
end
