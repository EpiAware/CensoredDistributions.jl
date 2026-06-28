@testitem "truncated single-delay right-truncation normaliser" begin
    using Distributions

    # The right-truncation log-normaliser of `truncated(delay; upper)` must equal
    # the andv index-case term -logcdf(delay, window): logpdf of the truncated
    # object is logpdf(delay, x) - logcdf(delay, window).
    delay = LogNormal(1.5, 0.5)
    for window in (2.0, 6.0, 12.0)
        td = truncated(delay; upper = window)
        for x in 0.5:0.7:min(window, 10.0)
            @test logpdf(td, x) ≈ logpdf(delay, x) - logcdf(delay, window) atol=1e-10
            @test cdf(td, x) ≈ cdf(delay, x) / cdf(delay, window) atol=1e-10
        end
        # Mass beyond the window is excluded.
        @test cdf(td, window + 1.0) ≈ 1.0 atol=1e-10
        @test pdf(td, window + 1.0) == 0.0
    end
end

@testitem "truncated convolved-delay right-truncation normaliser" begin
    using Distributions

    # A convolved input selects the convolved-chain denominator: the
    # log-normaliser equals -logcdf(convolution, window).
    conv = convolved(LogNormal(1.5, 0.5), Gamma(2.0, 1.0))
    window = 9.0
    td = truncated(conv; upper = window)
    for x in 1.0:1.5:8.5
        @test logpdf(td, x) ≈ logpdf(conv, x) - logcdf(conv, window) atol=1e-7
        @test cdf(td, x) ≈ cdf(conv, x) / cdf(conv, window) atol=1e-7
    end
end

@testitem "truncated empty-window clamp primitive" begin
    using Distributions
    using CensoredDistributions: _truncate_window

    # A non-positive window (horizon already passed) gives an empty-support
    # truncation: no mass, matching the andv floatmin guard on a zero
    # denominator. The internal `_truncate_window` clamps upper up to the
    # distribution minimum (AD-safe), giving a degenerate point mass there.
    delay = Gamma(2.0, 1.0)
    td = _truncate_window(delay, -1.0)
    @test minimum(td) == maximum(td)
    @test cdf(td, 5.0) == 1.0   # degenerate point mass at the lower edge
end

@testitem "truncated reproduces andv index vs sourced chain terms" begin
    using Distributions
    using CensoredDistributions: _collapse_to_observation

    # Faithful reproduction of the two andv truncation_model @addlogprob!
    # terms. Index cases: single observed delay, single-delay denominator.
    # Sourced cases: intermediate event unobserved, convolved denominator. The
    # chain collapse + `truncated(...; upper)` is the single truncation verb.
    inc_dist = LogNormal(1.5, 0.5)
    delta_dist = Gamma(2.0, 1.0)
    obs_time = 10.0

    # Index: anchored at infection, observed delay is inc_dist alone.
    T_inf = 4.0
    window_inc = obs_time - T_inf
    index = truncated(
        _collapse_to_observation((inc_dist,), ()); upper = window_inc)
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
    sourced = truncated(
        _collapse_to_observation((inc_dist, delta_dist), (false,));
        upper = window_onset)
    conv = convolved(inc_dist, delta_dist)
    # andv: cdf(ConvolvedDelays(inc_dist, delta_dist), obs_time - T_onset)
    expected_sourced_norm = -logcdf(conv, window_onset)
    for x in 1.0:1.5:window_onset
        @test logpdf(sourced, x) ≈
              logpdf(conv, x) + expected_sourced_norm atol=1e-7
    end

    # The two denominators genuinely differ for the same window.
    @test logcdf(inc_dist, window_onset) != logcdf(conv, window_onset)
end

@testitem "chain collapse picks the trailing unobserved run" begin
    using Distributions
    using CensoredDistributions: _collapse_to_observation

    a = LogNormal(1.0, 0.5)
    b = Gamma(2.0, 1.0)
    c = Exponential(2.0)
    window = 9.0

    # An observed boundary closes the run before it; the truncation
    # distribution is the convolution of the trailing unobserved segments.

    # All unobserved -> full convolution.
    full = truncated(
        _collapse_to_observation((a, b, c), (false, false)); upper = window)
    conv_full = convolved(a, b, c)
    @test logpdf(full, 4.0) ≈
          logpdf(conv_full, 4.0) - logcdf(conv_full, window) atol=1e-6

    # Middle observed -> trailing single segment c.
    trailing = truncated(
        _collapse_to_observation((a, b, c), (false, true)); upper = window)
    @test logpdf(trailing, 2.0) ≈
          logpdf(truncated(c; upper = window), 2.0) atol=1e-10

    # First observed -> trailing convolution of b and c.
    bc = convolved(b, c)
    tail_conv = truncated(
        _collapse_to_observation((a, b, c), (true, false)); upper = window)
    @test logpdf(tail_conv, 3.0) ≈
          logpdf(bc, 3.0) - logcdf(bc, window) atol=1e-6
end

@testitem "chain collapse validation" begin
    using Distributions
    using CensoredDistributions: _collapse_to_observation

    a = Gamma(2.0, 1.0)
    b = LogNormal(1.0, 0.5)
    # observed must have one fewer entry than segments.
    @test_throws ArgumentError _collapse_to_observation((a, b), (true, false))
    @test_throws ArgumentError _collapse_to_observation((a, b), ())
    @test_throws ArgumentError _collapse_to_observation((), ())
end

@testitem "truncated δ-bounded window normaliser" begin
    using Distributions

    # A finite window `[lower, upper]` has log-normaliser
    # log(cdf(delay, upper) - cdf(delay, lower)), the finite-window mass.
    delay = LogNormal(1.5, 0.5)
    for (upper, δ) in ((6.0, 4.0), (10.0, 3.0), (8.0, 6.5))
        lower = upper - δ
        td = truncated(delay; lower = lower, upper = upper)
        lognorm = log(cdf(delay, upper) - cdf(delay, lower))
        for x in (lower + 0.2):0.7:(upper - 0.1)
            @test logpdf(td, x) ≈ logpdf(delay, x) - lognorm atol=1e-10
        end
        # Mass below the lower edge and above the upper edge is excluded.
        @test pdf(td, lower - 0.5) == 0.0
        @test pdf(td, upper + 0.5) == 0.0
    end
end

@testitem "truncated upper-only is the no-lower-edge special case" begin
    using Distributions

    # The upper-only `truncated(delay; upper)` is the δ-bounded form with the
    # lower edge at the distribution minimum (`cdf(minimum) = 0`), so the density
    # is byte-identical.
    for delay in (LogNormal(1.5, 0.5), Gamma(2.0, 1.0),
        convolved(LogNormal(1.2, 0.4), Gamma(2.0, 0.8)))
        for upper in (3.0, 6.0, 12.0)
            th = truncated(delay; upper = upper)
            wide = truncated(delay; lower = minimum(delay), upper = upper)
            for x in 0.5:0.9:(upper - 0.1)
                @test logpdf(wide, x) ≈ logpdf(th, x) atol=1e-12
            end
        end
    end
end

@testitem "truncated δ-bounded window matches Monte Carlo" begin
    using Distributions, Random, Statistics
    rng = MersenneTwister(20240618)

    # The δ-bounded truncated delay matches samples kept only inside the finite
    # window [lower, upper].
    delay = LogNormal(1.2, 0.4)
    upper, δ = 8.0, 5.0
    lower = upper - δ
    td = truncated(delay; lower = lower, upper = upper)

    n = 600_000
    draws = rand(rng, delay, n)
    accepted = filter(t -> lower <= t <= upper, draws)
    @test length(accepted) > 10_000
    for x in (lower + 0.5, (lower + upper) / 2, upper - 0.5)
        @test cdf(td, x) ≈ mean(accepted .<= x) atol=5e-3
    end
end

@testitem "truncated δ-bounded AD over the params (ForwardDiff)" begin
    using Distributions, ForwardDiff

    # The δ-bounded normaliser cdf(upper) - cdf(lower) must differentiate w.r.t.
    # the delay params. LogNormal `logcdf` is ForwardDiff-friendly (the same
    # restriction the whole-compose truncation AD tests document).
    upper, lower = 6.0, 3.0
    function nll(theta)
        d = LogNormal(theta[1], theta[2])
        return -logpdf(truncated(d; lower = lower, upper = upper), 4.0)
    end
    g = ForwardDiff.gradient(nll, [1.4, 0.5])
    @test all(isfinite, g)
    # The lower edge MOVES the gradient relative to the upper-only form.
    g_up = ForwardDiff.gradient(
        theta -> -logpdf(
            truncated(LogNormal(theta[1], theta[2]); upper = upper), 4.0),
        [1.4, 0.5])
    @test !isapprox(g, g_up)
end

@testitem "truncated convolved chain right-truncation matches Monte Carlo" begin
    using Distributions, Random, Statistics
    using CensoredDistributions: _collapse_to_observation
    rng = MersenneTwister(20240617)

    # The right-truncated convolved chain should match samples from the
    # convolution discarded above the window. Compare the truncated CDF at a
    # grid against the empirical CDF of accepted Monte Carlo draws.
    inc_dist = LogNormal(1.2, 0.4)
    delta_dist = Gamma(2.0, 0.8)
    window = 8.0
    chain = truncated(
        _collapse_to_observation((inc_dist, delta_dist), (false,));
        upper = window)

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
