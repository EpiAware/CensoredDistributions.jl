@testitem "Test CensoringIndicator constructor" begin
    using Distributions

    ic = interval_censored(Normal(0, 1), 1.0)
    d = indicate_censoring(ic)

    @test typeof(d) <: CensoredDistributions.CensoringIndicator
    @test d.dist === ic

    # Also works with Distributions.jl's own Censored/Truncated types (#893:
    # these should be first-class components everywhere a plain leaf is).
    censored_d = indicate_censoring(censored(Normal(0, 1); upper = 2.0))
    @test censored_d.dist isa Distributions.Censored

    truncated_d = indicate_censoring(truncated(Normal(0, 1); lower = -1.0))
    @test truncated_d.dist isa Distributions.Truncated
end

@testitem "Test CensoringIndicator distribution interface delegates to dist" begin
    using Distributions

    ic = interval_censored(LogNormal(1.5, 0.5), 1.0)
    d = indicate_censoring(ic)

    @test minimum(d) == minimum(ic)
    @test maximum(d) == maximum(ic)
    @test insupport(d, 2.0) == insupport(ic, 2.0)
    @test insupport(d, -1.0) == insupport(ic, -1.0)
    @test params(d) == params(ic)
    @test eltype(d) == eltype(ic)

    x = 2.0
    @test pdf(d, x) == pdf(ic, x)
    @test cdf(d, x) == cdf(ic, x)
    @test logcdf(d, x) == logcdf(ic, x)
    @test ccdf(d, x) == ccdf(ic, x)
    @test logccdf(d, x) == logccdf(ic, x)
    @test quantile(d, 0.4) == quantile(ic, 0.4)
end

@testitem "Test CensoringIndicator scalar scoring defaults to censored" begin
    using Distributions

    ic = interval_censored(Normal(0, 1), 1.0)
    d = indicate_censoring(ic)
    x = 0.5

    # With no indicator supplied, a bare scalar observation scores exactly
    # as `dist` (the censored leaf) would, for every scoring function.
    @test pdf(d, x) == pdf(ic, x)
    @test logpdf(d, x) == logpdf(ic, x)
    @test cdf(d, x) == cdf(ic, x)
    @test logcdf(d, x) == logcdf(ic, x)
    @test ccdf(d, x) == ccdf(ic, x)
    @test logccdf(d, x) == logccdf(ic, x)
end

@testitem "Test CensoringIndicator joint observation selects exact vs censored" begin
    using Distributions

    base = Normal(0, 1)
    ic = interval_censored(base, 1.0)
    d = indicate_censoring(ic)
    x = 0.5

    # censored = true: `dist` itself (the bounded/censored leaf).
    @test logpdf(d, (value = x, censored = true)) == logpdf(ic, x)

    # censored = false: the exact density of the UNDERLYING (unwrapped)
    # dist - the invariant this feature exists for (CensoredDistributions#894).
    @test logpdf(d, (value = x, censored = false)) == logpdf(base, x)

    # And that is NOT the same value as the censored contribution (interval
    # censoring genuinely changes the density here), so the two branches
    # are not accidentally aliasing.
    @test logpdf(d, (value = x, censored = true)) !=
          logpdf(d, (value = x, censored = false))
end

@testitem "Test CensoringIndicator joint observation dispatch for every scorer" begin
    using Distributions

    base = Normal(0, 1)
    ic = interval_censored(base, 1.0)
    d = indicate_censoring(ic)
    x = 0.5

    # pdf, cdf, logcdf, ccdf, and logccdf all recognise the same
    # `(value, censored)` joint observation as logpdf, and dispatch the
    # same way: `censored = true` scores against `dist`, `censored = false`
    # scores against the unwrapped `get_dist(dist)`.
    @test pdf(d, (value = x, censored = true)) == pdf(ic, x)
    @test pdf(d, (value = x, censored = false)) == pdf(base, x)

    @test cdf(d, (value = x, censored = true)) == cdf(ic, x)
    @test cdf(d, (value = x, censored = false)) == cdf(base, x)

    @test logcdf(d, (value = x, censored = true)) == logcdf(ic, x)
    @test logcdf(d, (value = x, censored = false)) == logcdf(base, x)

    @test ccdf(d, (value = x, censored = true)) == ccdf(ic, x)
    @test ccdf(d, (value = x, censored = false)) == ccdf(base, x)

    @test logccdf(d, (value = x, censored = true)) == logccdf(ic, x)
    @test logccdf(d, (value = x, censored = false)) == logccdf(base, x)
end

@testitem "Test CensoringIndicator with PrimaryCensored" begin
    using Distributions

    pc = primary_censored(LogNormal(1.5, 0.5), Uniform(0, 1))
    d = indicate_censoring(pc)
    x = 3.0

    @test logpdf(d, (value = x, censored = true)) == logpdf(pc, x)
    @test logpdf(d, (value = x, censored = false)) == logpdf(get_dist(pc), x)
end

@testitem "Test CensoringIndicator with Distributions.censored and truncated" begin
    using Distributions

    base = Gamma(2.0, 1.5)

    cens = censored(base; upper = 4.0)
    d_cens = indicate_censoring(cens)
    @test logpdf(d_cens, (value = 4.0, censored = true)) == logpdf(cens, 4.0)
    @test logpdf(d_cens, (value = 4.0, censored = false)) == logpdf(base, 4.0)

    trunc = truncated(base; lower = 0.5)
    d_trunc = indicate_censoring(trunc)
    @test logpdf(d_trunc, (value = 1.0, censored = true)) == logpdf(trunc, 1.0)
    @test logpdf(d_trunc, (value = 1.0, censored = false)) == logpdf(base, 1.0)
end

@testitem "Test CensoringIndicator loglikelihood over a mixed table" begin
    using Distributions

    base = Normal(0, 1)
    ic = interval_censored(base, 1.0)
    d = indicate_censoring(ic)

    values = [0.5, 1.5, -0.5]
    censored_flags = [true, false, true]
    obs = (values = values, censored = censored_flags)

    expected = sum(logpdf(d, (value = v, censored = c))
    for (v, c) in zip(values, censored_flags))
    @test loglikelihood(d, obs) ≈ expected

    # Cross-check against manual per-row selection, tying the table form
    # back to the same invariant as the scalar joint-observation test.
    manual = sum(
        c ? logpdf(ic, v) : logpdf(base, v) for (v, c) in zip(values, censored_flags))
    @test loglikelihood(d, obs) ≈ manual
end

@testitem "Test CensoringIndicator loglikelihood for a single joint observation" begin
    using Distributions

    ic = interval_censored(Normal(0, 1), 1.0)
    d = indicate_censoring(ic)
    obs = (value = 0.5, censored = false)

    @test loglikelihood(d, obs) == logpdf(d, obs)
end

@testitem "Test CensoringIndicator sampling delegates to dist" begin
    using Distributions, Random, Statistics

    ic = interval_censored(Normal(5.0, 2.0), 1.0)
    d = indicate_censoring(ic)

    samples_ic = rand(MersenneTwister(1), ic, 5000)
    samples_d = rand(MersenneTwister(1), d, 5000)

    @test samples_ic == samples_d
end

@testitem "Test CensoringIndicator type stability" begin
    using Distributions

    ic = interval_censored(Normal(0.0, 1.0), 1.0)
    d = indicate_censoring(ic)

    @test d isa
          CensoredDistributions.CensoringIndicator{<:CensoredDistributions.IntervalCensored}
    @test logpdf(d, 0.5) isa Float64
    @test logpdf(d, (value = 0.5, censored = true)) isa Float64
    @test logpdf(d, (value = 0.5, censored = false)) isa Float64
    @test pdf(d, (value = 0.5, censored = true)) isa Float64
    @test cdf(d, (value = 0.5, censored = true)) isa Float64
end
