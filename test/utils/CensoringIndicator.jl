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

@testitem "Test CensoringIndicator scalar logpdf defaults to the censored contribution" begin
    using Distributions

    ic = interval_censored(Normal(0, 1), 1.0)
    d = indicate_censoring(ic)
    x = 0.5

    # With no indicator supplied, a bare scalar observation scores exactly
    # as `dist` (the censored leaf) would.
    @test logpdf(d, x) == logpdf(ic, x)
end

@testitem "Test CensoringIndicator joint observation selects exact vs censored" begin
    using Distributions

    base = Normal(0, 1)
    ic = interval_censored(base, 1.0)
    d = indicate_censoring(ic)
    x = 0.5

    # exact = true: the exact density of the UNDERLYING (unwrapped) dist.
    @test logpdf(d, (value = x, exact = true)) == logpdf(base, x)

    # exact = false: the invariant this feature exists for — a bounded
    # record scored via the indicator equals the same record scored by
    # placing the censored leaf directly in the tree (CensoredDistributions#894).
    @test logpdf(d, (value = x, exact = false)) == logpdf(ic, x)

    # And that is NOT the same value as the exact contribution (interval
    # censoring genuinely changes the density here), so the two branches
    # are not accidentally aliasing.
    @test logpdf(d, (value = x, exact = true)) != logpdf(d, (value = x, exact = false))
end

@testitem "Test CensoringIndicator with PrimaryCensored" begin
    using Distributions

    pc = primary_censored(LogNormal(1.5, 0.5), Uniform(0, 1))
    d = indicate_censoring(pc)
    x = 3.0

    @test logpdf(d, (value = x, exact = true)) == logpdf(get_dist(pc), x)
    @test logpdf(d, (value = x, exact = false)) == logpdf(pc, x)
end

@testitem "Test CensoringIndicator with Distributions.censored and truncated" begin
    using Distributions

    base = Gamma(2.0, 1.5)

    cens = censored(base; upper = 4.0)
    d_cens = indicate_censoring(cens)
    @test logpdf(d_cens, (value = 4.0, exact = true)) == logpdf(base, 4.0)
    @test logpdf(d_cens, (value = 4.0, exact = false)) == logpdf(cens, 4.0)

    trunc = truncated(base; lower = 0.5)
    d_trunc = indicate_censoring(trunc)
    @test logpdf(d_trunc, (value = 1.0, exact = true)) == logpdf(base, 1.0)
    @test logpdf(d_trunc, (value = 1.0, exact = false)) == logpdf(trunc, 1.0)
end

@testitem "Test CensoringIndicator loglikelihood over a mixed table" begin
    using Distributions

    base = Normal(0, 1)
    ic = interval_censored(base, 1.0)
    d = indicate_censoring(ic)

    values = [0.5, 1.5, -0.5]
    exacts = [true, false, true]
    obs = (values = values, exacts = exacts)

    expected = sum(logpdf(d, (value = v, exact = e)) for (v, e) in zip(values, exacts))
    @test loglikelihood(d, obs) ≈ expected

    # Cross-check against manual per-row selection, tying the table form
    # back to the same invariant as the scalar joint-observation test.
    manual = sum(
        e ? logpdf(base, v) : logpdf(ic, v) for (v, e) in zip(values, exacts))
    @test loglikelihood(d, obs) ≈ manual
end

@testitem "Test CensoringIndicator loglikelihood for a single joint observation" begin
    using Distributions

    ic = interval_censored(Normal(0, 1), 1.0)
    d = indicate_censoring(ic)
    obs = (value = 0.5, exact = false)

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
    @test logpdf(d, (value = 0.5, exact = true)) isa Float64
    @test logpdf(d, (value = 0.5, exact = false)) isa Float64
end
