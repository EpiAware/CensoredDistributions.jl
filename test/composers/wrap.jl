# External censoring wrappers over composers. The semantics:
# combine first, then censor. A Sequential collapses to the convolution
# of its steps (its observed total) before censoring; a Parallel distributes
# the wrapper into every branch; Convolved/Resolve are already univariate and
# flow through the existing wrapper methods unchanged.

@testitem "observed_distribution: combine-then-censor scalar lowering" begin
    using Distributions

    # This is the COMBINE-THEN-CENSOR layer: you observe the chain TOTAL,
    # the single elapsed time origin -> terminal event, with every intermediate
    # event MARGINALISED. That total is the convolution of the chain steps, so a
    # `Sequential` lowers to a univariate `Convolved` and the `Convolved`
    # assertion below is correct for THIS layer.
    #
    # This is DISTINCT from the general per-record lowering
    # (`composed_distribution_model` / `logpdf(::Sequential, events)`),
    # where OBSERVED intermediates do NOT marginalise but FACTORISE into one
    # independent per-edge term each and only fully-missing intermediates
    # marginalise. The contrast test in
    # `test/composers/observed_intermediate_contrast.jl` pins that difference.

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

@testitem "Convolved/Resolve flow through wrappers (already univariate)" begin
    using Distributions

    conv = convolve_distributions(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    dic = double_interval_censored(
        conv; primary_event = Uniform(0, 1), interval = 1.0)
    @test dic isa CensoredDistributions.IntervalCensored
    @test cdf(dic, 4.0) isa Real

    comp = Resolve(:death => (Gamma(1.5, 1.0), 0.3),
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

    # The canonical example: combine first, then censor.
    d = double_interval_censored(
        convolve_distributions(Gamma(2.0, 1.0), LogNormal(0.5, 0.4));
        primary_event = Uniform(0, 1), upper = 10.0, interval = 1.0)
    @test d isa CensoredDistributions.IntervalCensored
    @test cdf(d, 5.0) isa Real
    @test 0.0 <= cdf(d, 5.0) <= 1.0
    @test pdf(d, 3.0) >= 0.0
end

# ---------------------------------------------------------------------------
# Coverage matrix: every composer x every wrapper constructs and gives a
# finite logpdf. This is the deliverable the maintainer asked for: truncation /
# censoring / interval-censoring compose over a composed distribution as you'd
# expect. truncate_to_horizon now joins the censoring wrappers over composers,
# and Choose distributes a wrapper into its alternatives.
# ---------------------------------------------------------------------------

@testitem "every composer x every wrapper constructs and scores finitely" begin
    using Distributions

    # A finite-logpdf check at a valid point for each composer realisation.
    function score_finite(d)
        if d isa CensoredDistributions.Parallel
            return isfinite(logpdf(d, fill(2.0, length(d))))
        elseif d isa CensoredDistributions.Choose
            return isfinite(logpdf(d, 2.0; kind = first(d.names)))
        else
            return isfinite(logpdf(d, 2.0))
        end
    end

    composers = (
        ("Sequential",
            Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))),
        ("Parallel",
            Parallel(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))),
        ("Choose",
            choose(:index => Gamma(2.0, 1.0),
                :sourced => LogNormal(0.5, 0.4))),
        ("Resolve",
            Resolve(:a => (Gamma(2.0, 1.0), 0.6),
                :b => (LogNormal(0.5, 0.4), 0.4))),
        ("Compete",
            Compete(:a => Gamma(2.0, 1.0),
                :b => LogNormal(0.5, 0.4))),
        ("Convolved",
            convolve_distributions(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))),
        ("Seq[leaf,Resolve]",
            Sequential(Gamma(2.0, 1.0),
                Resolve(:a => (Gamma(1.5, 1.0), 0.5),
                    :b => (LogNormal(0.3, 0.4), 0.5))))
    )

    wrappers = (
        ("truncate_to_horizon", d -> truncate_to_horizon(d, 10.0)),
        ("primary_censored", d -> primary_censored(d, Uniform(0, 1))),
        ("interval_censored", d -> interval_censored(d, 1.0)),
        ("double_interval_censored",
            d -> double_interval_censored(
                d; primary_event = Uniform(0, 1), upper = 10.0,
                interval = 1.0))
    )

    for (cname, d) in composers, (wname, w) in wrappers

        wrapped = w(d)
        @test score_finite(wrapped)
    end
end

@testitem "single-edge Sequential collapse is exact (== bare leaf)" begin
    using Distributions

    leaf = Gamma(2.0, 1.0)
    seq = Sequential(leaf)

    # The collapse is identity at the distribution level.
    @test observed_distribution(seq) === leaf

    # Every wrapper over the single-edge chain is density-identical to the
    # wrapper over the bare leaf (collapse-then-wrap must be exact).
    for x in (0.5, 1.0, 2.5, 5.0)
        @test logpdf(truncate_to_horizon(seq, 8.0), x) ≈
              logpdf(truncate_to_horizon(leaf, 8.0), x)
        @test logpdf(primary_censored(seq, Uniform(0, 1)), x) ≈
              logpdf(primary_censored(leaf, Uniform(0, 1)), x)
        @test logpdf(interval_censored(seq, 1.0), x) ≈
              logpdf(interval_censored(leaf, 1.0), x)
        @test logpdf(
            double_interval_censored(seq; primary_event = Uniform(0, 1),
                upper = 10.0, interval = 1.0), x) ≈
              logpdf(
            double_interval_censored(leaf; primary_event = Uniform(0, 1),
                upper = 10.0, interval = 1.0), x)
    end
end

@testitem "truncate_to_horizon over Sequential collapses then truncates" begin
    using Distributions

    seq = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    conv = convolve_distributions(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))

    t_seq = truncate_to_horizon(seq, 8.0)
    t_conv = truncate_to_horizon(conv, 8.0)
    @test t_seq isa Truncated
    @test logpdf(t_seq, 3.0) ≈ logpdf(t_conv, 3.0)
end

@testitem "truncate_to_horizon over Parallel distributes into branches" begin
    using Distributions

    par = Parallel(Gamma(2.0, 1.0), LogNormal(1.0, 0.5))
    t = truncate_to_horizon(par, 8.0)
    @test t isa CensoredDistributions.Parallel
    @test t.components[1] isa Truncated
    @test t.components[2] isa Truncated
    # The distributed truncation scores each branch independently.
    @test logpdf(t, [2.0, 3.0]) ≈
          logpdf(truncate_to_horizon(Gamma(2.0, 1.0), 8.0), 2.0) +
          logpdf(truncate_to_horizon(LogNormal(1.0, 0.5), 8.0), 3.0)
end

@testitem "Wrapping a Choose distributes into its alternatives" begin
    using Distributions

    sel = choose(:index => Gamma(2.0, 1.0),
        :sourced => LogNormal(0.5, 0.4))

    # Each wrapper distributes into every alternative, preserving the
    # names/selector, and scoring a record routes (via `kind`) to it.
    for (w,
        leaf_w) in (
        (d -> primary_censored(d, Uniform(0, 1)),
            ld -> primary_censored(ld, Uniform(0, 1))),
        (d -> interval_censored(d, 1.0), ld -> interval_censored(ld, 1.0)),
        (d -> truncate_to_horizon(d, 8.0), ld -> truncate_to_horizon(ld, 8.0)),
        (
            d -> double_interval_censored(d; primary_event = Uniform(0, 1),
                upper = 10.0, interval = 1.0),
            ld -> double_interval_censored(ld; primary_event = Uniform(0, 1),
                upper = 10.0, interval = 1.0)))
        wrapped = w(sel)
        @test wrapped isa CensoredDistributions.Choose
        @test wrapped.names === sel.names
        @test wrapped.selector === sel.selector
        @test logpdf(wrapped, 2.0; kind = :index) ≈
              logpdf(leaf_w(Gamma(2.0, 1.0)), 2.0)
        @test logpdf(wrapped, 2.0; kind = :sourced) ≈
              logpdf(leaf_w(LogNormal(0.5, 0.4)), 2.0)
    end
end
