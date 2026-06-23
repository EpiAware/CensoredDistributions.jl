# Node-level censoring/truncation wrappers over composers (tenet 7). A modifier
# applies at ANY level — to a leaf OR a whole composed node — and keeps the TREE
# SHAPE unchanged: it distributes the shared observation resolution DOWN into the
# node's leaf cores, so a wrapped `Sequential`/`Parallel`/`Choose` is still
# record-scoreable. Each leaf is reduced to its continuous core before re-wrapping
# so an already-censored node is re-resolved, not double-wrapped.
#
# The SCALAR combine-then-censor lowering (observe the chain TOTAL with every
# intermediate marginalised) stays available EXPLICITLY through
# `modifier(observed_distribution(seq))` / `modifier(convolve_distributions(...))`;
# `observed_distribution` is the dual, scalar direction, distinct from the
# node-level wrap. Convolved/Resolve are already univariate and flow through the
# existing `UnivariateDistribution` wrapper methods unchanged.

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

@testitem "Node-level wrap of a Sequential distributes into leaf cores" begin
    using Distributions

    seq = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))

    # A modifier over a Sequential keeps the tree shape: it rebuilds the SAME
    # chain (same step names) with every leaf core wrapped, NOT a scalar collapse.
    pc_seq = primary_censored(seq, Uniform(0, 1))
    @test pc_seq isa CensoredDistributions.Sequential
    @test pc_seq.names === seq.names
    @test all(c -> c isa CensoredDistributions.PrimaryCensored, pc_seq.components)

    ic_seq = interval_censored(seq, 1.0)
    @test ic_seq isa CensoredDistributions.Sequential
    @test all(c -> c isa CensoredDistributions.IntervalCensored, ic_seq.components)

    dic_seq = double_interval_censored(
        seq; primary_event = Uniform(0, 1), upper = 10.0, interval = 1.0)
    @test dic_seq isa CensoredDistributions.Sequential

    # The node-level wrap is density-identical to the canonical per-leaf
    # construction (wrap each leaf, then chain).
    perleaf = Sequential(
        double_interval_censored(Gamma(2.0, 1.0);
            primary_event = Uniform(0, 1), upper = 10.0, interval = 1.0),
        double_interval_censored(LogNormal(0.5, 0.4);
            primary_event = Uniform(0, 1), upper = 10.0, interval = 1.0))
    rows = [(event_1 = 0.0, event_2 = 2.0, event_3 = 5.0)]
    @test logpdf(dic_seq, rows) ≈ logpdf(perleaf, rows)
end

@testitem "Scalar combine-then-censor stays explicit via observed_distribution" begin
    using Distributions

    seq = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    conv = convolve_distributions(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))

    # The SCALAR combine-then-censor total (intermediate marginalised) is now the
    # EXPLICIT `modifier(observed_distribution(seq))` form and reproduces the old
    # numbers: it equals the wrapper over the convolved total.
    pc = primary_censored(observed_distribution(seq), Uniform(0, 1))
    @test cdf(pc, 3.0) ≈ cdf(primary_censored(conv, Uniform(0, 1)), 3.0)

    ic = interval_censored(observed_distribution(seq), 1.0)
    @test pdf(ic, 3.0) ≈ pdf(interval_censored(conv, 1.0), 3.0)

    dic = double_interval_censored(observed_distribution(seq);
        primary_event = Uniform(0, 1), upper = 10.0, interval = 1.0)
    dic_conv = double_interval_censored(conv;
        primary_event = Uniform(0, 1), upper = 10.0, interval = 1.0)
    @test cdf(dic, 4.0) ≈ cdf(dic_conv, 4.0)

    # Keyword-only primary_censored over the observed total uses the default
    # primary.
    @test cdf(primary_censored(observed_distribution(seq)), 3.0) ≈
          cdf(primary_censored(conv), 3.0)
end

@testitem "interval_censored(observed_distribution) matches a Monte Carlo total" begin
    using Distributions, Random, Statistics

    Random.seed!(20240601)
    seq = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    # The scalar combine-then-censor total: the day-discretised PMF of the chain
    # total. With the node-level wrap distributing into leaves, the scalar total
    # is now the explicit `observed_distribution` form.
    ic = interval_censored(observed_distribution(seq), 1.0)

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
#
# A node-level wrap keeps the tree shape, so a wrapped Sequential/Parallel is
# multivariate: it scores an EVENT vector (one slot per event, the latent origin
# E_0 plus each step/branch), not a scalar.
# ---------------------------------------------------------------------------

@testitem "every composer x every wrapper constructs and scores finitely" begin
    using Distributions

    # A finite-logpdf check at a valid point for each composer realisation. A
    # wrapped node keeps the tree shape and is multivariate, so it scores a
    # fully-observed, ascending vector with the right slot layout: a CENSORED node
    # scores the flat event vector `[E_0, ..., E_k]` (a latent origin then one slot
    # per event, length `length(event_names(d))`), a PLAIN node the per-value
    # vector (length `length(d)`). Ascending values keep every gap positive. A
    # univariate (Convolved/Resolve) wrap scores a scalar and a Choose routes by
    # `kind`. (A fully-observed vector avoids the `missing` slots `rand(d)` can
    # emit for a non-firing branch, which `logpdf` is not meant to take here.)
    function score_finite(d)
        if d isa Union{CensoredDistributions.Sequential,
            CensoredDistributions.Parallel}
            if CensoredDistributions._tree_primary_event(d) === nothing
                # Plain node: the per-value vector (length `length(d)`).
                return isfinite(logpdf(d, collect(Float64, 1:length(d))))
            end
            # Censored node: the flat event vector `[E_0, ..., E_k]`, a
            # `Missing`-admitting vector so it routes to the event scorer.
            n = length(CensoredDistributions.event_names(d))
            ev = Vector{Union{Missing, Float64}}(collect(Float64, 1:n))
            return isfinite(logpdf(d, ev))
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

@testitem "single-edge Sequential wrap is exact (each leaf == bare leaf)" begin
    using Distributions

    leaf = Gamma(2.0, 1.0)
    seq = Sequential(leaf)

    # A single-edge chain wraps to a one-step Sequential whose lone leaf is the
    # wrapped bare leaf (distribute-into-leaf is exact at the leaf level). The
    # scalar combine-then-censor stays the explicit observed_distribution form.
    @test observed_distribution(seq) === leaf

    for (w,
        lw) in (
        (d -> truncate_to_horizon(d, 8.0), ld -> truncate_to_horizon(ld, 8.0)),
        (d -> primary_censored(d, Uniform(0, 1)),
            ld -> primary_censored(ld, Uniform(0, 1))),
        (d -> interval_censored(d, 1.0), ld -> interval_censored(ld, 1.0)),
        (
            d -> double_interval_censored(d; primary_event = Uniform(0, 1),
                upper = 10.0, interval = 1.0),
            ld -> double_interval_censored(ld; primary_event = Uniform(0, 1),
                upper = 10.0, interval = 1.0)))
        wseq = w(seq)
        @test wseq isa CensoredDistributions.Sequential
        @test length(wseq.components) == 1
        for x in (0.5, 1.0, 2.5, 5.0)
            @test logpdf(only(wseq.components), x) ≈ logpdf(lw(leaf), x)
        end
    end
end

@testitem "truncate_to_horizon over Sequential distributes into leaves" begin
    using Distributions

    seq = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))

    t_seq = truncate_to_horizon(seq, 8.0)
    @test t_seq isa CensoredDistributions.Sequential
    @test all(c -> c isa Truncated, t_seq.components)
    # Each leaf is right-truncated at the shared window.
    for (c, leaf) in zip(t_seq.components, (Gamma(2.0, 1.0), LogNormal(0.5, 0.4)))
        @test logpdf(c, 3.0) ≈ logpdf(truncate_to_horizon(leaf, 8.0), 3.0)
    end

    # The scalar combine-then-truncate total stays the explicit form.
    conv = convolve_distributions(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    t_total = truncate_to_horizon(observed_distribution(seq), 8.0)
    @test t_total isa Truncated
    @test logpdf(t_total, 3.0) ≈ logpdf(truncate_to_horizon(conv, 8.0), 3.0)
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

# ---------------------------------------------------------------------------
# Node-level censoring is record-scoreable (#655). Wrapping a WHOLE composed
# node in a censoring modifier keeps the tree shape, so the wrapped node scores
# a table of records exactly as the canonical per-leaf construction does.
# ---------------------------------------------------------------------------

@testitem "node-level dic over a Sequential scores records (== per-leaf)" begin
    using Distributions

    seq = Sequential(LogNormal(1.2, 0.5), Gamma(2.0, 1.0))
    wrapped = double_interval_censored(
        seq; primary_event = Uniform(0, 1), interval = 1.0)
    @test wrapped isa CensoredDistributions.Sequential

    # Reference: censor each leaf with the SAME resolution, then chain.
    perleaf = Sequential(
        double_interval_censored(LogNormal(1.2, 0.5);
            primary_event = Uniform(0, 1), interval = 1.0),
        double_interval_censored(Gamma(2.0, 1.0);
            primary_event = Uniform(0, 1), interval = 1.0))

    names = CensoredDistributions.event_names(wrapped)
    rows = [NamedTuple{names}((0.0, 2.0, 5.0)),
        NamedTuple{names}((1.0, 3.0, 7.0)),
        # Fully-missing intermediate marginalises identically in both forms.
        NamedTuple{names}((0.0, missing, 6.0))]

    scored = logpdf(wrapped, rows)
    @test isfinite(scored)
    @test scored ≈ logpdf(perleaf, rows)

    # Per-record assembly agrees with the front-door sum.
    recs = CensoredDistributions.record_distributions(wrapped, rows)
    @test sum(logpdf(recs[i], collect(values(rows[i]))) for i in 1:2) ≈
          logpdf(wrapped, rows[1:2])
end

@testitem "node-level dic over a Parallel scores records (== per-leaf)" begin
    using Distributions

    par = Parallel((LogNormal(1.2, 0.5), Gamma(2.0, 1.0)), (:a, :b))
    wrapped = double_interval_censored(
        par; primary_event = Uniform(0, 1), interval = 1.0)
    @test wrapped isa CensoredDistributions.Parallel

    # Reference: censor each branch core with the SAME shared origin, then fan out.
    perleaf = Parallel(
        (
            double_interval_censored(LogNormal(1.2, 0.5);
                primary_event = Uniform(0, 1), interval = 1.0),
            double_interval_censored(Gamma(2.0, 1.0);
                primary_event = Uniform(0, 1), interval = 1.0)),
        (:a, :b))

    names = CensoredDistributions.event_names(wrapped)
    rows = [NamedTuple{names}((0.0, 2.0, 3.0)),
        NamedTuple{names}((1.0, 4.0, 5.0)),
        # A missing branch drops from the joint in both forms.
        NamedTuple{names}((0.0, missing, 3.0))]

    scored = logpdf(wrapped, rows)
    @test isfinite(scored)
    @test scored ≈ logpdf(perleaf, rows)
end

@testitem "node-level wrap does not double-censor an already-censored node" begin
    using Distributions

    # Wrapping a node whose leaves are ALREADY primary-censored re-resolves them
    # at the new resolution (the leaf is reduced to its continuous core first),
    # so it is density-identical to wrapping the bare-core node — no stacked
    # PrimaryCensored{PrimaryCensored{...}}.
    bare = Parallel((LogNormal(1.2, 0.5), Gamma(2.0, 1.0)), (:a, :b))
    censored = Parallel(
        (primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
            primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))), (:a, :b))

    w_bare = double_interval_censored(
        bare; primary_event = Uniform(0, 1), interval = 1.0)
    w_censored = double_interval_censored(
        censored; primary_event = Uniform(0, 1), interval = 1.0)

    @test typeof(w_bare) == typeof(w_censored)
    names = CensoredDistributions.event_names(w_bare)
    rows = [NamedTuple{names}((0.0, 2.0, 3.0))]
    @test logpdf(w_bare, rows) ≈ logpdf(w_censored, rows)
end
