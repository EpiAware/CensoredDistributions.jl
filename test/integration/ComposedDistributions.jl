@testitem "ComposedDistributions extension loads" begin
    using ComposedDistributions

    ext = Base.get_extension(
        CensoredDistributions, :CensoredDistributionsComposedDistributionsExt)
    @test ext !== nothing
end

@testitem "censored leaves peel to their inner free delay" begin
    using ComposedDistributions: free_leaf
    using Distributions

    inner = LogNormal(1.5, 0.5)

    # A primary-censored leaf peels to the delay being estimated: the primary
    # event and the solver method are fixed structure, not free parameters.
    @test free_leaf(primary_censored(inner)) === inner

    # An interval-censored leaf peels through to the same inner delay, and the
    # double-censored stack peels through every wrapper in one call.
    @test free_leaf(interval_censored(inner, 1)) === inner
    @test free_leaf(
        double_interval_censored(inner; interval = 1, upper = 10)) === inner
end

@testitem "rewrap_leaf rebuilds the censoring around a new inner delay" begin
    using ComposedDistributions: free_leaf, rewrap_leaf
    using Distributions

    inner = LogNormal(1.5, 0.5)
    new_inner = LogNormal(2.0, 0.75)

    pc = primary_censored(inner)
    rebuilt = rewrap_leaf(pc, new_inner)
    @test free_leaf(rebuilt) === new_inner
    # The fixed censoring structure survives the rebuild unchanged.
    @test rebuilt.primary_event == pc.primary_event
    @test rebuilt.method === pc.method

    ic = interval_censored(inner, 1)
    ic_rebuilt = rewrap_leaf(ic, new_inner)
    @test free_leaf(ic_rebuilt) === new_inner
    @test ic_rebuilt.boundaries == ic.boundaries

    # Round-tripping the same inner delay reproduces the original leaf.
    @test rewrap_leaf(pc, free_leaf(pc)) == pc
end

@testitem "params_table lists only a censored leaf's free delay params" begin
    using ComposedDistributions
    using Distributions

    inner = LogNormal(1.5, 0.5)
    tree = compose((
        onset = double_interval_censored(inner; interval = 1, upper = 10),
        report = Gamma(2.0, 1.0)))

    tbl = params_table(tree)
    rows = collect(zip(tbl.edge, tbl.param))

    # The censored edge contributes exactly the inner delay's free params --
    # not the primary event's, and not the censoring boundaries.
    @test filter(r -> first(r) == :onset, rows) ==
          [(:onset, :mu), (:onset, :sigma)]
    # The plain leaf is unaffected.
    @test filter(r -> first(r) == :report, rows) ==
          [(:report, :shape), (:report, :scale)]

    # The censored edge's support is the inner delay's, not the wrapper's.
    onset_support = first(
        [s for (e, s) in zip(tbl.edge, tbl.support) if e == :onset])
    @test onset_support == (minimum(inner), maximum(inner))
end

@testitem "update rebuilds a censored leaf and keeps its censoring" begin
    using ComposedDistributions
    using ComposedDistributions: free_leaf
    using Distributions

    leaf = double_interval_censored(
        LogNormal(1.5, 0.5); interval = 1, upper = 10)
    tree = compose((onset = leaf, report = Gamma(2.0, 1.0)))

    updated = update(tree,
        (onset = (mu = 2.0, sigma = 0.75), report = (shape = 3.0, scale = 1.5)))

    new_leaf = event(updated, :onset)
    # Same wrapper stack, new inner delay, censoring carried over untouched.
    @test nameof(typeof(new_leaf)) == nameof(typeof(leaf))
    @test params(free_leaf(new_leaf)) == (2.0, 0.75)
    @test new_leaf.boundaries == leaf.boundaries

    @test params(event(updated, :report)) == (3.0, 1.5)
end

@testitem "an uncertain inner delay survives the censoring wrapper" begin
    using ComposedDistributions
    using Distributions

    # A prior attached to the inner delay must stay discoverable once that delay
    # is wrapped in censoring, otherwise build_priors would silently drop it and
    # treat the parameter as fixed.
    leaf = double_interval_censored(
        uncertain(LogNormal(1.5, 0.5); mu = Normal(1.5, 0.5));
        interval = 1, upper = 10)
    tree = compose((onset = leaf, report = Gamma(2.0, 1.0)))

    tbl = params_table(tree)
    attached = [p
                for (e, pn, p) in zip(tbl.edge, tbl.param, tbl.prior)
                if e == :onset && pn == :mu]
    @test length(attached) == 1
    @test attached[1] == Normal(1.5, 0.5)

    # The attached prior reaches build_priors rather than being defaulted.
    priors = build_priors(tbl)
    @test priors.onset.mu == Normal(1.5, 0.5)
end
