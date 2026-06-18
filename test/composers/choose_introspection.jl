# Choose wired into the params/prior introspection interface (#377). The
# name-keyed nested params path must treat a `Choose` node like the other
# composers: its alternatives namespaced per alternative name, a tag shared
# across alternatives inventoried once, and the whole `params_table` /
# `build_priors` / `update` / `event_*` surface working on a `Choose`. Kept in
# its own file (not appended to `choose.jl`) to avoid #460 merge conflicts.

@testitem "Choose params is nested and name-keyed (standalone) (#377)" begin
    using CensoredDistributions, Distributions
    const CD = CensoredDistributions

    sel = choose(:index => Gamma(2.0, 1.0), :sourced => LogNormal(0.5, 0.4))

    # The PUBLIC `params(::Choose)` stays POSITIONAL, mirroring
    # `params(::Resolve)`: a tuple of each alternative's params in order.
    @test params(sel) == ((2.0, 1.0), (0.5, 0.4))

    # The NAME-keyed nested params (what prior introspection threads) is keyed by
    # the alternative names, each value that alternative's own params.
    ps = CD._choose_params(sel)
    @test keys(ps) == (:index, :sourced)
    @test ps.index == (2.0, 1.0)
    @test ps.sourced == (0.5, 0.4)
    # `_child_params` of a `Choose` is the name-keyed form.
    @test CD._child_params(sel) == ps
end

@testitem "Choose nested under a composer yields a name-keyed subtree (#377)" begin
    using CensoredDistributions, Distributions
    const CD = CensoredDistributions

    # A `Choose` as a child of a `Sequential` (the tree the event/introspection
    # layer walks). Before #377 `_child_params` fell through to the positional
    # `params(::Choose)`, injecting a bare tuple into an otherwise name-keyed
    # tree; now it recurses to a name-keyed subtree.
    sel = choose(:index => Gamma(2.0, 1.0), :sourced => LogNormal(0.5, 0.4))
    seq = CD.Sequential((Gamma(3.0, 1.0), sel), (:lead, :branch))
    p = params(seq)
    @test keys(p) == (:lead, :branch)
    @test p.lead == (3.0, 1.0)
    # The Choose child is a NamedTuple keyed by its alternative names, not a
    # positional tuple.
    @test p.branch isa NamedTuple
    @test keys(p.branch) == (:index, :sourced)
    @test p.branch.index == (2.0, 1.0)
    @test p.branch.sourced == (0.5, 0.4)

    # A composer alternative recurses further (name-keyed all the way down).
    sel2 = choose(:index => Gamma(2.0, 1.0),
        :sourced => compose((onset_admit = Gamma(2.0, 1.0),
            admit_death = LogNormal(0.5, 0.4))))
    cp = CD._child_params(sel2)
    @test cp.index == (2.0, 1.0)
    @test cp.sourced == (onset_admit = (2.0, 1.0), admit_death = (0.5, 0.4))
end

@testitem "params_table row-groups a Choose per alternative (#377)" begin
    using CensoredDistributions, Distributions
    const Tables = CensoredDistributions.Tables

    # Independent per-branch params are namespaced per alternative
    # (`index.…` / `sourced.…`), one row-group per alternative.
    sel = choose(:index => Gamma(2.0, 1.0), :sourced => LogNormal(0.5, 0.4))
    tbl = params_table(sel)

    @test Tables.istable(tbl)
    @test Tables.columnnames(Tables.columns(tbl)) ==
          (:edge, :param, :value, :support)
    @test tbl.edge == [:index, :index, :sourced, :sourced]
    @test tbl.param == [:shape, :scale, :mu, :sigma]
    @test tbl.value == [2.0, 1.0, 0.5, 0.4]
    @test all(s -> s == (0.0, Inf), tbl.support)

    # A composer alternative dots its inner edges under the alternative name.
    sel2 = choose(:index => Gamma(2.0, 1.0),
        :sourced => compose((onset_admit = Gamma(2.0, 1.0),
            admit_death = LogNormal(0.5, 0.4))))
    tbl2 = params_table(sel2)
    @test Symbol("sourced.onset_admit") in tbl2.edge
    @test Symbol("sourced.admit_death") in tbl2.edge
    @test :index in tbl2.edge
end

@testitem "params_table inventories a shared tag once across Choose (#377)" begin
    using CensoredDistributions, Distributions

    # A parameter tied across alternatives via `shared(:inc, ...)` is inventoried
    # ONCE under its tag, so the tied value is listed once not per branch.
    inc = shared(:inc, Gamma(2.0, 1.0))
    delta = LogNormal(0.5, 0.4)
    tree = choose(:index => inc,
        :sourced => compose((delta = delta, inc = inc)))
    tbl = params_table(tree)

    # `inc` appears once (shape + scale, both under the `:inc` tag edge); the
    # second occurrence in the sourced branch is deduped.
    @test count(==(:inc), tbl.edge) == 2
    @test tbl.edge == [:inc, :inc,
        Symbol("sourced.delta"), Symbol("sourced.delta")]
    @test tbl.param == [:shape, :scale, :mu, :sigma]
    @test tbl.value == [2.0, 1.0, 0.5, 0.4]
end

@testitem "build_priors round-trips a Choose params table (#377)" begin
    using CensoredDistributions, Distributions

    sel = choose(:index => Gamma(2.0, 1.0), :sourced => LogNormal(0.5, 0.4))
    tbl = params_table(sel)

    # Support-derived defaults everywhere: a complete nested prior NamedTuple
    # keyed by the alternative names then the parameter names.
    nested = build_priors(tbl)
    @test Set(keys(nested)) == Set((:index, :sourced))
    @test Set(keys(nested.index)) == Set((:shape, :scale))
    @test Set(keys(nested.sourced)) == Set((:mu, :sigma))
    # Positive scale/shape -> positive-truncated; a LogNormal mu is unbounded.
    @test nested.index.shape isa Truncated
    @test minimum(nested.index.shape) == 0
    @test nested.sourced.mu isa Normal

    # A per-(edge, param) override takes precedence.
    over = build_priors(tbl;
        priors = Dict((:index, :shape) => Normal(2, 0.5)))
    @test over.index.shape == Normal(2, 0.5)

    # A shared tag round-trips as a single top-level prior group.
    inc = shared(:inc, Gamma(2.0, 1.0))
    shtree = choose(:index => inc,
        :sourced => compose((delta = LogNormal(0.5, 0.4), inc = inc)))
    shnested = build_priors(params_table(shtree))
    @test haskey(shnested, :inc)
    @test Set(keys(shnested.inc)) == Set((:shape, :scale))
end

@testitem "update reconstructs a Choose from a nested NamedTuple (#377)" begin
    using CensoredDistributions, Distributions
    const CD = CensoredDistributions

    sel = choose(:index => Gamma(2.0, 1.0), :sourced => LogNormal(0.5, 0.4))
    upd = update(sel, (index = (shape = 3.0, scale = 1.5),
        sourced = (mu = 0.7, sigma = 0.5)))

    @test upd isa CD.Choose
    @test upd.names == sel.names
    @test upd.selector == sel.selector
    @test CD._pick(upd, :index) == Gamma(3.0, 1.5)
    @test CD._pick(upd, :sourced) == LogNormal(0.7, 0.5)

    # Scoring matches a hand-built Choose with the new parameters.
    hand = choose(:index => Gamma(3.0, 1.5), :sourced => LogNormal(0.7, 0.5))
    @test logpdf(upd, 2.0; kind = :index) == logpdf(hand, 2.0; kind = :index)

    # A shared tag updates every occurrence from one top-level entry.
    inc = shared(:inc, Gamma(2.0, 1.0))
    shtree = choose(:index => inc,
        :sourced => compose((delta = LogNormal(0.5, 0.4), inc = inc)))
    shupd = update(shtree, (inc = (shape = 3.0, scale = 1.5),
        sourced = (delta = (mu = 0.7, sigma = 0.5),)))
    @test get_dist(CD._pick(shupd, :index)) == Gamma(3.0, 1.5)
    @test get_dist(event(CD._pick(shupd, :sourced), :inc)) == Gamma(3.0, 1.5)
end

@testitem "Choose event_tree / event_names name introspection (#377)" begin
    using CensoredDistributions, Distributions

    sel = choose(:index => Gamma(2.0, 1.0),
        :sourced => compose((onset_admit = Gamma(2.0, 1.0),
            admit_death = LogNormal(0.5, 0.4))))

    # The flat event names of a Choose are its alternative names (no single flat
    # layout; the active alternative is data-selected).
    @test event_names(sel) == (:index, :sourced)

    # The nested event tree keys the alternatives, recursing into a composer
    # alternative.
    et = event_tree(sel)
    @test keys(et) == (:index, :sourced)
    @test et.index == :index
    @test keys(et.sourced) == (:onset_admit, :admit_death)

    # `event` fetches an alternative by name and descends a nested path.
    @test event(sel, :index) == Gamma(2.0, 1.0)
    @test event(sel, :sourced, :admit_death) == LogNormal(0.5, 0.4)
    @test event(sel, Symbol("sourced.onset_admit")) == Gamma(2.0, 1.0)
    @test_throws KeyError event(sel, :missing_alt)
end
