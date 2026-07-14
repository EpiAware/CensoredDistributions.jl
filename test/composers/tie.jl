# `tie(d, paths...; name)` is the tree-level, path-based spelling of
# `shared(:tag, dist)`: it walks the composed tree to each named leaf and wraps
# it in `Shared(name, leaf)`, producing the EXACT artefact a hand-written
# `shared` build would. The tests below pin that equivalence (params_table,
# build_priors, update and logpdf all identical), the loud failure on a bad or
# non-leaf path, and that `name` is required. AD coverage lives in
# `test/ad/tie_ad.jl`.

# Helper: compare two params_table column tables structurally (the wrapper has
# no custom `==`, so compare the underlying NamedTuple of column vectors).
@testitem "tie == shared: params_table, build_priors, update, logpdf" begin
    using CensoredDistributions, Distributions
    using Tables: Tables

    cols(t) = Tables.columns(t)

    # A Choose with the same incubation `inc` in both branches: tie it tree-level
    # vs hand-wiring `shared(:inc, …)` at each leaf.
    incp = (2.0, 1.0)
    hand = choose(
        :index => compose((inc = shared(:inc, Gamma(incp...)),)),
        :sourced => compose((src = LogNormal(0.5, 0.4),
            inc = shared(:inc, Gamma(incp...)))))

    base = choose(
        :index => compose((inc = Gamma(incp...),)),
        :sourced => compose((src = LogNormal(0.5, 0.4),
            inc = Gamma(incp...))))
    tied = tie(base, (:index, :inc), (:sourced, :inc); name = :inc)

    # The tied tree and the hand-shared tree are the SAME composed object.
    @test tied == hand

    # params_table is row-for-row identical (deduped under the `:inc` tag once).
    @test cols(params_table(tied)) == cols(params_table(hand))
    # The `:inc` group appears once, not twice.
    @test count(==(:inc), params_table(tied).edge) == 2  # shape + scale, once

    # build_priors over the table yields the same nested prior structure.
    @test build_priors(params_table(tied)) == build_priors(params_table(hand))

    # update through the tied/hand trees gives the same reconstructed object.
    nt = (inc = (shape = 3.0, scale = 1.5),
        sourced = (src = (mu = 0.2, sigma = 0.3),))
    @test update(tied, nt) == update(hand, nt)

    # Score equality is the load-bearing check: same densities everywhere. The
    # `:sourced` branch scores `src` (LogNormal) then the shared `inc` (Gamma).
    ev = [1.0, 2.0]
    @test logpdf(tied, ev; kind = :sourced) ≈
          logpdf(hand, ev; kind = :sourced) atol=1e-12

    # A dotted-path Symbol resolves the SAME leaves as the tuple form.
    tied_dotted = tie(base, Symbol("index.inc"), Symbol("sourced.inc");
        name = :inc)
    @test tied_dotted == hand
    @test cols(params_table(tied_dotted)) == cols(params_table(hand))
end

@testitem "tie: update of the tied group sets every occurrence" begin
    using CensoredDistributions, Distributions

    base = choose(
        :index => compose((inc = Gamma(2.0, 1.0),)),
        :sourced => compose((src = LogNormal(0.5, 0.4),
            inc = Gamma(2.0, 1.0))))
    tied = tie(base, (:index, :inc), (:sourced, :inc); name = :inc)

    hand = choose(
        :index => compose((inc = shared(:inc, Gamma(2.0, 1.0)),)),
        :sourced => compose((src = LogNormal(0.5, 0.4),
            inc = shared(:inc, Gamma(2.0, 1.0)))))

    # Update the shared `:inc` group; both occurrences must take the new value,
    # identically to the hand-shared build.
    nt = (inc = (shape = 4.0, scale = 0.5),
        sourced = (src = (mu = 0.5, sigma = 0.4),))
    ut = update(tied, nt)
    uh = update(hand, nt)
    @test ut == uh
    ev = [1.5, 2.5]
    @test logpdf(ut, ev; kind = :sourced) ≈
          logpdf(uh, ev; kind = :sourced) atol=1e-12
end

@testitem "tie: bad path errors loudly (not a silent no-op)" begin
    using CensoredDistributions, Distributions

    base = choose(
        :index => compose((inc = Gamma(2.0, 1.0),)),
        :sourced => compose((src = LogNormal(0.5, 0.4),
            inc = Gamma(2.0, 1.0))))

    # A typo'd leaf name: no such child.
    @test_throws Exception tie(base, (:index, :typo), (:sourced, :inc);
        name = :inc)

    # A path that stops at a composer subtree, not a leaf.
    @test_throws ArgumentError tie(base, :index, (:sourced, :inc); name = :inc)

    # A path that runs past a leaf.
    @test_throws Exception tie(base, (:index, :inc, :nope); name = :inc)

    # No paths at all is an error.
    @test_throws ArgumentError tie(base; name = :inc)
end

@testitem "tie: parameter-incompatible leaves error" begin
    using CensoredDistributions, Distributions

    # `inc` is a Gamma, `src` a LogNormal: tying them into one group must error
    # since they cannot be one free parameter.
    base = choose(
        :index => compose((inc = Gamma(2.0, 1.0),)),
        :sourced => compose((src = LogNormal(0.5, 0.4),
            inc = Gamma(2.0, 1.0))))
    @test_throws ArgumentError tie(base, (:index, :inc), (:sourced, :src);
        name = :inc)
end

@testitem "tie: name keyword is required" begin
    using CensoredDistributions, Distributions

    base = choose(
        :index => compose((inc = Gamma(2.0, 1.0),)),
        :sourced => compose((src = LogNormal(0.5, 0.4),
            inc = Gamma(2.0, 1.0))))
    @test_throws UndefKeywordError tie(base, (:index, :inc), (:sourced, :inc))
end
