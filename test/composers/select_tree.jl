# A nested `Select` inside a composed TREE routes each record to ONE of its
# alternatives by the row's selector (`:kind`), scored as that alternative's edge.
# This is distinct from the flat value-vector nesting (which commits to the first
# alternative for the data-free round-trip): the EVENT/tree path must route by the
# selector so different rows score different alternatives.
#
# `_nested_trait` counts a Select as nesting (a tree with a Select recurses,
# rather than flat-segment scoring it), and `_tree_step(::Select)` picks the
# alternative. On the DATA path the per-record build resolves the selector into
# the tree, so a wrong silent commit to alternative 1 cannot happen; the data-free
# round-trip keeps the deterministic first-alternative default.

@testitem "nested Select makes the tree trait nested" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: _nested_trait, _Nested

    inner = selecting(:a => primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)),
        :b => primary_censored(Gamma(5.0, 1.0), Uniform(0, 1)))
    seq = Sequential(
        (primary_censored(Gamma(3.0, 1.0), Uniform(0, 1)), inner),
        (:onset_admit, :admit_death))
    @test _nested_trait(seq.components) isa _Nested

    par = Parallel(
        (primary_censored(Gamma(3.0, 1.0), Uniform(0, 1)), inner),
        (:onset_admit, :onset_death))
    @test _nested_trait(par.components) isa _Nested
end

@testitem "nested Select in a Sequential routes per row by selector" begin
    using CensoredDistributions, Distributions

    e1 = primary_censored(Gamma(3.0, 1.0), Uniform(0, 1))
    a = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))
    b = primary_censored(Gamma(5.0, 1.0), Uniform(0, 1))
    inner = selecting(:a => a, :b => b)
    seq = Sequential((e1, inner), (:onset_admit, :admit_death))

    # Two records routing to DIFFERENT alternatives. The chain is
    # onset -> admit (edge1) -> death (the selected Select edge).
    rows = [(onset = 0.0, admit = 2.0, death = 5.0, kind = :a),
        (onset = 0.0, admit = 1.0, death = 7.0, kind = :b)]
    recs = CensoredDistributions.record_distributions(seq, rows)

    # Hand-built reference: each record scores edge1 at (admit - onset) and the
    # SELECTED alternative at (death - admit).
    ref1 = logpdf(e1, 2.0 - 0.0) + logpdf(a, 5.0 - 2.0)
    ref2 = logpdf(e1, 1.0 - 0.0) + logpdf(b, 7.0 - 1.0)

    @test logpdf(recs[1], [0.0, 2.0, 5.0]) ≈ ref1
    @test logpdf(recs[2], [0.0, 1.0, 7.0]) ≈ ref2
    # Routing matters: scoring row 1 with alternative b would differ.
    @test !isapprox(logpdf(recs[1], [0.0, 2.0, 5.0]),
        logpdf(e1, 2.0 - 0.0) + logpdf(b, 5.0 - 2.0))
end

@testitem "nested Select in a Parallel routes per row by selector" begin
    using CensoredDistributions, Distributions

    shared = Uniform(0, 1)
    e1 = primary_censored(Gamma(3.0, 1.0), shared)
    a = primary_censored(Gamma(2.0, 1.0), shared)
    b = primary_censored(Gamma(5.0, 1.0), shared)
    inner = selecting(:a => a, :b => b)
    # Two branches off a shared origin: a fixed branch and a Select branch.
    par = Parallel((e1, inner), (:onset_admit, :onset_death))

    rows = [(onset = 0.0, admit = 2.0, death = 5.0, kind = :a),
        (onset = 0.0, admit = 1.0, death = 7.0, kind = :b)]
    recs = CensoredDistributions.record_distributions(par, rows)

    # Reference: the SAME Parallel with the routed alternative inlined (a plain
    # tree, no Select). Routing must reproduce the resolved tree's score exactly.
    par_a = Parallel((e1, a), (:onset_admit, :onset_death))
    par_b = Parallel((e1, b), (:onset_admit, :onset_death))
    @test logpdf(recs[1], [0.0, 2.0, 5.0]) ≈
          logpdf(par_a, Vector{Union{Missing, Float64}}([0.0, 2.0, 5.0]))
    @test logpdf(recs[2], [0.0, 1.0, 7.0]) ≈
          logpdf(par_b, Vector{Union{Missing, Float64}}([0.0, 1.0, 7.0]))
    # Routing matters: the two alternatives give different scores.
    @test !isapprox(
        logpdf(par_a, Vector{Union{Missing, Float64}}([0.0, 2.0, 5.0])),
        logpdf(par_b, Vector{Union{Missing, Float64}}([0.0, 2.0, 5.0])))
end

@testitem "data-free nested Select round-trip uses the default alternative" begin
    using CensoredDistributions, Distributions, Random
    using CensoredDistributions: _tree_step, _Flat

    e1 = primary_censored(Gamma(3.0, 1.0), Uniform(0, 1))
    a = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))
    b = primary_censored(Gamma(5.0, 1.0), Uniform(0, 1))
    inner = selecting(:a => a, :b => b)
    seq = Sequential((e1, inner), (:onset_admit, :admit_death))

    # The data-free EVENT-vector logpdf (no selector available) commits to the
    # FIRST alternative deterministically, so a constructed flat vector
    # round-trips through logpdf without a selector.
    ev = Vector{Union{Missing, Float64}}([0.0, 2.0, 5.0])
    expected = logpdf(e1, 2.0 - 0.0) + logpdf(a, 5.0 - 2.0)
    @test logpdf(seq, ev) ≈ expected

    # rand produces a full NAMED event record (a nested censored tree) that
    # round-trips back through logpdf via its flat event vector.
    Random.seed!(7)
    draw = rand(seq)
    @test length(draw) == 3
    @test isfinite(logpdf(seq,
        Vector{Union{Missing, Float64}}(collect(draw))))
end

@testitem "nested Select data path errors when the selector is missing" begin
    using CensoredDistributions, Distributions

    e1 = primary_censored(Gamma(3.0, 1.0), Uniform(0, 1))
    inner = selecting(:a => primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)),
        :b => primary_censored(Gamma(5.0, 1.0), Uniform(0, 1)))
    seq = Sequential((e1, inner), (:onset_admit, :admit_death))

    # A data row with no selector field cannot route the nested Select, so the
    # per-record build errors rather than silently scoring alternative 1.
    rows = [(onset = 0.0, admit = 2.0, death = 5.0)]
    @test_throws ArgumentError CensoredDistributions.record_distributions(
        seq, rows)
end
