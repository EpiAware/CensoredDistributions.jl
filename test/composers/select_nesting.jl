# A `Select` may be a CHILD of a composer (Sequential / Parallel / compose), not
# just a top node. The flat value-vector nesting machinery
# (`_child_nleaves` / `_child_logpdf` / `_child_rand!`) treats a nested Select as a
# fixed-width node that commits to one alternative: the data-free flat path uses
# the first alternative (deterministic, so `rand`/`logpdf` round-trip), while the
# row-driven record path selects by the row's `:kind`.

@testitem "Select reports a child leaf count" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: _child_nleaves

    # All alternatives share one leaf width, so a nested Select occupies that
    # many flat value slots.
    s = select_branch(:a => Gamma(2.0, 1.0), :b => LogNormal(0.5, 0.4))
    @test _child_nleaves(s) == 1

    # A composer alternative widens the slot to its own leaf count; every
    # alternative must agree.
    s2 = select_branch(
        :flat => Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4)),
        :also => Sequential(Gamma(1.0, 1.0), Gamma(3.0, 1.0)))
    @test _child_nleaves(s2) == 2

    # Disagreeing widths cannot occupy a fixed flat slot.
    bad = select_branch(:one => Gamma(2.0, 1.0),
        :two => Sequential(Gamma(1.0, 1.0), Gamma(2.0, 1.0)))
    @test_throws ArgumentError _child_nleaves(bad)
end

@testitem "Select nests in a Parallel: logpdf and rand" begin
    using CensoredDistributions, Distributions, Random

    inner = select_branch(:a => Gamma(2.0, 1.0), :b => LogNormal(0.5, 0.4))
    par = Parallel(Gamma(3.0, 1.0), inner)
    # Two leaves: the plain branch and the Select child (its first alternative).
    @test length(par) == 2

    x = [1.5, 2.3]
    # The flat value-vector logpdf scores the Select child as its first
    # alternative (the data-free default).
    manual = logpdf(Gamma(3.0, 1.0), x[1]) + logpdf(Gamma(2.0, 1.0), x[2])
    @test logpdf(par, x) ≈ manual

    # rand fills both slots; the Select slot is its first alternative's draw, so
    # logpdf at the draw is finite and consistent.
    Random.seed!(3)
    draw = rand(par)
    @test length(draw) == 2
    @test isfinite(logpdf(par, draw))
end

@testitem "Select nests in a Sequential: logpdf and rand" begin
    using CensoredDistributions, Distributions, Random

    inner = select_branch(:a => Gamma(2.0, 1.0), :b => LogNormal(0.5, 0.4))
    seq = Sequential(Gamma(3.0, 1.0), inner)
    @test length(seq) == 2

    x = [1.5, 2.3]
    manual = logpdf(Gamma(3.0, 1.0), x[1]) + logpdf(Gamma(2.0, 1.0), x[2])
    @test logpdf(seq, x) ≈ manual

    Random.seed!(4)
    draw = rand(seq)
    @test length(draw) == 2
    @test isfinite(logpdf(seq, draw))
end

@testitem "Select nests inside a compose front-end" begin
    using CensoredDistributions, Distributions, Random

    inner = select_branch(:a => Gamma(2.0, 1.0), :b => LogNormal(0.5, 0.4))
    # A Select as a compose component nests as a Parallel branch.
    outer = compose((pick = inner, side = Normal(0.0, 1.0)))
    @test outer isa CensoredDistributions.Parallel
    @test length(outer) == 2

    x = [0.7, 1.2]
    manual = logpdf(Gamma(2.0, 1.0), x[1]) + logpdf(Normal(0.0, 1.0), x[2])
    @test logpdf(outer, x) ≈ manual

    Random.seed!(5)
    @test length(rand(outer)) == 2
end
