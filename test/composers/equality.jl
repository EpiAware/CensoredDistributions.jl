# `==` / `hash` contract for the composer types (`src/composers/equality.jl`).
# The `Choose` case is covered in `choose.jl`; this pins the rest of the module:
# the STRUCTURAL equality of `Sequential` / `Parallel` (names ignored), the
# name-bearing equality of `Resolve` / `Compete`, the `NoEvent`
# singleton, and the cross-type `Resolve != Compete`. The hash/==
# invariant (equal values hash equal) is checked alongside each case.

@testitem "Sequential/Parallel compare structurally, ignoring names" begin
    using CensoredDistributions, Distributions

    comps = (Gamma(2.0, 1.0), LogNormal(0.5, 0.4))

    # Same components, DIFFERENT edge names -> structurally equal and equal
    # hash (names are relaxable display metadata, see equality.jl).
    s1 = Sequential(comps, (:onset_admit, :admit_death))
    s2 = Sequential(comps, (:a_b, :b_c))
    @test s1 == s2
    @test hash(s1) == hash(s2)

    p1 = Parallel(comps, (:onset_admit, :onset_notif))
    p2 = Parallel(comps, (:x_y, :x_z))
    @test p1 == p2
    @test hash(p1) == hash(p2)

    # Different components -> not equal.
    s3 = Sequential((Gamma(2.0, 1.0), LogNormal(0.6, 0.4)),
        (:onset_admit, :admit_death))
    @test s1 != s3

    # A Sequential and a Parallel of the same components are never equal
    # (the type tag is hashed in).
    @test s1 != Parallel(comps, (:onset_admit, :onset_notif))
    @test hash(s1) != hash(Parallel(comps, (:onset_admit, :onset_notif)))
end

@testitem "Resolve keeps names in ==/hash" begin
    using CensoredDistributions, Distributions

    c1 = Resolve(:death => (Gamma(1.5, 1.0), 0.3),
        :discharge => (Gamma(2.0, 1.5), 0.7))
    c2 = Resolve(:death => (Gamma(1.5, 1.0), 0.3),
        :discharge => (Gamma(2.0, 1.5), 0.7))
    @test c1 == c2
    @test hash(c1) == hash(c2)

    # Different outcome NAMES -> not equal (outcome identities are intrinsic).
    c3 = Resolve(:death => (Gamma(1.5, 1.0), 0.3),
        :recover => (Gamma(2.0, 1.5), 0.7))
    @test c1 != c3

    # Different branch probabilities -> not equal.
    c4 = Resolve(:death => (Gamma(1.5, 1.0), 0.4),
        :discharge => (Gamma(2.0, 1.5), 0.6))
    @test c1 != c4
end

@testitem "Compete equality and Resolve cross-type" begin
    using CensoredDistributions, Distributions

    h1 = Compete(:death => Gamma(1.5, 1.0), :recover => Gamma(2.0, 1.0))
    h2 = Compete(:death => Gamma(1.5, 1.0), :recover => Gamma(2.0, 1.0))
    @test h1 == h2
    @test hash(h1) == hash(h2)

    h3 = Compete(:death => Gamma(1.6, 1.0), :recover => Gamma(2.0, 1.0))
    @test h1 != h3

    # A mixture (Resolve) and a racing-hazard node are never equal, even
    # with matching names/delays.
    c = Resolve(:death => (Gamma(1.5, 1.0), 0.5),
        :recover => (Gamma(2.0, 1.0), 0.5))
    @test c != h1
    @test h1 != c
end

@testitem "NoEvent is a singleton under ==/hash" begin
    using CensoredDistributions
    CD = CensoredDistributions

    @test CD.NoEvent() == CD.NoEvent()
    @test hash(CD.NoEvent()) == hash(CD.NoEvent())
end
