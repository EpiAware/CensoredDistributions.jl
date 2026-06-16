# A `Competing` node flows through the `compose` front-end as a `Parallel` branch
# (it is univariate, so it nests off the shared latent origin of the other
# censored branches). Since #333 a NESTED `Competing` exposes one EVENT slot per
# OUTCOME (its death/discharge columns), so the event vector carries those slots
# rather than a single opaque resolution event, and the nested scorer
# self-dispatches on which outcome is observed.

@testitem "compose: Competing nests as a Parallel branch (NamedTuple)" begin
    using Distributions

    cmp = Competing(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.7))
    d_notif = primary_censored(LogNormal(0.5, 0.4), Uniform(0, 1))

    d = compose((resolution = cmp, notification = d_notif))

    # Lowers to a Parallel whose first branch IS the Competing, equal to the
    # explicitly built Parallel (names label branches but do not change stack).
    @test d isa CensoredDistributions.Parallel
    @test d.components[1] === cmp
    @test d == Parallel(cmp, d_notif)
end

@testitem "compose: nested Competing branch self-dispatches (#333)" begin
    using Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    death_d, disch_d = Gamma(1.5, 1.0), Gamma(2.0, 1.5)
    cmp = Competing(:death => (death_d, 0.3), :disch => (disch_d, 0.7))
    d_notif = primary_censored(LogNormal(0.5, 0.4), Uniform(0, 1))
    d = compose((resolution = cmp, notification = d_notif))

    # The Competing branch contributes one EVENT slot per OUTCOME (#333), so the
    # event vector is [origin, death, disch, notification]. The observed outcome
    # is identified positionally; death observed here conditions on that branch.
    o, y_notif = 0.2, 2.5
    ev = Vector{Union{Missing, Float64}}([o, 3.0, missing, y_notif])
    direct = logpdf(d, ev)
    @test isfinite(direct)

    # Scores through the generic per-record entry by name: the submodel
    # log-density equals the direct censored-composer logpdf. The origin and
    # notification edges have no derivable split names, so they take the
    # positional event names (`event_1` / `event_2`); the Competing outcomes are
    # named (`death` / `disch`).
    @model demo(dd, r) = obs ~ to_submodel(composed_distribution_model(dd, r))
    row = (event_1 = o, death = 3.0, disch = missing, event_2 = y_notif)
    @test only(logjoint(demo(d, row), (;))) ≈ direct

    # Death observed at gap `3.0 - o` from the shared origin: condition on the
    # death branch (log p_death + its delay logpdf). The notif branch conditions
    # on its OWN declared censoring at its gap (the nested-tree convention,
    # #345), not the bare core.
    expected = log(0.3) + logpdf(death_d, 3.0 - o) +
               logpdf(d_notif, y_notif - o)
    @test direct ≈ expected

    # Discharge observed instead conditions on the discharge branch.
    evd = Vector{Union{Missing, Float64}}([o, missing, 2.0, y_notif])
    @test logpdf(d, evd) ≈
          log(0.7) + logpdf(disch_d, 2.0 - o) +
          logpdf(d_notif, y_notif - o)
end

@testitem "compose: matrix front-end carries a Competing entry" begin
    using Distributions

    cmp = Competing(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.7))
    d_notif = primary_censored(LogNormal(0.5, 0.4), Uniform(0, 1))

    # A 2x1 column matrix is two parallel branches; the Competing is the first.
    mat = reshape([cmp, d_notif], 2, 1)
    dm = compose(mat)
    @test dm isa CensoredDistributions.Parallel
    @test dm.components[1] === cmp
    @test dm == Parallel(cmp, d_notif)

    # Scores identically to the NamedTuple front-end for the same structure.
    # The Competing contributes its two outcome slots (#333), so the event
    # vector is [origin, death, disch, notification].
    ev = Vector{Union{Missing, Float64}}([0.2, 3.0, missing, 2.5])
    @test logpdf(dm, ev) ≈ logpdf(compose((a = cmp, b = d_notif)), ev)
end

@testitem "compose: table prob/compete columns build a Competing branch" begin
    using Distributions

    d_death = Gamma(1.5, 1.0)
    d_disc = Gamma(2.0, 1.5)
    d_notif = primary_censored(LogNormal(0.5, 0.4), Uniform(0, 1))
    cfr = 0.3

    # An edge-list table whose `compete` column folds the death/discharge rows
    # into ONE Competing node (their `prob` values are the branch
    # probabilities), while the notification row stays an ordinary leaf branch.
    table = (
        name = [:death, :discharge, :notification],
        dist = [d_death, d_disc, d_notif],
        compete = [1, 1, 0],
        prob = [cfr, 1 - cfr, missing],
        chain = [0, 0, 0]
    )

    d = compose(table)
    expected = Parallel(
        Competing(:death => (d_death, cfr), :discharge => (d_disc, 1 - cfr)),
        d_notif)
    @test d isa CensoredDistributions.Parallel
    @test d.components[1] isa CensoredDistributions.Competing
    @test d == expected
    @test d.components[1].names == (:death, :discharge)
    @test d.components[1].branch_probs == (cfr, 1 - cfr)
end

@testitem "compose: table Competing works without a chain column" begin
    using Distributions

    d_death = Gamma(1.5, 1.0)
    d_disc = Gamma(2.0, 1.5)
    d_notif = primary_censored(LogNormal(0.5, 0.4), Uniform(0, 1))
    cfr = 0.4

    # The `chain` column is optional; a `compete`/`prob` table alone still
    # builds the Competing branch (compete id 0 rows stay leaf branches).
    table = (
        name = [:death, :discharge, :notification],
        dist = [d_death, d_disc, d_notif],
        compete = [2, 2, 0],
        prob = [cfr, 1 - cfr, missing]
    )

    d = compose(table)
    @test d == Parallel(
        Competing(:death => (d_death, cfr), :discharge => (d_disc, 1 - cfr)),
        d_notif)
end

@testitem "compose: table Competing branch scores like the NamedTuple form" begin
    using Distributions

    d_death = Gamma(1.5, 1.0)
    d_disc = Gamma(2.0, 1.5)
    d_notif = primary_censored(LogNormal(0.5, 0.4), Uniform(0, 1))
    cmp = Competing(:death => (d_death, 0.3), :disch => (d_disc, 0.7))

    table = (
        name = [:death, :disch, :notification],
        dist = [d_death, d_disc, d_notif],
        compete = [1, 1, 0],
        prob = [0.3, 0.7, missing]
    )

    dt = compose(table)
    dn = compose((resolution = cmp, notification = d_notif))
    ev = Vector{Union{Missing, Float64}}([0.2, 3.0, missing, 2.5])
    @test logpdf(dt, ev) ≈ logpdf(dn, ev)
end

@testitem "compose: table Competing validates branch probs" begin
    using Distributions

    d_death = Gamma(1.5, 1.0)
    d_disc = Gamma(2.0, 1.5)

    # Branch probabilities in a compete group must sum to one.
    bad_sum = (
        name = [:death, :disch],
        dist = [d_death, d_disc],
        compete = [1, 1],
        prob = [0.3, 0.3]
    )
    @test_throws ArgumentError compose(bad_sum)

    # A `compete` row needs a non-missing `prob`.
    missing_prob = (
        name = [:death, :disch],
        dist = [d_death, d_disc],
        compete = [1, 1],
        prob = [0.3, missing]
    )
    @test_throws ArgumentError compose(missing_prob)

    # A `prob` column without a `compete` column is rejected (ambiguous).
    no_compete = (
        name = [:death, :disch],
        dist = [d_death, d_disc],
        prob = [0.3, 0.7]
    )
    @test_throws ArgumentError compose(no_compete)
end

@testitem "Competing inner constructor validates every path" begin
    using Distributions

    d1, d2 = Gamma(1.5, 1.0), Gamma(2.0, 1.5)

    # Direct struct construction (the form used by equality round-trips and the
    # `update` value/node edits) validates the structural invariants: bounds, the
    # outcome count, and equal-length tuples.
    @test_throws ArgumentError CensoredDistributions.Competing(
        (:a, :b), (d1, d2), (-0.1, 1.1))           # negative branch prob
    @test_throws ArgumentError CensoredDistributions.Competing(
        (:a,), (d1,), (1.0,))                       # fewer than two outcomes
    @test_throws ArgumentError CensoredDistributions.Competing(
        (:a, :b), (d1, d2), (0.5,))                 # length mismatch
    @test_throws ArgumentError CensoredDistributions.Competing(
        (:a, :b, :c), (d1, d2), (0.5, 0.5))         # names vs delays mismatch

    # The sum-to-one requirement is enforced at the USER-facing `Pair...`
    # constructor (and at `as_mixture`), not the inner struct constructor: the
    # DynamicPPL extension legitimately rebuilds a Competing from branch probs
    # sampled independently from priors that need not sum to one.
    @test_throws ArgumentError Competing(
        :a => (d1, 0.3), :b => (d2, 0.3))          # user path: sum 0.6 errors
    unnorm = CensoredDistributions.Competing((:a, :b), (d1, d2), (0.3, 0.3))
    @test unnorm isa CensoredDistributions.Competing  # struct path: allowed
    # `as_mixture` (and the moments routed through it) needs a normalised set,
    # so it rejects the unnormalised node with a clear error.
    @test_throws ArgumentError CensoredDistributions.as_mixture(unnorm)

    # A valid direct construction still builds and round-trips through ==.
    c = CensoredDistributions.Competing((:a, :b), (d1, d2), (0.4, 0.6))
    @test c == Competing(:a => (d1, 0.4), :b => (d2, 0.6))

    # AD `Dual` branch probabilities pass validation (no AD stripping).
    using ForwardDiff: Dual
    p = Dual(0.4, 1.0)
    cad = CensoredDistributions.Competing((:a, :b), (d1, d2), (p, 1 - p))
    @test cad isa CensoredDistributions.Competing
end

@testitem "Competing.logpdf differentiates branch probs (AD)" begin
    using Distributions
    using ForwardDiff: derivative

    d1, d2 = Gamma(1.5, 1.0), Gamma(2.0, 1.5)
    x = 1.7

    # logpdf w.r.t. the first branch probability must carry a non-zero,
    # correct gradient. The marginal is log(p f1(x) + (1-p) f2(x)); its
    # derivative w.r.t. p is (f1(x) - f2(x)) / (p f1(x) + (1-p) f2(x)).
    f(p) = logpdf(
        CensoredDistributions.Competing((:a, :b), (d1, d2), (p, 1 - p)), x)
    g = derivative(f, 0.3)
    f1, f2 = pdf(d1, x), pdf(d2, x)
    expected = (f1 - f2) / (0.3 * f1 + 0.7 * f2)
    @test g ≈ expected
    @test g != 0

    # pdf gradient is exp(logpdf) * logpdf-gradient.
    fp(p) = pdf(
        CensoredDistributions.Competing((:a, :b), (d1, d2), (p, 1 - p)), x)
    @test derivative(fp, 0.3) ≈ exp(f(0.3)) * expected
end

@testitem "competing sugar mirrors the Competing constructor" begin
    using Distributions

    cfr = 0.3
    a = competing(:death => (Gamma(1.5, 1.0), cfr),
        :disch => (Gamma(2.0, 1.5), 1 - cfr))
    b = Competing(:death => (Gamma(1.5, 1.0), cfr),
        :disch => (Gamma(2.0, 1.5), 1 - cfr))
    @test a isa CensoredDistributions.Competing
    @test a == b
    @test logpdf(a, 2.0) ≈ logpdf(b, 2.0)

    # The sugar nests through compose just like the struct constructor.
    d_notif = primary_censored(LogNormal(0.5, 0.4), Uniform(0, 1))
    @test compose((resolution = a, notification = d_notif)) ==
          Parallel(b, d_notif)
end
