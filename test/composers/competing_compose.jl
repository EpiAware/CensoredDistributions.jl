# A `Competing` node flows through the `compose` front-end TODAY as a univariate
# leaf (#329/#333): being univariate it nests as an ordinary `Parallel` branch,
# off the shared latent origin of the other (censored) branches. This pins that
# the either/or branch composes and scores now, before any table-column
# competing convention is built (deferred, #333).

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

@testitem "compose: Competing branch scores off the shared origin" begin
    using Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    cmp = Competing(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.7))
    d_notif = primary_censored(LogNormal(0.5, 0.4), Uniform(0, 1))
    d = compose((resolution = cmp, notification = d_notif))

    # The notification branch carries the shared primary; the Competing branch
    # is a univariate leaf delay off that same shared origin. The event vector
    # is [origin, resolution, notification].
    ev = Vector{Union{Missing, Float64}}([0.2, 3.0, 2.5])
    direct = logpdf(d, ev)
    @test isfinite(direct)

    # Scores through the generic per-record entry: the submodel log-density
    # equals the direct censored-composer logpdf.
    @model demo(dd, r) = obs ~ to_submodel(composed_distribution_model(dd, r))
    row = (origin = 0.2, resolution = 3.0, notification = 2.5)
    @test only(logjoint(demo(d, row), (;))) ≈ direct

    # The Competing branch behaves as a univariate leaf: conditioning on a
    # present origin, its contribution is logpdf(competing, resolution - origin).
    o, y_res, y_notif = 0.2, 3.0, 2.5
    expected = logpdf(Uniform(0, 1), o) +
               logpdf(cmp, y_res - o) +
               logpdf(get_dist_recursive(d_notif), y_notif - o)
    @test direct ≈ expected
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
    ev = Vector{Union{Missing, Float64}}([0.2, 3.0, 2.5])
    @test logpdf(dm, ev) ≈ logpdf(compose((a = cmp, b = d_notif)), ev)
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
