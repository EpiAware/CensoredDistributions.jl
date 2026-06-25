@testitem "mean/var/std collapse a Sequential to its overall scalar" begin
    using CensoredDistributions, Distributions

    # mean(seq) is the OVERALL observed-delay moment (a scalar): the moment of
    # observed_distribution(seq), i.e. the convolved total. For independent
    # steps this is the sum of the step means / variances.
    seq = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    @test mean(seq) isa Real
    @test mean(seq) ≈ mean(Gamma(2.0, 1.0)) + mean(LogNormal(0.5, 0.4))
    @test mean(seq) ≈ mean(observed_distribution(seq))
    @test var(seq) isa Real
    @test var(seq) ≈ var(Gamma(2.0, 1.0)) + var(LogNormal(0.5, 0.4))
    @test std(seq) ≈ sqrt(var(seq))
end

@testitem "mean(latent(seq)) is the full per-event NamedTuple" begin
    using CensoredDistributions, Distributions

    dic(d) = double_interval_censored(
        d; primary_event = Uniform(0, 1), interval = 1.0)

    # A censored chain's latent view exposes the full per-event NamedTuple keyed
    # by event_names(d), in the same layout as rand(latent(d)): the origin event
    # then one free-delay moment per leaf edge (censoring seen through).
    seq = Sequential((dic(Gamma(1.2, 3.0)), dic(Gamma(2.0, 3.5))),
        (:onset_admit, :admit_death))
    lat = latent(seq)
    em = mean(lat)
    @test em isa NamedTuple
    @test keys(em) == keys(rand(lat)) == event_names(seq)
    @test em.onset ≈ mean(Uniform(0, 1))
    @test em.admit ≈ mean(Gamma(1.2, 3.0))
    @test em.death ≈ mean(Gamma(2.0, 3.5)) ≈ 7.0

    ev = var(latent(seq))
    @test ev isa NamedTuple
    @test ev.admit ≈ var(Gamma(1.2, 3.0))
    @test ev.death ≈ var(Gamma(2.0, 3.5))
    es = std(latent(seq))
    @test es.admit ≈ sqrt(ev.admit)
    @test es.death ≈ sqrt(ev.death)
end

@testitem "mean(par) is the per-endpoint NamedTuple, latent the full path" begin
    using CensoredDistributions, Distributions

    dic(d) = double_interval_censored(
        d; primary_event = Uniform(0, 1), interval = 1.0)

    # A Parallel is genuinely multivariate: mean(par) is the per-ENDPOINT
    # NamedTuple (one overall moment per branch endpoint, NOT the origin), seeing
    # through censoring, keyed by the branch endpoint names.
    par = Parallel(dic(Gamma(2.0, 1.0)), dic(Gamma(1.5, 2.0)))
    m = mean(par)
    @test m isa NamedTuple
    @test length(keys(m)) == length(par.components)
    @test m.branch_1 ≈ mean(Gamma(2.0, 1.0))
    @test m.branch_2 ≈ mean(Gamma(1.5, 2.0))

    v = var(par)
    @test v.branch_1 ≈ var(Gamma(2.0, 1.0))
    @test v.branch_2 ≈ var(Gamma(1.5, 2.0))
    s = std(par)
    @test s.branch_1 ≈ sqrt(v.branch_1)
    @test s.branch_2 ≈ sqrt(v.branch_2)

    # latent(par) is the FULL per-event NamedTuple incl. the shared origin.
    lat = latent(par)
    lm = mean(lat)
    @test keys(lm) == keys(rand(lat)) == event_names(par)
    origin, admit, death = event_names(par)
    @test lm[origin] ≈ mean(Uniform(0, 1))   # origin slot
    @test lm[admit] ≈ mean(Gamma(2.0, 1.0))
    @test lm[death] ≈ mean(Gamma(1.5, 2.0))
end

@testitem "latent per-event vector sees through censored leaves" begin
    using CensoredDistributions, Distributions

    dic(d) = double_interval_censored(
        d; primary_event = Uniform(0, 1), interval = 1.0)

    # A bdbv-shaped tree: onset -> {admit -> Resolve(death, discharge), notif}.
    resolution = Resolve(:death => (dic(Gamma(2.0, 3.5)), 0.4),
        :discharge => (dic(Gamma(1.0, 8.0)), 0.6))
    admit_path = Sequential((dic(Gamma(1.2, 3.0)), resolution),
        (:onset_admit, :admit_resolution))
    tree = compose((admit_path = admit_path,
        onset_notif = dic(Gamma(0.7, 20.0))))

    lat = latent(tree)
    em = mean(lat)
    @test keys(em) == keys(rand(lat)) == event_names(tree)
    @test em.onset ≈ mean(Uniform(0, 1))
    @test em.admit ≈ mean(Gamma(1.2, 3.0))
    @test em.death ≈ mean(Gamma(2.0, 3.5)) ≈ 7.0
    @test em.discharge ≈ mean(Gamma(1.0, 8.0))
    @test em.notif ≈ mean(Gamma(0.7, 20.0))

    ev = var(lat)
    @test ev.death ≈ var(Gamma(2.0, 3.5))
    @test ev.discharge ≈ var(Gamma(1.0, 8.0))
end

@testitem "Resolve mixture moment sees through censored leaves" begin
    using CensoredDistributions, Distributions

    dic(d) = double_interval_censored(
        d; primary_event = Uniform(0, 1), interval = 1.0)

    # A nested Resolve inside a chain: the latent per-event vector exposes each
    # outcome slot, while the Resolve node's own scalar mixture moment is built
    # from the FREE per-outcome moments.
    inner = Resolve(:fast => (dic(Gamma(1.0, 1.0)), 0.5),
        :slow => (dic(Gamma(2.0, 2.0)), 0.5))
    chain = Sequential((dic(Gamma(1.2, 3.0)), inner),
        (:onset_admit, :admit_resolution))

    em = mean(latent(chain))
    @test em.fast ≈ mean(Gamma(1.0, 1.0))
    @test em.slow ≈ mean(Gamma(2.0, 2.0))

    # The Resolve's own scalar mixture mean (free-leaf transparent), used when
    # a Resolve is a value child rather than its own event slots.
    mix = 0.5 * mean(Gamma(1.0, 1.0)) + 0.5 * mean(Gamma(2.0, 2.0))
    @test CensoredDistributions._one_of_mix_mean(inner) ≈ mix
end

@testitem "mean(latent(d)) handles a plain (uncensored) tree per-step layout" begin
    using CensoredDistributions, Distributions

    # A plain (uncensored) Sequential collapses to the overall scalar.
    seq = Sequential((Gamma(2.0, 1.0), LogNormal(0.5, 0.4)), (:a, :b))
    @test mean(seq) isa Real
    @test mean(seq) ≈ mean(Gamma(2.0, 1.0)) + mean(LogNormal(0.5, 0.4))

    # Its latent view is the per-step value NamedTuple keyed by the step names
    # (no censored origin event for a plain tree).
    m = mean(latent(seq))
    @test m isa NamedTuple
    @test keys(m) == keys(rand(latent(seq))) == (:a, :b)
    @test m.a ≈ mean(Gamma(2.0, 1.0))
    @test m.b ≈ mean(LogNormal(0.5, 0.4))

    v = var(latent(seq))
    @test v.a ≈ var(Gamma(2.0, 1.0))
    @test v.b ≈ var(LogNormal(0.5, 0.4))
end

@testitem "latent walks Convolved and weighted leaves through free-leaf" begin
    using CensoredDistributions, Distributions

    dic(d) = double_interval_censored(
        d; primary_event = Uniform(0, 1), interval = 1.0)

    # A Convolved edge reuses its additive mean (sum of component means). This
    # tree is PLAIN (uncensored), so the latent view is the per-step value
    # NamedTuple keyed by the step names (no censored origin event).
    conv = convolve_distributions(Gamma(2.0, 1.0), Gamma(3.0, 1.0))
    ct = compose((c = conv, n = Gamma(1.0, 1.0)))
    cm = mean(latent(ct))
    @test keys(cm) == keys(rand(latent(ct))) == (:c, :n)
    @test cm.c ≈ mean(conv) ≈ mean(Gamma(2.0, 1.0)) + mean(Gamma(3.0, 1.0))
    @test cm.n ≈ mean(Gamma(1.0, 1.0))

    # A weighted censored leaf reports its inner free-delay moment.
    wt = compose((onset_admit = weight(dic(Gamma(2.0, 3.5)), 2.0),
        onset_notif = Gamma(0.7, 20.0)))
    wm = mean(latent(wt))
    @test wm.admit ≈ 7.0
end

@testitem "observed_distribution collapses a chain to its terminal scalar" begin
    using CensoredDistributions, Distributions

    seq = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    obs = observed_distribution(seq)
    @test obs isa UnivariateDistribution
    # mean(seq) IS the terminal scalar mean (the overall scalar).
    @test mean(seq) ≈ mean(obs)
end
