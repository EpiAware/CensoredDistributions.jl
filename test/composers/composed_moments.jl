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

@testitem "mean(latent(seq)) is the full per-event vector" begin
    using CensoredDistributions, Distributions

    dic(d) = double_interval_censored(
        d; primary_event = Uniform(0, 1), interval = 1.0)

    # A censored chain's latent view exposes the full per-event vector matching
    # rand(latent(d)) / event_names(d): the origin event then one free-delay
    # moment per leaf edge (censoring seen through).
    seq = Sequential((dic(Gamma(1.2, 3.0)), dic(Gamma(2.0, 3.5))),
        (:onset_admit, :admit_death))
    lat = latent(seq)
    m = mean(lat)
    @test m isa Vector
    @test length(m) == length(rand(lat)) == length(event_names(seq))
    em = NamedTuple{event_names(seq)}(Tuple(m))
    @test em.onset ≈ mean(Uniform(0, 1))
    @test em.admit ≈ mean(Gamma(1.2, 3.0))
    @test em.death ≈ mean(Gamma(2.0, 3.5)) ≈ 7.0

    ev = NamedTuple{event_names(seq)}(Tuple(var(latent(seq))))
    @test ev.admit ≈ var(Gamma(1.2, 3.0))
    @test ev.death ≈ var(Gamma(2.0, 3.5))
    @test std(latent(seq)) ≈ sqrt.(var(latent(seq)))
end

@testitem "mean(par) is the per-endpoint vector, latent the full path" begin
    using CensoredDistributions, Distributions

    dic(d) = double_interval_censored(
        d; primary_event = Uniform(0, 1), interval = 1.0)

    # A Parallel is genuinely multivariate: mean(par) is the per-ENDPOINT vector
    # (one overall moment per branch endpoint, NOT the origin), seeing through
    # censoring.
    par = Parallel(dic(Gamma(2.0, 1.0)), dic(Gamma(1.5, 2.0)))
    m = mean(par)
    @test m isa Vector
    @test length(m) == length(par.components)
    @test m[1] ≈ mean(Gamma(2.0, 1.0))
    @test m[2] ≈ mean(Gamma(1.5, 2.0))

    v = var(par)
    @test v[1] ≈ var(Gamma(2.0, 1.0))
    @test v[2] ≈ var(Gamma(1.5, 2.0))
    @test std(par) ≈ sqrt.(var(par))

    # latent(par) is the FULL per-event vector incl. the shared origin.
    lat = latent(par)
    lm = mean(lat)
    @test length(lm) == length(rand(lat)) == length(event_names(par))
    @test lm[1] ≈ mean(Uniform(0, 1))   # origin slot
    @test lm[2] ≈ mean(Gamma(2.0, 1.0))
    @test lm[3] ≈ mean(Gamma(1.5, 2.0))
end

@testitem "latent per-event vector sees through censored leaves" begin
    using CensoredDistributions, Distributions

    dic(d) = double_interval_censored(
        d; primary_event = Uniform(0, 1), interval = 1.0)

    # A bdbv-shaped tree: onset -> {admit -> Competing(death, discharge), notif}.
    resolution = Competing(:death => (dic(Gamma(2.0, 3.5)), 0.4),
        :discharge => (dic(Gamma(1.0, 8.0)), 0.6))
    admit_path = Sequential((dic(Gamma(1.2, 3.0)), resolution),
        (:onset_admit, :admit_resolution))
    tree = compose((admit_path = admit_path,
        onset_notif = dic(Gamma(0.7, 20.0))))

    lat = latent(tree)
    m = mean(lat)
    @test length(m) == length(rand(lat)) == length(event_names(tree))
    em = NamedTuple{event_names(tree)}(Tuple(m))
    @test em.onset ≈ mean(Uniform(0, 1))
    @test em.admit ≈ mean(Gamma(1.2, 3.0))
    @test em.death ≈ mean(Gamma(2.0, 3.5)) ≈ 7.0
    @test em.discharge ≈ mean(Gamma(1.0, 8.0))
    @test em.notif ≈ mean(Gamma(0.7, 20.0))

    ev = NamedTuple{event_names(tree)}(Tuple(var(lat)))
    @test ev.death ≈ var(Gamma(2.0, 3.5))
    @test ev.discharge ≈ var(Gamma(1.0, 8.0))
end

@testitem "Competing mixture moment sees through censored leaves" begin
    using CensoredDistributions, Distributions

    dic(d) = double_interval_censored(
        d; primary_event = Uniform(0, 1), interval = 1.0)

    # A nested Competing inside a chain: the latent per-event vector exposes each
    # outcome slot, while the Competing node's own scalar mixture moment is built
    # from the FREE per-outcome moments.
    inner = Competing(:fast => (dic(Gamma(1.0, 1.0)), 0.5),
        :slow => (dic(Gamma(2.0, 2.0)), 0.5))
    chain = Sequential((dic(Gamma(1.2, 3.0)), inner),
        (:onset_admit, :admit_resolution))

    em = NamedTuple{event_names(chain)}(Tuple(mean(latent(chain))))
    @test em.fast ≈ mean(Gamma(1.0, 1.0))
    @test em.slow ≈ mean(Gamma(2.0, 2.0))

    # The Competing's own scalar mixture mean (free-leaf transparent), used when
    # a Competing is a value child rather than its own event slots.
    mix = 0.5 * mean(Gamma(1.0, 1.0)) + 0.5 * mean(Gamma(2.0, 2.0))
    @test CensoredDistributions._competing_mix_mean(inner) ≈ mix
end

@testitem "mean(latent(d)) handles a plain (uncensored) tree per-step layout" begin
    using CensoredDistributions, Distributions

    # A plain (uncensored) Sequential collapses to the overall scalar.
    seq = Sequential((Gamma(2.0, 1.0), LogNormal(0.5, 0.4)), (:a, :b))
    @test mean(seq) isa Real
    @test mean(seq) ≈ mean(Gamma(2.0, 1.0)) + mean(LogNormal(0.5, 0.4))

    # Its latent view is the per-step value vector (no censored origin event).
    m = mean(latent(seq))
    @test length(m) == length(rand(latent(seq)))
    @test m[1] ≈ mean(Gamma(2.0, 1.0))
    @test m[2] ≈ mean(LogNormal(0.5, 0.4))

    v = var(latent(seq))
    @test v[1] ≈ var(Gamma(2.0, 1.0))
    @test v[2] ≈ var(LogNormal(0.5, 0.4))
end

@testitem "latent walks Convolved and weighted leaves through free-leaf" begin
    using CensoredDistributions, Distributions

    dic(d) = double_interval_censored(
        d; primary_event = Uniform(0, 1), interval = 1.0)

    # A Convolved edge reuses its additive mean (sum of component means). This
    # tree is PLAIN (uncensored), so the latent view is the per-step value
    # vector (no censored origin event).
    conv = convolve_distributions(Gamma(2.0, 1.0), Gamma(3.0, 1.0))
    ct = compose((c = conv, n = Gamma(1.0, 1.0)))
    cm = mean(latent(ct))
    @test length(cm) == length(rand(latent(ct)))
    @test cm[1] ≈ mean(conv) ≈ mean(Gamma(2.0, 1.0)) + mean(Gamma(3.0, 1.0))
    @test cm[2] ≈ mean(Gamma(1.0, 1.0))

    # A weighted censored leaf reports its inner free-delay moment.
    wt = compose((onset_admit = weight(dic(Gamma(2.0, 3.5)), 2.0),
        onset_notif = Gamma(0.7, 20.0)))
    wm = NamedTuple{event_names(wt)}(Tuple(mean(latent(wt))))
    @test wm.admit ≈ 7.0
end

@testitem "endpoint collapses a chain to its terminal scalar" begin
    using CensoredDistributions, Distributions

    seq = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    # endpoint is an alias for observed_distribution.
    @test endpoint(seq) == observed_distribution(seq)
    @test mean(endpoint(seq)) ≈ mean(observed_distribution(seq))
    # mean(seq) IS the endpoint mean now (the overall scalar).
    @test mean(seq) ≈ mean(endpoint(seq))
end
