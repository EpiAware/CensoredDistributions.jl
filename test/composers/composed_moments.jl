@testitem "mean / var report per-event free-leaf delay moments" begin
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

    # mean/var are per-event Vectors matching rand(d) / event_names(d).
    m = mean(tree)
    @test m isa Vector
    @test length(m) == length(rand(tree)) == length(event_names(tree))
    em = NamedTuple{event_names(tree)}(Tuple(m))

    # The origin slot is the primary (origin) event's moment.
    @test em.onset ≈ mean(Uniform(0, 1))
    # Each later slot reports its inner FREE delay's mean (censoring seen
    # through), so the Gamma(2, 3.5) death edge reports 7.0, not the censored
    # mean.
    @test em.admit ≈ mean(Gamma(1.2, 3.0))
    @test em.death ≈ mean(Gamma(2.0, 3.5)) ≈ 7.0
    @test em.discharge ≈ mean(Gamma(1.0, 8.0))
    @test em.notif ≈ mean(Gamma(0.7, 20.0))

    ev = NamedTuple{event_names(tree)}(Tuple(var(tree)))
    @test ev.admit ≈ var(Gamma(1.2, 3.0))
    @test ev.death ≈ var(Gamma(2.0, 3.5))
    @test ev.discharge ≈ var(Gamma(1.0, 8.0))

    # std is the elementwise sqrt of var.
    @test std(tree) ≈ sqrt.(var(tree))
end

@testitem "Competing mixture moment sees through censored leaves" begin
    using CensoredDistributions, Distributions

    dic(d) = double_interval_censored(
        d; primary_event = Uniform(0, 1), interval = 1.0)

    # A nested Competing inside a chain: the per-event vector exposes each
    # outcome slot, while the Competing node's own scalar mixture moment is built
    # from the FREE per-outcome moments.
    inner = Competing(:fast => (dic(Gamma(1.0, 1.0)), 0.5),
        :slow => (dic(Gamma(2.0, 2.0)), 0.5))
    chain = Sequential((dic(Gamma(1.2, 3.0)), inner),
        (:onset_admit, :admit_resolution))

    em = NamedTuple{event_names(chain)}(Tuple(mean(chain)))
    @test em.fast ≈ mean(Gamma(1.0, 1.0))
    @test em.slow ≈ mean(Gamma(2.0, 2.0))

    # The Competing's own scalar mixture mean (free-leaf transparent), used when
    # a Competing is a value child rather than its own event slots.
    mix = 0.5 * mean(Gamma(1.0, 1.0)) + 0.5 * mean(Gamma(2.0, 2.0))
    @test CensoredDistributions._competing_mix_mean(inner) ≈ mix
end

@testitem "mean handles a plain (uncensored) tree per-step layout" begin
    using CensoredDistributions, Distributions

    tree = compose((a = Gamma(2.0, 1.0), b = LogNormal(0.5, 0.4)))
    m = mean(tree)
    # A plain tree's rand is the per-step value vector, so mean matches that
    # layout (length == length(d)).
    @test length(m) == length(tree) == length(rand(tree))
    @test m[1] ≈ mean(Gamma(2.0, 1.0))
    @test m[2] ≈ mean(LogNormal(0.5, 0.4))

    v = var(tree)
    @test v[1] ≈ var(Gamma(2.0, 1.0))
    @test v[2] ≈ var(LogNormal(0.5, 0.4))
end

@testitem "mean walks Convolved and weighted leaves through free-leaf" begin
    using CensoredDistributions, Distributions

    dic(d) = double_interval_censored(
        d; primary_event = Uniform(0, 1), interval = 1.0)

    # A Convolved edge reuses its additive mean (sum of component means). This
    # tree is PLAIN (uncensored), so mean is the per-step value vector.
    conv = convolve_distributions(Gamma(2.0, 1.0), Gamma(3.0, 1.0))
    ct = compose((c = conv, n = Gamma(1.0, 1.0)))
    cm = mean(ct)
    @test length(cm) == length(rand(ct))
    @test cm[1] ≈ mean(conv) ≈ mean(Gamma(2.0, 1.0)) + mean(Gamma(3.0, 1.0))
    @test cm[2] ≈ mean(Gamma(1.0, 1.0))

    # A weighted censored leaf reports its inner free-delay moment.
    wt = compose((onset_admit = weight(dic(Gamma(2.0, 3.5)), 2.0),
        onset_notif = Gamma(0.7, 20.0)))
    wm = NamedTuple{event_names(wt)}(Tuple(mean(wt)))
    @test wm.admit ≈ 7.0
end

@testitem "endpoint collapses a chain to its terminal scalar" begin
    using CensoredDistributions, Distributions

    seq = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    # endpoint is an alias for observed_distribution.
    @test endpoint(seq) == observed_distribution(seq)
    @test mean(endpoint(seq)) ≈ mean(observed_distribution(seq))
end
