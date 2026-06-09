@testitem "edge_means / edge_vars report free-leaf delay moments" begin
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

    em = edge_means(tree)
    # Each edge reports its inner FREE delay's mean (censoring is seen through),
    # so the Gamma(2, 3.5) death edge reports 7.0, not the censored mean.
    @test em.admit_path.onset_admit ≈ mean(Gamma(1.2, 3.0))
    @test em.admit_path.admit_resolution.death ≈ mean(Gamma(2.0, 3.5)) ≈ 7.0
    @test em.admit_path.admit_resolution.discharge ≈ mean(Gamma(1.0, 8.0))
    @test em.onset_notif ≈ mean(Gamma(0.7, 20.0))
    # The Competing node also reports the branch-prob-weighted mixture mean.
    @test em.admit_path.admit_resolution.mixture ≈
          0.4 * mean(Gamma(2.0, 3.5)) + 0.6 * mean(Gamma(1.0, 8.0))

    ev = edge_vars(tree)
    @test ev.admit_path.onset_admit ≈ var(Gamma(1.2, 3.0))
    @test ev.admit_path.admit_resolution.death ≈ var(Gamma(2.0, 3.5))
    # Mixture variance: Σ p_i (σ_i² + μ_i²) − (Σ p_i μ_i)².
    m1, m2 = mean(Gamma(2.0, 3.5)), mean(Gamma(1.0, 8.0))
    v1, v2 = var(Gamma(2.0, 3.5)), var(Gamma(1.0, 8.0))
    mm = 0.4 * m1 + 0.6 * m2
    @test ev.admit_path.admit_resolution.mixture ≈
          0.4 * (v1 + m1^2) + 0.6 * (v2 + m2^2) - mm^2
end

@testitem "edge_means walks Select, Latent, Convolved and bare leaves" begin
    using CensoredDistributions, Distributions

    dic(d) = double_interval_censored(
        d; primary_event = Uniform(0, 1), interval = 1.0)

    # An andv-shaped Select: an index branch versus a sourced two-step chain.
    sel = select(:index => dic(Gamma(2.0, 1.0)),
        :sourced => Sequential(
            (dic(Gamma(4.0, 1.5)), dic(Gamma(1.0, 2.0))), (:a, :b)))
    sm = edge_means(sel)
    @test sm.index ≈ mean(Gamma(2.0, 1.0))
    @test sm.sourced.a ≈ mean(Gamma(4.0, 1.5))
    @test sm.sourced.b ≈ mean(Gamma(1.0, 2.0))

    # A Latent of a composer reports the wrapped composer's edge means.
    tree = compose((onset_admit = dic(Gamma(2.0, 3.5)),
        onset_notif = Gamma(0.7, 20.0)))
    lt = CensoredDistributions.Latent(tree)
    @test edge_means(lt) == edge_means(tree)
    @test edge_vars(lt) == edge_vars(tree)

    # A Convolved edge reuses its additive mean (sum of component means).
    conv = convolve_distributions(Gamma(2.0, 1.0), Gamma(3.0, 1.0))
    ct = compose((c = conv, n = Gamma(1.0, 1.0)))
    @test edge_means(ct).c ≈ mean(conv) ≈
          mean(Gamma(2.0, 1.0)) + mean(Gamma(3.0, 1.0))

    # A bare (possibly censored or weighted) leaf reports its free-delay moment.
    @test edge_means(dic(Gamma(2.0, 3.5))) ≈ 7.0
    @test edge_vars(dic(Gamma(2.0, 3.5))) ≈ var(Gamma(2.0, 3.5))
    @test edge_means(weight(dic(Gamma(2.0, 3.5)), 2.0)) ≈ 7.0
end
