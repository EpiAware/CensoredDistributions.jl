# Tests for the recursive event-tree / nesting layer (#298). References use a
# self-contained fine trapezoid integral (the integrate-the-latents
# equivalence) and a Monte-Carlo per-case likelihood estimate; no extra deps.

@testitem "EventTree constructor and topology validation" begin
    using Distributions

    tree = event_tree(:onset,
        [:onset => :admit => Gamma(2.0, 1.0),
            :onset => :notif => LogNormal(1.0, 0.5),
            :admit => :death => Gamma(1.5, 1.0),
            :admit => :disch => Gamma(2.0, 1.5)])

    @test tree isa CensoredDistributions.EventTree
    @test length(tree) == 5
    @test Set(event_names(tree)) ==
          Set([:onset, :admit, :notif, :death, :disch])
    @test tree.root == :onset
    @test tree.primary_event === nothing

    # A two-parent (shared child) structure is rejected: not a conjunctive
    # tree node.
    @test_throws ArgumentError event_tree(:a,
        [:a => :c => Gamma(2.0, 1.0), :b => :c => Gamma(1.0, 1.0)])

    # A cycle is rejected.
    @test_throws ArgumentError event_tree(:a,
        [:a => :b => Gamma(2.0, 1.0), :b => :a => Gamma(1.0, 1.0)])

    # The root cannot be a child.
    @test_throws ArgumentError event_tree(:a,
        [:a => :b => Gamma(2.0, 1.0), :b => :a => Gamma(1.0, 1.0)])

    # Malformed edge.
    @test_throws ArgumentError event_tree(:a, [(:a, :b, Gamma(2.0, 1.0))])

    # At least one edge.
    @test_throws ArgumentError event_tree(:a, [])
end

@testitem "EventTree single-edge equals primary_censored" begin
    using Distributions

    delay = LogNormal(1.0, 0.5)
    pe = Uniform(0.0, 1.0)
    tree = event_tree(:a, [:a => :b => delay]; primary_event = pe)
    pc = primary_censored(delay, pe)

    # Latent root marginalised over the primary window, child observed: the
    # single-edge tree reduces to scalar primary_censored.
    for y in [1.0, 2.0, 3.5]
        @test logpdf(tree, (a = missing, b = y))≈logpdf(pc, y) atol=1e-6
    end
end

@testitem "EventTree all-observed factorises" begin
    using Distributions

    D_oa = Gamma(2.0, 1.0)
    D_on = LogNormal(1.0, 0.4)
    D_ad = Gamma(1.5, 1.0)
    D_ai = Gamma(2.0, 0.8)
    tree = event_tree(:onset,
        [:onset => :admit => D_oa, :onset => :notif => D_on,
            :admit => :death => D_ad, :admit => :disch => D_ai])

    obs = (onset = 0.0, admit = 2.0, death = 5.0, disch = 4.0, notif = 1.5)
    ref = logpdf(D_oa, 2.0) + logpdf(D_on, 1.5) +
          logpdf(D_ad, 3.0) + logpdf(D_ai, 2.0)
    @test logpdf(tree, obs) ≈ ref
    @test pdf(tree, obs) ≈ exp(ref)

    # Dict and positional-vector observations agree with the NamedTuple.
    dict_obs = Dict(:onset => 0.0, :admit => 2.0, :death => 5.0,
        :disch => 4.0, :notif => 1.5)
    @test logpdf(tree, dict_obs) ≈ ref
    vec_obs = [getproperty(obs, n) for n in event_names(tree)]
    @test logpdf(tree, vec_obs) ≈ ref
end

@testitem "EventTree bdbv shared-latent and MC equivalence" begin
    using Distributions, Random

    # Fine trapezoid reference (integrate-the-latents equivalence).
    function tref(f, lo, hi; n = 200_000)
        xs = range(lo, hi; length = n)
        h = (hi - lo) / (n - 1)
        s = (f(xs[1]) + f(xs[end])) / 2
        @inbounds for i in 2:(n - 1)
            s += f(xs[i])
        end
        return s * h
    end

    D_oa = Gamma(2.0, 1.0)
    D_on = LogNormal(1.0, 0.4)
    D_ad = Gamma(1.5, 1.0)
    D_ai = Gamma(2.0, 0.8)

    tree = event_tree(:onset,
        [:onset => :admit => D_oa, :onset => :notif => D_on,
            :admit => :death => D_ad, :admit => :disch => D_ai])

    notif_term = logpdf(D_on, 1.5)

    # admit MISSING (shared interior latent), onset observed. admit couples
    # death and disch; it must be integrated jointly, once.
    obs2 = (onset = 0.0, admit = missing, death = 5.0, disch = 4.0,
        notif = 1.5)
    integ2 = tref(
        a -> pdf(D_oa, a) * pdf(D_ad, 5.0 - a) * pdf(D_ai, 4.0 - a),
        0.0, 4.0)
    @test logpdf(tree, obs2)≈(notif_term + log(integ2)) atol=1e-6

    # admit latent AND death a latent leaf (drops): only disch constrains.
    obs3 = (onset = 0.0, admit = missing, death = missing, disch = 4.0,
        notif = 1.5)
    integ3 = tref(a -> pdf(D_oa, a) * pdf(D_ai, 4.0 - a), 0.0, 4.0)
    @test logpdf(tree, obs3)≈(notif_term + log(integ3)) atol=1e-6

    # Latent root: marginalise onset over the primary window.
    tree_pe = event_tree(:onset,
        [:onset => :admit => D_oa, :onset => :notif => D_on,
            :admit => :death => D_ad, :admit => :disch => D_ai];
        primary_event = Uniform(0.0, 1.0))
    obs4 = (onset = missing, admit = 2.0, death = 5.0, disch = 4.0,
        notif = 1.5)
    dd = logpdf(D_ad, 3.0) + logpdf(D_ai, 2.0)
    integ4 = tref(
        o -> pdf(Uniform(0, 1), o) * pdf(D_oa, 2.0 - o) *
             pdf(D_on, 1.5 - o), 0.0, 1.0)
    @test logpdf(tree_pe, obs4)≈(dd + log(integ4)) atol=1e-6

    # Hardest case: onset AND admit both latent (connected latent block,
    # nested joint integration; admit shared across death and disch).
    obs5 = (onset = missing, admit = missing, death = 5.0, disch = 4.0,
        notif = 1.5)
    integ5 = tref(
        o -> pdf(Uniform(0, 1), o) * pdf(D_on, 1.5 - o) *
             tref(
                 a -> pdf(D_oa, a - o) * pdf(D_ad, 5.0 - a) *
                      pdf(D_ai, 4.0 - a),
                 o, 4.0; n = 4000),
        0.0, 1.0; n = 4000)
    @test logpdf(tree_pe, obs5)≈log(integ5) rtol=1e-4

    # Monte-Carlo per-case likelihood for the onset+admit-latent target.
    Random.seed!(2024)
    target = (death = 5.0, disch = 4.0, notif = 1.5)
    lp = logpdf(tree_pe,
        (onset = missing, admit = missing,
            death = target.death, disch = target.disch,
            notif = target.notif))
    function mc_density(tr, tgt, hw, N)
        cnt = 0
        for _ in 1:N
            s = rand(tr)
            if abs(s.death - tgt.death) < hw &&
               abs(s.disch - tgt.disch) < hw &&
               abs(s.notif - tgt.notif) < hw
                cnt += 1
            end
        end
        return cnt / N / (2hw)^3
    end
    mc = mc_density(tree_pe, target, 0.25, 8_000_000)
    @test exp(lp)≈mc rtol=0.1
end

@testitem "EventTree rand returns full event-time path" begin
    using Distributions, Random, Statistics

    D_oa = Gamma(2.0, 1.0)
    D_on = LogNormal(1.0, 0.4)
    D_ad = Gamma(1.5, 1.0)
    tree = event_tree(:onset,
        [:onset => :admit => D_oa, :onset => :notif => D_on,
            :admit => :death => D_ad];
        primary_event = Uniform(0.0, 1.0))

    Random.seed!(7)
    s = rand(tree)
    @test s isa NamedTuple
    @test Set(keys(s)) == Set(event_names(tree))
    # Tree ordering: each child time exceeds its parent time.
    @test s.admit > s.onset
    @test s.notif > s.onset
    @test s.death > s.admit

    # Mean gaps recover the delay means.
    Random.seed!(11)
    n = 40_000
    gaps_oa = Float64[]
    gaps_ad = Float64[]
    for _ in 1:n
        r = rand(tree)
        push!(gaps_oa, r.admit - r.onset)
        push!(gaps_ad, r.death - r.admit)
    end
    @test mean(gaps_oa)≈mean(D_oa) rtol=0.05
    @test mean(gaps_ad)≈mean(D_ad) rtol=0.05
end

@testitem "EventTree horizon right-truncates edges" begin
    using Distributions

    delay = LogNormal(1.0, 0.5)
    pe = Uniform(0.0, 1.0)
    horizon = 5.0
    tree = event_tree(:a, [:a => :b => delay];
        primary_event = pe, horizon = horizon)

    # With the latent root and an observed child the single-edge tree equals
    # primary_censored of the horizon-truncated delay.
    pc = primary_censored(truncated(delay; upper = horizon), pe)
    for y in [1.0, 2.0, 4.0]
        @test logpdf(tree, (a = missing, b = y))≈logpdf(pc, y) atol=1e-6
    end

    @test truncated(tree, horizon) isa CensoredDistributions.EventTree
end

@testitem "EventTree interval censoring scores observed windows" begin
    using Distributions

    D = Gamma(2.0, 1.0)
    tree = event_tree(:a, [:a => :b => D]; interval = 1.0)

    # Observed parent and child with an interval width: the edge factor is the
    # interval-censored logpdf at the gap.
    gap = 3.0
    @test logpdf(tree, (a = 0.0, b = gap)) ≈
          logpdf(interval_censored(D, 1.0), gap)
end
