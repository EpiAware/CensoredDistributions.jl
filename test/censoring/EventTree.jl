# Tests for the recursive event-tree / nesting layer (#298). The front-end is a
# Tables.jl edge list (columns parent, child, delay). References use a
# self-contained fine trapezoid integral (the integrate-the-latents
# equivalence) and a Monte-Carlo per-case likelihood estimate; no extra deps.

@testitem "EventTree edge-list constructor and topology validation" begin
    using Distributions

    # The bdbv edge list: onset -> {admit, notif}; admit -> {death, disch}.
    edges = (parent = [:onset, :onset, :admit, :admit],
        child = [:admit, :notif, :death, :disch],
        delay = [Gamma(2.0, 1.0), LogNormal(1.0, 0.5),
            Gamma(1.5, 1.0), Gamma(2.0, 1.5)])
    tree = primary_censored(edges, Uniform(0.0, 1.0))

    @test tree isa CensoredDistributions.EventTree
    @test length(tree) == 5
    @test Set(event_names(tree)) ==
          Set([:onset, :admit, :notif, :death, :disch])
    # The root is inferred (the single event that is never a child).
    @test tree.root == :onset
    @test tree.primary_event == Uniform(0.0, 1.0)

    # A vector of NamedTuple rows is also a Tables.jl source.
    row_edges = [(parent = :a, child = :b, delay = Gamma(2.0, 1.0))]
    @test primary_censored(row_edges, Uniform(0.0, 1.0)) isa
          CensoredDistributions.EventTree

    # A two-parent (shared child) structure is rejected: not a conjunctive node.
    @test_throws ArgumentError primary_censored(
        (parent = [:a, :b], child = [:c, :c],
            delay = [Gamma(2.0, 1.0), Gamma(1.0, 1.0)]),
        Uniform(0.0, 1.0))

    # A cycle is rejected (no event is a root).
    @test_throws ArgumentError primary_censored(
        (parent = [:a, :b], child = [:b, :a],
            delay = [Gamma(2.0, 1.0), Gamma(1.0, 1.0)]),
        Uniform(0.0, 1.0))

    # Two disconnected trees give two roots and are rejected.
    @test_throws ArgumentError primary_censored(
        (parent = [:a, :c], child = [:b, :d],
            delay = [Gamma(2.0, 1.0), Gamma(1.0, 1.0)]),
        Uniform(0.0, 1.0))

    # A missing required column is rejected.
    @test_throws ArgumentError primary_censored(
        (parent = [:a], child = [:b]), Uniform(0.0, 1.0))

    # A non-table source is rejected.
    @test_throws ArgumentError primary_censored(
        [:a => :b => Gamma(2.0, 1.0)], Uniform(0.0, 1.0))

    # A non-distribution delay entry is rejected.
    @test_throws ArgumentError primary_censored(
        (parent = [:a], child = [:b], delay = [1.0]), Uniform(0.0, 1.0))
end

@testitem "EventTree single-edge equals primary_censored" begin
    using Distributions

    delay = LogNormal(1.0, 0.5)
    pe = Uniform(0.0, 1.0)
    tree = primary_censored(
        (parent = [:a], child = [:b], delay = [delay]), pe)
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
    pe = Uniform(0.0, 1.0)
    edges = (parent = [:onset, :onset, :admit, :admit],
        child = [:admit, :notif, :death, :disch],
        delay = [D_oa, D_on, D_ad, D_ai])
    tree = primary_censored(edges, pe)

    # Observed root: the primary_event is the root prior at the onset time.
    obs = (onset = 0.0, admit = 2.0, death = 5.0, disch = 4.0, notif = 1.5)
    ref = logpdf(pe, 0.0) +
          logpdf(D_oa, 2.0) + logpdf(D_on, 1.5) +
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
    pe = Uniform(0.0, 1.0)
    edges = (parent = [:onset, :onset, :admit, :admit],
        child = [:admit, :notif, :death, :disch],
        delay = [D_oa, D_on, D_ad, D_ai])
    tree = primary_censored(edges, pe)

    # onset observed at 0.0, so the prior term is logpdf(pe, 0.0).
    root_term = logpdf(pe, 0.0)
    notif_term = logpdf(D_on, 1.5)

    # admit MISSING (shared interior latent), onset observed. admit couples
    # death and disch; it must be integrated jointly, once.
    obs2 = (onset = 0.0, admit = missing, death = 5.0, disch = 4.0,
        notif = 1.5)
    integ2 = tref(
        a -> pdf(D_oa, a) * pdf(D_ad, 5.0 - a) * pdf(D_ai, 4.0 - a),
        0.0, 4.0)
    @test logpdf(tree, obs2)≈(root_term + notif_term + log(integ2)) atol=1e-6

    # admit latent AND death a latent leaf (drops): only disch constrains.
    obs3 = (onset = 0.0, admit = missing, death = missing, disch = 4.0,
        notif = 1.5)
    integ3 = tref(a -> pdf(D_oa, a) * pdf(D_ai, 4.0 - a), 0.0, 4.0)
    @test logpdf(tree, obs3)≈(root_term + notif_term + log(integ3)) atol=1e-6

    # Latent root: marginalise onset over the primary window.
    obs4 = (onset = missing, admit = 2.0, death = 5.0, disch = 4.0,
        notif = 1.5)
    dd = logpdf(D_ad, 3.0) + logpdf(D_ai, 2.0)
    integ4 = tref(
        o -> pdf(Uniform(0, 1), o) * pdf(D_oa, 2.0 - o) *
             pdf(D_on, 1.5 - o), 0.0, 1.0)
    @test logpdf(tree, obs4)≈(dd + log(integ4)) atol=1e-6

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
    @test logpdf(tree, obs5)≈log(integ5) rtol=1e-4

    # Monte-Carlo per-case likelihood for the onset+admit-latent target.
    Random.seed!(2024)
    target = (death = 5.0, disch = 4.0, notif = 1.5)
    lp = logpdf(tree,
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
    mc = mc_density(tree, target, 0.25, 8_000_000)
    @test exp(lp)≈mc rtol=0.1
end

@testitem "EventTree rand returns full event-time path" begin
    using Distributions, Random, Statistics

    D_oa = Gamma(2.0, 1.0)
    D_on = LogNormal(1.0, 0.4)
    D_ad = Gamma(1.5, 1.0)
    edges = (parent = [:onset, :onset, :admit],
        child = [:admit, :notif, :death],
        delay = [D_oa, D_on, D_ad])
    tree = primary_censored(edges, Uniform(0.0, 1.0))

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
    tree = primary_censored(
        (parent = [:a], child = [:b], delay = [delay]), pe;
        horizon = horizon)

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
    pe = Uniform(0.0, 1.0)
    tree = primary_censored(
        (parent = [:a], child = [:b], delay = [D]), pe; interval = 1.0)

    # Observed parent and child with an interval width: the edge factor is the
    # interval-censored logpdf at the gap (plus the observed-root prior term).
    gap = 3.0
    @test logpdf(tree, (a = 0.0, b = gap)) ≈
          logpdf(pe, 0.0) + logpdf(interval_censored(D, 1.0), gap)
end

@testitem "EventTree show prints the tree structure" begin
    using Distributions

    edges = (parent = [:onset, :onset, :admit, :admit],
        child = [:admit, :notif, :death, :disch],
        delay = [Gamma(2.0, 1.0), LogNormal(1.0, 0.5),
            Gamma(1.5, 1.0), Gamma(2.0, 1.5)])
    tree = primary_censored(edges, Uniform(0.0, 1.0))

    str = sprint(show, MIME("text/plain"), tree)
    @test occursin("EventTree with 5 events", str)
    @test occursin("root: onset", str)
    @test occursin("onset -> admit", str)
    @test occursin("admit -> death", str)
    @test occursin("Gamma", str)
end
