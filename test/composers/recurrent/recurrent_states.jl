@testitem "recur builds a cyclic state graph and classifies states" begin
    using CensoredDistributions, Distributions
    const CD = CensoredDistributions

    m = recur(
        :susceptible => (:infected => Gamma(2.0, 4.0)),
        :infected => (:recovered => Gamma(2.0, 3.0), :dead => Gamma(2.0, 8.0)),
        :recovered => (:susceptible => Gamma(2.0, 30.0)))

    @test m isa RecurrentStates
    @test m.start == :susceptible
    @test CD.transient_states(m) == [:infected, :recovered, :susceptible]
    @test CD.absorbing_states(m) == [:dead]
    @test CD.is_absorbing(m, :dead)
    @test !CD.is_absorbing(m, :infected)
    # The infected state has two racing edges, so its node is a Compete.
    @test m.nodes[:infected] isa Compete
    # A single-edge state stays a lone `dest => dist` pair.
    @test m.nodes[:susceptible] isa Pair
end

@testitem "recur fixed-probability split builds a Resolve node" begin
    using CensoredDistributions, Distributions

    m = recur(
        :ill => (:recovered => (Gamma(2.0, 3.0), 0.7),
            :dead => (Gamma(2.0, 8.0), 0.3)),
        :recovered => (:ill => Gamma(2.0, 20.0)))
    @test m.nodes[:ill] isa Resolve
    @test collect(probs(m.nodes[:ill])) == [0.7, 0.3]
end

@testitem "recur accepts a NamedTuple transition spelling" begin
    using CensoredDistributions, Distributions

    # A NamedTuple of edges is the positional spelling of the Pairs form and
    # builds the same racing-hazard node.
    nt = recur(:ill => (recover = Gamma(2.0, 3.0), die = Gamma(3.0, 2.0)),
        :recover => (:ill => Gamma(2.0, 10.0)))
    pr = recur(:ill => (:recover => Gamma(2.0, 3.0), :die => Gamma(3.0, 2.0)),
        :recover => (:ill => Gamma(2.0, 10.0)))
    @test CensoredDistributions.component_names(nt.nodes[:ill]) ==
          CensoredDistributions.component_names(pr.nodes[:ill])
    @test nt.nodes[:ill] isa Compete
end

@testitem "Resolve fixed-split step samples and scores consistently" begin
    using CensoredDistributions, Distributions, Random
    const CD = CensoredDistributions

    # A fixed-probability split state: rand draws a path and logpdf scores it
    # through the Resolve conditioned term (cause independent of timing).
    m = recur(
        :ill => (:recovered => (Gamma(2.0, 3.0), 0.7),
            :dead => (Gamma(2.0, 8.0), 0.3)),
        :recovered => (:ill => Gamma(2.0, 20.0)))
    path = rand(MersenneTwister(9), m; horizon = 60.0)
    @test isfinite(logpdf(m, path))
    # A single ill -> recovered jump scores log(0.7) + logpdf(Gamma, t).
    one_jump = [(from = :ill, to = :recovered, dwell = 4.0)]
    @test logpdf(m, one_jump) ≈ log(0.7) + logpdf(Gamma(2.0, 3.0), 4.0)
end

@testitem "recur rejects malformed graphs" begin
    using CensoredDistributions, Distributions

    @test_throws ArgumentError recur()
    # A non-Symbol destination.
    @test_throws ArgumentError recur(:a => ("b" => Gamma(2.0, 1.0)))
    # A duplicate state.
    @test_throws ArgumentError recur(
        :a => (:b => Gamma(2.0, 1.0)),
        :a => (:c => Gamma(2.0, 1.0)))
    # A start state that neither has edges nor is any edge's destination.
    @test_throws ArgumentError recur(:a => (:b => Gamma(2.0, 1.0)); start = :z)
end

@testitem "rand simulates a cyclic path that returns to a state" begin
    using CensoredDistributions, Distributions, Random
    const CD = CensoredDistributions

    m = recur(
        :susceptible => (:infected => Gamma(2.0, 4.0)),
        :infected => (:recovered => Gamma(2.0, 3.0),),
        :recovered => (:susceptible => Gamma(2.0, 10.0)))
    # No absorbing state, so a long horizon yields a multi-cycle path.
    path = rand(MersenneTwister(1), m; horizon = 200.0)
    @test path isa StatePath
    @test path.start == :susceptible
    states = CD.visited_states(path)
    # The path must revisit :susceptible (a genuine cycle) at least once after
    # the start, which the acyclic grammar cannot produce.
    @test count(==(:susceptible), states) >= 2
    @test path.stop in (:absorbed, :censored, :maxjumps)
end

@testitem "rand stops at an absorbing state" begin
    using CensoredDistributions, Distributions, Random

    m = recur(
        :well => (:ill => Gamma(2.0, 2.0)),
        :ill => (:dead => Gamma(2.0, 2.0),))
    path = rand(MersenneTwister(2), m)
    @test path.stop == :absorbed
    @test last(path.jumps).to == :dead
end

@testitem "logpdf scores a path by summing per-sojourn terms" begin
    using CensoredDistributions, Distributions, Random
    const CD = CensoredDistributions

    m = recur(
        :well => (:ill => Gamma(2.0, 5.0)),
        :ill => (:well => Gamma(2.0, 3.0), :dead => Gamma(2.0, 10.0)))
    path = rand(MersenneTwister(3), m; horizon = 80.0)

    # The path score equals the sum of the independent per-jump transition
    # log sub-densities (the clock resets each step, so terms factorise).
    jump_terms = [CD._transition_logpdf(m.nodes[j.from], j.to, j.dwell)
                  for j in path.jumps]
    survival = path.stop == :censored ?
               CD._edge_logsurvival(m.nodes[path.censored_state],
        path.censored_for) : 0.0
    @test logpdf(m, path) ≈ sum(jump_terms) + survival
end

@testitem "logpdf of a Compete step is the cause-resolved sub-density" begin
    using CensoredDistributions, Distributions
    const CD = CensoredDistributions

    m = recur(:ill => (:recover => Gamma(2.0, 3.0), :die => Gamma(3.0, 2.0)))
    node = m.nodes[:ill]
    t = 4.0
    # Scoring a `(to, dwell)` transition is exactly the standalone Compete
    # cause-resolved sub-density `log f_j + sum_{k != j} log S_k`.
    j = findfirst(==(:recover), CD.component_names(node))
    @test CD._transition_logpdf(node, :recover, t) ≈
          CD._hazard_cause_logpdf(node, j, t)
end

@testitem "horizon censoring adds the final-sojourn survival term" begin
    using CensoredDistributions, Distributions, Random
    const CD = CensoredDistributions

    # A long single sojourn so a short horizon reliably censors it.
    m = recur(:well => (:ill => Gamma(5.0, 5.0)),
        :ill => (:well => Gamma(2.0, 3.0),))
    path = rand(MersenneTwister(7), m; horizon = 3.0)
    @test path.stop == :censored
    @test path.censored_state == :well
    @test path.censored_for == 3.0
    # The censored path's score is purely the survival of the unfinished sojourn
    # (no jump completed before the horizon).
    @test logpdf(m, path) ≈ CD._edge_logsurvival(m.nodes[:well], 3.0)
end

@testitem "censored and composed sojourns compose on edges" begin
    using CensoredDistributions, Distributions, Random

    # Any UnivariateDistribution is a valid sojourn, including an interval-
    # censored leaf: it scores and samples through the same path machinery.
    m = recur(
        :well => (:ill => interval_censored(Gamma(2.0, 5.0), 1.0)),
        :ill => (:well => Gamma(2.0, 3.0),))
    path = rand(MersenneTwister(4), m; horizon = 40.0)
    @test isfinite(logpdf(m, path))
end

@testitem "path logpdf differentiates through edge parameters" begin
    # A fast ForwardDiff smoke check run in the main suite. The full per-backend
    # AD matrix (ForwardDiff, ReverseDiff, Mooncake, Enzyme) lives in the
    # ADFixtures `:recurrent` scenario group (test/ADFixtures, test/ad).
    using CensoredDistributions, Distributions, Random
    using ForwardDiff

    m = recur(
        :well => (:ill => Gamma(2.0, 5.0)),
        :ill => (:well => Gamma(2.0, 3.0), :dead => Gamma(2.0, 10.0)))
    paths = [rand(MersenneTwister(s), m; horizon = 120.0) for s in 1:30]

    function nll(logscales)
        s = exp.(logscales)
        mm = recur(
            :well => (:ill => Gamma(2.0, s[1])),
            :ill => (:well => Gamma(2.0, s[2]), :dead => Gamma(2.0, s[3])))
        return -sum(p -> logpdf(mm, p), paths)
    end

    g = ForwardDiff.gradient(nll, log.([5.0, 3.0, 10.0]))
    @test all(isfinite, g)
    @test length(g) == 3
end
