@testitem "ctmc assembles a valid generator matrix" begin
    using CensoredDistributions
    const CD = CensoredDistributions

    m = ctmc(
        :well => (:ill => 0.2),
        :ill => (:well => 0.3, :dead => 0.1))
    @test m isa CTMCStates
    @test m.states == (:well, :ill, :dead)
    # Rows of a generator sum to zero.
    for i in 1:3
        @test sum(m.Q[i, :]) ≈ 0 atol = 1e-10
    end
    # Off-diagonals are the supplied rates.
    @test m.Q[1, 2] == 0.2
    @test m.Q[2, 1] == 0.3
    @test m.Q[2, 3] == 0.1
end

@testitem "ctmc rejects negative rates" begin
    using CensoredDistributions
    @test_throws ArgumentError ctmc(:a => (:b => -0.1))
end

@testitem "transition_probability is a stochastic matrix" begin
    using CensoredDistributions
    const CD = CensoredDistributions

    m = ctmc(:well => (:ill => 0.2), :ill => (:well => 0.3, :dead => 0.1))
    P = CD.transition_probability(m, 2.0)
    # Each row of exp(Q t) is a probability vector.
    for i in 1:3
        @test sum(P[i, :]) ≈ 1 atol = 1e-8
        @test all(P[i, :] .>= -1e-10)
    end
    # P(0) is the identity.
    P0 = CD.transition_probability(m, 0.0)
    @test P0 ≈ [1.0 0 0; 0 1 0; 0 0 1] atol = 1e-10
    # The dead state is absorbing: it stays dead.
    @test P[3, 3] ≈ 1 atol = 1e-10
end

@testitem "transition_probability matches a two-state analytic solution" begin
    using CensoredDistributions
    const CD = CensoredDistributions

    # A reversible two-state chain a <-> b with rates λ, μ has the closed-form
    # P_aa(t) = (μ + λ e^{-(λ+μ)t}) / (λ + μ).
    λ, μ = 0.4, 0.25
    m = ctmc(:a => (:b => λ), :b => (:a => μ))
    t = 3.0
    P = CD.transition_probability(m, t)
    s = λ + μ
    @test P[1, 1] ≈ (μ + λ * exp(-s * t)) / s atol = 1e-8
    @test P[1, 2] ≈ (λ - λ * exp(-s * t)) / s atol = 1e-8
end

@testitem "logpdf scores state-at-visit panel observations" begin
    using CensoredDistributions
    const CD = CensoredDistributions

    m = ctmc(:well => (:ill => 0.2), :ill => (:well => 0.3, :dead => 0.1))
    panel = [(0.0, :well), (3.0, :ill), (7.0, :well)]
    # `logpdf` is the single front door; it dispatches on the observation shape,
    # so a `(time, state)` panel routes to the `exp(Q Δt)` kernel with no bespoke
    # scoring name.
    lp = logpdf(m, panel)
    # The panel likelihood is the product of the per-gap transition
    # probabilities, marginalising the hidden jumps.
    P1 = CD.transition_probability(m, 3.0)
    P2 = CD.transition_probability(m, 4.0)
    @test lp ≈ log(P1[1, 2]) + log(P2[2, 1])
    # A `(time = , state = )` NamedTuple panel scores identically (order-free).
    panel_nt = [(time = 0.0, state = :well), (time = 3.0, state = :ill),
        (time = 7.0, state = :well)]
    @test logpdf(m, panel_nt) ≈ lp
end

@testitem "logpdf dispatches panel vs jump chain on one model" begin
    using CensoredDistributions
    const CD = CensoredDistributions

    m = ctmc(:well => (:ill => 0.2), :ill => (:well => 0.3, :dead => 0.1))
    # A `(time, state)` panel and a `(from, to, dwell)` jump chain are two
    # different likelihoods; the same `logpdf(model, data)` covers both by
    # recognising the element shape, so `panel_logpdf` is unnecessary.
    panel = [(0.0, :well), (3.0, :ill)]
    jumps = [(from = :well, to = :ill, dwell = 4.0)]
    @test logpdf(m, panel) ≈ log(CD.transition_probability(m, 3.0)[1, 2])
    @test logpdf(m, jumps) ≈ log(0.2) - 0.2 * 4.0
end

@testitem "ctmc jump-chain logpdf is the exponential semi-Markov term" begin
    using CensoredDistributions, Distributions
    const CD = CensoredDistributions

    rates = (ill = 0.3, dead = 0.1)
    m = ctmc(:well => (:ill => 0.2), :ill => (:well => 0.3, :dead => 0.1))
    # A line-list path with known transition times.
    jumps = [(from = :well, to = :ill, dwell = 4.0),
        (from = :ill, to = :dead, dwell = 2.0)]
    # well: only exit rate 0.2, so log(0.2) - 0.2*4; ill: total exit 0.4, taking
    # the dead edge (rate 0.1): log(0.1) - 0.4*2.
    expected = (log(0.2) - 0.2 * 4.0) + (log(0.1) - 0.4 * 2.0)
    @test logpdf(m, jumps) ≈ expected
end

@testitem "ctmc rand respects the start state and exponential sojourns" begin
    using CensoredDistributions, Distributions, Random
    const CD = CensoredDistributions

    m = ctmc(:well => (:ill => 0.2), :ill => (:well => 0.3, :dead => 0.1))
    path = rand(MersenneTwister(5), m; start = :well, horizon = 50.0)
    @test path.start == :well
    @test first(path.jumps).from == :well
    # The path's score reads back finite under the model.
    @test isfinite(logpdf(m, path))
end

@testitem "ctmc panel logpdf differentiates through rates" begin
    # A fast ForwardDiff smoke check run in the main suite. The full per-backend
    # AD matrix (including the Enzyme-forward exp(Qt) gap) lives in the
    # ADFixtures `:recurrent` scenario group (test/ADFixtures, test/ad).
    using CensoredDistributions
    using ForwardDiff

    panel = [(0.0, :well), (3.0, :ill), (7.0, :well), (10.0, :ill)]
    function nll(logr)
        r = exp.(logr)
        m = ctmc(:well => (:ill => r[1]),
            :ill => (:well => r[2], :dead => r[3]))
        return -logpdf(m, panel)
    end
    g = ForwardDiff.gradient(nll, log.([0.2, 0.3, 0.1]))
    @test all(isfinite, g)
end

@testitem "all-exponential recur auto-dispatches to the CTMC fast path" begin
    using CensoredDistributions, Distributions
    const CD = CensoredDistributions

    # Every edge exponential and every state racing: recur returns a CTMCStates.
    m = recur(:well => (:ill => Exponential(5.0)),
        :ill => (:well => Exponential(1 / 0.3), :dead => Exponential(10.0)))
    @test m isa CTMCStates
    # The generator rates are 1 / scale of each exponential edge.
    i = CD.state_index(m, :well)
    j = CD.state_index(m, :ill)
    @test m.Q[i, j] ≈ 1 / 5.0
    # A Gamma edge keeps the semi-Markov RecurrentStates; a fixed-probability
    # Resolve split (even all-exponential) is not a CTMC, so it stays too.
    @test recur(:well => (:ill => Gamma(2.0, 5.0)),
        :ill => (:dead => Gamma(2.0, 3.0),)) isa RecurrentStates
    @test recur(
        :ill => (:rec => (Exponential(3.0), 0.7),
            :dead => (Exponential(8.0), 0.3)),
        :rec => (:ill => Exponential(20.0))) isa RecurrentStates
end

@testitem "all-exponential semi-Markov score equals the CTMC score" begin
    using CensoredDistributions, Distributions
    const CD = CensoredDistributions

    # A hand-built exponential RecurrentStates (bypassing recur's auto-dispatch)
    # scores a jump chain by the semi-Markov term; its ctmc(...) conversion gives
    # the identical number, confirming the CTMC IS the exponential special case.
    rs = CD.RecurrentStates(
        Dict(:well => (:ill => Exponential(5.0)),
            :ill => CD.Compete(:well => Exponential(1 / 0.3),
                :dead => Exponential(10.0))), :well)
    cv = ctmc(rs)
    @test cv isa CTMCStates
    jumps = [(from = :well, to = :ill, dwell = 2.0),
        (from = :ill, to = :dead, dwell = 1.5)]
    @test CD._jumps_logpdf(rs, jumps) ≈ logpdf(cv, jumps)
    # Panel data on the semi-Markov model routes through the CTMC representation.
    panel = [(0.0, :well), (3.0, :ill), (7.0, :well)]
    @test logpdf(rs, panel) ≈ logpdf(cv, panel)
    # A non-exponential model cannot convert: ctmc(...) and panel scoring error.
    g = recur(:well => (:ill => Gamma(2.0, 5.0)),
        :ill => (:dead => Gamma(2.0, 3.0),))
    @test_throws ArgumentError ctmc(g)
    @test_throws ArgumentError logpdf(g, panel)
end
