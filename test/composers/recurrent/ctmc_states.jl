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

@testitem "panel_logpdf scores state-at-visit observations" begin
    using CensoredDistributions
    const CD = CensoredDistributions

    m = ctmc(:well => (:ill => 0.2), :ill => (:well => 0.3, :dead => 0.1))
    panel = [(0.0, :well), (3.0, :ill), (7.0, :well)]
    lp = CD.panel_logpdf(m, panel)
    # The panel likelihood is the product of the per-gap transition
    # probabilities, marginalising the hidden jumps.
    P1 = CD.transition_probability(m, 3.0)
    P2 = CD.transition_probability(m, 4.0)
    @test lp ≈ log(P1[1, 2]) + log(P2[2, 1])
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

@testitem "ctmc panel_logpdf differentiates through rates" begin
    # A fast ForwardDiff smoke check run in the main suite. The full per-backend
    # AD matrix (including the Enzyme-forward exp(Qt) gap) lives in the
    # ADFixtures `:recurrent` scenario group (test/ADFixtures, test/ad).
    using CensoredDistributions
    using ForwardDiff
    const CD = CensoredDistributions

    panel = [(0.0, :well), (3.0, :ill), (7.0, :well), (10.0, :ill)]
    function nll(logr)
        r = exp.(logr)
        m = ctmc(:well => (:ill => r[1]),
            :ill => (:well => r[2], :dead => r[3]))
        return -CD.panel_logpdf(m, panel)
    end
    g = ForwardDiff.gradient(nll, log.([0.2, 0.3, 0.1]))
    @test all(isfinite, g)
end
