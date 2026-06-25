@testitem "linear_chain_stages lowers a single Exp/Erlang leaf" begin
    using CensoredDistributions, Distributions

    # An Exponential is one stage at rate 1/scale.
    es = linear_chain_stages(Exponential(2.0))
    @test length(es) == 1
    @test es[1].stages == 1
    @test es[1].rate ≈ 0.5
    @test es[1].name == :delay

    # An Erlang(k, θ) is k stages at rate 1/θ; the chain mean matches the delay.
    gs = linear_chain_stages(Gamma(3.0, 1.5))
    @test length(gs) == 1
    @test gs[1].stages == 3
    @test gs[1].rate ≈ 1 / 1.5
    @test gs[1].stages / gs[1].rate ≈ mean(Gamma(3.0, 1.5))
end

@testitem "linear_chain_stages lowers a Sequential chain step by step" begin
    using CensoredDistributions, Distributions

    chain = Sequential(Gamma(2.0, 1.0), Exponential(0.5))
    s = linear_chain_stages(chain)
    @test length(s) == 2
    @test [st.stages for st in s] == [2, 1]
    @test s[1].rate ≈ 1.0
    @test s[2].rate ≈ 2.0
    # Positional construction names the steps :step_1, :step_2.
    @test [st.name for st in s] == [:step_1, :step_2]

    # Total compartments is the sum of stages.
    @test sum(st.stages for st in s) == 3
end

@testitem "linear_chain_stages threads compose step names through" begin
    using CensoredDistributions, Distributions

    chain = Sequential((Gamma(2.0, 1.0), Exponential(0.5)),
        (:exposed, :infectious))
    s = linear_chain_stages(chain)
    @test [st.name for st in s] == [:exposed, :infectious]
end

@testitem "linear_chain_stages sees through censoring wrappers" begin
    using CensoredDistributions, Distributions

    # A double-interval-censored Erlang leaf still lowers to its free delay's
    # (rate, stages); the censoring bounds are fixed structure, not stages.
    dic = double_interval_censored(Gamma(2.0, 1.0); upper = 20, interval = 1)
    s = linear_chain_stages(dic)
    @test length(s) == 1
    @test s[1].stages == 2
    @test s[1].rate ≈ 1.0
end

@testitem "linear_chain_stages rejects non-Exp/Erlang delays" begin
    using CensoredDistributions, Distributions

    # Non-integer Gamma shape has no exact finite linear chain.
    @test_throws ArgumentError linear_chain_stages(Gamma(2.5, 1.0))
    # Other families are out of scope for the exact lowering.
    @test_throws ArgumentError linear_chain_stages(LogNormal(1.0, 0.5))
    @test_throws ArgumentError linear_chain_stages(Weibull(2.0, 1.0))
end

@testitem "linear_chain_stages moment-matches a non-Erlang leaf" begin
    using CensoredDistributions, Distributions

    # A non-integer-shape Gamma lowers by matching its mean and squared
    # coefficient of variation to the nearest Erlang chain.
    d = Gamma(2.5, 1.0)
    s = linear_chain_stages(d; moment_match = true)
    @test length(s) == 1
    # SCV = 1/shape = 0.4, so the matched Erlang shape is round(1/0.4) = 2 or 3.
    k = s[1].stages
    @test k == round(Int, 1 / (var(d) / mean(d)^2))
    # The chain mean matches the delay mean exactly (rate = k / mean).
    @test k / s[1].rate ≈ mean(d)
    @test s[1].name == :delay
end

@testitem "linear_chain_stages moment-matches a LogNormal leaf" begin
    using CensoredDistributions, Distributions

    d = LogNormal(1.0, 0.5)
    s = linear_chain_stages(d; moment_match = true)
    @test length(s) == 1
    scv = var(d) / mean(d)^2
    @test s[1].stages == round(Int, 1 / scv)
    @test s[1].stages / s[1].rate ≈ mean(d)
end

@testitem "linear_chain_stages moment_match keeps the exact path exact" begin
    using CensoredDistributions, Distributions

    # An Erlang under moment matching gives the same stages as the exact path.
    exact = linear_chain_stages(Gamma(3.0, 1.5))
    matched = linear_chain_stages(Gamma(3.0, 1.5); moment_match = true)
    @test matched[1].stages == exact[1].stages
    @test matched[1].rate ≈ exact[1].rate
end

@testitem "linear_chain_stages moment-matches a Sequential chain" begin
    using CensoredDistributions, Distributions

    chain = Sequential(Gamma(2.5, 1.0), LogNormal(0.5, 0.3))
    s = linear_chain_stages(chain; moment_match = true)
    @test length(s) == 2
    @test s[1].stages / s[1].rate ≈ mean(Gamma(2.5, 1.0))
    @test s[2].stages / s[2].rate ≈ mean(LogNormal(0.5, 0.3))
end

@testitem "linear_chain_stages rejects under-dispersed moment matching" begin
    using CensoredDistributions, Distributions

    # An Erlang needs SCV ≤ 1 (shape ≥ 1); a distribution with SCV > 1 has no
    # Erlang chain that matches both moments, so moment matching errors.
    @test_throws ArgumentError linear_chain_stages(
        LogNormal(0.0, 1.5); moment_match = true)
end
