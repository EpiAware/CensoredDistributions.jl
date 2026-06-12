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
