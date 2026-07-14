@testitem "compartment_stages lowers a single Exp/Erlang leaf" begin
    using CensoredDistributions, Distributions

    # An Exponential is one stage at rate 1/scale.
    es = compartment_stages(Exponential(2.0))
    @test length(es) == 1
    @test es[1].stages == 1
    @test es[1].rate ≈ 0.5
    @test es[1].name == :delay

    # An Erlang(k, θ) is k stages at rate 1/θ; the chain mean matches the delay.
    gs = compartment_stages(Gamma(3.0, 1.5))
    @test length(gs) == 1
    @test gs[1].stages == 3
    @test gs[1].rate ≈ 1 / 1.5
    @test gs[1].stages / gs[1].rate ≈ mean(Gamma(3.0, 1.5))
end

@testitem "compartment_stages lowers a Sequential chain step by step" begin
    using CensoredDistributions, Distributions

    chain = Sequential(Gamma(2.0, 1.0), Exponential(0.5))
    s = compartment_stages(chain)
    @test length(s) == 2
    @test [st.stages for st in s] == [2, 1]
    @test s[1].rate ≈ 1.0
    @test s[2].rate ≈ 2.0
    # Positional construction names the steps :step_1, :step_2.
    @test [st.name for st in s] == [:step_1, :step_2]

    # Total compartments is the sum of stages.
    @test sum(st.stages for st in s) == 3
end

@testitem "compartment_stages threads compose step names through" begin
    using CensoredDistributions, Distributions

    chain = Sequential((Gamma(2.0, 1.0), Exponential(0.5)),
        (:exposed, :infectious))
    s = compartment_stages(chain)
    @test [st.name for st in s] == [:exposed, :infectious]
end

@testitem "compartment_stages sees through censoring wrappers" begin
    using CensoredDistributions, Distributions

    # A double-interval-censored Erlang leaf still lowers to its free delay's
    # (rate, stages); the censoring bounds are fixed structure, not stages.
    dic = double_interval_censored(Gamma(2.0, 1.0); upper = 20, interval = 1)
    s = compartment_stages(dic)
    @test length(s) == 1
    @test s[1].stages == 2
    @test s[1].rate ≈ 1.0
end

@testitem "compartment_stages rejects non-Exp/Erlang delays" begin
    using CensoredDistributions, Distributions

    # Non-integer Gamma shape has no exact finite linear chain.
    @test_throws ArgumentError compartment_stages(Gamma(2.5, 1.0))
    # Other families are out of scope for the exact lowering.
    @test_throws ArgumentError compartment_stages(LogNormal(1.0, 0.5))
    @test_throws ArgumentError compartment_stages(Weibull(2.0, 1.0))
end

@testitem "compartment_stages moment-matches a non-Erlang leaf" begin
    using CensoredDistributions, Distributions

    # A non-integer-shape Gamma lowers by matching its mean and squared
    # coefficient of variation to the nearest Erlang chain.
    d = Gamma(2.5, 1.0)
    s = compartment_stages(d; moment_match = true)
    @test length(s) == 1
    # SCV = 1/shape = 0.4, so the matched Erlang shape is round(1/0.4) = 2 or 3.
    k = s[1].stages
    @test k == round(Int, 1 / (var(d) / mean(d)^2))
    # The chain mean matches the delay mean exactly (rate = k / mean).
    @test k / s[1].rate ≈ mean(d)
    @test s[1].name == :delay
end

@testitem "compartment_stages moment-matches a LogNormal leaf" begin
    using CensoredDistributions, Distributions

    d = LogNormal(1.0, 0.5)
    s = compartment_stages(d; moment_match = true)
    @test length(s) == 1
    scv = var(d) / mean(d)^2
    @test s[1].stages == round(Int, 1 / scv)
    @test s[1].stages / s[1].rate ≈ mean(d)
end

@testitem "compartment_stages moment_match keeps the exact path exact" begin
    using CensoredDistributions, Distributions

    # An Erlang under moment matching gives the same stages as the exact path.
    exact = compartment_stages(Gamma(3.0, 1.5))
    matched = compartment_stages(Gamma(3.0, 1.5); moment_match = true)
    @test matched[1].stages == exact[1].stages
    @test matched[1].rate ≈ exact[1].rate
end

@testitem "compartment_stages moment-matches a Sequential chain" begin
    using CensoredDistributions, Distributions

    chain = Sequential(Gamma(2.5, 1.0), LogNormal(0.5, 0.3))
    s = compartment_stages(chain; moment_match = true)
    @test length(s) == 2
    @test s[1].stages / s[1].rate ≈ mean(Gamma(2.5, 1.0))
    @test s[2].stages / s[2].rate ≈ mean(LogNormal(0.5, 0.3))
end

@testitem "compartment_stages rejects under-dispersed moment matching" begin
    using CensoredDistributions, Distributions

    # An Erlang needs SCV ≤ 1 (shape ≥ 1); a distribution with SCV > 1 has no
    # Erlang chain that matches both moments, so moment matching errors.
    @test_throws ArgumentError compartment_stages(
        LogNormal(0.0, 1.5); moment_match = true)
end

@testitem "compartment_stages moment match reproduces mean and variance" begin
    using CensoredDistributions, Distributions

    # Regression: the matched Erlang chain must reproduce the target delay's
    # moments. The chain mean (`stages / rate`) is always exact (rate is set to
    # `stages / mean`). The chain variance (`stages / rate²` = `mean² / stages`)
    # equals the target variance exactly when `1 / scv` is an integer, since the
    # matched stage count `k = round(1 / scv)` then makes the chain scv `1/k`
    # hit the target. A LogNormal tuned to scv = 1/4 is that exact case.
    sigma = sqrt(log(1.25))            # exp(σ²) - 1 = 0.25 = scv
    d = LogNormal(0.0, sigma)
    s = compartment_stages(d; moment_match = true)
    @test length(s) == 1
    @test s[1].stages == 4             # round(1 / 0.25)
    chain_mean = s[1].stages / s[1].rate
    chain_var = s[1].stages / s[1].rate^2
    @test chain_mean ≈ mean(d)
    @test chain_var ≈ var(d)

    # For a general non-integer 1/scv the mean stays exact and the variance is
    # reproduced to within the stage-count rounding (here 1/scv = 2.5 -> k = 2).
    g = Gamma(2.5, 1.0)
    sg = compartment_stages(g; moment_match = true)
    @test sg[1].stages / sg[1].rate ≈ mean(g)
    @test isapprox(sg[1].stages / sg[1].rate^2, var(g); rtol = 0.25)
end
