# Tests for the DynamicPPL extension submodels: `primary_censored_model`,
# `interval_censored_model`, `double_interval_censored_model`. They check that
# the submodels compile, that the marginal submodel log-density equals the direct
# `logpdf`, that the `weight` and `origin` keywords behave, and that a short NUTS
# run samples. DynamicPPL/Turing are test dependencies; the core stays
# Turing-free.

@testitem "primary_censored_model: marginal == logpdf" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    d = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))

    @model demo(d, y) = obs ~ to_submodel(primary_censored_model(d, y))

    for y in (0.5, 2.0, 5.0)
        @test only(logjoint(demo(d, y), (;))) ≈ logpdf(d, y)
    end
end

@testitem "primary_censored_model: weight scales the contribution" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    d = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))

    @model function demo(d, y, w)
        obs ~ to_submodel(primary_censored_model(d, y; weight = w))
    end

    y = 2.0
    @test only(logjoint(demo(d, y, 3), (;))) ≈ 3 * logpdf(d, y)
    # `nothing` weight leaves the contribution unweighted.
    @model demo_nothing(d, y) = obs ~ to_submodel(
        primary_censored_model(d, y; weight = nothing))
    @test only(logjoint(demo_nothing(d, y), (;))) ≈ logpdf(d, y)
end

@testitem "primary_censored_model: latent samples p inside the model" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint, condition, VarInfo,
                      @varname

    d = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))
    ld = latent(d)

    # The latent submodel declares the primary `p` INSIDE the model; the user
    # never passes it. It appears in the model's variables (prefixed by the
    # submodel LHS `obs`).
    @model demo(ld, y) = obs ~ to_submodel(primary_censored_model(ld, y))

    vi = VarInfo(demo(ld, 3.0))
    pkeys = collect(keys(vi))
    @test any(vn -> occursin("p", string(vn)), pkeys)
    # `p` is sampled inside, not passed: the only model variable is the latent.
    @test length(pkeys) == 1

    # Conditioning on a value of the latent `p`, the log-density equals the
    # primary prior plus the conditional of the observed given that `p`.
    p, y = 0.4, 3.0
    conditioned = condition(demo(ld, y), (@varname(obs.p) => p))
    @test logjoint(conditioned, (;)) ≈
          logpdf(get_primary_event(d), p) +
          primary_conditional_logpdf(d, p, y)
end

@testitem "primary_censored_model: latent model generates [p, y]" begin
    using CensoredDistributions, Distributions, Random
    using DynamicPPL: @model, to_submodel

    d = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))
    ld = latent(d)

    # The latent submodel returns the observed `y`. With `y` unobserved both `p`
    # and `y` are sampled (`~`), so calling the submodel generates `[p, y]`. The
    # generated observed time exceeds the primary (`y = p + delay`, delay > 0).
    @model function gen(ld)
        inner = to_submodel(primary_censored_model(ld, missing), false)
        y ~ inner
        return y
    end

    Random.seed!(4)
    draws = [gen(ld)() for _ in 1:200]
    @test all(>(0), draws)
end

@testitem "primary_censored_model: weight applies on the latent path" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint, condition, @varname

    d = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))
    ld = latent(d)
    p, y, w = 0.4, 3.0, 3

    @model function demo(ld, y, w)
        obs ~ to_submodel(primary_censored_model(ld, y; weight = w))
    end

    # The weight scales the conditional likelihood (the observation), not the
    # primary prior: prior(p) + w * conditional(y | p).
    conditioned = condition(demo(ld, y, w), (@varname(obs.p) => p))
    @test logjoint(conditioned, (;)) ≈
          logpdf(get_primary_event(d), p) +
          w * primary_conditional_logpdf(d, p, y)
end

@testitem "interval_censored_model: marginal == logpdf" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    d = interval_censored(LogNormal(1.5, 0.75), 1.0)

    @model demo(d, y) = obs ~ to_submodel(interval_censored_model(d, y))

    for y in (1.0, 2.0, 4.0)
        @test only(logjoint(demo(d, y), (;))) ≈ logpdf(d, y)
    end

    @model function demo_w(d, y, w)
        obs ~ to_submodel(interval_censored_model(d, y; weight = w))
    end
    @test only(logjoint(demo_w(d, 2.0, 5), (;))) ≈ 5 * logpdf(d, 2.0)
end

@testitem "double_interval_censored_model: marginal == logpdf" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    d = double_interval_censored(
        LogNormal(1.5, 0.75); upper = 10, interval = 1)

    @model demo(d, y) = obs ~ to_submodel(double_interval_censored_model(d, y))

    for y in (1.0, 3.0, 6.0)
        @test only(logjoint(demo(d, y), (;))) ≈ logpdf(d, y)
    end

    @model function demo_w(d, y, w)
        obs ~ to_submodel(double_interval_censored_model(d, y; weight = w))
    end
    @test only(logjoint(demo_w(d, 3.0, 4), (;))) ≈ 4 * logpdf(d, 3.0)
end

@testitem "primary_censored_model: short NUTS run samples" begin
    using CensoredDistributions, Distributions, Random
    using Turing: Turing, NUTS, sample, @model, to_submodel, @addlogprob!

    rng = Random.MersenneTwister(1)
    truth = LogNormal(1.5, 0.5)
    n = 50
    samples = rand(rng, truth, n) .+ rand(rng, Uniform(0, 1), n)

    # A single distribution scores every record. The submodel exposes no latent
    # in the marginal case, so the records can be accumulated directly.
    @model function fit(samples)
        mu ~ Normal(1.5, 1.0)
        sigma ~ truncated(Normal(0.5, 0.5); lower = 0.05)
        d = primary_censored(LogNormal(mu, sigma), Uniform(0, 1))
        for s in samples
            @addlogprob! logpdf(d, s)
        end
    end

    chain = sample(rng, fit(samples), NUTS(), 100; progress = false)
    @test size(chain, 1) == 100
    @test all(isfinite, chain[:mu])
end

@testitem "primary_censored_model: submodel composes in a NUTS run" begin
    using CensoredDistributions, Distributions, Random
    using Turing: Turing, NUTS, sample, @model, to_submodel

    rng = Random.MersenneTwister(2)
    truth = LogNormal(1.5, 0.5)
    sample_val = rand(rng, truth) + rand(rng, Uniform(0, 1))

    # A single-record model exercising the submodel inside a sampled `@model`.
    @model function fit(y)
        mu ~ Normal(1.5, 1.0)
        sigma ~ truncated(Normal(0.5, 0.5); lower = 0.05)
        d = primary_censored(LogNormal(mu, sigma), Uniform(0, 1))
        obs ~ to_submodel(primary_censored_model(d, y))
    end

    chain = sample(rng, fit(sample_val), NUTS(), 100; progress = false)
    @test size(chain, 1) == 100
    @test all(isfinite, chain[:mu])
end

@testitem "primary_censored_model: latent flow samples in a NUTS run" begin
    using CensoredDistributions, Distributions, Random
    using Turing: Turing, NUTS, sample, @model, to_submodel
    using DynamicPPL: prefix

    rng = Random.MersenneTwister(3)
    truth = LogNormal(1.5, 0.5)
    n = 20
    samples = rand(rng, truth, n) .+ rand(rng, Uniform(0, 1), n)

    # Latent flow: each record's primary `p` is sampled inside its submodel, so
    # the sampler explores `mu`, `sigma`, and one latent `p` per record. Each
    # submodel is prefixed so the per-record latents get distinct names.
    @model function fit(samples)
        mu ~ Normal(1.5, 1.0)
        sigma ~ truncated(Normal(0.5, 0.5); lower = 0.05)
        d = latent(primary_censored(LogNormal(mu, sigma), Uniform(0, 1)))
        for i in eachindex(samples)
            x ~ to_submodel(
                prefix(primary_censored_model(d, samples[i]),
                    Symbol("rec", i)), false)
        end
    end

    chain = sample(rng, fit(samples), NUTS(), 100; progress = false)
    @test size(chain, 1) == 100
    @test all(isfinite, chain[:mu])
end

# ===========================================================================
# Composer models (#335, PR3d): NamedTuple-row data interface
# ===========================================================================
# `composed_distribution_model(d, row)` is the single generic record entry: a
# leaf/univariate `d` delegates to the matching leaf model, a composed `d`
# recurses. The composed methods take observations as a `NamedTuple` ROW keyed by
# event name; `missing` fields drive per-record marginalise-vs-condition; a
# reserved `weight`/`count` field or a `weight =` kwarg scales the likelihood.
# Marginal vs latent is dispatch on the struct type (bare composer marginal,
# `latent`-wrapped composer latent).

@testitem "composed_distribution_model: leaf delegation routes correctly" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint, VarInfo

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    # PrimaryCensored leaf -> primary_censored_model: bare value and one-event
    # row both score the marginal logpdf; a reserved weight scales it.
    pc = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))
    @test only(logjoint(demo(pc, 2.0), (;))) ≈ logpdf(pc, 2.0)
    @test only(logjoint(demo(pc, (delay = 2.0,)), (;))) ≈ logpdf(pc, 2.0)
    @test only(logjoint(demo(pc, (delay = 2.0, weight = 3)), (;))) ≈
          3 * logpdf(pc, 2.0)

    # IntervalCensored leaf -> interval_censored_model.
    ic = interval_censored(LogNormal(1.5, 0.75), 1.0)
    @test only(logjoint(demo(ic, 2.0), (;))) ≈ logpdf(ic, 2.0)

    # double_interval_censored / other univariate -> the marginal univariate
    # leaf model (fallback).
    dic = double_interval_censored(
        LogNormal(1.5, 0.75); upper = 10, interval = 1)
    @test only(logjoint(demo(dic, 3.0), (;))) ≈ logpdf(dic, 3.0)

    # Latent PrimaryCensored leaf -> the latent leaf path: `p` sampled inside.
    ld = latent(pc)
    @test length(keys(VarInfo(demo(ld, 3.0)))) == 1
end

@testitem "composer model: marginal Sequential submodel == direct logpdf" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint, VarInfo

    seq = Sequential(
        primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    # All-observed row: log-density equals the censored composer logpdf on the
    # positional event vector built from the field order.
    row = (onset = 0.0, admit = 2.0, death = 5.0)
    ev = Vector{Union{Missing, Float64}}([0.0, 2.0, 5.0])
    @test only(logjoint(demo(seq, row), (;))) ≈ logpdf(seq, ev)

    # A missing intermediate marginalises (convolution): still matches the
    # direct censored composer logpdf for that missingness pattern.
    row2 = (onset = 0.0, admit = missing, death = 5.0)
    ev2 = Vector{Union{Missing, Float64}}([0.0, missing, 5.0])
    @test only(logjoint(demo(seq, row2), (;))) ≈ logpdf(seq, ev2)

    # The marginal composer declares no latent: no VarInfo variables.
    @test isempty(keys(VarInfo(demo(seq, row))))
end

@testitem "composer model: marginal Parallel submodel == direct logpdf" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    par = Parallel(
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)),
        primary_censored(LogNormal(1.0, 0.5), Uniform(0, 1)))

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    # Shared-origin event row [origin, y1, y2]; present origin conditions.
    row = (origin = 0.3, y1 = 2.5, y2 = 3.1)
    ev = Vector{Union{Missing, Float64}}([0.3, 2.5, 3.1])
    @test only(logjoint(demo(par, row), (;))) ≈ logpdf(par, ev)

    # Missing origin marginalises by the shared 1-D integral.
    row2 = (origin = missing, y1 = 2.5, y2 = 3.1)
    ev2 = Vector{Union{Missing, Float64}}([missing, 2.5, 3.1])
    @test only(logjoint(demo(par, row2), (;))) ≈ logpdf(par, ev2)
end

@testitem "composer model: Resolve marginal == mixture logpdf" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    cmp = Resolve(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.7))

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    for y in (1.0, 4.0, 8.0)
        @test only(logjoint(demo(cmp, (resolve = y,)), (;))) ≈ logpdf(cmp, y)
    end
    # `count` reserved field scales the likelihood.
    @test only(logjoint(demo(cmp, (resolve = 4.0, count = 5)), (;))) ≈
          5 * logpdf(cmp, 4.0)
end

@testitem "Resolve self-dispatch: observed outcome conditions (#329)" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    cmp = Resolve(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.7))

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    # Exactly one outcome time observed -> condition on that branch:
    # log(p[obs]) + logpdf(delay[obs], gap). Death observed at 4.0.
    lj = only(logjoint(demo(cmp, (death = 4.0, disch = missing)), (;)))
    @test lj ≈ log(0.3) + logpdf(Gamma(1.5, 1.0), 4.0)

    # Discharge observed instead.
    lj2 = only(logjoint(demo(cmp, (death = missing, disch = 2.5)), (;)))
    @test lj2 ≈ log(0.7) + logpdf(Gamma(2.0, 1.5), 2.5)

    # A reserved weight scales the conditioned contribution.
    lj3 = only(logjoint(demo(cmp, (death = 4.0, disch = missing, weight = 3)),
        (;)))
    @test lj3 ≈ 3 * (log(0.3) + logpdf(Gamma(1.5, 1.0), 4.0))

    # Two observed outcomes is an error (at most one resolves).
    @test_throws ArgumentError logjoint(
        demo(cmp, (death = 4.0, disch = 2.5)), (;))
end

@testitem "Resolve self-dispatch: unknown outcome marginalises (#329)" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    cmp = Resolve(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.7))

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    # Resolved at a known time but the outcome unknown (`resolved` field, all
    # outcome columns missing) -> the mixture marginal.
    for y in (1.0, 4.0, 8.0)
        lj = only(logjoint(
            demo(cmp, (resolved = y, death = missing, disch = missing)), (;)))
        @test lj ≈ logpdf(cmp, y)
    end

    # All outcome columns missing and no resolution time -> fully missing,
    # contributes nothing.
    lj0 = only(logjoint(demo(cmp, (death = missing, disch = missing)), (;)))
    @test lj0 ≈ 0.0
end

@testitem "Resolve self-dispatch: per-row branch-prob override (#329)" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    cmp = Resolve(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.7))

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    # A per-record `branch_probs` field OVERRIDES the stored probabilities
    # (covariate CFR flows in here). NamedTuple form.
    row = (death = 4.0, disch = missing, branch_probs = (death = 0.6, disch = 0.4))
    lj = only(logjoint(demo(cmp, row), (;)))
    @test lj ≈ log(0.6) + logpdf(Gamma(1.5, 1.0), 4.0)
    @test lj != log(0.3) + logpdf(Gamma(1.5, 1.0), 4.0)

    # Scalar form for a two-outcome node is the FIRST outcome's probability.
    row_s = (death = 4.0, disch = missing, branch_probs = 0.6)
    @test only(logjoint(demo(cmp, row_s), (;))) ≈ lj

    # The override also reweights the marginalise (unknown-outcome) path.
    lj_marg = only(logjoint(
        demo(cmp, (resolved = 4.0, branch_probs = (death = 0.6, disch = 0.4))),
        (;)))
    @test lj_marg ≈ logpdf(
        MixtureModel([Gamma(1.5, 1.0), Gamma(2.0, 1.5)], [0.6, 0.4]), 4.0)

    # Out-of-range / non-summing overrides error.
    @test_throws ArgumentError logjoint(
        demo(cmp, (death = 4.0, branch_probs = (death = 1.2, disch = -0.2))), (;))
    @test_throws ArgumentError logjoint(
        demo(cmp, (death = 4.0, branch_probs = (death = 0.6, disch = 0.6))), (;))
end

@testitem "Resolve self-dispatch: covariate CFR recovers beta (#329)" begin
    using CensoredDistributions, Distributions, Random
    using Turing: Turing, NUTS, sample, @model, to_submodel
    using DynamicPPL: prefix

    _logistic(z) = 1 / (1 + exp(-z))

    rng = Random.MersenneTwister(202)
    death_delay = Gamma(2.0, 1.5)
    disch_delay = Gamma(2.0, 1.0)
    # True logistic CFR: p_death(x) = logistic(beta0 + beta1 * x).
    beta0, beta1 = -0.4, 1.1
    n = 400
    xs = randn(rng, n)
    rows = map(1:n) do i
        p = _logistic(beta0 + beta1 * xs[i])
        if rand(rng) < p
            (death = rand(rng, death_delay), disch = missing)
        else
            (death = missing, disch = rand(rng, disch_delay))
        end
    end

    # The regression is PLAIN TURING; the per-record probability is PASSED IN to
    # the node via the reserved `branch_probs` field (the node only consumes it).
    @model function fit(rows, xs)
        b0 ~ Normal(0, 1)
        b1 ~ Normal(0, 1)
        cmp = Resolve(:death => (death_delay, 0.5),
            :disch => (disch_delay, 0.5))
        for i in eachindex(rows)
            p = _logistic(b0 + b1 * xs[i])
            row = merge(rows[i], (branch_probs = p,))
            x ~ to_submodel(
                prefix(composed_distribution_model(cmp, row),
                    Symbol("rec", i)), false)
        end
    end

    chain = sample(rng, fit(rows, xs), NUTS(), 300; progress = false)
    @test mean(chain[:b1]) > 0.5
    @test abs(mean(chain[:b0]) - beta0) < 0.5
    @test abs(mean(chain[:b1]) - beta1) < 0.6
end

@testitem "Resolve self-dispatch: AD flows through the MARGINALISED path (#372)" begin
    using CensoredDistributions, Distributions
    using Turing: Turing, @model, to_submodel, AutoForwardDiff
    using DynamicPPL: DynamicPPL, LogDensityFunction
    import DynamicPPL.LogDensityProblems as LDP

    _logistic(z) = 1 / (1 + exp(-z))
    death_d, disch_d = Gamma(2.0, 3.0), Gamma(2.0, 1.0)
    # UNKNOWN-outcome (marginalised) records: a resolved resolution time but no
    # per-outcome column, so the mixture is evaluated. The per-record CFR is
    # logistic(beta * x), passed into the node, so the gradient w.r.t. beta MUST
    # flow through the branch probabilities on the MARGINALISE path (#372: the old
    # `float.(probs)` stripped the Dual and zeroed this gradient).
    specs = [(t = 4.0, x = 0.6), (t = 2.5, x = -0.9), (t = 6.0, x = 1.3)]

    @model function fit(specs)
        beta ~ Normal(0.0, 1.0)
        cmp = Resolve(:death => (death_d, 0.5), :disch => (disch_d, 0.5))
        for i in eachindex(specs)
            p = _logistic(beta * specs[i].x)
            row = (resolved = specs[i].t, death = missing, disch = missing,
                branch_probs = p)
            y ~ to_submodel(composed_distribution_model(cmp, row), false)
        end
    end

    ldf = LogDensityFunction(fit(specs); adtype = AutoForwardDiff())
    v, g = LDP.logdensity_and_gradient(ldf, [0.4])
    @test length(g) == 1
    @test all(isfinite, g)
    # The beta gradient is non-zero ONLY if the Dual reaches the marginalised
    # mixture weights, which is exactly the #372 fix.
    @test any(!iszero, g)
end

@testitem "Resolve self-dispatch: AD through conditioned + per-row prob" begin
    using CensoredDistributions, Distributions
    using Turing: Turing, @model, to_submodel, AutoForwardDiff
    using DynamicPPL: DynamicPPL, LogDensityFunction
    import DynamicPPL.LogDensityProblems as LDP

    _logistic(z) = 1 / (1 + exp(-z))
    disch_delay = Gamma(2.0, 1.0)
    # One conditioned-death record, one conditioned-discharge record, and one
    # unknown-outcome (marginalised) record. The per-record prob is
    # logistic(beta * x), passed into the node.
    specs = [
        (kind = :death, t = 3.0, x = 0.4),
        (kind = :disch, t = 2.0, x = -0.7),
        (kind = :resolved, t = 4.5, x = 1.2)
    ]

    function _row(spec, beta)
        p = _logistic(beta * spec.x)
        bp = (branch_probs = p,)
        spec.kind === :death &&
            return merge((death = spec.t, disch = missing), bp)
        spec.kind === :disch &&
            return merge((death = missing, disch = spec.t), bp)
        return merge((resolved = spec.t,), bp)
    end

    # `sh` parameterises a conditioned death branch; `beta` drives the passed-in
    # per-record prob. The log-density flows through both the conditioned and the
    # per-row-prob paths, so a ForwardDiff gradient over (sh, beta) exercises both.
    @model function fit(specs)
        sh ~ truncated(Normal(2.0, 0.5); lower = 0.2)
        beta ~ Normal(0.0, 1.0)
        cmp = Resolve(:death => (Gamma(sh, 1.5), 0.5),
            :disch => (disch_delay, 0.5))
        for i in eachindex(specs)
            y ~ to_submodel(
                composed_distribution_model(cmp, _row(specs[i], beta)), false)
        end
    end

    ldf = LogDensityFunction(fit(specs); adtype = AutoForwardDiff())
    v, g = LDP.logdensity_and_gradient(ldf, [2.0, 0.8])
    @test length(g) == 2
    @test all(isfinite, g)
    @test any(!iszero, g)
end

@testitem "Nested Resolve: bdbv tree scores per-record by name (#333)" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    edge(mu,
        sigma) = double_interval_censored(LogNormal(mu, sigma);
        primary_event = Uniform(0, 1), interval = 1.0)
    e_oa, e_on = edge(1.4, 0.4), edge(1.9, 0.5)
    cfr = 0.3
    death_d, disch_d = Gamma(2.0, 3.0), Gamma(2.0, 1.0)
    cmp = Resolve(:death => (death_d, cfr), :discharge => (disch_d, 1 - cfr))
    # onset -> {admit -> Resolve(death, discharge), notif}.
    d = Parallel(Sequential(e_oa, cmp), e_on)

    @model demo(dd, r) = obs ~ to_submodel(composed_distribution_model(dd, r))

    # A record's death/discharge/notif columns, by name; missingness selects the
    # observed outcome. The onset/admit edges are positional (`event_1/2`).
    death_row = (event_1 = 0.0, event_2 = 4.0, death = 12.0,
        discharge = missing, event_3 = 9.0)
    ref_death = logpdf(e_oa, 4.0) + log(cfr) + logpdf(death_d, 8.0) +
                logpdf(e_on, 9.0)
    @test only(logjoint(demo(d, death_row), (;))) ≈ ref_death

    disch_row = (event_1 = 0.0, event_2 = 4.0, death = missing,
        discharge = 11.0, event_3 = 9.0)
    ref_disch = logpdf(e_oa, 4.0) + log(1 - cfr) + logpdf(disch_d, 7.0) +
                logpdf(e_on, 9.0)
    @test only(logjoint(demo(d, disch_row), (;))) ≈ ref_disch
end

@testitem "Nested Resolve: per-row branch_probs override (#333)" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    edge(mu,
        sigma) = double_interval_censored(LogNormal(mu, sigma);
        primary_event = Uniform(0, 1), interval = 1.0)
    e_oa, e_on = edge(1.4, 0.4), edge(1.9, 0.5)
    death_d, disch_d = Gamma(2.0, 3.0), Gamma(2.0, 1.0)
    cmp = Resolve(:death => (death_d, 0.3), :discharge => (disch_d, 0.7))
    d = Parallel(Sequential(e_oa, cmp), e_on)

    @model demo(dd, r) = obs ~ to_submodel(composed_distribution_model(dd, r))

    base = logpdf(e_oa, 4.0) + logpdf(death_d, 8.0) + logpdf(e_on, 9.0)
    # NamedTuple override OVERRIDES the stored branch_probs for this record.
    row = (event_1 = 0.0, event_2 = 4.0, death = 12.0, discharge = missing,
        event_3 = 9.0, branch_probs = (death = 0.6, discharge = 0.4))
    @test only(logjoint(demo(d, row), (;))) ≈ base + log(0.6)
    @test !isapprox(only(logjoint(demo(d, row), (;))), base + log(0.3))

    # Scalar override (first outcome's probability of the two-outcome node).
    row_s = (event_1 = 0.0, event_2 = 4.0, death = 12.0, discharge = missing,
        event_3 = 9.0, branch_probs = 0.6)
    @test only(logjoint(demo(d, row_s), (;))) ≈ base + log(0.6)

    # Out-of-range / non-summing overrides error.
    bad = (event_1 = 0.0, event_2 = 4.0, death = 12.0, discharge = missing,
        event_3 = 9.0, branch_probs = (death = 1.2, discharge = -0.2))
    @test_throws ArgumentError logjoint(demo(d, bad), (;))
end

@testitem "Nested Resolve: N-ary override and conditioning (#333)" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    edge(mu,
        sigma) = double_interval_censored(LogNormal(mu, sigma);
        primary_event = Uniform(0, 1), interval = 1.0)
    e_oa, e_on = edge(1.4, 0.4), edge(1.9, 0.5)
    death_d, disch_d, trans_d = Gamma(2.0, 3.0), Gamma(2.0, 1.0), Gamma(3.0, 1.0)
    cmp = Resolve(:death => (death_d, 0.2),
        :discharge => (disch_d, 0.5), :transfer => (trans_d, 0.3))
    d = Parallel(Sequential(e_oa, cmp), e_on)

    @model demo(dd, r) = obs ~ to_submodel(composed_distribution_model(dd, r))

    base = logpdf(e_oa, 4.0) + logpdf(trans_d, 6.0) + logpdf(e_on, 9.0)
    # A three-entry override summing to one, transfer observed.
    row = (event_1 = 0.0, event_2 = 4.0, death = missing, discharge = missing,
        transfer = 10.0, event_3 = 9.0,
        branch_probs = (death = 0.1, discharge = 0.6, transfer = 0.3))
    @test only(logjoint(demo(d, row), (;))) ≈ base + log(0.3)

    # A scalar override is rejected for a three-outcome node (ambiguous).
    bad = (event_1 = 0.0, event_2 = 4.0, death = missing, discharge = missing,
        transfer = 10.0, event_3 = 9.0, branch_probs = 0.3)
    @test_throws ArgumentError logjoint(demo(d, bad), (;))
end

@testitem "Nested Resolve: AD through conditioned tree + per-row prob" begin
    using CensoredDistributions, Distributions
    using Turing: Turing, @model, to_submodel, AutoForwardDiff
    using DynamicPPL: DynamicPPL, LogDensityFunction
    import DynamicPPL.LogDensityProblems as LDP

    _logistic(z) = 1 / (1 + exp(-z))
    e_on = double_interval_censored(LogNormal(1.9, 0.5);
        primary_event = Uniform(0, 1), interval = 1.0)
    # One conditioned-death and one conditioned-discharge record over the bdbv
    # tree; the per-record CFR is logistic(beta * x), passed into the node.
    specs = [
        (kind = :death, t = 12.0, x = 0.4),
        (kind = :discharge, t = 11.0, x = -0.7)
    ]

    function _row(spec, beta)
        p = _logistic(beta * spec.x)
        bp = (branch_probs = (death = p, discharge = 1 - p),)
        d_ev = spec.kind === :death ? spec.t : missing
        c_ev = spec.kind === :discharge ? spec.t : missing
        return merge(
            (event_1 = 0.0, event_2 = 4.0, death = d_ev, discharge = c_ev,
                event_3 = 9.0), bp)
    end

    # `sh` parameterises the death branch delay; `beta` drives the passed-in
    # per-record prob. The gradient over (sh, beta) flows through both the
    # conditioned tree edges and the per-row-prob path.
    @model function fit(specs)
        sh ~ truncated(Normal(2.0, 0.5); lower = 0.2)
        beta ~ Normal(0.0, 1.0)
        e_oa = double_interval_censored(LogNormal(1.4, 0.4);
            primary_event = Uniform(0, 1), interval = 1.0)
        cmp = Resolve(:death => (Gamma(sh, 3.0), 0.5),
            :discharge => (Gamma(2.0, 1.0), 0.5))
        d = Parallel(Sequential(e_oa, cmp), e_on)
        for i in eachindex(specs)
            y ~ to_submodel(
                composed_distribution_model(d, _row(specs[i], beta)), false)
        end
    end

    ldf = LogDensityFunction(fit(specs); adtype = AutoForwardDiff())
    v, g = LDP.logdensity_and_gradient(ldf, [2.0, 0.8])
    @test length(g) == 2
    @test all(isfinite, g)
    @test any(!iszero, g)
end

@testitem "latent Nested Resolve: AD through sampled admit + per-row prob (#363)" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: latent
    using Turing: Turing, @model, to_submodel, AutoForwardDiff
    using DynamicPPL: DynamicPPL, LogDensityFunction
    import DynamicPPL.LogDensityProblems as LDP

    _logistic(z) = 1 / (1 + exp(-z))
    # Latent-wrapped chain onset -> admit -> {death, discharge}; the admission is
    # SAMPLED per record (a latent), the recorded outcome conditions its branch,
    # and the per-record CFR is logistic(beta * x). The gradient over (sh, beta)
    # flows through the sampled-admit latent edges and the conditioned branch.
    specs = [
        (kind = :death, t = 12.0, x = 0.4),
        (kind = :discharge, t = 11.0, x = -0.7)
    ]

    function _row(spec, beta)
        p = _logistic(beta * spec.x)
        bp = (branch_probs = (death = p, discharge = 1 - p),)
        d_ev = spec.kind === :death ? spec.t : missing
        c_ev = spec.kind === :discharge ? spec.t : missing
        # admit (event_2) is MISSING -> sampled as a latent per record.
        return merge(
            (onset = 0.0, admit = missing, death = d_ev, discharge = c_ev), bp)
    end

    @model function fit(specs)
        sh ~ truncated(Normal(2.0, 0.5); lower = 0.2)
        beta ~ Normal(0.0, 1.0)
        e_oa = double_interval_censored(LogNormal(1.4, 0.4);
            primary_event = Uniform(0, 1), interval = 1.0)
        cmp = Resolve(:death => (Gamma(sh, 3.0), 0.5),
            :discharge => (Gamma(2.0, 1.0), 0.5))
        d = latent(Sequential((e_oa, cmp), (:onset_admit, :admit)))
        for i in eachindex(specs)
            y ~ to_submodel(
                DynamicPPL.prefix(
                    composed_distribution_model(d, _row(specs[i], beta)),
                    Symbol(:rec, i)), false)
        end
    end

    ldf = LogDensityFunction(fit(specs); adtype = AutoForwardDiff())
    # 2 params + 2 sampled admit latents (one per record).
    x0 = [2.0, 0.8, 3.0, 3.0]
    v, g = LDP.logdensity_and_gradient(ldf, x0)
    @test length(g) == 4
    @test all(isfinite, g)
    @test any(!iszero, g)
end

@testitem "Nested Resolve: covariate CFR recovers beta in bdbv tree (#333)" begin
    using CensoredDistributions, Distributions, Random
    using Turing: Turing, NUTS, sample, @model, to_submodel
    using DynamicPPL: prefix

    _logistic(z) = 1 / (1 + exp(-z))

    rng = Random.MersenneTwister(333)
    death_d, disch_d = Gamma(2.0, 1.5), Gamma(2.0, 1.0)
    beta0, beta1 = -0.4, 1.1
    n = 400
    xs = randn(rng, n)
    # Simulate the bdbv tree per record: onset at 0, admit + a delay, then the
    # one_of outcome (death w.p. logistic(Xbeta)), and notif. Only the
    # one_of branch carries the covariate; the rest are fixed delays.
    onset_admit = Gamma(2.0, 1.0)
    onset_notif = Gamma(2.0, 1.5)
    rows = map(1:n) do i
        p = _logistic(beta0 + beta1 * xs[i])
        a = rand(rng, onset_admit)
        nt = rand(rng, onset_notif)
        if rand(rng) < p
            (event_1 = 0.0, event_2 = a, death = a + rand(rng, death_d),
                discharge = missing, event_3 = nt)
        else
            (event_1 = 0.0, event_2 = a, death = missing,
                discharge = a + rand(rng, disch_d), event_3 = nt)
        end
    end

    # The regression is PLAIN TURING; the per-record CFR is PASSED IN to the
    # nested Resolve via the reserved `branch_probs` field.
    @model function fit(rows, xs)
        b0 ~ Normal(0, 1)
        b1 ~ Normal(0, 1)
        e_oa = double_interval_censored(LogNormal(0.0, 0.6);
            primary_event = Uniform(0, 1), interval = 1.0)
        e_on = double_interval_censored(LogNormal(0.4, 0.6);
            primary_event = Uniform(0, 1), interval = 1.0)
        cmp = Resolve(:death => (death_d, 0.5),
            :discharge => (disch_d, 0.5))
        d = Parallel(Sequential(e_oa, cmp), e_on)
        for i in eachindex(rows)
            p = _logistic(b0 + b1 * xs[i])
            row = merge(rows[i], (branch_probs = p,))
            x ~ to_submodel(
                prefix(composed_distribution_model(d, row),
                    Symbol("rec", i)), false)
        end
    end

    chain = sample(rng, fit(rows, xs), NUTS(), 300; progress = false)
    @test mean(chain[:b1]) > 0.5
    @test abs(mean(chain[:b0]) - beta0) < 0.5
    @test abs(mean(chain[:b1]) - beta1) < 0.6
end

@testitem "composer model: weight via reserved field and kwarg" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    seq = Sequential(
        primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))
    ev = Vector{Union{Missing, Float64}}([0.0, 2.0, 5.0])
    base = logpdf(seq, ev)

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))
    @model function demo_w(d, r, w)
        obs ~ to_submodel(composed_distribution_model(d, r; weight = w))
    end

    # Reserved `weight` field scales the contribution; excluded from events.
    row_w = (onset = 0.0, admit = 2.0, death = 5.0, weight = 3)
    @test only(logjoint(demo(seq, row_w), (;))) ≈ 3 * base
    # `count` field likewise.
    row_c = (onset = 0.0, admit = 2.0, death = 5.0, count = 4)
    @test only(logjoint(demo(seq, row_c), (;))) ≈ 4 * base
    # The `weight =` kwarg overrides / supplies the multiplicity.
    row = (onset = 0.0, admit = 2.0, death = 5.0)
    @test only(logjoint(demo_w(seq, row, 6), (;))) ≈ 6 * base
end

@testitem "Nested Resolve: latent-wrapped tree conditions on outcome (#363)" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: latent
    using DynamicPPL: @model, to_submodel, logjoint

    edge(mu,
        sigma) = double_interval_censored(LogNormal(mu, sigma);
        primary_event = Uniform(0, 1), interval = 1.0)
    e_oa = edge(1.4, 0.4)
    cfr = 0.3
    death_d, disch_d = Gamma(2.0, 3.0), Gamma(2.0, 1.0)
    cmp = Resolve(:death => (death_d, cfr), :discharge => (disch_d, 1 - cfr))
    # A latent-wrapped composer with a nested Resolve conditions on the recorded
    # outcome (data), exactly like the marginal Resolve path: the observed admit
    # anchors the conditioned branch (both endpoints observed -> declared edge), so
    # the latent contribution equals the marginal `log p[i] + logpdf(delay[i], gap)`.
    seq = Sequential(e_oa, cmp)
    md = seq
    ld = latent(seq)

    @model demo(dd, r) = obs ~ to_submodel(composed_distribution_model(dd, r))

    # Death observed (admit observed -> declared onset->admit edge + conditioned
    # death branch at its gap from admit).
    death_row = (event_1 = 0.0, event_2 = 4.0, death = 12.0, discharge = missing)
    ref_death = logpdf(e_oa, 4.0) + log(cfr) + logpdf(death_d, 8.0)
    @test only(logjoint(demo(ld, death_row), (;))) ≈ ref_death
    # The latent form matches the marginal form on the same record.
    @test only(logjoint(demo(ld, death_row), (;))) ≈
          only(logjoint(demo(md, death_row), (;)))

    # Discharge observed.
    disch_row = (event_1 = 0.0, event_2 = 4.0, death = missing, discharge = 11.0)
    ref_disch = logpdf(e_oa, 4.0) + log(1 - cfr) + logpdf(disch_d, 7.0)
    @test only(logjoint(demo(ld, disch_row), (;))) ≈ ref_disch
    @test only(logjoint(demo(ld, disch_row), (;))) ≈
          only(logjoint(demo(md, disch_row), (;)))

    # Per-record branch_probs override (covariate CFR) flows in for the latent path
    # exactly as for the marginal.
    bp_row = (event_1 = 0.0, event_2 = 4.0, death = 12.0, discharge = missing,
        branch_probs = (death = 0.6, discharge = 0.4))
    @test only(logjoint(demo(ld, bp_row), (;))) ≈
          logpdf(e_oa, 4.0) + log(0.6) + logpdf(death_d, 8.0)

    # A composer-subtree Resolve outcome is still out of scope for the latent
    # path (its outcome carries internal latents); rejected clearly.
    sub = Resolve(:death => (Sequential(e_oa, edge(1.0, 0.3)), 0.3),
        :discharge => (disch_d, 0.7))
    ld_sub = latent(Sequential(e_oa, sub))
    sub_row = (event_1 = 0.0, event_2 = 4.0, death = missing, discharge = 11.0)
    @test_throws ArgumentError logjoint(demo(ld_sub, sub_row), (;))
end

@testitem "composer model: varying-data batch (mixed missing patterns)" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint, prefix

    seq = Sequential(
        primary_censored(LogNormal(1.2, 0.4), Uniform(0, 1)),
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))

    # Each row a different missingness pattern and multiplicity; the batch log
    # density is the sum of per-row censored composer logpdfs with weights.
    rows = [
        (onset = 0.0, admit = 2.0, death = 5.0, weight = 2),
        (onset = 0.0, admit = missing, death = 6.0),
        (onset = 0.5, admit = 3.0, death = missing, count = 4)
    ]

    @model function fit(rows)
        for i in eachindex(rows)
            x ~ to_submodel(
                prefix(composed_distribution_model(seq, rows[i]),
                    Symbol("rec", i)), false)
        end
    end

    ev(r) = Vector{Union{Missing, Float64}}([r.onset, r.admit, r.death])
    manual = 2 * logpdf(seq, ev(rows[1])) + logpdf(seq, ev(rows[2])) +
             4 * logpdf(seq, ev(rows[3]))
    @test logjoint(fit(rows), (;)) ≈ manual
end

@testitem "latent batch model matches the manual per-record loop" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, prefix, logjoint, VarInfo, condition,
                      @varname

    seq = Sequential(
        (primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
            primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))),
        (:onset_admit, :admit_death))
    lseq = latent(seq)

    rows = [(onset = 0.3, admit = missing, death = 5.4),
        (onset = 1.0, admit = 3.2, death = 6.1),
        (onset = 0.7, admit = missing, death = 4.9)]

    # The manual per-record loop (hand-prefixed), wrapped in one
    # `obs ~ to_submodel(...)` so its varnames carry the same `obs.` namespace
    # the batch entry adds.
    @model function manual_loop(d, rs)
        for i in eachindex(rs)
            inner ~ to_submodel(
                prefix(composed_distribution_model(d, rs[i]), Symbol(:rec, i)),
                false)
        end
    end
    @model function manual(d, rs)
        obs ~ to_submodel(manual_loop(d, rs))
    end

    # The batch entry collapses the loop + prefix into one tilde.
    @model function batch(d, rs)
        obs ~ to_submodel(composed_distribution_model(d, rs))
    end

    # Same VarInfo: one latent per record's missing admit (records 1 and 3),
    # none for the fully-observed record 2 -> two latents, identically named.
    vman = VarInfo(manual(lseq, rows))
    vbat = VarInfo(batch(lseq, rows))
    @test length(keys(vman)) == 2
    @test Set(string.(keys(vman))) == Set(string.(keys(vbat)))

    # The batch model's logjoint equals the manual loop on the same conditioned
    # latents (the two missing admits).
    a1, a3 = 2.1, 1.9
    cman = condition(manual(lseq, rows),
        (@varname(obs.rec1.e[2]) => a1, @varname(obs.rec3.e[2]) => a3))
    cbat = condition(batch(lseq, rows),
        (@varname(obs.rec1.e[2]) => a1, @varname(obs.rec3.e[2]) => a3))
    @test logjoint(cman, (;)) ≈ logjoint(cbat, (;))
end

@testitem "latent batch model accepts a Tables.jl table" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint
    import Tables

    seq = Sequential(
        (primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
            primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))),
        (:onset_admit, :admit_death))
    lseq = latent(seq)

    rows = [(onset = 0.3, admit = 2.1, death = 5.4),
        (onset = 1.0, admit = 3.2, death = 6.1)]
    tbl = Tables.columntable(rows)
    @test Tables.istable(tbl)
    @test !(tbl isa AbstractVector)

    @model demo_vec(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))
    @model demo_tbl(d, t) = obs ~ to_submodel(composed_distribution_model(d, t))

    # All events observed -> no latents, so logjoint is well-defined and the
    # table path matches the vector-of-rows path.
    @test only(logjoint(demo_tbl(lseq, tbl), (;))) ≈
          only(logjoint(demo_vec(lseq, rows), (;)))
end

@testitem "latent batch model rejects an empty table and obs_time" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, VarInfo

    seq = Sequential(
        (primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
            primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))),
        (:onset_admit, :admit_death))
    lseq = latent(seq)

    @test_throws ArgumentError composed_distribution_model(
        lseq, NamedTuple[])

    # obs_time horizon is rejected under latent, per-record and so per-batch.
    @model bad(d, rs) = obs ~ to_submodel(composed_distribution_model(d, rs))
    rows = [(onset = 0.3, admit = 2.1, death = 5.4, obs_time = 7.0)]
    @test_throws Exception VarInfo(bad(lseq, rows))
end

@testitem "composer model: latent Sequential samples internal events inside" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint, VarInfo, condition,
                      @varname

    seq = Sequential(
        primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))
    lseq = latent(seq)

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    # A fully-missing row turns every internal event into a sampled latent that
    # lives in the VarInfo (one per event, never passed in).
    row_miss = (onset = missing, admit = missing, death = missing)
    vi = VarInfo(demo(lseq, row_miss))
    @test length(keys(vi)) == 3
    @test all(vn -> occursin("e", string(vn)), keys(vi))

    # Conditioning on every event time, the log-density decomposes into the
    # origin primary prior plus each edge's conditional at the realised gap, scored
    # on the edge's DECLARED censoring (matching the marginal chain).
    o, a, dd = 0.3, 2.1, 5.4
    cond = condition(demo(lseq, (onset = o, admit = a, death = dd)),
        (@varname(obs.e[1]) => o, @varname(obs.e[2]) => a,
            @varname(obs.e[3]) => dd))
    manual = logpdf(get_primary_event(seq.components[1]), o) +
             logpdf(seq.components[1], a - o) +
             logpdf(seq.components[2], dd - a)
    @test logjoint(cond, (;)) ≈ manual
end

@testitem "composer model: latent model generates the full event path" begin
    using CensoredDistributions, Distributions, Random
    using DynamicPPL: @model, to_submodel

    seq = Sequential(
        primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))
    lseq = latent(seq)

    # With every event missing, calling the latent submodel samples the FULL
    # event path `[E_0, E_1, E_2]`, which is monotone increasing (each event is
    # the previous plus a positive delay).
    @model function gen(d)
        inner = to_submodel(
            composed_distribution_model(d, (a = missing, b = missing, c = missing)),
            false)
        path ~ inner
        return path
    end

    Random.seed!(11)
    draws = [gen(lseq)() for _ in 1:300]
    @test all(p -> p[1] >= 0 && p[2] >= p[1] && p[3] >= p[2], draws)
end

@testitem "composer model: latent Parallel shares one origin" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint, VarInfo, condition,
                      @varname

    par = Parallel(
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)),
        primary_censored(LogNormal(1.0, 0.5), Uniform(0, 1)))
    lpar = latent(par)

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    # Origin plus two branches: three latents (shared origin + two branch
    # observations) when all missing.
    vi = VarInfo(demo(lpar, (origin = missing, y1 = missing, y2 = missing)))
    @test length(keys(vi)) == 3

    # Conditioning: prior(origin) + branch conditionals at y_i - origin.
    o, y1, y2 = 0.3, 2.5, 3.1
    cond = condition(demo(lpar, (origin = o, y1 = y1, y2 = y2)),
        (@varname(obs.e[1]) => o, @varname(obs.e[2]) => y1,
            @varname(obs.e[3]) => y2))
    manual = logpdf(get_primary_event(par.components[1]), o) +
             logpdf(get_dist(par.components[1]), y1 - o) +
             logpdf(get_dist(par.components[2]), y2 - o)
    @test logjoint(cond, (;)) ≈ manual
end

@testitem "composer model: latent twin recurses through a nested tree" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint, VarInfo, condition,
                      @varname

    # A nested tree onset -> {admit -> {b1, b2 off admit}}: the inner Parallel
    # branches share the ADMIT event (E_1) as their origin, not the root onset.
    # The latent twin must recurse so each branch hangs off the right event.
    oa = primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1))
    b1 = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))
    b2 = primary_censored(LogNormal(0.8, 0.4), Uniform(0, 1))
    seq = Sequential(oa, Parallel(b1, b2))
    lseq = latent(seq)

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    # The flat event vector is [E_0, E_1(admit), E_2(b1), E_3(b2)]: four events,
    # so a fully-missing row samples four latents.
    vi = VarInfo(demo(lseq, (a = missing, b = missing, c = missing,
        d = missing)))
    @test length(keys(vi)) == 4

    # Conditioning on the full path: origin prior + the onset->admit edge at
    # (E_1 - E_0) + each inner branch at (E_i - E_1), the SHARED admit event.
    e0, e1, e2, e3 = 0.0, 2.0, 5.0, 4.5
    cond = condition(demo(lseq, (a = e0, b = e1, c = e2, d = e3)),
        (@varname(obs.e[1]) => e0, @varname(obs.e[2]) => e1,
            @varname(obs.e[3]) => e2, @varname(obs.e[4]) => e3))
    manual = logpdf(get_primary_event(oa), e0) +
             logpdf(oa, e1 - e0) +
             logpdf(b1, e2 - e1) +
             logpdf(b2, e3 - e1)
    @test logjoint(cond, (;)) ≈ manual

    # The conditioned latent log-density equals the marginal: a NESTED Parallel
    # conditions each branch on its declared edge (the marginal `_tree_score`), so
    # the latent twin keeps the same edges and the two agree even with censoring.
    row = (a = e0, b = e1, c = e2, d = e3)
    marg = only(logjoint(demo(seq, row), (;)))
    @test logjoint(cond, (;)) ≈ marg rtol=1e-10
end

@testitem "composer model: marginal == latent for interval-censored edges" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint, condition, @varname

    # An observed edge carrying SECONDARY interval censoring must score the same
    # in the marginal and the latent forms: both condition the observed gap on the
    # edge's DECLARED censoring. The latent form must not strip the censoring down
    # to the continuous core (which would silently mis-score the edge).
    seq = Sequential(
        primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
        double_interval_censored(LogNormal(1.5, 0.75); interval = 1))
    lseq = latent(seq)

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    o, a, dd = 0.3, 2.0, 7.0
    row = (onset = o, admit = a, death = dd)
    marg = only(logjoint(demo(seq, row), (;)))
    cond = condition(demo(lseq, row),
        (@varname(obs.e[1]) => o, @varname(obs.e[2]) => a,
            @varname(obs.e[3]) => dd))
    lat = logjoint(cond, (;))
    @test marg ≈ lat rtol=1e-10

    # The conditioned latent log-density also equals the explicit decomposition:
    # origin prior + each edge's DECLARED-censoring logpdf at the realised gap.
    manual = logpdf(get_primary_event(seq.components[1]), o) +
             logpdf(seq.components[1], a - o) +
             logpdf(seq.components[2], dd - a)
    @test lat ≈ manual rtol=1e-10
end

@testitem "composer model: latent dic chain initialises in-support (#423)" begin
    using CensoredDistributions, Distributions, Random
    using DynamicPPL: @model, to_submodel, logjoint, condition, @varname

    # A `double_interval_censored` chain in LATENT form samples a CONTINUOUS
    # origin per record, but the observed downstream events are DISCRETISED
    # (floored to the interval). When a record's first downstream event lands in
    # the SAME interval as the origin (an observed gap of 0), the origin -> first
    # event edge must stay in-support for any continuous origin in
    # `[0, interval)`, so the latent joint is finite and NUTS finds a valid init.
    # Previously the edge scored `logpdf(edge, admit - origin)` with a continuous
    # origin, giving a negative gap (out of support, `-Inf`) for any origin above
    # the floored event.
    dic(d) = double_interval_censored(d; primary_event = Uniform(0, 1),
        interval = 1.0)
    seq = Sequential((dic(Gamma(2.0, 1.5)), dic(Gamma(1.5, 2.0))),
        (:onset_admit, :admit_death))
    lseq = latent(seq)

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    # admit floors into the SAME interval as the (continuous) origin: gap is 0.
    row = (onset = 0.0, admit = 0.0, death = 5.0)

    # Conditioning the observed events, the joint must be finite for a RANGE of
    # continuous origins in `[0, 1)`, not only the boundary `origin = 0`.
    for o in (0.0, 0.25, 0.5, 0.9)
        cond = condition(demo(lseq, row),
            (@varname(obs.e[1]) => o, @varname(obs.e[2]) => 0.0,
                @varname(obs.e[3]) => 5.0))
        @test isfinite(logjoint(cond, (;)))
    end

    # The discretised gap matches the marginal with the floored origin observed:
    # the latent origin's sub-interval position does not change the scored gap.
    cond = condition(demo(lseq, row),
        (@varname(obs.e[1]) => 0.4, @varname(obs.e[2]) => 0.0,
            @varname(obs.e[3]) => 5.0))
    lat = logjoint(cond, (;))
    origin = CensoredDistributions._origin_primary_event(seq.components[1])
    manual = logpdf(origin, 0.4) +
             logpdf(seq.components[1], 0.0) +
             logpdf(seq.components[2], 5.0)
    @test lat ≈ manual rtol = 1e-10
end

@testitem "composer model: latent dic chain NUTS inits (#423)" tags = [:turing] begin
    using CensoredDistributions, Distributions, Random
    using DynamicPPL, Turing
    using FlexiChains: VNChain

    # End-to-end: a latent fit of a `double_interval_censored` chain whose data
    # includes a same-interval record must find valid initial parameters and
    # sample (it failed to initialise under NUTS before #423).
    dic(d) = double_interval_censored(d; primary_event = Uniform(0, 1),
        interval = 1.0)
    template = Sequential((dic(Gamma(2.0, 1.5)), dic(Gamma(1.5, 2.0))),
        (:onset_admit, :admit_death))
    priors = build_priors(params_table(template))

    @model function latent_fit(t, p, rows)
        delays ~ to_submodel(composed_parameters_model(t, p))
        ld = latent(delays)
        for i in eachindex(rows)
            obs ~ to_submodel(
                DynamicPPL.prefix(
                    composed_distribution_model(ld, rows[i]), Symbol(:rec, i)),
                false)
        end
    end

    # Include a same-interval record (onset == admit == 0) that triggered the
    # init failure, plus a few ordinary ones.
    rows = [(onset = 0.0, admit = 0.0, death = 5.0),
        (onset = 0.0, admit = 2.0, death = 3.0),
        (onset = 0.0, admit = 1.0, death = 4.0)]

    chain = sample(Xoshiro(1), latent_fit(template, priors, rows),
        NUTS(0.8; adtype = AutoForwardDiff()), 20;
        chain_type = VNChain, progress = false)
    @test chain isa VNChain
end

@testitem "composer model: marginal == latent for a Parallel interval edge" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint, condition, @varname

    # The latent Parallel form must likewise score each observed branch on its
    # DECLARED censoring, matching the marginal Parallel logpdf.
    par = Parallel(
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)),
        double_interval_censored(LogNormal(1.0, 0.5); interval = 1))
    lpar = latent(par)

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    o, y1, y2 = 0.3, 2.5, 4.0
    row = (origin = o, y1 = y1, y2 = y2)
    marg = only(logjoint(demo(par, row), (;)))
    cond = condition(demo(lpar, row),
        (@varname(obs.e[1]) => o, @varname(obs.e[2]) => y1,
            @varname(obs.e[3]) => y2))
    lat = logjoint(cond, (;))
    @test marg ≈ lat rtol=1e-10
end

@testitem "composer model: latent rejects a per-record obs_time horizon" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, VarInfo

    # An `obs_time` horizon is not supported under the latent form; reject it
    # clearly rather than silently dropping it.
    seq = Sequential(
        primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))
    lseq = latent(seq)

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    row = (onset = 0.3, admit = 2.0, death = 5.0, obs_time = 10.0)
    @test_throws ArgumentError VarInfo(demo(lseq, row))

    # A latent leaf row carrying an obs_time is likewise rejected.
    lleaf = latent(primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1)))
    leaf_row = (delay = 2.0, obs_time = 10.0)
    @test_throws ArgumentError VarInfo(demo(lleaf, leaf_row))
end

@testitem "composer model: marginal Sequential short NUTS run" begin
    using CensoredDistributions, Distributions, Random
    using Turing: Turing, NUTS, sample, @model, to_submodel
    using DynamicPPL: prefix

    rng = Random.MersenneTwister(21)
    truth = Sequential(
        primary_censored(LogNormal(1.2, 0.4), Uniform(0, 1)),
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))
    rows = map(1:25) do _
        p = rand(rng, truth)
        (onset = p[1], admit = p[2], death = p[3])
    end

    @model function fit(rows)
        mu ~ Normal(1.2, 0.5)
        sh ~ truncated(Normal(2.0, 0.5); lower = 0.2)
        seq = Sequential(
            primary_censored(LogNormal(mu, 0.4), Uniform(0, 1)),
            primary_censored(Gamma(sh, 1.0), Uniform(0, 1)))
        for i in eachindex(rows)
            x ~ to_submodel(
                prefix(composed_distribution_model(seq, rows[i]),
                    Symbol("rec", i)), false)
        end
    end

    chain = sample(rng, fit(rows), NUTS(), 60; progress = false)
    @test size(chain, 1) == 60
    @test all(isfinite, chain[:mu])
end

@testitem "composer model: latent Sequential short NUTS run" begin
    using CensoredDistributions, Distributions, Random
    using Turing: Turing, NUTS, sample, @model, to_submodel
    using DynamicPPL: prefix

    rng = Random.MersenneTwister(22)
    truth = Sequential(
        primary_censored(LogNormal(1.2, 0.4), Uniform(0, 1)),
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))
    # Observe origin and death, leave the intermediate missing: the sampler then
    # explores `mu` and a latent intermediate event time per record.
    rows = map(1:12) do _
        p = rand(rng, truth)
        (onset = p[1], admit = missing, death = p[3])
    end

    @model function fit(rows)
        mu ~ Normal(1.2, 0.5)
        d = latent(Sequential(
            primary_censored(LogNormal(mu, 0.4), Uniform(0, 1)),
            primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))))
        for i in eachindex(rows)
            x ~ to_submodel(
                prefix(composed_distribution_model(d, rows[i]),
                    Symbol("rec", i)), false)
        end
    end

    chain = sample(rng, fit(rows), NUTS(), 50; progress = false)
    @test size(chain, 1) == 50
    @test all(isfinite, chain[:mu])
end

@testitem "composer model: a linelist NamedTuple row passes straight in" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    seq = Sequential(
        primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))

    # A linelist row IS a NamedTuple (event columns plus a reserved `count`), so
    # each iterated row drops straight into the model (no glue). The second
    # row's missing intermediate marginalises automatically.
    rows = [
        (onset = 0.0, admit = 2.0, death = 5.0, count = 2),
        (onset = 0.5, admit = missing, death = 6.0, count = 3)
    ]

    @model function fit(rows)
        for r in rows
            x ~ to_submodel(composed_distribution_model(seq, r), false)
        end
    end

    ev(r) = Vector{Union{Missing, Float64}}([r.onset, r.admit, r.death])
    manual = 2 * logpdf(seq, ev(rows[1])) + 3 * logpdf(seq, ev(rows[2]))
    @test logjoint(fit(rows), (;)) ≈ manual
end

@testitem "by-name row: reordered row scores identically (#362)" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    # A NAMED chain: edge names `:onset_admit`, `:admit_death` derive the EVENT
    # names `(:onset, :admit, :death)` (origin + targets), so a row keys by event.
    seq = Sequential(
        (primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
            primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))),
        (:onset_admit, :admit_death))
    @test event_names(seq) ==
          (:onset, :admit, :death)

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    in_order = (onset = 0.0, admit = 2.0, death = 5.0)
    shuffled = (death = 5.0, onset = 0.0, admit = 2.0)
    ev = Vector{Union{Missing, Float64}}([0.0, 2.0, 5.0])
    base = only(logjoint(demo(seq, in_order), (;)))
    # The reordered row maps each field to its event slot BY NAME, so it scores
    # identically to the in-order row and to the direct logpdf.
    @test only(logjoint(demo(seq, shuffled), (;))) ≈ base ≈ logpdf(seq, ev)

    # A reserved weight still rides along regardless of position.
    w_first = (weight = 3, death = 5.0, onset = 0.0, admit = 2.0)
    @test only(logjoint(demo(seq, w_first), (;))) ≈ 3 * base
end

@testitem "by-name row: mixed missingness factorises by name (#362)" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    seq = Sequential(
        (primary_censored(LogNormal(1.2, 0.4), Uniform(0, 1)),
            primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))),
        (:onset_admit, :admit_death))

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    # A missing intermediate marginalises; the by-name mapping places `missing`
    # in the admit slot regardless of field order.
    row = (death = 6.0, admit = missing, onset = 0.0)
    ev = Vector{Union{Missing, Float64}}([0.0, missing, 6.0])
    @test only(logjoint(demo(seq, row), (;))) ≈ logpdf(seq, ev)
end

@testitem "by-name row: name mismatch and missing event error (#362)" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    seq = Sequential(
        (primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
            primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))),
        (:onset_admit, :admit_death))

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    # A field the tree has no event for is rejected.
    bad = (onset = 0.0, admit = 2.0, demise = 5.0)
    @test_throws ArgumentError logjoint(demo(seq, bad), (;))

    # A missing REQUIRED event name (no `death` field at all) is rejected.
    short = (onset = 0.0, admit = 2.0)
    @test_throws ArgumentError logjoint(demo(seq, short), (;))
end

@testitem "by-name row: nested tree keys by name at every depth (#362)" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    # onset -> admit, then admit -> {death, discharge} off the SHARED admit event.
    oa = primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1))
    ad = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))
    adisch = primary_censored(LogNormal(0.8, 0.4), Uniform(0, 1))
    inner = Parallel((ad, adisch), (:admit_death, :admit_discharge))
    seq = Sequential((oa, inner), (:onset_admit, :admit_resolution))

    # Events: onset (E_0), admit (E_1), death (E_2), discharge (E_3), the inner
    # parallel sharing the admit event as its origin.
    @test event_names(seq) ==
          (:onset, :admit, :death, :discharge)

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    ev = Vector{Union{Missing, Float64}}([0.0, 2.0, 5.0, 4.5])
    in_order = (onset = 0.0, admit = 2.0, death = 5.0, discharge = 4.5)
    shuffled = (discharge = 4.5, death = 5.0, admit = 2.0, onset = 0.0)
    base = only(logjoint(demo(seq, in_order), (;)))
    @test base ≈ logpdf(seq, ev)
    @test only(logjoint(demo(seq, shuffled), (;))) ≈ base
end

@testitem "by-name row: positional default names fall back positionally (#362)" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    # A positionally-constructed chain has `:step_i` edges -> positional `:event_i`
    # events, so a row is matched POSITIONALLY (the documented fallback).
    seq = Sequential(
        primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))
    @test event_names(seq) ==
          (:event_1, :event_2, :event_3)

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    row = (onset = 0.0, admit = 2.0, death = 5.0)
    ev = Vector{Union{Missing, Float64}}([0.0, 2.0, 5.0])
    @test only(logjoint(demo(seq, row), (;))) ≈ logpdf(seq, ev)
end

@testitem "per-record horizon: whole-compose truncation at D (#329)" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    seq = Sequential(
        (primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
            primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))),
        (:onset_admit, :admit_death))

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    D = 8.0
    # Endpoint-observed record (intermediate admit unobserved): the whole compose
    # collapses to the origin->death total, right-truncated at D - origin. The
    # log-density includes the -logcdf(total, window) correction vs untruncated.
    ev = Vector{Union{Missing, Float64}}([0.0, missing, 5.0])
    seg = CensoredDistributions._sequential_segment(
        seq.components, 1, 3, Uniform(0, 1))
    expected = logpdf(seg, 5.0) - logcdf(seg, D - 0.0)
    row = (onset = 0.0, admit = missing, death = 5.0, obs_time = D)
    @test only(logjoint(demo(seq, row), (;))) ≈ expected
    # The correction is exactly the right-truncation term.
    @test only(logjoint(demo(seq, row), (;))) ≈ logpdf(seq, ev) - logcdf(seg, D)

    # A non-zero observed origin shifts the window to D - origin.
    o = 1.5
    seg2 = CensoredDistributions._sequential_segment(
        seq.components, 1, 3, Uniform(0, 1))
    row2 = (onset = o, admit = missing, death = 5.0, obs_time = D)
    @test only(logjoint(demo(seq, row2), (;))) ≈
          logpdf(seg2, 5.0 - o) - logcdf(seg2, D - o)

    # No obs_time field -> no truncation (back-compat).
    row3 = (onset = 0.0, admit = missing, death = 5.0)
    @test only(logjoint(demo(seq, row3), (;))) ≈ logpdf(seq, ev)
end

@testitem "per-record horizon: index single vs sourced convolved (#329)" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    # An index case observes a single delay onset->resolution; a sourced case
    # observes the total over two unobserved-intermediate segments. The whole-
    # compose denominator is the single delay vs the convolution, selected by the
    # record's missingness, exactly the andv split.
    inc = primary_censored(LogNormal(1.5, 0.5), Uniform(0, 1))
    delta = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))
    seq = Sequential((inc, delta), (:onset_mid, :mid_obs))
    D = 6.0

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    # Sourced: intermediate `mid` unobserved -> convolved origin->obs denominator.
    sourced = CensoredDistributions._sequential_segment(
        seq.components, 1, 3, Uniform(0, 1))
    row_src = (onset = 0.0, mid = missing, obs = 5.0, obs_time = D)
    @test only(logjoint(demo(seq, row_src), (;))) ≈
          logpdf(sourced, 5.0) - logcdf(sourced, D)

    # Index: a single-segment chain (one delay), endpoint observed -> single
    # delay denominator.
    seq1 = Sequential((inc,), (:onset_obs,))
    single = CensoredDistributions._sequential_segment(
        seq1.components, 1, 2, Uniform(0, 1))
    row_idx = (onset = 0.0, obs = 4.0, obs_time = D)
    @test only(logjoint(demo(seq1, row_idx), (;))) ≈
          logpdf(single, 4.0) - logcdf(single, D)
end

@testitem "per-record horizon: non-positive window guard (#329)" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    seq = Sequential(
        (primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
            primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))),
        (:onset_admit, :admit_death))

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    # Horizon before the observed total -> empty-support truncation -> -Inf, no
    # error or NaN (the primitive guards the non-positive window).
    row = (onset = 2.0, admit = missing, death = 5.0, obs_time = 1.0)
    lj = only(logjoint(demo(seq, row), (;)))
    @test lj == -Inf
end

@testitem "per-record horizon: observed-intermediate whole-compose (#366)" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    inc = primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1))
    delta = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))
    seq = Sequential((inc, delta), (:onset_admit, :admit_death))

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    # With the intermediate admit OBSERVED and a per-record horizon, #366 scores
    # whole-compose TOTAL truncation: the factorised per-segment numerator over a
    # single conv-to-last-observed denominator. Match the hand-rolled per-record
    # decomposition (the andv index-vs-sourced terms generalised to an observed
    # intermediate).
    D = 8.0
    o, a, dth = 0.0, 2.0, 5.0
    # Numerator: each observed segment conditions on its own edge.
    numerator = logpdf(inc, a - o) + logpdf(delta, dth - a)
    # Denominator: the convolution of all components origin->last observed
    # (death), truncated at D - origin.
    conv = CensoredDistributions._sequential_segment(
        seq.components, 1, 3, Uniform(0, 1))
    expected = numerator - logcdf(conv, D - o)
    row = (onset = o, admit = a, death = dth, obs_time = D)
    @test only(logjoint(demo(seq, row), (;))) ≈ expected

    # The whole-compose denominator is conv-to-LAST-observed, NOT a per-segment
    # product of each edge's own window: the two genuinely differ.
    per_segment = logcdf(inc, D - o) + logcdf(delta, D - a)
    @test !isapprox(logcdf(conv, D - o), per_segment)
end

@testitem "per-record horizon: leaf record truncates at D (#329)" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    # A leaf record (observed from the origin) truncates at the horizon itself.
    pc = primary_censored(LogNormal(1.5, 0.5), Uniform(0, 1))
    row = (delay = 2.0, obs_time = 6.0)
    @test only(logjoint(demo(pc, row), (;))) ≈
          logpdf(pc, 2.0) - logcdf(pc, 6.0)
    # No obs_time -> untruncated (and keeps the named primary-censored routing).
    @test only(logjoint(demo(pc, (delay = 2.0,)), (;))) ≈ logpdf(pc, 2.0)
end

@testitem "per-record horizon: per-record loop with differing D (#329)" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint, prefix

    inc = primary_censored(LogNormal(1.5, 0.5), Uniform(0, 1))
    delta = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))
    seq = Sequential((inc, delta), (:onset_mid, :mid_obs))

    # Each record carries its OWN observation horizon; the batch log density sums
    # the per-record whole-compose truncated contributions.
    rows = [
        (onset = 0.0, mid = missing, obs = 4.0, obs_time = 6.0),
        (onset = 0.5, mid = missing, obs = 5.0, obs_time = 9.0, weight = 2)
    ]

    @model function fit(rows)
        for i in eachindex(rows)
            x ~ to_submodel(
                prefix(composed_distribution_model(seq, rows[i]),
                    Symbol("rec", i)), false)
        end
    end

    seg = CensoredDistributions._sequential_segment(
        seq.components, 1, 3, Uniform(0, 1))
    m1 = logpdf(seg, 4.0 - 0.0) - logcdf(seg, 6.0 - 0.0)
    m2 = logpdf(seg, 5.0 - 0.5) - logcdf(seg, 9.0 - 0.5)
    @test logjoint(fit(rows), (;)) ≈ m1 + 2 * m2
end

@testitem "per-record horizon: AD through the truncated path (#329)" begin
    using CensoredDistributions, Distributions
    using Turing: Turing, @model, to_submodel, AutoForwardDiff
    using DynamicPPL: DynamicPPL, LogDensityFunction
    import DynamicPPL.LogDensityProblems as LDP

    rows = [
        (onset = 0.0, mid = missing, obs = 4.0, obs_time = 6.0),
        (onset = 0.5, mid = missing, obs = 5.0, obs_time = 9.0)
    ]

    @model function fit(rows)
        mu ~ Normal(1.5, 0.3)
        sh ~ truncated(Normal(2.0, 0.5); lower = 0.2)
        seq = Sequential(
            (primary_censored(LogNormal(mu, 0.5), Uniform(0, 1)),
                primary_censored(Gamma(sh, 1.0), Uniform(0, 1))),
            (:onset_mid, :mid_obs))
        for i in eachindex(rows)
            x ~ to_submodel(composed_distribution_model(seq, rows[i]), false)
        end
    end

    ldf = LogDensityFunction(fit(rows); adtype = AutoForwardDiff())
    v, g = LDP.logdensity_and_gradient(ldf, [1.5, 2.0])
    @test length(g) == 2
    @test all(isfinite, g)
    @test any(!iszero, g)
end

@testitem "composed_distribution_model: Choose routes by the data selector" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    # An index case (its own short origin) vs a sourced case (a longer coupled
    # delay), selected by the row's `:kind` field (#356).
    idx = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))
    src = primary_censored(Gamma(4.0, 1.5), Uniform(0, 1))
    d = choose(:index => idx, :sourced => src)

    @model demo(dd, r) = obs ~ to_submodel(composed_distribution_model(dd, r))

    # The selector value picks WHICH alternative scores the record.
    lj_idx = only(logjoint(demo(d, (kind = :index, y = 3.0)), (;)))
    lj_src = only(logjoint(demo(d, (kind = :sourced, y = 3.0)), (;)))
    @test lj_idx ≈ logpdf(idx, 3.0)
    @test lj_src ≈ logpdf(src, 3.0)
    @test lj_idx != lj_src

    # A reserved weight on the row scales the selected alternative's likelihood.
    lj_w = only(logjoint(demo(d, (kind = :index, y = 3.0, weight = 2.5)), (;)))
    @test lj_w ≈ 2.5 * logpdf(idx, 3.0)
end

@testitem "composed_distribution_model: Choose alternative may be a composer" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    # The sourced case is a whole chain; the index case is a single leaf. The
    # data routes the record to the selected alternative's full handling.
    chain = Sequential(
        primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))
    leaf = primary_censored(Gamma(3.0, 1.0), Uniform(0, 1))
    d = choose(:sourced => chain, :index => leaf)

    @model demo(dd, r) = obs ~ to_submodel(composed_distribution_model(dd, r))

    row_chain = (kind = :sourced, onset = 0.0, admit = 2.0, death = 5.0)
    ev = Vector{Union{Missing, Float64}}([0.0, 2.0, 5.0])
    @test only(logjoint(demo(d, row_chain), (;))) ≈ logpdf(chain, ev)

    @test only(logjoint(demo(d, (kind = :index, y = 4.0)), (;))) ≈
          logpdf(leaf, 4.0)
end

@testitem "composed_distribution_model: Choose branches keep independent anchors" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    # The hanta index-vs-sourced split (#323, #356): two ALTERNATIVE WHOLE
    # records with INDEPENDENT anchors, NOT shared-origin branches. The index
    # case is its own primary-censored leaf; the sourced case is a longer,
    # independently-anchored chain. Unlike a shared-origin Parallel, Choose makes
    # no shared-primary assumption: each selected branch scores as its OWN
    # complete distribution, and the non-selected branch contributes nothing.
    index_rec = primary_censored(LogNormal(1.5, 0.5), Uniform(0, 1))
    sourced_rec = Sequential(
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)),
        primary_censored(Gamma(3.0, 1.0), Uniform(0, 1)))
    d = choose(:index => index_rec, :sourced => sourced_rec)

    @model demo(dd, r) = obs ~ to_submodel(composed_distribution_model(dd, r))

    # Choosing the index branch scores EXACTLY its own standalone leaf logpdf
    # (no shared-origin path, no contribution from the sourced branch).
    @test only(logjoint(demo(d, (kind = :index, y = 3.0)), (;))) ≈
          logpdf(index_rec, 3.0)

    # Choosing the sourced branch scores EXACTLY its own standalone chain
    # logpdf, routed through the same nested-composer recursion.
    ev = Vector{Union{Missing, Float64}}([0.0, 2.0, 5.0])
    @test only(logjoint(
        demo(d, (kind = :sourced, onset = 0.0, admit = 2.0, death = 5.0)),
        (;))) ≈ logpdf(sourced_rec, ev)
end

@testitem "composed_distribution_model: Choose selector field is non-event" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    # A custom selector name, and the selector field must hold a Symbol.
    idx = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))
    src = primary_censored(Gamma(4.0, 1.5), Uniform(0, 1))
    d = choose(:index => idx, :sourced => src; selector = :case)

    @model demo(dd, r) = obs ~ to_submodel(composed_distribution_model(dd, r))

    # The selector field is stripped before delegating, so the leaf sees only
    # its single event value.
    @test only(logjoint(demo(d, (case = :sourced, y = 2.0)), (;))) ≈
          logpdf(src, 2.0)

    # A non-Symbol selector value is rejected.
    @model bad(dd, r) = obs ~ to_submodel(composed_distribution_model(dd, r))
    @test_throws Exception logjoint(bad(d, (case = 1, y = 2.0)), (;))
end
