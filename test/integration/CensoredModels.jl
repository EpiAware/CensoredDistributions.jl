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

@testitem "composer model: Competing marginal == mixture logpdf" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    cmp = Competing(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.7))

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    for y in (1.0, 4.0, 8.0)
        @test only(logjoint(demo(cmp, (resolve = y,)), (;))) ≈ logpdf(cmp, y)
    end
    # `count` reserved field scales the likelihood.
    @test only(logjoint(demo(cmp, (resolve = 4.0, count = 5)), (;))) ≈
          5 * logpdf(cmp, 4.0)
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
    # origin primary prior plus each edge's conditional at the realised gap.
    o, a, dd = 0.3, 2.1, 5.4
    cond = condition(demo(lseq, (onset = o, admit = a, death = dd)),
        (@varname(obs.e[1]) => o, @varname(obs.e[2]) => a,
            @varname(obs.e[3]) => dd))
    manual = logpdf(get_primary_event(seq.components[1]), o) +
             logpdf(get_dist(seq.components[1]), a - o) +
             logpdf(get_dist(seq.components[2]), dd - a)
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
             logpdf(get_dist(oa), e1 - e0) +
             logpdf(get_dist(b1), e2 - e1) +
             logpdf(get_dist(b2), e3 - e1)
    @test logjoint(cond, (;)) ≈ manual
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
    @test CensoredDistributions.tree_event_names(seq) ==
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
    @test CensoredDistributions.tree_event_names(seq) ==
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
    @test CensoredDistributions.tree_event_names(seq) ==
          (:event_1, :event_2, :event_3)

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    row = (onset = 0.0, admit = 2.0, death = 5.0)
    ev = Vector{Union{Missing, Float64}}([0.0, 2.0, 5.0])
    @test only(logjoint(demo(seq, row), (;))) ≈ logpdf(seq, ev)
end

@testitem "composed_distribution_model: Select routes by the data selector" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    # An index case (its own short origin) vs a sourced case (a longer coupled
    # delay), selected by the row's `:kind` field (#356).
    idx = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))
    src = primary_censored(Gamma(4.0, 1.5), Uniform(0, 1))
    d = select(:index => idx, :sourced => src)

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

@testitem "composed_distribution_model: Select alternative may be a composer" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    # The sourced case is a whole chain; the index case is a single leaf. The
    # data routes the record to the selected alternative's full handling.
    chain = Sequential(
        primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))
    leaf = primary_censored(Gamma(3.0, 1.0), Uniform(0, 1))
    d = select(:sourced => chain, :index => leaf)

    @model demo(dd, r) = obs ~ to_submodel(composed_distribution_model(dd, r))

    row_chain = (kind = :sourced, onset = 0.0, admit = 2.0, death = 5.0)
    ev = Vector{Union{Missing, Float64}}([0.0, 2.0, 5.0])
    @test only(logjoint(demo(d, row_chain), (;))) ≈ logpdf(chain, ev)

    @test only(logjoint(demo(d, (kind = :index, y = 4.0)), (;))) ≈
          logpdf(leaf, 4.0)
end

@testitem "composed_distribution_model: Select branches keep independent anchors" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    # The hanta index-vs-sourced split (#323, #356): two ALTERNATIVE WHOLE
    # records with INDEPENDENT anchors, NOT shared-origin branches. The index
    # case is its own primary-censored leaf; the sourced case is a longer,
    # independently-anchored chain. Unlike a shared-origin Parallel, Select makes
    # no shared-primary assumption: each selected branch scores as its OWN
    # complete distribution, and the non-selected branch contributes nothing.
    index_rec = primary_censored(LogNormal(1.5, 0.5), Uniform(0, 1))
    sourced_rec = Sequential(
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)),
        primary_censored(Gamma(3.0, 1.0), Uniform(0, 1)))
    d = select(:index => index_rec, :sourced => sourced_rec)

    @model demo(dd, r) = obs ~ to_submodel(composed_distribution_model(dd, r))

    # Selecting the index branch scores EXACTLY its own standalone leaf logpdf
    # (no shared-origin path, no contribution from the sourced branch).
    @test only(logjoint(demo(d, (kind = :index, y = 3.0)), (;))) ≈
          logpdf(index_rec, 3.0)

    # Selecting the sourced branch scores EXACTLY its own standalone chain
    # logpdf, routed through the same nested-composer recursion.
    ev = Vector{Union{Missing, Float64}}([0.0, 2.0, 5.0])
    @test only(logjoint(
        demo(d, (kind = :sourced, onset = 0.0, admit = 2.0, death = 5.0)),
        (;))) ≈ logpdf(sourced_rec, ev)
end

@testitem "composed_distribution_model: Select selector field is non-event" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    # A custom selector name, and the selector field must hold a Symbol.
    idx = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))
    src = primary_censored(Gamma(4.0, 1.5), Uniform(0, 1))
    d = select(:index => idx, :sourced => src; selector = :case)

    @model demo(dd, r) = obs ~ to_submodel(composed_distribution_model(dd, r))

    # The selector field is stripped before delegating, so the leaf sees only
    # its single event value.
    @test only(logjoint(demo(d, (case = :sourced, y = 2.0)), (;))) ≈
          logpdf(src, 2.0)

    # A non-Symbol selector value is rejected.
    @model bad(dd, r) = obs ~ to_submodel(composed_distribution_model(dd, r))
    @test_throws Exception logjoint(bad(d, (case = 1, y = 2.0)), (;))
end
