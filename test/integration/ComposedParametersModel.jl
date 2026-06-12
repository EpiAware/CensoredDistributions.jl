# Tests for `composed_parameters_model` (#353): the priors -> Turing-submodel
# helper that samples a composed distribution's parameters from user priors and
# returns the reconstructed distribution, ready to score.

@testitem "composed_parameters_model: round-trip structure + names" tags=[:turing] begin
    using CensoredDistributions, Distributions, DynamicPPL, Random

    template = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    priors = (
        onset_admit = (shape = truncated(Normal(2, 0.5); lower = 0),
            scale = truncated(Normal(1, 0.3); lower = 0)),
        admit_death = (mu = Normal(0.5, 0.2),
            sigma = truncated(Normal(0.4, 0.1); lower = 0)))

    @model function pm(t, p)
        d ~ to_submodel(composed_parameters_model(t, p))
        return d
    end

    Random.seed!(101)
    m = pm(template, priors)
    d = m()

    # The reconstructed distribution has the template's structure and names.
    @test d isa CensoredDistributions.Parallel
    @test event_names(d) == event_names(template)
    @test keys(params(d)) == keys(params(template))
    @test event(d, :onset_admit) isa Gamma
    @test event(d, :admit_death) isa LogNormal

    # Sampled parameter varnames carry the edge path (Option A prefixing).
    vns = Set(string.(collect(keys(VarInfo(m)))))
    @test "d.onset_admit.shape" in vns
    @test "d.onset_admit.scale" in vns
    @test "d.admit_death.mu" in vns
    @test "d.admit_death.sigma" in vns
end

@testitem "composed_parameters_model: bare leaf template" tags=[:turing] begin
    using CensoredDistributions, Distributions, DynamicPPL, Random

    @model function pm(t, p)
        d ~ to_submodel(composed_parameters_model(t, p))
        return d
    end

    Random.seed!(102)
    m = pm(Gamma(2.0, 1.0),
        (shape = truncated(Normal(2, 0.5); lower = 0),
            scale = truncated(Normal(1, 0.3); lower = 0)))
    d = m()
    @test d isa Gamma
    vns = Set(string.(collect(keys(VarInfo(m)))))
    @test "d.shape" in vns
    @test "d.scale" in vns
end

@testitem "composed_parameters_model: nested Sequential + Competing" tags=[:turing] begin
    using CensoredDistributions, Distributions, DynamicPPL, Random

    template = compose((
        chain = [Gamma(2.0, 1.0), LogNormal(0.5, 0.4)],
        resolution = Competing(:death => (Gamma(1.5, 1.0), 0.3),
            :disch => (Gamma(2.0, 1.5), 0.7))))
    priors = (
        chain = (
            step_1 = (shape = truncated(Normal(2, 0.5); lower = 0),
                scale = truncated(Normal(1, 0.3); lower = 0)),
            step_2 = (mu = Normal(0.5, 0.2),
                sigma = truncated(Normal(0.4, 0.1); lower = 0))),
        resolution = (
            death = (shape = truncated(Normal(1.5, 0.3); lower = 0),
                scale = truncated(Normal(1, 0.2); lower = 0)),
            disch = (shape = truncated(Normal(2, 0.3); lower = 0),
                scale = truncated(Normal(1.5, 0.2); lower = 0)))
    )

    @model function pm(t, p)
        d ~ to_submodel(composed_parameters_model(t, p))
        return d
    end

    Random.seed!(103)
    m = pm(template, priors)
    d = m()

    @test event_names(d) == event_names(template)
    @test keys(params(d)) == keys(params(template))
    # Competing branch probabilities kept fixed from the template by default.
    @test event(d, :resolution).branch_probs == (0.3, 0.7)

    vns = Set(string.(collect(keys(VarInfo(m)))))
    @test "d.chain.step_1.shape" in vns
    @test "d.chain.step_2.sigma" in vns
    @test "d.resolution.death.scale" in vns
    @test "d.resolution.disch.shape" in vns
end

@testitem "composed_parameters_model: Competing branch_probs prior" tags=[:turing] begin
    using CensoredDistributions, Distributions, DynamicPPL, Random

    template = compose((resolution = Competing(
        :death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.7)),))
    priors = (resolution = (
        death = (shape = truncated(Normal(1.5, 0.3); lower = 0),
            scale = truncated(Normal(1, 0.2); lower = 0)),
        disch = (shape = truncated(Normal(2, 0.3); lower = 0),
            scale = truncated(Normal(1.5, 0.2); lower = 0)),
        branch_probs = (death = Uniform(0, 1), disch = Uniform(0, 1))),)

    @model function pm(t, p)
        d ~ to_submodel(composed_parameters_model(t, p))
        return d
    end

    Random.seed!(104)
    m = pm(template, priors)
    d = m()
    bp = event(d, :resolution).branch_probs
    @test length(bp) == 2
    @test all(0 .<= bp .<= 1)

    vns = Set(string.(collect(keys(VarInfo(m)))))
    @test "d.resolution.branch_probs.death" in vns
    @test "d.resolution.branch_probs.disch" in vns
end

@testitem "composed_parameters_model: missing/extra prior errors" tags=[:turing] begin
    using CensoredDistributions, Distributions, DynamicPPL

    template = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))

    @model function pm(t, p)
        d ~ to_submodel(composed_parameters_model(t, p))
        return d
    end

    full = (
        onset_admit = (shape = Normal(2, 0.5), scale = Normal(1, 0.3)),
        admit_death = (mu = Normal(0.5, 0.2),
            sigma = truncated(Normal(0.4, 0.1); lower = 0)))

    # Missing a leaf parameter.
    miss_param = merge(full, (; onset_admit = (shape = Normal(2, 0.5),)))
    @test_throws ArgumentError pm(template, miss_param)()

    # Extra leaf parameter.
    extra_param = merge(full,
        (;
            onset_admit = (shape = Normal(2, 0.5),
            scale = Normal(1, 0.3), bogus = Normal(0, 1))))
    @test_throws ArgumentError pm(template, extra_param)()

    # Missing an edge.
    miss_edge = (; onset_admit = full.onset_admit)
    @test_throws ArgumentError pm(template, miss_edge)()

    # Extra edge.
    extra_edge = merge(full, (; ghost = (mu = Normal(0, 1), sigma = Normal(1, 1))))
    @test_throws ArgumentError pm(template, extra_edge)()
end

@testitem "composed_parameters_model: AD gradient through full loop" tags=[:turing] begin
    using CensoredDistributions, Distributions, DynamicPPL, Turing, Random
    const LDP = DynamicPPL.LogDensityProblems

    # AD must flow through prior-sample -> reconstruct -> logpdf. Build the
    # log-density function with ForwardDiff and check a finite, non-zero gradient
    # over the whole sampled-and-reconstructed loop.
    template = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    priors = (
        onset_admit = (shape = truncated(Normal(2, 0.5); lower = 0),
            scale = truncated(Normal(1, 0.3); lower = 0)),
        admit_death = (mu = Normal(0.5, 0.2),
            sigma = truncated(Normal(0.4, 0.1); lower = 0)))

    @model function fit(t, p, ys)
        d ~ to_submodel(composed_parameters_model(t, p))
        for y in ys
            DynamicPPL.@addlogprob! logpdf(d, y)
        end
    end

    Random.seed!(106)
    ys = [[0.5, 2.0], [1.0, 3.0]]
    m = fit(template, priors, ys)
    # Link to the unconstrained space and start from a valid prior draw so the
    # gradient is taken at an in-support point of the full sampled-reconstructed
    # loop.
    vi = DynamicPPL.link(VarInfo(m), m)
    ldf = DynamicPPL.LogDensityFunction(
        m, DynamicPPL.getlogjoint_internal, vi; adtype = AutoForwardDiff())
    x0 = vi[:]
    lp, grad = LDP.logdensity_and_gradient(ldf, x0)

    @test isfinite(lp)
    @test length(grad) == 4
    @test all(isfinite, grad)
    @test any(!iszero, grad)
end

@testitem "composed_parameters_model: censored leaf round-trips" tags=[:turing] begin
    using CensoredDistributions, Distributions, DynamicPPL, Random

    # A template carrying a censored leaf directly: the params layer shows only
    # the inner delay's free params and the model rebuilds the SAME censored
    # leaf (no doc-side re-censoring needed).
    cens = double_interval_censored(Gamma(2.0, 1.5);
        primary_event = Uniform(0, 1), upper = 10.0, interval = 1.0)
    template = compose((obs = cens,))
    priors = (obs = (shape = truncated(Normal(2, 0.5); lower = 0),
        scale = truncated(Normal(1.5, 0.3); lower = 0)),)

    @model function pm(t, p)
        d ~ to_submodel(composed_parameters_model(t, p))
        return d
    end

    Random.seed!(110)
    d = pm(template, priors)()
    leaf = event(d, :obs)
    # The rebuilt leaf is the censored type, not the bare delay.
    @test typeof(leaf) == typeof(cens)

    # At the template's parameters the rebuilt leaf scores exactly the original
    # censored leaf logpdf (the fixed censoring carried through).
    exact = update(template, (obs = (shape = 2.0, scale = 1.5),))
    exact_leaf = event(exact, :obs)
    for x in (1.0, 2.0, 3.0, 5.0)
        @test isapprox(logpdf(exact_leaf, x), logpdf(cens, x); atol = 1e-10)
    end
end

@testitem "build_priors drives composed_parameters_model" tags=[:turing] begin
    using CensoredDistributions, Distributions, DynamicPPL, Random

    template = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    # Build the nested prior NamedTuple from the flat table.
    priors = build_priors(params_table(template);
        default = row -> truncated(Normal(row.value, 1); lower = 0))

    @model function pm(t, p)
        d ~ to_submodel(composed_parameters_model(t, p))
        return d
    end

    Random.seed!(111)
    d = pm(template, priors)()
    @test event_names(d) == event_names(template)
    @test keys(params(d)) == keys(params(template))
end

@testitem "chain_to_params + update reconstruct from a chain" tags=[:turing] begin
    using CensoredDistributions, Distributions, DynamicPPL, Turing, Random
    using FlexiChains: Prefixed, VNChain
    import Statistics

    template = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    priors = (
        onset_admit = (shape = truncated(Normal(2, 0.5); lower = 0),
            scale = truncated(Normal(1, 0.3); lower = 0)),
        admit_death = (mu = Normal(0.5, 0.2),
            sigma = truncated(Normal(0.4, 0.1); lower = 0)))

    @model function fit(t, p, ys)
        d ~ to_submodel(composed_parameters_model(t, p))
        for y in ys
            DynamicPPL.@addlogprob! logpdf(d, y)
        end
        return d
    end

    Random.seed!(112)
    ys = [[0.5, 2.0], [1.0, 3.0], [0.8, 2.5]]
    chain = sample(fit(template, priors, ys), NUTS(), 60;
        chain_type = VNChain, progress = false)

    # Posterior means read into the nested NamedTuple `update` consumes.
    means = chain_to_params(template, chain)
    @test Set(keys(means)) == Set(keys(params(template)))
    @test Set(keys(means.onset_admit)) == Set((:shape, :scale))

    ready = update(template, means)
    @test ready isa CensoredDistributions.Parallel
    # Matches a hand-rebuild from the same chain means.
    sh = Statistics.mean(chain[Prefixed(@varname(onset_admit.shape))])
    sc = Statistics.mean(chain[Prefixed(@varname(onset_admit.scale))])
    @test event(ready, :onset_admit) == Gamma(sh, sc)

    # A single draw reads that iteration's value.
    one = chain_to_params(template, chain; draw = 5)
    @test isapprox(one.onset_admit.shape,
        chain[Prefixed(@varname(onset_admit.shape))][5]; atol = 1e-12)

    # The reconstructed distribution is ready to sample.
    Random.seed!(113)
    @test length(rand(ready)) == 2
end

@testitem "chain_to_params: configurable summary and draws subset" tags=[:turing] begin
    using CensoredDistributions, Distributions, DynamicPPL, Turing, Random
    using FlexiChains: Prefixed, VNChain
    import Statistics

    template = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    priors = (
        onset_admit = (shape = truncated(Normal(2, 0.5); lower = 0),
            scale = truncated(Normal(1, 0.3); lower = 0)),
        admit_death = (mu = Normal(0.5, 0.2),
            sigma = truncated(Normal(0.4, 0.1); lower = 0)))

    @model function fit(t, p, ys)
        d ~ to_submodel(composed_parameters_model(t, p))
        for y in ys
            DynamicPPL.@addlogprob! logpdf(d, y)
        end
        return d
    end

    Random.seed!(112)
    ys = [[0.5, 2.0], [1.0, 3.0], [0.8, 2.5]]
    chain = sample(fit(template, priors, ys), NUTS(), 60;
        chain_type = VNChain, progress = false)

    col = vec(chain[Prefixed(@varname(onset_admit.shape))])

    # A non-mean summary reduces each parameter's draws with the supplied
    # reduction (here the median).
    med = chain_to_params(template, chain; summary = Statistics.median)
    @test isapprox(med.onset_admit.shape, Statistics.median(col); atol = 1e-12)
    # A custom reduction (a high quantile) works too.
    q90 = chain_to_params(template, chain;
        summary = x -> Statistics.quantile(x, 0.9))
    @test isapprox(q90.onset_admit.shape,
        Statistics.quantile(col, 0.9); atol = 1e-12)

    # A `draws` subset reduces over only the selected iterations: a range, and a
    # predicate over the iteration index (a warmup-drop / thinning).
    sub = chain_to_params(template, chain; draws = 30:60)
    @test isapprox(sub.onset_admit.shape,
        Statistics.mean(col[30:60]); atol = 1e-12)
    keep = chain_to_params(template, chain; draws = i -> i > 30)
    idx = [i for i in 1:length(col) if i > 30]
    @test isapprox(keep.onset_admit.shape,
        Statistics.mean(col[idx]); atol = 1e-12)

    # The `update` chain overload threads the same kwargs.
    upd = update(template, chain; summary = Statistics.median)
    @test event(upd, :onset_admit) ==
          Gamma(Statistics.median(col),
        Statistics.median(vec(chain[Prefixed(@varname(onset_admit.scale))])))
end

@testitem "chain_to_params skips latent per-record event vectors" tags=[:turing] begin
    using CensoredDistributions, Distributions, DynamicPPL, Turing, Random
    using FlexiChains: VNChain

    # A latent fit also samples per-record event vectors, named `recN.e`. These
    # are vector-valued varnames the template walk never requests; reading their
    # mean would error, so `chain_to_params`/`update` must skip them.
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

    truth = update(
        template, (onset_admit = (shape = 2.0, scale = 1.5),
            admit_death = (shape = 1.5, scale = 2.0)))
    rng = MersenneTwister(909)
    rows = map(1:6) do _
        rand(rng, truth)
    end

    chain = sample(Xoshiro(1), latent_fit(template, priors, rows),
        NUTS(0.8), 30; chain_type = VNChain, progress = false)

    # The latent event vectors are present in the chain but must not break the
    # read; only the `delays.*` scalars are reconstructed.
    means = chain_to_params(template, chain; prefix = :delays)
    @test Set(keys(means)) == Set(keys(params(template)))
    fit = update(template, chain; prefix = :delays)
    @test all(>(0), mean(fit))
    # A single draw works the same.
    @test update(template, chain; prefix = :delays, draw = 5) isa
          CensoredDistributions.Sequential
end

@testitem "latent batch entry: one-tilde fit matches the manual loop" tags=[:turing] begin
    using CensoredDistributions, Distributions, DynamicPPL, Turing, Random
    using FlexiChains: VNChain

    # The batch latent entry collapses the per-record loop + prefix into one
    # tilde, symmetric with the marginal batch entry. It must produce the same
    # `recN.e` varnames and fit through the same update/edge_means read.
    pc(d) = primary_censored(d, Uniform(0, 1))
    template = Sequential((pc(Gamma(2.0, 1.5)), pc(Gamma(1.5, 2.0))),
        (:onset_admit, :admit_death))
    priors = build_priors(params_table(template))

    @model function marginal_fit(t, p, rows)
        delays ~ to_submodel(composed_parameters_model(t, p))
        obs ~ to_submodel(composed_distribution_model(delays, rows))
    end
    @model function latent_fit(t, p, rows)
        delays ~ to_submodel(composed_parameters_model(t, p))
        obs ~ to_submodel(composed_distribution_model(latent(delays), rows))
    end

    truth = update(
        template, (onset_admit = (shape = 2.0, scale = 1.5),
            admit_death = (shape = 1.5, scale = 2.0)))
    rng = MersenneTwister(909)
    rows = map(1:8) do _
        rand(rng, truth)
    end

    mchain = sample(Xoshiro(1), marginal_fit(template, priors, rows),
        NUTS(0.8), 25; chain_type = VNChain, progress = false)
    lchain = sample(Xoshiro(1), latent_fit(template, priors, rows),
        NUTS(0.8), 25; chain_type = VNChain, progress = false)

    # Both fits run and read back finite, positive edge means via the same
    # update/mean path the per-record loop uses; the latent fit's per-record
    # `recN.e` event vectors are skipped by the template read.
    mfit = update(template, mchain; prefix = :delays)
    lfit = update(template, lchain; prefix = :delays)
    @test all(>(0), mean(mfit))
    @test all(>(0), mean(lfit))
end

@testitem "update(template, chain) == update via chain_to_params" tags=[:turing] begin
    using CensoredDistributions, Distributions, DynamicPPL, Turing, Random
    using FlexiChains: VNChain

    template = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    priors = build_priors(params_table(template))

    @model function fit(t, p, ys)
        d ~ to_submodel(composed_parameters_model(t, p))
        for y in ys
            DynamicPPL.@addlogprob! logpdf(d, y)
        end
        return d
    end

    Random.seed!(212)
    ys = [[0.5, 2.0], [1.0, 3.0], [0.8, 2.5]]
    chain = sample(fit(template, priors, ys), NUTS(), 60;
        chain_type = VNChain, progress = false)

    # The chain overload matches threading chain_to_params by hand.
    @test update(template, chain) ==
          update(template, chain_to_params(template, chain))
    # A single draw too.
    @test update(template, chain; draw = 5) ==
          update(template, chain_to_params(template, chain; draw = 5))
    # Result is the right structure and ready to sample.
    ready = update(template, chain)
    @test ready isa CensoredDistributions.Parallel
    @test event_names(ready) == event_names(template)
end

@testitem "composed_distribution_model accepts a Tables.jl table" tags=[:turing] begin
    using CensoredDistributions, Distributions, DynamicPPL
    import Tables

    seq = Sequential(
        primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))
    rows = [(onset = 0.0, admit = 2.0, death = 5.0),
        (onset = 1.0, admit = missing, death = 7.0)]
    # A column table is a Tables.jl source (what a DataFrame is, structurally),
    # not a single NamedTuple row nor a Vector of rows.
    tbl = Tables.columntable(rows)
    @test Tables.istable(tbl)
    @test !(tbl isa AbstractVector)

    @model demo_tbl(d, t) = obs ~ to_submodel(composed_distribution_model(d, t))
    @model demo_vec(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    # The table path matches the per-record (vector) path exactly.
    @test only(logjoint(demo_tbl(seq, tbl), (;))) ≈
          only(logjoint(demo_vec(seq, rows), (;)))

    # A non-table second argument errors clearly.
    @test_throws ArgumentError composed_distribution_model(seq, 3)
end

@testitem "composed_parameters_model: shared param sampled once, placed everywhere" tags=[:turing] begin
    using CensoredDistributions, Distributions, DynamicPPL, Random

    # `inc` is shared across the index and sourced branches of a select: one
    # free parameter, sampled once, placed in both occurrences.
    inc = shared(:inc, Gamma(2.0, 1.0))
    template = selecting(:index => inc,
        :sourced => compose((delta = LogNormal(0.5, 0.4), inc = inc)))
    priors = (
        inc = (shape = truncated(Normal(2, 0.5); lower = 0),
            scale = truncated(Normal(1, 0.3); lower = 0)),
        sourced = (delta = (mu = Normal(0.5, 0.2),
            sigma = truncated(Normal(0.4, 0.1); lower = 0)),))

    @model function pm(t, p)
        d ~ to_submodel(composed_parameters_model(t, p))
        return d
    end

    Random.seed!(120)
    m = pm(template, priors)
    d = m()

    idx = CensoredDistributions._pick(d, :index)
    src = CensoredDistributions._pick(d, :sourced)
    # The SAME inc value flows to both occurrences.
    @test params(get_dist(idx)) == params(get_dist(event(src, :inc)))

    # The shared group is sampled ONCE (under its tag), not per occurrence; the
    # sourced delta is its own free parameter.
    vns = Set(string.(collect(keys(VarInfo(m)))))
    @test "d.inc.shape" in vns
    @test "d.inc.scale" in vns
    @test "d.sourced.delta.mu" in vns
    @test "d.sourced.delta.sigma" in vns
    # No per-occurrence inc names.
    @test !("d.sourced.inc.shape" in vns)
    @test !("d.index.inc.shape" in vns)

    # A missing shared prior errors.
    bad = (sourced = priors.sourced,)
    @test_throws ArgumentError pm(template, bad)()
end

@testitem "composed_parameters_model: shared param AD + NUTS recovery" tags=[:turing] begin
    using CensoredDistributions, Distributions, DynamicPPL, Turing, Random
    using FlexiChains: Prefixed, VNChain
    const LDP = DynamicPPL.LogDensityProblems
    import Statistics

    inc = shared(:inc, Gamma(2.0, 1.0))
    template = selecting(:index => inc,
        :sourced => compose((delta = LogNormal(0.5, 0.4), inc = inc)))
    priors = (
        inc = (shape = truncated(Normal(3, 1); lower = 0),
            scale = truncated(Normal(1.5, 0.5); lower = 0)),
        sourced = (delta = (mu = Normal(0.7, 0.3),
            sigma = truncated(Normal(0.5, 0.2); lower = 0)),))

    @model function fit(t, p, idx_obs, src_obs)
        d ~ to_submodel(composed_parameters_model(t, p))
        idx = CensoredDistributions._pick(d, :index)
        src = CensoredDistributions._pick(d, :sourced)
        for y in idx_obs
            DynamicPPL.@addlogprob! logpdf(idx, y)
        end
        for y in src_obs
            DynamicPPL.@addlogprob! logpdf(src, y)
        end
    end

    Random.seed!(121)
    true_inc = Gamma(3.0, 1.5)
    idx_obs = rand(true_inc, 150)
    src_obs = [[rand(LogNormal(0.7, 0.5)), rand(true_inc)] for _ in 1:150]
    m = fit(template, priors, idx_obs, src_obs)

    # AD flows through the shared sample -> both occurrences -> logpdf. The free
    # dimension is the deduped set (inc.shape, inc.scale, delta.mu, delta.sigma).
    vi = DynamicPPL.link(VarInfo(m), m)
    ldf = DynamicPPL.LogDensityFunction(
        m, DynamicPPL.getlogjoint_internal, vi; adtype = AutoForwardDiff())
    lp, grad = LDP.logdensity_and_gradient(ldf, vi[:])
    @test isfinite(lp)
    @test length(grad) == 4
    @test all(isfinite, grad)
    @test any(!iszero, grad)

    # NUTS recovers the shared inc from both branches' data.
    chain = sample(m, NUTS(), 300; chain_type = VNChain, progress = false)
    sh = Statistics.mean(chain[Prefixed(@varname(inc.shape))])
    sc = Statistics.mean(chain[Prefixed(@varname(inc.scale))])
    @test isapprox(sh * sc, 4.5; atol = 1.0)  # true mean = 3.0 * 1.5
end

@testitem "composed_parameters_model: NUTS samples the full loop" tags=[:turing] begin
    using CensoredDistributions, Distributions, DynamicPPL, Turing, Random
    using FlexiChains: Prefixed, niters

    template = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    priors = (
        onset_admit = (shape = truncated(Normal(2, 0.5); lower = 0),
            scale = truncated(Normal(1, 0.3); lower = 0)),
        admit_death = (mu = Normal(0.5, 0.2),
            sigma = truncated(Normal(0.4, 0.1); lower = 0)))

    # Sample the priors, reconstruct, and score step-value vectors directly
    # through the composer logpdf (a self-contained likelihood).
    @model function fit(t, p, ys)
        d ~ to_submodel(composed_parameters_model(t, p))
        for y in ys
            DynamicPPL.@addlogprob! logpdf(d, y)
        end
        return d
    end

    Random.seed!(105)
    ys = [[0.5, 2.0], [1.0, 3.0], [0.8, 2.5]]
    chain = sample(
        fit(template, priors, ys), NUTS(), 50;
        chain_type = VNChain, progress = false)

    @test niters(chain) == 50
    # The chain carries the edge-path-prefixed parameter names.
    shape = chain[Prefixed(@varname(onset_admit.shape))]
    sigma = chain[Prefixed(@varname(admit_death.sigma))]
    @test all(isfinite.(shape))
    @test all(sigma .> 0)
end

@testitem "chain_to_params + update reconstruct a Select template" tags=[:turing] begin
    using CensoredDistributions, Distributions, DynamicPPL, Turing, Random
    using FlexiChains: Prefixed, VNChain
    import Statistics

    # A `Select` with one leaf alternative and one nested-composer alternative.
    # Each alternative carries its own free parameters under the select path, so
    # the chain bridge must walk the alternatives like the core
    # `params`/`update`.
    template = selecting(:index => Gamma(2.0, 1.0),
        :sourced => compose((delta = LogNormal(0.5, 0.4),
            inc = Gamma(2.0, 1.0))))
    priors = (
        index = (shape = truncated(Normal(2, 0.5); lower = 0),
            scale = truncated(Normal(1, 0.3); lower = 0)),
        sourced = (
            delta = (mu = Normal(0.5, 0.2),
                sigma = truncated(Normal(0.4, 0.1); lower = 0)),
            inc = (shape = truncated(Normal(2, 0.5); lower = 0),
                scale = truncated(Normal(1, 0.3); lower = 0))))

    @model function fit(t, p, idx_obs, src_obs)
        d ~ to_submodel(composed_parameters_model(t, p))
        idx = CensoredDistributions._pick(d, :index)
        src = CensoredDistributions._pick(d, :sourced)
        for y in idx_obs
            DynamicPPL.@addlogprob! logpdf(idx, y)
        end
        for y in src_obs
            DynamicPPL.@addlogprob! logpdf(src, y)
        end
        return d
    end

    Random.seed!(130)
    idx_obs = rand(Gamma(2.0, 1.0), 30)
    src_obs = [[rand(LogNormal(0.5, 0.4)), rand(Gamma(2.0, 1.0))]
               for _ in 1:30]
    chain = sample(fit(template, priors, idx_obs, src_obs), NUTS(), 60;
        chain_type = VNChain, progress = false)

    # Posterior means read into the nested NamedTuple `update` consumes; the
    # keys mirror `params(template)` (every alternative walked).
    means = chain_to_params(template, chain)
    @test Set(keys(means)) == Set(event_names(template))
    @test Set(keys(means.sourced)) == Set((:delta, :inc))

    ready = update(template, means)
    @test ready isa CensoredDistributions.Select
    # Each alternative's leaf matches a hand-rebuild from the same chain means.
    ish = Statistics.mean(chain[Prefixed(@varname(index.shape))])
    isc = Statistics.mean(chain[Prefixed(@varname(index.scale))])
    @test CensoredDistributions._pick(ready, :index) == Gamma(ish, isc)
    src = CensoredDistributions._pick(ready, :sourced)
    dmu = Statistics.mean(chain[Prefixed(@varname(sourced.delta.mu))])
    dsig = Statistics.mean(chain[Prefixed(@varname(sourced.delta.sigma))])
    @test event(src, :delta) == LogNormal(dmu, dsig)

    # A single draw reads that iteration's value.
    one = chain_to_params(template, chain; draw = 5)
    @test isapprox(one.index.shape,
        chain[Prefixed(@varname(index.shape))][5]; atol = 1e-12)

    # The chain overload matches threading chain_to_params by hand.
    @test update(template, chain) ==
          update(template, chain_to_params(template, chain))
end

@testitem "chain_to_params + update reconstruct a shared-tagged template" tags=[:turing] begin
    using CensoredDistributions, Distributions, DynamicPPL, Turing, Random
    using FlexiChains: Prefixed, VNChain
    import Statistics

    # `inc` is shared across the index and sourced branches: the chain carries
    # the deduped `d.inc.shape`/`d.inc.scale`, not the per-occurrence paths, so
    # the bridge must read the tag ONCE and place it in both occurrences.
    inc = shared(:inc, Gamma(2.0, 1.0))
    template = selecting(:index => inc,
        :sourced => compose((delta = LogNormal(0.5, 0.4), inc = inc)))
    priors = (
        inc = (shape = truncated(Normal(3, 1); lower = 0),
            scale = truncated(Normal(1.5, 0.5); lower = 0)),
        sourced = (delta = (mu = Normal(0.7, 0.3),
            sigma = truncated(Normal(0.5, 0.2); lower = 0)),))

    @model function fit(t, p, idx_obs, src_obs)
        d ~ to_submodel(composed_parameters_model(t, p))
        idx = CensoredDistributions._pick(d, :index)
        src = CensoredDistributions._pick(d, :sourced)
        for y in idx_obs
            DynamicPPL.@addlogprob! logpdf(idx, y)
        end
        for y in src_obs
            DynamicPPL.@addlogprob! logpdf(src, y)
        end
        return d
    end

    Random.seed!(131)
    idx_obs = rand(Gamma(3.0, 1.5), 30)
    src_obs = [[rand(LogNormal(0.7, 0.5)), rand(Gamma(3.0, 1.5))]
               for _ in 1:30]
    chain = sample(fit(template, priors, idx_obs, src_obs), NUTS(), 60;
        chain_type = VNChain, progress = false)

    # The shared tag is inventoried ONCE under its tag, so the nested NamedTuple
    # carries a top-level `inc` entry (matching `params_table`'s tag edge), not
    # a per-occurrence entry under `index`/`sourced`.
    means = chain_to_params(template, chain)
    @test haskey(means, :inc)
    @test Set(keys(means.inc)) == Set((:shape, :scale))

    ready = update(template, means)
    @test ready isa CensoredDistributions.Select
    # The SAME inc value flows to both occurrences, read from `d.inc.*`.
    sh = Statistics.mean(chain[Prefixed(@varname(inc.shape))])
    sc = Statistics.mean(chain[Prefixed(@varname(inc.scale))])
    idx = CensoredDistributions._pick(ready, :index)
    src = CensoredDistributions._pick(ready, :sourced)
    @test get_dist(idx) == Gamma(sh, sc)
    @test get_dist(event(src, :inc)) == Gamma(sh, sc)

    # The chain overload matches threading chain_to_params by hand.
    @test update(template, chain) ==
          update(template, chain_to_params(template, chain))
end
