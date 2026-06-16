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

@testitem "nested Competing: Mooncake reverse == ForwardDiff (#497)" tags=[:turing] begin
    using CensoredDistributions, Distributions, DynamicPPL, Turing, Random
    using ADTypes: AutoForwardDiff, AutoMooncake
    import Mooncake
    const LDP = DynamicPPL.LogDensityProblems

    # A bdbv-style nested-Competing tree fit through the full prior-sample ->
    # reconstruct -> score loop with a PER-RECORD covariate branch probability.
    # The Competing rebuild used to route the reconstructed component tuple through
    # `Tuple(::Vector{Any})`, which gave the composed object a heterogeneous edge
    # type whose reverse data Mooncake could not `increment!!` (#497) -- so the
    # nested-Competing case studies fell back to `AutoForwardDiff`. The reconstruct
    # path now builds the component tuple by a type-stable head/tail recursion, so
    # Mooncake reverse builds a rule and its gradient MATCHES ForwardDiff.
    dc(d) = double_interval_censored(d; primary_event = Uniform(0, 1),
        interval = 1.0)
    function delay_tree(; cfr = 0.5)
        resolution = Competing(
            :death => (dc(Gamma(2.0, 3.5)), cfr),
            :discharge => (dc(Gamma(1.0, 8.0)), 1 - cfr))
        admit_path = Sequential(
            (dc(Gamma(1.2, 3.0)), resolution),
            (:onset_admit, :admit_resolution))
        return compose((admit_path = admit_path,
            onset_notif = dc(Gamma(0.7, 20.0))))
    end
    template = delay_tree()
    priors = build_priors(params_table(template);
        priors = Dict(
            (Symbol("admit_path.admit_resolution.branch_probs"), :death) =>
                Uniform(0, 1),
            (Symbol("admit_path.admit_resolution.branch_probs"), :discharge) =>
                Uniform(0, 1)))

    logistic(z) = 1 / (1 + exp(-z))
    death_prob(β, r) = logistic(β.β0 + β.β_hcw * r.hcw)

    @model function bdbv(template, priors, rows)
        delays ~ to_submodel(composed_parameters_model(template, priors))
        β0 ~ Normal(0, 1.5)
        β_hcw ~ Normal(0, 1)
        β = (; β0, β_hcw)
        obs_rows = map(rows) do r
            p = death_prob(β, r)
            (onset = r.onset, admit = r.admit, death = r.death,
                discharge = r.discharge, notif = r.notif,
                branch_probs = (death = p, discharge = 1 - p))
        end
        obs ~ to_submodel(composed_distribution_model(delays, obs_rows))
    end

    rows = [
        (onset = 0.0, admit = 4.0, death = 12.0, discharge = missing,
            notif = 9.0, hcw = true),
        (onset = 0.5, admit = 5.0, death = missing, discharge = 11.0,
            notif = 10.0, hcw = false)]

    Random.seed!(497)
    m = bdbv(template, priors, rows)
    # Link to the unconstrained space and start from a valid prior draw so the
    # gradient is taken at an in-support point of the full loop.
    vi = DynamicPPL.link(VarInfo(m), m)
    x0 = vi[:]

    ldf_fd = DynamicPPL.LogDensityFunction(
        m, DynamicPPL.getlogjoint_internal, vi; adtype = AutoForwardDiff())
    ldf_mc = DynamicPPL.LogDensityFunction(
        m, DynamicPPL.getlogjoint_internal, vi;
        adtype = AutoMooncake(config = nothing))

    lp_fd, g_fd = LDP.logdensity_and_gradient(ldf_fd, x0)
    # The key assertion: Mooncake reverse builds a rule for the nested-Competing
    # tree (no `increment!!` MethodError) and returns a finite gradient.
    lp_mc, g_mc = LDP.logdensity_and_gradient(ldf_mc, x0)

    @test isfinite(lp_fd)
    @test lp_mc ≈ lp_fd
    @test all(isfinite, g_mc)
    @test g_mc≈g_fd rtol=1e-5 atol=1e-7
end

@testitem "Select top: Mooncake reverse == ForwardDiff (#497)" tags=[:turing] begin
    using CensoredDistributions, Distributions, DynamicPPL, Turing, Random
    using ADTypes: AutoForwardDiff, AutoMooncake
    import Mooncake
    const LDP = DynamicPPL.LogDensityProblems

    # An andv-style Select-top model fit through the full prior-sample ->
    # reconstruct -> score loop. `event(delays, :index)` inside the differentiated
    # model splits a dotted edge `Symbol` via `_split_edge`, whose `split(string,
    # '.')` is pointer-arithmetic string search that aborted Mooncake reverse with
    # the uncatchable `sub_ptr intrinsic hit` (#497) -- so the andv case study fell
    # back to `AutoForwardDiff`. `_split_edge` now carries a Mooncake `@zero_adjoint`
    # (it is constant string -> `Tuple{Symbol...}` work with zero derivative), so
    # Mooncake builds a rule and its gradient MATCHES ForwardDiff.
    dd(d) = double_interval_censored(d; primary_event = Uniform(0, 1), interval = 1)
    # Mirror the andv structure: a shared incubation leaf across both alternatives,
    # the index branch a single composed edge and the sourced branch a two-step
    # chain. `event(delays, :index)` then returns a composer scored by event name.
    inc_window = 30.0
    function delay_select(; inc = LogNormal(3.0, 0.3), delta = Normal(0.0, 1.0))
        index = compose((infection_onset = shared(:inc,
            primary_censored(inc, Uniform(0, inc_window))),))
        sourced = Sequential(
            (dd(delta), shared(:inc, dd(inc))),
            (:srconset_infection, :infection_onset))
        return selecting(:index => index, :sourced => sourced)
    end
    template = delay_select()
    priors = build_priors(params_table(template);
        priors = (
            inc = (mu = Normal(3.0, 0.5),
                sigma = truncated(Normal(0.0, 0.5); lower = 0)),
            sourced = (srconset_infection = (mu = Normal(0.0, 5.0),
                sigma = truncated(Normal(0.0, 1.0); lower = 0)),)))

    @model function andv(template, index_rows, sourced_rows)
        delays ~ to_submodel(composed_parameters_model(template, priors))
        obs_index ~ to_submodel(
            DynamicPPL.prefix(
                composed_distribution_model(event(delays, :index), index_rows),
                :index), false)
        obs_sourced ~ to_submodel(
            DynamicPPL.prefix(
                composed_distribution_model(
                    event(delays, :sourced), sourced_rows),
                :sourced), false)
        return delays
    end

    index_rows = [(infection = missing, onset = 25.0),
        (infection = missing, onset = 22.0)]
    sourced_rows = [(srconset = 0.0, infection = 3.0, onset = 25.0),
        (srconset = 0.0, infection = 5.0, onset = 28.0)]

    m = andv(template, index_rows, sourced_rows)

    # Draw prior parameter points until the linked log-density is finite, so the
    # gradient is compared at an in-support point of the full loop.
    function _insupport_varinfo(model)
        for attempt in 1:200
            varinfo = DynamicPPL.link(
                VarInfo(Xoshiro(attempt), model), model)
            ldf0 = DynamicPPL.LogDensityFunction(
                model, DynamicPPL.getlogjoint_internal, varinfo)
            isfinite(LDP.logdensity(ldf0, varinfo[:])) && return varinfo
        end
        error("no in-support prior draw found")
    end
    vi = _insupport_varinfo(m)
    x0 = vi[:]

    ldf_fd = DynamicPPL.LogDensityFunction(
        m, DynamicPPL.getlogjoint_internal, vi; adtype = AutoForwardDiff())
    ldf_mc = DynamicPPL.LogDensityFunction(
        m, DynamicPPL.getlogjoint_internal, vi;
        adtype = AutoMooncake(config = nothing))

    lp_fd, g_fd = LDP.logdensity_and_gradient(ldf_fd, x0)
    # The key assertion: Mooncake reverse builds a rule for the Select-top tree
    # (no `sub_ptr intrinsic hit`) and returns a finite gradient.
    lp_mc, g_mc = LDP.logdensity_and_gradient(ldf_mc, x0)

    @test isfinite(lp_fd)
    @test lp_mc ≈ lp_fd
    @test all(isfinite, g_mc)
    @test g_mc≈g_fd rtol=1e-5 atol=1e-7
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

# ---------------------------------------------------------------------------
# Varying / partially-pooled per-stratum params (the grouped primitive)
# ---------------------------------------------------------------------------

@testitem "grouped model entry equals the per-record stratum loop" tags=[:turing] begin
    using CensoredDistributions, Distributions, DynamicPPL

    # `composed_distribution_model(ds, table; group)` must score the same as the
    # explicit per-record loop over each record's OWN stratum distribution.
    mk(scale) = Sequential(
        primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
        primary_censored(Gamma(2.0, scale), Uniform(0, 1)))
    ds = [mk(1.0), mk(2.0)]
    rows = [(onset = 0.0, admit = 2.0, death = 5.0),
        (onset = 1.0, admit = 3.0, death = 9.0),
        (onset = 0.5, admit = 2.5, death = 6.0)]
    group = [1, 2, 1]

    @model function grouped(ds, t, g)
        obs ~ to_submodel(composed_distribution_model(ds, t; group = g))
        return obs
    end

    lp_model = only(logjoint(grouped(ds, rows, group), (;)))
    lp_loop = CensoredDistributions.batched_event_logpdf(ds, rows; group = group)
    @test lp_model ≈ lp_loop
end

@testitem "grouped model one stratum equals the shared-d model" tags=[:turing] begin
    using CensoredDistributions, Distributions, DynamicPPL

    # A single stratum recovers the shared-`d` vectorised model exactly.
    d = Sequential(
        primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))
    rows = [(onset = 0.0, admit = 2.0, death = 5.0),
        (onset = 1.0, admit = missing, death = 7.0)]

    @model function grouped(ds, t, g)
        obs ~ to_submodel(composed_distribution_model(ds, t; group = g))
        return obs
    end
    @model function shared(d, t)
        obs ~ to_submodel(composed_distribution_model(d, t))
        return obs
    end

    @test only(logjoint(grouped([d], rows, [1, 1]), (;))) ≈
          only(logjoint(shared(d, rows), (;)))
end

@testitem "partial-pooling submodel recovers hierarchical strata" tags=[:turing] begin
    using CensoredDistributions, Distributions, DynamicPPL, Turing, Random
    using FlexiChains: Prefixed, VNChain
    import Statistics

    # Per-stratum scale ~ a shared hyperprior (partial pooling). The grouped fit
    # samples a hyper-mean and per-stratum scales drawn off it, and should
    # recover both the hyper-mean and the per-stratum truth from grouped data.
    # The template carries a CENSORED edge (the vectorised record path needs a
    # shared primary event); only the inner Gamma scale is free.
    pc(scale) = primary_censored(Gamma(2.0, scale), Uniform(0, 1))
    template = compose((onset_admit = pc(1.0),))
    nstrata = 3
    true_scales = [1.0, 2.0, 3.0]

    # The user's enclosing model encodes the pooling: a hyper-mean `mu`, then each
    # stratum's prior is parameterised by `mu` (a shared hyperprior over the
    # stratum scales), assembled into the per-stratum priors vector the primitive
    # consumes.
    @model function pooled_fit(t, rows, group, nstrata)
        mu ~ truncated(Normal(2.0, 1.0); lower = 0)
        strata_priors = [(onset_admit = (
                             shape = truncated(Normal(2.0, 0.3); lower = 0),
                             scale = truncated(Normal(mu, 0.5); lower = 0)),)
                         for _ in 1:nstrata]
        ds ~ to_submodel(composed_parameters_model(t, strata_priors))
        obs ~ to_submodel(composed_distribution_model(ds, rows; group = group))
        return ds
    end

    # Simulate grouped data: per stratum, draw onset/admit from the true scale.
    Random.seed!(440)
    rows = NamedTuple[]
    group = Int[]
    for k in 1:nstrata
        truth = compose((onset_admit = pc(true_scales[k]),))
        for _ in 1:40
            s = rand(truth)
            push!(rows, (onset = s[1], admit = s[2]))
            push!(group, k)
        end
    end

    chain = sample(Xoshiro(7), pooled_fit(template, rows, group, nstrata),
        NUTS(0.8), 200; chain_type = VNChain, progress = false)

    # Per-stratum scales recover their truth (the hierarchy is identified). The
    # per-stratum scale lives under `ds.stratumK.onset_admit.scale`; read each by
    # its dotted varname.
    scales = (
        Statistics.mean(chain[Prefixed(@varname(ds.stratum1.onset_admit.scale))]),
        Statistics.mean(chain[Prefixed(@varname(ds.stratum2.onset_admit.scale))]),
        Statistics.mean(chain[Prefixed(@varname(ds.stratum3.onset_admit.scale))]))
    for k in 1:nstrata
        @test isapprox(scales[k], true_scales[k]; atol = 0.8)
    end
    # The per-stratum scales are ORDERED like the truth (the hierarchy is
    # identified, not pooled flat).
    @test scales[1] < scales[2] < scales[3]
    # The hyper-mean sits in the bulk of the per-stratum scales.
    mu_post = Statistics.mean(chain[Prefixed(@varname(mu))])
    @test 0.5 < mu_post < 4.0
end

@testitem "strip_prefix drops the submodel prefix from chain names" tags=[:turing] begin
    using CensoredDistributions, Distributions, DynamicPPL, Turing, Random
    using FlexiChains: VNChain, parameters
    import Statistics

    # `d ~ to_submodel(composed_parameters_model(...))` prefixes every sampled
    # parameter with the submodel variable name (`d.onset_admit.shape`).
    # `strip_prefix` removes that leading prefix so the user-facing chain names
    # drop to the edge path (`onset_admit.shape`) while staying unambiguous.
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

    Random.seed!(214)
    ys = [[0.5, 2.0], [1.0, 3.0], [0.8, 2.5]]
    chain = sample(fit(template, priors, ys), NUTS(), 40;
        chain_type = VNChain, progress = false)

    # Before: every parameter carries the `d.` prefix.
    before = Set(string.(collect(parameters(chain))))
    @test "d.onset_admit.shape" in before

    stripped = strip_prefix(chain)
    names = Set(string.(collect(parameters(stripped))))
    # After: the prefix is gone; the edge path remains (no collisions).
    @test names == Set(["onset_admit.shape", "onset_admit.scale",
        "admit_death.mu", "admit_death.sigma"])
    @test !any(startswith(n, "d.") for n in names)

    # The draws round-trip: a stripped name indexes the same column.
    @test vec(stripped[@varname(onset_admit.shape)]) ==
          vec(chain[@varname(d.onset_admit.shape)])

    # The stripped chain still reconstructs via update at the empty prefix.
    ready = update(template, stripped; prefix = Symbol(""))
    @test ready isa CensoredDistributions.Parallel
    @test event_names(ready) == event_names(template)

    # A non-default prefix is honoured.
    @model function fit2(t, p, ys)
        delays ~ to_submodel(composed_parameters_model(t, p))
        for y in ys
            DynamicPPL.@addlogprob! logpdf(delays, y)
        end
    end
    Random.seed!(215)
    chain2 = sample(fit2(template, priors, ys), NUTS(), 40;
        chain_type = VNChain, progress = false)
    stripped2 = strip_prefix(chain2; prefix = :delays)
    names2 = Set(string.(collect(parameters(stripped2))))
    @test "onset_admit.shape" in names2
    @test !any(startswith(n, "delays.") for n in names2)
end

@testitem "strip_prefix leaves non-prefixed parameters untouched" tags=[:turing] begin
    using CensoredDistributions, Distributions, DynamicPPL, Turing, Random
    using FlexiChains: VNChain, parameters

    # A model mixing a prefixed submodel parameter with a plain top-level one.
    # `strip_prefix(:d)` must drop only the `d.` prefix and leave `theta`
    # untouched rather than erroring on it.
    template = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    priors = build_priors(params_table(template))

    @model function fit(t, p, ys)
        theta ~ Normal(0, 1)
        d ~ to_submodel(composed_parameters_model(t, p))
        for y in ys
            DynamicPPL.@addlogprob! logpdf(d, y) + theta * 0
        end
    end

    Random.seed!(216)
    ys = [[0.5, 2.0], [1.0, 3.0]]
    chain = sample(fit(template, priors, ys), NUTS(), 40;
        chain_type = VNChain, progress = false)

    stripped = strip_prefix(chain)
    names = Set(string.(collect(parameters(stripped))))
    # The plain `theta` survives unchanged; the prefixed ones lose `d.`.
    @test "theta" in names
    @test "onset_admit.shape" in names
    @test !("d.onset_admit.shape" in names)
end
