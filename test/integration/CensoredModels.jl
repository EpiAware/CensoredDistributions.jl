# Tests for the DynamicPPL extension submodels (issue #88):
# `primary_censored_model`, `interval_censored_model`, and
# `double_interval_censored_model`. These exercise the extension that loads when
# `DynamicPPL`/`Turing` is present (the core package stays Turing-free), checking
# that the submodels compile, that the marginal submodel's log-density matches
# the direct `logpdf`, that the easy marginal-versus-latent switch works by
# flipping only the `mode` on `d`, and that a short NUTS run samples.

@testitem "primary_censored_model marginal matches logpdf" tags=[:turing] begin
    using CensoredDistributions
    using CensoredDistributions: MarginalPrimaryCensored
    using Distributions
    using DynamicPPL
    using DynamicPPL: to_submodel

    d = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))
    @test d isa MarginalPrimaryCensored

    @model function fit(d, y)
        inner ~ to_submodel(primary_censored_model(d, y), false)
    end

    # The marginal submodel exposes no latent, so the model log-joint at the
    # (empty) parameter set equals the pure marginal logpdf.
    for y in (1.0, 2.0, 4.0)
        m = fit(d, y)
        @test logjoint(m, (;)) ≈ logpdf(d, y)
    end
end

@testitem "primary_censored_model marginal weighting via weight" tags=[
    :turing] begin
    using CensoredDistributions
    using Distributions
    using DynamicPPL
    using DynamicPPL: to_submodel

    d = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))

    @model function fit(d, y, w)
        inner ~ to_submodel(primary_censored_model(d, y; weight = w), false)
    end

    # Weighting goes through the `weight` distribution wrapper inside the `~`
    # (not an `@addlogprob!` hack), so the contribution is `w * logpdf`.
    for (y, w) in ((2.0, 7.0), (3.0, 3.0))
        m = fit(d, y, w)
        @test logjoint(m, (;)) ≈ w * logpdf(d, y)
    end
end

@testitem "primary_censored_model latent declares internal primary" tags=[
    :turing] begin
    using CensoredDistributions
    using CensoredDistributions: LatentPrimaryCensored
    using Distributions
    using DynamicPPL
    using DynamicPPL: to_submodel

    # Flip ONLY the mode to switch to latent; everything else is identical.
    d = primary_censored(
        LogNormal(1.5, 0.75), Uniform(0, 1); mode = Latent())
    @test d isa LatentPrimaryCensored

    @model function fit(d, y)
        inner ~ to_submodel(primary_censored_model(d, y), false)
    end

    m = fit(d, 2.0)
    vi = VarInfo(m)
    # The latent primary `p` lives inside the submodel (prefixed under `inner`)
    # and is a free variable the sampler owns; the user never declared it.
    varnames = string.(collect(keys(vi)))
    @test any(contains("p"), varnames)
end

@testitem "Easy marginal<->latent switch via mode only" tags=[:turing] begin
    using CensoredDistributions
    using CensoredDistributions: MarginalPrimaryCensored, LatentPrimaryCensored
    using Distributions
    using DynamicPPL
    using DynamicPPL: to_submodel

    # The SAME model definition, used with two distributions that differ only in
    # their `mode`, selects the marginal or latent submodel automatically. No
    # code in `fit` changes between the two.
    @model function fit(d, y)
        inner ~ to_submodel(primary_censored_model(d, y), false)
    end

    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0, 1)

    d_marg = primary_censored(delay, pe)                  # Marginal default
    d_lat = primary_censored(delay, pe; mode = Latent())  # flip only the mode

    @test d_marg isa MarginalPrimaryCensored
    @test d_lat isa LatentPrimaryCensored

    # Marginal: no latent variables.
    @test isempty(keys(VarInfo(fit(d_marg, 2.0))))
    # Latent: a sampler-owned latent primary appears.
    @test !isempty(keys(VarInfo(fit(d_lat, 2.0))))
end

@testitem "primary_censored_model injected origin (coupled case)" tags=[
    :turing] begin
    using CensoredDistributions
    using CensoredDistributions: get_dist
    using Distributions
    using DynamicPPL
    using DynamicPPL: to_submodel

    delay = LogNormal(1.5, 0.75)
    d = primary_censored(delay, Uniform(0, 1); mode = Latent())

    # With an injected origin the submodel scores ONLY the conditional delay
    # `logpdf(delay, y - origin)` and draws no primary-event prior: the caller
    # owns the prior over the coupled origin.
    @model function fit(d, y, o)
        inner ~ to_submodel(
            primary_censored_model(d, y; origin = o), false)
    end
    for (y, o) in ((2.0, 0.4), (3.0, 0.2))
        @test logjoint(fit(d, y, o), (;)) ≈ logpdf(get_dist(d), y - o)
        # No latent is drawn inside the submodel for the injected case.
        @test isempty(keys(VarInfo(fit(d, y, o))))
    end

    # Omitted origin (the default) is unchanged: the latent primary is drawn.
    @model function fit_plain(d, y)
        inner ~ to_submodel(primary_censored_model(d, y), false)
    end
    @test !isempty(keys(VarInfo(fit_plain(d, 2.0))))
end

@testitem "Coupled two-record origin NUTS smoke test" tags=[:turing] begin
    using CensoredDistributions
    using Distributions
    using Turing
    using DynamicPPL: to_submodel, prefix
    using Random

    Random.seed!(3)
    mu_t, sig_t = 1.5, 0.75
    o1_t = 0.5
    o2_t = o1_t + rand(LogNormal(mu_t, sig_t))   # transmission link
    y1 = o1_t + rand(LogNormal(mu_t, sig_t))
    y2 = o2_t + rand(LogNormal(mu_t, sig_t))

    # Record 2's origin is a function of record 1's latent origin: a coupled
    # prior the caller owns and injects via `origin`.
    @model function coupled(y1, y2)
        mu ~ Normal(1.0, 1.0)
        sigma ~ truncated(Normal(0.5, 0.5); lower = 0.01)
        d = primary_censored(
            LogNormal(mu, sigma), Uniform(0, 1); mode = Latent())
        r1 ~ to_submodel(prefix(primary_censored_model(d, y1), :r1), false)
        link ~ LogNormal(mu, sigma)
        o2 = r1.p + link
        r2 ~ to_submodel(
            prefix(primary_censored_model(d, y2; origin = o2), :r2), false)
    end

    chain = sample(coupled(y1, y2), NUTS(), 60; progress = false)
    @test all(isfinite, chain[:mu])
    @test all(chain[:sigma] .> 0)
    @test all(chain[:link] .> 0)
end

@testitem "interval_censored_model matches logpdf" tags=[:turing] begin
    using CensoredDistributions
    using Distributions
    using DynamicPPL
    using DynamicPPL: to_submodel

    d = interval_censored(Normal(5, 2), 1.0)

    @model function fit(d, y)
        inner ~ to_submodel(interval_censored_model(d, y), false)
    end

    for y in (3.0, 5.0, 7.0)
        @test logjoint(fit(d, y), (;)) ≈ logpdf(d, y)
    end

    # Weighting via the `weight` wrapper.
    @model function fitw(d, y, w)
        inner ~ to_submodel(interval_censored_model(d, y; weight = w), false)
    end
    @test logjoint(fitw(d, 5.0, 4.0), (;)) ≈ 4.0 * logpdf(d, 5.0)
end

@testitem "double_interval_censored_model matches logpdf" tags=[:turing] begin
    using CensoredDistributions
    using Distributions
    using DynamicPPL
    using DynamicPPL: to_submodel

    d = double_interval_censored(
        LogNormal(1.5, 0.75); upper = 10, interval = 1)

    @model function fit(d, y)
        inner ~ to_submodel(double_interval_censored_model(d, y), false)
    end

    for y in (1.0, 3.0, 6.0)
        @test logjoint(fit(d, y), (;)) ≈ logpdf(d, y)
    end

    # Weighting via the `weight` wrapper.
    @model function fitw(d, y, w)
        inner ~ to_submodel(
            double_interval_censored_model(d, y; weight = w), false)
    end
    @test logjoint(fitw(d, 3.0, 5.0), (;)) ≈ 5.0 * logpdf(d, 3.0)
end

@testitem "primary_censored_model NUTS smoke test (marginal)" tags=[
    :turing] begin
    using CensoredDistributions
    using Distributions
    using Turing
    using DynamicPPL: to_submodel, prefix
    using Random

    Random.seed!(42)
    d_gen = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))
    ys = rand(d_gen, 60)

    @model function marg_fit(ys)
        mu ~ Normal(1.0, 1.0)
        sigma ~ truncated(Normal(0.5, 0.5); lower = 0.01)
        d = primary_censored(LogNormal(mu, sigma), Uniform(0, 1))
        for i in eachindex(ys)
            obs ~ to_submodel(
                prefix(primary_censored_model(d, ys[i]), Symbol("obs", i)),
                false)
        end
    end

    chain = sample(marg_fit(ys), NUTS(), 50; progress = false)
    @test all(isfinite, chain[:mu])
    @test all(chain[:sigma] .> 0)
    # Loose recovery check: the posterior mean should be near the truth.
    @test abs(mean(chain[:mu]) - 1.5) < 0.5
end

@testitem "primary_censored_model NUTS smoke test (latent)" tags=[:turing] begin
    using CensoredDistributions
    using Distributions
    using Turing
    using DynamicPPL: to_submodel, prefix
    using Random

    Random.seed!(42)
    d_gen = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))
    ys = rand(d_gen, 40)

    # Same model body as the marginal smoke test, but the delay is built with
    # mode = Latent(): each observation gets its own sampler-owned primary.
    @model function lat_fit(ys)
        mu ~ Normal(1.0, 1.0)
        sigma ~ truncated(Normal(0.5, 0.5); lower = 0.01)
        d = primary_censored(
            LogNormal(mu, sigma), Uniform(0, 1); mode = Latent())
        for i in eachindex(ys)
            obs ~ to_submodel(
                prefix(primary_censored_model(d, ys[i]), Symbol("obs", i)),
                false)
        end
    end

    chain = sample(lat_fit(ys), NUTS(), 50; progress = false)
    @test all(isfinite, chain[:mu])
    @test all(chain[:sigma] .> 0)
end

# ===========================================================================
# Composed-type submodels (#309 sequential, #317 parallel, #318 tree,
# #320 competing)
# ===========================================================================

@testitem "Sequential submodel matches logpdf" tags=[:turing] begin
    using CensoredDistributions
    using Distributions
    using DynamicPPL
    using DynamicPPL: to_submodel

    delays = [Gamma(2.0, 1.0), Gamma(1.0, 1.0), LogNormal(0.5, 0.4)]
    d = sequential_distribution(delays)
    obs = [0.0, missing, 3.0, 5.0]   # middle event unobserved (marginalised)

    @model function fit(d, obs)
        inner ~ to_submodel(primary_censored_model(d, obs), false)
    end

    @test logjoint(fit(d, obs), (;)) ≈ logpdf(d, obs)

    # Weighting scales the joint log-density by the multiplicity.
    @model function fitw(d, obs, w)
        inner ~ to_submodel(
            primary_censored_model(d, obs; weight = w), false)
    end
    @test logjoint(fitw(d, obs, 4.0), (;)) ≈ 4.0 * logpdf(d, obs)

    # A zero weight contributes -Inf (matching the univariate `weight`
    # convention), not a `0 * logpdf` that could become NaN out of support.
    @test logjoint(fitw(d, obs, 0.0), (;)) == -Inf
end

@testitem "Parallel submodel matches logpdf and latent switch" tags=[
    :turing] begin
    using CensoredDistributions
    using Distributions
    using DynamicPPL
    using DynamicPPL: to_submodel

    d = primary_censored(
        [Gamma(2.0, 1.0) LogNormal(1.0, 0.5)], Uniform(0.0, 1.0))
    observed = [2.0, 3.0]

    @model function fit(d, observed)
        inner ~ to_submodel(primary_censored_model(d, observed), false)
    end

    # Marginal mechanic: prepend a missing shared origin, integrated inside
    # logpdf.
    @test logjoint(fit(d, observed), (;)) ≈ logpdf(d, vcat(missing, observed))

    # Latent mechanic: the shared origin is sampled inside the submodel and is
    # a free variable the sampler owns.
    @model function fit_latent(d, observed)
        inner ~ to_submodel(
            primary_censored_model(d, observed; latent = true), false)
    end
    @test !isempty(keys(VarInfo(fit_latent(d, observed))))
end

@testitem "Parallel latent: distinct VarNames and correct multi-branch score" tags=[
    :turing] begin
    # Regression test for the collapsed-VarName bug: the parallel-latent loop
    # must NOT reuse one `delay` VarName for every branch (which both collapsed
    # the chain and mis-scored the branches as resampled latents). Only the
    # shared origin `p` is latent; each branch's implied delay is a
    # deterministic likelihood contribution scored exactly once.
    using CensoredDistributions
    using CensoredDistributions: get_primary_event
    using Distributions
    using DynamicPPL
    using DynamicPPL: to_submodel, fix, @varname

    b1 = Gamma(2.0, 1.0)
    b2 = LogNormal(1.0, 0.5)
    b3 = Gamma(1.5, 2.0)
    d = primary_censored([b1 b2 b3], Uniform(0.0, 1.0))
    observed = [2.0, 3.0, 4.0]

    @model function fit_latent(d, observed)
        inner ~ to_submodel(
            primary_censored_model(d, observed; latent = true), false)
    end
    m = fit_latent(d, observed)

    # The shared origin `p` is the ONLY latent; the branch delays are
    # deterministic given `p`, so no per-branch `delay`/`branch_delay` VarName
    # is created (the collapse bug would have left a single shared one).
    vns = string.(collect(keys(VarInfo(m))))
    @test vns == ["p"]
    @test !any(contains("delay"), vns)

    # Correct multi-branch conditional log-density at a fixed origin: the origin
    # prior plus the sum over the three present branches, matching the pure
    # `ParallelPrimaryCensored` conditional `logpdf([p, observed...])`.
    p = 0.4
    mfix = fix(m, (@varname(p) => p,))
    ref = logpdf(get_primary_event(d), p) +
          logpdf(b1, observed[1] - p) +
          logpdf(b2, observed[2] - p) +
          logpdf(b3, observed[3] - p)
    @test logjoint(mfix, (;)) ≈ ref
    @test logjoint(mfix, (;)) ≈ logpdf(d, vcat(p, observed))
end

@testitem "Parallel latent: per-record origins are distinctly named" tags=[
    :turing] begin
    # Multi-record latent models namespace each record's shared origin with the
    # caller's `prefix` (the EpiAware `cases.y_t`-style scheme), giving distinct,
    # groupable VarNames `o1.p`, `o2.p`, ... rather than a single collapsed name.
    using CensoredDistributions
    using Distributions
    using DynamicPPL
    using DynamicPPL: to_submodel, prefix

    d = primary_censored(
        [Gamma(2.0, 1.0) LogNormal(1.0, 0.5)], Uniform(0.0, 1.0))
    records = [[2.0, 3.0], [2.5, 3.5], [1.5, 4.0]]

    @model function fit(d, records)
        for i in eachindex(records)
            r ~ to_submodel(
                prefix(
                    primary_censored_model(d, records[i]; latent = true),
                    Symbol("o", i)), false)
        end
    end

    vns = string.(collect(keys(VarInfo(fit(d, records)))))
    # One distinctly-named latent origin per record.
    @test Set(vns) == Set(["o1.p", "o2.p", "o3.p"])
end

@testitem "Event-tree submodel matches logpdf" tags=[:turing] begin
    using CensoredDistributions
    using Distributions
    using DynamicPPL
    using DynamicPPL: to_submodel

    edges = (parent = [:onset, :onset, :admit, :admit],
        child = [:admit, :notif, :death, :disch],
        delay = [Gamma(2.0, 1.0), LogNormal(1.0, 0.5),
            Gamma(1.5, 1.0), Gamma(2.0, 1.5)])
    tree = primary_censored(edges, Uniform(0.0, 1.0))
    # `disch` missing is marginalised; `admit` is the shared interior node.
    obs = (onset = 0.0, admit = 2.0, death = 5.0, disch = missing,
        notif = 1.5)

    @model function fit(d, obs)
        inner ~ to_submodel(primary_censored_model(d, obs), false)
    end

    @test logjoint(fit(tree, obs), (;)) ≈ logpdf(tree, obs)
end

@testitem "Competing-node submodels match logpdf" tags=[:turing] begin
    using CensoredDistributions
    using CensoredDistributions: as_mixture
    using Distributions
    using DynamicPPL
    using DynamicPPL: to_submodel

    # In a tree: the competing node lowers to a mixture and its branch is
    # marginalised inside the tree logpdf.
    cfr = 0.3
    cfr_edges = (parent = [:onset, :admit],
        child = [:admit, :outcome],
        delay = [Gamma(2.0, 1.0),
            Competing(:death => (Gamma(1.5, 1.0), cfr),
                :disch => (Gamma(2.0, 1.5), 1 - cfr))])
    cfr_tree = primary_censored(cfr_edges, Uniform(0.0, 1.0))
    obs = (onset = 0.0, admit = 2.0, death = 5.0, disch = missing)

    @model function fit_tree(d, obs)
        inner ~ to_submodel(primary_censored_model(d, obs), false)
    end
    @test logjoint(fit_tree(cfr_tree, obs), (;)) ≈ logpdf(cfr_tree, obs)

    # Stand-alone competing node: score the resolution gap against the mixture.
    node = Competing(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.7))

    @model function fit_node(c, gap)
        inner ~ to_submodel(primary_censored_model(c, gap), false)
    end
    @test logjoint(fit_node(node, 3.0), (;)) ≈ logpdf(as_mixture(node), 3.0)
end

@testitem "Parallel submodel NUTS smoke test" tags=[:turing] begin
    using CensoredDistributions
    using Distributions
    using Turing
    using DynamicPPL: to_submodel, prefix
    using Random

    Random.seed!(7)
    gen = primary_censored(
        [Gamma(2.0, 1.0) LogNormal(1.0, 0.5)], Uniform(0.0, 1.0))
    obs = [rand(gen)[2:3] for _ in 1:30]   # branch observations per record

    @model function par_fit(obs)
        shape ~ truncated(Normal(2.0, 1.0); lower = 0.1)
        d = primary_censored(
            [Gamma(shape, 1.0) LogNormal(1.0, 0.5)], Uniform(0.0, 1.0))
        for i in eachindex(obs)
            inner ~ to_submodel(
                prefix(primary_censored_model(d, obs[i]), Symbol("o", i)),
                false)
        end
    end

    chain = sample(par_fit(obs), NUTS(), 40; progress = false)
    @test all(isfinite, chain[:shape])
    @test all(chain[:shape] .> 0)
end

@testitem "Competing-tree NUTS recovers the CFR" tags=[:turing] begin
    using CensoredDistributions
    using CensoredDistributions: Competing
    using Distributions
    using Turing
    using DynamicPPL: to_submodel, prefix
    using Random

    Random.seed!(7)

    @model function tree_fit(obss)
        cfr ~ Beta(2, 2)
        edges = (parent = [:onset, :admit], child = [:admit, :outcome],
            delay = [Gamma(2.0, 1.0),
                Competing(:death => (Gamma(1.5, 1.0), cfr),
                    :disch => (Gamma(2.0, 1.5), 1 - cfr))])
        tree = primary_censored(edges, Uniform(0.0, 1.0))
        for i in eachindex(obss)
            inner ~ to_submodel(
                prefix(primary_censored_model(tree, obss[i]),
                    Symbol("c", i)), false)
        end
    end

    # 12 deaths and 28 discharges: the realised-branch likelihood should pull
    # the CFR towards 0.3.
    obss = vcat(
        [(onset = 0.0, admit = 2.0, death = 4.0, disch = missing)
         for _ in 1:12],
        [(onset = 0.0, admit = 2.0, death = missing, disch = 5.0)
         for _ in 1:28])

    chain = sample(tree_fit(obss), NUTS(), 60; progress = false)
    @test all(isfinite, chain[:cfr])
    @test all(0 .<= chain[:cfr] .<= 1)
    @test abs(mean(chain[:cfr]) - 0.3) < 0.2
end
