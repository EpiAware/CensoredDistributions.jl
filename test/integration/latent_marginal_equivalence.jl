# Marginal == latent density-equivalence regression tests (#453).
#
# The marginal and latent forms of a censored composer are ONE model scored two
# ways: the marginal integrates the latent origin/intermediate events out inside
# `logpdf`; the latent SAMPLES them. Integrating the sampled latents back out of
# the latent joint MUST reproduce the marginal density. The bug fixed in #453 was
# that the latent form re-applied PRIMARY censoring on a SAMPLED-origin edge (the
# origin/intermediate was already sampled), double-counting the within-window
# uncertainty, so latent-integrated != marginal and neither matched the target
# models. The rule: an edge between two OBSERVED events keeps its declared
# censoring (#419); an edge with a SAMPLED endpoint scores the BARE core.
#
# These tests integrate the latent joint over its sampled latents on a grid and
# assert it equals the marginal `logpdf` to grid error, across representative
# shapes (leaf, bare chain, primary-censored chain, parallel, the andv & bdbv
# tree shapes), plus an andv term-by-term match against the target's bare-delay /
# bare-convolution scoring.

@testitem "marginal == latent-integrated: bare two-edge chain" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: latent, Sequential, convolve_distributions,
                                 composed_distribution_model, event_names
    using DynamicPPL: VarInfo, logjoint

    inc = LogNormal(1.5, 0.4)
    delta = Normal(0.3, 0.6)
    seq = Sequential((delta, inc), (:src_inf, :inf_onset))
    en = event_names(seq)                     # (:src, :inf, :onset)
    chain = latent(seq)

    # Origin observed (srconset = 0), intermediate (infection) sampled, terminal
    # (onset) observed. Marginal marginalises the intermediate -> bare convolution.
    function latent_int(onset; n = 40_000)
        xs = range(-6.0, onset - 1e-4; length = n)
        dx = step(xs)
        s = 0.0
        for x in xs
            row = NamedTuple{(en[1], en[2], en[3])}((0.0, x, onset))
            m = composed_distribution_model(chain, row)
            s += exp(logjoint(m, VarInfo(m))) * dx
        end
        return log(s)
    end

    conv = convolve_distributions(delta, inc)
    for onset in (2.0, 4.0, 7.0)
        ev = Vector{Union{Missing, Float64}}([0.0, missing, onset])
        marg = logpdf(seq, ev)
        # The marginal of a bare endpoint-observed chain IS the bare convolution.
        @test isapprox(marg, logpdf(conv, onset); atol = 1e-8)
        @test isapprox(latent_int(onset), marg; atol = 5e-3)
    end
end

@testitem "marginal == latent-integrated: leaf primary-censored" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: latent, get_dist

    inc = LogNormal(1.0, 0.5)
    d = primary_censored(inc, Uniform(0, 1))
    core = get_dist(d)
    # The leaf latent samples the primary p ~ Uniform(0, 1) and scores the
    # conditional core at y - p; integrating p reproduces the marginal logpdf.
    function latent_int(y; n = 40_000)
        xs = range(1e-6, 1 - 1e-6; length = n)
        dx = step(xs)
        s = 0.0
        for p in xs
            s += pdf(core, y - p) * dx     # Uniform(0,1) density is 1
        end
        return log(s)
    end
    for y in (1.5, 3.0, 5.0)
        @test isapprox(latent_int(y), logpdf(d, y); atol = 5e-3)
    end
end

@testitem "marginal == latent-integrated: 3-edge chain, sampled intermediate" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: latent, Sequential, composed_distribution_model,
                                 event_names
    using DynamicPPL: VarInfo, logjoint

    # A three-edge bare chain: origin observed, the first intermediate SAMPLED, the
    # second intermediate and terminal observed. The marginal convolves the bare
    # cores across the single unobserved-intermediate run; the latent samples that
    # intermediate. Integrating it reproduces the marginal — and the BOTH-OBSERVED
    # final edge keeps its declared scoring (#419), so the test mixes both policies.
    e1 = Normal(0.4, 0.5)
    e2 = Gamma(2.0, 0.8)
    e3 = LogNormal(0.6, 0.3)
    seq = Sequential((e1, e2, e3), (:a_b, :b_c, :c_d))
    en = event_names(seq)                     # (:a, :b, :c, :d)
    chain = latent(seq)

    function latent_int(b_unused, c, d; n = 50_000)
        # Origin a = 0 observed, b (first intermediate) sampled, c and d observed.
        xs = range(-6.0, c - 1e-4; length = n)
        dx = step(xs)
        s = 0.0
        for x in xs
            row = NamedTuple{(en[1], en[2], en[3], en[4])}((0.0, x, c, d))
            m = composed_distribution_model(chain, row)
            s += exp(logjoint(m, VarInfo(m))) * dx
        end
        return log(s)
    end

    for (c, d) in ((2.0, 4.0), (3.0, 6.0))
        ev = Vector{Union{Missing, Float64}}([0.0, missing, c, d])
        @test isapprox(latent_int(nothing, c, d), logpdf(seq, ev); atol = 5e-3)
    end
end

@testitem "marginal == latent-integrated: parallel shared origin" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: latent, Parallel, composed_distribution_model,
                                 event_names
    using DynamicPPL: VarInfo, logjoint

    # Two branches sharing one latent origin (the common primary). Origin sampled,
    # both branch events observed. Integrating the origin out of the latent joint
    # reproduces the marginal `logpdf(::Parallel, [missing, y1, y2])`.
    b1 = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))
    b2 = primary_censored(LogNormal(1.0, 0.4), Uniform(0, 1))
    par = Parallel((b1, b2), (:o_y1, :o_y2))
    en = event_names(par)                     # (:o, :y1, :y2)
    chain = latent(par)

    function latent_int(y1, y2; n = 40_000)
        xs = range(1e-6, 1 - 1e-6; length = n)
        dx = step(xs)
        s = 0.0
        for o in xs
            row = NamedTuple{(en[1], en[2], en[3])}((o, y1, y2))
            m = composed_distribution_model(chain, row)
            s += exp(logjoint(m, VarInfo(m))) * dx
        end
        return log(s)
    end

    for (y1, y2) in ((1.0, 2.5), (2.0, 4.0))
        ev = Vector{Union{Missing, Float64}}([missing, y1, y2])
        @test isapprox(latent_int(y1, y2), logpdf(par, ev); atol = 5e-3)
    end
end

@testitem "marginal == latent: nested Competing, observed anchor (#363)" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: latent, Sequential, Competing,
                                 composed_distribution_model
    using DynamicPPL: VarInfo, logjoint, @model, to_submodel

    # A latent-wrapped chain onset -> admit -> {death, discharge}. The recorded
    # outcome is data; the latent form CONDITIONS on it exactly like the marginal
    # Competing path. With the admit anchor OBSERVED both endpoints of the
    # conditioned branch are observed, so the latent edge keeps its declared
    # censoring and the latent term equals the marginal term edge-for-edge.
    edge(mu,
        sigma) = double_interval_censored(LogNormal(mu, sigma);
        primary_event = Uniform(0, 1), interval = 1.0)
    e_oa = edge(1.4, 0.4)
    cfr = 0.3
    death_d, disch_d = Gamma(2.0, 3.0), Gamma(2.0, 1.0)
    cmp = Competing(:death => (death_d, cfr), :discharge => (disch_d, 1 - cfr))
    seq = Sequential(e_oa, cmp)

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    for (oc, t, dly) in ((:death, 12.0, death_d), (:discharge, 11.0, disch_d))
        row = oc === :death ?
              (event_1 = 0.0, event_2 = 4.0, death = t, discharge = missing) :
              (event_1 = 0.0, event_2 = 4.0, death = missing, discharge = t)
        p = oc === :death ? cfr : 1 - cfr
        marg = only(logjoint(demo(seq, row), (;)))
        lat = only(logjoint(demo(latent(seq), row), (;)))
        ref = logpdf(e_oa, 4.0) + log(p) + logpdf(dly, t - 4.0)
        @test isapprox(marg, ref; atol = 1e-10)
        @test isapprox(lat, marg; atol = 1e-10)   # latent == marginal, exact
    end
end

@testitem "latent Competing: sampled anchor integrates to bare mixture term (#363)" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: latent, Sequential, Competing,
                                 convolve_distributions,
                                 composed_distribution_model, event_names
    using DynamicPPL: VarInfo, logjoint

    # onset(observed) -> admit(SAMPLED) -> {death(observed), discharge}. The latent
    # samples the unobserved admission; integrating it out must reproduce the
    # branch-prob-weighted BARE convolution at the observed death time:
    #   log p_death + logpdf(conv(bare_oa, bare_death), t_death).
    # Both edges go bare on a sampled endpoint (#453), so the integral is the bare
    # onset->death convolution, scaled by the death branch probability. (The
    # marginal nested-Competing scorer needs an observed anchor, so this sampled-
    # anchor marginalisation is a capability the latent form adds.)
    oa = Normal(0.4, 0.5)
    death_d = Gamma(2.0, 0.8)
    disch_d = LogNormal(0.6, 0.3)
    cfr = 0.3
    cmp = Competing(:death => (death_d, cfr), :discharge => (disch_d, 1 - cfr))
    seq = Sequential((oa, cmp), (:onset_admit, :admit))
    en = event_names(seq)                      # (:onset, :admit, :death, :discharge)
    chain = latent(seq)
    conv = convolve_distributions(oa, death_d)

    function latent_int(tdeath; n = 50_000)
        xs = range(-6.0, tdeath - 1e-4; length = n)
        dx = step(xs)
        s = 0.0
        for a in xs
            row = NamedTuple{en}((0.0, a, tdeath, missing))
            m = composed_distribution_model(chain, row)
            s += exp(logjoint(m, VarInfo(m))) * dx
        end
        return log(s)
    end

    for tdeath in (3.0, 5.0)
        ref = log(cfr) + logpdf(conv, tdeath)
        @test isapprox(latent_int(tdeath), ref; atol = 5e-3)
    end
end

@testitem "andv sourced: latent == marginal == bare convolution (target)" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: latent, Sequential, convolve_distributions,
                                 composed_distribution_model, completeness_probability,
                                 event_names, truncate_to_horizon
    using DynamicPPL: VarInfo, logjoint

    # The andv sourced branch at the published posterior means. Source onset is the
    # observed continuous anchor (srconset = 0), the infection is the sampled
    # latent, both edges are BARE (the #453 fix), so the marginal sourced delay is
    # the BARE convolution `delta + inc` and the latent integrates to it.
    inc = LogNormal(3.06, 0.32)
    delta = Normal(0.17, 0.62)
    seq = Sequential((delta, inc), (:srconset_infection, :infection_onset))
    en = event_names(seq)
    chain = latent(seq)
    conv = convolve_distributions(delta, inc)

    function latent_int(onset; n = 60_000)
        xs = range(-30.0, onset - 1e-4; length = n)
        dx = step(xs)
        s = 0.0
        for x in xs
            row = NamedTuple{(en[1], en[2], en[3])}((0.0, x, onset))
            m = composed_distribution_model(chain, row)
            s += exp(logjoint(m, VarInfo(m))) * dx
        end
        return log(s)
    end

    for onset in (15.0, 22.0, 30.0)
        # Term-by-term: the latent integrates to the BARE convolution density, the
        # exact density the target model scores (`logpdf(inc, T_onset - T_inf)` +
        # `logpdf(delta, T_inf - T_onset_src)` with both internal times sampled).
        @test isapprox(latent_int(onset), logpdf(conv, onset); atol = 1e-2)
    end

    # Sourced right-truncation: the denominator is the BARE convolution cdf at the
    # source-relative window (the target's `cdf(ConvolvedDelays(inc, δ), w)`), NOT
    # the censored-convolution cdf. `completeness_probability` on the bare chain is
    # exactly that cdf.
    for w in (20.0, 35.0, 60.0)
        @test isapprox(completeness_probability(conv, w), cdf(conv, w); atol = 1e-10)
    end
end

@testitem "andv index: declared double-interval-censored leaf (unchanged)" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: composed_distribution_model
    using DynamicPPL: @model, to_submodel, logjoint, VarInfo

    # The index branch is observed from the case's own exposure, so it keeps its
    # DECLARED double-interval censoring (#419) and right-truncation at its own
    # `obs_time` window. This guards that the #453 sampled-origin change does not
    # touch the observed-origin (index) edge scoring.
    inc = LogNormal(3.06, 0.32)
    pwindow = 1.0
    swindow = 1.0
    leaf = double_interval_censored(inc;
        primary_event = Uniform(0, pwindow), interval = swindow)

    @model demo(d, row) = obs ~ to_submodel(composed_distribution_model(d, row))

    for (delay, w) in ((5.0, 40.0), (12.0, 30.0))
        row = (delay = delay, obs_time = w)
        m = demo(leaf, row)
        lj = logjoint(m, VarInfo(m))
        # Equals the declared leaf truncated at the horizon, scored at the delay.
        ref = logpdf(CensoredDistributions.truncate_to_horizon(leaf, w), delay)
        @test isapprox(lj, ref; atol = 1e-8)
    end
end
