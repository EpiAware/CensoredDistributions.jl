# Marginal == latent density-equivalence regression tests.
#
# The marginal and latent forms of a censored composer are ONE model scored two
# ways: the marginal integrates the latent origin/intermediate events out inside
# `logpdf`; the latent SAMPLES them. Integrating the sampled latents back out of
# the latent joint MUST reproduce the marginal density. An earlier bug was
# that the latent form re-applied PRIMARY censoring on a SAMPLED-origin edge (the
# origin/intermediate was already sampled), double-counting the within-window
# uncertainty, so latent-integrated != marginal and neither matched the target
# models. The rule: an edge between two OBSERVED events keeps its declared
# censoring; an edge with a SAMPLED endpoint scores the BARE core.
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
    # final edge keeps its declared scoring, so the test mixes both policies.
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

@testitem "marginal == latent: nested Resolve, observed anchor" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: latent, Sequential, Resolve,
                                 composed_distribution_model
    using DynamicPPL: VarInfo, logjoint, @model, to_submodel

    # A latent-wrapped chain onset -> admit -> {death, discharge}. The recorded
    # outcome is data; the latent form CONDITIONS on it exactly like the marginal
    # Resolve path. With the admit anchor OBSERVED both endpoints of the
    # conditioned branch are observed, so the latent edge keeps its declared
    # censoring and the latent term equals the marginal term edge-for-edge.
    edge(mu,
        sigma) = double_interval_censored(LogNormal(mu, sigma);
        primary_event = Uniform(0, 1), interval = 1.0)
    e_oa = edge(1.4, 0.4)
    cfr = 0.3
    death_d, disch_d = Gamma(2.0, 3.0), Gamma(2.0, 1.0)
    cmp = Resolve(:death => (death_d, cfr), :discharge => (disch_d, 1 - cfr))
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

@testitem "marginal == latent: nested Resolve, right-truncated" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: latent, Sequential, Resolve,
                                 composed_distribution_model, event_names,
                                 truncate_to_horizon, _nested_tree_logpdf,
                                 _origin_primary_event, _first_origin_node
    using DynamicPPL: logjoint, @model, to_submodel
    using ForwardDiff: gradient

    # The nested-Resolve scorer must honour a per-record right-truncation
    # `horizon` exactly as the top-level Resolve path does. Previously
    # the nested scorer ignored the horizon, scoring the UNtruncated branch density
    # (a likelihood mis-specification) and breaking the top-level-vs-nested
    # symmetry. The chain is onset -> admit -> {death, discharge} with the admit
    # anchor OBSERVED, so the conditioned branch is right-truncated at the remaining
    # window `horizon - admit` (the time left to observe the outcome from its
    # anchor), mirroring the top-level node truncated at the horizon from origin 0.
    edge(mu,
        sigma) = double_interval_censored(LogNormal(mu, sigma);
        primary_event = Uniform(0, 1), interval = 1.0)
    e_oa = edge(1.4, 0.4)
    cfr = 0.3
    death_d, disch_d = Gamma(2.0, 3.0), Gamma(2.0, 1.0)
    cmp = Resolve(:death => (death_d, cfr), :discharge => (disch_d, 1 - cfr))
    seq = Sequential(e_oa, cmp)

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    admit = 4.0
    horizon = 20.0
    prim = _origin_primary_event(_first_origin_node(seq))

    for (oc, t, dly) in ((:death, 12.0, death_d), (:discharge, 11.0, disch_d))
        p = oc === :death ? cfr : 1 - cfr
        slot = oc === :death ? 3 : 4
        ev = Vector{Union{Missing, Float64}}([0.0, admit, missing, missing])
        ev[slot] = t

        # The marginal nested-tree scorer with vs without a horizon. (The flat
        # public marginal entry guards nested-tree horizons separately; the SCORER
        # itself threads the horizon.)
        marg_trunc = _nested_tree_logpdf(seq, ev, prim, Float64, horizon)
        marg_untrunc = _nested_tree_logpdf(seq, ev, prim, Float64)

        # The latent form scores the same record through the public model, which
        # routes the nested Resolve through `_latent_one_of_logprob`.
        row = oc === :death ?
              (event_1 = 0.0, event_2 = admit, death = t, discharge = missing,
            obs_time = horizon) :
              (event_1 = 0.0, event_2 = admit, death = missing, discharge = t,
            obs_time = horizon)
        row0 = oc === :death ?
               (event_1 = 0.0, event_2 = admit, death = t, discharge = missing) :
               (event_1 = 0.0, event_2 = admit, death = missing, discharge = t)
        lat_trunc = only(logjoint(demo(latent(seq), row), (;)))
        lat_untrunc = only(logjoint(demo(latent(seq), row0), (;)))

        # (a) The truncated nested contribution DIFFERS from the untruncated one
        # (the bug scored the untruncated branch density).
        @test !isapprox(marg_trunc, marg_untrunc)

        # The expected truncated term: the onset->admit edge, the branch weight, and
        # the death/discharge branch RIGHT-TRUNCATED at the remaining window.
        ref_branch = logpdf(truncate_to_horizon(dly, horizon - admit), t - admit)
        ref_trunc = logpdf(e_oa, admit) + log(p) + ref_branch
        @test isapprox(marg_trunc, ref_trunc; atol = 1e-10)

        # (b) The nested truncated branch term MATCHES the analogous top-level
        # Resolve truncation (anchored at 0, window `horizon - admit`).
        top = Resolve(:death => (death_d, cfr),
            :discharge => (disch_d, 1 - cfr))
        tlrow = oc === :death ?
                (death = t - admit, discharge = missing,
            obs_time = horizon - admit) :
                (death = missing, discharge = t - admit,
            obs_time = horizon - admit)
        tl = only(logjoint(demo(top, tlrow), (;)))
        @test isapprox(tl, log(p) + ref_branch; atol = 1e-10)

        # (c) marginal == latent for the right-truncated nested node (the project
        # invariant: the two forms must be density-identical).
        @test isapprox(marg_trunc, lat_trunc; atol = 1e-10)

        # (d) horizon = nothing reproduces today's value EXACTLY (byte-identical).
        @test marg_untrunc ==
              _nested_tree_logpdf(seq, ev, prim, Float64, nothing)
        @test isapprox(lat_untrunc, marg_untrunc; atol = 1e-10)
    end

    # ForwardDiff AD smoke over the branch params through the truncated nested
    # scorer (the horizon is DATA, the params differentiate). The truncation
    # denominator routes through the branch `logcdf`, so the outcomes are LogNormal
    # (its `logcdf` is ForwardDiff-friendly; a Gamma `logcdf` is not, the same
    # restriction the whole-compose truncation AD tests document). The event vector
    # stays the data type `Float64`; the Dual rides the branch params.
    ev = Vector{Union{Missing, Float64}}([0.0, admit, 12.0, missing])
    function nll(theta)
        mu_d, mu_s = theta
        cc = Resolve(:death => (LogNormal(mu_d, 0.4), cfr),
            :discharge => (LogNormal(mu_s, 0.5), 1 - cfr))
        s = Sequential(e_oa, cc)
        return -_nested_tree_logpdf(s, ev, prim, Float64, horizon)
    end
    g = gradient(nll, [1.5, 1.2])
    @test all(isfinite, g)
    # The truncation MOVES the gradient (the horizon enters the denominator).
    g0 = gradient(
        theta -> -_nested_tree_logpdf(
            Sequential(e_oa,
                Resolve(:death => (LogNormal(theta[1], 0.4), cfr),
                    :discharge => (LogNormal(theta[2], 0.5), 1 - cfr))),
            ev, prim, Float64, nothing), [1.5, 1.2])
    @test !isapprox(g, g0)
end

@testitem "marginal == latent: nested Resolve via PUBLIC flat entry" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: latent, Sequential, Resolve,
                                 composed_distribution_model, event_logpdf,
                                 truncate_to_horizon, _nested_tree_logpdf,
                                 _origin_primary_event, _first_origin_node,
                                 _tree_acc_type
    using DynamicPPL: logjoint, @model, to_submodel
    using ForwardDiff: gradient

    # The PUBLIC marginal entry `event_logpdf(::Sequential, events; horizon)` must
    # accept a nested-tree record carrying a per-record `horizon` and thread it down
    # to the (now horizon-capable) nested scorer. Previously the entry
    # GUARDED/REJECTED a nested-tree record with a horizon (it threw an
    # `ArgumentError`), even though the underlying `_nested_tree_logpdf` already
    # truncates each nested Resolve node at the per-record window. The chain is
    # onset -> admit -> {death, discharge}, admit OBSERVED, so the conditioned branch
    # is right-truncated at the remaining window `horizon - admit`, mirroring the
    # top-level Resolve node truncated at `horizon` from origin 0.
    edge(mu,
        sigma) = double_interval_censored(LogNormal(mu, sigma);
        primary_event = Uniform(0, 1), interval = 1.0)
    e_oa = edge(1.4, 0.4)
    cfr = 0.3
    death_d, disch_d = Gamma(2.0, 3.0), Gamma(2.0, 1.0)
    cmp = Resolve(:death => (death_d, cfr), :discharge => (disch_d, 1 - cfr))
    seq = Sequential(e_oa, cmp)

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    admit = 4.0
    horizon = 20.0
    prim = _origin_primary_event(_first_origin_node(seq))

    for (oc, t, dly) in ((:death, 12.0, death_d), (:discharge, 11.0, disch_d))
        p = oc === :death ? cfr : 1 - cfr
        slot = oc === :death ? 3 : 4
        ev = Vector{Union{Missing, Float64}}([0.0, admit, missing, missing])
        ev[slot] = t

        # The PUBLIC flat entry now THREADS the horizon instead of throwing.
        pub_trunc = event_logpdf(seq, ev; horizon = horizon)
        pub_untrunc = event_logpdf(seq, ev; horizon = nothing)

        # The direct nested scorer the entry threads into, and the latent form of
        # the same record through the public model.
        direct = _nested_tree_logpdf(seq, ev, prim, _tree_acc_type(seq, ev),
            horizon)
        row = oc === :death ?
              (event_1 = 0.0, event_2 = admit, death = t, discharge = missing,
            obs_time = horizon) :
              (event_1 = 0.0, event_2 = admit, death = missing, discharge = t,
            obs_time = horizon)
        lat_trunc = only(logjoint(demo(latent(seq), row), (;)))

        # The expected truncated term: the onset->admit edge, the branch weight, and
        # the death/discharge branch RIGHT-TRUNCATED at the remaining window.
        ref_branch = logpdf(truncate_to_horizon(dly, horizon - admit), t - admit)
        ref_trunc = logpdf(e_oa, admit) + log(p) + ref_branch

        # (a) public marginal == direct scorer == latent == closed-form reference,
        # the project invariant (all density-identical to ~1e-10).
        @test isapprox(pub_trunc, direct; atol = 1e-10)
        @test isapprox(pub_trunc, lat_trunc; atol = 1e-10)
        @test isapprox(pub_trunc, ref_trunc; atol = 1e-10)

        # (b) the public truncated term MATCHES the analogous top-level Resolve
        # truncation (anchored at 0, window `horizon - admit`), scored through the
        # model's own per-record `obs_time` route.
        top = Resolve(:death => (death_d, cfr),
            :discharge => (disch_d, 1 - cfr))
        tlrow = oc === :death ?
                (death = t - admit, discharge = missing,
            obs_time = horizon - admit) :
                (death = missing, discharge = t - admit,
            obs_time = horizon - admit)
        tl = only(logjoint(demo(top, tlrow), (;)))
        @test isapprox(tl, log(p) + ref_branch; atol = 1e-10)

        # (c) horizon = nothing reproduces today's value BYTE-FOR-BYTE (it is exactly
        # the back-compat `logpdf(seq, ev)` path, untouched by this change).
        @test pub_untrunc === logpdf(seq, ev)
    end

    # ForwardDiff AD smoke through the PUBLIC entry over the branch params (the
    # horizon is DATA, the params differentiate). The truncation denominator routes
    # through the branch `logcdf`, so the outcomes are LogNormal (its `logcdf` is
    # ForwardDiff-friendly; a Gamma `logcdf` is not, the documented existing
    # restriction). The event vector stays Float64; the Dual rides the branch params.
    ev = Vector{Union{Missing, Float64}}([0.0, admit, 12.0, missing])
    function nll(theta)
        mu_d, mu_s = theta
        cc = Resolve(:death => (LogNormal(mu_d, 0.4), cfr),
            :discharge => (LogNormal(mu_s, 0.5), 1 - cfr))
        s = Sequential(e_oa, cc)
        return -event_logpdf(s, ev; horizon = horizon)
    end
    g = gradient(nll, [1.5, 1.2])
    @test all(isfinite, g)
    # The truncation MOVES the gradient (the horizon enters the denominator).
    g0 = gradient(
        theta -> -event_logpdf(
            Sequential(e_oa,
                Resolve(:death => (LogNormal(theta[1], 0.4), cfr),
                    :discharge => (LogNormal(theta[2], 0.5), 1 - cfr))),
            ev; horizon = nothing), [1.5, 1.2])
    @test !isapprox(g, g0)
end

@testitem "marginal == latent: δ-bounded leaf right-truncation" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: composed_distribution_model, truncate_to_window
    using DynamicPPL: @model, to_submodel, logjoint, VarInfo

    # A δ-bounded leaf record: the reserved `obs_window` δ adds a LOWER edge a
    # width δ below the `obs_time` horizon, so the leaf is truncated to the finite
    # window `[obs_time - δ, obs_time]` (normaliser cdf(upper) - cdf(lower)). The
    # marginal leaf model scores the δ-bounded truncated leaf; this guards that the
    # row's `obs_window` threads through to the δ-bounded truncation and that
    # `obs_window` absent reproduces the upper-only form exactly.
    inc = LogNormal(3.06, 0.32)
    leaf = double_interval_censored(inc;
        primary_event = Uniform(0, 1), interval = 1.0)

    @model demo(d, row) = obs ~ to_submodel(composed_distribution_model(d, row))

    for (delay, D, δ) in ((5.0, 40.0, 30.0), (12.0, 30.0, 25.0))
        rowδ = (delay = delay, obs_time = D, obs_window = δ)
        lj = logjoint(demo(leaf, rowδ), VarInfo(demo(leaf, rowδ)))
        ref = logpdf(truncate_to_window(leaf, D, δ), delay)
        @test isapprox(lj, ref; atol = 1e-8)

        # The δ-bounded score DIFFERS from the upper-only horizon score (the lower
        # edge changes the normaliser), so the δ is genuinely applied.
        row_up = (delay = delay, obs_time = D)
        lj_up = logjoint(demo(leaf, row_up), VarInfo(demo(leaf, row_up)))
        @test !isapprox(lj, lj_up)
        # And the upper-only row reproduces the upper-only truncation exactly.
        @test isapprox(lj_up,
            logpdf(CensoredDistributions.truncate_to_horizon(leaf, D), delay);
            atol = 1e-10)
    end
end

@testitem "marginal == latent: δ-bounded nested Resolve right-truncation" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: latent, Sequential, Resolve,
                                 composed_distribution_model,
                                 truncate_to_window, _nested_tree_logpdf,
                                 _origin_primary_event, _first_origin_node,
                                 WindowedHorizon, event_logpdf, _tree_acc_type
    using DynamicPPL: logjoint, @model, to_submodel
    using ForwardDiff: gradient

    # The δ-bounded variant of the nested-Resolve right-truncation: the
    # conditioned branch is truncated to the FINITE window `[window - δ, window]`
    # (window = horizon - anchor) rather than `(-∞, window]`. The marginal nested
    # scorer, the public flat entry, and the latent form must all be density-
    # identical (the project invariant), and δ → full window must reproduce the
    # upper-only value exactly. The chain is onset -> admit -> {death, discharge}
    # with the admit anchor OBSERVED.
    edge(mu,
        sigma) = double_interval_censored(LogNormal(mu, sigma);
        primary_event = Uniform(0, 1), interval = 1.0)
    e_oa = edge(1.4, 0.4)
    cfr = 0.3
    death_d, disch_d = Gamma(2.0, 3.0), Gamma(2.0, 1.0)
    cmp = Resolve(:death => (death_d, cfr), :discharge => (disch_d, 1 - cfr))
    seq = Sequential(e_oa, cmp)

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    admit = 4.0
    horizon = 20.0
    δ = 10.0                                    # window [horizon-admit-δ, .] inc.
    wh = WindowedHorizon(horizon, δ)
    prim = _origin_primary_event(_first_origin_node(seq))

    for (oc, t, dly) in ((:death, 12.0, death_d), (:discharge, 11.0, disch_d))
        p = oc === :death ? cfr : 1 - cfr
        slot = oc === :death ? 3 : 4
        ev = Vector{Union{Missing, Float64}}([0.0, admit, missing, missing])
        ev[slot] = t

        marg = _nested_tree_logpdf(seq, ev, prim, Float64, wh)
        pub = event_logpdf(seq, ev; horizon = wh)

        # Closed-form reference: the onset->admit edge, the branch weight, and the
        # branch δ-bounded to the finite window `[horizon-admit-δ, horizon-admit]`.
        ref_branch = logpdf(
            truncate_to_window(dly, horizon - admit, δ), t - admit)
        ref = logpdf(e_oa, admit) + log(p) + ref_branch

        row = oc === :death ?
              (event_1 = 0.0, event_2 = admit, death = t, discharge = missing,
            obs_time = horizon, obs_window = δ) :
              (event_1 = 0.0, event_2 = admit, death = missing, discharge = t,
            obs_time = horizon, obs_window = δ)
        lat = only(logjoint(demo(latent(seq), row), (;)))

        # (a) marginal == public == latent == closed-form reference, all density-
        # identical (the project invariant).
        @test isapprox(marg, ref; atol = 1e-10)
        @test isapprox(pub, marg; atol = 1e-10)
        @test isapprox(lat, marg; atol = 1e-10)

        # (b) the δ-bounded score DIFFERS from the upper-only horizon score (the
        # lower edge enters the denominator).
        marg_up = _nested_tree_logpdf(seq, ev, prim, Float64, horizon)
        @test !isapprox(marg, marg_up)

        # (c) δ → full window reproduces the upper-only value EXACTLY. A δ wider
        # than the window collapses the lower edge to the branch minimum, the
        # upper-only truncation.
        wide = WindowedHorizon(horizon, horizon + 1000.0)
        @test _nested_tree_logpdf(seq, ev, prim, Float64, wide) === marg_up
        @test event_logpdf(seq, ev; horizon = wide) === marg_up
    end

    # ForwardDiff AD over the branch params through the δ-bounded nested scorer
    # (the horizon/δ are DATA, the params differentiate). The δ-bounded
    # denominator routes through the branch `logcdf` at both edges, so the
    # outcomes are LogNormal (ForwardDiff-friendly `logcdf`).
    ev = Vector{Union{Missing, Float64}}([0.0, admit, 12.0, missing])
    function nll(theta)
        mu_d, mu_s = theta
        cc = Resolve(:death => (LogNormal(mu_d, 0.4), cfr),
            :discharge => (LogNormal(mu_s, 0.5), 1 - cfr))
        s = Sequential(e_oa, cc)
        return -_nested_tree_logpdf(s, ev, prim, Float64, wh)
    end
    g = gradient(nll, [1.5, 1.2])
    @test all(isfinite, g)
    # The δ lower edge MOVES the gradient relative to the upper-only form.
    g_up = gradient(
        theta -> -_nested_tree_logpdf(
            Sequential(e_oa,
                Resolve(:death => (LogNormal(theta[1], 0.4), cfr),
                    :discharge => (LogNormal(theta[2], 0.5), 1 - cfr))),
            ev, prim, Float64, horizon), [1.5, 1.2])
    @test !isapprox(g, g_up)
end

@testitem "latent Resolve: sampled anchor integrates to bare mixture term" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: latent, Sequential, Resolve,
                                 convolve_distributions,
                                 composed_distribution_model, event_names
    using DynamicPPL: VarInfo, logjoint

    # onset(observed) -> admit(SAMPLED) -> {death(observed), discharge}. The latent
    # samples the unobserved admission; integrating it out must reproduce the
    # branch-prob-weighted BARE convolution at the observed death time:
    #   log p_death + logpdf(conv(bare_oa, bare_death), t_death).
    # Both edges go bare on a sampled endpoint, so the integral is the bare
    # onset->death convolution, scaled by the death branch probability. (The
    # marginal nested-Resolve scorer needs an observed anchor, so this sampled-
    # anchor marginalisation is a capability the latent form adds.)
    oa = Normal(0.4, 0.5)
    death_d = Gamma(2.0, 0.8)
    disch_d = LogNormal(0.6, 0.3)
    cfr = 0.3
    cmp = Resolve(:death => (death_d, cfr), :discharge => (disch_d, 1 - cfr))
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
    # latent, both edges are BARE, so the marginal sourced delay is
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

@testitem "bdbv: interval-observed intermediate factorises == latent" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: Sequential, latent_segments, latent_records,
                                 latent_observed_logpdf, event_names
    using QuadGK: quadgk

    # #647 (CONTRACT HOLDS): the bdbv/andv onset -> admit -> resolution chain with
    # the intermediate `admit` OBSERVED as an interval difference (day-resolution
    # double-interval-censored edges). The contract (memory: an OBSERVED -> OBSERVED
    # edge keeps its declared censoring; the chain FACTORISES at the observed
    # intermediate) must be density-identical to the project's latent form, which
    # decomposes the chain into per-segment latent edges each anchored at the
    # OBSERVED admit. The earlier "marginalise the continuous admit JOINTLY across
    # both edges" reference is a DIFFERENT model (admit latent/unobserved), not the
    # observed-intermediate model scored here.
    oa = double_interval_censored(LogNormal(1.2, 0.4);
        primary_event = Uniform(0, 1), interval = 1.0)
    ar = double_interval_censored(Gamma(2.0, 1.5);
        primary_event = Uniform(0, 1), interval = 1.0)
    seq = Sequential((oa, ar), (:onset_admit, :admit_resolution))
    @test event_names(seq) == (:onset, :admit, :resolution)

    segs = latent_segments(seq)

    for row in ((onset = 0.0, admit = 3.0, resolution = 7.0),
        (onset = 0.0, admit = 5.0, resolution = 9.0))
        ev = Vector{Union{Missing, Float64}}(
            [row.onset, row.admit, row.resolution])
        marg = logpdf(seq, ev)

        # (a) the marginal factorises at the OBSERVED admit: each edge keeps its
        # declared double-interval censoring, scored at its own observed gap.
        ref = logpdf(oa, row.admit - row.onset) +
              logpdf(ar, row.resolution - row.admit)
        @test isapprox(marg, ref; atol = 1e-12)

        # (b) the project's latent form, integrating each per-segment sampled
        # primary out exactly (quadgk), reproduces the marginal to machine
        # precision -- the marginal == latent invariant for an interval-observed
        # intermediate. The segments are independent, so the joint integral is the
        # product of the per-segment integrals (anchored at the observed admit).
        lrows = latent_records(seq, [row])
        @test length(lrows) == 2
        acc = 1.0
        for k in eachindex(lrows)
            val,
            _ = quadgk(
                p -> exp(latent_observed_logpdf(segs, [lrows[k]], [p])),
                0.0, 1.0; rtol = 1e-10)
            acc *= val
        end
        @test isapprox(marg, log(acc); atol = 1e-8)

        # (c) the latent decomposition anchors the second segment at the OBSERVED
        # admit (gap = resolution - admit), confirming it does NOT jointly
        # marginalise the continuous admit across edges.
        ar_seg = lrows[2]
        @test ar_seg.kind == :admit_resolution
        @test ar_seg.resolution == row.resolution - row.admit
    end
end

@testitem "andv index: declared double-interval-censored leaf (unchanged)" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: composed_distribution_model
    using DynamicPPL: @model, to_submodel, logjoint, VarInfo

    # The index branch is observed from the case's own exposure, so it keeps its
    # DECLARED double-interval censoring and right-truncation at its own
    # `obs_time` window. This guards that the sampled-origin change does not
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

# ===========================================================================
# Single `latent(tree)` wrapper: composed marginal == latent on the SAME rows
# ===========================================================================
#
# North-star tenet 4: converting a COMPOSED model from marginal to latent must be
# the single change of wrapping the composed object in `latent(...)`, on the SAME
# record rows, with the SAME model structure and priors. These tests prove the
# single wrapper is density-identical to the marginal for the composed node types,
# including the nested-Parallel bdbv shape (a nested-chain branch's first edge must
# stay DECLARED, matching the marginal `_nested_tree_logpdf`, not be forced bare)
# and the top-level Choose andv shape (the latent routes per record to the chosen
# alternative's latent form).

@testitem "single latent wrapper: nested-Parallel branch edge stays declared" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: latent, composed_distribution_model
    using DynamicPPL: @model, to_submodel, logjoint

    # A Parallel whose FIRST branch is a nested chain (onset -> admit -> resolution)
    # and whose second branch is a leaf (onset -> notif), the bdbv tree shape. The
    # MARGINAL routes this through `_nested_tree_logpdf`, which scores every edge
    # with its DECLARED censoring when both endpoints are observed (origin observed,
    # so all edges declared). The latent wrapper must match edge-for-edge: forcing
    # the branches BARE (the FLAT shared-origin Parallel rule) over-strips the
    # nested-chain first edge and the leaf branch, diverging by ~4e-4. With the
    # origin observed (onset = 0) NOTHING is sampled, so the latent logjoint is a
    # point density that must EQUAL the marginal to machine precision.
    dic(d) = double_interval_censored(d;
        primary_event = Uniform(0, 1), interval = 1.0)
    res = resolve(:death => (dic(Gamma(2.0, 3.5)), 0.3),
        :discharge => (dic(Gamma(1.0, 8.0)), 0.7))
    admit_path = sequential(:onset_admit => dic(Gamma(1.2, 3.0)),
        :admit_resolution => res)
    tree = compose((admit_path = admit_path,
        onset_notif = dic(Gamma(0.7, 20.0))))
    @test tree isa CensoredDistributions.Parallel

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    rows = [
        (onset = 0.0, admit = 4.0, death = 12.0, discharge = missing,
            notif = 18.0, branch_probs = (death = 0.3, discharge = 0.7)),
        (onset = 0.0, admit = 5.0, death = missing, discharge = 9.0,
            notif = 20.0, branch_probs = (death = 0.4, discharge = 0.6))]
    for row in rows
        marg = only(logjoint(demo(tree, [row]), (;)))
        lat = only(logjoint(demo(latent(tree), [row]), (;)))
        @test isapprox(marg, lat; atol = 1e-10)
    end
    # The whole batch (one `~` over both records) is also identical.
    marg_all = only(logjoint(demo(tree, rows), (;)))
    lat_all = only(logjoint(demo(latent(tree), rows), (;)))
    @test isapprox(marg_all, lat_all; atol = 1e-10)
end

@testitem "single latent wrapper: nested-Parallel sampled-admit integrates" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: latent, composed_distribution_model, event_names,
                                 Sequential, Parallel, Resolve
    using DynamicPPL: @model, to_submodel, logjoint, VarInfo

    # The bdbv-shaped nested Parallel with admit UNOBSERVED but death observed: the
    # latent samples the admission time; integrating it over the onset->admit window
    # must reproduce the marginal nested-Resolve scoring with a sampled anchor (the
    # bare onset->death convolution weighted by the death branch probability). This
    # is the capability the single wrapper adds over the marginal (which needs an
    # observed anchor), and integrating it must land on the same bare-convolution
    # term the equivalence invariant predicts.
    oa = Normal(0.4, 0.5)
    death_d = Gamma(2.0, 0.8)
    disch_d = LogNormal(0.6, 0.3)
    cfr = 0.3
    res = Resolve(:death => (death_d, cfr), :discharge => (disch_d, 1 - cfr))
    admit_path = Sequential((oa, res), (:onset_admit, :admit))
    leaf = Normal(2.0, 0.5)
    tree = Parallel((admit_path, leaf), (:admit_path, :onset_notif))
    en = event_names(tree)   # (:onset, :admit, :death, :discharge, :notif)
    @test tree isa CensoredDistributions.Parallel

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))
    conv = CensoredDistributions.convolve_distributions(oa, death_d)

    function latent_int(tdeath, tnotif; n = 50_000)
        xs = range(-6.0, tdeath - 1e-4; length = n)
        dx = step(xs)
        s = 0.0
        for a in xs
            row = NamedTuple{en}((0.0, a, tdeath, missing, tnotif))
            m = demo(latent(tree), [row])
            s += exp(logjoint(m, VarInfo(m))) * dx
        end
        return log(s)
    end

    for (tdeath, tnotif) in ((3.0, 2.5), (5.0, 1.8))
        # onset observed (0), notif observed leaf (bare branch, sampled-free origin
        # reference), death observed via the bare onset->death convolution.
        ref = logpdf(conv, tdeath) + log(cfr) + logpdf(leaf, tnotif)
        @test isapprox(latent_int(tdeath, tnotif), ref; atol = 5e-3)
    end
end

@testitem "single latent wrapper: top-level Resolve == marginal" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: latent, composed_distribution_model
    using DynamicPPL: @model, to_submodel, logjoint, VarInfo

    # A top-level Resolve has no origin latent of its own (outcomes hang off the
    # implicit origin reference), so `latent(resolve)` is density-identical to the
    # marginal resolve: same conditioned-outcome score, same branch probability.
    dic(d) = double_interval_censored(d;
        primary_event = Uniform(0, 1), interval = 1.0)
    res = resolve(:death => (dic(Gamma(2.0, 3.5)), 0.3),
        :discharge => (dic(Gamma(1.0, 8.0)), 0.7))

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    for row in (
        (death = 12.0, discharge = missing,
            branch_probs = (death = 0.3, discharge = 0.7)),
        (death = missing, discharge = 9.0,
            branch_probs = (death = 0.4, discharge = 0.6)))
        marg = logjoint(demo(res, row), VarInfo(demo(res, row)))
        lat = logjoint(demo(latent(res), row), VarInfo(demo(latent(res), row)))
        @test isapprox(marg, lat; atol = 1e-12)
    end
end

@testitem "single latent wrapper: top-level Choose routes to alt latent" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: latent, composed_distribution_model, event,
                                 event_names
    using DynamicPPL: @model, to_submodel, logjoint, VarInfo

    # A top-level Choose (the andv shape) routes a record to one alternative by its
    # selector, then scores that alternative's LATENT form. With each record fully
    # observed (nothing sampled) the latent wrapper is a point density that must
    # EQUAL the marginal Choose. The `index` alternative is a single-leaf compose
    # (Parallel) and the `sourced` alternative a two-edge Sequential with its
    # intermediate observed, so both alternatives' observed-observed edges stay
    # declared under the latent wrapper.
    inc_window = 1.0
    dd(d) = double_interval_censored(d; primary_event = Uniform(0, 1), interval = 1)
    index = compose((infection_onset = primary_censored(
        LogNormal(3.0, 0.3), Uniform(0, inc_window)),))
    sourced = sequential(:srconset_infection => dd(Normal(0.0, 1.0)),
        :infection_onset => dd(LogNormal(3.0, 0.3)))
    tree = choose(:index => index, :sourced => sourced)
    @test tree isa CensoredDistributions.Choose

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    rows = [(kind = :index, infection = 0.0, onset = 4.0),
        (kind = :sourced, srconset = 0.0, infection = 2.0, onset = 6.0),
        (kind = :index, infection = 0.0, onset = 5.0)]
    for row in rows
        marg = logjoint(demo(tree, row), VarInfo(demo(tree, row)))
        lat = logjoint(
            demo(latent(tree), row), VarInfo(demo(latent(tree), row)))
        @test isapprox(marg, lat; atol = 1e-10)
    end
end

@testitem "single latent wrapper: andv index leaf-latent integrates" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: latent, composed_distribution_model, event,
                                 event_names
    using DynamicPPL: @model, to_submodel, logjoint, VarInfo

    # The andv `index` alternative is a single-leaf compose whose origin (infection)
    # is UNOBSERVED: the marginal integrates the primary, the latent samples it.
    # Integrating the sampled primary over its window must reproduce the marginal
    # leaf logpdf (the leaf marginal == latent invariant, now reached through the
    # single composed wrapper).
    inc_window = 1.0
    idx = compose((infection_onset = primary_censored(
        LogNormal(3.0, 0.3), Uniform(0, inc_window)),))
    en = event_names(idx)   # (:infection, :onset)

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    function latent_int(onset; n = 40_000)
        s = 0.0
        for p in range(1e-6, inc_window - 1e-6; length = n)
            row = NamedTuple{en}((p, onset))
            m = demo(latent(idx), [row])
            s += exp(logjoint(m, VarInfo(m))) * (inc_window / n)
        end
        return log(s)
    end

    for onset in (4.0, 5.0)
        row = (infection = missing, onset = onset)
        marg = only(logjoint(demo(idx, [row]), (;)))
        @test isapprox(latent_int(onset), marg; atol = 5e-4)
    end
end

@testitem "single latent wrapper: plain-leaf and Choose-of-leaves" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: latent, composed_distribution_model
    using DynamicPPL: @model, to_submodel, logjoint, VarInfo

    # A `latent`-wrapped PLAIN leaf (a double-interval-censored pipeline, not a bare
    # primary-censored node) has no compositional latent to sample: its marginal
    # already integrates the primary analytically, so `latent(leaf)` is density-
    # identical to the marginal leaf. A top-level Choose whose alternatives are such
    # leaves routes each record to the chosen alternative's latent form, which must
    # therefore equal the marginal Choose. This guards the leaf-fallback method the
    # Choose routing relies on (and that a bare `latent(primary_censored)` keeps its
    # own primary-sampling, scored separately by the equivalence integral above).
    dic(d) = double_interval_censored(d;
        primary_event = Uniform(0, 1), interval = 1.0)
    leaf_a = dic(Gamma(2.0, 3.0))
    leaf_b = dic(Gamma(1.0, 5.0))

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    # Plain-leaf latent == marginal.
    for delay in (4.0, 6.0)
        row = (delay = delay,)
        marg = logjoint(demo(leaf_a, row), VarInfo(demo(leaf_a, row)))
        lat = logjoint(
            demo(latent(leaf_a), row), VarInfo(demo(latent(leaf_a), row)))
        @test isapprox(marg, lat; atol = 1e-10)
    end

    # Choose of plain leaves: latent routes to the chosen leaf's latent (== marginal).
    ch = choose(:a => leaf_a, :b => leaf_b)
    for row in ((kind = :a, delay = 6.0), (kind = :b, delay = 4.0))
        marg = logjoint(demo(ch, row), VarInfo(demo(ch, row)))
        lat = logjoint(demo(latent(ch), row), VarInfo(demo(latent(ch), row)))
        @test isapprox(marg, lat; atol = 1e-10)
    end
end
