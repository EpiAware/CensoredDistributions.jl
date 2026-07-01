# Regression for the genuine latent-variable capability (#722/#724).
#
# A latent-wrapped composer must sample the intermediate event time when the
# record leaves it unobserved, so the sampled latent event time then lives in the
# model's VarInfo and is visible in the posterior chain. A future refactor that
# silently turned latent back into a re-marginalisation (e.g. by reintroducing a
# scalar observed density on `Latent`) would drop that latent variable; these
# tests pin it so that regression fails loudly.

@testitem "latent composer samples the unobserved intermediate event" begin
    using Distributions
    using DynamicPPL: VarInfo, @varname
    using CensoredDistributions: composed_distribution_model, latent

    leaf(g) = double_interval_censored(
        g; primary_event = Uniform(0, 1), interval = 1.0)
    tree = sequential(:onset_admit => leaf(Gamma(2.0, 2.0)),
        :admit_death => leaf(Gamma(2.0, 3.5)))

    # Admission unobserved: the intermediate event time must be sampled, so the
    # VarInfo carries the latent event var `e[2]` (the onset origin is observed
    # at 0.0, the death terminal is observed data).
    missing_row = (onset = 0.0, admit = missing, death = 12.0)
    vi_missing = VarInfo(
        composed_distribution_model(latent(tree), missing_row))
    @test @varname(e[2]) in keys(vi_missing)

    # Admission observed: nothing intermediate is latent, so the VarInfo is empty
    # (every event is conditioned on as data). This is the contrast the latent
    # form must preserve: a missing intermediate samples, an observed one does
    # not.
    observed_row = (onset = 0.0, admit = 4.0, death = 12.0)
    vi_observed = VarInfo(
        composed_distribution_model(latent(tree), observed_row))
    @test isempty(keys(vi_observed))
end

@testitem "bdbv tree latent: unobserved admit sampled, not marginal-in-a-hat" begin
    using Distributions
    using DynamicPPL: VarInfo, @varname, @model, to_submodel, logjoint
    using CensoredDistributions: composed_distribution_model, latent,
                                 resolve, sequential, compose,
                                 double_interval_censored, Parallel

    # The published bdbv `delays` object (#724): a nested Parallel of an
    # onset -> admit -> {death, discharge} chain and an onset -> notif leaf, every
    # edge double-interval censored. When a resolved record leaves admission
    # UNOBSERVED, the latent form must SAMPLE the admission time, so the latent
    # event var lands in the VarInfo and the joint density depends on the sampled
    # value. A refactor that silently re-marginalised `latent` would drop that
    # variable (empty VarInfo) and return a value constant in the (now absent)
    # intermediate -- a marginal in a hat. These pin both against that regression.
    dic(d) = double_interval_censored(d;
        primary_event = Uniform(0, 1), interval = 1.0)
    res = resolve(:death => (dic(Gamma(2.0, 3.5)), 0.3),
        :discharge => (dic(Gamma(1.0, 8.0)), 0.7))
    admit_path = sequential(:onset_admit => dic(Gamma(1.2, 3.0)),
        :admit_resolution => res)
    tree = compose((admit_path = admit_path,
        onset_notif = dic(Gamma(0.7, 20.0))))
    @test tree isa Parallel

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))
    bp = (death = 0.3, discharge = 0.7)

    # Admission UNOBSERVED, death observed: the intermediate admission time is the
    # sampled latent, keyed `rec1.e[2]` (the second event slot of the first
    # record). The marginal nested-Resolve scorer needs an observed anchor, so
    # sampling this intermediate is a capability the latent form adds.
    miss = (onset = 0.0, admit = missing, death = 12.0, discharge = missing,
        notif = 18.0, branch_probs = bp)
    m_miss = composed_distribution_model(latent(tree), [miss])
    vi_miss = VarInfo(m_miss)
    @test @varname(rec1.e[2]) in keys(vi_miss)

    # Admission OBSERVED: every event is conditioned on as data, nothing
    # intermediate is latent, so the VarInfo is empty. This is the contrast the
    # latent form must preserve.
    obsv = (onset = 0.0, admit = 4.0, death = 12.0, discharge = missing,
        notif = 18.0, branch_probs = bp)
    @test isempty(keys(VarInfo(
        composed_distribution_model(latent(tree), [obsv]))))

    # With admission observed both endpoints of every edge are observed, so the
    # latent form keeps each declared censoring and is density-identical to the
    # marginal (the project marginal == latent invariant), exact.
    marg = only(logjoint(demo(tree, [obsv]), (;)))
    lat = only(logjoint(demo(latent(tree), [obsv]), (;)))
    @test isapprox(marg, lat; atol = 1e-10)

    # Not a marginal in a hat: with admission unobserved the latent joint is a
    # function of the sampled intermediate, so independent draws of `rec1.e[2]`
    # give different logjoints. A re-marginalised `latent` would carry no sampled
    # intermediate and return one constant value regardless of the draw.
    lj1 = logjoint(m_miss, VarInfo(m_miss))
    lj2 = logjoint(m_miss, VarInfo(m_miss))
    lj3 = logjoint(m_miss, VarInfo(m_miss))
    @test lj1 != lj2 || lj2 != lj3
end

@testitem "latent fit posterior contains the sampled intermediate latent" begin
    using Distributions, Turing
    using DynamicPPL: to_submodel
    using FlexiChains: VNChain, parameters
    using CensoredDistributions: composed_distribution_model, latent

    # A short NUTS fit of the same latent composer over records with the
    # intermediate unobserved must produce a chain whose parameters include the
    # per-record sampled latent event time, not just the delay parameters. This
    # is the headline latent-variable behaviour: the intermediate event is an
    # explicitly sampled latent in the posterior.
    leaf(g) = double_interval_censored(
        g; primary_event = Uniform(0, 1), interval = 1.0)
    tree = sequential(:onset_admit => leaf(Gamma(2.0, 2.0)),
        :admit_death => leaf(Gamma(2.0, 3.5)))

    rows = [(onset = 0.0, admit = missing, death = Float64(d))
            for d in (10, 12, 14)]

    @model function latent_fit(d, rows)
        obs ~ to_submodel(composed_distribution_model(d, rows))
    end

    chn = sample(latent_fit(latent(tree), rows), NUTS(), 20;
        chain_type = VNChain, progress = false)

    # One sampled latent event time per record appears in the chain, keyed by the
    # record prefix and the event slot (`obs.recN.e[2]`). The marginal form would
    # carry none of these.
    names = string.(parameters(chn))
    @test count(n -> occursin("e[2]", n), names) == length(rows)
end

@testitem "single-leaf latent(double_interval_censored) samples the primary" begin
    using Distributions
    using DynamicPPL: VarInfo, @varname
    using CensoredDistributions: composed_distribution_model, latent,
                                 PrimaryConditional, get_primary_event

    # The genuine single-leaf latent (#723): `latent(double_interval_censored)`
    # must SAMPLE the primary event (not analytically marginalise it), so a single
    # record realises the primary `p` in the VarInfo, and the joint is the primary
    # prior plus the interval-censored, truncated conditional secondary -- NOT the
    # bare-delay shift (stripping the interval was the bug).
    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0, 1)
    leaf = double_interval_censored(delay; primary_event = pe, interval = 1.0)
    ld = latent(leaf)

    # Single-record VarInfo carries the sampled primary `p`.
    vi = VarInfo(composed_distribution_model(ld, (report = 4.0,)))
    @test @varname(p) in keys(vi)

    # Joint = primary prior + conditional secondary, and the conditional keeps the
    # interval (so it differs from the bare-delay shift).
    p, y = 0.4, 4.0
    cond = PrimaryConditional(ld, p)
    @test logpdf(ld, [p, y]) ≈ logpdf(get_primary_event(ld), p) + logpdf(cond, y)
    @test logpdf(cond, y) != logpdf(delay, y - p)
end

@testitem "single-leaf latent fit realises the primary in the chain" begin
    using Distributions, Turing
    using DynamicPPL: to_submodel
    using FlexiChains: VNChain, parameters
    using CensoredDistributions: composed_distribution_model, latent

    # A short fit of `latent(double_interval_censored)` over single-event records
    # must put one sampled primary per record in the chain (`obs.recN.p`), the
    # evidence that the primary is a sampled latent, not marginalised.
    delay = LogNormal(1.5, 0.75)
    leaf = double_interval_censored(delay; primary_event = Uniform(0, 1),
        interval = 1.0)
    rows = [(report = Float64(r),) for r in (3, 5, 7)]

    @model function fit(d, rows)
        obs ~ to_submodel(composed_distribution_model(d, rows))
    end

    chn = sample(fit(latent(leaf), rows), NUTS(), 20;
        chain_type = VNChain, progress = false)
    names = string.(parameters(chn))
    @test count(n -> occursin(".p", n), names) == length(rows)
end
