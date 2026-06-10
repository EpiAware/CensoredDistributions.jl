# Latent ergonomics: a `Select` may carry a `latent`-wrapped alternative, and a
# `latent`-wrapped composer splits each record's events into observed (conditioned
# on their edge) and unobserved (sampled), driven by the row's missingness pattern
# rather than relying on the caller to condition each observed event by hand.

@testitem "Select holds a Latent alternative" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: latent

    # A Latent is a Multivariate distribution; it must compose as a Select
    # alternative so the index-vs-sourced split can route a sourced case to its
    # latent chain.
    d = select_branch(
        :index => primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)),
        :sourced => latent(primary_censored(Gamma(4.0, 1.5), Uniform(0, 1))))
    @test d isa CensoredDistributions.Select
    @test CensoredDistributions._n_alternatives(d) == 2
    @test CensoredDistributions._pick(d, :sourced) isa CensoredDistributions.Latent

    # A latent-wrapped Sequential also composes (the sourced chain).
    chain = Sequential(
        primary_censored(Normal(0.0, 1.0), Uniform(0, 1)),
        primary_censored(LogNormal(3.0, 0.3), Uniform(0, 1)))
    d2 = select_branch(:index => primary_censored(LogNormal(3.0, 0.3),
            Uniform(0, 1)),
        :sourced => latent(chain))
    @test CensoredDistributions._pick(d2, :sourced) isa
          CensoredDistributions.Latent
end

@testitem "Latent is composable" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: latent, _is_composable

    @test _is_composable(latent(primary_censored(Gamma(2.0, 1.0),
        Uniform(0, 1))))
    @test _is_composable(latent(Sequential(
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)),
        primary_censored(LogNormal(1.0, 0.5), Uniform(0, 1)))))
end

@testitem "Select routes a sourced record to its latent leaf branch" begin
    using CensoredDistributions, Distributions, Random
    using CensoredDistributions: latent, get_primary_event, get_dist
    using DynamicPPL: @model, to_submodel, logjoint, VarInfo, condition,
                      @varname

    d = select_branch(
        :index => primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)),
        :sourced => latent(primary_censored(Gamma(4.0, 1.5), Uniform(0, 1))))

    @model demo(dist, row) = obs ~ to_submodel(
        composed_distribution_model(dist, row))

    # An index record scores the marginal selected alternative directly.
    idx_lp = only(logjoint(demo(d, (kind = :index, delay = 3.0)), (;)))
    @test idx_lp ≈ logpdf(primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)), 3.0)

    # A sourced record routes to the latent leaf model: the primary `p` is
    # sampled inside, so a fully-observed delay leaves one latent in the VarInfo.
    vi = VarInfo(demo(d, (kind = :sourced, delay = 4.0)))
    @test length(keys(vi)) == 1
end

@testitem "latent Sequential conditions observed events, samples missing" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: latent, get_primary_event, get_dist
    using DynamicPPL: @model, to_submodel, logjoint, VarInfo, condition,
                      @varname

    seq = Sequential(
        primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))
    lseq = latent(seq)

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    # Onset (origin) + death observed, the intermediate admit unobserved. The
    # observed onset/death CONDITION on their edge rather than being re-sampled as
    # free latents; only the unobserved admit is a genuine latent.
    o, dd = 0.3, 5.4
    row = (onset = o, admit = missing, death = dd)
    vi = VarInfo(demo(lseq, row))
    # One genuine latent: the unobserved admit (e[2]).
    @test length(keys(vi)) == 1

    # The row-driven likelihood is in the joint with no manual conditioning of the
    # observed events: it equals the origin prior at the observed onset plus each
    # observed edge's conditional at the latent admit. Condition on the sampled
    # admit to pin its value and compare against the hand-written decomposition.
    e1 = 2.1
    cond = condition(demo(lseq, row), (@varname(obs.e[2]) => e1,))
    manual = logpdf(get_primary_event(seq.components[1]), o) +
             logpdf(get_dist(seq.components[1]), e1 - o) +
             logpdf(get_dist(seq.components[2]), dd - e1)
    @test logjoint(cond, (;)) ≈ manual
end

@testitem "latent Sequential fully observed conditions every event" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: latent, get_primary_event, get_dist
    using DynamicPPL: @model, to_submodel, logjoint, VarInfo

    seq = Sequential(
        primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))
    lseq = latent(seq)

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    # Every event observed: no latents at all, and the log-density equals the
    # full decomposition with no conditioning needed.
    o, a, dd = 0.3, 2.1, 5.4
    row = (onset = o, admit = a, death = dd)
    vi = VarInfo(demo(lseq, row))
    @test length(keys(vi)) == 0

    lp = only(logjoint(demo(lseq, row), (;)))
    manual = logpdf(get_primary_event(seq.components[1]), o) +
             logpdf(get_dist(seq.components[1]), a - o) +
             logpdf(get_dist(seq.components[2]), dd - a)
    @test lp ≈ manual
end

@testitem "latent Sequential fully missing samples the whole path" begin
    using CensoredDistributions, Distributions, Random
    using CensoredDistributions: latent
    using DynamicPPL: @model, to_submodel, VarInfo

    seq = Sequential(
        primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))
    lseq = latent(seq)

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    row = (onset = missing, admit = missing, death = missing)
    vi = VarInfo(demo(lseq, row))
    @test length(keys(vi)) == 3

    @model function gen(d)
        inner = to_submodel(
            composed_distribution_model(d,
                (a = missing, b = missing, c = missing)), false)
        path ~ inner
        return path
    end
    Random.seed!(11)
    draws = [gen(lseq)() for _ in 1:200]
    @test all(p -> p[1] >= 0 && p[2] >= p[1] && p[3] >= p[2], draws)
end

@testitem "latent Parallel conditions observed branches, samples missing" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: latent, get_primary_event, get_dist
    using DynamicPPL: @model, to_submodel, logjoint, VarInfo, condition,
                      @varname

    par = Parallel(
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)),
        primary_censored(LogNormal(1.0, 0.5), Uniform(0, 1)))
    lpar = latent(par)

    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    # Origin + first branch observed, second branch missing: the shared origin is
    # observed (conditions), branch 1 conditions, branch 2 samples -> one latent.
    o, y1 = 0.3, 2.5
    row = (origin = o, y1 = y1, y2 = missing)
    vi = VarInfo(demo(lpar, row))
    @test length(keys(vi)) == 1

    e2 = 3.1
    cond = condition(demo(lpar, row), (@varname(obs.e[3]) => e2,))
    manual = logpdf(get_primary_event(par.components[1]), o) +
             logpdf(get_dist(par.components[1]), y1 - o) +
             logpdf(get_dist(par.components[2]), e2 - o)
    @test logjoint(cond, (;)) ≈ manual
end

@testitem "latent Select sourced chain conditions onset, samples infection" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: latent, get_primary_event, get_dist
    using DynamicPPL: @model, to_submodel, logjoint, VarInfo, condition,
                      @varname

    # The andv sourced branch: a two-edge latent chain
    # source_onset -> infection (latent) -> case_onset, the intermediate
    # infection sampled, the case onset observed and conditioned.
    delta = Normal(0.0, 1.0)
    inc = LogNormal(3.0, 0.3)
    chain = Sequential(
        primary_censored(delta, Uniform(0, 1)),
        primary_censored(inc, Uniform(0, 1)))
    d = select_branch(
        :index => primary_censored(inc, Uniform(0, 1)),
        :sourced => latent(chain))

    @model demo(dist, row) = obs ~ to_submodel(
        composed_distribution_model(dist, row))

    # source_onset observed (= 0), infection latent, case_onset observed.
    src_onset, case_onset = 0.0, 20.0
    row = (kind = :sourced, source_onset = src_onset, infection = missing,
        onset = case_onset)
    vi = VarInfo(demo(d, row))
    # Two latents: the origin source_onset (observed below) is conditioned, the
    # infection is sampled. The origin here is observed, so only infection is the
    # genuine latent -> one latent.
    @test length(keys(vi)) == 1

    inf = 5.0
    cond = condition(demo(d, row), (@varname(obs.obs.e[2]) => inf,))
    manual = logpdf(get_primary_event(chain.components[1]), src_onset) +
             logpdf(get_dist(chain.components[1]), inf - src_onset) +
             logpdf(get_dist(chain.components[2]), case_onset - inf)
    @test logjoint(cond, (;)) ≈ manual
end
