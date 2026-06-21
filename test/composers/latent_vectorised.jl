# Vectorised scoring for the LATENT representation. A whole table of latent
# records scores in a two-statement vectorised pair rather than a per-record
# `to_submodel` loop:
#
#   primaries ~ product_distribution(latent_primary_priors(d, rows))
#   @addlogprob! latent_observed_logpdf(d, rows, primaries)
#
# A mixed Choose table (some marginal rows, some latent rows) works: marginal
# rows score through their marginal record logpdf, latent rows through the
# vectorised primary/observed pair.

@testitem "latent_primary_priors stacks one prior per latent row" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: latent, latent_primary_priors,
                                 get_primary_event

    d = latent(primary_censored(Gamma(4.0, 1.5), Uniform(0, 1)))
    rows = [(delay = 3.0,), (delay = 5.0,), (delay = 2.0,)]
    priors = latent_primary_priors(d, rows)
    @test length(priors) == 3
    @test all(p -> p == get_primary_event(d), priors)
    # The stacked priors are the input to a single product_distribution `~`.
    @test product_distribution(priors) isa Distribution
end

@testitem "vectorised latent leaf equals the per-record latent logjoint" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, prefix, logjoint, @addlogprob!,
                      condition, @varname
    using CensoredDistributions: latent, latent_primary_priors,
                                 latent_observed_logpdf

    d = latent(primary_censored(Gamma(4.0, 1.5), Uniform(0, 1)))
    rows = [(delay = 3.0,), (delay = 5.0,), (delay = 2.0,)]

    # The new VECTORISED latent model: sample all primaries at once, then
    # condition all observed events at once.
    @model function vectorised(d, rows)
        primaries ~ product_distribution(latent_primary_priors(d, rows))
        @addlogprob! latent_observed_logpdf(d, rows, primaries)
        return primaries
    end

    # The per-record reference: each row in its own prefixed latent submodel.
    @model function per_record(d, rows)
        n = length(rows)
        parts = Vector(undef, n)
        for i in 1:n
            parts[i] ~ to_submodel(
                prefix(composed_distribution_model(d, rows[i]),
                    Symbol("r", i)), false)
        end
    end

    # Pin the same set of latent primaries in both models and compare logjoint.
    ps = [0.3, 0.6, 0.2]
    cv = condition(vectorised(d, rows), (@varname(primaries) => ps,))
    cp = condition(per_record(d, rows),
        (@varname(r1.p) => ps[1], @varname(r2.p) => ps[2],
            @varname(r3.p) => ps[3]))
    @test logjoint(cv, (;)) ≈ logjoint(cp, (;))
end

@testitem "vectorised latent matches the marginal where equivalent" begin
    using CensoredDistributions, Distributions, Random
    using CensoredDistributions: latent, latent_primary_priors,
                                 latent_observed_logpdf

    # Integrating out the sampled primaries of the vectorised latent must
    # recover the marginal log-density. Check by Monte-Carlo: averaging exp of
    # the latent observed conditional over draws approximates the marginal pdf.
    pc = primary_censored(Gamma(4.0, 1.5), Uniform(0, 1))
    d = latent(pc)
    rows = [(delay = 4.0,)]
    prior = latent_primary_priors(d, rows)
    Random.seed!(42)
    N = 200_000
    samples = [exp(latent_observed_logpdf(d, rows, [rand(prior[1])]))
               for _ in 1:N]
    mc = sum(samples) / N
    @test isapprox(mc, exp(logpdf(pc, 4.0)); rtol = 0.05)
end

@testitem "vectorised mixed Choose table: marginal + latent rows" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, prefix, logjoint, @addlogprob!,
                      condition, @varname
    using CensoredDistributions: latent, latent_primary_priors,
                                 latent_observed_logpdf

    # An index case scored marginally, a sourced case scored latent.
    d = choose(
        :index => primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)),
        :sourced => latent(primary_censored(Gamma(4.0, 1.5), Uniform(0, 1))))

    rows = [(kind = :index, delay = 3.0),
        (kind = :sourced, delay = 5.0),
        (kind = :index, delay = 2.0),
        (kind = :sourced, delay = 7.0)]

    # Vectorised: stack the latent primaries (sourced rows only), condition all.
    @model function vectorised(d, rows)
        primaries ~ product_distribution(latent_primary_priors(d, rows))
        @addlogprob! latent_observed_logpdf(d, rows, primaries)
        return primaries
    end

    # Only the two sourced rows carry a latent primary.
    @test length(latent_primary_priors(d, rows)) == 2

    # Per-record reference mixing marginal (index) and latent (sourced) rows.
    @model function per_record(d, rows)
        n = length(rows)
        parts = Vector(undef, n)
        for i in 1:n
            parts[i] ~ to_submodel(
                prefix(composed_distribution_model(d, rows[i]),
                    Symbol("r", i)), false)
        end
    end

    ps = [0.4, 0.7]
    cv = condition(vectorised(d, rows), (@varname(primaries) => ps,))
    # sourced rows are r2 (ps[1]) and r4 (ps[2]); index rows have no latent.
    cp = condition(per_record(d, rows),
        (@varname(r2.obs.p) => ps[1], @varname(r4.obs.p) => ps[2]))
    @test logjoint(cv, (;)) ≈ logjoint(cp, (;))
end

@testitem "latent_primary_priors stacks one prior per latent edge" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: latent, latent_primary_priors,
                                 get_primary_event

    # A two-edge latent chain: each row's latent variables are the origin draw
    # E_0 and the first gap (E_1 - E_0); the terminal E_2 conditions. So each
    # chain row stacks TWO priors: the origin prior and the first edge's BARE core.
    e1 = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))
    e2 = primary_censored(Gamma(3.0, 1.5), Uniform(0, 1))
    chain = latent(Sequential((e1, e2), (:onset_admit, :admit_death)))

    rows = [(onset = missing, admit = missing, death = 6.0),
        (onset = missing, admit = missing, death = 9.0)]
    priors = latent_primary_priors(chain, rows)
    # Two rows, two latent values each, so four stacked priors.
    @test length(priors) == 4
    # The origin prior is the first edge's primary event; the second latent value
    # is the first edge's BARE core: the endpoint-observed chain samples the
    # origin and the intermediate, so the intermediate gap is the bare delay (no
    # primary smear), matching the per-record chain submodel and the marginal.
    @test priors[1] == get_primary_event(e1)
    @test priors[2] == CensoredDistributions._bare_latent_edge(e1)
    @test product_distribution(priors) isa Distribution
end

@testitem "vectorised latent chain equals the per-record chain logjoint" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, prefix, logjoint, @addlogprob!,
                      condition, @varname
    using CensoredDistributions: latent, latent_primary_priors,
                                 latent_observed_logpdf

    e1 = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))
    e2 = primary_censored(Gamma(3.0, 1.5), Uniform(0, 1))
    chain = latent(Sequential((e1, e2), (:onset_admit, :admit_death)))

    # Endpoint-observed chain rows: origin + intermediate latent, terminal data.
    rows = [(onset = missing, admit = missing, death = 6.0),
        (onset = missing, admit = missing, death = 9.0)]

    @model function vectorised(d, rows)
        primaries ~ product_distribution(latent_primary_priors(d, rows))
        @addlogprob! latent_observed_logpdf(d, rows, primaries)
        return primaries
    end

    @model function per_record(d, rows)
        n = length(rows)
        parts = Vector(undef, n)
        for i in 1:n
            parts[i] ~ to_submodel(
                prefix(composed_distribution_model(d, rows[i]),
                    Symbol("r", i)), false)
        end
    end

    # Per-record samples E_0 (e[1]) and E_1 (e[2]); the vectorised samples the
    # origin draw and the first gap (E_1 - E_0). Pin matching values: the gap is
    # E_1 - E_0, so the two parameterisations score identically (shift Jacobian
    # is 1).
    e0_1, e1_1 = 0.4, 2.3
    e0_2, e1_2 = 0.6, 3.1
    ps = [e0_1, e1_1 - e0_1, e0_2, e1_2 - e0_2]
    cv = condition(vectorised(chain, rows), (@varname(primaries) => ps,))
    cp = condition(per_record(chain, rows),
        (@varname(r1.e[1]) => e0_1, @varname(r1.e[2]) => e1_1,
            @varname(r2.e[1]) => e0_2, @varname(r2.e[2]) => e1_2))
    @test logjoint(cv, (;)) ≈ logjoint(cp, (;))
end

@testitem "vectorised latent chain floors the shift on an interval edge" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, prefix, logjoint, @addlogprob!,
                      condition, @varname
    using CensoredDistributions: latent, latent_primary_priors,
                                 latent_observed_logpdf

    # A SINGLE-edge chain whose edge carries SECONDARY interval censoring
    # (double_censored). The origin is the latent primary; the terminal is
    # observed. The per-record path floors the sampled origin to the edge's
    # interval; the vectorised path used to reconstruct from the CONTINUOUS
    # origin without flooring, diverging for an interval edge. Floored and
    # continuous shifts disagree here, so this guards the flooring.
    e1 = double_interval_censored(Gamma(2.0, 1.0); primary_event = Uniform(0, 1),
        interval = 1.0)
    chain = latent(Sequential((e1,), (:onset_admit,)))
    rows = [(onset = missing, admit = 6.3), (onset = missing, admit = 9.1)]

    @model function vectorised(d, rows)
        primaries ~ product_distribution(latent_primary_priors(d, rows))
        @addlogprob! latent_observed_logpdf(d, rows, primaries)
        return primaries
    end
    @model function per_record(d, rows)
        n = length(rows)
        parts = Vector(undef, n)
        for i in 1:n
            parts[i] ~ to_submodel(
                prefix(composed_distribution_model(d, rows[i]),
                    Symbol("r", i)), false)
        end
    end

    # Each row's only latent is its origin draw; pin FRACTIONAL origins so the
    # floor changes the scored gap (floor(0.7) = 0, floor(0.4) = 0).
    o1, o2 = 0.7, 0.4
    cv = condition(vectorised(chain, rows), (@varname(primaries) => [o1, o2],))
    cp = condition(per_record(chain, rows),
        (@varname(r1.e[1]) => o1, @varname(r2.e[1]) => o2))
    @test logjoint(cv, (;)) ≈ logjoint(cp, (;))

    # The vectorised score equals the FLOORED-shift edge score, NOT the
    # continuous-shift score (the earlier bug).
    floored = logpdf(e1, 6.3 - floor(o1)) + logpdf(e1, 9.1 - floor(o2))
    continuous = logpdf(e1, 6.3 - o1) + logpdf(e1, 9.1 - o2)
    @test latent_observed_logpdf(chain, rows, [o1, o2]) ≈ floored
    @test !isapprox(floored, continuous)
end

@testitem "vectorised mixed table: latent leaf + latent chain + marginal" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, prefix, logjoint, @addlogprob!,
                      condition, @varname
    using CensoredDistributions: latent, latent_primary_priors,
                                 latent_observed_logpdf

    # A Choose over an index leaf (marginal), a sourced latent leaf, and a
    # two-edge latent chain. Each row routes by `:kind` to its alternative.
    e1 = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))
    e2 = primary_censored(Gamma(3.0, 1.5), Uniform(0, 1))
    d = choose(
        :index => primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)),
        :sourced => latent(primary_censored(Gamma(4.0, 1.5), Uniform(0, 1))),
        :chain => latent(Sequential((e1, e2), (:onset_admit, :admit_death))))

    rows = [(kind = :index, delay = 3.0),
        (kind = :sourced, delay = 5.0),
        (kind = :chain, onset = missing, admit = missing, death = 6.0)]

    @model function vectorised(d, rows)
        primaries ~ product_distribution(latent_primary_priors(d, rows))
        @addlogprob! latent_observed_logpdf(d, rows, primaries)
        return primaries
    end

    @model function per_record(d, rows)
        n = length(rows)
        parts = Vector(undef, n)
        for i in 1:n
            parts[i] ~ to_submodel(
                prefix(composed_distribution_model(d, rows[i]),
                    Symbol("r", i)), false)
        end
    end

    # Latent values in row order: sourced leaf (1), then chain (origin + gap, 2).
    p_src = 0.5
    e0, e1v = 0.3, 2.0
    ps = [p_src, e0, e1v - e0]
    @test length(latent_primary_priors(d, rows)) == 3

    cv = condition(vectorised(d, rows), (@varname(primaries) => ps,))
    cp = condition(per_record(d, rows),
        (@varname(r2.obs.p) => p_src,
            @varname(r3.obs.e[1]) => e0, @varname(r3.obs.e[2]) => e1v))
    @test logjoint(cv, (;)) ≈ logjoint(cp, (;))
end

@testitem "latent_primary_priors returns a concretely-typed vector" begin
    using CensoredDistributions, Distributions, Test
    using CensoredDistributions: latent, latent_primary_priors, choose,
                                 Sequential

    # The homogeneous latent leaf is the common case: every prior is the same
    # primary-event type, so the return type is the concrete element vector and
    # `latent_primary_priors` is `@inferred`-clean. This keeps
    # `product_distribution(latent_primary_priors(...))` type-stable so Mooncake
    # compiles one tight gradient rule (fast cold start) rather than a broad rule
    # off a runtime-narrowed `Vector{Any}`.
    d = latent(primary_censored(Gamma(4.0, 1.5), Uniform(0, 1)))
    rows = [(delay = 3.0,), (delay = 5.0,)]
    p = @inferred latent_primary_priors(d, rows)
    @test p isa Vector{Uniform{Float64}}
    rt = only(Base.return_types(latent_primary_priors, (typeof(d), typeof(rows))))
    @test isconcretetype(rt)

    # A mixed Choose table with one (homogeneous) latent alternative stays
    # concretely typed too: the marginal alternative contributes no priors.
    ds = choose(
        :index => primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)),
        :sourced => latent(primary_censored(Gamma(4.0, 1.5), Uniform(0, 1))))
    srows = [(kind = :index, delay = 3.0), (kind = :sourced, delay = 5.0)]
    ps = @inferred latent_primary_priors(ds, srows)
    @test ps isa Vector{Uniform{Float64}}

    # The empty case (all-marginal table) returns a correctly-typed EMPTY vector,
    # so the call site can skip the `primaries ~ ...` statement without a
    # `Vector{Any}`/`Union{}` regression.
    erows = [(kind = :index, delay = 3.0), (kind = :index, delay = 2.0)]
    ep = @inferred latent_primary_priors(ds, erows)
    @test ep isa Vector{Uniform{Float64}}
    @test isempty(ep)

    # A heterogeneous chain (origin primary + bare edge core, different types)
    # still works: the return type widens to the promotion, not an error.
    e1 = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))
    e2 = primary_censored(Gamma(3.0, 1.5), Uniform(0, 1))
    chain = latent(Sequential((e1, e2), (:onset_admit, :admit_death)))
    crows = [(onset = missing, admit = missing, death = 6.0)]
    cp = @inferred latent_primary_priors(chain, crows)
    @test length(cp) == 2
    @test product_distribution(cp) isa Distribution
end
