# Vectorised scoring for the LATENT representation. A whole table of latent
# records scores in a two-statement vectorised pair rather than a per-record
# `to_submodel` loop:
#
#   primaries ~ product_distribution(latent_primary_priors(d, rows))
#   @addlogprob! latent_observed_logpdf(d, rows, primaries)
#
# A mixed Select table (some marginal rows, some latent rows) works: marginal
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

@testitem "vectorised mixed Select table: marginal + latent rows" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, prefix, logjoint, @addlogprob!,
                      condition, @varname
    using CensoredDistributions: latent, latent_primary_priors,
                                 latent_observed_logpdf

    # An index case scored marginally, a sourced case scored latent.
    d = selecting(
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
    # chain row stacks TWO priors: the origin prior and the first DECLARED edge.
    e1 = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))
    e2 = primary_censored(Gamma(3.0, 1.5), Uniform(0, 1))
    chain = latent(Sequential((e1, e2), (:onset_admit, :admit_death)))

    rows = [(onset = missing, admit = missing, death = 6.0),
        (onset = missing, admit = missing, death = 9.0)]
    priors = latent_primary_priors(chain, rows)
    # Two rows, two latent values each, so four stacked priors.
    @test length(priors) == 4
    # The origin prior is the first edge's primary event; the second latent value
    # is the first DECLARED edge (the intermediate gap prior), matching the
    # per-record chain submodel and the marginal.
    @test priors[1] == get_primary_event(e1)
    @test priors[2] == e1
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

@testitem "vectorised mixed table: latent leaf + latent chain + marginal" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, prefix, logjoint, @addlogprob!,
                      condition, @varname
    using CensoredDistributions: latent, latent_primary_priors,
                                 latent_observed_logpdf

    # A Select over an index leaf (marginal), a sourced latent leaf, and a
    # two-edge latent chain. Each row routes by `:kind` to its alternative.
    e1 = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))
    e2 = primary_censored(Gamma(3.0, 1.5), Uniform(0, 1))
    d = selecting(
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
