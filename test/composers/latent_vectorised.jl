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
    d = select_branch(
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
