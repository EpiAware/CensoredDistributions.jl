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
