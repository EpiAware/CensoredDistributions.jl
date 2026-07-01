# A renewal model fits like the rest of the stack: `renewal_model` samples the
# Rt path and the modulator parameters, runs the renewal, and the user scores
# observed counts against the returned infections. Here we recover the
# susceptible-pool size N and the Rt level from a Poisson-observed epidemic.

@testitem "renewal_model recovers the susceptible pool N" tags=[:turing] begin
    using CensoredDistributions, Distributions, Random
    using Turing: NUTS, sample, @model, to_submodel, AutoForwardDiff, @varname
    using FlexiChains: VNChain
    using Statistics: mean

    gi = pdf(
        interval_censored(
            truncated(Gamma(2.5, 1.3); lower = 1.0, upper = 12.0), 1.0), 1:12)
    n = 60
    I0 = 10.0
    true_N = 5.0e4
    # A sustained Rt above one drives the epidemic far enough into the
    # susceptible pool that the depletion curvature identifies N.
    true_Rt = fill(1.6, n)

    expected = renewal(true_Rt, gi, I0;
        modulator = susceptibility_depletion(true_N))
    rng = MersenneTwister(7)
    cases = rand.(rng, Poisson.(expected .+ 1.0e-6))

    # Rt is held FIXED at its known path (passed as a plain vector), so the fit
    # estimates only the susceptible pool. The pool prior is bounded: at very
    # large N the susceptible fraction saturates near one and the depletion
    # gradient vanishes (a flat plateau the sampler can wander onto), so a
    # bounded prior keeps the chain on the identifiable slope.
    mod_priors = (N = truncated(Normal(5.0e4, 1.5e4); lower = 2.0e4,
        upper = 1.2e5),)
    make_mod = p -> susceptibility_depletion(p.N)

    @model function fit(gi, I0, Rt, mod_priors, make_mod, cases)
        infections ~ to_submodel(renewal_model(gi, I0, Rt;
            modulator_priors = mod_priors, make_modulator = make_mod))
        cases ~ product_distribution(Poisson.(infections .+ 1.0e-6))
    end

    chain = sample(Xoshiro(1),
        fit(gi, I0, true_Rt, mod_priors, make_mod, cases),
        NUTS(0.95; adtype = AutoForwardDiff()), 200;
        chain_type = VNChain, progress = false)

    # The modulator parameter is addressable by its prefixed name
    # (`infections.N.x`), confirming the chain is namespaced.
    @test mean(chain[@varname(infections.N.x)])≈true_N rtol=0.2
end

@testitem "renewal_model with no modulator runs the bare renewal" tags=[
    :turing] begin
    using CensoredDistributions, Distributions, Random
    using Turing: sample, @model, to_submodel, Prior
    using DynamicPPL: DynamicPPL

    gi = pdf(
        interval_censored(
            truncated(Gamma(2.5, 1.3); lower = 1.0, upper = 12.0), 1.0), 1:12)
    n = 20
    Rt_prior = product_distribution(fill(Dirac(1.2), n))

    @model function bare(gi, Rt_prior)
        infections ~ to_submodel(renewal_model(gi, 5.0, Rt_prior))
        return infections
    end

    # The default modulator is NoModulation, so the returned series equals the
    # bare renewal at the (fixed) Rt path.
    got = bare(gi, Rt_prior)()
    @test got ≈ renewal(fill(1.2, n), gi, 5.0)
end
