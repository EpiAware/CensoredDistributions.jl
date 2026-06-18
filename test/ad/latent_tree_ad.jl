# AD coverage for the marginal -> latent wrapper over a composed tree.
#
# `latent_segments(tree)` + `latent_records(tree, rows)` feed the vectorised
# latent path. A latent fit differentiates `latent_observed_logpdf(segments,
# table, primaries)` wrt the leaf delay PARAMETERS (riding in `segments`), so the
# scorer must be ForwardDiff-safe. Full Mooncake-reverse coverage is exercised
# end-to-end by the bdbv tutorial's NUTS fit with `AutoMooncake` (the main test
# env has Turing + Mooncake; this `:ad` env does not), so this file guards the
# ForwardDiff parameter gradient.

@testitem "latent wrapper parameter gradient: ForwardDiff" tags=[
    :ad, :forwarddiff] begin
    using CensoredDistributions, Distributions
    using ADTypes: AutoForwardDiff
    using DifferentiationInterface: gradient
    using ForwardDiff: ForwardDiff
    using CensoredDistributions: latent_observed_logpdf

    # A resolved record with admission observed and a notification, the bdbv
    # shape: the gradient flows through the gap delays' shape parameters.
    rows = [(onset = 0.0, admit = 4.0, death = 12.0, discharge = missing,
        notif = 18.0, branch_probs = (death = 0.3, discharge = 0.7))]

    function nll(θ)
        dic(d) = double_interval_censored(d;
            primary_event = Uniform(0, 1), interval = 1.0)
        tree = compose((
            admit_path = sequential(
                :onset_admit => dic(Gamma(θ[1], 3.0)),
                :admit_resolution => resolve(
                    :death => (dic(Gamma(θ[2], 3.5)), 0.3),
                    :discharge => (dic(Gamma(1.0, 8.0)), 0.7))),
            onset_notif = dic(Gamma(θ[3], 20.0))))
        seg = latent_segments(tree)
        tab = latent_records(tree, rows)
        return -latent_observed_logpdf(seg, tab, [0.3, 0.4, 0.5])
    end

    θ = [1.2, 2.0, 0.7]
    g = gradient(nll, AutoForwardDiff(), θ)
    @test all(isfinite, g)
    # The shape parameters of the OBSERVED gap delays (onset->admit, admit->death,
    # onset->notif) all enter the score, so the gradient is non-trivial.
    @test any(!iszero, g)
end
