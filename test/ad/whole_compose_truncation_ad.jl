# AD coverage for the whole-compose conv-to-last-observed right-truncation
# denominator (#366). With an OBSERVED intermediate and a per-record horizon the
# Sequential record scores a factorised numerator over a single
# `-logcdf(conv-to-last-observed, window)` denominator. The gradient w.r.t. the
# delay params must flow: the denominator convolves the leaf cores and evaluates
# a CDF at the (constant) window, all on the differentiated param type.

@testitem "whole-compose truncation gradient: ForwardDiff (#366)" tags=[
    :ad, :forwarddiff] begin
    using CensoredDistributions, Distributions
    using ADTypes: AutoForwardDiff
    using DifferentiationInterface: gradient
    using ForwardDiff: ForwardDiff

    # One observed-intermediate record and one endpoint-observed record, both
    # right-truncated at their own horizon, so the gradient exercises both the
    # factorised (observed-intermediate) numerator and the conv-to-last-observed
    # denominator. LogNormal delays keep the truncation `logcdf` off the Gamma
    # shape-derivative path (`_gamma_inc`), matching the established AD-safe
    # right-truncation fixtures (`ADFixtures`).
    # events: [origin, mid, obs]; horizon per record.
    evs = [Vector{Union{Missing, Float64}}([0.0, 2.0, 5.0]),
        Vector{Union{Missing, Float64}}([0.5, missing, 7.0])]
    Ds = [8.0, 9.0]

    # θ = [inc μ, inc σ, delta μ, delta σ].
    function f(θ)
        inc = primary_censored(LogNormal(θ[1], θ[2]), Uniform(0, 1))
        delta = primary_censored(LogNormal(θ[3], θ[4]), Uniform(0, 1))
        seq = Sequential((inc, delta), (:onset_mid, :mid_obs))
        return sum(eachindex(evs)) do i
            CensoredDistributions.event_logpdf(seq, evs[i]; horizon = Ds[i])
        end
    end

    θ = [1.0, 0.5, 0.5, 0.4]
    val = f(θ)
    @test isfinite(val)

    g = gradient(f, AutoForwardDiff(), θ)
    @test g isa AbstractVector && length(g) == 4 && all(isfinite, g)
    # Every parameter feeds both a numerator factor and the denominator, so each
    # partial is non-zero.
    @test all(!=(0), g)
end
