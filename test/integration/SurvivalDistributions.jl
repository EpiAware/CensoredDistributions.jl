# Integration tests for SurvivalDistributions.jl families as composable delay
# leaves (issue #465).
#
# SurvivalDistributions.jl provides parametric delay families (LogLogistic,
# GeneralizedGamma, PowerGeneralizedWeibull, ExponentiatedWeibull) and a
# piecewise-constant hazard distribution. They subtype
# `ContinuousUnivariateDistribution` and define `logpdf`/`cdf`/`rand`/`params`,
# and (as of v0.1) already define `minimum`/`maximum` over `[0, Inf)` plus the
# generic `insupport`, so they satisfy the leaf contract (`<:
# UnivariateDistribution`) with NO package-side methods. These tests lock that
# in: the analytic families compose, censor and convolve as leaves with zero
# new code, and no weak-dep extension is needed (see the PR for that decision).
#
# The piecewise-constant hazard type is verified for the bare-leaf support and
# density contract only: its upstream `logcdf` throws in v0.1.1 (an unimported
# `log1mexp`), so it cannot yet route the numeric censoring quadrature, which
# the analytic families cover.

@testitem "SurvivalDistributions families satisfy the leaf interface" begin
    using CensoredDistributions
    using CensoredDistributions.TestUtils: test_interface
    import SurvivalDistributions as SD
    using Distributions: Uniform

    # Helper: double-interval-censor a leaf the way a reporting delay would be
    # observed (primary-event width + secondary interval).
    dic(x) = double_interval_censored(
        x; primary_event = Uniform(0, 1), interval = 1.0)

    # The four analytic families. They have no closed-form `mean`/`var`, so the
    # bare-leaf overall-moment check is skipped (`overall = :none`), exactly as
    # the package's own "bare censored leaf" fixture does.
    leaves = (
        "LogLogistic" => SD.LogLogistic(1.0, 2.0),
        "GeneralizedGamma" => SD.GeneralizedGamma(1.0, 1.5, 2.0),
        "PowerGeneralizedWeibull" => SD.PowerGeneralizedWeibull(1.0, 1.5, 2.0),
        "ExponentiatedWeibull" => SD.ExponentiatedWeibull(1.0, 1.5, 2.0)
    )

    for (nm, leaf) in leaves
        # Bare leaf: minimum/maximum/insupport, a finite logpdf on an in-support
        # draw, a monotone cdf in [0, 1], and params all hold.
        test_interface(leaf; name = "bare $nm", draw = 1.5,
            univariate = true, overall = :none)
        # Double-interval-censored leaf: the censoring stack scores and has a
        # monotone cdf, proving the family drops into the censoring path.
        test_interface(dic(leaf); name = "dic $nm", draw = 3.0,
            univariate = true, overall = :none)
    end
end

@testitem "SurvivalDistributions leaf inside a composer" begin
    using CensoredDistributions
    using CensoredDistributions.TestUtils: test_interface
    import SurvivalDistributions as SD
    using Distributions: Uniform, Gamma

    dic(x) = double_interval_censored(
        x; primary_event = Uniform(0, 1), interval = 1.0)

    # A Parallel of a SurvivalDistributions leaf and a stock Distributions leaf,
    # both double-interval-censored: the survival family composes alongside the
    # existing families with no new code. The moment views are skipped
    # (`overall = :none`, `latent_moments = false`): a censored leaf has no
    # closed-form mean, and the survival families define no `mean`/`var` either
    # (the generic `Statistics.mean` iterate-fallback errors), so the structural
    # checks (length/rand, logpdf, event layout) are what apply.
    # The in-support event draw is supplied explicitly as `[origin, branch1,
    # branch2]`: the package's `_insupport_event_draw` helper builds a draw from
    # `mean(core)`, which the survival families do not define, so a hand-built
    # in-support vector (positive gaps clear of each core's lower support edge)
    # is used instead. This is the layout the composer `logpdf` scores.
    par = Parallel(dic(SD.GeneralizedGamma(1.0, 1.5, 2.0)),
        dic(Gamma(2.0, 1.0)))
    test_interface(par; name = "Parallel survival+Gamma",
        draw = Vector{Union{Missing, Float64}}([0.0, 3.0, 3.0]),
        overall = :none, latent_moments = false, has_endpoint = false)

    # A Sequential chain whose first step is a SurvivalDistributions leaf: the
    # chain collapses to its observed total and scores event-by-event. The
    # overall moment is skipped for the same no-closed-form-mean reason. The
    # event draw is `[origin, onset_admit, admit_death]` with strictly
    # increasing times.
    seq = Sequential((dic(SD.LogLogistic(1.0, 2.0)), dic(Gamma(1.5, 2.0))),
        (:onset_admit, :admit_death))
    test_interface(seq; name = "Sequential survival-first",
        draw = Vector{Union{Missing, Float64}}([0.0, 3.0, 7.0]),
        path = (:onset_admit,), overall = :none, latent_moments = false)
end

@testitem "SurvivalDistributions primary_censored numeric CDF is monotone" begin
    using CensoredDistributions
    import SurvivalDistributions as SD
    using Distributions: Uniform, cdf, logcdf, logpdf, truncated, Normal
    using QuadGK: quadgk

    # A LogLogistic leaf with a truncated-Normal primary event forces the
    # numeric Gauss-Legendre quadrature path (no analytic
    # `primarycensored_cdf(::LogLogistic, ::Truncated, ...)` exists).
    leaf = SD.LogLogistic(1.0, 2.0)
    primary = truncated(Normal(0.5, 0.3), 0.0, 1.0)
    pc = primary_censored(leaf, primary)

    # Independent QuadGK reference for the primary-censored CDF, using the same
    # integrand the package's own numeric path does (see test/integration of
    # the Integrals.jl interface): integrate `cdf(delay, u) * pdf(primary, x - u)`
    # in `u` over `[max(x - max(primary), min(delay)), x - min(primary)]`.
    function reference_pc_cdf(delay, prim, x)
        lower = max(x - maximum(prim), minimum(delay))
        upper = x - minimum(prim)
        upper <= lower && return 0.0
        val, _ = quadgk(
            u -> exp(logcdf(delay, u) + logpdf(prim, x - u)), lower, upper)
        return clamp(val, 0.0, 1.0)
    end

    xs = range(0.0, 8.0; length = 9)
    cs = [cdf(pc, x) for x in xs]
    @test issorted(cs)
    @test all(c -> 0.0 - 1e-9 <= c <= 1.0 + 1e-9, cs)
    for x in (1.0, 2.0, 3.0, 5.0)
        @test cdf(pc, x) ≈ reference_pc_cdf(leaf, primary, x) atol=1e-6
    end
end

@testitem "SurvivalDistributions convolve_distributions constructs and scores" begin
    using CensoredDistributions
    import SurvivalDistributions as SD
    using Distributions: cdf, LogNormal

    # A convolution of a SurvivalDistributions leaf with a stock leaf: the sum
    # of two independent delays. The numeric convolution quadrature uses the
    # leaves' minimum/maximum, so this exercises the leaf-support contract.
    conv = convolve_distributions(
        SD.GeneralizedGamma(1.0, 1.5, 2.0), LogNormal(0.5, 0.4))
    cs = [cdf(conv, x) for x in range(0.0, 10.0; length = 11)]
    @test issorted(cs)
    @test all(c -> 0.0 - 1e-9 <= c <= 1.0 + 1e-9, cs)
end

@testitem "SurvivalDistributions piecewise-constant hazard leaf" begin
    using CensoredDistributions
    import SurvivalDistributions as SD
    using Distributions: cdf, logpdf, insupport

    # The piecewise-constant hazard distribution (e.g. from a Kaplan-Meier fit)
    # is a valid leaf on `[0, Inf)`: it has a monotone cdf and a finite logpdf,
    # the support contract the composers need. It lacks `params`, so it is not
    # run through the full `test_interface` checklist (which calls `params`).
    #
    # The numeric censoring path (`primary_censored`/`double_interval_censored`)
    # is NOT exercised here: SurvivalDistributions v0.1.1's
    # `logcdf(::PiecewiseConstantHazardDistribution)` references an unimported
    # `log1mexp` and throws (an upstream bug), and the quadrature integrand calls
    # `logcdf`. The analytic families above cover the censoring path; this leaf
    # is verified for the bare-leaf support/density contract it satisfies today.
    leaf = SD.PiecewiseConstantHazardDistribution(
        [1.0, 2.0, 3.0], [0.5, 0.3, 0.2])
    @test minimum(leaf) == 0.0
    @test maximum(leaf) == Inf
    @test insupport(leaf, 0.5)
    @test isfinite(logpdf(leaf, 1.0))
    cs = [cdf(leaf, x) for x in range(0.0, 6.0; length = 13)]
    @test issorted(cs)
    @test all(c -> 0.0 - 1e-9 <= c <= 1.0 + 1e-9, cs)
end
