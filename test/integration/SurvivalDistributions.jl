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
#
# The CENSORED GeneralizedGamma path additionally routes its leaf CDF/survival
# through the package's AD-safe `_gamma_cdf` helper via the
# `CensoredDistributionsSurvivalDistributions` weak-dep extension (the GG inner
# `Gamma`'s stock `logccdf` → `StatsFuns._gammalogccdf` has no Dual/Tracked
# method). The gradient-parity testitem at the end of this file locks that in.
# `LogLogistic` cannot carry AD-tracked parameters at all: its constructor stores
# `Logistic{eltype(X)}` where `eltype(Logistic{Dual}) == Float64`, so the struct
# is forced to `LogLogistic{Float64}` and `convert`ing the `Dual` Logistic into
# it throws — an upstream bug no package-side routing can fix, so the survival
# AD coverage is GeneralizedGamma only.

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

@testitem "SurvivalDistributions GeneralizedGamma censored AD parity" begin
    using CensoredDistributions
    import SurvivalDistributions as SD
    using Distributions: logpdf, logccdf, Uniform
    import ForwardDiff

    # The censored GeneralizedGamma path routes its leaf survival/CDF through the
    # package's AD-safe `_gamma_cdf` (the SurvivalDistributions extension), so the
    # interval-, primary- and double-interval-censored pipelines differentiate
    # w.r.t. the three GG parameters (shape σ, scale ν, power γ). Without the
    # extension the inner Gamma's `logccdf` hits `StatsFuns._gammalogccdf`, which
    # has no `ForwardDiff.Dual` method and errors.
    #
    # Central finite-difference gradient of `f` at `θ` (reference of the primal).
    function fd_grad(f, θ; h = 1e-6)
        g = similar(θ)
        for i in eachindex(θ)
            θp = copy(θ)
            θp[i] += h
            θm = copy(θ)
            θm[i] -= h
            g[i] = (f(θp) - f(θm)) / (2h)
        end
        return g
    end

    θ₀ = [1.0, 1.5, 2.0]
    obs = [0.5, 1.2, 2.5, 3.8, 5.1]
    obs_int = [0.0, 1.0, 2.0, 3.0, 4.0]
    # Interior support points for the exact-primitive FD reference: deep-tail
    # `logccdf` values (e.g. at t = 5.1, where t^γ ≈ 26) lose precision to
    # catastrophic cancellation in a central difference, so the FD reference is
    # unstable there even though the AD gradient is exact. Points near the mode
    # keep the FD reference well-conditioned.
    obs_in = [0.4, 0.7, 1.0, 1.3]

    # Exact parity on ALL FOUR AD-safe primitives the fix adds. Each
    # GeneralizedGamma `_*_ad_safe` routes through `_gamma_cdf` with NO internal
    # finite-differencing, so its ForwardDiff gradient must match a central FD
    # reference of the same primal to tight tolerance. These are the actual
    # functions the censoring integrands call; matching FD here proves the gradient
    # is correct, not merely finite. `_logccdf_ad_safe`/`_cdf_ad_safe` and their
    # complements `_ccdf_ad_safe`/`_logcdf_ad_safe` are all covered so the survival
    # and CDF directions both lock in.
    safe_lccdf(θ) = sum(
        t -> CensoredDistributions._logccdf_ad_safe(
            SD.GeneralizedGamma(θ[1], θ[2], θ[3]), t),
        obs_in)
    safe_cdf(θ) = sum(
        t -> CensoredDistributions._cdf_ad_safe(
            SD.GeneralizedGamma(θ[1], θ[2], θ[3]), t),
        obs_in)
    safe_ccdf(θ) = sum(
        t -> CensoredDistributions._ccdf_ad_safe(
            SD.GeneralizedGamma(θ[1], θ[2], θ[3]), t),
        obs_in)
    safe_lcdf(θ) = sum(
        t -> CensoredDistributions._logcdf_ad_safe(
            SD.GeneralizedGamma(θ[1], θ[2], θ[3]), t),
        obs_in)
    for (nm,
        f) in (("_logccdf_ad_safe", safe_lccdf),
        ("_cdf_ad_safe", safe_cdf), ("_ccdf_ad_safe", safe_ccdf),
        ("_logcdf_ad_safe", safe_lcdf))
        gf = ForwardDiff.gradient(f, θ₀)
        gref = fd_grad(f, θ₀; h = 1e-5)
        @test all(isfinite, gf)
        @test gf≈gref rtol=1e-4 atol=1e-6
    end

    # Interval-censored pipeline: its CDF differences are exact (no internal FD),
    # so its ForwardDiff gradient also matches a central FD reference exactly.
    ic(θ) = sum(
        x -> logpdf(
            interval_censored(SD.GeneralizedGamma(θ[1], θ[2], θ[3]), 1.0), x),
        obs_int)
    gf_ic = ForwardDiff.gradient(ic, θ₀)
    @test all(isfinite, gf_ic)
    @test gf_ic≈fd_grad(ic, θ₀) rtol=1e-4 atol=1e-6

    # Numeric quadrature pipelines (primary-/double-interval-censored over GG).
    # Their `logpdf` finite-differences the CDF internally with a hardcoded
    # `h = 1e-8`, so an external central FD of the whole logpdf is not a reliable
    # reference (the two FD steps compound). ForwardDiff's Dual propagation
    # through the internal FD is the exact gradient of the package's own logpdf,
    # and is cross-checked against ReverseDiff / Mooncake / Enzyme in `test/ad`.
    # Here we lock that these paths differentiate at all (finite gradient) — the
    # property that was broken before the AD-safe routing.
    pc(θ) = sum(
        x -> logpdf(
            primary_censored(
                SD.GeneralizedGamma(θ[1], θ[2], θ[3]), Uniform(0.0, 1.0)),
            x),
        obs)
    dic(θ) = sum(
        x -> logpdf(
            double_interval_censored(
                SD.GeneralizedGamma(θ[1], θ[2], θ[3]);
                primary_event = Uniform(0.0, 1.0), upper = 10.0, interval = 1.0),
            x),
        obs)
    for (nm, f) in (("primary_censored", pc), ("double_interval_censored", dic))
        g = ForwardDiff.gradient(f, θ₀)
        @test all(isfinite, g)
        @test !all(iszero, g)
    end
end

@testitem "SurvivalDistributions GeneralizedGamma bare logcdf is AD-safe" begin
    using CensoredDistributions
    import SurvivalDistributions as SD
    using Distributions: logcdf, cdf
    import ForwardDiff

    # A DIRECT `logcdf(GeneralizedGamma(θ...), t)` (not via a censoring
    # pipeline) must differentiate cleanly. SurvivalDistributions defines no
    # `logcdf` for GeneralizedGamma, so the generic `Distributions.logcdf`
    # evaluates the inner `Gamma`'s `logcdf` -> `StatsFuns._gammalogcdf`, which
    # has no `ForwardDiff.Dual` method and STRIPS the Dual / throws. The
    # extension routes the public `logcdf` through the AD-safe
    # `_logcdf_ad_safe` (`_gamma_cdf`-backed) so the bare call differentiates
    # w.r.t. the three GG parameters. Locks in #46 Task 2b.
    function fd_grad(f, θ; h = 1e-6)
        g = similar(θ)
        for i in eachindex(θ)
            θp = copy(θ)
            θp[i] += h
            θm = copy(θ)
            θm[i] -= h
            g[i] = (f(θp) - f(θm)) / (2h)
        end
        return g
    end

    θ₀ = [1.0, 1.5, 2.0]
    obs_in = [0.4, 0.7, 1.0, 1.3]
    f(θ) = sum(t -> logcdf(SD.GeneralizedGamma(θ[1], θ[2], θ[3]), t), obs_in)
    g = ForwardDiff.gradient(f, θ₀)
    @test all(isfinite, g)
    @test g≈fd_grad(f, θ₀; h = 1e-5) rtol=1e-4 atol=1e-6

    # The AD-routed primal equals the package's `_logcdf_ad_safe` value (and so
    # the stock `log(cdf)`), i.e. the routing changes the gradient path only.
    for t in obs_in
        gg = SD.GeneralizedGamma(θ₀[1], θ₀[2], θ₀[3])
        @test logcdf(gg, t) ≈ log(cdf(gg, t))
    end
end
