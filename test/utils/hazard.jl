# Tests for the discrete-time reporting-hazard layer: PMF <-> hazard round-trip,
# logit-scale effect modification, and the per-(reference, report) expected-count
# matrix with right-truncation.

@testitem "delay_hazard / hazard_to_pmf round-trip a normalised PMF" begin
    using CensoredDistributions, Distributions

    # A discretised, truncated delay PMF sums to one over its grid.
    pmf = CensoredDistributions._delay_pmf(
        truncated(LogNormal(1.2, 0.5); upper = 10.0), 10, 1.0)
    pmf = pmf ./ sum(pmf)

    h = CensoredDistributions.delay_hazard(pmf)
    @test all(0 .<= h .<= 1)
    @test h[end] ≈ 1                       # maximum-delay constraint

    back = CensoredDistributions.hazard_to_pmf(h)
    @test back ≈ pmf
    @test sum(back) ≈ 1
end

@testitem "delay_hazard handles an exhausted survival without NaN" begin
    using CensoredDistributions

    # A PMF that puts all mass early: later survival hits zero.
    pmf = [0.6, 0.4, 0.0, 0.0]
    h = CensoredDistributions.delay_hazard(pmf)
    @test all(isfinite, h)
    @test h[1] ≈ 0.6
    @test h[2] ≈ 1.0                       # 0.4 / (1 - 0.6)
    @test h[end] ≈ 1.0
end

@testitem "apply_hazard_effects with zero effects is the identity" begin
    using CensoredDistributions, Distributions

    pmf = CensoredDistributions._delay_pmf(
        truncated(LogNormal(1.0, 0.6); upper = 8.0), 8, 1.0)
    pmf = pmf ./ sum(pmf)

    modified = CensoredDistributions.apply_hazard_effects(pmf, zeros(length(pmf)))
    @test modified ≈ pmf
end

@testitem "apply_hazard_effects shifts mass and stays normalised" begin
    using CensoredDistributions, Distributions

    pmf = CensoredDistributions._delay_pmf(
        truncated(LogNormal(1.2, 0.5); upper = 10.0), 10, 1.0)
    pmf = pmf ./ sum(pmf)

    # A positive early-delay effect speeds reporting: more mass at short delays.
    effects = vcat(fill(1.5, 3), zeros(length(pmf) - 3))
    faster = CensoredDistributions.apply_hazard_effects(pmf, effects)

    @test sum(faster) ≈ 1
    # Cumulative mass by delay 2 is higher under the faster hazard.
    @test sum(faster[1:3]) > sum(pmf[1:3])
    @test all(faster .>= 0)
end

@testitem "apply_hazard_effects rejects a mismatched effect length" begin
    using CensoredDistributions

    @test_throws DimensionMismatch CensoredDistributions.apply_hazard_effects(
        [0.5, 0.3, 0.2], [0.1, 0.2])
end

@testitem "reference_report_matrix rows recover expected when untruncated" begin
    using CensoredDistributions, Distributions

    expected = fill(100.0, 12)
    pmf = CensoredDistributions._delay_pmf(
        truncated(LogNormal(1.1, 0.5); upper = 7.0), 7, 1.0)
    pmf = pmf ./ sum(pmf)

    M = CensoredDistributions.reference_report_matrix(expected, pmf; now = nothing)
    @test size(M) == (12, 8)
    # Each row sums to its expected final count (PMF sums to one).
    @test all(t -> isapprox(sum(M[t, :]), expected[t]; rtol = 1e-8), 1:12)
end

@testitem "reference_report_matrix right-truncates beyond now" begin
    using CensoredDistributions, Distributions

    expected = fill(50.0, 10)
    pmf = CensoredDistributions._delay_pmf(
        truncated(LogNormal(1.0, 0.5); upper = 6.0), 6, 1.0)
    pmf = pmf ./ sum(pmf)
    now = 10

    M = CensoredDistributions.reference_report_matrix(expected, pmf; now = now)
    # Cells with report date t + (d - 1) > now are zero.
    for t in 1:10, d in 1:size(M, 2)

        report_date = t + (d - 1)
        if report_date > now
            @test M[t, d] == 0
        end
    end
    # The most recent reference date sees only delay-0 reports.
    @test M[10, 1] > 0
    @test all(M[10, 2:end] .== 0)
end

@testitem "reference_report_matrix reference effect varies the delay profile" begin
    using CensoredDistributions, Distributions

    expected = fill(100.0, 6)
    pmf = CensoredDistributions._delay_pmf(
        truncated(LogNormal(1.2, 0.5); upper = 8.0), 8, 1.0)
    pmf = pmf ./ sum(pmf)

    # Later reference dates report faster (positive hazard shift).
    ref = collect(range(-1.0, 1.0; length = 6))
    M = CensoredDistributions.reference_report_matrix(expected, pmf;
        reference_effects = ref, now = nothing)

    # Faster reporting -> more mass at delay 0 for later reference dates.
    @test M[6, 1] > M[1, 1]
    # Still normalised per row.
    @test all(t -> isapprox(sum(M[t, :]), 100.0; rtol = 1e-8), 1:6)
end

@testitem "reference_report_matrix report effect tracks the report date" begin
    using CensoredDistributions, Distributions

    expected = fill(100.0, 8)
    pmf = CensoredDistributions._delay_pmf(
        truncated(LogNormal(1.0, 0.5); upper = 7.0), 7, 1.0)
    pmf = pmf ./ sum(pmf)

    # A weekly report effect: boost reporting on report dates ≡ 1 (mod 7).
    dow = [s -> (mod1(s, 7) == 1 ? 1.0 : 0.0)][1]
    M = CensoredDistributions.reference_report_matrix(expected, pmf;
        report_effect = dow, now = nothing)
    @test size(M) == (8, 8)
    @test all(t -> isapprox(sum(M[t, :]), 100.0; rtol = 1e-8), 1:8)
end

@testitem "hazard layer is AD-safe through expected and effects" begin
    using CensoredDistributions, Distributions
    using ForwardDiff

    pmf = CensoredDistributions._delay_pmf(
        truncated(LogNormal(1.2, 0.5); upper = 8.0), 8, 1.0)
    pmf = pmf ./ sum(pmf)

    # Differentiate the observed-so-far total w.r.t. a log-expected level and a
    # reference-effect scale, as a fit would.
    function observed_total(theta)
        loglam, eff = theta[1], theta[2]
        expected = fill(exp(loglam), 8)
        ref = eff .* collect(range(-1.0, 1.0; length = 8))
        M = CensoredDistributions.reference_report_matrix(expected, pmf;
            reference_effects = ref, now = 8)
        return sum(M)
    end

    g = ForwardDiff.gradient(observed_total, [log(100.0), 0.3])
    @test all(isfinite, g)
end
