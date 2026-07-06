# Empirical test of the latent truncation normaliser Z.
#
# The latent secondary conditional scores f(y | p) = mass(y | p) / Z. Two
# choices of Z are compared:
#   * MARGINAL Z (current code, `_secondary_logZ`): the primary-integrated
#     truncation mass P(lower <= P + delay <= upper), a constant shared across p.
#   * PER-P Z (`_secondary_logZ_perp`): F_D(upper - p) - F_D(lower - p), the
#     delay-cdf mass of the shifted window for this particular primary p.
#
# The question: which normaliser makes the primary-marginalised conditional
#   m(y) = E_p[f(y | p)] = int g(p) f(y | p) dp
# equal the analytic `double_interval_censored` marginal? Three configurations:
#   (A) plain prior g(p)              + MARGINAL Z   (current code)
#   (B) plain prior g(p)              + PER-P Z
#   (C) reweighted prior g(p)Z_p/Z_m + PER-P Z
#
# Run: julia --project=test test/experiments/latent_truncation_z.jl

using CensoredDistributions
using Distributions
using Random
using Printf

const CD = CensoredDistributions

# Trapezoidal integral of samples `f` on ordered nodes `x`.
function trapz(x::AbstractVector, f::AbstractVector)
    s = zero(eltype(f))
    @inbounds for i in 1:(length(x) - 1)
        s += (x[i + 1] - x[i]) * (f[i + 1] + f[i]) / 2
    end
    return s
end

# One experiment: return the analytic marginal pmf/pdf over `ygrid` and the
# three marginalised-conditional reconstructions (A, B, C), plus diagnostics.
function run_case(; delay, primary_event, lower, upper, interval, ygrid,
        np = 20_001)
    dpipe = double_interval_censored(delay; primary_event = primary_event,
        lower = lower, upper = upper, interval = interval)

    # Primary prior nodes and density g(p) over the primary support.
    plo, phi = minimum(primary_event), maximum(primary_event)
    ps = collect(range(plo, phi; length = np))
    g = pdf.(primary_event, ps)

    # Per-p conditional (built once per p; reused across y).
    conds = [CD._conditional(dpipe, p) for p in ps]

    # Per-p truncation mass Z_p(p) and the marginal Z (integral of g*Z_p).
    Zp = [exp(CD._secondary_logZ_perp(c)) for c in conds]
    Zmarg_int = trapz(ps, g .* Zp)
    # Marginal Z straight from the primary-censored cdf (code path A uses this).
    Zmarg_code = exp(CD._secondary_logZ(conds[1]))
    gtilde = (g .* Zp) ./ Zmarg_int          # reweighted (selection) prior

    analytic = [exp(logpdf(dpipe, y)) for y in ygrid]
    mA = similar(analytic)
    mB = similar(analytic)
    mC = similar(analytic)
    for (j, y) in enumerate(ygrid)
        fA = [exp(logpdf(c, y)) for c in conds]              # marginal Z
        fB = [exp(CD._secondary_logpdf_perp(c, y)) for c in conds]  # per-p Z
        mA[j] = trapz(ps, g .* fA)
        mB[j] = trapz(ps, g .* fB)
        mC[j] = trapz(ps, gtilde .* fB)
    end
    return (; ygrid, analytic, mA, mB, mC, Zmarg_int, Zmarg_code, ps, g,
        gtilde, Zp)
end

maxabs(a, b) = maximum(abs.(a .- b))

function report(name, r)
    println("\n=== $name ===")
    @printf("  marginal Z: code path = %.8f, integral of g*Z_p = %.8f\n",
        r.Zmarg_code, r.Zmarg_int)
    println("    (match => primary-censored cdf IS the marginal mass)")
    println("  Z_p range over primary support: " *
            @sprintf("[%.4f, %.4f] (spread %.1f%%)", minimum(r.Zp),
        maximum(r.Zp), 100 * (maximum(r.Zp) / minimum(r.Zp) - 1)))
    println()
    println("   y        analytic     A:plain+margZ   B:plain+perpZ   " *
            "C:rewt+perpZ")
    for j in eachindex(r.ygrid)
        @printf("  %6.2f   %10.6f    %10.6f      %10.6f      %10.6f\n",
            r.ygrid[j], r.analytic[j], r.mA[j], r.mB[j], r.mC[j])
    end
    @printf("  sum        %8.6f     %8.6f       %8.6f       %8.6f\n",
        sum(r.analytic), sum(r.mA), sum(r.mB), sum(r.mC))
    dA = maxabs(r.mA, r.analytic)
    dB = maxabs(r.mB, r.analytic)
    dC = maxabs(r.mC, r.analytic)
    println()
    @printf("  max|A - analytic| = %.3e\n", dA)
    @printf("  max|B - analytic| = %.3e\n", dB)
    @printf("  max|C - analytic| = %.3e\n", dC)
    return (; dA, dB, dC)
end

# --- Simulation cross-check: the selection-truncation DGP -------------------
# Draw (p, delay), keep pairs whose secondary p+delay lands in [lower, upper],
# floor to the interval. The empirical pmf is the true observed marginal; the
# accepted primaries reveal the selection reweighting of g(p).
function simulate(; delay, primary_event, lower, upper, interval, ygrid,
        n = 20_000_000, seed = 20240706)
    rng = MersenneTwister(seed)
    counts = Dict{Float64, Int}()
    kept = 0
    psum = 0.0
    for _ in 1:n
        p = rand(rng, primary_event)
        total = p + rand(rng, delay)
        (lower <= total <= upper) || continue
        kept += 1
        psum += p
        y = interval === nothing ? total :
            CD.floor_to_interval(total, interval)
        counts[y] = get(counts, y, 0) + 1
    end
    # Simulated draws are floored to interval starts (integers for width 1),
    # so key the empirical pmf by each grid point's own interval start.
    key(y) = interval === nothing ? y : CD.floor_to_interval(y, interval)
    emp = [get(counts, key(y), 0) / kept for y in ygrid]
    return (; emp, kept, n, pmean = psum / kept)
end

println(repeat("#", 72))
println("# Latent truncation-Z experiment")
println(repeat("#", 72))

# Realistic short-delay scenario. Lower bound sits near the delay mode so a
# one-day primary shift swings Z_p materially; window truncates both tails.
delay = LogNormal(1.0, 0.5)          # median 2.72, mode 2.11, mean 3.08
primary_event = Uniform(0, 1)        # primary within a day
lower, upper = 2.0, 6.0

println("\ndelay = LogNormal(1.0, 0.5); primary = Uniform(0,1); " *
        "window = [$lower, $upper]")

# --- Interval-censored (daily) ---------------------------------------------
ic_grid = [2.5, 3.5, 4.5, 5.5]       # interval-start representatives (width 1)
ric = run_case(; delay, primary_event, lower, upper, interval = 1.0,
    ygrid = ic_grid)
dic = report("interval-censored (interval = 1, daily)", ric)
sim = simulate(; delay, primary_event, lower, upper, interval = 1.0,
    ygrid = ic_grid)
reweighted = abs(sim.pmean - 0.5) > 0.005 ? "REWEIGHTED by selection" :
             "~unchanged"
@printf("\n  simulation (%d kept of %d): mean accepted p = %.4f\n",
    sim.kept, sim.n, sim.pmean)
println("    plain-prior mean is 0.5 => accepted primaries are $reweighted")
@printf("  max|simulated pmf - analytic| = %.3e (MC noise)\n",
    maxabs(sim.emp, ric.analytic))

# --- Continuous truncated (no interval), fine grid for a sharp density -------
ct_grid = collect(range(lower + 0.01, upper - 0.01; length = 41))
rct = run_case(; delay, primary_event, lower, upper, interval = nothing,
    ygrid = ct_grid)
println("\n=== continuous truncated (interval = nothing), 41-point grid ===")
@printf("  density integral over window: analytic=%.6f A=%.6f B=%.6f C=%.6f\n",
    trapz(ct_grid, rct.analytic), trapz(ct_grid, rct.mA),
    trapz(ct_grid, rct.mB), trapz(ct_grid, rct.mC))
@printf("  max|A - analytic| = %.3e\n", maxabs(rct.mA, rct.analytic))
@printf("  max|B - analytic| = %.3e   <- per-p + plain prior\n",
    maxabs(rct.mB, rct.analytic))
@printf("  max|C - analytic| = %.3e\n", maxabs(rct.mC, rct.analytic))

# --- Scenario panel: how the B bias scales with the primary window ----------
# The bias of (B) is a covariance across p between the conditional shape and
# 1/Z_p; it grows as the primary window widens relative to the delay/truncation
# scale. Daily primary censoring is the mild end; a multi-day primary window is
# the strong end. (A) and (C) track the analytic marginal throughout.
scenarios = [
    ("daily primary, window [2,6]",
        LogNormal(1.0, 0.5), Uniform(0, 1), 2.0, 6.0, 1.0),
    ("daily primary, tight window [2,4]",
        LogNormal(1.0, 0.5), Uniform(0, 1), 2.0, 4.0, 1.0),
    ("2-day primary, window [1,5]",
        LogNormal(0.5, 0.6), Uniform(0, 2), 1.0, 5.0, 1.0),
    ("3-day primary, tight window [1,4]",
        LogNormal(0.5, 0.6), Uniform(0, 3), 1.0, 4.0, 1.0)
]
println("\n", repeat("=", 72))
println("SCENARIO PANEL (interval-censored). ",
    "dB_rel = max|B-analytic| / max(analytic pmf)")
@printf("%-38s %10s %10s %10s %8s\n", "scenario", "dA", "dB", "dC", "dB_rel")
panel = map(scenarios) do (name, dl, pe, lo, up, iv)
    starts = collect((floor(lo / iv) * iv + iv / 2):iv:(up - iv / 2))
    r = run_case(; delay = dl, primary_event = pe, lower = lo, upper = up,
        interval = iv, ygrid = starts)
    dA = maxabs(r.mA, r.analytic)
    dB = maxabs(r.mB, r.analytic)
    dC = maxabs(r.mC, r.analytic)
    relB = dB / maximum(r.analytic)
    @printf("%-38s %10.2e %10.2e %10.2e %7.2f%%\n", name, dA, dB, dC,
        100 * relB)
    (; name, dA, dB, dC, relB)
end

# --- Data-driven verdict ----------------------------------------------------
# Confirmed when A and C sit at the trapezoid floor while B stands orders of
# magnitude above it (a real structural bias, not integration noise).
println("\n", repeat("#", 72))
floorAC = maximum(max(p.dA, p.dC) for p in panel)
minB = minimum(p.dB for p in panel)
aOK = dic.dA < 1e-6
cOK = dic.dC < 1e-6
bBreaks = dic.dB > 1000 * max(dic.dA, dic.dC)
println("VERDICT:")
println("  (A) plain prior + marginal Z  == analytic marginal   : $aOK")
println("  (C) reweighted prior + per-p Z == analytic marginal  : $cOK")
println("  (B) plain prior + per-p Z  != analytic marginal      : $bBreaks")
@printf("  numerical floor max(dA,dC) over panel = %.2e; min dB = %.2e\n",
    floorAC, minB)
if aOK && cOK && bBreaks
    println("=> HYPOTHESIS CONFIRMED. Per-p Z is same-in-expectation ONLY with")
    println("   a selection-reweighted primary prior (C). With the plain prior")
    println("   the latent sampler uses, per-p Z (B) biases the marginal; the")
    println("   bias is small for daily data and grows with the primary window.")
else
    println("=> HYPOTHESIS NOT CONFIRMED as stated; see numbers above.")
end
println(repeat("#", 72))
