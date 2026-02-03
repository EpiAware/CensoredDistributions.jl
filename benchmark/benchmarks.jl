using BenchmarkTools
using CensoredDistributions
using Distributions

const SUITE = BenchmarkGroup()

# Test point for evaluation
const TEST_X = 2.5
const TEST_P = 0.5

# ============================================================================
# PrimaryCensored Benchmarks
# ============================================================================

SUITE["PrimaryCensored"] = BenchmarkGroup()

# --- Gamma + Uniform (analytical and numerical) ---

SUITE["PrimaryCensored"]["Gamma+Uniform"] = BenchmarkGroup()

let
    delay = Gamma(2.0, 1.0)
    primary = Uniform(0, 1)

    # Analytical (default)
    d_analytical = primary_censored(delay, primary)
    SUITE["PrimaryCensored"]["Gamma+Uniform"]["analytical"] = BenchmarkGroup()
    SUITE["PrimaryCensored"]["Gamma+Uniform"]["analytical"]["cdf"] = @benchmarkable cdf($d_analytical, $TEST_X)
    SUITE["PrimaryCensored"]["Gamma+Uniform"]["analytical"]["pdf"] = @benchmarkable pdf($d_analytical, $TEST_X)
    SUITE["PrimaryCensored"]["Gamma+Uniform"]["analytical"]["logpdf"] = @benchmarkable logpdf(
        $d_analytical, $TEST_X)
    SUITE["PrimaryCensored"]["Gamma+Uniform"]["analytical"]["rand"] = @benchmarkable rand($d_analytical)

    # Numerical (force_numeric=true)
    d_numeric = primary_censored(delay, primary; force_numeric = true)
    SUITE["PrimaryCensored"]["Gamma+Uniform"]["numerical"] = BenchmarkGroup()
    SUITE["PrimaryCensored"]["Gamma+Uniform"]["numerical"]["cdf"] = @benchmarkable cdf($d_numeric, $TEST_X)
    SUITE["PrimaryCensored"]["Gamma+Uniform"]["numerical"]["pdf"] = @benchmarkable pdf($d_numeric, $TEST_X)
    SUITE["PrimaryCensored"]["Gamma+Uniform"]["numerical"]["logpdf"] = @benchmarkable logpdf(
        $d_numeric, $TEST_X)
    SUITE["PrimaryCensored"]["Gamma+Uniform"]["numerical"]["rand"] = @benchmarkable rand($d_numeric)
end

# --- LogNormal + Uniform (analytical and numerical) ---

SUITE["PrimaryCensored"]["LogNormal+Uniform"] = BenchmarkGroup()

let
    delay = LogNormal(1.5, 0.75)
    primary = Uniform(0, 1)

    # Analytical (default)
    d_analytical = primary_censored(delay, primary)
    SUITE["PrimaryCensored"]["LogNormal+Uniform"]["analytical"] = BenchmarkGroup()
    SUITE["PrimaryCensored"]["LogNormal+Uniform"]["analytical"]["cdf"] = @benchmarkable cdf(
        $d_analytical, $TEST_X)
    SUITE["PrimaryCensored"]["LogNormal+Uniform"]["analytical"]["pdf"] = @benchmarkable pdf(
        $d_analytical, $TEST_X)
    SUITE["PrimaryCensored"]["LogNormal+Uniform"]["analytical"]["logpdf"] = @benchmarkable logpdf(
        $d_analytical, $TEST_X)
    SUITE["PrimaryCensored"]["LogNormal+Uniform"]["analytical"]["rand"] = @benchmarkable rand($d_analytical)

    # Numerical (force_numeric=true)
    d_numeric = primary_censored(delay, primary; force_numeric = true)
    SUITE["PrimaryCensored"]["LogNormal+Uniform"]["numerical"] = BenchmarkGroup()
    SUITE["PrimaryCensored"]["LogNormal+Uniform"]["numerical"]["cdf"] = @benchmarkable cdf($d_numeric, $TEST_X)
    SUITE["PrimaryCensored"]["LogNormal+Uniform"]["numerical"]["pdf"] = @benchmarkable pdf($d_numeric, $TEST_X)
    SUITE["PrimaryCensored"]["LogNormal+Uniform"]["numerical"]["logpdf"] = @benchmarkable logpdf(
        $d_numeric, $TEST_X)
    SUITE["PrimaryCensored"]["LogNormal+Uniform"]["numerical"]["rand"] = @benchmarkable rand($d_numeric)
end

# --- Weibull + Uniform (analytical and numerical) ---

SUITE["PrimaryCensored"]["Weibull+Uniform"] = BenchmarkGroup()

let
    delay = Weibull(2.0, 3.0)
    primary = Uniform(0, 1)

    # Analytical (default)
    d_analytical = primary_censored(delay, primary)
    SUITE["PrimaryCensored"]["Weibull+Uniform"]["analytical"] = BenchmarkGroup()
    SUITE["PrimaryCensored"]["Weibull+Uniform"]["analytical"]["cdf"] = @benchmarkable cdf(
        $d_analytical, $TEST_X)
    SUITE["PrimaryCensored"]["Weibull+Uniform"]["analytical"]["pdf"] = @benchmarkable pdf(
        $d_analytical, $TEST_X)
    SUITE["PrimaryCensored"]["Weibull+Uniform"]["analytical"]["logpdf"] = @benchmarkable logpdf(
        $d_analytical, $TEST_X)
    SUITE["PrimaryCensored"]["Weibull+Uniform"]["analytical"]["rand"] = @benchmarkable rand($d_analytical)

    # Numerical (force_numeric=true)
    d_numeric = primary_censored(delay, primary; force_numeric = true)
    SUITE["PrimaryCensored"]["Weibull+Uniform"]["numerical"] = BenchmarkGroup()
    SUITE["PrimaryCensored"]["Weibull+Uniform"]["numerical"]["cdf"] = @benchmarkable cdf($d_numeric, $TEST_X)
    SUITE["PrimaryCensored"]["Weibull+Uniform"]["numerical"]["pdf"] = @benchmarkable pdf($d_numeric, $TEST_X)
    SUITE["PrimaryCensored"]["Weibull+Uniform"]["numerical"]["logpdf"] = @benchmarkable logpdf(
        $d_numeric, $TEST_X)
    SUITE["PrimaryCensored"]["Weibull+Uniform"]["numerical"]["rand"] = @benchmarkable rand($d_numeric)
end

# --- Exponential + Uniform (numerical only - no analytical solution) ---

SUITE["PrimaryCensored"]["Exponential+Uniform"] = BenchmarkGroup()

let
    delay = Exponential(2.0)
    primary = Uniform(0, 1)

    # Exponential has no analytical solution, so both should use numerical
    d = primary_censored(delay, primary)
    SUITE["PrimaryCensored"]["Exponential+Uniform"]["numerical"] = BenchmarkGroup()
    SUITE["PrimaryCensored"]["Exponential+Uniform"]["numerical"]["cdf"] = @benchmarkable cdf($d, $TEST_X)
    SUITE["PrimaryCensored"]["Exponential+Uniform"]["numerical"]["pdf"] = @benchmarkable pdf($d, $TEST_X)
    SUITE["PrimaryCensored"]["Exponential+Uniform"]["numerical"]["logpdf"] = @benchmarkable logpdf($d, $TEST_X)
    SUITE["PrimaryCensored"]["Exponential+Uniform"]["numerical"]["rand"] = @benchmarkable rand($d)
end

# --- Gamma + Exponential (numerical only - non-Uniform primary) ---

SUITE["PrimaryCensored"]["Gamma+Exponential"] = BenchmarkGroup()

let
    delay = Gamma(2.0, 1.0)
    primary = Exponential(0.5)

    # Non-Uniform primary forces numerical
    d = primary_censored(delay, primary)
    SUITE["PrimaryCensored"]["Gamma+Exponential"]["numerical"] = BenchmarkGroup()
    SUITE["PrimaryCensored"]["Gamma+Exponential"]["numerical"]["cdf"] = @benchmarkable cdf($d, $TEST_X)
    SUITE["PrimaryCensored"]["Gamma+Exponential"]["numerical"]["pdf"] = @benchmarkable pdf($d, $TEST_X)
    SUITE["PrimaryCensored"]["Gamma+Exponential"]["numerical"]["logpdf"] = @benchmarkable logpdf($d, $TEST_X)
    SUITE["PrimaryCensored"]["Gamma+Exponential"]["numerical"]["rand"] = @benchmarkable rand($d)
end

# --- LogNormal + Exponential (numerical only - non-Uniform primary) ---

SUITE["PrimaryCensored"]["LogNormal+Exponential"] = BenchmarkGroup()

let
    delay = LogNormal(1.5, 0.75)
    primary = Exponential(0.5)

    # Non-Uniform primary forces numerical
    d = primary_censored(delay, primary)
    SUITE["PrimaryCensored"]["LogNormal+Exponential"]["numerical"] = BenchmarkGroup()
    SUITE["PrimaryCensored"]["LogNormal+Exponential"]["numerical"]["cdf"] = @benchmarkable cdf($d, $TEST_X)
    SUITE["PrimaryCensored"]["LogNormal+Exponential"]["numerical"]["pdf"] = @benchmarkable pdf($d, $TEST_X)
    SUITE["PrimaryCensored"]["LogNormal+Exponential"]["numerical"]["logpdf"] = @benchmarkable logpdf(
        $d, $TEST_X)
    SUITE["PrimaryCensored"]["LogNormal+Exponential"]["numerical"]["rand"] = @benchmarkable rand($d)
end

# ============================================================================
# IntervalCensored Benchmarks
# ============================================================================

SUITE["IntervalCensored"] = BenchmarkGroup()

# --- Regular intervals ---

SUITE["IntervalCensored"]["Regular"] = BenchmarkGroup()

let
    underlying = LogNormal(1.5, 0.75)
    interval = 1.0

    d = interval_censored(underlying, interval)
    SUITE["IntervalCensored"]["Regular"]["cdf"] = @benchmarkable cdf($d, $TEST_X)
    SUITE["IntervalCensored"]["Regular"]["pdf"] = @benchmarkable pdf($d, $TEST_X)
    SUITE["IntervalCensored"]["Regular"]["logpdf"] = @benchmarkable logpdf($d, $TEST_X)
    SUITE["IntervalCensored"]["Regular"]["rand"] = @benchmarkable rand($d)
end

# --- Arbitrary intervals ---

SUITE["IntervalCensored"]["Arbitrary"] = BenchmarkGroup()

let
    underlying = LogNormal(1.5, 0.75)
    boundaries = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]

    d = interval_censored(underlying, boundaries)
    SUITE["IntervalCensored"]["Arbitrary"]["cdf"] = @benchmarkable cdf($d, $TEST_X)
    SUITE["IntervalCensored"]["Arbitrary"]["pdf"] = @benchmarkable pdf($d, $TEST_X)
    SUITE["IntervalCensored"]["Arbitrary"]["logpdf"] = @benchmarkable logpdf($d, $TEST_X)
    SUITE["IntervalCensored"]["Arbitrary"]["rand"] = @benchmarkable rand($d)
end

# --- Exponential with regular intervals ---

SUITE["IntervalCensored"]["Exponential"] = BenchmarkGroup()

let
    underlying = Exponential(2.0)
    interval = 1.0

    d = interval_censored(underlying, interval)
    SUITE["IntervalCensored"]["Exponential"]["cdf"] = @benchmarkable cdf($d, $TEST_X)
    SUITE["IntervalCensored"]["Exponential"]["pdf"] = @benchmarkable pdf($d, $TEST_X)
    SUITE["IntervalCensored"]["Exponential"]["logpdf"] = @benchmarkable logpdf($d, $TEST_X)
    SUITE["IntervalCensored"]["Exponential"]["rand"] = @benchmarkable rand($d)
end

# ============================================================================
# DoubleIntervalCensored Benchmarks
# ============================================================================

SUITE["DoubleIntervalCensored"] = BenchmarkGroup()

# --- LogNormal + Uniform ---

SUITE["DoubleIntervalCensored"]["LogNormal+Uniform"] = BenchmarkGroup()

let
    delay = LogNormal(1.5, 0.75)
    primary = Uniform(0, 1)

    d = double_interval_censored(delay; primary_event = primary, upper = 10, interval = 1)
    SUITE["DoubleIntervalCensored"]["LogNormal+Uniform"]["cdf"] = @benchmarkable cdf($d, $TEST_X)
    SUITE["DoubleIntervalCensored"]["LogNormal+Uniform"]["pdf"] = @benchmarkable pdf($d, $TEST_X)
    SUITE["DoubleIntervalCensored"]["LogNormal+Uniform"]["logpdf"] = @benchmarkable logpdf($d, $TEST_X)
    SUITE["DoubleIntervalCensored"]["LogNormal+Uniform"]["rand"] = @benchmarkable rand($d)
end

# --- Exponential + Uniform ---

SUITE["DoubleIntervalCensored"]["Exponential+Uniform"] = BenchmarkGroup()

let
    delay = Exponential(2.0)
    primary = Uniform(0, 1)

    d = double_interval_censored(delay; primary_event = primary, upper = 10, interval = 1)
    SUITE["DoubleIntervalCensored"]["Exponential+Uniform"]["cdf"] = @benchmarkable cdf($d, $TEST_X)
    SUITE["DoubleIntervalCensored"]["Exponential+Uniform"]["pdf"] = @benchmarkable pdf($d, $TEST_X)
    SUITE["DoubleIntervalCensored"]["Exponential+Uniform"]["logpdf"] = @benchmarkable logpdf($d, $TEST_X)
    SUITE["DoubleIntervalCensored"]["Exponential+Uniform"]["rand"] = @benchmarkable rand($d)
end
