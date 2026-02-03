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
