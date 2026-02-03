SUITE["DoubleIntervalCensored"] = BenchmarkGroup()

# --- LogNormal + Uniform ---

SUITE["DoubleIntervalCensored"]["LogNormal+Uniform"] = BenchmarkGroup()

let
    delay = LogNormal(1.5, 0.75)
    primary = Uniform(0, 1)

    d = double_interval_censored(delay; primary_event = primary, upper = 10, interval = 1)
    SUITE["DoubleIntervalCensored"]["LogNormal+Uniform"]["cdf"] = @benchmarkable cdf.($d, $TEST_XS)
    SUITE["DoubleIntervalCensored"]["LogNormal+Uniform"]["pdf"] = @benchmarkable pdf.($d, $TEST_XS)
    SUITE["DoubleIntervalCensored"]["LogNormal+Uniform"]["logpdf"] = @benchmarkable logpdf.($d, $TEST_XS)
    SUITE["DoubleIntervalCensored"]["LogNormal+Uniform"]["rand"] = @benchmarkable rand($d, 100)
end

# --- Exponential + Uniform ---

SUITE["DoubleIntervalCensored"]["Exponential+Uniform"] = BenchmarkGroup()

let
    delay = Exponential(2.0)
    primary = Uniform(0, 1)

    d = double_interval_censored(delay; primary_event = primary, upper = 10, interval = 1)
    SUITE["DoubleIntervalCensored"]["Exponential+Uniform"]["cdf"] = @benchmarkable cdf.($d, $TEST_XS)
    SUITE["DoubleIntervalCensored"]["Exponential+Uniform"]["pdf"] = @benchmarkable pdf.($d, $TEST_XS)
    SUITE["DoubleIntervalCensored"]["Exponential+Uniform"]["logpdf"] = @benchmarkable logpdf.($d, $TEST_XS)
    SUITE["DoubleIntervalCensored"]["Exponential+Uniform"]["rand"] = @benchmarkable rand($d, 100)
end
