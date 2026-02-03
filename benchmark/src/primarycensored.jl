SUITE["PrimaryCensored"] = BenchmarkGroup()

# --- Gamma + Uniform (analytical and numerical) ---

SUITE["PrimaryCensored"]["Gamma+Uniform"] = BenchmarkGroup()

let
    delay = Gamma(2.0, 1.0)
    primary = Uniform(0, 1)

    # Analytical (default)
    d_analytical = primary_censored(delay, primary)
    SUITE["PrimaryCensored"]["Gamma+Uniform"]["analytical"] = BenchmarkGroup()
    SUITE["PrimaryCensored"]["Gamma+Uniform"]["analytical"]["cdf"] = @benchmarkable cdf.(
        $d_analytical, $TEST_XS)
    SUITE["PrimaryCensored"]["Gamma+Uniform"]["analytical"]["pdf"] = @benchmarkable pdf.(
        $d_analytical, $TEST_XS)
    SUITE["PrimaryCensored"]["Gamma+Uniform"]["analytical"]["logpdf"] = @benchmarkable logpdf.(
        $d_analytical, $TEST_XS)
    SUITE["PrimaryCensored"]["Gamma+Uniform"]["analytical"]["rand"] = @benchmarkable rand($d_analytical, 100)

    # Numerical (force_numeric=true)
    d_numeric = primary_censored(delay, primary; force_numeric = true)
    SUITE["PrimaryCensored"]["Gamma+Uniform"]["numerical"] = BenchmarkGroup()
    SUITE["PrimaryCensored"]["Gamma+Uniform"]["numerical"]["cdf"] = @benchmarkable cdf.($d_numeric, $TEST_XS)
    SUITE["PrimaryCensored"]["Gamma+Uniform"]["numerical"]["pdf"] = @benchmarkable pdf.($d_numeric, $TEST_XS)
    SUITE["PrimaryCensored"]["Gamma+Uniform"]["numerical"]["logpdf"] = @benchmarkable logpdf.(
        $d_numeric, $TEST_XS)
    SUITE["PrimaryCensored"]["Gamma+Uniform"]["numerical"]["rand"] = @benchmarkable rand($d_numeric, 100)
end

# --- LogNormal + Uniform (analytical and numerical) ---

SUITE["PrimaryCensored"]["LogNormal+Uniform"] = BenchmarkGroup()

let
    delay = LogNormal(1.5, 0.75)
    primary = Uniform(0, 1)

    # Analytical (default)
    d_analytical = primary_censored(delay, primary)
    SUITE["PrimaryCensored"]["LogNormal+Uniform"]["analytical"] = BenchmarkGroup()
    SUITE["PrimaryCensored"]["LogNormal+Uniform"]["analytical"]["cdf"] = @benchmarkable cdf.(
        $d_analytical, $TEST_XS)
    SUITE["PrimaryCensored"]["LogNormal+Uniform"]["analytical"]["pdf"] = @benchmarkable pdf.(
        $d_analytical, $TEST_XS)
    SUITE["PrimaryCensored"]["LogNormal+Uniform"]["analytical"]["logpdf"] = @benchmarkable logpdf.(
        $d_analytical, $TEST_XS)
    SUITE["PrimaryCensored"]["LogNormal+Uniform"]["analytical"]["rand"] = @benchmarkable rand(
        $d_analytical, 100)

    # Numerical (force_numeric=true)
    d_numeric = primary_censored(delay, primary; force_numeric = true)
    SUITE["PrimaryCensored"]["LogNormal+Uniform"]["numerical"] = BenchmarkGroup()
    SUITE["PrimaryCensored"]["LogNormal+Uniform"]["numerical"]["cdf"] = @benchmarkable cdf.(
        $d_numeric, $TEST_XS)
    SUITE["PrimaryCensored"]["LogNormal+Uniform"]["numerical"]["pdf"] = @benchmarkable pdf.(
        $d_numeric, $TEST_XS)
    SUITE["PrimaryCensored"]["LogNormal+Uniform"]["numerical"]["logpdf"] = @benchmarkable logpdf.(
        $d_numeric, $TEST_XS)
    SUITE["PrimaryCensored"]["LogNormal+Uniform"]["numerical"]["rand"] = @benchmarkable rand($d_numeric, 100)
end

# --- Weibull + Uniform (analytical and numerical) ---

SUITE["PrimaryCensored"]["Weibull+Uniform"] = BenchmarkGroup()

let
    delay = Weibull(2.0, 3.0)
    primary = Uniform(0, 1)

    # Analytical (default)
    d_analytical = primary_censored(delay, primary)
    SUITE["PrimaryCensored"]["Weibull+Uniform"]["analytical"] = BenchmarkGroup()
    SUITE["PrimaryCensored"]["Weibull+Uniform"]["analytical"]["cdf"] = @benchmarkable cdf.(
        $d_analytical, $TEST_XS)
    SUITE["PrimaryCensored"]["Weibull+Uniform"]["analytical"]["pdf"] = @benchmarkable pdf.(
        $d_analytical, $TEST_XS)
    SUITE["PrimaryCensored"]["Weibull+Uniform"]["analytical"]["logpdf"] = @benchmarkable logpdf.(
        $d_analytical, $TEST_XS)
    SUITE["PrimaryCensored"]["Weibull+Uniform"]["analytical"]["rand"] = @benchmarkable rand($d_analytical, 100)

    # Numerical (force_numeric=true)
    d_numeric = primary_censored(delay, primary; force_numeric = true)
    SUITE["PrimaryCensored"]["Weibull+Uniform"]["numerical"] = BenchmarkGroup()
    SUITE["PrimaryCensored"]["Weibull+Uniform"]["numerical"]["cdf"] = @benchmarkable cdf.($d_numeric, $TEST_XS)
    SUITE["PrimaryCensored"]["Weibull+Uniform"]["numerical"]["pdf"] = @benchmarkable pdf.($d_numeric, $TEST_XS)
    SUITE["PrimaryCensored"]["Weibull+Uniform"]["numerical"]["logpdf"] = @benchmarkable logpdf.(
        $d_numeric, $TEST_XS)
    SUITE["PrimaryCensored"]["Weibull+Uniform"]["numerical"]["rand"] = @benchmarkable rand($d_numeric, 100)
end

# --- Exponential + Uniform (numerical only - no analytical solution) ---

SUITE["PrimaryCensored"]["Exponential+Uniform"] = BenchmarkGroup()

let
    delay = Exponential(2.0)
    primary = Uniform(0, 1)

    d = primary_censored(delay, primary)
    SUITE["PrimaryCensored"]["Exponential+Uniform"]["numerical"] = BenchmarkGroup()
    SUITE["PrimaryCensored"]["Exponential+Uniform"]["numerical"]["cdf"] = @benchmarkable cdf.($d, $TEST_XS)
    SUITE["PrimaryCensored"]["Exponential+Uniform"]["numerical"]["pdf"] = @benchmarkable pdf.($d, $TEST_XS)
    SUITE["PrimaryCensored"]["Exponential+Uniform"]["numerical"]["logpdf"] = @benchmarkable logpdf.(
        $d, $TEST_XS)
    SUITE["PrimaryCensored"]["Exponential+Uniform"]["numerical"]["rand"] = @benchmarkable rand($d, 100)
end

# --- Gamma + Exponential (numerical only - non-Uniform primary) ---

SUITE["PrimaryCensored"]["Gamma+Exponential"] = BenchmarkGroup()

let
    delay = Gamma(2.0, 1.0)
    primary = Exponential(0.5)

    d = primary_censored(delay, primary)
    SUITE["PrimaryCensored"]["Gamma+Exponential"]["numerical"] = BenchmarkGroup()
    SUITE["PrimaryCensored"]["Gamma+Exponential"]["numerical"]["cdf"] = @benchmarkable cdf.($d, $TEST_XS)
    SUITE["PrimaryCensored"]["Gamma+Exponential"]["numerical"]["pdf"] = @benchmarkable pdf.($d, $TEST_XS)
    SUITE["PrimaryCensored"]["Gamma+Exponential"]["numerical"]["logpdf"] = @benchmarkable logpdf.($d, $TEST_XS)
    SUITE["PrimaryCensored"]["Gamma+Exponential"]["numerical"]["rand"] = @benchmarkable rand($d, 100)
end

# --- LogNormal + Exponential (numerical only - non-Uniform primary) ---

SUITE["PrimaryCensored"]["LogNormal+Exponential"] = BenchmarkGroup()

let
    delay = LogNormal(1.5, 0.75)
    primary = Exponential(0.5)

    d = primary_censored(delay, primary)
    SUITE["PrimaryCensored"]["LogNormal+Exponential"]["numerical"] = BenchmarkGroup()
    SUITE["PrimaryCensored"]["LogNormal+Exponential"]["numerical"]["cdf"] = @benchmarkable cdf.($d, $TEST_XS)
    SUITE["PrimaryCensored"]["LogNormal+Exponential"]["numerical"]["pdf"] = @benchmarkable pdf.($d, $TEST_XS)
    SUITE["PrimaryCensored"]["LogNormal+Exponential"]["numerical"]["logpdf"] = @benchmarkable logpdf.(
        $d, $TEST_XS)
    SUITE["PrimaryCensored"]["LogNormal+Exponential"]["numerical"]["rand"] = @benchmarkable rand($d, 100)
end
