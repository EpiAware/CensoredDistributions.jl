# Per-record `:field` modifier parameters.
#
# A `Symbol` passed to a modifier constructor on a composed node names a column
# read per record; a constant stays constant for every row. This generalises the
# per-record observation horizon (today only the truncation upper bound) to any
# modifier parameter (the truncation `lower`, the interval width, ...). The
# field-bound form must equal the per-record loop that builds each record's node
# with that row's concrete value.

@testitem "per-record truncation lower equals the per-record loop" begin
    using CensoredDistributions, Distributions

    seq = compose((onset_admit = primary_censored(
            LogNormal(1.2, 0.5), Uniform(0, 1)),
        admit_death = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))))

    rows = [(onset = 0.0, admit = 2.0, death = 5.0, lo = 1.0),
        (onset = 0.5, admit = 3.0, death = 7.0, lo = 0.5),
        (onset = 0.0, admit = 2.5, death = 6.0, lo = 1.5)]

    bound = truncated(seq; lower = :lo)
    got = logpdf(bound, rows)

    # The per-record loop builds each record's node with that row's concrete
    # lower bound, then scores the row (the `lo` field is not an event).
    want = sum(rows) do row
        node = truncated(seq; lower = row.lo)
        ev = (onset = row.onset, admit = row.admit, death = row.death)
        only(CensoredDistributions.batched_event_logpdf(node, [ev]))
    end
    @test got ≈ want
end

@testitem "per-record truncation lower and upper together" begin
    using CensoredDistributions, Distributions

    seq = compose((onset_admit = primary_censored(
            LogNormal(1.0, 0.4), Uniform(0, 1)),
        admit_death = primary_censored(Gamma(1.5, 1.2), Uniform(0, 1))))

    rows = [(onset = 0.0, admit = 2.0, death = 5.0, lo = 0.5, hi = 9.0),
        (onset = 0.3, admit = 2.5, death = 6.0, lo = 1.0, hi = 10.0)]

    bound = truncated(seq; lower = :lo, upper = :hi)
    got = logpdf(bound, rows)

    want = sum(rows) do row
        node = truncated(seq; lower = row.lo, upper = row.hi)
        ev = (onset = row.onset, admit = row.admit, death = row.death)
        only(CensoredDistributions.batched_event_logpdf(node, [ev]))
    end
    @test got ≈ want
end

@testitem "per-record interval equals the per-record loop" begin
    using CensoredDistributions, Distributions

    seq = compose((onset_admit = primary_censored(
            LogNormal(1.2, 0.5), Uniform(0, 1)),
        admit_death = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))))

    rows = [(onset = 0.0, admit = 2.0, death = 5.0, interval = 1.0),
        (onset = 0.5, admit = 3.0, death = 7.0, interval = 0.5),
        (onset = 0.0, admit = 2.5, death = 6.0, interval = 2.0)]

    bound = interval_censored(seq, :interval)
    got = logpdf(bound, rows)

    want = sum(rows) do row
        node = interval_censored(seq, row.interval)
        ev = (onset = row.onset, admit = row.admit, death = row.death)
        only(CensoredDistributions.batched_event_logpdf(node, [ev]))
    end
    @test got ≈ want
end

@testitem "per-record double_interval_censored interval field" begin
    using CensoredDistributions, Distributions

    seq = Sequential(
        (LogNormal(1.2, 0.5), Gamma(2.0, 1.0)),
        (:onset_admit, :admit_death))

    rows = [(onset = 0.0, admit = 2.0, death = 5.0, interval = 1.0),
        (onset = 0.5, admit = 3.0, death = 7.0, interval = 2.0)]

    bound = double_interval_censored(seq; interval = :interval)
    got = logpdf(bound, rows)

    want = sum(rows) do row
        node = double_interval_censored(seq; interval = row.interval)
        ev = (onset = row.onset, admit = row.admit, death = row.death)
        only(CensoredDistributions.batched_event_logpdf(node, [ev]))
    end
    @test got ≈ want
end

@testitem "constant modifier params are unchanged by the field mechanism" begin
    using CensoredDistributions, Distributions

    # A constant value stays constant for every row: the field-aware constructor
    # must reproduce the plain constant-bound node exactly.
    seq = compose((onset_admit = primary_censored(
            LogNormal(1.2, 0.5), Uniform(0, 1)),
        admit_death = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))))
    rows = [(onset = 0.0, admit = 2.0, death = 5.0),
        (onset = 0.5, admit = 3.0, death = 7.0)]

    @test logpdf(truncated(seq; lower = 0.5), rows) ==
          CensoredDistributions.batched_event_logpdf(
        truncated(seq; lower = 0.5), rows)
    @test logpdf(interval_censored(seq, 1.0), rows) ==
          CensoredDistributions.batched_event_logpdf(
        interval_censored(seq, 1.0), rows)
end

@testitem "per-record field params keep the event names of the inner node" begin
    using CensoredDistributions, Distributions

    seq = compose((onset_admit = primary_censored(
            LogNormal(1.2, 0.5), Uniform(0, 1)),
        admit_death = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))))
    bound = truncated(seq; lower = :lo)
    @test event_names(bound) == event_names(seq)
end

@testitem "per-record field params AD-safe gradient" tags=[:turing] begin
    using CensoredDistributions, Distributions, ForwardDiff

    rows = [(onset = 0.0, admit = 2.0, death = 5.0, lo = 1.0, interval = 1.0),
        (onset = 0.5, admit = 3.0, death = 7.0, lo = 0.5, interval = 2.0)]

    function nll(theta)
        seq = compose((onset_admit = primary_censored(
                LogNormal(theta[1], 0.5), Uniform(0, 1)),
            admit_death = primary_censored(Gamma(2.0, theta[2]), Uniform(0, 1))))
        bound = truncated(seq; lower = :lo)
        return -logpdf(bound, rows)
    end

    g = ForwardDiff.gradient(nll, [1.2, 1.0])
    @test length(g) == 2
    @test all(isfinite, g)
    @test any(!iszero, g)
end
