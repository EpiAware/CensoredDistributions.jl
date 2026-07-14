# Per-record `:field` modifier parameters.
#
# A `Symbol` passed to a modifier constructor on a composed node names a column
# read per record; a constant stays constant for every row. This generalises the
# per-record observation horizon (today only the truncation upper bound) to any
# modifier parameter (the truncation `lower`, the interval width, ...). The
# field-bound form must equal the per-record loop that builds each record's node
# with that row's concrete value.

@testsnippet PerRecordFieldSeq begin
    using CensoredDistributions, Distributions

    # A named censored Sequential chain (onset -> admit -> death). Node-level
    # modifiers re-resolve the chain's leaves, keeping the chain scoreable.
    seq_chain() = Sequential(
        (primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
            primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))),
        (:onset_admit, :admit_death))

    ev_of(row) = (onset = row.onset, admit = row.admit, death = row.death)

    # The per-record loop: build each record's node with that row's concrete
    # value and score the row (the bound field is not an event).
    function loop_total(build, rows)
        return sum(rows) do row
            only(CensoredDistributions.batched_event_logpdf(
                build(row), [ev_of(row)]))
        end
    end
end

@testitem "per-record truncation lower equals the per-record loop" setup=[PerRecordFieldSeq] begin
    seq = seq_chain()
    rows = [(onset = 0.0, admit = 2.0, death = 5.0, lo = 1.0),
        (onset = 0.5, admit = 3.0, death = 7.0, lo = 0.5),
        (onset = 0.0, admit = 2.5, death = 6.0, lo = 1.5)]

    got = logpdf(truncated(seq; lower = :lo), rows)
    want = loop_total(row -> truncated(seq; lower = row.lo), rows)
    @test got ≈ want
end

@testitem "per-record truncation lower and upper together" setup=[PerRecordFieldSeq] begin
    seq = seq_chain()
    rows = [(onset = 0.0, admit = 2.0, death = 5.0, lo = 0.5, hi = 9.0),
        (onset = 0.3, admit = 2.5, death = 6.0, lo = 1.0, hi = 10.0)]

    got = logpdf(truncated(seq; lower = :lo, upper = :hi), rows)
    want = loop_total(
        row -> truncated(seq; lower = row.lo, upper = row.hi), rows)
    @test got ≈ want
end

@testitem "per-record interval equals the per-record loop" setup=[PerRecordFieldSeq] begin
    seq = seq_chain()
    rows = [(onset = 0.0, admit = 2.0, death = 5.0, interval = 1.0),
        (onset = 0.5, admit = 3.0, death = 7.0, interval = 0.5),
        (onset = 0.0, admit = 2.5, death = 6.0, interval = 2.0)]

    got = logpdf(interval_censored(seq, :interval), rows)
    want = loop_total(row -> interval_censored(seq, row.interval), rows)
    @test got ≈ want
end

@testitem "per-record double_interval_censored interval field" setup=[PerRecordFieldSeq] begin
    seq = Sequential(
        (LogNormal(1.2, 0.5), Gamma(2.0, 1.0)),
        (:onset_admit, :admit_death))
    rows = [(onset = 0.0, admit = 2.0, death = 5.0, interval = 1.0),
        (onset = 0.5, admit = 3.0, death = 7.0, interval = 2.0)]

    got = logpdf(double_interval_censored(seq; interval = :interval), rows)
    want = loop_total(
        row -> double_interval_censored(seq; interval = row.interval), rows)
    @test got ≈ want
end

@testitem "constant modifier params are unchanged by the field mechanism" setup=[PerRecordFieldSeq] begin
    # A constant value stays constant for every row: the field-aware constructor
    # must reproduce the plain constant-bound node exactly.
    seq = seq_chain()
    rows = [(onset = 0.0, admit = 2.0, death = 5.0),
        (onset = 0.5, admit = 3.0, death = 7.0)]

    @test logpdf(truncated(seq; lower = 0.5), rows) ==
          CensoredDistributions.batched_event_logpdf(
        truncated(seq; lower = 0.5), rows)
    @test logpdf(interval_censored(seq, 1.0), rows) ==
          CensoredDistributions.batched_event_logpdf(
        interval_censored(seq, 1.0), rows)
end

@testitem "per-record field params keep the event names of the inner node" setup=[PerRecordFieldSeq] begin
    seq = seq_chain()
    bound = truncated(seq; lower = :lo)
    @test event_names(bound) == event_names(seq)
end

@testitem "per-record field params score in a DynamicPPL model" setup=[PerRecordFieldSeq] tags=[:turing] begin
    using DynamicPPL: @model, to_submodel, logjoint

    seq=seq_chain()
    rows=[(onset = 0.0, admit = 2.0, death = 5.0, lo = 1.0),
        (onset = 0.5, admit = 3.0, death = 7.0, lo = 0.5)]
    bound=truncated(seq; lower = :lo)

    @model demo(d, t) = obs ~ to_submodel(composed_distribution_model(d, t))
    lp=only(logjoint(demo(bound, rows), (;)))
    # The model scores byte-equal to the table front-door (same resolved nodes).
    @test lp ≈ logpdf(bound, rows)
end

@testitem "per-record field params AD-safe gradient" setup=[PerRecordFieldSeq] tags=[:turing] begin
    using ForwardDiff

    # The per-record `:field` interval is read per row (constant data), so the
    # field-scored path differentiates w.r.t. the leaf params exactly as the
    # constant-interval node does.
    rows=[(onset = 0.0, admit = 2.0, death = 5.0, interval = 1.0),
        (onset = 0.5, admit = 3.0, death = 7.0, interval = 2.0)]

    function nll(theta)
        seq=Sequential(
            (primary_censored(LogNormal(theta[1], 0.5), Uniform(0, 1)),
                primary_censored(Gamma(2.0, theta[2]), Uniform(0, 1))),
            (:onset_admit, :admit_death))
        return -logpdf(interval_censored(seq, :interval), rows)
    end

    g=ForwardDiff.gradient(nll, [1.2, 1.0])
    @test length(g) == 2
    @test all(isfinite, g)
    @test any(!iszero, g)
end
