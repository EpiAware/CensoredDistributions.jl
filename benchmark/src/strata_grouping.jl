# Per-stratum grouping (the varying-params primitive / #370): records fall into a
# few STRATA, each with its own (sampled) delay params. Building each record's
# composed distribution INDEPENDENTLY rebuilds the expensive censored edge once
# per record (cost N_records); GROUPING by the integer stratum id builds each
# distinct stratum's edge ONCE and reuses it across that stratum's records (cost
# Σ_strata, i.e. n_strata builds), then scores all records. With many records per
# stratum the grouped build-once is far cheaper.

SUITE["StrataGrouping"] = BenchmarkGroup()

let
    # A censored two-event edge whose construction (numeric primary-censoring) is
    # genuinely expensive to rebuild, so per-record rebuilding shows up.
    mk(scale) = Sequential(
        primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
        primary_censored(Gamma(2.0, scale), Uniform(0, 1)))

    n_strata = 5
    per_stratum = 60
    n_records = n_strata * per_stratum

    ds = [mk(0.5 + 0.5 * k) for k in 1:n_strata]
    rows = NamedTuple[]
    group = Int[]
    for k in 1:n_strata
        for _ in 1:per_stratum
            push!(rows, (onset = 0.0, admit = 2.0, death = 5.0))
            push!(group, k)
        end
    end

    # Naive: build each record's distribution from scratch (one stratum-`d` per
    # record) and score it - the per-edge construction repeats per record.
    SUITE["StrataGrouping"]["per_record_build"] = @benchmarkable begin
        total = 0.0
        for i in 1:($n_records)
            recs = CensoredDistributions.record_distributions(
                $ds[$group[i]], [$rows[i]])
            total += logpdf(recs[1], [0.0, 2.0, 5.0])
        end
        total
    end

    # Grouped: build each stratum's edge ONCE (keyed on the integer stratum id),
    # reuse across that stratum's records, and score the whole table.
    # `batched_event_logpdf` is the Turing-friendly grouped primitive: a plain
    # `logpdf`-style scalar that drops into a `@model` via `@addlogprob!` and
    # differentiates under ForwardDiff / Mooncake (the `group` ids are integer
    # data; the sampled params ride inside `ds`). It accepts a vector of composers
    # OR bare leaves per stratum. See its docstring for the partial-pooling
    # `@model` pattern.
    SUITE["StrataGrouping"]["grouped"] = @benchmarkable begin
        CensoredDistributions.batched_event_logpdf(
            $ds, $rows; group = $group)
    end
end
