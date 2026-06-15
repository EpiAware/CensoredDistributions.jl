# Vectorised per-record scoring + sampling of a composed distribution.
#
# `record_distributions(d, rows)` assembles a vector of per-record distributions
# whose product log density must EQUAL the per-record loop
# `sum(event_logpdf(d, ev_r; horizon = h_r) * w_r)` exactly, while building each
# shared convolution segment once. These tests pin that equality across a
# Sequential chain, a Parallel set, mixed observed-patterns, per-record
# truncation and weights, sampling, the construction sharing, type stability,
# and the dual-purpose DynamicPPL entry.

@testitem "vectorised Sequential equals per-record loop" begin
    using Distributions

    function ref_loop(d, rows)
        total = 0.0
        for row in rows
            ev = CensoredDistributions._row_event_vector(d, row)
            w = CensoredDistributions._row_weight_field(row, nothing)
            h = CensoredDistributions._row_horizon_field(row)
            lp = CensoredDistributions.event_logpdf(d, ev; horizon = h)
            total += w === nothing ? lp : w * lp
        end
        return total
    end
    function vec_total(d, rows)
        recs = CensoredDistributions.record_distributions(d, rows)
        return sum(eachindex(recs)) do i
            ev = CensoredDistributions._row_event_vector(d, rows[i])
            logpdf(recs[i], [e === missing ? 0.0 : e for e in ev])
        end
    end

    seq = compose((onset_admit = primary_censored(
            LogNormal(1.2, 0.5), Uniform(0, 1)),
        admit_death = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))))

    rows = [(onset = 0.0, admit = missing, death = 5.0),
        (onset = 0.5, admit = missing, death = 7.0),
        (onset = 0.0, admit = 2.0, death = 6.0),
        (onset = 0.3, admit = 4.0, death = 9.0)]

    @test vec_total(seq, rows) ≈ ref_loop(seq, rows)
end

@testitem "vectorised Sequential honours weights and horizon" begin
    using Distributions

    function ref_loop(d, rows)
        total = 0.0
        for row in rows
            ev = CensoredDistributions._row_event_vector(d, row)
            w = CensoredDistributions._row_weight_field(row, nothing)
            h = CensoredDistributions._row_horizon_field(row)
            lp = CensoredDistributions.event_logpdf(d, ev; horizon = h)
            total += w === nothing ? lp : w * lp
        end
        return total
    end
    function vec_total(d, rows)
        recs = CensoredDistributions.record_distributions(d, rows)
        return sum(eachindex(recs)) do i
            ev = CensoredDistributions._row_event_vector(d, rows[i])
            logpdf(recs[i], [e === missing ? 0.0 : e for e in ev])
        end
    end

    seq = compose((onset_admit = primary_censored(
            LogNormal(1.0, 0.4), Uniform(0, 1)),
        admit_death = primary_censored(Gamma(1.5, 1.2), Uniform(0, 1))))

    rows = [(onset = 0.0, admit = missing, death = 5.0, weight = 3.0,
            obs_time = 8.0),
        (onset = 0.5, admit = missing, death = 7.0, weight = 2.0,
            obs_time = 10.0),
        (onset = 0.0, admit = missing, death = 4.0, count = 5.0)]

    @test vec_total(seq, rows) ≈ ref_loop(seq, rows)
end

@testitem "vectorised Parallel equals per-record loop" begin
    using Distributions

    function ref_loop(d, rows)
        total = 0.0
        for row in rows
            ev = CensoredDistributions._row_event_vector(d, row)
            w = CensoredDistributions._row_weight_field(row, nothing)
            h = CensoredDistributions._row_horizon_field(row)
            lp = CensoredDistributions.event_logpdf(d, ev; horizon = h)
            total += w === nothing ? lp : w * lp
        end
        return total
    end
    function vec_total(d, rows)
        recs = CensoredDistributions.record_distributions(d, rows)
        return sum(eachindex(recs)) do i
            ev = CensoredDistributions._row_event_vector(d, rows[i])
            logpdf(recs[i], [e === missing ? 0.0 : e for e in ev])
        end
    end

    shared = Uniform(0, 1)
    par = Parallel(
        primary_censored(LogNormal(1.0, 0.5), shared),
        primary_censored(Gamma(2.0, 1.0), shared))

    rows = [(event_1 = 0.0, event_2 = 3.0, event_3 = 5.0),
        (event_1 = 0.5, event_2 = missing, event_3 = 6.0),
        (event_1 = 0.0, event_2 = 2.0, event_3 = missing),
        (event_1 = 0.3, event_2 = 5.0, event_3 = 8.0, weight = 4.0)]

    @test vec_total(par, rows) ≈ ref_loop(par, rows)
end

@testitem "vectorised Parallel with horizon equals per-record loop" begin
    using Distributions

    function ref_loop(d, rows)
        total = 0.0
        for row in rows
            ev = CensoredDistributions._row_event_vector(d, row)
            w = CensoredDistributions._row_weight_field(row, nothing)
            h = CensoredDistributions._row_horizon_field(row)
            lp = CensoredDistributions.event_logpdf(d, ev; horizon = h)
            total += w === nothing ? lp : w * lp
        end
        return total
    end
    function vec_total(d, rows)
        recs = CensoredDistributions.record_distributions(d, rows)
        return sum(eachindex(recs)) do i
            ev = CensoredDistributions._row_event_vector(d, rows[i])
            logpdf(recs[i], [e === missing ? 0.0 : e for e in ev])
        end
    end

    shared = Uniform(0, 1)
    par = Parallel(
        primary_censored(LogNormal(1.0, 0.5), shared),
        primary_censored(Gamma(2.0, 1.0), shared))

    rows = [(event_1 = 0.0, event_2 = 3.0, event_3 = 5.0, obs_time = 9.0),
        (event_1 = 0.5, event_2 = missing, event_3 = 6.0, obs_time = 10.0,
            weight = 2.0)]

    @test vec_total(par, rows) ≈ ref_loop(par, rows)
end

@testitem "vectorised shares the segment construction across records" begin
    using Distributions

    # Many records sharing one endpoint-observed pattern build the collapsed
    # origin->terminal segment ONCE: the prebuilt-segment vector has a single
    # entry, the SAME bundle reused by every record.
    seq = Sequential(
        primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))
    rows = [(onset = Float64(i), admit = missing, death = Float64(i + 5))
            for i in 1:20]

    recs = CensoredDistributions.record_distributions(seq, rows)
    bundle = recs[1].segs
    @test length(bundle.segs) == 1
    @test all(r -> r.segs === bundle, recs)

    # A mix of two observed-patterns builds exactly the distinct runs once.
    rows2 = [(onset = 0.0, admit = missing, death = 5.0),
        (onset = 1.0, admit = missing, death = 7.0),
        (onset = 0.0, admit = 2.0, death = 6.0)]
    recs2 = CensoredDistributions.record_distributions(seq, rows2)
    runs = Set((s.a, s.b) for s in recs2[1].segs.segs)
    @test runs == Set([(1, 3), (1, 2), (2, 3)])
end

@testitem "vectorised value equals a non-shared reference build" begin
    using Distributions

    # The shared-construction result equals rebuilding every segment fresh per
    # record: the dedup is a performance optimisation, not a value change.
    seq = Sequential(
        primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))
    rows = [(onset = 0.0, admit = missing, death = 5.0),
        (onset = 0.5, admit = missing, death = 7.0),
        (onset = 0.0, admit = 2.0, death = 6.0)]

    recs = CensoredDistributions.record_distributions(seq, rows)
    shared_total = sum(eachindex(recs)) do i
        ev = CensoredDistributions._row_event_vector(seq, rows[i])
        logpdf(recs[i], [e === missing ? 0.0 : e for e in ev])
    end
    fresh_total = sum(rows) do row
        ev = CensoredDistributions._row_event_vector(seq, row)
        CensoredDistributions.event_logpdf(seq, ev; horizon = nothing)
    end
    @test shared_total ≈ fresh_total
end

@testitem "vectorised Sequential rand samples the full event path" begin
    using Distributions, Random

    seq = Sequential(
        primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))
    rows = [(onset = missing, admit = missing, death = missing)]
    recs = CensoredDistributions.record_distributions(seq, rows)

    Random.seed!(13)
    draws = [rand(recs[1]) for _ in 1:300]
    # The full path is monotone: each event is the previous plus a positive delay.
    @test all(p -> p[1] >= 0 && p[2] >= p[1] && p[3] >= p[2], draws)
end

@testitem "vectorised logpdf is type-stable (@inferred)" begin
    using Distributions, Test

    seq = Sequential(
        primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))
    rows = [(onset = 0.0, admit = missing, death = 5.0),
        (onset = 0.5, admit = missing, death = 7.0)]
    recs = CensoredDistributions.record_distributions(seq, rows)
    @test (@inferred logpdf(recs[1], Float64[0.0, 0.0, 5.0])) isa Float64

    shared = Uniform(0, 1)
    par = Parallel(
        primary_censored(LogNormal(1.0, 0.5), shared),
        primary_censored(Gamma(2.0, 1.0), shared))
    prows = [(event_1 = 0.0, event_2 = 3.0, event_3 = 5.0),
        (event_1 = 0.5, event_2 = missing, event_3 = 6.0)]
    precs = CensoredDistributions.record_distributions(par, prows)
    @test (@inferred logpdf(precs[1], Float64[0.0, 3.0, 5.0])) isa Float64
    @test (@inferred logpdf(precs[2], Float64[0.5, 0.0, 6.0])) isa Float64
end

@testitem "vectorised scores like the per-record submodel sum (DynamicPPL)" begin
    using Distributions
    using DynamicPPL: @model, to_submodel, prefix, logjoint

    seq = compose((onset_admit = primary_censored(
            LogNormal(1.2, 0.5), Uniform(0, 1)),
        admit_death = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))))

    rows = [(onset = 0.0, admit = missing, death = 5.0),
        (onset = 0.5, admit = 3.0, death = 7.0),
        (onset = 0.0, admit = missing, death = 4.0, weight = 2.0)]

    # The vectorised entry scores the whole table in one `~`.
    @model batched_demo(d, t) = obs ~ to_submodel(
        composed_distribution_model(d, t))
    # The per-record reference scores each row in its own prefixed submodel.
    @model function per_record_demo(d, t)
        n = length(t)
        parts = Vector(undef, n)
        for i in 1:n
            parts[i] ~ to_submodel(
                prefix(composed_distribution_model(d, t[i]), Symbol("r", i)),
                false)
        end
    end

    lp_vec = only(logjoint(batched_demo(seq, rows), (;)))
    lp_loop = only(logjoint(per_record_demo(seq, rows), (;)))
    @test lp_vec ≈ lp_loop
end

@testitem "vectorised entry generates a full event matrix when missing" begin
    using Distributions, Random
    using DynamicPPL: @model, to_submodel

    seq = Sequential(
        primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))

    # A fully-missing table samples the full event paths: a 3-event x n-record
    # matrix whose every column is a monotone path.
    miss = [(onset = missing, admit = missing, death = missing) for _ in 1:6]
    @model gen(d, t) = obs ~ to_submodel(composed_distribution_model(d, t))

    Random.seed!(5)
    M = gen(seq, miss)()
    @test size(M) == (3, 6)
    @test all(j -> M[2, j] >= M[1, j] && M[3, j] >= M[2, j], 1:6)
end

# Shared fixtures + references for the nested-Competing (bdbv) and Select (hanta)
# vectorised tests: the case-study trees and the per-record loop the vectorised
# `record_distributions` path must equal.
@testsnippet CaseStudyRecords begin
    using Distributions

    # A bdbv nested-Competing tree with NAMED edges so the event names are
    # onset/admit/death/discharge/notif: onset -> {admit -> Competing(death,
    # discharge), notif}.
    function bdbv_tree()
        edge(mu,
            sigma) = double_interval_censored(LogNormal(mu, sigma);
            primary_event = Uniform(0, 1), interval = 1.0)
        cmp = Competing(:death => (Gamma(2.0, 3.0), 0.3),
            :discharge => (Gamma(2.0, 1.0), 0.7))
        seq = Sequential((edge(1.4, 0.4), cmp),
            (:onset_admit, :admit_resolution))
        return Parallel((seq, edge(1.9, 0.5)), (:onset_admit, :onset_notif))
    end

    # The per-record loop reference: each row builds its own (branch_probs-overridden)
    # tree and scores via `event_logpdf`, exactly the vectorised path's contract.
    function bdbv_ref_loop(d, rows)
        total = 0.0
        for row in rows
            scored = haskey(row, :branch_probs) ?
                     CensoredDistributions._override_competing_outcome_probs(d,
                CensoredDistributions._coerce_branch_probs(
                    CensoredDistributions._the_competing_node(d),
                    row.branch_probs)) : d
            ev = CensoredDistributions._row_event_vector(d, row)
            w = CensoredDistributions._row_weight_field(row, nothing)
            h = CensoredDistributions._row_horizon_field(row)
            lp = CensoredDistributions.event_logpdf(scored, ev; horizon = h)
            total += w === nothing ? lp : w * lp
        end
        return total
    end

    function bdbv_vec_total(d, rows)
        recs = CensoredDistributions.record_distributions(d, rows)
        return sum(eachindex(recs)) do i
            ev = CensoredDistributions._row_event_vector(d, rows[i])
            logpdf(recs[i], [e === missing ? 0.0 : e for e in ev])
        end
    end

    # A hanta Select top: an index case (its own origin) vs a sourced case (a longer
    # delay), selected by the row's `:kind`.
    function hanta_select()
        return selecting(
            :index => primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)),
            :sourced => primary_censored(Gamma(4.0, 1.5), Uniform(0, 1)))
    end

    # The per-record loop reference for a Select top: each row picks its alternative
    # and scores its single observed value, right-truncated at the row's obs_time.
    function hanta_ref_loop(d, rows)
        total = 0.0
        for row in rows
            chosen = CensoredDistributions._pick(d, row.kind)
            h = CensoredDistributions._row_horizon_field(row)
            w = CensoredDistributions._row_weight_field(row, nothing)
            lp = CensoredDistributions.event_logpdf(
                chosen, Float64(row.delay); horizon = h)
            total += w === nothing ? lp : w * lp
        end
        return total
    end

    function hanta_vec_total(d, rows)
        recs = CensoredDistributions.record_distributions(d, rows)
        return sum(eachindex(recs)) do i
            logpdf(recs[i], [Float64(rows[i].delay)])
        end
    end
end

@testitem "vectorised bdbv nested Competing equals per-record loop" setup=[CaseStudyRecords] begin
    using Distributions

    d = bdbv_tree()
    # Each row carries a DIFFERENT per-record branch_probs (covariate CFR), a
    # different observed outcome, and assorted weights.
    rows = [
        (onset = 0.0, admit = 4.0, death = 12.0, discharge = missing,
            notif = 9.0, branch_probs = (death = 0.2, discharge = 0.8)),
        (onset = 0.5, admit = 5.0, death = missing, discharge = 11.0,
            notif = 10.0, branch_probs = (death = 0.6, discharge = 0.4)),
        (onset = 1.0, admit = 3.0, death = 14.0, discharge = missing,
            notif = 8.0, branch_probs = 0.45, weight = 2.0),
        (onset = 0.0, admit = 6.0, death = missing, discharge = missing,
            notif = 12.0, branch_probs = (death = 0.3, discharge = 0.7))]

    @test bdbv_vec_total(d, rows) ≈ bdbv_ref_loop(d, rows)
end

@testitem "vectorised bdbv samples a full named event matrix when missing" setup=[CaseStudyRecords] begin
    using Distributions, Random
    using DynamicPPL: @model, to_submodel

    d = bdbv_tree()
    # Fully-missing rows generate: each column is a full event path, exactly one
    # of the death/discharge outcome slots filled (the resolved outcome).
    miss = [(onset = missing, admit = missing, death = missing,
                discharge = missing, notif = missing) for _ in 1:8]
    @model gen(d, t) = obs ~ to_submodel(composed_distribution_model(d, t))

    Random.seed!(7)
    M = gen(d, miss)()
    # 5 event slots (onset, admit, death, discharge, notif) x 8 records.
    @test size(M) == (5, 8)
end

@testitem "vectorised bdbv is type-stable (@inferred)" setup=[CaseStudyRecords] begin
    using Distributions, Test

    d = bdbv_tree()
    rows = [(onset = 0.0, admit = 4.0, death = 12.0, discharge = missing,
        notif = 9.0, branch_probs = (death = 0.2, discharge = 0.8))]
    recs = CensoredDistributions.record_distributions(d, rows)
    @test (@inferred logpdf(recs[1], Float64[0.0, 4.0, 12.0, 0.0, 9.0])) isa
          Float64
end

@testitem "vectorised hanta Select equals per-record loop" setup=[CaseStudyRecords] begin
    using Distributions

    d = hanta_select()
    # Each row selects a DIFFERENT alternative and carries a DIFFERENT obs_time.
    rows = [(kind = :index, delay = 3.0, obs_time = 8.0),
        (kind = :sourced, delay = 5.0, obs_time = 12.0),
        (kind = :index, delay = 2.0, obs_time = 6.0, weight = 2.0),
        (kind = :sourced, delay = 7.0, obs_time = 15.0)]

    @test hanta_vec_total(d, rows) ≈ hanta_ref_loop(d, rows)
end

@testitem "vectorised hanta Select samples per row when missing" setup=[CaseStudyRecords] begin
    using Distributions, Random
    using DynamicPPL: @model, to_submodel

    d = hanta_select()
    miss = [(kind = :index, delay = missing) for _ in 1:4]
    @model gen(d, t) = obs ~ to_submodel(composed_distribution_model(d, t))

    Random.seed!(11)
    M = gen(d, miss)()
    @test size(M) == (1, 4)
    @test all(M .>= 0)
end

@testitem "vectorised hanta Select scores like the per-record submodel" setup=[CaseStudyRecords] begin
    using Distributions
    using DynamicPPL: @model, to_submodel, prefix, logjoint

    d = hanta_select()
    rows = [(kind = :index, delay = 3.0, obs_time = 8.0),
        (kind = :sourced, delay = 5.0, obs_time = 12.0),
        (kind = :index, delay = 2.0, obs_time = 6.0, weight = 2.0)]

    @model batched(d, t) = obs ~ to_submodel(composed_distribution_model(d, t))
    @model function per_record(d, t)
        n = length(t)
        parts = Vector(undef, n)
        for i in 1:n
            parts[i] ~ to_submodel(
                prefix(composed_distribution_model(d, t[i]), Symbol("r", i)),
                false)
        end
    end

    lp_vec = only(logjoint(batched(d, rows), (;)))
    lp_loop = only(logjoint(per_record(d, rows), (;)))
    @test lp_vec ≈ lp_loop
end

@testitem "vectorised Select rejects heterogeneous-length alternatives" begin
    using CensoredDistributions, Distributions

    # A leaf alternative is one event slot; a composer alternative is several. A
    # table that mixes rows selecting alternatives of DIFFERENT event-slot counts
    # has no rectangular event matrix. The records are heterogeneous-length, so
    # `product_distribution` would throw Distributions.jl's opaque "all
    # distributions must be of the same size". `record_distributions` raises a
    # clear `ArgumentError` instead.
    d = selecting(
        :leaf => Gamma(2.0, 1.0),
        :pair => Sequential(Gamma(1.0, 1.0), LogNormal(0.5, 0.4)))
    rows = [(kind = :leaf, value = 2.0),
        (kind = :pair, onset = 0.0, admit = 3.0)]
    err = try
        CensoredDistributions.record_distributions(d, rows)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("same number", err.msg)
    # A homogeneous table (all rows pick the same-length alternative) is fine.
    homog = [(kind = :leaf, value = 2.0), (kind = :leaf, value = 3.0)]
    recs = CensoredDistributions.record_distributions(d, homog)
    @test length(recs) == 2
end

# ---------------------------------------------------------------------------
# Grouped per-stratum assembly (varying-params primitive)
# ---------------------------------------------------------------------------

@testitem "grouped record_distributions equals the per-stratum loop" begin
    using Distributions

    mk(scale) = Sequential(
        primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
        primary_censored(Gamma(2.0, scale), Uniform(0, 1)))
    ds = [mk(1.0), mk(2.0), mk(0.5)]
    rows = [(onset = 0.0, admit = 2.0, death = 5.0),
        (onset = 1.0, admit = 3.0, death = 9.0),
        (onset = 0.5, admit = 2.5, death = 6.0),
        (onset = 0.2, admit = 1.8, death = 4.0)]
    group = [1, 2, 1, 3]
    obs = [[0.0, 2.0, 5.0], [1.0, 3.0, 9.0], [0.5, 2.5, 6.0], [0.2, 1.8, 4.0]]

    recs = CensoredDistributions.record_distributions(ds, rows; group = group)
    # The grouped scoring equals the per-record loop over each record's OWN
    # stratum distribution (the invariant).
    grouped = sum(logpdf(recs[i], obs[i]) for i in eachindex(recs))
    manual = sum(
        CensoredDistributions.event_logpdf(
            ds[group[i]], Vector{Union{Missing, Float64}}(obs[i]))
    for i in eachindex(obs))
    @test grouped ≈ manual

    # `batched_event_logpdf` is the same value as a direct call (no model).
    @test CensoredDistributions.batched_event_logpdf(ds, rows; group = group) ≈
          grouped
end

@testitem "grouped record_distributions one stratum == shared-d path" begin
    using Distributions

    d = Sequential(
        primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))
    rows = [(onset = 0.0, admit = 2.0, death = 5.0),
        (onset = 1.0, admit = missing, death = 7.0),
        (onset = 0.5, admit = 2.5, death = 6.0)]
    obs = [[0.0, 2.0, 5.0], [1.0, 0.0, 7.0], [0.5, 2.5, 6.0]]
    group = [1, 1, 1]

    grouped = CensoredDistributions.record_distributions([d], rows; group = group)
    shared = CensoredDistributions.record_distributions(d, rows)
    # Degenerate single stratum is bit-identical to the shared-`d` fast path.
    @test all(logpdf(grouped[i], obs[i]) == logpdf(shared[i], obs[i])
    for i in eachindex(obs))
end

@testitem "grouped record_distributions validates group ids" begin
    using Distributions

    mk(scale) = Sequential(
        primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
        primary_censored(Gamma(2.0, scale), Uniform(0, 1)))
    ds = [mk(1.0), mk(2.0)]
    rows = [(onset = 0.0, admit = 2.0, death = 5.0),
        (onset = 1.0, admit = 3.0, death = 9.0)]

    # Wrong-length group.
    @test_throws ArgumentError CensoredDistributions.record_distributions(
        ds, rows; group = [1])
    # Out-of-range stratum id.
    @test_throws ArgumentError CensoredDistributions.record_distributions(
        ds, rows; group = [1, 3])
    # Non-integer (float) id - the AD footgun guard.
    @test_throws ArgumentError CensoredDistributions.record_distributions(
        ds, rows; group = [1.0, 2.0])
end

@testitem "grouped record_distributions AD-safe gradient" tags=[:turing] begin
    using CensoredDistributions, Distributions, ForwardDiff

    # The grouped scoring must differentiate w.r.t. the per-stratum params (the
    # group key is an integer data-pass id, never keyed on a float). Build `ds`
    # from a Dual-carrying parameter vector and check a finite gradient.
    rows = [(onset = 0.0, admit = 2.0, death = 5.0),
        (onset = 1.0, admit = 3.0, death = 9.0),
        (onset = 0.5, admit = 2.5, death = 6.0)]
    group = [1, 2, 1]

    function nll(theta)
        mk(scale) = Sequential(
            primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
            primary_censored(Gamma(2.0, scale), Uniform(0, 1)))
        ds = [mk(theta[1]), mk(theta[2])]
        return -CensoredDistributions.batched_event_logpdf(ds, rows; group = group)
    end

    g = ForwardDiff.gradient(nll, [1.0, 2.0])
    @test length(g) == 2
    @test all(isfinite, g)
    @test any(!iszero, g)
end
