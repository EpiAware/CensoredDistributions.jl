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

@testitem "node-level truncation over a primary-censored Parallel assembles" begin
    using Distributions

    # Regression for #741: a node-level `truncated(node; lower)` over a Parallel
    # of primary-censored branches must keep the censored leaves
    # record-assemblable. Before the fix the truncation distribute stripped each
    # branch to its continuous core, so the shared primary was lost and
    # `record_distributions` rejected the (now plain-branch) Parallel.
    shared = Uniform(0, 1)
    par = compose((
        a = primary_censored(Gamma(2.0, 1.0), shared),
        b = primary_censored(LogNormal(0.5, 0.4), shared)))

    trunc = truncated(par; lower = 0.5)
    # The branches keep their primary censoring through the truncation layer.
    @test trunc isa CensoredDistributions.Parallel
    for c in trunc.components
        @test c isa Truncated
        @test c.untruncated isa CensoredDistributions.PrimaryCensored
        @test CensoredDistributions._origin_primary_event(c) == shared
    end

    rows = [(origin = 0.0, a = 2.0, b = 3.0),
        (origin = 0.5, a = missing, b = 4.0)]
    recs = CensoredDistributions.record_distributions(trunc, rows)
    @test length(recs) == 2

    # Density-identical to the canonical per-leaf construction
    # `compose((a = truncated(primary_censored(...); lower), ...))`.
    canon = compose((
        a = truncated(primary_censored(Gamma(2.0, 1.0), shared); lower = 0.5),
        b = truncated(
            primary_censored(LogNormal(0.5, 0.4), shared); lower = 0.5)))
    recs_canon = CensoredDistributions.record_distributions(canon, rows)
    for i in eachindex(rows)
        ev = CensoredDistributions._row_event_vector(trunc, rows[i])
        x = [e === missing ? 0.0 : e for e in ev]
        @test logpdf(recs[i], x) ≈ logpdf(recs_canon[i], x)
    end
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

# Shared fixtures + references for the nested-Resolve (bdbv) and Choose (hanta)
# vectorised tests: the case-study trees and the per-record loop the vectorised
# `record_distributions` path must equal.
@testsnippet CaseStudyRecords begin
    using Distributions

    # A bdbv nested-Resolve tree with NAMED edges so the event names are
    # onset/admit/death/discharge/notif: onset -> {admit -> Resolve(death,
    # discharge), notif}.
    function bdbv_tree()
        edge(mu,
            sigma) = double_interval_censored(LogNormal(mu, sigma);
            primary_event = Uniform(0, 1), interval = 1.0)
        cmp = Resolve(:death => (Gamma(2.0, 3.0), 0.3),
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
                     CensoredDistributions._override_one_of_outcome_probs(d,
                CensoredDistributions._coerce_branch_probs(
                    CensoredDistributions._the_one_of_node(d),
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

    # A hanta Choose top: an index case (its own origin) vs a sourced case (a longer
    # delay), selected by the row's `:kind`.
    function hanta_select()
        return choose(
            :index => primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)),
            :sourced => primary_censored(Gamma(4.0, 1.5), Uniform(0, 1)))
    end

    # The per-record loop reference for a Choose top: each row picks its alternative
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

@testitem "vectorised bdbv nested Resolve equals per-record loop" setup=[CaseStudyRecords] begin
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

@testitem "vectorised hanta Choose equals per-record loop" setup=[CaseStudyRecords] begin
    using Distributions

    d = hanta_select()
    # Each row selects a DIFFERENT alternative and carries a DIFFERENT obs_time.
    rows = [(kind = :index, delay = 3.0, obs_time = 8.0),
        (kind = :sourced, delay = 5.0, obs_time = 12.0),
        (kind = :index, delay = 2.0, obs_time = 6.0, weight = 2.0),
        (kind = :sourced, delay = 7.0, obs_time = 15.0)]

    @test hanta_vec_total(d, rows) ≈ hanta_ref_loop(d, rows)
end

@testitem "vectorised hanta Choose samples per row when missing" setup=[CaseStudyRecords] begin
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

@testitem "vectorised hanta Choose scores like the per-record submodel" setup=[CaseStudyRecords] begin
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

@testitem "vectorised Choose rejects heterogeneous-length alternatives" begin
    using CensoredDistributions, Distributions

    # A leaf alternative is one event slot; a composer alternative is several. A
    # table that mixes rows choose alternatives of DIFFERENT event-slot counts
    # has no rectangular event matrix. The records are heterogeneous-length, so
    # `product_distribution` would throw Distributions.jl's opaque "all
    # distributions must be of the same size". `record_distributions` raises a
    # clear `ArgumentError` instead.
    d = choose(
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

# ---------------------------------------------------------------------------
# Bare-leaf records (a single-delay model, no Sequential wrapper)
# ---------------------------------------------------------------------------

@testitem "bare-leaf record == single-edge Sequential wrapper" begin
    using CensoredDistributions, Distributions

    # A single-delay model scores a BARE censored leaf directly: no need to wrap
    # it in a one-edge `Sequential`. The bare-leaf record must be density-equal to
    # the one-edge-`Sequential`-wrapped form (observed from a zero origin) and to
    # the per-record loop.
    leaf = primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1))
    rows = [(delay = 2.0,), (delay = 3.5,), (delay = 4.0,)]
    obs = [[2.0], [3.5], [4.0]]

    bare = CensoredDistributions.record_distributions(leaf, rows)
    bare_total = sum(logpdf(bare[i], obs[i]) for i in eachindex(bare))

    # Single-edge `Sequential(leaf)` over `[E_0, E_1]` from a zero origin.
    seq = Sequential(leaf)
    wrows = [(onset = 0.0, delay = 2.0), (onset = 0.0, delay = 3.5),
        (onset = 0.0, delay = 4.0)]
    wobs = [[0.0, 2.0], [0.0, 3.5], [0.0, 4.0]]
    wrapped = CensoredDistributions.record_distributions(seq, wrows)
    wrapped_total = sum(logpdf(wrapped[i], wobs[i]) for i in eachindex(wrapped))

    # The per-record loop value.
    loop = sum(logpdf(leaf, obs[i][1]) for i in eachindex(obs))

    @test bare_total ≈ wrapped_total
    @test bare_total ≈ loop
    @test CensoredDistributions.batched_event_logpdf(leaf, rows) ≈ bare_total
end

@testitem "bare-leaf record horizon and weight" begin
    using CensoredDistributions, Distributions

    leaf = primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1))
    # A reserved `obs_time` right-truncates the leaf; a `weight` scales it.
    hrec = only(CensoredDistributions.record_distributions(
        leaf, [(delay = 2.0, obs_time = 10.0)]))
    @test logpdf(hrec, [2.0]) ≈
          logpdf(CensoredDistributions.truncate_to_horizon(leaf, 10.0), 2.0)

    wrec = only(CensoredDistributions.record_distributions(
        leaf, [(delay = 2.0, weight = 3.0)]))
    @test logpdf(wrec, [2.0]) ≈ 3 * logpdf(leaf, 2.0)

    # An empty table errors clearly (matching the composer entries).
    @test_throws ArgumentError CensoredDistributions.record_distributions(
        leaf, NamedTuple[])
end

@testitem "bare-leaf record rejects an extra non-reserved field" begin
    using CensoredDistributions, Distributions

    # A bare leaf scores ONE observed delay per row. A row that carries a second
    # non-reserved data column (e.g. a stray `D` left alongside the reserved
    # `obs_time` horizon) would otherwise be miscounted as a two-event record;
    # the leaf path rejects it with a message naming the offending fields, and
    # the same path serves both a bare leaf and a leaf `Choose` alternative so
    # the message must not be Choose-specific.
    leaf = primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1))
    ds = [leaf, primary_censored(Gamma(2.0, 1.5), Uniform(0, 1))]
    rows = [(delay = 2.0, D = 10.0, obs_time = 10.0),
        (delay = 3.0, D = 12.0, obs_time = 12.0)]
    group = [1, 2]

    err = try
        CensoredDistributions.record_distributions(ds, rows; group = group)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("one event value", err.msg)
    @test occursin("delay", err.msg)
    @test occursin("D", err.msg)
    @test !occursin("Choose", err.msg)

    # Dropping the stray `D` (keeping only the event and the reserved horizon)
    # scores cleanly, matching the per-record loop.
    ok = [(delay = 2.0, obs_time = 10.0), (delay = 3.0, obs_time = 12.0)]
    recs = CensoredDistributions.record_distributions(ds, ok; group = group)
    loop = sum(logpdf(
                   CensoredDistributions.truncate_to_horizon(ds[group[i]],
                       ok[i].obs_time), ok[i].delay)
    for i in eachindex(ok))
    @test sum(logpdf(recs[i], [ok[i].delay]) for i in eachindex(recs)) ≈ loop
end

@testitem "bare-leaf grouped == wrapped == per-stratum loop" begin
    using CensoredDistributions, Distributions

    # Bare leaves per stratum (partial-pooling style): each record scores its
    # OWN stratum leaf directly. Grouped == single-edge-Sequential-wrapped ==
    # the per-record loop.
    leaf1 = primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1))
    leaf2 = primary_censored(Gamma(2.0, 1.5), Uniform(0, 1))
    ds = [leaf1, leaf2]
    rows = [(delay = 2.0,), (delay = 3.0,), (delay = 4.0,), (delay = 5.0,)]
    group = [1, 2, 1, 2]
    obs = [[2.0], [3.0], [4.0], [5.0]]

    recs = CensoredDistributions.record_distributions(ds, rows; group = group)
    grouped = sum(logpdf(recs[i], obs[i]) for i in eachindex(recs))

    loop = sum(logpdf(ds[group[i]], obs[i][1]) for i in eachindex(obs))
    @test grouped ≈ loop
    # The leaf genuinely CONSTRAINS the observed delay (not a silent zero): the
    # record scores `logpdf(leaf, delay)`, so the contribution is non-trivial.
    @test grouped != 0.0
    @test all(logpdf(recs[i], obs[i]) != 0.0 for i in eachindex(recs))

    # Wrapped: one-edge Sequential per stratum, origin slot added.
    dsseq = [Sequential(leaf1), Sequential(leaf2)]
    wrows = [(onset = 0.0, delay = r.delay) for r in rows]
    wobs = [[0.0, o[1]] for o in obs]
    wrecs = CensoredDistributions.record_distributions(
        dsseq, wrows; group = group)
    wrapped = sum(logpdf(wrecs[i], wobs[i]) for i in eachindex(wrecs))
    @test grouped ≈ wrapped

    @test CensoredDistributions.batched_event_logpdf(ds, rows; group = group) ≈
          grouped
end

@testitem "bare-leaf grouped honours per-record obs_time truncation" begin
    using CensoredDistributions, Distributions

    # A reserved per-record `obs_time = D` field right-truncates each record's
    # leaf at `D`, threaded THROUGH the grouped (per-stratum) path. The grouped
    # value must equal baking `upper = D` into each record's leaf (the ebola
    # right-truncation), per stratum and per record.
    leaf1 = primary_censored(Gamma(2.0, 1.5), Uniform(0, 1))
    leaf2 = primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1))
    ds = [leaf1, leaf2]
    group = [1, 2, 1]
    D = [8.0, 9.0, 7.0]
    y = [2.0, 3.0, 4.0]
    rows = [(delay = y[i], obs_time = D[i]) for i in eachindex(y)]

    grouped = CensoredDistributions.batched_event_logpdf(ds, rows; group = group)

    # `upper = D` baked into the leaf (the explicit right-truncation).
    upper_baked = sum(
        logpdf(truncated(ds[group[i]]; upper = D[i]), y[i])
    for i in eachindex(y))
    @test isapprox(grouped, upper_baked; atol = 1e-6)

    # Equal to the internal `truncate_to_horizon` primitive (exact).
    horizon_baked = sum(
        logpdf(CensoredDistributions.truncate_to_horizon(ds[group[i]], D[i]),
            y[i])
    for i in eachindex(y))
    @test grouped ≈ horizon_baked

    # The truncation actually moves the value (a real data constraint).
    untrunc = CensoredDistributions.batched_event_logpdf(
        ds, [(delay = y[i],) for i in eachindex(y)]; group = group)
    @test grouped != untrunc
end

@testitem "bare-leaf grouped one stratum == shared-leaf path" begin
    using CensoredDistributions, Distributions

    leaf = primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1))
    rows = [(delay = 2.0,), (delay = 3.0,), (delay = 4.0,)]
    obs = [[2.0], [3.0], [4.0]]
    group = [1, 1, 1]

    grouped = CensoredDistributions.record_distributions(
        [leaf], rows; group = group)
    shared = CensoredDistributions.record_distributions(leaf, rows)
    @test all(logpdf(grouped[i], obs[i]) == logpdf(shared[i], obs[i])
    for i in eachindex(obs))
end

@testitem "bare-leaf grouped AD-safe gradient" tags=[:turing] begin
    using CensoredDistributions, Distributions, ForwardDiff

    # The grouped bare-leaf scoring must differentiate w.r.t. the per-stratum
    # params (the group key is an integer data-pass id, never keyed on a float).
    rows = [(delay = 2.0,), (delay = 3.0,), (delay = 4.0,)]
    group = [1, 2, 1]

    function nll(theta)
        mk(scale) = primary_censored(Gamma(2.0, scale), Uniform(0, 1))
        ds = [mk(theta[1]), mk(theta[2])]
        return -CensoredDistributions.batched_event_logpdf(
            ds, rows; group = group)
    end

    g = ForwardDiff.gradient(nll, [1.5, 2.0])
    @test length(g) == 2
    @test all(isfinite, g)
    @test any(!iszero, g)
end

# ---------------------------------------------------------------------------
# Batched record-aware rand: the forward-simulation dual to the scoring path.
# ---------------------------------------------------------------------------

@testitem "batched rand(d, rows) draws one labelled path per row" begin
    using CensoredDistributions, Distributions, Random

    # A named Sequential chain: each event chains off the previous, so a drawn
    # path is monotone (onset <= admit <= death).
    seq = Sequential(
        (primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
            primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))),
        (:onset_admit, :admit_death))
    rows = [(onset = missing, admit = missing, death = missing),
        (onset = missing, admit = missing, death = missing),
        (onset = missing, admit = missing, death = missing)]

    draws = rand(Random.Xoshiro(1), seq, rows)
    @test length(draws) == length(rows)
    @test all(d -> d isa NamedTuple, draws)
    @test all(d -> keys(d) == (:onset, :admit, :death), draws)
    # Each labelled path is a monotone event sequence (a chained Sequential).
    @test all(d -> d.onset >= 0 && d.admit >= d.onset && d.death >= d.admit,
        draws)
end

@testitem "batched rand row equals the record's own rand (same rng)" begin
    using CensoredDistributions, Distributions, Random

    seq = compose((onset_admit = primary_censored(
            LogNormal(1.0, 0.4), Uniform(0, 1)),
        admit_death = primary_censored(Gamma(1.5, 1.2), Uniform(0, 1))))
    rows = [(onset = missing, admit = missing, death = missing),
        (onset = missing, admit = missing, death = missing)]

    # The batched rand of a row must equal the single `rand` of that row's
    # `record_distributions` entry under the SAME seeded rng.
    recs = CensoredDistributions.record_distributions(seq, rows)
    rng1 = Random.Xoshiro(42)
    rng2 = Random.Xoshiro(42)
    batched = rand(rng1, seq, rows)
    perrec = [rand(rng2, r) for r in recs]
    for i in eachindex(rows)
        @test collect(values(batched[i])) == perrec[i]
    end
end

@testitem "batched rand is reproducible under a seeded rng" begin
    using CensoredDistributions, Distributions, Random

    par = Parallel(
        primary_censored(LogNormal(1.0, 0.5), Uniform(0, 1)),
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))
    rows = [(event_1 = missing, event_2 = missing, event_3 = missing)
            for _ in 1:5]

    a = rand(Random.Xoshiro(7), par, rows)
    b = rand(Random.Xoshiro(7), par, rows)
    @test a == b
end

@testitem "batched rand routes a Choose tree per row by :kind" begin
    using CensoredDistributions, Distributions, Random

    # A Choose top: an index case (short delay) vs a sourced case (longer
    # delay), selected by the row's `:kind`.
    d = choose(
        :index => primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)),
        :sourced => primary_censored(Gamma(4.0, 1.5), Uniform(0, 1)))
    # Each row selects a DIFFERENT alternative by :kind; the draw must come from
    # the alternative that row routes to. The index alternative is Gamma(2, 1)
    # (mean 2); the sourced is Gamma(4, 1.5) (mean 6), so a large sample of each
    # separates clearly on the mean.
    rows = [(kind = i <= 200 ? :index : :sourced, delay = missing)
            for i in 1:400]

    draws = rand(Random.Xoshiro(3), d, rows)
    @test length(draws) == 400
    idx = [only(values(draws[i])) for i in 1:200]
    src = [only(values(draws[i])) for i in 201:400]
    # The routed-by-kind draws separate on their alternative's mean.
    @test all(>=(0), idx)
    @test all(>=(0), src)
    @test mean(idx) < mean(src)
end

@testitem "batched rand row equals the routed record's rand (Choose)" begin
    using CensoredDistributions, Distributions, Random

    d = choose(
        :index => primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)),
        :sourced => primary_censored(Gamma(4.0, 1.5), Uniform(0, 1)))
    rows = [(kind = :sourced, delay = missing),
        (kind = :index, delay = missing)]

    recs = CensoredDistributions.record_distributions(d, rows)
    rng1 = Random.Xoshiro(99)
    rng2 = Random.Xoshiro(99)
    batched = rand(rng1, d, rows)
    perrec = [rand(rng2, r) for r in recs]
    for i in eachindex(rows)
        @test collect(values(batched[i])) == perrec[i]
    end
end

@testitem "batched rand defaults the rng (no explicit rng arg)" begin
    using CensoredDistributions, Distributions, Random

    seq = Sequential(
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)),
        primary_censored(Gamma(1.5, 1.0), Uniform(0, 1)))
    rows = [(event_1 = missing, event_2 = missing, event_3 = missing)
            for _ in 1:4]

    Random.seed!(123)
    draws = rand(seq, rows)
    @test length(draws) == 4
    @test all(d -> d isa NamedTuple, draws)
    @test all(d -> d.event_3 >= d.event_2 >= d.event_1 >= 0, draws)
end

@testitem "count form rand(d, n) batches n records (no overflow)" begin
    using CensoredDistributions, Distributions, Random

    # Regression for #675: the count form `rand(d, n)` StackOverflowed on the
    # composers (multivariate but `rand(rng, d)` draws a NamedTuple, so the
    # generic `rand(::Multivariate, ::Int)` matrix fallback recursed). It must
    # batch into n independent labelled event records, the same self-describing
    # shape as the record-aware `rand(d, rows)` path.
    seq = Sequential(
        primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))
    rng = MersenneTwister(1)
    draws = rand(rng, seq, 5)
    @test draws isa AbstractVector
    @test length(draws) == 5
    @test all(d -> d isa NamedTuple, draws)
    schema = keys(rand(MersenneTwister(0), seq))
    @test all(d -> keys(d) == schema, draws)
    # A chained Sequential draw is a monotone event path.
    @test all(d -> d.event_3 >= d.event_2 >= d.event_1 >= 0, draws)

    # Parallel and Choose batch too, with the right per-draw schema.
    par = Parallel(
        primary_censored(LogNormal(1.0, 0.5), Uniform(0, 1)),
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))
    pdraws = rand(MersenneTwister(2), par, 3)
    @test length(pdraws) == 3
    @test all(d -> d isa NamedTuple, pdraws)
    @test all(d -> keys(d) == keys(rand(MersenneTwister(0), par)), pdraws)

    # A Choose over univariate leaves draws a scalar per alternative, so the
    # batch is a vector of scalars (each draw matches a single `rand(ch)`).
    ch = choose(
        :index => primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)),
        :sourced => primary_censored(Gamma(4.0, 1.5), Uniform(0, 1)))
    cdraws = rand(MersenneTwister(3), ch, 4)
    @test length(cdraws) == 4
    one_draw = rand(MersenneTwister(0), ch)
    @test all(d -> d isa typeof(one_draw), cdraws)

    # The no-rng count form batches too, and a seeded rng is reproducible.
    @test length(rand(seq, 6)) == 6
    @test rand(MersenneTwister(9), seq, 3) == rand(MersenneTwister(9), seq, 3)
end

# The public front-door `logpdf(d, rows)` scores a whole table / vector of
# records directly, byte-identical to the internal batched path
# (`batched_event_logpdf` / the `record_distributions` product). One PUBLIC
# scoring entry over a table; the per-record event-vector scorer is unchanged.

@testitem "logpdf(d, rows) equals the batched path (leaf, Sequential)" begin
    using CensoredDistributions, Distributions

    # A single-delay model: the leaf wrapped in a one-edge `Sequential` (the
    # canonical composed form). The bare leaf itself keeps the
    # `record_distributions` / `batched_event_logpdf` path (no pirating
    # `logpdf(::leaf, rows)` method), and both forms agree.
    leaf = primary_censored(LogNormal(1.4, 0.5), Uniform(0, 1))
    leaf_seq = Sequential((leaf,), (:onset_delay,))
    leaf_rows = [(onset = 0.0, delay = 2.0), (onset = 0.0, delay = 3.5)]
    @test logpdf(leaf_seq, leaf_rows) ==
          CensoredDistributions.batched_event_logpdf(leaf_seq, leaf_rows)

    # Sequential chain over mixed observed-pattern records.
    seq = compose((onset_admit = primary_censored(
            LogNormal(1.2, 0.5), Uniform(0, 1)),
        admit_death = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))))
    seq_rows = [(onset = 0.0, admit = missing, death = 5.0),
        (onset = 0.5, admit = 2.0, death = 7.0),
        (onset = 0.3, admit = 4.0, death = 9.0, weight = 2.0)]
    @test logpdf(seq, seq_rows) ==
          CensoredDistributions.batched_event_logpdf(seq, seq_rows)

    # A column table (NamedTuple of vectors) is the same multi-record source.
    ct = (onset = [0.0, 0.5], admit = [1.0, 2.0], death = [5.0, 7.0])
    @test logpdf(seq, ct) ==
          CensoredDistributions.batched_event_logpdf(seq, ct)
end

@testitem "logpdf(d, rows) equals the batched path (Parallel)" begin
    using CensoredDistributions, Distributions

    shared = Uniform(0, 1)
    par = Parallel(
        primary_censored(LogNormal(1.0, 0.5), shared),
        primary_censored(Gamma(2.0, 1.0), shared))
    rows = [(event_1 = 0.0, event_2 = 3.0, event_3 = 5.0),
        (event_1 = 0.5, event_2 = missing, event_3 = 6.0),
        (event_1 = 0.3, event_2 = 5.0, event_3 = 8.0, weight = 4.0)]
    @test logpdf(par, rows) ==
          CensoredDistributions.batched_event_logpdf(par, rows)
end

@testitem "logpdf(d, rows) equals the batched path (Resolve tree)" setup=[CaseStudyRecords] begin
    using CensoredDistributions, Distributions

    d = bdbv_tree()
    rows = [
        (onset = 0.0, admit = 4.0, death = 12.0, discharge = missing,
            notif = 9.0, branch_probs = (death = 0.2, discharge = 0.8)),
        (onset = 0.5, admit = 5.0, death = missing, discharge = 11.0,
            notif = 10.0, branch_probs = (death = 0.6, discharge = 0.4)),
        (onset = 1.0, admit = 3.0, death = 14.0, discharge = missing,
            notif = 8.0, branch_probs = 0.45, weight = 2.0)]

    @test logpdf(d, rows) ==
          CensoredDistributions.batched_event_logpdf(d, rows)
    # And equals the per-record loop reference the batched path reproduces.
    @test logpdf(d, rows) ≈ bdbv_ref_loop(d, rows)
end

@testitem "logpdf(d, rows) keeps the single event-vector scorer" begin
    using CensoredDistributions, Distributions

    # A single flat event vector still routes to the per-record event scorer, NOT
    # the table front-door, so the two entries stay distinct.
    seq = Sequential(
        primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))
    ev = Vector{Union{Missing, Float64}}([0.0, 2.0, 5.0])
    single = logpdf(seq, ev)
    @test single isa Real
    @test single == CensoredDistributions.event_logpdf(seq, ev)
    # The one-record table of the same event scores equal to that single value.
    @test logpdf(seq, [(onset = 0.0, admit = 2.0, death = 5.0)]) ≈ single
end
