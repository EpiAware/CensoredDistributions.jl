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
