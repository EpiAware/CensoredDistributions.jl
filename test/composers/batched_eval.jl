# Batched / shared evaluation of a composed distribution (#364, approach 1).
#
# The batched log density `batched_event_logpdf(d, rows)` must EQUAL the
# per-record loop `sum(event_logpdf(d, ev_r; horizon = h_r) * w_r)` exactly,
# while building each shared convolution segment once. These tests pin that
# equality across a Sequential chain, a Parallel set, mixed observed-patterns,
# per-record truncation and weights, gradients, and the DynamicPPL submodel.

@testitem "batched Sequential equals per-record loop" begin
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

    seq = compose((onset_admit = primary_censored(
            LogNormal(1.2, 0.5), Uniform(0, 1)),
        admit_death = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))))

    # Endpoint-observed (intermediate missing) + all-observed rows mixed.
    rows = [(onset = 0.0, admit = missing, death = 5.0),
        (onset = 0.5, admit = missing, death = 7.0),
        (onset = 0.0, admit = 2.0, death = 6.0),
        (onset = 0.3, admit = 4.0, death = 9.0)]

    @test CensoredDistributions.batched_event_logpdf(seq, rows) ≈
          ref_loop(seq, rows)
end

@testitem "batched Sequential honours weights and horizon" begin
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

    seq = compose((onset_admit = primary_censored(
            LogNormal(1.0, 0.4), Uniform(0, 1)),
        admit_death = primary_censored(Gamma(1.5, 1.2), Uniform(0, 1))))

    rows = [(onset = 0.0, admit = missing, death = 5.0, weight = 3.0,
            obs_time = 8.0),
        (onset = 0.5, admit = missing, death = 7.0, weight = 2.0,
            obs_time = 10.0),
        (onset = 0.0, admit = missing, death = 4.0, count = 5.0)]

    @test CensoredDistributions.batched_event_logpdf(seq, rows) ≈
          ref_loop(seq, rows)
end

@testitem "batched Parallel equals per-record loop" begin
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

    shared = Uniform(0, 1)
    par = Parallel(
        primary_censored(LogNormal(1.0, 0.5), shared),
        primary_censored(Gamma(2.0, 1.0), shared))

    rows = [(event_1 = 0.0, event_2 = 3.0, event_3 = 5.0),
        (event_1 = 0.5, event_2 = missing, event_3 = 6.0),
        (event_1 = 0.0, event_2 = 2.0, event_3 = missing),
        (event_1 = 0.3, event_2 = 5.0, event_3 = 8.0, weight = 4.0)]

    @test CensoredDistributions.batched_event_logpdf(par, rows) ≈
          ref_loop(par, rows)
end

@testitem "batched Parallel with horizon equals per-record loop" begin
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

    shared = Uniform(0, 1)
    par = Parallel(
        primary_censored(LogNormal(1.0, 0.5), shared),
        primary_censored(Gamma(2.0, 1.0), shared))

    rows = [(event_1 = 0.0, event_2 = 3.0, event_3 = 5.0, obs_time = 9.0),
        (event_1 = 0.5, event_2 = missing, event_3 = 6.0, obs_time = 10.0,
            weight = 2.0)]

    @test CensoredDistributions.batched_event_logpdf(par, rows) ≈
          ref_loop(par, rows)
end

@testitem "batched matches per-record submodel sum (DynamicPPL)" begin
    using Distributions
    using DynamicPPL: @model, to_submodel, prefix, logjoint

    seq = compose((onset_admit = primary_censored(
            LogNormal(1.2, 0.5), Uniform(0, 1)),
        admit_death = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))))

    rows = [(onset = 0.0, admit = missing, death = 5.0),
        (onset = 0.5, admit = 3.0, death = 7.0),
        (onset = 0.0, admit = missing, death = 4.0, weight = 2.0)]

    # The batched entry scores the whole table in one submodel.
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

    lp_batched = only(logjoint(batched_demo(seq, rows), (;)))
    lp_loop = only(logjoint(per_record_demo(seq, rows), (;)))
    @test lp_batched ≈ lp_loop
end

@testitem "batched scorer is type-stable (@inferred)" begin
    using Distributions, Test

    seq = compose((onset_admit = primary_censored(
            LogNormal(1.2, 0.5), Uniform(0, 1)),
        admit_death = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))))
    rows = [(onset = 0.0, admit = missing, death = 5.0),
        (onset = 0.5, admit = missing, death = 7.0)]

    recs = CensoredDistributions._collect_records(seq, rows)
    @inferred CensoredDistributions._batched_logpdf_dispatch(seq, recs)
end
