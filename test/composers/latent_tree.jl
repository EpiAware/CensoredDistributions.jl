# The marginal -> latent WRAPPER over a composed tree: `latent_segments(tree)`
# lowers a composer to the per-segment latent `Choose`, and `latent_records(tree,
# rows)` derives the per-segment rows the vectorised latent path scores. The bar
# is that the wrapper is density-identical to the marginal `composed_distribution_
# model` over the same records (the project marginal == latent invariant) and
# reproduces the old hand-rolled bdbv latent scoring exactly.

@testitem "latent_segments names the segments by origin_target" begin
    using CensoredDistributions, Distributions

    dic(d) = double_interval_censored(d;
        primary_event = Uniform(0, 1), interval = 1.0)
    res = resolve(:death => (dic(Gamma(2.0, 3.5)), 0.3),
        :discharge => (dic(Gamma(1.0, 8.0)), 0.7))
    admit = sequential(:onset_admit => dic(Gamma(1.2, 3.0)),
        :admit_resolution => res)
    tree = compose((admit_path = admit,
        onset_notif = dic(Gamma(0.7, 20.0))))

    seg = latent_segments(tree)
    # One alternative per leaf segment, keyed by its origin_target event names.
    @test event_names(seg) ==
          (:onset_admit, :admit_death, :admit_discharge, :onset_notif)
end

@testitem "latent_segments lowers a top-level Choose tree" begin
    using CensoredDistributions, Distributions

    dic(d) = double_interval_censored(d;
        primary_event = Uniform(0, 1), interval = 1.0)
    # A standalone Choose tree: its default alternative defines the segment
    # layout (the alternatives share the event-slot width).
    tree = choose(
        :fast => sequential(:onset_admit => dic(Gamma(1.2, 3.0))),
        :slow => sequential(:onset_admit => dic(Gamma(0.7, 20.0))))

    # One leaf segment, so the wrapper is a single latent chain (not a Choose).
    seg = latent_segments(tree)
    @test event_names(seg) == (:onset, :admit)
    tab = latent_records(tree, [(onset = 0.0, admit = 4.0)])
    @test isequal(tab, [(kind = :onset_admit, onset = missing, admit = 4.0)])
end

@testitem "latent_records derives the observed segment rows" begin
    using CensoredDistributions, Distributions

    dic(d) = double_interval_censored(d;
        primary_event = Uniform(0, 1), interval = 1.0)
    res = resolve(:death => (dic(Gamma(2.0, 3.5)), 0.3),
        :discharge => (dic(Gamma(1.0, 8.0)), 0.7))
    admit = sequential(:onset_admit => dic(Gamma(1.2, 3.0)),
        :admit_resolution => res)
    tree = compose((admit_path = admit,
        onset_notif = dic(Gamma(0.7, 20.0))))

    # A death record (admit observed) and a notif-only record.
    rows = [
        (onset = 0.0, admit = 4.0, death = 12.0, discharge = missing,
            notif = 18.0, branch_probs = (death = 0.3, discharge = 0.7)),
        (onset = 0.0, admit = missing, death = missing, discharge = missing,
            notif = 25.0, branch_probs = (death = 0.4, discharge = 0.6))]
    tab = latent_records(tree, rows)

    # Three segments from the death record, one from the notif-only record.
    @test length(tab) == 4
    # onset->admit segment: origin sampled (missing), observed gap = admit.
    @test isequal(tab[1], (kind = :onset_admit, onset = missing, admit = 4.0))
    # admit->death segment: gap = death - admit, carrying the death branch prob.
    @test isequal(
        tab[2], (kind = :admit_death, admit = missing, death = 8.0,
            branch_prob = 0.3))
    # onset->notif segment of the FIRST record.
    @test isequal(tab[3], (kind = :onset_notif, onset = missing, notif = 18.0))
    # The second record has only its notif segment (no admit -> no resolution).
    @test isequal(tab[4], (kind = :onset_notif, onset = missing, notif = 25.0))
    tab2 = latent_records(tree, [rows[2]])
    @test isequal(tab2, [(kind = :onset_notif, onset = missing, notif = 25.0)])
end

@testitem "latent_segments/records reproduce the hand-rolled bdbv scoring" begin
    using CensoredDistributions, Distributions, Random
    using CensoredDistributions: latent, latent_primary_priors,
                                 latent_observed_logpdf, event

    dic(d) = double_interval_censored(d;
        primary_event = Uniform(0, 1), interval = 1.0)
    res = resolve(:death => (dic(Gamma(2.0, 3.5)), 0.3),
        :discharge => (dic(Gamma(1.0, 8.0)), 0.7))
    admit = sequential(:onset_admit => dic(Gamma(1.2, 3.0)),
        :admit_resolution => res)
    tree = compose((admit_path = admit,
        onset_notif = dic(Gamma(0.7, 20.0))))

    # The OLD hand-rolled bdbv latent machinery, verbatim.
    function delay_leaves(d)
        (onset_admit = event(d, :admit_path, :onset_admit),
            admit_death = event(d, :admit_path, :admit_resolution, :death),
            admit_discharge = event(d, :admit_path, :admit_resolution,
                :discharge),
            onset_notif = event(d, :onset_notif))
    end
    function hand_segments(leaves)
        edge(leaf, name) = latent(sequential(name => leaf))
        choose(:onset_admit => edge(leaves.onset_admit, :onset_admit),
            :admit_death => edge(leaves.admit_death, :admit_death),
            :admit_discharge => edge(leaves.admit_discharge, :admit_discharge),
            :onset_notif => edge(leaves.onset_notif, :onset_notif))
    end
    function hand_rows(r)
        rows = NamedTuple[]
        r.admit !== missing &&
            push!(rows, (kind = :onset_admit, onset = missing, admit = r.admit))
        if r.death !== missing
            push!(rows, (kind = :admit_death, admit = missing,
                death = r.death - r.admit))
        elseif r.discharge !== missing
            push!(rows,
                (kind = :admit_discharge, admit = missing,
                    discharge = r.discharge - r.admit))
        end
        r.notif !== missing &&
            push!(rows, (kind = :onset_notif, onset = missing, notif = r.notif))
        rows
    end

    rng = MersenneTwister(7)
    p = 0.3
    rows = [(onset = 0.0, admit = Float64(rand(rng, 3:6)),
                death = i % 2 == 0 ? Float64(rand(rng, 8:14)) : missing,
                discharge = i % 2 == 1 ? Float64(rand(rng, 6:11)) : missing,
                notif = Float64(rand(rng, 15:25)),
                branch_probs = (death = p, discharge = 1 - p)) for i in 1:6]

    new_seg = latent_segments(tree)
    new_tab = latent_records(tree, rows)
    hand_seg = hand_segments(delay_leaves(tree))
    hand_tab = reduce(vcat, map(hand_rows, rows))

    # The stacked primary priors match row-for-row.
    @test latent_primary_priors(new_seg, new_tab) ==
          latent_primary_priors(hand_seg, hand_tab)

    # The observed logpdf matches once the hand-rolled CFR term (added separately
    # in the old model) is folded in, which the new path does via `branch_prob`.
    primaries = randn(MersenneTwister(99),
        length(latent_primary_priors(hand_seg, hand_tab)))
    cfr = sum(rows) do r
        (r.death === missing && r.discharge === missing) && return 0.0
        r.death !== missing ? log(p) : log(1 - p)
    end
    hand_lp = latent_observed_logpdf(hand_seg, hand_tab, primaries) + cfr
    new_lp = latent_observed_logpdf(new_seg, new_tab, primaries)
    @test isapprox(hand_lp, new_lp; atol = 1e-12)
end

@testitem "marginal == latent: the wrapper is density-identical to the marginal" begin
    using CensoredDistributions, Distributions, Random
    using CensoredDistributions: latent_primary_priors, latent_observed_logpdf,
                                 composed_distribution_model
    using DynamicPPL: logjoint, @model, to_submodel

    dic(d) = double_interval_censored(d;
        primary_event = Uniform(0, 1), interval = 1.0)
    res = resolve(:death => (dic(Gamma(2.0, 3.5)), 0.3),
        :discharge => (dic(Gamma(1.0, 8.0)), 0.7))
    admit = sequential(:onset_admit => dic(Gamma(1.2, 3.0)),
        :admit_resolution => res)
    tree = compose((admit_path = admit,
        onset_notif = dic(Gamma(0.7, 20.0))))

    rng = MersenneTwister(11)
    rows = [(onset = 0.0, admit = Float64(rand(rng, 3:6)),
                death = i % 2 == 0 ? Float64(rand(rng, 8:14)) : missing,
                discharge = i % 2 == 1 ? Float64(rand(rng, 6:11)) : missing,
                notif = Float64(rand(rng, 15:25)),
                branch_probs = (death = 0.3, discharge = 0.7)) for i in 1:8]

    @model marg_model(d, rs) = obs ~ to_submodel(
        composed_distribution_model(d, rs))
    marg = only(logjoint(marg_model(tree, rows), (;)))

    seg = latent_segments(tree)
    tab = latent_records(tree, rows)
    priors = latent_primary_priors(seg, tab)
    # Every admit is observed here, so each segment is observed-to-observed and
    # the latent observed logpdf is primary-independent: it IS the full latent
    # marginal likelihood, which must equal the marginal model's logjoint.
    obs_lp = latent_observed_logpdf(seg, tab, rand.(priors))
    @test isapprox(marg, obs_lp; atol = 1e-8)
end
