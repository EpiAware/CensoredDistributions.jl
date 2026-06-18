# A `shared(:tag, ...)` leaf is TRANSPARENT to scoring: a tree carrying a
# shared censored leaf must score IDENTICALLY to the same tree built from
# independent identical (untagged) leaves. The censored-tree traversal strips
# every other wrapper (`Truncated`, `IntervalCensored`, `Weighted`) to recover a
# leaf's origin primary event and secondary interval; the shared wrapper must be
# stripped the same way. Previously the origin-primary and
# leaf-interval traversals did NOT descend through `Shared`, so a
# `shared(:inc, primary_censored(...))` first step silently scored as an
# UNCENSORED origin and a shared `double_interval_censored` leaf dropped its
# interval discretisation — diverging from the untagged leaf even though
# `cdf`/`logpdf`/`_marginal_core` delegated correctly.

@testitem "shared censored leaf scores identically to an untagged leaf" begin
    using CensoredDistributions, Distributions

    # A nested (Choose-bearing) tree with a shared `:inc` as BOTH the origin edge
    # and the routed alternative. Each position must recover `inc`'s primary event
    # through the shared wrapper, matching the untagged build.
    inc_params = (2.0, 1.0)
    b = primary_censored(Gamma(5.0, 1.0), Uniform(0, 1))
    ev = Vector{Union{Missing, Float64}}([missing, 2.0, 5.0])

    inc = shared(:inc, primary_censored(Gamma(inc_params...), Uniform(0, 1)))
    tagged = Sequential((inc, choose(:a => inc, :b => b)),
        (:onset_admit, :admit_death))

    # Independent, identical, UNtagged leaves in the same positions.
    inc1 = primary_censored(Gamma(inc_params...), Uniform(0, 1))
    inc2 = primary_censored(Gamma(inc_params...), Uniform(0, 1))
    untagged = Sequential((inc1, choose(:a => inc2, :b => b)),
        (:onset_admit, :admit_death))

    @test logpdf(tagged, ev) ≈ logpdf(untagged, ev) atol=1e-12
    # The origin primary is now recovered through the shared wrapper, so the score
    # is NOT the uncensored-origin value the stripped traversal used to produce.
    @test isfinite(logpdf(tagged, ev))
end

@testitem "shared leaf recovers origin primary and interval" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: _origin_primary_event, _leaf_interval,
                                 _tree_primary_event

    bare = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))
    tagged = shared(:inc, bare)
    # The origin primary event survives the shared wrapper, exactly as for the
    # untagged leaf.
    @test _origin_primary_event(tagged) == _origin_primary_event(bare)
    @test _origin_primary_event(tagged) !== nothing
    @test _tree_primary_event(tagged) == _tree_primary_event(bare)

    # The secondary interval survives the shared wrapper for an interval-censored
    # leaf (here through the `Truncated` wrapper `double_interval_censored` adds).
    dic = double_interval_censored(Gamma(2.0, 1.0); upper = 10.0, interval = 1.0)
    @test _leaf_interval(shared(:d, dic)) === _leaf_interval(dic)
    @test _leaf_interval(shared(:d, dic)) !== nothing
end

@testitem "shared censored leaf round-trips through record scoring" begin
    using CensoredDistributions, Distributions

    # A shared `:inc` origin edge plus a Choose, scored per record. The shared and
    # untagged builds must produce the same per-record log densities.
    inc_params = (3.0, 1.0)
    b = primary_censored(Gamma(5.0, 1.0), Uniform(0, 1))
    rows = [(onset = 0.0, admit = 2.0, death = 5.0, kind = :a),
        (onset = 0.0, admit = 1.0, death = 7.0, kind = :b)]
    obs = [[0.0, 2.0, 5.0], [0.0, 1.0, 7.0]]

    inc = shared(:inc, primary_censored(Gamma(inc_params...), Uniform(0, 1)))
    tagged = Sequential((inc, choose(:a => inc, :b => b)),
        (:onset_admit, :admit_death))

    inc1 = primary_censored(Gamma(inc_params...), Uniform(0, 1))
    inc2 = primary_censored(Gamma(inc_params...), Uniform(0, 1))
    untagged = Sequential((inc1, choose(:a => inc2, :b => b)),
        (:onset_admit, :admit_death))

    rt = CensoredDistributions.record_distributions(tagged, rows)
    ru = CensoredDistributions.record_distributions(untagged, rows)
    for i in eachindex(rt)
        @test logpdf(rt[i], obs[i]) ≈ logpdf(ru[i], obs[i]) atol=1e-12
    end
end
