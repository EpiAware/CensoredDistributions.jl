# Structural interventions on a composed tree: intervene / swap_child /
# cut_branch / splice. Tested on the bdbv (nested Competing) and andv (Select)
# trees plus plain composers, checking the edits produce a valid composed
# distribution that scores and `rand`s.

@testsnippet InterveneTrees begin
    using CensoredDistributions, Distributions

    dic(d) = double_interval_censored(d; primary_event = Uniform(0, 1),
        interval = 1.0)

    # A bdbv-style tree: an admission chain (with a nested two-arm Competing
    # resolution) parallel to a notification branch, all doubly interval
    # censored.
    function bdbv_tree(; cfr = 0.3)
        resolution = Competing(:death => (dic(Gamma(2.0, 3.5)), cfr),
            :discharge => (dic(Gamma(1.0, 8.0)), 1 - cfr))
        admit_path = Sequential((dic(Gamma(1.2, 3.0)), resolution),
            (:onset_admit, :admit_resolution))
        return compose((admit_path = admit_path,
            onset_notif = dic(Gamma(0.7, 20.0))))
    end

    # A three-outcome resolution so a cut leaves a valid two-arm node.
    function bdbv_three()
        resolution = Competing(:death => (dic(Gamma(2.0, 3.5)), 0.3),
            :discharge => (dic(Gamma(1.0, 8.0)), 0.4),
            :transfer => (dic(Gamma(1.0, 4.0)), 0.3))
        admit_path = Sequential((dic(Gamma(1.2, 3.0)), resolution),
            (:onset_admit, :admit_resolution))
        return compose((admit_path = admit_path,
            onset_notif = dic(Gamma(0.7, 20.0))))
    end

    # An andv-style Select tree over an index vs a coupled sourced branch.
    function andv_tree()
        index = dic(LogNormal(1.5, 0.4))
        sourced = dic(Sequential(Normal(0.2, 0.6), LogNormal(1.5, 0.4)))
        return CensoredDistributions.select(:index => index,
            :sourced => sourced; selector = :kind)
    end
end

@testitem "intervene replaces a deep node on the bdbv tree" setup=[InterveneTrees] begin
    tree = bdbv_tree()
    tree2 = intervene(tree,
        (:admit_path, :admit_resolution, :death) => dic(Gamma(3.0, 2.0)))
    # Same outer structure, different death arm.
    @test typeof(tree2) == typeof(tree)
    res = get_event(get_event(tree2, :admit_path), :admit_resolution)
    @test event_names(res) == (:death, :discharge)
    # Scores and rands.
    v = [2.0, 5.0, 4.0]
    @test isfinite(logpdf(tree2, v))
    @test logpdf(tree2, v) != logpdf(tree, v)
    @test rand(tree2) isa NamedTuple
end

@testitem "intervene takes multiple edits and a single Symbol path" setup=[InterveneTrees] begin
    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    tree2 = intervene(tree, :onset_admit => Gamma(3.0, 1.5),
        :admit_death => LogNormal(0.8, 0.3))
    @test get_event(tree2, :onset_admit) == Gamma(3.0, 1.5)
    @test get_event(tree2, :admit_death) == LogNormal(0.8, 0.3)
    @test isfinite(logpdf(tree2, [1.5, 2.0]))
end

@testitem "swap_child swaps a named child of a node" setup=[InterveneTrees] begin
    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    tree2 = swap_child(tree, (), :onset_admit => Gamma(4.0, 1.0))
    @test get_event(tree2, :onset_admit) == Gamma(4.0, 1.0)
    @test typeof(tree2) == typeof(tree)
end

@testitem "cut_branch drops a Competing arm and renormalises probs" setup=[InterveneTrees] begin
    tree = bdbv_three()
    tree2 = cut_branch(tree, (:admit_path, :admit_resolution, :transfer))
    res = get_event(get_event(tree2, :admit_path), :admit_resolution)
    @test event_names(res) == (:death, :discharge)
    @test sum(res.branch_probs) ≈ 1.0
    # Original 0.3 / 0.4 renormalise over the kept 0.7 mass.
    @test res.branch_probs[1] ≈ 0.3 / 0.7
    @test res.branch_probs[2] ≈ 0.4 / 0.7
    @test isfinite(logpdf(tree2, [2.0, 5.0, 4.0]))
    @test rand(tree2) isa NamedTuple
end

@testitem "cut_branch drops a Sequential / Parallel step" setup=[InterveneTrees] begin
    tree = compose((a = Gamma(2.0, 1.0), b = LogNormal(0.5, 0.4),
        c = Normal(0.0, 1.0)))
    tree2 = cut_branch(tree, :b)
    @test event_names(tree2) == (:a, :c)
    @test length(tree2) == 2
    @test isfinite(logpdf(tree2, [1.5, 0.0]))
end

@testitem "splice inserts before/after steps at a node" setup=[InterveneTrees] begin
    tree = bdbv_tree()
    tree2 = splice(tree, :onset_notif;
        after = :notif_report => dic(Gamma(1.0, 2.0)))
    spliced = get_event(tree2, :onset_notif)
    @test spliced isa CensoredDistributions.Sequential
    @test event_names(spliced) == (:onset_notif, :notif_report)
    # One extra leaf value.
    @test length(tree2) == length(tree) + 1
    @test isfinite(logpdf(tree2, [2.0, 5.0, 4.0, 1.0]))
end

@testitem "intervene / cut / splice on the andv Select tree" setup=[InterveneTrees] begin
    d = andv_tree()
    @test event_names(d) == (:index, :sourced)

    # Replace the index alternative; the sourced score is untouched.
    d2 = intervene(d, :index => dic(LogNormal(2.0, 0.5)))
    @test typeof(d2) == typeof(d)
    @test logpdf(d2, 3.0; kind = :sourced) == logpdf(d, 3.0; kind = :sourced)
    @test logpdf(d2, 3.0; kind = :index) != logpdf(d, 3.0; kind = :index)
    @test rand(d2; kind = :index) isa Real

    # Splice a reporting step onto the sourced branch.
    d3 = splice(d, :sourced; after = :report => dic(Gamma(1.0, 1.0)))
    @test get_event(d3, :sourced) isa CensoredDistributions.Sequential
end

@testitem "cut_branch on a three-way Select drops an alternative" setup=[InterveneTrees] begin
    d = CensoredDistributions.select(:a => Gamma(2.0, 1.0),
        :b => LogNormal(0.5, 0.4), :c => Normal(0.0, 1.0))
    d2 = cut_branch(d, :b)
    @test event_names(d2) == (:a, :c)
    @test logpdf(d2, 1.0; kind = :a) == logpdf(d, 1.0; kind = :a)
end

@testitem "intervention input validation" setup=[InterveneTrees] begin
    tree = bdbv_tree()
    # Unknown child name.
    @test_throws ArgumentError intervene(tree, :nope => Gamma(1.0, 1.0))
    # Path runs past a leaf.
    @test_throws ArgumentError intervene(tree,
        (:onset_notif, :deeper) => Gamma(1.0, 1.0))
    # Cutting a two-arm Competing below the minimum.
    @test_throws ArgumentError cut_branch(tree,
        (:admit_path, :admit_resolution, :death))
    # Splice with no step.
    @test_throws ArgumentError splice(tree, :onset_notif)
    # Empty cut path.
    @test_throws ArgumentError cut_branch(tree, ())
end
