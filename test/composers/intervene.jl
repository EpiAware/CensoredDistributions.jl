# Structural edits on a composed tree: `update` (node replace), `prune`,
# `splice`. Tested on the bdbv (nested Competing) and andv (Select) trees plus
# plain composers, checking the edits produce a valid composed distribution that
# scores and `rand`s. The deprecated `intervene` / `swap_child` / `cut_branch`
# aliases are checked to still call through.

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
        return CensoredDistributions.selecting(:index => index,
            :sourced => sourced; selector = :kind)
    end
end

@testitem "update replaces a deep node on the bdbv tree" setup=[InterveneTrees] begin
    tree = bdbv_tree()
    tree2 = update(tree,
        (:admit_path, :admit_resolution, :death) => dic(Gamma(3.0, 2.0)))
    # Same outer structure, different death arm.
    @test typeof(tree2) == typeof(tree)
    res = event(tree2, :admit_path, :admit_resolution)
    @test keys(event_tree(res)) == (:death, :discharge)
    # Scores and rands.
    v = [2.0, 5.0, 4.0]
    @test isfinite(logpdf(tree2, v))
    @test logpdf(tree2, v) != logpdf(tree, v)
    @test rand(tree2) isa NamedTuple
end

@testitem "update takes multiple node edits and a single Symbol path" setup=[InterveneTrees] begin
    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    tree2 = update(tree, :onset_admit => Gamma(3.0, 1.5),
        :admit_death => LogNormal(0.8, 0.3))
    @test event(tree2, :onset_admit) == Gamma(3.0, 1.5)
    @test event(tree2, :admit_death) == LogNormal(0.8, 0.3)
    @test isfinite(logpdf(tree2, [1.5, 2.0]))
end

@testitem "update node-replace and value-update coexist on one verb" setup=[InterveneTrees] begin
    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    # Pair argument dispatches to the node-replace method.
    node_edit = update(tree, :onset_admit => Gamma(3.0, 1.5))
    @test event(node_edit, :onset_admit) == Gamma(3.0, 1.5)
    # NamedTuple argument dispatches to the value-update method, untouched.
    value_edit = update(tree, (onset_admit = (shape = 3.0, scale = 1.5),
        admit_death = (mu = 0.7, sigma = 0.5)))
    @test event(value_edit, :onset_admit) == Gamma(3.0, 1.5)
    @test event(value_edit, :admit_death) == LogNormal(0.7, 0.5)
end

@testitem "prune drops a Competing arm and renormalises probs" setup=[InterveneTrees] begin
    tree = bdbv_three()
    tree2 = prune(tree, (:admit_path, :admit_resolution, :transfer))
    res = event(tree2, :admit_path, :admit_resolution)
    @test keys(event_tree(res)) == (:death, :discharge)
    @test sum(res.branch_probs) ≈ 1.0
    # Original 0.3 / 0.4 renormalise over the kept 0.7 mass.
    @test res.branch_probs[1] ≈ 0.3 / 0.7
    @test res.branch_probs[2] ≈ 0.4 / 0.7
    @test isfinite(logpdf(tree2, [2.0, 5.0, 4.0]))
    @test rand(tree2) isa NamedTuple
end

@testitem "prune drops a Sequential / Parallel step" setup=[InterveneTrees] begin
    tree = compose((a = Gamma(2.0, 1.0), b = LogNormal(0.5, 0.4),
        c = Normal(0.0, 1.0)))
    tree2 = prune(tree, :b)
    @test keys(event_tree(tree2)) == (:a, :c)
    @test length(tree2) == 2
    @test isfinite(logpdf(tree2, [1.5, 0.0]))
end

@testitem "splice inserts before/after steps at a node" setup=[InterveneTrees] begin
    tree = bdbv_tree()
    tree2 = splice(tree, :onset_notif;
        after = :notif_report => dic(Gamma(1.0, 2.0)))
    spliced = event(tree2, :onset_notif)
    @test spliced isa CensoredDistributions.Sequential
    @test keys(event_tree(spliced)) == (:onset_notif, :notif_report)
    # One extra leaf value.
    @test length(tree2) == length(tree) + 1
    @test isfinite(logpdf(tree2, [2.0, 5.0, 4.0, 1.0]))
end

@testitem "update / prune / splice on the andv Select tree" setup=[InterveneTrees] begin
    d = andv_tree()
    @test event_names(d) == (:index, :sourced)

    # Replace the index alternative; the sourced score is untouched.
    d2 = update(d, :index => dic(LogNormal(2.0, 0.5)))
    @test typeof(d2) == typeof(d)
    @test logpdf(d2, 3.0; kind = :sourced) == logpdf(d, 3.0; kind = :sourced)
    @test logpdf(d2, 3.0; kind = :index) != logpdf(d, 3.0; kind = :index)
    @test rand(d2; kind = :index) isa Real

    # Splice a reporting step onto the sourced branch.
    d3 = splice(d, :sourced; after = :report => dic(Gamma(1.0, 1.0)))
    @test event(d3, :sourced) isa CensoredDistributions.Sequential
end

@testitem "prune on a three-way Select drops an alternative" setup=[InterveneTrees] begin
    d = CensoredDistributions.selecting(:a => Gamma(2.0, 1.0),
        :b => LogNormal(0.5, 0.4), :c => Normal(0.0, 1.0))
    d2 = prune(d, :b)
    @test event_names(d2) == (:a, :c)
    @test logpdf(d2, 1.0; kind = :a) == logpdf(d, 1.0; kind = :a)
end

@testitem "edit input validation" setup=[InterveneTrees] begin
    tree = bdbv_tree()
    # Unknown child name.
    @test_throws ArgumentError update(tree, :nope => Gamma(1.0, 1.0))
    # Path runs past a leaf.
    @test_throws ArgumentError update(tree,
        (:onset_notif, :deeper) => Gamma(1.0, 1.0))
    # Cutting a two-arm Competing below the minimum.
    @test_throws ArgumentError prune(tree,
        (:admit_path, :admit_resolution, :death))
    # Splice with no step.
    @test_throws ArgumentError splice(tree, :onset_notif)
    # Empty cut path.
    @test_throws ArgumentError prune(tree, ())
end

@testitem "path forms for update / prune / splice match event" setup=[InterveneTrees] begin
    tree = bdbv_three()
    new = dic(Gamma(3.0, 2.0))
    tup = (:admit_path, :admit_resolution, :death)
    dotted = Symbol("admit_path.admit_resolution.death")

    # `event` reads the same address `update` writes: varargs and dotted forms
    # resolve to the same node.
    @test event(tree, tup...) == event(tree, dotted)

    # update: tuple, dotted Symbol all hit the same node.
    u_tup = update(tree, tup => new)
    u_dot = update(tree, dotted => new)
    @test u_tup == u_dot
    @test event(u_tup, tup...) == new

    # prune: varargs, dotted Symbol, tuple all drop the same arm.
    p_var = prune(tree, :admit_path, :admit_resolution, :transfer)
    p_dot = prune(tree, Symbol("admit_path.admit_resolution.transfer"))
    p_tup = prune(tree, (:admit_path, :admit_resolution, :transfer))
    @test p_var == p_dot == p_tup

    # splice: varargs, dotted Symbol, tuple all wrap the same node.
    step = :extra => dic(Gamma(1.0, 1.0))
    s_var = splice(tree, :admit_path, :admit_resolution, :death; after = step)
    s_dot = splice(tree, Symbol("admit_path.admit_resolution.death");
        after = step)
    s_tup = splice(tree, (:admit_path, :admit_resolution, :death);
        after = step)
    @test s_var == s_dot == s_tup
end

@testitem "sequential / parallel build the same objects as the structs" setup=[InterveneTrees] begin
    # Named pairs match the two-arg struct constructor (components + names).
    s = sequential(:a => Gamma(2.0, 1.0), :b => LogNormal(0.5, 0.4))
    s_struct = Sequential((Gamma(2.0, 1.0), LogNormal(0.5, 0.4)), (:a, :b))
    @test s == s_struct
    @test CensoredDistributions.component_names(s) ==
          CensoredDistributions.component_names(s_struct)

    p = parallel(:x => Gamma(2.0, 1.0), :y => LogNormal(1.0, 0.5))
    p_struct = Parallel((Gamma(2.0, 1.0), LogNormal(1.0, 0.5)), (:x, :y))
    @test p == p_struct
    @test CensoredDistributions.component_names(p) ==
          CensoredDistributions.component_names(p_struct)

    # Positional steps match the default-named struct constructor.
    sp = sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    @test sp == Sequential((Gamma(2.0, 1.0), LogNormal(0.5, 0.4)))
    @test CensoredDistributions.component_names(sp) == (:step_1, :step_2)

    pp = parallel(Gamma(2.0, 1.0), LogNormal(1.0, 0.5))
    @test pp == Parallel((Gamma(2.0, 1.0), LogNormal(1.0, 0.5)))
    @test CensoredDistributions.component_names(pp) == (:branch_1, :branch_2)

    # A vector of steps/branches matches the vector struct constructor.
    sv = sequential([Gamma(2.0, 1.0), LogNormal(0.5, 0.4)])
    @test sv == Sequential([Gamma(2.0, 1.0), LogNormal(0.5, 0.4)])
end

@testitem "deprecated intervene / swap_child / cut_branch call through" setup=[InterveneTrees] begin
    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    # `intervene` -> `update` node-replace.
    @test intervene(tree, :admit_death => Gamma(3.0, 1.5)) ==
          update(tree, :admit_death => Gamma(3.0, 1.5))
    # `swap_child` -> `update` with the full path.
    @test swap_child(tree, (), :onset_admit => Gamma(4.0, 1.0)) ==
          update(tree, (:onset_admit,) => Gamma(4.0, 1.0))
    # `cut_branch` -> `prune`.
    node = competing(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.5),
        :transfer => (Gamma(1.0, 1.0), 0.2))
    tt = compose((resolution = node, onset = Gamma(1.0, 1.0)))
    @test cut_branch(tt, (:resolution, :transfer)) ==
          prune(tt, (:resolution, :transfer))
end
