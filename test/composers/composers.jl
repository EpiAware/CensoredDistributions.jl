@testitem "Sequential composes plain dists and scores step values" begin
    using Distributions

    s = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    @test s isa CensoredDistributions.Sequential
    @test length(s) == 2
    # logpdf is the sum of the per-step leaf log-densities.
    @test logpdf(s, [1.5, 2.0]) ≈
          logpdf(Gamma(2.0, 1.0), 1.5) + logpdf(LogNormal(0.5, 0.4), 2.0)
    @test pdf(s, [1.5, 2.0]) ≈ exp(logpdf(s, [1.5, 2.0]))
    @test length(rand(s)) == 2
    # vector and varargs constructors agree.
    @test Sequential([Gamma(2.0, 1.0), LogNormal(0.5, 0.4)]) == s

    @test_throws DimensionMismatch logpdf(s, [1.0])
    @test_throws ArgumentError Sequential(())
end

@testitem "Parallel composes plain dists and sums branch logpdfs" begin
    using Distributions

    p = Parallel(Gamma(2.0, 1.0), LogNormal(1.0, 0.5))
    @test p isa CensoredDistributions.Parallel
    @test length(p) == 2
    @test logpdf(p, [2.0, 3.0]) ≈
          logpdf(Gamma(2.0, 1.0), 2.0) + logpdf(LogNormal(1.0, 0.5), 3.0)
    @test pdf(p, [2.0, 3.0]) ≈ exp(logpdf(p, [2.0, 3.0]))
    @test length(rand(p)) == 2
    @test Parallel([Gamma(2.0, 1.0), LogNormal(1.0, 0.5)]) == p

    @test_throws DimensionMismatch logpdf(p, [1.0, 2.0, 3.0])
    @test_throws ArgumentError Parallel(())
end

@testitem "Competing lowers to a MixtureModel" begin
    using Distributions

    cfr = 0.3
    c = Competing(:death => (Gamma(1.5, 1.0), cfr),
        :disch => (Gamma(2.0, 1.5), 1 - cfr))
    @test c isa CensoredDistributions.Competing
    mix = as_mixture(c)
    @test mix isa MixtureModel
    # The univariate interface delegates to the mixture lowering.
    @test logpdf(c, 2.0) ≈ logpdf(mix, 2.0)
    @test pdf(c, 2.0) ≈ pdf(mix, 2.0)
    @test cdf(c, 2.0) ≈ cdf(mix, 2.0)
    @test mean(c) ≈ mean(mix)
    @test minimum(c) == minimum(mix)

    # Branch probabilities must be valid and sum to one; >= 2 outcomes.
    @test_throws ArgumentError Competing(:a => (Gamma(1.0, 1.0), 0.5))
    @test_throws ArgumentError Competing(
        :a => (Gamma(1.0, 1.0), 0.3), :b => (Gamma(1.0, 1.0), 0.3))
    @test_throws ArgumentError Competing(
        :a => (Gamma(1.0, 1.0), 1.5), :b => (Gamma(1.0, 1.0), -0.5))
end

@testitem "Composers nest recursively (the nesting is the tree)" begin
    using Distributions

    c = Competing(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.7))
    # Competing (univariate) inside Sequential inside Parallel.
    nested = Parallel(Sequential(Gamma(2.0, 1.0), c), LogNormal(1.0, 0.5))
    @test length(nested) == 3   # two leaves in the chain plus one branch

    r = rand(nested)
    @test length(r) == 3

    # logpdf consumes a contiguous slice per child and recurses.
    lp = logpdf(nested, [1.5, 2.0, 3.0])
    @test lp ≈
          logpdf(Gamma(2.0, 1.0), 1.5) + logpdf(c, 2.0) +
          logpdf(LogNormal(1.0, 0.5), 3.0)

    # A nested Parallel inside a Parallel flattens its leaf values.
    deep = Parallel(Parallel(Gamma(1.0, 1.0), Gamma(2.0, 1.0)), Normal(0.0, 1.0))
    @test length(deep) == 3
    @test logpdf(deep, [0.5, 1.5, 0.0]) ≈
          logpdf(Gamma(1.0, 1.0), 0.5) + logpdf(Gamma(2.0, 1.0), 1.5) +
          logpdf(Normal(0.0, 1.0), 0.0)
end

@testitem "compose: NamedTuple, table and matrix are structurally equal" begin
    using Distributions

    # The three front-ends build the same nested STRUCTURE; their node names may
    # differ (each format carries its own), and `==` compares structure only
    # (#351, Option A). These assertions exercise that relaxed equivalence.

    # --- irregular tree: NamedTuple, column table, row table agree ---
    nt = (a = Gamma(2.0, 1.0), b = [LogNormal(0.5, 0.4), Gamma(1.0, 1.0)])
    column_table = (name = [:a, :b1, :b2],
        dist = [Gamma(2.0, 1.0), LogNormal(0.5, 0.4), Gamma(1.0, 1.0)],
        chain = [0, 1, 1])
    row_table = [(name = :a, dist = Gamma(2.0, 1.0), chain = 0),
        (name = :b1, dist = LogNormal(0.5, 0.4), chain = 1),
        (name = :b2, dist = Gamma(1.0, 1.0), chain = 1)]
    target = Parallel(Gamma(2.0, 1.0),
        Sequential(LogNormal(0.5, 0.4), Gamma(1.0, 1.0)))

    @test compose(nt) == target
    @test compose(column_table) == target
    @test compose(row_table) == target
    @test compose(nt) == compose(column_table) == compose(row_table)

    # --- regular grid: NamedTuple, table and Matrix agree ---
    nt2 = (r1 = [Gamma(2.0, 1.0), LogNormal(0.5, 0.4)],
        r2 = [Gamma(1.0, 1.0), Gamma(3.0, 1.0)])
    table2 = (name = [:a, :b, :c, :d],
        dist = [Gamma(2.0, 1.0), LogNormal(0.5, 0.4),
            Gamma(1.0, 1.0), Gamma(3.0, 1.0)],
        chain = [1, 1, 2, 2])
    mat2 = [Gamma(2.0, 1.0) LogNormal(0.5, 0.4)
            Gamma(1.0, 1.0) Gamma(3.0, 1.0)]
    target2 = Parallel(Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4)),
        Sequential(Gamma(1.0, 1.0), Gamma(3.0, 1.0)))

    @test compose(nt2) == target2
    @test compose(table2) == target2
    @test compose(mat2) == target2
    @test compose(nt2) == compose(table2) == compose(mat2)
end

@testitem "compose: matrix orientation and flat equivalences" begin
    using Distributions

    # Rows are parallel branches, columns are sequential steps.
    flat_parallel = compose(reshape(
        [Gamma(2.0, 1.0), LogNormal(1.0, 0.5)], 2, 1))
    @test flat_parallel == compose((a = Gamma(2.0, 1.0), b = LogNormal(1.0, 0.5)))
    @test flat_parallel ==
          compose((name = [:a, :b], dist = [Gamma(2.0, 1.0), LogNormal(1.0, 0.5)]))

    # A one-row matrix is a single sequential branch.
    chain = compose([Gamma(2.0, 1.0) LogNormal(0.5, 0.4) Gamma(1.0, 1.0)])
    @test chain == Parallel(Sequential(
        Gamma(2.0, 1.0), LogNormal(0.5, 0.4), Gamma(1.0, 1.0)))

    # A composed stack scores and samples like any composer.
    d = compose((r1 = [Gamma(2.0, 1.0), LogNormal(0.5, 0.4)],
        r2 = [Gamma(1.0, 1.0), Gamma(3.0, 1.0)]))
    @test length(rand(d)) == 4
    @test isfinite(logpdf(d, rand(d)))
end

@testitem "compose: input validation" begin
    using Distributions

    @test_throws ArgumentError compose((name = [:a], extra = [1]))
    @test_throws ArgumentError compose((name = [:a], dist = [1.0]))
    @test_throws ArgumentError compose(42)
    # A NamedTuple sequential child must hold distributions.
    @test_throws ArgumentError compose((a = [1.0, 2.0],))
    # Negative chain group ids would collide with auto-generated leaf ids.
    @test_throws ArgumentError compose((name = [:a, :b],
        dist = [Gamma(2.0, 1.0), LogNormal(0.5, 0.4)], chain = [-1, -1]))
end

@testitem "compose accepts pre-built composer children" begin
    using CensoredDistributions, Distributions
    const CD = CensoredDistributions

    # A compose result drops into a parent compose as a child and builds the same
    # structure as the equivalent direct constructor calls.
    inner = compose((a = Gamma(2.0, 1.0), b = LogNormal(0.5, 0.4)))
    outer = compose((x = inner, y = Normal(0.0, 1.0)))
    @test outer == CD.Parallel(
        CD.Parallel(Gamma(2.0, 1.0), LogNormal(0.5, 0.4)), Normal(0.0, 1.0))

    # A select result nests as a compose child.
    sel = select(:s1 => Gamma(2.0, 1.0), :s2 => Gamma(5.0, 1.0))
    withsel = compose((k = sel, m = Normal(0.0, 1.0)))
    @test withsel == CD.Parallel(sel, Normal(0.0, 1.0))

    # A composer is allowed inside a Vector chain step (no longer leaf-only).
    chained = compose((leg = [Gamma(2.0, 1.0),
        CD.Parallel(Gamma(1.0, 1.0), Gamma(2.0, 1.0))],))
    @test chained == CD.Parallel(CD.Sequential(
        Gamma(2.0, 1.0), CD.Parallel(Gamma(1.0, 1.0), Gamma(2.0, 1.0))))
    @test length(chained) == 3   # leaf step plus two-branch parallel step

    # The whole stack scores and samples like any composer.
    @test length(rand(outer)) == 3
    @test isfinite(logpdf(outer, rand(outer)))
end

@testitem "compose keeps user step names on a named chain" begin
    using CensoredDistributions, Distributions
    const CD = CensoredDistributions

    # A Sequential value carrying names nests as a chain child WITHOUT the
    # (name, dist, chain) table form, keeping readable step names.
    named = CD.Sequential((LogNormal(1.5, 0.4), Gamma(2.0, 1.0)),
        (:onset_admit, :admit_death))
    tree = compose((path = named, other = Normal(0.0, 1.0)))
    chain = tree.components[1]
    @test CD.component_names(chain) == (:onset_admit, :admit_death)
    @test CD.component_names(tree) == (:path, :other)
end

@testitem "Composers show readably" begin
    using Distributions

    s = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    p = Parallel(Gamma(2.0, 1.0), LogNormal(1.0, 0.5))
    c = Competing(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.7))

    @test occursin("Sequential", sprint(show, s))
    @test occursin("Sequential (2 steps)", sprint(show, MIME"text/plain"(), s))
    @test occursin("Parallel", sprint(show, p))
    @test occursin(
        "Parallel (2 branches)", sprint(show, MIME"text/plain"(), p))
    @test occursin("Competing", sprint(show, c))
    @test occursin("death", sprint(show, MIME"text/plain"(), c))
end

@testitem "Composers show as a recursive indented tree" begin
    using Distributions

    # A nested stack must print the WHOLE structure as one indented tree, with
    # the child node types/names at the right depth (not just the top level).
    d = Parallel(
        Gamma(2.0, 1.0),
        Sequential(LogNormal(0.5, 0.4), Gamma(3.0, 1.0)),
        Competing(:death => (Gamma(1.5, 1.0), 0.3),
            :disch => (Gamma(2.0, 1.5), 0.7)))
    out = sprint(show, MIME"text/plain"(), d)
    lines = split(out, '\n')

    # Root node header and its three branches.
    @test occursin("Parallel (3 branches)", lines[1])
    # The nested Sequential and Competing children are rendered (recursively),
    # each indented one level under the root with a tree connector.
    seq_line = findfirst(l -> occursin("Sequential (2 steps)", l), lines)
    comp_line = findfirst(l -> occursin("Competing (2 outcomes)", l), lines)
    @test seq_line !== nothing
    @test comp_line !== nothing
    @test occursin("├─ ", lines[seq_line])
    @test occursin("└─ ", lines[comp_line])

    # The Sequential's own leaf steps are indented one further level (the
    # continuation prefix `│  ` carries the parent connector down).
    @test any(l -> occursin("│  ", l) && occursin("LogNormal", l), lines)
    @test any(l -> occursin("│  ", l) && occursin("Gamma", l), lines)

    # The Competing outcomes carry their names and branch probabilities, nested
    # under the last branch (so a spaces continuation prefix, not `│`).
    @test any(l -> occursin("death (p = 0.3)", l), lines)
    @test any(l -> occursin("disch (p = 0.7)", l), lines)

    # A composer nested inside Competing also recurses: a Competing outcome may
    # itself be a (univariate) Competing, whose subtree then prints nested.
    # (Competing outcomes are univariate by design, so a Sequential/Parallel
    # cannot be an outcome; a nested Competing can.)
    nested_comp = Sequential(
        Gamma(2.0, 1.0),
        Competing(
            :a => (Competing(:a1 => (Normal(0.0, 1.0), 0.5),
                    :a2 => (Normal(1.0, 1.0), 0.5)),
                0.4),
            :b => (Normal(2.0, 1.0), 0.6)))
    out2 = sprint(show, MIME"text/plain"(), nested_comp)
    @test occursin("Sequential (2 steps)", out2)
    # Both the outer and the nested Competing headers appear.
    @test count("Competing (2 outcomes)", out2) == 2
    # The outer Competing outcome `a` is itself a Competing, nested under it.
    @test occursin("a (p = 0.4): Competing (2 outcomes)", out2)
    @test occursin("a1 (p = 0.5)", out2)
end

@testitem "compose threads names through every input format (#351)" begin
    using Distributions
    const CD = CensoredDistributions

    nt = (onset_admit = LogNormal(1.5, 0.4), admit_death = Gamma(2.0, 1.0))
    tbl = (name = [:onset_admit, :admit_death],
        dist = [LogNormal(1.5, 0.4), Gamma(2.0, 1.0)])
    mat = reshape([LogNormal(1.5, 0.4), Gamma(2.0, 1.0)], 2, 1)

    c_nt = compose(nt)
    c_tbl = compose(tbl)
    c_mat = compose(mat; names = (:onset_admit, :admit_death))
    c_mat_default = compose(mat)

    # User names pass through whichever format carries them.
    @test CD.component_names(c_nt) == (:onset_admit, :admit_death)
    @test CD.component_names(c_tbl) == (:onset_admit, :admit_death)
    @test CD.component_names(c_mat) == (:onset_admit, :admit_death)
    # Positional fallback only when no names are supplied.
    @test CD.component_names(c_mat_default) == (:branch_1, :branch_2)

    # Structurally equal regardless of names (relaxed equivalence, #351).
    @test c_nt == c_tbl == c_mat == c_mat_default

    # Chained table: branch named by its first row; steps by their own rows.
    tc = (name = [:incub, :rep, :solo],
        dist = [Gamma(2.0, 1.0), LogNormal(0.5, 0.4), Normal(1.0, 0.5)],
        chain = [1, 1, 0])
    cc = compose(tc)
    @test CD.component_names(cc) == (:incub, :solo)
    @test CD.component_names(cc.components[1]) == (:incub, :rep)

    # Matrix step names label the columns within a multi-step row.
    m2 = [Gamma(2.0, 1.0) LogNormal(0.5, 0.4)
          Gamma(1.0, 1.0) Gamma(3.0, 1.0)]
    cm = compose(m2; names = (:r1, :r2), step_names = (:s1, :s2))
    @test CD.component_names(cm) == (:r1, :r2)
    @test CD.component_names(cm.components[1]) == (:s1, :s2)
end

@testitem "params is nested and name-keyed for composed dists (#351)" begin
    using Distributions

    tree = compose((onset_admit = LogNormal(1.5, 0.4),
        admit_death = Gamma(2.0, 1.0)))
    @test params(tree) == (onset_admit = (1.5, 0.4), admit_death = (2.0, 1.0))

    # Recurses into a nested chain (default step names) and another NamedTuple.
    nested = compose((
        leaf = Normal(0.0, 1.0),
        chain = [Gamma(2.0, 1.0), LogNormal(1.0, 0.5)],
        sub = (a = Normal(0.0, 1.0), b = Gamma(1.0, 2.0))))
    p = params(nested)
    @test keys(p) == (:leaf, :chain, :sub)
    @test p.leaf == (0.0, 1.0)
    @test p.chain == (step_1 = (2.0, 1.0), step_2 = (1.0, 0.5))
    @test p.sub == (a = (0.0, 1.0), b = (1.0, 2.0))

    # Competing contributes name-keyed outcomes plus branch_probs.
    c = Competing(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.7))
    pc = CensoredDistributions._competing_params(c)
    @test pc.death == (1.5, 1.0)
    @test pc.disch == (2.0, 1.5)
    @test pc.branch_probs == (0.3, 0.7)
end

@testitem "params_table flattens to edge|param|value|support (#351)" begin
    using Distributions
    # `Tables` is a CensoredDistributions dependency; reach it through the
    # package rather than adding a direct test dep.
    const Tables = CensoredDistributions.Tables

    tree = compose((onset_admit = LogNormal(1.5, 0.4),
        admit_death = Gamma(2.0, 1.0)))
    tbl = params_table(tree)

    # It is a Tables.jl table with the documented columns.
    @test Tables.istable(tbl)
    @test Tables.columnnames(Tables.columns(tbl)) ==
          (:edge, :param, :value, :support)

    @test tbl.edge == [:onset_admit, :onset_admit, :admit_death, :admit_death]
    @test tbl.param == [:mu, :sigma, :shape, :scale]
    @test tbl.value == [1.5, 0.4, 2.0, 1.0]
    # LogNormal and Gamma are positive-supported delays.
    @test all(s -> s == (0.0, Inf), tbl.support)

    # Constraints reflect each edge's support: a Normal edge is unbounded.
    ntree = compose((x = Normal(0.0, 1.0),))
    ntbl = params_table(ntree)
    @test all(s -> s == (-Inf, Inf), ntbl.support)

    # Competing branch probabilities appear as [0, 1]-supported rows.
    ctree = Parallel(
        (Gamma(2.0, 1.0),
            Competing(:death => (Gamma(1.5, 1.0), 0.3),
                :disch => (Gamma(2.0, 1.5), 0.7))),
        (:incub, :resolution))
    ct = params_table(ctree)
    bp = findall(==(Symbol("resolution.branch_probs")), ct.edge)
    @test length(bp) == 2
    @test all(i -> ct.support[i] == (0.0, 1.0), bp)
    @test ct.value[bp] == [0.3, 0.7]
end

@testitem "name introspection: event_names and get_event (#351)" begin
    using Distributions

    tree = compose((onset_admit = LogNormal(1.5, 0.4),
        admit_death = Gamma(2.0, 1.0)))
    @test event_names(tree) == (:onset_admit, :admit_death)
    @test get_event(tree, :admit_death) == Gamma(2.0, 1.0)
    @test_throws KeyError get_event(tree, :missing_edge)

    # Competing: outcome names and delays by name.
    c = Competing(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.7))
    @test event_names(c) == (:death, :disch)
    @test get_event(c, :death) == Gamma(1.5, 1.0)
end
