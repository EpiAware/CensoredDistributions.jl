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

@testitem "Composers reject wrong-shape logpdf/pdf with a clear message" begin
    using Distributions

    s = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    p = Parallel(Gamma(2.0, 1.0), LogNormal(1.0, 0.5))

    # A scalar where the per-step/branch value vector is expected used to hit a
    # confusing MethodError; it now names the expected length and keys.
    err = try
        logpdf(s, 3.0)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("Sequential", err.msg)
    @test occursin("length-2", err.msg)
    @test occursin("NamedTuple", err.msg)
    @test occursin("multivariate", err.msg)

    @test_throws ArgumentError logpdf(s, 3.0)
    @test_throws ArgumentError pdf(s, 3.0)
    @test_throws ArgumentError logpdf(p, 3.0)
    @test_throws ArgumentError pdf(p, 3.0)

    # A censored composer names the flat EVENT vector form.
    oa = primary_censored(LogNormal(1.5, 0.4), Uniform(0, 1))
    ad = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))
    tree = compose((onset_admit = [oa, ad],))
    cerr = try
        logpdf(tree, 3.0)
        nothing
    catch e
        e
    end
    @test cerr isa ArgumentError
    @test occursin("event vector", cerr.msg)

    # A wrong-length vector names the expected length.
    derr = try
        logpdf(s, [1.0])
        nothing
    catch e
        e
    end
    @test derr isa DimensionMismatch
    @test occursin("length-2", derr.msg)
    @test occursin("length 1", derr.msg)

    # Correct calls are unchanged.
    @test logpdf(s, [1.5, 2.0]) ≈
          logpdf(Gamma(2.0, 1.0), 1.5) + logpdf(LogNormal(0.5, 0.4), 2.0)
    @test logpdf(s, rand(s)) isa Real
end

@testitem "Records reject wrong-length event vectors clearly" begin
    using Distributions

    seq = Sequential(
        primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))
    recs = CensoredDistributions.record_distributions(
        seq, [(onset = 0.0, admit = 2.0, death = 5.0)])
    r = recs[1]
    @test length(r) == 3
    @test logpdf(r, [0.0, 2.0, 5.0]) isa Real

    # A wrong-length event vector used to hit a deep BoundsError; it now names
    # the expected slot count.
    err = try
        logpdf(r, [0.0, 2.0])
        nothing
    catch e
        e
    end
    @test err isa DimensionMismatch
    @test occursin("length-3", err.msg)
    @test occursin("event vector", err.msg)
    @test_throws DimensionMismatch logpdf(r, [3.0])
end

@testitem "Records error on a missing or unknown event field" begin
    using Distributions

    oa = primary_censored(LogNormal(1.5, 0.4), Uniform(0, 1))
    ad = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))
    seq = sequential(:onset_admit => oa, :admit_death => ad)
    @test CensoredDistributions.event_names(seq) == (:onset, :admit, :death)

    # A row missing a required event field names it.
    merr = try
        CensoredDistributions.record_distributions(
            seq, [(onset = 0.0, admit = 2.0)])
        nothing
    catch e
        e
    end
    @test merr isa ArgumentError
    @test occursin(":death", merr.msg)

    # A row carrying an unknown field names it.
    uerr = try
        CensoredDistributions.record_distributions(
            seq, [(onset = 0.0, admit = 2.0, death = 5.0, bogus = 1.0)])
        nothing
    catch e
        e
    end
    @test uerr isa ArgumentError
    @test occursin(":bogus", uerr.msg)

    # The same missing-field error fires on the NamedTuple logpdf path.
    @test_throws ArgumentError logpdf(seq, (onset = 0.0, admit = 2.0))
end

@testitem "Resolve lowers to a MixtureModel" begin
    using Distributions

    cfr = 0.3
    c = Resolve(:death => (Gamma(1.5, 1.0), cfr),
        :disch => (Gamma(2.0, 1.5), 1 - cfr))
    @test c isa CensoredDistributions.Resolve
    mix = as_mixture(c)
    @test mix isa MixtureModel
    # The univariate interface delegates to the mixture lowering.
    @test logpdf(c, 2.0) ≈ logpdf(mix, 2.0)
    @test pdf(c, 2.0) ≈ pdf(mix, 2.0)
    @test cdf(c, 2.0) ≈ cdf(mix, 2.0)
    @test mean(c) ≈ mean(mix)
    @test minimum(c) == minimum(mix)

    # Branch probabilities must be valid and sum to one; >= 2 outcomes.
    @test_throws ArgumentError Resolve(:a => (Gamma(1.0, 1.0), 0.5))
    @test_throws ArgumentError Resolve(
        :a => (Gamma(1.0, 1.0), 0.3), :b => (Gamma(1.0, 1.0), 0.3))
    @test_throws ArgumentError Resolve(
        :a => (Gamma(1.0, 1.0), 1.5), :b => (Gamma(1.0, 1.0), -0.5))
end

@testitem "Composers nest recursively (the nesting is the tree)" begin
    using Distributions

    c = Resolve(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.7))
    # Resolve (univariate) inside Sequential inside Parallel.
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

@testitem "compose: NamedTuple and table are structurally equal" begin
    using Distributions

    # Both front-ends build the same nested STRUCTURE; their node names may
    # differ (each format carries its own), and `==` compares structure only
    # (Option A). These assertions exercise that relaxed equivalence. A column
    # table is passed as an explicit row table (a Tables.jl source); a
    # NamedTuple is always read structurally, never as a column table.

    # --- irregular tree: NamedTuple and row table agree ---
    nt = (a = Gamma(2.0, 1.0), b = [LogNormal(0.5, 0.4), Gamma(1.0, 1.0)])
    row_table = [(name = :a, dist = Gamma(2.0, 1.0), chain = 0),
        (name = :b1, dist = LogNormal(0.5, 0.4), chain = 1),
        (name = :b2, dist = Gamma(1.0, 1.0), chain = 1)]
    target = Parallel(Gamma(2.0, 1.0),
        Sequential(LogNormal(0.5, 0.4), Gamma(1.0, 1.0)))

    @test compose(nt) == target
    @test compose(row_table) == target
    @test compose(nt) == compose(row_table)

    # --- regular grid: NamedTuple and table agree ---
    nt2 = (r1 = [Gamma(2.0, 1.0), LogNormal(0.5, 0.4)],
        r2 = [Gamma(1.0, 1.0), Gamma(3.0, 1.0)])
    table2 = [(name = :a, dist = Gamma(2.0, 1.0), chain = 1),
        (name = :b, dist = LogNormal(0.5, 0.4), chain = 1),
        (name = :c, dist = Gamma(1.0, 1.0), chain = 2),
        (name = :d, dist = Gamma(3.0, 1.0), chain = 2)]
    target2 = Parallel(Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4)),
        Sequential(Gamma(1.0, 1.0), Gamma(3.0, 1.0)))

    @test compose(nt2) == target2
    @test compose(table2) == target2
    @test compose(nt2) == compose(table2)
end

@testitem "compose: flat equivalences across front-ends" begin
    using Distributions

    # A flat NamedTuple of leaves and the equivalent row table agree.
    flat_parallel = compose((a = Gamma(2.0, 1.0), b = LogNormal(1.0, 0.5)))
    flat_table = [(name = :a, dist = Gamma(2.0, 1.0)),
        (name = :b, dist = LogNormal(1.0, 0.5))]
    @test flat_parallel == compose(flat_table)

    # A single-chain row table is one sequential branch.
    chain = compose([(name = :s1, dist = Gamma(2.0, 1.0), chain = 1),
        (name = :s2, dist = LogNormal(0.5, 0.4), chain = 1),
        (name = :s3, dist = Gamma(1.0, 1.0), chain = 1)])
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

    @test_throws ArgumentError compose(42)
    # A NamedTuple sequential child must hold distributions.
    @test_throws ArgumentError compose((a = [1.0, 2.0],))
    # An explicit table needs `name` and `dist` columns.
    @test_throws ArgumentError compose([(name = :a, extra = 1)])
    @test_throws ArgumentError compose([(name = :a, dist = 1.0)])
    # Negative chain group ids would collide with auto-generated leaf ids.
    @test_throws ArgumentError compose([
        (name = :a, dist = Gamma(2.0, 1.0), chain = -1),
        (name = :b, dist = LogNormal(0.5, 0.4), chain = -1)])
end

@testitem "compose: a name/dist NamedTuple is always structural" begin
    using Distributions
    const CD = CensoredDistributions

    # A NamedTuple is ALWAYS read structurally, even one whose keys are
    # `:name`/`:dist`: there is no column-table auto-detection. Here the two
    # fields are user-named chain branches (a structural NamedTuple), NOT a
    # (name, dist) column table; the keys `:name`/`:dist` are just branch names.
    nt = (name = [Gamma(2.0, 1.0), LogNormal(0.5, 0.4)],
        dist = [Gamma(1.0, 1.0), Gamma(3.0, 1.0)])
    target = Parallel(Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4)),
        Sequential(Gamma(1.0, 1.0), Gamma(3.0, 1.0)))
    @test compose(nt) == target
    @test CD.component_names(compose(nt)) == (:name, :dist)
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

    # A Choose with equal-width alternatives IS a valid compose child (it occupies
    # a fixed flat slot); the flat path commits to its first alternative. The
    # supported direction also includes a composer INSIDE a Choose.
    sel = choose(:s1 => Gamma(2.0, 1.0), :s2 => Gamma(5.0, 1.0))
    withsel = compose((k = sel, m = Normal(0.0, 1.0)))
    @test withsel isa CD.Parallel
    @test logpdf(withsel, [1.0, 0.5]) ≈
          logpdf(Gamma(2.0, 1.0), 1.0) + logpdf(Normal(0.0, 1.0), 0.5)

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
    c = Resolve(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.7))

    @test occursin("Sequential", sprint(show, s))
    @test occursin("Sequential (2 steps)", sprint(show, MIME"text/plain"(), s))
    @test occursin("Parallel", sprint(show, p))
    @test occursin(
        "Parallel (2 branches)", sprint(show, MIME"text/plain"(), p))
    @test occursin("Resolve", sprint(show, c))
    @test occursin("death", sprint(show, MIME"text/plain"(), c))
end

@testitem "Composers show as a recursive indented tree" begin
    using Distributions

    # A nested stack must print the WHOLE structure as one indented tree, with
    # the child node types/names at the right depth (not just the top level).
    d = Parallel(
        Gamma(2.0, 1.0),
        Sequential(LogNormal(0.5, 0.4), Gamma(3.0, 1.0)),
        Resolve(:death => (Gamma(1.5, 1.0), 0.3),
            :disch => (Gamma(2.0, 1.5), 0.7)))
    out = sprint(show, MIME"text/plain"(), d)
    lines = split(out, '\n')

    # Root node header and its three branches.
    @test occursin("Parallel (3 branches)", lines[1])
    # The nested Sequential and Resolve children are rendered (recursively),
    # each indented one level under the root with a tree connector.
    seq_line = findfirst(l -> occursin("Sequential (2 steps)", l), lines)
    comp_line = findfirst(l -> occursin("Resolve (2 outcomes)", l), lines)
    @test seq_line !== nothing
    @test comp_line !== nothing
    @test occursin("├─ ", lines[seq_line])
    @test occursin("└─ ", lines[comp_line])

    # The Sequential's own leaf steps are indented one further level (the
    # continuation prefix `│  ` carries the parent connector down).
    @test any(l -> occursin("│  ", l) && occursin("LogNormal", l), lines)
    @test any(l -> occursin("│  ", l) && occursin("Gamma", l), lines)

    # The Resolve outcomes carry their names and branch probabilities, nested
    # under the last branch (so a spaces continuation prefix, not `│`).
    @test any(l -> occursin("death (p = 0.3)", l), lines)
    @test any(l -> occursin("disch (p = 0.7)", l), lines)

    # A composer nested inside Resolve also recurses: a Resolve outcome may
    # itself be a (univariate) Resolve, whose subtree then prints nested.
    # (Resolve outcomes are univariate by design, so a Sequential/Parallel
    # cannot be an outcome; a nested Resolve can.)
    nested_comp = Sequential(
        Gamma(2.0, 1.0),
        Resolve(
            :a => (Resolve(:a1 => (Normal(0.0, 1.0), 0.5),
                    :a2 => (Normal(1.0, 1.0), 0.5)),
                0.4),
            :b => (Normal(2.0, 1.0), 0.6)))
    out2 = sprint(show, MIME"text/plain"(), nested_comp)
    @test occursin("Sequential (2 steps)", out2)
    # Both the outer and the nested Resolve headers appear.
    @test count("Resolve (2 outcomes)", out2) == 2
    # The outer Resolve outcome `a` is itself a Resolve, nested under it.
    @test occursin("a (p = 0.4): Resolve (2 outcomes)", out2)
    @test occursin("a1 (p = 0.5)", out2)
end

@testitem "Composer show is compact; inspect gives detail" begin
    using Distributions
    const CD = CensoredDistributions

    # A censored leaf holds a Gauss-Legendre quadrature solver. The compact
    # show of a composed tree (and of a bare leaf) must NOT dump the solver's
    # node/weight arrays, only the structure and short leaf labels.
    leaf = double_interval_censored(Gamma(1.8, 1.4); interval = 1.0)
    leaf_t = double_interval_censored(
        Gamma(1.8, 1.4); interval = 1.0, upper = 10.0)
    stack = compose((onset_admit = leaf, admit_death = leaf_t))

    # A node from one of the leaf solver's 64-point quadrature arrays. Its
    # absence is the marker that no array was dumped.
    quad = "0.9993050"

    out3 = sprint(show, MIME"text/plain"(), stack)
    out2 = sprint(show, stack)
    @test !occursin(quad, out3)
    @test !occursin(quad, out2)
    # The tree show is a handful of lines (header + one per branch), not the
    # hundreds the array dump produced.
    @test count(==('\n'), out3) <= 6
    # Each leaf is summarised on one short line with its family and key params.
    @test occursin("Parallel (2 branches)", out3)
    @test occursin("IntervalCensored", out3)
    @test occursin("PrimaryCensored", out3)
    @test occursin("Gamma", out3)
    @test occursin("interval=1.0", out3)

    # A bare leaf's own show (2-arg and text/plain) is compact too.
    @test !occursin(quad, sprint(show, leaf))
    @test !occursin(quad, sprint(show, MIME"text/plain"(), leaf))
    # A truncated (upper-bounded) leaf stays compact, including its bound.
    @test !occursin(quad, sprint(show, leaf_t))
    @test occursin("Truncated", sprint(show, leaf_t))
    @test occursin("upper=10.0", sprint(show, leaf_t))

    # The solver itself shows as its type and node COUNT, never the arrays.
    @test sprint(show, CD.AnalyticalSolver()) ==
          "AnalyticalSolver(GaussLegendre(64))"
    @test sprint(show, CD.GaussLegendre(; n = 64)) == "GaussLegendre(64)"

    # `inspect` is the opt-in detailed render: it still avoids the arrays but
    # expands each leaf, naming the inner delay, primary event and solver.
    det = sprint(inspect, stack)
    @test !occursin(quad, det)
    @test occursin("primary_event", det)
    @test occursin("method", det)
    @test occursin("GaussLegendre(64)", det)
    @test occursin("Gamma", det)
    # Detail is longer than the compact tree, line for line.
    @test count(==('\n'), det) > count(==('\n'), out3)
end

@testitem "compose threads names through every input format" begin
    using Distributions
    const CD = CensoredDistributions

    nt = (onset_admit = LogNormal(1.5, 0.4), admit_death = Gamma(2.0, 1.0))
    tbl = [(name = :onset_admit, dist = LogNormal(1.5, 0.4)),
        (name = :admit_death, dist = Gamma(2.0, 1.0))]

    c_nt = compose(nt)
    c_tbl = compose(tbl)

    # User names pass through whichever format carries them.
    @test CD.component_names(c_nt) == (:onset_admit, :admit_death)
    @test CD.component_names(c_tbl) == (:onset_admit, :admit_death)

    # Structurally equal regardless of names (relaxed equivalence).
    @test c_nt == c_tbl

    # Chained table: branch named by its first row; steps by their own rows.
    tc = [(name = :incub, dist = Gamma(2.0, 1.0), chain = 1),
        (name = :rep, dist = LogNormal(0.5, 0.4), chain = 1),
        (name = :solo, dist = Normal(1.0, 0.5), chain = 0)]
    cc = compose(tc)
    @test CD.component_names(cc) == (:incub, :solo)
    @test CD.component_names(cc.components[1]) == (:incub, :rep)
end

@testitem "params is nested and name-keyed for composed dists" begin
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

    # Resolve contributes name-keyed outcomes plus branch_probs.
    c = Resolve(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.7))
    pc = CensoredDistributions._one_of_params(c)
    @test pc.death == (1.5, 1.0)
    @test pc.disch == (2.0, 1.5)
    @test pc.branch_probs == (0.3, 0.7)
end

@testitem "params_table flattens to edge|param|value|support" begin
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

    # Resolve branch probabilities appear as [0, 1]-supported rows.
    ctree = Parallel(
        (Gamma(2.0, 1.0),
            Resolve(:death => (Gamma(1.5, 1.0), 0.3),
                :disch => (Gamma(2.0, 1.5), 0.7))),
        (:incub, :resolution))
    ct = params_table(ctree)
    bp = findall(==(Symbol("resolution.branch_probs")), ct.edge)
    @test length(bp) == 2
    @test all(i -> ct.support[i] == (0.0, 1.0), bp)
    @test ct.value[bp] == [0.3, 0.7]
end

@testitem "name introspection: event_names, event_tree and event" begin
    using Distributions

    tree = compose((onset_admit = LogNormal(1.5, 0.4),
        admit_death = Gamma(2.0, 1.0)))
    # event_names is the per-RECORD key space: exactly the keys of `rand(d)`.
    # This PLAIN (uncensored) stack realises one value per edge with no latent
    # origin, so the keys are the branch/edge names (NOT a split flat path).
    @test event_names(tree) == (:onset_admit, :admit_death)
    @test event_names(tree) == keys(rand(tree))
    # event_tree keys are the top-level child names.
    @test keys(event_tree(tree)) == (:onset_admit, :admit_death)
    @test event(tree, :admit_death) == Gamma(2.0, 1.0)
    @test event(tree, :onset_admit) == LogNormal(1.5, 0.4)
    @test_throws KeyError event(tree, :missing_edge)

    # A nested path round-trips by multiple names and by a dotted Symbol.
    nested = compose((
        admit_path = compose((onset_admit = Gamma(2.0, 1.0),
            admit_death = LogNormal(0.5, 0.4))),
        onset_recover = Gamma(3.0, 1.0)))
    @test event(nested, :admit_path, :admit_death) == LogNormal(0.5, 0.4)
    @test event(nested, Symbol("admit_path.admit_death")) == LogNormal(0.5, 0.4)

    # Resolve: outcome names via event_tree, delays by name via event.
    c = Resolve(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.7))
    @test keys(event_tree(c)) == (:death, :disch)
    @test event(c, :death) == Gamma(1.5, 1.0)
end

@testitem "event_names equals the actual rand/logpdf record keys (#712)" begin
    using Distributions, Random

    # A PLAIN named-branch Parallel keys its record on the BRANCH names, NOT a
    # positional `:event_i` flat path. `event_names` must match `keys(rand(d))`
    # (the bug: it returned `(:event_1, :event_2, :event_3)`).
    par = compose((a = LogNormal(1.5, 0.5), b = Gamma(2.0, 1.0)))
    @test event_names(par) == (:a, :b)
    @test event_names(par) == keys(rand(Xoshiro(1), par))
    @test event_names(par) == keys(rand(Xoshiro(1), latent(par)))

    # A Sequential with UNDERSCORE edge names: a plain stack keys on the edge
    # names; the underscore split is only the latent/censored flat path.
    seq = compose((onset_admit = LogNormal(1.5, 0.5),
        admit_death = Gamma(2.0, 1.0)))
    @test event_names(seq) == (:onset_admit, :admit_death)
    @test event_names(seq) == keys(rand(Xoshiro(1), seq))

    # The SAME stack censored realises the flat event path, and `event_names`
    # follows that schema, still matching `keys(rand(d))`.
    cseq = compose((
        onset_admit = double_interval_censored(
            LogNormal(1.5, 0.5); primary_event = Uniform(0, 1), interval = 1.0),
        admit_death = double_interval_censored(
            Gamma(2.0, 1.0); primary_event = Uniform(0, 1), interval = 1.0)))
    @test event_names(cseq) == (:onset, :admit, :death)
    @test event_names(cseq) == keys(rand(Xoshiro(1), cseq))

    # A standalone Choose: its record keys are the selector form `(:kind,
    # :value)` from `rand`, but its `event_names` is the alternative names (the
    # active alternative is data-selected). Documented as the alternative names.
    ch = choose((a = Gamma(2.0, 1.0), b = LogNormal(1.0, 0.5)))
    @test event_names(ch) == (:a, :b)

    # A standalone Resolve realises the flat event record directly, so
    # `event_names == keys(rand(c))`.
    res = resolve(:death => (Gamma(2.0, 1.0), 0.3),
        :disch => (LogNormal(1.0, 0.5), 0.7))
    @test event_names(res) == keys(rand(Xoshiro(1), res))
end

@testitem "params_table is transparent to a censored leaf" begin
    using Distributions

    delay = Gamma(2.0, 1.5)
    cens = double_interval_censored(delay; primary_event = Uniform(0, 1),
        upper = 10.0, interval = 1.0)
    tree = compose((obs = cens,))
    tbl = params_table(tree)

    # Only the inner delay's free params appear; the censoring bounds
    # (interval, truncation, primary event) never leak as parameters.
    @test tbl.param == [:shape, :scale]
    @test tbl.value == [2.0, 1.5]
    @test all(s -> s == (minimum(delay), maximum(delay)), tbl.support)

    # The public `params` of the censored leaf is unchanged (still leaks the
    # bounds); only the introspection layer is transparent.
    @test length(params(cens)) > 2
end

@testitem "params_table surfaces a thin weight as a [0, 1] row (#642b)" begin
    using Distributions

    # A `thin(delay, p)` leaf surfaces the inner delay params PLUS a `:thin` row
    # for the reporting probability `p`, supported on [0, 1].
    tree = compose((inc = Gamma(2.0, 1.0), cases = thin(LogNormal(1.5, 0.4), 0.3)))
    tbl = params_table(tree)

    @test tbl.edge == [:inc, :inc, :cases, :cases, :cases]
    @test tbl.param == [:shape, :scale, :mu, :sigma, :thin]

    # The inner delay params keep the delay's support; the thin row is [0, 1].
    rows = collect(zip(tbl.edge, tbl.param, tbl.value, tbl.support))
    thin_row = only(filter(r -> r[2] === :thin, rows))
    @test thin_row[1] === :cases
    @test thin_row[3] == 0.3
    @test thin_row[4] == (0.0, 1.0)
    # The inner LogNormal params keep the positive half-line support.
    inner_rows = filter(r -> r[1] === :cases && r[2] !== :thin, rows)
    @test all(r -> r[4] == (0.0, Inf), inner_rows)

    # A thinned, CENSORED leaf still surfaces the inner free params + thin.
    ctree = compose((cases = thin(
        double_interval_censored(Gamma(2.0, 1.5); primary_event = Uniform(0, 1),
            interval = 1.0), 0.4),))
    ctbl = params_table(ctree)
    @test ctbl.param == [:shape, :scale, :thin]
    @test ctbl.value == [2.0, 1.5, 0.4]
    @test ctbl.support[end] == (0.0, 1.0)

    # `cumulative` carries no numeric weight, so it surfaces NO extra row.
    cum = compose((inc = cumulative(Gamma(2.0, 1.0)),))
    @test params_table(cum).param == [:shape, :scale]
end

@testitem "build_priors gives a thin weight a [0, 1] default (#642b)" begin
    using Distributions

    tree = compose((cases = thin(LogNormal(1.5, 0.4), 0.3),))
    tbl = params_table(tree)

    # The thin row's default prior is the [0, 1]-supported `Uniform(0, 1)`.
    nested = build_priors(tbl)
    @test nested.cases.thin == Uniform(0, 1)
    @test minimum(nested.cases.thin) == 0
    @test maximum(nested.cases.thin) == 1

    # `default_prior` classifies a `:thin` row by its [0, 1] support.
    @test default_prior((; edge = :cases, param = :thin, value = 0.3,
        support = (0.0, 1.0))) == Uniform(0, 1)

    # A user can override the thin prior like any other parameter.
    over = build_priors(tbl; priors = (cases = (thin = Beta(2, 5),),))
    @test over.cases.thin == Beta(2, 5)
end

@testitem "update round-trips a thin weight (#642b)" begin
    using Distributions

    tree = compose((cases = thin(LogNormal(1.5, 0.4), 0.3),))

    # `update` re-routes a new thin weight into the `ThinOp`, keeping the inner
    # delay params, and round-trips the value.
    updated = update(tree, (cases = (mu = 1.0, sigma = 0.5, thin = 0.6),))
    leaf = event(updated, :cases)
    @test leaf isa CensoredDistributions.Transformed
    @test leaf.op isa CensoredDistributions.ThinOp
    @test leaf.op.factor == 0.6
    @test CensoredDistributions.get_dist(leaf) == LogNormal(1.0, 0.5)

    # The new thin weight surfaces back through params_table (round-trip).
    rt = params_table(updated)
    @test rt.value[findfirst(==(:thin), rt.param)] == 0.6

    # A censored thinned leaf round-trips inner params + thin + censoring.
    ctree = compose((cases = thin(
        double_interval_censored(Gamma(2.0, 1.5); primary_event = Uniform(0, 1),
            interval = 1.0), 0.4),))
    cup = update(ctree, (cases = (shape = 3.0, scale = 1.0, thin = 0.7),))
    cleaf = event(cup, :cases)
    @test cleaf.op.factor == 0.7
    @test CensoredDistributions.free_leaf(cleaf) == Gamma(3.0, 1.0)
    @test cleaf.dist isa CensoredDistributions.IntervalCensored
end

@testitem "update rebuilds a composed distribution from a nested NamedTuple" begin
    using Distributions

    tpl = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    upd = update(tpl, (onset_admit = (shape = 3.0, scale = 1.5),
        admit_death = (mu = 0.7, sigma = 0.5)))

    @test event_names(upd) == event_names(tpl)
    @test event(upd, :onset_admit) == Gamma(3.0, 1.5)
    @test event(upd, :admit_death) == LogNormal(0.7, 0.5)

    # Equivalent to a hand-rebuilt distribution.
    hand = compose((onset_admit = Gamma(3.0, 1.5),
        admit_death = LogNormal(0.7, 0.5)))
    @test logpdf(upd, [1.5, 2.5]) == logpdf(hand, [1.5, 2.5])

    # A censored leaf round-trips: only the inner free params are supplied and
    # the fixed censoring is carried through.
    cens = double_interval_censored(Gamma(2.0, 1.5); upper = 10.0, interval = 1.0)
    ct = compose((obs = cens,))
    cu = update(ct, (obs = (shape = 2.0, scale = 1.5),))
    @test typeof(event(cu, :obs)) == typeof(cens)
    @test logpdf(event(cu, :obs), 3.0) == logpdf(cens, 3.0)

    # Missing / extra keys error.
    @test_throws ArgumentError update(tpl,
        (onset_admit = (shape = 1.0,),
            admit_death = (mu = 0.1, sigma = 0.2)))
    @test_throws ArgumentError update(tpl,
        (onset_admit = (shape = 1.0, scale = 1.0, bogus = 0.0),
            admit_death = (mu = 0.1, sigma = 0.2)))
end

@testitem "update handles Resolve branch_probs" begin
    using Distributions

    tree = compose((res = Resolve(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.7)),))

    # branch_probs omitted: the template's fixed probabilities are kept.
    keep = update(tree, (res = (death = (shape = 1.0, scale = 1.0),
        disch = (shape = 2.0, scale = 2.0)),))
    @test event(keep, :res).branch_probs == (0.3, 0.7)

    # branch_probs supplied: each outcome's probability is replaced.
    set = update(tree,
        (res = (death = (shape = 1.0, scale = 1.0),
            disch = (shape = 2.0, scale = 2.0),
            branch_probs = (death = 0.4, disch = 0.6)),))
    @test event(set, :res).branch_probs == (0.4, 0.6)
end

@testitem "build_priors assembles nested priors from the params table" begin
    using Distributions

    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    tbl = params_table(tree)

    # Default-prior function over the table rows.
    nested = build_priors(tbl;
        default = row -> truncated(Normal(row.value, 1); lower = -10))
    @test Set(keys(nested)) == Set((:onset_admit, :admit_death))
    @test Set(keys(nested.onset_admit)) == Set((:shape, :scale))
    @test Set(keys(nested.admit_death)) == Set((:mu, :sigma))
    @test nested.onset_admit.shape isa Distribution

    # Explicit per-(edge, param) priors take precedence over the default.
    mixed = build_priors(tbl;
        priors = Dict((:onset_admit, :shape) => Normal(2, 0.5)),
        default = row -> truncated(Normal(row.value, 1); lower = -10))
    @test mixed.onset_admit.shape == Normal(2, 0.5)

    # With no default and an uncovered row, building errors.
    @test_throws ArgumentError build_priors(tbl; default = nothing)
end

@testitem "update overrides priors-table fields by path" begin
    using Distributions

    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    priors = build_priors(params_table(tree))

    shape_prior = truncated(Normal(1.0, 1.5); lower = 0.05)
    sigma_prior = truncated(Normal(0.0, 0.5); lower = 0)

    # A single Symbol addresses a top-level name; a tuple a nested path. The
    # RHS is a NamedTuple of per-field overrides.
    updated = update(priors,
        :onset_admit => (shape = shape_prior,),
        :admit_death => (sigma = sigma_prior,))

    # The named fields are replaced exactly.
    @test updated.onset_admit.shape === shape_prior
    @test updated.admit_death.sigma === sigma_prior
    # Sibling fields keep their support-derived defaults.
    @test updated.onset_admit.scale === priors.onset_admit.scale
    @test updated.admit_death.mu === priors.admit_death.mu
    # The input is not mutated.
    @test priors.onset_admit.shape !== shape_prior
end

@testitem "update on a priors table walks a nested path" begin
    using Distributions

    inner = compose((death = Gamma(1.5, 1.0), discharge = Gamma(2.0, 1.5)))
    tree = compose((admit_resolution = inner, onset_notif = Gamma(2.0, 1.0)))
    priors = build_priors(params_table(tree))

    shape_prior = truncated(Normal(1.0, 1.5); lower = 0.05)
    updated = update(priors,
        (:admit_resolution, :death) => (shape = shape_prior,),
        :onset_notif => (shape = shape_prior,))

    @test updated.admit_resolution.death.shape === shape_prior
    @test updated.onset_notif.shape === shape_prior
    # Untouched leaf and field unchanged.
    @test updated.admit_resolution.discharge.shape ===
          priors.admit_resolution.discharge.shape
    @test updated.admit_resolution.death.scale ===
          priors.admit_resolution.death.scale
end

@testitem "update on a priors table errors on an unknown path" begin
    using Distributions

    tree = compose((onset_admit = Gamma(2.0, 1.0),))
    priors = build_priors(params_table(tree))
    prior = Normal(0.0, 1.0)

    # Unknown top-level name.
    @test_throws ArgumentError update(priors, :missing => (shape = prior,))
    # Unknown nested step.
    @test_throws ArgumentError update(priors,
        (:onset_admit, :nope) => (shape = prior,))
    # Unknown field at a known leaf.
    @test_throws ArgumentError update(priors,
        :onset_admit => (bogus = prior,))
end

@testitem "build_priors derives support-based defaults (brms-style)" begin
    using Distributions

    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    tbl = params_table(tree)

    # No priors/default supplied: every row gets a support-derived default.
    nested = build_priors(tbl)
    @test Set(keys(nested)) == Set((:onset_admit, :admit_death))
    # Positive-support scale/shape -> positive-truncated Normal.
    @test nested.onset_admit.shape isa Truncated
    @test minimum(nested.onset_admit.shape) == 0
    @test nested.onset_admit.scale isa Truncated
    # A location parameter (LogNormal mu) -> unconstrained Normal.
    @test nested.admit_death.mu isa Normal
    @test nested.admit_death.sigma isa Truncated

    # default_prior is the per-row default and classifies by the parameter.
    @test default_prior((; edge = :e, param = :p, value = 0.5,
        support = (0.0, 1.0))) == Uniform(0, 1)
    @test default_prior((; edge = :e, param = :scale, value = 2.0,
        support = (0.0, Inf))) isa Truncated
    @test default_prior((; edge = :e, param = :mu, value = -1.0,
        support = (0.0, Inf))) isa Normal
end

@testitem "default_prior classifies by parameter, not variate support" begin
    using Distributions

    # A location-family delay (Normal) has unbounded variate support, but its
    # scale parameter still lives on the positive half-line: its default prior
    # must be positively constrained, not an unconstrained Normal that puts mass
    # on negative scale.
    tree = compose((onset = Normal(1.0, 2.0),))
    tbl = params_table(tree)
    nested = build_priors(tbl)
    @test nested.onset.mu isa Normal
    @test minimum(nested.onset.sigma) >= 0
    @test nested.onset.sigma isa Truncated

    # Same when the leaf is wrapped in an Affine (still a location family with
    # unbounded support); the inner sigma must stay positively constrained.
    atree = compose((onset = affine(Normal(1.0, 2.0); scale = 2.0,
        shift = 0.5),))
    anested = build_priors(params_table(atree))
    @test anested.onset.mu isa Normal
    @test minimum(anested.onset.sigma) >= 0

    # Scale/shape parameters are positive regardless of the variate support.
    @test minimum(default_prior((; edge = :e, param = :sigma, value = 2.0,
        support = (-Inf, Inf)))) >= 0
    @test minimum(default_prior((; edge = :e, param = :scale, value = 2.0,
        support = (-Inf, Inf)))) >= 0
    @test minimum(default_prior((; edge = :e, param = :shape, value = 2.0,
        support = (-Inf, Inf)))) >= 0
    @test minimum(default_prior((; edge = :e, param = :rate, value = 2.0,
        support = (-Inf, Inf)))) >= 0

    # Location parameters stay unbounded.
    @test default_prior((; edge = :e, param = :mu, value = -1.0,
        support = (-Inf, Inf))) isa Normal
    @test default_prior((; edge = :e, param = :location, value = -1.0,
        support = (-Inf, Inf))) isa Normal

    # A [0, 1] probability parameter is Uniform(0, 1).
    @test default_prior((; edge = :e, param = :p, value = 0.3,
        support = (0.0, 1.0))) == Uniform(0, 1)
end

@testitem "build_priors takes a nested-NamedTuple partial override" begin
    using Distributions

    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    tbl = params_table(tree)

    # Override only one parameter; the rest keep their support-based defaults.
    nested = build_priors(tbl;
        priors = (onset_admit = (shape = Normal(9, 9),),))
    @test nested.onset_admit.shape == Normal(9, 9)
    @test nested.onset_admit.scale isa Truncated
    @test nested.admit_death.mu isa Normal
end

@testitem "build_priors front-door matches the table form" begin
    using Distributions

    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))

    # The front-door builds the params_table internally; same nested priors.
    @test build_priors(tree) == build_priors(params_table(tree))
    # param_priors is a thin alias over the same machinery.
    @test param_priors(tree) == build_priors(params_table(tree))

    # The keyword surface is forwarded unchanged.
    @test build_priors(tree;
        default = row -> truncated(Normal(row.value, 1); lower = -10)) ==
          build_priors(params_table(tree);
        default = row -> truncated(Normal(row.value, 1); lower = -10))
    @test build_priors(tree;
        priors = Dict((:onset_admit, :shape) => Normal(2, 0.5))) ==
          build_priors(params_table(tree);
        priors = Dict((:onset_admit, :shape) => Normal(2, 0.5)))

    # Overrides via update still work off the front-door result.
    shape_prior = truncated(Normal(2, 0.5); lower = 0)
    updated = update(build_priors(tree),
        (:onset_admit,) => (shape = shape_prior,))
    @test updated.onset_admit.shape === shape_prior
    @test updated.admit_death.mu === build_priors(tree).admit_death.mu

    # An unknown override path still errors clearly.
    @test_throws ArgumentError update(build_priors(tree),
        :nope => (shape = shape_prior,))
end

@testitem "build_priors front-door works on a bare leaf" begin
    using Distributions

    leaf = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))

    # A bare leaf keys the priors flat by parameter name (no edge wrapper).
    priors = build_priors(leaf)
    @test Set(keys(priors)) == Set((:mu, :sigma))
    @test priors == build_priors(params_table(leaf))
    @test param_priors(leaf) == priors
    @test priors.mu isa Normal
    @test priors.sigma isa Truncated
end

@testitem "event pulls a named subtree from a composed dist" begin
    using Distributions

    tree = compose((
        admit_path = compose((onset_admit = Gamma(2.0, 1.0),
            admit_death = LogNormal(0.5, 0.4))),
        onset_recover = Gamma(3.0, 1.0)))

    # A single name matches event.
    @test event(tree, :admit_path) === event(tree, :admit_path)
    @test event(tree, :onset_recover) == Gamma(3.0, 1.0)
    # A multi-name path descends to the leaf.
    @test event(tree, :admit_path, :admit_death) == LogNormal(0.5, 0.4)
    # A dotted-path Symbol (as in params_table's edge column) is equivalent.
    @test event(tree, Symbol("admit_path.onset_admit")) == Gamma(2.0, 1.0)
    # A bad name throws.
    @test_throws KeyError event(tree, :admit_path, :missing_edge)
    @test_throws ArgumentError event(tree)
end

@testitem "shared tags a leaf and is transparent to scoring" begin
    using Distributions

    base = Gamma(2.0, 1.0)
    s = shared(:inc, base)
    @test s isa Shared
    @test CensoredDistributions._shared_tag(s) == :inc
    # Every distribution method delegates to the wrapped leaf.
    for x in (0.5, 1.5, 3.0)
        @test logpdf(s, x) == logpdf(base, x)
        @test cdf(s, x) == cdf(base, x)
    end
    @test params(s) == params(base)
    @test minimum(s) == minimum(base) && maximum(s) == maximum(base)

    # The tag survives censoring wrappers.
    cs = shared(:inc, double_interval_censored(base; upper = 10.0, interval = 1.0))
    @test CensoredDistributions._shared_tag(cs) == :inc
    @test CensoredDistributions.free_leaf(cs) == base
end

@testitem "params_table dedups a shared tag across branches" begin
    using Distributions

    inc = shared(:inc, Gamma(2.0, 1.0))
    delta = LogNormal(0.5, 0.4)
    # `inc` appears in BOTH the index and sourced branches of a choose.
    tree = choose(:index => inc,
        :sourced => compose((delta = delta, inc = inc)))
    tbl = params_table(tree)

    # `inc` is inventoried ONCE (under its tag), `delta` once: no duplicate inc.
    @test tbl.edge == [:inc, :inc, Symbol("sourced.delta"), Symbol("sourced.delta")]
    @test tbl.param == [:shape, :scale, :mu, :sigma]
    @test count(==(:inc), tbl.edge) == 2  # shape + scale of the one inc group
    @test tbl.value == [2.0, 1.0, 0.5, 0.4]
end

@testitem "update places a shared value in every occurrence" begin
    using Distributions

    inc = shared(:inc, Gamma(2.0, 1.0))
    delta = LogNormal(0.5, 0.4)
    tree = choose(:index => inc,
        :sourced => compose((delta = delta, inc = inc)))

    # One top-level `inc` entry updates both occurrences.
    upd = update(tree, (inc = (shape = 3.0, scale = 1.5),
        sourced = (delta = (mu = 0.7, sigma = 0.5),)))
    idx = CensoredDistributions._pick(upd, :index)
    src = CensoredDistributions._pick(upd, :sourced)
    @test get_dist(idx) == Gamma(3.0, 1.5)
    @test get_dist(event(src, :inc)) == Gamma(3.0, 1.5)
    @test event(src, :delta) == LogNormal(0.7, 0.5)

    # A missing shared entry errors clearly.
    @test_throws ArgumentError update(tree,
        (sourced = (delta = (mu = 0.7, sigma = 0.5),),))
end

@testitem "shared leaf is transparent to rand on a composed tree" begin
    using Distributions, Random

    inc = shared(:inc, Gamma(2.0, 1.0))
    # A composed tree carrying a shared leaf draws a full event path; the tag does
    # not change the realisation type (eltype delegates to the wrapped leaf).
    tree = compose((delta = LogNormal(0.5, 0.4), inc = inc))
    @test eltype(tree) == Float64
    @test length(rand(MersenneTwister(3), tree)) == 2

    # A shared leaf's own draw delegates to the wrapped distribution.
    a = rand(MersenneTwister(5), inc)
    b = rand(MersenneTwister(5), Gamma(2.0, 1.0))
    @test a == b
end
