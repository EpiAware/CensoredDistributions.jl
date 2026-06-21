# Every composer verb accepts BOTH a `:name => dist` Pairs spelling and a
# positional `(name = dist, …)` NamedTuple, and `compose` accepts both a
# NamedTuple and the equivalent Pairs. The two spellings build an `==` object
# with the same names and children; the NamedTuple is positional, so a config
# keyword (`choose`'s `selector`) stays separate.

@testitem "sequential: Pairs and NamedTuple build the same chain" begin
    using CensoredDistributions, Distributions

    d1 = Gamma(2.0, 1.0)
    d2 = LogNormal(0.5, 0.4)
    pairs = sequential(:onset_admit => d1, :admit_death => d2)
    nt = sequential((onset_admit = d1, admit_death = d2))
    @test pairs == nt
    @test typeof(pairs) == typeof(nt)
    @test CensoredDistributions.component_names(nt) ==
          (:onset_admit, :admit_death)
    @test CensoredDistributions.component_names(pairs) ==
          CensoredDistributions.component_names(nt)
    # show round-trips without error.
    @test repr(nt) isa String
end

@testitem "parallel: Pairs and NamedTuple build the same branch set" begin
    using CensoredDistributions, Distributions

    d1 = Gamma(2.0, 1.0)
    d2 = LogNormal(1.0, 0.5)
    pairs = parallel(:admit => d1, :notif => d2)
    nt = parallel((admit = d1, notif = d2))
    @test pairs == nt
    @test typeof(pairs) == typeof(nt)
    @test CensoredDistributions.component_names(nt) == (:admit, :notif)
    @test CensoredDistributions.component_names(pairs) ==
          CensoredDistributions.component_names(nt)
    @test repr(nt) isa String
end

@testitem "resolve: Pairs and NamedTuple build the same mixture node" begin
    using CensoredDistributions, Distributions

    cfr = 0.3
    pairs = resolve(:death => (Gamma(1.5, 1.0), cfr),
        :disch => (Gamma(2.0, 1.5), 1 - cfr))
    nt = resolve((death = (Gamma(1.5, 1.0), cfr),
        disch = (Gamma(2.0, 1.5), 1 - cfr)))
    @test pairs == nt
    @test typeof(pairs) == typeof(nt)
    @test mean(pairs) ≈ mean(nt)
    @test repr(nt) isa String
end

@testitem "resolve: the omitted-last-prob residual via Pairs" begin
    using CensoredDistributions, Distributions

    cfr = 0.3
    # The discharge probability is the residual `1 - cfr`, omitted on the
    # last outcome. Pairs and NamedTuple both fill it.
    pairs = resolve(:death => (Gamma(1.5, 1.0), cfr),
        :disch => Gamma(2.0, 1.5))
    nt = resolve((death = (Gamma(1.5, 1.0), cfr), disch = Gamma(2.0, 1.5)))
    @test pairs == nt
    @test mean(pairs) ≈ mean(nt)
end

@testitem "compete: Pairs and NamedTuple build the same racing node" begin
    using CensoredDistributions, Distributions

    pairs = compete(:death => Gamma(2.0, 3.0), :recover => Gamma(3.0, 2.0))
    nt = compete((death = Gamma(2.0, 3.0), recover = Gamma(3.0, 2.0)))
    @test pairs == nt
    @test typeof(pairs) == typeof(nt)
    @test values(probs(pairs)) == values(probs(nt))
    @test repr(nt) isa String
end

@testitem "choose: NamedTuple alternatives stay separate from selector" begin
    using CensoredDistributions, Distributions

    d1 = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))
    d2 = primary_censored(Gamma(4.0, 1.5), Uniform(0, 1))
    pairs = choose(:index => d1, :sourced => d2)
    nt = choose((index = d1, sourced = d2))
    @test pairs == nt
    @test typeof(pairs) == typeof(nt)
    # The `selector` keyword works alongside the positional NamedTuple.
    sel = choose((index = d1, sourced = d2); selector = :kind)
    @test sel == pairs
    sel2 = choose((index = d1, sourced = d2); selector = :origin)
    @test sel2 != pairs
    @test logpdf(nt, 3.0; kind = :index) == logpdf(pairs, 3.0; kind = :index)
    @test repr(nt) isa String
end

@testitem "compose: NamedTuple and Pairs build the same stack" begin
    using CensoredDistributions, Distributions

    d1 = Gamma(2.0, 1.0)
    d2 = LogNormal(0.5, 0.4)
    nt = compose((a = d1, b = d2))
    pairs = compose(:a => d1, :b => d2)
    @test nt == pairs
    @test typeof(nt) == typeof(pairs)
    @test CensoredDistributions.component_names(pairs) == (:a, :b)
    # A computed/data-driven name only the Pairs spelling can express.
    names = [Symbol("branch_$i") for i in 1:2]
    dyn = compose((names[1] => d1, names[2] => d2)...)
    @test dyn == nt
    @test CensoredDistributions.component_names(dyn) ==
          (:branch_1, :branch_2)
    # A non-Symbol branch name is rejected.
    @test_throws ArgumentError compose("a" => d1, "b" => d2)
end

@testitem "compose Pairs nest vectors and NamedTuples as children" begin
    using CensoredDistributions, Distributions

    oa = Gamma(2.0, 1.0)
    ad = LogNormal(0.5, 0.4)
    # A vector child nests as a Sequential, matching the NamedTuple form.
    nt = compose((chain = [oa, ad], leaf = ad))
    pairs = compose(:chain => [oa, ad], :leaf => ad)
    @test nt == pairs
end
