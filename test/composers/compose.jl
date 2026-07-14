
@testitem "single-branch compose rand/mean over a nested censored child" begin
    using CensoredDistributions, Distributions, Random
    using Statistics: mean

    # A top-level NamedTuple is always a Parallel over its named branches, so a
    # one-entry NamedTuple is a Parallel-of-one. When that single branch is
    # itself a NESTED censored composer (a Parallel carrying a Resolve), the
    # whole tree shares one latent origin; `rand` must walk the tree, not fall
    # through to the plain per-leaf-value path.
    dic(x) = double_interval_censored(x; primary_event = Uniform(0.0, 1.0),
        interval = 1.0)
    nested_comp = Resolve(:death => (dic(Gamma(2.0, 3.5)), 0.4),
        :discharge => (dic(Gamma(1.0, 8.0)), 0.6))
    inner = compose((onset_admit = dic(Gamma(1.2, 3.0)),
        admit_resolution = nested_comp))

    single = compose((path = inner,))
    @test single isa CensoredDistributions.Parallel
    @test length(single.components) == 1

    Random.seed!(436)
    # `rand` must not error and must produce the nested named event record.
    rec = rand(single)
    @test rec isa NamedTuple
    @test keys(rec) == event_names(single)

    # `mean(latent(...))` over the per-event view must also not error and
    # returns the per-event NamedTuple keyed by event_names(single).
    m = mean(latent(single))
    @test m isa NamedTuple
    @test keys(m) == event_names(single)
    @test length(m) == length(event_names(single))
    @test all(isfinite, skipmissing(values(m)))
end

@testitem "single-branch compose stays a Parallel-of-one (no collapse)" begin
    using CensoredDistributions, Distributions

    # A top-level NamedTuple is documented to be a Parallel over its named
    # branches; a single-branch `compose((x = ...,))` is a Parallel-of-one and
    # does NOT collapse to the bare child. Collapsing would drop the branch name
    # the NamedTuple front-end exists to thread through, and break the
    # `compose(nt)::Parallel` invariant other front-ends match by `==`.
    chain = Sequential((Gamma(2.0, 1.0), LogNormal(0.5, 0.4)))
    single = compose((path = chain,))
    @test single isa CensoredDistributions.Parallel
    @test length(single.components) == 1
    @test single.components[1] === chain
    @test CensoredDistributions.component_names(single) == (:path,)

    # A bare leaf branch likewise stays a Parallel-of-one.
    leaf = compose((only = Gamma(2.0, 1.0),))
    @test leaf isa CensoredDistributions.Parallel
    @test CensoredDistributions.component_names(leaf) == (:only,)
end

@testitem "compose(origin; branches...) shares an origin across branches" begin
    using CensoredDistributions, Distributions

    incub = Gamma(2.5, 1.3)
    combined = compose(incub; cases = thin(Gamma(1.5, 1.2), 0.3),
        deaths = thin(Gamma(3.0, 4.0), 0.012))
    # A shared origin then a Parallel of the branch tails.
    @test combined isa CensoredDistributions.Sequential
    @test combined.components[2] isa CensoredDistributions.Parallel
    @test CensoredDistributions.component_names(combined.components[2]) ==
          (:cases, :deaths)

    # Equivalent to the explicit Sequential-ending-in-Parallel form.
    explicit = Sequential((incub,
        Parallel((thin(Gamma(1.5, 1.2), 0.3), thin(Gamma(3.0, 4.0), 0.012)),
            (:cases, :deaths))))
    @test combined == explicit

    # At least one branch is required.
    @test_throws ArgumentError compose(incub)
end

@testitem "compose(dist, n) repeats a distribution into a chain or branches" begin
    using CensoredDistributions, Distributions
    const CD = CensoredDistributions

    d = Gamma(2.0, 1.0)
    # The default repeats into a Sequential chain of n identical steps.
    chain = compose(d, 3)
    @test chain isa CD.Sequential
    @test length(chain.components) == 3
    @test all(==(d), chain.components)
    @test CD.component_names(chain) == (:step_1, :step_2, :step_3)
    # `chain = false` repeats into a Parallel of n identical branches, equal to
    # the hand-written NamedTuple of the same leaf repeated.
    branches = compose(d, 3; chain = false)
    @test branches isa CD.Parallel
    @test branches == compose((branch_1 = d, branch_2 = d, branch_3 = d))
    # A pre-built composer subtree repeats too (one shared child object).
    sub = compose((a = Gamma(1.0, 1.0), b = LogNormal(0.5, 0.4)))
    @test compose(sub, 2) isa CD.Sequential
    # n must be at least one.
    @test_throws ArgumentError compose(d, 0)
end
