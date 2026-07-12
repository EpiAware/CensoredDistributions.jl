# Shared setup for the `as_turing` AD gradient test items. The genuinely
# NEW differentiated surface `as_turing`'s model body exercises (beyond
# DynamicPPL's own well-trodden tilde/`@addlogprob!` machinery) is
# `ComposedDistributions.logdensity`'s reconstruction: `unflatten -> update
# -> loglik`, over a tree that mixes CD leaves with plain `Distributions`
# leaves. This differentiates that function directly (PPL-neutral, no
# Turing/DynamicPPL needed) across the same backend matrix as the rest of
# the AD suite, reusing `ADFixtures.backends()` for the exact configs.

@testsnippet ComposedTuringADHelpers begin
    using ADTypes
    using DifferentiationInterface
    using DifferentiationInterface: Constant
    using ADFixtures
    using ComposedDistributions
    using ComposedDistributions: as_logdensity, logdensity
    using Distributions
    using ForwardDiff, ReverseDiff, Enzyme, Mooncake

    _composed_entry(name) = only(filter(e -> e.name == name, ADFixtures.backends()))

    # Built once per test process the tree is fixed; `prob` travels as a
    # `Constant` DI context rather than a closure capture, matching
    # `ADFixtures.scenarios`' convention (keeps every backend, including
    # Enzyme, needing no `function_annotation = Duplicated`).
    function _composed_logdensity_scenario()
        tree = compose((
            onset_admit = uncertain(Gamma(2.0, 1.0);
                shape = LogNormal(log(2.0), 0.2),
                scale = LogNormal(0.0, 0.2)),
            admit_death = uncertain(LogNormal(0.5, 0.4);
                mu = Normal(0.5, 0.5),
                sigma = LogNormal(log(0.4), 0.2))))
        data = [[0.5, 2.0], [1.0, 3.0], [0.8, 2.5]]
        prob = as_logdensity(tree, data)
        x0 = [2.0, 1.0, 0.5, 0.4]
        f(x, p) = logdensity(p, x)
        ref = ForwardDiff.gradient(x -> f(x, prob), x0)
        return f, x0, prob, ref
    end

    # Backends confirmed broken on this scenario (both Mooncake directions):
    # `ComposedDistributions.logdensity`/`unflatten` each guard their flat
    # vector's length with `throw(DimensionMismatch("... $(prob.dist) ..."))`
    # -- an UNREACHED branch here (the lengths always match), but Mooncake's
    # whole-program rule derivation still needs a differentiation rule for it,
    # including the string interpolation, which calls `show`/`string` on the
    # composed tree. That recurses into `Base`'s UTF-8 string-indexing
    # continuation machinery (`_thisind_continued`/`getindex_continued`/
    # `string_index_err`), where a `sub_ptr` pointer-arithmetic intrinsic has
    # no Mooncake rule (reverse: "sub_ptr intrinsic hit"; forward: no
    # `frule!!` method) -- a Mooncake/upstream limitation on an error-message
    # branch, not a correctness issue in this package. Every other backend
    # (ForwardDiff, ReverseDiff, Enzyme fwd+rev) differentiates this scenario
    # cleanly. Tracked upstream; repointing here once fixed just means
    # dropping the name from this set.
    const COMPOSED_LOGDENSITY_BROKEN_BACKENDS = Set([
        "Mooncake reverse", "Mooncake forward"])

    function test_composed_logdensity_backend(name)
        backend = _composed_entry(name).backend
        f, x0, prob, ref = _composed_logdensity_scenario()
        if name in COMPOSED_LOGDENSITY_BROKEN_BACKENDS
            ok = try
                g = DifferentiationInterface.gradient(
                    f, backend, x0, Constant(prob))
                g isa AbstractVector && all(isfinite, g) &&
                    isapprox(g, ref; rtol = 5e-2, atol = 1e-6)
            catch
                false
            end
            @test_broken ok
            return nothing
        end
        g = DifferentiationInterface.gradient(f, backend, x0, Constant(prob))
        @test g isa AbstractVector
        @test all(isfinite, g)
        @test isapprox(g, ref; rtol = 5e-2, atol = 1e-6)
    end
end
