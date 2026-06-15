# Shared-dist scoring (#395): a `shared(:tag, ...)` censored leaf appears in N
# positions of a nested tree. The issue proposed reusing the leaf's
# param-dependent precomputation across occurrences. In the current architecture
# the heavy work (the primary-censored CDF quadrature) is parameterised by the
# OBSERVED gap, so it differs per occurrence and cannot be reused; the wrapper
# objects (`PrimaryCensored`, `Convolved`) precompute nothing and the integration
# rule is a global constant. This suite measures the end-to-end logpdf of a tree
# with a shared censored leaf vs the SAME tree built from independent identical
# (untagged) leaves, so the two paths can be compared directly and any future
# reuse re-measured. They score IDENTICALLY (the shared wrapper is transparent),
# so this is a like-for-like cost comparison, not a correctness contrast.

SUITE["SharedReuse"] = BenchmarkGroup()

let
    # A nested (Select-bearing) tree whose origin edge AND routed alternative are
    # the SAME shared `:inc` censored leaf, so the leaf is scored in two positions
    # per record. Many records share the one tree (fixed params).
    inc_params = (2.0, 1.0)
    b = primary_censored(Gamma(5.0, 1.0), Uniform(0, 1))

    selecting = CensoredDistributions.selecting
    inc = CensoredDistributions.shared(
        :inc, primary_censored(Gamma(inc_params...), Uniform(0, 1)))
    tagged = Sequential((inc, selecting(:a => inc, :b => b)),
        (:onset_admit, :admit_death))

    inc1 = primary_censored(Gamma(inc_params...), Uniform(0, 1))
    inc2 = primary_censored(Gamma(inc_params...), Uniform(0, 1))
    untagged = Sequential((inc1, selecting(:a => inc2, :b => b)),
        (:onset_admit, :admit_death))

    # Latent-origin records (the marginalisation path the shared leaf feeds).
    n_records = 50
    ev = Vector{Union{Missing, Float64}}([missing, 2.0, 5.0])
    evs = [copy(ev) for _ in 1:n_records]

    # Shared-tagged tree: the shared leaf is rebuilt at each occurrence today.
    SUITE["SharedReuse"]["tagged"] = @benchmarkable begin
        acc = 0.0
        for e in $evs
            acc += logpdf($tagged, e)
        end
        acc
    end

    # Untagged identical-leaf tree: the same per-occurrence rebuild, no tag.
    SUITE["SharedReuse"]["untagged"] = @benchmarkable begin
        acc = 0.0
        for e in $evs
            acc += logpdf($untagged, e)
        end
        acc
    end
end
