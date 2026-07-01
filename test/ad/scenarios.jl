# Per-backend AD gradient scenario suite. Each backend is its own test
# item, tagged so the per-backend CI can select it with a tag
# filter (e.g. `julia test/ad/runtests.jl enzyme_reverse`). With no tag
# every item runs, as `task test-ad` does locally.
#
# The MARGINAL and LATENT scenario groups are SEPARATE test items per backend:
# `test_working_backend(name)` defaults to the marginal group, and
# `test_working_backend(name; category = :latent)` runs the latent /
# augmented-primary group (`Latent*`, `PrimaryConditional`). Keeping them apart
# means the marginal AD sweep is purely marginal (no latent scenarios riding
# along), and the latent path has its own coverage. The standalone vectorised
# latent items live in `latent_tree_ad.jl` / `latent_vectorised_ad.jl`.

# === Marginal scenario group ===

@testitem "ForwardDiff gradients (marginal)" tags=[:ad, :forwarddiff] setup=[ADHelpers] begin
    test_working_backend("ForwardDiff")
end

@testitem "ReverseDiff (tape) gradients (marginal)" tags=[:ad, :reversediff] setup=[ADHelpers] begin
    test_working_backend("ReverseDiff (tape)")
end

@testitem "Enzyme reverse gradients (marginal)" tags=[:ad, :enzyme, :enzyme_reverse] setup=[ADHelpers] begin
    test_working_backend("Enzyme reverse")
end

@testitem "Mooncake reverse gradients (marginal)" tags=[:ad, :mooncake, :mooncake_reverse] setup=[ADHelpers] begin
    test_working_backend("Mooncake reverse")
end

@testitem "Mooncake forward gradients (marginal)" tags=[:ad, :mooncake, :mooncake_forward] setup=[ADHelpers] begin
    test_working_backend("Mooncake forward")
end

@testitem "Enzyme forward gradients (marginal)" tags=[:ad, :enzyme, :enzyme_forward] setup=[ADHelpers] begin
    test_working_backend("Enzyme forward")
end

# === Latent / augmented-primary scenario group ===
# The single-record latent scenarios (`Latent*`, `PrimaryConditional`) are
# all-continuous arithmetic through the delay logpdf and the augmented
# primaries, so they differentiate on every backend (no entries in the
# broken/skip registries). The vectorised `latent_observed_logpdf` path is
# covered separately in `latent_tree_ad.jl` / `latent_vectorised_ad.jl`.

@testitem "ForwardDiff gradients (latent)" tags=[:ad, :forwarddiff] setup=[ADHelpers] begin
    test_working_backend("ForwardDiff"; category = :latent)
end

@testitem "ReverseDiff (tape) gradients (latent)" tags=[:ad, :reversediff] setup=[ADHelpers] begin
    test_working_backend("ReverseDiff (tape)"; category = :latent)
end

@testitem "Enzyme reverse gradients (latent)" tags=[:ad, :enzyme, :enzyme_reverse] setup=[ADHelpers] begin
    test_working_backend("Enzyme reverse"; category = :latent)
end

@testitem "Mooncake reverse gradients (latent)" tags=[:ad, :mooncake, :mooncake_reverse] setup=[ADHelpers] begin
    test_working_backend("Mooncake reverse"; category = :latent)
end

@testitem "Mooncake forward gradients (latent)" tags=[:ad, :mooncake, :mooncake_forward] setup=[ADHelpers] begin
    test_working_backend("Mooncake forward"; category = :latent)
end

@testitem "Enzyme forward gradients (latent)" tags=[:ad, :enzyme, :enzyme_forward] setup=[ADHelpers] begin
    test_working_backend("Enzyme forward"; category = :latent)
end

# === Recurrent / cyclic multi-state scenario group ===
# The `RecurrentStates` path likelihood and the `CTMCStates` jump-chain / panel
# likelihoods, differentiated w.r.t. the edge sojourn params / generator rates.
# The CTMC panel scenario stresses the backends through the `exp(Qt)` matrix
# exponential; genuine per-backend limitations are in the broken/skip registries.

@testitem "ForwardDiff gradients (recurrent)" tags=[:ad, :forwarddiff] setup=[ADHelpers] begin
    test_working_backend("ForwardDiff"; category = :recurrent)
end

@testitem "ReverseDiff (tape) gradients (recurrent)" tags=[:ad, :reversediff] setup=[ADHelpers] begin
    test_working_backend("ReverseDiff (tape)"; category = :recurrent)
end

@testitem "Enzyme reverse gradients (recurrent)" tags=[:ad, :enzyme, :enzyme_reverse] setup=[ADHelpers] begin
    test_working_backend("Enzyme reverse"; category = :recurrent)
end

@testitem "Mooncake reverse gradients (recurrent)" tags=[:ad, :mooncake, :mooncake_reverse] setup=[ADHelpers] begin
    test_working_backend("Mooncake reverse"; category = :recurrent)
end

@testitem "Mooncake forward gradients (recurrent)" tags=[:ad, :mooncake, :mooncake_forward] setup=[ADHelpers] begin
    test_working_backend("Mooncake forward"; category = :recurrent)
end

@testitem "Enzyme forward gradients (recurrent)" tags=[:ad, :enzyme, :enzyme_forward] setup=[ADHelpers] begin
    test_working_backend("Enzyme forward"; category = :recurrent)
end
