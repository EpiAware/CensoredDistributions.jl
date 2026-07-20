# `as_turing`'s differentiated surface (ComposedDistributions.logdensity's
# unflatten -> update -> loglik reconstruction), one test item per backend so
# the per-backend CI (#269) can select a single one.

@testitem "as_turing logdensity: ForwardDiff gradients" tags=[:ad, :forwarddiff] setup=[ComposedTuringADHelpers] begin
    test_composed_logdensity_backend("ForwardDiff")
end

@testitem "as_turing logdensity: ReverseDiff (tape) gradients" tags=[:ad, :reversediff] setup=[ComposedTuringADHelpers] begin
    test_composed_logdensity_backend("ReverseDiff (tape)")
end

@testitem "as_turing logdensity: Enzyme reverse gradients" tags=[
    :ad, :enzyme, :enzyme_reverse] setup=[ComposedTuringADHelpers] begin
    test_composed_logdensity_backend("Enzyme reverse")
end

@testitem "as_turing logdensity: Mooncake reverse gradients" tags=[
    :ad, :mooncake, :mooncake_reverse] setup=[ComposedTuringADHelpers] begin
    test_composed_logdensity_backend("Mooncake reverse")
end

@testitem "as_turing logdensity: Mooncake forward gradients" tags=[
    :ad, :mooncake, :mooncake_forward] setup=[ComposedTuringADHelpers] begin
    test_composed_logdensity_backend("Mooncake forward")
end

@testitem "as_turing logdensity: Enzyme forward gradients" tags=[
    :ad, :enzyme, :enzyme_forward] setup=[ComposedTuringADHelpers] begin
    test_composed_logdensity_backend("Enzyme forward")
end
