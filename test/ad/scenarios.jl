# Per-backend AD gradient scenario suite. Each backend is its own test
# item, tagged so the per-backend CI (#269) can select it with a tag
# filter (e.g. `julia test/ad/runtests.jl enzyme_reverse`). With no tag
# every item runs, as `task test-ad` does locally.

@testitem "ForwardDiff gradients" tags=[:ad, :forwarddiff] setup=[ADHelpers] begin
    test_working_backend("ForwardDiff")
end

@testitem "ReverseDiff (tape) gradients" tags=[:ad, :reversediff] setup=[ADHelpers] begin
    test_working_backend("ReverseDiff (tape)")
end

@testitem "Enzyme reverse gradients" tags=[:ad, :enzyme, :enzyme_reverse] setup=[ADHelpers] begin
    test_working_backend("Enzyme reverse")
end

@testitem "Mooncake reverse gradients" tags=[:ad, :mooncake, :mooncake_reverse] setup=[ADHelpers] begin
    test_working_backend("Mooncake reverse")
end

@testitem "Mooncake forward gradients" tags=[:ad, :mooncake, :mooncake_forward] setup=[ADHelpers] begin
    test_working_backend("Mooncake forward")
end

@testitem "Enzyme forward gradients (partial, #225)" tags=[:ad, :enzyme, :enzyme_forward] setup=[ADHelpers] begin
    test_partial_backend("Enzyme forward")
end
