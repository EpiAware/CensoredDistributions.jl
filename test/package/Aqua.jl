@testitem "Aqua.jl quality" tags=[:quality] begin
    using EpiAwarePackageTools: test_aqua
    # The standard Aqua suite (unbound args, undefined exports, project
    # extras, stale deps, deps compat, undocumented names, piracies,
    # ambiguities) over the package, via the shared kit wrapper. CD runs the
    # full suite, so no checks are relaxed here.
    test_aqua(CensoredDistributions)
end
