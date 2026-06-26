@testitem "Aqua.jl" tags=[:quality] begin
    using EpiAwarePackageTools: test_aqua
    # The full standard Aqua suite (unbound args, undefined exports, project
    # extras, stale deps, deps compat, undocumented names, piracies,
    # ambiguities) over CensoredDistributions. The per-check wiring lives in
    # `EpiAwarePackageTools.test_aqua`; nothing here is CD-specific, so CD just
    # calls it.
    test_aqua(CensoredDistributions)
end
