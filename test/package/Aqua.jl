
@testitem "Aqua.jl - Unbound args" tags=[:quality] begin
    using Aqua
    Aqua.test_unbound_args(CensoredDistributions)
end

@testitem "Aqua.jl - Undefined exports" tags=[:quality] begin
    using Aqua
    Aqua.test_undefined_exports(CensoredDistributions)
end

@testitem "Aqua.jl - Project extras" tags=[:quality] begin
    using Aqua
    Aqua.test_project_extras(CensoredDistributions)
end

@testitem "Aqua.jl - State deps" tags=[:quality] begin
    using Aqua
    Aqua.test_stale_deps(CensoredDistributions)
end

@testitem "Aqua.jl - Deps compat" tags=[:quality] begin
    using Aqua
    Aqua.test_deps_compat(CensoredDistributions)
end

@testitem "Aqua.jl - Undocumented names" tags=[:quality] begin
    using Aqua
    Aqua.test_undocumented_names(CensoredDistributions)
end

@testitem "Aqua.jl - Piracies" tags=[:quality] begin
    using Aqua
    Aqua.test_piracies(CensoredDistributions)
end

@testitem "Aqua.jl - Ambiguities" tags=[:quality] begin
    using Aqua
    Aqua.test_ambiguities(CensoredDistributions)
end
