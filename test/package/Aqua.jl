
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
    using ConvolvedDistributions
    # The continuous `convolve_series` method deliberately extends a foreign
    # ConvolvedDistributions function on a foreign Distributions type — the
    # intended discretisation route for continuous delays
    # (CensoredDistributions#921). That is the Julia style guide's sanctioned
    # "coupled packages" form of type piracy (we organise both packages), so
    # Aqua is told to treat `ConvolvedDistributions.convolve_series` as ours to
    # extend here. Tracked in #927.
    Aqua.test_piracies(CensoredDistributions;
        treat_as_own = (ConvolvedDistributions.convolve_series,
            ConvolvedDistributions.delay_masses))
end

@testitem "Aqua.jl - Ambiguities" tags=[:quality] begin
    using Aqua
    Aqua.test_ambiguities(CensoredDistributions)
end
