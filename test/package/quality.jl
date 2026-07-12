# MANAGED by EpiAwarePackageTools.scaffold — do not edit by hand.
# Re-run `scaffold(pkgdir(MyPackage))` to update; the scheduled template-sync
# regenerates this file. Package-specific inputs (ignore lists, extension names,
# broken quarantines) live in the package-owned `qa_config.jl` this file reads.
#
# Standard package-quality testset. Routes every generic QA check through the
# shared EpiAwarePackageTools helpers over the package's own module. The package
# module and the QA config are supplied by `qa_config.jl`, which must define a
# `QA_CONFIG` NamedTuple (see the template for its fields).

@testitem "Quality: Aqua" tags=[:quality] begin
    using EpiAwarePackageTools
    include(joinpath(@__DIR__, "qa_config.jl"))
    test_aqua(QA_CONFIG.mod; QA_CONFIG.aqua...)
end

@testitem "Quality: ExplicitImports" tags=[:quality] begin
    using EpiAwarePackageTools
    include(joinpath(@__DIR__, "qa_config.jl"))
    test_explicit_imports(QA_CONFIG.mod; ignore = QA_CONFIG.ei_ignore)
end

@testitem "Quality: import centralisation" tags=[:quality] begin
    using EpiAwarePackageTools
    include(joinpath(@__DIR__, "qa_config.jl"))
    test_import_centralisation(QA_CONFIG.mod)
end

@testitem "Quality: docstring format" tags=[:quality] begin
    using EpiAwarePackageTools
    include(joinpath(@__DIR__, "qa_config.jl"))
    test_docstring_format(QA_CONFIG.mod;
        crossref_ignore = QA_CONFIG.crossref_ignore,
        QA_CONFIG.docstring...)
end

@testitem "Quality: README sections" tags=[:quality] begin
    using EpiAwarePackageTools
    include(joinpath(@__DIR__, "qa_config.jl"))
    # `readme` is a newer package-owned `QA_CONFIG` field; `qa_config.jl` is
    # package-owned (not re-applied by `scaffold_update`), so an adopter predating it has
    # no `readme` key. Default to the repo-root README with the standard section
    # requirements (#163) rather than erroring on the missing field. Warn, so a
    # typoed key does not quietly revert to the defaults (#188).
    cfg = if hasproperty(QA_CONFIG, :readme)
        QA_CONFIG.readme
    else
        @warn "QA_CONFIG has no `readme` field; checking the repo-root " *
              "README with the standard sections. Add one to qa_config.jl " *
              "to configure (or confirm) this."
        (; path = joinpath(@__DIR__, "..", ".."))
    end
    test_readme_sections(cfg.path;
        (k => v for (k, v) in pairs(cfg) if k !== :path)...)
end

@testitem "Quality: doctest" tags=[:quality] begin
    using EpiAwarePackageTools
    include(joinpath(@__DIR__, "qa_config.jl"))
    test_doctest(QA_CONFIG.mod)
end

@testitem "Quality: formatting" tags=[:quality] begin
    using EpiAwarePackageTools
    include(joinpath(@__DIR__, "qa_config.jl"))
    test_formatting(QA_CONFIG.mod)
end

@testitem "Quality: linting (JET)" tags=[:quality] begin
    using EpiAwarePackageTools
    include(joinpath(@__DIR__, "qa_config.jl"))
    test_linting(QA_CONFIG.mod; env = QA_CONFIG.jet_env)
end

@testitem "Quality: extension ambiguities" tags=[:quality] begin
    using EpiAwarePackageTools
    include(joinpath(@__DIR__, "qa_config.jl"))
    for ext in QA_CONFIG.extensions
        # Load the trigger packages, then check the extension's surface.
        for trigger in ext.triggers
            Base.require(Main, Symbol(trigger))
        end
        test_ext_ambiguities(QA_CONFIG.mod, ext.name;
            prefixes = ext.prefixes,
            expect_phantoms = get(ext, :expect_phantoms, false),
            broken = get(ext, :broken, false))
    end
end
