@testitem "Docstring format" tags=[:quality] begin
    using EpiAwarePackageTools: test_docstring_format
    using CensoredDistributions

    # Docstring-convention checks (field docs on structs, `# Arguments` /
    # `# Keyword Arguments` sections, `@example` blocks on exported/public
    # functions, a shown signature, and resolvable `@ref` cross-references)
    # run through the shared kit helper. The kit defaults match CD's previous
    # behaviour: field docs required, examples required only for
    # exported/public symbols. `crossref_ignore` carries the upstream
    # Distributions names CD legitimately links to in "See also" sections.
    test_docstring_format(CensoredDistributions;
        crossref_ignore = (:pdf, :cdf, :logpdf, :logcdf, :rand, :quantile))
end
