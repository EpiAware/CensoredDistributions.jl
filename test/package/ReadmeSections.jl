@testitem "README sections" tags=[:readme] begin
    using EpiAwarePackageTools: test_readme_sections
    # Assert the README carries the standard EpiAware section structure (H1
    # title, the managed badge markers, then Why/Overview, Getting
    # started/Usage, Documentation, Contributing, Citing/License) via the
    # shared kit check. CD's README is hand-curated and orders "Supporting and
    # citing" before "Contributing", so presence is checked but not order.
    root = pkgdir(CensoredDistributions)
    test_readme_sections(root; order = false)
end
