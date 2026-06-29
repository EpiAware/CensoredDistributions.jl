# Bundled data: Epuyén 2018-19 Andes virus line list

`linelist.csv` is the Epuyén Andes hantavirus (ANDV) outbreak line list,
hand-encoded from Table S2 of the supplementary appendix of:

> Martínez VP, Di Paola N, Alonso DO, et al.
> "Super-spreaders" and person-to-person transmission of Andes virus in
> Argentina.
> N Engl J Med 2020;383:2230-41.
> [doi:10.1056/NEJMoa2009040](https://doi.org/10.1056/NEJMoa2009040)

The copy here is taken from the re-analysis at
[epiforecasts/andv-linelist-analysis](https://github.com/epiforecasts/andv-linelist-analysis),
which is released under the MIT licence (Copyright (c) 2026 Sebastian Funk
and contributors; see `LICENSE`).
It is redistributed here under that licence for the case-study page
[Real-time Andes virus delays](../../andv-linelist-analysis.md).
No values have been modified in this copy.

Columns: patient ID, age, sex, residence, exposure place, exposure window
(lower / upper), onset date, attributed source (or `index` for the zoonotic
case), relationship to source, transmission wave, observed offspring count
`Z`, and free-text notes.
The two `_alt` rows are alternative-source sensitivity records that the
upstream main fit excludes; the page drops them as well.
