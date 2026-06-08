# Bundled Andes virus line list

`andv-linelist.csv` is a verbatim copy of `data/linelist.csv` from the
[epiforecasts/andv-linelist-analysis](https://github.com/epiforecasts/andv-linelist-analysis)
repository, taken from the real-time extension branch (pull request #45,
`seabbs-bot/hantavirus@realtime`, commit `dade5020`).

The line list is hand-encoded from Table S2 of the supplementary appendix of
Martínez et al. 2020, the 2018-19 Epuyén Andes hantavirus (ANDV) outbreak.

## Provenance and licence

The source repository is released under the MIT licence, which permits
redistribution with attribution.
The original `LICENSE` text is reproduced in `LICENSE` next to this file.

Source data citation.

> Martínez VP, Di Paola N, Alonso DO, et al. "Super-spreaders" and
> person-to-person transmission of Andes virus in Argentina.
> N Engl J Med 2020;383:2230-41.
> [doi:10.1056/NEJMoa2009040](https://doi.org/10.1056/NEJMoa2009040)

Analysis code and the line-list encoding.

> Funk S, Abbott S. andv-linelist-analysis. epiforecasts, 2026.
> MIT licence. https://github.com/epiforecasts/andv-linelist-analysis

## Columns

`patient_id`, `age`, `sex`, `residence`, `exposure_place`,
`exposure_lower`, `exposure_upper` (the recorded infection window),
`onset_date`, `source_case` (the attributed source, or `index` for the
zoonotic index case via the `relationship` column), `relationship`, `wave`,
`Z` (observed offspring count), and `notes`.
