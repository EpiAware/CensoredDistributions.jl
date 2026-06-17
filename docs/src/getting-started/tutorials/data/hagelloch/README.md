# Bundled data: 1861 Hagelloch measles outbreak

`linelist.csv` is the 1861 Hagelloch (Germany) measles line list (n = 188
children), the canonical dataset for pairwise survival analysis of
transmission.
The data were originally collected by Pfeilsticker (1863) and re-analysed
by Oesterle (1992), and have been used repeatedly to test transmission
models because every case carries a putative infector and a household.

The copy here is taken verbatim from the
[outbreaks](https://github.com/reconhub/outbreaks) R package
(`measles_hagelloch_1861`), released under the MIT licence (see
`LICENSE`), which in turn formatted it from `hagelloch.df` in the
[surveillance](https://CRAN.R-project.org/package=surveillance) R package.
No values have been modified in this copy.

Columns:

- `case_ID`: case id number.
- `infector`: case id of the putative source of infection (the
  who-infected-whom ground truth used here only as a comparison target).
- `date_of_prodrome`: date of onset of prodromal symptoms (taken here as
  the onset of infectiousness that starts the contact-interval clock).
- `date_of_rash`: date of onset of rash.
- `date_of_death`: date of death (`NA` means recovered).
- `age`: age in years.
- `gender`: `f` / `m`.
- `family_ID`: household (family) id, the close-contact group structure.
- `class`: school class (`0` preschool, `1` first class, `2` second
  class).
- `complications`, `x_loc`, `y_loc`: complications flag and house
  coordinates (unused here).

If you use this data in your own work, please cite:

> Pfeilsticker, A. (1863). Beiträge zur Pathologie der Masern.
> M.D. Thesis, Eberhard-Karls-Universität Tübingen.
>
> Oesterle, H. (1992). Statistische Reanalyse einer Masernepidemie 1861
> in Hagelloch. M.D. Thesis, Eberhard-Karls-Universität Tübingen.
>
> Neal, P. J. and Roberts, G. O. (2004). Statistical inference and model
> selection for the 1861 Hagelloch measles epidemic.
> Biostatistics 5(2):249-261.

The pairwise survival method analysed on this page is from:

> Kenah, E. (2011). Contact intervals, survival analysis of epidemic
> data, and estimation of R0. Biostatistics 12(3):548-566.
> [doi:10.1093/biostatistics/kxq068](https://doi.org/10.1093/biostatistics/kxq068)
