# Bundled data: 2014-2016 Sierra Leone Ebola line list

`linelist.csv` is the Sierra Leone Ebola virus disease line list
(n = 8358) bundled in the epidist R package as `sierra_leone_ebola_data`,
collated by Fang et al. (2016):

> Fang LQ, Yang Y, Jiang JF, et al.
> Transmission dynamics of Ebola virus disease and intervention
> effectiveness in Sierra Leone.
> Proc Natl Acad Sci USA 2016;113(16):4488-93.
> [doi:10.1073/pnas.1518587113](https://doi.org/10.1073/pnas.1518587113)

The copy here is taken from
[epinowcast/epidist](https://github.com/epinowcast/epidist), which is
released under the MIT licence (Copyright (c) 2022 epidist authors; see
`LICENSE`).
It is redistributed here under that licence for the case-study page
[Stratified Ebola delays](../ebola-stratified-delays.md).
No values have been modified in this copy.

Columns: case id, age, sex (`Female`/`Male`/`NA`), symptom onset date,
sample tested (positive test) date, district (14 Sierra Leone districts),
and chiefdom.

The modelled delay is symptom onset to positive sample test, the same
delay analysed in the epidist
[Ebola vignette](https://epidist.epinowcast.org/articles/ebola.html) on
which this page is based.
If you use this data in your own work, please cite the Fang et al. (2016)
paper above.
