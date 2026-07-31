I assume the same **40 µL aqueous droplet volume** used in the JACS work, with **0.01 wt% Tween 20 in DI water as the backfill**, followed by oil segmentation. The Experimental Procedures specify 16 mM citrate, 2 mM HAuCl₄, and 10 mM HCl/NaOH stocks; using the 16 mM citrate stock exactly reproduces the integer-volume recipes below. The reported standard condition was 100 °C with a nominal 10 min reaction age. ([Brookhaven National Laboratory][1])

## Three JACS reference recipes

The three high-performing conditions highlighted in Figure 4 were:

| Reference    | Final concentrations: Cit / signed titrant / Au | 40 µL recipe                                             | Reported result                        |
| ------------ | ----------------------------------------------: | -------------------------------------------------------- | -------------------------------------- |
| **JACS-U**   |                          4.80 / +1.75 / 0.95 mM | 12 µL citrate + 7 µL NaOH + 19 µL HAuCl₄ + 2 µL backfill | (d\approx13.4) nm, (\sigma\approx0.11) |
| **JACS-30A** |                              0.40 / 0 / 0.80 mM | 1 µL citrate + 16 µL HAuCl₄ + 23 µL backfill             | (d\approx31.8) nm, (\sigma\approx0.17) |
| **JACS-30B** |                          0.80 / +0.25 / 0.85 mM | 2 µL citrate + 1 µL NaOH + 17 µL HAuCl₄ + 20 µL backfill | (d\approx28.2) nm, (\sigma\approx0.16) |

Here, positive titrant means NaOH and negative titrant means HCl. These three conditions correspond to the Run-1 global dispersity minimum and the two Run-2 optima near the 30 nm target. ([Brookhaven National Laboratory][1])

## Recommended first campaign: 12 unique recipes

This matrix includes the three JACS references, symmetric acid/base perturbations, citrate-ratio variation, and low/high gold conditions. All aqueous volumes total **40 µL**.

| ID                  | 16 mM citrate, µL | 10 mM HCl, µL | 10 mM NaOH, µL | 2 mM HAuCl₄, µL | Backfill, µL | Final Cit / titrant / Au, mM | Cit/Au |
| ------------------- | ----------------: | ------------: | -------------: | --------------: | -----------: | ---------------------------- | -----: |
| **R1 JACS-U**       |                12 |             0 |              7 |              19 |            2 | 4.80 / +1.75 / 0.95          |   5.05 |
| **R2 JACS-30A**     |                 1 |             0 |              0 |              16 |           23 | 0.40 / 0 / 0.80              |   0.50 |
| **R3 JACS-30B**     |                 2 |             0 |              1 |              17 |           20 | 0.80 / +0.25 / 0.85          |   0.94 |
| R4 low-Cit acid     |                 1 |             2 |              0 |              16 |           21 | 0.40 / −0.50 / 0.80          |   0.50 |
| R5 low-Cit base     |                 1 |             0 |              2 |              16 |           21 | 0.40 / +0.50 / 0.80          |   0.50 |
| R6 center neutral   |                 6 |             0 |              0 |              16 |           18 | 2.40 / 0 / 0.80              |   3.00 |
| R7 center acid      |                 6 |             2 |              0 |              16 |           16 | 2.40 / −0.50 / 0.80          |   3.00 |
| R8 center base      |                 6 |             0 |              2 |              16 |           16 | 2.40 / +0.50 / 0.80          |   3.00 |
| R9 high-Cit neutral |                12 |             0 |              0 |              16 |           12 | 4.80 / 0 / 0.80              |   6.00 |
| R10 high-Cit base   |                12 |             0 |              4 |              16 |            8 | 4.80 / +1.00 / 0.80          |   6.00 |
| R11 low-Au          |                 6 |             0 |              2 |              12 |           20 | 2.40 / +0.50 / 0.60          |   4.00 |
| R12 high-Au         |                 6 |             0 |              2 |              24 |            8 | 2.40 / +0.50 / 1.20          |   2.00 |

The general conversion is:

[
[\mathrm{Cit}]*{\rm final}=0.4V*{\rm Cit},
\quad
[\mathrm{Au}]*{\rm final}=0.05V*{\rm Au},
\quad
[\mathrm{H/OH}]*{\rm final}=\pm0.25V*{\rm titrant}
]

with concentrations in mM and volumes in µL for a 40 µL droplet.

### Fifteen-run sequence with drift checks

Use R1 at the beginning, middle, and end, and repeat R8 once:

**R1, R6, R11, R9, R2, R8, R12, R1, R5, R3, R10, R8, R7, R4, R1**

This gives 12 unique chemistries and four measurements of repeatability/drift. R4 is placed near the end because the combination of low citrate and acid is more likely to produce broad or aggregated products and possible wall deposition. The JACS analysis similarly found acidic conditions to give larger and more variable outcomes, whereas neutral-to-basic conditions were more reproducible. ([Brookhaven National Laboratory][1])

## Kinetic acquisition

A 10 min endpoint alone will produce a synthesis–outcome model, not a kinetic model. Each recipe should provide spectra at multiple effective reaction ages. A useful grid is:

[
t=5,\ 10,\ 20,\ 30,\ 45,\ 60,\ 90,\ 120,\ 180,\ 300,\ 600\ {\rm s}
]

Acquire at approximately 1 Hz whenever possible, especially during the first 60–120 s. In a conventional Turkevich synthesis, the seed population—and therefore much of the final-size information—can be established within roughly the first 30 s, even though precursor reduction and particle growth continue for minutes. ([ACS Publications][2])

For consistent (t=0):

1. Premix citrate, HCl or NaOH, and backfill.
2. Introduce HAuCl₄ last, immediately before the static mixer.
3. Define (t=0) at the mixer exit.
4. Correct the nominal age for transfer time and thermal equilibration before the solution reaches 100 °C.

If UV–vis is available at only one reactor position, obtain different ages by changing residence time or using stop-flow. Keep the mixing conditions unchanged as much as possible.

## Quantities to extract from every spectrum

Do not model only (\lambda_{\max}), because the LSPR position and width are jointly affected by particle size, morphology, aggregation, and dielectric environment. This convolution was one reason SAXS provided more quantitative size feedback than UV–vis in the JACS study. ([Brookhaven National Laboratory][1])

Extract at least:

* (A_{400}(t)): preliminary Au(0) conversion proxy.
* LSPR integrated area, preferably over approximately 480–650 nm.
* (\lambda_{\max}(t)).
* LSPR FWHM.
* (A_{650}/A_{520}) or a similar red-tail metric for aggregation.
* Baseline/scattering level around 700–750 nm.

Absorbance at 400 nm can estimate colloidal Au(0), but its accuracy depends on particle size, surface chemistry, and oxidation state. Published validation reported uncertainties as large as approximately 30% across arbitrary systems, but below about 5% after calibration for a specific Turkevich system. Therefore, normalize (A_{400}) against a fully reacted endpoint for each chemistry region rather than immediately treating it as an absolute conversion. ([ACS Publications][3])

A practical first model is:

[
X_{\mathrm{Au(0)}}(t)
=====================

\frac{A_{400}(t)-A_{400}(0)}
{A_{400}(\infty)-A_{400}(0)}
]

followed by a lagged stretched-exponential fit,

[
X(t)=
\begin{cases}
0,&t\leq t_{\rm lag}\
1-\exp[-k(t-t_{\rm lag})^n],&t>t_{\rm lag}
\end{cases}
]

Then regress (t_{\rm lag}), (k), (n), final LSPR area, final (\lambda_{\max}), and FWHM against ([\mathrm{Au}]), (\log(\mathrm{Cit/Au})), and signed titrant concentration. Treat signed HCl/NaOH concentration as the controlled variable; it is a **pH proxy**, not the actual high-temperature pH.

[1]: https://www.bnl.gov/tcp/uploads/files/2024-007j.3.pdf?utm_source=chatgpt.com "Autonomous Nanoparticle Synthesis Guided by In Situ Multiscale ..."
[2]: https://pubs.acs.org/doi/10.1021/acsnano.5b01579?utm_source=chatgpt.com "Turkevich in New Robes: Key Questions Answered for the Most Common Gold Nanoparticle Synthesis | ACS Nano"
[3]: https://pubs.acs.org/doi/10.1021/ac502053s?utm_source=chatgpt.com "In Situ Determination of Colloidal Gold Concentrations with UV–Vis Spectroscopy: Limitations and Perspectives | Analytical Chemistry"
