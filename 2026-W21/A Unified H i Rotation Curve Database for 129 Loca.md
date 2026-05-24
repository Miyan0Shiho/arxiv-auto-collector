# A Unified H i Rotation Curve Database for 129 Local Volume Dwarf and Irregular Galaxies

**Authors**: David C. Flynn

**Published**: 2026-05-21 08:34:06

**PDF URL**: [https://arxiv.org/pdf/2605.22163v1](https://arxiv.org/pdf/2605.22163v1)

## Abstract
We present a unified H i rotation curve database for 129 dwarf and irregular galaxies drawn from four Local Volume surveys: the Local Volume H i Survey (LVHIS; 33 galaxies), VLA-ANGST (29), LITTLE THINGS (26), and WALLABY DR2 (41). The database provides standardised kinematic parameters, distance estimates, morphological classifications, and rotation curve data in machine-readable JSON, JSONL, and CSV formats with a documented 27-field schema, supporting retrieval-augmented generation (RAG) applications and cross-survey kinematic analysis. Quality tiers distinguish 26 galaxies with full multi-point tilted-ring rotation curves from 103 with single-ring or profile-width estimates. Three worked examples demonstrate corpus queries, including application of the ω correction to DDO 154 (LITTLE THINGS). This work is presented as a data resource; no new dynamical model is proposed. The database and all computation scripts are available at Zenodo (https://doi.org/10.5281/zenodo.20320362).

## Full Text


<!-- PDF content starts -->

Draft version May 22, 2026
Typeset using L ATEX default style in AASTeX631
A Unified HiRotation Curve Database for 129 Local Volume Dwarf and Irregular Galaxies
D. C. Flynn1
1EPS Research, Laurel, MD 20708, USA
(Received; Revised; Accepted)
Submitted to PASP
ABSTRACT
We present a unified Hirotation curve database for 129 dwarf and irregular galaxies drawn from
four Local Volume surveys: the Local Volume HiSurvey (LVHIS; 33 galaxies), VLA-ANGST (29),
LITTLE THINGS (26), and WALLABY DR2 (41). The database provides standardised kinematic
parameters, distance estimates, morphological classifications, and rotation curve data in machine-
readable JSON, JSONL, and CSV formats with a documented 27-field schema, supporting retrieval-
augmented generation (RAG) applications and cross-survey kinematic analysis. Quality tiers distin-
guish 26 galaxies with full multi-point tilted-ring rotation curves from 103 with single-ring or profile-
width estimates. Three worked examples demonstrate corpus queries, including application of the
ωcorrection to DDO 154 (LITTLE THINGS). This work is presented as a data resource; no new
dynamical model is proposed. The database and all computation scripts are available at Zenodo
(https://doi.org/10.5281/zenodo.20320362).
Keywords:catalogs — galaxies: dwarf — galaxies: irregular — galaxies: kinematics and dynamics —
galaxies: fundamental parameters — radio lines: galaxies
1.INTRODUCTION
Dwarf and irregular galaxies occupy a unique position in galactic kinematic studies. Their shallow potential wells,
gas-dominated baryonic inventories, and high dark matter fractions (e.g., Oh et al. 2015; Koribalski et al. 2019) make
them sensitive probes of the relationship between baryonic mass and observed kinematics — a relationship that remains
contested in the context of modified gravity frameworks (Milgrom 1983; McGaugh et al. 2016) and ΛCDM halo models
(Navarro et al. 1997).
The Hirotation curves of Local Volume dwarfs have been systematically measured by four major interferometric
surveys. The Local Volume HiSurvey (LVHIS; Koribalski et al. 2019) used the Australia Telescope Compact Array
(ATCA) to observe 82 galaxies within 10 Mpc. The VLA-ANGST survey (VLA-ANGST; Ott et al. 2012) targeted 35
dwarf galaxies in the Local Volume with the Very Large Array (VLA). The LITTLE THINGS survey (Local Irregulars
That Trace Luminosity Extremes; Hunter et al. 2012; Oh et al. 2015) provided high-resolution VLA Hiimaging and
tilted-ring rotation curves for 41 nearby dwarf irregular galaxies. The WALLABY HiAll-Sky Survey Pilot Data
Release 2 (WALLABY DR2; Koribalski et al. 2020) with ASKAP extends coverage to larger distances with a blind
survey approach.
Despite the scientific importance of this population, no unified kinematic corpus covering all four surveys in a
machine-readable, schema-consistent format has been published. Individual survey catalogs use different column
conventions, distance methods, and quality flags, making cross-survey comparison and retrieval-augmented generation
(RAG) applications challenging.
davidflynn@eps-research.com
Corresponding author: D. C. FlynnarXiv:2605.22163v1  [astro-ph.GA]  21 May 2026

2
The present work addresses this gap. We assemble and standardise rotation curve data from all four surveys into
a unified JSON/CSV corpus with a documented schema, quality tiers, and omega-readiness flags. Theωcorrection
is treated throughout as an empirical kinematic descriptor; no modification to Newtonian or relativistic gravity is
proposed or implied.
This corpus is the third in a series:
1. Flynn & Cannaliato (2025) introduced theωkinematic correction and demonstrated its empirical performance
across 84 SPARC Q=1 disk galaxies (Flynn & Cannaliato 2025).
2. Flynn (2026a) validated theωcorrection across the same 84 SPARC galaxies with full baryonic decomposition,
achieving a 2.4×improvement over MOND on the same target (Flynn 2026a).
3. Flynn (2026b) published a unified Hirotation curve corpus of 438 galaxies from SPARC, THINGS, LITTLE
THINGS, and WALLABY DR2, submitted toAstronomy & Computing(Flynn 2026b, ASCOM-D-26-00129).
4. This work provides the dwarf/irregular extension corpus (129 galaxies) and applies theωcorrection to the 24
omega-ready LITTLE THINGS galaxies.
Beyond its value as a data resource, this corpus enables a qualitatively new test of theωcorrection: does the
kinematic regularity identified in massive spirals persist into genuinely dark-matter-dominated dwarfs where baryonic
disk physics is fundamentally different?
2.SURVEY SOURCES AND SELECTION
2.1.LVHIS
The Local Volume HiSurvey (Koribalski et al. 2019) provides ATCA Hiobservations of 82 galaxies within ap-
proximately 10 Mpc, with published tilted-ring kinematic models, systemic velocities, inclinations, and position an-
gles. We ingest 33 galaxies after excluding six massive spirals and ellipticals with peak rotation velocities exceed-
ing 150 km s−1and non-Irr/Im/dIrr morphological classifications: NGC 253 (SABc, 200 km s−1), NGC 1313 (SBd,
220 km s−1), NGC 4945 (SBcd, 174 km s−1), NGC 5128/Cen A (S0, 260 km s−1), NGC 5236/M 83 (Sc, 150 km s−1), and
Circinus (Sb, 161 km s−1). The 33 retained LVHIS galaxies are ingested in seed mode from Table 9 of Koribalski et
al. (2019): each entry carries a single best-fit rotation velocity from the published tilted-ring model and is assigned
quality tier 2.
2.2.VLA-ANGST
The VLA-ANGST survey (Ott et al. 2012) targeted 35 dwarf galaxies in the Local Volume with the VLA in B and C
configurations. We ingest all 29 galaxies with published kinematic parameters. VLA-ANGST entries are single-point
profile-width estimates (V rotfromW 20/2 sini) and are assigned quality tier 2. HI radii are converted from published
arcsecond values using individual distance estimates.
2.3.LITTLE THINGS
The LITTLE THINGS survey (Hunter et al. 2012; Oh et al. 2015) provides VLA Hiimaging and 2DBAT tilted-ring
rotation curves for 41 nearby dwarf irregulars. We transfer all 26 galaxies with Q=1 quality flags from the Flynn
(2026b) v7.0 corpus (Flynn 2026b), retaining the full multi-point rotation curves. Radii are sorted monotonically on
ingest. LITTLE THINGS galaxies with regular rotation fields (26 of 41 per Oh et al. quality selection) are assigned
quality tier 1.
2.4.WALLABY DR2
WALLABY DR2 (Koribalski et al. 2020) provides ASKAP blind Hisurvey data. We filter 41 dwarf and irregular
candidates from the Flynn (2026b) v7.0 WALLABY DR2 component based on morphological classification and peak
rotation velocity. WALLABY entries include published Barolo tilted-ring models where available. Five WALLABY
galaxies have distances between 55 and 92 Mpc, reflecting the larger survey volume of ASKAP; these are retained with
a documented range note. The WALLABY DR2 blind survey catalog does not publish HI masses, dynamical masses,
HI radii, or optical morphological classifications; these fields (mhi logmsun,log mdyn,mhi mdyn ratio,rhi kpc,
hubble type) are absent for all 41 WALLABY entries and documented in theknown issuesfield of each record.

3
3.CORPUS CONSTRUCTION
3.1.Schema Design
Each galaxy entry in the corpus is a JSON object with 27 standardised fields covering identification, coordinates, mor-
phology, distance, kinematics, HI properties, rotation curve data, and quality metadata. Table 1 lists the core fields and
their units. The full 27-field schema is documented in the Zenodo deposit (https://doi.org/10.5281/zenodo.20320362).
Table 1.Core corpus schema fields.
Field Type Description
galaxy string Identifier
survey string LVHIS|VLA ANGST|LITTLE THINGS|WALLABY
distance mpc float Distance [Mpc]
distance method string TRGB|Cepheid|flow|SBF
incdeg float Inclination [◦]
padeg float Position angle [◦]
vrot max kms float Peak rotation velocity [km s−1]
npoints int Rotation curve points
rminkpc float Innermost radius [kpc]
rmax kpc float Outermost radius [kpc]
rhikpc float HI radius [kpc]
quality tier int 1 = full RC; 2 = seed/estimate
omega ready bool Suitable forωcorrection
data array [Rad(kpc),Vrot(km s−1),errV]
3.2.Quality Tiers
We define two quality tiers:
Tier 1(26 galaxies): Full multi-point tilted-ring rotation curves with published kinematic models. All LITTLE
THINGS galaxies and multi-point WALLABY entries qualify.
Tier 2(103 galaxies): Single-ring or profile-width velocity estimates from published best-fit models. All LVHIS and
VLA-ANGST entries, which provide single best-fit rotation velocities from Table 9 of Koribalski et al. (2019) and Ott
et al. (2012) respectively.
3.3.Omega-Readiness
A galaxy is flaggedomega ready=Trueif it satisfies: (1)n points≥5; (2) quality tier 1; and (3) the rotation field is
classified as kinematically regular. This yields 24 omega-ready galaxies, all from LITTLE THINGS.
Two LITTLE THINGS galaxies are explicitly excluded from the omega-ready subsample despite meeting the point-
count criterion:
IC 10: The innermost boundary hasV 1= 1.95 km s−1withσ V1= 12.71 km s−1— the observational uncertainty
exceeds the signal atR 1= 0.060 kpc. A negativeV obsatR= 0.080 kpc further indicates disturbed kinematics. The
maximum radiusR max= 0.54 kpc is insufficient for a reliable boundary fit.
NGC 3738: The inclination of 22.6◦falls below the reliable threshold for inclination-corrected rotation velocities.
Duplicate radii appear at both inner and outer boundaries, and the maximum rotation velocity (132.7 km s−1) is
anomalously high for a dIrr classification.
4.WORKED EXAMPLES
To demonstrate corpus utility, we provide three representative query examples. Full worked examples with retrievable
JSON contexts are provided inrag examples v1.jsonin the Zenodo deposit.
4.1.Example 1: Single-Galaxy Kinematic Query

4
Query:What is the inclination, position angle, and peak rotation velocity of IC 5152, and what group does it belong
to?
Result:IC 5152 (LVHIS) is a dwarf irregular (Im) atD= 1.97 Mpc with inclination 49.9◦, position angle 107◦, and
peak rotation velocityV rot= 39.5 km s−1. It is a member of the Sculptor group.
4.2.Example 2: Population Filter
Query:Which galaxies in the Centaurus A group have peak rotation velocities below 80 km s−1?
Result:Queryinggroup member = "Cen A"andvrot maxkms < 80returns 11 galaxies from the LVHIS component,
spanning Hubble types Im through dIrr, with distances 3.4–4.9 Mpc.
4.3.Example 3: Kinematic Correction Applied to DDO 154
Query:Apply the Flynn & Cannaliato (2025) two-boundary kinematic correction to DDO 154 (LITTLE THINGS).
Result:Using LITTLE THINGS boundary points from this corpus and the SPARC rotmod baryonic decomposition
for the Vbary curve (Lelli et al. 2016), and the innermost and outermost boundary points (R 1= 0.140 kpc,V 1=
6.20 km s−1;R2= 7.890 kpc,V 2= 50.51 km s−1), the angular velocity offset is:
ω=V2
R2−V1
R1R1
R23/2
=50.51
7.890−6.20
0.1400.140
7.8903/2
= 6.297 km s−1kpc−1= 6.162 rad Gyr−1,(1)
consistent with the SPARC spiral population mean of 7.06 rad Gyr−1(Flynn & Cannaliato 2025). Figure 1 shows
the four-curve decomposition for DDO 154 using the SPARC rotmod file (Lelli et al. 2016), which provides the full
baryonic decomposition.
5.DISCUSSION
We discuss the omega correction results and corpus limitations below.
5.1.Omega Correction: Descriptive Results
The 24 omega-ready LITTLE THINGS galaxies yieldωvalues from 3.6 to 32.7 rad Gyr−1with median 9.94 rad Gyr−1,
overlapping the SPARC spiral distribution (mean 7.06±3.26 rad Gyr−1; Flynn & Cannaliato 2025) within approxi-
mately one standard deviation. All 24 outer gapsV adj(R2)−V obs(R2) are negative, consistent with the Flynn (2026a)
result across 84 SPARC Q=1 spiral galaxies.
This corpus provides the data needed to test whetherωcorrelates with baryonic surface density, dark matter fraction,
or other physical parameters across the dwarf-to-spiral transition. Such tests require full baryonic decomposition of
the LITTLE THINGS galaxies, which is not available in the present corpus but is a natural target for future cross-
matching with available HI mass models. Cosmological simulation with RAMSES/osyris (Teyssier 2002) provides an
independent test of whetherωemerges from disk formation physics.
5.2.Limitations
The present corpus has several limitations that future versions should address:
No baryonic decomposition for LITTLE THINGS.The omega-ready subsample computesV adjbut cannot
complete the full RMSE(V adj, Vbary) comparison without Vgas/Vdisk baryonic decomposition. The DDO 154 example
uses the SPARC rotmod file as an exception. Future versions should cross-match LITTLE THINGS with available HI
mass models to enable full baryonic validation.
LVHIS and VLA-ANGST in seed mode.The 62 LVHIS and VLA-ANGST galaxies carry single-point velocity
estimates rather than full rotation curves. FITS-mode ingestion of the published data cubes would yield multi-point
profiles for many of these galaxies, significantly increasing the omega-ready subsample.
Missing vsys for LVHIS.Systemic velocities are absent for all 33 LVHIS entries in the current corpus. These can
be ingested from Table 4 of Koribalski et al. (2019) in a future version.
Distance heterogeneity.Five WALLABY galaxies have distances between 55 and 92 Mpc, beyond the strict Local
Volume definition. These are retained with documentation but may not be representative of the Local Volume dwarf
population.

5
Figure 1.Rotation curve decomposition for DDO 154 from the SPARC rotmod file (Lelli et al. 2016). Blue:V obs; green:
Vadj=V obs−Rω; orange dotted:V Kep; red dashed:V bary (Υ = 0.5). The SPARC rotmod boundary conditions yield
ω= 6.864 rad Gyr−1, slightly higher than the LITTLE THINGS corpus value of 6.162 rad Gyr−1(Section 4.3), reflecting the
independent kinematic modeling pipelines of the two surveys. Both values are consistent with the SPARC population mean
(Flynn & Cannaliato 2025). The outer gapV adj(R2)−V obs(R2) =−12.28 km s−1is negative, consistent with results across 84
SPARC Q=1 spiral galaxies.
6.SUMMARY
We have presented the Dwarf/Irregular Galaxy HiRotation Curve Corpus v1.0, a unified kinematic database of 129
dwarf and irregular galaxies assembled from LVHIS, VLA-ANGST, LITTLE THINGS, and WALLABY DR2. The
principal results are:
1. The corpus provides 129 galaxies in unified JSON/CSV/JSONL format with a 27-field schema, quality tiers, and
omega-readiness flags. All entries pass schema validation.
2. 24 LITTLE THINGS galaxies are classified omega-ready (quality tier 1,n points≥5, regular rotation fields).
3. Three worked examples demonstrate corpus queries: single-galaxy kinematic lookup (IC 5152), population
filter (Cen A group dwarfs), and a kinematic correction applied to DDO 154 (LITTLE THINGS), yielding
ω= 6.162 rad Gyr−1, consistent with the SPARC spiral population (Flynn & Cannaliato 2025).
The corpus and all scripts are available at Zenodo (https://doi.org/10.5281/zenodo.20320362).
The SPARC database is maintained by F. Lelli and S. McGaugh (CWRU). This work was conducted as independent
research by EPS Research and received no external funding.1
2
AI assistance.All numerical results, tables, and figures in this paper were computed exclusively by the author in
JupyterLab on a local HPC cluster (Node1: Intel i9-14900K, 128 GB DDR5, Ubuntu 24.04), directly from published
survey data. Large language models were used solely for manuscript review, L ATEX formatting, literature cross-checking,
and language editing. No AI system performed, verified, or generated any numerical analysis. All scientific content,
interpretations, and conclusions are the sole responsibility of the author.3
4
5
6
7

6
Software:Python (Van Rossum & Drake 2009), NumPy (Harris et al. 2020), SciPy (Virtanen et al. 2020), Matplotlib
(Hunter 2007), Astropy (Astropy Collaboration et al. 2022)
REFERENCES
Astropy Collaboration et al. 2022, ApJ, 935, 167
Flynn, D. C. 2026a, Baryonic Validation of the Omega
Kinematic Correction Across 84 SPARC Galaxies,
Preprint: Zenodo,
https://doi.org/10.5281/zenodo.20132805
Flynn, D. C. 2026b, A Unified HI Rotation Curve Corpus,
submitted toAstronomy and Computing
(ASCOM-D-26-00129); Zenodo,
https://doi.org/10.5281/zenodo.19563417;
arXiv:2604.13489
Flynn, D. C., & Cannaliato, J. 2025, Frontiers in
Astronomy and Space Sciences, 12, 1680387,
https://doi.org/10.3389/fspas.2025.1680387
Harris, C. R., et al. 2020, Nature, 585, 357,
https://doi.org/10.1038/s41586-020-2649-2
Hunter, J. D. 2007, Computing in Science & Engineering, 9,
90, https://doi.org/10.1109/MCSE.2007.55
Hunter, D. A., Ficut-Vicas, D., Ashley, T., et al. 2012, AJ,
144, 134, https://doi.org/10.1088/0004-6256/144/5/134
Koribalski, B. S., et al. 2019, MNRAS, 483, 4,
https://doi.org/10.1093/mnras/sty2440Koribalski, B. S., et al. 2020, Ap&SS, 365, 118,
https://doi.org/10.1007/s10509-020-03831-4
Lelli, F., McGaugh, S. S., & Schombert, J. M. 2016, AJ,
152, 157, https://doi.org/10.3847/0004-6256/152/6/157
McGaugh, S. S., Lelli, F., & Schombert, J. M. 2016, PRL,
117, 201101,
https://doi.org/10.1103/PhysRevLett.117.201101
Milgrom, M. 1983, ApJ, 270, 365,
https://doi.org/10.1086/161130
Navarro, J. F., Frenk, C. S., & White, S. D. M. 1997, ApJ,
490, 493, https://doi.org/10.1086/304888
Oh, S.-H., Hunter, D. A., Brinks, E., et al. 2015, AJ, 149,
180, https://doi.org/10.1088/0004-6256/149/6/180
Ott, J., et al. 2012, AJ, 144, 123,
https://doi.org/10.1088/0004-6256/144/4/123
Teyssier, R. 2002, A&A, 385, 337,
https://doi.org/10.1051/0004-6361:20011817
Van Rossum, G., & Drake, F. L. 2009, Python 3 Reference
Manual, Scotts Valley, CA: CreateSpace
Virtanen, P., et al. 2020, Nature Methods, 17, 261,
https://doi.org/10.1038/s41592-019-0686-2