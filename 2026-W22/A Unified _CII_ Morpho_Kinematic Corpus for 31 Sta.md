# A Unified [CII] Morpho-Kinematic Corpus for 31 Star-Forming Galaxies at z = 4.26-5.68: The High-z Kinematic Corpus Z1

**Authors**: David C. Flynn

**Published**: 2026-05-25 01:53:26

**PDF URL**: [https://arxiv.org/pdf/2605.25339v1](https://arxiv.org/pdf/2605.25339v1)

## Abstract
We present the High-z Kinematic Corpus Z1, a structured machine-readable dataset of ALMA [C ii] 158 um morpho-kinematic data for 31 star-forming main-sequence galaxies at z = 4.26-5.68 drawn from the ALPINE survey (Jones et al. 2021; Le Fevre et al. 2020). The corpus is the fourth entry in the EPS Research RAG Astrophysics Corpus Series, extending coverage from Milky Way globular clusters and local HI rotation curves (Flynn 2026a,b,c) to the epoch approaching cosmic reionization. Eight confirmed rotators carry quality tier 1 per-ring rotation curves from 3DBarolo tilted-ring fits (Di Teodoro & Fraternali 2015), with 2-3 rings per galaxy, Vrot and sigma per ring, and dynamical mass estimates; the remaining 23 galaxies carry morpho-kinematic classification only (quality tier 2). All entries include stellar mass (Faisst et al. 2020), star formation rate, Wisnioski et al. (2015) disk criteria, and geometric parameters. The corpus is distributed as a single structured JSON file with nested per-ring kinematic data, a flat CSV for catalog-level filtering, a RAG-ready JSONL archive (one galaxy per line), and a per-galaxy ZIP archive. Three worked Jupyter notebook examples demonstrate single-galaxy [C ii] rotation curve analysis, corpus-level population statistics, and cross-corpus application of the Flynn & Cannaliato (2025) omega kinematic correction. Applying the omega formula to all 8 tier-1 rotators yields negative values (median -13.05 rad/Gyr), contrasting with positive values at z = 0 (+7.06 rad/Gyr for SPARC spirals; +9.94 rad/Gyr for local dwarfs), consistent with the known evolution from centrally concentrated high-z systems to extended rotating disks. The corpus is publicly available at Zenodo (DOI: 10.5281/zenodo.20369285) under CC BY 4.0.

## Full Text


<!-- PDF content starts -->

Draft version May 26, 2026
Typeset using L ATEXtwocolumnstyle in AASTeX631
A Unified [CII] Morpho-Kinematic Corpus for 31 Star-Forming Galaxies atz= 4.26–5.68: The High-z
Kinematic Corpus Z1
David C. Flynn1
1EPS Research, Laurel, MD, USA; ORCID: 0000-0002-2768-6650
ABSTRACT
We present the High-z Kinematic Corpus Z1, a structured machine-readable dataset of ALMA [Cii]
158µm morpho-kinematic data for 31 star-forming main-sequence galaxies atz= 4.26–5.68 drawn from
the ALPINE survey (Jones et al. 2021; Le F` evre et al. 2020). The corpus is the fourth and final entry
in the EPS Research RAG Astrophysics Corpus Series, extending coverage from Milky Way globular
clusters and local HI rotation curves (Flynn 2026a,b,c) to the epoch approaching cosmic reionization.
Eight confirmed rotators carry quality tier 1 per-ring rotation curves from 3DBarolo tilted-ring fits
(Di Teodoro & Fraternali 2015), with 2–3 rings per galaxy,V rotandσper ring, and dynamical mass
estimates; the remaining 23 galaxies carry morpho-kinematic classification only (quality tier 2). All
entries include stellar mass (Faisst et al. 2020), star formation rate, Wisnioski et al. (2015) disk criteria,
and geometric parameters. The corpus is distributed as a single structured JSON file with nested per-
ring kinematic data, a flat CSV for catalog-level filtering, a RAG-ready JSONL archive (one galaxy
per line), and a per-galaxy ZIP archive. Three worked Jupyter notebook examples demonstrate single-
galaxy [Cii] rotation curve analysis, corpus-level population statistics, and cross-corpus application
of the Flynn & Cannaliato (2025) omega kinematic correction. Applying the omega formula to all 8
tier-1 rotators yields negative values (median−13.05 rad Gyr−1), contrasting with positive values at
z= 0 (+7.06 rad Gyr−1for SPARC spirals; +9.94 rad Gyr−1for local dwarfs), consistent with the
known evolution from centrally concentrated high-zsystems to extended rotating disks. The corpus is
publicly available at Zenodo (DOI: 10.5281/zenodo.20369285) under CC BY 4.0.
Keywords:high-redshift galaxies — galaxy kinematics — ALPINE survey — ALMA — [Cii] 158µm
— rotation curves — RAG — LLM — data release — omega kinematic correction
1.INTRODUCTION
The kinematic properties of galaxies atz∼4–6 en-
code critical information about the assembly of rotating
disk structures and the distribution of baryonic and dark
matter in the early universe. At these redshifts, ALMA
observations of the [Cii] 158µm fine-structure line pro-
vide the most reliable tracer of cold gas morphology and
kinematics (Le F` evre et al. 2020; B´ ethermin et al. 2020).
The ALPINE survey (ALMA Large Program to INvesti-
gate [Cii] at Early Times) observed 118 main-sequence
star-forming galaxies atz= 4.4–5.9 (Le F` evre et al.
2020), of which Jones et al. (2021) published 3DBarolo
tilted-ring kinematic models for a subset of confirmed ro-
davidflynn@eps-research.comtators and morpho-kinematic classifications for the full
sample.
Despite the scientific importance of these data, no
unified machine-readable corpus currently exists that
combines the ALPINE kinematic catalog, per-ring ro-
tation curve data, Wisnioski et al. (2015) disk crite-
ria, stellar masses, and geometric parameters in a single
self-describing schema. This data-fragmentation prob-
lem is particularly acute for computational workflows—
including LLM-based retrieval-augmented generation
(RAG) pipelines—that require structured, consistently
typed input data.
This paper presents the High-z Kinematic Corpus
Z1, a structured JSON dataset containing morpho-
kinematic data for 31 ALPINE galaxies atz= 4.26–
5.68. The corpus is the fourth entry in the EPS Research
RAG Astrophysics Corpus Series:arXiv:2605.25339v1  [astro-ph.GA]  25 May 2026

2Flynn
1. Unified HI Rotation Curve Corpus v7.0 (Flynn
2026a): 438 galaxies,z= 0, HI 21 cm
2. Dwarf/Irregular HI Rotation Curve Corpus v1.0
(Flynn 2026b): 129 galaxies,z= 0, HI 21 cm
3. Milky Way Globular Cluster Corpus v1.3.1 (Flynn
2026c): 174 clusters, multi-survey
4.High-z Kinematic Corpus Z1 (this work):31
galaxies,z= 4.3–5.7, [Cii] 158µm
Together these four corpora form a unified RAG
knowledge base spanning Milky Way stellar clusters, lo-
cal spiral and dwarf galaxies, and the epoch approach-
ing cosmic reionization. We describe the corpus archi-
tecture (Section 2), the schema and quality tier sys-
tem (Section 3), the ingestion and verification proce-
dures (Section 4), three usage examples (Section 5), the
LLM/RAG application (Section 6), known limitations
(Section 7), and data availability (Section 8).
2.SOURCE DATA AND SURVEY COVERAGE
2.1.ALPINE Survey
The ALMA Large Program to INvestigate [Cii] at
Early Times (ALPINE; Le F` evre et al. 2020; B´ ethermin
et al. 2020) observed 118 main-sequence star-forming
galaxies atz= 4.4–5.9 with ALMA in the [Cii] 158µm
line (ALMA project 2017.1.00428.L). The survey was
designed to characterize the cold gas content, morphol-
ogy, and kinematics of typical star-forming galaxies dur-
ing the epoch of peak cosmic star formation. The syn-
thesized beam is approximately 1 arcsec FWHM, corre-
sponding to∼6–7 kpc atz∼5, setting the practical
spatial resolution floor for kinematic analysis.
2.2.Jones et al. (2021) Kinematic Catalog
Jones et al. (2021) applied the 3DBarolo tilted-ring
fitter (Di Teodoro & Fraternali 2015) to a subset of
ALPINE galaxies with sufficient [Cii] signal-to-noise
for kinematic modeling, publishing per-ring rotation ve-
locities, velocity dispersions, and dynamical masses in
their Table C1. Morpho-kinematic classifications (ROT,
MER, DIS, UNC) are provided for the full sample in
their Table 1, supplemented by the five Wisnioski et al.
(2015) disk criteria (Wisnioski et al. 2015). This corpus
ingests both tables as the primary kinematic source.
2.3.Ancillary Data
Stellar masses and star formation rates are drawn
from the ALPINE multi-wavelength photometry cata-
log (Faisst et al. 2020), which provides SED-fittedM ∗
and SFR for all ALPINE targets. The ALPINE DataTable 1.Corpus coverage summary.
Property Value
Total galaxies 31
Redshift rangez= 4.26–5.68
Maximum redshiftz= 5.6773 (DC773957)
Survey ALPINE
Telescope ALMA
Tracer [Cii] 158µm
Confirmed rotators (ROT) 8
Mergers (MER) 5
Dispersion-dominated (DIS) 3
Uncertain (UNC) 15
Quality tier 1 8
Quality tier 2 23
Release 1 (B´ ethermin et al. 2020) provides the [Cii] flux
catalog and morphological parameters.
Table 1 summarizes the corpus coverage.
3.CORPUS ARCHITECTURE AND SCHEMA
3.1.File Formats
The corpus is distributed in four com-
plementary formats. The master file
high zkinematic corpus Z1.jsonis a single JSON
document containing all 31 galaxy entries within a uni-
fied schema, together with a top-level metadata block.
The flat tablehigh zkinematic corpus Z1flat.csv
provides one row per galaxy (31 rows) with sum-
mary statistics for sample selection and filtering. The
JSONL filehigh zkinematic corpus Z1.jsonlcon-
tains one self-describing JSON object per line, opti-
mized for LLM/RAG ingestion. The per-galaxy archive
high zkinematic corpus Z1bygalaxy.zipcontains
31 individual JSON files, each self-contained with full
corpus metadata and the complete rotation curve array.
3.2.Quality Tier System
A two-tier quality annotation is applied at the galaxy
level. Tier 1 (8 confirmed rotators) denotes galaxies with
per-ringV rot,σ, andM dynfrom the Jones et al. (2021)
3DBarolo fits (their Table C1). Tier 2 (23 galaxies)
denotes morpho-kinematic classification only, with no
reliable rotation curve. The tier system enables down-
stream analyses to filter by data quality without inspect-
ing individual galaxies.
Tier-1 qualification note:Although all 8 tier-1 en-
tries carry per-ring fits from a peer-reviewed source,

High-z Kinematic Corpus Z13
Table 2.Per-galaxy schema summary.
Field Type Description
galaxystr Source name
redshiftfloat Spectroscopic redshift
class jones2021str ROT|MER|DIS|UNC
isrotatorbool Boolean flag
w15criteriadict Five W15 disk criteria
inckindegfloat Kinematic inclination
pakindegfloat Kinematic PA (deg)
vrot maxkmsfloat Peak rotation velocity
sigma mean kmsfloat Mean velocity dispersion
vover sigmafloat Kinematic state indicator
logmdyn msunfloat log 10dynamical mass
logmstar msunfloat log 10stellar mass
sfrmsun yrfloat Star formation rate
quality tierint 1 (per-ring)|2 (class only)
beam smearedbool True (all ALPINE)
datalist Per-ring RC (tier 1 only)
Note—Per-ringdataentries containR kpc,Vrot kms,
eVrot kms,sigma kms,e sigma kms,Mdyn msun, and
vover sigma. Tier-2 galaxies have an emptydataar-
ray and null kinematic fields.
the astrophysical quality is limited by ALMA resolution
(∼1 arcsec beam,∼6–7 kpc atz∼5) and ring counts
of 2–3 per galaxy. These are the best currently avail-
able spatially resolved [Cii] rotation curves atz∼4–6,
but are marginal by the standards of local HI surveys.
Users applying quantitative kinematic analyses should
treat tier-1 Z1 entries with commensurate caution.
3.3.Per-Galaxy Schema
Table 2 summarizes the per-galaxy fields available in
the corpus.
3.4.Omega Correction Compatibility
The corpus schema is designed to support the Flynn
& Cannaliato (2025) omega kinematic correction:
ω=V2
R2−V1
R1R1
R23/2
[rad Gyr−1] (1)
where (R 1, V1) and (R 2, V2) are the innermost and out-
ermost fitted ring boundary points. For tier-1 galax-
ies,data[0]provides (R 1, Vrot,1) anddata[-1]provides
(R2, Vrot,2).
4.INGESTION AND VERIFICATION4.1.ALPINE Data Ingestion
Kinematic parameters were ingested from Jones et al.
(2021) Tables 1 and C1. Per-ring rotation velocities, ve-
locity dispersions, dynamical masses, inclinations, and
position angles for the 8 confirmed rotators were digi-
tized from Table C1. Morpho-kinematic classifications,
Wisnioski et al. (2015) criteria, and integrated kinematic
parameters were ingested from Table 1. Stellar masses
and SFRs were cross-matched from Faisst et al. (2020).
Spectroscopic redshifts were drawn from the ALPINE
catalog (B´ ethermin et al. 2020).
The ingestion pipeline (alpine ingest.py) performs
five steps: (1) galaxy-by-galaxy data ingestion from
primary tables; (2) schema validation against the Z1
schema; (3) export to JSON, JSONL, CSV, and per-
galaxy ZIP formats; (4) RAG example generation; and
(5) README generation. All steps are fully scripted
and reproducible from the deposited source files.
4.2.Schema Validation
A dedicated schema validator (schema validator hz.py)
checks each galaxy entry for required field presence
and type correctness, controlled vocabulary compliance
(ROT|MER|DIS|UNC), Wisnioski criterion internal
consistency (boolean flags vs.n passedcount), boolean
flag consistency withclass jones2021, per-ring data
monotonicity, and tier-1/tier-2 data completeness rules.
All 31 galaxies pass validation with zero errors in stan-
dard mode; strict mode raises expected warnings for
null kinematic fields in tier-2 galaxies, which carry no
per-ring data by design.
4.3.Known Data Anomalies
Three data anomalies inherited from the primary
source are documented in the corpusknown issues
field:
CG32 and DC396844:logM dyn<logM ∗by 0.16
and 0.35 dex respectively. Dynamical mass below stellar
mass is physically implausible and likely reflects incli-
nation uncertainty (DC396844e inc= 31◦) compounded
by the 2-ring boundary constraint. Values are reported
as published in Jones et al. (2021) Table C1.
HZ9 outer-ring dispersion:The outermost ring of HZ9
hasσ= 4.82 km s−1, anomalously low relative to the
inner rings (σ∼71–75 km s−1), producingV/σ= 36.6
at the outer boundary. This likely reflects a 3DBarolo
fit artifact at the beam resolution limit.
DC519281 redshift uncertainty:e z= 0.02, ap-
proximately 40×larger than the typical uncertainty
(∼0.0005) in the sample.
5.USAGE EXAMPLES

4Flynn
The following three examples demonstrate the corpus
utility for common high-zkinematic analyses. Each ex-
ample is provided as a self-contained Jupyter notebook
(deposited alongside the corpus) loading data directly
from the JSON with no external preprocessing. All code
uses Python 3 with onlynumpyandmatplotlib. Out-
put figures are deposited as supplementary material.
5.1.Example 1: [Cii] Rotation Curve of a Confirmed
Rotator
Figure 1 shows J0817, the highest-velocity confirmed
rotator in the corpus (V rot,max = 252 km s−1atz=
4.26), with three panels extracted directly from the
corpus JSON:V rotwith error bars (top), velocity dis-
persion profile (bottom left), andV/σper ring (bot-
tom right). The metadata annotation (inclination, PA,
logM dyn, W15 criteria) is extracted directly from the
JSON. The point for this data descriptor is that all three
panels come from a single JSON load in under 15 lines
of Python.
J0817 rotation curve from corpus JSON.
import json, numpy as np, matplotlib.pyplot as plt
with open(’high_z_kinematic_corpus_Z1.json’) as f:
corpus = json.load(f)
g = next(g for g in corpus[’galaxies’]
if g[’galaxy’] == ’J0817’)
d = g[’data’]
R = np.array([p[’R_kpc’] for p in d])
Vrot = np.array([p[’Vrot_kms’] for p in d])
eVrot = np.array([p[’e_Vrot_kms’] for p in d])
sigma = np.array([p[’sigma_kms’] for p in d])
vos = np.array([p[’v_over_sigma’] for p in d])
plt.errorbar(R, Vrot, yerr=eVrot, fmt=’o-’)
Table 3 lists all 8 tier-1 rotators with outermost ring
properties.
5.2.Example 2: Corpus-Level Population Statistics
Figure 2 provides the corpus population overview
loaded from the flat CSV. Panel (a) shows the morpho-
kinematic class fractions (ROT: 8, MER: 5, DIS: 3,
UNC: 15). Panel (b) shows the redshift distribution by
class as a stacked histogram, with thez= 5 boundary
marked; 9 of 31 galaxies lie atz >5, with 4 UNC, 3
ROT, 1 DIS, and 1 MER in this subsample. Panel (c)
showsV/σvs. redshift for the 8 tier-1 rotators, with
the W15 thresholds atV/σ= 1 andV/σ= 3 marked.
Panel (d) shows a heatmap of the five Wisnioski et al.
(2015) disk criteria across all 8 rotators sorted by red-
shift.
Figure 1.J0817 (ALPINE confirmed rotator,z= 4.2605)
loaded from the Z1 corpus JSON.Top:V rotwith 3DBarolo
error bars (beam-smear corrected).Bottom left:Velocity
dispersion profile.Bottom right:V/σper ring. Metadata
(inc = 48◦, PA = 93◦, logM dyn= 10.623, W15: 3/5) ex-
tracted directly from the JSON.
Population statistics from flat CSV.
import csv, json
from collections import Counter
galaxies_csv = []
with open(’high_z_kinematic_corpus_Z1_flat.csv’) as f:
for row in csv.DictReader(f):
galaxies_csv.append(row)
classes = [r[’class_jones2021’] for r in galaxies_csv]
redshifts= [float(r[’redshift’]) for r in galaxies_csv]
class_counts = Counter(classes)
n_z5 = sum(1 for z in redshifts if z > 5.0)
Notably, two of the threez >5 rotators (DC552206,
σmean = 83.1 km s−1; HZ9,σ mean = 50.3 km s−1) have
mean dispersions exceeding 50 km s−1, consistent with
the elevated turbulence expected in high-zstar-forming
disks (Wisnioski et al. 2015). These values are available
only for tier-1 galaxies; the 6 non-ROT galaxies atz >5
have no reliableσestimate.
5.3.Example 3: Cross-Corpus Omega Kinematic
Application
Figure 3 demonstrates an application of the Flynn &
Cannaliato (2025) omega kinematic correction to the Z1
tier-1 rotators, connecting the high-zcorpus to thez= 0
EPS Research corpora. This example is illustrative; the
caveats in Section 7 apply in full.
Applying Equation (1) to all 8 tier-1 rotators us-
ing thedata[0]anddata[-1]boundary points yields

High-z Kinematic Corpus Z15
Table 3.Tier-1 confirmed rotators: outermost ring properties.
Galaxyz N rings rmax Vrot,out eVrot σout V/σW15
(kpc) (km s−1) (km s−1) (km s−1)
J0817 4.2605 2 2.84 252.09 14.94 35.98 7.006 3/5
CG32 4.4105 2 3.50 115.04 26.96 19.14 6.010 4/5
DC396844 4.5424 2 3.75 80.42 17.66 19.84 4.053 2/5
VC5110377875 4.5506 2 3.84 102.85 19.84 60.84 1.690 3/5
DC881725 4.5778 2 3.44 62.07 12.54 48.40 1.282 3/5
DC552206 5.5016 3 5.53 172.84 27.63 65.34 2.645 3/5
HZ9 5.5413 3 2.68 176.63 25.45 4.82 36.645 2/5
DC494057 5.5446 2 3.13 80.31 12.75 44.98 1.785 2/5
Note—r max: outermost ring radius.V rot,out ,σout: velocity and dispersion at outermost ring.
Inclination uncertainty not propagated intoV roterrors (3DBarolo systematic; see Jones et
al. 2021 Section 5.3.2). HZ9 outer-ringσ= 4.82 km s−1is anomalously low relative to inner
rings (σ∼71–75 km s−1); see Section 4.
Figure 2.High-z Kinematic Corpus Z1 population overview
(N= 31 galaxies,z= 4.3–5.7). (a) Morpho-kinematic class
fractions from Jones et al. (2021). (b) Redshift distribution
by class (stacked histogram); dashed line marksz= 5. (c)
V/σvs. redshift for tier-1 rotators; dotted lines atV/σ= 1
andV/σ= 3 (W15 threshold). (d) Wisnioski et al. (2015)
disk criteria heatmap for all 8 rotators sorted by redshift.
CG32 passes 4/5 criteria (most robustly confirmed disk);
HZ9 and DC396844 pass 2/5.
the values in Table 4. All 8 values are negative (me-
dianω=−13.05 rad Gyr−1, range−33.22 to−2.96
rad Gyr−1), indicating falling angular velocity profiles
(V/Rdecreasing outward). This contrasts with pos-
itive omega values atz= 0: SPARC mean +7.06±
3.26 rad Gyr−1across 84 galaxies (Flynn & Cannaliato
2025), and dwarf/irregular median +9.94 rad Gyr−1
across 24 omega-ready galaxies (Flynn 2026b).Omega computation from CII boundary velocities.
import json, numpy as np
with open(’high_z_kinematic_corpus_Z1.json’) as f:
corpus = json.load(f)
rotators = [g for g in corpus[’galaxies’]
if g.get(’is_rotator’) and
g.get(’quality_tier’) == 1]
results = []
for g in rotators:
d = g[’data’]
R1, V1 = d[0][’R_kpc’], d[0][’Vrot_kms’]
R2, V2 = d[-1][’R_kpc’], d[-1][’Vrot_kms’]
omega = (V2/R2 - V1/R1) * (R1/R2)**1.5
results.append({’galaxy’: g[’galaxy’],
’z’: g[’redshift’],
’omega’: omega})
The sign reversal across cosmic time is consistent with
the known evolution from compact, centrally concen-
trated systems at highzto extended rotating disks at
z= 0, but does not constitute a test of the omega frame-
work in the absence of baryonic decomposition. The
Z1 corpus provides the observational boundary condi-
tion for future RAMSES cosmological simulations trac-
ing this kinematic evolution fromz= 6 initial conditions
toz= 0.
6.APPLICATION TO LLM-BASED INFERENCE
A secondary design goal of the corpus is to serve as a
retrieval corpus for LLM-based RAG pipelines in astro-
physical research. The per-galaxy ZIP archive is opti-

6Flynn
Table 4.Omega values for tier-1 rotators.
Galaxyz R 2 Vrot,2 ω
(kpc) (km s−1) (rad Gyr−1)
J0817 4.2605 2.84 252.09−33.22
CG32 4.4105 3.50 115.04−13.05
DC396844 4.5424 3.75 80.42−14.48
VC5110377875 4.5506 3.84 102.85−12.73
DC881725 4.5778 3.44 62.07−13.05
DC552206 5.5016 5.53 172.84−2.96
HZ9 5.5413 2.68 176.63−20.14
DC494057 5.5446 3.13 80.31−9.53
Note—Omega computed from Equation (1) using boundary
pointsdata[0]anddata[-1]. All values negative, indicat-
ing fallingV/Rprofiles. Medianω=−13.05 rad Gyr−1.
DC552206 has a risingV rotprofile (V 2> V 1) but negative
ωbecauseV/Rdecreases outward. No baryonic decompo-
sition available; direct comparison toz= 0 requires caution
(Section 7).
Figure 3.Cross-corpus omega application. (a)ωvs. red-
shift for Z1 tier-1 rotators (diamonds) withz= 0 ref-
erence lines: SPARC mean +7.06 rad Gyr−1(blue solid),
dwarf/irregular median +9.94 rad Gyr−1(green dashed),
DDO161 +4.69 rad Gyr−1(red dotted). Orange diamonds:
negativeω; all 8 rotators show falling angular velocity pro-
files atz∼4–6. (b)|ω|vs. logM dyn. (c) EPS Research
corpus trilogy summary.Note:Z1 omega values are observa-
tional kinematics only; no baryonic decomposition available
atz∼5 (Section 7).mized for this use case: each file is a self-contained JSON
document of∼3–5 KB, well within typical LLM context
limits, containing all metadata and per-ring data needed
to answer kinematic queries without external lookups.
To assess whether the JSON schema is sufficiently self-
describing for automated consumption, we conducted a
structured usability evaluation with four LLMs: Google
Gemini Pro, Anthropic Claude, Microsoft Copilot Pro,
and Gemma 4 31B Dense (self-hosted via LM Studio
on Node1). Each model was presented with a single
per-galaxy JSON document and asked to perform three
benchmark tasks without additional documentation: (1)
plot the [Cii] rotation curve with error bars; (2) com-
puteV/σfor all tier-1 rotators from the flat CSV; and
(3) apply the Flynn & Cannaliato (2025) omega formula
to a specified galaxy. All four models successfully gen-
erated syntactically correct Python for all three tasks
on first attempt, requiring no additional prompting be-
yond a natural-language research question, consistent
with the findings of Flynn (2026a) for the local HI cor-
pora. These results suggest that the corpus’s explicit
column definitions, unit annotations, and quality flags
provide sufficient context for LLM-based code genera-
tion without external documentation.
7.KNOWN LIMITATIONS
Ten limitations should be noted by users of this cor-
pus.
(1) Maximum redshift isz= 5.6773, notz= 6.The
corpus designation “Z1” reflects its role as the high-
zanchor of the EPS Research series, approaching but
not reachingz= 6. The highest-redshift galaxy is
DC773957 (z= 5.6773). Truez= 6 spatially resolved
[Cii] rotation curves at astrophysically useful resolution
do not yet exist. The schema is designed to accommo-
date future REBELS, CRISTAL, and JWST IFU sam-
ples.
(2) Only 8/31 galaxies have per-ring rotation curve
data.The remaining 23 carry morpho-kinematic classi-
fication only.
(3) 2–3 rings per tier-1 galaxy.Omega and dynamical
mass estimates are sensitive to boundary conditions with
so few radial points.
(4) No baryonic decomposition.V gasandV diskare
not available atz∼5; omega values are observational
kinematics, not baryonic-model-corrected values. Direct
comparison toz= 0 omega requires future baryonic
modeling.
(5) Beam smearing.All ALPINE data has∼1 arcsec
beam (∼6–7 kpc atz∼5). 3DBarolo mitigates but does
not eliminate beam-smearing effects.

High-z Kinematic Corpus Z17
(6) ALPINE selection bias.The survey targets SFR>
fewM ⊙yr−1, missing the true progenitor population of
local dwarf irregulars (logM ∗∼7.5 atz= 0). Progen-
itor candidates identified in Example 3 are upper-mass
analogs, not confirmed DDO161-class progenitors.
(7) [Cii] vs. HI tracer difference.Direct kinematic
comparison between Z1 ([Cii]) and thez= 0 corpora
(HI 21 cm) requires caution; the tracers sample different
gas phases and spatial scales.
(8)M dyn< M ∗for CG32 and DC396844.Dynam-
ical mass below stellar mass is physically implausible
and likely reflects inclination uncertainty compounded
by the 2-ring boundary constraint. Values are reported
as published in Jones et al. (2021).
(9) HZ9 outer-ring dispersion anomaly.The outer-
most ring of HZ9 hasσ= 4.82 km s−1, anomalously
low relative to inner rings (σ∼71–75 km s−1), likely
a 3DBarolo fit artifact at the beam resolution limit.
(10) DC519281 redshift uncertainty.e z= 0.02, ap-
proximately 40×larger than the typical sample uncer-
tainty, reflecting genuine spectroscopic uncertainty in
this source.
8.DATA AVAILABILITY
The corpus is publicly available at Zenodo under
DOI: 10.5281/zenodo.20369285. The deposit includes
the master JSON, flat CSV, JSONL, per-galaxy ZIP
archive, three Jupyter notebooks with output figures,
and the ingestion and validation scripts. The corpus
schema, normalization, and unified structure are origi-
nal work by D.C. Flynn / EPS Research and are released
under CC BY 4.0. All underlying kinematic data are
drawn from published, publicly available sources; users
should cite both this corpus and the relevant primary
survey papers listed in the references.
This corpus is the fourth in the EPS Re-
search RAG Astrophysics Corpus Series. The
companion corpora are available at: Unified HI
Corpus v7.0 (Flynn 2026a), DOI: 10.5281/zen-
odo.19563417; Dwarf/Irregular Corpus v1.0 (Flynn
2026b), DOI: 10.5281/zenodo.20320362; Milky Way
GC Corpus v1.3.1 (Flynn 2026c), DOI: 10.5281/zen-
odo.19907765.This work was conducted as independent research by
EPS Research without external funding or institutional
affiliation. The author thanks Jim Cannaliato for col-
laboration on the omega correction framework. The
ALPINE survey team is thanked for making their kine-
matic catalog and data products publicly available. The
ALMA Observatory is operated by ESO, AUI/NRAO,
and NAOJ.1
2
3
4
5
6
7
8
DECLARATION OF GENERATIVE AI USE
In accordance with standard practice for AI-assisted
research, the author discloses the following. Four large
language models were used during the creation and val-
idation of this corpus and manuscript: Google Gemini
Pro, Anthropic Claude (Sonnet and Opus), Microsoft
Copilot Pro, and Gemma 4 31B Dense (self-hosted). Be-
cause the corpus is explicitly designed for LLM-based
RAG pipelines, these models served as both develop-
ment tools and validation instruments:
(1) Corpus schema design and ingestion code.LLMs
assisted in drafting and debugging the Python ingestion
scripts, JSON schema design, and flat CSV generation.
All code was reviewed, tested, and validated by the au-
thor against primary source data.
(2) Multi-model schema validation.As described
in Section 6, Gemini Pro, Claude, Copilot Pro, and
Gemma 4 31B Dense were each tested as downstream
consumers of the corpus JSON to verify that the schema
is sufficiently self-describing for LLM-based scientific
analysis without additional prompting.
(3) Cross-model review.Prior to Zenodo submis-
sion, five LLMs (Gemini Pro, Claude, Copilot Pro,
Gemma 4 31B, and AstroSage-70B) were asked to re-
view the corpus files and README for data anomalies
and documentation inconsistencies. This process identi-
fied the maximum redshift attribution error (DC417567
vs. DC773957) that was corrected prior to publication.
(4) Manuscript preparation.Claude (Anthropic) as-
sisted in drafting, formatting, and assembling this
manuscript. All scientific content, interpretations, data
provenance decisions, and editorial judgments are the
sole responsibility of the author.
No generative AI output was accepted without human
review. The author takes full responsibility for the con-
tent of this publication.
REFERENCES
B´ ethermin, M., Fudamoto, Y., Ginolfi, M., et al. 2020,
A&A, 643, A2.
https://doi.org/10.1051/0004-6361/202037649Di Teodoro, E.M., & Fraternali, F. 2015, MNRAS, 451,
3021. https://doi.org/10.1093/mnras/stv1213
Faisst, A.L., Schaerer, D., Lemaux, B.C., et al. 2020, ApJS,
247, 61. https://doi.org/10.3847/1538-4365/ab7a3c

8Flynn
Flynn, D.C., & Cannaliato, J. 2025, Frontiers in Astronomy
and Space Sciences, 12.
https://doi.org/10.3389/fspas.2025.1680387
Flynn, D.C. 2026a, Astronomy and Computing (submitted;
preprint arXiv:2604.13489). Zenodo.
https://doi.org/10.5281/zenodo.19563417
Flynn, D.C. 2026b, Dwarf/Irregular Galaxy HI Rotation
Curve Corpus v1.0. Zenodo.
https://doi.org/10.5281/zenodo.20320362
Flynn, D.C. 2026c, Milky Way Globular Cluster Corpus
v1.3.1. Zenodo. https://doi.org/10.5281/zenodo.19907765Jones, G.C., Vergani, D., Romano, M., et al. 2021, MNRAS,
507, 3540. https://doi.org/10.1093/mnras/stab2703
Le F` evre, O., B´ ethermin, M., Faisst, A., et al. 2020, A&A,
643, A1. https://doi.org/10.1051/0004-6361/201936965
Wisnioski, E., F¨ orster Schreiber, N.M., Wuyts, S., et al.
2015, ApJ, 799, 209.
https://doi.org/10.1088/0004-637X/799/2/209