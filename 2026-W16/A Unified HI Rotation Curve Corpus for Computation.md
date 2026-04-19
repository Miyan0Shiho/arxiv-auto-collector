# A Unified HI Rotation Curve Corpus for Computational Astrophysics: 438 Galaxies from SPARC, THINGS, LITTLE THINGS, and WALLABY DR2

**Authors**: David C. Flynn

**Published**: 2026-04-15 05:24:25

**PDF URL**: [https://arxiv.org/pdf/2604.13489v1](https://arxiv.org/pdf/2604.13489v1)

## Abstract
We present a unified corpus of 8,963 spatially resolved HI rotation curve measurements across 423 galaxies (438 total catalog entries including 15 metadata-only THINGS galaxies), drawn from four major surveys: SPARC (175), THINGS (34), LITTLE THINGS (26), and WALLABY DR2 (203). The corpus is distributed as a single structured JSON file with nested per-ring kinematic data, survey metadata, column definitions, and data-quality annotations, accompanied by a 438-row flat CSV for catalog-level filtering. All radii are in kiloparsecs, all velocities in km/s. Kinematic parameters have been verified against scanned primary tables. A two-tier quality system distinguishes hand-curated rotation curves with per-point uncertainties (Tier 1) from automated pipeline products (Tier 2). The corpus was designed for both traditional numerical analysis and Large Language Model retrieval-augmented generation (RAG) pipelines. Three worked examples demonstrate single-galaxy rotation curve plotting, multi-component baryonic analysis, and corpus-level parameter-space exploration, each requiring fewer than 15 lines of Python. The corpus is publicly available at Zenodo (DOI: 10.5281/zenodo.19563417) under CC BY 4.0.

## Full Text


<!-- PDF content starts -->

A Unified HI Rotation Curve Corpus for Computational
Astrophysics:
438 Galaxies from SPARC, THINGS, LITTLE THINGS,
and WALLABY DR2
David C. Flynna,∗
aEPS Research, Laurel MD, USA
Abstract
We present a unified corpus of 8,963 spatially resolved HI rotation curve mea-
surements across 423 galaxies (438 total catalog entries including 15 metadata-
onlyTHINGSgalaxies),drawnfromfourmajorsurveys: SPARC(175),THINGS
(34), LITTLE THINGS (26), and WALLABY DR2 (203). The corpus is dis-
tributed as a single structured JSON file with nested per-ring kinematic data,
survey metadata, column definitions, and data-quality annotations, accompa-
nied by a 438-row flat CSV for catalog-level filtering. All radii are in kiloparsecs,
all velocities in km/s. Kinematic parameters have been verified against scanned
primary tables. A two-tier quality system distinguishes hand-curated rotation
curves with per-point uncertainties (Tier 1) from automated pipeline products
(Tier 2). The corpus was designed for both traditional numerical analysis and
Large Language Model retrieval-augmented generation (RAG) pipelines. Three
workedexamplesdemonstratesingle-galaxyrotationcurveplotting,multi-component
baryonic analysis, and corpus-level parameter-space exploration, each requiring
fewer than 15 lines of Python. The corpus is publicly available at Zenodo (DOI:
10.5281/zenodo.19563417) under CC BY 4.0.
Keywords:galaxy rotation curves, HI kinematics, SPARC, THINGS, LITTLE
THINGS, WALLABY, data release, LLM, RAG
∗Corresponding author. ORCID: 0000-0002-2768-6650
Email address:davidflynn@eps-research.com(David C. Flynn)arXiv:2604.13489v1  [astro-ph.GA]  15 Apr 2026

1. Introduction
Galaxy rotation curves remain the primary observational evidence for the
mass discrepancy problem in disk galaxies. The gap between observed circular
velocities and those predicted from visible baryonic matter has motivated dark
matter halo models [1], modified gravity theories [2, 3], and empirical correction
frameworks [4]. All of these approaches require access to high-quality, spatially
resolved rotation curve data with consistent units and metadata.
Four major HI surveys currently provide the bulk of published rotation
curves for nearby galaxies. SPARC (Spitzer Photometry and Accurate Rota-
tion Curves [5]) offers 175 galaxies with full baryonic decomposition at 3.6µm.
THINGS (The HI Nearby Galaxy Survey [6, 7]) provides high-resolution VLA
tilted-ring fits for 19 galaxies. LITTLE THINGS [8] contributes 26 dwarf irreg-
ular galaxies observed at comparable VLA resolution. WALLABY DR2 (Wide-
field ASKAP L-band Legacy All-sky Blind Survey [9, 10, 11]) adds 203 galaxies
from the ASKAP automated pipeline, extending coverage to previously unchar-
acterized systems.
Despite the scientific importance of these datasets, no unified, machine-
readablecorpuscurrentlyexiststhatcombinesallfoursurveysinasingleschema.
While the individual survey products remain publicly available, they are dis-
tributed across heterogeneous platforms with incompatible formats, differing
column conventions (arcseconds vs. kiloparsecs, varying velocity definitions),
and no common schema—conditions that impose a significant barrier to repro-
duciblecomputationalanalysisandeffectivelyrenderthecombineddatasetinac-
cessible to automated pipelines. Researchers wishing to compare rotation curves
across surveys must individually download these disparate source files, reconcile
unit and naming conventions, and manually verify kinematic parameters against
scatteredprimarypublications. Thisdata-fragmentationproblemisparticularly
acute for computational workflows—including automated fitting pipelines, sta-
tisticalmeta-analyses, andemergingLLM-basedretrieval-augmentedgeneration
2

(RAG) architectures—that require structured, self-describing input data. The
problem is likely to intensify as earlier-generation HI surveys hosted on aging or
deprecated infrastructure become candidates for similar standardization efforts.
This paper presents theUnified Galaxy HI Rotation Curve Corpus(v7.0), a
structured JSON dataset containing 8,963 individually resolved rotation curve
measurements across 423 galaxies from all four surveys, with an additional 15
THINGS galaxies contributing verified kinematic metadata. We describe the
corpus architecture (Section 3), present the catalog-level CSV schema (Sec-
tion 4), document the ingestion and verification procedures (Section 5), demon-
strate three concrete usage examples with embedded figures (Section 6), dis-
cuss the LLM/RAG application (Section 7), and enumerate known limitations
(Section 8). The corpus is publicly available at Zenodo (DOI: 10.5281/zen-
odo.19563417) under CC BY 4.0.
2. Survey Coverage and Source Data
Table 1 summarizes the survey coverage. The corpus contains rotation curve
data for 423 galaxies and kinematic metadata for a further 15, totaling 438
catalog entries and 8,963 radial measurement points.
Table 1: Survey coverage summary.
Survey Galaxies Data Points Tier Primary Reference
SPARC 175 3,391 1 Lelli et al. (2016) [5]
THINGS 34 (19 w/data) 2,110 1 de Blok et al. (2008) [7]
LITTLE THINGS 26 1,716 1 Oh et al. (2015) [8]
WALLABY DR2 203 1,746 2 Deg et al. (2022) [10];
Murugeshan et al. (2024) [11]
Total 438 8,963
3

2.1. SPARC
The SPARC database [5] provides 175 galaxies with HI/Hαrotation curves
and full baryonic decomposition at Spitzer 3.6µm. Each galaxy carries observed
velocity (V obs), gas velocity (V gas), disk velocity (V disk), bulge velocity (V bul),
and surface brightness profiles at each radial point, enabling mass modeling
without external photometry. Kinematic parameters (inclination, distance, and
their uncertainties) were verified against Lelli et al. [5] Table 1 (Lelli2016c.mrt;
scanned). SPARC coordinates (RA, Dec), systemic velocities, and position an-
gles are absent from the corpus because they were not published in a unified
SPARC table and remain distributed across approximately 50 individual source
papers.
2.2. THINGS
The THINGS survey [6] observed 34 nearby galaxies at high spectral and
spatial resolution with the VLA. De Blok et al. [7] published tilted-ring rota-
tion curves for 19 of these galaxies. The remaining 15 were excluded from the
kinematic analysis due to morphological disturbance, low inclination, strong
non-circular motions, or other factors. These 15 galaxies are included in the cor-
pus with verified metadata (distance, inclination, position angle, coordinates)
but without per-point rotation curve data. Fourteen also appear in SPARC
and carry full baryonic decomposition under their SPARC entries. Kinematic
parameters were verified against de Blok et al. [7] Tables 1 and 2 (scanned).
2.3. LITTLE THINGS
LITTLE THINGS (Local Irregulars That Trace Luminosity Extremes, The
HI Nearby Galaxy Survey [8]) contributes 26 dwarf irregular galaxies observed
with the VLA, providing rotation velocities (V rot) with per-point uncertainties.
All kinematic parameters were verified against Oh et al. [8] Table 1 (scanned).
The characteristic radiusr 0.3and corresponding velocityv 0.3(the radius at
which the logarithmic slope of the rotation curve equals 0.3) are included in the
catalog CSV.
4

2.4. WALLABY DR2
TheWALLABYDR2kinematiccatalogue[10,11]providesspatiallyresolved
rotation curves for 203 galaxies processed through the WALLABY Kinematic
AnalysisPipelineProducts(WKAPP)automatedpipeline: the3DBarolotilted-
ring fitter [12] combined with the Fully Automated TiRiFiC (FAT) pipeline [13].
The ASKAP synthesised beam is 30 arcseconds (FWHM), setting a practical
spatial resolution floor.V rotvalues below 50 km/s are subject to beam-smearing
and should be treated with caution. Distances are derived from Hubble flow at
H0= 75 km s−1Mpc−1unless Cosmicflows-4 distances are available. Per-ring
rotation velocity uncertainties are not published in the DR2 catalogue; WAL-
LABY entries therefore carryquality_tier= 2. Of 203 WALLABY galaxies,
202 carry quality flag 0 (reliable model fit) and 1 carries flag 1 (marginal). Ring
counts range from 3 to 47 (median 7). Per-ring data include radius (kpc),V rot
(km/s), velocity dispersion (km/s), inclination (degrees), and position angle
(degrees).
3. Corpus Architecture and Schema
3.1. File formats
The corpus is distributed in three complementary formats. The master file
rotation_curve_corpus_v7.jsonis a single JSON document (∼2.0 MB) contain-
ingall438galaxyentrieswithinaunifiedschema,togetherwithatop-levelmeta-
data block encoding version, survey counts, quality tier definitions, and citation.
The flat tablerotation_curve_corpus_v7_flat.csvprovides one row per galaxy
(438 rows, 29 columns) with summary statistics for sample selection and filter-
ing. Theper-galaxyarchiverotation_curve_corpus_v7_by_galaxy.zipcontains
438 individual JSON files organised into subdirectories by survey (SPARC/,
THINGS/, LITTLE_THINGS/, WALLABY/), each self-contained with full
corpusmetadataandthecompleterotationcurvearray,optimisedforLLM/RAG
ingestion where each galaxy constitutes a single retrieval document.
5

3.2. Quality tier system
A two-tier quality annotation is applied at the galaxy level. Tier 1 (SPARC,
THINGS, LITTLE THINGS) denotes hand-curated rotation curves with per-
point uncertainties, verified kinematic parameters, and—for SPARC—full bary-
onicdecomposition. Tier2(WALLABYDR2)denotesautomatedpipelineprod-
uctsfromtheWKAPPsystem, peer-reviewedbutwithoutper-ringuncertainties
or baryonic components. The tier system enables downstream analyses to filter
by data provenance without inspecting individual galaxies.
3.3. Units and conventions
All radii are stored in kiloparsecs. THINGS radii were converted from arc-
seconds usingR[kpc] =R[arcsec]×D[Mpc]×1000×π/648000. All velocities are
in km/s. SPARC baryonic velocity components (V gas,Vdisk,Vbul) are stored at
mass-to-light ratioΥ = 1. The gas velocityV gasmay be negative at inner radii
due to the sign-preserving quadrature convention used throughout SPARC, in
which thermal pressure exceeds rotational support. The total baryonic velocity
is computed via:
Vbar=q
Υ⋆V2
disk+ ΥbV2
bul+ sign(V gas)V2gas (1)
Thevgas_negative_rowscolumn in the CSV records the count of inner-disk
radii whereV gas<0for each SPARC galaxy. All tables presented in this
manuscriptareroundedtoinstrument-appropriateprecision; themachine-readable
JSON and CSV files retain full floating-point values to preserve round-trip fi-
delity for programmatic use (see Section 8, Limitation 5).
3.4. JSON schema by survey
Because the four surveys provide fundamentally different observables, the
per-galaxy JSON schema varies by survey. Table 2 summarizes the per-ring
columns available for each.
Code accessing per-point data across surveys should check for both keys:
6

Table 2: Per-ring data columns by survey. SPARC/THINGS/LITTLE THINGS use thedata
key; WALLABY usesrotation_curve. Note that within thedatakey, SPARC and LITTLE
THINGS useVobs/errVwhile THINGS usesVrot/e_Vrot.
Survey Data Key Per-Ring Columns
SPARC (175)dataRad, Vobs, errV, Vgas, Vdisk, Vbul, SBdisk, SBbul
THINGS (19)dataRad, Vrot, e_Vrot
LITTLE THINGS (26)dataRad, Vobs, errV
WALLABY (203)rotation_curverad_kpc, vrot_kms, vdisp_kms, inc_deg, pa_deg
points = galaxy.get(’data’) or galaxy.get(’rotation_curve’,
[])
4. Catalog-Level CSV Schema
The flat CSV contains 29 columns per galaxy. Table 3 lists each field with
its survey coverage. The CSV is derived from the JSON and is designed for
rapid sample selection; per-point data require loading the JSON.
5. Data Ingestion and Verification
5.1. SPARC, THINGS, and LITTLE THINGS
SPARC rotation curves were ingested from the publicly available flat files
at astroweb.cwru.edu/SPARC. THINGS rotation curves were digitised from de
Blok et al. [7] Table 2; radii were converted from arcseconds to kiloparsecs using
the published distances. LITTLE THINGS data were obtained from VizieR
(J/AJ/149/180). All kinematic parameters were cross-verified against scanned
primarytables: Lellietal.[5]Table1forSPARC,deBloketal.[7]Tables1and2
for THINGS, and Oh et al. [8] Table 1 for LITTLE THINGS. This verification
step caught an arcsec-to-kpc conversion bug (a missing×1000factor) in an
earlier ingestion iteration, underscoring the value of systematic primary-source
checking.
7

Table 3: Flat CSV schema (29 columns). LT = LITTLE THINGS; W = WALLABY.
Field Surveys Coverage Description
galaxyAll 438/438 Galaxy identifier
surveyAll 438/438 SPARC / THINGS / LITTLE_THINGS / WALLABY
quality_tierAll 438/438 1 = hand-curated, 2 = automated
telescopeTHINGS,LT,W 263/438 Instrument identifier
ra_degTHINGS,LT,W 263/438 Right ascension (J2000, deg)
dec_degTHINGS,LT,W 263/438 Declination (J2000, deg)
distance_mpcAll 438/438 Distance (Mpc)
e_distance_mpcSPARC 175/438 Distance uncertainty (Mpc)
vsys_kmsTHINGS,LT,W 263/438 Systemic velocity (km/s)
inc_degAll 438/438 Inclination (deg); verified
e_inc_degSPARC 175/438 Inclination uncertainty (deg)
pa_degTHINGS,LT,W 261/438 Position angle (deg)
hubble_typeSPARC 175/438 Hubble type integer (0–11)
m2l_diskSPARC 175/438 Mass-to-light ratio at [3.6]
n_pointsAll 423/438 Rotation curve point count
r_min_kpcAll 423/438 Innermost radius (kpc)
r_max_kpcAll 423/438 Outermost radius (kpc)
vrot_min_kmsAll 423/438 Min rotation velocity (km/s)
vrot_mean_kmsAll 423/438 Mean rotation velocity (km/s)
vrot_max_kmsAll 423/438 Peak rotation velocity (km/s)
vdisp_mean_kmsWALLABY 203/438 Mean velocity dispersion (km/s)
has_bulgeSPARC 201/438 Boolean bulge flag
vgas_negative_rowsSPARC 175/438 Inner radii withV gas<0
r0p3_kpcLT 26/438 Radius at log slope = 0.3 (kpc)
v0p3_kmsLT 26/438 Velocity atr 0.3(km/s)
beam_arcsecWALLABY 203/438 Beam FWHM (arcsec)
qflag_modelWALLABY 203/438 3DBarolo quality flag
referenceAll 438/438 Primary citation
notesAll 41/438 Data quality notes
8

5.2. WALLABY DR2
WALLABYrotationcurveswereingestedfromthepublished3DBarolomodel
files available at the Canadian Astronomy Data Centre (CADC). A Python
script(wallaby_ingest.py, includedassupplementarymaterial)readstheCADC
per-galaxy model output, extracts per-ring kinematic parameters, and cross-
matches against the WALLABY DR2 kinematic catalogue (303 entries) to at-
tachgalaxy-levelmetadata(coordinates, distance, systemicvelocity, inclination,
position angle). Of 303 catalogue entries, 204 had Barolo model files at CADC;
after quality screening, 203 were ingested. Ring counts range from 3 to 47
(median 7), with a total of 1,746 radial measurement points.
5.3. Cross-matching
Fourteen galaxies appear in two surveys (primarily SPARC and THINGS).
The crossmatch index is stored in the JSON metadata block and enables down-
stream analyses to identify duplicate entries, apply survey-specific treatments,
ormergecomplementarydata(e.g.,SPARCbaryonicdecompositionwithTHINGS
tilted-ring kinematics for the same galaxy).
6. Usage Examples
The following three examples demonstrate the corpus’s utility for common
rotation curve analyses. Each example loads data directly from the JSON with
no external preprocessing. All code is Python 3 using only the standard library,
numpy, and matplotlib. The figure-generation script (make_figures_v7.py) is
included as supplementary material.
6.1. Example 1: Multi-component baryonic analysis (SPARC)
Figure1showsDDO161(aSPARCTier1dwarfirregulargalaxyat7.4Mpc)
with four curves extracted directly from the corpus JSON:V obswith SPARC
error bars (blue circles),V barfrom sign-preserving quadrature atΥ = 1(red
squares), the omega-corrected velocityV obs−Rω(green triangles, applying the
9

empirical correctionV obs=VKepler +Rω, whereωis a per-galaxy angular veloc-
ity offset [4], hereω= 4.69rad/Gyr; note that 1 rad/Gyr≈1.022 km/s/kpc,
soωin these units yieldsVin km/s when multiplied byRin kpc), and the
expected Keplerian baseline (orange dashed). The gap between blue and red is
the mass discrepancy that dark matter, MOND, or the omega correction each
attempt to explain. The point for this data descriptor is that all four curves
come from a single JSON load with no preprocessing.
import json, numpy as np, matplotlib.pyplot as plt
with open(’rotation_curve_corpus_v7.json’) as f:
corpus = json.load(f)
g = next(g for g in corpus[’galaxies’] if g[’galaxy’]==’
DDO161’)
d = g[’data’]
R = np.array([p[’Rad’] for p in d])
Vobs = np.array([p[’Vobs’] for p in d])
errV = np.array([p[’errV’] for p in d])
Vgas = np.array([p[’Vgas’] for p in d])
Vdisk= np.array([p[’Vdisk’] for p in d])
Vbul = np.array([p[’Vbul’] for p in d])
# Baryonic velocity: sign-preserving quadrature
Vbar = np.where(Vgas<0, -np.sqrt(Vgas**2+Vdisk**2+Vbul**2),
np.sqrt(Vgas**2+Vdisk**2+Vbul**2))
omega = 4.69 # rad/Gyr, [4] Table 2
V_omega = Vobs - R * omega
6.2. Example 2: WALLABY rotation curve with caution zone (Tier 2)
Figure2demonstratestheTier2pipelineoutputforWALLABYJ165901−601241,
a galaxy with 37 rings and a clean rising curve to∼167 km/s. All points lie well
above the 50 km/s beam-smearing caution zone (shaded). The metadata anno-
tation (distance, inclination, ring count) is extracted directly from the JSON,
10

Figure 1: DDO 161 (SPARC Tier 1) loaded from corpus JSON. Blue circles:V obswith SPARC
error bars. Red squares:V barfrom sign-preserving quadrature atΥ = 1. Green triangles:
Vobs−Rω(ω= 4.69rad/Gyr [4]). Orange dashed: expected Keplerian baseline.
showing the corpus carries everything needed for quality assessment.
wg = next(g for g in corpus[’galaxies’]
if g.get(’galaxy’)==’WALLABY_J165901 -601241’)
rc = wg[’rotation_curve’]
R_w = [p[’rad_kpc’] for p in rc]
V_w = [p[’vrot_kms’] for p in rc]
plt.plot(R_w, V_w, ’o-’, color=’#B2182B’)
plt.axhspan(0, 50, alpha=0.15, color=’red’,
label=’Vrot␣<␣50␣km/s␣(caution)’)
plt.text(0.03, 0.95, f"D={wg[’distance_mpc’]:.1f}␣Mpc\n"
f"inc={wg[’inc_deg’]:.1f}␣deg\n{len(rc)}␣rings",
transform=plt.gca().transAxes, va=’top’)
11

Figure 2: WALLABY J165901−601241 (Tier 2) loaded from corpus JSON. The 50 km/s
beam-smearing caution zone is shaded. Metadata (D= 15.2Mpc, inc= 50.7◦, 37 rings) is
extracted directly from the JSON.
6.3. Example 3: Corpus-level parameter-space exploration
Figure 3 provides the corpus-level overview. Panel (a) shows the peak rota-
tion velocity distribution across all four surveys as a stacked histogram: SPARC
andWALLABYdominatedifferentvelocityranges, whileTHINGSandLITTLE
THINGS fill intermediate coverage. Panel (b) shows theR maxvs.V rot,maxpa-
rameter space colored by quality tier, demonstrating that the corpus spans from
∼3 kpc dwarf irregulars to∼100 kpc massive spirals.
# Panel (a): stacked histogram by survey
for survey in [’SPARC’,’THINGS’,’LITTLE_THINGS’,’WALLABY’]:
vals = [max(p.get(’Vobs’,p.get(’vrot_kms’,0))
for p in g.get(’data’) or g.get(’rotation_curve’
,[]))
for g in corpus[’galaxies’] if g[’survey’]==
survey
and (g.get(’data’) or g.get(’rotation_curve’))]
ax[0].hist(vals, bins=36, range=(0,350),
label=f’{survey}␣({len(vals)})’, alpha=0.8)
12

# Panel (b): scatter by tier
ax[1].scatter(r_max[tier==1], v_max[tier==1], label=’Tier␣1’
)
ax[1].scatter(r_max[tier==2], v_max[tier==2], label=’Tier␣2’
)
Figure 3: Corpus population overview. (a) Peak rotation velocity distribution across all four
surveys (stacked histogram,N= 423galaxies with rotation curve data). (b)R maxvs.V rot,max
parameter space colored by quality tier (Tier 1: 220 galaxies, black; Tier 2: 203 galaxies, red).
The corpus spans from∼3 kpc dwarf irregulars to∼100 kpc massive spirals.
7. Application to LLM-Based Inference
A secondary design goal of the corpus is to serve as a retrieval corpus for
LLM-based RAG pipelines in astrophysical research. In a RAG architecture, a
user query (e.g., “plot the rotation curve of DDO 161 and compute its baryonic
mass model”) triggers a retrieval step that locates the relevant galaxy document,
which is then injected into the LLM’s context window alongside the query. The
per-galaxy ZIP archive described in Section 3.1 is optimised for this use case:
each file is a self-contained JSON document of∼1–5 KB, well within typical
context limits, containing all metadata and per-ring data needed to answer the
query without external lookups.
To assess whether the JSON schema is sufficiently self-describing for auto-
mated consumption, we conducted a structured usability evaluation with four
13

LLMs: Google Gemini Pro, Anthropic Claude, AstroSage-Llama-3.1-70B (a
domain-specific astronomy foundation model), and Microsoft Copilot Pro. Each
model was presented with a single per-galaxy JSON document and asked to
perform three benchmark tasks without additional documentation: (1) plot
the rotation curve with error bars, (2) compute the baryonic velocity via sign-
preserving quadrature, and (3) apply the omega correction from [4]. All four
models successfully generated syntactically correct Python scripts for all three
tasks on first attempt, requiring no additional prompting beyond a natural-
language research question. Gemini Pro and Claude produced the most fluent
end-to-end workflows, generating matplotlib figures directly within their native
interfaces. Copilot Pro generated correct Python but required script execu-
tion outside the chat environment to render figures. AstroSage-70B, tested
via LM Studio, proved effective at identifying corpus-level defects and schema
inconsistencies but was more sensitive to the host platform’s capabilities; RAG-
optimised deployment environments may yield stronger results. In all cases, the
most reliable workflow was to request Python code generation and execute the
resulting scripts in a Jupyter notebook, which enabled the full range of rota-
tion curve analysis, baryonic decomposition, and omega correction application.
These results suggest that the corpus’s explicit column definitions, unit annota-
tions, and quality flags provide sufficient context for LLM-based code generation
and analysis without external documentation, and that the per-galaxy JSON
format integrates naturally into interactive computational environments.
8. Known Limitations
Five limitations should be noted by users of this corpus.
(1) Fifteen THINGS galaxies lack per-point rotation curve data.
As noted in Section 2.2, these galaxies were excluded from the de Blok et al. [7]
tilted-ring analysis for legitimate astrophysical reasons and are marked with
n_points = nullin the CSV.
(2)SPARCentrieslackcoordinatesandsystemicvelocities.SPARC
14

RA, Dec,V sys, and PA are absent because Lelli et al. [5] did not publish these
in a unified table. These parameters are recoverable from NED or SIMBAD for
individual galaxies; they were not included in the corpus to avoid introducing a
secondary provenance layer beyond the original survey publications.
(3)WALLABYrotationcurvescarrynoper-ringuncertaintiesand
no baryonic decomposition.V rotbelow 50 km/s is unreliable due to beam
smearing at 30 arcsec ASKAP resolution.
(4) JSON schema is not fully harmonised across surveys.SPAR-
C/THINGS/LITTLETHINGSuseadatakey; WALLABYusesrotation_curve
with different field names. This reflects genuine differences in the underlying
observables.
(5) Decimal precision may exceed measurement accuracy in some
derived columns.Floating-point artifacts from unit conversions (e.g., arcsec
to kpc) are retained in the JSON to preserve round-trip fidelity. Radii are stored
to three decimal places (∼1 pc); velocities to two decimal places (∼0.01 km/s).
Users applying uncertainty-propagation analysis should round to instrumental
precision before reporting results. Typical instrumental uncertainties are 1–
5 km/s for rotation velocities and∼0.1 kpc for radii at the distances of these
surveys.
9. Data Availability
ThecorpusispubliclyavailableatZenodounderversion-specificDOI:10.5281/zen-
odo.19563417(concept DOI for all versions: 10.5281/zenodo.19425427). The
deposited files are named v7.0 to reflect internal development versioning; this
Zenodo record represents the first peer-reviewed public release of the unified
corpus. The deposit includes the master JSON, flat CSV, per-galaxy ZIP
archive, corpus description sheet, and the WALLABY ingestion script. The
corpus schema, normalisation, annotations, and unified structure are original
work by D.C. Flynn / EPS Research and are released under CC BY 4.0. All
underlying rotation curve data are drawn from published, publicly available
15

sources; users should cite both this corpus and the relevant survey papers listed
in the references.
Acknowledgements
This work was conducted as independent research by EPS Research without
external funding or institutional affiliation. The author thanks Jim Cannaliato
for collaboration on the omega correction framework. The SPARC database
is maintained by Federico Lelli and Stacy McGaugh at Case Western Reserve
University. THINGS data products are based on observations with the Karl G.
Jansky Very Large Array of the National Radio Astronomy Observatory. WAL-
LABY data products are based on observations with the Australian Square
Kilometre Array Pathfinder (ASKAP) and were accessed via the Canadian As-
tronomy Data Centre (CADC).
Declaration of Generative AI Use
In accordance with Elsevier’s policy on the use of generative AI and AI-
assisted technologies, the author discloses the following. Four large language
modelswereusedduringthecreationandvalidationofthiscorpusandmanuscript:
Google Gemini Pro, Anthropic Claude (Opus and Sonnet), AstroSage-Llama-
3.1-70B, and Microsoft Copilot Pro. Because the corpus is explicitly designed
for LLM-based retrieval-augmented generation (RAG) pipelines, these models
served as both development tools and validation instruments:
(1)Corpus schema design and ingestion code.LLMs assisted in drafting
and debugging the Python ingestion scripts (including wallaby_ingest.py), the
JSON schema design, and the flat CSV generation code. All code was reviewed,
tested, and validated by the author against primary source data.
(2)Data verification and error detection.LLMs were used to cross-check
ingested values against scanned primary tables [5, 7, 8], which led to the detec-
tion and correction of an arcsec-to-kpc conversion bug in an earlier ingestion
iteration.
16

(3)RAG validation testing.As described in Section 7, Gemini Pro, Claude,
AstroSage-70B, and Copilot Pro were each tested as downstream consumers of
thecorpusJSONtoverifythattheschemaissufficientlyself-describingforLLM-
basedscientificanalysis—includingrotationcurveplotting,sign-preservingbary-
onicquadrature, andomegacorrectionapplication—withoutadditionalprompt-
ing beyond natural-language research questions.
(4)Manuscript preparation.Claude (Anthropic) assisted in drafting, for-
matting, and assembling this manuscript, including figure generation code and
document production. All scientific content, interpretations, data provenance
decisions, and editorial judgments are the sole responsibility of the author.
No generative AI output was accepted without human review. The author
takes full responsibility for the content of this publication.
References
[1] J.F. Navarro, C.S. Frenk, S.D.M. White, The structure of cold dark matter
halos, ApJ 462 (1996) 563.
[2] M. Milgrom, A modification of the Newtonian dynamics as a possible al-
ternative to the hidden mass hypothesis, ApJ 270 (1983) 365.
[3] S.S.McGaugh, Themassdiscrepancy–accelerationrelation, ApJ609(2004)
652.
[4] D.C. Flynn, J. Cannaliato, A new empirical fit to galaxy rota-
tion curves, Frontiers in Astronomy and Space Sciences 12 (2025).
doi:10.3389/fspas.2025.1680387.
[5] F. Lelli, S.S. McGaugh, J.M. Schombert, SPARC: Mass models for 175
disk galaxies with Spitzer photometry and accurate rotation curves, AJ
152 (2016) 157.
[6] F. Walter, E. Brinks, W.J.G. de Blok, et al., THINGS: The HI Nearby
Galaxy Survey, AJ 136 (2008) 2563.
17

[7] W.J.G.deBlok, F.Walter, E.Brinks, etal., High-resolutionrotationcurves
and galaxy mass models from THINGS, AJ 136 (2008) 2648.
[8] S.-H. Oh, D.A. Hunter, E. Brinks, et al., High-resolution mass models of
dwarf galaxies from LITTLE THINGS, AJ 149 (2015) 180.
[9] T. Westmeier, N. Deg, K. Spekkens, et al., WALLABY: an SKA Pathfinder
HI survey, PASA 39 (2022) e058.
[10] N. Deg, K. Spekkens, T. Westmeier, et al., WALLABY kinematic mod-
elling, PASA 39 (2022) e059.
[11] C. Murugeshan, N. Deg, T. Westmeier, et al., WALLABY pilot survey:
public data release of∼1800 HI sources and high-resolution cut-outs from
Pilot Survey Phase 2, PASA 41 (2024) e088.
[12] E.M. Di Teodoro, F. Fraternali, 3D Barolo: a new 3D algorithm to derive
rotation curves of galaxies, MNRAS 451 (2015) 3021.
[13] P. Kamphuis, G.I.G. Józsa, S.-H. Oh, et al., Automated kinematic mod-
elling of warped galaxies, MNRAS 452 (2015) 3139.
18