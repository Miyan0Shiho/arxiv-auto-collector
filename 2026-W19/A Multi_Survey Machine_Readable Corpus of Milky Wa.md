# A Multi-Survey Machine-Readable Corpus of Milky Way Globular Cluster Parameters for Retrieval-Augmented Generation Applications

**Authors**: David C. Flynn

**Published**: 2026-05-04 19:29:15

**PDF URL**: [https://arxiv.org/pdf/2605.03099v1](https://arxiv.org/pdf/2605.03099v1)

## Abstract
We present the Milky Way Globular Cluster Corpus v1.3.1, a unified machinereadable database of fundamental parameters for 174 Milky Way globular clusters assembled from four independent published surveys. Each cluster record integrates photometric and structural parameters from Harris [1996] (2010 revision), Gaia EDR3 proper motions from Vasiliev and Baumgardt [2021], N-body dynamical masses and orbital parameters from Baumgardt et al. [2023],, and mean chemical abundances from the APOGEE DR17 globular cluster Value Added Catalog of Schiavon et al. [2024]. The corpus contains 17,438 non-null data points across 174 clusters stored in JSONL, JSON, and flat CSV formats with consistent native-typed fields (float, int, bool, null), embedded provenance blocks, and fully documented schema. Survey coverage is 157/174 clusters for Harris photometry, 170/174 for Gaia EDR3 proper motions, 154/174 for Baumgardt N-body dynamics, and 72/174 for APOGEE DR17 chemistry. The corpus was designed as a Retrieval-Augmented Generation (RAG) knowledge base for large language model applications in astrophysics research, following the same multi-survey integration methodology as the Unified Galaxy HI Rotation Curve Corpus [Flynn, 2026b], and has been validated for structured context injection with instruction-following language models. It is equally suitable for traditional quantitative analyses including orbit modeling, cluster classification, chemical tagging, and multi-survey cross-validation.
  The dataset is available at Zenodo DOI: 10.5281/zenodo.19907766

## Full Text


<!-- PDF content starts -->

Submitted to PASP Flynn (2026) — MW GC Corpus v1.3.1
A Multi-Survey Machine-Readable Corpus
of Milky Way
Globular Cluster Parameters for
Retrieval-Augmented
Generation Applications
David C. Flynn
EPS Research, Laurel, MD 20707, USA
ORCID: 0000-0002-2768-6650
davidflynn@eps-research.com
Submitted toPublications of the Astronomical Society of the Pacific
Abstract
We present the Milky Way Globular Cluster Corpus v1.3.1, a unified machine-
readable database of fundamental parameters for 174 Milky Way globular clusters
assembled from four independent published surveys. Each cluster record integrates
photometric and structural parameters from Harris [1996] (2010 revision), Gaia
EDR3 proper motions from Vasiliev and Baumgardt [2021], N-body dynamical
masses and orbital parameters from Baumgardt et al. [2023], and mean chemi-
cal abundances from the APOGEE DR17 globular cluster Value Added Catalog of
Schiavon et al. [2024]. The corpus contains 17,438 non-null data points across 174
clusters stored in JSONL, JSON, and flat CSV formats with consistent native-typed
fields (float,int,bool,null), embedded provenance blocks, and fully documented
schema. Surveycoverageis157/174clustersforHarrisphotometry,170/174forGaia
EDR3 proper motions, 154/174 for Baumgardt N-body dynamics, and 72/174 for
APOGEE DR17 chemistry. The corpus was designed as a Retrieval-Augmented
Generation (RAG) knowledge base for large language model applications in astro-
physics research, following the same multi-survey integration methodology as the
Unified Galaxy HI Rotation Curve Corpus [Flynn, 2026b], and has been validated
for structured context injection with instruction-following language models. It is
equally suitable for traditional quantitative analyses including orbit modeling, clus-
ter classification, chemical tagging, and multi-survey cross-validation. The dataset
is available at Zenodo DOI: 10.5281/zenodo.19907766.
1arXiv:2605.03099v1  [astro-ph.GA]  4 May 2026

Submitted to PASP Flynn (2026) — MW GC Corpus v1.3.1
Keywords:globular clusters — Milky Way — stellar kinematics — chemical abundances
—N-bodymodels—propermotions—GaiaEDR3—APOGEE—catalogs—databases
— methods: data analysis
1 Introduction
The Milky Way globular cluster system is one of the most thoroughly studied populations
in astrophysics. These old, dense stellar systems serve as tracers of Galactic structure
and chemical evolution, probes of dark matter distribution, and benchmarks for stellar
population models. Decades of multi-wavelength observational campaigns have produced
a rich but fragmented literature: photometric catalogs, N-body dynamical models, Gaia
astrometry, and high-resolution spectroscopic surveys exist as independent publications,
each with its own format, naming convention, and coverage.
Theriseoflargelanguagemodels(LLMs)andRetrieval-AugmentedGeneration(RAG)
architectures in scientific research creates a new demand: machine-readable corpora that
are not merely tabular databases but structured knowledge representations accessible to
both programmatic analysis and natural language inference pipelines. A RAG corpus
must be consistently typed, null-safe, self-describing, and organized so that each record is
a semantically coherent unit — a single cluster carrying all available information about
that object from all available sources.
This paper describes the construction and content of the Milky Way Globular Cluster
Corpus v1.3.1, which satisfies these requirements. The corpus follows the design principles
established in the Unified Galaxy HI Rotation Curve Corpus [Flynn, 2026b], extending
the multi-survey integration approach from galaxy kinematics to globular cluster physics.
Thefoursourcesurveyswerechosentoprovideorthogonalphysicalinformation: Harris
[1996] for photometry and classical structure; Vasiliev and Baumgardt [2021] for space
astrometry; Baumgardt et al. [2023] for dynamical modeling; and Schiavon et al. [2024]
for high-resolution spectroscopic chemistry. No single survey covers all 174 clusters in
the corpus; coverage fractions range from 41% (APOGEE DR17) to 98% (Vasiliev &
Baumgardt 2021), with the Harris catalog forming the 157-cluster backbone.
2 Source Surveys
2.1 Harris (1996, 2010 Edition)
The Harris catalog [Harris, 1996], revised 2010, is the standard reference compilation
for Milky Way GC parameters, providing photometric and structural data for 157 clus-
ters. Parameters include: positions in equatorial and Galactic coordinates; heliocentric
distanceR ⊙and Galactocentric distanceR GC; GalactocentricX/Y/Zcoordinates; red-
2

Submitted to PASP Flynn (2026) — MW GC Corpus v1.3.1
deningE(B−V); metallicity[Fe/H]; apparent and absoluteVmagnitudes; distance
modulus; horizontal branch magnitudeV HB; color indicesU−B,B−V,V−R,V−I;
ellipticity; spectraltype; Kingconcentrationparameterc[King,1966]; core-collapseclassi-
fication; core and half-light radii in arcmin and kpc; central surface brightnessµ V; central
logarithmic densitylogρ 0; core and half-mass relaxation times; mean radial velocity and
velocity dispersion.
2.2 Vasiliev & Baumgardt (2021)
Vasiliev and Baumgardt [2021] published a comprehensive Gaia EDR3 astrometric analy-
sis of 170 Milky Way GCs. Their mixture model analysis yields mean proper motionsµ α∗
andµδwith errors and correlation coefficient, mean Gaia parallax with Lindegren et al.
[2021] zero-point correction applied per-star, a Plummer scale radius, and the number of
member stars with good astrometry.
This survey provides 17 clusters not present in the Harris 2010 catalog, representing
systems discovered after the Harris compilation. Four clusters (2MS-GC01, 2MS-GC02,
GLIMPSE01, GLIMPSE02) lie behind extreme dust columns (E(B−V)ranging from
5.2 to 34.5) that Gaia optical astrometry cannot penetrate; theirgaia_edr3blocks are
null throughout.
2.3 Baumgardt et al. (2023), v4 N-body Database
The Baumgardt et al. N-body database [v4, March 2023; Baumgardt and Hilker, 2018,
Baumgardt and Vasiliev, 2021, Baumgardt et al., 2023] provides parameters derived by
fitting multimass N-body models to compilations of ground-based radial velocities, Gaia
DR3 proper motions, and HST-based stellar mass functions. The database covers 154
clusters in the current corpus.
Two sub-tables were ingested. The orbits table provides precise distances with er-
rors, mean radial velocity, Gaia DR3 proper motions, GalactocentricX/Y/Zpositions
andU/V/Wspace velocities with errors, and orbital pericenterr periand apocenterr apo
computed in the Irrgang et al. [2013] Galactic potential. The structural parameters table
provides total dynamical mass with uncertainty,V-band magnitude andM/Lratio, core
radiusrc, projected half-light radiusr hl, 3D half-mass radiusr hm, and tidal radiusr t— all
in parsecs — central and half-mass logarithmic densities, half-mass relaxation time, initial
mass, dissolution timescale, global IMF slopeα(Salpeter=−2.3), central 1D velocity
dispersionσ 0, central escape velocity, mass segregation parametersη candηh[Trenti and
van der Marel, 2013], and rotation amplitudeA rotwith detection probabilityP rot[Sollima
et al., 2019].
Five clusters present in the Baumgardt database but not in Harris or Vasiliev are
appended: Gran 2, Gran 3, Gran 5, Patchick 126, and VVV-CL160.
3

Submitted to PASP Flynn (2026) — MW GC Corpus v1.3.1
2.4 Schiavon et al. (2024), APOGEE DR17 GC Value Added
Catalog
Schiavon et al. [2024] constructed the APOGEE DR17 GC Value Added Catalog by
identifying member stars in 72 Milky Way GCs from the APOGEE DR17 allStar catalog
[Abdurro’uf et al., 2022], yielding 6,422 member star entries. APOGEE acquires high-
resolution (R∼22,500)H-band spectra using the 300-fiber APOGEE spectrographs.
StellarparametersandabundanceswerederivedusingtheASPCAPpipeline[GarcíaPérez
et al., 2016] with the DR17synspec_rev1synthetic grid incorporating NLTE treatment
for Na, Mg, K, and Ca.
From this VAC we ingest cluster-level mean values: mean[Fe/H], mean radial velocity
with error, heliocentric and Galactocentric distances from Baumgardt and Vasiliev [2021],
cluster mass in units of104M⊙, Jacobi radius in degrees, and total number of member
star entries in the VAC. The 72-cluster coverage (41%) reflects APOGEE fiber allocation
priorities andH-band extinction limits.
3 Data Integration Methodology
3.1 Cluster Identification and Name Resolution
Primary cluster identifiers follow the Harris 2010 naming convention where available.
Baumgardt and Vasiliev catalog names were mapped to Harris primary identifiers through
an explicit lookup table, verified against the SIMBAD astronomical database.
3.2 Data Type Standardization
All numeric values are stored as native Pythonfloatorintprimitives. Missing or
unmeasured values are stored as JSONnull. Errors are stored as separate_errfields.
Boolean classifications (core_collapsed,inner_galaxy,sgr_stream) are native JSON
booleans. No unit conversions were performed without documentation. Radii in arcmin
(Harris) and parsecs (Baumgardt) are both preserved in the schema because the two sets
of values are derived from different methods and should be kept distinct.
3.3 Null Handling and Coverage Gaps
Null values arise from three distinct causes.Physical inaccessibility: the four Gaia-
invisible clusters have null Gaia blocks because their dust columns prevent optical astrom-
etry.Survey coverage limits: APOGEE DR17 observed 72 of 174 clusters; Baumgardt v4
covers 154 of 174; uncovered clusters lack sufficient member stars for N-body fitting or
fell outside observed fields.Catalog vintage: the 22 clusters not in Harris 2010 have null
4

Submitted to PASP Flynn (2026) — MW GC Corpus v1.3.1
Harris fields because they were discovered after the 2010 revision. All three causes are
documented in the provenance blocks embedded in each cluster record.
3.4 Provenance Embedding
Each data block carries a provenance sub-object containing the source citation, DOI or
URL, and methodological notes. Provenance is embedded at the block level to balance
completeness with compactness, ensuring that any record retrieved from the JSONL cor-
pus carries its own attribution.
3.5 Known Issues and Bug Fix History
v1.3.1 patch.Three bugs identified during peer review were corrected before the Zenodo
deposit.Bug 1: 14 clusters with two-word alt-names (NGC 104/47 Tuc, NGC 1904/M 79,
NGC 4590/M 68, NGC 5024/M 53, NGC 5272/M 3, NGC 5904/M 5, NGC 6205/M 13,
NGC 6218/M 12, NGC 6093/M 80, NGC 6266/M 62, NGC 6273/M 19, NGC 6656/M 22,
NGC 6779/M 56, Ton 2) had a one-column rightward shift inl,b, and Harris distance
fields. Thebaumgardt2023positionalfieldsserveastheauthoritativesourceforpositional
queries for all 154 covered clusters.Bug 2: 2MS-GC01 and GLIMPSE01 had incorrect
feh,feh_weight, andebvvalues due to a zero-weight parse error.Bug 3: the horizontal
branch magnitudev_hbwas absent from the flat CSV export. All three are fixed in v1.3.1.
Note on theinner_galaxyflag
Theinner_galaxybooleanisinheritedfromanupstreamclassificationlistandisnotcom-
puted from the currentbaumgardt2023.r_gc_kpcvalues in this corpus. Cross-checking
the flag against the Baumgardt (2023) Galactocentric distances reveals inconsistencies
that arise because the upstream flag was assigned using older distance estimates (pre-
dominantly Harris 2010) that have since been superseded. For example, UKS 1 carries
inner_galaxy = truebased on its Harris distance ofR GC= 0.7kpc, but the Baumgardt
(2023) N-body fit places it atR GC= 7.47kpc; the flag was not updated when the new
distance was ingested. NGC 104 (47 Tuc) likewise carries the flag set totruedespite a
canonical halo/thick-disk location (l= 306◦,b=−45◦,RGC= 7.5kpc), reflecting an ar-
tifact of the upstream list rather than a current chemodynamical assignment. Conversely,
NGC 6553 atR GC= 2.37kpc is flaggedfalse. Users requiring a clean radial cut should
filter directly onbaumgardt2023.r_gc_kpcrather than relying on the flag. The flag is re-
tained in v1.3.1 for backward compatibility with consumers of the upstream classification;
a forthcoming v1.4 release will recomputeinner_galaxyfrom the Baumgardt distances
and document the threshold explicitly.
5

Submitted to PASP Flynn (2026) — MW GC Corpus v1.3.1
4 Schema Description
Table 1 gives the complete per-cluster JSON schema. The flat CSV exposes all fields at
one level using block-prefix naming: no prefix for Harris/identity fields,b_for Baumgardt
2023,gaia_for Vasiliev 2021,a_for APOGEE DR17.
Table 1: Per-cluster JSON schema for the Milky Way Globular Cluster Corpus v1.3.1.
Block Field(s) Type Description
(top)cluster_idstr Primary identifier
(top)alt_namestr|null Common name
positionra_hms, dec_dmsstr|null Equatorial coords
positionl_deg, b_degfloat|null Galacticl,b(deg)
distancesr_sun_kpcfloat|null Heliocentric distance (kpc)
distancesr_gc_kpcfloat|null Galactocentric distance (kpc)
distancesx/y/z_kpcfloat|null GalactocentricX/Y/Z(kpc)
metallicityfehfloat|null[Fe/H](Harris)
metallicityfeh_weightint|null Metallicity weight
metallicityebvfloat|null ReddeningE(B−V)
photometryv_hb, dist_modfloat|nullV HB, distance modulus
photometryv_t, m_v_tfloat|null Apparent/absoluteVmag
photometryellipticityfloat|null Ellipticity
photometrycolors
{ub,bv,vr,vi}float|null Color indices
kinematicsv_r_kms±errfloat|null Heliocentric RV (kms−1)
kinematicssig_v_kms±errfloat|null Velocity dispersion (kms−1)
structureking_concentrationfloat|null King concentrationc
structurecore_collapsedbool Core-collapse flag
structurer_core/half_arcminfloat|null Core/half-light radii (arcmin)
structurer_core/half_kpcfloat|null Core/half-light radii (kpc)
structuremu_v_central,
log_rho0float|null Central brightness/density
dynamicslog_t_rc/rh_yrfloat|null Relaxation times (logyr)
flagsinner_galaxybool Inner-Galaxy association flag
(legacy; see Sec. 3.5)
flagssgr_streambool Sgr dSph association
gaia_edr3mu_alpha_mas_yr±
errfloat|null PM inαcosδ(masyr−1)
gaia_edr3mu_delta_mas_yr±
errfloat|null PM inδ(masyr−1)
6

Submitted to PASP Flynn (2026) — MW GC Corpus v1.3.1
Table 1 – continued
Block Field(s) Type Description
gaia_edr3corr_mufloat|null PM correlation coefficient
gaia_edr3parallax_mas±errfloat|null Gaia parallax (mas)
gaia_edr3n_members_gaiaint|nullNstars with good astrometry
baumgardt2023r_sun/gc_kpc±errfloat|null Distances (kpc)
baumgardt2023rv_kms±errfloat|null Mean RV (kms−1)
baumgardt2023x/y/z_kpc±errfloat|null Galactocentric position (kpc)
baumgardt2023u/v/w_kms±errfloat|null Space velocities (kms−1)
baumgardt2023r_peri/apo_kpc±
errfloat|null Orbital pericenter/apocenter
(kpc)
baumgardt2023n_rv, n_pmint|nullNstars with RV/PM
baumgardt2023mass_msun±errfloat|null Dynamical mass (M ⊙)
baumgardt2023rc/rhl/rhm/rt_pcfloat|null Structural radii (pc)
baumgardt2023sigma0_kms,
v_esc_kmsfloat|null Central dispersion,v esc
baumgardt2023mf_slope±errfloat|null Global IMF slopeα
baumgardt2023eta_c, eta_hfloat|null Mass segregation parameters
baumgardt2023a_rot_kms±err,
p_rot_pctfloat|null Rotation amplitude/probabil-
ity
apogee_dr17feh_apogeefloat|null Mean[Fe/H](APOGEE)
apogee_dr17rv_mean_kms±errfloat|null Mean RV (kms−1)
apogee_dr17mass_1e4_msunfloat|null Mass (104M⊙)
apogee_dr17r_jacobi_degfloat|null Jacobi radius (deg)
apogee_dr17n_membersint|nullNmember stars in VAC
5 Example Records
The following three records illustrate the range of data coverage. Provenance sub-blocks
are omitted for brevity.
5.1 NGC 104 (47 Tuc) — Full Four-Survey Record
NGC 104 is the most data-rich record in the corpus. All four source surveys contribute:
Harris provides the foundational photometry; Vasiliev and Baumgardt [2021] yield 39,932
Gaia member stars (σ µ≈0.008masyr−1); Baumgardt et al. [2023] derive a total mass
of853,000M ⊙withσ 0= 11.9kms−1and an essentially circular orbit (r peri= 5.47kpc,
rapo= 7.51kpc); and Schiavon et al. [2024] measure a mean[Fe/H] =−0.74from 297
7

Submitted to PASP Flynn (2026) — MW GC Corpus v1.3.1
member giants, consistent with the Harris photometric value of−0.72.
{" cluster_id ": "NGC 104" , " alt_name ": "47 Tuc",
" position ": {" l_deg ": 305.89 , " b_deg ": -44.89} ,
" metallicity ": {" feh ": -0.72 , "ebv ": 0.04} ,
" structure ": {" king_concentration ": 2.07 , " core_collapsed ":
false ,
" r_core_kpc ": 0.032 , " r_half_kpc ": 0.2821} ,
" gaia_edr3 ": {" mu_alpha_mas_yr ": 5.252 , " mu_alpha_err ": 0.021 ,
" mu_delta_mas_yr ": -2.551 , " n_members_gaia ":
39932} ,
" baumgardt2023 ": {" mass_msun ": 853000.0 , " rc_pc ": 0.61 ,
" sigma0_kms ": 11.9 , " r_peri_kpc ": 5.47 ,
" r_apo_kpc ": 7.51 , " mf_slope ": -0.65} ,
" apogee_dr17 ": {" feh_apogee ": -0.74 , " n_members ": 297}}
5.2 2MS-GC01 — Gaia-Invisible Record
2MS-GC01 was discovered through infrared photometry. Its extreme reddening (E(B−
V) = 6.80) renders it invisible to Gaia optical detectors. Harris provides limited photome-
tryandaradialvelocityfromnear-infraredspectroscopy. Thegaia_edr3,baumgardt2023,
andapogee_dr17blocks are null throughout, representing physical inaccessibility rather
than a survey coverage gap.
{" cluster_id ": "2MS - GC01 ", " alt_name ": "2 MASS - GC01 ",
" position ": {" l_deg ": 10.48 , " b_deg ": 0.11} ,
" metallicity ": {" feh ": null , " feh_weight ": 0, " ebv ": 6.80} ,
" kinematics ": {" v_r_kms ": 0.85 , " sig_v_kms ": 8.43} ,
" gaia_edr3 ": null ,
" baumgardt2023 ": null ,
" apogee_dr17 ": null }
5.3 Bliss 1 — Post-Harris Discovery, Gaia-Only Record
Bliss 1 was discovered after the Harris 2010 revision. All Harris blocks are null. The
Gaia EDR3 block provides a proper motion solution from 10 member stars. Baumgardt
and APOGEE blocks are null because 10 members are insufficient for N-body fitting.
Galactic coordinates were computed from the Gaia RA/Dec usingastropy[Astropy
Collaboration et al., 2022].
{" cluster_id ": " Bliss 1", " alt_name ": null ,
" position ": {" l_deg ": 290.8321 , " b_deg ": 19.6528} ,
8

Submitted to PASP Flynn (2026) — MW GC Corpus v1.3.1
" metallicity ": {" feh ": null , "ebv ": null },
" gaia_edr3 ": {" ra_deg ": 177.511 , " dec_deg ": -41.772 ,
" mu_alpha_mas_yr ": -2.34 , " mu_alpha_err ": 0.042 ,
" mu_delta_mas_yr ": 0.138 , " mu_delta_err ": 0.038 ,
" n_members_gaia ": 10} ,
" baumgardt2023 ": null ,
" apogee_dr17 ": null }
6 Coverage Analysis
Figure 1 shows the sky distribution of all 174 clusters in Galactic coordinates, coloured by
survey coverage tier. The strong concentration toward the Galactic centre (l∼0◦,b∼0◦)
reflects the underlying distribution of the MW GC system, with APOGEE coverage (blue)
followingthesametrendsinceAPOGEEtargetswerepreferentiallydrawnfromaccessible,
non-reddened fields. Figure 2 summarises the per-block coverage fractions.
Table 2 gives the coverage summary. The 12 clusters with Gaia PM data only are
among the most recently discovered systems and will accumulate additional parameters
as follow-up observations are published.
Table 2: Coverage summary by source block.
Source Clusters Fraction Primary gap reason
Harris (1996, 2010 ed.) 157/174 90% 17 post-2010 discoveries
Vasiliev & Baumgardt (2021) 170/174 98% 4 Gaia-invisible (extreme extinction)
Baumgardt et al. (2023) 154/174 89% Insufficient members for N-body
Schiavon et al. (2024) APOGEE 72/174 41% Fiber coverage + extinction limits
Total 174 17,438 non-null data points
9

Submitted to PASP Flynn (2026) — MW GC Corpus v1.3.1
Figure 1: Sky distribution of the 174 clusters in the Milky Way Globular Cluster Corpus
v1.3.1, shown in Galactic coordinates (Aitoff projection) and coloured by survey coverage
tier. Blue: all four surveys (72 clusters). Green: Harris + Gaia + Baumgardt (82).
Orange: Harris + Gaia only (5). Red: Gaia only, post-Harris discovery (12). Purple:
Harris only, Gaia-invisible (4). The strong concentration nearl= 0◦reflects the Galactic
bulge and inner halo GC population.
Figure 2: Survey coverage by source block. Each bar shows the number and percentage
of clusters with non-null data in that block. Grey portions indicate uncovered clusters;
the reason for each gap differs by block (see Table 2 and Section 3).
10

Submitted to PASP Flynn (2026) — MW GC Corpus v1.3.1
7 Scientific Figures
Figure 3 shows the relationship between cluster metallicity and dynamical mass for both
the Harris photometric[Fe/H](left,n= 149) and the APOGEE DR17 spectroscopic
[Fe/H](right,n= 72). The broad scatter with no strong correlation is consistent with the
known independence of GC mass and metallicity across the full Galactic GC population.
The two panels demonstrate the internal consistency of the corpus across independent
survey blocks.
Figure 4 shows the Gaia EDR3 proper motion diagram for all 170 clusters with PM
measurements, coloured by Galactocentric distanceR GC. The concentration of inner-
halo clusters (yellow/orange) near the origin reflects the near-zero net PM expected
for an isotropic, pressure-supported population at smallR GC, while outer-halo clusters
(blue/purple) show the larger PM amplitudes associated with more radial orbits.
Figure 3: Metallicity vs. dynamical mass for the Milky Way globular cluster corpus. Left:
Harris (1996, 2010) photometric[Fe/H]vs. Baumgardt et al. (2023)log10dynamical mass
(n= 149clusters). Right: APOGEE DR17 spectroscopic[Fe/H]vs. the same mass
(n= 72). The broad scatter and lack of strong correlation is consistent with the known
independence of GC mass from metallicity across the full population.
11

Submitted to PASP Flynn (2026) — MW GC Corpus v1.3.1
Figure 4: Gaia EDR3 proper motion diagram (µ α∗vs.µδ) for 170 clusters with Vasiliev
& Baumgardt (2021) measurements, coloured by Galactocentric distanceR GCfrom the
Baumgardt et al. (2023) database. The concentration near zero reflects the bulk Galactic
rotation frame, with outer-halo clusters (blue/purple) exhibiting larger proper motion
amplitudes from more eccentric orbits.
8 Use Cases
The corpus is designed to support a broad range of scientific and computational applica-
tions.
RAG and LLM applications.The JSONL format allows each cluster record to
be retrieved as a semantically complete unit by a vector database or retrieval pipeline.
Embedded provenance blocks ensure retrieved facts carry their citations, supporting
verifiable AI-assisted literature synthesis. The deterministic JSONL structure stabi-
lizes the token context window for inference; the corpus has been validated against ten
instruction-following LLM systems spanning frontier cloud models (Claude Opus 4.6,
Claude Haiku 4.5, Microsoft Copilot Pro, Google Gemini Pro) and local open-weight
models from 1.5B to 70B parameters, and is suitable for deployment with any instruction-
following model supporting structured context injection.
Cluster classification and machine learning.The flat CSV provides a 82-column
feature matrix suitable for use in scikit-learn, PyTorch, or R-based classifiers. Natural
targets include core-collapse classification (29 known cases, boolean label), Sagittarius
12

Submitted to PASP Flynn (2026) — MW GC Corpus v1.3.1
stream membership (6 clusters), and inner-Galaxy vs. halo membership.
Orbit modeling and dynamical studies.Thebaumgardt2023block provides pre-
computed pericenter and apocenter radii, 3D space velocities (U/V/W), and dissolution
timescales for 154 clusters under the Irrgang et al. [2013] Galactic potential, suitable for
use directly or as starting conditions for custom integrations ingalpyoragama.
Chemical tagging and population studies.Theapogee_dr17block provides
mean[Fe/H]for 72 clusters from a homogeneous high-resolution spectroscopic pipeline,
enablingdirectcomparisonwithBaumgardtdynamicalmassesandGaiakinematicswithin
a single record.
Multi-surveycross-validation.Threeindependentradialvelocitysourcesarepresent
in the corpus: Harris mean RV, Baumgardt N-body fit RV, and APOGEE mean member
RV. Two independent distance estimates are also available (Harris vs. Baumgardt). The
corpus structure makes systematic cross-survey comparisons straightforward.
Reproducibility.All code used to construct the corpus is available at the Zenodo
record [Flynn, 2026a] inbuild_scripts_v1.3.1.zip.
8.1 RAG Demonstration Examples
To validate the corpus as a RAG knowledge base, we demonstrate three representative
queriessubmittedtoinstruction-followingLLMs(ClaudeOpus4.6andClaudeSonnet4.6,
Anthropic)suppliedwiththerelevantclusterrecordextractedfromharris_gc_corpus_v1.3.1.jsonl
as sole context. The LLM receives no prior knowledge of the dataset beyond the retrieved
JSON block. The three examples exercise the Baumgardt orbital block, the APOGEE
chemistry block, and a cross-corpus scientific calculation linking the GC kinematics to
the empirical omega kinematic correction of Flynn and Cannaliato [2025].
Example 1 — Orbital classification (Baumgardt block)
Query:
Is NGC 6121 (M 4) on a prograde or retrograde orbit relative to Galactic
rotation, and what is its orbital eccentricity?
Retrieved context (key fields):
{" cluster_id ": "NGC 6121" ,
" baumgardt2023 ": {
" r_peri_kpc ": 0.62 , " r_peri_err ": 0.07 ,
" r_apo_kpc ": 6.52 , " r_apo_err ": 0.13 ,
" v_kms ": 45.28 ,
" rv_kms ": 71.22 , " rv_err ": 0.13 ,
" mu_alpha_mas_yr ": -12.511 ,
13

Submitted to PASP Flynn (2026) — MW GC Corpus v1.3.1
" mu_delta_mas_yr ": -19.017 ,
" r_gc_kpc ": 6.45}}
Reference response (Jupyter-verified):
NGC 6121 has aVGalactocentric velocity component of+45.28kms−1. The
sign convention adopted by Baumgardt and Vasiliev [2021] places the Sun at
X=−8.2kpc withV ⊙= +232kms−1; a positiveVin this frame indicates
prograde rotation (co-rotating with the Galactic disk). NGC 6121 is therefore
on aprograde orbit. The orbital eccentricity ise= (r apo−r peri)/(r apo+
rperi) = (6.52−0.62)/(6.52 + 0.62) = 0.826, indicating a highly radial orbit
despite the prograde sense.
Example 2 — Chemical population query (APOGEE block)
Query:
Which inner-Galaxy globular clusters (R GC<3kpc) have APOGEE spectro-
scopic[Fe/H]>−0.5, suggesting a bulge population origin?
Retrieved context (three representative records):
{" cluster_id ": "NGC 6528" , " flags ": {" inner_galaxy ": true },
" baumgardt2023 ": {" r_gc_kpc ": 0.70} ,
" apogee_dr17 ": {" feh_apogee ": -0.16 , " n_members ": 4}}
{" cluster_id ": "NGC 6553" , " flags ": {" inner_galaxy ": false },
" baumgardt2023 ": {" r_gc_kpc ": 2.37} ,
" apogee_dr17 ": {" feh_apogee ": -0.19 , " n_members ": 17}}
{" cluster_id ": " Liller 1", " flags ": {" inner_galaxy ": true },
" baumgardt2023 ": {" r_gc_kpc ": 0.74} ,
" apogee_dr17 ": {" feh_apogee ": -0.14 , " n_members ": 30}}
{" cluster_id ": "NGC 6304" , " flags ": {" inner_galaxy ": false },
" baumgardt2023 ": {" r_gc_kpc ": 2.19} ,
" apogee_dr17 ": {" feh_apogee ": -0.48 , " n_members ": 34}}
Note:theinner_galaxyflag uses a stricterR GC<2kpc threshold than the query’s
RGC<3kpc filter, which is why NGC 6553 and NGC 6304 satisfy the query but are
flaggedfalse.
Reference response (Jupyter-verified):
Filtering the corpus forR GC<3kpc and[Fe/H] APOGEE>−0.5identifies
four clusters: Liller 1 (−0.14), NGC 6528 (−0.16), NGC 6553 (−0.19), and
NGC 6304 (−0.48). Terzan 5 ([Fe/H] =−0.78) and NGC 6441 (R GC= 4.78
kpc) lie inside the inner Galaxy but fail the metallicity and distance cuts
14

Submitted to PASP Flynn (2026) — MW GC Corpus v1.3.1
respectively. The three metal-richest (Liller 1, NGC 6528, NGC 6553) all have
[Fe/H]>−0.2, consistent with the metal-rich bulge GC population identified
by Baumgardt and Vasiliev [2021]. APOGEEn_membersis low (≤34) for
all four, so mean metallicities carry non-negligible sampling uncertainty.
Example 3 — Cross-corpus omega correction validation (Baumgardt×Flynn
& Cannaliato 2025)
Query:
Using the empirical omega kinematic correction of Flynn & Cannaliato (2025)
withω= 7.06kms−1kpc−1, what is the expected rotational velocity contri-
bution at the Galactocentric radius of NGC 6266 (M 62), and how does this
compare to the cluster’s observedVspace velocity component?
Retrieved context:
{" cluster_id ": "NGC 6266" ,
" baumgardt2023 ": {
" r_gc_kpc ": 2.46 ,
" v_kms ": 106.94 ,
" u_kms ": 87.09 ,
" w_kms ": 59.64 ,
" r_peri_kpc ": 1.08 , " r_apo_kpc ": 2.69}}
Reference response (Jupyter-verified):
The Flynn & Cannaliato (2025) omega correction [Flynn and Cannaliato,
2025] proposesV obs=V Kepler +Rω, whereω= 7.06kms−1kpc−1is the mean
kinematic field amplitude derived from 84 SPARC galaxies. At the Galac-
tocentric radius of NGC 6266 (R GC= 2.46kpc), the predicted rotational
contribution isRω= 2.46×7.06 = 17.4kms−1.
The Baumgardt et al. (2023)Vcomponent for NGC 6266 is+106.94kms−1.
Subtracting the omega contribution gives a correctedV corr= 106.94−17.4 =
89.5kms−1, representing a∼16% reduction in the prograde rotational com-
ponent. This is consistent in sign and order of magnitude with the omega
correction applied to inner-disk galaxies in Flynn and Cannaliato [2025], sug-
gesting the kinematic field may extend into the inner Galactic halo where
bulge GCs reside. This application of the corpus to GC kinematics represents
a natural extension of the omega correction framework beyond the galaxy disk
regime for which it was originally derived.
Jupyterverification:Allthreeexampleswereindependentlyverifiedbydirectcomputa-
tionagainstthecorpusinJupyterLabusingthePythonscriptsgc_rag_example1_orbital.py,
15

Submitted to PASP Flynn (2026) — MW GC Corpus v1.3.1
gc_rag_example2_chemistry.py,andgc_rag_example3_omega.py,includedinbuild_scripts_v1.3.1.zip.
Numerical results agree with reference responses to within rounding precision.
Scientific note: omega correction in the GC population
Example 3 extends the Flynn & Cannaliato (2025) omega kinematic correction [Flynn
and Cannaliato, 2025] beyond its original disk-galaxy context to the GC population.
ApplyingV ω=R GC×ωto all 64 inner-Galaxy clusters (R GC<4kpc) with BaumgardtV
components yields fractional corrections spanning∼3% to∼970% of|V|, with a median
of∼15%. For the majority of clusters the correction is modest (5–20%), consistent with
what is found for inner-disk galaxies at similar radii. The high-tail clusters (Terzan 10
at 970%, NGC 6402 at 134%, NGC 6541 at 116%, NGC 6539 at 85%) all share the
same diagnostic: their observedVcomponent is small in magnitude (e.g., Terzan 10 has
V=−1.58kms−1), so the ratio diverges not becauseωis large in absolute terms but
because these clusters are on highly radial or near-retrograde orbits where the GC velocity
field is pressure-supported rather than rotationally ordered.
This result constitutes a meaningful null: the omega kinematic field, as derived from
84 rotationally-supported SPARC disk galaxies, isnota systematic organizing effect in
the GC population. The 64-cluster sample showsσ(V obs) = 105.6kms−1before correction
andσ(V corr) = 106.3kms−1after; the omega subtraction increases population scatter by
0.7 kms−1rather than reducing it. GCVcomponents span∼−250to+250km/s with no
preferred rotational sense as a population. This is physically expected if omega represents
a property of disk kinematics rather than a universal Galactic potential term, and the GC
corpus provides the first direct test of the correction outside the disk regime. The finding
appropriately bounds the domain of applicability of the Flynn & Cannaliato (2025) result.
8.2 Multi-System AI Validation
The three RAG examples were tested against ten LLM systems to assess the generaliz-
ability of the corpus as a structured context source. Testing consisted of supplying each
system with the relevant cluster JSON record(s) and the query text verbatim. Grad-
ing criteria: Pass = correct numerical result and correct physical interpretation (2 pts);
Partial = correct computation with incomplete or incorrect interpretation, or one of two
required outputs correct (1 pt); Fail = incorrect numerical result, no JSON grounding,
or no attempt (0 pts). Each example is worth 2 points, for a maximum of 6 points per
system. Numerical correctness was verified independently in Jupyter Lab for all examples
prior to LLM testing.
Allfourfrontiercloudsystemsachievedperfectscoresof6/6. Amonglocalopen-weight
models,thecommunity-distilledGemma-4-31B-Instruct-Claude-Opus-Distillachieved6/6
byfollowingthecorrecteccentricityformulae= (r apo−rperi)/(r apo+rperi), whileQwen3.6-
16

Submitted to PASP Flynn (2026) — MW GC Corpus v1.3.1
35B-A3B (Alibaba’s Mixture-of-Experts model with 35B total / 3B active parameters)
scored 5/6, failing only on Example 1 through use of the incorrect eccentricity formula
e= 1−r peri/rapo≈0.905in place of the standard formulation. AstroSage-70B [de Haan
et al., 2025], the domain-specialized 70B model fine-tuned from Llama-3.1-70B by the
AstroMLab collaboration, scored 4/6: it computed Example 1 with the correct formula
but reportede≈0.843(a minor numerical drift from the verified0.826), and on Exam-
ple 2 retrieved only two of the four qualifying clusters. AstroSage-8B scored 3/6, failing
Example 1 (e= 0.94, mis-computation) and partially passing Example 2 (only one of
four clusters identified). Sub-10B general-purpose models showed markedly limited RAG
grounding: Mistral-7B-Instruct (v0.2, Q5_K_M quantization) attempted only Exam-
ple 3, returning a slight rounding artefact (V ω= 17.19km/s vs. verified17.37km/s)
for 1/6 total. DeepSeek-R1-Distil-1.5B ignored the injected JSON context entirely and
responded from prior knowledge with fabricated coordinates for NGC 6121, scoring 0/6.
A notable operational finding is that Claude Haiku 4.5, despite achieving a perfect
score, required three identical prompts before correctly attaching the uploaded corpus as
context (the query text was unchanged across attempts). This pattern is consistent with
an attachment-loading inconsistency in the API/UI rather than a reasoning failure, but
represents a deployment friction point for automated RAG pipelines and suggests retry
logic may be advisable when integrating Haiku for structured-context tasks.
Results are summarized in Table 3.
The threshold pattern is informative: all 30B+ parameter models that engaged with
theJSONcontextachievedatleastPartialcreditoneveryexample,whilesub-10Bgeneral-
purpose models either grounded poorly (Mistral 7B) or hallucinated entirely (DeepSeek-
R1-Distil-1.5B). The domain-specialized AstroSage models, despite their astronomical
fine-tuning, showed retrieval gaps on Example 2’s filter-and-list task, suggesting that
domain specialization for Q&A does not automatically translate to structured-context fil-
tering performance. The two systems that arrived ate= 0.905on Example 1 (Qwen 3.6)
and the formula-correcte≈0.83(Gemma distillation, AstroSage 70B) suggest that the
orbital eccentricity formula is genuinely a discriminating test for orbital-mechanics knowl-
edge.
9 Data Access
The corpus is available at Zenodo DOI: 10.5281/zenodo.19907766 [Flynn, 2026a] in three
formats:
•harris_gc_corpus_v1.3.1.jsonl(622.7 KB) — primary format, one JSON object
per line per cluster
•harris_gc_corpus_v1.3.1_flat.csv(66.4 KB) — flat table, 82 columns
17

Submitted to PASP Flynn (2026) — MW GC Corpus v1.3.1
Table 3: LLM system validation results for the three RAG demonstration examples. Pass
= correct numerical result and physical interpretation (2 pts); Partial = correct computa-
tion with incomplete physical interpretation or one of two required outputs correct (1 pt);
Fail = incorrect result or no JSON grounding (0 pts). Score = total out of 6 possible
points.
System Provider /
TypeEx. 1 Ex. 2 Ex. 3 Score Notes
Claude Opus 4.6 Anthropic /
cloudPass Pass Pass 6/6 Perfect; extended omega
discussion unprompted.
Copilot Pro Microsoft/cloud Pass Pass Pass 6/6 All correct; clean formatted
responses.
Gemini Pro Google / cloud Pass Pass Pass 6/6 All correct; concise, no
extrapolation.
Claude Haiku 4.5 Anthropic /
cloudPass Pass Pass 6/6 Same query yielded the
correct answer on the third
attempt without prompt
modification, indicating
attachment-loading
inconsistency rather than a
reasoning failure.
Gemma-4-31B-Instruct-
Claude-Opus-Distillcommunity fine-
tune / localPass Pass Pass 6/6 Correct formula; all four
clusters identified.
Qwen 3.6-35B-A3B (MoE) Alibaba / local Partial Pass Pass 5/6 Ex. 1: wrong eccentricity
formula (e= 0.905).
AstroSage 70B (build
20251009)AstroMLab / lo-
calPartial Partial Pass 4/6 Ex. 1:e= 0.843(numerical
drift). Ex. 2: found only
Liller 1 + NGC 6528 of four.
AstroSage 8B AstroMLab / lo-
calFail Partial Pass 3/6 Ex. 1:e= 0.94(computation
error); could not determine
orbit sense.
Mistral 7B-Instruct v0.2
(Q5_K_M)Mistral / local
quant— — Partial 1/6 Only Ex. 3 attempted; slight
rounding (V ω= 17.19vs.
17.37).
DeepSeek-R1-Distil-1.5B community
quant / localFail — — 0/6 Ignored JSON; fabricated
NGC 6121 coordinates from
prior knowledge.
Totals (10 systems)Pass: 19, Partial: 4, Fail/N-A: 7 43/60 72% benchmark score
•harris_gc_corpus_v1.3.1.json(868 KB) — full nested JSON with metadata
header
A Python loading snippet:
importjson
clusters = [ json . loads ( line )
forlinein open(" harris_gc_corpus_v1 .3.1. jsonl ")]
# Get NGC 104
ngc104 =next(cforcinclusters
ifc[" cluster_id "] == "NGC 104 ")
# All clusters with APOGEE chemistry
feh = {c[" cluster_id "]: c[" apogee_dr17 "][" feh_apogee "]
forcinclusters
18

Submitted to PASP Flynn (2026) — MW GC Corpus v1.3.1
ifc.get (" apogee_dr17 ")
andc[" apogee_dr17 "]. get(" feh_apogee ")is notNone }
10 Relationship to Previous Work
This corpus is a companion to the Unified Galaxy HI Rotation Curve Corpus [Flynn,
2026b], whichfollowsidenticaldesignprinciplesfor438spiralanddwarfirregulargalaxies.
Both corpora use the same data-type standards, the same block-plus-provenance schema
structure, the same JSONL-primary file format, and the same philosophy of including
all available data with explicit null annotation rather than restricting to a complete-data
subset. This consistency allows the two corpora to serve as a unified astrophysical RAG
knowledge base covering both resolved stellar systems (GCs) and unresolved extragalactic
objects.
11 Future Versions
v1.4:Addition of cluster ages from a homogeneous compilation [e.g., Marín-Franch et al.,
2009, VandenBerg et al., 2013].
v1.5:Individual per-star APOGEE DR17 abundance records (up to 20 elements per
star, 6,422 stars) as a separate linked file.
Future:Na–O and Mg–Al anticorrelation data [Carretta et al., 2009] for the∼19
clusters with published measurements; multi-model profile parameters from McLaughlin
and van der Marel [2005].
12 Summary
We have presented the Milky Way Globular Cluster Corpus v1.3.1, a production-ready
multi-survey machine-readable database of 174 Milky Way globular clusters with 17,438
non-null data points. The corpus integrates four independent published surveys spanning
photometry, spaceastrometry, N-bodydynamics, andhigh-resolutionspectroscopy, stored
inaconsistent, typed, null-safeJSONL/CSV/JSONstructurewithembeddedprovenance.
It is the globular cluster analog of the Unified Galaxy HI Rotation Curve Corpus and is
available at Zenodo DOI 10.5281/zenodo.19907766 under a CC BY 4.0 license.
Acknowledgments
This work used no external funding. Computational infrastructure provided by EPS
Research.
19

Submitted to PASP Flynn (2026) — MW GC Corpus v1.3.1
AI Use Acknowledgment.Claude Opus 4.6 and Claude Sonnet 4.6 (Anthropic)
were used as the primary development and RAG validation systems throughout corpus
construction, schema design, and manuscript preparation. Claude Haiku 4.5 (Anthropic),
Microsoft Copilot Pro, and Google Gemini Pro were used for multi-system RAG vali-
dation and manuscript review. AstroSage-70B (AstroMLab, build 20251009; de Haan
et al. 2025) and AstroSage-8B (AstroMLab) were used for domain-specialized validation.
Qwen 3.6-35B-A3B (Alibaba), Gemma-4-31B-Instruct-Claude-Opus-Distill (community
fine-tune), Mistral-7B-Instruct v0.2 (Mistral), and DeepSeek-R1-Distil-1.5B (community
quantization) were used for local open-weight model validation. All AI-assisted outputs
were verified by the human author; all scientific claims, numerical results, and data val-
ues were independently verified in Jupyter Lab using the Python scripts provided in the
Zenodo deposit. The AI systems did not generate original data; all corpus values derive
from the four primary published sources cited in the text. No AI system is listed as an
author.
This paper made use of: the Harris (1996, 2010) GC catalog at McMaster University;
the Vasiliev & Baumgardt (2021) Gaia EDR3 catalog (ESA Gaia mission); the Baum-
gardt et al. N-body database at the University of Queensland; the APOGEE DR17 Value
Added Catalog from the Sloan Digital Sky Survey; andastropy[Astropy Collaboration
et al., 2022]. SDSS-IV is managed by the Astrophysical Research Consortium for the
Participating Institutions of the SDSS Collaboration.
References
Abdurro’uf et al. The seventeenth data release of the Sloan Digital Sky Survey.ApJS,
259:35, 2022.
Astropy Collaboration et al. The Astropy project: Sustaining and growing a community-
developed open-source project and status of the v5.0 core package.ApJ, 935:167, 2022.
H. Baumgardt and M. Hilker. A catalogue of masses, structural parameters and velocity
dispersion profiles of 112 Milky Way globular clusters.MNRAS, 478:1520, 2018.
H. Baumgardt and E. Vasiliev. Accurate distances to Galactic globular clusters through
a combination of Gaia and ground-based data.MNRAS, 505:5957, 2021.
H. Baumgardt et al. Multimass models of 144 Milky Way globular clusters.MNRAS,
521:3991, 2023. doi: 10.1093/mnras/stad631.
E. Carretta et al. Na-O anticorrelation and HB. VIII.A&A, 505:117, 2009.
TijmendeHaanetal. AstroMLab4: Benchmark-toppingperformanceinastronomyQ&A
20

Submitted to PASP Flynn (2026) — MW GC Corpus v1.3.1
with a 70B-parameter domain-specialized reasoning model.arXiv e-prints, 2025. URL
https://arxiv.org/abs/2505.17592.
David C. Flynn. Milky way globular cluster corpus v1.3.1, 2026a. URLhttps://doi.
org/10.5281/zenodo.19907766.
David C. Flynn. Unified galaxy HI rotation curve corpus v7.0, 2026b. URLhttps:
//doi.org/10.5281/zenodo.19491084.
David C. Flynn and Jim Cannaliato. A new empirical fit to galaxy rotation curves.
Frontiers in Astronomy and Space Sciences, 12, 2025. doi: 10.3389/fspas.2025.1680387.
URLhttps://doi.org/10.3389/fspas.2025.1680387.
A. E. García Pérez et al. ASPCAP: The APOGEE stellar parameter and chemical abun-
dances pipeline.AJ, 151:144, 2016.
W. E. Harris. A catalog of parameters for globular clusters in the Milky Way.AJ, 112:
1487, 1996. 2010 revision available athttps://physics.mcmaster.ca/~harris/mwgc.
dat.
A. Irrgang et al. Milky Way mass models for orbit calculations.A&A, 549:A137, 2013.
I. R. King. The structure of star clusters. III.AJ, 71:64, 1966.
L. Lindegren et al. Gaia Early Data Release 3: The astrometric solution.A&A, 649:A2,
2021.
A. Marín-Franch et al. The ACS survey of Galactic globular clusters. VII. relative ages.
ApJ, 694:1498, 2009.
D. E. McLaughlin and R. P. van der Marel. Resolved massive star clusters in the Milky
Way and its satellites: Brightness profiles and a catalog of fundamental parameters.
ApJS, 161:304, 2005. doi: 10.1086/497429.
R. P. Schiavon et al. The APOGEE value added catalogue of Galactic globular cluster
stars.MNRAS, 528:1393, 2024. doi: 10.1093/mnras/stad3419.
A. Sollima, H. Baumgardt, and M. Hilker. The stellar rotation and the kinematic prop-
erties of 28 Milky Way globular clusters.MNRAS, 485:1460, 2019.
M. Trenti and R. van der Marel. No energy equipartition in globular clusters.ApJ, 775:
L2, 2013.
D. A. VandenBerg et al. The ages of 55 globular clusters as determined using an improved
δvHB
TOmethod.ApJ, 775:134, 2013.
21

Submitted to PASP Flynn (2026) — MW GC Corpus v1.3.1
E.VasilievandH.Baumgardt. GaiaEDR3propermotionsofMilkyWayglobularclusters.
MNRAS, 505:5978, 2021. doi: 10.1093/mnras/stab1475.
22