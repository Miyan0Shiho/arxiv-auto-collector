# VERIRAG: Healthcare Claim Verification via Statistical Audit in Retrieval-Augmented Generation

**Authors**: Shubham Mohole, Hongjun Choi, Shusen Liu, Christine Klymko, Shashank Kushwaha, Derek Shi, Wesam Sakla, Sainyam Galhotra, Ruben Glatt

**Published**: 2025-07-23 21:32:50

**PDF URL**: [http://arxiv.org/pdf/2507.17948v1](http://arxiv.org/pdf/2507.17948v1)

## Abstract
Retrieval-augmented generation (RAG) systems are increasingly adopted in
clinical decision support, yet they remain methodologically blind-they retrieve
evidence but cannot vet its scientific quality. A paper claiming "Antioxidant
proteins decreased after alloferon treatment" and a rigorous multi-laboratory
replication study will be treated as equally credible, even if the former
lacked scientific rigor or was even retracted. To address this challenge, we
introduce VERIRAG, a framework that makes three notable contributions: (i) the
Veritable, an 11-point checklist that evaluates each source for methodological
rigor, including data integrity and statistical validity; (ii) a Hard-to-Vary
(HV) Score, a quantitative aggregator that weights evidence by its quality and
diversity; and (iii) a Dynamic Acceptance Threshold, which calibrates the
required evidence based on how extraordinary a claim is. Across four
datasets-comprising retracted, conflicting, comprehensive, and settled science
corpora-the VERIRAG approach consistently outperforms all baselines, achieving
absolute F1 scores ranging from 0.53 to 0.65, representing a 10 to 14 point
improvement over the next-best method in each respective dataset. We will
release all materials necessary for reproducing our results.

## Full Text


<!-- PDF content starts -->

VERIRAG: Healthcare Claim Verification via Statistical Audit in
Retrieval-Augmented Generation
Shubham Mohole
sam88@cornell.edu
Cornell University
USAHongjun Choi
choi22@llnl.gov
Lawrence Livermore
National Laboratory
USAShusen Liu
liu42@llnl.gov
Lawrence Livermore
National Laboratory
USAChristine Klymko
klymko1@llnl.gov
Lawrence Livermore
National Laboratory
USAShashank
Kushwaha
sk89@illinois.edu
University of Illinois
Urbana-Champaign
USA
Derek Shi
derekshi@ucla.edu
University of
California, Los
Angeles
USAWesam Sakla
sakla1@llnl.gov
Lawrence Livermore
National Laboratory
USASainyam Galhotra
sg@cs.cornell.edu
Cornell University
USARuben Glatt
glatt1@llnl.gov
Lawrence Livermore
National Laboratory
USA
Abstract
Retrieval-augmented generation (RAG) systems are increasingly
adopted in clinical decision support, yet they remain methodolog-
ically blind —they retrieve evidence but cannot vet its scientific
quality. A paper claiming "Antioxidant proteins decreased after
alloferon treatment" and a rigorous multi-laboratory replication
study will be treated as equally credible, even if the former lacked
scientific rigor or was even retracted. To address this challenge, we
introduce VERIRAG , a framework that makes three notable con-
tributions: (i) the Veritable , an 11-point checklist that evaluates
each source for methodological rigor, including data integrity and
statistical validity; (ii) a Hard-to-Vary (HV) Score , a quantitative
aggregator that weights evidence by its quality and diversity; and
(iii) a Dynamic Acceptance Threshold , which calibrates the
required evidence based on how extraordinary a claim is. Across
four datasets—comprising retracted, conflicting, comprehensive,
and settled science corpora—the VERIRAG approach consistently
outperforms all baselines, achieving absolute F1 scores ranging
from 0.53 to 0.65, representing a 10 to 14 point improvement over
the next-best method in each respective dataset. We will release all
materials necessary for reproducing our results.
CCS Concepts
•Applied computing →Health informatics ;•Computing
methodologies→Natural language processing .
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
ACM BCB’25, Philadelphia,PA
©2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-1-4503-9876-5/25/06
https://doi.org/10.1145/XXXXX.XXXXXKeywords
Retrieval-Augmented Generation, RAG, Healthcare AI, Evidence-
Based Medicine, Scientific Integrity, Claim Verification, LLM
ACM Reference Format:
Shubham Mohole, Hongjun Choi, Shusen Liu, Christine Klymko, Shashank
Kushwaha, Derek Shi, Wesam Sakla, Sainyam Galhotra, and Ruben Glatt.
2025. VERIRAG: Healthcare Claim Verification via Statistical Audit in Retrieval-
Augmented Generation. In Proceedings of Proceedings of the 16th ACM Con-
ference on Bioinformatics, Computational Biology, and Health Informatics
(ACM BCB’25). ACM, New York, NY, USA, 15 pages. https://doi.org/10.1145/
XXXXX.XXXXX
Figure 1: Illustration of VERIRAG’s Effectiveness based on
a sample record in test set: a standard RAG system treats a
potentially flawed paper as an equally valid source, while
VERIRAG performs a methodological audit.
1 Introduction
"Extraordinary claims require extraordinary evidence" [ 30]. Carl
Sagan’s maxim is a key guidepost for scientific inquiry; yet, in
healthcare research, its systematic adoption faces growing hur-
dles today. The biomedical literature has grown to a point wherearXiv:2507.17948v1  [cs.IR]  23 Jul 2025

ACM BCB’25, XXX XX–YY, 2025, Philadelphia,PA Mohole et al.
researchers can no longer feasibly review it [ 14]. PubMed now
contains more than 35 million citations and continues to expand
by approximately 1.5 million new publications each year [ 21]. As
healthcare institutions rapidly adopt AI technologies, with 80%
either experimenting or scaling implementations [ 10], these sys-
tems are increasingly synthesizing evidence for clinical decisions,
but without the methodological vigilance that characterizes expert
human judgment.
The shortcomings of such systems extend beyond retrieval accu-
racy or LLM hallucination to the missing basic step of evidence val-
idation. Evaluating the strength of retrieved context is fundamental
to making any RAG framework reliable and trustworthy. Current AI
systems, particularly those employing Retrieval-Augmented Gener-
ation (RAG), lack this methodological discrimination. A retracted
paper is an extreme but illustrative example of this issue. As illus-
trated in Figure 1, an example from our experiments shows the
severity and impact on decision making of such shortcomings in
the current RAG systems. This validation gap means that a cancer
biomarker search could surface p-hacked results without flagging
their statistical manipulation [ 31], propagating flawed evidence
directly into clinical decision-making.
Researchers have explored solutions to mitigate this challenge,
but these solutions face limitations. Self-correcting RAG architec-
tures, such as Self-RAG [ 2] and CRAG [ 39], introduce reflection
mechanisms where LLMs critique their outputs. However, these self-
assessments evaluate semantic coherence without formal method-
ological criteria. The system cannot distinguish between a well-
written paper with fabricated data and legitimate research because
it lacks the framework to assess statistical validity or experimental
design. Other work relies on external credibility mapping systems
(CIBER [ 37], Valsci [ 12]). Often, such approaches conflate visibility
with validity, as high-impact publications can harbor serious flaws
while rigorous work at niche venues or new publications remains
undervalued. These systems fail to examine the internal validity
that determines research reliability. Finally, the related thread of
uncertainty quantification in reinforcement learning [ 44] is also
ill-suited for medical evidence assessment, which demands high-
stakes decisions without the opportunity for iterative, reward-based
learning.
In VERIRAG, we directly address these limitations through three
integrated innovations that add methodological scrutiny to AI-
driven evidence synthesis:
(1)Formal Methodological Audit: We implement an 11-point
Veritable checklist, grounded in biostatistical principles, that
systematically evaluates each source for checks such as data
integrity, sample size adequacy, and control of confounders.
(2)Quantitative Evidence Synthesis: Our framework calcu-
lates a Hard-to-Vary (HV) score inspired by ideas from [ 9]
that quantifies the quality of evidence. This metric combines
methodological rigor (from the audit), source diversity (via
redundancy penalties), and the balance of supporting and
refuting evidence.
(3)Dynamic Rigor Calibration: Our system sets a claim-
specific acceptance threshold based on its specificity and
testability, as determined by a model calibrated on human
expert ratings. This applies Sagan’s principle by adjusting
standards based on the nature of the claim itself.Through this approach, VERIRAG demonstrates improved claim
verification accuracy compared to baselines. Thus, we show that
when methodological checks are built into the retrieval process,
RAG shifts from a simple semantic search to a system that actively
audits against established research standards.
2 Methodological Audit Framework
Our system uses a modular framework to assess if research claims
meet methodological standards. Unlike traditional statistical vali-
dation, which requires access to raw data, our framework operates
by performing a deep semantic analysis of the research artifacts
themselves. This process is enabled by two main components: a
structured, machine-readable representation of each document, and
a systematic taxonomy of statistical checks applied to that repre-
sentation.
2.1 Preliminaries
A healthcare research claim is not merely an assertion; it is the
culmination of a complex process of data collection, analysis, and
interpretation. Our framework deconstructs this process as it is
represented in the text.
DEFINITION 2.1 (EVIDENCE-BASED CLAIM). A research claim
cis a tuple ( A, E, M ), where Ais the core Assertion being made,
Eis the Evidence Set (e.g., reported statistics, figures, tables) pre-
sented in its support, and Mis the Methodological Context (e.g.,
study design, statistical methods, inclusion criteria) that qualifies
the evidence.
DEFINITION 2.2 (METHODOLOGICAL AUDIT VIA NL). A
validation check Vis an operator that assesses the integrity of a
claim by analyzing the natural language (NL) representation of its
evidence and methods. It evaluates the consistency, sufficiency, and
rigor of the reported information to identify potential threats to
the validity of the findings.
2.2 Document Representation
To operationalize the methodological audit, VERIRAG transforms
each source paper into a standardized, two-part representation.
This ensures that all downstream processes, from evidence retrieval
to the Veritable checks, can operate on consistent and predictable
data objects.
First, the raw text of each paper is processed into a set of content-
aware chunks. This ingestion pipeline, leveraging Google’s Docu-
ment AI, parses the source PDF to preserve its semantic structure,
identifying headers, paragraphs, lists, and tables. These text chunks
form a common evidence base that is used by both VERIRAG and
the baseline models, ensuring a fair comparison across all systems.
In parallel, a second LLM-driven process generates a focused,
structured JSON object containing high-level methodological sig-
nals extracted from the paper. This analysis schema represents a
metadata layer designed to support the audit. Its high-level struc-
ture enables the key objectives of VERIRAG:
global_integrity_signals: This object captures top-level in-
dicators of research quality, such as funding source trans-
parency, the presence of a conflict of interest statement, and
the data availability statement.

VERIRAG ACM BCB’25, XXX XX–YY, 2025, Philadelphia,PA
veritable_check_signals: This is the core of the analysis, con-
taining a nested object for each of the 11 Veritable Checks.
For each check, it provides a boolean flag for its methodolog-
ical applicability and a concise, objective analysis grounded
in the paper’s text.
This dual-component design—separating the raw evidence chunks
from the structured methodological analysis—is a key architectural
feature of our framework. We foresee VERIRAG’s audit functional-
ity to act as a modular analysis layer that can augment any standard
RAG pipeline, while operating on the shared base of text chunks.
With a standard schema for the metadata, we build an object that
can be queried for quantitative and qualitative assessments relevant
to the factuality check for the claims.
2.3 The Veritable Taxonomy
Our taxonomy, shown in Figure 2, organizes 11 distinct checks that
were designed by mapping core statistical analysis methods with
guidelines from CONSORT [ 8], STROBE [ 35], and PRISMA [ 23] into
two primary categories based on the aspect of the research process
they scrutinize. These checks are designed to be operationalized
via targeted information extraction and semantic analysis. A light-
weight JSONPath query on the paper’s structured JSON analysis
object serves as an applicability filter —it decides whether a check
is methodologically relevant for a given paper. The substantive
pass/fail judgment is delivered by the LLM prompt that follows,
which inspects the relevant text span and returns a structured JSON
verdict.
VeritableTaxonomyData QualityChecksInferential ValidityChecksEvidence AssessmentDataCompletenessStatistical RigorConsistencyAnalysisC1: Data IntegrityC3: SampleRepresentativenessC2: MissingData PatternsC4: OutcomeVariabilityC5: EstimationValidityC6: StatisticalPowerC7: OutlierInfluenceC8: ConfoundingControlC9: Source ConsistencyC10: EffectHomogeneityC11: SubgroupConsistency
Figure 2: A hierarchical taxonomy of the Veritable checks,
organized by the type of textual evidence they evaluate.
2.3.1 Data Quality Checks (Evidence-Based Assessment). These
checks evaluate the quality of the underlying data as described
in the text, identifying anomalies that are visible without access to
the raw dataset.
C1: Data Integrity
Definition: Detects reported anomalies, inconsistencies,
or corrections in the data that suggest potential issues with
data reliability.
Specific Checks: CONSORT 22a : For each group, the num-
bers of participants who were randomly assigned, received in-
tended intervention, and were analysed for the primary outcome?
CONSORT 26 : Numbers analysed, outcomes and estimationScoring Query (LLM Prompt Snippet): C1 (Data
Integrity): Check if participant/sample numbers are
consistent across sections and if data sources are
clearly described.
C2: Missing Data Patterns
Definition: Analyzes how missing data is handled, with
a focus on patterns that could introduce bias (i.e., non-
random attrition) [28].
Specific Checks: CONSORT 21c : How missing data were
handled in the analysis CONSORT 22b : For each group, losses
and exclusions after randomisation, together with reasons
Scoring Query (LLM Prompt Snippet): C2 (Missing
Data): Check if the paper mentions how missing data was
handled (e.g., imputation, exclusion) and if potential
biases are addressed.
C3: Sample Representativeness
Definition: Compares the described sample characteris-
tics against the population to which the claim is general-
ized [13].
Specific Checks: CONSORT 12a : Eligibility criteria for par-
ticipants
Scoring Query (LLM Prompt Snippet): C3 (Sample
Representativeness): Check if the sampling method
is described and if the sample is a reasonable
representation for the study’s claims.
C4: Outcome Variability
Definition: Assesses if the reported variance or hetero-
geneity in outcomes is appropriately acknowledged and
consistent with the claim.
Specific Checks: CONSORT 26 : Result for each group, and
the estimated effect size and its precision (such as 95% confidence
interval)
Scoring Query (LLM Prompt Snippet): C4 (Outcome
Variability): Check if measures of variability (e.g.,
confidence intervals, standard deviation) are reported
for the main outcomes.
2.3.2 Inferential Validity Checks (Methodological Assessment). These
checks evaluate the soundness of the analytical pipeline, encompass-
ing the statistical methods employed and the conclusions drawn.
C5: Estimation Validity
Definition: Assesses whether the described statistical
methods are appropriate for the study design and data
structure.

ACM BCB’25, XXX XX–YY, 2025, Philadelphia,PA Mohole et al.
Specific Checks: CONSORT 21a : Statistical methods used to
compare groups for primary and secondary outcomes, including
harms
Scoring Query (LLM Prompt Snippet): C5
(Estimation Validity): Check if the statistical
tests used are appropriate for the study design and
data type (e.g., t-test for two groups).
C6: Statistical Power
Definition: Evaluates whether the sample size is justi-
fied for the claimed effect, typically via a formal power
analysis [7].
Specific Checks: CONSORT 16a : How sample size was de-
termined, including all assumptions supporting the sample size
calculation
Scoring Query (LLM Prompt Snippet): C6
(Statistical Power): For experimental studies,
check if a power analysis was conducted to justify
the sample size.
C7: Outlier Influence
Definition: Looks for evidence that the study’s conclu-
sions were robust to the influence of extreme data points.
Specific Checks: CONSORT 21d : Methods for any additional
analyses (eg, subgroup and sensitivity analyses), distinguishing
prespecified from post hoc. CONSORT 28 : Any other analyses
performed, including subgroup and sensitivity analyses, distin-
guishing pre-specified from post hoc
Scoring Query (LLM Prompt Snippet): C7 (Outlier
Influence): Check if the authors report any checks
for statistical outliers or sensitivity analyses.
C8: Confounding Control
Definition: For observational studies, evaluates whether
key confounding variables were measured and appropri-
ately adjusted for in the analysis [24].
Specific Checks: STROBE 7 : Clearly define all outcomes, ex-
posures, predictors, potential confounders, and effect modifiers
Scoring Query (LLM Prompt Snippet): C8
(Confounding Control): For observational studies,
check if key confounding variables were identified
and adjusted for in the analysis.
C9: Source Consistency
Definition: Detects cherry-picking by assessing whether
the paper’s claims and citations are consistent with the
broader evidence base [11].Specific Checks: CONSORT 6 : Scientific background and ra-
tionale CONSORT 29 : Interpretation consistent with results, bal-
ancing benefits and harms, and considering other relevant evi-
dence
Scoring Query (LLM Prompt Snippet): C9 (Source
Consistency): Check if the paper’s introduction and
discussion accurately represent prior work, including
any contradictory findings.
C10: Effect Homogeneity
Definition: Assesses whether a single average effect is an
appropriate summary or if it masks significant underlying
heterogeneity.
Specific Checks: PRISMA 14 : Specify any methods used to
identify or quantify statistical heterogeneity (e.g. visual inspection
of results, a formal statistical test for heterogeneity, heterogeneity
variance(𝜏2), inconsistency (e.g. 12), and prediction intervals)
Scoring Query (LLM Prompt Snippet): C10 (Effect
Homogeneity): For meta-analyses, check if statistical
heterogeneity (e.g., I-squared) was assessed and
discussed.
C11: Subgroup Consistency
Definition: Evaluates whether a claim of an overall effect
holds across key subgroups, flagging potentially mislead-
ing post-hoc analyses.
Specific Checks: CONSORT 21d : Methods for any additional
analyses (eg, subgroup and sensitivity analyses), distinguishing
prespecified from post hoc. CONSORT 28 : Any other analyses
performed, including subgroup and sensitivity analyses, distin-
guishing pre-specified from post hoc
Scoring Query (LLM Prompt Snippet): C11 (Subgroup
Consistency): Check if subgroup analyses were
pre-specified and if results are interpreted with
caution (not presented as primary findings).
2.4 Operationalizing the Audit
The taxonomy serves as a foundation for VERIRAG’s auditing
pipeline design. Each check functions as a structured prompt snip-
pet within our framework. These prompts identify textual evidence
corresponding to specific methodological criteria. All prompt snip-
pets are accessible through our shared repository. The aggregated
results of this audit provide a multidimensional profile of a claim’s
credibility, enabling a more nuanced and reliable assessment than
any single metric could provide. Although not a substitute for vali-
dation based on raw data [ 6], this process allows for scalable and
context-aware validation using NL inference capabilities of modern
LLM systems.
3 The Quantitative Framework
After auditing the evidence, VERIRAG requires a formal mecha-
nism to synthesize the results into a single, justifiable verdict. This

VERIRAG ACM BCB’25, XXX XX–YY, 2025, Philadelphia,PA
section formalizes the mathematical machinery for this process.
We define two central quantities: (i) the Hard-to-Vary (HV) score,
which represents the aggregated, quality-weighted evidence for or
against a claim, and (ii) the Dynamic Acceptance Threshold (𝜏auto),
which sets a context-appropriate bar for acceptance.
3.1 Preliminaries and Notation
We first establish the notation for a claim 𝑐being evaluated against
a set of retrieved evidence documents 𝐸={𝑑1,...,𝑑𝑁}.
Definition 3.1 (Evidential State). To quantify its contribution,
each document 𝑑𝑖∈𝐸is assigned a stance 𝑠𝑖∈{− 1,0,1}(Refutes,
Neutral, Supports) and an audit vector 𝑣𝑖∈{0,0.5,1}𝐾across the𝐾=
11methodological checks, where the values encode {Fail, Uncertain,
Pass}.
The audit vector is derived from a richer structured object con-
taining detailed reasoning; the numerical values are used for the
aggregation computations below.
3.2 Hard-to-Vary (HV) Evidential Score
The HV score is calculated by first assessing each document’s indi-
vidual contribution and then aggregating these contributions.
3.2.1 Evidential Operators. We define a series of operators to distill
the evidential value of each document.
Definition 3.2 (Applicability Mask). To account for checks
that do not apply to all study types, for each document 𝑑𝑖we define a
binary mask 𝑚𝑖∈{0,1}𝐾, where𝑚𝑖,𝑘=1if check𝐶𝑘is applicable.
We denote the number of applicable checks as 𝐾𝑖=∥𝑚𝑖∥1.
Definition 3.3 (Intrinsic Document Quality). The method-
ological quality of a document 𝑑𝑖is the mean pass rate over all appli-
cable checks.
𝑞𝑖=1
𝐾𝑖𝐾∑︁
𝑘=1𝑣𝑖,𝑘𝑚𝑖,𝑘∈[0,1]. (1)
Definition 3.4 (Redundancy Penalty). To reward diverse evi-
dence and penalize redundant information from multiple sources, we
calculate a redundancy score at the evidence-chunk level. For each
evidence chunk 𝑐from a document 𝑑𝑖, its redundancy 𝜌𝑐is its highest
textual similarity to any preceding chunk in the evidence set for claim
𝑐. The final penalty for the document, 𝜌𝑖, is the mean redundancy
of its constituent chunks, calculated using a TF-IDF vectorizer and
cosine similarity. The information-gain weight is 𝑤𝑖=1−𝜌𝑖.
Definition 3.5 (Effective Contribution). The total evidential
contribution of a document 𝑑𝑖is its quality weighted by its novelty.
𝜂𝑖=𝑞𝑖·𝑤𝑖. (2)
3.2.2 Aggregation and Parameter Calibration. The individual con-
tributions are aggregated into a final score.
Definition 3.6 (Aggregate Tallies). We sum the effective con-
tributions for all supporting, refuting, and neutral documents.
𝐻𝑆=∑︁
𝑖:𝑠𝑖=+1𝜂𝑖, 𝐻𝑅=∑︁
𝑖:𝑠𝑖=−1𝜂𝑖, 𝐻𝑁=∑︁
𝑖:𝑠𝑖=0𝜂𝑖.Definition 3.7 (Log-Odds and HV Score). The final HV score is
the logistic transformation of the log-odds of the aggregated evidence.
logOdds(𝑐|𝐸)=log𝐻𝑆+𝜆
𝐻𝑅+𝜆−𝛼log 1+𝐻𝑁, (3)
and
HV=𝜎 logOdds(𝑐|𝐸), 𝜎(𝑧)=1
1+𝑒−𝑧. (4)
The parameters 𝛼(neutral evidence penalty) and 𝜆(regularization)
are not theoretical; they are empirically derived via a brute-force
grid search optimization against a ground-truth dataset to maximize
predictive accuracy.
3.3 Dynamic Acceptance Threshold 𝜏auto
To implement the principle that extraordinary claims require extra-
ordinary evidence, VERIRAG sets a dynamic, claim-specific accep-
tance threshold, 𝜏auto, using a multi-step heuristic process.
Definition 3.8 (Claim Features for Thresholding). For each
claim, we generate a feature set including:
•Specificity & Testability ( 𝑆,𝑇):Ratings from 1-10 generated
by an LLM.
•Required Standard ( 𝑅):A categorical label (e.g., ‘Robust
Study‘, ‘Settled Science‘) indicating the expected evidence bur-
den.
•Evidence Volume ( 𝑁𝑒𝑣):The number of evidence papers avail-
able for the claim in the current temporal scenario (e.g., TY0,
TY1).
Definition 3.9 (Threshold Calculation). The final threshold
𝜏autois computed by blending a regression model’s prediction with a
rule-based modifier:
𝜏base=(0.5·𝜋𝑅)+(0.5·𝑓(𝑆,𝑇,𝑅)) (5)
where𝜋𝑅is a hardcoded prior probability for the required standard
𝑅, and𝑓(𝑆,𝑇,𝑅)is a Ridge Regression model trained on expert-rated
claims to predict a boldness score. The base threshold is then adjusted
by the volume of available evidence:
𝜏auto=𝜏base+(𝐶·max(0,𝑁𝑒𝑣
𝑁base−1)) (6)
where𝑁baseis the evidence count in the initial scenario (TY0) and 𝐶is
a scaling factor. This ensures that as more evidence becomes available
over time, a higher bar is set for acceptance.
3.4 Theoretical Properties
The framework possesses several desirable mathematical proper-
ties.
Proposition 3.1 (Boundedness). The final scores are well-behaved,
with HV,𝜏auto∈(0,1).
Proof. HV is the output of a sigmoid function. 𝜏autois calculated
as a blend of probabilities and small modifiers, clamped to remain
within a sensible range of [0.5,0.95]. □
Proposition 3.2 (Monotonicity). The HV score correctly in-
creases with supporting evidence and decreases with refuting evidence,
i.e.,𝜕HV
𝜕𝐻𝑆>0and𝜕HV
𝜕𝐻𝑅<0.

ACM BCB’25, XXX XX–YY, 2025, Philadelphia,PA Mohole et al.
Figure 3: VERIRAG system architecture: data ingestion,
retrieval-audit pipeline and control layer.
Proof. From (3),𝜕log Odds/𝜕𝐻𝑆=(𝐻𝑆+𝜆)−1>0and
𝜕log Odds/𝜕𝐻𝑅=−(𝐻𝑅+𝜆)−1<0. The sigmoid function is mono-
tone increasing. □
Proposition 3.3 (Redundancy Immunity). Adding a semanti-
cally identical evidence chunk 𝑐𝑗to the evidence set for a claim does
not meaningfully change the total evidence tallies.
Proof. The redundancy of a chunk is its max similarity to any
preceding chunk. A duplicate chunk 𝑐𝑗would have a redundancy
score𝜌𝑐𝑗≈1, causing its information-gain weight 𝑤𝑗to approach
0. Its effective contribution 𝜂𝑗thus becomes negligible. □
4 System Architecture
The VERIRAG framework is implemented as a modular architecture
with three distinct, interoperating subsystems: the Data System, the
RAG System, and the Control System. This separation of concerns
ensures flexibility and clear delineation of responsibilities. The
overall architecture is illustrated in Figure 3.
4.1 Data System
The Data System serves as the dedicated layer, responsible for the
curation, structured representation, and efficient indexing of the
corpus.
•Dataset Manager: Handles the life cycle of the research
corpus, including the ingestion of documents from a source
manifest, metadata tracking, and efficient indexed storage
within a relational database.
•Claims Manager: This module uses an LLM-driven process
to learn features, pre-compute evidence mapping (shared
by all baselines), and meta-data needed for execution of
baselines (e.g. probes associated with CIBER [37] baseline.
•Embedding Store: Maintains vector embeddings for seman-
tic retrieval of evidence chunks.
4.2 RAG System
The RAG System forms the core of the inference pipeline. It retrieves
the messaging schema and relies on a dedicated handler for each
of the RAG approaches to execute the respective RAG pipeline.
•Retriever Engine: This orchestrator coordinates the mes-
sage schema tags in the RAG system to fetch relevant meta-
data and evidence for each claim.•Veritable Module: The VERIRAG specific module that col-
lates the 11-point methodological assessment prompt snip-
pets for applicable checks based on research papers matched
with the claim evidence.
•LLM Interface: An abstraction layer that mediates all in-
teractions with the underlying language models. It handles
prompt formatting, response schema validation, timeouts,
and retries, and provides reliability and flexibility for model
adoption.
4.3 Control System
The Control System provides visibility into the system’s validation
performance and manages context-appropriate settings.
•Computation: For VERIRAG, it implements the dynamic
acceptance threshold mechanism ( 𝜏auto), adjusting the re-
quired level of evidentiary support based on pre-computed
features of the claim under review.
•Settings: Manages the framework’s hyperparameters (e.g.,
𝜆,𝛼), declarative workflows the system, and allows secure
authorization and handshake through environment settings.
•Analytics: This module supports the validation of system
outputs and the calculation of performance metrics for each
processed claim, helping to monitor accuracy and reveal
patterns in methodological weaknesses.
4.4 Implementation Notes
VERIRAG’s architecture is designed for both configurability and
extensibility. The framework’s persistence layer is managed by
a MySQL database. The data ingestion pipeline utilizes Google’s
Document AI service to parse the layout and structure of source
PDFs prior to chunking. To ensure reproducibility, our reference
implementation uses Llama-4-Maverick-17B-128E
-Instruct-FP8 for all analysis and generation tasks and
togethercomputer/m2-bert-80M-32k-retrieval for generating
evidence-chunk embeddings. The modular design readily permits
the substitution of any component. We discuss in detail the prompts
used in the current implementation in Appendix C.
5 Evaluations
We designed a detailed evaluation plan to assess VERIRAG’s ca-
pabilities. Our experiments were structured around three primary
objectives: (1) to validate the core component of our framework -
the Veritable checks auditor perform the sub-tasks reliably; (2) to
benchmark VERIRAG’s end-to-end performance against state-of-
the-art baselines on the under evolving information conditions; and
(3) to analyze the system’s efficiency and failure modes through
targeted ablation studies and qualitative analysis.
5.1 Experimental Setup
5.1.1 Datasets and Task Formulation. Our evaluation is grounded
in a corpus constructed to simulate the real-world challenges of sci-
entific evidence synthesis. To achieve this, we developed a unique,
temporally-structured evaluation that moves beyond static bench-
marks to assess how systems perform under evolving information
conditions.

VERIRAG ACM BCB’25, XXX XX–YY, 2025, Philadelphia,PA
Table 1: Summary of Baseline RAG Frameworks and their Prompt-Based Simulation in VERIRAG’s Evaluation.
Baseline System Core Principle VERIRAG Simulation Method
COT-RAG [18, 38] Improves reasoning by generating step-by-step analysis
before reaching a conclusion.The LLM is instructed to first identify relevant statements,
assess their support, and then provide a final verdict and
justification in a single pass.
SELF-RAG [2] Critiques retrieved passages for relevance and support
before generating a final answer.A two-turn prompt chain: the first turn generates struc-
tured critiques for each evidence chunk; the second turn
provides those critiques back to the LLM to synthesize a
final, critique-aware verdict.
FLARE [17] Actively decides if more information is needed by gen-
erating questions and retrieving new documents mid-
generation.A two-turn prompt chain: the first turn makes an initial
assessment on snippets and decides if a full-text review is
needed. If so, the second turn provides the full text of the
corresponding paper for a final verdict.
CIBER [37] Probes LLM consistency by asking about a claim from mul-
tiple angles (e.g., agree, conflict, paraphrase) and fusing
the responses.First the COT call is repeated (for primary claim state-
ment) and then three separate "probe" prompts are sent
to the LLM for each claim. The final verdict is computed
by applying the Dempster-Shafer (WBU) fusion rule as
described in the original research.
•Corpus and Claims: The final test set consists of 100 scien-
tific claims derived from an underlying corpus of 200 source
papers with groundtruth validated by two healthcare experts.
This test claims benchmark, detailed in Appendix A, are a
mix of 73 ‘simple‘ (single assertion) and 27 ‘composite‘ (mul-
tiple sub-claims) types and cover a range of Medical Subject
Headings (MeSH [ 27]), including Diseases ,Chemicals and
Drugs , and Psychiatry and Psychology .
•Temporal Scenarios (TY): To add realism to the evolving
healthcare knowledge base, we created four distinct evidence
subsets (TY0, TY1, TY3, and TY5). This longitudinal design is
engineered to stress-test each framework against the dynamic
and often messy lifecycle of scientific discovery—from initial
(and potentially flawed) findings (TY0), to the emergence
of conflicting evidence (TY1), and finally toward a broader
scientific consensus (TY3 and TY5). This design is critical
because it assesses a system’s ability not just to evaluate
a fixed set of facts, but to reason under uncertainty and
revise its assessment as the evidence base evolves—a key
weakness in current methodologically-blind RAG systems.
In Appendix A, we detail how the supporting dataset corpus
is built.
•Task: For each of the 100 claims, every system under evalu-
ation is tasked with producing a verdict for each of the four
temporal scenarios, resulting in 400 unique experimental
runs per system. During the evidence retrieval phase for a
given claim, we persist the associated paper information and
later include an evidence snippet only if its source paper is
part of the specific temporal dataset (TY0, TY1, TY3, or TY5)
being evaluated.
5.1.2 Quantitative Framework Calibration. As detailed in Appen-
dix B, the quantitative components of VERIRAG were calibrated
using an 60 claims from SciFact [ 36] as detailed in the Appendix B.
Human ratings for claim specificity and testability were used to
train a regression model that informs the Dynamic AcceptanceThreshold (𝜏auto). The HV score parameters ( 𝛼,𝜆) were then tuned
on this same dataset to optimize performance.
5.2 Intrinsic Component Validation
Before comparing the performance of the end-to-end VERIRAG sys-
tem with that of baselines, we validated the consistency of Veritable
checks results derived through NL inference using LLM.
Table 2: System-Expert Agreement for the Veritable Audit.
Agreement reflects the percentage of papers where two inde-
pendent expert LLMs agreed on the Pass/Fail score for each
check generated by our primary model.
Check Agreement Check Agreement
C1: Data Integrity 81.3% C7: Outlier Influence 70.3%
C2: Missing Data 82.4% C8: Confounding Control 95.6%
C3: Sample Rep. 91.2% C9: Source Consistency 95.6%
C4: Outcome Var. 81.3% C10: Effect Homogeneity 98.9%
C5: Estimation Val. 86.8% C11: Subgroup Consist. 92.3%
C6: Statistical Power 96.7% Overall Kappa 0.631
5.2.1 Veritable Auditor Validation. To assess the reliability and
reproducibility of our automated Veritable audit, we used two inde-
pendent, state-of-the-art "expert" LLMs (Gemini 2.5 Pro and Claude
4 Opus) to evaluate the output of our primary model. For resource
conservation, we randomly selected 50% of the data for validation.
This approach, where LLMs perform structured reasoning and cri-
tique, has shown promise in recent work [ 42]. We measured the
consistency between these two external expert models as a proxy
for the audit’s stability. A high degree of agreement suggests that
the checks are well-defined and less susceptible to the biases of
a single model. As shown in Table 2, the overall agreement was
substantial, with a Cohen’s Kappa of 0.63.

ACM BCB’25, XXX XX–YY, 2025, Philadelphia,PA Mohole et al.
5.3 System-Level Performance
Modern, long-context LLMs (e.g. Gemini 2.5 Pro [ 15]) now accept
large-token prompts, which have altered the RAG practice in pro-
duction systems [ 5]. Instead of training task-specific retrievers
or rerankers, most real world pipelines compose off-the-shelf em-
bedding APIs with multi-turn prompting that handles relevance
checks, critique, and verification. This coincides with recent work
that confirms that sophisticated behaviors once achieved through
fine-tuning can be reproduced—or even surpassed—via prompting
alone. Language-Agent Tree Search (LATS) [ 43], Chain-of-Note
(CoN) [ 40], and Self-Refine [ 20] all show that an LLM can generate
hypotheses, criticize its own outputs, and refine answers without
any custom model training. We follow the same paradigm: VERI-
RAG and every baseline (including SELF-RAG [ 2]) are executed as
prompt-only agents over a shared, pre-selected evidence set, letting
us focus on the reasoning logic itself rather than the quality of a
custom retriever. Table 1 summarizes the principles of each baseline
and our simulation method. In Appendix C we share details of the
prompts used for VERIRAG and each of the baselines.
Table 3: Primary Performance Metric (Macro F1-Score) on
Temporal Year (TY) Datasets. Best performance is in bold.
Model TY0 TY1 TY3 TY5
COT-RAG 0.4017 0.4118 0.4902 0.4819
Self-RAG 0.3711 0.3800 0.3912 0.5315
FLARE 0.3316 0.3251 0.3205 0.4919
CIBER 0.3749 0.4243 0.4409 0.5228
VERIRAG (Ours) 0.5325 0.5686 0.5932 0.6542
5.3.1 Primary Metrics: Verification Correctness. We measure the
performance of all systems on their ability to correctly classify
claims according to the fixed ground truth. For the purpose of bi-
nary evaluation against our Valid/Invalid ground truth, any system
verdict that did not explicitly support a claim (e.g., ’Neutral’, ’Un-
certain’, or ’Refutes’) was mapped to the ’Invalid’ class. As shown
in Table 3, across every TY scenario, VERIRAG achieves a higher
F1 score than all compared baselines. While acknowledging the
significant class imbalance in our dataset (approx. 1:20 Valid to
Invalid), we also calculated the Matthews Correlation Coefficient
(MCC). VERIRAG consistently outperformed all baselines on this
metric as well, achieving MCC scores in the range of [0.25, 0.40]
while COT-RAG had a range of [-0.13, 0.08], Self-RAG [-0.04, 0.08],
FLARE [0.08, 0.16], and CIBER [0.08, 0.18].
5.4 Process Efficiency
A key consideration for the practical deployment of RAG systems is
token cost. We measured the average number of tokens consumed
by each framework to verify a single claim. As shown in Figure 4,
VERIRAG’s structured, single-pass batch audit wherein we send
evidence snippets along with each paper’s methodology metadata
filtered by applicable checks is economically competitive with more
complex, multi-turn baseline frameworks like CIBER, making it a
viable option for real-world applications.
Figure 4: Comparison of average token consumption across
five RAG methodologies (COT-RAG, SELF-RAG, FLARE,
CIBER, and VERIRAG) evaluated under four temporal sce-
narios (TY0, TY1, TY3, TY5).
Table 4: Ablation Study on the TY3 Dataset.
Model Variation Macro F1-Score
Full VERIRAG 0.5932
Ablations:
w/o HV Score 0.3686
w/o Dynamic Threshold ( 𝜏auto) 0.2168
w/o Redundancy Penalty ( 𝑤𝑖) 0.5610
5.5 Ablation Studies
To isolate the contribution of VERIRAG’s key quantitative com-
ponents, we conducted ablation studies on the TY3 dataset. The
results in Table 4 confirm that each component provides a signifi-
cant performance uplift. Each of the datasets TY0, TY1, TY3, and
TY5 showed similar results. The removal of the Hard-to-Vary (HV)
Score logic and the Dynamic Threshold resulted in the most signif-
icant performance degradation. The Redundancy Penalty showed
a smaller impact on this test set, likely due to the limited overlap
in the curated evidence snippets. However, in a real-world setting
with a large, uncurated corpus, this penalty would be critical to
prevent systems from overweighting multiple reports of the same
finding.
5.6 Qualitative Analysis and Case Studies
To supplement our quantitative results, we conducted a qualitative
review of outcomes. Certain patterns, as highlighted in the case
studies below, were notable.

VERIRAG ACM BCB’25, XXX XX–YY, 2025, Philadelphia,PA
Case Study 1: Invalid Claim Detection
We analyze a claim from a retracted paper in the TY0 dataset
that reported "SHR lungs expressed higher levels of NGF, BDNF,
TrkA, and both isoforms of TrkB receptor mRNA compared to
age-matched WKY rats," which was published in a now-retracted
paper titled "Elevated Neurotrophin and Neurotrophin Receptor
Expression in Spontaneously Hypertensive Rat Lungs" [26].
Challenge: This claim is Invalid , yet multiple baseline systems
incorrectly verified it.
VERIRAG’s Success: VERIRAG correctly identified the claim as
Invalid in its TY0 analysis because it found that the paper:
•Failed to conduct a power analysis (violating C6).
•Lacked reporting on how missing data was handled.
•Lacked checks for statistical outliers (violating C7).
Case Study 2: False Positive Analysis
We examine a case where VERIRAG produced a false positive.
This error highlights a key challenge for AI fact-checking: claims
requiring direct visual data interpretation.
The Claim: "Hypoxia induces an inflammatory response in all
plaque phenotypes as seen in Figure 2B," from the retracted arti-
cle "Evidence for markers of hypoxia and apoptosis in explanted
human carotid atherosclerotic plaques".
Analysis Outcome:
•Ground Truth: Invalid.
•VERIRAG’s Verdict: Valid (False Positive).
Root Cause: The system produced a false positive because it
relied on the authors’ textual description of the figure rather than
analyzing the visual data itself. An independent review of Figure
2B shows that the inflammatory response was not statistically
significant in early plaques, making the claim’s use of the word
"all" an overgeneralization.
5.7 Limitations
Our experiments demonstrated VERIRAG’s success in improving
the nuanced understanding of the quality of research work beyond
semantic matching of evidence to assess the factuality of the claims,
but several limitations of our experiments warrant acknowledg-
ment. First, we evaluated VERIRAG on 100 curated biomedical
claims (400 verdicts across four temporal scenarios). This focused
scale is in line with the demanding annotation requirements of
high-fidelity, expert-annotated evaluations in specialized biomedi-
cal NLP tasks such as BIOSSES [ 32] semantic similarity benchmark
of 100 sentence pairs and statistical queries focused tasks such
as Google’s DataGemma [ 25] RAG models that ground answers
to 101 statistical queries during testing. While this size meets the
requirements of a controlled experimental setup to validate the
framework’s approach, further work is needed to explore larger,
automatically-curated corpora to validate the generalization of re-
sults seen in our experiments. Second, our dataset design, which
used retracted papers as the source for claims, resulted in a signif-
icant class imbalance with invalid claims dominating the dataset.
Third, our framework, like all LLM-based systems, is susceptible tohallucinations and biases. A related and significant challenge is the
lack of interpretability in the LLM’s reasoning process; it is difficult
to determine whether the model is strictly following instructions or
drawing upon its own parametric "world knowledge." The fluctuat-
ing performance of the baseline models is a clear manifestation of
this issue, where the LLM likely contextualized the evidence unpre-
dictably rather than adhering strictly to the provided text. While
VERIRAG’s structured auditing approach is designed to mitigate
this by forcing the LLM into a more constrained reasoning path,
the underlying challenge of ensuring trustworthy models remains
a foundational issue for all LLM-based applications. Finally, as high-
lighted in our case study our current implementation focuses solely
on textual evidence and does not perform multi-modal analysis of
figures or charts, which can contain critical information especially
in the context of healthcare research validation work [3].
6 Related Work
VERIRAG builds upon and extends three primary research areas:
(1) automated systems for maintaining scientific integrity, (2) spe-
cialized frameworks for claim validation in high-stakes domains,
and (3) the architectural evolution of RAG.
6.1 Automated Scientific Integrity
The movement to automate the detection of flawed research in-
cludes statistical heuristic tools like Statcheck [ 22] and GRIM [ 4],
and community-driven post-publication review platforms like Pub-
Peer [ 1]. These tools are often narrow in scope or cannot scale
effectively. VERIRAG distinguishes itself by integrating a compre-
hensive, automated audit directly into the generative workflow,
providing a quantitative, evidence-based decision in real time. It
assesses broader methodological issues like sample representative-
ness (𝐶3) or confounding control ( 𝐶8).
6.2 Claim Validation in Healthcare and Clinical
Specialized claim validation systems in healthcare often focus on
fact-checking against established knowledge bases or use methods
like Evidence-Based Medicine (EBM) hierarchies [ 29] and biblio-
metrics. These approaches face challenges with novel or contested
research and can conflate visibility with validity. VERIRAG intro-
duces a complementary approach by performing a deep, semantic
audit of the internal methodological characteristics of the evidence,
allowing for a granular assessment of novel claims where no estab-
lished reference exists.
6.3 Retrieval-Augmented Generation (RAG)
Recent advances in Retrieval-Augmented Generation (RAG) have in-
creasingly focused on improving factual accuracy and self-correction
mechanisms. Notable frameworks such as CRAG [ 39] introduce
reflective and evaluative modules that guide the model to critique
its own generation using special reflective tokens, while CRAG em-
ploys a retrieval evaluator to remove irrelevant evidence. Building
on the success of these mechanisms, Li et al. [ 19] demonstrated
strong performance by applying Self-RAG and CRAG to biomedical
fact-checking tasks, particularly in the context of COVID-19. More
structured approaches, like CoV-RAG [ 16], explicitly incorporate
a verification module that assesses the accuracy of the retrieved

ACM BCB’25, XXX XX–YY, 2025, Philadelphia,PA Mohole et al.
evidence and generated answers through a chain-of-verification
process. RAFTS [ 41] enhances claim verification by generating both
supporting and refuting arguments using retrieved evidence, allow-
ing fine-grained credibility assessments. Despite these advances,
existing RAG frameworks largely rely on internal or heuristic-based
critique and often lack an understanding of formal, domain-specific
standards. As a result, they cannot detect more subtle yet critical
issues like p-hacking ( 𝐶11) or selective reporting ( 𝐶9). The core con-
tribution of VERIRAG is the introduction of a formal, statistically
informed verification layer into the RAG pipeline. Our Veritable
transforms the vague instruction to "critique the source" into a
systematic, 11-point audit, combined with a quantitative synthesis
model (the HV score) and an adaptive decision threshold ( 𝜏auto).
7 Conclusion and Future Work
VERIRAG addresses a critical gap in modern AI systems by shifting
the objective of evidence synthesis from mere semantic relevance
to methodological rigor. Our key innovations—the Veritable audit
framework, the Hard-to-Vary (HV) score, and the dynamic accep-
tance threshold—create a system that quantitatively evaluates re-
search quality. Based on our evaluations the VERIRAG approach
consistently outperforms all baselines across temporal scenarios,
validating the importance of methodological assessment for reli-
able scientific claim verification. Future work will extend VERIRAG
along three trajectories. First, we will adapt our document schema
to additional biomedical subfields, enabling automated quality as-
sessment across specialized domains. Second, we will develop VERI-
RAG into an interactive assistant for manuscript preparation and
peer review, providing real-time methodological feedback. Finally,
we will explore community partnerships, such as with The Black
Spatula Project [ 33], to validate and refine our approach through
student and researcher engagement. Through these initiatives, we
envision VERIRAG evolving from a verification tool into a trusted
aid for improving scientific rigor at the point of new research.
Acknowledgments
This work was performed under the auspices of the U.S. Depart-
ment of Energy by Lawrence Livermore National Laboratory under
Contract DE-AC52-07NA27344 and was supported by the LLNL-
LDRD Program under Project No. 25-SI-001. LLNL-CONF-2007737-
DRAFT.
References
[1] 2012. PubPeer. https://pubpeer.com/. Post-publication peer review platform.
[2]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2023.
Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.
arXiv:2310.11511 [cs.CL] https://arxiv.org/abs/2310.11511
[3]Elisabeth M. Bik, Arturo Casadevall, and Ferric C. Fang. 2016. The Prevalence of
Inappropriate Image Duplication in Biomedical Research Publications. mBio 7, 3
(June 2016), e00809–16. doi:10.1128/mBio.00809-16
[4]Nicholas J. L. Brown and James A. J. Heathers. 2017. The GRIM test: A simple
technique detects numerous anomalies in the reporting of results in psychology.
Social Psychological and Personality Science 8, 4 (2017), 363–369. doi:10.1177/
1948550616673876
[5]Harrison Chase. 2022. LangChain . https://github.com/langchain-ai/langchain
Python framework for developing applications powered by large language mod-
els.[6]Svetlana Churina, Anab Maulana Barik, and Saisamarth Rajesh Phaye. 2024.
Improving Evidence Retrieval on Claim Verification Pipeline through Ques-
tion Enrichment. In Proceedings of the Seventh Fact Extraction and VERifica-
tion Workshop (FEVER) , Michael Schlichtkrull, Yulong Chen, Chenxi White-
house, Zhenyun Deng, Mubashara Akhtar, Rami Aly, Zhijiang Guo, Christos
Christodoulopoulos, Oana Cocarascu, Arpit Mittal, James Thorne, and Andreas
Vlachos (Eds.). Association for Computational Linguistics, Miami, Florida, USA,
64–70. doi:10.18653/v1/2024.fever-1.6
[7]Jacob Cohen. 1988. Statistical Power Analysis for the Behavioral Sciences (2nd ed.).
Routledge. doi:10.4324/9780203771587
[8]CONSORT Group. 2025. CONSORT 2025 Statement: updated guideline for
reporting randomised trials. EQUATOR Network. https://www.equator-
network.org/reporting-guidelines/consort/ Accessed June 26, 2025.
[9]David Deutsch. 2011. The Beginning of Infinity: Explanations That Transform the
World . Viking, New York.
[10] Asif Dhar, Bill Fera, and Leslie Korenda. 2023. Can GenAI help make health care
affordable? Consumers think so. Health Forward Blog, Deloitte Center for Health
Solutions (2023).
[11] Kerry Dwan, Carrol Gamble, Paula R. Williamson, Jamie J. Kirkham, and Re-
porting Bias Group. 2013. Systematic review of the empirical evidence of study
publication bias and outcome reporting bias - an updated review. PLoS One 8, 7
(July 2013), e66844. doi:10.1371/journal.pone.0066844
[12] B. Edelman and J. Skolnick. 2025. Valsci: an open-source, self-hostable literature
review utility for automated large-batch scientific claim verification using large
language models. BMC Bioinformatics 26 (2025), 140. doi:10.1186/s12859-025-
06159-4
[13] Andrew Gelman and Jennifer Hill. 2007. Data analysis using regression and
multilevel/hierarchical models . Cambridge University Press, Cambridge. https:
//www.cambridge.org/highereducation/books/data-analysis-using-regression-
and-multilevel-hierarchical-models/32A29531C7FD730C3A68951A17C9D983
[14] Rita González-Márquez, Luca Schmidt, Benjamin M. Schmidt, Philipp Berens,
and Dmitry Kobak. 2024. The landscape of biomedical research. Patterns 5, 6
(2024), 100968. doi:10.1016/j.patter.2024.100968
[15] Google AI. 2024. Long context | Gemini API | Google AI for Developers. https:
//ai.google.dev/gemini-api/docs/long-context Accessed: 2025-07-08.
[16] Bolei He, Nuo Chen, Xinran He, Lingyong Yan, Zhenkai Wei, Jinchang Luo,
and Zhen-Hua Ling. 2024. Retrieving, Rethinking and Revising: The Chain-
of-Verification Can Improve Retrieval Augmented Generation. arXiv preprint
arXiv:2410.05801 (2024).
[17] Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-
Yu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active Retrieval
Augmented Generation. arXiv:2305.06983 [cs.CL] https://arxiv.org/abs/2305.
06983
[18] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel,
Sebastian Riedel, and Douwe Kiela. 2021. Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks. arXiv:2005.11401 [cs.CL] https://arxiv.org/abs/
2005.11401
[19] Hai Li, Jingyi Huang, Mengmeng Ji, Yuyi Yang, and Ruopeng An. 2025. Use
of Retrieval-Augmented Large Language Model for COVID-19 Fact-Checking:
Development and Usability Study. Journal of medical Internet research 27 (2025),
e66098.
[20] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah
Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank
Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir
Yazdanbakhsh, and Peter Clark. 2023. Self-Refine: Iterative Refinement with
Self-Feedback. In Advances in Neural Information Processing Systems (NeurIPS) .
https://arxiv.org/abs/2303.17651
[21] J. Novoa, M. Chagoyen, C. Benito, F. J. Moreno, and F. Pazos. 2023. PMIDigest:
Interactive Review of Large Collections of PubMed Entries to Distill Relevant
Information. Genes 14, 4 (April 2023), 942. doi:10.3390/genes14040942
[22] Michèle B. Nuijten, Chris H. J. Hartgerink, Marcel A. L. M. van Assen, Sacha
Epskamp, and Jelte M. Wicherts. 2016. The prevalence of statistical reporting
errors in psychology (1985–2013). Behavior Research Methods 48, 4 (2016), 1205–
1226. doi:10.3758/s13428-015-0664-2
[23] Matthew J. Page, Joanne E. McKenzie, Patrick M. Bossuyt, Isabelle Boutron,
Tammy C. Hoffmann, Cynthia D. Mulrow, Larissa Shamseer, Jennifer M. Tetzlaff,
Elie A. Akl, Sue E. Brennan, Roger Chou, Julie Glanville, Jeremy M. Grimshaw,
Asbjørn Hróbjartsson, Manoj M. Lalu, Tianjing Li, Elizabeth W. Loder, Evan
Mayo-Wilson, Steve McDonald, Luke A. McGuinness, Lesley A. Stewart, James
Thomas, Andrea C. Tricco, Vivian A. Welch, Penny Whiting, and David Moher.
2021. The PRISMA 2020 statement: an updated guideline for reporting systematic
reviews. BMJ 372 (29 mar 2021), n71. doi:10.1136/bmj.n71 CC BY Open access.
[24] Judea Pearl. 2009. Causality: Models, Reasoning, and Inference (2nd ed.). Cambridge
University Press, Cambridge.
[25] Prashanth Radhakrishnan, Jennifer Chen, Bo Xu, Prem Ramaswami, Hannah Pho,
Adriana Olmos, James Manyika, and R.V.Guha. 2024. Knowing When to Ask–
Bridging Large Language Models and Data. arXiv preprint arXiv:2409.XXXX.

VERIRAG ACM BCB’25, XXX XX–YY, 2025, Philadelphia,PA
https://docs.datacommons.org/papers/DataGemma-FullPaper.pdf Introduces
Google’s DataGemma models for grounding LLMs in Data Commons.
[26] A Ricci, E Bronzetti, F Mannino, L Felici, C Terzano, and S Mariotta. 2004. El-
evated neurotrophin and neurotrophin receptor expression in spontaneously
hypertensive rat lungs. Growth Factors 22, 3 (Sep 2004), 195–205. doi:10.1080/
08977190412331284353 Retracted in: Growth Factors. 2024 Aug;42(3):146. doi:
10.1080/08977194.2024.2392373.
[27] FB Rogers. 1963. Medical subject headings. Bulletin of the Medical Library
Association 51, 1 (Jan 1963), 114–116.
[28] Donald B. Rubin. 1976. Inference and missing data. Biometrika 63, 3 (1976),
581–592. doi:10.1093/biomet/63.3.581
[29] David L. Sackett, William M. Rosenberg, J. A. Muir Gray, R. Brian Haynes, and
W. Scott Richardson. 1996. Evidence based medicine: what it is and what it isn’t.
BMJ 312, 7023 (jan 1996), 71–72. doi:10.1136/bmj.312.7023.71
[30] Carl Sagan. 1979. Broca’s Brain: Reflections on the Romance of Science . Random
House, New York.
[31] Joseph P. Simmons, Leif D. Nelson, and Uri Simonsohn. 2011. False-Positive
Psychology: Undisclosed Flexibility in Data Collection and Analysis Allows
Presenting Anything as Significant. Psychological Science 22, 11 (2011), 1359–
1366. doi:10.1177/0956797611417632
[32] Gizem Sogancioglu, Hakime Öztürk, and Arzucan Özgür. 2017. BIOSSES: a
semantic sentence similarity estimation system for the biomedical domain. Bioin-
formatics 33, 14 (2017), i49–i58. doi:10.1093/bioinformatics/btx238
[33] The Black Spatula Project Contributors. 2024. The Black Spatula Project. Open
Research Initiative. https://the-black-spatula-project.github.io/ An open ini-
tiative to investigate the potential of large language models (LLMs) to identify
errors in scientific papers.
[34] Together AI. 2025. Structured outputs - JSON Mode. https://docs.together.ai/
docs/json-mode. Accessed July 8, 2025.
[35] Erik von Elm, Douglas G. Altman, Matthias Egger, Stuart J. Pocock, Peter C.
Gøtzsche, Jan P. Vandenbroucke, and STROBE Initiative. 2008. The Strengthening
the Reporting of Observational Studies in Epidemiology (STROBE) statement:
guidelines for reporting observational studies. Journal of Clinical Epidemiology
61, 4 (apr 2008), 344–349. doi:10.1016/j.jclinepi.2007.11.008
[36] David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen,
Arman Cohan, and Hannaneh Hajishirzi. 2020. Fact or Fiction: Verifying Scientific
Claims. In Proceedings of the 2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP) , Bonnie Webber, Trevor Cohn, Yulan He, and
Yang Liu (Eds.). Association for Computational Linguistics, Online, 7534–7550.
doi:10.18653/v1/2020.emnlp-main.609
[37] Siyuan Wang, James R. Foulds, Md Osman Gani, and Shimei Pan. 2025. LLM-based
Corroborating and Refuting Evidence Retrieval for Scientific Claim Verification.
arXiv:2503.07937 [cs.AI] https://arxiv.org/abs/2503.07937
[38] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei
Xia, Ed Chi, Quoc Le, and Denny Zhou. 2023. Chain-of-Thought Prompting
Elicits Reasoning in Large Language Models. arXiv:2201.11903 [cs.CL] https:
//arxiv.org/abs/2201.11903
[39] Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. 2024. Corrective Retrieval
Augmented Generation. arXiv:2401.15884 [cs.CL] https://arxiv.org/abs/2401.
15884
[40] Wenhao Yu, Hongming Zhang, Xiaoman Pan, Kaixin Ma, Hongwei Wang, and
Dong Yu. 2024. Chain-of-Note: Enhancing Robustness in Retrieval-Augmented
Language Models. In Proceedings of the 2024 Conference on Empirical Methods in
Natural Language Processing (EMNLP) . https://arxiv.org/abs/2311.09210
[41] Zhenrui Yue, Huimin Zeng, Lanyu Shang, Yifan Liu, Yang Zhang, and Dong
Wang. 2024. Retrieval Augmented Fact Verification by Synthesizing Contrastive
Arguments. arXiv preprint arXiv:2406.09815 (2024).
[42] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu,
Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang,
Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-Judge with MT-Bench
and Chatbot Arena. arXiv:2306.05685 [cs.CL] https://arxiv.org/abs/2306.05685
[43] Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang, and Yu-
Xiong Wang. 2024. Language Agent Tree Search Unifies Reasoning, Acting and
Planning in Language Models. In Proceedings of the 41st International Conference
on Machine Learning (ICML) . https://arxiv.org/abs/2310.04406
[44] YI Zhu, Jing Dong, and Henry Lam. 2022. Uncertainty Quantification and
Exploration for Reinforcement Learning. arXiv:1910.05471 [cs.LG] https:
//arxiv.org/abs/1910.05471Appendix A Corpus Construction and Ground
Truth Protocol
This appendix details the systematic, multi-phase protocol for con-
structing the evaluation corpus based on 200 research papers and
establishing ground truth for the claims related to these research
topics. The methodology is designed to test a system’s ability to
determine the fundamental methodological validity of a scientific
claim within an evolving evidence landscape.
A.1 Phase 1: Seeding with Flawed Research
The corpus was seeded with an initial set of 40 primary research
articles identified as having been retracted from PubMed for sub-
stantive methodological or ethical reasons. The selection criteria
were designed to isolate articles where the retraction was due to
issues with the data quality or analysis, rather than administrative
errors.
A.2 Phase 2: Claim Extraction and Ground
Truth Annotation
This phase establishes the fixed, unchanging ground truth for the
claims used in the evaluation via a multi-stage, human-in-the-loop
process.
Figure 5: The claim generation interface used by annotators
•Crowdsourced Claim Generation: We developed a cus-
tom web interface (see Figure 5) to guide annotators. For
each retracted paper, the interface presented a summary of
the original paper’s abstract and the official retraction notice.
Annotators were then prompted with eight distinct, human-
friendly patterns (e.g., "Primary Causal Finding," "Quantita-
tive Effect Size") to generate a diverse set of candidate claims
based on the provided context.
•Author Filtering: Of the crowdsourced claims reviewed, ap-
proximately 40% were excluded from analysis. These claims
were deemed unsuitable for verification because they were
either ambiguous or limited to verifying authorship and aca-
demic disciplines rather than substantive assertions.
•Expert QC and Test Set Finalization: The remaining
claims were independently evaluated by two healthcare ex-
perts. Their task was to assign a final ground truth label
(‘Valid’, ‘Invalid’, or ‘Uncertain’). The QC process had a sub-
stantial inter-rater reliability with a Gwet’s AC1 score of

ACM BCB’25, XXX XX–YY, 2025, Philadelphia,PA Mohole et al.
0.9165. Only claims where the two experts reached a con-
sensus after discussing disagreements and had either ‘Valid’
or ‘Invalid’ labels were included in the final test set of 100
claims. This final label is fixed and does not change in sub-
sequent evaluation stages. Below we list the sample claims
from our dataset.
(1)Restraint of miR-587 by lncRNA PANTR1 promotes War-
burg effect in HCC. DOI: https://doi.org/10.1155/2021/1736819
(2)The lncRNA PANTR1 upregulates BCL2A1 expression.
DOI: https://doi.org/10.1155/2021/1736819
(3)HIF-1𝛼was upregulated in carotid specimens with re-
spect to controls, with a significant p-value of less than
5 percent. DOI: https://www.jvascsurg.org/article/S0741-
5214(10)01344-3/fulltext
(4)The level of antioxidant proteins decreased after alloferon
treatment. DOI: https://link.springer.com/article/10.1007/
s11010-008-9746-0
(5)Aged serum contains factors that condition young niche
cells to induce a fivefold accumulation of young HSPCs.
DOI: https://www.nature.com/articles/nature08749
A.3 Phase 3: Construction of Temporal
Evidence Contexts
To simulate the progression of scientific discourse, the evidence
available to the systems was varied over time. The ground truth of
the claims remains fixed, but the context for verifying them evolves.
•Evidence Augmentation: For each original retracted ar-
ticle, we identified high-quality refuting papers and sub-
sequent research articles on the closely related topic via
systematic citation analysis.
•Temporal Year (TY) Datasets: The augmented corpus was
partitioned into four distinct evidence sets, each representing
the information available at a specific time.
–TY0 (New Information): Contains only the full text of
the source retracted papers.
–TY1 (Refuting Information): Augments the TY0 set
with high-quality refuting articles.
–TY3 (Broader Analysis): Expands the TY1 set by includ-
ing articles published on the topic after the availability of
both the TY0 and TY1 corpus papers.
–TY5 (Settled Science): Contains only the refuting papers
and the subsequent articles, excluding the original flawed
work. In all, our corpus evolved to a total of 200 research
papers across four temporal datasets.
Appendix B Expert Calibration and
Component Validation Protocol
This appendix outlines the protocol for calibrating VERIRAG’s quan-
titative framework. To ensure the integrity of the main evaluation,
all activities described here were performed on an external dataset
of 60 claims derived from the SciFact benchmark [36].B.1 Phase 1: Generation of a Synthetic
Calibration Dataset
To train our models, we first created a synthetic dataset where
methodologically sound claims were paired with simulated flawed
evidence.
•Data Source: We began with claims from the SciFact dataset.
•Simulating Methodological Flaws: For each claim, we
used an LLM to generate a plausible, but hypothetical, study
methodology. Crucially, the LLM was instructed to randomly
mark 2-4 of the 11 Veritable Checks as "Fail" in a correspond-
ing hypothetical audit report. This process created a dataset
of claims paired with evidence containing specific, known
methodological weaknesses.
B.2 Phase 2: Human-in-the-Loop Data
Collection
The synthetic dataset was then loaded into our custom VERIRAG
Expert Calibration tool (Figure 6) for human annotation. For each
of the 60 claims, raters performed a two-part task:
(1)Part A: Claim Boldness Rating: Raters first evaluated the
claim itself to calibrate the Dynamic Acceptance Threshold
(𝜏auto). They rated its Specificity and Testability on a 1-10
scale and assessed its "Burden of Proof" on a 3-point Sagan’s
Scale (Low, Medium, High).
Figure 6: The expert calibration interface for collecting data
to train the quantitative models

VERIRAG ACM BCB’25, XXX XX–YY, 2025, Philadelphia,PA
(2)Part B: Hypothetical Audit Verdict: Raters were then
shown the hypothetical methodology and the simulated
flawed audit report. Based on this flawed evidence, they pro-
vided a final verdict (‘Support‘, ‘Contradict‘, or ‘Uncertain‘)
and a confidence score (0-100).
B.3 Phase 3: Model Calibration
The submissions collected via the calibration tool from our peer
researchers were used to train and tune VERIRAG’s quantitative
models.
•Dynamic Threshold Model: The ratings for Specificity,
Testability, and Burden of Proof (from Part A) were used as
features to train the Ridge Regression model that predicts
the base value for 𝜏auto.
•HV Score Parameter Tuning: The verdict and confidence
ratings (from Part B) were used as the ground truth for tuning
the HV score’s aggregation parameters ( 𝛼,𝜆) via a brute-
force grid search, optimizing for predictive accuracy against
the human-provided verdicts.
Appendix C LLM Prompt and Schema
Engineering
We realize that the reliability and reproducibility of our framework
depend on manually engineered, structured prompts and support
from the host for the Meta Llama 4 model’s structured output
[34]. We highlight key prompts, which underlie the system’s core
analytical tasks.
Prompt 1: Paper & Claim Feature Generation
A. Paper Methodological Analysis (JSON Generation)
This prompt instructs the LLM to analyze a paper’s full text and
generate a structured JSON object containing the signals needed
for the Veritable audit.
### CORE TASK ###
Your task is to act as an expert methodological reviewer.
Analyze the
provided paper content and generate a single, valid JSON
object that
strictly adheres to the REQUIRED_JSON_SCHEMA.
### REQUIRED_JSON_SCHEMA ###
```json
{...schema_json_string...}
VERITABLE_CHECK_DEFINITIONS
This is the context for the veritable_check_signals
part of the schema...
{...veritable_checks_string...}
CRITICAL RULE
For every Veritable Check (C1-C11) you must first set
is_applicable.
If is_applicable is false, set objective_analysis to "N/A".If is_applicable is true, provide a concise,
factual analysis...
PAPER_CONTENT
{...paper_chunks_json...}
B. Claim Feature & Probe Generation
For each claim, this prompt generates key features used by the
quantitative framework, including boldness metrics and probe
questions for baselines.
TASK
As a meticulous scientific analyst, analyze the following
research claim
and generate a structured JSON object with ALL of the
following attributes.
Claim to Analyze:
"""
{claim_text}
"""
Attribute Requirements:
specificity_rating:
Rate the claim 's specificity... (1-10).
testability_rating:
Rate the claim 's testability... (1-10).
claim_type: Classify as 'simple 'or'composite '.
topic: Categorize into exactly ONE MeSH topic...
evidence_confidence_criteria: What is the
minimum standard of evidence
(Settled Science, Robust Study, Plausible Evidence)
you would need?
ciber_probe_questions: Generate exactly three probing
questions...
Figure 7: Prompts for the offline data preparation pipeline
(M1 & M2). Part A creates the methodological JSON analy-
sis for each paper. Part B enriches each claim with features
for VERIRAG’s dynamic threshold ( 𝜏textauto ) and probe ques-
tions for the CIBER and FLARE baselines.
Prompt 2: VERIRAG Batch Audit
This prompt executes the core VERIRAG audit. The LLM receives
the claim and a batch of evidence papers, each packaged with its
pre-computed methodological JSON analysis.
CORE TASK
You will be given a scientific claim and a JSON array named
'papers_to_audit '.
Each object in this array represents a single evidence
paper and contains its
ID, its structured methodological analysis

ACM BCB’25, XXX XX–YY, 2025, Philadelphia,PA Mohole et al.
(paper_json_content), and
relevant text chunks (evidence_text_chunks).
Your task is to iterate through EACH paper object...
and for each one:
Determine Stance: Decide if the paper 'Supports ',
'Refutes ', or is
'Neutral 'towards the claim.
Perform Audit: Based on the paper 's'paper_json_content ',
evaluate
ONLY the checks where 'is_applicable 'is true. For each of
these, provide a 'Pass ','Fail ', or 'Uncertain 'score
and a brief reasoning.
CLAIM TO VERIFY
"{claim_text}"
PAPERS TO AUDIT
{...paper_snippet_and_methodology_analysis_json_array...
}
REQUIRED JSON OUTPUT
Your entire response MUST be a single JSON object. This
object must contain a
key 'all_papers_audit 'which is an array of
audit results...
Figure 8: The primary online prompt for the VERIRAG sys-
tem (M5). It directs the LLM to perform a stance detection
and a formal audit for every evidence paper simultaneously,
leveraging the pre-computed JSON analysis to inform its
judgments.
Prompt 3: Baseline Example (COT-RAG)
This is a single-turn Chain-of-Thought baseline. It instructs the
LLM to perform step-by-step reasoning before arriving at a verdict,
based only on the provided evidence snippets.
### TASK ###
Analyze whether the provided evidence supports, refutes,
or you 've neutral
verdict about claim. Use ONLY information
explicitly stated in the evidence
below - do not rely on any external knowledge.
### Claim to Verify
"{claim_text}"
### Evidence (Your ONLY source)
---
{evidence_snippets}
---
### Analysis Requirements
1. Identify specific statements in the evidence that
relate to the claim2. Assess whether these statements support, contradict,
or are neutral
3. Base your verdict solely on what is explicitly
stated in the evidence
4. If the evidence doesn 't address key aspects, mark as
'Unverifiable '
### REQUIRED JSON ###
Respond with only the JSON object per the schema.
Figure 9: Prompt for the Chain-of-Thought (COT-RAG) base-
line. This simulation evaluates the LLM’s ability to perform
structured reasoning in a single pass.
Prompt 4: Baseline Example (SELF-RAG)
A. Step 1: Critique Generation The first turn in the two-step
SELF-RAG baseline asks the LLM to act as a critic, generating
structured critiques for each retrieved evidence snippet.
TASK
Critique each evidence passage based STRICTLY on its
explicit content.
Assess relevance and support using only what is stated in
the text.
Claim to Evaluate
"{claim_text}"
Evidence Passages
{evidence_snippets_with_ids}
Critique Guidelines
'Relevant '= passage explicitly discusses topics directly
related to the claim
'Fully Supported '= passage contains explicit statements
that directly
validate the claim...
B. Step 2: Synthesis The second turn provides the original evi-
dence andthe critiques from Step 1, asking the LLM to synthesize
them into a final verdict.
TASK
Synthesize a verdict using ONLY the evidence and critiques
below.
Claim
"{claim_text}"
Original Evidence
{evidence_snippets_with_ids}
Your Critiques
{critique_results_array}
Synthesis Rules
Only use passages marked as 'Relevant 'in your critiques

VERIRAG ACM BCB’25, XXX XX–YY, 2025, Philadelphia,PA
'Valid '= at least one 'Fully Supported 'critique with no
contradictions...
Figure 10: An example of a two-turn baseline prompt for
SELF-RAG. This structured, multi-step process allows for
more complex reasoning than a standard RAG pipeline but
does not use the formal Veritable checks.
Prompt 5: Baseline Example (FLARE)
A. Step 1: Initial Assessment
The first turn in the FLARE baseline prompts the LLM to make
an initial assessment and decide if a full-text review of a specific
paper is necessary to confidently verify the claim.
### TASK ###
Use only what is stated in the evidence. Do not fill in
gaps with external
knowledge. Analyze the claim and decide if you need a
full-text review of
one of the source papers to meet the suggested evaluation
standard.
### Claim to Verify
"{claim_text}"
### Suggested Evaluation Standard
"{suggested_standard}"
### Evidence Snippets (from papers: {paper_ids_list})
---
{evidence_snippets}
---
### Assessment Guidelines
1. Form initial verdict based ONLY on explicit statements
in the snippets
2. Evaluate if evidence meets the required standard
3. Request full review ONLY if snippets suggest the full
paper contains the
needed explicit evidence. Your value MUST match one of
the paper ids EXACTLY.
### REQUIRED JSON ###
Respond with ONLY a JSON object per the schema.
B. Step 2: Final Verdict
If a full review was requested, this second-turn prompt provides
the full text of the chosen paper and asks for a final, more informed
verdict.
### TASK ###
Your initial assessment was based on limited snippets. You
requested a full
review of paper "{paper_id_for_full_review}". Now,
using BOTH the original
snippets and the FULL TEXT of the requested paper,
provide a final evaluation.### Claim
"{claim_text}"
### Original Evidence Snippets
---
{step1_evidence_snippets}
---
### Full Text of Paper `{paper_id_for_full_review} `
---
{full_paper_text}
---
### REQUIRED JSON ###
Your entire response MUST be only the JSON object.
Figure 11: Prompts for the two-step FLARE baseline. This
simulates active retrieval, where the model can introspect
on the initial evidence and request more context before com-
mitting to a final verdict.
Prompt 6: Baseline Example (CIBER)
The CIBER baseline sends multiple "probe" questions to the LLM
to test for consistency. This is the generic template used for each
probe. The ‘{probe_question}‘ placeholder is dynamically filled
with an agree, conflict, or paraphrase question generated during
the offline feature-creation step.
### Probe Question
{probe_question}
### Evidence Snippets (Your ONLY source)
---
{evidence_snippets}
---
### Instructions
1. Answer based SOLELY on explicit statements
in the evidence above.
2. Provide a brief justification for your verdict,
citing the evidence.
### REQUIRED JSON ###
Respond with ONLY a JSON object with your verdict and
justification.
Figure 12: The generic prompt for the CIBER baseline. The
system sends multiple requests using this template, each
with a different probe question, and then fuses the verdicts
to arrive at a final consistency-checked conclusion.