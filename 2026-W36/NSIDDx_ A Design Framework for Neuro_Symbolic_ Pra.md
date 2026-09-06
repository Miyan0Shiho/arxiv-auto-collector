# NSIDDx: A Design Framework for Neuro-Symbolic, Practitioner-First Differential Diagnosis in Low-Resource Settings

**Authors**: Aarav Singh

**Published**: 2026-08-31 18:58:59

**PDF URL**: [https://arxiv.org/pdf/2609.00256v1](https://arxiv.org/pdf/2609.00256v1)

## Abstract
LLM-based diagnostic systems achieve high semantic accuracy on benchmarks, but open-ended evaluation on clinically uncommon presentations reveals a systematic gap between headline accuracy and verifiable clinical reliability. We evaluate an LLM+rare-disease-RAG pipeline across two cohorts and show that the paradigm produces confident outputs that are frequently unverifiable and systematically resistant to clinician interrogation. We present NSIDDx (Neuro-Symbolic Integrated Differential Diagnosis System), a design framework arguing that DDx systems in low-resource settings must treat the clinician as an active reasoning agent. We instantiate this through a neuro-symbolic pipeline with ternary symptom encoding, contradiction detection, audit strings, and practitioner override - running offline on consumer hardware. We distill five design principles for clinician-in-the-loop clinical NLP and invite the prospective studies needed to validate the claim at scale.

## Full Text


<!-- PDF content starts -->

NSIDDx: A Design Framework for Neuro-Symbolic, Practitioner-First
Differential Diagnosis in Low-Resource Settings
Aarav Singh
IIIT Naya Raipur
aarav24101@iiitnr.edu.in
Abstract
LLM-based diagnostic systems achieve high
semantic accuracy on benchmarks, but open-
ended evaluation on clinically uncommon pre-
sentations reveals a systematic gap between
headline accuracy and verifiable clinical re-
liability. We evaluate an LLM+rare-disease-
RAG pipeline across two cohorts and show
that the paradigm produces confident out-
puts that are frequently unverifiable and sys-
tematically resistant to clinician interrogation.
We presentNSIDDx(Neuro-Symbolic Inte-
grated Differential Diagnosis System), a de-
sign framework arguing that DDx systems in
low-resource settings must treat the clinician
as an active reasoning agent. We instantiate
this through a neuro-symbolic pipeline with
ternary symptom encoding, contradiction de-
tection, audit strings, and practitioner override
— running offline on consumer hardware. We
distill five design principles for clinician-in-
the-loop clinical NLP and invite the prospec-
tive studies needed to validate the claim at
scale.
1 Introduction
Differential diagnosis is among the most cogni-
tively demanding tasks in clinical medicine. A
general practitioner must retrieve relevant disease
knowledge, weigh present and absent symptoms,
consider rare conditions, and produce a reasoned
conclusion — often in under fifteen minutes with-
out specialist support. In low-resource settings,
this load falls on a single practitioner frequently
without any decision support tool. They do not
need a system that is right eighty percent of the
time if they cannot tell which twenty percent to
distrust. They need apractitioner-firstdesign
whose reasoning they can follow, augment, and
whose uncertainty is surfaced rather than con-
cealed.
Large language models have shown strong per-
formance on clinical benchmarks (Singhal et al.,2023; Nori et al., 2023), yet the clinician receiv-
ing an LLM-based diagnostic output gets an an-
swer, not a reasoning process they can interrogate,
modify, or reject on clinical grounds — a limita-
tion that persists even in KG-grounded pipelines
(Chandak et al., 2023; Gargano et al., 2024; Wang
et al., 2025b) where the graph informs genera-
tion but remains invisible to the clinician. Singhal
et al. (2023) themselves note that strong bench-
mark numbers conceal key gaps when models are
evaluated by clinicians rather than automated met-
rics — a finding our open-ended evaluation on 750
CUPCase cases independently reproduces: 75.2%
semantic accuracy at DDx@5 collapses to 6.6%
exact match, and 83.5% of rare scanner outputs
carry no tokenically verifiable (verifiable by words
(token) overlapping between disease names) rela-
tionship to the ground-truth diagnosis. The clin-
ical decision support literature consistently iden-
tifies automation bias and alert fatigue as major
barriers to adoption (Goddard et al., 2012). This
is not a failure of accuracy; it is a failure of design
philosophy.
Our Position.We argue that for low-resource
clinical settings, the most critical design con-
straint for diagnostic AI is not accuracy alone, but
tractability of disagreement: a system whose rea-
soning is transparent, auditable, and modifiable by
the clinician may in practice be more useful than
a more accurate system whose failures are invisi-
ble — because it invites collaboration rather than
anchoring.
This paper presentsNSIDDx, a neuro-symbolic
differential diagnosis pipeline built around that
philosophy. NSIDDx uses a large language model
for what it does well — synthesising clinical nar-
ratives, generating structured reasoning chains,
producing readable explanations — while inde-
pendently validating that reasoning through a sym-
bolic layer grounded in HPO/MONDO (Gargano
arXiv:2609.00256v1  [cs.CL]  31 Aug 2026

et al., 2024) and PrimeKG (Chandak et al., 2023).
The practitioner can read the Propositional Logic
(PL) audit string, inspect the graph, and disagree:
they can add a symptom the model missed, remove
a candidate they consider implausible, and rerun
the scoring pipeline. The loop is not closed with-
out them.
Our contributions are two-fold:
(A) Position and Design Principles.We dis-
till five design principles for clinician-in-the-loop
clinical NLP — surface contradiction, preserve
negation structurally, make override cheap, pro-
vide multi-level explanation, and design for offline
deployment — intended as actionable guidelines
for the community.
(B) System Instantiation and Failure Char-
acterisation.We present NSIDDx as a con-
crete instantiation of these principles, built
around a ternary symptom encoding scheme
(+1present,−1explicitly denied,0un-
recorded), a dual-formula scoring engine combin-
ing a contradiction-sensitive matrix score and a
coverage-penalising hybrid score, and a Negation-
Retaining Phenotype Graph that renders absent
findings as structural nodes rather than silently
discarding them. We evaluate this pipeline
under open-ended conditions with no multiple-
choice scaffolding, characterising systematic fail-
ure modes — vocabulary mismatch, confidence-
without-grounding, and negation inversion — that
make clinician intervention a routine requirement
rather than an edge case. NSIDDx’s symbolic
transparency layer is proposed as the architectural
response. We do not claim diagnostic superiority
— our contribution is architectural.
We acknowledge that no user study or clinical
validation has been conducted. The argument for
clinician involvement is theoretical and demon-
strated by one case study, and explicitly invites
prospective validation.
2 Related Work
LLM-based clinical diagnosis.Large language
models have demonstrated strong performance on
medical benchmarks (Singhal et al., 2023; Nori
et al., 2023). Health-LLM (Yu et al., 2025) ex-
tends this with a two-pass RAG pipeline and XG-
Boost classifier for personalised prediction, but the
clinician receives a label with no inspectable logic
chain and no mechanism for override.Human-AI collaborative diagnosis.AMIE (Tu
et al., 2025) shows conversational AI can match
physician accuracy in OSCE studies but does not
expose an inspectable reasoning chain or practi-
tioner override mechanism. In low-resource set-
tings, the ability to interrogate and correct AI sug-
gestions may be more valuable than raw accuracy.
Knowledge graph-guided iterative diagnosis.
MedKGI (Wang et al., 2025a) models multi-step
diagnosis as an iterative conversational loop using
information gain over KG subgraphs. It does not
model absent symptoms — negative findings are
neither elicited nor encoded, and clinicians can-
not see which absent symptoms are relevant to the
differential. NSIDDx is complementary: where
MedKGI optimises automated history-taking over
positive findings, NSIDDx makes negation ex-
plicit and inspectable after history collection.
Knowledge graph grounding for prognosis.
Wang et al. (2025b) fine-tune lightweight LLMs
on PrimeKG-scaffolded reasoning paths for next-
visit prediction, encoding visits as binary ICD-9
vectors — structurally unable to represent explicit
symptom denials (x i=−1). The reasoning chain
is not inspectable and absent findings are implicit.
Our two-cohort evaluation quantifies this opacity:
41.6% of DDx@5 semantic hits on CUPCase are
tokenically unverifiable without an audit trail.
Neuro-symbolic explainability.Mondal et al.
(2025) and Lu et al. (2025) apply LNN-based
methods to explainable diabetes prediction, expos-
ing auditable reasoning with negation support, but
operate on low-dimensional tabular data with stat-
ically defined rules and cannot parse unstructured
clinical narratives.
Rare disease detection.RADAR (Kim et al.,
2025) applies FAISS to rare disease retrieval in
brain MRI; Zelin et al. (2024) explore knowledge-
guided RAG for rare disease diagnosis. NSIDDx’s
scanner differs: it operates over clinical text, is
grounded in ZebraMap (Islam et al., 2026), and
scores candidates through the same dual-source
symbolic pipeline as the primary DDx, offline
without API access.
White-box diagnostic systems.Ada DX (Ron-
icke et al., 2019) demonstrates the value of accept-
ing present and absent findings with full practi-
tioner override, but operates as a probabilistic ex-
pert system on structured symptom input rather

System N O C U K
Health-LLM (Yu et al., 2025)✗ ✗ ✗ ✓ ✓
MedKGI (Wang et al., 2025a)✗ ✗ ✗ ✓ ✓
LNN (Mondal et al., 2025; Lu et al., 2025)✓ ✗ ✓ ✗ ✗
Ada DX (Ronicke et al., 2019)✓ ✓ ✓ ✗ ✓
AMIE (Tu et al., 2025)✗ ✗ ✗ ✓ ✗
NSIDDx❜ ✓ ✓ ✓ ✓
Table 1: Feature comparison. Columns:Negation
modelling,Override interface, offlineCapable,
Unstructured text input,KG grounding.✓full;
❜partial (feature exists but with major constraints:
e.g., negation requires manual entry, override is
single-direction, or offline mode excludes some
functionality);✗none.
than free-text narratives.
Feature comparison.Table 1 situates NSIDDx
among prior systems. The claim is not individ-
ual feature novelty but that thespecific integra-
tionof all five dimensions — negation modelling,
override interface, offline capability, unstructured
text input, and KG grounding — is novel in neuro-
symbolic clinical NLP.
3 System Architecture
NSIDDx is a modular pipeline with five core
stages.
Data sourcing.Knowledge base files are
sourced from HPO (hp.obo, phenotype.hpoa),
MONDO (mondo.owl), and PrimeKG
(kg.feather). HPO terms are extracted into a
symptom catalog; unannotated modifier and
inheritance terms are excluded. MONDO cross-
references unify OMIM and ORPHA identifiers;
entries with zero present-symptom annotations
are dropped, yielding roughly 10,800 disease
profiles. For PrimeKG, disease-phenotype edges
are bridged to HPO IDs; disease-disease edges are
excluded. Over 9,000 phenotype-bearing entities
enter the scoring pipeline. The NSIDDx pipeline,
including all preprocessing scripts and filter
thresholds, is publicly available athttps://
github.com/joetheguide2/NSIDDX-.
3.1 Stage 1: Clinical History Taking and
Summarisation
Clinical history is gathered via an LLM-driven
conversational agent and condensed into a struc-
tured summary preserving both positive and nega-
tive findings.3.2 Stage 2: Symptom Extraction and HPO
Resolution
Symptom extraction passes the clinical sum-
mary through a two-step pipeline. First, the
LLM reformats the summary into a structured
format withPresence of:andDenies:
prefixes. Second, surface forms are resolved
to HPO IDs via MedSpaCy TargetRule (lit-
eral) or Abhinand/MedEmbed-large-v0.1
(semantic) matching; each is tagged with its
resolution method. The result is two typed
sets:patient_hpo_ids(present) and
absent_hpo_ids(explicitly denied).
Disease name resolution uses exact then sub-
string match, preferring diseases with symptom
profiles over umbrella terms like “syndrome”.
3.3 Stage 3: LLM Differential Diagnosis and
PL Generation
The LLM differential diagnosis stage sends the
clinical summary together with the confirmed
present and explicitly denied symptom lists
to the LLM under a structured DDx prompt.
The prompt enforces a fixedEvidences→
Reasoning→Diagnosisformat per entry,
with anABSENT:marker for denied symptoms.
Propositional Logic (PL) strings — case-
specific conjunctions of symptom propositions im-
plying a diagnosis — are generated per diag-
nosis, either symbolically (from HPO-matched
symptoms) or via the LLM. Propositional state-
ments use absent symptoms as negated propo-
sitions (e.g.,¬(Raynaud phenomenon)). These
strings serve as practitioner-readable audit trails
for each diagnostic candidate.
3.4 Stage 4: Ternary-Hybrid Scoring Engine
The scoring engine runs two scoring formu-
las — thematrix scoreand thehybrid score
— independently against each knowledge source
(HPO/MONDO, PrimeKG), yielding independent
scores per candidate disease. Both formulas are
applied to all sources.
Score Formula 1: Matrix Score.A patient
symptom vectorV p∈ {−1,0,+1}Nis built from
the extracted HPO IDs:+1for confirmed present,
−1for explicitly denied,0for unrecorded. A dis-
ease profile vectorA j∈ {−1,0,+1}Nencodes
expected phenotypes. The score is the normalised

dot product:
Smatrix
j =Vp·Aj
|Aj|∈[−1,+1]
where|A j|is the disease’s profile size. Negative
scores indicate contradictions — the patient denies
a required symptom or presents one the disease
expects absent. Both formulas use standard mea-
sures (dot product, Jaccard); their specific combi-
nation with equal weights is novel.
Score Formula 2: Hybrid Score.The hybrid
score measures how well a disease explains the pa-
tient’s positive findings:
Shybrid
j = 0.5×|A∩B|
|A|+0.5×|A∩B|
|A∪B|∈[0,+1]
The inclusion term (|A∩B|/|A|) rewards covering
most of the patient’s symptoms; the Jaccard term
(|A∩B|/|A∪B|) penalises large profiles with only
incidental symptom overlap.
3.5 Stage 5: Negation-Retaining Phenotype
Graph and Rare Disease RAG Scanner
The negation graph stage builds aNegation-
Retaining Phenotype Graph— a directed multi-
graph using NetworkX from the current patient
payload and the last scoring output. The graph
contains five node types: present symptoms
(HPO-resolved), absent symptoms (explicitly de-
nied), past medical history, family history, and
DDx candidates. Absent symptom nodes remain
structurally present in the graph — visually dis-
tinguishable from confirmed symptoms — giving
the practitioner a direct view of pertinent negatives
alongside positive findings.
Edge types.The graph renders four edge
types from PrimeKG:explains(dis-
ease→symptom);presents(symptom
→disease);linked_to(symptom co-
occurrence); andphenotype_modifier_of
(modifier→base).Explainsedges
carry a PrimeKG-derived specificity weight
w= log10(total diseases/symptom count) + 1,
where higher weight indicates greater diagnostic
specificity. The scoring layer treats modifier terms
as independent phenotypes — a known limitation
(Section 8).
The rare disease RAG scanner provides an op-
tional safety net grounded in ZebraMap (Islamet al., 2026), encoded with MedEmbed and in-
dexed in a FAISS vector store. Retrieved candi-
dates are mapped from UMLS to MONDO IDs,
enabling scoring through the same dual-source
symbolic pipeline as the primary DDx. The mod-
ule is intended for presentations with eight or more
symptoms where the primary DDx may miss low-
prevalence conditions.
3.6 Stage 6: All-Disease Scanning
Beyond scoring LLM-nominated candidates,
NSIDDx provides a parallel, LLM-independent
diagnostic pathway that sweeps the entire knowl-
edge base, scoring every disease in HPO/MONDO
and PrimeKG against the current patient symptom
set using the hybrid score. The hybrid formula
is used by deliberate design: in full-database
sweeps, the matrix score suffers a structural
bias where diseases with tiny profiles achieve
artificially high scores (Table 2).
Disease Matrix Hybrid Explained
Intellectual dev. disorder1.0000.071 1/14
17q11.2 microduplication 0.6250.76212/14
Table 2: Matrix score bias in full-database scanning.
The hybrid score correctly reverses the ranking, which
is why all full-database sweeps use the hybrid formula
by default.
Practitioner override and closed-loop design.
At any stage, the practitioner can add or remove
symptoms and disease candidates, then re-run
scoring, graph generation, and PL compilation.
Changes propagate back into the LLM prompt,
giving direct control over the differential diagnosis
without modifying underlying weights.
4 Evaluation
This section evaluates the LLM+rare-disease-
RAG diagnostic pipeline on 750 CUPCase cases
across two cohorts, characterising failure modes
of the paradigm rather than of NSIDDx alone.
4.1 Setup
We evaluate on two cohorts drawn fromCUP-
Case(Perets et al., 2025), a publicly available
benchmark of 3,562 real-world patient case re-
ports sourced from BMC case report journals, ac-
cepted at AAAI 2025. The500-case random
sample(CUP) represents the full distribution of

clinically uncommon presentations — the real-
istic evaluation surface for any system deployed
on edge-case presentations. The250-case exact-
match sample(WELL) consists of cases whose
correct diagnosis resolves by exact name match in
HPO/MONDO or PrimeKG; this cohort represents
the upper bound of automated symbolic pipeline
performance within CUPCase, as the correct dis-
ease is guaranteed to have an ontology entry, a
curated phenotype profile, and a resolvable name.
Approximately 700 of 3,562 CUPCase cases meet
this criterion. Both cohorts are drawn from CUP-
Case; there is no domain shift between them. The
only variable is ontology coverage. All evaluation
used Qwen3.5-9B (Qwen Team, 2026) (IQ4_XS, a
4-bit GGUF quantization) on consumer hardware
without cloud API access. Median per-case run-
time was 87.6 seconds.
Metrics.We evaluate under four matching met-
rics of increasing permissiveness:exact(com-
plete string match),substring(one is a complete
substring of the other),token(significant token
overlap), andsemantic(cosine similarity of sen-
tence embeddings using Abhinand/MedEmbed-
large-v0.1, similarity threshold 0.7). We report all
four to make the gap between verifiable and appar-
ent accuracy explicit.
Statistical analysis.Accuracy differences be-
tween cohorts are tested using the chi-square
test of proportions on 2×2 contingency tables
(hit/miss per case); all reported expected cell
counts exceed five. Continuous distributions (phe-
notype counts, DDx label lengths, confidence
scores) are compared using the two-sided Mann-
Whitney U test, which makes no normality as-
sumption. Per-cohort confidence intervals are Wil-
son score intervals atα= 0.05. Significance
markers follow the convention∗p <0.05,∗∗p <
0.01,∗∗∗p <0.001.
4.2 Primary DDx Accuracy
The LLM component performs equivalently
across both cohorts: DDx@5 semantic accuracy
is 75.2% and 73.6% respectively (χ2,p= 0.700).
The divergence appears in exact accuracy (6.6%
vs 14.0%,p= 0.001) and token accuracy (34.4%
vs 46.8%,p <0.001) — metrics that depend on
vocabulary alignment between the LLM’s gener-
ated labels and ontology entries. Phenotype ex-
traction rates are also statistically indistinguish-
able (MWU,p= 0.976), ruling out differential in-Pipeline Exact Token Semantic
CUP (500-case random sample)
DDx (LLM) @1 2.4% 17.0% 41.8%
DDx (LLM) @3 5.6% 29.2% 67.8%
DDx (LLM) @5 6.6% 34.4%75.2%
MONDO @5 0.0% 1.8% 5.0%
KG @5 0.0% 1.4% 4.0%
Rare scanner any 2.6% 9.4% 23.6%
DDx@5∪Rare — — 78.4%
WELL (250-case exact-match sample)
DDx (LLM) @1 7.2% 25.6% 42.8%
DDx (LLM) @3 11.6% 40.8% 66.4%
DDx (LLM) @5 14.0% 46.8%73.6%
MONDO @5 0.4% 2.8% 8.0%
KG @5 0.8% 5.6% 8.0%
Rare scanner any 17.6% 27.6% 35.6%
DDx@5∪Rare — — 80.4%
Table 3: DDx semantic accuracy is statistically equiv-
alent between cohorts (χ2,p= 0.700). Exact and to-
ken accuracy diverge significantly (p <0.001–0.01),
reflecting vocabulary mismatch on uncommon cases
rather than differential LLM reasoning. CUP = 500-
case random sample; WELL = 250-case exact-match
sample (upper bound of ontology coverage within
CUPCase).
put quality as a confound. The failure is at the vo-
cabulary interface, not in the LLM’s clinical rea-
soning. The MONDO and KG symbolic scorers
achieve 5.0–8.0% semantic accuracy in both co-
horts, confirming that their near-zero performance
reflects HPO phenotype annotation sparsity, not a
scoring formula failure.
4.3 Rare Disease Scanner
The rare scanner recovers cases missed by the
LLM DDx (Table 4).
Metric CUP / WELL
Empty-match rate 83.5% / 84.4%
Mean empty-match con-
fidence0.641 / 0.645
Discriminative gap (cor-
rect vs missed)+0.061*** / +0.064***
Rare scanner sole recov-
ery3.2% / 6.8%
DDx@5∪Rare (com-
bined ceiling)78.4% / 80.4%
Table 4: Rare disease scanner quality analysis. The
empty-match rate is consistent across cohorts. The
discriminative gap is statistically significant (MWU,
p <0.001) but clinically insufficient as a decision
threshold.
Candidates are ranked by embedding similarity
without phenotypic pathway validation; the scan-

ner functions as a hypothesis generator requiring
clinician review.
4.4 Accuracy by Extraction Quality
Phenotype bin CUP WELL
nDDx@5nDDx@5
0 (extraction failure) 12 83.3%* 5 60.0%*
1–2 96 74.0% 54 66.7%
3–5 (modal) 173 72.3% 84 75.0%
6–9 131 79.4% 60 75.0%
10+ 88 75.0% 47 78.7%
Table 5: DDx semantic accuracy by extraction qual-
ity. No between-cohort difference is significant at any
phenotype bin (allp >0.44). The absence of signifi-
cant differences within bins confirms that performance
divergence is attributable to the vocabulary boundary
layer, not differential extraction quality.
Extraction errors are input-level failures that the
clinician can correct regardless of ontology cover-
age — motivating the practitioner override inter-
face.
5 Qualitative Failure Mode Analysis
Analysis of automated failure modes across both
cohorts reveals four intervention categories (Ta-
ble 6).
Category CUP (500) WELL (250)
Cat 1: Extraction failure 2.4% 2.0%
Cat 2: DDx complete
miss24.4% 25.6%
Cat 3: Semantic-only hit 41.6% 28.0%***
Cat 4: KG+MONDO
silent70.0% 62.8%
Any intervention 96.8% 93.2%*
Table 6: Failure taxonomy across both cohorts. Cate-
gories 1 and 2 are statistically equivalent (p >0.78),
indicating LLM blind spots and extraction failures are
domain-level phenomena not dependent on ontology
coverage. Category 3 diverges significantly (p <
0.001), localising the primary performance gap to the
vocabulary boundary.
Negation Inversion.Analysis of HPO resolver
mappings identified 26 cases where explicitly
denied symptoms were semantically mapped to
affirmative HPO terms, entering the scoring
pipeline as positive evidence. Examples in-
clude:DENIED: pain→Pain insensitivity,
DENIED: hemoptysis→Hemoptysis,Noremarkable family history→heredi-
tary fructose intolerance. In the sarcoidosis case
(Section 6), six of seven absent symptoms were
LLM-inferred rather than explicitly denied, caus-
ing matrix score−0.024for the correct diagno-
sis. This failure class is structurally undetectable
in any pipeline without explicit polarity encoding.
Implications for design.Categories 1 and 2 are
domain-level failures requiring clinician oversight
regardless of ontology coverage. Category 3 calls
for synonym normalisation and an audit trail so the
clinician can verify tokenically unverifiable hits.
Category 4 confirms that symbolic path confirma-
tion is unavailable for most cases; the PL string
and negation graph provide alternative explanation
modalities independent of score magnitude. Ac-
tive contradictions and negation inversions require
no system change — surfacing them is the design
goal.
6 Case Study
The following case falls within the category of
semantic-only hit under automation with active
symbolic contradiction and illustrates the override
mechanism converting a surfaced failure into a
confirmed diagnosis.
Case and automated extraction failure.A 64-
year-old Japanese woman presents with exertional
dyspnea, bilateral pleural effusions, bilateral hilar
and mediastinal lymphadenopathy, subcutaneous
nodules, and non-caseous epithelioid granulomas
on biopsy. The correct diagnosis is Sarcoidosis
(Kesici et al., 2014). The automated pipeline ex-
tracts 5 present and 7 absent symptoms:
Present (5):HP:0002094dyspnea,
HP:0032252granuloma,HP:0034388
hilar lymphadenopathy,HP:0100721medi-
astinal lymphadenopathy,HP:0002202pleural
effusion.
Absent (7):HP:0100749¬chest pain,
HP:0012735¬cough,HP:0001945¬fever,
HP:0002105¬hemoptysis,HP:0030166
¬night sweats,HP:0001962¬palpitations,
HP:0001824¬weight loss.
All seven absent symptoms are generated by
the LLM’s structured reformatting step, which in-
fers pertinent negatives from the narrative even
where the case text does not explicitly deny
them. Splenomegaly, documented via gallium-67
scintigraphy showing abnormal splenic uptake, is

not extracted because no explicit “splenomegaly”
surface form appears in the HPO synonym dictio-
nary.
The resulting automated score for Sarcoidosis is
MONDO m=−0.024, MONDO h= 0.512— the
negative matrix score places it last in the symbolic
ranking despite the LLM correctly nominating it
first in the DDx. The system does not suppress
this contradiction; the negative score is surfaced
in the scoring output alongside the PL audit trail.
Human-in-the-loop demonstration.A re-
searcher reviewed the raw case narrative alongside
the automated symptom vector and performed
three targeted interventions:
1. Corrects the absent list: six of the seven
absent symptoms (chest pain, cough, fever,
haemoptysis, night sweats, weight loss) were
not explicitly denied in the case text and
are removed, retaining only¬palpitations
(HP:0001962) as a true absent finding.
2. Adds splenomegaly by medical judgement:
the gallium-67 scintigraphy finding of
abnormal splenic uptake implies splenic
involvement, even though no explicit
“splenomegaly” string appears in the symp-
tom extraction pass. The clinician enters it as
/add_symptom splenomegaly.
3. Re-runs/score.
Removing the six spurious absent symptoms
resolves the matrix contradiction immediately:
Sarcoidosis moves from MONDO m=−0.024
to MONDO m= 0.073, MONDO h= 0.284.
Adding splenomegaly improves coverage further:
MONDO m= 0.098, MONDO h= 0.331. Sar-
coidosis rises to rank 1 in the scored DDx.
The corrected present vector contains 7 HPO
IDs (dyspnea, hilar lymphadenopathy, mediastinal
lymphadenopathy, pleural effusion, subcutaneous
nodules, granuloma, splenomegaly) and 1 absent
(¬palpitations). The full-database sweeps inde-
pendently confirm Sarcoidosis at top ranks across
all scoring sources.
The PL audit string for the corrected state reads:
(Dyspnea)∧(Mediastinal lym-
phadenopathy)∧(Pleural effusion)
∧(Splenomegaly)⇒sarcoidosis,
susceptibility to, 1Phenotypic graph.Figure 1 shows the
Negation-Retaining Phenotype Graph for the
corrected state.
This case demonstratesthe design philosophy:
the system surfaced a symbolic contradiction, the
practitioner corrected extraction errors via over-
ride, and convergent evidence confirmed the cor-
rected diagnosis. It does not represent typical
recovery rates (Section 5). The three interven-
tions required represent capabilities that Category
3 analysis indicates 208 CUPCase cases would
benefit from. We do not claim this recovery rate
is generalisable without a user study; we demon-
strate that the mechanism functions as designed in
this instance.
7 Position: Five Design Principles for
Clinician-in-the-Loop Clinical NLP
From the NSIDDx experience, we distill five de-
sign principles for clinical NLP systems that treat
the practitioner as an active reasoning agent rather
than a passive consumer of model outputs. These
principles are derived from the failure modes we
observed (Section 5) and the recovery patterns
demonstrated in the case study (Section 6). They
are well-motivated hypotheses, not empirically
validated claims; prospective clinician studies are
needed to test their usability and generalisability.
Principle 1: Surface Contradiction.When the
symbolic layer disagrees with the LLM, the sys-
tem should present both outputs side-by-side. The
contradictionisthe signal: it flags uncertainty, ex-
traction failure, or missing clinician knowledge. In
the sarcoidosis case (Section 6), the negative ma-
trix score (−0.024) correctly identified an extrac-
tion error — the system flagged its own mistake
rather than hiding it.
Principle 2: Preserve Negation Structurally.
Denied symptoms are first-class citizens in scor-
ing, graphs, and reasoning chains. NSIDDx en-
codes them as−1in the ternary vector and ren-
ders them as grey nodes in the negation graph, vi-
sually distinct from confirmed findings. The sar-
coidosis contradiction (matrix−0.024from spuri-
ous absent symptoms) motivates this design.
Yet a symptom may bestructurally absent(the
profile expects it) without beingclinically mean-
ingfully absentat a given stage of presentation.
The ternary vector cannot represent this temporal
distinction; the override interface exists to admit

Figure 1: Negation-Retaining Phenotype Graph for the corrected case. Red nodes: confirmed present symptoms.
Grey node: explicitly denied symptom. Green nodes: DDx candidates.
the clinical judgment that no encoding can capture
by design.
Principle 3: Enable Auditable Practitioner
Override.The practitioner can add, remove, or
modify symptoms at any stage, with changes prop-
agating through scoring, graph, and reasoning
chains in real time. Override actions are auditable:
what changed and why is visible. In NSIDDx,
/add_symptomand/remove_symptomtrig-
ger immediate re-scoring and regeneration of all
outputs.
Principle 4: Provide Multiple Levels of Expla-
nation.Different clinicians prefer different ex-
planation formats: some read PL audit strings,
others inspect graphs, others examine raw scores.
NSIDDx provides all three. No single explanation
format works for all users, and the system should
not force a choice. This aligns with the layered
explanation paradigm established in the XAI lit-
erature (Doshi-Velez and Kim, 2017). Numerical
scores alone can be insufficient; the PL string and
negation graph provide complementary modalities
independent of score magnitude.
Principle 5: Design for Offline Deployment
on Consumer Hardware.Low-resource set-
tings cannot rely on cloud APIs. Systems designedfor these settings must run locally on commodity
hardware. This is not a technical constraint — it
is a design requirement that shapes every architec-
tural choice, from the selection of quantized mod-
els (Qwen3.5-9B-IQ4_XS.gguf) to the decision to
use deterministic scoring alongside probabilistic
LLM output.
8 Conclusion
We presented NSIDDx, a neuro-symbolic pipeline
with explicit negation, ternary-hybrid scoring, and
full-stack override — and a design framework ar-
guing that diagnostic AI must prioritise tractabil-
ity of disagreement over raw accuracy. Eval-
uation characterises three failure modes of the
LLM+RAG paradigm under open-ended condi-
tions (vocabulary mismatch, confidence-without-
grounding, negation inversion) that make clinician
oversight a routine requirement, not an edge case.
Offline deployability on consumer hardware is a
prerequisite for equitable access in the settings this
system is designed to serve. A case study demon-
strates the override mechanism. We distill five
design principles for clinician-in-the-loop clinical
NLP and invite the prospective studies needed to
validate the claim at scale.

Limitations
We document the known limitations of NSIDDx
alongside the design choices that partially address
each.
Natural language symptom coverage.
MedSpaCy TargetRule matching performs literal
string lookup against a large synonym dictionary.
A semantic fallback (Abhinand/MedEmbed-large-
v0.1, cosine threshold 0.7) resolves colloquial
terms such as“SOB”to dyspnea, but the threshold
was not ablated and false positives are possible;
match quality is uncertain. Each extracted
symptom is tagged with its resolution method
(literal/semantic); the practitioner can inspect and
correct any mapping via/add_symptom.
PrimeKG phenotype coverage.PrimeKG has
limited coverage of common diseases: many con-
ditions a general practitioner would recognise con-
fidently (e.g., pharyngitis, bronchitis) lack curated
phenotype entries in the graph, producing zero
or low KG scores for diagnoses that are clini-
cally straightforward. The HPO/MONDO ma-
trix score is unaffected by this gap. NSIDDx is
most valuable for complex, multisystem presen-
tations where GPs genuinely benefit from struc-
tured support. This limitation is partially miti-
gated by the dual-source design: while PrimeKG
scores may be near-zero for common conditions,
the HPO/MONDO matrix score remains opera-
tional, and the LLM-generated DDx is not sup-
pressed by low KG scores.
LLM hallucination in DDx reasoning.The
LLM may generate plausible but factually incor-
rect reasoning chains, particularly for rare diseases
with sparse training data representation. The sym-
bolic scoring layer exists precisely as an indepen-
dent validation step: if the LLM proposes a diag-
nosis that the matrix score actively contradicts, the
practitioner sees both the narrative reasoning and
the symbolic contradiction. The system does not
suppress the LLM’s output; it presents it alongside
its symbolic assessment.
Absent symptom reliability.The pipeline treats
clinician-reported negations as structurally mean-
ingful diagnostic signals. In real settings, patients
may not volunteer absent symptoms unless ex-
plicitly asked. The history-taking agent partially
addresses this through targeted pertinent-negative
elicitation, but cannot guarantee completeness.Single-language support.The pipeline cur-
rently operates on English clinical text. Many
low-resource settings operate in Hindi, Swahili, or
other languages. Cross-lingual extension is out-
side the scope of this version; the modular archi-
tecture is designed to support alternative extrac-
tion front-ends.
Single-encounter reasoning.NSIDDx reasons
over a single clinical encounter. Chronic or evolv-
ing presentations with important longitudinal sig-
nal are outside the current design scope.
Phenotype modifier terms in scoring.The
graph correctly captures qualifier relation-
ships between HPO terms via PrimeKG’s
phenotype_modifier_ofedges. However,
the symbolic scoring layer does not exploit this
binding: modifier terms extracted by MedSpaCy
are entered intoV pas independent+1entries,
creating a gap between the graph’s semantic
fidelity and the scoring layer’s representational
granularity. Composing modifier-symptom pairs
into single weighted phenotype entries prior to
scoring is a direction for future work.
Evaluation scope.Evaluation is conducted on
750 cases from the CUPCase benchmark across
two cohorts — real-world published case reports.
The sample reflects hardware constraints (one to
three minutes per case on the target hardware).
Notably, CUPCase cases are selected for their di-
agnostic interest and completeness, which may
overestimate system performance compared to
routine clinical notes that are often fragmented or
incomplete. Real-world EHR validation and eval-
uation on the full CUPCase corpus remain neces-
sary next steps before clinical deployment.
Semantic matching permissiveness.The eval-
uation relies on semantic similarity (cosine simi-
larity of sentence embeddings) as a matching met-
ric. Across the CUPCase cohort, 83.5% of rare
scanner outputs and 41.6% of DDx@5 semantic
hits carry no tokenically verifiable relationship to
the ground-truth diagnosis. Whether these seman-
tic hits represent genuine synonymy or embedding
artefacts cannot be determined without expert re-
view — a core motivation for the practitioner over-
ride interface.
Human-in-the-loop demonstration.The over-
ride interventions in Section 6 were performed by
a researcher reviewing the case narrative alongside

the automated symptom vector, not by a clinician
in a live diagnostic setting. The case study demon-
strates the mechanism functions as designed but
provides no evidence about human behaviour with
the system. Prospective user studies with practis-
ing clinicians are necessary and explicitly invited.
Scoring formula ablation.The ternary-hybrid
scoring formulas were designed for interpretabil-
ity rather than diagnostic accuracy. No hy-
perparameter tuning was performed; the equal-
weight combination (0.5/0.5) was selected for
transparency. Future work should explore learned
weights that preserve interpretability while im-
proving discrimination.
Ethical and Societal Implications
Positive Impact Definition.For us, positive im-
pact in clinical NLP is not primarily measured in
benchmark accuracy. It is measured inaccess:
whether a practitioner in a clinic with no spe-
cialist referral network, no reliable internet con-
nection, and no clinical decision support infras-
tructure can use a system to surface a differential
diagnosis they might otherwise have missed, in-
spect the reasoning behind it, and make a better-
informed decision for their patient. NSIDDx is
designed with this practitioner in mind. Its of-
fline deployability, consumer hardware require-
ments, and practitioner-in-the-loop architecture
are not technical constraints — they are the de-
sign goal. Following the NLP4PI workshop’s em-
phasis on impact grounding (Pant et al., 2025),
we define positive impact as democratized access
to structured differential reasoning — the ability
of any practitioner, regardless of institutional re-
sources, to receive a transparent, auditable diag-
nostic aid that respects clinical expertise and local
data sovereignty.
Assistive-Only Design Principle.NSIDDx is
an assistive tool, not an autonomous diagnostic
system. It does not make diagnoses. Every out-
put — the DDx ranking, the PL string, the pheno-
typic graph — is presented to the practitioner as
structured information to reason with, not a deci-
sion to accept. The system is explicitly designed
so that disagreement is easy: the practitioner can
add symptoms, remove candidates, and rerun the
pipeline in seconds.
Automation Bias and Over-Reliance Risk.
Any decision support system carries a risk of au-tomation bias: the tendency of clinicians to an-
chor on system outputs even when their own clin-
ical judgment diverges. The PL strings and phe-
notypic graphs are designed toinvitedisagree-
ment — a practitioner who reads(Alopecia)∧
(Malar rash)⇒SLE and knows the patient also
has a finding not captured in the system has a clear
signal that the system’s evidence base is incom-
plete. Nevertheless, we acknowledge that the risk
of over-reliance cannot be fully designed away and
should be addressed through practitioner training
and deployment guidance.
Rare Disease Scanner and Diagnostic Heuris-
tics.We deliberately invert the hoofbeats heuris-
tic for low-resource settings: when specialist re-
ferral is unavailable, surfacing a rare disease can-
didate with moderate confidence is more ethical
than suppressing it. This inversion is bounded by
three constraints: the scanner is optional and off
by default; it activates only for presentations with
eight or more symptoms; and low-scoring candi-
dates (including negative matrix scores) are visu-
ally deprioritised.
Evaluation on real-world case reports.The
evaluation uses 750 cases from the CUPCase
benchmark across two cohorts, a publicly avail-
able collection of real-world BMC patient case
reports. The sample reflects inference time con-
straints on consumer hardware. Validation on
institution-specific EHR data, with appropriate
ethics approvals and data governance, is required
before any clinical deployment.
Review and Governance.NSIDDx is explicitly
positioned as a research prototype, not a clinical
tool. Any future deployment would require in-
stitutional review board approval, HIPAA/GDPR
compliance audits, and a phased clinical validation
protocol. The system’s architecture — local pro-
cessing, no cloud dependency, auditable reasoning
chains — is designed to facilitate, not circumvent,
these governance requirements.
Equity and language access.The instantiated
system currently operates only in English. This
limits immediate applicability in many of the low-
resource settings it is designed to serve. We recog-
nise this as a significant equity gap and iden-
tify multilingual extension as a priority for future
work.

Data Retention and Sovereignty.NSIDDx op-
erates entirely locally. No patient information is
sent to external servers or APIs. This is both a
practical requirement for offline deployment and
an ethical requirement for patient data sovereignty
in settings where data protection infrastructure
may be limited.
References
Payal Chandak, Kexin Huang, and Marinka Zitnik.
2023. Building a knowledge graph to enable pre-
cision medicine.Scientific Data, 10(1):67.
Finale Doshi-Velez and Been Kim. 2017. Towards a
rigorous science of interpretable machine learning.
InNIPS 2017 Symposium on Interpretable Machine
Learning.
Michael A Gargano, Nicolas Matentzoglu, Ben Cole-
man, Eunice B Addo-Lartey, Anna V Anagnos-
topoulos, Joel Anderton, Paul Avillach, Anita M
Bagley, Eduard Bakštein, James P Balhoff, Gareth
Baynam, Susan M Bello, Michael Berk, Holli
Bertram, Somer Bishop, Hannah Blau, David F Bo-
denstein, Pablo Botas, Kaan Boztug, and 157 others.
2024. The human phenotype ontology in 2024: phe-
notypes around the world.Nucleic Acids Research,
52(D1):D1333–D1346.
Kate Goddard, Abdul Roudsari, and Jeremy C Wyatt.
2012. Automation bias: a systematic review of fre-
quency, effect mediators, and mitigators.Journal
of the American Medical Informatics Association,
19(1):121–127.
Md Sanzidul Islam, Amani Jamal, and Ali Alkhathlan.
2026. ZebraMap: A multimodal rare disease knowl-
edge map with automated data aggregation & LLM-
enriched information extraction pipeline.Diagnos-
tics, 16(1):107.
Besir Kesici, Ahmet Burak Toros, Levent Bayraktar,
and Adem Dervisoglu. 2014. Sarcoidosis inciden-
tally diagnosed: A case report.Case Reports in Pul-
monology, 2014(1):702868.
Ha Young Kim, Jun Li, Ana Beatriz Solana, Car-
olin M. Pirkl, Benedikt Wiestler, Julia A. Schn-
abel, and Cosmin I. Bercea. 2025. Learning to rea-
son about rare diseases through retrieval-augmented
agents.Preprint, arXiv:2511.04720.
Qiuhao Lu, Rui Li, Elham Sagheb, Andrew Wen,
Jinlian Wang, Liwei Wang, Jungwei W. Fan, and
Hongfang Liu. 2025. Explainable diagnosis pre-
diction through neuro-symbolic integration.AMIA
Joint Summits on Translational Science Proceed-
ings, 2025:332–341.
Semanto Mondal, Antonino Ferraro, Fabiano Pecorelli,
and Giuseppe De Pietro. 2025. A logic tensornetwork-based neurosymbolic framework for ex-
plainable diabetes prediction.Applied Sciences,
15(21):11806.
Harsha Nori, Nicholas King, Scott Mayer McKinney,
Dean Carignan, and Eric Horvitz. 2023. Capabilities
of gpt-4 on medical challenge problems.Preprint,
arXiv:2303.13375.
Devesh Pant, Rishi Raj Grandhe, Jatin Agrawal,
Jushaan Singh Kalra, Sudhir Kumar, Saransh
Khanna, Vipin Samaria, Mukul Paul, Dr. Satish V
Khalikar, Vipin Garg, Dr. Himanshu Chauhan,
Dr. Pranay Verma, Akhil Vssg, Neha Khandelwal,
Soma S Dhavala, and Minesh Mathew. 2025. Health
sentinel: An AI pipeline for real-time disease out-
break detection. InProceedings of the Fourth Work-
shop on NLP for Positive Impact (NLP4PI), pages
23–42, Vienna, Austria. Association for Computa-
tional Linguistics.
Oriel Perets, Ofir Ben Shoham, Nir Grinberg, and
Nadav Rappoport. 2025. CUPCase: Clinically
uncommon patient cases and diagnoses dataset.
InProceedings of the AAAI Conference on Ar-
tificial Intelligence, volume 39, pages 28293–
28301. Dataset:https://huggingface.co/
datasets/ofir408/CupCase.
Qwen Team. 2026. Qwen3.5: Towards native multi-
modal agents.
Simon Ronicke, Martin C. Hirsch, Ewelina Türk,
Katharina Larionov, Daphne Tientcheu, and An-
nette D. Wagner. 2019. Can a decision support sys-
tem accelerate rare disease diagnosis? evaluating the
potential impact of Ada DX in a retrospective study.
Orphanet Journal of Rare Diseases, 14(1):69.
Karan Singhal, Shekoofeh Azizi, Tao Tu, S. Sara Mah-
davi, Jason Wei, Hyung Won Chung, Nathan Scales,
Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl,
Perry Payne, Martin Seneviratne, Paul Gamble,
Chris Kelly, Abubakr Babiker, Nathanael Schärli,
Aakanksha Chowdhery, Philip Mansfield, Dina
Demner-Fushman, and 13 others. 2023. Large lan-
guage models encode clinical knowledge.Nature,
620(7972):172–180.
Tao Tu, Mike Schaekermann, Anil Palepu, Khaled
Saab, Jan Freyberg, Ryutaro Tanno, Amy Wang,
Brenna Li, Mohamed Amin, Nenad Tomasev, and
1 others. 2025. Towards conversational diagnostic
artificial intelligence.Nature.
Qipeng Wang, Rui Sheng, Yafei Li, Huamin Qu,
Yushi Sun, and Min Zhu. 2025a. MedKGI: Itera-
tive differential diagnosis with medical knowledge
graphs and information-guided inquiring.Preprint,
arXiv:2512.24181.
Ruiyu Wang, Tuan Vinh, Ran Xu, Yuyin Zhou, Jiaying
Lu, Francisco Pasquel, Mohammed K. Ali, and Carl
Yang. 2025b. Knowledge graph augmented large
language models for disease prediction.Preprint,
arXiv:2512.01210.

Qinkai Yu, Mingyu Jin, Dong Shu, Chong Zhang,
Lizhou Fan, Wenyue Hua, Suiyuan Zhu, Yanda
Meng, Zhenting Wang, Mengnan Du, and Yongfeng
Zhang. 2025. Health-llm: Personalized retrieval-
augmented disease prediction system.Preprint,
arXiv:2402.00746.
Charlotte Zelin, Wendy K. Chung, Mederic Jeanne,
Gongbo Zhang, and Chunhua Weng. 2024. Rare
disease diagnosis using knowledge guided retrieval
augmentation for ChatGPT.Journal of Biomedical
Informatics, 157:104702.
A Sarcoidosis Case: Negation-Retaining
Phenotype Graph
The following is the Negation-Retaining Phe-
notype Graph output for the corrected Sar-
coidosis demonstration case from Section 6,
showing present and absent symptom nodes
with their HPO identifiers and PrimeKG-derived
explainsedge weights for each DDx candidate.
Patient – present (7):
dyspnea (HP:0002094)
hilar lymphadenopathy (HP:0034388)
mediastinal lymphadenopathy (HP:0100721)
pleural effusion (HP:0002202)
subcutaneous nodule (HP:0001482)
granuloma (HP:0032252)
splenomegaly (HP:0001744)
Patient – denies (1):
palpitations (HP:0001962)
Candidates (green nodes):
Sarcoidosis (ORPHA:797): explains dysp-
nea (w=3.739), pleural effusion (w=3.317),
subcutaneous nodule (w=3.165), mediastinal
lymphadenopathy (w=2.781); also linked to
¬palpitations via “denies”
Granulomatosis with polyangiitis: explains pleu-
ral effusion (w=3.317), subcutaneous nodule
(w=3.165); narrower coverage
Lung TB: no resolved HPO overlap, discon-
nected node with zero scores
NHL: no resolved HPO overlap, disconnected
node
Pulmonary fungal disease: no resolved HPO
overlap, disconnected node
B Evaluation Data
The evaluation set consists of 750 cases drawn
from the CUPCase benchmark (Perets et al., 2025)
across two cohorts: a 500-case random sam-
ple representing the full distribution of clinically
uncommon presentations, and a 250-case exact-
match sample representing the upper bound of on-
tology coverage within CUPCase. The full 3,562-
case corpus remains unevaluated due to inference
time constraints on consumer hardware.
The complete filtered dataset, together with all
evaluation scripts and code used in this paper,is publicly available athttps://github.
com/joetheguide2/NSIDDX-. The raw
per-case result files backing this evaluation
arepart_1_qwen9b_semantic.csv
(CUP, 500-case cohort) and
exact_qwen9b_semantic.csv(WELL,
250-case cohort). The repository also contains
earlier exploratory runs and threshold variants
retained for transparency; these do not reflect the
final reported numbers, which are the two files
named above. Some summary documents in the
repository (e.g. disease/symptom counts in the
top-level README) report raw, pre-filter counts
rather than the corrected, filtered counts used in
the scoring pipeline and reported in Section 3.
C Scoring Mechanism Detail
The scoring engine produces four scores per can-
didate disease, displayed asHPO_m/KG_m(ma-
trix) andHPO_h/KG_h(hybrid). Both formulas
are applied to both sources. Table 7 provides an in-
terpretation guide for reading score combinations
in clinical practice.
Matrix Hybrid Clinical interpretation
High High Strong match; disease fits both pos-
itive findings and full expected pro-
file
Low High Disease explains findings but patient
is missing symptoms the disease re-
quires
Negative Any Active contradiction; inspect grey
nodes in graph for specific conflicts
Any Low Weak explanatory fit; disease does
not account for the patient’s symp-
tom cluster
Zero Zero No phenotype overlap; consider
/rare_disease_scan
HPO
high, KG
low— KG resolution likely inaccurate;
checkKG→mapping
Table 7: Four-score interpretation guide. HPO/KG
discrepancies surface knowledge graph noise and are
flagged for the practitioner.