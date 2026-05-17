# ClinicalBench: Stress-Testing Assertion-Aware Retrieval for Cross-Admission Clinical QA on MIMIC-IV

**Authors**: Alex Stinard

**Published**: 2026-05-11 18:47:52

**PDF URL**: [https://arxiv.org/pdf/2605.11143v1](https://arxiv.org/pdf/2605.11143v1)

## Abstract
Reasoning benchmarks measure clinical performance on clean inputs. We evaluate the step before reasoning: retrieval over real EHR notes, where negation, temporality, and family-versus-patient attribution can flip a correct answer to a wrong one. EpiKG carries an assertion label and a temporality tag with every fact in a patient knowledge graph, then routes retrieval by question intent. ClinicalBench is a 400-question test over 43 MIMIC-IV patients across 9 assertion-sensitive categories. A 7-condition ablation tests each piece of EpiKG across six LLMs (Claude Opus 4.6, GPT-OSS 20B, MedGemma 27B, Gemma 4 31B, MedGemma 1.5 4B, Qwen 3.5 35B). Three physicians blindly adjudicated 100 paired items. The author-blind primary endpoint, leave-author-out paired exact McNemar on 50 unanimous-strict items rated by two external physicians, yields +22.0 percentage points (95 percent Newcombe CI [+5.1, +31.5], p=0.0192). The architectural novelty, intent-aware KG-RAG over a Contriever dense-RAG baseline (C2b to C4g_kw on the change-excluded n=362 endpoint), is +8.84 percentage points (paired McNemar p=1.79e-3); +12.43 percentage points under oracle intent. Sensitivities agree directionally: three-rater physician majority +24.0 percentage points (subject to single-author circularity); deterministic keyword reproducibility proxy +39.5 percentage points. Across the six models, the gain shrinks as the LLM-alone baseline rises (beta=-1.123, r=-0.921, p=0.009). With n=6 this looks more like regression to the mean than encoding substituting for model size. Physician adjudication identified 56 percent of auto-generated reference answers as defective, a methodological finding indicating that NLP-pipeline clinical-QA benchmarks require physician adjudication to be usable. ClinicalBench, the frozen evaluator, three-rater adjudication data, and the EpiKG output stack are publicly released.

## Full Text


<!-- PDF content starts -->

ClinicalBench: Stress-Testing Assertion-Aware Retrieval
for Cross-Admission Clinical QA on MIMIC-IV
Alex Stinard, MD
Department of Clinical Sciences, College of Medicine
University of Central Florida, Orlando, FL 32816
alex.stinard@ucf.edu
May 13, 2026
Preprint — arXiv version
Abstract
Objective.Reasoningbenchmarksmeasureclinicalperformanceoncleaninputs. Weevaluatethestepbeforereasoning: retrieval
over real EHR notes, where negation, temporality, and family-versus-patient attribution can flip a correct answer to a wrong one.
MaterialsandMethods.EpiKGcarriesanassertionlabelandatemporalitytagwitheveryfactinapatientknowledgegraph,then
routes retrieval by question intent. ClinicalBench is a 400-question test over 43 MIMIC-IV patients across 9 assertion-sensitive
categories. A 7-condition ablation tests each piece of EpiKG across six LLMs (Opus 4.6, GPT-OSS 20B, MedGemma 27B,
Gemma 4 31B, MedGemma 1.5 4B, Qwen 3.5 35B). Three physicians blindly adjudicated 100 paired items.
Results.Author-blindprimaryendpoint: leave-author-outpairedexactMcNemaron50items(Hird ×Nadeemunanimousstrict),
Δ=+22.0pp [+5.1pp,+31.5pp](95%Newcombe CI), 𝑝=0.0192 . ThearchitecturalnoveltyisC2b(Contrieverdense-RAG)
→C4g_kw(intent-awareKG-RAG)onthechange-excluded 𝑛=362endpoint:+8.84pp (pairedMcNemar 𝑝=1.79×10−3);
+12.43pp under oracle intent. Sensitivity analyses: three-rater physician majority +24.0pp (𝑝=0.0075 ; Fleiss’𝜅=0.413 ;
subjecttosingle-authorcircularitysincetheauthorisR1);deterministickeywordproxy +39.5pp overLLM-alone(reproducibility
tool, not a clinical correctness claim). The audit found 56% of auto-generated references defective.
Discussion.Across the six models, the gain shrinks as the LLM-alone baseline rises ( 𝛽=−1.123 ,𝑟=−0.921 ,𝑝=0.009 ).
With𝑛=6thislooksmorelikeregressiontothemeanthanencodingsubstitutingformodelsize. Theauthorbuiltthesystem,
generated the initial gold standard, and performed the internal audit. The primary endpoint uses external physician ratings with
the author left out.
Conclusion.Carryingassertionlabelsandroutingbyquestionintentimprovecross-admissionclinicalQAacrosssixLLMs.
ClinicalBench and the evaluation artifacts are public.
1 Background and Significance
Largelanguagemodelsmatchorexceedphysician-level
performanceonmedicallicensingexams[ 1–3],andreason-
ing benchmarks like HealthBench Professional, MedQA,
and USMLE-style items measure that last mile of clin-
ical reasoning given clean vignettes. Real EHR use
exposes a complementary, undermeasured layer:retrieval
faithfulnesson messy charts, where negation, temporal
drift, source conflict, and semantic compression must be
navigated before reasoning. The harder question is not
whether AI can reason like a physician but whether it
can read like one—physicians, of course, do both. A
single sentence—“patient denies chest pain, sister had
MI at 45, will consider statin if lipids remain elevated”—
encodesnegation,familyattribution,hypotheticalintent,
and an implicit present condition. Clinical NLP detectsthese assertions accurately [ 4,5], but RAG pipelines flat-
tenthe context, conflating “patientdenies” with“patient
has.” This is theepistemic propagation gap, inside a
broaderstructural-representationgap—assertiontyping,
temporal indexing, experiencer attribution preserved to
retrieval—that reasoning benchmarks do not probe.
To the best of current knowledge, no patient-level
clinical KG-RAG system jointly preserves assertion
state on graph edges and routes retrieval by ques-
tion intent. OMOP excludes negated conditions
from CONDITION_OCCURRENCE [6], and FHIR provides
verificationStatus forCondition resources only.
Existing graph-augmented RAG systems—including
GraphRAG [ 7], GFM-RAG [ 8], KARE [ 9], and Medical-
Graph-RAG[ 10]—buildKGsthatdiscardthemetadatadis-
tinguishing“patienthasdiabetes”from“ruleoutdiabetes.”
A paralleltemporal integration gapexists: clinical events
1arXiv:2605.11143v1  [cs.CL]  11 May 2026

admit bi-temporal storage (valid + transaction time, in the
Snodgrass tradition; cf. Zep [ 11]) plus an NLP-asserted
temporality label 𝜏𝑎∈ {Past,Current,Future} , yet
existing systems model at most a subset [11, 12].
The core empirical finding is interactional: assertion
preservationalonedoesnotimproveaggregateaccuracy
unless retrieval is also routed by question type. Three
contributions are made:
1.ClinicalBench.A 400-question single-site, same-
record stress test over 43 MIMIC-IV patients (con-
venience sample; 32 with two admissions, 11 single-
admission) and 9 assertion-sensitive categories, expos-
ingcategory×conditioninteractionsaggregatescores
hide. Ittargetsretrievalfaithfulnessonrealchartsrather
than reasoning on clean vignettes, complementing
exam-stylebenchmarksatadifferentlayer. SliceBench,
a small supporting case study on record complexity, is
also introduced.
2.EpiKGandtheepistemicpropagationgap.Theloss
of assertion metadata across clinical NLP pipelines
is formalized, an information-theoretic loss bound is
derived (Section 3.3, Appendix B), and a patient-level
clinicalKG-RAGsystemisimplementedthatpreserves
assertionandtemporalmetadatawhileroutingretrieval
by question intent.
3.Author-blindprimaryendpointandarchitectural
novelty.The author-blind primary is a paired
test: leave-author-out exact McNemar on 𝑛=50
unanimous-strict items adjudicated by two exter-
nal physicians, yielding Δ =+22.0pp (95% New-
combe CI[+5.1,+31.5] ,𝑝=0.0192 ). The architec-
tural novelty is the paired delta of intent-aware KG-
RAGoverastrongdense-RAGbaseline(Contriever),
C2b→C4g_kw+8.84pp onthechange-excluded 𝑛=
362endpoint (McNemar 𝑝=1.79×10−3; oracle
+12.43pp ). Secondary sensitivities are demoted:
three-ratermajority +24.0pp (single-authorcircularity
since the author is one rater) and a deterministic repro-
ducibility proxy (keyword evaluator) +39.5pp (not a
clinical-correctnessclaim). Cross-modelconvergence
across𝑛=6models is descriptive only: a linear re-
gression of C1 baseline against C1 →C4g_oracle delta
yields𝛽=−1.123 ,𝑟=−0.921 ,𝑝=0.009 , consis-
tentwithregressiontothemeanratherthanencoding
substituting for parameter count.
The author designed the benchmark, built the system,
and conducted the internal evaluation; this circularity
is structurally mitigated by frozen evaluation artifacts,externalphysicianevaluation,andcross-modelreplication,
butreadersshouldweightclaimsaccordingly(Section4.1).
Together these yield a benchmark-supported design hy-
pothesis: preserve epistemic metadata, route retrieval
by intent, and evaluate by category ×condition interac-
tion rather than aggregate score. The central research
question—when does structured epistemic context help,
hurt, or break even?—is answered interactionally on a
single-site,in-distributionstresstestdesignedforretrieval
faithfulness rather than cross-site generalization.
1.1 Related Work
Prior work is organized along four axes (extended discus-
sion and Table 6 in Appendix D.1).
Clinical reasoning benchmarks.Reasoning evalua-
tions are complementary. HealthBench Professional [ 13],
MedQA [ 14], and MedPaLM 2 [ 1] score reasoning on vi-
gnettes with facts pre-supplied; EpiKG measures retrieval
faithfulnessonreallongitudinalEHRswithdispersedfacts
and negation, temporal, and source ambiguity. The two
probedifferentstages: thelastmile(reasoninggivenclean
inputs)versusthefirstmile(readingtherightpatientfrom
messy charts).
MedicalRAGandclinicalQA.Graph-augmentedre-
trieval is a leading paradigm: GraphRAG [ 7], GFM-
RAG [8], and Medical-Graph-RAG [ 10] buildpopulation-
levelgraphsbutdonotpropagatenote-derivedassertionor
temporalmetadata(bi-temporalstoragewithNLP-asserted
scope label, in our framing). Existing benchmarks—
MedPaLM 2 [ 1], MIRAGE [ 15], emrQA [ 16]—target
factual recall or grounded retrieval, not assertion-faithful
longitudinal QA over real EHRs.
Clinical KG construction.Multi-LLM KG-RAG [ 17],
AutoRD[ 18],andRECAP-KG[ 19]applyLLMstoclinical
KGconstructionbutdonotpropagateassertionstatusinto
the final graph.
AssertiondetectionandtemporalKGs.NegEx[ 20],
ConText[ 21],andGuletal.[ 5]treatassertiondetection
asterminalannotation;MedTKG[ 12]andGraphiti[ 11]
implement temporal KGs butlack epistemic propagation.
Twostructuralgapsemerge: anepistemicpropagationgap
(assertionlabelsarenotpersistedintoKGs)andatemporal
integrationgap(temporalformalismsareannotationlayers,
not retrieval-participating edge attributes). EpiKG closes
2

both by carrying assertion and temporal metadata as first-
class properties through every pipeline stage.
2 Objective
To evaluate whether preserving assertion and temporal
metadata in a patient-level clinical knowledge graph,
thenroutingretrievalbyquestionintent,improvescross-
admission clinical question answering over electronic
health records.
3 Materials and Methods
3.1 Method Overview
EpiKG implements three ideas (Figure 1): (1) end-to-
end epistemic preservation, carrying assertion labels
through extraction, OMOP mapping, KG materializa-
tion, and retrieval; (2) bi-temporal edge storage (valid
time, transaction time; in the Snodgrass tradition, cf.
Graphiti [ 11]) plus an NLP-asserted temporality label
𝜏𝑎∈{Past, Current, Future} derived from clinical-text
scope (data-modeling clarification in Appendix C.1); and
(3) intent-aware routing matching graph traversal to ques-
tion type. The first two are infrastructure; the third is
where the performance gain originates.
3.2 Epistemic Assertion Schema
Clinicalnotescontainqualifiedstatements(“noevidenceof
pneumonia,”“possibleCHF,”“motherhadbreastcancer”)
that standard representations discard: OMOP excludes
negated conditions [ 6]; FHIR limits assertion metadata.
EpiKG defines a seven-value assertion taxonomy:
𝛼∈{Pres., Abs., Poss., Cond., Hypo., Fam.Hx., Hist.}(1)
extendingthei2b2six-classtaxonomy[ 4]byseparating
HistoricalfromFamily_History(AppendixQ).Arule-
basedclassifier(122scope-awaretriggerpatterns)assigns
𝛼withaconfidencescore,propagatedthrougheverystage.
Each edge carries bi-temporal metadata (valid + trans-
action time) plus an NLP-asserted temporality label 𝜏𝑎,
withAllen-styleintervalrelationsstoredasedgemetadata
(data-modeling clarification in Appendix C.1).
3.3 Formal Epistemic Preservation
Theepistemicinvariantisformalizedasatestablepipeline
property(AppendixB).Anassertion-blindpipelinecol-
lapses all labels toPresent, reducing assertion entropyto zero [22]; its faithfulness bound is 1−𝑓np(𝑐), where
𝑓npis the fraction of non-present mentions—substantially
below 1 for concepts like pneumonia or diabetes. Em-
pirical consequences are measured via category-stratified
accuracy in Section 4.
3.4 Intent-Aware Retrieval (C4g)
The base retrieval pipeline uses bidirectional BFS over
patient KG edges and OMOP vocabulary relationships
(Appendix C.2), but treats all questions uniformly. Differ-
ent clinical question types require fundamentally different
graph operations:changerequires cross-admission set
differencing,current-stateneeds the most recent valid
edges,historicalmust recover resolved conditions.
Intentclassifier.Arule-basedclassifiermapseachques-
tion toChange,Current_State,Historical, orDe-
fault(Algorithm 1, Appendix S). Primary results use
keyword-onlyclassification(production-realistic);oracle
classificationwithcategorymetadataisanupperbound.
Keyword classification reduces Opus C4g from 68.5% to
60.2% (−8.3pp; Section 4.1).
Routing strategies.Changepartitions edges by
hadm_id andcomputessetdifferencesacrossadmission
pairs.Current_Statefilters edges to 𝜏𝑎=Current
or open validity.Historicalselects 𝜏𝑎=Pastedges,
augmented by admission-based inference (concepts in
earlierbutnotthelatestadmissionarelabeled“resolved”).
Each intent triggers a type-specific prompt template
(Appendix U). Figure 2 shows a historical question an-
swered incorrectlyunderC1 (noevidence) andC4 (stale
PRESENTlabel)butcorrectlyunderC4g’stemporalfil-
tering.
ExistingclinicalQAbenchmarksevaluatefactualrecall
(MedQA [ 14]) or agent task completion [ 23] but do not
test epistemic qualifiers or cross-admission reasoning.
ClinicalBench1is a same-record retrieval-faithfulness
stress test on MIMIC-IV [ 24]; SliceBench is a small
supporting case study.
3.5ClinicalBench: Assertion-Sensitive Clinical
QA
ClinicalBenchcomprises400questionsover43MIMIC-
IV patients across two tasks (A: 200 negation-aware re-
trieval;B: 200 temporal reasoning) and 9 categories
1Unrelated to identically named benchmarks in other clinical NLP
subfields.
3

assertion label  α  preserved end-to-end
α∈:tri-temporal: tvalid | ttxn | τNLP
Clinical
NoteNLP
ExtractOMOP
MapClinical
FactKnowledge
GraphIntent
RouterStructured
EvidenceLLM
GenerateAnswer
Intent Router  —  route on  ι  (intent type)
CHANGE
Cross-admission
set differencing
Δ(adm1,adm2)CURRENT STATE
Latest-edge
filtering
argmaxt(v)HISTORICAL
Resolved condition
recovery
α∈{Absent,Hist.}DEFAULT
Bidirectional
BFS traversal
BFS(v,d≤2)
Present Absent Possible Conditional Hypothetical Family Hx HistoricalDischarge Note
"Pt denies chest
 pain. Family hx of
 MI at age 45.
 Will consider statin
 if lipids remain
 elevated."Extracted Mentions
chest_pain
  α = ABSENT
MI
  α = FAMILY_HX
statin
  α = CONDITIONALKG Edges
pt → chest_pain
  α=ABSENT  τ=CURRENT
pt → MI
  α=FAM_HX  exp=FAMILY
pt → statin
  α=COND   τ=FUTUREEvidence to LLM
Q: "Current meds?"
ι = CURRENT_STATE
Filtered edges:
  statin (α=COND, skip)
  metformin (α=PRES ✓)Data at each stageFigure 1: EpiKG system workflow with concrete data examples.Top: 9-stage pipeline from clinicalnote to answer,
with the gold 𝛼ribbon tracing assertion preservation end-to-end.Middle: actual data at each stage—a discharge note
withnegation,familyhistory,andconditionallanguageisextractedintoassertion-labeledmentions,materializedas
KG edges with temporality, and filtered by intent-aware routing.Bottom: the four routing strategies with formal
operations. TheexampleshowshowaCurrent_Statequeryfiltersoutconditionaledgeswhilepreservingconfirmed
medications.
Alt text: Multi-row workflow diagram. The top row shows clinical note ingestion through extraction, OMOP mapping, graph
construction, retrieval, and answer generation. A highlighted assertion label is preserved across stages. Middle panels show
example note text, extracted mentions, graph edges, and routed evidence. The bottom row compares routing operations for
default, change, current-state, and historical queries.
4

“Does the patient currently have cholelithiasis?”
Category: Historical   |   Expected: “No, resolved”
C1: LLM Alone
No retrieval
No patient data
LLM answers from
parametric knowledge only
“No indication … not in
diagnosis list.”
✗INCORRECT
Right answer, wrong reasoning:
claims missing, not resolved.C4: Assertion KG-RAG
Generic BFS (all edges)
All edges labeled PRESENT
-- no temporal filtering
Sees PRESENT -- answers “Yes.”
No temporal filtering.
✗INCORRECT
Stale PRESENT label from
unfiltered BFS traversal.C4g: Intent-Aware KG-RAG
HISTORICAL route
Temporal diff: listed Adm 1, absent Adm 2
“No, cholelithiasis is resolved.
Temporal status: RESOLVED.”
✓CORRECT
Correct answer + reasoning:
resolved from temporal evidence.
Progressive enrichment:  no context  -->  unfiltered graph  -->  intent-matched temporal evidenceRetrieved subgraph (unfiltered BFS)
Patientcholeli-
thiasisdiabetes
fatty
liver
sleep
apneaHTNRetrieved subgraph (HISTORICAL route)
Patient
choleli-
thiasis
Adm 1
PMHAdm 2
PMHPRESENT PRESENT
PRESENT
PRESENTRESOLVED
listed absentFigure 2: Worked example with retrieved knowledge subgraphs. AHistoricalquestion is answered under three
conditions. C1 (left) has no patient data. C4 (center) retrieves an unfiltered BFS subgraph where all edges are labeled
Present—including a stale label for cholelithiasis (red). C4g (right) appliesHistoricalrouting, filtering to resolved
status with temporal evidence from two admissions. Mini-graph insets show the actual retrieved subgraph structure.
Alt text: Three side-by-side answer cards for one historical question. The left C1 card lacks patient evidence, the center C4 card
retrieves a stale present edge, and the right C4g card filters by historical intent to recover resolved-status evidence across two
admissions.
5

(negation,conditional,uncertainty,family_history,se-
quence,current_state,duration,historical,change). The
v2 reference set began as auto-generated NLP labels with
54 physician corrections and is provisional. Questions
are authored from the same de-identified charts later
ingested—an in-distribution retrieval-faithfulness stress
test, not cross-site generalization. Four ablation condi-
tions progressively add components; two bookends (C6
all-notes,C7deterministic-KG)probedesignboundaries
(Table 1).
Cohort structure.The cohort comprises 43 MIMIC-
IVpatientsselectedasaconveniencesample(noformal
stratification or random sampling); 11 patients have a
single hospital admission, 32 have two admissions, and 0
havethreeormore. The‘cross-admission’framingapplies
cleanly to the 32-patient two-admission subset; on the
11-patient single-admission subset, retrieval still operates
overmultiplenoteswithinanadmission. Demographics
in Appendix Y.
Linguistic categories, not computable phenotypes.
ClinicalBench’s9categories(negation,conditional,uncer-
tainty,family_history,sequence,current_state,duration,
historical, change) arelinguistic constructs adaptedfrom
the i2b2 assertion taxonomy [ 4]. They are NOT com-
putable phenotypes in the PheKB / eMERGE / OHDSI
sense; the per-category C1 →C4g deltas should not be
interpreted as phenotype-validation evidence. Phenotype-
validatableevaluationrequiresOMOP/SNOMEDconcept-
sets, multi-site PPV, and chart-review-validated cohorts
(Newtonetal.,JAMIA2013;Hripcsak&Ryan,JBI2019),
none of which this work provides.
Experiencer-attributiondefects.Post-hocverification
identified8items(qidsinAppendixP.2)withexperiencer-
attribution defects: source section=Family History
butgold expected_answer assertsthediseaseasacur-
rent/historicalconditionofthepatient. Theseareexcluded
from the change-excluded keyword endpoint ( 𝑛=362,
nowreportedasasensitivitycomparator;seeEndpoints
below), reducing the keyword delta from +40.0pp to
+39.5pp. The items remain in the released v2 gold for
transparency; v3 corrections are planned post-publication.
Endpoints.Theprimary endpointis the leave-author-
out paired exact McNemar test on 𝑛=50matched (C1,
C4g) qid pairs from the three-rater externaladjudication,
restricted to Hird ×Nadeem unanimous strict ratings(Section 4.7). This is the substantive author-blind com-
parison;iteliminatessingle-authorcircularityatthecost
of𝑛=362→𝑛=50 statistical power.Secondary / sen-
sitivity: (i)deterministickeywordreproducibilityproxy
on the change-excluded 𝑛=362subset (C4g_keyword
vs. C1; reported with patient-level cluster bootstrap CIs
over43patients—notaclinical-correctnessclaim);(ii)
three-ratermajorityvote( 𝑛=100,subjecttosingle-author
circularity); (iii) oracle C4g upper bound; (iv) hard cross-
admission subset (change ∪current_state∪historical,
𝑛=122, post-selection-inference caveat); (v) C1b vs.
C4g+ extension ( 𝑛=240); (vi) full𝑛=400.Diagnos-
tic: per-category C1 →C4g deltas and the C3 →C4→C4g
decomposition. Thechangecategory is excluded from
secondary(i)duetoknownreferencedefects(Section4.6).
Cohort demographics in Appendix Y.
Endpointpre-registrationandprovenance.Theleave-
author-out paired exact McNemar statistic ( 𝑏=4,𝑐=15,
𝑝=0.0192 )wascomputedandreportedasasensitivity
analysisincommit 0c510d7 (v85.1,2026-04-26),priorto
theJAMIA-10 externalreview;itsunderlyingthree-rater
adjudicationdatawerefrozenearlier(commit 7b388db,
v74) and were not modified subsequently.Promotion
to primary endpoint occurred in v86 (2026-04-27)
post-hoc, in response to external-reviewer feedback
identifyingsingle-authorcircularityasthedominant
threattotheoriginally-declared 𝑛=362keywordpri-
mary(commit 96746e9,v82,2026-04-26). Becauseboth
endpoints are deterministic computations over already-
frozendata,thepromotiondoesnotinvolveadditionaldata
collectionormodelre-running;itdoes,however,change
the headline claim and the statistical-power profile, and
wethereforereporttheleave-author-outprimaryalongside
the originally-declared 𝑛=362keyword endpoint as a
sensitivity. The8-itemexperiencer-attributionexclusion
(𝑛=370→362 ) was identified post-hoc during exter-
nal review (the+0.5ppimpact is documented in §4.1).
The hard cross-admission 𝑛=122subset was selected
post-hocwithapost-selection-inferencecaveatin§results.
Pre-registration was not performed on AsPredicted or
OSF;futureversionsofthisbenchmarkwillpre-register
endpoints prior to data collection.
3.6SliceBench: Complexity-Stratified Case
Study
SliceBenchisasmallcasestudy(6MIMIC-IVpatients,
144questions,threecomplexitytiers)testingwhetherKG-
augmentedretrievalscaleswithrecordcomplexity. Five
6

Table 1: ClinicalBench conditions. C1–C4g: ablation ladder; C6/C7: bookend baselines; C1b/C4g+: extensions
(𝑛=240).
ID Short Name Condition Retrieval Assertion Temporal
C1LLM-aloneLLM Alone None None None
C2TF-IDF RAG+ Vanilla RAG (TF-IDF) TF-IDF doc chunks None None
C2bDense RAG+ Vanilla RAG (dense) Contriever doc chunks None None
C3KG-RAG+ KG-RAG (no assertions) Graph + Doc None None
C4KG-RAG+Assert+ Epistemic KG-RAG Graph + Doc Full (7-class) Bi-temporal+label
C4gKG-RAG+Route+ Intent-Aware KG-RAG Graph + Doc (type-specific) Full (7-class) Bi-temporal+label
C6Long ContextLong Context All notes None None
C7Deterministic KGDeterministic KG KG lookup (no LLM) Full Bi-temporal+label
C1bDischarge OnlyDischarge Summary Discharge doc only None None
C4g+KG-RAG+NotesKG-RAG + Full Notes Graph + Doc + All notes Full (7-class) Bi-temporal+label
conditions(B0–B4)formamonotonecontextprogression;
the critical comparison is B2 →B3, which adds struc-
turedKGcontextwithassertionmetadatawhileholding
documents fixed (Appendix K).
3.7 Evaluation Protocol
ClinicalBenchusesadeterministickeywordevaluator
(v2)—exact word-boundary matching with abstention-
detectiongate(AppendixP).Thephysician-auditedsubset
provides the most credible human accuracy estimate. Pri-
mary answering model: Claude Opus 4.6; cross-model:
MedGemma27B,GPT-OSS20B,Qwen3.535B,Gemma4
31B, MedGemma 1.5 4B.
SliceBenchuses LLM-as-judge with separated answer-
ing/judgingmodels[ 15](ClaudeSonnet4.5answers,Opus
4.6 judges).
Statistical reporting.The author-blind primary end-
point (leave-author-out paired exact McNemar, 𝑛=50)
is reported with two-sided exact binomial 𝑝-values and a
95%NewcombeCIonthepaireddifference;thisendpoint
does not depend on any bootstrap. For the keyword sensi-
tivity endpoint and other secondary contrasts we report
BCa bootstrap 95% CIs ( 𝑛=2,000 , seed 42) [ 25] with
patient-levelclusterbootstrap(43patients)primaryand
question-level secondary (caveat: 𝑛=43clusters is at
the low endof cluster-bootstrap reliability; cf. Cameron
& Miller [ 26]). McNemar’s test [ 27] with Benjamini–
Hochberg FDR correction is used for paired condition
comparisonsandcross-modelcontrasts. Safetyscorewith
asymmetric weighting (𝑤=2.0) in Appendix E.3.8 Physician Adjudication Protocol
The author (board-certified emergency physician, system
designer)conductedablindedinternalauditof120paired
C1/C4g questions with randomized A/B labels, rating
eachonfivedimensions: referencecorrectness,modelcor-
rectness, score fairness, safety, utility (Appendix O). The
adjudication supports C4g >C1 (+35.0pp strict,+31.7pp
lenient; paired exact McNemar 𝑝 <10−8strict) and re-
vealed a 56% reference-answer defect rate.
External physician adjudication.Two independent
physicians (senior attending, 20+ years; resident) com-
pleted the same blinded 100-item protocol, yielding a
three-ratermajorityvotewithFleiss’ 𝜅andexact-binomial
McNemar𝑝-values(Section4.7). Thisisin-distribution
physician adjudication, not multi-site phenotype valida-
tion.
4 Results
EpiKG is evaluated with ClinicalBench (same-record
retrieval-faithfulness stress test) and SliceBench (small
case study on patient complexity). ClinicalBench full-set
resultsuseadeterministickeywordproxy;thephysician-
adjudicated subset provides the most credible human
accuracyestimate;SliceBenchusesLLM-as-judge(Sec-
tion3.7). ClaudeOpus4.6istheprimaryansweringmodel;
cross-modelevaluationspansMedGemma27B,GPT-OSS
20B, Qwen3.5 35B, Gemma 4 31B, and MedGemma 1.5
4B (Appendix P). Both benchmarks report BCa bootstrap
95% CIs (𝑛=2,000, seed 42) [25].
7

C1
LLM AloneC2
+RetrievalC3
+Graph
StructureC4
+AssertionsC4g
(keyword)C4g
(oracle)010203040506070Accuracy (%)(a)
21.8%52.0%50.0%46.2%60.2%68.5%
+30.2+14.0+8.3
C6: Long Context = 59.2%Baseline
Positive delta
Negative delta
-2.0
-3.8
Evidence overload without routing:
47 assertions per query vs. 8 with
keyword routing. Noise drowns
out relevant evidence.
(b)  What changes in the model's answer
C1
LLM Alone"Cannot determine from
available information"
C2
+Retrieval"Pneumonia documented
in admission notes"
C4
+Assertions"Evidence overload: 47 facts
retrieved, conflicting signals"
C4g
+Routing"Pneumonia present admission 1,
resolved by admission 2"
0 20 40 60
Accuracy (%)Opus
GPT-OSS
Qwen 3.5
MedGemmaC4g (keyword) C1 (LLM Alone)(c)  KG benefit generalizes across models
+38.5pp 60.2%
21.8%
+32.3pp 52.5%
20.2%
+21.6pp 57.8%
36.2%
+21.7pp 47.2%
25.5%Figure 3: Ablation results.(a)Waterfall chart showing incremental accuracy changes (Opus, 𝑛=400). Retrieval
provides the largest gain ( +30.2pp); switching from flat to KG-structured retrieval is neutral ( −2.0pp); assertions
without routinghurt( −3.8pp); keyword routing recovers and extends ( +14.0pp); oracle routing adds +8.3pp.
(b)Qualitativeprogressionofmodelanswersacrossconditions.(c)KG-RAGbenefitgeneralizesacrossallsixmodels
(+20–47pp over C1).
Alt text: Three-panel ablation figure. A waterfall plot shows accuracy rising with retrieval, falling slightly with assertions alone,
and rising again with intent routing. A qualitative answer panel compares condition outputs. A cross-model panel shows
KG-RAG gains for every tested model.
8

Table 2: ClinicalBench full-setproxyresults (400 questions, Claude Opus 4.6, keyword evaluator v2 with abstention
detection). Reproducibility scaffolding only — not the primary endpoint (see Section 4.7). Ablation ladder (C1–C4g)
progressively adds components; C4 isolates assertion metadata from intent routing; C6 and C7 are bookend baselines.
C4gkw(keyword routing, secondary/sensitivity) andC4g oracle(oracle routing, upper bound) areshown separately. Sig:
*denotesBCaCIexcludingzero(caveat: 𝑛=43patientsisatthelowendofcluster-bootstrapreliability;cf.Cameron
& Miller [26]). Best inbold.
Condition AccuracyΔvs C1
C1 LLM Alone 21.8% —
C2 + Vanilla RAG (TF-IDF) 52.0%+30.2pp *
C2b + Vanilla RAG (dense) 50.8%+29.0pp *
C3 + KG-RAG (no assertions) 50.0%+28.2pp *
C4 + Assertions (no routing) 46.2%+24.5pp *
C4gkw+ Intent-Aware KG-RAG (keyword)60.2%+38.5pp *
C4goracle+ Intent-Aware KG-RAG (oracle) 68.5%+46.8pp *
C6 Long Context (all notes) 59.2%+37.5pp *
C7 Deterministic KG (no LLM) —†—
4.1 ClinicalBench: Primary Ablation
ClinicalBench provides three evaluators that yield di-
rectionally consistent estimates: physician three-rater
majority+24.0pp (𝑝=0.0075 ; sensitivity, subject to
single-author circularity; Section 4.7), internal author
adjudication+35.0pp strict(descriptive,single-rater;Sec-
tion 4.6), and a deterministic keyword reproducibility
proxy+39.5pp (NOTaclinicalcorrectnessclaim). The
keywordevaluatorisreproducibilityscaffolding—shallow
keyword matching with no polarity check, favoring C4g’s
structured-answer style—and is reported here for replica-
bility,notasthesubstantivecomparison(AppendixP.3).
On the change-excluded 𝑛=362 proxy endpoint (8
experiencer-attribution-defective items excluded), key-
word C4g reaches 62.4% versus C1 22.9% ( +39.5pp;
McNemar𝑝=2.44×10−30); oracle provides a +43.1pp
upperbound(66.0%). C6(longcontext)scores59.2%—
−9.3ppbelow C4g oracle(𝑝=0.001 ), gap concentrated in
current-state(C618.0%vs.C4g70.0%).†C7returnstem-
platerefusals on >98%of questions andis semantically
0%(AppendixX).The author-blindprimaryendpointis
the leave-author-out paired exact McNemar reported in
Section 4.7.
Ablation decomposition.C2 (TF-IDF) and C2b (Con-
triever dense) score comparably (52.0% vs. 50.8%;
𝑝=0.62), and C3 (50.0%) is similar: retrieval method
does not explain the C2 →C4g gap. C4 scores 46.2%—
belowC3 (−3.8pp, n.s.); intent routing recovers and
extends (+14.0pp keyword,+22.3pp oracle;𝑝 <10−6).
Per-category, assertions alone help assertion-sensitive(negation+22.7pp, uncertainty+15.0pp) but degrade
temporal (historical −30.0pp, sequence−45.0pp); rout-
ingreversesthese(AppendixX).TheC2b →C4garchitec-
turaldeltavs.dense-RAGbaselineisreportedseparately
below.
Intent classification sensitivity.Keyword-only C4g
(60.2%)outperformsC4withoutrouting( +14.0pp)and
the best non-KG baseline C2 ( +8.2pp), indicating the
architecture’s value does not require an oracle classifier
(per-category classifier accuracy in Appendix Table 23).
Evaluator hierarchy.LLM-as-judge ( +28.5pp; Ap-
pendix V) corroborates the physician and keyword esti-
mates. Thekeywordevaluatoristoostrictin40%ofcases
vs. too lenient in 5% (7.5:1 ratio).
Circularitydisclosure.TheauthordesignedClinical-
Bench,builtEpiKG,generatedinitialreferenceanswers,
and conducted internal adjudication—a degree of role
overlap that could bias results. Three structural mitiga-
tions bound this risk: frozen public release of evaluator
andpredictions;twoexternalphysiciansconfirmedC4g
under blinded majority vote ( +24.0pp); five additional
LLMsallshowsignificantbenefit( +20.4to+43.1pp or-
acle). Independent replication on a separately authored
benchmark is the definitive test.
Hard cross-admission subset.On the 122-item sub-
set requiring synthesis across ≥2admissions, C4g oracle
9

10 20 30 40 50 60 70
Accuracy (%)Claude Opus 4.6
GPT-OSS 20B
MedGemma 27B
Qwen3.5 35BC4: 46.2%
21.8
20.2
25.5
36.2C1: LLM alone C4g (keyword) C4g (oracle) C4 (46.2%)
+8.3 pp
60.2 68.5
+7.7 pp
52.5 60.2
+9.3 pp
47.2 56.5
-0.8 pp
57.0 57.8Figure 4: Oracle vs. keyword-only intent routing across
the four models with both routing variants. Dumbbells
spanfromkeywordC4g(lightblue)tooracleC4g(dark
blue); gray diamonds show C1 baselines. The dashed line
marksC4(norouting,46.2%). Mostmodelsgainmodestly
from oracle routing; Qwen3.5 (run with think:false
due to an Ollama repetition-penalty issue, Appendix P.1)
does not benefit from oracle routing in this configuration.
Alt text: Dumbbell chart comparing keyword and oracle
routing accuracy by model. Each row includes a low C1
baseline point and higher C4g points. Most models improve
with oracle routing.
reaches 72.1% versus C1 14.8% ( +57.4pp; 95% CI:
[+47.5pp, +66.0pp]).
Architectural novelty: structured intent-aware re-
trieval overa strong dense-RAG baseline.C2b(Con-
triever dense RAG) →C4gkw(intent-aware KG-RAG) on
𝑛=362yields+8.84pp (McNemar𝑝=1.79×10−3);
oracle classification yields +12.43pp . On full𝑛=400,
+9.50pp keyword /+17.75pp oracle. This isolates the
structural-retrieval-with-routing contributionoverarea-
sonable dense-retrieval baseline; it is the defensible ar-
chitectural novelty number, separating retrieval-vs-no-
retrieval from structured-retrieval-vs-flat-retrieval.
4.2Discharge Summary vs. KG-RAG (Cross-
Model Extension)
Aclinicallyrealisticcomparisonpitsdischargesummary
alone(C1b)againstEpiKGwithallnotes(C4g+full)on
𝑛=240acrossfourmodels. Threeoffourmodelsreach
significance under cluster bootstrap; GPT-OSS shows
a directional but non-significant gain: Opus +12.5pp
(57.5%→70.0%), Qwen3.5+10.4pp (58.3%→68.8%),
MedGemma+8.8pp(55.0%→63.8%),GPT-OSS+1.7pp
(60.4%→62.1%,𝑝=0.32). Therange+1.7to+12.5pp is
consistent with structured retrieval over full clinical notesgeneralizing across parameter and training differences
(per-category breakdowns in Appendix Table 16).
4.3 Category×Condition Interaction
Per-category accuracy (Figure 5; Appendix Table 7) re-
vealsacategory×conditioninteraction(diagnosticonly;
𝑛=20–30 per category, no multiplicity correction). C4g
improvesall9categoriesoverC1,withthelargestgains
incross-admissionsynthesis. C4helpsassertion-sensitive
categories but degrades temporal ones, resolved by intent
routing (Appendix Figure 7); C6 lags C4g most on cur-
rent state (−52pp) and conditional ( −5pp), with parity
elsewhere.
4.4 Cross-Model Evaluation
All six models benefit (Table 3; oracle deltas +20.4to
+43.1pp, all𝑞 <10−10after BH-FDR), with benefit
inverse to baseline strength (Appendix Table 7).
To assess whether the convergence of C4g oracleaccura-
cies(range55.8–66.0%) despiteC1 spanning21.8–39.5%
reflectsencodingpartiallysubstitutingforparametercount,
weregressedC1→C4goracledeltaonC1baseline. Slope
𝛽=−1.123 , Pearson𝑟=−0.921 ,𝑝=0.009 (𝑛=6
models). The strong negative slope is consistent with
regression to the mean rather than encoding-substitution;
thesubstitutionhypothesisisthereforenotsupportedby
thisevidenceandisremovedfromthecontributionclaims
(§1).
4.5 SliceBench
A small supporting case study (SliceBench, 6 patients,
144 questions) is consistent with a complexity-dependent
KGeffectbutdoesnotreachaggregatesignificance(Ap-
pendix I).
4.6 Physician Adjudication
A blinded internal audit of 120 paired questions (Sec-
tion 3.8, Appendix X) shows C4g at 62.5% strict / 84.2%
lenientvs.C1at27.5%/52.5%. Deltas: +35.0pp strict
(95% paired Wald CI: [+24.8 pp, +45.2pp]; paired ex-
act McNemar 𝑝 <10−8) and+31.7pp lenient; on a
3-level ordinal score (correct=1, partial=0.5, incorrect=0)
C4g higher in 64/120, tied in 46, lower in 10 (sign test
𝑝 <0.0001 ). The keyword evaluator overestimates the
strict delta by∼5pprelative to physician judgment; all
three methods agree on direction.
10

NegationConditionalUncertainty
Family
History
Sequence
Current
State
Duration
HistoricalChange20%40%60%80%100%
C6: Long Context (59.2%)
C1: LLM Alone (21.8%)
C2: Vanilla RAG (52.0%)
C3: KG-RAG no assert. (50.0%)
C4: +Assertions (46.2%)
C4g: +Intent-Aware (68.5%)Figure 5: ClinicalBench per-categoryaccuracies across six conditions (ClaudeOpus 4.6, evaluator v2). This plot is
descriptiveonly: categorycountsvary,nomultiplicitycorrectionisapplied,andcategorieswithknownevaluation
defects, especially change, can inflate apparent gaps. C4 is lower than C3 overall, while C4g is highest overall.
Alt text: Radar chart showing per-category accuracy for six ClinicalBench conditions. C4g covers the largest area overall,
especially on cross-admission categories. C4 is smaller than C3 in several temporal categories, illustrating the
assertion-without-routing regression.
11

Table 3: ClinicalBench cross-model results (change-excluded keyword sensitivity endpoint, 𝑛=362, keyword
evaluatorv2;thesubstantiveprimaryendpointistheleave-author-outpairedexactMcNemarinSection4.7). C4g oracle
shown as upper bound; all six models benefit (oracle deltas+20.4to+43.1pp, all𝑞<10−10after BH-FDR).
Model C1 C4g oracleΔ
Commercial Claude Opus 4.6 22.93%66.02%+43.1pp
Open GPT-OSS 20B 21.82% 59.39%+37.6pp
Medical MedGemma 27B 27.90% 55.80%+27.9pp
Open-weights Gemma 4 31B 36.74% 61.05%+24.3pp
Medical (small) MedGemma 1.5 4B 35.91% 56.35%+20.4pp
Reasoning Qwen3.5 35B 39.50% 60.77%+21.3pp
Evaluator agreement.Keyword agrees with physician
54.2% (Cohen’s 𝜅=0.18) [28]; on the 106 items with
physician-confirmedcorrectreferences,strictdeltarises
to+41.4pp—gold-standard defects attenuate, not inflate,
the benefit. C4g improves or ties on all 9 categories
(𝑝 <0.002 ); safe rate+15.8pp, helpful rate+36.7pp
(Appendix X).
4.7External Physician Adjudication (Three-
Rater)
Three physicians—Reviewer 1 (A.S., senior internal, 20+
yr), Reviewer 2 (C.H., senior external, 20+ yr), and Re-
viewer 3 (S.N., external resident)—independently rated
the same blinded 100-item subset (50 C1, 50 C4g).
Author-blind primary endpoint: leave-author-out
paired exact McNemar.The author-blind primary end-
point is the leave-author-out paired exact McNemar on 50
matched (C1, C4g) qid pairs from the three-rater external
adjudication, restricted to Hird ×Nadeem unanimous
strict ratings (no inclusion of the author). C1 12/50
(24.0%)→C4g23/50(46.0%), Δ=+22.0pp [95%New-
combe CI:+5.1pp,+31.5pp], two-sided exact McNemar
𝑝=0.0192 (𝑏=4favorC1,𝑐=15favorC4g). Thisisthe
substantive author-blindcomparison: paired,not subject
togold-standardcircularity(thetestiswhetherindepen-
dentratersagreeonsystemoutput),andcomputedfrom
the pre-existing v85.1 commit ( 0c510d7, 2026-04-26).
Promotion of this statistic from a sensitivity to the pri-
mary endpoint occurred in v86 (2026-04-27) post-hoc, in
response to external-reviewer feedback identifying single-
author circularity as the dominant threat to the originally-
declared𝑛=362keyword primary; both endpoints are
deterministiccomputationsoveralready-frozenthree-rater
data(commit 7b388db,v74),andpre-registrationwasnot
performed on AsPredicted or OSF (Section 3.5). Inter-
non-author Cohen’s 𝜅=0.36 (Hird×Nadeem) versus0.43–0.49for author-involving pairs.
Three-rater majority (sensitivity comparator).Un-
der majority vote (Table 4), C4g is correct on 32/50 vs.
C1 on 20/50,+24.0pp (𝑝=0.0075 ); change-excluded
+27.3pp. No reviewer shows an inversion on any non-
changecategory. Fleiss’ 𝜅=0.413 (strict) / 0.615 (gold
correctness)[ 28];pairwiseCohen’s 𝜅0.36–0.49. External
reviewersfound61–64%referencedefectrates(vs.56%
internal). Thismajorityresultissubjecttosingle-author
circularity(Reviewer1isanauthor)andisreportedasa
sensitivity comparator to the primary endpoint above.
Structurally-independent rater.The only structurally-
independent rater (Reviewer 3, no prior author relation-
ship) showed+2.0pp(calibration: 65/100 vs. 39–46/100
forseniors);with23discordantpairs,minimumdetectable
effect is∼22pp, so the resident’s data are consistent with
effects up to that magnitude (Appendix Z.1).
12

Table4: Three-raterexternalphysicianadjudication( 𝑛=100paireditems,50C1/50C4g,blinded). Per-reviewer
C4g−C1strictdeltasareshownwithexact-binomialMcNemar 𝑝-values. Thethree-ratermajorityvoteisreportedasa
sensitivitycomparator; theauthor-blindprimaryendpointistheleave-author-outpairedexactMcNemar(Section4.7).
Reviewer (training level) C1 strict C4g strictΔMcNemar𝑝
Reviewer 1 (senior internal, 20+ yr) 28.0% 64.0%+36.0pp 0.00028
Reviewer 2 (senior external, 20+ yr) 24.0% 54.0%+30.0pp 0.00073
Reviewer 3 (resident external) 64.0% 66.0%+2.0pp 1.0
3-rater majority vote (2+/3) 40.0% 64.0%+24.0pp 0.0075
5 Discussion
Whylongcontextunderperforms.Longcontextforces
the model to do two things at once: extract structure
(negation,temporality,experiencer)andreasontowardthe
answer. KG-RAG does thestructural work upstream and
hands the LLM a small set of typed facts. C6 (59.2%)
trails C4g_oracle (68.5%) by −9.3pp(𝑝=0.001 ), and
the gap is concentrated in current_state (C6 18.0% vs.
C4g 70.0%). GPT-OSS 20B narrows its gap to Opus
4.6 once both get structured context (59.4% vs. 66.0%).
The keyword evaluator is a reproducibility tool, not a
clinical-correctnessclaim: itscoreskeywordpresenceand
skips polarity for uncertainty, family history, conditional,
current_state, and historical questions. These shallow
rulesfavorC4g’sstructuredanswersoverC1’sabstentions,
so the keyword+39.5pp overestimates the true gain. The
physician adjudication ( +24to+36pp) is the comparison
that matters; the keyword number is its deterministic
proxy.
Cross-model convergence.All six models gain (Ta-
ble 3; oracle deltas +20.4to+43.1pp, all𝑞 <10−10
after BH-FDR). C4g_oracle accuracies land in a nar-
row55.8–66.0%band even though C1 baselines span
21.8–39.5%. Regressing the C1 →C4g delta on the C1
baseline gives 𝛽=−1.123 ,𝑟=−0.921 ,𝑝=0.009 . With
C4g_oracleclosetothekeywordevaluator’smeasurable
ceiling(∼70%),Tu(BMJ2005)showedthatpairedpre-
post designs can produce slopes near −1as a ceiling
artifact,sowecannottellwhetherstructuredretrievalis
substitutingformodelsizeorwhetherweareseeingregres-
siontothemean. With 𝑛=6models,thisisdescriptive,
notinferential. Onesignthatrepresentationaloneisnot
free: C4(assertionswithoutrouting)drops −3.8ppbelow
C3 (𝑝=0.26, n.s.; Appendix J). Routing in C4g recovers
and extends the gain.Possible implications beyond clinical NLP.Adding
metadata without aligning retrieval to it can hurt perfor-
mance, as the C3→C4 step shows. The C3 →C4→C4g
ablation may be useful as a test pattern wherever a system
extractsstructuredannotationsbutdoesnotintegratethem
into retrieval.
Evaluator calibration and reference-answer qual-
ity.The three evaluators agree directionally: key-
word+39.5pp,physician+24to+36pp,LLM-as-judge
+28.5pp. ThekeywordandLLM-judgemagnitudesdiffer,
butbothcomparethesameC1andC4ganswersagainst
thesamegoldstandard,soanyper-answerevaluatorbias
cancelsin thewithin-modelC1 →C4gpaired delta;what
survivesisanarchitecturalsignalthatthephysicianadjudi-
cation then independently corroborates. Physician review
found56%ofv2referenceanswersdefective. Thecauseis
a systematic NLPassertion-classifier error: the classifier
read “history of CHF” as “resolved” when clinical usage
means“chronic.” Themodeliscorrect59%ofthetime
whenthereferenceiswrong,and46%ofthetimewhen
it is right, so noisy labels appear to underestimate true
performance rather than inflate it. We do not report a
v3goldrescoreinthismanuscript: thecorrectionswere
identified during physician adjudication, applying them
andre-runningeverymodel ×conditionwouldconstitute
a second freezing of the benchmark, and we prefer to
keep v2 as the public test set with v3 reserved for an
independent follow-up.
Bi-temporal versus tri-temporal modeling.We use
the term “tri-temporal” loosely. The system stores bi-
temporal edges (valid time and transaction time, in the
Snodgrasstradition;cf.Graphiti[ 11])andaddsanNLP-
derived temporality label 𝜏𝑎∈{Past, Current, Future} .
The label comes from clinical NLP, not database time,
so the data model is bi-temporal in the strict sense. C4
enables the assertion and temporality axes together, so we
cannot separate the temporal contribution to the C4g gain.
13

That separation is future work.
Statistical caveats.BCa cluster bootstrap with 43 pa-
tientsisatthelowendofreliability(Cameron&Miller[ 26]
note≥50clustersispreferable);CIsshouldbeinterpreted
with this caveat. The hard cross-admission 𝑛=122
subsetwasselectedpost-hocandmeritsapost-selection-
inference caveat (cf. Berk et al. [ 29]). Cross-model
BH-FDR is reported; the more conservative Benjamini-
Yekutieli (2001)procedure underPRDS-violating depen-
denceyieldsequivalentconclusions(BYadjustmentfactor
2.45×; all𝑞<2.5×10−10).
Multi-site phenotype-validation roadmap.The cur-
rent cohort comprises 43 MIMIC-IV patients (BIDMC
ICU).Multi-sitevalidationintheeMERGE/OHDSItradi-
tion would require≥5sites with per-site PPV. Minimum-
viable transportabilitychecks (MIMIC-III ↔MIMIC-IV
intra-cohort, eICU,Syntheanarrative)areflaggedasfu-
ture work. NLP-portability literature [ 30] projects 10–
20ppPPV degradation on transport, which we have not
measured.
Deployment-realism scoping.ClinicalBench Opus
C1/C3/C4g require ∼22min per condition per patient
via API; the system has not been packaged for sub-
second EHR-sidebar latency, FURM-style governance
review [31], CHAI AssuranceReporting Checklist align-
ment,orpost-deploymentalgorithmovigilance[ 32]. These
are deployment prerequisites, not paper findings; a candi-
date SaMD/PCCP/CHAI/algorithmovigilance scaffold is
sketched in Appendix Z.5.
Threatstovalidityandclaimscope.ClinicalBench-v2
is a provisional in-distribution stress test (43 patients, 400
questions, MIMIC-IV [ 24]) measuring retrieval fidelity,
not external generalization or phenotype validation in
the eMERGE/PheKB sense. The change-excluded key-
word endpoint (now reported as a sensitivity, 𝑛=362)
excludes 8 experiencer-attribution-defective items (Ap-
pendix P.2); they remain in the released v2 gold for
transparency. Evaluator uncertainty is substantial (54%
keyword–physicianagreement);conclusionsfocusondi-
rectional agreement across three evaluators. The author’s
combined role as designer, builder, and primary eval-
uator creates circularity risk structurally mitigated but
not eliminated (Section 4.1); inter-non-author Cohen’s
𝜅=0.36(Hird×Nadeem) further attenuates the external-
adjudication signal; the leave-author-out 3-rater majority
is+22.0pp (paired exact McNemar 𝑝=0.0192 ; 4 vs.15 discordant pairs) versus +24.0pp with the author in-
cluded,supportingbothdirectionandsignificance. Model
dependenceisreal( +20.4–43.1pporacle,afterBH-FDR).
Thisworkperforms in-distributionevaluationandphysi-
cian adjudication, not multi-site phenotype validation;
it doesnotestablish clinical deployment readiness, and
multi-site replication and a prospective clinician-with-
systemcomparisonareencouraged. Detailedthreatsand
broader-impact discussion in Appendix Z.
6 Conclusion
The author-blind primary endpoint is leave-author-out
paired exact McNemar (Hird ×Nadeem unanimous strict,
𝑛=50; promoted post-hoc to primary in v86 from a
pre-existing v85.1 sensitivity in response to external re-
view—notpre-registered): +22.0pp [+5.1pp,+31.5pp],
𝑝=0.0192 . The architectural novelty is C2b (Contriever
dense-RAG)→C4g_kw(intent-awareKG-RAG)onthe
change-excluded 𝑛=362endpoint:+8.84pp (McNemar
𝑝=1.79×10−3). Sensitivities: three-rater physician
majority+24.0pp (𝑝=0.0075 ; subject to single-author
circularity since the author is R1); deterministic keyword
proxy+39.5pp (reproducibility proxy only, not a clini-
cal correctnessclaim). Cross-modelaccuracies converge
with strong negative slope vs C1 baseline ( 𝛽=−1.123 ,
𝑟=−0.921 ,𝑝=0.009 ), consistent with regression to
the mean rather than encoding substituting for parame-
ter count. The 56%reference-answer defect rate under-
scores a methodological lesson: automated NLP-pipeline
benchmarks require physician adjudication. Reasoning
benchmarkslikeHealthBenchmeasurethelastmilegiven
clean vignettes; retrieval-faithfulness on real charts is
the first mile, and representation is the undermeasured
prerequisite.
Acknowledgments
The author thanks non-author contributors Cindy Hird,
MDandShaheeraNadeem,MDforindependentphysician
adjudication of ClinicalBench items, and Yu Tian, PhD
(University of Central Florida) for feedback on an early
draft. MIMIC-IV data were accessed under PhysioNet
Credentialed Health Data Use Agreement.
Funding.This research received no specific grant from
any funding agency in the public, commercial, or not-for-
profit sectors.
14

Conflict of Interest.The author serves as founder of
Sulci.ai,aclinical-AIstartupthatisexploringdeployment
of EpiKG-derived technology. No commercial relation-
shipfundedthismanuscriptortheunderlyingexperiments.
The University of Central Florida is the institution of
academic record for this work.
AuthorContributions.AlexStinard,MD:Conceptu-
alization, Methodology, Software, Validation, Formal
Analysis, Investigation, DataCuration, Writing–Original
Draft, Writing – Review & Editing, Visualization, Project
Administration.
DataAvailability.ClinicalBenchquestions,reference
answers (v1 and v2), raw model predictions, the deter-
ministic keyword evaluator, and physician adjudication
data are publicly available at https://huggingface.
co/datasets/alexstinard/epikg-clinicalbench
(DOI: 10.57967/hf/8549) [ 33]. MIMIC-IV clinical notes
requireseparatecredentialedaccessviaPhysioNet( https:
//physionet.org/content/mimiciv/3.1/).
UseofAITools.ClaudeCode(Anthropic,ClaudeOpus
4.6)wasusedasaprogrammingassistantduringsystem
development,dataanalysis,andmanuscriptpreparation.
All AI-generated content was reviewed, verified, and
edited by the author. Claude Opus 4.6 is also the primary
answeringmodelevaluatedinClinicalBenchexperiments;
this dual role is disclosed throughout the paper.
References
[1]Karan Singhal, Shekoofeh Azizi, Tao Tu, S. Sara
Mahdavi, Jason Wei, Hyung Won Chung, Nathan
Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen
Pfohl,etal. Towardsexpert-levelmedicalquestion
answering with large language models.Nature, 620:
399–404, 2023. doi: 10.1038/s41586-023-06291-2.
[2]Khaled Saab, Tao Tu, Xavier Amatriain, et al. Capa-
bilitiesofGeminimodelsinmedicine.arXivpreprint
arXiv:2404.18416, 2024.
[3]TaoTu, AnilPalepu,MikeSchaekermann, Khaled
Saab, Jan Freyberg, Ryutaro Tanno, Amy Wang,
Brenna Li, Mohamed Amin, Nenad Tober, et al.
Towards conversational diagnostic AI.Nature, 2024.
[4]ÖzlemUzuner,BrettR.South,ShuyingShen,and
Scott L. DuVall. 2010 i2b2/VA challenge on con-cepts, assertions, and relations in clinical text.Jour-
naloftheAmericanMedicalInformaticsAssociation,
18(5):552–556, 2011.
[5]VeyselKocaman,YigitGul,M.AytugKaya,Hasham
Ul Haq, Mehmet Butgul, Cabir Celik, and David
Talby. Beyondnegationdetection: Comprehensive
assertion detection models for clinical NLP. In
Text2Story Workshop at European Conference on
Information Retrieval (ECIR), 2025.
[6]OHDSI Collaborative. OMOP common data model
v5.4.Observational Health Data Sciences and In-
formatics, 2024.
[7]Darren Edge et al. From local to global: A graph
RAG approach to query-focused summarization.
arXiv preprint arXiv:2404.16130, 2024.
[8]Tianjun Luo et al. GFM-RAG: Graph foundation
model for retrieval augmented generation. InAd-
vances in Neural Information Processing Systems,
2025.
[9]Peng Jiang et al. KARE: Knowledge graph aug-
mentedreasoningviallmsforclinicaldecisionsup-
port. InInternational Conference on Learning Rep-
resentations, 2025.
[10]Junde Wu et al. Medical-Graph-RAG: Towards safe
medical large language model via graph retrieval-
augmentedgeneration. InProceedingsofthe63rd
AnnualMeetingoftheAssociationforComputational
Linguistics, 2025.
[11]Preston Rasmussen, Pavlo Paliychuk, Travis Beau-
vais,JackRyan,andDanielChalef. Zep: Atemporal
knowledge graph architecture for agent memory.
arXiv preprint arXiv:2501.13956, 2025.
[12]Marco Postiglione, Daniel Bean, Zeljko Kraljevic,
RichardDobson,andVincenzoMoscato. Predicting
future disorders via temporal knowledge graphs and
medicalontologies.IEEEJournalofBiomedicaland
Health Informatics, 28(7):4238–4248, 2024. doi:
10.1109/JBHI.2024.3390419.
[13]OpenAI. HealthBenchProfessional: Evaluatingclin-
icalreasoning inlarge languagemodels. https://
openai.com/research/healthbench ,2026. Ac-
cessed 26 April 2026.
[14]Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung
Weng,HanyiFang,andPeterSzolovits.Whatdisease
15

does this patient have? a large-scale open domain
question answering dataset from medical exams.
Applied Sciences, 11(14):6421, 2021. doi: 10.3390/
app11146421.
[15]Guangzhi Xiong et al. MIRAGE: Medical infor-
mation retrieval-augmented generation evaluation.
InFindings of the Association for Computational
Linguistics: ACL, 2024.
[16]AnusriPampari,PreethiRaghavan,JenniferLiang,
and Jian Peng. emrQA: A large corpus for question
answering on electronic medical records. InPro-
ceedings of the Conference on Empirical Methods
in Natural Language Processing, pages 2357–2368,
2018.
[17]Wei Chen et al. Multi-LLM KG-RAG: End-to-
end clinical knowledge graph construction.arXiv
preprint arXiv:2601.01844, 2026.
[18]LangLietal. AutoRD:Anautomaticandend-to-end
systemforrarediseaseknowledgegraphconstruction.
JMIR Medical Informatics, 12, 2024.
[19]Rakhilya Lee Mekhtieva, Brandon Forbes, Dalal
Alrajeh, Brendan Delaney, and Alessandra Russo.
RECAP-KG:MiningknowledgegraphsfromrawGP
notesforremoteCOVID-19assessmentinprimary
care.arXiv preprint arXiv:2306.17175, 2023.
[20]Wendy W. Chapman, Will Bridewell, Paul Hanbury,
Gregory F. Cooper, and Bruce G. Buchanan. A
simple algorithm for identifying negated findings
and diseases in discharge summaries.Journal of
Biomedical Informatics, 34(5):301–310, 2001.
[21]Henk Harkema, John N. Dowling, Tyler Thornblade,
and Wendy W. Chapman. ConText: An algorithm
for determining negation, experiencer, and temporal
statusfromclinicalreports.JournalofBiomedical
Informatics, 42(5):839–851, 2009.
[22]Claude E. Shannon. A mathematical theory of com-
munication.Bell System Technical Journal, 27(3):
379–423, 1948.
[23]Yixing Jiang, Kameron C. Black, Gloria Geng,
Danny Park, James Zou, Andrew Y. Ng, and
JonathanH.Chen. MedAgentBench: AvirtualEHR
environment to benchmark medical LLM agents.
NEJM AI, 2(9), 2025. doi: 10.1056/AIdbp2500144.[24]Alistair E.W. Johnson, Lucas Bulgarelli, Lu Shen,
etal.MIMIC-IV,afreelyaccessibleelectronichealth
record dataset.Scientific Data, 10:1, 2023.
[25]Bradley Efron and Robert J. Tibshirani.An Intro-
ductiontotheBootstrap. ChapmanandHall/CRC,
1993.
[26] A. Colin Cameron and Douglas L. Miller. A practi-
tioner’s guide to cluster-robust inference.Journal of
Human Resources, 50(2):317–372, 2015.
[27]Quinn McNemar. Note on the sampling error of
the difference between correlated proportions or
percentages.Psychometrika, 12(2):153–157, 1947.
[28]J. Richard Landis and Gary G. Koch. The mea-
surement of observer agreement for categorical data.
Biometrics, 33(1):159–174, 1977.
[29]Richard Berk, Lawrence Brown, Andreas Buja, Kai
Zhang,andLindaZhao. Validpost-selectioninfer-
ence.Annals of Statistics, 41(2):802–837, 2013.
[30]Andre Bittar, Sumithra Velupillai, Johnny Downs,
Rosemary Sedgwick, and Rina Dutta. Portability
of natural language processing methods to detect
suicidalityfromunstructuredclinicaltextinusand
uk electronic health records.Journal of the Amer-
ican Medical Informatics Association Open, 6(3):
ooad078, 2023. doi: 10.1093/jamiaopen/ooad078.
[31]Nigam H. Shah, John D. Halamka, Suchi Saria,
MichaelPencina,TroyTazbaz,MickyTripathi,Al-
ison Callahan, Hailey Hildahl, and Brian Ander-
son. A nationwide network of health ai assurance
laboratories.JAMA, 331(3):245–249, 2024. doi:
10.1001/jama.2023.26930.
[32]Peter J. Embi. Algorithmovigilance—advancing
methods to analyze and monitor artificial
intelligence–driven health care for effectiveness and
equity.JAMA Network Open, 4(4):e214622, 2021.
doi: 10.1001/jamanetworkopen.2021.4622.
[33]Alex Stinard. [dataset] ClinicalBench: Assertion-
sensitive clinical question answering bench-
mark. https://huggingface.co/datasets/
alexstinard/epikg-clinicalbench ,2026.Ac-
cessed 25 April 2026.
[34]WernerCeustersandBarrySmith. Aboutness: To-
wards foundations for the information artifact on-
tology.International Conference on Biomedical
Ontology, 2015.
16

[35]Richard T. Snodgrass.Developing Time-Oriented
Database Applications in SQL. Morgan Kaufmann,
2000.
[36]JamesF.Allen. Maintainingknowledgeabouttem-
poral intervals.Communications of the ACM, 26
(11):832–843, 1983.
[37]Fei Li, Jianfu Hong, Cui Tao, et al. TEO: A time
event ontology for clinical narratives.Journal of the
AmericanMedicalInformaticsAssociation,27(10):
1560–1568, 2020.
[38]YanHuang,XiaojinLi,andGuo-QiangZhang. Tem-
poral cohort logic.AMIA Annual Symposium Pro-
ceedings, 2022:1237–1246, 2023.
[39]Yifan Lu, Tianyu Fu, et al. Doctorrag: Medical
rag emulating doctor-like reasoning. InAdvances in
Neural Information Processing Systems, 2025.
[40]Yusheng Wang et al. MedRAG: Enhancing medical
diagnosis through retrieval-augmented generation
withknowledgegraph-elicitedreasoning.Proceed-
ings of The Web Conference, 2025.
[41]Samuel Thio, Matthew Lewis, Spiros Denaxas,
and Richard J. B. Dobson. Unlocking electronic
health records: A hybrid graph RAG approach to
safe clinical AI for patient QA.arXiv preprint
arXiv:2602.00009, 2025.
[42]Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo,
HaizhouShi,ChuntaoHong,YanZhang,andSiliang
Tang. Graph retrieval-augmented generation: A
survey.ACMTransactionsonInformationSystems,
2025. doi: 10.1145/3777378.
[43]Lang Cao, Qingyu Chen, and Yue Guo. EHR-RAG:
Bridginglong-horizonstructuredelectronichealth
records and large language models via enhanced
retrieval-augmented generation.arXiv preprint
arXiv:2601.21340, 2026.
[44]Yanjun Gao, Ruizhe Li, Emma Croxford, John
Caskey,BrianW.Patterson,MatthewChurpek,Tim-
othy Miller, Dmitriy Dligach, and Majid Afshar.
Leveraging medical knowledge graphs into large
language models for diagnosis prediction: Design
and application study.JMIR AI, 4(1):e58670, 2025.
doi: 10.2196/58670.
[45]Gautier Izacard, Mathilde Caron, Lucas Hosseini,
Sebastian Riedel, Piotr Bojanowski, Armand Joulin,andEdouardGrave.Unsuperviseddenseinformation
retrieval withcontrastive learning.Transactions on
Machine Learning Research, 2022.
[46]YifanPeng,XiaosongWang,LeLu,Mohammadhadi
Bagheri,RonaldSummers,andZhiyongLu.NegBio:
Ahigh-performancetoolfornegationanduncertainty
detection in radiology reports.AMIA Summits on
Translational Science Proceedings, 2018.
17

A Notation Summary
Symbol Meaning
𝛼∈AAssertion label (7 values, Eq. 1)
𝝉𝑣 Valid time (event date, valid from/to)
𝝉𝑡 Transaction time (recorded at, doc date, created at)
𝜏𝑎 NLP-asserted temporality (Current / Past / Future)
𝑟∈RTemporal relation (9 values, Table 22)
𝑐Confidence score∈[0,1]
𝜉Experiencer (Patient / Family)
𝑒(𝑚)Epistemic state tuple(𝑐,𝛼,𝜉,𝜏)
𝑞,𝜋Clinical question, patient
𝜄Question intent (Change / Current_State / Historical / Default)
𝑓np(𝑐)Fraction of non-present assertions for concept𝑐
Table 5: Key notation used throughout the paper.
B Formal Epistemic Preservation
The epistemicinvariantis formalized for theepistemic invariant maintainedbythe EpiKG pipeline, buildingon the
principle that aboutness—the relationship between information artifacts and the entities they represent—must be
preserved across transformations [34].
Definition1(EpistemicState).Foraclinicalmention 𝑚,theepistemicstateisthetuple 𝑒(𝑚)=(𝑐,𝛼,𝜉,𝜏) ,where
𝑐is the OMOP concept identifier, 𝛼∈Ais the 7-value assertion label (Eq. 1), 𝜉∈{Patient,Family} is the
experiencer, and 𝜏∈{Current,Past,Future} is the temporality. A pipeline 𝑃epistemically preservesmention 𝑚if
𝑃(𝑒(𝑚))=𝑒(𝑚) ; that is, the epistemic state at the output of every pipeline stage is identical to the state at extraction.
Proposition 2(Assertion Entropy Loss).For concept 𝑐in patient𝜋’s record, let 𝐴𝜋
𝑐denote the assertion distribution
with empirical frequencies𝑝(𝛼 𝑖)=𝑛 𝑖/Í
𝑗𝑛𝑗over the|A|assertion classes. The assertion entropy is
𝐻(𝐴𝜋
𝑐)=−∑︁
𝑖𝑝(𝛼𝑖)log𝑝(𝛼 𝑖).(2)
An assertion-blind pipeline collapses all mentions to 𝛼=Present , yielding a degenerate distribution with 𝐻=0.
The information loss is Δ𝐻=𝐻(𝐴𝜋
𝑐)≥0[22], withΔ𝐻 >0strictly whenever any mention carries a non-present
assertion.
Proof.Under the assertion-blind mapping 𝜙:𝛼 𝑖↦→Present for all𝑖, the output distribution assigns probability 1 to
Presentand0toallotherclasses,so 𝐻(𝜙(𝐴𝜋
𝑐))=0. Bynon-negativityofShannonentropy, Δ𝐻=𝐻(𝐴𝜋
𝑐)−0=
𝐻(𝐴𝜋
𝑐)≥0, with equality iff all mentions already carry𝛼=Present.□
Corollary3(FaithfulnessBound).Let 𝑓np(𝑐)denotethefractionofmentionsofconcept 𝑐carryinganon-present
assertion (𝛼≠Present ). Without assertion labels, the maximum assertion-faithful accuracy for any downstream
task conditioned on 𝑐is bounded by 1−𝑓np(𝑐). In clinical records where negated and uncertain mentions are
prevalent—e.g., “no pneumonia,” “possible CHF”—this bound can be substantially below 1.
Proof.Without assertion labels, any predictor must assign a single assertion class to all mentions of 𝑐. Choosing
Present(themajorityclassinclinicaltext)yieldsaccuracy 1−𝑓np(𝑐);the𝑓np(𝑐)non-presentmentionsarenecessarily
misclassified. □
18

C Extended System Design
C.1 Temporal Knowledge Graph Model (Bi-Temporal Storage + NLP-Asserted Label)
Clinicaleventsunfold acrossmultipletimedimensions thatpriorsystemsmodel incompletely[ 11,12]. Bi-temporal
databases [ 35] distinguish validtime from transaction time; EpiKG’s per-edge representation addsan NLP-asserted
temporality label𝜏 𝑎and stores all three on every KG edge (data-modeling clarification below):
Definition4(TemporalEdge—bi-temporalstoragewithNLP-assertedlabel).Anedge 𝑒=(𝑠,𝑝,𝑜,𝛼,𝝉 𝑣,𝝉𝑡,𝜏𝑎,𝑟,𝑐)
where:
•𝑠,𝑜are source and target nodes;𝑝is the predicate (one of 24 edge types);
•𝛼∈Ais the assertion label (Eq. 1);
•𝝉𝑣=(event_date,valid_from,valid_to)is thevalid timeinterval—when the relationship held in the real world;
•𝝉𝑡=(recorded_at,doc_date,created_at)is thetransaction time—when it was recorded;
•𝜏𝑎∈{Current,Past,Future}is theNLP-asserted temporality;
•𝑟∈Ris an Allen interval algebra relation [36];𝑐∈[0,1]is temporal confidence.
Rcomprises seven of Allen’s 13 canonical relations [ 36] (merging symmetric pairs like Before/Meets) plus
ConcurrentandUnknown(ninetotal;fullmappinginTable22). UnlikeTEO[ 37]andTCL[ 38],whichuseAllen’s
relations as annotation-ontology classes or modal operators, EpiKG stores 𝑟and𝑐directly on materialized edges; this
is intended to enable temporal-interval filtering during traversal, although the current intent-aware retrieval algorithm
queries the categorical𝜏 𝑎label rather than Allen-relation predicates (see clarification below).
Data-modelingclarification (bi-temporalstorage +derived label).Whatwedescribe as“tri-temporal” ismore
precisely abi-temporal storage layer(valid time 𝝉𝑣+ transaction time 𝝉𝑡, in the Snodgrass tradition [ 35]; cf.
Graphiti [ 11]) plus anNLP-asserted temporality label 𝜏𝑎∈{Past,Current,Future} . The label is derived from
clinicalNLPratherthanfromdatabasetime,soondata-modelinggroundsthemodelisbi-temporal-with-derived-
attribute, not strictly tri-temporal. Allen-style interval relations 𝑟∈Rare stored on edges as metadata, but the current
intent-awareretrievalalgorithmqueriesthecategorical 𝜏𝑎labelratherthanAllen-relationpredicates;usingstored
intervals in retrieval (e.g., to enforceBefore/Overlapsconstraints) is flagged as future work.
C.2 Graph-Augmented Retrieval Details
Given a clinical question 𝑞and patient𝜋, the base retrieval pipeline: (1) extracts concepts from 𝑞via NLP + OMOP
enrichment; (2) traverses 2–3 hops via bidirectional breadth-first search (BFS) over patient KG edges and OMOP
vocabularyrelationships(20M+edges)viaPostgreSQLcommontableexpressions(CTEs)(AppendixL),pruning
edges with𝑐 <0.3; (3) groups edges into four temporal views (event timeline, current state, historical, conflicts);
(4) retrieves matching guideline sections; (5) scores edges by:
score(𝑒)=𝑐(𝑒)+⊮[type(𝑒)∈𝑄 rel]·0.2+⊮[𝜏 𝑎(𝑒)=Current]·0.1(3)
where the bonuses for question-relevant edge types ( 0.2) and current temporality ( 0.1) were set by manual tuning on a
development set of 20 questions. The surviving subgraph is serialized into structured text preserving assertion labels
(e.g., “Absent: pneumonia”); and (6) composes graph evidence, temporal context, guidelines, and source documents
into a single prompt.
D Gap Analysis
Partial marks indicate limited coverage: GFM-RAG, MedRAG, and KARE build population-level or hierarchical
KGs (notper-patient cross-admission graphs); KARE usesKG-augmented retrievalbut without assertion awareness;
Multi-LLM KG models uncertainty via entropy but not the full assertion taxonomy; MedTKG and Graphiti carry
bi-temporal annotations but lack the assertion-aware retrieval of EpiKG. MediGRAF is the closest patient-level
19

Table 6: Capability comparison across representative systems.✓= supported,◦= partial,×= not supported.
System Patient KGOMOP mapping Assertion (7-class) Bi-temp.+labelAssert.-aware RAG ExperiencerAllen’s algebraMulti-hop (≥3)
DoctorRAG [39]× × × × × × × ×
GFM-RAG [8]◦ × × × × × ×✓
MedRAG [40]◦ × × × × × × ◦
KARE [9]◦ × × × ◦ × ×✓
Multi-LLM KG [17]×✓◦ × × × × ×
MedTKG [12]✓× × ◦ × × ×✓
Graphiti [11]× × × ◦ × × ×✓
MediGRAF [41]✓× × × × × ×✓
EpiKG (this work)✓ ✓ ✓ ✓ ✓ ✓ ✓◦
competitor,constructingper-patientmedicalgraphsfromclinicalnotes,butitdoesnotpreserveassertionstatusor
temporal relations as edge attributes. EpiKG’s partial mark on multi-hop traversal ( ≥3hops) reflects a deliberate
architectural trade-off: PostgreSQL CTE-based traversal provides ACID compliance but degrades beyond 2 hops
(Appendix L).
D.1 Extended Related Work Discussion
This subsection expands the four-axis related work overview in Section 1.1.
Medical RAG systems.Graph-augmented retrieval has emerged as a leading paradigm for medical QA.
GraphRAG [ 7] introduced community-based summarization; GFM-RAG [ 8] trained a graph foundation model
across 60KGs; KARE [ 9] adaptedcommunity retrieval to clinicaldecision support; Medical-Graph-RAG [ 10] links
documentsviatriplegraphs. However,nopriorclinicalKG-RAGsystemjointlypropagatesnote-derivedassertion
classestogetherwithbi-temporalstorageplusanNLP-assertedtemporalitylabelasfirst-classgraphpropertiesthrough
extraction,storage,andretrieval. GFM-RAGandKAREbuildpopulation-levelgraphsratherthanpatient-levelgraphs
fromclinicaltext,andarecentsurvey[ 42]doesnotidentifysystemsthatcarryepistemicmetadatathroughthefull
retrieval stack.
ClinicalQAbenchmarksandsystems.MedPaLM2[ 1],Med-Gemini[ 2],andAMIE[ 3]targetmedicalknowledge
recallfrom parametric knowledge or literature. ClinicalBench targets a narrower task:assertion-faithful cross-
admissionreasoningoverrealEHRrecords,wherethechallengeisknowingwhetherthispatientisstillonmetformin,
whether a condition was ruled out or confirmed, and how the clinical picture changed across admissions. Prior
EHR QA benchmarks—emrQA [ 16], EHR-RAG [ 43], MIRAGE [ 15]—evaluate medically grounded retrieval but not
assertion-sensitive, cross-admission QA with category×condition ablations and physician adjudication.
Clinical KG construction.Multi-LLM KG-RAG [ 17] uses multi-agent prompting with schema-constrained
extractionforoncology;AutoRD[ 18]andRECAP-KG[ 19]applyLLMstorarediseaseandGPnotesrespectively.
All share a common limitation: assertion status is not propagated into the final graph.
Assertion detection.NegEx [ 20] introduced trigger-based negation; ConText [ 21] extended it with temporality
andexperiencer;Guletal.[ 5]fine-tunedLLMsto0.962accuracyonthei2b2/VAtaxonomy[ 4]. Alltreatassertion
20

detection as a terminal annotation task: labels are not carried into knowledge graphs or retrieval systems, and are not
temporally situated across encounters.
Temporalknowledgegraphs.MedTKG[ 12]constructstemporalKGswithtime-stampedsnapshots(eventtime
only); Graphiti [11] implements bitemporal edges but lacks clinical ontology alignment.
E Safety Score
The safety score 𝑆=1−1
𝑁Í𝑁
𝑖=1𝑤𝑖·⊮[ˆ𝑦 𝑖≠𝑦𝑖]applies𝑤𝑖=2.0for false-positive assertion errors (reporting a
negated or absent condition as present) and 𝑤𝑖=1.0otherwise, capturing asymmetric clinical risk where acting on a
falsely affirmed condition is more dangerous than missing a present one. The value 𝑤=2.0is treated as a reasonable
default; sensitivity to this choice is a direction for future work.
F Cross-Model Per-Category Results
Table7: ClinicalBenchper-categoryaccuracy(%)bymodelunderC1andC4g(intent-awareKG-RAG,evaluator
v2)forOpus,MedGemma27B,GPT-OSS,Gemma431B,MedGemma1.54B,andQwen3.5. Theseper-category
improvements are descriptive; several categories are small, and the change category has known label/evaluator defects
discussed in Section 4.6.
Claude Opus 4.6 MedGemma 27B GPT-OSS 20B Gemma 4 31B MedGemma 1.5 4B Qwen3.5 35B
Category𝑛C1 C4gΔC1 C4gΔC1 C4gΔC1 C4gΔC1 C4gΔC1 C4gΔ
Negation 110 44.5 81.8↑37.3pp 57.3 77.3↑20.0pp 45.5 80.9↑35.5pp 74.5 82.7↑8.2pp 59.1 64.5↑5.4pp 69.1 85.5↑16.4pp
Conditional 20 0.0 45.0↑45.0pp 15.0 35.0↑20.0pp 0.0 15.0↑15.0pp 0.0 45.0↑45.0pp 5.0 25.0↑20.0pp 5.0 20.0↑15.0pp
Uncertainty 40 12.5 50.0↑37.5pp 7.5 32.5↑25.0pp 5.0 30.0↑25.0pp 17.5 30.0↑12.5pp 17.5 45.0↑27.5pp 37.5 32.5↓5.0pp
Family hist. 30 3.3 56.7↑53.3pp 3.3 40.0↑36.7pp 3.3 40.0↑36.7pp 23.3 83.3↑60.0pp 3.3 33.3↑30.0pp 26.7 40.0↑13.3pp
Sequence 40 7.5 62.5↑55.0pp 2.5 37.5↑35.0pp 7.5 32.5↑25.0pp 17.5 40.0↑22.5pp 40.0 75.0↑35.0pp 25.0 57.5↑32.5pp
Current st. 50 26.0 70.0↑44.0pp 32.0 58.0↑26.0pp 36.0 56.0↑20.0pp 12.0 66.0↑54.0pp 36.0 58.0↑22.0pp 36.0 56.0↑20.0pp
Duration 30 33.3 60.0↑26.7pp 46.7 63.3↑16.7pp 16.7 66.7↑50.0pp 60.0 63.3↑3.3pp 53.3 86.7↑33.4pp 40.0 86.7↑46.7pp
Historical 50 6.0 60.0↑54.0pp 0.0 46.0↑46.0pp 0.0 60.0↑60.0pp 12.0 36.0↑24.0pp 12.0 36.0↑24.0pp 8.0 46.0↑38.0pp
Change 30 10.0 96.7↑86.7pp 0.0 73.3↑73.3pp 0.0 66.7↑66.7pp 3.3 36.7↑33.4pp 6.7 26.7↑20.0pp 3.3 16.7↑13.3pp
Overall 400 21.8 68.5↑46.8pp25.5 56.5↑31.0pp20.2 60.2↑40.0pp33.5 58.5↑25.0pp33.0 53.8↑20.8pp36.2 57.0↑20.7pp
G MedGemma Full Per-Category Results
G.1 MedGemma 1.5 4B Full Per-Category Results
21

Category𝑛C7 C1 C4g
Negation 110 — 57.3 77.3
Conditional 20 — 15.0 35.0
Uncertainty 40 — 7.5 32.5
Family History 30 — 3.3 40.0
Sequence 40 — 2.5 37.5
Current State 50 — 32.0 58.0
Duration 30 — 46.7 63.3
Historical 50 — 0.0 46.0
Change 30 — 0.0 73.3
Overall 400—25.2 56.2
Table 8: MedGemma 27B ClinicalBench per-category accuracy (%, keyword evaluator v2) for three conditions.
Per-category changes are descriptive; several categories are small, and the change category has known label/evaluator
defects discussed in Section 4.6.
Category𝑛C1 C4g KW C4g Oracle
Negation 110 59.1 64.5 64.5
Conditional 20 5.0 25.0 25.0
Uncertainty 40 17.5 45.0 45.0
Family History 30 3.3 23.3 33.3
Sequence 40 40.0 75.0 75.0
Current State 50 36.0 40.0 58.0
Duration 30 53.3 86.7 86.7
Historical 50 12.0 36.0 36.0
Change 30 6.7 26.7 26.7
Overall 400 33.0 50.7 53.8
Table9: MedGemma1.54BClinicalBenchper-categoryaccuracy(%,keywordevaluatorv2)forthreeconditions:
C1(LLMalone),C4gkeyword-onlyclassification,andC4goracleclassification. Keyword →oracledelta:+3.0pp
overall, concentrated in current state ( +18pp) and family history ( +10pp); all other categories are identical between
oracleandkeyword. Per-categorychangesaredescriptive;severalcategoriesaresmall,andthechangecategoryhas
known label/evaluator defects discussed in Section 4.6.
H Experiencer Attribute Propagation Ablation
Theexperiencer attributedistinguishespatientconditionsfromfamilymemberconditions;withoutit,thegraph
conflates the two, causing family history misattribution.
The fix improved family history by +10.0pp with zero regressions on guard categories (negation, conditional,
duration,sequenceallunchanged),confirmingthattheexperiencerattributeisload-bearingforassertion-sensitive
reasoning.
22

Table10: ImpactofexperiencerattributepropagationonOpusC4(400questions,priorevaluatorrun). Categories
with no change omitted.
Category Pre-fix Post-fixΔ
Family history 63.3%73.3%+10.0pp
Uncertainty 50.0%55.0%+5.0pp
Historical 34.0% 38.0%+4.0pp
Current state 46.0% 48.0%+2.0pp
Overall C4 68.2% 70.2%+2.0pp
I SliceBench
SliceBench (6 patients, 144 questions, 3 complexity tiers; LLM-as-judge evaluation) is consistent with a complexity-
dependent KG effect: the incremental KG layer (B2 →B3) contributes+2.2ppoverall (CI: [-1.5 pp, +5.9pp]), not
reachingaggregatesignificance,whileTierC(15+encounters)gains +5.0ppvs.TierA(1–2notes) +0.6pp(Table14).
Because the overall B2 →B3 comparison is not statistically distinguishable from zero and tier-level results are
descriptive only (𝑛=2patients per tier), this pattern is treated as exploratory.
23

Tier A
(1–2 enc.)Tier B
(5–10 enc.)Tier C
(15+ enc.)−2−101234567B2 → B3 Accuracy Gain (pp)
+0.6 pp+1.0 pp+5.0 pp
Overall: +2.2 pp
(CI: [-1.5, +5.9])
n = 2 patients/tier, 24 Q/patient
Point estimates; per-tier CIs not available
Tier A (1–2 enc.) Tier B (5–10 enc.) Tier C (15+ enc.)020406080100Accuracy (%)B0 LLM Alone B1 Latest Note B2 All Notes B3 KG-RAG B4 Full SystemFigure 6: SliceBench exploratory results ( 𝑛=2patients per tier, 24 questions each). Tier-specific B2 →B3 deltas
are point estimates only and should not be over-interpreted; the overall B2 →B3 delta is+2.2pp(95% CI [-1.5 pp,
+5.9pp]).
24

J Per-Category Delta Chart and Transition Analysis
−40 −20 0 20
Accuracy Change (pp)Negation
Conditional
Uncertainty
Family History
Change
Duration
Current State
Historical
Sequence+24
+20
+15
+7
-7
-13
-24
-30
-48Task A
(assertion)
Task B
(temporal)C3 vs C4
0 20 40 60 80 100
Accuracy Change (pp)-9
-5
-2
+7
+90
+20
+40
+48
+55C4 vs C4g
0 20 40 60 80 100
Accuracy Change (pp)+36
+45
+38
+53
+93
+37
+44
+54
+55C1 vs C4g
Figure 7: Per-category deltas for three paired contrasts (C3 vs. C4, C4 vs. C4g, and C1 vs. C4g). These contrasts are
descriptive,notaformaladditivedecomposition,andcategorieswithsmall 𝑛orknownevaluator/reference-answer
defects are shown for transparency only.
Table11reportsquestion-leveltransitionsbetweenC3(KG-RAGwithoutassertions)andC4(withassertions,no
routing), with C4g recovery rates.
Table11: Per-categoryC3 →C4transitionanalysis(ClaudeOpus4.6, 𝑛=400).Regr.: C3correct →C4incorrect;
Impr.: C3 incorrect→C4 correct;Recov.: regressions recovered by C4g.
Category𝑛C3 C4 Regr. Impr. Recov.
Temporal categories (net−52):
Current state 50 54% 30% 20 8 17
Sequence 40 55% 10% 19 1 19
Historical 50 42% 12% 18 3 15
Duration 30 57% 47% 10 7 8
Change 30 20% 7% 6 2 5
Assertion-sensitive categories (net+37):
Negation 110 66% 89% 6 31 5
Family history 30 43% 50% 8 10 7
Uncertainty 40 38% 52% 7 13 6
Conditional 20 30% 50% 1 5 1
Total400 50% 46% 95 80 83
Overall, 87.4% of regressions (83/95) are recovered by C4g. Temporal categories account for 73/95 regressions
but only 21/80 improvements; assertion-sensitive categories show the reverse pattern (22/95 regressions, 59/80
improvements). This confirms that assertions help on epistemic questions but suppress needed evidence for temporal
synthesis without intent-matched routing.
25

Qualitative examples.C4 regressions exhibit a consistent pattern: the evidence context includes raw assertion
summaries that distract the LLM from the question’s intent. For family-history questions, C4 states “The patient
has Lymphoma” (leaking family-history findings to patient state); for sequence questions, C4 lists assertion types
insteadoftemporalordering;forhistoricalquestions,C4notes“historyofdepression”andconcludesitisnolonger
active. Ineachcase,C4g’sintent-awareretrievaleliminatesthedistractionbyroutingtocategory-specificevidence
(cross-admission comparison, timeline traversal, or current-state filtering).
K SliceBench Conditions
SliceBench selects patients in three tiers:Tier A(2 patients, 1–2 encounters),Tier B(2 patients, 5–10), andTier C
(2 patients, 15+). Each patient receives 24 questions spanning hard cross-admission categories (cross-encounter
medication timelines, problem list reconciliation, causal chain tracing), yielding 144 total.
Table 12: SliceBench conditions. B0–B4 form a monotone context progression.
ID Condition Context Provided to LLM
B0 LLM Alone Question only (parametric knowledge)
B1 Latest Note Most recent clinical note
B2 All Notes RAG All notes, retrieved by relevance
B3 KG-RAG Knowledge graph context + all notes
B4 Full System KG-RAG + guidelines + calculators
L System Implementation Details
LLM baseline.On MedQA-USMLE (965 questions), Claude Opus 4.5 achieves 81.6% accuracy (5.1pp below
GPT-4at86.7%andMed-PaLM2at86.5%[ 23]),establishingthattheClaudemodelfamily—Sonnet4.5(SliceBench
answerer) and Opus 4.6 (ClinicalBench primary, SliceBench judge)—provides a reasonable LLM baseline (Table 13).
Model Accuracy Step 1 N
EpiKG (LLM alone) 81.6% 79.9% 965
GPT-4 (2023) 86.7% — 1,273
Med-PaLM 2 (2023) 86.5% — 1,273
Claude 3 Opus (2024) 78.0% — 1,273
Table 13: MedQA-USMLE results. The LLM-alone baseline is competitive.
KGscaleandtraversallatency.Thedeployedsystemprocesses145documentsacross85patients,materializing
3,100KGnodes and8,803edges. Graphtraversallatency issub-millisecondatthisscale: 0.57ms(1-hop), 0.75ms
(2-hop). The full system integrates 201 clinical calculators, 1,202 guideline sections, 20M+ OMOP vocabulary
relationships, 122 assertion trigger patterns, and 24 edge types across 13 node types.
Multi-hop traversal.On the DR.KNOWS diagnostic reasoning benchmark [ 44], which measures KG traversal
accuracy using PostgreSQL-backed clinical data, EpiKG achieves 0.420 overall (50% at 1-hop, 25% at 2-hop, 0% at
3-hop). Multi-hop degradation reflects a deliberate architectural trade-off: PostgreSQL CTE-based traversal provides
ACID compliance but scales poorly beyond 2 hops.
M SliceBench Tier-Stratified Results
26

Table 14: SliceBench results stratified by patient complexity tier. B2 →B3Δshows the incremental KG contribution.
Score (%)
Tier Encounters B0 B1 B2 B3 B4 B2→B3Δ
A 1–2 notes 51.7 66.7 81.1 81.8 82.6+0.6pp
B 5–10 notes 49.5 79.4 86.4 87.4 90.1+1.0pp
C 15+ notes 48.4 74.2 86.5 91.5 90.3+5.0pp
N ClinicalBench Full Per-Category Results (Opus)
Category𝑛C7 C6 C1 C2 C2b C3 C4 C4g
Negation 110 99.1†85.5 44.5 71.8 76.4 66.4 89.1 81.8
Conditional 20 0.0 45.0 0.0 35.0 50.0 30.0 50.0 45.0
Uncertainty 40 0.0 35.0 12.5 35.0 32.5 37.5 52.5 50.0
Family History 30 0.0 56.7 3.3 43.3 43.3 43.3 50.0 56.7
Sequence 40 0.0 82.5 7.5 60.0 45.0 55.0 10.0 62.5
Current State 50 0.0 18.0 26.0 32.0 32.0 54.0 30.0 70.0
Duration 30 0.0 43.3 33.3 66.7 63.3 56.7 46.7 63.3
Historical 50 0.0 62.0 6.0 48.0 46.0 42.0 12.0 60.0
Change 30 0.0 56.7 10.0 36.7 23.3 20.0 6.7 96.7
Overall 400 27.2†59.2 21.8 52.0 50.8 50.0 46.2 68.5
Table 15: Complete ClinicalBench per-category accuracy (%, Claude Opus 4.6, keyword evaluator v2) for all
conditions, computed over the full 400-item set. The change-excluded keyword endpoint (sensitivity comparator;
theprimaryendpointistheleave-author-outMcNemarinSection4.7)excludesthe30change-categoryitemsand
8 experiencer-attribution-defective items (Appendix P.2), yielding 𝑛=362; per-category counts above are for the
fullset. C2b(Contrieverdenseretrieval)scorescomparablytoC2(TF-IDF),confirmingtheretrievalmethoddoes
not drive the C2→C4g gap. C4 adds assertion metadata to C3 without intent routing; it helps assertion-sensitive
categories (negation: +22.7pp) but hurts temporal categories (historical: −30pp, sequence:−45pp). Only with
intentrouting(C4g)doesthefullsystemreach68.5%.†C7isreportedas“—(evaluatorartifact,semantic0%)”in
Table 2; raw 27.2% reflects template refusals coincidentally matching negation keywords (see Section 4.1).
N.1 Extension Condition Per-Category Results
All four models gain on the extension endpoint, with duration showing the largest gain across architectures (Opus
+10pp, Qwen3.5+36.7pp, GPT-OSS+43.4pp, MedGemma+40pp), indicating cross-admission temporal recall
benefits substantially from structured graph retrieval. Opus and Qwen3.5 also gain on historical (Opus +22pp,
Qwen3.5+20pp); GPT-OSS shows a slight decline on historical ( −8pp) and current state ( −12pp), consistent
with smaller-model difficulty stably extracting cross-admission state. The reference-answer defects in historical
(Section 4.6) attenuate measured gains on this category across all four models.
27

Table16: Per-categoryaccuracy(%)forextensionconditions( 𝑛=240,frozenevaluator+canonicalquestions_v2
gold). C1b: discharge summary only. C4g+full: intent-aware KG-RAG with all clinical notes. The extension subset
spans 4 categories: negation (𝑛=110), current state (𝑛=50), historical (𝑛=50), duration (𝑛=30).
Opus Qwen3.5 GPT-OSS MedGemma
Category C1b C4g+ C1b C4g+ C1b C4g+ C1b C4g+
Negation 82.7 92.7 82.7 86.4 84.5 85.5 80.9 85.5
Current st. 14.0 24.0 38.0 38.0 56.0 44.0 20.0 28.0
Historical 52.0 74.0 36.0 56.0 22.0 14.0 44.0 44.0
Duration 46.7 56.7 40.0 76.7 43.3 86.7 36.7 76.7
Overall 57.5 70.0 58.3 68.8 60.4 62.1 55.0 63.8
O Physician Adjudication Protocol and Summary
Design.Asingleboard-certifiedemergencyphysician(A.S.)independentlyscoredClinicalBenchanswersundertwo
conditions(C1and C4g)usingafive-dimensional rubric(reference-answercorrectness, model-answercorrectness,
auto-scorefairness,clinicalsafety,clinicalutility). Thereviewerwasblindedtoconditionassignment(conditions
randomized asA/B), so thisshould beinterpretedas a blinded internal expertaudit rather thanindependent external
validation. Thepairedadjudicationcovers120uniquequestions ×2conditions. Item-levelreference-answerandsafety
summariesusethefulladjudicationrecordset,whichincludesonerepeatedblinded-conditionreviewandtherefore
sumsto241records. Free-textphysiciannoteswererecordedfor165/241items(68.5%). Reviewswereconducted
using a custom scoring interface with access to full de-identified MIMIC-IV discharge summaries. Condition
assignmentwasrandomizedandapproximatelybalancedacrossblindedlabels;thisverifiesallocationbalance,not
successful deblinding prevention, because C4g outputs can be more structured than C1 outputs.
Endpoints.(1) Human–keyword evaluator concordance rate (fraction of automated scores confirmed by physician).
(2) Physician-rated C4g vs. C1 accuracy, safety, and utility. (3) Reference-answer error rate and defect taxonomy.
Reference-answererroranalysis.Of the241auditedrecords,only44%ofv2referenceanswerswereratedfully
correct;29.5%wereoutrightwrongand19.5%neededrevision. Errorsareconcentratedinchange(0%correct—NLP
conflated inpatient orders with discharge medications), historical (56.7% wrong—“history of X” misclassified as
resolved),anduncertainty(37.5%wrong—causaluncertaintyconflatedwithexistentialuncertainty). Becausethe
detecteddefectsarequestion-levelratherthancondition-specific,theyarelesslikelytoreversethedirectionofthe
C1–C4gcomparison. Theydo,however,materiallyaffectabsoluteaccuraciesand somecategory-levelmagnitudes,
especially for change and historical questions.
Auto-evaluator concordance.The keyword evaluator agreed with physician judgment in 54.2% of cases ( 𝑛=240).
When it disagreed, it was overwhelmingly too strict: 97 false negatives (40.4%) vs. 13 false positives (5.4%), a 7.5:1
strict-to-lenient ratio. The majority of evaluator errors (63% of false negatives) trace to reference-answer errors rather
than evaluator logic.
Disclosure and limitations.The reviewing physician (A.S.) is the author and system designer (see Section 3.8 for
primarydisclosure). Single-reviewerdesignlimitsinter-raterreliabilityassessment;three-raterexternalvalidation
results (including two independent external physicians) are reported in Section 4.7.
Status.This protocol was designed before full result synthesis. Full adjudication results ( 𝑛=120paired) are
reported in Section 4.6; complete per-category results in Appendix X.
28

Table17: Reference-answerqualitybycategory(physicianadjudicationofv2referenceset). Defectrate=fraction
of reference answers rated less than fully correct. Categories with the highest defect rates drive most evaluator
disagreements.
Category𝑛Correct Defect Rate
Change 31 0.0% 100.0%
Historical 30 36.7% 63.3%
Uncertainty 40 40.0% 60.0%
Current state 30 43.3% 56.7%
Negation 10 50.0% 50.0%
Sequence 40 50.0% 50.0%
Conditional 20 55.0% 45.0%
Duration 20 65.0% 35.0%
Family history 20 70.0% 30.0%
Overall 241 44.0% 56.0%
P Reproducibility Details
Models.ClinicalBench primary ablation: Claude Opus 4.6 (claude-opus-4-6, keyword evaluator v2). Cross-model:
MedGemma 27B (alibayram/medgemma:27b, 4-bit GGUF), GPT-OSS 20B (4-bit GGUF), Qwen3.5 35B, Gemma 4
31B( gemma4:31b ,4-bitGGUF, num_predict=2048 ),andMedGemma1.54B( alibayram/medgemma15:4b ,4-bit
GGUF, num_predict=2048 , with <unused94> /<unused95> stop tokens) via Ollama; same evaluator. SliceBench:
Claude Sonnet 4.5 (claude-sonnet-4-5-20250929) answers, Opus 4.6 judges. Temperature 0 throughout; 4-bit
quantization introduces GPU non-determinism ( ±10pprun-to-run for MedGemma), controlled via within-run paired
comparisons.
Evaluation.The deterministic keyword evaluator performs exact word-boundary matching (regex \bKEYWORD\b )
against reference-answer assertion and temporal keywords. Per-category keyword sets: negation (“no,” “denies,”
“absent,” “negative”), uncertainty (“possible,” “suspected,” “may,” “likely,” “consider”), temporal (“before,” “after,”
“during,”“changed,” “new”),pluscategory-specificterms. Evidence preambles(echoedgraphcontext)are stripped
before matching.
Retrieval.C2’s vanilla RAG uses TF-IDF over chunked clinical notes (512-token chunks, 64-token overlap),
retrieving the top-5 chunks by cosine similarity. C2b replaces TF-IDF with Contriever [ 45] dense embeddings
(256-tokenchunks,64-tokenoverlap,top- 𝑘upto6,000characters);C2bscores50.8%( Δ=−1.2pp vs.C2;𝑝=0.62),
confirming the retrieval method does not explain the C2 →C4g gap. These are the closest analogs to existing medical
RAG systems (DoctorRAG, MedRAG) on patient-level cross-admission data. ClinicalBench conditions map to
SliceBench: C1≈B0, C2≈B2, C4g≈B3, C4g+full≈B4.
Bootstrap. 𝑛=2,000 resamples, seed 42, BCa method [ 25]. Patient-level cluster bootstrap (resampling the 43
patients with replacement, including all their questions per draw) is the inferential method for keyword-endpoint
CIs. Question-levelresamplingisreportedasasecondarysensitivityanalysis. (Caveat: with 𝑛=43clustersthisis
at the low end of cluster-bootstrap reliability; cf. Cameron & Miller [ 26]; the substantive primary endpoint is the
leave-author-out paired exact McNemar in Section 4.7, which does not depend on cluster bootstrap.) Patient-level CIs
are slightly wider but all reported endpoints remain significant (Table 18).
McNemar’stest.BootstrapCIsaresupplementedwithCIswithMcNemar’stestforpairednominaldata,comparing
discordantpairs(C1wrong/C4grightvs.C1right/C4gwrong). Allthreepairwisecomparisonsaresignificant: C1
vs. C4g (𝜒2=155.8,𝑝 <10−6; discordant: 18 vs. 204), C1 vs. C3 ( 𝜒2=73.8,𝑝 <10−6; 30 vs. 143), and C3 vs.
C4g (𝜒2=47.2,𝑝<10−10; 20 vs. 93). On the hard cross-admission subset (secondary endpoint, 𝑛=122: change∪
29

current_state∪historical, after excluding the 8 experiencer-attribution-defective items in Appendix P.2), C4g oraclevs.
C1 yieldsΔ=+57.4pp,𝑝<10−15.
BH-FDR adjustment for cross-model contrasts.Across the six cross-model C4g oraclevs. C1 McNemar tests
reportedinTable18andSection4.4,Benjamini–Hochbergfalse-discovery-rateadjustmentyieldsq-values: Opus
𝑞=1.84×10−32,GPT-OSS𝑞=3.14×10−27,MedGemma27B 𝑞=1.22×10−17,Gemma431B 𝑞=3.81×10−17,
Qwen3.5 35B 𝑞=2.00×10−11, MedGemma 1.5 4B 𝑞=9.32×10−11. BH-FDR applied across the six cross-model
contrasts;all 𝑞<10−10. BH-FDRisreportedhere. ThemoreconservativeBenjamini–Yekutieli(2001)procedure
(which holds under arbitrary dependence, including PRDS violations of paired McNemars on overlapping items)
yields equivalent conclusions: BY adjustment factor 2.45×over BH on𝑚=6cross-model contrasts, with all BY
q-values<2.5×10−10.
Table 18: Statistical summary: patient-level (cluster) and question-level BCa bootstrap 95% CIs for the change-
excluded keyword sensitivity endpoint and other secondary contrasts. The author-blind primary endpoint is the
leave-author-out paired exact McNemar in Section 4.7; the keyword endpoint is change-excluded ( 𝑛=362, after
excluding the 8 experiencer-attribution-defective items in Appendix P.2) keyword C4g vs. C1.
ComparisonΔQuestion CI Patient CI McNemar𝑝
C4g−C1 (full-set oracle, secondary)+46.5pp [+40.5pp, +51.7pp] [+40.8pp, +52.4pp]<10−6
C2b−C2 (dense vs. keyword)−1.2pp [-6.2pp, +3.5pp] [-5.6pp, +3.5pp]0.62
C4g−C2b (KG-RAG vs. dense)+17.5pp [+11.8pp, +22.8pp] [+12.4pp, +22.7pp]<10−6
C3−C1 (retrieval)+28.2pp [+22.2pp, +34.2pp] [+23.1pp, +33.3pp]<10−6
C4−C3 (assertions)−3.8pp [-10.8pp, +1.8pp] [-10.7pp, +3.1pp]0.26
C4g−C4 (routing)+22.0pp [+15.5pp, +27.8pp] [+14.8pp, +29.4pp]<10−10
C4g−C3 (both)+18.2pp [+13.2pp, +22.8pp] [+13.7pp, +23.1pp]<10−10
Hard cross-admission (𝑛=122)+57.4pp [+47.4pp, +65.9pp] [+46.8pp, +68.6pp]<10−15
Data.De-identified MIMIC-IV clinical notes accessed under a PhysioNet Credentialed Health Data Use Agreement
(v3.1).
PhysioNet DUA compliance.The HuggingFace release of the ClinicalBench artifact (DOI 10.57967/hf/8549)
contains: (a) question text (authored from MIMIC-IV charts but rephrased; not raw note text), (b) reference answers
(paraphrased clinical findings; not direct note quotations), and (c) raw model predictions (commit da5f5b1 stripped
rawnoteexcerptsfromadjudicationitemspriortopublicrelease). MIMIC-IVnotetextremainsexclusivelyunder
PhysioNet credentialed access. Any user replicating the system requires separate PhysioNet authorization.
Released artifacts.ClinicalBench questions, reference answers (v1 and v2), raw model predictions for all evaluated
conditions, the deterministic keyword evaluator, and physician adjudication data are publicly released (Hugging Face
DOI 10.57967/hf/8549). The full EpiKG application stack (graph construction, intent-aware routing implementation,
retrieval algorithm) is NOT released in this work. Application-code release is planned for follow-on work; reviewers
anddownstreamusersshouldtreatthesystemcontributionhereasaproberatherthanareproduciblearchitectural
artifact.
Compute.ClinicalBenchOpusC1/C3/C4g: ∼22mineach(API);OpusC6: ∼3.5h(API);C7: <1min(noLLM);
MedGemma 27B C1–C4g: ∼3.6h (single GPU); MedGemma 27B C6: ∼1.5h; Gemma 4 31B C1/C4g: ∼10h (Apple
Silicon, Ollama); MedGemma 1.5 4B C1/C4g:∼1.5h (Apple Silicon, Ollama); SliceBench:∼2h (API).
30

Evaluatorevolution.Thekeywordevaluatorunderwentthreeversions: v0(substringmatching),v1(word-boundary
matching+evidencepreamblestripping),andv2(+abstentiondetectiongate+domain-specifickeywordrequirements
for sequence and change). The v1 evaluator awarded false positives when models responded with “insufficient
information in the notes” because negation and temporal keywords in the refusal text matched reference-answer
patterns. Thev2evaluatoraddsanabstentiondetectionlayer: answersmatchingabstentionpatterns(e.g.,“cannot
determine,” “not mentioned,” “insufficient evidence”) are scored as incorrect unless they contain clinical claim
patterns (e.g., “patient does not,” “denies”). Additionally, v2 requires sequence answers to contain ordering keywords
(“first,” “then,” “before”) and change answers to contain change keywords (“added,” “removed,” “discontinued”),
preventing term-overlap-only false positives. The duration category’s minimum-0.5 score floor for matching duration
keywordswasremoved. Thev1 →v2transitionreducedC1accuracy(from ∼50%to21.8%forOpus)bycorrectly
classifyingabstention responsesasincorrect;C4g accuracydecreasedmodestly(from ∼76%to68.5%)because the
model abstains less frequently when retrieval context is provided. All main-text numbers use v2.
Reproducibility package.A standalone reproducibility package is included in the supplementary
materials ( epikg-benchmark/ ) and available at https://huggingface.co/datasets/alexstinard/
epikg-clinicalbench . It contains all 400 ClinicalBench questions with reference answers, raw model pre-
dictionsforallLLM-basedconditions(OpusC1/C2/C3/C4/C4g/C6,MedGemma27BC1/C4g,GPT-OSSC1/C4g,
QwenC1/C4g,Gemma431BC1/C4g,MedGemma1.54BC1/C4g),scoredoutputsforthedeterministicbaseline
(C7), and the keyword evaluator v2 with abstention detection. The evaluator itself requires only Python 3.10+ with no
external dependencies; reproduce.py additionally requires NumPy and SciPy for bootstrap CIs. All ClinicalBench
accuracynumbersinTables2,7,and3arereproduciblefromthispackage. MIMIC-IVpatientidentifiersareincluded
sothatPhysioNet-credentialedreviewerscantracepredictionsbacktosourceclinicalnotes. ClinicalBenchispub-
liclyavailableat https://huggingface.co/datasets/alexstinard/epikg-clinicalbench withCroissant
metadata for machine-readable dataset discovery.
Results provenance.Table 19 maps each reported result to its source checkpoint file. MedGemma 27B C4g
has 2 empty predictions (timeouts); these are scored as incorrect ( 𝑛=400). All cross-model C1 and C4g results
(Table 3) are from the same system snapshot (February 2026), except Gemma 4 31B (April 2026, added after
the initial cross-model batch) and MedGemma 1.5 4B (April 2026, after fixing a checkpoint truncation bug and
adding <unused94> /<unused95> Gemma 3 stop tokens in the Ollama Modelfile); intermediate ablation conditions
(C2/C3/C4/C6) for non-Opus models were collected in a later batch (March 2026) after system updates and are
included in the reproducibility package for completeness but are not used in any paper table.
P.1 Infrastructure Bugs Discovered and Fixed During Evaluation
Duringfinalreview,threeinfrastructureissueswerediscoveredandcorrectedthataffecteddataintegrityinearlier
runs. They aredocumentedhereinthe interestofmethodologicaltransparency; allthreehavesince beenfixed,and
the residual impact on reported numbers is described for each.
Bug 1: Checkpoint serialization truncation (500-character limit).A line in qa_experiment_executor.py
wrote predicted_answer[:500] to checkpoint files, silently truncating every saved answer to 500 characters.
Original in-run scoring (the legacy internal evaluator) operated on full answers, but the frozen-evaluator rescoring—
whichisthesourceofallnumbersinthispaper—operatedonthetruncatedstrings. Thetruncationratescaledwith
per-model answer verbosity: GPT-OSS ≈0.2%of answers affected, MedGemma 27B ≈0.5–2.8%(all negligible,
<1ppimpact on frozen rescore), Qwen 3.5 ≈0.8–4.2%(<1pp), Opus 4.6≈3.5–16%(estimated 1–3ppdownward
bias on C4g oracle), and MedGemma 1.5 4B ≈7.5–35.5%(estimated 2–3ppdownward bias). Because the bug was
applied uniformly across models and conditions, relative rankings are preserved; absolute scores for the most verbose
models (Opus, MedGemma 1.5) would shift slightly upward if fully rerun. The slice was removed from the [:500]
sliceandreranMedGemma1.5end-to-end(themostaffectedmodel)toobtaincleandata;othermodelsretaintheir
original checkpoint data with this documented downward bias.
31

Table 19: Results provenance: checkpoint files for each condition ×model combination. All scored with keyword
evaluator v2.
Condition Model Checkpoint file𝑛
C1 Opus 4.6opus/C1_llm_alone.jsonl400
C2 Opus 4.6opus/C2_vanilla_rag.jsonl400
C2b Opus 4.6opus/C2b_dense_rag.jsonl400
C3 Opus 4.6opus/C3_kg_rag.jsonl400
C4 Opus 4.6opus/C4_epistemic_kg_rag.jsonl400
C4g Opus 4.6opus/C4g_intent_aware.jsonl400
C6 Opus 4.6opus/C6_long_context.jsonl400
C7 —opus/C7_deterministic.jsonl400
C1 MedGemma 27Bmedgemma/C1_llm_alone.jsonl400
C4g MedGemma 27Bmedgemma/C4g_intent_aware.jsonl400*
C1 GPT-OSS 20Bgptoss/C1_llm_alone.jsonl400
C4g GPT-OSS 20Bgptoss/C4g_intent_aware.jsonl400
C1 Qwen3.5 35Bqwen35/C1_llm_alone.jsonl400
C4g Qwen3.5 35Bqwen35/C4g_intent_aware.jsonl400
C1 Gemma 4 31Bgemma4/C1_llm_alone.jsonl400
C4g Gemma 4 31Bgemma4/C4g_intent_aware.jsonl400
C1 MedGemma 1.5 4Bmedgemma15/C1_llm_alone.jsonl400
C4g MedGemma 1.5 4Bmedgemma15/C4g_intent_aware.jsonl400
*MedGemma 27B C4g: 2 timeouts; empty answers scored as incorrect (𝑛=400).
Bug 2: MedGemma 1.5 special-token duplication.MedGemma 1.5 4B (Gemma 3 architecture) emitted the
special tokens <unused94> and<unused95> mid-generation, causing the model to regenerate the same answer
twice within a single response. The frozen evaluator then scored on the concatenated duplicate text, effectively
givingthe modeltwoindependent chancestomatch referencekeywords. Manualinspectionof per-questionoutputs
showed that 26%of MedGemma 1.5 C4g oracle answers contained these tokens, artificially inflating frozen-evaluator
scoresbyapproximately 2–5pp. OnlyMedGemma1.5wasaffected; thetokensareGemma3specific. Thefixwas
trivial—adding <unused94> and<unused95> totheOllamaModelfilestop-tokenlist—butthebenchmarkingimpact
was significant. All MedGemma 1.5 numbers in this paper were collected after the fix was applied and the model was
rerun across all three reported conditions.
Bug3: OllamasilentlydiscardsQwen3.5repetitionpenaltiesinthinkingmode.Qwen3.535Bisdesigned
as an extended-reasoning model; its model card explicitly recommends presence_penalty=1.5 to prevent
pathologicalrepetitionloopsduringchain-of-thoughtgeneration. However,Ollamasilentlydiscards repeat_penalty ,
presence_penalty , and frequency_penalty options when Qwen is used with think: true (Ollama issue
#14493). At temperature=0 themodelthenentersinfinitereasoningloops(e.g.,“Wait,I’llwrite: ‘X’.Okay,I’ll
write: ‘Y’.Wait,I’llwrite: ‘X’...”) andconsumestheentiretokenbudgetwithoutproducinganyvisiblecontent. The
issuewasverifiedbytestingeightsamplingconfigurationsthroughtheOllamaAPI( repeat_penalty from 1.3to
1.8,frequency_penalty=0.5 ,presence_penalty=0.5 ,temperature∈{0.3,0.5} ,mirostat=2 , and stacked
combinations); all eight produced identical 29,083-character thinking output with empty content, confirming that
OllamaisnotforwardingtheseoptionstoQwen’sthinkinggeneration. RelatedopenOllamaissuesinclude#14493
(Qwen 3.5 tool calling non-functional and repetition penalties silently ignored), #14421 ( qwen3.5:35b looping),
#10976(thinking+tools+qwen3 ⇒emptyoutput),#14716( qwen3.5 visionoutputroutedtothinkingfield),and
#10927 (LLM stuck in infinite loop of thinking).
As a workaround thinking was disabled ( think: false ) for all Qwen runs reported in this paper. This produces
direct answers but may not reflect Qwen’s optimal reasoning performance. Qwen’s reported numbers should therefore
32

be interpreted as “Qwen 3.5 35B without reasoning enabled” rather than “Qwen 3.5 35B at full capability.” This
workaround has a visible downstream effect on Qwen’s oracle-vs-keyword comparison: a −1.2pporacle inversion at
the aggregate level that does not match the pattern of the other five tested models. Per-category analysis shows Qwen
benefitssubstantiallyfromoracleroutingonhistoricalquestions( +16pp)butsufferson current_state (−10pp)
andconditional (−15pp)categories. Onehypothesis isthat Qwen’s constrainednon-thinking modeinteractspoorly
with oracle’s focused current_state retrieval, which discards historical context that non-thinking Qwen appears to
relyon. Whetherthisinversionwouldpersistwiththinkingenabled—viavLLMorQwen’snativeDashScopeAPI,
which do honor repetition penalties—is an open question is left for future work.
P.2 Experiencer-Attribution Defective Items
Duringpost-hocverification,8itemswereidentifiedwhosesource section="Family History" butwhosegold
expected_answer asserts the disease as a current or historical condition of the patient. The defect arises from
the upstream NLP pipeline mis-propagating an experiencer flag during gold-answer generation: the section tag
was correctly retained but the answer string nonetheless described the patient. These items are excluded from the
change-excluded keyword endpoint ( 𝑛=362, sensitivity comparator). The headline impact of removing them is
+40.0pp→+39.5pp on that endpoint. They remain in the released v2 gold standard for transparency.
Table 20: Eight experiencer-attribution-defective items excluded from the change-excluded keyword sensitivity
endpoint. All havesection="Family History"but gold answers describing the patient.
QID Category Disease Bug
bench_b_current_state_0a964177current_state Lung cancer gold says current; FH
bench_b_current_state_351fb38ecurrent_state Lung cancer gold says current; FH
bench_b_current_state_7080eb03current_state Hypertension gold says current; FH
bench_b_current_state_96ef4bd2current_state Breast cancer gold says current; FH
bench_b_current_state_9a0fed0bcurrent_state Colon cancer gold says active; FH
bench_b_current_state_ab1c0783current_state Colon cancer gold says current; FH
bench_b_current_state_e84d9e91current_state Breast cancer gold says current; FH
bench_b_historical_a701eaf4historical Colon cancer gold says historical; FH
P.3 Evaluator Polarity: Extended Disclosure
The keyword evaluator uses category-specific keyword lists and matches by word-boundary regex. For three
categories—uncertainty, family_history, and conditional—the rule is structurally:
is_correct = _has_match(predicted_lower, patterns)
where patterns isthecategory-definingkeywordlist(e.g., [’if’,’conditional’,’pending’,’depending’,’only
if’]for conditional). The gold expected_answer is not consulted: any prediction containing a category keyword
is scored correct.
For current_state and historical, the rule matches keyword presence ( "current" ,"active" ,"present" for
current_state; "was","former" ,"resolved" ,"history" for historical) without polarity check. Consequently,
a prediction asserting“NOT FOUND IN CURRENT RECORDS”matches the keyword "current" and is scored
correct against a gold of “currently active”—directionally opposed but lexically overlapping.
Theimplicationis thatC4g’sstructured-answerstyle(which routinelyechoesthequeriedcategory, e.g.,“Current
state: ...”) ismechanicallyadvantagedoverC1’sabstentionstyle(“insufficientinformation”),evenwhenneither
answer carries the right clinical content. This contributes to the keyword evaluator’s measured 7.5:1 strict-vs-lenient
asymmetry(Table28)andistheprincipalreasonwetreatthekeywordevaluatorasadeterministicreproducibility
proxy rather than a substantive truth criterion. Physician adjudication and LLM-as-judge are the substantively
interpretable evaluators; their deltas (Section 4.6) are the comparisons we ask readers to weight.
33

Q Assertion Category Definitions
Assertion Definition Example
presentCondition currently affirmed “has diabetes”
absentCondition explicitly negated “denies chest pain”
possibleCondition suspected, not confirmed “possible pneumonia”
conditionalContingent on specific circumstances “if febrile, start antibiotics”
hypotheticalDiscussed as a scenario “would need dialysis if...”
family_historyAttributed to a family member “mother had breast cancer”
historicalPreviously true, not necessarily current “former smoker”
Table 21: The 7-value assertion taxonomy, extending i2b2 by separatinghistoricalfromfamily_history.
R Temporal Relation Mapping
The nine temporal relations Rused on KG edges are derived from Allen’s 13 canonical interval relations [ 36] by
merging symmetric pairs.
Rvalue Allen source(s) Meaning
BeforeBefore, Meets𝐴ends before𝐵starts
AfterAfter, Met-by𝐴starts after𝐵ends
DuringDuring𝐴entirely within𝐵
ContainsContains𝐴entirely contains𝐵
OverlapsOverlaps, Overlapped-by𝐴and𝐵partially overlap
StartsStarts, Started-by𝐴and𝐵share start time
FinishesFinishes, Finished-by𝐴and𝐵share end time
ConcurrentEquals𝐴and𝐵have same interval
Unknown— Relation undetermined
Table 22: Mapping from Allen’s 13 interval relations to the 9 temporal relation values stored on KG edges.
S Intent-Aware Routing Algorithm
TheC4gintentclassifieroperatesintwomodes. Intheoraclemodeusedforbenchmarkevaluation,thequestion’s
category metadata determines the intent directly. In thekeyword-onlymode used for deployment, a rule-based
classifier infers intent from keyword patterns in the question text (e.g., “changed”, “new since” triggerChange;
“currently”,“activeproblem”triggerCurrent_State). Thekeywordclassifierachieves68%overallaccuracybutonly
20.0% on questions requiring targeted routing; per-category accuracy is reported in Table 23. All benchmark results
in the main text use oracle classification unless otherwise noted. Algorithm 1 details the full retrieval procedure.
S.1 Keyword-Only Intent Classifier Accuracy
Table23reportsper-categoryaccuracyofthekeyword-onlyintentclassifieralongsidetheoracle–keywordaccuracy
gaponOpusC4g. Thekeywordclassifierachieves68%overallclassificationaccuracybutonly20.0%onquestions
requiring targeted routing (historical: 0%, family history: 0%, current state: 34%, change: 50%). Categories routed
toDefault(negation,uncertainty,conditional,duration,sequence)areunaffectedbyclassificationerrorsbecause
both oracle and keyword paths use the same default BFS traversal.
34

Algorithm 1:Intent-Aware Retrieval (C4g)
Input:Question𝑞, patient𝜋
Output:Structured evidence𝐸
1C←ExtractConcepts(𝑞);// NLP + OMOP enrichment
2𝜄←ClassifyIntent(𝑞);//∈{Change,CurrSt,Hist,Default}
3if𝜄=Changethen
4Partition edges by admission:E 𝑘←{𝑒|hadm_id(𝑒)=𝑘};
5foreachadmission pair(𝑘,𝑘′)with𝑘 <𝑘′do
6A←C 𝑘′\C𝑘;R←C 𝑘\C𝑘′;S←C 𝑘∩C𝑘′;
7𝐸 𝑔←FormatChange(A,R,S);
8else if𝜄=CurrStthen
9𝐸 𝑔←FilterEdges(𝜋,C, 𝜏 𝑎=Current∨open validity);
10Deduplicate by concept; emit “Not Found” for missing𝑐∈C;
11else if𝜄=Histthen
12𝐸 𝑔←FilterEdges(𝜋,C, 𝜏 𝑎=Past);
13Augment: concepts in earlier admissions but absent from latest→“resolved”;
14else
15𝐸 𝑔←BidirectionalBFS(𝜋,C,hops=2–3, 𝑐 min=0.3);
16𝐸 𝑑←RetrieveDocuments(𝜋,C);
17returnCompose(𝐸 𝑔,𝐸𝑑,template(𝜄));
Table 23: Per-category keyword intent classifier accuracy and its impact on Opus C4g QA accuracy ( 𝑛=400).
“Classifieracc.” isthefractionofquestionswherethekeywordclassifiermatchesoracleintent. Categoriesmarked
“Default” are routed identically under both classifiers.
Category𝑛Classifier Acc. Oracle KeywordΔ
change 30 50.0% 96.7% 30.0%−66.7pp
current_state 50 34.0% 70.0% 66.0%−4.0pp
family_history 30 0.0% 56.7% 46.7%−10.0pp
historical 50 0.0% 60.0% 60.0%0.0pp
negation 110 Default 81.8% 75.5%−6.3pp
sequence 40 Default 62.5% 55.0%−7.5pp
uncertainty 40 Default 50.0% 50.0%0.0pp
conditional 20 Default 45.0% 45.0%0.0pp
duration 30 Default 63.3% 70.0%+6.7pp
Overall 40068.0%68.5% 60.2%−8.3pp
35

0 20 40 60 80 100
Keyword Classifier Accuracy (%)Negation
Conditional
Uncertainty
Sequence
Duration
Change
Current State
Family History
Historical100% n=110
100% n=20
100% n=40
100% n=40
100% n=30
50% n=30
34% n=50
0% n=30
0% n=50(A)  Classifier Accuracy by Category
No routing needed
Keywords work
Keywords fail
30 40 50 60 70 80 90 100
C4g (oracle) Accuracy (%)20304050607080C4g (keyword) Accuracy (%)
parityNegation  (-7pp)
Duration  (+10pp)
Current State
HistoricalSequence  (-8pp)
Uncertainty
Family History  (-10pp)
Conditional
Change  (-70pp)
n=20 n=50 n=110(B)  Downstream QA Impact (Opus)
Generic (no routing)
Targeted (keywords work)
Targeted (keywords fail)Figure 8: Keyword intent classifier analysis (Opus).(A)Per-category classifier accuracy: categories requiring no
targeted routing (top, green) achieve 100%; categories needing routing show 0–50% keyword accuracy (bottom,
red/orange).(B)Downstream QA impact: categories near the parity line(dashed) havesmall oracle–keyword gaps;
change (−70pp) is the major outlier, while duration (+10pp) benefits from keyword routing.
36

T Bookend and Diagnostic Conditions
C5 (Full System).C5 extends C4 with guideline retrieval (1,202 sections) and clinical calculators (201, e.g.,
CHA 2DS2-VASc,MELD).Inapriorevaluatorrun,C5scoredbelowC4g,likelybecausenon-relevantcomponents
(guidelinesandcalculators)dilutethecontextwindowwithoutcontributingtoClinicalBenchquestiontypes. C5is
excludedfromthecurrentablationladderbecauseitconflatesretrievalarchitecturewithadditionalknowledgesources.
C6 (Long Context).All patient documents are concatenated chronologically and presented to the LLM. For Opus
(200K token window), all notes fit; for MedGemma (8K window), later documents are truncated. C6 achieves
59.2% (Opus) overall—well above C1 (21.8%) and modestly below C4g oracle(68.5%,−9.3pp,𝑝=0.001 ). The gap
concentrates in current state (C6 18.0% vs. C4g 70.0%, −52pp); on temporal categories C6 is comparable (historical
62.0%, sequence 82.5%) or weaker only modestly (duration 43.3% vs. C4g 63.3%). This indicates brute-force
long context handles factual recall and temporal retrieval reasonably well but underperforms structured retrieval
on intent-sensitive queries, where the epistemic KG’s explicit assertion typing distinguishes resolved from active
conditions in ways implicit raw-text reading does not.
C7 (Deterministic KG).KG edges matching query concepts are returned directly without LLM reasoning. C7
nominallyscores27.2%overall,butthisisanevaluatorartifact: C7returnstemplaterefusals(“Norelevantknowledge
graph edges found”) for >98% of questions, and the word “No” in the template coincidentally matches negation-
category keywords (negation: 99.1%, all other categories: 0%). Semantic accuracy is 0%, confirming that structured
data without an LLM reasoner cannot answer clinical questions.
U Illustrative Retrieval Example
To illustrate why intent-aware routing matters, consider a change question:“What medications changed between this
patient’s first and second admissions?”
C1(LLMalone).Themodelseesonlythecurrentnote,whichmentionsmetoprolol,atorvastatin,and“discontinued
lisinopril.” Without prior admission records, it cannot determine what wasaddedvs.continued, and may hallucinate
prior medication lists.
C4g (intent-aware KG-RAG).The intent classifier routes toChangeretrieval, which partitions KG edges by
hadm_idand computes set differences:
•Added(admission 2 only): atorvastatin 40mg [Present]
•Removed(admission 1 only): lisinopril 10mg [Present→Absent]
•Continued: metoprolol 25mg [Present, both admissions]
The assertion labels (Present,Absent) disambiguate: “discontinued lisinopril” is not a current medication but a
historical one whose status changed—exactly the distinction the epistemic schema preserves.
V LLM-as-Judge Concordance
Tocomplementthedeterministickeywordevaluator,allwererescoredfor800predictions(400questions ×2conditions:
C1 and C4g) using Claude Opus 4.6 as an LLM judge. The judge prompt presents the question, reference answer,
andsystemanswer,andrequestsascoreof1(correct), 0.5(partiallycorrect),or0(incorrect)withaone-sentence
justification. Table 24 summarizes the concordance.
37

Table 24: Evaluator concordance comparison. LLM judge scores ≥0.5treated as correct for binary comparison.
Physician concordance computed on the𝑛=30pilot adjudication subset.
Metric Keyword v2 LLM Judge
vs. Keyword evaluator (𝑛=800)
Agreement — 78.9%
Cohen’s𝜅— 0.572
vs. Physician (𝑛=30)
Agreement 46.7% 56.7%
Too strict 43.3% 36.7%
Too lenient 10.0% 6.7%
Condition-level accuracy
C1 accuracy 21.8% 28.5%
C4g accuracy 68.5% 57.0%
C4g−C1Δ+46.8pp+28.5pp
Key findings.The LLM judge achieves higher physician concordance (56.7% vs. 46.7%) and is less prone to false
strictness(36.7%vs.43.3%), confirmingthatthekeywordevaluator’sconservativebiasinflatesthemeasured delta.
Under theLLMjudge, C1rises from21.8%to 28.5%(thejudgecreditsclinically reasonablehedging thatkeyword
matchingpenalizes), whileC4gdropsfrom 68.5%to57.0%(the judgepenalizespartialanswersthatpasskeyword
matching via term overlap). The resulting C4g −C1 delta under LLM judge ( +28.5pp) is lower than the keyword
delta (+46.8pp) but remains substantial and directionally consistent. The per-condition asymmetry—C1 gains, C4g
loses—is consistent with the physician finding that the keyword evaluator disproportionately penalizes C1 abstentions.
Per-category analysis.The LLM judge is particularly stricter on change (C4g: 62% mean score vs. 100% keyword)
because it penalizes partial medication lists that contain correct change keywords but miss specific drugs. Conversely,
thejudgeismoregenerousonhistorical(C4g: 62%vs.60%keyword)andnegation(C4g: 79%vs.81%keyword),
where its semantic understanding better captures correct answers that use varied phrasing.
Limitations.Usingthesamemodelfamily(Opus)forbothansweringandjudgingintroducespotentialself-preference
bias. The judge also shows moderate agreement with the keyword evaluator ( 𝜅=0.572 ), indicating they capture
partially overlapping but distinct aspects of answer quality.
W Assertion Classifier Evaluation
Therule-basedassertionclassifieruses122calibratedtriggerpatternsextendingtheNegEx[ 20]andConText[ 21]
frameworkstoa7-classtaxonomy(Table21). Table25summarizesthepatterninventoryandconfidencerangesby
category.
Corpusstatistics.Acrossthe43ClinicalBenchpatients,theknowledgegraphcontains3,943edgeslinkedtoclinical
facts. Of these, 618 (15.7%) carry non-present assertions: 442 absent (11.2%), 94 possible (2.4%), 71 historical
(1.8%), 8 conditional (0.2%), and 3 hypothetical (0.1%). At the mention level, 1,428 of 12,379 mentions (11.5%) are
non-present. Thisnon-presentfraction 𝑓np=0.157providestheempiricalboundfromCorollary3: anassertion-blind
pipeline cannot exceed 1−𝑓np=84.3% assertion-faithful accuracy on concepts where negated or uncertain mentions
are present.
Intrinsicevaluation.Astratifiedsampleof189mentionsfromthe43ClinicalBenchpatients,stratifiedbypredicted
assertion type (50 present, 51 absent, 15 possible, 28 historical, 19 conditional, 4 hypothetical, 10 family history), and
38

Table 25: Assertion trigger pattern inventory. Confidence ranges reflect per-trigger calibration.
Category Patterns Confidence Example triggers
Absent 31 0.85–0.98 “no evidence of,” “denies,” “ruled out”
Uncertain 32 0.35–0.75 “possible,” “likely,” “consistent with”
Present 27 0.85–0.98 “confirmed,” “diagnosed with,” “positive for”
Hypothetical 11 0.25–0.35 “risk of,” “screening for,” “prophylaxis”
Family history 8 0.85–0.95 “family history of,” “maternal history”
Conditional 7 0.20–0.40 “if,” “contingent on,” “depending on”
Historical 6 0.75–0.90 “history of,” “former,” “remote history”
Total 1220.20–0.98
aphysician(A.S.)annotatedreference-standardlabelsinablinded,randomizedreview. Table26reportsper-class
precision, recall, and F1.
Table26: Intrinsic assertion classifier evaluation on 189 physician-annotated MIMIC-IVmentions (stratified sample
from 43 ClinicalBench patients).
Assertion𝑛P R F1
Present 62 0.980 0.790 0.875
Absent 51 0.980 0.961 0.970
Possible 15 0.500 1.000 0.667
Historical 28 0.933 1.000 0.966
Conditional 19 1.000 0.789 0.882
Hypothetical 4 1.000 1.000 1.000
Family history 10 0.900 0.900 0.900
Weighted avg189 0.933 0.894 0.902
Overall accuracy is 89.4% (169/189; 95% Wilson CI: [84.2%, 93.0%]) with Cohen’s 𝜅=0.867 (strong agreement).
NegationdetectionachievesF1=0.970,consistentwithpublishedbenchmarks: NegExP=94.5%/R=77.8%[ 20],
NegBio P=96.3%/R=85.7% [ 46]. The dominant error pattern (11/20 errors) is over-triggering uncertainty: the
classifierassignspossibletomentionsnearhedginglanguage(“likely,” “concerningfor”)thatphysiciansjudgeas
present. This explains the low precision for thepossibleclass (P=0.50) despite perfect recall.
Functional evaluation.Rather than relying solely on intrinsic metrics, the C4 ablation provides a functional
evaluationofassertionquality. C4addsassertionmetadatatoC3’sgraphwithoutintentrouting: theclassifier’soutput
isdirectlyconsumedbytheretrievalpipeline. Onassertion-sensitivecategories(negation,conditional,uncertainty,
familyhistory),C4outperformsC3byanaverageof +16.1pp,confirmingthattheclassifierproducesusableassertions
for these categories. On temporal categories (historical, sequence, change, current state, duration), C4 underperforms
C3byanaverageof −24.5pp—notbecauseassertionsareincorrect,butbecauseuniformBFStraversalcannotexploit
them effectively. The C4→C4g routing correction recovers these losses and amplifies the gains, achieving+22.0pp
overall (𝑝<10−10).
X Physician Adjudication: Full Results and Reference Answer Evolution
Thissectionreportscompleteresultsfromtheblindedphysicianadjudication( 𝑛=120pairedquestions)anddocuments
the reference-answer evolution.
39

Table 27: Physician-judged accuracy by category and condition ( 𝑛per condition). Strict = correct only; lenient =
correct + partially correct. Categories sorted by C4g strict accuracy.
C1 C4g
Category𝑛Strict Lenient Keyword Strict Lenient Keyword
Historical 15 40.0% 66.7% 6.7% 86.7% 86.7% 46.7%
Conditional 10 60.0% 80.0% 0.0% 80.0% 100.0% 50.0%
Family hist. 10 50.0% 90.0% 0.0% 80.0% 90.0% 50.0%
Uncertainty 20 25.0% 45.0% 5.0% 75.0% 90.0% 50.0%
Duration 10 30.0% 60.0% 40.0% 70.0% 90.0% 50.0%
Current st. 15 33.3% 53.3% 20.0% 66.7% 86.7% 46.7%
Negation 5 0.0% 20.0% 40.0% 60.0% 80.0% 40.0%
Sequence 20 10.0% 45.0% 10.0% 55.0% 90.0% 50.0%
Change 15 6.7% 20.0% 0.0% 0.0% 46.7% 100.0%
Overall 120 27.5% 52.5% 10.8% 62.5% 84.2% 55.0%
X.1 Per-Category Physician Accuracy
The keyword evaluator dramatically underscores C1 on conditional (0% vs. 80% physician lenient) and family history
(0% vs. 90%), where the model gives clinically reasonable hedged answers that lack specific keywords. The change
categoryexhibitstheoppositepattern: thekeywordevaluatorreportsC4gat100%,butthephysicianratesitat0%
strict / 46.7% lenient—keyword matching catches medication names without verifying comparison logic. Family
history shows no physician-judged delta (both conditions score ∼90%lenient), suggesting that the LLM already
handles this category well without KG-RAG when evaluated by a physician—a finding masked by the keyword
evaluator, which scores both conditions at 0% on C1.
X.2 Evaluator Agreement
Table 28: Overall keyword evaluator agreement with physician judgment ( 𝑛=240items from full adjudication). The
evaluatorisoverwhelminglytoostrict(7.5:1strict-to-lenientratio),andthemajorityoferrorstracetoreference-answer
defects rather than evaluator logic.
Evaluator Outcome𝑛%
Agrees with physician 130 54.2%
Too strict (false negative) 97 40.4%
Too lenient (false positive) 13 5.4%
Strict:lenient ratio 7.5:1
Themajorityofevaluatorerrors(63%offalsenegatives,85%offalsepositives)tracetoreference-answererrors
rather thanevaluator logic: whenthe referenceanswer iscorrect, theevaluator achieves64.2% physicianagreement
withonlya1.9%false-positiverate. Thelow 𝜅(0.18)despite54.2%rawagreementconfirmstheevaluator’serrors
are systematic (overwhelmingly too strict) rather than random. Table 29 breaks this down by category.
X.3 Safety and Clinical Utility by Condition
Table 31 breaks safety down by category.
Change is the most safety-concerning category: 83.9% of change items have safety concerns (minor or harmful),
mirroring the reference-answer quality and model accuracy findings.
40

Table29: Keywordevaluatoragreementwithphysicianjudgmentbycategory. Categorieswheretheevaluatorfails
worst are highlighted.
Category𝑛Agreement Too Strict Too Lenient
Conditional 20 35.0% 65.0% 0.0%
Family hist. 20 35.0% 65.0% 0.0%
Historical 30 50.0% 50.0% 0.0%
Uncertainty 40 55.0% 42.5% 2.5%
Current state 30 56.7% 40.0% 3.3%
Sequence 40 57.5% 40.0% 2.5%
Duration 20 60.0% 35.0% 5.0%
Change 30 66.7% 6.7% 26.7%
Negation 10 70.0% 20.0% 10.0%
Table 30: Physician-judged clinical safety and utility by condition ( 𝑛=120paired questions). C4g is safer and more
useful than C1; the “misleading” rate is similar between conditions.
Dimension Rating C1 C4gΔ
SafetySafe 60.8% 76.7%+15.8pp
Minor concern 33.3% 20.8%−12.5pp
Potentially harmful 5.8% 2.5%−3.3pp
UtilityHelpful 30.8% 67.5%+36.7pp
Neutral 20.0% 15.0%−5.0pp
Not useful 34.2% 2.5%−31.7pp
Misleading 15.0% 15.0%0.0pp
X.4 Qualitative Themes from Physician Notes
Free-text notes (165/241 items, 68.5%) reveal five recurring themes:
1.Referenceanswerssystematicallywrongformedicationchangequestions(103notesmentioningreference-
answer issues): Every reference answer in the change category conflates inpatient medication orders(heparin, IV
antibiotics, CIWA protocol) with discharge medications.
2.C1 hallucinates from limited context(15 notes, exclusively C1): Without retrieval, C1 fabricates admission IDs,
medication names, and clinical scenarios. Zero C4g items received this complaint.
3.NLP assertion classifier propagates errors(8 notes): Boilerplate discharge instructions (“call if fever >101.5”)
tagged as clinical findings; “h/o recently diagnosed metastatic cancer” tagged as historical.
4.Safety-criticalerrors(10itemsflaggedaspotentiallyharmful): Codestatuserrors(model“hallucinatesDNR
confirmation” when chart says full code), active cancer missed from medication list, anticoagulation misclassified.
5.Modelpraisedwhenreferenceanswerswerewrong(63notes): Reviewernotedthemodelgaveclinicallycorrect
answers that the automated benchmark reference penalized.
X.5 Reference Answer Version History
The ClinicalBench reference answers have undergone iterative refinement:
•v1 (auto-generated reference set): Reference answers created by LLM from MIMIC-IV notes via the NLP
extraction pipeline. No physician review. 400 questions.
•v2 (partially corrected reference set): From the initial 𝑛=30pilot, 54 corrections were applied to the most
egregious errors.2This is the version used for all reported numbers. Both v1 and v2 are released.
2Theshipped corrections.json filedocuments53explicitcorrections;thediffbetweenv1andv2 expected_answer fieldscontains54
differing items, because one item was adjusted post-corrections.json during a final reconciliation pass.
41

Table 31: Clinical safety ratings by category in the audited record set. Categories sorted by safety concern rate.
Category𝑛Safe Minor Harmful
Change 31 16.1% 74.2% 9.7%
Uncertainty 40 57.5% 35.0% 7.5%
Current st. 30 63.3% 33.3% 3.3%
Negation 10 70.0% 20.0% 10.0%
Historical 30 73.3% 23.3% 3.3%
Sequence 40 87.5% 12.5% 0.0%
Conditional 20 90.0% 5.0% 5.0%
Duration 20 90.0% 10.0% 0.0%
Family hist. 20 95.0% 5.0% 0.0%
•v3 (planned physician-validated release): Full corrections from the 𝑛=120adjudication plus external validation.
Triage summary: 45 questions KEEP (37%), 55 FIX_GOLD (46%), 20 REPLACE_QUESTION (17%). Of the 55
FIX_GOLD items, 46 have drafted proposed corrections. This future release is not used for any numbers reported
in this paper.
X.6 Systematic Error Taxonomy
Five systematic failure modes account for all 78 problematic questions (of 120 adjudicated):
1.NLP assertion classifier error(28 questions, 36%): The dominant failure. Manifests as: “history of heart failure”
→“heartfailureisresolved”(clinicalidiommeansactivechroniccondition);“edema,likelyduetononcompliance”
→“edema is uncertain” (causal vs. existential uncertainty conflation); experiencer tag reversal (patient’s atrial
fibrillation labeled as family history).
2.Wrong answer / inverted truth(16 questions, 21%): The reference answer states the opposite of the chart.
Example: the reference says pitting edema is absent when PE documents “2+ pitting edema bilaterally.”
3.Non-clinical entity extraction(11 questions, 14%): NLP extracted boilerplate (“call if fever >101”), devices
(Foley catheter as diagnosis), lab values (blood sugar as diagnosis), or section headers (“Allergies” as medical
condition).
4.Medication list conflation(10 questions, 13%): Change questions compared wrong lists—inpatient orders
(heparin, IV antibiotics) vs. discharge medications, or admission med-rec vs. discharge list. PRN-only medications
(CIWA Valium) counted as prescribed.
5.Fabricated temporal relationship(8 questions, 10%): Sequence questions claimed ordering not supported by the
chart—both conditions in the same admission with no temporal anchoring, or based on NLP-extracted entities
from negated text.
X.7 Impact on Reported Numbers
Because the detected defects are question-level rather than condition-specific, they are less likely to reverse the
directionoftheC1–C4gcomparison. Theydo,however,materiallyaffectabsoluteaccuraciesandsomecategory-level
magnitudes,especiallyforchangeandhistoricalquestions. Thekeywordevaluatorachieves64.2%physicianagreement
andonly1.9%false-positiveratewhenrestrictedtoquestionswithcorrectreferenceanswers,confirmingthatevaluator
errors are dominated by reference-answer noise rather than matching logic.
Y Cohort Demographics and Subgroup Analysis
The 43 ClinicalBench patients are drawn from MIMIC-IV [ 24], a single academic medical center (Beth Israel
Deaconess Medical Center, Boston). The cohort skews female (60.5%), White (76.7%), and Medicare-insured
(44.2%), reflecting the source institution’s patient mix.
42

Subgroup accuracy.Table 32 reports C1 and C4g accuracy by demographic group. Because questions are
distributedunevenlyacrosspatientsandgroupsizesaresmall( 𝑛≤26patientsperstratum),thesecomparisonsare
severelyunderpowered; they arereported for transparency, not for drawing subgroupconclusions. Nostatistically
significant interaction between demographic group and condition was observed, but the analysis cannot rule out
meaningful effect modification.
Table32: ClinicalBenchaccuracybydemographicsubgroup(underpowered;reportedfortransparency). 𝑛𝑞=number
of questions in each stratum.
Characteristic Group𝑛 pts𝑛𝑞C1 (%) C4g (%)
SexFemale 26 249 22.5 69.9
Male 17 151 20.5 67.5
Age<65 28 258 20.5 69.4
≥65 15 142 23.9 68.3
RaceWhite 33 290 22.4 71.7
Non-White 10 110 20.0 61.8
Generalizability limitations.The single-site, predominantly White cohort limits external validity. MIMIC-IV
emergency department notes are heavily templated; performance on narrative-heavy specialties (psychiatry, palliative
care)orcommunityhospitaldocumentationstylesisunknown. Multi-sitevalidationwithdemographicallydiverse
cohorts is needed before deployment claims can be made.
Z Ethics, Broader Impacts, and Detailed Threat Analysis
This work uses de-identified clinical data from MIMIC-IV [ 24] under a PhysioNet Credentialed Health Data Use
Agreement. No patient re-identification was attempted. The system is designed for clinical decisionsupport, not
autonomous clinical decision-making.
Broaderimpacts.Improvedassertion-faithfulretrievalcouldreduceclinicalerrorscausedbynegationorfamily-
history misattribution, particularly in high-volume settings where physicians cannot review every note. However,
deploymentrisksincludeover-relianceonautomatedepistemiclabeling(falseconfidenceinassertionstatus),brittleness
to out-of-distribution clinical language, and the potential for structured outputs to appear more authoritative than their
accuracywarrants. Responsibledeploymentrequiresprospectiveevaluation,integrationwithclinicianworkflows,and
clear communication of system limitations. Because ClinicalBench is single-site, predominantly White, and built
fromheavilytemplatedMIMIC-IVdocumentation,asystemorbenchmarktunedonitcouldoverfitnotestyleand
underperform on other hospitals, specialties, or populations. This creates a coverage and fairness risk: strong results
on this stress test could be mistaken for portable performance when they may primarily reflect source-institution
conventions. ClinicalBench should be used for failure analysis and ablation, not as evidence of deployment readiness
or demographic robustness.
Evaluator bias characterization.The keyword evaluator has known limitations. Negation scoring checks keyword
presencewithoutverifyingdirection(e.g., “pneumoniaisabsent”and“patienthaspneumonia”couldbothmatch).
Longermodeloutputsnaturallycontainmorekeywordmatches,creatingaverbositybias(Opusaverages337characters
vs. GPT-OSS 139 characters). Echo-stripping may differentially affect structured vs. prose outputs. These biases
primarily affectcross-modelcomparisons; within-model ablations (C1 vs. C4g) use the same model’s output format
across conditions, so evaluator biases cancel.
43

Reference-answer correction methodology.Benchmark label quality is substantially imperfect: full physician
adjudication ( 𝑛=120questions, 241 audited records) found a 56% defect rate in provisional reference answers.
Defects are not random but trace to five systematic pipeline failures: NLP assertion classifier errors (36% of defects),
inverted-truth reference answers (21%), non-clinical entity extraction (14%), medication list conflation (13%),
and fabricated temporal relationships (10%). The NLP assertion classifier is the dominant source, systematically
misinterpreting “history of X” as “X is resolved” and conflating causal uncertainty with existential uncertainty
(Appendix X). Reference-answer corrections were made by the lead physician (A.S.), who also designed the system.
Toquantifyimpact,allwererescoredforconditionsagainstboththeoriginal(v1)andpartiallycorrected(v2)reference
sets: v1→v2correctionsimprovedallmodelssimilarly( +0.5–1.2pp),preservingwithin-modeldeltasinthathistorical
comparison. All results reported in this paper use the v2 reference set (54 corrections from an initial 𝑛=30pilot,
reconciled against the v1 →v2expected_answer diff); both v1 and v2 are released for reproducibility. A future v3
releasewillincorporateconsensuscorrectionsfromthemulti-revieweradjudicationoncethatstudyiscomplete;nov3
resultsareusedinthismanuscript. Thecompletedthree-rateradjudication(twoindependentexternalphysicians ×
100 items) addresses single-rater bias; results appear in Section 4.7.
Scopeandframing.Thisstudyisanablationanalysisratherthanahead-to-headleaderboardagainstexistingmedical
RAG systems [ 8,9,39]; C2 is used as a proxy baseline for document-level retrieval on this cross-admission task.
Runtimevariancefromquantizedlocalinference(e.g.,MedGemma ±10ppacrossruns)andsingle-siteprovenance
further limit external validity.
Z.1 Extended External Adjudication Details
Rater-calibration detail.Reviewer 3 rates 65/100 model answers as “correct” (vs. 46/100 Reviewer 1 and 39/100
Reviewer 2) and uses “partially correct” only 10/100 times (vs. 22/100 and 29/100 respectively): borderline answers
arecollapsedinto“correct”onbothconditions,compressingthebetween-conditiondelta. Excludingchange,herdelta
rises to+6.8pp(still underpowered, 𝑝=0.65). Ordinal linear-weighted pairwise Cohen’s 𝜅on the 3-class model-
answer scale averages 0.463 (quadratic-weighted 0.527) across the three reviewer pairs. The structurally-independent
inter-non-authorCohen’s 𝜅(Hird×Nadeem,binarystrict)is 0.36,versus 0.43–0.49forauthor-involvingpairsand
Fleiss’𝜅=0.413 on the full three-rater binary-strict scale; the lower non-author 𝜅is consistent with author influence
on the other two reviewer pairs and is one reason the three-rater majority-vote +24.0pp result should be weighted
with this dependency in mind. All 100 items contribute to 𝜅; the planned 10-item calibration phase was replaced by
written instructions only, so no items were excluded.
Reviewer 3 gold-standard pattern.Reviewer 3’s unusual combination of the strictest gold-standard ratings
(31/100fullycorrect)alongsidethemostlenientmodel-answerratings(65/100correct)isinternallycoherent: she
frequently judges the reference answer as wrong while still finding the model’s answer clinically reasonable—the
same phenomenon the internal audit reports, now independently replicated.
Scoring assumptions.Under lenient scoring (“correct or partially correct”), the 3-rater majority-vote delta is
+12.0pp (𝑝=0.18 , n.s.) on the full set and +15.9pp (𝑝=0.09 ) withchangeexcluded—directional but not
statisticallysignificant. Thethree-ratervalidationcharacterizesthephysician-perceivedmagnitudeunderthreescoring
assumptions: strict 2/3 majority ( +24.0pp, significant), lenient 2/3 majority ( +12.0pp, n.s.), and per-reviewer deltas
(+2to+36pp).
Z.2 Leave-Author-Out Sensitivity for Three-Rater Majority
Toassesssensitivityofthe +24.0pp three-ratermajorityresulttoauthorinclusion(Reviewer1=A.S.,theleadauthor),
werecomputedtheC1vs.C4gdeltausingonly thetwostructurally-independentexternalraters(HirdandNadeem).
Table 33 summarizes the alternative aggregations.
44

Table33: Leave-author-outsensitivityforthethree-raterexternalvalidation( 𝑛=50itemspercondition,strictscoring).
The author-involving 3-rater majority (top row, paper headline) is contrasted with author-excluded aggregations.
AggregationΔ(C4g−C1) McNemar𝑝
3-rater majority strict (headline)+24.0pp0.0075
Hird-only strict+30.0pp —
Nadeem-only strict+2.0pp —
Hird×Nadeem unanimous strict+22.0pp0.0192(paired exact)
Hird OR Nadeem strict+10.0pp —
Inter-non-author Cohen’s 𝜅=0.36(Hird×Nadeem) versus 0.43–0.49for author-involving pairs. The substantive
directionsurvivesauthorexclusionandthemagnitudeispreserved( −2pp,from+24.0pp to+22.0pp underunanimous
external agreement); the paired exact McNemar test on the 50 matched questions gives 4 discordant pairs favoring C1
vs. 15 favoring C4g, two-sided𝑝=0.0192, retaining significance at𝛼=0.05.
Z.3 Inter-Rater Agreement on the Gold Standard
Inter-rater agreement was measured onsystem output ratings(Fleiss 𝜅=0.413 on C1/C4g/equivalent labels; pairwise
Cohen𝜅∈{0.36,0.43,0.49} , see Appendix Z.1 and Section 4.7). Agreement on thecorrectness of the gold standard
answers themselveswas not formally measured in this study—a known gap in the benchmark methodology that limits
direct quantification of gold validity.
Indirectly, the external raters separately found 61–64% reference defect rates on the v2 gold standard (Section 4.6),
implyinglowgold-validityagreement betweenthev2 referenceand externalclinicaljudgment. A v3goldstandard
with multi-rater authoring (eachitem independently authoredby ≥2raters with consensusreconciliation) and explicit
IAA on the reference answers themselves is planned for post-publication release; no v3 numbers are reported in this
paper.
Z.4 Component-Level F1 of the Assertion Classifier
The 122-pattern rule-based assertion classifier (Appendix W) wasnotexternally benchmarked on i2b2-2010 or n2c2-
2010held-outsetsinthiswork;component-levelF1isreportedonlyfrominternalvalidationonaphysician-annotated
stratifiedsampleof189 ClinicalBenchmentions(accuracy89.4%,weightedF10.902,Cohen 𝜅=0.867 ; Table26).
Externalcomponent-levelevaluationinthei2b2/n2c2traditionisflaggedasfutureworkandwouldstrengthenthe
assertion-preservation claim by separating intrinsic classifier quality from downstream KG-RAG retrieval gains.
Z.5 Regulatory and Governance Framing
Scope.ThisworkdoesNOTestablishclinicaldeploymentreadiness. Theframingsbelowaregovernancescaffolding
for any future deployment effort, not claims about the current paper. EpiKG is evaluated here as a research probe; no
intended-use statement, no IRB-approved deployment protocol, and no prospective clinical study is in scope.
SaMD risk class (candidate).If deployed for patient-level clinical answers from EHR retrieval, EpiKG would
plausiblyfallunderFDASaMDClassIIb–IIIaunderIMDRFcriteria(high-impactdecisionsupportoverhigh-severity
conditions, where individual answers can influence diagnosis or therapy). Class assignment depends on intended use,
clinical context, and clinician oversight model: a tool used for care-team chart-review triage with mandatory clinician
adjudication would sit lower on the risk spectrum than one driving patient-facing answers. The system as evaluated is
research-only and does not have an intended-use statement.
PCCP elements (per FDA December 2024 final guidance).An AI-enabled SaMD deploying EpiKG would
specify a Predetermined Change Control Plan covering modifiable components: (a) the intent-aware retrieval policy
45

(routing rules that select between assertion-, temporal-, and entity-typed retrieval slices), (b) the 122-pattern assertion
classifier(ruleadditions,deletions,ormodifications),(c)theOMOPvocabularyversion(e.g.,conceptadditionsin
successiveAthenareleases),and(d)thebaseLLMversionsacrossthecross-modelstack(Claude,MedGemma,Qwen,
Gemma 4, GPT-OSS). Each component requires a Description of Modifications, a Modification Protocol, and an
ImpactAssessmentundertheDecember2024finalguidance. Noneofthesegovernanceartifactsarepresentinthis
work; they are flagged as deployment prerequisites.
CHAIAssuranceReportingChecklistself-mapping.TheCHAI2024AssuranceReportingChecklistandthe
Joint Commission–CHAI September 2025 governance guidance enumerate categories that any deployment effort
must address. A high-level self-assessment against those categories follows:
•Governance:no external clinical-AI governance committee, model risk-management board, or institutional review
structure is engaged.
•Privacy:de-identifiedMIMIC-IVunderPhysioNetcredentialedaccess;theHuggingFacereleasecontainsrephrased
questions, paraphrased reference answers, and predictions without raw note text (Appendix P).
•Transparency:modelversions,evaluatorcode,rawpredictions,andphysicianadjudicationdataarepubliclyreleased;
the full application stack is not (Appendix P).
•Data security:HuggingFace public release; commitda5f5b1stripped raw note excerpts.
•Safety event reporting:no post-deployment safety surveillance protocol; this is research-only.
•Riskandbiasassessment:demographictableandequitygapreported(AppendixY);nostructuredexternalbias
review.
Thepaperprovideshonestdisclosureonmostcategories;structuredexternalreviewandaformalassurance-checklist
filing are absent. Any deployment effort would need to close those gaps before clinical release.
Algorithmovigilance plan (sketch only).Per Embí (JAMIA 2021) and Davis, Embí, and Matheny (JAMIA 2024),
a clinical-AI algorithmovigilance program for EpiKG would minimally include: (a) drift monitoring—e.g., monthly
per-category accuracy on a held-out chart sample, with drift alerts triggered when any category accuracy moves
more than 5pprelative to a rolling baseline; (b) periodic re-validation—annual re-validation against a refreshed gold
standard, with attention to upstream NLP-pipeline regressions; and (c) equity surveillance—per-demographic-stratum
accuracy with alerting on >5ppgaps between strata. None of these are implemented in this work; they are flagged as
deployment prerequisites and are not in scope for the present manuscript.
Harm-pathwayanalysis.The10“potentiallyharmful”itemsidentifiedbyphysicianadjudication(AppendixX;
Table 31) span code status, oncology, and anticoagulation. For governance purposes these can be cast in an FDA-style
probability×severity matrix (Table 34). The matrix is illustrative; deployment would require structured FMEA with
multidisciplinary review.
Equitygapreframing.The 9.9ppNon-Whitevs.WhiteaccuracygapontheC4gendpointisreporteddescriptively
in the demographic table (Appendix Y) with the disclaimer “severely underpowered ( 𝑛=10Non-White).” We
additionallyframethisasanequitysignalwarrantinginvestigationinanyfuturedeploymenteffort,perCHAI/FDA
expectations on intended-use-populationfairness. The current sample sizecannot exclude either a true equitygap or
samplingnoise;deploymentwouldrequiredemographicallydiversemulti-sitevalidationwithpre-specifiedper-stratum
accuracy targets and stratum-conditional recalibration if gaps persist.
Joint Commission–CHAI September 2025 alignment.The Joint Commission–CHAI September 2025 “Responsi-
bleUseofAIinHealthCare”guidanceistheoperativegovernancestandardforclinical-AIdeploymentgoingforward.
EpiKG is not deployed in any healthcare setting in scope of this paper, and the present manuscript should not be read
as a Joint Commission compliance document. Any institutional deployment of EpiKG (or a derivative) would be
expected to map its governance, monitoring, and reporting practices to that guidance prior to clinical release.
46

Table 34: Illustrative probability ×severity matrix for the 10 potentially harmful items observed in the internal
120-paired-item physician adjudication ( 𝑛=240single-condition ratings; Table 31). Probability is the observed rate
within the audited subset; deployment-context base rates would differ.
Severity Probability Examples
High (acute risk to life) Unknown / unmea-
suredHallucinated DNR confirmation when chart
documents full code; missed active cancer in
current-state retrieval
Medium (clinically mate-
rial)Observed (e.g.,
1/240≈0.4%
on the internal
120-paired-item
adjudication,
i.e. 240 single-
condition ratings;
not the C1b
discharge-only
extension)Anticoagulationmisclassification(activevs.his-
torical); medication-list conflation between in-
patient and discharge orders
Low (low-acuity drift) Observed at low
rateVerbose-but-correct hedging that could be mis-
read as uncertainty; over-confident assertion-
status framing on edge-case phrasing
47