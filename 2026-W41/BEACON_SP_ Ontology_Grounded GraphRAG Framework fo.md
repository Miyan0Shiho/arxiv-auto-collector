# BEACON-SP: Ontology-Grounded GraphRAG Framework for Clinical Suicide Risk Assessment

**Authors**: Kemal Davaslioglu, Nathan Conger, Sastry Kompella, Yalin E. Sagduyu, Nathaniel D. Bastian

**Published**: 2026-10-06 19:23:51

**PDF URL**: [https://arxiv.org/pdf/2610.09026v1](https://arxiv.org/pdf/2610.09026v1)

## Abstract
We present BEACON-SP, an ontology-grounded Graph Retrieval-Augmented Generation (GraphRAG) framework for clinician-facing decision support in behavioral health settings such as suicide prevention, where effective assessment requires integrating heterogeneous clinical, behavioral, social, and temporal evidence. BEACON-SP combines patient knowledge graphs with ontology-guided retrieval to support multi-hop reasoning across diagnoses, medications, risk and protective factors, life events, and temporal relationships. The framework is enabled by a comprehensive suicide prevention ontology that integrates the Three-Step Theory, the Integrated Motivational-Volitional Model, and the Suicide Social Determinants of Health Ontology into a unified representation of patient risk factors. We construct ontology-grounded patient knowledge graphs and evaluate BEACON-SP for clinician-facing question answering. Compared with a vector-based retrieval-augmented generation (RAG) baseline on a 1,500-query benchmark spanning 15 clinical categories and 100 patients, BEACON-SP improves completeness, clinical relevance, and evidence grounding under a corrected comparative evaluation protocol, with a small gain on factual accuracy. In paired criterion-level comparisons, GraphRAG is preferred in 76.4% of cases. These results demonstrate the potential of ontology-guided GraphRAG to provide structured, contextualized patient evidence for clinical decision support.

## Full Text


<!-- PDF content starts -->

BEACON-SP: Ontology-Grounded GraphRAG
Framework for Clinical Suicide Risk Assessment
Kemal Davaslioglu1, Nathan Conger1, Sastry Kompella1, Yalin E. Sagduyu1, and Nathaniel D. Bastian2
1Nexcepta, Gaithersburg, MD, USA
2United States Military Academy, West Point, NY , USA
Abstract—We present BEACON-SP, an ontology-grounded
Graph Retrieval-Augmented Generation (GraphRAG) frame-
work for clinician-facing decision support in behavioral health
settings such as suicide prevention, where effective assessment
requires integrating heterogeneous clinical, behavioral, social,
and temporal evidence. BEACON-SP combines patient knowl-
edge graphs with ontology-guided retrieval to support multi-
hop reasoning across diagnoses, medications, risk and protective
factors, life events, and temporal relationships. The framework
is enabled by a comprehensive suicide prevention ontology that
integrates the Three-Step Theory, the Integrated Motivational-
Volitional Model, and the Suicide Social Determinants of Health
Ontology into a unified representation of patient risk factors.
We construct ontology-grounded patient knowledge graphs and
evaluate BEACON-SP for clinician-facing question answering.
Compared with a vector-based retrieval-augmented generation
(RAG) baseline on a 1,500-query benchmark spanning 15 clinical
categories and 100 patients, BEACON-SP improves completeness,
clinical relevance, and evidence grounding under a corrected
comparative evaluation protocol, with a small gain on factual
accuracy. In paired criterion-level comparisons, GraphRAG is
preferred in 76.4% of cases. These results demonstrate the
potential of ontology-guided GraphRAG to provide structured,
contextualized patient evidence for clinical decision support.
Index Terms—Clinical decision support, Knowledge graphs,
Ontology, Retrieval-augmented generation, GraphRAG, Suicide
prevention, Electronic Health Records.
I. INTRODUCTION
The growing complexity of modern healthcare has increased
the need for intelligent tools that can integrate heterogeneous
clinical, behavioral, social, and temporal information. Despite
advances in AI, synthesizing such evidence into actionable
clinical assessments remains challenging, particularly in sui-
cide prevention where risk is driven by interacting psychiatric,
social, environmental, and temporal factors.
Suicide is one of the leading causes of death in the United
States, posing a profound public health challenge. According
to the Substance Abuse and Mental Health Services Adminis-
tration (SAMHSA), more than 14 million people in the United
States contemplated or attempted suicide in 2024 [1]. Suicide
is the second leading cause of death among Veterans under age
45 [2]. In 2023, Veterans accounted for 6,398 of 47,711 U.S.
adult suicides, a rate of 35.2 per 100,000 Veterans [2]. Risk is
driven by psychiatric conditions, including anxiety, depression,
PTSD, and substance use disorders, as well as social deter-
minants such as housing, financial security, and healthcare
This research was, in part, funded by the U.S. Government. The views and
conclusions contained in this document are those of the authors and should
not be interpreted as representing the official policies, either expressed or
implied, of the U.S. Government.access. Despite this complexity, existing clinical tools do not
comprehensively integrate diagnoses, social determinants, and
behavioral indicators into a unified risk assessment. Conse-
quently, they provide limited support for timely, personalized
intervention and evaluation of veteran suicide risk.
In this paper, we present Better Evaluation and Assessment
for Comprehensive Mental Health Outcomes and Naviga-
tion Suicide Prevention (BEACON-SP), an ontology-grounded
Graph Retrieval Augmented Generation (GraphRAG) frame-
work for clinician-facing decision support in behavioral health
settings. BEACON-SP combines patient-specific knowledge
graphs, ontology-guided retrieval, and graph-based evidence
expansion to support multi-hop reasoning across diagnoses,
medications, risk factors, protective factors, life events, and
temporal relationships. The framework is not intended to au-
tonomously assess suicide risk or replace clinician judgment.
Instead, it serves as a clinician-assistive system that retrieves
and contextualizes patient evidence, identifies relevant risk and
protective factors, maps clinical findings to established suicide
theories, and supports structured risk formulation. BEACON-
SP incorporates a comprehensive suicide prevention ontology
and a reproducible pipeline for constructing patient knowledge
graphs from Electronic Health Records (EHRs).
This paper makes the following contributions:
•We introduce BEACON-SP, an ontology-grounded
GraphRAG framework that combines semantic retrieval
and graph traversal to support clinician-facing question
answering and structured suicide-risk assessment.
•We design a two-pass GraphRAG retrieval strategy that
combines cosine-similarity seed retrieval with breadth-
first graph expansion to capture clinically relevant rela-
tional and temporal context.
•We introduce a comprehensive suicide prevention ontol-
ogy integrating the Three-Step Theory (3ST), the In-
tegrated Motivational-V olitional (IMV) model, and the
Suicide SDoH Ontology (Suicide-SDoHO) into a unified
patient-level knowledge representation.
•We present a reproducible pipeline for constructing
ontology-grounded patient knowledge graphs from EHRs.
•We evaluate BEACON-SP GraphRAG against a vector-
based RAG baseline across 1,500 clinical queries span-
ning 15 suicide-prevention tasks and 100 patients.
•We identify and correct a systematic evaluation bias that
underestimates graph-based retrieval systems, and intro-
duce a comparative protocol that evaluates each system
against its retrieved evidence.
arXiv:2610.09026v1  [cs.AI]  6 Oct 2026

The remainder of this paper is organized as follows. Sec-
tion II reviews related work. Section III summarizes the
suicide-risk theories that motivate BEACON-SP. Section IV
presents the ontology and knowledge graph. Section V de-
scribes the RAG and ontology-guided GraphRAG workflows.
Section VI presents the benchmark design and evaluation
results. Section VII concludes the paper.
II. RELATEDWORK
Recent work has explored RAG, knowledge graphs, and
Large Language Model (LLM)-based assistants for healthcare,
mental health, and clinical decision support. A GraphRAG that
links a corpus of medical literature to MIMIC-IV EHRs is
applied in [3] to improve evidence-based medical responses
from LLMs. A two-stage architecture for clinical risk pre-
diction is developed in [4] in which a fine-tuned LLM ex-
tracts task-relevant information from clinical notes enriched by
knowledge-graph retrieval and a second stage combines this
unstructured evidence with structured data to generate final
predictions. TrumorGPT, a GraphRAG-based LLM framework
is introduced in [5] for fact-checking health-related claims us-
ing semantic health knowledge graphs and semantic reasoning.
A mental-health LLM is combined with GraphRAG in [6]
for depression detection. LLM-Therapist [7] combines RAG
and function calling in a multimodal healthcare assistant for
personalized responses. Our prior work [8] explored contextual
RAG for domain-specific knowledge discovery; BEACON-SP
extends this approach with ontology-grounded patient graphs
for relational and temporal reasoning.
In contrast to these approaches, BEACON-SP organizes pa-
tient evidence through an ontology-based schema that unifies
3ST theory, IMV model, and Suicide-SDoHO ontology. The
resulting patient knowledge graph explicitly represents psychi-
atric diagnoses, medications, social determinants, risk factors,
protective factors, suicide-theory constructs, and temporal risk
windows. GraphRAG then operates over this patient-specific
graph to support both single- and multi-hop clinician-facing
questions, which provides theory-grounded retrieval context
rather than only semantically similar text chunks or generic
mental health guidance.
III. SUICIDERISKMODELING
BEACON-SP draws on three complementary frameworks
for representing patient-level suicide risk.
Three-Step Theory (3ST)characterizes suicide risk
through psychological pain, hopelessness, connectedness, and
capacity for suicide [9]. Ideation emerges when pain and
hopelessness co-occur, its severity depends on whether con-
nectedness offsets pain, and progression to attempt depends
on dispositional, acquired, and practical capacity.
Integrated Motivational-Volitional (IMV) Modelorga-
nizes suicide risk into pre-motivational, motivational, and
volitional phases [10]. These phases capture background vul-
nerability, development of suicidal ideation through defeat and
entrapment, and transition from intent to behavior through
factors such as means access, impulsivity, exposure to suicide,
and fearlessness about death.
SP-PACO
Physical Activity
•Exercise
•Intensity
•Frequency 
•Sedentary
•ProgramsSP-TEO
Temporal Events
•Time instant
•Time interval
•Ordering
•Duration 
•TrajectoriesSP-3ST
Three Step Theory
•Pain
•Hopelessness 
•Connectedness
•Capability
•Ideation
SP-SDoHO
Social Determinants
•Housing
•Employment
•Access to care
•Discrimination 
•Social supportBEACON -SP-Bridge
Integration Layer
•Increases construct
•Decreases construct
•Measures construct
•Targets construct
•Has context
•Has warning sign
•Has risk level
•Occurs duringBEACON -SP-Core
➢Person, Actor, Organization
➢Episodes: Ideation, Crisis, Attempt
➢Assessments & Instruments
➢Interventions & Plans
➢Risk, Protective/Contextual Factors
➢Outcomes & Risk StatesFig. 1: BEACON-SP Ontology components.
Suicide SDoH Ontology (Suicide-SDoH)provides a struc-
tured vocabulary for social and contextual contributors to
suicide risk, including social context, behavior and lifestyle,
economic stability, and suicide-specific concepts [11].
Together, these frameworks support a unified patient-level
ontology linking clinical observations to theory-grounded risk
constructs and temporal context.
IV. ONTOLOGYFRAMEWORK
Figure 1 illustrates the high-level architecture of BEACON-
SP which shows how the 3ST, Suicide-SDoH, the Physical
Activity Concept Ontology (PACO) [12], and the Time En-
tity Ontology (TEO) [13] connect to the BEACON Core,
with the BEACON Bridge layer that unifies patient features.
The BEACON-SP ontology contains 247 attributes organized
across 12 feature domains which cover mental health infor-
mation, medications, life events, risk and protective factor
profiles, and a temporal component that represents processes
and events over time.
The feature domains and their principal additions are as
follows:Risk factorsare derived directly from the Suicide-
SDoH across ten categories (behavioral, demographic, devel-
opmental, economic, education, legal, neighborhood, neuro-
biological, psychological, and social), replacing earlier gen-
eralized categories with clinically grounded suicide-specific
representations.Protective factorscapture conditions associ-
ated with reduced suicide risk, including social connectedness,
coping skills, stable employment, means restriction, and effec-
tive clinical care.Suicide behavior spectrumrepresents the
progression from passive and active ideation through suicidal
intent, planning, and behavior, incorporating constructs from
both IMV and 3ST.Life eventsencode triggering clinical
and social episodes such as hospitalization, discharge, job
loss, eviction, bereavement, and military deployment, as times-
tamped nodes.Medical conditionsinclude disorders with
established associations to elevated suicide risk: chronic pain,
traumatic brain injury, sleep disorders, epilepsy, cancer, HIV ,
and chronic kidney disease.Medicationscover psychiatric
pharmacotherapy organized by class: antidepressants (e.g.,
bupropion, mirtazapine etc.), antipsychotics, anxiolytics, mood
stabilizers, and ketamine, with interaction effects modeled
where multiple agents are co-prescribed.Psychiatric diag-
nosesspan major disorder categories from the DSM5-TR with

documented suicide associations: mood, psychotic, personal-
ity, anxiety, substance use, trauma-related, and neurodevelop-
mental disorders.Interventionsinclude hospitalization, psy-
chotherapy modalities, safety planning, lethal means counsel-
ing, crisis intervention, and related crisis-response treatments.
Assessments and biomarkersincorporate validated clinical
instruments (e.g., PHQ-9, GAD-7, CSSRS, PCL-5, AUDIT,
Beck-SSI) alongside biological markers (cortisol, serotonin,
C-reactive protein, IL-6) that index biological vulnerability.
Temporal featuresrepresent four time-dependent constructs:
state transitions (discrete timestamped clinical or life changes),
clinical epochs (treatment or contextual phases such as hos-
pitalization or incarceration), observation series (longitudinal
repeated measurements of the same clinical variable), and
risk windows (bounded periods of elevated risk triggered by
specific events such as post-discharge or medication initiation).
V. PATIENTKNOWLEDGEGRAPH ANDGRAPHRAG
IMPLEMENTATION
In a vector-based RAG pipeline, documents and patient
records are first divided into chunks, and each chunk is
encoded into a dense vector representation. A user query is
encoded using the same embedding model, and the retriever
ranks candidate chunks by similarity in the embedding space.
In our baseline, this similarity is computed using cosine
similarity,sim(q,d i) =q⊤di/(∥q∥∥d i∥), whereqis the
query embedding andd iis the embedding of theith patient-
record chunk. The top-kchunks are then concatenated into
the LLM context window. This approach works well when
the answer is contained in one or a few semantically similar
passages. However, suicide risk formulation often requires
integrating evidence distributed across diagnoses, medications,
social history, life events, and longitudinal clinical transitions.
These facts may be clinically related even when they are not
lexically or semantically close in the original record.
GraphRAG addresses this limitation by making relation-
ships explicit before retrieval. Instead of treating the patient
record only as flat text, BEACON-SP maps structured patient
attributes and derived constructs into a typed knowledge graph.
Nodes represent entities such as diagnoses, medications, en-
counters, risk factors, protective factors, interventions, suicide-
behavior constructs, and temporal events. Edges represent
meaningful relationships such as diagnosis history, medication
exposure, co-occurrence, temporal ordering, risk-factor mem-
bership, and ontology alignment. At query time, BEACON-
SP uses semantic retrieval to identify an initial set of graph
seeds and then expands from those seeds over typed graph
edges. This graph traversal supplies the relational and temporal
context needed to answer multi-hop clinical questions.
Figure 2 presents our framework that uses the BEACON-SP
ontology as the schema to construct patient-level knowledge
graphs rather than embedding documents as flat text. Struc-
tured patient data are mapped into graph nodes and edges
according to the ontology. Suicide theory constructs along
with temporal features are added as higher-order nodes that
capture causal and temporal relations among clinical events.
The resulting graph is stored in Neo4j with patient-specific
Knowledge Graph
(Neo4j Patient )BEACON -SP
Ontology
EHR Data (nodes , edges , ontology 
concepts , patient facts )
Clinical
QueryQuery Embedding
+ Entity LinkingOntology -Guided Graph + 
Vector RetrievalRetrieved Evidence 
Subgraph
Patient -Specific 
Evidence Subgraph
GraphRAG
Context Builder
Local LLM
(gpt-oss:20b )Grounded 
Output
Answer + 
Evidence Trace 
+ Retrieved 
Patient FactsOntology -guided steps:
(Semantic match , Cypher/path traversal
neighborhood expansion , Evidence ranking )Fig. 2: BEACON-SP GraphRAG system architecture.
identifiers to prevent cross-patient contamination. A patient-
specific subgraph is retrieved at query time and passed as
structured context to the language model.
Operationally, BEACON-SP uses a two-pass retrieval strat-
egy. First, the query is embedded and matched to patient-
specific graph node descriptions using cosine similarity to
select relevant seed nodes. Second, a bounded breadth-first
expansion retrieves neighboring nodes and relations within the
same patient graph, constrained by patient identifier, edge type,
and traversal depth. The resulting subgraph is serialized as
structured evidence with node labels, edge types, timestamps,
and source attributes. This combines semantic retrieval with
ontology-guided graph traversal, enabling BEACON-SP to
support both direct questions, such as medication history, and
relational questions about how diagnoses, stressors, recent
events, and protective factors shape the risk profile of a patient.
VI. EVALUATIONS
A. Experimental Setup
In our evaluations, we used the patient records in the Vet-
erans Affairs Suicide Prevention Synthetic Dataset (V ASPSD)
dataset [14] to create the BEACON-SP knowledge graph. We
perform both qualitative and quantitative assessments.
Ourqualitative assessmentsfocus on the responses of the
two retrieval systems using the same query. To isolate the
effect of retrieval structure, both systems use identical answer-
generation models and system prompts, and differ only in
whether context is described as coming from a vector store
or a knowledge graph.
Forquantitative assessments, we compare the ontology-
grounded GraphRAG against a vector-based RAG baseline on
a benchmark of 1,500 clinical queries drawn from 100 pa-
tients across 15 suicide-prevention query categories: diagnosis,
medication, suicide indicators, stressors, summary generation,
comorbidity, temporal reasoning, protective factors, risk fac-
tors, substance interaction, ontology mapping, risk construct
inference, risk window identification, protective versus risk
balance, and lethal means assessment (100 queries per cate-
gory). Both systems share the same answer-generation model
and prompting and differ only in their retrieval mechanism.
RAG retrieves semantically similar patient-record chunks from
a vector store, whereas BEACON-SP GraphRAG retrieves
patient-specific nodes and relationships from the knowledge
graph using a two-pass strategy: a cosine-similarity seed
pass followed by breadth-first graph expansion. This process
surfaces structured relations among diagnoses, medications,

What is the temporal sequence of this patient’s psychiatric history, including onset
dates, episode progression, and key clinical transitions?Question
Chronology of Psychiatric Illness
Date Diagnostic Event Clinical Note / Episode
1993-04-12 Major Depressive First documented MDD onset.
Disorder (MDD) No psychotic features noted.
...
Detailed EHR logs are omitted here due to space constraints.
...
2021-02-08 Psychotic Depression Further episode recorded;
– additional onset no remission documented.
Episode Progression & Key Transitions
1)Initial MDD (1993)– First major depressive disorder episode without
psychosis.
2)First Psychotic Episode (2003)– Depression acquires psychotic features,
marking a more severe phenotype.
3)Emergence of Schizophrenia (2005)– New schizophrenia diagnosis; antipsy-
chotic therapy initiated.
4)Onset of Anxiety (2007)– Anxiety disorder added to clinical picture.
5)Formal Psychotic-Depression Diagnosis (2013)– Psychotic-depressive
episodes formally coded as “Psychotic Depression.”
6)Relapses (2016–2020)– Recurrent psychotic-depression episodes indicating
chronic, relapsing illness.
7)Co-morbid Schizophrenia & Anxiety– Persistent diagnoses throughout
2005–2020.
8)Last Documented Episode (2020–2021)– Ongoing psychotic-depression
with no remission recorded.
Clinical Take-away
•Long-standing recurrent MDD progressing to psychotic depression since 2003.
•Schizophrenia (2005) and anxiety (2007) add complexity to the psychiatric
profile.
•Psychotic-depression becomes a chronic, distinct diagnosis from 2013 onward.
•Persistent, relapsing psychotic-depression with no documented remission
through 2021.
•Ongoing management with long-term antidepressant and antipsychotic therapy.GraphRAG
I have no retrieved patient data available to outline the temporal sequence of this
patient’s psychiatric history.Vector-based RAG
Fig. 3: GraphRAG and Vector-based RAG responses to psy-
chiatric history and progression.
social determinants, risk factors, protective factors, suicide-
theory constructs, and temporal events, thereby isolating the
effect of retrieval structure.
B. Qualitative Evaluations
Figure 3 illustrates the qualitative difference between
GraphRAG and RAG using a representative multi-hop query
about the psychiatric history of a high-risk patient from the
V ASPSD with PTSD, Generalized Anxiety Disorder (GAD),
schizophrenia, and chronic pain.
When asked to reconstruct the patient’s psychiatric history,
GraphRAG returns a structured chronology anchored in dis-
crete diagnostic events, including Major Depressive Disorder
(MDD) onset in 1993, psychotic features in 2003, schizophre-
nia in 2005, anxiety in 2007, psychotic depression in 2013,
and a relapsing course through 2021 with no documented
remission. The response organizes these events into a dated
sequence and summarizes the trajectory in terms of chronicity,
comorbidity, and ongoing pharmacologic management.
RAG returns no relevant information for the same query. Al-
though the relevant facts are present in the underlying record,
vector retrieval fails to assemble the longitudinal diagnostic
evidence to answer a temporally structured question.TABLE I: Overall mean absolute judge scores (1,500 queries,
100 patients, 15 tasks). Best score per criterion in bold.
SystemFactual
AccuracyComple-
tenessClinical
RelevanceEvidence
Grounding
RAG0.4240.348 0.1930.432
GraphRAG 0.2310.342 0.3590.200
This example highlights the advantage of ontology-
grounded retrieval. Reconstructing psychiatric history requires
linking diagnoses across time, distinguishing onset from re-
lapse, and tracing accumulating comorbidity. These relations
are explicitly encoded in the graph and recoverable through
traversal, whereas flat chunk retrieval ranks locally similar text
without a mechanism for assembling the full clinical trajectory.
C. Quantitative Evaluations
We evaluated the system responses using an LLM-as-judge
protocol. Each response is scored on four criteria using a
trinary scale (0, 0.5, 1):
•Factual Accuracy:whether clinical claims are correct
with respect to the patient record.
•Completeness:whether all relevant evidence in the pa-
tient record is captured.
•Clinical Relevance:whether the answer interprets evi-
dence clinically rather than listing raw facts.
•Evidence Grounding:whether specific claims are trace-
able to patient-record evidence.
We employ two evaluation modes. Inabsolutemode, each
system is scored independently against the structured patient
record. Incomparativemode, the judge receives both answers
simultaneously alongside each system’s retrieved evidence
(text chunks for RAG, graph nodes for GraphRAG), and scores
both before declaring a winner per criterion.
The distinction carries methodological weight. Under abso-
lute scoring, graph nodes retrieved by GraphRAG cannot be
verified against the flat patient record, so ontological constructs
such as risk windows and theoretical framework nodes are
penalised as ungrounded even when correctly retrieved. The
comparative judge corrects this by treating each system’s
retrieved evidence as the appropriate verification source: RAG
answers are checked against the patient record and retrieved
chunks, GraphRAG answers against the patient recordand
retrieved graph nodes. This correction is necessary for a fair
assessment of ontology-grounded retrieval and constitutes a
methodological contribution of this work.
Table I reports absolute scores. The systems split by cri-
terion: RAG leads on factual accuracy (0.424 vs. 0.231) and
evidence grounding (0.432 vs. 0.200), while GraphRAG leads
on clinical relevance (0.359 vs. 0.193), with near parity on
completeness. This reflects a structural asymmetry in the ab-
solute judge: claims are verified against the flat patient record,
so RAG answers derived from the same text fields receive
higher grounding scores, whereas GraphRAG answers citing
knowledge-graph constructs absent from the flat record are
penalized as ungrounded. Thus, the absolute results conflate
proximity to the source representation with answer quality.

TABLE II: Paired comparative evaluation (1,500 queries×4
criteria, minus skipped pairs).
Outcome Count / Rate
GraphRAG wins 4,482
RAG wins 238
Ties 1,144
Skipped 136
Total comparisons 5,864
GraphRAG win rate 76.4%
RAG win rate 4.1%
Tie rate 19.5%
TABLE III: GraphRAG win rate by criterion under the com-
parative judge.
Criterion GR Wins RAG Wins Ties GR Win Rate
Factual Accuracy 716 70 680 48.8%
Completeness 1,231 100 135 84.0%
Clinical Relevance 1,377 23 66 93.9%
Evidence Grounding 1,158 45 263 79.0%
The comparative judge mitigates this representation asym-
metry. Across 5,864 criterion-level comparisons in Table II,
GraphRAG is preferred in 4,482 cases (76.4%), compared with
238 cases (4.1%) for RAG and 19.5% ties. Under this fairer
protocol, ontology-grounded retrieval produces consistently
stronger clinical answers. Table III shows that GraphRAG
wins 93.9% of clinical relevance and 84.0% of complete-
ness comparisons, reflecting the ontology’s ability to surface
structured relational evidence in clinically interpretable form.
Evidence grounding reaches 79.0%, reversing the absolute
judge’s finding and confirming that RAG’s earlier advantage
was an attribution artifact. Factual accuracy shows the smallest
comparative advantage: GraphRAG wins 48.8% of compar-
isons, while 46.4% are ties and 4.8% favor RAG.
Table IV reports the task-level absolute scores and shows
that the two retrieval strategies have different strengths.
RAG generally achieves higher factual accuracy and evi-
dence grounding, while GraphRAG achieves higher clinical
relevance in 12 of 15 tasks, trailing only on medication,
stressor, and suicide indicators. The clearest exception is the
suicide-indicators task, where RAG obtains the highest scores
across all four criteria (0.795, 0.810, 0.840, and 0.800). This
reflects the structure of the V ASPSD records, where suicide-
related indicators often appear in compact clinical entries that
vector retrieval can recover intact. In contrast, GraphRAG
distributes the same information across multiple node types,
so missing any relevant node can reduce coverage. This is
the one category where flat text retrieval retains a genuine ad-
vantage. Ontology mapping and protective-versus-risk balance
remain challenging for both systems, suggesting that these
tasks require stronger reasoning over theoretical constructs and
competing clinical evidence beyond retrieval alone.
For paired task-level comparison, Table V reports
GraphRAG win rates per task. Win rates exceed 80% on seven
tasks: risk factors (91.5%), ontology (89.4%), comorbidity
(85.1%), risk construct inference (84.7%), protective versusrisk balance (84.5%), lethal means assessment (83.0%), and
summary (82.1%). Each of these requires multi-hop reason-
ing across heterogeneous node types: integrating diagnoses
with theoretical suicide constructs, which links conditions to
temporal risk windows, or synthesizing competing risk and
protective evidence. The BEACON-SP ontology encodes these
relations explicitly; vector similarity cannot reconstruct them
from chunk co-occurrence.
In areas where RAG is competitive such as suicide indica-
tors (51.3%), protective factors (61.0%), medication (62.0%),
and stressor identification (64.8%), the relevant clinical infor-
mation is largely self-contained in a small number of struc-
tured EHR fields and retrieved accurately by both systems. For
such tasks, the relational scaffolding of the knowledge graph
adds less marginal value.
D. Discussion
Two findings emerge from our evaluations. The first finding
is methodological. When claims are verified only against flat
patient records, GraphRAG appears weaker in factual accuracy
and evidence grounding. This is partly a measurement artifact:
graph-derived constructs may not appear explicitly in the
source text. The comparative judge reduces this asymmetry
by evaluating each system against its own retrieved evidence.
Under this protocol, GraphRAG achieves a 79.0% evidence-
grounding win rate, while factual accuracy is essentially tied at
48.8%, suggesting that flat-text ground truth can underestimate
graph-based retrieval.
The second finding is clinical. After correcting evidence
attribution, BEACON-SP GraphRAG wins 76.4% of 5,864
head-to-head criterion evaluations, with the largest gains in
clinical relevance (93.9%) and completeness (84.0%). These
improvements are concentrated in relational reasoning tasks,
including risk factor identification, comorbidity analysis, risk
construct inference, protective-versus-risk balance, and sum-
mary synthesis. Such tasks benefit from the explicit structure
encoded in the BEACON-SP ontology, which flat passage
retrieval cannot recover from semantic similarity alone.
Suicide indicators are the main exception. RAG performs
better under both protocols, likely because dense clinical-
note passages capture these indicators more completely than
distributed graph nodes. Similar patterns for protective factors,
medications, and stressors motivate a hybrid design in which
graph retrieval supports relational inference, while passage
retrieval supplies dense clinical documentation. Both systems
remain below 0.5 on protective-versus-risk balance, ontol-
ogy mapping, and summary tasks, indicating that high-level
synthesis requires stronger reasoning and grounding beyond
retrieval architecture alone.
VII. CONCLUSION
This paper presented BEACON-SP, an ontology-grounded
GraphRAG framework for clinician-facing suicide-prevention
decision support. BEACON-SP combines patient knowledge
graphs with ontology-guided retrieval to provide structured
reasoning over clinical, behavioral, social, and temporal ev-
idence. The framework uses a comprehensive suicide preven-
tion ontology that integrates three suicide-prevention theories

TABLE IV: Mean absolute judge scores per task (100 patients). Best score per RAG–GraphRAG pair bolded.
RAG GraphRAG
TaskFactual
AccuracyComple-
tenessClinical
RelevanceEvidence
GroundingFactual
AccuracyComple-
tenessClinical
RelevanceEvidence
Grounding
Comorbidity 0.060 0.070 0.000 0.0350.260 0.260 0.485 0.195
Diagnosis 0.315 0.310 0.320 0.2800.140 0.305 0.360 0.135
Lethal Means Assessment0.625 0.6200.1450.5200.425 0.3900.2200.255
Medication0.580 0.6200.5650.6400.240 0.3900.4950.210
Ontology 0.000 0.045 0.000 0.0000.030 0.155 0.005 0.010
Protective Factors0.8250.405 0.0250.7850.2100.375 0.1150.155
Protective vs. Risk0.3150.270 0.1850.4000.1050.270 0.4500.185
Risk Construct Inference0.465 0.4050.0300.3700.310 0.3450.4600.270
Risk Factors 0.065 0.140 0.010 0.0950.165 0.360 0.275 0.145
Risk Window Identification0.690 0.3850.1100.5300.255 0.2850.5150.185
Stressor0.475 0.440 0.375 0.7700.270 0.470 0.250 0.400
Substance Interaction0.6500.320 0.0550.7000.2950.380 0.3600.195
Suicide Indicators0.795 0.810 0.840 0.8000.465 0.525 0.530 0.405
Summary0.2700.150 0.1850.2950.1750.405 0.4100.120
Temporal0.2300.230 0.0450.2600.1250.215 0.4500.130
Overall 0.4240.348 0.1930.4320.2310.342 0.3590.200
TABLE V: GraphRAG paired win rate by task using the
comparative judge (100 patients).
Task GraphRAG Win Rate
Risk Factors 91.5%
Ontology 89.4%
Comorbidity 85.1%
Risk Construct Inference 84.7%
Protective vs. Risk Balance 84.5%
Lethal Means Assessment 83.0%
Summary 82.1%
Diagnosis 78.8%
Temporal 78.5%
Risk Window Identification 76.3%
Substance Interaction 74.2%
Stressor 64.8%
Medication 62.0%
Protective Factors 61.0%
Suicide Indicators 51.3%
Overall 76.4%
into a unified representation of diagnoses, medications, med-
ical conditions, risk and protective factors, interventions, life
events, suicide-behavior constructs, and temporal risk win-
dows. We evaluated BEACON-SP for clinician-facing question
answering using an ontology-guided GraphRAG workflow. On
a benchmark of 1,500 queries spanning 15 clinical categories
and 100 patients, GraphRAG was preferred in 76.4% of head-
to-head comparisons, with gains in clinical relevance (93.9%)
and completeness (84.0%). We also identified and corrected an
evaluation bias that penalizes graph-based systems for citing
ontological constructs absent from flat EHR records. The re-
sulting comparative protocol provides a more equitable assess-
ment of ontology-grounded retrieval systems. BEACON-SP
establishes a foundation for interpretable, evidence-grounded
clinical decision support. Future work will focus on larger
benchmarks, clinician review, and hybrid graph-text retrieval.REFERENCES
[1] S. Abuse and M. H. S. Administration, “Key substance use and mental
health indicators in the United States: Results from the 2024 national
survey on drug use and health,”HHS Publication No. PEP25-07-007,
NSDUH Series H-60, 2025.
[2] U.S. Department of Veterans Affairs, “2024 national veteran suicide
prevention annual report: Part 1 of 2—in-depth reviews,” Dec. 2024.
[3] J. Wuet al., “Medical Graph RAG: Towards safe medical large lan-
guage model via graph retrieval-augmented generation,”arXiv preprint
arXiv:2408.04187, 2024.
[4] R. Dattaet al., “Improving hospital risk prediction with knowledge-
augmented multimodal EHR modeling,”ArXiv, vol. abs/2508.01970,
2025.
[5] C. N. Hang, P.-D. Yu, and C. W. Tan, “TrumorGPT: Graph-based
retrieval-augmented large language model for fact-checking,”IEEE
Transactions on Artificial Intelligence, vol. 6, no. 11, pp. 3148–3162,
2025.
[6] Y . Guo and Y . Guo, “A knowledge graph and large language model-
based framework for depression detection,” in2024 International Con-
ference on Image Processing, Computer Vision and Machine Learning
(ICICML), pp. 670–673, 2024.
[7] F. R. Shafi and M. A. Hossain, “LLM-therapist: A RAG-based multi-
modal behavioral therapist as healthcare assistant,” in2024 IEEE Global
Communications Conference (GLOBECOM), pp. 2129–2134, 2024.
[8] N. Conger, N. Scollar, K. Davaslioglu, Y . E. Sagduyu, and S. Kompella,
“How to discover knowledge for FutureG: Contextual RAG and LLM
prompting for O-RAN,” in2026 IEEE 23rd Consumer Communications
and Networking Conference (CCNC), pp. 1–6, 2026.
[9] E. L. Meerwijket al., “Development of a 3-step theory of suicide
ontology to facilitate 3st factor extraction from clinical progress notes,”
Journal of biomedical informatics, vol. 150, p. 104582, 2024.
[10] R. C. O’Connor and O. J. Kirtley, “The integrated motivational-volitional
model of suicidal behaviour,”Philosophical Transactions of the Royal
Society B: Biological Sciences, vol. 373, July 2018.
[11] S. Wanget al., “An NLP approach to identify SDoH-related circum-
stance and suicide crisis from death investigation narratives,”Journal of
the American Medical Informatics Association, vol. 30, no. 8, pp. 1408–
1417, 2023.
[12] H. Kimet al., “Developing a physical activity ontology to support the
interoperability of physical activity data,”Journal of medical Internet
research, vol. 21, no. 4, p. e12776, 2019.
[13] F. Liet al., “Time event ontology (TEO): to support semantic represen-
tation and reasoning of complex temporal relations of clinical events,”
Journal of the American Medical Informatics Association, vol. 27, no. 7,
pp. 1046–1056, 2020.
[14] D. of Veterans Affairs, “Veterans affairs suicide prevention synthetic
dataset,”VHA Open Data, April 2026.