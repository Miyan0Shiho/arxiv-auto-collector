# Clinical Knowledge Graph Construction and Evaluation with Multi-LLMs via Retrieval-Augmented Generation

**Authors**: Udiptaman Das, Krishnasai B. Atmakuri, Duy Ho, Chi Lee, Yugyung Lee

**Published**: 2026-01-05 07:16:29

**PDF URL**: [https://arxiv.org/pdf/2601.01844v1](https://arxiv.org/pdf/2601.01844v1)

## Abstract
Large language models (LLMs) offer new opportunities for constructing knowledge graphs (KGs) from unstructured clinical narratives. However, existing approaches often rely on structured inputs and lack robust validation of factual accuracy and semantic consistency, limitations that are especially problematic in oncology. We introduce an end-to-end framework for clinical KG construction and evaluation directly from free text using multi-agent prompting and a schema-constrained Retrieval-Augmented Generation (KG-RAG) strategy. Our pipeline integrates (1) prompt-driven entity, attribute, and relation extraction; (2) entropy-based uncertainty scoring; (3) ontology-aligned RDF/OWL schema generation; and (4) multi-LLM consensus validation for hallucination detection and semantic refinement. Beyond static graph construction, the framework supports continuous refinement and self-supervised evaluation, enabling iterative improvement of graph quality. Applied to two oncology cohorts (PDAC and BRCA), our method produces interpretable, SPARQL-compatible, and clinically grounded knowledge graphs without relying on gold-standard annotations. Experimental results demonstrate consistent gains in precision, relevance, and ontology compliance over baseline methods.

## Full Text


<!-- PDF content starts -->

Clinical Knowledge Graph Construction and Evaluation
with Multi-LLMs via Retrieval-Augmented Generation
Udiptaman Das1Krishnasai B. Atmakuri1Duy Ho2Chi Lee1
Yugyung Lee1
1University of Missouri–Kansas City, USA
2California State University, Fullerton, USA
{ud3d4, bka2bg}@umkc.edu duyho@fullerton.edu, {leech,leeyu}@umkc.edu
Abstract
Large language models (LLMs) offer new opportuni-
ties for constructing knowledge graphs (KGs) from
unstructured clinical narratives. However, existing
approaches often rely on structured inputs and lack
robust validation of factual accuracy and seman-
tic consistency—limitations that are especially prob-
lematic in oncology. We introduce an end-to-end
framework for clinical KG construction and evalua-
tion from free text using multi-agent prompting and
a schema-constrained Retrieval-Augmented Genera-
tion (KG-RAG) strategy. Our pipeline integrates:
(1) prompt-driven entity, attribute, and relation ex-
traction; (2) entropy-based uncertainty scoring; (3)
ontology-aligned RDF/OWL schema generation; and
(4) multi-LLM consensus validation for hallucina-
tion detection and semantic refinement. Beyond
static construction, the framework supports contin-
uous refinement and self-supervised evaluation to it-
eratively improve graph quality. Applied to two
oncology cohorts (PDAC and BRCA), our method
produces interpretable, SPARQL-compatible, and
clinically grounded knowledge graphs without gold-
standard annotations, achieving consistent gains in
precision, relevance, and ontology compliance over
baseline methods.
1 Introduction
Constructing accurate and clinically relevant knowl-
edge graphs (KGs) from unstructured medical narra-
tives is a foundational challenge in biomedical infor-
matics. Clinical KGs enable explainable AI, decision
support,andlongitudinalpatientmodeling,yettradi-
tional approaches remain limited. Rigid schemas like
FHIR often lack semantic flexibility [1], and while on-
tologies such as SNOMED CT [15], LOINC [12], and
RxNorm[17]offerstandardterminologies,theystrug-gle to capture the temporal, contextual, and inferen-
tial nuances needed for oncology and other complex
domains.
Conventional rule-based or manual KG construc-
tion pipelines are brittle, difficult to scale, and ill-
suited to the evolving language and structure of clin-
ical narratives. In contrast, large language models
(LLMs) have emerged as powerful tools for semantic
parsing, relation discovery, and context-aware gener-
ation. Models such asGemini 2.0 Flash[3],GPT-4o
[10], andGrok 3[18] show promise for automating
KG construction directly from text. However, their
outputs are prone to hallucinations, semantic drift,
andfactualinconsistency—issuesespeciallycriticalin
high-stakes domains like oncology [11, 2].
Recent works such asCancerKG.ORG[4],
EMERGE[23], andCLR2G[19] explore LLM inte-
gration with structured or multimodal sources, but
lack generalizable pipelines for KG construction and
validation directly from clinical text.
This paper introduces the first end-to-end frame-
work for constructing and evaluating clinical knowl-
edge graphs from free-text using multi-agent prompt-
ing and a graph-based Retrieval-Augmented Gener-
ation (KG-RAG) approach. Our pipeline supports
continuous refinement and self-supervised evaluation,
enabling both high-precision construction and dy-
namic graph improvement over time.
We leverage a multi-agent LLM pipeline combining
the complementary strengths of:
•Gemini 2.0 Flashfor schema-guided En-
tity–Attribute–Value (EAV) extraction;
•GPT-4ofor contextual enrichment, ontology
alignment, and reflection-based refinement;
•Grok 3for validation through contradiction test-
ing and conservative filtering.
1arXiv:2601.01844v1  [cs.AI]  5 Jan 2026

All triples are mapped to SNOMED CT,
LOINC, RxNorm, GO, and ICD, and encoded in
RDF/RDFS/OWL for semantic reasoning. Trust
metrics are derived from model agreement and align-
ment with biomedical ontologies. We demonstrate
our method on 40 clinical oncology reports from the
CORAL dataset [16], spanning PDAC and BRCA co-
horts, and evaluate triple correctness, ontology cov-
erage, relation diversity, and graph connectivity.
Our key contributions are as follows.
•We propose a KG-RAG framework that inte-
grates multi-agent prompting, LLM-based re-
finement, and continuous evaluation without
gold-standard labels.
•We introduce a schema-constrained, FHIR-
aligned EAV extraction module paired with on-
tology mapping and OWL-based encoding.
•We support inference over implicit, cross-
sentence, and multi-attribute relations with ro-
bust semantic validation.
•We encode the final graphs with semantic web
standards and introduce composite trust scoring
mechanisms.
•We empirically validate our system on oncology
narratives using metrics spanning factuality, se-
mantic grounding, and relational completeness.
Byunitingmulti-LLMconsensus, ontologyground-
ing, and iterative refinement, our system offers a scal-
able, explainable, and verifiable approach for clinical
KG construction—laying the foundation for real-time
decision support and next-generation clinical AI.
2 Related Work
Recent advancements in hybrid knowledge
graph–languagemodel(KG–LLM)systems, retrieval-
augmented generation (RAG), and contrastive learn-
ing have significantly influenced clinical informatics.
However, many existing approaches rely on prede-
fined schemas, curated corpora, or multimodal data.
Our work addresses a critical gap: enabling schema-
inductive, verifiable knowledge graph construction
directly from unstructured clinical narratives.
Schema-Constrained KG–LLM Hybrids
CancerKG.ORG[4] integrates large language mod-
els (LLMs) with curated colorectal cancer knowl-
edge using models such as GPT-4, LLaMA-2,
and FLAN-T5, alongside structured meta-profilesand guardrailed RAG. Other frameworks like
KnowGL[9],KnowGPT[21], andKALA[8] demon-
strate the efficacy of knowledge graph alignment but
depend on static ontologies. In contrast, our system
facilitates dynamic schema evolution through auto-
maticentity–attribute–value(EAV)extraction, prob-
abilistic triple scoring, and iterative ontology ground-
ing—eliminating the need for manual curation or
fixed schemas.
Multimodal RAG and Predictive Systems
Systems such asEMERGE[23],MedRAG[22], and
GatorTron-RAG[20] combine clinical notes, struc-
tured electronic health record (EHR) data, and
biomedical graphs for predictive tasks. While effec-
tive for diagnosis or risk stratification, these models
focus on classification rather than symbolic knowl-
edge representation. Our approach emphasizes inter-
pretable, symbolic triple construction using a fully
text-based pipeline—removing dependencies on ex-
ternal modalities or fixed task objectives.
ContrastiveandTrust-AwareLLMGeneration
Trust-aware LLM generation has been explored
through methods likeReflexion[13] andTruth-
fulQA[7], which employ self-reflection and halluci-
nation mitigation techniques. Similarly,CLR2G[19]
applies cross-modal contrastive learning for radiology
report generation. While these approaches enhance
semantic control, our framework advances this by in-
troducing multi-agent triple scoring, entropy-aware
trust filtering, and reflection-based validation across
LLMs such as Gemini, GPT-4o, and Grok—focusing
specifically on symbolic reasoning from clinical text.
Structured Retrieval and Table-Centric
Knowledge Graphs
Earlier platforms includingWebLens[6],Hy-
brid.JSON[14], andCOVIDKG[5] introduced scal-
able structural retrieval and metadata modeling for
heterogeneous clinical data tables.CancerKGex-
tends this direction using vertical and horizontal
metadata modeling over structured cancer databases.
Our work fundamentally differs in modality and
scope: rather than relying on table alignment or at-
tribute normalization, we construct evolving clinical
knowledge graphs directly from free-text reports us-
ing prompt-based decomposition, probabilistic scor-
ing, and ontology-backed reasoning.
In summary, while existing systems have laid es-
sential groundwork in structured knowledge extrac-
tion and LLM-KG integration, our contribution is
2

a self-evaluating, schema-flexible pipeline that au-
tonomously discovers, validates, and encodes clini-
cal knowledge from raw narratives. By removing
rigid schema dependencies and leveraging multi-LLM
agreement for factual trust, we enable scalable, in-
terpretable, and clinically relevant knowledge graph
constructionthatdynamicallyadaptstonewdomains
and document types.
3 Methodology
3.1 Overview of Multi-Agent KG
Construction and Evaluation
Pipeline
We propose a multi-agent framework for construct-
ing clinically verifiable and semantically interopera-
ble knowledge graphs (KGs) directly from oncology
narratives. The system orchestrates three state-of-
the-art large language models—Gemini 2.0 Flash,
GPT-4.o, andGrok 3—each assigned a special-
ized role across five modular stages. This architec-
ture is designed to produce structured, explainable,
andontology-alignedknowledgegraphsfromunstruc-
tured clinical text.
1.EAV Extraction (Gemini 2.0 Flash):Us-
ing tailored prompts and FHIR-aware templates,
Gemini 2.0 Flashperforms extraction ofEn-
tity–Attribute–Value(EAV) triples from free-
text narratives. Extracted entities are linked to
canonical clinical resource types.
2.Ontology Mapping (Gemini 2.0 Flash):
Attributes and values are mapped to standard
biomedical vocabularies such asSNOMED CT,
LOINC, andRxNorm. This step normalizes
terminology and facilitates semantic consistency
across clinical concepts.
3.Relation Discovery (Gemini 2.0 Flash +
GPT-4.o):Semantic relationships between
entities and attributes are identified through
prompt-driven extraction and refinement.Gem-
ini 2.0 Flashgenerates candidate links, while
GPT-4.orefines and validates relation types
using contextual understanding and ontology
alignment.
4.Semantic Web Encoding
(RDF/RDFS/OWL):All triples are en-
coded using Semantic Web standards such as
RDF, RDFS, and OWL. This enables sym-
bolic reasoning, knowledge graph queryingvia SPARQL, and integration with external
ontology-based systems.
5.KG Validation (Gemini 2.0 Flash + GPT-
4.o + Grok 3):A composite trust function is
applied to evaluate the reliability of each triple.
The trust score incorporates:
•Self-consistency from Gemini 2.0 Flash(via
re-prompting),
•Semantic grounding from GPT-4.o(evi-
dence retrieval and consistency checks),
•Robustness verification from Grok 3(coun-
terfactual and adversarial assessment).
This pipeline supports schema-flexible, ontology-
informed, and model-validated knowledge graph con-
struction from raw clinical narratives. The modular
architecture facilitates downstream reasoning, query-
ing, and integration into clinical decision support sys-
tems while maintaining traceability and explainabil-
ity of the extracted information.
3.2 Stage 1: FHIR-Guided EAV Ex-
traction
The first stage of our pipeline focuses on extract-
ing structured clinical knowledge in the form ofEn-
tity–Attribute–Value (EAV)triples. This is achieved
using schema-guided prompting withGemini 2.0
Flash, integrating syntactic cues from the narra-
tive and semantic constraints from theFHIR (Fast
Healthcare Interoperability Resources)specification.
Entities are concurrently typed using FHIR to ensure
semantic consistency and interoperability.
Triple Extraction.LetD={x 1, x2, . . . , x N}be
the corpus of clinical narratives. For eachx i∈ D,
we construct a structured promptπ(x i)tailored for
Gemini 2.0 Flash, which generates a candidate set of
EAV triples as:
Ti=fθ(π(x i)) ={(e j, aj, vj)}k
j=1,∀i∈[1, N]
Each triple(e j, aj, vj)includes:e j∈ EFHIR: a
FHIR-typed entity (e.g.,Procedure,Observation),
aj: a clinical attribute, andv j: a narrative-grounded
value.
FHIR Typing Function.Entities are normalized
via a typing functionϕ FHIR :E → E FHIR, ensuring
alignment to standard resource types for downstream
mapping.
3

Entropy-Based Value Confidence.To evaluate
confidence in value predictions, we compute token-
level entropy forv jgiven its sub-token distribution
P(vj) ={p 1, . . . , p m}:
H(v j) =−mX
t=1ptlogp t
Values withH(v j)> δ(thresholdδ) are flagged for
further validation or multi-model filtering.
Illustrative EAV Triples.Examples include:
(Procedure, performed_by,
SurgicalOncologist)
(Observation, hasLabResult, CA 19-9)
(HER2 Status, determines, Trastuzumab
Eligibility)
These EAVs serve as the foundation for ontology
mapping, relation discovery, and semantic web en-
coding in the subsequent stages.
3.3 Stage 2: Ontology Mapping &
Schema Construction
To enable semantic reasoning and interoperability,
extracted EAV concepts are mapped to standard-
ized biomedical ontologies via LLM-guided retrieval
and similarity alignment.Gemini 2.0 Flashorches-
trates this step and produces OWL/RDFS-compliant
schemas for graph construction.
Ontology Vocabulary.We define the ontology
set:
O={SNOMED CT,LOINC,RxNorm,ICD,GO}
EachO icontributes domain-specific concepts,
e.g., SNOMED:Weight Loss, LOINC:CA 19-9,
RxNorm:FOLFIRINOX.
Concept Mapping.Given raw termsC raw =
{c1, . . . , c M}, define a mapping:
µ:Craw→ Cmapped ⊆[
iOi
Scoring uses lexical and semantic similarity:
Score(c i, oj) =α·sim lex+β·sim sem, α+β= 1
Schema Construction.Mapped concepts are en-
coded in OWL/RDFS:
•Class Typing:Ifo j∈ OSNOMED, declareo jas
an OWLClass.•Property Semantics:
hasLabResult⊑ObjectProperty,
domain(hasLabResult)=Observation,
range(hasLabResult)=LabTest
•TBox Inclusion:ElevatedCA19_9⊑
AbnormalTumorMarker.
Persistent URIs.Each concept receives a resolv-
able URI, e.g.,http://snomed.info/id/267036007
for “Weight Loss.”
Outcome.This stage yields an ontology-aligned
schema that supports OWL reasoning, SPARQL
queries, and Linked Data integration, while ensuring
formal consistency and semantic traceability.
3.4 Stage 3: Relation Discovery
To move beyond isolated EAV triples, this stage
enriches the knowledge graph with typed relations
capturing diagnostic reasoning, temporal dependen-
cies, and treatment logic. Structured prompting and
multi-agent validation (Gemini 2.0 Flash, GPT-4.o,
Grok 3) are used to discover and filter semantic rela-
tions.
Relation Typing.Each relationr i∈ Ris classi-
fied as:
ri∈

REE(Entity–Entity)
REA(Entity–Attribute)
RAA(Attribute–Attribute)
Examples include:R EE:Biopsy→confirms
→TumorType,R EA:CT Scan→visualizes
→Pancreatic Mass,R AA:HER2 Status→
determines→Trastuzumab Eligibility.
Candidate Generation.Given narrativex, can-
didate relationsT rel={(h i, pi, ti)}are generated via
Gemini:
Tgen
rel=fGemini (πrel(x))
withh i, ti∈ E ∪ Aandp i∈ Vverb.
Semantic Validation.Each tripleτ iis scored by
GPT-4.o using contextual inference:
J(τi) =fGPT-4.o (πjudge(τi, x))∈[0,1]
Adversarial Filtering.Grok 3 perturbs each re-
lation and flags contradictions:
ξ(τi) =|{τ′
i∈ A(τ i)|contradictory}|
|A(τ i)|
4

Final Set.The accepted relation set filters for
high plausibility and low contradiction:
Ttrusted
rel ={τ i|J(τ i)> δ, ξ(τ i)≤ϵ}
Outcome.This step yields a validated set of
typedrelationsinterlinkingFHIR-groundedconcepts.
These enhance the inferential depth of the KG,
supporting diagnostic, prognostic, and treatment-
oriented reasoning.
3.5 Stage 4: Semantic Graph Encod-
ing
To enable reasoning and system interoperability, val-
idated triplesτ i= (s i, pi, oi)are encoded using Se-
mantic Web standards: RDF, RDFS, OWL, and
SWRL.
RDF Triples.Each assertion is modeled as:
(si, pi, oi)∈ GRDF, wheres i, oi∈ U ∪ Landp i∈ U;
Udenotes URIs andLliterals.
RDFS Constraints.Predicates include do-
main/range typing: domain(p i) =C s, range(p i) =
Co, with type assertions:rdf:type(s i, Cs)∧
rdf:type(o i, Co).
OWL Semantics.Logical rules include: Sub-
class:A⊑B, Equivalence:A≡B, Re-
striction:Biopsy⊓ ∃hasOutcome.Malignant⊑
PositiveFinding.
SPARQL Query.Query for high Ki-67 index:
PREFIX kg: <http://example.org/kg#>
SELECT ?p WHERE {
?p kg:hasAttribute ?a .
?a rdf:type kg:Ki67_Index .
?a kg:indicates ?v .
FILTER(?v > 20)
}
SWRL Rule.Example rule for identifying high-
risk PDAC patients:
Patient(p)∧hasAttribute(p, ca)∧CA19_9(ca)∧
indicates(ca, v 1)∧greaterThan(v 1,1000)∧
hasAttribute(p, w)∧WeightLoss(w)∧
indicates(w, v 2)∧greaterThan(v 2,10)
⇒HighRiskPatient(p)
Outcome.This stage produces an ontology-
compliant semantic graph supporting RDFS valida-
tion, OWL/SWRL inference, and SPARQL query-
ing—ideal for clinical integration and semantic an-
alytics.3.6 Stage 5: Multi-LLM Trust Valida-
tion
To ensure the reliability and semantic precision of the
constructedclinicalknowledgegraph,weimplementa
multi-layered validation framework. This framework
consists of two complementary tiers: (1) evaluating
the fidelity of Entity–Attribute–Value (EAV) triples
and (2) verifying semantic relation triples. Both tiers
are supported by a multi-agent setup involvingGem-
ini 2.0 Flash,GPT-4.o, andGrok 3, which collabora-
tively perform grounding verification, reflective rea-
soning, and adversarial testing.
EAV Validation: Grounding and Fidelity As-
sessment
Each EAV triple(e, a, v)is validated across multi-
ple quality dimensions, including textual grounding,
correctness, hallucination risk, recoverability, and
prompt-based consistency.
EAV Validation Metrics.
•Coverage:Whether the attributeaand valuev
are directly supported by the source text.
•Correctness Rate (CR):Proportion of grounded
EAV triples verified as correct.
•Hallucination Rate (HR):Fraction of generated
triples lacking explicit textual support.
•Rescue Rate (RR):Proportion of hallucinated
triples that become valid after normalization or
lexico-semantic heuristics.
•Self-Consistency:Agreement across multiple
prompting strategies applied to the same source.
Relation Validation: Semantic Integrity and
Clinical Utility
Each relation triple(s, p, o)is evaluated on criteria
that assess its semantic validity, ontological align-
ment, clinical relevance, and informativeness within
the graph.
Relation Evaluation Criteria.
•Semantic Validity:Verified via entailment and
alignment with external evidence or ontologies.
•Schema Compliance:Ensures each predicate re-
spects RDF domain and range constraints.
•Clinical Usefulness:Prioritizes relations with di-
agnostic, prognostic, or therapeutic significance.
5

Table 1: Three-Stage Matching Pipeline for EAV Triple Validation
Technique Formal Rule Illustrative Example
Stage 1: Baseline Matching Techniques
Case-Sensitive Matching vi∈ Texact(dj) “FOLFIRINOX” matched exactly in raw text
Regex Matching regex("1 ˙2 ?mg/dL") bilirubin: 1.2 mg/dLmatched as value
Fuzzy String Matching fuzz.ratio(v i, tk)> τ “FOLFRINOX”→“FOLFIRINOX”
N-Gram Phrase Matching ∃g⊂d j:sim(g, v i)> γ “2–3 episodes/day” matched to vomiting frequency
Boolean Inference “denies smoking”⇒smoking: false Negation mapped to absent finding
Stage 2: Heuristic Augmentation
Case-Insensitive Matching lower(v_i) = lower(t_k) “Male” == “male” == “MALE”
Custom Negation Detec-
tion“denies chills”→
observation_chills: absentHandcrafted negation patterns
spaCy Lemmatization lemma(a i) =t k “diaphoretic”→“diaphoresis”
Synonym Mapping ai∈Σ syn “dyspneic”↔“dyspnea”
Stage 3: Specialized Resolution
Sentence-Level Negation dep(t k) =neg in sentence(v i) “denies pruritus”→absent
Typo Correction fuzz.ratio(deneid,denied)>95 Handles transcription errors
Explicit Fixes “smking”→“smoking” Dictionary-based recovery of high-impact terms
Table 2: Relation-Level Validation Techniques and Examples
Technique Formal Rule or Criterion Illustrative Example
Evidence Alignment (En-
tailment)Sim entail (τ, ek)fore k∈ Eτ “HER2 is overexpressed” entailsHER2→determines→
Trastuzumab Eligibility
LLM Reflective Scoring J(τ)∈[0,1]from GPT-4o verifier
promptGPT-4o returns 0.92 plausibility for therapy eligibility
triple
Self-Consistency across
PromptsC(τ) =1
nPI[τ∈ T(i)] Relation appears in 4/5 prompt variants→C= 0.80
Domain-Range Schema
Checktype(s)∈domain(p); type(o)∈
range(p)Tumor→hasMarker→HER2invalid if HER2 is not range
of hasMarker
Redundancy Score cos(τ i, τj)> γin embedding space “confirms” and “verifies” flagged as duplicates
Semantic Clustering for
GapsUseUMAPclusteringoverembeddings Reveals that “initiated” and “started” lack edge alignment
•Structural Role and Frequency:Measures rela-
tional salience based on graph centrality and re-
currence.
•Redundancy and Gaps:Detects paraphrased or
missing links using embedding-based similarity.
•Multi-LLM Agreement:Confidence derived from
consensusacrossGemini 2.0,GPT-4.o,andGrok
3outputs.
Unified Trust Scoring
AcompositetrustscoreT(τ)isassignedtoeachtriple
τ= (s, p, o)or(e, a, v):
T(τ) =λ 1R(τ) +λ 2C(τ) +λ 3J(τ),X
λi= 1
Where:
•R(τ): Evidence alignment score from entailment
or retrieval.
•C(τ): Self-consistency across prompt variants.
•J(τ): Reflective plausibility score from model-
based judgment.
Only triples satisfyingT(τ)≥δ T(e.g.,δ T= 0.65)
are retained in the final knowledge graph.Final Graph Validation and Filtering
The final graphG finalis further validated through
graph-level filters:
•Ontology Compliance:Ensures that all relations
adhere to predefined schema constraints.
•Redundancy Elimination:Removes duplicate or
semantically equivalent edges.
•Clinical Coverage:Emphasizes inclusion of med-
ically relevant concepts (e.g., NCCN-aligned
markers, therapeutic eligibility).
Outcome.This multi-agent, multi-criteria valida-
tion framework ensures that the final knowledge
graph is clinically grounded, semantically coherent,
and structurally optimized. The result is a trust-
worthy KG suitable for deployment in applications
such as explainable AI, cohort identification, treat-
ment planning, and clinical decision support.
6

Figure 1: PDAC vs. BRCA Knowledge Graph Metrics by EAV Statistics, Ontology Mapping, Predicate
Typing, and Graph Structure. BRCA emphasizes molecular attributes; PDAC reflects procedural diversity.
Table 3: Consolidated knowledge graph metrics with examples (PDAC vs. BRCA).
Metric Category Metric PDAC BRCA Example / Description
EAV StatisticsTotal EAV triples 2,062 2,293 Structured triples such as(Observation, hasLabResult,
CA_19-9).
Entity instances 209 195 Entity types likeProcedure,Observation,Condition.
Unique attributes 1,172 1,353 Fine-grained attributes likeWeight_Loss,HER2_Status.
Ontology MappingMapped attributes 1,738 1,873 Attributes linked to SNOMED CT, RxNorm, and LOINC.
RxNorm terms 668 562FOLFIRINOX,Tamoxifen, chemotherapy agents.
GO terms 409 533 Genomic markers likeHER2,BRCA1,EGFR.
Unmapped rate 0.69% 0.85% Mostly generic strings, typos, or shorthand entries.
Predicate TypingEntity predicates 56 46 Action relations:(Biopsy, confirms, TumorType).
Attribute predicates 72 69 Diagnostic relations:(HER2Status, determines,
TherapyEligibility).
Total predicate instances 721 721 Total predicate uses across the graph.
Graph StructureTotal RDF triples 18,097 18,732 RDF-format atomic facts encoding EAV and relations.
Unique predicates 1,346 1,520 Vocabulary such ashasLabResult,performedBy.
Avg. node degree 5.73 5.74 Graph density and information connectivity.
Inconsistent entities 25 41 Domain–range issues caught during OWL validation.
4 Results and Evaluation
4.1 Experimental Setup
We evaluate our clinical knowledge graph (KG)
framework using 40 expert-annotated oncology re-
ports from theCORAL (Clinical Oncology Reports
to Advance Language Models)dataset. The reports
span two cancer types—Pancreatic Ductal Adenocar-
cinoma (PDAC)andBreast Cancer (BRCA)—which
differ significantly in diagnostic focus and treatment
documentation.
Our primary objectives are to assess the ground-
ing and accuracy of extractedEntity–Attribute–Value
(EAV)triples and to evaluate the structural and se-
mantic integrity of the resulting knowledge graphs.
The KG construction pipeline integrates four core
components. First,Gemini 2.0 Flashis used to ex-
tract FHIR-aligned EAV triples from unstructurednarratives. Second,GPT-4.o,Grok 3, and Gem-
ini collaborate to validate triples—filtering halluci-
nations and resolving inconsistencies. Third, ex-
tracted terms are normalized to standardized vocab-
ularies, including SNOMED CT, RxNorm, LOINC,
GO, and ICD. Finally, validated triples are encoded
into RDF/RDFS/OWL to support SPARQL and
SWRL-based rule reasoning.
Each of the 40 clinical reports—20 PDAC and
20 BRCA—contains a narrative file (.txt), ex-
pert annotations (.ann), and model-generated out-
puts (.json). The average report is approximately
145,000 tokens, with some extending to 180,000 to-
kens in more complex cases. The three LLMs play
distinct but complementary roles.Gemini 2.0 Flash
acts as the primary EAV extractor, aligning outputs
to FHIR standards and providing confidence scores.
GPT-4.ovalidates predicate plausibility and assigns
7

reflection-based trust levels.Grok 3performs ad-
versarial testing to detect redundancy and seman-
tic inconsistencies. This multi-agent framework en-
ables robust, interpretable, and ontology-compliant
knowledge graph construction directly from raw clin-
ical narratives.
4.2 Knowledge Graph Construction
Results
Figure 1 and Table 3 summarize the semantic and
structural features of knowledge graphs constructed
from PDAC and BRCA oncology reports. The vi-
sual and tabular comparisons reflect all key pipeline
stages—EAV extraction, ontology mapping, predi-
cate typing, and semantic modeling—and highlight
domain-specific characteristics across cancer types.
Together, they demonstrate the fidelity, flexibility,
and interpretability of our multi-agent LLM-based
KG construction approach.
Step 1: FHIR-Aligned EAV Extraction.BRCA
records yielded more EAV triples (2,293 vs. 2,062)
and a broader attribute set (1,353 vs. 1,172), reflect-
ing molecular precision. Examples:
•HER2 Status→determines→Trastuzumab
Eligibility
•Ki-67 Index→indicates→High
Proliferation Rate
PDAC emphasized diagnostic and procedural con-
cepts:
•CT Scan→visualizes→Pancreatic Mass
•Surgical Resection→treats→Primary
Tumor
Step 2: Ontology Mapping.BRCA had more
mapped attributes (1,873 vs. 1,738) and GO terms
(533 vs. 409), consistent with genomics-rich narra-
tives. PDAC showed stronger RxNorm alignment
(668 vs. 562), reflecting treatment-oriented notes.
Both cohorts had <1% unmapped rate.
Step 3: Predicate Typing.Both KGs included
721 predicate instances, with PDAC using a broader
predicate range (56 entity-level, 72 attribute-level)
than BRCA (46, 69). PDAC predicates emphasized
procedural workflows, BRCA prioritized stratifica-
tion and therapeutic relevance.
Step 4: Graph Structure and Semantic Mod-
eling.Both graphs showed dense connectivity (avg.
node degree∼5.7) and rich vocabularies (PDAC:
1,346 predicates; BRCA: 1,520). OWL validation
foundmoreinconsistenciesinBRCA(41)thanPDAC(25), often from genomics-based domain-range mis-
matches. TripleswereencodedinRDF/RDFS/OWL,
supporting SPARQL/SWRL:
Biopsy⊓hasOutcome.Malignant⊑PositiveFinding
Clinical Interpretability.Actionable triples
like(Biopsy, confirms, TumorType)and
(HER2 Status, determines, Trastuzumab
Eligibility)demonstrate support for clinical
decision-making and cohort stratification. Our
framework—Gemini 2.0 Flash for generation,
GPT-4o for validation, Grok 3 for semantic ro-
bustness—transforms unstructured narratives into
queryable, ontology-aligned knowledge graphs.
4.3 Evaluation of Knowledge Graphs
We evaluate the quality of LLM-generated clinical
knowledge graphs (KGs) using a two-tiered frame-
work: (1) analysis of Entity–Attribute–Value (EAV)
triplesfortextualandsemanticfidelity, and(2)struc-
tural and relation-level validation of the graph it-
self. Evaluation spans two cancer cohorts—PDAC
and BRCA—derived from the CORAL dataset.
4.3.1 Entity–Attribute–Value(EAV)Evalua-
tion
Evaluation of EAV Extraction Across PDAC
and BRCA Cohorts
We conducted a comparative evaluation of entity-
attribute-value (EAV) extraction across two cancer
cohorts: Pancreatic Ductal Adenocarcinoma (PDAC,
Patients 0–19) and Breast Cancer (BRCA, Patients
20–39). Key metrics include raw text and attribute
coverage, correctness, hallucinations, and error rates,
summarized in Table 4.
BRCA patients showed higher average raw text
coverage (37.97%) than PDAC (29.74%), suggesting
denser or more structured narratives in breast can-
cer reports. PDAC achieved slightly better extrac-
tion precision, with a lower incorrect attribute rate
(25.45 vs. 28.25) and a smaller overall error rate
(25.10% vs. 26.91%). Both cohorts demonstrated
excellent attribute coverage (99.83%), indicating con-
sistent LLM performance across cancer types. Hallu-
cination rates were minimal—fewer than one per five
patients—validating the reliability of prompt-based
extraction.
These results highlight a trade-off between recall
and precision. BRCA’s richer documentation may
increase coverage but introduces more noise, while
8

Table 4: EAV Quality Metrics Across Cancer Cohorts
MetricPDAC (P# 0–19) BRCA (P# 20–39)
Avg / Rate TotalAvg / Rate Total
Text & Attribute Coverage
Raw Text Coverage (%) 29.74% – 37.97% –
Attribute Coverage (%) 99.83% – 99.83% –
Total Attributes Extracted – 2,028 – 2,099
Correctness & Error Metrics
Correct Attributes (%) 73.22% 1,514 72.58% 1,524
Incorrect Attr. (Error#) 25.45 509 28.25 565
Errors (Attr.) (%) 25.10% – 26.91% –
Hallucination & Consistency
Hallucinated Attr. (Total) 0.20 4 0.15 3
PDAC yields fewer attributes with higher correct-
ness. This underscores the need for cohort-specific
fine-tuning or filtering in future extraction pipelines.
Attribute-Level Comparison
Figure 2 presents a comparison of raw coverage, cor-
rectness, and error rates across PDAC and BRCA co-
horts. PDAC patients show stable correctness with
fewer attribute-level errors, whereas BRCA patients
demonstrate greater variability, most notably among
Patients 30 and 31, who exhibit high attribute vol-
ume but reduced correctness.
Both cohorts demonstrate strong EAV extraction.
PDAC emphasizes higher correctness and reduced
noise, while BRCA contributes richer attribute di-
versity. These patterns highlight the importance of
tailoring extraction strategies to cohort-specific nar-
rative styles, especially for oncology KGs and adap-
tive clinical decision support.
4.3.2 KG Structural and Relation-Level Val-
idation
Multi-LLM Consensus Evaluation of EAV
Triples
To ensure semantic fidelity and factual grounding
of extracted Entity–Attribute–Value (EAV) triples,
we adopt a consensus-driven validation framework
using three complementary LLMs:Gemini 2.0
Flash(schema-aware extractor),Grok 3(evidence-
basedvalidator), andGPT-4o(semanticgeneralizer).
Triples are assessed across three dimensions: (1)Fac-
tuality—explicit grounding in source text or biomed-
ical ontologies, (2)Plausibility—semantically infer-
ablebutimplicittriples, and(3)Correction—revision
of ambiguous or hallucinated content.
Each model contributes distinct strengths:
Grok filters unsupported relations (e.g.,
hematuria_etiology→diagnosis); GPT-4o
proposes contextually inferred alternatives (e.g.,creatinine→kidney_function); Gemini pro-
vides precise, schema-aligned outputs.
Figure 3 shows Gemini achieves the highest fac-
tual accuracy (98–100%), followed by Grok (94–96%)
and GPT-4o (85–92%). Pearson correlations indi-
cate strong alignment between Gemini and Grok
(r= 0.88), with GPT-4o moderately correlated with
Gemini (r= 0.71) and Grok (r= 0.69), consistent
with its broader semantic scope.
In the absence of gold-standard annotations, we
accept triples validated by at least two mod-
els. Disagreements are resolved by prioritizing
ontology-compliant alternatives (e.g., SNOMED CT,
RxNorm), with uncertain triples flagged for reflection
andreranking. Together, theprecisionofGemini, the
filteringstrengthofGrok, andthecontextualbreadth
of GPT-4o enable robust, unsupervised validation of
clinical KGs. Correlation trends reinforce their com-
plementary roles in accurate, ontology-aligned graph
construction—without the need for human-labeled
supervision.
Statistical Analysis of Relation Diversity and
Source Coverage
We compared relation types, source coverage, and
structural patterns in knowledge graphs generated by
Gemini 2.0 Flash,Grok 3, andGPT-4oacross 40 on-
cology reports. Figure 4(a) summarizes key evalua-
tion metrics.
Gemini and Grok focused on core clinical predi-
cates (e.g.,indicates,treats,influences), while
GPT-4o generated a broader range of descriptive
and context-sensitive relations (e.g.,pertains_to,
assesses,documents), reflecting its strength in se-
mantic generalization.
All models extracted key clinical entities (Patient,
Condition,CarePlan) and biomarkers (age,
ca_19_9,tumor_grade). GPT-4o also surfaced
nuanced attributes (e.g.,fertility_preservation,
estrogen_exposure), showing sensitivity to subtle
contextual cues often missed by the others.
9

Figure 2: Comparison of attribute-level metrics for PDAC and BRCA cohorts. From left to right: raw
coverage percentage, correctness percentage, incorrect attribute counts, and overall averages.
Figure 3: Top: Per-patient factual correctness of EAV triples across three LLMs. Bottom: Pairwise correct-
ness differences and Pearson correlation trends across 40 patients.
Structurally, each model connected densely around
nodes likeConditionandTreatment. Gemini
emphasized procedural clusters (e.g.,Assessment,
Practitioner); GPT-4o incorporated diverse con-
cepts (e.g.,Therapy,FollowUp) and minimized re-
dundancy. Grok prioritized precision but exhib-
ited more repetition due to conservative extraction.
Figure 4(a) illustrates: GPT-4o excels in relational
breadthandabstraction; Grok3insemanticprecision
and redundancy control; and Gemini 2.0 in balanced
procedural accuracy.
Semantic Relation Analysis Across LLMs
To evaluate the semantic integrity of relationships in
the clinical knowledge graph, we conducted a com-
parative analysis of predicate usage byGemini 2.0
Flash,Grok 3, andGPT-4o. Rather than focus-
ing solely on factual correctness, this analysis em-
phasizes whether each model employs predicates that
are both clinically meaningful and ontologically valid
(e.g.,treats,leads_to,indicates). We identi-
fied and grouped common relational errors into four
categories: (1)Vagueness, involving imprecise predi-
catessuchasdrug_use vague; (2)Causal Overreach,
where unsupported causality is overstated (e.g.,bp
leads_to comorbidities); (3)Incorrect Semantics,such as the misuse ofcontraindicatesoris_a; and
(4)Overuse/Generality, with excessive use of broad
predicates likeindicatesandinfluences. Fig-
ure 4(b) presents a radar chart summarizing semantic
issues. GPT-4o shows more vagueness and generality
duetoitsgenerativestyle; Grok3minimizessemantic
errors through rigorous filtering; Gemini 2.0 exhibits
more causal overreach from broader predicate explo-
ration. These trends suggest that GPT-4o provides
relational diversity but may lack precision, Grok 3
emphasizes semantic clarity and restraint, and Gem-
ini2.0contributesexploratoryrichnesswithsomerisk
of drift. Together, their combined strengths enable
predicate refinement and clinical coherence in graph
construction.
Clinical Relevance Evaluation Across LLMs
To assess the clinical relevance of LLM-generated re-
lationships,wecomparedaverageentityandattribute
relevanceacross40oncologyreports. AsshowninTa-
ble 5, GPT-4o achieved the highest relevance (0.87
for entities and 0.89 for attributes), followed by Grok
(0.85 entity, 0.88 attribute), and Gemini (0.80 en-
tity, 0.79 attribute). GPT-4o’s strength lies in con-
textual reasoning; Grok excels in conservative filter-
ing; Gemini trades precision for broader coverage.
10

(a) Relation and structural metrics
 (b) Semantic issue distribution
 (c) Unified semantic evaluation
Figure 4: Radar chart comparison of Gemini 2.0 Flash, Grok 3, and GPT-4o: (a) relation diversity and
structural metrics, (b) semantic issue categories, and (c) unified evaluation across correctness, inference, and
data support.
Table 5: Comprehensive evaluation of clinical knowledge graph construction across LLMs with illustrative
examples.
Category Metric Gemini Grok GPT-4o Example / Description
Data SupportEntity supported (per
report)21.57 24.00 23.10 Entities per patient (Patient,Condition,
ImagingStudy).
Attribute supported
(per report)67.67 89.03 83.50 Examples:tumor_size,Ki-67_Index,
weight_loss.
Attribute diversity
(Unique/Total)0.52 0.60 0.68 Range of unique attributes, e.g., bothageand
lymph_node_count.
Inference/AbstractionEntity inferred (avg) 0.12 0.28 0.95 InfersHER2implies aBiomarkerentity.
Attribute inferred
(avg)0.70 1.35 3.20 InfersPrognostic_Markerfrom context.
Attribute independent
(no entity)0.25 0.10 1.05Obesityextracted without linkedPatient.
Implicit inference ra-
tio (%)1.01% 1.49% 3.92% Implicit triple:ER-positive→responds_to→
Tamoxifen.
Cross-sentence infer-
ence (#)1.0 1.5 3.2 Combines scattered mentions ofsurgeryand
blood_loss.
Gaps/SuggestionsAvg gaps per patient 1.00 2.00 2.00 Missing link:Diagnosis→associated_with→
Biopsy.
Avg suggestions per
patient1.00 1.00 1.50 Suggestsliver_functionfor elevated
bilirubin.
Gap-fill accuracy (%) 55.0% 61.0% 68.0% Fraction of suggestions matching expert-
annotated content.
Correctness/RelevanceEntity correct (%) 99.4% 23.3% 11.7% Correctly identifiesImagingStudyas a clinical
entity.
Attribute correct (%) 97.5% 75.3% 18.0% Example:CA_19-9used properly asLabResult.
Attribute relevance
(avg)0.80 0.89 0.87 Clinical utility ofBMIandTumor_Grade.
Attribute inferred (%) 0.95% 0.18% 2.49% InfersSmoking_StatusaffectsCancer_Risk.
Attribute independent
(%)0.34% 0.00% 1.21% Isolated attributes not linked to any entity.
Triple validation con-
fidence0.96 0.91 0.88 Confidence inBiopsy→confirms→
TumorType.
Semantic QualityHallucination rate (%) 0.18% 0.04% 0.92% False triple:Vitamin_D→prevents→Cancer.
Redundancy rate (%) 2.2% 0.7% 1.8% Relation repetition for the same attribute (e.g.,
indicates).
SPARQL-compatible
triples (%)98.1% 99.6% 96.2% Fraction of triples executable in SPARQL end-
points.
11

ThesecomplementarycapabilitiesmaketheLLMtrio
well-suitedforbuildingclinicallygroundedknowledge
graphs that balance accuracy and discovery, and can
be iteratively improved through expert feedback and
retrieval-based refinement.
Unified Evaluation of LLM Behavior in KG
Construction
We evaluatedGemini 2.0 Flash,Grok 3, andGPT-
4oacross six dimensions of clinical knowledge graph
construction: data support, inference, gap handling,
correctness, semantic quality, and SPARQL compat-
ibility. Table 5 provides a detailed comparison with
clinical examples.Gemini 2.0excels in precision,
achieving the highest entity (99.4%) and attribute
correctness (97.5%), and high SPARQL compliance
(98.1%), though with limited inference and attribute
diversity due to its schema-constrained approach.
Grok 3offers strong semantic rigor and moder-
ate inference, with high attribute relevance (0.89),
minimal hallucination (0.04%), and low redun-
dancy—ideal for conservative, high-precision graph
construction.GPT-4oleads in contextual infer-
ence and semantic enrichment, extracting diverse at-
tributes (e.g.,estrogen_exposure) and achieving
the highest gap fill accuracy (68.0%). While halluci-
nation is slightly higher, its strength lies in uncover-
ingimplicitrelationsandimprovinggraphexpressive-
ness. In the absence of ground truth, our multi-agent
ensemble ensures robustness by combining Gemini’s
structured extraction, Grok’s semantic validation,
and GPT-4o’s contextual enrichment. This yields
ontology-aligned, clinically meaningful graphs suit-
able for decision support and advanced analytics.
5 Limitations and Conclusion
Despite its strengths, the framework has several lim-
itations. It is not yet integrated with live clinical de-
cision support systems (CDSS), limiting real-world
deployment. It currently operates solely on unstruc-
tured clinical text, omitting other critical modalities
such as imaging (CT, MRI), biosignals (ECG, EMG),
and structured EHR data. Our evaluation, based on
40 oncology reports (PDAC and BRCA) from the
CORAL dataset, also limits generalizability across
broader clinical settings. While model consensus and
ontology alignment provide a form of weak supervi-
sion, expert-in-the-loop validation remains essential
for resolving ambiguous or domain-specific triples.
We introduce the first framework to construct
and evaluate clinical knowledge graphs directly from
free-text reports using multi-LLM consensus anda Retrieval-Augmented Generation (RAG) strategy.
Our pipeline combines schema-guided EAV extrac-
tion viaGemini 2.0 Flash, semantic refinement by
GPT-4oandGrok 3, and ontology-aligned encoding
usingSNOMEDCT,RxNorm, LOINC,GO,andICD
in RDF/RDFS/OWL with SWRL rules.
This KG-RAG approach supports continuous re-
finement, enabling iterative improvement of graph
quality through self-supervised validation and expert
feedback. By retrieving, validating, and grounding
extracted triples over time, the system evolves dy-
namically, bridging static KG generation with adap-
tive knowledge refinement. Gemini anchors high-
precision extraction, Grok filters hallucinations and
enforces semantic rigor, and GPT-4o contributes con-
textual depth and abstraction, yielding clinically rel-
evant, SPARQL-compatible graphs.
Future directions include real-time CDSS integra-
tion, multimodal fusion, large-scale validation across
institutions, and enriched temporal and causal mod-
eling. Ultimately, our framework provides a scalable
and explainable foundation for transforming unstruc-
tured narratives into trustworthy, ontology-grounded
clinical knowledge graphs.
References
[1] Muhammad Ayaz, Muhammad F Pasha, Mo-
hammed Y Alzahrani, Rahmat Budiarto, and
Deris Stiawan. The fast health interoperabil-
ity resources (fhir) standard: systematic liter-
ature review of implementations, applications,
challenges and opportunities.JMIR Medical In-
formatics, 9(7):e21929, 2021.
[2] Shuyang Cao, Lu Wang, and Dragomir Radev.
A survey on hallucination in natural language
generation.ACM Computing Surveys (CSUR),
55(12):1–38, 2023.
[3] Google DeepMind. Gemini 2.0 technical report.
https://deepmind.google/technologies/
gemini/, 2024. Accessed May 2025.
[4] Michael Gubanov, Anna Pyayt, and Aleksandra
Karolak. Cancerkg. org-a web-scale, interactive,
verifiable knowledge graph-llm hybrid for assist-
ing with optimal cancer treatment and care. In
Proceedings of the 33rd ACM International Con-
ference on Information and Knowledge Manage-
ment, pages 4497–4505, 2024.
[5] BhimeshKandibedala, AnnaPyayt, NickolasPi-
raino, Chris Caballero, and Michael Gubanov.
Covidkg.org–a web-scale covid-19 interactive,
12

trustworthy knowledge graph, constructed and
interrogatedforbiasusingdeep-learning. InPro-
ceedings of the International Conference on Ex-
tending Database Technology (EDBT), 2023.
[6] Rituparna Khan and Michael Gubanov. We-
blens: Towards web-scale data integration,
training the models. In2020 IEEE International
Conference on Big Data (Big Data), pages5727–
5729. IEEE, 2020.
[7] Stephanie Lin, Jacob Hilton, and Owain Evans.
Truthfulqa: Measuring how models mimic hu-
man falsehoods.Proceedings of the 60th Annual
Meeting of the Association for Computational
Linguistics (ACL), pages 3214–3252, 2022.
[8] Shizhe Liu, Chong Zheng, Yuxian Wang,
Zhengyan Zhang, Zhiyuan Liu, and Maosong
Sun. Kala: Knowledge-augmented lan-
guage model pretraining.arXiv preprint
arXiv:2306.11644, 2023.
[9] Zhiyuan Liu, Zhengyan Zhang, Yuxian Wang,
Yantao Shen, Zhicheng Liu, Jiliang Tang, and
Maosong Sun. Knowgl: Knowledge generation
and linking from text. InProceedings of the 60th
Annual Meeting of the Association for Com-
putational Linguistics (ACL), pages 4364–4378,
2022.
[10] OpenAI. Gpt-4o.https://openai.com/index/
gpt-4o, 2024. Accessed: 2025-05-30.
[11] Pranav Rajpurkar, Jeremy Irvin, Haoran Zhang,
Brandon Yang, Harsh Mehta, Katie Shpanskaya,
Bobak J Wang, Riley Jones, Kevin H Yu, and
Matthew P Lungren. Evaluating the factual
consistency of large language models in medical
applications.NPJ Digital Medicine, 6(1):1–13,
2023.
[12] Regenstrief Institute. Loinc: Logical observation
identifiers names and codes.https://loinc.
org, 2022. Accessed May 2025.
[13] Noah Shinn, Yuchen Tang, Harsha Nori, Vidhi
Agarwal, Jianfeng Gao, and Xinyun Chen. Re-
flexion: Language agents with verbal reinforce-
mentlearning.arXiv preprint arXiv:2303.11366,
2023.
[14] Mark Simmons, Daniel Armstrong, Dylan So-
derman, and Michael Gubanov. Hybrid.json:
High-velocity parallel in-memory polystore json
ingest. In2017 IEEE International Conference
on Big Data (Big Data),pages2741–2750.IEEE,
2017.[15] SNOMED International. Snomed ct: Sys-
tematized nomenclature of medicine – clinical
terms.https://www.snomed.org/snomed-ct,
2022. Accessed May 2025.
[16] Madhumita Sushil, Vanessa E Kennedy, Divneet
Mandair, Brenda Y Miao, Travis Zack, and
Atul J Butte. Coral: expert-curated oncology
reports to advance language model inference.
NEJM AI, 1(4):AIdbp2300110, 2024.
[17] U.S. National Library of Medicine. Rxnorm:
Normalized names for clinical drugs.https:
//www.nlm.nih.gov/research/umls/rxnorm/,
2022. Accessed May 2025.
[18] xAI. Grok: Ai by xai (elon musk’s company).
https://x.ai, 2024. Accessed May 2025.
[19] HongchenXue,QingzhiMa,GuanfengLiu,Jian-
feng Qu, Yuanjun Liu, and An Liu. Clr2g:
Cross modal contrastive learning on radiology
report generation. InProceedings of the 33rd
ACM International Conference on Information
and Knowledge Management, pages 2742–2752,
2024.
[20] Xi Yang, Yichi Zhang, Yuxing Si, Yuhao Zhang,
Meng Jiang, Fei Wang, and Hua Xu. Gatortron-
rag: A retrieval-augmented generation frame-
work for biomedical question answering.arXiv
preprint arXiv:2311.00964, 2023.
[21] Rui Zhang, Jack Syu, Edward Hu, Kai-Wei
Chang, and Hao Tan. Knowgpt: Benchmarking
andimprovingknowledgegroundingoflargelan-
guage models.arXiv preprint arXiv:2402.08600,
2024.
[22] Xuejiao Zhao, Siyan Liu, Su-Yin Yang, and
Chunyan Miao. Medrag: Enhancing retrieval-
augmented generation with knowledge graph-
elicited reasoning for healthcare copilot. InPro-
ceedings of the ACM on Web Conference 2025,
pages 4442–4457, 2025.
[23] Yinghao Zhu, Changyu Ren, Zixiang Wang,
Xiaochen Zheng, Shiyun Xie, Junlan Feng,
Xi Zhu, Zhoujun Li, Liantao Ma, and Cheng-
wei Pan. Emerge: Enhancing multimodal elec-
tronic health records predictive modeling with
retrieval-augmented generation. InProceedings
of the 33rd ACM International Conference on
Information and Knowledge Management, pages
3549–3559, 2024.
13