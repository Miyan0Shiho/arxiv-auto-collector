# GARAGE: Characterizing the Automation Boundary in LLM-based Attack Graph Generation

**Authors**: Daekwon Pi, Sangho Lee, Young Hun Lee, Huy Kang Kim

**Published**: 2026-07-20 16:08:23

**PDF URL**: [https://arxiv.org/pdf/2607.18108v1](https://arxiv.org/pdf/2607.18108v1)

## Abstract
While modern vehicle security depends on effective Cyber Threat Intelligence (CTI) synthesis, current automated tools struggle with unstructured data and automotive-specific architectural nuances. To bridge this gap, we introduce GARAGE, a RAG-powered framework that converts fragmented CTI into an actionable, domain-specific knowledge base for automated attack graph generation. GARAGE synthesizes a dataset of 12,786 CVEs and 140 incident reports into a STIX 2.1 and Auto-ISAC ATM-compliant knowledge base. By formalizing tactical-pattern-level scenarios through granular kill chain analysis, GARAGE achieves threat generation capabilities. Our 320 Leave-One-Out experiments reveal that the framework can accurately transfer security knowledge to entirely unseen vehicle architectures. Furthermore, we position GARAGE as a scalable TARA support tool within human-in-the-loop workflows, offering a comprehensive cost-performance analysis to guide its deployment across various LLM tiers.

## Full Text


<!-- PDF content starts -->

GARAGE: Characterizing the Automation Boundary in
LLM-based Attack Graph Generation
Daekwon Pi1, Sangho Lee2, Young Hun Lee3, and Huy Kang Kim1
1School of Cybersecurity, Korea University, Seoul, Republic of Korea
daekp@korea.ac.kr, cenda@korea.ac.kr
2Samsung Electronics, Suwon, Republic of Korea
s35.lee@samsung.com
3Agency for Defense Development (ADD), Daejeon, Republic of Korea
yh_lee@add.re.kr
Abstract.While modern vehicle security depends on effective Cyber Threat Intelligence (CTI) synthesis,
current automated tools struggle with unstructured data and automotive-specific architectural nuances. To
bridge this gap, we introduce GARAGE, a RAG-powered framework that converts fragmented CTI into an
actionable, domain-specific knowledge base for automated attack graph generation. GARAGE synthesizes a
dataset of 12,786 CVEs and 140 incident reports into a STIX 2.1 and Auto-ISAC ATM-compliant knowledge
base. By formalizing tactical-pattern-level scenarios through granular kill chain analysis, GARAGE achieves
threat generation capabilities. Our 320 Leave-One-Out experiments reveal that the framework can accurately
transfer security knowledge to entirely unseen vehicle architectures. Furthermore, we position GARAGE
as a scalable TARA support tool within human-in-the-loop workflows, offering a comprehensive cost-
performance analysis to guide its deployment across various LLM tiers.
Keywords:Attack Graph Generation ·Large Language Models ·Retrieval-Augmented Generation
·Vehicle Security·Cybersecurity
1 Introduction
While Large Language Models (LLMs) offer significant po-
tential for automated security analysis, their application in the
automotive industry faces distinct challenges. Much of the
knowledge required for cybersecurity analysis remains pri-
vate, fragmented, and difficult to share across organizations.
One promising application of LLMs to overcome this lim-
itation is generating vehicle-level attack graphs. Traditional
Threat Analysis and Risk Assessment (TARA) is often scoped
to individual Electronic Control Units (ECUs). In practice,
however, safety-critical attacks are rarely confined to a sin-
gle ECU. Instead, they exploit chained vulnerabilities across
multiple components and architectural layers, including in-
vehicle networks (e.g., CAN/Ethernet), operating systems,
and hardware interfaces. Constructing and validating such
end-to-end attack paths requires cross-domain expertise span-
ning networking, OS internals, and hardware, which is diffi-cult for human analysts to acquire and systematically apply.
LLMs can bridge this gap by aggregating distributed technical
knowledge to support multi-layer security reasoning.
Generating attack graphs requires modeling relationships
between assets, vulnerabilities, and attack actions (e.g., pen-
etration, exploitation). Graph-based approaches provide ex-
plicit structure but often fail to capture the semantic context
embedded in unstructured reports. Vector-based retrieval cap-
tures semantic similarity but often fails to preserve causal,
step-by-step chaining. In this work, we adopt a Hybrid
Retrieval-Augmented Generation (Hybrid RAG [21]) ap-
proach that jointly leverages graph-based structural knowl-
edge and vector-based semantic retrieval for attack graph
construction.
This challenge is further exacerbated by the manual nature
of current TARA practice [25], which prolongs assessment
duration and hinders timely incorporation of rapidly evolv-
ing, unstructured Cyber Threat Intelligence (CTI). To address
1
arXiv:2607.18108v1  [cs.CR]  20 Jul 2026

this, LLMs can act as automated analysts, directly generating
attack graphs and inferring vulnerability chaining to support
actionable, vehicle-level security assessment [19].
In this paper, we present GARAGE (GenerativeAI with
RAG-basedAttackGraphEngine), an end-to-end framework
that transforms publicly available automotive CTI into a
domain-specific Knowledge Base (KB) and automatically
generates vehicle-level attack graphs via RAG-based infer-
ence. Our main contributions are:
•We construct an automotive cybersecurity KB from
12,786 CVEs and 140 security event documents, aligned
with STIX 2.1 and the Auto-ISAC Automotive Threat
Matrix (ATM). An LLM-based pipeline extracts security-
relevant entities, relations, and events to populate both a
Knowledge Graph (KG) and a vector store. We publicly
release this KB to support reproducibility and future
research.1
•We design a dual-metric evaluation framework (Practical
Feasibility and Knowledge Reconstruction) that sepa-
rately assesses attack path viability and security knowl-
edge depth, enabling per-dimension evaluation of LLM
capabilities.
•Through 320 Leave-One-Out (LOO) experiments across
four real-world attacks and eight LLMs, we empirically
establish anautomation boundarybased onknowledge
granularity. Specifically, LLMs can generate high-level
tactical threat scenarios, whereas low-level implementa-
tion details require human expert intervention.
•We provide quantitative deployment guidance through
cost-performance trade-off analysis across proprietary
and open-weight model tiers, demonstrating the utility
of GARAGE as a practical TARA support tool within a
Human-in-the-loop workflow.
2 Methodology
2.1 Overview of the Pipeline
GARAGE is a Retrieval-Augmented Generation (RAG)-
based system that automatically generates vehicle-level attack
graphs. Inputs include unstructured CTI texts, CVE descrip-
tions, ATM technique definitions, and (optionally) target vehi-
cle architecture specifications. The output is an attack graph
in structured JSON format that captures step-by-step vulner-
ability chaining, supported by evidence retrieved from both
the vector store and KG.
The pipeline combines two complementary databases. An
LLM extracts security-relevant entities, relations, and events
from CTI texts and stores them in a KG and a vector store.
By integrating semantic similarity from the vector store with
1https://anonymous.4open.science/r/GARAGE-dataset-4265/explicit structural connectivity across ECUs, components, and
vulnerabilities stored in the KG, GARAGE captures both
semantic context and structural chaining.
During inference (Figure 1), GARAGE (i) formulates ATM-
guided queries and performs vector retrieval to collect relevant
CTI evidence, (ii) traverses the KG for high-confidence struc-
tural evidence around target assets/ECUs, and (iii) uses an
LLM to synthesize both evidence sources into attack paths.
To mitigate hallucinations, the LLM outputs are constrained
to remain consistent with retrieved entities/relations and are
supported by the underlying textual evidence.
2.2 Dataset
The CTI dataset was sourced from theAutoSec-Timeline
repository (as of December 17, 2025) [2], comprising12,786
CVEsand140 security events.
2.2.1 Vulnerability Data
To filter CVEs relevant to automotive software, we collected
software names and versions from the open-source license no-
tices of three OEMs (Hyundai MOBIS [16], V olkswagen [24],
Kia [14]), selected for the public availability of their detailed
software disclosures. Using this data, we applied regex-based
filtering to identify relevant vulnerabilities. Selected vulner-
ability descriptions were stored in the vector store, and ex-
tracted entities were ingested into the KG (see Section 2.3).
2.2.2 Security Event Data
Raw textual data were manually curated from various sources,
including technical blogs, reports, and presentations. To main-
tain format consistency, PDF and PPT documents were con-
verted to text using the pymupdf4llm library, while YouTube
content was processed via transcript extraction. For image-
based materials, we provided manually authored natural-
language descriptions. A total of 164 cases were initially
gathered, of which24were excluded due to source inacces-
sibility (e.g., 404 errors, inaccessible Tor hidden services) or
a lack of technical detail. This refinement process yielded
140documents, whose event descriptions were subsequently
embedded and indexed in the vector store.
2.3 Knowledge Base Construction
All KB construction stages (entity, relation, event extraction,
and rewriting) are independently performed by the same four
LLMs:Gemini 3.0 Flash, GPT-5.2, Claude 4.5 Sonnet, and
DeepSeek-V3.2. The goals are (i) to compare model-specific
knowledgeandreasoningcharacteristics in security analysis
tasks, and (ii) to evaluate how evidence grounding via RAG
enhances inference across model tiers.
2

Figure 1: Proposed GARAGE System Architecture and Data Flow. The system integrates graph-based structural retrieval with
vector-based semantic retrieval to improve attack graph generation quality.
Table 1: The Entity Types for Automotive CTI
Type Definition
componentHW/SW module within the vehicle
target_ecuExplicitly targeted ECU in an attack
vulnerabilityIdentified security defect (incl. CWE)
network_domain Vehicle network segment or logical domain
network_protocolCommunication protocol used in attack
attack_vectorMethod of attack delivery
supplierComponent supplier / manufacturer
researcherResearcher / org. that reported a vuln.
cveCVE identifier
cweCWE identifier
assetSafety-critical function / system affected
toolHW/SW tool used to perform an attack
2.3.1 Entity extraction
We design an entity extraction pipeline to identify cyberse-
curity elements from automotive CTI sources. As inputs, we
use the filtered CVE Description field and CTI documents
segmented with chunk_size=500 andchunk_overlap=150 .
Each chunk is processed independently, and results from the
same document are merged at the document level.
Each model extracts entity mentions under 12 predefined
types (Table 1). Results are stored in structured JSON and
provided as the normalized entity set (E extracted ) to constrain
the candidate space for relation extraction.
Our domain-specific entity schema resolves the se-
mantic ambiguity of generic STIX objects (e.g., Device )
that fail to distinguish automotive-specific components.
By explicitly separating entities like IVI and Gateway
ECU, it enables precise, domain-specific relations such as
(target_ecu)-[BELONGS_TO]-(network_domain).
All LLM engines share a standardized prompt framework
with three core directives: (1)Verbatim Extractionto pre-
vent hallucinations, (2)Negative Instructionto prohibit gen-
erating unmentioned entities, and (3)Structural Guidance
with clear definitions for consistent classification (full prompt
templates in Appendix A.2).2.3.2 Entity normalization
To eliminate semantic redundancy, we apply an embedding-
based normalization. Raw entities are integrated with CVE
product names, descriptions, and CWE information, then
embedded using OpenAI’s text-embedding-3-large (φ:
Eraw→Rd). We perform Agglomerative Hierarchical Clus-
tering with Average Linkage and a cosine distance thresh-
oldτ=0.3 , empirically tuned to balance under-clustering
(e.g., failing to merge ‘CAN’ and ‘Controller Area Network’)
against over-clustering of distinct entities. The shortest string
in each cluster is selected as the canonical representative
(canonical(e) =argmin e′∈cluster(e) |e′|), preferring standard
domain acronyms. In Neo4j, we adopt aCanonical Nodear-
chitecture: representative nodes (e.g., CanonicalComponent )
are created and linked from individual entities via SAME_AS
relationships, consolidating duplicate entries into canonical
representations.
2.3.3 Relation extraction
To identify semantic associations between extracted enti-
ties, we extract Subject-Predicate-Object (SPO) triplets T=
(es,r,e o)from the CTI corpus D. The same four LLMs inde-
pendently perform relation extraction, each receiving the raw
text alongside the normalized entity list (E extracted ).
All prompts enforce five core constraints: (1)Entity Con-
straint( es,eo∈E extracted ); (2)Relational Constraint, limit-
ingrto 9 predefined types (Table 2); (3)Verbatim Consis-
tency; (4)Negative Instructionto suppress hallucinations;
and (5)Structural Enforcementvia strict JSON schema. The
resulting KG, averaged across the four models, comprises ap-
proximately8,824unique entity nodes and9,265relation
edges.
2.3.4 Event extraction
To model dynamic causality and behavioral patterns, we ex-
tract security events from unstructured text using the same
four LLMs. Each model identifies security incidents and trans-
forms them into structured event nodes comprisingactor,ac-
tion,target, andpre/post-conditions. Each extracted event is
3

Table 2: The Predefined Relation Types for Triplet Extraction
Relation Type Definition
HAS_VULNERABILITYComponent/ECU contains a vulnerability
IDENTIFIED_BYVulnerability is associated with a CVE
CONNECTED_TO Component/ECU connects to a network
domain
USES_PROTOCOL Component/ECU/tool uses a network
protocol
DELIVERED_VIA Vulnerability is triggered via an attack
vector
PROVIDED_BYSupplier provides a component/ECU
REPORTED_BYResearcher/org. reported a vulnerability
AFFECTS_ASSETVulnerability affects a specific asset
USED_AGAINSTTool is used to attack a target
then mapped to specific ATM techniques via a fine-tuned clas-
sification model (detailed in Section 2.3.6), which assigns the
corresponding technique_id (e.g., ATM-T0059) and links
it through a CLASSIFIED_AS relationship. This hierarchical
structuring allows tracing how individual actions on specific
components map to broader attack tactics.
2.3.5 Rewriting
The Rewriting phase converts structured entities and rela-
tions back into natural language to improve retrieval quality.
Each of the four LLMs independently synthesizes factual
summaries from verified fact triplets, excluding the noise
present in raw CTI texts. The resulting summaries are stored
in the vector store, directing the RAG system to retrieve re-
fined factual information rather than noisy unstructured data
(Appendix A.2).
2.3.6 Technique extraction
To classify ATM techniques, we developed a fine-
tuned model via Supervised Fine-Tuning (SFT) based on
GPT-4.1-2025-04-14 . Zero-shot prompting showed high
costs and limited precision for granular technique classifica-
tion, necessitating the SFT approach. The training dataset was
constructed from the latest ATM data with virtual incident
reports synthesized in five narrative styles (research reports,
advisories, press releases, forum posts, and training materials;
Appendix A.2). Three measures ensure training robustness:
(1) label leakage prevention, which blacklists technique names
and IDs from generated narratives; (2) data augmentation via
character-level noise injection and random truncation; and
(3) semantic consistency validation, where each augmented
sample is re-classified by a separate LLM to filter inconsis-
tent examples. Additionally, negative samples drawn from
non-security contexts (e.g., vehicle maintenance logs) were
included to suppress false positives.2.4 Attack graph generation using Hybrid
RAG
2.4.1 Context-Aware Vector Retrieval
To reflect specific vehicle specifications in the retrieval pro-
cess, we employ a three-stage retrieval pipeline. (1)Dy-
namic Query Generation: the system analyzes the Vehicle
Specification to identify entities such as ECUs and proto-
cols. (2)ATM-Guided Query Formulation: queries are
constructed by concatenating each ATM technique’s title
and description to align retrieval with operational contexts.
(3)Cross-Encoder Reranking: initial candidates are re-
ranked using a Cross-Encoder (ms-marco-MiniLM-L-6-v2)
that evaluates query-document pairs simultaneously to select
the top-10 documents as final context.
2.4.2 Structured Knowledge Graph Traversal
To supplement unstructured text with explicit component
connectivity, we employ the Neo4j-based KG. ECU IDs
and protocol names from the vehicle spec are mapped
asSeed Nodes, and traversal is strictly filtered to fol-
low only security-critical relations ( HAS_VULNERABILITY ,
DELIVERED_VIA ,EXPLOITS ) with a maximum depth of 2-
hops. This constraint effectively captures core causal chains
such as “ECU →Component →Vulnerability,” avoiding ex-
haustive graph traversal. For densely-connected nodes, raw
SPO triples are replaced by an LLM-generated security sum-
mary that compiles the entity’s associated vulnerabilities and
attack vectors, following the community summarization prin-
ciple of GraphRAG [7]. This reduces prompt token consump-
tion while preserving essential security semantics.
2.4.3 Hybrid Context Integration and Attack Graph
Generation
Similar attack cases retrieved from the vector store and
component-vulnerability relationships traversed from the KG
are integrated into a unified prompt. When the two sources
conflict, a conservative approach is adopted: vulnerabilities
recorded in the KG are assumed to exist by default. Vector
store evidence (e.g., patch notes) overrides this assumption
only when it explicitly references the target vehicle’s make,
model, and year, in which case the feasibility of the corre-
sponding attack path is reduced or excluded. The LLM then
uses this hybrid context together with the target vehicle’s net-
work topology to infer multi-stage attack paths, outputting
the result as a structured attack graph.
3 Experiment
We designed experiments around three research questions to
evaluate the effectiveness, reliability, and practical utility of
GARAGE.
4

•RQ1: Inference Capability on Unknown Threats.
Evaluating whether the framework can infer potential
attack paths solely from vehicle architectural specifi-
cations when direct knowledge of the target attack is
absent.
•RQ2: Reliability as an Expert Assistant.Determin-
ing whether the framework maintains consistent results
across repeated trials, ensuring the stability required for
an automated expert assistant.
•RQ3: Cost-Performance Efficiency.Investigating
whether open-weight models provide a viable, cost-
effective alternative to proprietary models while main-
taining sufficient analytical depth.
3.1 Evaluation Benchmarks
We selected four well-documented automotive cyber incidents
(C1–C4) representing diverse architectures and attack vectors:
•C 1: Jeep Cherokee [11]— Remote exploitation via
cellular network targeting Uconnect and gateway (V850)
to control CAN bus. Validates attack paths in legacy
architectures.
•C 2: Mercedes-Benz W177 [13]— Exploitation of head
unit–backend communication and WebKit flaws. Evalu-
ates connected car analysis.
•C 3: Nissan Leaf [3]— Bluetooth stack buffer overflow
via the Hands-Free Profile (HFP) for RCE on IVI. Tests
short-range wireless vulnerability detection.
•C 4: BMW i3 [12]— Multi-vector attacks (GSM, Blue-
tooth, USB) compromising HU and TCU, with gateway
bypass via UDS diagnostic messages.
Ground Truth & Controls.We manually reconstructed at-
tack scenarios from detailed technical white papers, referenc-
ing ATM incident examples for C1andC4[1]. Ground truth
was established by mapping attack paths to ATM Tactics and
Techniques.
Two strict controls were enforced: (1) a LOO strategy
excluding target incident data from the KB, and (2)En-
tity Anonymizationto prevent parametric memory recall
and force architectural reasoning. We omitted vehicle make
and model names from queries, and strictly filtered out re-
trieved results from both the vector store and KG that con-
tained target-specific keywords (e.g., jeep ,uconnect forC1;
bmw,connecteddrive forC4). General technical terms (e.g.,
SPI, CAN, Bluetooth) are intentionally retained as domain-
common knowledge shared across vehicles.3.2 Experimental Environment
3.2.1 Knowledge Base Configuration
To evaluate end-to-end capability, we constructed independent
KBs using four SOTA models (GPT-5.2, Gemini 3.0 Flash,
Claude 4.5 Sonnet, DeepSeek-V3.2) as Extraction Models.
Since each model produces a unique knowledge representa-
tion, we created 16 hybrid KB instances ( 4Extraction Models
×4 scenarios), each with LOO applied to both stores:
•Graph DB (Neo4j):Graph topology varies across in-
stances because each LLM extracts distinct triples.
•Vector DB (FAISS):Unlike standard RAG, our vector
store is built via natural language rewriting of extracted
triples, making embeddings model-dependent and requir-
ing 16 separate indices.
The KGs contain an average of approximately 8,824 nodes
and 9,265 edges. Vectorization uses OpenAI text-embedding-
3-small (dim=1536).
3.2.2 Target LLMs
We selected eight LLMs including both proprietary and open-
weight categories. For the experiment, models are grouped by
theirfunctional rolerather than license type:
•Extraction & Inference (4 models):GPT-5.2, Gemini
3.0 Flash, Claude 4.5 Sonnet, DeepSeek-V3.2, accessed
via official APIs. These serve dual roles as both Extrac-
tion Models (Section 3.2.1) and inference targets eval-
uated on their own KB instances. Note that DeepSeek-
V3.2 is an open-weight model but is included in this
group due to its competitive extraction capability and
API accessibility.
•Inference Only (4 models):Qwen3-coder, Llama 4
Maverick, GPT-OSS-120b, GPT-OSS-20b, accessed via
OpenRouter API [17] for the inference phase only.
Evaluation Mapping.Extraction & Inference models query
their own KB instances, while Inference Only models are uni-
formly mapped to the GPT-5.2-generated KBs as the standard
retrieval pool, selected as the most capable extraction model
available at the time of experimentation.
In the Results section, models are regrouped byweight avail-
ability—Proprietary (Claude, GPT, Gemini) vs. Open-Weight
(DeepSeek, Qwen3, Llama, GPT-OSS)—to analyze perfor-
mance differences attributable to model architecture and scale.
Hyperparameters.Generation temperature is set to 1.0 for
all models to encourage exploration of diverse multi-vector
attack paths. The GARAGE framework constrains outputs
with retrieved evidence, ensuring structural validity despite
the elevated sampling temperature.
5

3.3 Evaluation Methodology: LLM-as-a-
Judge
A single metric cannot simultaneously capture attack-path
viability and knowledge accuracy; the dual design enables
separate analysis—for example, high KR but low PF indi-
cates strong domain knowledge yet poor path consistency, an
outcome commonly observed in open-weight models.
3.3.1 Practical Feasibility (PF)
PF evaluates attack path viability from a red-team perspective,
enforcing thecascade failureprinciple: if step kis invalidated,
all subsequent steps ( i≥k ) are scored as failed. PF comprises
three components:
(1)Chain Completeness (Max 60):Ratio of contiguous valid
steps before the first invalidation point k. Each step siis scored
as V ALID (1.0), PARTIAL (0.5), or BROKEN (0.0):
Score PF-Comp =∑N
i=1si
N×60 wheres i=0 fori≥k
(2)Attack Feasibility (Max 30):Logical transition validity
between adjacent steps, scored as FEASIBLE (1.0), QUES-
TIONABLE (0.5), or INFEASIBLE (0.0):
Score PF-Feas =∑k−1
j=1tj
N−1×30
(3)Critical Violations (up to −20):Penalties for physical/-
logical impossibilities (e.g., trust boundary crossing −10, pro-
tocol mismatch −5). No double jeopardy with elements al-
ready scored 0.
3.3.2 Knowledge Reconstruction (KR)
Unlike PF, KR measures the depth of security knowledge in-
dependently of logical chain integrity—accurately identifying
a subsequent step accrues points even if a preceding step is
missed.
(1)Entity & Step Identification (Max 50):A 5-level se-
mantic matching scale ( mi): FULL (1.0), SEMANTIC (0.75),
PARTIAL (0.5), WEAK (0.25), or NONE (0.0):
Score KR-Iden =∑N
i=1mi
N×50
(2)Structural Knowledge (Max 30):Transition connectivity
classified as CONNECTED (1.0), IMPLIED (0.5), or DIS-
CONNECTED (0.0):
Score KR-Struct =∑N−1
j=1cj
N−1×30
(3)Knowledge Precision (Max 20):Baseline 10 points;
bonuses for accurate details (e.g., exact CVEs), deductions
only for hallucinations (e.g., fabricated ECUs). Valid alterna-
tive attack paths are not penalized.Evaluation Setup.We adopt the LLM-as-a-Judge approach,
which recent studies have shown to achieve high agreement
with human expert ratings when guided by structured scoring
criteria [30]. To mitigate self-evaluation bias, we employ
Claude 4.5 Sonnet (temperature =0.3) as an independent
Judge. The fixed scoring categories and explicit criteria in
our PF/KR evaluation scheme further constrain subjective
variation. The evaluation covers 320 instances (8 models ×
4 scenarios ×10 iterations), yielding 640 evaluation profiles
(PF + KR per instance).
4 Results and Analysis
This section analyzes 320 experiments (8 models ×4 cases ×
10 iterations) conducted under strict LOO control. Appendix
Figure 6 illustrates a representative comparison between a
manually curated ground-truth attack graph and an LLM-
generated counterpart; the best-scoring generated graphs for
all four cases are provided in Appendix Figures 7–10.
4.1 Inference Capability on Unknown Threats
(Answering RQ1)
The main hypothesis of GARAGE is that attack paths for
unknown threats can be reconstructed bytransferringknowl-
edge from other vehicle attack cases, even when direct at-
tack knowledge for the target vehicle is absent from the
KB. We conducted 320 experiments across 8 models under
the LOO condition. Proprietary models (Claude 4.5 Son-
net, GPT 5.2, Gemini 3.0 Flash) achieved a mean PF of
59.0/100 (64.8/55.9/56.3; KR 64.1), while Open-Weight
models (Qwen3-Coder, DeepSeek-V3.2, Llama-4, GPT-OSS)
achieved a mean PF of 39.2/100 (KR 48.8).
However, these aggregate scores alone cannot explain the
inter-model and inter-case performance variations. As a RAG-
based system, GARAGE’s performance can be evaluated
across two dimensions: (1) theRetrievalstage, defined by the
availability of transferable references in the KB; and (2) the
Generationstage, representing the LLM’s accuracy in com-
posing retrieved knowledge into executable attack paths. Sec-
tion 4.1.1 analyzes model- and case-level patterns from the
heatmap (Figure 2), and Section 4.1.2 extends this analysis
from a tactical perspective (Figure 3).
4.1.1 Model and Case Analysis
We analyze the heatmap patterns based on four factors:knowl-
edge granularity, structural complexity, compositional rea-
soning ability, and RAG augmentation scope.
Observation 1: Knowledge granularity determines trans-
fer success.BMW i3 achieved the highest scores across
all 8 models (55.0–83.9), whereas Jeep Cherokee yielded the
lowest (17.5–64.0). To investigate this variance, we analyzed
6

BMW Jeep Mercedes Nissan
Test CaseClaude Sonnet 4.5
Gemini 3.0 Flash
GPT 5.2
Qwen3-Coder
DeepSeek-V3.2
(Thinking)
Llama-4
Maverick
GPT-OSS-120B
GPT-OSS-20BModel83.7 59.0 73.0 43.6
65.8 64.0 57.3 38.3
83.9 40.6 51.9 47.2
69.0 31.0 46.8 46.6
55.0 20.6 48.6 43.6
54.9 20.6 43.1 45.9
63.8 17.5 34.5 23.6
55.0 17.6 21.2 25.2Path Finding (PF) Score
BMW Jeep Mercedes Nissan
Test CaseClaude Sonnet 4.5
Gemini 3.0 Flash
GPT 5.2
Qwen3-Coder
DeepSeek-V3.2
(Thinking)
Llama-4
Maverick
GPT-OSS-120B
GPT-OSS-20BModel78.0 68.3 70.0 65.6
71.9 65.5 61.9 50.5
71.4 41.5 62.2 61.9
69.6 38.9 58.5 41.5
68.2 36.2 62.3 51.3
66.4 35.5 50.9 41.9
67.6 34.7 54.3 35.3
64.2 29.6 39.1 30.6Knowledge Reconstruction (KR) Score
020406080100
Score
020406080100
Score
RQ1: Model × Case Performance (n=10)Figure 2: PF/KR Score Heatmap across 8 models and 4 attack cases. Proprietary models (top three rows) consistently occupy
the higher score bands, the BMW column yields the highest scores across all models, and the Jeep column exhibits the largest
variance.
Table 3: KB Match Rate and mean PF score by attack case.
Higher KB Match Rate—reflecting GT definitions covered
by existing KB threat patterns—correlates with higher PF.
Case KB Match Mean PF GT Examples
Rate (%) (key_entities)
BMW88.866.4 USB Interface, Central
Gateway, CAN Bus
Benz53.447.1Code Execution ,
Privilege
Escalation
Jeep38.833.9/fs/mmc0 ,cmcioc.bin ,
0xF0 0x02
Nissan35.439.2CVE-2017-7932 ,
libevo_stack.so ,
dnscat2
thematch_status labels assigned by the Judge’s KR mod-
ule. TheKB Match Rate(the ratio of FULL or SEMANTIC
verdicts) represents the overlap between the GT abstraction
level and KB coverage. A higher rate indicates that the GT is
defined at the tactical-pattern level, where existing KB threat
patterns provide sufficient references. A lower rate indicates
that the GT requires target-specific implementation details
not covered by cross-vehicle knowledge.
These results indicate two key findings. First, GARAGE is
most effective for threat modeling defined at the tactical-
pattern level. Second, in high KB Match Rate environments
(BMW, 88.8%), performance gaps between models narrow.
For example, GPT-OSS-20B (PF 55.0) achieved performance
comparable to proprietary models. Notably, despite similar
KB Match Rates between Nissan (35.4%) and Jeep (38.8%),
PF scores differed (39.2 vs. 33.9), indicating that structural
complexity (Observation 2) and model reasoning ability (Ob-
servations 3–4) also significantly affect performance.Table 4: Jeep Cherokee: Per-model PF scores and step-level
V ALID rates. The SPI/firmware step serves as the key dis-
criminator between Proprietary and Open-Weight models.
Model Jeep PF V ALID Rate SPI/FW Persist.
Gemini64.0 55% 4/100/10
Claude59.0 46% 5/100/10
GPT 40.6 5% 0/10 0/10
Qwen3 31.0 8% 0/10 0/10
DeepSeek 20.6 8% 0/10 0/10
Llama 20.6 0% 0/10 0/10
OSS-20B 17.6 0% 0/10 0/10
OSS-120B 17.5 0% 0/10 0/10
Observation 2: Model capability divergence under refer-
ence sparsity.The Jeep Cherokee case exhibited a substan-
tial performance gap between Proprietary and Open-Weight
models, with the SPI-based V850 firmware flashing step as
the primary bottleneck.
This step involves hardware-dependent lateral movement
not observed in other KB cases. Claude and Gemini suc-
cessfully inferred paths by combining ATM techniques (e.g.,
ATM-T0068 Firmware Manipulation) with SPI protocol
references. For instance, Claude generated “Reflash V850
firmware via compromised SPI,” adapting generic firmware
techniques to the target architecture (V ALID 4–5/10). All
other models failed at this reasoning task. The Persistence
step, which relies on target-specific filesystem structures
(/fs/mmc0 ), recorded 0/10 V ALID across all models, confirm-
ing that knowledge granularity constrains the upper bound of
model performance.
Observation 3: Architecture recognition and cascade fail-
ure.The three Proprietary models recorded similar scores
7

Table 5: Jeep Cherokee step-level evaluation for Proprietary
models. V = V ALID, P = PARTIAL, B = BROKEN (counts
out of 10 iterations).
Step Phase Gemini Claude GPT
1 Reconnaissance10/0/09/0/1 1/2/7
2 Initial Access10/0/09/0/1 1/0/9
3 HU Compromise (RCE) 7/3/0 4/4/2 0/5/5
4 Persistence 0/7/3 0/5/5 0/1/9
5 Lateral Movement (FW)4/1/55/0/5 0/1/9
6 Command Translation5/0/55/0/5 0/1/9
7 Steering Control 3/2/5 0/5/5 1/2/7
8 Braking/Engine 5/2/3 5/1/4 1/3/6
Table 6: Benz case: V ALID rates for critical steps across two
parallel attack paths (Head Unit and T-Box).
Model V ALID HU IA HU CAN T-Box IA T-Box CAN
Rate -HMI -D
Claude64%10/1010/109/108/10
Gemini 48% 9/10 6/10 9/10 4/10
GPT 41% 6/101/10 10/105/10
on BMW (Claude 83.7, GPT 83.9, Gemini 65.8) but diverged
sharply on Jeep (Gemini 64.0, Claude 59.0, GPT 40.6). GPT
dropped 43.3 points on Jeep despite achieving the overall
highest score on BMW, because Jeep’s GT requires system-
specific protocols (D-Bus, SPI) that its generic-pattern strat-
egy could not handle.
GPT’s performance drop originated in the reference-rich
early steps (Steps 1–2): the model applied generic HTTP/TCP
80/443 patterns in 9/10 iterations instead of the D-Bus proto-
col (Port 6667) specified in the GT. Due to PF’s cascade fail-
ure property, this initial error nullified all subsequent scores.
In contrast, Gemini and Claude achieved 9–10/10 V ALID
on Steps 1–2 and maintained 4–5/10 V ALID on sparse inter-
mediate steps by formulating alternative mechanisms. This
outcome indicates thataccurate target architecture recogni-
tion is a prerequisite for overall path quality, and that archi-
tecture adaptation ability varies significantly even among Pro-
prietary models (cross-case SD: GPT 16.7 vs. Gemini 10.9).
Observation 4: Structural reasoning—parallel vs. sequen-
tial paths.On the Benz case, Claude 4.5 Sonnet scored 73.0,
a gap of 15.7+ points over the second-tier group (Gemini 57.3,
GPT 51.8). The Benz scenario involvestwo independent at-
tack chains(T-Box and Head Unit) converging at the CAN
bus.
Claude maintained high V ALID rates across both paths.
GPT, however, generated the T-Box path successfully (10/10)
but failed to construct the Head Unit path (1/10). Interest-
ingly, this structural difficulty caused aperformance re-
versalamong lower-tier models (Figure 2): Llama scored
Nissan 45.9 vs. Benz 43.1, and GPT-OSS-20B scored Nis-Table 7: PF–KR gap across all 8 models. The gap serves as
a proxy for compositional reasoning—the ability to convert
retrieved knowledge into executable attack paths.
Tier Model PF KR Gap
Proprietary Claude 4.5 Sonnet 64.8 70.5 +5.7
Proprietary Gemini 3.0 Flash 56.3 62.4 +6.1
Proprietary GPT 5.2 55.9 59.3 +3.4
Open-Weight Qwen3-coder 48.3 52.1 +3.8
Open-Weight DeepSeek-V3.2 42.0 54.5 +12.5
Open-Weight Llama-4 41.1 48.7 +7.6
Open-Weight GPT-OSS-120B 34.9 48.0 +13.1
Open-Weight GPT-OSS-20B 29.8 40.9 +11.1
san 25.2 vs. Benz 21.2. Because Nissan employs asequential
single-pathattack chain (via Bluetooth), lower-tier models
processed this linear structure more effectively than Benz’s
parallel construct, despite Nissan’s lower KB Match Rate
(35.4% vs. 53.4%). This finding highlights that model per-
formance depends not only on capability tier but also onthe
structural complexity of the attack paths.
Observation 5: RAG augmentation effect and PF–KR gap.
The PF–KR gap ranges from +3.4 to +13.1 across models (Ta-
ble 7), indicating a continuous range of compositional reason-
ing ability rather than a strict tier-based separation. This gap
arises because KR credits individual step recognition indepen-
dently of chain continuity, while PF’s cascade failure penal-
izes broken causal chains. Most Open-Weight models exhibit
large gaps (+7.6 to +13.1), indicating sufficient entity-level
knowledge to recognize attack components but insufficient
compositional reasoning to chain them into valid sequences.
Notably, Qwen3-coder (+3.8) achieves Proprietary-level com-
positional efficiency despite being an Open-Weight model.
As KB Match Rate decreases, these inter-model reasoning
differences re-emerge as the dominant performance factor.
Summary.Based on the five observations, GARAGE’s
LOO performance primarily depends on four interacting
factors: (1)knowledge granularity—the intersection of GT
abstraction and KB coverage determines transfer success
(Obs. 1); (2)structural complexity—parallel vs. sequential
path topology affects reasoning difficulty (Obs. 4); (3)model-
specific compositional reasoning—capability differences in
architecture recognition and hardware-dependent inference
become apparent (Obs. 2, 3); and (4)RAG augmentation
scope—the degree to which retrieved knowledge is converted
into executable paths (Obs. 5). The logical completion rate
across all 320 experiments was 55.6% (178/320), correlating
strongly with knowledge granularity.
8

0.0 0.2 0.4 0.6 0.8 1.0
Coverage (Recall)ExecutionCredential AccessDiscoveryExfiltrationManipulate EnvironmentCommand and ControlDefense EvasionPersistenceLateral MovementPrivilege EscalationInitial AccessAffect Vehicle FunctionTactic
0.150.190.240.300.310.330.410.600.660.820.950.99RQ1: Tactic Coverage (Overall)Figure 3: Auto-ISAC Kill Chain tactic coverage. Initial Ac-
cess (95%) and Affect Vehicle Function (99%) show high
coverage, while mid-chain steps (Execution 15%, C2 33%)
drop sharply. This pattern is consistent across all 8 models.
4.1.2 Kill Chain Coverage Analysis
Coverage is computed as the mean rate at which each tac-
tic was scored V ALID or PARTIAL across all 320 experi-
ments. The resulting distribution reveals distinct differences
in transferability: Initial Access (95%), Privilege Escalation
(82.5%), and Affect Vehicle Function (98.8%) exhibit high
coverage because their underlying entry vectors, escalation
methods, and CAN-bus-based control principles share com-
mon patterns across vehicle architectures. In contrast, Exe-
cution (15%), C2 (32.5%), and Persistence (60%) recorded
significantly lower coverage due to their reliance on target-
specific implementations. For example, the Execution tactic
inC3(Nissan) necessitates triggering a buffer overflow in
libevo_stack.so via the HFP Bluetooth profile—a library
exclusive to Nissan’s IVI platform that generic references can-
not anticipate. Similarly, C2 in C1(Jeep) relies on D-Bus IPC
communicating via NavTrailService on port 6667, which
is unique to Chrysler’s Uconnect system. Identifying such
specific elements requires reverse-engineering artifacts that
are inherently absent from cross-vehicle KBs. This pattern
demonstrates the limitations of automated knowledge transfer.
For steps dependent on target-specific firmware or software
internals, relying solely on knowledge retrieval is insufficient,
and reverse engineering or dynamic analysis via Human-in-
the-loop validation remains necessary.
RQ1 Conclusion:GARAGE demonstrates that under
the LOO condition, it can reconstruct valid attack paths
(mean PF 59.0/100 for Proprietary models) by transfer-
ring knowledge from other vehicle attack cases and CTI.
Performance depends on four key factors: knowledge
granularity, structural complexity, compositional reason-Table 8: PF score consistency metrics across 8 models (10
iterations×4 cases = 40 experiments per model).
Model Mean (PF) Std CV (%) 95% CI
Claude 4.5 Sonnet 64.8 19.530.1±6.0
Gemini 3.0 Flash 56.3 17.1 30.3±5.3
GPT 5.2 55.9 20.9 37.5±6.5
Qwen3-coder 48.3 18.3 37.9±5.7
DeepSeek-V3.2 42.0 18.6 44.3±5.8
Llama-4 41.1 18.0 43.8±5.6
GPT-OSS-120B 34.9 22.4 64.3±7.0
GPT-OSS-20B 29.8 18.2 61.2±5.6
ing, and RAG augmentation scope. GARAGE is most
effective fortactical-pattern-level threat modeling;
implementation details present fundamental limitations
that require Human-in-the-loop approaches.
4.2 Consistency Analysis (Answering RQ2)
To verify whether the performance differences observed in
RQ1 are reproducible, all experiments were repeated 10 times
and analyzed using the Coefficient of Variation (CV) and
95% Confidence Interval (CI). Figure 4 visualizes the per-
case PF distributions for each model: Proprietary models
exhibit compact interquartile ranges concentrated in the upper
score bands, while several Open-Weight models display wider
spreads with data points frequently clustering near zero—
illustrating the impact of cascade failures analyzed below.
Proprietary models exhibited a mean CV of 32.6%, substan-
tially lower than Open-Weight models (50.3%). The overall
mean 95% CI was ±5.9 points, narrower than inter-model
differences reported in RQ1 (e.g., Claude 64.8 vs. GPT 55.9,
∆≈9), confirming that the performance hierarchy is statisti-
cally meaningful and reproducible. The elevated overall CV
(43.7%) reflects PF’s bimodal score distribution from cas-
cade failures rather than true instability. Specifically, PF’s
cascade failure property—where a single broken early step
nullifies all subsequent scores—produces two distinct score
clusters: high scores when early steps succeed (allowing the
full chain to be evaluated) and near-zero scores when they
fail (invalidating the entire path). This bimodality artificially
increases variance-based metrics like CV without indicating
actual output inconsistency; accordingly, the 95% CI serves
as the more appropriate reliability indicator.
RQ2 Conclusion:GARAGE exhibits reproducibility
with a mean 95% CI of ±5.9 points. Proprietary models
(CV≈33%) demonstrate higher consistency than Open-
Weight models (CV ≈50%). The performance hierarchy
from RQ1 is a stable, reproducible phenomenon.
9

Jeep Mercedes Nissan BMW020406080100PF Score
Claude 4.5
=64.8, CV=30%
Jeep Mercedes Nissan BMW
Gemini 3.0
=56.3, CV=30%
Jeep Mercedes Nissan BMW
GPT 5.2
=55.9, CV=37%
Jeep Mercedes Nissan BMW
Qwen3-Coder
=48.3, CV=38%
Jeep Mercedes Nissan BMW020406080100PF Score
DeepSeek-V3.2
(Thinking)
=42.0, CV=44%
Jeep Mercedes Nissan BMW
Llama-4 Maverick
=41.1, CV=44%
Jeep Mercedes Nissan BMW
GPT-OSS-120B
=34.9, CV=64%
Jeep Mercedes Nissan BMW
GPT-OSS-20B
=29.8, CV=61%
RQ2: Score Distribution across 10 Iterations by ModelFigure 4: Distribution of PF scores across 10 independent trials for each model and case. Proprietary models (Claude, Gemini,
GPT) exhibit narrower interquartile ranges, indicating higher consistency, while Open-Weight models show wider variance.
Table 9: Per-analysis cost and performance metrics. Cost is
computed from measured token counts and API pricing (as of
Feb. 2026). Efficiency is defined as PF score per milli-dollar.
Model Tier In Tok Out Tok Cost PF Eff
($/run) (PF/m$)
Claude 4.5 Sonnet Proprietary 5,942 4,990 $0.093 64.8 700
GPT 5.2 Proprietary 5,946 4,005 $0.066 55.9 841
Gemini 3.0 Flash Proprietary 5,837 1,867 $0.009 56.36,614
Qwen3-coder Open-Weight 5,704 2,548 $0.004 48.3 12,712
DeepSeek-V3.2 Open-Weight 3,756 3,195 $0.001 42.0 29,007
Llama-4 Open-Weight 5,786 1,783 $0.002 41.1 21,217
GPT-OSS-120B Open-Weight 5,836 3,110 $0.001 34.9 42,578
GPT-OSS-20B Open-Weight 5,745 3,145 $0.001 29.8 48,560
4.3 Cost-Performance Trade-off (Answering
RQ3)
To assess practical deployability, we analyzed the trade-off
between per-analysis cost and performance across 8 models.
Costs were calculated from actual token counts measured
from experimental prompts and outputs.
The cost-performance distribution segments into two tiers
(Figure 5).Within the Proprietary tier, cost and perfor-
mance do not scale linearly: Gemini 3.0 Flash achieved 87%
of Claude’s performance at 1/10 the cost ($0.009 vs. $0.093),
yielding 9.4 ×higher cost efficiency.In the Open-Weight
tier, Qwen3-coder (PF 48.3, $0.004) achieved the highest
performance at 1/17–1/24 the cost of Claude/GPT, while
DeepSeek-V3.2 (PF 42.0, $0.001) maintained meaningfulperformance at minimal cost. Smaller models (GPT-OSS)
have the lowest API costs ( ≤$0.001) but PF scores of 30–35;
all Open-Weight models can be deployed on-premise, though
GPU infrastructure costs would apply.
RQ3 Conclusion:GARAGE’s cost-performance trade-
off is clearly stratified by model tier. Gemini 3.0 Flash
provides the optimal cost-efficiency among Proprietary
models (1/10 the cost of Claude at 87% performance),
while Qwen3-coder achieves the best cost-effectiveness
among Open-Weight models.
5 Discussion and Limitation
5.1 Discussion
Key Findings.Our 320 LOO experiments across four real-
world automotive attacks and eight LLMs reveal three key
findings. First,knowledge-transfer-based inference on un-
known threats is practically feasible: even with all direct
attack knowledge removed, proprietary models reconstructed
valid attack paths with a mean PF of 59.0/100. This transfer
occurs through two complementary pathways—explicit re-
trieval from the domain-specific KB and implicit recall from
the LLM’s parametric knowledge. While our ablation study
shows that both pathways achieve comparable PF scores, only
10

$1.00 $10.00 $100.00
Cost per Analysis (milli-dollars, log scale)203040506070PF Score (Practical Feasibility)Claude 4.5
Gemini 3.0 GPT 5.2
Qwen3-Coder
DeepSeek-V3.2Llama-4 Maverick
GPT-OSS-120B
GPT-OSS-20BRQ3: Cost vs. Performance Trade-off
(Each point = model average across 40 experiments)
Model Tier
Proprietary
Open-WeightFigure 5: Cost vs. Performance trade-off across eight
LLMs (log scale). Gemini 3.0 Flash achieves the optimal
cost-performance balance among proprietary models, while
Qwen3-coder represents the best value among open-weight
alternatives.
the KB-grounded pathway provides formal evidence trace-
ability, a prerequisite for regulatory compliance (seeRole of
Retrieval Modality). Second,performance is structurally
constrained by knowledge granularity: the strong correla-
tion between KB Match Rate and PF (BMW 88.8% →66.4
vs. Jeep 38.8% →33.9) indicates that the abstraction level at
which threats are defined sets an upper bound on automated
inference. This constraint persists regardless of model capabil-
ity. Tactical-pattern-level threats are highly suitable for auto-
mated reasoning, whereas implementation-detail-level threats
establish a structural boundary for knowledge transfer. Third,
the impact of model selection is highly context-dependent:
the best-worst performance gap (Claude 64.8 vs. GPT-OSS-
20B 29.8) far exceeds the 95% CI ( ±5.9), yet this variance is
amplified in scenarios characterized by low knowledge granu-
larity and high structural complexity. This suggests that the
choice of model must be tailored to the specific characteristics
of the threat type.
Practical Implications.As an automated tool supporting
TARA compliance under UN R155 and ISO/SAE 21434,
GARAGE can systematically generate initial threat scenario
candidates. The high Kill Chain coverage in Initial Access
(95.0%) and Affect Vehicle Function (98.8%), as reported in
Section 4.1.2, indicates that the system can cover entry points
and impact scenarios that analysts might otherwise overlook.
Conversely, the lower coverage in Execution (15.0%) and C2
(32.5%) confirms that GARAGE is most effective as anexpert
assistantwithin a Human-in-the-loop workflow, automating
initial threat exploration while leaving technical verification
and prioritization to domain experts.
In the context of the ISO/SAE 21434 TARA workflow,
this boundary of automation maps directly to specific pro-cess stages: GARAGE facilitatesthreat scenario enumer-
ationandinitial attack path generation, corresponding to
high-coverage tactics. However,attack feasibility validation
requires implementation-level knowledge of the target system
(involving low-coverage tactics such as Execution and C2)
and therefore remains a manual expert task.
For deployment, our RQ3 results provide quantitative cri-
teria for model selection: Claude 4.5 Sonnet (PF 64.8) is
suitable for accuracy-critical regulatory compliance; Gemini
3.0 Flash achieves 87% of Claude’s performance (PF 56.3
vs. 64.8) at a fraction of the cost, making it practical for
large-scale analyses; and Qwen3-Coder is an alternative for
air-gapped on-premise environments, though total cost of own-
ership (GPU infrastructure) should be considered beyond API
pricing.
Role of Retrieval Modality.An ablation study compar-
ing Graph-only, Vector-only, and No-RAG conditions against
the full Hybrid configuration showed negligible PF varia-
tions ( ∆PF<1.2 across all conditions). This indicates that
knowledge granularity—the quality and coverage of the
underlying KB—is the dominant factordictating inference
performance, rather than the specific retrieval modality. The
KB provides the foundational knowledge that all retrieval
strategies draw upon; accordingly, the core contributions of
this work are the domain-specific KB construction pipeline,
the dual-metric evaluation framework, and the empirical find-
ings on knowledge transfer boundaries.
Nevertheless, the domain-specific KB and RAG pipeline
are indispensable for practical deployment. While the No-
RAG condition attains comparable PF scores by relying on
parametric memory, its outputs lackformal evidence trace-
ability—the ability to trace each attack step back to a spe-
cific CVE, CTI report, or KG node. Under UN R155 and
ISO/SAE 21434, outputs without formal grounding cannot
serve as auditable compliance evidence. By design, the RAG-
based architecture grounds generated attack paths in retrieved
source artifacts, thereby providing the traceability demanded
by regulatory standards. Quantifying the grounding rate of
each generated step remains future work.
5.2 Limitation
L1: Limited GT Scale and Subjectivity.The evaluation re-
lies on four public automotive attack cases that are among
the most thoroughly documented in automotive cybersecu-
rity. However, the small sample size limits statistical gen-
eralizability. Because GT construction inevitably involves
researcher interpretation, three domain experts independently
reconstructed each GT by referencing ATM official exam-
ples ( C1,C4) or technical white papers ( C2,C3), resolving
any disagreements through consensus. We further mitigated
subjective bias by evaluating structural coherence and practi-
cal feasibility rather than demanding exact matches, thereby
accepting semantically equivalent alternative paths.
11

L2: LLM-as-a-Judge Bias.Utilizing Claude 4.5 Sonnet
as both a target model and an evaluator introduces potential
model-family bias. To mitigate this risk, we implemented four
distinct safeguards: (i) complete session isolation between
generation and evaluation phases; (ii) dual independent met-
rics (PF/KR); (iii) uniform evaluation criteria across all tested
models; and (iv) a structured output format requiring all mod-
els to produce attack graphs in an identical JSON schema,
eliminating stylistic preferences as a confounding factor. Em-
pirically, no systemic bias was observed; for instance, Gemini
(64.0) outperformed Claude (59.0) in the Jeep case. To further
validate judge reliability, we conducted cross-judge evalua-
tion on two cases ( C1,C4;N=160) using GPT-5.2 and Gemini
3.0 Flash. As detailed in Table 10, pairwise Spearman correla-
tions indicated strong agreement ( ρ= 0.74–0.84, p<0.001 ),
confirming that model rankings remain consistent regardless
of the judge model employed.
Table 10: Cross-judge agreement ( N=160, two cases com-
bined). All correlations are significant atp<0.001.
Judge Pair PFρKRρ
Claude vs GPT 0.76 0.77
Claude vs Gemini 0.75 0.74
GPT vs Gemini 0.82 0.84
L3: LLM Hallucination.Across all 320 experiments,
76.9% of the generated attack graphs contained at least one
hallucinated element (averaging 1.74 per experiment), with
hallucination rates ranging from 52.5% (GPT 5.2) to 95.0%
(Qwen3-Coder). Common hallucination types include fab-
ricated CVE identifiers, non-existent protocol names, and
generic network details not grounded in the input vehicle
specification. The mean Knowledge Precision score of 9.81
out of 20 indicates that approximately half of the generated
knowledge elements lack direct traceability to the retrieved
context. This result supports the need for a Human-in-the-loop
workflow: while GARAGE effectively generates candidate
attack paths, expert review remains essential to filter halluci-
nated artifacts before operational use.
L4: LOO vs. True Zero-Day.The LOO experimental de-
sign simulates unknown threats by systematically excluding
the target case; however, true zero-day scenarios may involve
entirely novel attack surfaces that inherently lack any transfer-
able references within the KB. Because our LOO condition
ensures the availability of at least three other authoritative
vehicle attack cases, it represents a controlled evaluation of
knowledge transfer capability rather than a strict worst-case
assessment.
L5: Vehicle Architecture Input Requirements.Our ex-
periments reconstruct vehicle architecture specifications from
publicly available technical reports and white papers, repre-
senting the maximum detail recoverable from open sources.
In practice, OEMs rarely disclose complete E/E architecturedetails or SBOMs, limiting specification fidelity for external
analysts. Furthermore, validating operational exploitability
would require physical testbed access. GARAGE is therefore
designed to generateplausible threat scenariosfor TARA
prioritization rather than confirmed exploit chains.
L6: Generalizability Across Architectures.Our evalua-
tion is based on vehicle architectures from the 2014–2018 era.
Modern software-defined vehicles incorporate stronger secu-
rity controls (e.g., HSMs, SecOC, and Automotive Ethernet)
and may exhibit different transferability characteristics. How-
ever, the framework can be extended to newer architectures
by updating the KB and threat taxonomy.
6 Related Work
Attack Graphs and Automotive Domain Application.AGs
are a key tool for modeling multi-step attack paths. Early
logic-based frameworks like MulV AL [18] inferred attack
paths from formalized vulnerability and configuration in-
puts. However, these approaches rely on predefined static
rules, creating a knowledge acquisition bottleneck that lim-
its integration of unstructured threat intelligence. Research
applying AGs to the automotive domain primarily focuses
on automating the TARA process mandated by the ISO/SAE
21434 standard. Early studies such as Salferet al. [20] and
GAPP [22] used AGs to analyze architectural vulnerabilities
in on-board networks. Subsequent tools like ThreatGet [5]
supported ISO/SAE 21434 compliance by automating asset-
driven attack tree analysis. Separately, dynamic AG genera-
tion algorithms based on ontology reasoning were proposed
for Internet of Vehicles environments. Nevertheless, these
approaches often focus on vehicle-level assessment, show-
ing limitations in addressing the ’function-level’ TARA re-
quired by ISO/SAE 21434. Furthermore, the unique protocols
(e.g., CAN, V2X) and architectures of vehicles create a do-
main knowledge gap that general-purpose IT tools cannot
capture. To address this, domain-specific knowledge bases
such as CAKG [27] (a vehicle-specific cybersecurity knowl-
edge graph) and the Acti dataset (manually annotated auto-
motive CTI reports) [26] have been introduced. Despite these
contributions, they do not provide an end-to-end solution for
automatically linking this knowledge to AG generation for
TARA.
CTI-based Knowledge Graph Construction and LLMs.
Extracting structured knowledge from unstructured CTI has
been extensively studied. Standardized ontologies like UCO
[23] provided a schema for knowledge representation. Early
NLP-based studies, such as Open-CyKG [10] and AttacKG
[15] employed specially trained neural models to extract enti-
ties and relations from CTI reports. However, these methods
were inflexible, overfit to specific datasets and schemas, and
incurred high retraining costs to adapt to new threats. Re-
cent advancements in LLMs have provided flexible alterna-
tives. LLMs use pre-trained knowledge to parse complex CTI
12

text without task-specific fine-tuning. LLM-TIKG [8] and
CTINexus [4] proposed flexible and adaptable CTI-to-KG
pipelines using LLM in-context learning (ICL). Furthermore,
CTIKG [9] introduced multi-agent and dual-memory architec-
tures to mitigate LLM hallucinations. Together, these studies
show that LLMs can reduce the knowledge acquisition bot-
tleneck in CTI analysis. Benchmarks such as CTI-HAL [6]
further enable systematic performance validation.
LLM-driven Attack Graph Automation.Recent work in-
tegrates LLMs directly into the AG generation logic, moving
beyond CTI-to-KG preprocessing. AttacKG+ [28] enhances
AG construction by utilizing LLMs to infer dependencies
between attack nodes. This trend has led to multimodal ap-
proaches like MM-AttacKG [29], which analyzes both text
and images (e.g., threat diagrams, screenshots) from CTI re-
ports. Despite these advancements, a critical gap remains
between these LLM-based methods and the specific TARA
requirements of the automotive domain.
Specifically, the latest LLM-AG frameworks are domain-
agnostic. They lack the built-in knowledge required to model
the unique constraints of an automotive E/E architecture or
the specific vulnerabilities of the CAN bus. Conversely, auto-
motive domain-specific research, such as CAKG [27] and the
Acti dataset [26], focuses primarily on building foundational
knowledge bases but does not extend to end-to-end attack
graph generation.
In summary, no existing framework effectively combines
the parsing flexibility of LLMs with automotive-specific CTI
processing to automatically generate TARA-ready AGs. This
paper presents GARAGE to address this gap. GARAGE is an
end-to-end framework that uses LLM-based CTI extraction,
a domain-specific automotive cybersecurity KB, and RAG-
based inference to generate vehicle-level attack graphs. We
evaluate this framework using a dual-metric approach via
empirical analysis across eight LLMs.
7 Conclusion
This paper presented GARAGE, an end-to-end framework that
transforms publicly available automotive CTI into a domain-
specific KB and automatically generates vehicle-level attack
graphs via RAG-based inference. The KB comprises 12,786
CVEs and 140 security events, structured following STIX 2.1
and ATM. We evaluated the generated AGs using a dual-
metric framework (PF and KR).
Through 320 LOO experiments across four real-world at-
tack cases and eight LLMs, we demonstrated cross-vehicle
knowledge transfer to unseen architectures. Specifically, pro-
prietary models reconstructed valid attack paths with a mean
PF of 59.0, even without direct knowledge of the target at-
tacks. Based on kill chain analysis andknowledge granularity,
we identified a clear automation boundary: automated gen-
eration is effective for tactical-pattern-level threat scenarios,
whereas implementation-detail-level stages require humanexpert intervention. An ablation study confirmed that KB
quality and coverage, rather than retrieval modality, is the
dominant performance factor, underscoring the centrality of
the KB construction pipeline. Our primary contributions are
therefore the domain-specific KB construction pipeline, the
evaluation methodology, and the empirical findings regarding
the automation boundary.
These results show that GARAGE serves as a practical
TARA support tool within a human-in-the-loop workflow.
Additionally, our findings provide cost-performance guidance
for deploying various LLM tiers. Future work will extend
the evaluation to software-defined vehicle architectures and
explore integration with real-time CTI feeds.
References
[1]Auto-ISAC. Auto-isac atm. https://atm.automoti
veisac.com/. Accessed: 25 Feb 2026.
[2]AutoSec. Autosec-timeline. https://autosec-tim
eline.delikely.eu.org/. Accessed: 25 Feb 2026.
[3]Black Hat. Remote exploitation of nissan leaf. https:
//i.blackhat.com/Asia-25/Asia-25-Evdokim
ov-Remote-Exploitation-of-Nissan-Leaf.pdf .
Accessed: 25 Feb 2026.
[4]Yutong Cheng, Osama Bajaber, Saimon Amanuel
Tsegai, Dawn Song, and Peng Gao. Ctinexus: Automatic
cyber threat intelligence knowledge graph construction
using large language models. In2025 IEEE 10th Euro-
pean Symposium on Security and Privacy (EuroS&P),
pages 923–938. IEEE, 2025.
[5]Sebastian Chlup, Korbinian Christl, Christoph Schmit-
tner, Abdelkader Magdy Shaaban, Stefan Schauer, and
Martin Latzenhofer. Threatget: towards automated at-
tack tree analysis for automotive cybersecurity.Infor-
mation, 14(1):14, 2022.
[6]Sofia Della Penna, Roberto Natella, Vittorio Orbinato,
Lorenzo Parracino, and Luciano Pianese. Cti-hal: A
human-annotated dataset for cyber threat intelligence
analysis. In2025 IEEE European Symposium on Secu-
rity and Privacy Workshops (EuroS&PW), pages 69–78.
IEEE, 2025.
[7]Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha
Metropolitansky, Robert Osazuwa Ness, and Jonathan
Larson. From local to global: A graph rag ap-
proach to query-focused summarization.arXiv preprint
arXiv:2404.16130, 2024.
[8]Yuelin Hu, Futai Zou, Jiajia Han, Xin Sun, and Yilei
Wang. Llm-tikg: Threat intelligence knowledge graph
13

construction utilizing large language model.Computers
& Security, 145:103999, 2024.
[9]Liangyi Huang and Xusheng Xiao. Ctikg: Llm-powered
knowledge graph construction from cyber threat intel-
ligence. InFirst Conference on Language Modeling,
2024.
[10] Sarhan Injy and Spruit Marco. Open-cykg: An open
cyber threat intelligence knowledge graph.Knowledge-
Based Systems, 233(1):1–13, 2021.
[11] IOActive. Jeep hacking. https://www.ioactive.c
om/wp-content/uploads/pdfs/IOActive_Remote_
Car_Hacking.pdf. Accessed: 25 Feb 2026.
[12] KeenLab. Experimental security assessment of bmw
cars. https://keenlab.tencent.com/en/whitepa
pers/Experimental_Security_Assessment_of_B
MW_Cars_by_KeenLab.pdf. Accessed: 25 Feb 2026.
[13] KeenLab. Mercedes benz security research report. ht
tps://keenlab.tencent.com/en/whitepapers/M
ercedes_Benz_Security_Research_Report_Fina
l.pdf. Accessed: 25 Feb 2026.
[14] KIA. Kia open source software notice. http://webm
anual.kia.com/STD_GEN5_WIDE/AVNT/EU/Italia
n/opensourcesoftwarenotice.html . Accessed: 25
Feb 2026.
[15] Zhenyuan Li, Jun Zeng, Yan Chen, and Zhenkai Liang.
Attackg: Constructing technique knowledge graph from
cyber threat intelligence reports. InEuropean Sympo-
sium on Research in Computer Security, pages 589–609.
Springer, 2022.
[16] Hyundai MOBIS. Hyundai mobis open sources. https:
//www.mobis.com/kr/tech/rnd.do . Accessed: 25
Feb 2026.
[17] OpenRouter. Openrouter api. https://openrouter
.ai/. Accessed: 25 Feb 2026.
[18] Xinming Ou, Sudhakar Govindavajhala, Andrew W Ap-
pel, et al. Mulval: A logic-based network security ana-
lyzer. InUSENIX security symposium, volume 8, pages
113–128. Baltimore, MD, 2005.
[19] Minh Pham and Kaiqi Xiong. A survey on security
attacks and defense techniques for connected and au-
tonomous vehicles.Computers & Security, 109:102269,
2021.
[20] Martin Salfer.Automotive Security Analyzer for Ex-
ploitability Risks. Springer, 2024.[21] Bhaskarjit Sarmah, Dhagash Mehta, Benika Hall, Ro-
han Rao, Sunil Patel, and Stefano Pasquali. Hybridrag:
Integrating knowledge graphs and vector retrieval aug-
mented generation for efficient information extraction.
InProceedings of the 5th ACM International Conference
on AI in Finance, pages 608–616, 2024.
[22] Mera Nizam-Edden Saulaiman, Miklos Kozlovszky, and
Akos Csilling. Graph-based automation of threat analy-
sis and risk assessment for automotive security.Infor-
mation, 16(6):449, 2025.
[23] Zareen Syed, Ankur Padia, M Lisa Mathews, Tim Finin,
Anupam Joshi, et al. Uco: A unified cybersecurity ontol-
ogy. InProceedings of the AAAI Workshop on Artificial
Intelligence for Cyber Security, pages 195–202, 2016.
[24] V olkswagen. V olkswagen third-party licence notes.ht
tps://www.volkswagen-newsroom.com/en/third
-party-licence-notes-38. Accessed: 25 Feb 2026.
[25] Yinghui Wang, Yilong Ren, Zhiyong Cui, and Haiyang
Yu. Proactive security defense: cyber threat intelligence
modeling for connected autonomous vehicles.arXiv
preprint arXiv:2410.16016, 2024.
[26] Yinghui Wang, Yilong Ren, Hongmao Qin, Zhiyong
Cui, Yanan Zhao, and Haiyang Yu. A dataset for cyber
threat intelligence modeling of connected autonomous
vehicles.Scientific Data, 12(1):366, 2025.
[27] Peng Yang, Wang Lijie, Li Yun, Song Xuedong, Wang
Yaxin, and Guo Biheng. Cakg: A framework for cyber-
security threat detection of automotive via knowledge
graph. In2023 8th International Conference on Data
Science in Cyberspace (DSC), pages 221–228. IEEE,
2023.
[28] Yongheng Zhang, Tingwen Du, Yunshan Ma, Xiang
Wang, Yi Xie, Guozheng Yang, Yuliang Lu, and Ee-
Chien Chang. Attackg+: Boosting attack graph con-
struction with large language models.Computers &
Security, 150:104220, 2025.
[29] Yongheng Zhang, Xinyun Zhao, Yunshan Ma, Haokai
Ma, Yingxiao Guan, Guozheng Yang, Yuliang Lu, and
Xiang Wang. Mm-attackg: A multimodal approach to
attack graph construction with large language models.
arXiv preprint arXiv:2506.16968, 2025.
[30] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuo-
han Li, Dacheng Li, Eric Xing, et al. Judging llm-as-
a-judge with mt-bench and chatbot arena.Advances
in neural information processing systems, 36:46595–
46623, 2023.
14

A Appendix
A.1 Ablation Study: Retrieval Modality
Table 11 reports overall PF/KR scores across four retrieval
conditions, and Table 12 provides a per-case breakdown.
Table 11: Ablation study: overall mean PF and KR across
retrieval conditions (320 experiments each).
Condition Mean PF Mean KR
Hybrid (baseline) 46.6 54.5
Vector-only (NO_GRAPH) 47.7 55.2
Graph-only (NO_VECTOR) 47.8 55.2
NO_RAG 47.8 54.9
Table 12: Ablation study: per-case mean PF scores.
Case Hybrid Vector-only Graph-only NO_RAG
Jeep 33.9 40.1 33.2 38.6
MBUX 47.1 46.2 46.7 41.6
Nissan 39.3 40.0 39.8 40.6
BMW 66.4 64.4 71.5 70.6
No condition consistently outperforms the others; per-case
effects are heterogeneous (e.g., MBUX drops under NO_RAG
while BMW improves). Notably, the Hybrid baseline records
a marginally lower overall PF (46.6) than single-modality or
No-RAG conditions (47.7–47.8); however, all differences fall
within ±1.2 points—well below the 95% CI of ±5.9 reported
in RQ2—indicating that these variations are not statistically
meaningful. These results support the Discussion finding that
knowledge granularity, not retrieval modality, is the dominant
performance factor.
A.2 Prompt Templates
This section presents the core prompt templates used in the
KB construction and attack graph generation stages.
A.2.1 Entity Extraction Prompt
The entity extraction prompt assigns the LLM the role of
an automotive cybersecurity analyst and applies three con-
straints: (1)Verbatim Extraction—all entities must be ex-
plicitly grounded in the input text; (2)Negative Instruc-
tion—entities not mentioned in the text must not be created;
and (3)Structural Guidance—12 entity types are defined
with explicit examples to reduce boundary ambiguity. Key
excerpts:
You are an expert automotive cybersecurity analyst
specializing in extracting structured information
from technical texts, incident reports, andvulnerability descriptions.
### Additional Instructions for Accuracy:
- Grounding: Ensure all identified entities are
derived explicitly from the input text.
Do not hallucinate entities not present in the text.
- Standardization: If an entity is referred to
multiple times with slight variations, always
standardize to the exact same name.
- Coreference Resolution: If a pronoun is used,
replace it with the specific entity name it
refers to. Do NOT extract pronouns as entities.
- Conciseness: Entities must be specific noun
phrases (typically 1-5 words). Do NOT extract
full sentences or descriptions of actions.
### Negative Instructions:
- Do NOT extract or infer relationships between
entities.
- Do NOT create entities that are not explicitly
mentioned in the text.
- Do NOT extract generic terms like "device",
"system", "vehicle" unless part of a specific
name (\textit{e.g.}, "Infotainment system").
- Do NOT extract verbs or action phrases
(\textit{e.g.}, "hijack the connection") as entities.
### Entity Definitions (with examples):
- component: (\textit{e.g.}, ’UDS module’, ’BT stack’)
- target_ecu: (\textit{e.g.}, ’Infotainment ECU’)
- vulnerability: (\textit{e.g.}, ’Buffer overflow
(CWE-120)’)
...[snip 9 more types]...
### Output Format:
{ "component": [], "target_ecu": [],
"vulnerability": [], ... }
A.2.2 Relation Extraction Prompt
The relation extraction prompt constrains outputs to nine
predefined relation types (Table 2) and requires that both
source and target entities come verbatim from the previously
extracted entity list (E extracted ):
Extract valid relationships only between the
entities in "Extracted Entities".
Always use entity strings verbatim from that list
for each "source" and "target".
### Allowed Relations
(Each line: Description - Format - Example)
1. HAS_VULNERABILITY
2. IDENTIFIED_BY
3. CONNECTED_TO
4. USES_PROTOCOL
5. DELIVERED_VIA
6. PROVIDED_BY
7. REPORTED_BY
8. AFFECTS_ASSET
9. USED_AGAINST
...[snip per-relation examples]...
If no valid relations exist, return "Relations": [].
15

A.2.3 Rewriting Prompt
The rewriting prompt converts structured data (entities and
relations) into dense, factual natural language paragraphs op-
timized for vector retrieval. It aims to align semantically with
ATM technique descriptions by emphasizing attack mecha-
nisms and impacts:
You are an expert in automotive cybersecurity
documentation. Your task is to rewrite the provided
structured data (Entities and Relations) into a
dense, factual paragraph optimized for vector
retrieval.
### Goal
Create a text that semantically aligns with
Automotive Threat Matrix (ATM) technique
descriptions. Describe the mechanism of the attack
(how it works) and the impact (what it achieves).
### Rules
1. Action-Oriented: Use strong verbs describing
the attack lifecycle (\textit{e.g.}, "intercepted",
"spoofed", "flooded", "extracted", "modified").
2. Mechanism & Impact: Explicitly state HOW the
vulnerability was exploited and WHAT the
consequence was.
3. Keyword Preservation: Keep specific entity
names (ECUs, Protocols, CVEs) exact.
4. Conciseness: Dense, technical summary.
5. Contextual Linking: Connect the ’Tool’ used
to the ’Vulnerability’ exploited and the
’Asset’ affected.
6. Grounding: Do not invent details not present
in the input.
...[snip example and I/O format]...
A.2.4 SFT Data Generation Prompt
The SFT data generation prompt assigns the LLM the role
of a Threat Intelligence Analyst. It instructs the model to
synthesize virtual incident reports for each ATM technique
across five styles with diverse scenarios. To prevent label
leakage, technique names and IDs are explicitly blacklisted
from the generated text:
You are an expert Threat Intelligence Analyst
specializing in automotive cybersecurity.
Your mission is to draft five distinct reports
that illustrate the SAME Auto-ISAC attack
technique through different scenarios.
## Safety & Ethics (MANDATORY)
- Keep tone professional and avoid sensationalism.
## Global Requirements
- All five reports must express the same technique,
each via a distinct scenario, attacker type,
target surface, and network domain.
- Each report must be 3-6 paragraphs.
- Focus on impacts, detection, and mitigations
over exploitation details.
- IMPORTANT: The narrative text MUST NOT containor repeat any of the following terms (to
prevent label leakage): [blocked_terms]
## Output Schema
- "style": one of ["research_lab",
"corporate_advisory", "press_release",
"anon_forum_post", "training_material"]
...[snip 10 more fields]...
## Diversity Constraints
- Vary at least three of: attacker_type,
target_surface, network_domain,
kill-chain entry points.
A.2.5 Graph Summarization Prompt
For densely-connected nodes, the following prompt condenses
raw SPO triples into a security summary:
You are an expert Automotive Cybersecurity Analyst
and Knowledge Graph Summarizer.
Analyze the provided SPO triplets connected to
a single Hub Node. Generate a summary paragraph
(max 80 words) capturing the core threat landscape.
...[snip detailed instructions]...
Focus on:
- Attack Categories: Group vulnerabilities into
2-3 high-level categories (\textit{e.g.},
Authentication
Bypass, Memory Corruption, Protocol Flaws).
- Impact: Primary exploit capabilities
(\textit{e.g.}, RCE, Physical Access).
- Context: Maintain automotive domain terminology.
HUB NODE: {entity_name}
--- RAW TRIPLET DATA ---
{relations}
A.2.6 Attack Graph Generation Prompt
The final inference prompt integrates both retrieval sources
(VectorDB and GraphDB) into a unified context, with explicit
conflict resolution rules and a structured JSON output schema:
You are an expert automotive cybersecurity analyst.
=== VEHICLE TOPOLOGY ===
[ECU Details, External Interfaces,
Network Connections from vehicle spec]
=== RETRIEVED KNOWLEDGE ===
Similar Attack Cases: [VectorDB results]
Graph Context: [GraphDB traversal results]
...[snip]...
=== CONFLICT RESOLUTION RULE ===
If VectorDB (\textit{e.g.}, patch notes) conflicts with
GraphDB (\textit{e.g.}, existing vulnerability):
1. Prioritize VectorDB ONLY IF it explicitly
mentions the target vehicle’s make, model,
and year.
2. Otherwise, assume the vulnerability exists
16

(Conservative Approach).
=== OUTPUT SCHEMA ===
{
"nodes": [{"id", "label", "type", "group"}],
"edges": [{"source", "target", "step_id",
"type", "label", "protocol",
"technique_ids": ["ATM-Txxxx"]}]
}
...[snip requirements]...
A.3 Representative Attack Graphs
17

External Layer
Infotainment Domain
Telematics DomainGateway DomainInternal Network
Body Domain
Powertrain DomainUSB Port
(Interface)
Head Unit (NBT)USB Connection
[USB 2.0]
Step 1:
[Physical] USB/E-NET Access
[USB / OBD-II]
OBD-II (E-NET)
(Interface)E-NET Connection
[Ethernet]
Rogue Base Station
(Interface)Telematics Control BoxCellular Connection
[Rogue Cell]
Step 1:
[Remote] MITM / BT Pairing
[USRP / Bluetooth]Bluetooth
(Interface)Bluetooth Connection
[Bluetooth]Step 2:
[Physical] Code Signing Bypass
[Update Service]
Central Gateway
(Gateway)K-CAN / QNET
[K-CAN]
Step 3:
[Physical] UDS via Gateway
[K-CAN/QNET]
Step 2:
[Remote] Memory Corruption RCE
[NGTP/SMS Stack]K-CAN Connection
[K-CAN]
Step 3:
[Remote] Lateral Movement
[K-CAN/QNET]K-CAN Bus
(Bus)CAN Bridge
[K-CAN]
Step 4:
CAN Injection (Door Unlock)
[K-CAN]
PT-CAN Bus
(Bus)CAN Bridge
[PT-CAN]
Step 4:
CAN Injection (Engine)
[PT-CAN]Door Lock ECUStep 4:
Door Unlock
[K-CAN ID]
Engine ECUStep 4:
Engine Control
[PT-CAN ID]CAN Connection
[K-CAN]
CAN Connection
[PT-CAN]
External Layer
Telematics DomainInfotainment Domain
Gateway DomainVehicle Network
Powertrain DomainComfort Domain
Cellular Network (GSM/UMTS)
(Interface)
Telematics Control Unit (TCB)Cellular Connection
[GSM/UMTS Network]
Step 1:
Exploit DMA Issue via SMS
[SMS Port]
Step 1:
SMS Credential Phishing
[SMS Gateway]Wi-Fi Interface
(Interface)Head Unit (HU_NBT)WiFi Connection
[802.11 Wireless]Step 1:
Unauthenticated D-Bus Access
[Port 6667]Bluetooth Interface
(Interface)Bluetooth Connection
[BT Classic/BLE]
Step 1:
Exploit Bluetooth Stack
[RFCOMM Channel]
USB Port (Head Unit)
(Interface)USB Connection
[USB 2.0 Port]
Step 1:
Malicious Media File
[USB Mass Storage]
OBD-II Diagnostic Port
(Interface)Central Gateway (ZGW)
(Gateway)
OBD-II Connection
[Pin 6/14 CAN]
Step 1:
Physical OBD Access
[Pin 6/14]Ethernet via OBD-II
(Interface)Ethernet Connection
[192.168.0.1/24]
TCB Root AccessStep 2:
Escalate to Root
[UART IPC]Step 2:
TCB-to-HU Lateral Movement
[NGTP Protocol]
TCB-Gateway Link
[CAN Interface]
Step 3:
Bridge to Gateway
[CAN Interface]HU_NBT Root Shell
Step 2:
Escalate via AppArmor Bypass
[Kernel Patch]Step 2:
Gain Persistent Root
[System Daemon]Step 2:
Code Execution to Root
[Media Parser]
Gateway Link
[Ethernet Bridge]
Step 3:
Bypass Gateway RSA Challenge
[UDS Authentication]Step 3:
Send UDS via Ethernet
[192.168.0.1] K-CAN Bus
(Bus)Step 3:
Access K-CAN via Gateway
[CAN Channel 1]
CAN Injection ControlStep 3:
Establish CAN Control
[All CAN Channels]
ZGW Privileged Mode
(Gateway)
Step 4:
Disable Gateway Filtering
[UDS Service 0x31]Step 2:
Flash Malicious Firmware
[UDS Service 0x34]
K-CAN Interface
[CAN Channel 1]
PT-CAN Bus
(Bus)PT-CAN Interface
[CAN Channel 2]
Step 4:
Route to PT-CAN
[CAN Channel 2]
Step 4:
Route to Powertrain
[CAN Channel 2]FlexRay Bus
(Bus) FlexRay Interface
[FR Channel A/B]
MOST Bus
(Bus)
MOST Interface
[MOST150] LIN Bus
(Bus)
LIN Interface
[LIN Master]Step 3:
Access FlexRay Network
[FR Channel A/B]Step 5:
Establish CAN Injection
[All CAN Channels]Door Lock ControllerDoor ECU Connection
[CAN ID 0x2C0]
Step 4:
Unlock Vehicle Doors
[CAN ID 0x2C0]
Steering Control ECUSteering ECU Connection
[CAN ID 0x180]
Battery Management SystemBMS Connection
[CAN ID 0x2F0]
Step 5:
Manipulate Battery Control
[CAN ID 0x2F0]
Step 5:
Battery DoS Attack
[CAN ID 0x2F0]Step 4:
Inject FlexRay Commands
[FlexRay Slot 12]Step 6:
Inject Steering Commands
[CAN ID 0x180]Step 5:
Steering Manipulation
[CAN ID 0x180]Step 4:
Remote Unlock Attack
[CAN ID 0x2C0]
Figure 6: Comparison of attack graphs for Case 4 (BMW i3). Top: Ground-truth—two validated paths (Physical via USB/OBD-II and Remote via rogue base station)
converging at the Central Gateway. Bottom: LLM-generated (Claude 4.5 Sonnet, LOO condition)—the model reconstructs both GT paths while exploring additional
entry surfaces and alternative attack techniques.
18

External LayerInfotainment Domain
Gateway Domain
Vehicle NetworkBody Domain
Chassis Domain
Powertrain DomainCellular Interface
(Interface)
Uconnect Head UnitCellular Connection
[Sprint Network]
Step 1:
Exploit HTTP Vulnerability
[Port 6667]
Step 1:
Remote Authentication Bypass
[TCP Port 6667]
Wi-Fi Interface
(Interface)Wi-Fi Connection
[802.11n]
Step 1:
WiFi MitM Attack
[TCP Port 80]
Bluetooth Interface
(Interface)Bluetooth Connection
[BT 2.1+EDR]
Step 1:
Bluetooth Heap Overflow
[RFCOMM Channel]
USB Interface
(Interface)USB Connection
[USB 2.0 Port]
Step 1:
Malicious USB Device
[USB Mass Storage]RADIO Root Access
Step 2:
Privilege Escalation
[Kernel Exploit]Step 2:
Inject Malicious Update
[OTA Update]Step 2:
Achieve Code Execution
[ROP Chain]Step 2:
ADB Shell Access
[USB Debug Bridge]Step 2:
Command Injection
[D-Bus Command]
Vehicle Interface Processor
(Gateway)
SPI Connection
[SPI Bus]Step 3:
SPI Bus Manipulation
[SPI MOSI/MISO]
Step 3:
Compromise Gateway via SPI
[SPI Flash Write]
Step 3:
SPI Firmware Update
[SPI CS Pin]
Step 3:
Gateway Bootloader Exploit
[SPI Boot Mode]
Step 3:
SPI Flash Dump
[SPI Read]V850 Firmware Layer
Step 4:
Firmware Reflash
[Bootloader Mode]Step 4:
Persistent Backdoor
[Bootloader Hook]Step 4:
Firmware Analysis and Backdoor
[Binary Patch]
CAN Bus
(Bus)CAN Gateway
[CAN Interface]
Step 4:
Direct CAN Injection
[CAN TX Buffer]
Step 4:
Replay CAN Traffic
[CAN Frame Buffer]Step 5:
CAN Frame Injection
[CAN TX]
Step 5:
CAN Message Spoofing
[CAN TX/RX]
Step 5:
Silent CAN Monitoring
[CAN RX Only]Parking Assist ModuleCAN Connection
[CAN ID Range 0x200-0x2FF]
Step 6:
Parking Sensor Manipulation
[CAN ID 0x200]
Anti-lock Braking SystemCAN Connection
[CAN ID 0x140]
Step 5:
ABS Manipulation
[CAN ID 0x140]
Step 6:
Brake Force Modulation
[CAN ID 0x142]
Powertrain Control ModuleCAN Connection
[CAN ID 0x7E0]
Step 6:
Engine Control Command
[CAN ID 0x7E0]
Step 5:
Transmission Control
[CAN ID 0x7E8]
Step 7:
Fuel Injection Control
[CAN ID 0x7E1]
Figure 7: Best-scoring LLM-generated attack graph for Case 1 (Jeep Cherokee). Generated by Claude 4.5 Sonnet, Iteration 4 (PF = 78.0, KR = 73.8). The graph
comprises 12 nodes and 39 edges, identifying 5 distinct attack paths from 4 entry points (Cellular, WiFi, Bluetooth, USB) converging through the RADIO head unit
and V850 gateway to reach safety-critical targets (PAM, ABS, PCM) via the CAN bus. Red edges denote attack steps; gray edges denote network connections.
19

External Layer
Telematics Domain
Infotainment DomainVehicle Network
Gateway Layer
Powertrain DomainBody DomainInternet
(Cloud) 4G LTE Interface
(Interface)Step 1:
Remote MQTT Backdoor Exploitation
[Port 1883/8883]
HERMES 2.1 TCUCellular Link
[LTE Band 3/7/20]Step 2:
Authentication Bypass
[Weak password: admin/admin]WiFi (bcm4359)
(Interface)WiFi Link
[802.11n]
Step 1:
Rogue WiFi AP Setup
[hostapd on TP-Link WN-722N]
5GHz WiFi (BCM89359)
(Interface)Step 5:
Lateral Movement via WiFi
[Shared WiFi network]
Head Unit (MBUX)WiFi Link
[5GHz 802.11ac]Bluetooth Interface
(Interface)Bluetooth Link
[BT 5.0]
Step 1:
Legacy Pairing Exploitation
[4-digit PIN bruteforce]USB (CarPlay/Android Auto)
(Interface)
USB Link
[USB 3.0]Step 1:
Malicious USB Drive Insertion
[Auto-mount script]Ethernet (eth0)
(Interface)Ethernet Link
[eth0 100BASE-TX]
Step 1:
Diagnostic Port Access
[DoIP port 13400]Step 2:
DNS/DHCP Manipulation
[Scapy tbone.py script]
HERMES Root AccessStep 3:
Privilege Escalation
[Kernel CVE exploit]
Step 3:
Command Injection via MQTT
[MQTT topic injection]
CAN-D Bus
(Bus)CAN Connection
[500 kbps]
Step 6:
Remote CAN Bus Access
[can0 interface]Serial (/dev/ttyAMA1)
(Interface)Step 4:
Serial Console Activation
[/dev/ttyAMA1 115200]
Step 4:
CAN Injection from TCU
[Arbitrary CAN frames]
Step 5:
CAN Bus Flooding
[500 kbps saturation]Serial Debug
[/dev/ttyAMA1]
Step 2:
RFCOMM Buffer Overflow
[OBEX PBAP payload]Step 2:
USB Script Runner Exploitation
[Python reverse shell]Step 2:
UDS Seed-Key Bypass
[Security Access 0x27]
HU Root AccessStep 3:
Code Execution & Privilege Escalation
[pam_pcc.so exploit]
Step 3:
Persistent Backdoor Installation
[ncat listener port 4444]Step 3:
Firmware Reflash
[UDS service 0x34/0x36] MOST Bus
(Bus)
MOST Connection
[MOST150]CAN-HMI Bus
(Bus)HMI CAN Connection
[500 kbps]
Step 4:
WiFi Credential Extraction
[wpa_supplicant.conf]Step 4:
MOST Network Access
[MOST150 channel]Step 4:
CAN Message Injection
[Missing payload validation]
Step 5:
CAN Frame Replay Attack
[Captured legitimate frames]Electronic Ignition Switch Gateway
(Gateway)Gateway Link
[500 kbps]
Step 5:
Gateway Filtering Bypass
[Hardcoded key 0xDEADBEEF]
Step 6:
Gateway DoS
[Overwhelmed message queue]
Gateway Link
[500 kbps]
Step 5:
Gateway Exploitation
[Exposed private keys]
CAN-B Bus
(Bus)
Powertrain Control ECU
Powertrain Link
[CAN ID 0x7E0-0x7E7]Step 7:
Engine Control Manipulation
[CAN ID 0x7E0]Body Control Module
Body Control Link
[CAN ID 0x200-0x2FF]Gateway Link
[500 kbps]
Step 6:
Access Body CAN
[Forwarded frames]Step 6:
Physical Control Commands
[CAN ID 0x220]
Figure 8: Best-scoring LLM-generated attack graph for Case 2 (Mercedes MBUX). Generated by Claude 4.5 Sonnet, Iteration 8 (PF = 83.5, KR = 80.7). The graph
comprises 20 nodes and 52 edges, identifying 4 major attack paths from 8 entry points (4G_LTE, WiFi, Bluetooth, USB, Ethernet, Serial, Internet) through the T-Box
(HERMES) and HU (NTG6) to reach Powertrain_ECU and Body_Control_ECU via multi-bus pivoting across CAN_D, CAN_HMI, and CAN_B.
20

External Layer
Infotainment DomainGateway DomainBody Domain
ADAS DomainChassis DomainTelematics Domain
Bluetooth
(Interface)
i.MX6 (IVI)
Paired link
[L2CAP/RFCOMM/AVCTP]
Step 1:
Exploit RFCOMM multiplexer bug
[L2CAP PSM 0x0003, RFCOMM]Wi-Fi (Client)
(Interface)
Wi-Fi client
[802.11 STA]
Step 30:
Exploit Wi-Fi driver
[crafted frames]USB
(Interface)
USB port
[USB 2.0]
Step 10:
Malicious USB update/backdoor
[MSC, update parser]Cellular (LTE)
(Interface)
TCU (Telematic Unit)LTE data
[APN PDP]Internet/Cloud
(Cloud)IP reachability
[Public IP/DNS]
Step 20:
Exploit TCU over DNS/HTTP
[DNS 53, HTTP/HTTPS]
Ethernet/USB Link to i.MX6
(Interface)USB tether
[CDC ECM/RNDIS]USB tether
[CDC ECM/RNDIS]
Internal SPI
(Interface) Board SPI
[SPI bus]RH850 (IVI)Board SPI
[SPI bus]
INC (Internal Network)
(Bus)INC
[INC channel]INC
[INC channel]
Step 2:
Privilege escalate to root
[local exploit]Step 11:
Establish persistence/root
[updater, dm-verity]Step 23:
Enable CAN message access
[socketcan]Step 31:
Escalate to root
[kernel vuln]Step 3:
Abuse INC/SPI to command MCU
[SPI channel 0]Step 12:
Command MCU via SPI/INC
[SPI channel 0]
Step 24:
Issue MCU CAN commands
[SPI/INC]Step 32:
Control MCU bridge
[SPI/INC]ITS CAN
(Bus)
CAN link
[500 kbps]Gateway (Central Gateway Unit)
(Gateway)
Step 4:
Bypass gateway policy via CAN
[ITS CAN]
Step 13:
Exploit gateway policy
[ITS CAN]
Step 25:
Cross gateway segmentation
[ITS CAN]
Step 33:
Abuse gateway routing
[ITS CAN]CAN link
[ITS CAN]Vehicle CAN
(Bus)Step 5:
Bridge frames to Vehicle CAN
[Routing rule]
Step 26:
Forward to Vehicle CAN
[Routing rule]
ADAS CAN
(Bus)Step 14:
Bridge to ADAS CAN
[Routing rule]Chassis CAN
(Bus)Step 34:
Bridge to Chassis CAN
[Routing rule]CAN link
[Vehicle CAN]BCM (Body Control Module)
CAN link
[Vehicle CAN]Step 6:
UDS control of BCM
[UDS DID 0x745]Step 27:
UDS control BCM
[UDS DID 0x745]
CAN link
[ADAS CAN]
ADAS (Advanced Driver Assistance System) CAN link
[ADAS CAN]Step 15:
UDS DoS ADAS
[UDS DID 0x75D]CAN link
[Chassis CAN]Step 22:
Pivot over USB Ethernet
[CDC ECM/RNDIS]Step 21:
Root via unsigned update
[update service]
Figure 9: Best-scoring LLM-generated attack graph for Case 3 (Nissan Leaf). Generated by GPT 5.2, Iteration 10 (PF = 78.0, KR = 73.0). The graph comprises
18 nodes and 47 edges, identifying 4 primary attack paths from 5 entry points (Bluetooth, WiFi, USB, Cellular_LTE, Internet). The longest chain spans 9 steps from
Internet through TCU and iMX6_IVI to BCM. The graph highlights ADAS-specific targeting via dedicated ADAS_CAN and weak gateway segmentation across
5 CAN buses.
21

External Layer
Telematics DomainInfotainment Domain
Gateway DomainVehicle Network
Powertrain DomainComfort Domain
Cellular Network (GSM/UMTS)
(Interface)
Telematics Control Unit (TCB)Cellular Connection
[GSM/UMTS Network]
Step 1:
Exploit DMA Issue via SMS
[SMS Port]
Step 1:
SMS Credential Phishing
[SMS Gateway]Wi-Fi Interface
(Interface)Head Unit (HU_NBT)WiFi Connection
[802.11 Wireless]Step 1:
Unauthenticated D-Bus Access
[Port 6667]Bluetooth Interface
(Interface)Bluetooth Connection
[BT Classic/BLE]
Step 1:
Exploit Bluetooth Stack
[RFCOMM Channel]
USB Port (Head Unit)
(Interface)USB Connection
[USB 2.0 Port]
Step 1:
Malicious Media File
[USB Mass Storage]
OBD-II Diagnostic Port
(Interface)Central Gateway (ZGW)
(Gateway)
OBD-II Connection
[Pin 6/14 CAN]
Step 1:
Physical OBD Access
[Pin 6/14]Ethernet via OBD-II
(Interface)Ethernet Connection
[192.168.0.1/24]
TCB Root AccessStep 2:
Escalate to Root
[UART IPC]Step 2:
TCB-to-HU Lateral Movement
[NGTP Protocol]
TCB-Gateway Link
[CAN Interface]
Step 3:
Bridge to Gateway
[CAN Interface]HU_NBT Root Shell
Step 2:
Escalate via AppArmor Bypass
[Kernel Patch]Step 2:
Gain Persistent Root
[System Daemon]Step 2:
Code Execution to Root
[Media Parser]
Gateway Link
[Ethernet Bridge]
Step 3:
Bypass Gateway RSA Challenge
[UDS Authentication]Step 3:
Send UDS via Ethernet
[192.168.0.1] K-CAN Bus
(Bus)Step 3:
Access K-CAN via Gateway
[CAN Channel 1]
CAN Injection ControlStep 3:
Establish CAN Control
[All CAN Channels]
ZGW Privileged Mode
(Gateway)
Step 4:
Disable Gateway Filtering
[UDS Service 0x31]Step 2:
Flash Malicious Firmware
[UDS Service 0x34]
K-CAN Interface
[CAN Channel 1]
PT-CAN Bus
(Bus)PT-CAN Interface
[CAN Channel 2]
Step 4:
Route to PT-CAN
[CAN Channel 2]
Step 4:
Route to Powertrain
[CAN Channel 2]FlexRay Bus
(Bus) FlexRay Interface
[FR Channel A/B]
MOST Bus
(Bus)
MOST Interface
[MOST150] LIN Bus
(Bus)
LIN Interface
[LIN Master]Step 3:
Access FlexRay Network
[FR Channel A/B]Step 5:
Establish CAN Injection
[All CAN Channels]Door Lock ControllerDoor ECU Connection
[CAN ID 0x2C0]
Step 4:
Unlock Vehicle Doors
[CAN ID 0x2C0]
Steering Control ECUSteering ECU Connection
[CAN ID 0x180]
Battery Management SystemBMS Connection
[CAN ID 0x2F0]
Step 5:
Manipulate Battery Control
[CAN ID 0x2F0]
Step 5:
Battery DoS Attack
[CAN ID 0x2F0]Step 4:
Inject FlexRay Commands
[FlexRay Slot 12]Step 6:
Inject Steering Commands
[CAN ID 0x180]Step 5:
Steering Manipulation
[CAN ID 0x180]Step 4:
Remote Unlock Attack
[CAN ID 0x2C0]
Figure 10: Best-scoring LLM-generated attack graph for Case 4 (BMW i3). Generated by Claude 4.5 Sonnet, Iteration 3 (PF = 88.5, KR = 87.5). The most complex
graph among the four cases, comprising 21 nodes and 59 edges with 5 distinct attack paths from 6 entry points (Cellular, WiFi, Bluetooth, USB, OBD-II, Ethernet).
The graph spans a multi-bus architecture (K-CAN, PT-CAN, FlexRay, MOST, LIN) converging through the ZGW central gateway to reach Steering_ECU, BMS_ECU,
and Door_Lock_ECU.
22