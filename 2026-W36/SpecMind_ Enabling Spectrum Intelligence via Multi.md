# SpecMind: Enabling Spectrum Intelligence via Multi-Agent Hybrid Retrieval-Augmented Generation

**Authors**: Songwei Dong, Bingyan Lu, Makayla Kienlen, J. Nicholas Laneman, Cong Shen

**Published**: 2026-08-31 22:04:59

**PDF URL**: [https://arxiv.org/pdf/2609.00427v1](https://arxiv.org/pdf/2609.00427v1)

## Abstract
The exponential growth of wireless devices is driving unprecedented spectrum demand, pushing spectrum management toward more fine-grained decisions across space, time, and device constraints. As a result, spectrum policymakers and engineers must process large volumes of data that come from diverse sources and take many different forms, such as text and tables. These data sources are often disaggregated and require significant time and effort to integrate, search, and interpret. Furthermore, most of this information is formatted for human understanding and is not readily accessible to automated systems. To address this challenge, we propose SpecMind, a novel Multi-Agent Retrieval-Augmented Generation (RAG) system for spectrum intelligence that performs reasoning over heterogeneous data sources. This system enables autonomous agents to coordinate specialized sub-agents that retrieve and synthesize knowledge across policy proceedings, legal regulations, and license databases. We develop SpecBench, a question and answer (Q&A) dataset based on real-world license records and policy proceedings, addressing the lack of evaluation resources for RAG systems in the spectrum domain. Experimental results demonstrate that SpecMind outperforms traditional, general-purpose RAG systems across spectrum-related tasks, achieving over 80% win rate against strong baselines. The agent-based design enables more accurate retrieval, better contextual reasoning, and improved task completion across diverse query types.

## Full Text


<!-- PDF content starts -->

SpecMind: Enabling Spectrum Intelligence via
Multi-Agent Hybrid Retrieval-Augmented
Generation
Songwei Dong∗‡, Bingyan Lu∗†, Makayla Kienlen§, J. Nicholas Laneman†and Cong Shen‡
‡Department of Electrical and Computer Engineering, University of Virginia
†Department of Electrical Engineering, University of Notre Dame
§Baskin School of Engineering, University of California, Santa Cruz
‡{hxt5ap, cong}@virginia.edu,†{blu, jnl}@nd.edu,§mkienlen@ucsc.edu
Abstract—The exponential growth of wireless devices is driving
unprecedented spectrum demand, pushing spectrum manage-
ment toward more fine-grained decisions across space, time,
and device constraints. As a result, spectrum policymakers and
engineers must process large volumes of data that come from
diverse sources and take many different forms, such as text
and tables. These data sources are often disaggregated and
require significant time and effort to integrate, search, and
interpret. Furthermore, most of this information is formatted for
human understanding and is not readily accessible to automated
systems. To address this challenge, we proposeSpecMind,
a novel Multi-Agent Retrieval-Augmented Generation (RAG)
system for spectrum intelligence that performs reasoning over
heterogeneous data sources. This system enables autonomous
agents to coordinate specialized sub-agents that retrieve and
synthesize knowledge across policy proceedings, legal regulations,
and license databases. We developSpecBench, a question and
answer (Q&A) dataset based on real-world license records and
policy proceedings, addressing the lack of evaluation resources
for RAG systems in the spectrum domain. Experimental results
demonstrate thatSpecMindoutperforms traditional, general-
purposed RAG systems across spectrum-related tasks, achieving
over 80% win rate against strong baselines. The agent-based de-
sign enables more accurate retrieval, better contextual reasoning,
and improved task completion across diverse query types.
Index Terms—spectrum management, agentic AI, retrieval-
augmented generation, large language models
I. INTRODUCTION
The wireless spectrum faces increasing demand as modern
services and devices compete for limited bandwidth. Spectrum
management has consequently become more complex, involv-
ing constraints across frequency, time, location, users, and reg-
ulatory requirements [2]. This complexity drives rapid growth
in spectrum data in both scale and heterogeneity. Routine tasks
such as transmission compliance verification and spectrum
policy development require integrating fragmented, heteroge-
neous information. These challenges necessitate scalable and
context-aware methods for spectrum data interpretation and
decision support [3].
This work was supported in part by SpectrumX, the National Science
Foundation (NSF) Spectrum Innovation Center, through grant AST 2132700
operated under Cooperative Agreement by the University of Notre Dame.
∗The first two authors contributed equally to this work.Large language models (LLMs) have demonstrated strong
capabilities in domain-specific understanding and reason-
ing [4], [5], while retrieval-augmented generation (RAG)
improves grounding by incorporating external knowledge
sources [6]–[8]. Recent work has begun to extend RAG to
spectrum and telecommunications domains. SpectrumRAG [9]
introduces an iterative framework that leverages query rewrit-
ing to refine retrieval. TelcoRAG [10] enhances retrieval over
telecommunications standards through query augmentation
and specialized pipelines. Radio Regulations GPT [11] pro-
poses a domain-specific RAG pipeline for regulatory under-
standing. However, these approaches remain largely centered
on text-based retrieval and lack adaptive mechanisms for
integrating heterogeneous, multi-source data, limiting their
effectiveness on increasingly complex tasks.
To address these limitations, we proposeSpecMind, an
Agentic RAG framework for spectrum intelligence over het-
erogeneous data. Our key contributions include the construc-
tion of a spectrum-specific corpus, the design of special-
ized tools and prompts, and the development of an agentic
framework that orchestrates these components for complex
reasoning and task execution that supports tasks such as
incumbent investigation, stakeholder analysis, and regulatory
compliance verification. We further introduce a benchmark for
joint license and proceeding tasks and conduct a systematic
evaluation ofSpecMind. Our contributions are summarized
as follows:
•Reusable, extensible spectrum knowledge databases:
We construct a unified corpus of FCC licensing records,
proceeding documents, and regulatory texts as comple-
mentary machine-readable resources. Licensing data are
organized in a relational SQL database, proceedings are
modeled as graphs capturing entities and cross-document
relations, and regulatory texts are indexed via embed-
dings for retrieval. The resulting corpus is reusable and
extensible, enabling efficient integration of new records.
•An expert-designed benchmark for heterogeneous
spectrum-focused RAG:We introduceSpecBench, an
expert-designed benchmark dataset for evaluating RAG
arXiv:2609.00427v1  [cs.AI]  31 Aug 2026

systems over heterogeneous spectrum data. It comprises
450 curated question–answer pairs reflecting realistic
spectrum analysis tasks, including fact verification, cross-
source synthesis, and multi-hop reasoning. Built upon the
proposed databases,SpecBenchenables standardized
and reproducible evaluation and serves as a valuable
resource for spectrum-oriented RAG.
•A multi-agent hybrid RAG system for heterogeneous
sources:To the best of our knowledge,SpecMindis
the first multi-agent hybrid RAG framework for spec-
trum intelligence that enables coordinated retrieval and
reasoning over heterogeneous spectrum data, including
tabular, graph-structured, and textual sources. By sup-
porting modality-aware retrieval and cross-source reason-
ing,SpecMindimproves response accuracy and com-
pleteness over existing spectrum-oriented RAG pipelines,
achieving over 80% win rate against strong baselines.
We have publicly released theSpecBenchdataset, the
reusable spectrum databases, and theSpecMindframework
implementation to support reproducibility and community
research [1].
II. METHODOLOGY
A. Database Construction
The construction process is illustrated in Figure 1. We
consider three primary data sources in the spectrum domain:
(1)license data, which appear as structured tables with
numerical attributes; (2)proceeding documents, released by
the FCC, each centered on a specific topic and containing
diverse comments from individuals and organizations during
regulatory deliberation; and (3)regulatory documents, which
codify existing rules and policies. Applying RAG to such
heterogeneous sources requires constructing modality-aware
databases tailored to their distinct characteristics, rather than
relying on a general-purposed semantic retrieval pipeline.
Forregulatory texts, we adopt a standard dense retrieval
paradigm. Documents are segmented into fixed-length chunks
(1000 tokens with 200-token overlap) to preserve semantic
continuity. Given a query, we retrieve top-kcandidates via
embedding similarity and further apply a reranking stage to
refine relevance. Specifically, we select the top 20 candidates
from initial retrieval and rerank them to obtain the top 5 final
contexts. This design aligns with the nature of regulatory texts,
which are concise and legally precise, making queries rely
heavily on accurate semantic matching.
In contrast,FCC proceedingcomments and reply comments
exhibit high volume, topic diversity, and complex multi-
entity interactions. A single entity may express inconsis-
tent positions across proceedings, and many queries require
cross-document synthesis (e.g., identifying agency stances or
comparing stakeholder views). Such requirements are poorly
served by chunk-level semantic retrieval. We therefore en-
hance GraphRAG [12], which models documents as graphs
with entities as nodes and relations as edges, and construct
hierarchical summaries to support multi-hop reasoning. We
Regulation Documents
Codiﬁed rules, concise text
FCC Proceeding Documents
Diverse comments, 
multi-entity interactions
License Data
Structured numerical attributesPhase 1: Heterogeneous
Data Sources
Standard Dense Retrieval Pipeline
Chunk  Embedding
Fixed-length segments
with overlap
Phase 2: Modality-A ware Pipeline
Text embedding
computation
Hierarchical
SummarizationKnowledge
ExtractionGraph-Based Pipeline
Extract entities (Nodes)
& relations (Edges)Support multi-hop
reasoningRegulation V ector DB
Per-Proceeding
graph DBs
License SQL  DBData 
TransformationStructured Relational Pipeline
Extract entities (Nodes)
& relations (Edges)SQL Conversion
Map to schema,
organize into tablesPhase 3:  Heterogeneous
Knowledge StorageFig. 1. Database construction pipeline for heterogeneous spectrum data.
build a separate graph database for each proceeding to ensure
structural consistency (e.g., avoiding cross-document conflicts
introduced by merging multiple proceedings into a single
graph) and enable modular extensibility without rebuilding the
entire corpus when new proceedings are introduced. Detailed
statistics of the graph corpora, including node distributions
across hierarchy levels, are summarized in Table I.
License datapresent a fundamentally different challenge,
as they are dominated by structured numerical fields with
weak semantic signals [13]. Traditional RAG methods per-
form poorly in this setting due to their reliance on semantic
similarity. We instead model license data using relational
databases and perform retrieval via SQL queries to enable
exact matching over structured attributes. The problem is thus
reformulated as generating executable SQL queries from nat-
ural language questions, which can be effectively handled by
LLMs. Concretely, we extract license records into intermedi-
ate JSON representations for preprocessing, and then convert
them into a relational SQL database. We further organize
the database into service-specific subtables to accommodate
heterogeneous schemas across different license types.
TABLE I
GRAPH CORPUS CONSTRUCTION RESULTS FORFCCPROCEEDINGS.
NTIA NSSDENOTES THENTIA NATIONALSPECTRUMSTRATEGY.
ProceedingNumber of Nodes
Level 0 Level 1 Level 2 Level 3 Level 4
FCC 19 - 38 477 445 133 57 -
FCC 24 - 72 375 329 163 23 -
FCC 25 - 59 1045 1008 419 40 -
FCC 22 - 352 2094 2051 1599 571 -
FCC 23 - 158 902 868 405 53 -
FCC 23 - 232 871 794 338 82 58
NTIA NSS 4079 3960 3074 679 345
B. Framework Design
We proposeSpecMind, a multi-agent hybrid RAG system
for reasoning over heterogeneous spectrum data. The overall
system design is illustrated in Figure 2. Each task instance is
represented as
(Q,K,A,Π,T, A),

Algorithm 1Supervisor Agent Control Loop
Require:User queryQ, knowledge sourcesK, agent setA=
{asup} ∪ A sub, agent policiesΠ
Ensure:Final answer ˆA
1:Initialize global task stateT 0← ∅
2:Set termination flagdone←false
3:Set iteration indext←1
4:while notdonedo
5:Think:
6:Determine next actionu t∼π sup(Q,T t−1)
7:Act:
8:ifu tinvokes a specialized agenta∈ A subthen
9:Executeausing its policyπ aand tools, obtain
resultr t
10:else
11:Execute internal operation (e.g., subtask rewrit-
ing or evidence extraction), obtain resultr t
12:end if
13:Observe:
14:Update task stateT t← T t−1∪ {r t}
15:Updatedonebased onT t
16:t←t+ 1
17:end while
18:Aggregate evidence inT t−1to generate final answer ˆA
19:return ˆA
whereQdenotes a user query,Ais the ground-truth answer,
andKcomprises heterogeneous knowledge sources, includ-
ing structured license databases, graph-structured proceeding
documents, and unstructured regulatory texts. The agent set
is defined as
A={a sup} ∪ A sub,A sub={a lic, aproc, areg}.
Each agenta∈ Ais associated with an action policyπ a∈Π.
The Supervisor Agenta supcoordinates task decomposition,
agent invocation, and result integration, while specialized
agents perform domain-specific actions.
Given a queryQ, the system maintains a global task state
Tthat accumulates intermediate evidence and partial results
across multiple agent interactions. The Supervisor Agent
coordinates the execution of specialized agents and iteratively
refines the task state until a final answer ˆAis produced.
The objective of the system is to generate an answer ˆA
that is factually grounded inKand consistent with the ground-
truth answerA, while enabling accurate and flexible reasoning
across heterogeneous spectrum data sources.
C. Prompt Engineering
Design principle.We formulate prompt engineering as
the explicit specification of agent-level action policiesΠ =
{πsup, πlic, πproc, πreg}, which govern task decomposition, tool
usage, and cross-agent coordination. Inspired by the ReAct
paradigm [14], which interleaves reasoning and action, our
design extends this principle to a multi-agent setting tailored
to spectrum-specific tasks. Rather than relying on implicitreasoning in LLMs, we enforce structured decision-making
through prompt-level control, enabling reliable interaction
with heterogeneous data sources.
Supervisor prompt.The Supervisor Agent is prompted
to follow a structuredThink–Act–Observeloop for multi-
step reasoning. The prompt explicitly instructs the agent to
decompose the query into subtasks, decide whether to invoke
a specialized agent or perform an internal operation, and
iteratively update the global task state. This design enables
adaptive planning and dynamic routing across heterogeneous
tools, while ensuring that intermediate results are incorporated
into subsequent reasoning steps.
License agent prompt.For structured license data, we
design a tool-aware prompt that explicitly specifies the usage
of four SQL tools, including schema inspection, query exe-
cution, and query validation. The action policyπ licenforces
a stepwise interaction pattern, where the agent first inspects
database structure, then generates executable SQL queries,
and finally refines results based on returned outputs. This
design enables precise handling of numerical and table-centric
queries beyond the capability of semantic retrieval.
Proceeding agent prompt.For graph-structured proceed-
ing data, the prompt encodes a hierarchical retrieval strategy.
The action policyπ procfirst performs proceeding selection via
topic matching, and then dynamically chooses among three
GraphRAG tools (basic, local, and global) according to query
granularity. Specifically, local retrieval is preferred for entity-
centric queries, whereas global retrieval is used for higher-
level summarization. To improve robustness under incomplete
graph construction, the prompt further incorporates a fallback
rule that invokes basic retrieval when structured retrieval fails
to return sufficient evidence.
Regulation agent prompt.For regulatory documents, we
design a minimal high-precision prompt that focuses on re-
trieval refinement rather than tool selection. The action policy
πregadopts a two-stage retrieve-and-rerank pipeline, where
embedding-based retrieval is followed by lightweight rerank-
ing withQwen3-Reranker-0.6B. This design improves
evidence precision for regulation-oriented queries, where cor-
rectness is critical.
III.SP E CBE N C HDATASET
We introduceSpecBench, a real-world benchmark dataset
for evaluating RAG systems in the spectrum domain.
SpecBenchaddresses the lack of benchmarks for heteroge-
neous spectrum data sources, including proceedings, license
databases, and regulations.SpecBenchadopts a question
and answer (Q&A) format, a standard paradigm for evaluating
both retrieval accuracy and generation faithfulness [11], [15].
Following established RAG evaluation dimensions [16], we
consider three core capabilities: (i)noise robustness, i.e., the
ability to identify correct evidence under noisy retrieval re-
sults; (ii)information integration, i.e., the ability to synthesize
evidence across multiple sources; and (iii)negative rejection,
i.e., the ability to detect insufficient supporting evidence

License  SQL 
DatabaseProceeding 
Graph  Docs      License Agent        Proceeding Agent
GraphRAG tools
design query  query
response query response
 queryresponse
         Regulation Agent
Regulation 
chunk databaseRegulation  RAG 
tools design retrievePrompt Loop 
Logic design                        Supervisor Agent
TaskTask Answer
Think Act Observe
SQL tools
design retrieveFig. 2. Overview of the proposedSpecMindframework.
and abstain from answering, thereby avoiding hallucinated
responses.
To unify evaluation across modalities, we define anevi-
dence unitas the minimal retrievable element, corresponding
to a document for unstructured data and a table cell for
structured data. A question is classified assingle-sourceif
it can be answered using one evidence unit, andmulti-source
otherwise. This distinction enables controlled evaluation of
noise robustness and information integration.
TABLE II
OVERVIEW OFSP E CBE N C HQUESTION CATEGORIES AND EVALUATED
CAPABILITIES.
Category Subtype Data Source Evaluated Capability
Proceeding Single-cellProceedingsNoise Robustness
Multi-cells Information Integration
License Single-cellLicensesNoise Robustness
Multi-cells Information Integration
Regulation WiLL [15] FCC Title 47 Information Integration
Compound ParallelMulti-source Information IntegrationSequential
Unanswerable - - Negative Rejection
As summarized in Table II,SpecBenchorganizes ques-
tions by data source and task structure. Single-source ques-
tions primarily evaluate noise robustness, while multi-source
and compound questions assess information integration. Com-
pound questions further test cross-agent coordination and are
categorized intoparallelandsequentialtypes. Inparallelquestions, required evidence can be retrieved independently
from different sources and combined at the final stage. By
contrast,sequentialquestions involve inter-step dependencies,
where intermediate results from one retrieval step are required
to formulate subsequent queries, making them more suscepti-
ble to error propagation. Unanswerable questions contain no
valid supporting evidence and evaluate negative rejection.
We constructSpecBenchbased on question types iden-
tified through real-world domain experts interviews, ensuring
coverage of practical tasks. For each question, we manually re-
trieve supporting documents and provide a evidence-grounded
reference answer. For regulation questions, we incorporate
adapted Q&A samples from theWiLLbenchmark [15] to
improve coverage of regulation-focused queries. The dataset
contains 450 Q&A pairs spanning proceeding (31.1%), license
(31.1%), regulation (13.3%), compound (14.4%), and unan-
swerable (10.0%), balancing realism and annotation quality.
The distribution of questions across data sources and task
types is illustrated in Fig. 3.
IV. EXPERIMENTS
A. Baselines
Web-search RAG.This baseline uses Google Search as the
retriever, collecting the top-20 results per query as generation
context. It serves as a practical baseline for spectrum question
answering using publicly accessible resources.
SpectrumRAG.We include SpectrumRAG [9] as a com-
petitive baseline, which is an iterative RAG framework that
leverages LLM-based query rewriting to refine retrieval for

Fig. 3. Distribution of questions inSpecBench. The inner ring shows
the proportion of questions across different sources, while the outer ring
distinguishes between single-source and multi-source questions.
spectrum policy question answering. This design improves re-
trieval quality and downstream generation, making it a strong
representative of advanced RAG systems in this domain.
B. Experimental Setup
Models.To ensure a fair comparison, we evaluate all
methods, including the baselines andSpecMind, using two
backbone LLMs of different scales, Qwen3-8B and GPT-5.2.
Each method uses the same backbone for both RAG and
reasoning, andtext-embedding-3-smallis used for
retrieval across all methods.
Metrics.We reportsuccess rateandwin ratefor each
query. The success rate evaluates whether a method pro-
duces a valid response under one of the two settings: (i)
for standard question types, including license, proceeding,
regulation, and compound queries, the method must return
a factually grounded answer when sufficient supporting evi-
dence exists; and (ii) for negative-rejection questions, where
no valid evidence is available, the method must correctly
abstain and avoid hallucination by issuing a rejection. The
win rate measures comparative answer quality. We use a
strong proprietary LLM, Gemini 3.1-Pro-Preview, as a unified
evaluator, which is distinct from the backbone models and
fixed across all experiments to ensure consistent comparisons.
For each query, the evaluator selects the best answer among all
candidates; if all answers are incorrect, no method is assigned
a win.
Evaluation Rubric Prompt
Given a question and its ground-truth answer,
evaluate the responses generated by different
methods.
For each response, assess:
1. Correctness: whether the answer is
factually consistent with the ground truth. 2.
Completeness: whether all key aspects of the
question are addressed. 3. Faithfulness: whether
the answer avoids hallucination and unsupportedclaims.
Select the response that is both correct and of
the highest overall quality. If all responses are
incorrect, do not select any answer.
C. Main Results
As shown in Table III,SpecMindconsistently outperforms
the baselines, including Web-search RAG and SpectrumRAG,
across question types under both backbone LLMs, achieving
substantial win-rate gains, especially on license and compound
queries, demonstrating effective structured retrieval and cross-
source reasoning.
In terms of success rate,SpecMindmaintains strong per-
formance across all categories, achieving near-perfect results
with GPT-5.2 and consistently high accuracy with Qwen3-8B.
In contrast, the baselines achieve significantly lower success
rates, particularly on license and compound tasks, highlighting
their limitations in precise retrieval and coordinated reasoning.
While Web-search RAG performs relatively well on regulation
queries due to strong semantic retrieval, it fails to generalize
to structured and multi-source settings.
For unanswerable questions, all methods achieve perfect
success rates. This is largely because the task primarily
requires abstention rather than evidence synthesis, and modern
backbone LLMs already exhibit strong intrinsic capabilities in
recognizing insufficient evidence and avoiding unsupported
generation. As a result, performance on this category is less
sensitive to the retrieval framework and more dependent on
the underlying model’s calibration.
D. Ablation Studies
To analyze the source of performance gains, we conduct
ablation studies on the three specialized agents under the GPT-
5.2 backbone by individually replacing each sub-agent with a
naive RAG module (without re-ranking) while preserving the
overall framework. Results are reported in Table IV.
We observe consistent performance degradation across all
ablations, confirming that each agent contributes to the overall
effectiveness. The impact is most pronounced on the cor-
responding question types, followed by compound queries,
reflecting error propagation in multi-step reasoning.
The degradation is particularly severe for the license agent,
with substantial drops in both win rate and success rate (up to
−87.8and−82.8), highlighting the importance of SQL-based
retrieval for structured data. The proceeding agent also plays
a critical role, with notable declines on proceeding questions
(−27.3win,−10.3success), primarily due to failures in multi-
document reasoning, underscoring the benefit of graph-based
retrieval. In contrast, the regulation agent shows relatively
smaller impact, as its improvements mainly arise from prompt
design and reranking over text-based retrieval, which offer
more limited gains than structured and graph-based mecha-
nisms.

TABLE III
PERFORMANCE ACROSS QUESTION TYPES UNDER DIFFERENT BACKBONES. WIN:WIN RATE; SUCCESS:SUCCESS RATE.
Model MethodQuestion Type
Proceeding License Regulation Compound Unanswerable Overall
Win Success Win Success Win Success Win Success Win Success Win Success
Qwen3-8BWeb-search RAG 2.9 70.2 0.0 15.5 29.7 88.3 4.8 9.2 -1005.4 37.8
SpectrumRAG 12.5 71.5 0.8 6.7 6.7 52.4 0.0 0.0 -1006.1 42.6
SpecMind60.4 91.8 79.3 87.9 42.0 96.7 70.5 81.7-100 66.2 89.8
GPT-5.2Web-search RAG 4.7 78.3 0.0 25.3 38.3 95.8 8.3 13.3 -1007.9 50.5
SpectrumRAG 18.8 78.8 1.2 8.2 8.3 60.6 0.0 0.0 -1008.5 56.5
SpecMind72.9 100 97.6 100 51.7 100 86.8 96.7-100 81.1 99.6
TABLE IV
ABLATION STUDY:REPLACING EACH SUB-AGENT WITHNAIVERAG (WITHOUT RERANKING).
MethodQuestion Type
Proc. Lic. Reg. Comp. Neg. All
W S W S W S W S W S W S
SpecMind72.9 100 97.6 100 51.7 100 86.8 96.7 - 100 81.1 99.6
w/o License Agent (RAG)70.2 (-2.7) 98.5 (-1.5) 9.8 (-87.8) 17.2 (-82.8) 48.9 (-2.8) 97.6 (-2.4) 66.5 (-20.3) 80.1 (-16.6) - 100(-0.0) 66.8 (-14.3) 84.4 (-15.2)
w/o Proceeding Agent (RAG)45.6 (-27.3) 89.7 (-10.3) 94.8 (-2.8) 98.7 (-1.3) 49.5 (-2.2) 97.9 (-2.1) 63.2 (-23.6) 79.4 (-17.3) - 100(-0.0) 63.1 (-18.0) 87.2 (-12.4)
w/o Regulation Agent (RAG)71.8 (-1.1) 99.3 (-0.7) 96.2 (-1.4) 99.1 (-0.9) 46.7 (-5.0) 93.6 (-6.4) 84.7 (-2.1) 95.1 (-1.6) - 100(-0.0) 80.3 (-0.8) 99.0 (-0.6)
SUPER VISOR AGENT
TaskRoute Subtask 1
SUBT ASK 1:
Find Satellite frequency
STATE UPDA TE:
Frequency band = 
12.7-13.25 GHz
REFINE QUESTION 1:
List licensees & check
INTELSA T
STATE UPDA TE:
Licensees identiﬁed
REFINE QUESTION 2:
Summarize & compare
licensee opinions
 12.7-13.25 GHz
Frequency Result
Route Subtask 2
INTELSA T, GLOBECOMM ...LICENSE AGENT
SQL Query
License DBLicensee list
PROCEEDING AGENT
Proceeding FilesGlobal Graph
RetrievalRoute Subtask 3
Position Synthesis
Consistent, AGAINST  sharingREGULA TION AGENT
Regulation Files          RAG
(Embedding Search)
Final Output
Fig. 4. Case study ofSpecMindon a representative compound query.
V. QUALITATIVEANALYSIS
A. Illustrative Example
To illustrate howSpecMindoperates, we present a com-
pound Q&A example jointly supported by all three specialized
agents:Within the 12.7–13.25 GHz band, which frequencies
are allocated to satellite services? Is Intelsat an incumbent in
this band, and if so, do Intelsat and other incumbents share the
same position on spectrum sharing with terrestrial services?
The reasoning process is illustrated in Fig. 4. The supervisor
agent operates in a sequential, state-driven manner, iteratively
selecting which agent to invoke and what subtask to issue.
It first queries the regulation agent to identify the relevant
frequency band (e.g., 12.7–13.25 GHz), establishing a con-
straint for subsequent steps. Based on this result, it invokes
the license agent to retrieve corresponding licensees (e.g.,
INTELSAT, GLOBECOMM) via SQL queries, grounding
the task in structured data. Finally, the proceeding agentperforms graph-based retrieval to compare these entities’
positions on spectrum sharing and reveal consistent opposition
among incumbents. At each step, intermediate results are
incorporated into the task state, allowing the supervisor to
adaptively refine subsequent actions and resolve inter-step
dependencies. This example demonstrates howSpecMind
supports coherent cross-source reasoning through dynamic
coordination of heterogeneous retrieval mechanisms.
B. Failure Analysis
To better understand the small set of remaining non-optimal
cases, we manually inspect queries where SpecMind does
not achieve the best result or produces a partially complete
answer. We observe three recurring edge conditions.(i) Fine-
grained attribution.In some proceeding graphs, entity–filing
relations indicate co-occurrence but do not always explic-
itly encode whether the entity authored the filing or was
referenced by another party, which can make fine-grained
attribution more challenging.(ii) Sparse evidence for long-
tail entities.Stakeholders that appear only a few times in
a proceeding naturally have sparser graph representations,
and retrieval for such entities may surface broader contextual
information rather than the most specific statement of position.
(iii) Ambiguity in structured queries.Some identifiers are
only unique within a service-specific subtable, so queries
that omit the service type can admit multiple reasonable
interpretations and lead to broader aggregation than intended.
Overall, these cases largely reflect edge conditions in evidence
representation and query specification rather than systematic
limitations of the coordination framework. They also point
to natural extensions, including richer authorship metadata in
graph construction and lightweight clarification mechanisms
for ambiguous queries.

VI. CONCLUSIONS
In this paper, we introducedSpecMind, a multi-agent
hybrid RAG framework for heterogeneous spectrum data,
together withSpecBench, a 450-question benchmark span-
ning licenses, regulations, proceedings, and cross-source tasks.
The results show thatSpecMindis particularly effective
when conventional semantic retrieval is poorly matched to the
underlying data structure: SQL-based retrieval is critical for
license records, graph-based retrieval benefits multi-document
proceedings, and their coordination yields substantial gains on
compound queries. In contrast, the smaller improvement on
regulation questions suggests that conventional RAG remains
competitive when the source is already well suited to semantic
retrieval. These findings indicate that the value ofSpecMind
lies in selecting and composing retrieval mechanisms accord-
ing to the structure of the evidence rather than relying on a
single universal retrieval pipeline.
Our analysis further characterizes the remaining non-
optimal cases and highlights opportunities for refinement,
particularly in fine-grained evidence representation and query
disambiguation. Future work can build on these directions
while expanding the scale and coverage of spectrum policy
datasets to support more comprehensive evaluation across
broader regulatory domains, frequency bands, and stake-
holder interactions. To facilitate such studies, we publicly
releaseSpecBench, the reusable spectrum databases, and
theSpecMindframework, providing a common basis for
evaluating retrieval-augmented reasoning over heterogeneous
spectrum data [1].
ACKNOWLEDGMENT
The authors would like to thank Zhiyu Shen, Yuxi Chen,
Omkar Mujumdar, Yankai Peng, Hasan Nazim Bicer, Christo-
pher Wahl, Solbee Kang, and Lucas Scholler in the Notre
Dame Wireless Institute for their valuable contributions to data
collection and the evaluation process.
In addition, we thank Caleb Reinking, Le Li Kruczek,
Connor Howington and Paul Brenner in the Notre Dame
Center for Research Computing for their contributions to code
optimization and the implementation of the system user inter-
face. Last but not the least, this project was conceived during a
SpectrumX center meeting, and subsequently received a seed
fund to carry out the research. We thank NSF and SpectrumX
for the support.REFERENCES
[1] “Specmind and SpecBench,” https://specmindrag.github.io.
[2] P. Marshall,Three-Tier Shared Spectrum, Shared Infrastructure, and a
Path to 5G. Cambridge University Press, 2017.
[3] H. Rutagemwa, L. Odilinye, and D. Mahmoud, “Accelerating regulatory
processes with large language models: A spectrum management case
study,” in2024 IEEE Pacific Rim Conference on Communications,
Computers and Signal Processing (PACRIM). IEEE, 2024, pp. 1–7.
[4] L. Bariah, H. Zou, Q. Zhao, B. Mouhouche, F. Bader, and M. Debbah,
“Understanding telecom language through large language models,” in
GLOBECOM 2023-2023 IEEE Global Communications Conference.
IEEE, 2023, pp. 6542–6547.
[5] B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan,
P. Shyam, G. Sastry, A. Askell, S. Agarwalet al., “Language models
are few-shot learners,”arXiv preprint arXiv:2005.14165, vol. 1, no. 3,
p. 3, 2020.
[6] A. Singh, A. Ehtesham, S. Kumar, and T. T. Khoei, “Agentic retrieval-
augmented generation: A survey on agentic RAG,”arXiv preprint
arXiv:2501.09136, 2025.
[7] Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai, J. Sun,
H. Wang, and H. Wang, “Retrieval-augmented generation for large
language models: A survey,”arXiv preprint arXiv:2312.10997, vol. 2,
no. 1, 2023.
[8] K. Guu, K. Lee, Z. Tung, P. Pasupat, and M.-W. Chang, “Realm:
Retrieval-augmented language model pre-training,” 2020. [Online].
Available: https://arxiv.org/abs/2002.08909
[9] B. Lu, C. Reinking, and J. N. Laneman, “Initial evaluation of retrieval-
augmented generation approaches in spectrum policy research,” in2025
IEEE International Symposium on Dynamic Spectrum Access Networks
(DySPAN). IEEE, 2025, pp. 1–5.
[10] A.-L. Bornea, F. Ayed, A. De Domenico, N. Piovesan, and A. Maatouk,
“Telco-RAG: Navigating the challenges of retrieval augmented language
models for telecommunications,” inGLOBECOM 2024-2024 IEEE
Global Communications Conference. IEEE, 2024, pp. 2359–2364.
[11] Z. E. Kassimi, F. Fourati, and M.-S. Alouini, “Retrieval-augmented gen-
eration for reliable interpretation of radio regulations,”arXiv preprint
arXiv:2509.09651, 2025.
[12] D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody,
S. Truitt, D. Metropolitansky, R. O. Ness, and J. Larson, “From local
to global: A graph RAG approach to query-focused summarization,”
2025. [Online]. Available: https://arxiv.org/abs/2404.16130
[13] E. Wallace, Y . Wang, S. Li, S. Singh, and M. Gardner, “Do
NLP models know numbers? probing numeracy in embeddings,” in
Proceedings of the 2019 Conference on Empirical Methods in Natural
Language Processing and the 9th International Joint Conference on
Natural Language Processing (EMNLP-IJCNLP), K. Inui, J. Jiang,
V . Ng, and X. Wan, Eds. Hong Kong, China: Association for
Computational Linguistics, Nov. 2019, pp. 5307–5315. [Online].
Available: https://aclanthology.org/D19-1534/
[14] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y . Cao,
“React: Synergizing reasoning and acting in language models,” 2023.
[Online]. Available: https://arxiv.org/abs/2210.03629
[15] A. Bansal, V . Muriga, J. Li, L. Duan, and S. Kumar, “Can we make
FCC experts out of LLMs?” inProceedings of the 26th International
Workshop on Mobile Computing Systems and Applications, 2025, pp.
85–90.
[16] J. Chen, H. Lin, X. Han, and L. Sun, “Benchmarking large language
models in retrieval-augmented generation,” inProceedings of the AAAI
Conference on Artificial Intelligence, vol. 38, no. 16, 2024, pp. 17 754–
17 762.