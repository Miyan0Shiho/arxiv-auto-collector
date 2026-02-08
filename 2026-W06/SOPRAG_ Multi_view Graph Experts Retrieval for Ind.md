# SOPRAG: Multi-view Graph Experts Retrieval for Industrial Standard Operating Procedures

**Authors**: Liangtao Lin, Zhaomeng Zhu, Tianwei Zhang, Yonggang Wen

**Published**: 2026-02-02 09:30:43

**PDF URL**: [https://arxiv.org/pdf/2602.01858v1](https://arxiv.org/pdf/2602.01858v1)

## Abstract
Standard Operating Procedures (SOPs) are essential for ensuring operational safety and consistency in industrial environments. However, retrieving and following these procedures presents unique challenges, such as rigid proprietary structures, condition-dependent relevance, and actionable execution requirement, which standard semantic-driven Retrieval-Augmented Generation (RAG) paradigms fail to address. Inspired by the Mixture-of-Experts (MoE) paradigm, we propose SOPRAG, a novel framework specifically designed to address the above pain points in SOP retrieval. SOPRAG replaces flat chunking with specialized Entity, Causal, and Flow graph experts to resolve industrial structural and logical complexities. To optimize and coordinate these experts, we propose a Procedure Card layer that prunes the search space to eliminate computational noise, and an LLM-Guided gating mechanism that dynamically weights these experts to align retrieval with operator intent. To address the scarcity of domain-specific data, we also introduce an automated, multi-agent workflow for benchmark construction. Extensive experiments across four industrial domains demonstrate that SOPRAG significantly outperforms strong lexical, dense, and graph-based RAG baselines in both retrieval accuracy and response utility, achieving perfect execution scores in real-world critical tasks.

## Full Text


<!-- PDF content starts -->

SOPRAG: Multi-view Graph Experts Retrieval for Industrial Standard
Operating Procedures
Liangtao Lin, Zhaomeng Zhu, Tianwei Zhang and Yonggang Wen
Nanyang Technological University, Singapore
liangtao002@e.ntu.edu.sg, {tianwei.zhang, ygwen}@ntu.edu.sg
Abstract
Standard Operating Procedures (SOPs) are
essential for ensuring operational safety and
consistency in industrial environments. How-
ever, retrieving and following these procedures
presents unique challenges, such as rigid propri-
etary structures, condition-dependent relevance,
and actionable execution requirement, which
standard semantic-driven Retrieval-Augmented
Generation (RAG) paradigms fail to address.
Inspired by the Mixture-of-Experts (MoE)
paradigm, we propose SOPRAG , a novel frame-
work specifically designed to address the above
pain points in SOP retrieval. SOPRAG replaces
flat chunking with specialized Entity, Causal,
and Flow graph experts to resolve industrial
structural and logical complexities. To opti-
mize and coordinate these experts, we propose
a Procedure Card layer that prunes the search
space to eliminate computational noise, and an
LLM-Guided gating mechanism that dynam-
ically weights these experts to align retrieval
with operator intent. To address the scarcity
of domain-specific data, we also introduce an
automated, multi-agent workflow for bench-
mark construction. Extensive experiments
across four industrial domains demonstrate that
SOPRAG significantly outperforms strong lex-
ical, dense, and graph-based RAG baselines
in both retrieval accuracy and response util-
ity, achieving perfect execution scores in real-
world critical tasks.
1 Introduction
The Standard Operating Procedure (SOP) is a set
of step-by-step instructions compiled by organi-
zations to guide personnel in performing routine
task, which is essential for ensuring safe, consistent,
and efficient operations in industrial environments.
As industrial systems grow increasingly complex,
SOP Retrieval, the capability to assist operators to
rapidly identify and follow the appropriate proce-
dure, has become ever more critical for minimizing
Server  Down!
What is the SOP? 
dRetrieved  SOP  Steps
(Step -by-Step Procedure)
Step1:         Call Supervisor 
(xxx)  immediately.
Step2:         Attempt a 
server reboot.
Step3:         Verify system 
status after reboot.
Step4:         Log incident 
and notify team.
Our SOPRAG  System
(MoE -Style SOP Retrieval & Generation)
Retriever Generator
Standard Operating Procedure (SOP) Document ID: IT -SOP -CRIT -005.A 
(Version 4.2)
1.Scope & Definitions
Applies to Tier 1 Support / Critical 
Outages Only. Refer to Appendix B for ...
2. Pre -Action Requirements
Required: SSH Client, Supervisor Contact 
List, Incident Log Form (ILF -01)...3. Detailed Mitigation Protocol
3.1 Call Supervisor (xxx) immediately upon confirmed ...
3.2 Attempt a hard server reboot via iDRAC/ILO after ...
3.3 Verify system status post -reboot using the Post ...
3.4 Log incident and notify team (DL -IT-03) and ...4. Review 
& Sign -off
Prepared by: 
________. 
Approved by: 
________. PC i PC j PC n PC kFigure 1:Overview of SOP Retrieval in practice.
(Top) An operator encounters a system failure and uses
ourSOPRAG to generate executable step-by-step guid-
ance. (Bottom) A representative sample of a SOP source
document used for retrieval.
downtime and mitigating operational risks (Akyar,
2012).
However, retrieving SOPs presents unique chal-
lenges compared to the general open-domain docu-
ment retrieval. Unlike common corpora structured
around narrative semantics, SOPs are engineered
for operational execution, characterized by rigid
dependencies between critical entities, conditional
logic, and procedural workflows. This distinct na-
ture renders standard semantic matching insuffi-
cient and imposes three primary hurdles for exist-
ing methods:
•(C1) Proprietary Structure:SOPs possess a
rigid hierarchy with high-level goals (titles), spe-
cific constraints (parameters, alarm codes, equip-
ment models), and execution logic (steps). Tra-
ditional chunking strategies often sever semantic
links between a procedural step and its governing
1arXiv:2602.01858v1  [cs.AI]  2 Feb 2026

context, rendering the retrieved chunk ambiguous
or actionable only for the wrong equipment.
•(C2) Condition-Dependent Relevance:Oper-
ator queries are inherently scenario-specific. A
symptom like "system overheat" may correspond
to entirely different procedures depending on the
underlying cause or system state. Relevance is
determined by causal logic and intent recognition,
rather than simple lexical or semantic overlap.
•(C3) Actionable Response Generation:Unlike
general Question Answering (QA), where a fac-
tual summary suffices, industrial operators re-
quire executable knowledge. The desired output
is a precise, step-by-step workflow that guides op-
erators through a specific scenario without hallu-
cinations that may lead to serious consequences.
Current Retrieval-Augmented Generation (RAG)
paradigms fall short in addressing these challenges.
While GraphRAG approaches (Peng et al., 2025)
have improved models’ reasoning capabilities, they
primarily construct graphs based on semantic simi-
larity or entity co-occurrence. They often neglect
the causal and sequential logic that defines a work-
flow, leading to suboptimal search quality and re-
sponses without procedural coherence. Further-
more, research in this niche is hindered by the lack
of comprehensive benchmarks, as most existing
datasets focus on general QA rather than the nu-
anced requirements of industrial retrieval.
Inspired by the Mixture-of-Experts (MoE)
paradigm (Shazeer et al., 2017), we propose
SOPRAG , a structure-aware framework to bridge
the above gaps by explicitly modeling the pro-
cedural "shape" of industrial knowledge. To dis-
mantle the core hurdles of proprietary structure
(C1), condition-dependency (C2), and actionable
execution (C3), we introduce three specialized
graph experts,Entity, Causal, and Flow graphs,
each engineered to resolve specific structural and
logical complexities. By utilizing graph-based
search and modeling instead of flat semantic chunk-
ing,SOPRAG directly overcomes context fragmenta-
tion while significantly elevating reasoning trans-
parency and precision.
Consistent with the principle of sparse activation,
we design aProcedure Cardlayer inspired by the
cognitive workflow of human experts who scan
high-level titles to prune irrelevant information be-
fore drilling into technical details. This layer serves
to purge computational noise and ensure that the un-
derlying specialists operate with maximum focus.
We further implement an intention-awareLLMRouteras our gating mechanism, which dynami-
cally modulates attention weights assigned to each
expert based on the operator’s specific query in-
tent. This elegant alignment reveals a robust, gated
system tailored for the high-precision demands of
industrial SOP retrieval.
Furthermore, to address the scarcity of domain-
specific data and rigorously evaluate our frame-
work, we introduce an agent-based simulation en-
vironment for automated benchmark construction,
which deploys a team of specialized agents, the
Collector,Archivist,Auditor, andExaminer, to em-
ulate the rigorous document processing lifecycle of
human experts.
Our main contributions are as follows:
•We identify the specific challenges of SOP re-
trieval and propose a novel agnentic framework
for its benchmark construction, which could fa-
cilitate future research.
• We introduceSOPRAG, a MoE-inspired structure-
aware retrieval and generation framework specifi-
cally tailored for SOPs, which integrates a Proce-
dure Card layer with Multi-view Graph Experts
to address the unique challenges in SOP retrieval.
•Extensive experiments demonstrate that our ap-
proach significantly outperforms strong lexical,
dense, and graph-based RAG baselines in both
retrieval accuracy and response utility.
2 Related Work
2.1 Procedural Question Answering
Procedural QA examines how to answer questions
grounded in stepwise texts. Early work by Baner-
jee and Bandyopadhyay (2012) formalized ques-
tion types and answering strategies for procedural
text, while Bosselut et al. (2018) modeled action
dynamics and state changes using Neural Process
Networks. Subsequent studies enhanced procedu-
ral understanding by incorporating external knowl-
edge (Zhang et al., 2021) or structural representa-
tions, such as graph-guided QA generation (Pham
et al., 2024). Procedural QA has also been extended
to multimodal settings, including RecipeQA (Yag-
cioglu et al., 2018) and ProMQA (Hasegawa et al.,
2025), as well as interactive, program-guided an-
swering (Maitra et al., 2020).
However, few studies have targeted domain-
critical procedural texts such as SOPs. Existing
work includes QA models for human–machine
interaction in manufacturing (Ruiz et al., 2023),
domain-centric RAG for smart manufacturing
2

(Zhang et al., 2021), or private knowledge-based
LLMs for enterprise SOP scenarios (Xie et al.,
2024). Nevertheless, these studies mainly focus
on question answering, whereas real industrial op-
erations require retrieving and following complete
SOP documents, which is the central gap our work
aims to address.
2.2 LLM Agents Enhanced by SOP
Recent work has increasingly explored incorpo-
rating SOPs into LLM-based agents to improve
controllability, reliability, and domain alignment.
SOP-Agent (Ye et al., 2025) proposes to em-
power general-purpose agents with domain-specific
SOPs as explicit behavioral guidance, while SOP-
Bench (Nandi et al., 2025) introduces a benchmark
of complex industrial SOPs to evaluate agents’ pro-
cedural understanding and execution capabilities.
Along this, Agent-S (Kulkarni, 2025) formulates
SOPs as agentic workflows, enabling LLM agents
to automate real-world operational procedures.
Other studies integrate SOPs into agent plan-
ning and coordination: ChatSOP (Li et al., 2025)
employs SOP-guided Monte Carlo Tree Search
for controllable dialogue planning, and Flow-of-
Action (Pei et al., 2025) enhances multi-agent sys-
tems with SOP constraints for root cause analysis.
Some efforts investigate SOP structuring and in-
terpretation, including generating structured plan
representations from procedural text (Garg et al.,
2025) and classifying SOP step components via
prompt engineering (Bashatah and Sherry, 2024).
Overall, these works treat SOPs primarily as
guidance or control mechanisms for LLM agents,
rather than as retrievable procedural knowledge,
leaving open the challenge of accurately retrieving
and grounding agents in complete SOP documents
in industrial settings.
2.3 Graph-based RAG
Retrieval-Augmented Generation enhances LLMs
by grounding generation in retrieved external docu-
ments, but conventional flat chunk-based retrieval
strategy ignores relationships across text segments.
Graph-based RAG (Peng et al., 2025) addresses
this limitation by organizing documents as graphs,
enabling multi-hop retrieval, global context ag-
gregation, and structured reasoning. Representa-
tive approaches include GraphRAG, where Edge
et al. (2024) construct document graphs and per-
form community-level summarisation for query-
focused global reasoning, and LightRAG (Guoet al., 2025), which introduces a lightweight graph
indexing scheme to jointly retrieve fine-grained
evidence and high-level concepts.
Beyond text graphs, Sun et al. (2023) propose
Think-on-Graph, enabling LLMs to traverse knowl-
edge graphs for multi-hop reasoning. Li et al.
(2024) introduce GraphReader, which organizes
long documents as graphs to support agent-style
coarse-to-fine retrieval. More recently, Gutiér-
rez et al. (2024, 2025) move from RAG to non-
parametric memory by structuring retrieved knowl-
edge as persistent graphs for continual learning.
Despite these advances, existing Graph-based RAG
methods remain largely semantic-driven and text-
centric, and do not explicitly model the procedural,
hierarchical, and step-dependent structures of SOP
documents. This motivates our structure-aware
Graph-based RAG design for industrial SOP.
3 SOPRAG
3.1 Overview
To address the unique challenges of SOP retrieval
(C1, C2, and C3), while simultaneously overcom-
ing the length and semantic constraints of chunking
and structural deficiencies in existing graph-based
RAG paradigms, we propose SOPRAG , as shown in
Figure 2. Inspired by the MoE paradigm, we intro-
duce three specialized graph-based experts, theEn-
tity, Causal, and Flow graphs, each specifically
engineered to dismantle these industrial hurdles.
To optimize this process, we introduce aProce-
dure Card (PC)layer as a Sparse Activation layer
to prune the reasoning scope and reduce computa-
tional noise. This design is inspired by the cogni-
tive workflow of human experts who prioritize scan-
ning high-level titles to filter candidate procedures
before drilling into technical execution. Finally,
anLLM-Guided Routeris employed for intent
recognition, acting as a gating mechanism that dy-
namically assigns attention weights to the graph
experts, ensuring the retrieval process is adaptively
tailored to the user’s specific query.
3.2 Specialized Expert Construction
LetS={S 1, S2, . . . , S N}be the set of atomic
SOPs. We construct a knowledge base Gwhere
each SOP is represented through a sparse activation
anchor and three specialized reasoning experts.
3.2.1 Procedure Card Layer
Consistent with the principle of sparse activation
in MoE, we abstract each SOP Siinto aProcedure
3

Procedure Cards (PCs) Layer
Title
SummaryPC1
Title
SummaryPC2
Title
SummaryPCN
RouterUser Query
Title
SummaryPCiPCi PCj PCn PCk
Pump X
Asset A
Temp
Alarm
Code 502 PUELow Water
PressureStep1
Based on SOP i :
•Step1: Call Supervisor 
(XXX) immediately.
•Step2: Attempt a 
server reboot.
•Step3: Verify system 
status after reboot.
Cooling
Failure
Temp
Rise
Overheat -
ingℛ𝐸 ℛ𝐶 ℛ𝐹
∑CAUSESRESULTS_IN
Step2A
Step3AStep2B
Step3BCOND
𝑤𝐶𝑤𝐸 𝑤𝐹
…𝑅𝑖
SOPj (Rank k)Step1: Call Supervisor
Step2: Server Reboot 
Step3: Verify StatusA. MoE -Inspired Expert Construction B. MoE -Style SOP Retrieval C. Structure -Aware 
Generation
…
SOPi (Rank 1)Figure 2:Architecture overview of SOPRAG .(A) Expert Construction: delineating the Procedure Card (PC) layer
and the three multi-view graph experts (Entity, Causal, and Flow); (B) MoE-Style Retrieval: illustrating the sparse
activation of the PC layer, the LLM-guided gating mechanism (Router), and the scoring and aggregation process of
the graph experts; (C) Structure-Aware Generation: showing the generation of a step-by-step actionable response
using the flow graph of retrieved SOP as a structured prompt context. (Further extensions are shown in Appendix G.)
Card( PCi), which consists of the SOP’s high-level
title and an LLM-synthesized one-line abstract:
PCi={Title(S i),Abstract(S i)} ∈ V Global (1)
The PC layer acts as the primary entry point, se-
lecting only the relevant "expert clusters" to ensure
the underlying reasoning components operate with
maximum focus.
3.2.2 Multi-view Graph Experts
To resolve the specific hurdles of SOPs, we re-
place generic semantic chunks with three special-
ized graph experts, each engineered to process a
distinct dimension of procedural information.
Entity Graph.To resolve the ambiguity of Pro-
prietary Structure (C1), where procedures are
strictly bound to specific equipment parameters
or alarm codes, we construct theEntity Graph( GE).
This subgraph serves as an entity-centric map, ex-
plicitly linking discrete domain entities to their gov-
erning Procedure Cards:
GE={(e, r, PC i)|e∈ V Entity},(2)
where VEntity represents the set of extracted en-
tities (e.g., assets, parameters) and rdenotes the
association relationship. This design prevents the
semantic decoupling common in traditional chunk-
ing, ensuring that queries specifying unique propri-
etary constraints are routed directly to the correct
equipment context.Causal Graph.To address Condition-Dependent
Relevance (C2), where the correct procedure de-
pends on the underlying root cause rather than just
surface symptoms, we build theCausal Graph( GC).
This graph models the non-linear condition-action
logic, tracing the path from observable states to the
resolving Procedure Card:
GC={(s u, sv)|suc− →sv} ∪ {(s, PC i)}(3)
Here,VState ={s u, sv, . . .} represents system
states (Symptoms, Causes), and the edges represent
causal transitions. By traversing the path (sstart→
··· →s end→PC i), the system can simulate
a diagnostic reasoning process, aligning retrieval
with the operator’s troubleshooting intent.
Flow Graph.To ensure Actionable Response
Generation and preserve the procedural integrity of
the response (C3), we generate a localFlow Graph
(GF) for each Procedure Card. Unlike flat text, this
graph captures the precise topological structure re-
quired for safe execution:
GF(PC i) = (V Step,EFlow),EFlow⊆ VStep×VStep
(4)
This structure GFenables procedure segmenta-
tion that respects the inherent stepwise organization
of each SOP, thereby avoiding the semantic frag-
mentation introduced by generic chunking strate-
gies (C1). Each resulting segment remains tightly
aligned with its corresponding PCi, facilitating
precise retrieval and providing operators with con-
cise, reliable, step-by-step operational guidance.
4

3.3 SOP Retrieval
SOPRAG formalizes retrieval as an MoE-style gated
inference pipeline. Adopting a "Coarse-to-Fine"
strategy, the system emulates sparse gating by first
pruning the search space via sparse activation and
then adaptively routing queries to specialized rea-
soning experts for precise execution. It performs
the following three specific steps.
3.3.1 Procedure Card Layer Anchoring
Given a query Q, the system first activates the Top-
KProcedure Cards to prune the search space. We
calculate a global alignment score based on the se-
mantic similarity between Qand Procedure Cards:
ATopK =TopKPCi∈VGlobal(Sim(Q, PC i))(5)
This design reflects how operators typically
search for SOPs by first scanning document titles,
as these titles generally indicate the primary func-
tion or purpose of each procedure.
3.3.2 LLM-Guided Routing
To navigate the experts effectively, an Intention-
AwareLLM Routerserves as the gating net-
work. It dynamically modulates the contribu-
tion of each expert by assigning attention weights
w= [w E, wC, wF] =Softmax(LLM Router(Q))1
based on the focus of the query:
•Entity Focus ( wE):For queries specifying
equipment IDs or parameters.
•Causal Focus ( wC):For "Why" or diagnosis-
oriented queries.
•Flow Focus ( wF):For "How" or execution-
oriented queries.
3.3.3 Subgraphs Reasoning
The final relevance score R(Q, S i)is computed by
aggregating the anchor score and the gated reason-
ing from the specialized experts:
R(Q, S i) =λ·Sim(Q, PC i) + (1−λ)
X
m∈{E,C,F}wm· Rm(Q, PC i)(6)
whereRmrepresents the specific scoring function
for each subgraph:
1.Entity Score ( RE):Measures the overlap of
1Details in Appendix A.2, Appendix Hproprietary terms:
RE=1
|VQ|X
e∈VQ
α·I(e∈ V Entity )
+(1−α)·max
e′∈VEntitySim(e, e′)
(7)
where VQdenotes the set of entities extracted
from Q,I(·)is the indicator function for Exact
Match, andα∈[0,1]is a balancing weight.
2.Causal Score ( RC):Evaluates the causal valid-
ity. It identifies a start state smatching Qand
measures the reachability to PCiby perform-
ing a k-hop neighborhood search on the causal
graphG C:
RC= max
s∈VState(Sim(Q, s)·Path(s→PC i))
(8)
3.Flow Score ( RF):Measures the semantic align-
ment with the executable steps within the local
flow graph:
RF= max
p∈VStep (PCi)Sim(Q, p)(9)
3.4 Structure-Aware Generation
Once the optimal SOP is identified, the system
retrieves its Flow Graph GF(PC i). This struc-
tured representation is linearized into a step-by-
step prompt context (Figure 2). By constraining
the LLM to this verified flow, we ensure the gener-
ation is precise and actionable, directly addressing
the hallucination risks in industrial settings.
4 SOP Benchmark Construction
To rigorously evaluate retrieval algorithms in the
domain of SOPs, we necessitate a high-quality,
domain-specific benchmark, especially geared to-
wards real-world industrial scenarios. Unlike gen-
eral open-domain QA, SOP retrieval requires pre-
cise alignment between operational queries and
procedural segments.
To this end, we propose an Agent-based Simula-
tion Environment for automated dataset construc-
tion. This pipeline simulates a real-world document
processing lifecycle (data collection →processing
→generation), ensuring both the diversity of the
source data and logical validity of the queries.
4.1 Agent Architecture
As illustrated in Figure 3, we conceptualize the
data generation pipeline as a collaborative work-
flow involving four specialized agents:Collector,
5

Goal: Build an 
SOP Retrieval 
Dataset for the 
Data Center field.
Agent -based Simulation Environment
     
DeepSearch OCR: Any -to-md Filtration & Segmentation Question GenerationRaw SOP
Documents
(PDF,
DOC…)Markdown
Format 
FilesValidated,    
Atomic 
SOPs
SOP-Query pairsBenchmark
Dataset
Collector Agent Archivist Agent Auditor Agent Examiner AgentFigure 3:The overview of the Agent-based Simulation Environment for automated dataset construction.
Given a specific domain, four specialized agents (Collector,Archivist,Auditor, andExaminer) mimic human expert
workflows to automatically generate a high-quality benchmark dataset.
Archivist,Auditor, andExaminer. This multi-agent
framework is designed to minimize human inter-
vention while maximizing data quality.
Collector.The foundation of the benchmark lies
in the acquisition of raw, domain-specific docu-
ments. TheCollectoris instantiated as an LLM-
powered agent equipped with DeepSearch capabil-
ities. Unlike general web scrapers, it operates in
a topic-driven manner. Given a specific industrial
theme (e.g., "Data Center Maintenance"), the agent
autonomously generates related search keywords
and navigates the web to locate relevant resources.
Its primary objective is to retrieve raw SOP doc-
uments that precisely align with the input topic,
ensuring the benchmark covers targeted domains
with high-quality source material.
Archivist.Raw data must be standardized and
verified to serve as a valid retrieval corpus. To ad-
dress the heterogeneity of file formats (e.g., PDFs,
spreadsheets, and presentation slides), theArchivist
employs a layout-aware parsing agent powered by
MinerU-2.5 (Niu et al., 2025). Its primary role is
document normalization: detecting document lay-
outs and converting diverse raw files into a unified,
clean Markdown format. This ensures subsequent
agents can process the content as pure text, while
simultaneously preserving the semantic integrity of
the original structure (e.g., headers, tables).
Auditor.TheAuditorserves as the quality assur-
ance layer, which performs two critical functions:
•Filtration:It analyzes the semantic content of
the Markdown files to filter out non-SOP docu-
ments (e.g., general news, pure data tables) that
may have bypassed theCollector.
•Segmentation:Since a single document often
contains multiple distinct procedures, theAuditor
splits multi-SOP files into atomic units. This
ensures that the retrieval granularity is at the level
of a single, executable procedure rather than a
coarse-grained file.Examiner.Once the SOPs are validated and seg-
mented into atomic units, we generate high-quality
evaluation pairs to establish the benchmark ground
truth. To construct realistic query-document pairs,
we employ theExaminer, a Question Generation
Agent. For each validated SOP, theExaminer
adopts the persona of a frontline operator to sim-
ulate diverse real-world information needs. It is
tasked with generating Ndistinct questions per pro-
cedure, spanning a spectrum from simple keyword-
based lookups to sophisticated, scenario-driven
queries. This multi-level generation strategy en-
sures the benchmark accurately reflects the com-
plex problem-solving requirements of industrial
personnel in operational environments.
4.2 Benchmark Statistics
Leveraging this automated pipeline, we construct
a comprehensive SOP benchmark comprising four
domain-specific datasets to evaluate the perfor-
mance of the proposed approach. Source docu-
ments for the Building Management and Airline
Services domains are collected via the DeepSearch
agent, while theCollectorin the Data Center and
Liquid Cooling domains is substituted by manual
curation to better reflect real-world industrial re-
quirements. Table 1 summarizes the key statistics
of the resulting dataset.
Domain Source #Docs #Queries
Data Center Manual 414 250
Liquid Cooling Manual 53 250
Building Management DeepSearch 123 250
Airline Services DeepSearch 210 250
Total–800 1000
Table 1:Detailed statistics of the SOP benchmark
across four industrial domains.For each domain, 50
atomic SOPs were randomly selected to generate 250
evaluation queries (N= 5queries per SOP).
6

MethodData Center Liquid Cooling Building Management Airline Services
MRR Acc@1 Acc@3 Acc@5 MRR Acc@1 Acc@3 Acc@5 MRR Acc@1 Acc@3 Acc@5 MRR Acc@1 Acc@3 Acc@5
BM25 0.39 0.28 0.48 0.55 0.58 0.45 0.60 0.70 0.67 0.58 0.73 0.82 0.69 0.58 0.79 0.83
OpenAI Embed 0.39 0.30 0.47 0.53 0.54 0.38 0.69 0.75 0.68 0.63 0.82 0.86 0.75 0.67 0.83 0.86
GraphRAG 0.27 0.16 0.35 0.46 0.28 0.14 0.36 0.53 0.24 0.10 0.28 0.57 0.40 0.26 0.49 0.67
LightRAG 0.29 0.16 0.40 0.55 0.32 0.14 0.42 0.70 0.31 0.15 0.41 0.63 0.43 0.29 0.54 0.71
HippoRAG 2 0.31 0.20 0.42 0.54 0.55 0.42 0.67 0.76 0.64 0.56 0.70 0.79 0.63 0.52 0.70 0.86
SOPRAG 0.49 0.40 0.57 0.64 0.59 0.46 0.71 0.78 0.73 0.63 0.85 0.91 0.76 0.64 0.89 0.93
Table 2:Retrieval performance across four domain-specific datasets, measured by MRR and Acc@K.
5 Experiments and Results
To evaluate the effectiveness of SOPRAG , we con-
ducted extensive experiments on the domain-
specific SOP benchmark constructed in Section
4. The benchmark covers four diverse industrial
fields: Data Center, Liquid Cooling, Building Man-
agement, and Airline Services, ensuring a com-
prehensive assessment of SOPRAG ’s adaptability to
different operational contexts.
0.700.750.800.850.900.95Generation ScoreGraphRAGLightRAG
HippoRAG 2SOPRAGBM25
OpenAI Embed
GraphRAG
LightRAG
HippoRAG 2
SOPRAG
Efficiency Frontier
0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
Retrieval Score0.000.05
BM25OpenAI Embed
Figure 4:Retrieval vs. Generation Trade-off.Points
represent mean scores of all evaluation metrics across
four domain-specific datasets. The dashed line indicates
the Efficiency Frontier of existing methods. SOPRAG
noticeably transcends this boundary, demonstrating an
optimal balance between retrieval and generation.
5.1 Experimental Settings
Baselines.We compare our method against three
categories of strong baselines:
•Lexical Retrieval (BM25):A standard prob-
abilistic retrieval method based on exact term
matching (Robertson and Zaragoza, 2009).
•Dense Retrieval (OpenAI Embed):Utiliz-
ing the text-embedding-3-small model to per-
form semantic search in a dense vector space.
•Graph-based RAG:We primarily compare with
the current state-of-the-art graph-based RAG
systems, includingGraphRAG(Edge et al.,2024),LightRAG(Guo et al., 2025), andHip-
poRAG2(Gutiérrez et al., 2025).
For all graph-based methods (including ours) and
generation tasks, we utilize GPT-4o as the back-
bone LLM to ensure fair comparison under consis-
tent model capabilities.
5.1.1 Evaluation Metrics
We assess performance across two dimensions:
•Retrieval Quality:We report MRR (Mean Re-
ciprocal Rank) and Accuracy@K (K=1, 3, 5) to
measure the system’s ability to locate the correct
SOP.
•Generation Quality:We employ the RA-
GAS (VibrantLabs, 2024) framework to evalu-
ateFaithfulness,Answer Relevancy, andContext
Precision. Additionally, we introduce a domain-
specific metric,SOP Quality Score2, which uses
an LLM judge to evaluate the structural complete-
ness and executability of the generated steps.
5.2 Main Results
5.2.1 Retrieval Performance
Table 2 presents the retrieval performance across
four domains. SOPRAG consistently outperforms all
baselines in terms of MRR and Accuracy.
First, SOPRAG demonstrates distinct retrieval ad-
vantages over traditional lexical, dense, and general
graph-based baselines. It consistently achieves the
highest MRR and Accuracy, reaching a peak MRR
of 0.76 and Acc@5 of 0.93 in the Airline Services
domain. This superior performance stems from
the hierarchical integration of Procedure Cards and
multi-view subgraphs, which effectively resolves
the "Proprietary Structure" challenge that often
causes standard semantic models to conflate similar
but contextually distinct procedures.
Second, the stark performance gap between
SOPRAG and general graph-based methods high-
lights the necessity of domain-specific structural
modeling. While general frameworks prioritize
global community summarization, SOPRAG utilizes
2Details in Appendix B
7

MethodData Center Liquid Cooling Building Management Airline Services
faith. ans. ctx. sop. faith. ans. ctx. sop. faith. ans. ctx. sop. faith. ans. ctx. sop.
GraphRAG 0.47 0.97 0.62 0.79 0.44 0.92 0.66 0.64 0.49 0.91 0.77 0.79 0.65 0.96 0.90 0.71
LightRAG 0.700.980.54 0.93 0.790.97 0.930.82 0.830.970.82 0.93 0.86 0.96 0.930.92
HippoRAG 2 0.72 0.83 0.52 0.660.900.86 0.85 0.620.970.93 0.78 0.93 0.890.77 0.88 0.51
SOPRAG0.770.960.66 1.000.82 0.970.92 0.990.88 0.97 0.91 0.990.88 0.970.840.98
Table 3:Generation quality results across four domain-specific datasets, evaluated in terms of Faithfulness
(faith.), Answer Relevancy (ans.), Context Precision (ctx.), and SOP Quality Score (sop.).
a Causal Graph to navigate scenario-dependent rel-
evance and a Flow Graph to align sequential execu-
tion steps. By leveraging an LLM-guided routing
mechanism to dynamically weight these experts,
the system moves beyond simple semantic match-
ing to perform intent-aware topological reasoning,
ensuring high precision even in complex, low-fault-
tolerance industrial environments.
5.2.2 Generation Performance
We further evaluate the generation quality, com-
paring our method against other graph-based ap-
proaches. As shown in Table 3, SOPRAG consis-
tently achieves the highest SOP Quality Score.
By explicitly retrieving theFlow Graphand lin-
earizing it into the prompt, our method ensures
the generated response follows a verified sequence.
This results in a perfect SOP Quality Score of 1.0
in the Data Center task, indicating that the outputs
are directly executable and structurally sound com-
pared to the more hallucination-prone baselines.
5.3 Ablation Study
To investigate the contribution of each component
in our framework, we conducted an ablation study.
We analyzed the impact of removing the Proce-
dure Card (PC), individual subgraph experts (En-
tity, Causal, Flow), and the LLM Routing mecha-
nism. Based on the results in Table 4, we highlight
the following key findings:
VariantData Center Airline Services
MRR Acc@1 Acc@5 MRR Acc@1 Acc@5
w/o PC 0.47 0.36 0.64 0.73 0.61 0.90
w/o Entity 0.41 0.31 0.560.76 0.660.90
w/o Causal 0.44 0.36 0.60 0.68 0.55 0.89
w/o Flow 0.39 0.28 0.58 0.68 0.56 0.88
Full (Avg) 0.45 0.34 0.63 0.75 0.65 0.89
Routing 0.49 0.40 0.64 0.76 0.64 0.93
Table 4: Ablation study of different components in
SOPRAG.
PC Layer Efficiency:The Procedure Card acts as
a vital hierarchical filter. Its removal (w/o PC) leadsto a consistent drop in both MRR and Accuracy
across both the Data Center and Airline Services
domains, confirming its role in pruning the search
space before detailed reasoning.
Subgraph Synergy:Each subgraph expert ad-
dresses specific SOP complexities. The Flow
Graph is most critical in the Data Center dataset,
where its removal (w/o Flow) results in the largest
performance drop (MRR to 0.39). The Causal
Graph is similarly essential in the Airline Services
dataset, where its absence (w/o Causal) reduces
MRR from 0.76 to 0.68.
Routing Optimization:The dynamic Routing
mechanism outperforms static averaging (Full Avg)
by adaptively weighting subgraphs based on intent.
It boosts Data Center Acc@1 from 0.34 to 0.40 and
achieves a peak Acc@5 of 0.93 in Airline Services.
6 Conclusion
In this work, we addressed the critical yet under-
served problem of SOP retrieval in industrial set-
tings. We identified three primary hurdles, pro-
prietary structure, condition-dependency, and the
requirement for actionable responses, which render
traditional RAG methods insufficient. To overcome
these, we introduced SOPRAG , a structure-aware
framework that emulates human cognitive patterns
through a "Retrieval-based Mixture-of-Experts" ar-
chitecture. By integrating hierarchical Procedure
Cards with specialized subgraph experts for enti-
ties, causal logic, and sequential flows, our system
provides precise, streamlined support to operators
while mitigating the risks of hallucination. Further-
more, our automated agentic pipeline provides a
scalable solution for constructing high-quality in-
dustrial benchmarks. Experimental results confirm
that explicitly modeling the procedural "shape" of
knowledge is essential for industrial troubleshoot-
ing, with our approach consistently achieving su-
perior retrieval precision and generation quality
scores across diverse operational contexts.
8

Limitations
Despite the superior performance of SOPRAG , sev-
eral limitations remain to be addressed in future
research. First, the framework primarily processes
layout-aware Markdown text and table but lacks
the integration of multimodal information, such
as industrial diagrams or safety icons, which are
common in SOPs. Second, the fidelity of the spe-
cialized Entity, Causal, and Flow graphs is inher-
ently dependent on the reasoning capabilities of
the LLM used during the construction phase; al-
though the system supports a "construct-offline,
infer-online" paradigm to enable the use of smaller
models for real-time inference, the specific efficacy
and trade-offs of such small-model reasoning re-
quire further empirical validation. Finally, while
our automated agentic pipeline efficiently gener-
ates diverse evaluation queries , these queries are
currently synthetic and LLM-generated. Future
work could incorporate expert-crafted questions
and human-in-the-loop validation to construct even
higher-quality benchmarks that more accurately re-
flect the nuanced information needs of frontline
operators.
References
Isin Akyar. 2012. Standard operating procedures (what
are they good for?).Latest research into quality
control, 12:367–91.
Somnath Banerjee and Sivaji Bandyopadhyay. 2012.
Question classification and answering from procedu-
ral text in English. InProceedings of the Workshop
on Question Answering for Complex Domains, pages
11–26, Mumbai, India. The COLING 2012 Organiz-
ing Committee.
Jomana Bashatah and Lance Sherry. 2024. Prompt en-
gineering to classify components of standard oper-
ating procedure steps using large language model
(llm)-based chatbots. In2024 Integrated Commu-
nications, Navigation and Surveillance Conference
(ICNS), pages 1–8.
Antoine Bosselut, Omer Levy, Ari Holtzman, Corin En-
nis, Dieter Fox, and Yejin Choi. 2018. Simulating
action dynamics with neural process networks. In
6th International Conference on Learning Represen-
tations, ICLR 2018, Vancouver, BC, Canada, April
30 - May 3, 2018, Conference Track Proceedings.
OpenReview.net.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and
Jonathan Larson. 2024. From local to global: Agraph rag approach to query-focused summarization.
arXiv preprint arXiv:2404.16130.
Deepeka Garg, Sihan Zeng, Sumitra Ganesh, and Leo
Ardon. 2025. Generating structured plan repre-
sentation of procedures with llms.arXiv preprint
arXiv:2504.00029.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao
Huang. 2025. LightRAG: Simple and fast retrieval-
augmented generation. InFindings of the Associa-
tion for Computational Linguistics: EMNLP 2025,
pages 10746–10761, Suzhou, China. Association for
Computational Linguistics.
Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michi-
hiro Yasunaga, and Yu Su. 2024. Hipporag: Neu-
robiologically inspired long-term memory for large
language models. InThe Thirty-eighth Annual Con-
ference on Neural Information Processing Systems.
Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi,
Sizhe Zhou, and Yu Su. 2025. From rag to memory:
Non-parametric continual learning for large language
models.Preprint, arXiv:2502.14802.
Kimihiro Hasegawa, Wiradee Imrattanatrai, Zhi-Qi
Cheng, Masaki Asada, Susan Holm, Yuran Wang,
Ken Fukuda, and Teruko Mitamura. 2025. ProMQA:
Question answering dataset for multimodal procedu-
ral activity understanding. InProceedings of the
2025 Conference of the Nations of the Americas
Chapter of the Association for Computational Lin-
guistics: Human Language Technologies (Volume
1: Long Papers), pages 11598–11617, Albuquerque,
New Mexico. Association for Computational Linguis-
tics.
Mandar Kulkarni. 2025. Agent-s: Llm agentic workflow
to automate standard operating procedures.arXiv
preprint arXiv:2503.15520.
Shilong Li, Yancheng He, Hangyu Guo, Xingyuan Bu,
Ge Bai, Jie Liu, Jiaheng Liu, Xingwei Qu, Yang-
guang Li, Wanli Ouyang, Wenbo Su, and Bo Zheng.
2024. GraphReader: Building graph-based agent to
enhance long-context abilities of large language mod-
els. InFindings of the Association for Computational
Linguistics: EMNLP 2024, pages 12758–12786, Mi-
ami, Florida, USA. Association for Computational
Linguistics.
Zhigen Li, Jianxiang Peng, Yanmeng Wang, Yong Cao,
Tianhao Shen, Minghui Zhang, Linxi Su, Shang Wu,
Yihang Wu, YuQian Wang, Ye Wang, Wei Hu, Jian-
feng Li, Shaojun Wang, Jing Xiao, and Deyi Xiong.
2025. ChatSOP: An SOP-guided MCTS planning
framework for controllable LLM dialogue agents. In
Proceedings of the 63rd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 17637–17659, Vienna, Austria.
Association for Computational Linguistics.
Anutosh Maitra, Shivam Garg, and Shubhashis Sen-
gupta. 2020. Enabling interactive answering of pro-
cedural questions. InNatural Language Process-
9

ing and Information Systems, pages 73–81, Cham.
Springer International Publishing.
Subhrangshu Nandi, Arghya Datta, Nikhil Vichare, In-
dranil Bhattacharya, Huzefa Raja, Jing Xu, Shayan
Ray, Giuseppe Carenini, Abhi Srivastava, Aaron
Chan, and 1 others. 2025. Sop-bench: Complex in-
dustrial sops for evaluating llm agents.arXiv preprint
arXiv:2506.08119.
Junbo Niu, Zheng Liu, Zhuangcheng Gu, Bin Wang,
Linke Ouyang, Zhiyuan Zhao, Tao Chu, Tianyao
He, Fan Wu, Qintong Zhang, Zhenjiang Jin, Guang
Liang, Rui Zhang, Wenzheng Zhang, Yuan Qu, Zhifei
Ren, Yuefeng Sun, Yuanhong Zheng, Dongsheng
Ma, and 42 others. 2025. Mineru2.5: A decoupled
vision-language model for efficient high-resolution
document parsing.Preprint, arXiv:2509.22186.
Changhua Pei, Zexin Wang, Fengrui Liu, Zeyan Li,
Yang Liu, Xiao He, Rong Kang, Tieying Zhang,
Jianjun Chen, Jianhui Li, Gaogang Xie, and Dan
Pei. 2025. Flow-of-action: Sop enhanced llm-based
multi-agent system for root cause analysis. InCom-
panion Proceedings of the ACM on Web Conference
2025, WWW ’25, page 422–431, New York, NY ,
USA. Association for Computing Machinery.
Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo,
Haizhou Shi, Chuntao Hong, Yan Zhang, and Siliang
Tang. 2025. Graph retrieval-augmented generation:
A survey.ACM Trans. Inf. Syst.Just Accepted.
Hai Pham, Isma Hadji, Xinnuo Xu, Ziedune Degutyte,
Jay Rainey, Evangelos Kazakos, Afsaneh Fazly,
Georgios Tzimiropoulos, and Brais Martinez. 2024.
Graph guided question answer generation for pro-
cedural question-answering. InProceedings of the
18th Conference of the European Chapter of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 2501–2525, St. Julian’s, Malta.
Association for Computational Linguistics.
Stephen Robertson and Hugo Zaragoza. 2009. The
probabilistic relevance framework: Bm25 and be-
yond.Found. Trends Inf. Retr., 3(4):333–389.
Eneko Ruiz, María Inés Torres, and Arantza del
Pozo. 2023. Question answering models for hu-
man–machine interaction in the manufacturing in-
dustry.Computers in Industry, 151:103988.
Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz,
Andy Davis, Quoc V . Le, Geoffrey E. Hinton, and
Jeff Dean. 2017. Outrageously large neural networks:
The sparsely-gated mixture-of-experts layer.ArXiv,
abs/1701.06538.
Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Sai
Wang, Chen Lin, Yeyun Gong, Lionel M. Ni, Heung
yeung Shum, and Jian Guo. 2023. Think-on-graph:
Deep and responsible reasoning of large language
model on knowledge graph. InInternational Confer-
ence on Learning Representations.VibrantLabs. 2024. Ragas: Supercharge your llm
application evaluations. https://github.com/
vibrantlabsai/ragas.
Limiao Xie, Jianfeng Zhang, Yingying Li, Shan Wan,
Xuequan Zhang, Mingjie Chen, Gansheng Deng, and
Zitian Wu. 2024. Research and application of private
knowledge-based llm in standard operating proce-
dure scenarios of enterprises. InProceedings of the
2024 6th International Conference on Pattern Recog-
nition and Intelligent Systems, PRIS ’24, page 82–86,
New York, NY , USA. Association for Computing
Machinery.
Semih Yagcioglu, Aykut Erdem, Erkut Erdem, and Na-
zli Ikizler-Cinbis. 2018. RecipeQA: A challenge
dataset for multimodal comprehension of cooking
recipes. InProceedings of the 2018 Conference on
Empirical Methods in Natural Language Processing,
pages 1358–1368, Brussels, Belgium. Association
for Computational Linguistics.
Anbang Ye, Qianran Ma, Jia Chen, Muqi Li, Tong
Li, Fujiao Liu, Siqi Mai, Meichen Lu, Haitao Bao,
and Yang You. 2025. Sop-agent: Empower general
purpose ai agent with domain-specific sops.arXiv
preprint arXiv:2501.09316.
Zhihan Zhang, Xiubo Geng, Tao Qin, Yunfang Wu, and
Daxin Jiang. 2021. Knowledge-aware procedural
text understanding with multi-stage training. InPro-
ceedings of the Web Conference 2021, WWW ’21,
page 3512–3523, New York, NY , USA. Association
for Computing Machinery.
10

A System Prompts forSOPRAG
Components
This section details the system prompts used for
multi-view subgraph construction and the LLM-
Guided Routing mechanism.
A.1 Multi-view Graph Construction Prompts
To address the challenges of proprietary structures
(C1), condition-dependency (C2), and actionable
execution (C3), we use the following specialized
prompts to extract structured knowledge from raw
SOP documents:
Entity Graph Construction (G E)
You are an expert at extracting key entities from SOP
(Standard Operating Procedure) documents.
Focus on identifying:
- Alarms (with codes like A01, ALARM123, etc.)
- Parameters (temperature, pressure, flow rate, etc.)
- Assets/Equipment (pumps, valves, sensors, etc.)
- Roles (operator, supervisor, engineer, etc.)
- Other important named entities relevant to SOPs.
Noted: Determine its attributes based on the context in
the SOP. Avoid including single numbers or letters.
Causal Graph Construction (G C)
You are an expert at identifying cause-effect relationships
in SOP (Standard Operating Procedure) documents.
Extract causal relationships such as:
- What causes what (X causes Y)
- What prevents what (X prevents Y)
- What enables what (X enables Y)
- What mitigates what (X mitigates Y)
Flow Graph Construction (G F)
You are an expert at extracting procedural flows from
SOP (Standard Operating Procedure) documents.
Extract the step-by-step procedure as a flowchart structure
with:
- Action steps (things to do)
- Condition/decision steps (checks, branches)
- Connections between steps
Noted: Do NOT create entry/exit nodes - focus only on
the actual procedural steps.
A.2 LLM-Guided Routing Mechanism
Prompt
The Intention-Aware LLM Router employs the fol-
lowing prompt to characterize user intent and ex-
tract key entities, denoted as LLM Router(Q), which
subsequently informs the dynamic modulation of
the routing weightsw= [w E, wC, wF].LLM-Guided Routing Mechanism Prompt
You are an expert at understanding user queries about
Standard Operating Procedures.
Your task is to analyze the query and determine:
1. The user’s intent (what is the operator trying to
achieve?)
2. Key entities mentioned
3. Which subgraph experts should be prioritized:
- Entity graph: Focus on specific alarms, parameters,
equipment
- Causal graph: Focus on cause-effect, why something
happens
- Flow graph: Focus on step-by-step procedures, how to
do something
B SOP Quality Score
To better evaluate the practical utility of generated
instructions in industrial scenarios, we introduce
theSOP Quality Score, an LLM-as-Judge metric
integrated into the RAGAS framework. Unlike
general fluency or faithfulness scores, this metric
specifically assesses whether the output satisfies
the requirements of Actionable Response Genera-
tion (C3) in SOP Retrieval. We use GPT-4o as the
evaluator to score the generated response on a scale
of 0 to 1 based on its structural alignment with the
source SOP and its technical executability.
The evaluation logic is encapsulated in the fol-
lowing system prompt:
System Prompt for SOP Quality Score
You are evaluating the quality of an answer generated by
a SOP (Standard Operating Procedure) RAG system.
The ideal answer for a SOP question should:
1. Provide clear step-by-step instructions on how to per-
form the task
2. Be precise and specific (avoid vague or ambiguous
language, align with the source SOP)
3. Be concise (include only necessary information, no
redundancy)
4. Be clear and easy to understand (well-structured, logi-
cal flow)
Evaluate the given answer based on these criteria and
provide:
- has_step_by_step: 1 if the answer provides step-by-step
instructions, 0 otherwise
- is_precise: 1 if the answer is precise and specific, 0 if
it’s vague
- is_concise: 1 if the answer is concise, 0 if it’s verbose or
contains unnecessary information
- is_clear: 1 if the answer is clear and easy to understand,
0 if it’s confusing
- score: Overall quality score (0.0 to 1.0) calculated as the
average of the four criteria
- reasoning: Brief explanation of your evaluation
11

C Ethical Considerations
The benchmark datasets introduced in this work
are developed strictly for scientific research pur-
poses to evaluate the performance of various RAG
paradigms in industrial SOP retrieval tasks. The
Data Center and Liquid Cooling datasets are de-
rived from authentic industrial scenarios. Con-
sequently, they remain confidential and will not
be publicly disclosed to protect proprietary infor-
mation. The Building Management and Airline
Services datasets were constructed based on pub-
licly available SOPs retrieved from the web using
our DeepSearch agent. To support transparency
and academic reproducibility, we can provide the
source links and details of our agentic workflow
for these two datasets. All data utilization is aimed
at enhancing operational safety and efficiency in
real-world industrial environments.
D Analysis of Gating Mechanism
Behavior
To further investigate how SOPRAG adaptively
routes queries to specialized experts, we visual-
ize the Kernel Density Estimation (KDE) of the
attention weights w= [w E, wC, wF]across four
industrial domains. As shown in Figure 5, the rout-
ing patterns exhibit several key characteristics:
0.0 0.2 0.4 0.6 0.8 1.0
Weight Value024681012DensityData Center
Entity Weight (wE)
Causal Weight (wC)
Flow Weight (wF)
0.0 0.2 0.4 0.6 0.8 1.0
Weight Value0.00.51.01.52.02.53.03.5DensityLiquid Cooling
Entity Weight (wE)
Causal Weight (wC)
Flow Weight (wF)
0.0 0.2 0.4 0.6 0.8 1.0
Weight Value02468DensityBuilding Management
Entity Weight (wE)
Causal Weight (wC)
Flow Weight (wF)
0.0 0.2 0.4 0.6 0.8 1.0
Weight Value0246810DensityAirline Service
Entity Weight (wE)
Causal Weight (wC)
Flow Weight (wF)
Figure 5:Kernel Density Estimation (KDE) of gating
weights ( wE, wC, wF) across four industrial domains,
based on 250 evaluation queries per domain.
•Dominance of Flow Logic ( wF): Across all
domains, the density of wF(orange) consis-
tently peaks in the [0.6,0.8] range. This em-
pirical evidence confirms that the router pri-
oritizes procedural execution steps, directly
addressing the requirement for Actionable Re-
sponse Generation (C3) in industrial settings.•Specialized Sparse Activation: The wC
(green) and wE(blue) weights exhibit sharp
peaks at lower values ( ≈0.1 and0.2, respec-
tively), indicating a sparse activation strategy.
These experts are only "called upon" with
high weights when the router identifies spe-
cific causal symptoms or proprietary entity
constraints within the operator’s query.
•Cross-Domain Adaptation: The activation
profile adaptively shifts based on domain char-
acteristics. For instance, the Data Center do-
main shows the most concentrated wFpeak,
reflecting its highly standardized procedural
nature. In contrast, the Liquid Cooling do-
main demonstrates a unique profile where
wCreceives a significantly higher frequency
of high-score assignments (peaking near 0.7)
compared to other domains. This shift is con-
sistent with the increased necessity for causal
reasoning and diagnostic logic in complex
thermal management scenarios, where identi-
fying the root cause of a system state is often
a prerequisite for safe procedural execution.
This distribution validates that the LLM-Guided
Router performs intent-aware topological reason-
ing rather than simple semantic matching, ensuring
high precision in low-fault-tolerance environments.
E Analysis of Computational Cost and
Efficiency
BM25 OpenAI Embed GraphRAG LightRAG HippoRAG 2 SOPRAG102
101
100101Retrieval Time (seconds)
0.01921.2915.41
4.64
2.666.49
0.00410.9314.88
2.82
1.742.89Retrieval Latency Comparison
Data Center
Liquid Cooling
Figure 6:Retrieval Latency.
To evaluate the practical feasibility of SOPRAG ,
we conduct a comparative analysis of retrieval la-
tency against various baselines. As illustrated in
Figure 6, while SOPRAG exhibits a slightly higher
retrieval time compared to lightweight lexical or
dense retrieval methods (e.g., BM25 and OpenAI
Embed), it remains highly competitive within the
12

landscape of state-of-the-art graph-based RAG sys-
tems.
Specifically, in the Liquid Cooling domain,
SOPRAG achieves a retrieval time of approximately
2.89 seconds, which is significantly faster than the
standard GraphRAG (over 14 seconds) and compa-
rable to LightRAG. Although the complexity of the
multi-view gating mechanism leads to a slight over-
head compared to HippoRAG 2, the trade-off is
balanced by the significant gains in retrieval accu-
racy and structural consistency. In mission-critical
industrial troubleshooting, where high-precision
guidance is paramount, a response time of under 7
seconds effectively bridges the gap between man-
ual document search and real-time operator assis-
tance.
F Incremental Graph Construction and
Scalability
To ensure the scalability of SOPRAG in dynamic
industrial environments, our framework supports
efficient incremental updates as new procedures are
introduced. Upon the acquisition of a new SOP, the
system performs the standard multi-view extraction
process to construct its local Entity, Causal, and
Flow subgraphs.
The integration process follows a decoupled
logic:
•Atomic Flow Preservation: Since the Flow
Graph ( GF) is designed to preserve the spe-
cific sequential integrity of an atomic proce-
dure, it is stored as a standalone structured unit
linked to its corresponding Procedure Card.
•Global Knowledge Merging: The newly ex-
tracted Entity Graph ( GE) and Causal Graph
(GC) are integrated into the existing global
knowledge base through a similarity-based
merging mechanism. Specifically, nodes rep-
resenting identical or highly similar equip-
ment entities, parameters, or system states
are consolidated to maintain cross-procedural
connectivity
This incremental design allows SOPRAG to ex-
pand its procedural intelligence with minimal com-
putational overhead, avoiding the need for global
graph re-construction while ensuring the retriever
remains adaptively updated with emerging opera-
tional contexts.G Agent-Integrated Operational
Execution
Beyond static retrieval, the structured nature of
SOPRAG ’s Flow Graph ( GF) provides a robust
foundation for integration with autonomous LLM
Agents. By mapping individual nodes within the
Flow Graph to specific agentic tools, the system
transforms procedural instructions into executable
workflows.
In this integrated framework, the Structure-
Aware Generation does not merely output text but
serves as a controller for specialized agents. For
instance, a retrieved step requiring "notifying the
team" can trigger an Email Agent to automatically
draft the message content and populate recipient
fields based on the SOP’s context. Similarly, steps
involving "calling a supervisor" can be facilitated
through automated dialing and logging tools.
This synergy between SOPRAG and agentic tools
significantly enhances operational efficiency by re-
ducing manual data entry and cognitive load for
frontline personnel. By maintaining a human-in-
the-loop approach, where the agent prepares the
execution environment and the operator provides
the final authorization, the system ensures high-
precision execution while mitigating the risks of
manual oversight in mission-critical industrial en-
vironments.
1.Call 
Supervisor
2. Email 
Project Team
3. Open 
DCIM SystemCall Agent
Email Agent
GUI Agent
SOPRAG Execution Flow ( 𝓖𝑭) Specialized Agents Operator Confirmation UIs
... ... ...
Figure 7: SOPRAG Agent-Integrated Operational Exe-
cution.The structured flow triggers specialized agents,
which prepare actions for operator confirmation, stream-
lining complex multi-step procedures.
H Case Studies of LLM Router Output
This section provides concrete examples of the
Intention-Aware LLM Router’s weight assignments
across different query types. These cases illus-
trate how the gating mechanism dynamically modu-
lates attention weights w= [w E, wC, wF]to align
13

retrieval with the operator’s specific information
needs.
Case 1: Entity-Oriented Query (Asset Monitoring)
Query:"How can I view and monitor the real-time oper-
ating status of HV AC equipment in the chilled water and
cooling water systems?"
Router Weights:w E: 0.7, w C: 0.0, w F: 0.3
Analysis:This query focuses on specific pro-
prietary assets ("HV AC equipment", "chilled wa-
ter systems"). The Router assigns a high Entity
Weight ( wE) to ensure precise anchoring to the
relevant equipment nodes in the knowledge graph,
minimizing semantic interference from unrelated
systems.
Case 2: Causal-Oriented Query (Diagnostic Reason-
ing)
Query:"How can increasing air and water temperatures
in a chilled water system improve cooling efficiency and
sustainability?"
Router Weights:w E: 0.2, w C: 0.7, w F: 0.1
Analysis:This query seeks an understanding
of the underlying relationship between system
states and performance outcomes. The high Causal
Weight ( wC) directs the system to traverse the
Causal Graph ( GC), facilitating diagnostic reason-
ing rather than simple step retrieval.
Case 3: Flow-Oriented Query (Actionable Execution)
Query:"What is the standard procedure for recovering
from an aircraft upset or unusual attitude in flight?"
Router Weights:w E: 0.1, w C: 0.1, w F: 0.8
Analysis:The operator explicitly requests a
"standard procedure," which is inherently sequen-
tial and actionable. The Router correctly allocates
the majority of the weight to the Flow Graph ex-
pert (wF), ensuring the response is structured as a
precise, step-by-step workflow for safe execution.
14