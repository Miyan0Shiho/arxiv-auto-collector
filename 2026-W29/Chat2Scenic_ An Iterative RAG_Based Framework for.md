# Chat2Scenic: An Iterative RAG-Based Framework for Scenario Generation in Autonomous Driving

**Authors**: Yuan Gao, Wenting Miao, Mattia Piccinini, Haoyu Wang, Qunying Song, Johannes Betz

**Published**: 2026-07-15 22:02:03

**PDF URL**: [https://arxiv.org/pdf/2607.14387v1](https://arxiv.org/pdf/2607.14387v1)

## Abstract
Validating autonomous driving systems requires diverse, regulation-compliant test scenarios. In simulation-based testing, scenarios are defined as executable scripts. Yet automatically generating such scripts from regulatory descriptions remains an open challenge, and existing approaches face fundamental trade-offs. Retrieval-assemble methods achieve reasonable compilation rates but lack scalability, whereas retrieval-based full-script generation suffers from low compilation success rates. We present Chat2Scenic, the first iterative retrieval-augmented framework to generate scenario scripts in Domain Specific Language (DSL). Specifically, Chat2Scenic provides a chatbot interface that supports interactive scenario refinement and integrates Retrieval-augmented Generation (RAG) to ground scenario generation in regulatory knowledge and DSL syntax. Furthermore, we propose an open benchmark for scenario generation comprising 123 scenarios from various regulations, including NHTSA and United Nations Vehicle Regulations, as well as other sources. Extensive evaluation with State-of-the-Art (SOTA) Large Language Models (LLMs) demonstrates that Chat2Scenic achieves 76.42% Compilation Success Rate (CSR) and 58.17% Framework Accuracy (FA), outperforming existing methods (Retrieval Assemble with 30.08% CSR, 11.03% FA and Retrieval full script generation with 16.26% CSR, 10.86% FA). To facilitate future research, we release our code as open source at https://github.com/TUM-AVS/chat2scenic.

## Full Text


<!-- PDF content starts -->

Chat2Scenic: An Iterative RAG-Based Framework for Scenario Generation
in Autonomous Driving
Yuan Gao1,∗, Wenting Miao1,∗, Mattia Piccinini1, Haoyu Wang1, Qunying Song2, Johannes Betz1
Abstract— Validating autonomous driving systems requires
diverse, regulation-compliant test scenarios. In simulation-
based testing, scenarios are defined as executable scripts. Yet au-
tomatically generating such scripts from regulatory descriptions
remains an open challenge, and existing approaches face funda-
mental trade-offs. Retrieval-assemble methods achieve reason-
able compilation rates but lack scalability, whereas retrieval-
based full-script generation suffers from low compilation suc-
cess rates. We present Chat2Scenic, the first iterative retrieval-
augmented framework to generate scenario scripts in Domain
Specific Language (DSL). Specifically, Chat2Scenic provides
a chatbot interface that supports interactive scenario refine-
ment and integrates Retrieval-augmented Generation (RAG)
to ground scenario generation in regulatory knowledge and
DSL syntax. Furthermore, we propose an open benchmark
for scenario generation comprising 123 scenarios from various
regulations, including NHTSA and United Nations Vehicle
Regulations, as well as other sources. Extensive evaluation
with State-of-the-Art (SOTA) Large Language Models (LLMs)
demonstrates that Chat2Scenic achieves 76.42% Compila-
tion Success Rate (CSR) and 58.17% Framework Accuracy
(FA), outperforming existing methods (Retrieval Assemble with
30.08% CSR, 11.03% FA and Retrieval full script generation
with 16.26% CSR, 10.86% FA). To facilitate future research, we
release our code as open source athttps://github.com/
TUM-AVS/chat2scenic.
I. INTRODUCTION
Autonomous driving has matured significantly, with com-
panies like Waymo deploying SAE Level 4 robotaxis [1] in
urban environments, demonstrating fully-autonomous capa-
bilities within specific Operational Design Domain (ODD).
This progress requires developing and validating highly
reliable Automated Driving System (ADS). Traditional val-
idation approaches depend on real-world testing, such as
on-road trials [2]. However, physical testing alone cannot
adequately capture the diversity of driving situations and
edge cases. To overcome these limitations, researchers and
industry have increasingly adopted virtual scenario-based
testing, which enables cost-effective simulation of diverse
and realistic driving conditions [3]. Recent advances in Large
Language Models (LLMs) offer such potential: by leveraging
knowledge of Domain Specific Language (DSL) syntax,
they can automatically translate natural language scenario
descriptions into executable scripts in DSL.
Despite this progress, LLM-based generation of diverse
and regulation-grounded scenarios remains challenging. As
1Y . Gao, W. Miao, M. Piccinini, H. Wang and J. Betz are with the
Professorship of Autonomous Vehicle Systems, TUM School of Engineering
and Design, Technical University of Munich, 85748 Garching, Germany;
Munich Institute of Robotics and Machine Intelligence (MIRMI)
2Q. Song is with University College London, London, United Kingdom
∗Equal contribution.
Fig. 1: Comparison of DSL-based scenario generation ap-
proaches and Chat2Scenic. While prior methods assemble
existing DSL components or directly generate DSL pro-
grams, Chat2Scenic iteratively generates DSL blocks that in-
teract with the database to create complex driving scenarios.
shown in Fig. 1, existing automated approaches primarily
rely on simplified scenario descriptions rather than complex
ADS testing regulatory specifications. Moreover, current
methods face fundamental trade-offs: retrieval-assemble ap-
proaches [4]–[6] achieve reasonable compilation rates by
assembling pre-existing DSL code snippets but lack gen-
eralizability to new scenarios, while direct generation [7]–
[12] offers better flexibility but suffers from low compilation
success rates. This motivates us to develop a framework
that can handle complex ADS testing regulatory descriptions
while maintaining both high compilation success and gener-
alizability through iterative component-wise generation.
A. Related Work
1) Classical Scenario Generation:Many studies have ex-
plored the generation of ADS testing scenarios for scenario-
based testing. One impactful survey [13] categorizes gener-
ation methods into data-driven, adversarial, and knowledge-
based approaches. In more detail, data-driven methods lever-
age real-world datasets to generate realistic scenarios. Ad-
versarial approaches aim to expose ADS weaknesses by
synthesizing rare but high-risk situations through techniques.
Knowledge-based methods integrate domain expertise, such
as traffic rules, to enhance controllability. Aligned with the
ISO/WD PAS 21448 Safety of the Intended Functional-
This work has been accepted to the IEEE IROS 2026. Copyright may be transferred without notice, after which this version may no longer be accessible.
arXiv:2607.14387v1  [cs.AI]  15 Jul 2026

TABLE I: Comparison of LLM-based Scenario Generation
with DSL format
Paper UIRAG Prompting TechniquesReg.G
C D CP CoT ICL RAG-ICLAssemble2023 TARGET [4]× × ×✓×✓× ×C
2024 ChatScene [5]×✓× × ×✓× ×C
2025 Text2Scenario [6]× × × ×✓ ✓× ×CGeneration2024 LEADE [7]× × × ×✓ ✓× ×F
2024 LeGEND [8]× × × × ×✓× ×F
2024 ScenicNL [9]× × ×✓ ✓ ✓× ×I
2024 Talk2Traffic [10]✓ ✓ ✓×✓ ✓× ×C
2025 Rubavicius [11]✓ ✓× × ×✓ ✓×F
2025 NL2Scenic [12]×✓×✓ ✓ ✓ ✓×F
Ours ✓✓✓✓✓✓ ✓ ✓ I
UI: User Interaction. C: Code. D: Documentation RAG. CP: Contextual Prompting.
CoT: Chain of Thought. ICL: In-Context Learning. RAG-ICL: Retrieval-Augmented
In-Context Learning. Reg.: Regulatory Test Specifications. G - Granularity: F (Full
DSL), C (Component-wise), I (Iterative Component-wise).
ity (SOTIF) standard, Schutt et al. [14] examine scenario
generation across functional, logical, and concrete abstrac-
tion levels, covering machine learning-based generation,
optimization-driven exploration, scenario extraction from
driving data, and manual design.
2) LLM-based Scenario Generation:LLMs have emerged
as powerful tools for generating scenarios directly from
natural language inputs [15]. Depending on the application,
the underlying simulators fall into two categories: vehicle-
centric simulators (e.g., CARLA [16]) and traffic-centric
simulators (e.g., SUMO [17]) for large-scale traffic gener-
ation. For traffic scenario generation, ChatSUMO [18] and
OmniTester [19] leveraged LLMs to produce realistic urban
traffic simulations from text, showcasing the scalability of
language-driven scenario generation.
Recent survey [15] shows that LLMs have been used
to generate scenarios in CARLA for autonomous driving
based on DSL formats like Scenic [5], [9]–[12] or OpenSce-
nario [4], [6], or using simulator APIs [20]–[24]. However,
API methods depend on large databases and mapping rules
that bind LLMs-extracted elements to predefined templates,
limiting expressiveness. DSLs provide a formal and human-
readable representation for parameterized driving scenarios,
corresponding to logical-level specifications that can be
instantiated into concrete simulation setups. These DSL-
based methods are typically divided into two categories:
Retrieval Assemble and Direct Generation, as shown in
Table I. Retrieval Assemble retrieves the code snippets from
the database or other code repositories and assembles them
into a complete DSL script. Direct Generation generates the
DSL script from scratch with or without retrieval.
Retrieval Assemble:TARGET [4] parses traffic rules into
logical scenarios via multi-stage prompting, then assembles
concrete scripts through dictionary lookup and hierarchical
map-based route retrieval. ChatScene [5] retrieves and as-
sembles code snippets mapping behaviors to Scenic imple-
mentations using LLM with Retrieval-augmented Generation
(RAG). Text2Scenario [6] performs LLM-based hierarchical
scenario parsing with subsequent priority-based DSL as-sembly to generate executable test cases. However, these
retrieval-assemble methods lack generalizability to novel
scenarios beyond their predefined code databases.
Direct Generation:LEADE [7] employs multi-modal
prompting to generate executable scenario DSL from abstract
representations completely, followed by dual-layer optimiza-
tion search for safety violations. LeGEND [8] uses two-
stage LLM processing: translating reports into parameter
patterns inserted into predefined templates, then generating
concrete scripts from logical specifications. ScenicNL [9]
chains multiple LLMs with constrained decoding and uses
compiler feedback for iterative refinement. Talk2Traffic [10]
performs LLM-based retrieval-augmented snippet generation
independently and followed by LLM-based component as-
sembly. Rubavicius [11] employs retrieval-augmented di-
rect code generation with multi-turn dialogue refinement
for Scenic-based autonomous driving scenario synthesis.
NL2Scenic [12] adopts full-program LLM-based Scenic gen-
eration with optional retrieval-augmented prompting. How-
ever, direct generation approaches suffer from low com-
pilation rates due to the difficulty of generating complete
scripts or assembling newly generated components, which
are individually valid but mutually incompatible.
B. Critical Summary
To the best of our knowledge, the existing literature is
limited by at least one of the following aspects:
1) Limited generative paradigms: Retrieval-based assem-
bly methods [4]–[6] construct DSL programs from
stored code snippets, reducing generalization and sce-
nario diversity.
2) Compilation reliability and evaluation gaps: SOTA
methods focusing on direct DSL generation via ad-
vanced prompting techniques [7], [9]–[12] face com-
pilation reliability challenges, as LLMs generate either
full scripts or components to assemble. Moreover,
these approaches lack a comprehensive evaluation of
compilation success rates and semantic accuracy.
3) Limited input complexity: Most frameworks focus on
simplified scenario descriptions rather than regulation-
grounded specifications, which are essential for ADS
validation in real-world deployment.
C. Contribution
The key contributions of this paper are as follows:
1) We proposeChat2Scenic, a chatbot-based iterative
component-wise generation framework with advanced
prompting techniques and RAG, outperforming State-
of-the-Art (SOTA) methods with publicly available
code across compilation success rate, semantic accu-
racy, and framework accuracy.
2) We curate 120+ ADS scenario descriptions from the
CARLA Leaderboard, NHTSA, and United Nations
Vehicle Regulations, and propose dedicated metrics
to evaluate both framework-level performance (e.g.,
compilation success rate and framework accuracy) and
application-level performance (semantic accuracy).

Fig. 2: Overview of the Chat2Scenic framework, consisting of three core modules: theInteractive Moduleinteractively
refines and interprets user queries into logical components; theGeneration Modulesynthesizes related DSL snippets based
on these components, integrating theRAG Moduleto retrieve domain specific knowledge.
3) We evaluate Chat2Scenic across closed- and open-
source LLMs, conduct ablation studies on prompting
techniques, and benchmark against SOTA methods.
II. METHODOLOGY
Chat2Scenic consists of three parts (Fig. 2): an interactive
module, a RAG module, and a generation module, followed
by a simulation interface. The interactive module parses the
input description into a logical structured representation and
supports interactive refinement. The RAG module retrieves
domain-specific knowledge from a DSL snippet database and
a documentation database. The generation module iteratively
generates Scenic code components until a complete Scenic
program is produced, which is compatible with 3D simula-
tors such as CARLA [16].
A. Interactive Module
Chat2Scenic provides an interactive framework through
a chatbot interface supported by Gradio1, as illustrated in
Fig. 3. Users can describe scenarios in natural language,
query regulation-grounded scenarios retrieved from RAG
databases, and iteratively refine their description.
1) Logical Structure Schema:To enable controllable,
modular scenario generation, we represent each user request
using a logical structureSby identifying the building blocks
of the DSL. Specifically, we define the structured scenario
1https://www.gradio.app/representationS(Fig. 3) as a set of components composed
of global configurationGand scenario componentsS int:
S={G} ∪ S int={G, R, E, O, T}(1)
Here,Grepresents the global configuration (map, weather,
vehicle models) and is independently extracted by the
Global Configuration Generator(Section II-C.1). TheIn-
terpreterextracts the scenario-specific componentsS int=
{R, E, O, T}, whereRrepresents spatial relations (e.g., road
topology and relative positioning of entities),Edenotes ego
behavior (e.g., ego parameters),O={O 1, ..., O N}is the
set ofNobjects (e.g., adversarial vehicles), andTspecifies
restrictions (e.g., initial conditions and termination criteria).
2) Interpreter:We employ an LLM-based interpreter in
the workflow. The interpreter is prompted using explicit
instructions and few-shot examples to extractS intfrom
scenario descriptions. Each component is abstracted into a
single-sentence description, which is later used as a semantic
query for component code generation. This abstraction aligns
the interpretation with the structure of code snippets in
the database and the Scenic syntax, facilitating effective
retrieval and generation. The interpreter uses LangGraph’s
checkpointing mechanism2with MemorySaver to maintain
conversation state, enabling iterative refinement where users
can provide feedback to update the logical interpretation
without restarting the workflow.
2https://www.langchain.com/langgraph

Fig. 3: Chatbot UI for interactive refinement.
B. RAG Module
To enhance the LLM performance for domain-specific
tasks, we integrate RAG supported by the LangChain3
framework. The RAG technique retrieves relevant exter-
nal knowledge and returns it to the LLM during scenario
generation, improving accuracy and reducing hallucinations.
Specifically, we construct two complementary databases: a
code snippet database and a documentation database, and
design specialized retrievers for each.
1) Code Snippet Database:We construct a component-
level database by collecting codes from official Scenic
sources [25]. Each scenario is decomposed into independent
component-level code units covering theS intcomponents
(R,E,O,T). For each component, an LLM generates
a concise single-sentence description (e.g., “A car travels
forward and makes a right turn”). Each entry stores both the
natural language description and the corresponding Scenic
code snippet. The description is embedded using sentence
transformers (all-MiniLM-L6-v24) for semantic retrieval,
while the code snippet is returned as context to the generator.
2) Documentation Database:We construct a documenta-
tion database from official Scenic documentation and reg-
ulatory documents (e.g., UN Vehicle Regulations). Docu-
ments are collected via web crawling, cleaned to remove
irrelevant content, and split into chunks using structure-
aware hierarchical separators to preserve document hierarchy.
Chunks are then embedded using an embedding model for
semantic matching and indexed with Best Match 25 (BM25)
for keyword-based retrieval, enabling a hybrid search that
combines both approaches.
3) Dual Retriever Architecture:We employ a dual-
retriever architecture to provide both code-level and
language-level domain-specific context during generation.
Thecode retrieverqueries the code snippet database using
pure semantic search: given a component description from
Sint(e.g.,E: “Ego vehicle travels forward”), it retrieves
3https://www.langchain.com/
4https://huggingface.co/sentence-transformers/
all-MiniLM-L6-v2Algorithm 1Generation Pipeline
1:Input:user query(Scenario description)
2:Output:S code (Scenic program)
3:— Step 1: Logical Structure Extraction —
4:Sint←Interpreter(user query)▷extract{R, E, O, T}
5:— Step 2: Global Configuration —
6:(detected, confidence)←SettingsDetector(user query)
7:forp∈ {map, weather, models}do
8:ifconfidence[p]≥0.6anddetected[p]̸=nullthen
9:G settings [p]←detected[p]▷use detected value
10:else
11:G settings [p]←default[p]▷fall back to default
12:end if
13:end for
14:G code←HeaderGenerator(G settings )▷Scenic header
15:— Step 3: Iterative Component Generation —
16:R code←Generator R(R, G code)
17:E code←Generator E(E, G code, Rcode)
18:fori= 1toNdo
19:O i,code←Generator O(Oi, Gcode, Rcode, Ecode, O1:i−1,code )
20:end for
21:T code←Generator T(T, G code, Rcode, Ecode, O1:N,code )
22:S code← {G code, Rcode, Ecode, O1:N,code , Tcode}
23:returnS code
the top-3 most similar snippets filtered by component type
via COSINE similarity. Thedocumentation retrieveruses
a hybrid retrieval approach. For the same component de-
scription (e.g.,E: “Ego vehicle travels forward”), BM25
matches keywords and retrieves chunks containing explicit
Scenic API definitions such asFollowLaneBehavior,
while the embedding model retrieves semantically related
content (e.g., general lane-following behavior descriptions).
Reciprocal Rank Fusion (RRF) then merges and re-ranks
both lists, so the returned chunks cover both exact API defini-
tions and broader semantic background. This dual approach
ensures the generator receives both concrete code patterns
and domain-specific language knowledge.
C. Generation Module
The generation module transforms the logical represen-
tationSinto executable Scenic codeS code as shown in
Algorithm 1. The process consists of two main stages: (1)
global configuration generation and (2) iterative component-
wise generation with context accumulation, followed by
simple concatenation to assemble the final program. Fol-
lowing SOTA approaches [15], [26], we employ prompting
techniques rather than fine-tuning: pre-trained LLMs al-
ready demonstrate strong performance and can be effectively
adapted to domain-specific tasks via prompting techniques.
Additionally, this approach preserves generalizability, en-
abling seamless deployment across different LLM models.
1) Global Configuration Generator:The Global Config-
uration Generator operates as a two-stage pipeline to detect
and formalize scenario settings by usingSettingsDetector
andHeaderGeneratorprompts. First, theSettingsDetec-
toranalyzes the scenario description to detect global set-
tingsG settings using rule-based keyword matching patterns
and example-based reasoning. The detector returns settings
with confidence scores (only confidence≥0.6 accepted,
low-confidence or null detections use defaults:Town05,

Fig. 4: Prompting strategies for component generation.
ClearNoon,Lincoln mkz). Second, theHeaderGenera-
tortransforms the detected settings into an executable Scenic
headerG code←HeaderGenerator(map, weather, models),
covering CARLA map parameters, weather, and vehicles.
2) Iterative Component Generator:Components are gen-
erated iteratively in dependency order using specialized gen-
erators: Generator Rproduces Spatial Relation code (R code),
Generator Eproduces Ego code (E code), Generator Oitera-
tively produces Object codes (O 1:N,code ), and Generator T
produces Restriction code (T code). Finally, all components
are concatenated to form the complete executable Scenic
programS code ={G code, Rcode, Ecode, O1:N,code , Tcode}.
Each generator takes inputs including the logical description
from related components inS intand previously accumulated
generated code context, ensuring compatibility between com-
ponents. All generators share the same unified prompting
strategy structure as shown in Fig. 4.
Contextual Prompting (CP):Enhances the model’s do-
main understanding by injecting domain-specific knowledge
directly into the prompt, including syntax constraints, hierar-
chical type systems (e.g., for Generator R: NetworkElement
→LinearElement→ {Road, Lane}), and available operators.
This approach remains flexible, allowing additional domain
knowledge to be incorporated.
Chain of Thought (CoT):Structures the generation task
into explicit reasoning steps tailored per component (e.g., for
Generator E: “Understand requirements→Examine context
→Select type→Define parameters→Define behavior→
Instantiate→Validate”), enabling the model to complete
each step incrementally, improving both understanding and
generation accuracy.
In-Context Learning (ICL):Employs few-shot examples
where the model learns from demonstration examples di-
rectly in the prompt context. Positive examples showcase cor-
rect implementation patterns and proper syntax, while nega-
tive examples present error-correction pairs. This contrastive
learning approach anchors the model to concrete instances,
reducing ambiguity and preventing common pitfalls.
Retrieval-Augmented In-Context Learning (RAG-
ICL):Extends ICL with dynamic retrieval: the dual-retrieverTABLE II: Distribution of scenarios in our benchmark.
Source Number SCE VRU TC Temporal DB
Number Types Vehicle VRU
CARLA Leaderboard 24 24 3 2 3 11 13 2
NHTSA Crash 16 16 3 1 4 16 8 1
NHTSA PreCrash 31 31 4 2 4 27 13 2
UN R152 4 3 1 2 1 2 7 1
UN R157 12 12 6 2 0 12 6 4
UN R171 36 35 9 3 0 35 8 3
Total 123 121 26 3 12 103 20 10
SCE: Safety-critical Edge Cases, VRU: Vulnerable Road Users such as pedestrians
and cyclists, TC: Traffic Control mechanisms such as signals and signs, Temporal:
conditions affecting weather and time-of-day, DB: Dynamic Behaviors capturing
motion patterns.
(Section II-B) dynamically fetches the most relevant ex-
amples per query: Code In-Context Learning (CodeICL)
retrieves top-3 semantically similar code snippets, while
Documentation In-Context Learning (DocICL) retrieves rel-
evant documentation chunks. Unlike fixed-example ICL, the
dynamic retrieval adapts to each query, and the underlying
database can be extended independently, making the frame-
work flexible and generalizable to new domains.
Ablation studies on these prompting techniques are pre-
sented in Section III-B.1.
III. RESULTS& DISCUSSION
A. Experimental Setup
We conduct experiments using the 3D simulator CARLA
0.9.15 with the compatible DSL Scenic 3.1.0. All experi-
ments are performed on a Dell Alienware R15 equipped with
an NVIDIA RTX 4090 GPU with 24GB VRAM.
1) Task Definition and Benchmark:To evaluate our
framework’s performance, we propose a regulation-grounded
benchmark with 123 scenario specifications drawn from
multiple sources: the CARLA Leaderboard5(24 scenarios),
NHTSA6crash and pre-crash datasets (47 scenarios), and
United Nations Vehicle Regulations including UN R152,
R157, and R171 (52 scenarios7). Table II presents the de-
tailed distribution across these sources.
2) Evaluated Large Language Models:Our evaluation
includes leading proprietary and open-source LLMs. For the
proprietary models, we evaluate across Qwen (Flash, Plus),
DeepSeek-V3.2 in both Chat and Reasoner modes, Google
Gemini-3 (Pro, Flash (no thinking mode)), specifically with
Qwen3-Coder (Plus, Flash). Regarding the open-source mod-
els, we adopt Ollama8with GPT-OSS 20B, Mistral-Small-
3.2, Gemma-3-27B, Qwen3:30B and the code-specialized
Qwen3-Coder:30B.
3) Evaluation Metrics:To comprehensively evaluate our
framework, we employ two categories of metrics:
Framework Performance Metrics:We assess the oper-
ational efficiency and reliability of the framework using: (i)
Compilation Success Rate (CSR): percentage of generated
5https://leaderboard.carla.org/
6https://www.nhtsa.gov/
7https://unece.org/transport/vehicle-regulations
8https://ollama.com/

Fig. 5: Example scenarios from three sources: a) CARLA leaderboard, b) UN R171, c) NHTSA Crash.
Scenic programs that successfully execute in the simulator;
(ii)Response Time (RT): average generation time per sce-
nario from description to code; (iii)Token Usage (Tokens):
average number of tokens consumed per scenario.
Scenario Generation Accuracy:For scenarios that suc-
cessfully compile, we evaluate how accurately the generated
scenario reflects the original description. Since our output
is executable code, language quality metrics (e.g., BLEU,
ROUGE-L) are not applicable. Instead, we adopt a human
evaluation structured by a layers model [27], which decom-
poses a driving scenario into semantically distinct layers. Hu-
man evaluator compares each layer of the generated scenario
against the scenario description and assigns a correctness
score. Specifically, the evaluated layers are: (i)Road: ge-
ometry and topology; (ii)Traffic Infrastructure: traffic light
and signs; (iii)Temporal Modifications: construction zones;
(iv)Objects: Static and Dynamic Objects; (v)Environment:
weather and time-of-day conditions; (vi)Scenario Quality
(SQ): the average alignment score across layers. Additionally,
we report theFramework Accuracy (FA), computed as FA =
CSR × SQ, which represents the overall end-to-end accuracy.
B. Experiments and Results
Fig. 5 shows CARLA-rendered qualitative examples from
three benchmark sources: (a) an ego vehicle negotiating
an unsignalized intersection (CARLA Leaderboard), (b) anego vehicle following a lead car that swerves to avoid a
stationary obstacle (UN R171), and (c) a pedestrian crossing
a multi-lane roadway undetected (NHTSA Crash). We further
conduct comprehensive ablation studies and cross-model
comparisons on the proposed benchmark.
1) Ablation Study:As discussed in Section II-C.2, we
combine CP, CoT, ICL, and RAG-ICL to inject domain-
specific knowledge and guide LLM generation. We system-
atically evaluate 12 configurations (C1-C12 in Table III)
organized into four tiers using Gemini 3 Flash: (i)Zero/Few-
shot Tier(C1-C3): baseline and ICL/CodeICL without CP;
(ii)Contextual Tier(C4-C7): CP as foundation with various
combinations; (iii)Thinking Tier(C8-C11): CP+CoT with
various combinations; (iv)Extended(C12): all techniques
including DocICL. Results are summarized in Table III.
The results demonstrate clear performance improvements
as more advanced techniques are incorporated. Starting from
the zero-shot baseline (C1) with CSR and FA of 0%,
adding CP alone (C4) increases CSR to 12.20% and FA to
8.51%. Combining CP with ICL (C5) significantly boosts
performance CSR of 47.15% and FA of 32.62%. Further
adding CoT to create the thinking tier (C9) achieves CSR
of 54.47% and FA of 39.09%. The best performance is
achieved by C11, which incorporates CP, CoT, ICL, and
CodeICL, achieving a CSR of 76.42%, SQ of 76.12%, and

TABLE III: Ablation study results across all prompting technique combinations (based on Gemini-3-Flash model).
ID CP CoT ICLRAG-ICLCSR (%)↑ RT (s)↓Tokens↓Scenario Alignment(%)↑FA(%)↑
CodeICL DocICL RL↑TI↑TM↑DO↑EN↑ SQ↑
Zero/Few-shot Tier
C1∗– – – – – 0.00 170.63 32572 0.00 0.00 0.00 0.00 0.00 0.00 0.00
C2 – –✓– – 4.06 102.58 22889 100.00 100.00 – 66.67 100.00 88.89 3.61
C3 – – –✓– 8.10 157.75 27796 64.58 75.00 100.00 41.67 75.00 54.86 4.44
Contextual Tier
C4✓– – – – 12.20 435.59 43657 82.29 100.00 100.00 52.08 100.00 69.79 8.51
C5✓–✓– – 47.15 196.60 39702 88.10 83.33 76.67 43.12 100.00 69.18 32.62
C6✓– –✓– 58.54 318.92 47085 84.43 90.00 75.00 45.61 97.62 69.17 40.49
C7✓–✓ ✓– 60.10 172.27 48606 88.22 78.57 85.00 54.89 98.00 74.93 45.03
Thinking Tier
C8✓ ✓– – – 13.01 624.12 55153 84.31 100.00 – 41.18 100.00 65.52 8.52
C9✓ ✓ ✓– – 54.47 520.47 52689 87.81 92.31 100.00 49.50 97.22 71.77 39.09
C10✓ ✓–✓– 60.98 300.47 46005 84.46 77.78 83.33 51.58 100.00 70.70 43.11
C11 ✓✓ ✓ ✓ – 76.42 222.11 52339 90.65 75.00 83.33 58.16 96.00 76.12 58.17
Extended Configuration
C12✓ ✓ ✓ ✓ ✓56.90 563.36 70225 86.62 78.57 75.00 54.93 100.00 73.30 41.71
Arrows indicate if higher (↑) or lower (↓) values are better.∗C1 is a zero-shot baseline model. CP: Contextual Prompting, CoT: Chain-of-Thought, ICL: In-Context Learning,
RAG-ICL: Retrieval-Augmented Generation with In-Context Learning, including Code and Doc. CSR: Compilation Success Rate (%), RT: Response Time (s), Tokens: Token
usage. Scenario Alignment Layers: RL: Road, TI: Traffic Infrastructure, TM: Temporal Modifications, DO: Dynamic Objects, EN: Environment. SQ: Scenario Quality, FA:
Framework Accuracy.
TABLE IV: Comparison across LLM backbones and SOTA methods evaluated on the proposed 123-scenario benchmark.
Model Variant CSR (%)↑ RT (s)↓Tokens↓Scenario Alignment(%)↑FA(%)↑
RL↑TI↑TM↑DO↑EN↑ SQ↑
Open-Source Models
Qwen3-Coder:30B 0.00 62.95 21098 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Qwen3:30B 0.00 135.23 30659 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Gemma3:27B 0.00 121.88 35705 0.00 0.00 0.00 0.00 0.00 0.00 0.00
GPT-OSS:20B 0.82 96.69 33199 100.00 100.00 – 100.00 – 100.00 0.82
Mistral-Small3.2:24B 1.62 125.74 32129 100.00 33.33 – 11.11 100.00 58.33 0.94
Proprietary Models
Qwen-Flash 1.63 15.00 22081 83.33 100.00 – 22.22 – 62.04 1.01
Qwen-Plus 5.70 14.05 21937 100.00 100.00 – 66.67 100.00 87.30 4.98
Qwen3-Coder-Plus 10.57 19.67 22072 85.71 80.00 – 34.52 91.67 65.28 6.90
Qwen3-Coder-Flash 15.44 13.85 21980 97.37 60.00 33.33 39.47 100.00 70.61 10.90
DeepSeek-V3.2 (Chat) 12.20 28.04 22676 85.09 70.00 0.00 45.61 100.00 72.37 8.83
DeepSeek-V3.2 (Reasoner) 14.63 460.62 35290 93.75 60.00 100.00 56.25 100.00 78.30 11.46
Gemini-3-Pro 60.16 165.39 47928 92.22 61.54 85.71 53.33 100.00 73.50 44.22
Gemini-3-Flash 76.42 222.11 52339 90.65 75.00 83.33 58.16 96.00 76.12 58.17
SOTA (Gemini-3 Flash)
ChatScene [5] (Retrieval Assemble) 30.08 10.02 4476 65.03 37.88 0 7.52 46.67 36.68 11.03
NL2Scenic [12] (Retrieval Generation) 16.26 43.44 20444 96.15 100.00 66.67 33.33 100.00 66.77 10.86
Arrows indicate if higher (↑) or lower (↓) values are better.
FA of 58.17% with a response time of 222.11 seconds.
Notably, adding DocICL (C12) does not improve perfor-
mance, as the Scenic documentation is better suited for
our interactive chats to understand code concepts, rather
than providing practical guidance for code generation, which
incurs higher time and token costs. A crucial observation
is that providing more code examples through CodeICL
significantly improves generation efficiency: this emerges by
comparing C7 (CP+ICL+CodeICL, RT=172.27s) with C5
(CP+ICL, RT=196.60s), and C11 (CP+CoT+ICL+CodeICL,
RT=222.11s) with C9 (CP+CoT+ICL, RT=520.47s), as code
snippets help the model converge faster to correct solutions.
2) Comparison across Models and SOTA Methods:As
shown in Table IV, we evaluate our Chat2Scenic framework
(C11 configuration) across leading proprietary, open-source
LLM and SOTA methods. The Gemini-3 family models
outperform other proprietary models, likely due to their
powerful capabilities and support for long context windows.
Notably, Gemini-3-Flash achieves the best overall perfor-mance with CSR of 76.42% and FA of 58.17%, significantly
outperforming Gemini-3-Pro (CSR 60.16%, FA 44.22%).
This aligns with findings from [28] that internal reasoning
of thinking models can be hindered by external prompting
strategies, whereas instruction-optimized models like Flash
are better suited for structured prompting techniques. In
contrast, open-source models perform poorly: most achieve
near-zero CSR (0%–1.62%), primarily due to their smaller
scale, which limits general capabilities, less extensive pre-
training data compared to proprietary models, and difficulty
following complex prompting strategies with multiple inter-
leaved techniques.
Furthermore, based on the same Gemini-3-Flash model,
we compare our framework against SOTA methods,
specifically ChatScene [5] (Retrieval Assemble) and
NL2Scenic [12] (Retrieval Generation), which provide pub-
licly available implementations. ChatScene achieves better
performance (CSR 30.08%) than NL2Scenic, as it leverages
a pre-built code database from CARLA Leaderboard sce-

narios for retrieval-based assembly. In contrast, NL2Scenic’s
direct retrieval of complete Scenic scripts struggles with
complex scenarios despite advanced prompting, as full-
script generation lacks the flexibility and stability needed for
diverse scenario specifications. Our framework significantly
outperforms both methods, achieving a CSR of 76.42%,
demonstrating that iterative component-wise generation with
diverse prompting strategies improves the effectiveness and
robustness of scenario generation. While our approach re-
quires longer response time (RT of 222.11 seconds) due to
iterative generation, this is an acceptable compromise for
offline scenario generation, where accuracy and reliability
are prioritized over generation speed.
IV. CONCLUSION ANDFUTUREWORK
We presented Chat2Scenic, the first iterative retrieval-
augmented framework for ADS scenario generation. To
enable systematic evaluation, we introduced a regulation-
grounded benchmark comprising 123 scenarios derived
from multiple sources, along with comprehensive evaluation
metrics. Through ablation studies and cross-model com-
parisons, we demonstrated that advanced prompting tech-
niques (CP+CoT+ICL+CodeICL) achieve CSR of 76.42%,
SQ of 76.12%, and FA of 58.17% (with Gemini-3-Flash),
significantly outperforming existing retrieval-assemble and
full-script generation approaches. Future work will extend
the framework with multimodal inputs (e.g., images and
sketches) and simulation-in-the-loop feedback to enhance
scenario specification and generation accuracy.
ACKNOWLEDGMENT
The manuscript was initially drafted by the authors, with
AI tools used to improve grammar and clarity.
REFERENCES
[1] J. Betz, M. Lutwitzi, and S. Peters, “A new taxonomy for automated
driving: Structuring applications based on their operational design
domain, level of automation and automation readiness,” in2024 IEEE
Intelligent Vehicles Symposium (IV), 2024.
[2] M. B ¨aumler and E. Cuvillier,Generating Test Scenarios for Assessing
Automated Driving Systems Using Scenario Fusion, ser. Schriftenreihe
des Lehrstuhls Kraftfahrzeugtechnik. Cuvillier Verlag, 2024.
[3] Q. Song, K. Tan, P. Runeson, and S. Persson, “Critical scenario
identification for realistic testing of autonomous driving systems,”
Software Quality Journal, 2023.
[4] Y . Deng, J. Yao, Z. Tu, X. Zheng, M. Zhanget al., “Target: Automated
scenario generation from traffic rules for testing autonomous vehicles,”
arXiv preprint arXiv:2305.06018, 2023.
[5] J. Zhang, C. Xu, and B. Li, “Chatscene: Knowledge-enabled safety-
critical scenario generation for autonomous vehicles,” inIEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2024.
[6] X. Cai, X. Bai, Z. Cui, D. Xie, D. Fuet al., “Text2scenario: Text-
driven scenario generation for autonomous driving test,”Automotive
Innovation, 2026.
[7] H. Tian, X. Han, Y . Zhou, G. Wu, A. Guoet al., “Lmm-
enhanced safety-critical scenario generation for autonomous driv-
ing system testing from non-accident traffic videos,”arXiv preprint
arXiv:2406.10857, 2024.
[8] S. Tang, Z. Zhang, J. Zhou, L. Lei, Y . Zhouet al., “Legend: A top-
down approach to scenario generation of autonomous driving systems
assisted by large language models,” in39th IEEE/ACM International
Conference on Automated Software Engineering, 2024.[9] K. Elmaaroufi, D. Shanker, A. Cismaru, M. Vazquez-Chanlatte,
A. Sangiovanni-Vincentelliet al., “Scenicnl: generating proba-
bilistic scenario programs from natural language,”arXiv preprint
arXiv:2405.03709, 2024.
[10] B.-K. Ruan, H.-T. Tsui, Y .-H. Li, and H.-H. Shuai, “Traffic scene
generation from natural language description for autonomous vehicles
with large language model,”arXiv preprint arXiv:2409.09575, 2024.
[11] R. Rubavicius, A. V . Miceli-Barone, A. Lascarides, and S. Ramamoor-
thy, “Conversational code generation: a case study of designing a dia-
logue system for generating driving scenarios for testing autonomous
vehicles,”arXiv preprint arXiv:2410.09829, 2024.
[12] P. Bauerfeind, A. Salarpour, D. Fernandez, P. MohajerAnsari,
J. Reschkeet al., “David vs. goliath: A comparative study of different-
sized llms for code generation in the domain of automotive scenario
generation,”arXiv preprint arXiv:2510.14115, 2025.
[13] W. Ding, C. Xu, M. Arief, H. Lin, B. Liet al., “A survey on safety-
critical driving scenario generation—a methodological perspective,”
IEEE Transactions on Intelligent Transportation Systems, 2023.
[14] B. Sch ¨utt, J. Ransiek, T. Braun, and E. Sax, “1001 ways of scenario
generation for testing of self-driving cars: A survey,” in2023 IEEE
Intelligent Vehicles Symposium (IV). IEEE, 2023.
[15] Y . Gao, M. Piccinini, Y . Zhang, D. Wang, K. Molleret al., “Foundation
models in autonomous driving: A survey on scenario generation and
scenario analysis,”IEEE Open Journal of Intelligent Transportation
Systems, 2026.
[16] A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, and V . Koltun, “Carla:
An open urban driving simulator,” inConference on robot learning.
PMLR, 2017.
[17] P. A. Lopez, M. Behrisch, L. Bieker-Walz, J. Erdmann, Y .-P. Fl ¨otter ¨od
et al., “Microscopic traffic simulation using sumo,” inThe 21st
IEEE International Conference on Intelligent Transportation Systems.
IEEE, 2018.
[18] S. Li, T. Azfar, and R. Ke, “Chatsumo: Large language model for
automating traffic scenario generation in simulation of urban mobility,”
IEEE Transactions on Intelligent Vehicles, 2024.
[19] Q. Lu, X. Wang, Y . Jiang, G. Zhao, M. Maet al., “Multimodal
large language model driven scenario testing for autonomous vehicles,”
arXiv preprint arXiv:2409.06450, 2024.
[20] A. Aiersilan, “Generating traffic scenarios via in-context learning
to learn better motion planner,” inAAAI Conference on Artificial
Intelligence, 2025.
[21] A. Guo, Y . Zhou, H. Tian, C. Fang, Y . Sunet al., “Sovar: Build
generalizable scenarios from accident reports for autonomous driving
testing,” inProceedings of the 39th IEEE/ACM International Confer-
ence on Automated Software Engineering, 2024.
[22] Q. Shi, Q. Meng, H. Cheng, Q. Xu, and J. Wang, “Linguasim:
Interactive multi-vehicle testing scenario generation via natural lan-
guage instruction based on large language models,”arXiv preprint
arXiv:2510.08046, 2025.
[23] Q. Lu, M. Ma, X. Dai, X. Wang, and S. Feng, “Realistic corner case
generation for autonomous vehicles with multimodal large language
model,”arXiv preprint arXiv:2412.00243, 2024.
[24] X. Zhou, Y . Huang, J. Zhang, J. Shao, D. Panet al., “Automatic
generation method for autonomous driving simulation scenarios based
on large language model,” inInternational Conference on Artificial
Intelligence and Autonomous Transportation. Springer, 2024.
[25] D. J. Fremont, T. Dreossi, S. Ghosh, X. Yue, A. L. Sangiovanni-
Vincentelliet al., “Scenic: a language for scenario specification and
scene generation,” in40th ACM SIGPLAN conference on programming
language design and implementation, 2019.
[26] Y . Gao, M. Piccinini, K. Moller, A. Alanwar, and J. Betz, “From
words to collisions: Llm-guided evaluation and adversarial generation
of safety-critical driving scenarios,” in2025 IEEE 28th International
Conference on Intelligent Transportation Systems (ITSC). IEEE,
2025.
[27] M. Scholtes, L. Westhofen, L. R. Turner, K. Lotto, M. Schuldeset al.,
“6-layer model for a structured description and categorization of urban
traffic and environment,”IEEE Access, 2021.
[28] Y . Gao, M. Piccinini, R. Brusnicki, Y . Zhang, and J. Betz, “Nurisk:
A visual question answering dataset for agent-level risk assessment
in autonomous driving,” in2026 IEEE International Conference on
Robotics and Automation (ICRA), 2026.