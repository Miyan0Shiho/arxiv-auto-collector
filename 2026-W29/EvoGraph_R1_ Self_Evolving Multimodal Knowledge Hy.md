# EvoGraph-R1: Self-Evolving Multimodal Knowledge Hypergraphs for Agentic Retrieval

**Authors**: Jiashi Lin, Changhong Jiang, Xiangru Lin, Ruifei Zhang, Xinyi Zhu, Jiyao Liu, Cheng Tang, Ye Du, Shujian Gao, Junzhi Ning, Lihao Liu, Ziyan Huang, Tianbin Li, Jin Ye, Junjun He

**Published**: 2026-07-14 13:34:07

**PDF URL**: [https://arxiv.org/pdf/2607.12764v1](https://arxiv.org/pdf/2607.12764v1)

## Abstract
Retrieval-augmented generation (RAG) has emerged as a critical paradigm for grounding Multimodal Large Language Models (MLLMs) in external knowledge. Recent GraphRAG methods introduce structured entity-relation graphs to improve retrieval and reasoning. However, they remain limited by treating knowledge graphs as static data structures built offline and queried in a single pass. This static paradigm misaligns with the interactive, iterative nature of knowledge-intensive reasoning, creating three bottlenecks: (i) text-centric fragmentation that impedes cross-modal reasoning, (ii) frozen structures unable to incorporate new evidence or correct errors, and (iii) rigid single-pass retrieval without adaptive refinement. To overcome these limitations, we introduce EvoGraph-R1, a self-evolving GraphRAG framework that reconceptualizes knowledge graphs as dynamic environments shaped through agent interactions. We formulate retrieval as a Markov Decision Process (MDP) where the agent observes the graph state and executes actions to query (GraphRetrieve), expand (WebSearch), refine (GraphEdit), or terminate (Answer) the reasoning. These actions reshape the hypergraph structure and generate feedback signals that guide subsequent evolution. Through this closed loop, the hypergraph evolves by integrating new evidence, correcting errors, and refining structure to support multi-hop reasoning. Experiments on multimodal VQA and text QA benchmarks demonstrate substantial improvements over existing RAG baselines in accuracy, coverage, and traceability, establishing self-evolving knowledge graphs as a fundamental paradigm across modalities.

## Full Text


<!-- PDF content starts -->

EvoGraph-R1: Self-Evolving Multimodal Knowledge Hypergraphs for Agentic
Retrieval
Jiashi Lin1,2*Changhong Jiang1†Xiangru Lin3*Ruifei Zhang5Xinyi Zhu1
Jiyao Liu2Cheng Tang2Ye Du2Shujian Gao2Junzhi Ning2Lihao Liu2
Ziyan Huang2Tianbin Li2Jin Ye2,4Junjun He2
1Northwestern Polytechnical University,2Shanghai Artificial Intelligence Laboratory
3The University of Hong Kong,4Monash University,5The Chinese University of Hong Kong (Shenzhen)
https://evograph-r1.github.io/
Abstract
Retrieval-augmented generation (RAG) has emerged as a
critical paradigm for grounding Multimodal Large Lan-
guage Models (MLLMs) in external knowledge. Recent
GraphRAG methods introduce structured entity-relation
graphs to improve retrieval and reasoning. However,
they remain limited by treating knowledge graphs as
static data structures built offline and queried in a sin-
gle pass. This static paradigm misaligns with the in-
teractive, iterative nature of knowledge-intensive reason-
ing, creating three bottlenecks: (i) text-centric fragmenta-
tion that impedes cross-modal reasoning, (ii) frozen struc-
tures unable to incorporate new evidence or correct er-
rors, and (iii) rigid single-pass retrieval without adap-
tive refinement. To overcome these limitations, we intro-
duceEvoGraph-R1, a self-evolving GraphRAG framework
that reconceptualizes knowledge graphs asdynamic envi-
ronmentsshaped through agent interactions. We formu-
late retrieval as a Markov Decision Process (MDP) where
the agent observes the graph state and executes actions
to query (GRAPHRETRIEVE), expand (WEBSEARCH), re-
fine (GRAPHEDIT), or terminate (ANSWER) the reason-
ing. These actions reshape the hypergraph structure and
generate feedback signals that guide subsequent evolu-
tion. Through this closed loop, the hypergraph evolves
by integrating new evidence, correcting errors, and re-
fining structure to support multi-hop reasoning. Experi-
ments on multimodal VQA and text QA benchmarks demon-
strate substantial improvements over existing RAG base-
lines in accuracy, coverage, and traceability, establishing
self-evolving knowledge graphs as a fundamental paradigm
*Equal contribution
†Corresponding authoracross modalities.
1. Introduction
Who was the sultan of the ottoman empire 
when construction of this memorial began?
(a) RAG-based 
Lost in the vast document space.
(c) EvoGraph-R1
Answer: Abdul Hamid I.
Self-evolving Over Time
S3:GraphEdit (Insert, Update, Delete)
Insert
Hyper
edge_1
Hyper
edge_3Hyper
edge_2
1905
 Ottoman EmpireAbdul Hamid II
1905
William McKinley
National Memorial 
Update
Delete
Abdul Hamid I
(b) Graph-based
William McKinley 
National Memorial1905Ottoman 
EmpireAbdul Hamid IIruled by
ruled in year
start time
Reasoning breaks when 
key facts are absent.
？S1:GraphRetrieval
 S2:WebSearch
<websearch>Construction 
of the William McKinley 
National Memorial began 
in 1905.</websearch>
 Question
Answer:  Abdul Hamid II.
The agent 
identifies missing 
facts, retrieves 
external evidence, 
and iteratively 
refines the 
HyperGraph.
 1789
Figure 1.Comparison of Retrieval Paradigms.(a) Vanilla RAG
loses information in document spaces. (b) Static graph-based
RAG breaks when key facts are missing. (c) EvoGraph-R1 self-
evolves through agent-driven GRAPHRETRIEVE, WEBSEARCH,
and GRAPHEDITto complete reasoning chains.
Multimodal large language models (MLLMs) excel at
visual reasoning tasks [1, 19], yet their parametric mem-
arXiv:2607.12764v1  [cs.CV]  14 Jul 2026

ory remains limited [23]. No model can internalize the
vast, ever-evolving knowledge of the open world. Retrieval-
augmented generation (RAG) addresses this by enabling
dynamic access to external knowledge at inference time,
improving factual grounding in knowledge-intensive visual
question answering (VQA) [11]. However, existing mul-
timodal RAG frameworks struggle with complex compo-
sitional reasoning [16], as retrieved evidence is often re-
duced to disjoint text snippets or coarse page-level screen-
shots [6, 22, 33], hindering fine-grained structural modeling
and cross-modal alignment.
Recent GraphRAG methods [5, 8, 9, 21] mitigate these
limitations by leveraging knowledge graphs for retrieval and
reasoning, thereby facilitating structured reasoning. The
typical pipeline comprises three stages: (i) constructing
graphs via LLM-based entity-relation extraction [41]; (ii)
retrieving relevant subgraphs; and (iii) generating answers
conditioned on graph knowledge. However, we identify
a key limitation:existing systems treat the knowledge
graph as a static structure, built offline in a single pass.
This one-shot process makes graph quality difficult to guar-
antee, especially on long or multimodal inputs. As illus-
trated in Figure 1, such static pipelines often yield incom-
plete or noisy graphs that propagate errors throughout re-
trieval and reasoning. Consequently, these limitations man-
ifest as three bottlenecks:(1) Text-centric fragmentation.
LLM-based extraction collapses rich multimodal evidence
into isolated textual tuples, leading to missing relations,
fragmented structures, and loss of long-range or cross-
modal dependencies [35].(2) Static structure.Graphs
are constructed offline and remain fixed during inference,
preventing the incorporation of new evidence, error correc-
tion, or coverage of unseen topics [26].(3) Rigid retrieval.
Retrieval is executed as a single-pass operation without it-
erative refinement. When initial evidence is insufficient, the
system cannot revise its strategy, explore alternatives, or in-
voke external search [34].
These limitations stem from treating the knowledge
graph as apassive data structurerather than anactive rea-
soning substrate. Human reasoning over knowledge is in-
herently interactive and iterative [2]: we collect initial evi-
dence, identify gaps, seek additional information, reconcile
contradictions, and progressively refine our understanding.
This mirrors the agent-environment interaction paradigm in
reinforcement learning [20], where agents perceive states,
take actions that modify those states, and receive feedback
to guide future decisions. Inspired by this paradigm, we
reconceptualize multimodal GraphRAG as aMarkov De-
cision Process (MDP)[17], where the knowledge graph
becomes a dynamic environment that co-evolves with the
reasoning process:
•Environment State: the multimodal knowledge hyper-
graph, encoding retrieved evidence (entities, relations, vi-sual objects) and their structural organization (connectiv-
ity, alignment, relevance scores).
•Actions: operations thatquerythe graph
(GRAPHRETRIEVE),expandit (WEBSEARCH) for
external knowledge),refineit (GRAPHEDITto insert/up-
date/delete elements), orterminate(ANSWER).
•Rewards: trajectory-level signals that assess reasoning
quality (structural correctness and answer accuracy) and
efficiency (computational cost).
•Policy: the agent’s action selection strategy, learned
through reinforcement learning to maximize answer ac-
curacy while maintaining efficiency.
This formulation enables aself-evolving paradigm[3]: the
agent does not merely retrieve from a fixed graph but ac-
tively shapes it by adding verified evidence, correcting in-
consistencies, pruning noise, and reorganizing structures to
support multi-hop inference.
To instantiate this paradigm, we presentEvoGraph-R1,
an agentic multimodal GraphRAG framework that unifies
retrieval and knowledge evolution into a closed-loop in-
teraction process. Initialized from a minimal hypergraph,
EvoGraph-R1 progressively refines its structure through
agent–environment interactions. At each step, the agent ob-
serves the current graph state to identify knowledge gaps,
then selects an action that reshapes the hypergraph and
generates feedback signals to guide subsequent decisions.
Through this iterative cycle, EvoGraph-R1 transforms an
initially sparse, noisy hypergraph into a coherent, query-
specific knowledge state. Our self-evolving MDP paradigm
generalizes beyond multimodal settings. By removing vi-
sual components while retaining the agent-driven evolution
mechanism, the framework naturally adapts to text-only
scenarios. We demonstrate state-of-the-art performance on
both multimodal [12, 30] and text QA benchmarks [40],
validating that treating knowledge graphs as interactive en-
vironments provides value independent of modality. Our
main contributions are as follows:
•Self-evolving retrieval paradigm.We introduce the first
framework that models multimodal knowledge graphs as
MDP environments, unifying retrieval, reasoning, and
knowledge evolution through agent-environment interac-
tion.
•EvoGraph-R1 framework.We design an autonomous
agent operating over multimodal knowledge hypergraphs
via four action types (GRAPHRETRIEVE, WEBSEARCH,
GRAPHEDIT, ANSWER), implementing closed-loop co-
evolution of graph structure and reasoning.
•State-of-the-art performance.We conduct multiple
experiments on knowledge-intensive multimodal VQA
and text QA benchmarks. Experimental results show
our framework substantially outperforms existing RAG,
GraphRAG, and search-augmented baselines in accuracy,
efficiency, and evidential traceability.

2. Related Work
RAG and GraphRAG.Retrieval-augmented generation
enhances LLM factuality by retrieving external evi-
dence during inference [18]. While standard RAG re-
trieves flat text chunks without structured reasoning [27],
GraphRAG [5] represents knowledge as entity-relation
graphs, retrieving subgraphs and reasoning paths [5,
13]. Graph-R1 [20] frames retrieval as interactive
agent-environment interaction over hypergraphs, optimiz-
ing multi-turn exploration with reinforcement learning [7,
20]. Despite these advances, most GraphRAG systems as-
sume offline, fixed graphs with one-shot retrieval. Graphs
cannot be updated with new evidence or repaired at in-
ference time, limiting robustness in open-world settings.
EvoGraph-R1 departs from this static paradigm by enabling
online graph expansion, updating, and pruning through
agent-driven evolution.
Multimodal RAG.Multimodal RAG extends text-
only retrieval across modalities through modality-specific
pipelines: document-oriented systems preserve layouts as
images [6, 36, 39] but lack relational grounding; scene-
graph approaches link visual regions to text but flatten struc-
tured elements [35]. VLM-based methods [22, 32, 33] re-
trieve document images to avoid OCR errors but introduce
redundancy. RAG-Anything [9] constructs graphs encod-
ing cross-modal relationships, yet still assumes static, of-
fline structures. EvoGraph-R1 incrementally builds, up-
dates, and prunes multimodal hypergraphs during inference,
shifting retrieval from a static one-shot answering paradigm
to maintaining a persistent, evolving knowledge state.
Web-search Equipped MLLMs.Recent work equips
MLLMs with active web access for autonomous search and
result integration. Search-R1 [14] uses RL to interleave rea-
soning with web search; MMSearch-R1 [38] extends this
to multimodal settings with end-to-end RL. While demon-
strating the value of interactive retrieval, these systems treat
retrieved content as transient prompt context. EvoGraph-R1
instead regards each retrieval step, including web evidence,
as a state update to a persistent hypergraph that is continu-
ously expanded, revised, and filtered, thereby enabling sys-
tematic multi-hop reasoning with explicit traceability.
3. Method
3.1. Overview
EvoGraph-R1 comprises three components: (1)Multi-
modal Hypergraph Constructioncreates a unified cross-
modal knowledge representation; (2)Agent-Based Graph
Evolutionimplements iterative graph refinement through
four actions; (3)Reinforcement Learningoptimizes the
agent policy using trajectory-level rewards. An overview is
presented in Figure 2.
<Answer><Retrieved
Knowledge><Think>
Reward 
ModelingUser 
Querstion
Action Pool
GraphRetrivel
WebSearch
GraphEditAgent Policy
s+ = (Question,
Dynamic HyperGraph) MDP
StateEnd-to-End RL
 Refine and  
Self-evolving
ExtractorDomain 
Knowledge
Hyper
edge_1
Hyper
edge_3Hyper
edge_2 1905
Ottoman 
EmpireAbdul Hamid IIInsert
1905
William McKinley
National Memorial Update
Abdul Hamid IDelete
Insert
Update
DeleteGraphEdit
Figure 2.EvoGraph-R1 framework overview.The agent op-
erates over a dynamic hypergraph formulated as an MDP. At
each step, it selects actions GRAPHRETRIEVE, WEBSEARCH,
GRAPHEDIT, or ANSWER. GRAPHEDITincludes INSERT, UP-
DATE, and DELETE, enabling continuous refinement. The agent
policy is optimized via end-to-end RL with trajectory-level re-
wards, forming a closed-loop system in which the hypergraph co-
evolves with the reasoning process.
3.2. Problem Setup
In knowledge-driven VQA, the model receives an image-
question pair(I q, q)and generates an answeraby retriev-
ing relevant knowledge fromB={(T i, Ii)}N
i=1, where each
document contains textT iand imageI i. This requires vi-
sual and textual reasoning, demanding that the model align
and synthesize cross-modal content effectively. Unlike tra-
ditional RAG systems that retrieve static documents, our
framework maintains a persistent knowledge hypergraph
that is incrementally expanded, updated, and pruned in re-
sponse to queries, enabling systematic reasoning over inter-
connected multimodal evidence.
3.3. Multimodal Hypergraph Construction
We construct a unified cross-modal knowledge hypergraph
GH= (V,E)from a multimodal corpus through three
stages: textual extraction, visual construction, and cross-
modal fusion. This hypergraph serves as the initial environ-
ment state for agent-based retrieval and reasoning.
Textual Subgraph Extraction. Given the input text
d, we segment it into knowledge fragments represented
as hyperedgesE(d)
H={e 1, . . . , e k}, where eache i=
(etext
i,Vei, ri, σi)contains a natural-language description,
entity set, relation type, and confidence scoreσ i∈(0,10].
Following prior work on hypergraph-structured knowl-
edge [8, 21], we employ ann-ary relation extraction prompt
with an MLLM-based extractorπ ext:
F(n)
d={f 1, . . . , f k} ∼π ext(d),(1)
where each extractedf ℓ= (e ℓ,Veℓ)associates a hyperedge
eℓwith its participating entity setV eℓ. This formulation al-

lows each text segment to contribute multiple, potentially
overlappingn-ary relational facts to the unified textual hy-
pergraphGtext
H= (Vtext,Etext), facilitating high-order rela-
tional reasoning beyond binary relations.
Visual Subgraph Construction. While text naturally
lends itself to entity-relation extraction, visual content re-
quires special handling to preserve scene-level semantics.
For each imagex∈I i, we employ extractorπ extto gen-
erate a detailed natural language scene description together
with the primary object names detected within the image.
This scene description is then encoded as a high-order hy-
peredge, serving as an “anchor” for all entities and relations
grounded in the image:
ux= (id=x,type=image,mod=vis),(2)
whereu xis the anchor node representing the image. En-
tities and finer-grained visual relational facts are subse-
quently extracted from the scene description and associated
withu x, yielding
(Vvis
x,Evis
x) =π ext(x).(3)
Each visual hyperedgee= (V e, re, σe,mod=vis)in-
cludes the anchor node (u x∈ V e) ensuring all extracted
information remains grounded in its image context, prevent-
ing visual facts from floating without source attribution. All
image-specific subgraphsGvis
x= ({u x} ∪ Vvis
x,Evis
x)are in-
tegrated across the corpus, forming the complete visual sub-
graphGvis
H= (Vvis,Evis).
Cross-Modal Graph Fusion. To unify textual and vi-
sual subgraphs, we normalize entity labels and apply the
entity-resolution functionϕ, matching visual and textual
mentions using string similarity and embedding proximity.
Matched entities merge into canonical nodes, inheriting at-
tributes and contextual edges from both modalities. All
hyperedges from original subgraphs reconnect to merged
nodes, yielding the fused graphG H= (V,E):
V=Resolve(Vtext∪ Vvis, ϕ),E=Etext∪ Evis.(4)
Offline Graph Indexing. For efficient retrieval during
agent interaction, we embed all graph elements (textual en-
tities, visual entities, hyperedges) using multimodal encoder
f(·)(e.g., GME [42]) into a shared semantic space:
ϕentity=f(vtext∪vvis), ϕ edge=f(e),∀e∈ E.(5)
This unified hypergraphG H= (V,E)retains fine-grained
cross-modal semantics and forms the initial environment
state for agent-driven evolution.
3.4. Agent-Based Graph Evolution
While the initial hypergraphG Hprovides structured knowl-
edge, automatically constructed graphs are often incom-
plete and noisy. Moreover, static graphs fundamentally can-
not adapt to query-specific information needs or incorporatenewly discovered evidence. To address these limitations,
we formulate retrieval as an agent-environment interaction
process where the knowledge graph becomes a dynamic en-
vironment that co-evolves with reasoning.
3.4.1. MDP Formulation
We model the agent-graph interaction as a discrete-time
Markov decision process over stepst= 0,1, . . . , T. At ini-
tialization, the agent receives the multimodal hypergraphG 0
and queryq. The process terminates when the agent selects
ANSWERaction or reaches step limitT, then it generates
a final answer conditioned on the evolved hypergraphG T.
We formalize the key components of this MDP as:
Environment State.At stept, the environment state
captures the evolving knowledge context:
st= (G t,Ht, q),(6)
whereG t= (V t,Et)is the current hypergraph encod-
ing retrieved evidence (entities, relations, visual regions)
and their structural organization (connectivity, cross-modal
alignments, relevance scores),H tis the action history (a se-
quence of tuples(a k, rk,Gk+1)fork < t), andqis the input
query. This representation ensures the agent conditions its
behavior on both current graph structure and reasoning tra-
jectory, enabling coherent multi-turn reasoning and credit
assignment.
Actions.The agent operates over a discrete action space
Awith four action types:
(1)GRAPHRETRIEVEqueries the current environ-
ment via elements in hypergraphG t. It supports cross-
modal entity-level lookups over surrounding subgraphs and
hyperedge-level vector retrieval, enabling the agent to ex-
ploit structured, semantically linked higher-order facts.
(2)WEBSEARCHis invoked when in-graph evidence
proves insufficient. The agent formulates a queryq weband
retrieves external information from web search engines.
The returned content is aligned with the current state and
appended to the textual context for subsequent generation,
allowing the agent to incorporate external evidence beyond
the static knowledge base.
(3)GRAPHEDITrefines the environment through direct
graph-structure modification via three complementary sub-
operations: INSERT, which introduces new entities or hy-
peredges derived from retrieved content and expands the
graph with verified evidence; UPDATE, which revises ex-
isting hyperedges to correct errors, resolve conflicts, or
strengthen factual grounding based on newly discovered in-
formation; and DELETE, which performs soft removal by
lowering the confidence scores of low-quality or contra-
dicted elements, thereby pruning noise. Together, these op-
erations enable self-correction and adaptive refinement of
the knowledge graph as reasoning progresses, effectively
addressing the inherent incompleteness and noise of auto-
matically constructed graphs.

(4)ANSWERterminates the reasoning process. Once the
agent determines that the environment has evolved suffi-
ciently and accumulated adequate evidence, it invokes this
action to produce the final response based on the current
graph stateG t.
Rewards.After each action, the agent receives re-
ward signals based on three factors: (1) whether reason-
ing steps follow the intended interaction protocol (structural
correctness), (2) semantic alignment between predicted and
ground-truth answers (answer quality), and (3) the compu-
tational cost of executed actions (efficiency). These rewards
guide policy learning to maximize answer accuracy while
maintaining efficient graph evolution (Section 3.5).
Policy.The agent’s policyπ θ(at|st)determines action
selection based on current state, learned through reinforce-
ment learning (Section 3.5) to maximize answer accuracy
while maintaining efficiency.
3.4.2. MDP State Transition Dynamics
The hypergraph is dynamic and responsive to agent actions,
enabling self-evolution. At each stept, actiona tinduces a
transition via evolution policyπ evolve:
Gt+1=π evolve(at,Gt),(7)
where different action types induce different state changes:
• GRAPHRETRIEVEmarks which entities and relations
have been accessed without altering structure.
• WEBSEARCHacts as an external interface, retrieving ev-
idence to validate and augment the graph with new facts
and cross-modal alignments.
• GRAPHEDIToperations directly refine graph quality by
consolidating fragmented knowledge, correcting incon-
sistencies, and removing noise.
Each transition is appended to the action history:
Ht+1=H t∪(a t, rt,Gt+1),(8)
forming a cumulative record that guides online decision-
making and provides supervision signals for reinforcement
learning. The final output is computed as:
a∗←πfinal
resp(q,H T).(9)
Through this recurrent update mechanism, the agent
gradually transforms an initially sparse, noisy hypergraph
G0into a coherent, query-specific knowledge stateG Topti-
mized for reasoning. This evolutionary process yields three
benefits that address the limitations of static GraphRAG:
(1) fine-grained multimodal entities and cross-modal align-
ments are dynamically constructed rather than pre-defined,
overcoming text-centric fragmentation; (2) the graph con-
tinuously evolves to integrate new evidence and correct er-
rors rather than remaining frozen, eliminating static struc-
ture constraints; (3) retrieval adapts through multi-turn
exploration guided by intermediate reasoning rather than
single-pass lookup, removing rigid pipeline limitations.3.5. Reinforcement Learning Optimization
We train the agent’s policy to efficiently evolve the knowl-
edge hypergraph using Group Relative Policy Optimization
(GRPO) [31] with a trajectory-level reward function that
balances answer correctness, reasoning coherence, and re-
trieval efficiency.
3.5.1. Training Setup
Given a dataset of questionsD Q, we train the agent through
experience collected from its interactions with the hyper-
graph environment. For each questionq∈ D Q, an ini-
tial states 1∼P(q)is sampled (consisting of the query
q, a relevant initial subgraph, and an empty history). At
stept, the agent observes(s t−1,GH)and selects an action
at∼πθ(· |s t−1;GH)according to the current policy pa-
rameterized byθ. Ifa tis an GRAPHEDITaction, the agent
also proposes specific modificationsm′
tto commit toG H.
A complete trajectory is defined as:
τ= (s 1, a1, s2, a2, . . . , s |τ|)∈ T q,(10)
where only GRAPHEDITactions modify the graph state
GH, while other actions (e.g., GRAPHRETRIEVE, WEB-
SEARCH) gather information without altering structure.
3.5.2. Reward Design and Policy Optimization
To optimize both reasoning quality and graph evolution ef-
ficiency, we define a trajectory-level reward functionR(τ)
composed of three complementary components:
(i) Structural Rewardevaluates whether each reason-
ing step follows the intended interaction protocol. At each
step(s t, at), the agent produces an intermediate block
(athink
t, αt, aout
t). Well-formed steps are rewarded, up to a
capped total:
Rstruct(τ) = min
1.0, η|τ|X
t=1Ivalid(t)
,(11)
whereI[·]denotes the indicator function, andηis a step-
wise scaling factor set to 0.5. This component encour-
ages disciplined reasoning and syntactic consistency across
multi-turn trajectories.
(ii) Answer Rewardmeasures semantic fidelity using a
token-level overlap between the predicted answer and the
ground-truth labely∗
q:
Rans(aans
T) =2|tok(aans
T)∩tok(y∗
q)|
|tok(aans
T)|+|tok(y∗q)|,(12)
where tok(·)applies normalization such as lowercasing and
whitespace removal. This F1-style score rewards semantic
alignment without requiring exact string matching.
(iii) Overall Outcome Rewardintegrates correctness,
structure, and efficiency:

R(τ) =−λ|τ|X
t=1c(at) +R struct(τ)
+I[R struct(τ) = 1.0]·R ans(aans
T),(13)
wherec(a t)denotes the normalized cost of executing ac-
tiona t(e.g., retrieval or graph edit), andλ >0is the
efficiency coefficient. This formulation ensures correct-
ness is rewarded only when the reasoning process remains
structurally coherent, thereby promoting high-quality, cost-
efficient graph evolution.
(iv) Policy Optimization. We optimize the policy using
GRPO, which computes group-level advantages by compar-
ing each trajectory’s reward to the group mean. This enables
stable updates and efficient learning from multiple rollouts
per question. Through this optimization, the agent learns to
navigate the action space, selectively retrieving and editing
graph elements to construct query-specific knowledge states
that enable accurate question answering.
4. Experiments
We conduct a comprehensive evaluation of EvoGraph-R1 to
address the following questions:RQ1:Does EvoGraph-R1
outperform existing baselines in reasoning accuracy?RQ2:
To what extent do the individual components contribute to
its effectiveness?RQ3:What is the cost of constructing the
unified heterogeneous hypergraph?RQ4:How well does
it generate accurate and coherent answers? Finally,RQ5:
how does it perform in low-resource settings?
4.1. Experiment Setup
Datasets.We evaluate on bothmultimodalandtext-
onlysettings to demonstrate the generalizability of our
self-evolving paradigm. For multimodal tasks, we
use E-VQA [12], InfoSeek [4], and OK-VQA [24],
with Wikipedia as the knowledge source filtered via
EchoSight [11]. For text-only tasks, we use 2WikiMul-
tiHopQA [10], HotpotQA [40], and Natural Questions
(NQ) [18, 40] with official Wikipedia dumps.
Evaluation Metrics.We use the LLM-as-Judge frame-
work for multimodal accuracy, F1 [15] for text-only perfor-
mance, and G-E [28] for answer quality.
Baselines.We compare against three categories:(i)
Standard RAG methods: NaiveRAG [18], EchoSight [11],
ReflectiV A [32], and MMKB-RAG;(ii) GraphRAG meth-
ods: GraphRAG [5], LightRAG [8], HippoRAG2, and Hy-
perGraphRAG [21];(iii) RL-augmented methods: Search-
R1 [14], MMSearch-R1 [38] and Graph-R1 [20].
Implementation Details. We employ GPT-4o-mini
for knowledge construction in both the EvoGraph-R1 and
GraphRAG baselines. For retrieval, we use GME [42].
EvoGraph-R1 is implemented with Qwen2.5-VL-7B asthe base MLLM for multimodal tasks and Qwen2.5-7B-
Instruct for text-only tasks. We conduct experiments using
three random seeds and run all experiments on four 80GB
NVIDIA A100 GPUs. Additional implementation details
are provided in the supplementary document .
4.2. Comparative Performance Analysis - RQ1
Tables 1 and 2 present comprehensive comparisons across
text-only and multimodal benchmarks. EvoGraph-R1 con-
sistently outperforms all baselines, with improvements
more pronounced on multimodal tasks.
Text-Only Performance.We outperform the strongest
baseline (Graph-R1) by +3.5% F1 on 2WikiMultiHopQA,
+2.7% on HotpotQA, and +6.9% on Natural Questions,
achieving 68.5%, 65.4%, and 56.8% respectively. These
improvements underscore the effectiveness of our dynamic
graph evolution and retrieval framework. By progressively
refining entity relations and evidence pathways, our method
surpasses static text-based and graph-based approaches.
Multimodal Performance.On E-VQA, we achieve
43.6% accuracy, outperforming MMSearch-R1 by +6.7%
and MMKB-RAG by +7.7%. On OK-VQA, we reach
68.6%, surpassing GPT-4o-mini by +2.7%. These substan-
tial improvements confirm that dynamically constructing
and refining multimodal knowledge hypergraphs is more ef-
fective than retrieving static documents.
Comparison with Web-search equipped Methods.
EvoGraph-R1 maintains a persistent, evolving knowledge
state instead of discarding evidence after each step. This
enables cumulative reasoning and structured integration
of noisy web content. It outperforms MMSearch-R1 by
+21.8% and Search-R1 by +17.5% on text-only bench-
marks, while also surpassing MMSearch-R1 by +6.7% on
E-VQA and +8.7% on OK-VQA, confirming the advantage
of dynamic graph evolution over transient retrieval across
both text-only and multimodal benchmarks. Importantly,
EvoGraph-R1’s gains stem from the synergy between the
persistent hypergraph environment and RL-driven policy
evolution, which together provide both accuracy and effi-
ciency across text-only and multimodal reasoning.
4.3. Ablation Study - RQ2
Table 3 quantifies individual component contributions on
2WikiMultiHopQA and E-VQA.
Multimodal Hypergraph Environment.Removing the
multimodal hypergraph causes a 5.4% drop on 2Wiki F1
and 4.8% drop on E-VQA accuracy, with retrieval rounds
increasing. This underscores its role in enabling structured
reasoning and cross-modal alignment through unified tex-
tual–visual representation.
Graph Edit Operations.INSERT is most critical: re-
moving it causes an 8.4% drop on 2Wiki and a 6.8% drop
on E-VQA, with 0.8 additional inefficient retrieval rounds.

Table 1.Performance comparison on text-only datasets. Bold
indicates best performance.
Method2Wiki HotpotQA NQ Avg.
Without Retrieval
Qwen2.5-7B-Instruct 12.2 16.6 13.2 14.00
GPT-4o-mini 17.0 31.8 21.6 23.47
Retrieval-Augmented Models
NaiveRAG 20.1 44.3 24.6 29.67
Graph Retrieval-Augmented Models
GraphRAG 16.0 31.7 20.3 22.67
LightRAG 17.6 30.7 19.1 22.47
HippoRAG2 23.5 44.8 24.1 30.80
Retrieval-Augmented Models with Reinforcement Learning
Search-R1-7B 41.3 50.9 45.9 46.03
MMSearch-R1-7B 38.5 45.3 41.6 41.80
Graph-R1-7B 65.0 62.7 49.9 59.20
EvoGraph-R1-7B 68.5 65.4 56.8 63.57
Table 2.Performance comparison on multimodal datasets.
Boldindicates best performance.
MethodE-VQA InfoSeek OK-VQA Avg.
Without Retrieval
Qwen2.5-VL-7B 18.8 26.4 35.0 26.73
GPT-4o-mini 27.2 35.9 65.9 43.00
Retrieval-Augmented Models
NaiveRAG 27.4 28.4 56.1 37.30
Echosight 21.7 30.4 – 26.05
ReflectiV A 29.2 40.1 – 34.65
MMKB-RAG 35.9 36.4 65.4 45.90
Graph Retrieval-Augmented Models
GraphRAG 19.8 24.3 51.8 31.97
LightRAG 17.7 20.3 46.3 28.10
HippoRAG2 22.6 25.6 48.2 32.13
Retrieval-Augmented Models with Reinforcement Learning
MMSearch-R1-7B 36.9 41.3 59.9 46.03
Graph-R1-7B 28.5 29.8 53.6 37.30
EvoGraph-R1-7B 43.6 42.3 68.6 51.50
UPDATE provides substantial gains: removing it results in
drops of 5.5% and 3.9% on 2Wiki and E-VQA, respectively,
confirming the benefits of error correction. DELETE has a
smaller but consistent impact, with 2.4% and 1.5% drops
when removed, validating the benefits of noise filtering.
Web Search.Removing WEBSEARCHleads to substan-
tial performance drops of 9.6% on 2Wiki and 11.2% on
E-VQA, revealing the limitations of static corpora in sup-
porting open-domain and fine-grained multimodal reason-
ing, especially when handling long-tail knowledge. More-
over, one-shot graph construction exacerbates this limita-
tion by introducing information loss and structural ambigu-
ity, which disrupts the continuity of reasoning.Table 3.Ablation study on 2WikiMulti-HopQA and E-VQA.
We report F1/Acc (%) and average retrieval rounds as an efficiency
indicator.
Setting2Wiki E-VQA
F1.↑Rounds Acc.↑Rounds
Full (all components) 68.5 2.57 43.6 1.65
−MM Hypergraph 63.1 2.82 38.8 2.29
−INSERT 60.1 3.48 36.8 2.45
−UPDATE 63.0 2.95 39.7 2.10
−DELETE 66.1 2.70 42.1 1.85
−WEBSEARCH 58.9 3.17 32.4 2.22
4.4. Retrieval Efficiency Analysis - RQ3
We analyze the computational efficiency through retrieval
rounds and response length. As shown in Table 3,
EvoGraph-R1 completes queries in 2.57 rounds on 2Wiki
and 1.65 rounds on E-VQA, while ablated variants re-
quire significantly more:−INSERT needs 3.48 rounds, a
+35.4% increase, and−WEBSEARCHrequires 3.17 rounds,
a +23.3% increase, indicating inefficient searches for un-
available information.
Figure 3 further demonstrates this efficiency advantage.
EvoGraph-R1, shown in purple, converges to fewer retrieval
turns of approximately 2.4 and generates more concise re-
sponses of around 1,300 tokens. This compares favorably
to variants without graph editing, shown in blue with ap-
proximately 3.1 turns and 2,850 tokens, and MMSearch-
R1, shown in green with approximately 3.5 turns and 2,200
tokens. This dual efficiency gain validates that persistent
hypergraph refinement enables targeted knowledge integra-
tion. The agent avoids redundant retrieval by maintaining
a refined state and generates focused answers without com-
pensatory verbosity.
Figure 3.Average Retrieval Turns and Response Length on
2Wiki.
4.5. Generation Quality Evaluation - RQ4
Following HelloBench [28], which employs GPT-4o-mini
as LLM-as-Judge, we assess answer quality across seven di-
mensions. As shown in Figure 4, EvoGraph-R1 consistently
surpasses all baselines, achieving notably higher relevance
and correctness than MMSearch-R1 and Graph-R1. It also
excels in comprehensiveness, factuality, logical coherence,

Figure 4.Comprehensive quality comparison across seven
evaluation dimensions.EvoGraph-R1 (red) consistently out-
performs all baselines, demonstrating superior generation quality
through self-evolving graph refinement.
and knowledgeability, demonstrating that self-evolving
graph refinement reduces hallucination while preserving
reasoning quality. Moreover, EvoGraph-R1 attains the
highest diversity score, confirming that WEBSEARCHin-
tegration and dynamic graph expansion effectively incorpo-
rate varied evidence sources.
4.6. Performance in Low-Resource Settings - RQ5
We evaluate robustness on E-VQA by restricting Wikipedia
to 1%, 5%, and 10% of its full size, simulating limited-
resource settings. Figure 5 shows resilience. At 1% corpus,
EvoGraph-R1 achieves 37.2% accuracy and surpasses base-
lines ranging from 13.2% to 18.9%. Notably, the perfor-
mance gap widens as data decreases. Compared to MMKB-
RAG, the gain increases from +7.7 points at full corpus to
+13.2 points at 1% corpus, and remains large at 5% with
+13.8 and at 10% with +12.9. This trend underscores the
importance of WEBSEARCHand dynamic INSERT when
static knowledge is insufficient.
Figure 5.Performance under limited corpus size on E-VQA.
EvoGraph-R1 (blue) consistently outperforms all baselines, with
the performance gap widening as corpus size decreases.
4.7. Graph Refinement Analysis
We visualize the subgraph centered on “NARRATIVE” be-
fore and after refinement (Figure 6). The pre-refinementTable 4.Graph refinement statistics before and after refine-
ment.Percentage change is computed as(After−Before)/Before.
Metric Before After∆%
Nodes 120,499 123,631 +2.60%
Hyperedges 177,408 181,418 +2.26%
Graph Density 0.781 0.842 +7.81%
Clustering Coefficient 0.024 0.028 +16.67%
Edge Semantic Similarity 0.664 0.685 +3.16%
graph is shallow and loosely organized, with fragmented
single-hop associations and weak thematic cohesion. Af-
ter refinement, the graph consolidates into denser and more
coherent clusters, revealing clearer multi-hop thematic con-
nections and a more structured narrative space.
For quantitative evaluation, we employ standard
complex-network metrics [25] to measure entity and edge
counts, graph density, and the Watts–Strogatz clustering co-
efficient [37], capturing structural cohesion. To assess se-
mantic consistency, we compute edge semantic similarity
using Sentence-BERT embeddings [29]. As summarized
in Table 4, refinement yields consistent improvements in
all structural and semantic metrics, indicating a more con-
nected, coherent, and aligned graph.
(a) Before refinement
 (b) After refinement
Figure 6.Visualization of the hypergraph centered on “NAR-
RATIVE” before and after refinement.The refined graph ex-
hibits clearer thematic structure and deeper reasoning paths.
5. Conclusion
We introduced EvoGraph-R1, a self-evolving GraphRAG
framework that reconceptualizes knowledge graphs as dy-
namic MDP environments rather than static structures. An
autonomous agent iteratively refines a multimodal knowl-
edge hypergraph, transforming sparse initial graphs into co-
herent, query-specific knowledge states. Comprehensive
experiments demonstrate state-of-the-art performance on
both text and multimodal benchmarks, validating that self-
evolving graphs generalize across modalities. This work
establishes agent-driven graph evolution as a fundamental
paradigm for adaptive retrieval, enabling robust reasoning
over incomplete knowledge and dynamic environments.

References
[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ah-
mad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida,
Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al.
Gpt-4 technical report.arXiv preprint arXiv:2303.08774,
2023. 1
[2] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Han-
naneh Hajishirzi. Self-rag: Learning to retrieve, generate,
and critique through self-reflection. InThe Twelfth Interna-
tional Conference on Learning Representations. 2
[3] Xusheng Cao, Haori Lu, Linlan Huang, Fei Yang, Xialei Liu,
and Ming-Ming Cheng. Knowledge graph enhanced gen-
erative multi-modal models for class-incremental learning.
arXiv preprint arXiv:2503.18403, 2025. 2
[4] Yang Chen, Hexiang Hu, Yi Luan, Haitian Sun, Soravit
Changpinyo, Alan Ritter, and Ming-Wei Chang. Can
pre-trained vision and language models answer visual
information-seeking questions? InProceedings of the 2023
Conference on Empirical Methods in Natural Language Pro-
cessing, pages 14948–14968, 2023. 6
[5] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley,
Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropoli-
tansky, Robert Osazuwa Ness, and Jonathan Larson. From
local to global: A graph rag approach to query-focused sum-
marization.arXiv preprint arXiv:2404.16130, 2024. 2, 3,
6
[6] Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani,
Gautier Viaud, C ´eline Hudelot, and Pierre Colombo. Col-
pali: Efficient document retrieval with vision language mod-
els.arXiv preprint arXiv:2407.01449, 2024. 2, 3
[7] Lee T Gettler. Becoming dads: Considering the role of cul-
tural context and developmental plasticity for paternal so-
cioendocrinology.Current Anthropology, 57(S13):S38–S51,
2016. 3
[8] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao
Huang. Lightrag: Simple and fast retrieval-augmented gen-
eration.arXiv preprint arXiv:2410.05779, 2024. 2, 3, 6
[9] Zirui Guo, Xubin Ren, Lingrui Xu, Jiahao Zhang, and Chao
Huang. Rag-anything: All-in-one rag framework.arXiv
preprint arXiv:2510.12323, 2025. 2, 3
[10] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and
Akiko Aizawa. Constructing a multi-hop qa dataset for com-
prehensive evaluation of reasoning steps. InProceedings
of the 28th International Conference on Computational Lin-
guistics, pages 6609–6625, 2020. 6
[11] Ziniu Hu, Ahmet Iscen, Chen Sun, Zirui Wang, Kai-Wei
Chang, Yizhou Sun, Cordelia Schmid, David A Ross, and
Alireza Fathi. Reveal: Retrieval-augmented visual-language
pre-training with multi-source multimodal knowledge mem-
ory. InProceedings of the IEEE/CVF conference on com-
puter vision and pattern recognition, pages 23369–23379,
2023. 2, 6
[12] Chengyue Huang, Brisa Maneechotesuwan, Shivang
Chopra, and Zsolt Kira. Frames-vqa: Benchmarking
fine-tuning robustness across multi-modal shifts in visual
question answering. InProceedings of the Computer Visionand Pattern Recognition Conference, pages 3909–3918,
2025. 2, 6
[13] Xinke Jiang, Rihong Qiu, Yongxin Xu, Yichen Zhu, Ruizhe
Zhang, Yuchen Fang, Chu Xu, Junfeng Zhao, and Yasha
Wang. Ragraph: A general retrieval-augmented graph learn-
ing framework.Advances in Neural Information Processing
Systems, 37:29948–29985, 2024. 3
[14] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan
Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-
r1: Training llms to reason and leverage search engines with
reinforcement learning.arXiv preprint arXiv:2503.09516,
2025. 3, 6
[15] Jiajie Jin, Yutao Zhu, Zhicheng Dou, Guanting Dong, Xinyu
Yang, Chenghao Zhang, Tong Zhao, Zhao Yang, and Ji-Rong
Wen. Flashrag: A modular toolkit for efficient retrieval-
augmented generation research. InCompanion Proceedings
of the ACM on Web Conference 2025, pages 737–740, 2025.
6
[16] Fucai Ke, Joy Hsu, Zhixi Cai, Zixian Ma, Xin Zheng, Xindi
Wu, Sukai Huang, Weiqing Wang, Pari Delir Haghighi,
Gholamreza Haffari, et al. Explain before you answer: A
survey on compositional visual reasoning.arXiv preprint
arXiv:2508.17298, 2025. 2
[17] Yongqi Leng, Yikun Lei, Xikai Liu, Meizhi Zhong, Bojian
Xiong, Yurong Zhang, Yan Gao, Yao Hu, Deyi Xiong, et al.
Decex-rag: Boosting agentic retrieval-augmented generation
with decision and execution optimization via process super-
vision. InProceedings of the 2025 Conference on Empirical
Methods in Natural Language Processing: Industry Track,
pages 1412–1425, 2025. 2
[18] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
K¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel, et al.
Retrieval-augmented generation for knowledge-intensive nlp
tasks.Advances in neural information processing systems,
33:9459–9474, 2020. 3, 6
[19] Shilong Liu, Hao Cheng, Haotian Liu, Hao Zhang, Feng Li,
Tianhe Ren, Xueyan Zou, Jianwei Yang, Hang Su, Jun Zhu,
et al. Llava-plus: Learning to use tools for creating multi-
modal agents. InEuropean conference on computer vision,
pages 126–142. Springer, 2024. 1
[20] Haoran Luo, Guanting Chen, Qika Lin, Yikai Guo, Fangzhi
Xu, Zemin Kuang, Meina Song, Xiaobao Wu, Yifan Zhu,
Luu Anh Tuan, et al. Graph-r1: Towards agentic graphrag
framework via end-to-end reinforcement learning.arXiv
preprint arXiv:2507.21892, 2025. 2, 3, 6
[21] Haoran Luo, Guanting Chen, Yandan Zheng, Xiaobao Wu,
Yikai Guo, Qika Lin, Yu Feng, Zemin Kuang, Meina Song,
Yifan Zhu, et al. Hypergraphrag: Retrieval-augmented gen-
eration via hypergraph-structured knowledge representation.
arXiv preprint arXiv:2503.21322, 2025. 2, 3, 6
[22] Quentin Mac ´e, Ant ´onio Loison, and Manuel Faysse. Vidore
benchmark v2: Raising the bar for visual retrieval.arXiv
preprint arXiv:2505.17166, 2025. 2, 3
[23] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel
Khashabi, and Hannaneh Hajishirzi. When not to trust lan-
guage models: Investigating effectiveness of parametric and

non-parametric memories. InProceedings of the 61st An-
nual Meeting of the Association for Computational Linguis-
tics (Volume 1: Long Papers), pages 9802–9822, 2023. 2
[24] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and
Roozbeh Mottaghi. Ok-vqa: A visual question answer-
ing benchmark requiring external knowledge. In2019
IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pages 3190–3199. IEEE, 2019. 6
[25] Mark EJ Newman. The structure and function of complex
networks.SIAM review, 45(2):167–256, 2003. 8
[26] Jeff Pan, Simon Razniewski, Jan-Christoph Kalo, Sneha
Singhania, Jiaoyan Chen, Stefan Dietze, Hajira Jabeen,
Janna Omeliyanenko, Wen Zhang, Matteo Lissandrini, et al.
Large language models and knowledge graphs: Opportuni-
ties and challenges.Transactions on Graph Data and Knowl-
edge, 2023. 2
[27] Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou
Shi, Chuntao Hong, Yan Zhang, and Siliang Tang. Graph
retrieval-augmented generation: A survey.arXiv preprint
arXiv:2408.08921, 2024. 3
[28] Haoran Que, Feiyu Duan, Liqun He, Yutao Mou, Wangchun-
shu Zhou, Jiaheng Liu, Wenge Rong, Zekun Moore Wang,
Jian Yang, Ge Zhang, et al. Hellobench: Evaluating long
text generation capabilities of large language models.arXiv
preprint arXiv:2409.16191, 2024. 6, 7
[29] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence
embeddings using siamese bert-networks. InProceedings of
the 2019 Conference on Empirical Methods in Natural Lan-
guage Processing and the 9th International Joint Conference
on Natural Language Processing (EMNLP-IJCNLP), pages
3982–3992, 2019. 8
[30] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark,
Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A
benchmark for visual question answering using world knowl-
edge. InEuropean conference on computer vision, pages
146–162. Springer, 2022. 2
[31] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao
Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li,
Yang Wu, et al. Deepseekmath: Pushing the limits of math-
ematical reasoning in open language models.arXiv preprint
arXiv:2402.03300, 2024. 5
[32] Yubo Sun, Chunyi Peng, Yukun Yan, Shi Yu, Zhenghao Liu,
Chi Chen, Zhiyuan Liu, and Maosong Sun. Visrag 2.0:
Evidence-guided multi-image reasoning in visual retrieval-
augmented generation.arXiv preprint arXiv:2510.09733,
2025. 3, 6
[33] Ryota Tanaka, Taichi Iki, Taku Hasegawa, Kyosuke Nishida,
Kuniko Saito, and Jun Suzuki. Vdocrag: Retrieval-
augmented generation over visually-rich documents. InPro-
ceedings of the Computer Vision and Pattern Recognition
Conference, pages 24827–24837, 2025. 2, 3
[34] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and
Ashish Sabharwal. Interleaving retrieval with chain-of-
thought reasoning for knowledge-intensive multi-step ques-
tions. InProceedings of the 61st annual meeting of the as-
sociation for computational linguistics (volume 1: long pa-
pers), pages 10014–10037, 2023. 2[35] Xueyao Wan and Hang Yu. Mmgraphrag: Bridging vi-
sion and language with interpretable multimodal knowledge
graphs.arXiv preprint arXiv:2507.20804, 2025. 2, 3
[36] Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang,
Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan Qu,
Fukai Shang, et al. Mineru: An open-source solution
for precise document content extraction.arXiv preprint
arXiv:2409.18839, 2024. 3
[37] Duncan J Watts and Steven H Strogatz. Collective dynam-
ics of ‘small-world’networks.nature, 393(6684):440–442,
1998. 8
[38] Jinming Wu, Zihao Deng, Wei Li, Yiding Liu, Bo You, Bo
Li, Zejun Ma, and Ziwei Liu. Mmsearch-r1: Incentivizing
lmms to search.arXiv preprint arXiv:2506.20670, 2025. 3,
6
[39] Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei,
and Ming Zhou. Layoutlm: Pre-training of text and layout
for document image understanding. InProceedings of the
26th ACM SIGKDD international conference on knowledge
discovery & data mining, pages 1192–1200, 2020. 3
[40] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christopher D
Manning. Hotpotqa: A dataset for diverse, explainable
multi-hop question answering. InProceedings of the 2018
conference on empirical methods in natural language pro-
cessing, pages 2369–2380, 2018. 2, 6
[41] Hongbin Ye, Ningyu Zhang, Hui Chen, and Huajun Chen.
Generative knowledge graph construction: A review. InPro-
ceedings of the 2022 Conference on Empirical Methods in
Natural Language Processing, pages 1–17, 2022. 2
[42] Xin Zhang, Yanzhao Zhang, Wen Xie, Mingxin Li, Ziqi Dai,
Dingkun Long, Pengjun Xie, Meishan Zhang, Wenjie Li, and
Min Zhang. Gme: Improving universal multimodal retrieval
by multimodal llms.arXiv preprint arXiv:2412.16855, 2024.
4, 6