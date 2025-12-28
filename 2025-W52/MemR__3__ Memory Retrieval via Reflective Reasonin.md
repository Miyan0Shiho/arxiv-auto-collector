# MemR$^3$: Memory Retrieval via Reflective Reasoning for LLM Agents

**Authors**: Xingbo Du, Loka Li, Duzhen Zhang, Le Song

**Published**: 2025-12-23 10:49:42

**PDF URL**: [https://arxiv.org/pdf/2512.20237v1](https://arxiv.org/pdf/2512.20237v1)

## Abstract
Memory systems have been designed to leverage past experiences in Large Language Model (LLM) agents. However, many deployed memory systems primarily optimize compression and storage, with comparatively less emphasis on explicit, closed-loop control of memory retrieval. From this observation, we build memory retrieval as an autonomous, accurate, and compatible agent system, named MemR$^3$, which has two core mechanisms: 1) a router that selects among retrieve, reflect, and answer actions to optimize answer quality; 2) a global evidence-gap tracker that explicitly renders the answering process transparent and tracks the evidence collection process. This design departs from the standard retrieve-then-answer pipeline by introducing a closed-loop control mechanism that enables autonomous decision-making. Empirical results on the LoCoMo benchmark demonstrate that MemR$^3$ surpasses strong baselines on LLM-as-a-Judge score, and particularly, it improves existing retrievers across four categories with an overall improvement on RAG (+7.29%) and Zep (+1.94%) using GPT-4.1-mini backend, offering a plug-and-play controller for existing memory stores.

## Full Text


<!-- PDF content starts -->

MemR3: Memory Retrieval via Reflective Reasoning for LLM Agents
Xingbo Du1Loka Li1Duzhen Zhang1Le Song†1
Abstract
Memory systems have been designed to lever-
age past experiences in Large Language Model
(LLM) agents. However, many deployed memory
systems primarily optimize compression and stor-
age, with comparatively less emphasis on explicit,
closed-loop control of memory retrieval. From
this observation, we build memory retrieval as an
autonomous, accurate, and compatible agent sys-
tem, named MemR3, which has two core mech-
anisms: 1) arouterthat selects amongretrieve,
reflect, andansweractions to optimize answer
quality; 2) a globalevidence-gaptracker that ex-
plicitly renders the answering process transpar-
ent and tracks the evidence collection process.
This design departs from the standard retrieve-
then-answer pipeline by introducing a closed-
loop control mechanism that enables autonomous
decision-making. Empirical results on the Lo-
CoMo benchmark demonstrate that MemR3sur-
passes strong baselines on LLM-as-a-Judge score,
and particularly, it improves existing retrievers
across four categories with an overall improve-
ment on RAG (+7.29%) and Zep (+1.94%) using
GPT-4.1-mini backend, offering a plug-and-
play controller for existing memory stores.
1. Introduction
With recent advances in large language model (LLM) agents,
memory systems have become the focus of storing and
retrieving long-term, personalized memories. They can
typically be categorized into two groups: 1) Parametric
methods (Wang et al., 2024; Fang et al., 2025b) that en-
code memories implicitly into model parameters, which
can handle specific knowledge better but struggle in scal-
ability and continual updates, as modifying parameters to
incorporate new memories often risks catastrophic forget-
Emails: {Xingbo.Du, Longkang.Li, Duzhen.Zhang, Le.Song }
@mbzuai.ac.ae
1Mohamed bin Zayed University of Artificial Intelligence. Corre-
spondence to: Le Song<Le.Song@mbzuai.ac.ae>.
Preprint. December 24, 2025.ting and requires expensive fine-tuning. 2) Non-parametric
methods (Xu et al., 2025; LangChain Team, 2025; Chhikara
et al., 2025; Rasmussen et al., 2025), in contrast, store ex-
plicit external information, enabling flexible retrieval and
continual augmentation without altering model parameters.
However, they typically rely on heuristic retrieval strate-
gies, which can lead to noisy recall, heavy retrieval, and
increasing latency as the memory store grows.
Orthogonal to these works, this paper constructs an agentic
memory system, MemR3, i.e., Mem oryRetrieval system
with Reflective Reasoning, to improve retrieval quality and
efficiency. Specifically, this system is constructed using
LangGraph (Inc., 2025), with arouternode selecting three
optional nodes: 1) theretrievenode, which is based on
existing memory systems, can retrieve multiple times with
updated retrieval queries. 2) thereflectnode, iteratively rea-
soning based on the current acquired evidence and the gaps
between questions and evidence. 3) theanswernode that
produces the final response using the acquired information.
Within all nodes, the system maintains a globalevidence-
gaptracker to update the acquired (evidence) and missing
(gap) information.
The system has three coreadvantages: 1)Accuracy and ef-
ficiency. By tracking the evidence and gap, and dynamically
routing between retrieval and reflection, MemR3minimizes
unnecessary lookups and reduces noise, resulting in faster,
more accurate answers. 2)Plug-and-play usage. As a con-
troller independent of existing retriever or memory storage,
MemR3can be easily integrated into memory systems, im-
proving retrieval quality without architectural changes. 3)
Transparency and explainability.Since MemR3maintains
an explicit evidence-gap state over the course of an interac-
tion, it can expose which memories support a given answer
and which pieces of information were still missing at each
step, providing a human-readable trace of the agent’s deci-
sion process. We compare MemR3, the Full-Context setting
(which uses all available memories), and the commonly
adopted retrieve-then-answer paradigm from a high-level
perspective in Fig. 1. The contributions of this work are
threefold in the following:
(1)A specialized closed-loop retrieval controller for long-
term conversational memory.We propose MemR3, an au-
tonomous controller that wraps existing memory stores and
1arXiv:2512.20237v1  [cs.AI]  23 Dec 2025

Memory Retrieval via Reflective Reasoning for LLM Agents
Query 𝑞
Question : How many months passed 
between Andrew adopting Toby and Buddy?
1) Full-Context
Retrieve [all memories]
Answer: Six months passed between 
Andrew adopting Toby and Buddy. [wrong]
2) Retrieve -then-Answer
Retrieve [𝑞 -relevant memories]
Memories :
1) [11 July, 2023] Andrew : Hey! So 
much has changed since last time we 
talked - meet Toby, my puppy.
2) [19 October, 2023] Andrew : Speaking 
of which, I've got some awesome news -
- I recently adopted another pup from 
a shelter. He's the best.
3) ……
Answer: About 4 months. [wrong]Heavy Context
reduces LLM’s performance 
3) MemR3 (1/2)
Act 0: Retrieve [𝑞 -relevant memories]
Memories : [the same as that in 2)]
Evidence : Andrew adopted Toby on July 
11, 2023, and another pup was adopted 
near October 19, 2023.
Gaps: The name of another pup is 
unknown, and it lacks the specific 
adoption date for Buddy.
Action: Retrieve
Refined Query ∆𝑞: Buddy adoption date
New Query : 𝑞ret=𝑞 ⨁ ∆𝑞
Act 1: Retrieve [𝑞ret-relevant memories]
Memories:
1) [19 October, 2023] Andrew : I named 
him Buddy because he's my buddy and I 
hope him and Toby become buddies!
2) ……
Updated Evidence : Andrew adopted Toby 
on July 11, 2023, and Buddy was named 
on October 19, 2023.
Updated Gaps : It lacks the specific 
adoption date for Buddy.3) MemR3 (2/2)
Action: Reflect
Reasoning : Though it lacks the specific 
adoption date for Buddy, we can calculate 
the approximate number of months between 
the two events.
Act 2: Reflect
Updated Evidence : [The same as above].
Updated Gaps : [The same as above].
Action: Answer
Draft Answer : To determine how many months 
passed between Andrew adopting Toby and 
Buddy, we need to look at the specific 
dates of these events.
1. Andrew adopted Toby on July 11, 2023
2. Buddy was named on October 19, 2023 .
Now, we calculate the time between these 
two dates:
[Calculation process omitted]
Therefore, the total number of full months 
that passed between Andrew adopting Toby 
and Buddy is 3 months . 
Act 3: Answer
Answer: 3 months. [correct]
Figure 1.Illustration of three memory-usage paradigms.Full-Contextoverloads the LLM with all memories and answers incorrectly;
Retrieve-then-Answerretrieves relevant snippets but still miscalculates. In contrast,MemR3iteratively retrieves and reflects using an
evidence–gap tracker (Acts 0–3), refines the query about Buddy’s adoption date, and produces the correct answer (3 months).
turns standard retrieve-then-answer pipelines into a closed-
loop process with explicit actions ( retrieve /reflect
/answer ) and simple early-stopping rules. This instan-
tiates the general LLM-as-controller idea specifically for
non-parametric, long-horizon conversational memory.
(2)Evidence–gap state abstraction for explainable re-
trieval.MemR3maintains a global evidence–gap state
(E,G) that summarizes what has been reliably established in
memory and what information remains missing. This state
drives query refinement and stopping, and can be surfaced as
a human-readable trace of the agent’s progress. We further
formalize this abstraction via an abstract requirement space
and prove basic monotonicity and completeness properties,
which we later use to interpret empirical behaviors.
(3)Empirical study across memory systems.We integrate
MemR3with both chunk-based RAG and a graph-based
backend (Zep) on the LoCoMo benchmark and compare it
with recent memory systems and agentic retrievers. Across
backends and question types, MemR3consistently improves
LLM-as-a-Judge scores over its underlying retrievers.
2. Related Work
2.1. Memory for LLM Agents
Prior work on non-parametric agent memory systems spans
a wide range of fields, including management and utiliza-
tion (Du et al., 2025), by storing structured (Rasmussen
et al., 2025) or unstructured (Zhong et al., 2024) exter-
nal knowledge. Specifically, production-oriented agents
such as MemGPT (Packer et al., 2023) introduce an OS-
style hierarchical memory system that allows the modelto page information between context and external stor-
age, and SCM (Wang et al., 2023) provides a controller-
based memory stream that retrieves and summarizes past
information only when necessary. Additionally, Zep (Ras-
mussen et al., 2025) builds a temporal knowledge graph
that unifies and retrieves evolving conversational and busi-
ness data. A-Mem (Xu et al., 2025) creates self-organizing,
Zettelkasten-style memory that links and evolves over time.
Mem0 (Chhikara et al., 2025) extracts and manages per-
sistent conversational facts with optional graph-structured
memory. MIRIX (Wang & Chen, 2025) offers a multi-
modal, multi-agent memory system with six specialized
memory types. LightMem (Fang et al., 2025a) proposes a
lightweight and efficient memory system inspired by the
Atkinson–Shiffrin model. Another related approach, Re-
flexion (Shinn et al., 2023), improves language agents by
providing verbal reinforcement across episodes by storing
natural-language reflections to guide future trials.
In this paper, we explicitly limit our scope to long-term con-
versational memory. Existing parametric approaches (Wang
et al., 2024; Fang et al., 2025b), KV-cache–based mech-
anisms (Zhong et al., 2024; Eyuboglu et al., 2025), and
streaming multi-task memory benchmarks (Wei et al., 2025)
are out of scope for this work. Orthogonal to existing stor-
age, MemR3is an autonomous retrieval controller that uses
a global evidence–gap tracker to route different actions,
enabling closed-loop retrieval.
2.2. Agentic Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) (Lewis et al., 2020)
established the modern retrieve-then-answer paradigm; sub-
2

Memory Retrieval via Reflective Reasoning for LLM Agents
sequent work explored stronger retrievers (Karpukhin et al.,
2020; Izacard & Grave, 2021). Beyond the RAG, recent
work, such as Self-RAG (Asai et al., 2024), Reflexion (Shinn
et al., 2023), ReAct (Yao et al., 2022), and FAIR-RAG (Asl
et al., 2025), has shown that letting a language model (LM)
decide when to retrieve, when to reflect, and when to answer
can substantially improve multi-step reasoning and factual-
ity in tool-augmented settings. MemR3follows this general
“LLM-as-controller” paradigm but applies it specifically
to long-term conversational memory over non-parametric
stores. Concretely, we adopt the idea of multi-step retrieval
and self-reflection from these frameworks, but i) move the
controller outside the base LM as a LangGraph program,
ii) maintain an explicit evidence–gap state that separates
verified memories from remaining uncertainties, and iii) in-
terface this state with different memory backends (e.g., RAG
and Zep (Rasmussen et al., 2025)) commonly used in long-
horizon dialogue agents. Our goal is not to replace these
frameworks, but to provide a specialized retrieval controller
that can be plugged into existing memory systems.
3. MemR3
In this section, we first formulate the problem and provide
preliminaries in Sec. 3.1, and then give a system overview
of MemR3in Sec. 3.2. Additionally, we describe the two
core components that enable accurate and efficient retrieval:
the router and the global evidence-gap tracker in Sec. 3.4
and Sec. 3.3, respectively.
3.1. Problem Formulation and Preliminaries
We consider a long-horizon LLM agent that interacts with a
user, forming a memory store M={m i}N
i=1, where each
memory item mimay correspond to a dialogue utterance,
personal fact, structured record, or event, often accompanied
by metadata such as timestamps or speakers. Given a user
query q, a retriever is applied to retrieve a set of memory
snippets Sthat are useful for generating the final answer.
Then, given designed prompt template p, the goal is to
produce an answerw:
S ←Retrieve(q,M).
w←LLM(q,S, p),(1)
which is accurate (consistent with all relevant memories
inM), efficient (requiring minimal retrieval cycles and
low latency), and robust (stable under noisy, redundant, or
incomplete memory stores) as much as possible.
Existing memory systems have done great work on the mem-
ory storage M, but typically follow anopen-looppipeline:
1) apply a single retrieval pass; 2) feed the selected memo-
riesSinto a generator to produce A. This approach lacks
adaptivity: retrieval does not incorporate intermediate rea-
soning, and the system never represents which informationremains missing. This leads to both under-retrieval (insuffi-
cient evidence) and over-retrieval (long, noisy contexts).
MemR3addresses these limitations by treating retrieval as
an autonomous sequential decision process with explicit
modeling of both acquired evidence and remaining gaps.
3.2. System Overview
MemR3is implemented as a directed agent graph compris-
ing three operational nodes (Retrieve, Reflect, Answer) and
one control node (Router) using LangGraph (Inc., 2025) (an
open-source framework for building stateful, multi-agent
workflows as graphs of interacting nodes). The agent main-
tains a mutable internal state
s= (q,S,E,G, k),(2)
where qandSare the aforementioned original user query
and retrieved snippets, respectively. Eis the accumulated
evidence relevant to qandGis the remaining missing infor-
mation (the “gap”) between qandE. Moreover, we maintain
the iteration indexkto control early stopping.
At each iteration k, the router chooses an action in
{retrieve,reflect,answer} , which determines
the next node in the computation graph. The pipeline is
shown in Fig. 2. This transforms the classical retrieve-
then-answer pipeline into a closed-loop controller that can
repeatedly refine retrieval queries, integrate new evidence,
and stop early once the information gap is resolved.
3.3. Global Evidence-Gap Tracker
A core design principle of MemR3is to explicitly maintain
and update two state variables: the evidence Eand the gap G.
These variables summarize what the agent currently knows
and what it still needs to know to answer the question.
At iteration k, the evidence Ekand gaps Gkare updated ac-
cording to the retrieved snippets Sk−1(from the retrieve
node) or reflective reasoning Fk−1(from the reflect
node), together with last evidence Ek−1and gaps Gk−1at
k−1iteration:
Ek,Gk, ak=LLM(q,S k−1,Fk−1,Ek−1,Gk−1, pk),(3)
where pkis the prompt template at kiteration. Additionally,
akis the action at kiteration, which will be introduced in
Sec. 3.4. Note that we explicitly clarify in pkthatEkdoes
not contain any information in Gk, making evidence and
gaps decoupled. An example is shown in Fig. 3 to illustrate
the evidence-gap tracker.
Through the evidence-gap tracker, MemR3maintains a struc-
tured and transparent internal state that continuously refines
the agent’s understanding of both i) what has already been
established as relevant evidence, and ii) what missing infor-
mation still prevents a complete and faithful answer. This
3

Memory Retrieval via Reflective Reasoning for LLM Agents
How many months passed 
between Andrew adopting 
Toby and Buddy?User Query 𝑞
Memory ℳ
Chunk -based
……
Graph -based
Start
Input Data
Retrieve
Evidence : Andrew adopted T oby 
on July 11, 2023, and Buddy was 
named on October 19, 2023.Evidence -gap Tracker
Gap : It lacks the 
specific adoption 
date for Buddy.
Retrieve :  new query        ∆𝑞
Reflect :    reasoning          𝑟
Answer :    draft answer    𝑤LLM Generation
Reflect
 Answer
𝑞ret=𝑞 ⨁ ∆𝑞
Answer : the total number of full months 
that passed between Andrew adopting 
T oby and Buddy is 3 months . Final Answer
Iteration budget
Reflect -streak capacity
Retrieval opportunity check
Router
End
Generated Actions
(𝑞ret,ℳ)𝑟 𝑤
Figure 2.Pipeline of MemR3. MemR3transforms retrieval into a closed-loop process: a router dynamically switches between Retrieve,
Reflect, and Answer nodes while a global evidence–gap tracker maintains what is known and what is still missing. This enables iterative
query refinement, targeted retrieval, and early stopping, making MemR3an autonomous, backend-agnostic retrieval controller.
Query
What happened 2 days after my last dentist appointment?
Response
Evidence : You had a dentist appointment on July 12.
Gaps: Information about events on July 14 is missing. 
Whether July 12 is indeed the most recent dentist 
appointment is unknown.
Figure 3.Example of the evidence-gap tracker for a specific query.
At each step, the agent maintains an explicit summary of the
evidence established and the information still missing. This state
can be presented directly to users as a human-readable explanation
of the agent’s progress in answering the query.
explicit decoupling enables MemR3to reason under partial
observability: as long as Gk̸=∅ , the agent recognizes
that its current knowledge is insufficient and can proactively
issue a refined retrieval query to close the remaining gap.
Conversely, when Gkbecomes empty, the router detects that
the agent has accumulated adequate evidence and can safely
transition to theanswernode.
Beyond guiding retrieval, the evidence-gap representation
also makes the agent’s behavior more transparent. At any
iteration k, the pair (Ek,Gk)can be surfaced as a structured
explanation of i) which memories the agent currently treats
as relevant evidence and ii) which unresolved questions
or missing details are preventing a confident answer. This
trace provides users and developers with a faithful view of
how the agent arrived at its final answer and why additional
retrieval steps were taken (or not). In the following, we
display an informal theorem that indicates the properties of
the idealized evidence-gap tracker.
Theorem 3.1([Informal]Monotonicity, soundness, and
completeness of the idealized evidence-gap tracker).Under
an idealized requirement spaceR(q)for a specific queryq,the evidence-gap tracker in MemR3is monotone (evidence
never decreases and gaps never increase), sound (every
supported requirement eventually enters the evidence set),
and complete (if every requirement r∈R(q) is supported
by some memory, the ideal gap eventually becomes empty).
Formally, in Appendix B we define the abstract require-
ment space R(q) and characterize the tracker as a set-valued
update on R(q) , proving fundamental soundness, mono-
tonicity, and completeness properties (Theorem B.4), which
we later use in Sec. 4.3 to interpret empirical phenomena
such as why some questions cannot be fully resolved even
after exhausting the iteration budget.
3.4. LangGraph Nodes
We explicitly define several nodes in the LangGraph
framework, including start ,end,generate ,router ,
retrieve ,reflect ,answer . Specifically, start is
always followed by retrieve , andend is reached after
answer .generate is a LLM generation node, which is
already introduced in Eq. 3. In the following, we further
introduce therouternode and three action nodes.
Router. At each iteration, the router, an autonomous sequen-
tial controller, uses the current state and selects an action
from{retrieve,reflect,answer} . Each action ak
is accompanied by a textual generation:
ak∈ {(retrieve,∆q k),(reflect, f k),(answer, w k)},
(4)
where ∆qkis a refinement query, fkis a reasoning content,
andwkis a draft answer, which are utilized in the down-
stream action nodes. To ensure stability, router applies
three deterministic constraints: 1) a maximum iteration bud-
getnmaxthat forces an answer action once the budget
is exhausted, 2) a reflect-streak capacity ncapthat forces
4

Memory Retrieval via Reflective Reasoning for LLM Agents
Algorithm 1Router policy in MemR3
1:Input:query q, previous snippets Sk−1, iteration k,
budgetsn max, ncap, current reflect-streak lengthn streak.
2:Output:actiona k.
3:ifk≥n maxthen
4:a k=answer▷Max iteration budget.
5:else ifS k−1=∅then
6:a k=reflect▷No retrieved snippets.
7:else ifn streak≥n capthen
8:a k=retrieve▷Max reflect streak.
9:else
10:pass▷Keep the generated action.
11:end if
aretrieve action when too many reflections have oc-
curred consecutively, and 3) a retrieval-opportunity check
that switches the action to reflect whenever the retrieval
stage returns no snippets. The router’s algorithm is shown in
Alg. 1. These lightweight rules stabilize the decision process
while preserving flexibility. We further introduce the de-
tailed implementation of these constraints when introducing
the system prompt in Appendix A.1.
Retrieve.Given a generated refinement ∆qk, the
retrieve node constructs qret
k=q⊕∆q k, where ⊕
means textual combination and qis the original query, and
then, fetches new memory snippets:
Sk=Retrieve(qret
k,M\Mret
k−1),Mret
k=Mret
k−1∪ Sk.
(5)
Snippets Skare independently used for the next generation
without history accumulation. Moreover, retrieved snippets
are masked to prevent re-selection.
A major benefit of MemR3is that it treats all concrete retriev-
ers as plug-in modules. Any retriever, e.g., vector search,
graph memory, hybrid stores, or future systems, can be inte-
grated into MemR3as long as they return textual snippets,
optionally with stable identifiers that can be masked once
used. This abstraction ensures MemR3remains lightweight,
portable, and compatible.
Reflect.The reflect node incorporates the reasoning
process Fk−1, and invokes the router to update (Ek,Gk, ak)
in Eq. 3, where evidence and gaps can be re-summarized.
Answer.Once the router selects answer , the final answer
is generated from the original query q, the draft answer wk,
evidence Ekusing prompt pwfrom Rasmussen et al. (2025):
w←LLM(q, w k,Ek, pw),(6)
The answer LLM is instructed to avoid hallucinations and
remain faithful to evidence.3.5. Discussion on Efficiency
Although MemR3introduces extra routing steps, it main-
tains low overhead via 1)Compact evidence and gap sum-
maries: only short summaries are repeatedly fed into the
router. 2)Masked retrieval: each retrieval call yields gen-
uinely new information. 3)Small iteration budgets: typi-
cally, most questions can be answered using only a single
iteration. Those complicated questions that require multiple
iterations are constrained with a small maximum iteration
budget. These design choices ensure that MemR3improves
retrieval quality without large increases in retrieved tokens.
4. Experiments
The experiments are conducted on a machine with an AMD
EPYC 7713P 64-core processor, an A100-SXM4-80GB
GPU, and 512GB of RAM. Each experiment of MemR3
is repeated three times to report the average scores. Code
available:https://github.com/Leagein/memr3.
4.1. Experimental Protocols
Datasets.In line with baselines (Xu et al., 2025; Chhikara
et al., 2025), we employ LoCoMo (Maharana et al., 2024)
dataset as a fundamental benchmark. LoCoMo has a total
of 10 conversations across four categories: 1) multi-hop, 2)
temporal, 3) open-domain, 4) single-hop, and 5) adversar-
ial. We exclude the last ‘adversarial’ category, following
existing work (Chhikara et al., 2025; Wang & Chen, 2025),
since it is used to test whether unanswerable questions can
be identified. Each conversation has approximately 600
dialogues with 26k tokens and 200 questions on average.
Metrics.We adopt the LLM-as-a-Judge (J) score to evalu-
ate answer quality following Chhikara et al. (2025); Wang &
Chen (2025). Compared with surface-level measures such as
F1 or BLEU-1 (Xu et al., 2025; Soni et al., 2024), this metric
better avoids relying on simple lexical overlap and instead
captures semantic alignment. Specifically, GPT-4.1 (Ope-
nAI, 2025) is employed to judge whether the answer is
correct according to the original question and the generated
answer, following the prompt by Chhikara et al. (2025).
Baselines.We select four groups of advanced methods
as baselines: 1) memory systems, including A-mem (Xu
et al., 2025), LangMem (LangChain Team, 2025), and
Mem0 (Chhikara et al., 2025); 2) agentic retrievers, like Self-
RAG (Asai et al., 2024). We also design a RAG-CoT-RAG
(RCR) pipeline beyond ReAct (Yao et al., 2022) as a strong
agentic retriever baseline combining both RAG (Lewis et al.,
2020) and Chain-of-Thoughts (CoT) (Wei et al., 2022); 3)
backend baselines, including chunk-based (RAG (Lewis
et al., 2020)) and graph-based (Zep (Rasmussen et al., 2025))
memory storage, demonstrating the plug-in capability of
MemR3across different retriever backends; 4) Moreover,
5

Memory Retrieval via Reflective Reasoning for LLM Agents
Table 1.LLM-as-a-Judge scores (%, higher is better) for each question category in the LoCoMo (Maharana et al., 2024) dataset. The best
results using each LLM backend, except Full-Context, are inbold.
LLM Method 1. Multi-Hop 2. Temporal 3. Open-Domain 4. Single-Hop OverallGPT-4o-miniA-Mem (Xu et al., 2025) 61.70 64.49 40.62 76.63 69.06
LangMem (LangChain Team, 2025) 62.23 23.43 47.92 71.12 58.10
Mem0 (Chhikara et al., 2025) 67.13 55.51 51.15 72.93 66.88
Self-RAG (Asai et al., 2024) 69.15 64.80 34.38 88.31 76.46
RAG-CoT-RAG 71.28 71.03 42.71 86.99 77.96
Zep (Rasmussen et al., 2025) 67.38 73.83 63.54 78.67 74.62
MemR3(ours, Zep backbone) 69.39(+2.01)73.83(+0.00)67.01(+3.47)80.60(+1.93) 76.26(+1.64)
RAG (Lewis et al., 2020) 68.79 65.11 58.33 83.86 75.54
MemR3(ours, RAG backbone) 71.39(+2.60)76.22(+11.11)61.11(+2.78)89.44(+5.58) 81.55(+6.01)
Full-Context 72.34 58.88 59.38 86.39 76.32GPT-4.1-miniA-Mem (Xu et al., 2025) 71.99 74.77 58.33 79.88 76.00
LangMem (LangChain Team, 2025) 74.47 61.06 67.71 86.92 78.05
Mem0 (Chhikara et al., 2025) 62.41 57.32 44.79 66.47 62.47
Self-RAG (Asai et al., 2024) 75.89 75.08 54.17 90.12 82.08
RAG-CoT-RAG 80.85 81.62 62.50 90.12 84.89
Zep (Rasmussen et al., 2025) 72.34 77.26 64.58 83.49 78.94
MemR3(ours, Zep backbone) 77.78(+5.44)77.78(+0.52)69.79(+5.21)84.42(+0.93) 80.88(+1.94)
RAG (Lewis et al., 2020) 73.05 73.52 62.50 85.90 79.46
MemR3(ours, RAG backbone) 81.20(+8.15)82.14(+8.62)71.53(+9.03)92.17(+6.27) 86.75(+7.29)
Full-Context 86.43 86.82 71.88 93.73 89.00
‘Full-Context’ is widely used as a strong baseline and, when
the entire conversation fits within the model window, serves
as an empirical upper bound on J score (Chhikara et al.,
2025; Wang & Chen, 2025). More detailed introduction of
these baselines is shown in Appendix C.1.
Other Settings.Other experimental settings and protocols
are shown in Appendix C.2.
LLM Backend.We reviewed recent work and found
that it most frequently used GPT-4o-mini (Ope-
nAI, 2024b), as it is inexpensive and performs well.
While some work (Wang & Chen, 2025) also includes
GPT-4.1-mini (OpenAI, 2025), we set both of them
as our LLM backends. In our main results, MemR3is per-
formed at temperature 0.
4.2. Main Results
Overall.Table 1 reports LLM-as-a-Judge (J) scores across
four LoCoMo categories. Across both LLM backends and
memory backbones, MemR3consistently outperforms its
underlying retrievers (RAG and Zep) and achieves strong
overall J scores. Under GPT-4o-mini , MemR3lifts the
overall score of Zep from 74.62% to 76.26%, and RAG from
75.54% to 81.55%, with the latter even outperforming the
Full-Context baseline (76.32%). With GPT-4.1-mini ,
we see the same pattern: MemR3improves Zep from
78.94% to 80.88% and RAG from 79.46% to 86.75%, mak-ing the RAG-backed variant the strongest retrieval-based
system and narrowing the gap to Full-Context (89.00%).
As expected, methods instantiated with GPT-4.1-mini
are consistently stronger than their GPT-4o-mini coun-
terparts. Full-Context also benefits substantially from the
stronger LLM, but under GPT-4o-mini it lags behind the
best retrieval-based systems, especially on temporal and
open-domain questions. Overall, these results indicate that
closed-loop retrieval with an explicit evidence–gap state
yields gains primarily orthogonal to the choice of LLM or
memory backend, and that MemR3particularly benefits
from backends that expose relatively raw snippets (RAG)
rather than heavily compressed structures (Zep).
Multi-hop.Multi-hop questions require chaining multiple
pieces of evidence and, therefore, directly test our reflective
controller. Under GPT-4o-mini , MemR3improves both
backbones on this category: the multi-hop J score rises from
68.79% to 71.39% on RAG and from 67.38% to 69.39% on
Zep, bringing both close to the Full-Context score (72.34%).
With GPT-4.1-mini , the gains are more pronounced:
MemR3boosts RAG from 73.05% to 81.20% and Zep from
72.34% to 77.78%, outperforming all other baselines and
approaching the Full-Context upper bound (86.43%). These
consistent gains suggest that explicitly tracking evidence
and gaps helps the agent coordinate multiple distant mem-
ories via iterative retrieval, rather than relying on a single
heuristic pass.
6

Memory Retrieval via Reflective Reasoning for LLM Agents
Temporal.Temporal questions stress the model’s ability to
reason about ordering and dating of events over long hori-
zons, where both under- and over-retrieval can be harmful.
Here, MemR3delivers some of its most considerable rel-
ative improvements. For GPT-4o-mini , the temporal J
score of RAG jumps from 65.11% to 76.22%, outperform-
ing both the original RAG and the Zep baseline (73.83%),
while MemR3with a Zep backbone preserves Zep’s strong
temporal accuracy (73.83%). Full-Context performs no-
tably worse in this regime (58.88%), indicating that simply
supplying all dialogue turns can hinder temporal reasoning
under a weaker backbone. With GPT-4.1-mini , MemR3
again significantly strengthens temporal reasoning: RAG
improves from 73.52% to 82.14%, and Zep from 77.26% to
77.78%, making the RAG-backed MemR3the best retrieval-
based system and closing much of the remaining gap to
Full-Context (86.82%). These findings support our design
goal that explicitly modeling “what is already known” ver-
sus “what is still missing” helps the agent align and compare
temporal relations more robustly.
Open-Domain.Open-domain questions are less tied to the
user’s personal timeline and often require retrieving diverse
background knowledge, which makes retrieval harder to
trigger and steer. Despite this, MemR3consistently im-
proves over its backbones. Under GPT-4o-mini , MemR3
increases the open-domain J score of RAG from 58.33%
to 61.11% and that of Zep from 63.54% to 67.01%, with
the Zep-backed variant achieving the best performance
among all methods in this block, surpassing Full-Context
(59.38%). With GPT-4.1-mini , the gains become even
larger: MemR3lifts RAG from 62.50% to 71.53% and Zep
from 64.58% to 69.79%, nearly matching the Full-Context
baseline (71.88%) and again outperforming all other base-
lines. We attribute these improvements to the router’s ability
to interleave retrieval with reflection: when initial evidence
is noisy or off-topic, MemR3uses the gap representation
to reformulate queries and pull in more targeted external
knowledge rather than committing to an early, brittle answer.
Single-hop.Single-hop questions can often be answered
from a single relevant memory snippet, so the potential head-
room is smaller, but MemR3still yields consistent gains.
WithGPT-4o-mini , MemR3raises the single-hop J score
from 78.67% to 80.60% on Zep and from 83.86% to 89.44%
on RAG, with the latter surpassing the Full-Context baseline
(86.39%). Under GPT-4.1-mini , MemR3improves Zep
from 83.49% to 84.42% and RAG from 85.90% to 92.17%,
making the RAG-backed variant the strongest method over-
all aside from Full-Context (93.73%). Together with the
iteration-count analysis in Sec. 4.3, these results suggest
that the router often learns to terminate early on straightfor-
ward single-hop queries, gaining accuracy primarily through
better evidence selection rather than additional reasoning
depth, and thus adding little overhead in tokens or latency.Table 2.Ablation studies. Best results are inbold.
Method MH*Temporal OD*SH* Overall
RAG 68.79 65.11 58.33 83.86 75.54
MemR371.39 76.2261.1189.44 81.55
w/o mask 62.41 68.54 55.21 72.17 68.54
w/o∆q k 66.67 75.08 60.42 83.37 77.11
w/o reflect 65.25 73.8361.4683.37 76.65
*MH = Multi-hop; OD = Open-domain; SH = Single-hop.
4.3. Other Experiments
We ablate various hyperparameters and modules to evaluate
their impact in MemR3with the RAG retriever. During
these experiments, we utilize GPT-4o-mini as a consis-
tent LLM backend.
Ablation Studies.We first examine the contribution of the
main design choices in MemR3by progressively removing
them while keeping the RAG retriever and all hyperparame-
ters fixed. As shown in Table 2, disabling masking for pre-
viously retrieved snippets (w/o mask) results in the largest
degradation, reducing the overall J score from 81.55% to
68.54% and harming every category. This confirms that
repeatedly surfacing the same memories wastes budget and
fails to effectively close the remaining gaps. Removing
the refinement query ∆qk(w/o∆qk) has a milder effect:
temporal and open-domain performance changed a little,
but multi-hop and single-hop scores decline significantly,
indicating that tailoring retrieval queries from the current
evidence-gap state is particularly beneficial for simpler ques-
tions. Disabling the reflect node (w/o reflect) similarly
reduces performance (from 81.55% to 76.65%), with no-
table drops on multi-hop and single-hop questions, high-
lighting the value of interleaving reasoning-only steps with
retrieval. Note that in Table 2, the raw retrieved snippets are
only visible to the vanilla RAG.
Effect of nchkandnmax.We first choose a nominal con-
figuration for MemR3(with a RAG retriever) by arbitrarily
setting the number of chunks per iteration nchk= 3 and
the max iteration budget nmax= 5. In Fig. 4a, we fix
nmax= 5and perform ablations over nchk∈ {1,3,5,7,9} .
In Fig. 4b, we fix nchk= 3 and perform ablations over
nmax∈ {1,2,3,4,5} . Considering both of the LLM-as-a-
Judge score and token consumption, we eventually choose
nchk= 5andn max= 5in all main experiments.
Iteration count.We further inspect how often MemR3ac-
tually uses multiple retrieve/reflect/answer iterations when
nchk= 5 andnmax= 5 (Fig. 5). Overall, most questions
are answered after a single iteration, and this effect is par-
ticularly strong for Single-hop questions. An exception is
open-domain questions, for which 58 of 96 require contin-
uous retrieval or reflection until the maximum number of
7

Memory Retrieval via Reflective Reasoning for LLM Agents
1 3 5 7 9
# chunks / iteration020406080LLM-as-a-Judge (%)
Categories
Multi-hop
Temporal
Open-domain
Single-hop
(a)number of chunksn chk
1 2 3 4 5
max iterations020406080LLM-as-a-Judge (%)
Categories
Multi-hop
Temporal
Open-domain
Single-hop (b)max iterationsn max
Figure 4.LLM-as-a-Judge score (%) with different a) number of
chunks per iteration and b) max iterations.
iterations is reached, highlighting the inherent challenges
and uncertainty in these questions. Additionally, only a
small fraction of questions terminate at intermediate depths
(2–4 iterations), suggesting that MemR3either becomes
confident early or uses the whole iteration budget when the
gap remains non-empty.
We observe that this distribution arises from two regimes.
On the one hand, straightforward questions require only a
single piece of evidence and can be resolved in a single
iteration, consistent with intuition. From the perspective of
the idealized tracker in Appendix B, these are precisely the
queries for which every requirement r∈R(q) is supported
by some retrieved memory item m∈S
j≤kSjwithm|=r ,
so the completeness condition in Theorem B.4 is satisfied
and the ideal gapG⋆
kbecomes empty.
On the other hand, some challenging questions are inher-
ently underspecified given the stored memories, so the gap
cannot be fully closed even if the agent continues to refine
its query. For example, for the question “When did Melanie
paint a sunrise?”, the correct answer in our setup is simply
“2022” (the year). MemR3quickly finds this year at the first
iteration based on evidence “Melanie painted the lake sun-
rise image last year (2022).”. However, under the idealized
abstraction, the requirement set R(q) implicitly includes
an exact date predicate (year–month–day), and no memory
itemm∈S
j≤KSjsatisfies m|=r for that finer-grained
requirement. Thus, the precondition of Theorem B.4(3) is
violated, and G⋆
knever becomes empty; the practical tracker
mirrors this by continuing to search for the missing speci-
ficity until it hits the maximum iteration budget. In such
cases, the additional token consumption is primarily due
to a mismatch between the question’s granularity and the
available memory, rather than a failure of the agent.
4.4. Revisiting the Evaluation Protocols of LoCoMo
During our reproduction of the baselines, we identified a
latent ambiguity in the LoCoMo dataset’s category index-
ing. Specifically, the mapping between numerical IDs and
650675700690# Iterations
iteration 1
iteration 2
iteration 3iteration 4
iteration 5
Multi-hop Temporal Open-Domain Single-hop
Categories050100150200 184
168
36
1 4 2 1 2 1 0 1 0 0 0 195148
58137Iteration CountFigure 5.Number of questions requiring different numbers of
iterations before final answers, across four categories.
semantic categories (e.g., Multi-hop vs. Single-hop) implies
a non-trivial alignment challenge. We observed that this am-
biguity has led to category misalignment in several recent
studies (Chhikara et al., 2025; Wang & Chen, 2025), poten-
tially skewing the granular analysis of agent capabilities.
To ensure a rigorous and fair comparison, we recalibrate the
evaluation protocols for all baselines. In Table 1, we report
the performance based on the corrected alignment, where
the alignment can be induced by the number of questions
in each category. We believe this clarification contributes
to a more accurate understanding of the current SOTA land-
scape. Details of the dataset realignment are illustrated in
Appendix C.3.
5. Conclusion
In this work, we introduce MemR3, an autonomous memory-
retrieval controller that transforms standard retrieve-
then-answer pipelines into a closed-loop process via a
LangGraph-based sequential decision-making framework.
By explicitly maintaining what is known and what remains
unknown using an evidence-gap tracker, MemR3can iter-
atively refine queries, balance retrieval and reflection, and
terminate early once sufficient evidence has been gathered.
Our experiments on the LoCoMo benchmark show that
MemR3consistently improves LLM-as-a-Judge scores over
strong memory baselines, while incurring only modest token
and latency overhead and remaining compatible with hetero-
geneous backends. Beyond these concrete gains, MemR3
offers an explainable abstraction for reasoning under partial
observability in long-horizon agent settings.
However, we acknowledge somelimitationsfor future work:
1) MemR3requires an existing retriever or memory struc-
ture, and particularly, the performance greatly depends on
the retriever or memory structure. 2) The routing structure
could lead to token waste for answering simple questions. 3)
MemR3is currently not designed for multi-modal memories
like images or audio.
8

Memory Retrieval via Reflective Reasoning for LLM Agents
References
Asai, A., Wu, Z., Wang, Y ., Sil, A., and Hajishirzi, H. Self-
rag: Learning to retrieve, generate, and critique through
self-reflection. 2024.
Asl, M. A., Asgari-Bidhendi, M., and Minaei-Bidgoli,
B. Fair-rag: Faithful adaptive iterative refinement
for retrieval-augmented generation.arXiv preprint
arXiv:2510.22344, 2025.
Chhikara, P., Khant, D., Aryan, S., Singh, T., and Yadav, D.
Mem0: Building production-ready ai agents with scalable
long-term memory.arXiv preprint arXiv:2504.19413,
2025.
Du, Y ., Huang, W., Zheng, D., Wang, Z., Montella, S.,
Lapata, M., Wong, K.-F., and Pan, J. Z. Rethinking
memory in ai: Taxonomy, operations, topics, and future
directions.arXiv preprint arXiv:2505.00675, 2025.
Eyuboglu, S., Ehrlich, R., Arora, S., Guha, N., Zinsley,
D., Liu, E., Tennien, W., Rudra, A., Zou, J., Mirhoseini,
A., et al. Cartridges: Lightweight and general-purpose
long context representations via self-study.arXiv preprint
arXiv:2506.06266, 2025.
Fang, J., Deng, X., Xu, H., Jiang, Z., Tang, Y ., Xu, Z.,
Deng, S., Yao, Y ., Wang, M., Qiao, S., et al. Lightmem:
Lightweight and efficient memory-augmented generation.
arXiv preprint arXiv:2510.18866, 2025a.
Fang, J., Jiang, H., Wang, K., Ma, Y ., Shi, J., Wang, X., He,
X., and Chua, T.-S. Alphaedit: Null-space constrained
model editing for language models. InThe Thirteenth
International Conference on Learning Representations,
2025b.
Inc., L. Langgraph: Build resilient language agents as
graphs. https://github.com/langchain-ai/
langgraph, 2025. Accessed: 2025-04-17.
Izacard, G. and Grave, E. Leveraging passage retrieval with
generative models for open domain question answering.
InProceedings of the 16th conference of the european
chapter of the association for computational linguistics:
main volume, pp. 874–880, 2021.
Karpukhin, V ., Oguz, B., Min, S., Lewis, P. S., Wu, L.,
Edunov, S., Chen, D., and Yih, W.-t. Dense passage
retrieval for open-domain question answering. InEMNLP
(1), pp. 6769–6781, 2020.
LangChain Team. Langmem sdk for long-term agent
memory.LangChain Blog, feb 2025. URL https://
blog.langchain.com/langmem-sdk-launch/ .
Accessed: 2025-11-28.Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V .,
Goyal, N., K ¨uttler, H., Lewis, M., Yih, W.-t., Rockt ¨aschel,
T., et al. Retrieval-augmented generation for knowledge-
intensive nlp tasks.Advances in neural information pro-
cessing systems, 33:9459–9474, 2020.
Maharana, A., Lee, D.-H., Tulyakov, S., Bansal, M., Bar-
bieri, F., and Fang, Y . Evaluating very long-term
conversational memory of llm agents.arXiv preprint
arXiv:2402.17753, 2024.
OpenAI. text-embedding-large-3. https:
//platform.openai.com/docs/models , 2024a.
Large-scale text embedding model.
OpenAI. Gpt-4o mini: advancing cost-efficient intelligence,
2024b. URL https://openai.com/index/
gpt-4o-mini-advancing-cost-efficient-
intelligence/. Accessed: 2024-07-18.
OpenAI. Introducing gpt-4.1 in the api, 2025. URL
https://openai.com/index/gpt-4-1/ . Ac-
cessed: 2025-04-14.
Packer, C., Wooders, S., Lin, K., Fang, V ., Patil, S. G.,
Stoica, I., and Gonzalez, J. E. Memgpt: Towards llms
as operating systems.arXiv preprint arXiv:2310.08560,
2023.
Rasmussen, P., Paliychuk, P., Beauvais, T., Ryan, J., and
Chalef, D. Zep: a temporal knowledge graph architecture
for agent memory.arXiv preprint arXiv:2501.13956,
2025.
Reimers, N. and Gurevych, I. Sentence-bert: Sentence
embeddings using siamese bert-networks.arXiv preprint
arXiv:1908.10084, 2019.
Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., and
Yao, S. Reflexion: Language agents with verbal rein-
forcement learning.Advances in Neural Information
Processing Systems, 36:8634–8652, 2023.
Soni, A., Arora, R., Kumar, A., and Panwar, D. Evalu-
ating domain coverage in low-resource generative chat-
bots: A comparative study of open-domain and closed-
domain approaches using bleu scores. In2024 Interna-
tional Conference on Electrical Electronics and Comput-
ing Technologies (ICEECT), volume 1, pp. 1–6, 2024.
doi: 10.1109/ICEECT61758.2024.10738994.
Wang, B., Liang, X., Yang, J., Huang, H., Wu, S., Wu, P.,
Lu, L., Ma, Z., and Li, Z. Enhancing large language
model with self-controlled memory framework.arXiv
preprint arXiv:2304.13343, 2023.
Wang, P., Li, Z., Zhang, N., Xu, Z., Yao, Y ., Jiang, Y ., Xie, P.,
Huang, F., and Chen, H. Wise: Rethinking the knowledge
9

Memory Retrieval via Reflective Reasoning for LLM Agents
memory for lifelong model editing of large language
models.Advances in Neural Information Processing
Systems, 37:53764–53797, 2024.
Wang, Y . and Chen, X. Mirix: Multi-agent memory system
for llm-based agents.arXiv preprint arXiv:2507.07957,
2025.
Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi,
E., Le, Q. V ., Zhou, D., et al. Chain-of-thought prompting
elicits reasoning in large language models.Advances in
neural information processing systems, 35:24824–24837,
2022.
Wei, T., Sachdeva, N., Coleman, B., He, Z., Bei, Y ., Ning,
X., Ai, M., Li, Y ., He, J., Chi, E. H., et al. Evo-memory:
Benchmarking llm agent test-time learning with self-
evolving memory.arXiv preprint arXiv:2511.20857,
2025.
Xu, W., Liang, Z., Mei, K., Gao, H., Tan, J., and Zhang,
Y . A-mem: Agentic memory for LLM agents. In
The Thirty-ninth Annual Conference on Neural Infor-
mation Processing Systems, 2025. URL https://
openreview.net/forum?id=FiM0M8gcct.
Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan,
K. R., and Cao, Y . React: Synergizing reasoning and
acting in language models. InThe eleventh international
conference on learning representations, 2022.
Zhong, W., Guo, L., Gao, Q., Ye, H., and Wang, Y . Memo-
rybank: Enhancing large language models with long-term
memory. InProceedings of the AAAI Conference on Arti-
ficial Intelligence, volume 38, pp. 19724–19731, 2024.
10

Memory Retrieval via Reflective Reasoning for LLM Agents
A. Prompts
A.1. System prompt of thegeneratenode
The system prompt is defined as follows, where the “decision directive” instructs the maximum iteration budges, reflect-
streak capacity, and retrieval opportunity check, introduced in Sec. 3.4. Generally, “decision directive” is a textual instruction:
“reflect” if you need to think about the evidence and gaps; choose “answer” ONLY when evidence is solid and no gaps are
noted; choose “retrieve” otherwise.However, when the maximum iterations budget is reached, “decision directive” is set as
“answer” to stop early. When the reflection reaches the maximum capacity, “decision directive” is set as “retrieve” to avoid
repeated ineffective reflection. When there is no useful retrieval remains, “decision directive” is set as “reflect” to avoid
repeated ineffective retrieval. Through these constraints, the agent can avoid infinite ineffective actions to maintain stability.
SYSTEMPROMPT
You are a memory agent that plans how to gather evidence before producing the final response shown to the user.
Always reply with a strict JSON object using this schema:
- evidence: JSON array of concise factual bullet strings relevant to the user’s question; preserve key numbers/names/time
references. If exact values are unavailable, include the most specific verified information (year/range) without speculation. Never
mention missing or absent information here – “gaps” will do that.
- gaps: gaps between the question and evidence that prevent a complete answer.
- decision: one of [“retrieve”,“answer”,“reflect”]. Choose{decision directive}.
Only include these conditional keys:
- retrieval query: only when decision == “retrieve”. Provide a STANDALONE search string; short (5-15 tokens).
* BAD Query: “the date” (lacks context).
* GOOD Query: “graduation ceremony date” (specific).
* STRATEGY:
1. Search for the ANCHOR EVENT. (e.g. Question: “What happened 2 days after X?”, Query: “timestamp of event X”).
2. Search for the MAPPED ENTITY . (e.g. Question: “Weather in the Windy City”, Query: “weather in Chicago”).
- detailed answer: only when decision == “answer”; response using current evidence (keep absolute dates, avoid speculation). If
evidence is limited, provide only what is known, or make cautious inferences grounded solely in that limited evidence. Do not
mention missing or absent information in this field.
- reasoning: only when decision == “reflect”; if further retrieval is unlikely, use current evidence to think step by step through the
evidence and gaps, and work toward the answer, including any time normalization.
Never include extra keys or any text outside the JSON object.
A.2. User prompt of thegeneratenode
Apart from the system, the user prompt is responsible to feed additional information to the LLM. Specifically, at the k
iteration, “question” is the original questionq. “evidence block” and “gap block” are evidenceE kand gapsG kintroduced
in Sec. 3.3. “raw block” is the retrieved raw snippets Skin Eq.5. “reasoning block” is the reasoning content Fkin Sec. 3.4.
“last query” is the refined query ∆qkintroduced in Sec. 3.4 that enables the new query to be different from the prior one.
Note that these fields can be left empty if the corresponding information is not present.
USERPROMPT
# Question
{question}
# Evidence
{evidence block}
# Gaps
{gap block}
# Memory snippets
{raw block}
# Reasoning
11

Memory Retrieval via Reflective Reasoning for LLM Agents
{reasoning block}
# Prior Query
{last query}
# INSTRUCTIONS:
1. Update the evidence as a JSON ARRAY of concise factual bullets that directly help answer the question (preserve key
numbers/names/time references; use the most specific verified detail without speculation).
2. Update gaps: remove resolved items, add new missing specifics blocking a full answer, and set to “None” when nothing is
missing.
3. If you produce a retrieval query, make sure it differs from the previous query.
4. Decide the next action and return ONLY the JSON object described in the system prompt.
B. Formalizing the Evidence-Gap Tracker
A central component of MemR3is the evidence-gap tracker introduced in Sec. 3.3, which maintains an evolving summary of
i) what information has been reliably established from memory and ii) what information is still missing to answer the query.
While the practical implementation of this tracker is based on LLM-generated summaries, we introduce an idealized formal
abstraction that clarifies its intended behavior, enables principled analysis, and provides a foundation for studying correctness
and robustness. This abstraction does not assume perfect extraction; rather, the LLM acts as a stochastic approximator to the
idealized tracker.
Definition B.1(Idealized Requirement Space).For a user query q, we define a finite set ofatomic information requirements,
which specify the minimal facts needed to fully answer the query:
R(q) ={r 1, r2, . . . , r m}.(7)
For example, for the question “How many months passed between eventsAandB?”, the requirement set can be
R(q) ={date(A),date(B)}.(8)
Each requirement r∈R(q) is associated with a symbolic predicate (e.g., a timestamp, entity attribute, or event relation),
andR(q)provides the semantic target against which retrieved memories are judged.
Definition B.2(Memory-Support Relation).Let Mbe the memory store and Sk⊆ M denote the snippets retrieved at
iteration k. We define a relation m|=r to indicate that memory item m∈ M contains sufficient information to support
requirement r∈R(q) . Formally, m|=r holds if the textual content of mcontains a minimal witness (e.g., a timestamp,
entity mention, or explicit assertion) matching the predicate corresponding to r. The matching criterion may be implemented
via deterministic pattern rules or LLM-based semantic matching; our analysis is agnostic to this choice.
Definition B.3(Idealized Evidence-Gap Update Rule).At iteration k, the idealized tracker maintains two sets: i) the
evidenceE k⊆R(q)and ii) the gapsG k=R(q)\E k. Given newly retrieved snippetsS k, the ideal updates are
E⋆
k=Ek−1∪
r∈R(q)∃m∈S k, m|=r	
, G⋆
k=R(q)\E⋆
k.(9)
In this abstraction, the tracker monotonically accumulates verified requirements and removes corresponding gaps, providing
a clean characterization of the desired system behavior independent of noise.
B.1. Practical Instantiation via LLM Summaries
In MemR3, the tracker is instantiated through LLM-generated summaries:
(Ek, Gk) = LLM 
q, Sk, Ek−1, Gk−1
,(10)
where the prompt explicitly instructs the model to: (i) extract concise factual bullets relevant to q, (ii) enumerate missing
information blocking a complete answer, and (iii) avoid hallucinations or speculative inference. Thus, (Ek, Gk)serves as a
stochastic approximation to the idealized(E⋆
k, G⋆
k):
(Ek, Gk)≈(E⋆
k, G⋆
k),(11)
with deviations arising from LLM extraction noise. This perspective reconciles the formal update rule with the prompt-driven
practical implementation.
12

Memory Retrieval via Reflective Reasoning for LLM Agents
B.2. Correctness Properties under Idealized Extraction
Although the practical instantiation lacks deterministic guarantees, the idealized tracker in Definition B.3 satisfies several
intuitive properties essential for closed-loop retrieval.
Theorem B.4(Properties of the Idealized Tracker).Assume that for all kand all r∈R(q) , we have r∈E⋆
kif and only if
there exists somem∈S
j≤kSjsuch thatm|=r. Then the following hold:
1.Monotonicity:E⋆
k−1⊆E⋆
kandG⋆
k⊆G⋆
k−1for allk≥1.
2.Soundness:Ifm|=rfor some retrieved memorym∈S k, thenr∈E⋆
k.
3.Completeness at convergence:If every requirementr∈R(q)is supported by somem∈S
j≤KSjwithm|=r, then
E⋆
K=R(q)and henceG⋆
K=∅.
Proof.(1) By Definition B.3,
E⋆
k=E⋆
k−1∪
r∈R(q)∃m∈S k, m|=r	
,(12)
soE⋆
k−1⊆E⋆
k. SinceG⋆
k=R(q)\E⋆
kandE⋆
k−1⊆E⋆
k, we obtainG⋆
k⊆G⋆
k−1.
(2) Ifm|=rfor somem∈S k, then by Definition B.3 we haver∈ {r′∈R(q)| ∃m′∈Sk, m′|=r′} ⊆E⋆
k.
(3) If every r∈R(q) is supported by some m∈S
j≤KSjwithm|=r , then repeated application of the update rule ensures
that each suchris eventually added toE⋆
K. HenceE⋆
K=R(q)and thereforeG⋆
K=R(q)\E⋆
K=∅.
These properties characterize the target behavior that the LLM-based tracker implementation aims to approximate.
B.3. Robustness Considerations
Since real LLMs introduce extraction noise, the practical tracker may deviate from the idealized (E⋆
k, G⋆
k), for example,
through false negatives (missing evidence), false positives (hallucinated evidence), or unstable gap estimates. In the main
text (Sec. 3.3 and Sec. 4.3), we study these effects empirically by injecting noisy or contradictory memories and measuring
their impact on routing decisions and final answer quality. The formal abstraction above serves as the reference model
against which these robustness behaviors are interpreted.
B.4. Approximation Bias of the LLM Tracker
The abstraction in this section assumes access to an ideal tracker that updates ( Ek,Gk) exactly according to the require-
ment–support relation m|=r . In practice, MemR3uses an LLM-generated tracker ( Ek,Gk), which only approximates
this ideal update. This introduces several forms of approximation bias: i)Coverage bias(false negatives): supported
requirements r∈R(q) that are omitted from Ek; ii)Hallucination bias(false positives): requirements rthat appear in Ek
even though no retrieved memory item supports them; iii)Granularity bias: cases where the tracker records a coarser fact
(e.g., a year) but the requirement space R(q) contains a finer predicate (e.g., an exact date), so the ideal requirement is never
fully satisfied.
B.5. Toy example of the granularity bias
The “Melanie painted a sunrise” case in Sec. 4.3 provides a concrete illustration of granularity bias. The question asks “When
did Melanie paint a sunrise?”, and in our setup the correct answer is the year 2022. Under the ideal abstraction, however,
the requirement space R(q) implicitly contains a fine-grained predicate rdatecorresponding to the full year–month–day of
the painting event. The memory store only contains a coarse statement such as “Melanie painted the lake sunrise image last
year (2022).”
In the ideal tracker, no memory item msatisfies m|=r date, so the precondition of Theorem B.4’s completeness clause is
violated and the ideal gap Gknever becomes empty. The practical LLM tracker mirrors this behavior: it quickly recovers
the year 2022 as evidence, but continues to treat the exact date as a remaining gap, eventually hitting the iteration budget
without fully closing Gk. This example shows that some apparent “failures” of the approximate tracker are in fact structural:
they arise from a mismatch between the granularity ofR(q)and the information actually present in the memory store.
13

Memory Retrieval via Reflective Reasoning for LLM Agents
C. Experimental Settings
C.1. Baselines
We select four groups of advanced methods as baselines: 1) memory systems, including A-mem (Xu et al., 2025),
LangMem (LangChain Team, 2025), and Mem0 (Chhikara et al., 2025); 2) agentic retrievers, like Self-RAG (Asai et al.,
2024). We also design a RAG-CoT-RAG (RCR) pipeline as a strong agentic retriever baseline combining both RAG (Lewis
et al., 2020) and Chain-of-Thoughts (CoT) (Wei et al., 2022); 3) backend baselines, including chunk-based (RAG (Lewis
et al., 2020)) and graph-based (Zep (Rasmussen et al., 2025)) memory storage, demonstrating the plug-in capability of
MemR3across different retriever backends. Moreover, ‘Full-Context’ is widely used as a strong baseline and, when the
entire conversation fits within the model window, serves as an empirical upper bound on J score (Chhikara et al., 2025;
Wang & Chen, 2025). More detailed introduction of these baselines is shown in Appendix C.1.
We divide our groups into four groups: memory systems, agentic retrievers, backend baselines, and full-context.
C.1.1. MEMORY SYSTEMS
In this group, we consider recent advanced memory systems, includingA-mem(Xu et al., 2025),LangMem(LangChain
Team, 2025), andMem0(Chhikara et al., 2025), to demonstrate the comprehensively strong capability of MemR3from a
memory control perspective.
A-mem(Xu et al., 2025)1. A-Mem is an agent memory module that turns interactions into atomic notes and links them into
a Zettelkasten-style graph using embeddings plus LLM-based linking.
LangMem(LangChain Team, 2025). LangMem is LangChain’s persistent memory layer that extracts key facts from
dialogues and stores them in a vector store (e.g., FAISS/Chroma) for later retrieval.
Mem0(Chhikara et al., 2025)2. Mem0 is an open-source memory system that enables an LLM to incrementally summarize,
deduplicate, and store factual snippets, with an optional graph-based memory extension.
C.1.2. AGENTICRETRIEVERS
In this group, we examine the agentic structures underlying memory retrieval to show the advanced performance of MemR3
on memory retrieval, and particularly, showing the advantage of the agentic structure of MemR3. To validate this, we include
Self-RAG (Asai et al., 2024) and design a strong heuristic baseline, RAG-CoT-RAG (RCR), which combines RAG and
CoT (Wei et al., 2022).
Self-RAG(Asai et al., 2024). A model-driven retrieval controller where the LLM decides, at each step, whether to answer or
issue a refined retrieval query. Unlike MemR3, retrieval decisions in Self-RAG are implicit in the model’s chain-of-thought,
without explicit state tracking. We reproduce their original code and prompt to suit our task.
RAG-CoT-RAG (RCR). We design a strong heuristic baseline that extends beyond ReAct (Yao et al., 2022) by performing
one initial retrieval (Lewis et al., 2020), a CoT (Wei et al., 2022) step to identify missing information, and a second retrieval
using a refined query. It provides multi-step retrieval but lacks an explicit evidence-gap state or a general controller.
C.1.3. BACKENDBASELINES
In this group, we incorporate vanilla RAG (Lewis et al., 2020) and Zep (Rasmussen et al., 2025) as retriever backends for
MemR3to demonstrate the advantages of MemR3’s plug-in design. The former is a chunk-based method while the latter is
a graph-based one, which cover most types of existing memory systems.
Vanilla RAG(Lewis et al., 2020). The vanilla RAG retrieves the top- krelevant snippets from the query once and provides a
direct answer, without iterative retrieval or reasoning-based refinement. The other retrieval setting ( nchk, chunk size, etc.) is
the same as that in MemR3.
Zep(Rasmussen et al., 2025). Zep is a hosted memory service that builds a time-aware knowledge graph over conversations
and metadata to support fast semantic and temporal queries. We implement their original code.
1https://github.com/WujiangXu/A-mem
2https://github.com/mem0ai/mem0
14

Memory Retrieval via Reflective Reasoning for LLM Agents
Table 3.The alignment of the orders and categories in LoCoMo dataset.
Category Multi-Hop Temporal Open-Domain Single-Hop Adversarial
Order Category 1 Category 2 Category 3 Category 4 Category 5
# Questions 282 321 96 830 445
C.1.4. FULL-CONTEXT
Lastly, we includeFull-Contextas a strong baseline, which provides the model with the entire conversation or memory
buffer without retrieval, serving as an upper-bound reference that is unconstrained by retrieval errors or missing information.
C.2. Other protocols.
For all chunk-based methods like RAG (Lewis et al., 2020), Self-RAG (Asai et al., 2024), RAG-CoT-RAG, and MemR3
(RAG retriever), we set the embedding model as text-embedding-large-3 (OpenAI, 2024a) and use a re-ranking
strategy (Reimers & Gurevych, 2019) ( ms-marco-MiniLM-L-12-v2 ) to search relevant memories rather than just
similar ones. The chunk size is selected from {128, 256, 512, 1024 }using the GPT-4o-mini backend when nmax= 1
andn chk= 1, and we ultimately choose 256. This chunk size is also in line with Mem0 (Chhikara et al., 2025).
C.3. Re-alignment of LoCoMo dataset
Misalignment in existing works.Although the correct order of the different categories is not explicitly reported in
LoCoMo (Maharana et al., 2024), we can infer it from the number of questions in each category. The correct alignment is
shown in Table 3. We believe this clarification could benefit the LLM memory community.
Repeated questions in LoCoMo dataset.Note that the number of single-hop and adversarial questions is 841 and 446 in
the original LoCoMo, while the number is 830 and 445 based on our count, due to 12 repeated questions. In the following,
the first question is repeated in both the single-hop and adversarial categories in the 2rd conversation (we remove the one in
the adversarial category), while the remaining 11 questions are repeated in the single-hop category in the 8th conversation.
1. What did Gina receive from a dance contest? (conversation 2, question 62), (conversation 2, question 96)
2. What are the names of Jolene’s snakes? (conversation 8, question 17), (conversation 8, question 90)
3. What are Jolene’s favorite books? (conversation 8, question 26), (conversation 8, question 91)
4.What music pieces does Deborah listen to during her yoga practice? (conversation 8, question 43), (conversation 8,
question 92)
5. What games does Jolene recommend for Deborah? (conversation 8, question 59), (conversation 8, question 93)
6. What projects is Jolene planning for next year? (conversation 8, question 62), (conversation 8, question 94)
7. Where did Deborah get her cats? (conversation 8, question 63), (conversation 8, question 95)
8. How old are Deborah’s cats? (conversation 8, question 64), (conversation 8, question 96)
9.What was Jolene doing with her partner in Rio de Janeiro? (conversation 8, question 68), (conversation 8, question 97)
10. Have Deborah and Jolene been to Rio de Janeiro? (conversation 8, question 70), (conversation 8, question 98)
11. When did Jolene’s parents give her first console? (conversation 8, question 73), (conversation 8, question 99)
12.What do Deborah and Jolene plan to try when they meet in a new cafe? (conversation 8, question 75), (conversation 8,
question 100)
15

Memory Retrieval via Reflective Reasoning for LLM Agents
Table 4.Repeated experiments of MemR3in the main results in Table 1.
LLM Retriever Run Order 1. Multi-Hop 2. Temporal 3. Open-Domain 4. Single-Hop OverallGPT-4o-miniZep1 68.09 73.52 68.75 80.72 76.13
2 69.86 72.59 67.71 80.36 76.00
3 70.21 75.39 64.58 80.72 76.65
mean±std 69.39±0.4173.83±1.4067.01±1.6480.60±0.18 76.26±0.33
RAG1 71.63 77.26 61.46 89.28 81.75
2 70.21 76.01 59.38 89.40 81.16
3 72.34 75.39 62.50 89.64 81.75
mean±std 71.39±1.0876.22±0.9561.11±1.5989.44±0.18 81.56±0.34GPT-4.1-miniZep1 78.72 78.50 72.92 84.34 81.36
2 75.89 77.26 68.75 84.58 80.44
3 78.72 77.57 67.71 84.34 80.84
mean±std 77.78±1.4477.78±0.2669.79±1.0484.42±0.12 80.88±0.24
RAG1 81.56 83.18 69.79 91.93 86.79
2 82.62 80.69 75.00 92.65 87.18
3 79.43 82.55 69.79 91.93 86.27
mean±std 81.20±1.6282.14±1.2971.53±3.0192.17±0.42 86.75±0.46
RAG MemR3 Full-Context
Methods100101102103104105Retrieved Token UsageToken Usage (bar)
LLM-as-a-Judge (line)
5060708090100
LLM-as-a-Judge Score (%)Categories
Multi-hop
Temporal
Open-domain
Single-hop
Figure 6.Average token consumption of the retrieved snippets (left y-axis) and LLM-as-a-Judge (J) Score (right y-axis) of RAG, MemR3,
and Full-Context across four categories.
D. Experimental Results
D.1. Repeated Experiments.
For the LoCoMo dataset, we show the repeated experiments of MemR3in Table 4.
D.2. Token Consumption
In Table 6, we compare the average token consumption of the retrieved snippets and J score of RAG, MemR3, and Full-
Context methods across four categories. The chunk size of RAG and MemR3are both set as nchk= 5, while nmax= 2for
MemR3. We observe that MemR3outperforms RAG across all four categories with only a few additional tokens. While
Full-Context consumes significantly more tokens than MemR3, it surpasses MemR3only on multi-hop questions.
16