# ComoRAG: A Cognitive-Inspired Memory-Organized RAG for Stateful Long Narrative Reasoning

**Authors**: Juyuan Wang, Rongchen Zhao, Wei Wei, Yufeng Wang, Mo Yu, Jie Zhou, Jin Xu, Liyan Xu

**Published**: 2025-08-14 07:52:09

**PDF URL**: [http://arxiv.org/pdf/2508.10419v1](http://arxiv.org/pdf/2508.10419v1)

## Abstract
Narrative comprehension on long stories and novels has been a challenging
domain attributed to their intricate plotlines and entangled, often evolving
relations among characters and entities. Given the LLM's diminished reasoning
over extended context and high computational cost, retrieval-based approaches
remain a pivotal role in practice. However, traditional RAG methods can fall
short due to their stateless, single-step retrieval process, which often
overlooks the dynamic nature of capturing interconnected relations within
long-range context. In this work, we propose ComoRAG, holding the principle
that narrative reasoning is not a one-shot process, but a dynamic, evolving
interplay between new evidence acquisition and past knowledge consolidation,
analogous to human cognition when reasoning with memory-related signals in the
brain. Specifically, when encountering a reasoning impasse, ComoRAG undergoes
iterative reasoning cycles while interacting with a dynamic memory workspace.
In each cycle, it generates probing queries to devise new exploratory paths,
then integrates the retrieved evidence of new aspects into a global memory
pool, thereby supporting the emergence of a coherent context for the query
resolution. Across four challenging long-context narrative benchmarks (200K+
tokens), ComoRAG outperforms strong RAG baselines with consistent relative
gains up to 11% compared to the strongest baseline. Further analysis reveals
that ComoRAG is particularly advantageous for complex queries requiring global
comprehension, offering a principled, cognitively motivated paradigm for
retrieval-based long context comprehension towards stateful reasoning. Our code
is publicly released at https://github.com/EternityJune25/ComoRAG

## Full Text


<!-- PDF content starts -->

ComoRAG: A Co gnitive-Inspired M emory-O rganized RAG
for Stateful Long Narrative Reasoning
Juyuan Wang*1Rongchen Zhao*1Wei Wei2Yufeng Wang1
Mo Yu4Jie Zhou4Jin Xu1,3Liyan Xu†4
1School of Future Technology, South China University of Technology
2Independent Researcher3Pazhou Lab, Guangzhou
4Pattern Recognition Center, WeChat AI, Tencent
Abstract
Narrative comprehension on long stories and novels has been
a challenging domain attributed to their intricate plotlines
and entangled, often evolving relations among characters and
entities. Given the LLM’s diminished reasoning over ex-
tended context and high computational cost, retrieval-based
approaches remain a pivotal role in practice. However, tra-
ditional RAG methods can fall short due to their stateless,
single-step retrieval process, which often overlooks the dy-
namic nature of capturing interconnected relations within
long-range context. In this work, we propose ComoRAG,
holding the principle that narrative reasoning is not a one-
shot process, but a dynamic, evolving interplay between new
evidence acquisition and past knowledge consolidation, anal-
ogous to human cognition when reasoning with memory-
related signals in the brain. Specifically, when encountering a
reasoning impasse, ComoRAG undergoes iterative reasoning
cycles while interacting with a dynamic memory workspace.
In each cycle, it generates probing queries to devise new ex-
ploratory paths, then integrates the retrieved evidence of new
aspects into a global memory pool, thereby supporting the
emergence of a coherent context for the query resolution.
Across four challenging long-context narrative benchmarks
(200K+ tokens), ComoRAG outperforms strong RAG base-
lines with consistent relative gains up to 11% compared to the
strongest baseline. Further analysis reveals that ComoRAG
is particularly advantageous for complex queries requiring
global comprehension, offering a principled, cognitively mo-
tivated paradigm for retrieval-based long context comprehen-
sion towards stateful reasoning. Our code is publicly released
at https://github.com/EternityJune25/ComoRAG.
1 Introduction
The core challenge of long-context narrative comprehension
lies not merely in connecting discrete pieces of evidence,
a task more naturally defined as multi-hop Question An-
swering (QA), but in performing a dynamic cognitive syn-
thesis to grasp necessary background and content progres-
sion (Xu et al. 2024a). Unlike multi-hop QA (Yang et al.
2018), which seeks a static path through fixed facts, narrative
comprehension requires emulating a human reader: contin-
uously building and revising a global mental model of the
*These authors contributed equally.
†Project lead. Correspondence to: <liyanlxu@tencent.com >
(c) ComoRAG
(b) Multi-step RAGWhy Snape Kill DumbledoreAlbus?StatelessReasoningStatefulReasoning
Fragmented EvidenceLack of FusionMemory-OrganizedStatefulComprehension
ContradictoryEvidence Motive UnclearAn act of loyaltynot betrayal
SnapeBullyHarrySnape KillsAlbusSnapeProtectHarry
(a) Single-step RAGShallow/Superficial Understanding
Snape was a loyal Death EaterThe Half-BloodPrince where Snape kills Albus
Coherent Context Formed
Apparent ContradictionCausally Incomplete Event
Snape KillsAlbus
SnapeProtectHarry
Snape KillsAlbus
SnapeProtectHarry
Snape KillsAlbusSnape UnbreakableVow
Context-Grounded Exploratory Probing
DynamicMemoryWorkspaceContext-Grounded Exploratory ProbingKnowledgeConsolidationKnowledgeConsolidationKnowledgeConsolida1onFigure 1: Comparison of RAG reasoning paradigms.
plot, characters, and their evolving motivations (Johnson-
Laird 1983). The complexity of this process is well exem-
plified by a classic narrative question “Why did Snape kill
Dumbledore?” from the Harry Potter series. Answering this
requires weaving a complete web of evidence from disparate
clues spanning multiple books—Dumbledore’s terminal ill-
ness, the Unbreakable V ow, and Snape’s deeply concealed
loyalty. The true significance of these clues is only fully rec-
onciled in hindsight. This capability is what we term state-
ful reasoning : it demands more than linking static evidence;
it requires maintaining a dynamic memory of the narrative,
one that is constantly updated as new revelations emerge.
Long-context LLMs have demonstrated promising perfor-
mance on benchmarks such as the “Needle in a Haystack”
test (Eisenschlos, Yogatama, and Al-Rfou 2023) in recent
years. However, their capacity to process long narratives
(200k+ tokens) remains limited by finite context windows.
Furthermore, as the input length increases, these models are
prone to the “lost in the middle” problem (Liu et al. 2024),
which raises perplexity and impairs generation quality. This
limitation is particularly pronounced in narrative compre-arXiv:2508.10419v1  [cs.CL]  14 Aug 2025

hension tasks which require stateful reasoning. As a result,
retrieval-augmented generation (RAG) (Lewis et al. 2020)
has emerged as an important strategy for tackling long con-
text comprehension with LLMs.
However, existing RAG methods still struggle to effec-
tively address this challenge. Advanced single-step retrieval
remains limited by its static index. This includes methods
such as RAPTOR (Sarthi et al. 2024), which clusters and
summarizes text chunks to retrieve at different levels of de-
tails; HippoRAGv2 (Guti ´errez et al. 2025), which mimics
the human hippocampus by building a knowledge graph
to achieve multi-hop reasoning in a single retrieval step.
Nonetheless, single-step methods rely on one-shot static re-
trieval, which may lead to shallow comprehension. For ex-
ample, in Figure 1(a), the evidence about Snape can mislead
the model into making a false inference.
As a remedy, multi-step retrieval methods offer a more
promising direction, such as IRCoT (Trivedi et al. 2023),
which interleaves the retrieval process with Chain-of-
Thought reasoning (Wei et al. 2022); Self-RAG (Asai et al.
2024), which trains a model to adaptively retrieve and re-
flect on evidence; and MemoRAG (Qian et al. 2025), which
uses a dual-system architecture to generate clues from com-
pressed global context. These methods all target to obtain
richer context through iterative retrieval. However, their re-
trieval steps are typically independent, which lack coherent
reasoning throughout explicit narrative progression, featur-
ing fragmented evidence with a stateless comprehension. As
illustrated in Figure 1(b), due to a lack of dynamic memory,
multi-step retrieval fails to integrate contradictory evidence
such as “Snape protects/bullies Harry” and cannot under-
stand the evolution of his actions, ultimately unable to yield
the correct answer.
In this work, we seek inspiration from the function of Pre-
frontal Cortex (PFC) in human brains, which employs a so-
phisticated reasoning process called Metacognitive Regula-
tion (Fernandez-Duque, Baird, and Posner 2000). This pro-
cess is not a single action but a dynamic interplay between
new evidence acquisition , driven by goal-directed memory
probes (Dobbins and Han 2006; Miller and Constantinidis
2024), and subsequent knowledge consolidation . During
consolidation, new findings are integrated with past informa-
tion to construct an evolving, coherent narrative. This itera-
tive cycle allows the PFC to continuously assess its under-
standing and revise its strategy, providing a direct cognitive
blueprint for our framework’s stateful reasoning approach.
We introduce ComoRAG, a co gnitive-inspired, m emory-
organized RAG framework, imitating the human Prefrontal
Cortex (PFC) for achieving true stateful reasoning. At its
core is a dynamic cognitive loop operating on a memory
workspace, which actively probes and integrates new evi-
dence to build a coherent narrative comprehension.
This process, as illustrated in Figure 1(c), is a closed loop
of evolving reasoning states. Faced with a complex query
like“Why did Snape kill Dumbledore?” , the system’s mem-
ory state evolves from an initial “causally incomplete event”
(Snape kills Albus ), to an “apparent contradiction” upon
finding contradictory information ( Snape protects Harry ),
and ultimately to a logically consistent coherent contextthrough deeper exploration and evidence fusion. Only in this
final, complete cognitive state can ComoRAG perform the
correct stateful reasoning, deriving the profound insight that
it was “an act of loyalty, not betrayal” .
This cognitively-inspired design yields substantial im-
provements across four challenging long-context narrative
benchmarks. ComoRAG is shown to consistently outper-
form all categories of strong baselines across each dataset.
Our analysis reveals several key findings. First, these gains
stem directly from the cognitive loop, which transforms a
static knowledge base into a dynamic reasoning engine; for
instance, accuracy on EN.MC jumps from a static-retrieval
baseline of 64.6% to 72.9%, with performance efficiently
converging in around 2-3 cycles. Second, our framework ex-
cels on narrative queries that require global understanding
of plot progression, achieving up to a 19% relative F1 im-
provement on these challenging question types where oth-
ers falter. Finally, our framework demonstrates remarkable
modularity and generalizability. Its core loop can be flexi-
bly integrated to existing RAG methods such as RAPTOR,
which directly yields a 21% relative accuracy gain). Also,
switching to a stronger model as the backbone LLM agents
can upgrade reasoning in the entire cognitive loop, attaining
accuracy from 72.93% to 78.17%. These results collectively
validate that ComoRAG provides a principled, cognitively-
inspired new paradigm for retrieval-based long narrative
comprehension towards stateful reasoning.
2 Methodology
We introduce ComoRAG, an autonomous cognitive archi-
tecture designed to formalize and implement the process of
Metacognitive Regulation outlined in the Introduction. The
architecture’s design is directly inspired by the functional
mechanisms of the Prefrontal Cortex (PFC) and is founded
on three conceptual pillars: (1) a Hierarchical Knowledge
Source for deep contextual understanding; (2) a Dynamic
Memory Workspace for tracking and integrating the multi-
turn reasoning; and (3) a Metacognitive Control Loop that
drives the entire resolving procedure.
2.1 Problem Formulation: Towards Principled
Narrative Reasoning
Our objective is to design a framework for stateful reason-
ing in RAG scenarios. Especially, it aims to resolve those
queries that require global context comprehension in the
first place, commonly seen in narratives, where conventional
RAG may fail to recognize relevant context based on the sur-
face form of queries. Formally, denote the initial query as
qinit, and a knowledge source Xderived upon the original
context, our framework Fleverages a series of adaptive op-
erations to yield the final answer, Afinal , through discrete
time steps t= 1, . . . , T with underlying memory control.
At the beginning of each step t,Fdetermines its focus
of reasoning—a set of new probing queries P(t), represent-
ing new information to seek that may logically deepen the
query comprehension and ultimately complement the an-
swer resolution. With newly retrieved information by P(t)
at each step, the framework utilizes the global memory pool

NewStep[A] Response to the government's letter[B] Sending a birthday card to my niece[C] Write a letter to the Sunday Comet newspaper[D] Blurring Photos
QueryMrs. MacIntyre never writes letters, so what is the sudden purpose of buying ink?
𝑷𝒓𝒐𝒃𝒆:Mrs. MacIntyrenever writes letters, so what is the sudden purpose of buyingink?
Cue	: A page of the newspaper had aportion cut out𝑪𝒖𝒆𝟏	Among Mrs. McGinty’s belongings was a copy of the Sunday Comet.𝑪𝒖𝒆𝟐 A page of the newspaper had a portion cut out.𝑪𝒖𝒆𝟑 Mrs. McGinty may have sent a letter to the newspaper. 𝑪𝒖𝒆𝟒   ......Cue	: Mrs. McGinty may have sent a letter to the newspaper. I will create a cue from theretrieved evidence to complement the resolution of theQueryI will find the most relevant historical memory unitsand create a fused cue
Mem-UpdateMrs. McGinty’s recognized someone in the photo and tried to sell it to the Sunday Comet. So the answer is [C].Query+	𝑪(𝟐)FinalAnswer𝑪(𝟐)=∪𝑴𝒑𝒐𝒐𝒍(𝟐)=𝑴𝒑𝒐𝒐𝒍(𝟏)⋃
Mem-EncodeAfter reviewing past probe, I will generate 3new probing queriesSelf-Probe
Mem-Fuse
𝑴(𝟎)
𝑷𝒉𝒊𝒔𝒕(𝟏)𝑴(𝟎)𝑪𝒇𝒖𝒔𝒆(𝟐)
Tri-RetrieveTry-Answer
𝑷(𝟐)𝓔𝒑𝒕𝒚𝒑𝒆
𝑪𝒑𝒕𝒚𝒑𝒆𝑴𝒆𝒏𝒄𝒐𝒅𝒆(𝟐)
Hierarchical Knowledge Source𝑴(𝟏)𝑷𝒓𝒐𝒃𝒆:What did Mrs. McGinty recognize in the clipping, and how did she try to profit from it?𝑴(𝟐)One Probe yieldsthree types of evidences
Steps 1
Steps 0
Failure
𝑷𝒓𝒐𝒃𝒆:What did Miss Pamela Hosford recall about a letter sent to the Sunday Comet?𝑴(𝟏)
Cue	: Mrs. McGinty wanted to make some money from the photo.Repeat𝑴(𝟎)𝑴(𝟏)Repeat
Evidence	: “…receiving a vague letter from a woman about a photograph…”Evidence	: “…The paper she bought before her death had a rectangular section cut out…”Evidence	: “…She liked eavesdropping, knew who was in the photo, and asked what we‘d pay...”
MemoryUnits(step2)MemoryUnits(step1)MemoryUnits(step0)high-levelBackgroundsummaryQAAgentComprehensionAgentIntegrationAgentRegulationAgentEvidence（three types）SynthesizedcueProbingQuery Failure
𝑪(𝟏)𝑴(𝟐)Legend
Input
Figure 2: An illustration of ComoRAG. Triggered by a reasoning impasse (Failure), the Metacognitive Regulation loop con-
sists of five core operations described in Section 2.3: 1) Self-Probe to devise new exploratory probing queries based on past
memory units; 2) Tri-Retrieve to retrieve evidence from three knowledge sources; 3) Mem-Encode to form new memory units on
how the latest evidence of new aspects could complement the final query resolution; 4) Mem-Fuse to generate cues integrating
new and past memory units; 5) Try-Answer to perform query answering using new memory information produced in this cycle.
maintained till the prior step M(t−1)
pool, and produces either
the final answer, or a Failure Signal, indicating a reason-
ing impasse—and updates the memory pool to M(t)
pool, ac-
complishing a cognitive cycle that synergizes between the
knowledge source, memory space and retrieval operations.
2.2 The Hierarchical Knowledge Source
To overcome the limitations of a monolithic representation
of the given context, our framework first builds a hierarchi-
cal knowledge index Xfor retrieval that models the raw text
from three complementary cognitive dimensions, analogous
to how the PFC integrates different memory types from var-
ious brain regions, particularly supporting cross-layer rea-
soning from raw evidence to abstract relationships.
Veridical Layer: Grounding in Factual Evidence. To
ensure all reasoning is traceable to source evidence, a veridi-
cal layer Xveris firstly established, constituted by raw text
chunks directly, analogous to the precise recall of factual de-
tails in human memory. For more accurate retrieval on text
chunks, we instruct a LLM to generate knowledge triples
(subject -predicate -object ) for each text chunk. These triples
participate in each retrieval, and strengthen the matching be-
tween an incoming query and the corresponding text chunk,
which is proven effective by HippoRAG (Jimenez Gutierrez
et al. 2024). Further details are described in Appendix B.
Semantic Layer: Abstracting Thematic Structure. To
capture thematic and conceptual connections that transcend
across long-range contextual dependencies, a semantic layer
Xsemis built, inspired by the prior work RAPTOR that
employs a GMM-driven clustering algorithm to recursivelysummarize semantically similar text chunks into a hierar-
chical summary tree. We reckon such semantic abstraction
is necessary for deeper comprehension and follow the same
formulism. These summary nodes enable the framework to
retrieve conceptual information beyond the surface level.
Episodic Layer: Reconstructing Narrative Flow. The
previous two layers equip views of both factual details and
high-level concepts. However, they lack temporal devel-
opment or plot progression that can be especially crucial
for narratives. To enable such view with long-range causal
chains, we introduce the episodic layer, Xepi, which aims
to reconstruct the plotline and story arc by capturing the se-
quential narrative development. The process features a slid-
ing window summarization across text chunks; each result-
ing node is then a summary that aggregates the narrative de-
velopment of continuous or causally related events accord-
ing to the timeline. Optionally, the sliding window process
can be applied recursively to form higher-level views of con-
tent progression, extracting different levels of narrative flow
as part of the knowledge source.
2.3 The Architecture of Metacognitive Regulation
The core of ComoRAG is a control loop that fully realizes
the concept of metacognitive regulation. It is composed of
aRegulatory Process for reflection and planning at each
step, and a Metacognitive Process for executing reasoning
and memory management with the Memory Workspace .
Dynamic Memory Workspace. The memory workspace
contains memory units that serve as the bridge for a co-
hesive multi-step exploration and reasoning by metacog-

nitive regulation. Each memory unit mfunctionally con-
cludes one retrieval operation , denoted as a tuple of three
elements: m= (p,Etype
p,Ctype
p), where pis the probing
query that triggers this retrieval; Etype
p is the homogeneous
set of evidence retrieved from a single knowledge layer
(type∈ {ver, sem, epi }); and Ctype
p is a synthesized cue
that reflects how these retrieved evidence by the probe p
could complement the comprehension and resolution of the
original query qinit. Concretely, Ctype
p is generated by a
LLM in the role of Comprehension Agent, πcue, denoted as
Ctype
p=πcue(qinit, p,Etype
p).
The formation of a memory unit (p,Etype
p,Ctype
p)by each
retrieval is defined as a Mem-Encode operation. The memory
workspace/pool will be utilized and updated throughout the
reasoning cycle described below.
The Regulatory Process. The regulatory process is in-
voked at the beginning of a reasoning cycle/step tif the pre-
ceding cycle t−1is concluded in failure. The core operation,
Self-Probe , plans new probing queries of which retrieved in-
formation may contribute to the final answer, thereby de-
vising new exploratory paths to break the impasse. It is or-
chestrated by a Regulation Agent ,πprobe , whose decisions
are informed by the reflection on the prior failure, explor-
ing for more necessary background or relevant information
towards a full context comprehension to resolve the origi-
nal query. Self-Probe takes three inputs: (1) the ultimate goal
qinit; (2) the complete exploration probing history P(t−1)
histup
to the end of the last step; and (3) the immediate knowledge
gaps that caused the failure, concretized by all synthesized
cues of memory units generated in the prior step, denoted as
{C}(t−1). Its output P(t)is a new, strategic set of retrieving
probes for the current cycle t:
P(t)=πprobe 
qinit,P(t−1)
hist,{C}(t−1)
(1)
The Metacognitive Process. The metacognitive process
takes the new probes for this cycle P(t), and performs rea-
soning towards resolving the original query while keeping
track of the progress with the memory space. It comprises a
series of operations, described in details as follows.
Tri-Retrieve : for each probing query p∈ P(t), a retrieval
is conducted on each knowledge layer Xtypewhere type∈
{ver, sem, epi }, such that evidence of high embedding sim-
ilarity to pper layer is retrieved in a standard Dense Passage
Retrieval paradigm, with each evidence being either the raw
text chunk, a semantically clustered summary, or a narrative
flow summary.
Mem-Encode : for each pandtype, the retrieved evidence is
immediately processed by the aforementioned Mem-Encode ,
to generate a new memory unit that keeps track of how this
specific probing could complement to the final answer. The
number of all generated memory units at this step can be
denoted as |M(t)
encode|= 3× |P(t)|.
Mem-Fuse : new memory units in the above step M(t)
encodemainly emphasize aspects probed in the current cycle. To
fully utilize the past experience and historical knowledge,
the framework further identifies relevant synthesized cues
from past units in the existing memory pool Mt−1
pool, thengenerates a new synthesized cue for fusing past relevant evi-
dence. Let Mt−1
pool◦qinitrepresent past memory units whose
cues are of high embedding similarity with qinit, and denote
a LLM as Integration Agent πfuse that synthesizes these
relevant past evidence into a high-level background sum-
mary, the new cue fusing past memory C(t)
fuseis then:
C(t)
fuse=πfuse 
qinit,Mt−1
pool◦qinit
(2)
Try-Answer : with the new probing evidence in M(t)
encode
and the past-fusing cue C(t)
fuse, aQA Agent ,πQA, is applied
to these contexts to produce the cycle’s final output O(t):
O(t)=πQA 
qinit,M(t)
encode,C(t)
fuse
(3)
Specifically, a LLM is instructed to take these latest evi-
dence and the past background as the context, and determine
whether the original query can be resolved. It either yields
thefinal answer and terminates the entire reasoning loop, or
signals Failure and continues to the next step.
Mem-Update : this last step in a cycle simply incorporates
the newly generated memory units into the global pool, with
their embedding encoded, for future retrieval and reasoning:
M(t)
pool← M(t−1)
pool∪ M(t)
encode(4)
ComoRAG With the above six steps from Tri-Retrieve to
Mem-Update , one cycle of the cognitive loop is realized. For
the initial step as in t= 0, ComoRAG starts with one round
of Tri-Retrieve followed by Try-Answer . If Failure is signaled,
it initiates the Metacognitive loop of stateful reasoning on
exploratory paths, characterized by the interlocking opera-
tions with the memory workspace, which enables to tackle
complex narrative comprehension.
In essence, our framework grasps on the principle that for
long context comprehension, especially in narratives where
the entire context is cohesively interconnected through the
underlying plot progression (Xu et al. 2024a), the query res-
olution is not a linear pipeline; rather, it is a dynamic, evolv-
ing interplay between new evidence acquisition andpast
knowledge consolidation , analogous to the human cogni-
tive process. The overall process is further depicted in the
algorithm of Appendix A; detailed prompts used by each
LLM agent are provided in Appendix D.
3 Experimental Settings
Datasets Our experiments cover four long-context narra-
tive understanding datasets for comprehensive evaluation,
featuring both question answering through free generation
(QA), and multi-choice questions by selecting the best op-
tion (MC).
•NarrativeQA (Kocisk ´y et al. 2017): a QA dataset consist-
ing of books and movie scripts. For ease of computation,
we follow prior works and randomly sample 500 questions
from the test set, with average context length 58k tokens.
•EN.QA from∞BENCH (Zhang et al. 2024): a QA dataset
with 351 questions on classic novels, with average context
length over 200k tokens.

Category Method NarrativeQA EN.QA EN.MC DetectiveQA QA Avg. MC Avg.
F1 EM F1 EM ACC ACC F1 EM ACC
LLM GPT-4o-mini 27.29 7.00 29.83 12.82 30.57 30.68 28.56 9.91 30.63
Naive RAGBGE-M3(0.3B) 23.16 15.10 23.71 16.24 59.82 54.54 23.44 15.67 57.18
NV-Embed-v2 (7B) 27.18 17.80 34.34 24.57 61.13 62.50 30.76 21.19 61.82
Qwen3-Embed-8B 24.19 15.60 25.79 17.95 65.50 61.36 24.99 16.78 63.43
Enhanced RAGRAPTOR 27.84 17.80 26.33 19.65 57.21 57.95 27.09 18.73 57.58
HippoRAGv2 23.12 15.20 24.45 17.09 60.26 56.81 23.79 16.15 58.54
Multi-step
RAGSelf-RAG 19.60 6.40 12.84 4.27 59.83 52.27 16.22 5.34 56.05
MemoRAG 23.29 15.20 19.40 11.64 55.89 51.13 21.35 13.42 53.51
RAPTOR+IRCoT 31.35 16.00 32.09 19.36 63.76 64.77 31.72 17.68 64.27
HippoRAGv2+IRCoT 28.98 13.00 29.27 18.24 64.19 62.50 29.13 15.62 63.35
ComoRAG (Ours) 31.43 18.60 34.52 25.07 72.93 68.18 32.98 21.84 70.56
Table 1: Evaluation results on four long narrative comprehension datasets. For fair comparison, all methods use GPT-4o-mini
as the LLM backbone, and all non-naive RAG methods use BGE-M3 for retrieval (details in Section 3). We highlight the best
and second-best results. ComoRAG is shown consistently outperform all baselines across all datasets.
•EN.MC from∞BENCH: a MC dataset with 229 ques-
tions on classic novels of similar length as EN.QA.
•DetectiveQA (Xu et al. 2024b): a MC dataset consisting
of detective fiction with average length over 100k tokens.
We randomly sample 20% of all stories to reduce the com-
putational cost.
For evaluation metrics, we report both F1 and Exact Match
(EM) scores for QA datasets, and report Accuracy (ACC) for
MC datasets. To ensure fairness in resolving multiple-choice
questions, we only expose the options during Try-Answer ,
such that no retrieval-related actions can utilize potential
hints present in the options.
Baselines We employ four types of baselines as follows,
covering different paradigms for long context QA.
•LLM : the non-RAG setting, where the entire context
(capped by length 128k) is provided to the LLM directly.
•Naive RAG : the standard RAG setting that splits the raw
context by chunks for retrieval. We set the max chunk
length as 512 tokens in all experiments.
•Enhanced RAG : RAG methods with augmented retrieval
index, including RAPTOR (Sarthi et al. 2024) that con-
structs a semantic summary tree over text chunks, and
HippoRAGv2 (Guti ´errez et al. 2025) that builds the
knowledge base for entities in text chunks. We also ex-
perimented with GraphRAG (Edge et al. 2025); however,
it requires exponential computational cost for building the
retrieval index, being less practical for full evaluation. We
separately report GraphRAG on a subset in Appendix B.
•Multi-step RAG : RAG methods with multi-step or itera-
tive retrieval strategies. IRCoT (Trivedi et al. 2023) lever-
ages Chain-of-Thought (CoT) as intermediate queries that
iteratively retrieve evidence. Self-RAG (Asai et al. 2024)
trains a dedicated critic model to control when to stop re-
trieval. MemoRAG (Qian et al. 2025) trains a model that
compresses the global context, which generates clues as
intermediate queries.Implementation Details For the Hierarchical Knowledge
Source, we follow the procedures of HippoRAGv2 and
RAPTOR respectively to build the Veridical and Semantic
layers; the Episodic layer employs an adaptive sliding win-
dow for narrative summaries described in Appendix B.
For LLMs, our main experiments adopt GPT-4o-mini in
all approaches to ensure fair comparison. We additionally
tested GPT-4.1 and Qwen3-32B (Yang et al. 2025) for gen-
eralization analysis in Section 4.3. For all RAG methods,
we adopt the popular model BGE-M3 (Chen et al. 2024)
for retrieval. Additionally, for naive RAG, we also experi-
ment with larger but less practical embedding models, in-
cluding NV-Embed-v2 (Lee et al. 2025) and Qwen3-Embed-
8B (Zhang et al. 2025). The LLM context length for all RAG
methods, including ComoRAG, is capped at 6k tokens.
For the Metacognitive Regulation loop, we set the frame-
work to iterate for a maximum of 5 rounds.
More regarding implementation details are further pro-
vided in Appendix B.
4 Experimental Results
4.1 Main Results
Evaluation results of our main experiments are shown in
Table 1. Remarkably, ComoRAG achieves the best perfor-
mance upon all baselines across all datasets. Despite using
the lightweight 0.3B BGE-M3 for retrieval, it significantly
outperforms RAG with much larger 8B embedding models.
Overall, ComoRAG demonstrates consistent improvement
for tackling long narrative comprehension, surpassing strong
prior RAG methods of various paradigms.
Upon closer examination, ComoRAG exhibits distinct ad-
vantages on the two ∞BENCH datasets featuring ultra-long
contexts. More broadly, Figure 3 illustrates that ComoRAG
is more robust and insensitive to longer contexts, sustaining
its efficacy over HippoRAGv2, with the accuracy gap peak-
ing at +24.6% for documents exceeding 150k tokens, which

Method EN.MC EN.QA
ACC F1 EM
ComoRAG 72.93 34.52 25.07
Baselines
HippoRAGv2 60.26 24.45 17.09
RAPTOR 57.21 26.33 19.65
Index
w/o Veridical 51.97 22.24 15.88
w/o Semantic 64.63 30.82 22.65
w/o Episodic 64.63 31.48 21.47
Retrieval
w/o Metacognition 62.01 26.95 18.53
w/o Regulation 55.02 27.95 20.59
w/o Both 54.15 25.64 17.35
Table 2: Ablation studies of ComoRAG.
highlights the importance of stateful multi-step reasoning for
query resolution over long and coherent contexts.
4.2 Ablation Studies
We perform ablation studies on EN.MC and EN.QA datasets
by systematically removing key modules in ComoRAG. The
results are shown in Table 2.
Hierarchical Knowledge Source All three knowledge
layers contribute supplementary enhancements to the final
performance, with the Veridical Layer being the most sig-
nificant retrieval index. It provides the basis for factual-
grounded reasoning, as confirmed by the 30% relative per-
formance drop upon its removal.
Metacognition Removing the Metacognition process es-
sentially disables the memory workspace, where all agents
operate on retrieved evidence directly, without knowledge
consolidation by synthesized cues . Disabling this module
leads to a significant performance drop, as seen by the 22%
relative decrease in F1 score on EN.QA, and an approximate
15% decrease in accuracy on EN.MC, underscoring the crit-
ical role of dynamic memory organization.
Regulation Removing the Regulation process cuts off the
goal-oriented guidance, such that each cycle uses the same
initial query for new evidence retrieval (duplicated evidence
is removed), without generating probing queries that are
crucial to new evidence acquisition. Disabling this module
severely impacts retrieval efficiency, causing a 24% drop in
accuracy on EN.MC and a 19% drop in F1 score on EN.QA.
Notably, removing both Metacognition and Regulation
further degrades performance, effectively reducing the sys-
tem to a one-shot resolver without multi-step reasoning.
Overall, the ablation study results corroborate that the en-
hancement offered by ComoRAG stems from the synergy
between its memory consolidation and dynamic evidence
exploration, facilitated by the hierarchical knowledge index
to provide enriched semantic information. Removing any of
the core components would significantly weaken its narra-
tive reasoning capabilities.
>50 >100 >150 >200
Number of Tokens (K)505560657075808590Accuracy (%)+14.7%+24.6%74.5
68.980.8
66.9
59.864.0
56.2
54.2
ComoRAG HippoRAGv2Figure 3: Averaged accuracy across different document
lengths on Multi-Choice datasets. ComoRAG is shown more
robust to long contexts over the baseline.
Step0 Step1 Step2 Step3 Step4 Step56065707580Metrics
59.168.268.276.1
64.672.971.678.2
Step0 Step1 Step2 Step3 Step4 Step520.022.525.027.530.032.535.037.540.0
21.231.431.835.4
27.434.5
31.138.8
051015202530354045
020406080100120140160180
Failure Signal Count
DetectiveQA EN.MC NarrativeQA EN.QA
ComoRAG GPT-4.1 Metrics Failure Signal
Figure 4: Performance gains from iterative probing. GPT-4.1
marks the evaluation by using the stronger GPT-4.1 as LLM
agents in ComoRAG (as opposed to GPT-4o-mini).
4.3 In-Depth Analysis of Iterative Retrieval
To further investigate the source of ComoRAG’s effective-
ness, this section presents a quantitative analysis of its core
iterative retrieval process.
Source of Gains: From Static Bottleneck to Dynamic
Reasoning Our analysis suggests that the stateful multi-
step reasoning enabled by the Metacognitive loop is the key
factor driving the observed improvement.
We first identify a “static bottleneck”: after the initial re-
trieval using the original query at step 0, the single-step
evaluation score shows no significant advantage over strong
baselines, with less than 1% compared to the best baseline
HippoRAGv2+IRCoT. However, upon activating the cog-
nitive loop, there presents a sustained and significant im-
provement, raising the accuracy to 72.93% on EN.MC, as
shown in Figure 4. This further supports the findings from
the ablation studies, which demonstrate a significant perfor-
mance drop upon removing the entire loop. Additionally,
Figure 4 illustrates that the majority of the improvement
occurs within 2-3 cycles, confirming the efficiency of the
process. The few remaining unresolved queries are tied to
the inherent reasoning limitation of the base LLM, where
our next analysis shows that the ceiling performance of Co-
moRAG can be boosted simply by switching to more capa-
ble LLMs.
Model-agnostic Generalization ComoRAG demon-
strates generalization with different LLM backbones, with
stronger LLMs further enhancing the reasoning process and
final query resolution. To validate this, we replace GPT-4o-
mini with GPT-4.1 and Qwen3-32B in the Metacognitive
loop, using the same knowledge source for retrieval. The

Method NarQA EN.QA EN.MC DetQA
F1 F1 ACC ACC
ComoRAG 31.43 34.52 72.93 68.18
w/ Qwen3-32B 32.17 35.29 74.24 69.32
w/ GPT-4.1 35.43 38.82 78.17 76.14
HippoRAGv2 23.12 24.45 60.26 56.81
+ Our Loop 29.12 31.76 68.56 63.64
RAPTOR 27.84 26.33 57.21 57.95
+ Our Loop 30.55 34.31 69.00 62.50
Table 3: Efficacy of ComoRAG on model-agnostic general-
ization and Plug-and-Play flexibility.
results, presented in Figure 4 and the upper section of
Table 3, show a notable improvement particularly with
GPT-4.1, boosting the F1 score on EN.QA from 34.52 to
38.82, and increases the accuracy on EN.MC from 72.93 to
78.17. These results demonstrate that ComoRAG effectively
leverages and unleashes the model’s capabilities during its
stateful iterative reasoning process.
Plug-and-Play: Flexibility To examine the modularity of
our framework, we conduct further experiments by apply-
ing the Metacognitive loop of ComoRAG on existing RAG
methods. As shown in the bottom section of Table 3, the cog-
nitive loop can be seamlessly integrated with different RAG
index including HippoRAGv2 and RAPTOR. This integra-
tion consistently results in significant performance improve-
ments across all benchmarks, with accuracy on EN.MC in-
creasing by over 8% for HippoRAGv2 and nearly 12% for
RAPTOR (a similar trend is observed on EN.QA). These
results demonstrate that ComoRAG could serve as a robust
and flexible plug-and-play solution to enhance query resolu-
tion of existing RAG methods.
4.4 In-Depth Analysis of Query Resolution
To deepen the understanding of narrative query resolution,
we roughly categorize all questions in our experimented
datasets into three query types: factoid ,narrative , and in-
ferential , described as follows(details in Appendix C).
•Factoid Queries : queries answerable by a single, spe-
cific piece of information, often knowledge-seeking, e.g.,
“What religion is Octavio Amber?”
•Narrative Queries : queries that require an understanding
of plot progression as a coherent background context, e.g.,
“Where does Trace choose to live at the end of the novel?”
•Inferential Queries : queries demanding reasoning be-
yond the literal text to understand implicit motivations,
e.g., “What is the main reason that Nils first visits Aiden
in his apartment?”
To systematically investigate the dynamics of ComoRAG
reasoning, we first pose the question: what is the bottle-
neck in long-narrative reasoning for existing RAG meth-
ods? Figure 5 pictures a clear diagnosis. While one-shot re-
trieval suffices for factoid queries, which account for over
60% of initial solution, our iterative cognitive loop is essen-
tial for resolving complex narrative queries involving global
Step0 Step1-5 Unresolved
Processing Stages01020304050607080Percentage(%)68.5
41.7
20.0
13.650.0
10.017.9
8.370.0Factoid Narrative InferentialFigure 5: Distribution of solved question types by processing
stage.
60.8%
0.3540.3040.18067.9%
65.2%77.1%
76.1%
58.8%
0.3630.3630.223Factoid
MC
Narrative
MC
Inferential
MC
Factoid
QANarrative
QAInferential
QA
HippoRAGv2-IRCOT RAPTOR-IRCOT
HippoRAGv2 RAPTOR Ours
Figure 6: Benchmarking RAG methods across query types.
context comprehension and deeper reasoning. These consti-
tute nearly 50% of the problems that are solved exclusively
through the Metacognitive loop.
This leads to the second question: how does our frame-
work’s performance on this specific bottleneck com-
pared to strong baselines? Figure 6 demonstrates that our
method’s advantage is most pronounced precisely in this
area. On narrative queries, ComoRAG substantially out-
performs the strongest baselines, achieving a 19% relative
F1 improvement on EN.QA and a 16% accuracy gain on
EN.MC. By addressing these queries, we demonstrate that
the success of our framework is not merely a general im-
provement, but a targeted and effective solution to the nar-
rative query type - a cornerstone to achieve true narrative
comprehension that has posed challenges for prior retrieval-
based narrative reasoning approaches.
Qualitatively, Figure 2 illustrates the dynamic reasoning
mechanism with the query qinit:“Mrs. MacIntyre never
writes letters, so what is the sudden purpose of buying ink?”
A standard, single-step retrieval would fail on this query, as
it would only find a vague clue about a “cut out newspa-
per,” which is insufficient to form an answer. In contrast, Co-
moRAG initiates an iterative reasoning process by dynami-
cally probing new queries and evidence towards full resolu-
tion, thereby constructing a complete evidence chain to de-
duce the final answer : Mrs. McGinty recognized a photo,
wanted to sell the story, and intended to write to the news-
paper. We provide full reasoning details in Appendix E.
5 Conclusion
In this work, we propose ComoRAG for long narrative rea-
soning, aiming to address the “stateless” limitation of con-
ventional RAG. ComoRAG is especially inspired by the hu-

man brain’s Prefrontal Cortex; through a dynamic mem-
ory workspace and iterative probes, it fuses fragmented evi-
dence into a coherent context to achieve stateful reasoning
over narrative progression. Experiments validate that Co-
moRAG overcomes the bottleneck of existing methods by
excelling at complex narrative and inferential queries, mark-
ing a paradigm shift from information retrieval to cognitive
reasoning towards deeper long context comprehension.
References
Asai, A.; Wu, Z.; Wang, Y .; Sil, A.; and Hajishirzi, H.
2024. Self-RAG: Learning to Retrieve, Generate, and Cri-
tique through Self-Reflection. In The Twelfth International
Conference on Learning Representations .
Chen, J.; Xiao, S.; Zhang, P.; Luo, K.; Lian, D.; and
Liu, Z. 2024. M3-Embedding: Multi-Linguality, Multi-
Functionality, Multi-Granularity Text Embeddings Through
Self-Knowledge Distillation. In Ku, L.-W.; Martins, A.; and
Srikumar, V ., eds., Findings of the Association for Compu-
tational Linguistics: ACL 2024 , 2318–2335. Bangkok, Thai-
land: Association for Computational Linguistics.
Dobbins, I. G.; and Han, S. 2006. Cue- versus Probe-
dependent Prefrontal Cortex Activity during Contextual Re-
membering. Journal of Cognitive Neuroscience , 18(9):
1439–1452.
Edge, D.; Trinh, H.; Cheng, N.; Bradley, J.; Chao, A.; Mody,
A.; Truitt, S.; Metropolitansky, D.; Ness, R. O.; and Larson,
J. 2025. From Local to Global: A Graph RAG Approach to
Query-Focused Summarization. arXiv:2404.16130.
Eisenschlos, J. M.; Yogatama, D.; and Al-Rfou, R. 2023.
Needle In A Haystack: Where Is It? Finding Factual Asso-
ciations in Long Texts. arXiv preprint arXiv:2307.09288 .
Fernandez-Duque, D.; Baird, J. A.; and Posner, M. I. 2000.
Executive Attention and Metacognitive Regulation. Con-
sciousness and Cognition , 9(2): 288–307.
Guti´errez, B. J.; Shu, Y .; Qi, W.; Zhou, S.; and Su, Y . 2025.
From RAG to Memory: Non-Parametric Continual Learning
for Large Language Models. In Forty-second International
Conference on Machine Learning .
Jimenez Gutierrez, B.; Shu, Y .; Gu, Y .; Yasunaga, M.; and
Su, Y . 2024. Hipporag: Neurobiologically inspired long-
term memory for large language models. Advances in Neu-
ral Information Processing Systems , 37: 59532–59569.
Johnson-Laird, P. N. 1983. Mental Models: Towards a Cog-
nitive Science of Language, Inference, and Consciousness .
Cambridge, MA: Harvard University Press.
Kocisk ´y, T.; Schwarz, J.; Blunsom, P.; Dyer, C.; Hermann,
K. M.; Melis, G.; and Grefenstette, E. 2017. The Narra-
tiveQA Reading Comprehension Challenge. Transactions of
the Association for Computational Linguistics , 6: 317–328.
Lee, C.; Roy, R.; Xu, M.; Raiman, J.; Shoeybi, M.; Catan-
zaro, B.; and Ping, W. 2025. NV-Embed: Improved Tech-
niques for Training LLMs as Generalist Embedding Mod-
els. In The Thirteenth International Conference on Learning
Representations .Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V .;
Goyal, N.; K ¨uttler, H.; Lewis, M.; Yih, W.-t.; Rockt ¨aschel,
T.; Riedel, S.; and Kiela, D. 2020. Retrieval-Augmented
Generation for Knowledge-Intensive NLP Tasks. In
Larochelle, H.; Ranzato, M.; Hadsell, R.; Balcan, M.; and
Lin, H., eds., Advances in Neural Information Processing
Systems , volume 33, 9459–9474. Curran Associates, Inc.
Liu, N. F.; Lin, K.; Hewitt, J.; Paranjape, A.; Bevilacqua,
M.; Petroni, F.; and Liang, P. 2024. Lost in the Middle: How
Language Models Use Long Contexts. Transactions of the
Association for Computational Linguistics , 12: 157–173.
Miller, J. A.; and Constantinidis, C. 2024. Timescales of
learning in prefrontal cortex. Nature Reviews Neuroscience ,
25(9): 597–610.
Qian, H.; Liu, Z.; Zhang, P.; Mao, K.; Lian, D.; Dou, Z.; and
Huang, T. 2025. Memorag: Boosting long context process-
ing with global memory-enhanced retrieval augmentation.
InProceedings of the ACM on Web Conference 2025 , 2366–
2377.
Sarthi, P.; Abdullah, S.; Tuli, A.; Khanna, S.; Goldie, A.;
and Manning, C. 2024. RAPTOR: Recursive Abstractive
Processing for Tree-Organized Retrieval. In International
Conference on Learning Representations (ICLR) .
Trivedi, H.; Balasubramanian, N.; Khot, T.; and Sabharwal,
A. 2023. Interleaving Retrieval with Chain-of-Thought Rea-
soning for Knowledge-Intensive Multi-Step Questions. In
Rogers, A.; Boyd-Graber, J.; and Okazaki, N., eds., Pro-
ceedings of the 61st Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers) ,
10014–10037. Toronto, Canada: Association for Computa-
tional Linguistics.
Wei, J.; Wang, X.; Schuurmans, D.; Bosma, M.; brian ichter;
Xia, F.; Chi, E. H.; Le, Q. V .; and Zhou, D. 2022. Chain
of Thought Prompting Elicits Reasoning in Large Language
Models. In Oh, A. H.; Agarwal, A.; Belgrave, D.; and Cho,
K., eds., Advances in Neural Information Processing Sys-
tems.
Xu, L.; Li, J.; Yu, M.; and Zhou, J. 2024a. Fine-Grained
Modeling of Narrative Context: A Coherence Perspective
via Retrospective Questions. In Ku, L.-W.; Martins, A.; and
Srikumar, V ., eds., Proceedings of the 62nd Annual Meeting
of the Association for Computational Linguistics (Volume 1:
Long Papers) , 5822–5838. Bangkok, Thailand: Association
for Computational Linguistics.
Xu, Z.; Ye, J.; Liu, X.; Sun, T.; Liu, X.; Guo, Q.; Li, L.; Liu,
Q.; Huang, X.; and Qiu, X. 2024b. DetectiveQA: Evaluat-
ing Long-Context Reasoning on Detective Novels. ArXiv ,
abs/2409.02465.
Yang, A.; Li, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.;
Yu, B.; Gao, C.; Huang, C.; Lv, C.; et al. 2025. Qwen3
technical report. arXiv preprint arXiv:2505.09388 .
Yang, Z.; Qi, P.; Zhang, S.; Bengio, Y .; Cohen, W.; Salakhut-
dinov, R.; and Manning, C. D. 2018. HotpotQA: A Dataset
for Diverse, Explainable Multi-hop Question Answering. In
Riloff, E.; Chiang, D.; Hockenmaier, J.; and Tsujii, J., eds.,
Proceedings of the 2018 Conference on Empirical Methods

in Natural Language Processing , 2369–2380. Brussels, Bel-
gium: Association for Computational Linguistics.
Zhang, X.; Chen, Y .; Hu, S.; Xu, Z.; Chen, J.; Hao, M.; Han,
X.; Thai, Z.; Wang, S.; Liu, Z.; et al. 2024. ∞Bench: Ex-
tending long context evaluation beyond 100k tokens. In
Proceedings of the 62nd Annual Meeting of the Associa-
tion for Computational Linguistics (Volume 1: Long Papers) ,
15262–15277.
Zhang, Y .; Li, M.; Long, D.; Zhang, X.; Lin, H.; Yang, B.;
Xie, P.; Yang, A.; Liu, D.; Lin, J.; Huang, F.; and Zhou, J.
2025. Qwen3 Embedding: Advancing Text Embedding and
Reranking Through Foundation Models. arXiv:2506.05176.

A ComoRAG Algorithm
Algorithm 1: ComoRAG (Described in Section 2)
Require: Initial Query qinit, Knowledge Source X, Max
Iterations T
Ensure: The final answer Oor a failure signal
1:function COMO RAG( qinit,X, T)
2: M(0)
pool,P(0)
hist,{C}(0)← ∅,∅,∅▷Initialize Memory
Pool, Probing History, and Synthesized Cues
3: E(0)← Tri-Retrieve ({qinit},X)
4: O(0)← Try-Answer (qinit,E(0))
5: ifO(0)̸=FailureSignal then
6: return O(0)▷Return immediately if successful
7: end if
▷Triggered only if initial attempt fails
8: M(0)
encode← Mem-Encode (qinit,P(0)
hist,E(0))
9: M(0)
pool← Mem-Update (M(0)
pool,M(0)
encode)
10: P(0)
hist←qinit
11: {C}(0)← M(0)
pool
12: fort= 1, . . . , T do
13: P(t)← Self-Probe (qinit,P(t−1)
hist,{C}(t−1))
14: E(t)← Tri-Retrieve (P(t),X)
15: M(t)
encode← Mem-Encode (qinit,P(t),E(t))
16: C(t)
fuse← Mem-Fuse (qinit,M(t−1)
pool◦qinit
17: O(t)← Try-Answer (qinit,M(t)
encode,C(t)
fuse)
18: ifO(t)̸=FailureSignal then return O(t)
19: end if
20: M(t)
pool← Mem-Update (M(t−1)
pool,M(t)
encode)
21: P(t)
hist← P(t−1)
hist∪ P(t)
22: {C}(t)← M(t)
pool
23: end for
24: return FailureSignal
25:end function
B Implementation Details
B.1 Veridical Layer
As described in Section 2.2, ComoRAG empowers Large
Language Models by constructing a hierarchical knowledge
source, whereby the Veridical Layer is a foundational com-
ponent comprising text chunks of the original context. We
largely follow the construction process of HippoRAGv2
(Guti ´errez et al. 2025) to add a mapping between knowl-
edge graphs (KGs) and text chunks to facilitate retrieval. To
construct the KG, a Large Language Model (LLM) is lever-
aged to extract (subject-predicate-object) knowledge triples.
These triples from a document are then aggregated to form a
unified knowledge graph. Finally, a retrieval-optimized en-
coder adds supplementary edges to this graph by identifying
and linking semantically similar entities (synonyms). The
retrieval of the Veridical Layer thus follows HippoRAGv2
to utilize KGs towards more accurate retrieval. Statistics for
this layer are detailed in Table 4.Layer Count NarQA EN.QA EN.MC DetQA
Veridical# of Chunks 4446 26 465 47 074 2406
# of Entities 33 810 292 170 401 040 30 969
# of Triples 51 012 372 339 576 595 33 696
Table 4: Statistics of the Veridical Layer across Datasets.
B.2 Episodic Layer
To construct the Episodic Layer, a sequence of text chunks
is summarized. Since the context lengths can vary signif-
icantly, the choice of a sliding window size for this sum-
marization presents a trade-off: a large window can be too
coarse for short narratives, while a small window may be in-
efficient and fail to capture long-range dependencies in long-
form content. Therefore, we dynamically adjust the window
size W according to the total number of text chunks, N, in
the document. The specific heuristic is as follows.
• For short to medium-length narratives ( N≤200
chunks): stepped window sizes (3, 5, 8, and 10) are used
for documents up to 20, 50, 100, and 200 chunks respec-
tively, aiming to preserve details for shorter contexts.
• For long narratives ( N > 200): A logarithmic scaling
function is applied to prevent the window from becom-
ing excessively large. This sub-linear growth is intended
to increase the summary scope for massive texts more
slowly. The window size is calculated as follows to keep
the window size between 10 to 20:
W= min(20 ,max(10 ,⌊log2(N)×2⌋))
For each window, the contained text chunks are concate-
nated and provided to an LLM agent (GPT-4o-mini in our
experiments). The agent is instructed to generate a con-
cise summary that maintains chronological order and identi-
fies key events and causal relationships. The resulting sum-
maries are then collected and sorted by their original win-
dow order to form the nodes of the Episodic Layer.
B.3 GraphRAG Experiments
GraphRAG is a structured-augmented RAG method similar
to HippoRAGv2, which involves the construction of a com-
prehensive knowledge graph from source documents, which
is then used to identify interconnected information for re-
trieval. However, its formulation requires heavy computa-
tion for building the retrieval index that includes multi-level
node relations and summaries.
We conducted preliminary experiments on a data subset to
evaluate its viability. The results, detailed in Table 5, demon-
strated that GraphRAG not only had significantly higher to-
ken consumption, but also attained lower scores compared to
other baselines adopted in our experiments. Considering the
trade-offs between its computational cost and performance,
we ultimately did not include GraphRAG as a primary base-
line for a full-scale evaluation.
B.4 Hyperparameters for ComoRAG
The key hyperparameters for our ComoRAG framework are
detailed in Table 6. All cognitive agents employ GPT-4o-

ComoRAG GraphRAG
Performance Metrics
F1 Score 33.61 (100.0%) 14.20 (42.3%)
EM Score 21.43 (100.0%) 8.00 (37.3%)
Token Usage
Tokens 5.90M (100.0%) 27.12M (459.7%)
Average Time Taken (sec)
Index 291 (100.0%) 1936 (665.3%)
Retrieve 25 (100.0%) 29 (116.0%)
Table 5: Comparison of Performance, Token Usage, and Av-
erage Time for ComoRAG and GraphRAG.
mini, with retrieval powered by the widely-used BGE-M3
embedding model. For retrieval settings, The dynamic cog-
nitive loop is configured to run for a maximum of 5 iter-
ations, generating up to 3 new probing queries per cycle.
The context for QA is capped at 6k tokens, in consistent
with all RAG baselines in our experiments. This context is
assembled via a proportional 8:2:2:1 allocation of evidence
from the Veridical, Semantic, Episodic, and fused Histori-
cal memory, respectively. The “Mem-Fuse Threshold” is set
to 0.5, indicating the proportion of evidences retrieved from
the memory pool that are forwarded to the Integration Agent
for memory fusion and summary generation.
Hyperparameter Value
LLM Agents ( πprobe , etc.) GPT-4o-mini
Retrieval Model BGE-M3
Chunk Size 512 tokens
Context Length 6,000 tokens
Random Seed 0
Max Iterations 5
Max Probing Queries 3
Context Construction Proportional Allocation (8:2:2:1 ra-
tio for V:S:E:H)
Mem-Fuse Threshold 0.5
Table 6: Hyperparameter settings for ComoRAG in our ex-
periments. V , S, E, H refer to Veridical, Semantic, Episodic,
and Historical evidence.
C Query Types for Narratives
Dataset Factoid Narrative Inferential Total
EN.QA 224 84 43 351
EN.MC 132 46 51 229
Table 7: Distribution of query types across the two datasets.
To facilitate a fine-grained analysis of our model’s per-
formance, we (authors of this work) manually annotated the
types of all questions in the EN.QA and EN.MC datasets.Each question is classified into one of the three categories
based on the cognitive processes required to answer it, de-
scribed in Section 4.4:
•Factoid : questions answerable by locating a single, spe-
cific piece of information from the text.
•Narrative : questions that demand an understanding of
plot progression, requiring the aggregation of informa-
tion from multiple text parts.
•Inferential : questions that necessitate reasoning beyond
the literal text to understand implicit motivations or
causal links.
The final distribution of the annotated query types is pre-
sented in Table 7.

D Prompting Templates
Self-Probe Instruction Template for Probing Query Generation in Regulation Agent
Role:
You are an expert in multi-turn retrieval-oriented probe generation. Your job is to extract diverse and complementary
retrieval probes from queries to broaden and enrich subsequent corpus search results.
Input Materials:
•Original Query : A question or information need that requires comprehensive information retrieval.
•Context : Available background information, partial content, or relevant summaries.
•Previous Probes : Previously generated probes from earlier iterations (if any).
Task:
Based on the query and context, generate up to 3 non-overlapping retrieval probes that explore the query from distinct
angles.
Critical Requirements:
•Semantic Differentiation : Ensure new probes are semantically distinct from any previous probes provided.
•Complementary Coverage : New probes should cover different information dimensions not addressed by previous
probes.
•Relevance Maintenance : All probes must remain directly relevant to answering the original query.
Each probe should:
• Target different information dimensions relevant to the query type:
–Character-related : actions, motivations, relationships, timeline, consequences
–Event-related : participants, causes, sequence, location, outcomes
–Object-related : description, origin, usage, significance, connections
–Location-related : events occurred, people involved, time periods, significance
• Expand search scope beyond obvious keywords to capture related content.
• Avoid semantic overlap with previous probes while maintaining query relevance.
• Be formulated as effective search terms or phrases.
Probe Generation Strategy:
•When previous probes exist:
1. Analyze Previous Coverage: Identify what semantic domains/angles have been covered.
2. Gap Identification: Find unexplored but relevant information dimensions.
3. Alternative Angles: Generate probes from different conceptual perspectives.
4. Semantic Distance: Ensure sufficient semantic distance from previous probes.
•When no previous probes exist:
–Probe 1: Direct elements explicitly mentioned in the query.
–Probe 2: Contextual elements that might contain the answer.
–Probe 3: Related concepts or alternative formulations.
Output Format:
‘‘‘json
{
"probe1": "Content of probe 1",
...
}
‘‘‘
Notes:
• For simple queries, you may generate only 1–2 probes.
• If previous probes have covered most relevant angles, generate fewer new probes to avoid redundancy.
• Prioritize quality and semantic distinctiveness over quantity.

Mem-Encode Instruction Template for Synthesized Cue Generation in Comprehension Agent
Role
You are an expert narrative analyst capable of identifying, extracting, and analyzing key information from narrative
texts to provide accurate and targeted answers to specific questions.
Material
You are given the following:
1. A final objective to be resolved
2. A specific question that needs to be answered
3. Content: Direct excerpts, facts, and specific information from the narrative text
Task
1. Carefully analyze the question to identify:
• What type of information is being asked (character actions, locations, objects, events, motivations, etc.)
• Which narrative elements are relevant to answering it
• The specific details that need to be extracted
2. Systematically scan the content for:
• Direct mentions of relevant elements (names, places, objects, events)
• Contextual probes that help answer the question
• Temporal and spatial relationships
• Cause-and-effect connections
3. Analyze the identified information considering:
• Explicit statements (directly stated facts)
• Implicit information (suggested through context, dialogue, or narrative)
• Logical connections between different narrative elements
• Chronological sequence of events if relevant
4. Synthesize findings to construct a precise answer to the question.
Response Format
Provide a structured analysis with up to 5 key findings:
‘‘‘
Key Finding :<Most directly relevant information answering the question >
Key Finding :<Supporting evidence or context >
Key Finding :<Additional relevant details >
Key Finding :<Clarifying information if needed >
Key Finding :<Resolution of any ambiguities >
‘‘‘
Mem-Fuse Instruction Template for Cue Generation in Integration Agent
Role:
You are an expert narrative synthesis specialist who excels at integrating and analyzing information from multiple
narrative sources to create coherent and comprehensive insights.
Input Material:
•Previous Analysis: Results from earlier memory fusion operations that contain analyzed narrative information.
•Current Query: A question or information request that needs to be addressed.
Task:
1.Review and understand the previous memory fusion outputs:
• Identify key narrative elements and their relationships.

• Note any established facts, character developments, or plot points.
• Recognize patterns and connections across different analyses.
2.Analyze the current query in context:
• Determine how it relates to previously established information.
• Identify any new aspects or angles that need to be addressed.
• Consider how previous insights can inform the current response.
3.Synthesize the information:
• Integrate relevant previous findings with new analysis.
• Create a coherent narrative that addresses the current query.
• Ensure continuity and consistency with previous analyses.
• Highlight any new insights or developments.
4.Provide a comprehensive response that:
• Directly answers the current query.
• Incorporates relevant previous context.
• Maintains narrative coherence.
• Offers clear and insightful analysis.
Response Format:
Provide a cohesive narrative response that integrates previous insights with new analysis to address the current query.
Focus on creating a flowing, well-structured response.
Try-Answer Prompt Template for Query Resolution in QA Agent
Role:
You are an expert on reading and understanding books and articles.
Task:
Given the following detailed article, semantic summary, Episodic summary from a book, and a related question with
different options, you need to analyze which option is the best answer for the question.
Inputs:
•Detail Article: {context }
•Summary by Semantic: {semantic summary }
•Summary by Episodic: {Episodic summary }
•History Info: {history info}
•Question: {question }
Limits:
• Do not infer. Respond only based on the provided content strictly.
• Pick the choice only if you find at least 2 places that support the answer.
Response Format:
1.Content Understanding :
Start with a brief summary of the content in no more than three sentences. Begin this section with ### Content
Understanding .
2.Question Analysis :
Based on the question, analyze and list all relevant items using a markdown list. Begin this section with ###
Question Analyse .
3.Options Analysis :
Extract the key points related to 4 options, also using a markdown list. Begin this section with ### Options
Analyse .
Note: Only analyze based on the provided materials, do not make guesses.
4.Final Answer :
Provide your final answer with a heading. Begin this section with ### Final Answer , followed by the best
option in the format of [A] or[B] or[C] or[D]. If you cannot answer, give a failure signal: *.

E Case Study on Narrative Reasoning
Input Data (No Options)
Query: Mrs. MacIntyre never writes letters, so what is the sudden purpose of buying ink?
Options: [A] Response to the government’s letter [B] Sending a birthday card to my niece [C] Write a letter to the Sunday Comet
newspaper. [D] Blurring Photos
ComoRAG’s Choice Result
Memory Pool M(0)
pool:
- A page of the newspaper hadaportioncutout...
Step1
Probes P(1):
- What did Mrs. McGinty recognize in the clipping, and how did she try to profit from it?
...
Retrieved Passages:
...The narrative offers insight into Miss Pamela Hosford’s role at the Sunday Comet, as she casually recalls receiving a vague letter from a
woman about a photograph but fails to retrieve it...
CuesC(1):
-Key Finding:Mrs. McGinty usuallyhadJoehelp herreplytoletters.;
-Key Finding:Mrs. McGinty may have sentalettertothenews paper.;...
Memory Pool M(1)
pool:
- A page of the newspaper hadaportioncutout...
- Mrs. MacIntyre sent a letter to theSundayComet...
Step2
Probes P(2):
- What did Miss Pamela Hosford recall about a letter sent to the Sunday Comet, and what might it imply about Mrs. McGinty?
...
Retrieved assages:
...Miss Pamela Hosford’s role at the Sunday Comet, as she casually recalls receivingavague letterfrom awoman about aphotograph but
fails to retrieve it...She liked eavesdropping, knew who was in the photo, and asked what we‘d pay...
CuesC(2):
-Key Finding:Mrs. McGinty wanted tomake some money from thephoto.;...
Chosen: C. (Correct)
(C) Write a letter to the Sunday Comet newspaper: Strong textual probes support this option. Mrs. McGinty cut out a part of the
newspaper, recognized someone in a photo, asked about payment, and unusually bought ink—suggesting she intended to write to the
paper. Final Answer: [C]
Table 8: Case Study on Narrative Reasoning. We present a case to demonstrate our model’s performance in long-context
understanding, showing the final round of the Metacognitive Control Loop. Different colors are used to highlight the nature of
the processed information: Blue is used for the key evidence that contributes to the correct answer, while Orange is used for the
key cues.