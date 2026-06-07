# IA-RAG: Interval-Algebra-Driven Temporal Reasoning for Dynamic Knowledge Retrieval

**Authors**: Xiaoman Wang, Yaoze Zhang, Wenzhuo Fan, Hongwei Zhang, Ding Wang, Guohang Yan, Song Mao, Botian Shi, Yunshi Lan, Pinlong Cai

**Published**: 2026-06-04 11:39:50

**PDF URL**: [https://arxiv.org/pdf/2606.06044v1](https://arxiv.org/pdf/2606.06044v1)

## Abstract
Retrieval-Augmented Generation (RAG) has shown strong effectiveness in grounding Large Language Models (LLMs) with external knowledge. However, existing RAG and Graph RAG frameworks largely treat knowledge as static or associate time with coarse-grained timestamps or metadata, failing to capture rich temporal structures such as duration, overlap, and containment. We propose IA-RAG, a hierarchical temporal RAG framework that models knowledge as time intervals and performs retrieval under formal temporal constraints. IA-RAG represents facts as Interval Event Units (IEUs) and organizes them into a hierarchical Thematic Forest, where temporal dependencies are governed by Allen's Interval Algebra. To handle incomplete or uncertain temporal boundaries, IA-RAG further introduces a Sub-graph Time Tightening mechanism that refines fuzzy intervals through logical constraints within connected event subgraphs. In addition, IA-RAG supports implicit temporal semantic retrieval through interval-algebra-guided traversal. Experiments on multiple temporal question answering benchmarks, including TimeQA, TempReason, and ComplexTR, demonstrate that IA-RAG achieves strong temporal retrieval and reasoning performance, particularly on complex compositional temporal reasoning tasks. Our code is released at https://github.com/xiaoAugenstern/LogicalRAG_TemporalQA.

## Full Text


<!-- PDF content starts -->

IA-RAG: Interval-Algebra–Driven Temporal Reasoning for Dynamic
Knowledge Retrieval
Xiaoman Wang1*, Yaoze Zhang2,3*, Wenzhuo Fan2,4, Hongwei Zhang1,2,
Ding Wang2,Guohang Yan2,Song Mao2,Botian Shi2,Yunshi Lan1†,Pinlong Cai2†
1East China Normal University,2Shanghai Artificial Intelligence Laboratory,
3University of Shanghai for Science and Technology,4Harbin Engineering University
xmwang@stu.ecnu.edu.cn,razzzhang@gmail.com,
yslan@dase.ecnu.edu.cn,caipinlong@pjlab.org.cn
Abstract
Retrieval-Augmented Generation (RAG) has
shown strong effectiveness in grounding Large
Language Models (LLMs) with external knowl-
edge. However, existing RAG and Graph RAG
frameworks largely treat knowledge as static
or associate time with coarse-grained times-
tamps or metadata, failing to capture rich tem-
poral structures such as duration, overlap, and
containment. We proposeIA-RAG, a hierar-
chical temporal RAG framework that models
knowledge as time intervals and performs re-
trieval under formal temporal constraints. IA-
RAG represents facts asInterval Event Units
(IEUs)and organizes them into a hierarchical
Thematic Forest, where temporal dependencies
are governed by Allen’s Interval Algebra. To
handle incomplete or uncertain temporal bound-
aries, IA-RAG further introduces aSub-graph
Time Tighteningmechanism that refines fuzzy
intervals through logical constraints within con-
nected event subgraphs. In addition, IA-RAG
supports implicit temporal semantic retrieval
through interval-algebra-guided traversal. Ex-
periments on multiple temporal question an-
swering benchmarks, including TimeQA, Tem-
pReason, and ComplexTR, demonstrate that
IA-RAG achieves strong temporal retrieval and
reasoning performance, particularly on com-
plex compositional temporal reasoning tasks.
Our code is released at https://github.com/
xiaoAugenstern/LogicalRAG_TemporalQA.
1 Introduction
Large Language Models (LLMs) have achieved
remarkable progress on knowledge-intensive
tasks (Naveed et al., 2025). However, they remain
limited by outdated parametric knowledge and
factual inconsistency. Retrieval-Augmented Gen-
eration (RAG) mitigates these issues by ground-
ing generation in external evidence (Lewis et al.,
*Equal contribution.
†Corresponding author.2020; Zhang et al., 2026a). More recently, Graph
RAG methods extend this paradigm by model-
ing relational structures through entity-relation
graphs (Edge et al., 2024; Guo et al., 2024).
Despite these advances, existing RAG frame-
works still lack an expressive mechanism for tempo-
ral reasoning. Recent temporal-aware approaches
mainly incorporate timestamps through metadata
filtering (Li et al., 2025), temporal anchoring (Sun
et al., 2025), or chronological organization (Zhang
et al., 2026b). However, real-world facts are not
isolated time points; they are intervals with dura-
tion, overlap, and containment. Treating time as a
shallow attribute leads to three major limitations:
(1)Temporal Boundary Ambiguity: Semantic re-
trievers struggle to distinguish between states of
the same subject across different periods. For in-
stance, queries regarding “policiesbefore2008"
and “policiesafter2008" yield nearly identical
embeddings, causing the retrieval of temporally
irrelevant facts. (2)Inability to Refine Implicit
Temporal Attributes: Many real-world facts lack
explicit timestamps. Current methods fail to utilize
the complementary relationships between events
to infer or impute these unknown temporal inter-
vals, leaving a significant portion of the knowl-
edge base temporally unanchored and unsearch-
able. (3)Topological Reasoning Deficiency: Com-
plex queries often require inferring relationships
between multiple events, such as determining if one
event occurred during another or if they overlap.
Existing RAG systems return disjoint chunks with-
out modeling these interval-based dependencies,
forcing LLMs to rely on brittle linguistic inference
rather than structured temporal logic.
To bridge these gaps, we present IA-RAG, a hi-
erarchical temporal RAG framework that leverages
Allen’s Interval Algebra (IA) to rigorously model
and reason over the topological structure of events.
To systematically address the limitations of point-
based approaches, our framework introduces three
1arXiv:2606.06044v1  [cs.CL]  4 Jun 2026

key innovations aligned with the identified chal-
lenges. First, to overcome topological reasoning
deficiencies in existing event representations, we
propose theInterval Event Unit (IEU)and the
Thematic Forest. Unlike previous graphs that rely
on shallow time-proximity links, our structure or-
ganizes atomic IEUs into a hierarchy where edges
are governed by the 13 fundamental IA relations
(e.g.,meets,overlaps). Second, to tackle the inabil-
ity to refine implicit temporal attributes, we intro-
duce aSub-graph Time Tightening mechanism.
By exploiting the logical constraints within con-
nected subgraphs, this mechanism infers missing
timestamps and refines fuzzy boundaries based on
the precise temporal context of neighboring events.
Finally, to overcome temporal boundary ambigu-
ity, IA-RAG employs anInterval-Algebra–guided
traversal strategy. Instead of relying solely on
semantic similarity, this approach performs direc-
tional retrieval anchored to specific event states,
ensuring that queries are resolved with strict adher-
ence to their temporal logic constraints.
Our contributions are summarized as follows:
•We introduce an interval-based temporal repre-
sentation grounded in Allen’s Interval Algebra,
which is the first to formally model event dura-
tion, topology, and temporal relations in retrieval-
augmented generation, thereby enabling reason-
ing over intervals, in contrast to existing RAG
methods that model time as discrete, unordered
point timestamps.
•We propose IA-RAG, a hierarchical temporal
GraphRAG framework that integrates logical
temporal connectivity, ambiguity-aware bound-
ary refinement, and constraint-guided retrieval.
This enables accurate answering of implicit tem-
poral state queries and interval-level reasoning
tasks that are fundamentally inaccessible to con-
ventional timestamp-based RAG systems.
•Extensive experiments on multiple temporal QA
benchmarks, including TimeQA, TempReason,
and ComplexTR, show that IA-RAG consistently
delivers strong temporal retrieval and reasoning
performance, especially on complex composi-
tional temporal reasoning tasks.
2 Related Work
2.1 Knowledge Graph Based
Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) improves
the factual reliability of Large Language Mod-els (LLMs) by grounding generation in external
evidence sources (Lewis et al., 2020; Yan et al.,
2025). To overcome the limitations of flat dense
retrieval for multi-hop reasoning, graph-based
RAG frameworks explicitly model relational struc-
tures among knowledge units. GraphRAG (Edge
et al., 2024) constructs entity-centric graphs for
structured traversal, while HippoRAG (Gutiérrez
et al., 2024) improves long-range relational re-
call through neuro-inspired memory mechanisms.
Subsequent works such as LightRAG (Guo et al.,
2024) and MemoRAG (Qian et al., 2025) further en-
hance retrieval efficiency via hierarchical indexing
and memory compression. Event-centric designs
like EventRAG (Yang et al., 2025) extract events
to capture logico-deductive narratives, and neuro-
symbolic extensions like NeuSTIP (Singh et al.,
2023) couple temporal graph completion with path-
mining rules. Despite their advancements, these
graph-based systems focus predominantly on static
semantic relations. They relegate temporal dimen-
sions to isolated node attributes or unlinked meta-
data, making them inherently incapable of captur-
ing the topological variations of evolving facts. In
contrast, IA-RAG introduces interval-aware event
graphs and hierarchical thematic forests to support
structured temporal retrieval.
2.2 Temporal-Aware RAG
Recent studies have explored incorporating tem-
poral signals into the RAG pipeline for reason-
ing over evolving knowledge and time-sensitive
facts (Piryani et al., 2025). TA-RAG (Lau et al.,
2025) addresses diachronic question answering
by retrieving temporally coherent evidence across
queried periods, while T-GRAG (Li et al., 2025)
organizes historical knowledge into timestamped
subgraphs to reduce temporal conflicts and redun-
dancy. TG-RAG (Han et al., 2025) further in-
troduces hierarchical temporal graphs for multi-
granularity retrieval, and DyG-RAG (Sun et al.,
2025) models dynamic event-centric retrieval us-
ing temporally grounded event units. In parallel,
E2RAG (Zhang et al., 2026b) preserves evolving
entity states through a dual entity-event graph archi-
tecture. However, existing temporal RAG systems
primarily model time as discrete timestamps or
temporal filters, lacking a formal semantic repre-
sentation of interval relations such as overlap and
duration. As a result, they struggle with queries in-
volving implicit temporal states or complex interval
constraints. IA-RAG addresses this limitation by
2

IA-RAG (Ours) GraphRAG Temporal RAG Vanilla RAG
Temporal 
Represen-
tationQuery : Which employer did Solomon Lefschetz work for 7 years and 7 months after July 1910 ?
...
Chunk1 : Solomon took positions in 
University of Nebraska in 1900 ...
Chunk2 : Solomon Lefschetz moved to 
Princeton University in 1924 ...
IEU Solomon edited the Annals of
Mathematics from 1928 to 1958.
Time Interval: [1928-01-01, 1958-12-31]Solomon Lefschetz traveled 
to Mexico in 1945.
Time : 1945
Chunk3 : ...Clark University in 
Worcester, Massachusetts in 1911. 
Community : 
Entity :
Relation :(Solomon, moved to, 
Princeton University
time=1924)Event1T1
Event2T2
Event3 T3...
...Event4T4•Time Inference: February 1918
Event1T1
encoderMEU MEU MEUMEU MEU
...
IEU IEU 
IEU IEU IEU 
IEU IEU 
Temporal 
Reasoning
Fails to link hidden 
timelinesRelational focus 
weak temporal inferencePoint-based
struggles with implicit spansInterval-centric
superior implicit temporal inferenceQuery Query Embedding
Top-k semantic chunks
Chunk 1
1907
Chunk 2
1900
Chunk 5
1902Chunk 4
1901
Chunk 31918
Query Entity linking Query
Implicit 
Temporal 
Inference
relevant 
unrelevantSolomon 
Lefschetz
Princeton 
UniversityNebraska
UniversitySolomon 
Lefschetz
Clark 
UniversityTime:1924Time:1900Time:1911
1910
1918Seed19001911
19281912 1914
1917
1910SeedPath 
sampingTime-blind
fragmented text chunksStructural links
static temporal attributesSingle timestamps
implicit point encodingExplicit intervals
duration- aware modeling
Low Accuracy
 Limited Accuracy Moderate Accuracy High AccuracyQueryinterval-
topologyTop-k 
IEU s,  MEU s
MEU
MEU
MEU 1924-01~1945-01overlapped_by
before1917-11~1917-12
1918-02~1924-03
IEU IEU 
IEU IEU meets1905-02~1907-08
1902-09~1910-071910-07~1910-08
1918-01~1918-10beforebeforeFigure 1: Comparison of representative RAG methods and IA-RAG.
representing events as explicit temporal intervals
and performing retrieval under Allen’s Interval Al-
gebra (Allen, 1984), enabling structured reasoning
over temporal topologies.
3 Method
3.1 IA-RAG Workflow
In this section, we presentIA-RAG, a tempo-
ral GraphRAG framework that models events as
time intervals and performs retrieval under interval-
algebraic constraints. As shown in Figure 2, IA-
RAG consists of three stages: (1) interval event
unit extraction, (2) interval-aware event graph con-
struction, and (3) query-driven temporal retrieval.
3.2 Interval Event Unit Extraction
Definition of Interval Event Unit.An IEU is a
minimal and self-contained representation of a fac-
tual statement that describes either an instantaneous
event or a persistent state over time. Formally, an
IEUe iis defined as
ei=⟨s i, ci,[tstart
i, tend
i], fi, src i⟩,(1)
where sidenotes the event sentence, cirepresents
optional contextual information, [tstart
i, tend
i]spec-
ifies the temporal boundaries of the event, fi∈
{0,1} is a temporal uncertainty indicator, and srci
identifies the originating document.
To extract IEUs, we first segment each source
document into overlapping text chunks to preserve
local coherence. Events with explicit timestampsare directly aligned to precise intervals with fi= 0,
while vague expressions (e.g., “early 1990s”) are
mapped to coarse-grained intervals inferred from
context and marked as uncertain withf i= 1.
IEU Filtering and DeduplicationThe initial ex-
traction phase inevitably yields redundant IEUs
due to cross-document repetition and overlapping
segmentation. To mitigate this redundancy, we
perform IEU-level deduplication using alocal se-
mantic neighborhoodstrategy. For each IEU ei, we
retrieve its top- Qmost semantically similar neigh-
borsNQ(ei)based on cosine similarity. The set
{ei} ∪ N Q(ei)is jointly examined by an LLM to
determine whether a subset of IEUs represents du-
plicate descriptions of the same event.
3.3 Interval-Aware Event Graph Construction
After filtering and deduplication, the extracted In-
terval Event Units (IEUs) serve as atomic nodes for
temporal knowledge organization. We construct an
interval-aware event graphby explicitly modeling
both semantic relatedness and qualitative temporal
relations among IEUs.
Definition of Interval-Aware Edge.Given two
IEUs eiandej, an interval-aware temporal relation
is established only when they satisfy both semantic
relevance and temporal comparability.
Lethi,hj∈Rddenote dense semantic embed-
dings of the event descriptions. We first compute a
3

IEU 
IEU IEU 
IEU IEU 
IEU ...
...
IEU 
IEU IEU 
IEU IEU 
IEU 
CorpusInterval Event Unit Extraction
Chunk
Chunk...
 sim (IEU i, IEU j) > �Merged Sentence: 
From 1924 to 1958, Solomon 
Lefschetz was employed at ...
Time : 1924-01-01 ~ 1958-12-31
Thematic Forest IA-Guided Traversal Query Temporal AnchoringEdge Construction Graph Construction Sub-graph Time Tightening
fuzzy=TrueTime : 1924-01-01 ~ 1924-12-31 IEU
fuzzy=FalseTime : 1924-12-01 ~ 1924-12-31 IEU 
MEU fuzzy=Falseupdate
1924-
01-011924-
12-311928-
12-011958-
12-311945-
03-011928-
01-01IEU1 IEU2 IEU3
IEU2 
IEU3 IEU1 
IEU2 before
overlaps
Query : Which employer did Solomon 
Lefschetz work for 7 years and 7 
months after July 1910 ?
Top-k1 MEUs          Top-k2 IEUs
MEU MEU ...MEU MEU MEUMEU MEU
...
IEU IEU 
IEU IEU IEU 
IEU 
Answer : Between 1918 and 1924, 
Solomon Lefschetz was working at the 
University of Kansas.
IEU
IEU...
 IEU =    sentence  +  context  +  [ t_start, t_end ]  +  fuzzy  +  Doc_id IEU 1Solomon took positions in 
University of Nebraska ...Princeton University 
until 1953...1924-01-01 ~ 1924-12-31 True D1
IEU 2Solomon edited the 
Annals of Mathematics ...During this period, 
the journal ...1928-01-01 ~ 1958-12-31FalseD2
IEU fuzzy=True
IEU fuzzy=False
summary
IEU
LLM
Temporal Window: 1918-02-01~1918-02-28
Query Embedding                     
Seed SupplementaryMEUMEU
MEU 1924-01~1945-01overlapped_by
before1917-11~1917-12
1918-02~1924-03
IEU IEU 
IEU IEU meets1905-02~1907-08
1902-09~1910-071910-07~1910-08
1918-01~1918-10beforebeforeFigure 2: Overall framework of IA-RAG.
semantic compatibility indicator:
S(ei, ej) =Ihi·hj
∥hi∥∥hj∥> τ sem
,(2)
where τsemis a predefined similarity threshold.
Only pairs with S(ei, ej) = 1 are considered for
temporal relation assignment.
Allen Interval Relations.For semantically com-
patible IEUs, we determine their qualitative tempo-
ral relation using Allen’s Interval Algebra (Allen,
1984). Allen’s Interval Algebra provides a rigor-
ous formalism for modeling the relative ordering
of time intervals, enabling the construction of a
temporal topology among events. Given the tem-
poral intervals [tstart
i, tend
i]and[tstart
j, tend
j], their
relation is defined as:
R(ei, ej)∈ A,(3)
where Adenotes the set of 13 mutually exclusive
Allen relations. Relations are detailed in Table 1.
Sub-graph Time Tightening.Contextual frag-
mentation and sparse textual evidence often pro-
duce IEUs with uncertain temporal boundaries,
marked by a fuzzy flag ( f= 1 ). To reduce such
ambiguity, IA-RAG leverages the interval-awareRelation Definition Inverse
beforetend
i< tstart
j after
meetstend
i=tstart
j met_by
overlapststart
i< tstart
j< tend
i< tend
j overlapped_by
startststart
i=tstart
j, tend
i< tend
j started_by
duringtstart
j< tstart
i< tend
i< tend
j contains
finisheststart
i> tstart
j, tend
i=tend
j finished_by
equalststart
i=tstart
j, tend
i=tend
j -
Table 1: The 13 mutually exclusive Allen’s interval
relations, including 7 basic relations and their corre-
sponding inverses.
event graph to recover latent temporal dependen-
cies. Formally, we identify the set of connected
componentsCvia graph decomposition:
C=ϕ(G) ={C 1,C2, . . . ,C N},(4)
where ϕ(·) denotes the connected component com-
putation. By incorporating Interval Algebra, dis-
crete temporal signals are seamlessly bridged, al-
lowing fragmented information to converge. This
integration catalyzes the evolution of loose, theme-
centric events into structured, interconnected com-
ponents.
Based on this observation, given an IEU with the
original temporal interval τ≜
tstart, tend
and
the fuzzy status with f= 1 , we designate its en-
compassing component Ck∈C as the grounded
4

local context. Within Ck, the surrounding IEUs and
their explicit Allen relations impose logical con-
straints that implicitly bound the target’s timeline.
The refined temporal interval τ∗is then derived via
an LLM-based reasoning operatorL:
τ∗=L(τ| C k),(5)
Upon reasoning, if the contextual evidence in Ck
supports a precise temporal alignment, the fuzzy
flag is updated to f= 0 ; otherwise, the IEU re-
mains marked as fuzzyf= 1.
Macro Event Units.Connected components nat-
urally group thematically related events. Based on
this observation, we construct Macro Event Units
(MEUs) by aggregating the events within each com-
ponent. MEUs preserve interval-aware temporal
semantics while providing higher-level thematic
abstraction. Allen relations are then computed
between MEU intervals, producing a higher-level
interval-aware graph and enabling recursive hierar-
chical construction.
We generalize this process by anchoring the base
hierarchy to the initial global graph, defined as
G(0)≡G with component set eC(0)≡C . Subse-
quently, this aggregation process for llayer to l+ 1
layer is defined as:
U(l+1)=n
g(Ci)| Ci∈eC(l)o
,(6)
G(l+1)=
U(l+1),E(l+1)
,(7)
where g(·)denotes a composite abstraction opera-
tor that unifies the temporal boundaries and sum-
marizes the semantic content of the events within
each component. G(l+1),U(l+1), andE(l+1)repre-
sent the event graph, the component set of MEUs,
and the component set of qualitative temporal con-
straints among these MEUs atl+ 1layer.
Thematic Forest Construction.Combining the
Eqs. (4)-(8), the recursive abstraction process in-
duces a genealogical structure across layers. Specif-
ically, each unit u∈ U(l+1)is synthesized from a
unique component e∈ C(l), and a directed contain-
ment relationship exists h(·,·) between the high-
layer unit and its constituent events, defined as:
T(l)=h(u, e)≜u↔e,(8)
where eis an IEU when l= 0 or a MEU when
alll >0 . Collectively, these relationships form a
set of hierarchical trees, where the roots representthe most abstract narrative themes and the leaves
correspond to atomic IEUs. We define this result-
ing multi-layered structure as theThematic Forest,
which organizes temporal knowledge from granular
details to broad narrative arcs.
3.4 Query-Driven Event Retrieval
Query Temporal Scope Anchoring.Given a
user query q, our objective is to ground the query
by inferring its temporal scope. To achieve this, we
employ an LLM-based parser Φd(·)to extract the
corresponding temporal window:
Tw= [w s, we] = Φ d(q),(9)
where wsandwedenote the inferred start and
end timestamps.
Event-to-Forest Matching.Existing temporal
RAG methods typically operate on flat graph struc-
tures, treating temporal information as isolated em-
bedding attributes (Sun et al., 2025). Such designs
overlook high-level thematic organization and of-
ten require expensive global traversal.
In contrast, IA-RAG adopts a hierarchical coarse-
to-fine retrieval strategy over the proposed The-
matic Forest, where IEUs provide fine-grained fac-
tual evidence, and MEUs capture high-level the-
matic and temporal context. This hierarchy enables
more accurate retrieval while preserving both fac-
tual precision and temporal coherence. Given a
query q, we independently retrieve relevant MEUs
and IEUs.:
M∗= TopK 1
u∈USim(h q,hu),(10)
I∗= TopK 2
e∈VSim(h q,he),(11)
wherehq,hu, andhedenote the embeddings of the
query, MEUs, and IEUs. UandVdenote the MEU
and IEU spaces. M∗andI∗represent the retrieved
thematic units and event units, respectively. K1and
K2are retrieval hyperparameters1.
Interval-Algebra–Guided Traversal.Due to
the absence of explicit temporal awareness, purely
semantic retrieval may retrieve semantically rele-
vant events that occurred at incorrect time periods.
To address this challenge, IA-RAG performs tem-
poral traversal over both the MEU-level forest and
the IEU-level event graph according to the rela-
tive position between retrieved intervals and the
1Hyperparameter experiments are provided in the Ap-
pendix B.2
5

inferred query window Tw. For each retrieved unit
xi∈ {M∗,I∗}with temporal interval
Ti= [tstart
i, tend
i],(12)
the traversal direction is dynamically determined
as:
Ψ(Ti, Tw) =

RF, tend
i< ws,
RB, tstart
i> we,
RF∪ RB,otherwise.(13)
Here,RFandRBdenote the forward and back-
ward Allen relation sets, respectively:
RF={before,meets,overlaps,
starts,during,finishes,equals},(14)
RB={after,met_by,overlapped_by,
started_by,contains,finished_by,equals}.(15)
Traversal over both hierarchical MEU graphs
and fine-grained IEU graphs proceeds forward,
backward, or bidirectionally according to the Allen
temporal relation between the retrieved unit and
the target window:
Msup=n
uj∃ui∈ M∗, uj∈ N(u i),
R(u i, uj)∈Ψ(T i, Tw)o
,(16)
Isup=n
ej∃ei∈ I∗, ej∈ N(e i),
R(ei, ej)∈Ψ(T i, Tw)o
,(17)
whereN(·) denotes temporal neighbors connected
through Allen interval relations. Finally, the supple-
mentary units are merged with the initial retrieval
results to form the final retrieval set:
Itarget=I∗∪ Isup∪ M∗∪ Msup.(18)
4 Experimental Setup
In our experiments, we aim to answer the following
research questions:
•RQ1:How does IA-RAG compare against state-
of-the-art baselines on temporal QA tasks?
•RQ2:Can IA-RAG achieve superior tempo-
ral reasoning performance while reducing the
amount of retrieved context?
•RQ3:How do the hierarchical forest and sub-
graph time tightening modules affect retrieval
and answer quality?•RQ4:How effective is the proposed interval-
algebra-guided traversal strategy for capturing
directional temporal dependencies?
•RQ5:How does IA-RAG perform across differ-
ent temporal reasoning types?
4.1 Datasets and Evaluations
We conduct experiments on three widely used
temporal question answering benchmarks:
TimeQA(Chen et al., 2021),TempReason(Tan
et al., 2023), andComplexTR(Tan et al.,
2024). Following the experimental setup of
DyG-RAG (Sun et al., 2025), we directly adopt
their processed document corpus and the corre-
sponding question–answer pairs, ensuring a fair
and controlled comparison. For evaluation, we
follow the token-level evaluation protocol (Zhou
et al., 2025), reporting both Accuracy and Recall,
computed based on the overlap between the
predicted answers and the gold-standard answer.
4.2 Baselines
We compare IA-RAG with representative Vanilla
RAG, Graph RAG, and Temporal-Aware RAG
methods:
•Vanilla RAG (Lewis et al., 2020): Dense retrieval
over vectorized document chunks.
•GraphRAG (Edge et al., 2024): Community-
based graph retrieval with local (L) and global
(G) variants.
•LightRAG (Guo et al., 2024): Dual-level index-
ing framework with local (L), global (G), and
hybrid (H) retrieval.
•HippoRAG (Gutiérrez et al., 2024): Mimics hu-
man memory by running personalized PageRank
over knowledge graphs for multi-hop retrieval.
•E2GraphRAG (Zhao et al., 2025): Builds bidi-
rectional indexes across a summary tree and an
entity graph to capture many-to-many document
relationships.
•TG-RAG (Han et al., 2025): Structures times-
tamped edges into bi-level graphs with hierarchi-
cal time nodes for multi-granularity summaries.
•T-GRAG (Li et al., 2025): Models historical evo-
lution via timestamped subgraphs and a three-
layer retriever to resolve knowledge conflicts.
•TA-RAG (Lau et al., 2025): Calibrates semantic
matching against temporal coherence by separat-
ing queries into subjects and time windows.
•DyG-RAG (Sun et al., 2025): It introduces dy-
namic event units and constructs a temporally
6

MethodsTimeQA TempReason ComplexTR
Acc. Recall Acc. Recall Acc. Recall
Traditional RAG Methods
Vanilla RAG 56.33 64.98 70.96 82.53 55.01 68.68
GraphRAG-L 40.26 46.28 56.11 67.55 43.16 54.29
GraphRAG-G 10.10 13.83 8.81 11.34 20.97 30.81
LightRAG-L 34.94 39.89 54.23 66.82 41.03 49.46
LightRAG-G 6.66 8.12 13.54 15.27 12.46 13.56
LightRAG-H 36.36 43.06 50.01 62.95 42.68 53.59
HippoRAG 39.99 45.39 69.80 80.54 44.68 55.28
E2GraphRAG 40.48 50.19 61.29 73.58 38.29 54.99
Temporal QA Retrieval Methods
T-GRAG 29.89 34.21 55.42 65.32 27.66 34.78
TG-RAG 47.80 56.09 30.91 38.47 50.15 63.54
TA-RAG 42.25 48.92 58.66 67.97 29.26 37.22
DyG-RAG 58.78 67.02 84.75 91.4755.62 69.88
IA-RAG (ours)61.72(+2.94)69.25(+2.23) 80.21 89.64 65.95(+10.33)77.42(+7.54)
Table 2: Overall performance comparison on temporal QA benchmarks. Acc. and Recall denote answer accuracy
and retrieval recall, respectively. The best results are highlighted inbold, the second-best results are underlined ,
and improvements over the strongest baseline are shown in teal.
grounded event graph to support time-aware re-
trieval and multi-hop temporal reasoning.
4.3 Implementation Details
We employ Qwen2.5-14B-Instruct (Qwen et al.,
2025) as the large language model for all graph-
construction and generation tasks. And all retrieval
modules use the BGE-M3 (Chen et al., 2024) en-
coder. All documents are segmented using a sliding
window approach with a fixed chunk size of 1,200
tokens and an overlap of 64 tokens. Hierarchical
thematic forest is constructed with a maximum
depth of 4 layers. Additional implementation de-
tails, hyperparameter settings, and sensitivity anal-
yses are provided in Appendix B.
5 Results and Analyses
5.1 Overall Performance Comparison (RQ1)
Table 2 reports the overall results on three tempo-
ral QA benchmarks. Overall, IA-RAG achieves
the best performance on TimeQA and ComplexTR,
demonstrating the effectiveness of interval-algebra-
guided traversal for complex temporal reasoning
and multi-hop chronological inference. Compared
with existing RAG and temporal retrieval baselines,
IA-RAG consistently improves retrieval precision
by explicitly modeling temporal interval relations
and dynamically constraining traversal directions.
On TempReason, DyG-RAG achieves slightly bet-
ter results. We attribute this to the fact that manyTempReason questions focus on single timestamp-
oriented temporal matching, where DyG-RAG’s
temporal-semantic embedding alignment and local
graph expansion are particularly effective. In con-
trast, IA-RAG shows larger advantages on bench-
marks requiring more complex directional temporal
reasoning and interval-level dependency modeling.
5.2 Efficiency Analysis (RQ2)
Figure 3 compares answer accuracy and input token
consumption across different methods. Although
IA-RAG performs temporal traversal at both the
MEU and IEU levels, it maintains relatively com-
pact prompt sizes while achieving strong perfor-
mance. IA-RAG significantly reduces input tokens
on ComplexTR while obtaining higher accuracy, in-
dicating that the proposed interval-algebra-guided
traversal retrieves more temporally relevant evi-
dence instead of relying on exhaustive graph ex-
pansion. Overall, the results demonstrate that IA-
RAG achieves a better balance between retrieval
effectiveness and efficiency for temporal QA.
5.3 Effect of Core IA-RAG Components
(RQ3)
Table 3 presents the ablation results of IA-RAG
across three benchmarks. Removing the interval-
algebra-guided traversal (w/o IA) precipitates the
most severe performance drop, underscoring that
pure semantic similarity is insufficient for complex
7

Figure 3: Comparison between answer accuracy and
input token consumption across temporal QA methods.
MethodsTimeQA TempReason ComplexTR
Acc. Recall Acc. Recall Acc. Recall
IA-RAG 61.72 69.25 80.21 89.64 65.95 77.42
w/o IA57.90 65.48 77.41 88.09 61.70 74.55
w/o Tight.60.81 68.66 79.66 89.12 63.52 76.34
w/o Hier.60.47 68.28 78.98 88.44 62.31 74.08
Table 3: Ablation study of IA-RAG. We evaluate the
contribution of interval-algebra-guided traversal (IA),
hierarchical thematic retrieval (Hier.), and sub-graph
temporal tightening (Tight.).
temporal QA, which necessitates reasoning beyond
isolated seed facts.
Concurrently, removing the sub-graph time tight-
ening mechanism (w/o Tight.) consistently erodes
accuracy; this confirms that factual evidence from
fragmented corpora suffers from temporal bound-
ary uncertainty, and tightening fuzzy intervals via
local topological dependencies successfully elimi-
nates temporal noise to preserve information-rich,
chronologically coherent events. We further hy-
pothesize that the slightly marginal gain from tight-
ening on TempReason stems from the inherent tem-
poral sensitivity bottlenecks of the backbone model
in grounding explicit timestamps.
Removing the hierarchical structure (w/o Hier.)
harms performance, especially on ComplexTR, val-
idating that our coarse-to-fine paradigm prevents
retrieval from collapsing into flat event-level vector
matching and reduces matching ambiguity.
5.4 Traversal Strategy Analysis (RQ4)
Table 4 compares different traversal strategies. Us-
ing all Allen relations without directional filtering
leads to clear performance degradation, showing
that unrestricted traversal introduces temporally ir-
relevant neighbors and noisy evidence. Replacing
dynamic traversal with LLM-predicted top- 5Allen
relations also underperforms IA-RAG. Although
LLM-predicted relations capture partial temporal
dependencies, they often lack stable chronologicalMethodsTimeQA TempReason ComplexTR
Acc. Recall Acc. Recall Acc. Recall
IA-RAG 61.72 69.25 80.21 89.64 65.95 77.42
w/All60.54 68.89 77.49 88.10 62.91 75.75
w/Top-561.50 68.98 78.30 88.32 63.22 76.25
Table 4: Ablation study on traversal strategies.All
denotes traversal over all Allen relations, whileTop-5
uses the top-5 Allen relations predicted by an LLM.
Methods Explicit Relative Comp. Overall
TG-RAG 42.98 68.49 46.67 50.15
Vanilla RAG 52.07 68.49 50.37 55.01
DyG-RAG 51.24 64.38 54.81 55.62
IA-RAG (Ours) 57.85 73.97 68.88 65.95
Improvement+5.78 +5.48 +14.07 +10.33
Table 5: Accuracy breakdown on ComplexTR across
different temporal reasoning categories.
consistency. In contrast, IA-RAG dynamically se-
lects forward, backward, or bidirectional traversal
according to the relative position between the seed
event and the target temporal window. This allows
the model to focus on chronologically relevant re-
gions while avoiding unnecessary expansion.
5.5 Breakdown Analysis on Temporal
Reasoning Types (RQ5)
Table 5 reports the performance breakdown on
ComplexTR across different temporal reasoning
categories2. IA-RAG consistently achieves the best
performance across explicit, relative, and compo-
sitional temporal reasoning tasks. The largest im-
provement is observed in compositional reasoning,
where IA-RAG surpasses the strongest baseline
by more than 14%, demonstrating the effective-
ness of interval-algebra-guided traversal for multi-
hop temporal dependency modeling and temporal
computation. IA-RAG also shows clear gains in
relative temporal reasoning, indicating that dynam-
ically constrained traversal effectively captures di-
rectional interval relations and event-conditioned
temporal dependencies.
6 Conclusion
In this paper, we present IA-RAG, a hierarchical
temporal RAG framework that models knowledge
as time intervals under Allen’s Interval Algebra.
2Detailed reasoning type definitions and dataset statistics
are provided in Appendix D.
8

Through interval-aware retrieval, hierarchical the-
matic organization, and sub-graph time tightening,
IA-RAG enables structured temporal reasoning be-
yond conventional timestamp-based retrieval. In ad-
dition, the proposed interval-algebra-guided traver-
sal supports implicit temporal semantic retrieval un-
der complex temporal dependencies. Experiments
on multiple temporal QA benchmarks demonstrate
strong performance, particularly on complex com-
positional temporal reasoning tasks.
Limitations
Although IA-RAG achieves strong performance
on temporal reasoning tasks, several limitations re-
main. First, the framework relies on the quality
of temporal extraction and interval normalization,
which may introduce errors when handling ambigu-
ous or noisy temporal expressions. Second, the
current framework mainly focuses on interval-level
temporal reasoning and does not explicitly model
causal relations or evolving event dynamics, which
we leave for future work.
References
James F Allen. 1984. Towards a general theory of action
and time.Artificial intelligence, 23(2):123–154.
Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu
Lian, and Zheng Liu. 2024. BGE M3-embedding:
Multi-lingual, multi-functionality, multi-granularity
text embeddings through self-knowledge distillation.
arXiv preprint arXiv:2402.03216, 4(5).
Wenhu Chen, Xinyi Wang, and William Yang Wang.
2021. A dataset for answering time-sensitive ques-
tions.arXiv preprint arXiv:2108.06314.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and
Jonathan Larson. 2024. From local to global: A
graph rag approach to query-focused summarization.
arXiv preprint arXiv:2404.16130.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tian Ao, and
Chao Huang. 2024. Lightrag: Simple and fast
retrieval-augmented generation.arXiv preprint
arXiv:2410.05779, 2(3).
Bernal J Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Ya-
sunaga, and Yu Su. 2024. Hipporag: Neurobiologi-
cally inspired long-term memory for large language
models.Advances in neural information processing
systems, 37:59532–59569.
Jiale Han, Austin Cheung, Yubai Wei, Zheng Yu,
Xusheng Wang, Bing Zhu, and Yi Yang. 2025. Rag
meets temporal graphs: Time-sensitive modeling andretrieval for evolving knowledge.arXiv preprint
arXiv:2510.13590.
Kwun Hang Lau, Ruiyuan Zhang, Weijie Shi, Xiaofang
Zhou, and Xiaojun Cheng. 2025. Reading between
the timelines: Rag for answering diachronic ques-
tions.arXiv preprint arXiv:2507.22917.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Dong Li, Yichen Niu, Ying Ai, Xiang Zou, Biqing Qi,
and Jianxing Liu. 2025. T-grag: A dynamic graphrag
framework for resolving temporal conflicts and re-
dundancy in knowledge retrieval. InProceedings of
the 33rd ACM International Conference on Multime-
dia, pages 11880–11889.
Humza Naveed, Asad Ullah Khan, Shi Qiu, Muhammad
Saqib, Saeed Anwar, Muhammad Usman, Naveed
Akhtar, Nick Barnes, and Ajmal Mian. 2025. A com-
prehensive overview of large language models.ACM
Transactions on Intelligent Systems and Technology,
16(5):1–72.
Bhawna Piryani, Abdelrahman Abdallah, Jamshid
Mozafari, Avishek Anand, and Adam Jatowt. 2025.
It’s high time: A survey of temporal question answer-
ing.Preprint, arXiv:2505.20243.
Hongjin Qian, Zheng Liu, Peitian Zhang, Kelong Mao,
Defu Lian, Zhicheng Dou, and Tiejun Huang. 2025.
Memorag: Boosting long context processing with
global memory-enhanced retrieval augmentation. In
Proceedings of the ACM on Web Conference 2025,
pages 2366–2377.
Qwen, An Yang, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan
Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan
Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin
Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, and
24 others. 2025. Qwen2.5 technical report.Preprint,
arXiv:2412.15115.
Ishaan Singh, Navdeep Kaur, Garima Gaur, and 1 oth-
ers. 2023. Neustip: A novel neuro-symbolic model
for link and time prediction in temporal knowledge
graphs.arXiv preprint arXiv:2305.11301.
Qingyun Sun, Jiaqi Yuan, Shan He, Xiao Guan, Haonan
Yuan, Xingcheng Fu, Jianxin Li, and Philip S Yu.
2025. Dyg-rag: Dynamic graph retrieval-augmented
generation with event-centric reasoning.arXiv
preprint arXiv:2507.13396.
Qingyu Tan, Hwee Tou Ng, and Lidong Bing. 2023.
Towards benchmarking and improving the temporal
reasoning capability of large language models. In
Proceedings of the 61st Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 14820–14835.
9

Qingyu Tan, Hwee Tou Ng, and Lidong Bing. 2024.
Towards robust temporal reasoning of large lan-
guage models via a multi-hop qa dataset and pseudo-
instruction tuning. InFindings of the Association for
Computational Linguistics: ACL 2024, pages 6272–
6286.
Guohang Yan, Yue Zhang, Pinlong Cai, Ding Wang,
Song Mao, Hongwei Zhang, Yaoze Zhang, Hairong
Zhang, Xinyu Cai, and Botian Shi. 2025. Het-
arag: Hybrid deep retrieval-augmented genera-
tion across heterogeneous data stores.Preprint,
arXiv:2509.21336.
Zairun Yang, Yilin Wang, Zhengyan Shi, Yuan Yao, Lei
Liang, Keyan Ding, Emine Yilmaz, Huajun Chen,
and Qiang Zhang. 2025. Eventrag: Enhancing llm
generation with event knowledge graphs. InProceed-
ings of the 63rd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Pa-
pers), pages 16967–16979.
Yaoze Zhang, Rong Wu, Pinlong Cai, Xiaoman Wang,
Guohang Yan, Song Mao, Ding Wang, and Botian
Shi. 2026a. Leanrag: Knowledge-graph-based gen-
eration with semantic aggregation and hierarchical
retrieval. InProceedings of the AAAI Conference
on Artificial Intelligence, volume 40, pages 34862–
34869.
Ze Yu Zhang, Zitao Li, Yaliang Li, Bolin Ding,
and Bryan Kian Hsiang Low. 2026b. Respecting
temporal-causal consistency: Entity-event knowl-
edge graph for retrieval-augmented generation. In
Proceedings of the 19th Conference of the European
Chapter of the Association for Computational Lin-
guistics (Volume 1: Long Papers), pages 2017–2054.
Yibo Zhao, Jiapeng Zhu, Ye Guo, Kangkang He, and
Xiang Li. 2025. Eˆ 2graphrag: Streamlining graph-
based rag for high efficiency and effectiveness.arXiv
preprint arXiv:2505.24226.
Yingli Zhou, Yaodong Su, Youran Sun, Shu Wang, Tao-
tao Wang, Runyuan He, Yongwei Zhang, Sicong
Liang, Xilin Liu, Yuchi Ma, and 1 others. 2025. In-
depth analysis of graph-based rag in a unified frame-
work.arXiv preprint arXiv:2503.04338.
10

A Use of Large Language Models
The research presented in this paper, including
the core ideas, experimental design, and quanti-
tative results, is the original work of the authors. A
large language model was used as a writing assis-
tant for tasks such as polishing prose, improving
clarity, and correcting grammatical errors in the
manuscript. All final content was reviewed and
edited by the authors to ensure it accurately reflects
our research and contributions.
B Experimental Setup
B.1 Setting
Model and Inference.All stages of our
pipeline, including Interval Event Unit (IEU) ex-
traction, semantic filtering, deduplication, and
question answering, are implemented using
Qwen2.5-14B-Instruct3.
Hardware Configuration.All experiments were
conducted on a machine equipped with 4 ×
NVIDIA A100 GPUs (80GB memory per GPU).
Inference is performed in parallel where applicable.
Similarity Thresholds.For IEU-level semantic
filtering and deduplication, we compute cosine sim-
ilarity between event embeddings and apply a sim-
ilarity threshold of τsem= 0.8 . Only event pairs
exceeding this threshold are considered semanti-
cally related.
Thematic Forest Construction.During The-
matic Forest construction, we apply a semantic
similarity threshold of 0.8at the first abstraction
layer ( U(0)→ U(1)) to form Macro Event Units
(MEUs). The same threshold is used consistently
across subsequent layers unless otherwise speci-
fied.
Retrieval Configuration.In the retrieval stage,
we adopt a two-step coarse-to-fine retrieval strat-
egy. Specifically, we first retrieve the top- K1most
relevant MEUs, followed by the top- K2IEUs. For
TimeQA and ComplexTR, we set K1= 10 , while
K1= 20 is used for TempReason. Across all
datasets, we fixK 2= 20.
B.2 Sensitivity Analysis on Forest Depth and
MEU Granularity
Tables 7 and 8 present the sensitivity analysis
of IA-RAG with respect to thematic forest depth
3https://huggingface.co/Qwen/Qwen2.
5-14B-InstructParameter Value Description
K50 Number of nearest
neighbors
Tsem 0.8 Semantic similarity
threshold
MIN_CLUSTER_SIZE 2 Minimum size for ag-
gregation
MAX_LEVELS 4 Maximum hierarchy
depth
DECAY 0.95 Threshold decay fac-
tor
MIN_THRESHOLD 0.6 Lower bound of simi-
larity threshold
MAX_THRESHOLD 0.85 Initial similarity
threshold
MAX_WORKERS 32 Parallel processing
threads
Table 6: Key hyperparameters used in graph construc-
tion.
(MAX_LEVELS) and MEU retrieval granularity.
For thematic forest depth, setting MAX_LEVELS =0
degenerates IA-RAG into a flat graph without hi-
erarchical thematic abstraction, resulting in consis-
tent performance drops across all benchmarks. In-
troducing moderate hierarchical depth significantly
improves both accuracy and recall, indicating that
multi-level thematic organization effectively nar-
rows the retrieval space before fine-grained event
reasoning. However, excessively deep hierarchies
slightly degrade performance, likely because over-
fragmented thematic partitions weaken semantic
coherence and increase retrieval sparsity.
For MEU retrieval granularity, a moderate num-
ber of retrieved MEUs achieves the best overall
performance. Retrieving too few MEUs limits the-
matic coverage and may overlook relevant tempo-
ral evidence, while retrieving too many introduces
noisy or weakly related subgraphs that reduce re-
trieval precision. These results demonstrate that
IA-RAG benefits from a balanced coarse-to-fine
retrieval strategy that preserves thematic diversity
while maintaining temporal relevance.
C Complexity and Theoretical Analysis
LetNdenote the number of atomic events and K
the number of nearest neighbors retrieved for each
event. The overall complexity of the framework
can be decomposed into three stages: retrieval,
graph construction, and hierarchical summariza-
tion.
Retrieval Complexity.At level 0, each atomic
event retrieves its top- Knearest neighbors using
11

MAX_LEVELSTimeQA TempReason ComplexTR
Acc. Recall Acc. Recall Acc. Recall
0 60.47 68.28 78.98 88.44 62.31 74.08
461.72 69.25 80.21 89.64 65.95 77.42
6 60.70 68.49 77.67 87.89 62.31 75.70
10 60.19 68.18 77.49 87.88 63.52 76.54
Table 7: Sensitivity analysis of thematic forest depth ( MAX_LEVELS ).MAX_LEVELS =0 corresponds to a flat graph
without hierarchical thematic organization.
MEUTimeQA TempReason ComplexTR
Acc. Recall Acc. Recall Acc. Recall
5 59.85 67.81 77.10 87.53 63.22 74.64
1061.72 69.2576.52 87.4865.95 77.42
15 60.93 68.47 77.91 88.19 62.91 75.89
20 59.89 67.6480.21 89.6460.79 74.04
40 59.51 67.55 76.52 86.23 60.79 74.49
Table 8: Sensitivity analysis of MEU retrieval granularity. MEU denotes the number of retrieved thematic units
used for coarse-to-fine retrieval.
an approximate nearest neighbor (ANN) index.
Assuming ANN retrieval requires approximately
O(logN) time per query, the total retrieval com-
plexity is:
Tretrieval =O(NlogN)(19)
Graph Construction Complexity.After re-
trieval, each node connects to at most Kneighbors,
resulting in at most O(NK) edges. Temporal re-
lation computation for each edge is constant time,
and connected component detection can be per-
formed in linear time with respect to the number of
nodes and edges. Therefore, the graph construction
complexity at each level is:
Tgraph=O(NK)(20)
For hierarchical graph construction, let Nlde-
note the number of nodes at level l. Assuming the
graph size decreases geometrically across levels:
Nl+1≤αN l,0< α <1(21)
the cumulative graph construction cost across all
levels becomes:
X
lO(N lK) =O(NK)(22)Hierarchical Summarization Complexity.At
each hierarchy level, connected components are
summarized using an LLM. Let Cldenote the num-
ber of connected components at level l, and let tLLM
denote the average inference cost of one summa-
rization call. The total summarization complexity
is therefore:
TLLM=X
lCl·tLLM (23)
Overall Complexity.Combining all stages, the
overall complexity of the framework is:
Ttotal=O(NlogN) +O(NK) +X
lCl·tLLM
(24)
Since Kis a small constant in practice, the dom-
inant non-LLM cost is the ANN-based retrieval
stage with approximate complexity:
Ttotal≈O(NlogN) +X
lCl·tLLM (25)
D Temporal Query Type Definition and
Dataset Distribution
To systematically analyze temporal reason-
ing complexity, we categorize temporal ques-
tions using an LLM-based taxonomy classifier
12

Dataset T1 T2 T3 T4 T5 Total
TimeQA 1589 1007 10 7 0 2613
TempReason 5397 0 0 0 0 5397
ComplexTR 72 49 42 73 93 329
Table 9: Distribution of fine-grained temporal reasoning
types across datasets.
(Qwen2.5-14B-Instruct ). We organize the
queries into three high-level reasoning types, which
encompass five fine-grained categories:
Explicit Temporal Reasoning These queries con-
tain explicit timestamps or intervals, primarily
requiring direct temporal matching. This group
includes:
•T1: Time-Anchored.Queries with explicit
timestamps or fixed ranges.
Ex: “Which position did Brian Smith hold after
May 2001?”
•T2: Explicit Interval.Queries with clearly
specified start and end times.
Ex: “Which team did Attaphol Buspakom play
for between May 1989 and Oct 1990?”
Relative Temporal Reasoning These queries de-
fine time implicitly through event relations, re-
quiring external temporal grounding. This group
includes:
•T4: Event-to-Event Relative.Queries defin-
ing temporal relations relative to another event.
Ex: “Where was Beate Meinl-Reisinger edu-
cated before being a member of NEOS?”
Compositional Temporal Reasoning These
queries involve temporal arithmetic or multi-hop
reasoning over both events and durations. This
group includes:
•T3: Numerical Offset.Queries involving arith-
metic operations over explicit timestamps.
Ex: “Which employer did Gustav Ludwig Hertz
work for 8 years before April 1926?”
•T5: Multi-hop Reasoning.Queries combining
event relations with duration-based computa-
tion.
Ex: “Which employer did Manuel García Ve-
larde work for 3 years after working for Uni-
versité libre de Bruxelles?”
As shown in Table 10, ComplexTR contains sub-
stantially more relative and composite temporal
reasoning questions than TimeQA and TempRea-
son, making it significantly more challenging for
retrieval-based temporal QA systems.Dataset Explicit Relative Compositional
TimeQA 2596 7 10
TempReason 5397 0 0
ComplexTR 121 73 135
Table 10: Distribution of high-level temporal reasoning
types across datasets.
EDistribution of Allen Interval Relations
Across Temporal QA Benchmarks
To better understand the temporal topology charac-
teristics of existing temporal QA benchmarks, we
analyze the distribution of Allen interval relations
within the constructed interval-aware event graphs.
Figure 4 presents the relation distributions across
TimeQA, TempReason, and ComplexTR.
First, although chronological relations such as
before/afterdominate all datasets, a substantial
number of non-trivial interval relations also ex-
ist, includingoverlaps,during/contains, andmeets.
This indicates that temporal reasoning in real-world
QA benchmarks extends beyond simple timestamp
ordering. Second, the existence of diverse inter-
val topologies empirically justifies the necessity of
interval-aware retrieval and traversal mechanisms.
These findings support our motivation that model-
ing time as isolated timestamps is insufficient for
capturing the rich temporal dependencies present
in complex QA tasks.
F Visualization of Hierarchical Thematic
Forest
To further illustrate the hierarchical organization
mechanism of IA-RAG, we visualize representa-
tive thematic forest structures constructed on the
ComplexTR benchmark. Figure 5 presents the hi-
erarchical relationship between Macro Event Units
(MEUs) and atomic Interval Event Units (IEUs).
At the lower level, IEUs represent fine-grained
temporally grounded events connected through
Allen interval relations. IA-RAG further aggre-
gates semantically and temporally coherent IEUs
into higher-level MEUs, which capture broader nar-
rative themes and temporal subtopics. Through
recursive abstraction, the resulting thematic forest
organizes temporal knowledge in a coarse-to-fine
manner, enabling efficient retrieval and directional
temporal reasoning.
The visualization demonstrates that IA-RAG can
simultaneously preserve detailed event-level tem-
poral structure while constructing high-level the-
13

Figure 4: Distribution of Allen interval relations across temporal QA benchmarks. Inverse relation pairs are merged
for clarity (e.g.,before/after,during/contains).
Hier_424 Hier_511L3
L2
EVT_709
EVT_708afterEVT_702
EVT_703EVT_705EVT_716
EVT_712EVT_714before
afterbeforeafter
beforeEVT_704EVT_710
EVT_707
EVT_706EVT_713EVT_711
EVT_713EVT_715
before
afterafter
equalsafterafter
beforeL0Hier_322 Hier_295 Hier_655overlapsL1Hier_342 Hier_643 Hier_807duringThematic Forest
......
...
...
Figure 5: Visualization of the hierarchical thematic forest constructed by IA-RAG on ComplexTR.
matic abstractions, thereby improving retrieval lo-
calization and reducing temporal ambiguity during
multi-hop reasoning.
GClarification on TG-RAG Performance
on TempReason
During experimental reproduction, we observed
that a portion of the events in the TempReason
dataset contain missing temporal annotations rep-
resented as UNKNOWN . This issue directly affects
the graph construction mechanism of TG-RAG,
which relies on explicit temporal information to
establish temporal connections between events. To
ensure a fair comparison, we excluded nodes with
UNKNOWN timestamps during graph construction and
conducted experiments on the resulting graph, from
which the reported performance was obtained. We
further conjecture that the relatively weak perfor-
mance of TG-RAG on TempReason may stem from
the limited robustness of its graph constructionstrategy under incomplete temporal supervision,
particularly when temporal information is missing
or ambiguously specified.
H Case Study
To further illustrate the effectiveness of IA-RAG,
we present several representative case studies in Ta-
bles 11, 12, and 13. These examples demonstrate
how IA-RAG performs temporal interval ground-
ing under complex temporal reasoning scenarios,
while existing RAG baselines often retrieve tempo-
rally irrelevant or logically inconsistent evidence.
I Prompt Templates
This appendix presents the prompt templates used
in IA-RAG, including interval event unit extrac-
tion, IEU deduplication, temporal interval tight-
ening, temporal reasoning type classification, and
answer generation. The corresponding templates
are shown in Figures 6–10.
14

QUESTION: Which employer did Karl Menger work for 3 years and 2 months before February 1930?
Gold answer: University of Amsterdam
(a) Chunk-based Answer: ×Incorrect
Based on the information provided in the text chunks, Karl Menger worked at the University of Vienna from 1927 to 1930.
Given the timeline provided, Karl Menger worked at the University of Vienna for approximately 2 years and 3 months
before February 1930. Therefore, the employer he worked for 3 years and 2 months before February 1930 does not align
exactly with any single employment stint mentioned. However, the closest match is his position at the University of Vienna,
which lasted less than the specified duration. The question might contain an error or assumption that doesn’t fit perfectly
with the given facts about Karl Menger’s career timeline.
(b) DyG-RAG Answer: ×Incorrect
To determine Karl Menger’s employer 3 years and 2 months before February 1930, we need to look back to December
1926. Based on the provided events and text chunks, we can analyze the timeline:
Events Analysis
• Event #10 states that Karl Menger accepted a professorship at the University of Vienna in 1927.
• Event #11 mentions Hertz becoming ordinarius professor of experimental physics and director of the Physics Institute
of the Technische Hochschule Berlin in 1928, which is irrelevant to Menger’s employment.
•Event #12 talks about Albert Einstein nominating Heisenberg, Born, and Jordan for the Nobel Prize in Physics in 1928,
which does not relate to Menger’s employment.
•Event #20 notes that Karl Menger was a visiting lecturer at Harvard University and The Rice Institute in 1930 and 1931.
Given this information, we can deduce that Menger was not at Harvard University or The Rice Institute prior to 1930.
Conclusion
From the timeline and the events provided, Karl Menger was at the University of Vienna from 1927 to 1929. Thus, 3
years and 2 months before February 1930 would place us in December 1926, during which time Menger was still at the
University of Vienna.
Answer:The employer of Karl Menger 3 years and 2 months before February 1930 was the University of Vienna.
(c) IA-RAG Answer: ✓Correct
To determine Karl Menger’s employer 3 years and 2 months before February 1930, we need to identify his employment
status in October 1926.
Timeline Analysis
• Event #1 (1924): Karl Menger received his PhD from the University of Vienna.
• Event #2 (1925): L. E. J. Brouwer invited Karl Menger to teach at the University of Amsterdam in 1925.
• Event #3 (1927): Karl Menger returned to Vienna to accept a professorship there in 1927.
Conclusion
According to the timeline, Menger taught at the University of Amsterdam from 1925 to 1927, before returning to Vienna.
Therefore, in October 1926, Karl Menger was employed at the University of Amsterdam. This inference is consistent with
the information in Chunk 1.
Answer:Karl Menger worked for the University of Amsterdam 3 years and 2 months before February 1930.
Table 11: Case study 1
15

QUESTION
Where was Stefan Mazurkiewicz educated after he/she studied at Ludwig Maximilian University of Munich?
Gold answer:University of Göttingen
(a) Chunk-based Answer: ×Incorrect
Based on the information provided in the text chunks, Stefan Mazurkiewicz, after studying at Ludwig Maximilian University
of Munich, became a professor at the University of Paris for a time. However, the text does not specify the exact educational
institution he attended immediately after Ludwig Maximilian University of Munich. It is clear that he spent most of his
career as a professor at the University of Warsaw. Therefore, the immediate educational step after Ludwig Maximilian
University of Munich mentioned is his period as a professor at the University of Paris, though it is not specified if he
pursued further formal education there or elsewhere before taking up that position.
(b) DyG-RAG Answer: ×Incorrect
None of the provided events or text chunks contain information about Stefan Mazurkiewicz’s educational background
after Ludwig Maximilian University of Munich. Therefore, the question cannot be answered based on the given data. The
information provided focuses on individuals and events unrelated to Stefan Mazurkiewicz’s educational history.
(c) IA-RAG Answer: ✓Correct
Based on the information provided in the text chunks and events, Stefan Mazurkiewicz was educated at the University of
Göttingen after studying at Ludwig Maximilian University of Munich.
From the text chunk:
“Stefan Mazurkiewicz was a Polish mathematician who worked in mathematical analysis, topology, and probability.
He was a student of Wacław Sierpi´ nski and a member of the Polish Academy of Learning (“PAU”). ”
And specifically from Event 3:
“He then went to the University of Göttingen on a grant from the Rockefeller Foundation to study mathematics under
David Hilbert. ”
Thus, after studying at Ludwig Maximilian University of Munich, Mazurkiewicz proceeded to the University of Göttingen
to further his education in mathematics.
Table 12: Case study 2
16

QUESTION: Where was Mary, Crown Princess of Denmark educated 15 years and 6 months before he/she was
married to Frederik, Crown Prince of Denmark?
Gold answer:Hobart College
(a) Chunk-based Answer: ×Incorrect
Based on the provided text chunks, Mary, Crown Princess of Denmark was educated at the University of Tasmania from
1990 to 1994, where she graduated with a combined Bachelor of Commerce and Bachelor of Laws degree on 27 May 1995.
To determine where she was educated 15 years and 6 months before her marriage to Frederik, Crown Prince of Denmark,
we need to calculate the dates:
Mary and Frederik got engaged on 8 October 2003 and were married on 14 May 2004.
15 years and 6 months before her marriage (14 May 2004) would be 14 November 1988.
Given this timeframe, we can see that during 14 November 1988, Mary was in her final year of high school at Taroona High
School in Hobart, Tasmania, as she was completing her secondary schooling from 1984 to 1987 and matriculation from
18 to 1989. Therefore, 15 years and 6 months before her marriage, Mary was attending Taroona High School in Hobart,
Tasmania.
(b) DyG-RAG Answer: ×Incorrect
To determine where Mary, Crown Princess of Denmark, was educated 15 years and 6 months before her marriage to
Frederik, Crown Prince of Denmark, we need to calculate the relevant timeline:
Timeline Calculation
Event #16 indicates that Mary and Frederik were married on 14 May 2004.
15 years and 6 months before 14 May 2004 would be 14 May 1988.
Education Analysis
From the provided text chunks, specifically Chunk 7:
• Mary Elizabeth Donaldson was born on 5 February 1972.
• She attended Clear Lake City Elementary School in Houston, Texas, from 1974 to 1977.
• Her primary education was at Waimea Heights from 1978 to 1983.
• Her secondary schooling was at Taroona High School from 1984 to 1987.
•She studied at the University of Tasmania from 1990 to 1994, graduating with a combined Bachelor of Commerce and
Bachelor of Laws degree on 27 May 1995.
Conclusion
Since 14 May 1988 falls within her secondary schooling period, Mary was attending Taroona High School at that time.
Answer:Mary, Crown Princess of Denmark, was educated at Taroona High School in Hobart, Tasmania, 15 years and 6
months before her marriage to Frederik, Crown Prince of Denmark.
(c) IA-RAG Answer: ✓Correct
To determine where Mary, Crown Princess of Denmark, was educated 15 years and 6 months before she was married to
Frederik, Crown Prince of Denmark, we need to identify the date of her marriage and then go back 15 years and 6 months
from that date.
Timeline Analysis
- From Event #12 (EVT002091): Frederik and Mary married on14 May 2004in Copenhagen Cathedral.
- Going back 15 years and 6 months from 14 May 2004 lands us on1 November 1988.
Education Verification
From Event #9 (EVT002087): Mary Elizabeth Donaldson graduated with a combined Bachelor of Commerce and Bachelor
of Laws degree on 27 May 1995.
From Chunk 2: “She studied at the University of Tasmania from 1990 to 1994. . . ”
Since 1990 is later than 1 November 1988, we need to check earlier educational activities from Chunk 2:
“Her primary education, from 1978 to 1983, was at Waimea Heights with her secondary schooling (1984–1987) being
at Taroona High School, and matriculation (1988–1989) at Hobart College. ”
Conclusion
Therefore, 15 years and 6 months before her marriage in 2004, Mary was finishing her matriculation at Hobart College in
Tasmania, Australia.
Answer:Mary, Crown Princess of Denmark, was finishing her matriculation at Hobart College in Tasmania, Australia, 15
years and 6 months before she was married to Frederik, Crown Prince of Denmark.
Table 13: Case study 3
17

Prompt: Temporal Reasoning Types
You are an expert in Temporal Question Answering. Your task is to classify a given question into one of the following
temporal reasoning types:
T1: Time-Anchored (Point or Range-based)
- Definition: The temporal constraint is a FIXED timestamp (Year, Month, or Day).
- Includes: "In 2001", "Since 2001", "Before/After May 2001".
- Rule: NO other events or numerical durations (e.g., "5 years") are involved.
- Example: "Which position did Brian Smith hold after May 2001?"
T2: Explicit Interval
- Definition: The query specifies a fixed START and END point.
- Keywords: "Between X and Y", "From X to Y".
- Example: Which team did Attaphol Buspakom play for between May 1989 and Oct 1990?"
T3: Numerical Offset from Timestamp (Time-to-Time)
- Definition: A specific duration (X years, Y months) applied to a FIXED timestamp.
- Logic: [Fixed Date] +/- [Duration].
- Rule: Does NOT reference any specific event as the base.
- Example: "Who did A work for 8 years before April 1926?"
eg. "Which employer did Gustav Ludwig Hertz work for 8 years and 2 months before April 1926?",
eg. Which political party did Jozo Radoš belong to 13 years and 2 months before October 2017?
eg. Which political party did Marcel Dassault belong to 39 years and 1 months after February 1946?
T4: Event-to-Event Relation (Purely Relative)
- Definition: One event’s time is defined relative to another event WITHOUT numerical offsets.
- Keywords: "Before he joined X", "After her term at Y".
- Rule: NO numbers/durations (years/months) allowed.
- Example: "Which party did Jüri Adams belong to after he was a member of Isamaa?"
eg.Where was Beate Meinl-Reisinger educated before he/she was the member of NEOS – The New Austria?",
eg. Which political party did Jüri Adams belong to after he/she was the member of Isamaa?
eg. Which employers did Marian Salzman work for when he/she held the position of chief strategy officer?
T5: Multi-hop Reasoning (Event + Offset/Multi-step)
- Definition: Complex reasoning combining events with durations OR linking multiple events.
- Logic: [Event A] +/- [Duration] relative to [Event B].
- Rule: If you see a "Duration" (e.g., 2 years) AND an "Event" (e.g., "worked at X"), it MUST be T5.
- Example: "Which employer did A work for 3 years after he worked for University B?"
eg. Which employer did Manuel García Velarde work for 3 years and 11 months after he/she worked for Université libre de
Bruxelles?",
eg. "Which employer did Selig Brodetsky work for 44 years and 6 months after he/she studied at Central Foundation Boys’
School?"
eg. Where was Jessica Stegrud educated 29 years before he/she held the position of member of the Swedish Riksdag?
Output JSON schema:
{{
"type": "T1" | "T2" | "T3" | "T4" | "T5"
}}
Instructions:
- Return ONLY valid JSON
- Do NOT explain your reasoning
- Do NOT infer events or time intervals
- Only judge relevance of relations
Question: {question}
Figure 6: Prompt template used for temporal reasoning type classification.
18

Prompt specification for interval event unit extraction
Role.You are an information-extraction assistant for building dynamic knowledge graphs. Your task is to extractevery
Interval Event Unit (IEU)that can be directly retrieved from the given text.
An IEU is acomplete factual statement(one clause or several tightly linked clauses) that: (i) occurs at aspecific time
point or interval; (ii) hashigh information content; and (iii) contains anexplicit subject with no pronouns.
1. Temporal Reasoning Protocol
1.1 Reference-Time Stack. Maintain a reference-time stack by pushing every explicit absolute date (Year / Year–Month /
Year–Month–Day). The top of the stack serves as the default anchor for resolving relative time expressions.
1.2 Priority Parsing (apply before other reasoning). Prioritize the following patterns: (i) bracketed dates: (YYYY),
(YYYY–MM), (YYYY–MM–DD); (ii) hyphenated intervals: 1995–1998, 2003–05; (iii) parenthetical intervals: (1990–1995), (from
2000 to 2005); (iv) sentence-ending standalone years (e.g., “established in 1995”).
1.3 Relative and Vague Time Resolution. Resolve ambiguous time expressions using the following precedence: (1) immediately
preceding dated event; (2) earlier explicit date in the same sentence or paragraph; (3) document metadata or header timeline;
(4) paragraph-level inheritance from neighboring sentences.
Time Interval Conversion (YYYY–MM–DD Format).All time expressions must be normalized into a closed interval[t start, tend].
•Exact day (YYYY–MM–DD):t start=tend, FuzzyFlag=false.
•Exact month (YYYY–MM):first day to last day of the month, FuzzyFlag=false.
•Exact year (YYYY):YYYY-01-01 to YYYY-12-31; FuzzyFlag=true for point events (e.g., appointment), false for states or
durations.
•Explicit interval (e.g., YYYY–YYYY):expanded to full granularity, FuzzyFlag=false.
•“Until X”:unknown start, end at X, FuzzyFlag=true.
•Relative dates (e.g., “two days later”):resolved via calendar arithmetic, FuzzyFlag=false.
Fuzzy Time Expressions. Expressions such as “early YYYY”, “mid YYYY”, “late YYYY”, or “around YYYY” must always be marked
with FuzzyFlag=true and mapped to their canonical sub-intervals.
2. Information-Value FilterEach candidate sentence is scored from 0 to 4, receiving one point for each satisfied criterion:
•Specific actor:contains a named entity or definite noun phrase uniquely identifying the subject.
•Action or change:describes an action, role change, appointment, or bounded continuation.
•Result or magnitude:includes quantitative detail, outcome, or measurable consequence.
•Temporal anchoring:time is resolvable to at least month-level precision or bounded by clear intervals.
Retention Rule. Keep the sentence if the score≥1, or if it matches explicit role-duration patterns (e.g., “served as
. . . from . . . to . . . ”), contains clear temporal markers, or describes a continued state with explicit boundaries. If
temporal anchoring is missing, paragraph-level inheritance must be attempted; otherwise, discard the sentence.
3. Mandatory Subject RulesEvery extracted IEU must contain a clear, explicit subject. Use the most complete entity
form available (full names, official titles, full organization names).No pronouns(e.g., “he”, “she”, “they”, “it”) are
permitted. Scan the entire document, including titles, to recover complete entity names.
4. Sentence Formation GuidelinesThe sentence field must be self-contained, preserve the original time expression, and
contain no pronouns. The optional context field may include up to 80 tokens of supplementary information drawn only from the
same text chunk. Never merge information across different time expressions; sentences containing multiple distinct times
must be split into separate IEUs.
5. Output FormatReturnONLYa valid JSON object. No explanations, no additional text.JSON Schema.
{ "events": [ { "event_id": "E1", "sentence": "<explicit-subject factual sentence>", "context": "<optional,
≤80 tokens>", "t_start": "YYYY-MM-DD | unknown", "t_end": "YYYY-MM-DD | unknown", "FuzzyFlag": true | false
} ] }
If no valid IEU can be extracted, or if any subject cannot be fully resolved (e.g., remaining pronouns), return:
{ "events": [] }
Example.Input Text. Professor Lin began directing the Quantum Materials Laboratory in early 2018. From 2019 to 2021, the
laboratory developed three prototype superconducting chips. Two months after completing the third prototype, Professor Lin
signed a collaboration agreement with Tsinghua University. Later that year, the joint team published preliminary findings,
though the exact month was not specified. Expected Output (JSON).
{ "events": [ { "event_id": "E1", "sentence": "Professor Lin began directing the Quantum Materials Laboratory in
early 2018.", "context": "", "t_start": "2018-01-01", "t_end": "2018-04-30", "FuzzyFlag": true }, { "event_id":
"E2", "sentence": "The Quantum Materials Laboratory developed three prototype superconducting chips from 2019
to 2021.", "context": "", "t_start": "2019-01-01", "t_end": "2021-12-31", "FuzzyFlag": false }, { "event_id":
"E3", "sentence": "Professor Lin signed a collaboration agreement with Tsinghua University two months after
completing the third prototype.", "context": "The third prototype was completed within 2021.", "t_start":
"2022-02-01", "t_end": "2022-02-28", "FuzzyFlag": true }, { "event_id": "E4", "sentence": "The joint team
published preliminary findings later that year.", "context": "The publication occurred after the collaboration
agreement.", "t_start": "2022-09-01", "t_end": "2022-12-31", "FuzzyFlag": true } ] }
Notice.
•If you are uncertain about extracting any IEU, or cannot fully resolve all pronouns to explicit entities, output {
"events": [] }.
•Donotreveal internal scoring, heuristics, or reasoning steps.
•Sentences should remain maximally complete and precise to support direct downstream retrieval as event evidence.
Real Data Interface.Text: {input_text}
Figure 7: Prompt template used for interval event unit extraction.
19

Prompt: Strict IEU-level Deduplication
Role. You are a Strict Data Auditor for IEU (Interval Event Unit) deduplication. This task is ONLY about filtering
redundant records. This is NOT summarization, NOT aggregation, and NOT timeline construction.
The “Atomic IEU” Rule (CRITICAL). A valid IEU must represent exactly one atomic fact:
- One subject
- One specific role or action
- One single continuous time interval or time point
Task. Analyze {len(events)} IEUs and determine whether any subset refers to theexact same real-world occurrence.
- If events are distinct (even if related),do not merge.
- Only mergeredundant mentions of the same atomic fact.
Strict Prohibitions (Do Not Merge If): - Organization mismatch: different institutions imply distinct events (e.g., “Yale
University” vs. “Harvard University”).
- Sequential chaining: do not merge consecutive career stages.
- Role or title change: e.g., “Assistant Professor” vs. “Dean”.
- Information loss: merging would obscure a change in location or institution.
- Temporal chaining: do not construct a career summary.
- Multiple time references: a sentence must not mention more than one time span.
- Time gaps: if there is a temporal gap, events must remain separate.
Allowed Merge (Only If):
The IEUs are redundant mentions of the same single occurrence at the same organization and during the same continuous
period.
- Example: “A played for AC Milan in 2010” + “A contract at AC Milan from 2010 to 2012”→merge into a single event
spanning 2010–2012.
Merging Rules for Sentence and Time:
- Sentence: must be canonical, atomic, and contain no “and”.
- Precision first: prefer Day>Month>Year.
- No expansion: the merged interval must not exceed the real duration of a single appointment or stay.
Notes.
- merged_event_ids must include only the merged IEUs.
- All other IEUs are implicitly preserved without modification.
Input IEUs:
{events_list_str}
Output Format (Strict). Returnonlya valid JSON object. Do not include explanations.
If no events should be merged:
{
"if_merge": false
}
If one mergeable subset is found:
{
"if_merge": true,
"merged_event_ids": ["event_global_id1", "event_global_id2"],
"sentence": "Single atomic sentence with no conjunctions.",
"context": "Integrated context from all source IEUs.",
"t_start": "YYYY-MM-DD",
"t_end": "YYYY-MM-DD",
"FuzzyFlag": true
}
Figure 8: Prompt template used for strict IEU-level deduplication and redundancy removal.
20

Prompt: Temporal Interval Tightening and Logical Inference
Role
You are aHigh-Precision Temporal Auditor. Your role is to performTemporal Interval Tightening and Logical Inferenceon a
set of semantically linked Interval Event Units (IEUs).
Task Goal
1.Tightening: Identify IEUs marked as FuzzyFlag: true and refine them into precise intervals using neighboring evidence.
2.Inference: Reason about unknown or fuzzy dates by leveraging thecausal or sequential logicwithin the connected component.
Core Constraints
•Absolute Accuracy: Every update must be anchored in the provided text. NO world knowledge.
•Atomic Preservation: Do NOT merge, rewrite, or reinterpret events. Only update t_start, t_end, and FuzzyFlag.
•Directional Logic: A refinement must narrow the uncertainty, never expand it.
Time Interval Conversion (YYYY-MM-DD Format)
All time expressions must be converted into a precise [t_start, t_end] interval.
Rule Expression Type t_start t_end FuzzyFlag Notes
A Exact Day (YYYY-MM-DD) Same Day Same Day false
B Exact Month (YYYY-MM) 1st of Month Last of Month false
C Exact Year (YYYY) YYYY-01-01 YYYY-12-31 See below
D Explicit Interval Start Boundary End Boundary false Full granularity
E “Until X” (Duration End) unknown Date X true Start not explicit
G Relative Date Calendar Arith. Calendar Arith. false Anchored precisely
FuzzyFlag Specifics:
•Year (C):true for point events (join, begin, appointment); false for states/durations (served during 1999).
•Fuzzy Expressions (F):Always true.
•Other True Conditions:Time is inferred, approximated, expanded by default rules, or based on a fuzzy anchor.
Fuzzy Time & Interval Table
Expression t_start t_end FuzzyFlag
Year–year YYYY1-01-01 YYYY2-12-31 false
Month–month YYYY1-MM1-01 YYYY2-MM2-last_day false
Early YYYY YYYY-01-01 YYYY-04-30 true
Mid YYYY YYYY-05-01 YYYY-08-31 true
Late YYYY YYYY-09-01 YYYY-12-31 true
Around YYYY YYYY-01-01 YYYY-12-31 true
Early YYYY-MM YYYY-MM-01 YYYY-MM-10 true
Mid YYYY-MM YYYY-MM-11 YYYY-MM-20 true
Late YYYY-MM YYYY-MM-21 YYYY-MM-last true
Strict Prohibitions (Zero Tolerance)
•NO Hallucination: Do not use world knowledge.
•NO Merging: Do not combine two sentences into one.
•NO Speculation: If logic does not strictly dictate a date, keep FuzzyFlag: true.
Input Data (Connected Component)
{events_list_str}
Output Instructions (JSON ONLY)
If no refinement is possible:
{ "if_update": false }
If one or more IEUs can be updated:
{
"if_update": true,
"updates_ieus": [
{
"event_global_id": "EVT000123",
"updated_t_start": "YYYY-MM-DD",
"updated_t_end": "YYYY-MM-DD",
"updated_FuzzyFlag": false,
"reasoning": "Evidence-based justification referencing other IEUs."
}
]
}
Figure 9: Prompt template used for temporal interval tightening and logical temporal inference.
21

Prompt: IA-RAG Answer Generation
You are an AI assistant that answers questions based on both temporal event sequences and relevant text chunks.
TASK
Answer the question using both the provided events and text chunks. The events show temporal relationships, while the text
chunks provide additional context and details.
TEMPORAL RELEVANCE GUIDELINE
Prioritize information from time periods that align with the question’s temporal scope.If a question asks about a specific
time period, focus on events and information that occurred within or near that timeframe.
ANALYSIS APPROACH
1.Query Semantics Understanding: Analyze the semantic intent:
•Existential queries: What exists/happens at a specific time
•Continuity queries: Ongoing states, processes, or relationships
•Boundary queries: Beginnings, endings, or transitions
•Aggregate queries: Synthesis across multiple time points
2.Temporal Scope Identification: Identify exact time constraints:
•Subject Consistency Check: Verify the event’s subject matches the question’s focus
•Extract specific dates, time periods, or temporal references
•Mark temporal qualifiers (e.g., “during”, “between”, “before”, “after”)
3.Evidence Time Filtering: Before using any event or chunk:
•Assess temporal relevance: Evaluate whether evidence falls within or near the question’s time scope
•Consider contextual value: Include nearby time periods if essential background
•Prioritize temporal proximity: Give preference to evidence closer to the target time
4.Event Analysis: Review the temporal sequence of events:
•Temporal Ordering: Analyze chronological relationships
•State Persistence and Change: Persistent states / Instantaneous events / Processes / Transitions
•Office/Role Incumbency Reasoning: Treat holding an office as a persistent state; if no explicit end event is found
before the query time, assume the holder remains in position
5.Entity-Event Relationships: Analyze connections:
•Agent: Who performs or causes an action
•Patient: Who/what is affected
•Locative: Where something happens
•Attributive: Properties or characteristics at specific times
6.Chunk Analysis: Extract relevant information from text chunks
7.Cross-Reference: Connect information between events and chunks
RESPONSE GUIDELINES
•TEMPORAL RELEVANCE ASSESSMENT: Assess whether evidence timestamps are relevant
•Reference specific events (e.g., “Event #3”) or chunks when supporting your answer
•Apply temporal logic: if an event occurred at T1 and no change is mentioned by T2, assume continuity
•If limited information exists: state temporal limitations, then infer based on: (1) Career Continuity, (2) Role Persistence,
(3) Institutional Affiliation — label inferences clearly
•Start with the direct answer followed by justification citing key events
•If based on inference, state: “Based on inference from surrounding evidence: [answer]”
QUESTION
{question}
EVENTS
{events_data}
TEXT CHUNKS
{chunks_data}
If there is any conflict between EVENTS and TEXT CHUNKS,TEXT CHUNKS take precedenceand must be trusted as the authoritative
source.
ANSWER
Figure 10: Prompt template used for IA-RAG answer generation based on retrieved temporal evidence.
22