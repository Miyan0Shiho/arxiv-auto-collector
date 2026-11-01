# TCC-RAG: Temporal Causal Chain-Guided RAG for Temporal Knowledge Graph Question Answering

**Authors**: Xinxun Zhang, Yuxiang Wang, Shanshan Feng, Jun Zhang, Tieyun Qian, Binhang Yuan, Kai Yin, Xiao Yan, Jiawei Jiang

**Published**: 2026-10-28 07:03:44

**PDF URL**: [https://arxiv.org/pdf/2610.38366v1](https://arxiv.org/pdf/2610.38366v1)

## Abstract
Recent advances in large language models (LLMs) have spurred growing interest in temporal knowledge graph question answering (TKGQA). Most existing methods rely on semantic retrieval augmented generation (RAG), which suffers from fundamental limitations in complex temporal reasoning. On the one hand, semantic-based RAG methods struggle to capture the multi-hop causal knowledge required for complex temporal reasoning. On the other hand, these methods introduce excessive redundant knowledge, trapping LLMs in spurious reasoning chains that potentially result in erroneous answers. To address these problems, we propose a Temporal Causal Chain-guided RAG (TCC-RAG) to retrieve a concise yet sufficient set of causal knowledge to support complex temporal reasoning. Specifically, we reformulate the temporal knowledge graph into a multi-view causal cue graph, with causal cues inducing connections that explicitly encode temporal evolution and causal dynamics. Based on the causal cue graph, we construct a query-specific temporal causal chain to guide the retrieval process, enabling the model to focus on a concise set of causally relevant knowledge. Finally, we introduce a causal evidence filter to refine the retrieved knowledge, ensuring its causal consistency with the underlying temporal logic of the query. Experiments show that TCC-RAG achieves substantial performance gains of 10.72% and 8.40% on two TKGQA benchmarks, and delivers a remarkable 62.97% improvement for multi-step temporal reasoning.

## Full Text


<!-- PDF content starts -->

TCC-RAG: Temporal Causal Chain–Guided RAG
for Temporal Knowledge Graph Question
Answering
Xinxun Zhang1*, Yuxiang Wang1*, Shanshan Feng1, Jun Zhang2, Tieyun Qian1,
Binhang Yuan3, Kai Yin1†, Xiao Yan1, Jiawei Jiang1,4†
1School of Computer Science, Wuhan University2Tencent Inc.
3Hong Kong University of Science and Technology
4School of Computer Science, Central China Normal University
{xxzhangstu,nai.yxwang,victor fengss,qty}@whu.edu.cn; neoxzhang@tencent.com
biyuan@ust.hk;{yinkai,yanxiaosunny,jiawei.jiang}@whu.edu.cn
Abstract—Recent advances in large language models (LLMs)
have spurred growing interest in temporal knowledge graph
question answering (TKGQA). Most existing methods rely on
semantic retrieval augmented generation (RAG), which suffers
from fundamental limitations in complex temporal reasoning. On
the one hand, semantic-based RAG methods struggle to capture
the multi-hop causal knowledge required for complex temporal
reasoning. On the other hand, these methods introduce excessive
redundant knowledge, trapping LLMs in spurious reasoning
chains that potentially result in erroneous answers. To address
these problems, we propose a Temporal Causal Chain–guided
RAG (TCC-RAG) to retrieve a concise yet sufficient set of causal
knowledge to support complex temporal reasoning. Specifically,
we reformulate the temporal knowledge graph into a multi-view
causal cue graph, with causal cues inducing connections that
explicitly encode temporal evolution and causal dynamics. Based
on the causal cue graph, we construct a query-specific temporal
causal chain to guide the retrieval process, enabling the model to
focus on a concise set of causally relevant knowledge. Finally,
we introduce a causal evidence filter to refine the retrieved
knowledge, ensuring its causal consistency with the underlying
temporal logic of the query. Experiments show that TCC-RAG
achieves substantial performance gains of 10.72% and 8.40% on
two TKGQA benchmarks, and delivers a remarkable 62.97%
improvement for multi-step temporal reasoning.
Index Terms—Temporal Knowledge Graph Question Answer-
ing, Retrieval-Augmented Generation, Causal Cue Graph
I. INTRODUCTION
Large Language Models (LLMs), such as GPT-4 [1] and the
LLaMA series [2], have brought a paradigm shift to artificial
intelligence by demonstrating strong capabilities in semantic
understanding, reasoning, and generalization across a wide
range of tasks [3]–[8]. These advances have made LLMs
increasingly attractive for question answering (QA), where
models are expected to understand natural language queries
and produce accurate, knowledge-grounded responses [4]. To
improve factual reliability, Retrieval-Augmented Generation
(RAG) [9]–[11] has become a widely adopted paradigm for
QA systems. By retrieving relevant knowledge from external
*Equal contribution.†Corresponding authors.
(a) Semantic Based RAG 
Danish Ministry Make a visit Iraq in 2006 -01-05
Jack Straw Make a visit Iraq in 2006 -01-06
Barack Obama Make a visit Iraq in 2006 -01-07 
Iraq Host a visit Barack Obama in 2006 -01-07
Barack Obama Make a visit Iraq in 2006 -01-08
…
Spurious Reasoning Chain:
Danish Ministry visit Iraq in 2006 -01-05, based 
on the retrieved facts, the first visit Iraq is 
Barack Obama in 2006 -01-07. 
(b) Temporal Causal Chain –Guided RAG 
Iraq
Denm
arkJack 
Straw
Temporal Causal Reasoning Chain:
Danish Ministry visit Iraq in 2006 -01-05, based 
on the retrieved facts, the first visit Iraq is 
Jack Straw in 2006 -01-06. 
Iraq
Denm
arkJack 
StrawBarack
ObamaEvan
BayhPeter
pace
Causal Cue -Graph Temporal Causal ChainafterQ: After the Danish Ministry of Defence  and 
Security, who was the first to visit Iraq?
❶Causal Facts
❷Noise FactsMissin gFig. 1: The limitations of semantic-based RAG methods and
design of our temporal causal chain-guided RAG approach.
sources and incorporating it into the LLM context, RAG ef-
fectively supplements LLMs and improves factual grounding,
particularly in knowledge-intensive reasoning tasks.
Temporal Knowledge Graph Question Answering
(TKGQA) is an important QA task that requires reasoning over
knowledge that evolves over time [12], [13]. Early TKGQA
methods primarily relied on embedding-based approaches
[14]–[16], which learn representations of temporal entities
and relations to support inference over temporal knowledge
graphs. More recently, with the rapid development of LLMs,
researchers have begun to explore LLM-based TKGQA
frameworks that leverage the reasoning ability of LLMs
arXiv:2610.38366v1  [cs.CL]  28 Oct 2026

together with external temporal knowledge [17]. Most of
these methods adopt semantic retrieval-augmented generation,
where temporally relevant facts are retrieved based on
semantic similarity and provided to the LLM for reasoning
[18]–[22]. However, such semantic RAG pipelines struggle
to support complex multi-step temporal reasoning, where
answering a query requires reconstructing temporally ordered
reasoning chains across multiple events. In particular, existing
methods suffer from two fundamental limitations in complex
temporal reasoning scenarios, as illustrated in Figure 1(a).
❶Missing Causal Facts.Semantic-based RAG methods treat
knowledge as isolated and unordered text units, ignoring the
temporal evolution inherent in the data. Complex multi-step
reasoning questions usually require multi-hop causal knowl-
edge to reconstruct the correct temporal reasoning process.
However, multi-hop causal knowledge that is causally es-
sential but only partially relevant to the query semantics
may be overlooked, preventing LLMs from capturing the
true temporal causal chain in multi-step temporal reasoning
scenarios. As illustrated in Figure 1(a), for the multi-step
temporal reasoning question”After the Danish Ministry of
Defence and Security, who was the first to visit Iraq?”,
semantic-based retrieval fails to retrieve the causally neces-
sary fact”Jack Straw made a visit to Iraq on 2006-01-06”
as it has limited semantic overlap with the query despite
being causally essential for multi-step temporal reasoning.
❷Introducing Noise Facts.Semantic-based RAG methods re-
trieve a large number of knowledge units through similarity
matching, most of which are redundant, while only a small
portion is truly causally relevant to the query. The presence
of enormous noisy knowledge can lead LLMs to follow
spurious reasoning chains, which may ultimately induce
incorrect answers. As illustrated in Figure 1(a), semantic-
based RAG methods return facts that are semantically re-
lated but causally irrelevant, such as”Iraq hosted a visit
by Barack Obama on 2006-01-07”and”Barack Obama
made a visit to Iraq on 2006-01-08”. The presence of
these two causally irrelevant facts aboutBarack Obama
may cause the LLM to follow spurious reasoning chains
based onBarack Obama’s visits when performing temporal
reasoning, leading to incorrect responses.
Our Solution: TCC-RAG.To address the aforementioned
limitations, we propose Temporal Causal Chain–guided
Retrieval-Augmented Generation (TCC-RAG), a novel frame-
work that retrieves a concise yet sufficient set of causal
knowledge to support complex temporal reasoning. Specifi-
cally, we reformulate temporal knowledge facts into a multi-
view causal cue graph, where temporal evolution is explicitly
encoded as causal cues across one event graph and two
trie-structured indices. This transformation exposes temporal
evolution as explicit causal logic between events, capturing
structural dynamics that plain text semantics cannot. More-
over, we introduce a temporal causal chain-guided retrieval
strategy that progressively parses queries and retrieves only
the causally relevant facts along the routed view of the causalcue graph. This design enables the retrieval process to focus on
a concise yet sufficient causal knowledge, effectively avoiding
the introduction of redundant knowledge that misleads LLMs
into incorrect reasoning paths. Finally, to further ensure the
causal consistency of the retrieved knowledge, we propose a
causal evidence filter that distills the retrieved evidence into
a minimal causally sufficient subset through two deterministic
steps: equivalence-class deduplication and sufficiency-driven
selection. The filter yields a purified evidence set that provides
more faithful support for temporal reasoning.
We conduct a comprehensive evaluation of TCC-RAG on
two widely used TKGQA benchmarks against over 10 base-
lines, covering diverse temporal reasoning scenarios. TCC-
RAG consistently outperforms all baselines, achieving overall
improvements of 10.72% and 8.40% in Hits@1, respectively.
Notably, it yields a substantial improvement of 62.97% on
complex multi-step temporal reasoning, demonstrating its
strong capability in complex temporal reasoning. Ablation
studies and case studies further validate the effectiveness
of each component and the quality of the retrieved causal
evidence. Efficiency analysis shows that TCC-RAG reduces
token consumption by over 57% compared with the strongest
baseline, confirming its practical efficiency and scalability.
The main contributions of this work are as follows:
•We identify the limitations of existing semantic-based RAG
methods for TKGQA, including the absence of causal facts
and the introduction of enormous noisy facts, which hinder
complex multi-step temporal reasoning.
•We propose TCC-RAG, which organizes temporal knowl-
edge into a multi-view causal cue graph to explicitly encode
temporal evolution dynamics as structured causal cues.
•We design a temporal causal chain-guided retrieval strategy
with a causal evidence filter to retrieve and distill relevant
facts into causal evidence for complex temporal reasoning.
II. RELATEDWORK
Retrieval-Augmented Generation.Retrieval-Augmented
Generation (RAG) enhances large language models by
retrieving external knowledge to support reasoning and
generation [9]–[11], [23]. A typical RAG framework consists
of a retriever that identifies relevant documents or facts and a
generator that produces answers conditioned on the retrieved
evidence. For instance, [24] conduct a comprehensive
benchmark of LLMs in RAG settings, revealing that retrieval
noise and context length are key bottlenecks for faithful
generation. HippoRAG [25] draws inspiration from human
memory consolidation to build a long-term associative
memory structure, enabling multi-hop retrieval by linking
retrieved passages through a hippocampal-like index. These
works highlight that the quality and structure of retrieved
evidence are critical to downstream reasoning performance.
GraphRAG has attracted increasing attention due to its
ability to model structured knowledge and capture multi-hop
contextual dependencies [26]–[29]. Instead of retrieving inde-
pendent text chunks, GraphRAG methods organize knowledge

as graphs and perform structure-aware retrieval. For example,
Think-on-Graph (ToG) [27] iteratively explores knowledge
graph paths via LLM-guided beam search, retrieving multi-
hop reasoning chains in a depth-first manner. GNN-RAG [28]
combines graph neural networks with RAG by first extracting
candidate answer entities through GNN reasoning and then
retrieving shortest paths as evidence for LLM generation.
Existing GraphRAG approaches can generally be categorized
into subgraph retrieval methods [30] and path-based reasoning
methods [31], both of which aim to leverage graph structure
to improve reasoning quality. However, how GraphRAG can
effectively support complex temporal reasoning, especially
multi-step temporal reasoning, remains largely underexplored.
Temporal Knowledge Graph Question Answering.Tempo-
ral Knowledge Graph Question Answering, a crucial paradigm
for modeling evolving knowledge, has recently garnered
significant research attention [12], [13]. Existing TKGQA
methods can be broadly categorized into embedding-based
approaches and LLM-based approaches [17].
Embedding-based methods typically learn representations
of temporal entities and relations and perform reasoning in
the embedding space. CronQA [32] encodes temporal con-
straints directly into entity and relation embeddings via a
time-aware scoring function, enabling the model to answer
questions anchored to specific timestamps. TempoQR [16]
extends this by grounding question representations into the
temporal knowledge graph through entity and time mention
linking, achieving stronger alignment between query semantics
and temporal facts. Other approaches rely on graph neural
networks to model temporal relational structures [33]–[36],
capturing local and global structural dependencies within
the knowledge graph. While effective on simpler temporal
queries, these embedding-based methods generally struggle to
generalize to complex multi-step temporal reasoning scenarios.
More recently, LLM-based TKGQA methods have emerged
as a flexible alternative [17]. TimeR4 [18] proposes a time-
aware retrieval strategy that incorporates temporal signals into
the retrieval process, providing LLMs with temporally fil-
tered evidence for question answering. RTQA [19] introduces
a recursive thinking mechanism that decomposes complex
temporal questions into sub-questions and iteratively retrieves
supporting facts, improving multi-step reasoning performance.
Despite these advances, most existing LLM-based methods
still rely on semantic similarity-based retrieval [18], [19], [21],
which fails to capture the multi-hop causal knowledge required
for complex temporal reasoning, motivating our causal chain-
guided approach.
III. PRELIMINARIES
Temporal Knowledge Graph.A temporal knowledge graph
(TKG) is defined as a structured knowledge base that repre-
sents relational facts evolving over time. Formally, a TKG is
denoted as:G= (E,R,T,F),whereEdenotes the entity
set,Rthe relation set,Tthe temporal domain, andFthe
complete temporal fact set. Each temporal fact is representedas a quadruple(h, r, t, τ)∈ F,whereh, t∈ Eare the head and
tail entities,r∈ Ris the relation type, andτ∈ Tdenotes the
temporal annotation indicating when the relation holds. The
temporal annotationτcan be a discrete timestamp, a time
interval, or an ordered temporal index.
Temporal Knowledge Graph Question Answering.Given a
temporal knowledge graphGand a natural language question
Q, the goal of temporal knowledge graph question answering
(TKGQA) is to infer the correct answerAgrounded inGwhile
respecting temporal constraints and event evolution patterns.
Formally, the task is to learn a mapping:f: (Q,G)→ A,
whereA ∈ E,A ∈ T, orA ∈ E × T, depending on the
question type.
Complex Temporal Reasoning.Unlike static KGQA,
TKGQA often requires multi-step reasoning over time. Specif-
ically, the answeramay depend on a sequence of tempo-
rally ordered facts:π= (h 1, r1, t1, τ1), . . . ,(h n, rn, tn, τn),
where the timestamps satisfy temporal ordering constraints:
τ1≺τ 2≺ ··· ≺τ n,and collectively provide sufficient
evidence to derive the answer:π|=a| Q.
Temporal Causal Cue.Acausal cueis a query-conditioned
temporal dependency signal that constrains how query-
relevant events are ordered or selected. We distinguish two
types of cues. The first type is theordering cuec∈
{before,after,same}, which constrains the temporal re-
lation between an event and a boundary timestampτ b. Given a
queryQwith ordering cueˆc, an evente= (h, r, t, τ)iscue-
consistentwithQif its timestampτsatisfies the constraint
imposed byˆcrelative toτ b:
τ≺ˆcτb⇔

τ < τ b,ˆc=before,
τ > τ b,ˆc=after,
τ=τ b,ˆc=same.(1)
The second type is theselection cuec∈ {first,last},
which selects the boundary event from a temporally ordered
set of cue-consistent events. Specifically,firstselects the
earliest event andlastselects the latest one. Selection
cues are typically composed with ordering cues in multi-step
queries, for example“Before Mexico, which country did China
visit last?”, wherebeforefirst restricts the candidate events
andlastthen selects the answer among them.
In this work, the term causal is used in an operational
sense for temporal question answering rather than in the
intervention-based sense of causal inference. We regard tem-
porally ordered event dependencies that are relevant to a query
as causal signals for reasoning.
IV. METHODOLOGY
TCC-RAG consists of three core components. First, we
construct a multi-view causal cue graph to explicitly encode
temporal causal relations among events, transforming implicit
temporal information into structured causal cues organized
over a graph and two indices. Second, we design a query-
aware temporal causal chain retrieval module that progres-
sively parses the query and traces causally relevant evidence

𝐹!
a) Multi-View Causal Cue GraphChina make a visit to Vietnam in 2005-01-09China make a visit to Iran in 2005-03-01China make a visit to Mexico in 2005-03-01China make a visit to Mexico in 2005-06-25Vietnam make a visit to China in 2005-06-22Mexico make a visit to China in 2005-09-15…ChinaVietnamIranMexico𝑡!𝑡"𝑡#𝑡#,𝑡$𝑡%TKSTemporal Event Graph𝒢	'ChinaVietnamIranMexicoSubject Causal Cue Index𝒟(VietnamMexicoObject Causal Cue Index𝒟)
China𝑡"𝑡#,𝑡$𝑡#𝑡!𝑡%“After” Cue“Before” Cue“Same” Cuemake a visit
b) Temporal Causal Chain-Guided Retrieval
Q: Before Mexico, which country did China visit last?
NERLLM
𝐄*+,
𝐄--.
ConsistentInconsistent 
ConsistencyCheck [Mexico, China]
FinalEntity 
Extract Relation  
𝑟~
roleExtract Causal Cue  
[Before, Last][visit,  subject][Mexico,China]       ,  visit [Before Last],subject,        object
Dynamic Query Routingrole=         ans=
structured tuple
ChinaIranMexico𝑡$𝑡#…China make a visit to Iran in 2005-03-01China make a visit to Mexico in 2005-06-25Last
Temporal causal chain
c) Causal Evidence Filter
Answer the question based on retrieved causal facts
d) Causal Chain-Guided ReasoningChina visit Vietnam in 2005-01-09China visit Iran in 2005-03-01 China visit Iran in 2005-03-01China visit Mexico in 2005-06-25 China visit Brazil in 2005-09-15China visit Japan in 2005-11-20repeatCausal-Equivalent DeduplicationCausal Sufficiency Selection
noisyChina visit Vietnam in 2005-01-09China visit Iran in 2005-03-01 China visit Mexico in 2005-06-25 boundary event: China visit Mexico in 2005-06-25; cue: before, last
𝐹!Fig. 2: The framework of TCC-RAG. (a) Temporal facts are organized into a Temporal Event Graph (TEG) and two temporally
sorted indices: the Subject Causal Cue Index (SCCI) and Object Causal Cue Index (OCCI). (b) NER and LLM extraction
ground query entities through matching and consistency checking. Relation, cue, and role parsing then guide view routing
and temporal chain retrieval. (c) The Causal Evidence Filter removes equivalent duplicates and selects a sufficient subset of
query-consistent facts. (d) The LLM reasons over the filtered evidence to generate answers. When structured retrieval yields
no facts, semantic retrieval provides fallback candidates that also pass through the filter.
over the routed view, ensuring that only essential causal facts
are retrieved. Finally, we introduce a causal evidence filter
that refines the retrieved evidence into a minimal causally
sufficient subset through equivalence-class deduplication and
sufficiency-driven selection. Together, these components form
a unified, training-free, and plug-and-play framework that
effectively enhances LLM-based complex temporal reasoning.
The overall architecture of TCC-RAG is shown in Figure 2.
A. Multi-View Causal Cue Graph Construction
Different temporal queries exhibit heterogeneous logical
structures and require varying reasoning depths. Simple
queries that involve a known entity pair only require direct
event lookup, while complex multi-step queries anchored on
a single entity demand traversal across temporally ordered
events. A single monolithic structure cannot serve both ef-
ficiently. To this end, we design the causal cue graph as
three complementary views: aTemporal Event Graph(TEG)
for paired-entity queries, aSubject Causal Cue Index(SCCI)
for subject-anchored multi-step queries, and anObject Causal
Cue Index(OCCI) for object-anchored multi-step queries. The
model dynamically routes each query to the most suitableview based on the role of its grounded entities, achieving both
precision and efficiency across diverse query patterns.
Temporal Event Graph.TEG targets paired-entity queries
that ask for events directly connecting two known entities,
e.g.,“When did China negotiate with Iraq?”. For such queries,
retrieval reduces to a direct edge lookup between two nodes,
which is naturally captured by a graph. Formally, we construct
TEG as a multi-relational temporal graph:Gt= (E,F),where
Eis the entity node set (as defined in Section III), and each
edge(h, r, t, τ)∈ Fencodes a temporal fact linking head
hto tailtvia relationrat timestampτ. Given the grounded
subject–object pair(ˆe s,ˆeo), retrieval scans directed edges from
ˆestoˆeoand retains those matching the query relation, costing
O(deg(ˆe s,ˆeo)), wheredegcounts these directed edges.
Subject Causal Cue Index.A more challenging case arises
in multi-step queries anchored on a subject entity, such as
“After Mexico, which country did China visit?”. The answer
requires traversing all events initiated by the subject under
the same relation, ordered along the temporal dimension. A
naive solution is to materialize a causal cue edge between
every pair of temporally related events sharing the same
(subject, relation)key. However, this incursO(d2)storage

TABLE I: Complexity of the multi-view causal cue graph.
|F|: total facts;d max: maximum posting list length;k: number
of cue-consistent events per query.Per-querymeasures index
retrieval with grounded keys and a resolved temporal boundary
(see Section IV-E).
Structure Storage Construction Per-query
TEG (Gt)O(|F|)O(|F|)O(deg)
SCCI (Ds)O(|F|)O(|F|logd max)O(logd max+k)
OCCI (Do)O(|F|)O(|F|logd max)O(logd max+k)
cost per key, whereddenotes the number of events per
(subject, relation)pair, becoming prohibitive in dense entity
neighborhoods.
We observe that traversing along causal cue edges is func-
tionally equivalent to scanning a list of events sorted by
timestamp. Causal cues encode the relative temporal order
between events, and this order is fully recoverable from the
position of each event in the sorted list. Building on this
insight, we design SCCI as an inverted index that encodes
causal cues implicitly through temporal ordering, rather than as
physically materialized edges. Concretely, SCCI is a hash map
keyed by(subject, relation)pairs, with each value storing a
temporally ordered posting list:
Ds: (u, r)7→
e1, e2, . . . , e d
, τ e1≤τe2≤ ··· ≤τ ed.
(2)
The causal cues among events in the posting list are re-
covered on-the-fly: ifτ ei< τej, thenbefore(e i, ej)and
after(e j, ei)hold. Ifτ ei=τej, onlysame(e i, ej)holds.
Each predicate describes its first event relative to its second.
List position alone does not imply strict temporal precedence.
This implicit encoding preserves the full causal cue semantics
defined in Section III, while reducing storage fromO(d2)to
O(d)per key.
The SCCI design delivers two efficiency benefits over a
fully materialized causal cue graph. First, storage is reduced
fromO(d2)toO(d)per key, yieldingO(|F|)total. Second,
a hash lookup locates the posting list, binary search locates
the temporal range, and a scan returns itskcue-consistent
events, costingO(logd max+k). Crucially, this efficiency
comes without sacrificing causal cue semantics: every retrieval
result reachable via materialized edges remains reachable via
posting list scan.
Object Causal Cue Index.Symmetric to SCCI, OCCI handles
multi-step queries anchored on an object entity, e.g.,“Before
the United States, who visited Iraq?”. Following the same
design rationale, we construct OCCI as a hash map keyed by
(relation, object)pairs:
Do: (r, u)7→
e1, e2, . . . , e d
, τ e1≤τe2≤ ··· ≤τ ed.
(3)
The causal cue semantics, storage costO(|F|), and per-
query retrieval costO(logd max+k)are identical to those ofAlgorithm 1Temporal Causal Chain-Guided Retrieval
Require:QueryQ, viewsS= (Gt,Ds,Do), KG vocabularyE cand,
relationsR.
Ensure:EvidenceFcfor answer generation.
1:// Hybrid Entity Grounding
2:M ner←NER(Q)
3:M llm←LLM(Q; promptent)
4:Match both mention sets toE cand using Eq. (4)
5:Obtain ˆEby agreement or arbitration in Eq. (5)
6:// Relation, Cue, and Role Parsing
7:ˆc←RuleMatch(Q;P)
8:(˜r,role, z ref)←LLM(Q, ˆE; promptrel)
9:r∗←arg max r∈Rcos(ϕ(˜r), ϕ(r))
10:Inferans(Q)and filled slots fromQandrole
11:Resolveτ bfromQandz ref, using⊥if absent or unresolved
12:// Route to One View and Construct the Chain
13:F Q← ∅
14:ifpair(Q)then
15:F Q←DirectedEdges(Gt,ˆes, r∗,ˆeo)
16:else ifrole(ˆe) = subjectandans(Q) = objectthen
17:F Q← Ds[(ˆe, r∗)]
18:else ifrole(ˆe) = objectandans(Q) = subjectthen
19:F Q← Do[(r∗,ˆe)]
20:end if
21:F Q←TemporalSelect(F Q,ˆc, τ b)
22:ifF Q=∅then
23:F Q←SemanticRetrieve(Q)
24:end if
25:OrderF Qby ascending timestamp
26:Fc←CEF(F Q,Q){Section IV-C}
27:returnFc
SCCI, providing the same efficiency guarantees on the object-
anchored side.
Summary.We denote the multi-view structure byS=
(Gt,Ds,Do), distinct from the original TKGG. Construction
populates the three views in one pass overFand then sorts the
posting lists, takingO(|F|logd max)time andO(|F|)storage.
Table I summarizes the view-specific retrieval costs. Together,
the views preserve causal cue semantics through direct edges
and implicit temporal ordering.
B. Temporal Causal Chain-Guided Retrieval
Given the multi-view causal cue graph, the central challenge
is to ground a natural-language queryQinto the appropriate
index keys and extract the events that participate in the
underlying temporal causal chain. We address this challenge
through a progressive query parsing pipeline that decomposes
parsing into three stages of decreasing difficulty: hybrid entity
grounding, context-aware relation and cue parsing, and dy-
namic query routing. Each stage takes the signals resolved by
the previous one as additional context, so that the search space
narrows along the pipeline and progressively lighter machinery
suffices for the remaining decisions. The structured output
is finally resolved into a temporal causal chain by a chain
construction procedure over the routed view.
Hybrid Entity Grounding.Entity grounding is the most chal-
lenging stage of parsing because it operates on the raw query
with no prior context. Real-world temporal queries frequently

involve aliases, composite references, and role descriptors, all
of which may map to canonical KG entities in non-obvious
ways. A single extractor, whether a NER model or an LLM,
may fail to handle such surface-form variability reliably. To
address this, we adopt a dual-path extraction design with
mutual consistency verification.
For queryQ, NER and the LLM independently extract
mention setsM nerandM llm. Both are matched against the KG
entity vocabularyE cand =E, not a query-derived candidate
set:
Ener=M(M ner,Ecand),
Ellm=M(M llm,Ecand).(4)
HereE nerandE llmcontain matched KG entities, andM(·)
is a hybrid matching function. For each extracted mention,
M(·)first applies semantic matching via cosine similarity in
a pretrained embedding space. If the maximum similarity ex-
ceeds thresholdθ 1, the semantic result is adopted. Otherwise,
it falls back to string-based fuzzy matching.
We then apply a consistency check to produce the final
grounded entity set ˆE:
ˆE=(
Ener,ifE ner=E llm,
LLM(Q,E ner,Ellm; promptcom),otherwise.
(5)
When the two paths agree, the shared result is directly
adopted. When they disagree, the LLM is prompted with both
candidates and the original query to resolve the conflict and
select the correct entity set. This dual-path design enables
self-correction without labeled data or fine-tuning, improving
grounding robustness across diverse and linguistically complex
temporal queries.
Context-Aware Relation and Cue Parsing.With entities
grounded, the second stage extracts two predicate-level sig-
nals, the temporal cueˆcand the canonical relationr∗, which
together determine the target posting list withinDsorDo, or
the target edge withinGt. Because cues and relations exhibit
different parsing complexities, we adopt different extraction
strategies for the two.
For temporal cues, we observe that nearly all temporal
queries express their cue through a small closed set of op-
erator words such asbefore,after,first, andlast,
often in regular surface patterns. We therefore extractˆcvia
deterministic pattern matching on the query surface:
ˆc= RuleMatch(Q;P),(6)
wherePis a small set of cue-pattern rules.
For relations, we leverage the LLM’s predicate-parsing
capability conditioned on the grounded entities ˆE. The LLM
extracts a relation phrase˜r, subject/object role assignments,
and a temporal referencez refwhen present:
(˜r,role, z ref) = LLM(Q, ˆE; promptrel),(7)wherepromptreldenotes the relation-extraction template. The
extracted phrase˜ris then grounded to a canonical KG relation
via embedding similarity over the relation setR:
r∗= arg max
r∈Rcos(ϕ(˜r), ϕ(r)).(8)
The answer slotans(Q)follows from the interrogative and
missing argument in the parsed subject–relation–object struc-
ture. For entity answers, the filled subject or object serves as
the traversal anchorˆe. An entity or date associated with the
temporal cue provides the temporal referencez ref. For time
answers,ˆe sandˆe odenote the filled subject and object. These
roles, together withr∗andˆc, determine routing and temporal
selection.
Dynamic Query Routing.With the parsed signals in hand,
the third stage dynamically maps each query to the appropriate
causal cue graph view based on the query structure. In
particular, a multi-step subject-anchored or object-anchored
query may still contain two grounded entities: one serves
as the traversal anchor, while the other only provides the
temporal reference point that determinesτ b. We therefore
route by the filled argument slot and the answer slot. Let
ans(Q)∈ {subject,object,time}denote the slot being asked,
and letpair(Q)indicate that both subject and object slots are
filled and the query asks for a direct temporal fact. The routing
functionρ:Q 7→ {Gt,Ds,Do}is defined as
ρ(Q) =

Gt,pair(Q) = true,
Ds,role(ˆe) = subject∧ans(Q) = object,
Do,role(ˆe) = object∧ans(Q) = subject.
(9)
For example, in the query“Before Mexico, which country
did China visit last?”, bothChinaandMexicoare grounded.
However,Chinais the subject anchor andMexicoonly de-
termines the boundary timestamp, so the query is routed
to the SCCI dictionaryDsrather than to TEG. The three
branches are mutually exclusive for the supported query
patterns. Reference-event lookup supplies a boundary, not
a second answer traversal. The executor does not perform
arbitrary cross-view, multi-relation joins.
Causal Chain Construction.After routing, we construct the
temporal causal chain by retrieving cue-consistent events from
the routed view. The construction proceeds along two paths
corresponding to the structural property of the view, followed
by a unified temporal selection step.
If the query is routed to TEG, retrieval uses the known
subjectˆe sand objectˆe o. The query-specific candidate setF Q
contains directed edges matchingr∗:
FQ={(h, r, t, τ)∈ F |h= ˆe s, t= ˆe o, r=r∗}.(10)
For SCCI or OCCI, we locate the posting listLkeyed
by(ˆe, r∗)or(r∗,ˆe), respectively. An explicit date directly
suppliesτ b. Otherwise, TEG retrieves the reference event
connecting the anchor and reference entity underr∗. Its

timestamp suppliesτ b. With the boundary resolved, binary
search followed by a range scan selects cue-consistent events:
FQ={e∈L|τ e≺ˆcτb},(11)
where≺ ˆcis defined in Section III. We useτ b=⊥when
the boundary is absent or unresolved. Without an ordering
constraint, the full list is eligible for selection. A missing key
or an unresolved required boundary yieldsF Q=∅.
The unified TEMPORALSELECToperation applies these
temporal constraints, then retains the earliest or latest event(s)
forfirst/last, if requested. Thus, compound cues such
asbefore-lastfirst restrict the range and then select its
temporal extreme. Queries without temporal restrictions retain
all relation-matched candidates. The selected events inF Qare
arranged in temporal order to form the temporal causal chain
π=e 1ˆc− →e 2ˆc− → ···ˆc− →ek,(12)
which serves as the structured causal evidence for downstream
answer generation.
Parsing uses two LLM calls (mention extraction and predi-
cate parsing), plus one arbitration call only when the ground-
ing paths disagree. When structured retrieval returnsF Q=∅,
the retrieval pipeline falls back to semantic similarity search
to obtain alternative candidates. Grounding errors, relation
mismatches, or missing facts can cause such empty results.
Routing alone does not trigger fallback. Candidates from either
retrieval path are passed to CEF before answer generation, as
summarized in Algorithm 1.
C. Causal Evidence Filter
Retrieved evidence can contain redundant or temporally
irrelevant facts. CEF processes the query-specific candidates
FQfrom either structured retrieval or semantic fallback in
two deterministic steps, producing deduplicated evidenceFd
and final filtered evidenceFc. Already concise, cue-consistent
evidence can pass through unchanged, while redundant or cue-
inconsistent candidates are refined before answer generation.
Causal-Equivalent Deduplication.The first kind of noise
comes from facts that are structurally distinct yet carry iden-
tical causal information under the query. We capture this via
a query-conditioned causal signature:
σ(e| Q) = 
h, r, t, κ(τ,ˆc, τ b)
,(13)
whereκ(τ,ˆc, τ b)maps the timestampτto a discrete causal
token. Letc κbe the ordering component ofˆcif present,
otherwise its selection component (or⊥if neither exists).
Compound cues use the ordering component for normalization
and retain the selection component for sufficiency selection.
LetD b(τ)be(τ−τ b).daysfor a resolved boundary, or
τ.toordinal()otherwise. Then
κ(τ,ˆc, τ b) =

⌊Db(τ)/δ⌋, c κ∈ {before,after},
rank(τ;c κ), c κ∈ {first,last},
τ.toordinal(),otherwise.
(14)Prompt for final answer generation
Instruction:
Y ou are a temporal knowledge question answering as-
sistant. Given a list of Historical facts and a Question,
return ONL Y a JSON object in this exact format:
{"answers": [...]}
Rules:
•Return all possible answers and do NOT provide
explanations.
•Every returned entity must appear EXACTL Y as
written in the Historical facts.
•Dates must follow ISO format (YYYY -MM-DD). If
the question asks for:
–year→return YYYY
–month→return YYYY -MM
–day→return YYYY -MM-DD
Fig. 3: The prompt template for causal chain-guided temporal
reasoning based answer generation.
whereδis the temporal bucket width (set to 1 day, aligned
with the finest granularity of MULTITQ), andrank(τ;c κ)is
the ordinal position within the(h, r, t)group, sorted ascending
forfirstor descending forlast. Thus, range cues use
day buckets, whilesameand absent cues preserve day-level
identity. Two facts are causally equivalent underQif and
only if they share the same signatureσ. We partitionF Qinto
equivalence classes underσand retain the earliest-timestamp
fact as the canonical representative of each class, yielding the
deduplicated setFd. This stage runs inO(|F Q|)via hash-
based grouping and is information-lossless: every fact inF Q
has an equivalent representative inFd.
Causal Sufficiency Selection.The second kind of noise comes
from facts that fail to participate in the causal chain required by
the query. Eligible facts satisfy the ordering component when
τbis resolved. When it is unresolved, CEF skips boundary-
dependent tests to avoid unsupported pruning of fallback
evidence. We select a minimal subsetFc⊆Fdby a greedy
procedure that enforces three conditions in order:
(i) Boundary preservation.For afirst/lastselection
component, including compound cues, the algorithm seedsFc
with the earliest/latest eligible fact inFd. Forsamewith a
resolvedτ b, it retains all facts on the anchor day. This ensures
that the pivotal boundary event anchoring the causal chain is
always retained.
(ii) Cue coverage.The algorithm adds the canonical rep-
resentative of each eligible class inFd, using the ordering
component ofˆcwhen present (i.e.,τ < τ bforbefore,
τ > τ bforafter, or the same day forsame, with a resolved
τb). Classes that are cue-inconsistent are discarded. This step
ensures that every distinct causal pattern satisfying the query’s
temporal constraint is represented inFc.
(iii) Chain connectivity.WithFcordered by ascending
timestamp, the algorithm verifies that consecutive facts share

the grounded relationr∗and respect the temporal direction
implied byˆc(i.e., timestamps are non-decreasing). If this
condition is violated, indicating that the selected facts do not
form a coherent causal chain, the algorithm safely falls back
toFdto avoid over-pruning.
Preserving the chronological order supplied by chain con-
struction, selection and connectivity checks takeO(|Fd|)time.
Together with hash-based deduplication, CEF takesO(|F Q|)
time and produces filtered evidenceFc.
D. Causal Chain-Guided Reasoning
With the evidence setFcin hand, we prompt the LLM
to perform temporal reasoning and generate the final answer.
Specifically,Fcis serialized as a structured fact list and
provided to the LLM alongside the original queryQ, enabling
it to follow the temporal causal chain to derive the answer.
The prompt template used for answer generation is shown
in Figure 3. Compared with semantic-based RAG methods,
which feed a large number of loosely relevant facts into the
LLM, TCC-RAG supplies a concise yet causally sufficient
evidence set. This precision reduces reasoning noise and
allows the LLM to focus on the true causal chain, yielding
more accurate answers with substantially fewer facts.
E. Complexity Analysis
Time Complexity.Offline construction costsO(|F|logd max),
dominated by posting-list sorting. Query parsing uses two
LLM calls, plus one optional grounding-arbitration call.
Answer generation adds one further call. Entity and re-
lation matching depend on their vocabulary sizes and are
separate from index retrieval. With grounded keys and
a resolved boundary, TEG lookup costsO(deg(ˆe s,ˆeo)),
while SCCI/OCCI range retrieval costsO(logd max+k).
A reference-event boundary additionally requires a TEG
edge lookup. Ordering initially unsorted candidates costs
O(|F Q|log|F Q|). Posting-list results already inherit temporal
order. CEF then costsO(|F Q|). Table I reports index retrieval
alone, excluding parsing, evidence filtering, generation, and
the semantic backend used only for fallback. Thus, structured
retrieval accesses query-relevant neighborhoods rather than
scanning all|F|facts.
Space Complexity.The three structured views requireO(|F|)
storage: each fact appears once in TEG and once in each
of SCCI and OCCI. These views require no fact-level dense
vectors. Entity/relation embedding caches and the semantic
fallback backend are additional storage, separate from this
structural bound.
F . Limitations
Entity Grounding and Query Parsing Noise.Ambiguous
entity mentions or misaligned relations can produce empty or
incorrect structured evidence even when the query is routed
to a view. Semantic retrieval provides a safety net when the
structured result is empty, but does not guarantee detection of
nonempty, incorrect evidence. Robust grounding and relation
disambiguation remain important challenges.TABLE II: Statistics of question categories in MULTITQ.
Category Train Dev Test
SingleEqual 135,890 18,983 17,311
Before/After 75,340 11,655 11,073
First/Last 72,252 11,097 10,480
MultipleEqual Multi 16,893 3,213 3,207
After First 43,305 6,499 6,266
Before Last 43,107 6,532 6,247
Total 386,787 57,979 54,584
TABLE III: Statistics of question categories in TimeQuestions.
Category Train Dev Test
Explicit 2,724 1,302 1,311
Implicit 651 291 292
Temporal 2,657 1,073 1,067
Ordinal 938 570 567
Total 6,970 3,236 3,237
Sparse Knowledge Graph.The SCCI/OCCI indices gain effi-
ciency when multiple facts share the same(subject, relation)
or(relation, object)key. On sparse KGs, most posting lists
contain a single entry. In such cases, multi-step causal chain
retrieval degenerates to a single fact lookup, and the multi-
view design provides limited benefit. Adapting TCC-RAG to
sparse KG settings is left to future work.
V. EXPERIMENTS
A. Experimental Setup
Datasets.We evaluate the effectiveness of TCC-RAG on
two widely used TKGQA benchmarks: MULTITQ [13] and
TimeQuestions [36], as summarized in Table II and Table III.
MULTITQ is a large-scale TKGQA dataset containing over
500K questions with diverse temporal granularities. The ques-
tions span multiple temporal reasoning patterns, requiring
varying numbers of reasoning steps over temporal knowledge
graphs. Its scale and diversity make it particularly suitable
for evaluating both retrieval accuracy and reasoning robust-
ness under complex temporal settings. TimeQuestions is a
complementary benchmark consisting of approximately 16K
questions, categorized into four distinct temporal reasoning
types. Across both datasets, different question types inherently
require different levels of temporal reasoning, ranging from
single-step temporal lookup to multi-step causal chain reason-
ing, making them well aligned with our evaluation objectives.
Baselines.We compare TCC-RAG with a diverse set of
baselines covering pretrained language models, embedding-
based QA methods, and large language model (LLM)-based
approaches. On the MULTITQ dataset, we evaluate against the
following methods:
•Pretrained Language Models:BERT [14], DistilBERT
[37], and ALBERT [15], which fine-tune pretrained trans-
formers for temporal QA.

TABLE IV: Results of TCC-RAG and baselines on Hits@1 and Hits@10 on MULTITQ. The best and second best results are
marked in bold and underlined, respectively. The bottom row is the relative improvement of TCC-RAG over the best baseline.
ModelHits@1 Hits@10
Overall Question Type Answer Type Overall Question Type Answer Type
Multiple Single Entity Time Multiple Single Entity Time
BERT 0.083 0.061 0.092 0.101 0.040 0.441 0.392 0.461 0.531 0.222
DistilBERT 0.083 0.074 0.087 0.102 0.037 0.482 0.426 0.505 0.591 0.216
ALBERT 0.108 0.086 0.116 0.139 0.032 0.484 0.415 0.512 0.589 0.228
EmbedKGQA 0.206 0.134 0.235 0.290 0.001 0.459 0.439 0.467 0.648 0.001
CronKGQA 0.279 0.134 0.337 0.328 0.156 0.608 0.453 0.671 0.696 0.392
MultiQA 0.293 0.159 0.347 0.349 0.157 0.635 0.519 0.682 0.733 0.396
LLaMA 3.1 8B 0.032 0.008 0.041 0.037 0.019 0.056 0.015 0.071 0.064 0.039
GPT-4o mini 0.064 0.019 0.081 0.071 0.048 0.094 0.030 0.118 0.115 0.048
DeepSeek-Chat 0.104 0.034 0.129 0.115 0.081 0.153 0.049 0.191 0.181 0.090
ARI 0.380 0.210 0.680 0.394 0.344 - - - - -
TimeR40.728 0.335 0.887 0.639 0.945 - - - - -
RTQA 0.765 0.424 0.902 0.692 0.942 0.768 0.427 0.907 0.697 0.942
TCC-RAG
LLaMA 3.1 8B 0.762 0.580 0.836 0.715 0.876 0.823 0.648 0.894 0.789 0.901
GPT-4o mini 0.817 0.646 0.879 0.767 0.923 0.857 0.709 0.911 0.821 0.936
DeepSeek-Chat0.847 0.691 0.911 0.801 0.959 0.876 0.746 0.928 0.839 0.961
Imp (%) 10.72 62.97 1.00 15.75 1.48 14.06 43.74 2.32 14.46 2.02
•Embedding-based Methods:EmbedKGQA [38],
CronKGQA [32], and MultiQA [13], which rely on
learned knowledge graph embeddings for temporal
reasoning.
•LLM-based Methods:LLaMA 3.1 8B [39], GPT-4o-mini
[1], and DeepSeek-Chat [40] answer questions through
prompt-based inference. ARI [21], TimeR4[18], and RTQA
[19] augment LLMs with semantic RAG pipelines.
On the TimeQuestions dataset, we compare against the fol-
lowing categories of baselines:
•KGQA Methods:PullNet [41], GRAFTNet [42], and
Uniqorn [43], which perform reasoning over static or
hybrid knowledge graphs. TKGQA-specific methods in-
clude CronKGQA [32], TempoQR [16], EXAQT [44],
LGQA [34], and TwiRGCN [36].
•LLM-based Methods:LLaMA2 7B [2], ChatGPT [1], Gen-
TKGQA [20], TimeR4[18], and RTQA [19], which leverage
large language models with varying retrieval strategies.
These baselines collectively cover a wide spectrum of
paradigms, enabling a comprehensive evaluation of TCC-RAG
across different temporal reasoning and retrieval settings.
Implementation Details.
•LLM & Model Configuration.For the progressive query
parsing module (Section IV-B), we use a locally hosted
LLaMA 3.1 8B [39] for both entity extraction and relation
extraction. The NER component adopts roberta-large-ner-
english [45] as the surface-form recognition model, and
multi-qa-MiniLM-L6-cos-v1 [46], [47] for cosine similarity
computation between entity mentions and KG entities. For
final answer generation, we evaluate TCC-RAG with three
LLM backbones: DeepSeek-Chat v3.1-20250922 [40], GPT-
4o-mini [1], and LLaMA 3.1 8B [39]. On TimeQues-tions, TCC-RAG is evaluated using DeepSeek-Chat v3.1-
20250922 as the generator. The entity-grounding similarity
threshold is set toθ 1= 0.85. Candidates below this
threshold fall back to fuzzy string matching. The temporal-
bucket widthδin the causal signatureκ(τ,ˆc, τ b)is set to
1day for range cues (before,after). Forsame,first,
andlast,κuses exact-timestamp or ordinal-rank encoding
without bucketing.
•Baseline Setup.OnMULTITQ, results for non-LLM
baselines (BERT/DistilBERT/ALBERT, EmbedKGQA,
CronKGQA, MultiQA) and ARI/RTQA are taken directly
from RTQA [19]. TimeR4is evaluated using its publicly
released checkpoint with its original pretrained LLaMA
backbone. TCC-RAG uses DeepSeek-Chat, GPT-4o-
mini, and LLaMA 3.1 8B through public APIs. On
TimeQuestions, results for non-LLM baselines (PullNet,
Uniqorn, GRAFT-Net, CronKGQA, TempoQR, EXAQT,
TwiRGCN, LGQA) are taken directly from TimeR4[18].
For LLM-based RAG baselines (TimeR4and RTQA), we
re-run them with the same frozen DeepSeek-Chat v3.1-
20250922 generator as TCC-RAG, rather than using
their originally reported numbers. This is because the
TimeQuestions training and test splits have partial overlap,
which can cause data leakage when a fine-tuned backbone
is used, inflating reported scores. By replacing all LLM
generators with the same frozen DeepSeek-Chat v3.1-
20250922, we eliminate backbone confounds and ensure a
fair comparison that isolates each method’s retrieval and
reasoning strategy.
•Hardware & Decoding.All experiments are conducted on
a server equipped with an NVIDIA A6000 GPU and an
Intel Xeon Gold 6330 CPU @ 2.00GHz. All LLM calls
use deterministic decoding with temperature= 0and top-

p= 0.9to ensure reproducibility.
B. Overall Performance
Results on MULTITQ.Table IV reports the experimen-
tal results on the MULTITQ dataset. Overall, TCC-RAG
consistently outperforms all baseline methods across evalu-
ation metrics. In particular, compared with the state-of-the-art
RTQA, TCC-RAG achieves relative improvements of 10.72%
and 14.06% on Hits@1 and Hits@10, respectively. A closer
inspection reveals that the performance gains are primarily
driven by theMultiplequestion category, where TCC-RAG
achieves improvements of 62.97% on Hits@1 and 74.71%
on Hits@10. This category exclusively consists of complex
temporal questions requiring multi-step reasoning, such as
Before Last,After First, andEqual Multi. Existing semantic-
based RAG methods struggle in this setting, as they are unable
to retrieve coherent multi-hop causal evidence and often in-
troduce redundant or irrelevant facts. In contrast, TCC-RAG
explicitly leverages causal cue-guided traversal to retrieve
multi-hop causal knowledge. By following temporal causal
chains while suppressing redundant information, the proposed
method is able to recover causally consistent evidence paths,
leading to substantial improvements under complex multi-step
temporal reasoning scenarios.
In addition, we further evaluate TCC-RAG with differ-
ent LLM backbones. Notably, TCC-RAG equipped with
LLaMA 3.1 8B achieves comparable Hits@1 performance to
RTQA using DeepSeek-Chat, while surpassing it by 7.16% on
Hits@10. This result demonstrates that even with a lightweight
backbone, TCC-RAG is able to compensate for limited reason-
ing capacity through effective causal retrieval. When equipped
with GPT-4o-mini, TCC-RAG further establishes clear ad-
vantages over RTQA, achieving improvements of 6.80% and
11.58% on Hits@1 and Hits@10, respectively. These consis-
tent gains across different backbone capacities indicate that
the performance improvements of TCC-RAG are not solely
attributed to stronger language models, but rather stem from
its superior ability to retrieve and organize temporally coherent
causal knowledge. By explicitly capturing intrinsic temporal
causal structures through causal-guided retrieval, TCC-RAG
effectively mitigates the reasoning limitations of smaller-scale
LLMs and amplifies the strengths of more capable backbones,
enabling it to outperform LLM-based baselines across model
sizes consistently.
Results on TimeQuestions.The experimental results on the
TimeQuestions dataset are reported in Table V. As shown,
TCC-RAG consistently achieves the best performance across
all question types. Overall, TCC-RAG outperforms the best-
performing baseline RTQA by 8.40%, demonstrating its strong
generalization ability on diverse temporal reasoning tasks.
Similar to the observations on MULTITQ, the performance
gains on TimeQuestions are primarily attributed to question
types that require multi-step temporal reasoning. In partic-
ular, on theImplicitquestion category, TCC-RAG achieves
a notable improvement of 13.58% over the best baselineTABLE V: Performance comparison (Hits@1) on different
question and answer types on TimeQuestions. The best is
bolded and the second best is underlined. The bottom row is
the relative improvement of TCC-RAG over the best baseline.
Model OverallQuestion Type Answer Type
Explicit Implicit Temporal Ordinal
PullNet 0.105 0.022 0.081 0.234 0.029
Uniqorn 0.331 0.318 0.316 0.392 0.202
GRAFT-Net 0.452 0.445 0.428 0.515 0.322
CronKGQA 0.462 0.466 0.445 0.511 0.369
TempoQR 0.416 0.465 0.036 0.400 0.349
EXAQT 0.572 0.568 0.512 0.642 0.420
TwiRGCN 0.605 0.602 0.586 0.641 0.518
LGQA 0.529 0.532 0.506 0.605 0.402
LLaMA2 7B 0.271 0.268 0.325 0.279 0.234
ChatGPT 0.459 0.433 0.511 0.465 0.481
GenTKGQA 0.584 0.596 0.611 0.563 0.578
TimeR40.617 0.609 0.512 0.754 0.426
RTQA 0.655 0.648 0.601 0.762 0.503
TCC-RAG0.710 0.711 0.694 0.781 0.597
Imp (%) 8.40 9.72 13.58 2.49 3.29
TABLE VI: Ablation studies (Hits@1) of TCC-RAG on MUL-
TITQ.
Model Overall Multiple Single Entity Time
w/o CCG0.647 0.127 0.837 0.517 0.933
w/o PQP0.778 0.653 0.824 0.759 0.821
w/o CEF0.841 0.664 0.910 0.791 0.958
TCC-RAG 0.847 0.691 0.911 0.801 0.959
and 35.54% over semantic-based RAG methods such as
TimeR4. Such questions require inferring multi-hop temporal
and causal dependencies that are not explicitly stated in the
query or directly retrievable via semantic similarity. These
results further indicate that TCC-RAG is able to enhance the
reasoning capability of LLMs by explicitly retrieving multi-
step temporal causal chains. By capturing structured causal
knowledge rather than isolated facts, TCC-RAG effectively
supports complex temporal reasoning, consistently validating
its causal knowledge mining ability across different datasets
and reasoning settings.
C. Ablation Study
We conduct ablation studies on MULTITQ to evaluate the
effectiveness of each component in TCC-RAG. The results
are reported in Table VI. Specifically,w/o CCGremoves the
Causal Cue Graph and relies solely on semantic retrieval,w/o
PQPdisables the progressive query parsing pipeline and uses
a single-shot LLM call to extract all signals (entities, relation,
cue, role) at once, andw/o CEFremoves the causal evidence
filter.
Effect of Causal Cue Graph.Removing the Causal Cue
Graph results in a substantial performance degradation across
all metrics. In particular, the overall performance drops by
23.61%, with the most significant decline observed in theMul-

Q: After the Cabinet Council of Ministers of Kazakhstan, who was the first to express the intention to negotiate with China?1.China Engage in negotiation Envoy (United States) in 2010-02-252.Envoy (United States) Express intent to meet or negotiate China in 2010-02-223.China Engage in negotiation Iran in 2010-02-244.Cabinet Council of Ministers Advisors (United States) Express intent to meet or negotiate China in 2010-03-015.'Cabinet Council of Ministers Advisors (United States) Express intent to meet or negotiate China in 2010-02-27Top5 of Retrieved 15 Facts
TimeR4Answer:[‘Envoy (United States)’]Ground Truth: ['Citizen (North Korea)']1.Cabinet Council of Ministers Advisors (Kazakhstan) Express intent to meet or negotiate China in 2010-02-202.Kazakhstan Engage in negotiation China in 2005-07-043.China Engage in negotiation Iran in 2010-02-244.Cabinet Council of Ministers Advisors (United States) Express intent to meet or negotiate China in 2010-03-015.'Cabinet Council of Ministers Advisors (United States) Express intent to meet or negotiate China in 2010-02-27RTQAAnswer:[‘Japan 2010-03-01’]1.Cabinet Council of Ministers Advisors (Kazakhstan) Express intent to meet or negotiate China in 2010-02-202.South Korea Express intent to meet or negotiate China in 2010-02-223.Citizen (North Korea) Express intent to meet or negotiate China in 2010-02-224.Envoy (United States) Express intent to meet or negotiate China 2010-02-22TCC-RAGAnswer:[‘South Korea ’,’ Citizen (North Korea) ’, ‘Envoy (United States)’]
Top5 of Retrieved 50 FactsOnly 4 Retrieved FactsFig. 4: A comparison of the retrieved facts between semantic-
based RAG methods and TCC-RAG on MULTITQ. Facts
highlighted in red are causally essential evidence.
tiplequestion category, where the score decreases by 81.62%.
TheEntity-type answers also suffer a notable drop of 35.45%,
while the performance degradation onSingleandTime-type
questions is relatively minor (8.12% and 2.71%, respectively).
This phenomenon is expected, asMultiplequestions exclu-
sively consist of complex multi-step temporal reasoning tasks.
Similarly,Entity-type answers tend to involve more multi-hop
temporal dependencies thanTime-type answers. These results
demonstrate that the causal cue graph is critical for modeling
complex temporal causal structures, and its removal severely
impairs the model’s ability to perform multi-step temporal
reasoning.
Effect of Progressive Query Parsing.Disabling progressive
query parsing and relying on a single-shot LLM call leads
to consistent performance degradation across all categories,
with an overall drop of 8.14%. This degradation is particularly
evident inMultiple(5.49%) andTime-type (14.3%) questions.
The results indicate that time-sensitive queries exhibit high
linguistic and structural diversity, making accurate entity and
relation extraction challenging when all signals are produced
in a single LLM pass. Without the staged decomposition into
entity grounding, context-aware relation parsing, and rule-
based cue extraction, errors accumulate and prevent effective
grounding of queries onto the causal cue graph. Consequently,
the model fails to fully exploit the structured causal retrieval
mechanism, resulting in reduced performance.
Effect of Causal Evidence Filter.Removing the causal evi-
dence filter also leads to performance degradation, particularly
onMultipleandEntity-type questions, where the scores drop
by 3.90% and 1.24%, respectively. These observations suggest
that the filter primarily contributes to refining causal evidence
in complex temporal reasoning scenarios, where retrieval may
still introduce redundant or noisy facts due to data noise. The
relatively small overall performance drop (0.71%) is expected,
as the filter operates as a structural refinement layer on top
of an already concise retrieval output. Since most queries can
already retrieve minimal and precise causal knowledge through
the causal cue graph, only a small portion of queries require
further refinement by the filter.
Semantic Fallback Analysis.Semantic retrieval is invokedTABLE VII: Retrieval-path coverage and conditional Hits@1.
Dataset Path Coverage Hits@1
MULTITQStructured 84.46% 0.912
Fallback 15.54% 0.492
TimeQuestionsStructured 91.82% 0.738
Fallback 8.18% 0.398
when structured retrieval returns no facts. Table VII re-
ports coverage on the complete test sets and conditional
Hits@1 within each retrieval path. On MULTITQ, struc-
tured retrieval handles 46,102 of 54,584 queries, while fall-
back handles the remaining 8,482. On TimeQuestions, struc-
tured coverage reaches 91.82%. With fallback disabled, un-
covered queries receive no prediction, and overall Hits@1
is 0.770/0.678 on MULTITQ/TimeQuestions, still exceeding
RTQA’s 0.765/0.655. Thus, causal-cue retrieval is effective
independently, while semantic fallback improves coverage.
D. Insights and Case Study
Results for Multi-Granularity Time.To evaluate the effec-
tiveness of TCC-RAG under different temporal resolutions,
we conduct experiments across multiple time granularities,
includingday,month, andyear. The results are shown in
Figure 5. Overall, TCC-RAG achieves the best performance
across most temporal granularities and question types, demon-
strating its robustness in handling temporal reasoning under
different time resolutions. In particular, consistent improve-
ments are observed in theBefore/AfterandEqual Multicat-
egories, indicating that temporal causal chain-guided retrieval
is effective for capturing causal temporal dependencies across
events at different granularities. We note that RTQA slightly
outperforms TCC-RAG in theEqual-MonthandEqual-Multi-
Monthsettings. These cases mainly involve direct temporal
alignment at a single reasoning step, where semantic retrieval
is often sufficient. Nevertheless, TCC-RAG maintains stable
performance across coarse to fine-grained temporal reasoning
tasks, confirming the general effectiveness of the proposed
causal retrieval framework.
Case Study.To verify whether TCC-RAG can retrieve truly
causal knowledge, we conduct a retrieval case study comparing
TCC-RAG with semantic-based methods. The results are illus-
trated in Figure 4. TimeR4employs rewrite-and-rerank strate-
gies, while RTQA utilizes question decomposition. However,
both methods fail to retrieve the key causal facts required for
multi-step temporal reasoning and introduce a large number
of redundant facts, which ultimately prevents the model from
producing the completely correct answer. TCC-RAG leverages
the causal cue graph to retrieve only the most relevant facts.
With merely four retrieved facts, the model can construct
a complete temporal causal chain and successfully perform
multi-step reasoning. This demonstrates that TCC-RAG is
capable of mining concise yet sufficient causal knowledge.
Furthermore, this example reveals an implicit temporal am-
biguity. Although the ground-truth answer isCitizen (North

Day Month Year0.00.20.40.60.81.0Hits@1
0.425
0.389
0.3310.445
0.393
0.3500.916
0.959
0.9670.950
0.869
0.971
(a) EqualDay Month Year0.00.20.40.60.81.0Hits@1
0.375
0.474
0.4500.379
0.548
0.5250.842
0.898
0.7870.845
0.903
0.883
(b) Before/AfterDay Month Year0.00.20.40.60.81.0Hits@1
0.295
0.333
0.2510.308
0.321
0.2830.729
0.758
0.5780.791
0.672
0.695
(c) Equal MultiCronKGQA MultiQA RTQA TCC-RAGFig. 5: Experiment results of Hits@1 across multiple temporal granularities on the MULTITQ dataset.
T oken Consumption Time Consumption040080012001600token / query
0.00.51.01.52.02.5
seconds / queryTCC-RAG RTQA
Fig. 6: Comparison of token cost and inference time between
TCC-RAG and RTQA on MULTITQ.
Korea), multiple events, includingSouth KoreaandEnvoy
(United States), occur on the same day. TCC-RAG accurately
captures this temporal coincidence through causal cue-guided
retrieval and returns all temporally consistent answers. This
observation further demonstrates that TCC-RAG can uncover
latent causal structures within the temporal knowledge store
rather than relying solely on surface-level semantic matching.
E. Efficiency Analysis
Figure 6 presents a detailed comparison of token cost and
inference time between TCC-RAG and RTQA on MULTITQ,
highlighting the efficiency advantages of TCC-RAG.
Token Consumption.We compare the average input token
consumption of RTQA and TCC-RAG on the MULTITQ
dataset, where both methods are deployed using DeepSeek-
Chat. In RTQA, token usage mainly comes from the recursive
question decomposition process and the subsequent answer
aggregation for each sub-question. In contrast, TCC-RAG
consumes tokens primarily during query parsing and final
answer generation. Experimental results show that RTQA
consumes an average of 1,403.8 tokens per question, whereas
TCC-RAG only uses 604.8 tokens per question on average,
reducing token consumption by over 57%. This substantial
reduction is mainly attributed to the concise yet sufficiently
informative set of causal facts retrieved by TCC-RAG, which
yields a much smaller evidence context for the LLM. These
results demonstrate that causal-chain-guided retrieval not only
improves reasoning accuracy but also significantly reducestoken consumption compared with decomposition-based RAG
pipelines.
Inference Time.We further compare the average end-to-
end processing time per query. TCC-RAG takes 2.11s per
query on average, slightly higher than the 1.82s of RTQA.
This minor gap mainly stems from the entity grounding
stage, where TCC-RAG invokes both a NER model and an
LLM for dual-path extraction and consistency verification,
introducing a small additional cost. Nevertheless, this overhead
is well justified by the substantial accuracy gains, and remains
acceptable in practice. Since all LLM components are accessed
through asynchronous API calls, multiple queries can be
processed concurrently, allowing TCC-RAG to deliver answers
efficiently at scale despite the slightly higher per-query latency.
VI. CONCLUSION
In this paper, we propose TCC-RAG, a novel RAG frame-
work for TKGQA. To overcome the limitations of existing
semantic-based RAG methods, which suffer from missing
causally relevant facts and introducing noisy evidence, we
introduce a multi-view causal cue graph that explicitly encodes
temporal causal relations. Built on this structure, we design a
progressive causal chain-guided retrieval strategy that traces
causally relevant evidence along the routed graph view, paired
with a causal evidence filter that distills the retrieved set into a
minimal causally sufficient subset. Extensive experiments on
two widely used TKGQA benchmarks demonstrate that TCC-
RAG consistently outperforms state-of-the-art methods, with
especially significant improvements on complex multi-step
temporal reasoning. Ablation studies verify the effectiveness of
each component, while a case study further demonstrates the
ability of TCC-RAG to retrieve concise yet sufficient causal
evidence. Moreover, efficiency analysis confirms the practical
efficiency and scalability of the proposed framework.
ACKNOWLEDGMENTS
This work was sponsored by the National Natural Sci-
ence Foundation of China (No.62472327, 62572361), CCF-
Tencent Rhino-Bird Open Research Fund (CCF-Tencent
RAGR20260132), and the Sichuan Clinical Research Center
for Imaging Medicine (YXYX2402).

AI-GENERATEDCONTENTACKNOWLEDGMENTS
AI tools (ChatGPT and Claude) were used for language pol-
ishing and to suggest improvements to the visual presentation
of selected figures, including colors and formatting. AI was not
used for research conception, method design, or experiments.
REFERENCES
[1] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman,
D. Almeida, J. Altenschmidt, S. Altman, S. Anadkatet al., “Gpt-4
technical report,”arXiv preprint arXiv:2303.08774, 2023.
[2] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y . Babaei,
N. Bashlykov, S. Batra, P. Bhargava, S. Bhosaleet al., “Llama
2: Open foundation and fine-tuned chat models,”arXiv preprint
arXiv:2307.09288, 2023.
[3] D. Li, B. Jiang, L. Huang, A. Beigi, C. Zhao, Z. Tan, A. Bhattacharjee,
Y . Jiang, C. Chen, T. Wuet al., “From generation to judgment: Oppor-
tunities and challenges of llm-as-a-judge,” inProceedings of the 2025
Conference on Empirical Methods in Natural Language Processing,
2025, pp. 2757–2791.
[4] M. Yu, F. Meng, X. Zhou, S. Wang, J. Mao, L. Pan, T. Chen, K. Wang,
X. Li, Y . Zhanget al., “A survey on trustworthy llm agents: Threats and
countermeasures,” inProceedings of the 31st ACM SIGKDD Conference
on Knowledge Discovery and Data Mining V . 2, 2025, pp. 6216–6226.
[5] H. Han, K. Hu, W. Cai, D. Zhang, B. Qin, Y . Wang, J. Jiang, X. Yan,
and B. Du, “MiCU: End-to-end smart home command understanding
with large language model,” inProceedings of the 32nd ACM SIGKDD
Conference on Knowledge Discovery and Data Mining V . 2, 2026, pp.
7311–7321.
[6] H. Luo, Y . Lin, X. Yan, X. Hu, Y . Wang, Q. Zeng, H. Wang, and
J. Jiang, “Guiding LLM-based smart contract generation with finite
state machine,” inProceedings of the Thirty-Fourth International Joint
Conference on Artificial Intelligence, 2025, pp. 5869–5877.
[7] H. Luo, X. Yan, X. Li, Q. Zeng, Y . Lin, S. Feng, H. Wang, and J. Jiang,
“AdaMix: Adaptive mixing for short and long reasoning adapters,”
inProceedings of the 64th Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), 2026, pp. 40 126–
40 147.
[8] H. Zheng, Y . Zhang, F. Fu, X. Zhou, H. Luo, H. Zhu, Y . Zhu, H. Wang,
X. Yan, and J. Jiang, “Scheduling LLM inference with uncertainty-aware
output length predictions,” inProceedings of the 43rd International
Conference on Machine Learning, ser. Proceedings of Machine Learning
Research, vol. 306. PMLR, 06–11 Jul 2026, pp. 163 894–163 918.
[9] W. Fan, Y . Ding, L. Ning, S. Wang, H. Li, D. Yin, T.-S. Chua, and
Q. Li, “A survey on rag meeting llms: Towards retrieval-augmented large
language models,” inProceedings of the 30th ACM SIGKDD conference
on knowledge discovery and data mining, 2024, pp. 6491–6501.
[10] S. Zhao, Y . Yang, Z. Wang, Z. He, L. K. Qiu, and L. Qiu, “Retrieval
augmented generation (rag) and beyond: A comprehensive survey on
how to make your llms use external data more wisely,”arXiv preprint
arXiv:2409.14924, 2024.
[11] Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai, J. Sun,
M. Wang, and H. Wang, “Retrieval-augmented generation for large
language models: A survey,”arXiv preprint arXiv:2312.10997, 2023.
[12] M. Su, Z. Li, Z. Chen, L. Bai, X. Jin, and J. Guo, “Temporal knowledge
graph question answering: A survey,”arXiv preprint arXiv:2406.14191,
2024.
[13] Z. Chen, J. Liao, and X. Zhao, “Multi-granularity temporal question
answering over knowledge graphs,” inProceedings of the 61st Annual
Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers), 2023, pp. 11 378–11 392.
[14] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training
of deep bidirectional transformers for language understanding,” inPro-
ceedings of the 2019 conference of the North American chapter of the
association for computational linguistics: human language technologies,
volume 1 (long and short papers), 2019, pp. 4171–4186.
[15] Z. Lan, M. Chen, S. Goodman, K. Gimpel, P. Sharma, and R. Soricut,
“Albert: A lite bert for self-supervised learning of language representa-
tions,” inInternational Conference on Learning Representations, 2020.
[16] C. Mavromatis, P. L. Subramanyam, V . N. Ioannidis, A. Adeshina, P. R.
Howard, T. Grinberg, N. Hakim, and G. Karypis, “Tempoqr: temporal
question reasoning over knowledge graphs,” inProceedings of the AAAI
conference on artificial intelligence, vol. 36, no. 5, 2022, pp. 5825–5833.[17] B. Piryani, A. Abdullah, J. Mozafari, A. Anand, and A. Jatowt, “It’s
high time: A survey of temporal information retrieval and question
answering,”arXiv preprint arXiv:2505.20243, 2025.
[18] X. Qian, Y . Zhang, Y . Zhao, B. Zhou, X. Sui, L. Zhang, and K. Song,
“Timer4: Time-aware retrieval-augmented large language models for
temporal knowledge graph question answering,” inProceedings of the
2024 Conference on Empirical Methods in Natural Language Process-
ing, 2024, pp. 6942–6952.
[19] Z. Gong, J. Li, Z. Liu, L. Liang, H. Chen, and W. Zhang, “Rtqa:
Recursive thinking for complex temporal knowledge graph question
answering with large language models,” inProceedings of the 2025
Conference on Empirical Methods in Natural Language Processing,
2025, pp. 9853–9870.
[20] Y . Gao, L. Qiao, Z. Kan, Z. Wen, Y . He, and D. Li, “Two-stage
generative question answering on temporal knowledge graph using large
language models,” inFindings of the Association for Computational
Linguistics ACL 2024, 2024, pp. 6719–6734.
[21] Z. Chen, D. Li, X. Zhao, B. Hu, and M. Zhang, “Temporal knowledge
question answering via abstract reasoning induction,” inProceedings
of the 62nd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), Aug. 2024, pp. 4872–4889.
[22] Q. Hu, X. Tu, C. Guo, and S. Zhang, “Time-aware ReAct agent for
temporal knowledge graph question answering,” inFindings of the
Association for Computational Linguistics: NAACL 2025. Association
for Computational Linguistics, 2025, pp. 6028–6039.
[23] Q. Huang, K. Liu, L. Deng, S. Zhang, C. Hu, T. Qian, X. Yan,
and J. Jiang, “DM-RAG: Enhancing user support in dameng databases
with retrieval-augmented generation,” in2026 IEEE 42nd International
Conference on Data Engineering (ICDE). IEEE, 2026, pp. 4067–4080.
[24] J. Chen, H. Lin, X. Han, and L. Sun, “Benchmarking large language
models in retrieval-augmented generation,” inProceedings of the AAAI
Conference on Artificial Intelligence, vol. 38, no. 16, 2024, pp. 17 754–
17 762.
[25] B. Jimenez Gutierrez, Y . Shu, Y . Gu, M. Yasunaga, and Y . Su, “Hip-
porag: Neurobiologically inspired long-term memory for large language
models,”Advances in Neural Information Processing Systems, vol. 37,
pp. 59 532–59 569, 2024.
[26] D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody, S. Truitt,
D. Metropolitansky, R. O. Ness, and J. Larson, “From local to global:
A graph rag approach to query-focused summarization,”arXiv preprint
arXiv:2404.16130, 2024.
[27] J. Sun, C. Xu, L. Tang, S. Wang, C. Lin, Y . Gong, L. Ni, H.-Y . Shum,
and J. Guo, “Think-on-graph: Deep and responsible reasoning of large
language model on knowledge graph,” inThe Twelfth International
Conference on Learning Representations, 2024.
[28] C. Mavromatis and G. Karypis, “Gnn-rag: Graph neural retrieval for
efficient large language model reasoning on knowledge graphs,” in
Findings of the Association for Computational Linguistics: ACL 2025,
2025, pp. 16 682–16 699.
[29] Q. Zeng, X. Hu, Y . Lin, H. Luo, S. X. Rao, X. Yan, and J. Jiang, “PoP-
RAG: Holistic query planning for GraphRAG,”Transactions on Graph
Intelligence and Network Applications, 2026.
[30] M. Li, S. Miao, and P. Li, “Simple is effective: The roles of graphs and
large language models in knowledge-graph-based retrieval-augmented
generation,” inThe Thirteenth International Conference on Learning
Representations, 2025.
[31] X. Tan, X. Wang, Q. Liu, X. Xu, X. Yuan, and W. Zhang, “Paths-over-
graph: Knowledge graph empowered large language model reasoning,”
inProceedings of the ACM on Web Conference 2025, 2025, pp. 3505–
3522.
[32] A. Saxena, S. Chakrabarti, and P. Talukdar, “Question answering over
temporal knowledge graphs,”arXiv preprint arXiv:2106.01515, 2021.
[33] Z. Jia, P. Christmann, and G. Weikum, “Faithful temporal question
answering over heterogeneous sources,” inProceedings of the ACM Web
Conference 2024, 2024, pp. 2052–2063.
[34] Y . Liu, D. Liang, M. Li, F. Giunchiglia, X. Li, S. Wang, W. Wu,
L. Huang, X. Feng, and R. Guan, “Local and global: Temporal question
answering via information fusion,” inProceedings of the Thirty-Second
International Joint Conference on Artificial Intelligence, IJCAI-23. In-
ternational Joint Conferences on Artificial Intelligence Organization,
2023, pp. 5141–5149.
[35] C. Xue, D. Liang, P. Wang, and J. Zhang, “Question calibration and
multi-hop modeling for temporal question answering,” inProceedings

of the AAAI Conference on Artificial Intelligence, vol. 38, no. 17, 2024,
pp. 19 332–19 340.
[36] A. Sharma, A. Saxena, C. Gupta, M. Kazemi, P. Talukdar, and
S. Chakrabarti, “Twirgcn: Temporally weighted graph convolution for
question answering over temporal knowledge graphs,” inProceedings
of the 17th Conference of the European Chapter of the Association for
Computational Linguistics, 2023, pp. 2049–2060.
[37] V . Sanh, L. Debut, J. Chaumond, and T. Wolf, “Distilbert, a distilled
version of bert: smaller, faster, cheaper and lighter,”arXiv preprint
arXiv:1910.01108, 2019.
[38] A. Saxena, A. Tripathi, and P. Talukdar, “Improving multi-hop question
answering over knowledge graphs using knowledge base embeddings,”
inProceedings of the 58th annual meeting of the association for
computational linguistics, 2020, pp. 4498–4507.
[39] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle,
A. Letman, A. Mathur, A. Schelten, A. Vaughan, A. Yang, A. Fan
et al., “The Llama 3 herd of models,”arXiv preprint arXiv:2407.21783,
2024. [Online]. Available: https://arxiv.org/abs/2407.21783v3
[40] A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng,
C. Zhang, C. Ruanet al., “Deepseek-v3 technical report,”arXiv preprint
arXiv:2412.19437, 2024.
[41] H. Sun, T. Bedrax-Weiss, and W. Cohen, “Pullnet: Open domain
question answering with iterative retrieval on knowledge bases and
text,” inProceedings of the 2019 conference on empirical methods in
natural language processing and the 9th international joint conference
on natural language processing (EMNLP-IJCNLP), 2019, pp. 2380–
2390.
[42] H. Sun, B. Dhingra, M. Zaheer, K. Mazaitis, R. Salakhutdinov, and
W. Cohen, “Open domain question answering using early fusion of
knowledge bases and text,” inProceedings of the 2018 conference on
empirical methods in natural language processing, 2018, pp. 4231–4242.
[43] S. Pramanik, J. Alabi, R. S. Roy, and G. Weikum, “Uniqorn: unified
question answering over rdf knowledge graphs and natural language
text,”Journal of Web Semantics, vol. 83, p. 100833, 2024.
[44] Z. Jia, S. Pramanik, R. Saha Roy, and G. Weikum, “Complex temporal
question answering on knowledge graphs,” inProceedings of the 30th
ACM international conference on information & knowledge manage-
ment, 2021, pp. 792–802.
[45] Y . Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis,
L. Zettlemoyer, and V . Stoyanov, “RoBERTa: A robustly optimized
BERT pretraining approach,”arXiv preprint arXiv:1907.11692, 2019.
[46] W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, and M. Zhou, “Minilm:
Deep self-attention distillation for task-agnostic compression of pre-
trained transformers,”Advances in neural information processing sys-
tems, vol. 33, pp. 5776–5788, 2020.
[47] N. Reimers and I. Gurevych, “Sentence-bert: Sentence embeddings
using siamese bert-networks,” inProceedings of the 2019 conference
on empirical methods in natural language processing and the 9th
international joint conference on natural language processing (EMNLP-
IJCNLP), 2019, pp. 3982–3992.