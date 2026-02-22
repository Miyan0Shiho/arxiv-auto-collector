# HyperRAG: Reasoning N-ary Facts over Hypergraphs for Retrieval Augmented Generation

**Authors**: Wen-Sheng Lien, Yu-Kai Chan, Hao-Lung Hsiao, Bo-Kai Ruan, Meng-Fen Chiang, Chien-An Chen, Yi-Ren Yeh, Hong-Han Shuai

**Published**: 2026-02-16 05:15:55

**PDF URL**: [https://arxiv.org/pdf/2602.14470v1](https://arxiv.org/pdf/2602.14470v1)

## Abstract
Graph-based retrieval-augmented generation (RAG) methods, typically built on knowledge graphs (KGs) with binary relational facts, have shown promise in multi-hop open-domain QA. However, their rigid retrieval schemes and dense similarity search often introduce irrelevant context, increase computational overhead, and limit relational expressiveness. In contrast, n-ary hypergraphs encode higher-order relational facts that capture richer inter-entity dependencies and enable shallower, more efficient reasoning paths. To address this limitation, we propose HyperRAG, a RAG framework tailored for n-ary hypergraphs with two complementary retrieval variants: (i) HyperRetriever learns structural-semantic reasoning over n-ary facts to construct query-conditioned relational chains. It enables accurate factual tracking, adaptive high-order traversal, and interpretable multi-hop reasoning under context constraints. (ii) HyperMemory leverages the LLM's parametric memory to guide beam search, dynamically scoring n-ary facts and entities for query-aware path expansion. Extensive evaluations on WikiTopics (11 closed-domain datasets) and three open-domain QA benchmarks (HotpotQA, MuSiQue, and 2WikiMultiHopQA) validate HyperRAG's effectiveness. HyperRetriever achieves the highest answer accuracy overall, with average gains of 2.95% in MRR and 1.23% in Hits@10 over the strongest baseline. Qualitative analysis further shows that HyperRetriever bridges reasoning gaps through adaptive and interpretable n-ary chain construction, benefiting both open and closed-domain QA.

## Full Text


<!-- PDF content starts -->

HyperRAG: Reasoning N-ary Facts over Hypergraphs for
Retrieval Augmented Generation
Wen-Sheng Lien
National Yang Ming Chiao Tung
University
Hsinchu, Taiwan
vincentlien.ii13@nycu.edu.twYu-Kai Chan
National Yang Ming Chiao Tung
University
Hsinchu, Taiwan
ctw33888.ee13@nycu.edu.twHao-Lung Hsiao
National Yang Ming Chiao Tung
University
Hsinchu, Taiwan
hlhsiao.cs13@nycu.edu.tw
Bo-Kai Ruan
National Yang Ming Chiao Tung
University
Hsinchu, Taiwan
bkruan.ee11@nycu.edu.twMeng-Fen Chiang
National Yang Ming Chiao Tung
University
Hsinchu, Taiwan
meng.chiang@nycu.edu.twChien-An Chen
E.SUN Bank
Taipei, Taiwan
lukechen-15953@esunbank.com
Yi-Ren Yeh
National Kaohsiung Normal
University
Kaohsiung, Taiwan
yryeh@nknu.edu.twHong-Han Shuai
National Yang Ming Chiao Tung
University
Hsinchu, Taiwan
hhshuai@nycu.edu.tw
Abstract
Graph-based Retrieval-Augmented Generation (RAG) typically op-
erates on binary Knowledge Graphs (KGs). However, decomposing
complex facts into binary triples often leads to semantic fragmenta-
tion and longer reasoning paths, increasing the risk of retrieval drift
and computational overhead. In contrast, 𝑛-ary hypergraphs pre-
serve high-order relational integrity, enabling shallower and more
semantically cohesive inference. To exploit this topology, we pro-
poseHyperRAG, a framework tailored for 𝑛-ary hypergraphs fea-
turing two complementary retrieval paradigms: (i) HyperRetriever
learns structural-semantic reasoning over 𝑛-ary facts to construct
query-conditioned relational chains. It enables accurate factual
tracking, adaptive high-order traversal, and interpretable multi-hop
reasoning under context constraints. (ii) HyperMemory leverages
the LLM’s parametric memory to guide beam search, dynamically
scoring𝑛-ary facts and entities for query-aware path expansion.
Extensive evaluations on WikiTopics (11 closed-domain datasets)
and three open-domain QA benchmarks (HotpotQA, MuSiQue,
and 2WikiMultiHopQA) validate HyperRAG’s effectiveness. Hy-
perRetriever achieves the highest answer accuracy overall, with
average gains of 2.95% in MRR and 1.23% in Hits@10 over the
strongest baseline. Qualitative analysis further shows that Hyper-
Retriever bridges reasoning gaps through adaptive and interpretable
𝑛-ary chain construction, benefiting both open and closed-domain
QA. Our codes are publicly available at https://github.com/Vincent-
Lien/HyperRAG.git.
This work is licensed under a Creative Commons Attribution 4.0 International License.
WWW ’26, Dubai, United Arab Emirates.
©2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2307-0/2026/04
https://doi.org/10.1145/3774904.3792710CCS Concepts
•Information systems →Retrieval models and ranking;Lan-
guage models;Question answering.
Keywords
Hypergraph-based Retrieval-Augmented Generation, N-ary Rela-
tional Knowledge Graphs, Multi-hop Question Answering, Memory-
Guided Adaptive Retrieval
ACM Reference Format:
Wen-Sheng Lien, Yu-Kai Chan, Hao-Lung Hsiao, Bo-Kai Ruan, Meng-Fen
Chiang, Chien-An Chen, Yi-Ren Yeh, and Hong-Han Shuai. 2026. Hyper-
RAG: Reasoning N-ary Facts over Hypergraphs for Retrieval Augmented
Generation. InProceedings of the ACM Web Conference 2026 (WWW ’26),
April 13–17, 2026, Dubai, United Arab Emirates.ACM, New York, NY, USA,
12 pages. https://doi.org/10.1145/3774904.3792710
1 Introduction
Retrieval-Augmented Generation (RAG) has established itself as a
critical mechanism for augmenting Large Language Models (LLMs)
with non-parametric external knowledge during inference [12, 17,
19,20]. By dynamically retrieving verifiable information from ex-
ternal corpora without the need for extensive fine-tuning, RAG
effectively mitigates intrinsic LLM limitations such as hallucina-
tions and temporal obsolescence. This paradigm has proven par-
ticularly transformative for knowledge-intensive tasks, including
open-domain question answering (QA), fact verification, and com-
plex information extraction, driving significant innovation across
both academia and industry.
Current RAG methodologies broadly fall into three categories:
document-based, graph-based, and hybrid approaches. Document-
based methods utilize dense vector retrieval to match queries with
textual segments, offering scalability but often failing to capture
complex structural dependencies [ 5,6]. Conversely, graph-basedarXiv:2602.14470v1  [cs.CL]  16 Feb 2026

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates. Wen-Sheng Lien et al.
TV 101
Sam
Pillsbury
Eric
Laneuville
English
California(b) Hyper graph
Bruce  
Seth Gr een
Sam
WeismanBruce Seth Green, Sam
Weisman, Sam Pillsbury, and
Eric Laneuville directs TV 101 in
English in California.N-ary relation
Relational chain
EntityHyperedge Question: What other shows or movies were directed by directors
who also directed shows that Bruce Seth Green directed?
(a) Kno wledge Gr aph
Bruce 
Seth GreenTV 101
Unchained
Hear t
Sam
Weisman
Sam
Pillsbur y
Born  
Into ExileDickie
Rober ts
Eric
Laneuville
TV 101Film Film DirectorBinary relation
Relational chain
Entity
Figure 1: Structural Comparison of (a) Knowledge Graphs
and (b) Hypergraphs. For a given question 𝑞, (a) requires
3-hop reasoning over binary facts, while (b) enables single-
hop inference via an 𝑛-ary relational fact, yielding a more
compact and expressive multi-entity representation.
methods leverage Knowledge Graphs (KGs) to explicitly model re-
lationships, enabling multi-hop reasoning over structured data [ 15,
31]. Hybrid approaches attempt to bridge these paradigms, bal-
ancing comprehensiveness with efficiency. However, despite the
reasoning potential of graph-based methods, the prevailing reliance
on binary KGs presents fundamental topological limitations.
Traditional graph-based RAG methods predominantly rely on bi-
nary knowledge graphs, which suffer from notable limitations when
applied to closed-domain question-answering scenarios. Specifi-
cally, binary KG approaches encounter two fundamental structural
limitations. First,Semantic Fragmentationarises because binary
relations limit the expressiveness required to capture complex multi-
entity interactions, forcing the decomposition of holistic facts into
disjoint triples that fail to represent intricate semantic nuances.
Second, this fragmentation leads toPath Explosion, where con-
ventional approaches incur significant computational costs due to
the need for deep traversals over the vast binary relation space to
reconnect these facts, enabling error propagation and undermin-
ing real-world practicality [ 18,37]. To address these limitations,
recent work advocates hypergraphs for structured retrieval in RAG.
Hypergraphs natively encode higher-order ( 𝑛-ary) relations that
bind multiple entities and roles, providing a richer semantic sub-
strate than binary graphs [ 26]. As illustrated in Figure 1, the Path
Explosion issue is evident when answering a question grounded on
the topic entity “Bruce Seth Green,” which requires a 3-hop binary
traversal on a standard KG. In contrast, this reduces to a single
hop through an 𝑛-ary relation in a hypergraph, yielding a more
compact representation. Hypergraphs enable the direct modelingof higher-order relational chains, effectively mitigating Semantic
Fragmentation and reducing the reasoning steps required to capture
complex dependencies.
Motivated by these insights, we introduceHyperRAG, an inno-
vative retrieval-augmented generation framework designed explic-
itly for reasoning over 𝑛-ary hypergraphs. HyperRAG integrates
two novel adaptive retrieval variants: (i)HyperRetriever, which uses
a multilayer perceptron (MLP) to fuse structural and semantic em-
beddings, constructing query-conditioned relational chains that
enable accurate and interpretable evidence aggregation within con-
text and token constraints; and (ii)HyperMemory, which leverages
the parametric memory of an LLM to guide beam search, dynam-
ically scoring 𝑛-ary facts and entities for query-adaptive path ex-
pansion. By combining higher-order reasoning with shallower yet
more expressive chains that locate key evidence without multi-hop
traversal. Replacement of the 𝑛-ary structure with a binary reduces
the average MRR from36 .45%to34.15%and the average Hits@10
from40.59%to36.82%(Table 3), indicating gains in response quality.
Our key contributions are summarized as follows.
•We propose HyperRAG, a pioneering framework that shifts the
graph-RAG paradigm from binary triples to 𝑛-ary hypergraphs,
tackling the issues of semantic fragmentation and path explosion.
•We introduce HyperRetriever, a trainable MLP-based retrieval
module that fuses structural and semantic signals to extract pre-
cise, interpretable evidence chains with low latency.
•We develop HyperMemory, a synergistic retrieval approach that
utilizes LLM parametric knowledge to guide symbolic beam
search over hypergraphs for complex query adaptive reasoning.
•Extensive evaluation across closed-domain and open-domain
benchmarks demonstrates that HyperRAG consistently outper-
forms strong baselines, offering a superior trade-off between
retrieval accuracy, reasoning interpretability, and system latency.
2 Preliminaries
2.1 Background
Definition 2.1( 𝑛-ary Relational Knowledge Graph).An 𝑛-ary
relational knowledge graph, or hypergraph, represents relational
facts involving two or more entities and one or more relations.
Formally, following the definition in [ 43], a hypergraph is defined
asG=(E ,R,F), whereEdenotes the set of entities, Rdenotes the
set of relations, and Fthe set of𝑛-ary relational facts (hyperedges).
Each𝑛-ary fact𝑓𝑛∈F, which consists of two or more entities, is
represented as: 𝑓𝑛={𝑒𝑖}𝑛
𝑖=1, where{𝑒𝑖}𝑛
𝑖=1⊆E is a set of𝑛entities
with𝑛≥2.
Unlike binary knowledge graphs, 𝑛-ary representation inher-
ently captures higher-order relational dependencies among multi-
ple entities. 𝑛-ary relations cannot be faithfully decomposed into
combinations of binary relations without losing structural integrity
or introducing ambiguity in semantic interpretation [ 1,9,35]. We
formalize faithful reduction and show that any straightforward bi-
nary scheme violates at least one of: (i) recoverability of the original
tuples, (ii) role preservation, or (iii) multiplicity of co-participations.
Please refer to Appendix A for more details on the recoveryability
of role-preserving hypergraph reduction, roles, and multiplicity.

HyperRAG: Reasoning N-ary Facts over Hypergraphs for Retrieval Augmented Generation WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates.
2.2 Problem Formulation
Problem(Hypergraph-based RAG).Given a question 𝑞, a hyper-
graphGrepresenting 𝑛-ary relational structures, and a collection
of source documents D, the goal of hypergraph-based retrieval-
augmented generation (RAG) is to generate faithful and contextu-
ally grounded answers 𝑎by leveraging salient multi-hop relational
chains fromGand extracting relevant textual evidence fromD.
Complexity: Native 𝑛-ary Hypergraph Retrieval.Let 𝑁𝑒=|E|,
𝑁𝑓=|F| , and ¯𝑛be the average arity. A query binds 𝑘role-typed
arguments, 𝑞={(𝑟𝑖:𝑎𝑖)}𝑘
𝑖=1, and asks for the remaining 𝑛−𝑘 roles.
We maintain sorted posting lists over role incidences, P(𝑟:𝑎)=
{𝑓∈F :(𝑟:𝑎)∈𝑓} , with length 𝑑(𝑟:𝑎) . To answer 𝑞, the𝑛-ary
based retriever intersects the 𝑘posting listsby hyperedge IDsand
reads the missing roles from each surviving hyperedge. Let 𝑛★be
the (max/avg) arity among matches. The running time is given by:
𝑇HYP(𝑞)=O𝑘∑︁
𝑖=1𝑑(𝑟𝑖:𝑎𝑖) +out
,(1)
where outis the number of matching facts. In typical schemas, the
relation arity is often bounded by a small constant (e.g., triadic,
𝑛≤3). As a result, for each match the retriever touches exactly
one hyperedge record to materialize the unbound roles, yielding
per-outputoverheadO(1).
Complexity: Standard Binary KG Retrieval.Suppose each 𝑛-
ary fact𝑓is reified as an event node 𝑒𝑓with𝑛role-typed binary
edges (e.g., role𝑗(𝑒𝑓,𝑎𝑗)). For each binding (𝑟𝑖:𝑎𝑖), use the list of
event IDs postedPevent(𝑟𝑖:𝑎𝑖)and intersect the 𝑘lists to obtain
candidate events to mirror the hypergraph intersection. For each
surviving𝑒𝑓, follow its remaining (𝑛−𝑘) role-edges to materialize
unbound arguments. Let 𝑑event(𝑟:𝑎)=|P event(𝑟:𝑎)| and let𝑛★be
the (max/avg) arity over matches. The running time is given by:
𝑇BIN(𝑞)=O𝑘∑︁
𝑖=1𝑑event(𝑟𝑖:𝑎𝑖) +out·(𝑛★−𝑘)
.(2)
Under a schema-bounded arity, theper-resultoverhead is up to ¯𝑛
role lookups to materialize the remaining arguments. In contrast,
the hypergraph returns them from a single record.
Complexity Gap.In a native hypergraph, all arguments of an𝑛-
ary fact co-reside in asinglehyperedge record, thus materializing a
hit, is one read, i.e., O(1)per result under bounded arity. In contrast,
in an event-reified binary KG, the fact is split across 𝑛role-typed
edges, reachable only via the intermediate event node 𝑒𝑓. As a result,
materializing requires up to (𝑛−𝑘) pointer chases, yielding out· ¯𝑛
term, and usually incurs extra indirections/cache misses.
3 Methodology
We proposeHyperRAG, a novel framework that enhances answer
fidelity by integrating reasoning over condensed 𝑛-ary relational
facts with textual evidence. As depicted in Figure 2, HyperRAG
features two retrieval paradigms: (i)HyperRetriever, which per-
forms adaptive structural-semantic traversal to build interpretable,
query-conditioned relational chains; (ii)HyperMemory, which uti-
lizes the parametric knowledge of the LLM to guide symbolic beam
search. Both variants ground the generation process in hypergraph
structures, ensuring faithful and accurate multi-hop reasoning.
HyperRetriever HyperMemory
MLPFrontier Entities
Entities
Hyperedges
Chunks
ContextFrontier Entities
Entities
Hyperedges
Chunks
Context
LLM
Memory-Guided 
Beam Retriever
Budget-aware Contextualized GeneratorSubgraphAdapted SearchHypergraphDocuments
Relational
ChainsSubgraph
Relational
ChainsBeam SearchWhat other shows or movies were directed by di-
rectors who also directed shows that Bruce Seth
Green directed?Question
Answer: TV 101Bruce Seth
Green
Generator
LLMFigure 2: The overall framework of HyperRAG.
3.1 HyperRetriever: Relational Chains Learning
The motivation behind learning to extract fine-grained 𝑛-ary re-
lational chains over hypergraph structures stems from two key
challenges: (i) the well-documented tendency of LLMs to halluci-
nate factual content and (ii) the vast combinatorial search space
of hypergraphs under limited token and context budgets [ 25]. To
mitigate these challenges, we introduce a lightweight yet expres-
sive retriever that integrates structural and semantic cues to rank
salient𝑛-ary facts aligned with query intent.
3.1.1 Topic Entity Extraction.The purpose of obtaining the topic
entity is to ground the query semantics onto hypergraphs G. For-
mally, given a query 𝑞, we request an LLM with prompt 𝑝topicto
identify a set of topic entities that appear in 𝑞in an LLM as follows:
E𝑞=LLM 𝑝topic,𝑞,
whereE𝑞denotes the set of extracted entities in the query𝑞.
3.1.2 Hyperedge Retrieval and Triple Formation.For each extracted
topic entity 𝑒𝑠∈E𝑞, we retrieve its incident hyperedges from F,
formally defined as follows:
F𝑒𝑠={𝑓𝑛∈F:𝑒𝑠∈𝑓𝑛}.
Each hyperedge 𝑓𝑛∈F𝑒𝑠defines an𝑛-ary relation over a subset of
𝑛entities. To enable pairwise reasoning, we derive a set of pseudo-
binary triples by enumerating ordered entity pairs within each
hyperedge for query𝑞as follows:
T𝑞={(𝑒ℎ,𝑓𝑛,𝑒𝑡)|𝑓𝑛∈F𝑒𝑠, 𝑒ℎ∈𝑓𝑛,𝑒𝑡∈𝑓𝑛},(3)

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates. Wen-Sheng Lien et al.
where each pseudo-binary triple (𝑒ℎ,𝑓𝑛,𝑒𝑡)consists of a head entity,
the originating hyperedge, and a tail entity.
3.1.3 Structural Proximity Encoding.To capture the structural prox-
imity between entities in the hypergraph, we adapt the directional
distance encoding (DDE) mechanism from SubGraphRAG [ 21], ex-
tending it from binary relations to 𝑛-ary hyperedges. Formally, for
each candidate triple (𝑒ℎ,𝑓𝑛,𝑒𝑡)∈T𝑞, we compute its directional
encoding in the following steps:
•One-Hot Initialization:For each entity (𝑒ℎ,𝑓𝑛,𝑒𝑡), we initialize
a one-hot indicator for the head entity:
𝑠(0)
𝑒=1,if∃(𝑒 ℎ,𝑓𝑛,𝑒𝑡)∈T𝑞such that𝑒=𝑒 ℎ,
0,otherwise.(4)
•Bi-directional Feature Propagation:For each layer 𝑙=0,...,𝐿 ,
we propagate features over the set of derived triples T𝑞. Forward
propagation simulates how the head entity 𝑒ℎreaches out to the
tail entity𝑒 𝑡as follows:
𝑠(𝑙+1)
𝑒=1
|{𝑒′|(𝑒′,·,𝑒)∈T𝑞}|∑︁
(𝑒′,·,𝑒)∈T𝑞𝑠(𝑙)
𝑒′.(5)
In contrast, backward propagation updates head encodings based
on tail-to-head influence:
𝑠(𝑟,𝑙+1)
𝑒 =1
|{𝑒′|(𝑒,·,𝑒′)∈T𝑞}|∑︁
(𝑒,·,𝑒′)∈T𝑞𝑠(𝑟,𝑙)
𝑒′.(6)
•Bi-directional Encoding:After 𝐿rounds of propagation, we
concatenate the forward and backward encodings to obtain the
final vector for each entity𝑒as follows:
𝑠𝑒=[𝑠(0)
𝑒∥𝑠(1)
𝑒∥···∥𝑠(𝐿)
𝑒∥𝑠(𝑟,1)
𝑒∥···∥𝑠(𝑟,𝐿)
𝑒],(7)
where∥denotes vector concatenation. Note that the backward
propagation starts from 𝑙=1, as𝑙=0is shared in both directions.
•Triple Encoding:For each candidate triple (𝑒ℎ,𝑓𝑛,𝑒𝑡), we define
its structural proximity encoding as follows:
𝛿(𝑒ℎ,𝑓𝑛,𝑒𝑡)=
𝑠𝑒ℎ∥𝑠𝑒𝑡
,(8)
which is passed to a lightweight parametric neural function to
compute the plausibility score for each candidate triple (𝑒ℎ,𝑓𝑛,𝑒𝑡)
given query𝑞.
3.1.4 Contrastive Plausibility Scoring.To reduce the search space in
the hypergraph structure, we address the challenge that similarity-
based retrieval often introduces noisy or irrelevant triples. To miti-
gate this, we train a lightweight MLP classifier 𝑓𝜃to estimate the
plausibility of each triple candidate and prune uninformative ones.
To this end, the training set is prepared with positive and nega-
tive samples. Let 𝑃∗
𝑞denote the shortest path of triples connecting
the topic entity to a correct answer in the hypergraph G. The
positive samplesT+
𝑖at hop𝑖consist of triples in 𝑃∗
𝑞, denoted as
T+
𝑖={(𝑒ℎ,𝑖,𝑓𝑛
𝑖,𝑒𝑡,𝑖)}. Negative samples 𝑇−
𝑖consist of all other
triples incident to the head entity 𝑒𝑖at hop𝑖that are not in 𝑃∗
𝑞. At
each exploration step, only positive triples are expanded at each
hop, while negative ones are excluded. Each triple (𝑒ℎ,𝑓𝑛,𝑒𝑡)is
encoded in a feature vector by concatenating its contextual and
structural encodings:
x=
𝜑(𝑞)∥𝜑(𝑒 ℎ)∥𝜑(𝑓𝑛)∥𝜑(𝑒𝑡)∥𝛿(𝑒ℎ,𝑓𝑛,𝑒𝑡)
,(9)where𝜑denotes an embedding model that maps the textual content
of the query ( 𝑞), head entity ( 𝑒ℎ), hyperedge ( 𝑓𝑛), and tail entity
(𝑒𝑡), into vector representations, forming the candidate pseudo-
binary triple(𝑒ℎ,𝑓𝑛,𝑒𝑡). The classifier outputs a plausibility score
𝑓𝜃(x)∈[0,1], trained using binary cross-entropy as follows:
L=−1
𝑁𝑁∑︁
𝑖=1h
𝑦𝑖log 𝑓𝜃(x𝑖)+ (1−𝑦𝑖)log 1−𝑓𝜃(x𝑖)i
.(10)
3.1.5 Adaptive Search.At inference time, we initiate the retrieval
process with initial triples of topic entities and compute their plau-
sibility scores using the trained MLP, 𝑓𝜃(x). Triples exceeding a
plausibility threshold 𝜏are retained, and their tail entities are used
as frontier entities in the next hop. This expansion–filtering cy-
cle continues until no new triples satisfy the threshold. However,
using a fixed threshold 𝜏can be problematic: it may be too strict
in sparse hypergraphs, limiting retrieval, or too lenient in dense
hypergraphs, leading to an overload of irrelevant triples. To mit-
igate this, we implement an adaptive thresholding strategy. We
initialize with 𝜏0=0.5, allow a maximum of 𝑁max=5threshold
reductions, and define 𝑀= 50as the minimum acceptable num-
ber of hyperedges per hop. At hop 𝑖, we retrieve the set of triples,
T𝑞,≥𝜏𝑗={(𝑒ℎ,h,𝑒𝑡)|𝑓𝜃(𝑥)≥𝜏𝑗}under the current threshold 𝜏𝑗. If
|T𝑞,≥𝜏𝑗|<𝑀, we iteratively reduce the threshold as follows:
𝜏𝑗+1=𝜏𝑗−𝑐, 𝑗=0,...,𝑁 max−1,(11)
where𝑐= 0.1is the decay factor. This process continues until
||T𝑞,≥𝜏𝑗||≥𝑀 or the reduction limit is reached. To further adapt to
structural variations in the hypergraph, we incorporate a density-
aware thresholding policy. Given the density of the hypergraph
Δ(G) and the predefined lower and upper bounds ΔloandΔup,
we classify the hypergraph and adjust 𝜏0accordingly to balance
coverage and precision as follows:
MG= 
Mlow,Δ(G)≤Δ lo,
Mmid,Δ lo<Δ(G)≤Δ up,
Mhigh,Δ(G)>Δ up(12)
After convergence or exhaustion of threshold reduction attempts,
the retrieval strategy is adjusted based on the assigned graph density
category. For low-density graphs ( Mlow), the retriever selects from
previously discarded triples those that satisfy the final plausibility
threshold. For medium and high-density graphs ( MmidandMhigh),
the strategy additionally expands from the tail entities of these
newly accepted triples to increase the depth of reasoning. This
density-aware adjustment prevents over-retrieval in sparse graphs
while enabling more profound and broader exploration in dense
graphs. To further control expansion in high-density settings, where
the number of candidate hyperedges may become excessive, we
impose an upper bound on the number of retrieved triples per
hop. This constraint effectively limits entity expansion, accelerates
retrieval, and reduces the inclusion of low-utility information.
3.1.6 Budget-aware Contextualized Generator.After completion
of the retrieval process, we organize the selected elements into a
structured input for the generator. Following the context layout
protocol of HyperGraphRAG [ 25], we include (i) entities and their
associated descriptions, (ii) hyperedges along with their participat-
ing entities, and (iii) supporting source text chunks linked to each

HyperRAG: Reasoning N-ary Facts over Hypergraphs for Retrieval Augmented Generation WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates.
entity or hyperedge. Due to input length constraints, we prioritize
components based on their utility. As shown in the ablation study
of HyperGraphRAG, 𝑛-ary relational facts (i.e., hyperedges) con-
tribute the most to reasoning performance, followed by entities
and then source text. We therefore allocate the token budget ac-
cordingly: 50% for hyperedges, 30% for entities, and 20% for source
chunks. To further maximize informativeness, we order hyperedges
and entities according to their plausibility scores 𝑓𝜃(·), with graph
connectivity as a secondary criterion. The selected components
are then sequentially filled in the order: hyperedges, entities, and
source chunks. Components are filled in priority order and any
unused budget is passed to the next category. The contextualized
evidence resulting context , together with the original query 𝑞, is
then passed to the LLM to generate the final answerAnsweras:
Answer:=LLM(Context,𝑞).(13)
3.2 HyperMemory: Relational Chain Extraction
To improve interpretability and context awareness in path retrieval,
we avoid naive top- 𝑘heuristics with LLM-guided scoring that lever-
ages the model’s parametric memory to assess the salience of hyper-
edges and entities. This enables retrieval to be guided by contextual
priors and query intent, facilitating more targeted and meaningful
relational exploration.
3.2.1 Memory-Guided Beam Retriever.Specifically, we design beam
search with width 𝑤= 3and depth 𝑑= 3, where𝑤denotes the
number of paths ranked in the top order retained at each iteration,
and𝑑specifies the maximum number of expansion steps. Following
the process of theLearnable Relational Chain Retriever, we begin by
identifying the set of topic entities E𝑞from the input query 𝑞using
an LLM-based entity extractor. For each topic entity 𝑒𝑠∈E𝑞, we
retrieve its incident hyperedge set F𝑒𝑠. Each hyperedge 𝑓𝑛∈F𝑒𝑠is
scored for relevance to both𝑒 𝑠and𝑞using a prompt𝑝 edge:
SF(𝑓𝑛|𝑒𝑠,𝑞)∼LLM(𝑝 edge,𝑒𝑠,𝑓𝑛,𝑞).(14)
We retain the top- 𝑤hyperedges, denoted 𝐻+
𝑒𝑠, based on the score
SF(·). Next, for each 𝑓𝑛∈F+
𝑒𝑠, we identify unvisited tail entities
𝑒𝑡and score their relevance using a second prompt𝑝 entity:
SE(𝑒𝑡|𝑓𝑛,𝑞) ∼LLM(𝑝 entity, 𝑓𝑛, 𝑒𝑡, 𝑞).(15)
Next, each resulting candidate triple (𝑒𝑠,𝑓𝑛,𝑒𝑡)receives a weighted
composite score as follows:
S(𝑒𝑠,𝑓𝑛,𝑒𝑡)=SF(𝑓𝑛|𝑒𝑠,𝑞) ·SE(𝑒𝑡|𝑓𝑛,𝑞).(16)
From the current set of candidate triples, we retain the top- 𝑤based
on the final triple scorer S(·). The tail entities of these selected paths
define the next expansion frontier. At each depth 𝑖, we evaluate
whether the accumulated evidence suffices to answer the query. All
retrieved triples are assembled into a contextualized component 𝐶𝑖,
which is passed to the LLM for an evidence sufficiency check:
LLM(𝑝 ctx,𝐶𝑖,𝑞) −→ {yes,no},Reason.(17)
If the result is yes, terminate the search and proceed to generation.
Otherwise, if𝑖<𝑑, the search continues until the next iteration.3.2.2 Contextualized Generator.The entities and hyperedges re-
trieved are organized in a fixed format context, as defined in Eq.(13).
This contextualized evidence Context , combined with the original
query𝑞, is then passed to the LLM to generate the finalAnswer.
4 Experiments
We quantitatively evaluate the effectiveness and efficiency of Hyper-
Retriever against RAG baselines both in-domain and cross-domain
settings. Ablation studies highlight the benefits of adaptive expan-
sion and𝑛-ary relational chain learning, complemented by qual-
itative analyzes that illustrate the precision and efficiency of the
adaptive retrieval process.
4.1 Experimental Setup
4.1.1 Datasets.We conduct experiments under both open-domain
and closed-domain multi-hop question answering (QA) settings.
For in-domain evaluation, we use three widely adopted bench-
mark datasets: HotpotQA [ 42], MuSiQue [ 38], and 2WikiMulti-
HopQA [ 16]. To evaluate cross-domain generalization, we adopt
the WikiTopics-CLQA dataset [ 11], which tests zero-shot induc-
tive reasoning over unseen entities and relations at inference time.
Comprehensive dataset statistics are summarized in Appendix B.2.
4.1.2 Evaluation Metrics.We employ four standard metrics to as-
sess performance, aligning with established protocols for each
benchmark type. For open-domain QA datasets, where the objective
is precise answer generation, we report Exact Match (EM) and F1
scores. For WikiTopics-CLQA, which involves ranking correct enti-
ties from a candidate list, we utilize Mean Reciprocal Rank (MRR)
and Hits@k to evaluate retrieval fidelity. All metrics are reported as
percentages (%), with higher values indicating better performance.
4.1.3 Baselines.To evaluate the effectiveness of our approach, we
compare HyperRAG with RAG baselines with varying retrieval
granularities, enabling a systematic analysis of how evidence struc-
ture affects retrieval effectiveness and answer generation in both
open- and closed-domain settings. Specifically, we include: RAP-
TOR [ 33], which retrieves tree-structured nodes; HippoRAG [ 14],
which retrieves free-text chunks; ToG [ 37], which retrieves rela-
tional subgraphs; and HyperGraphRAG [ 25], which retrieves a
heterogeneous mixture of entities, relations, and textual spans.
4.1.4 Implementation Details.All baselines and our proposed meth-
ods utilize gpt-4o-mini as the core model for both graph construc-
tion and question answering. For HyperRetriever, we additionally
employ the pretrained text encoder gte-large-en-v1.5 to pro-
duce dense embeddings for entities, relations, and queries. With
434M parameters, this GTE-family model achieves strong perfor-
mance on English retrieval benchmarks, such as MTEB, and of-
fers an efficient balance between inference speed and embedding
quality, making it well-suited for semantic subgraph retrieval. All
experiments were implemented in Python 3.11.13 with CUDA 12.8
and conducted on a single NVIDIA RTX 3090 (24 GB). Peak GPU
memory usage remained within 24 GB due to dynamic allocation.
4.2 Open-domain Answering Performance
4.2.1 Setup.ForHyperRetriever, a lightweight MLP 𝑓𝜃scores the
plausibility of candidate hyperedges, enabling aggressive pruning

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates. Wen-Sheng Lien et al.
TopicRAPTOR HippoRAG ToG HyperGraphRAG HyperRetriever HyperMemory Rel. Gain (%)
MRR Hits@10 MRR Hits@10 MRR Hits@10 MRR Hits@10 MRR Hits@10 MRR Hits@10 MRR Hits@10
art3.44 4.13 8.42 9.77 2.99 3.20 17.18 21.68 19.31 24.3115.63 19.17 12.40 12.13
award20.57 25.13 32.80 38.65 8.70 9.35 51.64 63.43 52.66 65.2847.34 56.98 1.98 2.93
edu4.94 5.90 23.82 26.37 9.09 9.49 43.44 50.05 44.79 51.6341.68 46.95 3.11 3.16
health18.85 22.04 25.72 29.59 7.14 7.95 31.46 37.94 32.68 39.2627.48 33.13 3.88 3.48
infra10.95 12.79 23.88 27.11 9.87 10.67 37.18 44.82 38.92 45.7735.77 41.69 4.68 2.12
loc16.55 18.68 19.88 23.08 3.45 3.83 29.92 34.38 31.80 36.8530.73 35.95 6.28 7.18
org12.00 14.54 36.20 41.70 6.61 7.3364.68 74.89 62.87 71.21 52.26 59.84 -2.80 -4.91
people10.74 13.10 15.39 18.28 3.90 4.40 20.67 28.10 21.62 28.4818.96 25.29 4.60 1.35
sci6.84 8.66 15.62 18.86 6.87 7.2825.92 34.54 25.15 32.30 21.50 27.53 -2.97 -6.49
sport11.31 13.28 22.78 26.01 7.51 8.53 37.40 44.91 39.37 45.5633.64 39.72 5.27 1.45
tax10.48 11.08 24.77 26.65 6.22 6.50 35.15 40.94 37.20 40.9833.65 38.19 5.83 0.10
AVG11.52 13.58 22.66 26.01 6.58 7.14 35.88 43.24 36.94 43.7832.60 38.59 2.95 1.23
Table 1: Performance comparison of domain generalization across 11 diverse topics. The “Rel. Gain” column highlights the
substantial relative improvement of our approach over the best baseline, averaged across all domains (metrics in %).
ModelHotpotQA MuSiQue 2WikiMultiHopQA
EM(%) F1(%) EM(%) F1(%) EM(%) F1(%)
RAPTOR 35.50 41.56 15.00 16.31 22.50 22.95
HippoRAG 49.50 55.8714.50 17.43 30.00 30.44
ToG 10.08 11.00 2.70 2.69 5.20 5.34
HyperGraphRAG51.0042.6922.00 20.02 42.5030.17
HyperRetriever 42.50 43.65 13.50 14.15 34.00 34.06
HyperMemory 35.50 41.51 8.00 12.96 31.50 32.56
Rel. Gain (%) -16.67 -21.87 -38.64 -29.32 -20.00 11.89
Table 2: Performance comparison on HotpotQA, MuSiQue,
and 2WikiMultiHopQA. Rel. Gain (%) indicates the relative
performance gains achieved by our model compared with
the best baselines. The best results are bolded, and the second
best are underlined .
that reduces traversal complexity without compromising reason-
ing quality. ForHyperMemory, we set beam width 𝑤= 3and
depth𝑑=3to balance retrieval coverage against computational
cost. Comprehensive prompt definitions for edge scoring ( 𝑝edge),
entity ranking ( 𝑝entity), context evaluation ( 𝑝ctx), and generation
are provided in the Appendix.
4.2.2 Results.Table 2 details the Exact Match (EM) and F1 scores
across three open-domain QA benchmarks. HyperRetriever consis-
tently outperforms the HyperMemory variant on HotpotQA and
MuSiQue, demonstrating superior capability in identifying eviden-
tial relational chains. This advantage is attributed to its learnable
MLP-based plausibility scorer and density-aware expansion strat-
egy, which affords precise control over retrieval depth. In contrast,
HyperMemory relies on the fixed parametric memory of the LLM,
rendering it less adaptable to domain-specific relational patterns.
When compared to external KG-based RAG baselines, we observe
a performance divergence based on graph topology. On HotpotQA
and MuSiQue, HyperRetriever exhibits a performance gap (e.g.,38.64% lower EM on MuSiQue), likely because these datasets re-
quire the rigid structural guidance of explicit KG priors for cross-
document navigation. However, on 2WikiMultiHopQA, HyperRe-
triever reverses this trend, achieving an 11.89% relative F1 improve-
ment. This suggests that while KG priors aid in sparse settings,
HyperRetriever is uniquely effective at exploiting the denser, com-
plex relational contexts found in 2WikiMultiHopQA.
4.3 Closed-domain Generalization Performance
To evaluate adaptability to closed-domain 𝑛-ary knowledge graphs,
we evaluate the performance ofHyperRAGon the WikiTopics-
CLQA dataset (Table 1). The results demonstrate a strong gener-
alization across diverse topic-specific hypergraphs. In particular,
our learnable variant, HyperRetriever, achieved the highest over-
all answer precision, with average improvements of 2.95% (MRR)
and 1.23% (Hits@10) compared to the second-best baseline, Hyper-
GraphRAG. These gains are statistically significant ( 𝑝≪ 0.001),
with𝑡-test values of1 .46×10−17for MRR and2 .41×10−6for Hits@10,
suggesting the empirical reliability of our approach. HyperRetriever
secures top performance in 9 out of the 11 categories—for instance,
achieving relative gains of 12.40% (MRR) and 12.13% (Hits@10) in
theArtdomain—and consistently ranks second in the remaining
two. This broad efficacy highlights the robustness of HyperRe-
triever’s adaptive retrieval mechanism. Unlike baselines that are
sensitive to domain-specific graph density, HyperRetriever’s learn-
able MLP scorer dynamically calibrates its expansion strategy to suit
varying𝑛-ary topologies, ensuring high precision even in complex
reasoning tasks. In contrast, our memory-guided variant,Hyper-
Memory, consistently underperforms against to HyperRetriever.
This variant serves as a critical ablation to probe the limitations of
an LLM’s intrinsic parametric memory for 𝑛-ary retrieval. The re-
sults confirm that prompt-based scoring alone, without the explicit
structural learning provided by HyperRetriever, is insufficient for
multi-hop reasoning in closed domains.

HyperRAG: Reasoning N-ary Facts over Hypergraphs for Retrieval Augmented Generation WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates.
TopicFull w/o Entities w/o Hyperedges w/o Chunks w/o Adaptive Search w Binary KG
MRR Hits@10 MRR Hits@10 MRR Hits@10 MRR Hits@10 MRR Hits@10 MRR Hits@10
art26.03 31.00 27.2831.00 24.03 27.00 24.17 27.00 26.33 31.00 14.00 15.00
award56.9170.00 43.22 61.00 55.95 69.00 55.01 66.00 52.98 66.00 48.92 53.00
edu49.0056.00 43.24 52.00 47.93 52.00 42.67 47.00 47.53 53.00 38.20 42.00
health41.2547.00 37.17 43.00 37.70 40.00 39.33 47.00 39.20 46.00 36.17 39.00
infra34.85 43.00 35.17 43.00 30.87 39.0038.7544.00 35.50 45.00 30.50 32.00
loc38.75 42.5044.5847.50 37.50 40.00 33.13 37.50 41.67 47.50 39.58 42.50
org46.79 58.9758.7565.00 45.92 55.00 53.00 60.00 38.07 45.00 47.50 47.50
people14.20 22.0021.2328.00 13.73 19.00 20.03 26.00 13.37 20.00 19.33 22.00
sci25.91 36.00 18.67 22.00 24.53 32.0026.0938.00 21.14 32.00 24.00 27.00
sport31.04 40.00 35.83 40.00 35.00 45.50 29.58 40.00 33.33 37.5042.0847.50
tax36.25 40.00 29.17 35.00 33.54 36.25 33.13 36.2536.8840.00 35.42 37.50
AVG36.4540.59 35.85 42.50 35.15 41.34 35.90 42.61 35.64 42.91 34.15 36.82
Table 3: Ablation on the Contribution of Context Formation and Adaptive Search. The full model incorporates all components
essential for context formation, including entities, hyperedges involved in learnable relational chains, and retrieved chunks.
The best results in MRR are bolded, and the best in Hits@10 are underlined .
Dimension RAPTOR [33] HippoRAG [14] ToG [37] HyperGraphRAG [25] OG-RAG [34] HyperRetriever / Memory
Structure type Doc tree (summ.) KG (binary) KG (binary) Hypergraph (𝑛-ary) Object graph (mostly bin.) Hypergraph (𝑛-ary)
Unit of fact Passage / summary Entity-entity edge Step / subgoal Hyperedge (𝑛-ary fact) Object-object edge Hyperedge (𝑛-ary fact)
Candidate growth Additive (levels) Additive on edge LLM-var. Additive on hyperedges Additive on objects Additive on hyperedges
Per-query overhead Tokens onlyO(𝑛−𝑘)Var.O(1)†O(1) O(1)†
Depth for reasoning chain Deep Deep (pairwise) LLM-var. Shallow (𝑛-ary edges) Deep (pairwise) Shallow (𝑛-ary edges)
Retrieval strategy Dense tree search Graph walk + dense LLM on graph Static Object-centric walk Adaptive / LLM on graph
LLM at retrieval Low-Med Low Med-High (LLM) Low Low Low / Med (LLM)
Ontology✗ ✗ ✗ ✗ ✓ ✗
Table 4: Method Comparison. HyperRetriever utilizes adaptive search on 𝑛ary hyperedges, enabling higher-order reasoning
with shallow chains and near constant per-query retrieval overhead O(1). In contrast, static or object-centric walks on binary
graphs entail deeper pairwise chains and materialization cost. †denotes bounded arity; ✓indicates an ontology requirement.
4.4 Ablation Study
To evaluate the effectiveness of our approach, we conduct a series
of ablation studies targeting two key aspects: (i) the contribution
of individual components to context formation, and (ii) the impact
of the adaptive search policy on retrieval performance.
4.4.1 Higher-Order Reasoning Chains.Compared with binary KG
RAG,HyperRAGsupports higher-order reasoning on 𝑛-ary hyper-
graphs. An𝑛-ary hyperedge jointly binds multiple entities and roles,
capturing fine-grained dependencies beyond pairwise links. Exploit-
ing this structure yields shallower yet more expressive reasoning
chains, enabling the model to surface key evidence without multi-
hop traversal. Empirically (Table 3), replacing the 𝑛-ary structure
with a binary one lowers average MRR from36 .45%to34.15%(-2.3%)
and the average Hits @ 10 from40 .59%to36.82%(-3.77%), indicat-
ing gains in both accuracy and efficiency. Additional qualitative
examples appear in Appendix C.
4.4.2 Impact of Context Formation.Table 3 presents a component-
wise ablation study conducted on a representative 1% subset to
isolate the contributions of (i) entities, (ii) structural relations (hy-
peredges), and (iii) textual context. We observe that removing any
component consistently degrades Mean Reciprocal Rank (MRR),though Hits@10 exhibits higher variance. This divergence high-
lights the distinction between ranking fidelity (MRR) and candidate
inclusion (Hits@10). For instance, in theorgandlocdomains,
certain ablated variants maintain competitive Hits@10 scores but
suffer sharp declines in MRR. This indicates that while the correct
answer remains within the top candidates, the loss of structural or
semantic signals causes it to drift down the ranking list, degrading
precision. Crucially, hyperedges emerge as the dominant factor in
effective context formation. Their exclusion precipitates the most
significant performance drops across both metrics, underscoring
the necessity of high-order topological structure for reasoning. In
contrast, removing entities yields less severe degradation, as enti-
ties primarily provide node-level descriptions, whereas hyperedges
capture the joint dependencies between them. Text chunks offer
complementary unstructured semantics but lack the relational preci-
sion of the graph structure. Ultimately, the superior performance of
the full model validates the synergistic integration of entity-aware
signals, hypergraph topology, and adaptive textual evidence.
4.4.3 Impact of Adaptive Search.Removing the adaptive search
component results in a noticeable decline in MRR across most cate-
gories, whereas its impact on Hit@10 is minimal and in some cases

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates. Wen-Sheng Lien et al.
0 100 200 300 400
Average Retrieval Time (s)01020304050Average Hits@10 (%)
Figure 3: The visualization shows the efficiency-effectiveness
tradeoff in multi-hop QA: retrieval time ( 𝑥-axis), answer qual-
ity (Hits@10, 𝑦-axis), and context volume (bubble size, log-
scaled by retrieved tokens).
(e.g.,infra,loc), even marginally positive. This pattern suggests
that while correct answers remain retrievable among the top 10
candidates, they tend to be ranked lower in the absence of adaptive
search, resulting in a reduced overall ranking precision.
4.5 Efficiency Study
4.5.1 Setup.To assess retrieval efficiency, we draw a stratified
1% from each WikiTopics-CLQA category, yielding approximately
1,000 questions evenly distributed across 11 topic domains, and
evaluate all baselines on this set. Figure 3 depicts the three-way
trade off among retrieval time ( 𝑥-axis), Hits@10 accuracy ( 𝑦-axis),
and context volume (bubble size, logarithmically scaled by retrieved
tokens). Models in the upper left quadrant achieve the best balance
between efficiency and effectiveness, combining low latency with
high Hits@10 while retrieving compact contexts.
4.5.2 Empirical Evidence.HyperRetriever achieves the shortest
retrieval time and the highest Hits@10. Although it retrieves more
tokens than some baselines, top performers consistently rely on
larger contexts, highlighting a common trade-off between answer
quality and retrieval volume. Our empirical findings align with
the theoretical analysis in §2.2.HyperRetrieveremploys adaptive
search over 𝑛-ary hyperedges, enabling higher-order reasoning
with shallow chains and nearly constant per query overhead O(1).
In contrast, static or object-centric walks in binary graphs require
deeper pairwise chains and incur an event materialization cost
O(𝑛−𝑘) . We further benchmark our approach against five publicly
available graph-based RAG systems, covering both 𝑛-ary and binary
KG designs, and summarize in Table 4.
5 Related Work
Retrieval-Augmented Generation.RAG fundamentally aug-
ments the parametric memory of LLMs with external data, serving
as a critical countermeasure against hallucination in knowledge-
intensive tasks. The standard pipeline operates by retrieving top- 𝑘
document chunks via dense similarity search before conditioning
generation on this augmented context [ 2,12,17]. However, conven-
tional dense retrieval methods [ 6,20] treat data as flat text, often
overlooking the complex structural and relational signals requiredfor deep reasoning. To address this, iterative multi-step retrieval
approaches have been proposed [ 18,36,39]. Yet, these methods of-
ten suffer from diminishing returns: they increase inference latency
and retrieve redundant information that dilutes the context signal.
This noise contributes to the “lost-in-the-middle” effect, where fi-
nite context windows prevent the LLM from effectively attending
to dispersed evidence [24, 41].
Graph-based RAG.Graph-based RAG frameworks incorporate
inter-document and inter-entity relationships into retrieval to en-
hance coverage and contextual relevance [ 3,15,31,32]. Early ap-
proaches queried curated KGs (e.g., WikiData, Freebase) for factual
triples or reasoning chains [ 4,22,27,40], while recent methods fuse
KGs with unstructured text [ 8,23] or build task-specific graphs from
raw corpora [ 7]. To improve efficiency, LightRAG [ 13], HippoRAG
[14], and MiniRAG [ 10] adopt graph indexing via entity links, per-
sonalized PageRank, or incremental updates [ 28,29]. However,
KG-based RAGs often face a trade-off between breadth and pre-
cision: broader retrieval increases noise, while narrower retrieval
risks omitting key evidence. Methods using fixed substructures (e.g.,
paths, chunks) simplify reasoning [ 33,44] but may miss global con-
text, and challenges are amplified by LLM context window limits,
vast KG search spaces [ 18,30,37], and the high latency of iterative
queries [ 37]. Moreover, most graph-based RAG methods rely on
binary relational facts, limiting the expressiveness and coverage
of knowledge. Hypergraph-based representations capture richer 𝑛-
ary relational structures [ 26]. HyperGraphRAG [ 25] advances this
line by leveraging 𝑛-ary hypergraphs, outperforming conventional
KG-based RAGs, yet suffers from noisy retrieval and reliance on
dense retrievers. OG-RAG [ 34] addresses these issues by grounding
hyperedge construction and retrieval in domain-specific ontologies,
enabling more accurate and interpretable evidence aggregation.
However, its dependence on high-quality ontologies constrains
scalability in fast-changing or low-resource domains. Most graph-
based and hypergraph-based RAG methods still face challenges,
particularly due to the use of static or object-centric walks on binary
graphs, which entail deeper pairwise chains and higher material-
ization costs. Table 4 compares existing methods withHyperRAG.
6 Conclusion
We introduced HyperRAG, a novel framework that advances multi-
hop Question Answering by shifting the retrieval paradigm from
binary triples to 𝑛-ary hypergraphs featuring two strategies: Hyper-
Retriever, designed for precise, structure-aware evidential reason-
ing, and HyperMemory, which leverages dynamic, memory-guided
path expansion. Empirical results demonstrate that HyperRAG
effectively bridges reasoning gaps by enabling shallower, more
semantically complete retrieval chains. Notably, HyperRetriever
consistently outperforms strong baselines across diverse open- and
closed-domain datasets, proving that modeling high-order depen-
dencies is crucial for accurate and interpretable RAG systems.
Acknowledgments
This work is partially supported by the National Science and Tech-
nology Council (NSTC), Taiwan (Grants: NSTC-112-2221-E-A49-
059-MY3, NSTC-112-2221-E-A49-094-MY3, 114-2222-E-A49-004, and
114-2639-E-A49-001-ASP).

HyperRAG: Reasoning N-ary Facts over Hypergraphs for Retrieval Augmented Generation WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates.
References
[1]Serge Abiteboul, Richard Hull, and Victor Vianu. 1995.Foundations of Databases.
Addison-Wesley.
[2]Gabor Angeli, Melvin Jose Johnson Premkumar, and Christopher D. Manning.
2015. Leveraging Linguistic Structure For Open Domain Information Extraction.
InProceedings of the Annual Meeting of the Association for Computational Linguis-
tics and the 7th Int’l Joint Conference on Natural Language Processing, Chengqing
Zong and Michael Strube (Eds.). 344–354.
[3]Mariam Barry, Gaetan Caillaut, Pierre Halftermeyer, Raheel Qader, Mehdi
Mouayad, Fabrice Le Deit, Dimitri Cariolaro, and Joseph Gesnouin. 2025.
GraphRAG: Leveraging Graph-Based Efficiency to Minimize Hallucinations in
LLM-Driven RAG for Finance Data. InProceedings of the Workshop on Generative
AI and Knowledge Graphs (GenAIK). 54–65.
[4]Kurt Bollacker, Colin Evans, Praveen Paritosh, Tim Sturge, and Jamie Taylor.
2008. Freebase: a collaboratively created graph database for structuring human
knowledge. InProceedings of the ACM SIGMOD Int’l Conf. on Management of
Data (SIGMOD ’08). 1247–1250.
[5]Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu.
2024. M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity
Text Embeddings Through Self-Knowledge Distillation. InFindings of the Associ-
ation for Computational Linguistics: ACL 2024. 2318–2335.
[6]Gabriel de Souza P. Moreira, Radek Osmulski, Mengyao Xu, Ronay Ak, Benedikt
Schifferer, Mengyao Xu, Ronay Ak, Benedikt Schifferer, and Even Oldridge. 2024.
NV-Retriever: Improving text embedding models with effective hard-negative
mining. arXiv:2407.15831
[7]Jialin Dong, Bahare Fatemi, Bryan Perozzi, Lin F. Yang, and Anton Tsitsulin. 2024.
Don’t Forget to Connect! Improving RAG with Graph-based Reranking.arXiv
preprint arXiv: 2405.18414(2024).
[8]Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva
Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan
Larson. 2025. From Local to Global: A Graph RAG Approach to Query-Focused
Summarization. arXiv:2404.16130
[9]Ronald Fagin. 1977. Multivalued Dependencies and a New Normal Form for
Relational Databases.ACM Transactions on Database Systems2, 3 (Sept. 1977),
262–278. doi:10.1145/320557.320571
[10] Tianyu Fan, Jingyuan Wang, Xubin Ren, and Chao Huang. 2025. MiniRAG:
Towards Extremely Simple Retrieval-Augmented Generation.arXiv preprint
arXiv: 2501.06713(2025).
[11] Jianfei Gao, Yangze Zhou, Jincheng Zhou, and Bruno Ribeiro. 2023. Double
Equivariance for Inductive Link Prediction for Both New Nodes and New Relation
Types. InNeurIPS 2023 Workshop: New Frontiers in Graph Learning.
[12] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Ji-
awei Sun, Meng Wang, and Haofen Wang. 2024. Retrieval-Augmented Generation
for Large Language Models: A Survey. arXiv:2312.10997
[13] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2024. LightRAG:
Simple and Fast Retrieval-Augmented Generation.arXiv preprint arXiv: 2410.05779
(2024).
[14] Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su.
2024. HippoRAG: Neurobiologically Inspired Long-Term Memory for Large
Language Models. InThe Annual Conf. on Neural Information Processing Systems.
[15] Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Jiayuan Ding, Yongjia Lei, Ma-
hantesh Halappanavar, Ryan A. Rossi, Subhabrata Mukherjee, Xianfeng Tang, Qi
He, Zhigang Hua, Bo Long, Tong Zhao, Neil Shah, Amin Javari, Yinglong Xia, and
Jiliang Tang. 2025. Retrieval-Augmented Generation with Graphs (GraphRAG).
arXiv:2501.00309
[16] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020.
Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reason-
ing Steps. InProceedings of the Int’l Conf. on Computational Linguistics. 6609–6625.
[17] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian
Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting
Liu. 2025. A Survey on Hallucination in Large Language Models: Principles,
Taxonomy, Challenges, and Open Questions.ACM Transactions on Information
Systems43, 2 (Jan. 2025), 1–55.
[18] Zhengbao Jiang, Frank Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-
Yu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active Retrieval
Augmented Generation. InProceedings of the Conf. on Empirical Methods in
Natural Language Processing. 7969–7992.
[19] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey
Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for Open-
Domain Question Answering. InProceedings of the Conf. on Empirical Methods in
Natural Language Processing (EMNLP). 6769–6781.
[20] Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi,
Bryan Catanzaro, and Wei Ping. 2025. NV-Embed: Improved Techniques for
Training LLMs as Generalist Embedding Models. InInt’l Conf. on Learning Repre-
sentations.
[21] Mufei Li, Siqi Miao, and Pan Li. 2025. Simple is Effective: The Roles of Graphs
and Large Language Models in Knowledge-Graph-Based Retrieval-Augmented
Generation. InInt’l Conf. on Learning Representations.[22] Shiyang Li, Yifan Gao, Haoming Jiang, Qingyu Yin, Zheng Li, Xifeng Yan, Chao
Zhang, and Bing Yin. 2023. Graph Reasoning for Question Answering with
Triplet Retrieval. InFindings of the Association for Computational Linguistics: ACL
2023. 3366–3375.
[23] Lei Liang, Zhongpu Bo, Zhengke Gui, Zhongshu Zhu, Ling Zhong, Peilong
Zhao, Mengshu Sun, Zhiqiang Zhang, Jun Zhou, Wenguang Chen, Wen Zhang,
and Huajun Chen. 2025. KAG: Boosting LLMs in Professional Domains via
Knowledge Augmented Generation. InCompanion Proceedings of the ACM on
Web Conf.334–343.
[24] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua,
Fabio Petroni, and Percy Liang. 2024. Lost in the Middle: How Language Models
Use Long Contexts.Transactions of the Association for Computational Linguistics
12 (2024), 157–173.
[25] Haoran Luo, Haihong E, Guanting Chen, Yandan Zheng, Xiaobao Wu, Yikai Guo,
Qika Lin, Yu Feng, Zemin Kuang, Meina Song, Yifan Zhu, and Luu Anh Tuan. 2025.
HyperGraphRAG: Retrieval-Augmented Generation via Hypergraph-Structured
Knowledge Representation. arXiv:2503.21322
[26] Haoran Luo, Haihong E, Yuhao Yang, Tianyu Yao, Yikai Guo, Zichen Tang,
Wentai Zhang, Shiyao Peng, Kaiyang Wan, Meina Song, Wei Lin, Yifan Zhu,
and Anh Tuan Luu. 2024. Text2NKG: Fine-Grained N-ary Relation Extraction
for N-ary relational Knowledge Graph Construction. InAdvances in Neural
Information Processing Systems, Vol. 37. Curran Associates, Inc., 27417–27439.
[27] Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. 2024. Reasoning
on Graphs: Faithful and Interpretable Large Language Model Reasoning. InInt’l
Conf. on Learning Representations.
[28] Linhao Luo, Zicheng Zhao, Gholamreza Haffari, Dinh Phung, Chen Gong, and
Shirui Pan. 2025. GFM-RAG: Graph Foundation Model for Retrieval Augmented
Generation.arXiv preprint arXiv:2502.01113(2025).
[29] Costas Mavromatis and George Karypis. 2025. GNN-RAG: Graph Neural Retrieval
for Efficient Large Language Model Reasoning on Knowledge Graphs. InFindings
of the Association for Computational Linguistics: ACL 2025. 16682–16699.
[30] Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, and Xindong Wu.
2024. Unifying Large Language Models and Knowledge Graphs: A Roadmap.
IEEE Transactions on Knowledge and Data Engineering36, 7 (2024), 3580–3599.
[31] Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, Chuntao Hong,
Yan Zhang, and Siliang Tang. 2024. Graph Retrieval-Augmented Generation: A
Survey. arXiv:2408.08921
[32] Ian Robinson, Jim Webber, and Emil Eifrem. 2015.Graph Databases: New Oppor-
tunities for Connected Data(2nd ed.). O’Reilly Media, Inc.
[33] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and
Christopher D Manning. 2024. RAPTOR: Recursive Abstractive Processing for
Tree-Organized Retrieval. InThe Twelfth Int’l Conf. on Learning Representations.
[34] Kartik Sharma, Peeyush Kumar, and Yunqing Li. 2024. OG-RAG: Ontology-
Grounded Retrieval-Augmented Generation For Large Language Models.
arXiv:2412.15235
[35] Abraham Silberschatz, Henry F. Korth, and S. Sudarshan. 2010.Database System
Concepts(6 ed.). McGraw-Hill.
[36] Weihang Su, Yichen Tang, Qingyao Ai, Zhijing Wu, and Yiqun Liu. 2024. DRAGIN:
Dynamic Retrieval Augmented Generation based on the Real-time Information
Needs of Large Language Models. InProceedings of the Annual Meeting of the
Association for Computational Linguistics (Vol. 1: Long Papers). 12991–13013.
[37] Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun
Gong, Lionel Ni, Heung-Yeung Shum, and Jian Guo. 2024. Think-on-Graph: Deep
and Responsible Reasoning of Large Language Model on Knowledge Graph. In
The Twelfth Int’l Conf. on Learning Representations.
[38] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2022. MuSiQue: Multihop Questions via Single-hop Question Composition.
Transactions of the Association for Computational Linguistics10 (2022), 539–554.
[39] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2023. Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-
Intensive Multi-Step Questions. InProceedings of the Annual Meeting of the
Association for Computational Linguistics (Vol. 1: Long Papers). 10014–10037.
[40] Denny Vrandečić and Markus Krötzsch. 2014. Wikidata: a free collaborative
knowledgebase.Commun. ACM57, 10 (Sept. 2014), 78–85.
[41] Peng Xu, Wei Ping, Xianchao Wu, Lawrence McAfee, Chen Zhu, Zihan Liu,
Sandeep Subramanian, Evelina Bakhturina, Mohammad Shoeybi, and Bryan
Catanzaro. 2024. Retrieval meets Long Context Large Language Models. InInt’l
Conf. on Learning Representations.
[42] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan
Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for
Diverse, Explainable Multi-hop Question Answering. InProceedings of the Conf.
on Empirical Methods in Natural Language Processing. 2369–2380.
[43] Dengyong Zhou, Jiayuan Huang, and Bernhard Schölkopf. 2006. Learning with
Hypergraphs: Clustering, Classification, and Embedding. InAdvances in Neural
Information Processing Systems.
[44] Yingli Zhou, Yaodong Su, Youran Sun, Shu Wang, Taotao Wang, Runyuan He,
Yongwei Zhang, Sicong Liang, Xilin Liu, Yuchi Ma, and Yixiang Fang. 2025. In-
depth Analysis of Graph-based RAG in a Unified Framework. arXiv:2503.04338

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates. Wen-Sheng Lien et al.
A Reduction to Binary Knowledge Graphs
Definition A.1(Faithful Reduction to Binaries).Let Fbe a set of
𝑛-ary facts(𝑛≥ 3)over entitiesEwith role-typed arguments. A
reductionis a mapping Φ:P(F)→P(E×E) that introduces no
new auxiliary nodes and satisfies, for all𝐹,𝐹′⊆F:
(1)Recoverability: 𝐹is uniquely determined by Φ(𝐹) without
spurious or missing tuples; and
(2)Role preservation:argument roles in 𝐹are recoverable from
Φ(𝐹); and
(3)Multiplicity:distinct co-participation instances remain distin-
guishable (no accidental merging).
The three conditions cannot be met under a binary-only schema
Φ. Intuitively, triadic and higher-arity facts imposejointconstraints
across all arguments, whereas binaries encode onlypairwiseco-
occurrence. Removing the joint carrier hyperedge either obscures
“who did what with which role” or merges parallel events. Therefore,
an auxiliary event node (or equivalent mechanism) is necessary to
preserve tuple identity and roles. An illustrative example follows.
Example(Role Ambiguity).
𝐹1={give(Alice,Bob,Book),give(Alice,Carol,Pen)},
𝐹2={give(Alice,Bob,Pen),give(Alice,Carol,Book)}.
Naive pairwise projection (no event node):
Φ(𝐹)=gaveTo(Alice,Bob),gaveTo(Alice,Carol),
gaveItem(Alice,Book),gaveItem(Alice,Pen)
.
Then Φ(𝐹 1)=Φ(𝐹 2): the (receiver, item) pairing is unrecoverable,
violatingrecoverabilityandrole preservation.
Prompt for Entity Salience Scoring (             )
Please score the entities' contribution to the
question on a scale from 0 to 1 (the sum of the
scores of all entities is 1).
Example:
Q: Who directed the movie that won Best Picture in
1998?
Hyperedge: Titanic, directed by James Cameron, won
the Academy Award for Best Picture in 1998.
Entities: Titanic; James Cameron; 1998; Academy Award
Score: 0.3, 0.6, 0.05, 0.05
"James Cameron" is the director of Titanic, the movie
that won Best Picture in 1998. Therefore, "James
Cameron" receives the highest score. "Titanic" is the
movie in question and gets a moderate score. "1998"
and "Academy Award" provide context and get lower
scores.
---
Q: {query}
Hyperedge: {hyperedge}
Entities: {entities}
Score: 
entity
Figure 4: Prompt for Entity Salience Scoring (𝑝 entity ).
B Reproducibility Details
B.1 Hyperparameter Setting
HyperRetriever Hyperparameters.HyperRetriever is trained
using nn.BCEWithLogitsLoss with a batch size of 32, learning rateof1×10−4, and early stopping (patience = 10) over 50 epochs. For the
retrieval phase of HyperRetriever, we followed the hyperparameters
specified in the methodology: initial plausibility threshold 𝜏0=0.5,
maximum threshold reductions 𝑁max=5, minimum number of
hyperedges per question𝑀=50, and decay coefficient𝑐=0.1. To
further adapt retrieval behavior based on the graph structure, we
design hypergraph’s density lower and upper bounds Δlo=2.35
and𝐷𝑒𝑙𝑡𝑎 up=5.
HyperMemory Hyperparameters.For HyperMemory, we set
the beam width 𝑤= 3and the maximum search depth 𝑑= 3.
This approach prevents the retriever from managing an excessive
number of paths while still providing sufficient information for
effective retrieval.
B.2 Dataset Statistics
Comprehensive statistics for open-domain and closed-domain QA
benchmarks, including dataset splits, are presented in Table 5.
Dataset Train Validation Test Total
Wikitopics 89815 89726 89749 269290
HotpotQA 640 160 200 1,000
MuSiQue 640 160 200 1,000
2WikiMultiHopQA 640 160 200 1,000
Table 5: Statistics of QA benchmarks across domain settings
B.3 Github Repository
Our anonymized code is available at https://github.com/Vincent-
Lien/HyperRAG.git.
C Additional Qualitative Results
Figure 6 provides a qualitative comparison of evidential 𝑛-ary re-
lational chains extracted by the strong baseline, ToG, versus our
proposed HyperRetriever, alongside the Ground Truth (GT). The
analysis reveals that HyperRetriever exploits hypergraph topology
to preserve the semantic integrity of dense 𝑛-ary facts, resulting
in structurally concise reasoning paths. Conversely, ToG is con-
strained by binary graph decomposition, necessitating longer, more
fragmented traversal paths to capture equivalent dependencies.
D Prompt Templates
Edge Plausibility Scoring ( 𝑝edge).The template for 𝑝edgeis de-
picted in Figure 7a.
Entity Salience Scoring ( 𝑝entity).The template for 𝑝edgeis de-
picted in Figure 4.
Context Relevance Evaluation ( 𝑝ctx).The template for 𝑝ctxis
depicted in Figure 7b.
Question Answering.We generate the final answers for both Hy-
perRetriever and HyperMemory using the same prompt and dataset.
For open-domain QA benchmarks such as HotpotQA, MuSiQue
and 2WikiMultiHopQA, the answer is usually a single entity or
sentence. Therefore, we design the prompt to guide the model to-
ward a clear, single factual reply. In contrast, the closed-domain
WikiTopics-CLQA dataset expects a list of multiple entities. In this

HyperRAG: Reasoning N-ary Facts over Hypergraphs for Retrieval Augmented Generation WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates.
Prompt for Open-Domain Question Answering
---Role---
You are a helpful assistant responding to questions
about data in the tables provided.
---Goal---
Generate a response that lists exactly one entity
that can answer the user's question.
If you don't know the answer, just say so. Do not
make anything up.
Do not include information where the supporting
evidence for it is not provided.
---Target response length and format---
A JSON array containing exactly one entity name (no
other text). 
Example:
["The Romantic Englishwoman"]
---Data tables---
{context_data}
Add sections and commentary to the response as
appropriate for the length and format. Style the
response in markdown.
(a) Open-Domain Question Answering
Prompt for Closed-Domain Question Answering
---Role---
You are a helpful assistant responding to questions
about data in the tables provided.
---Goal---
Generate a response that lists exactly which entities
can answer the user's question.
If you don't know the answer, just say so. Do not
make anything up.
Do not include information where the supporting
evidence for it is not provided.
---Target response length and format---
A JSON array of entity names (no other text). 
Example:
["The Romantic Englishwoman", "Love in the
Wilderness", Brave", "Ring of Bright Water"]
---Data tables---
{context_data}
Add sections and commentary to the response as
appropriate for the length and format. Style the
response in markdown.
(b) Closed-Domain Question Answering
Figure 5: Prompt templates for (a) Open-Domain Question
Answering, and (b) Closed-Domain Question Answering.
case, we shape the prompt to ensure the model produces a list of
all relevant entities, thus ensuring the output matches the required
multi-item format.
Prompt for Open-Domain Question Answering:The template
for open-domain question answering is illustrated in Figure 5a.Prompt for Closed-Domain Question Answering:The tem-
plate for closed-domain question answering is given in Figure 5b.
Question: Which stations are connected by the same line as the line
that connects Sawajiri Station’s adjacent station?
HyperRetriever
Hanawa
Line
Hanawa
LineSawajiri
Station
Sawajiri Station is adjacent to Dobukai Station and
Jūnisho Station on the Hanawa Line.
The Hanawa Line connects various stations including
Rikuchū-Ōsato Station, Hachimantai Station, and
Kazunohanawa Station.
Kazunohanawa
Station
Question: Which genes are associated with multiple sclerosis?
GT
ToG
HyperRetriever
chst12
multiple sclerosis
Genetic association chst12 has been linked to
multiple sclerosis.
chst12
Question: Who received an award from the University of Florida
Athletic Hall of Fame?
GT
ToG
HyperRetriever
Doug dickey
was honored by
was inducted into
College Football
Hall of FameUniversity of Florida
Athletic Hall of Fame
Doug Dickey received an award from the
University of Florida Athletic Hall of Fame and
College Football Hall of Fame.
award
 Doug dickey
Doug dickey
athletic hall of fame
university of florida
university of florida
is associated with
is linked to
multiple sclerosis
chst12
genes
multiple sclerosis
GT
ToG
Kazunohanawa Station
is adjacent to
is connected byis on
Dobukai Station
Kazunohanawa StationHanawa Line
Sawajiri Station
Hanawa LineDobukai Station
Figure 6: Comparison of evidential 𝑛-ary relational chains.
We contrast Ground-Truth (GT) answers with reasoning
paths derived by ToG and HyperRetriever. While ToG op-
erates on standard knowledge graphs restricted to binary
relations, HyperRetriever leverages hypergraphs to preserve
the semantic integrity of dense𝑛-ary facts.

WWW ’26, April 13–17, 2026, Dubai, United Arab Emirates. Wen-Sheng Lien et al.
Prompt for Edge Plausibility Scoring (           )
Please retrieve %s hyperedges (each hyperedge is a passage) that contribute to answering the question and rate
their contribution on a scale from 0 to 1 (the sum of the scores of the %s hyperedges must equal 1).
Example:
Q: Where did Albert Einstein publish his paper on general relativity?
Topic Entity: Albert Einstein
Hyperedges: 
1. "In 1905, Einstein published four groundbreaking papers on the photoelectric effect, Brownian motion, special
relativity, and mass–energy equivalence in the journal Annalen der Physik." 
2. "In November 1915, Einstein presented the field equations of general relativity to the Prussian Academy of
Sciences in Berlin." 
3. "Einstein received the 1921 Nobel Prize in Physics for his explanation of the photoelectric effect." 
4. "During World War I, scientific exchange in Europe was severely limited." 
A: 
1. {{2. "In November 1915, Einstein presented the field equations of general relativity to the Prussian Academy
of Sciences in Berlin." (Score: 0.70)}}: This passage directly states where his general relativity work was
presented, making it the most relevant. 
2. {{1. "In 1905, Einstein published four groundbreaking papers on the photoelectric effect, Brownian motion,
special relativity, and mass–energy equivalence in the journal Annalen der Physik." (Score: 0.20)}}: Although
this lists multiple papers, it mentions the same journal which provides context on Einstein's publication
venues. 
3. {{4. "During World War I, scientific exchange in Europe was severely limited." (Score: 0.10)}}: Offers
historical context but does not directly answer the publication venue. 
---
Q: {query}
Topic Entity: {topic_entity}
Hyperedges: 
{hyperedges} 
A:edge
(a) Edge Plausibility Scoring (𝑝 edge)
Prompt for Context Relevance Evaluation (        )
You are given a question and a set of related knowledge statements (hyperedges), where each statement connects
multiple entities. You are also given descriptions of the involved entities. Your task is to judge whether the
provided information is sufficient to answer the question, considering your own knowledge and the given context.
Answer with either {{Yes}} or {{No}}, and explain your reasoning briefly.
Example:
Q: Who is the spouse of the person who played Hermione Granger in Harry Potter?
Entity Descriptions:
Emma Watson: British actress known for her role as Hermione Granger in Harry Potter. 
Hermione Granger: A fictional character from the Harry Potter series. 
Harry Potter: A fantasy film and book series featuring a young wizard. 
Hyperedges:
1. "Emma Watson played the role of Hermione Granger in the Harry Potter film series." 
 Connected Entities: [Emma Watson, Hermione Granger, Harry Potter] 
2. "Emma Watson is a British actress born in 1990." 
 Connected Entities: [Emma Watson] 
3. "Emma Watson has been involved in various humanitarian activities." 
 Connected Entities: [Emma Watson]
A: {{No}}. The provided statements confirm that Emma Watson played Hermione Granger, but they do not include any
information about her spouse. Additional data is needed to answer the question.
---
Q: {query}
Entity Descriptions:
{entity_descriptions}
Hyperedges:
{hyperedges}
A: 
ctx
(b) Context Relevance Evaluation (𝑝 ctx)
Figure 7: Prompt for (a) Edge Plausibility Scoring, and (b) Context Relevance Evaluation.