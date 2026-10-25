# Graph Topology Meets Energy-Driven Pathfinding: Preserving Contextual Consistency in Retrieval-Augmented Generation

**Authors**: Yufeng Li, Zaijing Li, Fuchen Zhang, Zongzong Wu, Ming Zhao, Fengxiao Tang

**Published**: 2026-10-21 16:05:44

**PDF URL**: [https://arxiv.org/pdf/2610.29933v1](https://arxiv.org/pdf/2610.29933v1)

## Abstract
Retrieval-augmented generation (RAG) frequently integrates knowledge graphs (KGs) for complex multi-hop question answering. However, their performance is increasingly bottlenecked by the knowledge structuring and reasoning processes. (i) Severe semantic loss: Distilling continuous texts into discrete symbols strips away contextual nuances and narrative coherence. (ii) Pronounced modality mismatch: Fragmented graph structures clash with the continuous texts LLMs are pre-trained on, while rewriting them into natural language introduces generative noise and omits information. To address these limitations, we propose ThreadRAG, a semantic-topology-based RAG framework for context-preserving knowledge structuring and modality-native evidence reasoning. ThreadRAG introduces a Semantic Topology Mapper (STM), which organizes retrieved documents into context-intact Semantic Nodes and constructs a Semantic Topology Graph (STG) through narrative, semantic, and lexical edges. It further incorporates Energy-Driven Pathfinding (EDP), which identifies query-relevant subgraphs via energy diffusion and deterministically linearizes them into coherent textual evidence paths. Experiments on three multi-hop QA benchmarks demonstrate that ThreadRAG consistently outperforms strong baselines. Notably, it achieves a 14.4% EM improvement over text-centric methods on HotpotQA and surpasses the best baseline by 5.8% EM on 2WikiMultihop, while reducing token cost to 23.3% of the strongest baseline.

## Full Text


<!-- PDF content starts -->

Graph Topology Meets Energy-Driven Pathfinding: Preserving Contextual
Consistency in Retrieval-Augmented Generation
Yufeng Li1, Zaijing Li2, Fuchen Zhang1, Zongzong Wu1, Ming Zhao1, Fengxiao Tang1,*
1Central South University, Changsha, China
2Harbin Institute of Technology, Shenzhen, China
liyufeng@csu.edu.cn
Abstract
Retrieval-augmented generation (RAG) fre-
quently integrates knowledge graphs (KGs) for
complex multi-hop question answering. How-
ever, their performance is increasingly bottle-
necked by the knowledge structuring and rea-
soning processes. (i) Severe semantic loss: Dis-
tilling continuous texts into discrete symbols
strips away contextual nuances and narrative
coherence. (ii) Pronounced modality mismatch:
Fragmented graph structures clash with the con-
tinuous texts LLMs are pre-trained on, while
rewriting them into natural language introduces
generative noise and omits information. To ad-
dress these limitations, we proposeThread-
RAG, a semantic-topology-based RAG frame-
work for context-preserving knowledge struc-
turing and modality-native evidence reasoning.
ThreadRAG introduces aSemanticTopology
Mapper (STM), which organizes retrieved doc-
uments into context-intact Semantic Nodes
and constructs aSemanticTopologyGraph
(STG) through narrative, semantic, and lexical
edges. It further incorporatesEnergy-Driven
Pathfinding (EDP), which identifies query-
relevant subgraphs via energy diffusion and
deterministically linearizes them into coherent
textual evidence paths. Experiments on three
multi-hop QA benchmarks demonstrate that
ThreadRAG consistently outperforms strong
baselines. Notably, it achieves a 14.4% EM
improvement over text-centric methods on Hot-
potQA and surpasses the best baseline by 5.8%
EM on 2WikiMultihop, while reducing token
cost to 23.3% of the strongest baseline.
1 Introduction
Retrieval-augmented generation (RAG) has be-
come a powerful paradigm for enhancing large
language models (LLMs) on knowledge-intensive
tasks by grounding generation in external evidence
(Gao et al., 2023). However, complex multi-hop
question answering (QA) remains challenging for
*Corresponding author.existing RAG systems (Lewis et al., 2020; Izacard
et al., 2022; Trivedi et al., 2023). Unlike single-
hop QA, multi-hop QA requires models to identify
scattered evidence across multiple documents, con-
nect them through latent relations, and perform
multi-step reasoning (Shao et al., 2023).
Graph-enhanced RAG methods address this chal-
lenge by introducing structural priors over retrieved
knowledge. Despite their promise, existing graph-
RAG paradigms often suffer from a mismatch be-
tween how knowledge is structured and how LLMs
perform reasoning. As illustrated in Figure 1, this
mismatch manifests in two key limitations. First,
existing graph construction pipelines can incur sub-
stantialsemantic loss. Triplet-based knowledge
graphs compress continuous text into rigid sym-
bolic tuples, often discarding contextual nuances,
while chunk-based graphs split documents into iso-
lated segments, leading to semantic truncation and
noisy boundaries. Such fragmentation weakens the
continuous contextual pathways required for multi-
hop reasoning. Second, existing graph representa-
tions introduce amodality mismatchat inference
time. LLMs are primarily optimized to consume co-
herent natural language sequences, whereas graph-
RAG methods typically provide fragmented graph
paths or rely on generative verbalization to con-
vert subgraphs into text. These conversions can
introduce redundancy, hallucinated transitions, or
omission of critical evidence, thereby limiting the
effectiveness of structural priors.
To address these limitations, we propose
ThreadRAG, a semantic-topology-based RAG
framework that preserves textual context dur-
ing knowledge structuring and converts graph
evidence into LLM-native reasoning sequences.
ThreadRAG consists of two key components: a
SemanticTopologyMapper (STM) for high-
fidelity knowledge structuring, and anEnergy-
DrivenPathfinding module (EDP) for query-aware
evidence selection and topological linearization.
1
arXiv:2610.29933v1  [cs.CL]  21 Oct 2026

Chunk 1
Mickey's Safari in 
Letterland  is a 1993 
educational Nintendo 
Entertainment System 
video game starr -ing 
Disney's Mickey 
Mouse , … The succe -
ssor title, Walt 
Disney 's Comics and 
Stories (WDC&S), 
described by Barrier as 
a true comic book, 
began publication with 
the Oct 1940 issue and 
had the Gottfr -edson . ..
Mickey's Safari in 
Letterland1993
Mickey Mouse
DisneyWDC&S
Barrier
starringstarringrelease yearrelease year
associated withassociated withdescribed featuresfeatures
Gottfreds
on serials
Gottfreds
on serials
Dependencies 
missing due to 
context loss
Mickey's Safari... starring 
Disney's Mickey Mouse
The successor title, Walt 
Disney's Comics...
Semantic 
truncation<Mickey's Safari, starring, Mickey Mouse>, 
<Mickey Mouse, associated with, Disney>,  ...
Fragmented triples lose semantic. 
Mickey's Safari      (starring)      Mickey Mouse      
(featured)      Gottfredson serials
Graph  paths mislead LLM reasoning.
Mickey Mouse is associated with Disney. 
WDC&S features Gottfredson.
Text rewriting omits information .
Chunk k
Chunk 2The succe -ssor title, Walt Disney's  Co-mics 
and Stories (WDC&S), ...
Chunks cause semantic truncation .Semantic Nodes 
SN n
SN 2
Mickey's Safari in Letter -land 
is a 1993 educational NES 
video game .SN 1(a) Text to STG
Triple Topological  
mechanism
SN9
SN8SN3
SN7
SN6SN4SN1
SN5SN2SN9
SN8SN3
SN7
SN6SN4SN1
SN5SN2
Modality -Aligned 
PathfindingSN1: Mickey's Safari in 
Letterland is a 1993 
educational NES video 
game, starring Disney's 
Mickey Mouse.
Semantic Topology Path
Preserves semantic context .
Aligns modality .
(c) Text to Chunk KGSN4: Mickey Mouse is a 
cartoon character cocr -
eated in 1928 by Walt 
Disney andUb Iwerks. 
SN7: The successor 
title, Walt Disney's 
Comics and Stories 
(WDC&S)...(b) Text to Triplet KGFigure 1: Comparison of graph-RAG paradigms.Left: Our proposed ThreadRAG (a) structures raw texts into a
Semantic Topology Graph (STG) and extracts Semantic Topology Paths (ST-Paths), preserving semantic context
and aligning modality for LLM deduction.Right: Existing paradigms, Triplet KG (b) and Chunk KG (c), fragment
texts into discrete components, introducing severe semantic loss and modality mismatch.
A Semantic Topology Mapper for high-fidelity
knowledge structuring.To mitigate semantic
loss introduced by conventional graph construc-
tion, STM structures retrieved documents into an
LLM-orientedSemanticTopologyGraph (STG).
Instead of extracting symbolic triplets or relying
on isolated chunks, STM organizes retrieved doc-
uments into self-containedSemantic Nodesthat
preserve local textual context. It then connects
these nodes through complementary topological
relations, including narrative, semantic, and lexical
edges. Narrative edges preserve the original textual
order, while semantic and lexical edges capture
cross-node associations beyond surface adjacency.
This design allows STG to retain the semantic con-
tinuity of raw text while exposing the relational
structure needed for evidence navigation. These
relations transform fragmented retrieved evidence
into a navigable topology without discarding the
natural context required for multi-hop reasoning.
An Energy-Driven Pathfinding module for
modality-native knowledge representation.To
bridge the modality mismatch between discrete
graph structures and sequence-based LLM reason-
ing, EDP converts query-relevant topology into
continuous textual evidence paths. Given a query,
EDP first identifies relevant Semantic Nodes as rea-
soning anchors, and then performs energy-driven
propagation and dynamic pruning over the STG
to obtain a compact, query-specific evidence sub-
graph. Instead of directly prompting the LLM with
discrete graph paths or using free-form generation
to verbalize the subgraph, EDP deterministically
linearizes the selected nodes according to their
topological and narrative structure. This process
preserves the structural priors of the graph whilepresenting evidence in a form naturally compati-
ble with LLM pretraining. The resultingSeman-
tic Topology Pathsprovide continuous, context-
preserving, and sequence-native evidence, enabling
more coherent and reliable multi-hop reasoning.
We conduct comprehensive evaluations across
three multi-hop QA benchmarks. ThreadRAG
consistently surpasses the best mainstream graph-
enhanced paradigms across all three datasets (e.g.,
achieving a 3.4% EM improvement over Know-
Trace on HotpotQA). In graph representation eval-
uations, replacing traditional global graphs with
our STG yields a 12.6% EM improvement. Fur-
thermore, our sequence-native STG-Paths strictly
dominate all prevailing knowledge representation
modalities, peaking at 0.588 EM on 2WikiMulti-
hop. These results indicate that high-fidelity knowl-
edge structuring and topological linearization sig-
nificantly boost LLM deduction accuracy. Our con-
tributions are as follows:
•We propose aSemanticTopologyMapper that
reduces semantic loss by organizing retrieved
text into a context-preservingSemanticTopology
Graph.
•We introduce anEnergy-DrivenPathfinding mod-
ule that selects query-relevant evidence and lin-
earizes discrete graph structures into sequence-
native prompts for LLM reasoning.
•We presentThreadRAG, a unified graph-
enhanced RAG framework that consistently im-
proves multi-hop QA performance over strong
baselines across three benchmarks.
2 Related Work
Semantic Loss in Graph RAG.Integrating KGs
into RAG provides explicit structural priors for
2

SN nSN nSN 2SN 2
 Query: Who had the best 
singles ranking, Roberta 
Vinci or Jorge Lozano?Subquery :[Roberta Vinci, singles ranking]
[Jorge Lozano , singles ranking ]
Local Search Top-k ContextOriginal Context: Sara Errani and 
Roberta Vinci were the defending 
champions, but Errani chose not to 
participate this year. Vinci played 
alongside Karin Knapp, ...
SN1:”Sara Errani and Roberta Vinci 
were the defending champions ”
Doc_id: doc_0 ;   Position:0 -58Anchor 
Identification
Energy Diffusion
Graph Pruning
Topological Linearization
Roberta Vinci Women's 
Singles Overview Current 
rankings.
Career -high rankings WTA 
Singles Ranking (09 May 
2016)7.
...N1
N2
N3
Discriminator
P={N1,N2...}SufficientInsufficient
Roberta VinciRetrieval STM EDPRe-Query:
Jorge Lozano tennis 
career high singles 
ranking.Data Stream
Narrative Edge
Semantic EdgeLexical Edge
Anchor Node
Control Stream
Merge operationIterative update
Only i nitial input
Hybrid 
RetrievalSemantic 
Topology GraphRetrievalSemantic Topology Mapper
(STM)Energy -Driven  Pathfinding
(EDP) 
χ1χ1χ2χ2χ3χ3...χnχn
iσ(Wh + b)
SN 1SN 1MLP
MLP
EncoderEncoderTokenization
Doc A1
SN2:SN1:
SN3:Doc A1
SN2:SN1:
SN3:SN3SN3
SN1SN1
SN2SN2SN5SN5SN7SN7SN8SN8 SN9SN9
SN6SN6SN4SN4
Doc A2
SNm:SNn:
Doc B2Doc A3
Doc B3Roberta
RobertaSNn:
SNm:
nar

cos(θ)
sem

lex
Figure 2: Overview of ThreadRAG. Given a query and top- kretrieved contexts, the STM structures raw text into
STG. The EDP then isolates a query-relevant subgraph via energy diffusion and linearizes it into modality-native
prompts. Finally, an adaptive agent uses these prompts to iteratively refine retrieval or synthesize answers.
complex reasoning. Mainstream paradigms like
GraphRAG (Edge et al., 2024) extract rigid entity-
relation triplets, which inadvertently strip away
contextual nuances and cause severe semantic loss.
To preserve semantics, early NLP QA approaches
(Fang et al., 2020; Qiu et al., 2019; Ding et al.,
2019) retain original text spans. However, their
heterogeneous node designs target GNN aggrega-
tion rather than LLM prompt reconstruction, ir-
reversibly fragmenting the logical sequence. To
better align with LLMs, recent methods such as
HippoRAG (Gutiérrez et al., 2024) utilize broader
text-chunk nodes. While mitigating symbolic ex-
traction loss, their unconstrained chunk assembly
introduces semantic truncation and structural noise.
Consequently, these paradigms suffer from seman-
tic loss, depriving LLMs of the continuous con-
textual pathways essential for multi-hop reasoning.
We address this bottleneck by proposing a topolog-
ical framework that seamlessly embeds structural
priors without compromising narrative integrity.
Modality Alignment in Graph RAG.A crit-
ical bottleneck in graph-augmented reasoning is
the inherent modality gap between discrete graphs
and continuous natural language. Several struc-
tural reasoning methods (Luo et al., 2024; Sun
et al., 2024; Tan et al., 2025) compel LLMs to
process fragmented symbolic data over discrete
entity-relation paths. This modality mismatch per-
manently strips away crucial contextual nuances
like temporal constraints. To bridge this, genera-tive rewriting approaches (Wu et al., 2024; Sarthi
et al., 2024) employ LLMs to rewrite subgraphs
into natural language, introducing substantial to-
ken overhead and hallucination risks. Conversely,
chunk-as-node approaches (Pham et al., 2026; Yue
et al., 2026) preserve text spans but merely con-
catenate nodes as isolated chunks, failing to recon-
struct cross-document narrative coherence. Thread-
RAG addresses these limitations via a Topologi-
cal Pathfinding mechanism. Using topological lin-
earization, it restructures fragments into a coherent
sequence. This aligns graph topologies with the
LLM’s native modality, enabling multi-hop com-
prehension without generative rewriting overhead.
3 ThreadRAG
In this section, we present the overall architec-
ture of ThreadRAG (Figure 2). The framework
primarily consists of an STM module that struc-
tures retrieved contexts into an STG (Sec. 3.1),
a EDP module that extracts and linearizes query-
relevant subgraphs into continuous text sequences
(Sec. 3.2), and an adaptive reasoning agent that
iteratively refines the evidence base to synthesize
the final answer (Sec. 3.3).
3.1 Semantic Topology Mapper
Existing graph-RAG methods fragment continuous
texts into discrete components, destroying narra-
tive continuity and inducing severe semantic loss.
Motivated by this, ourSemanticTopologyMapper
(STM) extracts context-intact semantic nodes and
3

interconnects them via narrative, semantic, and lex-
ical edges into an STG. This fundamentally anchors
structural reasoning priors directly into native text
sequences, thereby transforming fragmented docu-
ments into a high-fidelity network.
Semantic Node extractionGiven a query q, an
LLM extracts key entities Ekto construct a retrieval
query set Qret={q} ∪ E k. These queries are in-
dependently issued to a hybrid retrieval index over
a local Wikipedia corpus, combining BM25 lexi-
cal matching with dense semantic retrieval using
equal weights. The retrieved passages are pooled,
deduplicated, and ranked according to the fused
retrieval score to yield the top-kcontext set:
D= top-k[
qj∈Q retHybridRetrieve(q j)
(1)
For each document di∈ D, the raw text is to-
kenized into a sequence X={x 1, x2, . . . , x n}.
This sequence is processed through a lightweight
encoder to obtain contextualized hidden represen-
tations hj∈Rdfor each token xj. To identify
semantic breakpoints, each representation hjis
mapped to a boundary probability pj∈[0,1] via a
MLP(see Appendix D for detailed configurations):
pj=σ(W 2(ReLU(W 1hj+b 1)) +b 2)(2)
where σdenotes the sigmoid activation function,
andW1,W 2,b1, b2are learnable parameters. To-
kens satisfying pj> θ, given a predefined thresh-
oldθ, act as cut-off points. The continuous docu-
ment is deterministically partitioned at these bound-
aries to yield a set of Semantic Nodes V=
{v1, v2, . . . , v m}. Each node vk∈ V is formally
represented as a tuple:
vk= (t k, dk, sk, τk)(3)
where tkis the raw natural language text span, dk
is the document identifier, and [sk, τk)denotes the
exact character-level position interval of tkwithin
dk. Compared to existing LLM-based extraction
paradigms, this feed-forward mechanism achieves
context-intact segmentation with millisecond-level
latency and strictly avoids generative noise.
Semantic Topology Graph constructionGiven
the node set V, the STG is completed by construct-
ing three complementary types of edges that cap-
ture distinct relational signals between SNs. Each
edge type encodes a different dimension of inter-
node relatedness, and their weights are fused under
a unified formulation.Narrative edges Enarconnect positionally adja-
cent nodes within the same document to preserve
the original reading order. Nodes sharing diare
sorted by start position sj, with a directed edge
(vj, vj+1)∈ E naradded between consecutive pairs.
Semantic edges Esemcapture cross-document
topical relatedness. Each node viis represented
by a dense embedding hi, connecting top- kcross-
document pairs(d i̸=dj)with weight:
Sim sem(vi, vj) =hi·hj
∥hi∥∥h j∥(4)
Lexical edges Elexcapture explicit entity co-
reference. For named entities Niextracted from vi,
cross-document pairs sharing entities are connected
via Jaccard similarity:
Sim lex(vi, vj) =|Ni∩ N j|
|Ni∪ N j|(5)
The final edge weight W(v i, vj)balances local
narrative continuity against global cross-document
association:
W(v i, vj) = (1−γ) 
α·Sim sem(vi, vj)
+β·Sim lex(vi, vj)
+γ·1 nar(vi, vj)(6)
where 1nar(vi, vj) = 1 iff(vi, vj)∈ E nar,γ∈
(0,1) scales the narrative prior, and α+β= 1 . The
complete edge set E=E nar∪ E sem∪ E lexalongside
Vdefines the STGG= (V,E).
3.2 Energy-Driven Pathfinding
Current methods relying on fragmented graph com-
ponents or generative rewriting struggle with a
modality mismatch and suffer from generative
noise or critical information omission. OurEnergy-
DrivenPathfinding (EDP) module addresses this
by executing a context-preserving pathfinding
mechanism over the STG. By isolating a query-
specific subgraph and deterministically reconstruct-
ing it into continuous textual sequences, EDP over-
comes structural fragmentation via anchor-driven
energy diffusion and topological linearization.
Anchor Identification.Given the STG G= (V,E)
and a query q, the pathfinding stage begins by iden-
tifying a set of anchor nodes as entry points. Each
node viis scored by combining its semantic rel-
evance to qwith an entity-level matching signal:
si=Sim sem(hi,hq) +λ·max
ej∈EkSim sem(hi,hej)
(7)
4

where hqandhejare the embeddings of the query
and extracted entities, respectively, and λis the
entity boost weight. The top- Knodes ranked by si
form the anchor setA ⊆ V.
Energy Diffusion.With the anchor set established,
we construct an initial energy vector r. Each anchor
is assigned an energy proportional to its score, mod-
ulated by a provenance weight withat prioritizes
locally retrieved nodes:
ri=(
si·wi·(1 +µ·b i)ifv i∈ A
0otherwise(8)
bi= max
ej∈EkSim sem(hi,hej)(9)
where µcontrols the entity bonus strength. After
ℓ1-normalization of r, energy is propagated across
the STG via Personalized PageRank (PPR). The
stationary distributionpsatisfies:
p= (1−α)Wp+αr(10)
where Wis the row-normalized adjacency matrix
weighted by the fused edge weights, and α∈(0,1)
is the restart probability. This diffusion process al-
lows scattered evidence to organically accumulate
in a high-energy topological subregion.
Graph Pruning.Upon convergence, nodes with
stationary energy pifalling below a threshold ϵ
are pruned. A connectivity constraint is then en-
forced via a BFS traversal rooted at A, retaining
only reachable nodes to produce a compact, topo-
logically grounded subgraph G′= (V′,E′). The
retained nodes V′′⊆ V′(after deduplication and
budget filtering) are subsequently deterministically
linearized. We sort these nodes based on their phys-
ical coordinates (di, si), ordering them primarily
by document retrieval relevance and secondarily
by intra-document narrative sequence. The final
modality-native context is assembled as:
C= concat(t i|vi∈ V′′)(11)
This linearization operation effectively unravels
the discrete pruned graph back into a continuous
textual sequence, providing the downstream LLM
with a contextually intact and strictly modality-
aligned prompt.
3.3 Adaptive Reasoning Agent
The linearized context Cobtained from topologi-
cal pathfinding is passed to a reasoning agent that
iteratively refines its evidence base until the query
can be answered with sufficient confidence.At each iteration, a sufficiency discriminator
evaluates the current context Cagainst the query
q. IfCis deemed sufficient, the agent proceeds di-
rectly to answer synthesis; otherwise, the discrimi-
nator outputs the missing information as structured
knowledge triples, which are forwarded to the sub-
query generator to produce targeted retrieval enti-
ties. These entities are issued as new queries to
the hybrid index, and the retrieved documents are
processed through Steps 2–4 of the ThreadRAG
pipeline—segmentation, graph augmentation, and
topological pathfinding—to yield a refined context
C′that replacesCfor the next iteration.
This loop repeats for at most Niterations. Upon
termination, the final context is passed to an answer
synthesis module that reasons step by step and pro-
duces a concise answer enclosed in a structured
output tag, from which the final answer string is
extracted. The overall procedure is summarized in
Algorithm 1.
Algorithm 1Inference Process of ThreadRAG
Require:LLM M; BM25 index over corpus; max
iterations N; energy threshold ϵ; MLP threshold θ
Input:questionq
Output:linearized contextC; final answera
1:// Phase 1: Semantic Topology Mapper
2:Ek←M extract(q);Q ret← {q} ∪ E k
3:D ←top-kS
qj∈Q retHybridRetrieve(q j)
4:// Yieldsv i= (t i, di, si, τi)
5:V ←MLP_Segmentation(D, θ)
6:E ← E nar∪ E sem∪ E lex;G ←(V,E)
7:// Phase 2: Energy-Driven Pathfinding
8:A ←Identify_Anchors(G, q,E k)
9:p←PPR_Diffusion(G,Init_Energy(A))
10:V′← {v i∈ V |p i≥ϵ}
11:G′←BFS_Traversal(V′,A)
12:C←concat(Sort_by_Coordinates(v i∈ G′))
13:// Phase 3: Adaptive Reasoning Agent
14:foriter= 1toNdo
15:{FLAG,E miss} ←M eval(C, q)
16:ifFLAGthen
17:break
18:end if
19:D new←HybridRetrieve(E miss)
20:G ←STM_Augment(G,D new)
21:C←EDP_Linearize(G, q)
22:end for
23:a←M synth(C, q)
24:// Final Answer Synthesis
25:returnC, a
5

MethodsLLaMA3-8B-Instruct Qwen3.5 plus GPT-4o mini
HotpotQA 2Wiki MuSiQue HotpotQA 2Wiki MuSiQue HotpotQA 2Wiki MuSiQue
EM F1 EM F1 EM F1 EM F1 EM F1 EM F1 EM F1 EM F1 EM F1
Non-Graph RAG
RA-CoT .204 .330 .202 .261 .142 .210 .312 .429 .268 .372 .162 .249 .310 .425 .264 .359 .158 .243
ReAct .268 .379 .244 .335 .216 .311 .358 .476 .342 .487 .248 .378 .349 .475 .338 .483 .242 .383
IRCoT .318 .430 .288 .367 .224 .327 .442 .565 .374 .519 .262 .376 .438 .559 .372 .496 .258 .373
Self-Ask .256 .373 .222 .331 .190 .281 .356 .471 .326 .466 .204 .325 .348 .465 .328 .471 .200 .317
Iter-RetGen .288 .388 .268 .351 .231 .315 .424 .534 .370 .503 .250 .373 .418 .529 .366 .489 .252 .374
Graph RAG
HippoRAG .324 .403 .318 .411 .202 .274 .418 .557 .466 .569 .234 .333 .412 .549 .458 .572 .219 .327
LightRAG .286 .357 .288 .365 .182 .249 .328 .429 .342 .424 .204 .303 .312 .402 .326 .429 .194 .293
CoTKR .304 .367 .312 .366 .212 .309 .354 .411 .368 .451 .226 .321 .368 .435 .376 .487 .234 .303
EFSUM .264 .343 .266 .351 .206 .271 .322 .417 .294 .386 .224 .297 .320 .411 .316 .409 .218 .279
ToG .318 .401 .284 .355 .202 .317 .446 .557 .456 .521 .243 .351 .448 .565 .434 .512 .258 .367
PoG .336 .426 .312 .380 .230 .321 .516 .639 .518 .601 .282 .371 .508 .623 .492 .587 .278 .366
ERA-CoT .354 .439 .304 .371 .253 .350 .478 .613 .442 .553 .278 .385 .464 .608 .436 .538 .270 .379
knowtrace .396 .486 .338 .413 .292 .379 .552 .695 .546 .638 .318 .414 .534 .679 .528 .629 .320 .413
ThreadRAG .422 .513 .372 .453 .324 .417 .586 .704 .596 .671 .342 .453 .576 .685 .586 .663 .328 .437
Table 1: Main results of multi-hop QA datasets. Best results are inboldand second best are underlined .
4 Experiments
4.1 Experimental Setup
Datasets & Metrics.We evaluate ThreadRAG on
three standard open-domain multi-hop QA bench-
marks: HotpotQA (Yang et al., 2018), 2WikiMul-
tihopQA (Ho et al., 2020), and MuSiQue (Trivedi
et al., 2022). We use the same data splits as previ-
ous works (Trivedi et al., 2023; Li and Du, 2023)
for evaluation. Following IRCoT (Trivedi et al.,
2023), our retrieval corpus aggregates all support-
ing and distractor passages to simulate a realistic
open-domain setting. For evaluation, we report
Exact Match (EM) and F1 scores to measure abso-
lute correctness and token-level overlap, applying
standard text normalization to both predictions and
ground-truth answers prior to computation.
Baselines.We evaluate ThreadRAG against
five categories of baselines: text-centric multi-hop
methods reasoning purely over plain text (RA-CoT
(Wei et al., 2022), ReAct (Yao et al., 2022), IR-
CoT (Trivedi et al., 2023), Self-Ask (Press et al.,
2023), Iter-RetGen (Shao et al., 2023)), KG-Chunk
RAG methods treating text chunks as nodes (Hip-
poRAG (Gutiérrez et al., 2024), LightRAG (Guo
et al., 2024)), KG-Rewriting RAG methods con-
verting subgraphs into natural language (CoTKR
(Wu et al., 2024), EFSUM (Ko et al., 2024)), KG-
Path RAG methods reasoning along discrete struc-
tural paths (ToG (Sun et al., 2024), PoG (Tan et al.,
2025)), and KG-Triplet RAG methods prompting
with discrete relational triplets (ERA-CoT (Liu
et al., 2024), KnowTrace (Li et al., 2025)). Forfair comparison, we adapt ToG and PoG to use
the same local KG as LightRAG, replacing their
external KGs.
Implementation Details.We use Llama-3-8B
(Grattafiori et al., 2024), Qwen-3.5-Plus (Yang
et al., 2025), and GPT-4o-mini (OpenAI et al.,
2024) withnomic-embed-textfor embeddings
(Nussbaum et al., 2024). Chunk size is 1200 charac-
ters with 150 overlap. We retrieve top-5 global con-
texts via Elasticsearch using equal BM25 and dense
weights. STG settings are: 5-NN, 0.5/0.3 seman-
tic/lexical thresholds, and edge weights γ= 0.3 ,
α= 0.35 ,β= 0.65 . Pathfinding parameters are:
3 anchors, 0.5 PPR restart, 0.01 pruning threshold,
and 3 max iterations. All experiments run on 4
NVIDIA A6000 GPUs.
4.2 Inference Performance Comparison
Table 1 summarizes the main experimental results
across three multi-hop QA datasets. First, we ob-
serve that KG-Chunk methods (e.g., HippoRAG)
outperform KG-Rewriting approaches to a certain
extent (e.g., CoTKR, achieving 0.324 vs. 0.304
EM on HotpotQA with LLaMA3-8B-Instruct). We
attribute this to the generative noise and hallucina-
tions introduced when rewriting discrete subgraphs
into text. Second, iterative KG-Triplet prompting
(e.g., KnowTrace) significantly surpasses KG-Path
methods (e.g., PoG, 0.396 vs. 0.336 EM). This
indicates that faulty path retrieval can easily inject
erroneous logical chains, which LLMs struggle to
verify or reject due to the severe loss of surrounding
6

Ablation Setting Datasets
SN STG EDP HotpotQA 2Wiki MuSiQue
✓✓✓ .422 .372 .324
.238(↓18.4%).224(↓14.8%).172(↓15.2%)
✓ .326(↓9.6%).308(↓6.4%).232(↓9.2%)
✓ ✓ .398(↓2.4%).334(↓3.8%).286(↓3.8%)
✓ ✓ .388(↓3.4%).342(↓3%).292(↓3.2%)
Table 2: Ablation study of the STM (SN, STG) and
EDP modules across three datasets.
narrative context in discrete nodes.
Beyond these flawed paradigms, ThreadRAG di-
rectly grounds structural guidance within native
text sequences to achieve state-of-the-art results.
For example, it outperforms the strongest baseline
KnowTrace, by up to 5.8% EM on 2Wiki (GPT-4o
mini). Furthermore, it surpasses the text-centric
IRCoT by a substantial 14.4% EM on the complex
HotpotQA dataset (Qwen3.5 plus). These signif-
icant gains demonstrate the effectiveness of our
STG in providing high-fidelity structural priors.
By providing explicit reasoning pathways without
fragmenting the original narrative flow, Thread-
RAG bridges the modality gap, bypassing both the
semantic loss of symbolic extraction and the hallu-
cination risks of generative rewriting.
4.3 Ablation Study
Many unexplored questions surround the optimal
knowledge representation and reasoning modalities
for multi-hop QA. In this section, we conduct an
extensive ablation study to validate the core compo-
nents of ThreadRAG and summarize key findings.
The STM preserves crucial contextual in-
tegrity.As shown in Table 2, removing the entire
STM module (i.e., SN and STG) and using LLM-
extracted triplets leads to a severe decline in per-
formance, with the EM score dropping by 9.2% on
MuSiQue. Furthermore, replacing our MLP-based
Semantic Nodes (SN) with fixed-length chunks re-
sults in a noticeable 3.4% drop on HotpotQA. We
attribute this to the fact that traditional triplet extrac-
tion or arbitrary chunking inevitably destroys the
continuous narrative flow, resulting in irreversible
semantic loss that hinders the LLM’s ability to de-
duce complex answers.
EDP effectively bridges the modality gap.As
shown in Table 2, feeding the discrete STG nodes
directly into the LLM without the EDP module’s
topological linearization causes a noticeable de-
cline, such as an 3.8% decrease on the MuSiQueMethodHotpotQA MuSiQue
Const. Inf. Total Const. Inf. Total
LightRAG 26.2M 47.4k 26.25M 36.4M 57.6k 36.46M
KnowTrace 0.15M 72.6k 222.6k 0.22M 94.4k 314.4k
ThreadRAG 0 58.5k 58.5k 0 73.4k 73.4k
Table 3: Comparison of construction (Const.), inference
(Inf.), and total token costs.
dataset. We attribute this to the inherent modal-
ity mismatch between discrete graph structures and
the LLM’s native continuous language space. With-
out EDP to reconstruct the narrative sequence, the
LLM struggles to effectively interpret the frag-
mented subgraphs, which introduces reasoning
noise and degrades overall performance.
4.4 More Analysis
Cost Analysis.To quantify the computational over-
head, we measure token consumption across 100
randomly sampled queries per dataset using the
tiktoken library (GPT-4o-mini standard). As
shown in Table 3, ThreadRAG demonstrates a de-
cisive advantage in computational efficiency by
fundamentally decoupling structural knowledge
generation from expensive LLM pre-computation
across all datasets. In contrast, baseline methods
like LightRAG and KnowTrace consume massive
token budgets—up to 36.4M and 220K tokens, re-
spectively—merely to build their graph structures.
Furthermore, when evaluating the overall pipeline,
ThreadRAG operates highly efficiently: its total
token consumption accounts for merely 23.3% to
26.3% of the overhead incurred by the strongest
baseline, KnowTrace. This confirms that Thread-
RAG effectively resolves the prohibitive I/O bottle-
necks inherent in traditional graph-enhanced RAG
paradigms.
Hyperparameter analysis.To evaluate frame-
work robustness, we conduct a hyperparameter
analysis on the HotpotQA dataset using the GPT-
4o-mini model. As illustrated in Figure 3, we inves-
tigate four key variables: initial anchors, retrieved
documents, reasoning iterations, and the energy
pruning threshold ( ϵ). ThreadRAG demonstrates
remarkable stability, consistently maintaining Ex-
act Match (EM) scores above 0.512 across all ex-
treme settings. Performance optimally peaks at an
EM of 0.574 with 3 anchors, 5 documents, 3 iter-
ations, and a 0.01 threshold. Notably, exceeding
these values yields marginal diminishing returns
rather than sharp performance drops. This con-
7

/uni00000014/uni00000015/uni00000016/uni00000017/uni00000018/uni00000019/uni0000001a
/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni00000024/uni00000051/uni00000046/uni0000004b/uni00000052/uni00000055/uni00000056/uni00000013/uni00000011/uni00000018/uni00000013/uni00000013/uni00000013/uni00000011/uni00000018/uni00000015/uni00000018/uni00000013/uni00000011/uni00000018/uni00000018/uni00000013/uni00000013/uni00000011/uni00000018/uni0000001a/uni00000018/uni00000028/uni0000005b/uni00000044/uni00000046/uni00000057/uni00000003/uni00000030/uni00000044/uni00000057/uni00000046/uni0000004b/uni00000003/uni0000000b/uni00000028/uni00000030/uni0000000c
/uni00000013/uni00000011/uni00000018/uni0000001a/uni00000017/uni0000000b/uni00000044/uni0000000c/uni00000003/uni00000037/uni00000052/uni00000053/uni00000010/uni0000004e/uni00000003/uni00000024/uni00000051/uni00000046/uni0000004b/uni00000052/uni00000055/uni00000056
/uni00000014/uni00000015/uni00000016/uni00000017/uni00000018/uni00000019/uni0000001a/uni0000001b/uni0000001c
/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni00000027/uni00000052/uni00000046/uni00000058/uni00000050/uni00000048/uni00000051/uni00000057/uni00000056/uni00000013/uni00000011/uni00000018/uni00000013/uni00000013/uni00000013/uni00000011/uni00000018/uni00000015/uni00000018/uni00000013/uni00000011/uni00000018/uni00000018/uni00000013/uni00000013/uni00000011/uni00000018/uni0000001a/uni00000018
/uni00000013/uni00000011/uni00000018/uni0000001a/uni00000017/uni0000000b/uni00000045/uni0000000c/uni00000003/uni00000037/uni00000052/uni00000053/uni00000010/uni0000004e/uni00000003/uni00000035/uni00000048/uni00000057/uni00000055/uni0000004c/uni00000048/uni00000059/uni00000048/uni00000047/uni00000003/uni00000027/uni00000052/uni00000046/uni00000056
/uni00000014/uni00000015/uni00000016/uni00000017/uni00000018/uni00000019/uni0000001a
/uni0000002c/uni00000057/uni00000048/uni00000055/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000003/uni00000026/uni00000052/uni00000058/uni00000051/uni00000057/uni00000013/uni00000011/uni00000018/uni00000013/uni00000013/uni00000013/uni00000011/uni00000018/uni00000015/uni00000018/uni00000013/uni00000011/uni00000018/uni00000018/uni00000013/uni00000013/uni00000011/uni00000018/uni0000001a/uni00000018
/uni00000013/uni00000011/uni00000018/uni0000001a/uni00000017/uni0000000b/uni00000046/uni0000000c/uni00000003/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni0000002c/uni00000057/uni00000048/uni00000055/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000051/uni00000056
/uni00000013/uni00000011/uni00000013/uni00000013/uni00000014/uni00000013/uni00000011/uni00000013/uni00000013/uni00000016/uni00000013/uni00000011/uni00000013/uni00000013/uni00000019/uni00000013/uni00000011/uni00000013/uni00000014/uni00000013/uni00000011/uni00000013/uni00000015/uni00000013/uni00000011/uni00000013/uni00000018/uni00000013/uni00000011/uni00000014
/uni00000037/uni0000004b/uni00000055/uni00000048/uni00000056/uni0000004b/uni00000052/uni0000004f/uni00000047/uni00000003/uni00000039/uni00000044/uni0000004f/uni00000058/uni00000048/uni00000013/uni00000011/uni00000018/uni00000013/uni00000013/uni00000013/uni00000011/uni00000018/uni00000015/uni00000018/uni00000013/uni00000011/uni00000018/uni00000018/uni00000013/uni00000013/uni00000011/uni00000018/uni0000001a/uni00000018
/uni00000013/uni00000011/uni00000018/uni0000001a/uni00000017/uni0000000b/uni00000047/uni0000000c/uni00000003/uni00000028/uni00000051/uni00000048/uni00000055/uni0000004a/uni0000005c/uni00000003/uni00000037/uni0000004b/uni00000055/uni00000048/uni00000056/uni0000004b/uni00000052/uni0000004f/uni00000047/uni00000003/uni0000000b/uni0000000c
Figure 3: Hyperparameter sensitivity analysis on the HotpotQA dataset.
/uni0000002b/uni00000052/uni00000057/uni00000053/uni00000052/uni00000057/uni00000034/uni00000024 /uni00000015/uni0000003a/uni0000004c/uni0000004e/uni0000004c/uni00000030/uni00000058/uni0000004f/uni00000057/uni0000004c/uni0000004b/uni00000052/uni00000053 /uni00000030/uni00000058/uni00000036/uni0000004c/uni00000034/uni00000058/uni00000048/uni00000013/uni00000011/uni00000013/uni00000013/uni00000011/uni00000014/uni00000013/uni00000011/uni00000015/uni00000013/uni00000011/uni00000016/uni00000013/uni00000011/uni00000017/uni00000013/uni00000011/uni00000018/uni00000028/uni0000005b/uni00000044/uni00000046/uni00000057/uni00000003/uni00000030/uni00000044/uni00000057/uni00000046/uni0000004b
/uni00000013/uni00000011/uni00000015/uni0000001b/uni0000001b/uni00000013/uni00000011/uni00000016/uni00000013/uni00000017
/uni00000013/uni00000011/uni00000014/uni0000001b/uni00000019/uni00000013/uni00000011/uni00000017/uni00000014/uni00000017/uni00000013/uni00000011/uni00000016/uni0000001c/uni00000019
/uni00000013/uni00000011/uni00000015/uni00000019/uni00000017/uni0000002a/uni00000055/uni00000044/uni00000053/uni0000004b/uni00000035/uni00000024/uni0000002a/uni0000000d /uni00000037/uni0000004b/uni00000055/uni00000048/uni00000044/uni00000047/uni00000035/uni00000024/uni0000002a/uni0000000d
Figure 4: Performance comparison of EM scores using
different underlying graph representations.
firms ThreadRAG effectively balances computa-
tional efficiency with reasoning accuracy, avoiding
unnecessary over-exploration.
Graph Representation Comparison.This ex-
periment compares the conventional GraphRAG
paradigm, which relies on the strict extraction
of rigid entity-relation triplets, against our STG
paradigm, which inherently preserves native se-
mantic context and narrative flow. To facilitate
a rigorous evaluation, we implement two spe-
cific variants:GraphRAG∗, which utilizes a pre-
constructed global KG, andThreadRAG∗, which
adopts our proposed Semantic Topology Graph.
All other components, including retrieval indices
and prompting templates, are kept strictly identical.
As illustrated in Figure 4, ThreadRAG∗consis-
tently achieves superior performance across all
benchmarks, securing substantial EM improve-
ments ranging from 7.8% to 12.6%. Notably, on
the HotpotQA dataset, our approach yields a sig-
nificant 12.6% improvement in EM (from 0.288
to 0.414). We attribute these substantial gains to
STG’s ability to preserve rich semantic narrative
flows and ensure precise modality alignment via
topological paths. In contrast, the discrete, sym-
bolic triplets employed by GraphRAG∗tend to frag-
ment the original context, leading to inevitable se-
mantic loss during complex multi-hop reasoning.
/uni00000013/uni00000011/uni00000016 /uni00000013/uni00000011/uni00000017 /uni00000013/uni00000011/uni00000018 /uni00000013/uni00000011/uni00000019
/uni00000028/uni0000005b/uni00000044/uni00000046/uni00000057/uni00000003/uni00000030/uni00000044/uni00000057/uni00000046/uni0000004b/uni0000002b/uni00000052/uni00000057/uni00000053/uni00000052/uni00000057/uni00000034/uni00000024 /uni00000015/uni0000003a/uni0000004c/uni0000004e/uni0000004c /uni00000030/uni00000058/uni00000036/uni0000004c/uni00000034/uni00000058/uni00000048/uni00000013/uni00000011/uni00000018/uni00000018/uni00000019
/uni00000013/uni00000011/uni00000018/uni00000018/uni00000017
/uni00000013/uni00000011/uni00000016/uni00000014/uni00000013/uni00000013/uni00000011/uni00000018/uni00000018/uni0000001b
/uni00000013/uni00000011/uni00000018/uni00000019/uni00000019
/uni00000013/uni00000011/uni00000016/uni00000014/uni00000019/uni00000013/uni00000011/uni00000018/uni00000019/uni00000013
/uni00000013/uni00000011/uni00000018/uni00000019/uni00000017
/uni00000013/uni00000011/uni00000016/uni00000013/uni00000017/uni00000013/uni00000011/uni00000018/uni0000001a/uni00000017
/uni00000013/uni00000011/uni00000018/uni0000001b/uni0000001b
/uni00000013/uni00000011/uni00000016/uni00000015/uni0000001b/uni0000002e/uni0000002a/uni00000010/uni00000037/uni00000055/uni0000004c/uni00000053/uni0000004f/uni00000048/uni00000057/uni00000056
/uni0000002e/uni0000002a/uni00000010/uni00000037/uni00000048/uni0000005b/uni00000057/uni00000056/uni0000002e/uni0000002a/uni00000010/uni00000033/uni00000044/uni00000057/uni0000004b/uni00000056
/uni00000036/uni00000037/uni0000002a/uni00000010/uni00000033/uni00000044/uni00000057/uni0000004b/uni00000056/uni00000003/uni0000000b/uni00000032/uni00000058/uni00000055/uni00000056/uni0000000cFigure 5: Performance comparison of various knowl-
edge representation strategies.
knowledge representation strategies.Since
a critical concern lies in how to optimally for-
mat retrieved knowledge for LLM inference, we
investigate four knowledge representation strate-
gies: KG-Triplets, KG-Paths, KG-Texts, and our
STG-Paths, respectively corresponding to discrete
relational triplets, linear structural paths, LLM-
rewritten natural language, and topological text
paths. As illustrated in Figure 5, STG-Paths con-
sistently achieves the highest performance. We
observe that forcing an LLM to rewrite triplets
into natural language (KG-Texts) occasionally un-
derperforms even naive KG-Triplets, which we at-
tribute to hallucinations and semantic distortions
introduced during rewriting. Conversely, while
KG-Triplets and KG-Paths avoid rewriting risks,
they suffer from severe semantic fragmentation and
lack deep structural context. In stark contrast, STG-
Paths resolves this dilemma. By directly extracting
topological text paths, it preserves undistorted raw
information while injecting structural priors. This
dual advantage of strict semantic fidelity and logi-
cal coherence renders STG-Paths the most effective
representation for our framework.
8

5 Conclusion
In this paper, we propose ThreadRAG to tackle
complex multi-hop reasoning challenges. Thread-
RAG integrates a Semantic Topology Graph (STG)
for knowledge structuring and a Modality-Aligned
Reasoning Mechanism via Energy-Driven pathfind-
ing. As a core contribution, ThreadRAG linearizes
discrete graph structures into continuous natural
language sequences, effectively bridging the modal-
ity gap and preserving crucial narrative context.
Extensive experimental results demonstrate that
ThreadRAG achieves superior performance across
multiple benchmarks, surpassing existing state-of-
the-art baselines with low computational overhead.
Limitations
Our work has limitations in certain scenarios. First,
ThreadRAG relies on capturing structural priors
through narrative, semantic, and lexical edges, so
its performance improvement may not be signifi-
cant in tasks with extremely sparse document links
or where cross-document evidence chains are virtu-
ally non-existent, such as single-document standard
QA tasks. Additionally, due to the dynamic nature
of text segmentation and the complexity of multi-
hop paths, even after energy diffusion and graph
pruning, there may still be a few long-range contex-
tual dependencies that the pathfinding mechanism
fails to perfectly linearize in the exact chronologi-
cal or logical order. This could result in minor nar-
rative fragmentation during prompt reconstruction,
affecting the reasoning accuracy of downstream
LLMs. In future work, we aim to address these two
issues. We plan to explore whether the model can
effectively optimize the topological path selection
and conduct a more fine-grained alignment of nar-
rative flows to further enhance the effectiveness of
LLM prompts.
Ethics Statement
This work uses only publicly available benchmark
datasets, including HotpotQA, 2WikiMultihopQA,
and MuSiQue, and does not involve the collection
of personally identifiable or sensitive information,
human-subject experiments, or user studies. All
datasets and pretrained models are used solely for
academic research and evaluation purposes in ac-
cordance with their intended licenses and usage
conditions.References
Ming Ding, Chang Zhou, Qibin Chen, Hongxia Yang,
and Jie Tang. 2019. Cognitive graph for multi-hop
reading comprehension at scale. InProceedings of
the 57th annual meeting of the association for com-
putational linguistics, pages 2694–2703.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and
Jonathan Larson. 2024. From local to global: A
graph rag approach to query-focused summarization.
arXiv preprint arXiv:2404.16130.
Yuwei Fang, Siqi Sun, Zhe Gan, Rohit Pillai, Shuohang
Wang, and Jingjing Liu. 2020. Hierarchical graph
network for multi-hop question answering. InPro-
ceedings of the 2020 conference on empirical meth-
ods in natural language processing (EMNLP), pages
8823–8838.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,
and Haofen Wang. 2023. Retrieval-augmented gen-
eration for large language models: A survey.arXiv
preprint arXiv:2312.10997, 2(1):32.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schel-
ten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh
Goyal, Anthony Hartshorn, Aobo Yang, Archi Mi-
tra, Archie Sravankumar, Artem Korenev, Arthur
Hinsvark, and 542 others. 2024. The llama 3 herd of
models.arXiv preprint arXiv:2407.21783.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tian Ao, and
Chao Huang. 2024. Lightrag: Simple and fast
retrieval-augmented generation.arXiv preprint
arXiv:2410.05779, 2(3).
Bernal J Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Ya-
sunaga, and Yu Su. 2024. Hipporag: Neurobiologi-
cally inspired long-term memory for large language
models.Advances in neural information processing
systems, 37:59532–59569.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-hop
qa dataset for comprehensive evaluation of reasoning
steps. InProceedings of the 28th International Con-
ference on Computational Linguistics, pages 6609–
6625.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lu-
cas Hosseini, Fabio Petroni, Timo Schick, Jane
Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and
Edouard Grave. 2022. Few-shot learning with re-
trieval augmented language models.arXiv preprint
arXiv:2208.03299, 1(2):4.
Sungho Ko, Hyunjin Cho, Hyungjoo Chae, Jinyoung
Yeo, and Dongha Lee. 2024. Evidence-focused fact
summarization for knowledge-augmented zero-shot
9

question answering. InProceedings of the 2024 Con-
ference on Empirical Methods in Natural Language
Processing, pages 10636–10651.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks.Advances in neural information
processing systems, 33:9459–9474.
Rui Li, Quanyu Dai, Zeyu Zhang, Xu Chen, Zhenhua
Dong, and Ji-Rong Wen. 2025. Knowtrace: Boot-
strapping iterative retrieval-augmented generation
with structured knowledge tracing. InProceedings
of the 31st ACM SIGKDD Conference on Knowledge
Discovery and Data Mining V . 2, pages 1470–1480.
Ruosen Li and Xinya Du. 2023. Leveraging structured
information for explainable multi-hop question an-
swering and reasoning. InFindings of the Associa-
tion for Computational Linguistics: EMNLP 2023,
pages 6779–6789.
Yanming Liu, Xinyue Peng, Tianyu Du, Jianwei Yin,
Weihao Liu, and Xuhong Zhang. 2024. Era-cot: Im-
proving chain-of-thought through entity relationship
analysis. InProceedings of the 62nd Annual Meet-
ing of the Association for Computational Linguistics
(Volume 1: Long Papers), pages 8780–8794.
Linhao Luo, Yuan-Fang Li, Reza Haffari, and Shirui
Pan. 2024. Reasoning on graphs: Faithful and inter-
pretable large language model reasoning. InInter-
national Conference on Learning Representations,
volume 2024, pages 14400–14423.
Zach Nussbaum, John X Morris, Brandon Duderstadt,
and Andriy Mulyar. 2024. Nomic embed: Training
a reproducible long context text embedder.arXiv
preprint arXiv:2402.01613.
OpenAI, :, Aaron Hurst, Adam Lerer, Adam P. Goucher,
Adam Perelman, Aditya Ramesh, Aidan Clark,
AJ Ostrow, Akila Welihinda, Alan Hayes, Alec
Radford, Aleksander M ˛ adry, Alex Baker-Whitcomb,
Alex Beutel, Alex Borzunov, Alex Carney, Alex
Chow, Alex Kirillov, and 401 others. 2024. Gpt-4o
system card.arXiv preprint arXiv:2410.21276.
Viet Thanh Pham, Minghan Wang, Hao-Han Liao, and
Thuy Vu. 2026. Discourse graph guided document
translation with large language models. InProceed-
ings of the 19th Conference of the European Chapter
of the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 1607–1627.
Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt,
Noah A Smith, and Mike Lewis. 2023. Measuring
and narrowing the compositionality gap in language
models. InFindings of the Association for Computa-
tional Linguistics: EMNLP 2023, pages 5687–5711.
Lin Qiu, Yunxuan Xiao, Yanru Qu, Hao Zhou, Lei Li,
Weinan Zhang, and Yong Yu. 2019. Dynamicallyfused graph network for multi-hop reasoning. InPro-
ceedings of the 57th annual meeting of the associa-
tion for computational linguistics, pages 6140–6150.
Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh
Khanna, Anna Goldie, and Christopher Manning.
2024. Raptor: Recursive abstractive processing for
tree-organized retrieval. InInternational Conference
on Learning Representations, volume 2024, pages
32628–32649.
Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie
Huang, Nan Duan, and Weizhu Chen. 2023. En-
hancing retrieval-augmented large language models
with iterative retrieval-generation synergy. InFind-
ings of the Association for Computational Linguistics:
EMNLP 2023, pages 9248–9274.
Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo
Wang, Chen Lin, Yeyun Gong, Lionel Ni, Heung-
Yeung Shum, and Jian Guo. 2024. Think-on-graph:
Deep and responsible reasoning of large language
model on knowledge graph. InInternational Con-
ference on Learning Representations, volume 2024,
pages 3868–3898.
Xingyu Tan, Xiaoyang Wang, Qing Liu, Xiwei Xu, Xin
Yuan, and Wenjie Zhang. 2025. Paths-over-graph:
Knowledge graph empowered large language model
reasoning. InProceedings of the ACM on Web Con-
ference 2025, pages 3505–3522.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. Musique: Multi-
hop questions via single-hop question composition.
Transactions of the Association for Computational
Linguistics, 10:539–554.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving retrieval
with chain-of-thought reasoning for knowledge-
intensive multi-step questions. InProceedings of
the 61st annual meeting of the association for com-
putational linguistics (volume 1: long papers), pages
10014–10037.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V . Le,
and Denny Zhou. 2022. Chain-of-thought prompting
elicits reasoning in large language models.Advances
in neural information processing systems, 35:24824–
24837.
Yike Wu, Yi Huang, Nan Hu, Yuncheng Hua, Guilin Qi,
Jiaoyan Chen, and Jeff Z Pan. 2024. Cotkr: Chain-of-
thought enhanced knowledge rewriting for complex
knowledge graph question answering. InProceed-
ings of the 2024 Conference on Empirical Methods
in Natural Language Processing, pages 3501–3520.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao,
Chengen Huang, Chenxu Lv, Chujie Zheng, Dayi-
heng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge,
Haoran Wei, Huan Lin, Jialong Tang, and 41 oth-
ers. 2025. Qwen3 technical report.arXiv preprint
arXiv:2505.09388.
10

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D Manning. 2018. Hotpotqa: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 conference on empiri-
cal methods in natural language processing, pages
2369–2380.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik Narasimhan, and Yuan Cao. 2022.
React: Synergizing reasoning and acting in language
models.arXiv preprint arXiv:2210.03629.
Xiao Yue, Guangzhi Qu, and Lige Gan. 2026.
Litesemrag: Lightweight llm-free semantic-aware
graph retrieval for robust rag.arXiv preprint
arXiv:2604.16350.
A Dataset Details
To comprehensively evaluate the multi-hop rea-
soning and knowledge structuring capabilities of
ThreadRAG, we conduct our experiments across
three standard benchmarks. Detailed descriptions
of each dataset are provided below:
HotpotQA(Yang et al., 2018) is a widely-
used, crowd-sourced dataset derived from English
Wikipedia, comprising approximately 113k ques-
tions. It specifically requires models to bridge infor-
mation across the introductory paragraphs of mul-
tiple distinct articles to deduce the correct answer.
While the original dataset provides golden para-
graphs and human-annotated supporting sentences,
we evaluate our framework strictly under a realistic
open-domain setting. In this setting, golden con-
texts are not provideda priori; instead, the system
must retrieve and structure relevant information
from a massive corpus, which rigorously tests the
robustness of our Semantic Topology Graph (STG)
against retrieval noise and distractors.
2WikiMultihopQA (2Wiki)(Ho et al., 2020)
focuses on evaluating complex 2-hop reasoning,
placing a strong emphasis on compositional deduc-
tion and comparative analysis. The questions in this
dataset are systematically constructed by combin-
ing both structured knowledge (from Wikidata) and
unstructured texts (from Wikipedia). This struc-
tural heterogeneity makes it an ideal testbed for
evaluating whether our framework can seamlessly
align and linearize varied knowledge formats into
a cohesive, modality-native narrative flow without
semantic loss.
MuSiQue(Trivedi et al., 2022) is a highly de-
manding dataset carefully constructed by compos-
ing bottom-up single-hop queries into intricatemulti-hop reasoning chains. Compared to Hot-
potQA and 2Wiki, MuSiQue poses a significantly
greater challenge as it features deep 2- to 4-hop
questions and is meticulously designed to prevent
reasoning shortcuts. This dataset effectively stress-
tests the capability of our Energy-Driven Pathfind-
ing (EDP) mechanism to maintain long-range logi-
cal coherence and prevent structural truncation in
extremely deep reasoning scenarios.
B Baseline Descriptions
To comprehensively evaluate the performance of
ThreadRAG and clearly demonstrate the char-
acteristics of different knowledge representation
paradigms, we benchmark our framework against
five distinct categories of state-of-the-art baselines.
Non-Graph RAG.This category encompasses
text-centric paradigms that tackle multi-hop reason-
ing over plain text without relying on explicit graph
structures. Instead of using structural priors, these
approaches lean heavily on the LLM’s intrinsic ca-
pacity to process continuous contexts and deduce
logical chains. To evaluate this paradigm, we select
several prominent baselines:RA-CoT(Wei et al.,
2022) andReAct(Yao et al., 2022), which enhance
standard text reasoning by integrating retrieval with
chain-of-thought and acting, respectively;IRCoT
(Trivedi et al., 2023) andIter-RetGen(Shao et al.,
2023), which build continuous reasoning chains
by iteratively interleaving retrieval steps with gen-
eration; andSelf-Ask(Press et al., 2023), which
structures the context by decomposing complex
queries into manageable textual sub-queries.
KG-Chunk RAG.Methods in this paradigm
aim to mitigate the semantic fragmentation caused
by fine-grained triplet extraction by treating larger
text chunks as the fundamental nodes in the knowl-
edge graph. This approach provides structural con-
nectivity while preserving local contextual nuances.
We evaluate againstHippoRAG(Gutiérrez et al.,
2024), a neurobiologically inspired framework sim-
ulating human long-term memory, andLightRAG
(Guo et al., 2024), which focuses on simple, fast,
and lightweight graph-based retrieval.
KG-Rewriting RAG.Recognizing the modal-
ity mismatch between discrete graph structures
and the continuous text sequences LLMs are pre-
trained on, this category attempts to bridge the
gap through generative rewriting. These meth-
ods retrieve relevant subgraphs and employ LLMs
to translate them into natural language narratives
11

prior to final reasoning. Representative baselines
includeCoTKR(Wu et al., 2024), which enhances
knowledge rewriting via chain-of-thought, andEF-
SUM(Ko et al., 2024), which focuses on generat-
ing evidence-focused summaries from graph struc-
tures.
KG-Path RAG.This paradigm provides LLMs
with explicit, step-by-step structural guidance by
forcing them to reason along discrete relational
paths extracted from the knowledge graph. The
selected baselines are Think-on-Graph (ToG) (Sun
et al., 2024) and Paths-over-Graph (PoG) (Tan
et al., 2025). For fair comparison, we adapt ToG
and PoG to use the same local KG as LightRAG,
replacing their external KGs.
KG-Triplet RAG.Representing the most fine-
grained end of the structural extraction spectrum,
these methods distill documents into rigid entity-
relation triplets and use them as discrete prompts.
While offering high-density factual associations,
this paradigm often strips away the original narra-
tive context entirely. We compare againstERA-
CoT(Liu et al., 2024), which improves reasoning
through entity relationship analysis, andKnow-
Trace(Li et al., 2025), a robust iterative framework
guided by structured knowledge tracing.
C Example
To provide a concrete illustration of the end-to-end
reasoning process in ThreadRAG, we present a
representative example from MuSiQue in Table 4.
The example traces the complete pipeline, includ-
ing query analysis and retrieval, semantic node
construction, energy-driven evidence selection and
topological linearization, and final answer synthe-
sis. It demonstrates how ThreadRAG progressively
transforms initially retrieved passages into a com-
pact, context-preserving evidence sequence that
supports the final multi-hop reasoning process.
DImplementation Details of the Semantic
Segmentation Model
In response to the necessity for high-fidelity,
context-intact partitioning during the Semantic
Topology Mapper (STM) phase, we explicitly de-
tail the architecture, data curation, and training
pipeline of the MLP-based boundary predictor (in-
troduced in Equation 2).
Encoder Architecture and Inference Pipeline.
The “lightweight encoder” utilized in our frame-
work is implemented using the pre-trainedRoBERTa-base model. To prepare the inputs,
raw continuous documents are first segmented into
natural sentences using the NLTK library. To maxi-
mize contextual awareness while strictly adhering
to the 512-token limit of the RoBERTa architec-
ture, we dynamically pack consecutive sentences
into continuous blocks of under 512 tokens. The
model processes these text blocks to extract dense
representations, simultaneously evaluating the se-
mantic continuity at the boundaries between all
adjacent sentences within the chunk. This batch-
processing approach provides a robust foundation
for capturing localized semantic shifts without in-
curring the massive latency associated with using
LLMs for real-time inference. During generation,
a semantic breakpoint is formally triggered if the
predicted boundary probability piexceeds a confi-
dence threshold ofθ= 0.65.
Data and Supervision Signal (Silver Labels).
To circumvent the prohibitive costs of human
annotation while maintaining high segmentation
quality, we employ a Knowledge Distillation
(KD) paradigm using pseudo-labels (Silver Labels).
Specifically, we sample 100 complex multi-hop
documents from the Wikipedia corpus. We then
prompt powerful teacher models (e.g., GPT-4o-
mini and Qwen3.5-Plus) to perform rigorous "Se-
mantic Node Partitioning" on this subset, instruct-
ing them to split the texts along natural narrative
breaks. This zero-shot distillation yields a high-
quality dataset comprising 1,553 distinct semantic
nodes. The binary boundaries generated by these
teacher models serve as the positive supervision
signals ( yi= 1for a narrative boundary, yi= 0for
continuous context) to train our lightweight bound-
ary predictor.
Training Setup and Loss Function.The model
is optimized using the AdamW optimizer. To en-
sure stable convergence while fine-tuning, we apply
a learning rate of 1×10−5for the MLP prediction
head. The MLP is optimized using the standard
Binary Cross-Entropy (BCE) loss:
Lseg=−1
NNX
i=1h
yilog(p i)+(1−y i) log(1−p i)i
(12)
where Nis the total number of evaluated sentence
boundaries within the input chunks, yi∈ {0,1} is
the distillation pseudo-label, and piis the predicted
boundary probability derived from the MLP head.
Generalizability (Shared Across Datasets).It
is crucial to note that the segmentation model is
12

Question:Who sings home alone tonight with the singer of you can crash my party anytime?
[Phase 1: Query Analysis & Retrieval]
# Entity Extraction:
- Entities to retrieve: "Home Alone Tonight", "Crash My Party"
# Retrieve Relevant Textual Passages:
-Take Me Home Tonight (song):"Take Me Home Tonight" is a song by American rock singer Eddie Money...
-Luke Bryan discography:American country music singer and songwriter Luke Bryan has released...
-Crash My Party (song):"Crash My Party" is a song recorded by American country artist Luke Bryan...
-Home Alone Tonight:"Home Alone Tonight" is a song recorded by American country music artist...
[Phase 2: Semantic Topology Mapper (STM)]
# Semantic Node Extraction:
-SN 1:"Take Me Home Tonight" is a song by American rock singer Eddie Money.
-SN 2:It was released in August 1986 as the lead single from his album "Can’t Hold Back".
# Semantic Topology Graph (STG) Construction:
- Graph initialized with 270 context-intact nodes and 878 structural (narrative/semantic/lexical) edges.
[Phase 3: Energy-Driven Pathfinding (MAP)]
# Anchor Identification & Energy Diffusion:
- 5 query-relevant anchor nodes identified based on the extracted entities.
# Graph Pruning & Topological Linearization:
- 9 highly activated nodes retained out of 270 after energy threshold pruning.
-Linearized Context (Modality-Native Prompt):
(1) "Crash My Party" is a song written by Rodney Clawson and Ashley Gorley, and recorded by American country artist
Luke Bryan.
(2) Crash My Party is the fourth studio album by American country music artist Luke Bryan.
(3) "Home Alone Tonight" is a song recorded by American country music artist Luke Bryan as a duet with Karen Fairchild
of American country music group Little Big Town for his fifth studio album, "Kill the Lights" (2015).
(4) ...
[Phase 4: Agent Evaluation & Answer Synthesis]
# Reasoning Thought Process:
- Step 1: Identify "the singer of you can crash my party anytime". From the linearized context, "Crash My Party" is a song
recorded by Luke Bryan.
- Step 2: Identify who sings "Home Alone Tonight" with Luke Bryan. From the context, "Home Alone Tonight" is a duet
with Karen Fairchild.
- Step 3: Therefore, the person who sings "Home Alone Tonight" with Luke Bryan is Karen Fairchild.
Answer:Karen Fairchild
Table 4: Case on MuSiQue.
notdynamically re-trained or overfitted to individ-
ual evaluation sets. The RoBERTa-base encoder
and the MLP head are trained exactly once using
the distilled pseudo-labels. The resulting weights
are then frozen and uniformly shared across all
downstream evaluations (HotpotQA, 2WikiMul-
tihopQA, and MuSiQue). This zero-shot infer-
ence approach ensures strict evaluation fairness
and proves that our STM possesses strong gener-
alization capabilities across unseen open-domain
texts.
Prompt for Dataset Generation.To ensure
reproducibility, the exact instruction template pro-
vided to the teacher models for generating the silver
labels is presented in Table 5.E Additional Ablation and Intrinsic
Evaluation
To further address the concerns regarding graph
topology, evidence selection, and semantic segmen-
tation, we conduct additional controlled ablations
and intrinsic evaluations. These experiments com-
plement the main-text analyses by more explicitly
isolating the contributions of graph-based evidence
organization, Energy-Driven Pathfinding (EDP),
and the semantic boundary predictor.
E.1 Graph-Free Ablation
To more cleanly isolate the contribution of graph-
based evidence organization from the remain-
13

Prompt 3: Dataset Generation
You are an expert linguistic annotator. Your task is to per-
form “Semantic Node Partitioning” on a given multi-hop
document. Your goal is to divide the continuous text into
self-contained “Semantic Nodes” along natural narrative,
logical, or topical breaks.
## Rules for Partitioning:
1.Contextual Integrity:Each semantic node should be a
coherent unit of information (e.g., a complete description
of an entity, a specific event, or a logical step). Do not split
sentences that are tightly coupled (e.g., pronouns referring
to the immediately preceding sentence).
2.Granularity:A node typically consists of 1 to 2 sen-
tences. Avoid creating overly long nodes that merge distinct
topics, and avoid overly granular nodes (e.g., splitting a
single compound sentence).
3.Exact Match:The text of the partitioned nodes MUST
perfectly reconstruct the original document when concate-
nated. Do not alter, add, or remove any words.
## Output Format:
You MUST output a valid JSON array of strings, where
each string is a continuous segment (Semantic Node) of the
original text.
## Example:
Original Text:“Mickey’s Safari in Letterland is a 1993 edu-
cational NES video game. It stars Disney’s Mickey Mouse.
The successor title, Walt Disney’s Comics and Stories, be-
gan publication with the Oct 1940 issue. It featured the
Gottfredson serials as a prominent feature.”
Output:
[
"Mickey’s Safari in Letterland is a
1993 educational NES video
game. It stars Disney’s Mickey
Mouse.",
"The successor title, Walt Disney’s
Comics and Stories, began
publication with the Oct 1940 issue.
It featured the
Gottfredson serials as a prominent
feature."
]
## Input Document:
{document_text}
Output ONLY the JSON array:
Table 5: Prompt Templates of Dataset Generation.ing components of ThreadRAG, we construct a
Graph-Freevariant. This variant retains the same
retrieval pipeline, iterative reasoning procedure,
downstream LLM, prompts, and evidence budget
as ThreadRAG, while removing Semantic Topol-
ogy Graph (STG) construction and EDP. Instead,
retrieved passages are selected according to query–
passage similarity and concatenated following their
original document order.
Method HotpotQA 2Wiki MuSiQue
Graph-Free 0.274 0.258 0.206
ThreadRAG0.422 0.372 0.324
Table 6: Controlled Graph-Free ablation in terms of
Exact Match (EM). The Graph-Free variant preserves
retrieval and iterative reasoning while removing graph
construction and EDP.
As shown in Table 6, removing graph-based evi-
dence organization leads to consistent absolute EM
decreases of 0.148, 0.114, and 0.118 on HotpotQA,
2Wiki, and MuSiQue, respectively. These results
indicate that the performance gain of ThreadRAG
cannot be attributed solely to iterative retrieval or
reasoning, and that explicitly organizing retrieved
evidence through the semantic topology contributes
substantially to multi-hop reasoning performance.
E.2 Comparison of Subgraph Selection
Strategies
We further investigate whether the advantage of
EDP arises specifically from its topology-aware
evidence propagation rather than merely from se-
lecting a connected subgraph. We compare EDP
against two standard graph traversal strategies,
breadth-first search (BFS) and depth-first search
(DFS), as well as a condition without subgraph-
based selection. All variants operate on the same
candidate graph and use an identical evidence bud-
get; only the evidence selection strategy is changed.
Subgraph Strategy EM∆vs. EDP
No subgraph 0.286−0.038
BFS (depth = 3) 0.294−0.030
DFS (depth = 3) 0.302−0.022
EDP 0.324–
Table 7: Comparison of evidence selection strategies on
MuSiQue using LLaMA3-8B-Instruct. All methods use
the same candidate graph and evidence budget.
Table 7 shows that both BFS and DFS underper-
form EDP, with absolute EM decreases of 0.030
14

Precision Recall F1
0.948 0.861 0.902
Table 8: Intrinsic evaluation of the semantic boundary
predictor on 50 held-out Wikipedia documents. Scores
measure agreement with teacher-generated reference
boundaries.
and 0.022, respectively. The no-subgraph condition
exhibits an even larger decrease of 0.038. Fixed-
depth traversal strategies are constrained by local
graph exploration: BFS tends to expand broadly
within a limited depth, whereas DFS follows indi-
vidual structural branches. In contrast, EDP propa-
gates energy jointly from multiple query-relevant
anchors over the graph, allowing evidence dis-
tributed across distant regions to contribute to the
final selection. This property is particularly useful
for MuSiQue, where answering a question often
requires aggregating evidence across 2–4 reasoning
hops.
E.3 Intrinsic Evaluation of Semantic
Segmentation
In addition to the downstream ablation reported
in the main paper, we directly evaluate the bound-
ary prediction quality of the semantic segmentation
model. We construct an independent held-out eval-
uation set containing 50 Wikipedia documents that
are disjoint from the 100 documents used to train
the boundary predictor. Teacher-generated seman-
tic boundaries are used as reference annotations,
resulting in 719 reference boundaries in total.
As shown in Table 8, the lightweight boundary
predictor achieves a precision of 0.948, recall of
0.861, and F1 score of 0.902. The high precision
indicates that the model rarely introduces unsup-
ported semantic breaks, while the recall suggests
that most reference boundaries are successfully
identified. Together with the downstream perfor-
mance reported in the main paper, these results
provide complementary evidence that the learned
segmentation model can preserve coherent seman-
tic units without relying on dataset-specific retrain-
ing.
F Prompt Templates
In this section, we detail the core prompt templates
utilized in ThreadRAG. To ensure reproducibility,
we present the exact instructions provided to the
underlying Large Language Models for both entityPrompt 1: Entity Extraction
You are an entity extraction assistant.
Extract key entities from the question
for knowledge retrieval.
Rules:
1. Extract only key entities (persons, places, organizations,
products, etc.)
2. Return ONLY a JSON array of entity strings
3. DO NOT include generic words like "who", "what",
"where", "the", "a", etc.
4. For multi-hop questions, include ALL entities needed
for reasoning
5. Keep entity names concise (2-4 words max)
Examples:
Question: "Who directed Inception?"
Entities: ["Inception"]
Question: "Aside from the Apple Remote, what other de-
vice can control Front Row?"
Entities: ["Apple Remote", "Front Row"]
Question: "What sport did the person who founded Google
play in college?"
Entities: ["Google", "Larry Page"]
Table 9: Prompt Templates of Entity Extraction.
extraction and the iterative information sufficiency
evaluation.
F.1 Entity Extraction Prompt
This prompt (Table 9) is designed to instruct the
LLM to extract key reasoning entities from the
user’s multi-hop query, stripping away generic stop
words to form precise search anchors.
F.2 Information Sufficiency Discriminator
Prompts
To prevent premature hallucination, ThreadRAG
employs an Information Sufficiency Discriminator.
As detailed in Table 10, this system prompt estab-
lishes strict evaluation criteria, guiding the LLM
to carefully assess whether the provided context
contains adequate factual information to fully re-
solve the user’s query before proceeding to the final
answer generation.
G Qualitative Error Analysis
To systematically investigate the boundary condi-
tions and inherent limitations of the ThreadRAG
framework, we conducted a rigorous qualitative er-
ror analysis on the HotpotQA dataset (in Figure 6).
15

Prompt 2: Discriminator
You are an Information Sufficiency
Discriminator.
Your task is to evaluate whether the provided context is
sufficient to answer the user’s question.
## Output Format (JSON):
{"status": "Sufficient" or "Insufficient", "reason": "brief
explanation if insufficient, empty string if sufficient"}
## Evaluation Criteria:
- If the context contains enough factual information to an-
swer the question, output {"status": "Sufficient", "reason":
""}
- If the context is missing key information, output {"sta-
tus": "Insufficient", "reason": "what specific information is
missing"}
- Be strict: only output "Sufficient" if you are confident the
question can be fully answered from the context.
Table 10: Prompt Templates of Discriminator.
Retrieval Failure149
(74.5%)LLM Analysis Error
42
(21.0%) Pathfinding
9
(4.5%)Failure Mode Distribution
Figure 6: Distribution of failure modes analyzed from
200 incorrect predictions.
Specifically, we randomly sampled 200 incorrect
prediction cases from the evaluation pool and man-
ually scrutinized their underlying reasoning traces.
These failures were comprehensively classified into
three distinct categories based on their root causes:
•Retrieval Failure (74.5%, 149 cases):This
dominant failure mode occurs when the upstream
initial retriever fails to surface any documents
containing the gold supporting evidence within
the top- kretrieval pool. Consequently, the sub-
sequently constructed Semantic Topology Graph
(STG) lacks the necessary factual foundation,
rendering correct deduction impossible.
•LLM Analysis Error (21.0%, 42 cases):In this
scenario, the ThreadRAG pipeline functions flaw-lessly, successfully linearizing a perfect, context-
intact prompt that encapsulates the complete
multi-hop reasoning chain. However, the down-
stream baseline Large Language Model suffers
from intrinsic logical deficiencies or hallucina-
tions during the final step, failing to deduce the
correct target answer despite having access to the
pristine context.
•Pathfinding Error (4.5%, 9 cases):This rep-
resents the internal boundary condition of our
system. During exceptionally long-range or com-
plex multi-hop path exploration, the energy prop-
agated via our Energy-Driven Pathfinding (EDP)
module decays beyond acceptable margins, or
falls slightly below the rigid structural pruning
threshold. As a result, critical evidence nodes are
inadvertently truncated or filtered out during the
topological linearization phase.
This empirical breakdown of failure modes
strongly validates the efficacy, robustness, and
architectural soundness of our proposed Energy-
Driven Pathfinding (EDP) mechanism. The statisti-
cal distribution clearly illustrates that a staggering
95.5% of the total failure cases are exclusively trig-
gered by orthogonal, external bottlenecks—namely,
the hard recall limits of the upstream retriever and
the intrinsic reasoning ceilings of the baseline LLM.
In sharp contrast, errors stemming from our inter-
nal graph pathfinding and structural pruning com-
ponents constitute an absolute minority (merely
4.5%). These quantitative insights demonstrate
that the EDP module successfully navigates com-
plex knowledge topologies to eliminate redundant
context noise while maintaining an exceptionally
high fidelity in preserving critical factual chains for
downstream generation.
16