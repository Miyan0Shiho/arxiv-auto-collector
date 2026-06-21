# SproutRAG: Attention-Guided Tree Search with Progressive Embeddings for Long-Document RAG

**Authors**: Amirhossein Abaskohi, Issam H. Laradji, Peter West, Giuseppe Carenini

**Published**: 2026-06-16 18:28:00

**PDF URL**: [https://arxiv.org/pdf/2606.18381v1](https://arxiv.org/pdf/2606.18381v1)

## Abstract
Retrieval-augmented generation (RAG) systems must balance retrieval granularity with contextual coherence, a challenge that existing methods address through LLM-guided chunking, single-level context expansion, or hierarchical summarization. These approaches variously depend on costly LLM calls during indexing or retrieval, limit context aggregation to a single granularity level, or introduce information loss through summarization. We present SproutRAG, an attention-guided hierarchical RAG framework that addresses this trade-off by organizing sentence-level chunks into progressively larger but semantically coherent units, using learned inter-sentence attention to construct a binary chunking tree. Unlike prior approaches that rely on external LLMs, fixed context expansion, or lossy summarization, SproutRAG learns which attention heads and layers best capture semantic document structure, enabling multi-granularity retrieval without additional LLM calls or compressed summaries. At retrieval time, SproutRAG uses hierarchical beam search to retrieve candidates at multiple granularities, capturing multi-sentence relevance beyond flat retrieval. The framework is trained end-to-end with a joint objective that improves both embeddings and tree structure. Experiments across four benchmarks spanning scientific, legal, and open-domain settings demonstrate that SproutRAG improves information efficiency (IE) by 6.1% on average over the strongest baseline. Code is available on https://github.com/AmirAbaskohi/SproutRAG.

## Full Text


<!-- PDF content starts -->

SPROUTRAG: Attention-Guided Tree Search with Progressive
Embeddings for Long-Document RAG
Amirhossein Abaskohi1*, Issam H. Laradji1,2,Peter West1, Giuseppe Carenini1
1University of British Columbia,2ServiceNow Research
Abstract
Retrieval-augmented generation (RAG) sys-
tems must balance retrieval granularity with
contextual coherence, a challenge that existing
methods address through LLM-guided chunk-
ing, single-level context expansion, or hierar-
chical summarization. These approaches var-
iously depend on costly LLM calls during in-
dexing or retrieval, limit context aggregation to
a single granularity level, or introduce informa-
tion loss through summarization. We present
SPROUTRAG, an attention-guided hierarchi-
cal RAG framework that addresses this trade-
off by organizing sentence-level chunks into
progressively larger but semantically coherent
units, using learned inter-sentence attention to
construct a binary chunking tree. Unlike prior
approaches that rely on external LLMs, fixed
context expansion, or lossy summarization,
SPROUTRAG learns which attention heads and
layers best capture semantic document struc-
ture, enabling multi-granularity retrieval with-
out additional LLM calls or compressed sum-
maries. At retrieval time, SPROUTRAG uses hi-
erarchical beam search to retrieve candidates at
multiple granularities, capturing multi-sentence
relevance beyond flat retrieval. The frame-
work is trained end-to-end with a joint objec-
tive that improves both embeddings and tree
structure. Experiments across four benchmarks
spanning scientific, legal, and open-domain set-
tings demonstrate that SPROUTRAG improves
information efficiency (IE) by 6.1% on average
over the strongest baseline1.
1 Introduction
Retrieval-augmented generation (RAG) has be-
come the dominant paradigm for grounding large
language models (LLMs) in external knowledge,
helping reduce hallucinations, support domain-
specific reasoning, and improve performance on
*Corresponding author:aabaskoh@cs.ubc.ca
1Code is available on/githubGitHub.knowledge-intensive tasks (Lewis et al., 2020; Au-
genstein et al., 2024). As LLMs are increasingly
applied to complex tasks involving long docu-
ments (Jin et al., 2025a), directly providing entire
documents as input becomes impractical due to
context-length constraints and degraded attention
over extended sequences (Jin et al., 2025b; Liu
et al., 2024). Consequently, RAG frameworks seg-
ment documents into chunks and retrieve the most
relevant pieces to construct focused, high-quality
evidence for generation.
The effectiveness of this retrieval step hinges
critically on how documents are segmented. Large
chunks preserve contextual coherence but intro-
duce redundant noise that dilutes key information,
while fine-grained chunks offer precision but suf-
fer from semantic fragmentation and broken inter-
chunk relationships (Tao et al., 2025; Zhao et al.,
2025a). This problem is particularly acute for cross-
paragraph retrieval, where answering a query re-
quires synthesizing information scattered across
multiple document sections, as in multi-hop rea-
soning and summarization tasks (Liu et al., 2025).
Recent work addresses this challenge from sev-
eral directions. SAKI-RAG (Tao et al., 2025) uses
a SLLM (An et al., 2024) to merge semantically re-
lated sentence pairs and relies on an external LLM
to filter retrieval candidates. However, extending
this pairwise expansion to multi-chunk relevance
greatly increases the candidate space and makes
LLM filtering expensive. LLM-guided chunk-
ing methods such as Meta-Chunking (Zhao et al.,
2025b) and MoC (Zhao et al., 2025a) improve
segmentation quality, but discard cross-chunk de-
pendencies after chunking. Hierarchical meth-
ods such as RAPTOR (Sarthi et al., 2024) sup-
port multi-granularity retrieval through clustering
and summarization, yet clustering treats chunks
within a group as interchangeable and summaries
can lose evidence. Graph-based methods such as
GraphRAG (Edge et al., 2025) model entity re-
1arXiv:2606.18381v1  [cs.CL]  16 Jun 2026

Build a hierarchy from sentence attention
1Long Document + Query
Document Snippet
William Shakespeare was born in Stratford-upon-
Avon in 1564. At age 18, he married Anne Hathaway,
with whom he had three children. His early life
centered on family and his local community,
experiences that later shaped themes of community
and human nature in his works. Between 1585 and
1592, he began a successful career in London as an
actor, writer, and part-owner of a playing company.What changes in
Shakespeare’s life
influenced his works?
2Sentence-level Chunks and SLLM attention to find related sentences
Shakespeare was born in Stratford-
upon-Avon.
At age 18, he married Anne Hathaway.
During his early life, he focused on
family and local community.
These experiences later influenced
themes of community and human
nature.
Between 1585 and 1592, he began a
successful career in London.
He became an actor, writer, and part-
owner of a playing company.1
2
3
4
5
6
Inter-sentence attention
3Attention-guided tree retrieval
Root
(all sentences 1-6)
Node A
(1-4)
5
career in
London (1585-
1592)Node A2
(3-4)Node B
(5-6)
QueryQuery
Node A1
(1-2)
1
born
in Stratford -
upon-Avon3
family and
local
community2
married Anne
Hathaway 6
actor, writer,
part-owner of
company 4
themes of
community &
human natureRetrieved NodesFigure 1: SPROUTRAG segments a long document into
sentence-level chunks, uses SLLM attention to identify
semantically related sentences, and organizes them into
an attention-guided binary tree. Retrieval then selects
evidence across fine-grained leaves, mid-level nodes,
and broader subtrees, preserving precision while recov-
ering coherence.
lations, but are less effective when fine-grained
chunks contain sparse entity information.
In this paper, we presentSPROUTRAG, an
attention-guided hierarchical RAG framework that
organizes sentence-level chunks into a learned doc-
ument structure, preserving cross-chunk dependen-
cies while avoiding LLM inference overhead dur-
ing retrieval. As illustrated in Figure 1, this struc-
ture enables retrieval at multiple semantic granu-
larities. SPROUTRAG encodes documents at sen-
tence granularity using an SLLM and constructs a
binary tree bottom-up, where merge order is deter-
mined by a learned weighted aggregation of inter-
sentence attention across transformer heads and lay-
ers. This aggregation replaces naive uniform aver-
aging, which we show introduces a proximity bias
that weakens the global tree index; instead, learn-
able scalar weights allow the model to discover
which head types best reflect semantic co-relevance
for document structure. Each internal node storesa progressive embedding that compositionally rep-
resents its subtree, enabling multi-granularity re-
trieval via hierarchical beam search that collects
candidates across all tree levels. SPROUTRAG
is trained end-to-end to jointly optimize retrieval
quality and tree structure, requiring no external
LLM calls at any stage. It captures emergent multi-
sentence relevance that pairwise or flat retrieval
methods cannot, while remaining efficient enough
for deployment. We evaluate SPROUTRAG on
four benchmarks spanning scientific, legal, and
open-domain settings. On average, SPROUTRAG
improves information efficiency (IE) by 6.1% over
the strongest baseline, especially in the cases where
evidence is often dispersed across paragraphs.
In summary, the contributions of this paper are
as follows.(1)We introduce SPROUTRAG, an
attention-guided hierarchical RAG framework that
constructs a binary tree over sentence-level chunks
using learned inter-sentence attention, enabling
multi-granularity retrieval without any LLM calls
at inference time.(2)We identify and address the
proximity bias introduced by uniform attention av-
eraging in sentence-level transformers, replacing
it with a learned weighted aggregation that allows
the model to discover which attention heads best
reflect semantic co-relevance for document struc-
ture.(3)We introduce a joint training objective that
jointly improves retrieval quality and tree structure,
eliminating the need for external LLM filtering or
lossy summarization at any pipeline stage.
2 Related Work
Chunking and adaptive retrieval.The effec-
tiveness of RAG depends strongly on how docu-
ments are segmented into retrievable units. Stan-
dard RAG pipelines often rely on rule-based split-
ters, such as fixed-length or delimiter-based chunk-
ing, which are efficient but insensitive to seman-
tic boundaries (Team, 2024). Recent methods
aim to improve this granularity choice. Late-
Chunking (Günther et al., 2025) contextualizes to-
ken representations before forming chunk embed-
dings, while Meta-Chunking (Zhao et al., 2025b)
and MoC (Zhao et al., 2025a) use LLM-based sig-
nals or routing mechanisms to produce more adap-
tive chunk boundaries. Dense X Retrieval (Chen
et al., 2024) moves toward finer granularity by de-
composing text into atomic propositions, improv-
ing precision but weakening broader contextual
continuity. Other methods adapt retrieval after
2

chunks have been formed. ReflectiveRAG (Verma
et al., 2026) introduces a self-reflective retrieval
loop that evaluates evidence sufficiency and refor-
mulates queries to improve factual grounding, but
it does not change the underlying flat organization
of retrieval units. Most related to SPROUTRAG,
SAKI-RAG (Tao et al., 2025) uses a SLLM to
estimate inter-sentence attention and expand re-
trieved chunks with related sentences. Unlike
SAKI-RAG’s pairwise expansion with LLM filter-
ing, SPROUTRAG builds a global sentence-level
hierarchy that supports multi-granularity retrieval
without inference-time LLM calls.
Structured and hierarchical retrieval.Beyond
flat chunk retrieval, structured RAG methods or-
ganize document content into higher-level repre-
sentations. RAPTOR (Sarthi et al., 2024) recur-
sively clusters chunks and summarizes each cluster
into a tree, enabling retrieval from multiple levels;
however, its structure is based on embedding-space
clustering and relies on LLM-generated summaries,
which can discard fine-grained details. Graph-
based approaches such as GraphRAG (Edge et al.,
2025) and LightRAG (Guo et al., 2025) represent
documents through entities and relations, support-
ing traversal-based retrieval but depending on suc-
cessful entity extraction and relation construction.
PropRAG (Wang and Han, 2025) replaces entity
triples with propositions and performs LLM-free
beam search over proposition paths, while Beam
Retrieval (Zhang et al., 2024) shows the benefit
of maintaining multiple retrieval hypotheses for
multi-hop passage retrieval. PageIndex (Zhang
et al., 2025a) similarly explores reasoning-based,
vectorless retrieval over document tree structures,
but relies on document-level structural organiza-
tion rather than learned sentence-level attention.
SPROUTRAG instead builds an attention-guided
binary tree over sentence-level chunks, with com-
positional internal nodes and joint retrieval over
all nodes. This preserves cross-sentence dependen-
cies without lossy summarization, entity-centric
structures, or external LLM calls.
3 SPROUTRAG
As illustrated in Figure 2, SPROUTRAG replaces
flat chunk retrieval with atrained attention-
guided hierarchyover sentence-level chunks. Dur-
ing offline indexing, a SLLM encodes the docu-
ment and provides both sentence embeddings and
inter-sentence attention signals. These signals areaggregated with learnable head–layer weights and
used to build a binary tree, where leaves represent
fine-grained chunks and internal nodes storepro-
gressive embeddingsof merged sentence groups.
During online retrieval, SPROUTRAG encodes the
query and performshierarchical beam search, col-
lecting candidates from leaves, internal nodes, and
subtrees. As described in Section 3.3, the frame-
work is trained with a joint objective that improves
bothretrieval qualityandtree structure, enabling
multi-granularity retrieval without external LLM
calls during retrieval.
3.1 Attention-Guided Indexing
Given a document D, we first split it into sentence-
level chunks S={s 1, . . . , s n}. We encode the full
sequence with a SLLM, obtaining contextualized
sentence embeddings {e(s i)}n
i=1and attention ma-
trices from all layers and heads. For layer land
headh, we denote the corresponding attention ma-
trix asAttn(l,h)∈Rn×n.
A uniform average over all heads and layers
can overemphasize local sentence proximity, since
some attention heads primarily capture sequential
patterns (V oita et al., 2019). To reduce thisprox-
imity bias, SPROUTRAG learns a weighted aggre-
gation over heads and layers:
Aij=LX
l=1HX
h=1wl,hAttn(l,h)
ij,(1)
wherew l,his defined as:
wl,h=exp(α l,h)PL
l′=1PH
h′=1exp(α l′,h′).(2)
The learnable scalars αl,hallow the model to em-
phasize attention heads that better capture semantic
co-relevance. We then symmetrize the aggregated
attention to obtain a mutual relation score:
Mij=Aij+A ji
2.(3)
The tree is built bottom-up. Initially, each sen-
tence chunk is a leaf node. At each step, we merge
the pair of active nodes with the highest mutual
attention score. The parent embedding is computed
as aprogressive embeddingof its children:
e(p) =e(u) +e(v)
2,(4)
where uandvare the merged child nodes. Af-
ter merging, the new parent inherits its strongest
3

Phase 1:  Attention-Guided Indexing (Offline)
1
Sentence-level Chunking
S1 S2 Sn...Split the document into fine-grained sentences
S3
2SLLM Encdoing
SLLMe1 e2 en...
Sentence embeddings
Inter-sentence
Attention Matrix A
(learned aggregation
across heads and layers)
3Attention-Guided Tree Construction
e(p12)
e(p1) e(p2)
e(s1s3) e(s4s2) e(s5s7) e(s6sn)
e1 e2 e3 e4 e6 e5 e7 ...
At each step, merge the pair of nodes with the highest learned
attention score to form a parent node.Phase 2:  Hierarchical Retrieval (Online)
1Query Encoding
What changes in
Shakespeare’s life
influenced his works?
SLLMQuery Vector e(q)
2Hierarchical Beam Search
e1 e2 e3 e4 e6 e5 e7 ...selected node
parent node
leaf node
pruned path
selected path
3Collect Candidates from Multiple Levels
4Similarity Reranking & Selectionscore...Top-k
ChunksPassed to the LLM
for answer Geneaio
Figure 2:Overview of SPROUTRAG.In the offline indexing phase (Phase 1), documents are split into sentence-
level chunks and encoded with a SLLM to obtain sentence embeddings and inter-sentence attention. Learned
aggregation over attention heads and layers guides bottom-up tree construction, producing an attention tree with
sentence embeddings at the leaves and progressive embeddings at internal nodes. In the online retrieval phase(Phase
2), a query is encoded and hierarchical beam search traverses the tree, collecting candidates from multiple levels
before similarity reranking selects the top-kchunks for answer generation.
relation to each remaining node:
Mpr= max(M ur,Mvr).(5)
This single-linkage update preserves long-range
semantic connections as the hierarchy grows. The
result is anattention tree Twhose leaves retain
sentence-level precision and whose internal nodes
represent broader semantic units.
3.2 Hierarchical Retrieval
Given a query q, SPROUTRAG first encodes it
with the same SLLM used during indexing to ob-
tain a query embedding e(q). Retrieval is then
performed over the attention-guided binary tree,
where each node vrepresents a document span at
a specific granularity. Leaf nodes correspond to
sentence-level chunks, while internal nodes rep-
resent progressively larger groups of semantically
related sentences. This allows SPROUTRAG to
retrieve evidence at the level most appropriate for
the query, rather than relying on a fixed chunk size.
Each candidate node is scored by cosine simi-
larity between the query embedding and the noderepresentation:
sim(q, v) =e(q)⊤e(v)
∥e(q)∥∥e(v)∥.(6)
Starting from the root node, retrieval proceeds
via hierarchical beam search. Let Btdenote the
active beam at depth t, with B0={v root}. At
each step, SPROUTRAG expands the children of
the current beam nodes and retains the top- bmost
relevant nodes:
Bt+1= Topb [
v∈BtChild(v),sim(q,·)!
,(7)
where bis the beam width. This search strategy fo-
cuses computation on the most promising branches
of the tree while still allowing the retriever to ex-
plore multiple semantically relevant regions of the
document.
In parallel, SPROUTRAG collects relevant nodes
encountered during traversal. Let Vvisitdenote the
set of all nodes scored during beam search:
Vvisit=[
t[
v∈BtChild(v).(8)
4

Algorithm 1SPROUTRAG Indexing and Retrieval
Require: Document D, query q, beam width b,
thresholdδ, top-k
Ensure:Retrieved evidenceF
1:{Offline indexing}
2:S←SplitSentences(D)
3:E,A ←SLLM(S)
4:A←P
l,hwl,hA(l,h)
5:M←(A+A⊤)/2
6:N ← {Leaf(s i, ei)|si∈S, e i∈E}
7:while|N|>1do
8:(u, v)←arg max u̸=v;u,v∈N Muv
9:p←Node(u, v)
10:e p←(e u+ev)/2
11:forr∈ N \ {u, v}do
12:M pr←max(M ur,Mvr)
13:M rp←M pr
14:end for
15:N ←(N \ {u, v})∪ {p}
16:end while
17:T ←root(N)
18:{Online retrieval}
19:e q←SLLM(q)
20:C ← ∅,B ← {T }
21:whileB ̸=∅do
22:C ← C ∪ {v∈ B |sim(e q, ev)≥δ}
23:X ←S
v∈BChildren(v)
24:B ←TopB(X, b,sim(e q,·))
25:end while
26:F←TopK(Rerank(C, q), k)
27:returnF
The retrieval candidate set is then defined as all
visited nodes whose similarity exceeds alearned
thresholdδ:
C={v∈ V visit : sim(q, v)≥δ}.(9)
The candidate set Ccontains evidence at multiple
granularities, from sentence-level leaves to larger
subtrees, allowing SPROUTRAG to retrieve either
precise facts or broader multi-sentence context as
needed. The collected candidates are reranked by
similarity or a lightweight reranker, and the top- k
chunks are passed to the answer generator. The
complete indexing and retrieval procedure is sum-
marized in Algorithm 1.
3.3 Joint Training
The pretrained SLLM is not optimized for retrieval
or for constructing retrieval-oriented document
structures. We therefore fine-tune SPROUTRAGwith a joint objective that improves both the em-
bedding space and the attention tree.
Retrieval objective.We train the SLLM embed-
dings with contrastive learning over query–passage
pairs. Given a query q, a positive passage p+, and
hard negatives {pj}, each passage is represented
by mean-pooling its sentence embeddings. We op-
timize:
Lret=−logexp(sim(q, p+)/τ)P
jexp(sim(q, p j)/τ),(10)
where τis a temperature parameter. This objective
aligns queries with relevant passages and separates
them from hard negatives.
Structure objective.Good embeddings alone do
not guarantee a useful hierarchy. Since the tree
depends on the learned attention matrix, we add
an attention regularizer that encourages co-relevant
sentence pairs to receive high mutual attention. Let
Gbe the set of sentence pairs within a positive
passage that jointly support the query. We define:
Lattn=−1
|G|X
(si,sj)∈GlogAij+A ji
2
.(11)
This objective directly shapes the learned head–
layer aggregation, encouraging the induced tree to
group semantically related evidence into coherent
and retrievable subtrees.
Final objective.The final training loss is:
L=L ret+λL attn,(12)
where λcontrols the strength of structure regular-
ization. After training, the learned aggregation
weights are used during offline indexing, and re-
trieval requires only query encoding, tree traversal,
and reranking. Thus, SPROUTRAG avoids external
LLM filtering and lossy LLM-based summarization
while enabling efficient multi-granularity retrieval.
4 Experiments and Results
4.1 Experimental Setup
Benchmarks.We evaluate SPROUTRAG on four
retrieval benchmarks spanning scientific, legal, and
open-domain settings: SCI-DOCS (Cohan et al.,
2020), LegalBench-RAG (Pipitone and Alami,
2024), Dragonball (Zhu et al., 2025), and MS
MARCO (Nguyen et al., 2016). For end-to-end
5

answer generation, we further evaluate on Hot-
potQA (Yang et al., 2018), WebQuestions (Berant
et al., 2013), and Dragonball. See Appendix A.1
for more details.
Baselines.We compare against representative
chunking and structured retrieval methods, in-
cluding Dense X Retrieval (Chen et al., 2024),
Meta-Chunking (Zhao et al., 2025b), MoC (Zhao
et al., 2025a), RAPTOR (Sarthi et al., 2024),
LightRAG (Guo et al., 2025), PropRAG (Wang
and Han, 2025), and SAKI-RAG (Tao et al.,
2025). GraphRAG (Edge et al., 2025), Reflec-
tiveRAG (Verma et al., 2026), PageIndex (Zhang
et al., 2025a), and REFRAG (Lin et al., 2025) are
reported only for final task performance, as they
primarily involve LLM-heavy reasoning, genera-
tion, summarization, or decoding-time optimiza-
tion rather than efficient retrieval. For fair com-
parison, methods requiring an LLM or reranker
use the same QWEN3-8B (Team, 2025) gener-
ator and QWEN3-RERANKER-4B (Zhang et al.,
2025b) reranker. Other than unifying the generator
and reranker, we follow the settings recommended
in the original papers for each baseline. See Ap-
pendix A.2 for more details.
Metrics.For retrieval evaluation, we report Re-
call, Precision, and Information Efficiency (IE),
where IE = Recall×Precision . To keep the ev-
idence budget comparable across methods, kde-
notes the number of underlying evidence units used
for evaluation rather than the number of retrieved
tree nodes. For hierarchical outputs, retrieved in-
ternal nodes are expanded into their underlying
evidence units, with each contained unit counted to-
ward the same top- kbudget. We compute each met-
ric at k∈ {1,3,5} and report the average across
these three cutoffs. For end-to-end generation, we
report F1 on HotpotQA and WebQuestions, and
ROUGE-L (Lin, 2004), METEOR (Banerjee and
Lavie, 2005), and BERTScore (Zhang* et al., 2020)
on Dragonball. We also report online efficiency us-
ing Tok/Q and latency, where Tok/Q counts online
model-token usage per query, excluding offline in-
dexing and output tokens. All reported results are
averaged over three independent runs.
Implementation and Training Details.We use
the 1.3B-parameter SLLM (An et al., 2024)2as the
sentence encoder and split documents into chunks
of up to two sentences. SPROUTRAG is trained
on 30K query–passage examples sampled from
2https://github.com/cavedweller509/SentenceVAECLaRa (He et al., 2025), fine-tuning the SLLM and
learned head–layer aggregation weights with the
joint objective in Eq. 12. Unless otherwise stated,
we use the following hyperparameters as the default
setting: 3 training epochs with AdamW, learning
rates of 2×10−5for the SLLM and 1×10−3
for the aggregation scalars, batch size 32, temper-
ature τ= 0.05 , attention weight λ= 0.1 , and 5%
linear warmup. At retrieval time, the default set-
ting uses beam width b= 5 , collects candidates
from all tree levels, and reranks them with QWEN3-
RERANKER-4B. Final answers are generated with
QWEN3-8B. For fair comparison, baselines that
require an LLM or reranker use the same models;
otherwise, we follow their original settings. All ex-
periments use 8 NVIDIA A100 80GB GPUs, and
results are averaged over three runs.
4.2 Retrieval Quality
Table 1 reports retrieval performance across four
benchmarks. SPROUTRAG achieves the highest IE
on all datasets, improving over the strongest base-
line by8.06points on Dragonball,4.65on SCI-
DOCS,4.90on LegalBench-RAG, and6.83on MS
MARCO. These improvements are not driven by re-
call alone: SPROUTRAG also obtains thebest pre-
cision on every benchmark. This suggests that the
attention-guided hierarchy helps retrieve broader
supporting context while avoiding the noise intro-
duced by overly large or weakly related chunks.
The comparison with SAKI-RAG is especially
informative. While SAKI-RAG achieves strong
precision, particularly on Dragonball and SCI-
DOCS, its pairwise expansion limits evidence
aggregation, reducing recall and IE. In contrast,
SPROUTRAG converts sentence-level attention
into aglobal tree structure, enabling retrieval over
individual chunks, internal nodes, and subtrees.
This preserves SAKI-RAG’s precision benefits
while improving IE across all datasets. Structured
baselines such as RAPTOR, LightRAG, PropRAG,
and MoC improve recall over flat or boundary-
based chunking, but their clustering, graph, propo-
sition, or routing structures do not explicitly model
learned multi-sentence composition. SPROUTRAG
bridges this gap: leaves retain fine-grained evi-
dence, while internal nodes recover coherent con-
text, yielding the strongest recall–precision trade-
off. Appendix C provides a qualitative example.
6

MethodDragonball SCI-DOCS
Rec.↑Pre.↑IE↑Rec.↑Pre.↑IE↑
Dense X Retrieval3.18 ±0.09 5.76±0.14 0.18±0.02 95.64 ±0.31 88.13 ±0.42 84.31 ±0.47
Meta-Chunking-PPL42.91 ±0.54 44.62 ±0.63 19.14 ±0.37 22.16 ±0.71 18.73 ±0.64 4.15±0.19
Meta-Chunking-MSP32.74 ±0.47 42.08 ±0.58 13.77 ±0.31 96.37 ±0.29 88.42 ±0.46 85.21 ±0.43
RAPTOR32.19 ±0.61 41.92 ±0.76 13.49 ±0.34 97.72 ±0.24 93.03 ±0.39 90.92 ±0.41
LightRAG47.38 ±0.69 54.16 ±0.72 25.67 ±0.46 97.51 ±0.27 92.84 ±0.44 90.53 ±0.45
PropRAG49.81 ±0.64 58.24 ±0.79 29.01 ±0.52 98.07 ±0.22 94.36 ±0.35 92.55 ±0.38
MoC51.26 ±0.57 60.47 ±0.68 30.99 ±0.44 98.18 ±0.19 94.73 ±0.31 93.01 ±0.34
SAKI-RAG36.83 ±0.52 78.61 ±0.64 28.95 ±0.41 91.82 ±0.33 97.43 ±0.28 89.46 ±0.36
SPROUTRAG 45.76 ±0.43 85.34 ±0.51 39.05 ±0.38 98.74 ±0.16 98.91 ±0.18 97.66 ±0.21
MethodLegalBench-RAG MS MARCO
Rec.↑Pre.↑IE↑Rec.↑Pre.↑IE↑
Dense X Retrieval24.83 ±0.38 35.74 ±0.51 8.88±0.21 58.41 ±0.46 62.68 ±0.55 36.61 ±0.48
Meta-Chunking-PPL27.82 ±0.44 38.41 ±0.57 10.69 ±0.26 60.84 ±0.52 63.97 ±0.49 38.92 ±0.45
Meta-Chunking-MSP28.31 ±0.41 39.26 ±0.49 11.12 ±0.24 61.73 ±0.43 64.36 ±0.52 39.73 ±0.39
RAPTOR28.64 ±0.48 39.83 ±0.62 11.41 ±0.29 62.48 ±0.56 65.81 ±0.67 41.12 ±0.53
LightRAG30.62 ±0.53 43.97 ±0.58 13.47 ±0.33 66.13 ±0.51 68.42 ±0.61 45.25 ±0.57
PropRAG31.47 ±0.46 45.28 ±0.54 14.25 ±0.32 66.92 ±0.48 69.83 ±0.58 46.73 ±0.49
MoC32.18 ±0.42 46.11 ±0.51 14.84 ±0.27 67.31 ±0.45 70.26 ±0.53 47.29 ±0.46
SAKI-RAG31.38 ±0.39 46.27 ±0.47 14.52 ±0.25 68.04 ±0.42 71.58 ±0.49 48.70 ±0.44
SPROUTRAG 36.91 ±0.34 53.48 ±0.41 19.74 ±0.29 72.86 ±0.37 76.21 ±0.44 55.53 ±0.40
Table 1: Retrieval performance across four benchmarks. Recall, Precision, and IE are averaged over @1, @3, and
@5, with IE computed at each cutoff before averaging. Values report the mean over three independent runs, and
the red ±values indicate the corresponding standard deviation. The shaded row marks SPROUTRAG. Refer to
Appendix B for the results @1, @3, and @5.
4.3 End-to-End Performance and Efficiency
We next evaluate whether the retrieval improve-
ments translate into stronger final answers. Ta-
ble 2 compares SPROUTRAG with system-level
RAG methods on HotpotQA, WebQuestions,
and Dragonball. PageIndex achieves the high-
est final answer scores, but it requires sub-
stantially more online computation due to its
reasoning-based search and evidence construc-
tion. REFRAG improves efficiency compared
with reflection-heavy or reasoning-heavy systems,
but SPROUTRAG still provides the strongest
performance–efficiency tradeoff: it outperforms
GraphRAG, ReflectiveRAG, and REFRAG across
all final-performance metrics, while using only
4.38K online tokens per queryand193 msla-
tency. The reported cost measures online per-query
inference and excludes offline training and index-
ing.SPROUTRAG does require an upfront train-
ing stage: we fine-tune the SLLM and attentionaggregation weights on a 30K-example subset of
CLaRa. However,this cost is paid once and
reused across datasets, similar to other systems
with offline preparation or model adaptation costs.
The cross-dataset results show that the learned
attention-guided hierarchy generalizes without re-
training for each benchmark, making the training
cost amortized rather than query-time overhead.
4.4 Ablation Study
Training objectives.In the first group of Ta-
ble 3, evaluates the role of the training objectives.
TheNot trainedvariant performs worst across all
datasets, showing that the pretrained SLLM atten-
tion and embeddings are not sufficient for retrieval-
oriented tree construction. Removing Lretsubstan-
tially reduces both recall and IE, since the query
and evidence embeddings are no longer explicitly
aligned. Removing Lattnis less damaging than
removing the retrieval loss, but still causes a con-
sistent drop, especially in IE. This confirms that
7

MethodFinal Performance
HotpotQA WebQuestions Dragonball Average Cost
F1↑F1↑R-L↑MTR↑BRT↑Tok/Q↓Lat. (ms)↓
GraphRAG 72.18 64.73 0.346 0.361 0.637 16238 2317
ReflectiveRAG 70.64 63.29 0.334 0.349 0.622 11274 1186
REFRAG 73.42 65.38 0.351 0.368 0.641 5436 492
PageIndex79.36 70.81 0.389 0.406 0.69124620 2847
SPROUTRAG 76.47 68.12 0.372 0.389 0.671 4382 193
Table 2: End-to-end answer quality and online efficiency. HotpotQA and WebQuestions are evaluated with F1,
while Dragonball uses ROUGE-L (R-L), METEOR (MTR), and BERTScore (BRT). Tok/Q counts online model
input tokens per query, excluding offline training, indexing, and output tokens; Lat. reports online per-query latency.
All methods use the same generator and reranker when applicable.
VariantDragonball SCI-DOCS LegalBench-RAG MS MARCO
Rec.↑Pre.↑IE↑Rec.↑Pre.↑IE↑Rec.↑Pre.↑IE↑Rec.↑Pre.↑IE↑
SPROUTRAG (b= 5, λ= 0.1) 45.76 85.34 39.05 98.74 98.91 97.66 36.91 53.48 19.74 72.86 76.21 55.53
Training Objectives
Not trained 34.28 69.41 23.79 90.36 92.18 83.28 29.47 42.53 12.53 64.18 67.42 43.28
w/oL ret 37.64 73.52 27.67 92.71 94.86 87.96 31.08 45.17 14.04 66.24 69.73 46.18
w/oL attn 41.39 78.26 32.39 96.42 96.88 93.41 33.52 48.31 16.20 69.18 72.64 50.25
Tree and Retrieval Design
Uniform attention aggregation 39.18 76.94 30.15 95.83 96.12 92.11 32.74 47.36 15.51 68.41 71.52 48.92
Embedding-similarity tree 40.72 79.38 32.32 96.31 96.74 93.17 33.18 48.42 16.06 69.36 72.48 50.28
Leaf-only retrieval 38.26 83.19 31.83 94.87 98.12 93.07 31.94 52.61 16.80 67.83 75.42 51.16
Greedy search 39.84 82.47 32.86 96.25 98.34 94.65 32.81 51.76 16.98 68.92 74.86 51.60
Hyperparameter Sensitivity
Beam widthb= 3 44.12 84.71 37.37 98.31 98.83 97.16 35.72 52.86 18.88 71.94 75.63 54.42
Beam widthb= 10 46.24 82.63 38.21 98.91 98.42 97.35 37.28 51.92 19.36 73.41 74.32 54.56
λ= 0.05 44.68 83.92 37.50 98.42 98.63 97.07 35.96 52.47 18.87 72.14 75.48 54.46
λ= 0.20 45.18 82.74 37.38 98.58 98.37 96.97 36.42 51.83 18.88 72.49 74.93 54.31
Table 3: Ablation study on retrieval performance. Metrics are averaged over @1, @3, and @5. The blue row is the
default SPROUTRAG setting ( b= 5, λ= 0.1 ). The three groups evaluate training objectives, tree/retrieval design,
and sensitivity tobandλ.
embedding quality and tree quality require com-
plementary supervision: the retrieval loss aligns
query–passage representations, while the attention-
structure loss shapes the hierarchy used for multi-
granularity retrieval.
Tree and retrieval design.The second group
in Table 3 examines the necessity of the attention-
guided hierarchy. Uniform attention aggregation
reduces performance, highlighting that averaging
heads and layers introducesproximity biasand
weakens the tree. An embedding-similarity tree
also underperforms, showing that SLLM atten-
tion encodes structural information beyond embed-
dings. Leaf-only retrieval maintains high precision
but lowers recall and IE, while greedy search suf-
fers from early path commitment. These results
demonstrate thatlearned attention aggregation,
internal-node retrieval, andbeam searchare allcrucial for balancing precise evidence with broader
contextual coverage.
Hyperparameter sensitivity.The final group in
Table 3 studies beam width band attention regu-
larization weight λ. Reducing the beam width to
b= 3 slightly lowers IE because fewer semantic
paths are explored, while increasing it to b= 10 im-
proves recall but slightly reduces precision, yield-
ing no overall advantage over the default b= 5 .
Similarly, both λ= 0.05 andλ= 0.20 underper-
form the default λ= 0.1 : a weaker structure loss
provides insufficient guidance for tree construc-
tion, while a stronger one can overemphasize atten-
tion alignment at the expense of retrieval precision.
Overall, SPROUTRAG is stable across reasonable
settings, with b= 5 andλ= 0.1 providing the best
recall–precision tradeoff.
8

5 Conclusion and Future Work
We introduced SPROUTRAG, an attention-guided
hierarchical RAG framework that organizes
sentence-level chunks into a learned tree for multi-
granularity retrieval. Rather than relying on fixed
chunk boundaries, pairwise context expansion,
lossy summarization, or inference-time LLM filter-
ing, SPROUTRAG uses learned SLLM attention ag-
gregation to construct a retrieval-oriented hierarchy.
At inference time, hierarchical beam search selects
evidence from sentence leaves, internal nodes, and
broader subtrees, allowing the retriever to balance
fine-grained precision with contextual coherence.
Across benchmarks, SPROUTRAG improves re-
trieval information efficiency by 6.1% on average,
offering a strongperformance–efficiency tradeoff
that approaches LLM-heavy systems while using
far fewer online tokens and lower latency. While
SPROUTRAG generalizes well after one-time train-
ing, several directions remain open. Future work
can explore richer node composition functions be-
yond mean pooling, such as gated or attention-
based composition, and dynamic tree adaptation
or query-dependent traversal policies for complex
multi-hop retrieval.
Limitations
While SPROUTRAG improves multi-granularity
retrieval without inference-time LLM filtering, it
has some limitations. First, the hierarchy is cur-
rently built as abinary tree, which may be restric-
tive when several sentences jointly form a coher-
ent semantic unit and should be grouped together
simultaneously. Multi-branch trees could better
capture such many-to-many dependencies. Second,
SPROUTRAG requires an upfront training stage for
the SLLM and attention aggregation weights. Al-
though this is a one-time cost that transfers across
datasets in our experiments, it is still more expen-
sive than using an off-the-shelf retriever without
adaptation. Finally, tree construction is offline and
fixed during retrieval. While this makes inference
efficient and avoids rebuilding the index per query,
it may be less flexible when queries require evi-
dence reorganized by query-specific relevance.
References
Hongjun An, Yifan Chen, Zhe Sun, and Xuelong Li.
2024. Sentencevae: Enable next-sentence predic-
tion for large language models with faster speed,higher accuracy and longer context.Preprint,
arXiv:2408.00655.
Isabelle Augenstein, Timothy Baldwin, Meeyoung Cha,
Tanmoy Chakraborty, Giovanni Luca Ciampaglia,
David Corney, Renee DiResta, Emilio Ferrara, Scott
Hale, Alon Halevy, and 1 others. 2024. Factuality
challenges in the era of large language models and
opportunities for fact-checking.Nature Machine In-
telligence, 6(8):852–863.
Satanjeev Banerjee and Alon Lavie. 2005. METEOR:
An automatic metric for MT evaluation with im-
proved correlation with human judgments. InPro-
ceedings of the ACL Workshop on Intrinsic and Ex-
trinsic Evaluation Measures for Machine Transla-
tion and/or Summarization, pages 65–72, Ann Arbor,
Michigan. Association for Computational Linguis-
tics.
Jonathan Berant, Andrew Chou, Roy Frostig, and Percy
Liang. 2013. Semantic parsing on Freebase from
question-answer pairs. InProceedings of the 2013
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 1533–1544, Seattle, Wash-
ington, USA. Association for Computational Linguis-
tics.
Tong Chen, Hongwei Wang, Sihao Chen, Wenhao Yu,
Kaixin Ma, Xinran Zhao, Hongming Zhang, and
Dong Yu. 2024. Dense X retrieval: What retrieval
granularity should we use? InProceedings of the
2024 Conference on Empirical Methods in Natural
Language Processing, pages 15159–15177, Miami,
Florida, USA. Association for Computational Lin-
guistics.
Arman Cohan, Sergey Feldman, Iz Beltagy, Doug
Downey, and Daniel Weld. 2020. SPECTER:
Document-level representation learning using
citation-informed transformers. InProceedings
of the 58th Annual Meeting of the Association
for Computational Linguistics, pages 2270–2282,
Online. Association for Computational Linguistics.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and
Jonathan Larson. 2025. From local to global: A
graph rag approach to query-focused summarization.
Preprint, arXiv:2404.16130.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao
Huang. 2025. LightRAG: Simple and fast retrieval-
augmented generation. InFindings of the Associa-
tion for Computational Linguistics: EMNLP 2025,
pages 10746–10761, Suzhou, China. Association for
Computational Linguistics.
Michael Günther, Isabelle Mohr, Daniel James Williams,
Bo Wang, and Han Xiao. 2025. Late chunking: Con-
textual chunk embeddings using long-context embed-
ding models.Preprint, arXiv:2409.04701.
Jie He, Richard He Bai, Sinead Williamson, Jeff Z.
Pan, Navdeep Jaitly, and Yizhe Zhang. 2025. Clara:
9

Bridging retrieval and generation with continuous
latent reasoning.Preprint, arXiv:2511.18659.
Bowen Jin, Jinsung Yoon, Jiawei Han, and Sercan O
Arik. 2025a. Long-context LLMs meet RAG: Over-
coming challenges for long inputs in RAG. InThe
Thirteenth International Conference on Learning
Representations.
Bowen Jin, Jinsung Yoon, Jiawei Han, and Sercan O
Arik. 2025b. Long-context LLMs meet RAG: Over-
coming challenges for long inputs in RAG. InThe
Thirteenth International Conference on Learning
Representations.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. InAdvances in Neural Infor-
mation Processing Systems, volume 33, pages 9459–
9474. Curran Associates, Inc.
Chin-Yew Lin. 2004. ROUGE: A package for auto-
matic evaluation of summaries. InText Summariza-
tion Branches Out, pages 74–81, Barcelona, Spain.
Association for Computational Linguistics.
Xiaoqiang Lin, Aritra Ghosh, Bryan Kian Hsiang Low,
Anshumali Shrivastava, and Vijai Mohan. 2025. Re-
frag: Rethinking rag based decoding.Preprint,
arXiv:2509.01092.
Hao Liu, Zhengren Wang, Xi Chen, Zhiyu Li, Feiyu
Xiong, Qinhan Yu, and Wentao Zhang. 2025. Ho-
pRAG: Multi-hop reasoning for logic-aware retrieval-
augmented generation. InFindings of the Associa-
tion for Computational Linguistics: ACL 2025, pages
1897–1913, Vienna, Austria. Association for Compu-
tational Linguistics.
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024. Lost in the middle: How language mod-
els use long contexts.Transactions of the Association
for Computational Linguistics, 12:157–173.
Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng
Gao, Saurabh Tiwary, Rangan Majumder, and
Li Deng. 2016. MS MARCO: A human gener-
ated machine reading comprehension dataset.CoRR,
abs/1611.09268.
Nicholas Pipitone and Ghita Houir Alami. 2024.
Legalbench-rag: A benchmark for retrieval-
augmented generation in the legal domain.Preprint,
arXiv:2408.10343.
Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh
Khanna, Anna Goldie, and Christopher D Manning.
2024. RAPTOR: Recursive abstractive processing
for tree-organized retrieval. InThe Twelfth Interna-
tional Conference on Learning Representations.Wenyu Tao, Xiaofen Xing, Zeliang Li, and Xiangmin
Xu. 2025. SAKI-RAG: Mitigating context fragmen-
tation in long-document RAG via sentence-level at-
tention knowledge integration. InProceedings of
the 2025 Conference on Empirical Methods in Natu-
ral Language Processing, pages 1195–1213, Suzhou,
China. Association for Computational Linguistics.
LangChain Team. 2024. Langchain: A framework for
developing applications powered by language mod-
els.
Qwen Team. 2025. Qwen3 technical report.Preprint,
arXiv:2505.09388.
Akshay Verma, Swapnil Gupta, Siddharth Pillai, Prateek
Sircar, and Deepak Gupta. 2026. ReflectiveRAG: Re-
thinking adaptivity in retrieval-augmented generation.
InProceedings of the 19th Conference of the Euro-
pean Chapter of the Association for Computational
Linguistics (Volume 5: Industry Track), pages 377–
384, Rabat, Morocco. Association for Computational
Linguistics.
Elena V oita, David Talbot, Fedor Moiseev, Rico Sen-
nrich, and Ivan Titov. 2019. Analyzing multi-head
self-attention: Specialized heads do the heavy lift-
ing, the rest can be pruned. InProceedings of the
57th Annual Meeting of the Association for Computa-
tional Linguistics, pages 5797–5808, Florence, Italy.
Association for Computational Linguistics.
Jingjin Wang and Jiawei Han. 2025. PropRAG: Guid-
ing retrieval with beam search over proposition paths.
InProceedings of the 2025 Conference on Empiri-
cal Methods in Natural Language Processing, pages
6212–6227, Suzhou, China. Association for Compu-
tational Linguistics.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. 2018. HotpotQA: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing, pages
2369–2380, Brussels, Belgium. Association for Com-
putational Linguistics.
Jiahao Zhang, Haiyang Zhang, Dongmei Zhang, Liu
Yong, and Shen Huang. 2024. End-to-end beam re-
trieval for multi-hop question answering. InProceed-
ings of the 2024 Conference of the North American
Chapter of the Association for Computational Lin-
guistics: Human Language Technologies (Volume
1: Long Papers), pages 1718–1731, Mexico City,
Mexico. Association for Computational Linguistics.
Mingtian Zhang, Yu Tang, and PageIndex Team. 2025a.
Pageindex: Next-generation vectorless, reasoning-
based rag.PageIndex Blog.
Tianyi Zhang*, Varsha Kishore*, Felix Wu*, Kilian Q.
Weinberger, and Yoav Artzi. 2020. Bertscore: Eval-
uating text generation with bert. InInternational
Conference on Learning Representations.
10

Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang,
Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren
Zhou. 2025b. Qwen3 embedding: Advancing text
embedding and reranking through foundation models.
Preprint, arXiv:2506.05176.
Jihao Zhao, Zhiyuan Ji, Zhaoxin Fan, Hanyu Wang,
Simin Niu, Bo Tang, Feiyu Xiong, and Zhiyu Li.
2025a. MoC: Mixtures of text chunking learners for
retrieval-augmented generation system. InProceed-
ings of the 63rd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Pa-
pers), pages 5172–5189, Vienna, Austria. Associa-
tion for Computational Linguistics.
Jihao Zhao, Zhiyuan Ji, Yuchen Feng, Pengnian Qi,
Simin Niu, Bo Tang, Feiyu Xiong, and Zhiyu Li.
2025b. Meta-chunking: Learning text segmenta-
tion and semantic completion via logical perception.
Preprint, arXiv:2410.12788.
Kunlun Zhu, Yifan Luo, Dingling Xu, Yukun Yan,
Zhenghao Liu, Shi Yu, Ruobing Wang, Shuo Wang,
Yishan Li, Nan Zhang, Xu Han, Zhiyuan Liu, and
Maosong Sun. 2025. RAGEval: Scenario specific
RAG evaluation dataset generation framework. In
Proceedings of the 63rd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 8520–8544, Vienna, Austria.
Association for Computational Linguistics.
A Benchmark and Baseline Details
A.1 Benchmark Dataset Details
We include four retrieval-focused benchmarks—
SCI-DOCS, LegalBench-RAG, Dragonball, and
MS MARCO—and three end-to-end genera-
tion benchmarks—HotpotQA, WebQuestions, and
Dragonball. Together, these datasets cover scien-
tific retrieval, legal retrieval, open-domain passage
retrieval, multi-hop question answering, short-form
factual QA, and multi-domain RAG evaluation.
SCI-DOCS (Cohan et al., 2020)is a scientific
document representation benchmark introduced
with SPECTER. It contains multiple document-
level tasks, including citation prediction, document
classification, and recommendation. We use SCI-
DOCS as a scientific retrieval benchmark because
scientific abstracts are dense, terminology-heavy,
and often contain multiple related concepts within
a short span. This makes the dataset useful for test-
ing whether SPROUTRAG can construct coherent
sentence-level hierarchies in technical domains.
LegalBench-RAG (Pipitone and Alami, 2024)
is designed specifically to evaluate retrieval in
legal RAG pipelines. It contains 6,858 query-
answer pairs over legal documents such as NDAs,M&A agreements, commercial contracts, and pri-
vacy policies. Unlike broad document retrieval,
LegalBench-RAG emphasizes precise snippet re-
trieval: the model must identify minimal legal ev-
idence rather than simply retrieve a generally rel-
evant document. This makes it a strong test of
fine-grained precision.
Dragonball (Zhu et al., 2025)is a multi-domain
and multilingual RAG benchmark released as part
of RAGEval. It contains questions across finance,
legal, and medical scenarios in English and Chi-
nese. We use Dragonball for both retrieval and
end-to-end generation because it combines het-
erogeneous domains, long evidence contexts, and
domain-specific terminology. This setting tests
whether retrieval methods can recover relevant evi-
dence without introducing excessive distractor con-
text.
MS MARCO (Nguyen et al., 2016)is a large-
scale open-domain passage retrieval benchmark
built from real web search queries. Its corpus con-
sists of millions of short passages, and the task
requires identifying passages that answer natural
language questions. Compared with SCI-DOCS
and LegalBench-RAG, MS MARCO has shorter re-
trieval units and more direct query-passage match-
ing, providing a complementary test of retrieval
effectiveness when evidence is already compact.
HotpotQA (Yang et al., 2018)is an open-
domain multi-hop QA benchmark built from
Wikipedia. Its questions require reasoning over
multiple supporting documents or facts, and the
dataset provides sentence-level supporting-fact an-
notations. We use HotpotQA for end-to-end an-
swer generation because it directly tests whether
retrieved evidence supports multi-step reasoning,
which aligns with SPROUTRAG’s goal of retriev-
ing evidence across multiple granularities.
WebQuestions (Berant et al., 2013)is an open-
domain factual QA benchmark built from natu-
ral language questions collected from web search
logs. The answers are typically short entities or
phrases, making token-level F1 a suitable evalua-
tion metric. We include WebQuestions to evaluate
whether SPROUTRAG also improves short-form
factual QA, where retrieval must remain precise
and avoid adding unnecessary context.
11

A.2 Baseline Details
We compare SPROUTRAG against two groups of
baselines: efficient retrieval-oriented methods used
in the retrieval evaluation, and system-level RAG
methods used in the end-to-end generation com-
parison. Unless otherwise stated, we follow the
configurations recommended in the original pa-
pers. For methods requiring an LLM generator
or reranker, we use the same QWEN3-8B genera-
tor and QWEN3-RERANKER-4B reranker for fair
comparison.
Dense X Retrieval (Chen et al., 2024)decom-
poses documents into fine-grained propositions and
uses these propositions as retrieval units. This im-
proves precision by making each unit more atomic
and self-contained. However, proposition-level
retrieval can weaken broader contextual continu-
ity, since related facts are retrieved independently
rather than as coherent multi-sentence evidence.
Meta-Chunking (Zhao et al., 2025b)uses LLM-
based signals to identify semantically meaningful
chunk boundaries instead of relying on fixed-size
segmentation. We evaluate both variants:Meta-
Chunking-PPL, which uses perplexity changes
to detect boundaries, andMeta-Chunking-MSP,
which uses margin-sampling-based boundary deci-
sions. These methods improve chunk coherence,
but once chunks are formed, cross-chunk semantic
dependencies are not explicitly modeled.
MoC (Zhao et al., 2025a)improves over single-
strategy chunking by dynamically routing text to
different chunking strategies or granularity choices.
It is a strong adaptive chunking baseline because
it can better match the segmentation strategy to
the local document structure. However, MoC still
primarily operates at the chunk-construction stage
and does not build a retrieval-time hierarchy over
sentence-level evidence.
RAPTOR (Sarthi et al., 2024)recursively clus-
ters chunks and summarizes each cluster to build
a hierarchical tree. Retrieval can then operate
over both lower-level chunks and higher-level sum-
maries. This provides a natural multi-granularity
baseline, but its hierarchy is based on embedding-
space clustering and LLM-generated summaries,
which can introduce information loss and addi-
tional indexing cost.
LightRAG (Guo et al., 2025)augments retrieval
with graph-structured knowledge and combines lo-cal and global retrieval signals. It is designed to
improve retrieval over connected evidence by ex-
ploiting entity and relation structure. We include
it as a structured retrieval baseline, especially for
settings where graph-style evidence organization
can improve coverage.
PropRAG (Wang and Han, 2025)represents
documents using propositions and performs beam-
style traversal over proposition paths. It is closely
related to our use of beam search, but differs in
its underlying structure: PropRAG searches over a
proposition graph, whereas SPROUTRAG searches
over an attention-guided sentence hierarchy. Pro-
pRAG is therefore a strong baseline for testing
whether hierarchical sentence-level structure pro-
vides benefits beyond proposition-path retrieval.
SAKI-RAG (Tao et al., 2025)uses a Sentence-
Level Large Language Model (SLLM) to esti-
mate inter-sentence attention and expand retrieved
chunks with related sentences. It is the closest base-
line to SPROUTRAG because it also uses sentence-
level attention signals. However, SAKI-RAG per-
forms pairwise expansion and relies on LLM filter-
ing during retrieval, while SPROUTRAG converts
learned attention into a global tree and retrieves
across multiple granularities without inference-
time LLM filtering.
GraphRAG (Edge et al., 2025)constructs an
entity-relation graph from the corpus and uses
graph structure to support retrieval and generation.
It is effective for queries that align well with entity-
centric evidence, but it depends on reliable entity
extraction and relation construction. We include
GraphRAG in the end-to-end comparison because
it is a system-level RAG method with substantial
LLM-based preprocessing and reasoning.
ReflectiveRAG (Verma et al., 2026)introduces
adaptive retrieval and generation through self-
reflection. Instead of using a fixed retrieval bud-
get, it evaluates whether retrieved evidence is suffi-
cient and can reformulate or expand retrieval when
needed. This makes it a strong final-performance
baseline, but its main contribution lies in adaptive
evidence use and generation rather than efficient
retrieval structure.
PageIndex (Zhang et al., 2025a)replaces con-
ventional vector retrieval with a reasoning-based hi-
erarchical page index. An LLM navigates the index
and selects evidence through multi-step reasoning,
12

which can improve final answer quality. However,
because it performs LLM-heavy online search and
evidence construction, we include it only in end-
to-end performance comparisons rather than as an
efficient retrieval baseline.
REFRAG (Lin et al., 2025)focuses on
generation-side efficiency for RAG by exploiting
sparsity in retrieved contexts. It compresses re-
trieved chunks and selectively expands them during
decoding, reducing the effective context processed
by the generator. Since its main contribution is
decoding-time optimization rather than retrieval or
indexing, we include it in the final-performance
and efficiency comparison.
B Retrieval Performance at Different
Cutoffs
Tables 4, 5, and 6 report retrieval performance at
cutoffs k=1,k=3, and k=5, respectively, across
all six benchmarks. As expected, both precision
and recall increase monotonically with kfor all
methods, since retrieving more documents provides
greater coverage of relevant passages. The relative
ordering of methods remains consistent across all
cutoffs: SPROUTRAG outperforms all non-oracle
baselines at every depth. This consistency demon-
strates that the gains from topic-guided retrieval
are not specific to any particular cutoff, but reflect
a robust improvement in retrieval quality across the
full range of evaluation settings reported here.
C Qualitative Analysis: Recovering
Multi-Sentence Legal Evidence
Table 7 presents a qualitative example that illus-
trates why retrievinginternal tree nodesis useful.
The query asks whether a software services agree-
ment limits the provider’s liability and what ex-
ceptions apply. A complete answer requires more
than a single sentence or a pair of related sen-
tences: the model must recover the excluded dam-
ages, the aggregate liability cap, the scope of the
limitation across legal theories, and the carve-outs
for indemnification, confidentiality breaches, gross
negligence, and willful misconduct. These pieces
form a coherent clause-level unit, but they are dis-
tributed across four sentences. MoC retrieves a
locally coherent chunk containing the liability cap,
but this evidence is too narrow to answer the full
query. SAKI-RAG improves over local chunking
by linking the damage-exclusion sentence with theliability-cap sentence; however, its pairwise expan-
sion still misses the later exception sentence, which
is essential for a legally complete answer. In con-
trast, SPROUTRAG retrieves the internal node v1:4,
which groups all four relevant sentences into a sin-
gle clause-level unit. This allows the generator
to answer both parts of the query: the agreement
limits liability through damage exclusions and a
monetary cap, but the limitation does not apply to
the specified carve-outs.
13

MethodLegalBench-RAG MSMARCO SCI-DOCS Dragonball
IE↑Prec.↑Rec.↑IE↑Prec.↑Rec.↑IE↑Prec.↑Rec.↑IE↑Prec.↑Rec.↑
DenseXRetrieval 1.93 32.74 5.88 19.14 58.68 32.61 70.89 86.13 82.31 0.01 1.64 1.12
Meta-Chunking-PPL 2.72 35.41 7.69 20.94 59.97 34.92 0.36 16.73 2.15 6.72 41.62 16.14
Meta-Chunking-MSP 2.94 36.26 8.12 21.57 60.36 35.73 71.91 86.42 83.21 4.21 39.08 10.77
RAPTOR 3.10 36.83 8.41 22.94 61.81 37.12 80.94 91.03 88.92 4.08 38.92 10.49
LightRAG 4.29 40.97 10.47 26.57 64.42 41.25 80.42 90.84 88.53 11.60 51.16 22.67
PropRAG 4.76 42.28 11.25 28.13 65.83 42.73 83.63 92.36 90.55 14.37 55.24 26.01
MoC 5.10 43.11 11.84 28.68 66.26 43.29 84.39 92.73 91.01 16.09 57.47 27.99
SAKI-RAG 4.98 43.27 11.52 30.21 67.58 44.70 83.46 95.43 87.46 19.62 75.61 25.95
SPROUTRAG 8.45 50.48 16.74 37.21 72.21 51.53 93.91 98.04 95.79 29.68 82.34 36.05
Table 4: Retrieval performance at depth k=1 across four benchmarks (IE ↑, Precision ↑, Recall ↑).Bold= best;
underline = second-best. SPROUTRAG rows are shaded.
MethodLegalBench-RAG MSMARCO SCI-DOCS Dragonball
IE↑Prec.↑Rec.↑IE↑Prec.↑Rec.↑IE↑Prec.↑Rec.↑IE↑Prec.↑Rec.↑
DenseXRetrieval 2.84 34.99 8.13 21.96 61.68 35.61 73.44 87.63 83.81 0.01 5.01 0.14
Meta-Chunking-PPL 3.74 37.66 9.94 23.88 62.97 37.92 0.67 18.23 3.65 8.07 43.87 18.39
Meta-Chunking-MSP 3.99 38.51 10.37 24.54 63.36 38.73 74.48 87.92 84.71 5.38 41.33 13.02
RAPTOR 4.17 39.08 10.66 26.00 64.81 40.12 83.67 92.53 90.42 5.25 41.17 12.74
LightRAG 5.50 43.22 12.72 29.83 67.42 44.25 83.13 92.34 90.03 13.31 53.41 24.92
PropRAG 6.01 44.53 13.50 31.48 68.83 45.73 86.40 93.86 92.05 16.25 57.49 28.26
MoC 6.39 45.36 14.09 32.06 69.26 46.29 87.17 94.23 92.51 18.06 59.72 30.24
SAKI-RAG 6.27 45.52 13.77 33.67 70.58 47.70 86.23 96.93 88.96 21.96 77.86 28.20
SPROUTRAG 10.01 52.73 18.99 41.01 75.21 54.53 95.92 98.69 97.19 32.40 84.59 38.30
Table 5: Retrieval performance at depth k=3 across four benchmarks (IE ↑, Precision ↑, Recall ↑).Bold= best;
underline = second-best. SPROUTRAG rows are shaded.
MethodLegalBench-RAG MSMARCO SCI-DOCS Dragonball
IE↑Prec.↑Rec.↑IE↑Prec.↑Rec.↑IE↑Prec.↑Rec.↑IE↑Prec.↑Rec.↑
DenseXRetrieval 4.99 39.49 12.63 28.16 67.68 41.61 78.68 90.63 86.81 0.04 9.51 0.40
Meta-Chunking-PPL 6.09 42.16 14.44 30.29 68.97 43.92 1.41 21.23 6.65 11.07 48.37 22.89
Meta-Chunking-MSP 6.40 43.01 14.87 31.02 69.36 44.73 79.75 90.92 87.71 8.03 45.83 17.52
RAPTOR 6.61 43.58 15.16 32.66 70.81 46.12 89.24 95.53 93.42 7.87 45.67 17.24
LightRAG 8.22 47.72 17.22 36.89 73.42 50.25 88.69 95.34 93.03 17.04 57.91 29.42
PropRAG 8.83 49.03 18.00 38.71 74.83 51.73 92.07 96.86 95.05 20.31 61.99 32.76
MoC 9.27 49.86 18.59 39.35 75.26 52.29 92.86 97.23 95.51 22.31 64.22 34.74
SAKI-RAG 9.14 50.02 18.27 41.12 76.58 53.70 91.90 99.93 91.96 26.93 82.36 32.70
SPROUTRAG 13.44 57.23 23.49 49.16 81.21 60.53 100.00 100.00 100.00 38.13 89.09 42.80
Table 6: Retrieval performance at depth k=5 across four benchmarks (IE ↑, Precision ↑, Recall ↑).Bold= best;
underline = second-best. SPROUTRAG rows are shaded.
14

Method / Unit Retrieved Evidence Analysis
Query:Does the agreement limit the provider’s liability, and what exceptions or exclusions apply?
MoC Retrieved chunk: liability cap.Provider’s aggregate
liability under this Agreement shall not exceed the fees
paid by Client during the twelve (12) months preceding
the event giving rise to the claim.MoC identifies a locally coherent chunk
around the monetary cap, but the re-
trieved unit is too narrow for the query. It
answershow muchliability is capped, but
misses the excluded damages, the scope
across legal theories, and the exceptions.
SAKI-RAG Retrieved pairwise expansion: damage exclusion +
liability cap.
1.In no event shall Provider be liable for any in-
direct, incidental, special, consequential, exem-
plary, or punitive damages.
2.Provider’s aggregate liability shall not exceed the
fees paid during the prior twelve months.SAKI-RAG improves over a single chunk
by linking two related sentences. How-
ever, the evidence remains pairwise,
so it captures the main limitation but
misses the later sentence listing excep-
tions such as indemnification, confiden-
tiality breach, gross negligence, and will-
ful misconduct.
Leafs 1 In no event shall Provider be liable for any indirect, in-
cidental, special, consequential, exemplary, or punitive
damages arising out of or relating to this Agreement.Identifies excluded damages, but does not
provide the monetary cap or exceptions.
Leafs 2 Provider’s aggregate liability under this Agreement
shall not exceed the fees paid by Client during the
twelve (12) months preceding the event giving rise to
the claim.Provides the liability cap, but not the
scope or carve-outs.
Leafs 3 The foregoing limitation shall apply regardless of the
form of action, whether in contract, tort, strict liability,
or otherwise.Clarifies that the limitation applies across
legal theories.
Leafs 4 The limitations in this Section shall not apply to
Provider’s indemnification obligations, breach of confi-
dentiality, gross negligence, or willful misconduct.Provides the exceptions required for a
complete legal answer.
SPROUTRAG inter-
nal nodev 1:4Retrieved clause-level node containings 1–s4:
1. excluded damages,
2. aggregate liability cap,
3. scope across legal theories,
4. exceptions and carve-outs.SPROUTRAG retrieves the middle
node containing more than two sen-
tences.This gives the generator the full
limitation-of-liability clause, enabling a
complete answer that includes both the
limitation and the exceptions.
Table 7: Qualitative comparison on a limitation-of-liability query. MoC retrieves a locally coherent but incomplete
chunk, and SAKI-RAG retrieves a related sentence pair that captures the main limitation but misses the exception
sentence. SPROUTRAG retrieves an internal clause-level node containing four sentences, allowing the answer to
include excluded damages, the liability cap, legal-theory scope, and carve-outs.
15