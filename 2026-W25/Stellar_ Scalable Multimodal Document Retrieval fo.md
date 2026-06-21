# Stellar: Scalable Multimodal Document Retrieval for Natural Language Queries

**Authors**: Yuxiang Guo, Zhonghao Hu, Yuren Mao, Yuhang Liu, Congcong Ge, Xiaolu Zhang, Jun Zhou, Yunjun Gao

**Published**: 2026-06-18 08:57:26

**PDF URL**: [https://arxiv.org/pdf/2606.19960v1](https://arxiv.org/pdf/2606.19960v1)

## Abstract
Multimodal document retrieval--selecting the most relevant multimodal document from a large corpus to answer a natural language query--plays an essential role in Retrieval-Augmented Generation (RAG) systems. State-of-the-art methods represent each document and query with multiple token-level embeddings and use late interaction to achieve high effectiveness. However, such multi-vector representations incur substantial memory overhead during retrieval, leading to poor scalability and hindering real-world deployment. In this paper, we present Stellar, a scalable multimodal document retrieval framework that stores token-level document embeddings on disk and loads only a small set of candidate embeddings into memory for late interaction. Stellar comprises two key components: (i) Lexical Representation-based Filtering (LRF), which fine-tunes a Multimodal Large Language Model (MLLM) as a sparse encoder to produce high-quality lexical representations, enabling efficient and effective document filtering to significantly reduce the candidate set; (ii) Efficient Disk-backed Late Interaction (DLI), which designs an on-disk token embedding storage layout guided by a balanced clustering algorithm, and dynamically loads only the necessary token embeddings into memory using a simple yet effective cost model. Extensive experiments on four real-world benchmarks and a newly presented large-scale dataset demonstrate that Stellar reduces memory overhead and query latency by 1-2 orders of magnitude compared to existing methods without compromising retrieval effectiveness.

## Full Text


<!-- PDF content starts -->

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING, VOL. XX, NO. XX, XXX XXXX 1
Stellar: Scalable Multimodal Document Retrieval
for Natural Language Queries
Yuxiang Guo, Zhonghao Hu, Yuren Mao, Yuhang Liu, Congcong Ge, Xiaolu Zhang,
Jun Zhou, and Yunjun Gao,Senior Member, IEEE
Abstract—Multimodal document retrieval—selecting the most
relevant multimodal document from a large corpus to answer
a natural language query—plays an essential role in Retrieval-
Augmented Generation (RAG) systems. State-of-the-art methods
represent each document and query with multiple token-level
embeddings and use late interaction to achieve high effectiveness.
However, such multi-vector representations incur substantial
memory overhead during retrieval, leading to poor scalability
and hindering real-world deployment. In this paper, we present
STELLAR, a scalable multimodal document retrieval framework
that stores token-level document embeddings on disk and loads
only a small set of candidate embeddings into memory for
late interaction. STELLARcomprises two key components: (i)
Lexical Representation-based Filtering (LRF), which fine-tunes
a Multimodal Large Language Model (MLLM) as a sparse
encoder to produce high-quality lexical representations, enabling
efficient and effective document filtering to significantly reduce
the candidate set; (ii) Efficient Disk-backed Late Interaction
(DLI), which designs an on-disk token embedding storage layout
guided by a balanced clustering algorithm, and dynamically loads
only the necessary token embeddings into memory using a simple
yet effective cost model. Extensive experiments on four real-
world benchmarks and a newly presented large-scale dataset
demonstrate that STELLARreduces memory overhead and query
latency by 1-2 orders of magnitude compared to existing methods
without compromising retrieval effectiveness.
Index Terms—Multimodal Document Retrieval, Document Fil-
tering, Late Interaction, Scalability
I. INTRODUCTION
Large language models (LLMs) [1], [2] have demonstrated
strong performance across diverse tasks, yet hallucination
remains a major challenge for their reliable deployment in
domain-specific applications [3]. Retrieval-Augmented Gen-
eration (RAG) [4] mitigates this issue by enabling LLMs
to incorporate external knowledge through information re-
trieval. While most existing RAG systems operate over text
corpora [5]–[7], in practice, knowledge is often embedded in
multimodal documents that integrate text, charts, and other
visual elements [8], [9]. This motivates the need for mul-
timodal document retrieval to locate relevant documents to
answer users’ natural language (NL) queries.
As an example shown in Figure 1, given a user query,
multimodal document retrieval aims to identify the document
d2from the large corpus, as it contains the information about
Y . Guo, Z. Hu, Y . Mao, Y . Liu, C. Ge, and Y . Gao are with Zhejiang
University, Hangzhou 310027, China (e-mail: guoyx@zju.edu.cn;
zhonghao.hu@zju.edu.cn; yuren.mao@zju.edu.cn; lyh65535@zju.edu.cn;
gcc@zju.edu.cn; gaoyj@zju.edu.cn).
X. Zhang and J. Zhou are with Ant Group, Hangzhou 310013, China (e-mail:
yueyin.zxl@antfin.com; jun.zhoujun@antfin.com)
Quer y: Ho w man y  of emplo y ees ar e t her e in t he U .S. Stat e 
Depar tment, and which cat egor y has ?cat egories
t he most emplo y ees
...Stat e Depar tment Emplo y ees
(in t housand)Multimodal Document CorpusR etrie v e r ele v ant document s
d 1d 2d 3F or eign Ser viceCivil Ser viceLocally Emplo y ed StaffF amily Members
Fig. 1. Multimodal document retrieval for NL queries.
“employee categories” and “employee counts” represented by
legends and bar charts, which can be used to answer the
query. Onced 2is retrieved, it can be provided to a generator
to generate a response. The quality of downstream question
answering depends on the retrieval step.
Traditionally, multimodal document retrieval relies on Op-
tical Character Recognition (OCR) [10] to extract text, fol-
lowed by standard text-based retrieval methods [11], [12].
Such pipelines often suffer from limited effectiveness due
to OCR errors and the loss of visual information. With the
rapid development of Multimodal Large Language Models
(MLLMs) [13], recent studies [9], [14] have shifted toward
vision-centric paradigms: each document page is rendered as
an image and directly encoded by MLLMs. As shown in
Figure 2(a), single-vector-based methods [9], [15] encode each
document page into a single dense vector; however, such a
global representation often fails to capture the fine-grained
semantics of complex documents. To address this, multi-vector
methods [14], [16] encode both queries and documents into
multiple token-level embeddings, as shown in Figure 2(b),
and further employ the late interaction mechanism [17], where
each query token interacts with all document tokens to com-
pute the overall query-document relevance.
Although multi-vector methods have achieved state-of-the-
art (SOTA) retrieval effectiveness [14], [16], they suffer from
poor scalability. At retrieval time, they require all token-
level document embeddings to reside in memory and perform
late interaction, incurring substantial computation and memory
overhead. For example, the SOTA method ColPali [14] is es-
timated to require over 800 GB of memory to store token em-
beddings for a 2.4-million-document corpus [18]. This resultsarXiv:2606.19960v1  [cs.IR]  18 Jun 2026

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING, VOL. XX, NO. XX, XXX XXXX 2
MLLM MLLMS Score
Query Document Page
(b)  Multi -vector method...
......
...Text token embedding
Visual token embedding
MLLM
Query MLLMScore
Document Page
(a) Single -vector methodQuery embedding
Document embedding
MaxSim MaxSim MaxSim
Fig. 2. An illustration of the single-vector-based method and multi-vector
method with late interaction.
in considerable deployment costs, exceeding $8,500 per month
on a standard cloud instance (e.g., AWS r7iz.32xlarge1). Al-
though quantization-based approximate search methods [19],
[20] can reduce the memory footprint of token-level embed-
dings by compressing them offline and keeping the quantized
representations resident in memory during retrieval, this design
does not fundamentally address the scalability challenge. First,
the offline compression stage requires loading all original
token embeddings into memory, resulting in the same peak
memory footprint as the uncompressed representations. Sec-
ond, retrieval-time memory usage still grows rapidly with
corpus size, incurring substantial memory footprints at scale.
This motivates a paradigm shift: instead of compressing token
embeddings to fit in memory,we store original token embed-
dings on inexpensive disk and selectively load only a small
subset into memory for late interaction. However, introducing
disk-backed late interaction raises two new challenges.
Challenge I:How to design an effective and lightweight filter?
In a disk-based retrieval paradigm, loading all documents’
token embeddings from disk into memory for late interaction
is impractical, as it would undermine the purpose of memory
saving. Therefore, an effective filtering stage is essential to
identify a small candidate set. However, designing such a filter
is non-trivial, as it must balance efficiency and recall: the
filter should be lightweight to avoid introducing substantial
memory overhead or latency to the overall framework, yet
powerful enough to ensure that relevant candidates are not
prematurely discarded. Existing solutions typically employ
dense representations to retrieve top-ranked candidates [15],
[19]. While effective, the high-dimensional dense embeddings
incur substantial storage overhead and become a major bot-
tleneck for end-to-end retrieval at large corpus scales (see
Section VII-D).
Challenge II:How to reduce disk-to-memory data loading
time?After candidate documents are identified, their token-
level embeddings must be loaded into memory for late
interaction, making disk-to-memory data loading a critical
efficiency bottleneck due to I/O latency. This challenge is
twofold, involving both offline data organization and online
loading strategy. First, without careful layout design, token
embeddings of semantically related documents are stored non-
contiguously across disk blocks, resulting in expensive random
I/O. Second, even with an optimized storage layout, the
1https://instances.vantage.sh/aws/ec2/r7iz.32xlarge?currency=USDsystem faces a dynamic online loading dilemma: sequentially
loading entire block embeddings improves I/O efficiency but
introduces overhead from non-candidate embeddings, while
selectively loading only candidate embeddings minimizes data
transfer but triggers random I/O. Consequently, a one-size-fits-
all strategy would inevitably lead to suboptimal performance.
To surmount these challenges, we present STELLAR, a
scalable ret rievalframework via efficient fil tering and disk-
backed la te inter action. To addressChallenge I, we design a
siamese sparse dual-encoder based on a pre-trained MLLM,
repurposing its next-token prediction head as a shared pro-
jection layer that maps both documents and queries into
the model’s vocabulary space, followed by a sparsification
step that retains only the most informative dimensions. The
resulting sparse representations support effective and efficient
document filtering via an in-memory inverted index. To tackle
Challenge II, we propose a balanced clustering algorithm that
groups semantically related documents based on the learned
sparse lexical representations. The token embeddings of all
documents within each cluster are stored contiguously in a sin-
gle disk block, preserving semantic locality while maintaining
balanced block sizes for stable I/O performance. Building on
this layout, we further introduce a cost-aware loading strategy
guided by a simple yet effective cost model, which adaptively
decides whether to load entire blocks or only the required
token embeddings, thereby optimizing data loading efficiency.
Our main contributions are as follows:
•Scalable retrieval framework.We introduce STELLAR,
a scalable framework for multimodal document retrieval
that addresses the memory-intensive limitations of existing
approaches through a representation-storage co-design.
•MLLM-based lexical representation.We propose a novel
lexical representation learning method that repurposes the
pre-trained prediction head of MLLMs to project complex
multimodal document pages into a sparse lexical space,
enabling effective and efficient document filtering.
•Efficient disk-to-memory loading.We design a balanced
clustering-based on-disk layout, and a cost-aware data load-
ing strategy guided by a simple yet effective cost model,
substantially reducing I/O overhead and enabling low-
latency retrieval.
•Large-scale benchmark.Since no public large-scale multi-
modal document retrieval dataset is available, we construct
LargeDoc, a large-scale benchmark dataset, and publicly
release it for future research.
•Extensive experiments.Our extensive experiments on five
real-world datasets show that STELLARreduces memory us-
age and query latency by 1-2 orders of magnitude compared
with existing multi-vector methods, while maintaining state-
of-the-art effectiveness.
Roadmap.The remainder of this paper is organized as fol-
lows. Section II reviews related works. Section III provides the
preliminaries. Section IV overviews the framework STELLAR.
Section V and Section VI introduces two key components of
STELLAR. Section VII reports experimental results and our
findings. Section VIII concludes this paper.

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING, VOL. XX, NO. XX, XXX XXXX 3
II. RELATEDWORK
A. Multimodal Document Retrieval
Multimodal retrieval [21]–[24] is a foundational task aimed
at bridging the semantic gap between disparate modali-
ties. Early paradigms predominantly focus on cross-modal
alignment, where models like CLIP [25], BLIP [26], and
SigLIP [27] leverage contrastive learning [28] to map images
and short text into a unified latent space. While effective
for cross-modal matching, these methods often falter when
applied to multimodal document retrieval. Unlike cross-modal
retrieval (e.g., retrieve images by text), each multimodal doc-
ument page encompasses elements of multiple modalities and
complex layout structures. As highlighted in [15], CLIP-style
encoders are suboptimal for complex multimodal document
representation. With the development of powerful Multimodal
Large Language Models (MLLMs), recent studies [9], [14],
[15] try to render each document page as an image and
leverage MLLMs to generate high-quality embeddings for
dense similarity search, and achieve promising effectiveness.
B. Multi-Vector Late Interaction
Single-vector representations are often insufficient for cap-
turing the rich semantics of complex documents. To address
this, multi-vector late-interaction methods [17] are proposed
to encode both documents and queries as sets of token-
level embeddings, and compute relevance between token
pairs. Representative approaches include ColPali [14] for
multimodal document retrieval and M3DocRAG [16], which
adopts ColPali as its retriever. Despite their effectiveness,
such approaches incur substantial computational and memory
overhead, as they require storing and comparing large numbers
of token embeddings.
While quantization-based approximate search methods [19],
[20] can compress token embeddings into centroids and
quantized residuals, they do not fundamentally resolve the
scalability challenge and typically trade retrieval effectiveness
for efficiency. In contrast, widely adopted approximate nearest
neighbor (ANN) search techniques, including in-memory ap-
proaches such as HNSW [29], and disk-based methods such
as DiskANN [30] and Starling [31], are primarily designed
for single-vector retrieval and are not directly compatible with
multi-vector late interaction. Although some existing vector
search systems (e.g., Milvus [32]), extend single-vector re-
trieval to multi-vector settings, they merely aggregate similar-
ity scores from isolated single-vector searches—a design that
fundamentally diverges from the token-level late interaction
paradigm prevalent in multimodal document retrieval.
C. Sparse Retrieval
Sparse retrieval traditionally relies on literal term matching
within a shared vocabulary space to enable efficient candi-
date filtering via inverted indexes. Early approaches such as
BM25 [11] calculate relevance scores based on term-frequency
and inverse document-frequency statistics. While highly effi-
cient, these methods suffer from the lexical mismatch as they
cannot capture semantic synonyms. To address this, recentneural sparse retrievers like SPLADE [33] leverage BERT [34]
to jointly learn term expansion and term weighting, sub-
stantially improving retrieval effectiveness while preserving
sparsity. This paradigm has recently been extended to the
multimodal domain. For instance, STAIR [35] and MLSR [36]
map images and text into sparse token spaces for cross-
modal tasks. However, these methods typically use small
encoder-only models and are ill-suited for complex multimodal
document pages. In contrast, STELLARbridges this gap by
introducing the powerful MLLM, and adapts the decoder as a
sparse encoder for document filtering.
III. PRELIMINARIES
In this section, we first provide the problem statement, and
then introduce the backgrounds of multimodal large language
models and late interaction mechanism.
A. Problem Statement
Letddenote a multimodal document page, as exemplified
in Figure 1, which comprises both textual content and visual
elements. Following previous studies [9], [14], we represent
each document page as a rendered image, referred to as a
document image or document screenshot. This rendered page
image serves as the atomic unit of retrieval. In this paper,
we use the terms “document” and “page” interchangeably. A
multimodal document corpusD={d 1, . . . , d |D|}is a set of
such documents.
Given a multimodal document corpusDand a natural
language (NL) queryq, multimodal document retrieval aims
to identify query-relevant documents that contain the answer
to the queryq. This relevance is determined using a similarity
metricsim(q, d). Following previous studies [15], [37], we
focus on single-document retrieval. We leave multi-document
retrieval, in which answers are distributed across various
multimodal documents, as future work.
B. Multimodal Large Language Models
An MLLM typically consists of a vision encoder and an
LLM decoder with a pre-defined vocabularyV. Given a
visual input (e..g, a document image), the MLLM first divides
it into a sequence of fixed-size patches. These patches are
processed by the vision encoder to extract visual features,
which are then mapped into the language embedding space via
a learnable adapter to produce visual tokens:{tvis
1, . . . ,tvis
m},
wheremis the number of patches, which depends on the
resolution of the input image and the patch size. For a textual
input, such as a queryq, the LLM’s tokenizer first yields a
sequence of discrete token indices, which are subsequently
transformed into textual tokens:{ttext
1, . . . ,ttext
n}, wherenis
the number of tokens. The sequences from both modalities
are concatenated to form a unified input sequence, which is
sent to the LLM decoder to compute hidden states. Finally, the
LLM autoregressively generates textual output by projecting
the hidden states through a language modeling headf headto
compute a probability distribution over the vocabularyV.

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING, VOL. XX, NO. XX, XXX XXXX 4
in memoryQuery qSparse
Representation
Sparse
RepresentationFilteringSparse
vectors... ... Inverted
Indexing
How many
catogries...
Sparse
vectord3 1
2 d3,d6
d1,d5 |   |
...
Top-k 1 docs...
d1 d|k |1Docs DMulti-Vector
Representation... Cluster-based 
Storage Layout
Cost-aware
Loading...
Clusters on disk Multi-vec
of D
Multi-Vector
Representation𝒱 
Late Interactio n
 & Score Fusi o
 multi-vec
of D' in memoryMulti-vec
of q
query  qMLLM-based Lexical 
Representation
Document
Filtering... ... Inverted 
Index
How many
categories ...
candidates   Dk1...corpus  DMulti-Vector
Representation... Cluster-based
 On-Disk Storage
Cost-Aware
Loadingblocks on disk multi-vecs of D
Multi-Vector
RepresentationLate Interaction
& Score Fusion  
multi-vec of  qmulti-vecs of Dk1 
reranked docsinverted index
(a) Lexical Representation-based  Filtering (b) Efficient Disk-backed Late Interactionoffline
onlineoffline
onlined7
in memory1
2
...
|   |d1d3
d4d6
d1
d2d3
......
𝒱 
MLLM-based Lexical 
Representationsparse vecs of D 
sparse vec of q 
Fig. 3. Overview of STELLAR. (a) Lexical Representation-based Filtering enables efficient and effective document filtering. (b) Efficient Disk-backed Late
Interaction supports low-cost multi-vector similarity computation and sparse-dense score fusion.
C. Late Interaction Mechanism
To capture fine-grained semantic alignments between
queries and documents, multi-vector representation learn-
ing [14] avoids compressing an entire document into a single
vector. Instead, it encodes each documentdand queryqinto a
sequence of token-level embeddings, denoted as{ed
1, . . . ,ed
m}
and{eq
1, . . . ,eq
n}, respectively. Here,mandnrepresent the
number of visual tokens (e.g., image patches) in documentd
and text tokens in queryq. Based on these representations, the
late interaction mechanism [17] employs aMaxSimoperator
to compute the relevance score. This mechanism allows each
query token to independently align with the most semantically
similar part of the document, thereby preserving fine-grained
semantic relations. The similarity score is formally defined as:
score mul(q, d) =nX
i=1mmax
j=1 
eq
i·ed
j
(1)
whereeq
ianded
jdenote theL 2-normalized embeddings of the
i-th query token and thej-th document token, respectively;n
andmare the corresponding token counts; and(·)denotes the
inner product.
IV. OVERVIEW OFSTELLAR
Figure 3 illustrates the overview of STELLAR, which com-
prises two components: lexical representation-based filtering
(LRF), and efficient disk-backed late interaction (DLI).
Lexical Representation-based Filtering.During theoffline
stage, each documentdin the corpusDis encoded into a
sparse vector using our proposed MLLM-based sparse lexi-
cal representation method. Each sparse vector has a dimen-
sionality equal to the MLLM’s vocabulary size|V|, where
most dimensions are zero. These sparse vectors are indexed
via an in-memory inverted index that maps each vocabulary
dimension (of size|V|) to documents with non-zero values
along that dimension, enabling efficient document filtering. In
theonlinephase,the user queryqis encoded into a sparse
vector. The top-k 1candidate documentsDk1⊂Dare filtered
by computing sparse similarity scores between the query and
document sparse embeddings via the inverted index.
Efficient Disk-backed Late Interaction.During theoffline
stage, each documentd∈Dis encoded into multiple token
embeddings. These token embeddings are stored on disk andorganized using the proposed balanced clustering algorithm. In
theonlinephase, we design a cost-aware loading strategy to
determine whether to load the entire block or specific token
embeddings of the candidate documentsDk1into memory.
Meanwhile, the user queryqis encoded into a set of token-
level embeddings, which interact with document token em-
beddings via late interaction to compute multi-vector dense
scores. These dense scores are then fused with the first-stage
sparse scores to obtain hybrid scores, which are used to rank
the candidate documentsDk1as the final results.
V. LEXICALREPRESENTATION-BASEDFILTERING
In this section, we first present the MLLM-based lexical
representation method. Then, we describe how to index the
sparse vectors and perform document filtering.
A. MLLM-based Lexical Representation
We design a siamese dual-encoder built upon an MLLM
to encode both documents and queries into a unified sparse
lexical space, which serves as an efficient filtering mechanism
for scalable multimodal document retrieval.
Unified Lexical Projection.We project each documentd∈D
and text queryqinto a shared vocabulary space using the
MLLM. Specifically, documents and queries are independently
processed to extract visual and textual tokens, which are then
encoded into last-layer hidden states. Each hidden state is
projected into the model’s vocabulary spaceV, yielding a
|V|-dimensional lexical vector. For documents, this projection
translates input visual tokens into semantically relevant lexical
tokens inV; for text queries, it naturally enables semantic
expansion by activating related vocabulary tokens. To support
this unified lexical mapping, we repurpose the MLLM’s pre-
trained language modeling headf headas a lexical projection
layer. Although originally designed for next-token prediction,
fheadlearns to map hidden states into a shared vocabulary
space during large-scale vision-language pretraining. We there-
fore initialize the projection layer with the pre-trainedf headto
fully exploit its capability, and further fine-tune it to align with
our retrieval objective.
Formally, for each tokent i(i= 1, . . . , m) in a multimodal
documentd, leth idenote its last-layer hidden state. The
corresponding lexical vectorw iis computed as:
wi=f head(hi) = [w i1, wi2, . . . , w i|V|]∈R|V|(2)

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING, VOL. XX, NO. XX, XXX XXXX 5
wherew ijrepresents the weight assigned to thej-th vocabu-
lary token in representing the tokent i. A high value ofw ij
indicates strong semantic relevance between the tokent iand
thej-th vocabulary token, while a negative value suggests
irrelevance and can be set to zero. To implement this, we apply
a ReLU function to each vectorw i, eliminating all negative
values. Finally, max pooling is applied across tokens for each
dimensionjto extract salient features, yielding the final sparse
representationenc spa(d) = [z 1, z2, . . . , z |V|]∈R|V|, where
eachz jis calculated as follows:
zj= max
ilog (1 + ReLU (w ij)), i∈ {1,2, . . . , m}(3)
The sparse lexical representation of a text queryq, denoted
asenc spa(q), is constructed analogously.
MLLM Fine-tuning.High-quality lexical representations are
expected to exhibit high similarity between a query and its
relevant documents, while maintaining low similarity with
irrelevant ones. This enables effective document filtering via
similarity search. To achieve this, we employ contrastive
learning [28] to fine-tune the MLLM as a sparse encoder
enc spa, using in-batch negatives:
LInfoNCE =−1
NNX
i=1logexp
sim(q i, d+
i)/τ
exp
sim(q i, d+
i)/τ
+P
j̸=iexp
sim(q i, d−
j)/τ
(4)
wheresim(q, d) =enc spa(q)·enc spa(d)denotes the similarity
between the embeddings of the anchor queryqand either
the positive (relevant) documentd+or negative (irrelevant)
documentd−;Ndenotes the mini-batch size, andτis the
temperature parameter.
To ensure vector sparsity and improve retrieval efficiency,
we incorporate a FLOPs-based regularization loss [33] to
control the sparsity of the output document representations:
Ld
FLOPS =|V|X
j=1 
1
NNX
i=1zj(di)!2
(5)
wherez j(di)denotes thej-th component of the sparse rep-
resentation for documentd i, which is computed as shown in
Equation (3). The regularization loss for query representation
is denoted asLq
FLOPS with a similar definition toLd
FLOPS .
The overall loss is as follows:
L=L InfoNCE +λ1Ld
FLOPS +λ2Lq
FLOPS (6)
whereλ 1andλ 2control the strength of the FLOPs regulariza-
tion for the document and query vectors, respectively. During
fine-tuning, LoRA [38] is applied to all linear layers in the
Transformer blocks and the language modeling headf head.
B. Inverted Index and Document Filtering
Offline Indexing.We use the fine-tuned MLLM as the sparse
encoder to generate lexical representations for all the docu-
mentsd∈D. Then, we construct an inverted index over
sparse dimensions. Specifically, for each documentd i∈D,
we record a posting entry(i, z j(di))for each dimension
j∈ I di, whereI didenotes the set of non-zero dimensions
in its sparse representation,iis the documentID, andz j(di)is the corresponding sparse weight. Posting entries are then
grouped by dimension to form the inverted index:
Index[j] ={(i, z j(di))|z j(di)̸= 0}, j∈ {1,2, . . . ,|V|}
Online Filtering.At query time, we compute the sparse query
representationenc spa(q)and identify its non-zero dimensions
Iq. Instead of scanning all documents, we traverse only the
posting lists corresponding to dimensions inI q. For eachj∈
Iq, we accessIndex[j]and accumulate partial scores for the
associated documents. This process effectively computes the
sparse relevance score defined as:
score spa(q, d) =X
j∈Iq∩Idzj(q)·z j(d),(7)
whereI ddenotes the set of non-zero dimensions of document
d.
Based on the sparse scores, we filter the top-k 1(k1≪ |D|)
candidate documentsDk1⊂Dand discard the remaining
documents from subsequent late-interaction computations.
VI. DISK-BACKEDLATEINTERACTION
In this section, we first review token-level multi-vector
representation learning, then introduce the cluster-aware on-
disk layout design, followed by the cost-aware data loading
strategy and score fusion mechanism.
A. Multi-Vector Representation
To obtain token-level embeddings, we reuse the MLLM
backbone in Section V-A and fine-tune it as a dense multi-
vector encoder. Given a multimodal document or a query,
we extract the MLLM’s last-layer hidden stateh iof each
visual or text token, and project it into a lower-dimensional
embedding space via a linear transformation:e i=Wh i+b,
whereWandbdenote the projection matrix and bias vector,
respectively. A document and a query are represented as sets
of token embeddings, denoted byenc mul(d) ={ed
1, . . . ,ed
m}
andenc mul(q) ={eq
1, . . . ,eq
n}, wheremandndenote the
numbers of visual tokens in documentdand text tokens in
queryq, respectively.
We adopt a pairwise contrastive loss [14] to fine-tune the
MLLM as an effective multi-vector dense encoder:
Lpw=1
NNX
i=1log 
1 + exp 
s−
i−s+
i
(8)
whereNis the number of training examples
in each mini-batch,s+
i = score mul(qi, d+
i), and
s−
i= max j̸=iscore mul(qi, d−
j), both computed using
Equation (1). We adopt LoRA for fine-tuning.
B. Cluster-based On-Disk Storage
We store all learned token-level embeddings of the doc-
ument corpus on disk rather than in memory, which incurs
disk-to-memory loading overhead during online retrieval due
to random I/O [39]. At query time, the first-stage (LRF)
dynamically filters a set of candidate documentsDk1. If
the token embeddings of these filtered retrieved candidate

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING, VOL. XX, NO. XX, XXX XXXX 6
Algorithm 1:Balanced Clustering Algorithm (BCA)
Input:Sparse vectorsE D, expected sizes exp, minimum size
smin
Output:Balanced clustersC res
1K← ⌈|D|/s exp⌉,C tmp← ∅
2C init←k-means(E D, K)
3foreach clusterc∈ C initdo
4if|c|> s expthen
5C tmp← C tmp∪RecursiveSplit(c, s exp)
6else
7C tmp← C tmp∪ {c}
8C small← {c∈ C tmp| |c|< s min}
9C surv← C tmp\ C small
10foreach documentd∈S
c∈C smallcdo
11c target←argmaxc∈C survsim(d,centroid(c))
12Movedtoc target
13returnC res← C surv
14ProcedureRecursiveSplit(c,s exp):
15n k← ⌈|c|/s exp⌉,C out← ∅
16C sub←k-means(c, n k)
17foreachs∈ C subdo
18if|s|> s expthen
Cout← C out∪RecursiveSplit(s, s exp)
19elseC out← C out∪ {s}
20returnC out
documents are stored contiguously on disk, the cost of random
I/O can be substantially reduced. Although the candidate
set varies across queries, the following observation enables
such an on-disk organization in practice. Since the first-
stage filtering is based on learned lexical representations, the
retrieved candidate documentsDk1tend to exhibit similar
sparse embeddings. This property motivates clustering doc-
uments by their lexical representations and storing the token
embeddings of documents within each cluster contiguously
as ablock. However, naive clustering often produces highly
imbalanced block sizes, leading to unpredictable I/O behavior
and inefficient data loading (see Section VII-E). To address
this, we design a balanced document clustering algorithm to
organize token embeddings on disk.
Balanced Clustering Algorithm (BCA).The algorithm pro-
ceeds in three stages, with pseudocode shown in Algorithm 1.
Initial Partitioning. BCAfirst sets the initial number of clus-
ters toK=⌈|D|/s exp⌉and performs a preliminaryk-means
partitioning (lines 1–2), wheres expdenotes the target cluster
capacity.
Recursive Splitting. Then,BCAidentifies clusters whose sizes
exceed the capacitys expand triggers a recursive splitting
strategy (lines 3–5). Within theRecursiveSplitprocedure
(lines 14–20), the local cluster numbern kis dynamically
recomputed based on the current cluster size. This hierarchical
refinement continues until all sub-clusters are size-compliant,
ensuring that no single disk access loads an oversized block
that would stall I/O.
Locality-Aware Reassignment. To avoid overly small clusters,
we define a minimum cluster sizes min. Clusters with fewer
thans mindocuments are classified asC small and dissolved
(lines 8–9). Their constituent documents are then reassignedto the clusters in the surviving setC survwhose centroids
exhibit the highest similarity to the document embeddings
(lines 10–12). This reassignment ensures that each final cluster
inC resmaintains sufficient document density, maximizing the
effective utilization of each disk access and reducing overall
random I/O overhead. After balanced clustering, the token-
level embeddings of documents belonging to the same cluster
are stored contiguously on disk, forming a storage blockb.
Two-level Index.To efficiently manage and load on-disk token
embeddings, we design two lightweight in-memory indexes.
Document-level index :IDd7→(ID b,count,offset,len),
whereID dis a globally unique document identifier;ID bde-
notes the block containing the document’s token embeddings;
countspecifies the number of token vectors for documentd;
offsetindicates the starting byte offset of these embeddings
within the block; andlendenotes their total byte length.
Block-level index :IDb7→(doc_list,total), whereID b
denotes the block identifier,doc_listis the list of document
IDs contained in the block, andtotalindicates the number
of vectors in the block.
C. Cost-Aware Loading
During the online stage, we aim to fetch the token em-
beddings of candidate documentsDk1={d 1, . . . , d k1}from
disk. We first identify the set ofhit blocksB∗={b∗
i}, each
containing embeddings of at least one candidate document.
For each hit blockb∗
i, we consider two data loading modes:
(i) Full Block Loading (FBL) : This mode leverages high se-
quential I/O throughput by loading the entire block into
memory in a single continuous read. Subsequently, the block-
level index is used to locate and retain only the vector slices
corresponding to the candidate documents.
(ii) Specific Vector Loading (SVL) : This mode eliminates the
overhead of irrelevant data loading by utilizing the document-
level index to directly locate and load embeddings of candidate
documents within a hit block.FBLbenefits from high sequen-
tial I/O throughput but incurs extra overhead from loading
embeddings of non-candidate documents, making it inefficient
when candidates occupy only a small fraction of a block.SVL
is efficient when the hit block contains only a small number of
candidate documents, but its efficiency is constrained by the
low throughput of non-contiguous random disk accesses. The
key challenge is thus to determine the optimal loading model
for each hit block during online stage.
We design a cost-aware loading strategy that selects between
FBLandSVLbased on an estimated I/O cost. The estimation
is guided by a simple yet effective cost model, whose key
parameters include the vector dimensionV dim, the byte size
per floatB float, and the disk’s sequential and random read
rates, denoted byR seqandR rand, respectively. To measureR seq,
we sequentially read a temporary binary file (e.g., 1 GB) and
compute the throughput as the total bytes read divided by the
elapsed time. To estimateR rand, we perform a large number
of small random reads (e.g., 10,000 reads of 100 KB) from
the same file and compute the average throughput. For each
hit blockb∗
i∈ B∗, we obtain the total number of vectors
Ni,total in the block and the number of required vectorsN i,req

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING, VOL. XX, NO. XX, XXX XXXX 7
for the candidate documents using the block-level index. We
then estimate the loading time forFBLandSVLusing the
following cost model, and choose the strategy with the lower
estimated cost.
TFBL=Ni,total×V dim×B float
Rseq(9)
TSVL=Ni,req×V dim×B float
Rrand(10)
Discussion.AlthoughFBLincurs higher per-iteration memory
usage in Algorithm 1, it does not affect STELLAR’s peak
memory footprint. Specifically, our objective is to load token
embeddings fork 1documents. UnderFBL, approximately
sexpdocument’s embeddings are loaded per block, wheres exp
denotes the expected cluster size defined in Algorithm 1.
Crucially,s expis configured to be much smaller thank 1;
otherwise, loading just a single block would exceed the total
number of required documents, resulting in inefficient memory
usage. Therefore, the peak memory usage is dominated by
the late interaction stage rather than the data loading stage.
The primary difference in memory usage betweenFBLand
SVLarises from index storage:FBLmaintains both block-
level and document-level indexes in memory, whileSVLonly
requires the document-level index. However, these indexes are
lightweight and introduce negligible memory overhead (see
Section VII-E). Consequently, memory usage is excluded from
our cost model.
D. Late Interaction and Score Fusion
After disk-to-memory data loading, we obtain all the token
embeddings of candidate documentsDk1. The user query is
also encoded into multiple token embeddings, followed by late
interaction with each documentd i∈Dk1using Equation (1)
to obtain the multi-vector dense scoresscore mul(q, d i).
Instead of relying solely on dense scores to rank candidate
documents, we fuse them with the sparse scorescore spa(q, d i)
obtained from the first-stage filtering, as the two scores pro-
vide complementary retrieval signals [40]. The fused score is
defined below:
score fus(q, di) =α·Z(score spa(q, di)) +Z(score mul(q, di))(11)
whereαis a coefficient between 0 and 1, and the function
Z(·)denotes thez−normalization, i.e.,Z(x) =x−µ
δwhereµ
is the mean value andδis the standard deviation. We use the
fusedscore fusto rank candidate documents inDk1to obtain
the final results.
VII. EXPERIMENTS
In this section, we evaluate STELLARon four existing
benchmark datasets and a newly proposed large-scale dataset
LargeDoc. The code and datasets are publicly available at
https://github.com/ZJU-DAILY/Stellar.
A. Experimental Setup
Datasets.We adopt four existing benchmark datasets and
propose a large-scale dataset. The datasets with statistics are
summarized in Table I.TABLE I
STATISTICS OF THE DATASETS USED IN EXPERIMENTS.
Scale Dataset Document Type Train Evaluation
#(q, d)#q#d
SmallDocVQA Industrial Docs 10,624 591 741
InfoVQA Infographics 17,664 718 459
MediumArXivQA ArXiv Figures 25,856 816 8,066
PlotQA Scientific Plots 56,192 863 9,593
Large LargeDoc Mixed — 94 400,000
Existing benchmark datasets. We adopt four widely-used
benchmark datasets in document retrieval and open-domain
QA tasks [9]: (1)DocVQA[41], consisting of scanned
industrial documents with dense textual layouts, complex
forms, and tabular structures; (2)InfoVQA[42], composed
of visually-rich infographics with complex spatial layouts; (3)
ArXivQA[43], containing scientific figures extracted from
academic papers; and (4)PlotQA[44], comprising scientific
plots with structured axes and markers.
Large-scale dataset. To the best of our knowledge, there is no
publicly available large-scale dataset for multimodal document
retrieval. To fill this gap, we constructLargeDoc, a large-scale
benchmark consisting of 400,000 multimodal documents and
94 test queries, each paired with a ground-truth document.
LargeDoc is constructed based on Docmatix [18], which
contains 2.4 million PDFs spanning diverse types such as
papers, forms, charts, and slides, and 9.5 million questions
paired with related documents. The construction process com-
prises three steps. (1) Query selection. Since many Docmatix
questions are designed for closed-domain QA rather than
retrieval (e.g., “What is this page about?”), we manually curate
94 high-quality query-document pairs suitable for retrieval
evaluation. (2) Corpus scaling. To emulate a realistic large-
scale retrieval setting, we randomly sample over 400,000
documents from Docmatix and merge the 94 ground-truth
documents into this pool. (3) False-negative cleaning. Large-
scale retrieval benchmarks often suffer from false negatives
— unlabeled documents that also satisfy a query. To mitigate
this issue, we apply state-of-the-art text-based and vision-
based retrievers (gte-Qwen2, DSE, and ColPali) to retrieve
the top-100 candidates for each query, and take the union
of all retrieved candidates. These candidates are then verified
by GPT-4o, followed by manual validation from three post-
graduate students. Any document confirmed by at least two
annotators as a valid alternative to the ground-truth document
is removed from the pool. Finally, we remove excess irrelevant
documents to limit the corpus size to 400,000 documents.
Baselines.We compare STELLARwith three categories of
multimodal document retrieval methods. For fairness, all meth-
ods performexact similarity searchexcept forQColPali.
Text-centric methods. These methods extract text from doc-
ument pages using PP-OCR [45] and perform text retrieval
using text retrieval methods.
•BM25[11]: A classic sparse retrieval method that evalu-
ates document relevance by measuring term frequency and
statistical significance within the collection.
•BGE-Large[12]: A powerful dense encoder that maps
textual content into a high-dimensional vector space to

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING, VOL. XX, NO. XX, XXX XXXX 8
TABLE II
RETRIEVAL PERFORMANCE ACROSS SIX DATASETS,EVALUATED WITHRECALL@10 (R@1), RECALL@10 (R@10),ANDMRR@10 (M@10).
MethodsDocVQA InfoVQA ArXivQA PlotQA LargeDoc
R@1 R@10 M@10 R@1 R@10 M@10 R@1 R@10 M@10 R@1 R@10 M@10 R@1 R@10 M@10
BM25 60.74 86.80 75.27 45.41 82.59 66.94 30.15 54.29 43.65 30.93 76.01 57.28 57.45 70.21 62.77
BGE-Large 45.57 68.19 50.76 58.39 88.16 72.38 28.23 48.65 39.29 28.46 73.12 51.33 60.64 73.40 65.21
gte-Qwen2 59.89 83.25 68.11 64.76 92.83 78.69 32.84 58.14 43.12 31.05 78.66 53.69 62.77 75.53 69.05
DSE 63.96 90.19 72.79 76.46 96.38 84.02 67.28 84.44 72.71 31.87 73.46 44.00 71.28 80.85 73.74
VisRet 61.25 87.64 70.79 75.77 96.10 83.08 64.83 82.72 70.52 30.48 72.89 42.22 68.09 77.66 71.28
Bi-Qwen 61.59 88.49 71.05 75.77 96.10 83.65 64.09 82.84 70.16 26.88 69.29 39.14 67.02 78.72 70.14
ColPali 82.74 97.27 88.26 86.49 98.89 91.61 75.74 88.48 80.02 56.55 87.02 66.27 74.47 86.17 77.30
QColPali 80.03 96.62 86.53 83.71 98.33 89.71 73.65 87.01 78.14 55.50 87.02 65.88 72.34 85.11 75.89
STELLAR(ours) 84.43 97.29 89.05 86.91 99.16 92.10 75.75 88.97 80.24 56.9086.10 66.01 75.53 87.23 79.47
TABLE III
PEAK MEMORY USAGE(MB)AND QUERY LATENCY(MS)OF THE VISION-CENTRIC MULTI-VECTOR METHODS.
MethodsSmall Scale Medium Scale Large Scale
DocVQA InfoVQA ArXivQA PlotQA LargeDoc
memory latency memory latency memory latency memory latency memory latency
ColPali 315.90 73.44 183.76 59.80 3,301.28 649.35 3,765.12 518.49 147,355 42,115
QColPali 40.0045.81 25.60 42.39 398.40 261.59 480.00 302.43 13,699 613
STELLAR(ours) 32.5150.16 38.66 44.73 49.56 42.35 45.27 43.02 988 110
facilitate semantic similarity matching.
•gte-Qwen2[46]: An LLM-based embedding model re-
leased by Alibaba, built upon the sameQwen2-1.5Bback-
bone as the MLLMQwen2-VL-2B.
Vision-centric single-vector methods: These methods treat
each multimodal document page as an image, and encode each
query and document page into a single vector via MLLMs.
•DSE[15]: A method appends a special token to the input
sequence of MLLMs and uses the final hidden state of this
token as dense representations.
•VisRet[9]: A method uses position-weighted mean pooling
over the final hidden states of MLLMs to get dense repre-
sentations
•Bi-Qwen: A method performs mean pooling over the final
hidden states of MLLMs to obtain dense representations.
Vision-centric multi-vector methods: Methods in this category
encode each document page and query into multiple token
embeddings, enabling fine-grained late interaction.
•ColPali[14]: The state-of-the-art multi-vector multimodal
document retrieval method which performs exact late inter-
action of query-document token embeddings [17].
•QColPali: An approximate variant ofColPalithat incorpo-
rates the PLAID [19] method. Each token embedding is
represented by a centroid and a quantized residual vector.
Following PLAID,QColPalifirst filtersk 1candidate docu-
ments using centroid embeddings and then reconstructs the
quantized token embeddings for late interaction.
Evaluation metrics.To evaluate the effectiveness, we follow
the previous study [9] to adopt Recall-at-k (R@k) aggregated
over the test queries, and Mean Reciprocal Rank-at-k (M@k).
We also report peak memory usage and query latency to
evaluate the scalability of the proposed method.
Implementation Details.All vision-centric methods use
Qwen2-VL-2B[47] as the backbone, which consists of aVision Transformer and theQwen2-1.5Blanguage model.
To ensure a fair evaluation, the backbone model is trained
for a single epoch on the same mixed training set, which
combines all benchmarks’ training splits and another corpus
provided by the previous study [9]. We adopt LoRA [38]
for fine-tuning, with rankr= 32. The temperatureτis
set to 1. Following insights from the previous study [33],
we set the FLOPs regularization strengths to 9e-5 and 6e-5
for documents and queries, respectively. We filterk 1= 100
candidate documents during the first stage. In our balanced
clustering algorithm, we empirically set the expected cluster
sizes expto 50 and the minimum cluster sizes minto 3. All
experiments were conducted on a server equipped with an
Intel(R) Xeon(R) Silver 4316 CPU (2.30GHz), 256GB RAM.
The online retrieval process is performed solely on the CPU.
B. Overall Performance
Effectiveness.Table II reports the R@1, R@10, and M@10
of STELLARand other baselines. Results inboldare the best;
underlined are the second. We have several key observations:
•Vision-centric methods consistently outperform text-centric
methods on all benchmarks, which is consistent with the
observations from previous studies [9], [14].
•Wthin the vision-centric category, single-vector methods
underperform multi-vector methods by an average of 15
points in R@1, underscoring the limitation of representing
an entire document with a single global vector.
•Compared with existing multi-vector methods, STELLAR
achieves the best R@1 and consistently high R@10 and
M@10 across all datasets. Its accuracy is comparable to, and
even outperformsColPali, which performs inefficient late
interaction between query and all documents. Its superior
accuracy arises from (1) lexical representation-based filter-
ing (see Section VII-D), and (2) the score fusion mechanism

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING, VOL. XX, NO. XX, XXX XXXX 9
20 40 60 80 100
Latency (ms)DocVQA
InfoVQA
ArxivQA
PlotQA
LargeDocLRF 
DLI 
Fig. 4. Runtime of lexical representation-based filtering (LRF) and disk-
backed late interaction (DLI).
50 100 150 200 250 300
Number of Non-Zero DimensionsDocVQA
InfoVQA
ArxivQA
PlotQA
LargeDocQuery
Doc
Fig. 5. Average number of non-zero dimensions of our sparse representations
(vocabulary dimension|V|= 151,936).
that integrates both sparse and dense signals to rank the
candidate documents (see Section VII-E). In contrast,QCol-
Palireduces memory via quantization, resulting in inevitable
effectiveness degradation.
Memory and Latency.Table III reports the memory us-
age and query latency of vision-centricmulti-vectormethods
across datasets of different scales. We observe that on small-
scale datasets, i.e., DocVQA and InfoVQA,QColPaliachieves
memory and latency reductions comparable to STELLAR.
However, as the corpus size increases, the limitations of the
baselines become apparent. On medium-scale datasets ArX-
ivQA and PlotQA,ColPaliandQColPalirequire 74.5×and
9.3×the memory of STELLAR, respectively. On LargeDoc,
STELLARrequires only0.96 GBof memory, achieving a150×
reduction compared toColPaliand a14×reduction relative
toQColPali.
We conduct an in-depth analysis of the runtime of the
two phases in STELLAR, as illustrated in Figure 4. The first
observation is that the LRF stage exhibits extremely low
latency, attributed to the high sparsity of the learned represen-
tations. Specifically, our learned sparse representation reduces
the original 151,936-dimensional vocabulary-based vectors to
sparse vectors with only around 200 active (non-zero) dimen-
sions on average, achieving a sparsity rate of approximately
99.87%, as shown in Figure 5. This high sparsity not only
minimizes memory usage but also enables the use of inverted
index structures for fast large-scale candidate filtering. The
second observation is the DLI stage’s latency depends on both
the dataset and its scale. Generally speaking, larger datasets
require processing more hit blocks and thus incur higher
latency. Note that DocVQA contains high-resolution document
images, leading to more visual tokens and consequently higher
late-interaction latency.
ColPali QColPali STELLA
0.1 1 2 3 4
Dataset Size (×105)101
100101102Memory (GB)
(a) Memory w.r.t different scales
0.1 1 2 3 4
Dataset Size (×105)101
100101102Latency (s)
 (b) Latency w.r.t different scales
Fig. 6. Scalability study with different dataset size.
DSE QColPali-Filter STELLA-LRF
110 30 50 100
k160708090100Recall
(a) Recall@k 1on DocVQA
110 30 50 100
k160708090100Recall
 (b) Recall@k 1on LargeDoc
DSE QColPali-Filter Stellar-LRF
0.1 1 2 3 4
Dataset Size (×105)036912Memory (GB)
(c) Memory usage on LargeDoc
0.1 1 2 3 4
Dataset Size (×105)00.150.300.450.60Latency (s) (d) Runtime on LargeDoc
Fig. 7. Performance of different filtering methods.
C. Scalability Study
We measure the memory usage and query latency of the
vision-centric multi-vector methods on LargeDoc at varying
scales. For memory usage, as shown in Figure 6(a),ColPali
exhibits the largest memory footprint. As the corpus scale in-
creases, its memory consumption grows most rapidly, resulting
in more than 150 GB when reaching 400K documents. Al-
thoughQColPalicompresses high-dimensional token embed-
dings, its memory usage still grows rapidly with corpus size.
Moreover, it results in memory overflow on our 256 GB RAM
machine when the dataset exceeds 400K documents, primarily
due to the costly compression stage. In contrast, STELLAR
maintains a low memory footprint, reducing memory usage by
1-2orders of magnitude compared withColPaliandQColPali.
Moreover, its memory consumption grows more than10×
more slowly with increasing corpus size thanQColPali, high-
lighting its superior scalability. For query latency, as shown in
Figure 6(b),ColPaliis 2 orders of magnitude slower than the
other two methods due to its exhaustive late interaction across
all documents. In contrast, bothQColPaliand STELLARlimit
late interaction to the top-k 1candidates. However, STELLAR
achieves higher efficiency due to the efficient LRF stage.
D. Effectiveness of LRF
We compare our lexical representation-based filtering
(STELLAR-LRF) with two alternative filtering methods: (i)
DSE, the strongest dense single-vector retriever (see Table II),
which serves as a dense first-stage filter; and (ii)QColPali-

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING, VOL. XX, NO. XX, XXX XXXX 10
TABLE IV
MEMORY USAGE(MB)AND LATENCY(MS)WITH DIFFERENT DOCUMENT CLUSTERING AND DATA LOADING STRATEGIES.
MethodsDocVQA InfoVQA ArXivQA PlotQA LargeDoc
memory latency memory latency memory latency memory latency memory latency
STELLAR 32.5150.16 38.6644.73 49.5642.35 45.2743.02 988 110
NCA 33.11 60.33 38.63 51.06 50.21 55.81 45.81 57.45 1014 133
FBL 34.60 178.11 39.31 189.22 52.51 158.65 45.89 176.59 1002 351
SVL 31.0263.20 38.3560.68 48.6364.47 44.5359.47 989 142
TABLE V
PERFORMANCECOMPARISON OFSTELLAR ANDSTELLAR W/OSCOREFUSION(SF).
MethodsDocVQA InfoVQA ArXivQA PlotQA LargeDoc
R@1 R@10 M@10 R@1 R@10 M@10 R@1 R@10 M@10 R@1 R@10 M@10 R@1 R@10 M@10
STELLAR 84.43 97.29 89.05 86.91 99.16 92.10 75.75 88.97 80.24 56.9086.1066.01 75.5387.2379.47
STELLARw/o SF 82.91 96.95 88.25 86.49 98.89 91.63 75.49 88.73 79.86 56.2086.4465.80 70.2188.3075.69
0.0 0.5 1.0
92.092.593.093.5R@10
(a) Effect ofαon average R@10
0.0 0.5 1.0
81.082.083.084.0M@10
 (b) Effect ofαon average M@10
Fig. 8. Sensitivity analysis ofαin score fusion.
Filter, which filters candidates using centroids obtained by
clustering token-level embeddings.
We show Recall@k 1across differentk 1on a small-scale
dataset and a large-scale dataset in Figure 7(a)-(b); other
datasets exhibit similar trends. Notably, our LRF consistently
achieves higher recall than the two dense-vector-based filtering
methods, demonstrating the effectiveness of lexical represen-
tations in capturing lexical cues often overlooked by dense
embeddings. Whenk 1reaches 100, all three methods attain
stable and high recall. However, LRF is more efficient in
both time and memory. As shown in Figure 7(c)-(d), on the
LargeDoc, LRF exhibits lower memory usage and runtime.
It is worth noting that on the 400K-scale, using denseDSE
for first-stage filtering requires 2.3 GB of memory—more
than twice the end-to-end memory consumption of STELLAR
(0.96 GB; see Table III). This indicates thatdense filtering
can become a major memory bottleneck, underscoring the
necessity of our sparse lexical filtering.
E. Impact of Strategies in DLI
We conduct an ablation study to validate the necessity of
our designs in disk-backed late interaction (DLI), focusing
on: (1) the effect of balanced clustering for on-disk layout
optimization on memory and latency, (2) the effect of the
selective loading strategy, and (3) the effectiveness of score
fusion.
Impact of Clustering Strategy.We replace our balanced doc-
ument clustering algorithm with a naive clustering algorithm
(NCA), which performs clustering without adjusting cluster
sizes. As shown in Table IV,NCAresults in consistently
higher latency than STELLARacross all datasets, highlightingthe importance of well-balanced clusters for efficient disk-to-
memory data loading.
Impact of Loading Strategy.We report the memory usage
and query latency by replacing our cost-aware loading strategy
with (1) full block loading (FBL), and (2) specific vector load-
ing (SVL). Table IV reports the results. The first observation
is that the memory costs are similar across loading strategies,
consistent with our discussion in Section VI-C. The second
observation is that bothFBLandSVLincur higher latency
than our method, highlighting the effectiveness of our cost
model in dynamically selecting the optimal strategy.
Impact of Score Fusion.We remove the score fusion (SF)
component and rank documents solely based on the dense late-
interaction scores following first-stage filtering, as shown in
Table V. The results demonstrate the complementary nature
of sparse and dense signals, highlighting the importance of
score fusion in enhancing accuracy. To balance sparse and
dense scores, we introduce a weighting parameterα. As
shown in Figure 8, sensitivity analysis shows that accuracy
first improves with increasingα, peaks at 0.3, then declines,
offering useful guidance for tuningαin practice.
VIII. CONCLUSION
In this work, we present STELLAR, a scalable framework
for multimodal document retrieval. Instead of compressing
the token embeddings of large document collections to fit
in memory, STELLARstores them on disk. We introduce a
lexical representation method that encodes both multimodal
documents and text queries into a unified lexical space, en-
abling effective and efficient candidate filtering. To mitigate
disk I/O overhead, we propose a balanced clustering–based
storage optimization method, along with a cost-aware on-
line loading strategy for efficient retrieval. Comprehensive
experiments demonstrate that STELLARsubstantially reduces
memory usage and latency while maintaining high retrieval
effectiveness. In the future, we plan to further accelerate disk-
to-memory loading through hardware-aware prefetching or
lightweight embedding caching, and explore multi-document
retrieval in more complex scenarios.

IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING, VOL. XX, NO. XX, XXX XXXX 11
REFERENCES
[1] T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal,
A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-
V osset al., “Language models are few-shot learners,” inNeurIPS, 2020.
[2] A. Hurst, A. Lerer, A. P. Goucher, A. Perelman, A. Ramesh, A. Clark,
A. Ostrow, A. Welihinda, A. Hayes, A. Radfordet al., “Gpt-4o system
card,”arXiv preprint arXiv:2410.21276, 2024.
[3] L. Huang, W. Yu, W. Ma, W. Zhong, Z. Feng, H. Wang, Q. Chen,
W. Peng, X. Feng, B. Qin, and T. Liu, “A survey on hallucination in large
language models: Principles, taxonomy, challenges, and open questions,”
ACM Trans. Inf. Syst., vol. 43, no. 2, pp. 42:1–42:55, 2025.
[4] W. Fan, Y . Ding, L. Ning, S. Wang, H. Li, D. Yin, T. Chua, and Q. Li,
“A survey on RAG meeting llms: Towards retrieval-augmented large
language models,” inSIGKDD, 2024, pp. 6491–6501.
[5] A. Asai, Z. Wu, Y . Wang, A. Sil, and H. Hajishirzi, “Self-rag: Learning
to retrieve, generate, and critique through self-reflection,” inICLR, 2024.
[6] Y . Mao, X. Dong, W. Xu, Y . Gao, B. Wei, and Y . Zhang, “FIT-RAG:
black-box RAG with factual information and token reduction,”ACM
Trans. Inf. Syst., vol. 43, no. 2, pp. 40:1–40:27, 2025.
[7] H. Tan, S. Zhan, H. Lin, H. Zheng, and W. K. Chan, “QAEA-DR: A
unified text augmentation framework for dense retrieval,”IEEE Trans.
Knowl. Data Eng., vol. 37, no. 6, pp. 3669–3683, 2025.
[8] Z. Li, Y . Zhong, C. Chai, Z. Sun, Y . Deng, Y . Yuan, G. Wang, and
L. Cao, “Docdb: A database for unstructured document analysis,”Proc.
VLDB Endow., vol. 18, no. 12, pp. 5387–5390, 2025.
[9] S. Yu, C. Tang, B. Xu, J. Cui, J. Ran, Y . Yan, Z. Liu, S. Wang,
X. Han, Z. Liu, and M. Sun, “Visrag: Vision-based retrieval-augmented
generation on multi-modality documents,” inICLR, 2025.
[10] B. Shi, X. Wang, P. Lyu, C. Yao, and X. Bai, “Robust scene text
recognition with automatic rectification,” inCVPR, 2016, pp. 4168–
4176.
[11] S. E. Robertson and H. Zaragoza, “The probabilistic relevance frame-
work: BM25 and beyond,”Found. Trends Inf. Retr., vol. 3, no. 4, pp.
333–389, 2009.
[12] S. Xiao, Z. Liu, P. Zhang, and N. Muennighoff, “C-pack: Pack-
aged resources to advance general chinese embedding,”CoRR, vol.
abs/2309.07597, 2023.
[13] S. Yin, C. Fu, S. Zhao, K. Li, X. Sun, T. Xu, and E. Chen, “A survey on
multimodal large language models,”arXiv preprint arXiv:2306.13549,
2023.
[14] M. Faysse, H. Sibille, T. Wu, B. Omrani, G. Viaud, C. Hudelot, and
P. Colombo, “Colpali: Efficient document retrieval with vision language
models,” inICLR, 2025.
[15] X. Ma, S. Lin, M. Li, W. Chen, and J. Lin, “Unifying multimodal
retrieval via document screenshot embedding,” inEMNLP, 2024, pp.
6492–6505.
[16] J. Cho, D. Mahata, O. Irsoy, Y . He, and M. Bansal, “M3docrag:
Multi-modal retrieval is what you need for multi-page multi-document
understanding,”arXiv preprint arXiv:2411.04952, 2024.
[17] O. Khattab and M. Zaharia, “Colbert: Efficient and effective passage
search via contextualized late interaction over BERT,” inSIGIR, 2020,
pp. 39–48.
[18] H. Laurenc ¸on, A. Marafioti, V . Sanh, and L. Tronchon, “Building
and better understanding vision-language models: insights and future
directions.” inarXiv preprint arXiv:2408.12637, 2024.
[19] K. Santhanam, O. Khattab, C. Potts, and M. Zaharia, “PLAID: an
efficient engine for late interaction retrieval,” inCIKM, 2022, pp. 1747–
1756.
[20] K. Santhanam, O. Khattab, J. Saad-Falcon, C. Potts, and M. Zaharia,
“Colbertv2: Effective and efficient retrieval via lightweight late interac-
tion,” inNAACL-HLT, 2022, pp. 3715–3734.
[21] W. Wang, B. C. Ooi, X. Yang, D. Zhang, and Y . Zhuang, “Effective
multi-modal retrieval based on stacked auto-encoders,”Proc. VLDB
Endow., vol. 7, no. 8, pp. 649–660, 2014.
[22] M. Erfanian, H. V . Jagadish, and A. Asudeh, “Chameleon: Foundation
models for fairness-aware multi-modal data augmentation to enhance
coverage of minorities,”Proc. VLDB Endow., vol. 17, no. 11, pp. 3470–
3483, 2024.
[23] T. Li, X. Yang, Y . Ke, B. Wang, Y . Liu, and J. Xu, “Alleviating the
inconsistency of multimodal data in cross-modal retrieval,” inICDE,
2024, pp. 4643–4656.
[24] M. Wang, X. Ke, X. Xu, L. Chen, Y . Gao, P. Huang, and R. Zhu,
“MUST: an effective and scalable framework for multimodal search of
target modality,” inICDE, 2024, pp. 4747–4759.[25] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal,
G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever,
“Learning transferable visual models from natural language supervi-
sion,” inICML, 2021, pp. 8748–8763.
[26] J. Li, D. Li, S. Savarese, and S. C. H. Hoi, “BLIP-2: bootstrapping
language-image pre-training with frozen image encoders and large
language models,” inICML, 2023, pp. 19 730–19 742.
[27] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, “Sigmoid loss for
language image pre-training,” inICCV, 2023, pp. 11 941–11 952.
[28] K. He, H. Fan, Y . Wu, S. Xie, and R. Girshick, “Momentum contrast
for unsupervised visual representation learning,” inCVPR, 2020, pp.
9729–9738.
[29] Y . A. Malkov and D. A. Yashunin, “Efficient and robust approxi-
mate nearest neighbor search using hierarchical navigable small world
graphs,”IEEE Trans. Pattern Anal. Mach. Intell., vol. 42, no. 4, pp.
824–836, 2018.
[30] S. J. Subramanya, Devvrit, R. Kadekodi, R. Krishaswamy, and H. V .
Simhadri, “Diskann: fast accurate billion-point nearest neighbor search
on a single node,” inNeurIPS, 2019, pp. 13 766–13 776.
[31] M. Wang, W. Xu, X. Yi, S. Wu, Z. Peng, X. Ke, Y . Gao, X. Xu,
R. Guo, and C. Xie, “Starling: An i/o-efficient disk-resident graph
index framework for high-dimensional vector similarity search on data
segment,”Proc. ACM Manag. Data, vol. 2, no. 1, pp. 14:1–14:27, 2024.
[32] J. Wang, X. Yi, R. Guo, H. Jin, P. Xu, S. Li, X. Wang, X. Guo, C. Li,
X. Xu, K. Yu, Y . Yuan, Y . Zou, J. Long, Y . Cai, Z. Li, Z. Zhang, Y . Mo,
J. Gu, R. Jiang, Y . Wei, and C. Xie, “Milvus: A purpose-built vector
data management system,” inSIGMOD, 2021, pp. 2614–2627.
[33] M. Formal, X. Martinet, N. Thakur, H. Sajjad, C. Lioma, F. Scholer,
M. Worring, M. Bendersky, and F. Diaz, “Splade v2: Sparse lexi-
cal and expansion model for first stage retrieval,” inarXiv preprint
arXiv:2106.05222, 2021.
[34] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of
deep bidirectional transformers for language understanding,” inNAACL-
HLT, 2019, pp. 4171–4186.
[35] C. Chen, B. Zhang, L. Cao, J. Shen, T. Gunter, A. M. Jose, A. Toshev,
Y . Zheng, J. Shlens, R. Pang, and Y . Yang, “Stair: Learning sparse text
and image representation in grounded tokens,” inEMNLP, 2023, pp.
15 079–15 094.
[36] T. Nguyen, M. Hendriksen, A. Yates, and M. de Rijke, “Multimodal
learned sparse retrieval with probabilistic expansion control,” inECIR,
2024, pp. 114–131.
[37] Y . Guo, Z. Hu, Y . Mao, B. Zheng, Y . Gao, and M. Zhou, “BIRDIE: nat-
ural language-driven table discovery using differentiable search index,”
Proc. VLDB Endow., vol. 18, no. 7, pp. 2070–2083, 2025.
[38] E. J. Hu, Y . Shen, P. Wallis, Z. Allen-Zhu, Y . Li, S. Wang, L. Wang,
and W. Chen, “Lora: Low-rank adaptation of large language models,”
inICLR, 2022.
[39] J. S. Vitteret al., “I/o-efficient algorithms and environments,”ACM
Computing Surveys, vol. 31, no. 2, pp. 256–257, 1999.
[40] M. I. L. Balaka, D. Alexander, Q. Wang, Y . Gong, A. Krisnadhi,
and R. Castro Fernandez, “Pneuma:leveraging llms for tabular data
representation and retrieval in an end-to-end system,”Proc. ACM Manag.
Data, vol. 3, no. 3, 2025.
[41] R. Tito, D. Karatzas, and E. Valveny, “Hierarchical multimodal trans-
formers for multipage docvqa,”Pattern Recognit., vol. 144, p. 109834,
2023.
[42] M. Mathew, V . Bagal, R. Tito, D. Karatzas, E. Valveny, and C. V .
Jawahar, “Infographicvqa,” inWACV, 2022, pp. 2582–2591.
[43] L. Li, Y . Wang, R. Xu, P. Wang, X. Feng, L. Kong, and Q. Liu,
“Multimodal arxiv: A dataset for improving scientific comprehension
of large vision-language models,” inACL, 2024, pp. 14 369–14 387.
[44] N. Methani, P. Ganguly, M. M. Khapra, and P. Kumar, “Plotqa: Rea-
soning over scientific plots,” inWACV, 2020, pp. 1527–1536.
[45] “Paddleocr: Multi-language ocr toolkit based on paddlepaddle,” https:
//github.com/PaddlePaddle/PaddleOCR, 2020.
[46] C. Lee, R. Roy, M. Xu, J. Raiman, M. Shoeybi, B. Catanzaro, and
W. Ping, “Nv-embed: Improved techniques for training llms as generalist
embedding models,” inICLR, 2025.
[47] P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu,
J. Wang, W. Ge, Y . Fan, K. Dang, M. Du, X. Ren, R. Men, D. Liu,
C. Zhou, J. Zhou, and J. Lin, “Qwen2-vl: Enhancing vision-language
model’s perception of the world at any resolution,”arXiv preprint
arXiv:2409.12191, 2024.