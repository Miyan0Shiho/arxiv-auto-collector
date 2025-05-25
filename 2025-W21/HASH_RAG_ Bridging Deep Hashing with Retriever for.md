# HASH-RAG: Bridging Deep Hashing with Retriever for Efficient, Fine Retrieval and Augmented Generation

**Authors**: Jinyu Guo, Xunlei Chen, Qiyang Xia, Zhaokun Wang, Jie Ou, Libo Qin, Shunyu Yao, Wenhong Tian

**Published**: 2025-05-22 02:22:11

**PDF URL**: [http://arxiv.org/pdf/2505.16133v1](http://arxiv.org/pdf/2505.16133v1)

## Abstract
Retrieval-Augmented Generation (RAG) encounters efficiency challenges when
scaling to massive knowledge bases while preserving contextual relevance. We
propose Hash-RAG, a framework that integrates deep hashing techniques with
systematic optimizations to address these limitations. Our queries directly
learn binary hash codes from knowledgebase code, eliminating intermediate
feature extraction steps, and significantly reducing storage and computational
overhead. Building upon this hash-based efficient retrieval framework, we
establish the foundation for fine-grained chunking. Consequently, we design a
Prompt-Guided Chunk-to-Context (PGCC) module that leverages retrieved
hash-indexed propositions and their original document segments through prompt
engineering to enhance the LLM's contextual awareness. Experimental evaluations
on NQ, TriviaQA, and HotpotQA datasets demonstrate that our approach achieves a
90% reduction in retrieval time compared to conventional methods while
maintaining considerate recall performance. Additionally, The proposed system
outperforms retrieval/non-retrieval baselines by 1.4-4.3% in EM scores.

## Full Text


<!-- PDF content starts -->

arXiv:2505.16133v1  [cs.IR]  22 May 2025HASH-RAG: Bridging Deep Hashing with Retriever for Efficient, Fine
Retrieval and Augmented Generation
Jinyu Guo1, Xunlei Chen1Qiyang Xia1, Zhaokun Wang1, Jie Ou1,
Libo Qin2,Shunyu Yao2Wenhong Tian1
1University of Electronic Science and Technology of China2Central South University
3Big data and artificial intelligent institute, China Telecom Research Institute
guojinyu@uestc.edu.cn
Abstract
Retrieval-Augmented Generation (RAG) en-
counters efficiency challenges when scaling to
massive knowledge bases while preserving con-
textual relevance. We propose Hash-RAG, a
framework that integrates deep hashing tech-
niques with systematic optimizations to ad-
dress these limitations. Our queries directly
learn binary hash codes from knowledgebase
code, eliminating intermediate feature extrac-
tion steps, and significantly reducing storage
and computational overhead. Building upon
this hash-based efficient retrieval framework,
we establish the foundation for fine-grained
chunking. Consequently, we design a Prompt-
Guided Chunk-to-Context (PGCC) module that
leverages retrieved hash-indexed propositions
and their original document segments through
prompt engineering to enhance the LLM’s con-
textual awareness. Experimental evaluations on
NQ, TriviaQA, and HotpotQA datasets demon-
strate that our approach achieves a 90% reduc-
tion in retrieval time compared to conventional
methods while maintaining considerate recall
performance. Additionally, The proposed sys-
tem outperforms retrieval/non-retrieval base-
lines by 1.4-4.3% in EM scores.
1 Introduction
In the era of rapidly expanding data, an increasing
number of downstream tasks rely on large language
models (LLMs). Within these tasks, Retrieval-
Augmented Generation (RAG) is a popular tech-
nical framework that incorporates external knowl-
edge sources to tackle knowledge-intensive prob-
lems (Lewis et al., 2020). By combining a non-
parametric retrieval module with the main model,
RAG effectively alleviates hallucination issues in
large models (Yao et al., 2022; Bang et al., 2023).
Moreover, this retrieval-and-generation mechanism
expands the capabilities of LLMs in few- or zero-
shot settings (Brown et al., 2020; Chowdhery et al.,
2023), which is now widely considered a standard
Figure 1: Framework of Deep Supervised Hashing
with Pairwise Similarity. The framework computes
similarity-preserving loss by aligning pairwise rela-
tionships between hash codes and their corresponding
ground truth.
solution for addressing factual shortcomings in tra-
ditional LLMs (Ma et al., 2023).
Behind the effectiveness of RAG, the huge scale
of the knowledge bases is to ensure the quality and
professionalism of the output (Izacard et al., 2023).
With the rapid growth of the model scale and the
amount of knowledge data, the scale of the RAG
knowledgebase that is growing increasingly has
become a trend (Chen et al., 2021). In view of this,
the efficiency and performance of knowledgebase
retrieval are more important than ever.
Current research has dedicated significant ef-
forts to optimize the retrieval process in RAG sys-
tems. Within them, Several approaches focus on
multi-round retrieval iterations to obtain more com-
prehensive and higher-quality content (Yao et al.,
2023). Others employ task-specific or domain-
adapted model fine-tuning to enhance performance
in targeted scenarios. Alternative strategies involve
implementing diversified retrieval techniques or ag-
gregating information from heterogeneous sources
(Huang et al., 2023). Chunk optimization strategies
demonstrate effectiveness in improving retrieval re-
sults by adjusting the chunk size (Sarthi et al., 2024)
while mitigating risks of model performance degra-

dation (Zhao et al., 2024). Despite these advance-
ments in performance enhancement, RAG systems
consistently encounter significant efficiency chal-
lenges as knowledge bases expand.
In the domain of large-scale data retrieval, Ap-
proximate Nearest Neighbor (ANN) search attracts
significant attention due to its capability to sub-
stantially reduce search complexity under most
scenarios (Do et al., 2016; Wang et al., 2017).
Among ANN techniques (Luo et al., 2021), hash-
ing methods emerged as one of the most widely
adopted approaches, offering exceptional storage
efficiency and rapid retrieval capabilities (Cantini
et al., 2021). In particular, deep hashing methods
(Xia et al., 2014) employ deep neural networks
to learn discriminative feature representations and
convert them into compact hash codes, substan-
tially enhancing retrieval efficiency while reducing
storage requirements and computational costs (Lai
et al., 2015). This approach marks an unprece-
dented breakthrough in large-scale image retrieval
and significantly outperforms conventional method
(Chen et al., 2021).
In the era of RAG, the knowledge bases signifi-
cantly surpass traditional image retrieval datasets
in scale and growth velocity. In light of this, ANN
techniques exemplified by hashing demonstrate sig-
nificant potential for RAG applications with their
capacity to rapidly target results and reduce compu-
tational complexity in large-scale data processing.
In this paper, we introduce ANN-based tech-
niques into RAG frameworks and propose Hash-
RAG through systematic integration of deep hash-
ing methods. Specifically, our architecture con-
verts query embeddings into binary hash codes via
sign function operations. For knowledge bases, we
adopt an asymmetric processing strategy to opti-
mize training efficiency by directly learning binary
hash codes without feature learning. Based on this,
we achieve fine-grained retrieval through corpus
chunking, which filters redundant content while
preserving precision. Nevertheless, we notice that
existing chunking approaches result in retrieved
segments lacking essential contextual information,
which substantially degrades the quality of gener-
ated outputs. To address this, we also propose a
Prompt-Guided Chunk-to-Context (PGCC) mod-
ule, which splits documents into factual fragments
(i.e., propositions) as retrieval units. These propo-
sitions are structured in a concise, self-contained
natural language format and indexed to their origi-nal documents. During generation, LLM processes
hash-based retrieved propositions and their con-
texts through specifically designed prompts to gen-
erate, achieving optimal coordination between ac-
curacy and efficiency.
We conducted experiments on open-domain
question-answering datasets, including Natural
Questions (NQ) (Kwiatkowski et al., 2019), TRIV-
IAQA (Joshi et al., 2017), and the more complex
multi-hop HOTPOTQA (Yang et al., 2018). Exper-
imental results show that our model significantly
reduces retrieval time, which requires only 10% of
the time needed for conventional retrieval methods
while maintaining advanced recall performance.
In combination with the PGCC module, we have
achieved a performance increase in the generation
task while retaining efficiency.
Main contributions of this paper are as follows:
1.We propose HASH-RAG, a framework that
systematically integrates deep hashing into
RAG with stage-wise optimizations. This
approach significantly enhances computa-
tional efficiency in large-scale knowledge re-
trieval, thereby accelerating end-to-end infer-
ence throughout the overall RAG.
2.Building upon our hash-based efficient re-
trieval framework, we propose the PGCC
module that enables fine-grained retrieval
while enhancing contextual information
through prompt-based optimization.
3.Experimental results on multiple datasets
demonstrate that HASH-RAG significantly
improves the efficiency of the retrieval.
With PGCC module, our method surpasses
RAG baseline models in overall performance,
achieving optimal coordination between effi-
ciency and performance.
2 Related Work
2.1 Retrieval-Augmented Generation
RAG mitigates LLM hallucinations through non-
parametric memory integration and compensates
for factual deficiencies via external knowledge
retrieval (Gao et al., 2023). Early implementa-
tions relied on statistical similarity metrics (TF-
IDF (Robertson and Walker, 1997), BM25 (Robert-
son et al., 2009)) before transitioning to vector-
ized representations (Karpukhin et al., 2020), en-
abling end-to-end tunable systems with context-
aware retrieval. Recent efforts focus on two phases:

pre-retrieval query-data matching to enhance pre-
cision (Ma et al., 2023) and post-retrieval content
re-ranking or transformation to optimize generator
input (Glass et al., 2022). A persistent challenge
lies in chunking strategy design, balancing gran-
ularity trade-offs: coarse chunks risk redundancy
despite contextual richness (Shi et al., 2023), while
fine-grained units sacrifice semantic completeness
for precision (Raina and Gales, 2024). Scalability-
induced efficiency bottlenecks in retrieval optimiza-
tion now critically constrain RAG advancement.
2.2 Deep Hashing Methods
Approximate Nearest Neighbor (ANN) algorithms
address large-scale search inefficiency by trading
exact precision for efficiency. Unlike tree-based
(Annoy (Bernhardsson, 2015)), quantization-based
(PQ (Jegou et al., 2010)), or graph-based (HNSW
(Malkov and Yashunin, 2018)) methods, hashing
reduces memory requirements and search latency
while preserving vector locality, which makes it a
mainstream ANN solution. Early techniques like
Locality-Sensitive Hashing (LSH) (Charikar, 2002;
Indyk and Motwani, 1998) relied on predefined
mappings to hash buckets, requiring multiple tables
for satisfactory recall. Learning-to-hash methods
(e.g., Spectral Hashing (Weiss et al., 2008), Se-
mantic Hashing (Salakhutdinov and Hinton, 2009))
later optimized hash functions to improve retrieval
efficiency and accuracy. Current research focuses
on deep-supervised hashing methods (e.g., Con-
volutional neural network hashing (CNNH) (Xia
et al., 2014): two-phase of binary codes generation
and convolutional neural networks (CNNs) training,
Deep supervised hashing (DSH) (Liu et al., 2016):
pairwise similarity loss with regularization for end-
to-end training, Maximum-margin hamming hash-
ing (MMHH) (Kang et al., 2019): discriminative
enhancement through t-distribution and semi-batch
optimization). These methods demonstrate adapt-
ability to large-scale retrieval, establishing techni-
cal foundations for optimizing RAG’s chunking
strategies and retrieval efficiency.
3 Method
In this study, we propose a Hash Retrieval-
Augmented Generation (Hash-RAG) framework, a
pipeline that enhances RAG from the perspectives
of deep hashing and contextual enhancement. Fig-
ure 2 illustrates the comprehensive architecture of
Hash-RAG. Sections 3.1 and 3.2 respectively intro-duce the Hash-Based Retriever (HbR) and the mod-
ule Prompt-Guided Chunk-to-Context (PGCC).
3.1 Hash-Based Retriever
Hash-based Encoder (HbE) HbE module com-
prises two components: a query encoder Eqand
a proposition encoder Ep, which generate binary
hash codes for queries qand propositions p(de-
rived from knowledge base document segmenta-
tion), respectively. Inspired by ADSH (Jiang and
Li, 2018), we design asymmetric encoding strate-
gies for queries and propositions. The query en-
coder Eqemploys BERT-base-uncased (Kenton
and Toutanova, 2019) as its embedding model to
map queries into a d-dimensional vector space
(d= 768 ). The query embedding vector vq∈Rd:
vq=BERT (q, θ) (1)
where θdenotes BERT’s parameter. Subsequently,
the binary hash code of query qis computed
through the sign function in the hashing layer:
hq=sign(vq). To address the vanishing gradi-
ent problem of the sign function, we approximate
it using a scaled tanh function (Cao et al., 2017).
ehq=tanh(βvq)∈ {− 1,1}l(2)
where lis the fixed hash code length and βis the
scaling hyperparameter controlling approximation
smoothness ( β→ ∞ converges to the sign func-
tion). We set σ= 0.1andβ=√σ·step+ 1,
where step counts completed training steps.
The proposition encoder Epdirectly learns
binary hash codes without embedding model
training through specialized loss functions and
an alternating optimization strategy to reduce
training overhead. For training data ∆ =
{⟨qi, p+
i,{p−
i,j}n
j=1⟩}m
i=1containing minstances,
the supervision matrix S∈ {− 1,1}m×(n+1)labels
positive ( Si,j= 1) and negative ( Si,j=−1) sam-
ples. We minimize L2loss between binary code
inner products and supervision information:
LHbE=mX
i=1nX
j=1h
fhqiThpj−lSiji2
(3)
where hpdenotes the proposition hash code. For
proposition-only training data ∆p={qj}n
j=1, we
construct simulated queries by randomly sampling
mpropositions from all proposition indices set
Γ ={1,2, . . . , n }, forming index subset Ω =

Similarity 
information
(B) InferenceAsymmetric 
Pairwise Loss
(A) Training
Knowledgebase 
CodeHbEHash-Based Retriever
Hash CodeRetrieveEmbedding
Model
Hash 
LayerHash-Based Encoder
Query
Query
 Hash Code
Propositionizer
PropositionsDocuments
Directly
 Learn
Knowledgebase
 Code
Doc.
Corresponding IndexPrompt
Prop.
Answer
 Generator
Prompt-Guided 
Chunk-to-ContextFigure 2: Framework Overview. (a) Training. The hash-based encoder generates compact query hash codes,
while the knowledge base creates binarized propositional codes from factually chunked corpora. Both components
are jointly optimized through an asymmetric pairwise loss with similarity constraints. (b) Inference. The hash-
based retriever efficiently fetches relevant propositions, augmented by indexed document references for contextual
grounding. The generator synthesizes evidence from these elements using optimized prompts to produce responses.
{i1, i2, . . . , i m} ⊆Γ. The extended loss function
is as follows:
LHbE=X
i∈ΩX
j∈Γ
tanh( βvpi)Thpj−lSij2
+γX
i∈Ω
hpj−tanh( βvpi)2(4)
where γconstrains hpiandfhpi= tanh( βvpi)to
be as close as possible, which enables effective
optimization of proposition hash codes through
iterative parameter updates.
We implement an alternating optimization strat-
egy, alternately fixing and updating the neural net-
work parameters θand the proposition sentence
hash code matrix H.
Update θwithHfixed WithHfixed, we com-
pute the gradient of LHbE w.r.t. vpi:
∂LHbE
∂vpi=(
2X
j∈Γh
fhpiThpj−lSij
hpji
+ 2γ
fhpi−hpi)
⊙
1−fhpi2(5)
The chain rule propagates this gradient through
BERT’s parameters θ, which are then updated via
backpropagation.
Update Hwithθfixed With θfixed, we rewrite
Equation 4 in matrix form:LHbE=∥eV HT−lS∥2
F+γ∥HΩ−eV∥2
F
=∥eV HT∥2
F−2ltr(HTSTeV)
−2γtr(HΩeVT) + const(6)
whereeV= [fvp1,fvp2, . . . ,gvpm]T∈[−1,+1]m×l,
andHΩ= [hp1, hp2, . . . , h pm]Tdenotes the sam-
pled proposition hash codes.
To update H, we adopt a column-wise updating
strategy. For the k-th column H∗kwith residual
matrices cHk(excluding column k), and the k-th
column gV∗kwith residual matrices bVk(excluding
column k), the optimisation objective function is:
L(H∗k) =tr
H∗kh
2gV∗kTbVkdHkT+QT
∗ki
+ const
(7)
whereQ=−2lSTeV−2γV, with the k-th column
Q∗k. The optimal solution is:
H∗k=−sign
2cHkcVkTgV∗k+Q∗k
(8)
The alternating optimization between θandH
drives gradual convergence through multiple itera-
tions, ultimately producing an effective query hash
function and robust proposition hash code.
3.2 Prompt-Guided Chunk-to-Context
Retrieval Unit Granularity We employ the in-
formation bottleneck theory (Tishby et al., 2000) to

optimize retrieval unit selection, where proposition-
based chunks preserve maximal generator-relevant
information while minimizing noise. Given the
joint probability distribution p(X, Y)between doc-
ument Xand generator output Y, we quantify the
information content about Ycontained within com-
pressed proposition eXthrough mutual information:
I(eX;Y) =Z
eXZ
Yp(ex, y) logp(ex, y)
p(ex)p(y)dexdy
(9)
The compression objective minimizes LIB=
I(eX;X)−βI(eX;Y), where the Lagrange mul-
tiplier βbalances information retention and com-
pression. Unlike conventional sentence/paragraph
units (Karpukhin et al., 2020), we adopt proposition
units (Min et al., 2023) that capture atomic seman-
tic expressions. For document Doc, we extract k
interrelated propositions X= [x1, . . . , x k], with
relevance scores computed through hybrid scoring:
Xf=αXdoc+ (1−α)nX
k=1wkxk (10)
where Xdocandxkdenote document-level and
BERT-based proposition scores respectively, with
αandwkoptimized via cross-validation.
Hash-based retrieval optimizes proposition se-
lection through Hamming distance relationships:
distH(hqi, hpj) =1
2(d− ⟨hqi, hpj⟩) (11)
where dis the binary code dimension. We itera-
tively expand the Hamming radius until selecting
the top αpropositions:
TopPj= arg max
i∈{1,...,α}⟨vqi, hpj⟩ (12)
Deduplication over proposition-document map-
pings yields the final top kretrieved documents
{Doc 1, . . . , Doc k}=Duplicates (P1∪. . .∪Pj).
This dual optimization of semantic compression
and hash-based retrieval ensures maximal informa-
tion extraction with minimal noise.
Prompt Optimization We employ LLAMA2
as the generator with optimized prompts. The
Hash-Retriever identifies top- jpropositions Pj=
{P1, . . . , P j}and their corresponding document
indices Doc k, forming the generator’s context
through three key components: (1) AdditionalPrompt instructing semantic integration of propo-
sitions and indexed documents for precise re-
sponses, (2) Retrieved Segments containing
similarity-ranked propositions Pjwith document
references Doc j, and (3) Indexed Documents
Doc 1, . . . , Doc kproviding contextual grounding.
The prompt template activates the generator’s
capability through chunk-to-context: propositions
supply direct evidence while documents offer
broader context, enabling accurate intent under-
standing with balanced retrieval-context integra-
tion. For details, see appendix A.
4 Experiment
4.1 Experimental Settings
Datasets and Retrieval Corpus We evaluated our
model on three QA benchmarks using development
sets. These datasets contain Wikipedia and web-
sourced questions, representing diverse knowledge-
intensive tasks: NQ (Kwiatkowski et al., 2019)
and TRIVIAQA (Joshi et al., 2017) assess direct
knowledge recall, while HOTPOTQA (Yang et al.,
2018) requires multi-hop reasoning across docu-
ments. Different retrieval granularities from sen-
tences to full documents refer to Appendix B.
Metrics With more retrieval units, we retrieve
additional propositions, map them to source doc-
uments, deduplicate, and return the top kunique
documents. We evaluate using document recall@k
and retrieval efficiency (index size/query time). For
generation, Exact Match (EM) assesses whether the
ground truth appears exactly in the output.
Implementation Details In our paper, the en-
coders (Embedding) utilize BERT base, large, AL-
BERT, and ALBERT, with each model initialized
using the official pre-trained weights. The number
of top jpropositions is fixed at 100. The generator
(generator) LLMs include LLaMA2-7B and 13B
(Touvron et al., 2023). For the HbR model used in
our primary experiments, the training batch size is
set to 128, with one additional BM25 negative pas-
sage per question. Each encoder is trained for 40
epochs, employing linear scheduling with warm-up
and a dropout rate of 0.1.
Baselines We compare HbR with BM25 (Robert-
son et al., 2009), DPR (Karpukhin et al., 2020),
SimCSE (Gao et al., 2021), Contriever (Izac-
ard et al., 2021), Model-enhanced Vector Index
(MEVI) (Zhang et al., 2024), LSH (Charikar, 2002),
and DSH (Liu et al., 2016).
BM25 (Robertson et al., 2009) employs TF-

ModelTop 5 Top 20 Top 100 Index Query
NQ TQA HQA NQ TQA HQA NQ TQA HQA size time
BM25 45.2 55.7 - 59.1 66.9 - 73.7 76.7 - 7.4 913.8
SimCSE 28.8 44.9 26.7 44.3 59.4 44.1 47.0 62.4 46.1 64.6 548.2
Contriever 47.8 59.4 42.5 67.8 74.2 67.4 82.1 83.2 76.9 64.6 608.0
MEVI†75.5 - - 82.8 - - 87.3 - - 151.0 222.5
DPR 66.0 71.6 54.4 78.4 79.4 73.0 85.4 85.0 80.3 64.6 456.9
LSH‡43.2 48.0 38.4 63.9 65.2 60.5 77.2 76.9 71.1 2.0 28.8
DSH‡57.2 64.7 44.2 77.9 77.9 66.2 85.7 84.5 80.4 2.2 38.1
HbR(Ours)72.4 78.3 57.7 80.3 87.0 80.2 87.5 88.4 81.44.6 42.3
±0.2±0.2±0.1±0.2±0.1±0.1±0.3±0.1±0.1
Table 1: Top krecall on test sets with the index size(GB) and query time(ms) of HbR and baselines. †Model
selected is MEVI Top-100 & HNSW from the main experiments. ‡Integration of hash with the encoder, selecting
DPR, DSH model selected is hash table lookup with candidate = 1000.
ModelLLAMA2-7B LLAMA2-13B
NQ TQA HQA NQ TQA HQA
ToolFormer♢17.7 48.8 14.5 22.1 51.7 19.2
RRR 25.2 54.9 19.8 27.1 59.7 24.4
FILCO 25.8 55.0 19.4 27.3 60.4 23.9
REPLUG♦27.1 57.1 20.5 29.4 62.7 26.8
Hash-RAG 28.5 ±0.157.1±0.122.1±0.234.9±0.264.5±0.131.1±0.3
Table 2: EM of open-domain QA. ♢Generation models in this experiment involve the GPT series, all of which are
modified to the LLAMA2 series and w/o train reader in this experiment. ♦Contriever and a zero-shot setting are
selected.
IDF principles for document relevance ranking,
while DPR (Karpukhin et al., 2020) utilizes a dual-
encoder architecture; SimCSE (Gao et al., 2021),
an unsupervised learning architecture, optimizes
semantic representations via positive/negative pair
discrimination; Contriever (Izacard et al., 2021)
employs Transformer-based encoder and optimize
a contrastive loss function; MEVI (Zhang et al.,
2024) employs the clusters of Residual Quantiza-
tion (RQ) to search for documents semantically;
LSH’s (Charikar, 2002) hash-bucket mapping re-
duces the scope of nearest neighbor search; DSH
(Liu et al., 2016) integrates deep feature extraction
with semantic label optimization in hash space.
4.2 Main Result
Table 1 illustrates HbR’s recall@k ( k∈5,20,100)
and latency on NQ, TQA, and HotpotQA bench-
marks. Our framework reduces query latency to
one-tenth of conventional retrievers while achiev-
ing 0.2-8.6% higher recall@20/100. Notably, HbR
achieves optimal performance at k= 20 . While H-RAG underperforms MEVI at k= 5, such small- k
scenarios are secondary since users typically re-
quire≥20 passages for answer generation. The
hashing mechanism maintains index size advan-
tages despite data volume increases from chunking.
Next, Table 2 compares Hash-RAG with base-
line RAG systems using LLAMA2-7B/13B. Our
framework outperforms retrieval-optimized meth-
ods (FILCO (Wang et al., 2023), RRR (Ma et al.,
2023)) and LLAMA-Retrieval hybrids (REPLUG
(Shi et al., 2024)), with all retrieval-augmented
models surpassing Toolformer’s non-retrieval base-
line (Schick et al., 2023). By feeding top-20 results
to generators, we optimally balance context volume
and generation quality. This design choice lever-
ages the complementary strengths of hashing-based
retrieval and modern LLMs, demonstrating signifi-
cant EM improvements across all benchmarks.
4.3 Ablations
Encoder Version We investigated the com-
patibility of various encoder versions (ALBERT,

Model Top 5 Top 20 Top 100
ALBERT 63.2 78.1 82.4
Bert-base 72.4 80.3 87.5
RoBERTa 72.6 84.7 87.6
Bert-large 73.1 85.8 87.9
Table 3: Top krecall of HbR using different versions of
embedding models on NQ Dataset.
Chunk Strategy Recall@20
sentence 62.9
paragraph 68.8
prop. 80.2
Prompt Optimization EM
HbRw/o prop. 25.3
HbRw/o doc. 24.8
HbR 29.4
HbRw/prompt (Ours) 31.1
Table 4: Metrics (Recall@20 and EM) of different
chunk strategies and prompt optimizations on HotpotQA
dataset, with prop. denoting the propositions and doc.
representing the original source documents associated
with the retrieved propositions.
Bert-base, Bert-large, RoBERTa) with our model.
As shown in Table 3, proposition-level chunk-
ing achieves significantly superior retrieval perfor-
mance compared to sentence-level and paragraph-
level strategies in terms of Recall@20 metrics. Al-
though BERT-large achieved the highest Top-k re-
call rates, to ensure a fair comparison with existing
models that employ BERT-base as their encoder
architecture, we adopt the identical configuration
in our implementation.
Chunk Strategy The performance hierarchy in
Table 4 demonstrates proposition-level chunking
surpassing sentence- and paragraph-level strate-
gies on Recall@20 metrics respectively. Exper-
imental analysis reveals sentence-level segmenta-
tion fractures predicate-argument coherence, while
paragraph-level processing incorporates extrane-
ous content. The performance hierarchy reflects
proposition-level chunking’s dual advantage: pre-
serving self-contained semantic units and system-
atically eliminating contextual noise.
Prompt Optimization Table 4 presents our com-
parative analysis of prompt optimization strate-
gies, where the PGCC module empirically demon-
strates superior EM performance over all base-
lines. Notably, the w/o doc. configuration out-
performs w/o prop., suggesting that in multi-hop
Figure 3: Prompt Guidance Attention Heat Map
datasets under Recall@20 settings, self-contained
propositions still require cross-verification with
source documents even when guided by optimized
prompts. The non-prompted configuration achieves
secondary performance due to sufficient contextual
data supporting LLM reasoning, while prompt in-
tegration enhances the model’s focus on retrieved
results through structured attention guidance.
5 Analysis
5.1 Chunk-Unit & Prompt-Guided
Information Bottleneck of Proposition To
demonstrate how HASH-RAG’s chunking strategy
leverages the information bottleneck to enhance
text generation capabilities, we analyze content
preserved through document segmentation. Ex-
perimental analysis based on Table 5 reveals a
potential correlation between compression rates
and the conciseness of conditional mutual informa-
tionI(˜X;X|Y;Q), comparing exact and greedy
search methods across context lengths for gener-
ator, mutual information metrics, and EM scores.
We propose that applying information bottleneck
principles to factual chunking generates concise
intermediate answer representations with support-
ing evidence, outperforming alternative strategies
in multi-hop queries. This indicates conventional
chunking methods cannot achieve comparable in-
formation density optimization.
Prompt Guidance on Attention To investigate
how prompts influence attention mechanisms dur-
ing LLM text generation, we employ Recall@1 to
identify a document providing optimal factual sup-
port, thereby validating the effectiveness of prompt
optimization. We generate comparative attention
heatmaps (Figure 3) illustrating model behavior
with versus without prompts. The prompt-free con-
dition exhibits concentrated self-referential atten-
tion along the diagonal axis. In contrast, prompted
generation demonstrates vertical attention patterns

Dataset Filtering Candidates Words I(˜X;X|Y;Q)EM
NQExactParagraph-Level 78.1 0.597 21.2
Sentence-Level 28.4 0.561 23.8
GreedyQuery & Answer 26.2 0.562 19
Answer 18.2 0.556 24.3
Exact Proposition-Level 33.6 0.594 27.4
HotpotQAExactParagraph-Level 120.0 0.679 26.3
Sentence-Level 41.2 0.619 27.8
GreedyQuery & Supporting Facts & Answer 32.5 0.614 25.8
Supporting Facts & Answer 14.8 0.604 26.9
Exact Proposition-Level 42.6 0.679 28.7
Table 5: The effectiveness of the information bottleneck theory on the filtering data compression rate and concise
mutual information in the test sets of the PGCC module for NQ and HotpotQA.
Figure 4: Training time on NQ dataset.
focusing on proposition tokens Pj, accompanied
by significant off-diagonal highlights indicating
strengthened long-range dependencies between an-
swer generation positions and critical propositions.
5.2 Training of deep hashing algorithms
Time Complexity We compare our hash method
to other deep hashing baselines on NQ dataset, with
the training time results illustrated in Figure 4. In
our evaluation framework, DSH (Liu et al., 2016)
and DHN (Zhu et al., 2016) represent conventional
deep hashing baselines trained on 10,000 sampled
data points, while DSH-D and DHN-D denote their
full-database counterparts. The result reveals that
full-database training of baselines needs over 80
minutes for convergence, which inspires the sam-
pled training of the large-scale datasets. Moreover,
our method achieves significantly faster conver-
gence than both sampled and full-database base-
lines while maintaining the highest accuracy.
Sensitivity to Parameters Figure 5 illustrates
the hashing hyperparameter γsensitivity on NQ
dataset with 24-bit codes. Our method shows stabil-
Figure 5: Hyperparameter γon NQ dataset.
ity across a broad range (1 < γ < 500), with Mean
Average Precision (MAP) fluctuating within 0.01.
It could potentially be attributed to NQ’s hierarchi-
cal semantic structure, which exhibits tolerance to
hash-induced local perturbations. This parameter
invariance reduces deployment optimization com-
plexity while ensuring multi-scenario reliability.
Conclusion
We bridge deep hashing with retrieval-augmented
generation for efficient, fine-grained knowledge
retrieval and context-augmented generation, bal-
ancing the trade-off between the query processing
time and recall. Not only as an evaluation frame-
work for hash retrievers, Our proposed PGCC mod-
ule further improves the accuracy and relevance
of retrieval by optimizing the chunking strategy
and addressing contextual information limitations.
Experimental results demonstrate that our hash re-
triever significantly outperforms baseline methods
and achieves impressive metrics in the generator.
In future work, we plan to explore the application
of hash techniques to other tasks and structures,
such as the knowledge graph.

Limitations
The focus of this paper is to deeply integrate deep
hashing techniques with the RAG model. The ex-
perimental framework assumes that the external
knowledge base is static. If incremental updates are
required, such as adding new documents or revising
content, the hash encoder needs to be retrained to
incorporate the new data, which is computationally
expensive. In the future, developing an efficient
adaptation strategy for dynamic hashing encoding
remains an open challenge.
References
Yejin Bang, Samuel Cahyawijaya, Nayeon Lee, Wen-
liang Dai, Dan Su, Bryan Wilie, Holy Lovenia, Ziwei
Ji, Tiezheng Yu, Willy Chung, et al. 2023. A multi-
task, multilingual, multimodal evaluation of chatgpt
on reasoning, hallucination, and interactivity. arXiv
preprint arXiv:2302.04023 .
Erik Bernhardsson. 2015. Annoy (approx-
imate nearest neighbors oh yeah). URL
https://github.com/spotify/annoy .
Tom Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, et al. 2020. Language models are few-shot
learners. Advances in neural information processing
systems , 33:1877–1901.
Riccardo Cantini, Fabrizio Marozzo, Giovanni Bruno,
and Paolo Trunfio. 2021. Learning sentence-to-
hashtags semantic mapping for hashtag recommen-
dation on microblogs. ACM Transactions on Knowl-
edge Discovery from Data (TKDD) , 16(2):1–26.
Zhangjie Cao, Mingsheng Long, Jianmin Wang, and
Philip S Yu. 2017. Hashnet: Deep learning to hash
by continuation. In Proceedings of the IEEE inter-
national conference on computer vision , pages 5608–
5617.
Moses S Charikar. 2002. Similarity estimation tech-
niques from rounding algorithms. In Proceedings of
the thiry-fourth annual ACM symposium on Theory
of computing , pages 380–388.
Qi Chen, Bing Zhao, Haidong Wang, Mingqin Li,
Chuanjie Liu, Zengzhong Li, Mao Yang, and Jing-
dong Wang. 2021. Spann: Highly-efficient billion-
scale approximate nearest neighborhood search. Ad-
vances in Neural Information Processing Systems ,
34:5199–5212.
Aakanksha Chowdhery, Sharan Narang, Jacob Devlin,
Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul
Barham, Hyung Won Chung, Charles Sutton, Sebas-
tian Gehrmann, et al. 2023. Palm: Scaling language
modeling with pathways. Journal of Machine Learn-
ing Research , 24(240):1–113.Thanh-Toan Do, Anh-Dzung Doan, and Ngai-Man Che-
ung. 2016. Learning to hash with binary deep neural
network. In Computer Vision–ECCV 2016: 14th
European Conference, Amsterdam, The Netherlands,
October 11-14, 2016, Proceedings, Part V 14 , pages
219–234. Springer.
T Gao, X Yao, and Danqi Chen. 2021. Simcse: Sim-
ple contrastive learning of sentence embeddings. In
EMNLP 2021-2021 Conference on Empirical Meth-
ods in Natural Language Processing, Proceedings .
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen
Wang. 2023. Retrieval-augmented generation for
large language models: A survey. arXiv preprint
arXiv:2312.10997 .
Michael Glass, Gaetano Rossiello, Md Faisal Mahbub
Chowdhury, Ankita Rajaram Naik, Pengshan Cai,
and Alfio Gliozzo. 2022. Re2g: Retrieve, rerank,
generate. In Annual Conference of the North Amer-
ican Chapter of the Association for Computational
Linguistics .
Wenyu Huang, Mirella Lapata, Pavlos V ougiouklis,
Nikos Papasarantopoulos, and Jeff Pan. 2023. Re-
trieval augmented generation with rich answer encod-
ing. In Proceedings of the 13th International Joint
Conference on Natural Language Processing and the
3rd Conference of the Asia-Pacific Chapter of the
Association for Computational Linguistics (Volume
1: Long Papers) , pages 1012–1025.
Piotr Indyk and Rajeev Motwani. 1998. Approximate
nearest neighbors: towards removing the curse of
dimensionality. In Proceedings of the thirtieth an-
nual ACM symposium on Theory of computing , pages
604–613.
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Se-
bastian Riedel, Piotr Bojanowski, Armand Joulin,
and Edouard Grave. 2021. Unsupervised dense infor-
mation retrieval with contrastive learning.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas
Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-
Yu, Armand Joulin, Sebastian Riedel, and Edouard
Grave. 2023. Atlas: Few-shot learning with retrieval
augmented language models. Journal of Machine
Learning Research , 24(251):1–43.
Herve Jegou, Matthijs Douze, and Cordelia Schmid.
2010. Product quantization for nearest neighbor
search. IEEE transactions on pattern analysis and
machine intelligence , 33(1):117–128.
Qing-Yuan Jiang and Wu-Jun Li. 2018. Asymmetric
deep supervised hashing. In Proceedings of the AAAI
conference on artificial intelligence , volume 32.
Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke
Zettlemoyer. 2017. Triviaqa: A large scale distantly
supervised challenge dataset for reading comprehen-
sion. In Proceedings of the 55th Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers) , pages 1601–1611.

Rong Kang, Yue Cao, Mingsheng Long, Jianmin Wang,
and Philip S Yu. 2019. Maximum-margin hamming
hashing. In Proceedings of the IEEE/CVF interna-
tional conference on computer vision , pages 8252–
8261.
Vladimir Karpukhin, Barlas O ˘guz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for
open-domain question answering. arXiv preprint
arXiv:2004.04906 .
Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina
Toutanova. 2019. Bert: Pre-training of deep bidirec-
tional transformers for language understanding. In
Proceedings of naacL-HLT , volume 1. Minneapolis,
Minnesota.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, et al. 2019. Natural questions: a benchmark
for question answering research. Transactions of the
Association for Computational Linguistics , 7:453–
466.
Hanjiang Lai, Yan Pan, Ye Liu, and Shuicheng Yan.
2015. Simultaneous feature learning and hash coding
with deep neural networks. In Proceedings of the
IEEE conference on computer vision and pattern
recognition , pages 3270–3278.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich K ¨uttler, Mike Lewis, Wen-tau Yih, Tim
Rockt ¨aschel, et al. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks. Advances
in Neural Information Processing Systems , 33:9459–
9474.
Haomiao Liu, Ruiping Wang, Shiguang Shan, and Xilin
Chen. 2016. Deep supervised hashing for fast image
retrieval. In Proceedings of the IEEE conference
on computer vision and pattern recognition , pages
2064–2072.
Xiao Luo, Daqing Wu, Chong Chen, Jinwen Ma, and
Minghua Deng. 2021. Deep unsupervised hashing by
global and local consistency. In 2021 IEEE Interna-
tional Conference on Multimedia and Expo (ICME) ,
pages 1–6. IEEE.
Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao,
and Nan Duan. 2023. Query rewriting in retrieval-
augmented large language models. In Proceedings
of the 2023 Conference on Empirical Methods in
Natural Language Processing , pages 5303–5315.
Yu A Malkov and Dmitry A Yashunin. 2018. Efficient
and robust approximate nearest neighbor search us-
ing hierarchical navigable small world graphs. IEEE
transactions on pattern analysis and machine intelli-
gence , 42(4):824–836.Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis,
Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettle-
moyer, and Hannaneh Hajishirzi. 2023. Factscore:
Fine-grained atomic evaluation of factual precision
in long form text generation. In Proceedings of the
2023 Conference on Empirical Methods in Natural
Language Processing , pages 12076–12100.
Vatsal Raina and Mark Gales. 2024. Question-based
retrieval using atomic units for enterprise rag. arXiv
preprint arXiv:2405.12363 .
Stephen Robertson, Hugo Zaragoza, et al. 2009. The
probabilistic relevance framework: Bm25 and be-
yond. Foundations and Trends ®in Information Re-
trieval , 3(4):333–389.
Stephen E Robertson and Steve Walker. 1997. On rel-
evance weights with little relevance information. In
Proceedings of the 20th annual international ACM
SIGIR conference on Research and development in
information retrieval , pages 16–24.
Ruslan Salakhutdinov and Geoffrey Hinton. 2009. Se-
mantic hashing. International Journal of Approxi-
mate Reasoning , 50(7):969–978.
Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh
Khanna, Anna Goldie, and Christopher D Man-
ning. 2024. Raptor: Recursive abstractive pro-
cessing for tree-organized retrieval. arXiv preprint
arXiv:2401.18059 .
Timo Schick, Jane Dwivedi-Yu, Roberto Dess `ı, Roberta
Raileanu, Maria Lomeli, Eric Hambro, Luke Zettle-
moyer, Nicola Cancedda, and Thomas Scialom. 2023.
Toolformer: Language models can teach themselves
to use tools. Advances in Neural Information Pro-
cessing Systems , 36:68539–68551.
Freda Shi, Xinyun Chen, Kanishka Misra, Nathan
Scales, David Dohan, Ed H Chi, Nathanael Sch ¨arli,
and Denny Zhou. 2023. Large language models can
be easily distracted by irrelevant context. In Inter-
national Conference on Machine Learning , pages
31210–31227. PMLR.
Weijia Shi, Sewon Min, Michihiro Yasunaga, Min-
joon Seo, Richard James, Mike Lewis, Luke Zettle-
moyer, and Wen-tau Yih. 2024. Replug: Retrieval-
augmented black-box language models. In Proceed-
ings of the 2024 Conference of the North American
Chapter of the Association for Computational Lin-
guistics: Human Language Technologies (Volume 1:
Long Papers) , pages 8364–8377.
Naftali Tishby, Fernando C Pereira, and William Bialek.
2000. The information bottleneck method. arXiv
preprint physics/0004057 .
Hugo Touvron, Louis Martin, Kevin Stone, Peter Al-
bert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti
Bhosale, et al. 2023. Llama 2: Open founda-
tion and fine-tuned chat models. arXiv preprint
arXiv:2307.09288 .

Jingdong Wang, Ting Zhang, Nicu Sebe, Heng Tao
Shen, et al. 2017. A survey on learning to hash.
IEEE transactions on pattern analysis and machine
intelligence , 40(4):769–790.
Zhiruo Wang, Jun Araki, Zhengbao Jiang, Md Rizwan
Parvez, and Graham Neubig. 2023. Learning to filter
context for retrieval-augmented generation. arXiv
preprint arXiv:2311.08377 .
Yair Weiss, Antonio Torralba, and Rob Fergus. 2008.
Spectral hashing. Advances in neural information
processing systems , 21.
Rongkai Xia, Yan Pan, Hanjiang Lai, Cong Liu, and
Shuicheng Yan. 2014. Supervised hashing for im-
age retrieval via image representation learning. In
Proceedings of the AAAI conference on artificial in-
telligence , volume 28.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D Manning. 2018. Hotpotqa: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing , pages
2369–2380.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik Narasimhan, and Yuan Cao. 2022.
React: Synergizing reasoning and acting in language
models. arXiv preprint arXiv:2210.03629 .
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik Narasimhan, and Yuan Cao. 2023.
React: Synergizing reasoning and acting in language
models. In International Conference on Learning
Representations (ICLR) .
Hailin Zhang, Yujing Wang, Qi Chen, Ruiheng Chang,
Ting Zhang, Ziming Miao, Yingyan Hou, Yang Ding,
Xupeng Miao, Haonan Wang, et al. 2024. Model-
enhanced vector index. Advances in Neural Informa-
tion Processing Systems , 36.
Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhengren
Wang, Yunteng Geng, Fangcheng Fu, Ling Yang,
Wentao Zhang, and Bin Cui. 2024. Retrieval-
augmented generation for ai-generated content: A
survey. arXiv preprint arXiv:2402.19473 .
Han Zhu, Mingsheng Long, Jianmin Wang, and Yue
Cao. 2016. Deep hashing network for efficient sim-
ilarity retrieval. In Proceedings of the AAAI confer-
ence on Artificial Intelligence , volume 30.A Prompt Template
Open-domain QA for LLaMA-2-7B
Please consider all relevant details in the
retrieved segments and offer a concise, in-
formative, and contextually appropriate
response. If necessary, carefully review
the corresponding indexed documents
to support your answer with just a few
words:
IdxID:23 Title: Mount Everest
Propositions: Mount Everest is Earth’s
highest mountain above sea level.
IdxID:23 Title: Mount Everest
Propositions: Mount Everest is known in
Tibetan as Chomolungma.
IdxID:59 Title: Lhasa Tibetan
Propositions: Verbs in Tibetan always
come at the end of the clause.
....
ID=23 Title: Mount Everest
Doc: Mount Everest, known locally as
Sagarmatha or Qomolangma,[note 4] is
Earth’s highest mountain above sea level,
located in the Mahalangur Himal sub-range
of the Himalayas. The China–Nepal border
....
ID=59 Title: Lhasa Tibetan
Doc: In the traditional ”three-branched”
classification of the Tibetic languages, the
Lhasa dialect belongs to the Central Tibetan
branch (the other two being Khams Tibetan
and Amdo Tibetan).[4] In terms of
....
Question: What is the highest mountain in
the world?
The answer is: Mount Everest
B Dataset
B.1 Dataset statistics
We use three datasets built from Wikipedia articles
as supporting documents for answer, response, and
judgment generation, as listed in Table 6.

Dataset# Examples
train dev test
NQ 79.2 8.7 3.6
TQA 78.8 8.8 11.3
HOTPOTQA 88.9 5.6 5.6
Table 6: Dataset statistics
B.2 Units of Wikipedia
We refer to the processed corpus as Prop-WIKI.
The statistics of Prop-WIKI are shown in Table
7. Notably, the values presented here correspond
to the average segment length of the processed
Wikipedia corpus, while Table 5 specifically re-
ports the average input sequence length fed into the
text generator during inference.
Table 7: Statistics of text units in the English Wikipedia.
# units Avg. # words
Passages 41,393,528 58.5
Sentences 114,219,127 21.0
Propositions 261,125,423 23.2