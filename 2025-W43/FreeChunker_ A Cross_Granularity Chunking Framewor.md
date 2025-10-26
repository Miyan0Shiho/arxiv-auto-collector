# FreeChunker: A Cross-Granularity Chunking Framework

**Authors**: Wenxuan Zhang, Yuan-Hao Jiang, Yonghe Wu

**Published**: 2025-10-23 08:57:00

**PDF URL**: [http://arxiv.org/pdf/2510.20356v1](http://arxiv.org/pdf/2510.20356v1)

## Abstract
Chunking strategies significantly impact the effectiveness of
Retrieval-Augmented Generation (RAG) systems. Existing methods operate within
fixed-granularity paradigms that rely on static boundary identification,
limiting their adaptability to diverse query requirements. This paper presents
FreeChunker, a Cross-Granularity Encoding Framework that fundamentally
transforms the traditional chunking paradigm: the framework treats sentences as
atomic units and shifts from static chunk segmentation to flexible retrieval
supporting arbitrary sentence combinations. This paradigm shift not only
significantly reduces the computational overhead required for semantic boundary
detection but also enhances adaptability to complex queries. Experimental
evaluation on LongBench V2 demonstrates that FreeChunker achieves superior
retrieval performance compared to traditional chunking methods, while
significantly outperforming existing approaches in computational efficiency.

## Full Text


<!-- PDF content starts -->

FreeChunker: A Cross-Granularity Chunking Framework
Wenxuan Zhang1Yuan-Hao Jiang1Yonghe Wu2
Abstract
Chunking strategies significantly impact the ef-
fectiveness of Retrieval-Augmented Generation
(RAG) systems. Existing methods operate within
fixed-granularity paradigms that rely on static
boundary identification, limiting their adaptabil-
ity to diverse query requirements. This paper
presentsFreeChunker, a Cross-Granularity En-
coding Framework that fundamentally transforms
the traditional chunking paradigm: the framework
treats sentences as atomic units and shifts from
static chunk segmentation to flexible retrieval sup-
porting arbitrary sentence combinations. This
paradigm shift not only significantly avoids the
computational overhead required for semantic
boundary detection, but also enhances adaptabil-
ity to complex queries. Experimental evaluation
on LongBench V2 demonstrates thatFreeChun-
kerachieves superior retrieval performance com-
pared to traditional chunking methods, while sig-
nificantly outperforming existing methods in com-
putational efficiency.
1. Introduction
The rapid advancement of large language models (LLMs)
has substantially propelled progress in natural language pro-
cessing, enabling strong generalization capabilities across
diverse tasks (Brown et al., 2020). However, LLMs typically
rely on static, parameterized knowledge, often making them
less suitable for scenarios requiring up-to-date or domain-
specific information. This limitation frequently leads to
hallucinated content that may deviate from factual correct-
ness (Huang et al., 2025; Li et al., 2023).
To address this limitation, RAG has emerged as a promising
paradigm. By incorporating external knowledge sources,
RAG enhances the generation process with more timely
and verifiable content, achieving strong performance on
1Shanghai Institute of Artificial Intelligence for Education,
East China Normal University, China2Education Technology, East
China Normal University, China. Correspondence to: Yonghe Wu
<yhwu@deit.ecnu.edu.cn>.
The Silence Between StarsThere is a kind of silence that does not belong to night, nor to the absence of sound, but to the space between thoughts—between the pulse of one heartbeat and the next. It is the silence that stretches between stars, where light travels for ages before ﬁnding another witness.Beneath that endless sky, the world forgets its borrowed name. Everything that seems solid—our memories, our fears, even the faces we love—glimmers for a moment, then dissolves like breath on a mirror. And yet, from that vanishing, something persists. Each spark that dies gives birth to another. Life is not a line but a rhythm, a ﬂicker of light endlessly exchanged.The rivers understand this better than we do. They move without hesitation, whispering to the stones with silver tongues. Through forests that have seen centuries of dreams, their murmurs carry the memory of rains, of lost voices, of time itself humming softly beneath the roots.We, too, are wanderers between what fades and what stays. We live in the brief glow between two eternities—the one before birth, and the one after death. And yet, within that fragile interval, something inﬁnite stirs. Perhaps it is not the stars that shine upon us, but we who momentarily remember how to shine.In the silence that remains after everything is said, we discover a truth both tender and terrifying: the soul and the self are the same, and neither truly belongs to us. They are fragments of a larger music, a vast harmony that plays whether or not we are listening.And so, as we walk beneath the quiet constellations, we learn not to fear the void, but to listen to it. For in that silence between stars—where all beginnings and endings blur—we are not alone. We are the echo, the spark, the song itself.
The Silence Between StarsThere is a kind of silence that does not belong to night, nor to the absence of sound, but to the space between thoughts—between the pulse of one heartbeat and the next. It is the silence that stretches between stars, where light travels for ages before ﬁnding another witness.Beneath that endless sky, the world forgets its borrowed name. Everything that seems solid—our memories, our fears, even the faces we love—glimmers for a moment, then dissolves like breath on a mirror. And yet, from that vanishing, something persists. Each spark that dies gives birth to another. Life is not a line but a rhythm, a ﬂicker of light endlessly exchanged.The rivers understand this better than we do. They move without hesitation, whispering to the stones with silver tongues. Through forests that have seen centuries of dreams, their murmurs carry the memory of rains, of lost voices, of time itself humming softly beneath the roots.We, too, are wanderers between what fades and what stays. We live in the brief glow between two eternities—the one before birth, and the one after death. And yet, within that fragile interval, something inﬁnite stirs. Perhaps it is not the stars that shine upon us, but we who momentarily remember how to shine.In the silence that remains after everything is said, we discover a truth both tender and terrifying: the soul and the self are the same, and neither truly belongs to us. They are fragments of a larger music, a vast harmony that plays whether or not we are listening.And so, as we walk beneath the quiet constellations, we learn not to fear the void, but to listen to it. For in that silence between stars—where all beginnings and endings blur—we are not alone. We are the echo, the spark, the song itself.
0.6−0.30.9⋮−0.80.10.0−1.00.5−0.6⋮0.3−0.40.7−0.60.7−1.0⋮0.20.00.4
0.0−0.20.3⋮−0.70.4−1.0Traditional ChunkerFreeChunker
0.20.00.8⋮−0.5−0.21.0−0.70.4−0.9⋮0.60.3−0.1
0.1−0.80.5⋮−0.20.9−0.3−0.41.0−0.1⋮0.5−0.60.7
−0.60.4−0.2⋮0.7−0.30.5Figure 1.Cross-Granularity ChunkingFreeChunkerbreaks the
limitation of fixed-granularity text chunking, enabling flexible
extraction of sentence combinations at arbitrary granularities.
knowledge-intensive tasks such as open-domain question
answering and fact verification (Lewis et al., 2020). How-
ever, the effectiveness of RAG critically depends on the
quality of retrieved content, which is substantially influ-
enced by document chunking strategies (Ru et al., 2024).
Prior research has demonstrated that chunking granularity
often directly affects intra-chunk coherence, semantic cov-
erage, and downstream generation quality. Inappropriate
chunking strategies may disrupt semantic structures, poten-
tially leading to information fragmentation or redundancy,
ultimately undermining both retrieval and generation perfor-
mance (Brådland et al., 2025; Lyu et al., 2025).
To improve chunking quality, recent efforts have explored
semantic-aware chunking techniques. For instance, Seman-
ticChunker (LangChain, 2024) segments documents based
on sentence embedding similarity. Meta-Chunking (Zhao
et al., 2024) identifies semantic boundaries through perplex-
ity analysis and probability-based segmentation decisions.
LumberChunker (Duarte et al., 2024) leverages LLMs to
predict chunk boundaries more aligned with human interpre-
tation. However, existing approaches often lack the flexibil-
ity to adapt to diverse query intents or varying deployment
constraints. Moreover, some methods rely on computa-
tionally expensive large models, where the marginal gains
1arXiv:2510.20356v1  [cs.CL]  23 Oct 2025

FreeChunker: A Cross-Granularity Chunking Framework
of semantic chunking may not always justify the compu-
tational overhead (Qu et al., 2025). More fundamentally,
these approaches still operate within a single-granularity
paradigm, consequently limiting their ability to simultane-
ously accommodate different information needs within the
same document collection.
To overcome the fundamental limitation of fixed chunking
schemes that cannot simultaneously provide multiple gran-
ularity levels,FreeChunkeris proposed in this paper—a
novel Cross-Granularity Encoding Framework that intro-
duces an alternative chunking paradigm. As illustrated in
Figure 1, unlike traditional chunking methods that are typ-
ically constrained by fixed text boundaries, the proposed
method enables retrievers to extract arbitrary combinations
of sentences from documents, thereby providing enhanced
flexibility in content retrieval. The following core contribu-
tions are offered by the proposed method:
•Cross-granularity chunking paradigm: A novel ap-
proach is proposed that breaks the limitation of fixed-
granularity text chunking, enabling flexible extraction
of sentence combinations at arbitrary granularities.
•Computational efficiency: The framework reduces
chunking overhead by treating sentences as atomic
units, avoiding expensive boundary identification pro-
cesses.
•Superior performance: The method achieves compet-
itive retrieval performance by adapting multiple granu-
larity levels to diverse query requirements, demonstrat-
ing enhanced effectiveness across various information
retrieval scenarios in LongBench V2 Task.
2. Related Work
Limitations of Fixed-granularity ChunkingAs a foun-
dational step in RAG, text chunking substantially influences
overall system performance. The most straightforward strat-
egy divides documents into fixed-granularity chunks, which
is simple to implement but often misaligns with semantic
boundaries, consequently leading to information fragmenta-
tion or redundancy (Ru et al., 2024). More critically, fixed
chunking forces a single granularity choice across the entire
document, thereby preventing the system from simultane-
ously accessing both fine-grained details and coarse-grained
context that different queries may require. This granularity
rigidity fundamentally limits the retrieval system’s ability to
adapt to diverse information needs within a single document
collection.
Semantic-aware Chunking TechniquesTo address the
aforementioned semantic boundary issues, recent work has
…………ChunkPatternMask+
!  WQ
!  WK…………AttentionMatrixe3e1h1−2h2−3h3−4h1−3e2e40.1
0.3−0.40.1−0.10.20.2−0.3−0.100000
00.100000.2
0.2−0.10SoftMax
!  WVe1e2e3⋮en−2en−1en·−∞0−∞−∞00000000e3e1e2e4h1−2h2−3h3−4h1−3−∞−∞−∞−∞−∞000h1−3000h1−30
!  WV
…………ChunkPatternMask+
!  WQ
!  WK…………AttentionMatrixe3
e1h1−2h2−3h3−4e2e40.1
0.3−0.40.1−0.10.20.2−0.3−0.100000
00.100000.2
0.2−0.10SoftMax
!  WVe1e2e3⋮en−2en−1en·
−∞0−∞−∞00000000e3e1e2e4h1−2h2−3h3−4h1−3−∞−∞−∞−∞
−∞000h2−4000h1−40
!  WV
[e1,e2,e3,…,en−2,en−1,en][h1−2,h2−3,…,h1−n] Sentenizer+  EmbeddingModel
[e1,e2,e3,…,en−2,en−1,en,…,e1−2,e2−3,…,e1−n]
LongText:"s1+s2+s3+…+sn−2+sn−1+sn"…h1−3h2−4h1−4Figure 2. The architecture ofFreeChunker. Long text input is
processed by the Sentenizer and Embedding Model to produce sen-
tence embeddings. The Attention Matrix and Chunk Pattern Mask
work together to generate multiple granularity chunk embeddings
in a single forward pass, enabling flexible on-demand access to
different chunk sizes.
explored more adaptive chunking strategies. SemanticChun-
ker (LangChain, 2024) relies on sentence-level embedding
similarity to enhance intra-chunk semantic coherence by
identifying natural breakpoints based on semantic similarity
thresholds. Meta-Chunking (Zhao et al., 2024) proposes two
adaptive segmentation algorithms leveraging large language
models’ logical awareness: Perplexity Chunking identifies
logical boundaries by analyzing inter-sentence perplexity
variations, while Margin Sampling Chunking determines
boundaries by comparing model probability differences for
segmentation decisions. LumberChunker (Duarte et al.,
2024) leverages large language models to determine chunk
boundaries more aligned with human interpretation. How-
ever, despite their semantic awareness, these methods still
operate under a single granularity paradigm—each docu-
ment is ultimately represented at one chosen level of gran-
ularity, whether sentence-level, paragraph-level, or custom
semantic units.
Adaptive AttemptsInspired by the concept of mixture
of experts (MoE) (Jacobs et al., 1991; Jordan & Jacobs,
1994), recent methods like MoC (Zhao et al., 2025) and
MoG (Mix-of-Granularity) (Zhong et al., 2025) attempt to
enhance adaptability by treating different chunking strate-
gies as expert modules and employing routing mechanisms
to dynamically select the most suitable chunker based on
query characteristics. However, these MoE-inspired meth-
2

FreeChunker: A Cross-Granularity Chunking Framework
ods essentially aggregate existing chunkers as experts; con-
sequently, both the quality of semantic chunking and the
degree of granularity control remain fundamentally bounded
by the capabilities of the underlying chunker experts.
Despite these advances, a fundamental limitation persists:
current methods still operate within a single-chunking,
single-granularity paradigm—whether fixed, semantics-
driven, or query-adaptive. Consequently, retrieval systems
cannot simultaneously access multiple levels of informa-
tion granularity, which is precisely what is needed to better
accommodate diverse query intents. Therefore, there re-
mains a critical need for a paradigm that can transcend these
single-granularity constraints.
3. Methodology
InFreeChunker, sentences are treated as atomic units and
chunk embeddings are generated across multiple granular-
ities simultaneously. This approach typically enables re-
trieval systems to access any desired chunk size on demand,
ranging from fine-grained single sentences to coarse-grained
multi-sentence contexts.
3.1. Cross-Granularity Chunk Pattern
The core innovation ofFreeChunkerlies in proposing a
flexible multi-granularity paradigm that overcomes the fixed
granularity limitation of existing methods by defining multi-
ple chunk patterns, thereby enabling on-demand granularity
access.
Given a document with nsentences D={s 1, s2, . . . , s n},
a Chunk Pattern Mask P∈ {0,−∞}m×nis constructed,
where each row P[i,:] represents the i-th chunk pattern, and
P[i, j] = 0 typically indicates that sentence sjbelongs to
that pattern.
The key insight is that the matrix can encode arbitrary gran-
ularity combinations. For any granularity gand starting
positions, chunk patterns are defined as:
Pg,s[i, j] =(
0ifs≤j < s+g
−∞otherwise(1)
Therefore, the framework can support any combination of
chunk sizes and overlapping strategies.
3.2. Cross-Granularity Encoder
After obtaining the Chunk Pattern Mask P, independently
encoding each chunk would typically cause extensive redun-
dant computation. Sentenizer is introduced to split docu-
ments into sentences and encode them using a token-level
embedding model M, thereby obtaining the sentence em-
bedding matrix E= [⃗ e 1, . . . , ⃗ e n]∈Rn×d. For any chunkspanning from sentence sito sentence sj, its true embed-
ding is denoted as ⃗ ei,j=M([s i, si+1, . . . , s j]), where
[si, si+1, . . . , s j]represents the concatenation of sentences.
A specialized transformer architecture is then modified.
For each layer, a learnable chunk embedding hchk∈Rd
is introduced that is replicated mtimes to form H=
[hchk,hchk, . . . ,h chk]T∈Rm×dand attention is used to
generatemchunk embeddings in parallel:
Q=W QH,K=W KE,V=W VE(2)
Attn(H,E) = softmax
QK⊤
√
d+P
V(3)
The complete chunk encoding layer then applies layer nor-
malization and feed-forward networks:
V=LayerNorm(H+Attn(H,E))(4)
Output=LayerNorm(V+FFN(V))(5)
This design typically allows parallel generation of all chunk
embeddings in a single forward pass, with sentence-level
embeddings being reused across different granularities,
thereby avoiding redundant encoding of sentences.
4. Theoretical Analysis
A central concern forFreeChunkeris whether the cross-
granularity encoder induces non-negligible substitution loss
relative to directly encoding concatenated text chunks. This
section develops a geometric analysis of vector substitution,
establishing worst-case and average-case guarantees that are
verifiable in practice.
4.1. Setup and Constraints
Let⃗ ei,j=M([s i, sj])denote thetruechunk embeddings
of the concatenated chunk [si, sj], and let ⃗ vi,jbe theapprox-
imatechunk embeddings produced by the cross-granularity
encoder. Let ⃗ qbe a query embedding. Throughout this
section all vectors are assumed to be unit-norm.
The substitution loss is studied as follows:
ε:=cos(⃗ q,⃗ e i,j)−cos(⃗ q,⃗ v i,j).(6)
Two parameters characterize the geometric configuration:
ρ= cos(⃗ e i,j,⃗ vi,j),
s= cos(⃗ q,⃗ e i,j),(7)
where ρquantifies substitution quality and scaptures query
alignment with the true chunk.
3

FreeChunker: A Cross-Granularity Chunking Framework
Figure 3. Training Dynamics Across Model Scales.Training and validation curves (top) and test set loss distributions (bottom) for three
different model configurations, demonstrating consistent convergence and effective generalization.
4.2. Worst-Case Substitution Bound
Theorem 1 (Vector Substitution Loss Upper Bound).
Given the parameters in (7),
ε≤s(1−ρ) +p
1−s2p
1−ρ2.(8)
Proof.Let α=∠(⃗ q,⃗ e i,j)andβ=∠(⃗ e i,j,⃗ vi,j), sos=
cosαandρ= cosβ. By the spherical cosine law,
cos(⃗ q,⃗ v i,j) = cosαcosβ+ sinαsinβcosφ,(9)
where φis the relative azimuth of ⃗ qand⃗ vi,jaround the axis
⃗ ei,j. The adversarial configuration is cosφ=−1 , which
yields cos(⃗ q,⃗ v i,j)≥sρ−√
1−s2p
1−ρ2. Subtracting
fromcos(⃗ q,⃗ e i,j) =sgives (8).□
Interpretation.The first term s(1−ρ) is typically the
radial component; the second term√
1−s2p
1−ρ2is tan-
gential and becomes dominant when the query is weakly
aligned with the true chunk ( ssmall) and the substitution
deviates from the truth (ρsmall).
4.3. Expected Error under Random Azimuth
The worst-case bound (8)assumes an adversarial azimuth
between ⃗ qand⃗ vi,j. In practice, the relative azimuth φbe-
tween these vectors around ⃗ ei,jis typically random rather
than adversarial. A mild and empirically verifiable assump-
tion is therefore introduced.
Assumption 1 (Random Azimuth).The relative orienta-
tion between query and approximate vectors around the true
chunk embeddings is uniformly random.
Theorem 2 (Expected Substitution Loss).Under Assump-
tion 1 and given the parameters in (7), the expected substi-tution loss is bounded by:
Eφ[ε|s, ρ]≤s(1−ρ) +2
πp
1−s2p
1−ρ2.(10)
Proof.Using ε≤s(1−ρ) +√
1−s2p
1−ρ2|cosφ| and
E[|cosφ|] = 2/πfor uniformφyields the bound.□
Interpretation.The bound consists of a radial term
s(1−ρ) and a tangential term with coefficient 2/π≈0.636 ,
showing that random orientations substantially reduce ex-
pected degradation compared to worst-case scenarios.
5. Training Process
5.1. Training Setup
Token-level Embedding ModelsThree token-level em-
bedding models are employed as encoders Mfor the
Sentenizer: jina-embeddings-v2-small-en(Günther et al.,
2023)(33M parameters), nomic-embed-text-v1.5 (Nuss-
baum et al., 2024) (137M parameters) and BGE-M3 (Chen
et al., 2024) (570M parameters). Our sentence-level Cross-
Granularity encoder (330M parameters) reuses partial struc-
tures and weights from BGE-M3, with dimension prun-
ing support enabling token-level embedding model reuse
across different dimensions. Documents are split into
sentences and encoded to generate sentence embeddings
E= [⃗ e 1, . . . , ⃗ e n]∈Rn×d.
Dataset and PreprocessingThe training dataset com-
prises 100K documents sampled from The Pile (Gao et al.,
2020). For each document, the first 8000 tokens are ex-
tracted to construct training pairs. Documents are split
into sentences using the Sentenizer, typically yielding 10-
200 sentences per document. Training pairs (⃗ e,⃗ v) are con-
4

FreeChunker: A Cross-Granularity Chunking Framework
Table 1.Results of different chunking methods on LongBench V2 (without Top-1 results)
Methodjina-embeddings-v2-small-en nomic-embed-text-v1.5 BGE-M3
Top-5 Top-10 Time (↓) Top-5 Top-10 Time (↓) Top-5 Top-10 Time (↓)
I. Single-Document QA
Traditional30.86% 30.86%2.11s 35.43% 32.00%3.76s 36.00% 34.29%8.93s
SemanticChunker 33.71%33.71% 12.87s 29.14% 32.57% 15.48s 32.57% 30.86% 19.83s
PPL Chunking31.43% 28.00% 33.09s 32.57% 34.29% 34.73s 33.14% 33.71% 39.82s
Margin Sampling30.29%36.57%364.64s 35.43%36.00%368.68s 34.29% 28.00% 375.22s
FreeChunker32.57%36.57%2.75s 36.00% 36.00%5.29s 38.29% 36.57%9.64s
structed across 5 different granularities (2, 4, 8, 16, 32
sentences), where ⃗ erepresents ground-truth chunk embed-
dings obtained by directly encoding concatenated sentence
spans, and ⃗ vdenotes the corresponding cross-granularity
encoder outputs.
OptimizationThe training objective is formulated to min-
imize the cosine similarity loss:
L= 1−1
|B|X
(⃗ e,⃗ v)∈Bcos(⃗ e,⃗ v)(11)
where Bdenotes the training batch. The AdamW optimizer
is employed with learning rate 1×10−4and batch size 1.
Training is conducted for 2 epochs with cosine schedule
learning rate decay, incorporating warmup over the first one-
third of total training steps. Model validation is performed
every 1000 steps on a held-out validation set of 200 samples
to monitor generalization performance.
Computational ResourcesAll training experiments are
conducted on NVIDIA A800 GPUs with 80GB memory.
The total computational cost exceeds 500 GPU hours, en-
compassing dataset encoding construction and model train-
ing processes.
5.2. Training Dynamics and Evaluation
Convergence AnalysisThe upper panels of Figure 3
present the training and validation cosine loss curves for
three base models. All configurations demonstrate smooth
and stable convergence within 200K steps, validating the
effectiveness of the overall optimization strategy. Notably,
the smallest model, jina-embeddings-v2-small-en (33M pa-
rameters), exhibits the fastest and cleanest convergence with
lower training loss. As model parameters increase, both
training and validation losses show degradation across the
three models. This phenomenon may be attributed to the use
of a fixed 330M-parameter sentence-level encoder for all
three base models, suggesting that scaling the sentence-level
encoder capacity could potentially improve fitting perfor-
mance for BGE-M3 (570M parameters).Test Set Generalization AnalysisThe lower panels of
Figure 3 illustrate the cosine loss distributions on the inde-
pendent test set. We employ a threshold of 0.06 to distin-
guish well-fitted samples from under-fitted ones, enabling
coarse-grained assessment of chunk embedding accuracy.
For jina-embeddings-v2-small-en, 100% of test samples fall
below this threshold, with loss distribution highly concen-
trated in the 0.01–0.02 range, reflecting excellent general-
ization capability and minimal approximation error. In con-
trast, nomic-embed-text-v1.5 (137M parameters) exhibits
a more dispersed distribution: 96.7% of samples remain
below the 0.06 threshold, yet long-tail high-loss samples
emerge, indicating increased representation noise or model
uncertainty. BGE-M3 (570M parameters) shows further
distribution broadening, consistent with the training loss
patterns observed.
6. Experiments
6.1. Experimental Setup
DatasetsComprehensive experiments are conducted on
LongBench V2 (Bai et al., 2024), a challenging benchmark
specifically designed for long-context understanding tasks.
This dataset is suitable for evaluating chunking methods as
it contains documents with extensive contexts and covers
diverse task types with varying difficulty levels, thereby
providing a realistic testbed for assessing the effectiveness
of different chunking strategies in practical RAG scenarios.
Embedding ModelsTo ensure fair comparison, identi-
cal text encoding models are utilized across all methods.
Three embedding models are employed for comprehensive
evaluation: jina-embeddings-v2-small-en (33M parameters),
nomic-embed-text-v1.5 (137M parameters), and BGE-M3
(570M parameters). These models typically represent dif-
ferent scales and architectures, thereby enabling thorough
assessment of chunking performance across varying embed-
ding capacities.
Generative ModelFor text generation, Qwen3-8B (Team,
2025) is employed as the backbone language model, de-
5

FreeChunker: A Cross-Granularity Chunking Framework
ployed via vLLM for efficient inference. To guarantee fair
comparison and reproducibility, greedy decoding with tem-
perature set to 0 is employed by Qwen3-8B.
Baseline MethodsFreeChunkeris compared against sev-
eral representative chunking approaches.(1) Traditional
Chunkingrepresents the traditional approach that first splits
text by periods to preserve semantic integrity, then itera-
tively accumulates sentences until exceeding predefined
token limits (256 tokens);(2) Semantic Chunkingemploys
SemanticChunker (LangChain, 2024) to perform embed-
ding similarity-based chunking using sentence embeddings
to identify natural breakpoints in the text. The breakpoint
threshold type set to "percentile" and breakpoint thresh-
old amount set to 50.0;(3) Meta-Chunking(Zhao et al.,
2024) employs perplexity-based semantic boundary detec-
tion with two sub-methods: PPL Chunking and Margin
Sampling, utilizing Qwen/Qwen2.5-1.5B-Instruct model
(Yang et al., 2024) for text chunking. PPL Chunking adopts
the threshold of 0.5 as recommended in the original paper,
while Margin Sampling uses default settings;(4) Lumber-
Chunker(Duarte et al., 2024) provides LLM-guided chunk
boundary prediction; however, due to excessive process-
ing time for long texts, it is excluded from comparisons
involving lengthy documents.
6.2. Main Results
Table 1 presents the results across three embedding mod-
els on LongBench V2.FreeChunkerdemonstrates superior
performance in complex query environments while achiev-
ing significant computational efficiency gains compared to
traditional approaches.
Regarding computational efficiency,FreeChunkermaintains
processing time comparable to traditional chunking methods
while delivering superior retrieval performance. In contrast,
existing semantic chunking approaches require significantly
longer processing time without achieving comparable per-
formance gains. This combination of enhanced effectiveness
and computational efficiency validates the core contributions
of the cross-granularity paradigm in practical deployment
scenarios.
7. Conclusion
This paper introducesFreeChunker, a novel cross-
granularity chunking framework that fundamentally trans-
forms the traditional text chunking paradigm. By shifting
from static boundary identification to flexible semantic com-
bination retrieval, the proposed method significantly reduces
computational overhead associated with semantic analysis
for boundary detection while providing enhanced adaptabil-
ity to diverse query requirements. Unlike conventional static
chunking approaches, this flexible semantic combinationstrategy accommodates varying and unpredictable chunking
demands across different query intents.
Beyond the adjacent sentence combination approach ex-
plored in our experiments, the framework’s chunk pattern
control mechanism Pg,s[i, j] enables non-contiguous sen-
tence linking capabilities. This opens promising research
directions for incorporating predefined network structures
within document collections to construct sophisticated non-
adjacent sentence connections. Such extensions represent
compelling avenues for future investigation, potentially en-
abling more complex semantic relationship modeling across
document structures.
The experimental results demonstrate thatFreeChunker
achieves competitive retrieval performance while maintain-
ing computational efficiency comparable to baseline meth-
ods. The framework’s ability to provide multiple granularity
levels simultaneously addresses fundamental limitations of
existing single-granularity approaches, offering enhanced
flexibility for diverse information retrieval scenarios in prac-
tical RAG applications.
References
Bai, Y ., Tu, S., Zhang, J., Peng, H., Wang, X., Lv, X.,
Cao, S., Xu, J., Hou, L., Dong, Y ., Tang, J., and Li,
J. Longbench v2: Towards deeper understanding and
reasoning on realistic long-context multitasks.arXiv
preprint arXiv:2412.15204, 2024.
Brådland, H., Goodwin, M., Andersen, P.-A., Nossum, A. S.,
and Gupta, A. A new hope: Domain-agnostic automatic
evaluation of text chunking. InProceedings of the 48th
International ACM SIGIR Conference on Research and
Development in Information Retrieval (SIGIR ’25), pp.
1–10, Padua, Italy, 2025. ACM. ISBN 979-8-4007-1592-
1. doi: 10.1145/3726302.3729882. URL https://
doi.org/10.1145/3726302.3729882 . © 2025
Copyright held by the owner/author(s). Licensed under
CC BY 4.0.
Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan,
J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G.,
Askell, A., Agarwal, S., Herbert-V oss, A., Krueger, G.,
Henighan, T., Child, R., Ramesh, A., Ziegler, D. M., Wu,
J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M.,
Gray, S., Chess, B., Clark, J., Berner, C., McCandlish,
S., Radford, A., Sutskever, I., and Amodei, D. Language
models are few-shot learners. In Larochelle, H., Ranzato,
M., Hadsell, R., Balcan, M., and Lin, H. (eds.),Advances
in Neural Information Processing Systems 33: Annual
Conference on Neural Information Processing Systems
2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.
Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., and
6

FreeChunker: A Cross-Granularity Chunking Framework
Liu, Z. Bge m3-embedding: Multi-lingual, multi-
functionality, multi-granularity text embeddings through
self-knowledge distillation, 2024.
Duarte, A. V ., Marques, J., Graça, M., Freire, M., Li, L., and
Oliveira, A. L. Lumberchunker: Long-form narrative doc-
ument segmentation. In Al-Onaizan, Y ., Bansal, M., and
Chen, Y . (eds.),Findings of the Association for Computa-
tional Linguistics: EMNLP 2024, Miami, Florida, USA,
November 12-16, 2024, pp. 6473–6486. Association for
Computational Linguistics, 2024.
Gao, L., Biderman, S., Black, S., Golding, L., Hoppe, T.,
Foster, C., Phang, J., He, H., Thite, A., Nabeshima, N.,
et al. The pile: An 800gb dataset of diverse text for
language modeling.arXiv preprint arXiv:2101.00027,
2020.
Günther, M., Ong, J., Mohr, I., Abdessalem, A., Abel, T.,
Akram, M. K., Guzman, S., Mastrapas, G., Sturua, S.,
Wang, B., Werk, M., Wang, N., and Xiao, H. Jina embed-
dings 2: 8192-token general-purpose text embeddings for
long documents, 2023.
Huang, L., Yu, W., Ma, W., Zhong, W., Feng, Z., Wang, H.,
Chen, Q., Peng, W., Feng, X., Qin, B., et al. A survey on
hallucination in large language models: Principles, taxon-
omy, challenges, and open questions.ACM Transactions
on Information Systems, 43(2):1–55, 2025.
Jacobs, R. A., Jordan, M. I., Nowlan, S. J., and Hinton, G. E.
Adaptive mixtures of local experts.Neural computation,
3(1):79–87, 1991.
Jordan, M. I. and Jacobs, R. A. Hierarchical mixtures of
experts and the em algorithm.Neural computation, 6(2):
181–214, 1994.
LangChain. Semantic chunker. https:
//python.langchain.com/docs/how_to/
semantic-chunker/, 2024. Accessed: 2025-06-06.
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V .,
Goyal, N., Küttler, H., Lewis, M., Yih, W.-t., Rocktäschel,
T., et al. Retrieval-augmented generation for knowledge-
intensive nlp tasks.Advances in neural information pro-
cessing systems, 33:9459–9474, 2020.
Li, J., Cheng, X., Zhao, X., Nie, J., and Wen, J. Halueval: A
large-scale hallucination evaluation benchmark for large
language models. In Bouamor, H., Pino, J., and Bali, K.
(eds.),Proceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing, EMNLP 2023,
Singapore, December 6-10, 2023, pp. 6449–6464. Associ-
ation for Computational Linguistics, 2023. doi: 10.18653/
V1/2023.EMNLP-MAIN.397. URL https://doi.
org/10.18653/v1/2023.emnlp-main.397.Lyu, Y ., Li, Z., Niu, S., Xiong, F., Tang, B., Wang, W.,
Wu, H., Liu, H., Xu, T., and Chen, E. Crud-rag: A com-
prehensive chinese benchmark for retrieval-augmented
generation of large language models.ACM Transactions
on Information Systems, 43(2):1–32, 2025.
Nussbaum, Z., Morris, J. X., Duderstadt, B., and Mulyar, A.
Nomic embed: Training a reproducible long context text
embedder, 2024.
Qu, R., Tu, R., and Bao, F. S. Is semantic chunking
worth the computational cost? In Chiruzzo, L., Rit-
ter, A., and Wang, L. (eds.),Findings of the Associa-
tion for Computational Linguistics: NAACL 2025, Albu-
querque, New Mexico, USA, April 29 - May 4, 2025, pp.
2155–2177. Association for Computational Linguistics,
2025. URL https://aclanthology.org/2025.
findings-naacl.114/.
Ru, D., Qiu, L., Hu, X., Zhang, T., Shi, P., Chang, S., Ji-
ayang, C., Wang, C., Sun, S., Li, H., Zhang, Z., Wang, B.,
Jiang, J., He, T., Wang, Z., Liu, P., Zhang, Y ., and Zhang,
Z. Ragchecker: A fine-grained framework for diagnos-
ing retrieval-augmented generation. In Globersons, A.,
Mackey, L., Belgrave, D., Fan, A., Paquet, U., Tomczak,
J. M., and Zhang, C. (eds.),Advances in Neural Infor-
mation Processing Systems 38: Annual Conference on
Neural Information Processing Systems 2024, NeurIPS
2024, Vancouver, BC, Canada, December 10 - 15, 2024,
2024.
Team, Q. Qwen3 technical report, 2025. URL https:
//arxiv.org/abs/2505.09388.
Yang, A., Yang, B., Hui, B., Zheng, B., Yu, B., Zhou, C.,
Li, C., Li, C., Liu, D., Huang, F., Dong, G., Wei, H., Lin,
H., Tang, J., Wang, J., Yang, J., Tu, J., Zhang, J., Ma, J.,
Xu, J., Zhou, J., Bai, J., He, J., Lin, J., Dang, K., Lu, K.,
Chen, K., Yang, K., Li, M., Xue, M., Ni, N., Zhang, P.,
Wang, P., Peng, R., Men, R., Gao, R., Lin, R., Wang, S.,
Bai, S., Tan, S., Zhu, T., Li, T., Liu, T., Ge, W., Deng,
X., Zhou, X., Ren, X., Zhang, X., Wei, X., Ren, X., Fan,
Y ., Yao, Y ., Zhang, Y ., Wan, Y ., Chu, Y ., Liu, Y ., Cui, Z.,
Zhang, Z., and Fan, Z. Qwen2 technical report.arXiv
preprint arXiv:2407.10671, 2024.
Zhao, J., Ji, Z., Qi, P., Niu, S., Tang, B., Xiong, F., and Li,
Z. Meta-chunking: Learning efficient text segmentation
via logical perception.CoRR, abs/2410.12788, 2024. doi:
10.48550/ARXIV .2410.12788. URL https://doi.
org/10.48550/arXiv.2410.12788.
Zhao, J., Ji, Z., Fan, J. Z., Wang, H., Niu, S., Tang, B.,
Xiong, F., and Li, Z. Moc: Mixtures of text chunk-
ing learners for retrieval-augmented generation system.
CoRR, abs/2503.09600, 2025. doi: 10.48550/ARXIV .
7

FreeChunker: A Cross-Granularity Chunking Framework
2503.09600. URLhttps://doi.org/10.48550/
arXiv.2503.09600.
Zhong, Z., Liu, H., Cui, X., Zhang, X., and Qin, Z. Mix-
of-granularity: Optimize the chunking granularity for
retrieval-augmented generation. In Rambow, O., Wanner,
L., Apidianaki, M., Al-Khalifa, H., Eugenio, B. D., and
Schockaert, S. (eds.),Proceedings of the 31st Interna-
tional Conference on Computational Linguistics, COL-
ING 2025, Abu Dhabi, UAE, January 19-24, 2025, pp.
5756–5774. Association for Computational Linguistics,
2025.
8