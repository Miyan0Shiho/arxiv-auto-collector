# SCAR: Semantic Continuity-Aware Retrieval for Efficient Context Expansion in RAG

**Authors**: Nathanaël Langlois

**Published**: 2026-06-15 12:55:49

**PDF URL**: [https://arxiv.org/pdf/2606.16661v1](https://arxiv.org/pdf/2606.16661v1)

## Abstract
Fixed-length chunking in Retrieval-Augmented Generation (RAG) often leads to boundary fragmentation, where critical evidence is split across segments, degrading retrieval recall. While static windowing and parent retrieval improve recall, they introduce significant token overhead. We propose SCAR (Semantic Continuity-Aware Retrieval), an adaptive retrieval policy that selectively expands neighboring chunks by weighing query-neighbor relevance against a structural continuity penalty. SCAR uses a relative expansion threshold tied to each retrieved chunk's own query-relevance, yielding an approximately scale-invariant decision rule that transfers across embedding models without recalibration. Across four diverse corpora (RFC, GDPR, a 10-K report, and a Merger agreement; N=320 queries; 160 boundary-fragmented), SCAR achieves 92.8% recall on boundary-fragmented queries with only 7.84 chunks, a 22.9% reduction compared to static windowing (10.16 chunks). Paired bootstrap tests (B=10,000) confirm the chunk reduction is highly significant (p<0.0001, Cohen's d=-1.49, large effect), with a small recall difference (Cohen's d=-0.33). The policy transfers across three embedding models (text-embedding-3-large, BGE-large-en-v1.5, zembed-1) using the same single hyperparameter setting, and downstream RAGAS evaluation on the 10-K corpus confirms SCAR preserves generation faithfulness while reducing context tokens by 27.1%.

## Full Text


<!-- PDF content starts -->

SCAR: Semantic Continuity-Aware Retrieval for Efficient Context
Expansion in RAG
Nathanaël Langlois
Horizon Flow
Abstract
Fixed-length chunking in Retrieval-Augmented Generation (RAG)
often leads to boundary fragmentation, where critical evidence
is split across segments, degrading retrieval recall. While static
windowing and parent retrieval improve recall, they introduce sig-
nificant token overhead. We proposeSCAR(Semantic Continuity-
Aware Retrieval), an adaptive retrieval policy that selectively ex-
pands neighboring chunks by weighing query-neighbor relevance
against a structural continuity penalty. SCAR uses a relative expan-
sion threshold tied to each retrieved chunk’s own query-relevance,
yielding an approximately scale-invariant decision rule that trans-
fers across embedding models without recalibration. Across four
diverse corpora (RFC, GDPR, a 10-K report, and a Merger agree-
ment;𝑁= 320queries; 160 boundary-fragmented), SCAR achieves
92.8% recall on boundary-fragmented queries with only 7.84 chunks,
a 22.9% reduction compared to static windowing (10.16 chunks).
Paired bootstrap tests (B=10,000) confirm the chunk reduction is
highly significant ( 𝑝< 0.0001, Cohen’s 𝑑=− 1.49, large effect),
with a small recall difference (Cohen’s 𝑑=− 0.33). The policy trans-
fers across three embedding models ( text-embedding-3-large ,
BGE-large-en-v1.5 ,zembed-1 ) using the same single hyperpa-
rameter setting, and downstream RAGAS evaluation on the 10-K
corpus confirms SCAR preserves generation faithfulness while re-
ducing context tokens by 27.1%.
CCS Concepts
•Information systems →Document representation;Retrieval
models and ranking;Retrieval efficiency.
Keywords
RAG, Information Retrieval, Context Expansion, Efficiency
1 Introduction
Retrieval-Augmented Generation (RAG) has emerged as the stan-
dard for grounding Large Language Models (LLMs) in external
knowledge. However, document chunking remains a fundamental
bottleneck in the retrieval pipeline. Standard strategies, such as
recursive character splitting, prioritize local structural integrity
but create a rigid, static index. This leads to the boundary frag-
mentation problem, where critical evidence required for a specific
query is split across adjacent segments. Because optimal chunking
is query-dependent and constrained by the inherent structure of
the source documents, even advanced paragraph-aware splitters
cannot prevent logical context breaks at runtime.
2 Problem Statement
In technical, legal, financial, or other complex structured documents,
critical evidence often spans across standard indexing boundaries.
When a retrieval system returns only one fragment of a bifurcatedlogical unit (e.g., the header of a financial table without its corre-
sponding rows, or the first half of a legal definition), the downstream
LLM suffers from incomplete context.
Boundary Fragmentationoccurs when the evidence 𝐸required
to answer query 𝑄is split such that 𝐸={𝑐 𝑖,𝑐𝑖+1}, but the retriever
only identifies 𝑐𝑖as relevant. While a dense retriever may occasion-
ally capture both chunks in a Top- 𝑘result, it offers no guarantee
of logical continuity. Standard solutions likeStatic Windowing
(always retrieving 𝑐𝑖±1) orParent Retrieval(retrieving the en-
tire section) are token-inefficient. These methods often inflate the
context window with irrelevant "noise," which can degrade LLM
performance and increase inference costs.
3 Related Work
Existing approaches to boundary fragmentation cluster into two
camps.Indexing-time coherencemethods build semantically
continuous units before retrieval (e.g.,TextTiling[ 4] via lexi-
cal cohesion,Max–Min semantic chunking[ 6] via sentence-
embedding similarity,Late Chunking[ 1] via long-context con-
textualized embeddings, andSentGraph[ 7] via hierarchical sen-
tence graphs), but fix boundaries at indexing time and cannot adapt
to query-specific informational density.Retrieval-time expan-
sionmethods complement this. Static heuristics (fixed sliding win-
dows, parent–child retrieval) deterministically include neighbors
and are token-inefficient. Iterative methods (FLARE[ 5],Self-RAG
[2],DIVER[ 8]) interleave retrieval with generation or rerank-
ing through repeated LLM calls. Utility-trained retrievers such as
SCARLet[ 9] improve quality but require specialized data synthe-
sis and training. SCAR fills the gap as a single-shot, training-free
retrieval-time policy that adaptively expands only when query rel-
evance and structural continuity jointly justify it, and it composes
orthogonally with any indexing-time chunking strategy.
4 Methodology: SCAR
We introduceSCAR(Semantic Continuity-Aware Retrieval), an
adaptive retrieval policy designed to selectively recover fragmented
logical context at runtime.
4.1 Continuity Score
For a retrieved chunk 𝑐and a candidate neighbor 𝑛, we define a
boundary penalty:
𝑏𝑐,𝑛=1−cos(𝑒 𝑐,𝑒𝑛)(1)
where𝑒𝑐,𝑒𝑛are the chunk embeddings. The boundary penalty 𝑏𝑐,𝑛
quantifies the semantic discontinuity between adjacent chunks; a
higher cosine similarity indicates stronger continuity.arXiv:2606.16661v1  [cs.IR]  15 Jun 2026

Preprint, 2026, Nathanaël Langlois
4.2 Expansion Decision
We define the expansion score for a neighbor𝑛of𝑐:
𝑆𝑐,𝑛=cos(𝑒 𝑞,𝑒𝑛)−𝜆𝑏 𝑐,𝑛 (2)
where𝑒𝑞is the query embedding and 𝜆is a hyperparameter
controlling the strength of the continuity penalty.
A neighbor 𝑛is appended to the context if its expansion score
exceeds arelativethreshold tied to the retrieved chunk’s own query-
relevance:
𝑆𝑐,𝑛>𝛾·cos(𝑒 𝑞,𝑒𝑐)(3)
where𝛾∈( 0,1)is a ratio hyperparameter. Intuitively, we expand 𝑛
if, after the continuity penalty, it is at least 𝛾as query-relevant as
the chunk it is adjacent to.
This formulation has two design properties.Scale robustness:
the relevance terms on both sides of Eq. 3 scale with query-chunk
similarity, so the decision is approximately robust to the absolute
similarity scale of the embedding model (exactly so in the 𝜆=0limit;
for small𝜆the continuity penalty introduces only a minor scale-
dependent correction), largely removing the need for per-model
threshold recalibration, which we confirm empirically in Section 6.5.
Adaptive thresholding:the expansion bar adapts per retrieved
chunk, since a weakly-retrieved chunk (low cos(𝑒 𝑞,𝑒𝑐)) triggers
a correspondingly lower threshold, which is desirable when the
retriever is itself uncertain.
5 Experimental Setup
5.1 Datasets and Preprocessing
We evaluate SCAR on four diverse corpora selected for their com-
plex structural dependencies: the TCP specification (RFC 9293), the
GDPR regulation, a Microsoft 10-K annual report, and a corporate
merger agreement (Table 1). To prepare the documents, we utilize
a hierarchical chunking strategy withContextual Prepending.
We employ a recursive character splitter with a target chunk size
of 600 characters and a 60-character overlap.
For each logical segment, we extract the structural hierarchy
(e.g.,Section > Subsection) and prepend this metadata to the text
fragment before embedding. This ensures that each chunk carries
latent structural information, providing a rigorous baseline for our
adaptive expansion policy.
Table 1: Statistics of evaluation corpora.
Corpus Domain Total Chunks
TCP Technical 273
GDPR Legal 797
Microsoft 10-K Financial 654
Merger Contractual 461
5.2 Retrieval and Evaluation
We employ the text-embedding-3-large model to generate 3072-
dimensional embeddings for all chunks and queries. Retrieval is
performed using𝑘-NN search with𝑘=5and cosine similarity.To evaluate SCAR, we generated 80 queries per corpus ( 𝑁= 320)
using an LLM-assisted pipeline, followed by expert manual verifica-
tion and a three-rater inter-annotator-agreement study (described
below). This set is balanced between:
•Atomic Queries (40 per corpus): Information is contained
within a single chunk.
•Boundary-Fragmented Queries (40 per corpus): Infor-
mation spans two or more contiguous chunks, requiring
logical expansion for a complete answer.
Each query is paired with a set ofgold chunk IDsfor ground-
truth verification. Our primary metric isRecall, defined as the
proportion of gold chunks successfully recovered within the dy-
namically expanded context window. We evaluate this alongside
Chunk Efficiency, defined as Recall divided by the average num-
ber of unique chunks retrieved per query: higher is better, capturing
recall achieved per unit of context volume. Our code and evaluation
datasets are available at https://github.com/scarmethod/SCAR
Annotation Quality and Inter-Annotator Agreement.To validate the
LLM-generated gold labels, three annotators independently iden-
tified relevant chunks without access to the gold: one labeled the
full query set (N=77) and two labeled overlapping ∼20-query sub-
sets. Per-chunk Cohen’s 𝜅against the gold was 0.71 and 0.79 for
the two careful annotators (both substantial) and 0.55 for a third
rapid pass; 91% of queries were rated well-formed. Adjudication
produced three benchmark corrections (two gold-set edits, one
query reformulation), applied in all reported results.
5.3 Baselines
We compare SCAR against:
•Baseline (Top-𝑘): Retrieve only the top-𝑘chunks
•Window (±1): Always retrieve immediate neighbors
•Window (±2): Expand to±2neighbors
•Parent Retrieval: Retrieve all chunks from the same docu-
ment section
•Cross-Encoder Reranker: Rerank the top-20 dense candi-
dates withbge-reranker-largeand keep the top-𝑛
5.4 Hyperparameters
SCAR has two hyperparameters: the boundary-penalty weight 𝜆
and the relative-threshold ratio 𝛾from Eq. 3. We fix 𝜆=0.1and
𝛾=0.80across all datasets and embedding models. We evaluate
SCAR at expansion radius 𝑟∈{ 1,2}. We verify the stability of these
choices through a leave-one-corpus-out grid search and confirm
that𝛾=0.80transfers across embedding models without recalibra-
tion (Section 6.5).
6 Results and Discussion
6.1 Main Results: Boundary-Fragmented
Queries
Table 2 evaluates SCAR against the static windowing baseline on
queries where evidence is split across segments. SCAR ( ±1) achieves
92.8% average recall using 7.84 unique chunks on average. In com-
parison, the Static Window baseline requires 10.16 chunks: a 22.9%
reduction in context volume with a 3.9 percentage point recall

SCAR: Semantic Continuity-Aware Retrieval for Efficient Context Expansion in RAG Preprint, 2026,
gap. Against naive fixed-budget retrieval, SCAR’s selectivity is also
evident: a top-8 retriever reaches only 0.905 recall (2.3 points be-
low SCAR), and matching SCAR’s 0.928 needs top-10 (about 28%
more chunks). The savings reflect adaptive selection, not a smaller
candidate set.
The gain is most pronounced on the Microsoft 10-K corpus (29.9%
chunk reduction, 7.47 vs 10.65, at 92.3% recall vs Window’s 99.2%),
where SCAR filters the redundant “neighbor noise” common in
dense financial tables.
Table 2: Recall and average chunk volume on boundary-
fragmented queries ( 𝑁= 160,𝑘= 5,𝜆= 0.1,𝛾= 0.80). W
= Window (±1), S = SCAR (±1); chunk counts are unique seg-
ments after deduplication.
Recall Chunks
Dataset W S W S
TCP 0.992 0.950 9.75 7.90
GDPR 0.941 0.926 10.57 8.10
Microsoft 10-K 0.992 0.923 10.65 7.47
Merger 0.943 0.914 9.68 7.88
Average 0.967 0.928 10.16 7.84
Figure 1 shows Pareto curves demonstrating the recall-chunk
tradeoff as the expansion threshold varies. SCAR’s adaptive expan-
sion consistently provides superior token efficiency at comparable
recall to static windowing across the efficiency frontier. The gold
star marks our chosen operating point under the relative-threshold
formulation (𝛾=0.80).
Figure 1: Recall-chunk Pareto curves for SCAR- 𝛾(±1) as𝛾
varies in[0.60,0.90](𝑁= 40boundary-fragmented queries
per corpus). The gold star marks our 𝛾= 0.80default, the
orange square static Window ( ±1), and the gray triangle the
no-expansion Top-5 baseline. SCAR dominates the frontier:
at comparable recall, it uses fewer chunks; at comparable
chunk count, it achieves higher recall.6.2 Method Comparison and Expansion Radius
Table 3 compares all methods on the boundary-fragmented set. To
assess whether SCAR’s adaptive policy justifies larger expansion
radii, we compare SCAR ( ±2) against Window ( ±2): while Window
(±2) blindly retrieves all neighbors (14.4 chunks average), SCAR
(±2) selectively expands only to semantically continuous chunks
(9.2 vs 14.4), a 36% chunk reduction at only 3.6 percentage points
lower recall (0.949 vs 0.985), translating to roughly 50% higher
chunk efficiency.
Table 3: Efficiency comparison on boundary-fragmented
queries (𝑁= 160,𝑘= 5,𝜆= 0.1,𝛾= 0.80). The reranker
keeps the top-8 of 20 reranked candidates, matching SCAR’s
chunk budget.
Method Recall Chunks Efficiency
Baseline (Top-𝑘) 0.830 5.0 0.166
SCAR (±1) 0.928 7.84 0.118
Window (±1) 0.967 10.16 0.095
SCAR (±2) 0.949 9.2 0.103
Window (±2) 0.985 14.4 0.069
Reranker→top-8 0.854 8.0 0.107
Parent 0.968 50.2 0.019
SCAR dominates the efficiency frontier at both radii, offering
recall-chunk tunability that static methods lack.
Beyond expansion: cross-encoder reranking.As a non-expansion
alternative, we reranked the top-20 dense candidates with a widely-
used cross-encoder ( bge-reranker-large ) and kept the top-8. Al-
though97.2%of gold chunks lie within the candidate pool (Re-
call@20), the reranker reaches only0 .854recall at top-8 and0 .878
at top-10 (Table 3): above naive top-5 dense (0 .830) but well short
of SCAR’s0 .928at the same budget. Even a high-quality point-
wise scorer tends to demotecontinuationchunks, whose standalone
query relevance is lower than their keyword-bearing boundary
partners. This suggests the gap likely stems from the ranking objec-
tive rather than candidate availability: adjacent-boundary recovery
appears to benefit from structural-continuity reasoning beyond
point-wise ranking alone.
SCAR’s expansion is also query-dependent: on atomic queries
(𝑁= 160), SCAR retrieves 7.6 chunks on average versus Window’s
11.3, a 58.2% reduction inunnecessaryexpansion above the Top-5
baseline, providing a “zero-regret” profile when expansion is not
required.
6.3 Ablation Study
To evaluate the individual contribution of the continuity penalty
𝑏𝑐,𝑛, we compare the full SCAR policy ( 𝜆=0.1) against aRelevance-
Onlyexpansion policy ( 𝜆=0). This ablation determines whether
query relevance alone is sufficient for selective expansion or if
the structural continuity signal is required to prune redundant
neighbors.
As shown in Table 4, removing the continuity penalty increases
chunk volume by up to 7.2% with little recall change. The full SCAR
policy consistently reduces overhead across all four corpora. This

Preprint, 2026, Nathanaël Langlois
confirms that the continuity penalty acts as an effective semantic fil-
ter, suppressing irrelevant adjacent content. Finally, embedding raw
chunk textwithoutthe prepended structural metadata preserves
SCAR’s chunk-efficiency advantage (37.1% reduction vs Window,
versus 22.9% with metadata), confirming the continuity penalty
reflects content continuity rather than breadcrumb overlap.
Table 4: Ablation Study: Impact of Continuity Penalty ( 𝜆)
at𝑘= 5,𝛾= 0.80,𝑁= 160boundary-fragmented queries.
Chunks represent the average unique segments retrieved.
Recall Avg. Chunks
Dataset𝜆=0𝜆=0.1𝜆=0𝜆=0.1
TCP 0.967 0.950 8.35 7.90
GDPR 0.941 0.926 8.68 8.10
Microsoft 10-K 0.944 0.923 7.95 7.47
Merger 0.928 0.914 8.25 7.88
6.4 Statistical Significance
We test whether the observed chunk-volume and recall differences
between SCAR (±1) and Window (±1) are statistically reliable. Be-
cause each query is evaluated under both methods on the same
retrieval state, all comparisons arepaired.
For each query 𝑞we record the per-query paired differences
Δchunks
𝑞 =|𝑆 𝑞|−|𝑊 𝑞|andΔrecall
𝑞=rec 𝑆(𝑞)−rec 𝑊(𝑞). We compute: (i)
a paired bootstrap test with 𝐵=10,000resamples of the pair indices,
yielding both a two-sided 𝑝-value and a 95% confidence interval;
(ii) a Wilcoxon signed-rank test as a non-parametric complement;
and (iii) Cohen’s 𝑑on the paired differences as a standardized effect
size.
On the boundary-fragmented set ( 𝑁= 160), the chunk reduction
is highly significant: mean difference −2.32chunks, 95% bootstrap
CI[−2.57,−2.09],𝑝boot<10−4,𝑝wilcox <10−4, Cohen’s𝑑=− 1.49
(large effect). Per-corpus tests all yield 𝑝< 10−4on chunks. The
recall difference is smaller: mean −0.039, 95% CI[−0.058,−0.022],
𝑝boot<10−4,𝑝wilcox =0.011, Cohen’s 𝑑=− 0.33(small effect).
Together, these results indicate that SCAR’s chunk savings are a
large, robust effect, while the recall cost is measurable but small.
Cross-Embedding-Model Transfer.The same default (𝜆=
0.1,𝛾= 0.80)transfers without recalibration to two additional
embedding models with different dimensionalities and similarity
distributions: BGE-large-en-v1.5 (12.7% chunk reduction at 0.913
recall) and zembed-1 (17.2% chunk reduction at 0.956 recall), con-
firming that the relative-threshold formulation transfers across
similarity scales without recalibration.
6.5 Hyperparameter Robustness
Sweeping𝜆∈[ 0,0.3]at𝛾=0.80yields a smooth, monotonic trade-
off (mean recall drops only 1.7 points over [0,0.20]);𝜆=0.1sits
near the efficiency elbow, with any value in [0.05,0.15]compara-
ble. A leave-one-corpus-out (𝜆,𝛾) grid search places the default
(𝜆= 0.1,𝛾= 0.80)in a broad plateau where mean recall is ≥0.90;
CV-selected parameters trade chunks against recall inconsistentlyacross held-out corpora, confirming the fixed default does not ex-
ploit per-corpus peculiarities.
6.6 Robustness to Gold-Label Noise
To rule out the possibility that SCAR’s advantage stems from LLM-
generated gold favoring its heuristic, we re-evaluated every method
under four alternative gold definitions derived from the human
annotations (replace, permissive union, conservative intersection,
and 2-of-3 majority on the triple-overlap subset). Across all variants,
SCAR’s recall lead over the Top- 𝑘baseline on boundary-fragmented
queries remains positive (gap range +0.034to+0.098), and SCAR
continues to match or approach Window ±1recall while using fewer
chunks. The annotation imperfections surfaced by IAA propagate
symmetrically to all methods, so SCAR’s measured advantage is
not attributable to gold-label noise.
6.7 Downstream Generation Quality
Retrieval recall is a proxy for downstream usefulness; ultimately
what matters is whether a generator can answer correctly from
SCAR’s context. We evaluate on all 80 queries of the Microsoft
10-K corpus using GPT-4o-mini as the generator and an LLM-as-
judge protocol following RAGAS [ 3] for faithfulness and answer
relevancy.
Compared with Window ( ±1), SCAR (±1) produces essentially
identical faithfulness (4.99/5 vs. 4.99/5), slightly higher answer rele-
vancy (4.79 vs. 4.74), and a markedly smaller context: 934 tokens on
average vs. 1,281, a 27.1% token reduction. Context precision im-
proves from 0.17 to 0.23 while context recall (fraction of gold chunks
present) remains at 0.99. These results indicate that SCAR’s leaner
context does not degrade grounded-answer quality at generation
time.
7 Conclusion
SCAR is a training-free, retrieval-time policy that selectively ex-
pands context by weighing query-neighbor relevance against a
structural continuity penalty, with a relative threshold that trans-
fers across embedding models without recalibration. Across 160
boundary-fragmented queries it cuts chunk volume by 22.9% versus
static windowing ( 𝑝< 10−4,𝑑=− 1.49) at a 3.9-point recall cost,
while RAGAS on the 10-K corpus confirms faithfulness is preserved
at a 27.1% smaller context, making it a drop-in efficiency upgrade
for production RAG. Its scope is adjacent-chunk fragmentation: it
cannot recover evidence the retriever fails to rank in top- 𝑘, nor non-
adjacent evidence; learned continuity penalties and non-adjacent
logical-neighbor expansion are natural next directions. A further
limitation is that SCAR assumes a needed adjacent chunk retains
enough standalone query relevance to pass the expansion gate:
for semantically thin continuations (e.g., table values separated
from their headers, or paraphrased restatements), the neighbor’s
low direct relevance can cause SCAR to skip evidence that static
windowing would recover. Structure-aware chunk construction
(contextual prepending, header/row carryover) mitigates this and
is complementary to SCAR.

SCAR: Semantic Continuity-Aware Retrieval for Efficient Context Expansion in RAG Preprint, 2026,
GenAI Usage Disclosure
Generative-AI components are part of the retrieval pipeline under
evaluation: text-embedding-3-large ,BGE-large-en-v1.5 , and
zembed-1 produce the dense chunk and query embeddings, and
the 320-query evaluation benchmark was bootstrapped using an
LLM-assisted query-generation pipeline, followed by expert manual
verification and the three-rater inter-annotator-agreement study
described in Section 5. The downstream-generation evaluation in
Section 6.7 uses GPT-4o-mini as the generator under an LLM-as-
judge (RAGAS) protocol.
Generative-AI assistants were additionally used to draft and
revise prose passages and to suggest code refactors during anal-
ysis. All experimental design, hyperparameter choices, statistical
methodology, dataset curation, annotation procedures, manual ad-
judication of inter-annotator disagreements, and final scientific
claims were formulated and verified by the human authors.
References
[1]Jina AI. 2024. Late Chunking: Contextual Chunk Embeddings. https://jina.ai/
news/late-chunking-in-long-context-embedding-models/.
[2]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024.
Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. InThe Twelfth International Conference on Learning Representations. https://arxiv.
org/abs/2310.11511
[3]Shahul Es, Jithin James, Luis Espinosa-Anke, and Steven Schockaert.
2025. Ragas: Automated Evaluation of Retrieval Augmented Generation.
arXiv:2309.15217 [cs.CL] https://arxiv.org/abs/2309.15217
[4]Marti A. Hearst. 1997. TextTiling: segmenting text into multi-paragraph subtopic
passages.Comput. Linguist.23, 1 (March 1997), 33–64.
[5]Zhengbao Jiang, Frank Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu,
Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active Retrieval Augmented
Generation. InProceedings of the 2023 Conference on Empirical Methods in Natural
Language Processing, Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association
for Computational Linguistics, Singapore, 7969–7992. doi:10.18653/v1/2023.emnlp-
main.495
[6]Csaba Kiss, Marcell Nagy, and Péter Szilágyi. 2025. Max–Min semantic chunking
of documents for RAG application.Discover Computing28 (06 2025). doi:10.1007/
s10791-025-09638-7
[7]Junli Liang, Pengfei Zhou, Wangqiu Zhou, Wenjie Qing, Qi Zhao, Ziwen Wang,
Qi Song, and Xiangyang Li. 2026. SentGraph: Hierarchical Sentence Graph for
Multi-hop Retrieval-Augmented Question Answering. arXiv:2601.03014 [cs.CL]
https://arxiv.org/abs/2601.03014
[8]Meixiu Long, Duolin Sun, Dan Yang, Junjie Wang, Yecheng Luo, Yue Shen, Jian
Wang, Hualei Zhou, Chunxiao Guo, Peng Wei, Jiahai Wang, and Jinjie Gu. 2025.
DIVER: A Multi-Stage Approach for Reasoning-intensive Information Retrieval.
arXiv:2508.07995 [cs.IR] https://arxiv.org/abs/2508.07995
[9]Yilong Xu, Jinhua Gao, Xiaoming Yu, Yuanhai Xue, Baolong Bi, Huawei Shen, and
Xueqi Cheng. 2026. Training a Utility-based Retriever Through Shared Context
Attribution for Retrieval-Augmented Language Models. arXiv:2504.00573 [cs.CL]
https://arxiv.org/abs/2504.00573