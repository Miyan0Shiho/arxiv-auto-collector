# ViSAR: Training-Free Adaptive-$k$ Retrieval for Visual Document Question Answering

**Authors**: Adrien Mialland, Marc Plantevit, Julien Gallois, Céline Robardet

**Published**: 2026-09-02 11:54:58

**PDF URL**: [https://arxiv.org/pdf/2609.02486v2](https://arxiv.org/pdf/2609.02486v2)

## Abstract
Document Visual Question Answering (DocVQA) often leverages Retrieval-Augmented Generation (RAG), where late-interaction encoders are commonly used to identify document pages relevant to a user query, before answer generation by a Large Vision-Language Model (LVLM). Existing approaches typically retrieve a fixed top-$k$ number of pages regardless of query complexity, which increases LVLM latency and may degrade answer accuracy. We introduce ViSAR (Visual Semantic Activation Retrieval), a training-free adaptive-$k$ retrieval method for late-interaction visual document retrieval. ViSAR operates directly in the embedding space to construct a query-conditioned page-level similarity matrix that highlights query-relevant semantics and dynamically determines the number of pages to retrieve. Across multiple encoders and LVLMs, ViSAR retrieves compact, query-adapted page sets that reduce RAG latency by up to 58.7\%, while maintaining or improving answer accuracy compared with fixed top-$k$ and adaptive retrieval heuristics. Furthermore, we show that the similarity matrix structure correlates with answer accuracy, suggesting future directions for retrieval quality-aware document understanding.

## Full Text


<!-- PDF content starts -->

ViSAR: Training-Free Adaptive-kRetrieval for
Visual Document Question Answering
Adrien Mialland
INSA Lyon, CNRS, LIRIS UMR 5205, F-69621 Villeurbanne, France
mialland.a@gmail.com
Marc Plantevit
EPITA Research Laboratory (LRE), FR-94276, Le Kremlin-Bicêtre, France
marc.plantevit@epita.fr
Julien Gallois
Lowit, FR-69003 Lyon, France
julien.gallois@lowit.fr
Céline Robardet
INSA Lyon, CNRS, LIRIS UMR 5205, F-69621 Villeurbanne, France
celine.robardet@insa-lyon.fr
Abstract
Document Visual Question Answering (DocVQA) often leverages Retrieval-
Augmented Generation (RAG), where late-interaction encoders are commonly
used to identify document pages relevant to a user query, before answer gen-
eration by a Large Vision-Language Model (LVLM). Existing approaches typ-
ically retrieve a fixed top- knumber of pages regardless of query complexity,
which increases LVLM latency and may degrade answer accuracy. We intro-
duce ViSAR (Visual Semantic Activation Retrieval), a training-free adaptive- k
retrieval method for late-interaction visual document retrieval. ViSAR operates
directly in the embedding space to construct a query-conditioned page-level simi-
larity matrix that highlights query-relevant semantics and dynamically determines
the number of pages to retrieve. Across multiple encoders and LVLMs, ViSAR
retrieves compact, query-adapted page sets that reduce RAG latency by up to
58.7%, while maintaining or improving answer accuracy compared with fixed
top-kand adaptive retrieval heuristics. Furthermore, we show that the similar-
ity matrix structure correlates with answer accuracy, suggesting future directions
for retrieval quality-aware document understanding. Our code is available at
https://github.com/adrienmialland/ViSAR.
Keywords—Document Visual Question Answering, Large Language Model, Retrieval-Augmented Generation
1 Introduction
Document Visual Question Answering (DocVQA) aims to answer a user query about visually rich documents
using page screenshots, and has attracted growing interest across many applications [ 1–5]. These documents
often include text, images, charts, or tables that blend into document-specific layouts, requiring methods capable
arXiv:2609.02486v2  [cs.IR]  3 Sep 2026

⋯
𝑞1𝑞2𝑞m
MaxSim
Page-level similarity matrixQuery semantic localisation Query-conditionned
Sim(p, p')
1.0
0.50
0.0
Page p0
6
12
18
0 6 12 18
Page p'Page p
∑
⋯
𝑃1S𝑃2S𝑃𝑁SFixed top-k retrieval generation
wrong answer
increased latency
Adaptive-k retrieval
generation
correct answer
reduced latency⋯𝑃1
𝑃𝑁𝑃2⋯𝑞1𝑞2𝑞m
no encoder training - operates in the embeddings spaceApplied to each page individuallyLate-Interaction:
Cannot adapt the selection to the querya
bFigure 1: Document page retrieval mechanisms. (a) Late-interaction fixed top- k. (b) The proposed
ViSAR adaptive- kmethod. While late-interaction enables fine-grained query-page matching using
multi-vector representations, it relies on a fixed top- kretrieval that cannot adapt to the query, intro-
ducing irrelevant pages and unnecessary latency. ViSAR leverages these multi-vector representations
without additional training to enable adaptive- kretrieval, reducing irrelevant pages, lowering latency,
and improving answer accuracy.
of jointly understanding the query and heterogeneous visual content to localise relevant information and generate
an answer. DocVQA is therefore commonly addressed using Large Vision-Language Models (LVLMs) [ 6–8],
which reason over textual and visual modalities jointly. Recent progress has demonstrated close-to-human
performance in single-page DocVQA [ 9–11], but effectively processing multiple-page documents requires
processing only the relevant subset of pages.
In this setting, Retrieval-Augmented Generation (RAG) emerges as a promising paradigm, where an encoder
projects a query and document pages into a shared embedding space, and a retrieval step selects the most relevant
pages for an LVLM to generate an answer. While late-interaction models have significantly improved page
representation [ 12–14,3], determining how many pages should be retrieved remains an open challenge, despite
its critical impact: too many pages increase LVLM latency and may introduce irrelevant context that degrades
answer accuracy, whereas too few pages risk omitting relevant evidence.
State-of-the-art retrieval methods rely on a late-interaction mechanism [ 12], were the encoder independently
maps the query and pages into multi-vector representations. Page relevance is obtained by aggregating the best
matches of each query embedding across page embeddings, enabling fine-grained semantic matching, offline
page encoding, and scalable online retrieval. However, late-interaction produces independent page relevance
scores. As a result, it relies on a fixed top- kretrieval that cannot adapt to the query and may include irrelevant
pages or omit relevant ones. Moreover, by treating pages independently and assigning uniform importance to
embeddings, late-interaction does not exploit the semantic structure across pages (Figure 1-a).
Text retrieval has mitigated this through embedding weighting based on token frequency statistics [ 15–18], or
adaptive- kretrieval [ 19], but these approaches rely on discrete token structures that do not extend to visual
embeddings and often require additional training. In contrast, we show that late-interaction representations reveal
sparse, query-dependent semantic content across pages, enabling dynamic localization and adaptive retrieval.
We introduce ViSAR (Visual Semantic Activation Retrieval), a training-free retrieval mechanism that operates
directly in the embedding space of late-interaction encoders. ViSAR constructs a query-conditioned page-level
similarity matrix that highlights query semantics and enables adaptive- kretrieval, dynamically determining the
number of retrieved pages at inference time (Figure 1-b). We also show that the structure of the similarity matrix
correlates with answer accuracy, suggesting a general principle for quality-aware document understanding. Our
main contributions are as follows:
2

•ViSAR:A training-free adaptive- kretrieval mechanism for visual retrieval using late-interaction
encoders.
•Semantic localization:A query-conditioned page-level similarity matrix whose structure reflects
evidence localization and correlates with answer accuracy.
•Experiments:ViSAR retrieves fewer pages than fixed top- kretrieval on average and reduces RAG
latency by up to 58.7%, while maintaining or improving answer accuracy across multiple encoders
and LVLMs.
2 Related Work
2.1 Multi-Vector Retrieval and Embedding Weighting.
Multi-vector retrieval via late-interaction was introduced by ColBERT [ 12] for text documents. It represents
queries and pages as independent sets of token-level embeddings, and page relevance is computed by aggregating
query embeddings’ best similarity score to page embeddings. Subsequent works improved its efficiency [ 20,21],
introduced token weighting mechanisms based on frequency statistics, learned importance estimation, or sparse
representations [ 15,17,16,18,22–24], and explored retrieval refinement through pseudo-relevance feedback
[25]. However, these approaches rely on Optical Character Recognition (OCR) to extract the text content, cannot
capture visual content, and often require additional training. Recent works such as ColPali [ 13,14] extend
late-interaction by encoding each page as a set of visual patch embeddings, emerging as the state-of-the-art
paradigm for OCR-free visual document retrieval [ 26,13]. However, it still relies on independent page scores
and fixed top- kretrieval, leaving the semantic organisation of pages unexplored. ViSAR instead exploits this
structure to enable adaptive-kretrieval.
2.2 Adaptive-kRetrieval.
Adaptive retrieval has mainly been explored via iterative approaches, where LLMs or LVLMs reason over
multiple rounds of fixed top- kretrieval, making kvary implicitly across iterations [ 27–31]. However, adaptive- k
methods directly estimate the number of pages to retrieve in a single pass. Prior work determines the retrieval
cut-off via heuristics applied to late-interaction scores by identifying the largest gap between consecutive scores
[19] or by clustering scores to detect transitions [ 32,33]. However, these methods are text-based or require
additional training, and rely solely on score distributions, while ViSAR performs training-free adaptive- k
retrieval by exploiting the semantic structure encoded in late-interaction representations.
2.3 Visual Document Retrieval and DocVQA.
These advances in retrieval have directly influenced recent OCR-free DocVQA systems, where effective page
selection is a key component of the reasoning pipeline combined with LVLM-based reasoning [ 34]. VisRAG [ 3]
introduced a dedicated single-vector visual retriever, while M3DocRAG [ 1] demonstrated the effectiveness of
multi-vector late-interaction retrieval with ColPali. Other frameworks [ 35,2,36,37] improve visual encoders,
iterative retrieval, or LVLM reasoning. However, none exploit the semantic structure induced by late-interaction
representations, which ViSAR addresses while remaining agnostic to the encoder and LVLM.
3 Visual Semantic Activation Retrieval (ViSAR)
Figure 2 depicts our method, ViSAR, which addresses visual retrieval in DocVQA. ViSAR enables adaptive- k
retrieval by weighting Query-to-Page, Page-to-Query, and Page-to-Page interactions without encoder retraining.
It uses the MaxSim operator from late-interaction [ 12,13] to exploit the fine-grained semantic structure in the
embedding space and improve retrieval. Algorithm S1 (supplementary material) summarizes the full pipeline.
3.1 Preliminaries: MaxSim and Late-Interaction
We denote a document Das a set of pages D={P1, . . . , PN}. Each page Ppand query Qare denoted as
Pp={vp
j}np
j=1andQ={q i}m
i=1, their respective sets of multi-vector representations in a common embedding
spaceRD, such that Pp∈Rnp×DandQ∈Rm×D. The relevance score of PptoQis estimated via late-
interaction between their sets of embeddings (Equation 1), summing each qimaximum similarity over all vp
j,
wheresim(·,·) =⟨·,·⟩defines the cosine similarity between two embeddings.
SQ,Pp=mX
i=1max
j∈{1,...,n p}sim 
qi, vp
j
(1)
3

⋯𝑤 𝑣1𝑝
1𝑝𝒘 𝒗j𝒑𝑤 𝑣n𝑝
𝒘 𝒗𝟏𝒑′ ⋯𝑤 𝑣j'𝑝′𝒘 𝒗𝐧𝒑′Query-to-Page interaction matrix⋯ 𝒘𝟏𝒒𝟏𝑤i𝑞2 𝒘 𝒒𝐦 𝐦𝑤1𝑃1 ⋯ 𝒘p𝑷p𝑤 𝑃𝑁
𝑁
Page-to-Query interaction matrix
Page-to-Page interaction matrix⋯𝑃1
𝑃𝑁𝑃2
Efficient Adaptive-k page retrieval𝒌∗=𝟐1 2
3 4
⋯
⋯
𝑞1𝑞2𝑞m
Ap,m
Ap,2
Ap,1
MaxSimMaxSim
MaxSimPage p∑
j𝒑
n𝑝
𝟏𝒑′
j'𝑝′
𝐧𝒑′Figure 2: ViSAR overview. The method performs successive interactions in embedding space – 1.
Query-to-page: Unlike late-interaction, MaxSim activations are analyzed across pages rather than
summed over tokens, to derive token- and page-level weights (pink dotted arrows). 2. Page-to-Query:
These weights modulate patch-query similarities to derive patch-level weights (green dotted arrows).
3. Page-to-Page: These weights modulate patch-patch similarities to compute query-conditioned
similarities between pages (blue dotted arrow), forming a page-level similarity matrix that highlights
the shared query-related content. 4. Adaptive- k: The similarity matrix is used to retrieve an adaptive-
kset of pages.
3.2 Adaptive Multi-Level Interaction Weighting
Given the uniform contribution of qiandvp
jin Equation 1, we estimate their importance directly from the
embedding space. Since cosine similarity measures embedding alignments in a semantically structured space,
the MaxSim operator over all vp
jcaptures how strongly the semantic encoded by qiis realized in Pp. We refer to
Ap,i(Equation 2) as theactivationscore. Unlike late-interaction (Equation 1), which aggregates the activations
overq iinto a single score, we exploit the full activation matrix to identify discriminative query semantics.
Ap,i= max
j∈{1,...,n p}sim 
qi, vp
j
= max
j∈{1,...,n p}⟨qi, vp
j⟩(2)
3.2.1 Query-to-Page Interaction Weighting
We use Ap,ito derive weights for vectors qiand pages Ppto highlight strong and spatially localized semantics.
Ap,iis rescaled by its mean across pages into ˆAp,i, and modulated by its normalized inter-page standard
deviationˆσ i, yielding ˜Ap,i(Equation 4).
ˆAp,i=Ap,i
mean p(Ap,i)ˆσi=stdp(Ap,i)
mean i′(stdp(Ap,i′))(3)
˜Ap,i=ˆAp,i·ˆσi (4)
Aggregating ˜Ap,iover pages penalizes ubiquitous semantic content and highlights sparse activations, resulting
in a weightw ifor eachq i(Equation 5).
wi= logN
1 +a i,witha i=X
p˜Ap,i (5)
We then compute a page-level semantic co-activation Cp
i,i′, which is normalized by the mean activation of each
qiacross pages, and a weight wpis obtained for each page by aggregating the mean co-activation of all qi
(Equation 7).
Cp
i,i′=˜Ap,i·˜Ap,i′ ¯Ai= mean p ˜Ap,i
(6)
4

wp=mX
i=1mean i′Cp
i,i′
¯Ai·¯Ai′
(7)
3.2.2 Page-to-Query Interaction Weighting
Min-Max normalization is applied to wiandwp, yielding ˆwi,ˆwp∈[0,1] , which modulate the activations ˜Ap,i
and weight the cosine similarity between all qiandvp
j. The MaxSim operator is then applied over all qi(i.e.,
patch-to-query direction, reversing standard late-interaction) to capture how strongly the semantic encoded
byvp
jis realized in the query Q. We use this weighted maximum similarity as the relevance score rp
jofvp
j. A
weightwp
jis obtained for eachvp
jof a pagePpby centering and thresholding the relevancerp
j(Equation 9).
rp
j= max
i∈{1,...,m}h
⟨vp
j, qi⟩ · ˜Ap,i·ˆwi·ˆwp2i
(8)
wp
j= max 
0, rp
j−mean p,j 
rp
j
(9)
3.2.3 Page-to-Page Interaction Weighting
A Min-Max normalization is applied to wp
j, yielding ˆwp
j∈[0,1] , which modulates the importance of page
embeddings and enables a similarity measure between pages. For a source page Ppand target page Pp′, the
cosine similarity between all vp
jandvp′
j′is weighted by the target embedding weights ˆwp′
j′. The MaxSim operator
is then applied over all vp′
j′to capture how strongly the semantic encoded by vp
jis realized in Pp′. The resulting
score is then weighted by the source embedding weightˆwp
jto obtainSp→p′
j (Equation 10).
Sp→p′
j = ˆwp
j·max
j′h
⟨vp
j, vp′
j′⟩ ·ˆwp′
j′i
(10)
Finally, the similarity matrix Sim(p, p′)∈RN×Nis computed by averaging the Tlargest interactions of
Sp→p′
j across source embeddings vp
j, denoted by T ⊂ {1, . . . , n p}, and taking the square root (Equation 11).
Since each source embedding independently searches for its best match in the target page (Equation 10), this
formulation naturally produces a directional page similarity, whereSim(p, p′)̸= Sim(p′, p).
Sim(p, p′) =s
1
TX
j∈TSp→p′
j (11)
3.2.4 Adaptive-kRetrieval
We leverage the structure of Sim(p, p′)to develop an adaptive- kretrieval method that adapts to the query.
The self-similarity sp= Sim(p, p) of a page is used to rank the pages and define a candidate relevant set Rk,
containing the khighest-scoring pages. The remaining N−k pages constitute the irrelevant set Ik. For each
pagep∈ R k, we compute its coherence with the relevant set, cp
k, and the leakage from the irrelevant set, lp
k,
as the mean similarity to pages in RkandIk, respectively (Equation 12). The scores spare also normalized
to yield wspsumming to one, and combined with cp
kandlp
kin the cost function J(k) . The optimal number
of pages k⋆is obtained by minimizing J(k) , requiring the evaluation of at most Ncandidate sets, while an
exhaustive search would require evaluating2Nsets.
cp
k= mean p′∈Rk
Sim(p, p′)
(12)
lp
k= mean p′∈Ik
Sim(p, p′)
(13)
J(k) =X
p∈Rkwsp
cp
k−γ lp
k
(14)
The parameter γ >0 controls the leakage penalty. In addition, since the minimum J(k⋆)may imply residual
leakage from pages in Ik⋆, we evaluate whether k⋆corresponds to a sharp transition by accepting Rk⋆+1only
ifJvaries more sharply beyond k⋆than before it, indicating a large drop in leakage caused by the transitioning
page (Algorithm S1).
3.2.5 Implementation Details
We exploit the natural sparsity induced by ViSAR. Equation 9 produces a subset of inactive pages whose patch
weights are all zero, which cannot contribute to the similarity matrix (Equation 11). These pages are excluded
from the computation of Equation 10 and the optimization of J(k) . Equation 10 is also evaluated block-wise
rather than as a dense tensor, reducing peak memory usage. These implementation choices preserveexact
mathematical equivalence while avoiding unnecessary computation.
5

4 Experiments
We evaluate three research questions: (RQ1) Can ViSAR effectively adapt the number of retrieved pages
compared with fixed top- klate-interaction retrieval, and how does this affect DocVQA accuracy? (RQ2) How
does ViSAR’s adaptive retrieval strategy impact latency? (RQ3) Can ViSAR’s similarity matrix provide insights
into answer quality?
4.1 Experimental Setup
All experiments were conducted on an NVIDIA A6000 GPU (48 GB of memory) and run once per configuration.
We set T= 50 in Equation 11 and γ= 105in Equation 14. Further details on compute resources and sensitivity
analyses for both hyperparameters are provided in thesupplementary material.
Datasets.MMLongBench [ 38] and LongDocURL [ 39] datasets were used, which provide the answer evidence
pages for page ranking evaluation. They cover multiple scenarios to evaluate the answer generation, requiring
textual and visual reasoning across pages, and efficient retrieval over multi-page documents.
Encoders.We evaluate ViSAR on multi-vector embeddings from three OCR-free visual encoders: ColPali
[13], ColQwen2.5 [ 13], and ColModernVBERT [ 14]. For broader comparison, we consider ColBERTv2 [ 20], a
multi-vector text encoder operating on OCR-extracted text, and VisRAG-Ret [ 3], a single-vector visual encoder.
Retrieval Baselines.Multi-vector embeddings use late-interaction with fixed top- k, whereas single-vector
embeddings use cosine similarity with fixed top- k[3]. We also adapt two adaptive- kheuristic methods originally
proposed for late-interaction text retrieval to visual retrieval: aLargest-Gapcriterion [ 19] and aScore-Cluster
approach [32].
Oracle.Using the late-interaction ranking, the Oracle adapts the number of retrieved pages by selecting the
smallest top- kthat contains all evidence pages. It provides an upper bound on adaptive- kunder a standard and
fixed ranking, which may include irrelevant pages.
ViSAR Evaluation.We examine the effectiveness of ViSAR’s adaptive behaviour by reporting retrieval statistics,
Precision, Recall, and F1-scores at rank k⋆, and comparing ViSAR with Oracle, Largest-Gap, and Score-Cluster
adaptive methods. We evaluate ViSAR’s ranking quality against late-interaction ranking using Recall and
normalized discounted cumulative gain (NDCG) at ranks 5 and 10. We then assess the impact of adaptive
retrieval on answer generation using Qwen2.5-VL-7B-Instruct (greedy decoding) as the primary model, plus
additional generation models with default configurations. All methods are evaluated under a maximum budget
of Max- kpages, i.e, top-k=Max-k for fixed top- kretrieval. We report the answer accuracy, leveraging
LLM-as-a-judge evaluation [ 40,41], with Qwen2.5-14B-Instruct LLM [ 42] as the evaluator. It receives the
query, the reference answer and the generated answer, and predicts a binary correctness label based on semantic
equivalence, using few-shot examples and a constrained structured output. We then analyse how ViSAR latency
scales with document size, and compare its retrieval and generation costs with fixed top- k, Largest-Gap, and
Score-Cluster. Finally, we provide an ablation study to assess the contribution of ViSAR’s components.
4.2 Experimental Results
Preliminary experiments revealed numerical instabilities using ColModernVBERT on a few LongDocURL pages
(7.31% of queries). Careful investigation confirmed that the issue occurred during encoding, independently of
ViSAR. These samples were excluded from evaluation.
4.2.1 Adaptive Retrieval
Table 1 reports mean, median, Recall@ k⋆, Precision @k⋆, and F1 @k⋆across encoders for ViSAR, Oracle,
Largest-Gap and Score-Cluster methods. The mean of Oracle shows that retrieving all evidence pages often
requires significantly more pages than the number of evidence pages, while its lower median indicates that
compact retrieval is sufficient for many queries. This reflects the known limitations of late-interaction ranking
[14,43], which may struggle to capture all relevant pages for complex queries, requiring the retrieval of
intermediate irrelevant pages that may lead to unnecessarily large contexts. Adaptive retrieval methods exhibit a
similar pattern to Oracle but explicitly balance Recall and Precision, with ViSAR retrieving fewer pages than
Oracle on average while Largest-Gap and Score-Cluster retrieve more.
Figure 3-a further characterises this behaviour as a function of kOracle. ViSAR closely follows Oracle at low
kOracle and retrieves fewer pages as kOracle increases, limiting the inclusion of irrelevant pages that accumulate
in complex queries. In contrast, Largest-Gap and Score-Cluster over-retrieve across most kOracle, consistent
with their higher mean k⋆(Table 1). Consequently, ViSAR tends to favour precision, while Largest-Gap and
Score-Cluster tend to favour recall.
6

Table 1: Adaptive- kretrieval: means ( m), medians ( md), Recall@ k⋆(R@k⋆), Precision@ k⋆(P@k⋆),
and F1@ k⋆. Oracle retrieves substantially more pages than the number of evidence pages, reflecting
the known limitations [ 14,43] of late-interaction ranking. Adaptive methods balance Recall and
Precision by adjusting the number of retrieved pages, with ViSAR favouring more compact retrieved
sets on average, avoiding unnecessarily large retrieval sets. Full distributions in Figure S1–S2
MMLongBench LongDocURL
Encoder Retrieval m mdR@k⋆P@k⋆F1@k⋆m mdR@k⋆P@k⋆F1@k⋆
ColQwen2.5 Oracle 8.3 2 100.0 67.30 80.45 10.7 3 100.0 61.07 75.83
Score-Cluster 18.6 8 85.57 29.51 43.89 36.6 20 89.49 18.85 31.14
Largest-Gap 15.4 2 81.12 45.13 58.00 28.7 3 82.09 45.15 58.26
ViSAR (ours) 4.7 3 75.16 50.37 60.32 7.9 5 81.24 38.41 52.16
ColPali Oracle 8.8 2 100.0 62.16 76.67 12.4 3 100.0 57.70 73.18
Score-Cluster 18.7 8 82.64 26.12 39.70 35.8 18 87.56 19.15 31.43
Largest-Gap 16.2 3 79.16 41.94 54.83 24.7 2 79.37 46.79 58.87
ViSAR (ours) 5.3 3 73.57 46.11 56.69 8.1 6 80.16 34.38 48.12
ColModern Oracle 10.5 2 100.0 58.87 74.11 13.0 3 100.0 53.35 69.58
VBERT Score-Cluster 23.0 12 84.28 23.59 36.86 47.4 44 87.90 15.61 26.51
Largest-Gap 20.0 4 80.57 36.81 50.53 39.3 4 83.57 34.94 49.28
ViSAR (ours) 7.5 4 74.07 34.43 46.99 13.5 11 83.99 17.47 28.92
evidence pages 1.9 1 – – – 1.9 2 – – –
20
1.0
0.8
0.6
0.4
0.2
0.01.0
0.8
0.6
0.4
0.2
0.010
0
-10
-30
-40
-50
-60-20Mean(k − kOracle )(a)F1-score
% of Queries
kOracle123456 7 8 910 11-15 16-20 21-25 26-35 36+
Largest-Gap
Score-ClusterViSAROracle
Query distribution(b)
Figure 3: Adaptive- kretrieval as a function of kOracle , using ColQwen2.5 on MMLongBench (consis-
tent trends across encoders and datasets in Figure S3–S4). (a): mean difference k−k Oracle . ViSAR
closely follows Oracle at low kOracle and retrieves fewer pages as kOracle increases, limiting irrelevant
pages retrieval. Largest-Gap and Score-Cluster over-retrieve across most kOracle . (b): F1-score with
the corresponding percentage of queries. ViSAR achieves higher F1, especially at lowerk Oracle .
7

Table 2: Page ranking evaluation, comparing ViSAR and late-interaction at ranks 5 and 10 on
MMLongBench (similar trends for LongDocURL in Table S1). ViSAR consistently improves page
ranking, providing stronger support for its adaptive-kretrieval mechanism.
Recall NDCG
Ranking Method@5 @10 @5 @10
Using Colpali
Late-Interaction 75.00 84.49 0.730 0.746
ViSAR (ours)76.36 86.68 0.734 0.756
Using ColQwen2.5
Late-Interaction 78.60 86.82 0.780 0.793
ViSAR (ours)79.73 87.73 0.790 0.799
Using ColModernVBERT
Late-Interaction 72.79 83.14 0.704 0.723
ViSAR (ours)73.66 83.56 0.709 0.725
Accordingly, dataset-level F1-scores (Table 1) show that ViSAR performs best on MMLongBench, while
Largest-Gap performs best on LongDocURL. However, this aggregate metric depends on the query distribution
overkOracle. When analysed across kOracle, ViSAR achieves higher F1-scores on MMLongBench (Figure 3-b),
particularly for lower kOracle where most queries are located. On LongDocURL (Figure S4), ViSAR outperforms
Largest-Gap for most kOracle, except kOracle= 1, suggesting that Largest-Gap is mostly effective for single-page
queries, which constitute the majority of queries. This contributes to its higher dataset-level F1-score, whereas
ViSAR provides a more consistent adaptive retrieval across query complexities. Overall (Figure 3-b, Figure S4),
ColQwen2.5 shows the best results, followed by ColPali, while ColModernVBERT exhibits a weaker trend.
Table 2 confirms this trend, showing that ViSAR’s adaptive behaviour is grounded in improved ranking,
outperforming late-interaction ranking across Recall and NDCG at ranks 5 and 10, indicating that its weighting
mechanism better aligns page scoring to the query. Yet, ColModernVBERT shows more modest improvements,
consistent with its smaller size (250M vs. 3B parameters), which likely limits its ability to separate semantics,
resulting in less compact sets (Table 1) and lower ranking performance overall.
4.2.2 Answer Accuracy
Table S2 compares ViSAR with literature baselines, where Qwen2.5-VL-7B-Instruct LVLM and Qwen2.5-7B-
Instruct LLM are used, respectively, as generation models for visual retrieval methods and text-based retrieval
method (ColBERTv2, OCR-extracted text using Tesseract [ 44]). Max- kdenotes an input budget of at most k
pages, where fixed top- kmethods always use the full budget while adaptive methods only do so when required.
All adaptive methods outperform fixed top- kretrieval, with ViSAR achieving competitive or superior accuracy
across all settings. Notably, Largest-Gap and Score-Cluster also improve over fixed top- kdespite their simplicity,
indicating the inherent benefit of adaptive-kretrieval.
Table 4 shows that ViSAR generalizes across encoders, maintaining or improving accuracy over late-interaction
fixed top- k(McNemar’s test, p <0.05 ): ColQwen2.5 and ColPali show a general trend toward improvement,
with significance on LongDocURL. ColModernVBERT shows a non-significant trend toward fixed top- k,
consistent with its weaker retrieval performance (Table 1 and Table 2). Supplementary Table S3–S7 confirm
this trend using additional generation models: Across 60 configurations (3 encoders ×5 LVLMs ×2 Max- k
budgets ×2 datasets), ViSAR improves accuracy in 24 cases and maintains in the remaining 36. The largest
improvements are observed for LVLMs more sensitive to longer contexts (Table S6,S7), consistent with ViSAR
reducing both retrieved and irrelevant pages (Table 1).
4.2.3 RAG Latency
The reduction in context size enabled by adaptive retrieval lowers LVLM processing cost. While fixed top- k
retrieval always maximises the LVLM input budget, ViSAR dynamically adjusts the number of pages and only
uses the full budget when required. Figure 4 reports retrieval, generation, and end-to-end (retrieval + generation)
latencies, showing that ViSAR introduces a retrieval overhead that contributes only marginally, achieving a
substantial reduction in generation latency that dominates the total cost. This result in an end-to-end latency
reductions of up to 58.7% on MMLongBench and 38.5% on LongDocURL at a Max-10 LVLM budget. Table S8
further compares ViSAR with Largest-Gap and Score-Cluster, showing that both methods also reduce RAG
8

Table 3: Answer generation accuracy against literature baselines. Qwen2.5-VL-7B-Instruct is used
as the generation model for all visual methods, while ColBERTv2 operates on OCR-extracted text
(Tesseract [ 44]) with Qwen2.5-7B-Instruct. Max- kdenotes an input budget of at most kpages. Bold
indicates highest results, with confidence intervals and Adaptive- kusing other encoders in Table S2.
ViSAR achieves competitive or superior accuracy across settings.
MMLongBench LongDocURL
Retrieval MethodMax-5 Max-10 Max-5 Max-10
Fixed top-k
ColBERTv2 24.51 24.70 47.18 47.70
M3DocRAG 34.86 35.08 59.31 58.71
VisRAG-Ret 34.48 35.69 57.29 58.02
Adaptive-k(using ColQwen2.5)
Largest-Gap 35.79 35.88 61.01 60.89
Score-Cluster 36.25 35.97 60.00 59.83
ViSAR (ours)36.53 36.63 61.06 60.97
Table 4: Answer generation accuracy across encoders. Qwen2.5-VL-7B-Instruct is used as the
generation model. Max- kdenotes an input budget of at most kpages. Bold indicates statistically
significant differences according to McNemar’s test ( p <0.05 ). ViSAR consistently maintains or
improves accuracy, with no significant decreases observed, while retrieving fewer pages on average.
MMLongBench LongDocURL
Retrieval MethodMax-5 Max-10 Max-5 Max-10
Using ColQwen2.5
Fixed top-k35.04 35.69 59.79 59.27
ViSAR (ours) 36.53 36.6361.06 60.97
Using ColPali
Fixed top-k34.86 35.08 59.31 58.71
ViSAR (ours) 35.42 35.8860.77 60.13
Using ColModernVBERT
Fixed top-k34.75 34.95 58.42 58.84
ViSAR (ours) 34.01 34.28 58.28 58.33
latency on average. However, Score-Cluster remains consistently slower than ViSAR, while Largest-Gap only
provides a noticeable advantage on LongDocURL, with gains of up to 47.2% over fixed top- kat Max-10 budget.
Although ViSAR’s overhead increases with document size (Figure 4), it only becomes noticeable for the largest
MMLongBench document (468 pages, Figure S10). In such cases, the supplementary material presents strategies
that approximate the similarity matrix to preserve latency gains.
4.2.4 Similarity Structure Reflects Accuracy
ViSAR’s query-conditioned similarity matrix Sim(p, p′)provides an explicit representation of how query-
relevant information is distributed across a document. Figure 5 illustrates representative examples and shows
that when query semantics are localised, Sim(p, p′)is sparse and produces a sharp minimum in J(k) , enabling
a reliable retrieval boundary (Figure 5-a). Conversely, when query semantics are distributed across many pages,
Sim(p, p′)becomes denser, yielding a shallow minimum and a less reliable stopping decision (Figure 5-b).
Figure 5-c confirms this trend, showing that higher similarity matrix sparsity consistently correlates with higher
answer accuracy across encoders on MMLongBench (similar trend on LongDocURL in Figure S14 and Table S9).
WhileJ(k) is designed to maximise coherence and minimise leakage within this structure, these results provide
post-hoc validation that its components capture semantically meaningful retrieval properties. This suggests that
analysing the structure in Sim(p, p′)is a promising direction for understanding the conditions of retrieval success
9

# Document pages Max-k retrieved pagesRetrieval Latency Generation Latency End-to-end Latency
Max-k retrieved pages50-75 76-100 101-125 126-1501.0
0.8
0.6
0.4
0.2
0.0Latency (s)
2 4 6 8 1014
12
10
8
6
4
214
12
10
8
6
4
2
2 4 6 8 10
2 4 6 8 10 2 4 6 8 1014
12
10
8
6
4
214
12
10
8
6
4
22.5
2.0
1.5
1.0
0.5
0.0
1-50 51-100 101-150 151-200 468Latency (s)
MMLongBench
LongDocURL
ViSAR
Fixed top-k
−59%
−39.5% −38.5%−58.7%Figure 4: Average RAG latency comparing ViSAR and fixed top- k(shaded: std). Although ViSAR
introduces a retrieval overhead that grows with document size, it reduces generation latency by
limiting LVLM input context. The net effect is an end-to-end latency reduction of up to 58.7% at
k= 10 . The retrieval overhead becomes significant only for the longest MMLongBench document
(468 pages, Figure S10). For such extreme cases, computational approximations of the similarity
matrix are provided in the supplementary material to preserve latency gains.
Page p' # Pages retrievedJ(k)Page p
1.0
0.8
0.6
0.4
0.2
0.00
5
10
15
20
25
30
051015 20 25 30
1.0
0.8
0.6
0.4
0.2
0.014
kOracle=5
kViSAR=15
8 16 12 20 24 30 32J(k)Page p
1.0
0.8
0.6
0.4
0.2
0.00
3
6
9
12
15
18
0 3 6 9 12 15 18
1 3 5 7 9 1113 15 17 1.0  
0.8  
0.6  
0.4  
0.2  
0.0
Accuracy (%)55
50
45
40
35
50 60 70 80 90
Similarity Matrix Sparsity (%)
max-5
max-10
ColQwen2.5ColPali
ColModernVBERTcorrect answer
(a)
(b)(c)
wrong answer
kViSAR=kOracle=2
Figure 5: Similarity matrix structure and answer accuracy on MMLongBench. (a, b): Query-
conditioned page-level similarity matrices (zero values in black) and the corresponding retrieval cost
function J(k) , with k⋆
ViSAR andk⋆
Oracle marked. (a): A sparse matrix allows the query semantics
to be effectively localized, which produces a sharp minimum in J(k) , enabling accurate adaptive
retrieval and correct answer generation. (b): A dense matrix yields a shallow minimum, causing
over-retrieval and incorrect answer generation. (c): Answer accuracy as a function of the percentage
of inactive entries in the similarity matrix, showing that higher sparsity consistently correlates with
higher accuracy across encoders.
and failure, potentially enabling the conception of a label-free feedback signal for future retrieval strategies, such
as iterative query refinement or evidence selection, without additional models or training.
4.2.5 Ablation Studies
We evaluate simplified variants by disabling the query embedding weights wi, page weights wp, patch embedding
weights wp
j, and by replacing the optimization objective J(k) with fixed top- k, Largest-Gap, and Score-Cluster
retrieval. Table S10 shows that each modification degrades retrieval quality and downstream answer accuracy,
indicating that both the weighting strategy and the optimization objective contribute to ViSAR’s performance.
10

5 Conclusion
We introduced ViSAR, atraining-freevisual embedding weighting mechanism enablingadaptive- kretrievalfor
document visual question answering. It operates in the embedding space of late-interaction encoders to construct
a query-conditioned page-level similarity matrix that drives compact and adaptive page selection, reducing RAG
latency by up to 58.7% while maintaining or improving answer accuracy. We further showed that the structure of
the similarity matrix reflects the localization of query-relevant information and correlates with answer accuracy,
suggesting that it may provide a useful feedback signal for future retrieval strategies.
6 Supporting Information
Figure S1–S14 and Table S1–S10 are available in the supplementary material pdf.
References
[1]Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie He, and Mohit Bansal. M3docrag: Multi-modal retrieval
is what you need for multi-page multi-document understanding.arXiv preprint arXiv:2411.04952, 2024.
[2]Siwei Han, Peng Xia, Ruiyi Zhang, Tong Sun, Yun Li, Hongtu Zhu, and Huaxiu Yao. Mdocagent: A
multi-modal multi-agent framework for document understanding.arXiv preprint arXiv:2503.13964, 2025.
[3]Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang, Xu Han,
Zhiyuan Liu, et al. Visrag: Vision-based retrieval-augmented generation on multi-modality documents. In
International Conference on Learning Representations, volume 2025, pages 21074–21098, 2025.
[4]Rubèn Tito, Dimosthenis Karatzas, and Ernest Valveny. Hierarchical multimodal transformers for multipage
docvqa.Pattern Recognition, 144:109834, 2023.
[5]Gilles Baechler, Srinivas Sunkara, Maria Wang, Fedir Zubach, Hassan Mansoor, Vincent Etter, Victor
C˘arbune, Jason Lin, Jindong Chen, and Abhanshu Sharma. Screenai: A vision-language model for ui and
infographics understanding.arXiv preprint arXiv:2402.04615, 2024.
[6]Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei
Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report.arXiv preprint arXiv:2511.21631, 2025.
[7]Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi
Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal
models with open-source suites.Science China Information Sciences, 67(12):220101, 2024.
[8]Camille Barboule, Benjamin Piwowarski, and Yoan Chabot. Survey on question answering over visually
rich documents: Methods, challenges, and trends.arXiv preprint arXiv:2501.02235, 2025.
[9]Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images.
InProceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2200–2209,
2021.
[10] Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark
for question answering about charts with visual and logical reasoning. InFindings of the association for
computational linguistics: ACL 2022, pages 2263–2279, 2022.
[11] Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar.
Infographicvqa. InProceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision,
pages 1697–1706, 2022.
[12] Omar Khattab and Matei Zaharia. Colbert: Efficient and effective passage search via contextualized late
interaction over bert. InProceedings of the 43rd International ACM SIGIR conference on research and
development in Information Retrieval, pages 39–48, 2020.
[13] Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, and Pierre
Colombo. Colpali: Efficient document retrieval with vision language models. InInternational Conference
on Learning Representations, volume 2025, pages 61424–61449, 2025.
[14] Paul Teiletche, Quentin Macé, Max Conti, Antonio Loison, Gautier Viaud, Pierre Colombo, and Manuel
Faysse. Modernvbert: Towards smaller visual document retrievers.arXiv preprint arXiv:2510.01149,
2025.
11

[15] Xiao Wang, Craig Macdonald, Nicola Tonellotto, and Iadh Ounis. Pseudo-relevance feedback for multiple
representation dense retrieval. InProceedings of the 2021 ACM SIGIR International Conference on Theory
of Information Retrieval, pages 297–306, 2021.
[16] Karan Samel, Cheng Li, Weize Kong, Tao Chen, Mingyang Zhang, Shaleen Gupta, Swaraj Khadanga,
Wensong Xu, Xingyu Wang, Kashyap Kolipaka, et al. End-to-end query term weighting. InProceedings of
the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 4778–4786, 2023.
[17] S Archish, Ankit Garg, Kirankumar Shiragur, and Neeraj Kayal. Incorporating token importance in
multi-vector retrieval. InProceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages
32860–32866, 2026.
[18] Minghan Li, Sheng-Chieh Lin, Xueguang Ma, and Jimmy Lin. Slim: Sparsified late interaction for multi-
vector retrieval with inverted indexes. InProceedings of the 46th International ACM SIGIR Conference on
Research and Development in Information Retrieval, pages 1954–1959, 2023.
[19] Chihiro Taguchi, Seiji Maekawa, and Nikita Bhutani. Efficient context selection for long-context qa: No
tuning, no iteration, just adaptive-k. InProceedings of the 2025 Conference on Empirical Methods in
Natural Language Processing, pages 20116–20141, 2025.
[20] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. Colbertv2:
Effective and efficient retrieval via lightweight late interaction. InProceedings of the 2022 Conference
of the North American Chapter of the Association for Computational Linguistics: Human Language
Technologies, pages 3715–3734, 2022.
[21] Keshav Santhanam, Omar Khattab, Christopher Potts, and Matei Zaharia. Plaid: an efficient engine for
late interaction retrieval. InProceedings of the 31st ACM International Conference on Information &
Knowledge Management, pages 1747–1756, 2022.
[22] Thibault Formal, Carlos Lassance, Benjamin Piwowarski, and Stéphane Clinchant. Splade v2: Sparse
lexical and expansion model for information retrieval.arXiv preprint arXiv:2109.10086, 2021.
[23] Zhuyun Dai and Jamie Callan. Context-aware sentence/passage term importance estimation for first stage
retrieval.arXiv preprint arXiv:1910.10687, 2019.
[24] Rodrigo Kataishi. Enhancing retrieval-augmented generation with topic-enriched embeddings: A hybrid
approach integrating traditional nlp techniques.arXiv preprint arXiv:2601.00891, 2025.
[25] Xiao Wang, Craig Macdonald, Nicola Tonellotto, and Iadh Ounis. Colbert-prf: Semantic pseudo-relevance
feedback for dense passage and document retrieval.ACM Transactions on the Web, 17(1):1–39, 2023.
[26] António Loison, Quentin Macé, Antoine Edy, Victor Xing, Tom Balough, Gabriel Moreira, Bo Liu, Manuel
Faysse, Céline Hudelot, and Gautier Viaud. Vidore v3: A comprehensive evaluation of retrieval augmented
generation in complex real-world scenarios.arXiv preprint arXiv:2601.08620, 2026.
[27] S Yao, J Zhao, D Yu, N Du, I Shafran, KR Narasimhan, and Y Cao. React: Synergizing reasoning and
acting in language models. in the eleventh international conference on learning representations, 2023.
[28] Akari Asai, Zeqiu Wu, Yizhong Wang, Avi Sil, and Hannaneh Hajishirzi. Self-rag: Learning to retrieve,
generate, and critique through self-reflection. InInternational conference on learning representations,
volume 2024, pages 9112–9141, 2024.
[29] Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C Park. Adaptive-rag: Learning
to adapt retrieval-augmented large language models through question complexity. InProceedings of the
2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers), pages 7036–7050, 2024.
[30] Chelsi Jain, Yiran Wu, Yifan Zeng, Jiale Liu, Shengyu Dai, Zhenwen Shao, Qingyun Wu, and Huazheng
Wang. Simpledoc: Multi-modal document understanding with dual-cue page retrieval and iterative
refinement. InProceedings of the 2025 Conference on Empirical Methods in Natural Language Processing,
pages 28398–28415, 2025.
[31] Junda Wu, Yu Xia, Tong Yu, Xiang Chen, Sai Sree Harsha, Akash V Maharaj, Ruiyi Zhang, Victor Bursztyn,
Sungchul Kim, Ryan A Rossi, et al. Doc-react: Multi-page heterogeneous document question-answering.
InProceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 2:
Short Papers), pages 67–78, 2025.
[32] Yifan Xu, Vipul Gupta, Rohit Aggarwal, Varsha Mahadevan, and Bhaskar Krishnamachari. Cluster-based
adaptive retrieval: Dynamic context selection for rag applications.arXiv preprint arXiv:2511.14769, 2025.
12

[33] Zongmin Li, Yachuan Li, Lei Kang, Dimosthenis Karatzas, and Wenkang Ma. Avir: Adaptive visual
in-document retrieval for efficient multi-page document question answering. InProceedings of the 7th
ACM International Conference on Multimedia in Asia, pages 1–7, 2025.
[34] Xu Zheng, Ziqiao Weng, Yuanhuiyi Lyu, Lutao Jiang, Haiwei Xue, Bin Ren, Danda Paudel, Nicu Sebe,
Luc Van Gool, and Xuming Hu. Retrieval augmented generation and understanding in vision: A survey
and new outlook.arXiv preprint arXiv:2503.18016, 2025.
[35] Ryota Tanaka, Taichi Iki, Taku Hasegawa, Kyosuke Nishida, Kuniko Saito, and Jun Suzuki. Vdocrag:
Retrieval-augmented generation over visually-rich documents. InProceedings of the Computer Vision and
Pattern Recognition Conference, pages 24827–24837, 2025.
[36] Manan Suri, Puneet Mathur, Franck Dernoncourt, Kanika Goswami, Ryan A Rossi, and Dinesh Manocha.
Visdom: Multi-document qa with visually rich elements using multimodal retrieval-augmented generation.
InProceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for
Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6088–6109,
2025.
[37] Ziyu Gong, Chengcheng Mai, and Yihua Huang. Mhier-rag: Multi-modal rag for visual-rich document
question-answering via hierarchical and multi-granularity reasoning.arXiv preprint arXiv:2508.00579,
2025.
[38] Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan
Ma, Xiaoyi Dong, et al. Mmlongbench-doc: Benchmarking long-context document understanding with
visualizations.Advances in Neural Information Processing Systems, 37:95963–96010, 2024.
[39] Chao Deng, Jiale Yuan, Pi Bu, Peijie Wang, Zhong-Zhi Li, Jian Xu, Xiao-Hui Li, Yuan Gao, Jun Song,
Bo Zheng, et al. Longdocurl: a comprehensive multimodal long document benchmark integrating
understanding, reasoning, and locating. InProceedings of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), pages 1135–1159, 2025.
[40] Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen,
Shengjie Ma, Honghao Liu, et al. A survey on llm-as-a-judge, 2025.URL https://arxiv. org/abs/2411.15594,
2(8), 2024.
[41] Michael Krumdick, Charles Lovering, Varshini Reddy, Seth Ebner, and Chris Tanner. No free labels:
Limitations of llm-as-a-judge without human grounding.arXiv preprint arXiv:2503.05061, 2025.
[42] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li,
Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2.5 technical report, 2025.
[43] Qi Liu, Gang Guo, Jiaxin Mao, Zhicheng Dou, Ji-Rong Wen, Hao Jiang, Xinyu Zhang, and Zhao Cao. An
analysis on matching mechanisms and token pruning for late-interaction models.ACM Transactions on
Information Systems, 42(5):1–28, 2024.
[44] Ray Smith. An overview of the tesseract ocr engine. InNinth international conference on document
analysis and recognition (ICDAR 2007), volume 2, pages 629–633. IEEE, 2007.
13

A ViSAR Implementation Details
A.1 ViSAR Algorithm
Algorithm S1ViSAR: Adaptive-kRetrieval via Multi-Level Interaction Weighting
Require:Q={q i}m
i=1;D={Pp}N
p=1;Pp={vp
j}np
j=1;T;γ
Ensure:Adaptive relevant page setR k⋆
1:Stage 1 – Query-to-Page Weighting
2:foreach pagep, query embeddingq ido
3:A p,i←max j⟨qi, vp
j⟩▷Eq. 2
4:end for
5:Compute ˆAp,i,ˆσi, and ˜Ap,i ▷Eq. 3–4
6:Compute weightw ifor eachq i ▷Eq. 5
7:Compute co-activationCp
i,i′ ▷Eq. 6
8:Compute weightw pfor each pagep ▷Eq. 7
9:
10:Stage 2 – Page-to-Query Weighting
11:Min-Max normalizew i→ˆw i,wp→ˆw p
12:foreach pagepdo
13:foreach patch embeddingvp
jdo
14:rp
j= max ih
⟨vp
j, qi⟩ · ˜Ap,i·ˆwi·ˆwp2i
▷Eq. 8
15:wp
j←max(0, rp
j−mean j(rp
j))▷Eq. 9
16:end for
17:end for
18:
19:Stage 3 – Page-to-Page Similarity
20:Min-Max normalizewp
j→ˆwp
j
21:foreach source pagepdo
22:foreach target pagep′do
23:foreach source patchvp
jdo
24:Sp→p′
j←ˆwp
j·max j′h
⟨vp
j, vp′
j′⟩ ·ˆwp′
j′i
▷Eq. 10
25:end for
26:Sim(p, p′)←q
1
TP
j∈TSp→p′
j ▷Eq. 11, top-Tpatches
27:end for
28:end for
29:
30:Stage 4 – Adaptive-kRetrieval
31:Rank pages by self-similaritys p= Sim(p, p)
32:fork= 1toNdo
33:R k←top-kpages bys p
34:I k←N−kremaining pages
35:Computecp
kandlp
k▷Eq. 12–13
36:J(k)←P
p∈Rkwsp(cp
k−γ lp
k)▷Eq. 14
37:end for
38:k⋆←arg min kJ(k)
39:n←round(ln(k⋆))▷Limit page inclusion whenk⋆is big
40:v +←Pn−1
i=0|J(k⋆+i)− J(k⋆+i+ 1)|▷variation beyondk⋆
41:v−←Pn−1
i=0|J(k⋆−i)− J(k⋆−i−1)|▷variation beforek⋆
42:ifv +> v−then
43:k⋆←k⋆+ 1
44:end if
45:
46:returnR k⋆
14

A.2 Computational Resources
•Hardware.Experiments were conducted on a workstation equipped with:
–NVIDIA RTX A6000 GPU (48 GB VRAM, driver 560.35.03);
–two Intel Xeon Silver 4210R CPUs (20 physical cores, 40 threads);
–251 GB of system memory.
•Software.Experiments were performed on Debian GNU/Linux 12 (Bookworm) using:
–Python 3.12.9;
–PyTorch 2.4.0 with CUDA 12.1;
–Transformers 4.57.0;
–FlashAttention 2.7.4.post1;
–ColPali Engine 0.3.13.
–Tesseract 5.5.2 with pytesseract 0.3.13
The complete list of Python dependencies and their versions is provided in the released code repository
(requirements.txt).
•Models.Experiments were conducted using publicly available models, identified via Hugging Face
model names or GitHub repository links as listed below:
– Retrieval – ViSAR’s encoders:
*vidore/colpali-v1.2;
*vidore/colqwen2.5-v0.1;
*ModernVBERT/colmodernvbert.
– Retrieval – Reproduced baselines:
*openbmb/VisRAG-Ret.
*https://github.com/bloomberg/m3docrag.
*colbert-ir/colbertv2.0 , using https://github.com/AnswerDotAI/RAGatouille .
– Retrieval – Adapted baselines:
*Largest-Gap:
·Adapted from:Taguchi, C.; Maekawa, S.; and Bhutani, N. 2025. Efficient Context Selec-
tion for Long-Context QA: No Tuning, No Iteration, Just Adaptive-k. In Proceedings of the
2025 Conference on Empirical Methods in Natural Language Processing, 20116–20141.
·Implementation in: ./pipeline/retrievers.py::HeuristicsSelector.LargestGap
*Score-Cluster:
·Adapted from:Xu, Y.; Gupta, V .; Aggarwal, R.; Mahadevan, V .; and Krishnamachari, B.
2025. Cluster-based Adaptive Retrieval: Dynamic Context Selection for RAG Applications.
arXiv preprint arXiv:2511.14769.
·Implementation in: ./pipeline/retrievers.py::HeuristicsSelector.ScoreCluster
– Answer generation:
*Qwen/Qwen2.5-VL-7B-Instruct;
*Qwen/Qwen3-VL-8B-Instruct;
*OpenGVLab/InternVL3_5-8B-Instruct;
*HuggingFaceM4/Idefics3-8B-Llama3;
*llava-hf/llava-onevision-qwen2-7b-ov-hf;
*Qwen/Qwen2.5-7B-Instruct.
– Evaluation:
*Qwen/Qwen2.5-14B-Instruct.
15

B ViSAR Adaptive-kRetrieval
B.1 Retrieved Page Numbers Distribution.
# Retrieved PagesLongDocURLQueries (%)
1 3 5 7 9 11 13 15 17 19 21 23 25 27 2940
30
20
10
0
oracle ColPali ViSAR (                                                                         ) ColQwen ColModernVBERT evidence pages
1357 9 1113 15 17192123405060
30
20
10
0
# Retrieved pagesMMLongBench
Figure S1:Number of retrieved pages distribution. ViSAR and Oracle, along with evidence pages.
They show an exponentially decaying distribution (capped at the maximum evidence pages size for
clarity – statistics are reported in Table 1 of the main text). ColPali and ColQwen2.5 show similar
trends to Oracle. ColModernVBERT shows a weaker trend, especially on LongDocURL.
Queries (%) Queries (%)
60
50
40
30
20
010
1357 9 1113 15 17192123
50
40
30
20
10
0
# Retrieved Pages1 3 5 7 9 11 13 15 17 19 21 23 25 27 2950
40
30
20
10
0
# Retrieved Pages1 3 5 7 9 11 13 15 17 19 21 23 25 27 29
1357 9 1113 15 1719212360
50
40
30
20
010
50
40
30
20
10
0
# Retrieved Pages1 3 5 7 9 1113 15 17 19 21 23 25 27 291357 9 1113 15 1719212360
50
40
30
20
010
ColPali ViSAR (                                                                       ) ColQwen ColModernVBERT evidence pages Score-Clustering Largest-Gap
MMLongBench
LongDocURL
Figure S2:Number of retrieved pages distribution. ViSAR, Largest-Gap, and Score-Cluster
methods, along with evidence pages.Score-Cluster: identifies an adaptive cut-off by clustering the
late-interaction scores.Largest-Gap: uses the largest gap in the late-interaction scores as an adaptive
cut-off. All methods show an exponentially decaying distribution (capped at the maximum evidence
page set size for clarity; statistics are reported in Table 1 of the main text). ViSAR tends to retrieve
compact sets, favouring lower and more varied page counts. In contrast, Largest-Gap tends to favour
single page retrieval, while Score-Cluster shows a weaker trend overall.
16

B.2 Retrieval Quality vs Oracle: Mean Difference in Retrieved size.
Mean(k - kOracle )MMLongBench LongDocURL
Mean(k - kOracle )
Mean(k - kOracle )
ViSAR - Colqwen2.5
 ViSAR - Colqwen2.5
ViSAR - ColPali
 ViSAR - ColPali
ViSAR - ColModernVBERT
 ViSAR - ColModernVBERT
Figure S3:Mean difference of k−k Oracle as a function of kOracle .ViSAR closely follows Oracle
at low kOracle and retrieves fewer pages as kOracle increases, limiting the retrieval of irrelevant pages.
Largest-Gap and Score-Cluster over-retrieve across most kOracle . The trend is consistent across all
three encoders and two datasets.
17

B.3 Retrieval Quality vs Oracle: F1-score.
MMLongBench
ViSAR - Colqwen2.5
 ViSAR - Colqwen2.5LongDocURL
ViSAR - ColPali
 ViSAR - ColPali
ViSAR - ColModernVBERT
 ViSAR - ColModernVBERT
Figure S4:F1-score as a function of kOracle .Gray histograms show the corresponding percentage of
queries. ColQwen2.5 shows the best results, followed by ColPali, while ColModernVBERT exhibits
a weaker trend. ViSAR achieves a higher F1-score on MMLongBench, particularly for lower kOracle
where most queries are located. On LongDocURL, ViSAR outperforms Largest-Gap for most kOracle ,
except kOracle = 1, suggesting that Largest-Gap is mostly effective for single-page queries, which
constitute the majority of queries. Overall, ViSAR provides a more consistent adaptive retrieval, in
accordance with its compact but more varied retrieved set size in Figure S1 and S2
18

B.4 Retrieval Quality vs Oracle: Precision.
MMLongBench LongDocURL
ViSAR - Colqwen2.5
 ViSAR - Colqwen2.5
ViSAR - ColPali
 ViSAR - ColPali
ViSAR - ColModernVBERT
 ViSAR - ColModernVBERT
Figure S5:Precision as a function of kOracle .Gray histograms show the corresponding percentage
of queries. ColQwen2.5 shows the best results, followed by ColPali, while ColModernVBERT
exhibits a weaker trend. ViSAR achieves a higher precision on MMLongBench, particularly for
lower kOracle where most queries are located. On LongDocURL, when using ColQwen2.5, ViSAR
and Largest-Gap achieve competitive performance on most kOracle , except kOracle = 1, while ColPali
and ColModernVBERT favour Largest-Gap.
19

B.5 Retrieval Quality vs Oracle: Recall.
MMLonBench LongDocURL
ViSAR - Colqwen2.5
 ViSAR - Colqwen2.5
ViSAR - ColPali
 ViSAR - ColPali
ViSAR - ColModernVBERT
 ViSAR - ColModernVBERT
Figure S6:Precision as a function of kOracle .Gray histograms show the corresponding percentage
of queries. ColQwen2.5 shows the best results, followed by ColPali, while ColModernVBERT
exhibits a weaker trend. While Oracle always maximises Recall, by definition, adaptive methods
favour lower numbers of retrieved pages on average. They reduce the number of irrelevant pages as
kOracle increases, with Largest-Gap and Score-Cluster shows higher precision overall than ViSAR,
consistent with their larger numbers of retrieved pages at eachk Oracle in Figure S3.
20

B.6 Ranking Quality Metrics
Table S1: Page ranking evaluation on MMLongBench and LongDocURL, comparing ViSAR and late-
interaction at ranks 5 and 10, reporting Recall, normalized discounted cumulative gain (NDCG), hit
rate (Hits), and mean reciprocal rank (MRR). ViSAR consistently improves page ranking, contributing
to the effectiveness of its adaptive retrieval behaviour. ColQwen2.5 shows the best results, followed
by ColPali, while ColModernVBERT exhibits more modest improvements.
(a) MMLongBench
Recall (%) NDCG (%) Hits MRR
Ranking Method@5 @10 @5 @10 @5 @10 @5 @10
Using ColPali
Late-Interaction 75.00 84.49 0.730 0.746 84.92 90.97 0.696 0.704
ViSAR (ours)76.36 86.68 0.734 0.756 85.39 92.76 0.702 0.712
Using ColQwen2.5
Late-Interaction 78.60 86.82 0.780 0.793 87.89 92.87 0.754 0.760
ViSAR (ours)79.73 87.73 0.790 0.799 88.72 93.23 0.762 0.768
Using ColModernVBERT
Late-Interaction 72.79 83.14 0.704 0.723 83.14 90.26 0.668 0.678
ViSAR (ours)73.66 83.56 0.709 0.725 83.85 90.26 0.671 0.679
(b) LongDocURL
Recall (%) NDCG (%) Hits MRR
Ranking Method@5 @10 @5 @10 @5 @10 @5 @10
Using ColPali
Late-Interaction 75.07 83.15 0.777 0.788 88.72 93.97 0.750 0.758
ViSAR (ours)76.44 84.44 0.790 0.798 90.14 95.01 0.762 0.769
Using ColQwen2.5
Late-Interaction 77.54 84.98 0.805 0.810 90.70 94.88 0.779 0.785
ViSAR (ours)78.69 85.69 0.817 0.820 91.56 95.09 0.793 0.798
Using ColModernVBERT
Late-Interaction 71.55 81.10 0.724 0.739 85.18 92.34 0.690 0.700
ViSAR (ours)72.30 81.57 0.736 0.748 85.88 92.38 0.703 0.712
21

C ViSAR Sensitivity Analyses
We analyse the sensitivity of ViSAR to T(Equation 11 in main text ) and γ(Equation 14 in main text). Figures
S7 and S8 show NDCG and Recall at ranks 5 and 10 as functions of T∈[1,103]. Figure S9 shows k⋆as a
function ofγ∈[1,108].
C.1 Ranking Sensitivity toTin Equation 11
NDGC@5 Gains NDGC@5 Gains Recall@5 Gains Recall@5 Gains
top-T top-T top-Ttop-T top-T top-T0.012
0.010
0.008
0.006
0.004
0.002
0.0000.012
0.010
0.008
0.006
0.004
0.002
0.0000.012
0.010
0.008
0.006
0.004
0.002
0.000
1 510 25 50 75100 150 200 300 500 750 1000 1 510 25 50 75100 150 200 300 500 750 1000 1 510 25 50 75100 150 200 300 500 750 1000
1 510 25 50 75100 150 200 300 500 750 1000 1 510 25 50 75100 150 200 300 500 750 1000 1 510 25 50 75100 150 200 300 500 750 1000
1 510 25 50 75100 150 200 300 500 750 1000 1 510 25 50 75100 150 200 300 500 750 1000 1 510 25 50 75100 150 200 300 500 750 1000
1 510 25 50 75100 150 200 300 500 750 1000 1 510 25 50 75100 150 200 300 500 750 1000 1 510 25 50 75100 150 200 300 500 750 1000
MMLongBench : Colpali ColQwen ColModernVBERT Late-Interaction
1.75
1.50
1.25
1.00
0.75
0.50
0.25
0.001.75
1.50
1.25
1.00
0.75
0.50
0.25
0.001.75
1.50
1.25
1.00
0.75
0.50
0.25
0.00
0.018
0.015
0.012
0.010
0.007
0.005
0.003
0.0000.018
0.015
0.012
0.010
0.007
0.005
0.003
0.0000.018
0.015
0.012
0.010
0.007
0.005
0.003
0.000
1.75
1.50
1.25
1.00
0.75
0.50
0.25
0.001.75
1.50
1.25
1.00
0.75
0.50
0.25
0.001.75
1.50
1.25
1.00
0.75
0.50
0.25
0.00Recall@5 = 78.60
NDCG@5 = 0.777 NDCG@5 = 0.805 NDCG@5 = 0.724
Recall@5 = 75.07 Recall@5 = 77.54 Recall@5 = 71.55NDCG@5 = 0.730 NDCG@5 = 0.780 NDCG@5 = 0.704
Recall@5 = 75.00 Recall@5 = 72.79
LongDocURL : Colpali ColQwen ColModernVBERT Late-Interaction
Figure S7:Gains in NDCG@5 and Recall@5 as a function of T∈[1,103]– Equation 11 in the
main text. Gains are relative to late-interaction (zero dashed line) and are shown for both datasets.
When using ColQwen2.5 and ColPali, all Tvalues yield improvements over late-interaction across
metrics and datasets. When using ColModernVBERT, only lower Tyield consistent improvements.
No clear trends emerges, and the optimal Tvalue depends on the metric, encoder and dataset. As no
singleTis uniformly optimal,T= 50was selected empirically as a compromise.
22

Gains in NDGC@10 Gains in NDGC@10 Gains in Recall@10 Gains in Recall@10
top-T top-T top-Ttop-T top-T top-T0.0120.014
0.010
0.008
0.006
0.004
0.002
0.0000.0120.014
0.010
0.008
0.006
0.004
0.002
0.0000.0120.014
0.010
0.008
0.006
0.004
0.002
0.000
1 510 25 50 75100 150 200 300 500 750 1000 1 510 25 50 75100 150 200 300 500 750 1000 1 510 25 50 75100 150 200 300 500 750 1000
1 510 25 50 75100 150 200 300 500 750 1000 1 510 25 50 75100 150 200 300 500 750 1000 1 510 25 50 75100 150 200 300 500 750 1000
1 510 25 50 75100 150 200 300 500 750 1000 1 510 25 50 75100 150 200 300 500 750 1000 1 510 25 50 75100 150 200 300 500 750 1000
1 510 25 50 75100 150 200 300 500 750 1000 1 510 25 50 75100 150 200 300 500 750 1000 1 510 25 50 75100 150 200 300 500 750 10002.50
2.00
1.50
1.00
0.50
0.002.50
2.00
1.50
1.00
0.50
0.002.50
2.00
1.50
1.00
0.50
0.00
0.014
0.012
0.010
0.008
0.006
0.004
0.002
0.0000.014
0.012
0.010
0.008
0.006
0.004
0.002
0.0000.014
0.012
0.010
0.008
0.006
0.004
0.002
0.000
1.40
1.20
1.00
0.80
0.60
0.40
0.20
0.001.40
1.20
1.00
0.80
0.60
0.40
0.20
0.00
1.40
1.20
1.00
0.80
0.60
0.40
0.20
0.00
NDCG@10 = 0.788 NDCG@10 = 0.810
Recall@10 = 83.15NDCG@10 = 0.746NDCG@10 = 0.793 NDCG@10 = 0.723
Recall@10 = 84.49 Recall@10 = 86.82
Recall@10 = 83.14
NDCG@10 = 0.739
Recall@10 = 84.98 Recall@10 = 81.10
MMLongBench : Colpali ColQwen ColModernVBERT Late-Interaction
LongDocURL : Colpali ColQwen ColModernVBERT Late-InteractionFigure S8:Gains in NDCG@10 and Recall@10 as a function of T∈[1,103]–Equation 11
in main text. The gains are relative to late-interaction (zero dashed line) and are shown for both
datasets. When using ColQwen2.5 and ColPali, all Tvalues yield consistent improvements over
late-interaction across metrics and datasets. When using ColModernVBERT, only lower Tyield
improvements. No clear trends emerges, and the optimal Tvalue depends on the metric, encoder and
dataset. As no singleTis uniformly optimal, T=50 was selected empirically as a compromise.
23

C.2 Adaptive-kSensitivity toλin Equation 14
Mean k* Mean |Δk*|
101102103104105106107108
101100
γ102103104105106107108
18
16
14
12
10
8
6
4
2012
10
8
6
4
2
101102103104105106107108
101100
γ10210310410510610710816
14
12
10
8
6
4
2012
10
8
6
4
2
101100
γ102103104105106107108
012
10
8
6
4
2
14
12
10
8
6
4
2Mean k* Mean |Δk*|025
20
15
10
5
025
20
15
10
5
50
40
30
20
10
250
40
30
20
10
2101102103104105106107108
101102103104105106107108101102103104105106107108101102103104105106107108
Colpali Mean (                                                                         ) ColQwen ColModernVBERT Median MMLongBench : Q1-Q3
025
20
15
10
5
101100
γ102103104105106107108101100
γ102103104105106107108101100
γ1021031041051061071085060
40
30
20
10
2
Colpali Mean (                                                                         ) ColQwen ColModernVBERT Median LongDocuURL : Q1-Q3
Figure S9: k⋆as a function of γ∈[1,108](Equation 14 in the main text).Results are shown
for each dataset.Top:Mean( |∆k⋆|) across queries, computed as the average absolute change in k⋆
between consecutive values γiandγi−1, with the abscissa reporting γi.Bottom:Mean( k⋆) across
queries as a function of γ. Small values of γtend to produce unstable and overly large values of
k⋆. Beyond approximately γ= 103, both Mean( |∆k⋆|) and Mean( k⋆) stabilize,indicating that
adaptive-kretrieval is largely insensitive to the precise choice ofγin this regime.
24

D ViSAR Accuracy
D.1 Comparison with Literature Retrieval Baselines
Table S2: Answer generation accuracy across retrieval methods. Qwen2.5-VL-7B-Instruct is used
as the generation model for all visual methods, while ColBERTv2 operates on OCR-extracted text
(Tesseract (Smith 2007)) with Qwen2.5-7B-Instruct. Max- kdenotes an input budget of at most
kpages. Values are mean accuracies with 95% confidence intervals in parentheses, obtained via
bootstrap resampling over queries (10,000 samples). Bold indicates highest accuracy. ViSAR achieves
competitive or superior accuracy across all settings.
(a)MMLongBench
Retrieval MethodMax-5 Max-10
Fixed top-k
ColBERTv2 24.51 (21.90 – 27.12) 24.70 (22.18 – 27.31)
M3DocRAG 34.86 (32.06 – 37.74) 35.08 (32.28 – 37.96)
VisRAG-Ret 34.48 (31.59 – 37.28) 35.69 (32.81 – 38.58)
Adaptive-k(using ColQwen2.5)
Largest-Gap 35.79 (32.90 – 38.68) 35.88 (32.99 – 38.77)
Score-Cluster 36.25 (33.36 – 39.14) 35.97 (33.08 – 38.77)
ViSAR (ours)36.53(33.64 – 39.32)36.63(33.74 – 39.52)
Adaptive-k(using ColPali)
Largest-Gap 33.83 (31.03 – 36.72) 34.20 (31.41 – 37.00)
Score-Cluster 33.83 (31.03 – 36.63) 34.02 (31.22 – 36.91)
ViSAR (ours) 35.42 (32.53 – 38.30) 35.88 (32.99 – 38.77)
Adaptive-k(using ColModernVBERT)
Largest-Gap 32.99 (30.02 – 35.79) 33.36 (30.57 – 36.16)
Score-Cluster 34.48 (31.59 – 37.28) 35.04 (32.25 – 37.84)
ViSAR (ours) 34.01 (31.19 – 36.88) 34.28 (31.54 – 37.22)
(b) LongDocURL
Retrieval MethodMax-5 Max-10
Fixed top-k
ColBERTv2 47.18 (45.12 – 49.20) 47.70 (45.68 – 49.76)
M3DocRAG 59.31 (57.29 – 61.38) 58.71 (56.69 – 60.69)
VisRAG-Ret 57.29 (55.27 – 59.31) 58.02 (56.04 – 60.04)
Adaptive-k(using ColQwen2.5)
Largest-Gap 61.01 (59.03 – 62.99) 60.89 (58.92 – 62.92)
Score-Cluster 60.00 (57.98 – 62.02) 59.83 (57.85 – 61.81)
ViSAR (ours)61.06(59.12 – 63.04)60.97(59.00 – 62.95)
Adaptive-k(using ColPali)
Largest-Gap 60.04 (58.06 – 62.06) 60.47 (58.49 – 62.44)
Score-Cluster 59.53 (57.51 – 61.59) 59.70 (57.72 – 61.68)
ViSAR (ours) 60.77 (58.71 – 62.80) 60.13 (58.11 – 62.15)
Adaptive-k(using ColModernVBERT)
Largest-Gap 58.00 (55.92 – 60.09) 57.91 (55.82 – 60.00)
Score-Cluster 58.05 (55.96 – 60.19) 58.56 (56.47 – 40.60)
ViSAR (ours) 58.28 (56.16 – 60.13) 58.33 (56.24 – 60.37)
25

D.2 Comparison with Fixed Top-kRetrieval across 5 LVLM Models
The following 5 tables show answer generation accuracy using 5 different LVLMs, comparing late-interaction
fixed top-k with ViSAR adaptive- kretrieval. Max- kdenotes an input budget of at most kpages, where fixed
top-kstrategies maximise the input budget, adaptive- kstrategies dynamically adapt it. Bold indicates statistically
significant differences according to McNemar’s test ( p <0.05 ). Across 60 configurations (3 encoders ×5
LVLMs ×2 Max- kbudgets ×2 datasets), ViSAR improves accuracy in 24 cases and matches late-interaction in
the remaining 36, with no significant decrease observed. The largest improvements are observed for Idefics3-8B-
Llama3 LVLM (Table S5) and Llava-OV-Qwen2-7B LVLM (Table S6), which were more sensitive to longer
contexts, consistent with ViSAR reducing both the number of irrelevant pages and the total number of pages
(Table 1 and Figure 3 in the main text). The best performance overall are obtained using ColQwen2.5 encoder
and Qwen3-VL-8B-Instruct LVLM (Table S3), both independently and in combination.
Table S3:Qwen2.5-VL-7B-Instruct answer accuracywith an input budget of Max-5 and Max-10
pages, comparing late-interaction fixed top-k and ViSAR adaptive- k. Bold indicates statistically
significant differences, according to McNemar’s test (p <0.05).
MMLongBench LongDocURL
Retrieval MethodMax-5 Max-10 Max-5 Max-10
Using ColQwen2.5
Fixed top-k35.04 35.69 59.79 59.27
ViSAR (ours) 36.53 36.6360.86 60.77
Using ColPali
Fixed top-k34.86 34.86 59.31 58.71
ViSAR (ours) 35.42 35.8860.77 60.13
Using ColModernVBERT
Fixed top-k34.79 34.95 58.42 58.84
ViSAR (ours) 33.94 34.26 58.28 58.33
Table S4:Qwen3-VL-8B-Instruct answer accuracywith an input budget of Max-5 and Max-10
pages, comparing late-interaction fixed top-k and ViSAR adaptive- k. Bold indicates statistically
significant differences, according to McNemar’s test (p <0.05).
MMLongBench LongDocURL
Retrieval MethodMax-5 Max-10 Max-5 Max-10
Using ColQwen2.5
Fixed top-k40.07 40.73 66.75 67.69
ViSAR (ours) 39.33 38.77 67.27 67.18
Using ColPali
Fixed top-k38.68 39.51 64.56 67.35
ViSAR (ours) 37.93 38.68 65.64 66.88
Using ColModernVBERT
Fixed top-k39.14 38.96 63.85 65.85
ViSAR (ours) 38.30 38.68 63.29 65.99
26

Table S5:InternVL3.5-8B-Instruct answer accuracywith an input budget of Max-5 and Max-10
pages, comparing late-interaction fixed top-k and ViSAR adaptive- k. Bold indicates statistically
significant differences, according to McNemar’s test (p <0.05).
MMLongBench LongDocURL
Retrieval MethodMax-5 Max-10 Max-5 Max-10
Using ColQwen2.5
Fixed top-k31.69 29.64 58.54 55.40
ViSAR (ours) 33.7433.74 60.68 60.17
Using ColPali
Fixed top-k32.15 30.75 58.67 55.74
ViSAR (ours) 32.71 32.53 59.5759.01
Using ColModernVBERT
Fixed top-k31.03 30.94 57.54 55.55
ViSAR (ours) 30.66 30.48 59.01 56.33
Table S6:Idefics3-8B-Llama3 answer accuracywith an input budget of Max-5 and Max-10 pages,
comparing late-interaction fixed top-k and ViSAR adaptive- k. Bold indicates statistically significant
differences, according to McNemar’s test (p <0.05).
MMLongBench LongDocURL
Retrieval MethodMax-5 Max-10 Max-5 Max-10
Using ColQwen2.5
Fixed top-k22.27 20.60 40.90 37.94
ViSAR (ours) 23.5823.39 42.58 41.59
Using ColPali
Fixed top-k21.76 19.66 40.77 37.76
ViSAR (ours) 22.8322.8341.7641.20
Using ColModernVBERT
Fixed top-k21.81 20.69 40.84 37.87
ViSAR (ours) 20.88 20.88 41.02 38.52
Table S7:Llava-OV-Qwen2-7B answer accuracywith an input budget of Max-5 and Max-10 pages,
comparing late-interaction fixed top-k and ViSAR adaptive- k. Bold indicates statistically significant
differences, according to McNemar’s test (p <0.05).
MMLongBench LongDocURL
Retrieval MethodMax-5 Max-10 Max-5 Max-10
Using ColQwen2.5
Fixed top-k16.03 10.25 26.11 16.95
ViSAR (ours)20.69 19.94 32.56 30.32
Using ColPali
Fixed top-k15.94 9.97 26.41 17.03
ViSAR (ours)19.76 20.04 30.80 27.14
Using ColModernVBERT
Fixed top-k15.66 9.32 25.66 16.15
ViSAR (ours)18.27 16.5926.8220.97
27

E ViSAR RAG Latency
E.1 Comparison with Largest-Gap and Score-Cluster
Table S8:Latencies comparing ViSAR, Largest-Gap, and Score-Cluster.Max- kdenotes an input
budget of at most kpages, where late-interaction fixed top- kretrieval always maximises the LVLM
input budget and adaptive methods only do so when required. Values are average latencies for retrieval,
generation, and end-to-end (retrieval + generation), with gain percentages over late-interaction in
parentheses. All adaptive methods significantly improved generation latency. Although ViSAR
introduces a retrieval overhead, it contributes only marginally as by achieving a substantial reduction
in generation latency that dominates the total cost. This results in end-to-end latency reductions of up
to 58.7% on MMLongBench and 38.5% on LongDocURL at a Max-10 LVLM budget. In contrast,
Score-Cluster end-to-end latency is consistently the slowest method, while Largest-Gap end-to-end
latency provides only a noticeable advantage on LongDocURL, with gains of up to 47.2% over fixed
top-kat Max-10 budget.
(a) Retrieval latency
MMLongBench LongDocURL
Fixed top-k
Late-Interaction 0.049 0.009
Adaptive-k
Clustering 0.118 0.192
Largest-Gap 0.053 0.009
ViSAR (ours) 0.050 0.106
(a) Generation latency
MMLongBench LongDocURL
MethodMax-5 Max-10 Max-5 Max-10
Fixed top-k
Late-Interaction 4.88 10.05 5.20 10.65
Adaptive-k
Clustering 4.48 (-8.2%) 6.61 (-34.2%) 4.67 (-10.2%) 8.17 (-23.3%)
Largest-Gap 3.30 (-33.4%) 5.40 (-46.2%)3.75(-27.9%)5.62(-47.2%)
ViSAR (ours)3.23(-33.8%)4.12(-59.0%) 4.21 (-19.0%) 6.44 (-39.5%)
(a) End-to-end latency
MMLongBench LongDocURL
MethodMax-5 Max-10 Max5 Max-10
Fixed top-k
Late-Interaction 4.93 10.10 5.21 10.66
Adaptive-k
Clustering 4.60 (-6.69%) 6.73 (-33.37%) 4.86 (-6.72%) 8.36 (-21.58%)
Largest-Gap 3.35 (-32.05%) 5.45 (-46.04%)3.76(-27.83%)5.63(-47.19%)
ViSAR (ours)3.28(-33.47%)4.17(-58.71%) 4.32 (-17.08%) 6.55 (-38.51%)
28

E.2 Acceleration Strategies for Very Long Documents
E.2.1 Approximation of the Similarity Matrix
Although the retrieval overhead of ViSAR is negligible compared to its generation latency gains (main text
Figure 4, Table S8), it may become noticeable for very long documents. In such cases, we provide strategies to
approximate the similarity matrix and preserve latency gains. We introduce ViSAR (Approx.), which reduces
computational cost by leveraging sparsity ofeffective contributions. Since Equation 11 aggregates only the
highest-weighted patch interactions, Equation 10 can be restricted to the top- jpatches per page and the top- p
pages ranked by maximum patch weight. Additionally, the search over J(k) can be limited to the Max- k
input budget of the LVLM. We therefore set s= 10 , and empirically choose j= 450 andp= 75 , which
preserveRecall and k⋆across all evaluations. Therefore, although ViSAR (Approx.) requires tuning additional
hyperparameters tuning, it substantially reduces computational cost (Figure S11) while preserving retrieval
performance. In practice, these hyperparameters only require lower bound values: retrieval performance remains
unchanged oncejandpare sufficiently large, and only degrades when they are set too low.
E.2.2 Latency Decomposition across Document sizes
2 4 6 8 1014
12
10
8
6
4
2
All queriesEnd-to-end latency: MMLongBench
End-to-end latency: LongDocURLMax-k retrieved pages2 4 6 8 1014
12
10
8
6
4
2
100-150 page documents
2 4 6 8 1014
12
10
8
6
4
2
Max-k retrieved pages2 4 6 8 1014
12
10
8
6
4
2151-200 page documents
2 4 6 8 1014
12
10
8
6
4
2
Max-k retrieved pages2 4 6 8 1014
12
10
8
6
4
2468 page document
2 4 6 8 1012
10
8
6
4
2Latency (s) Latency (s)
Max-k retrieved pages2 4 6 8 1012
10
8
6
4
2
14
12
10
8
6
4
2
0
14
12
10
8
6
4
2
02 4 6 8 10
Max-k retrieved pages2 4 6 8 10All QueriesLatency (s) Latency (s)75-100 page documents 100-125 page documents 125-150 page documents
14
12
10
8
6
4
2
014
12
10
8
6
4
2
0
2 4 6 8 10
Max-k retrieved pages2 4 6 8 10
14
12
10
8
6
4
2
0
2 4 6 8 10
Max-k retrieved pages2 4 6 8 10
14
12
10
8
6
4
2
02 4 6 8 10
Max-k retrieved pages2 4 6 8 10
10
8
6
4
2
10
8
6
4
2
ViSAR (Approx.)
Fixed top-k
ViSAR (Approx.)
Fixed top-k
ViSAR
Fixed top-k
ViSAR
Fixed top-k
Figure S10:End-to-end (retrieval + generation) latency across document sizes.(Shaded: std).
ViSAR (Approx.) Introduces computational shortcuts that stillpreserve exact retrieval results, leaving
the retrieved sets unchanged. We therefore report only end-to-end latency as a function of document
size, since retrieval and generation are identical. Both ViSAR and ViSAR (Approx.) yield significant
latency gains on almost all documents. The only case with increased latency is ViSAR on queries
targeting the largest document (468 pages), whereas ViSAR (Approx.) still improves end-to-end
latency on the same queries, albeit more modestly.
29

E.2.3 Latency Decomposition across ViSAR’s Components
1-50 51-100 101-150 151-200 201-499
ViSAR
ViSAR (Approx.)
1-50 51-100 101-150 151-200 201-4992.5
2.0
1.5
1.0
0.5
0.0Latency (s)
1.0
0.8
0.6
0.4
0.4
0.0
-0.2Latency (s)0.35
0.30
0.25
0.20
0.15
0.10
0.05
0.00Latency (s)0.100.12
0.08
0.06
0.04
0.02
0.00Latency (s)
Weights Similarity matrix J(k) Total
50-75 76-100 101-125 126-150
# Document pages50-75 76-100 101-125 126-150
# Document pages
MMLongBench
LongDocURL
Figure S11:Retrieval latency decomposition across components.ViSAR (Approx.) introduces
computational shortcuts (section E.2.1) that stillpreserve exact retrieval results. Most of the compu-
tational cost comes from similarity matrix computation and optimization of J(k) . ViSAR (Approx.)
substantially reduces the impact of both stages and on both datasets.
E.2.4 Latency Decomposition across Acceleration Strategies
1-50 51-100 101-150 151-200 468
# Document pagesMMLongBench LongDocURL
50-75 76-100 101-125 126-150
# Document pages2.0
1.5
1.0
0.5
0.0Latency (s)0.4
0.3
0.2
0.1
0.0 Fixed top-kViSAR (Approx.)top_j, top_ptop_p, top_stop_ptop_j, top_stop_sViSARtop_j
Fixed top-kViSAR (Approx.)top_j, top_p
top_p, top_stop_p
top_j, top_stop_sViSAR
top_j
Figure S12:Retrieval latency decomposition across acceleration strategies.ViSAR (Approx.)
introduces computational shortcuts (section E.2.1) that stillpreserve exact retrieval results. top- j:
keep the top- jpatches per page for similarity matrix computation. top- p: keep the top- ppages
– ranked by their maximum patch weight – for similarity matrix computation. top- s: restrict the
search space of J(k) to the top- spages – based on their self-similarity sp= Sim(p, p) . Although
the effectiveness of individual shortcuts varies across datasets, combining top- j, top- p, and top- s
consistently yields the largest improvement, substantially reducing latency for larger documents.
30

E.2.5 Retrieval, Generation, End-to-end latencies.
# Document pages Max-k retrieved pagesRetrieval Latency Generation Latency End-to-end Latency
Max-k retrieved pages50-75 76-100 101-125 126-1501.0
0.8
0.6
0.4
0.2
0.0Latency (s)
2 4 6 8 1014
12
10
8
6
4
2
014
12
10
8
6
4
2
0
2 4 6 8 10
2 4 6 8 10 2 4 6 8 1014
12
10
8
6
4
214
12
10
8
6
4
22.5
2.0
1.5
1.0
0.5
0.0
1-50 51-100 101-150 151-200 468Latency (s)
MMLongBench
LongDocURL
ViSAR
Fixed top-kViSAR (Approx.)
Figure S13:Average RAG Latency.(shaded: std). ViSAR (Approx.) introduces computational
shortcuts (section E.2.1) that stillpreserve exact retrieval results. On average, ViSAR and ViSAR
(Approx.) achieve nearly identical end-to-end latency, as the average retrieval overhead remains
negligible compared to generation latency. ViSAR (Approx.) becomes beneficial only for very long
documents, where retrieval overhead is more pronounced (Figure S10).
31

F ViSAR Similarity Matrix Structure Analysis
Accuracy (%)74
72
70
68
66
64
62
60
58LongDocURL
50 60 70 80 90
Similarity Matrix Sparsity (%)
max-5
max-10
ColQwen2.5ColPali
ColModernVBERT
Figure S14: Answer accuracy as a function of the similarity matrix level of sparsity (%), on Long-
DocURL. It shows that higher sparsity consistently correlates with higher accuracy across encoders.
Table S9: Average level of similarity matrix sparsity (%), conditioned on LVLM answer binary
correctness, using Qwen2.5-VL-7B-Instruct. Max- kdenotes the maximum input budget of the
LVLM. Sparsity levels are reported as mean±std with the observed min–max range.
(a) MMLongBench
Similarity Matrix Sparsity (%)
Encoder Answer TypeMax-5 Max-10
ColQwen2.5-v0.1 Correct60.21±22.47(10–98.11)60.14±22.64(10–98.11)
Wrong55.14±21.19(6.67–98.53)55.17±21.10(6.67–98.53)
All56.99±21.81(6.67–98.53)56.99±21.81(6.67–98.53)
ColPali-v1.2 Correct58.59±20.99(5–98.11)57.99±21.04(5–98.11)
Wrong54.30±19.67(5.88–98.53)54.60±19.70(5.88–98.53)
All55.82±20.25(5–98.53)55.82±20.25(5–98.53)
ColModernVBERT Correct46.98±18.53(5.88–94.12)46.70±18.43(5.88–94.12)
Wrong43.74±16.27(2.94–91.67)43.86±16.35(2.94–91.67)
All44.83±17.14(2.94–94.12)44.83±17.14(2.94–94.12)
(a) LongDocURL
Similarity Matrix Sparsity (%)
Encoder Answer TypeMax-5 Max-10
ColQwen2.5-v0.1 Correct63.17±22.29(0.83–98.77)63.23±22.17(0.83–98.77)
Wrong58.67±21.93(0.77–97.78)58.58±22.10(0.77–97.78)
All61.41±22.26(0.77–98.77)61.41±22.26(0.77–98.77)
ColPali-v1.2 Correct61.41±19.91(1.45–98.36)61.87±19.75(1.45–98.36)
Wrong55.05±20.22(1.49–98.28)54.45±20.23(1.49–98.28)
All58.91±20.27(1.45–98.36)58.91±20.27(1.45–98.36)
ColModernVBERT Correct42.66±16.96(4.40–94.44)42.65±17.02(4.40–94.44)
Wrong39.29±16.20(3–89.47)39.30±16.13(3–93.65)
All41.25±16.73(3–94.44)41.25±16.73(3–94.44)
32

G ViSAR Ablation study
We evaluate simplified variants of ViSAR by disabling the query embedding weights wq, page weights wp, patch
embedding weights wp,j, and by replacing the optimization objective J(k) with fixed top- k, Largest-Gap, and
Score-Cluster retrieval.
Table S10:Ablation study of ViSAR.It uses ColPali encoder and Qwen2.5-VL-7B-Instruct LVLM.
(a) Effect of the weighting strategy on page ranking, evaluated suing normalized discounted cumula-
tive gain (NDCG: N) and Recall (R) at ranks 5 and 10. (b) Effect on accuracy of replacing the adaptive
retrieval objective J(k) with fixed top- k, Largest-Gap, and Score-Cluster retrieval methods. Max-10
denotes an LVLM input budget of at most 10 pages. Each modification degrades retrieval quality
and downstream answer accuracy, indicating that both the weighting strategy and the optimization
objective contribute to ViSAR’s performance. Furthermore, setting wp
j= 1removes the patch-level
weighting normalization in Equation 10, causing all self-similarities sp= Sim(p, p) to become equal
to 1 and the optimization problem to degenerate, as no page ranking is possible.
(a) Ranking quality: Weighting strategy ablation
MMLongBench LongDocURL
VariantN@5 N@10 R@5 R@10 N@5 N@10 R@5 R@10
ViSAR 0.7340.756 76.36 86.68 0.790 0.798 76.44 84.44
wi= 10.7350.748 75.04 85.15 0.772 0.779 74.52 83.08
wp= 10.684 0.702 70.48 81.50 0.716 0.730 71.07 80.45
wi=wp= 10.615 0.646 66.09 80.15 0.561 0.594 58.29 72.50
wp
j= 1Degenerate Degenerate
(b) Answer Accuracy: Adaptive retrieval objective ablation
MMLongBench LongDocURL
VariantMeank⋆Max-10 Meank⋆Max-10
ViSAR 5.335.888.160.13
Fixed top-10– 35.78 – 58.84
Largest-Gap 1.7 33.55 2.2 59.36
Score-Cluster 11.6 35.14 22.6 59.18
33