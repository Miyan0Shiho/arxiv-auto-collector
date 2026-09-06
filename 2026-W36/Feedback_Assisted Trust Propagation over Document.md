# Feedback-Assisted Trust Propagation over Document Relation Graphs for Retrieval-Augmented Generation

**Authors**: Zhuoheng Li, Ying Chen

**Published**: 2026-09-01 01:30:15

**PDF URL**: [https://arxiv.org/pdf/2609.00543v1](https://arxiv.org/pdf/2609.00543v1)

## Abstract
Retrieval-augmented generation (RAG) systems rely on external corpora that may contain outdated, contradictory, noisy, or unreliable documents, introducing reliability risks. Prior work has leveraged document relations to improve the answer reliability of RAG. To propagate reliability signals beyond directly compared document pairs, we propose TrustPropRAG, which structures document relations as a graph and estimates document reliability through multi-hop propagation across the graph. TrustPropRAG anchors this propagation with a limited set of human feedback on document reliability, extending these costly-to-collect feedback-based reliability signals across the whole corpus. Specifically, based on the constructed document relation graph, TrustPropRAG estimates a trust score for each document by formulating and solving an optimization problem that jointly captures pairwise document relations and user feedback. These scores are then used to improve the selection of reliable documents and support trust-aware answer generation. Evaluation results show that TrustPropRAG improves both retrieval quality and exact match over baselines, and remains robust under sparse and noisy feedback.

## Full Text


<!-- PDF content starts -->

Feedback-Assisted Trust Propagation over Document Relation Graphs for
Retrieval-Augmented Generation
Zhuoheng Li, Ying Chen
College of Information Sciences and Technology, The Pennsylvania State University
{zml5515,yingchen}@psu.edu
Abstract
Retrieval-augmented generation (RAG) sys-
tems rely on external corpora that may con-
tain outdated, contradictory, noisy, or unreli-
able documents, introducing reliability risks.
Prior work has leveraged document relations
to improve the answer reliability of RAG.
To propagate reliability signals beyond di-
rectly compared document pairs, we propose
TrustPropRAG , which structures document re-
lations as a graph and estimates document re-
liability through multi-hop propagation across
the graph. TrustPropRAG anchors this propa-
gation with a limited set of human feedback on
document reliability, extending these costly-
to-collect feedback-based reliability signals
across the whole corpus. Specifically, based
on the constructed document relation graph,
TrustPropRAG estimates a trust score for each
document by formulating and solving an opti-
mization problem that jointly captures pairwise
document relations and user feedback. These
scores are then used to improve the selection of
reliable documents and support trust-aware an-
swer generation. Evaluation results show that
TrustPropRAG improves both retrieval quality
and exact match over baselines, and remains
robust under sparse and noisy feedback.
1 Introduction
Retrieval-augmented generation (RAG) has
emerged as an effective approach for mitigating
the limitations of large language models (LLMs),
whose parametric knowledge may be incomplete or
outdated (Lewis et al., 2020; Borgeaud et al., 2022;
Izacard et al., 2023). By retrieving knowledge
from external corpora, LLMs can incorporate
relevant and up-to-date information during
answer generation. However, RAG’s reliance on
external corpora can introduce reliability risks,
as the corpora may contain noisy, outdated, or
contaminated content (Zhong et al., 2023; Zou
et al., 2025), as well as knowledge conflicts (Xie
et al., 2024; Chen et al., 2022).Prior work aims to improve RAG reliability in
both the retrieval (Ma et al., 2023) and answer gen-
eration (Wang et al., 2025a) phases. Notably, a line
of work leverages relationships among documents
to improve the answer reliability of RAG, such
as aggregating answers from different documents
through majority agreement (Xiang et al., 2024), or
selecting a subset of documents that are mutually
consistent before answer generation (Shen et al.,
2025). Document relations can be inferred by us-
ing natural language inference (NLI) models or
leveraging source metadata.
We propose TrustPropRAG, a framework that
structures document relations as a graph, as graph
representations enable information to propagate
throughmulti-hop relations. Recent studies have
explored graph-enhanced RAG, but they mainly
focus on organizing knowledge graphs (Edge
et al., 2024), or constructing index graphs for re-
trieval (Guo et al., 2025). In contrast, we use the
graph structure to estimate document reliability and
propagate these estimates across the corpus through
multi-hop relations. Graph propagation, however,
requires anchoring, as document relations reveal
consistency or conflict, but not document reliability
itself. We anchor this propagation with a set of hu-
man feedback on document reliability. In practice,
explicit signals (e.g., thumbs-up or thumbs-down
ratings) and implicit signals (e.g., clicks or dwell
time) can be attributed to the documents that con-
tribute to the generated answer using traceback
methods (Wang et al., 2025b; Cohen-Wang et al.,
2024), thereby providing positive or negative sig-
nals for those documents. Since user feedback
is costly to collect, TrustPropRAG leverages doc-
ument relation graphs to propagate sparse feed-
back across the corpus, similar to how label propa-
gation methods in semi-supervised learning (Zhu
et al., 2003; Zhou et al., 2003) spread a few labeled
examples across many unlabeled ones. TrustPro-
pRAG then turns the propagated document reli-
arXiv:2609.00543v1  [cs.AI]  1 Sep 2026

ability scores into performance improvements in
retrieval and answer generation.
To estimate document reliability and propagate
estimates across the corpus, TrustPropRAG first
constructs a document relation graph, where nodes
represent documents and edges capture document
relations (e.g., shared source, mutual support, or
contradiction). Based on this graph, TrustPro-
pRAG estimates a trust score for each document
by formulating and solving an optimization prob-
lem. The problem includes two objective function
terms to jointly capture pairwise document rela-
tions and incorporate user feedback. At query time,
these trust scores are used for trust-aware rescor-
ing, which helps select more reliable documents,
and for trust-aware prompting, where the LLM is
informed of document reliability during answer
generation. Trust score optimization is performed
offline over the corpus, amortizing its computa-
tional cost across future queries.
Our contributions are threefold. First, we pro-
pose a framework that estimates document trust
scores by propagating human feedback over a doc-
ument relation graph, and uses these scores to im-
prove both retrieval and answer generation. Second,
we formulate and solve an optimization problem for
trust score estimation. We also provide a theoreti-
cal analysis of how feedback influences trust score
propagation, as well as the convergence rate of the
propagation. Third, we evaluate TrustPropRAG
with three open-domain question-answering (QA)
benchmarks, three retrievers, and six LLMs, and
further test its robustness under sparse and noisy
feedback. Our code is publicly available at https:
//github.com/zhliOvO/TrustPropRAG.
2 Related Work
RAG reliability.A growing body of work has
studied how to improve RAG reliability. For ex-
ample, InstructRAG guides LLMs to denoise re-
trieved contexts using rationales (Wei et al., 2025).
AstuteRAG compares retrieved evidence with the
LLM’s parametric knowledge to handle knowl-
edge conflicts (Wang et al., 2025a). A notable
line of work uses relationships (agreements or con-
tradictions) among documents to improve reliabil-
ity. For instance, RobustRAG isolates retrieved
documents, generates per-document answers in-
dependently, and aggregates them through major-
ity agreement (Xiang et al., 2024). TrustRAG
leverages similarity relations among retrieved doc-uments by clustering them and then filtering out
contaminated documents before generation (Zhou
et al., 2025). ReliabilityRAG detects contradictions
between documents, selects a subset of documents
that are mutually consistent, and produces the final
answer via keyword aggregation (Shen et al., 2025).
However, these relationships are considered within
a subset of documents, and reliability is not prop-
agated beyond the directly compared documents.
To move beyond this one-hop relation, TrustPro-
pRAG structures document relations as a graph
and captures multi-hop relations among documents
through the graph representation.
Graph-empowered RAG.Graphs have long been
used to model relationships among textual evi-
dence in tasks such as multi-document summa-
rization (Christensen et al., 2013) and fact veri-
fication (Zhong et al., 2020). Recent work has
also incorporated graph structures into RAG. For
example, GraphRAG organizes corpora with en-
tity knowledge graphs, and leverages community
summaries for all groups of closely related enti-
ties (Edge et al., 2024). LightRAG incorporates
graph structures into text indexing and relevant in-
formation retrieval for efficient RAG (Guo et al.,
2025). NodeRAG introduces heterogeneous graph
structures to better leverage the structural nature of
graphs (Xu et al., 2025). However, these methods
primarily use graphs to organize knowledge and
guide retrieval. In contrast, we use graph structure
to propagate document reliability estimates across
the corpus that may contain noisy, outdated, or
contradictory information.
Human feedback for enhancing LLMs.Hu-
man feedback has been used to align LLM sys-
tems with user preferences or intents. For example,
InstructGPT trains language models with human
feedback using reinforcement learning (Ouyang
et al., 2022). RAG-Reward uses preference-based
reward models to improve the quality of gener-
ated answers (Zhang et al., 2025a). Pistis-RAG
aligns human feedback by training a ranking model
to improve content ranking and retrieval mecha-
nisms (Bai et al., 2024). Some methods further
study feedback in interactive settings. For exam-
ple, Liu et al. (2025b) leverages implicit feedback
in human–LLM dialogues. Zhang et al. (2025b)
leverages engagement and disengagement signals
for improving LLM-based recommenders. These
works mainly use feedback to fine-tune model be-
havior or improve outputs in human-LLM interac-
tions. TrustPropRAG instead uses limited feedback

to estimate document reliability across the large-
scale corpus without model fine-tuning.
3 Methodology
Our objective is to enhance RAG by selecting and
using more reliable documents from an external
corpus for answer generation. The external doc-
ument corpus may contain unreliable documents,
such as those with misinformation, outdated in-
formation, or conflicting claims (Xu et al., 2024;
Zou et al., 2025). User feedback can indicate
whether certain documents are reliable or unreli-
able. For example, correct and incorrect RAG-
generated answers, reported by users or flagged
by fact-verification systems, can be traced back to
their contributing documents (e.g., using traceback
methods (Wang et al., 2025b; Cohen-Wang et al.,
2024)), providing positive or negative feedback
for the corresponding documents. However, user
feedback can cover only a small fraction of a large
external corpus. To address this, TrustPropRAG
takes advantage of document relations to propagate
sparse feedback across the corpus.
As shown in Figure 1, TrustPropRAG consists
of four main stages. First, we construct a graph
where each node represents a document, and each
edge encodes a relation between two documents,
such as a same-source relation, mutual support,
or contradiction (Section 3.1). Second, based on
the document relation graph, we formulate and
solveoptimization problems for estimating trust
scores(Section 3.2). We estimate a trust score
τi∈[0,1] for each document diby jointly op-
timizing these scores over all documents in the
corpus C={d 1, d2, . . . , d N}withNdocuments.
Third, we performtrust-aware rescoring(Section
3.3). Given a user query q, the objective is to pro-
duce a ranked list Rk(q)ofkdocuments that are
provided to the LLM for answer generation, fa-
voring query-relevant documents with higher trust
scores. Finally, we perform trust-aware prompting
(Section 3.4) by leveraging document trust scores
during answer generation.
TrustPropRAG separates offline trust estima-
tion from query-time answer generation. Graph
construction and trust score optimization are per-
formed offline over the corpus, amortizing their
computational cost across future queries, while
trust-aware rescoring and prompting are performed
for each query with limited additional overhead.3.1 Document Relation Graph Construction
Given the corpusC, we construct a document rela-
tion graph G= (C, E) , where each node represents
a document, and each edge eijencodes a relation
between documents dianddj. Each edge eijis
assigned a relation label ℓij. Specifically, ℓij= +1
means that the two documents are expected to re-
ceive similar trust scores, ℓij=−1 means that they
contain conflicting claims and should not both be
highly trusted, and ℓij= 0means that no reliable
relation can be identified. Each edge also has a
confidence weight wij∈[0,1] , which reflects the
confidence associated with the relation labelℓ ij.
The relation label and confidence weight can be
obtained in different ways. For example, we can
use source metadata to connect two documents di
anddjfrom the same source group with support-
ive edges ( ℓij= +1 ), because they are expected
to have correlated trustworthiness and thus similar
trust scores. Document relations can also be in-
ferred using an NLI model, which derives whether
documents dianddjare supportive ( ℓij= +1 ),
contradictory ( ℓij=−1 ), or neutral ( ℓij= 0). For
edges derived by NLI, the confidence weight wijis
set to the softmax probability associated with the
relation label.
3.2 Trust Score Optimization
Pairwise consistency loss.We encourage two doc-
uments with positive relation labels to have similar
trust scores, while enforcing that contradictory doc-
uments have diverging trust scores. Specifically,
ifdianddjare mutually supportive ( ℓij= +1 ),
their trust scores τiandτjshould be close to each
other, i.e., we aim to minimize (τi−τj)2; whereas
ifℓij=−1 , meaning that dianddjare contradic-
tory, their trust scores should be far apart so that
at most one document can be trusted, i.e., we aim
to minimize (τi+τj−1)2. Hence, we define the
pairwise consistency lossL pair:
Lpair=X
eij∈E
ℓij=+1wij(τi−τj)2+X
eij∈E
ℓij=−1wij(τi+τj−1)2.
Feedback loss.Over time, we maintain a feed-
back set F=F+∪ F−:F+for documents
marked as reliable through user feedback, and F−
for documents marked as unreliable. We incorpo-
rate this feedback into the optimization problem
of trust score estimation. Nodes in F+are encour-
aged to have high trust scores, e.g., by minimizing

Document relation graph construction
Offline stage
Positive 
feedback
Negative
feedback
Optimization -based trust score estimation
Feedback loss term
Positive 
feedbackTrust 
score 𝜏1
 Negative
feedback
Pairwise consistency loss term
Supportive
relation𝑑1Trust 
score 𝜏2𝑑2
𝑑1 𝑑2𝜏1𝜏2Pull 
closeContradictory
relation𝑑1 𝑑2𝜏1𝜏2Push 
apart
Query time
?QueryTrust -aware 
rescoringRetrieverTop-k relevant and 
reliable documents
LLMTrust -aware 
answer generationTrust scoresFigure 1: Overview of the TrustPropRAG pipeline.
(τi−τ+)2, where τ+is close to 1. In contrast,
nodes in F−are pushed toward lower trust scores,
e.g., by minimizing (τi−τ−)2, where τ−is close to
0, to limit their influence on answer generation. We
implement this by adding a feedback term Lfbto
the optimization problem of trust score estimation.
Lfbis formulated as
Lfb=X
di∈F(τi−yi)2, y i=(
τ+,ifd i∈ F+
τ−,ifd i∈ F−,
wherey iis the feedback value.
Formulating trust score optimization problem.
To capture pairwise consistency of documents and
incorporate user feedback, we jointly optimize ob-
jective function terms LpairandLfb. We formulate
the optimization problem to assign trust scores to
all documents in the corpus:
τ∗= arg min
τ∈[0,1]NLpair+λ fLfb,(1)
where τ={τ 1, τ2,···, τ N}is the set of trust
scores for all documents, and λfcontrols the trade-
off between the two terms.
Solving trust score optimization problem.Eq. (1)
is a quadratic programming (QP) problem. We
solve the problem with projected gradient descent
(PGD). The document relation graph can contain
many nodes but remains sparse (i.e., each node
has supportive or contradictory relation labels with
only a small number of other nodes). As a result,
each PGD step only needs to compute the loss and
gradients for document pairs connected by these
relations, rather than all possible document pairs.
Section 4.2 reports the measured cost of each stage
in TrustPropRAG.
Before running PGD, we initialize each docu-
ment’s trust score using its relations with otherdocuments in the document relation graph. Specifi-
cally, for each document di, we compute an initial
score by aggregating the signed weights of its con-
nected edges: init(d i) =P
(i,j)∈E ℓij·wij. Val-
ues are min-max normalized to [τmin, τmax], with
0< τ min<0.5< τ max<1. Intuitively, docu-
ments connected to others through stronger sup-
portive relations receive higher initial trust scores,
while documents involved in stronger contradictory
relations receive lower initial trust scores. Graph
construction and trust score optimization are per-
formedofflineover the corpus, amortizing their
computational cost across future queries.
Theoretical analysis.Trust scores exhibitmulti-
hop propagation. Feedback on a document shifts
its trust score; through pairwise document relations,
this effect then spreads to its neighbors, which
further propagate the influence to their neighbors.
We formalize this behavior in two lemmas below.
SinceL=L pair+λ fLfbis quadratic in τ, its
Hessian A=∇2Lis symmetric positive semidef-
inite. We assume that every connected compo-
nent of Gcontains at least one document with
feedback, which is sufficient to make Astrictly
positive definite (see proof in Appendix C). Let
σmin≤ ··· ≤σ maxdenote its eigenvalues and
κ=σ max/σminits condition number. The un-
constrained minimizer satisfies the linear system
Aτ∗=b. We use this unconstrained solution to
analyze trust propagation and convergence rate.
Lemma 1 (Trust propagation) Let f1, . . . , f m
be the feedback documents in F. Suppose the
feedback value ykat each fk∈ F is perturbed
with a change of ∆yfk. Then the resulting change

in the trust score of documentd isatisfies
|∆τi| ≤mX
k=1γk·qdG(i, fk), q=√κ−1√κ+ 1,
where γk= 2λ fγ|∆y fk|,γ= (1 +√κ)2/(2σ max) =O(1/σ min), and dG(i, j) is the
shortest-path distance between nodes iandjinG.
Proof.See Appendix C.1.
This lemma shows that trust propagation is
distance-weighted and cumulative. Each document
receives influence from all documents with feed-
back. Each influence term is weighted by qdG(i,fk),
which depends on its graph distance to the cor-
responding document with feedback. The result
suggests that sparse feedback can still be effec-
tive when the documents with feedback are well
positioned in the relation graph, allowing their in-
fluence to reach other documents through short
relation paths.
Lemma 2 (Convergence) With the fixed step size
η⋆= 2/(σ min+σ max), the projected gradient
descent iterates satisfy
∥τ(t)−τ∗∥ ≤ρ·∥τ(t−1)−τ∗∥, ρ=κ−1
κ+ 1.
Proof.See Appendix C.2.
At each iteration, the error contracts by a factor
of at most ρ, so the number of iterations needed to
reach an ϵ-accurate solution (i.e. ∥τ(t)−τ∗∥ ≤ϵ )
scales as O(κlog(1/ϵ)) . In our experiments, PGD
converges within 1000 iterations on tested datasets.
3.3 Trust-aware Rescoring
We select documents with both high trust scores
and high query relevance, and provide them to the
LLM as retrieved context for answer generation.
To this end, we perform trust-aware rescoring for
every document by using its estimated trust score
and its similarity score. The similarity score is
computed by the retriever, e.g., as a dot product
score, to measure how relevant the document is to
the query. Specifically, for document di, we com-
bine its trust score τiwith its normalized similarity
score siand define the trust-aware ranking score as
ri=ατ i+(1−α)s i, where α∈[0,1] controls the
trade-off between document trustworthiness and its
relevance to the query. We select the top- kdocu-
ments with the largest riand provide them to the
LLM for answer generation.3.4 Trust-aware Answer Generation
Apart from using trust-aware rescoring for docu-
ment retrieval, we further leverage the estimated
trust scores during answer generation. Specifi-
cally, for the top- kdocuments with the largest trust-
aware ranking scores, we format the prompt so that
each document is accompanied by its trust score.
We also include an instruction in the prompt to
guide the LLM to place greater reliance on docu-
ments with higher trust scores during answer gen-
eration. This trust-aware prompting complements
trust-aware rescoring: while rescoring filters out
less reliable documents before answer generation,
this trust-aware prompting helps the LLM account
for document reliability during answer generation.
4 Evaluation
4.1 Evaluation Setup
Datasets.We evaluate TrustPropRAG
with three open-domain QA datasets:
MS MARCO (Bajaj et al., 2016), Natural
Questions (NQ) (Kwiatkowski et al., 2019), and
TriviaQA (Joshi et al., 2017). These datasets
are widely adopted in retrieval-augmented and
open-domain QA evaluation, and together they
cover diverse question types. Following the
evaluation settings commonly used in prior RAG
reliability and robustness studies (Shen et al., 2025;
Zhong et al., 2023; Wei et al., 2025; Wang et al.,
2025a), we randomly sample 300 query–answer
instances for each dataset. Half of the sampled
queries are paired with synthetic contradictory
documents generated by following the method in
PoisonedRAG (Zou et al., 2025). These documents
are written to remain fluent and relevant to the
query while supporting answers that conflict with
the ground-truth answers. The remaining queries
are paired only with factual documents from the
original datasets. In this way, we construct the
evaluation corpus from the relevant documents
associated with the sampled queries, resulting in
805, 893, and 1,037 documents for MS MARCO,
NQ, and TriviaQA, respectively. This setup
evaluates whether the end-to-end system can
produce correct answers when the corpus contains
both reliable evidence and fluent but misleading
contradictory content.
Retrievers.We evaluate with both sparse
and dense retrievers. We use BM25 (Robert-
son and Zaragoza, 2009), a sparse lexical re-
triever, and two dense bi-encoder retrievers: Con-

triever (Izacard et al., 2022) and MiniLM based
onall-MiniLM-L6-v2 (Reimers and Gurevych,
2019). For each query, we retrieve k=5 documents.
Graph construction.We construct a document
relation graph using two types of information: doc-
ument sources and semantic relations between doc-
uments. Documents iandjfrom the same source
are connected with supportive edges ( ℓij= +1 ). In
TriviaQA, we determine whether two documents
share the same source based on their source Uni-
form Resource Locators (URLs). For MS MARCO
and NQ, which do not provide source metadata,
we use simulated source grouping: factual docu-
ments are partitioned into multiple source groups,
and contradictory documents are partitioned into
another set of source groups. Documents in the
same group are treated as sharing the same source.
We adopt this simulation as a proxy for real source
metadata, following recent source-reliability-aware
RAG work that simulates sources with varying re-
liability (Hwang et al., 2025). We also apply an
NLI model ( nli-deberta-v3-small ) to each doc-
ument pair. If the model predicts that two docu-
ments iandjsupport each other, we add a sup-
portive edge ( ℓij= +1 ). If it predicts that they
contradict each other, we add a contradictory edge
(ℓij=−1 ). Appendix F reports the precision
and recall of the NLI-derived edges on all three
datasets. Table 4 evaluates the impact of removing
the source-group edges to isolate the contribution
of the NLI-derived edges.
Feedback construction.We construct the feed-
back set by randomly sampling a fraction of factual
and contradictory documents and simulating feed-
back labels for them. Sampled factual documents
are assigned positive feedback labels, while sam-
pled contradictory documents are assigned negative
feedback labels. We refer to this sampling fraction
as the feedback ratio (FB); for instance, FB = 30%
means that feedback labels are provided for 30%
of the documents. As real-world feedback may
be imperfect, we evaluate TrustPropRAG under
noisy feedback in Figure 2, where a portion of the
feedback labels are incorrect.
Trust optimization.For PGD, we use a step size of
η= 0.01 , run for at most T= 1000 iterations, and
stop when the convergence tolerance reaches ϵ=
10−6. We set feedback loss weight λf=1.0 . Trust
scores are optimized over all corpus documents,
resulting in large-scale optimization problems over
805, 893, and 1,037 documents for MS MARCO,
NQ, and TriviaQA, respectively.Rescoring.We set the trust coefficient to α=0.6 .
We also test the impact of αin Section 4.2. The
top-kdocuments, where k= 5 in our experiments,
are provided as retrieved context to the LLM. We
also test the impact ofkin Appendix B.
Answer generation.We use proprietary and
open-source LLMs, including GPT-4o-mini, GPT-
5, Gemini 2.5 Flash, Gemini 2.5 Pro, Llama-3.1-
8B-Instruct, and Mistral-7B-Instruct. TrustPro-
pRAG leverages the trust-aware prompt that anno-
tates each document with its trust score (as detailed
in Section 3.4).
Evaluation metrics.We adopt exact match (EM)
and fact precision@ k(FP@ k) as evaluation met-
rics. EM is a widely used metric in open-domain
QA. EM checks whether the normalized generated
answer contains the ground-truth answer, with nor-
malization including lowercasing and the removal
of articles and punctuation. FP@ kevaluates re-
trieval quality, which measures the reliability of
the top-ranked documents used for answer gener-
ation. Specifically, let DfactandDcontr denote the
sets of factual and contradictory documents, respec-
tively, and let Rk(q)denote the top- kdocuments
for query q. FP@ kis defined as the fraction of
factual documents among these top- kdocuments:
FP@k=|Rk(q)∩D fact|
|Rk(q)∩(D fact∪Dcontr)|.
Baselines.We compare TrustPropRAG against the
following methods: (1)VanillaRAGranks docu-
ments using the relevance scores produced by the
retriever and feeds the top-ranked documents to the
LLM. (2)InstructRAG(Wei et al., 2025) guides the
LLM to better use retrieved evidence during gen-
eration. It decomposes retrieved documents into
atomic claims, and prompts the LLM to generate
a rationale connecting relevant claims to the final
answer. (3)AstuteRAG(Wang et al., 2025a) com-
bines the LLM’s internal knowledge with retrieved
evidence. It first prompts the LLM to produce an
answer-relevant document from its internal knowl-
edge, and then compares this document with the
retrieved documents and resolves conflicts before
generating the answer. (4)ReliabilityRAG(Shen
et al., 2025) uses NLI to identify contradictions
among retrieved documents. It then solves a max-
imum independent set problem to select a subset
of documents with no detected pairwise contradic-
tions and uses keyword aggregation to generate the
final answer. (5)TrustRAG(Zhou et al., 2025) in-
cludes a first-stage filtering procedure that clusters
the retrieved documents by embedding similarity

Dataset Method Contriever BM25 MiniLM
MS MARCOVanillaRAG 0.27 0.24 0.26
InstructRAG 0.35 0.40 0.37
AstuteRAG 0.38 0.39 0.37
ReliabilityRAG 0.42 0.40 0.39
TrustRAG stage 1 0.36 0.35 0.33
TrustPropRAG0.45 0.47 0.44
NQVanillaRAG 0.37 0.37 0.39
InstructRAG 0.46 0.48 0.43
AstuteRAG 0.44 0.50 0.42
ReliabilityRAG 0.45 0.50 0.48
TrustRAG stage 1 0.44 0.46 0.41
TrustPropRAG0.55 0.59 0.59
TriviaQAVanillaRAG 0.57 0.62 0.49
InstructRAG 0.68 0.67 0.69
AstuteRAG 0.70 0.71 0.72
ReliabilityRAG 0.75 0.72 0.67
TrustRAG stage 1 0.66 0.65 0.63
TrustPropRAG0.78 0.75 0.77
Table 1: EM averaged over GPT-4o-mini, Gemini 2.5
Flash, and GPT-5 ( k=5, FB=30%). Best results are
bolded; second-best results are underlined .
Dataset Retriever Baselines TrustPropRAG
MS MARCOContriever 44%77%
BM25 57%92%
MiniLM 54%88%
NQContriever 53%79%
BM25 46%88%
MiniLM 48%96%
TriviaQAContriever 38%77%
BM25 57%90%
MiniLM 49%92%
Table 2: Retrieval quality measured by FP@5 ( k=5,
FB=30%).
and filters out clusters identified as suspicious. We
use this first-stage procedure as the baseline.
4.2 Evaluation Results
TrustPropRAG outperforms baselines on EM
and FP@5.Tables 1 and 2 report the EM
and FP@5 averaged over GPT-4o-mini, Gem-
ini 2.5 Flash, and GPT-5. Table 1 shows that Trust-
PropRAG achieves the best EM across all datasets
and retrievers, with absolute EM gains of 0.03–0.11
compared with the strongest baseline. Table 2
shows FP@5 (the fraction of factual documents
among top- kdocuments) of TrustPropRAG and
baselines. All baselines have the same FP@5 as
VanillaRAG under this retrieval-level metric: In-
structRAG, AstuteRAG, and TrustRAG stage 1 oper-
ate after the retriever selects the top- kdocumentsbased on query relevance and thus do not change
the ranked top- kdocuments; although Reliabili-
tyRAG filters documents before final answer gen-
eration, its filtering is driven by intermediate LLM-
generated answers and their consistency, rather
than by producing a new top- klist. The baseline
methods have an FP@5 of 38–57%, while Trust-
PropRAG raises it to 77–96%. The improvement
is observed with all three retrievers, indicating that
the method is not tied to a specific retriever. These
improvements in FP@5 help explain the EM gains
of TrustPropRAG. Specifically, a higher FP@5 in-
dicates that the top- klist contains a larger fraction
of factual documents, which suggests that trust
score optimization assigns higher trust to more re-
liable documents, and trust-aware rescoring pro-
motes them into the top- klist. As a result, the
LLM inputs contain more documents that support
the correct answer.
Ablation study.We conduct an ablation study
to examine the contribution of each component
in TrustPropRAG. Table 3 evaluates five vari-
ants on MS MARCO with FB=30% using Gem-
ini 2.5 Flash. We observe that the combination
of trust score optimization and trust-aware rescor-
ing leads to a substantial improvement. Compared
with VanillaRAG, adding these two components in-
creases FP@5 from 54% to 87% and improves EM
from 0.28 to 0.41. This suggests that the optimized
trust scores are effective for identifying more trust-
worthy documents and that rescoring can promote
them into the top- kcontext used for answer gener-
ation. We also observe that trust-aware prompting
provides an additional benefit when combined with
trust score optimization and trust-aware rescoring,
increasing EM from 0.41 to 0.46. This suggests that
some low-trust documents may still remain in the
top-klist after rescoring, and trust-aware prompt-
ing helps the LLM reduce their influence during
answer generation. Finally, we observe that using
trust-aware prompting alone does not improve per-
formance. Without trust score optimization, FP@5
remains at 54%, and EM slightly decreases from
0.28 to 0.27 compared with VanillaRAG. This sug-
gests the trust-aware prompt becomes useful only
when the trust scores have been optimized and can
provide a meaningful signal for answer generation.
The effects of trust propagation and source-
group edges.To separate the benefit of trust prop-
agation from simply having access to feedback,
thetrivial feedbackvariant uses the same feed-
back set as TrustPropRAG but does not propagate

Variant Opt. Rescore Prompt EM FP@5
VanillaRAG – – – 0.28 54%
Opt.+Prompt✓–✓ 0.35 54%
Opt.+Rescore✓ ✓– 0.41 87%
Prompt only – –✓ 0.27 54%
Full TrustPropRAG✓ ✓ ✓ 0.46 88%
Table 3: Effects of different TrustPropRAG compo-
nents.Opt: trust score optimization,Rescore: trust-
aware rescoring, andPrompt: trust-aware prompting.
Variant EM FP@5
VanillaRAG 0.31 51%
TrustPropRAG (trivial feedback) 0.36 62%
TrustPropRAG (NLI-only) 0.44 79%
TrustPropRAG 0.49 92%
Table 4: Effects of trust propagation and source-group
edges (MS MARCO and NQ, MiniLM, GPT-4o-mini,
k=5, FB=30%).
the feedback through the graph. Documents with
positive feedback receive a boost to their retrieval
scores, while documents with negative feedback
are not included in the top- klist. To isolate the
contribution of the NLI-derived edges from source-
group edges, theNLI-onlyvariant removes all simu-
lated source-group edges and retains only the NLI-
derived edges.
From Table 4, the trivial feedback variant
achieves an EM of 0.36 and an FP@5 of 62%. This
confirms that directly using feedback is beneficial.
However, its performance remains substantially be-
low TrustPropRAG, which achieves an EM of 0.49
and an FP@5 of 92%, respectively. This gap shows
that the gain of TrustPropRAG cannot be attributed
only to access to the feedback set. Propagating
feedback through document relation graphs pro-
vides trust estimates for documents without direct
feedback. Table 4 also shows that the NLI-only
variant achieves an EM of 0.44 and an FP@5 of
79%. Although removing the source-group edges
leads to a performance drop, the NLI-only variant
still outperforms the trivial feedback variant, show-
ing that propagation remains beneficial even when
the graph contains only NLI-derived relations.
Performance under noisy feedback.Figure 2
evaluates the robustness of TrustPropRAG by vary-
ing the accuracy of feedback labels from 60% to
100%. Feedback accuracy denotes the fraction of
feedback labels that are correct. For example, 80%
feedback accuracy means that 20% of the feed-
back labels are flipped: a factual document may
be labeled as low-trust, or a contradictory docu-
/uni00000013 /uni00000015/uni00000013 /uni00000017/uni00000013 /uni00000019/uni00000013 /uni0000001b/uni00000013 /uni00000014/uni00000013/uni00000013
/uni00000029/uni00000048/uni00000048/uni00000047/uni00000045/uni00000044/uni00000046/uni0000004e/uni00000003/uni00000055/uni00000044/uni00000057/uni0000004c/uni00000052/uni00000003/uni0000000b/uni00000008/uni0000000c/uni00000013/uni00000011/uni00000016/uni00000013/uni00000013/uni00000011/uni00000016/uni00000016/uni00000013/uni00000011/uni00000016/uni00000019/uni00000013/uni00000011/uni00000016/uni0000001c/uni00000013/uni00000011/uni00000017/uni00000015/uni00000013/uni00000011/uni00000017/uni00000018/uni00000013/uni00000011/uni00000017/uni0000001b/uni00000028/uni00000030
/uni00000014/uni00000013/uni00000013/uni00000008/uni00000003/uni00000049/uni00000048/uni00000048/uni00000047/uni00000045/uni00000044/uni00000046/uni0000004e/uni00000003/uni00000044/uni00000046/uni00000046/uni00000058/uni00000055/uni00000044/uni00000046/uni0000005c
/uni0000001b/uni00000013/uni00000008/uni00000003/uni00000049/uni00000048/uni00000048/uni00000047/uni00000045/uni00000044/uni00000046/uni0000004e/uni00000003/uni00000044/uni00000046/uni00000046/uni00000058/uni00000055/uni00000044/uni00000046/uni0000005c
/uni00000019/uni00000013/uni00000008/uni00000003/uni00000049/uni00000048/uni00000048/uni00000047/uni00000045/uni00000044/uni00000046/uni0000004e/uni00000003/uni00000044/uni00000046/uni00000046/uni00000058/uni00000055/uni00000044/uni00000046/uni0000005cFigure 2: Impact of feedback ratio and feedback accu-
racy on EM (MS MARCO, MiniLM, Gemini 2.5 Flash).
/uni00000013/uni00000011/uni00000013 /uni00000013/uni00000011/uni00000015 /uni00000013/uni00000011/uni00000017 /uni00000013/uni00000011/uni00000019 /uni00000013/uni00000011/uni0000001b /uni00000014/uni00000011/uni00000013
/uni00000037/uni00000055/uni00000058/uni00000056/uni00000057/uni00000003/uni00000046/uni00000052/uni00000048/uni00000049/uni00000049/uni0000004c/uni00000046/uni0000004c/uni00000048/uni00000051/uni00000057/uni00000003
/uni00000013/uni00000011/uni00000014/uni00000013/uni00000013/uni00000011/uni00000015/uni00000013/uni00000013/uni00000011/uni00000016/uni00000013/uni00000013/uni00000011/uni00000017/uni00000013/uni00000013/uni00000011/uni00000018/uni00000013/uni00000028/uni00000030
/uni00000028/uni00000030
/uni00000029/uni00000033/uni00000023/uni00000018
/uni00000017/uni00000013/uni00000008/uni00000018/uni00000018/uni00000008/uni0000001a/uni00000013/uni00000008/uni0000001b/uni00000018/uni00000008/uni00000014/uni00000013/uni00000013/uni00000008
/uni00000029/uni00000033/uni00000023/uni00000018/uni00000003/uni0000000b/uni00000008/uni0000000c
Figure 3: Impact of trust coefficient αon EM and FP@5.
(MS MARCO, MiniLM, Gemini 2.5 Flash, FB=30%)
ment may be labeled as high-trust. Figure 2 re-
ports EM on MS MARCO with Gemini 2.5 Flash.
We observe that TrustPropRAG is robust to noisy
feedback. Even when feedback accuracy is only
60% (i.e., 40% of feedback labels are incorrect),
TrustPropRAG still improves EM over the setting
without feedback across all feedback ratios. This
is because individual feedback errors can be miti-
gated through trust propagation over the document
relation graph, where relations among documents
help mitigate the impact of noisy feedback signals.
The impact of feedback ratio.Figure 2 also
shows EM as a function of the feedback ratio on
MS MARCO with Gemini 2.5 Flash. Compared
with the setting without feedback, TrustPropRAG
improves sharply when only 10% of documents
receive feedback, after which the improvement be-
comes more gradual. This indicates that a small
portion of feedback is sufficient for trust propaga-
tion to produce reliable trust estimates across the
document relation graph.
The impact of trust coefficient α.Figure 3 shows
EM and FP@5 as the trust coefficient αvaries
from 0 to 1. When α= 0 , rescoring uses only the
retrieval relevance score. As αincreases from 0,
both EM and FP@5 improve, showing that incor-
porating trust scores helps promote documents that

ModelMS MARCO NQ TriviaQA
∆VanillaRAG TrustPropRAG VanillaRAG TrustPropRAG VanillaRAG TrustPropRAG
Proprietary
GPT-4o-mini 0.21 0.40 0.41 0.58 0.50 0.76 +0.21
GPT-5 0.30 0.49 0.43 0.62 0.53 0.85 +0.23
Gemini 2.5 Flash 0.28 0.46 0.33 0.56 0.44 0.69 +0.22
Gemini 2.5 Pro 0.32 0.48 0.37 0.60 0.46 0.77 +0.23
Open-source
Llama-3.1-8B-Instruct 0.19 0.32 0.38 0.55 0.48 0.77 +0.20
Mistral-7B-Instruct 0.22 0.35 0.33 0.47 0.51 0.74 +0.17
Table 5: Performance with different LLMs across three datasets. ∆denotes TrustPropRAG’s absolute EM
improvement over VanillaRAG, averaged across the three datasets.
support the correct answer into the top- kcontext.
EM reaches its best value at α= 0.6 , and remains
stable between 0.5and0.8. However, when αis
too large, the top- klist tends to include documents
with high trust scores but lower query relevance,
leading to a slight decrease in EM. As TrustPro-
pRAG is not sensitive to the exact choice of α
within a range (e.g., between 0.5 and 0.8), we set
α=0.6for the rest of the experiments.
Performance with different LLMs.We evaluate
the performance of TrustPropRAG with a diverse
set of proprietary and open-source LLMs of differ-
ent model scales in Table 5. Since FP@5 depends
only on retrieval and rescoring, it is unchanged
across LLMs; therefore, we report EM as the main
metric. We use MiniLM as the retriever and set
k= 5 and FB=30%. Table 5 shows that TrustPro-
pRAG consistently outperforms VanillaRAG across
all evaluated LLMs. The average EM improvement
ranges from 0.17 to 0.23 across both proprietary
models (GPT-4o-mini, GPT-5, Gemini 2.5 Flash,
Gemini 2.5 Pro) and smaller open-source models
(Llama-3.1-8B-Instruct, Mistral-7B-Instruct). No-
tably, TrustPropRAG still achieves an average EM
gain of 0.23 with strong LLMs such as GPT-5 and
Gemini 2.5 Pro. This suggests that improving the
reliability of the retrieved context with trust propa-
gation remains useful even for strong LLMs.
Computational cost.Table 6 reports the com-
putational cost of TrustPropRAG, including of-
fline graph construction and trust optimization, as
well as query-time retrieval. The computational
resources are detailed in Appendix A. Constructing
the graph accounts for most of the offline cost, tak-
ing 578.3–1077.1 seconds. Given the constructed
graph, trust score optimization is efficient, requir-
ing only 0.3–0.4 seconds, indicating that trust prop-
agation itself introduces negligible computationalStage MS MARCO NQ TriviaQA
Offline
Graph construction 578.3 s 852.4 s 1077.1 s
Trust optimization (PGD) 0.3 s 0.3 s 0.4 s
Query time
Retrieval 4.4 s 4.8 s 4.8 s
Table 6: Computational cost breakdown.
overhead. Both graph construction and trust opti-
mization are performed offline and their costs are
therefore amortized across future queries. At query
time, retrieval takes 4.4–4.8 seconds and is shared
by TrustPropRAG and the baselines. Beyond re-
trieval, TrustPropRAG only performs lightweight
trust-aware rescoring using the precomputed trust
scores, introducing limited additional query-time
overhead.
5 Conclusions
We present TrustPropRAG, a framework that im-
proves RAG reliability by propagating sparse hu-
man feedback over a document relation graph.
TrustPropRAG estimates a trust score for each doc-
ument by optimizing an objective that combines
pairwise consistency loss and feedback loss. The re-
sulting trust scores are used to select more reliable
documents and guide trust-aware answer genera-
tion. TrustPropRAG achieves absolute EM gains of
0.03–0.11 over the strongest baseline, and demon-
strates robustness to noisy feedback.
Acknowledgments
We thank the anonymous reviewers for insightful
feedback. This work was supported by Seed Grant
of IST, and the National Science Foundation under
Grants 2550742, 2623125, 2555329, and 2549266.

Limitations
TrustPropRAG incorporates human feedback as bi-
nary labels, marking documents as either reliable
or unreliable. In practice, feedback can be graded
rather than binary. Studying TrustPropRAG under
graded feedback is left to future work. Follow-
ing common practice in RAG evaluation, we focus
on unreliable content that is injected as synthetic
contradictions rather than drawn from naturally
occurring conflicts. Finally, we focus on open-
domain QA with short factual answers. Applying
trust propagation to long-form or multi-hop genera-
tion, where reliability interacts with compositional
reasoning, is left to future work.
Ethical Considerations
This work addresses the problem of unreliable con-
tent in retrieval-augmented generation. Our exper-
iments use synthetically generated contradictions.
All datasets and models used in this work are pub-
licly released and licensed for research use, and
our usage is consistent with their intended research
purposes. MS MARCO, NQ, and TriviaQA are
publicly available open-domain QA benchmarks
released for research. The retrievers (BM25, Con-
triever, and the all-MiniLM-L6-v2 encoder) and the
NLI model (nli-deberta-v3-small) are also publicly
released.
References
Yu Bai, Yukai Miao, Li Chen, Dawei Wang, Dan Li,
Yanyu Ren, Hongtao Xie, Ce Yang, and Xuhui Cai.
2024. Pistis-RAG: Enhancing retrieval-augmented
generation with human feedback.arXiv preprint
arXiv:2407.00072.
Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng,
Jianfeng Gao, Xiaodong Liu, Rangan Majumder, An-
drew McNamara, Bhaskar Mitra, Tri Nguyen, Mir
Rosenberg, Xia Song, Alina Stoica, Saurabh Tiwary,
and Tong Wang. 2016. MS MARCO: A human gen-
erated machine reading comprehension dataset. In
Proceedings of the Workshop on Cognitive Computa-
tion: Integrating Neural and Symbolic Approaches
(CoCo@NIPS).
Michele Benzi and Nader Razouk. 2007. Decay bounds
and O( n) algorithms for approximating functions of
sparse matrices.Electronic Transactions on Numeri-
cal Analysis, 28:16–39.
Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann,
Trevor Cai, Eliza Rutherford, Katie Millican, George
van den Driessche, Jean-Baptiste Lespiau, Bogdan
Damoc, Aidan Clark, Diego de Las Casas, AureliaGuy, Jacob Menick, Roman Ring, Tom Hennigan,
Saffron Huang, Loren Maggiore, Chris Jones, Albin
Cassirer, and 9 others. 2022. Improving language
models by retrieving from trillions of tokens. InPro-
ceedings of the 39th International Conference on Ma-
chine Learning (ICML), volume 162 ofProceedings
of Machine Learning Research, pages 2206–2240.
PMLR.
Hung-Ting Chen, Michael J.Q. Zhang, and Eunsol Choi.
2022. Rich knowledge sources bring complex knowl-
edge conflicts: Recalibrating models to reflect con-
flicting evidence. InProceedings of the 2022 Con-
ference on Empirical Methods in Natural Language
Processing, pages 2292–2307, Abu Dhabi, United
Arab Emirates. Association for Computational Lin-
guistics.
Janara Christensen, Mausam, Stephen Soderland, and
Oren Etzioni. 2013. Towards coherent multi-
document summarization. InProceedings of the
2013 Conference of the North American Chapter of
the Association for Computational Linguistics: Hu-
man Language Technologies, pages 1163–1173, At-
lanta, Georgia. Association for Computational Lin-
guistics.
Benjamin Cohen-Wang, Harshay Shah, Kristian
Georgiev, and Aleksander M ˛ adry. 2024. ContextCite:
Attributing model generation to context. InAdvances
in Neural Information Processing Systems (NeurIPS),
volume 37.
Stephen Demko, William F. Moss, and Philip W. Smith.
1984. Decay rates for inverses of band matrices.
Mathematics of Computation, 43(168):491–499.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and
Jonathan Larson. 2024. From local to global: A
graph RAG approach to query-focused summariza-
tion.arXiv preprint arXiv:2404.16130.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao
Huang. 2025. LightRAG: Simple and fast retrieval-
augmented generation. InFindings of the Association
for Computational Linguistics: EMNLP 2025, pages
10746–10761.
Jeongyeon Hwang, Junyoung Park, Hyejin Park, Dong-
woo Kim, Sangdon Park, and Jungseul Ok. 2025.
Retrieval-augmented generation with estimation of
source reliability. InProceedings of the 2025 Con-
ference on Empirical Methods in Natural Language
Processing, pages 34279–34303.
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebas-
tian Riedel, Piotr Bojanowski, Armand Joulin, and
Edouard Grave. 2022. Unsupervised dense informa-
tion retrieval with contrastive learning.Transactions
on Machine Learning Research.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas
Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-
Yu, Armand Joulin, Sebastian Riedel, and Edouard

Grave. 2023. Few-shot learning with retrieval aug-
mented language models.Journal of Machine Learn-
ing Research, 24(251):1–43.
Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke
Zettlemoyer. 2017. TriviaQA: A large scale distantly
supervised challenge dataset for reading comprehen-
sion. InProceedings of the 55th Annual Meeting of
the Association for Computational Linguistics.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, Kristina Toutanova, Llion Jones, Matthew
Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob
Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natu-
ral questions: A benchmark for question answering
research.Transactions of the Association for Compu-
tational Linguistics, 7:452–466.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive NLP tasks. InAdvances in Neural Informa-
tion Processing Systems, volume 33.
Siyi Liu, Qiang Ning, Kishaloy Halder, Zheng Qi, Wei
Xiao, Phu Mon Htut, Yi Zhang, Neha Anna John,
Bonan Min, Yassine Benajiba, and 1 others. 2025a.
Open domain question answering with conflicting
contexts. InFindings of the Association for Computa-
tional Linguistics: NAACL 2025, pages 1838–1854.
Yuhan Liu, Michael JQ Zhang, and Eunsol Choi. 2025b.
User feedback in human-LLM dialogues: a lens to un-
derstand users but noisy as a learning signal. InPro-
ceedings of the 2025 Conference on Empirical Meth-
ods in Natural Language Processing, pages 2666–
2681.
Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao,
and Nan Duan. 2023. Query rewriting in retrieval-
augmented large language models. InProceedings
of the 2023 Conference on Empirical Methods in
Natural Language Processing, pages 5303–5315.
Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Car-
roll L. Wainwright, Pamela Mishkin, Chong Zhang,
Sandhini Agarwal, Katarina Slama, Alex Ray, John
Schulman, Jacob Hilton, Fraser Kelton, Luke Miller,
Maddie Simens, Amanda Askell, Peter Welinder,
Paul Christiano, Jan Leike, and Ryan Lowe. 2022.
Training language models to follow instructions with
human feedback. InAdvances in Neural Information
Processing Systems, volume 35.
Nils Reimers and Iryna Gurevych. 2019. Sentence-
BERT: Sentence embeddings using siamese BERT-
networks. InProceedings of the 2019 Conference on
Empirical Methods in Natural Language Processing,
pages 3982–3992. Association for Computational
Linguistics.Stephen Robertson and Hugo Zaragoza. 2009. The
probabilistic relevance framework: BM25 and be-
yond.Foundations and Trends in Information Re-
trieval, 3(4):333–389.
Zeyu Shen, Basileal Imana, Tong Wu, Chong Xiang,
Prateek Mittal, and Aleksandra Korolova. 2025. Re-
liabilityRAG: Effective and provably robust defense
for RAG-based web-search. InAdvances in Neural
Information Processing Systems.
Fei Wang, Xingchen Wan, Ruoxi Sun, Jiefeng Chen,
and Sercan Ö. Arik. 2025a. Astute RAG: Overcom-
ing imperfect retrieval augmentation and knowledge
conflicts for large language models. InProceedings
of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 30553–30571, Vienna, Austria. Association
for Computational Linguistics.
Yanting Wang, Wei Zou, Runpeng Geng, and Jinyuan
Jia. 2025b. TracLLM: A generic framework for at-
tributing long context llms. InUSENIX Security
Symposium.
Zhepei Wei, Wei-Lin Chen, and Yu Meng. 2025. In-
structRAG: Instructing retrieval-augmented genera-
tion via self-synthesized rationales. InProceedings
of the Thirteenth International Conference on Learn-
ing Representations (ICLR).
Chong Xiang, Tong Wu, Zexuan Zhong, David Wagner,
Danqi Chen, and Prateek Mittal. 2024. Certifiably ro-
bust RAG against retrieval corruption.arXiv preprint
arXiv:2405.15556.
Jian Xie, Kai Zhang, Jiangjie Chen, Renze Lou, and
Yu Su. 2024. Adaptive chameleon or stubborn sloth:
Revealing the behavior of large language models in
knowledge conflicts. InProceedings of the Twelfth
International Conference on Learning Representa-
tions (ICLR).
Rongwu Xu, Zehan Qi, Zhijiang Guo, Cunxiang Wang,
Hongru Wang, Yue Zhang, and Wei Xu. 2024.
Knowledge conflicts for LLMs: A survey. InPro-
ceedings of the 2024 Conference on Empirical Meth-
ods in Natural Language Processing, pages 8541–
8565, Miami, Florida, USA. Association for Compu-
tational Linguistics.
Tianyang Xu, Haojie Zheng, Chengze Li, Haoxiang
Chen, Yixin Liu, Ruoxi Chen, and Lichao Sun. 2025.
NodeRAG: Structuring graph-based rag with hetero-
geneous nodes.arXiv preprint arXiv:2504.11544.
Hanning Zhang, Juntong Song, Juno Zhu, Yuanhao Wu,
Tong Zhang, and Cheng Niu. 2025a. RAG-Reward:
Optimizing RAG with reward modeling and RLHF.
arXiv preprint arXiv:2501.13264.
Jizhi Zhang, Chongming Gao, Wentao Shi, Xin Chen,
Jingang Wang, Xunliang Cai, and Fuli Feng. 2025b.
Leveraging unpaired feedback for long-term LLM-
based recommendation tuning. InFindings of the
Association for Computational Linguistics: EMNLP
2025, pages 24507–24521.

Wanjun Zhong, Jingjing Xu, Duyu Tang, Zenan Xu,
Nan Duan, Ming Zhou, Jiahai Wang, and Jian Yin.
2020. Reasoning over semantic-level graph for fact
checking. InProceedings of the 58th Annual Meet-
ing of the Association for Computational Linguistics,
pages 6170–6180. Association for Computational
Linguistics.
Zexuan Zhong, Ziqing Huang, Alexander Wettig, and
Danqi Chen. 2023. Poisoning retrieval corpora by
injecting adversarial passages. InProceedings of the
2023 Conference on Empirical Methods in Natural
Language Processing.
Dengyong Zhou, Olivier Bousquet, Thomas N. Lal, Ja-
son Weston, and Bernhard Schölkopf. 2003. Learn-
ing with local and global consistency. InAdvances in
Neural Information Processing Systems, volume 16,
pages 321–328.
Huichi Zhou, Kin-Hei Lee, Zhonghao Zhan, Yue Chen,
Zhenhao Li, Zhaoyang Wang, Hamed Haddadi, and
Emine Yilmaz. 2025. TrustRAG: Enhancing robust-
ness and trustworthiness in retrieval-augmented gen-
eration.arXiv preprint arXiv:2501.00879.
Xiaojin Zhu, Zoubin Ghahramani, and John D. Laf-
ferty. 2003. Semi-supervised learning using Gaus-
sian fields and harmonic functions. InProceedings
of the 20th International Conference on Machine
Learning, pages 912–919.
Wei Zou, Runpeng Geng, Binghui Wang, and Jinyuan
Jia. 2025. PoisonedRAG: Knowledge corruption at-
tacks to retrieval-augmented generation of large lan-
guage models. In34th USENIX Security Symposium
(USENIX Security 25), pages 3827–3844. USENIX
Association.
A Implementation Details
Models and computational resources. Among
the LLMs we evaluate, Llama-3.1-8B-Instruct and
Mistral-7B-Instruct are open-source models with
8B and 7B parameters, respectively; GPT-4o-mini,
GPT-5, Gemini 2.5 Flash, and Gemini 2.5 Pro
are proprietary models accessed through their re-
spective APIs. The NLI model (nli-deberta-v3-
small) has approximately 44M parameters, and
the dense retrievers (Contriever and all-MiniLM-
L6-v2) are sentence-encoder models with approx-
imately 110M and 22M parameters respectively.
All local computation, including NLI-based edge
construction, dense retrieval, and trust score op-
timization, was run on an NVIDIA RTX 5070
GPU. For our largest corpus (1,037 documents,
TriviaQA), this took approximately 0.3 GPU-hours.
Trust score optimization via projected gradient de-
scent is lightweight, converging within 1,000 iter-
ations and taking under 1 minute per corpus. Wek=1k=3k=5k=10k=20
EM 0.41 0.43 0.46 0.45 0.42
FP@k95% 92% 88% 84% 73%
Table 7: Impact of the number of retrieved documents k
on EM and FP@ k(MS MARCO, MiniLM, Gemini 2.5
Flash, FB=30%).
implement retrieval, NLI inference, and trust score
optimization using standard Python libraries, and
provide exact package versions and scripts in the
released code.
Hyperparameters.We use the same hyperpa-
rameters across all datasets, retrievers, and LLMs.
For trust score initialization, we set τmin= 0.3
andτmax= 0.7 . We set feedback loss weight
λf=1.0 . We set the feedback values τ+= 0.9 and
τ−= 0.1.
B Number of Retrieved Documentsk
The number of retrieved documentskcontrols the
size of the context provided to the LLM. A smaller
kyields a more selective context but may omit rel-
evant evidence, while a larger kprovides broader
coverage at the cost of allowing more unreliable
documents into the prompt. We evaluate TrustPro-
pRAG across k∈ {1,3,5,10,20} on MS MARCO
with MiniLM as the retriever, Gemini 2.5 Flash as
the backbone LLM, and FB =30%. All other hy-
perparameters follow Section 4.
FP@kfalls steadily from 95% ( k=1) to 73%
(k=20) as lower-trust documents enter the top- k
selected documents. EM is less sensitive, staying
within 0.41–0.46 and peaking at k=5 and k=10.
Trust-aware rescoring places reliable evidence at
the top, so small kalready provides sufficient fac-
tual support; at larger k, the trust-aware prompt
helps the LLM put less weight to lower-trust doc-
uments that are provided in the context. EM is
smaller at k=1, where the retrieved context may
provide insufficient context, and at k=20, where
the larger retrieval set exposes the LLM to more
contradictory context.
Considering that EM peaks at k=5 and k=10, we
adopt k=5, matching prior work (Zou et al., 2025;
Shen et al., 2025).
C Proofs
We provide proofs of the two lemmas stated in
Section 3.2. Both lemmas rely on the fact that the
trust optimization objective L=L pair+λfLfb

is a quadratic in τ, so the first-order optimality
condition for the unconstrained quadratic problem
is given by the linear system Aτ∗=b, where
A=∇2Lis the Hessian. We first verify that
Ais positive definite under the assumption that
every connected component of Gcontains at least
one document with feedback, then prove the two
lemmas.
Positive definiteness of A.Let ei∈RNdenote
thei-th standard basis vector, whose i-th entry is
one and all other entries are zero. Expanding the
Hessian gives
A= 2X
eij∈E
ℓij=+1wij(ei−ej)(ei−ej)⊤
+ 2X
eij∈E
ℓij=−1wij(ei+ej)(ei+ej)⊤
+ 2λ fX
i∈Feie⊤
i,(2)
which is a sum of positive semidefinite matrices,
hence A⪰0 . Next, we prove A≻0 . A vector v
lies in ker(A) iff (a) vi=vjon every supportive
edge ( ℓij= +1 ), (b) vi=−v jon every contra-
dictory edge ( ℓij=−1 ), and (c) vi= 0on every
node with feedback. If a node with value zero is
connected to another node by a supportive edge,
condition (a) implies that the neighboring node also
has value zero. If it is connected by a contradictory
edge, condition (b) again implies that the neighbor-
ing node has value zero. Under our assumption that
every connected component contains at least one
feedback node, we have v=0 globally, so A≻0 .
We write its eigenvalues as σmin≤ ··· ≤σ max
and its condition number asκ=σ max/σmin.
C.1 Proof of Lemma 1 (Trust Propagation)
We fix the set of documents with feedback F=
{f1, . . . , f m}and consider perturbations of their
feedback values. Let ∆yfkdenote the perturbation
applied to the feedback value yfk. These pertur-
bations leave Aunchanged and only modify the
right-hand side bat entries corresponding to docu-
ments in F. Let∆bdenote the resulting change in
b, and let ∆bfkdenote its entry corresponding to
feedback documentf k. We have
∆bfk= 2λ f∆yfk.(3)
Step 1: Linearity and superposition.Because
the perturbations only change the feedback targetvalues, the matrix Aremains fixed. Therefore, the
unconstrained optimum satisfies τ∗=A−1b.Af-
ter the perturbation, the right-hand side becomes
b+ ∆b , and the corresponding change in the opti-
mized trust scores is
∆τ∗=A−1∆b.
Since ∆bis nonzero only at documents with feed-
back, the change in the trust score of document di
can be written as
∆τi=mX
k=1(A−1)ifk∆bfk,(4)
where (A−1)ifkdenotes the entry in the i-th row
andf k-th column ofA−1.
Step 2: Entry-wise decay of A−1.The feedback
term contributes only to the diagonal entries of A.
The off-diagonal entry Aij(i̸=j ) is nonzero when
documents iandjare joined by an edge ( ℓij=±1 ).
Specifically, from Eq. (2), each pairwise relation
term adds ±2w ijat positions (i, j) and(j, i) of
A. The sparsity pattern of Atherefore coincides
with the adjacency structure of G. For a symmet-
ric positive definite matrix with this property, the
Demko–Moss–Smith decay bound (Demko et al.,
1984; Benzi and Razouk, 2007) gives an entry-wise
decay of A−1that is exponential in graph distance:
|(A−1)ij| ≤γqdG(i,j), q=√κ−1√κ+ 1,(5)
whereγ= (1 +√κ)2/(2σ max) =O(1/σ min).
Step 3: Superposition bound.Combin-
ing Eqs. (3)-(5)and using the triangle inequality,
we obtain
|∆τi| ≤mX
k=1(A−1)ifk· |∆b fk|
≤mX
k=12γλf|∆yfk|qdG(i,fk).
C.2 Proof of Lemma 2 (Convergence)
Leteτ(t)=τ(t−1)−η∇L 
τ(t−1)
denote the
gradient descent step before projection, and let
τ(t)=P[0,1]N
eτ(t)
denote the projected iterate.
SinceLis quadratic with Hessian A, for any τwe

have∇L(τ) =A(τ−τ∗). Therefore,
eτ(t)−τ∗=τ(t−1)−η∇L
τ(t−1)
−τ∗
=τ(t−1)−τ∗−ηA
τ(t−1)−τ∗
= (I−ηA)
τ(t−1)−τ∗
.
(6)
With the fixed step size η⋆= 2/(σ min+σ max),
each eigenvalue σofAis mapped to an eigenvalue
1−η⋆σofI−η⋆A. Since σ∈[σ min, σmax], these
eigenvalues lie inh
−κ−1
κ+1,κ−1
κ+1i
.Thus, the largest
absolute eigenvalue of I−η⋆Aisρ=κ−1
κ+1.Be-
cause I−η⋆Ais symmetric, its spectral norm is
equal to its largest absolute eigenvalue, so
∥I−η⋆A∥2=ρ.
Taking norms in (6) then gives
eτ(t)−τ∗≤ρτ(t−1)−τ∗.
The projection operator P[0,1]N(·)is non-
expansive, and hence
∥τ(t)−τ∗∥ ≤ ∥eτ(t)−τ∗∥ ≤ρ∥τ(t−1)−τ∗∥.
D Impact of Mislabeled Edges
A mislabeled edge between documents iandj
changes the Hessian Ain Eq. (2)only at the entries
involving iandj. Under the same assumptions
used in Lemma 1, the entry-wise decay bound on
A−1from Demko et al. (1984) implies that the re-
sulting perturbation to the optimized trust scores
decreases exponentially with the shortest-path dis-
tance from the mislabeled edge. Therefore, the
influence of an edge error diminishes rapidly as it
propagates to more distant documents.
E Prompt Templates
Trust-aware prompt.Each retrieved document
is annotated with both its trust score τiand a trust
label. We assign the label based on the trust score:
high if τi≥0.7 , low if τi≤0.3 , and medium
otherwise. The system instruction is:
You are a QA system. Each document has
a trust score. Strongly prefer HIGH-trust
documents. Ignore or discount LOW-trust
documents, they may contain deliberate
misinformation. Give a short, direct
answer.
Documents are formatted as:DatasetContradictory Supportive
Precision Recall Precision Recall
MS MARCO 0.65 0.18 0.71 0.14
NQ 0.69 0.21 0.68 0.17
TriviaQA 0.58 0.13 0.62 0.12
Table 8: Precision and recall of the NLI-derived contra-
dictory and supportive edges.
[Document 1] (Trust: HIGH, score=0.92):
<text>
[Document 2] (Trust: LOW, score=0.15):
<text>
...
Baseline prompt.VanillaRAG uses the follow-
ing prompt. The other baselines use the prompts
specified in their original papers.
Answer the following question based on the
provided documents. Give a short, direct
answer.
Documents: [Document 1]: <text> ...
Question: <query>
Short answer:
F Quality of NLI-derived Edges
We evaluate the NLI-derived edges against the
known factual and contradictory structure of the
corpus on the three datasets. We treat an edge
between a factual document and a contradictory
document as a ground-truth contradictory relation,
and an edge between two documents of the same
category as a supportive relation. Table 8 reports
the precision and recall of the contradictory and
supportive edges produced by the NLI model.
The NLI-derived edges have moderate precision
and low recall, as the NLI model predicts most doc-
ument pairs as neutral. The low recall indicates that
many relations are omitted, resulting in a sparse
graph, which reflects practical settings where only
a limited subset of inter-document relations can be
identified. The imperfect precision indicates that
the graph also contains some incorrectly labeled
edges. Nevertheless, the NLI-only ablation in Ta-
ble 4 shows that TrustPropRAG still outperforms
the baselines when relying only on these imper-
fect edges, which suggests that trust propagation is
robust to noise and sparsity in graph construction.
G Performance on QACC
We evaluate TrustPropRAG on naturally occurring
conflicts without any synthetic injection. We used

Method EM
VanillaRAG 0.48
InstructRAG 0.54
AstuteRAG 0.50
ReliabilityRAG 0.56
TrustRAG stage 1 0.50
TrustPropRAG0.59
Table 9: EM on QACC (GPT-4o-mini, MiniLM, k=5,
FB=30%).
QACC (Liu et al., 2025a), a human-annotated open-
domain QA dataset in which unambiguous ques-
tions are paired with real web contexts retrieved via
Google Search. The conflicts among these contexts
arise naturally on the web rather than from injected
documents. We randomly sampled 100 questions
from QACC and constructed the evaluation cor-
pus from their retrieved contexts. As shown in Ta-
ble 9, TrustPropRAG achieves the best EM (0.59)
on this naturally conflicting corpus, outperforming
the strongest baseline ReliabilityRAG (0.56) and
improving over VanillaRAG by 0.11. These results
indicate that the effectiveness of TrustPropRAG is
not limited to synthetically injected contradictions
and extends to naturally occurring conflicts.