# Private-RAG: Answering Multiple Queries with LLMs while Keeping Your Data Private

**Authors**: Ruihan Wu, Erchi Wang, Zhiyuan Zhang, Yu-Xiang Wang

**Published**: 2025-11-10 21:12:32

**PDF URL**: [https://arxiv.org/pdf/2511.07637v1](https://arxiv.org/pdf/2511.07637v1)

## Abstract
Retrieval-augmented generation (RAG) enhances large language models (LLMs) by retrieving documents from an external corpus at inference time. When this corpus contains sensitive information, however, unprotected RAG systems are at risk of leaking private information. Prior work has introduced differential privacy (DP) guarantees for RAG, but only in single-query settings, which fall short of realistic usage. In this paper, we study the more practical multi-query setting and propose two DP-RAG algorithms. The first, MURAG, leverages an individual privacy filter so that the accumulated privacy loss only depends on how frequently each document is retrieved rather than the total number of queries. The second, MURAG-ADA, further improves utility by privately releasing query-specific thresholds, enabling more precise selection of relevant documents. Our experiments across multiple LLMs and datasets demonstrate that the proposed methods scale to hundreds of queries within a practical DP budget ($\varepsilon\approx10$), while preserving meaningful utility.

## Full Text


<!-- PDF content starts -->

Preprint.
PRIVATE-RAG: ANSWERINGMULTIPLEQUERIES WITH
LLMS WHILEKEEPINGYOURDATAPRIVATE
Ruihan Wu∗
Computer Science and Engineering
University of California, San Diego
ruw076@ucsd.eduErchi Wang∗
Halıcıo ˘glu Data Science Institute
University of California, San Diego
erw011@ucsd.edu
Zhiyuan Zhang
Department of Computer Science
University of California, Los Angeles
hollyzhang03@ucla.eduYu-Xiang Wang
Halıcıo ˘glu Data Science Institute
University of California, San Diego
yuxiangw@ucsd.edu
ABSTRACT
Retrieval-augmented generation (RAG) enhances large language models (LLMs) by
retrieving documents from an external corpus at inference time. When this corpus
contains sensitive information, however, unprotected RAG systems are at risk of
leaking private information. Prior work has introduced differential privacy (DP)
guarantees for RAG, but only in single-query settings, which fall short of realistic
usage. In this paper, we study the more practical multi-query setting and propose
two DP-RAG algorithms. The first, MURAG, leverages an individual privacy filter
so that the accumulated privacy loss only depends on how frequently each document
is retrieved rather than the total number of queries. The second, MURAG-ADA,
further improves utility by privately releasing query-specific thresholds, enabling
more precise selection of relevant documents. Our experiments across multiple
LLMs and datasets demonstrate that the proposed methods scale to hundreds of
queries within a practical DP budget ( ε≈10 ), while preserving meaningful utility.
1 INTRODUCTION
Retrieval-augmented generation (RAG) has become a popular approach for deploying large language
models (LLMs) in real-world applications. A core feature of RAG is its reliance on an external
dataset as the primary knowledge source at inference time. For example, a medical RAG system
may retrieve historical patient records to answer clinical questions more accurately. However, such
external datasets often contain sensitive or confidential information. In domains like healthcare or
law, the retrieved content may expose private records, raising serious privacy concerns. Prior work
has shown that RAG systems without proper safeguards are vulnerable to information leakage (Naseh
et al., 2025; Liu et al., 2025; Anderson et al., 2024; Li et al., 2025; Zhang et al., 2025; Zeng et al.,
2024a; Jiang et al., 2024; Peng et al., 2024), compromising data owner privacy and user trust.
Differential privacy (DP) is a widely adopted framework for providing rigorous guarantees on
individual data protection. Recent work (Koga et al., 2024) has proposed DPSparseV oteRAG, a RAG
system that ensures the generated answer satisfies DP with respect to the external dataset, fora single
user query. Empirical results demonstrate that this approach outperforms the baseline using a public
LLM without the external dataset, while achieving anε-DP guarantee withε≈10.
In realistic deployments, many queries may be issued by one or more users. A naïve approach
that applies DPSparseV oteRAG to each query and relies on standard composition theorems quickly
exhausts a reasonable privacy budget. As our experimental results (Figure 2) show, to achieve
reasonable utility, this approach may require a privacy budget as large as ε= 1000 , which is generally
considered too weak. This raises a key question:
∗Equal contribution.
1arXiv:2511.07637v1  [cs.LG]  10 Nov 2025

Preprint.
Can we design a differentially private RAG algorithm that handles hundreds of queries while ensuring
both meaningful privacy and utility?
We answer this question affirmatively and summarize our contributions below.
Circumventing Query-Composition Overhead with Per-Document Rényi Filters.We propose
a novel framework for multi-query differentially private RAG. Rather than composing a sequence of
single-query DP-RAG executions, where the privacy budget grows with the number of queries, we
leverageindividual R’enyi filters(Feldman & Zrnic, 2021). These filters bound privacy loss based on
how many times each document is retrieved, yielding substantial savings when queries access largely
disjoint documents. To the best of our knowledge, this is the first application of privacy filters in the
RAG setting. Our framework can incorporate any single-query private RAG algorithm.
Two DP Multi-RAG Algorithms for Varying Test Query Dependencies.We propose two dif-
ferentially private RAG algorithms for the multi-query setting through threshold-based screening of
relevant documents and their are tailored to the degree of relevance among test-time queries. MURAG
(Algorithm 1) uses a fixed relevance threshold across all queries and is sufficient to work well for
settings where queries are independent and do not share relevant private documents. MURAG-ADA
(Algorithm 2) allocates a small portion of the privacy budget to release a query-specific relevance
threshold, enabling more efficient use of the budget when queries are related and share overlapping
relevant documents.
Practical Multi-Query RAG with Non-Trivial Privacy Guarantees.We evaluate our algorithms
through extensive experiments on three LLMs (OPT-1.3B, Pythia-1.4B, and Mistral-7B). Our evalua-
tion spans three types of datasets: standard RAG benchmarks (Natural Questions,Trivia Questions), a
more challenging multi-hop QA dataset (MQuAKE) with correlated questions, and a privacy-sensitive
application (ChatDoctor) consisting of patient–doctor QA pairs. Empirical results show that both
of our methods can answer hundreds of queries within a total privacy budget of ε≈10 while
maintaining reasonable utility, a trade-off no baseline method achieves. Furthermore, we demonstrate
that our approaches with ε= 10 effectively defend against a state-of-the-art multi-query membership
inference attack for RAG.
2 DIFFERENTIALPRIVATERETRIEVAL-AUGMENTEDGENERATION
Notation.Let Vdenote a finite vocabulary, and let x∈ V∗represent a prompt of arbitrary length.
A document set of arbitrary size is denoted by D={z 1, z2, . . .} , where each document zi∈ V∗. For
convenience, we denote byZthe document space, i.e., the set of all finite-length sequences overV.
Differential Privacy.We denote the data space by X. Two datasets D, D′∈ X∗are said to be
neighboring if they differ in at most one element. In this work, we studydocument-level privacyunder
the add/remove neighboring relation, where the data universe is V∗and two datasets are neighbors if
they differ by exactly one document.
Definition 1(Differential Privacy (Dwork et al., 2006b)).A randomized algorithm M:X∗→Ω
satisfies (ε, δ) -differential privacy if, for all neighboring datasets X, X′∈ X∗and all measurable
subsetsO⊆Ω,Pr[M(X)∈O]≤eεPr[M(X′)∈O] +δ.
Definition 2(Rényi Differential Privacy (Mironov, 2017)).A randomized algorithm M:X∗→
Ωsatisfies (α, ε) -Rényi Differential Privacy (RDP) if, for all neighboring datasets X, X′∈
X∗, the Rényi divergence of order α >1 between M(X) andM(X′)is at most ε, i.e.
Dα(M(X)∥M(X′))≤ε.
We may also considerindividual-levelRDP, where the Rényi divergence is evaluated on neighboring
datasets that differ in a particular data point zi. LetS(zi, n)denote the set of dataset pairs (S,˜S)
such that|S|,| ˜S|< nandz i∈S△ ˜S, i.e. exactly one ofS, ˜Scontainsz i.
Definition 3(Individual Rényi Differential Privacy).A randomized algorithm M:X∗→Ω satisfies
(α, ε)-individual RDP at pointz iif, for all(X, X′)∈ S(z i, n),D α(M(X)∥M(X′))≤ε
Aprivacy filteris a stopping rule that tracks cumulative privacy loss and halts execution once the
privacy budget is exceeded, thereby ensuring that the designed privacy guarantees are never violated.
For completeness, we briefly introduce individual RDP filters; for a rigorous treatment, we refer
readers to Feldman & Zrnic (2021).
2

Preprint.
Definition 4((Individual) Rényi Differential Privacy Filters (Feldman & Zrnic, 2021)).A random
variable Fα,B: Ω∗→ {CONT,HALT} is a privacy filter for (α, B) -RDP if it halts the execution
of an algorithm before its accumulated (individual) privacy loss, measured in α-Rényi divergence,
exceedsB.
Problem Setting.We study retrieval-augmented generation (RAG) with a sensitive external
document collection. A decoder-only LLM with greedy decoding is modeled as a function
LLM :V∗× Z → V . Given a user prompt x∈ V∗, the system retrieves a subset of doc-
uments Dx=R k(x, D) from a private external corpus D∈ Z , where the retrieval function
Rk:V∗× Z → Z returns the kmost relevant documents. The corpus Dcontains sensitive
documents, each potentially corresponding to private user information.
We adopt a threat model in which the adversary has no direct access to the corpus Dbut may
issue arbitrary prompts xto the RAG system. The underlying LLM is assumed to be public and
independent of D. Our objective is to design a differentially private RAG mechanism that, given
a set of queries {q1, . . . , q T}, the sensitive corpus D, a public LLM, and a total privacy budget ε,
generates high-utility responses while guaranteeingε-differential privacy with respect to corpusD.
3 METHODOLOGY
3.1 TECHNICALOVERVIEW
Improved Privacy Accounting via Per-Document Privacy Filters.In retrieval-augmented gen-
eration (RAG), each query interacts with only a small, query-specific subset of the corpus D. This
sparsity implies that most documents are accessed only rarely1. We leverage this by introducing a
per-document privacy filter that monitors cumulative privacy loss and blocks further retrieval once a
document’s budget is exhausted. Because privacy cost is incurred only upon retrieval, this accounting
scheme naturally scales with the frequency of document access rather than the total number of queries.
Screening Relevant Documents via Relevance Thresholding.If RAG were applied directly to
the entire corpus, every document would be touched by each query, and per-document privacy filters
would provide no benefit. To prevent this, MURAG employs a global relevance threshold τ2: only
documents whose scores exceed τare retrieved and incur privacy cost. A document is excluded from
all future retrievals once its privacy budget is exhausted. Since τis fixed in advance and independent
of the data, introducing this threshold does not consume additional privacy budget.
Handling Correlated Queries via Adaptive Thresholding.When queries arecorrelated, meaning
their sets of relevant documents substantially overlap, a fixed relevance threshold τcan lead to
inefficiencies. Specifically, since the relevance score distribution may shift across queries, a uniform
threshold can cause some queries to retrieve more documents than necessary, prematurely exhausting
the budgets of relevant documents and limiting their availability for later queries. To mitigate this, we
propose MURAG-ADA, which privately selects a query-specific threshold τttailored to the relevance
distribution of each query. By combining per-document privacy accounting with the private release
of cumulative statistics, MURAG-ADA restricts retrieval to the most relevant documents, thereby
reducing unnecessary budget consumption and preserving utility across correlated queries.
Single-Query DP RAG after Screening.After thresholding, per-document privacy filters ensure
that each retrieved document incurs loss only when used and is removed once its budget is exhausted.
The resulting set is then passed to a single-query DP-RAG algorithm to generate the response. As
shown in Algorithms 1 and 2, our multi-query framework is modular, supporting any private single-
query RAG method. In this work, we instantiate it with a pure-DP variant of the algorithm from Koga
et al. (2024) (Algorithm 7).
3.2 DP-RAGWITH AFIXEDTHRESHOLD
InMURAG , we impose a fixed relevance threshold τto screen documents before retrieval. The
threshold can either be publicly specified or privately estimated using a small portion of the privacy
1We provide a more detailed discussion of this sparsity in Appendix B.
2Intuitively, the threshold τcan be viewed as a chosen percentile of the relevance score distribution for a
given query, ensuring that only the top-ranked documents contribute to privacy cost.
3

Preprint.
budget. The complete procedure is summarized in Algorithm 1 and the privacy guarantee is given
in Theorem 1. At a high level, the algorithm maintains a per-document privacy budget that is
decremented whenever the document is retrieved. For each query, it first updates the active set of
documents and then filters out most documents with scores below τ. Among the remaining documents,
the top- kare selected by relevance, and a differentially private single-query RAG procedure is invoked
to generate the response.
Since whether a document exceeds the constant threshold τdepends only on its own score and not
on the scores of other documents, the use of(Individual) Rényi Differential Privacy Filtersis valid.
Consequently, for each query, privacy loss is charged only to the small subset of documents that pass
the threshold, using a per-query budget εq, rather than to the entire corpus. The privacy guarantee of
MURAG is stated in Theorem 1, and the proof is deferred to Appendix D.
Theorem 1(Privacy Guarantee of Algorithm 1). MURAG satisfies ε-differential privacy provided
that the initial privacy budget assigned to each documentz∈Dis at mostε.
Algorithm 1:MURAG: Differentially PrivateMulti-QueryRetrieval-AugmentedGeneration
Input:Private datasetD, sequence of queries{q 1, . . . , q T}, per-query DP budgetε q,#retrieved
documentsk, maximum retrievals per documentM, relevance thresholdτ
Set:Initialize individual budget for each documentz∈D:E(z) =M·ε q
1fort= 1, . . . , Tdo
2A t={z∈D| E(z)≥ε q}▷Update active document set
3D qt={z∈A t|r(z, q t)> τ}▷Filter relevant documents
4forz∈D qtdo
5E(z)← E(z)−ε q ▷Update budget for retrieved documents
6Dk
qt=TOP-K(D qt, k, r(·, q t))▷Select top-krelevant documents
7a t=DP-RAG(x, Dk
qt,LLM, ε q)▷Generate DP response via Algo. 7
8return(a 1, . . . , a T)
3.3 DP-RAGWITHADAPTIVETHRESHOLD
The score distribution can vary substantially across different questions, making a single global
threshold ineffective. To guarantee the performance of single-query DP-RAG, the threshold must be
set low enough to retrieve sufficient documents for all queries. However, this often results in many
unnecessary documents being retrieved: although single-query DP-RAG uses at most Kdocuments,
any additional documents above Kstill incur privacy loss, wasting budget on unused data. This
inefficiency can significantly degrade performance when those documents are needed by later queries.
To overcome this limitation, we propose MURAG-ADA, which privately releases a query-specific
thresholdτ tadapted to the relevance distribution of each query.
The adaptive procedure works by discretizing the relevance scores into bins and then releasing noisy
prefix sums until the cumulative count of retrieved documents exceeds K. This mechanism tailors
the cutoff of documents to each query, reducing unnecessary budget consumption on irrelevant
documents and preserving utility across multiple queries. We will see in the experimental section that
this approach especially yields clear utility gains on datasets with high correlated queries. The full
procedure is summarized in Algorithm 2.
Notice that in Algorithm 2, we use kas a stopping criterion instead of releasing differentially
private top- krelevance scores. This is because releasing a noisy top- kscore for each query would
make the privacy budget grow linearly with the number of queries and incur loss on all documents,
thereby breaking the per-document privacy filter. By contrast, our prefix-sum approach (Step 1 of
Algorithm 2) incurs privacy loss only on the documents that appear in the released prefix sums, while
all other documents remain untouched. This concentrates the privacy cost of this step still on a small
subset, yielding tighter accounting and more efficient budget use across multiple queries. The privacy
guarantee of MURAG-ADAis stated in Theorem 2, and the proof is deferred to Appendix D.
Theorem 2(Privacy Guarantee of Algorithm 2). MURAG-ADA satisfies ε-differential privacy
provided that the initial privacy budget allocated to each documentz∈Dis at mostε.
4

Preprint.
Algorithm 2:MURAG-ADA: DPMulti-QueryRAGwithAdaptive Threshold
Input: Private dataset D, sequence of queries {q1, . . . , q T}, per-query budget εq, number of
retrieved documentsk, maximum retrievals per documentM
Set:Initialize budget for eachz∈D:E(z)←M·ε q. Split budget:ε q=εthr+εRAG.
Require:Discretization of similarity scores into bins[a i, ai+1)B
i=1
1fort= 1, . . . , Tdo
/*Step 1: Adaptive thresholding via noisy prefix sums */
2˜s←0,A t←∅
3fori= 1, . . . , Bdo
4A(i)
t={z∈D|r(z, q t)∈[a i, bi],E(z)≥ε thr}
5˜s←˜s+|A(i)
t|+ Lap(1/ε thr)
6A t←A t∪A(i)
t
7forz∈A(i)
tdo
8E(z)← E(z)−ε thr
9if˜s≥kthen
10τ t=ai; break▷Release threshold
/*Step 2: DP-RAG on adaptively selected active set */
11A′
t={z∈A t| E(z)≥ε RAG}
12D qt=TOP-K(A′
t, k, r(·, q t))
13a t=DP-RAG(x, D qt,LLM, ε RAG;τt)▷single-query RAG, Algorithm 7
14forz∈A′
tdo
15E(z)← E(z)−ε RAG
16return(a 1, . . . , a T)
1.0 1.2 1.4 1.6 1.8 2.0
# Questions Where the Document is in T op-50102103# DocumentsNatural Questions
1.0 1.2 1.4 1.6 1.8 2.0
# Questions Where the Document is in T op-50102103# DocumentsTrivia Questions
0 10 20 30 40
# Questions Where the Document is in T op-50100101102103104# DocumentsMQuAKE Questions
Figure 1: Histogram of document reuse across questions. Each bar shows how many questions a document
appears in among the top- Kretrieved results ( K= 50 ). The x-axis indicates the number of questions per
document, and the y-axis shows the count of such documents.
4 EXPERIMENT
4.1 DATASET
Datasets set-up.We first evaluate our methods ontwo independent question sets:Natural Questions
andTrivia Questions. These are standard benchmarks for evaluating RAG systems and have been
used in prior work on per-query DP for RAG (Koga et al., 2024). Following their setup, we randomly
subsample 100 questions from each dataset to reduce computational overhead. Importantly, the
questions are independent of one another, and each requires a disjoint set of relevant documents
from the external database. To quantify document reuse, we examine how frequently each document
appears in the top- Kretrieved results ( K= 50 ) across questions. As shown in Figure 1, in both
Natural QuestionsandTrivia Questions, most documents are retrieved for only one or two queries.
Thus, we expectMURAGto perform sufficiently well on these two datasets.
Second, we consider acorrelated question set,MQuAKE(Zhong et al.). This dataset contains
sequences of semantically related single-hop questions that together form multi-hop reasoning chains.
We select 100 such sequences, yielding 400 individual questions for evaluation. Since questions
5

Preprint.
in the same sequence share entities (subjects or objects), their relevant documents substantially
overlap. As shown in Figure 1, many documents appear across multiple questions.We therefore
expectMURAG-ADAto have an advantage overMURAG.
Finally, we evaluate onChatDoctor(Li et al., 2023), aprivacy-sensitive application of RAGin
the healthcare domain. This dataset consists of QA interactions between patients and doctors. We
sample 100 patient questions as our test set.This evaluation tests the effectiveness of our methods in
a real-world sensitive setting and their robustness against privacy attacks.
External datasets reflecting both standard and privacy-sensitive settings.For Natural Questions,
Trivia Questions, and MQuAKE Questions, we use Wikipedia of ∼20M documents as the external
knowledge source following the standard RAG setup (Chen et al., 2017; Lewis et al., 2020). For
ChatDoctor Questions, the external dataset consists of the remaining ∼200K QA pairs from the
original ChatDoctor dataset, excluding the 100 patient questions used for testing. This setup reflects
a realistic privacy-sensitive application, where the external corpus contains private information.
QA evaluation metric.For Natural Questions, Trivia Questions and MQuAKE Questions, the
datasets provide a list of all acceptable correct answers for each question. Following the evaluation
protocol of Koga et al. (2024), we use theMatch Accuracymetric: a prediction is scored as 1 if it
contains any correct answer, and 0 otherwise. For Chatdoctor Questions, we adopt the evaluation
metric from the original dataset paper, using the F1 score of BERTScore (Zhang et al., 2020) to
measure semantic similarity between the predicted response and the ground-truth answer.
4.2 MODEL ANDMETHOD SET-UP
Model set-up.Our RAG pipeline integrates three pre-trained LLMs: OPT-1.3B (Zhang et al., 2022),
Pythia-1.4B (Biderman et al., 2023), and Mistral-7B (Jiang et al., 2023). For document retrieval, we
use the Dense Passage Retriever (DPR) (Karpukhin et al., 2020) to compute dense query-document
relevance scores.
Baseline methods.We compare our two proposed methods with five baselines. The first is
NAIVE-MULTI-RAG (Algorithm 8), which applies the per-question DP RAG method, DPSparse-
V oteRAG, independently to each query and uses the standard sequential composition theorem (Dwork
et al., 2006a) to compute the overall privacy guarantee. The second baseline applies subsampling
amplification to the first baseline, NAIVE-MULTI-RAG, which we calledSUBSAMPLING-MULTI-
RAG. Specifically, for each query, we first subsample the external dataset using Poisson sampling
with rate η, and then apply DPSparseV oteRAG on this subsampled dataset. The overall privacy
guarantee is then computed using sequential composition combined with the amplification by subsam-
pling (Balle et al., 2018). The third baseline privatizes the external dataset of RAG under differential
privacy (DP) and then uses the resulting synthetic dataset as the knowledge source for evaluation.
In this setup, the answers are guaranteed to satisfy DP since they are derived from a privatized
dataset. We adoptPrivate Evolution(PE; Xie et al. (2024)), a state-of-the-art DP synthetic text
generation method that also aligns with the query-access setting of RAG. Specifically, PE first queries
an LLM to produce an initial dataset within the same domain as the private corpus, and then refines
its distribution under DP to better approximate that of the private dataset. To ensure consistency, for
each pretrained LLM used in RAG, we use the same model as the query API in PE. The other two
are non-private baselines:Non-RAG, which generates answers using the pretrained LLM without
retrieval, andNon-Private-RAG, which performs retrieval-augmented generation without any privacy
mechanism. We describe implementation details in Appendix E.
Privacy budget setup for DP algorithms.Following the setup in Koga et al. (2024), we vary the
per-query RAG privacy budget εq∈ {2,5,10,15,20,30,40} to explore the privacy-utility trade-off.
ForNAIVE-MULTI-RAG , the total privacy budget is T·ε q, where Tis the number of questions.
For MURAG and MURAG-ADA, the total budget is M·ε q, where Mis the number of retrieved
documents with nonzero privacy loss3. In our main results, we conservatively set M= 1 for a
realistic privacy region in MURAG and MURAG-ADAand set εthras1.0in MURAG-ADA4. For
the baseline SUBSAMPLING-MULTI-RAG, we consider the subsampling rate η= 0.1,0.01,0.001
3To enable a meaningful comparison, we convert our privacy guarantee, originally expressed in (∞, ε) -RDP,
into an equivalentε-DP guarantee (Mironov, 2017).
4We will see the detailed analysis of the choices ofMandε thrin Section 4.4
6

Preprint.
101102103
Total Budget 
3×101
4×101
6×101
Average Match Accuracy
Natural Questions, OPT-1.3B
Non-RAG
Non-Private-RAG
Naive-Multi-RAG
Subsampling-Multi-RAG
MuRAG (Ours)
MuRAG-Ada (Ours)
PE
101102103
Total Budget 
2×101
3×101
4×101
Average Match Accuracy
Natural Questions, Pythia-1.4B
Non-RAG
Non-Private-RAG
Naive-Multi-RAG
Subsampling-Multi-RAG
MuRAG (Ours)
MuRAG-Ada (Ours)
PE
101102103
Total Budget 
7×101
8×101
Average Match Accuracy
Natural Questions, Mistral-7B
Non-RAG
Non-Private-RAG
Naive-Multi-RAG
Subsampling-Multi-RAG
MuRAG (Ours)
MuRAG-Ada (Ours)
PE
101102103
Total Budget 
4×101
5×101
6×101
7×101
Average Match Accuracy
Trivia Questions, OPT-1.3B
Non-RAG
Non-Private-RAG
Naive-Multi-RAG
Subsampling-Multi-RAG
MuRAG (Ours)
MuRAG-Ada (Ours)
PE
101102103
Total Budget 
4×101
5×101
6×101
7×101
Average Match Accuracy
Trivia Questions, Pythia-1.4B
Non-RAG
Non-Private-RAG
Naive-Multi-RAG
Subsampling-Multi-RAG
MuRAG (Ours)
MuRAG-Ada (Ours)
PE
101102103
Total Budget 
8×101
9×101
Average Match Accuracy
Trivia Questions, Mistral-7B
Non-RAG
Non-Private-RAG
Naive-Multi-RAG
Subsampling-Multi-RAG
MuRAG (Ours)
MuRAG-Ada (Ours)
PE
101102103
Total Budget 
6×101
7×101
Average Match Accuracy
MQuAKE Questions, OPT-1.3B
Non-RAG
Non-Private-RAG
Naive-Multi-RAG
Subsampling-Multi-RAG
MuRAG (Ours)
MuRAG-Ada (Ours)
PE
101102103
Total Budget 
5×101
6×101
7×101
Average Match Accuracy
MQuAKE Questions, Pythia-1.4B
Non-RAG
Non-Private-RAG
Naive-Multi-RAG
Subsampling-Multi-RAG
MuRAG (Ours)
MuRAG-Ada (Ours)
PE
101102103
Total Budget 
8×101
9×101
Average Match Accuracy
MQuAKE Questions, Mistral-7B
Non-RAG
Non-Private-RAG
Naive-Multi-RAG
Subsampling-Multi-RAG
MuRAG (Ours)
MuRAG-Ada (Ours)
PE
Figure 2: Privacy-Utility tradeoffs of our two proposed methods (MURAG and MURAG-ADA) compared to
baselines across three pretrained LLMs and two categories of question sets.
and calculate the corresponding εqto satisfy the the varying total budget {2,5,10,15,20,30,40} .
For the baseline PE, we test withε∈ {10,200}.
Membership inference attack in RAG.To assess the effectiveness of our privacy-preserving
methods, we evaluate them against the membership inference attack (MIA). The objective of MIA is
as follows: given a candidate document xand a model system R(·;D) trained on a private dataset D,
the adversary aims to determine whether x∈D by computing a membership score s(x, R(·;D)) .
Without loss of generality, we assume higher scores indicate higher membership likelihood. Applying
the attack to an in-distribution set Din⊂D and an out-of-distribution set Dout(with no overlap with
D) allows us to derive the TPR–FPR curve and compute the AUC, which serves as the evaluation
metric for attack success.
We focus on scenarios where the adversary can issue multiple queries to the system, as this set-
ting substantially amplifies the attack strength. To model this, we adopt theInterrogation Attack
(IA)(Naseh et al., 2025), a state-of-the-art MIA specifically designed to exploit multi-query access
in RAG systems. For each document x, IA generates m= 30 tailored questions together with their
corresponding answers implied by x. Then each question is concatenated with the necessary context
to ensure the target document can be retrieved, and the query is then submitted to the RAG system.
The membership score is defined as the accuracy of the RAG system across these mquestions, where
higher accuracy implies a greater likelihood that the document is present in the external dataset and
is being retrieved to answer the queries. Additional implementation details, including the question
generation process, are provided in Appendix E.
4.3 MAINRESULTS
Results on two standard RAG benchmarks (independent question sets).Figure 2 shows the
performance of our two proposed methods compared with three baselines across three pretrained
LLMs onNatural QuestionsandTrivia Questions. Both of our methods outperform the Non-RAG
baseline in most cases under a total privacy budget ofε= 10.
7

Preprint.
0.00.20.40.60.81.0False Positive Rate0.00.20.40.60.81.0True Positive RateNon-Private-RAG (AUC = 0.606)MuRAG≤= 10(AUC = 0.504)MuRAG-Ada≤= 10(AUC = 0.531)
Figure 3:Left:Privacy-utility tradeoffs of our two methods and baselines.Right:TPR-FPR curves of IA
(Membership Inference Attack with multiple queries). Both experiments are conducted with Mistral-7B and
ChatDoctor datasets.
In contrast, all DP baselines (NAIVE-MULTI-RAG, SUBSAMPLING-MULTI-RAG, PE) either under-
perform the Non-RAG model or require an impractically large privacy budget to achieve comparable
performance. The baseline NAIVE-MULTI-RAG requires an impractically large budget, exceeding
ε= 103, to achieve comparable utility. This highlights that our approaches make differential privacy
practical in the multi-query RAG setting by leveraging more tailored compositions, enabling strong
utility within a realistic privacy budget. The SUBSAMPLING-MULTI-RAG baseline consistently
underperforms the Non-RAG model. This degradation is likely due to the reduced number of effective
documents (that provide the ground truth answers) after subsampling. For example, if there are 50
relevant documents for a query, subsampling at a rate of 0.1leaves only about 5accessible documents,
making it difficult for DPSPARSEVOTERAG to produce correct answers within the per-query budget
εq≈0.71 (computed from the overall budget ε, total queries T= 100 , and sampling rate η= 0.1 ).
The results demonstrate that the individual privacy accounting framework provides a more effective
composition mechanism than subsampling amplification for multi-query RAG problem; a more
detailed discussion of this limitation is provided in Section 6. The PE baseline performs even worse
than Non-RAG at ε= 200 for many settings, which we attribute to objective misalignment: PE
optimizes for distributional similarity (e.g., measured by Fréchet Inception Distance (FID; Heusel
et al. (2017))) rather than preserving factual content. Indeed, we find PE achieves a better FID score
atε= 200 but yields lower task performance than at ε= 10 on the setting of Trivia Questions and
OPT-1.3B, further supporting this explanation.5
Lastly, on these two datasets, MURAG outperforms MURAG-ADA, which aligns with our ex-
pectations. Since the questions are independent, adaptive thresholding provides little benefit and
additionally consumes extra privacy budget.
Results on multi-hop questions (correlated question set).Figure 2 shows the performance of our
two proposed methods compared with three baselines across three pretrained LLMs onMQuAKE
Questions. Overall, the relative trends between our methods and the baselines are consistent with the
independent question setting. However, a key difference emerges in the comparison between our two
approaches: MURAG-ADAperforms significantly better than MURAG. This result is aligned with
our intuition, as adaptive thresholding is particularly advantageous when questions are correlated and
share overlapping relevant documents.
Results on privacy-sensitive application.The left plot in Figure 3 shows the performance of our
methods and baselines on Mistral-7B with the ChatDoctor dataset. The results mirror the trends
observed in the previous benchmarks: both of our methods outperform the baselines in this practical,
privacy-sensitive setting. In particular, MURAG surpasses the Non-RAG baseline atε= 10.
We also evaluate robustness against the Interrogation Attack (IA) on ChatDoctor. Specifically, we
test three RAG systems: Non-Private-RAG, MURAG ( ε= 10 ), and MURAG-ADA( ε= 10 ). The
right plot in Figure 3 reports the corresponding TPR–FPR curves. Without protection, IA achieves
a non-trivial AUC of ≈0.6 . In contrast, both of our methods reduce the AUC to ≈0.5 , making
5We confirm that the FID score improves from ε= 10 toε= 200 (0.066 to0.036 ; lower is better) on the
setting of Trivia Questions and OPT-1.3B, yet RAG utility drops, underscoring the mismatch between FID and
factual fidelity required for RAG.
8

Preprint.
Table 1: Precision of retrieved documents under different thresholding approaches, measured as the
percentage of truly top-50 relevant documents among the retrieved.
Independent Question Set Correlated Question Set
Natural Questions Trivia Questions MQuAKE Questions
Constant Thresholding (in MURAG) 78.8% 72.2% 17.6%
Adaptive Thresholding (in MURAG-ADA) 92.6% 94.6% 40.7%
Adaptive Thresholding (Non-private top-K-release) 99.4% 99.6% 43.5%
M=1 M=50.00.10.20.30.40.50.60.7Precision in T op-50MURAG
MURAG-ADA
101102103
Total Budget 
0.680.700.720.740.76Average Match Accuracy
MQuAKE Questions, OPT-1.3B
Non-RAG
Non-Private-RAG
Naive-Multi-RAG
MuRAG (M=1)
MuRAG (M=5)
MuRAG-Ada (M=1)
MuRAG-Ada (M=5)
101102103
Total Budget 
0.620.640.660.680.700.72Average Match Accuracy
MQuAKE Questions, Pythia-1.4B
Non-RAG
Non-Private-RAG
Naive-Multi-RAG
MuRAG (M=1)
MuRAG (M=5)
MuRAG-Ada (M=1)
MuRAG-Ada (M=5)
101102103
Total Budget 
0.880.890.900.910.920.93Average Match Accuracy
MQuAKE Questions, Mistral-7B
Non-RAG
Non-Private-RAG
Naive-Multi-RAG
MuRAG (M=1)
MuRAG (M=5)
MuRAG-Ada (M=1)
MuRAG-Ada (M=5)
Figure 4: Comparison of M= 1 andM= 5 in the individual privacy accounting framework. The left plot
shows the retrieval precisions of two methods with M= 1,5 . Right three plots show the trade-off between the
QA performance and theε total in DP.
the attack ineffective. These findings demonstrate that our approaches provide practical privacy
protection atε= 10in a real-world sensitive application.
Takeaway.Across all evaluations, our methods consistently outperform baseline approaches under
practical privacy budgets. On independent question sets, MURAG achieves strong performance as
expected, while on correlated multi-hop questions, MURAG-ADAshows clear advantages due to its
adaptive thresholding. Finally, in the privacy-sensitive ChatDoctor application, both methods not
only improve utility over baselines but also effectively mitigate state-of-the-art membership inference
attacks. Together, these results demonstrate that our approaches make differentially private RAG both
practical and robust across diverse settings.
4.4 FURTHERANALYSIS OFMURAGANDMURAG-ADA
Comparison between thresholding approaches in our two methods.The two methods have
different performance as discussed above, and the difference is between the constant thresholding
and the DP-released adaptive thresholding. To quantify this effect, Table 1 reports the precision
under bothconstant thresholds(in MURAG) andadaptive thresholds(in MURAG-ADA), where we
measure the percentage of truly top-50 documents among the retrieved documents for each question
and calculate the average over questions as the precision. We observe that precision under MURAG
is particularly low for the correlated question setMQuAKE Questions, whereas MURAG-ADA
significantly improves retrieval precision on these datasets through its adaptive thresholds. This
improvement in retrieval quality directly contributes to the superior performance of MURAG-ADA
in the setting ofcorrelated question set.
Effect of different Min the individual privacy accounting framework.Both of our proposed
methods include a hyperparameter M, which controls the maximum number of queries for which
an individual document’s privacy budget can be consumed. In our main results (Figure 2), we set
M= 1 to ensure strict per-document privacy usage. However, this setting may limit utility: once a
document is used for one query, it becomes unavailable for future queries, even if it would have been
highly relevant. To better understand the impact of M, we evaluate our two methods with a larger
value of M= 5 . The left plot in Figure 4 shows a substantial increase in Top-50 retrieval precision
when using M= 5 , indicating better access to relevant documents. This improvement translates into
higher end-to-end RAG utility, as shown in the three plots on the right. However, increasing Malso
leads to a higher total privacy cost (ε total=M·ε q).
9

Preprint.
0.5 1.0 2.0 10.0 20.0
thr
0.00.20.40.60.81.01.2Absolute Error
Figure 5: Absolute error of releas-
ingτ tin MURAG-ADA.Budget allocation in MURAG-ADA.An important hyperpa-
rameter, εthr, controls the privacy budget allocated for releasing
the threshold τt. Figure 5 shows the absolute error between the
true top- Kthreshold and the estimated threshold returned by
the DP threshold-release procedure (Lines 3–10 in Algorithm 2)
on theTrivia Questionsdataset. As shown, the estimation error
remains small (absolute error ≤0.2 ) when εthr≥1.0 , which
is quite reasonable given that most scores lie between 70 and
100. Based on this trade-off, we choose εthrso that it achieves
a small absolute error while consuming only a small fraction of
the total budget ε, leaving the remaining budget for the DP-RAG
token-generation steps.
5 RELATEDWORK
Recent studies identify two main privacy risks in retrieval-augmented generation (RAG) systems.
The first is membership inference attacks (MIA) (Shokri et al., 2017), which test whether a specific
document is in the private external dataset, often via adversarial prompts (Naseh et al., 2025; Liu
et al., 2025; Anderson et al., 2024) or scoring mechanisms (Li et al., 2025). The second is data
reconstruction attacks, which aim to recover document content using adversarial prompts (Zhang
et al., 2025; Zeng et al., 2024a; Jiang et al., 2024) or poisoning triggers (Peng et al., 2024). Together,
these works highlight the growing need for principled privacy-preserving algorithms for RAG.
Several DP-based defenses have been proposed. Koga et al. (2024) introduced a single-query DP-RAG
system, and others (Yao & Li; Grislain, 2025) studied DP release of document identifiers. However,
none of these methods address the realistic multi-query setting. In addition to DP based methods,
empirical defenses have also been explored, including paraphrasing retrieved documents (Yao & Li)
and dataset privatization (Zeng et al., 2024b), but these lack formal privacy guarantees and remain
vulnerable to strong adversarial attacks. A complementary line of work considers protecting user
queries in cloud-hosted RAG (Cheng et al., 2024), which addresses a different threat model than ours.
For additional related work on the use of differential privacy in large language models and the line of
individual privacy accounting, we refer readers to Appendix A.
6 DISCUSSION
Why Privacy Filter rather than Amplification by Subsampling?As surveyed in Section A,
privacy amplification by subsampling (Balle et al., 2018; Wang et al., 2019; Zhu & Wang, 2019)
is widely used in DP LLM applications, such as DP prompt tuning and DP in-context learning, to
enhance generation quality. However, this technique is not well-suited for DP RAG as shown in the
experiment section. We would like to discuss the reason behind:
•In prompt tuning, the goal is to learn a single task-specific prompt that can generalize to all future
queries. In DP in-context learning, a small number of example inputs are selected under DP
constraints and reused across queries. In contrast, RAG does not allow for such "unified" prompts
or examples: each test-time query requires retrieving and using query-specific documents, which
must be handled privately, which makes individual privacy filter a more suitable choice.
•Moreover, in prompt tuning and in-context learning, all data points in the private dataset can
meaningfully contribute to the learned prompt or selected example set. This property enables the
use of subsampling-based amplification techniques in algorithm design. In RAG, however, only
a sparse subset of documents in the large external corpus are relevant to any given query—most
documents provide no utility.
These two key differences, the lack of reusable prompts and the sparsity of useful data, motivate the
development of our new DP RAG algorithms using Rènyi filter rather than amplification by sampling.
Leveraging Historical QA.As shown in Table 1 and Figure 1, when the relevant documents for
different questions exhibit significant overlap, the quality of answers to later questions degrades. This
occurs because the documents required to answer the queries may exhaust their privacy budgets and
10

Preprint.
are subsequently filtered out from the active set passed to the RAG algorithm. In the extreme case
where a user repeatedly submits the same query, only the first response may retain high quality, while
subsequent answers degrade due to the unavailability of relevant documents.
A potential remedy is to reuse historical answers as auxiliary documents in future queries. This
can be done without incurring any additional privacy cost, owing to the post-processing property of
differential privacy.
7 CONCLUSION
We proposed the first differentially private (DP) framework for retrieval-augmented generation (RAG)
that supports answering multiple queries while protecting a sensitive external dataset. We introduced
two algorithms: MURAG and MURAG-ADAdiffer in how they select documents for each query
under DP guarantees, which have their advantage for different types of question set. Through
comprehensive experiments on various question datasets and three LLMs, we demonstrated that
our methods achieve the utility that outperforms a Non-RAG baseline for answering 100questions
under a realistic budget of ε= 10 . We also showed that MURAG-ADAperforms particularly well
on correlated question sets. We hope our contributions provide a foundation for more practical and
principled privacy-preserving RAG systems.
REFERENCES
Kareem Amin, Alex Bie, Weiwei Kong, Alexey Kurakin, Natalia Ponomareva, Umar Syed, Andreas
Terzis, and Sergei Vassilvitskii. Private prediction for large-scale synthetic text generation.arXiv
preprint arXiv:2407.12108, 2024.
Kareem Amin, Salman Avestimehr, Sara Babakniya, Alex Bie, Weiwei Kong, Natalia Ponomareva,
and Umar Syed. Clustering and median aggregation improve differentially private inference.arXiv
preprint arXiv:2506.04566, 2025.
Maya Anderson, Guy Amit, and Abigail Goldsteen. Is my data in your retrieval database? membership
inference attacks against retrieval augmented generation.arXiv preprint arXiv:2405.20446, 2024.
Borja Balle, Gilles Barthe, and Marco Gaboardi. Privacy amplification by subsampling: Tight
analyses via couplings and divergences.Advances in neural information processing systems, 31,
2018.
Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien, Eric
Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al.
Pythia: A suite for analyzing large language models across training and scaling. InInternational
Conference on Machine Learning, pp. 2397–2430. PMLR, 2023.
Zachary Charles, Arun Ganesh, Ryan McKenna, H Brendan McMahan, Nicole Mitchell, Krishna
Pillutla, and Keith Rush. Fine-tuning large language models with user-level differential privacy.
arXiv preprint arXiv:2407.07737, 2024.
Danqi Chen, Adam Fisch, Jason Weston, and Antoine Bordes. Reading wikipedia to answer open-
domain questions. InProceedings of the 55th Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pp. 1870–1879, 2017.
Yihang Cheng, Lan Zhang, Junyang Wang, Mu Yuan, and Yunhao Yao. Remoterag: A privacy-
preserving llm cloud rag service.arXiv preprint arXiv:2412.12775, 2024.
Haonan Duan, Adam Dziedzic, Nicolas Papernot, and Franziska Boenisch. Flocks of stochastic
parrots: Differentially private prompt learning for large language models.Advances in Neural
Information Processing Systems, 36:76852–76871, 2023.
David Durfee and Ryan M Rogers. Practical differentially private top-k selection with pay-what-you-
get composition.Advances in Neural Information Processing Systems, 32, 2019.
11

Preprint.
Cynthia Dwork, Krishnaram Kenthapadi, Frank McSherry, Ilya Mironov, and Moni Naor. Our data,
ourselves: Privacy via distributed noise generation. InAnnual international conference on the
theory and applications of cryptographic techniques, pp. 486–503. Springer, 2006a.
Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating noise to sensitivity in
private data analysis. InProceedings of the Third Conference on Theory of Cryptography, TCC’06,
pp. 265–284, 2006b.
Cynthia Dwork, Aaron Roth, et al. The algorithmic foundations of differential privacy.Foundations
and trends® in theoretical computer science, 9(3–4):211–407, 2014.
Vitaly Feldman and Tijana Zrnic. Individual privacy accounting via a renyi filter.Advances in Neural
Information Processing Systems, 34:28080–28091, 2021.
Nicolas Grislain. Rag with differential privacy. In2025 IEEE Conference on Artificial Intelligence
(CAI), pp. 847–852. IEEE, 2025.
Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans
trained by a two time-scale update rule converge to a local nash equilibrium.Advances in neural
information processing systems, 30, 2017.
Junyuan Hong, Jiachen T. Wang, Chenhui Zhang, Zhangheng LI, Bo Li, and Zhangyang Wang.
DP-OPT: Make large language model your privacy-preserving prompt engineer. InThe Twelfth
International Conference on Learning Representations, 2024. URL https://openreview.
net/forum?id=Ifz3IgsEPX.
Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot,
Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier,
Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas
Wang, Timothée Lacroix, and William El Sayed. Mistral 7b, 2023. URL https://arxiv.
org/abs/2310.06825.
Changyue Jiang, Xudong Pan, Geng Hong, Chenfu Bao, and Min Yang. Rag-thief: Scalable extraction
of private data from retrieval-augmented generation applications with agent-based attacks.arXiv
preprint arXiv:2411.14110, 2024.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick SH Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. InEMNLP
(1), pp. 6769–6781, 2020.
Tatsuki Koga, Ruihan Wu, and Kamalika Chaudhuri. Privacy-preserving retrieval augmented genera-
tion with differential privacy.arXiv preprint arXiv:2412.04697, 2024.
Antti Koskela, Marlon Tobaben, and Antti Honkela. Individual privacy accounting with gaussian
differential privacy.arXiv preprint arXiv:2209.15596, 2022.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented genera-
tion for knowledge-intensive nlp tasks.Advances in neural information processing systems, 33:
9459–9474, 2020.
Xuechen Li, Florian Tramer, Percy Liang, and Tatsunori Hashimoto. Large language models can be
strong differentially private learners.arXiv preprint arXiv:2110.05679, 2021.
Yunxiang Li, Zihan Li, Kai Zhang, Ruilong Dan, Steve Jiang, and You Zhang. Chatdoctor: A medical
chat model fine-tuned on a large language model meta-ai (llama) using medical domain knowledge.
Cureus, 15(6), 2023.
Yuying Li, Gaoyang Liu, Chen Wang, and Yang Yang. Generating is believing: Membership
inference attacks against retrieval-augmented generation. InICASSP 2025-2025 IEEE International
Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1–5. IEEE, 2025.
12

Preprint.
Mingrui Liu, Sixiao Zhang, and Cheng Long. Mask-based membership inference attacks for retrieval-
augmented generation. InProceedings of the ACM on Web Conference 2025, pp. 2894–2907,
2025.
Ilya Mironov. Rényi differential privacy. In2017 IEEE 30th computer security foundations symposium
(CSF), pp. 263–275. IEEE, 2017.
Ali Naseh, Yuefeng Peng, Anshuman Suri, Harsh Chaudhari, Alina Oprea, and Amir Houmansadr.
Riddle me this! stealthy membership inference for retrieval-augmented generation.arXiv preprint
arXiv:2502.00306, 2025.
Yuefeng Peng, Junda Wang, Hong Yu, and Amir Houmansadr. Data extraction attacks in retrieval-
augmented generation via backdoors.arXiv preprint arXiv:2411.01705, 2024.
Ryan M Rogers, Aaron Roth, Jonathan Ullman, and Salil Vadhan. Privacy odometers and filters:
Pay-as-you-go composition.Advances in Neural Information Processing Systems, 29, 2016.
Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. Membership inference attacks
against machine learning models. In2017 IEEE symposium on security and privacy (SP), pp. 3–18.
IEEE, 2017.
Adam Smith and Abhradeep Thakurta. Fully adaptive composition for gaussian differential privacy.
arXiv preprint arXiv:2210.17520, 2022.
Xinyu Tang, Richard Shin, Huseyin A Inan, Andre Manoel, Fatemehsadat Mireshghallah, Zinan Lin,
Sivakanth Gopi, Janardhan Kulkarni, and Robert Sim. Privacy-preserving in-context learning with
differentially private few-shot generation. InThe Twelfth International Conference on Learning
Representations, 2024. URLhttps://openreview.net/forum?id=oZtt0pRnOl.
Vishnu Vinod, Krishna Pillutla, and Abhradeep Guha Thakurta. Invisibleink: High-utility and
low-cost text generation with differential privacy.arXiv preprint arXiv:2507.02974, 2025.
Yu-Xiang Wang, Borja Balle, and Shiva Prasad Kasiviswanathan. Subsampled rényi differential
privacy and analytical moments accountant. InThe 22nd international conference on artificial
intelligence and statistics, pp. 1226–1235. PMLR, 2019.
Justin Whitehouse, Aaditya Ramdas, Ryan Rogers, and Steven Wu. Fully-adaptive composition in
differential privacy. InInternational conference on machine learning, pp. 36990–37007. PMLR,
2023.
Tong Wu, Ashwinee Panda, Jiachen T. Wang, and Prateek Mittal. Privacy-preserving in-context
learning for large language models. InThe Twelfth International Conference on Learning Repre-
sentations, 2024. URLhttps://openreview.net/forum?id=x4OPJ7lHVU.
Chulin Xie, Zinan Lin, Arturs Backurs, Sivakanth Gopi, Da Yu, Huseyin A Inan, Harsha Nori,
Haotian Jiang, Huishuai Zhang, Yin Tat Lee, Bo Li, and Sergey Yekhanin. Differentially private
synthetic data via foundation model apis 2: Text.ICML, 2024.
Dixi Yao and Tian Li. Private retrieval augmented generation with random projection. InICLR 2025
Workshop on Building Trust in Language Models and Applications.
Da Yu, Saurabh Naik, Arturs Backurs, Sivakanth Gopi, Huseyin A Inan, Gautam Kamath, Janardhan
Kulkarni, Yin Tat Lee, Andre Manoel, Lukas Wutschitz, et al. Differentially private fine-tuning of
language models.arXiv preprint arXiv:2110.06500, 2021.
Shenglai Zeng, Jiankun Zhang, Pengfei He, Yiding Liu, Yue Xing, Han Xu, Jie Ren, Yi Chang,
Shuaiqiang Wang, Dawei Yin, et al. The good and the bad: Exploring privacy issues in retrieval-
augmented generation (rag). InFindings of the Association for Computational Linguistics ACL
2024, pp. 4505–4524, 2024a.
Shenglai Zeng, Jiankun Zhang, Pengfei He, Jie Ren, Tianqi Zheng, Hanqing Lu, Han Xu, Hui Liu,
Yue Xing, and Jiliang Tang. Mitigating the privacy issues in retrieval-augmented generation (rag)
via pure synthetic data.arXiv preprint arXiv:2406.14773, 2024b.
13

Preprint.
Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher
Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language
models.arXiv preprint arXiv:2205.01068, 2022.
Tailai Zhang, Yuxuan Jiang, Ruihan Gong, Pan Zhou, Wen Yin, Xingxing Wei, Lixing Chen, and
Daizong Liu. DEAL: High-efficacy privacy attack on retrieval-augmented generation systems via
LLM optimizer, 2025. URLhttps://openreview.net/forum?id=sx8dtyZT41.
Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. Bertscore: Evaluating
text generation with bert. InInternational Conference on Learning Representations, 2020. URL
https://openreview.net/forum?id=SkeHuCVFDr.
Zexuan Zhong, Zhengxuan Wu, Christopher D Manning, Christopher Potts, and Danqi Chen. Mquake:
Assessing knowledge editing in language models via multi-hop questions. InThe 2023 Conference
on Empirical Methods in Natural Language Processing.
Yuqing Zhu and Yu-Xiang Wang. Poission subsampled rényi differential privacy. In Kamalika
Chaudhuri and Ruslan Salakhutdinov (eds.),Proceedings of the 36th International Conference
on Machine Learning, volume 97 ofProceedings of Machine Learning Research, pp. 7634–7642.
PMLR, 09–15 Jun 2019.
14

Preprint.
A EXTENDEDRELATEDWORK
DIFFERENTIALPRIVACY INLARGELANGUAGEMODELS
Beyond our focus on DP for RAG, differential privacy has also been explored in a variety of LLM
settings, including pre-training and fine-tuning (Charles et al., 2024; Yu et al., 2021; Li et al., 2021),
prompt tuning (Duan et al., 2023; Hong et al., 2024), and in-context learning (Tang et al., 2024; Wu
et al., 2024). These tasks differ structurally and thus require different DP mechanisms. In pre-training
and fine-tuning, the challenge lies in optimizing model parameters while maintaining stability under
DP noise, whereas in RAG, the emphasis is on protecting privacy during inference-time retrieval and
generation. Closer to our setting are DP methods for prompt tuning and in-context learning. Still, the
structural differences between these tasks and RAG lead to distinct algorithmic requirements (see
Section 6 for discussion). Another line of research investigates differentially private synthetic test
generation under varying levels of model access. Vinod et al. (2025); Amin et al. (2025; 2024) focus
on next-token prediction with logits access, while Xie et al. (2024) studies the API-access setting,
which we also include in our comparisons.
INDIVIDUALPRIVACYACCOUNTING ANDPRIVACYFILTERS
Individual privacy accounting tracks the privacy loss of a single data point, often yielding tighter
bounds than worst-case analyses over all neighboring datasets (Dwork et al., 2006b). This perspective
was introduced by Feldman & Zrnic (2021) in the context of Rényi Differential Privacy and later
extended to Gaussian Differential Privacy by Koskela et al. (2022). See Feldman & Zrnic (2021,
Section 1.2) for a detailed overview. Within this framework, privacy filters provide a general
mechanism for adaptively enforcing privacy constraints by halting an algorithm once the cumulative
privacy loss reaches a budget. Individual privacy filters (Feldman & Zrnic, 2021; Koskela et al.,
2022) refine this idea by operating at the granularity of single data points, excluding them from
further computation once their budgets are exhausted. For additional developments and extensions,
see Rogers et al. (2016); Feldman & Zrnic (2021); Koskela et al. (2022); Smith & Thakurta (2022);
Whitehouse et al. (2023).
B DISCUSSION OFSPARSITY INRAG
In retrieval-augmented generation (RAG), relevance isinherently sparse: for any given query, only a
small subset of the external corpus contains the necessary information, while the vast majority are
irrelevant. We illustrate this sparsity with representative examples from the four datasets used in this
paper, as shown in Table 2. For instance, in Natural Questions, the query “what is the story behind
Five Nights at Freddy’s?” is mainly supported by the corresponding Wikipedia article.
Table 2: Example questions drawn from official sources: Natural Questions (visualization page);
TriviaQA (example page); MQuAKE (GitHub repository); and ChatDoctor (Hugging Face page).
Dataset Example Question
Natural Questions what is the story behind 5 nights at freddy’s
TriviaQA Miami Beach in Florida borders which ocean?
MQuAKEWhat country is the birthplace of the sport associated with Hampshire Cricket Board?
Where was the sport associated with Hampshire Cricket Board originated? Which
country is credited with creating the sport associated with Hampshire Cricket Board?
ChatDoctor"instruction": "If you are a doctor, please answer the medical questions based on
the patient’s description." "input": "Doctor, I think I’ve been poisoned. I drank
some ethylene glycol by mistake. "
15

Preprint.
C SUPPLEMENTARYALGORITHMS
This section contains additional algorithms that were excluded from the main body of the paper for
space reasons.
C.1 AUXILIARYALGORITHMS
(Top-K selection)Algorithm 3 selects the top- Kdocuments from the dataset Daccording to
the score function r. If|D|< K , it pads the output with empty strings so that the result
always contains exactly Kelements, as required for the privacy accounting (see Lemma 2).
Algorithm 3:TOP-K(D, K, r)
Input:datasetD, sample sizeK, score functionr
1if|D| ≥Kthen
2D′←top-Kdocuments fromDranked byr ▷assume no ties
3else
4D′←D∪ {""}K−|D|▷pad with empty strings to sizeK
5returnD′
(Poisson Subsampling)Algorithm 4 implements Poisson subsampling: it in-
dependently includes each data point zi∈D in the subsample Swith
probability γ, resulting in a (random) subset whose expected size is γn.
Algorithm 4:POISSONSAMPLING(D, γ)
Input :DatasetD={z 1, . . . , z n}, sampling rateγ∈(0,1)
1S← ∅▷Initialize subsample
2fori←1tondo
3Drawb i∼Bernoulli(γ)
4ifb i= 1then
5S←S∪ {z i}
6returnS
(Token Counting)Algorithm 5 computes the token count vector over a fixed vocabulary:
given a (multi)set of tokens Sand vocabulary V, it iterates over each vocabulary item vj
and counts how many times vjappears in S, returning the resulting count vector ⃗ u∈N|V|.
Algorithm 5:COUNT(S,V)
Input:A (multi)set of tokensS∈ V∗, a vocabularyV={v 1, v2, . . . , v |V|}.
output :Count vector⃗ u∈N|V|, whereu jis the number of timesv jappears inS.
1forj∈ {1,2, . . . ,|V|}do
2u j←P
x∈S1{x=v j}
3return⃗ u
(Exponential Mechanism)Algorithm 6 implements the exponential mechanism: given a candidate
setVand utility scores ujwith sensitivity ∆u, it assigns each candidate vjan unnormalized weight
exp εuj
2∆u
, normalizes these to probabilities, and then samples an output vJfrom the resulting
categorical distribution, ensuringε-DP.
C.2 DIFFERENTIALLYPRIVATERAGFORSINGLE-QUERYQUESTIONANSWERING
(DP-RAG)Algorithm 7 describes our differentially private RAG procedure for single-question
answering: at each decoding step, it compares a baseline token (without retrieval) to votes from m
RAG “voters” over disjoint document subsets, uses a noisy threshold test (via Laplace noise) to decide
whether retrieval can be used, and when it does, privately selects the next token with the exponential
mechanism under a per-token budget ε0, stopping when either an ⟨EOS⟩ token is generated or the
total privacy budget εis exhausted. Algorithm 7 can be seen as a variant of Koga et al. (2024,
Algorithm 2), where the LimitedDomain mechanism (Durfee & Rogers, 2019) is replaced by the
16

Preprint.
Algorithm 6:EXPOMECH(⃗ u,V, ε)
Input:Candidate setV={v 1, v2, . . . , v |V|}, privacy parameterε, utility scores
⃗ u= (u 1, . . . , u |V|)withu j:=u(v j)
output :A selected elementv∈ V
1forj∈ {1,2, . . . ,|V|}do
2w j←exp ε·uj
2∆u
▷unnormalized weight
3Z←P|V|
j=1wj ▷normalizer / partition function
4forj∈ {1,2, . . . ,|V|}do
5p j←w j/Z ▷sampling probability forv j
6SampleJ∼Categorical(p 1, . . . , p |V|)
7returnv J
exponential mechanism in the private token-generation step, yielding a stronger pure-DP guarantee
and simplifying the privacy analysis.
Algorithm 7:DP-RAG(x, D,LLM, ε)
Input:Promptx; document collectionD; language modelLLM; total budgetε.
Require:Per-token budgetε 0; max tokensT max; votersm; docs per voterk; retrieverR;
vote thresholdθ.
Set:ε Lap←ε Expo←ε 0/2; discoveries leftc← ⌊ε/ε 0⌋
1ˆθ←θ+ Lap(2/ε Lap)▷noisy threshold
2Dx←R(x, D;mk); splitD xuniformly intomchunks{D(i)
x}m
i=1.
3fort←1toT maxdo
4b←LLM(x,∅ |y <t)▷baseline token (no RAG)
5fori←1tomdo
6v i←LLM(x, D(i)
x|y<t)
7⃗ u←COUNT({v i}m
i=1,V);s←H[b]▷Algorithm 5
8ifs+ Lap(4/ε Lap)≤ˆθthen
9y t←EXPOMECH(⃗ u,V, ε Expo)▷Algorithm 6
10c←c−1
11 ˆθ←θ+ Lap(2/ε Lap)
12else
13y t←b ▷keep baseline
14ify t=⟨EOS⟩orc= 0then
15return(y 1, . . . , y t)
16return(y 1, . . . , y Tmax)
We now give the privacy guarantee for Algorithm 7.
Lemma 1(Privacy Guarantee for Algorithm 7).Algorithm 7 satisfies ε-DP under add/remove
relationship.
Proof. Notice that Algorithm 7 is an instantiation of AboveThreshold (Dwork et al. (2014, Algo-
rithm 1)) with at most cdiscoveries. It therefore suffices to show that each discovery event (i.e., each
use of the exponential mechanism) satisfiesε 0-DP, wherec=⌊ε/ε 0⌋.
We first verify that the added noise meets the requirements of the stated privacy guarantee, namely
for the threshold perturbation (Line 8 of Algorithm 7) and for the exponential mechanism (Line 9
of Algorithm 7). Without loss of generality, assume the input document set has size larger than
mk. Consider two neighboring datasets DandD′such that |D\D′|+|D′\D| ≤1 . This implies
|Dx\D′
x|+|D′
x\Dx| ≤2 , since the retriever Rranks documents by relevance and selects the
top-mkentries. Replacing a single token in the voting results can change at most one bin count in the
17

Preprint.
histogram by1, so the score function satisfies
|s(D)−s(D′)|=|s(D x)−s(D′
x)| ≤1,
where sis defined in Line 7 of Algorithm 7. Similarly, the utility function has unit sensitivity for
each token, i.e.,
|uj(Dx)−u j(D′
x)| ≤1,∀j∈ {1, . . . ,|V|},
whereu jis thej-th coordinate of⃗ u.
Thus, by Dwork et al. (2014, Theorem 3.23) and adaptive composition, each discovery is ε0-DP. Since
the number of discoveries satisfies c=⌊ε/ε 0⌋, basic composition implies that the entire execution of
Algorithm 7 satisfiesε-DP.
C.3 BASELINEALGORITHMS FORDIFFERENTIALLYPRIVATEMULTI-QUERYRAG
(NAIVE-MULTI-RAG)Algorithm 8 defines a naïve baseline for DP multi-query RAG: it answers
each query qtindependently by invoking the single-query DP-RAG procedure (Algorithm 7) on the
private datasetDwith per-query budgetε q, yielding responses{a t}T
t=1.
Algorithm 8:NAIVE-MULTI-RAG
Input:Private external datasetD, query sequence{q 1, q2, . . . , q T}, per-query budgetε q
1fort= 1, . . . , Tdo
2a t←DP-RAG(q t, D,LLM, ε q)▷Apply Algorithm 7
3return(a 1, a2, . . . , a T)
Lemma 2(Privacy guarantee of Algorithm 8).Algorithm 8 satisfies Tεq-DP under add/remove
relationship.
Proof. By Lemma 1, every call to DP-RAG satisfies εq-DP. Applying basic composition (Dwork
et al., 2014) toTsuch calls introduces an extra factor ofTin the privacy bound.
(SUBSAMPLING-MULTI-RAG)Algorithm 9 defines a baseline for DP multi-query RAG using
subsampling: for each query qt, it first applies Poisson subsampling to the private dataset Dwith
rateγto obtain Dt, then runs the single-query DP-RAG procedure (Algorithm 7) on (qt, Dt)with
per-query budgetε q, producing answers(a 1, . . . , a T).
Algorithm 9:SUBSAMPLING-MULTI-RAG
Input:Private external datasetD, query sequence{q 1, q2, . . . , q T}, per-query privacy
budgetε q, Poisson sampling rateγ
1fort= 1, . . . , Tdo
2D t←POISSONSAMPLING(D, γ)▷Apply Algorithm 4
3a t←DP-RAG(q t, Dt,LLM, ε q)▷Apply Algorithm 7
4return(a 1, a2, . . . , a T)
Lemma 3.Algorithm 9 satisfiesT×log(1 +γ(eεq−1))-DP under add/remove neigh
Proof. Since each call of the DP-RAG satisfies εq-DP, by Balle et al. (2018, Theorem 8), the Poisson
subsampled DP-RAG satisfies log(1 +γ(eεq−1)) -DP. Applying basic composition (Dwork et al.,
2014) toTsuch calls introduces an extra factor ofTin the privacy bound.
DPRIVACYGUARANTEE OFDIFFERENTIALLYPRIVATEMULTI-QUERYRAG
ALGORITHMS
PRIVACYGUARANTEE FORALGORITHM2
Theorem(Restatement of Theorem 2). MURAG-ADA (Algorithm 2) satisfies ε-differential privacy
under the add/remove neighboring relation, provided that the ex-ante individual privacy budget of
everyz∈Dis at mostε.
18

Preprint.
Proof. The proof follows the approach of Feldman & Zrnic (2021, Theorem 4.5). We first bound
the individual privacy loss of the t-th prefix-sum release algorithm, denoted by At. Consider
S,˜S∈ S(z i, n), and without loss of generality assume zi∈S.Conditioned on the trajectory r(t−1)
from the previous t−1 rounds, for any possible output sequence b(q):= (b 1, b2, . . . , b q)withq≤B ,
the only interesting regime is when there exists j∈[q] such that zicontributes to bj. Otherwise, we
have
At(S|r(t−1))d=A t(˜S|r(t−1)).
In the former case, we can perform the decomposition using Bayes’ rule:
logP(A t(S) =b(q))
P(A t(˜S) =b(q))
= logP(A t(S)[j+ 1 :q] =b(j+1:q)|b(j))
P(A t(˜S)[j+ 1 :q] =b(j+1:q) |b(j))
| {z }
(a)
+ log 
P(A t(S)[j] =b j|b(j−1))
P(A t(˜S)[j] =b j|b(j−1))!
| {z }
(b)+ logP(A t(S) =b(j−1))
P(A t(˜S) =b(j−1))
| {z }
(c)
≤εthr
Observe that the bins are disjoint, which implies that the privacy budget consumption is independent
across different data points. Consequently, we have(a) = (c) = 0and(b)≤ε thr.
Next, consider the RAG step. The non-trivial case arises when zi∈A′
t. In this case, by the
composition theorem, the privacy loss of DP-RAG◦TOP-K is bounded above byε RAG.
Moreover, E(zi)constitutes a valid stopping time, as the privacy budget is updated after each
invocation of the algorithms, and ziis only used when its budget remains sufficient. Therefore, by
Feldman & Zrnic (2021, Corollary 3.3), the overall privacy guarantee is given by E(z) , which is
upper bounded byε.
Remark 1.Algorithm 2 employs a fixed, data-independent threshold k(Line 9), rather than a
data-dependent choice such as a DP quantile. If, instead, we were to use a privately released
data-dependent threshold, the resulting selection would become coupled to the data, thereby violating
the assumptions underlying the individual-filter guarantee.
PRIVACYGUARANTEE FORALGORITHM1
Theorem(Restatement of Theorem 1). MURAG satisfies ε-differential privacy if, for every z∈D ,
the ex-ante individual privacy budget is at mostε.
Proof. SinceE(z)≤ε for every z∈D , by an analysis analogous to the proof of Theorem 2, the
claimed privacy guarantee follows directly from Feldman & Zrnic (2021, Corollary 3.3).
E EXPERIMENTALDETAILS
Implementation details of our methods and baseline methods.All four DP algorithms rely on
shared hyperparameters from DPSparseV oteRAG, including the number of retrieved documents k, the
per-token privacy budget εtoken, and the SVT threshold τsvt. Following Koga et al. (2024), we evaluate
each method under a grid of settings with k∈ {30,40,50} ,εtoken∈ {0.5,1.0,2.0} , and τsvt=k/2 .
For MURAG-ADA, the bins for discretizatizing the similarity scores are the bins between 70and100
with the bin size 0.2. For the Non-Private-RAG, we retrieve {1, 3, 5, 10} documents in the context
for each question. We report the best performance for each method over these configurations. For PE,
we adopt the same hyperparameter configuration used for unconditional generation on the PubMed
dataset (Table 14 in Xie et al. (2024)) and generate2000synthetic samples.
Implementation details of the membership inference attack.We implement the Interrogation
Attack (IA; Naseh et al. (2025)) to assess the privacy protection offered by our methods. For each
test document x, the attack generates m= 30 tailored questions. Following the setup in Naseh et al.
19

Preprint.
(2025), we reuse their prompt template (Figure 14 in their paper) and employ GPT-4 to generate the
questions for documents in the ChatDoctor dataset.
The following is an example of one target document and three generated questions.
Document: Patient’s description: I have noticed that my husband (black, male 47 yrs old) has
increasingly began involuntary head movements (as if his right ear is trying to touch his right
shoulder). They are sporadic in nature and he is aware that they happen but is not able to
control them. Can you please help explain what may be causing this? I have notice them
happening more often than not. ### Doctor’s answer: Hi, some people get this increase of
such movements in time of stress, so make him relaxed, you can observe such movements
when he is tensed due to some reasons, meditation could control that. Wish him good health.
Generated Questions and Corresponding Answers:
Q: Did the patient experience sharp abdominal pain on the right side? A: Yes
Q: Is there increased pain on the lower right side during and after sex? A: Yes
Q: Did the patient report pain only during sex, not afterwards? A: No
The construction of query, a combination of generated question and the necessary context, is as follow
I am inquiring about a patient-doctor conversation where the doctor talked "{doctor’s answer
in the target document}" {generated question} Please answer with ’Yes,’ or ’No’.
QA templates.For Trivia, NQ, and MQuAKE QA datasets, we adopt the following template for
Non-RAG QA
Instruction: Give a simple short answer for the question
Question: {question}
Answer:
and the following template for RAG QA
Instruction: Give a simple short answer for the question based on the context
Context: {document1;···; documentm}. Question: {question}
Answer:
For ChatDoctor dataset, we adopt the following template for Non-RAG QA
Instruction: if you are a doctor, please answer the medical questions based on the patient’s
description
Question: {question}
Answer:
and the following template for RAG QA
Instruction: if you are a doctor, please answer the medical questions based on the patient’s
description and the given example
Example: {document1;···; documentm}. Question: {question}
Answer:
Implementation details of the private evolution (PE, (Xie et al., 2024)).Since the external
datasets used in our RAG setup are quite large, applying a synthetic text generation method directly
on these private datasets can be computationally inefficient. To alleviate this overhead—and to give
the baseline a favorable setup—we adopt an approximation: for each QA dataset, we select the top-50
20

Preprint.
document for each question and attain a joint document set. Then we run PE on this smaller but
question-focused subset of the private dataset.
In our experiment, we are using the following prompts for the random API and variation API as
follows:
Random APIFor the ChatDoctor dataset, we adopt the following template:
Instruction: {example} Using a variety of sentence structures, write a dialogue between a
patient describing their condition and a doctor giving suggestions
Answer:
and the following template for Trivia, NQ, and MQuAKE QA datasets
Instruction: Using a variety of sentence structures, for answering the question {question},
write a Wikipedia paragraph
Answer:
In the ChatDoctor random API template, the placeholder example is filled with a sample dialogue
in which a patient describes their condition and a doctor provides suggestions. In contrast, the
random API templates for Trivia, NQ, and MQuAKE use the placeholder question , sampled from
the corresponding question set in a round-robin manner. As the number of API calls exceeds the set
size, the sampling ensures every question is used at least once, guaranteeing full coverage in the PE
generation.
Variation APIFor the ChatDoctor dataset, we adopt the following template:
Instruction: Please rephrase the following tonesentences as a dialogue between a patient
describing their condition and a doctor giving suggestions
Answer:
and the following template for Trivia, NQ, and MQuAKE QA datasets
Instruction: Please rephrase the following sentences as a Wikipedia paragraph
Answer:
21