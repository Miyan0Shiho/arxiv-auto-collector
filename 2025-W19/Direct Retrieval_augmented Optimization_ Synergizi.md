# Direct Retrieval-augmented Optimization: Synergizing Knowledge Selection and Language Models

**Authors**: Zhengliang Shi, Lingyong Yan, Weiwei Sun, Yue Feng, Pengjie Ren, Xinyu Ma, Shuaiqiang Wang, Dawei Yin, Maarten de Rijke, Zhaochun Ren

**Published**: 2025-05-05 23:54:53

**PDF URL**: [http://arxiv.org/pdf/2505.03075v1](http://arxiv.org/pdf/2505.03075v1)

## Abstract
Retrieval-augmented generation (RAG) integrates large language models ( LLM
s) with retrievers to access external knowledge, improving the factuality of
LLM generation in knowledge-grounded tasks. To optimize the RAG performance,
most previous work independently fine-tunes the retriever to adapt to frozen
LLM s or trains the LLMs to use documents retrieved by off-the-shelf
retrievers, lacking end-to-end training supervision. Recent work addresses this
limitation by jointly training these two components but relies on overly
simplifying assumptions of document independence, which has been criticized for
being far from real-world scenarios. Thus, effectively optimizing the overall
RAG performance remains a critical challenge.
  We propose a direct retrieval-augmented optimization framework, named DRO,
that enables end-to-end training of two key components: (i) a generative
knowledge selection model and (ii) an LLM generator. DRO alternates between two
phases: (i) document permutation estimation and (ii) re-weighted maximization,
progressively improving RAG components through a variational approach. In the
estimation step, we treat document permutation as a latent variable and
directly estimate its distribution from the selection model by applying an
importance sampling strategy. In the maximization step, we calibrate the
optimization expectation using importance weights and jointly train the
selection model and LLM generator. Our theoretical analysis reveals that DRO is
analogous to policy-gradient methods in reinforcement learning. Extensive
experiments conducted on five datasets illustrate that DRO outperforms the best
baseline with 5%-15% improvements in EM and F1. We also provide in-depth
experiments to qualitatively analyze the stability, convergence, and variance
of DRO.

## Full Text


<!-- PDF content starts -->

Direct Retrieval-augmented Optimization:
Synergizing Knowledge Selection and Language Models
ZHENGLIANG SHI, Shandong University, China
LINGYONG YAN, Baidu Inc., China
WEIWEI SUN, Carnegie Mellon University, United States
YUE FENG, University of Birmingham, UK
PENGJIE REN, Shandong University, China
XINYU MA, Baidu Inc., China
SHUAIQIANG WANG, Baidu Inc., China
DAWEI YIN, Baidu Inc., China
MAARTEN DE RIJKE, University of Amsterdam, Netherland
ZHAOCHUN REN∗,Leiden University, Netherland
Retrieval-augmented generation ( RAG ) integrates large language models ( LLM s) with retrievers to access
external knowledge, improving the factuality of LLM generation in knowledge-grounded tasks. To optimize
the RAG performance, most previous work independently fine-tunes the retriever to adapt to frozen LLM s or
trains the LLM s to use documents retrieved by off-the-shelf retrievers, lacking end-to-end training supervision.
Recent work addresses this limitation by jointly training these two components but relies on overly simplifying
assumptions of document independence, which has been criticized for being far from real-world scenarios.
Thus, effectively optimizing the overall RAG performance remains a critical challenge.
We propose a direct retrieval-augmented optimization framework, named DRO , that enables end-to-end
training of two key components: (i) a generative knowledge selection model and (ii) an LLM generator. DRO
alternates between two phases: (i) document permutation estimation and (ii) re-weighted maximization,
progressively improving RAG components through a variational approach. In the estimation step, we treat
document permutation as a latent variable and directly estimate its distribution from the selection model by
applying an importance sampling strategy. In the maximization step, we calibrate the optimization expectation
using importance weights and jointly train the selection model and LLM generator. Our theoretical analysis
reveals that DRO is analogous to policy-gradient methods in reinforcement learning. Extensive experiments
conducted on five datasets illustrate that DRO outperforms the best baseline with 5%–15% improvements
in EM and F1. We also provide in-depth experiments to qualitatively analyze the stability, convergence, and
variance of DRO .1
CCS Concepts: •Information systems →Retrieval models and ranking ;Retrieval models and ranking .
∗Corresponding author.
1Code is available on /gtbGitHub.
Authors’ Contact Information: Zhengliang Shi, Shandong University, Qingdao, China, zhengliang.shii@gmail; Lingyong
Yan, Baidu Inc., Beijing, China, lingyongy@gmail.com; Weiwei Sun, Carnegie Mellon University, Pittsburgh, United States,
sunweiwei@gmail.com; Yue Feng, University of Birmingham, Birmingham, UK, y.feng.6@bham.ac.uk; Pengjie Ren, Shandong
University, Qingdap, China, jay.r@outlook.com; Xinyu Ma, Baidu Inc., Beijing, China, xinyuma2016@gmail.com; Shuaiqiang
Wang, Baidu Inc., Beijing, China, shqiang.wang@gmail.com; Dawei Yin, Baidu Inc., Beijing, China, yindawei@acm.org;
Maarten de Rijke, University of Amsterdam, Amsterdam, Netherland, m.derijke@uva.nl; Zhaochun Ren, Leiden University,
Leiden, Netherland, z.ren@liacs.leidenuniv.nl.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee
provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and
the full citation on the first page. Copyrights for third-party components of this work must be honored. For all other uses,
contact the owner/author(s).
Conference acronym ’XX,
©2025 Copyright held by the owner/author(s).
https://doi.org/10.1145/nnnnnnn.nnnnnnn
, Vol. 1, No. 1, Article . Publication date: May 2025.arXiv:2505.03075v1  [cs.IR]  5 May 2025

2 Shi et al. (SDU, Baidu, CMU, UoB, UvA, Leiden)
QueryAnswerEnd-to-end supervisionDocument permutationLLM generatorKnowledge selectionAnswer generationGenerativeselectorRetrieval
Fig. 1. Overview of DRO objective. The selection model directly estimate a document permutation for the
generator to predict an answer, with both components trained jointly.
Additional Key Words and Phrases: Retrieval-augmented generation, List-wise knowledge selection, Impor-
tance sampling, Expectation-Maximization principle
ACM Reference Format:
Zhengliang Shi, Lingyong Yan, Weiwei Sun, Yue Feng, Pengjie Ren, Xinyu Ma, Shuaiqiang Wang, Dawei Yin,
Maarten de Rijke, and Zhaochun Ren. 2025. Direct Retrieval-augmented Optimization: Synergizing Knowledge
Selection and Language Models. In Proceedings of 2025 (Conference acronym ’XX). ACM, New York, NY, USA,
26 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn
1 Introduction
Large language models (LLMs) have shown remarkable text generation abilities; however, they
often provide factually incorrect content [ 4,53,73] due to the hallucination [ 16] or out-of-date
information [9]. To mitigate these limitations, retrieval-augmented generation (RAG) is proposed
to integrate external retrievers with LLMs, which enables the model to access extensive corpora
and retrieve relevant documents for references, thereby enhancing factuality. By integrating the
retriever with LLMs, RAG has shown superior performance in knowledge-intensive tasks such as
question answering [49, 61] and conversational information seeking [5, 24, 68].
Following the most widely used architecture [ 9,11,23],RAG typically includes two components
to answer an input query: (i) knowledge selection , where retrieval and re-ranking models select target
documents, (ii) answer generation , where an LLM generator generates correct answers conditioned
on the selected documents. To enhance coverage and improve answer quality, RAG models often
provide multiple retrieved documents as input to the generator. The interrelationships among
these documents are crucial for final performance [ 15,28,32,72]. We refer to a specific selection of
retrieved documents as a document permutation .
Improving RAG performance. To optimize RAG performance, some studies improve knowledge
accuracy by fine-tuning the retrieval or ranking model with relevance criteria [ 33,38,48]. Others
enhance the robustness of LLMs against irrelevant content through supervised fine-tuning [ 10,71]
or in-context learning [ 49], teaching them to summarize key points from retrieved documents.
However, these approaches optimize either the selection or the generation component separately
while neglecting a dual enhancement, which may lead to sub-optimal overall performance [ 27,45].
To address the above limitation, some recent studies train both the retriever and the LLM
generator [ 23,25,51]. However, to avoid optimizing over the complex permutation distribution of
inter-related documents, most of them simplify the retriever outputs into point-wise documents
rather than the overall permutations [ 72]. Specifically, they first retrieve the top-k documents using
a dense retriever, independently feed each document into downstream LLMs for training, and
train a point-wise dense retriever to assign a higher similarity to the document that leads to better
downstream outcomes. However, this is far from reality, as RAG models often gather useful content
from multiple documents [62,72], which suffers from a pronounced limitation in complex scenarios,
such as multi-hop QA. Besides, training a dense retriever requires the frequent updating of the
, Vol. 1, No. 1, Article . Publication date: May 2025.

Direct Retrieval-augmented Optimization: Synergizing Knowledge Selection and Language Models 3
document index [ 13], which is hard to operate and potentially non-compatible with established
retrieval applications. Thus, a natural question is: How to synergize (i) knowledge selection; and
(ii) answer generation to optimize holistic RAG performance?
In this paper, we propose DRO , a direct retrieval-augmented optimization method that synergizes
(i) a list-wise selection model ( aka, selector) to generate target document permutations, and (ii) an
LLM generator to predict the answer, enabling end-to-end improvement. As shown in Figure 1, the
core idea is to directly treat the document permutation as a latent variable and estimate its distribution
to maximize the log-likelihood of question answering. To achieve this, DRO iteratively alternates
between two steps: (i) document permutation estimation and (ii) re-weighted maximization within
the Expectation-Maximization principle [33].
In the permutation estimation step, we first define an ideal posterior distribution of document
permutation inspired by the classic expectation-maximization algorithm [ 33], introducing a tractable
evidence lower bound (ELBO) for the log-likelihood objective. Considering exactly computing the
posterior distribution is typically impractical, we employ an importance sampling strategy [ 8,21]
to directly estimate the document permutation distribution by sampling from the selection model.
Specifically, for each input query, we first recall relevant documents using an off-the-shelf dense
retriever, filtering documents with no semantic relevance to narrow down the candidate documents.
The generative selector then selects a subset of documents by generating their document identifiers
in an auto-regressive manner (e.g., "[1] > [2] > [3]" ).
In the re-weighted maximization step, we optimize the ELBO constructed in the estimation
step, thereby improving the overall log-likelihood for question answering. We first re-weight the
collected samples using importance weights to calibrate the bias introduced by sampling shifting
from the importance sampling strategy. Then, we jointly optimize the selection model and the
LLM generator by maximizing this re-weighted expectation, where both two components are
trained with end-to-end supervision. By alternating the estimation and maximization steps, DRO
progressively improves the holistic RAG performance.
Theoretical analysis. DRO differs from prior work such as [ 23,25,51] by: (i) enabling end-to-end
optimization for both knowledge selection and answer generation, rather than optimizing individual
processes; and (ii) directly estimating the distribution of document permutations for optimization,
relaxing the assumption of independent top- 𝑘marginalization posed in prior works; (iii) iteratively
aligning of selection and generation models, which achieves a consistent improvement until
convergence. To investigate the advantages of DRO , we provide a theoretical analysis of the
learning objective and optimization process within our framework. We prove that DRO shares
similarities with policy-based reinforcement learning approaches. In DRO , the selection module
improves by reinforcing document permutations that enhance generation performance, while the
generator, in turn, benefits from improved document permutations, creating a synergistic loop that
optimizes the entire RAG process. Additionally, we provide theoretical analysis about the training
convergence and stability of DRO . We reveal that importance sampling with normalized weights
can guarantees variance reduction and non-decreasing ELBO across iterations.
Experiments. We conduct extensive experiments across a wide range of datasets, including Natural
Questions [ 22], HotpotQA [ 69], 2WikiMultihopQA [ 14], MusiQue [ 59], and Wizard-of-Wikipedia [ 5].
The results show that the proposed DRO outperforms best baselines with 5–15% improvement in
EM and F1 metrics. Additionally, the selection model trained using our method achieves an average
precision improvement of 17.78% in identifying target documents. We further conduct fine-grained
analyses to examine the variance, convergence, and stability of the DRO during the training process.
We observe substantial variance decay with increased sampling size, and consistent improvements
, Vol. 1, No. 1, Article . Publication date: May 2025.

4 Shi et al. (SDU, Baidu, CMU, UoB, UvA, Leiden)
(e.g., F1 score) over iterations, indicating stable and convergent optimization. These findings verify
thatDRO achieves not only strong performance but also robust training dynamics across datasets.
Contributions. The main contributions of this paper are: (i) We propose DRO , a direct retrieval-aug-
mented optimization method that treats the document permutation as a latent variable to enable
an end-to-end improvement; (ii) We provide theoretical analysis for the learning objective of the
proposed method and demonstrate its convergence and training stability; and (iii) extensive experi-
ments conducted on five datasets show the improvement of our method, e.g., 5%–15% improvement
compared with state-of-the-art baselines.
2 Related work
2.1 Retrieval-augmented generation
Retrieval-augmented generation (RAG) aims to integrate external knowledge into LLMs, improving
their factuality [ 11]. Given an input query, the first process of RAG is to select relevant knowledge,
which is typically done by retrieval [ 20] or ranking model [ 33]. Subsequently, an LLM generator
incorporates these candidate documents to generate an answer. An active research question is how to
improve the overall RAG performance. Some studies aim to improve the knowledge accuracy [ 6,43],
such as fine-tuning an answer-aware dense retriever [ 44,48] or introducing additional modules for
document filtering [ 63,66]. Other work alternatively enhances the robustness of LLMs to irrelevant
content, enabling LLMs to adaptively extract supporting facts from the retrieved documents [ 70,
76]. However, these methods either optimize the retrieval or the generation process without
dual enhancement, potentially leading to sub-optimal performance [ 27]. Although existing work
proposes the end-to-end training paradigm, they overly simplify a marginalization optimization
through independent top-k approximation [43,72], where they simply feed top-k documents into
downstream LLMs one-by-one and re-score their relevance to optimize the retriever [ 23,27].
This has been criticized far from the practical scenarios as the RAG system typically consumes
multiple documents [ 72], while exhaustively enumerating all possible document permutations is
cost-intensive and typically infeasible in practice. In this work, we propose DRO , which directly
treats the document permutation as a latent variable and estimates its distribution for optimization.
2.2 Knowledge Selection for RAG
In RAG, the knowledge selection process aims to select target documents that can maximize LLM
generation performance [ 12,44]. To achieve this, prior work typically trains point-wise rankers
(e.g., MonoT5 [ 33], BGE [ 65]) on conventional retrieval benchmarks (e.g., MS-MARCO [ 34]) and
separately judges the relevance of each document to the input query. In contrast, our method applies
a generative list-wise selection model [ 55], which selects target documents for the input query by
generating corresponding document identifiers auto-regressively [ 37,38]. Compared with point-
wise ranking, our list-wise selection enables the comparison between multiple documents [ 30,
62], which can be inherently used to estimate the permutation distribution in our framework.
Additionally, unlike previous ranking models trained with semantic relevance criteria [ 41], our
selection model is jointly trained with the generator to maximize end-to-end RAG performance.
2.3 Variational approach for optimization
The variational approach has been widely applied in unsupervised scenarios and optimization
involving latent variables [ 52,56], such as GLEM in graph learning [ 75] and SSDM in speech mod-
eling [ 26]. As a general principle, the variational approach such as the Expectation-Maximization
algorithm (EM) [ 33], alternates between the Expectation step to compute the posterior distribution
, Vol. 1, No. 1, Article . Publication date: May 2025.

Direct Retrieval-augmented Optimization: Synergizing Knowledge Selection and Language Models 5
and the Maximization step to update the model parameter, optimizing the marginalization pro-
gressively. Inspired by this principle, in our work, we treat the document permutation as a latent
variable. Besides, we directly estimate the permutation distribution using a list-wise selection model
through the importance sampling [ 21]. This strategy diverges from the standard EM algorithm by
avoiding the exact computation of the posterior distribution, a process that is impractical due to
the large-scale document permutation space.
3 Preliminaries
3.1 Task Definition
Given an input query 𝑥, the task of retrieval-augmented generation (RAG) typically consists
of two processes: (i) knowledge selection to acquire a set of relevant documents; and (ii) answer
generation to generate a correct answer 𝑦by referring to the acquired documents. In the proposed
DRO , our first process starts by using an off-the-shelf retrieval model (e.g., ColBERT [ 46]) to
recall relevant documents 𝒅={𝑑1,𝑑2,...,𝑑|𝒅|}through semantic matching, thereby filtering the
irrelevant contents to narrow down the candidates. Then, a generative re-ranking model 𝜃𝑠reads
these documents and selects a document permutation 𝒛by generating the corresponding document
identifiers such as [1] > [2] > . . . > [k] . Subsequently, an LLM generator 𝜃𝑔predicts the
answer𝑦based on the input query 𝑥and selected documents 𝒅𝒛, which can be formulated as
𝑝(𝑦|𝑥,𝒅𝒛;𝜃𝑔)=Î|𝑦|
𝑡=1𝑝(𝑦𝑡|𝑦<𝑡,𝑥,𝒅𝒛;𝜃𝑔).
3.2 Generative Knowledge Selection for RAG
The generative selection model identifies the target documents in a list-wise manner, directly
taking the query 𝑥and the candidate documents 𝒅as input, and generates an ordered sequence of
document identifiers auto-regressively:
𝑝(𝒛|𝑥;𝜃𝑠)=Ö𝐾
𝑡=1𝑝(𝒛𝑡|𝒛<𝑡,𝑥,𝒅;𝜃𝑠). (1)
Here, 𝒛={𝒛𝑡;𝑡∈ [𝐾]}represents the document permutation consisting of 𝐾tokens, where
each token corresponds to a document identifier (e.g., [1]or[2]). By mapping the document
permutations back to the original documents, we obtain 𝐾selected documents, denoted as 𝒅𝒛=
{𝒅𝒛𝑡|𝑡∈[𝐾]}. Compared with traditional point-wise ranking, which assigns each individual
document a relevance score to the query, we use this generative selection model for two reasons:
(i) it inherently enables the comparison of multiple candidate documents through the attention
mechanism [ 54,60]; and (ii) it allows the direct modeling of document permutations, ordered lists of
documents that capture their interrelationships by autoregressively generating the docid list [ 31].
4 Direct RAG optimization
In this work, the proposed DRO method improves the holistic performance of RAG by synergizing
(i) knowledge selection and (ii) answer generation processes. To achieve this, DRO enables the
end-to-end training of (i) a list-wise generative selector with parameters 𝜃𝑠to estimate the target
document permutation for the input query 𝑥; and (ii) an LLM generator with parameters 𝜃𝑔to
generate accurate answers. For simplicity, we use 𝜃=(𝜃𝑠,𝜃𝑔)to represent the overall tunable
parameters of DRO throughout the paper. Below, we first derive the learning objective of DRO and
then relate it to how to achieve the end-to-end optimization of the selection and generation model
for such an objective.
, Vol. 1, No. 1, Article . Publication date: May 2025.

6 Shi et al. (SDU, Baidu, CMU, UoB, UvA, Leiden)
w(z) ∝𝑃(𝑦	|	𝑥,𝒅𝒛)SamplingDoc list
E-step: document permutation estimationQuery xDoc[3,1,.4]𝒅𝒛={𝑑"!,𝑑""	,…}Answer yAnswer yM-step: re-weighted maximizationGeneratorTarget permutationGeneratorDocidpermutation z	
[1]
[…]
[n]. .Document retrievalMapping
[1]
[…]
[n]. .Selector
InputDoc[3,1,.4]SelectorQuery xInput+ w(z)∇ℒ$+w(z)∇ℒ𝒢ℒ𝒢ℒ&[3][1]…[4][2][3][1]…[4][2]w(z) re-weighting
Fig. 2. The overall framework for DRO alternates between the (i) E-step: document permutation estimation
(Section 4.2); and (ii) M-step: re-weighted maximization (Section 4.3) to progressively optimize the holistic
RAG performance.
4.1 Deriving the DRO objective
As pointed out by prior work, generating a correct answer grounded on reference knowledge
is identified as one of the most crucial goals of RAG tasks [ 11,23,45]. In DRO , we define the
optimization objective as maximizing the marginal log-likelihood of generating the correct answer
𝑦to the input query 𝑥using external documents 𝒅𝒛. This can be formulated as:
log𝑝(𝑦|𝑥;𝜃)=log∑︁
𝒛𝑝(𝑦,𝒅𝒛|𝑥;𝜃).(2)
Since summing over all possible (𝑦,𝒛)is typically intractable, we employ a variational approach to
construct a tractable lower bound for log𝑝(𝑦|𝑥;𝜃)that we then maximize. Specifically, we introduce
a variational distribution 𝑞(𝒛|𝑥)and apply Jensen’s inequality tologÍ
𝒛𝑝(𝑦,𝒅𝒛|𝑥;𝜃):
log∑︁
𝒛𝑞(𝒛|𝑥)𝑝(𝑦,𝒅𝒛|𝑥;𝜃)
𝑞(𝒛|𝑥)≥∑︁
𝒛𝑞(𝒛|𝑥)log𝑝(𝑦,𝒅𝒛|𝑥;𝜃)
𝑞(𝒛|𝑥)
=E𝒛∼𝑞(𝒛|𝑥)
log𝑝(𝑦,𝒅𝒛|𝑥;𝜃)
𝑞(𝒛|𝑥)
.
|                                {z                                }
Evidence Lower BOund (ELBO(𝑞,𝜃))(3)
Here, we define the E𝒛∼𝑞(𝒛|𝑥)h
log𝑝(𝑦,𝒅𝒛|𝑥;𝜃)
𝑞(𝒛|𝑥)i
as an evidence lower bound (ELBO)forlog𝑝(𝑦|𝑥;𝜃).
Based on Jensen’s inequality, the ELBO(𝑞,𝜃)=log𝑝(𝑦|𝑥;𝜃)if and only if when the𝑝(𝑦,𝒅𝒛|𝑥;𝜃)
𝑞(𝒛|𝑥)≡𝑐,
where𝑐is a constant.
From Eq. (3), we can derive the DRO objective as a progressive optimization process, which
includes: (i) estimating the distribution of document permutation 𝒛to achieve log𝑝(𝑦|𝑥;𝜃)≈ELBO ,
and (ii) maximizing the ELBO to improve log𝑝(𝑦|𝑥;𝜃).
Expectation-Maximization for DRO training. To optimize the DRO objective, we adopt the
expectation-maximization algorithm with an importance sample strategy. In more detail, we start
by demonstrating the condition for the alignment in Eq. (3), which is formulated in the following
lemma.
Lemma 1. ForELBO(𝑞,𝜃)=log𝑝(𝑦|𝑥;𝜃), there exists a variational distribution 𝑞(𝑧|𝑥)such that
𝑞(𝒛|𝑥)=𝑝(𝒛|𝑥,𝑦;𝜃).
, Vol. 1, No. 1, Article . Publication date: May 2025.

Direct Retrieval-augmented Optimization: Synergizing Knowledge Selection and Language Models 7
Proof. This lemma can be proved by considering the case where the importance weight𝑝(𝑦,𝒛|𝑥;𝜃)
𝑞(𝒛|𝑥)
is constant, denoted as 𝑐, across all 𝒛. This is formulated as below:
𝑝(𝑦,𝒛|𝑥;𝜃)
𝑞(𝒛|𝑥)≡𝑐∀𝑥,𝑦
Then, we can sum both sides over 𝒛and obtain:Í
𝒛𝑝(𝑦,𝒛|𝑥;𝜃)=𝑐Í
𝒛𝑞(𝒛|𝑥)≡𝑐. Here𝑞(𝒛|𝑥)is
a probability distribution over 𝒛, i.e., it sums to 1. Therefore, we solve for 𝑞(𝒛|𝑥)as:
𝑞(𝒛|𝑥)=𝑝(𝑦,𝒛|𝑥;𝜃)Í
𝒛𝑝(𝑦,𝒛|𝑥;𝜃)=𝑝(𝒛|𝑥,𝑦;𝜃) (4)
Therefore, the variational distribution 𝑞(𝑧|𝑥)that matches the true posterior 𝑝(𝑧|𝑥,𝑦;𝜃)achieves
the exact ELBO =E𝒛∼𝑞(𝒛|𝑥)h
log𝑝(𝑦,𝒅𝒛|𝑥;𝜃)
𝑞(𝒛|𝑥)i
. □
The Lemma 1 shows an intuitive solution to achieve log𝑝(𝑦|𝑥;𝜃) ≈ ELBO(𝑞,𝜃)by exactly
computing the posterior distribution 𝑝(𝒛|𝑥,𝑦;𝜃)for latent document permutation 𝑧. However, it
is often impractical due to the large-scale permutation space. To address this challenge, we use an
importance sampling strategy, where the 𝑞(𝒛|𝑥)is directly set to 𝑝(𝒛|𝑥;𝜃𝑠). Consequently, the
training of DRO is achieved within an expectation-maximization principle, including:
(i)E-step: document permutation estimation (Section 4.2) to estimate the distribution of document
permutations by sampling from the selection model; and
(ii)M-step: re-weighted maximization (Section 4.3) to jointly optimize the selection model and
generator using the importance-weighted samples.
By iteratively alternating these two step, DRO progressively improves the holistic RAG objective
log𝑝(𝑦|𝑥;𝜃).
4.2 E-step: Document permutation estimation
This step aims to estimate the distribution of document permutations. Specifically, at the 𝑡th
iteration, we first assume 𝑞(𝒛|𝑥)≈𝑝(𝒛|𝑦,𝑥;𝜃𝑡)and transform the ELBO (𝜃,𝜃𝑡)in Eq. (3) into:
E𝒛∼𝑝(𝒛|𝑥,𝑦;𝜃𝑡)
log𝑝(𝑦,𝒛|𝑥;𝜃)−log𝑝(𝒛|𝑥,𝑦;𝜃𝑡)
=E𝒛∼𝑝(𝒛|𝑥,𝑦;𝜃𝑡)[log𝑝(𝑦,𝒛|𝑥;𝜃)]+H 𝑝(𝒛|𝑥,𝑦;𝜃𝑡,(5)
where the 𝒛is ideally sampled from the posterior distribution 𝑝(𝒛|𝑥,𝑦;𝜃𝑡)to compute the
expectation. TheH(𝑝(𝒛|𝑥,𝜃𝑡))indicates the entropy of 𝑝(𝒛|𝑥,𝜃𝑡), which is independent to 𝜃
and can be viewed as a constant. Since the posterior distribution 𝑝(𝒛|𝑦,𝑥;𝜃)is intractable, we
alternatively adopt an importance sampling strategy to directly sample the document permutation
𝒛from our selection model 𝜃𝑠via𝒛∼𝑝(𝒛|𝑥;𝜃𝑡
𝑠). To correct the bias introduced by the sampling
shifting, we also employ an importance weight to calibrate the expectation. Formally, it is presented
as follows:
ELBO(𝜃,𝜃𝑡)=E𝒛∼𝑝(𝒛|𝑥;𝜃𝑡𝑠)[𝑤(𝒛)log𝑝(𝑦,𝒛|𝑥;𝜃)]+constant
𝑤(𝒛)=𝑝(𝒛|𝑥,𝑦;𝜃𝑡)
𝑝(𝒛|𝑥;𝜃𝑡𝑠).(6)
Here we denote the 𝑤(𝒛)as the importance weight . According to Bayes’ theorem, where 𝑝(𝒛|
𝑦,𝑥;𝜃𝑡)∝𝑝(𝒛|𝑥;𝜃𝑡
𝑠)×𝑝(𝑦|𝒅𝒛,𝑥;𝜃𝑡
𝑔), we can then simplify the 𝑤(𝒛)as:
𝑤(𝒛)∝𝑝(𝒛|𝑥;𝜃𝑡
𝑠)×𝑝(𝑦|𝒅𝒛,𝑥;𝜃𝑡
𝑔)
𝑝(𝒛|𝑥;𝜃𝑡𝑠)=𝑝(𝑦|𝑥,𝒅𝒛;𝜃𝑡
𝑔). (7)
Intuitively, the weight 𝑤(𝒛)reflects the utility of a set of documents 𝒅𝒛for the LLM generator
𝜃𝑔in generating ground-truth answers. By sampling from the generative selection model 𝜃𝑠(𝒛∼
, Vol. 1, No. 1, Article . Publication date: May 2025.

8 Shi et al. (SDU, Baidu, CMU, UoB, UvA, Leiden)
Algorithm 1: The algorithm for DRO , which alternates between Estimation and
Maximization steps to progressively improve the overall RAG performance.
Input: selection model 𝜃𝑠; LLM generator 𝜃𝑔; training iteration number 𝑁; training dataT.
for𝑡=1to𝑁do
// Permutation Estimation (E-step)
foreach input query 𝑥in training set do
Sample document permutations: 𝒛∼𝑝(𝒛|𝑥;𝜃𝑡
𝑠)
Compute importance weight: 𝑤(𝒛)=𝑝(𝑦|𝑥,𝒛;𝜃𝑡
𝑔)
// Re-weighting (M-step)
LS(𝑥;𝜃𝑠):=−E𝒛∼𝑝(𝒛|𝑥;𝜃𝑡𝑠)[𝑤(𝒛)log𝑝(𝒛|𝑥;𝜃𝑠)]
LG(𝑥;𝜃𝑔):=−E𝒛∼𝑝(𝒛|𝑥;𝜃𝑡𝑠)
𝑤(𝒛)log𝑝(𝑦|𝑥,𝒅𝒛;𝜃𝑔)
// Re-weighted Maximization (M-step)
𝜃𝑡+1
𝑠=arg max𝜃𝑠E(𝑥,𝑦)∼T[−LS(𝑥;𝜃𝑠)]
𝜃𝑡+1
𝑔=arg max𝜃𝑔E(𝑥,𝑦)∼T
−LG(𝑥;𝜃𝑔)
ifno improvement on validation set then
Stop training Maximization // Early Stop
Output:(𝜃𝑠,𝜃𝑔)
𝑝(𝒛|𝑥;𝜃𝑠)) and calibrating the expectation with importance weights, we directly estimate the
distribution of the latent variable 𝒛(i.e., the document permutation) for unbiased optimization,
without explicitly computing the posterior distribution 𝑝(𝒛|𝑥,𝑦;𝜃𝑡).
4.3 M-step: Re-weighted maximization
After the permutation estimation step, the maximization step aims to update the tunable parameter
𝜃in the RAG system. Formally, the optimization objective is defined as:
𝜃𝑡+1=arg max
𝜃E𝒛∼𝑝(𝒛|𝑥;𝜃𝑡𝑠)[𝑤(𝒛)log𝑝(𝑦,𝒛|𝑥;𝜃)]. (8)
Here, the𝜃=(𝜃𝑠,𝜃𝑔)denotes the tunable parameters of knowledge selection model 𝜃𝑠and LLM
generator𝜃𝑔. Based on Bayes’ theorem, we have 𝑝(𝑦,𝒛|𝑥;𝜃)=𝑝(𝒛|𝑥;𝜃𝑠)·𝑝(𝑦|𝑥,𝒅𝒛;𝜃𝑔). Then,
the ELBO(𝜃,𝜃𝑡)can be rewritten as a decomposition of two parts:
E𝒛∼𝑝(𝒛|𝑥;𝜃𝑡𝑠)
𝑤(𝒛)log𝑝(𝒛|𝑥;𝜃𝑠)
|                    {z                    }
Learning to select+𝑤(𝒛)log𝑝(𝑦|𝑥,𝒅𝒛;𝜃𝑔)
|                        {z                        }
Learning to generate
. (9)
Thus, we derive two loss functions, namely (i) selection optimization LSand (ii) generation
optimizationLG:
LS:=−E𝒛∼𝑝(𝒛|𝑥;𝜃𝑡𝑠)[𝑤(𝒛)log𝑝(𝒛|𝑥;𝜃𝑠)], (10)
LG:=−E𝒛∼𝑝(𝒛|𝑥;𝜃𝑡𝑠)
𝑤(𝒛)log𝑝(𝑦|𝑥,𝒅𝒛;𝜃𝑔)
. (11)
Learning to select. The functionLSoptimizes the selection model 𝜃𝑠for document permutation
generation. SinceLSis weighted by 𝑤(𝒛)and𝑤(𝒛)∝𝑝(𝑦|𝑥,𝒅𝒛;𝜃𝑡
𝑠), the model𝜃𝑠learns to generate
the document permutation that maximizes the end-to-end performance. Based on Eq. (10), the
gradient ofLSwith respect to 𝜃𝑠is:
∇𝜃𝑠LS=−E𝒛∼𝑝(𝒛|𝑥;𝜃𝑡𝑠)
𝑤(𝒛)∇𝜃𝑠log𝑝(𝒛|𝑥;𝜃𝑠)
. (12)
, Vol. 1, No. 1, Article . Publication date: May 2025.

Direct Retrieval-augmented Optimization: Synergizing Knowledge Selection and Language Models 9
Learning to generate. The loss function for 𝜃𝑔is defined asLG:=−E𝒛∼𝑝(𝒛|𝑥;𝜃𝑡𝑠)
𝑤(𝒛)log𝑝(𝑦|𝑥,𝒛;𝜃𝑔)
,
training the LLM generator to understand the selected documents and generate correct answers.
The gradient ofLGwith respect to 𝜃𝑔can be formulated as:
∇𝜃𝑔LG=−E𝒛∼𝑝(𝒛|𝑥;𝜃𝑡𝑠)
𝑤(𝒛)∇𝜃𝑔log𝑝(𝑦|𝑥,𝒅𝒛;𝜃𝑔)
. (13)
4.4 Upshot: Explanation for DRO with pseudo algorithm
To provide a more intuitive explanation of the overall optimization process, we present a pseudo
algorithm in Algorithm 1. In DRO , the document permutation distribution is first estimated using
the selection model 𝜃𝑠(E-step). Subsequently, the model 𝜃𝑠learns to select documents that max-
imize the generation performance while the generator 𝜃𝑠, in turn, further learns to leverage the
selected documents (M-step). These two steps are iteratively alternated, enabling the progressive
improvement of the holistic RAG system.
5 Theoretical analysis
To further interpret DRO , we offer a theoretical analysis of its advantages and training stability,
and prove its convergence.
5.1 What do the selector and generator learn?
The format of the optimization gradient in Eq. (12)and Eq. (13)is similar to classic policy gradient
approaches [ 1,57,74] used in the reinforcement learning (RL). Generally, given an input task 𝑥
and the action 𝜏generated by the policy model 𝜋, the policy gradient objective J(𝜋)to maximize
an reward function 𝑟(𝜏)can be presented as:
J(𝜋)=E𝜏∼𝑝(𝜏|𝑥;𝜋)[𝑟(𝜏)],
∇𝜋J(𝜋)=E𝜏∼𝑝(𝜏|𝑥;𝜋)[𝑟(𝜏)·∇𝜋log𝑝(𝜏|𝑥;𝜋)].(14)
Below, we make several comparisons to relate RL with our DRO .
•𝑤(𝒛) ⇐⇒𝑟(𝜏): In our learning objectives, 𝑤(𝒛)plays a role similar to the reward 𝑟(𝜏)in RL. It
represents the importance of the document permutation 𝒛and serves as a weighting factor to
evaluate the utility of documents to downstream tasks, analogous to how rewards 𝑟(𝜏)shape the
policy model in RL.
•𝑝(𝒛|𝑥;𝜃𝑡
𝑠) ⇐⇒𝑝(𝜏|𝑥;𝜋𝜃): The sampling distribution over permutations 𝑝(𝒛|𝑥;𝜃𝑡
𝑠)reflects
the state of the selection model at iteration 𝑡, similar to how 𝑝(𝜏|𝑥;𝜋𝜃)represents the current
policy distribution over trajectories in RL.
•log𝑝(𝑦,𝒛|𝑥;𝜃) ⇐⇒ log𝑝(𝜏|𝑥;𝜋): The log𝑝(𝑦,𝒛|𝑥;𝜃)in our Eq. 8 is analogous to the
log𝑝(𝜏|𝑥;𝜋)in RL. In both cases, the gradient is updated to increase the likelihood of actions
(or permutations 𝒛) that yield higher rewards (or weights 𝑤(𝒛)).
From the RL perspective, the objective functions in Eq. (9)can be interpreted as a two-agent
system collaborating within the RAG task. In the estimation step, directly sampling a document
permutation from the selection model 𝜃𝑠is essentially a Monte Carlo process, while 𝑤(𝒛)=
𝑝(𝑦|𝑥,𝒅𝒛;𝜃𝑔)in the maximization step serves as the reward. Similar to the RL, where policy
models improve by reinforcing high-reward actions, the selection model 𝜃𝑠improves by selecting
documents that lead to better generation. The generator 𝜃𝑔, in turn, learns to leverage the selected
documents to generate correct answers, creating a synergistic loop that optimizes the entire RAG
performance.
, Vol. 1, No. 1, Article . Publication date: May 2025.

10 Shi et al. (SDU, Baidu, CMU, UoB, UvA, Leiden)
5.2 Impact of the importance sampling
In the permutation estimation step, we apply an importance sampling strategy to directly sample
document permutations from the knowledge selection model (i.e., 𝒛∼𝑝(𝒛|𝑥;𝜃𝑠)) instead of the
posterior distribution 𝑝(𝒛|𝑥,𝑦;𝜃). We analyze its impact on the expectation and variance of the
training objective. For simplicity, we denote log𝑝(𝑦,𝒛|𝑥;𝜃𝑡)in the ELBO as a function 𝑓(𝑦,𝒛).
Expectation is unchanged. In the permutation estimation step, we employ importance sampling
to approximate the posterior distribution 𝑝(𝒛|𝑥,𝑦;𝜃)with samples drawn from the proposal
distribution 𝑝(z|x;𝜃𝑠), which is parameterized by the selection model. To compensate for the
discrepancy between the target and proposal distributions, we apply an importance weight the
weight𝑤(𝒛)=𝑝(𝑦|𝒅𝒛,𝑥;𝜃)adjusts for the different between the target distribution 𝑝(𝒛|𝑥;𝜃𝑡
𝑠)
and the sampling distribution 𝑝(𝒛|𝑥;𝜃𝑠), which can be presented as:
E𝑝(𝒛|𝑥,𝑦;𝜃𝑡)[𝑓(𝒛)]=E𝑝(𝒛|𝑥;𝜃𝑡𝑠)𝑝(𝑦|𝑥,𝒅𝒛;𝜃𝑡)
𝑝(𝒛|𝑥;𝜃𝑡𝑠)𝑓(𝒛)
=E𝑝(𝒛|𝑥;𝜃𝑡𝑠)[𝑤(𝒛)𝑓(𝒛)].(15)
The key property of importance sampling is that it provides an unchanged estimator of the expec-
tation under the true posterior. This fundamental property of importance sampling ensures that
the expectation in the variational optimization objective remains unchanged.
Variance decreases. While importance sampling preserves the expectation, it crucially affects
thevariance of the estimator, which directly impacts the stability and efficiency of gradient-based
training. We begin by formulating the vanilla variance , i.e., the variance before applying importance
sampling.
Before using importance sampling, the vanilla variance Var𝑝(𝒛|𝑥,𝑦;𝜃𝑡)[𝑓(𝑦,𝒛)]is formulated as:
E𝑝(𝒛|𝑥,𝑦;𝜃𝑡)
𝑓(𝑦,𝒛)2
−
E𝑝(𝒛|𝑥;𝜃𝑡𝑠)[𝑓(𝑦,𝒛)]2
. (16)
After using importance sampling, we denote the new variance asVar𝑝(𝒛|𝑥;𝜃𝑡𝑠)[𝑤(𝒛)𝑓(𝑦,𝒛)], which
becomes:
E𝑝(𝒛|𝑥;𝜃𝑡𝑠)
𝑤(𝒛)2𝑓(𝑦,𝒛)2
−
E𝑝(𝒛|𝑥;𝜃𝑡𝑠)[𝑤(𝒛)𝑓(𝑦,𝒛)]2
. (17)
To evaluate the effect of importance sampling on variance, we derive the difference 𝛥Varbetween
the new and old variances.
𝛥Var=Var𝑝(𝒛|𝑥;𝜃𝑡𝑠)[𝑤(𝒛)𝑓(𝑦,𝒛)]−Var𝑝(𝒛|𝑥,𝑦;𝜃𝑡)[𝑓(𝑦,𝒛)]
=E𝑝(𝒛|𝑥,𝑦;𝜃𝑡)
𝑓(𝑦,𝒛)2(𝑤(𝒛)−1)
.(18)
Since𝑤(𝒛)∝𝑝(𝑦|𝑥,𝒅𝒛;𝜃𝑡
𝑔)and the 0≤𝑝(𝑦|𝑥,𝒅𝒛;𝜃𝑡
𝑔)≤1, it follows that 𝛥Var≤0if𝑤(𝒛)is
normalized. This expression reveals a key insight: if 𝑤(𝒛)≤1for all 𝒛, then𝛥Var≤0, indicating
that the variance of the importance-weighted estimator is strictly lower than the original variance.
o further guarantee numerical stability and prevent rare outlier samples with disproportionately
large weights, we apply normalization to 𝑤(𝒛)in practice:
𝑤(𝒛)=𝑤(𝒛)Í
𝒛′𝑤(𝒛′). (19)
This normalization step ensures bounded gradients and facilitates smoother convergence through-
out the DRO training process.
, Vol. 1, No. 1, Article . Publication date: May 2025.

Direct Retrieval-augmented Optimization: Synergizing Knowledge Selection and Language Models 11
Training process gradually stabilizes. From Eq. (16)and Eq. (17), we notice that the 𝑤(𝒛)plays
a key role in shaping the variance, which affects the training stability. Since 𝑤(𝒛)∝𝑝(𝑦|𝑥,𝒅𝒛;𝜃𝑡
𝑔),
we have the following analysis to demonstrate the training stability of our DRO : (i) initially, the
generator𝜃𝑡
𝑔is less optimized, which potentially results in high variance in 𝑝(𝑦|𝑥,𝒛;𝜃𝑡
𝑔); (ii) as
generator𝜃𝑡
𝑔is optimized through the loss function in Eq. (11),𝑝(𝑦|𝑥,𝒛;𝜃𝑡
𝑔)stabilizes, which leads
to a progressively reduced variance.
5.3 Convergence of the optimization process
To prove the convergence of DRO , we show the non-decreasing and upper-bounded property of
log𝑝(𝑦|𝑥,𝜃𝑡)during the training.
We first prove that the log-likelihood log𝑝(𝑦|𝑥,𝜃𝑡+1)is non-decreasing after each training
iteration, which is formulated as: log𝑝(𝑦|𝑥,𝜃𝑡+1)≥log𝑝(𝑦|𝑥,𝜃𝑡). Here𝜃= 𝜃𝑠,𝜃𝑔consists of
the parameters of the selection model 𝜃𝑠and the LLM generator 𝜃𝑔.
Proof. At each iteration 𝑡, we start by estimating the distribution of the latent variable 𝒛, i.e.,
document permutation. Since we apply an importance sampling strategy to directly sample 𝒛from
the selection model 𝜃𝑡
𝑠, the ELBO is transformed as in Eq. (6):
ELBO(𝜃,𝜃𝑡):=E𝒛∼𝑝(𝒛|𝑥;𝜃𝑡𝑠)[𝑤(𝒛)log𝑝(𝑦,𝒛|𝑥;𝜃)].
In the maximization step, we update 𝜃by maximizing the ELBO: 𝜃𝑡+1=arg max𝜃ELBO(𝜃,𝜃𝑡). This
ensures that ELBO(𝜃𝑡+1,𝜃𝑡)≥ELBO(𝜃𝑡,𝜃𝑡). We further observe that the marginal log-likelihood
in Eq. 3 satisfies:
log𝑝(𝑦|𝑥;𝜃𝑡+1)≥ELBO(𝜃𝑡+1,𝜃𝑡)
≥ELBO(𝜃𝑡,𝜃𝑡)=log𝑝(𝑦|𝑥;𝜃𝑡).(20)
Thus, we establish that log𝑝(𝑦|𝑥;𝜃𝑡+1)≥ log𝑝(𝑦|𝑥;𝜃𝑡), demonstrating the non-decreasing
nature of the optimization process. Next, we examine the boundedness of the log-likelihood. Given
that 0≤𝑝(𝑦|𝑥;𝜃)≤1, it follows that−∞≤ log𝑝(𝑦|𝑥;𝜃)≤0. Then, we introduce an existing
theorem from [3] as follows.
Theorem 5.1. Monotone Convergence Theorem: If a sequence{𝑎𝑛}is monotonic (either non-
decreasing or non-increasing) and bounded, then it converges to a finite limit.
Applying Theorem 5.1, the non-decreasing and upper-bounded nature of {log𝑝(𝑦|𝑥;𝜃𝑡)}∞
𝑡=1
ensures that the sequence converges to a finite limit, proving the convergence of the DRO training.
In practice, when the performance on the validation set shows no further improvement, the model
is considered to have converged. □
5.4 Upshot
Below, we summarize three key insights from the above theoretical analysis:
(i)The objective of DRO can be interpreted as optimizing a collaborative two-agent system
within the RAG framework. The selection model 𝜃𝑠improves by identifying document
permutations that enhance the generator’s performance. In turn, the generator 𝜃𝑔learns to
better utilize the selected documents to produce correct answers. This mutual improvement
forms a synergistic loop that jointly optimizes the overall RAG performance.
(ii)The use of importance sampling in DRO preserves the expectation of the learning objective
while reducing variance, particularly when the importance weights are normalized. This
contributes to more stable and efficient training.
, Vol. 1, No. 1, Article . Publication date: May 2025.

12 Shi et al. (SDU, Baidu, CMU, UoB, UvA, Leiden)
(iii)The convergence of DRO is theoretically guaranteed by the monotonic increase of the log-
likelihood and its upper boundedness. In practice, this implies that the model either improves
or maintains its performance over iterations, and training can be safely terminated when
validation performance saturates, ensuring both convergence and training efficiency.
6 Experimental setup
6.1 Datasets
Following previous work [ 23,24,71,72], we conduct experiments on full development set of five
commonly used question-answering benchmarks: Natural Question (NQ) [ 22], HotpotQA [ 69],
MuSiQue [ 59], 2Wikimultihopqa (2WikiQA) [ 14], and Wizard-of-Wikipedia (WoW) [ 5]. Table 1
presents the statistics of these datasets.
6.2 Evaluation metrics
In line with previous studies [ 25,49,64], we use F1andExactly Match (EM) metrics from KILT [ 36]
for evaluation. The F1score is used to measure the token-level overlap between the generated
answer and the ground truth answer, which represents the harmonic mean of precision and recall,
where the recall is determined by considering the number of overlaps with the correct answer
tokens, while precision is determined by considering the number of overlaps with all generated
tokens. The EMmetric checks if the predicted strings exactly match the ground truth. Besides the
end-to-ene evaluation, we also use the Recall@K (K=1,3,5) to evaluate the document re-ranking
performance of our selection model following previous work [ 24,36,45], in which the Recall@K is
set to 1 if and only if the top-K documents contains the ground-truth answer.
6.3 Baselines
We compare the proposed DRO with four categories of baselines based on how they integrate the
knowledge selection process with the answer generation process.
Prompting-based methods. These methods guide LLMs to leverage external knowledge through
prompt learning. We evaluate:
(i)RetGen [47], which interleaves query formulation, retrieval, and answer generation iteratively;
(ii)GenGround [49], which instructs a LLM to generate answers and then revise them using
retrieved documents;
(iii)In-context RAG [40], which truncates the top k retrieved documents as context for the LLM
to generate answers.
For the first two baselines, we use GPT-3.5-turbo as the backbone, following their official im-
plementations. For in-context RAG , we evaluate various LLMs with the same three in-context
examples.
Retrieval Tuning. These methods improve the entire RAG performance by enhancing the retrieval
process. We evaluate:
(i)REPLUG [48] and DPA-RAG [6], which adapt a dense retriever or point-wise ranker to a
frozen LLMs by re-scoring the relevance;
(ii)FLICO [63] and RECOMP , which filter irrelevant content from retrieved documents; and
(iii) Re-ranking Models , which re-rank retrieved documents, passing the top- Kranked results for
LLMs.
We benchmark point-wise (MonoT5 [ 35] and BGE [ 65]) and list-wise rankers ( RankVicuna [ 37]
and RankZephyr [38]).
, Vol. 1, No. 1, Article . Publication date: May 2025.

Direct Retrieval-augmented Optimization: Synergizing Knowledge Selection and Language Models 13
Table 1. Statistics of our experimental datasets, where we provide the amount of training and evaluation
dataset, the average length of input query (word) as well as the retrieval corpus.
Experimental
BenchmarksTraining
Data SizeQuery Length
(Train)Evaluation
Data SizeQuery Length
(Evaluation)Retrieval
Corpus
Nature Question [22] 58,622 9.21 6,489 9.16 Wiki2018
Hotpot QA [69] 90,185 17.85 7,384 15.63 Wiki2018
MusiQue QA [59] 19,938 15.96 2,417 18.11 Wiki2018
2WikiMultiHopQA [14] 167,454 12.74 12,576 11.97 Wiki2018
Wizard-of-Wikipedia [5] 63,734 70.41 3,054 70.25 Wiki2018
LLM Fine-tuning. These methods fine-tune LLMs through instruction tuning, improving LLMs’
ability to utilize retrieved documents. We evaluate:
(i)Vanilla supervised fine-tuning (SFT), which train LLMs by maximizing the answer likelihood
based on query and documents;
(ii)ChatQA [29] and RankRAG [71], which post-train LLMs on diverse knowledge-grounded
tasks;
(iii) RetRobust [70],InstructRAG [64],Self-RAG [2] and RAAT-7B [10], which train LLMs to identify
relevant content from retrieved documents for QA.
End-to-end training. These methods train RAG components with an end-to-end objective. We
benchmark:
(i)Atlas [17], a pre-trained retrieval-augmented LLM;
(ii)RA-DIT [27], which initially trains a generator and subsequently fine-tune a dual-encoder
retriever; and
(iii)DDR-RAG [25], which employs DPO [ 39] (Direct Preference Optimization) to jointly train a
point-wise ranker and an answer generator.
To ensure fairness, we set the size of retrieval documents as 20 for all the baselines and the 𝐾=5,
aligning with the implementation of DRO . Since the code or model checkpoints are unavailable for
some baselines, we mark their incomplete results as “–”. In such cases, we report results from the
original papers but only for reference.
6.4 Implementation details
We use the Llama-3-8B [ 7] as the backbone model for both our ranker and LM generator. We also
alternate it with another model, i.e., Mistral-7B [ 18], to evaluate the generalizability of DRO across
different backbones. Given a query 𝑥, we initially use the off-the-shelf ColBERTv2.0 [ 46] to retrieve
top20documents as input for the selection model 𝜃𝑠, which selects a permutation containing 𝐾=5
documents to the generator 𝜃𝑔. Following previous work [ 20,25,67], the document corpus for the
retrieval is based on the Wikipedia passage dump from Dec. 20, 2018. We set the maximum training
iteration𝑁=5(§ 4.3) and report the performance for each iteration checkpoint for a comprehensive
analysis. The sampling number for document permutation is set to 8. We use the DeepSpeed ZeRO
strategy [ 42] during training, with learning rates of 1𝑒−5and a weight decay coefficient of 0.01.
The training of the DRO can be done within 20 hours with 8 NVIDIA A100-PCIE-80GB GPUs.
7 Experimental results
7.1 Overall performance
Single-hop QA benchmark. We first analyze the performance of the proposed DRO on open-
domain QA and dialogue datasets, including NQ and WoW. The inputs in these datasets are complex,
posing challenges for neural models to comprehend [ 22]. Table 2 presents the results, where we
, Vol. 1, No. 1, Article . Publication date: May 2025.

14 Shi et al. (SDU, Baidu, CMU, UoB, UvA, Leiden)
Table 2. Experimental results on five benchmarks, where we highlight the best results in bold. We also
compute the average EM or F1 across all datasets. Scale indicates the parameter size of the LLM generator.∗
denotes the baselines based on closed-source gpt-3.5 .†and‡indicate significant improvements over best
open-source baselines with p-value < 0.05 and 0.01.
Tasks NQ HotpotQA MuSiQue 2WikiQA WoW Avg.
Metrics Scale EM F1 EM F1 EM F1 EM F1 F1 EM F1
Prompting-based Method
RetGen∗[47] 175B 37.75 39.78 39.04 42.10 17.69 21.04 33.00 39.17 16.97 31.87 31.81
GenGround∗[49] 175B 40.60 42.31 41.27 44.71 20.77 24.36 39.61 42.58 17.76 35.56 34.34
In-context RAG [40]
- w/ Llama3-inst.-70B [7] 70B 39.38 47.26 37.38 39.62 16.43 21.16 37.26 41.46 17.06 32.61 33.31
- w/ Llama2-70B-chat [58] 70B 38.07 44.69 37.14 40.27 16.78 20.11 38.51 41.02 15.75 32.62 32.37
- w/ Mixtral-inst.-8x7B [19] 56B 39.34 46.34 39.63 42.53 16.65 20.73 37.03 38.50 16.66 33.16 32.95
- w/ Llama2-13B-chat [58] 13B 35.27 41.54 35.43 40.37 16.87 18.37 35.47 38.63 13.12 30.76 30.41
Retrieval tuning
REPLUG [66] 65B 28.80 – 32.00 – – – – – – – –
RECOMP [66] 20B 37.47 42.67 38.72 42.72 17.34 24.96 32.17 38.26 15.11 31.43 32.74
DPA-RAG [6] 8B 37.29 44.31 37.15 40.53 18.45 20.36 39.02 39.66 14.73 32.98 31.92
FLICO [63] 7B 35.32 37.24 38.74 39.21 14.53 16.81 29.74 33.05 14.23 29.58 28.11
MonoT5 [35] w/Mistral† 7B 31.78 37.68 32.38 38.76 14.31 19.16 35.96 37.11 14.27 28.61 29.40
RankVicuna [ 37]w/Mistral† 7B 33.78 39.58 34.74 41.25 15.38 21.87 36.72 38.93 14.45 30.16 31.22
RankZephyr [ 38]w/Mistral† 7B 34.77 45.22 36.08 42.77 16.03 21.75 35.07 38.31 14.94 30.49 32.60
BGE-ranker [ 65]w/Mistral† 7B 35.21 40.50 37.61 38.10 16.61 21.37 37.16 38.49 14.77 31.65 30.65
LLM Fine-tuning
RetRobust [70] 13B 37.03 43.82 35.59 40.54 18.11 18.16 38.65 39.11 17.04 32.34 31.73
ChatQA [66] 8B 23.64 34.54 33.40 44.60 16.64 17.05 26.80 31.90 16.22 25.12 28.86
RankRAG [66] 8B – – 35.30 46.70 – – 31.40 36.90 – – –
InstructRAG [64] 8B 35.90 38.21 30.69 34.71 14.94 25.88 35.92 20.01 14.57 29.36 26.68
Vanilla SFT w/Llama3 [58] 8B 35.25 38.46 25.07 32.57 14.35 17.82 30.65 30.43 13.91 26.33 26.64
Vanilla SFT w/Mistral [18] 7B 34.65 37.52 25.75 30.65 13.35 17.97 30.43 30.65 13.83 26.05 26.12
Self-RAG [2] 7B 29.74 31.63 16.30 27.30 9.43 21.50 23.52 27.33 13.24 19.75 24.2
RAAT [10] 7B 33.53 37.85 33.01 31.67 17.08 21.793 29.69 32.68 15.37 28.33 27.87
End-to-end optimization
RA-DIT [27] 65B 35.20 – 39.70 – – – – – – – –
Atlas [17] 11B 26.70 – 34.70 – – – – – – – –
DDR-RAG [25] 8B 40.74 28.76 31.71 40.04 13.54 10.57 35.44 38.40 16.21 30.36 26.80
DRO -Mistral-7B ( Ours ) 7B42.41 51.01 40.37 47.87 21.36 25.32 42.12 43.65 18.31 36.56 37.23
DRO -Llama-3-8B ( Ours ) 8B45.76‡55.42‡42.23†49.27‡20.64†25.97†40.12†44.12‡18.76‡37.19‡38.71‡
observe that our DRO achieves the best performance, such as pushing the EM from 40.74 to
45.76 in NQ dataset (12.32% relative improvements). To investigate the reasons for our superior
performance, we conduct a coarse-to-fine analysis. We begin by identifying some evaluation cases
that incorrectly answered by strong baselines (e.g., InstructRAG) while correctly answered by
our method. Then we compare the outputs from both approaches for these cases. The selection
model, trained collaboratively with the LLM generator in DRO , selects more useful documents for
answer generation. This demonstrates that our selection model learns to understand utility-oriented
document selection criteria in RAG tasks, contributing to holistic improvement.
Multi-hop QA benchmark. We further validate the superiority of the proposed method on multi-
hop QA tasks, which are more challenging since they require the model to aggregate evidence from
multiple documents to derive the answer. For example, based on our statistic, queries in HotpotQA
, Vol. 1, No. 1, Article . Publication date: May 2025.

Direct Retrieval-augmented Optimization: Synergizing Knowledge Selection and Language Models 15
InitialretrievalRankVicunaRankZephyrDRO-Selection
49.71 67.87 74.31 52.22 67.10 73.54 59.48 71.12 75.47 60.89 73.29 79.75 
45.055.065.075.085.0
R@1R@3R@5Recall on NQ
35.50 48.66 54.85 38.16 49.41 54.86 41.81 54.19 58.96 40.93 54.96 60.36 
30.038.046.054.062.0
R@1R@3R@5
35.57 46.03 51.87 36.07 47.88 51.08 36.68 47.31 52.69 42.47 53.57 55.17 
28.036.044.052.060.0
R@1R@3R@5Recall on HotpotQA
Recall on 2Wiki
20.27 31.67 37.37 21.05 31.20 38.05 22.57 32.68 38.35 28.14 37.16 42.79 
15.023.031.039.047.0
R@1R@3R@5Recall on MuSiQue
Fig. 3. Recall@ K(k=1, 3, 5) score of the initial retrieval (Colbertv2.0), two re-ranking baselines (i.e., RankVicuna
and RankZephyer) and our selection model 𝜃𝑠, respectively.
involve at least 2.21 hops, while those in MuSiQue require an average of 2.65 hops. As shown in
Table 2, our DRO substantially outperforms existing baselines, such as achieving a F1 score of
49.27 and an EM score of 42.23 in HotpotQA dataset. The reasons for this improvement are: (1) the
weight𝑤(𝒛)in the selection model’s learning objective severs as a reward function, reinforcing
selecting documents that maximize end-to-end generation performance, and (2) the LLM generator
is optimized synchronously, further enhancing overall effectiveness.
Document selection evaluation. In addition to the end-to-end evaluation presented in Table 2, we
further evaluate the Recall@ Kof our selection models on documents re-ranking task. Following [ 20,
22], we set the Recall@K to 1 if the top-K documents contains the ground-truth answer. As shown
in Figure 3, our selection model improves the recall of the initial retrieval of ColBERTv2.0 by
on average 17.78%, such as pushing the accuracy@1 from 49.71 to 60.89 on NQ dataset. We also
compared with the similar list-wise re-ranking baselines, where we find that the selection model,
trained within DRO , achieves the highest recall. This validates the need to enable end-to-end
supervision for knowled ge selection process.
Table 3. F1 score for checkpoints in each training iteration.
Iteration NQ HotpotQA MuSiQue 2WikiQA Average 𝛥
DRO -Mistral-7B
1 41.56 39.65 20.79 36.5 –
2 47.19 ↑11.9% 43.71↑9.3% 23.26↑10.6% 39.53↑7.7% 9.8%
3 49.98 ↑5.6% 46.03↑5.0% 24.87↑6.9% 41.54↑4.8% 5.5%
4 50.82 ↑1.7% 47.24↑2.6% 25.07↑0.8% 41.75↑0.5% 1.4%
5 50.97 ↑0.3% 47.87↑1.3% 25.32↑1.0% 42.12↑0.9% 0.9%
DRO -Llama-3-8B
1 44.84 41.07 21.43 38.43 –
2 49.11 ↑8.7% 44.73↑8.2% 24.06↑10.9% 41.27↑6.9% 8.7%
3 54.98 ↑10.7% 48.24↑7.3% 25.24↑4.7% 43.02↑4.1% 6.7%
4 55.01 ↑0.1% 49.06↑1.7% 25.94↑2.7% 43.65↑1.4% 1.5%
5 55.42 ↑0.7% 49.27↑0.4% 25.97↑0.1% 44.12↑1.1% 0.6%
, Vol. 1, No. 1, Article . Publication date: May 2025.

16 Shi et al. (SDU, Baidu, CMU, UoB, UvA, Leiden)
1.0E-071.0E-061.0E-051.0E-041.0E-031.0E-02
246810121416Iteration 1Iteration 2Iteration 3
1.0E-071.0E-061.0E-051.0E-041.0E-031.0E-02
246810121416Iteration 1Iteration 2Iteration 3
1.0E-071.0E-061.0E-051.0E-041.0E-031.0E-02
246810121416Iteration 1Iteration 2Iteration 3Sampling sizeSampling size
Sampling sizeVariance on NQ (Mistral-7B)Variance on NQ (Llama-3-8B)
Variance on HotpotQA(Llama-3-8B)
1.0E-071.0E-061.0E-051.0E-041.0E-031.0E-02
246810121416Iteration 1Iteration 2Iteration 3Sampling sizeVariance on HotpotQA(Mistral-7B)
Fig. 4. Variance during the training process of our method (logarithmic scale).
7.2 Training Convergence
As illustrated in Algorithm 1, DRO iteratively optimizes the model parameters 𝜃=(𝜃𝑠,𝜃𝑔)to
improve performance. To analyze convergence, we evaluate the model checkpoint at each training
iteration. Table 3 presents the results on four experimental results. For both DRO -Mistral-7B and
DRO -LLaMA-3-8B, we observe a consistent and substantial increase in F1 scores during the first
three iterations. From iteration 3 to 5, the gains begin to taper off, indicating the model has entered a
convergence phase. Notably, DRO outperforms competitive baselines such as BGE and RankZephyr
after just two iterations, demonstrating the promising performance of our method.
7.3 Training Stability
In our maximization step, we employ an importance weight 𝑤(𝒛)to calibrate the optimization
expectation. Our theoretical analysis in Section 5.2 shows that the training variance decreases as
training progresses, since 𝑤(𝒛)∝𝑝(𝑦|𝑥,𝒅𝒛;𝜃𝑔). To validate this finding, we compute the variance
of sampling from 𝑝(𝑦|𝑥;𝒅𝒛)at each training iteration. We vary the number of samples from 1 to
16 to compute the variance, respectively. See Figure 4. The variance substantially decreases with
the optimization of model 𝜃𝑔during training progress, validating the correctness of our theoretical
analysis from Section 5.2. We also find that increasing the number of samples per iteration can
reduces variance at each training step, which is an straightforward solution to improve the training
stability. Besides, in our experiments, we observe that increasing the number of samples per iteration
reduces variance at each training step, offering a straightforward strategy to improve training
robustness, especially during early-stage training (See Section 8.1 from more details).
Remark 7.1. The reduction of variance during training is a direct consequence of the improved
confidence of the generator 𝜃𝑔in predicting correct answers. As 𝑝(𝑦|𝑥,𝒅𝒛;𝜃𝑔)stabilizes, the importance
weight𝑤(𝒛)becomes less volatile, leading to lower variance and enhanced training stability. This
, Vol. 1, No. 1, Article . Publication date: May 2025.

Direct Retrieval-augmented Optimization: Synergizing Knowledge Selection and Language Models 17
1 2 3 4 5
Training Iteration40.043.847.551.255.0F1 score
NQ
1 2 3 4 5
Training Iteration35.038.842.546.250.0F1 score
HotpotQA
1 2 3 4 5
Training Iteration15.018.822.526.230.0F1 score
MusiQue
1 2 3 4 5
Training Iteration35.037.540.042.545.0F1 score
2wiki
1 2 3 4 5
Training Iteration1214161820F1 score
WoWOurs Ours w/o Selection Ours w/o Generator
Fig. 5. Ablation study on five datasets to demonstrate the effectiveness of training the selection model 𝜃𝑠
and generator 𝜃𝑔.
validates the use of importance weighting in DRO as not only theoretically sound but also practically
stabilizing.
7.4 Ablation studies
The proposed DRO method enables end-to-end optimization of two tightly coupled components:
the selection model 𝜃𝑠and the LLM generator 𝜃𝑔. To quantify the individual contributions of these
components, we conduct an ablation study by isolating the training of each module while freezing
the other. Formally, we compare the full DRO method against two ablated variants:
(i)DRO -w/o Selector : only updates the generator 𝜃𝑔, keeping𝜃𝑠fixed.
(ii)DRO -w/o Generator : only updates the selector 𝜃𝑠, keeping𝜃𝑔fixed.
Figure 5 reports the F1 scores across five datasets (NQ, HotpotQA, MuSiQue, 2WikiMultihopQA,
and WoW) over five training iterations. We observe consistent and notable improvements of the
fullDRO method over both ablated variants. On the NQdataset, the vanilla method achieves an F1
score of 51.01 at iteration 5, while the two variants, i.e., DRO -w/o Selector andDRO -w/o Generator ,
obtain 46.77 and 48.71 respectively, suffering from drops of 4.24 and 2.30 points. Similar patterns
are observed in other datasets:
(i)HotpotQA :DRO achieves 47.87 F1 vs. 44.60 ( DRO -w/o Selector ) and 43.11 ( DRO -w/o Genera-
tor).
(ii)MuSiQue : full model achieves 25.32, outperforming 22.59 and 21.77.
(iii)2Wiki : the full pipeline reaches 43.65 F1, compared to 40.12 and 39.01.
(iv)WoW :DRO improves F1 to 18.31, surpassing 16.47 and 15.73.
These results demonstrate two key findings. First, training the selection model 𝜃𝑠is critical for
learning high-quality document permutations that benefit the generator. Second, optimizing the
generator𝜃𝑔to better utilize selected documents further amplifies performance. The combined
training of both components yields cumulative gains that neither alone can achieve.
Remark 7.2. The ablation study highlights the necessity of joint optimization in DRO . It validates
the design principle that the selector and generator must co-evolve during training to fully realize the
benefits of retrieval-augmented generation.
7.5 Human evaluation
Considering the potential bias of automatic metrics [ 50], we conduct a human evaluation with three
educated individuals assessing the correctness of 100 randomly sampled cases from five benchmarks,
using a three-point scale. Each query is paired with the corresponding golden documents and ground
truth answers from the original datasets, which serve as references for the human evaluators. We
ask at least two annotators to evaluate the same case repeatedly. If there is a discrepancy between
, Vol. 1, No. 1, Article . Publication date: May 2025.

18 Shi et al. (SDU, Baidu, CMU, UoB, UvA, Leiden)
0 1 2 3 4 5
Iteration3035404550F1
43.8242.67F1 Scores (Mistral-7B) on NQ
0 1 2 3 4 5
Iteration253035404550EM
37.0337.47EM Scores (Mistral-7B) on NQSample-14
Sample-12Sample-10
Sample-8Sample-6
Sample-4Sample-2
Sample-1RetRobust
RECOMP
Fig. 6. Performance under different sampling number (§ 8.1).
two annotators, and ask a third annotator to recheck it. The results are presented in Table 4. The
DRO achieves a correctness score of 0.41, while strong open-source baselines only score between
0.32 and 0.37, demonstrating the advantage of our proposed method. The average Kappa statistic
for our human evaluation is 0.751, indicating strong agreement.
Table 4. Human evaluation on 100 randomly sampled cases.
DPA-RAG RetRobust DDR-RAG DRO -Mistral
Correctness 37/100 32/100 33/100 41/100
8 Discussion
8.1 Impact of the Sampling Number
In our training procedure, for each input query, we sample 𝑚=8document permutations 𝑧from
the selection model to construct the ELBO objective via importance sampling. To further examine
the effect of this sampling number on training dynamics and model performance, we vary 𝑚
across{1,2,4,6,8,10,12,14}and evaluate the resulting models under the same experimental setup
described in Table 2. Figure 6 summarizes the results. We derive several observations from the
analysis: (i) Higher sampling improves performance. Overall, increasing the sampling number
𝑚consistently improves the final F1 score. This trend suggests that using more samples provides
better approximation of the expected objective, thereby guiding the optimization process more
accurately. (ii) Sampling number affects convergence speed. Larger values of 𝑚not only lead to
better performance but also accelerate convergence. For example, when 𝑚=14, the model reaches
peak performance within just two training iterations. In contrast, with only 𝑚=1, the model
requires up to four iterations to achieve similar results. This aligns with our variance analysis in
Section 5.2, as increased sampling reduces training variance and facilitates faster optimization.
(iii)Trade-off between cost and effectiveness. While higher values of 𝑚yield better and faster
results, they also incur greater computational overhead. In practice, we find that using 𝑚=4
strikes a good balance, offering competitive performance with reasonable training efficiency. These
findings highlight the importance of sampling strategies in importance-weighted optimization and
suggest practical guidelines for choosing sampling numbers based on resource availability and
convergence requirements.
, Vol. 1, No. 1, Article . Publication date: May 2025.

Direct Retrieval-augmented Optimization: Synergizing Knowledge Selection and Language Models 19
1 2 3 4 5
Training Iteration30.037.545.052.560.0F1
F1 score on NQ
1 2 3 4 5
Training Iteration30.036.242.548.855.0EM
EM score on NQ
1 2 3 4 5
Training Iteration35.038.842.546.250.0F1
F1 score on HotpotQA
1 2 3 4 5
Training Iteration3034384246EM
EM score on HotpotQALlama-3-1B Llama-3B Mistral-7B Llama-3-8B RetRobust RECOMP
Fig. 7. Performance with different parameter sizes of selection model 𝜃𝑠.
8.2 Performance with scaling-down parameter
In our experiment, we follow prior work (e.g., RankVicuna [ 37]) and employ a general-purpose LLM
(e.g., Llama) as the backbone for list-wise selection. Recent studies typically scale up parameter
sizes to explore performance upper bounds. However, since the document selection task involves
long context sequences as input with a concern of increased latency, we investigate a more practical
low bound by scaling down the size of selection model 𝜃𝑠inDRO . Specifically, we implement our
method using Llama-3-8B as the generator and pairing it with smaller LLMs as selection models. We
evaluate the performance under the same conditions as Table 2 and present the results in Figure 7.
We observe that as the parameter size of model 𝜃𝑠increases from 1B to 8B, performance improves
substantially. The Llama-3-8B generator, when paired with the smallest selector (Llama-3-1B),
outperforms strong baselines such as RetRobust, pushing the F1 score from 43.82 to 46.83 on NQ
dataset. Additionally, as an empirical suggestion, training a 3B model (e.g., Llama-3-3B) as the
selection model offers a balanced trade-off between performance and computational cost in our
method.
8.3 Case study
We conduct case studies to intuitively analyze the advantages and disadvantages of DRO . Below,
we first show the system prompt to the document selection model 𝜃𝑠and answer generation model
𝜃𝑔. Then, we present the correctly completed cases and the bad cases, respectively, for comparison.
System prompt for document selection models. In the prompt for the selection model 𝜃𝑠, we
instruct the model to generate a ranked list of useful documents in a generative, auto-regressive
manner, where document identifiers are produced in descending order of utility. Documents that
contain the correct answer are considered more useful and are expected to be ranked higher.
You are RankLLM , an intelligent assistant that can rank passages based on
their relevance and usefulness to the user 's query .
, Vol. 1, No. 1, Article . Publication date: May 2025.

20 Shi et al. (SDU, Baidu, CMU, UoB, UvA, Leiden)
I will provide you with { n_docs } passages . Please rank these passages
based on their usefulness in answering the user 's search query : "{
question }".
A passage 's usefulness is defined as:
1. Relevance to the question .
2. Contains necessary information to help the user .
The passages are listed below , each with a numerical identifier [].
{ docs }
Rank the { n_docs } passages based on their usefulness in descending order .
Use the format [] > [], e.g., [2] > [1]. Only respond with the ranking
results ; do not provide explanations .
Search Query : { question }
Your output :
System prompt for answer generation model. In the prompt for the answer generation model
𝜃𝑔, we provide the documents selected by the selection model as references, and instruct the model
to generate the final answer grounded in the information contained within these documents.
You are an artificial intelligence assistant . You should gives helpful
and precise answers to the user 's questions based on the context . The
context here refer to external passages related to user 's questions .
Please answer the question : "{ question }" using provided passages . Each
passage is indicated by a numerical identifier [].
Here are the related passages for your reference .
{ docs }
Question : { question }
Your answer :
Correctly completed cases. Below, we provide a case in which the input query asks: Which games,
Strange Synergy or Qwirkle, is a card game published by Steve Jackson Games? , with the ground
truth being Strange Synergy . To answer this question, it is essential to retrieve not only detailed
descriptions of Strange Synergy but also background information about Qwirkle for comparative
analysis.
Among the initially retrieved documents, document [1],[2], and [17] are all individually
necessary to resolve the query. Specifically, passages [1]and[2]provide overlapping but essential
descriptions of Strange Synergy, consistently identifying it as a card game published by Steve
Jackson Games. Meanwhile, passage [17] , though it offers necessary contrastive information about
Qwirkle, ranked much lower initially. In DRO , the selection model successfully identifies [1],[2],
and[17] as the top-ranked passages, demonstrating its ability to select the most relevant target
information. Based on these selected passages, the generator compares the publisher information
from [1]and[17] , accurately associating Strange Synergy with Steve Jackson Games as the correct
answer. This example highlights the strength of our DRO in enabling the cooperation between the
selection and generation models.
INPUT QUERY : Which games , Strange Synergy or Qwirkle , is a card game
published by Steve Jackson Games ? ( Ground truth : Strange Synergy )
, Vol. 1, No. 1, Article . Publication date: May 2025.

Direct Retrieval-augmented Optimization: Synergizing Knowledge Selection and Language Models 21
INITIALLY RETRIEVED DOCUMENTS : [0] Title : Strange Synergy . Content :
Strange Synergy Strange Synergy is a card game published by Steve Jackson
Games in which players build a team of super heroes to battle an
opponent 's team . The game supports up to four players ... [1] Title :
Strange Synergy . Content : enemies for three turns in a row , or capture an
opposing team 's flag . Strange Synergy Strange Synergy is a card game
published by Steve Jackson Games in which players build a team of super
heroes to battle an opponent 's team . The game supports up to four players
. However , there are bases and flags for two more players . [2] ... [3]
... [4] ... [5] ... [6] ... [7] ... [8] ... [9] ... [10] ... [11] ...
[12] ... [13] ... [14] ... [15] ... [16]... [17] Title : Qwirkle . Content :
Qwirkle Qwirkle is a tile - based game for 2 to 4 players , designed by
Susan McKinley Ross and published by MindWare . Qwirkle shares some
characteristics with the games Rummikub and Scrabble . It is distributed
in Canada by game and puzzle company , Outset Media . Qwirkle is considered
by MindWare to be its most awarded game of all time . [18] ... [19] ...
OUTPUT OF SELECTION MODEL : [1] > [2] > [17]
OUTPUT OF GENERATOR MODEL : Strange Synergy
Bad case of DRO .We also investigate the bad case of DRO to analyze potential disadvantages
of our method. The Table 5 summarize two key type of errors: incorrect selection and generation
mismatch. First, the incorrect selection indicates that the selector fails to comprehensively include
all documents that are necessary for answer the question. Without such contrastive evidence,
the generator tends to produce incorrect answers, even when relevant documents are present in
the retrieved pool. This suggests the need for diversity-aware selection strategies that explicitly
promote comparative reasoning. Second, the generation mismatch indicates that the generator
model produce a incorrect answer though the ground truth documents have been incorporated into
the context. This often occurs when the input contains partially relevant or distracting content, i.e.,
the noise. These cases highlight the challenge of robust generation.
Table 5. Representative failure cases of DRO. We summarize two typical types of failure, i.e., selection and
generation mismatch, and suggest potential remedies.
Case Type Description Observed Behavior Potential Reme-
dies
Selector misses con-
trastive documentThe selector fails to
comprehensively in-
clude documents (e.g.,
different publishers),
which are necessary for
disambiguation.The generator outputs an
incorrect answer due to
the lack of comparative
evidence.Incorporate multi-
view selection or
contrastive supervi-
sion to enforce di-
versity in document
selection.
Generator is misled by
noisy contentAlthough the selector
provides useful docu-
ments, the generator fail
to generate the correct
answer.The LLM misun-
derstands the input
documents and predicts
incorrect answerIntroduce answer
verification module
chain-of-thought
reasoning process
for LLM before
generating the final
answer.
, Vol. 1, No. 1, Article . Publication date: May 2025.

22 Shi et al. (SDU, Baidu, CMU, UoB, UvA, Leiden)
9 Conclusion
We have presented DRO , a direct retrieval-augmented optimization method for the RAG task that (i)
treats document permutations of the retriever as latent variables, and (ii) enables the co-training of
a list-wise document selection model and an LLM generator through an expectation-maximization
approach. Specifically, DRO alternates between two steps: (1) directly estimating the distribution of
document permutations from the selection model using an importance sampling strategy; and (2)
maximizing the importance-weighted evidence lower bound to jointly train the selection model
and the LLM generator. Through theoretical analysis, we have proven that the learning objectives
inDRO are analogous to policy-gradient reinforcement learning, reinforcing the selection and
generation models with an end-to-end training reward. Extensive experiments conducted on five
datasets have validated the superiority of our DRO . For future work, we aim to extend this approach
to multi-modal RAG scenarios. We also plan to explore the co-training of additional retrieval
modules within DRO , such as a query re-writer.
, Vol. 1, No. 1, Article . Publication date: May 2025.

Direct Retrieval-augmented Optimization: Synergizing Knowledge Selection and Language Models 23
References
[1]Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and
Sara Hooker. 2024. Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback
in LLMs. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024 . Association for Computational Linguistics.
[2]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024. Self-RAG: Learning to Retrieve,
Generate, and Critique through Self-Reflection. In The Twelfth International Conference on Learning Representations,
ICLR 2024, Vienna, Austria, May 7-11, 2024 . OpenReview.net.
[3]John Bibby. 1974. Axiomatisations of the average and a further generalisation of monotonic sequences. Glasgow
Mathematical Journal 1 (1974). https://doi.org/10.1017/S0017089500002135
[4]Shehzaad Dhuliawala, Mojtaba Komeili, Jing Xu, Roberta Raileanu, Xian Li, Asli Celikyilmaz, and Jason Weston. 2023.
Chain-of-verification reduces hallucination in large language models. arXiv preprint arXiv:2309.11495 (2023).
[5]Emily Dinan, Stephen Roller, Kurt Shuster, Angela Fan, Michael Auli, and Jason Weston. 2018. Wizard of wikipedia:
Knowledge-powered conversational agents. arXiv preprint arXiv:1811.01241 (2018).
[6]Guanting Dong, Yutao Zhu, Chenghao Zhang, Zechen Wang, Zhicheng Dou, and Ji-Rong Wen. 2024. Understand what
llm needs: Dual preference alignment for retrieval-augmented generation. arXiv preprint arXiv:2406.18676 (2024).
[7]Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil
Mathur, Alan Schelten, Amy Yang, Angela Fan, et al .2024. The Llama 3 Herd of Models. arXiv preprint arXiv:2407.21783
(2024).
[8] Víctor Elvira and Luca Martino. 2021. Advances in importance sampling. arXiv preprint arXiv:2102.05407 (2021).
[9]Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing Li. 2024. A
survey on rag meeting llms: Towards retrieval-augmented large language models. In Proceedings of the 30th ACM
SIGKDD Conference on Knowledge Discovery and Data Mining .
[10] Feiteng Fang, Yuelin Bai, Shiwen Ni, Min Yang, Xiaojun Chen, and Ruifeng Xu. 2024. Enhancing Noise Robustness of
Retrieval-Augmented Language Models with Adaptive Adversarial Training. In Proceedings of the 62nd Annual Meeting
of the Association for Computational Linguistics .
[11] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen Wang. 2023.
Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997 (2023).
[12] Jiafeng Guo, Yixing Fan, Liang Pang, Liu Yang, Qingyao Ai, Hamed Zamani, Chen Wu, W Bruce Croft, and Xueqi
Cheng. 2020. A deep look into neural ranking models for information retrieval. Information Processing & Management
6 (2020).
[13] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. 2020. Retrieval augmented language
model pre-training. In International conference on machine learning . PMLR.
[14] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020. Constructing A Multi-hop QA Dataset
for Comprehensive Evaluation of Reasoning Steps. In Proceedings of the 28th International Conference on Computational
Linguistics . International Committee on Computational Linguistics.
[15] Sebastian Hofstätter, Jiecao Chen, Karthik Raman, and Hamed Zamani. 2023. Fid-light: Efficient and effective retrieval-
augmented text generation. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development
in Information Retrieval .
[16] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng,
Xiaocheng Feng, Bing Qin, et al .2023. A survey on hallucination in large language models: Principles, taxonomy,
challenges, and open questions. ACM Transactions on Information Systems (2023).
[17] Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand
Joulin, Sebastian Riedel, and Edouard Grave. 2023. Atlas: Few-shot learning with retrieval augmented language models.
Journal of Machine Learning Research 251 (2023).
[18] Albert Qiaochu Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de
Las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, L’elio Renard Lavaud, Marie-Anne
Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023.
Mistral 7B. ArXiv (2023).
[19] Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh
Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al .2024. Mixtral of experts. arXiv preprint
arXiv:2401.04088 (2024).
[20] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick S. H. Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau
Yih. 2020. Dense Passage Retrieval for Open-Domain Question Answering. In Proceedings of the 2020 Conference
on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020 . Association for
Computational Linguistics.
, Vol. 1, No. 1, Article . Publication date: May 2025.

24 Shi et al. (SDU, Baidu, CMU, UoB, UvA, Leiden)
[21] Teun Kloek and Herman K Van Dijk. 1978. Bayesian estimates of equation system parameters: an application of
integration by Monte Carlo. Econometrica: Journal of the Econometric Society (1978).
[22] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle
Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang,
Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural Questions: A Benchmark for Question
Answering Research. Transactions of the Association for Computational Linguistics (2019).
[23] Patrick Lewis, Ethan Perez, Aleksandara Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler,
Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation
for Knowledge-Intensive NLP Tasks. In Proceedings of the 34th International Conference on Neural Information Processing
Systems .
[24] Xiaoxi Li, Zhicheng Dou, Yujia Zhou, and Fangchao Liu. 2024. Corpuslm: Towards a unified language model on corpus
for knowledge-intensive tasks. In Proceedings of the 2024 ACM SIGIR International Conference on Theory of Information
Retrieval .
[25] Xinze Li, Sen Mei, Zhenghao Liu, Yukun Yan, Shuo Wang, Shi Yu, Zheni Zeng, Hao Chen, Ge Yu, Zhiyuan Liu, et al .
2024. RAG-DDR: Optimizing Retrieval-Augmented Generation Using Differentiable Data Rewards. arXiv preprint
arXiv:2410.13509 (2024).
[26] Jiachen Lian, Xuanru Zhou, Zoe Ezzes, Jet Vonk, Brittany Morin, David Baquirin, Zachary Mille, Maria Luisa Gorno
Tempini, and Gopala Krishna Anumanchipalli. 2024. Ssdm: Scalable speech dysfluency modeling. arXiv preprint
arXiv:2408.16221 (2024).
[27] Xi Victoria Lin, Xilun Chen, Mingda Chen, Weijia Shi, Maria Lomeli, Rich James, Pedro Rodriguez, Jacob Kahn, Gergely
Szilvasy, Mike Lewis, et al .2023. Ra-dit: Retrieval-augmented dual instruction tuning. arXiv preprint arXiv:2310.01352
(2023).
[28] Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost
in the middle: How language models use long contexts. Transactions of the Association for Computational Linguistics
(2024).
[29] Zihan Liu, Wei Ping, Rajarshi Roy, Peng Xu, Chankyu Lee, Mohammad Shoeybi, and Bryan Catanzaro. 2024. Chatqa:
Surpassing gpt-4 on conversational qa and rag. arXiv preprint arXiv:2401.10225 (2024).
[30] Xueguang Ma, Xinyu Zhang, Ronak Pradeep, and Jimmy Lin. 2023. Zero-shot listwise document reranking with a
large language model. arXiv preprint arXiv:2305.02156 (2023).
[31] Sewon Min, Kenton Lee, Ming-Wei Chang, Kristina Toutanova, and Hannaneh Hajishirzi. 2021. Joint Passage Ranking
for Diverse Multi-Answer Retrieval. ArXiv (2021).
[32] Sewon Min, Julian Michael, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2020. AmbigQA: Answering ambiguous
open-domain questions. arXiv preprint arXiv:2004.10645 (2020).
[33] Todd K Moon. 1996. The expectation-maximization algorithm. IEEE Signal processing magazine 6 (1996).
[34] Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and Li Deng. 2016. MS
MARCO: A Human Generated MAchine Reading COmprehension Dataset. In Proceedings of the Workshop on Cognitive
Computation: Integrating neural and symbolic approaches 2016 co-located with the 30th Annual Conference on Neural
Information Processing Systems (NIPS 2016), Barcelona, Spain, December 9, 2016 . CEUR-WS.org.
[35] Rodrigo Nogueira, Zhiying Jiang, Ronak Pradeep, and Jimmy Lin. 2020. Document Ranking with a Pretrained
Sequence-to-Sequence Model. In Findings of the Association for Computational Linguistics: EMNLP 2020 . Association for
Computational Linguistics.
[36] Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani, Nicola De Cao, James Thorne, Yacine
Jernite, Vladimir Karpukhin, Jean Maillard, Vassilis Plachouras, Tim Rocktäschel, and Sebastian Riedel. 2021. KILT:
a Benchmark for Knowledge Intensive Language Tasks. In Proceedings of the 2021 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language Technologies . Association for Computational
Linguistics.
[37] Ronak Pradeep, Sahel Sharifymoghaddam, and Jimmy Lin. 2023. Rankvicuna: Zero-shot listwise document reranking
with open-source large language models. arXiv preprint arXiv:2309.15088 (2023).
[38] Ronak Pradeep, Sahel Sharifymoghaddam, and Jimmy Lin. 2023. RankZephyr: Effective and Robust Zero-Shot Listwise
Reranking is a Breeze! arXiv preprint arXiv:2312.02724 (2023).
[39] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2024. Direct
preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing
Systems (2024).
[40] Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham.
2023. In-Context Retrieval-Augmented Language Models. Transactions of the Association for Computational Linguistics
(2023).
, Vol. 1, No. 1, Article . Publication date: May 2025.

Direct Retrieval-augmented Optimization: Synergizing Knowledge Selection and Language Models 25
[41] Jinfeng Rao, Linqing Liu, Yi Tay, Wei Yang, Peng Shi, and Jimmy Lin. 2019. Bridging the gap between relevance
matching and semantic matching for short text similarity modeling. In Proceedings of the 2019 Conference on Empirical
Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing
(EMNLP-IJCNLP) .
[42] Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. 2020. DeepSpeed: System Optimizations Enable
Training Deep Learning Models with Over 100 Billion Parameters. In Proceedings of the 26th ACM SIGKDD International
Conference on Knowledge Discovery & Data Mining . Association for Computing Machinery.
[43] Devendra Sachan, Mostofa Patwary, Mohammad Shoeybi, Neel Kant, Wei Ping, William L. Hamilton, and Bryan
Catanzaro. 2021. End-to-End Training of Neural Retrievers for Open-Domain Question Answering. In Proceedings of
the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on
Natural Language Processing (Volume 1: Long Papers) . Association for Computational Linguistics.
[44] Devendra Singh Sachan, Mike Lewis, Mandar Joshi, Armen Aghajanyan, Wen-tau Yih, Joelle Pineau, and Luke
Zettlemoyer. 2022. Improving passage retrieval with zero-shot question generation. arXiv preprint arXiv:2204.07496
(2022).
[45] Alireza Salemi and Hamed Zamani. 2024. Evaluating Retrieval Quality in Retrieval-Augmented Generation. In
Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval .
Association for Computing Machinery.
[46] Keshav Santhanam, O. Khattab, Jon Saad-Falcon, Christopher Potts, and Matei A. Zaharia. 2021. ColBERTv2: Effective
and Efficient Retrieval via Lightweight Late Interaction. In North American Chapter of the Association for Computational
Linguistics .
[47] Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, and Weizhu Chen. 2023. Enhancing Retrieval-
Augmented Large Language Models with Iterative Retrieval-Generation Synergy. In Findings of the Association for
Computational Linguistics: EMNLP 2023 .
[48] Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, and Wen-tau
Yih. 2023. Replug: Retrieval-augmented black-box language models. arXiv preprint arXiv:2301.12652 (2023).
[49] Zhengliang Shi, Weiwei Sun, Shen Gao, Pengjie Ren, Zhumin Chen, and Zhaochun Ren. 2024. Generate-then-ground
in retrieval-augmented generation for multi-hop question answering. arXiv preprint arXiv:2406.14891 (2024).
[50] Zhengliang Shi, Weiwei Sun, Shuo Zhang, Zhen Zhang, Pengjie Ren, and Zhaochun Ren. 2023. RADE: Reference-
Assisted Dialogue Evaluation for Open-Domain Dialogue. ArXiv (2023).
[51] Devendra Singh, Siva Reddy, Will Hamilton, Chris Dyer, and Dani Yogatama. 2021. End-to-end training of multi-
document reader and retriever for open-domain question answering. Advances in Neural Information Processing
Systems .
[52] Alessandro Sordoni, Eric Yuan, Marc-Alexandre Côté, Matheus Pereira, Adam Trischler, Ziang Xiao, Arian Hosseini,
Friederike Niedtner, and Nicolas Le Roux. 2024. Joint prompt optimization of stacked llms using variational inference.
Advances in Neural Information Processing Systems (2024).
[53] Weihang Su, Yichen Tang, Qingyao Ai, Changyue Wang, Zhijing Wu, and Yiqun Liu. 2024. Mitigating entity-level
hallucination in large language models. In Proceedings of the 2024 Annual International ACM SIGIR Conference on
Research and Development in Information Retrieval in the Asia Pacific Region .
[54] Weiwei Sun, Pengjie Ren, and Zhaochun Ren. 2023. Generative Knowledge Selection for Knowledge-Grounded
Dialogues. In EACL Findings .
[55] Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, and Zhaochun Ren.
2023. Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents. In Proceedings of the
2023 Conference on Empirical Methods in Natural Language Processing . Association for Computational Linguistics.
[56] Zhiqing Sun and Yiming Yang. 2020. An em approach to non-autoregressive conditional sequence generation. In
International Conference on Machine Learning . PMLR.
[57] Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. 1999. Policy gradient methods for reinforce-
ment learning with function approximation. Advances in neural information processing systems (1999).
[58] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya
Batra, Prajjwal Bhargava, Shruti Bhosale, et al .2023. Llama 2: Open foundation and fine-tuned chat models. arXiv
preprint arXiv:2307.09288 (2023).
[59] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2022. MuSiQue: Multihop Questions
via Single-hop Question Composition. In Transactions of the Association for Computational Linguistics: TACL .
[60] A Vaswani. 2017. Attention is all you need. Advances in Neural Information Processing Systems (2017).
[61] Shuting Wang, Jiongnan Liu, Shiren Song, Jiehan Cheng, Yuqi Fu, Peidong Guo, Kun Fang, Yutao Zhu, and Zhicheng
Dou. 2024. Domainrag: A chinese benchmark for evaluating domain-specific retrieval-augmented generation. arXiv
preprint arXiv:2406.05654 (2024).
, Vol. 1, No. 1, Article . Publication date: May 2025.

26 Shi et al. (SDU, Baidu, CMU, UoB, UvA, Leiden)
[62] Shuting Wang, Xin Yu, Mang Wang, Weipeng Chen, Yutao Zhu, and Zhicheng Dou. 2024. Richrag: Crafting rich
responses for multi-faceted queries in retrieval-augmented generation. arXiv preprint arXiv:2406.12566 (2024).
[63] Zhiruo Wang, Jun Araki, Zhengbao Jiang, Md Rizwan Parvez, and Graham Neubig. 2023. Learning to filter context for
retrieval-augmented generation. arXiv preprint arXiv:2311.08377 (2023).
[64] Zhepei Wei, Wei-Lin Chen, and Yu Meng. 2024. InstructRAG: Instructing Retrieval-Augmented Generation via
Self-Synthesized Rationales. (2024).
[65] Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighoff. 2023. C-Pack: Packaged Resources To Advance
General Chinese Embedding. arXiv:2309.07597 [cs.CL]
[66] Fangyuan Xu, Weijia Shi, and Eunsol Choi. 2023. Recomp: Improving retrieval-augmented lms with compression and
selective augmentation. arXiv preprint arXiv:2310.04408 (2023).
[67] Shicheng Xu, Liang Pang, Huawei Shen, Xueqi Cheng, and Tat-Seng Chua. 2024. Search-in-the-Chain: Towards
Accurate, Credible and Traceable Large Language Models for Knowledge-intensive Tasks. In WWW .
[68] Diji Yang, Jinmeng Rao, Kezhen Chen, Xiaoyuan Guo, Yawen Zhang, Jie Yang, and Yi Zhang. 2024. Im-rag: Multi-round
retrieval-augmented generation through learning inner monologues. In Proceedings of the 47th International ACM
SIGIR Conference on Research and Development in Information Retrieval .
[69] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D.
Manning. 2018. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. In Conference on
Empirical Methods in Natural Language Processing (EMNLP) .
[70] Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan Berant. 2023. Making retrieval-augmented language models robust
to irrelevant context. arXiv preprint arXiv:2310.01558 (2023).
[71] Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan You, Chao Zhang, Mohammad Shoeybi, and Bryan Catanzaro.
2024. Rankrag: Unifying context ranking with retrieval-augmented generation in llms. arXiv preprint arXiv:2407.02485
(2024).
[72] Hamed Zamani and Michael Bendersky. 2024. Stochastic rag: End-to-end retrieval-augmented generation through ex-
pected utility maximization. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development
in Information Retrieval .
[73] Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang, Yulong
Chen, et al .2023. Siren’s song in the AI ocean: a survey on hallucination in large language models. arXiv preprint
arXiv:2309.01219 (2023).
[74] Hanyang Zhao, Wenpin Tang, and David Yao. 2024. Policy optimization for continuous reinforcement learning.
Advances in Neural Information Processing Systems (2024).
[75] Jianan Zhao, Meng Qu, Chaozhuo Li, Hao Yan, Qian Liu, Rui Li, Xing Xie, and Jian Tang. 2022. Learning on large-scale
text-attributed graphs via variational inference. arXiv preprint arXiv:2210.14709 (2022).
[76] Junda Zhu, Lingyong Yan, Haibo Shi, Dawei Yin, and Lei Sha. 2024. ATM: Adversarial Tuning Multi-agent System
Makes a Robust Retrieval-Augmented Generator. arXiv preprint arXiv:2405.18111 (2024).
, Vol. 1, No. 1, Article . Publication date: May 2025.