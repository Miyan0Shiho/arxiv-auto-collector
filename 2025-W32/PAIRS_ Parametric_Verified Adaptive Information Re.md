# PAIRS: Parametric-Verified Adaptive Information Retrieval and Selection for Efficient RAG

**Authors**: Wang Chen, Guanqiang Qi, Weikang Li, Yang Li, Deguo Xia, Jizhou Huang

**Published**: 2025-08-06 03:33:01

**PDF URL**: [http://arxiv.org/pdf/2508.04057v1](http://arxiv.org/pdf/2508.04057v1)

## Abstract
Retrieval-Augmented Generation (RAG) has become a cornerstone technique for
enhancing large language models (LLMs) with external knowledge. However,
current RAG systems face two critical limitations: (1) they inefficiently
retrieve information for every query, including simple questions that could be
resolved using the LLM's parametric knowledge alone, and (2) they risk
retrieving irrelevant documents when queries contain sparse information
signals. To address these gaps, we introduce Parametric-verified Adaptive
Information Retrieval and Selection (PAIRS), a training-free framework that
integrates parametric and retrieved knowledge to adaptively determine whether
to retrieve and how to select external information. Specifically, PAIRS employs
a dual-path generation mechanism: First, the LLM produces both a direct answer
and a context-augmented answer using self-generated pseudo-context. When these
outputs converge, PAIRS bypasses external retrieval entirely, dramatically
improving the RAG system's efficiency. For divergent cases, PAIRS activates a
dual-path retrieval (DPR) process guided by both the original query and
self-generated contextual signals, followed by an Adaptive Information
Selection (AIS) module that filters documents through weighted similarity to
both sources. This simple yet effective approach can not only enhance
efficiency by eliminating unnecessary retrievals but also improve accuracy
through contextually guided retrieval and adaptive information selection.
Experimental results on six question-answering (QA) benchmarks show that PAIRS
reduces retrieval costs by around 25% (triggering for only 75% of queries)
while still improving accuracy-achieving +1.1% EM and +1.0% F1 over prior
baselines on average.

## Full Text


<!-- PDF content starts -->

PAIRS: Parametric–Verified Adaptive Information Retrieval and Selection for
Efficient RAG
Wang Chen1,2, Guanqiang Qi1, Weikang Li3, Yang Li1*, Deguo Xia1, Jizhou Huang1
1Baidu Inc
2The University of Hong Kong
3Peking University
{chenwang05, qiguanqiang, liyang164, xiadeguo, huangjizhou01 }@baidu.com, wavejkd@pku.edu.cn
Abstract
Retrieval–Augmented Generation (RAG) has become a cor-
nerstone technique for enhancing large language models
(LLMs) with external knowledge. However, current RAG
systems face two critical limitations: (1) they inefficiently re-
trieve information for every query, including simple questions
that could be resolved using the LLM’s parametric knowledge
alone, and (2) they risk retrieving irrelevant documents when
queries contain sparse information signals. To address these
gaps, we introduce Parametric–verified Adaptive Information
Retrieval and Selection ( PAIRS ), a training-free framework
that integrates parametric and retrieved knowledge to adap-
tively determine whether to retrieve and how to select external
information. Specifically, PAIRS employs a dual-path gener-
ation mechanism: First, the LLM produces both a direct an-
swer and a context–augmented answer using self-generated
pseudo-context. When these outputs converge, PAIRS by-
passes external retrieval entirely, dramatically improving the
RAG system’s efficiency. For divergent cases, PAIRS acti-
vates a dual-path retrieval ( DPR ) process guided by both
the original query and self-generated contextual signals, fol-
lowed by an Adaptive Information Selection ( AIS) module
that filters documents through weighted similarity to both
sources. This simple yet effective approach can not only en-
hance efficiency by eliminating unnecessary retrievals but
also improve accuracy through contextually guided retrieval
and adaptive information selection. Experimental results on
six question–answering (QA) benchmarks show that PAIRS
reduces retrieval costs by around 25% (triggering for only
75% of queries) while still improving accuracy—achieving
+1.1% EM and +1.0% F1 over prior baselines on average.
Introduction
Large language models (LLMs) have dramatically advanced
natural language processing (NLP), achieving comparable
or even better performance than human beings (Touvron
et al. 2023; Achiam et al. 2023; Guo et al. 2025; Yang et al.
2025). However, previous studies (Ji et al. 2023) found that
LLMs can only answer questions or accomplish tasks by
leveraging their parametric knowledge, which cannot update
the latest information nor private datasets. To address this is-
sue, retrieval–augmented generation (RAG) framework was
designed to enhance the generation performance of LLMs
using the retrieved information (Lewis et al. 2020; Guu et al.
*Corresponding author2020; Karpukhin et al. 2020). The effectiveness of RAG has
been validated across various NLP tasks, achieving impres-
sive improvement compared with pure LLM systems (Ram
et al. 2023; Gao et al. 2023b).
Despite its widespread adoption, conventional RAG sys-
tems suffer from two critical shortcomings. First, they inef-
ficiently trigger retrieval for every query, including straight-
forward questions that could be resolved solely using the
LLM’s internal parametric knowledge, resulting in unneces-
sary computational latency and resource consumption. Sec-
ond, they exhibit vulnerability to sparse queries—concise
user inputs with limited contextual signals—which often
yield irrelevant or low-quality retrievals due to inadequate
semantic cues, ultimately degrading answer accuracy and re-
liability (Wang, Yang, and Wei 2023; Gao et al. 2023a).
Existing approaches address these challenges through
two primary strategies: query augmentation (e.g., appending
LLM–generated pseudo-documents or external data to en-
rich sparse queries) (Jagerman et al. 2023; Buss et al. 2023;
Jeong et al. 2024) and retrieval optimization (e.g., adopting
reinforcement learning or rerankers to refine document re-
trieval or selection) (Asai et al. 2024; Jin et al. 2025; Chang
et al. 2025). However, most of these methods may suffer
from high computational costs and overlook LLMs’ inher-
ent capability to resolve simple queries without retrieval.
Recently, the emerging LLMs contain billions of or even
more than 1 trillion parameters and are trained on massive
datasets, endowing them with rich parametric knowledge
that enables them to answer many simple queries without ex-
ternal retrieval. To explore how this capability can enhance
both the efficiency and effectiveness of RAG systems, we
propose PAIRS (Parametric–verified Adaptive Information
Retrieval and Selection), a training-free , simple yet effec-
tive framework that enhances RAG by adaptively integrat-
ing parametric and retrieved knowledge (as shown in Fig.
1). The key contributions of this work are summarized as
follows:
•Parametric–verified dual-path mechanism : PAIRS in-
troduces a novel parametric verification approach, where
the LLM generates both a direct answer and a pseudo-
context–augmented answer. If the two answers agree,
it confidently bypasses retrieval, significantly improving
efficiency.
•DPR-AIS module : For divergent cases, we propose a
1arXiv:2508.04057v1  [cs.CL]  6 Aug 2025

𝑠(𝑑)=𝑠𝑑,𝑞+𝑠(𝑑,𝑝)
dQueryContextqpAdaptiveInformation Selection (AIS)
Query
ContextDual-Path Retrieval (DPR)
Retrieve
…Corpus
LLM
Query+AnsGeneration(c) DPR-AISmodule(b) Parametric-verifieddual-generation mechanism(a) PAIRSAgree?YesNoAns
LLMQueryContextAns2Ans1Dual-generationmechanismParametricverificationDPR-AISQuery
LLMAns
Not to retrieveTo retrieve
AISFigure 1: Illustration of the PAIRS framework. (a) PAIRS can adaptively determine whether to retrieve and how to select external
information. (b) PAIRS applies a dual-generation mechanism to verify the correctness of answers. (c) PAIRS adaptively selects
effective information based on the dual-path retrieval for the final generation.
Dual-Path Retrieval (DPR) strategy that retrieves docu-
ments using both the query and the generated pseudo-
context. These are then filtered via the Adaptive Infor-
mation Selection (AIS) module, which scores each doc-
ument by weighted similarity to both sources.
•Efficiency with accuracy : Experimental results demon-
strate that PAIRS can reduce reliance on external re-
trieval, triggering it for only approximately 75% of
queries, while still achieving higher EM (+1.1%) and
F1 (+1.0%) scores than prior baselines across six QA
datasets on average.
•Flexible integration : The framework is modular and can
be seamlessly combined with other enhancements, such
as reranking models, to further boost performance. For
example, DPR-AIS-rerank achieves state-of-the-art re-
sults, outperforming strong reranking baselines by 2.3%
EM and 2.5% F1 on average across six datasets.
Related work
In this section, we review the related works in recent years
and further discuss the research gaps. Specifically, we pri-
mally focus on two folds: (1) how to augment a query to
enhance retrievals and (2) how to retrieve and select relevant
information.
Query augmentation
A query from users may be concise and contain relatively
sparse information, which can result in suboptimal retrievals
and poor answers. To address this issue, many studies have
proposed to augment the query by appending additional
terms extracted from retrieved documents (Lavrenko and
Croft 2017; Lv and Zhai 2009) or generated by neural mod-
els (Zheng et al. 2020; Mao et al. 2021). Recently, manystudies leveraged advanced LLMs to augment queries and
achieved notable improvement (Buss et al. 2023; Wang,
Yang, and Wei 2023; Jagerman et al. 2023; Gao et al. 2023a;
Lin et al. 2023). For example. Wang, Yang, and Wei (2023)
first prompted LLMs to generate a pseudo document of the
query, which was then concatenated with the original query
to enhance the retrieval. In addition, a few studies (Jeong
et al. 2022, 2024) used an external database, such as relevant
tabular data, to augment original queries, which can enhance
query representations but require additional data. Further-
more, a few studies (Chan et al. 2024; Zhang et al. 2025a)
trained models to refine or encode queries to enhance the
performance of information retrieval and question answer-
ing. However, the aforementioned approaches did not fully
leverage the parametric knowledge of LLMs that can answer
a proportion of queries directly. In addition, they did not ex-
plore more effective combination methods of the query and
generated context to retrieve and select more relevant infor-
mation.
Information retrieval and selection
How to retrieve and select relevant information is a key issue
in RAG systems. As the remarkable success of advancing
LLMs using reinforcement learning (RL), many studies have
adopted RL to train LLMs to enhance information retrieval
and reranking (Asai et al. 2024; Jin et al. 2025; Li et al.
2025; Song et al. 2025; Yu et al. 2024). Also, a few studies
(Jiang et al. 2025; Yan and Ling 2025) aligned the retriever’s
preferences with LLMs to retrieve more relevant informa-
tion and improve the generation performance. These meth-
ods could achieve significant improvement but may suffer
from high computational costs. In addition, with the acces-
sibility to advanced reranking models, such as bge-reranker
(Xiao et al. 2023) and Qwen3-reranker (Zhang et al. 2025b),
2

a straightforward method is to first rerank the retrievals using
a reranker and then select the top- krelevant retrievals (Glass
et al. 2022; Chang et al. 2025). This method could further
select retrievals with a higher semantic similarity with the
query and thus enhance the generation performance. How-
ever, it may not perform well or even degrade answer ac-
curacy in QA tasks due to the sparse query or large corpus
(Kim et al. 2024; Jiang et al. 2025).
Methodology
In this section, we introduce the preliminary problem and
the details of our proposed method.
Preliminary
In a retrieval–augmented generation (RAG) system, a collec-
tion of documents is first split into small chunks D, which
are then embedded into vectors using an embedding model
f. Given a query q, the retriever Rfirst searches top krele-
vant chunks Dk={d1, d2, ..., d k} ⊂Dfrom the collection
according to the similarity between the query and chunks,
i.e.,Dk=R(D, q). The similarity sis calculated using the
inner product (IP) or L2 norm between the query embedding
q=f(q)and chunk embeddings d=f(d),∀d∈D. Fi-
nally, the retrievals and query are incorporated into a prompt
P(q, Dk), which is used as the input of the generator model
gto generate final answer, i.e., ˆa=G(P(q, Dk)). We list all
notations and abbreviations in the appendix.
Parametric verification mechanism
Modern LLMs (Achiam et al. 2023; Guo et al. 2025; Yang
et al. 2025) are equipped with powerful parametric knowl-
edge acquired from large-scale pretraining corpora, enabling
them to answer many questions without the need for ex-
ternal information. To exploit this capability, PAIRS in-
corporates a parametric verification mechanism that adap-
tively determines whether external retrieval is necessary. At
its core is a dual-generation strategy, where the LLM pro-
duces two responses to a given query q: one generated di-
rectly from its parametric knowledge, i.e., ˆa1=G(P(q)),
and the other based on a self-generated pseudo-context, i.e.,
ˆa2=G(P(q, p)). This pseudo-context pmimics retrieved
content but is synthesized internally by the LLM, ensuring
the generation process remains efficient and self-contained.
If the two responses converge— the same or demonstrat-
ing semantic consistency—PAIRS concludes that external
retrieval is unnecessary and directly returns the answer. This
approach not only reduces computational overhead but also
avoids the potential pitfalls of retrieving irrelevant or noisy
documents. On the other hand, if the two responses di-
verge, suggesting uncertainty in the model’s parametric un-
derstanding, the system proceeds to initiate external docu-
ment retrieval and selection. In this way, the parametric ver-
ification mechanism serves as a lightweight and effective de-
cision gate that enhances both the efficiency and robustness
of the RAG pipeline.
Dual-path retrieval
One of the central challenges in RAG is handling sparse or
underspecified queries, which often lead to the retrieval of ir-relevant or low-quality documents (Zhao et al. 2024; Singh
et al. 2025). This issue becomes particularly pronounced
when relying solely on the original query to perform re-
trieval (please refer to the appendix for further analysis).
However, the previously generated pseudo-context can be
used to enhance the retrieval. An intuitive solution might
be to concatenate the query with LLM–generated pseudo-
context to enrich the signal (Wang, Yang, and Wei 2023;
Jagerman et al. 2023); however, such concatenation could
be inefficient, as the query and pseudo-context serve differ-
ent semantic roles and may not integrate cohesively. To ad-
dress this, PAIRS adopts a dual-path retrieval (DPR) mech-
anism that treats the query and the self-generated context as
complementary sources of information. Instead of combin-
ing them into a single retrieval signal, the system executes
two parallel retrieval operations—one conditioned on the
query and the other on the pseudo-context. Specifically, as
shown in Fig. 2 (a), the system retrieves top nrelevant doc-
uments using the embeddings of the query q=f(q)and the
generated context p=f(p), respectively, and the retrieved
documents can be denoted as D2n={d1, d2, ..., d 2n} ⊂D.
This dual-path strategy enhances the relevance and diversity
of the retrieved documents, which is especially beneficial
for complex or ambiguous queries where either source alone
may fall short.
rqprqpd𝜃!𝜃"𝜃#Retrievalembeddingsa.Dual-path retrievalb.Similarity calculation
Figure 2: Illustration of (a) dual-path retrieval mechanism
and (b) similarity calculation of retrievals.
Adaptive information selection
Once the dual-path retrieval yields a set of 2ncandidate doc-
uments based on the original query and the LLM-generated
pseudo-context, PAIRS introduces an adaptive information
selection (AIS) mechanism to refine this candidate set. The
goal is to prioritize documents that are simultaneously rel-
evant to qandp, ensuring that the final context passed to
the LLM is not only topically aligned but also semantically
coherent from different perspectives. Formally, we compute
the score for each document s(d)as follows:
s(d) =s1(d, q) +s2(d, p),∀d∈D2n, (1)
where s1(d, q)ands2(d, p)denote the relevance between the
document dand the query qand LLM–generated context p,
respectively.
As shown in Fig. 2 (b), a straightforward method is to
calculate the relevance ( s1ands2) as the IP between their
corresponding embeddings, as follows:
s1(d, q) = cos( θ1) =⟨d,q⟩, (2)
3

s2(d, q) = cos( θ2) =⟨d,p⟩, (3)
where q=f(q),p=f(p), and d=f(d),∀d∈D2n.
Remind that q,p, and dare normalized vectors. However,
such aggregation may not fully capture the joint relevance,
especially if either qorpintroduces semantic noise. There-
fore, AIS leverages a geometric intuition: if both qandp
align well with a given document d, then the combined an-
gleθ=θ1+θ2should be minimized. This motivates the
maximization of s(d) = cos( θ) = cos( θ1+θ2), which ex-
pands to:
s(d) = cos( θ1+θ2)
= cos( θ1)·cos(θ2)−p
1−(cos(θ1))2·p
1−(cos(θ2))2
=s1·s2−p
1−(s1)2·p
1−(s2)2. (4)
This formulation implicitly rewards documents that are
jointly aligned with both qandpwhile penalizing those that
diverge from either direction. The DPR-AIS algorithm, i.e.,
dual-path retrieval–based adaptive information selection, is
formulated as Algorithm 1.
Algorithm 1: DPR-AIS
Input : Query q, LLM–generated pseudo-context p, encoder
f(·), retriever R, corpus D={d1, d2, . . . , d N}, retrieval
sizen, and selection size k
Output : Filtered relevant documents Dk⊂D
1:Compute embeddings: q←f(q),p←f(p),d←
f(d)
2:Retrieve top- ndocuments using q:Dq← R (D,q)
3:Retrieve top- ndocuments using p:Dp← R (D,p)
4:Merge retrieved sets: D2n←Dq∪Dp
5:foreachd∈D2ndo
6: Compute s1=⟨q,d⟩ands2=⟨p,d⟩
7: Compute joint relevance score using Eq. (4)
8:end for
9:Select top- kdocuments from D2nby descending s(d)
values
10:return selected document set Dk
Extension and beyond
The flexibility of the proposed DPR-AIS mechanism opens
several promising avenues for extension and enhancement.
A key observation is that the relative importance of the orig-
inal query qand the LLM-generated pseudo-context pmay
vary across different scenarios. For instance, when qis un-
derspecified or ambiguous, pmay provide more concrete se-
mantic signals; conversely, when pis noisy or misaligned,
qmay remain the more reliable anchor for retrieval. To ac-
commodate this variability, we extend our selection metric
by introducing a dynamic weighting scheme that modulates
the contribution of each signal. Specifically, we define the
weighted retrieval direction as:
θdynamic =α·θ1+ (1−α)·θ2, (5)
where the coefficient α∈[0,1]serves as an importance
factor that can be determined dynamically, and a larger αimplies greater trust in the query and less in the pseudo-
context. One principled approach is to compute θ0, the angle
between qandp(as shown in Fig. 2 (b)), and use it to set
α, thereby reflecting the alignment between qandp. Specif-
ically, given a training set of queries and their correspond-
ing documents, generated contexts, and their associated rel-
evance judgments, we can empirically determine αfor each
instance as:
α=θ2
θ1+θ2, (6)
which reflects the relative informativeness of the query q
compared to the generated context pwhen retrieving a rele-
vant document d. Then, we can fit a model to predict αfrom
θ0at inference time.
Furthermore, DPR-AIS can be integrated with exist-
ing reranking architectures. Rather than relying purely
on embedding–based similarity, we can use the reranking
scores between qand candidate document dass1(d, q), and
between panddass2(d, p). Therefore, the score for a doc-
ument can be calculated as follows:
s(d) =K(d, q) +K(d, p), (7)
whereK(·,·)denotes the reranking model that evaluates the
semantic relevance between two documents. This provides a
novel extension to traditional reranking pipelines by consid-
ering both the explicit query and its latent expansion.
Experiments
In this section, we report our experiment setups, including
implementation details, evaluation datasets, and baselines,
main experimental results, ablation study results, and in-
depth discussions.
Experiment setup
Evaluations datasets. We evaluate the effectiveness of the
proposed method on open domain QA datasets: (1) Natu-
ralQA (Kwiatkowski et al. 2019), (2) WebQuestions (Be-
rant et al. 2013), (3) SQuAD (Rajpurkar et al. 2016), and
(4) TriviaQA (Joshi et al. 2017). In addition, we further test
the method using two complex multi-hop QA datasets, i.e.,
HotpotQA (Yang et al. 2018) and 2WikiMultiHopQA (Ho
et al. 2020). For the open domain datasets, we use the 21M
English Wikipedia dump as the corpus for retrieval, while
for the multi-hop datasets, we use their original corpus. The
statistics of all datasets are listed in Tab. 1.
Datasets #Questions #Documents
NaturalQA 3,610 21 M
WebQuestions 2,032 21 M
SQuAD 10,570 21 M
TriviaQA 11,313 21 M
HotpotQA 7,405 5 M
2WikiMultiHopQA 12,576 431 K
Table 1: Statistics of evaluation datasets.
4

MethodHotpotQA 2WikiQA NaturalQA WebQuestions SQuAQ TriviaQA Average
EM F1 EM F1 EM F1 EM F1 EM F1 EM F1 EM F1
No Retrieval 17.27 23.92 21.55 24.97 15.54 18.85 17.67 24.42 10.54 16.96 38.40 20.25 21.16 21.56
Standard RAG 34.84 44.75 29.19 34.17 34.29 36.06 20.08 27.68 28.99 34.41 54.69 30.89 33.68 34.66
HyDE 32.74 41.93 23.47 28.17 34.60 36.38 22.69 30.74 25.53 31.03 56.19 31.26 32.54 33.25
Q2D 33.71 43.41 24.25 29.07 35.68 37.36 22.79 30.77 29.20 34.79 58.12 32.56 33.96 34.66
CoT 32.96 42.50 24.05 28.88 35.67 37.36 22.15 30.52 28.38 34.23 58.32 32.39 33.59 34.13
PAIRS 35.14 45.20 26.99 31.73 36.87 38.17 23.08 31.15 28.85 34.90 59.23 32.92 35.03 35.68
RA ratio (74.4%) (69.1%) (82.8%) (78.6%) (86.3%) (61.6%) (75.5%)
DPR-AIS 36.61 47.02 28.48 33.48 37.42 38.95 22.00 30.63 30.00 35.87 59.09 33.32 35.60 36.55
Rerank 38.56 49.48 31.02 36.04 33.60 35.12 20.47 28.83 26.20 31.48 57.58 32.64 34.57 35.60
DPR-AIS-rerank 39.42 50.21 30.34 35.10 36.12 37.36 21.85 30.30 29.99 35.81 59.78 33.88 36.25 37.11
Table 2: Main results across six QA datasets. RA ratio indicates the proportion of queries for which the retriever was activated
by PAIRS. Bold and underlined values represent the best and second-best scores, respectively.
Baselines. We compare the proposed method with other
training–free baselines. (1) No Retrieval leverages the para-
metric knowledge of LLM to directly generate response to
the given query without retrieval. (2) Standard RAG in-
corporates the retrieved top- kdocuments with the query as
the input of LLM through prompting. (3) HyDE (Gao et al.
2023a) leverages the LLM–generated passage to enhance
the retrieval. (4) Q2D (Wang, Yang, and Wei 2023) con-
catenates multiple queries and the LLM–generated pseudo-
document to perform the retrieval. (5) CoT (Jagerman et al.
2023) first prompts the LLM to generate the answer as
well as the rationale to the given query, and then com-
bines multiple queries and the LLM outputs into the retrieval
signal. In addition, we also compare DPR-AIS–powered
reranking (denoted as DPR-AIS-rerank ) with the tradi-
tional reranking mechanism. Hence, we have the last base-
line: (6) Rerank (Glass et al. 2022; Chang et al. 2025) se-
lects the top- kdocuments from retrievals using a reranker.
Evaluation metrics. To comprehensively assess the qual-
ity of generated answers, we adopt two widely used metrics
in open-domain question answering: Exact Match (EM) and
F1 score. The EM metric quantifies the proportion of predic-
tions that exactly match any one of the ground-truth answers,
serving as a strict measure of correctness. On the contrary,
the F1 score provides a more forgiving evaluation by com-
puting the token-level overlap between the predicted and
reference answers. We normalize the predicted and ground
truth answers following the implementation of Fang, Meng,
and Macdonald (2025).
Implementation details. We implement our PAIRS
framework using Qwen2.5-7B-Instruct (QwenTeam 2024)
as the backbone LLM for answer and pseudo-context gen-
eration. To ensure deterministic outputs and eliminate vari-
ability due to random sampling, we set the temperature to
0.0 during decoding (Kim et al. 2024). For dense retrieval,
we employ the bge-large-en-v1.5 embedding model (Xiao
et al. 2023) with a hidden dimension of 768, using inner
product (IP) as the similarity metric. In the dual-path re-
trieval step, we retrieve the top 5 most relevant documents
independently for both the query and the generated pseudo-
context, resulting in a combined candidate pool of 10 docu-
ments ( 2n= 10 ). For reranking, we adopt the bge-reranker-base model (Xiao et al. 2023) to reorder the top-10 docu-
ments retrieved by the query-only baseline. Across all ex-
periments, we pass the top 3 documents to the LLM for fi-
nal answer generation, except No Retrieval . All remaining
experimental configurations of baselines follow the imple-
mentations reported in their respective original papers. We
present the prompts used in this study in the appendix.
Main results
Tab. 2 summarizes the performance of PAIRS and its vari-
ants against a range of baselines on six QA datasets. We
report both EM and F1 scores, along with the retriever ac-
tivation ratio ( RA ratio ) of PAIRS across datasets. Over-
all, the proposed PAIRS framework consistently outper-
forms baseline methods. Specifically, (1) PAIRS surpasses
all prior baselines—including Standard RAG, HyDE, Q2D,
and CoT—with an average of 1.1% EM and 1.0% F1 score
improvement while maintaining a retriever activation ratio
of only 75.5%. This demonstrates PAIRS’s ability to en-
hance both accuracy and efficiency by bypassing unneces-
sary retrieval for simple queries. (2) DPR-AIS , which inte-
grates dual-path retrieval and adaptive information selection
for all queries, further improves performance, achieving the
highest average EM (35.60%) and F1 score (36.55%) among
non-reranking models. The consistent gains across various
datasets highlight the robustness of DPR-AIS in both open-
domain and multi-hop QA scenarios. (3) DPR-AIS-rerank
extends these gains even further by incorporating a reranker,
respectively improving the EM and F1 score by 2.3% and
2.5% on average. Also, it improves over the Rerank base-
line by 1.7% EM and 1.5% F1 score on average, reach-
ing state-of-the-art results in 5 out of 6 datasets. These im-
provements validate the compatibility and synergy between
DPR-AIS and post-retrieval reranking mechanisms. (4) No-
tably, on datasets like WebQuestions, where queries con-
tain sparse signals and retrieval noise is common, PAIRS
achieves higher performance than DPR-AIS, illustrating that
adaptive retrieval can improve generation accuracy by skip-
ping unhelpful or distracting documents.
5

MethodHotpotQA NaturalQA SQuAQ TriviaQA Average
EM F1 EM F1 EM F1 EM F1 EM F1
DPR-AIS 36.61 47.02 37.42 38.95 30.00 35.87 59.09 33.32 40.78 38.79
w/o DPR (w/ q) 36.14 46.36 36.21 37.92 30.10 35.99 58.08 32.85 40.13 38.28
w/o DPR (w/ p) 35.04 44.90 36.73 38.09 28.34 34.17 57.77 32.40 39.47 37.39
w/o AIS (w/ 2 q+ 1p) 33.94 43.49 31.99 33.90 26.11 31.43 53.05 29.72 36.27 34.64
w/o AIS (w/ 1 q+ 2p) 28.55 37.38 28.73 31.53 22.00 26.76 48.76 27.38 32.01 30.76
Table 3: Ablation experiments on four datasets, where w/o DPR (w/ q) and w/o DPR (w/ q) remove dual-path retrieval but only
usesqandpto retrieve documents, respectively; and w/o AIS (w/ 2 q+ 1p) and w/o AIS (w/ 1 q+ 2p) use fixed ratios of retrieved
documents instead of adaptive selection. Bold and underlined values represent the best and second-best scores, respectively.
Ablation study
To better understand the contributions of different com-
ponents within the DPR-AIS framework, we conduct a
series of ablation studies across four representative QA
datasets. The results are summarized in Tab. 3; the complete
DPR-AIS model consistently achieves the best performance
across all four evaluated datasets, highlighting the effective-
ness of our design.
Specifically, to assess the role of DPR, we examine two
variants where one retrieval path is removed: w/o DPR (w/
q)andw/o DPR (w/ p). In these settings, the system retrieves
the top 10 documents using only the query qor the generated
pseudo-context p, respectively, and then applies AIS to se-
lect the final top 3 documents for generation. We find that
removing either path leads to a performance drop. Notably,
the drop is larger when using only p(w/o DPR (w/ p)), sug-
gesting that the LLM-generated pseudo-context is not suffi-
cient on its own. On the other hand, when relying only on the
query ( w/o DPR (w/ q)), the performance remains relatively
close to the full DPR-AIS model, indicating the robustness
of query–based retrieval and its dominance in guiding rele-
vance.
We then assess the effectiveness of the Adaptive Informa-
tion Selection (AIS) module by replacing it with fixed doc-
ument selection heuristics. In w/o AIS (w/ 2 q+ 1p)andw/o
AIS (w/ 1 q+ 2p), we retrieve documents using fixed propor-
tions (e.g., two from query and one from pseudo-context,
or vice versa), bypassing the adaptive scoring mechanism.
These variants perform significantly worse than DPR-AIS
across all datasets, with particularly steep drops on SQuAQ
and TriviaQA. This highlights the necessity of adaptively
weighing relevance from both sources. Please refer to the
appendix for a more detailed analysis of the relationship
among the query, pseudo-context, and retrievals.
Additional analysis
Dynamic weighting between query and pseudo-context.
To better understand the impact of weighting between the
query and the LLM–generated pseudo-context in the AIS
module, we randomly sampled 5,000 queries from the Hot-
potQA training set and generated pseudo-contexts for each
query using the LLM. We then explored the correlation be-
tween α(calculated using Eq. (6)) and θ0(the angle between
the query and the pseudo-context) to examine how the diver-
gence between the query and the generated context affectstheir relative importance (please refer to the appendix for de-
tails). We conducted a simple linear regression and obtained
the relationship, expressed as:
α= 0.058θ0+ 0.455. (8)
This result suggests that as the semantic gap between the
query and pseudo-context increases, the optimal strategy is
to place more emphasis on the original query during docu-
ment selection. We then integrated this dynamic weighting
mechanism into the AIS module and evaluated its effective-
ness. As shown in Tab. 4, the dynamically weighted vari-
ant (DPR-AIS-dynamic) slightly improves over the fixed-
weight version, achieving higher EM and F1 scores on Hot-
potQA. This result highlights the adaptability of AIS and
the benefit of learning to balance information sources on a
per-sample basis. In future work, more advanced strategies
such as learned weighting mechanisms or neural attention
over context sources could be explored to further enhance
adaptive retrieval and selection.
MethodHotpotQA
EM F1
DPR-AIS 36.61 47.02
DPR-AIS-dynamic 36.81 47.03
Table 4: Experimental results on HotpotQA with and with-
out considering dynamic weighting between the query and
LLM–generated pseudo-context.
HotpotQA2WikiQANaturalQAWebQSQuAQ TriviaQA0.00.10.20.30.40.50.60.7EMNon-DQ-retrieval
DQ
DQ-retrieval
Figure 3: EM scores of different queries across six datasets
6

MethodHotpotQA 2WikiQA NaturalQA WebQuestions SQuAQ TriviaQA Average
EM F1 EM F1 EM F1 EM F1 EM F1 EM F1 EM F1
PAIRS 35.14 45.20 26.99 31.73 36.87 38.17 23.08 31.15 28.85 34.90 59.23 32.92 35.03 35.68
RA ratio (74.4%) (69.1%) (82.8%) (78.6%) (86.3%) (61.6%) (75.5%)
Exclude num 35.30 45.38 27.00 31.75 37.04 38.33 22.98 31.13 29.22 35.28 59.18 32.93 35.12 35.80
RA ratio (76.1%) (69.8%) (87.9%) (80.7%) (89.9%) (64.2%) (78.1%)
Table 5: Effects of LLM hallucinations on directly generated answers using parametric knowledge.
Who was born first Am Rong or Ava DuVernay?QueryAnswerAmRongGenerated pseudo-contextAmRongandAvaDuVernaywerebothinfluentialfiguresintheirrespectivefields,….AmRong,….ShewasbornonJanuary31,1905,inLosAngeles,California.AvaDuVernay,…,wasbornonMarch26,1972,inLongBeach,California.…,itisclearthatAmRongwasbornbeforeAvaDuVernay.Ans1Ans2AmRongAmRongRetrievals1.AvaDuVernay:AvaMarieDuVernay(bornAugust24,1972)isanAmericandirector,screenwriter,filmmarketer,andfilmdistributor.…2.EmayatzyCorinealdi:EmayatzyEvettCorinealdi(bornJanuary14,1980)….SheisbestknownforherleadingroleintheAvaDuVernayfilm"MiddleofNowhere"(2012).”3.ARRAY:ARRAY,…,isanindependentdistributioncompanylaunchedbyfilmmakerandformerpublicistAvaDuVernayin2010….AnsAva DuVernay
Figure 4: Qualitative comparison between PARIS and stan-
dard RAG.
Parametric knowledge can generate accurate answers.
To further investigate the LLM’s ability to generate accurate
answers without relying on external retrieval, we categorize
the test queries into three groups: (1) queries that are an-
swered by the LLM directly, referred to as Directly Answer-
able Queries ( DQ); (2) the remaining queries that require re-
trievals, referred to as Non-DQ-retrieval ; and (3) a control
setting where the same DQs are answered using retrieval, de-
noted as DQ-retrieval . Fig. 3 presents the exact match (EM)
results across six datasets (with corresponding F1 scores and
other quantitative results provided in the appendix). We ob-
serve that the EM scores achieved by the LLM on DQ are
consistently high, and in some cases, even surpass those of
DQ-retrieval. Both DQ and DQ-retrieval substantially out-
perform Non-DQ-retrieval, indicating that DQs are gener-
ally simpler or factually grounded queries that fall within
the LLM’s pre-trained knowledge base. These results indi-
cate that our dual-path generation mechanism not only al-
lows accurate answer generation directly but also supports
selective retrieval when needed—offering a more efficient
and adaptive RAG framework.
Effects of LLM hallucinations. To assess the impact of
hallucinations when large language models (LLMs) gener-
ate answers without retrieval, we conduct a controlled ex-
periment based on a simple heuristic: if a generated answer
contains numeric values, it is more likely to be affected byhallucination. This is because LLMs are generally less reli-
able when producing precise facts such as numbers, dates,
or counts from parametric memory alone (Ji et al. 2023;
Singh et al. 2025). We filter out all directly answered queries
(DQs) whose generated answers contain numbers, and we
then rerun our DPR-AIS for these queries (referred to Ex-
clude num). The results are reported in Tab. 5. Overall, ex-
cluding numeric DQs results in slightly improved perfor-
mance. The average exact match (EM) increases from 35.03
to 35.12, and the average F1 score improves from 35.68 to
35.80. While these gains are modest, they come with an in-
crease in the retriever activation (RA) ratio—from 75.5%
to 78.1%. These findings suggest that hallucinations from
LLMs, especially when dealing with factual numeric con-
tent, can indeed affect answer quality in retrieval-free set-
tings. However, their overall impact remains limited in our
setting, as the performance gain is marginal. This reinforces
the strength of PAIRS: it naturally handles simple queries
with direct parametric answers with trivial effects of LLM
hallucinations.
Case study. Fig. 4 demonstrates a representative exam-
ple. The query is a simple factual comparison that lies well
within the LLM’s parametric knowledge, and as shown in
the generated pseudo-context, the model accurately gener-
ates birth dates and correctly answers Am Rong . However,
the standard RAG generates an incorrect answer because
the retrievals present Ava DuVernay ’s birth year (1972)
only, lacking complete evidence on Am Rong . This exam-
ple demonstrates that LLMs can generate accurate answers
to simple queries using their internal parametric knowledge,
while noisy or incomplete retrievals—especially those miss-
ing critical comparative context—can mislead the final gen-
eration. We present more cases in the appendix.
Conclusion
In this paper, we propose PAIRS, a simple yet effective
framework that adaptively integrates parametric and re-
trieved knowledge for question answering. By verifying the
LLM’s parametric response and selectively activating re-
trieval with dual-path signals and adaptive selection, PAIRS
improves both accuracy and efficiency. Experiments across
multiple QA benchmarks demonstrate that PAIRS consis-
tently outperforms existing baselines and can be easily ap-
plied to real-world RAG systems.
7

References
Achiam, J.; Adler, S.; Agarwal, S.; Ahmad, L.; Akkaya,
I.; Aleman, F. L.; Almeida, D.; Altenschmidt, J.; Alt-
man, S.; Anadkat, S.; et al. 2023. Gpt-4 technical report.
arXiv:2303.08774.
Asai, A.; Wu, Z.; Wang, Y .; Sil, A.; and Hajishirzi, H.
2024. Self-RAG: Learning to Retrieve, Generate, and Cri-
tique through Self-Reflection. In The Twelfth International
Conference on Learning Representations .
Berant, J.; Chou, A.; Frostig, R.; and Liang, P. 2013. Se-
mantic parsing on freebase from question-answer pairs. In
Proceedings of the 2013 conference on empirical methods in
natural language processing , 1533–1544.
Buss, C.; Mosavi, J.; Tokarev, M.; Termehchy, A.; Maier, D.;
and Lee, S. 2023. Generating Data Augmentation Queries
Using Large Language Models. In VLDB Workshops .
Chan, C.-M.; Xu, C.; Yuan, R.; Luo, H.; Xue, W.; Guo, Y .;
and Fu, J. 2024. Rq-rag: Learning to refine queries for re-
trieval augmented generation. arXiv:2404.00610.
Chang, C.-Y .; Jiang, Z.; Rakesh, V .; Pan, M.; Yeh, C.-C. M.;
Wang, G.; Hu, M.; Xu, Z.; Zheng, Y .; Das, M.; and Zou,
N. 2025. MAIN-RAG: Multi-Agent Filtering Retrieval-
Augmented Generation. In Proceedings of the 63rd Annual
Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers) , 2607–2622. Vienna, Austria: As-
sociation for Computational Linguistics.
Fang, J.; Meng, Z.; and Macdonald, C. 2025. Ki-
RAG: Knowledge-Driven Iterative Retriever for Enhancing
Retrieval-Augmented Generation. arXiv:2502.18397.
Gao, L.; Ma, X.; Lin, J.; and Callan, J. 2023a. Precise zero-
shot dense retrieval without relevance labels. In Proceedings
of the 61st Annual Meeting of the Association for Computa-
tional Linguistics (Volume 1: Long Papers) , 1762–1777.
Gao, Y .; Xiong, Y .; Gao, X.; Jia, K.; Pan, J.; Bi, Y .; Dai,
Y .; Sun, J.; Wang, H.; and Wang, H. 2023b. Retrieval-
augmented generation for large language models: A survey.
arXiv:2312.10997.
Glass, M.; Rossiello, G.; Chowdhury, M. F. M.; Naik, A. R.;
Cai, P.; and Gliozzo, A. 2022. Re2G: Retrieve, rerank, gen-
erate. arXiv:2207.06300.
Guo, D.; Yang, D.; Zhang, H.; Song, J.; Zhang, R.; Xu, R.;
Zhu, Q.; Ma, S.; Wang, P.; Bi, X.; et al. 2025. Deepseek-r1:
Incentivizing reasoning capability in llms via reinforcement
learning. arXiv:2501.12948.
Guu, K.; Lee, K.; Tung, Z.; Pasupat, P.; and Chang, M.
2020. Retrieval augmented language model pre-training. In
International conference on machine learning , 3929–3938.
PMLR.
Ho, X.; Nguyen, A.-K. D.; Sugawara, S.; and Aizawa, A.
2020. Constructing a multi-hop qa dataset for comprehen-
sive evaluation of reasoning steps. arXiv:2011.01060.
Jagerman, R.; Zhuang, H.; Qin, Z.; Wang, X.; and Bender-
sky, M. 2023. Query expansion by prompting large language
models. arXiv:2305.03653.Jeong, S.; Baek, J.; Cho, S.; Hwang, S. J.; and Park,
J. C. 2022. Augmenting Document Representations for
Dense Retrieval with Interpolation and Perturbation. In
60th Annual Meeting of the Association-for-Computational-
Linguistics (ACL) , 442–452. ASSOC COMPUTATIONAL
LINGUISTICS-ACL.
Jeong, S.; Baek, J.; Cho, S.; Hwang, S. J.; and Park, J. C.
2024. Database-Augmented Query Representation for In-
formation Retrieval. arXiv:2406.16013.
Ji, Z.; Lee, N.; Frieske, R.; Yu, T.; Su, D.; Xu, Y .; Ishii, E.;
Bang, Y . J.; Madotto, A.; and Fung, P. 2023. Survey of hal-
lucination in natural language generation. ACM computing
surveys , 55(12): 1–38.
Jiang, Y .; Zhao, S.; Li, J.; Wang, H.; and Qin, B. 2025. Gain-
RAG: Preference Alignment in Retrieval-Augmented Gen-
eration through Gain Signal Synthesis. arXiv:2505.18710.
Jin, B.; Zeng, H.; Yue, Z.; Yoon, J.; Arik, S.; Wang, D.; Za-
mani, H.; and Han, J. 2025. Search-r1: Training llms to rea-
son and leverage search engines with reinforcement learn-
ing.arXiv preprint arXiv:2503.09516 .
Joshi, M.; Choi, E.; Weld, D. S.; and Zettlemoyer, L. 2017.
TriviaQA: A Large Scale Distantly Supervised Challenge
Dataset for Reading Comprehension. In Proceedings of the
55th Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers) , 1601–1611.
Karpukhin, V .; Oguz, B.; Min, S.; Lewis, P. S.; Wu, L.;
Edunov, S.; Chen, D.; and Yih, W.-t. 2020. Dense Pas-
sage Retrieval for Open-Domain Question Answering. In
EMNLP (1) , 6769–6781.
Kim, J.; Nam, J.; Mo, S.; Park, J.; Lee, S.-W.; Seo, M.; Ha,
J.-W.; and Shin, J. 2024. SuRe: Summarizing Retrievals us-
ing Answer Candidates for Open-domain QA of LLMs. In
The Twelfth International Conference on Learning Repre-
sentations . Vienna, Austria.
Kwiatkowski, T.; Palomaki, J.; Redfield, O.; Collins, M.;
Parikh, A.; Alberti, C.; Epstein, D.; Polosukhin, I.; Devlin,
J.; Lee, K.; et al. 2019. Natural questions: a benchmark for
question answering research. Transactions of the Associa-
tion for Computational Linguistics , 7: 453–466.
Lavrenko, V .; and Croft, W. B. 2017. Relevance-based lan-
guage models. In ACM SIGIR Forum , volume 51, 260–267.
ACM New York, NY , USA.
Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V .;
Goyal, N.; K ¨uttler, H.; Lewis, M.; Yih, W.-t.; Rockt ¨aschel,
T.; et al. 2020. Retrieval-augmented generation for
knowledge-intensive nlp tasks. Advances in neural infor-
mation processing systems , 33: 9459–9474.
Li, X.; Dong, G.; Jin, J.; Zhang, Y .; Zhou, Y .; Zhu, Y .; Zhang,
P.; and Dou, Z. 2025. Search-o1: Agentic search-enhanced
large reasoning models. arXiv:2501.05366.
Lin, S.-C.; Asai, A.; Li, M.; Oguz, B.; Lin, J.; Mehdad, Y .;
Yih, W.-t.; and Chen, X. 2023. How to Train Your Dragon:
Diverse Augmentation Towards Generalizable Dense Re-
trieval. In Findings of the Association for Computational
Linguistics: EMNLP 2023 , 6385–6400.
8

Lv, Y .; and Zhai, C. 2009. A comparative study of methods
for estimating query language models with pseudo feedback.
InProceedings of the 18th ACM conference on Information
and knowledge management , 1895–1898.
Mao, Y .; He, P.; Liu, X.; Shen, Y .; Gao, J.; Han, J.; and Chen,
W. 2021. Generation-augmented retrieval for open-domain
question answering. In Joint Conference of the 59th An-
nual Meeting of the Association for Computational Linguis-
tics and the 11th International Joint Conference on Natural
Language Processing, ACL-IJCNLP 2021 , 4089–4100. As-
sociation for Computational Linguistics (ACL).
QwenTeam. 2024. Qwen2 technical report.
arXiv:2407.10671.
Rajpurkar, P.; Zhang, J.; Lopyrev, K.; and Liang, P. 2016.
Squad: 100,000+ questions for machine comprehension of
text. arXiv:1606.05250.
Ram, O.; Levine, Y .; Dalmedigos, I.; Muhlgay, D.; Shashua,
A.; Leyton-Brown, K.; and Shoham, Y . 2023. In-context
retrieval-augmented language models. Transactions of the
Association for Computational Linguistics , 11: 1316–1331.
Singh, A.; Ehtesham, A.; Kumar, S.; and Khoei, T. T. 2025.
Agentic retrieval-augmented generation: A survey on agen-
tic rag. arXiv:2501.09136.
Song, H.; Jiang, J.; Min, Y .; Chen, J.; Chen, Z.; Zhao, W. X.;
Fang, L.; and Wen, J.-R. 2025. R1-searcher: Incentiviz-
ing the search capability in llms via reinforcement learning.
arXiv:2503.05592.
Touvron, H.; Lavril, T.; Izacard, G.; Martinet, X.; Lachaux,
M.-A.; Lacroix, T.; Rozi `ere, B.; Goyal, N.; Hambro, E.;
Azhar, F.; et al. 2023. Llama: Open and efficient founda-
tion language models. arXiv:2302.13971.
Wang, L.; Yang, N.; and Wei, F. 2023. Query2doc: Query
Expansion with Large Language Models. In Proceedings
of the 2023 Conference on Empirical Methods in Natural
Language Processing , 9414–9423.
Xiao, S.; Liu, Z.; Zhang, P.; and Muennighoff, N. 2023.
C-Pack: Packaged Resources To Advance General Chinese
Embedding. arXiv:2309.07597.
Yan, S.-Q.; and Ling, Z.-H. 2025. RPO: Retrieval Preference
Optimization for Robust Retrieval-Augmented Generation.
arXiv:2501.13726.
Yang, A.; Li, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.;
Yu, B.; Gao, C.; Huang, C.; Lv, C.; et al. 2025. Qwen3
technical report. arXiv:2505.09388.
Yang, Z.; Qi, P.; Zhang, S.; Bengio, Y .; Cohen, W.; Salakhut-
dinov, R.; and Manning, C. D. 2018. HotpotQA: A Dataset
for Diverse, Explainable Multi-hop Question Answering. In
Proceedings of the 2018 Conference on Empirical Methods
in Natural Language Processing , 2369–2380.
Yu, Y .; Ping, W.; Liu, Z.; Wang, B.; You, J.; Zhang, C.;
Shoeybi, M.; and Catanzaro, B. 2024. Rankrag: Unify-
ing context ranking with retrieval-augmented generation in
llms. Advances in Neural Information Processing Systems ,
37: 121156–121184.
Zhang, W.; Lin, X. V .; Stratos, K.; Yih, W.-t.; and Chen,
M. 2025a. ImpRAG: Retrieval-Augmented Generation with
Implicit Queries. arXiv:2506.02279.Zhang, Y .; Li, M.; Long, D.; Zhang, X.; Lin, H.; Yang, B.;
Xie, P.; Yang, A.; Liu, D.; Lin, J.; et al. 2025b. Qwen3
Embedding: Advancing Text Embedding and Reranking
Through Foundation Models. arXiv:2506.05176.
Zhao, P.; Zhang, H.; Yu, Q.; Wang, Z.; Geng, Y .; Fu, F.;
Yang, L.; Zhang, W.; Jiang, J.; and Cui, B. 2024. Retrieval-
augmented generation for ai-generated content: A survey.
arXiv:2402.19473.
Zheng, Z.; Hui, K.; He, B.; Han, X.; Sun, L.; and Yates,
A. 2020. BERT-QE: Contextualized Query Expansion for
Document Re-ranking. In Findings of the Association for
Computational Linguistics: EMNLP 2020 , 4718–4728.
9

Notation List
The main notations and abbreviations used in this paper are
listed in Tab. 6.
Notation Explanation
D Chunks of Documents
Dk Retrieved kchunks
f Embedding model
q Query
q Query embedding
d A chunk
d Chunk embedding
R Retriever
P Prompt
G LLM generator
ˆa Generated answer
p LLM–generated pseudo-context
p Pseudo-context embedding
θ0 The angle between qandp
θ1 The angle between qandd
θ2 The angle between pandd
s(d) Score of d
α Importance factor of θ1
K Reranker
Abbreviation Explanation
PAIRS Parametric–verified adaptive information
retrieval and selection
DPR Dual-path retrieval
AIS Adaptive information selction
IP Inner product
EM Exact match
RA ratio Retriever activation ratio
Table 6: Explanation of main notations and abbreviations
used in this study.
Retrieval Analysis
In this section, we further analyze the properties of retrievals
using the query or LLM–generated pseudo-context. Specif-
ically, we first compare the documents retrieved using the
query with the ground-truth (GT) documents. We found that
the retrieved documents may be significantly different from
the GT documents. In addition, we analyze the spatial distri-
bution of query–based, pseudo-context–based, and GT doc-
uments. We discovered that the documents retrieved using
the pseudo-context may compensate for this shortcoming.
Similarity between query and GT documents
To figure out how the query–based documents differ from
the GT documents, we first calculated the similarity scores
between the ground-truth(GT) documents and the queries in
the HotpotQA dataset. As shown in Fig. 5, the similarity
scores of the most GT documents are below 0.8, indicating
that the query may differ from the GT documents, which can
lead to irrelevant retrievals.
0.3 0.4 0.5 0.6 0.7 0.8 0.9
GT Document Scores02004006008001000FrequencyGT Document Scores DistributionFigure 5: Similarity scores between queries and ground truth
documents.
Furthermore, we ranked the GT documents in the query–
based retrievals. As shown in Fig. 6, a significant propor-
tion of the GT documents rank 20+ in the documents re-
trieved using the query. This further demonstrates that using
the query to retrieve documents only can lead to irrelevant
results, which in turn result in inaccurate final answers.
01234567891011121314151617181920+
Ranking010002000300040005000FrequencyGT Document Ranking
Figure 6: Ranking of ground truth documents in query–
based retrievals.
Spatial distribution of query, context, and GT
documents
To figure out the relationship between the query, context,
and GT documents, we projected these documents into the
2D space. Figs. 7 and 8 illustrate their relationship in xy and
polar coordinates, respectively. Remind that we only visu-
alize 200 samples, and we set the origin of the coordinates
as the query. Most of the query documents are located be-
sides the origin (i.e., the query), while the GT documents
are kind of far from the origin, demonstrating that the GT
documents may differ from the query documents in the vec-
tor space. However, the context documents can compensate
for this gap as their distribution is similar to that of the GT
documents when they are far away from the origin. These
results further demonstrate that the LLM–generated pseudo-
context can compensate for the sparse queries, enhancing the
performance of RAG systems.
10

1.0
 0.5
 0.0 0.5 1.01.0
0.5
0.00.51.0PCA in XY Coordinates
Query docs
Context docs
GT docs
QueryFigure 7: Relationship between query, context, and GT doc-
uments in the XY coordinate.
1.0
 0.5
 0.0 0.5 1.01.00
0.75
0.50
0.25
0.000.250.500.751.00PCA in Polar Coordinates
Query docs
Context docs
GT docs
Query
Figure 8: Relationship between query, context, and GT doc-
uments in the polar coordinate.
Prompt Template
The prompt templates used in this paper for generating
pseudo-context or answers are presented in Fig. 9.
Details of Dynamic Weighting
To analyze the relationship between queries, pseudo-
contexts, and ground-truth documents, we randomly sam-
pled 5,000 queries from the HotpotQA training set and gen-
erated pseudo-contexts for each using an LLM. For ev-
ery query, we measured three key angles in the embedding
space: θ0,θ1, and θ2. Using these angles, we derived the
optimal weighting factor α=θ2
θ1+θ2. We then investigated
howαcorrelates with θ0to assess the influence of seman-
tic divergence between queries and pseudo-contexts on their
relative importance in document selection. As illustrated in
Fig. 10, a linear regression analysis revealed a clear posi-
tive relationship, which indicates that as the semantic gap
(θ0) widens, the optimal strategy increasingly prioritizes the
original query over the generated pseudo-context.
Yourtaskistowriteadetailedpassagetoanswerthegivenquestion.Question:{q}(a)Prompt for generating pseudo-contextYourtaskistoanswerthegivenquestion.Pleasedirectlyprovidetheconciseanswerinside<answer>and</answer>withoutdetaileddescription,e.g.,<answer>Beijing</answer>.Question:{q}(b)Prompt for generating answersdirectly
YourtaskistoanswerthegivenquestionbasedontheContext.Pleasedirectlyprovidetheanswerinside<answer>and</answer>withoutdetaileddescription,e.g.,<answer>Beijing</answer>.Context:{context}Question:{q}(c)Prompt for generating answerswith contextFigure 9: Prompt templates used in this paper, including (a)
generating pseudo-context, (b) generating answers directly,
and (c) generating answers based on context. q and context
denote the query and generated or retrieved context, respec-
tively.
0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
0
0.250.300.350.400.450.500.550.60
 = 0.058 0 + 0.455
Data Points
Regression Line
Figure 10: Relationship between the importance factor αand
the angle between the embeddings of the query and LLM–
generated pseudo-context. A linear relationship can be ob-
tained through regression as α= 0.058θ0+ 0.455.
Additional Results on Parametric Generation
Fig. 11 presents the F1 scores across six datasets. Similar to
the EM scores, the F1 scores achieved on both DQ and DQ-
retrieval substantially outperform those achieved on Non-
DQ-retrieval, indicating that the parametric knowledge can
generate accurate answers to simple queries. Tab. 8 presents
the averaged exact match (EM) and F1 scores across six
datasets. We observe that Non-DQ-retrieval queries yield
the lowest performance, with an average EM of 29.61 and
F1 of 32.22. In contrast, DQ queries—those answered with-
out retrieval—achieve a significantly higher EM of 50.50
and F1 of 47.37. Interestingly, when retrieval is enforced for
these DQ queries (i.e., DQ-retrieval), the performance only
slightly improves to an EM of 53.49 and F1 of 51.39. These
11

MethodHotpotQA 2WikiQA NaturalQA WebQuestions SQuAQ TriviaQA Average
EM F1 EM F1 EM F1 EM F1 EM F1 EM F1 EM F1
Standard RAG 36.49 46.84 30.15 35.04 35.46 37.48 21.31 29.10 31.29 37.22 57.36 32.49 35.34 36.36
PAIRS 35.07 45.10 28.09 32.89 36.65 38.63 23.67 31.93 31.03 37.35 60.70 33.68 35.87 36.60
RA ratio (74.4%) (69.1%) (82.8%) (78.6%) (86.3%) (61.6%) (75.5%)
DPR-AIS 36.79 47.06 29.71 34.70 37.06 39.22 22.54 31.29 32.59 38.70 61.20 34.29 36.65 37.54
Table 7: Results across six QA datasets with top-5 documents. RA ratio indicates the proportion of queries for which the
retriever was activated by PAIRS. Bold and underlined values represent the best and second-best scores, respectively.
results indicate that the proposed dual-path generation and
verification mechanism can achieve comparable accuracy to
that based on retrievals on relatively simple queries.
HotpotQA2WikiQANaturalQAWebQSQuAQ TriviaQA0.00.10.20.30.40.50.6F1Non-DQ-retrieval
DQ
DQ-retrieval
Figure 11: F1 scores of different queries across six datasets
QueriesAverage
EM F1
Non-DQ-retrieval 29.61 32.22
DQ 50.50 47.37
DQ-retrieval 53.49 51.39
Table 8: Average experimental results of different queries
across six datasets.
Additional Experiments
We also conducted experiments with top-5 documents, and
the results are listed in Tab. 7. Please note that we only con-
sider the standard RAG baseline and our proposed method
because (1) the experimental results with top-3 documents
have demonstrated that the standard RAG can be represen-
tative and achieve comparable accuracy with the other base-
lines, and (2) conducting experiments across all baselines
can be time-consuming. The experimental results are con-
sistent with those achieved with top-3 documents, where
PAIRS can achieve higher accuracy with triggering only
75% retrievals. Also, DPR-AIS can further improve the per-
formance by selecting the top-5 documents for all queries.
These results further validate the effectiveness of the pro-
posed method, which can not only enhance the efficiencybut also improve the performance of RAG systems.
Additional Case Studies
Are Ganzhou and Jimo District both located in China?QueryAnswerYesGenerated pseudo-contextYes,bothGanzhouandJimoDistrictarelocatedinChina.Ganzhouisaprefecture-levelcitysituatedinsoutheasternJiangxiProvince,knownforitsrichculturalheritageandnaturalbeauty.Ontheotherhand,JimoDistrictispartofQingdaoCity,whichislocatedineasternShandongProvince.…Ans 1Ans 2YesYesRetrievals1.Ganzhou(disambiguation):Ganzhouisaprefecture-levelcityinJiangxi,China.2.Gantian,Zhuzhou:GanzhouTown()isanurbantowninZhuzhouCounty,ZhuzhouCity,HunanProvince,People'sRepublicofChina.3.GanzhouDistrict:GanzhouDistrict,formerlytheseparatecityofGanzhouorKanchow,isadistrictinandtheseatoftheprefecture-levelcityofZhangyeinGansuProvince…AnsNo
Figure 12: Comparison between dual-path generation mech-
anism and standard RAG.
Seven Brief Lessons on Physics was written by an Italian physicist that has worked in France since what year?QueryAnswer2000Retrievals1.SevenBriefLessonsonPhysics:SevenBriefLessonsonPhysics(Italian:"")isashortbookbytheItalianphysicistCarloRovelli.OriginallypublishedinItalianin2014,…2.EttoreMajorana:EttoreMajorana(bornon5August1906\u2013probablydiedafter1959)wasanItaliantheoreticalphysicistwhoworkedonneutrinomasses.…3.CarloBecchi:CarloMariaBecchi(born20October1939)isanItaliantheoreticalphysicist.Ans 2014DPR-AIS Retrievals1.SevenBriefLessonsonPhysics:SevenBriefLessonsonPhysics(Italian:"")isashortbookbytheItalianphysicistCarloRovelli.OriginallypublishedinItalianin2014,…2.CarloRovelli:CarloRovelli(born3May1956)isanItaliantheoreticalphysicistandwriterwhohasworkedinItaly,theUnitedStatesandsince2000,inFrance.…3.GabrieleVeneziano:GabrieleVeneziano(born7September1942)isanItaliantheoreticalphysicistandoneofthepioneersofstringtheory.…Ans2000
Figure 13: Comparison between standard RAG and DPR-
AIS.
In this section, we present additional case studies that
12

demonstrate the effectiveness of our proposed method
(PAIRS) and the standard RAG baseline. Fig. 12 illustrates
that PAIRS can generate accurate answers to queries that lie
in the LLM’s parametric domain, and the dual-path genera-
tion mechanism can effectively verify the final answer. How-
ever, irrelevant documents retrieved using the query can lead
to an inaccurate answer.
Fig. 13 compares the retrievals and answers achieved by
standard RAG and DPR-AIS, respectively. Using the query
to retrieve documents only can lead to similar but irrelevant
results, which in turn results in a wrong final answer. How-
ever, our proposed dual-path retrieval mechanism can com-
pensate for sparse queries and retrieve more relevant infor-
mation, ultimately leading to accurate answers.
13