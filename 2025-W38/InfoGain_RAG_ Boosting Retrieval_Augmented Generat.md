# InfoGain-RAG: Boosting Retrieval-Augmented Generation via Document Information Gain-based Reranking and Filtering

**Authors**: Zihan Wang, Zihan Liang, Zhou Shao, Yufei Ma, Huangyu Dai, Ben Chen, Lingtao Mao, Chenyi Lei, Yuqing Ding, Han Li

**Published**: 2025-09-16 07:28:07

**PDF URL**: [http://arxiv.org/pdf/2509.12765v1](http://arxiv.org/pdf/2509.12765v1)

## Abstract
Retrieval-Augmented Generation (RAG) has emerged as a promising approach to
address key limitations of Large Language Models (LLMs), such as hallucination,
outdated knowledge, and lacking reference. However, current RAG frameworks
often struggle with identifying whether retrieved documents meaningfully
contribute to answer generation. This shortcoming makes it difficult to filter
out irrelevant or even misleading content, which notably impacts the final
performance. In this paper, we propose Document Information Gain (DIG), a novel
metric designed to quantify the contribution of retrieved documents to correct
answer generation. DIG measures a document's value by computing the difference
of LLM's generation confidence with and without the document augmented.
Further, we introduce InfoGain-RAG, a framework that leverages DIG scores to
train a specialized reranker, which prioritizes each retrieved document from
exact distinguishing and accurate sorting perspectives. This approach can
effectively filter out irrelevant documents and select the most valuable ones
for better answer generation. Extensive experiments across various models and
benchmarks demonstrate that InfoGain-RAG can significantly outperform existing
approaches, on both single and multiple retrievers paradigm. Specifically on
NaturalQA, it achieves the improvements of 17.9%, 4.5%, 12.5% in exact match
accuracy against naive RAG, self-reflective RAG and modern ranking-based RAG
respectively, and even an average of 15.3% increment on advanced proprietary
model GPT-4o across all datasets. These results demonstrate the feasibility of
InfoGain-RAG as it can offer a reliable solution for RAG in multiple
applications.

## Full Text


<!-- PDF content starts -->

InfoGain-RAG: Boosting Retrieval-Augmented Generation via Document
Information Gain-based Reranking and Filtering
Zihan Wang1,*Zihan Liang1,*Zhou Shao2,*Yufei Ma1
Huangyu Dai1Ben Chen1,†Lingtao Mao1Chenyi Lei1
Yuqing Ding1Han Li1
1Kuaishou Technology, Beijing, China
2Peking University, Beijing, China
Abstract
Retrieval-Augmented Generation (RAG) has
emerged as a promising approach to address
key limitations of Large Language Models
(LLMs), such as hallucination, outdated knowl-
edge, and lacking reference. However, current
RAG frameworks often struggle with identify-
ing whether retrieved documents meaningfully
contribute to answer generation. This short-
coming makes it difficult to filter out irrelevant
or even misleading content, which notably im-
pacts the final performance. In this paper, we
propose Document Information Gain (DIG), a
novel metric designed to quantify the contribu-
tion of retrieved documents to correct answer
generation. DIG measures a document’s value
by computing the difference of LLM’s genera-
tion confidence with and without the document
augmented. Further, we introduce InfoGain-
RAG, a framework that leverages DIG scores
to train a specialized reranker, which prioritizes
each retrieved document from exact distinguish-
ing and accurate sorting perspectives. This ap-
proach can effectively filter out irrelevant docu-
ments and select the most valuable ones for bet-
ter answer generation. Extensive experiments
across various models and benchmarks demon-
strate that InfoGain-RAG can significantly out-
perform existing approaches, on both single
and multiple retrievers paradigm. Specifically
on NaturalQA, it achieves the improvements
of 17.9%, 4.5%, 12.5% in exact match accu-
racy against naive RAG, self-reflective RAG
and modern ranking-based RAG respectively,
and even an average of 15.3% increment on
advanced proprietary model GPT-4o across all
datasets. These results demonstrate the feasibil-
ity of InfoGain-RAG as it can offer a reliable
solution for RAG in multiple applications.
1 Introduction
Recent advancements in Natural Language Process-
ing (NLP) have been significantly propelled by the
*Equal Contribution.
†Corresponding Author.emergence of LLMs (Brown et al., 2020; Achiam
et al., 2024), which demonstrates remarkable ca-
pabilities across many knowledge-intensive tasks.
However, maintaining reliability remains an on-
going challenge for LLMs, as they often struggle
with issues such as hallucination, outdated infor-
mation and lacking reference. RAG has emerged
as a promising solution to the aforementioned is-
sues. It can enhance responses by augmenting
prompts with external information, especially when
the model’s inherent knowledge is limited (Ram
et al., 2023). However, the generation quality heav-
ily depends on both the selection of relevant doc-
uments and their sequential ordering within the
LLMs’ context window (Liu et al., 2023).
Research addressing RAG document prioritiza-
tion spans multiple perspectives, of which three
pipelines gain significant attention. (Bajaj et al.,
2018; Gao et al., 2023; Ho et al., 2020) The
first pipeline focuses on retriever optimization,
which enhances retrieval performance through task-
specific training (Lewis et al., 2020; Shi et al., 2023;
Chen et al., 2024a). However, this approach be-
comes impractical when working with multiple re-
trievers (Fan et al., 2024). The second pipeline
leverages LLMs’ self-reflection capabilities to eval-
uate the utility of documents. It employs LLMs
to analyze each document and determine whether
it should be used. Although feasible, the multi-
ple LLM calls introduce substantial computational
overhead (Asai et al., 2024; Yan et al., 2024; Chang
et al., 2024). The third pipeline adds a reranker
after the retrieval stage to reorder all retrieved doc-
uments (Chen et al., 2024b; Li et al., 2024). While
this approach can effectively address multiple re-
trievers, the only consideration on semantic similar-
ity may fail to select the most useful documents for
generation (as shown in the Figure 6 of Appendix
A). All these shortcomings limit their further prac-
tical application.
To address these limitations, we propose a novelarXiv:2509.12765v1  [cs.IR]  16 Sep 2025

RAG framework, InfoGain-RAG, to filter out ir-
relevant or even misleading documents, and pri-
oritize the most valuable ones for answer genera-
tion. Specifically, we firstly introduce a new metric
named Document Information Gain (DIG), which
calculates the change in LLM’s generation confi-
dence with and without the document augmented.
A higher DIG score means the document has higher
information value. Then, a multi-task training strat-
egy is designed, enabling one newly added rerank-
ing module to predict the DIG score for each docu-
ment. Only those with a score greater than a certain
threshold will be augmented into the LLM for final
generation. This reranking module is plug-and-
play across diverse models and tasks. Furthermore,
it can efficiently handle documents from multiple
retrievers by invoking LLM only once for the entire
process and the low computational overhead makes
it feasible for the real application.
Extensive evaluations on two different types
of tasks: open-domain question answering (Trivi-
aQA (Joshi et al., 2017), NaturalQA (Kwiatkowski
et al., 2019), and PopQA (Mallen et al., 2023)) and
fact verification (FM2 (Eisenschlos et al., 2021))
spanning both proprietary LLMs (GPT, Claude)
(Wu et al., 2023; Eisele-Metzger et al., 2024)
and open-source models (LLaMA, Qwen, Gemma,
DeepSeek) (Touvron et al., 2023; Bai et al., 2023;
Team et al., 2024; Liu et al., 2024), demonstrate
substantial improvements of InfoGain-RAG over
existing methods. Specifically on NaturalQA, it
achieves significant gains in Exact Match accuracy:
outperforming naive RAG by 17.9%, retriever-
optimized RAG by 6.8%, self-reflective RAG by
4.5%, and modern ranking-based RAG by 12.5%.
Notably, even compared to the proprietary state-of-
the-art reranker GTE-7B (Zhang et al., 2024), our
method (335M) still demonstrates a 3.4% improve-
ment. These consistent performance gains extend
across TriviaQA, PopQA and FM2, validating our
approach’s effectiveness across diverse scenarios.
Our main contributions include:
•We introduce a novel metric calledDocument
Information Gain (DIG), to quantify each re-
trieved document’s impact on the LLM’s gen-
eration confidence. Different from semantic
similarity, DIG can more accurately evaluate
whether the document is helpful for generat-
ing a correct answer;
•We develop a multi-task training strategy,
which is used to optimize one reranker addedafter the retriever, with the aim of fitting the
DIG score for each document. This strategy
is designed from the exact distinguishing and
accurate sorting perspectives, so as to filter
out the irrelevant and select the most valuable
documents for answer generation.
•Integrating the DIG and the multi-task
reranker, we proposeInfoGain-RAG, a com-
prehensive framework for enhancing RAG.
This framework can improve the quality of
generation with both single and multiple re-
trievers, showing strong adaptability across
vairous real-world settings with only an effi-
cient, plug-and-play reranking module.
2 Related Work
RAG has emerged as a promising solution to ad-
dress fundamental limitations of LLMs. However,
a key challenge in RAG systems lies in effectively
evaluating and selecting the most valuable docu-
ments for answer generation. Existing document
selections in RAG broadly follow three approaches:
The first approach optimizes retrievers through
training on task-specific datasets. RePlug (Shi et al.,
2023) proposed a training pipeline that uses black-
box LLM outputs as supervision signals to optimize
the retriever, aiming to reduce LLM perplexity. RA-
DIT (Lin et al., 2023) proposed a dual instruction
tuning framework that jointly optimizes both the
LLM and retriever. Though useful, they struggle
with multiple retrievers.
The second approach aims to evaluate retrieved
documents utility by LLMs’s self-reflection capa-
bilities (Asai et al., 2024; Yan et al., 2024). Self-
RAG introduces reflection tokens that allow the
LLM to adaptively retrieve passages on-demand
and critique both the retrieved content and its own
generations. While effective in identifying valuable
documents, multiple LLM calls introduce substan-
tial computation overhead.
The third approach incorporates a reranker to re-
order retrieved documents, typically including the
open-source reranker BGE (Chen et al., 2024b) and
proprietary GTE-7B (Zhang et al., 2024). BGE is
a small encoder initially trained on over 300M text
pairs, then supervised fine-tuning on high-quality
labeled data, while GTE-7B trains a large long-
context LLM to learn the hybrid document repre-
sentations (both dense and sparse). However, BGE
is mainly trained to capture fine-grained semantic
relationships, which may fail to select truly helpful

documents, and GTE is computationally expensive
for practical deployment.
3 Method
In this section, we present InfoGain-RAG to ad-
dress the key challenges discussed earlier. Our
framework consists of two main components: (1)
Document Information Gain (DIG), a metric that
quantifies a document’s contribution to correct an-
swer generation by measuring changes in LLM’s
generation confidence scores, along with an effi-
cient pipeline for collecting high-quality training
data, and (2) a multi-task reranker that combines
document relevance classification and ranking ob-
jectives to optimize document selection. By in-
corporating these, our framework enables effec-
tive document selection without requiring multiple
LLM calls, making it both computationally effi-
cient and practical for real-world applications.
3.1 Document Information Gain
The core of InfoGain-RAG lies in quantifying each
document’s contribution to correct answer gener-
ation through calculating the information gain of
each retrieval. This section details our methodol-
ogy for computing DIG and utilizing it to build
high-quality training data. The complete data col-
lection pipeline is presented in Algorithm 1. To
compute DIG, we first propose a robust approach
for estimating LLM’s generation confidence, and
then use this estimation to measure the information
gain provided by each document.
Algorithm 1DIG Data Collection Pipeline
Require:Query setQ, Document corpusD, LLMϕ
Ensure:DIG datasetT
1:T ← ∅
2:foreach queryx∈ Qdo
3: Retrieve candidate documentsD xfromD
4: Get confidence pϕ(y|x) (defined in equation 2) without
documents
5:foreach docd∈D xdo
6: Get confidencep ϕ(y|x,d)with document
7: Calculate DIG (defined in equation 3)
8:T ← T ∪ {(x,d,DIG(d|x))}
9:end for
10:end for
11:returnT
3.1.1 Answer Generation Probability
A key challenge in computing DIG is estimating the
probability of a specific answer. A straightforward
way would be to multiply the probabilities of indi-
vidual tokens as the final confidence score. How-
ever, this approach faces two key challenges: First,it suffers from the length bias problem (Shi et al.,
2021) where longer sequences tend to receive lower
scores as any single low token probability severely
impacts the overall score. Second, treating all to-
kens equally fails to capture the strongest signal
for generation quality (Gangi Reddy et al., 2024)
which initial tokens often provide. To address these,
we propose a two-component approach:
Sliding Window Smoothing: To mitigate the
length bias problem, we implement a sliding win-
dow smoothing mechanism. For each token tiin
the answer sequence, its smoothed probability is
calculated as:
psmooth (ti) =1
Wi+⌊W/2⌋X
j=i−⌊W/2⌋p(tj)(1)
where Wis the window size and p(tj)represents
the original token probability, obtained by normal-
izing LLM logits (Yenduri et al., 2024).
Token Importance Weighting: It is reported
that initial tokens often carry stronger signals in
model generation(Gangi Reddy et al., 2024). Incor-
porating this observation, we apply higher weights
to the first ktokens when computing probability
scores, as they typically contain core semantic in-
formation for the response. The final formula is as
follows:
pϕ(y|x) =kY
i=1(psmooth(ti))ωi·α·|y|Y
j=k+1(psmooth(tj))1−α
(2)
where ωiare the importance weights for the first k
tokens, αis a weight hyper-parameter, and |y|is
the answer length.
3.1.2 Calculation of DIG
With a reliable approach to estimate answer gen-
eration probability, we now define the calculation
of DIG, as shown in Figure 1(NOTE part). Unlike
traditional relevance metrics that rely on lexical
overlap or semantic similarity, DIG directly mea-
sures how much a document improves the LLM’s
confidence in generating the correct answer.
Formally, given an LLM ϕ, a query x, and
its corresponding ground truth answer y, the
DIG for a document retrieved di(di∈ D,D=
{d1,d2, . . . ,d |D|})is defined as:
DIG(d i|x)def=p ϕ(y|x,d i)−p ϕ(y|x)(3)
where pϕ(y|x,d i)represents the model’s output
confidence with both the query and the document,
andp ϕ(y|x)is the query-only confidence.

STEP2:DocumentInformationGainDatasetCollectionforRerankerSTEP4:Inference with InfoGain-RAGNOTE:CalculationofDocumentInformationGain
RetrieverDocCandidatesDocCandidatesDIG>0DIG<	0
DIGDataset
Retriever
ProficientQueryChallengingQuery
Query
ProficientQueryChallengingQuery
Answer
LLM
QDALogits
QALogitsDIG
Query+Doc+Ans
Query+Ans
Query+Doc+DIG
BaseModelRerankerTrain
LLM
LLMLLM
Query
RetrieverReranker
LLMFilterDocCandidates
SelectedDocsSTEP1:QueryCategorizationSTEP3:RerankerTrainingFigure 1: Illustrations of InfoGain-RAG.STEP 1: Distinguish proficient queries from challenging ones;STEP 2:
Retrieve top-k documents for each query and calculate their DIG scores;STEP 3: Train the multi-task reranker;
STEP 4: Inference with InfoGain-RAG;NOTE: Calculation of DIG.
Based on above, we establish a data collection
pipeline that begins by categorizing queries based
on the model’s baseline performance without re-
trieved documents, shown in Figure 1 (STEP 1):
•Model-Proficient Queries: Queries that the
LLM can answer correctly using only its in-
herent knowledge (i.e., high pϕ(y|x) ). These
queries are particularly effective for identify-
ing noisy documents through DIG<0 , while
positive DIG samples are naturally rare since
external correct information adds little value
to already-known answers.
•Model-Challenging Queries: Queries that
the LLM shows low confidence without ex-
ternal information (i.e., low pϕ(y|x) ). These
queries facilitate us to identify helpful docu-
ments, as confidence increases (DIG>0).
Based on DIG, documents are categorized intothree groups (see Figure 5):
•DIG >0: Documents that enhance the
model’s confidence, containing relevant and
helpful information that should be prioritized
during reranking.
•DIG≈0: Documents that neither improve
nor diminish confidence and occur in two sce-
narios: (1) the document contains no mean-
ingful information for answering the query, or
(2) LLM has already mastered the required
knowledge during pre-training, making addi-
tional correct information unnecessary.
•DIG<0: Documents that reduce confidence
and contain misleading or contradictory infor-
mation that should be filtered out.
This categorization offers two key advantages:
1) quantitative measurement of document utility

through DIG scores, enabling both automatic iden-
tification of high-quality documents and precisely
filtering noise; and 2) fine-grained document prior-
itization through continuous DIG scores, which al-
lows optimal document ordering during inference.
By computing DIG across diverse query-
document pairs, we create a rich training dataset
capturing both absolute relevance and relative im-
portance of documents. This dataset serves as the
foundation for training our specialized reranker, as
detailed in the following section.
3.2 Multi-task Reranker
Building on DIG-scored training data collected
above, we propose a multi-task learning strategy to
train our reranker to select the most valuable docu-
ments for correct answer generation. The training
objective combines Cross-Entropy (CE) loss and
Margin loss to filter out noisy content and prioritize
highly effective documents based on DIG scores.
CE loss enables the model to distinguish between
helpful and noisy documents through binary clas-
sification, while margin loss optimizes document
ordering based on their DIG values. This unified
training approach enables our reranker to simulta-
neously learn discriminative document classifica-
tion and fine-grained ranking preferences, leading
to robust document selection for RAG.
3.2.1 Document Relevance Classification
The first task focuses on the relevance determina-
tion of the retrievals through binary classification.
Building upon the former collected data, we train
the reranker to distinguish documents that have
substantial contributions or potential harm to an-
swer generation. Specifically, we employ CE loss
to optimize the reranker θto achieve this objective:
min
θLCE=1
NNX
i=1h
−yilog(p(x i,di))
−(1−y i) log(1−p(x i,di))i
s.t.p(x i,di)∈[0,1], y i∈ {0,1},∀i= 1, . . . , N
(4)
Here, p(xi,di)represents the predicted probabil-
ity that document diwill achieve a positive DIG
score for query xi. The label yiis determined by
our previously computed DIG scores, with yi= 1
for documents whose score is above upper decision
boundary b1andyi= 0for those below lower deci-
sion boundary b2. These thresholds effectively sep-
arate helpful documents from harmful ones. Thesehyper-parameters selection will be detailed in the
experiment section. This classification-based learn-
ing not only helps identify useful documents but
also facilitates better learning of relative document
ordering through the joint training process.
3.2.2 Document Ranking Optimization
The second task focuses on learning relative doc-
ument importance through pairwise comparison.
Inspired by Circle Loss (Sun et al., 2020), we in-
troduce a margin-based learning objective that ex-
plicitly models the relative ordering of documents
based on their DIG values. Given a query, this
objective constrains the maximum score of neg-
ative query-document pairs to be lower than the
minimum score of positive pairs:
min
θLMargin = [max (s n)−min (s p)]+
with[x] += max(x,0)(5)
where snandspdenote scores for pairs with
DIG values above b1and below b2respectively,
andθdenotes reranker. To involve all samples in
one process, we employ the LogSumExp function
to approximate extremal value:
max{x 1, . . . , x n}= log (exp (max (x i)))≈LSE(x n),
min{x 1, . . . , x n}=−max{−x 1, . . . ,−x n}
≈ −LSE(−x n)
(6)
where LSE(x n)is the LogSumExp function,
with detailed derivation provided in Appendix B.1.
Substitute the LogSumExp approximations into
equation (5) and yield:
min
θLMargin≈[LSE(γ(s n))−(−LSE(−γ(s p)))]+
≈log"
1 +KX
i=1LX
j=1exp
γ
sj
n−si
p#
(7)
where γis a scaling factor controlling the contribu-
tion of non-extremal pairs and KandLdenote the
number of positive and negative document pairs.
Detailed derivation is provided in Appendix B.2.
Softplus is used to smooth the ReLU function:
Softplus(x) = log (1 +ex)≈[x] + (8)
By integrating CE loss and margin loss with
weight β, our multi-task training objective enables
the reranker to jointly optimize DIG and inter-
document relationships:
Ltotal=βL CE+ (1−β)L Margin (9)
This unified approach produces a robust reranker
that considers both absolute document relevance

Figure 2: The relationship between the hyper-parameter
βand accuracy on TriviaQA, LLaMA3.1-8B achieves
optimum atβ= 0.8, while Qwen2.5-14B at 0.7.
and relative ordering preferences within the re-
trieved documents, leading to more effective docu-
ment reranking and filtering for RAG systems (see
Figure 2 for empirical study on balancing these two
objectives via hyper-parameterβ).
During inference, InfoGain-RAG enhances naive
RAG pipelines by adding an efficient document
reranking step while maintaining low computa-
tional overhead, as illustrated in Figure 1 (STEP
4). The process begins with document retrieval, fol-
lowed by our trained reranker which both reorders
documents and filters out those below a quality
threshold. The filtered and reranked documents are
then passed to LLM for final answer generation
while only calling once.
4 Experiment
We evaluate InfoGain-RAG in four experiment se-
ries. First, we compare it with the open-source
reranker of BGE-Reranker-Large (Chen et al.,
2024b) trained on 300M samples, and the state-of-
the-art proprietary reranker GTE-7B (Zhang et al.,
2024). Second, we compare with retriever opti-
mization approaches like RePlug (Shi et al., 2023)
and RADIT (Lin et al., 2023), and self-reflection
approaches like Self-RAG (Asai et al., 2024) and
CRAG (Yan et al., 2024). Third, we test InfoGain-
RAG on combined documents retrieved from Con-
triever (Lei et al., 2023), BM25 (Robertson and
Zaragoza, 2009) and DPR (Karpukhin et al., 2020)
to demonstrate its capability to handle multiple
retrievers. Last, several ablation studies are con-
ducted to verify the effectiveness from different
aspects. The datasets and models we used are pub-
licly accessible.4.1 Setup
Tasks and Datasets.We experiment on two tasks
of four English datasets: (1)open-domain ques-
tion answering, including TriviaQA (Joshi et al.,
2017), NaturalQA (Kwiatkowski et al., 2019), and
PopQA (Mallen et al., 2023); (2)fact verification,
FM2 (Eisenschlos et al., 2021). We use the Decem-
ber 2018 Wikipedia dump (Karpukhin et al., 2020)
as the retrieval corpus.
Models and Metrics.All evaluations are con-
ducted across both proprietary LLMs (GPT-
4o-20241120, ChatGPT-20240125, and Claude-
3.5-Sonnet-20241022) and open-source models
(LLaMA3.1, Qwen2, Gemma2, DeepSeek-V3, and
DeepSeek-R1). We adpot Exact Match (EM) ac-
curacy (Rajpurkar et al., 2016) as the metric. EM
provides a strict evaluation of response accuracy
while accommodating multiple correct answer for-
mats, as it compares the model outputs with all
valid answers provided.
Implementation Details.We sample 110K
queries from TriviaQA dataset (with train-test
overlap removed) and calculate DIG scores for
all collected<query, answer, document>triplets
using Qwen2.5-7B. The scoring results in three
categories: 70K triplets with high positive gain
(>b1= 0.5 ), 150K triplets with negative gain
(<b2=−0.2 ), and 1200K triplets showing neg-
ligible information gain ( −0.05∼0.05 ). From
these scored triplets, we create a unified training
dataset of 88K samples through different sampling
strategies for each loss: for CE loss, we sample bal-
anced query-document pairs with equal numbers of
positive and negative samples (68K), while for mar-
gin loss, we sample query-document groups (34K)
where each query is paired with 3-5 high-DIG doc-
uments and augmented with additional negative
and negligible documents.
For experimental settings, we implement our
reranker using RoBERTa-large (Liu et al., 2019)
to rerank the top 100 documents retrieved by Con-
triever (Lei et al., 2023). Our reranker is trained
on an A800 GPU using Adam optimizer with a
learning rate of 5e-6 and βvalue of 0.75. For DIG
calculation, we set importance weights ωito 0.8
for the first k= 3 tokens and use α= 0.6 for bal-
ancing token probabilities. During inference, we
select the top 4 documents and employ a document
filtering threshold of 0.2 while retaining all can-
didates that exceed this threshold. This threshold

Table 1: Performance Comparison of RAG Reranking approaches with single-retriever (Contriever).
ModelTriviaQA NaturalQA PopQA FM2
RAG BGE(550M)§GTE(7B)3Ours(355M) RAG BGE(550M)§GTE(7B)3Ours(355M) RAG BGE(550M)§GTE(7B)3Ours(355M) RAG BGE(550M)§GTE(7B)3Ours(355M)
Qwen2.5-0.5B 48.5% 48.6% 49.5%55.8% 22.5% 27.3% 29.5%35.3% 26.5% 35.7% 35.3%36.5% 53.0% 52.3% 55.6%58.7%
Qwen2.5-1.5B 50.4% 59.1% 63.3%66.3% 30.7% 39.5% 45.2%47.2% 31.3% 41.3%44.2%43.0% 69.1% 69.5% 71.1%73.9%
Qwen2.5-7B 52.9% 67.0% 69.5%72.1% 36.3% 41.8% 49.9%53.6% 32.4% 43.4% 43.7%47.6% 72.5% 74.5% 77.8%79.9%
Qwen2.5-14B 56.1% 68.4% 71.1%72.9% 36.0% 42.7% 52.5%53.8% 31.8% 44.1% 45.9%49.4% 72.6% 75.7% 76.4%79.4%
Qwen2.5-32B 58.7% 70.3% 72.0%74.7% 36.4% 42.1% 53.7%55.9% 32.3% 45.5% 48.1%50.5% 73.7% 75.6% 79.0%81.2%
Qwen2.5-72B 59.9% 70.6% 73.4%76.3% 40.3% 44.9% 53.9%58.1% 34.0% 44.8% 49.5%51.4% 73.6% 75.9% 80.4%83.4%
Qwen3-8B 57.9% 67.6% 71.1%72.3% 34.0% 41.5% 50.9%52.6% 32.1% 43.6% 46.5%49.1% 71.4% 76.1%80.9%80.0%
LLaMA3.1-8B 55.1% 65.5% 67.5%70.4% 33.6% 39.4% 46.9%50.7% 31.7% 41.3% 44.6%47.1% 74.3% 77.6% 79.5%81.2%
LLaMA3.1-70B 54.5% 67.9% 67.4%71.3% 35.1% 39.9% 48.6%51.6% 30.4% 43.0% 47.2%47.6% 77.0% 79.5% 81.1%82.4%
LLaMA3.1-405B 56.7% 69.2% 73.8%74.6% 35.8% 41.5% 52.3%53.3% 30.5% 43.4% 47.3%49.5% 75.9% 77.6% 80.6%83.1%
Gemma-2-9B 54.3% 64.4% 69.0%71.3% 34.3% 39.6% 44.6%56.6% 31.4% 43.9% 45.5%49.3% 75.4% 78.5% 80.9%81.5%
Gemma-2-27B 59.6% 68.5% 70.9%74.3% 37.6% 42.3% 51.5%57.4% 33.1% 45.4% 49.4%50.3% 76.3% 78.4%82.1%81.6%
DeepSeek-V3 56.0% 68.0% 72.0%73.4% 37.6% 42.5% 50.7%55.1% 30.8% 43.4% 48.6%49.7% 75.7% 77.5% 78.6%80.2%
DeepSeek-R1 60.4% 71.7%75.7%75.2% 40.8% 44.8% 56.8%58.8% 31.2% 45.3% 51.1%51.6% 77.1% 78.9% 80.3%83.8%
Claude-Sonnet†54.5% 68.4% 70.7%73.9% 36.7% 41.1% 52.4%55.2% 31.6% 43.1% 48.9%50.4% 76.0% 78.4%80.9%80.8%
ChatGPT‡62.0% 69.0% 72.1%74.1% 37.1% 42.7%55.9%54.5% 32.0% 43.5% 48.0%48.5% 71.9% 73.2% 75.0%75.3%
GPT-4o⋆57.2% 69.2% 74.4%75.4% 37.5% 41.6% 53.1%57.2% 31.4% 43.3% 48.3%49.2% 76.6% 75.1% 78.8%82.2%
GPT-4.1§58.6% 70.8% 76.1%76.4% 35.1% 41.7% 55.6%56.2% 30.9% 45.4%51.3%50.4% 75.2% 77.1% 76.4%80.4%
§BGE-Reranker-Large (550M).3Proprietary GTE-Reranker (7B).†241022 version.‡240125 version.⋆241120 version.§
20250414 version.
is slight different from b1, as the the addition of
margin loss would widen the score distribution of
valid samples. Notably, to ensure minimal context
for generation, we retain at least 2 documents if
fewer exceed the filtering threshold.
4.2 Results
We first present InfoGain-RAG’s performance with
single retriever across different LLMs and bench-
marks, comparing it with naive RAG and reranking
approaches. We then show its effectiveness in mul-
tiple retriever settings. Finally, we demonstrate
our method’s advantages over self-reflection and
retriever-optimization approaches.
Comparison to Reranking approaches with Sin-
gle Retriever.Table 1 compares InfoGain-RAG
(355M) against naive RAG, BGE-Reranker (550M)
and GTE-Reranker (7B, SOTA) across different
models and datasets. As shown in the results,
InfoGain-RAG substantially improves over naive
RAG and BGE-Reranker, while surpassing the far
larger GTE-Reranker in most cases. On TriviaQA,
for instance, DeepSeek-V3 achieves 72.0% with
GTE-Reranker and 73.4% with InfoGain-RAG,
while Qwen2.5-72B reaches 76.3% with InfoGain-
RAG, surpassing naive RAG by 16.4%, BGE-
Reranker by 5.7%, and GTE-Reranker by 2.9%.
Moveover, these improvements hold across both
model scales and families - from smaller models
like Qwen2.5-1.5B (+15.9% over naive RAG) to
larger ones like LLaMA3.1-405B (+17.9%).
Trained on TriviaQA, InfoGain-RAG demon-
strates strong generalization ability across different
datasets and tasks. It improves Qwen2.5-72B’s
accuracy on NaturalQA by 17.8% and PopQA by
17.4% over naive RAG, with particularly notable
Figure 3: Performance comparison of Qwen2.5-7B
across different datasets with single retriever and multi-
ple retrievers.
gains on FM2 from 73.6% to 83.4%.
In particular, our reranker achieves these results
with just 88K training samples and merely 335M
parameters, compared to BGE-Reranker’s 300M
samples and GTE-Reranker’s 7B parameters(see
Appendix C for comparisons with the GTE family).
Comparison to Reranking approaches with Mul-
tiple Retrievers.InfoGain-RAG maintains con-
sistent superiority with multiple retrievers.
As shown in Figure 3, our reranker achieves the
best performance on all four tasks. Specifically, it
improves by 9.9% over BGE-Reranker on Natu-
ralQA and by 4.9% over GTE-Reranker on PopQA.
Additionally, we observe that all rerankers show im-
provements in the multi-retriever setting compared
to the single-retriever setting. Notably, our method
achieves the largest performance gains (when com-
paring multi-retriever to single-retriever settings)
on most tasks, with an average improvement of
3.8%. This clearly demonstrates the superior effec-
tiveness of our reranker in multi-retriever scenarios.
Comparison with Self-Reflection and Retriever-
Optimization approaches.As shown in Fig-

ure 4, we evaluate InfoGain-RAG against two types
of RAG approaches. For self-reflection, our ap-
proach outperforms both Self-RAG(Asai et al.,
2024) and CRAG(Yan et al., 2024). With LLaMA2-
13B as the base model, InfoGain-RAG achieves
76.2% accuracy on TriviaQA and 51.9% on Natu-
ralQA, surpassing Self-RAG (69.3%, 49.5%) and
CRAG (74.5%, 48.2%) while avoiding multiple
LLM inference calls. For retriever-optimization,
InfoGain-RAG shows substantial improvements
using LLaMA-65B, reaching 78.2% on TriviaQA
and 54.3% on NaturalQA. This outperforms both
RePlug(Shi et al., 2023) (74.9%, 42.3%) and RA-
DIT(Lin et al., 2023) (75.1%, 43.9%).
Figure 4: Performance Comparison with self-reflection
(7B, 13B) and retriever-optimization (65B) approaches
on TriviaQA (a) and NaturalQA (b). We strictly fol-
lowed the experimental settings of each baseline ap-
proach for fair comparison.
4.3 Ablation Study
In this section, we conduct comprehensive abla-
tion studies to systematically evaluate the critical
components across InfoGain-RAG: 1) examining
whether using different base models to generate
DIG data will affect the final effect, 2) verifying
whether the multi-task learning strategy can bring
greater improvement compared to each individual
task, and 3) assessing the impact of document fil-
tering during inference.
LLM-agnostic DIG-data Collection.Table 2
demonstrates that InfoGain-RAG’s performance re-
mains consistent regardless of which LLM is used
for DIG data collection. Despite the changes in
the DIG scores of each model due to factors such
as structure and size, the trained reranker achieves
similar accuracy on TriviaQA. This performance
shows that InfoGain-RAG can identify the intrinsic
query-document correlations independent of the
LLM used for data collection, validating its robust-
ness as a general framework.
Single or Multi-task Reranker Training.Ta-
ble 3 compares the performance differences of sin-Table 2: Compared results of rerankers trained using
DIG scores from different base LLMs on TriviaQA.
Model RAGOurs
(DIG-Qwen)Ours
(DIG-LLaMA)
Qwen2.5-7B 52.9%72.1%68.8%
Qwen2.5-14B 56.1% 72.9%74.2%
Qwen2.5-72B 59.9%76.3%75.0%
LLaMA3.1-8B 55.1% 70.4%72.1%
LLaMA3.1-70B 54.5%71.3%70.2%
LLaMA3.1-405B 56.7%74.6%73.0%
gle CE or Margin task to the multi-task training.
We can see that the combined strategy consistently
outperforms individual loss across two types of
models. For example, Qwen2.5-72B can get an
accuracy of 76.8% with the multi-task training on
TriviaQA, but only 73.0% for CE and 71.4% for
margin loss. The large improvement demonstrates
that the absolute relevance judgments can be com-
bined with the relative rankings to achieve more
robust document selection.
Table 3: Performance differences of single CE or Mar-
gin task to the multi-task training across models. The
testings is conducted on TriviaQA.
ModelOurs
(CE loss)Ours
(Margin loss)Ours
(Multi-loss)
Qwen2.5-7B 67.6% 68.2%71.8%
Qwen2.5-14B 70.1% 67.9%72.7%
Qwen2.5-72B 73.0% 71.4%76.8%
LLaMA3.1-8B 68.2% 65.3%70.7%
LLaMA3.1-70B 69.5% 67.1%71.4%
LLaMA3.1-405B 73.6% 70.8%74.2%
Document Filtering during Inference.In ta-
ble 4 we test the effectiveness of document filtering
during inference with the threshold of 0.2. Here,
non-filtering means all retrieved documents are
ranked without being filtered. It can be observed
that peformances are better with filtering than non-
filtering. For instance, Qwen2.5-72B improves
from 73.6% to 76.8%, and LLaMA3.1-405B gains
from 71.2% to 74.6%. These observations jointly
confirm that identifying and removing potentially
noisy contents is beneficial for final performance.
5 Conclusion
In this paper, we present a novel framework
InfoGain-RAG to address the critical challenge of
RAG about filtering out semantically misaligned
and noisy retrieved content. By introducing a
principled DIG metric coupled with a multi-task

Table 4: Performance validations of retrieved document
filtering operations. All results are tested on TriviaQA.
Model RAGOurs
(Non-filtering)Ours
(Filtering)
Qwen2.5-7B 52.9% 68.2%71.8%
Qwen2.5-14B 56.1% 71.8%72.9%
Qwen2.5-72B 59.9% 73.6%76.3%
LLaMA3.1-8B 55.1% 67.8%70.4%
LLaMA3.1-70B 54.5% 68.2%71.3%
LLaMA3.1-405B 56.7% 71.2%74.6%
reranker learning strategy, InfoGain-RAG effec-
tively quantifies document utility and optimizes
both filtering and reranking processes. Compre-
hensive experiments across proprietary and open-
source LLMs demonstrate substantial improve-
ments across multiple benchmarks while maintain-
ing lower computational overhead compared to ex-
isting approaches. The effectiveness and economic
applicability of the framework suggest the feasi-
bility of InfoGain-RAG, as it can offer a reliable
solution for RAG in partical application.
6 Limitation
While InfoGain-RAG demonstrates strong perfor-
mance improvements, several limitations warrant
discussion. The current implementation has only
been tested on text modalities, though it is the-
oretically extensible to other modalities such as
visual or code data. Computational constraints
limit the reranker to 335M parameters rather than
larger models (7B+), which could offer better per-
formance but may significantly increase inference
latency in practical applications. Additionally, the
DIG metric, while effective, cannot distinguish fac-
tual inaccuracies in retrieved documents, which
may require an extra module to address this issue.
We hope more efforts can be devoted to addressing
these limitations collaboratively in the future.
References
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama
Ahmad, Ilge Akkaya, et al. 2024. GPT-4 technical
report.arXiv preprint arXiv:2303.08774.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-rag: Learning to re-
trieve, generate, and critique through self-reflection.
InProceedings of the 12th International Conference
on Learning Representations (ICLR), Vienna, Aus-
tria.Jinze Bai, Shuai Bai, Yunfei Chu, et al. 2023. Qwen
technical report.arXiv preprint arXiv:2309.16609.
Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng,
Jianfeng Gao, Xiaodong Liu, Rangan Majumder,
Andrew McNamara, Bhaskar Mitra, Tri Nguyen,
Mir Rosenberg, Xia Song, Alina Stoica, Saurabh
Tiwary, and Tong Wang. 2018. Ms marco: A human
generated machine reading comprehension dataset.
Preprint, arXiv:1611.09268.
Tom Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared D Kaplan, et al. 2020. Language
models are few-shot learners.Advances in neural
information processing systems (NeurIPS), 33:1877–
1901.
Chia-Yuan Chang, Zhimeng Jiang, Vineeth Rakesh,
et al. 2024. Main-rag: Multi-agent filtering
retrieval-augmented generation.arXiv preprint
arXiv:2501.00332.
Ben Chen, Huangyu Dai, Xiang Ma, Wen Jiang, and
Wei Ning. 2024a. Robust interaction-based relevance
modeling for online e-commerce search. InMachine
Learning and Knowledge Discovery in Databases.
Applied Data Science Track: European Conference,
ECML PKDD 2024, Berlin, Heidelberg.
Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu
Lian, and Zheng Liu. 2024b. Bge m3-embedding:
Multi-lingual, multi-functionality, multi-granularity
text embeddings through self-knowledge distillation.
arXiv preprint arXiv:2402.03216.
Angelika Eisele-Metzger, Judith-Lisa Lieberum,
Markus Toews, Waldemar Siemens, Felix Heilmeyer,
Christian Haverkamp, Daniel Boehringer, and Joerg J
Meerpohl. 2024. Exploring the potential of claude 2
for risk of bias assessment: Using a large language
model to assess randomized controlled trials with
rob 2.medRxiv, pages 2024–07.
Julian Eisenschlos, Bhuwan Dhingra, Jannis Bulian,
Benjamin Börschinger, and Jordan Boyd-Graber.
2021. Fool me twice: Entailment from Wikipedia
gamification. InProceedings of the 2021 Conference
of the North American Chapter of the Association for
Computational Linguistics (NAACL), pages 352–365.
Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang,
Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing
Li. 2024. A survey on rag meeting llms: Towards
retrieval-augmented large language models.Preprint,
arXiv:2405.06211.
Revanth Gangi Reddy, JaeHyeok Doo, Yifei Xu,
Md Arafat Sultan, Deevya Swain, Avirup Sil, and
Heng Ji. 2024. FIRST: Faster improved listwise
reranking with single token decoding. InThe Asso-
ciation for Computational Linguistics: ACL 2024,
pages 8642–8652, Miami, Florida, USA.
Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen.
2023. Enabling large language models to generate

text with citations. InProceedings of the 2023 Con-
ference on Empirical Methods in Natural Language
Processing, Singapore.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-hop
QA dataset for comprehensive evaluation of reason-
ing steps. InProceedings of the 28th International
Conference on Computational Linguistics, Barcelona,
Spain (Online). International Committee on Compu-
tational Linguistics.
Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke
Zettlemoyer. 2017. TriviaQA: A large scale distantly
supervised challenge dataset for reading comprehen-
sion. InProceedings of the 55th Annual Meeting of
the Association for Computational Linguistics (ACL),
pages 1601–1611, Vancouver, Canada.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. InProceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pages 6769–6781,
Virtual.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, et al. 2019.
Natural questions: a benchmark for question answer-
ing research.Transactions of the Association for
Computational Linguistics, 7:453–466.
Yibin Lei, Liang Ding, Yu Cao, Changtong Zan, An-
drew Yates, and Dacheng Tao. 2023. Unsupervised
dense retrieval with relevance-aware contrastive pre-
training. InFindings of the Association for Computa-
tional Linguistics: ACL 2023, pages 10932–10940,
Toronto, Canada.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, et al. 2020. Retrieval-
augmented generation for knowledge-intensive NLP
tasks. InAdvances in Neural Information Processing
Systems (NeurIPS), volume 33, pages 9459–9474,
Virtual.
Jiarui Li, Ye Yuan, and Zehua Zhang. 2024. En-
hancing llm factual accuracy with rag to counter
hallucinations: A case study on domain-specific
queries in private knowledge-bases.arXiv preprint
arXiv:2403.10446.
Xi Victoria Lin, Xilun Chen, Mingda Chen, Weijia Shi,
Maria Lomeli, Rich James, Pedro Rodriguez, Jacob
Kahn, Gergely Szilvasy, Mike Lewis, et al. 2023.
Ra-dit: Retrieval-augmented dual instruction tuning.
arXiv preprint arXiv:2310.01352.
Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, et al.
2024. Deepseek-v3 technical report.arXiv preprint
arXiv:2412.19437.
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, et al. 2023. Lost in the
middle: How language models use long contexts.Transactions of the Association for Computational
Linguistics (TACL), 12:157–173.
Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Man-
dar Joshi, et al. 2019. RoBERTa: A robustly opti-
mized BERT pretraining approach.arXiv preprint
arXiv:1907.11692.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,
Daniel Khashabi, and Hannaneh Hajishirzi. 2023.
When not to trust language models: Investigating
effectiveness of parametric and non-parametric mem-
ories. InProceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (ACL),
pages 9802–9822, Toronto, Canada.
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and
Percy Liang. 2016. SQuAD: 100,000+ questions for
machine comprehension of text. InProceedings of
the 2016 Conference on Empirical Methods in Natu-
ral Language Processing, pages 2383–2392, Austin,
Texas. Association for Computational Linguistics.
Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay,
Amnon Shashua, et al. 2023. In-context retrieval-
augmented language models.Transactions of the
Association for Computational Linguistics, 11:1316–
1331.
Stephen Robertson and Hugo Zaragoza. 2009. The
probabilistic relevance framework: Bm25 and be-
yond.Foundations and Trends in Information Re-
trieval, 3(4):333–389.
Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon
Seo, Rich James, et al. 2023. Replug: Retrieval-
augmented black-box language models.arXiv
preprint arXiv:2301.12652.
Xuewen Shi, Heyan Huang, Ping Jian, and Yi-Kun
Tang. 2021. Reducing length bias in scoring neu-
ral machine translation via a causal inference method.
InChinese Computational Linguistics, pages 3–15,
Cham. Springer International Publishing.
Yifan Sun, Changmao Cheng, Yuhan Zhang, Chi Zhang,
Liang Zheng, Zhongdao Wang, and Yichen Wei.
2020. Circle loss: A unified perspective of pair simi-
larity optimization. InProceedings of the IEEE/CVF
conference on computer vision and pattern recogni-
tion, pages 6398–6407.
Gemma Team, Morgane Riviere, Shreya Pathak,
Pier Giuseppe Sessa, et al. 2024. Gemma 2: Im-
proving open language models at a practical size.
arXiv preprint arXiv:2408.00118.
Hugo Touvron, Louis Martin, Kevin Stone, et al. 2023.
Llama 2: Open foundation and fine-tuned chat mod-
els.arXiv preprint arXiv:2307.09288.
Tianyu Wu, Shizhu He, Jingping Liu, Siqi Sun, Kang
Liu, et al. 2023. A brief overview of ChatGPT: The
history, status quo and potential future development.
IEEE/CAA Journal of Automatica Sinica, 10(5):1122–
1136.

Shiqi Yan, Jiachen Gu, Zhuyun, and Zhenhua Ling.
2024. Corrective retrieval augmented generation.
arXiv preprint arXiv:2401.15884.
Gokul Yenduri, M. Ramalingam, G. Chemmalar
Selvi, Y . Supriya, Gautam Srivastava, Praveen Ku-
mar Reddy Maddikunta, G. Deepti Raj, Rutvij H.
Jhaveri, B. Prabadevi, Weizheng Wang, Athanasios V .
Vasilakos, and Thippa Reddy Gadekallu. 2024. Gpt
(generative pre-trained transformer)— a comprehen-
sive review on enabling technologies, potential appli-
cations, emerging challenges, and future directions.
IEEE Access, 12:54608–54649.
Xin Zhang, Yanzhao Zhang, Dingkun Long, Wen Xie,
Ziqi Dai, Jialong Tang, Huan Lin, Baosong Yang,
Pengjun Xie, Fei Huang, Meishan Zhang, Wenjie
Li, and Min Zhang. 2024. mGTE: Generalized long-
context text representation and reranking models for
multilingual text retrieval. InProceedings of the
2024 Conference on Empirical Methods in Natural
Language Processing: Industry Track, pages 1393–
1412, Miami, Florida, US.

A DIG Cases
QueryInwhatcitywasAbrahamRaimbachborn?DocumentAthisdeath,heheldagoldmedalawardedforhis"VillagePoliticians"attheParisExhibitionof1814.HewaselectedcorrespondingmemberoftheAcadémiedesBeaux-Artsin1835.HeisburiedinStMary'sChurchyard,Hendon.AbrahamRaimbachAbrahamRaimbach(16February1776inLondon17January1843),wasanEnglishline-engraverofSwissdescent.HewasborninCecilCourtintheWestEndofLondon.EducatedatArchbishopTenison'sLibrarySchool,hewasapprenticedtotheengraverJ.Hallfrom1789to1796.
QueryWhatisthecapitalofIceland?DocumentManyfjordspunctuateIceland's4,970-km-long(3,088-mi)coastline,whichisalsowheremostsettlementsaresituated.Theisland'sinterior,theHighlandsofIceland,isacoldanduninhabitablecombinationofsand,mountains,andlavafields.ThemajortownsarethecapitalcityofReykjavík,alongwithitsoutlyingtownsofKópavogur,Hafnarfjörður,andGarðabær,nearbyReykjanesbærwheretheinternationalairportislocated,andthetownofAkureyriinnorthernIceland.TheislandofGrímseyontheArcticCirclecontainsthenorthernmosthabitationofIceland,whereasKolbeinseycontainsthenorthernmostpointofIceland.
QueryWhoistheauthorofB²FHpaper?DocumentPhases,ashellmodelthatwasnecessaryforHoyle's1954picturetoworkassimultaneousejectionoftheabundancesfromeachburningphase.UnderstandingthisculturalrevolutionofcomputingtakesonefarinunderstandingwhyHoyle(1954)wasforgottenandB2FHappearedtohavebeentheworkthatfoundedstellarnucleosynthesis,asmanyevenclaimed.B2FHpaper,namedaftertheinitialsoftheauthorsofthepaper,MargaretBurbidge,GeoffreyBurbidge,WilliamA.Fowler,andFredHoyle,isalandmarkpaperontheoriginofthechemicalelementspublishedin"ReviewsofModernPhysics"in1957.
QueryInwhatcitywasGiulioBisegniborn?DocumentGiuliotriestoscamperbackdowntheledge,butfallsandfractureshisankle.ThebossgiveschaseandGiuliolimpstohismotorbike,barelyescaping.Giulio'smothertakeshimtothedoctorwhotellshimhe'llhavetobeinacastforseveralweeks.Ariannacomesovertohisplacetohelphimmend,butwhenhestartstellingherwhathappenedandhistheoryofhowFedericawantsSashatokillherboss,Ariannasayshe'scrazierthaneverandstormsout,sayingsheneverwantstoseehimagain.
QueryInwhatcitywasGloriaPorrasVallesborn?Documentherfatherworkingasatailorandhermother,ahousewife.Shewastaughtvaluesofcleanlinessandaccountability.Glorialivedwithhergrandmotherafterherparentsdivorced.Gloria‘sgrandmothertoldherstoriesabouthergreat-grandfatherwhowashunginthemountainsofMinasGerais.Daughterofaslavemother,hergrandmotherwasalsoabeneficiaryofthe1871LawofFreeBirthand,thus,bornfree.HergrandmothertaughtherthatsheneededtoworktobefreeandGloriadecidedtofocusoncombattingracialprejudice.Gloriawasoncemarriedbutseparatedbecauseshedidn’twanttolive.
QueryWhatgenreisEnter?Documentonetypeofstorybest.Inlaterperiodsgenresproliferatedanddevelopedinresponsetochangesinaudiencesandcreators.Genrebecameadynamictooltohelpthepublicmakesenseoutofunpredictableart.Becauseartisoftenaresponsetoasocialstate,inthatpeoplewrite/paint/sing/danceaboutwhattheyknowabout,theuseofgenreasatoolmustbeabletoadapttochangingmeanings.Genresuffersfromtheillsofanyclassificationsystem.Ithasbeensuggestedthatgenresresonatewithpeoplebecauseofthefamiliarity,theshorthandcommunication.(b)
(c)
QueryInwhatcitywasAkiHataborn?DocumentHehasrecentlybeguntobringbackthe"GuitarZamurai"charactersporadically,toplayonitsnostalgicappeal.In2008,heformedacliqueofotheronehitwondercomediansonthequiz/varietyshow,Quiz!HexagonIIcalled"Ippatsuya2008"(Ұൃ԰2008).YokuHataYōkuHata(೾ాཅ۠,"HataYōku",realname:AkiraHada(೾ాߊ,"HadaAkira"),bornJune5,1975inShimonoseki,YamaguchiPrefecture)isastandupcomedianinJapan.Herosetopopularityin2004withhischaracter"TheGuitarZamurai(Samurai)"(Ϊλʔࣆ)ontheprogram"TheGodofEntertainment"(Τϯλͷਆ༷).
QueryWhowastheproducerofTheMist?DocumentBecauseIgrewuplisteningtohis[Alan's]musicandneverthoughtthatonedayIwouldhavesuchafantasticopportunityofmeetinghim.AlanandIspokealot,especiallyaboutthebusinesssideandinnerrelationsoftheindustry,whichwasincrediblyvaluableformeatthatearlystageofmycareer.Ostinelli'ssoundtrackfor"TheMist"wasreleasedin2017byBMGRecords.TherecordcontainsexclusivelyOstinelli'sscorefortheshow.BasedontheStephenKing'snovellaofthesamename,"TheMist"hasbeenreimaginedfortelevisionbyChristianTorpeandstarsFrances
Query2011ladygagaalbumthathasedgeofglory?DocumentFemaleVideoandBestVideowithaSocialMessageawardsatthe2011MTVVideoMusicAwards.Inthefollowingvideo,"Judas",sheportraysMaryMagdalene,andNormanReedusplaysthetitlerole.Thevideofor"TheEdgeofGlory"consistsmostlyofinterchangingshotsofGagadancingandsingingonthestreetandwasconsideredthesimplestofhercareer.Inthesameyear,shereleased"YouandI",whichfocusesonhertryingtogetherboyfriendbackinNebraska.ShealsointroduceshermalealteregoJoCalderoneinthevideo.(a)
Figure 5: Retrieved documents of which DIG > 0 (a), DIG≈0 (b), and DIG < 0 (c) for the given query.

QueryWhowastheproducerofTheImitationGame?CorrectAnswerTeddySchwarzmanLLMAnswerwithRetrievedDocumentsTeddySchwarzmanRerankedDocumentsbyInfoGain-RAGReranker[Document1]TeddySchwarzmanproducedTheImitationGamethroughhisproductioncompanyBlackBearPictures,whowonacompetitivebidagainstNoraGrossmanandIdoOstrowskywhowantedtoacquirethescript."TheImitationGame"receivedmanyaccolades,includingAcademyAwardandBAFTAAwardnominationsforBestPictureandBestBritishFilm,respectively.Schwarzman,GrossmanandOstrowskywerealsonominatedforaProducersGuildofAmericaAward.SchwarzmanmarriedEllenMarieZajac,aNewYorklawyerwhomhemetatDukeUniversity,inNovember2007inMontegoBay,Jamaica.Theyhavethreechildren.TeddySchwarzmanEdwardFrank"Teddy"Schwarzman(bornMay29,1979)isanAmericanfilmproducerandformercorporatelawyer.[Document2]TeddySchwarzmanwhowantedtoacquirethescript."TheImitationGame"receivedmanyaccolades,includingAcademyAwardandBAFTAAwardnominationsforBestPictureandBestBritishFilm,respectively.Schwarzman,GrossmanandOstrowskywerealsonominatedforaProducersGuildofAmericaAward.SchwarzmanmarriedEllenMarieZajac,aNewYorklawyerwhomhemetatDukeUniversity,inNovember2007inMontegoBay,Jamaica.Theyhavethreechildren.TeddySchwarzmanEdwardFrank"Teddy"Schwarzman(bornMay29,1979)isanAmericanfilmproducerandformercorporatelawyer.Heisthefounder,presidentandchiefexecutiveofBlackBearPictures.[Document3]TheUSdistributorTWCstatedthatthefilmwouldinitiallydebutinfourcinemasinLosAngelesandNewYork,expandingtosixnewmarketson12DecemberbeforebeingreleasednationwideonChristmasDay."TheImitationGame"wasreleasedon31March2015intheUnitedStatesintwoformats:aone-discstandardDVDandaBlu-raywithadigitalcopyofthefilm."TheImitationGame"grossed$91.1millioninNorthAmericaand$142.4millioninotherterritoriesforaworldwidetotalof$233.5million,againstabudgetof$14million.Itwasthetop-grossingindependentfilmdistributedbySchwarzman.[Document4]TheImitationGame(play)TheImitationGameisatelevisionplaywrittenbyIanMcEwananddirectedbyRichardEyre,aBBC"PlayforToday",firstbroadcaston26April1980.Itis1940inFrintonand19-year-oldCathyRaineturnsdownajobatthelocalmunitionsfactoryand,muchtotheconsternationofherparentsandboyfriendTony,joinstheATS.Sheisassignedtoawirelesslisteningstation,transcribingEnigmacodedmorsetransmissionsfromNaziGermanyandmakesfriendswithMary.
QueryWhowastheproducerofTheImitationGame?CorrectAnswerTeddySchwarzmanLLMAnswerwithRetrievedDocumentsTheWeinsteinCompanyRerankedDocumentsbyBGEReranker[Document1]Adjustedforinflation,theImitationGameoutperformedtheWeinsteinCompany'sownOscar-winningfilms"TheKing'sSpeech"($88,863in2010)and"TheArtist"($51,220in2011),whichwerealsoreleasedonThanksgivingweekend.Thefilmexpandedintoadditionalmarketson12DecemberandwasreleasednationwideonChristmasDay.OnRottenTomatoes,thefilmhasanapprovalratingof91%basedon258reviews,withanaverageratingof7.7/10.Thesite'scriticalconsensusreads,"WithanoutstandingstarringperformancefromBenedictCumberbatchilluminatingitsfact-basedstory,"TheImitationGame"servesasaneminentlywell-madeentryintheprestigebiopicgenre.[Document2]hiscolleaguesworkedduringthewar,andCentralSaintMartinscampusonSouthamptonRowinLondon.OtherlocationsincludedtownsinEnglandsuchasNettlebed(JoyceGroveinOxfordshire)andChesham(Buckinghamshire).SceneswerealsofilmedatBicesterAirfieldandoutsidetheLawSocietybuildinginChanceryLane,andatWestLondonFilmStudios.Principalphotographyfinishedon11November2013.ThebombeseeninthefilmisbasedonareplicaofTuring'soriginalmachine,whichishousedinthemuseumatBletchleyPark.However,productiondesignerMariaDjurkovicadmittedthatherteammadethemachinemorecinematic.[Document3]TheImitationGameTheImitationGameisa2014AmericanhistoricaldramafilmdirectedbyMortenTyldumandwrittenbyGrahamMoore,basedonthebiography""byAndrewHodges.ItstarsBenedictCumberbatchasBritishcryptanalystAlanTuring,whodecryptedGermanintelligencecodesfortheBritishgovernmentduringtheSecondWorldWar.KeiraKnightley,MatthewGoode,RoryKinnear,CharlesDance,andMarkStrongalsostar.ThescreenplaytoppedtheannualBlackListforbestunproducedHollywoodscriptsin2011.TheWeinsteinCompanyacquiredthefilmfor$7millioninFebruary2014,thehighestamounteverpaidforU.S.distribution.[Document4]TheImitationGamewasannouncedthatAlexandreDesplatwouldprovidetheoriginalscoreofthefilm.ItwasrecordedbytheLondonSymphonyOrchestraatAbbeyRoadStudiosinLondon.DesplatusescontinuouspianoarpeggiostorepresentbothTuring‘sthinkingmindandtheworkingsofamechanicalmachine.Hesaidofthecomplexityofthecontinuityandstructureofthescore:[W]henthecameraattheendofthefilmhasthosebeautifulshotsoftheyoungboy,theyoungAlan,andhe’smeetingwiththeprofessorwho‘stellinghimhisfriendChristopherisdead,andthecameraispushinginonhim.
QueryWhatgenreisInside?CorrectAnswerhorrorfilmLLMAnswerwithRetrievedDocumentsHorrorfilmRerankedDocumentsbyInfoGain-RAGReranker[Document1]title:"Inside(2007film)"text:"guychasingafteryounggirls;it’soneoftheclichésofthegenre.Sothefirstmainideawaschangingtheidentityofthebadguy.Wewonderedwhatwasthemotivationforawomantohuntanotherwoman?"Thefilmwasgivenabudgetof1.7millionEuros."Inside"receivedmostlypositivereviews.RottenTomatoesreportedthat83%ofcriticsgaveitapositivereview.BloodyDisgustingrankedthefilmtwelfthintheirlistofthe'Top20HorrorFilmsoftheDecade',withthearticlesaying"Oneofthemostaudacious,brutal,unrelentinghorrorfilmsevermade,"Inside""[Document2]title:"TheInside(film)"text:"shehitshimovertheheadwithastone,escapesfromthewarehouseandishitbyacar."TheInside"wasshotonaprosumerHDcamcorderinsixdays.Theassaultscenewasperformedinasingle14-minutetake.EionMackenstatesinthemaking-ofthattherearealotofthingsinIrishhistorytoinspirehorrorfilms;andheeschewsbanshees,leprechauns,andsídheinfavorofsomethingmoreabstractly-formed(partlyinspiredbytheworstinman,aswellaspossiblyapaucityofimagination)."TheInside"premieredatEmpire,LeicesterSquareaspartofthe"[Document3]title:"InsideIn/InsideOut"text:"wasthefirstalbumhebought,attheageofnine.Hecreditedtherecordwithinspiringhimtobecomeamusician.InsideIn/InsideOutInsideIn/InsideOutisthedebutstudioalbumbyBritishindierockbandTheKooks.Itwasreleasedon23January2006onVirginRecords.Itcontainsthesingles,"Eddie'sGun","SofaSong","YouDon'tLoveMe","Naïve","SheMovesinHerOwnWay",and"OohLa".ThealbumwasproducedbyTonyHofferofrecordlabelVirginRecords.ReachingNo.2intheUKAlbumsChart,thealbumhassold"[Document4]title:"BloodInside"text:"tookformitdidnotfitanymore.Thentherewasthe"Heart"Album.Evenwethoughtthatwasabitpretentious.SoJørnandIarewalking/talkingoutsideonenighttryingtofigureoutwhatitisallabout.Thinkingonlyinkeywords:heart,blood,red,rose,beauty,violence,body,life,death,ambulance,hospitalandsoforth.Thenitstruckus:"BloodInside".”"Intermsofgenre,itismorerockthanelectronica.Evenabitpsychedelicand/orprogressiveattimes.Themoodiskindofsanctifiedandsad.Ithasafewfrivolousmomentsaswell,butas"
QueryWhatgenreisInside?CorrectAnswerhorrorfilmLLMAnswerwithRetrievedDocumentsDramaRerankedDocumentsbyBGEReranker[Document1]title:"TheWomanInside"text:"Inside.TheWomanInsideTheWomanInsideisa1981(butshotin1978)dramafilmmadeby20thCenturyFox,anddirectedbyJosephVanWinklewhoco-wrotescreenplaywithSteveFisher(uncredited).ThisdramafilmportraystheactionsofatoughVietnamvetwhowantstohaveasex-changeoperation.Heraunt(JoanBlondell)strugglestounderstandwhyshewouldwantdosuchathing.ThefilmwasreleasedafterBlondell'sdeath,endingacareerspanningmorethanhalfacentury.ThesonofEddyLawrenceMansonnowhasreleasedanewProjectnamedafterthefilmThe"[Document2]title:"Filmgenre"text:"Afilm'sgenrewillinfluencetheuseoffilmmakingstylesandtechniques,suchastheuseofflashbacksandlow-keylightinginfilmnoir,tightframinginhorrorfilms,fontsthatlooklikerough-hewnlogsforthetitlesofWesternfilms,orthe"scrawled"title-fontandcreditsof"Se7en"(1995),afilmaboutaserialkiller.Aswell,genreshaveassociatedfilm-scoringconventions,suchaslushstringorchestrasforromanticmelodramasorelectronicmusicforscience-fictionfilms.Thebasicgenresincludefictionanddocumentary,fromwhichsubgenreshaveemerged"[Document3]title:"PrettyontheInside"text:"Thefirst3,000pressingsoftheLPfeaturedbluevinyl,whilethefollowingpressingswereinstandardblack."PrettyontheInside"wasreceivedwithpositiveacclaimbymanyBritishandAmericanalternativepress.Inareviewby"NME",thealbumwaspositivelycomparedtoPattiSmith's"Horses",aswellasthedebutalbumsoftheRamones,Television,andNewYorkDolls,andwasbrandedasbeingin"aclassofitsown",whileElizabethWurtzelwrotein"TheNewYorker"that""PrettyontheInside"issuchacacophony...veryfewpeoplearelikelytogetthroughit"[Document4]title:"Interstitialart"text:"workinginasystemthatclearlylabelsoneshelfforromances,asecondshelfforfantasies,andathirdshelffortalesofhorror?There'snosingle,obviousanswer,becausesuchanovelisinterstitialfiction,itsessenceresidingsomewhereinbetweentheboundariesofthesegenres.OrconsidertheperformanceartistLaurieAnderson:Shemightgoonstageandsing,tellaspoken-wordstory,projectshadowpuppetsonascreen,andplayahackedviolinwhosebowisstrungwithaudiotape.Issheasinger,amonologist,apuppeteer,orsomekindoftinkeringinstrumentalist?Classifyingsuchanact"Figure 6: Comparison of documents retrieved by InfoGain-RAG reranker and BGE reranker.

B Mathematical Derivations
In this section, we provide detailed mathematical
derivations for two key components of margin loss:
(1) how LogSumExp (LSE) function approximates
the maximum function, and (2) the complete deriva-
tion steps of our margin loss formulation based on
LSE.
B.1 LSE Approximation of Maximum
Function
The LogSumExp function is defined as:
LSE(x 1, . . . , x n) = log(nX
i=1exp(x i))(10)
First, we prove that LSE provides an upper bound
for the maximum function. For anyi:
LSE(x 1, . . . , x n) = log nX
j=1exp(x j)!
≥log(exp(x i))
=xi(11)
Since this holds for alli, we have:
LSE(x 1, . . . , x n)≥max(x 1, . . . , x n)(12)
Letx∗= max(x 1, . . . , x n). We can rewrite LSE
as:
LSE(x 1, . . . , x n) = log nX
i=1exp(x i)!
= log 
exp(x∗)nX
i=1exp(x i−x∗)!
=x∗+ log
1 +X
i:xi̸=x∗exp(x i−x∗)
(13)
Since xi−x∗≤0for all i(with equality only
when xi=x∗), and typically xi−x∗≪0 for
xi̸=x∗, we have:
exp(x i−x∗)→0whenx i−x∗≪0(14)
Therefore:
log(1 +X
i:xi̸=x∗exp(x i−x∗))→0(15)
This yields our final approximation:
LSE(x 1, . . . , x n)≈x∗= max(x 1, . . . , x n)(16)
The approximation becomes more accurate as
the differences between the maximum value and
other values increase.B.2 Derivation of Margin Loss
Starting from the initial margin loss formulation:
LMargin ≈[LSE(γ(s n))−(−LSE(−γ(s p)))]+(17)
We can expand this expression:
[LSE(γ(s n))−(−NLSE(γ(s p)))]+
="
logLX
j=1exp
γ
sj
n
+ logKX
i=1exp
γ
−si
p#
+
="
log LX
j=1exp
γ
sj
nKX
i=1exp
γ
−si
p!#
+
="
logKX
i=1LX
j=1exp
γ
sj
n−si
p#
+
(18)
Finally, using the softplus function to smooth
the ReLU operation:
LMargin≈log"
1 +KX
i=1LX
j=1exp
γ
sj
n−si
p#
(19)
This completes the derivation of our margin loss
formulation.
C Comparisons with GTE Family
Table 5: Comparative analysis of InfoGain-RAG and
various GTE models as rerankers, with Qwen2.5 as the
answer generation model on TriviaQA. Results demon-
strate InfoGain-RAG’s superior performance across all
tested configurations.
Method GTE-1.5B GTE-7B GTE-Proprietary InfoGain-RAG
Qwen2.5-0.5B 45.3% 46.5% 49.5%55.8%
Qwen2.5-1.5B 59.7% 61.7% 63.3%66.3%
Qwen2.5-3B 63.3% 65.6% 65.8%68.2%
Qwen2.5-7B 67.4% 69.2% 69.5%72.1%
Qwen2.5-14B 67.5% 70.5% 71.1%72.9%
Qwen2.5-32B 70.1% 71.9% 72.0%74.7%
Qwen2.5-72B 69.2% 72.2% 73.4%76.3%
D Information of Datasets
TriviaQA1consists of 174,000 questions based on
Wikipedia pages, with answers and their justifica-
tions also determined from Wikipedia, including
138,000 for the training set, 17,900 for the valida-
tion set, and 17,200 for the test set. NaturalQA2
is a dataset consists of 307,373 training questions,
7,830 validation questions, and 7,842 test questions.
1https://huggingface.co/datasets/mandarjoshi/
trivia_qa
2https://huggingface.co/datasets/
sentence-transformers/natural-questions

where all questions originate from Google’s search
records, with answers derived from Wikipedia.
PopQA3contains approximately 14,000 questions
all sourced from the Wikidata database. PopQA
focuses on long-tail entities and can effectively as-
sess how well a LLM can grasp infrequent factual
knowledge.
FM24is a dataset that contains 10,400 training
questions, 1,170 validation qustions and 1,380 test
questions, which are designed to test the ability of
LLMs to answer simple, factual questions. These
questions cover a wide range of topics and are col-
lected from various online sources. The answers to
these questions are also provided, making it a valu-
able resource for training and evaluating question-
answering systems.
The December 2018 Wikipedia dump is a com-
prehensive collection of the content available on
Wikipedia up to December 2018. This dump in-
cludes nearly 23 millions articles, discussions, and
metadata, providing a vast amount of information
on a diverse range of topics. It is a valuable re-
source for natural language processing tasks, such
as information extraction, text summarization, and
question answering. Researchers and developers
can use this dump to train and test their models on a
large and diverse corpus of text, helping to improve
the performance and accuracy of their systems.
3https://huggingface.co/datasets/akariasai/
PopQA
4https://huggingface.co/datasets/tasksource/
fool-me-twice/viewer