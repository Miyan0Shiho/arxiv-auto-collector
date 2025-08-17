# SciRerankBench: Benchmarking Rerankers Towards Scientific Retrieval-Augmented Generated LLMs

**Authors**: Haotian Chen, Qingqing Long, Meng Xiao, Xiao Luo, Wei Ju, Chengrui Wang, Xuezhi Wang, Yuanchun Zhou, Hengshu Zhu

**Published**: 2025-08-12 08:36:23

**PDF URL**: [http://arxiv.org/pdf/2508.08742v1](http://arxiv.org/pdf/2508.08742v1)

## Abstract
Scientific literature question answering is a pivotal step towards new
scientific discoveries. Recently, \textit{two-stage} retrieval-augmented
generated large language models (RAG-LLMs) have shown impressive advancements
in this domain. Such a two-stage framework, especially the second stage
(reranker), is particularly essential in the scientific domain, where subtle
differences in terminology may have a greatly negative impact on the final
factual-oriented or knowledge-intensive answers. Despite this significant
progress, the potential and limitations of these works remain unexplored. In
this work, we present a Scientific Rerank-oriented RAG Benchmark
(SciRerankBench), for evaluating rerankers within RAG-LLMs systems, spanning
five scientific subjects. To rigorously assess the reranker performance in
terms of noise resilience, relevance disambiguation, and factual consistency,
we develop three types of question-context-answer (Q-C-A) pairs, i.e., Noisy
Contexts (NC), Semantically Similar but Logically Irrelevant Contexts (SSLI),
and Counterfactual Contexts (CC). Through systematic evaluation of 13 widely
used rerankers on five families of LLMs, we provide detailed insights into
their relative strengths and limitations. To the best of our knowledge,
SciRerankBench is the first benchmark specifically developed to evaluate
rerankers within RAG-LLMs, which provides valuable observations and guidance
for their future development.

## Full Text


<!-- PDF content starts -->

SciRerankBench: Benchmarking Rerankers Towards Scientific
Retrieval-Augmented Generated LLMs
Haotian Chen1, Qingqing Long1, Meng Xiao1, Chengrui Wang1, Xiao Luo2, Wei Ju2,
Xuezhi Wang1, Yuanchun Zhou1, Hengshu Zhu1∗
1Computer Network Information Center, Chinese Academy of Sciences
2Peking University
Abstract
Scientific literature question answering is a pivotal step towards
new scientific discoveries. Recently, two-stage retrieval-augmented
generated large language models (RAG-LLMs) have shown im-
pressive advancements in this domain. Such two-stage framework,
especially the second stage ( reranker ), is particularly essential to-
wards scientific domain, where subtle differences in terminology
may have greatly negative impacts on the final factual-oriented
or knowledge-intensive answers. Despite this significant progress,
the potential and limitations of these works remain unexplored. In
this work, we present a Scientific Rerank -oriented RAG Bench mark
(SciRerankBench ), for evaluating rerankers within RAG-LLMs
systems, spanning 5 scientific subjects.It is derived from over 250
million scholarly works with more than 100 million authors. To rig-
orously assess the reranker performance in terms of noise resilience,
relevance disambiguation, and factual consistency, we develop 3
types of question-context-answer (Q-C-A) pairs, i.e., Noisy Contexts
(NC) ,Semantically Similar but Logically Irrelevant Contexts (SSLI) ,
andCounterfactual Contexts (CC) . Through systematic evaluation of
13 widely used rerankers on 5 families of LLMs, we provide detailed
insights into their relative strengths and limitations. To the best of
our knowledge, SciRerankBench is the first benchmark specifically
developed to evaluate rerankers within RAG-LLMs, which provides
valuable observations and guidance for their future development.
CCS Concepts
•Computing methodologies →Search methodologies ;Natural
language generation .
Keywords
RAG, LLM, Reranker, Reranking, Scientific Literature, Retrieval
Augmented Generation
ACM Reference Format:
Haotian Chen1, Qingqing Long1, Meng Xiao1, Chengrui Wang1, Xiao Luo2,
Wei Ju2,, Xuezhi Wang1, Yuanchun Zhou1, Hengshu Zhu1. xx. SciRerankBench:
∗Corresponding Author.
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
Conference acronym ’XX, Woodstock, NY
©xx Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-x-xxxx-xxxx-x/YYYY/MM
https://doi.org/XXXXXXX.XXXXXXXBenchmarking Rerankers Towards Scientific Retrieval-Augmented Gener-
ated LLMs. In Proceedings of Make sure to enter the correct conference title
from your rights confirmation email (Conference acronym ’XX). ACM, New
York, NY, USA, 17 pages. https://doi.org/XXXXXXX.XXXXXXX
1 Introduction
Scientific literature question answering has always been a pivotal
steps for scientific new discoveries [ 12,14,16,27,28,62]. For ex-
ample, a biologist studying CRISPR gene-editing must consult hun-
dreds of papers to understand off-target effects [ 3,15]. In COVID-19
research [ 46,57], clinicians have to rapidly analyze thousands of
emerging papers each week to make informed public health de-
cisions. Recently, retrieval-augmented generated large language
models (RAG-LLMs) [ 1,26,59] have gained substantial progress
in scientific literature analysis. By effectively combining retrieval
and generative capabilities, RAG-LLMs demonstrate significant ad-
vantages when addressing knowledge-intensive, factually-oriented,
and reliability-critical tasks [ 9,18,51]. This integration allows mod-
els to improve factual accuracy while generating responses that
are more interpretable and reliable, making them well-suited for
scientific tasks requiring trustworthy and rigorous reasoning.
Current RAG-LLMs generally adopts a two-stage pipeline. The
first stage conducts fast but low-precision ( embedding model ) re-
trievals, while the second stage ( rerankers ) emphasizes high-precision
but higher computational costs for a trade-off between effectiveness
and efficiency. Such two-stage framework, especially the second
stage, is particularly essential in the scientific domain, where subtle
differences in terminology may have greatly negative impacts on
the final factual-oriented or knowledge-intensive answers [ 19,24,
30,36]. As illustrated in Figure 1, sorts of widely-used rerankers,
such as BCE [ 31], BGE [ 6], MXBAI [ 38] and ColBert [ 37], have
been developed into mature products and have received significant
attention from both academia and industry.
Despite significant attention, the capabilities of rerankers in
scientific domains remain largely unexplored. Recent research pri-
marily investigates the overall RAG-LLM systems, focusing pre-
dominantly on the quality of their final outputs, and neglects eval-
uating the effectiveness of individual components. For example,
Rankify [ 1] and FlashRAG [ 18] provide a comprehensive and modu-
lar Python toolkit for retrieval, reranking, and generation, support-
ing reproducible experimentation. Johannesson and Olsson [ 22]
propose a benchmark to evaluate re-ranking methods in LLM-based
search systems. However, these studies notably overlook detailed
evaluation of the reranking module. Furthermore, their datasets
are sourced from general domains, thus failing to effectively assess
the deeper reasoning capabilities required by RAG-LLMs in the
scientific domain. Other researchers concentrate on constructingarXiv:2508.08742v1  [cs.CL]  12 Aug 2025

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Haotian Chen et al.
What is the key component of gene
regulation and biological fate?Retriver
LLMAnswerReRanker
 ...N relevent corpus (contexts)
2022 2025 2024 2023
MiniLM
In-Rank
SPLADE
 MXBAI
 Jina
ColBert
 BGE
BCE
ListT5
 RankT5
Twolar
 RankGPT
LLM2V ec
...K relevent corpus (contexts)
Generator
ReaRank
Figure 1: Overall architecture of the two-stage RAG-LLM pipeline. The first stage conducts fast but low-precision ( embedding model ) retrievals,
while the second stage ( rerankers ) emphasizes high-precision but higher computational costs for a trade-off between effectiveness and efficiency
in RAG-LLMs.This two-stage setup first narrows the candidate set efficiently, then rerank results for higher quality generation.
scientific question answering (QA) benchmarks without incorpo-
rating retrieval-augmented generation, meaning their evaluations
do not involve external knowledge as input contexts for LLMs.
Typical works include SciEval [ 40], SciBench [ 48], AGIEval [ 61],
SciHorizon [ 35], MATH [ 13], and AquaRat [ 25] all of which aim to
systematically evaluate AI-for-Science models and datasets. How-
ever, these benchmarks neglect external retrieved contexts, making
them unsuitable for evaluating the deep reasoning capabilities of
RAG-LLMs when addressing knowledge-intensive, fact-oriented,
and high-reliability tasks. Table 1 provides a detailed comparison
between this paper and related studies.
In response, we propose a Scientific Rerank -oriented RAG Bench -
mark ( SciRerankBench ). It is derived from over 250 million schol-
arly works with more than 100 million authors [ 33]. Our synthetic
dataset contains 4.5K question-context-answer (Q-C-A) pairs, which
spanning 5 typical scientific domains: biology, physics, chemistry,
geography, and mathematics. Specifically, we construct 3 types of
Q-C-A pairs, for evaluating the complicated real world reasoning
capabilities of rerankers towards scientific domains. Noisy Contexts
(NC) are designed to evaluating the discernability of rerankers to-
wards noisy contexts. Semantically Similar but Logically Irrelevant
Contexts (SSLI) are constructed to evaluating the reasoning ability
of rerankers towards similar but do not supporting correct answer
contexts. Counterfactual Contexts (CC) are built to evaluate the
ability of rerankers when facing factually incorrect or contradic-
tory knowledge. Finally, we running 13 popular rerankers on 11
LLMs. Experimental results reveal that rerankers can significantly
improves performance in RAG-LLMs, with cross-encoder architec-
tures offering the strongest gains. However, in more complicated
multi step reasoning tasks, the quality of final answers heavily de-
pends on LLMs’ internal reasoning capacity. We summarize our
contributions as follows :
•To the best of our knowledge, this work is the first benchmark
designed for evaluating rerankers in RAG-LLMs.
•We construct 4 types of diagnostic datasets to deeply assess
the capabilities of rerankers towards scientific domains.
•Through evaluating popular 13 rerankers on 5 families of
LLMs, we summarize several key observations, which reveal
their advantages and limitations.2 Related Work
To position our work in the context of existing research, we review
prior efforts across three key areas: scientific QA datasets, bench-
marks for scientific QA tasks, and reranking methods that have
been explored in RAG-based LLM systems
2.1 Scientific Question Answering Datasets
Existing QA datasets can be broadly divided into two categories:
open-domain and scientific datasets. In open domain, Natural Ques-
tions (NQ) [ 24], TriviaQA [ 23], and WebQuestions [ 49] have served
as popular open-domain QA datasets. Furthermore, HotpotQA [ 54]
and MuSiQue [ 43] emphasize multi-hop reasoning and complex
context synthesis. Scientific literature plays a critical role in advanc-
ing human knowledge, and accurate comprehension and retrieval
of scientific documents is essential for research progress [ 20]. In the
biomedical domain, BioASQ [ 44] are generated based on PubMed
abstracts, with expert annotated factoid and list type questions. Pub-
MedQA [ 19] further extends this by focusing on yes/no/maybe ques-
tions derived from clinical research papers. MATH [ 13] is designed
to assess the procedural reasoning abilities of LLMs across various
mathematical domains. AquaRat [ 25] offers algebra word problems
with step by step rationales. OpenBookQA [ 29] and ARC [ 7] are
general scientific QA datasets. They construct science exam style
questions at elementary and high school levels, emphasizing both
retrieval and reasoning.
Sorts of studies [ 1,18] shown that question related context, i.e.,
external knowledge or facts, can help LLM generate more reliable
and accurate answers. However, except for PubMedQA and BioASQ,
most datasets do not provide a large amount of contextual informa-
tion.They assume that relevant context is already retrieved and do
not simulate real world conditions where the model must reason
over partially relevant, noisy, or even misleading documents. This
limits their ability to reveal the nuanced performance of retrieval
and rerankers.
2.2 Scientific Question Answering Benchmarks
Scientific QA has become a popular use case for evaluating LLMs
due to its demand for factual precision and deep domain under-
standing. Several recent benchmarks have been proposed to test
LLMs on scientific reasoning tasks. Similarly, APBench [ 50] focuses
exclusively on astrodynamics, evaluating models on real world

SciRerankBench: Benchmarking Rerankers Towards Scientific Retrieval-Augmented Generated LLMs Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Table 1: Comparison of SciRerankBench among related datasets, toolkits and benchmarks.
Type Work Source Data ExplainabilityScientific
FieldKnowledge
LevelRerank
OrientedDataset
Size
DatasetNQ [24] Google search log ✗ ✗ General ✗307k (train),
7.8k (dev/test)
HotpotQA [54] Wikipedia ✓ ✗ General ✗ 113k
PubMedQA [19] PubMed abstracts ✓ Bio. Expert ✗1k expert,
61k unlabeled,
211k generated
ToolkitFlashRAG [18] Public benchmarks ✓ ✗ General ✓ unknown
rankify [1] Public benchmarks ✓ ✗ General ✓ unknown
Bench.CRAG [53] Web search log ✓ ✗ General ✗ 4.4K
AGIEval [61] Exams ✗Geo., Bio., His.,Chem., Phy.,
Eng., Chi., Math, Law, Log.High school ✗ 8k
SciEval [40] Public datasets ✗ Phy., Chem., Bio. College ✗ 15k
SciBench [48] University exams ✓ Phy., Chem., Math University ✗ 2.1k
PHYBench [36] Physic exams ✓ Phy.High School,
College,
Olympiad✗ 500
SciHorizon [35] Public datasets ✓Math., Phys., Chem.
Bio., Earth & SpaceCollege ✗ unknown
SciRepEval [39] Scientific literature ✗ 20 subjects Expert ✗ 12.4M
Ours Scientific literature ✓ Phy., Chem., Bio., Geo., Math. Expert ✓ 4.5k
aerospace tasks. ChemBench [ 30] benchmarks LLMs on diverse
chemistry tasks, evaluating both factual knowledge and reasoning.
PHYBench [ 36] focuses on physical reasoning through tasks like
dimensional analysis and estimation. However, these benchmarks
are typically limited to a single scientific domain, which restricts
their ability to evaluate the generalizability and robustness of LLMs
across diverse fields of scientific inquiry.
To overcome the limitations of domain specific benchmarks, sev-
eral multidisciplinary QA benchmarks have been proposed to evalu-
ate LLMs across a broader spectrum of scientific fields. SciEval [ 40]
focuses on a comprehensive multi axis evaluation framework, while
SciBench [48] pushes the level of questions to a university setting
and incorporates multimodal prompts. AGIEval [ 61] expands the
evaluation of LLMs beyond factual recall, incorporating high diffi-
culty tasks across multiple disciplines , many of which are based
on real world human exams. SciRepEval [ 39] introduces a unified
benchmark comprising 25 diverse to evaluate the generalization
ability of scientific document representation models across multiple
task formats. SciHorizon [ 35] provides a comprehensive benchmark
to evaluate the AI-for-Science readiness of both scientific datasets
and LLMs. Rankify [ 1] and FlashRAG [ 18] both provide a compre-
hensive and modular Python toolkit for retrieval, reranking, and
generation, supporting reproducible experimentation and system-
atic benchmarking. While SciEval, SciBench, AGIEval, and SciRepE-
val offer broad and diverse evaluation settings, they do not isolate or
assess the rerankers critical to RAG pipelines. Conversely, Rankify
and FlashRAG provide infrastructure for evaluating rerankers, but
their datasets remain largely grounded in general domains rather
than the scientific literature.2.3 Rerankers in Retrieval-Augmented
Generated LLMs
Rerankers in RAG-LLMs can be categorized into the following
architectures. Dense cross-encoder rerankers: BGE [ 6], Jina [ 21],
BCE [ 31], and MXBAI [ 38]. They compute contextualized relevance
scores via bi-sequence encoders and have demonstrated strong
performance on various ranking tasks. Sparse lexical rerankers:
such as SPLADE [ 11] utilize term level matching with sparsity
constraints to maintain interpretability and alignment with classi-
cal information retrieval principles. Lightweight alternatives like
MiniLM [ 47] leverage distilled self-attention mechanisms to re-
duce inference cost while maintaining reasonable accuracy. Late-
interaction rerankers like ColBert [ 37] delay token level interactions
until the scoring stage, offering an efficiency accuracy trade-off suit-
able for large scale retrieval. More recently, LLM-based rerankers,
such as RankGPT [ 41], LLM2Vec [ 4] exploit the reasoning capabili-
ties of large generative models to perform zero-shot or instruction
guided reranking, along with increased computational overhead.
Sequence-to-sequence rerankers, such as In-Rank [ 60], tokenize
query-document pairs and employ generative decoding to assess
relevance, making them suitable for instruction driven reranking
while not relying on LLMs. Additionally, listwise ranking models :
RankT5 [ 63], ListT5 [ 55], adopt groupwise learning objectives to
optimize the global ordering of candidate contexts. Twolar [ 2] pro-
poses a two-step LLM augmented distillation framework for passage
reranking, achieving strong performance by transferring knowl-
edge from powerful LLMs to lightweight student models. Rank-
R1 [64] enhances the reasoning capability of document rerankers
by introducing a reinforcement learning framework optimized for

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Haotian Chen et al.
240M  Papers
TFs control activity
of gene
transcription ...What are the categories
of TF isoforms ?
What are the isoforms categories
of  genes  that control activity of
gene transcription ?100M Authors
Scientific  Literatur e Multi-Step Reasoning  Q-A  Pairs Rerank-Oriented Q-C-A  Pairs120K 
Publication
Venues... TF isoforms :
rewirers and negative
regulators...
Bridge: TFContext (Abstracts)
Context (Abstracts)One-Step Question
Multi-Step  Question... activities of gene transcription  can
be influenced by isoforms derived
from various tissues  ...
... genes that regulate transcription
produce only a single isoform , as
alternative splicing does not occur  ...Counterf actual
ContextsSemantically
Similar but
Logical-IrrelevantNoisy
Contexts
Intergate... genes responsible for coat color in
rabbits  has multiple alleles  that
determine a range of phenotypes  ...
Figure 2: Overview of the synthetic dataset process. We first generate multi-hop reasoning and diverse types of distractors to simulate real world
complicated retrieved contexts. Distractors include Noisy Contexts ,Semantically Similar but Logically Irrelevant Contexts andCounterfactual
Contexts .
answer aware rewards. It directly optimizes reranking performance
based on downstream QA accuracy rather than traditional relevance
labels. G-RAG [ 8], agraph based reranker that models semantic and
relational connections between retrieved documents using graph
neural networks (GNNs). Agent-based rerankers explicitly formu-
late the reranking process as a decision making task. Rearank [ 58]
introduces a reasoning agent trained via reinforcement learning
to iteratively examine candidate passages and generate a globally
optimal ranking.
However, existing benchmarks do not evaluate these rerankers
independently in a reliable way. They are usually designed to evalu-
ate end-to-end RAG systems and assume that the retrieved contexts
are already of high quality. This makes it difficult to assess how well
a reranker can distinguish relevant from irrelevant or misleading
passages, especially in scientific tasks where accuracy, terminology,
and reasoning precision are crucial.
3 Reranking-Oriented Benchmarking Dataset
This section outlines our data construction process, including source
data collection, multi-step QA generation, and the synthesis of
reranking-oriented question-context-answer (Q-C-A) pairs.
3.1 Scientific Literature Source Data
We have collected a large number of scientific papers from open
access academic repositories [33]. The selected papers span multi-
ple disciplines, including computer science, biology, physics, and
materials science, ensuring coverage of a diverse range of scien-
tific terminologies. Each paper contains structured metadata (title,
abstract, authors, publication year) and unstructured content (ab-
stracts and full texts). These information is used in 2 ways: generate
QA pairs and serve as the RAG retrieval corpus. Our dataset covers 5
major disciplines: physics, chemistry, biology, geography, and math-
ematics, ensuring broad topical diversity and scientific relevance.
All records were imported into a vector database [ 34], enabling
efficient dense retrieval. The abstract serves as the primary source
for both question generation and retrieval context.3.2 Synthetic Multi-Step Reasoning Q-A Pairs
To construct QA pairs from scientific literature, we apply a dual-
method approach that includes both single-hop and multi-hop ques-
tion generation techniques. This ensures a diverse set of questions
ranging from basic fact extraction to multi-step reasoning. For
single-hop question generation, we adopt the LMQG [45]framework,
a language model based method to generate natural language ques-
tions from input passages. Specifically, we use the unsupervised
mode of LMQG, where only the abstract text is required as in-
put. This approach allows for large scale, automatic generation of
factoid style questions from scientific abstracts with minimal hu-
man intervention. For multi-hop question generation, we adopt the
Unsupervised-Multi-hop-QA [32] framework to construct questions
that require reasoning across 2 semantically linked contexts.The
overall process is illustrated in Figure 2. Specifically, for each source
abstract in our scientific papers, we retrieve the most semantically
similar abstract based on dense embeddings. The resulting pair
comprising the original abstract and its nearest neighbor—is con-
catenated and fed into the multi-hop question generation model.
This setup are designed to generate questions that require inte-
grating information across 2 distinct but related scientific papers,
simulating real world complicated scientific reasoning scenarios.
3.3 Synthetic Reranking-Oriented Q-C-A Pairs
Fang et al. [ 10] analyzed how different types of noisy retrieved texts
affect the performance of LLMs. They identified 3 common noise
types: contexts that are superficially related to the query but lack
the correct answer,contexts that are irrelevant to the query , and
contexts that are topically related to the query but contain incorrect
information. Their findings show that these types of context can
easily confuse large language models. To evaluate the ability of
rerankers towards scientific domains, we construct the reranking-
oriented Q-C-A pairs in this paper. As illustrated in Figure 2, we
construct 3 types of disruptive contexts:
Noisy Contexts (NC) : A common challenge in real-world scien-
tific retrieval scenarios is the inevitable presence of irrelevant or
unrelated passages mixed in with relevant contents [ 24,25,30,40].

SciRerankBench: Benchmarking Rerankers Towards Scientific Retrieval-Augmented Generated LLMs Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
30405060Avg Recall
Mistral-7B
DeepSeek-7B Qwen-7BLLaMA-7B InternLM2-7BInternLM2-20BMistral-24B
Qwen-32BLLaMA2-70B
Qwen-72BMistral-7B
DeepSeek-7BQwen-7B
LLaMA-7BInternLM2-7B InternLM2-20B
Mistral-24B Qwen-32BLLaMA2-70BQwen-72BMistral-7B
DeepSeek-7BQwen-7B
LLaMA-7BInternLM2-7B InternLM2-20B
Mistral-24BQwen-32B
LLaMA2-70BQwen-72B
Contexts + Reranker
Contexts
w/o Contexts
Figure 3: Evaluation results under 3 different settings: without any context, with retrieved contexts, and with contexts after applying the
reranking stage by recall scores.
To simulate this practical challenge and evaluate the ability of
rerankers to distinguish relevant information within noisy data,
we design the Noisy Contexts setting. Each query in this setting
is paired with 5 passages confirmed to be relevant, mixed with 95
randomly sampled unrelated passages. This setup serves as a fun-
damental test for assessing the discernability of rerankers towards
noisy contexts.
Semantically Similar but Logically Irrelevant Contexts (SSLI) :
Passages that share high semantic similarity with the query but are
logically irrelevant often appear as challenge. Such contexts can
easily mislead rerankers [ 19,36], as they match surface level key-
words and phrases, yet do not contain the necessary information to
answer the question. This mismatch between semantic similarity
and logical relevance requires rerankers to possess strong reasoning
and contextual understanding abilities. To simulate this challenge
and rigorously evaluate rerankers’ reasoning skills, we design the
Semantically Similar but Logically Irrelevant setting. Each query is
paired with 90 standard candidate passages and 10 passages that are
semantically similar but logically irrelevant. This setup aims to test
the model’s capacity for fine grained logical discrimination, going
beyond simple semantic matching to accurately identify answer
bearing content.
Counterfactual Contexts (CC) :Retrieval passages may contain
information that is factually incorrect or contradictory to estab-
lished knowledge [ 1,35,39,50]. Such passages pose a significant
challenge because they can mislead rerankers relying solely on se-
mantic similarity, requiring models to assess not just relevance but
also factual correctness. To simulate this challenge and rigorously
evaluate rerankers’ counterfactual reasoning abilities, we design
the Counterfactual Contexts setting. Each question includes 90 stan-
dard candidate passages and 10 passages containing plausible yet
factually incorrect or contradictory information. This setup tests
the model’s ability to discern truthfulness and accuracy in addition
to semantic alignment.This setting presents a more realistic and
stringent scenario for scientific QA systems, where distinguishing
between fact and fiction is essential for trustworthy ouputs.
4 Experiment
We evaluate 13 widely used rerankers on the SciRerankBench bench-
mark, covering their effectiveness, robustness, reasoning capability,
and efficiency. The following subsections describe our experimental
setup and detailed results across various evaluation dimensions.4.1 Experimental Settings
Rerankers. We categorize all rerank baselines into 8 groups based
on their architectures. (1) Dense rerankers: BGE [ 6], BCE [ 31],
Jina [ 21]and MXBAI [ 38]. (2) Sparse lexical rerankers: SPLADE [ 11].
(3)Listwise rerankers: RankT5 [ 63] and ListT5 [ 55]. (4) Seq2Seq
rerankers: In-Rank [ 60]. (5) Lightweight student rerankers: Twolar[ 2]
and MiniLM[ 47]. (6) Late-interaction rerankers: ColBert[ 37] (7)
LLM-based rerankers: RankGPT [ 41],LLM2Vec[ 4] (8) Agent-based
rerankers: Rearank [58]1.
LLMs. In our experiments, we evaluate a diverse set of 11 open-
source LLMs drawn from different families, as summarized in Ta-
ble 3 in the Appendix. Mistral models (7B and 24B) [ 17] incorporate
architectural enhancements such as Sliding Window Attention (SWA)
andGrouped-Query Attention (GQA) to improve long-context model-
ing and computational efficiency. The LLaMA2-70B model [ 42] sim-
ilarly employs GQA along with Rotary Positional Embeddings (RoPE)
to better handle extended input sequences. The Qwen model [ 52]
uses the standard multi-head attention (MHA) mechanism, com-
bined with GQA and RoPE, to support multi-language and long-
context input while maintaining performance. DeepSeek-V3 [ 26]
introduces a sparse attention mechanism, mixed experts (MoE)
and multi-layer attention (MLA) to improve large-scale reason-
ing efficiency and knowledge capacity.We include InternLM2.5 [ 5],
a multilingual, open-weight transformer model supporting long-
context understanding (up to 200K tokens) and optimized for both
instruction-following and reasoning tasks.
Evaluation Protocols and Metrics. To comprehensively evaluate
model performance, we design a two-level evaluation strategy tar-
geting both rerankers and LLMs. For rerankers, we compare the top
contexts against annotated relevance labels to assess their ability
to retrieve the most informative and factually correct passages for
a given scientific question. Recall@10 evaluate how many of the
known relevant passages are ranked within the 10, respectively,
thus testing retrieval completeness. For LLMs, we assess their ability
to generate accurate and complete answers by comparing outputs
to golden answers. We use a token level metrics: Recall , which cap-
tures how completely the model reproduces the reference content.
1As the codes of some rerankers, i.e., Rank-R1 [ 64] and G-RAG [ 8], are unavailable,
we do not evaluate these models. Besides, the estimated time of evaluating RankGPT
would be approximately 1,611 hours. Thus we do not run RankGPT [ 41] in this paper.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Haotian Chen et al.
/uni00000025/uni00000026/uni00000028 /uni00000025/uni0000002a/uni00000028 /uni0000002d/uni0000004c/uni00000051/uni00000044 /uni0000002f/uni0000004c/uni00000056/uni00000057/uni00000037/uni00000018 /uni00000030/uni0000004c/uni00000051/uni0000004c/uni0000002f/uni00000030 /uni00000035/uni00000044/uni00000051/uni0000004e/uni00000037/uni00000018 /uni00000036/uni00000033/uni0000002f/uni00000024/uni00000027/uni00000028 /uni00000037/uni0000005a/uni00000052/uni0000004f/uni00000044/uni00000055 /uni00000026/uni00000052/uni0000004f/uni00000025/uni00000048/uni00000055/uni00000057 /uni00000030/uni0000003b/uni00000025/uni00000024/uni0000002c /uni0000002c/uni00000051/uni00000010/uni00000035/uni00000044/uni00000051/uni0000004e /uni0000002f/uni0000002f/uni00000030/uni00000015/uni00000039/uni00000048/uni00000046 /uni00000035/uni00000048/uni00000044/uni00000055/uni00000044/uni00000051/uni0000004e10203040506070/uni00000036/uni00000046/uni00000052/uni00000055/uni00000048
/uni00000026/uni00000026 /uni00000030/uni00000058/uni0000004f/uni00000057/uni0000004c/uni00000010/uni0000002b/uni00000052/uni00000053 /uni00000025/uni00000044/uni00000056/uni00000048 /uni00000031/uni00000026 /uni00000036/uni00000036/uni0000002f/uni0000002c
Figure 4: Evaluation of rerankers on five tasks using Recall metric
/uni00000025/uni00000026/uni00000028 /uni00000025/uni0000002a/uni00000028 /uni0000002d/uni0000004c/uni00000051/uni00000044 /uni0000002f/uni0000004c/uni00000056/uni00000057/uni00000037/uni00000018 /uni00000030/uni0000004c/uni00000051/uni0000004c/uni0000002f/uni00000030 /uni00000035/uni00000044/uni00000051/uni0000004e/uni00000037/uni00000018 /uni00000036/uni00000033/uni0000002f/uni00000024/uni00000027/uni00000028 /uni00000037/uni0000005a/uni00000052/uni0000004f/uni00000044/uni00000055 /uni00000026/uni00000052/uni0000004f/uni00000025/uni00000048/uni00000055/uni00000057 /uni00000030/uni0000003b/uni00000025/uni00000024/uni0000002c /uni0000002c/uni00000051/uni00000010/uni00000035/uni00000044/uni00000051/uni0000004e /uni0000002f/uni0000002f/uni00000030/uni00000015/uni00000039/uni00000048/uni00000046 /uni00000035/uni00000048/uni00000044/uni00000055/uni00000044/uni00000051/uni0000004e102030405060708090100/uni00000024/uni00000046/uni00000046/uni00000058/uni00000055/uni00000044/uni00000046/uni0000005c/uni00000003/uni0000000b/uni00000008/uni0000000c
/uni00000026/uni00000026 /uni00000030/uni00000058/uni0000004f/uni00000057/uni0000004c/uni00000010/uni0000002b/uni00000052/uni00000053 /uni00000025/uni00000044/uni00000056/uni00000048 /uni00000031/uni00000026 /uni00000036/uni00000036/uni0000002f/uni0000002c
Figure 5: Evaluation of rerankers on five tasks using Recall@10 metric.
This evaluation framework allows us to analyze the individual con-
tribution and limitations of rerankers versus generation models in
the RAG pipeline, especially under challenging scientific question
answering scenarios. Definitions of these metrics are provided in
Appendix A.2. This two-level evaluation helps us figure out whether
mistakes come from retrieving the wrong information, or from the
model not using the information well when generating answers. By
looking at both parts separately, we can better understand where
the system needs improvement. This is especially important for
handling complex scientific questions where both accurate retrieval
and strong reasoning matter.This approach also helps compare dif-
ferent system components fairly, making it easier to identify which
part contributes most to the overall performance.
Implementation Details. The rerankers are conducted on 4 NVIDIA
A100 GPUs. All LLMs and rerankers are used with their default
configurations. We conduct 3 independent experiments to ensure
the reliability and stability of our evaluation results. To ensure a
fair comparison, all models are evaluated under a zero-shot set-
ting, without task-specific fine-tuning or additional adaptations.
We collect 100,000 scientific articles and store their abstracts in a
Qdrant vector database [ 34] to facilitate dense retrieval for RAG-
based systems. Then we preprocess and filter the abstract texts
to ensure quality and consistency. Specifically, we retained only
those abstracts with a length between 100 and 500 characters. This
range was selected to exclude low information or overly terse ab-
stracts, e.g., placeholder texts, while also avoiding excessively long
passages that could complicate automatic question generation or
overload retrieval models. Following previous works [ 1,18,56],
100 candidate contexts are first retrieved with BGE as the retriever,
then the top 10 are selected by rerankers, which are used as input
to LLMs for final answer generation. In addition, a sampling testdataset showed that RankGPT required 3.5 hours to process 100
samples. Based on this, the estimated time to complete the evalua-
tion on the full dataset would be approximately 1,611 hours. Due
to this high computational cost, we do not running RankGPT in
this paper.
4.2 Evaluating the Usefulness of Rerankers in
RAG Pipelines.
To evaluate the impact of context retrieval and reranking on answer
generation quality, we compare LLM performance across 3 distinct
settings: (1) zero-context generation (no retrieved context), (2) naive
retrieval (Top-10 dense retrieval without reranking), and (3) RAG
with reranked contexts (Top-10 after reranking). As shown in Fig-
ure 3, all architectures benefit significantly from adding external
contexts, with contexts alone contributing 20–30 point gains in re-
call on average. Among all families, InternLM and Qwen exhibit the
strongest improvements from reranking, suggesting that these mod-
els better leverage high quality contexts when ranked effectively.
InternLM2-20B consistently achieves the highest recall across all
settings, while Qwen models also demonstrate robust gains, likely
aided by their multilingual pretraining and long-context optimiza-
tion. In contrast, DeepSeek models show moderate performance
without context but respond well to retrieval, albeit with limited
further gains from reranking.
Takeaways.
•While all LLMs benefit from retrievers and rerankers, the ability to
exploit high quality reranked context varies across different families
and architectures.
•After reranking the context with rerankers, InternLM and Qwen achieve
the best performance.

SciRerankBench: Benchmarking Rerankers Towards Scientific Retrieval-Augmented Generated LLMs Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
4.3 Evaluating the Discernability of Rerankers
towards Noisy Contexts
In the NCdataset, where 5 relevant contexts are mixed with 95
random distractors. We observe that most rerankers are able to
retrieve nearly all relevant contexts within the top-10 results, as in-
dicated by the high recall@10 scores across most evaluated models.
The presence of a large number of unrelated contexts appears to
disperse model attention, impairing the models’ ability to maintain
coherent focus on relevant information. Even models with strong
multi context reasoning capabilities often struggle to prioritize truly
important passages over distractors. As a result, the effective recall
of useful information during answer generation degrades, leading
to lower final answer quality. Furthermore, when irrelevant con-
texts are intermixed, models may mistakenly rely on misleading
information during both retrieval and generation stages. This phe-
nomenon highlights the necessity of not only retrieving relevant
documents but also ensuring high contextual purity. Doing so helps
avoid attention diffusion and knowledge contamination in scientific
question answering pipelines.
Takeaways.
•LLMs often struggle to effectively filter out irrelevant contexts when
handling tasks with a large amount of unrelated information, which
affects the quality of final answers.
•Effective scientific QA requires both precise retrieval of relevant infor-
mation and minimizing the influence of irrelevant content.
4.4 Evaluating the Reasoning Ability of
Rerankers towards Semantically Similar but
Logical-Irrelevant Contexts
The SSLI dataset focus on evaluating rerankers’ ability to handle
semantically challenging cases, where contexts are semantically
similar yet unanswerable. In these complex scenarios, cross-encoder
architectures demonstrate significant advantages. By jointly encod-
ing the query and document, cross-encoders allow fine-grained
feature interactions, capturing subtle semantic differences crucial
for distinguishing between true and misleading information. In
particular, MXBAI achieves the highest recall scores across nearly
all evaluation settings, highlighting the strength of cross-encoder
designs in complex reasoning tasks.In contrast, sparse retrieval
models, which rely on masked language model objectives to pro-
duce sparse representations, exhibit significantly lower recall (ap-
proximately 44%). Although sparse models offer faster inference
speed—achieving roughly 71.4% the inference time of dense archi-
tectures—their limited semantic granularity hinders their ability
to identify fine-grained factual inconsistencies, leading to notable
performance degradation on nuanced reasoning tasks. Late interac-
tion models such as ColBert, which independently encode queries
and documents before performing token wise interaction, main-
tain strong recall in random distractor settings but struggle when
deeper semantic comprehension is required. ColBert’s recall drops
to 43.47% on counterfactual tasks and 45.17% on semantic confu-
sion tasks, illustrating the limitations of independent representation
without full context fusion. LLM-based embedding methods like
LLM2Vec underperform on semantically challenging tasks, scoring
only 33.72% on counterfactual and 33.04% on semantic confusion.Although it leverages frozen LLMs for dense representation, the
lack of joint encoding limits its ability to capture subtle semantic
mismatches crucial for accurate reranking.
Takeaways.
•Cross-encoders outperform other rerankers on semantically challenging
tasks, due to their fine-grained query-document interaction.
4.5 Evaluating the Ranking Ability of
Rerankers
To evaluate the effectiveness of rerankers independently of final
answer generation, we additionally present recall@10 to directly
measure how well rerankers prioritize relevant information in the
top-ranked results.For Recall@10, most rerankers achieve relatively
high scores across datasets, including multi-hop and complex QA
tasks. However, a discrepancy emerges when examining the final
answer quality: despite high Recall@10, the generated answers
from LLMs are often of lower quality compared to those from the
base retrieval-only setup. This indicates that, while rerankers are
effective in ensuring that relevant contexts are included within
the top-10 candidates, the ultimate answer quality remains con-
strained by the LLM’s own generation capabilities. The finding
highlights a limitation where retrieval improvements alone cannot
fully compensate for deficiencies in the LLM’s ability to synthesize
and reason over retrieved knowledge. In multi-hop tasks, rerankers
can successfully retrieve sufficient relevant context. However, even
with ample context, LLMs fail to achieve expected performance
due to inherent limitations in their reasoning capabilities, which
ultimately restrict the quality of final answers. Beyond limitations
in LLM generation, differences in reranker design also contribute
to performance gaps across tasks. Although REARANK uses a list-
wise reinforcement learning framework with local window -based
ranking over sliding windows, its architecture lacks global cross-
document encoding or token-level interaction. As a result, it cannot
effectively aggregate complementary evidence scattered across pas-
sages, leading to lower Recall@10 (62.41) on multi-hop reasoning
tasks.
Takeaways.
•For multi-hop tasks, retrieval and rerankers are able to select high-
relevant contexts, but the final performance still depends on the inherent
reasoning capacity of LLMs.
•Reranker architectures also influence performance on muti-hop task.
4.6 Evaluating the Running Efficiency of
Rerankers
This section analyze the running efficiency. The average inference
time of one question is shown in Figure 6. We observe clear run-
ning efficiency differences across reranker architectures. Sparse
models like SPLADE achieve the fastest inference (0.49s), owing to
their non-interactive, inverted-index design. Lightweight sentence
encoders such as MiniLM also offer rapid performance (1.34s), mak-
ing them attractive for latency-sensitive applications. In contrast,
cross-encoder models demonstrate a trade-off: BCE and BGE are
relatively efficient (3.5s), while deeper or more expressive models

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Haotian Chen et al.
0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0
/uni00000024/uni00000059/uni00000048/uni00000055/uni00000044/uni0000004a/uni00000048/uni00000003/uni0000002c/uni00000051/uni00000049/uni00000048/uni00000055/uni00000048/uni00000051/uni00000046/uni00000048/uni00000003/uni00000037/uni0000004c/uni00000050/uni00000048/uni00000003/uni0000000b/uni00000056/uni0000000c/uni00000030/uni00000052/uni00000047/uni00000048/uni0000004f
/uni00000016/uni00000011/uni00000018/uni0000001c/uni00000016/uni00000011/uni0000001b/uni0000001b/uni0000001a/uni00000011/uni00000018/uni00000015/uni00000018/uni00000011/uni00000014/uni00000016/uni00000014/uni00000011/uni00000016/uni00000017/uni00000016/uni00000011/uni00000016/uni00000014/uni00000013/uni00000011/uni00000017/uni0000001c/uni00000014/uni0000001c/uni00000011/uni00000013/uni00000016/uni00000016/uni00000011/uni00000014/uni00000013/uni00000014/uni00000013/uni00000011/uni0000001c/uni00000019/uni00000017/uni00000011/uni00000014/uni00000018/uni00000018/uni00000011/uni00000017/uni0000001a/uni00000019/uni00000011/uni0000001b/uni00000015
120 125 130 135/uni00000031/uni00000020/uni00000014/uni00000015/uni0000001b/uni00000011/uni0000001c/uni00000015
/uni00000025/uni00000026/uni00000028
/uni00000025/uni0000002a/uni00000028
/uni0000002d/uni0000004c/uni00000051/uni00000044/uni0000002f/uni0000004c/uni00000056/uni00000057/uni00000037/uni00000018
/uni00000030/uni0000004c/uni00000051/uni0000004c/uni0000002f/uni00000030
/uni00000035/uni00000044/uni00000051/uni0000004e/uni00000037/uni00000018/uni00000036/uni00000033/uni0000002f/uni00000024/uni00000027/uni00000028
/uni00000037/uni0000005a/uni00000052/uni0000004f/uni00000044/uni00000055
/uni00000026/uni00000052/uni0000004f/uni00000025/uni00000048/uni00000055/uni00000057/uni00000030/uni0000003b/uni00000025/uni00000024/uni0000002c
/uni0000002c/uni00000051/uni00000010/uni00000035/uni00000044/uni00000051/uni0000004e
/uni0000002f/uni0000002f/uni00000030/uni00000015/uni00000039/uni00000048/uni00000046/uni00000035/uni00000048/uni00000044/uni00000055/uni00000044/uni00000051/uni0000004e
/uni00000035/uni00000044/uni00000051/uni0000004e/uni0000002a/uni00000033/uni00000037
Figure 6: Running efficiency analysis of rerankers.
0 10 20 30 40 50 60 70 80 90
Context IndexBase (Before)
Base (After)
NC (Before)
NC (After)
CC (Before)
CC (After)
SSLI (Before)
SSLI (After)Task
0 20 40 60 80 100
Relevance Score
Figure 7: Visualization of question-context relevance (before and after rerankering) under 4 evaluation tasks.
like MXBAI incur higher cost (10.96s) due to full query-document
interaction. Twolar emphasizes ranking accuracy through a two-
stage, listwise-aware architecture. Thus, it takes more time than
others. LLM2Vec uses frozen LLMs to compute dense vector simi-
larities without decoding, resulting in a moderate inference time
of 5.47s due to representation extraction and comparison. Rearank
adopts an agent-based framework with iterative LLM reasoning,
leading to higher latency (6.82s) mainly from multi-round docu-
ment evaluation and reordering. RankGPT leverages the reasoning
power of large language models for reranking through a generative,
prompt-based approach. Its extremely slow inference speed, which
is caused by having to process each candidate document one at a
time through a LLM.
Takeaways.
•Better reranking performance often comes at the cost of increased infer-
ence time.
4.7 Question-Context Relevance Under Four
Task Settings
To better understand the effectiveness of the rerankers in distin-
guishing relevant from irrelevant contexts, we visualize the context
ranking distributions before and after reranking in 4 tasks. We
use DeepSeek-V3 671B to assess the semantic relevance between
contexts and the given question. The visualization results of the rel-
evance heatmap are shown in Figure 7. In the heatmap visualization,
we group the results of the same task into pairs, allowing us to ob-
serve changes in color intensity to evaluate the effectiveness of thereranking. An overview of the 4 tasks reveals that after reranking,
the relevant contexts are clearly moved to the top positions across
all tasks. Particularly in the NC task, even when faced with a large
amount of irrelevant context, the rerankers is still able to correctly
prioritize the relevant contexts. However, in the SSLI and CC tasks,
the model incorrectly ranks irrelevant contexts among the top ,
leading to poor performance in LLM as discussed in Section 4.4.
This suggests that the rerankers limitations in its ability to perform
fine-grained semantic reasoning beyond surface-level relevance.
Takeaways.
•Rerankers effectively promote relevant contexts to higher positions across
multiple tasks, especially in noisy settings, but still struggle in scenarios
that require deep semantic understanding beyond surface-level similar-
ity.
5 Conclusion
In this paper, we introduce SciRerankBench. To the best of our
knowledge, it is the first publicly available benchmark for evaluat-
ing rerankers in RAG-LLMs towards scientific domains. We design
datasets covering noisy contexts, semantically similar but logically
irrelevant contexts and counterfactual contexts comprehensively
evaluate the rerankers. Evaluating 11 rerankers and 11 LLMs on
these tasks, our experiments show that while rerankers struggle to
filter logically irrelevant yet semantically similar passages. More-
over, final answer quality remains constrained by the inherent
reasoning limitations of current LLMs.

SciRerankBench: Benchmarking Rerankers Towards Scientific Retrieval-Augmented Generated LLMs Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
References
[1]Abdelrahman Abdallah, Bhawna Piryani, Jamshid Mozafari, Mohammed Ali,
and Adam Jatowt. 2025. Rankify: A comprehensive python toolkit for retrieval,
re-ranking, and retrieval-augmented generation. arXiv preprint arXiv:2502.02464
(2025).
[2]Davide Baldelli, Junfeng Jiang, Akiko Aizawa, and Paolo Torroni. 2024. TWOLAR:
a TWO-step LLM-Augmented distillation method for passage Reranking. In
European Conference on Information Retrieval . Springer, 470–485.
[3]Rodolphe Barrangou and Jennifer A Doudna. 2016. Applications of CRISPR
technologies in research and beyond. Nature biotechnology 34, 9 (2016), 933–941.
[4]Parishad BehnamGhader, Vaibhav Adlakha, Marius Mosbach, Dzmitry Bahdanau,
Nicolas Chapados, and Siva Reddy. 2024. Llm2vec: Large language models are
secretly powerful text encoders. arXiv preprint arXiv:2404.05961 (2024).
[5]Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun
Chen, Zehui Chen, Zhi Chen, Pei Chu, et al .2024. Internlm2 technical report.
arXiv preprint arXiv:2403.17297 (2024).
[6]Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024.
Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text
embeddings through self-knowledge distillation. arXiv preprint arXiv:2402.03216
(2024).
[7]Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa
Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering?
try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457 (2018).
[8]Jialin Dong, Bahare Fatemi, Bryan Perozzi, Lin F Yang, and Anton Tsitsulin.
2024. Don’t forget to connect! improving rag with graph-based reranking. arXiv
preprint arXiv:2405.18414 (2024).
[9]José Cassio dos Santos Junior, Rachel Hu, Richard Song, and Yunfei Bai. 2024.
Domain-Driven LLM Development: Insights into RAG and Fine-Tuning Practices.
InProceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and
Data Mining . 6416–6417.
[10] Feiteng Fang, Yuelin Bai, Shiwen Ni, Min Yang, Xiaojun Chen, and Ruifeng Xu.
2024. Enhancing Noise Robustness of Retrieval-Augmented Language Models
with Adaptive Adversarial Training. In Proceedings of the 62nd Annual Meeting of
the Association for Computational Linguistics (ACL) . 10028–10039.
[11] Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. 2021. SPLADE:
Sparse lexical and expansion model for first stage ranking. In Proceedings of
the 44th International ACM SIGIR Conference on Research and Development in
Information Retrieval . 2288–2292.
[12] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin
Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al .2025. Deepseek-r1:
Incentivizing reasoning capability in llms via reinforcement learning. arXiv
preprint arXiv:2501.12948 (2025).
[13] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric
Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem
solving with the math dataset. arXiv preprint arXiv:2103.03874 (2021).
[14] Robert Hoffmann and Alfonso Valencia. 2004. A gene network for navigating
the literature. Nature genetics 36, 7 (2004), 664–664.
[15] Patrick D Hsu, David A Scott, Joshua A Weinstein, F Ann Ran, Silvana Konermann,
Vineeta Agarwala, Yinqing Li, Eli J Fine, Xuebing Wu, Ophir Shalem, et al .2013.
DNA targeting specificity of RNA-guided Cas9 nucleases. Nature biotechnology
31, 9 (2013), 827–832.
[16] Tor-Kristian Jenssen, Astrid Lægreid, Jan Komorowski, and Eivind Hovig. 2001.
A literature network of human genes for high-throughput analysis of gene
expression. Nature genetics 28, 1 (2001), 21–28.
[17] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, De-
vendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel,
Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux,
Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix,
and William El Sayed. 2023. Mistral 7B. arXiv:2310.06825 [cs.CL] https:
//arxiv.org/abs/2310.06825
[18] Jiajie Jin, Yutao Zhu, Guanting Dong, Yuyao Zhang, Xinyu Yang, Chenghao
Zhang, Tong Zhao, Zhao Yang, Zhicheng Dou, and Ji-Rong Wen. 2024. Flashrag:
A modular toolkit for efficient retrieval-augmented generation research. arXiv
preprint arXiv:2405.13576 (2024).
[19] Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William W Cohen, and Xinghua Lu.
2019. Pubmedqa: A dataset for biomedical research question answering. arXiv
preprint arXiv:1909.06146 (2019).
[20] Qiao Jin, Robert Leaman, and Zhiyong Lu. 2024. PubMed and beyond: biomedical
literature search in the age of artificial intelligence. EBioMedicine 100 (2024).
[21] Jina AI. 2024. jina-reranker-v2-base-multilingual. https://huggingface.co/jinaai/
jina-reranker-v2-base-multilingual.
[22] Ebba Johannesson and Elise Olsson. 2025. Evaluating re-ranking methods in
LLM search. LU-CS-EX (2025).
[23] Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. 2017. Triviaqa:
A large scale distantly supervised challenge dataset for reading comprehension.
arXiv preprint arXiv:1705.03551 (2017).[24] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur
Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton
Lee, et al .2019. Natural questions: a benchmark for question answering research.
Transactions of the Association for Computational Linguistics 7 (2019), 453–466.
[25] Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. 2017. Program
induction by rationale generation: Learning to solve and explain algebraic word
problems. arXiv preprint arXiv:1705.04146 (2017).
[26] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Cheng-
gang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al .2024. Deepseek-v3
technical report. arXiv preprint arXiv:2412.19437 (2024).
[27] Qingqing Long, Shuai Liu, Ning Cao, Zhicheng Ren, Wei Ju, Chen Fang, Zhihong
Zhu, Hengshu Zhu, and Yuanchun Zhou. 2025. A survey of large language models
for traffic forecasting: Methods and applications. Authorea Preprints (2025).
[28] Qingqing Long, Lingjun Xu, Zheng Fang, and Guojie Song. 2021. Hgk-gnn:
Heterogeneous graph kernel based graph neural networks. In Proceedings of the
27th ACM SIGKDD conference on knowledge discovery & data mining . 1129–1138.
[29] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. 2018. Can a suit
of armor conduct electricity? a new dataset for open book question answering.
arXiv preprint arXiv:1809.02789 (2018).
[30] Adrian Mirza, Nawaf Alampara, Sreekanth Kunchapu, Martiño Ríos-García,
Benedict Emoekabu, Aswanth Krishnan, Tanya Gupta, Mara Schilling-Wilhelmi,
Macjonathan Okereke, Anagha Aneesh, et al .2024. Are large language models
superhuman chemists? arXiv preprint arXiv:2404.01475 (2024).
[31] Inc. NetEase Youdao. 2023. BCEmbedding: Bilingual and Crosslingual Embedding
for RAG. https://github.com/netease-youdao/BCEmbedding.
[32] Liangming Pan, Wenhu Chen, Wenhan Xiong, Min-Yen Kan, and William Yang
Wang. 2021. Unsupervised Multi-hop Question Answering by Question Gen-
eration. In Proceedings of the 2021 Conference of the North American Chapter
of the Association for Computational Linguistics: Human Language Technologies ,
Kristina Toutanova, Anna Rumshisky, Luke Zettlemoyer, Dilek Hakkani-Tur,
Iz Beltagy, Steven Bethard, Ryan Cotterell, Tanmoy Chakraborty, and Yichao
Zhou (Eds.). Association for Computational Linguistics, Online, 5866–5880.
doi:10.18653/v1/2021.naacl-main.469
[33] Jason Priem, Heather Piwowar, and Richard Orr. 2022. OpenAlex: A fully-open
index of scholarly works, authors, venues, institutions, and concepts. arXiv
preprint arXiv:2205.01833 (2022).
[34] Qdrant Team. 2024. Qdrant: Vector similarity search engine. https://github.com/
qdrant/qdrant.
[35] Chuan Qin, Xin Chen, Chengrui Wang, Pengmin Wu, Xi Chen, Yihang Cheng,
Jingyi Zhao, Meng Xiao, Xiangchao Dong, Qingqing Long, et al .2025. Scihorizon:
Benchmarking ai-for-science readiness from scientific data to large language
models. arXiv preprint arXiv:2503.13503 (2025).
[36] Shi Qiu, Shaoyang Guo, Zhuo-Yang Song, Yunbo Sun, Zeyu Cai, Jiashen Wei,
Tianyu Luo, Yixuan Yin, Haoxu Zhang, Yi Hu, et al .2025. PHYBench: Holistic
Evaluation of Physical Perception and Reasoning in Large Language Models.
arXiv preprint arXiv:2504.16074 (2025).
[37] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei
Zaharia. 2021. Colbertv2: Effective and efficient retrieval via lightweight late
interaction. arXiv preprint arXiv:2112.01488 (2021).
[38] Aamir Shakir, Darius Koenig, Julius Lipp, and Sean Lee. 2024. Boost Your Search
With The Crispy Mixedbread Rerank Models . https://www.mixedbread.ai/blog/
mxbai-rerank-v1
[39] Amanpreet Singh, Mike D’Arcy, Arman Cohan, Doug Downey, and Sergey Feld-
man. 2022. Scirepeval: A multi-format benchmark for scientific document repre-
sentations. arXiv preprint arXiv:2211.13308 (2022).
[40] Liangtai Sun, Yang Han, Zihan Zhao, Da Ma, Zhennan Shen, Baocai Chen, Lu
Chen, and Kai Yu. 2024. Scieval: A multi-level large language model evaluation
benchmark for scientific research. In Proceedings of the AAAI Conference on
Artificial Intelligence , Vol. 38. 19053–19061.
[41] Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, Zhumin
Chen, Dawei Yin, and Zhaochun Ren. 2023. Is ChatGPT good at search? investigat-
ing large language models as re-ranking agents. arXiv preprint arXiv:2304.09542
(2023).
[42] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yas-
mine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhos-
ale, et al .2023. Llama 2: Open foundation and fine-tuned chat models. arXiv
preprint arXiv:2307.09288 (2023).
[43] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2022. MuSiQue: Multihop Questions via Single-hop Question Composition.
Transactions of the Association for Computational Linguistics 10 (2022), 539–554.
[44] George Tsatsaronis, Michael Schroeder, Georgios Paliouras, Yannis Almirantis,
Ion Androutsopoulos, Eric Gaussier, Patrick Gallinari, Thierry Artieres, Michael R
Alvers, Matthias Zschunke, et al .2012. BioASQ: A Challenge on Large-Scale
Biomedical Semantic Indexing and Question Answering.. In AAAI fall symposium:
Information retrieval and knowledge discovery in biomedical text . Arlington, VA:
Citeseer.
[45] Asahi Ushio, Fernando Alva-Manchego, and Jose Camacho-Collados. 2023. An
Empirical Comparison of LM-based Question and Answer Generation Methods. In

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Haotian Chen et al.
Findings of the Association for Computational Linguistics: ACL 2023 , Anna Rogers,
Jordan Boyd-Graber, and Naoaki Okazaki (Eds.). Association for Computational
Linguistics, Toronto, Canada, 14262–14272. doi:10.18653/v1/2023.findings-acl.899
[46] Lucy Lu Wang, Kyle Lo, Yoganand Chandrasekhar, Russell Reas, Jiangjiang Yang,
Douglas Burdick, Darrin Eide, Kathryn Funk, Yannis Katsis, Rodney Kinney, et al .
2020. Cord-19: The covid-19 open research dataset. ArXiv (2020), arXiv–2004.
[47] Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou.
2020. Minilm: Deep self-attention distillation for task-agnostic compression of
pre-trained transformers. Advances in neural information processing systems 33
(2020), 5776–5788.
[48] Xiaoxuan Wang, Ziniu Hu, Pan Lu, Yanqiao Zhu, Jieyu Zhang, Satyen Subra-
maniam, Arjun R Loomba, Shichang Zhang, Yizhou Sun, and Wei Wang. 2023.
Scibench: Evaluating college-level scientific problem-solving abilities of large
language models. arXiv preprint arXiv:2307.10635 (2023).
[49] Zhenghao Wang, Shengquan Yan, Huaming Wang, and Xuedong Huang. 2014.
An overview of microsoft deep qa system on stanford webquestions benchmark.
Microsoft Corporation. Microsoft Research Technical Report MSR-TR-2014-121 18
(2014), 2014.
[50] Di Wu, Raymond Zhang, Enrico M Zucchelli, Yongchao Chen, and Richard
Linares. 2025. APBench and benchmarking large language model performance
in fundamental astrodynamics problems for space engineering. Scientific Reports
15, 1 (2025), 7944.
[51] Zhiyu Wu, Xiaokang Chen, Zizheng Pan, Xingchao Liu, Wen Liu, Damai Dai,
Huazuo Gao, Yiyang Ma, Chengyue Wu, Bingxuan Wang, et al .2024. Deepseek-
vl2: Mixture-of-experts vision-language models for advanced multimodal under-
standing. arXiv preprint arXiv:2412.10302 (2024).
[52] An Yang and Baosong Yang etc. 2024. Qwen2 Technical Report. arXiv preprint
arXiv:2407.10671 (2024).
[53] Xiao Yang, Kai Sun, Hao Xin, Yushi Sun, Nikita Bhalla, Xiangsen Chen, Sajal
Choudhary, Rongze Gui, Ziran Jiang, Ziyu Jiang, et al .2024. Crag-comprehensive
rag benchmark. Advances in Neural Information Processing Systems 37 (2024),
10470–10490.
[54] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan
Salakhutdinov, and Christopher D Manning. 2018. HotpotQA: A dataset for di-
verse, explainable multi-hop question answering. arXiv preprint arXiv:1809.09600
(2018).
[55] Soyoung Yoon, Eunbi Choi, Jiyeon Kim, Hyeongu Yun, Yireun Kim, and Seung-
won Hwang. 2024. Listt5: Listwise reranking with fusion-in-decoder improves
zero-shot retrieval. arXiv preprint arXiv:2402.15838 (2024).
[56] Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan You, Chao Zhang, Moham-
mad Shoeybi, and Bryan Catanzaro. 2024. Rankrag: Unifying context ranking
with retrieval-augmented generation in llms. Advances in Neural Information
Processing Systems 37 (2024), 121156–121184.
[57] Gianpaolo Zammarchi, Andrea Carta, Silvia Columbu, Luca Frigau, and Monica
Musio. 2024. A scientometric analysis of the effect of COVID-19 on the spread of
research outputs. Quality & Quantity 58, 3 (2024), 2265–2287.
[58] Le Zhang, Bo Wang, Xipeng Qiu, Siva Reddy, and Aishwarya Agrawal. 2025.
REARANK: Reasoning Re-ranking Agent via Reinforcement Learning. arXiv
preprint arXiv:2505.20046 (2025).
[59] Zhihao Zhang, Carrie-Ann Wilson, Rachel Hay, Yvette Everingham, and Usman
Naseem. 2025. BeefBot: Harnessing Advanced LLM and RAG Techniques for
Providing Scientific and Technology Solutions to Beef Producers. In Proceed-
ings of the 31st International Conference on Computational Linguistics: System
Demonstrations . 54–62.
[60] Jiawei Zhao, Yifei Zhang, Beidi Chen, Florian Schäfer, and Anima Anandkumar.
2023. Inrank: Incremental low-rank learning. arXiv preprint arXiv:2306.11250
(2023).
[61] Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang,
Amin Saied, Weizhu Chen, and Nan Duan. 2023. Agieval: A human-centric
benchmark for evaluating foundation models. arXiv preprint arXiv:2304.06364
(2023).
[62] Zhihong Zhu, Yunyan Zhang, Xianwei Zhuang, Fan Zhang, Zhongwei Wan,
Yuyan Chen, Qingqing Long, Yefeng Zheng, and Xian Wu. 2025. Can We Trust AI
Doctors? A Survey of Medical Hallucination in Large Language and Large Vision-
Language Models. In Findings of the Association for Computational Linguistics:
ACL 2025 . 6748–6769.
[63] Honglei Zhuang, Zhen Qin, Rolf Jagerman, Kai Hui, Ji Ma, Jing Lu, Jianmo Ni,
Xuanhui Wang, and Michael Bendersky. 2023. Rankt5: Fine-tuning t5 for text
ranking with ranking losses. In Proceedings of the 46th International ACM SIGIR
Conference on Research and Development in Information Retrieval . 2308–2313.
[64] Shengyao Zhuang, Xueguang Ma, Bevan Koopman, Jimmy Lin, and Guido Zuccon.
2025. Rank-R1: Enhancing Reasoning in LLM-based Document Rerankers via
Reinforcement Learning. arXiv preprint arXiv:2503.06034 (2025).

SciRerankBench: Benchmarking Rerankers Towards Scientific Retrieval-Augmented Generated LLMs Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
A Appendix
A.1 Detailed Dataset Construction Information
Source Literature. We choose OpenAlex [ 33] as our source data
to construct our synthetic scientific questions. OpenAlex is an
open, comprehensive scholarly metadata platform that provides
structured information on research papers, authors, institutions,
concepts, and more. This database contains over 250 million works,
more than 100 million unique author profiles, and metadata for
approximately 120,000 publication venues, including both journals
and conferences.
Dataset Construction Pseudocode. Algorithm 1 outlines the dataset
construction pipeline, including abstract preprocessing, single- and
multi-hop QA pair generation, and the creation of three datasets
which designed to evaluate different challenges.
Algorithm 1: Dataset Construction Process
Input: Source scientific corpus 𝐶
Output: QA Pairs𝐷𝑄𝐴, Reranking Set 𝐷𝑟𝑎𝑛𝑘
1// Step 1: Preprocessing & Indexing ;
2𝐶′←Filter abstracts: len ∈[100,500];
3𝑉←𝐸𝑏𝑔𝑒(𝐶′);
4Insert(𝑉into Qdrant index 𝐼);
5// Step 2: QA Generation ;
6foreach𝑎∈𝐶′do
7(𝑞1,𝑎𝑛𝑠 1)← LMQG(𝑎);
8 Add(𝑞1,𝑎𝑛𝑠 1)to𝐷𝑠𝑖𝑛𝑔𝑙𝑒
𝑄𝐴;
9𝑎′←NearestNeighbor(𝑎,𝐼);
10 if𝑎′≠NULL then
11(𝑞2,𝑎𝑛𝑠 2)← UMQG(𝑎,𝑎′);
12 Add(𝑞2,𝑎𝑛𝑠 2)to𝐷𝑚𝑢𝑙𝑡𝑖
𝑄𝐴;
13// Step 3: Construct Reranking Subsets ;
14foreach(𝑞,𝑎𝑛𝑠)∈𝐷𝑠𝑖𝑛𝑔𝑙𝑒
𝑄𝐴do
15𝑃←DenseRetrieve(𝑞,𝐼,𝑘 =100);
16𝑃𝑟𝑎𝑛𝑑←Mix(5·Rel,95·Random);
17𝑃𝑐𝑜𝑛𝑓←InjectConfusers(𝑃,10);
18𝑃𝑐𝑓𝑎𝑐𝑡←InjectCounterfacts(𝑃,10);
19 Add(𝑞,𝑃𝑟𝑎𝑛𝑑,𝑎𝑛𝑠)to𝐷𝑟𝑎𝑛𝑘;
20 Add(𝑞,𝑃𝑐𝑜𝑛𝑓,𝑎𝑛𝑠)to𝐷𝑟𝑎𝑛𝑘;
21 Add(𝑞,𝑃𝑐𝑓𝑎𝑐𝑡,𝑎𝑛𝑠)to𝐷𝑟𝑎𝑛𝑘;
22// Step 4: Final Evaluation Set ;
23𝐷eval←𝐷𝑟𝑎𝑛𝑘∪𝐷𝑚𝑢𝑙𝑡𝑖
𝑄𝐴;
Specific Dataset Instances/Samples. Here we provide a specific
dataset instances or sample, which comprising a scientific ques-
tion, its corresponding golden answer, and a set of 100 contextual
passages. The sample is shown in the Figure 9 to Figure 11.
Dataset Statistics. Table 2 provides a detailed breakdown of the
number of QA instances across five scientific subjects—Biology,
Math, Physics, Geology, and Chemistry—under five distinct tasktypes. These task types correspond to the benchmark datasets de-
scribed previously: NC,Base,CC,Multi-hop,SSLI.
Table 2: QA Counts for different subjects and tasks.
Subject Task Type Number
BiologyNC 2499
Base 2499
CC 2496
Multi-Hop 1246
SSLI 2497
MathNC 2494
Base 2494
CC 2491
Multi-Hop 1631
SSLI 2493
PhysicsNC 2491
Base 2491
CC 2494
Multi-Hop 1425
SSLI 2492
GeologyNC 2493
Base 2493
CC 2493
Multi-Hop 1598
SSLI 2496
ChemistryNC 2499
Base 2499
CC 2497
Multi-Hop 1087
SSLI 2498
A.2 Detailed Experimental Settings
Detailed LLM Information. Detailed information of LLM models
evaluated in this paper is shown in Table 3. These models are
from 5 LLM families, i.e., Mistral, LLaMA, DeepSeek, Qwen and
InternLM2 , with parameter sizes ranging from 7B to 72B+.This
diverse set of LLMs also enables us to introduce a new evaluation
perspective—using the quality of LLM-generated answers as an
indicator for how well a reranker supports real task performance.
Detailed Reranker Information. Detailed information of the rerankers
is shown in Table 4.
Prompting Template. To standardize the evaluation of reranking
models, we adopt a unified prompting template that guides models
to generate concise answers based solely on the provided contexts.
As illustrated in Figure 8, the prompt explicitly instructs the model
to refrain from using external knowledge and to return “No Answer
Present" when no relevant information is found.
Detailed Information of Evaluation Protocols and Metrics. For
rerankers, we compare the top-ranked contexts against annotated
relevance labels to assess their ability to retrieve the most informa-
tive and factually correct passages for a given scientific question.
Recall@10 evaluate how many of the known relevant passages are

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Haotian Chen et al.
Table 3: LLMs evaluated in this paper.
Model Family Size
Mistral-7B-v0.2 Mistral 7B
Mistral-24B Mistral 24B
LLaMA-7B Meta (LLaMA) 7B
LLaMA2-70B Meta (LLaMA2) 70B
DeepSeek-V3-7B DeepSeek 7B
DeepSeek-V3-671B DeepSeek 671B
Qwen-7B Qwen 7B
Qwen-32B Qwen 32B
Qwen-72B Qwen 72B
InternLM2-7B InternLM2 7B
InternLM2-20B InternLM2 20B
Table 4: RAG reranking methods evaluated in this paper.
Model Category Reranking Strategy
BGE Transformer Cross-Encoder
Jina Transformer Cross-Encoder
BCE Transformer Cross-Encoder
MXBAI Transformer Cross-Encoder
MiniLM Sentence Transformer distilled self-attention
ColBert BERT late interaction mechanism
In-Rank Seq2Seq model tokenizes query-document pairs
SPLADE Sparse MLMs
RankT5 T5-based ranking losses
ListT5 T5-based m-ary tournament sort
Twolar T5-based two-step distillation approach
LLM2Vec LLM-based LLM Embedding Similarity
RankGPT LLM-based sliding window
Rearank Agent-based reinforcement learning
Our Prompting Template
Instructions:
You are given a question and a set of contexts. Your task is to
provide a clear and concise answer to the question based on the
contexts provided.
Answer based only on the contexts. If none are relevant, say: No
Answer Present.
Keep your answer short — ideally within 2 sentences. Do not
include references or restate the question.
—
QUESTION: {QUESTION}
CONTEXTS: {CONTEXTS}
###
ANSWER:
Figure 8: Prompt used in the Rerank Evaluation Taskranked within the 10, respectively, thus testing retrieval complete-
ness. For LLMs, we assess their ability to generate accurate and
complete answers by comparing outputs to gold-standard answers.
We use a token-level metrics: Recall , which captures how completely
the model reproduces the reference content.
Recall is computed using tokenized outputs. Let 𝑃and𝑇be the
predicted and true token sets. Then:
Recall =|𝑃∩𝑇|
|𝑇|×100,
This metrics reflect the lexical similarity between the predicted
answer and the ground truth, helping to quantify completeness in
LLM-generated outputs.
Contain Answer Score evaluates whether the gold answer con-
tent is covered within the retrieved context passage. This is a soft
semantic measure, defined as:
ContainAnswer(𝐴,𝐶)=|tokens(𝐴)∩tokens(𝐶)|
|tokens(𝐴)|
If the overlap exceeds a predefined threshold (e.g., 0.6), the passage is
considered to contain the answer. This metric is particularly useful
for evaluating context sufficiency in open-domain and scientific
QA.
Recall@10 compute how many relevant contexts are successfully
retrieved within the top- 𝑘results:
Recall@𝑘=# relevant items in top 𝑘
Total relevant items×100
Table 5: Performance of LLM-based and Agent-based rerankers
across various evaluation tasks
Reranker Subject Multi-Hop NC CC SSLI Base
LLM2VecBio. 33.72±0.36 32.77±0.45 31.78±0.52 31.34±0.61 31.46±0.50
Geo. 44.34±3.09 41.84±2.70 38.45±2.10 37.06±1.85 35.99±1.75
Chem. 55.28±0.84 53.75±0.88 51.35±1.10 49.39±1.05 49.45±0.90
Phy. 50.18±0.83 47.82±0.75 44.42±0.82 43.96±0.65 45.69±0.88
Math. 33.04±0.66 29.49±1.10 31.05±0.92 29.26±1.15 28.99±1.90
RearankBio. 49.98±0.79 48.57±1.27 47.11±0.81 46.48±2.16 46.69±0.94
Geo. 32.56±1.36 30.73±1.92 26.71±2.38 24.83±0.84 23.48±1.17
Chem. 49.87±1.67 48.51±0.88 46.32±1.21 44.56±1.18 44.64±0.61
Phy. 52.25±0.46 49.76±0.23 46.27±0.70 45.86±0.32 47.64±0.87
Math. 51.45±1.02 45.90±1.34 48.35±0.50 45.62±1.23 45.22±3.21
A.3 Detailed Experimental Results
Detailed main results are shown in Table 6 and Table 7. These tables
report the performance metrics of various rerankers across multiple
tasks and subjects under two different backbone LLMs, Qwen and
LLaMA. The comparison highlights the general effectiveness and ro-
bustness of the rerankers, with only minor performance variations
across subjects, suggesting that the models generalize well across
domains. In addition, Table 5 compares two representative rerank-
ing approaches: the LLM-based LLM2Vec and the agent-based Rear-
ank. This table offers a more focused analysis of how rerankers with
different architectural designs and inference paradigms perform
under the same evaluation settings. Together, these results provide
a comprehensive view of reranker behavior across model types,
task types, and subject domains, serving as a reference for future
system design and benchmarking.

SciRerankBench: Benchmarking Rerankers Towards Scientific Retrieval-Augmented Generated LLMs Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Table 6: Performance of different rerankers across various evaluation tasks (Qwen-70B)
Reranker Subject Multi-Hop NC CC SSLI Base
BCEBio. 55.08±2.62 53.57±2.20 51.66±0.48 51.89±1.40 63.48±0.57
Geo. 52.44±1.23 51.84±1.53 48.94±1.01 48.14±1.73 60.20±0.42
Chem. 53.62±2.08 52.65±0.81 49.74±1.22 49.44±2.50 59.61±0.75
Phy. 49.77±0.90 51.93±1.66 45.32±1.12 46.05±1.45 61.94±0.74
Math. 52.51±1.48 52.38±0.18 49.26±1.47 47.84±1.52 62.14±2.30
BGEBio. 54.55±1.70 53.56±2.00 55.52±0.79 54.56±1.16 64.74±0.71
Geo. 51.13±0.84 52.25±1.18 52.99±0.59 51.37±1.17 60.91±0.57
Chem. 52.15±1.62 53.00±0.55 52.88±1.55 51.01±0.89 60.47±1.39
Phy. 50.45±0.23 52.57±1.47 49.00±1.60 50.37±1.43 61.66±0.73
Math. 51.45±1.69 52.70±0.29 52.43±0.94 50.92±1.61 62.16±1.63
JinaBio. 51.52±1.78 53.89±1.87 53.22±0.83 51.87±1.28 62.85±0.78
Geo. 47.44±0.95 51.99±1.41 50.92±0.89 49.92±1.36 60.26±0.61
Chem. 49.74±1.16 52.54±0.73 52.20±1.45 51.97±2.40 59.20±0.79
Phy. 47.53±1.19 52.01±1.46 48.35±1.62 49.67±2.22 60.33±0.69
Math. 51.69±1.17 52.76±0.32 52.37±0.54 51.28±1.37 61.30±1.77
ListT5Bio. 8.89±0.54 26.71±1.28 15.54±1.23 14.69±0.95 15.62±0.60
Geo. 10.19±2.28 27.20±1.56 15.39±0.56 13.54±1.01 12.83±0.81
Chem. 8.60±0.72 16.51±0.90 11.21±0.61 2.06±0.32 11.42±0.99
Phy. 8.80±0.77 20.97±0.95 12.28±0.82 12.01±1.32 10.95±0.85
Math. 9.17±1.19 28.30±1.46 13.68±0.74 13.41±0.94 12.71±0.89
MiniLMBio. 52.95±2.53 53.72±1.92 51.17±1.41 51.29±1.65 63.92±0.94
Geo. 48.78±0.38 51.75±1.74 48.82±1.30 48.18±2.28 60.29±0.70
Chem. 50.87±2.57 52.67±0.67 49.22±2.01 48.64±1.78 59.44±0.59
Phy. 46.71±1.17 52.24±1.70 45.25±0.74 46.55±3.55 60.98±0.66
Math. 49.64±1.64 52.77±0.69 49.15±0.48 47.83±2.21 61.75±0.50
RankT5Bio. 14.98±1.77 34.12±1.99 21.08±1.62 20.64±2.36 19.18±0.59
Geo. 16.78±1.76 34.69±1.10 22.76±1.31 19.44±2.47 19.57±1.55
Chem. 12.06±1.61 21.89±0.29 17.04±0.63 49.44±0.87 11.32±0.54
Phy. 14.83±0.78 27.76±1.34 18.48±2.70 18.89±2.50 14.00±1.23
Math. 18.73±0.76 35.56±0.91 21.57±4.64 20.37±1.78 18.47±0.29
SPLADEBio. 14.16±1.12 34.28±2.45 23.40±1.51 19.05±1.26 19.04±0.48
Geo. 17.27±0.37 34.26±1.30 22.45±1.28 20.65±0.55 19.39±1.10
Chem. 12.31±1.51 21.84±0.77 18.35±1.95 42.71±3.13 10.66±0.02
Phy. 14.21±1.25 28.27±1.96 20.49±1.81 21.10±2.76 14.43±0.44
Math. 19.05±0.53 35.26±0.27 20.39±3.23 20.30±2.48 18.58±0.47
TwoLARBio. 53.69±1.41 53.53±2.04 56.17±0.72 57.24±1.11 63.55±0.84
Geo. 50.78±0.42 52.05±1.76 55.36±0.96 53.59±1.30 61.12±0.54
Chem. 52.02±2.02 53.05±0.69 54.20±1.66 53.40±0.85 60.13±0.48
Phy. 49.57±1.20 52.40±1.70 51.65±0.90 52.19±1.05 61.91±0.37
Math. 52.10±1.39 53.07±0.47 54.33±0.23 52.90±1.60 62.43±0.41
ColBertBio. 53.35±1.96 53.63±1.98 44.92±0.32 44.38±0.74 62.87±0.69
Geo. 50.35±0.93 51.60±1.75 41.09±0.51 40.27±1.68 60.32±0.26
Chem. 52.07±1.80 52.82±0.65 42.12±1.04 42.24±1.51 58.96±0.88
Phy. 49.25±1.09 52.30±1.38 40.30±0.74 39.19±2.83 61.10±0.46
Math. 51.12±0.79 52.68±0.26 41.97±0.82 41.25±2.11 61.74±0.70
MXBAIBio. 53.56±3.25 53.62±1.71 57.18±0.53 58.40±0.96 62.25±0.67
Geo. 49.26±1.22 51.72±1.48 56.45±0.43 55.40±1.63 59.67±0.64
Chem. 51.43±1.56 53.05±0.59 54.29±1.12 55.42±1.05 58.83±0.47
Phy. 48.57±0.74 52.29±1.82 53.33±0.34 54.45±1.89 61.09±0.70
Math. 50.29±2.11 52.92±0.16 56.61±0.44 54.39±2.41 61.02±0.40
T5Bio. 54.45±1.46 53.64±1.75 53.92±1.13 53.81±1.17 63.96±0.99
Geo. 51.08±1.04 51.87±1.64 52.46±0.78 51.40±0.54 60.65±0.06
Chem. 53.08±1.68 52.76±0.59 52.44±1.87 51.66±0.98 59.67±0.94
Phy. 50.41±0.86 52.20±1.23 49.11±1.03 48.59±2.62 61.44±0.76
Math. 52.37±1.38 52.66±0.31 52.95±0.25 51.28±2.12 62.14±0.55

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Haotian Chen et al.
Table 7: Performance of different rerankers across various evaluation tasks (LLaMA2-70B)
Reranker Subject Multi-Hop NC CC SSLI Base
BCEBio. 48.97±1.83 51.82±0.48 49.73±0.89 50.50±1.35 60.57±0.48
Geo. 50.32±1.20 51.94±0.70 47.85±0.95 48.30±1.40 59.12±0.65
Chem. 50.40±1.15 51.40±0.60 48.92±1.05 49.40±1.20 58.84±0.73
Phy. 48.36±1.32 51.55±0.88 45.21±0.90 46.82±1.48 61.78±1.12
Math. 49.55±1.18 51.68±1.12 49.50±1.00 48.96±1.33 60.88±1.40
BGEBio. 52.08±1.66 52.84±0.36 52.32±1.25 53.63±1.10 60.95±0.74
Geo. 51.67±1.45 52.67±0.59 49.35±0.83 51.48±1.18 60.42±0.55
Chem. 51.74±1.30 52.40±0.64 50.23±1.00 51.92±1.35 60.26±0.88
Phy. 50.86±1.17 52.55±0.74 47.96±1.08 50.12±1.29 62.20±1.15
Math. 52.11±1.25 52.69±0.77 51.12±0.91 52.23±1.31 61.03±1.20
JinaBio. 47.34±1.38 53.19±0.67 52.00±0.65 51.41±1.55 60.38±1.19
Geo. 47.67±1.05 52.51±0.74 49.28±1.10 49.21±1.40 59.01±1.30
Chem. 48.20±1.25 52.13±0.85 50.01±1.03 49.97±1.32 58.32±1.20
Phy. 46.33±1.36 52.60±0.92 47.39±1.07 47.88±1.45 61.14±1.08
Math. 47.75±1.12 52.88±0.68 50.23±1.01 49.66±1.44 60.42±1.22
ListT5Bio. 16.70±0.29 25.82±1.73 25.69±1.19 26.42±0.42 26.23±1.57
Geo. 10.19±2.28 27.20±1.56 15.39±0.56 13.54±1.01 12.83±0.81
Chem. 8.60±0.72 16.51±0.90 11.21±0.61 2.06±0.32 11.42±0.99
Phy. 8.80±0.77 20.97±0.95 12.28±0.82 12.01±1.32 10.95±0.85
Math. 9.17±1.19 28.30±1.46 13.68±0.74 13.41±0.94 12.71±0.89
MiniLMBio. 48.14±1.91 51.68±0.32 49.83±0.35 50.19±1.59 60.56±0.83
Geo. 49.20±1.45 51.33±0.39 47.55±0.67 48.72±1.51 59.37±0.61
Chem. 48.97±1.33 51.11±0.45 48.40±0.58 49.45±1.34 59.06±0.78
Phy. 47.25±1.12 51.26±0.52 45.71±0.62 46.91±1.47 61.34±0.91
Math. 48.45±1.28 51.38±0.48 48.73±0.59 49.18±1.46 60.61±0.85
RankT5Bio. 49.45±2.88 52.41±0.61 51.07±0.03 51.33±1.57 60.49±0.21
Geo. 50.11±2.02 52.10±0.57 48.84±0.72 50.08±1.49 59.73±0.42
Chem. 49.85±1.89 51.78±0.62 49.71±0.60 50.89±1.40 59.61±0.64
Phy. 48.23±1.74 52.05±0.68 47.22±0.65 48.98±1.47 61.41±0.72
Math. 49.35±2.03 52.17±0.66 50.26±0.51 50.70±1.35 60.55±0.85
SPLADEBio. 45.21±1.67 52.62±0.56 43.96±0.75 45.38±1.35 59.05±0.52
Geo. 45.62±1.21 52.38±0.67 42.41±0.89 43.76±1.50 58.30±0.40
Chem. 45.95±1.00 52.12±0.74 42.85±0.93 44.44±1.42 57.78±0.60
Phy. 44.16±1.09 52.33±0.59 40.92±0.95 42.88±1.51 60.22±0.73
Math. 44.90±1.30 52.50±0.62 43.50±0.88 44.48±1.53 59.11±0.90
TwoLARBio. 50.66±2.18 52.57±0.87 54.50±0.27 55.16±0.89 60.15±0.04
Geo. 51.12±1.55 52.44±0.72 51.58±0.61 52.75±0.91 59.40±0.27
Chem. 51.30±1.29 52.13±0.64 52.62±0.52 53.66±0.97 59.23±0.32
Phy. 50.20±1.46 52.34±0.67 50.15±0.59 51.41±0.86 61.55±0.58
Math. 50.89±1.34 52.47±0.71 53.22±0.44 53.83±0.87 60.33±0.61
ColBertBio. 49.23±1.92 52.52±1.24 43.49±0.51 44.99±1.50 59.76±1.50
Geo. 49.40±1.63 52.39±1.05 41.88±0.94 43.57±1.65 58.91±1.42
Chem. 49.10±1.45 52.15±1.12 42.41±0.92 44.15±1.49 58.73±1.35
Phy. 48.11±1.76 52.28±1.09 40.55±0.87 42.28±1.62 61.12±1.23
Math. 48.80±1.81 52.36±1.18 43.12±0.83 44.09±1.63 59.81±1.38
MXBAIBio. 49.18±2.16 53.41±0.27 55.75±0.87 56.04±1.19 59.77±1.65
Geo. 49.55±1.88 53.22±0.40 53.01±0.66 53.83±1.24 58.89±1.53
Chem. 49.62±1.42 52.95±0.47 54.12±0.74 54.49±1.13 58.70±1.40
Phy. 48.58±1.36 53.20±0.54 52.46±0.68 52.76±1.22 61.33±1.45
Math. 49.35±1.63 53.33±0.51 55.12±0.59 54.88±1.20 59.89±1.52
T5Bio. 51.47±2.23 53.30±1.08 51.80±0.62 52.73±1.31 59.95±0.25
Geo. 51.91±1.89 53.01±0.92 49.50±0.55 50.62±1.24 59.20±0.33
Chem. 51.86±1.74 52.60±0.89 50.36±0.47 51.31±1.25 59.10±0.41
Phy. 50.78±1.60 52.96±1.02 48.42±0.51 49.66±1.31 61.48±0.35
Math. 51.55±1.95 53.13±1.00 51.02±0.46 51.40±1.29 60.01±0.39

SciRerankBench: Benchmarking Rerankers Towards Scientific Retrieval-Augmented Generated LLMs Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
A Counterfactual Synthetic (Question-Context-Answer) Sample.
Question: What are two viable large-scale energy storage technologies?
Golden Answer: Underground Hydrogen Storage (UHS) and Compressed Air Energy storage (CAES).
Context:
•Passage 1 : Underground hydrogen storage (UHS) and compressed air energy storage (CAES) are two viable large-scale energy storage
technologies for mitigating the intermittency of wind and solar power. Therefore, it is meaningful to compare the properties of hydrogen
and air with typical thermodynamic storage processes. This study employs a multi-physical coupling model to compare the operations of
CAES and UHS, integrating gas thermodynamics within caverns, thermal conduction, and mechanical deformation around rock caverns. Gas
thermodynamic responses are validated using additional simulations and the field test data. Temperature and pressure variations of air and
hydrogen within rock caverns exhibit similarities under both adiabatic and diabatic simulation modes. Hydrogen reaches higher temperature
and pressure following gas charging stage compared to air, and the ideal gas assumption may lead to overestimation of gas temperature and
pressure. Unlike steel lining of CAES, the sealing layer (fibre-reinforced plastic FRP) in UHS is prone to deformation but can effectively
mitigates stress in the sealing layer. In CAES, the first principal stress on the surface of the sealing layer and concrete lining is tensile stress,
whereas UHS exhibits compressive stress in the same areas. Our present research can provide references for the selection of energy storage
methods.
•Passage 2 : Intensive increases in electrical energy storage are being driven by electric vehicles (EVs), smart grids, intermittent renewable
energy, and decarbonization of the energy economy. Advanced lithium–sulfur batteries (LSBs) are among the most promising candidates,
especially for EVs and grid-scale energy storage applications. In this topical review, the recent progress and perspectives of practical LSBs
are reviewed and discussed; the challenges and solutions for these LSBs are analyzed and proposed for future practical and large-scale
energy storage applications. Major challenges for the shuttle effect, reaction kinetics, and anodes are specifically addressed, and solutions are
provided on the basis of recent progress in electrodes, electrolytes, binders, interlayers, conductivity, electrocatalysis, artificial SEI layers, etc.
The characterization strategies (including in situ ones) and practical parameters (e.g., cost-effectiveness, battery management/modeling,
environmental adaptability) are assessed for crucial automotive/stationary large-scale energy storage applications (i.e., EVs and grid energy
storage). This topical review will give insights into the future development of promising Li–S batteries toward practical applications, including
EVs and grid storage.
•Passage 3 : To support increasing renewable capacity for a net-zero future, energy storage will play a key role in maintaining grid stability. In
this paper, all current and near-future energy storage technologies are compared for three different scenarios: (1) fixed electricity buy-in price,
(2) market-based electricity buy-in price, and (3) energy storage integrated into a fully renewable electricity system. In the first part of this
study, an algorithm is devised to simulate strategic buy-in of electricity for energy storage. This analysis yields a qualitative decision-making
tool for a given energy storage duration and size. Building upon the first part’s findings, an integration study gives insight into expected
power prices and expected storage size in a typical northwestern European fully renewable energy system. The integration study shows
significant need for electricity storage with durations spanning from one to several days, typically around 40 h. Pumped Hydro Storage and
Pumped Thermal storage surface as the best options. The overall levelized costs of storage are expected to be in the USD 200–500/MWh
range. Integration of storage with renewables can yield a system-levelized cost of electricity of about USD 150/MWh. Allowing flexibility in
demand may lower the overall system-levelized cost of electricity to USD 100/MWh.","Climate change mitigation requires the large-scale
deployment of carbon capture and storage (CCS). Recent plans indicate an eight-fold increase in CCS capacity by 2030, yet the feasibility of
CCS expansion is debated. Using historical growth of CCS and other policy-driven technologies, we show that if plans double between 2023
and 2025 and their failure rates decrease by half, CCS could reach 0.37 GtCO
•Passage 4 : With the escalating utilization of intermittent renewable energy sources, demand for durable and powerful energy storage systems
has increased to secure stable electricity supply. Redox flow batteries (RFBs) have received ever-increasing attention as promising energy
storage technologies for grid applications. However, their broad market penetration is still obstructed by many challenges, such as high
capital cost and inferior long-term stability. In this work, combining the merits of both all-vanadium and iron-chromium RFB systems, a
vanadium-chromium RFB (V/Cr RFB) is designed and fabricated. This proposed system possesses a high theoretical voltage of 1.41 V while
achieving cost effectiveness by using cheap chromium as one of the reactive species. Experimentally, the system attains a peak power density
of over 900 mW cm-2 at 50 °C and demonstrates stable performance for 50 cycles with an energy efficiency of over 87%, presenting this
system as a promising candidate for large-scale energy storage.
•. . .
•Passage 99 : Two large-scale energy storage technologies with limited safety: Sodium-ion batteries, which are known for their safety, and
lithium-ion batteries, which have a higher risk of thermal runaway, are both unsuitable for use in a safe energy storage system.
•Passage 100 : Two large-scale energy storage technologies with limited environmental sustainability: Lead-acid batteries, which are known
for their environmental sustainability, and lithium-ion batteries, which have a higher environmental impact due to their production and
disposal, are both unsuitable for use in an environmentally sustainable energy storage system.
Figure 9: A Counterfactual synthetic (Question, Context, Answer) Sample

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Haotian Chen et al.
A SSLI Synthetic (Question-Context-Answer) Sample.
Question: What is the most lethal melanoma subtype?
Golden Answer: uveal melanomas
Context:
•Passage 1 : Activating mutations in GNAQ/GNA11 occur in over 90% of uveal melanomas (UMs), the most lethal melanoma subtype; however,
targeting these oncogenes has proven challenging and inhibiting their downstream effectors show limited clinical efficacy. Here, we
performed genome-scale CRISPR screens along with computational analyses of cancer dependency and gene expression datasets to identify
the inositol-metabolizing phosphatase INPP5A as a selective dependency in GNAQ/11-mutant UM cells in vitro and in vivo. Mutant cells
intrinsically produce high levels of the second messenger inositol 1,4,5 trisphosphate (IP3) that accumulate upon suppression of INPP5A,
resulting in hyperactivation of IP3-receptor signaling, increased cytosolic calcium and p53-dependent apoptosis. Finally, we show that
GNAQ/11-mutant UM cells and patients’ tumors exhibit elevated levels of IP4, a biomarker of enhanced IP3 production; these high levels are
abolished by GNAQ/11 inhibition and correlate with sensitivity to INPP5A depletion. Our findings uncover INPP5A as a synthetic lethal
vulnerability and a potential therapeutic target for GNAQ/11-mutant-driven cancers.
•Passage 2 : Background Although anti-Program-Death-1 (PD-1) is a standard adjuvant therapy for patients with resected melanoma. We
hypothesize that there are discrepancies of survival, recurrence pattern and toxicity to adjuvant PD-1 between different ethnicities and
melanoma subtypes. Objective/methodsWe performed a multicenter cohort study incorporating 6 independent institutions in Australia, China,
Japan, and US.The primary outcomes were RFS and OS.Secondary outcomes were disease recurrence patterns and toxicities. ResultsIn total 534
patients were included.East-Asian/Hispanic/African had significantly poorer RFS/OS.Non-acral-cutaneous/unknown primary (NAC/UP) had
the best RFS/OS, followed by acral; mucosal the poorest.Within the NAC/UP subtypes, East-Asian/Hispanic/African had significantly poorer
RFS/OS than Caucasian.In the multivariate analysis incorporating ethnicity/melanoma-subtype/age/sex/stage/LDH/BRAFmutation/adjuvant-
radiotherapy, East-Asian/Hispanic/African had independently significantly poorer outcomes (RFS: HR1.71 95%CI 1.19-2.44;OS: HR2.34,
95%CI 1.39-3.95),as was mucosal subtype (RFS: HR3.25, 95%CI 2.04-5.17;OS: HR3.20, 95%CI 1.68-6.08).Mucosal melanoma was an inde-
pendent risk factor for distant-metastasis, especially liver metastasis.East-Asian/Hispanic/African had significantly lower incidence of
GI/musculoskeletal/respiratory/other-rare-type-toxicities; but higher incidences of liver toxicities. Limitations retrospective study.Conclusions
Ethnicity and melanoma subtype are associated with survival and recurrence pattern in melanoma patients treated with adjuvant anti-PD-
1.Toxicity profile differs by ethnicity, and may require a precision toxicity surveillance strategy.
•Passage 3 : Melanoma is the third most common type of skin cancer, characterized by its heterogeneity and propensity to metastasize
to distant organs. Melanoma is a heterogeneous tumor, composed of genetically divergent subpopulations, including a small fraction of
melanoma-initiating cancer stem-like cells (CSCs) and many non-cancer stem cells (non-CSCs). CSCs are characterized by their unique
surface proteins associated with aberrant signaling pathways with a causal or consequential relationship with tumor progression, drug
resistance, and recurrence. Melanomas also harbor significant alterations in functional genes (BRAF, CDKN2A, NRAS, TP53, and NF1). Of
these, the most common are the BRAF and NRAS oncogenes, with 50% of melanomas demonstrating the BRAF mutation (BRAF
•Passage 4 : Melanoma, the deadliest form of skin cancer, poses a significant clinical challenge for the development of effective treatments.
Conventional in vivo animal studies have shown limited translational relevance to humans, raising strength to pre-clinical models for
melanoma research. This review provides an in-depth analysis of alternative pre-clinical models including in vitro and ex vivo platforms
such as reconstructed skin, spheroids, organoids, organotypic models, skin-on-a-chip, and bioprinting. Through a comprehensive analysis,
the specific attributes, advantages, and limitations of each model are elucidated. It discusses the points related to the uniqueness advantages,
from capturing complex interactions between melanoma cells and their microenvironment to enabling high-throughput drug screening
and personalized medicine approaches. This review is structured covering firstly the roadmap to identify the co-occurrence of discovering
new melanoma treatments and the development of its models, secondly it covers a comparative between the most used models followed by
a section discussing each of them: the in vitro and ex vivo models. It intends to serve as an asset for researchers of melanoma field and
clinicians involved in melanoma therapy, offering insights into the diverse preclinical models available for optimizing their integration into
the translational pipeline.
•Passage 5 : Melanoma incidence and mortality rates are historically higher for men than women. Although emerging studies have highlighted
tumorigenic roles for the male sex hormone androgen and its receptor (AR) in melanoma, cellular and molecular mechanisms underlying
these sex-associated discrepancies are poorly defined. Here, we delineate a previously undisclosed mechanism by which androgen-activated
AR transcriptionally upregulates fucosyltransferase 4 ( FUT4 ) expression, which drives melanoma invasiveness by interfering with adherens
junctions (AJs). Global phosphoproteomic and fucoproteomic profiling, coupled with in vitro and in vivo functional validation, further reveal
that AR-induced FUT4 fucosylates L1 cell adhesion molecule (L1CAM), which is required for FUT4-increased metastatic capacity. Tumor
microarray and gene expression analyses demonstrate that AR-FUT4-L1CAM-AJs signaling correlates with pathological staging in melanoma
patients. By delineating key androgen-triggered signaling that enhances metastatic aggressiveness, our findings help explain sex-associated
clinical outcome disparities and highlight AR/FUT4 and its effectors as potential prognostic biomarkers and therapeutic targets in melanoma.
•. . .
•Passage 99 : Deceptive: The most lethal melanoma subtype is often misdiagnosed as a harmless freckle, making early detection challenging.
•Passage 100 : Perplexing: It’s a common misconception that the most lethal melanoma subtype is the one that appears as a mole.
Figure 10: A SSLI synthetic (Question, Context, Answer) Sample

SciRerankBench: Benchmarking Rerankers Towards Scientific Retrieval-Augmented Generated LLMs Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
A Multi-Hop Synthetic (Question-Context-Answer) Sample.
Question: What are the isoforms categories of genes that control activity of gene transcription?
Golden Answer: rewirers and negative regulators
Context:
•Passage 1 : Most human Transcription factors (TFs) genes encode multiple protein isoforms differing in DNA binding domains, effector
domains, or other protein regions. The global extent to which this results in functional differences between isoforms remains unknown. Here,
we systematically compared 693 isoforms of 246 TF genes, assessing DNA binding, protein binding, transcriptional activation, subcellular
localization, and condensate formation. Relative to reference isoforms, two-thirds of alternative TF isoforms exhibit differences in one or
more molecular activities, which often could not be predicted from sequence. We observed two primary categories of alternative TF isoforms:
¨rewirersänd ¨negative regulators¨, both of which were associated with differentiation and cancer. Our results support a model wherein the
relative expression levels of, and interactions involving, TF isoforms add an understudied layer of complexity to gene regulatory networks,
demonstrating the importance of isoform-aware characterization of TF functions and providing a rich resource for further studies.
•Passage 2 : More than 1600 human transcription factors orchestrate the transcriptional machinery to control gene expression and cell
fate. Their function is conveyed through intrinsically disordered regions (IDRs) containing activation or repression domains but lacking
quantitative structural ensemble models prevents their mechanistic decoding. Here we integrate single-molecule FRET and NMR spectroscopy
with molecular simulations showing that DNA binding can lead to complex changes in the IDR ensemble and accessibility. The C-terminal
IDR of pioneer factor Sox2 is highly disordered but its conformational dynamics are guided by weak and dynamic charge interactions with
the folded DNA binding domain. Both DNA and nucleosome binding induce major rearrangements in the IDR ensemble without affecting
DNA binding affinity. Remarkably, interdomain interactions are redistributed in complex with DNA leading to variable exposure of two
activation domains critical for transcription. Charged intramolecular interactions allowing for dynamic redistributions may be common in
transcription factors and necessary for sensitive tuning of structural ensembles.
•Passage 3 : Transcription factors (TFs) control specificity and activity of gene transcription, but whether a relationship between these two
features exists is unclear. Here we provide evidence for an evolutionary trade-off between the activity and specificity in human TFs encoded
as submaximal dispersion of aromatic residues in their intrinsically disordered protein regions. We identified approximately 500 human TFs
that encode short periodic blocks of aromatic residues in their intrinsically disordered regions, resembling imperfect prion-like sequences.
Mutation of periodic aromatic residues reduced transcriptional activity, whereas increasing the aromatic dispersion of multiple human TFs
enhanced transcriptional activity and reprogramming efficiency, promoted liquid–liquid phase separation in vitro and more promiscuous
DNA binding in cells. Together with recent work on enhancer elements, these results suggest an important evolutionary role of suboptimal
features in transcriptional control. We propose that rational engineering of amino acid features that alter phase separation may be a strategy
to optimize TF-dependent processes, including cellular reprogramming.
•Passage 4 : Determining whether the RNA isoforms from medically relevant genes have distinct functions could facilitate direct targeting of
RNA isoforms for disease treatment. Here, as a step toward this goal for neurological diseases, we sequenced 12 postmortem, aged human
frontal cortices (6 Alzheimer disease cases and 6 controls; 50% female) using one Oxford Nanopore PromethION flow cell per sample. We
identified 1,917 medically relevant genes expressing multiple isoforms in the frontal cortex where 1,018 had multiple isoforms with different
protein-coding sequences. Of these 1,018 genes, 57 are implicated in brain-related diseases including major depression, schizophrenia,
Parkinson’s disease and Alzheimer disease. Our study also uncovered 53 new RNA isoforms in medically relevant genes, including several
where the new isoform was one of the most highly expressed for that gene. We also reported on five mitochondrially encoded, spliced RNA
isoforms. We found 99 differentially expressed RNA isoforms between cases with Alzheimer disease and controls.
•. . .
•Passage 99 : Pioneer transcription factors (TFs) exhibit a specialized ability to bind to and open closed chromatin, facilitating engagement by
other regulatory factors involved in gene activation or repression. Chemical probes are lacking for pioneer TFs, which has hindered their
mechanistic investigation in cells. Here, we report the chemical proteomic discovery of electrophilic small molecules that stereoselectively
and site-specifically bind the pioneer TF, FOXA1, at a cysteine (C258) within the forkhead DNA-binding domain. We show that these
covalent ligands react with FOXA1 in a DNA-dependent manner and rapidly remodel its pioneer activity in prostate cancer cells reflected in
redistribution of FOXA1 binding across the genome and directionally correlated changes in chromatin accessibility. Motif analysis supports a
mechanism where the covalent ligands relax the canonical DNA binding preference of FOXA1 by strengthening interactions with suboptimal
ancillary sequences in predicted proximity to C258. Our findings reveal a striking plasticity underpinning the pioneering function of FOXA1
that can be controlled by small molecules.
•Passage 100 : Alternative transcription start site usage (ATSS) is a widespread regulatory strategy that enables genes to choose between
multiple genomic loci for initiating transcription. This mechanism is tightly controlled during development and is often altered in disease
states. In this review, we examine the growing evidence highlighting a role for transcription start sites (TSSs) in the regulation of mRNA
isoform selection during and after transcription. We discuss how the choice of transcription initiation sites influences RNA processing
and the importance of this crosstalk for cell identity and organism function. We also speculate on possible mechanisms underlying the
integration of transcriptional and post-transcriptional processes.
Figure 11: A multi-hop synthetic (Question, Context, Answer) sample.