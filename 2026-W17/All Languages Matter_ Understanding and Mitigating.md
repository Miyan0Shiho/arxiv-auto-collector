# All Languages Matter: Understanding and Mitigating Language Bias in Multilingual RAG

**Authors**: Dan Wang, Guozhao Mo, Yafei Shi, Cheng Zhang, Bo Zheng, Boxi Cao, Xuanang Chen, Yaojie Lu, Hongyu Lin, Ben He, Xianpei Han, Le Sun

**Published**: 2026-04-22 05:33:06

**PDF URL**: [https://arxiv.org/pdf/2604.20199v1](https://arxiv.org/pdf/2604.20199v1)

## Abstract
Multilingual Retrieval-Augmented Generation (mRAG) leverages cross-lingual evidence to ground Large Language Models (LLMs) in global knowledge. However, we show that current mRAG systems suffer from a language bias during reranking, systematically favoring English and the query's native language. By introducing an estimated oracle evidence analysis, we quantify a substantial performance gap between existing rerankers and the achievable upper bound. Further analysis reveals a critical distributional mismatch: while optimal predictions require evidence scattered across multiple languages, current systems systematically suppress such ``answer-critical'' documents, thereby limiting downstream generation performance. To bridge this gap, we propose \textit{\textbf{L}anguage-\textbf{A}gnostic \textbf{U}tility-driven \textbf{R}eranker \textbf{A}lignment (LAURA)}, which aligns multilingual evidence ranking with downstream generative utility. Experiments across diverse languages and generation models show that LAURA effectively mitigates language bias and consistently improves mRAG performance.

## Full Text


<!-- PDF content starts -->

All Languages Matter: Understanding and Mitigating
Language Bias in Multilingual RAG
Dan Wang1,2,∗, Guozhao Mo1,2, ∗, Y afei Shi3, Cheng Zhang3, Bo Zheng3, Boxi Cao1,†,
Xuanang Chen1,†,Y aojie Lu1,Hongyu Lin1,Ben He1,2,Xianpei Han1,2,Le Sun1,2
1Chinese Information Processing Laboratory, Institute of Software, Chinese Academy of Sciences
2University of Chinese Academy of Sciences
3MYbank, AntGroup
{wangdan2023,moguozhao2024,caoboxi,chenxuanang}@iscas.ac.cn
{shiyafei.syf,zc481262,guangyuan}@mybank.cn benhe@ucas.ac.cn
{luyaojie,hongyu,sunle,xianpei}@iscas.ac.cn
Abstract
Multilingual Retrieval-Augmented Generation
(mRAG) leverages cross-lingual evidence to
ground Large Language Models (LLMs) in
global knowledge. However, we show that
current mRAG systems suffer from a lan-
guage bias during reranking, systematically
favoring English and the query’s native lan-
guage. By introducing an estimated oracle ev-
idence analysis, we quantify a substantial per-
formance gap between existing rerankers and
the achievable upper bound. Further analy-
sis reveals a critical distributional mismatch:
while optimal predictions require evidence
scattered across multiple languages, current
systems systematically suppress such “answer-
critical” documents, thereby limiting down-
stream generation performance. T o bridge this
gap, we propose Language- Agnostic Utility-
driven Reranker Alignment (LAURA) , which
aligns multilingual evidence ranking with
downstream generative utility . Experiments
across diverse languages and generation mod-
els show that LAURA effectively mitigates lan-
guage bias and consistently improves mRAG
performance.
1 Introduction
Retrieval-Augmented Generation (RAG), which
incorporates external documentary evidence into
the generation process, has emerged as a core
technique for improving the factual consistency,
knowledge coverage, and controllability of large
language models (LLMs) ( Lewis et al. ,2020 ;
Ram et al. ,2023 ). In these cases, multilingual
RAG (mRAG) has become a critical technology
to address the needs of a global user base for
LLMs ( Asai et al. ,2021b ;Li et al. ,2024 ). In
∗These authors contributed equally .
†Corresponding authors.
Q: 《第五元素》中饰演蓝色女士的是谁？
Q: Who plays the blue lady in The Fifth Element?
Retrieve
Rerank
Evidence Doc
Top-50 Retrieved Documents
A: 米拉·乔沃维奇
A: Milla Jovovich 
Top-5 Reranked DocumentsCorpus
Wasted EvidenceFigure 1: Illustration of failures induced by reranker lan-
guage bias.
real-world settings, knowledge is not uniformly dis-
tributed across languages. Instead, it exhibits in-
herently cross-lingual and complementary struc-
tures. Many region-specific facts, cultural con-
texts, policy details, and technical knowledge are
systematically documented only in particular lan-
guages. Therefore, an effective multilingual RAG
system should go beyond merely supporting mul-
tilingual input and output. Its objective should be
to select and integrate documents across various
languages, thereby providing the generation model
with an evidence set that maximizes informational
value.
Despite this ideal objective, prior studies have
reported the existence of bias in current mRAG sys-
tems ( Park and Lee ,2025 ;Amiraz et al. ,2025 ;Qi
et al. ,2025 ). Motivated by these observations, we
present a systematic analysis of language bias in
mRAG. Crucially, departing from previous stud-
ies that primarily focus on characterizing the pres-
ence of bias, we move beyond mere description
to investigate the underlying causes of such biases
and their significant impact on downstream predic-arXiv:2604.20199v1  [cs.CL]  22 Apr 2026

tions.
Specifically, based on MKQA dataset, we per-
form a comprehensive evaluation across multiple
rerankers and 13 languages. We first construct
multilingual candidate document pools and apply
standard multilingual retrieval and reranking pro-
cedures, after which we analyze the language com-
position of top-ranked documents. Our analysis
reveals a consistent pattern: current mRAG sys-
tems exhibit a pronounced language preference
bias during the reranking stage, systematically fa-
voring English and the original query language.
For instance, when using the widely adopted BGE
reranker, more than 70% of the top-5 retrieved doc-
uments, averaged across 13 languages, originate
from English and the query language alone. Such
a pronounced bias motivated us to dive into its root
causes and practical consequences.
Conceptually, such language preference bias
may stem from two distinct factors. First, it is pos-
sible that more accurate or richer information is in-
herently concentrated in certain languages for spe-
cific queries. Second, the bias may arise from the
limited multilingual capability of reranking mod-
els, which struggle to accurately identify relevant
evidence expressed in other languages. Disentan-
gling these two factors is essential for diagnosing
the core limitations of current mRAG systems. T o
this end, we propose a novel multilingual evidence
estimation method that approximates the oracle
distribution of evidence required to achieve opti-
mal downstream predictions, independent of the
reranker’s language preferences.
By comparing estimated oracle evidence dis-
tributions, we find that existing multilingual
rerankers exhibit limited cross-lingual capability
and often fail to provide suﬀiciently reliable evi-
dence for LLM generation. On the MKQA bench-
mark, standard rerankers underperform the ora-
cle by nearly 20%, revealing a large performance
gap. Further analysis indicates that this gap is
not caused by language concentration: oracle ev-
idence is distributed across multiple languages
rather than dominated by any single one. Al-
though high-quality evidence already exists in di-
verse languages within the candidate set, it is
systematically downweighted by language-biased
rerankers, which substantially limits downstream
performance.
T o address this misalignment, we propose
Language- Agnostic Utility-driven Reranker
Alignment (LAURA) , a training framework thatmitigates language bias in multilingual reranking
by aligning evidence selection with downstream
generation quality . Rather than relying solely
on semantic relevance signals, which often favor
the query language or high-resource languages,
LAURA derives supervision from multilingual
documents that lead to better generation outcomes
in practice. It then trains the reranker to prioritize
answer-critical evidence regardless of language.
This utility-driven alignment reduces systematic
language preferences in evidence selection and
yields consistent improvements in generation
performance.
Our major contributions are summarized as fol-
lows:
•We systematically investigate and quantify
language bias in mRAG. We further intro-
duce an estimated oracle evidence analysis
framework, revealing that such bias substan-
tially constrains the generation performance
of mRAG systems.
•We propose LAURA, an answer-utility-
driven reranking framework that leverages
generation outcomes as supervision signals.
LAURA effectively mitigates language bias
while consistently improving downstream
task performance.
2 Reranking Bias in mRAG Systems
While prior work has identified performance degra-
dation in mRAG, it largely focuses on pipeline-
level optimizations, such as translation-based
strategies, without rigorously quantifying the sys-
tem’s theoretical upper bound or identifying the
underlying causes. A key unresolved question is
whether current bottlenecks arise from insuﬀicient
relevant information in the retrieval pool or from
the selection mechanism ’s inability to identify ac-
curate multilingual evidence. T o bridge this gap,
we present a systematic analysis comparing stan-
dard retrieval pipelines against an oracle evidence
estimating setting, aiming to reveal the misalign-
ment between relevance-based selection and actual
answer utility .
2.1 Language Distribution Analysis
T o quantify the limitations of current mRAG
pipelines, we define two contrasting settings and
a method for analyzing language distribution.

Who plays the blue lady in The Fifth Element? Milla 
Jovovich  (×)
Final Answer: Maïwenn
EN docs
ZH docs
DE docsRerank
Multilingual Corpus
(13 languages)Retrieve
Top-50 Retrieved Documents
Group by 
languageMilla 
Jovovich  (×)
Rerank
Maï wenn (√)
Rerank
(wrong info)
(wrong info)
(contain evidence doc)(Evidence Doc)Figure 2: Illustration of the oracle evidence estimation strategy, where candidate documnents are grouped by lan-
guage and reranked independently to select the top-5 documents within each language group, and multilingual
evidence documents are selected based on correctness of the generated answer.
V anilla Document Reranking. Following the
standard multilingual RAG setup adopted in previ-
ous work ( Chirkova et al. ,2024 ), for each query
q∈ Q , we retrieve documents from a uni-
fied multilingual corpus that contains documents
from all evaluation languages (13 languages in to-
tal). The pipeline consists of two stages: first,
a multilingual retriever BGE-M3 ( Chen et al. )
fetches the top-50 candidate passages across all lan-
guages; second, a multilingual reranker, such as
BGE-Reranker- V2-M3 ( Chen et al. ) and Qwen3-
Reranker-0.6B ( Zhang et al. ,2025 ), selects the top-
5 most relevant passages. These passages are con-
catenated to form the context for the generator. The
quality of the generated answers is evaluated using
the metrics defined below .
Oracle Evidence Estimating. As show in Fig-
ure 2, to estimate the performance upper bound
given the retrieved candidates, we adopt a
language-wise reranking strategy . For a query
q∈ Q , the pool of 50 retrieved candidates is
partitioned by document language. Within each
language group, we select the top-5 documents
(or fewer if insuﬀicient candidates exist) to gen-
erate a language-specific answer. The final per-
formance for query qis defined as the maximum
score achieved across all language groups, serving
as an estimated upper limit for language selection.
We use BGE-M3 embeddings for retrieval and the
BGE-Reranker- V2-M3 for reranking.
Language Distribution Computation. T o un-
derstand the linguistic composition of selected ev-
idence, we calculate distribution metrics for both
settings:
•V anilla Distribution. For each query, we
calculate the proportion of each language
within the final top-5 documents chosen bythe reranker (e.g., three English and two Chi-
nese documents yield a distribution of 0.6 and
0.4, respectively). These per-query distribu-
tions are then averaged over all queries in a
specific query language to obtain the overall
context language distribution.
•Oracle Distribution. For each query, we
identify the document language(s) that pro-
duce the best-performing answer. We as-
sign an importance weight to languages based
on answer performance: if a single language
achieves the best score, it receives a weight of
1; if multiple languages tie for the best, the
weight is uniformly distributed among them
(e.g., a tie between English and Chinese re-
sults in 0.5 for each). Similar to the vanilla
setting, these per-query weights are averaged
across all queries for each query language.
2.2 Experimental Setups
Datasets. For the multilingual document corpus,
we use English Wikipedia1and Wikipedia in the
corresponding user languages2. Following the pre-
processing strategy of ( Chirkova et al. ,2024 ), we
split each Wikipedia article into chunks of 100
words. For languages without explicit whites-
pace segmentation, namely Chinese, Japanese, and
Thai, we instead split articles into chunks of 100
Unicode characters. The article title is prepended
to each chunk.
For multilingual question answering, we use the
MKQA ( Longpre et al. ,2021 ) dataset, following
the setup of Chirkova et al. (2024 ). MKQA is a
multilingual open-domain QA benchmark consist-
ing of 10,000 questions from the Natural Questions
1https://huggingface.co/datasets/facebook/
kilt_wikipedia
2https://huggingface.co/datasets/wikimedia/
wikipedia

ardeenesfifritjakoptruthzh
Query LanguageardeenesfifritjakoptruthzhDoc Language(a) BGE-Reranker-V2-M3
ardeenesfifritjakoptruthzh
Query Language(b) Qwen3-Reanker-0.6B
ardeenesfifritjakoptruthzh
Query Language(c) Oracle
0.00.10.20.30.40.5
ProportionFigure 3: Heatmaps showing the proportion of selected document languages (y-axis) for each query language (x-
axis). (a): Distribution from the BGE-Reranker- V2-M3 reranker. (b): Distribution from the Qwen3-Reranker-0.6B
reranker. (c): The oracle evidence distribution derived from our estimation strategy . Results for other reranking
models are detailed in Appendix E.
(NQ) dataset ( Kwiatkowski et al. ,2019 ), translated
into 25 languages. In our experiments, we focus on
a subset of languages used for evaluation. Specifi-
cally, we select 2.7K samples that overlap between
MKQA and the KIL T NQ dataset3, enabling access
to corresponding document-level relevance infor-
mation for the selected test languages.
Models. For retrieval, we use BGE-M3 ( Chen
et al. ), a strong and publicly available multilin-
gual embedding model capable of encoding all lan-
guages considered in our experiments.
For reranking, we adopt BGE-Reranker- V2-M3
and Qwen3-Reranker-0.6B ( Zhang et al. ,2025 )
as representatives of mainstream encoder-only
rerankers and LLM-based rerankers, respectively .
For answer generation, we evaluate two multi-
lingual large language models, including Qwen2.5-
7B-Instruct ( Qwen et al. ,2025 ), and Llama-3.1-
8B-Instruct ( Grattafiori et al. ,2024 ).
Evaluation Metric. Following Chirkova et al.
(2024 ), we evaluate model outputs using the
character-level 3-gram recall metric. The details
are shown in Appendix A.
2.3 Analysis Results
2.3.1 Multilingual Rerankers Exhibit
Systemic Language Bias
Conclusion 1. Current multilingual RAG sys-
tems exhibit a pronounced language preference
bias during the reranking stage, systematically fa-
voring English and the original query language.
3https://huggingface.co/datasets/facebook/
kilt_tasksT o understand the linguistic preferences of cur-
rent mRAG systems, we analyze the language dis-
tribution of the documents selected for genera-
tion. As illustrated in Figure 3, the heatmaps
display two dominant patterns: a strong diagonal
alignment reflecting a bias toward the query lan-
guage, and a pronounced horizontal alignment in-
dicating a systemic preference for English. T ak-
ing BGE-Reranker as a example, around 60% of
candidate documents are concentrated in English
and the query language. This distribution con-
firms that current rerankers heavily prioritize doc-
uments based on surface-level language matching
or dominant language priors (predominantly En-
glish), rather than assessing semantic relevance eq-
uitably across all candidate languages.
2.3.2 Reranking Bias as a Primary
Performance Bottleneck
Conclusion 2. These reranking biases consti-
tute a primary performance bottleneck in multilin-
gual RAG by causing the model to overlook gen-
uinely relevant evidence within the candidate pool,
thereby hindering the retrieval of optimal informa-
tion.
T o determine whether this pronounced language
bias stems from an intrinsic concentration of high-
quality information in dominant languages or a fun-
damental lack of multilingual capability in current
rerankers, we conduct a decoupled analysis by con-
trasting the standard pipeline with an Oracle Evi-
dence Estimating setting. This comparison allows
us to isolate the model’s selection bias from the
quality of the candidate pool, thereby identifying
the core defect in the current evidence selection

mechanism.
First, to quantify the extent to which reranking
limits system performance, we evaluated the gen-
eration quality under both settings and computed
the correlation between reranking scores and an-
swer utility . As shown in T able 1, simply select-
ing the correct documents from the existing re-
trieval pool yields substantial improvements rang-
ing from +12.9 to +20 points. This result con-
firms that the retrieval stage successfully recalls
the necessary information, but the reranker fails
to surface it. Furthermore, quantitative analysis
reveals a weak correlation between reranker rele-
vance scores and downstream answer quality, with
Pearson coeﬀicients consistently below 0.2 across
all models (T able 2). These indicate that cur-
rent multilingual rerankers fail to provide suﬀi-
ciently accurate and effective evidence, creating
a bottleneck that strictly limits the generation
potential of LLMs.
Next, to understand why valid evidence is over-
looked, we analyzed the language distribution of
estimated oracle evidence and conducted a case
study to observe model behavior. By analyzing
the language distribution under the Oracle Evi-
dence Estimating setting (Figure 3, c), we find that
true answer-critical evidence is broadly distributed
across diverse, non-query languages, rather than
being concentrated in the query language. How-
ever, the systemic bias identified in the previous
section filters these optimal documents out. This
phenomenon is exemplified in the Case Study (T a-
ble 11): for the query ”Who plays the blue lady in
The Fifth Element?”, the reranker prioritizes non-
informative query-language documents (ranks 1-
5) leading to hallucination, while suppressing the
decisive multilingual evidence to rank 10. Thus,
while genuinely relevant evidence is already
present within candidate documents across di-
verse languages, it is consistently marginalized
by the systemic language preferences of cur-
rent rerankers, thereby significantly constrain-
ing downstream performance.
3 Language-Agnostic Utility-driven
Reranker Alignment
In this section, we aim to mitigate language bias
in multilingual rerankers. Such bias leads models
to disproportionately favor documents in English
or the query language, even when higher quality
evidence exists in other languages. We hypothe-LangLlama-8B-Instruct Qwen2.5-7B-Instruct
BGE Qwen3 Oracle BGE Qwen3 Oracle
ar 32.7 28.9 53.6 33.8 31.4 51.4
de 62.8 60.9 76.5 59.6 58.0 73.7
en 70.1 67.8 79.3 65.4 63.3 76.2
es 63.0 62.7 76.8 62.3 61.3 75.6
fi 58.1 54.2 73.4 55.5 52.8 71.5
fr 64.4 63.7 76.7 56.6 54.4 71.6
it 63.9 62.2 77.0 60.2 57.9 73.8
ja 29.2 28.2 47.9 28.0 27.2 44.9
ko 25.5 23.3 41.0 26.5 24.7 38.9
pt 66.4 66.3 78.4 60.5 59.3 73.8
ru 51.9 47.4 68.0 45.7 42.3 63.1
th 26.4 24.8 44.1 23.5 22.3 39.0
zh 21.7 21.9 33.8 29.0 28.7 42.8
A VG 48.9 47.1 63.6 46.7 44.9 61.3
T able 1: Performance comparison (Recall@3-gram)
of vanilla reranking and oracle evidence estimating.
‘BGE’ and ‘Qwen3’ refer to BGE-Reranker- V2-M3 and
Qwen3-Reranker-0.6B models, respectively . ‘Oracle’
denotes the performance achieved under the estimated
oracle evidence.
Reranker Model Pearson p-value
BGE-RerankerLlama3-8B-Instruct 0.188 3.8×10−290
Qwen2.5-7B-Instruct 0.198 1.0×10−320
Qwen-RerankerLlama3-8B-Instruct 0.129 1.2×10−135
Qwen2.5-7B-Instruct 0.127 2.3×10−135
T able 2: Correlation between relevance scores (mean
top-5) and downstream answer performance (Recall@3-
gram) under different rerankers and generators.
size that skewed training data, in which high qual-
ity query and document annotations are scarce for
low resource languages, is a key factor driving this
disparity .
T o address this issue, we propose a language ag-
nostic utility driven reranker alignment framework
(LAURA). This framework reduces language bias
by grounding reranker supervision in answer utility
instead of relying on language dependent relevance
signals. Instead of defining positives based on lexi-
cal overlap or language matching, LAURA selects
documents according to their contribution to down-
stream answer quality, thereby reducing reliance
on language specific surface features. Specifi-
cally, LAURA uses a two stage data construction
pipeline (Figure 4) to generate language agnostic
supervision signals, followed by listwise reranker
fine tuning. This design promotes balanced cross
lingual supervision and aligns reranker preferences
with answer correctness, thereby mitigating the
over preference for high resource languages.
3.1 Answer Utility-driven Data Construction
Although many RAG QA datasets are publicly
available, most only provide annotations for the

correctness of the final answer, without explicit
query-document relevance labels. This absence
leads to language bias in reranker training. We aim
to automatically generate such annotations while
maintaining balanced multilingual coverage.
Given a query q, we retrieve a candidate docu-
ment set D from a multilingual corpus. Our objec-
tive is to select a positive subset D pos⊂D consist-
ing of documents that genuinely support answering
the query, free from language-specific bias. We
use the average answer quality produced by mul-
tiple generators conditioned on a document as a
proxy for its answer utility .
Stage 1: Language-Debiased Subset Selection.
Directly estimating answer utility on top-ranked
retrieved documents can amplify the inherent lan-
guage bias of multilingual rerankers, which of-
ten favor documents in high-resource or query-
matched languages. T o mitigate this effect, we pro-
pose a candidate debiasing stage that filters the can-
didate set before utility estimation while preserv-
ing overall utility .
Given a retrieved document set D, we partition
documents into disjoint subsets according to their
language and apply the same reranker to rank doc-
uments within each subset independently . From
each subset, we retain up to five top-ranked doc-
uments as utility candidates. This procedure does
not assume language-specific relevance. Instead, it
enforces equal exposure across linguistic subsets,
preventing the candidate pool from being domi-
nated by documents favored due to language priors
rather than informational content.
The retained documents are then evaluated by
multiple generators to estimate their average an-
swer quality . Documents achieving the highest
generation utility are selected for subsequent super-
vision construction. In cases where multiple sub-
sets yield identical maximal utility (e.g., all gener-
ators produce correct answers), all corresponding
documents are preserved. The resulting candidate
set is denoted as D balanced .
By decoupling candidate selection from global
reranker scores and restricting comparisons to
within each subset, this stage reduces language-
induced ranking bias while retaining documents
that are useful for downstream answer generation.
Stage 2: Document-Level Utility Estimation.
While Stage 1 ensures cross-lingual coverage, doc-
uments in D balanced may still vary in their actual
usefulness. We therefore perform fine-graineddocument-level utility estimation by evaluating
each document independently via generation.
T o avoid introducing an implicit language bias
through relative ranking alone, we apply an ab-
solute utility threshold θand retain only docu-
ments whose average generation performance ex-
ceeds this threshold. The final positive set D pos
thus consists of documents that demonstrably con-
tribute to answer correctness, independent of lan-
guage.
Overall, this two-stage procedure yields high-
quality, language-debiased training data that
grounds reranker supervision in answer utility
rather than language preference.
3.2 Listwise Reranker Fine- T uning
Using the constructed training data, we fine-tune
the reranker with a listwise learning objective.
Given a query qand a candidate set D, docu-
ments not selected into D pos are treated as nega-
tives, forming D neg .
During training, we construct training instances
consisting of one positive document and kneg-
ative documents, i.e., (q, d pos,{d(i)
neg}k
i=1), where
dpos∈D pos andd(i)
neg∈D neg . The reranker pro-
duces a relevance score s(q, d)for each document
d∈ D q. Training encourages the model to assign
the highest score to the positive document within
the list. We adopt a softmax cross-entropy loss:
L=−s(q, d pos) + log∑
d∈D qexp(
s(q, d))
(1)
For encoder-only rerankers, s(q, d)is produced
directly as a scalar logit. For LLM-based rerankers,
the score is derived from the relative logits of pre-
defined positive_token and negative_token ,
which represent the model ’s preference over rele-
vance labels.
3.3 Experimental Setups
LAURA Dataset. W e use data from the MKQA
benchmark, selecting only samples that are disjoint
from the evaluation test set to avoid data leakage.
In stage 1, for each question, we retrieve the
top-100 candidate documents from a multilingual
Wikipedia corpus using the BGE-M3 retriever.
Within each language group, we apply the multilin-
gual BGE reranker to select the top-5 documents,
yielding a language-debiased candidate set.
During evaluation, we prompt multiple genera-
tion models to answer the question conditioned on

Rerank
Candidate 
DocumentsGenerateEval Group
1.00
1.00
0.85
Stage 1: Language-Balanced Subset Selection
Selected
Subset
1.00
EN.1
0.75
EN.2
0.80
ZH.1
Stage 2: Document-Level Utility EstimationMAX 
Postive
DocumentsSplitGenerateEval
Threshold
Figure 4: T wo-stage data construction pipeline in the LAURA framework.
each document independently and measure answer
quality using character-level 3-gram recall. T o re-
duce model-specific bias, we compute each docu-
ment’s utility score as the average generation per-
formance across a diverse set of four generation
models, including Qwen2.5-7B-Instruct, Qwen2.5-
14B-Instruct, Llama3-8B-Instruct, and DeepSeek-
R1-Distill-Qwen-7B ( DeepSeek-AI et al. ,2025 ).
The threshold θis set to 0.8, ensuring that the doc-
uments retain high utility .
Finally, we construct a total of 18,360 query–
positive documents pairs. Among them, 1,000 are
randomly sampled as the dev set. Detailed statis-
tics of the constructed fine-tuning dataset are re-
ported in Appendix B.
Evaluation Metric. T o evaluate the effective-
ness of LAURA, we adopt Precision@k and
NDCG@k to assess the rerank performance on
positive documents in the dev set. In addition,
we use the PEER ( Y ang et al. ,2024 ) metric to
measure whether the reranker exhibits language-
specific bias. PEER is based on the assump-
tion that documents with equal relevance should
have similar average rankings across different lan-
guages. Higher PEER scores indicate weaker lan-
guage preference. The detailed definitions of the
evaluation metrics are provided in Appendix A.
Training Details. We fine-tune BGE-Reranker-
V2-M3 using the implementation provided by
FlagEmbedding4, and Qwen3-Reranker-0.6B us-
ing SWIFT ( Zhao et al. ,2025 ). For each query,
BGE is trained with 1 negative document, whereas
Qwen uses 7 negative documents, reflecting the
stronger capacity of the LLM-based reranker to
handle larger candidate lists. Both models are
optimized with AdamW ( Loshchilov and Hutter ,
2019 ), using a learning rate of 6×10−6, and are
trained for five epochs.
4https://github.com/FlagOpen/FlagEmbeddingSettingPrecision NDCG
PEER@5 @10 @5 @10
BGE-Reranker 0.3400 0.2712 0.4666 0.4904 0.5941
+ LAURA 0.3830 0.3149 0.5531 0.5925 0.6627
Qwen-Reranker 0.2702 0.2206 0.3695 0.3921 0.6606
+ LAURA 0.3546 0.2847 0.5214 0.5496 0.6720
T able 3: Reranking results of BGE-Reranker- V2-M3
(BGE-Reranker) and Qwen3-Reranker-0.6B (Qwen-
Reranker) on the dev set before and after LAURA train-
ing. PEER measures language bias in the reranker, with
higher values indicating weaker language preference.
3.4 Results of LAURA
LAURA improves multilingual rerankers’ abil-
ity to identify relevant documents. T o evaluate
whether rerankers can better identify positive can-
didates under the LAURA, we assess Precision,
NDCG on the dev set before and after training.
These metrics directly reflect the rerankers’ ability
to rank relevant candidates higher. As shown in T a-
ble 3, both BGE and Qwen rerankers exhibit con-
sistent improvements after being trained within the
LAURA. In particular, Precision@5 increases by
approximately 6 points, while NDCG@5 improves
by around 13 points across both model families, in-
dicating a stronger capability to place positive can-
didates at higher ranks.
LAURA improves multilingual rerankers’ lan-
guage fairness. We observe that LAURA leads
to consistent improvements in language fairness.
Beyond the quantitative gains on the dev set mea-
sured by the PEER metric, we analyze the language
distribution of reranker outputs on the MKQA test
set after LAURA training. As shown in T able 4,
the JS divergence and KL divergence between the
post-training distribution and the estimated ora-
cle evidence distribution are substantially reduced,
demonstrating that the learned distribution moves
closer to the desired target distribution. More-
over, we observe a consistent decrease in the pro-
portion of documents written in English and the

Setting JS KL Entropy
BGE-Reranker 0.203 0.186 2.03
+ LAURA 0.090 0.041 2.27
Qwen-Reranker 0.141 0.122 2.13
+ LAURA 0.129 0.094 2.14
T able 4: Language distributional metrics before and af-
ter LAURA training on the MKQA test set. JS and KL
denote the average distances between vanilla distribu-
tion and the estimated oracle distribution. Entropy indi-
cates the average entropy of the vanilla distribution of
each query language.
query language, suggesting that LAURA mitigates
the over-preference for dominant languages and en-
courages a more balanced multilingual ranking be-
havior. This indicates that LAURA effectively re-
duces the original language skew of rerankers. In
terms of PEER, LAURA yields an about +7 points
for the BGE reranker and about +0.5 points for the
Qwen reranker, suggesting that the method system-
atically mitigates language biases and promotes
more equitable performance across languages.
LAURA improves downstream generation per-
formance and ranking utility. LAURA is de-
signed to enhance reranking quality and improve
the alignment between reranking scores and down-
stream generation. T o investigate to what ex-
tent the improved reranking capability learned un-
der the LAURA transfers to downstream genera-
tion performance, we conduct experiments on the
MKQA test set using the setup in Section 2.1 .
The results are reported in T able 5. On the 3-
gram recall metric, incorporating LAURA leads
to an average improvement of 1.95 points for the
Qwen reranker and 1.0 points for the BGE reranker.
These results indicate that improving the rerankers’
ability to select higher-quality candidates can trans-
late into better downstream generate quality .
T o quantitatively assess the change in the rela-
tionship between ranking quality and generation
performance, we compute the Pearson correlation
between the average reranking score of the top-5
documents and the corresponding 3-gram recall
scores. After training with LAURA, the Pearson
correlation increases by approximately 25% for the
BGE reranker and by about 108% for the Qwen
reranker. This demonstrates that LAURA substan-
tially strengthens the correlation between rerank-
ing scores and generation performance, thereby im-
proving the practical utility of the reranking scores
for downstream generation.SettingLlama Qwen
3-gram Pearson 3-gram Pearson
BGE-Reranker 48.9 0.198 46.7 0.188
+ LAURA 49.9 0.236 47.7 0.247
Qwen-Reranker 47.1 0.129 44.9 0.127
+ LAURA 49.2 0.269 46.7 0.264
T able 5: Generation performance and Pearson corre-
lation of rerankers before and after LAURA training
on the MKQA test set. Pearson correlations are com-
puted between the average reranker scores of the top-5
reranked documents and character 3-gram recall perfor-
mance. All Pearson correlations are statistically signif-
icant with p-values < 0.001.
SettingLlama Qwen
3-gram Pearson 3-gram Pearson
BGE-Reranker 48.9 0.198 46.7 0.188
Self- Training 48.9 0.188 46.7 0.202
mMARCO 48.7 0.132 46.3 0.137
LAURA 49.9 0.236 47.7 0.247
T able 6: Performance comparison of LAURA against al-
ternative fine-tuning strategies, including Self- Training
(naive supervision using top-5 retrieved candidates) and
mMARCO fine-tuning (general-purpose multilingual
ranking data).
3.5 Comparison Against Fine-tuning
Baselines
We provide additional analysis on two alternative
fine-tuning strategies to further validate the effec-
tiveness of LAURA ’s data construction pipeline.
Self- Training Baseline. The first baseline fine-
tunes the reranker solely on its own top-ranked out-
puts as pseudo-positive supervision, directly treat-
ing the top-5 re-ranked documents as relevant and
all remaining candidates as non-relevant, without
any additional filtering or refinement. This setting
corresponds to the starting point of LAURA ’s data
construction pipeline and serves as an empty con-
trol to verify whether LAURA ’s additional filter-
ing and refinement steps contribute beyond naive
supervision. Under this paradigm, the model’s ex-
isting ranking preferences may be progressively re-
inforced, as no mechanism is introduced to correct
noisy or biased pseudo labels.
Fine-tuning on mMARCO. The second base-
line fine-tunes the reranker using mMARCO ( Boni-
facio et al. ,2022 ), a widely-used multilingual
dataset, to examine whether general-purpose train-
ing data can address the specific distribution imbal-
ance in mRAG. We randomly sample 20k queries

from mMARCO for training, comparable to the
17,360 queries used by LAURA, ensuring a fair
comparison in terms of training scale.
For both baselines, we use the same hyperparam-
eters as in our main experiments. In addition, we
ensure that LAURA and the Self- Training baseline
are trained on the exact same set of queries, iso-
lating the effect of the data construction strategy
rather than the training queries themselves.
As shown in T able 6, LAURA consistently
outperforms both baselines across all settings.
The Self- Training baseline fails to surpass BGE-
Reranker on certain metrics, indicating that naive
pseudo-label supervision can reinforce existing bi-
ases rather than correct them. The mMARCO base-
line also leads to a slight performance drop com-
pared to BGE-Reranker, suggesting that general
relevance signals cannot resolve the specific distri-
bution imbalance in mRAG. These results collec-
tively demonstrate that LAURA ’s filtering and re-
finement steps are essential for effective reranker
adaptation in mRAG settings.
4 Related Work
mRAG is pivotal for bridging global informa-
tion gaps and ensuring equitable knowledge ac-
cess across linguistic barriers. T o advance this ca-
pability, the community has established a robust
foundation spanning diverse benchmarks ( Asai
et al. ,2021a ,c;Liu et al. ,2025 ) and retrieval ar-
chitectures ( Gao et al. ,2022 ;Zhang et al. ,2023 ;
Chirkova et al. ,2024 ).
Previous studies have conducted preliminary
analyses of language preference phenomena in
mRAG systems. For instance, Amiraz et al. (2025 )
investigates multilingual retrieval biases over Ara-
bic–English corpora. Park and Lee (2025 ) eval-
uate language bias in multilingual RAG by mea-
suring retrieval ranking shifts. In comparison, our
work moves beyond merely characterizing these
preferences to systematically quantify the substan-
tial performance gap resulting from this linguistic
misalignment.
T o mitigate these biases, prior research has
largely relied on translation-centric strategies, such
as mapping queries or documents to a shared pivot
language ( Moon et al. ,2025 ;Amiraz et al. ,2025 ;
Park and Lee ,2025 ). However, these pipeline-
level heuristics depend heavily on the capability of
translation models and do not fundamentally cor-
rect the ranking objective. In comparison, we pro-pose to align rerankers directly with generation util-
ity, training the model to prioritize answer-critical
evidence regardless of the source language.
5 Conclusion
This work analyzes language bias in multilingual
retrieval-augmented generation (mRAG) systems,
showing that conventional rerankers favor English
and the query ’s original language, suppressing crit-
ical multilingual evidence. Using estimated oracle
evidence, we reveal the resulting performance gap
and cross-lingual distribution of answer-relevant
documents. T o address this, we propose LAURA,
a language-agnostic utility-driven reranker that
aligns evidence ranking with downstream genera-
tion, mitigating bias and improving performance
across languages and models.
Limitations
This work focuses on analyzing the alignment be-
tween reranker relevance and downstream answer
quality in multilingual RAG systems. Accordingly,
our study is limited to the reranking stage and does
not consider modifications to the retriever or the
generator, whose interactions with reranking re-
main an important direction for future work.
In addition, our evaluation relies on automatic,
task-specific metrics that may not fully capture
all aspects of generation utility, such as factual
completeness or cross-lingual reasoning. Finally,
while our experiments cover diverse multilingual
settings, the generalizability of our findings to
other architectures, domains, and low-resource lan-
guages warrants further investigation.
Acknowledgments
We sincerely thank the reviewers for their insight-
ful comments and valuable suggestions. This work
was supported by Beijing Natural Science Foun-
dation (L243006), the Natural Science Foundation
of China (No. 62536008, 62506354), the Post-
doctoral Fellowship Program of CPSF under Grant
Number GZC20251041, and MYbank, AntGroup.
References
Chen Amiraz, Y aroslav Fyodorov , Elad Haramaty, Zo-
har Karnin, and Liane Lewin-Eytan. 2025. The
cross-lingual cost: Retrieval biases in RAG over
Arabic-English corpora . In Proceedings of The

Third Arabic Natural Language Processing Confer-
ence , pages 69–83, Suzhou, China. Association for
Computational Linguistics.
Akari Asai, Jungo Kasai, Jonathan Clark, Kenton Lee,
Eunsol Choi, and Hannaneh Hajishirzi. 2021a. XOR
QA: Cross-lingual open-retrieval question answer-
ing . In Proceedings of the 2021 Conference of the
North American Chapter of the Association for Com-
putational Linguistics: Human Language T echnolo-
gies , pages 547–564, Online. Association for Com-
putational Linguistics.
Akari Asai, Xinyan Y u, Jungo Kasai, and Hanna Ha-
jishirzi. 2021b. One question answering model for
many languages with cross-lingual dense passage re-
trieval. Advances in Neural Information Processing
Systems , 34:7547–7560.
Akari Asai, Xinyan Y u, Jungo Kasai, and Hannaneh Ha-
jishirzi. 2021c. One question answering model for
many languages with cross-lingual dense passage re-
trieval. In Proceedings of the 35th International Con-
ference on Neural Information Processing Systems ,
NIPS ’21, Red Hook, NY , USA. Curran Associates
Inc.
Luiz Bonifacio, Vitor Jeronymo, Hugo Queiroz
Abonizio, Israel Campiotti, Marzieh Fadaee,
Roberto Lotufo, and Rodrigo Nogueira. 2022.
mmarco: A multilingual version of the ms marco pas-
sage ranking dataset .Preprint , arXiv:2108.13897.
Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo,
Defu Lian, and Zheng Liu. Bge m3-embedding:
Multi-lingual, multi-functionality, multi-granularity
text embeddings through self-knowledge distillation.
Nadezhda Chirkova, David Rau, Hervé Déjean,
Thibault Formal, Stéphane Clinchant, and V assilina
Nikoulina. 2024. Retrieval-augmented generation
in multilingual settings . In Proceedings of the
1st Workshop on T owards Knowledgeable Lan-
guage Models (KnowLLM 2024) , pages 177–188,
Bangkok, Thailand. Association for Computational
Linguistics.
DeepSeek-AI, Daya Guo, Dejian Y ang, Haowei Zhang,
Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao
Zhu, Shirong Ma, Peiyi W ang, Xiao Bi, Xiaokang
Zhang, Xingkai Y u, Y u Wu, Z. F. Wu, and others.
2025. Deepseek-r1: Incentivizing reasoning capa-
bility in llms via reinforcement learning .Preprint ,
arXiv:2501.12948.
Yifan Gao, Qingyu Yin, Zheng Li, Rui Meng, T ong
Zhao, Bing Yin, Irwin King, and Michael Lyu. 2022.
Retrieval-augmented multilingual keyphrase gener-
ation with retriever-generator iterative training . In
Findings of the Association for Computational Lin-
guistics: NAACL 2022 , pages 1233–1246, Seattle,
United States. Association for Computational Lin-
guistics.Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schel-
ten, Alex V aughan, Amy Y ang, Angela Fan, and 1
others. 2024. The llama 3 herd of models .Preprint ,
arXiv:2407.21783.
T om Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Al-
berti, Danielle Epstein, Illia Polosukhin, Jacob De-
vlin, Kenton Lee, Kristina T outanova, Llion Jones,
Matthew Kelcey, Ming-W ei Chang, Andrew M. Dai,
Jakob Uszkoreit, Quoc Le, and Slav Petrov . 2019.
Natural questions: A benchmark for question answer-
ing research . T ransactions of the Association for
Computational Linguistics , page 453 –466.
Patrick Lewis, Ethan Perez, Aleksandara Piktus,
Filippo Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih,
Tim Rocktäschel, Sebastian Riedel, and Douwe
Kiela. 2020. Retrieval-augmented generation for
knowledge-intensive nlp tasks. arXiv: Computation
and Language,arXiv: Computation and Language .
Bryan Li, Samar Haider, Fiona Luo, Adwait Agashe,
and Chris Callison-Burch. 2024. BordIRlines: A
dataset for evaluating cross-lingual retrieval aug-
mented generation . In Proceedings of the First Work-
shop on Advancing Natural Language Processing for
Wikipedia , pages 1–13, Miami, Florida, USA. Asso-
ciation for Computational Linguistics.
W ei Liu, Sony Trenous, Leonardo F. R. Ribeiro, Bill
Byrne, and Felix Hieber. 2025. XRAG: Cross-
lingual retrieval-augmented generation . In Findings
of the Association for Computational Linguistics:
EMNLP 2025 , pages 15669–15690, Suzhou, China.
Association for Computational Linguistics.
Shayne Longpre, Yi Lu, and Joachim Daiber. 2021.
Mkqa: A linguistically diverse benchmark for mul-
tilingual open domain question answering .T ransac-
tions of the Association for Computational Linguis-
tics , page 1389 –1406.
Ilya Loshchilov and Frank Hutter. 2019. Decoupled
weight decay regularization . In International Con-
ference on Learning Representations .
Hoyeon Moon, Byeolhee Kim, and Nikhil V erma.
2025. Quality-aware translation tagging in multilin-
gual RAG system . In Proceedings of the 5th Work-
shop on Multilingual Representation Learning (MRL
2025) , pages 161–177, Suzhuo, China. Association
for Computational Linguistics.
Jeonghyun Park and Hwanhee Lee. 2025. Investigat-
ing language preference of multilingual RAG sys-
tems . In Findings of the Association for Compu-
tational Linguistics: ACL 2025 , pages 5647–5675,
Vienna, Austria. Association for Computational Lin-
guistics.

Jirui Qi, Raquel Fernández, and Arianna Bisazza. 2025.
On the consistency of multilingual context utiliza-
tion in retrieval-augmented generation . In Proceed-
ings of the 5th Workshop on Multilingual Representa-
tion Learning (MRL 2025) , pages 199–225, Suzhuo,
China. Association for Computational Linguistics.
Qwen, :, An Y ang, Baosong Y ang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Y u, Chengyuan
Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan
Lin, Jian Y ang, Jianhong T u, Jianwei Zhang, Jianxin
Y ang, Jiaxi Y ang, Jingren Zhou, and 25 oth-
ers. 2025. Qwen2.5 technical report . Preprint ,
arXiv:2412.15115.
Ori Ram, Y oav Levine, Itay Dalmedigos, Dor Muhlgay,
Amnon Shashua, Kevin Leyton-Brown, and Y oav
Shoham. 2023. In-context retrieval-augmented lan-
guage models .T ransactions of the Association for
Computational Linguistics , 11:1316–1331.
Eugene Y ang, Thomas Jänich, James Mayfield, and
Dawn Lawrie. 2024. Language fairness in multilin-
gual information retrieval . In Proceedings of the
47th International ACM SIGIR Conference on Re-
search and Development in Information Retrieval ,
SIGIR ’24, page 2487 –2491, New Y ork, NY , USA.
Association for Computing Machinery .
Xinyu Zhang, Kelechi Ogueji, Xueguang Ma, and
Jimmy Lin. 2023. T oward best practices for training
multilingual dense retrieval models .ACM T rans. Inf.
Syst. , 42(2).
Y anzhao Zhang, Mingxin Li, Dingkun Long, Xin
Zhang, Huan Lin, Baosong Y ang, Pengjun Xie,
An Y ang, Dayiheng Liu, Junyang Lin, Fei Huang,
and Jingren Zhou. 2025. Qwen3 embedding: Ad-
vancing text embedding and reranking through foun-
dation models. arXiv preprint arXiv:2506.05176 .
Y uze Zhao, Jintao Huang, Jinghan Hu, Xingjun
W ang, Y unlin Mao, Daoze Zhang, Zeyinzi Jiang,
Zhikai Wu, Baole Ai, Ang W ang, Wenmeng Zhou,
and Yingda Chen. 2025. Swift: A scalable
lightweight infrastructure for fine-tuning .Proceed-
ings of the AAAI Conference on Artificial Intelli-
gence , 39(28):29733–29735.

A Metric Implementation Details
We report some evaluation metrics in our exper-
iments: character 3-gram Recall, Precision@k,
NDCG@k, and PEER. Below, we describe their
implementations in detail.
Character 3-gram Recall. Character 3-gram
Recall measures the lexical coverage between the
generated content and the reference text at the char-
acter level. We extract all contiguous character 3-
grams from both the reference text and the gener-
ated text. Let Cref denote the multiset of character
3-grams from the reference, and Cgen denote those
from the generated text. The character 3-gram Re-
call score is defined as:
Recall char-3 =|Cgen∩Cref|
|Cref|(2)
This metric is robust to tokenization differences
and is particularly suitable for multilingual evalu-
ation.
Precision@k. Precision@k measures the pro-
portion of relevant documents among the top- k
reranked results. Formally, given a ranked list of
documents Rkof length kand a binary relevance
function rel (·), Precision@k is defined as:
Precision@ k=1
kk∑
i=1rel(Ri) (3)
where rel (Ri) = 1 if the document at rank iis rel-
evant, and 0otherwise.
NDCG@k. Normalized Discounted Cumulative
Gain (NDCG@k) takes into both the relevance and
the ranking position of documents. We first com-
pute DCG@k as:
DCG@ k=k∑
i=12rel(Ri)−1
log2(i+ 1)(4)
where rel (Ri)denotes the relevance score of
the document at rank i. NDCG@k is obtained
by normalizing DCG@k with the ideal DCG@k
(IDCG@k), which corresponds to the optimal
ranking:
NDCG@ k=DCG@ k
IDCG@ k(5)
This normalization ensures that NDCG@k ranges
between 0and1.
17.1% en
13.2% es
12.7% it
12.4% de12.0% fr11.6% pt8.8% fi3.5% ru2.4% zh1.8% th1.7% ja1.4% ko1.3% arFigure 5: Language distribution of queries (inner ring)
and positive documents (outer ring).
PEER. W e compute PEER (Probability of Equal
Expected Rank) following Y ang et al. (2024 ), with
a task-specific adaptation: we only use positive
documents in the fairness test. Intuitively, PEER
evaluates whether relevant documents written in
different languages receive systematically different
ranks.
For each query q, we collect all retrieved doc-
uments labeled as positive and record their rank
positions in the final ranked list. We then parti-
tion these ranks by the document language ℓ∈ L ,
yielding groups {Rq,ℓ}ℓ∈L , where Rq,ℓ is the mul-
tiset of rank positions of positive documents in lan-
guage ℓ.
W e apply the Kruskal–W allis H test (KW) on
these rank groups, with the null hypothesis that the
rank distributions of positive documents are iden-
tical across languages (i.e., equal expected ranks).
We define PEER for query qas the resulting p-
value:
PEER (q) =p(
KW({Rq,ℓ}ℓ∈L))
(6)
where higher values (closer to 1) indicate that we
cannot reject the hypothesis of equal expected rank,
suggesting better language fairness. We report the
final PEER score as the mean of PEER (q)over all
queries.
B Statistics of the LAURA Dataset
As shown in T able 7, we report the number of
queries and positive documents in the constructed

Train Set Dev Set
Queries 17,360 1,000
Positive Documents 114,867 6,762
A vg. Languages per Query 2.90 2.90
T able 7: Statistics of the LAURA dataset. A vg. Lan-
guages per Query indicates the average number of
distinct languages among the positive candidate docu-
ments associated with each query .
Language Baseline LAURA Δ% p
Portuguese 63.1 65.9 +4.4 1.24e-24***
Finnish 55.1 57.8 +4.7 1.15e-17***
French 59.8 62.3 +4.3 1.26e-19***
Spanish 62.3 64.8 +4.0 2.94e-20***
Italian 61.1 63.5 +4.1 1.70e-18***
German 60.3 62.1 +2.9 1.46e-10***
Arabic 31.7 32.9 +3.8 5.22e-04***
English 66.6 67.8 +1.7 8.68e-06***
Russian 46.8 47.5 +1.5 4.62e-02*
Thai 24.2 24.9 +2.9 1.20e-02*
Japanese 28.1 28.6 +1.8 1.02e-01
Chinese 25.3 25.8 +1.8 8.04e-02
Korean 25.0 24.8 -0.8 4.98e-01
Overall 46.9 48.4 +3.1 8.62e-74***
T able 8: Per-language paired t-test results on 3-gram
scores.
LAURA dataset, as well as the average number of
languages per query among the candidate positive
documents. Figure 5illustrates the language distri-
butions of queries and positive documents.
C Detailed MKQA test Results
T o facilitate comparison, T able 10 reports detailed
per-language results on the MKQA test set used in
the main experiments.
D Case Study
T able 11 presents a case study illustrating the limi-
tations of relevance-based reranking in the vanilla
multilingual RAG setting.
E Language Distribution and Model
Performance
Figure 6and Figure 7show the document lan-
guage distribution and generation performance un-
der the vanilla and upper-limit settings, respec-
tively, across different rerankers.SettingLlama Qwen
3-gram Pearson 3-gram Pearson
BGE-Reranker 48.9 0.198 46.7 0.188
Stage 1 48.7 0.272 46.5 0.269
LAURA 49.9 0.236 47.7 0.247
T able 9: Ablation study comparing the full pipeline
against using only Stage 1 training data.
F Statistical Significance of LAURA
Improvements
T o assess whether the RAG performance gains
from LAURA are statistically reliable, we compute
per-query 3-gram scores and construct paired sam-
ples between LAURA and the corresponding base-
line under identical configurations (2 rerankers ×
2 generators), yielding approximately 4,000 paired
observations per language. We apply two-tailed
paired t-tests on the per-query score differences.
Results are reported in T able 8. LAURA achieves
statistically significant gains ( p < 0.05) in 10 out
of 13 languages, and the overall improvement is
highly significant ( p= 8.62×10−74). All conclu-
sions hold under Bonferroni correction, confirm-
ing that the improvements reflect a systematic ef-
fect rather than sampling variation.
G Ablation of LAURA
T o further validate the necessity of both Stage 1
and Stage 2 in our pipeline, we conduct an ablation
study using the training data obtained from Stage
1 alone.
As shown in T able 9, training solely with the
data from Stage 1 improves the correlation coef-
ficient. However, due to the lack of filtering, a
substantial number of false-positive documents are
included, making it diﬀicult to achieve meaning-
ful improvements in downstream performance. In
contrast, Stage 2 performs document-level evalu-
ation and filtering, which substantially improves
data quality and consequently leads to further gains
in the final generation performance.

Setting ar de en es fi fr it ja ko pt ru th zh A vg.
Llama3-8B-Instruct
Upper-limit 53.6 76.5 79.3 76.8 73.4 76.7 77.0 47.9 41.0 78.4 68.0 44.1 33.8 63.6
BGE-Reranker 32.7 62.8 70.1 63.0 58.1 64.4 63.9 29.2 25.5 66.4 51.9 26.4 21.7 48.9
+ LAURA 33.2 64.8 69.8 65.7 59.9 66.7 65.5 28.9 24.9 69.1 50.8 26.9 22.2 49.9
Qwen-Reranker 28.9 60.9 67.8 62.7 54.2 63.7 62.2 28.2 23.3 66.3 47.4 24.8 21.9 47.1
+ LAURA 31.6 63.3 70.1 65.5 57.3 66.1 65.0 28.2 23.8 68.4 50.5 26.7 22.7 49.2
Qwen2.5-7B-Instruct
Upper-limit 51.4 73.7 76.2 75.6 71.5 71.6 73.8 44.9 38.9 73.8 63.1 39.0 42.8 61.3
BGE-Reranker 33.8 59.6 65.4 62.3 55.5 56.6 60.2 28.0 26.5 60.5 45.7 23.5 29.0 46.7
+ LAURA 33.9 61.0 66.0 64.2 58.4 58.8 61.8 29.4 25.7 63.4 44.8 23.4 29.5 47.7
Qwen-Reranker 31.4 58.0 63.3 61.3 52.8 54.4 57.9 27.2 24.7 59.3 42.3 22.3 28.7 44.9
+ LAURA 33.1 59.2 65.2 63.9 55.3 57.8 61.9 28.1 25.0 62.6 44.1 22.7 28.7 46.7
T able 10: Performance comparison between the vanilla document reranking and the oracle evidence estimation set-
tings on MKQA. All results are reported using character 3-gram recall. Bolded results denote the best performance
among all non –upper-limit settings.
ardeenesfifritjakoptruthzh
Query LanguageardeenesfifritjakoptruthzhDoc Language0.000.250.500.751.00Performance0.330.600.66 0.640.57 0.580.61
0.290.250.63
0.46
0.240.29(a) BGE-Gemma
ardeenesfifritjakoptruthzh
Query Language0.310.600.67 0.64
0.55 0.58 0.60
0.28 0.250.62
0.45
0.210.30(b) BGE-Minicpm
ardeenesfifritjakoptruthzh
Query Language0.310.580.63 0.61
0.53 0.540.58
0.27 0.250.59
0.42
0.220.29(c) Qwen3-Reranker-0.6B
0.00.10.20.30.40.5
Proportion
Figure 6: V anilla document reranking with BGE-emma, BGE-Minicpm and Qwen3-Reranker-0.6B rerankers. The
heatmap shows the language distribution, while the bar chart reports Recall@3-gram of Qwen2.5-7B-Instruct.
ardeenesfifritjakoptruthzh
Query LanguageardeenesfifritjakoptruthzhDoc Language0.000.250.500.751.00Performance0.520.74 0.76 0.760.71 0.72 0.74
0.450.390.74
0.63
0.390.43(a) BGE-Gemma
ardeenesfifritjakoptruthzh
Query Language0.520.740.77 0.750.72 0.72 0.74
0.450.380.74
0.63
0.380.44(b) BGE-Minicpm
ardeenesfifritjakoptruthzh
Query Language0.520.73 0.76 0.760.71 0.71 0.74
0.450.390.74
0.62
0.380.43(c) Qwen3-Reranker-0.6B
0.00.10.20.30.40.5
Proportion
Figure 7: Oracle evidence estimation with BGE-Gemma, BGE-Minicpm and Qwen3-Reranker-0.6B rerankers. The
heatmap shows the language distribution, while the bar chart reports Recall@3-gram of Qwen2.5-7B-Instruct.

Query Who plays the blue lady in The Fifth Element?
Label Maïwenn
V anilla T op-5 [1] (es) El quinto elemento. El quinto elemento (en francés: Le Cinquième Élément) es una película
francesa (con coproducción de EE.UU.) de ciencia ficción y acción de 1997 dirigida por Luc Besson, con
Bruce Willis, Milla Jovovich y Gary Oldman en los papeles principales. Principalmente ambientada
en el , la trama central de la película involucra la supervivencia del planeta Tierra, que se convierte en
responsabilidad de Korben Dallas (Willis)...
[2] (en) The Fifth Element. The Fifth Element The Fifth Element () is a 1997 French science fiction action
film directed and co-written by Luc Besson. It stars Bruce Willis, Gary Oldman and Milla Jovovich.
Primarily set in the 23rd century , the film’s central plot involves the survival of planet Earth, which
becomes the responsibility of Korben Dallas (Willis), a taxicab driver and former special forces major,
after a young woman (Jovovich) falls into his cab ...
[3] (it) Il quinto elemento. Il quinto elemento (Le Cinquième Élément) è un film del 1997 diretto da Luc
Besson. Di produzione francese (benché girato in lingua inglese), fu la pellicola più costosa mai prodotta
in Europa allépoca della sua uscita. Il film, che ha per protagonisti Bruce Willis, Milla Jovovich e
Gary Oldman , venne presentato fuori concorso al 50º Festival di Cannes...
[4] (ru) Пятый элемент (фильм). того, для создания костюмов был привлечён известный модельер
Жан-Поль Готье. Он разработал все 900 костюмов, использованных в сценах на корабле «Флостон
Парадайз». Костюм Лилу из белых полос ткани Готье создал, вдохновившись картиной Фриды Кало
«Сломанная колонна». В течение года команда создала более 8000 рисунков. В это время Бессон
предложил на главную роль Брюса Уиллиса и Мела Гибсона, а также рассматривал Джулию Робертс
на роль Лилу...
[5] (zh)第五元素 (電影 ).米拉·乔沃维奇饰）的人形女性。莉露对周围的一切深感恐惧，逃出实验
室后，她从楼层的外沿跳了下去，正好掉进前特种部队少校科本·达拉斯（布鲁斯·威利斯饰）所
开的出租车裡 ...
Model Answer Milla Jovovich plays the blue lady, Leeloo, in The Fifth Element. Wrong
Oracle T op-5 [1] (de) Das fünfte Element. Das fünfte Element (Originaltitel: Le Cinquième Élément) ist ein Science-
Fiction-Film von Luc Besson mit Bruce Willis und Milla Jovovich aus dem Jahr 1997. Das fünfte El-
ement ist aufgrund seiner hohen Einspielergebnisse von über 260 Millionen US-Dollar einer der bisher
kommerziell erfolgreichsten europäischen Filme. Handlung Der Film beginnt im Jahr 1914 in Ägypten, in
einem verfallenen T empel, wo der Archäologe Professor Pacoli, begleitet vom Reporter Billy und einem
Priester, Inschriften über das unfassbar Böse findet...
[2*] (de) (rank 10 in baseline) Das fünfte Element. der Antagonist Zorg begegnen sich im Film kein
einziges Mal. Die Kostüme und Accessoires wurden von dem französischen Modeschöpfer Jean Paul
Gaultier entworfen. Als sich der Archäologe zu Beginn des Films plötzlich riesigen Mondoshawan-Aliens
gegenübersieht, fragt er in der deutschen Fassung: „Sind Sie hier von der Erde? “, während es im Original
heißt: „Are you German? “(dt. „Sind Sie Deutsche(r)? “). Der erste T eil der Arie der Diva ist aus der Oper
Lucia di Lammermoor von Gaetano Donizetti und wird hier von Inva Mula gesungen. Als Darstellerin
der Diva agierte jedoch Maïwenn , mit der Regisseur Besson zum Zeitpunkt der Dreharbeiten zusam-
menlebte und...
[3] (de) Milla Jovovich. Milica „Milla “Jovovich (* 17. Dezember 1975 in Kiew, Ukrainische SSR,
Sowjetunion, ukrainisch Милиця Богданівна Йовович ) ist eine US-amerikanische Schauspielerin und
Model serbisch-russischer Herkunft. Bekannt wurde sie nach Erfolgen in den Filmen Das fünfte Element
und Johanna von Orleans, aber besonders für ihre Hauptrolle in der Filmreihe Resident Evil...
[4] (de) Das fünfte Element. ins Weltall geschossen wird. In einer abschließenden Szene will sich der
Präsident bei den beiden „Helden “bedanken, die sich aber leidenschaftlich lieben und deshalb unabkömm-
lich sind. Auszeichnungen Der Film wurde im Jahr 1998 für den Oscar in der Kategorie Bester T onschnitt
nominiert. Er wurde 1998 in den Kategorien Bester Science-Fiction-Film, Beste Spezialeffekte, Beste
Kostüme und Beste Nebendarstellerin (Milla Jovovich) für den Saturn A ward nominiert...
[5] (de) Das fünfte Element. den Kategorien Bester Film, Beste Kostüme, Bester Schnitt, Beste Filmmusik
und Bester T on für den gleichen Preis nominiert. Milla Jovovich wurde 1998 für den Blockbuster Enter-
tainment A ward und (für die Beste Kampfszene) den MTV Movie A ward nominiert. Der Film gewann
1997 die Goldene Leinwand, den Bogey A ward in Silber und wurde für den Europäischen Filmpreis no-
miniert...
Model Answer The role of the blue alien diva Plavalaguna was played by Maïwenn. True
T able 11: A case study revealing a limitation of relevance-based reranking in multilingual RAG. The answer-critical
document (marked as [2*] ) is retrieved but ranked only 10th under the baseline, causing relevance-based reranking
to produce an incorrect answer.