# Healthier LLMs: Retrieval-Augmented Generation for Public Health Question Answering

**Authors**: Felix Feldman, Joshua Harris, Timothy Laurence, Leo Loman, Ollie Higgins, Fan Grayson, Poonam Soma, Bethany Pace-Bonello, Michael Borowitz, Toby Nonnenmacher

**Published**: 2026-07-07 14:47:42

**PDF URL**: [https://arxiv.org/pdf/2607.06641v1](https://arxiv.org/pdf/2607.06641v1)

## Abstract
Large language models (LLMs) achieve promising results on medical question answering benchmarks, yet their use in public health is constrained by hallucinations and the rapid evolution of official guidance. Retrieval-Augmented Generation (RAG) mitigates these risks by grounding responses in an explicitly maintained corpus, but end-to-end performance depends critically on retrieval configuration and on evaluation beyond multiple-choice formats. We extend PubHealthBench, a question answering (QA) benchmark of 7,929 questions derived from UK Government public health guidance, into a retrieval-augmented setting and systematically evaluate retrieval and generation choices. We compare dense, sparse, and hybrid retrieval across multiple embedding models and corpus variants, and show that hybrid retrieval consistently improves recall and ranking quality, with chunk length and topic interacting with ranking performance. Providing retrieved context substantially increases multiple-choice accuracy across a diverse set of LLMs, enabling smaller open-weight models to match or outperform larger models used without retrieval, with gains primarily driven by retrieval quality and careful context selection. To assess realistic free-form answering, we introduce a rubric-based LLM-as-a-judge covering faithfulness, completeness, clarity, and factual consistency, and validate it against dual human annotations. Judge-human agreement is strongest for faithfulness and completeness, while factual consistency and clarity are less reliably reproduced, motivating caution when interpreting those dimensions at scale. Overall, our results highlight retrieval as a primary lever for reliable public health QA and provide practical guidance for building and evaluating RAG systems grounded in official guidance.

## Full Text


<!-- PDF content starts -->

Healthier LLMs: Retrieval-Augmented Generation for
Public Health Question Answering
Felix Feldman Joshua Harris Timothy Laurence Leo Loman Ollie Higgins
Fan Grayson Poonam Soma Bethany Pace-Bonello Michael Borowitz
Toby Nonnenmacher
Abstract
Large language models (LLMs) achieve promising results on medical question
answering benchmarks, yet their use in public health is constrained by hallucina-
tions and the rapid evolution of official guidance. Retrieval-Augmented Generation
(RAG) mitigates these risks by grounding responses in an explicitly maintained
corpus, but end-to-end performance depends critically on retrieval configuration
and on evaluation beyond multiple-choice formats. We extend PubHealthBench, a
question answering (QA) benchmark of 7,929 questions derived from UK Govern-
ment public health guidance, into a retrieval-augmented setting and systematically
evaluate retrieval and generation choices. We compare dense, sparse, and hybrid
retrieval across multiple embedding models and corpus variants, and show that
hybrid retrieval consistently improves recall and ranking quality, with chunk length
and topic interacting with ranking performance. Providing retrieved context sub-
stantially increases multiple-choice accuracy across a diverse set of LLMs, enabling
smaller open-weight models to match or outperform larger models used without re-
trieval, with gains primarily driven by retrieval quality and careful context selection.
To assess realistic free-form answering, we introduce a rubric-based LLM-as-a-
judge covering faithfulness, completeness, clarity, and factual consistency, and
validate it against dual human annotations. Judge–human agreement is strongest
for faithfulness and completeness, while factual consistency and clarity are less
reliably reproduced, motivating caution when interpreting those dimensions at
scale. Overall, our results highlight retrieval as a primary lever for reliable public
health QA and provide practical guidance for building and evaluating RAG systems
grounded in official guidance.
1 Introduction
Artificial intelligence (AI) is playing an expanding role in public health, from chatbots that answer
health queries [ 1] to decision-support tools for public health professionals [ 2]. Large language
models (LLMs) can already generate coherent answers based on extensive training data, in some
cases approaching expert performance on medical question answering tasks [ 3]. This potential has
spurred interest in deploying LLMs in public health, both as public-facing tools and as decision aids
for public health professionals. Unlike many clinical decision-support settings—where tools are
designed to support decisions for an individual patient at the point of care, public health guidance is
population-level, often precautionary, and closely tied to official recommendations that are updated
as evidence and policy evolve [ 4,5,6]. However, LLMs can hallucinate information or provide
outdated advice, which in public health can have severe implications: even small inaccuracies may
harm individual health decision-making [ 7] and, at scale, drive inappropriate or dangerous behaviours
39th Conference on Neural Information Processing Systems (NeurIPS 2025).
arXiv:2607.06641v1  [cs.CL]  7 Jul 2026

across populations [ 8]. Therefore, ensuring that LLM responses are reliable and up to date is a
prerequisite for safe AI adoption in public health.
One established approach to improving LLM performance and reliability isRetrieval-Augmented
Generation(RAG) [ 9]. In RAG systems, an LLM is coupled with an external retrieval component
that selects relevant documents from a knowledge base; the model then conditions its output on this
retrieved context rather than relying solely on its parametric memory [ 10,11]. By incorporating
retrieval, an LLM can expand and update its effective knowledge beyond what is stored in its frozen
parameters, mitigating hallucinations and reducing the impact of outdated training data [ 12]. In
high-stakes fields such as public health, the ability to provide answers grounded in trusted evidence
is especially valuable. RAG systems have been shown to substantially improve performance on
knowledge-intensive tasks such as medical and guideline-based question answering compared with
LLMs that rely only on internal knowledge [13, 14, 15].
With the increasing adoption of LLM systems in use cases involving public health or medical
queries, it is crucial to measure their effectiveness on tasks that reflect real information needs of
both the public and professionals [ 16,17,18]. To this end, a variety of benchmarks for medical
Question Answering (QA) have been developed, though most existing benchmarks focus on clinical or
biomedical questions rather than public health guidance. To understand this issue for UK Government
public health information Harris et al. evaluated LLMs on a range of public health classification and
extraction tasks [ 19] before introducingPubHealthBench[ 1], a QA benchmark with nearly 8,000
questions derived from official UK public health guidance.
In this follow-up work, we explore RAG as a solution to improve LLM performance on public
health QA. We extend our benchmark setup by allowing models to retrieve information from the
same underlying corpus of UK public health guidance from which the questions were generated [ 1].
Specifically, we use PubHealthBench to address four questions. (1) How effective are different
retrieval setups, spanning embedding models and system design choices, at identifying the most
relevant chunks of public health guidance given questions about that guidance? (2) Does introducing
retrieval of trusted documents improve LLM performance on multiple-choice QA (MCQA), and how
does this depend on retrieval quality and context configuration? (3) To what extent do the observed
MCQA improvements generalise to free-from responses? (4) Can we develop an automated evaluation
framework to assess LLM responses that aligns with human expert decisions? By answering these
questions, we aim to demonstrate how coupling LLMs with a corpus of public health guidance can
yield more reliable and informed responses on knowledge-intensive tasks.
2 Related Work
2.1 Retrieval-Augmented Generation Foundations
RAG combines external knowledge retrieval with large language model (LLM) generation to enhance
factual accuracy and mitigate hallucinations [ 9,13]. For example, the RETRO model demonstrated
that pairing a smaller generator with a large retrieval data store can match much larger LLM-only
systems [ 20], and subsequent work has shown that retrieval scale and quality are critical performance
levers [ 21]. Recent surveys provide up-to-date taxonomies of RAG architectures [ 22,10,11] and
examine trust, safety, fairness, and accountability in retrieval-augmented systems [ 12]. Together,
these works highlight retrieval as a distinct driver of performance and underscore the challenge of
reliably integrating retrieved knowledge, particularly in safety-critical domains such as medicine and
public health.
2.2 Retrieval Methods and Context Handling
Modern retrieval systems largely rely on dense semantic embeddings to map text into vector space
representations, with trade-offs between model size, retrieval accuracy, and latency [ 23]. Hybrid
retrieval methods that combine dense embeddings with sparse models such as BM25 often boost
retrieval precision and recall [ 24]. Re-ranking models (e.g., GTE-ModernColBERT) and knowledge-
graph-based retrieval further support retrieving relevant information in domain-specific settings
[25,26,27]. Importantly, long context windows change but do not remove the need for careful
retrieval and context selection [ 28,29]. There is also extensive empirical work evaluating retrieval
design choices, such as, query expansion, chunking, memory-based architectures; underlining how
2

there are several configuration choices that can impact retrieval, and downstream QA, performance
[30, 31].
2.3 Benchmarking and Evaluation in Health QA
In medical and public-health domains, multiple-choice QA benchmarks (e.g., MedMCQA, MedQA)
have provided standard evaluation frameworks for assessing clinical knowledge [ 32,33,16]. However,
multiple-choice formats can overestimate model competence compared with free-form generation
tasks and may mask important reasoning failures [ 34]. For example, Singh et al. report an average
drop in accuracy of 39.4% when moving from MCQA to free-from answers for LLMs (vs. 22.3%
for humans). They also show that even with the question stem fully masked, MCQA accuracy
remains above chance (6.7% above random on average), suggesting MCQA performance can be
partly driven by answer-option cues rather than underlying understanding [ 34]. Consequently, newer
benchmarks have shifted toward free-form responses to better simulate real-world usage [ 35,36]. In
public health, studies show substantial knowledge gaps and uneven performance across domains:
AfriMed-QA reports that GPT-4o attains 79% overall accuracy on expert MCQs, but performance
varies substantially by specialty—ranging from >90% in top specialties (e.g., rheumatology) to
<60% in weaker areas such as pediatrics and obstetrics–gynecology [ 37], illustrating strong topic
dependence despite high aggregate QA accuracy, and PubHealthBench finds that while the strongest
models achieve >90% accuracy ( 92.5% for GPT-4.5) in MCQA, exceeding the 88% human baseline,
but no model scored higher than 75% in free-form QA[1].
Evaluation methods are also evolving. Overlap metrics such as ROUGE or BLEU correlate poorly
with human judgments in free-form QA [ 38,39]. The use of LLMs as automated evaluators (“LLM-
as-Judge”) has emerged as a scalable alternative, with specialised models such as Prometheus and
its successor showing strong alignment with human assessments across a range of generation tasks
[40,41,42,43]. In health-specific settings, rubric-guided LLM-as-judge frameworks that combine
physician-designed rubrics with model graders have achieved high agreement with clinician ratings
[18,44]. A recent comparative study of search engines, LLMs, and RAG variants on health questions
further demonstrates retrieval’s substantial impact on QA accuracy and highlights the benefits of
grounding answers in retrieved evidence [ 15]. Together, these developments underscore the need for
benchmarks that jointly assess retrieval quality and generation performance in health QA systems,
especially when systems are expected to align with official public health guidance.
3 Methods
3.1 Benchmark and Dataset
We use the PubHealthBench QA benchmark, which contains 7,929 multiple-choice questions derived
from 687 UK Government public health guidance documents covering 10 topic areas [ 1]. The source
documents were converted to markdown and split into chunks by markdown header levels, with full
header hierarchy appended for context. This produced 5,358 chunks, which form the retrieval corpus.
Table 1: PubHealthBench subsets [1].
Subset Size Creation method QA format Purpose
PubHealthBench-Full7,929 LLM generated multiple-choice questions cre-
ated from single chunks of public health guid-
ance via an automated pipeline.MCQA Broad coverage to assess LLM
performance across many public
health topics and guidance
audiences.
PubHealthBench-FreeForm760 Random subset of questions manually re-
viewed by experts (with ambiguous/invalid
items identified), presented without multiple-
choice options.Free-form More realistic open-ended
evaluation of LLM performance on
free-form QA.
3.2 Retrieval
We adopt five retrieval methods for our RAG pipeline:
1.Embedding-based retrieval:Each chunk is encoded into a dense embedding via a text-
embedding model, and queries are encoded with the same model. Retrieval ranks chunks
3

by cosine similarity between query and chunk embeddings. We evaluate eight embedding
models (e.g., NV-Embed-V2 [ 45], EmbeddingGemma [ 46], ModernBertBase [ 47], SFR-
Embedding-Mistral [ 48], Multilingual-E5-large [ 49], OpenAI’s text-embedding-3-large
[50]).
2.Keyword-based retrieval:We build sparse indices using the term frequency algorithms
TF-IDF and BM25. Retrieval ranks chunks by the overlap of key terms between the query
and chunks.
3.Hybrid retrieval:We perform both embedding-based and keyword-based retrieval as in
(1) and (2), and then merge the two ranked lists using a weighted Reciprocal Rank Fusion
(RRF):
R(d;α) =α1
c+r dense(d)+ (1−α)1
c+r sparse (d)(1)
Where αis a weighting factor, we set c= 60 consistent with the value used in foundational
RRF work [51].
4.Summary-based hybrid retrieval:Each chunk is summarised using GPT-4o-mini [ 52].
Hybrid retrieval (as in (3)) is applied to the summary corpus rather than full chunks. This
variant tests whether more compact context improves retrieval precision in the public-health
domain. Summaries are used only for retrieval and the corresponding full chunk is passed
down-stream for generation.
5.Reduced-corpus hybrid retrieval:Hybrid retrieval (as in (3)) is applied to a reduced
corpus, where any chunk longer than 512 tokens (for a given embedding model’s tokenizer)
is replaced with its generated summary (the 512-token cutoff aligns with the context capacity
of Multilingual-E5 [ 49]). As with (4), only the corresponding full chunk is passed down-
stream for generation.
3.3 Generation
The top- kranked chunks from retrieval are appended (with a chunk identifier and separator) and
concatenated into a context block. Following the prompt structure of Harris et al. [ 1], we insert the
question and the context block, with a simple instruction to make use of the inserted context when
answering. We assess 11 LLMs (e.g., Llama 3.3 [ 53], Phi-4 [ 54], Gemma-3 [ 55], MedGemma [ 56],
CommandR [57]) on the benchmark subsets shown in Table 1.
3.4 Experimental Setup
3.4.1 Retrieval Setup
Retrieval is run with each embedding model over all 7,929 queries on three corpora: CF(full
corpus), CS(summary-only corpus), and CR(reduced corpus). For model and each hybrid retrieval
configuration we find the the value of the weighting factor α∈ {0.50,0.55,0.6, ...,0.95} that results
in the best retrieval performance, where higher αmeans heavier weighting for embedding-based
ranking (see Equation 1). We report the optimal retrieval setup (method, corpus,α) for each model.
3.4.2 MCQA Setup
ForPubHealthBench-Full, we extract top- kchunks with k∈ {1,3,5,10} from three representative
retrieval configurations (NV-Embed-v2 ( CR), Multilingual-E5-large-instruct ( CR), ModernBertBase
(CF)). These models are chosen to test the impact of retrieval quality on MCQA accuracy, as they span
a range of model sizes and retrieval performance. We report MCQA accuracy onPubHealthBench-
Full, alongside accuracy without retrieval of the human baseline and GPT-4.5 [1].
3.4.3 Free-form Setup
In a large real world guidance corpus to achieve near perfect ( ∼100% ) retrieval recall, we may require
a large number of retrieved chunks, so for free form responses we use the same three configurations
forPubHealthBench-FreeForm, set k= 15 , and limit the context block to 10,000 tokens. Through
this setup we evaluate LLMs in a more realistic setting, where they are challenged to precisely extract
relevant information from a large amount of context, and use this to clearly and accurately answer a
4

public health query, without guiding answer options. In this setup, we evaluate free-form answers
using an LLM judge (see § 3.5).
3.5 Evaluation
3.5.1 Retrieval Evaluation
Because each query has exactly one target chunk, we report the following metrics:
•Recall@ k: Proportion of queries whose relevant chunk appears within the top k. Reflects
coverage of retrieval but all rankings in the topkare treated equally.
•Mean Reciprocal Rank (MRR): The average of 1/racross queries, where ris the rank of
the relevant chunk. Measures how early the correct chunk appears in the ranking.
•Normalized Discounted Cumulative Gain (nDCG@ k): Assesses ranking quality via
log-discounted gain, penalising relevant chunks appearing lower in the ranking [ 58,59].
This provides a finer-grained view of rank quality.
•Precision@1: Because there is exactly one relevant chunk per query, we report Precision@1
(1 if the chunk is ranked first, else 0) [60].
Together, these metrics allow us to evaluate whether the correct chunk is retrieved at all (Recall@ k),
how early it appears (MRR), and how well the ranking order is structured (nDCG).
3.5.2 Free-form Evaluation
To evaluate the answers forPubHealthBench-FreeForm, we follow HealthBench in applying a
rubric-based LLM-as-a-Judge to evaluate free form responses [ 18]. Through this we aim to create a
framework for automated evaluation of free-form responses to public health queries. In our setup,
each answer is assessed by the GPT-OSS-120B model, using a structured set of four criteria on factual
consistency, completeness, clarity, and faithfulness to official guidance. The criteria are designed to
be non-overlapping and objective, assessing an LLMs ability select only relevant information from
a large amount of context, and use this to accurately and clearly answer a public health query. The
criteria definitions are shown in in Table 2. To support reliable analysis, two human expert reviewers
independently annotated the same sample of 100 LLM responses against each criterion. We report
Cohen’s κand macro-F1, with 95% bootstrap confidence intervals for the agreement between the
reviewers and each reviewer with the LLM judge. All judgments are made with respect to the single
relevant chunk of guidance, along with a reference answer, question and LLM answer, and detailed
instructions for each criteria, aligning with HealthBench’s emphasis on grounded, context-aware
evaluation [18].
Table 2: LLM-as-a-Judge criteria definitions used to evaluate free-form responses.
Criterion Definition
FaithfulnessThe answer must not introduce any new claims, advice, or information not found in the provided official guidance.
Completeness The answer should include all key points and recommendations from the guidance necessary to fully address the question.
Factual ConsistencyThe parts of the answer that relate to the guidance must be factually accurate and reflect the intended meaning.
ClarityThe answer should clearly communicate its main point and be grammatically correct, focused, and easy to follow.
4 Results
4.1 Retrieval
We find that the retrieval performance of embedding models varies considerably on the task of re-
trieving relevant public health information. In dense-only retrieval configuration, the best performing
model,NV-Embed-v2, achieves the best results across all metrics ( 0.98 Recall@10, 0.85 MRR, 0.88
nDCG@10), a marked gain over other models tested; for example,EmbeddingGemmascores 49
ppts lower on Precision@1. Retrieval quality generally improves with model size and embedding
dimension, yet sparse-only retrieval using TF-IDF or BM25 outperforms some embedding models.
For every model, a hybrid setup improves performance across all retrieval metrics; the optimal fusion
weight αin our RRF setup is model-dependent. Notably, hybrid search narrows the gap between
5

smaller encoders (multilingual-E5-large,ModernBERT-base), which see a 5–10ppt increase across
metrics, andNV-Embed-v2, despite a factor of 10difference in parameter count. No model attains
its best results on the summary-only corpus (CS); peak scores arise on either the reduced (CR)or full
(CF)corpora, and this preference is not explained by context length alone.
Table 3:Retrieval metrics for sparse and dense-only retrievalMetrics across embedding models
and sparse baselines. NV-Embed-v2 is the strongest model in a dense-only setup.
Model metadata Retrieval metrics
Embedding model Params Dim Context len. Precision@1 Recall@5 Recall@10 MRR nDCG@10
NV-Embed-v2 8B 4096 327680.76 0.96 0.98 0.85 0.88
SFR-Embedding-Mistral 7B 4096 32768 0.71 0.94 0.97 0.81 0.85
text-embedding-3-large – 3072 8192 0.68 0.92 0.96 0.78 0.83
Multilingual-e5-large 0.6B 1024 512 0.68 0.91 0.96 0.78 0.83
Multilingual-e5-large-instruct 0.6B 1024 512 0.64 0.90 0.95 0.76 0.80
EmbeddingGemma 0.3B 768 2048 0.27 0.55 0.69 0.41 0.47
ModernBERT-base 0.15B 768 8192 0.63 0.90 0.95 0.75 0.79
MedCPT-Query-Encoder 0.1B 768 512 0.24 0.51 0.63 0.37 0.42
TF-IDF – – – 0.61 0.86 0.92 0.72 0.77
BM25 – – – 0.65 0.87 0.91 0.75 0.79
Table 4:Best-performing retrieval configurationsRetrieval metrics for models in hybrid retrieval
setups. NV-Embed-v2 remains the strongest but other models show significant improvements in
hybrid setups.
Retrieval configuration Retrieval metrics
Embedding model Corpus Method Sparse indexαPrecision@1 Recall@5 Recall@10 MRR nDCG@10
NV-Embed-v2C R hybrid TF-IDF 0.750.80 0.97 0.99 0.88 0.91
SFR-Embedding-MistralC F hybrid TF-IDF 0.75 0.76 0.96 0.99 0.85 0.88
text-embedding-3-largeC F hybrid TF-IDF 0.55 0.77 0.96 0.98 0.85 0.88
Multilingual-e5-largeC R hybrid TF-IDF 0.80 0.76 0.95 0.98 0.84 0.87
Multilingual-e5-large-instructC R hybrid TF-IDF 0.75 0.75 0.95 0.98 0.84 0.87
EmbeddingGemmaC R hybrid TF-IDF 0.50 0.63 0.88 0.93 0.74 0.78
ModernBERT-baseC F hybrid TF-IDF 0.55 0.74 0.94 0.98 0.83 0.86
MedCPT-Query-EncoderC F hybrid TF-IDF 0.50 0.64 0.88 0.94 0.74 0.79
Figure 1 shows retrieval performance declines across models as target chunk length increases. Recall
decreases only modestly, but rank-sensitive metrics (Precision@ k, MRR, nDCG@ k) drop sharply
once chunks exceed 700–800words. Using (CR)somewhat attenuates this decline relative to (CF),
although it does not mitigate it completely. As target length grows, the performance gap between
larger models (NV-Embed-v2,SFR-Embedding-Mistral) and smaller models widens.
In Figure 2, we observe topic-level deviations in ranking quality across embedding models, measured
as each topic’s departure from the model’s mean nDCG@10. A one-way ANOV A indicates that
guidance topic explains less than 1% of the variance in nDCG@10 across models ( η2≈0.006 –0.008 ).
Although this global effect size is small, topic means show systematic differences in ranking quality.
The gap between lower-performing topics (e.g.,Blood Safety, Hepatitis, STIs and HIV) and the
highest-performing topic (Tuberculosis, Travel, Zoonotic and Emerging Infections) is approximately
0.06–0.08 nDCG@10. The relative ordering of higher- and lower-performing topics is broadly
consistent across models.
In contrast, Recall@5 shows little evidence of topic dependence ( η2≈0.002 –0.005 ). Topic therefore
has minimal influence on whether relevant content appears within the top- kresults, but it does affect
the position at which relevant chunks are ranked once retrieved.
4.2 MCQA with Retrieval
Giving LLMs access to retrieved context significantly improves their MCQA performance; topic-wise
accuracy per model is shown in Table 5. Other thanGemma-3-1B, all models exceed the human
baseline (with cursory search engine use) of 88% [1]. Several models also meet or surpass the best
performing model without retrieved context (GPT -4.5,92.5% ) [1]. To summarise topic-wise stability,
we report the coefficient of variation CV =σ/µ across topic accuracies. With the exception of
6

Figure 1:Retrieval by target lengthRetrieval metrics for various setups broken down by word count
of the target chunk.
Table 5:MCQA accuracy by guidance topic.Best MCQA results per generation model across
PubHealthBench guidance topic areas.
Generation modelBlood safety
& STIs/HIVChemicals
& toxClimate
& healthGastro
& foodHCAI/AMR
& sepsisInclusion
healthOther RadiationTB/travel
& zoonosesVPDs
& imm.Total
Llama-3.3-70B∗0.995 0.997 1.000 0.994 1.000 0.9980.9921.000 0.992 0.997 0.995
Command-R-32B 0.989 0.988 0.997 0.979 0.992 0.992 0.983 0.974 0.986 0.991 0.987
MedGemma-27B 0.995 0.9951.0000.988 0.992 0.9960.9960.983 0.990 0.993 0.992
Gemma-3-27B 0.992 0.9921.0000.988 0.992 0.996 0.992 0.987 0.990 0.988 0.989
Gemma-2-27B 0.984 0.988 0.997 0.988 0.986 0.990 0.975 0.974 0.984 0.987 0.985
Phi-4-14B 0.986 0.9921.0000.985 0.986 0.992 0.979 0.991 0.991 0.990 0.989
Gemma-3-12B 0.989 0.994 0.997 0.985 0.984 0.993 0.992 0.996 0.985 0.992 0.989
Llama-3.1-8B 0.976 0.980 0.997 0.979 0.978 0.989 0.979 0.979 0.976 0.982 0.979
Phi-4-4B 0.954 0.969 0.987 0.959 0.970 0.980 0.958 0.979 0.975 0.973 0.971
Gemma-3-4B 0.957 0.970 0.980 0.962 0.953 0.965 0.962 0.949 0.958 0.956 0.956
Gemma-3-1B 0.651 0.692 0.647 0.649 0.619 0.679 0.732 0.634 0.658 0.656 0.664
∗Model used to generate the benchmark.
Gemma-3-1B, all models have CV<0.05 , indicating low between -topic variability compared with
LLM-only results [1].
Most models reach maximum MCQA accuracy with k∈ {3,5} context chunks, though all LLMs
exceed the highest score without retrieval even at k= 1 ; onlyLlama-3.3-70Bcontinues to improve
atk= 10 , and the marginal gain is small relative to the additional token cost. At k= 5 , LLMs had
access to the target chunk at least 95% of the time, and the increase in recall from k= 5 tok= 10
comes with additional noise from less relevant chunks. UsingNV-Embed-v2as the retriever raises
accuracy relative to the other embedding models shown. However, performance at k= 1 is strong
despite the retrieved chunk being fully relevant for only≤80%of queries.
The impact of retrieval quality is further shown in Figure 4. Retrieving the correct chunk yields the
largest performance gains, with ranking providing a consistent secondary benefit that is strongest
when longer context sections introduce more noise. Accuracy generally declines as the target chunk
moves down the ranked list, although smaller models (e.g.,Gemma-3-1B,Phi-4-4B,Gemma-3-
4B) exhibit noisier and occasionally counter-intuitive responses, including slight improvements at
lower ranks, whereas larger models (e.g.,Llama-3-70B,MedGemma-27B,Gemma-3-27B) show
smoother, more consistent declines and overall reduced sensitivity to rank.
7

Figure 2:Topic-wise deviations in ranking quality.Bars show the mean difference (percentage
points) between each guidance topic area’s nDCG@10 and the corresponding model’s overall mean
nDCG@10 (dashed line at0)
8

Figure 3:MCQA accuracy by number of context chunksThe MCQA scores of generation models
by the number of context chunks used in the prompt. Scores are shown for the 3 different retrieval
runs used. *Model used to generate benchmark
4.2.1 LLM-as-Judge Validation
Table 6: Cohen’s κand macro-F1 agreement between human reviewers and the LLM judge (GPT-
OSS-120B) with 95% bootstrap CIs.
Reviewer 1 vs Reviewer 2 Judge vs Reviewer 1 Judge vs Reviewer 2
Criterionκmacro-F1κmacro-F1κmacro-F1
Faithfulness 0.60 [0.43, 0.75] 0.80 [0.71, 0.88] 0.64 [0.46, 0.78] 0.82 [0.73, 0.89] 0.71 [0.54, 0.83] 0.85 [0.77, 0.92]
Completeness 0.76 [0.58, 0.89] 0.88 [0.79, 0.94] 0.59 [0.40, 0.73] 0.79 [0.69, 0.87] 0.57 [0.38, 0.72] 0.78 [0.68, 0.86]
Factual consistency 0.46 [0.27, 0.65] 0.73 [0.63, 0.82] 0.06 [-0.13, 0.27] 0.53 [0.44, 0.64] 0.17 [-0.03, 0.37] 0.59 [0.48, 0.69]
Clarity 0.35 [0.17, 0.51] 0.66 [0.56, 0.75] 0.26 [0.07, 0.44] 0.62 [0.52, 0.72] 0.61 [0.45, 0.76] 0.81 [0.72, 0.88]
Overall 0.56 [0.48, 0.64] 0.78 [0.74, 0.82] 0.42 [0.33, 0.51] 0.71 [0.66, 0.76] 0.55 [0.47, 0.63] 0.78 [0.73, 0.82]
We validated the LLM-as-Judge rubric on a sample of 100 responses independently annotated by two
human reviewers (Table 6). The validation reveals substantial variation in how consistently different
quality dimensions can be assessed, even by human raters, underscoring the difficulty of evaluating
free-form responses to public health questions.
Completeness was the most reliably assessed criterion, with high inter-reviewer agreement ( κ≈0.76 ;
macro-F1 ≈0.88 ) and strong judge–reviewer alignment ( κ≈0.57 –0.59). Faithfulness was also
reasonably consistent across both human raters ( κ≈0.60 ; macro-F1 ≈0.80 ) and between the
judge and reviewers ( κ≈0.6 –0.7). For both criteria, judge performance falls within the range of
human disagreement, and at the aggregate level, judge agreement with Reviewer 2 is essentially
indistinguishable from inter-reviewer agreement (overall κ≈0.55 vs.0.56; macro-F1 ≈0.78 in
both cases).
However, clarity and factual consistency proved much harder to assess reliably. Both criteria exhibited
lower inter-reviewer agreement, indicating that humans themselves struggle to apply these dimensions
consistently to public health responses. For factual consistency in particular, judge–human agreement
was weak ( κnear zero and substantially below inter-reviewer κ), meaning that automated scoring
for this criterion does not currently reproduce human judgments. Factual consistency would be
a critical dimension for public health applications, yet we find it the hardest to evaluate reliably
for both humans and the LLM judge. We therefore focus subsequent analysis on faithfulness and
completeness, where both human agreement and judge alignment are sufficient to support robust
9

(a)
(b)
Figure 4:Effects of target chunk retrieval and rank on MCQA accuracy(a) The differences in
MCQA accuracy when target chunk retrieved vs not retrieved and retrieved at rank r= 1 vsr≥2 ,
fork∈ {3,5,10} . (b) The difference in MCQA accuracy when target chunk is retrieved at rank r, or
not retrieved, compared tor=k, fork∈ {3,5,10}.
conclusions, while recognising that developing reliable evaluation methods for factual consistency
remains an important open challenge.
4.2.2 Free-form Response Evaluations
Figure 5 shows judge pass rates for free-form answers across models. Medium and large models
achieve consistently high completeness and clarity (typically ∼0.88–0.97), indicating that they
identify the key guidance needed to answer the question and present it in a legible, interpretable
way even without guiding answer options. The smallest model (Gemma-3-1B) is a clear outlier,
with substantially lower completeness and clarity. In contrast, faithfulness is the dominant failure
mode across all models: even among larger models, only around half to two-thirds of responses
meet the faithfulness criterion (e.g., 0.64 for the best-performing model), indicating a broad tendency
to incorporate guidance from the retrieved context beyond what is strictly required to answer the
question.
Retrieval ranking has a pronounced effect on faithfulness (Figure 6). As the target chunk rank r
increases, faithfulness drops sharply, consistent with models relying more heavily on earlier context
and drawing in extraneous guidance when the most relevant information appears later in the prompt.
By contrast, completeness and clarity show only modest degradation beyond r≥5 , suggesting
that once models locate the relevant guidance they can still produce a coherent answer, even if the
evidence appears lower in the context.
5 Limitations
Our study has several limitations. First, in retrieval evaluation, relevance is treated as binary with
a single target chunk per query. Many public health questions are supported by multiple passages,
so single-target metrics provide only a partial view of retrieval quality and may understate the
value of retrieving alternative, equally valid chunk. Second, our analysis of chunk-length effects
leverages the existing chunk structure rather than systematically varying chunk size over the same
10

Figure 5:Judge Rubric scores per modelProportion of free-form answers that meet each rubric
criteria for all LLMs according to the LLM Judge (GPT-OSS-120B)
underlying text, limiting causal interpretation. Third, while we observe topic-level variability in
ranking quality, our experiments do not isolate the underlying drivers (e.g., topic-specific language,
formatting conventions, or document structure).
Fourth, the multiple-choice format enables partial-credit behaviours such as option elimination,
meaning that MCQA gains may overestimate real-world capability. This concern is reinforced by the
observation that high MCQA accuracy persists even when the top-ranked chunk is not the labelled
target. This gap could reflect two non-exclusive mechanisms: the retrieved chunk may contain
sufficient information to eliminate incorrect options without fully answering the question, or it may
contain genuinely relevant guidance that our single-target labelling scheme does not capture.
In free-form evaluation, our rubric scores are computed with respect to a single reference chunk.
The judge cannot distinguish between genuinely unsupported additions and statements supported
elsewhere in the corpus, so faithfulness should be interpreted as “faithful to the provided evidence”
rather than “faithful to the entire guidance corpus”. Additionally, weak judge–human agreement on
factual consistency means that full-dataset scores for this criterion should be treated as indicative
rather than definitive. Finally, because PubHealthBench questions are derived from the guidance
itself, they are by construction answerable from the corpus and do not test multi-chunk reasoning or
out-of-corpus queries where safe systems should acknowledge uncertainty.
11

Figure 6:Effect of target chunk rank on criteria scoresProportion of free-form answers that meet
each judge criteria when the target chunk is retrieved in rank ranges r= 1,2≤r≤4,5≤r≤
7,8≤r≤15.
6 Discussion
6.1 Retrieval
The strong performance ofNV-Embed-v2across all metrics aligns with external evaluations where
NV-Embed models top the MTEB leaderboard, reflecting deliberate optimisation for embedding
and search tasks [ 45]. The universal gains from hybrid retrieval are consistent with prior work
showing that combining dense and sparse signals improves recall on domain-specific tasks by pairing
semantic similarity with specialised language matching. Importantly, hybrid retrieval enables small
and medium-sized models to perform comparably to much larger models, making it essential when
working within compute and memory constraints. Given the wide range of viable retrieval configura-
tions in our public health guidance setting, domain-specific benchmarks such as PubHealthBench are
essential for validating and selecting effective setups.
Chunk design also exerts a clear influence. The degradation of ranking quality with increasing
chunk length (§4.1, Figure 1) is consistent with longer chunks blurring topical boundaries, mixing
multiple concepts, and reducing the discriminative power of embeddings. That LLM-generated
summaries partially mitigate this effect suggests semantic compression can make relevant material
easier to retrieve; however, ranking quality still drops for the longest chunks, indicating that single-
shot summaries do not fully preserve the semantics of complex guidance sections. These findings
suggest pairing summarisation with more principled chunking (e.g., structure-aware or topic-aware
segmentation) rather than relying on summarisation alone.
The weak but systematic effect of topic on ranking quality but not recall (Figure 2) suggests that
general-purpose embedding models struggle to separate the most relevant text from superficially
similar content in narrow public-health subdomains. This consistency across models points to topic-
associated properties of the queries and guidance chunks, such as language patterns, structure, or
formatting, as the likely drivers. Where fine-grained improvements in ranking quality are needed, a
further systematic exploration of how these properties individually impact retrieval is required.
12

6.2 MCQA with Retrieval
Our results demonstrate that with high-quality retrieval, small and medium-sized open-weight models
reliably outperform a larger closed model used without retrieval. For knowledge-intensive QA over
PubHealthBench, retrieval quality can thus compensate for reduced parameter count: it is more
effective to pair a smaller LLM with a well-designed retrieval stack than to rely on a larger model
operating without retrieval. The optimal context size for smaller models is k∈ {3,5} chunks; beyond
that range, additional chunks introduce noise and yield diminishing or negative returns relative to the
extra token cost. This constrains the useful context budget for lightweight deployments and suggests
that optimisation effort should focus on what goes into those few chunks rather than simply increasing
k.
Within this constrained context window, retrieval quality, first in terms of recall, and then ranking
quality, is the primary driver of MCQA performance (Figure 4). The interaction between rank
sensitivity and model size creates a spectrum of viable design points. Smaller LLMs are more
sensitive to ranking quality and work best with fewer, well-ranked chunks, so they must be paired
with retrieval setups that keep relevant evidence near the top of a short list. Larger models can
partially mitigate weaker ranking because they are better at extracting information from noisier
contexts, making it possible to trade off retrieval quality against constraints such as latency, cost, or
infrastructure limitations.
6.3 Free-Form QA with Retrieval
Many of our MCQA findings generalise to the more realistic free-form setting. Medium and large
models generally identify relevant guidance from long retrieved contexts and express it in a clear,
usable form even without multiple-choice options. The main practical challenge is controlling
scope: models often draw broadly from retrieved context, producing answers that go beyond what is
required and may combine multiple recommendations. In a public health setting, this matters because
additional guidance, even when plausible, can reduce response precision and increase opportunities
for misunderstanding. The sharp decline in faithfulness as target chunk rank increases (Figure 6)
reinforces ranking quality as a key design consideration for RAG systems in free-form public health
QA.
Our human validation results inform the reliability of these automated scores. The judge shows
strong alignment with humans for faithfulness and reasonable consistency for completeness, so these
dimensions provide the most dependable basis for comparing retrieval setups and models at scale.
By contrast, weak judge–human agreement on factual consistency and low inter-rater agreement on
clarity highlight the need for further work on automated evaluation methods that more consistently
reproduce human judgments of response quality in public health QA.
6.4 Public Health Implications
Together, these findings have several implications for the design and deployment of RAG systems
in public health. In our setup, augmenting LLMs with UK public health documents is sufficient
to keep the system aligned with official guidance, to a high degree of accuracy ( ∼99% MCQA
accuracy), without requiring model retraining. This is consistent with recent work showing that access
to up-to-date sources is a key determinant of answer quality on our public health QA benchmark,
and that RAG yields more faithful and evidence-backed responses than closed-book LLMs alone
[1,15,18]. For public health agencies that revise guidance in response to emerging evidence or
outbreaks, the ability to update the knowledge base directly is important for maintaining the accuracy
of LLM applications.
Second, a well-designed retrieval stack allows small and medium-sized open-weight models to match
or exceed the performance of larger models running without retrieval. This makes RAG-based
systems a practical option for public health institutions operating under budget and infrastructure
constraints, offering both competitive performance and greater control over factors such as cost and
information update cadence.
Additionally, public health deployments bring specific data-governance requirements. Health data are
highly sensitive, and regulators increasingly emphasise data minimisation, strict control of sensitive
data, and careful consideration of where and how LLMs are hosted [ 61,62,63]. RAG is compatible
13

with deployment patterns in which models and knowledge bases are hosted on-premise or within
controlled environments, while still providing users with natural-language access to public health
guidance.
7 Conclusion
This work extends PubHealthBench into a retrieval-augmented setting by allowing models to retrieve
evidence from the same corpus of UK public health guidance used to construct the benchmark [ 1].
Using this setup, we make three key findings. First, we show that retrieval effectiveness in this public
health guidance domain is highly sensitive to embedding model choice and system design: hybrid
retrieval consistently improves retrieval quality, carefully crafting chunk length and uisng larger
embedding models also improve relevance ranking. Second, we find that introducing retrieval of
official guidance substantially improves MCQA performance, enabling smaller open-weight models
to match or outperform a larger models used without retrieval, with gains driven primarily by retrieval
quality and careful context selection. Third, we find that these improvements largely carry over to
free-form answering, where high-quality ranking yields more focused and faithful responses, but
also exposes task-design limitations that matter for real deployments (e.g., multi-evidence support
and out-of-corpus queries). Further, our LLM judge evaluations do not fully match human expert
assessments, indicating the need for further development in this area as robust automated evaluations
are a key requirement for deploying LLM based applications in public health use cases [64].
Overall, our results reinforce a practical conclusion for the application of RAG in public health:
the retrieval layer is one of the primary determinants of QA performance and a key lever for
deploying capable systems under realistic operational constraints. By grounding responses in an
well maintained guidance corpus, RAG provides a pathway to systems that are easier to keep up
to date as recommendations evolve, and more feasible to deploy in controlled environments where
governance and data security requirements are stringent [ 63]. Future work should strengthen the
benchmark’s retrieval supervision (e.g., multiple relevant chunks per query), broaden evaluation for
free-form responses, and introduce harder settings that require multi-chunk reasoning and explicitly
test out-of-corpus failure modes.
References
[1]Joshua Harris, Fan Grayson, Felix Feldman, Timothy Laurence, Toby Nonnenmacher, Oliver
Higgins, Leo Loman, Selina Patel, Thomas Finnie, Samuel Collins, and Michael Borowitz.
Healthy llms? benchmarking llm knowledge of uk government public health information.arXiv
preprint arXiv:2505.06046, 2025.
[2]Blog Editor. How we are pioneering artificial intelligence applications in public health , Mar
2025.
[3]Karan Singhal, Shekoofeh Azizi, Tao Tu, S. Sara Mahdavi, Jason Wei, Hyung Won Chung,
Nathan Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, Perry Payne, Martin Senevi-
ratne, Paul Gamble, Chris Kelly, Nathaneal Scharli, Aakanksha Chowdhery, Philip Mansfield,
Blaise Aguera y Arcas, Dale Webster, Greg S. Corrado, Yossi Matias, Katherine Chou, Juraj
Gottweis, Nenad Tomasev, Yun Liu, Alvin Rajkomar, Joelle Barral, Christopher Semturs, Alan
Karthikesalingam, and Vivek Natarajan. Large language models encode clinical knowledge,
2022.
[4]Reed T. Sutton, David Pincock, Daniel C. Baumgart, Daniel C. Sadowski, Richard N. Fedorak,
and Karen I. Kroeker. An overview of clinical decision support systems: benefits, risks, and
strategies for success.NPJ Digital Medicine, 3:17, February 2020.
[5]European Centre for Disease Prevention and Control. Evidence-based methodologies for public
health: How to assess the best available evidence when time is limited and there is lack of
sound evidence. Technical report, European Centre for Disease Prevention and Control (ECDC),
Stockholm, September 2011.
[6]World Health Organization. The living approach to who normative products and country
implementation: Member state briefing. Member State Briefing (PDF slides), October 2022.
14

[7]World Health Organization. Disinformation and public health. Questions and answers, February
2024. Accessed 2026-02-06.
[8]Israel Júnior Borges do Nascimento, Ana Beatriz Pizarro, Jussara M. Almeida, Natasha
Azzopardi-Muscat, Marisa A. Gonçalves, Mattias Björklund, and David Novillo-Ortiz. Info-
demics and health misinformation: a systematic review of reviews.Bulletin of the World Health
Organization, 100(9):544–561, September 2022.
[9]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and
Douwe Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks.Advances in
Neural Information Processing Systems (NeurIPS), 2020.
[10] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun,
et al. Retrieval-augmented generation for large language models: A survey.arXiv preprint
arXiv:2312.10997, 2023.
[11] Chaitanya Sharma. Retrieval-augmented generation: A comprehensive survey of architectures,
enhancements, and robustness frontiers.arXiv preprint arXiv:2506.00054, 2025.
[12] Bo Ni, Zheyuan Liu, Leyao Wang, Yongjia Lei, Yuying Zhao, Xueqi Cheng, Qingkai Zeng, Luna
Dong, Yinglong Xia, Krishnaram Kenthapadi, Ryan Rossi, Franck Dernoncourt, Md Mehrab
Tanjim, Nesreen Ahmed, Xiaorui Liu, Wenqi Fan, Erik Blasch, Yu Wang, Meng Jiang, and
Tyler Derr. Towards trustworthy retrieval augmented generation for large language models: A
survey.arXiv preprint arXiv:2502.06872, 2025.
[13] Yucheng Shi, Shaochen Xu, Tianze Yang, Zhengliang Liu, Tianming Liu, Quanzheng Li, Xiang
Li, and Ninghao Liu. Mkrag: Medical knowledge retrieval augmented generation for medical
question answering.arXiv preprint arXiv:2309.16035, 2024.
[14] Marius Vach, Michael Gliem, Daniel Weiss, Vivien Lorena Ivan, Frederik Hauke, Christian
Boschenriedter, Christian Rubbert, and Julian Caspers. Evaluating retrieval-augmented gen-
eration–enhanced large language models for question answering on german neurovascular
guidelines.Clinical Neuroradiology, 2025. (Online ahead of print).
[15] Marcos Fernández-Pichel, Juan C. Pichel, and David E. Losada. Evaluating search engines and
large language models for answering health questions.npj Digital Medicine, 8(1):153, 2025.
[16] Lawrence K. Q. Yan, Qian Niu, Ming Li, Yichao Zhang, Caitlyn H. Yin, Cheng Fei, Benji
Peng, Ziqian Bi, Pohsun Feng, et al. Large language model benchmarks in medical tasks.arXiv
preprint arXiv:2410.21348, 2024.
[17] Andrew Maranhão Ventura D’Addario. Healthqa-br: A system-wide benchmark reveals critical
knowledge gaps in large language models.arXiv preprint arXiv:2506.21578, 2025.
[18] Rahul K. Arora, Jason Wei, Rebecca Soskin Hicks, Preston Bowman, Joaquin Quiñonero-
Candela, Foivos Tsimpourlas, Michael Sharman, Meghan Shah, Andrea Vallone, Alex Beutel,
Johannes Heidecke, and Karan Singhal. Healthbench: Evaluating large language models towards
improved human health, 2025.
[19] Joshua Harris, Timothy Laurence, Leo Loman, Fan Grayson, Toby Nonnenmacher, Harry Long,
Loes WalsGriffith, Amy Douglas, Holly Fountain, Stelios Georgiou, Jo Hardstaff, Kathryn
Hopkins, Y-Ling Chi, Galena Kuyumdzhieva, Lesley Larkin, Samuel Collins, Hamish Mo-
hammed, Thomas Finnie, Luke Hounsome, Michael Borowitz, and Steven Riley. Evaluating
large language models for public health classification and extraction tasks, 2025.
[20] Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie
Millican, George van den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark,
Diego de Las Casas, Aurelia Guy, Jacob Menick, Roman Ring, Tom Hennigan, Saffron Huang,
Loren Maggiore, Chris Jones, Albin Cassirer, Andy Brock, Michela Paganini, Geoffrey Irving,
Oriol Vinyals, Simon Osindero, Karen Simonyan, Jack W. Rae, Erich Elsen, and Laurent
Sifre. Improving language models by retrieving from trillions of tokens.arXiv preprint
arXiv:2112.04426, 2022.
15

[21] Rulin Shao, Jacqueline He, Akari Asai, Weijia Shi, Tim Dettmers, Sewon Min, Luke Zettle-
moyer, and Pang Wei Koh. Scaling retrieval-based language models with a trillion-token
datastore.arXiv preprint arXiv:2407.12854, 2024.
[22] Mingyue Cheng, Yucong Luo, Jie Ouyang, Qi Liu, Huijie Liu, Li Li, Shuo Yu, Bohou Zhang,
Jiawei Cao, Jie Ma, and Daoyu Wang. A survey on knowledge-oriented retrieval-augmented
generation.arXiv preprint arXiv:2503.10677, 2025.
[23] Yan Fang, Jingtao Zhan, Qingyao Ai, Jiaxin Mao, Weihang Su, Jia Chen, and Yiqun Liu. Scaling
laws for dense retrieval, 2024.
[24] Sebastian Bruch, Siyu Gai, and Amir Ingber. An analysis of fusion functions for hybrid retrieval.
ACM Transactions on Information Systems, 42(1):1–35, 2023.
[25] Antoine Chaffin. Gte-moderncolbert, 2025.
[26] Junde Wu, Jiayuan Zhu, Yunli Qi, Jingkun Chen, Min Xu, Filippo Menolascina, and Vicente
Grau. Medical graph rag: Towards safe medical large language model via graph retrieval-
augmented generation, 2024.
[27] Martin Böckling, Heiko Paulheim, and Andreea Iana. Walk&retrieve: Simple yet effec-
tive zero-shot retrieval-augmented generation via knowledge graph walks.arXiv preprint
arXiv:2505.16849, 2025.
[28] Adithya Pratapa and Teruko Mitamura. Estimating optimal context length for hybrid retrieval-
augmented multi-document summarization.arXiv preprint arXiv:2504.12972, 2025.
[29] Bowen Jin, Jinsung Yoon, Jiawei Han, and Sercan O. Arik. Long-context llms meet rag:
Overcoming challenges for long inputs in rag. InThe Thirteenth International Conference on
Learning Representations (ICLR), 2025.
[30] Siran Li, Linus Stenzel, Carsten Eickhoff, and Seyed Ali Bahrainian. Enhancing retrieval-
augmented generation: A study of best practices. InProceedings of the 31st International
Conference on Computational Linguistics (COLING ’25), 2025.
[31] Qitao Qin, Yucong Luo, Yihang Lu, Zhibo Chu, and Xianwei Meng. Towards adaptive
memory-based optimization for enhanced retrieval-augmented generation.arXiv preprint
arXiv:2504.05312, 2025.
[32] Ankit Pal, Logesh Kumar Umapathi, and Malaikannan Sankarasubbu. Medmcqa: A large-scale
multi-subject multi-choice dataset for medical domain question answering. In Gerardo Flores,
George H Chen, Tom Pollard, Joyce C Ho, and Tristan Naumann, editors,Proceedings of
the Conference on Health, Inference, and Learning, volume 174 ofProceedings of Machine
Learning Research, pages 248–260. PMLR, 07–08 Apr 2022.
[33] Di Jin, Eileen Pan, Nassim Oufattole, Weihuong Weng, Hanyi Fang, and Peter Szolovits. What
disease does this patient have? a large-scale open domain question answering dataset from
medical exams.Applied Sciences, 11(14):6421, 2021.
[34] Shrutika Singh, Anton Alyakin, Daniel A. Alber, Jaden Stryker, Miguel Hernandez-Rovira,
Ki Yun Park, Eric K. Oermann, et al. It is too many options: Pitfalls of multiple-choice questions
in generative ai and medical education.arXiv preprint arXiv:2503.13508, 2025.
[35] Pedram Hosseini, Jessica M. Sin, Bing Ren, Bryceton G. Thomas, Elnaz Nouri, Ali Farahanchi,
and Saeed Hassanpour. A benchmark for long-form medical question answering.arXiv preprint
arXiv:2411.09834, 2024.
[36] Itay Manes, Naama Ronn, David Cohen, Ran Ilan Ber, Zehavi Horowitz-Kugler, and Gabriel
Stanovsky. K-qa: A real-world medical q&a benchmark. InProceedings of the 23rd Workshop
on Biomedical NLP (BioNLP 2023), pages 277–294, 2023.
16

[37] Charles Nimo, Tobi Olatunji, Abraham Toluwase Owodunni, Tassallah Abdullahi, Emmanuel
Ayodele, Mardhiyah Sanni, Ezinwanne C. Aka, Folafunmi Omofoye, Foutse Yuehgoh, Timothy
Faniran, Bonaventure F. P. Dossou, Moshood O. Yekini, Jonas Kemp, Katherine A Heller,
Jude Chidubem Omeke, Chidi Asuzu Md, Naome A Etori, Aïmérou Ndiaye, Ifeoma Okoh,
Evans Doe Ocansey, Wendy Kinara, Michael L. Best, Irfan Essa, Stephen Edward Moore,
Chris Fourie, and Mercy Nyamewaa Asiedu. AfriMed-QA: A pan-African, multi-specialty,
medical question-answering benchmark dataset. In Wanxiang Che, Joyce Nabende, Ekaterina
Shutova, and Mohammad Taher Pilehvar, editors,Proceedings of the 63rd Annual Meeting of
the Association for Computational Linguistics (Volume 1: Long Papers), pages 1948–1973,
Vienna, Austria, July 2025. Association for Computational Linguistics.
[38] Ning Xian, Yixing Fan, Ruqing Zhang, Maarten de Rijke, and Jiafeng Guo. An empirical study
of evaluating long-form question answering. InProceedings of the 48th International ACM
SIGIR Conference on Research and Development in Information Retrieval, 2025.
[39] Fangyuan Xu, Yixiao Song, Mohit Iyyer, and Eunsol Choi. A critical evaluation of evaluations
for long-form question answering. InProceedings of the 61st Annual Meeting of the Association
for Computational Linguistics (ACL), pages 3225–3245, 2023.
[40] Seungone Kim, Jamin Shin, Yejin Cho, Joel Jang, Shayne Longpre, Hwaran Lee, Sangdoo
Yun, Seongjin Shin, Sungdong Kim, James Thorne, and Minjoon Seo. Prometheus: Inducing
fine-grained evaluation capability in language models.arXiv preprint arXiv:2310.08491, 2023.
[41] Seungone Kim, Jamin Shin, Yejin Cho, Joel Jang, Shayne Longpre, Hwaran Lee, Sangdoo
Yun, Seongjin Shin, Sungdong Kim, James Thorne, and Minjoon Seo. Prometheus 2: An
open source language model specialized in evaluating other language models.arXiv preprint
arXiv:2405.01535, 2024.
[42] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang,
Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica.
Judging llm-as-a-judge with mt-bench and chatbot arena.arXiv preprint arXiv:2306.05685,
2023.
[43] Jiahui Gu et al. A survey on llm-as-a-judge.arXiv preprint arXiv:2411.15594, 2024.
[44] Euan Croxford et al. Automating evaluation of ai text generation in healthcare using medical
llm-as-a-judge.medRxiv, 2025. Version v2; PubMed PMID: 40313300.
[45] Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan
Catanzaro, and Wei Ping. Nv-embed: Improved techniques for training llms as generalist
embedding models.arXiv preprint arXiv:2405.17428, 2024. Includes NV-Embed-v1 and
NV-Embed-v2.
[46] Min Choi, Sahil Dua, Alice Lisak, and Google DeepMind. Introducing embeddinggemma: The
best-in-class open model for on-device embeddings. Google Developer Blog, September 4
2025.
[47] Benjamin Warner, Antoine Chaffin, Benjamin Clavié, Orion Weller, Oskar Hallström, Said
Taghadouini, Alexis Gallagher, Raja Biswas, Faisal Ladhak, Tom Aarsen, Nathan Cooper,
Griffin Adams, Jeremy Howard, and Iacopo Poli. Smarter, better, faster, longer: A modern
bidirectional encoder for fast, memory efficient, and long context finetuning and inference,
2024.
[48] Rui Meng, Ye Liu, Shafiq Rayhan Joty, Caiming Xiong, Yingbo Zhou, and Semih Yavuz.
Sfr-embedding-mistral: Enhance text retrieval with transfer learning. Salesforce AI Research
Blog / Hugging Face model card, October 2024.
[49] Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei.
Multilingual e5 text embeddings: A technical report.arXiv preprint arXiv:2402.05672, 2024.
Includes themultilingual-E5-largemodel.
[50] OpenAI. text-embedding-3-large: Openai’s next-generation large embedding model. OpenAI
API / Documentation, January 2024. Creates embeddings with up to 3072 dimensions.
17

[51] Gordon V . Cormack, Charles L. A. Clarke, and Stefan Büttcher. Reciprocal rank fusion
outperforms condorcet and individual rank learning methods. InProceedings of the 32nd
Annual International ACM SIGIR Conference on Research and Development in Information
Retrieval (SIGIR ’09), pages 758–759. ACM, 2009.
[52] Deven B. Dangi, Beni B. Dangi, and Oliver Steinbock. Evaluation of gpt-4o and gpt-4o-mini’s
vision capabilities for compositional analysis from dried solution drops, 2025.
[53] A. Grattafiori et al. The llama 3 herd of models.arXiv preprint arXiv:2407.21783, 2024.
Describes the Llama 3 family, including the 70B instruction-tuned model.
[54] Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar,
Michael Harrison, Russell J. Hewett, Mojan Javaheripi, Piero Kauffmann, James R. Lee,
Yin Tat Lee, Yuanzhi Li, Weishung Liu, Caio C. T. Mendes, Anh Nguyen, Eric Price, Gustavo
de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Xin Wang, Rachel Ward, Yue Wu, Dingli Yu,
Cyril Zhang, and Yi Zhang. Phi-4 technical report.arXiv preprint arXiv:2412.08905, 2024.
14B-parameter language model.
[55] Gemma Team, Google DeepMind. Gemma 3: A multimodal, multilingual, long-context open
model family.arXiv preprint arXiv:2503.19786, 2025.
[56] Andrew Sellergren, Sahar Kazemzadeh, Tiam Jaroensri, Atilla Kiraly, Madeleine Traverse,
Timo Kohlberger, Shawn Xu, Fayaz Jamil, Cían Hughes, Charles Lau, et al. Medgemma: A
multimodal generative model family for medical text and image comprehension.arXiv preprint
arXiv:2507.05201, 2025.
[57] Cohere Labs / CohereForAI. Cohere labs command-r (c4ai-command-r-v01). Hugging Face
model card, March 2024. 35B parameter generative model; 128K token context.
[58] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. Beir:
A heterogenous benchmark for zero-shot evaluation of information retrieval models.arXiv
preprint arXiv:2104.08663, 2021.
[59] Olivier Jeunen, Ivan Potapov, and Aleksei Ustimenko. On (normalised) discounted cumulative
gain as an off-policy evaluation metric for top- nrecommendation. InProceedings of the 30th
ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’24). Association
for Computing Machinery, 2024.
[60] Xiaoyue Wang, Jianyou Wang, Weili Cao, Kaicheng Wang, Ramamohan Paturi, and Leon
Bergen. Birco: A benchmark of information retrieval tasks with complex objectives.arXiv
preprint arXiv:2402.14151, 2024.
[61] HongYi Li, Jun-Fen Fu, and Andre Python. Implementing large language models in health
care: Clinician-focused review with interactive guideline.Journal of Medical Internet Research,
27(1):e71916, 2025.
[62] European Data Protection Board. Opinion 28/2024 on certain data protection aspects related to
the processing of personal data in the context of ai models. EDPB Opinion 28/2024, European
Data Protection Board, Brussels, Belgium, December 2024. Adopted on 17 December 2024.
[63] Department of Health and Social Care and Government Digital Service. A guide to good
practice for digital and data-driven health technologies. https://www.gov.uk/data-ethic
s-guidance/a-guide-to-good-practice-for-digital-and-data-driven-healt
h-technologies, 2025. Published 27 January 2025; accessed 9 December 2025.
[64] Karim Lekadir, Alejandro F. Frangi, Antonio R. Porras, Ben Glocker, Celia Cintas, Curtis P.
Langlotz, Eva Weicken, Folkert W. Asselbergs, Fred Prior, Gary S. Collins, Georgios Kaissis,
Gianna Tsakou, Irène Buvat, Jayashree Kalpathy-Cramer, John Mongan, Julia A. Schnabel,
Kaisar Kushibar, Katrine Riklund, Kostas Marias, Lameck M. Amugongo, Lauren A. Fromont,
Lena Maier-Hein, Leonor Cerdá-Alberich, Luis Martí-Bonmatí, M. Jorge Cardoso, Maciej
Bobowicz, Mahsa Shabani, Manolis Tsiknakis, Maria A. Zuluaga, Marie-Christine Fritzsche,
Marina Camacho, Marius George Linguraru, Markus Wenzel, Marleen De Bruijne, Martin G.
Tolsgaard, Melanie Goisauf, Mónica Cano Abadía, Nikolaos Papanikolaou, Noussair Lazrak,
18

Oriol Pujol, Richard Osuala, Sandy Napel, Sara Colantonio, Martijn P. A. Starmans, Smriti
Joshi, Stefan Klein, Susanna Aussó, Wendy A. Rogers, Zohaib Salahuddin, and FUTURE-
AI Consortium. Future-ai: international consensus guideline for trustworthy and deployable
artificial intelligence in healthcare.BMJ, 388:e081554, February 2025.
19