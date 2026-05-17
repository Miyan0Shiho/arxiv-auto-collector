# ASTRA-QA: A Benchmark for Abstract Question Answering over Documents

**Authors**: Shu Wang, Shansong Zhou, Xinyang Wang, Shiwei Wang, Hulong Wu, Yixiang Fang

**Published**: 2026-05-11 08:17:24

**PDF URL**: [https://arxiv.org/pdf/2605.10168v1](https://arxiv.org/pdf/2605.10168v1)

## Abstract
Document-based question answering (QA) increasingly includes abstract questions that require synthesizing scattered information from long documents or across multiple documents into coherent answers. However, this setting is still poorly supported by existing benchmarks and evaluation methods, which often lack stable abstract references or rely on coarse similarity metrics and unstable head-to-head comparisons. To alleviate this issue, we introduce ASTRA-QA, a benchmark for AbSTRAct Question Answering over documents. ASTRA-QA contains 869 QA instances over academic papers and news documents, covering five abstract question types and three controlled retrieval scopes. Each instance is equipped with explicit evaluation annotations, including answer topic sets, curated unsupported topics, and aligned evidence. Building on these annotations, ASTRA-QA assesses whether answers cover required key points and avoid unsupported content by directly scoring topic coverage and curated unsupported content, enabling scalable evaluation without exhaustive head-to-head comparisons. Experiments with representative Retrieval-Augmented Generation (RAG) methods spanning vanilla, graph-based, and hierarchical retrieval settings show that ASTRA-QA provides reference-grounded diagnostics for coverage, hallucination, and retrieval-scope robustness. Our dataset and code are available at https://xinyangsally.github.io/astra-benchmark.

## Full Text


<!-- PDF content starts -->

ASTRA-QA: A Benchmark for Abstract Question
Answering over Documents
Shu Wang1Shansong Zhou1Xinyang Wang1Shiwei Wang2Hulong Wu2Yixiang Fang1∗
1The Chinese University of Hong Kong, Shenzhen
2Data Science Group, Huolala
shuwang3@link.cuhk.edu.cn, fangyixiang@cuhk.edu.cn
Abstract
Document-based question answering (QA) increasingly includes abstract ques-
tions that require synthesizing scattered information from long documents or
across multiple documents into coherent answers. However, this setting is still
poorly supported by existing benchmarks and evaluation methods, which often
lack stable abstract references or rely on coarse similarity metrics and unstable
head-to-head comparisons. To alleviate this issue, we introduceASTRA-QA, a
benchmark forAbSTRAeructQuestionAnswering over documents. ASTRA-QA
contains 869 QA instances over academic papers and news documents, covering
five abstract question types and three controlled retrieval scopes. Each instance
is equipped with explicit evaluation annotations, including answer topic sets, cu-
rated unsupported topics, and aligned evidence. Building on these annotations,
ASTRA-QA assesses whether answers cover required key points and avoid un-
supported content by directly scoring topic coverage and curated unsupported
content, enabling scalable evaluation without exhaustive head-to-head compar-
isons. Experiments with representative Retrieval-Augmented Generation (RAG)
methods spanning vanilla, graph-based, and hierarchical retrieval settings show
that ASTRA-QA provides reference-grounded diagnostics for coverage, hallu-
cination, and retrieval-scope robustness. Our dataset and code are available at
https://xinyangsally.github.io/astra-benchmark.
1 Introduction
Retrieval-augmented generation (RAG) has been increasingly used for improving the faithfulness
of large language models (LLMs) in question answering (QA) by grounding generation in external
documents [ 1]. Recently, the abstract questions [ 2–7] in the RAG setting have received plenty of
attention, and they are prevalent in practical scenarios, including long-form QA [ 8], query-focused
summarization [ 2], and scientific document analysis [ 9,10]. For example, a user may ask“What are
the main differences between GraphRAG and LightRAG, and what are the strengths and limitations
of each method?”Compared with specific fact-seeking questions, they are high-level questions
requiring document-level synthesis rather than isolated fact retrieval. As a result, answering an
abstract question requires the RAG system to synthesize information over a broad document scope
and produce a coherent, selective, and faithful response.
Existing benchmarks have advanced long-form QA, scientific document QA, multi-hop reason-
ing, long-context understanding, and factual RAG evaluation, as seen in ASQA [ 11], Qasper [ 9],
PeerQA [ 10], HotpotQA [ 12], LongBench [ 13], and CRAG [ 14]. However, these benchmarks are
designed around different primary goals, such as ambiguous factoid resolution, local evidence lookup,
reasoning chains, general long-context understanding, and factual retrieval. As a result, they neither
directly target synthesis-heavy abstract questions that require methods to integrate information across
∗Corresponding author.
Preprint.arXiv:2605.10168v1  [cs.CL]  11 May 2026

Table 1: Comparison between ASTRA-QA and representative QA benchmarks. Hall-Eval and
Multi-Scope denote hallucination evaluation and multiple retrieval scopes, respectively.
Benchmark Target Evaluation Answer Topics Hall-Eval Multi-Scope
PeerQA [10] Scientific QA LLM judge
HotpotQA [12] Multi-hop QA EM/F1
LongBench [13] Long-context Mixed metrics
CRAG [14] Factual RAG LLM judge
RAG-QA Arena [15] Long-form RAG Head-to-head comparison
LiveRAG [16] Diverse QA Auto. + LLM metrics
ASTRA-QA (Ours)Abstract QA Topic-based scoring
broad document contexts, nor do they provide explicit evaluation annotations for fine-grained diag-
nostic evaluation. What remains inadequately explored is not merely the retrieval of more accurate
information, but rather the ability to perform abstract synthesis based on retrieved evidence under
controlled evaluation settings.
This mismatch also poses an evaluation challenge for abstract QA. Conventional lexical and semantic
metrics, such as ROUGE [ 17] and BERTScore [ 18], are too coarse-grained for long-form abstract
answers, whose wording may vary substantially while differing in topical coverage. QA-based and
factuality-oriented metrics, such as QuestEval [ 19], QAFactEval [ 20], and FActScore [ 21], improve
grounding analysis, but they mainly focus on consistency or factual precision rather than whether
a response covers all required answer topics. LLM-based evaluators, such as G-Eval [ 22] and
head-to-head comparison [ 2–4], can provide useful holistic judgments, but they are costly to scale
and do not explicitly ground evaluation in stable answers or source evidence. A benchmark for
abstract QA, therefore, requires explicit evaluation annotations that support direct and fine-grained
measurement of topic coverage and reference-targeted unsupported content. Table 1 summarizes the
key characteristics of ASTRA-QA and representative QA benchmarks.
Topic Set-based Evaluation
Head-to-headEvaluation
CorpusExtractedcommentsQA Instances
QA Refinement
Answer 1 contains the [1,2,3]topicsButmissesthetopics[4]andcoversthe [a]hallucinationtopicGrounded in topic sets, evaluation is objective and explainable
90T-F185T-Rec.10Hall.
Corpus
GeneratedAbstractiveQuestions
Win-rateResultLLMsays RAG1 is the best, but is thattrue?
RAG1:RAG2:RAG3:
Answer 1 is more concise and clearly …Answer 1 Win
RAG1RAG2R1R2R3R1507590R2255065R3103550
Figure 1: Comparison between existing head-to-head evaluation and our topic-based evaluation for
abstract QA.
To address these gaps, we introduceASTRA-QA, a benchmark forAbSTRActQAover documents,
as shown in Figure 1. Its design follows a simple grading intuition: a good answer should cover the
required key points while avoiding unsupported content. Accordingly, each ASTRA-QA instance
is annotated with a set of answer topics, a curated hallucination set, aligned evidence, and retrieval-
scope metadata. ASTRA-QA contains 869 QA instances over more than 700 paper documents
and 1,500 news articles, covering five synthesis-heavy question types and three retrieval scopes.
Built on these references, our evaluation method directly scores topic-based coverage and curated
hallucination-topic matches.
In summary, our principal contributions are as follows:
•We introduce ASTRA-QA, a new benchmark for abstract QA over academic papers and news doc-
uments, covering single-document summarization, two-way comparison, multi-way comparison,
thematic enumeration, and temporal reasoning.
• We propose a topic-based evaluation method that directly scores topic coverage and hallucinated
content, avoiding costly head-to-head judgments.
•We evaluate representative RAG methods on ASTRA-QA and analyze their behavior across five
question types and three retrieval scopes, highlighting the benchmark’s ability to surface coverage,
hallucination, and retrieval-scope challenges.
2

2 Related Work
RAG has been widely used for knowledge-intensive QA, where retrieved documents help LLMs
produce factual answers beyond parametric memory alone [ 1,23]. Recent RAG methods [ 2–4,7,24,
25] have moved further toward long-form and document-level synthesis through hierarchical retrieval,
graph-structured indexing, and long-context reading. These methods show that summarization,
comparison, and broader corpus-level synthesis have become realistic RAG targets. However,
abstract questions still lack dedicated datasets and an evaluation method to assess whether generated
answers are complete, grounded, and free of unsupported additions.
Benchmarks for RAG.Existing QA benchmarks for RAG include open-domain retrieval, multi-
hop reasoning, scientific document QA, and long-context understanding, as seen in ELI5 [ 26],
ASQA [ 11], HotpotQA [ 12], Qasper [ 9], PeerQA [ 10], and LongBench [ 13]. These resources
establish the importance of long answers, evidence supervision, and difficult retrieval settings, but
most still center on factoid resolution, local information seeking, or general long-context evaluation.
They do not jointly provide curated abstract references for comparison, enumeration, and temporal
synthesis, nor do they expose topic-level targets that make answer completeness and hallucination
behavior easy to diagnose. ASTRA-QA is designed for this missing setting by providing curated
QA instances with reference topic sets, hallucination sets, and aligned evidence spans over academic
papers and news documents.
Evaluation for Abstract QA.Early evaluation of answers to abstract questions relied on lexical or
semantic reference matching, such as ROUGE [ 17], BERTScore [ 18], and QA-based or factuality
metrics such as FEQA [ 27], QuestEval [ 19], QAFactEval [ 20], and FActScore [ 21]. These metrics
provide useful surface, semantic, or grounding signals, but they do not jointly diagnose whether a
response covers all required topics and avoids unsupported additions. With the development of LLM
judges, reference-guided evaluation offers another direction [ 28], as in RAG-QA Arena [ 15] and
G-Eval [ 22]. Head-to-head comparison is also widely used in abstract RAG evaluation, where an
LLM judge directly compares two RAG system outputs [ 2–4,6,7]. Such evaluations are useful for
holistic preference judgments, but they make it difficult to compare systems under stable reference
targets or to obtain diagnostic coverage and hallucination metrics. ASTRA-QA therefore emphasizes
interpretable reference targets for coverage and hallucination analysis rather than relying only on
answer-level similarity or pairwise preference.
3 Our ASTRA-QA Benchmark
In this section, we present the construction process of ASTRA-QA, including its design goals,
construction pipeline, question types, and dataset statistics.
3.1 Overview of ASTRA-QA
ASTRA-QA is a benchmark for abstract QA over documents, with a focus on evaluating RAG
methods. Unlike conventional QA benchmarks that emphasize short answers or extractive evidence
lookup, ASTRA-QA is designed to evaluate whether a RAG system can synthesize information from
long documents and produce responses that are coherent, selective, and faithful to the source content.
Such questions commonly require document-level summaries, structured comparisons, thematic
organization, and temporally grounded synthesis rather than isolated factual snippets. ASTRA-QA
is built over two source domains, namely academic papers and news documents, and covers five
abstract question types.
Formally, ASTRA-QA consists of a document corpus Dand a set of QA instances. Each QA instance
is represented as (Q, A, H, M) , where Qis an abstract question, A={τ 1, τ2,···, τ n}is the answer
topic set, H={h 1, h2,···, h k}is a curated hallucination set containing plausible but unsupported
topics, and Mdenotes the associated metadata, including the aligned evidence set E, the question
type, and the retrieval scope. Figure 2 provides an example of this representation, showing the
question, answer topic set, hallucination set, and metadata of a QA instance. The set Ais constructed
from high-level summary signals and used by our topic-based evaluation method to assess answer
coverage, while Hrecords relevant but unsupported topics and is used to assess whether responses
include these curated hallucination targets.
3

QuestionPlease summarize the paper“Encoding Recurrence into Transformers”.Answer Topic SetSelf-attention with recurrenceSample efficiencyGating functionMasked linear aggregation	⋯Hallucination Set Overfitting preventionMetadataQuestion Type:-Single-SumRetrievalScope:-Simple:[18]-Middle:[6,12,31]-Hard:[2,3,6,…,31]AlignedEvidenceSet:
It improves the sample efficiency ...use of self-attention with ...
RNNs are sample-efficient...
... masked linear aggregation ...use a gating function ...Figure 2: An example QA instance in the ASTRA-QA dataset.
3.2 Construction Pipeline
ASTRA-QA is constructed through a three-stage construction pipeline, as illustrated in Figure 3. The
pipeline converts heterogeneous source materials into abstract QA instances grounded in curated
document collections, with reference-grounded evidence and comprehensive topic-set answers.
OriginalDocuments
Extracted TextsCommentSourcesTags&Comments
Initial QA
Guidance
①Datacollection②QAGeneration③QARefinement
FinalQARAGCorpus
ASTRA-QA datasetAnswerRefinementTopicEnrichmentDeduplication
QuestionRefinement
IntentRevisionEvidenceAlignment
Figure 3: Workflow for constructing the ASTRA-QA dataset.
Step 1: Data collection.Original documents are collected from OpenReview, focusing on ICLR
2023 [ 29], survey papers from arXiv2, the tagged corpus of publications of Epstein et al. [ 30],
and news articles downloaded through the mediastack API3, forming an initial candidate pool of
700+ paper documents and 1,500+ news articles. 4,700+ comments are gathered separately from
OpenReview reviews4for ICLR papers (including reviewer comments: summary, strengths, and
weaknesses), tags from corpus of publications of Epstein et al. [ 30], and metadata associated with the
mediastack news records. We then performcontent extractionon both the original documents and the
matched comments. Specifically, for PDF documents, we use MinerU [ 31] to parse the document text,
whereas for HTML documents, we extract the page content directly. We finally align each comment
with its corresponding source document and merge all documents into a corpusD.
Step 2: QA instance generation.We use the processed materials, along withtype-specific guidance,
to generate initial QA instances with an LLM. Specifically, we use GPT-4o [ 32] throughout the
ASTRA-QA construction pipeline for generation and refinement. The LLM input includes the type
guidance, the matched comment text, the content of the original documents, and a small number of
examples. Each question qis required to match the question type, and its answer Ais represented as
atopic setextracted from matched comments, tags, and other summary-like context aligned to the
source documents, with 17.03 topics on average. We design five common abstract question types,
including summarization, pair comparison, multi-target comparison, enumeration, and temporal
reasoning, and instantiate a different prompt template for each type. The type-specific generation
details are described in Section 3.3.
Step 3: QA instance refinement.We refine the generated question and the topic-set answer
separately.(1) Question refinement. We performevidence alignmentby checking whether the
question can be answered from the selected evidence documents alone. To do so, we retrieve 10
related but non-evidence documents as probes and use an LLM to judge whether the question depends
on information outside the designated evidence scope. We then revise thequestion intentand enforce
consistency with the assigned question type, e.g., by making the comparison dimension explicit.
(2) Answer refinement. We first perform topic enrichment. We retrieve 5 similar questions to obtain
candidate supplemental topics. We then use an LLM to judge, based on the original question, the
original evidence, and the matched comment, whether each candidate topic should be included in
A. Accepted topics are added to A, while rejected but plausible unsupported topics are recorded
2https://arxiv.org/
3https://mediastack.com/
4https://openreview.net/
4

in the curatedhallucination set Hfor later evaluation. We nextdeduplicatethe enriched topic set
Aby using an LLM to merge or remove repeated and semantically equivalent topics. This stage
turns model-generated drafts into curated benchmark samples. After manual review, wefilter out
low-quality QA instances and their associated documents, resulting in the final ASTRA-QA dataset.
3.3 Question Type Design and Answer Annotation
We next describe the five common types of abstract questions covered by ASTRA-QA and explain
how the corresponding QA instances are constructed. Example QA instances for five question types
are provided in the appendix.
•SINGLE-SUM(single-document summarization).Single-Sum asks for a summary of one paper
or one document-centered source, targeting the overall content of the document rather than a local
fact or passage-level detail. We construct this question type by providing the LLM with the original
document, reviewer comments, and examples, and then generating a summary question together
with a topic-set answer.
•PAIR-COMP(two-way comparison).Pair-Comp is used when the source text presents two
methods, entities, documents, or events under an explicit contrast. We use survey passages or other
summary-like evidence that explicitly compare two targets, and require the answer to be organized
into two separate target-specific topic sets rather than a merged summary.
•MULTI-COMP(multi-way comparison).Multi-Comp extends comparison to settings with at
least three items. Its construction relies on survey-style comparison passages or other summary-like
text that place multiple targets under a shared discussion scope, and the answer is required to keep
a consistent target-by-target structure across all compared items.
•ENUMERATION(thematic enumeration).Enumeration asks for as many relevant items as possible
under a shared theme, such as contributions, reasons, or findings. We construct this question from
summary-like passages that discuss multiple aspects of the same topic, and require the answer to be
organized as a list of parallel topic items with minimal redundancy.
•TEMPORAL(temporal reasoning).Temporal asks about an event, fact, or state within a specified
time range, or about how it changes over time. Its construction relies on time-bounded news
documents and related summary-like text, and the answer is required to reflect temporal stages or
updates rather than a static summary.
To evaluate how the retrieval range affects RAG performance on abstract questions, we define three
retrieval settings, denoted as SIMPLE, MIDDLE, and HARD, by varying the retrieval span while
keeping the reference answer fixed. SIMPLE: the search space is restricted to the exact document or
tightly scoped documents that directly support the question; for single-document summarization, this
reduces to the source document itself. MIDDLE: we cluster semantically related questions and merge
the evidence documents within each cluster, yielding a broader but still relevant corpus. HARD: all
documents belonging to the same question type are pooled together, creating the largest retrieval
space and introducing substantially more distractors. This design allows ASTRA-QA to probe how
RAG methods behave as the search space expands from the most precise corpus scope to broader and
more noise-heavy corpora.
3.4 Dataset Statistics
Table 2 summarizes the ASTRA-QA dataset across the five question types and the retrieval difficulty
settings employed in our experiments. After filtering and deduplication, ASTRA-QA comprises 869
QA instances grounded in 2,095 unique source documents and 16,080,106 corpus tokens, organized
into 54 clusters under the MIDDLEsetting. The average token count for this setting is 347,963,
as indicated by theAMTok. column, while the largest MIDDLEcluster contains 736,089 tokens.
SINGLE-SUMconstitutes the largest portion of the benchmark, accounting for 48.6% of the total,
while the remaining question types provide complementary coverage of cross-document synthesis
and event-centric abstraction.
Another distinctive property of ASTRA-QA is its topic-set answer representation, as shown in
Figure 4. Instead of relying on a single free-form reference, each question is equipped with a set of
salient, source-grounded answer topics. Most answers contain 4–18 topics, with none being empty,
whereas hallucination sets are concentrated in the 0–3 range, and no hallucination sets exceed 13
5

Question Type #Q #Docs Corpus Tok.AMTok. #C
Single-Sum 422 422 9,681,570 322,719 30
Pair-Comp 99 54 1,565,393 597,178 5
Multi-Comp 42 57 1,670,693 457,473 5
Enumeration 150 64 1,728,257 427,368 7
Temporal 156 1,579 1,434,193 120,514 7
Total 869 2,095 16,080,106 347,963 54
Table 2: Data statistics of ASTRA-QA across question
types. The table uses the abbreviations for the five ques-
tion types. CandAMTok. denote cluster and average
token count of the MIDDLEsetting, respectively.
0 1-3 4-6 7-9 10-12 13-15 16-18 19-21 22-24 25-27 28-30
Number of topics in the answer set050100150200# of QA pairsT opics in answer
0 1-3 4-6 7-9 10-12 13-15 16-18 19-21 22-24 25-27 28-30
Number of topics in the hallucination set0200400# of QA pairsT opics in hallucinationFigure 4: Distribution of topics in the
answer and hallucination sets in our
ASTRA-QA.
topics. This design supports joint evaluation of completeness and faithfulness by measuring whether
a system covers the major answer topics while avoiding plausible but unsupported ones.
4 Topic-based Evaluation Method
4.1 Evaluation Method
Our core idea is to evaluate ASTRA-QA answers in the same spirit as grading a composition or a
reading-comprehension response by checking whether the answer covers the required key points. For
ASTRA-QA, a good answer should cover the answer topics in Aas completely as possible and avoid
hallucinated content.
During the ASTRA-QA evaluation, we denote the response from a RAG method to a question Q
asY. We then use an LLM-based extractor to convert Yinto a response-side topic set ˆA(Q, Y) =
{ˆτ1,ˆτ2, . . . ,ˆτ r}. The extractor is prompted with Q,A, and Hso that it can identify response-side
topics with reference to both AandH. IfYcontains content that goes beyond both AandH, that
content is also retained in ˆA(Q, Y)rather than discarded during extraction.
We use ˆτ≡τ to denote semantic equivalence between a response-side topic and an answer topic. We
next define the supported response-side topic set and the covered answer topic set as
S(Q, Y) ={ˆτ∈ ˆA(Q, Y)| ∃τ∈A,ˆτ≡τ}, C(Q, Y) ={τ∈A| ∃ˆτ∈ ˆA(Q, Y),ˆτ≡τ}.
We then compute topic precision, topic recall, and topic F1 as
T-Prec=|S(Q, Y)|
max(1,| ˆA(Q, Y)|),T-Rec=|C(Q, Y)|
|A|,T-F1=2T-Prec·T-Rec
T-Prec+T-Rec.
To evaluate curated hallucination targets, we define the covered hallucination set as
CH(Q, Y) ={h∈H| ∃ˆτ∈ ˆA(Q, Y),ˆτ≡h}.
Based on CH(Q, Y) , we define a topic-level hallucination score ( Htopic) and a response-level
hallucination rate (H resp) as
Htopic(Q, Y) =|CH(Q, Y)|
|H|, H resp(Q, Y) =I[|C H(Q, Y)|>0].
Here, I[·]denotes the indicator function. A lower Htopic means that the response covers fewer
hallucination topics in H, while Hresp(Q, Y) = 1 means that the response contains at least one
hallucination topic. For instances with an empty hallucination set, we set Htopic(Q, Y) = 0 and
Hresp(Q, Y) = 0.
4.2 Discussion of the Evaluation Method
We next discuss why this evaluation method fits abstract QA, what it reveals, and how its scope
should be interpreted.
6

Fit to abstract RAG.ASTRA-QA answers are structured syntheses rather than isolated facts, so
they require evaluation beyond short-form answer matching. The answer topic set Adefines the
expected coverage of a good answer, while the hallucination set Hprovides explicit negative targets
for hallucination checking.
Scalability and interpretability.Our evaluation method avoids exhaustive head-to-head comparisons
by scoring each method independently against the same topic references. This reduces evaluation
fromO(n2)pairwise comparisons to O(n) evaluations for nmethods, saving both annotation effort
and LLM inference cost. It also produces diagnostic scores because each result can be traced to
matched, missing, and hallucination-related topics. This makes the comparison more interpretable
than a single holistic preference label.
Scope and limitations.Evaluation quality still depends on the quality of the curated topic refer-
ences and on the reliability of the LLM-based extraction stage. The hallucination scores should be
interpreted as reference-targeted diagnostics rather than exhaustive factuality metrics: the curated
hallucination set Hcaptures plausible unsupported topics identified during benchmark construction,
but it cannot cover all hallucination patterns that may appear in model responses. Topic granularity
is also a design choice, and topic-set scoring does not fully capture discourse quality, response
organization, or stylistic preference.
5 Experiments
In this section, we evaluate representative RAG methods on ASTRA-QA and show that our benchmark
exposes coverage gaps, hallucination risks, and retrieval-scope sensitivity in current RAG systems.
5.1 Setup
Methods and implementations.We evaluate representative RAG methods targeted for abstract QA,
covering vanilla, graph-based, and hierarchical retrieval settings. The compared methods are Vanilla
RAG, RAPTOR [ 25], HippoRAG [ 24], GraphRAG [ 2], LightRAG [ 3], ArchRAG [ 4], KET-RAG [ 33],
and HiRAG [ 7]. For GraphRAG, we evaluate its global and local search, denoted as GGraphRAG
and LGraphRAG, respectively. Similarly, we evaluate the low, high, and hybrid retrieval modes for
LightRAG, denoted as LLightRAG, HiLightRAG, and HyLightRAG, respectively. We use the official
implementations for ArchRAG, KET-RAG, and HiRAG, and the Graph-based RAG framework [ 5]
for the remaining methods. All methods follow the recommended chunking and retrieval parameters
when available. For Vanilla RAG, we set the chunk size to 512 tokens and retrieve the top 10 chunks.
Models and evaluation settings.All RAG methods use Qwen3-8B [34] as the LLM backbone and
nomic-embed-text [ 35] as the embedding model. We use GPT-5.1 [ 36] for evaluation. All methods
are evaluated across three retrieval settings: SIMPLE, MIDDLE, and HARD. All model outputs are
scored with the topic-based evaluation method introduced in Section 4. Note that we report only
T-F1 ,Htopic, andHrespin this section, with the remaining metrics and additional details provided in
the appendix.
Table 3: Performance comparison of representative RAG methods on ASTRA-QA under the three
retrieval settings and overall, where TF1, HT, and HR denote T-F1,H topic, andH resp, respectively.
MethodSimple Middle Hard Overall
TF1 HT HR TF1 HT HR TF1 HT HR TF1 HT HR
Vanilla RAG 45.0 15.6 27.2 24.2 6.8 12.4 23.7 10.0 22.6 31.0 10.8 20.7
LLightRAG 55.0 9.8 20.7 36.7 15.9 30.4 33.9 17.9 37.0 41.8 14.5 29.4
HiLightRAG 54.0 8.2 16.7 48.4 12.4 29.9 46.0 15.4 31.3 49.5 12.0 26.0
HyLightRAG 40.7 7.6 19.5 40.0 13.8 26.7 38.4 17.5 34.9 39.7 13.0 27.0
LGraphRAG 57.8 11.0 22.3 39.3 11.9 27.5 38.3 13.9 27.9 45.1 12.3 25.9
GGraphRAG 23.1 5.2 12.7 22.5 7.2 15.5 21.8 7.7 17.4 22.5 6.7 15.2
HippoRAG 61.7 17.3 35.1 56.9 27.4 45.8 51.2 17.5 35.1 56.6 20.0 38.7
RAPTOR 64.0 16.4 31.7 53.3 16.6 37.6 52.1 21.1 45.5 55.3 18.0 38.3
ArchRAG 55.2 18.9 37.5 47.6 20.7 39.4 47.0 19.3 37.6 49.9 19.6 38.2
KET-RAG 32.7 4.0 6.2 27.8 3.3 6.6 12.9 5.6 12.8 24.5 4.3 8.5
HiRAG 68.9 18.1 29.2 45.2 13.8 33.0 35.7 13.4 34.0 49.9 15.1 32.1
7

Table 4: Performance comparison of representative RAG methods on ASTRA-QA across the five
question types, where TF1, HT, and HR denote T-F1,H topic, andH resp, respectively.
MethodSingle-Sum Pair-Comp Multi-Comp Enumeration Temporal
TF1 HT HR TF1 HT HR TF1 HT HR TF1 HT HR TF1 HT HR
Vanilla RAG 14.8 1.3 1.3 63.7 16.4 19.1 52.3 13.0 13.9 56.3 16.7 16.7 23.8 12.2 33.3
LLightRAG 48.1 3.8 3.8 52.3 16.5 21.3 48.4 10.7 15.3 41.6 16.7 16.7 16.6 20.4 52.2
HiLightRAG 56.9 4.6 5.5 59.1 12.3 18.4 56.9 13.5 23.1 56.6 19.4 19.4 14.5 14.4 39.6
HyLightRAG 44.5 2.5 3.3 43.9 12.2 17.8 44.4 6.9 9.3 42.4 0.0 0.0 20.5 17.4 41.2
LGraphRAG 52.7 2.5 4.3 48.7 12.4 16.3 59.1 13.0 20.8 48.3 0.0 0.0 15.8 17.2 43.1
GGraphRAG 35.2 0.6 2.0 12.9 2.3 5.7 11.2 0.3 1.9 15.2 0.0 0.0 7.2 13.6 30.5
HippoRAG 66.9 4.4 4.7 66.1 25.8 31.6 68.6 19.4 29.2 61.2 8.3 8.3 15.1 27.6 63.2
RAPTOR 65.2 6.4 6.4 66.9 15.1 25.4 68.1 13.6 21.9 58.7 16.7 16.7 14.3 22.0 52.7
ArchRAG 58.9 4.5 6.5 60.6 21.9 27.6 62.1 16.7 27.8 49.6 33.3 33.3 15.9 26.5 62.1
KET-RAG 29.3 0.5 1.0 25.5 5.3 8.5 14.1 2.1 4.2 34.3 16.7 16.7 3.9 5.2 12.0
HiRAG 49.3 5.8 5.8 54.8 12.0 14.8 43.5 7.6 10.4 68.1 0.0 0.0 33.5 22.1 56.0
5.2 Performance of RAG Methods on ASTRA-QA
We highlight the best score for each metric in each table in red, with higher TF1 and lower
Htopic/Hrespindicating better performance.
Overall results.Table 3 reports T-F1, Htopic, andHrespfor each method under the three retrieval
settings and their overall averages. In general, most methods degrade as the retrieval scope expands,
and stronger topic coverage is often accompanied by higher curated hallucination scores. HippoRAG
achieves the highest overall T-F1, reaching 56.6, but it also has the highest overall Htopic andHresp,
at 20.0 and 38.7. HiRAG obtains a strong overall T-F1 of 49.9 and the best T-F1 under SIMPLEwith
68.9, but its overall Hrespremains high at 32.1. In contrast, KET-RAG yields the lowest overall
Htopic andHresp, at 4.3 and 8.5, while its T-F1 drops to 24.5. These results highlight that ASTRA-QA
separates topic coverage from curated unsupported-topic behavior, rather than treating more verbose
or responses with excessive unsupported content as uniformly better.
Retrieval-scope comparison.Table 3 also separates results by retrieval scope, allowing direct
comparison across SIMPLE, MIDDLE, and HARDfor the same method. LLightRAG drops from 55.0
T-F1 under SIMPLEto 33.9 under HARD, while its Hrespincreases from 20.7 to 37.0. LGraphRAG
also declines from 57.8 to 38.3 T-F1, and KET-RAG decreases from 32.7 to 12.9. Some methods are
more stable in coverage, such as HiLightRAG, from 54.0 to 46.0 T-F1, and HippoRAG, from 61.7 to
51.2 T-F1. However, broader retrieval scopes still increase hallucination risk for several methods,
as shown by RAPTOR’s Hresprising from 31.7 to 45.5. These changes show that ASTRA-QA’s
retrieval settings not only increase corpus size, but also test whether a method can keep the answer
within the designated evidence scope.
Question-type comparison.Table 4 reports the performance of different RAG methods across the
five ASTRA-QA question types. Single-Sum favors methods that summarize a focused source, with
HippoRAG and RAPTOR reaching 66.9 and 65.2 T-F1. On ENUMERATION, HiRAG achieves the
highest T-F1 of 68.1 and several methods obtain 0 Htopic andHresp, because only 4 Enumeration
QA instances contain hallucination sets. TEMPORALis the most difficult question type, with the best
T-F1 reaching 33.5 and Hrespremaining high, peaking at 63.2 for HippoRAG. This difficulty mainly
comes from the much larger document sets in Temporal and the lack of temporal evidence selection
mechanisms in most compared methods. These variations across question types show thatASTRA-QA
tests distinct forms of abstract QA over documents rather than only scaling up retrieval difficulty.
5.3 Comparison with Head-to-Head Evaluation
To compare topic-based evaluation with the commonly used head-to-head evaluation, we randomly
sample 200 questions under the SIMPLEsetting and evaluate both ranking consistency and evaluation
cost. In the head-to-head setting, each cell represents the win rate of the row method against the
column method;single-orderevaluates each ordered pair once, whilebidirectionalalso evaluates the
reversed order and reports the average. We report both directions because head-to-head evaluation
is sensitive to the presentation order of RAG model responses, and the resulting win rates can vary
noticeably when the method order is reversed. The detailed bidirectional results are provided in the
appendix, which further suggests that head-to-head evaluation may yield unstable conclusions even
8

under the same question set. As shown in Figure 5 and Table 5, ASTRA-QA produces rankings that
are consistent with head-to-head results, with RAPTOR performing best and KET-RAG performing
worst. Figure 6 further shows that ASTRA-QA reduces evaluation time from 154,784 s to7,473 s
and token cost by over 21×, because each response is scored independently against fixed topic sets
rather than compared against every other method. Note that we report only the overall results for five
selected methods, and the complete results are provided in the appendix.
RA HR GL HY KR
RA 50 53 74 87 99
HR 47 50 66 76 98
GL 26 34 50 69 95
HY 13 24 31 50 88
KR 1 2 512 50
Figure 5: Head-to-head (bidirec-
tional, overall) win-rate compar-
ison, using the same method ab-
breviations as in Table 5.Method TF1 HT HR
RAPTOR 62.8 9.4 20.0
HippoRAG 61.7 15.4 35.5
LGraphRAG 55.6 7.5 16.3
HyLightRAG 40.8 6.4 13.8
KET-RAG 37.7 1.1 2.1
Table 5: ASTRA-QA evalua-
tion on 200 QA instances. TF1,
HT, and HR denote T-F1 ,Htopic,
andH resp, respectively.Time Tokens00.511154.8K 36.6M
76.0K 18.3M
7.5K 1.7MRelative CostH2H-D H2H-S ASTRA-QA
Figure 6: Normalized time and to-
ken costs relative to Head-to-Head
(bidirectional). Absolute values
are annotated above the bars.
5.4 Dataset Analysis
Benchmark diversity is particularly important for abstract QA, as it exposes systems to a broader
range of linguistic variation and synthesis demands, and therefore provides a more realistic test
of their generalization ability [ 16,37]. Table 6 compares ASTRA-QA with representative RAG
datasets [ 38,39,15,40,16] using three diversity metrics from prior text generation studies [ 41]: Q&A
n-gram diversity (Q&A-NGD), answer cluster entropy (A-CE), and answer embedding homogeneity
score (A-HS), together with average question and answer lengths; further details are provided in the
appendix. Answer-side metrics are unavailable for UltraDomain because it provides only questions
without answers, while A-HS is not applicable to MultiHop-RAG because each question has only
a single target answer. Although ASTRA-QA does not achieve the highest Q&A-NGD, its score
of 2.86 remains close to LiveRAG’s 2.97, indicating that its questions are still lexically diverse
rather than generated from narrow templates. ASTRA-QA also attains the best A-CE and A-HS,
because its topic-set answers are designed to cover more salient aspects of the source documents
and therefore span more diverse answer-side semantic regions. Overall, these results show that
ASTRA-QA provides strong diversity on both the question and answer sides, with particularly rich
semantic coverage and a balanced semantic distribution across its topic-set answers.
Table 6: Comparison of representative RAG datasets and ASTRA-QA. ↑/↓mark higher/lower-is-
better, and the best and second-best results are marked inboldand underlined , respectively.
Dataset Q&A-NGD↑Q-Length A-CE↑A-HS↓A-Length
PopQA [38] 2.37 6.70 3.76 0.45 3.68
MultiHop-RAG [39] 1.99 46.51 2.35 / 1.31
RAG-QA Arena [15] 2.82 9.89 3.83 0.37 80.16
UltraDomain [40] 2.31 12.56 / / /
LiveRAG [16]2.9715.09 3.84 0.36 60.61
ASTRA-QA (Ours)2.86 16.253.85 0.1417.17
6 Conclusion
In this paper, we introduce ASTRA-QA, a benchmark for abstract question answering over documents.
ASTRA-QA provides curated QA instances with answer topic sets, hallucination sets, and grounded
evidence, enabling evaluation beyond short-answer matching. We further present a topic-based
evaluation method that measures topic coverage and curated unsupported-topic matches in a more
interpretable and more scalable way than head-to-head judging. Experiments on representative RAG
baselines show that they still struggle on ASTRA-QA, especially when the retrieval scope expands
from focused evidence to broader and more distracting corpora. These results position ASTRA-QA as
a practical benchmark and evaluation testbed for studying more capable RAG methods with realistic
retrieval conditions for solving the abstract QA task.
9

References
[1]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented
generation for knowledge-intensive nlp tasks.Advances in Neural Information Processing
Systems, 33:9459–9474, 2020.
[2]Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven
Truitt, and Jonathan Larson. From local to global: A graph rag approach to query-focused
summarization.arXiv preprint arXiv:2404.16130, 2024.
[3]Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. Lightrag: Simple and fast
retrieval-augmented generation.arXiv e-prints, pages arXiv–2410, 2024.
[4]Shu Wang, Yixiang Fang, Yingli Zhou, Xilin Liu, and Yuchi Ma. Archrag: Attributed
community-based hierarchical retrieval-augmented generation. InProceedings of the AAAI
Conference on Artificial Intelligence, volume 40, pages 15868–15876, 2026.
[5]Yingli Zhou, Yaodong Su, Youran Sun, Shu Wang, Taotao Wang, Runyuan He, Yongwei Zhang,
Sicong Liang, Xilin Liu, Yuchi Ma, et al. In-depth analysis of graph-based rag in a unified
framework.arXiv preprint arXiv:2503.04338, 2025.
[6]Boyu Chen, Zirui Guo, Zidan Yang, Yuluo Chen, Junze Chen, Zhenghao Liu, Chuan Shi, and
Cheng Yang. Pathrag: Pruning graph-based retrieval augmented generation with relational
paths. InProceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages
30183–30191, 2026.
[7]Haoyu Huang, Yongfeng Huang, Junjie Yang, Zhenyu Pan, Yongqiang Chen, Kaili Ma, Hongzhi
Chen, and James Cheng. Retrieval-augmented generation with hierarchical knowledge.arXiv
preprint arXiv:2503.10150, 2025.
[8]Tianchi Cai, Zhiwen Tan, Xierui Song, Tao Sun, Jiyan Jiang, Yunqi Xu, Yinger Zhang, and Jinjie
Gu. Forag: Factuality-optimized retrieval augmented generation for web-enhanced long-form
question answering. InProceedings of the 30th ACM SIGKDD Conference on Knowledge
Discovery and Data Mining, pages 199–210, 2024.
[9]Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A Smith, and Matt Gardner. A dataset
of information-seeking questions and answers anchored in research papers.arXiv preprint
arXiv:2105.03011, 2021.
[10] Tim Baumgärtner, Ted Briscoe, and Iryna Gurevych. Peerqa: A scientific question answering
dataset from peer reviews. InProceedings of the 2025 Conference of the Nations of the Americas
Chapter of the Association for Computational Linguistics: Human Language Technologies
(Volume 1: Long Papers), pages 508–544, 2025.
[11] Ivan Stelmakh, Yi Luan, Bhuwan Dhingra, and Ming-Wei Chang. Asqa: Factoid questions meet
long-form answers. InProceedings of the 2022 Conference on Empirical Methods in Natural
Language Processing, pages 8273–8288, 2022.
[12] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan Salakhut-
dinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop
question answering.arXiv preprint arXiv:1809.09600, 2018.
[13] Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao
Du, Xiao Liu, Aohan Zeng, Lei Hou, et al. Longbench: A bilingual, multitask benchmark for
long context understanding. InProceedings of the 62nd annual meeting of the association for
computational linguistics (volume 1: Long papers), pages 3119–3137, 2024.
[14] Xiao Yang, Kai Sun, Hao Xin, Yushi Sun, Nikita Bhalla, Xiangsen Chen, Sajal Choudhary,
Rongze D Gui, Ziran W Jiang, Ziyu Jiang, et al. Crag-comprehensive rag benchmark.Advances
in Neural Information Processing Systems, 37:10470–10490, 2024.
10

[15] Rujun Han, Yuhao Zhang, Peng Qi, Yumo Xu, Jenyuan Wang, Lan Liu, William Yang Wang,
Bonan Min, and Vittorio Castelli. Rag-qa arena: Evaluating domain robustness for long-form
retrieval augmented question answering. InProceedings of the 2024 conference on empirical
methods in natural language processing, pages 4354–4374, 2024.
[16] David Carmel, Simone Filice, Guy Horowitz, Yoelle Maarek, Alex Shtoff, Oren Somekh, and
Ran Tavory. Liverag: A diverse q&a dataset with varying difficulty level for rag evaluation.
arXiv preprint arXiv:2511.14531, 2025.
[17] Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. InText summarization
branches out, pages 74–81, 2004.
[18] Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. Bertscore:
Evaluating text generation with bert.arXiv preprint arXiv:1904.09675, 2019.
[19] Thomas Scialom, Paul-Alexis Dray, Sylvain Lamprier, Benjamin Piwowarski, Jacopo Staiano,
Alex Wang, and Patrick Gallinari. Questeval: Summarization asks for fact-based evaluation.
InProceedings of the 2021 conference on empirical methods in natural language processing,
pages 6594–6604, 2021.
[20] Alexander Richard Fabbri, Chien-Sheng Wu, Wenhao Liu, and Caiming Xiong. Qafacteval:
Improved qa-based factual consistency evaluation for summarization. InProceedings of the 2022
Conference of the North American Chapter of the Association for Computational Linguistics:
Human Language Technologies, pages 2587–2601, 2022.
[21] Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Koh, Mohit Iyyer,
Luke Zettlemoyer, and Hannaneh Hajishirzi. Factscore: Fine-grained atomic evaluation of
factual precision in long form text generation. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing, pages 12076–12100, 2023.
[22] Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. G-eval:
Nlg evaluation using gpt-4 with better human alignment. InProceedings of the 2023 conference
on empirical methods in natural language processing, pages 2511–2522, 2023.
[23] Gautier Izacard and Edouard Grave. Leveraging passage retrieval with generative models for
open domain question answering. InProceedings of the 16th conference of the european chapter
of the association for computational linguistics: main volume, pages 874–880, 2021.
[24] Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. Hipporag:
Neurobiologically inspired long-term memory for large language models.arXiv preprint
arXiv:2405.14831, 2024.
[25] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D
Manning. Raptor: Recursive abstractive processing for tree-organized retrieval.arXiv preprint
arXiv:2401.18059, 2024.
[26] Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason Weston, and Michael Auli. Eli5:
Long form question answering. InProceedings of the 57th annual meeting of the association
for computational linguistics, pages 3558–3567, 2019.
[27] Esin Durmus, He He, and Mona Diab. Feqa: A question answering evaluation framework
for faithfulness assessment in abstractive summarization. InProceedings of the 58th annual
meeting of the association for computational linguistics, pages 5055–5070, 2020.
[28] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang,
Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and
chatbot arena.Advances in Neural Information Processing Systems, 36:46595–46623, 2023.
[29] David Soergel, Adam Saunders, and Andrew McCallum. Open scholarship and peer review: a
time for experimentation. InICML 2013 Workshop on Peer Reviewing and Publishing Models,
2013. URLhttps://openreview.net/forum?id=xf0zSBd2iufMg.
11

[30] Daniel A Epstein, Clara Caldeira, Mayara Costa Figueiredo, Xi Lu, Lucas M Silva, Lucretia
Williams, Jong Ho Lee, Qingyang Li, Simran Ahuja, Qiuer Chen, et al. Mapping and taking
stock of the personal informatics literature.Proceedings of the ACM on Interactive, Mobile,
Wearable and Ubiquitous Technologies, 4(4):1–38, 2020.
[31] Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen
Liu, Yuan Qu, Fukai Shang, et al. Mineru: An open-source solution for precise document
content extraction.arXiv preprint arXiv:2409.18839, 2024.
[32] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark,
AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card.arXiv
preprint arXiv:2410.21276, 2024.
[33] Yiqian Huang, Shiqi Zhang, and Xiaokui Xiao. Ket-rag: A cost-efficient multi-granular indexing
framework for graph-rag. InProceedings of the 31st ACM SIGKDD Conference on Knowledge
Discovery and Data Mining V . 2, pages 1003–1012, 2025.
[34] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu,
Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report.arXiv preprint
arXiv:2505.09388, 2025.
[35] Zach Nussbaum, John X. Morris, Brandon Duderstadt, and Andriy Mulyar. Nomic embed:
Training a reproducible long context text embedder, 2024.
[36] OpenAI. Gpt-5.1 instant and gpt-5.1 thinking system card addendum. https://openai.com/
index/gpt-5-system-card-addendum-gpt-5-1/, 2025. Accessed: 2026-04-14.
[37] Rui Han, Xiaoyi Lu, and Jiangtao Xu. On big data benchmarking. InWorkshop on Big Data
Benchmarks, Performance Optimization, and Emerging Hardware, pages 3–18. Springer, 2014.
[38] Ella Rabinovich, Samuel Ackerman, Orna Raz, Eitan Farchi, and Ateret Anaby Tavor. Predicting
question-answering performance of large language models through semantic consistency. In
Proceedings of the Third Workshop on Natural Language Generation, Evaluation, and Metrics
(GEM), pages 138–154, 2023.
[39] Yixuan Tang and Yi Yang. Multihop-rag: Benchmarking retrieval-augmented generation for
multi-hop queries.arXiv preprint arXiv:2401.15391, 2024.
[40] Hongjin Qian, Zheng Liu, Peitian Zhang, Kelong Mao, Defu Lian, Zhicheng Dou, and Tiejun
Huang. Memorag: Boosting long context processing with global memory-enhanced retrieval
augmentation. InProceedings of the ACM on Web Conference 2025, pages 2366–2377, 2025.
[41] Chantal Shaib, Venkata S Govindarajan, Joe Barrow, Jiuding Sun, Alexa F Siu, Byron C Wallace,
and Ani Nenkova. Standardizing the measurement of text diversity: A tool and a comparative
analysis of scores.arXiv preprint arXiv:2403.00553, 2024.
[42] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu,
Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large lan-
guage model serving with pagedattention. InProceedings of the ACM SIGOPS 29th Symposium
on Operating Systems Principles, 2023.
[43] Ollama. Ollama.https://github.com/ollama/ollama, 2024. Accessed: 2026-05-03.
12

A Additional Details of ASTRA-QA
A.1 Examples of ASTRA-QA benchmark
Table 7: Example QA instance of ASTRA-QA benchmark across the five question types.
Question Type Question Answer Topic Set
Single-Sum Please summarize the paper Leveraging
Large Language Models for Multiple
Choice Question Answering.[Multiple Choice Question Answer-
ing (MCQA), Cloze Prompting
(CP),···] (16 topics)
Pair-Comp How do NMT-based and LLM-based ap-
proaches differ in their methodologies
and capabilities for program repair?[Translation Task, Sequence-to-
Sequence, Copy Mechanism, Lim-
ited PL Knowledge, Attention Mech-
anism,···] (17 topics)
Multi-Comp How do single-functional, dual-
functional, and triple-functional signal
designs for ISCC compare in terms of
their approach to signal integration and
system complexity?[Independent Signals, Severe Inter-
ference, High Complexity Beam-
forming, Sensing and AirComp Sig-
nals,···] (10 topics)
Enumeration What mechanisms do gradient-based
jailbreak attacks employ to manipulate
LLMs, and what are the potential secu-
rity consequences of these mechanisms?[GCG, AutoDAN, ARCA, Append
adversarial suffix/prefix to prompts,
···] (7 topics)
Temporal What are the implications of US-China
economic interactions on consumer
costs and national economic resilience?[Increased costs for consumers;
Inflation, Rising prices affecting
households,···] (9 topics)
A.2 Discussion of ASTRA-QA
Broader impacts.ASTRA-QA is intended to support more transparent and reliable evaluation of
document-grounded QA and RAG systems, especially for abstract questions that require synthesis
across long or multiple documents. By providing reference topic sets, hallucination sets, and evidence
metadata, the benchmark may help researchers diagnose missing coverage and unsupported content,
reduce reliance on opaque pairwise preferences, and improve the reproducibility of RAG evaluation.
At the same time, benchmark results should not be interpreted as a complete measure of real-world
system safety or usefulness. Systems that perform well on ASTRA-QA may still propagate biases
or inaccuracies from source documents, fail under deployment-specific constraints, or produce
unsupported content outside the curated hallucination sets. In high-stakes domains such as scientific,
legal, medical, or policy decision support, outputs from RAG systems should therefore remain subject
to domain-expert review and should not be used as a substitute for human judgment.
A.3 Prompts in ASTRA-QA Benchmark
In this part, we report the main prompt templates used during ASTRA-QA construction and evalua-
tion.
We provide prompts for five different question types (SINGLE-SUM, PAIR-COMP, MULTI-COMP,
ENUMERATION, and TEMPORAL) in Figures 11, 12, 13, 14, and 15, respectively.
Additionally, we provide the complete prompt templates employed in ASTRA-QA and the head-to-
head evaluation method, facilitating reproducibility and further analysis in Figure 16, 17, respectively.
B Experimental Details
B.1 Details of Experimental Settings
Compute resources.All experiments were conducted on a Linux operating system running on
a high-performance server equipped with an Intel Xeon 2.0GHz CPU, 1024GB of memory, and 8
NVIDIA GeForce RTX A5000 GPUs, each with 24GB of VRAM. Following the experimental setup
13

in Section 5, all RAG methods use Qwen3-8B as the LLM backbone and nomic-embed-text as the
embedding model. We deploy Qwen3-8B with the vLLM [ 42] framework and serve nomic-embed-
text with Ollama [ 43]. GPT-5.1 is used for evaluation through API calls. We did not separately log
the total wall-clock time and API-token usage for the complete benchmark evaluation. However,
the total evaluation time can be estimated from the validation-subset comparison in Section 5: on
200 questions, our reference-based evaluation required 7,473.98 seconds and 1,687,701 tokens,
compared with 154,784.34 seconds and 36,623,666 tokens for bidirectional head-to-head evaluation,
corresponding to about a 95.2% reduction in evaluation time and a 95.4% reduction in token usage.
B.2 Metrics of Dataset Quality for QA
We evaluate the quality of QA data in each dataset using five metrics, all computed on a fixed random
sample of 500 QA instances per dataset for fair comparison. Let D={(q i, ai)}N
i=1denote the
sampled QA set, where N= 500 . Following LiveRAG [ 16], all embedding-based metrics use
the SentenceTransformer model sentence-transformers/all-MiniLM-L6-v2 . Specifically, we
reportQ&A-NGD,A-HS,Q-Length,A-Length, andA-CE.
•Q&A-NGD:Q&A-NGD (question and answer n-gram diversity) measures lexical diversity
on the concatenated question-answer text. A higher value indicates richer lexical variation.
For each sample, we concatenate the question and answer into xi= [qi;ai]. LetGn(D)be
the multiset of all n-grams extracted from {xi}N
i=1, and let Un(D)be the corresponding set
of uniquen-grams. We define
Q&A-NGD(D) =4X
n=1|Un(D)|
|Gn(D)|.
•A-HS:A-HS (answer embedding homogeneity score) measures within-answer semantic
consistency. For each answer ai, we first split it into a set of answer units {ui,j}mi
j=1. In
practice, if structured claim annotations are available, we use these atomic claims as answer
units; otherwise, for free-form answers, we first split the text by line breaks and then further
segment each block into sentence-like units using sentence-final punctuation. For list-style
answers, each non-empty item is treated as one unit. We then encode each unit as
zi,j=f(u i,j),
where f(·) denotes the shared embedding model introduced above. The homogeneity score
of answera iis defined as the mean pairwise cosine similarity among all unit embeddings:
si=2
mi(mi−1)X
1≤j<k≤m icos(z i,j,zi,k).
For answers with fewer than two units, the score is set to 0in implementation. The final
dataset-level metric is the mean score over all sampled QA pairs:
A-HS(D) =1
NNX
i=1si.
A lower value indicates lower semantic homogeneity within an answer, suggesting that
the answer tends to cover more diverse aspects and broader topical scope, and is therefore
generally more comprehensive.
•Q-Length:Question length is the average number of words per question.
•A-Length:Answer length is the average number of words in the answer.
•A-CE:A-CE (answer cluster entropy) measures the diversity of answer topics in a dataset.
We collect all answer units from the sampled QA pairs, embed them using the same model
f(·), and cluster them independently for each dataset using KMeans with K= 50 . Let
{ut}M
t=1be all answer units in the sampled set, let ct∈ {1, . . . , K} be the cluster assignment
of unitu t, and let
pk=1
MMX
t=11[ct=k]
14

denote the empirical proportion of clusterk. We then compute the Shannon entropy
A-CE(D) =−KX
k=1pklogp k, K= 50.
Here logdenotes the natural logarithm. A higher value indicates broader topical coverage
and a more balanced distribution over answer topics.
B.3 Complete Head-to-Head Results
We report the complete head-to-head results across four evaluation dimensions: Comprehensiveness,
Diversity, Empowerment, and the Overall Winner selected based on these three criteria. Each matrix
entry is a percentage and always reports the win rate of the row method against the column method.
For example, in Figure 7a, the value 92 in the first data column (VR) of the second data row (LL)
indicates that LLightRAG (LL) wins 92% of its pairwise comparisons against Vanilla RAG (VR)
under the Overall Winner criterion in the forward-order setting. The abbreviations VR, LL, LH, HY ,
GG, GL, HR, RA, AR, KR, and HI denote Vanilla RAG, LLightRAG, HiLightRAG, HyLightRAG,
GGraphRAG, LGraphRAG, HippoRAG, RAPTOR, ArchRAG, KET-RAG, and HiRAG, respectively.
Following the head-to-head evaluation prompt in Figure 17, theforward orderplaces the row
method’s output in the Answer 1 position and the column method’s output in the Answer 2 position.
Thereverse orderplaces the column method’s output in the Answer 1 position and the row method’s
output in the Answer 2 position. These two settings can lead to different win rates because the
presentation order may affect the LLM-based head-to-head judgment. For example, for the LL-vs-LH
comparison under the Overall Winner criterion, LL obtains a 33% win rate when LL is shown before
LH in Figure 7a (second row, third column), but a 47% win rate when LH is shown before LL in
Figure 7b (second row, third column).
VR LL LH HY GG GL HR RA AR KR HI
VR 50 8 5 36 41 17 3 1 1 91 14
LL 92 50 33 60 86 44 29 14 23 95 78
LH 95 67 50 60 90 52 37 26 31 98 82
HY 64 40 40 50 60 23 28 11 10 88 44
GG 59 14 10 40 50 21 19 2 1 73 30
GL 83 56 48 77 79 50 35 21 28 95 71
HR 97 71 63 72 81 65 50 40 49 98 96
RA 99 86 74 89 98 79 60 50 54 99 96
AR 99 77 69 90 99 72 51 46 50 99 99
KR 9 5 2 12 27 5 2 1 1 50 1
HI 86 22 18 56 70 29 4 4 1 99 50
(a) Overall Winner, forward orderVR LL LH HY GG GL HR RA AR KR HI
VR 50 11 6 44 49 22 4 1 1 92 19
LL 89 50 47 69 90 55 29 27 30 98 85
LH 94 53 50 71 92 57 41 37 40 98 86
HY 56 31 29 50 61 38 20 14 13 88 49
GG 51 10 8 39 50 20 4 1 2 73 41
GL 78 45 43 62 80 50 33 30 31 95 72
HR 96 71 59 80 96 67 50 54 55 98 6
RA 99 73 63 86 99 70 46 50 57 99 6
AR 99 70 60 87 98 69 45 43 50 99 99
KR 8 2 2 12 27 5 2 1 1 50 2
HI 81 15 14 51 59 28 4 4 1 98 50
(b) Overall Winner, reverse order
Figure 7: Head-to-head win rates for theOverall Winnercriterion. Each entry reports the row
method’s win rate against the column method; higher is better.
B.4 Topic Precision and Recall Results
Table 8 complements the complete topic precision and topic recall. We highlight the best score for
each column in red, with higher T-Prec and T-Rec indicating better performance.
HiRAG performs best under SIMPLE, reaching 77.3 T-Prec and 67.5 T-Rec, and also obtains the high-
est MIDDLET-Prec of 68.7, but its recall drops as the retrieval scope expands. In contrast, HippoRAG
and RAPTOR show stronger recall-oriented behavior under broader retrieval settings, achieving the
best MIDDLEand HARDT-Rec scores of 56.8 and 50.8, respectively, while HiLightRAG obtains
the best HARDT-Prec of 68.2. These results indicate that ASTRA-QA distinguishes conservative
high-precision behavior from broader recall-oriented topic coverage, complementing the results in
the main tables.
15

VR LL LH HY GG GL HR RA AR KR HI
VR 50 9 6 37 41 17 3 2 1 90 14
LL 91 50 36 60 89 45 23 19 23 96 77
LH 94 64 50 62 89 52 30 29 29 97 81
HY 63 40 38 50 61 31 17 13 8 91 45
GG 59 11 11 39 50 19 5 4 1 71 31
GL 83 55 48 69 81 50 29 25 26 96 72
HR 97 77 70 83 95 71 50 47 46 99 94
RA 98 81 71 87 96 75 53 50 50 99 92
AR 99 77 71 92 99 74 54 50 50 99 99
KR 10 4 3 9 29 4 1 1 1 50 1
HI 86 23 19 55 69 28 6 8 1 99 50
(a) Comprehensiveness, forward orderVR LL LH HY GG GL HR RA AR KR HI
VR 50 10 5 38 39 18 3 2 1 87 13
LL 90 50 42 66 89 52 29 26 23 97 83
LH 95 58 50 67 90 53 37 35 32 98 84
HY 62 34 33 50 60 38 19 15 10 88 46
GG 61 11 10 40 50 17 4 3 1 72 35
GL 82 48 47 62 83 50 33 28 23 96 73
HR 97 71 63 81 96 67 50 50 43 98 93
RA 98 74 65 85 97 72 50 50 46 99 92
AR 99 77 68 90 99 77 57 54 50 99 99
KR 13 3 2 12 28 4 2 1 1 50 1
HI 87 17 16 54 65 27 7 8 1 99 50
(b) Comprehensiveness, reverse order
Figure 8: Head-to-head win rates for theComprehensivenesscriterion. Each entry reports the row
method’s win rate against the column method; higher is better.
VR LL LH HY GG GL HR RA AR KR HI
VR 50 8 6 35 38 15 2 2 1 88 10
LL 92 50 36 57 89 42 23 18 10 96 77
LH 94 64 50 59 89 48 27 27 13 96 81
HY 65 43 41 50 62 28 16 13 4 92 44
GG 62 11 11 38 50 19 5 4 1 71 29
GL 85 58 52 72 81 50 27 23 12 96 72
HR 98 77 73 84 95 73 50 43 24 99 93
RA 98 82 73 87 96 77 57 50 24 99 92
AR 99 90 87 96 99 88 76 76 50 99 99
KR 12 4 4 8 29 4 1 1 1 50 1
HI 90 23 19 56 71 28 7 8 1 99 50
(a) Diversity, forward orderVR LL LH HY GG GL HR RA AR KR HI
VR 50 10 6 38 39 18 2 2 1 89 17
LL 90 50 46 66 90 55 34 33 20 96 86
LH 94 54 50 66 90 58 39 41 26 97 86
HY 62 34 34 50 61 41 21 20 9 90 47
GG 61 10 10 39 50 18 6 4 1 72 39
GL 82 45 42 59 82 50 38 34 23 97 72
HR 98 66 61 79 94 62 50 53 39 99 94
RA 98 67 59 80 96 66 47 50 39 99 92
AR 99 80 74 91 99 77 61 61 50 99 99
KR 11 4 3 10 28 3 1 1 1 50 2
HI 83 14 14 53 61 28 6 8 1 98 50
(b) Diversity, reverse order
Figure 9: Head-to-head win rates for theDiversitycriterion. Each entry reports the row method’s
win rate against the column method; higher is better.
VR LL LH HY GG GL HR RA AR KR HI
VR 50 7 5 36 40 17 4 1 1 91 11
LL 93 50 34 58 88 43 21 17 21 96 77
LH 95 66 50 60 88 47 27 27 32 97 78
HY 64 42 40 50 61 26 16 13 10 90 42
GG 60 12 12 39 50 19 5 4 1 71 29
GL 83 57 53 74 81 50 27 23 28 96 72
HR 96 79 73 84 95 73 50 42 48 99 92
RA 99 83 73 87 96 77 58 50 53 99 91
AR 99 79 68 90 99 72 52 47 50 99 99
KR 9 4 3 10 29 4 1 1 1 50 1
HI 89 23 22 58 71 28 8 9 1 99 50
(a) Empowerment, forward orderVR LL LH HY GG GL HR RA AR KR HI
VR 50 10 6 38 39 18 2 2 1 89 17
LL 90 50 46 66 90 55 34 33 31 96 86
LH 94 54 50 66 90 58 39 41 40 97 86
HY 62 34 34 50 61 41 21 20 13 90 47
GG 61 10 10 39 50 18 6 4 2 72 39
GL 82 45 42 59 82 50 38 34 31 97 72
HR 98 66 61 79 94 62 50 53 55 99 94
RA 98 67 59 80 96 66 47 50 58 99 92
AR 99 69 60 87 98 69 45 42 50 99 99
KR 11 4 3 10 28 3 1 1 1 50 2
HI 83 14 14 53 61 28 6 8 1 98 50
(b) Empowerment, reverse order
Figure 10: Head-to-head win rates for theEmpowermentcriterion. Each entry reports the row
method’s win rate against the column method; higher is better.
16

Table 8: Topic precision and recall of representative RAG methods on ASTRA-QA under the three
retrieval settings. T-Prec and T-Rec denote topic precision and topic recall, respectively.
MethodSimple Middle Hard
T-Prec T-Rec T-Prec T-Rec T-Prec T-Rec
Vanilla RAG 55.6 42.7 39.6 19.5 39.9 18.6
LLightRAG 68.7 50.7 53.8 33.7 49.5 30.9
HiLightRAG 66.1 50.4 61.7 46.2 68.2 41.1
HyLightRAG 66.4 34.8 60.1 34.4 52.6 34.1
LGraphRAG 73.3 53.8 59.0 35.2 56.7 34.2
GGraphRAG 40.2 18.2 46.1 16.7 35.5 17.4
HippoRAG 71.1 60.1 65.3 56.8 61.9 50.6
RAPTOR 71.9 63.5 63.6 54.1 64.5 50.8
ArchRAG 62.4 57.9 62.8 47.6 62.4 46.7
KET-RAG 59.5 24.1 51.6 20.4 22.1 10.1
HiRAG 77.3 67.5 68.7 38.0 67.0 27.8
Prompt for generating QA instance of Single-Sum
Please analyze the provided academic paper reviews and extract more than 10 technical terms or key phrases
that best represent the paper’s core contributions, methodological focus, and major technical concerns.
Instructions:
1. Read all review content, including summaries, strengths, and weaknesses.
2. Extract technical terms or key phrases that are explicitly mentioned or strongly implied by the reviews.
3. First identify the main topics mentioned in each individual review, then aggregate them across all reviews.
4. Prioritize compound technical phrases over single words.
5. Exclude:
− general academic verbs or adjectives
− generic praise or criticism
− explanations, commentary, or full sentences
Output Requirements:
− Output only technical terms or key phrases
− Do not include any introduction or explanation
− Use a comma−separated list
− Do not use markdown
− Do not use numbering
Output Format:
[term 1, term 2, term 3, term 4, term 5, term 6, term 7,...]
Input:
{reviews}
Figure 11: Prompt for generating Single-Sum QA instance.
17

Prompt for generating QA instance of Pair-Comp
You are a professional academic paper analysis expert. Please generate several sets of high−quality comparison
questions based on the content of one or several provided papers. Each set of Q&A must be strictly
based on the text content, without introducing external knowledge.
Input:
− Paper Content: {paper_content}
− Paper Title: {paper_title}
Task requirements are as follows:
Identify two different methods / techniques / models / methodologies. In the provided text, find two methods,
techniques, systems, models, or research paths that are explicitly described or contrasted (e.g., "
Government Agencies vs. Industry and AGI Labs" or "RAP vs. MCTS"). Then generate a single, specific,
and directly comparative question. The question should focus on one of the following aspects:
performance differences, efficiency comparison, applicability scenarios, limitations or risks, basic
principles or design philosophy, or governance roles / functional divisions. The question must be
expressed in a clear interrogative sentence and should encourage the revelation of differences, trade−offs,
or similarities.
Provide a concise keyword−style answer. The answer should be a structured list of 3−10 keywords, formatted
as:
[A: keyword1, keyword2, ...] [B: keyword1, keyword2, ...]
Keywords should be concise and accurate, reflecting the core functions, responsibilities, mechanisms, or
characteristics in the text. Avoid using complete sentences.
Provide supporting evidence paragraphs. The evidence field needs to contain all relevant sentences from the
original text used to support the Q&A, retaining all citation marks for subsequent supplementation of
reference content. All questions and answers must closely follow the text content, highlight technical
comparisons, and avoid generalization or vague statements.
Example output format (JSON list):
[
{
"question": "...",
"answer": "[A: ...],[B: ...]",
"evidence": "..."
}
]
notes:
− All information must originate from the provided text.
− Keywords should be short and powerful, reflecting core concepts.
− Do not generate repetitive or semantically similar questions.
− The complete content should be placed inside; do not summarize or rewrite the evidence sentences, they
must be the original text.
− Output in JSON format.
Figure 12: Prompt for generating QA instance of Pair-Comp.
18

Prompt for generating QA instance of Multi-Comp
You are a professional academic paper analysis expert. Please generate several sets of high−quality comparison
questions based on the content of one or several provided papers. Each set of Q&A must be strictly
based on the text content, without introducing external knowledge.
Input:
− Paper Content: {paper_content}
− Paper Title: {paper_title}
Task requirements are as follows:
Identify multiple methods / techniques / systems / models / research paths that are explicitly described or
contrasted in the provided text. There must be at least three distinct objects. Then generate a single,
specific, and directly comparative question. The question should focus on one of the following aspects:
performance differences, efficiency comparison, applicability scenarios, limitations or risks, basic
principles or design philosophy, or governance roles / functional divisions. The question must be a clear
interrogative sentence that reveals differences, trade−offs, or similarities.
Provide a concise keyword−style answer. The answer should be a structured list of 3−10 keywords for each
object, formatted as:
[A: keyword1, keyword2, ...] [B: keyword1, keyword2, ...] [C: keyword1, keyword2, ...]
Keywords should be concise and accurate, reflecting the core functions, responsibilities, mechanisms, or
characteristics in the text. Avoid using complete sentences.
Provide supporting evidence paragraphs. The evidence field needs to contain all relevant sentences from the
original text used to support the Q&A, retaining all citation marks for subsequent supplementation of
reference content. All questions and answers must closely follow the text content, highlight technical
comparisons, and avoid generalization or vague statements.
Example output format (JSON list):
[
{
"question": "...",
"answer": "[A: ...],[B: ...],[C: ...]",
"evidence": "..."
}
]
notes:
− All information must originate from the provided text.
− Keywords should be short and powerful, reflecting core concepts.
− Do not generate repetitive or semantically similar questions.
− The complete content should be placed inside; do not summarize or rewrite the evidence sentences, they
must be the original text.
− Output in JSON format.
Figure 13: Prompt for generating Multi-Comp QA instance.
19

Prompt for generating QA instance of Enum
You are an expert in analyzing academic survey papers. Your task is to carefully analyze the sections of the
provided review article, extract multiple distinct and specific discussion topics / concepts / methods /
techniques / key findings, and generate an independent and detailed Q&A pair for each one.
Input: Paper Content: {paper_content}; Paper Title: {paper_title}
Reading and identification: carefully read the input text and identify multiple clear, independent, and specific
discussion points. For each identified point, generate one Q&A pair while strictly following the format
and requirements below.
Question: the question must be a clear, specific, and independent interrogative sentence. It should directly
target the particular topic, concept, method, technique, or finding you identified. Example: What are the
key characteristics of method X discussed in the section? Or How does technique Y address challenge Z?
Provide a concise keyword−style answer. The answer must be a single list enclosed in square brackets [] and
contain more than 5 concise phrases or short expressions that accurately summarize the core information
of that topic. The required format is:
[core point 1, core point 2, core point 3, ...]
Keywords should be concise and accurate, reflecting the core functions, responsibilities, mechanisms, or
characteristics described in the text. Avoid using complete sentences.
Provide supporting evidence paragraphs including references. The evidence field must contain all relevant
original sentences from the text that support the Q&A pair, while preserving all citation markers for
subsequent completion of the reference content. Ensure technical accuracy and content relevance. All
questions and answers must remain closely tied to the source text, emphasize technical specificity, and
avoid vague or overly general statements.
Example output format (JSON list):
"example_output": [
{
"question": "...?",
"answer": "[..., ..., ..., ..., ...]",
"evidence": "..."
}
],
notes:
− All information must originate from the provided text.
− Keywords should be short and powerful, reflecting core concepts.
− Do not generate repetitive or semantically similar questions.
− The complete content should be placed inside; do not summarize or rewrite the evidence sentences, they
must be the original text.
− Output in JSON format.
Figure 14: Prompt for generating QA instance of Enum.
20

Prompt for generating QA instance of Temporal
You are a News Synthesis Assistant. Here is the list of news articles:
{news}
Task:
1. Scan the list of news articles above.
2. Identify a single, specific, and concrete topic or subject that is mentioned or discussed across multiple
articles. This should be a clear focal point, like ’Retrieval−Augmented Generation (RAG) techniques’, ’
Apple’s upcoming product features’, ’Impacts of a specific new policy’, or ’Performance of a particular
company’s recent quarter’. Avoid overly broad topics like ’Technology’ or ’World News’.
3. Formulate a question that asks for a list or enumeration of key points, methods, features, impacts, or other
relevant details specifically related to the chosen topic. The question should prompt an answer that lists
and briefly describes different aspects or examples. An example question format is: ’What are the key
features of the new iPhone as reported by various sources?’ or ’What are the main challenges faced by
companies adapting to new tariff structures?’.
4. Provide the answer to your question as a list of concise keywords or short phrases, summarizing the core
information from the selected articles related to the topic. Format the answer strictly as a list like this: [
Point 1, brief description; Point 2, brief description; Another relevant detail]. Do not include any
explanatory text or markdown in the answer.
5. Briefly explain in 1−2 sentences why you chose this specific topic and how the selected articles contribute
information to answer your question. This is the ’reason’.
6. List the titles of the news articles that are relevant to the chosen topic and used to formulate your question
and answer, separated by semicolons.
Please respond in JSON format with the following structure:
{
"question": "<Your generated question asking for a list related to the single topic>",
"answer": "[<Point 1, brief description; Point 2, brief description; ...>]",
"reason": "<Your reason for choosing this topic and how the articles contribute>",
"titles": "<Title of Relevant News 1; Title of Relevant News 2; ...>"
}
Remember: Only output the final JSON object. Do not include any other text, thoughts, or markdown code
blocks in your response.
Figure 15: Prompt for generating Temporal type.
Prompt for Evaluation in ASTRA-QA
You are an expert tasked with extracting topic lists from response of the question.
Task: Read the response and return the complete predicted topics list.
Topic normalization: When you extract a topic, check whether it is semantically equivalent to an existing topic
in the Common Errors List or Ground_truth list.
− If it matches a topic in the Common Errors List or Ground_truth List, use the Common Errors List or
Ground_truth List wording.
− Otherwise, write a concise topic label that accurately reflects the response.
Requirements:
− Extract all distinct topics supported by the response.
− Extract all the topics contained within the response.
− Use semantic matching, not literal matching.
− Keep each topic concise and complete.
− Merge duplicates and near−duplicates.
− The purpose of extraction is to analyze the coverage of the topic in the response. Therefore, you must be very
strict and cannot simply assume that the meanings are the same.
− Return only JSON.
Output format: { "predicted_topics": ["topic1", "topic2", "topic3"] }
Figure 16: Prompt for ASTRA-QA evaluation method.
21

Prompt for Evaluation-Head to Head
You are an expert tasked with evaluating two answers to the same question based on three criteria: **
Comprehensiveness **,**Diversity **, and **Empowerment **.
−**Comprehensiveness **: How much detail does the answer provide to cover all aspects and details of the
question?
−**Diversity **: How varied and rich is the answer in providing different perspectives and insights on the
question?
−**Empowerment **: How well does the answer help the reader understand and make informed judgments
about the topic?
For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an
overall winner based on these three categories.
Here is the question:
{query}
Here are the two answers:
**Answer 1: **{answer1}
**Answer 2: **{answer2}
Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.
Output your evaluation in the following JSON format:
{{
"Comprehensiveness": {{
"Winner": "[Answer 1 or Answer 2]",
"Explanation": "[Provide one sentence explanation here]"
}},
"Diversity": {{
"Winner": "[Answer 1 or Answer 2]",
"Explanation": "[Provide one sentence explanation here]"
}},
"Empowerment": {{
"Winner": "[Answer 1 or Answer 2]",
"Explanation": "[Provide one sentence explanation here]"
}},
"Overall Winner": {{
"Winner": "[Answer 1 or Answer 2]",
"Explanation": "[Briefly summarize why this answer is the overall winner based on the three criteria]"
}}
}}
Figure 17: Prompt for head-to-head evaluation.
22