# MemoryDocDataSet: A Benchmark for Joint Conversational Memory and Long Document Reasoning

**Authors**: Qiyang Xie, Jialun Wu, Xinjie He, Su Liu, Shuai Xiao, Zhiyuan Lin, Weikai Zhou

**Published**: 2026-06-03 04:44:50

**PDF URL**: [https://arxiv.org/pdf/2606.04442v1](https://arxiv.org/pdf/2606.04442v1)

## Abstract
AI systems increasingly need to combine two demanding capabilities: navigating multi-session conversation history and performing deep reading comprehension within long documents. Yet no existing benchmark evaluates both simultaneously. We introduce MemoryDocDataSet, a synthetic benchmark of 50 micro-worlds and 1,000 QA pairs in which each instance comprises 3-5 personas, a temporal event graph spanning months of activity, 3-5 real long documents (20,000-50,000 tokens each sourced from the Caselaw Access Project), multi-session conversations grounded on those documents, and 20 question-answer pairs across five reasoning categories. The defining feature is the Hybrid source tag: questions requiring a system to first navigate conversation history to identify which document is relevant, then extract the answer from within that document. Hybrid questions account for 75.1% of the dataset. Dataset quality is characterised through a prompt-sensitivity self-consistency analysis using LLM-as-judge, yielding a median Cohen's $κ= 0.634$ across all 50 micro-worlds. We evaluate six baseline configurations spanning truncated context, long-context LLMs, retrieval-augmented generation (RAG), and memory systems. The best baseline (RAG-Both) achieves 0.358 overall F1 and 0.342 on Hybrid. Document-only retrieval (RAG-Doc) collapses to 0.267 on Hybrid despite achieving 0.453 on Doc-only questions, demonstrating a clear joint-retrieval gap that motivates architectures unifying conversational memory with long-document navigation. We release the dataset, generation pipeline, and all baseline implementations.

## Full Text


<!-- PDF content starts -->

MemoryDocDataSet: A Benchmark for Joint Conversational
Memory and Long Document Reasoning
Qiyang Xie1Jialun Wu2Xinjie He3Su Liu4
Shuai Xiao4Zhiyuan Lin4Weikai Zhou4
1Northeastern University2Johns Hopkins University
3Columbia University4Independent Researcher
June 2026
Abstract
AI systems increasingly need to combine two demanding capabilities: navigating multi-session con-
versation history and performing deep reading comprehension within long documents. Yet no existing
benchmark evaluates both simultaneously. We introduceMemoryDocDataSet, a synthetic benchmark of 50
micro-worlds and 1,000 QA pairs in which each instance comprises 3–5 personas, a temporal event graph
spanning months of activity, 3–5 real long documents (20,000–50,000 tokens each sourced from the Caselaw
Access Project), multi-session conversations grounded on those documents, and 20 question-answer pairs
across five reasoning categories. The defining feature is theHybridsource tag: questions requiring a system to
first navigate conversation history to identify which document is relevant, then extract the answer from within
that document. Hybrid questions account for 75.1% of the dataset. Dataset quality is characterised through
a prompt-sensitivity self-consistency analysis using LLM-as-judge, yielding a median Cohen’s κ= 0.634
across all 50 micro-worlds. We evaluate six baseline configurations spanning truncated context, long-context
LLMs, retrieval-augmented generation (RAG) [ 1], and memory systems. The best baseline (RAG-Both)
achieves 0.358 overall F1 and 0.342 on Hybrid. Document-only retrieval (RAG-Doc) collapses to 0.267
on Hybrid despite achieving 0.453 on Doc-only questions, demonstrating a clear joint-retrieval gap that
motivates architectures unifying conversational memory with long-document navigation. We release the
dataset, generation pipeline, and all baseline implementations.
1 Introduction
Modern AI assistants operate in settings that demand two capabilities simultaneously. First, they must maintain
coherent memory across long conversations—tracking who said what, when, and how prior context informs
the current turn. Second, they must perform deep reading comprehension within long documents—contracts,
reports, filings—that far exceed the size of any conversational exchange.
These two capabilities have been studied largely in isolation. Memory benchmarks such as LoCoMo [ 2] and
LongMemEval [ 3] present systems with multi-session dialogues but attach no long documents; a system that
memorizes conversation facts perfectly can score highly. Long-document benchmarks such as L-Eval [ 4] and
ZeroSCROLLS [ 5] present systems with book-length or report-length texts but no conversation structure; a
system with a large context window can score highly without any notion of memory.
Neither family of benchmarks measures the combination. Yet real-world deployment routinely requires it.
Consider a legal assistant working with a client over several months: it must recall from prior sessions which
contract the client is asking about, then navigate 40,000 tokens of that contract to answer a specific question
about a penalty clause. This task is trivial for neither a pure memory system nor a pure long-context reader.
We introduceMemoryDocDataSet, a benchmark dataset designed precisely to test this joint capability. Each
1arXiv:2606.04442v1  [cs.CL]  3 Jun 2026

instance is amicro-world—a self-contained scenario comprising 3–5personaswith defined roles, expertise,
and relationships; atemporal event graphof 5–10 time-stamped events spanning at least six months; 3–5long
documentsof 20,000–50,000 tokens each sourced from real public domain legal corpora; fivemulti-session
conversationsgrounded on the event graph in which speakers naturally reference the attached documents; and
20 QA pairsper micro-world spanning five reasoning categories.
The key novelty is thesource dimension—every QA pair is tagged with whether its answer lies in the conversation
history only (Chat-only), in a long document only (Doc-only), or requires bridging both (Hybrid). Hybrid
questions require a system to first use conversation context to identify the relevant document, then read that
document deeply to extract the answer. We require at least 30% of QA pairs per micro-world to be Hybrid.
We evaluate six baseline configurations against our benchmark: a truncated-context base LLM, a long-context
LLM with full context, RAG over conversations only, RAG over documents only, RAG over both, and a memory
system baseline. Results show that all six configurations underperform on Hybrid questions, with RAG over
both achieving the highest F1 but still far below expected human performance (discussed in Section 6.4). This
gap motivates new systems that tightly integrate conversational memory with long-document retrieval.
Contributions.
1.A benchmark dataset with 50 micro-worlds, 1,000 QA pairs, and a novel source-dimension annotation that
no existing benchmark provides.
2.A fully automated, config-driven pipeline for generating micro-worlds at scale using real long documents
and LLM-generated synthetic structure.
3.Baseline results across six retrieval and memory configurations, establishing the first systematic comparison
on joint conversational memory and long-document reasoning.
2 Related Work
2.1 Conversational Memory Benchmarks
Several benchmarks evaluate long-term conversational memory. The Beyond Goldfish Memory paper [ 6], which
introduced the Multi-Session Chat (MSC) dataset, pioneered multi-session dialogue evaluation but focuses on
persona consistency over short ( ∼1K token) exchanges. LoCoMo [ 2] extends this to longer conversations (up to
300 turns, ∼9K tokens) and introduces temporal reasoning questions. LongMemEval [ 3] evaluates LLMs as
conversation partners across multiple sessions, with questions probing what was said in prior turns. MemBench
[7] provides a structured suite of memory operations including storage, retrieval, and update.
A common limitation across all these benchmarks is the absence of long documents. The context is entirely
conversational; systems that memorize conversation facts well can achieve high scores without any document
reading capability.
2.2 Long-Document and Multi-hop Benchmarks
A complementary line of work evaluates reading comprehension over long or multi-document inputs. L-Eval
[4] and ZeroSCROLLS [ 5] require models to process book-length or multi-document texts, but present these as
static reading tasks with no conversational component. HotpotQA [ 8] targets multi-hop reasoning across short
Wikipedia passages; it tests inference chains rather than long-document reading and carries no conversational
structure.
2

Table 1: Comparison of MemoryDocDataSet with related benchmarks.
Benchmark Multi-sess. Conv. Long Docs (20K+) Joint Reas. Venue
LoCoMo [2]✓× ×ACL 2024
LongMemEval [3]✓× ×ICLR 2025
MSC [6]✓× ×ACL 2022
MemBench [7]✓× ×ACL 2025
L-Eval [4]×✓×ACL 2024
ZeroSCROLLS [5]×✓×EMNLP 2023
HotpotQA [8]× × ×EMNLP 2018
Ours (MemoryDocDataSet)✓ ✓ ✓—
2.3 The Gap
Table 1 summarizes the key dimensions across related benchmarks. No existing benchmark combines multi-
session conversations with long documents (20K+ tokens) and requires joint reasoning across both. Memory-
DocDataSet is the first to impose this requirement through the Hybrid source tag.
2.4 Retrieval-Augmented Generation and Memory Systems
Retrieval-Augmented Generation (RAG) [ 1] has emerged as a dominant paradigm for grounding LLM responses
in external documents. However, standard RAG operates over a static document collection indexed at query time,
without modeling the temporal structure of conversations or the navigational relationship between conversation
context and specific documents. Memory systems such as Mem0 [ 9] and Zep [ 10] augment LLMs with graph-
based or fact-based memory stores that track entities and relationships across sessions, but are not designed for
deep reading within long documents. Our benchmark exposes both gaps.
3 The MemoryDocDataSet Benchmark
3.1 Task Definition
We define the task as follows. A system is given a micro-world Mconsisting of a set of conversation sessions
Cand a set of long documents D. Given a natural language question q, the system must produce a free-text
answer a. Questions may require evidence from Calone, from Dalone, or from both—and the system receives
no signal at test time about which source is required. Evaluation is performed using Exact Match (EM) and
token-level F1 against gold answers, with a separate abstention accuracy metric for adversarial questions.
This formulation is intentionally representation-agnostic: the system may represent CandDas a flat context, a
vector index, a knowledge graph, or any other structure. The benchmark does not prescribe a retrieval or memory
architecture.
3.2 Micro-World Structure
The fundamental unit of the dataset is themicro-world—a self-contained scenario that provides all context
needed to answer its associated questions. Figure 1 illustrates the structure. Formally, a micro-world is a tuple
M= (P, G, D, C, Q).
P(Personas). Pis a set ofpersonas, each described by a name, professional role, domain expertise, commu-
nication style, and a set of directed relationships to other personas (e.g.,reports to,collaborates with). Each
micro-world contains 3–5 personas.
3

...
Long Documents
(Caselaw Access Project)Source tags: Chat-only | Doc-only | Hybrid (75.1%)Doc 1
20K 50K tokens
Doc 2
20K 50K tokens
Doc 3
20K 50K tokens
Personas
35 per world
Event Graph
510 events (DAG)
 6-month span
Conversations
5 sessions
per worldQA Pairs
20 per world
5 categoriesFigure 1: Structure of a MemoryDocDataSet micro-world. Each micro-world comprises personas, a temporal
event graph, long documents, multi-session conversations, and QA pairs annotated with source tags (Chat-only,
Doc-only, Hybrid).
G(Event Graph). Gis atemporal event graph—a directed acyclic graph (DAG) over a set of time-stamped
events. Each evente∈Ghas a timestamp, a natural language description, a subset of personas involved, and a
subset of documents referenced. Events span a minimum of six months of simulated activity. The DAG structure
encodes causal dependencies between events.
D(Documents). Dis a set oflong documents, each containing 20,000–50,000 tokens of real legal text sourced
from the Caselaw Access Project [ 11]. Each micro-world contains 3–5 documents. Documents are shared across
personas and events within a micro-world, grounding the scenario in a consistent body of written evidence.
C(Conversations). Cis a set ofconversation sessions. Each session s∈C is anchored to a specific event
e∈G , involves a subset of personas from P, and consists of a sequence of utterances. Sessions are ordered
chronologically and collectively span the timeline of G. At least 40% of sessions explicitly reference one or
more documents in D, establishing the navigational relationship between conversation context and document
content that defines Hybrid questions.
Q(QA Pairs). Qis a set ofQA pairs. Each q∈Q consists of a natural language question, a gold free-text
answer, a category label (Section 3.3), a source tag (Section 3.4), and one or more evidence references pointing
to the specific utterance or document passage that supports the answer. Each micro-world contains 20 QA pairs.
3.3 Question Categories
We adopt five question categories following the taxonomy established in LoCoMo [ 2], which has become the
standard for memory benchmark evaluation. Table 2 defines each category with an example drawn from the
legal domain.
These five categories are not mutually exclusive in general, but in our annotation each QA pair is assigned exactly
one primary category. For Adversarial questions, we additionally record the adversarial type (false premiseor
unanswerable) and evaluate systems separately on abstention accuracy—the rate at which a system correctly
declines to answer rather than producing a hallucinated response.
4

Table 2: QA categories, definitions, and examples.
Category Definition Example
Single-hop A single fact retrievable from one
source without inference“What was the penalty clause amount in the
May 15th contract?”
Multi-hop Requires chaining two or more facts
across sessions or documents“Did the delivery deadline in the contract
match what was agreed in the March meet-
ing?”
Temporal Requires reasoning about the relative
or absolute ordering of events“Was the settlement offer made before or after
the expert witness deposition?”
Knowledge Update An earlier fact has been superseded
by a later one; the correct answer is
the most recent“What is the current delivery deadline?”
(original: June 1; amended: July 15)
Adversarial The question contains a false premise
or is unanswerable from the provided
context“What did the NDA signed on May 20th say
about non-compete?”(no NDA signed on
May 20th)
3.4 The Source Dimension
The defining contribution of MemoryDocDataSet is thesource dimension: an orthogonal annotation on every
QA pair that identifies which information sources are required to answer the question. Each QA pair carries
exactly one of three mutually exclusive source tags.
Chat-only.The gold answer is fully derivable from the conversation sessions C; no document in Dneeds
to be read. This tag tests standard conversational memory and is comparable to questions in LoCoMo [ 2] and
LongMemEval [3].
Doc-only.The gold answer requires reading one or more documents in D; the conversation sessions provide
no additional signal. This tag tests long-document comprehension and is comparable to questions in L-Eval [ 4]
and ZeroSCROLLS [5].
Hybrid.Answering requires two steps: first, the system must use the conversation sessions Cto identifywhich
document in Dis relevant—typically because a persona references a specific document during a session; second,
it must read that document to extract the answer. Neither step alone is sufficient. This tag has no equivalent in
any prior benchmark.
We enforce that at least 30% of QA pairs per micro-world carry the Hybrid tag. Critically, every Hybrid QA pair
is verified to satisfy two structural conditions: (1) at least one session inCcontains an explicit reference to the
document required to answer q, and (2) the gold answer cannot be derived from Calone, requiring the system to
actually read the referenced document.
3.5 Dataset Statistics
MemoryDocDataSet v1.0 comprises50 micro-worldsin the legal domain (US caselaw sourced from the
Caselaw Access Project), split 70/14/16 into train, validation, and test sets at the micro-world level (35/7/8
worlds respectively; the slight asymmetry reflects integer division of 50 whole micro-worlds). Table 3 summarizes
the measured structural properties of the released dataset, and Table 4 shows the QA pair distribution.
QA pairs are distributed uniformly across the five question categories (20% each) by construction. The Hybrid
source tag accounts for 75.1% of all pairs, substantially exceeding the 30% design floor, reflecting the fact that
5

Table 3: MemoryDocDataSet v1.0 structural properties. Documents are sourced from the Caselaw Access
Project [11].
Property Value
Total micro-worlds 50
Train / Val. / Test worlds 35 / 7 / 8
Train / Val. / Test QA pairs 700 / 140 / 160
Total QA pairs 1,000
QA pairs per micro-world 20 (uniform)
Total documents (pool) 790
Avg. documents per micro-world 4.0 (range 3–5)
Document length 20,000–50,000 tokens
Total conversation sessions 250
Avg. sessions per micro-world 5.0
Avg. utterances per session 10.6 (range 6–16)
Avg. personas per micro-world 3.9 (range 3–5)
Document source corpus Caselaw Access Project
Table 4: QA pair distribution by category, source tag, and difficulty.
Dimension Label Count %
CategorySingle-hop 200 20.0%
Multi-hop 200 20.0%
Temporal 200 20.0%
Know. Update 200 20.0%
Adversarial 200 20.0%
Source tagHybrid 751 75.1%
Chat-only 136 13.6%
Doc-only 113 11.3%
DifficultyMedium 389 38.9%
Hard 321 32.1%
Easy 290 29.0%
most questions in a scenario grounded on real documents naturally require bridging conversation and document
evidence. The test split (160 pairs, 8 worlds) is the evaluation set used in all baseline experiments reported in
Section 5.
3.6 Quality Analysis
We conduct a full-coverage automated quality analysis using an LLM-as-judge protocol applied to all 50 micro-
worlds. Two judge instances are instantiated from the same model (Claude Sonnet 4.6) with contrasting system
prompts: astrict reviewerinstructed to flag any QA pair whose gold answer is not unambiguously supported by
the cited evidence, and alenient reviewerinstructed to accept answers that are consistent with the evidence
under any reasonable reading. Each judge independently scores every QA pair in a world as correct or incorrect,
producing two binary label vectors per world over which Cohen’sκis computed.
This design measuresprompt-sensitivity self-consistency: the degree to which the pipeline’s quality signal is
stable across different interpretive stances toward the same evidence. It is not equivalent to human inter-annotator
agreement—the two judges share a model family with the generation pipeline, which may introduce leniency
bias—but it provides a reproducible, scalable proxy for labeling consistency.
6

200 docs
20K 50K tok
50 micro-
worlds35 sessions
per world1,000
QA pairs = 0.634
(median)Train/Val/
Test splitsF1 & EM
per baselineOutput:Stage 1
Document
CollectionStage 2
Persona &
Event GraphStage 3
Conversation
GenerationStage 4
QA
GenerationStage 5
Quality
VerificationStage 6
Dataset
PackagingStage 7
Baseline
EvaluationFigure 2: Seven-stage generation and evaluation pipeline. Each stage is independently reproducible; the pipeline
supports checkpoint/resume for long runs.
Across all 50 micro-worlds, the median prompt-sensitivity κis0.634(mean 0.619, range 0.000–1.000). 22
worlds (44%) achieve κ≥0.70 (substantial agreement), 29 worlds (58%) achieve κ≥0.60 , and 41 worlds
(82%) achieve κ≥0.40 . All 50 worlds are retained in the released dataset; the per-world κvalues are published
alongside the data to allow downstream users to apply their own quality filters. Worlds with low κpredominantly
arise from a systematic base-rate gap between the two prompts—the strict judge accepts roughly 40% of QA
pairs per world while the lenient judge accepts roughly 55%—rather than from disagreement about specific pairs.
This base-rate effect mathematically caps achievable κfor worlds where the gap is large, independent of actual
QA correctness. We discuss this limitation further in Section 6.4.
4 Collection Pipeline
MemoryDocDataSet is generated by an automated seven-stage pipeline that combines real document sourcing
with LLM-driven synthetic structure generation. The pipeline is fully reproducible, config-driven, and supports
checkpoint/resume to handle long runs.
4.1 Overview
The seven stages are: (1) Document Collection, (2) Persona & Event Graph Generation, (3) Conversation
Generation, (4) QA Generation, (5) Quality Verification, (6) Dataset Packaging, and (7) Baseline Evaluation.
All generation calls in Stages 1–5 use Claude Sonnet 4.6 via the Anthropic API. All LLM calls are routed
through a unified client supporting multiple providers (Groq, Ollama, Anthropic, OpenAI), making the pipeline
provider-agnostic. Figure 2 provides an overview of the full pipeline with per-stage outputs.
4.2 Stage 1: Document Collection
Long documents are sourced from theCaselaw Access Project[ 11], a Harvard Law School initiative that has
digitized 6.9 million US court opinions spanning all jurisdictions from the founding era to approximately 2020.
Individual case opinions typically range from 5,000 to 20,000 characters. We bundle full reporter volumes into
single documents, reaching our 20,000–50,000 token target. Documents are validated using the cl100k_base
tokenizer and must carry a permissive license. The current corpus covers 10 US jurisdictions with planned
expansion to all 50 states.
7

4.3 Stage 2: Persona and Event Graph Generation
For each micro-world, an LLM generates a set of 3–5 personas and a temporal event graph grounded on the
collected documents. The event graph is a DAG of 5–10 time-stamped events, each referencing one or more
personas and one or more documents, spanning a minimum of six months of simulated activity. Event graphs are
validated for structural correctness (no cycles, all referenced entities exist) before advancing.
4.4 Stage 3: Conversation Generation
An LLM generates 5 dialogue sessions for each micro-world, grounded on the event graph. Each session is
anchored to a specific event and involves a subset of the micro-world’s personas. A grounding ratio of at least
40% of sessions must reference at least one document, enforcing that conversations are not purely self-contained.
4.5 Stage 4: QA Generation
QA pairs are generated across five categories adapted from the LoCoMo taxonomy [ 2]: single-hop, multi-hop,
temporal, knowledge update, and adversarial. Each QA pair is additionally annotated with a source tag; at least
30% of QA pairs per micro-world must be Hybrid. Gold answers include evidence references pointing to the
specific conversation turn or document passage that supports the answer.
4.6 Stage 5: Quality Verification
All 50 micro-worlds are assessed using the LLM-as-judge protocol described in Section 3.6. The per-world
κvalues are recorded and published with the dataset release. All worlds are retained regardless of κvalue
(agreement_threshold: 0.0 );κis reported as a prompt-sensitivity quality characterisation rather than
a hard filter (see Section 3.6 for rationale).
4.7 Stage 6: Dataset Packaging
Verified micro-worlds are split 70/14/16 into train, validation, and test sets at the micro-world level. Splitting
is performed with document isolation: no document appears in more than one split, preventing leakage. The
dataset is serialized to JSON with accompanying statistics and a Datasheets for Datasets [ 12] documentation file.
5 Baseline Experiments
We evaluate six baseline configurations that span the design space of current approaches to conversational
memory and long-document retrieval. These baselines are intended to establish lower and upper bounds on the
benchmark and to characterize where each class of approach succeeds and fails—not to achieve state-of-the-art
performance.
5.1 Experimental Setup
All baselines useClaude Sonnet 4.5for answer generation. The dataset itself was generated with Claude Sonnet
4.6, keeping generation and evaluation models distinct to avoid self-evaluation bias. Documents are chunked into
512-token windows with 64-token overlap. Chunk embeddings are computed with the all-MiniLM-L6-v2
sentence encoder [ 13] and indexed in a persistent ChromaDB [ 14] vector store. Retrieval uses cosine similarity
withk=10 chunks; this value was not tuned on the validation split and hyperparameter sensitivity is left to
future work. All experiments are conducted on the test split only (160 QA pairs, 8 micro-worlds); the train and
validation splits are reserved for system development.
8

Table 5: Baseline configurations.
Baseline Context Source Context Size What It Tests
Base LLM Conv. + documents (truncated) 4,096 tokens Lower bound: parametric knowledge + short context
Long-Context LLM Conv. + documents (full) 60,000 tokens Long-context capacity with no retrieval
RAG-Conv Top-kconversation chunksk=10chunks Retrieval over conversation history only
RAG-Doc Top-kdocument chunksk=10chunks Retrieval over long documents only
RAG-Both Top-kchunks from bothk=10chunks Joint retrieval over both modalities
Memory System Mem0-style extracted facts Top-kfacts Graph-based memory augmentation
5.2 Baselines
We define six baseline configurations, summarized in Table 5.
Base LLMconcatenates all conversation sessions and document text for the relevant micro-world and truncates
to a 4,096-token context window. This serves as a lower bound, since it can only use whatever fits in the window.
Long-Context LLMprovides the full concatenation of conversation sessions and document text up to a
60,000-token context limit. It tests whether a long-context model can solve the benchmark purely through
extended attention.
RAG-Convindexes only conversation utterances. At inference time, the top- kutterance chunks most similar
to the question are retrieved. This configuration is directly comparable to retrieval-augmented approaches used
in existing memory benchmarks [2, 3], and has no access to document content.
RAG-Docindexes only document chunks. This is the natural RAG approach for long-document QA bench-
marks [4, 5] and serves as a strong Doc-only baseline. It has no access to conversation history.
RAG-Bothindexes both conversation utterances and document chunks into a unified vector store and retrieves
the top- kchunks across both modalities, ranked jointly by similarity score. This is the strongest retrieval baseline.
Memory Systemimplements a fact-extraction memory layer on top of ChromaDB [ 14]. For each conversation
session, an LLM prompt extracts up to 10 key facts as a structured list; these facts are embedded and stored
in a dedicated vector collection. At query time, the top- kmost similar facts are retrieved as context. Unlike
the RAG baselines, which operate on raw text chunks, the Memory System stores semantic abstractions of
conversations. Importantly, conversation sessions frequently reference and paraphrase document content, so
the extracted facts can capture document-derived information indirectly—explaining the system’s competitive
Doc-only score despite having no direct access to document chunks.
5.3 Evaluation Metrics
Exact Match (EM)A prediction is correct if, after normalization (lowercasing, removing articles and punctua-
tion), it exactly matches the gold answer string.
Token-level F1The harmonic mean of precision and recall computed over the bag of tokens in the predicted
and gold answers, after the same normalization.
9

Table 6: Token-level F1 per baseline by source tag (test split, n=160 ). Abstention Accuracy (AbstAcc) is
reported separately for Adversarial questions ( n=32 ). Bold indicates the highest value in each column. Test-split
counts for Doc-only ( n=23 ) and Chat-only ( n=16 ) are small; per-tag differences should be interpreted with
appropriate caution regarding statistical power.
Baseline Hybrid(n=121)Doc-Only(n=23)Chat-Only(n=16)Overall(n=160)AbstAcc
Base LLM 0.317 0.339 0.285 0.317 0.938
Long-Context LLM 0.330 0.318 0.283 0.323 0.844
RAG-Conv0.3480.351 0.275 0.341 0.938
RAG-Doc 0.267 0.453 0.292 0.2961.000
RAG-Both 0.3420.475 0.315 0.3580.969
Memory System 0.302 0.459 0.309 0.325 0.969
Table 7: Token-level F1 per baseline by question category (test split,n=160, 32 per category).
Baseline Single-Hop Multi-Hop Temporal Know. Update Adversarial
Base LLM 0.331 0.318 0.319 0.314 0.304
Long-Context LLM 0.310 0.310 0.340 0.329 0.327
RAG-Conv 0.3470.3240.3340.3720.329
RAG-Doc 0.408 0.277 0.220 0.233 0.342
RAG-Both0.4310.2980.3450.3670.350
Memory System 0.406 0.269 0.305 0.321 0.327
Abstention AccuracyFor Adversarial questions only, we evaluate whether the system correctly abstains rather
than producing a hallucinated answer. A system is instructed to respond with the special token ABSTAIN when
it cannot answer from the provided context. This metric is reported separately and not included in overall EM or
F1.
Token-level F1 and Abstention Accuracy are reported overall and broken down along two dimensions: (1)
question category and (2) source tag. Exact Match is omitted from the main tables as free-text answers rarely
produce exact string matches; F1 is the primary metric throughout. The per-source-tag F1 breakdown is the
primary diagnostic for measuring the joint reasoning gap.
5.4 Results
Tables 6 and 7 report F1 results on the test split.
The most striking pattern in Table 6 is theasymmetry of specialization. RAG-Doc achieves the second-highest
F1 on Doc-only questions (0.453) but thelowestoverall score (0.296) due to near-collapse on Hybrid questions
(0.267). This directly demonstrates that document retrieval without conversational navigation is insufficient
for joint reasoning. RAG-Both recovers this gap, improving Hybrid F1 from 0.267 to 0.342 by incorporating
conversation retrieval alongside document retrieval.
Long-Context LLM ranks fourth overall (0.323) despite receiving the full 60,000-token context window. Its
comparatively low abstention accuracy (0.844 vs. 0.938–1.000 for retrieval baselines) indicates that attending
over very long contexts increases the tendency to fabricate answers for unanswerable questions. The Memory
System (0.325 overall) is competitive with the long-context approach, particularly on Doc-only questions (0.459),
where its fact-extraction layer distils document-referenced facts from conversations into easily retrievable form.
In Table 7, all baselines show relatively uniform F1 across categories (range 0.267–0.431), with the exception of
RAG-Doc which degrades sharply on Temporal (0.220) and Knowledge Update (0.233) questions—categories
requiring tracking evolving state across sessions, information unavailable to document-only retrieval. Multi-hop
10

questions are consistently the weakest category for retrieval baselines (RAG-Both: 0.298), suggesting that
chaining evidence across two or more retrieval steps remains an open challenge.
6 Discussion
6.1 Do Hybrid Questions Expose a Genuine Joint-Retrieval Gap?
The answer is partially yes, but with important nuance. For baselines that specialize on one modality, Hybrid is
clearly the hardest source tag: RAG-Doc scores 0.267 on Hybrid versus 0.453 on Doc-only, and Memory System
scores 0.302 on Hybrid versus 0.459 on Doc-only. The pattern is consistent—a system that lacks access to one
modality is penalized precisely on the questions that require it.
However, the gap is less pronounced for baselines that have access to all context. Long-Context LLM actually
scoreshigheston Hybrid (0.330) among the three source tags, outperforming its own Doc-only score (0.318),
likely because Hybrid questions tend to have richer evidence in the conversation turns and because the question
text echoes conversational phrasing. The specialization asymmetry—strong on one source tag, weak on the
complementary tag—is the clearest signal that the benchmark is measuring something real.
6.2 Does RAG-Both Improve Over Specialised Baselines on Hybrid?
RAG-Both (Hybrid F1=0.342) substantially outperforms RAG-Doc (0.267) on Hybrid questions, confirming
that incorporating conversation retrieval recovers most of the gap caused by document-only indexing. However,
RAG-Both does not substantially outperform RAG-Conv (0.348) on Hybrid questions. This is a striking finding:
on Hybrid questions, conversation-side retrieval appears to carry most of the weight. The likely explanation is
that Hybrid question text tends to echo conversation language (persona names, event references) more closely
than it echoes specific document terminology, making conversation chunks more similar to queries in embedding
space.
This result suggests that the joint retrieval challenge in Hybrid questions is not symmetric: the hard part is
identifyingwhich documenta conversation references (a navigation problem), not retrieving passages from that
document once it is identified. No baseline we evaluate implements this two-stage strategy.
6.3 Does Long-Context LLM Outperform Base LLM?
Despite a 15 ×context size increase (4,096 →60,000 tokens), Long-Context LLM achieves only a marginal
overall improvement over Base LLM (0.323 vs. 0.317 F1). On Doc-only questions, Long-Context LLM is
actuallyworsethan Base LLM (0.318 vs. 0.339), and its abstention accuracy is the lowest of all baselines (0.844
vs. 0.938 for Base LLM). This pattern is consistent with the “lost in the middle” failure mode documented for
long-context transformers [ 15], in which relevant content embedded deep within a long context is systematically
underweighted by attention. The practical implication is that expanding the context window is not a substitute
for structured retrieval. Systems with access to the full context but no retrieval mechanism do not substantially
outperform systems that see only 4,096 tokens of it—while simultaneously becoming less reliable on adversarial
questions. Taken together, the results across all six baselines suggest that neither context length nor fact-based
memory substitutes for structured joint retrieval—the core design challenge this benchmark is intended to expose.
6.4 Human-System Gap
A formal human performance upper bound has not been measured in this work and remains an important
direction for future evaluation. Recruiting domain-knowledgeable annotators for caselaw material, ensuring
sufficient annotation time for long-document comprehension, and controlling for individual variance all require
infrastructure beyond the current scope.
11

We anticipate the human-system gap to be substantial, particularly on Hybrid questions. As a partial quality
signal, our LLM-as-judge verification (Section 3.6) found that the lenient judge accepted 67% of QA pairs
overall, with higher acceptance on Doc-only (96%) and Chat-only (61%) questions than on Hybrid (44%). The
lower Hybrid acceptance rate is consistent with the difficulty observed in baseline F1 scores. Future work
reporting human F1 on a sample of ≥50 test questions would anchor the benchmark’s difficulty and provide a
concrete target for system improvement.
6.5 Implications for System Design
The results collectively point to a specific architectural gap. No existing approach—long-context attention,
retrieval over one modality, retrieval over both, or fact-based memory—implements the two-step process that
Hybrid questions require: (1) use conversation context to identifywhichdocument is relevant, and (2) retrieve
specifically from that document. RAG-Both comes closest but does so with a flat similarity ranking that cannot
enforce the navigational dependency between step 1 and step 2.
A system that could close the Hybrid gap would need at minimum: (a) a representation of which conversation
sessions reference which documents (a citation graph), and (b) a retrieval strategy that conditions document-
chunk retrieval on the output of conversation-level retrieval. Memory systems designed around entity and
relationship graphs (e.g., Mem0 [ 9], Zep [ 10]) could in principle encode document references as graph edges, but
neither system in its current form is designed for this use case. MemoryDocDataSet makes the gap measurable
and provides the scaffolding for evaluating future approaches that attempt to close it.
Limitations
Domain scope.The current document corpus is drawn entirely from US court opinions sourced from the Caselaw
Access Project [ 11]. While legal text is a natural fit for the benchmark, it introduces domain bias. We plan to
expand to additional document types and domains in future releases.
LLM-generated conversation quality.Conversation sessions are generated by a language model conditioned
on the event graph and document set. LLM-generated dialogues may not capture the full range of natural
language phenomena found in real human conversations—including implicit references, pragmatic inferences,
code-switching, and repair sequences.
Automated QA generation.QA pairs and gold answers are produced by an LLM and validated through
structural checks. Automated generation cannot guarantee that every gold answer is uniquely correct, that
distractors in Adversarial questions are sufficiently realistic, or that Knowledge Update questions capture all
intermediate states.
LLM-as-judge verification limitations.Quality verification is performed by two LLM judge instances from
the same model family as the generation pipeline, introducing potential self-leniency bias. Additionally, the
prompt-sensitivity κwe measure reflects base-rate differences between strict and lenient prompts as much as
genuine label disagreement.
English only.The pipeline and document corpus are currently English-only.
Synthetic scenario realism.Micro-world personas, event graphs, and conversation sessions are synthetically
generated. The temporal event graphs enforce a clean DAG structure, which may be more regular than the
overlapping, ambiguous timelines found in real cases.
Fixed Hybrid threshold.The 30% Hybrid question floor is a dataset design parameter chosen to ensure reliable
per-source-tag metrics. It is not empirically derived from a study of real-world task distributions.
12

7 Conclusion
We introduced MemoryDocDataSet to close a gap that existing benchmarks leave open: no prior work evaluates
AI systems on the joint task of navigating multi-session conversation historyandperforming deep reading
within long documents simultaneously. The Hybrid source tag formalizes this requirement—questions that
are unanswerable without first using conversation context to identify the relevant document, then reading that
document to extract the answer.
Our baseline experiments reveal a clearspecialization asymmetrywith direct implications for system design.
RAG-Doc, the natural approach for long-document QA, collapses to 0.267 F1 on Hybrid despite achieving 0.453
on Doc-only—a 41% relative drop caused by the structural inability of document-only retrieval to resolve which
document a conversation is referring to. RAG-Both closes much of this gap (Hybrid F1=0.342) but the remaining
shortfall points to the core open problem: no existing baseline implements the two-step navigational strategy
that Hybrid questions require. A system that first uses conversation retrieval to identify the relevant document,
then applies targeted retrieval within that specific document, would be architecturally suited to this task—and no
current approach does so.
The practical implication is that the next generation of AI assistants will need to couple conversational memory
with long-document navigation at the architectural level, not just at the context-window level. MemoryDoc-
DataSet makes this gap measurable, reproducible, and quantifiable for the first time. We release the dataset,
generation pipeline, and all baseline implementations to support this line of research.
References
[1]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-
Augmented Generation for Knowledge-Intensive NLP Tasks. InAdvances in Neural Information Processing
Systems, volume 33, pages 9459–9474, 2020.
[2]Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang.
Evaluating Very Long-Term Conversational Memory of LLM Agents. InProceedings of the 62nd Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13851–13870,
2024.
[3]Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. LongMemEval:
Benchmarking Chat Assistants on Long-Term Interactive Memory. InProceedings of the Thirteenth
International Conference on Learning Representations, 2025.
[4]Chenxin An, Shansan Gong, Ming Zhong, Xingjian Zhao, Mukai Li, Jun Zhang, Lingpeng Kong, and
Xipeng Qiu. L-Eval: Instituting Standardized Evaluation for Long Context Language Models. InPro-
ceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers), pages 14388–14411, 2024.
[5]Uri Shaham, Maor Ivgi, Avia Efrat, Jonathan Berant, and Omer Levy. ZeroSCROLLS: A Zero-Shot
Benchmark for Long Text Understanding. InFindings of the Association for Computational Linguistics:
EMNLP 2023, pages 7977–7989, 2023.
[6]Jing Xu, Arthur Szlam, and Jason Weston. Beyond Goldfish Memory: Long-Term Open-Domain Conversa-
tion. InProceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume
1: Long Papers), pages 5180–5197, 2022.
13

[7]Haoran Tan, Zeyu Zhang, Chen Ma, Xu Chen, Quanyu Dai, and Zhenhua Dong. MemBench: Towards
More Comprehensive Evaluation on the Memory of LLM-based Agents. InFindings of the Association for
Computational Linguistics: ACL 2025, pages 19336–19352, 2025.
[8]Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and
Christopher D. Manning. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.
InProceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages
2369–2380, 2018.
[9]Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building
Production-Ready AI Agents with Scalable Long-Term Memory.arXiv preprint, 2025.
[10] Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, and Daniel Chalef. Zep: A Temporal
Knowledge Graph Architecture for Agent Memory.arXiv preprint, 2025.
[11] Harvard Law School. Caselaw Access Project. 2018.
[12] Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, Hanna Wallach, Hal
Daumé III, and Kate Crawford. Datasheets for Datasets.Communications of the ACM, 64(12):86–92, 2021.
[13] Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence Embeddings Using Siamese BERT-Networks.
InProceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th
International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992,
2019.
[14] Chroma. Chroma: The AI-Native Open-Source Embedding Database. 2022.
[15] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. Lost in the Middle: How Language Models Use Long Contexts.Transactions of the Association for
Computational Linguistics, 12:157–173, 2024.
Appendix
A Pipeline Stage Details
Stage 1 — Document Collection.Raw opinions are downloaded from the Caselaw Access Project bulk export
and processed into long document objects. Each document is tokenized with cl100k_base and filtered to
20,000–50,000 tokens. Documents are deduped by case ID and split into per-state shards to support reproducible
batched runs.
Stage 2 — Persona & Event Graph Generation.For each micro-world the pipeline selects 3–5 documents at
random (without replacement), infers a domain label from their content, and calls the persona and event-graph
generators in sequence. Persona count is sampled from personas_per_world (default 3–5). Event count is
sampled from events_per_graph (default 5–10). The event graph is validated as a DAG before acceptance;
malformed graphs are retried up to three times.
Stage 3 — Conversation Generation.Sessions are generated one per event in topological order of the event
graph, so that each session can reference prior-session context. A grounding flag controls whether a session
must include at least one document citation (applied to grounding_ratio_threshold× sessions). Each
session is generated independently, making the stage trivially parallelisable.
14

Stage 4 — QA Generation.Five category-specific prompts are called per micro-world. A post-generation
filter enforces at least min_qa_per_category pairs per category and at least doc_navigation_ratio
pairs with requires_doc_navigation=true . Pairs failing structural validation are discarded; the stage
retries if minimum counts are not met.
Stage 5 — Quality Verification.Two LLM judge instances (strict and lenient system prompts, same base
model) evaluate every micro-world in parallel. Each judge returns per-QA correctness labels ( qa_answer_correctness )
and per-world Likert quality scores (persona consistency, temporal coherence, document grounding accuracy,
1–5). Cohen’s κis computed over the qa_answer_correctness labels. All micro-worlds are retained
regardless of κvalue ( agreement_threshold: 0.0 );κis reported as a prompt-sensitivity quality
characterisation rather than a hard filter (see Section 3.6 for rationale).
Stage 6 — Dataset Packaging.Verified micro-worlds are shuffled (seed fixed to random_seed ) and split
70/14/16 into train, validation, and test sets at the micro-world level. No QA pair from one split shares a
micro-world with another split. Document-level leakage is also prevented: a document assigned to a micro-world
in the test split does not appear in any train or validation micro-world.
Stage 7 — Baseline Evaluation.Each baseline system is given the question text and the access mode
appropriate to its design. Predictions are scored against gold answers using token-level F1 and exact match
after lowercasing, article removal, and punctuation stripping. Abstention accuracy is computed separately for
Adversarial questions using a reservedABSTAINtoken.
B Prompt Templates
All LLM calls use a two-part message: a system prompt setting the generator’s role and a user prompt containing
the structured input and output schema. Templates are reproduced verbatim below; slot names in {braces}
are filled at runtime.
B.1 Persona Generation
System prompt
You are a creative writer specializing in generating realistic professional personas for synthetic dataset generation. You produce
well-structured JSON output that strictly follows the requested schema.
User prompt
Generate {count} unique professional personas for a micro-world in the “ {domain} ” domain. Each persona must have: a
unique full name; a professional role relevant to the domain; an expertise domain; a communication style; relationships to other
personas. Return a JSON array where each element has: name ,role ,expertise_domain ,communication_style ,
relationships . Document summaries for context: {document_summaries} . Return ONLY valid JSON, no markdown
fences or extra text.
B.2 Event Graph Generation
System prompt
You are an expert at creating realistic temporal event sequences for professional scenarios. You produce well-structured JSON output
that strictly follows the requested schema. Events must form a valid directed acyclic graph (DAG) with no cycles.
15

User prompt
Generate a temporal event graph with {event_count} events for a “ {domain} ” micro-world. Events must span at least
{time_span_months} months, form a valid DAG, reference at least one persona and one document each, and ensure every
document is referenced by at least one event. Return a JSON object with: time_range_start ,time_range_end , and
anevents array. Each event has: event_id ,timestamp (ISO 8601), description ,persona_ids ,document_ids ,
depends_on. Return ONLY valid JSON, no markdown fences or extra text.
B.3 Conversation Session Generation
System prompt
You are an expert dialogue writer who creates realistic multi-party professional conversations. You produce well-structured JSON
output that strictly follows the requested schema. Conversations must be grounded in the provided event context and reference
documents when appropriate.
User prompt
Generate a realistic conversation session for the following event in a “ {domain} ” scenario. Event: {event_description}
at{event_timestamp} . Participating personas: {personas_json} . Documents available for reference:
{documents_json} . Prior conversation context: {prior_context} . Instructions: generate natural dialogue; respect each
persona’s communication style; include temporal markers; {grounding_instruction} ; aim for {target_utterances}
utterances. Return a JSON object with anutterancesarray andreferenced_document_ids. Return ONLY valid JSON.
B.4 QA Generation
All five category prompts share the same system prompt:
System prompt (shared across all five categories)
You are an expert benchmark dataset creator who generates high-quality question-answer pairs for evaluating AI systems on joint
conversational memory and long document reasoning. You produce well-structured JSON output that strictly follows the requested
schema.
Category-specific core instructions are summarized in Table 8. All category prompts request the same out-
put schema per QA pair: question ,gold_answer ,evidence_references (with source_type ,
source_id,passage_span),difficulty,requires_doc_navigation.
Table 8: QA generation category-specific instructions.
Category Core instruction
Single-hop Each question answerable from a
single source; some require navigat-
ing from conversation to document.
Multi-hop Each question requires facts from
≥2 distinct sources; chain reasoning
across sessions or across conversa-
tion and document.
Temporal Each question requires reasoning
about chronological ordering; refer-
ence≥2 distinct events/timestamps.
Know. Update Each question tests use of the most
current information; at least one su-
perseded fact must exist.
Adversarial Mix of false_premise ,
unanswerable , and abstain
types.
16

B.5 LLM-as-Judge (Quality Verification)
Two instances of the same model are run with contrasting system prompts and an identical user prompt body.
Strict system prompt
You are a strict quality reviewer for a benchmark dataset of QA pairs grounded in conversations and long documents. Apply skeptical
scrutiny: if a QA pair’s gold answer is not unambiguously supported by the cited evidence, mark it incorrect. If a question’s category
label is debatable, mark it inaccurate. Err on the side of flagging. Output a single JSON object matching the requested schema. Do
not add commentary outside the JSON.
Lenient system prompt
You are a charitable quality reviewer for a benchmark dataset of QA pairs grounded in conversations and long documents. Apply
reasonable interpretation: if the gold answer is consistent with the cited evidence under any reasonable reading, mark it correct.
Only flag clear errors or unsupported claims. Output a single JSON object matching the requested schema. Do not add commentary
outside the JSON.
User prompt (shared).A compact context is rendered per micro-world containing the list of personas, the full
conversation sessions, and each QA pair with its cited evidence excerpts inlined. The judge returns a JSON object
withpersona_consistency (1–5), temporal_coherence (1–5), document_grounding_accuracy
(1–5), qa_answer_correctness (map of QA ID →bool), qa_category_accuracy (map of QA ID
→bool),flagged_qa_ids, andnotes.
C Dataset Example
Below is an illustrative excerpt from one micro-world in the caselaw domain (all names and case references are
from authentic Caselaw Access Project documents).
Personas (excerpt)
•Margaret Chen— Senior Partner; contract law; “formal and precise”
•David Rodriguez— Associate Attorney; litigation; “analytical and thorough”
•Sarah Kim— Legal Researcher; case law research; “methodical and detail-oriented”
Event graph (excerpt, 3 of 7 events)
•evt-1(2024-01-10): Initial client consultation on breach of contract; involves Chen, Rodriguez; references Case A.
•evt-3(2024-02-14): Discovery of precedent contradicting client position; involves Kim; references Case B.
•evt-5(2024-03-22): Strategy revision meeting; involves all three personas; references Cases A and B.
QA example — Single-hop
Q:Which attorney conducted the initial client consultation on the breach of contract claim?
A:Margaret Chen and David Rodriguez.
Source:evt-1conversation session.
QA example — Hybrid (doc navigation required)
Q:What specific contractual clause did the precedent case identified by Kim in February establish as unenforceable?
A:[answer drawn from Case B document after identifying it viaevt-3conversation]
Source:evt-3conversation session→Case B document.
QA example — Knowledge Update
Q:What was the legal team’s recommended strategy for the breach of contract case as of March 2024?
A:[revised strategy fromevt-5, superseding the initial strategy fromevt-1]
Source:evt-5conversation session;evt-1(superseded).
17