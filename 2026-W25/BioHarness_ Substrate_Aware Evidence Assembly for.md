# BioHarness: Substrate-Aware Evidence Assembly for Biomedical Question Answering across Literature, Knowledge Bases, and Biological Atlases

**Authors**: Meng Xiao, Chuan Qin, Jinmiao Chen, Yihang Cheng, Yuanchun Zhou, Hengshu Zhu

**Published**: 2026-06-17 06:25:12

**PDF URL**: [https://arxiv.org/pdf/2606.19396v1](https://arxiv.org/pdf/2606.19396v1)

## Abstract
Motivation: Biomedical question answering often requires evidence beyond topically retrieved literature, including gene alias resolution, database identifier normalization, and atlas-derived biological measurements. However, existing retrieval-augmented generation (RAG) systems typically follow a fixed workflow and lack an explicit mechanism for deciding when retrieved text is sufficient, when curated biomedical knowledge is required, or when executable evidence assembly over structured measurements should be invoked. This motivates a substrate-aware large language model (LLM) harness that selectively assembles sufficient evidence across literature, knowledge bases, and biological atlases.
  Results: We introduce BioHarness, an LLM harness for staged biomedical evidence assembly across literature retrieval, curated biomedical knowledge resources, and atlas-derived structured measurements. BioHarness first attempts to answer from reranked literature evidence and escalates through grounded cascade control to REPL-style evidence assembly only when the current evidence is uncertain, weakly grounded, or substrate-mismatched. Across 19,302 biomedical QA items spanning seven answer formats, BioHarness improves the pooled score from 65.9 to 71.0 over the strongest non-oracle baseline. Ablations, case studies, and backbone-scaling analyses show that these gains arise from repairing evidence-substrate mismatches through reranking, entity grounding, and structured measurement access, rather than from indiscriminately invoking more reasoning steps, retrieving additional literature, or relying on a particular answer-model scale.

## Full Text


<!-- PDF content starts -->

Bioinformatics,2022, 1–14
doi: DOI HERE
Advance Access Publication Date: Day Month Year
Paper
BioHarness: Substrate-Aware Evidence Assembly for
Biomedical Question Answering across Literature,
Knowledge Bases, and Biological Atlases
Meng Xiao ,1,2Chuan Qin ,1,3Jinmiao Chen ,2,4Yihang Cheng ,1
Yuanchun Zhou1,3,∗and Hengshu Zhu1,3,∗
1Computer Network Information Center, Chinese Academy of Sciences, South Fourth Street, 100180, Beijing,
China,2DUKE-NUS Medical School, National University of Singapore, 8 College Rd, 169857, Singapore,
3University of Chinese Academy of Sciences, Yanqihu East Rd, 101408, China and4Bioinformatics Institute,
Agency for Science, Technology and Research, 30 Biopolis Street, 138671, Singapore
∗Corresponding authors: Yuanchun Zhou (zyc@cnic.cn) and Hengshu Zhu (hszhu@cnic.cn)
FOR PUBLISHER ONLY Received on Date Month Year; revised on Date Month Year; accepted on Date Month Year
Abstract
Motivation: Biomedical question answering often requires evidence beyond topically retrieved literature,
including gene alias resolution, database identifier normalization, and atlas-derived biological measurements.
However, existing retrieval-augmented generation (RAG) systems typically follow a fixed workflow and lack
an explicit mechanism for deciding when retrieved text is sufficient, when curated biomedical knowledge
is required, or when executable evidence assembly over structured measurements should be invoked. This
motivates a substrate-aware large language model (LLM) harness that selectively assembles sufficient evidence
across literature, knowledge bases, and biological atlases.
Results: We introduce BioHarness, an LLM harness for staged biomedical evidence assembly across literature
retrieval, curated biomedical knowledge resources, and atlas-derived structured measurements. BioHarness
first attempts to answer from reranked literature evidence, and escalates through grounded cascade control to
REPL-style evidence assembly only when the current evidence is uncertain, weakly grounded, or substrate-
mismatched. Across 19,302 biomedical QA items spanning seven answer formats, BioHarness improves
the pooled score from 65.9 to 71.0 over the strongest non-oracle baseline. Ablations, case studies, and
backbone-scaling analyses show that these gains arise from repairing evidence-substrate mismatches through
reranking, entity grounding, and structured measurement access, rather than from indiscriminately invoking
more reasoning steps, retrieving additional literature, or relying on a particular answer-model scale.
Availability and implementation: Source code∗, reproducible benchmarking scripts†, and processed data‡are
all publicly available.
Key words:biomedical question answering; retrieval-augmented generation; large language models; evidence
assembly; biomedical knowledge bases; biological atlases
Introduction
Biomedical question answering with large language models (Luo
et al., 2025; Zhou et al., 2025; Zhang et al., 2026) is not a
single-substrate retrieval problem. A biomedical question may
require an answer-bearing literature statement (Jin et al., 2022,
∗https://github.com/coco11563/BioHarness_Chi
†https://github.com/coco11563/BioHarness_Eval_Framework
‡https://huggingface.co/datasets/Shaow/BioHarness_Eval2023), canonical entity grounding behind an ambiguous surface
mention (Wei et al., 2024; Gyori et al., 2022), or a measured
expression pattern stored in a bulk expression resource, tissue
atlas, or single-cell atlas (George et al., 2024; Li et al., 2022;
Consortium et al., 2022; Program et al., 2025). We refer to
these distinct forms of evidence as biomedical evidence substrates
because they differ not only in content but also in access patterns,
reliability profiles, and failure modes. Literature evidence is
accessed via semantic text retrieval, knowledge-base evidence via
©The Author 2022. Published by Oxford University Press. All rights reserved. For permissions, please e-mail: journals.permissions@oup.com1arXiv:2606.19396v1  [q-bio.QM]  17 Jun 2026

2 Meng Xiao et al.
Fig. 1: Heterogeneous evidence substrates in biomedical question
answering. Literature evidence supports claim-level reasoning
through semantic retrieval, knowledge-base evidence supports
entity-level grounding through identifier and relation lookup,
and atlas-derived evidence supports measurement-level reasoning
through executable queries over structured biological data.
identifier and relation lookup, and atlas-derived evidence via
executable operations on structured biological measurements. As
illustrated in Figure 1, fixed retrieve-then-generate pipelines can
fail through evidence-substrate mismatch: topically relevant text
may not contain answer-supporting evidence, surface names may
not resolve to the correct biomedical identity, and measured
biological values may be absent from narrative abstracts even when
they are available in structured resources.
Existing biomedical RAG systems improve individual parts
of this process, but rarely treat evidence-substrate selection as
an explicit control problem. Text-centered methods, including
dense retrieval, query expansion, reranking, and reflective evidence
filtering, improve the recall or ranking of relevant passages, but
still assume that the answer-supporting evidence can be recovered
from retrieved text (Lewis et al., 2020; Karpukhin et al., 2020; Gao
et al., 2023; Yu et al., 2024b; Asai et al., 2023; Yu et al., 2024a).
Biomedical tool-augmented and knowledge-graph-based methods
incorporate structured resources or entity-relation context, yet
these resources are often attached as fixed preprocessing, auxiliary
context, or graph summaries rather than invoked according to
evidence sufficiency (Jin et al., 2024; Wang et al., 2024; Li
et al., 2025; Edge et al., 2024; Matsumoto et al., 2024; Soman
et al., 2024). Agentic retrieval and reasoning frameworks can
further interleave search, tool use, and multi-step inference, but
indiscriminate execution increases costs and may still operate on
the wrong substrate when the question requires canonical entity
grounding or database-native measurements (Yao et al., 2022;
Trivedi et al., 2023; Shinn et al., 2023). Thus, the central problem
is not simply how to retrieve more passages, add more tools, orreason for more steps, but how to decide which evidence substrate
is sufficient for a given biomedical question.
To address this limitation, we introduce BioHarness, a
substrate-aware LLM harness for staged biomedical evidence
assembly and question answering. BioHarness first constructs
a literature evidence context through complementary retrieval
views and question-conditioned reranking, then applies grounded
cascade control to decide whether the current evidence is sufficient
for a fast answer. When the retrieved evidence is uncertain, weakly
grounded, or mismatched to the question requirement, BioHarness
escalates to a REPL-style evidence assembly loop, where the LLM
can inspect passages, invoke biomedical tools for entity grounding,
access router-gated atlas evidence, and assemble a compact
evidence state for final answer generation. A task-constrained
output head finally normalizes the answer to the required format,
allowing the same harness to support yes/no, multiple-choice,
factoid, list, summary, and expression-style questions.
Our contributions are threefold. First, we identify evidence-
substrate mismatch as a central failure mode in biomedical
RAG, showing that failures can arise not only from insufficient
retrieval but also from applying textual retrieval to questions that
require entity grounding or structured biological measurements.
Second, we propose BioHarness, a substrate-aware LLM
harness that connects literature evidence, curated biomedical
knowledge, and atlas-derived biological measurements through
dual evidence retrieval, question-conditioned reranking, cascade
control, biomedical tools, router-gated atlas access, REPL-style
evidence repair, and task-constrained answer generation. Third,
we conduct an extensive controlled analysis across heterogeneous
biomedical QA datasets, component ablations, efficiency settings,
backbone scales, and mechanistic case studies, demonstrating
that substrate-aware evidence assembly improves biomedical
QA beyond fixed retrieval, query-expansion, reflective, tool-
augmented, and graph-based RAG pipelines.
Method
As shown in Figure 2 and Algorithm 1, BioHarness implements
a substrate-aware biomedical evidence assembly pipeline that
first attempts to answer from literature evidence and escalates
only when additional evidence substrates are needed. When the
literature context is uncertain, weakly grounded, or substrate-
mismatched, REPL-style evidence repair incorporates biomedical
tool outputs and atlas-derived measurement records to construct
a sufficient evidence state.
Literature Evidence Assembly
The literature evidence component constructs an initial evidence
context from a biomedical literature corpus constructed from
PubMed abstracts. This component addresses a basic limitation
of single-query dense retrieval: top-ranked abstracts may be
topically related to the question, but still miss the specific
answer-bearing evidence needed by the downstream task. Within
the literature substrate, BioHarness combines complementary
retrieval views with question-conditioned reranking to improve
recall while preserving answer relevance.
As shown in Figure 2(a), the first retrieval view uses the original
question, preserving literal wording and entity-level precision. The
second retrieval view follows the hypothetical-document retrieval
principle (Gao et al., 2023): the LLM rewrites the question as a

BioHarness 3
Fig. 2: BioHarness implements a substrate-aware biomedical evidence assembly pipeline consisting of (a) literature evidence assembly
with reranking, (b) grounded evidence sufficiency control, and (c) executable evidence repair over biomedical tools and atlas-derived
structured measurements, followed by a task-constrained answer generation head.
Hypothetical Abstract Rewrite Prompt
Please write a short scientific PubMed-style abstract
that answers the question below. Write in one
paragraph, 4--6 sentences, focused on the biomedical
facts directly relevant to the question.
Question:{question}
Abstract:
Fig. 3: Prompt used for Hypothetical Abstract Rewrite. The
biomedical question is converted into a hypothetical abstract and
then used as an additional retrieval query.
short PubMed-style hypothetical abstract, as in the prompt in
Figure 3, thereby moving the retrieval query closer to a plausible
biomedical evidence context. The retrieved candidates from active
retrieval views are merged, de-duplicated by document identifier
when available, and reranked against the original question. This
reranking step mitigates hypothesis drift from the rewritten query
while retaining its recall benefit.
The resulting Literature Evidence context is passed to the
cascade controller as the initial evidence substrate. Importantly,
this component is not intended to solve all biomedical evidence
needs through literature retrieval alone. When a question requires
contrastive evidence, the literature retriever may include task-
aware query variants before reranking; when the primary evidence
need is canonical entity grounding or structured biological
measurement, the retrieved literature is retained as a lightweight
context and the unresolved evidence need is delegated to
downstream structured evidence substrates. Thus, literatureevidence assembly serves as the first substrate in a staged evidence
assembly process, rather than as a universal retrieval solution.
Grounded Evidence Sufficiency Control
After constructing the Literature Evidence context, BioHarness
applies a grounded evidence sufficiency controller to decide
whether the current evidence is adequate or whether additional
evidence substrates should be invoked. The controller is
designed to separate answer generation from evidence sufficiency
estimation: literature evidence is used when it provides a confident,
well-grounded answer, whereas structured evidence repair is
invoked when the draft answer is uncertain, unsupported, or
mismatched with the question’s requirements. As shown in
Figure 2(b) and Algorithm 1, given the literature evidence set
D, BioHarness first generates a draft answera(0)under the task
format and computes a lightweight confidence scorec= exp(ℓ),
whereℓis the log-probability of the first generated token. This
confidence score is used only as a routing proxy, rather than as a
calibrated estimate of answer correctness (Kadavath et al., 2022).
In parallel, the controller applies an answer-form-aware grounding
check to estimate whether the draft is supported by the retrieved
context. For answer forms whose support can be localized in text,
the check compares the normalized draft answer or its content
tokens against the retrieved titles and abstracts; when lexical
grounding is unreliable for a task format, routing relies primarily
on the confidence signal and downstream evidence repair.
The controller escalates a query when the draft answer is empty,
has low confidence, or is not grounded in the retrieved evidence. If
escalation is not triggered, the draft answer is returned as the final
answer. Otherwise, the query is passed to REPL-style evidence

4 Meng Xiao et al.
Algorithm 1Grounded Evidence Sufficiency Controller of
BioHarness.
Require:questionq, task typet, optionsOif applicable
Ensure:task-constrained answera
1:D ←LiteratureEvidenceAssembly(q, t)
2:(a(0), ℓ)←DraftPrediction(q,D;t, O)
3:c←exp(ℓ) # Confidence proxy
4:e←EvidenceGroundingCheck(a(0),D, t) # Evidence
grounding signal
5:π←CascadeDecision(t, c, e, a(0);θc)
6:ifπ=Fastthen
7:returna(0)# Literature-only path
8:end if
9:h←REPLEvidenceAssembly(q, t,D, O;N iter) #
Algorithm 2
10:a←FinalAnswer(q,EvidencePack(h,D);t, O)
11:returna
repair, where biomedical tools and atlas-derived measurement
records are initialized as structured evidence substrates. This
policy preserves the literature-only path when textual evidence
is sufficient, while routing uncertain or weakly grounded drafts to
the substrates required for evidence repair.
Structured Biomedical Evidence Substrates
When literature evidence is insufficient, BioHarness can access two
structured biomedical evidence substrates: the Biomedical Tool
Layer and the Atlas Evidence Layer1. Together, they provide
access patterns that are difficult to recover through dense retrieval
alone. The Biomedical Tool Layer supports entity-level grounding,
including gene-symbol normalization, alias and deprecated-symbol
resolution, identifier lookup, ontology traversal, and lightweight
database-style checks. These operations are treated as evidence-
producing functions rather than external text generators: their role
is to bridge ambiguous surface mentions to canonical biomedical
entities, identifiers, and curated relations.
The Atlas Evidence Layer supports measurement-level
grounding through a read-only atlas namespace. Atlas access
is controlled by a router that converts atlas-relevant questions
into an atlas query specification, including candidate genes,
tissues or cell types, measurement fields, and optional operations
such as filtering or thresholding. Only routed cases fetch the
corresponding atlas entries before REPL execution. The fetched
atlas object is represented as structured measurement evidence,
containing gene metadata, tissue-expression records, cell-type
annotations, quantitative values, and provenance fields when
available. During REPL execution, these structured records
can be inspected, filtered, aggregated, or thresholded through
deterministic operations.
The two substrates are deliberately separated because they
address different evidence needs. Biomedical tools resolve names,
identifiers, aliases, ontological terms, and curated database
relations, whereas atlas access supplies biological measurements
such as tissue-level or cell-type-level expression profiles. This
separation allows BioHarness to switch evidence substrates
1The details of involving biomedical tools and atlas can be found
in the attachment.Algorithm 2REPL-style Evidence Repair.
Require:questionq, task typet, literature evidenceD, options
Oif applicable, iteration capN iter
Ensure:evidence traceh
# Stage A – initialize heterogeneous evidence substrates
1:T←InitBiomedicalTools(q, t) # Initialize callable
grounding tools
2:r←AtlasRouter(q, t) # Generate atlas query specification
3:ifr.route atlasthen
4:A←FetchAtlasEvidence(r.query spec) # Structured
measurement evidence
5:else
6:A←∅
7:end if
8:h←InitContext(q, t,D, T, A, O)
# Stage B – iterative executable evidence repair
9:fori= 1, . . . , N iterdo
10:code i←CodeGeneration(h) #①Code generation
11:exec i←ReplExec(code i) #②REPL execution
12:obs i←ResultObservation(exec i) #③Result
observation
13:h←h∥code i∥obsi # Update evidence trace
14:ifIsFinal(obs i)then
15:break#④Stop if sufficient
16:end if
17:end for
18:returnh
according to the failure mode of the literature context, rather than
repeatedly querying the same textual corpus.
Executable Evidence Repair
When the controller rejects the literature-only path, BioHarness
enters a REPL-style evidence repair loop. As shown in
Algorithm 2, the loop is not used as an unconstrained reasoning
chain. Instead, it provides an executable workspace for substrate-
aware evidence repair: the LLM can inspect retrieved passages,
invoke read-only biomedical tools, access router-gated atlas
evidence, and assemble a compact evidence state through
executable operations.
This design targets two common repair needs after literature
retrieval. First, some questions require symbolic biomedical
grounding rather than additional text retrieval. Deprecated
aliases, surface-form ambiguity, and missing identifiers are better
resolved by gene normalization and database lookup tools than
by repeatedly searching the literature corpus. Second, atlas-
derived evidence is often structured and numeric rather than
narrative. Expression patterns, tissue distributions, and cell-
type measurements naturally require executable operations such
as filtering rows, applying thresholds, aggregating values, or
constructing entity sets. A REPL-style interface therefore matches
the access pattern of structured atlas evidence more closely than
a pure retrieve-then-generate pipeline.
The evidence tracehrecords the action–observation history
and is later compressed into an evidence pack for final
answer generation. The REPL interface gives the harness two
properties that ordinary retrieve-then-generate pipelines lack.
First, observations are stateful: intermediate outputs such as

BioHarness 5
resolved symbols, candidate entity sets, database records, tissue-
expression rows, and filtered atlas measurements remain available
for later operations. Second, execution is inspectable: the trace
exposes whether the repair came from entity normalization,
database lookup, atlas filtering, or another substrate-specific
operation. This makes evidence repair both substrate-aware and
auditable, rather than hidden inside free-form generation.
Task-constrained Answer Generation
After either the literature-only path or REPL-style evidence
repair, BioHarness produces the final answer through a task-
constrained output head. For literature-only cases, the draft
answer is returned when the controller determines that it is
sufficiently confident and grounded in the literature evidence.
For escalated cases, the REPL trace is compressed into an
evidence pack that includes retrieved literature snippets, resolved
biomedical entities, tool observations, atlas-derived measurement
records, and, when available, provenance information. The output
head then converts this evidence pack into an answer compatible
with the task specification and answer options.
This step separates evidence assembly from answer formatting.
Instead of relying on unconstrained free-form generation, the
output head constrains the response according to the required
output space, such as closed-set options, normalized entity
spans or entity sets, or concise evidence-grounded summaries.
This ensures that heterogeneous evidence traces from literature,
biomedical tools, and atlas-derived measurements are converted
into outputs compatible with the evaluation protocol.
Experimental Settings
Task Description
We evaluate biomedical QA in a unified evidence-assembly setting.
Given a queryq, task typet, and answer optionsOwhen available,
the system is required to produce an answerausing the assembled
evidence context. The benchmark covers heterogeneous answer
formats, including yes/no questions, single-answer multiple-choice
questions, multi-answer choice questions, short factoid answers,
list-style answers, summaries, and structured expression outputs.
Evaluation Metrics
We evaluate each item using a type-specific matching score and
report all scores on a 0–100 scale. For yes/no questions, we
use lowercased exact-match accuracy. For single-answer multiple-
choice questions, we use exact match on the predicted option
letter. For multi-answer choice questions, we compute set-F1
over the predicted and gold option sets. For factoid questions,
answers are usually short entities or phrases, so we use token-
level F1 (Rajpurkar et al., 2016) after answer normalization,
equivalent to unigram-overlap F1. For list-style questions, we
compute synonym-aware set-F1 between the predicted and gold
answer sets. For summary questions, we use ROUGE-L (Lin, 2004)
F1 because it captures longest-common-subsequence overlap and
better reflects content preservation in longer free-form answers.
For expression-style questions, we compute entity-set F1 over the
target entity set.
Fig. 4: MeSH-colored subsampled visualization of the PubMed
embedding space. Each point denotes a PubMed abstract encoded
by Qwen3-Embedding-0.6B, with colors assigned according to
article-associated MeSH terms and labels marking representative
MeSH categories.
Lets i∈[0,1] denote the type-specific matching score for item
i. For a datasetD, the reported dataset score is computed as
100×1
|D|X
i∈Dsi.
The overall score is computed analogously over the full benchmark
with sizeN:
100×1
NNX
i=1si,
Thus, all reported scores are pooled item-level averages of type-
specific matching scores.
Implementation Details
Experiments are conducted on a server equipped with eight
NVIDIA A800-SXM4-80GB GPUs. All LLMs are served
locally using vLLM2, with deterministic decoding and identical
generation settings across methods. Unless otherwise specified,
generation-based components use Qwen3.5-35B-A3B3as the
common backbone, including hypothetical abstract rewriting,
draft prediction, REPL-style evidence repair, and final answer
generation. For literature retrieval, we construct a unified
PubMed-derived corpus from approximately 20 million publicly
available abstracts. Questions and abstracts are encoded using
Qwen3-Embedding-0.6B4, and abstracts are indexed in Qdrant5
for dense vector retrieval. Retrieved candidates are reranked
under the original question using Qwen3-Reranker-8B6. Figure 4
provides a subsampled visualization of the PubMed abstract
embedding space, in which MeSH-colored neighborhoods indicate
that the retrieval corpus preserves the broad biomedical topic
structure. We use this visualization only as a qualitative sanity
2https://vllm.ai
3https://huggingface.co/Qwen/Qwen3.5-35B-A3B
4https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
5https://qdrant.tech
6https://huggingface.co/Qwen/Qwen3-Reranker-8B

6 Meng Xiao et al.
Table 1.Main results on eight biomedical QA benchmarks.Golden Contextuses dataset-provided gold evidence when available and is reported as an
oracle evidence upper bound; it is excluded from best/second-best highlighting. Among non-oracle methods, the best and second-best results are shown
inboldand underlined , respectively; ties are marked accordingly.
Method PubMedQA BioASQ GeneTuring SciHorizon MedMCQA MedQA-US MedQA-TW MedQA-CN Overall
Retrieval-free baseline
No-Context 50.6 45.4 7.9 42.1 72.7 82.0 89.1 89.3 62.1
w/ Golden Context†78.0 66.4 – – – – – – –
Dense retrieval and query augmentation
DenseRAG 75.4 47.2 8.9 43.3 73.1 84.0 89.0 89.4 63.6
HyDE 75.4 53.3 10.6 48.1 72.9 84.1 88.5 88.8 65.7
Reranking, filtering, and reflective retrieval
DenseRAG + Rerank 75.9 50.1 9.6 45.4 73.5 82.6 89.3 89.0 64.4
Chain-of-Note 73.4 49.9 7.7 44.8 71.9 82.1 88.0 88.9 63.8
Self-RAG 74.0 48.7 9.1 45.1 73.1 81.989.589.2 64.1
IRCoT 76.0 53.8 8.7 46.4 73.0 82.8 88.1 88.8 65.4
graph-based retrieval
GraphRAG 50.8 44.9 8.7 44.7 72.0 82.2 87.8 89.0 62.1
PathRAG 49.8 43.3 7.3 44.6 66.0 81.9 85.6 83.4 59.1
LightRAG 57.0 44.6 9.0 44.2 71.6 82.2 88.3 88.8 62.1
Biomedical, tool-augmented retrieval
FLARE 72.4 39.2 9.7 51.7 63.4 78.1 74.7 77.9 57.2
BiomedRAG 75.0 47.8 9.5 44.8 72.9 82.4 88.3 89.1 63.8
BioRAG 47.6 37.1 21.4 50.3 72.9 82.2 89.2 89.0 62.0
GeneGPT 75.4 47.0 34.5 50.6 72.8 83.2 88.5 89.0 65.9
BioHarness 76.4 54.1 54.6 60.3 74.9 85.9 89.3 89.7 71.0
†w/ Golden Contextis computed only on benchmarks with dataset-provided gold evidence, namely PubMedQA and BioASQ, covering 5,219
examples. “–” indicates that gold evidence is unavailable or that the corresponding overall score is not reported.
check of the literature retrieval substrate; all quantitative
comparisons are reported in the benchmark results. We set
the grounding thresholdθ gto 0.5 and the confidence threshold
θcto 0.7 in all experiments. Unless otherwise specified, all
non-oracle baselines are implemented under the same retrieval
corpus, embedding model, reranking model, answer backbone,
decoding and formatting strategy, and evaluation pipeline. This
controlled setup ensures that performance differences primarily
reflect evidence assembly and control strategies rather than
platform-level differences.
Benchmark Datasets
The benchmark integrates eight publicly available biomedical QA
datasets, unified under a consistent JSONL schema and evaluation
protocol7. In total, the benchmark contains 19,302 questions
covering seven task types and a broad spectrum of biomedical
subdomains, including molecular biology, clinical medicine, and
genomics. Each dataset contributes distinct evaluation signals:
•Clinical datasets, including MedMCQA (Pal et al., 2022),
MedQA-US, MedQA-TW, and MedQA-CN (Jin et al., 2020),
emphasize diagnostic reasoning and factual recall.
•Literature-based datasets, including BioASQ (Krithara et al.,
2023) and PubMedQA (Jin et al., 2019), assess document-
grounded inference and summarization.
•Genomics-focused datasets, including GeneTuring (Shang
et al., 2025) and SciHorizon-Gene (Qin et al., 2025; Huang
et al., 2026), stress domain-specific knowledge, entity
normalization, and biological reasoning.
7https://huggingface.co/datasets/Shaow/BioHarness_EvalAll datasets are normalized into a unified JSONL format, ensuring
consistent access to question text, task type, answer options when
applicable, and ground-truth annotations.
Baseline Methods
Following the controlled experimental platform described above,
all baselines are evaluated using the same processed corpus,
dense retriever, reranker when applicable, answer backbone,
constrained decoding strategy, and evaluation pipeline. The
compared methods cover retrieval-free, oracle-evidence, dense
retrieval, query augmentation, reranking, reflective retrieval,
graph-based retrieval, and biomedical tool-augmented retrieval
settings.
Retrieval-free and oracle settings.We includeNo-Context
as a retrieval-free baseline andw/ Golden Contextas an oracle-
evidence setting using dataset-provided gold evidence when
available.
Dense retrieval and query augmentation.We compare with
DenseRAG(Karpukhin et al., 2020), andHyDE(Gao et al., 2023).
Reflective and iterative retrieval.We includeDenseRAG+Rerank,
IRCoT(Trivedi et al., 2023),Chain-of-Note(Yu et al., 2024a),
Self-RAG(Asai et al., 2023), andFLARE(Zhang et al., 2024).
Graph-based retrieval.We compare withGraphRAG(Edge
et al., 2024),LightRAG(Guo et al., 2024), andPathRAG(Chen
et al., 2026).
Biomedical and tool-augmented retrieval.We further
compare withBiomedRAG(Li et al., 2025),BioRAG(Wang et al.,
2024), andGeneGPT(Jin et al., 2024).

BioHarness 7
Fig. 5: Score–efficiency comparison on the unified 19,302-question benchmark. The left panel compares BioHarness with all baselines,
and the right panel shows component ablations of BioHarness. Each point reports the pooled score against median per-question latency
on a log-scaled horizontal axis. Bubble area denotes the average total token count per query, and color denotes the average number of
LLM calls per query.
Results
Main Results on the Unified Biomedical QA Benchmark
Table 1 reports the main comparison on eight biomedical QA
benchmarks. BioHarness achieves the best overall performance,
improving the strongest non-oracle baseline from 65.9 to 71.0.
It obtains the best result on seven of the eight datasets,
with MedQA-TW being the only exception where Self-RAG is
slightly higher. The largest gains appear on benchmarks requiring
precise biomedical grounding: BioHarness substantially improves
GeneTuring and SciHorizon, suggesting that adaptive evidence
control is more effective than fixed retrieval or tool-use strategies.
The comparison also shows that stronger retrieval alone is
insufficient. Dense retrieval, reranking, reflective retrieval, and
graph-based retrieval each improve some settings, but their gains
remain uneven across the benchmark. Similarly, biomedical tool-
augmented methods such as BioRAG and GeneGPT improve
entity-centric tasks, but do not consistently generalize to the full
benchmark. These results suggest that biomedical QA benefits
more from adaptive control over retrieval, reranking, structured-
resource invocation, and evidence repair than from any single fixed
retrieval architecture.
The ‘w/ Golden Context’ provides an evidence upper bound
only for PubMedQA and BioASQ: BioHarness is close to the oracle
on PubMedQA but remains farther from it on BioASQ, reflecting
the greater difficulty of multi-format questions that require fine-
grained normalization and synthesis across snippets. In general,
the results support the central design of BioHarness: an accurate
biomedical quality assessment requires not only the retrieval of
relevant documents, but also the assembly and repair of evidence
according to the type of question, the substrate of the biomedical
entity, and the available structured resources.
Efficiency of Grounded Cascade Control
Figure 5 summarizes the score–compute trade-off on the full
benchmark. We measure median end-to-end latency per question,
average token usage, and average LLM calls under the same
evaluation setting as Table 1. For graph-based retrieval methods,latency includes query-time graph construction and reasoning, so
the reported operating point reflects a single-shot deployment
scenario without cross-query caching.
BioHarness lies on the empirical efficient frontier. It achieves
the highest pooled score, improving over the strongest non-oracle
baseline from 65.9 to 71.0, while using a moderate compute
budget. In contrast, several methods that spend substantially
more computation do not obtain comparable scores. GraphRAG,
PathRAG, and LightRAG require many LLM calls and large
token budgets because they construct or traverse query-local graph
structures, yet their scores remain below strong dense, query-
augmentation, or reflective retrieval baselines. This indicates that
the gain of BioHarness is not simply due to spending more
inference compute, but to allocating computation according to
evidence sufficiency.
The ablation panel further clarifies the role of cascade routing.
Removing cascade control, BioHarness −C, forces all questions
into the agent route and gives nearly the same pooled score as
the full pipeline, but shifts markedly toward the high-latency
and high-token regime. The full grounded cascade recovers
the same accuracy level with much lower latency and fewer
tokens by preserving confident fast-path answers and escalating
only uncertain, weakly grounded, or substrate-mismatched
cases. Conversely, disabling REPL iterative reasoning escalation,
BioHarness−A, yields a cheaper but substantially lower-scoring
configuration, confirming that iterative repair is useful when
needed but should not be applied indiscriminately.
Overall, the efficiency analysis supports the central design
choice of BioHarness: literature retrieval, question-conditioned
reranking, biomedical tools, atlas access, and REPL-style repair
should be treated as a staged evidence-allocation policy rather
than as uniformly applied modules. The cascade preserves the
score benefit of iterative repair while avoiding unnecessary agent
execution for questions that can already be resolved by the
fast path with reranked literature and tool-derived grounding.
Similarly, atlas access is router-gated rather than globally enabled:
only a small fraction of benchmark queries are routed to the Atlas
Evidence Layer, indicating that structured measurement queries

8 Meng Xiao et al.
Fig. 6: Component ablation and atlas-routed evidence analysis of BioHarness. (a) Pooled overall score under major component ablations
on the unified benchmark. (b) Distribution of atlas-induced Set-F1 changes on atlas-relevant expression questions, grouped by tissue-
specificity class. (c) Paired comparison between BioHarness and BioHarness −Don atlas-routed cases, where BioHarness −Ddisables
atlas-derived measurement access; points above the diagonal indicate cases improved by atlas evidence. (d) Per-dataset score changes
relative to the full model. (e) Per-task score changes relative to the full model. Here, BioHarness −Adisables REPL iterative reasoning
escalation while retaining fast-path tool-derived grounding, BioHarness −Cremoves cascade routing and forces all questions into the agent
route, BioHarness −Tdisables the Biomedical Tool Layer, BioHarness −Rremoves question-conditioned reranking, and BioHarness −D
disables atlas-derived measurement access.
are rare at the benchmark level but require a dedicated substrate
when they arise. This behavior supports the role of BioHarness as
a substrate-aware harness that selectively allocates computation
and switches evidence substrates, instead of applying all retrieval,
tool, and atlas operations to every question.
Component Ablation Studies
Figure 6 analyzes the contributions of the major retrieval, control,
repair, and atlas components of BioHarness. Disabling REPL
iterative reasoning escalation (BioHarness −A) yields the largest
pooled degradation in the current analysis, reducing the score from
71.0 to 66.3. Disabling the Biomedical Tool Layer (BioHarness −T)
reduces the score to 68.1, while removing question-conditioned
reranking (BioHarness −R) reduces it to 69.0. Removing both
reranking and cascade control (BioHarness −R,−C ) yields 69.3.
Removing cascade control alone (BioHarness −C), which forces allquestions into the agent route, gives a similar pooled score of 70.9
but changes the error profile. This indicates that the grounded
cascade is not only a cost-saving mechanism; it also protects
already-grounded fast-path answers from unnecessary agent-side
rewriting or over-processing.
Panels (d) and (e) show where these changes occur. At
the dataset level, disabling the Biomedical Tool Layer causes
the largest loss on GeneTuring, consistent with the importance
of tool-based alias resolution, canonical entity grounding, and
identifier normalization for gene-centric questions. By contrast,
removing cascade control causes only a small pooled change,
but it slightly degrades GeneTuring, where many questions can
already be answered correctly through fast-path retrieval plus
tool-derived grounding. This supports the intended role of the
grounded gate: it escalates uncertain or substrate-mismatched
cases while preserving confident fast-path answers when additional
agent reasoning is more likely to perturb than repair the answer.

BioHarness 9
Fig. 7: Backbone scaling analysis of BioHarness on a deterministic 10% subsample of the unified benchmark. The top panel reports the
pooled score across seven Qwen3.5 backbones ranging from 0.8B to 122B-A10B. The bottom panel shows paired item-level comparisons
between BioHarness and representative baselines. Blue bars denote items where BioHarness obtains a higher item-level score than the
baseline, and red bars denote the converse.
Disabling REPL iterative escalation mainly affects SciHorizon
and BioASQ, indicating that iterative repair is most useful when
the model must recover or synthesize evidence beyond a single
retrieved passage. At the task level, this degradation is most
pronounced for expression, multi-answer choice, and list-style
questions, which often require evidence aggregation or repeated
checking. Removing reranking also hurts BioASQ and SciHorizon,
and particularly affects list and multi-answer settings, supporting
its role in selecting answer-bearing evidence from topically similar
biomedical abstracts.
Panels (b) and (c) isolate the Atlas Evidence Layer, which
is router-gated and therefore not well summarized by a single
pooled benchmark score. Panel (b) shows that atlas-derived
measurements provide substantial gains for many atlas-relevant
expression questions, especially when tissue-level expression
patterns are broad or under-represented in the retrieved literature.
Panel (c) further compares BioHarness with BioHarness −D
on atlas-routed cases. Most points lie above the diagonal,
indicating that atlas evidence improves the routed examples, while
degradation is rare and concentrated in a small number of cases.
Qualitatively, the paired atlas analysis reveals two dominant
improvement modes and one residual failure mode. For genes with
under-reported literature evidence, atlas measurements recover
missing expression sites, as inGABRA4, where the literature-only
prediction retrieves only brain while atlas evidence completes the
ground truth with cardiac expression. For sparsely characterized
genes, atlas evidence can be decisive:LRRC66receives no usable
tissue assignment from literature retrieval, whereas atlas-derived
measurements recover a gastrointestinal expression pattern with
high Set-F1. The main degradation mode occurs when the
literature already provides a precise restricted answer, but the
atlas introduces broad low-level off-target signals, as inCRP, whereover-trusting noisy atlas breadth expands the answer beyond
liver and reduces Set-F1. Thus, atlas access is most useful when
literature priors are incomplete or weak, but requires conservative
filtering when structured measurements are broader than the
benchmark target.
Overall, the ablation results support the substrate-aware design
of BioHarness. Literature reranking improves answer-bearing
evidence selection, biomedical tools repair entity-level grounding
failures, REPL iterative escalation repairs unresolved cases beyond
the fast path, and atlas-derived measurements supply a targeted
substrate for measurement-centric questions.
Backbone Scaling Analysis
To examine whether the gains of BioHarness depend on a
particular answer-model scale, we re-evaluated BioHarness and
four representative backbone-swappable baselines, DenseRAG,
HyDE, Self-RAG, and BioRAG, on a deterministic 10% subsample
of the unified benchmark in Figure 7.
As shown in the top panel, BioHarness obtains the highest
pooled score at every tested backbone size. The margin is small
at 0.8B, where all methods remain in a low-score regime, but
becomes clearer from 2B onward. Although larger backbones
improve all methods, increasing answer-model capacity alone does
not remove the gap between BioHarness and the representative
retrieval baselines included in this analysis.
The bottom panel provides a paired item-level view of the
same trend. Across all tested backbones and all four baselines,
BioHarness maintains a positive win–loss balance, with more
items improved by BioHarness than items where the corresponding
baseline performs better. This indicates that the subsample-level

10 Meng Xiao et al.
Case I Dynamic Routing PubMedQA (yes/no)
−CAblation Tie Performance / 7.4×Faster
Q:“Cholesterol screening in school children: is family
history reliable to choose the ones to screen?”
A: No
BioHarness’s Routing Pipeline:
△Retrieved evidence (top-ranked literatures):
[PMID 17971187]:“...Despite 64.4% of subjects reporting
family history of hyperlipidemia, no relations between family
history and serum lipid levels were found....”
△Evidence status:answer explicitly supported by retrieved
passage✓
△Draft prediction:“No”confidence>threshold
△Routing decision:skip REPL execution (literature-only
path)
△Final answer:“No”Latency: 17.4 s
BioHarness −C:same final answer (“No”), but still executes
a redundant multi-step agent loop before termination.
Overhead:7.4×Latency: 129 s
Fig. 8: Cascade routing avoids unnecessary REPL-style evidence
repair.
gain is not driven only by a small number of high-scoring outliers,
but reflects consistent item-level improvements.
Together with the results in Table 1, this analysis supports the
claim that the gains of BioHarness do not arise merely from using
a particular answer backbone. Rather, they are consistent with the
intended design of the harness: improving the evidence available
before generation through literature evidence ranking, biomedical
entity grounding, and executable evidence repair.
Mechanistic Analysis of Evidence Assembly
Case I: Retrieval Saturation and Dynamic Escalation.
The case study from Figure 8 illustrates the primary role of
the cascade controller: avoiding unnecessary escalation when the
retrieved literature already provides explicit answer-supporting
evidence. The top-ranked literature passage directly states that
family history was not associated with serum lipid levels, allowing
BioHarness to produce a grounded negative answer from the
literature-only path. The controller therefore terminates after the
confidence and grounding checks are satisfied. In contrast, the
−Cablation disables selective routing by forcing REPL execution.
Although the retrieved evidence is already sufficient, the system
still executes a redundant multi-step repair loop before stopping.
This additional trajectory does not improve the final prediction,
which remains “No”, but increases inference overhead from 17.4,s
to 129,s. Rather than invoking REPL-style repair for every query,
BioHarness escalates only when the retrieved evidence remains
ambiguous, incomplete, or weakly grounded.
Case II: Entity Grounding via Biomedical Tool Calls.
The case study in Figure 9 illustrates a representative biomedical
entity-grounding failure that cannot be resolved through literatureCase II Biomedical Tool GeneTuring (factoid)
−TAblation Rescue Score 0→1
Q:“What is the official gene symbol ofAGTIL?”
(AGTILis a deprecated alias of Agouti Signaling Protein.)
A: ASIP
BioHarness’s Routing Pipeline:
△Retrieved evidence (top-ranked literatures):
PMID Title (excerpt) AGTIL ASIP
2045108 alanine: glyoxylate aminotrans.× ×
9441744 serpin gene cluster chr 14q32× ×
8954797 glycogen debranching (AGL) gene× ×
23946868 GT198 in breast/ovarian cancer× ×
18215257 RGST: rat gene symbol tracker× ×
...
△Evidence status:no retrieved abstracts mention AGTIL
△draft prediction:“unknown”
△Routing decision:Escalate to Biomedical Tool Layer
△REPL trace with Biomedical Tools
>>> gene resolver(‘‘AGTIL’’)
>>> aliases: [AGTI, AGTIL,...] # HGNC/MyGene alias lookup
>>> official symbol: ASIP
△Final answer:“ASIP”Score = 1 15.5 s
BioHarness −T:
△Retrieved evidence (top-ranked literatures): ...
△draft prediction:“unknown” confidence<threshold
△Routing decision:Escalate to Further Literature Search
△REPL trace:
search pubmed(“AGTIL gene”)
lexical reformulations over 8 iterations
no alias bridge discovered in top-K retrieval
Final answer:“Unknown”Score = 0 17.9 s
Fig. 9: Biomedical Tool Layer resolves alias chains that literature
retrieval cannot bridge.
retrieval alone. The query usesAGTIL, a deprecated alias of the
canonical gene symbolASIP. Because this alias is not represented
in the retrieved literature context, dense retrieval remains trapped
in an incorrect lexical neighborhood, returning unrelated genes
and symbol fragments despite repeated retrieval attempts. The
failure is therefore not due to insufficient retrieval depth, but to
the absence of a deterministic alias bridge between the surface
mention and the canonical biomedical entity. Even iterative
literature reformulation in BioHarness −Tremains confined to
lexical search and eventually terminates with “unknown” after
repeated unsuccessful retrieval rounds.
In contrast, the full BioHarness pipeline identifies the lack of
grounded evidence and escalates to REPL-style evidence repair,
where the Biomedical Tool Layer resolves the alias through a gene-
normalization call. Rather than continuing open-ended retrieval,
the system converts the problem into a structured identifier-
resolution task. The resolver maps the deprecated aliasAGTILto
the official HGNC symbolASIP, immediately recovering the correct
answer. This case demonstrates that biomedical QA errors can

BioHarness 11
Case III Atlas Evidence SciHorizon-Gene (expression)
−DAblation Rescue F1 Score 0.00→0.80
Q:“What is the tissue-level expression pattern of theFDXRgene?”
A:adrenal, testis, spleen, colon, ovary, skin, esophagus
BioHarness’s Routing Pipeline:
△Retrieved evidence (top-ranked literatures):
Retrieved Topic PMID
FDXR DNA-damage signaling 12370809, 33113898
Oxidative-stress response 28842474
Mitochondrial apoptosis 11590433, 16528528
...
△Evidence status:The retrieved abstracts only cover FDXR
disease mechanisms
△draft prediction:
“colon and leukocytes” # confidence<threshold
△Router decision:
Escalate to Atlas Evidence Layer
△REPL trace summary (details on the right panel):
load HPA bulk atlas
→readFDXR.tissue expression
→filter bynx≥1.0
△Final answer:adrenal, testis, spleen, esophagus, ovary, skin,
liver, brain(6 / 7 GT recovered) F1 = 0.80Details of BioHarness’s REPL Trace
>>> load atlas("hpa bulk")
>>> gene = "FDXR"
>>> rows = atlas[’genes’][gene] [’tissue expression’]
>>> inspect(rows)
adrenal: 41.2; testis: 14.5; spleen: 6.8
esophagus: 4.1; ovary: 3.9; skin: 3.4;
liver: 2.2; brain: 1.6; colon: 0.7
>>> expressed = []
>>> for tissue, nx in rows:
>>> if nx >= 1.0:
>>> expressed.append(tissue)
>>> return answer(expressed)
["adrenal", "testis", "spleen", "esophagus", "ovary", "skin",
"liver", "brain"] Elapsed: 2.3s
BioHarness −D
The LLM generates the same draft answer from the retrieved evidence, but the BioHarness −Dpipeline lacks an atlas substrate for
structured per-tissue measurements.
Final answer:colon and leukocytes F1 = 0.00
Fig. 10: Atlas-derived evidence provides structured tissue-level measurement grounding absent from the retrieved literature.
originate from entity-normalization failures rather than reasoning
deficiencies. For such questions, additional literature retrieval
provides diminishing returns because the missing link is structural
rather than textual. The Biomedical Tool Layer therefore acts
as an entity-grounding substrate that bridges aliases, deprecated
symbols, and identifier inconsistencies that are difficult to resolve
with retrieval-only pipelines.
Case III: Atlas-Driven Evidence beyond Retrieval.
Figure 10 illustrates a representative failure mode where the
required evidence substrate is structured rather than textual.
Although literature retrieval identifies passages related toFDXR,
the retrieved abstracts primarily discuss DNA-damage signaling,
oxidative stress, and mitochondrial apoptosis rather than tissue-
level expression measurements. As a result, the literature-
conditioned draft prediction remains weakly grounded and
incorrectly infers “colon and leukocytes” from indirect contextual
associations.
The cascade controller detects both the low-confidence
prediction and the mismatch between the retrieved evidence type
and the question requirement. Rather than continuing iterative
literature retrieval, BioHarness escalates to the Atlas Evidence
Layer, which exposes structured tissue-level measurements
through an executable atlas interface. As shown in Figure 10, the
REPL trace loads Human Protein Atlas tissue-expression recordsforFDXR, applies a deterministic atlas threshold, and returns an
atlas-derived tissue set from the structured substrate.
This case highlights an important limitation of retrieval-
only biomedical QA systems: some biomedical questions are
fundamentally not literature-summarization problems. Expression
profiles, atlas measurements, and structured omics annotations
often exist as database-native signals rather than declarative
textual statements in abstracts. The Atlas Evidence Layer enables
BioHarness to switch from unstructured literature reasoning to
executable biological measurement grounding, improving the item-
level score on measurement-centric biomedical QA.
Discussion
This work highlights evidence-substrate mismatch as a key
failure mode in biomedical RAG. BioHarness addresses this
issue by coordinating literature retrieval, biomedical entity
tools, and atlas-derived structured measurements within a staged
evidence assembly framework. The results indicate that selective
escalation is more effective than indiscriminately invoking multi-
step reasoning, because additional execution is useful mainly when
it repairs weakly grounded, uncertain, or substrate-mismatched
evidence. This suggests that future biomedical QA systems should
be designed as substrate-aware evidence harnesses that connect

12 Meng Xiao et al.
Fig. 11: Demo system of BioHarness. (a) The BioHarness landing page provides access to the life science question-answering interface.
(b) The workflow panel visualizes the execution trace of a user query, including query intake, literature chunk search, figure or document
linking, prompt construction, and answer generation. (c) The evidence panel displays retrieved literature snippets with document
identifiers, source links, and relevance scores, enabling users to inspect the evidence associated with the generated answer.
textual evidence with curated biomedical knowledge resources
and biological atlases, rather than treating all evidence as
retrievable text. Future work could improve evidence-sufficiency
estimation, expand coverage of structured biomedical resources,
and provide clearer provenance for answers supported by evidence
from the literature, knowledge-base records, and atlas-derived
measurements.
Availability
We also developed a web-based demo system to illustrate how
BioHarness can be used in an interactive biomedical QA setting.
As shown in Figure 11, the demo exposes both the user-facing
question-answering interface and the internal evidence workflow.
For each query, the system records major execution steps such as
literature chunk search, figure linking, prompt construction, and
answer generation. The retrieved evidence panel further displays
document identifiers, source links, and relevance scores, allowing
users to inspect the supporting literature behind the generated
answer. The purpose of the demo is not to introduce an additional
benchmark, but to demonstrate the practical instantiation of the
proposed LLM harness. A public release is planned following
stability and security testing.
Acknowledgement
This work is partially supported by the National Natural
Science Foundation of China (No.62506351), the Strategic
Priority Research Program of the Chinese Academy of Sciences
(No.XDB1350102), the National Natural Science Foundation of
China (No.92470204), and the Beijing Natural Science Foundation
(No.4254089). The development of this project was supported by
the SciMatrix platform and SciHorizon platform.References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh
Hajishirzi. Self-rag: Learning to retrieve, generate, and critique
through self-reflection. InThe Twelfth International Conference
on Learning Representations, 2023.
Boyu Chen, Zirui Guo, Zidan Yang, Yuluo Chen, Junze Chen,
Zhenghao Liu, Chuan Shi, and Cheng Yang. Pathrag: Pruning
graph-based retrieval augmented generation with relational
paths. InProceedings of the AAAI Conference on Artificial
Intelligence, volume 40, pages 30183–30191, 2026.
The Tabula Sapiens Consortium, Robert C Jones, Jim Karkanias,
Mark A Krasnow, Angela Oliveira Pisco, Stephen R Quake,
Julia Salzman, Nir Yosef, Bryan Bulthaup, Phillip Brown, et al.
The tabula sapiens: A multiple-organ, single-cell transcriptomic
atlas of humans.Science, 376(6594):eabl4896, 2022.
Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex
Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky,
Robert Osazuwa Ness, and Jonathan Larson. From local to
global: A graph rag approach to query-focused summarization.
arXiv preprint arXiv:2404.16130, 2024.
Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan.
Precise zero-shot dense retrieval without relevance labels. In
Proceedings of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), pages
1762–1777, 2023.
Nancy George, Silvie Fexova, Alfonso Munoz Fuentes, Pedro
Madrigal, Yalan Bi, Haider Iqbal, Upendra Kumbham,
Nadja Francesca Nolte, Lingyun Zhao, Anil S Thanki, et al.
Expression atlas update: insights from sequencing data at both
bulk and single cell level.Nucleic Acids Research, 52(D1):
D107–D114, 2024.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tian Ao, and Chao Huang.
Lightrag: Simple and fast retrieval-augmented generation.arXiv
preprint arXiv:2410.05779, 2(3), 2024.

BioHarness 13
Benjamin M Gyori, Charles Tapley Hoyt, and Albert Steppi.
Gilda: biomedical entity text normalization with machine-
learned disambiguation as a service.Bioinformatics advances,
2(1):vbac034, 2022.
Xiaohan Huang, Meng Xiao, Chuan Qin, Qingqing Long, Jinmiao
Chen, Yuanchun Zhou, and Hengshu Zhu. Scihorizon-
gene: Benchmarking llm for life sciences inference from
gene knowledge to functional understanding.arXiv preprint
arXiv:2601.12805, 2026.
Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi
Fang, and Peter Szolovits. What disease does this patient
have? a large-scale open domain question answering dataset
from medical exams.arXiv preprint arXiv:2009.13081, 2020.
Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William Cohen,
and Xinghua Lu. PubMedQA: A dataset for biomedical
research question answering. In Kentaro Inui, Jing Jiang,
Vincent Ng, and Xiaojun Wan, editors,Proceedings of the
2019 Conference on Empirical Methods in Natural Language
Processing and the 9th International Joint Conference on
Natural Language Processing (EMNLP-IJCNLP), pages 2567–
2577, Hong Kong, China, November 2019. Association for
Computational Linguistics. doi: 10.18653/v1/D19-1259. URL
https://aclanthology.org/D19-1259/.
Qiao Jin, Zheng Yuan, Guangzhi Xiong, Qianlan Yu, Huaiyuan
Ying, Chuanqi Tan, Mosha Chen, Songfang Huang, Xiaozhong
Liu, and Sheng Yu. Biomedical question answering: a survey of
approaches and challenges.ACM Computing Surveys (CSUR),
55(2):1–36, 2022.
Qiao Jin, Won Kim, Qingyu Chen, Donald C Comeau, Lana
Yeganova, W John Wilbur, and Zhiyong Lu. Medcpt:
Contrastive pre-trained transformers with large-scale pubmed
search logs for zero-shot biomedical information retrieval.
Bioinformatics, 39(11):btad651, 2023.
Qiao Jin, Yifan Yang, Qingyu Chen, and Zhiyong Lu. Genegpt:
augmenting large language models with domain tools for
improved access to biomedical information.Bioinformatics, 40
(2):btae075, 2024.
Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan,
Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-
Dodds, Nova DasSarma, Eli Tran-Johnson, et al. Language
models (mostly) know what they know.arXiv preprint
arXiv:2207.05221, 2022.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis,
Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih.
Dense passage retrieval for open-domain question answering.
InProceedings of the 2020 conference on empirical methods in
natural language processing (EMNLP), pages 6769–6781, 2020.
Anastasia Krithara, Anastasios Nentidis, Konstantinos
Bougiatiotis, and Georgios Paliouras. Bioasq-qa: A manually
curated corpus for biomedical question answering.Scientific
Data, 10(1):170, 2023.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni,
Vladimir Karpukhin, Naman Goyal, Heinrich K¨ uttler, Mike
Lewis, Wen-tau Yih, Tim Rockt¨ aschel, et al. Retrieval-
augmented generation for knowledge-intensive nlp tasks.
Advances in neural information processing systems, 33:9459–
9474, 2020.
Mengwei Li, Xiaomeng Zhang, Kok Siong Ang, Jingjing Ling,
Raman Sethi, Nicole Yee Shin Lee, Florent Ginhoux, and
Jinmiao Chen. Disco: a database of deeply integrated human
single-cell omics data.Nucleic acids research, 50(D1):D596–
D602, 2022.Mingchen Li, Halil Kilicoglu, Hua Xu, and Rui Zhang. Biomedrag:
A retrieval augmented large language model for biomedicine.
Journal of Biomedical Informatics, 162:104769, 2025.
Chin-Yew Lin. Rouge: A package for automatic evaluation of
summaries. InText summarization branches out, pages 74–81,
2004.
Junyu Luo, Weizhi Zhang, Ye Yuan, Yusheng Zhao, Junwei Yang,
Yiyang Gu, Bohan Wu, Binqi Chen, Ziyue Qiao, Qingqing Long,
et al. Large language model agent: A survey on methodology,
applications and challenges.arXiv preprint arXiv:2503.21460,
2025.
Nicholas Matsumoto, Jay Moran, Hyunjun Choi, Miguel E
Hernandez, Mythreye Venkatesan, Paul Wang, and Jason H
Moore. Kragen: a knowledge graph-enhanced rag framework
for biomedical problem solving using large language models.
Bioinformatics, 40(6):btae353, 2024.
Ankit Pal, Logesh Kumar Umapathi, and Malaikannan
Sankarasubbu. Medmcqa: A large-scale multi-subject multi-
choice dataset for medical domain question answering. In
Conference on health, inference, and learning, pages 248–260.
PMLR, 2022.
CZI Cell Science Program, Shibla Abdulla, Brian Aevermann,
Pedro Assis, Seve Badajoz, Sidney M Bell, Emanuele Bezzi,
Batuhan Cakir, Jim Chaffer, Signe Chambers, et al. Cz
cellxgene discover: a single-cell data platform for scalable
exploration, analysis and modeling of aggregated data.Nucleic
acids research, 53(D1):D886–D900, 2025.
Chuan Qin, Xin Chen, Chengrui Wang, Pengmin Wu, Xi Chen,
Yihang Cheng, Jingyi Zhao, Meng Xiao, Xiangchao Dong,
Qingqing Long, et al. Scihorizon: Benchmarking ai-for-
science readiness from scientific data to large language models.
InProceedings of the 31st ACM SIGKDD Conference on
Knowledge Discovery and Data Mining V. 2, pages 5754–5765,
2025.
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy
Liang. Squad: 100,000+ questions for machine comprehension
of text. InProceedings of the 2016 conference on empirical
methods in natural language processing, pages 2383–2392, 2016.
Xinyi Shang, Xu Liao, Zhicheng Ji, and Wenpin Hou.
Benchmarking large language models for genomic knowledge
with geneturing.Briefings in Bioinformatics, 26(5):bbaf492,
2025.
Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik
Narasimhan, and Shunyu Yao. Reflexion: Language agents with
verbal reinforcement learning.Advances in neural information
processing systems, 36:8634–8652, 2023.
Karthik Soman, Peter W Rose, John H Morris, Rabia E
Akbas, Brett Smith, Braian Peetoom, Catalina Villouta-Reyes,
Gabriel Cerono, Yongmei Shi, Angela Rizk-Jackson, et al.
Biomedical knowledge graph-optimized prompt generation for
large language models.Bioinformatics, 40(9):btae560, 2024.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and
Ashish Sabharwal. Interleaving retrieval with chain-of-thought
reasoning for knowledge-intensive multi-step questions. In
Proceedings of the 61st annual meeting of the association
for computational linguistics (volume 1: long papers), pages
10014–10037, 2023.
Chengrui Wang, Qingqing Long, Meng Xiao, Xunxin Cai,
Chengjun Wu, Zhen Meng, Xuezhi Wang, and Yuanchun Zhou.
Biorag: A rag-llm framework for biological question reasoning.

14 Meng Xiao et al.
arXiv preprint arXiv:2408.01107, 2024.
Chih-Hsuan Wei, Alexis Allot, Po-Ting Lai, Robert Leaman,
Shubo Tian, Ling Luo, Qiao Jin, Zhizheng Wang, Qingyu Chen,
and Zhiyong Lu. Pubtator 3.0: an ai-powered literature resource
for unlocking biomedical knowledge.Nucleic Acids Research, 52
(W1):W540–W546, 2024.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran,
Karthik Narasimhan, and Yuan Cao. React: Synergizing
reasoning and acting in language models.arXiv preprint
arXiv:2210.03629, 2022.
Wenhao Yu, Hongming Zhang, Xiaoman Pan, Peixin Cao, Kaixin
Ma, Jian Li, Hongwei Wang, and Dong Yu. Chain-of-note:
Enhancing robustness in retrieval-augmented language models.
InProceedings of the 2024 conference on empirical methods in
natural language processing, pages 14672–14685, 2024a.
Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan You, Chao
Zhang, Mohammad Shoeybi, and Bryan Catanzaro. Rankrag:Unifying context ranking with retrieval-augmented generation
in llms.Advances in Neural Information Processing Systems,
37:121156–121184, 2024b.
Liang Zhang, Katherine Jijo, Spurthi Setty, Eden Chung, Fatima
Javid, Natan Vidra, and Tommy Clifford. Enhancing large
language model performance to answer questions and extract
information more accurately.arXiv preprint arXiv:2402.01722,
2024.
Yunkun Zhang, Jin Gao, Zheling Tan, Lingfeng Zhou, Kexin Ding,
Mu Zhou, Shaoting Zhang, and Dequan Wang. Data-centric
foundation models in computational healthcare: A survey.ACM
Computing Surveys, 58(11):1–35, 2026.
Juexiao Zhou, Haoyang Li, Siyuan Chen, Zhangtianyi Chen,
Zhongyi Han, and Xin Gao. Large language models in
biomedicine and healthcare.npj Artificial Intelligence, 1(1):
44, 2025.