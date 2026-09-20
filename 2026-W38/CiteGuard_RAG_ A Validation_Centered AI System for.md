# CiteGuard-RAG: A Validation-Centered AI System for Evidence-Grounded Question Answering

**Authors**: Sumit Barua, Guan Hong, Halil Dursunoglu, Charles Rodgers, Alvis Fong

**Published**: 2026-09-14 16:30:54

**PDF URL**: [https://arxiv.org/pdf/2609.15830v1](https://arxiv.org/pdf/2609.15830v1)

## Abstract
Retrieval-augmented generation (RAG) can improve access to complex information; however, retrieving evidence alone does not ensure that answers are grounded, citation-valid, or appropriately refused. This paper introduces CiteGuard-RAG, a validation-centered AI system for evidence-grounded question answering. The system integrates hybrid semantic-lexical retrieval, citation-constrained generation, sentence-level grounding validation, and single-pass regeneration. Validation is used at runtime to determine whether a candidate answer should be accepted, refused, or regenerated before final delivery.
  CiteGuard-RAG is evaluated on 400 questions across a controlled housing-law dataset, PrivacyQA, and CUAD. In the controlled evaluation, it achieves 99.1% retrieval accuracy, 98.3% grounded-answer accuracy, and 98.3% citation validity, with no validation-detected hallucinations. Ablation results show that grounded-answer accuracy drops sharply when validation is removed, even when retrieval accuracy remains unchanged. External evaluation shows that while citation validity remains strong, evidence utilization, span alignment, and refusal calibration become harder under domain shift.
  These findings indicate that trustworthy RAG systems require explicit validation between retrieval and final answer delivery. CiteGuard-RAG provides a practical architecture for linking retrieval, generation, citation checking, abstention, and regeneration in high-stakes information access.

## Full Text


<!-- PDF content starts -->

CiteGuard-RAG: A Validation-Centered AI System
for Evidence-Grounded Question Answering
Sumit Barua, Guan Hong, Halil Dursunoglu, Charles Rodgers,
Alvis Fong
Department of Computer Science, Western Michigan University,
Kalamazoo, MI, 49008, USA.
*Corresponding author(s). E-mail(s): sumit.barua@wmich.edu;
Abstract
Retrieval-augmented generation (RAG) can improve access to complex infor-
mation; however, retrieving evidence alone does not ensure that answers
are grounded, citation-valid, or appropriately refused. This paper introduces
CiteGuard-RAG, a validation-centered AI system for evidence-grounded question
answering. The system integrates hybrid semantic–lexical retrieval, citation-
constrained generation, sentence-level grounding validation, and single-pass
regeneration. Validation is used at runtime to determine whether a candidate
answer should be accepted, refused, or regenerated before final delivery.
CiteGuard-RAG is evaluated on 400 questions across a controlled housing-law
dataset, PrivacyQA, and CUAD. In the controlled evaluation, it achieves 99.1%
retrieval accuracy, 98.3% grounded-answer accuracy, and 98.3% citation validity,
with no validation-detected hallucinations. Ablation results show that grounded-
answer accuracy drops sharply when validation is removed, even when retrieval
accuracy remains unchanged. External evaluation shows that while citation valid-
ity remains strong, evidence utilization, span alignment, and refusal calibration
become harder under domain shift.
These findings indicate that trustworthy RAG systems require explicit validation
between retrieval and final answer delivery. CiteGuard-RAG provides a practical
architecture for linking retrieval, generation, citation checking, abstention, and
regeneration in high-stakes information access.
Keywords:Retrieval-Augmented Generation, Evidence-Grounded Question
Answering, Citation Validation, Hallucination Mitigation
1
arXiv:2609.15830v1  [cs.CL]  14 Sep 2026

1 Introduction
1.1 Background and Motivation
Large language models (LLMs) have improved natural language understanding and
generation, but their use in high-stakes information access remains limited by hal-
lucination, weak source traceability, and unreliable refusal behavior [1, 2]. These
limitations are especially important in legal question answering, where answers must
remain faithful to source documents and unsupported claims can mislead users.
RAG addresses part of this problem by conditioning LLM outputs on retrieved
evidence rather than relying only on parametric memory [3, 4]. However, retrieval
success does not guarantee grounded generation. Even when relevant passages are
retrieved, a model may omit key evidence, overgeneralize from partial context, merge
retrieved text with parametric knowledge, or produce unsupported elaborations [5].
Legal information access further amplifies these challenges because legal corpora
are large, heterogeneous, and structurally complex [6]. Legal texts often contain hier-
archical organization, specialized language, exceptions, and argumentation patterns
that complicate retrieval, chunking, and evidence grounding [7]. As a result, trustwor-
thy legal RAG systems require explicit mechanisms for verifying whether generated
claims are supported by cited evidence, rather than assuming that retrieval alone
ensures reliability [8].
1.2 Literature Review and Research Gaps
1.2.1 Retrieval Strategies
Hybrid retrieval methods that combine lexical methods such as BM25 with dense
semantic retrieval have been widely studied in open–domain question answering, and
the findings show that lexical methods are better at exact term matching, while dense
methods are better at capturing semantic similarity but can retrieve topically related
but non-specific evidence [9, 10]. In the legal domain, transformer–based rerankers
designed for very long queries and documents showed considerable performance gains
while keeping computational efficiency [11]. Retrieval partitioning strategies have also
proven that the organization and incorporation of recovered content greatly improve
the quality of downstream generations [12]. Despite these developments, the interac-
tion of retrieval strategies and grounding quality remains insufficiently understood in
legal RAG systems.
1.2.2 Grounding and Hallucination
A recurrent difficulty for RAG systems is that retrieval quality alone does not guaran-
tee grounded generation; models can hallucinate when retrieved evidence is marginally
relevant or contradicts parametric knowledge [5]. This difficulty is exacerbated by the
“lost-in-the-middle” effect, where LLMs suffer substantial degradation when crucial
information is located in the middle of long input contexts. This phenomenon suggests
that models do not robustly employ all returned evidence [13]. To mitigate halluci-
nations in legal QA, constrained RAG techniques have been developed that leverage
2

aspect-based constraints throughout the retrieval and generation stages, demonstrat-
ing improvements in terms of accuracy and relevance of retrieved documents [14].
Likewise, systems based on natural language inference (NLI) and reinforcement learn-
ing have demonstrated the capacity of entailment-based validation to enhance the
factual consistency of generated text [15]. However, prior work does not have a uni-
fied framework to impose grounding during generation and verify it at fine–grained,
sentence–level resolution simultaneously.
1.2.3 Evaluation
The evaluation of RAG systems is difficult due to the connection between retrieval
and generation. Standard end-to-end metrics often conflate retrieval mistakes with
generation failures, such that system weaknesses are obscured [16, 17]. Benchmark-
ing studies have also demonstrated that RAG systems still suffer from robustness,
information integration, and negative rejection behavior [18]. Recent surveys in the
legal domain indicate the importance of specialized evaluation metrics that mea-
sure faithfulness, citation validity, and hallucination at fine granularity [8]. In-context
retrieval-augmented models show enhanced grounding and source attribution [4], but
there is no systematic evaluation of grounding reliability across system components.
1.2.4 Research Gaps
The existing literature leaves four main gaps. First, it remains unclear how effectively
RAG systems can enforce evidence-grounded and citation-valid answers in high-stakes
information access tasks, especially under domain shift. Second, the effect of lexical,
semantic, and hybrid retrieval strategies on downstream grounding and hallucination
behavior is not yet well characterized. Third, existing validation approaches often
remain post-hoc, answer-level, or evaluation-oriented, leaving limited understanding
of how sentence-level citation and evidence checks can control whether a candidate
answer is accepted, regenerated, or refused before final delivery. Fourth, retrieval,
generation, validation, and regeneration are often evaluated as separate components,
even though their interaction determines final answer reliability.
1.3 Proposed Approach
In this paper, we proposeCiteGuard-RAG, a validation-centered intelligent infor-
mation system for evidence-grounded question answering. The framework combines
page-aware document parsing, paragraph-aware chunking, hybrid semantic–lexical
retrieval, citation-constrained generation, sentence-level grounding validation, and
one-pass validation-guided regeneration. The overall system design is shown in
Figure 1.
The central idea is to treat grounding as a runtime control problem rather than only
as a post-hoc evaluation step. The pipeline first generates an internal candidate answer
and then checks whether its claims are supported by cited retrieved evidence before
any response is delivered. The validation result determines whether the candidate is
accepted as the final answer, revised through one-pass regeneration, or replaced with
a standardized refusal.
3

Fig. 1CiteGuard-RAG pipeline for evidence-grounded question answering. The framework inte-
grates paragraph-aware chunking, hybrid semantic–lexical retrieval with score fusion, citation-
constrained generation, post-generation grounding validation, and one-pass validation-guided regen-
eration.
1.4 Contributions
Motivated by the research gaps above, this study examines how retrieval, generation,
validation, refusal, and regeneration interact in an evidence-grounded RAG system.
The evaluation focuses on whether citation-grounding validation improves answer
reliability, how retrieval and design choices affect grounding behavior, and what fail-
ure modes emerge under controlled and external domain-shift settings. The main
contributions of this work are:
1.CiteGuard-RAG, a validation-centered intelligent information system that con-
nects retrieval, citation-constrained generation, grounding validation, abstention,
and one-pass regeneration for evidence-grounded question answering.
2. A sentence-level citation-grounding validation mechanism that jointly checks cita-
tion validity, lexical evidence overlap, and semantic support before final answer
delivery.
3. Empirical analysis of retrieval–generation–validation interaction, showing that high
retrieval accuracy alone does not guarantee grounded or citation-valid answers.
4. Multi-setting evaluation over 400 questions across a controlled housing-law dataset,
PrivacyQA, and CUAD, allowing in-domain performance to be compared with
external domain and task-shift behavior.
5. Failure case analysis identifying key reliability bottlenecks, including false refusal,
citation/evidence-compliance failures, answer-span mismatch, and evidence-
utilization errors.
2 Methodology
2.1 System Overview
Given a document corpusDand a user queryq, the system retrieves a ranked evidence
setE⊂Dand generates an internal candidate answera c. The candidate answer is
4

not delivered directly. Instead, each substantive claim is checked against the retrieved
evidence and its associated citation. The final output is either a validated answeraor
a standardized refusal. The pipeline is defined as:
D→Parsing→Chunking→Indexing
→Retrieval→Candidate Generation→Validation
→Accept/Regenerate/Refuse.
Algorithm 1 summarizes the proposed citation-constrained RAG workflow, includ-
ing retrieval, evidence filtering, validation, and one-pass regeneration.
Algorithm 1CiteGuard-RAG With Runtime Validation and One-Pass Regeneration
Require:Queryq, document corpusD, retrieverR, generatorG, validatorV
Ensure:Validated final answeraor standardized refusal
1:ParseDinto page-aware text units
2:Construct paragraph-aware evidence chunks{C i}
3:Retrieve hybrid-ranked evidence candidatesE′=R hybrid (q,{C i})
4:Deduplicate candidates and select top-kevidenceE⊆E′
5:ifEis insufficientthen
6:returnstandardized refusal
7:end if
8:Generate internal candidate answera 0=G(q, E) using citation-constrained
prompting
9:Compute validation resultv 0=V(a 0, E)
10:ifv 0passesthen
11:returnvalidated final answera 0
12:else
13:Generate regenerated candidatea 1=G(q, E, v 0) using validation feedback
14:Compute validation resultv 1=V(a 1, E)
15:ifv 1passesthen
16:returnvalidated final answera 1
17:else
18:returnstandardized refusal
19:end if
20:end if
2.2 Document Parsing and Chunk Construction
Input legal documents are processed using page-aware text extraction to preserve
evidence provenance. Each extracted page is represented as:
Pi={text i,pdf pagei,printed pagei},
5

where text iis normalized page text, pdf pageiis the internal PDF page index, and
printed pageiis an optional human-visible page number recovered from the document
when available. When printed page numbers cannot be reliably detected, the system
falls back to the canonical PDF page index. This preserves traceability across retrieval,
citation display, and evaluation.
Documents are segmented using a paragraph-aware chunking strategy. Paragraph
boundaries are preferred when preserved by PDF extraction; otherwise, a sentence-
level fallback is applied using punctuation-based boundary detection. Adjacent text
units are aggregated into chunks under a maximum token budget with overlap to
preserve local context across chunk boundaries. To avoid semantically diffuse evidence,
chunk expansion is constrained by page span, and low-information fragments such as
navigation text, tables of contents, and very short segments are removed. Each final
chunk is stored as:
Cj={text j,chunk idj,pdf rangej,printed rangej}.
2.3 Hybrid Retrieval
The retrieval layer combines dense semantic retrieval and sparse lexical retrieval. Dense
retrieval encodes each chunkC iand queryqusing a sentence-transformer embedding
modelf(·):
ci=f(C i),q=f(q).
Semantic relevance is computed using cosine similarity:
Simsemantic (q, C i) =q·ci
∥q∥∥c i∥.
In parallel, lexical relevance is estimated using BM25:
Score lexical (q, C i) =X
t∈qIDF(t)·tft,i(k1+ 1)
tft,i+k1
1−b+b·|Ci|
avgdl.
Semantic and lexical scores are min–max normalized and fused as:
Score hybrid =αˆssemantic + (1−α) ˆslexical ,
whereαcontrols the semantic–lexical trade-off. This hybrid formulation supports exact
legal phrase matching while retaining semantic flexibility for paraphrased queries.
2.4 Candidate Filtering and Optional Reranking
Retrieved candidates are deduplicated using normalized text similarity to reduce
repeated evidence caused by overlapping chunks and improve context diversity before
generation.
A lightweight heuristic reranker based on query overlap and legal cue words was
evaluated as an optional module. However, because reranking reduced performance
6

in the controlled evaluation, the final system uses deduplicated hybrid retrieval with-
out reranking. Neural cross-encoder rerankers are not used, as this study focuses on
lightweight, locally deployable RAG components.
2.5 Controlled Evidence-Grounded Generation
Retrieved chunks are formatted as numbered source blocks and passed to the
language model using a constrained prompt. The prompt requires the model to
answer only from retrieved sources, cite supporting evidence using inline source
markers, avoid external legal knowledge, and refuse when sufficient evidence is
unavailable. The standardized refusal response is:
The document does not contain sufficient information to answer this question.
This prompting strategy constrains the language model to function as a source-
grounded answer generator rather than a free-form legal advisor. For external datasets
with different document types, the same principle is preserved: answers must be
grounded in retrieved evidence, citations must point to valid retrieved sources, and
unsupported claims trigger refusal or regeneration.
2.6 Post-Generation Validation and Regeneration
The validation layer checks whether each internally generated candidate answer is
traceable to retrieved evidence before any answer is delivered as the final response.
Candidate answers, including regenerated candidates, are decomposed into sentences,
and each sentence is evaluated for citation validity and grounding support. Figure 2
illustrates the sentence-level validation workflow used to route candidate outputs
toward final acceptance, one-pass regeneration, or refusal.
A citation is considered valid only if it refers to one of the retrieved source blocks.
Lexical grounding is computed using token overlap between a candidate sentence and
its cited evidence:
Overlap(s, e) =|Ts∩Te|
|Ts|,
whereT sandT edenote normalized token sets for the candidate sentence and cited
evidence, respectively.
Semantic grounding is computed using sentence embeddings:
Simsemantic (s, e) =s·e
∥s∥∥e∥.
A sentence is considered supported if it satisfies citation validity and passes either the
lexical or semantic grounding threshold. The lexical-overlap and semantic-similarity
thresholds were set to 0.10 and 0.40, respectively, based on pilot validation runs to
balance strict hallucination control against valid paraphrased answers. These values
are tunable implementation parameters rather than fixed theoretical constants.
7

Fig. 2Sentence-level grounding validation pipeline. The workflow verifies citations, lexical overlap,
and semantic similarity against retrieved evidence, then routes the candidate output toward accep-
tance, one-pass regeneration, or refusal.
A candidate response is flagged as hallucinated if it contains one or more
unsupported claims. The supported sentence ratio is computed as:
SupportedRatio =n−u
n,
8

wherenis the total number of candidate sentences anduis the number of unsupported
sentences.
When validation detects a grounding-risk signal, the system performs a single
regeneration attempt using a stricter evidence-grounded prompt. Regeneration is
triggered by invalid citation behavior, weak evidence support, unsupported content,
malformed citation behavior, or refusal despite strong retrieved context. The regener-
ated candidate is accepted only if it improves validation compliance or, when evidence
is insufficient, produces a clean refusal. Regeneration is limited to one pass to control
inference cost and avoid unbounded correction loops. Thus, validation affects both
safety and coverage: unsupported candidate answers are blocked before delivery, but
valid answers may also be rejected when the system behaves conservatively.
2.7 Controlled Evaluation Set
The controlled evaluation set consists of 150 question–answer pairs derived from 18
housing-law documents across multiple U.S. states and cities. Each question is asso-
ciated with gold evidence pages and a refusal label indicating whether the document
contains sufficient information to answer the query. The dataset includes 123 answer-
able questions and 27 unanswerable questions. This design supports evaluation of both
answer generation and abstention behavior.
To reduce formulation bias, the evaluation set combines 100 human-authored ques-
tions with 50 LLM-assisted questions that were manually reviewed, edited, and verified
against the source documents. Questions are stratified across factual, procedural,
reasoning, and edge-case categories. Table 1 reports the question-type distribution.
Table 1Distribution of controlled evaluation question types
Category Count Description
Factual 51 Direct questions asking for explicit information stated in
the source document.
Procedural 40 Questions asking about steps, rights, obligations, dead-
lines, or required actions.
Reasoning 32 Questions requiring interpretation or combination of
related evidence from the retrieved text.
Edge Cases 27 Questions involving exceptions, unavailable evidence,
ambiguous conditions, or refusal behavior.
2.8 External Validation Design
External validation is conducted using two complementary datasets. PrivacyQA [19]
is used as the primary external QA-style validation dataset because it preserves the
natural document-grounded question-answering structure of the controlled housing-
law evaluation while shifting to privacy-policy documents. PrivacyQA consists of user
questions about privacy policies with expert-labeled relevant policy segments. We
evaluate a fixed 200-question subset and treat expert-labeled relevant segments as gold
9

evidence, allowing assessment of retrieval coverage, citation validity, evidence support,
and regeneration beyond the housing-law corpus.
CUAD [20] is used as a supplementary contract-domain stress test rather than as
a direct natural-QA benchmark. CUAD differs from the controlled evaluation because
it asks for contract clause identification and span-like evidence, introducing both legal
domain shift and task-formulation shift. We evaluate a fixed 50-question stratified
subset containing answerable and unanswerable clause-extraction examples to assess
retrieval coverage, citation validity, abstention behavior, hallucination control, and
regeneration under contract-domain shift.
2.9 Implementation Details
The system was implemented as a modular Python pipeline. Dense embeddings were
generated using BAAI/bge-base-en-v1.5 [21], while lexical retrieval used BM25 [22].
Hybrid retrieval combined min–max normalized semantic and lexical scores using a
0.65/0.35 weighting scheme. The final controlled configuration used 600-token chunks
with 120-token overlap, top-k= 8, and up to four retrieved chunks passed to the
generator. The heuristic reranker was disabled in the final system because it reduced
controlled-setting performance.
Generation was performed using locally deployable instruction–tuned models under
deterministic decoding with temperature set to 0. When validation failed, the system
allowed at most one regeneration attempt using validation feedback. For external val-
idation, the same RAG architecture, citation requirements, validation framework, and
one-pass regeneration mechanism were retained, with dataset-specific preprocessing
applied only to align PrivacyQA and CUAD with their native evidence structures.
2.10 Evaluation Protocol
Each controlled housing-law test instance is represented as:
(q, y refuse, G),
whereqis the query,y refuse∈ {0,1}indicates whether the system should abstain, and
Gdenotes the gold evidence pages. For PrivacyQA,Gcorresponds to expert-labeled
relevant policy segments. For CUAD,Gcorresponds to annotated contract evidence
spans or relevant clause text.
Evaluation separates retrieval coverage, answer correctness, citation grounding,
hallucination control, abstention behavior, and regeneration behavior. This separation
allows retrieval failures to be distinguished from generation, validation, and refusal-
calibration failures. The main metric groups are summarized in Table 2.
Two hallucination-related metrics are reported.Hallucination Ratemea-
sures broader answer-level failures, including incorrect, overextended, benchmark-
inconsistent, or evidence-inconsistent final responses under the dataset-specific evalua-
tion criteria.Unsupported Hallucination Rateis narrower and measures final responses
that contain claims not supported by the retrieved and cited evidence. Thus, a response
may count toward the broader hallucination rate when it fails a benchmark answer
10

criterion, while not counting as an unsupported hallucination if its claims remain
grounded in cited evidence.
Table 2Evaluation metric groups
Metric group Metrics and purpose
Retrieval coverage Page Hit@k, Span/Segment Hit@k, and Context Recall@k
measure whether gold evidence is retrieved and included in the
generation context.
Answer correctness Grounded Answer Accuracy and Token F1 measure whether
the final answer is correct, evidence-aligned, and sufficiently
overlaps with gold evidence or expert-labeled segments.
Citation and evidence
supportCitation Validity and Evidence Support Rate measure whether
final answers cite valid retrieved sources and satisfy lexical or
semantic grounding checks.
Hallucination control Hallucination Rate measures broader incorrect, overex-
tended, benchmark-inconsistent, or evidence-inconsistent final
responses. Unsupported Hallucination Rate measures claims
not supported by retrieved and cited evidence.
Abstention behavior True Refusal Rate, False Refusal Rate, and False Answer
Rate measure whether the system appropriately refuses when
evidence is insufficient and avoids answering unanswerable
questions.
Regeneration behavior Regeneration Attempt Rate and Regeneration Accepted Rate
measure how often validation triggers regeneration and how
often the regenerated candidate is accepted as the final output.
3 Results
3.1 Overall System Performance
Table 3 reports the performance of the final selected system configuration on the
controlled housing-law evaluation set.
Table 3Overall system performance
under the final selected configuration
Metric Value
Retrieval Accuracy 99.1%
Grounded Answer Accuracy 98.3%
Citation Validity 98.3%
Evidence Support Rate 98.3%
Hallucination Rate 0.0%
False Refusal Rate 1.8%
True Refusal Rate 96.4%
Regeneration Rate 0.9%
11

The system achieves high retrieval accuracy, grounded answer accuracy, citation
validity, and evidence support. No hallucination is detected under the implemented
validation criteria, while the false-refusal rate remains low. The low regeneration rate
indicates that most candidate answers satisfy the grounding constraints on the first
generation pass.
To assess whether this behavior depends on a specific generator, we also evaluated
Mistral [23] and Qwen [24] under the same retrieval, prompting, and validation con-
figuration. As shown in Table 4, both models achieve identical performance across the
primary reliability metrics.
Table 4Controlled-setting comparison of generation models
Model Grounded Acc. Citation Validity Hallucination True Refusal
Mistral 98.3% 98.3% 0.0% 96.4%
Qwen 98.3% 98.3% 0.0% 96.4%
The identical model-level results suggest that, in the controlled setting, system
behavior is primarily determined by retrieval, prompting, and validation constraints
rather than by generator-specific variability.
3.2 Component and Configuration Analysis
Table 5 compares three embedding models under the same hybrid retrieval configu-
ration. BAAI/bge-base-en-v1.5 achieves the strongest overall performance, with the
highest retrieval accuracy and grounded answer accuracy and the lowest hallucination
rate.
Table 5Embedding model comparison
Embedding Model Retrieval Grounded Hallucination False Refusal
all-MiniLM-L6-v2 [25] 90.8% 89.2% 3.1% 6.2%
intfloat/e5-base-v2 [26] 90.8% 90.8% 3.1%0.0%
BAAI/bge-base-en-v1.5 [21]99.1% 98.3% 0.0%1.8%
Table 6 compares lexical-only, semantic-only, and hybrid retrieval using the selected
BGE embedding model. Hybrid retrieval outperforms both single-mode baselines, sug-
gesting that legal QA benefits from combining exact phrase matching with semantic
similarity.
The validation layer has the largest effect on generation reliability. As shown in
Table 7, disabling validation leaves retrieval accuracy unchanged but substantially
reduces grounded answer accuracy and citation validity while increasing hallucination.
This result is important for legal QA because citation validity and evidence support
cannot be inferred from retrieval success alone; they require explicit post-generation
verification.
12

Table 6Comparison of retrieval strategies
Mode Retrieval Acc. Grounded Acc. Hallucination Exact Hit
Lexical Only 84.6% 83.1% 4.6% 81.5%
Semantic Only 92.3% 90.8% 3.1% 84.6%
Hybrid (0.65/0.35)99.1% 98.3% 0.0% 99.1%
Table 7Effect of validation layer
Setting Retrieval Acc. Grounded Acc. Citation Validity Hallucination
Validation ON 99.1%98.3% 98.3% 0.0%
Validation OFF 99.1% 8.7% 12.3% 18.6%
Reranking was evaluated as an optional retrieval-stage refinement. However, as
shown in Table 8, the heuristic reranker reduced both retrieval accuracy and grounded
answer accuracy while increasing false refusals. The final configuration therefore
disables reranking.
Table 8Effect of reranker
Setting Retrieval Acc. Grounded Acc. False Refusal Exact Hit
Reranker OFF99.1% 98.3% 1.8% 99.1%
Reranker ON 87.7% 86.2% 3.1% 83.1%
Chunk size and hybrid weighting also affect system behavior. Table 9 shows that
a 600-token chunk size provides the best trade-off between context completeness and
retrieval specificity.
Table 9Effect of chunk size
Chunk Size Retrieval Acc. Grounded Acc. False Refusal Hallucination
400 93.8% 92.3%1.5%1.5%
60099.1% 98.3%1.8%0.0%
800 95.4% 93.8% 3.1% 1.5%
Table 10 evaluates different semantic–lexical score combinations within hybrid
retrieval. The 0.65/0.35 configuration achieves the best overall balance, producing the
highest retrieval accuracy and grounded answer accuracy while eliminating detected
hallucination under the validation criteria. More semantic–heavy retrieval at 0.80/0.20
performs strongly but increases hallucination risk, indicating that maximizing seman-
tic dominance does not necessarily yield the safest legal-QA configuration.
13

Table 10Effect of hybrid semantic/lexical weighting
Semantic Lexical Retrieval Acc. Grounded Acc. Hallucination False Refusal
0.80 0.20 96.9% 96.9% 4.6%0.0%
0.65 0.3599.1% 98.3% 0.0%1.8%
0.50 0.50 90.8% 90.8% 1.5% 3.1%
0.35 0.65 89.2% 89.2% 1.5% 1.5%
3.3 External Validation and Stress-Test
Table 11 reports external validation results on PrivacyQA and CUAD. These datasets
evaluate whether the framework maintains retrieval, citation, grounding, refusal, and
regeneration behavior outside the controlled housing-law setting.
Table 11External validation and stress-test results
Metric PrivacyQA CUAD Stress Test
Evaluation Size 200 questions 50 questions
Retrieval Recall@k93.37% 100.00%
Citation Validity 100.00% 100.00%
Evidence Support Rate 81.33% 36.00%
Answer Accuracy 68.50% 54.00%
True Refusal Rate 76.47% 92.00%
False Refusal Rate 18.67% 56.00%
False Answer Rate 23.53% 8.00%
Hallucination Rate 22.00% 26.00%
Unsupported Hallucination Rate 0.00% 6.00%
Regeneration Attempt Rate 30.00% 28.00%
Regeneration Accepted Rate 29.50% 16.00%
On PrivacyQA, the system achieves high retrieval recall, citation validity, and
evidence support, with no unsupported hallucination detected under the validation
criteria. Regeneration is triggered for nearly one-third of the questions, indicating that
runtime validation remains active under external evaluation.
The CUAD stress test shows a larger shift in evidence-support and refusal behavior.
Retrieval recall and citation validity remain high, but evidence support decreases and
false refusal increases. Regeneration is also less often accepted as the final output.
3.4 Failure Case Analysis
To examine residual errors, we analyze representative failures across the controlled
evaluation and external validation. In the controlled setting, residual errors are rare
and primarily involve false refusal or citation/evidence-compliance failures rather than
unsupported generation. Under external validation, errors become more frequent and
dataset-specific: PrivacyQA errors include gold-segment mismatch and false-answer
behavior, while CUAD errors are dominated by over-refusal and clause-span extraction
difficulty.
14

Observed residual errors are grouped into five categories: (i) false refusal, where
the system abstains despite available evidence; (ii) citation or evidence-compliance
failure, where an otherwise supported answer fails strict citation or validation require-
ments; (iii) segment mismatch, where a concise supported answer does not fully match
the benchmark’s longer gold evidence segment; (iv) false answer, where the system
answers despite insufficient evidence; and (v) extraction or span-selection failure,
where relevant evidence is retrieved but not used completely or precisely.
Table 12 provides representative examples of these residual error types.
Table 12Representative residual error cases across controlled and external settings
Setting Type Observed Behavior Cause
Controlled False refusal Evidence is available but
the system abstainsConservative refusal
behavior under validation
Controlled Citation compliance Supported answer fails
strict citation or validation
formattingStrict citation-control
requirements
PrivacyQA Segment mismatch Supported concise answer
does not fully match gold
segmentGold labels are long pol-
icy segments rather than
canonical answers
PrivacyQA False answer System answers when evi-
dence is insufficient for the
queryDifficulty distinguishing
partial evidence from suf-
ficient evidence
CUAD Over-refusal Relevant clause is retrieved
but system abstainsClause-extraction task dif-
fers from natural QA
CUAD Extraction failure Relevant clause is retrieved
but answer is incompleteClause spans are dispersed
across context
4 Discussion
4.1 Findings and Interpretation
The results show that trustworthy evidence-grounded RAG requires more than strong
retrieval. Reliable answer generation depends on the interaction among retrieval,
citation-constrained generation, post-generation validation, and refusal behavior. In
the controlled housing-law evaluation, CiteGuard-RAG retrieves relevant evidence
and constrains generation to cited, source-supported content under the implemented
validation criteria.
The generator comparison further suggests that controlled-setting performance
is not driven mainly by a specific language model. Mistral and Qwen produce
effectively identical results under the same retrieval, prompting, and validation con-
figuration, indicating that the retrieval and validation layers strongly constrain model
behavior. This is important for intelligent information systems where reproducibility,
deployment control, and privacy-sensitive local execution may be required.
External validation provides a more realistic view of system behavior under task
and domain shift. PrivacyQA shows that citation validity and evidence support can
remain strong when the framework is transferred to another document-grounded QA
15

setting, although strict segment-overlap scoring reduces measured answer accuracy.
CUAD presents a harder contract-domain stress test, where retrieval and citation
behavior remain strong but false refusal and evidence-support errors increase. Over-
all, the framework generalizes most strongly in retrieval coverage, citation control,
and unsupported-hallucination prevention, while answer-span selection and abstention
calibration remain more sensitive to domain and task formulation.
4.2 Retrieval and Representation
The retrieval results show that hybrid semantic–lexical retrieval provides the most
reliable evidence access. Lexical retrieval helps preserve exact legal and policy wording,
while semantic retrieval improves matching when questions paraphrase the source text.
The best performance is achieved when these signals are balanced rather than used
independently.
The weighting and embedding ablations further show that retrieval design
affects downstream grounding, not only evidence recall. Excessive semantic weight-
ing increases hallucination risk, while stronger representations improve both retrieval
and answer reliability. This indicates that evidence representation is a core reliability
factor in RAG-based intelligent information systems.
4.3 Validation, Grounding, and Regeneration
The validation layer is the main mechanism for controlling unsupported generation.
When validation is removed, retrieval accuracy remains unchanged, but grounded
answer accuracy and citation validity decrease sharply. This confirms that retrieval
alone is not sufficient for trustworthy evidence-grounded QA.
The controlled 0.0% hallucination rate should be interpreted under the study’s
operational definition: no unsupported claims were detected relative to retrieved and
cited evidence. This does not mean that all possible legal or factual errors were elimi-
nated. The remaining controlled-setting errors were mainly false refusals and citation
or evidence-compliance failures rather than unsupported final answers.
The identical Mistral and Qwen results suggest that, in the controlled setting,
system behavior is primarily shaped by retrieval, prompting, and validation constraints
rather than by generator-specific variability. Regeneration further supports reliability
by giving the system one opportunity to revise candidate answers that fail validation,
although external results show that it cannot fully solve over-refusal or span-selection
errors.
4.4 System Design Trade-offs
The ablation results show that reliability depends on practical design choices. Mod-
erate chunking provides the best balance between preserving enough context and
avoiding irrelevant evidence. Smaller chunks can fragment legal rules, while larger
chunks can add noise.
The reranking experiment also shows that added complexity does not always
improve reliability. The heuristic reranker suppressed some relevant answer-bearing
16

chunks, so the final system uses deduplicated hybrid retrieval without reranking. Over-
all, the results support optimizing the full retrieval–generation–validation pipeline
rather than isolated components.
4.5 External Validation Insights
The external evaluations clarify the type of generalization achieved by CiteGuard-
RAG. PrivacyQA suggests that the pipeline transfers reasonably well to another
document-grounded legal/policy QA setting, particularly in citation control and
unsupported-hallucination prevention. This supports the role of runtime validation
beyond the controlled housing-law corpus.
CUAD reveals a different limitation. Unlike the controlled housing-law and Pri-
vacyQA settings, CUAD is closer to contract clause extraction than natural question
answering. It therefore introduces both domain shift and task-formulation shift.
Although the system retrieves relevant evidence and maintains valid citations, it
becomes more conservative and produces more false refusals. This suggests that clause-
level contract extraction requires more precise span-selection mechanisms than the
current answer-generation pipeline provides. CUAD should therefore be interpreted
as a stress test of robustness under task shift rather than as a direct comparison to
the controlled QA setting.
Together, the two external evaluations show that retrieval coverage and citation
validity generalize more robustly than answer extraction and refusal calibration. This
distinction is important for evidence-grounded RAG systems: a system may retrieve
the right evidence and cite valid sources, yet still fail to produce the exact answer
expected by a benchmark. Reliable high-stakes question answering therefore requires
both evidence access and evidence utilization.
4.6 Limitations and Future Work
Several limitations should be acknowledged. First, the controlled evaluation uses a
curated housing-law corpus, and the external evaluations are limited to PrivacyQA and
CUAD. Although these datasets provide meaningful evidence under domain and task
shift, broader testing across statutes, regulations, case law, administrative guidance,
and legal-aid documents would strengthen generalizability claims.
Second, the external benchmarks differ in answer format. PrivacyQA provides
expert-labeled policy segments rather than canonical short-form answers, while CUAD
focuses on clause-level contract evidence. As a result, answer accuracy and token-
overlap scores should be interpreted alongside citation validity, evidence support,
false-answer behavior, and unsupported hallucination.
Third, the validation framework relies on threshold-based lexical and semantic
similarity checks. These checks help detect unsupported outputs, but they may miss
implicit entailment, complex legal reasoning, or nuanced paraphrases. Future work
should explore stronger validation methods based on natural language inference,
learned evidence verification, and answer-span selection before generation.
Fourth, regeneration is limited to a single attempt for efficiency and reproducibil-
ity. Bounded multi-pass regeneration may improve recovery from difficult generation
17

failures, but it would require careful stopping criteria to avoid excessive inference cost
or unstable outputs.
Finally, the evaluated generators are general-purpose instruction-tuned models
without legal-domain adaptation. Future work should compare domain-specialized
and stronger general-purpose models under the same citation, grounding, refusal, and
regeneration constraints.
5 Conclusion
This work presentedCiteGuard-RAG, a validation-centered intelligent information
system for evidence-grounded question answering combining hybrid semantic–lexical
retrieval, citation-constrained generation, sentence-level grounding validation, absten-
tion, and one-pass validation-guided regeneration. Results from the controlled housing-
law evaluation show that strong retrieval alone is not sufficient for reliable answer
generation; explicit post-generation validation is needed to enforce citation valid-
ity, evidence support, and hallucination control. External evaluation on PrivacyQA
and CUAD further shows that retrieval coverage and citation validity transfer more
robustly than answer extraction and refusal calibration under domain and task shifts.
Overall, the findings suggest that trustworthy RAG systems should be designed as inte-
grated retrieval–generation–validation pipelines, where candidate answers are checked
against cited evidence before final delivery. Future studies should strengthen evidence
utilization through improved span selection, entailment-based validation, and bounded
multi-pass regeneration.
References
[1] Ji Z, Lee N, Frieske R, Yu T, Su D, Xu Y, et al. Survey of Hallucination in
Natural Language Generation. ACM Comput Surv. 2023 Mar;55(12). https:
//doi.org/10.1145/3571730.
[2] Zhao Z, Fan W, Li J, Liu Y, Mei X, Wang Y, et al. Recommender Systems in the
Era of Large Language Models (LLMs). IEEE Trans on Knowl and Data Eng.
2024 Nov;36(11):6889–6907. https://doi.org/10.1109/TKDE.2024.3392335.
[3] Fan W, Ding Y, Ning L, Wang S, Li H, Yin D, et al. A Survey on RAG Meeting
LLMs: Towards Retrieval-Augmented Large Language Models. In: Proceedings of
the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining.
New York, NY, USA: Association for Computing Machinery; 2024. p. 6491–6501.
[4] Ram O, Levine Y, Dalmedigos I, Muhlgay D, Shashua A, Leyton-Brown K, et al.
In-Context Retrieval-Augmented Language Models. Transactions of the Associa-
tion for Computational Linguistics. 2023;11:1316–1331. https://doi.org/10.1162/
tacla00605.
[5] Mallen A, Asai A, Zhong V, Das R, Khashabi D, Hajishirzi H. When Not to Trust
Language Models: Investigating Effectiveness of Parametric and Non-Parametric
18

Memories. In: Rogers A, Boyd-Graber J, Okazaki N, editors. Proceedings of the
61st Annual Meeting of the Association for Computational Linguistics (Volume
1: Long Papers). Toronto, Canada: Association for Computational Linguistics;
2023. p. 9802–9822. Available from: https://aclanthology.org/2023.acl-long.546/.
[6] Niklaus J, Matoshi V, St¨ urmer M, Chalkidis I, Ho D. MultiLegalPile: A 689GB
Multilingual Legal Corpus. In: Ku LW, Martins A, Srikumar V, editors. Proceed-
ings of the 62nd Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers). Bangkok, Thailand: Association for Computational
Linguistics; 2024. p. 15077–15094. Available from: https://aclanthology.org/2024.
acl-long.805/.
[7] Habernal I, Faber D, Recchia N, et al. Mining Legal Arguments in Court Deci-
sions. Artificial Intelligence and Law. 2024;32:1–38. https://doi.org/10.1007/
s10506-023-09361-y.
[8] Hindi M, Mohammed L, Maaz O, Alwarafy A. Enhancing the Precision and
Interpretability of Retrieval-Augmented Generation (RAG) in Legal Technology:
A Survey. IEEE Access. 2025;13:46171–46189. https://doi.org/10.1109/ACCESS.
2025.3550145.
[9] Arivazhagan MG, Liu L, Qi P, Chen X, Wang WY, Huang Z. Hybrid Hierarchical
Retrieval for Open-Domain Question Answering. In: Rogers A, Boyd-Graber J,
Okazaki N, editors. Findings of the Association for Computational Linguistics:
ACL 2023. Toronto, Canada: Association for Computational Linguistics; 2023. p.
10680–10689. Available from: https://aclanthology.org/2023.findings-acl.679/.
[10] Chirkova N, Rau D, D´ ejean H, Formal T, Clinchant S, Nikoulina V. Retrieval-
augmented generation in multilingual settings. In: Li S, Li M, Zhang MJ,
Choi E, Geva M, Hase P, et al., editors. Proceedings of the 1st Workshop on
Towards Knowledgeable Language Models (KnowLLM 2024). Bangkok, Thailand:
Association for Computational Linguistics; 2024. p. 177–188. Available from:
https://aclanthology.org/2024.knowllm-1.15/.
[11] Askari A, Verberne S, Abolghasemi A, Kraaij W, Pasi G. Retrieval for Extremely
Long Queries and Documents with RPRS: A Highly Efficient and Effective
Transformer-based Re-Ranker. ACM Trans Inf Syst. 2024 Apr;42(5). https:
//doi.org/10.1145/3631938.
[12] Wang Z, Teo S, Ouyang J, Xu Y, Shi W. M-RAG: Reinforcing Large Language
Model Performance through Retrieval-Augmented Generation with Multiple Par-
titions. In: Ku LW, Martins A, Srikumar V, editors. Proceedings of the 62nd
Annual Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers). Bangkok, Thailand: Association for Computational Linguistics;
2024. p. 1966–1978. Available from: https://aclanthology.org/2024.acl-long.108/.
19

[13] Liu NF, Lin K, Hewitt J, Paranjape A, Bevilacqua M, Petroni F, et al. Lost
in the Middle: How Language Models Use Long Contexts. Transactions of the
Association for Computational Linguistics. 2024;12:157–173. https://doi.org/10.
1162/tacl a00638.
[14] Nguyen HT, Satoh K. ConsRAG: Minimize LLM Hallucinations in the Legal
Domain. In: Legal Knowledge and Information Systems (JURIX 2024). vol. 385
of Frontiers in Artificial Intelligence and Applications. Amsterdam: IOS Press;
2024. .
[15] Roit P, Ferret J, Shani L, Aharoni R, et al. Factually Consistent Summarization
via Reinforcement Learning with Textual Entailment Feedback. In: Proceedings
of the 61st Annual Meeting of the Association for Computational Linguistics
(ACL 2023). Toronto, Canada; 2023. p. 6252–6272.
[16] Salemi A, Zamani H. Evaluating Retrieval Quality in Retrieval-Augmented Gen-
eration. SIGIR ’24. New York, NY, USA: Association for Computing Machinery;
2024. p. 2395–2400. Available from: https://doi.org/10.1145/3626772.3657957.
[17] Zhang Y, Zhang M, Yuan H, Liu S, Shi Y, Gui T, et al. LLMEval: a preliminary
study on how to evaluate large language models. In: Proceedings of the Thirty-
Eighth AAAI Conference on Artificial Intelligence and Thirty-Sixth Conference
on Innovative Applications of Artificial Intelligence and Fourteenth Symposium
on Educational Advances in Artificial Intelligence. AAAI’24/IAAI’24/EAAI’24.
AAAI Press; 2024. Available from: https://doi.org/10.1609/aaai.v38i17.29934.
[18] Chen J, Lin H, Han X, Sun L. Benchmarking large language models in retrieval-
augmented generation. In: Proceedings of the Thirty-Eighth AAAI Conference
on Artificial Intelligence and Thirty-Sixth Conference on Innovative Applications
of Artificial Intelligence and Fourteenth Symposium on Educational Advances in
Artificial Intelligence. AAAI’24/IAAI’24/EAAI’24. AAAI Press; 2024. Available
from: https://doi.org/10.1609/aaai.v38i16.29728.
[19] Ravichander A, Black AW, Wilson S, Norton T, Sadeh N. Question Answer-
ing for Privacy Policies: Combining Computational and Legal Perspectives. In:
Proceedings of the 2019 Conference on Empirical Methods in Natural Language
Processing and the 9th International Joint Conference on Natural Language
Processing (EMNLP-IJCNLP). Hong Kong, China: Association for Computa-
tional Linguistics; 2019. p. 4949–4959. Available from: https://www.aclweb.org/
anthology/D19-1500.
[20] Hendrycks D, Burns C, Chen A, Ball S. CUAD: An Expert-Annotated NLP
Dataset for Legal Contract Review. NeurIPS. 2021;.
[21] Xiao S, Liu Z, Zhang P, Muennighoff N, Lian D, Nie JY. C-Pack: Packed
Resources For General Chinese Embeddings. arXiv. 2023;https://doi.org/10.
48550/arxiv.2309.07597.
20

[22] Robertson S, Zaragoza H. The Probabilistic Relevance Framework: BM25 and
Beyond. Foundations and Trends in Information Retrieval. 2009;3(4):333–389.
https://doi.org/10.1561/1500000019.
[23] Jiang AQ, Sablayrolles A, Mensch A, Bamford C, Chaplot DS, de Las Casas D,
et al.: Mistral 7B. arXiv. Available from: https://arxiv.org/abs/2310.06825.
[24] Bai J, Bai S, Chu Y, Cui Z, Dang K, Deng X, et al. Qwen Technical Report.
arXiv. 2023;https://doi.org/10.48550/arxiv.2309.16609.
[25] Wang W, Wei F, Dong L, Bao H, Yang N, Zhou M. MiniLM: Deep Self-Attention
Distillation for Task-Agnostic Compression of Pre-Trained Transformers. arXiv.
2020;https://doi.org/10.48550/arxiv.2002.10957.
[26] Wang L, Yang N, Huang X, Jiao B, Yang L, Jiang D, et al. Text Embeddings
by Weakly-Supervised Contrastive Pre-training. arXiv. 2022;https://doi.org/10.
48550/arxiv.2212.03533.
21