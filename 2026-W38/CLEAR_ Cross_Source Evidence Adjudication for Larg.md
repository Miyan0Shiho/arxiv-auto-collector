# CLEAR: Cross-Source Evidence Adjudication for Large Language Models in Medicine

**Authors**: Shuai Wang, Yize Zhao, Qingyu Chen

**Published**: 2026-09-14 20:00:16

**PDF URL**: [https://arxiv.org/pdf/2609.16301v1](https://arxiv.org/pdf/2609.16301v1)

## Abstract
Medical knowledge evolves continuously, whereas the parametric knowledge encoded in large language models (LLMs) is fixed at training time. External retrieval, including retrieval-augmented generation (RAG), can provide access to newly available evidence, but retrieved information may be irrelevant, incomplete, or conflicting. As a result, external retrieval can in turn degrade the factual accuracy and evidence grounding of LLM outputs. To address this challenge, we propose \textbf{CLEAR}, an agentic framework for cross-source evidence adjudication in LLMs in medicine. CLEAR independently generates candidate answers from three complementary pathways---parametric knowledge, locally curated corpora, and dynamically retrieved evidence---reflecting three common sources of information available to LLMs. An aggregation verifier jointly evaluates the candidates, supporting evidence, provenance, and source-quality information to identify agreement and conflict across sources. An adjudication module then determines whether the current conclusion should be preserved or revised through complementary override-guard and challenge-audit mechanisms, while unresolved conflicts trigger targeted follow-up search and re-adjudication.

## Full Text


<!-- PDF content starts -->

CLEAR: Cross-Source Evidence Adjudication for Large Language
Models in Medicine
Shuai Wang1Yize Zhao1,∗Qingyu Chen2,∗
1Department of Biostatistics, Yale School of Public Health, Yale University, New Haven, CT, USA
2Department of Biomedical Informatics and Data Science, Yale School of Medicine, Yale University,
New Haven, CT, USA
Abstract
Medical knowledge evolves continuously, whereas the parametric knowledge encoded in
large language models (LLMs) is fixed at training time. External retrieval, including retrieval-
augmented generation (RAG), can provide access to newly available evidence, but retrieved
information may be irrelevant, incomplete, or conflicting. As a result, external retrieval can in
turn degrade the factual accuracy and evidence grounding of LLM outputs. To address this
challenge, we proposeCLEAR, an agentic framework for cross-source evidence adjudication in
LLMs in medicine. CLEAR independently generates candidate answers from three complementary
pathways—parametric knowledge, locally curated corpora, and dynamically retrieved evidence—
reflecting three common sources of information available to LLMs. An aggregation verifier jointly
evaluates the candidates, supporting evidence, provenance, and source-quality information to
identify agreement and conflict across sources. An adjudication module then determines whether
the current conclusion should be preserved or revised through complementary override-guard
and challenge-audit mechanisms, while unresolved conflicts trigger targeted follow-up search and
re-adjudication.
We evaluate CLEAR on ten public benchmarks spanning multiple-choice and free-text
settings using representative commercial and open-weight backbones (o3-mini and Qwen-3.5-
9B). We compare CLEAR with direct inference and established RAG baselines under matched
backbone settings. CLEAR achieves competitive performance across the ten benchmarks, with
the largest gains observed in settings where direct inference or conventional retrieval provides a
weaker baseline. In settings where parametric knowledge already supports strong performance,
CLEAR generally remains competitive, whereas additional retrieval does not consistently improve
performance. With Qwen-3.5-9B, the best CLEAR configurations improve performance by
11.30 percentage points on NEJM-QA and 13.69 percentage points on MedRBench over the
strongest evaluated baselines. Ablation studies further show that the full cross-source framework
achieves stronger aggregate performance than individual evidence-source settings, while targeted
follow-up search provides additional gains on unresolved cases. These findings suggest that an
agentic framework can benefit LLMs not only by acquiring new evidence, but also by explicitly
adjudicating conflicts across heterogeneous knowledge sources.
1 Introduction
Large language models (LLMs) have shown remarkable potential across a wide range of knowledge-
intensive tasks, demonstrating strong capabilities in language understanding, question answering,
and complex reasoning Brown et al. [2020], Chowdhery et al. [2023], Touvron et al. [2023], Achiam
*Co-corresponding authors:yize.zhao@yale.eduandqingyu.chen@yale.edu
1
arXiv:2609.16301v1  [cs.AI]  14 Sep 2026

et al. [2023]. By acquiring broad world knowledge from large-scale corpora during pretraining,
these models provide a general foundation that can be further adapted to downstream domains.
These capabilities have extended to medicine, where medically adapted LLMs have shown promise
in medical knowledge and reasoning, disease diagnosis, and patient communication Tian et al.
[2024], Liu et al. [2025a], Chen et al. [2025, 2026]. Taking medical question answering as an
example, Med-PaLM achieved 67.6% accuracy on MedQA and increased the proportion of long-form
answers judged to be aligned with scientific consensus from 61.9% for Flan-PaLM to 92.6% Singhal
et al. [2023]. Med-PaLM 2 further increased MedQA accuracy to 86.5%, approaching expert-level
performance on medical licensing examination questions Singhal et al. [2025]. Open-weight medical
LLMs have likewise demonstrated competitive performance across multiple medical benchmarks,
indicating that strong medical reasoning capabilities are not limited to proprietary models Chen
et al. [2023], Bolton et al. [2024], Xie et al. [2025a].
However, the medical knowledge encoded in LLM parameters is inherently bounded by the data
available during training. After training is completed, a model does not automatically incorporate
evidence published thereafter Dhingra et al. [2022]. Medical evidence, by contrast, evolves rapidly
through newly published clinical trials, updated practice guidelines, new drug approvals, and
emerging safety signals. In 2024, 289 clinical practice guidelines were published, representing a
29% increase from 2023 Guideline Central [2025]. PubMed now contains more than 40 million
biomedical citations, with more than one million new records added annually National Library of
Medicine [2024], Kim et al. [2026]. Consequently, LLMs may generate answers that are consistent
with the evidence available at the time of training but are no longer aligned with the latest clinical
evidence or recommendations. This mismatch may be particularly consequential in high-stakes
medical applications, for example, when a model generates treatment recommendations that conflict
with updated clinical guidelines Artsi et al. [2025], Guan et al. [2026], Wang et al. [2026b].
To address this challenge, previous studies have commonly adopted retrieval-augmented gen-
eration (RAG), which supplements the knowledge encoded in model parameters with external
evidence retrieved at inference time Amugongo et al. [2025], Liu et al. [2025b], Yang et al. [2026].
By retrieving relevant information from biomedical literature, medical textbooks, clinical guidelines,
and other curated knowledge resources, RAG enables LLMs to access specialized or newly published
medical evidence without repeatedly retraining the underlying model Zakka et al. [2024], Xiong
et al. [2024a]. Recent studies, however, have identified important limitations of RAG in medical
applications Kim et al. [2025], Xie et al. [2025b], Wong et al. [2025]. Retrieval may return irrelevant,
low-quality, or question-misaligned passages. Moreover, LLMs may fail to identify or correctly use
relevant evidence even when it has been successfully retrieved. These failures can weaken evidence
attribution, reduce factuality, and degrade the accuracy of the final output Kim et al. [2025].
These limitations point to a broader challenge: in medicine, LLMs must reason over multiple
knowledge and evidence sources that differ in coverage, timeliness, authority, and reliability. We
broadly categorize these sources into three types. First,parametric knowledgerefers to information
acquired by an LLM during pretraining or post-training. It provides broad medical priors but
may be incomplete, outdated, or insufficiently specific to the question at hand. Second,locally
curated evidence, such as institutional documents or medical textbooks, may be authoritative and
task-specific but is constrained by the coverage and time of corpus construction. Third,dynamic
evidence, such as information obtained through query-time web search, may capture recent or
context-specific developments but can vary substantially in relevance, credibility, and quality.
These sources may provide complementary support, but they may also lead to conflicting
conclusions. Prior studies have shown that effectively coordinating internal knowledge and retrieved
evidence remains an open challenge Xiong et al. [2024a], Tan et al. [2025], Guan et al. [2026]. For
example, a recent analysis found that conflicts between internal and external knowledge occurred
2

Figure 1: Overview of CLEAR. The framework independently generates candidate answers from
parametric knowledge, locally curated evidence, and dynamically retrieved evidence; aggregates and
adjudicates information across sources; and produces task-specific outputs.
in 19.2% of evaluated cases. Among these conflicts, internal knowledge was correct in 47.4% of
cases, whereas external evidence was correct in 52.6% Wang et al. [2025b]. These findings suggest
that neither source can be trusted by default. Recent extensions of RAG have introduced adaptive
retrieval, query reformulation, and follow-up question generation, allowing models to determine
whether additional retrieval is needed and how the search process should proceed Sohn et al. [2025],
Xiong et al. [2024b]. However, these methods primarily focus on whether and how to retrieve
additional evidence rather than on resolving conflicts among heterogeneous sources after retrieval. A
central challenge therefore remains: how to adjudicate heterogeneous sources and determine whether
the system should preserve an existing conclusion, revise it in light of newly acquired evidence, or
seek additional evidence when the conflict remains unresolved.
To address this challenge, we proposeCLEAR, an agentic framework for cross-source evidence
adjudication in LLMs in medicine. CLEAR generates source-specific candidate answers from
parametric knowledge, locally curated evidence, and dynamically retrieved evidence and explicitly
resolves their agreement and conflicts before producing the final response. CLEAR first constructs
three parallel reasoning pathways, allowing each source to independently contribute a candidate
answer and its supporting evidence. An aggregation verifier then reasons over the resulting evidence
bundle, determines whether the sources agree, conflict, or remain insufficiently supported, and assigns
an appropriate routing decision. When dynamic evidence challenges an agreement between the
parametric and local pathways, an adjudication module further determines whether the new evidence
is sufficiently relevant and reliable to justify revising the existing conclusion. This module combines
an override guard, which prevents weak or misaligned dynamic evidence from inappropriately
overturning a well-supported consensus, with a challenge audit, which allows credible new evidence
3

to correct a consensus that may be outdated or incomplete. When the available evidence cannot
resolve the conflict, the framework escalates the case to targeted follow-up search before making the
final decision.
We systematically evaluate CLEAR on ten public benchmarks spanning medical question
answering, clinical reasoning, and broader knowledge and reasoning tasks using representative
commercial and open-weight backbones. CLEAR remains competitive in settings where Direct
inference or conventional retrieval already performs strongly, while achieving substantial gains in
several other settings. In particular, with Qwen-3.5-9B, the best CLEAR configurations improve
performance by 13.69 percentage points on MedRBench and 11.30 percentage points on NEJM-
QA over the strongest evaluated baselines. These findings suggest that effective use of external
evidence requires not only retrieval, but also explicit adjudication of whether to preserve an existing
conclusion, revise it, or seek additional evidence. We also publicly release the implementation code
to support reproducibility and future research.
2 Method
2.1 Framework Overview
Figure 1 illustrates CLEAR. Given a query q, such as a medical question, the framework considers
three complementary sources of knowledge and evidence: (1) parametric knowledge encoded in the
backbone model, (2) evidence retrieved from locally curated corpora, and (3) dynamic evidence
acquired at inference time from external information sources, such as web search. Together, these
sources capture a common practical setting in which a model must combine its internal knowledge
with controlled local resources and newly acquired external information Wang et al. [2025c], Zhang
et al. [2023].
CLEAR comprises three stages. First, a multi-path generation module independently constructs
candidate answers from the parametric, local, and dynamic pathways. Maintaining separate pathways
preserves the provenance of each candidate and makes cross-source agreement and disagreement
observable. Second, an aggregation verifier assembles the candidate answers, retrieved evidence,
and available source-quality information into a unified evidence bundle. It evaluates the support
for each candidate and assigns a routing decision according to whether the sources agree, conflict,
or remain insufficient to support a conclusion. Third, an adjudication module determines whether
the current conclusion should be preserved, revised in light of dynamic evidence, or subjected to
targeted follow-up search.
CLEAR is conservative by design. Dynamic evidence is not preferred solely because it is newly
retrieved. When the parametric and local pathways agree, their consensus is preserved unless
the dynamic pathway provides sufficiently relevant and reliable evidence that directly resolves the
disagreement. Conversely, CLEAR does not assume that an existing consensus is necessarily correct.
A challenge audit allows credible dynamic evidence to revise a conclusion that may be outdated,
incomplete, or insufficiently sensitive to the specific context. When the available evidence cannot
resolve the conflict, the system performs a bounded follow-up search before producing the final
response. The individual components are described below.
2.2 Source-Specific Candidate Generation
Given the query q, CLEAR generates three source-specific candidate answers. Unless otherwise
specified, all candidate-generation pathways use the same backbone model M, thereby reducing
differences attributable to model capability and allowing the pathways to be compared primarily
4

with respect to the information available to each. Variants that introduce an additional model for
evidence assessment are described separately in the experimental setup.
Parametric pathway.The parametric pathway generates an answer using only the knowledge
encoded in the backbone model:
aparam =M(q).(1)
This pathway provides a retrieval-free reference and preserves the model’s learned medical priors.
It is particularly important when the query is already well supported by parametric knowledge and
additional retrieval would be redundant or potentially distracting Shi et al. [2024], Kim et al. [2025].
Local-evidence pathway.Let Klocaldenote a locally indexed corpus or collection of corpora, and
letRdenote a retrieval function over this collection. The retrieved local evidence, its provenance,
and the corresponding candidate answer are defined as
Elocal, Plocal=R(q, K local), a local=M(q, E local),(2)
where Plocalrecords available provenance information associated with the retrieved local evidence,
such as document identifiers or retrieval sources.
This pathway follows established RAG implementations Xiong et al. [2024a], Liu et al. [2025b].
The proposed framework is agnostic to the implementation of Rand can therefore support sparse,
dense, hybrid, or other retrieval approaches. The local pathway represents settings in which evidence
is curated, controlled, domain-specific, or institution-specific, but potentially limited by the coverage
and update time of the indexed corpus.
Dynamic-evidence pathway.The dynamic pathway acquires evidence from an external infor-
mation spaceK dynamic at inference time:
Edynamic , Pdynamic = Search(q, K dynamic ),(3)
where Pdynamic records the provenance of the retrieved external evidence. The framework further
assesses the quality of the retrieved evidence:
Sdynamic = AssessQuality (q, E dynamic , Pdynamic ), a dynamic =M(q, E dynamic ),(4)
where Sdynamic summarizes source-quality information derived from the retrieved evidence, such
as assessments based on source authority, recency, and content type. Dynamic retrieval may be
implemented using web search, literature search, external databases, or other query-time information
tools.
Each pathway produces a source-specific candidate record:
ci= (a i, Ei, Pi), i∈ {param,local,dynamic},(5)
where Pirecords the provenance of the candidate and its supporting evidence when available.
For the parametric pathway, Eparam =∅because its supporting knowledge is implicit in the model
parameters, and Pparam identifies the parametric pathway and backbone model. For the local and
dynamic pathways,E icontains the retrieved evidence andP irecords its retrieval provenance.
The resulting candidate set is
5

C={c param , clocal, cdynamic }.(6)
The purpose of multi-path generation is not to assume that using more sources necessarily im-
proves performance. Instead, it exposes source-specific conclusions and their supporting information
so that the subsequent modules can explicitly evaluate agreement, disagreement, and evidential
sufficiency.
2.3 Cross-Source Aggregation and Adjudication
2.3.1 Aggregation Verifier
The candidate pathways provide complementary answers, but they do not by themselves determine
which answer should govern the final response. The aggregation verifier therefore constructs a
unified evidence bundle:
B= Aggregate (q,C, S dynamic ).(7)
The bundle Bcontains the original query, the three candidate answers, the retrieved local and
dynamic evidence, the provenance of each candidate, and the available source-quality information.
The aggregation verifierVreasons over this bundle and returns a structured preliminary decision:
o0=V(B) = (ba 0, z0, j0, γ0),(8)
whereba0is the preliminary answer, z0is the routing decision, j0is the supporting rationale,
andγ0is a confidence signal associated with the decision. The confidence signal is not treated as a
calibrated probability.
The routing decision belongs to the following set:
Z={consensus,keep parametric,keep local,
accept dynamic,needs more search}.(9)
The routes have the following interpretations:
•consensus : At least two pathways converge on the same conclusion, and the available
supporting evidence does not reveal a material unresolved contradiction.
•keep parametric : The parametric answer is better supported than the alternatives, and the
retrieved evidence does not justify revising it.
•keep local : The locally grounded answer is better supported by the curated evidence than
the parametric or dynamic alternatives.
•accept dynamic : The dynamic answer is supported by sufficiently relevant and reliable
external evidence and appears to justify revising the current conclusion.
•needs more search : The available evidence is insufficient, internally inconsistent, or unable
to distinguish reliably among competing conclusions.
The aggregation verifier therefore goes beyond simple answer aggregation or majority voting.
It jointly evaluates the provenance, relevance, and support associated with each candidate and
determines whether the observed disagreement can be resolved using the current evidence bundle.
6

2.3.2 Adjudication of Cross-Source Conflicts
The adjudication module is invoked when the verifier identifies a meaningful conflict or when the
preliminary route may require revising an existing conclusion. A particularly important conflict
pattern is
aparam =alocal̸=adynamic ,(10)
in which the parametric and local pathways agree while the dynamic pathway proposes a different
conclusion. This pattern is inherently ambiguous. The dynamic pathway may have identified
newer or more context-specific evidence, but it may also have retrieved irrelevant, unreliable, or
question-misaligned information. The adjudication module therefore evaluates both the risk of an
inappropriate override and the risk of preserving an outdated or incomplete consensus.
Override guard.The override guard evaluates cases in which the preliminary decision favors
accept dynamic , particularly when the dynamic candidate conflicts with an agreement between the
parametric and local pathways or when all three pathways disagree. The guard permits an override
only when the dynamic evidence:
1. directly addresses the query and the observed disagreement;
2. originates from sufficiently credible and relevant sources;
3. provides explicit support for the dynamic conclusion; and
4. explains why the competing conclusion is less appropriate.
If these conditions are not adequately supported, the dynamic candidate is not immediately
accepted. The system either preserves the better-supported parametric or local conclusion or
initiates targeted follow-up retrieval. This mechanism reduces the risk that weak, unreliable, or
misaligned external evidence will overturn a stronger conclusion.
Challenge audit.The challenge audit addresses the opposite failure mode. A parametric–local
consensus may itself be outdated, incomplete, or insensitive to the specific clinical context. Therefore,
when the preliminary decision favors preserving the existing conclusion but the dynamic pathway
raises a plausible and well-supported challenge, the audit explicitly re-examines the consensus.
The challenge is accepted only when the dynamic evidence satisfies the specified relevance and
quality criteria, directly addresses the disagreement, and provides sufficient support for the revised
conclusion. Otherwise, the original conclusion is preserved. The challenge audit therefore prevents
agreement between the parametric and local pathways from being treated as automatically correct.
Together, the override guard and challenge audit balance two competing risks: allowing unreliable
dynamic evidence to inappropriately overturn a well-supported conclusion, and allowing an outdated
consensus to suppress credible and more current evidence.
2.3.3 Targeted Follow-Up Search
When the current evidence bundle cannot resolve the disagreement, the framework generates a
targeted follow-up query conditioned on the unresolved conflict:
eq=Q follow(q, B, o 0),(11)
7

where Qfollow identifies the specific evidential gap, competing claims, or clinical distinction
requiring clarification. Additional evidence and its provenance are then acquired using the targeted
query:
Efollow, Pfollow = Search(eq, K dynamic ).(12)
The framework assesses the quality of this additional evidence as
Sfollow = AssessQuality (eq, E follow, Pfollow),(13)
and merges the new evidence with the existing evidence bundle:
B′= Merge (B, E follow, Pfollow, Sfollow).(14)
The aggregation verifier and adjudication module are subsequently applied to the updated
bundle B′. Follow-up retrieval is bounded rather than open-ended and is used only when the current
evidence is insufficient to justify either preserving or revising the conclusion. Unlike unrestricted
repeated search, the follow-up query is directed at the specific conflict identified during adjudication.
The adjudication process returns a structured decision object:
d= (a⋆, z⋆, j⋆, γ⋆, B⋆),(15)
where a⋆is the selected answer, z⋆is the final route, j⋆is the adjudication rationale, γ⋆is the
final confidence signal, andB⋆is the final evidence bundle after any follow-up search.
2.4 Task-Specific Response Generation
The cross-source adjudication procedure above is shared across different tasks. Once the structured
decision dhas been obtained, a task-specific adapter converts the adjudicated conclusion and its
supporting evidence into the output format required by the target task.
Free-text medical tasks.For an open-ended task, the final response is generated as
rtext=G text(q, a⋆, j⋆, B⋆).(16)
The response generator does not independently repeat retrieval or overturn the adjudication
result. Its role is to express the selected conclusion and supporting evidence in a medically grounded
and task-appropriate form. Depending on the application, the response may include a diagnosis,
treatment recommendation, clinical explanation, or patient-facing answer.
Multiple-choice tasks.For a multiple-choice task with an admissible option set O, the task-
specific adapter maps the adjudicated conclusion to one or more valid answer options:
bO=T MCQ(q,O, d), bO ⊆ O.(17)
For single-answer tasks, |bO|= 1; for multiple-answer tasks, bOmay contain more than one
admissible option.
When the initial evidence is insufficient to distinguish among competing answer options, the
targeted follow-up query may be conditioned on those options so that the search directly evaluates
the relevant clinical distinctions. This option-aware retrieval occurs before the final adjudication
decisiondis produced and therefore does not constitute a separate decision process.
8

After adjudication, a format gate converts the selected option or options into the representation
required by the benchmark:
rMCQ = FormatGate( bO,O).(18)
The format gate does not alter the medical conclusion. It only ensures that the output contains
the appropriate number and representation of admissible options, thereby supporting both single-
answer and multiple-answer tasks.
2.5 Implementation Details
We instantiate CLEAR with GPT-o3-mini and Qwen-3.5-9B as representative commercial and
open-weight backbones, respectively. In the primary setting, the same backbone is used throughout
candidate generation, source-quality assessment, aggregation verification, adjudication, and final
response generation. Full model configurations and prompting details are provided in Appendix A.
For candidate generation, the parametric pathway directly queries the backbone without external
retrieval. The local pathway retrieves evidence from MedCorp Xiong et al. [2024a], following
the same corpus setting as the MedRAG baselines, using BM25 with the top k= 16 retrieved
snippets. The dynamic pathway obtains external evidence through web search. The primary
experiments use GPT-4o’s web-search capability for dynamic evidence acquisition. As a secondary
implementation-diversity experiment, we additionally instantiate the dynamic pathway using Tavily
on the NEJM-QA Internal Medicine subset (Appendix A), providing preliminary evidence that
CLEAR can be implemented with an alternative web-search backend.
Retrieved dynamic evidence is assigned categorical source-quality labels (high, medium, or low)
based on source authority, recency, and content type. These labels constitute Sdynamic and are
provided to the aggregation verifier and adjudication modules together with the candidate answers,
retrieved evidence, and provenance information. The aggregation verifier produces a preliminary
answer and routing decision. When dynamic evidence challenges another pathway, the adjudication
module applies two complementary mechanisms: anoverride guard, which verifies a verifier-proposed
dynamic override before accepting it, and achallenge audit, which allows rejected dynamic evidence
to challenge an existing parametric–local consensus when sufficiently strong supporting evidence is
available. Detailed triggering conditions and decision rules are provided in Appendix A.
Finally, task-specific adapters convert the adjudicated conclusion into benchmark-compatible out-
puts for multiple-choice tasks or evidence-grounded responses for free-text tasks, without introducing
additional retrieval or modifying the adjudication result.
2.6 Data and Evaluation
We evaluate CLEAR on ten public reasoning benchmarks, including eight medical question-answering
or clinical reasoning benchmarks and two broader knowledge and reasoning benchmarks. The
evaluation suite was selected to balance three considerations: comparability with prior work, task
difficulty, and diversity in answer format. Collectively, the benchmarks span multiple-choice and
free-text tasks as well as general and specialty-level medical knowledge.
Due to space constraints, we focus the main-text analysis on MedQA, PubMedQA, NEJM-QA,
MedRBench, and HealthBench. We additionally evaluate the framework on MedBullets Chen et al.
[2025], MedExQA Kim et al. [2024], AfriMed-QA Nimo et al. [2025], MMLU Hendrycks et al. [2020],
and MMLU-Pro Wang et al. [2024]. The magnitude of improvement varies across datasets and
backbone models; complete results on the five additional benchmarks are provided in Appendix A.1.
9

Multiple-choice benchmarks.We evaluate multiple-choice question answering on MedQA,
PubMedQA, and NEJM-QA. MedQA Jin et al. [2021] is a widely used benchmark based on
United States Medical Licensing Examination-style questions. Its English test set contains 1,273
questions, each with four answer options, and primarily evaluates professional medical knowledge
and exam-style clinical reasoning.
PubMedQA Jin et al. [2019] is constructed from biomedical research questions and corresponding
PubMed abstracts. Each question is answered asyes,no, ormaybebased on the associated abstract,
making the benchmark more focused on interpreting biomedical evidence than on isolated factual
recall. We use its 500 expert-labeled test questions.
NEJM-QA Wang et al. [2025a] is constructed from official board residency examinations released
through the NEJM-AI collection. It contains 655 questions spanning five specialties: general surgery,
internal medicine, psychiatry, pediatrics, and obstetrics and gynecology. Relative to MedQA and
PubMedQA, NEJM-QA emphasizes more specialized medical knowledge and provides a setting
with multiple clinically plausible alternatives. Some NEJM-QA questions admit multiple correct
answers; for these questions, a prediction is considered correct only when the complete predicted set
exactly matches the reference answer set.
We retain MedQA and PubMedQA not because they are expected to benefit most from retrieval,
but because they represent standard evaluation settings in which strong parametric models and
conventional local retrieval already perform well. These benchmarks therefore test whether the
proposed framework can preserve performance when additional external evidence is unnecessary or
potentially distracting.
Free-text benchmarks.We also evaluate CLEAR on free-text benchmarks MedRBench and
HealthBench. MedRBench Qiu et al. [2025] contains 1,453 structured patient cases with reference
reasoning derived from open-access clinical case reports. The benchmark spans 13 body systems
and 10 specialties and includes cases involving rare diseases. Following the benchmark protocol, we
evaluate the diagnosis and treatment stages using its automated reasoning evaluator.
HealthBench Arora et al. [2025] contains 5,000 realistic multi-turn health conversations graded
using conversation-specific rubrics developed by 262 physicians. Rather than requiring selection from
fixed answer options, HealthBench evaluates open-ended response quality across criteria including
clinical accuracy, completeness, safety, and communication. We use the HealthBench Hard subset,
comprising 1,000 queries.
Additional benchmarks.To assess robustness across a broader range of reasoning settings,
we additionally evaluate CLEAR on five benchmarks. MedBullets contains USMLE Step 2 and
Step 3-style multiple-choice questions with expert-written explanations. MedExQA covers multiple
medical specialties and provides multiple reference explanations for each question. AfriMed-QA is
a Pan-African, multi-specialty medical question-answering benchmark constructed across medical
institutions in multiple African countries. MMLU evaluates multitask knowledge across a broad
range of academic and professional subjects, while MMLU-Pro provides a more challenging reasoning-
oriented extension with a larger answer space. Detailed dataset statistics and complete results are
provided in Appendix A.1.
Evaluation metrics.Because the benchmarks differ in answer format and evaluation objective,
we use task-specific metrics. For single-answer multiple-choice questions, we report accuracy based
on the final selected option. For multiple-answer NEJM-QA questions, we use exact-set accuracy,
requiring the predicted option set to match the complete reference set. For MedRBench, we follow the
10

official evaluation protocol and report the automated reasoning scores for the diagnosis and treatment
stages, together with their aggregate result. For HealthBench, we report the physician-authored
rubric score, which measures the extent to which each response satisfies its conversation-specific
clinical criteria. All benchmark evaluators are used only for scoring and do not participate in
candidate generation, evidence adjudication, or final response generation.
2.7 Baselines and Framework Variants
We evaluate all methods using the same answer-generation backbones to control for differences in
the underlying model when comparing retrieval and evidence-adjudication strategies. Specifically,
we use o3-mini as a commercial backbone and Qwen-3.5-9B as an open-weight backbone. This
pairing allows us to examine whether the proposed framework remains effective across models with
different capability levels and accessibility.
Direct inference.TheDirectbaseline answers each query using only the parametric knowledge
encoded in the backbone model:
adirect =M(q).(19)
This retrieval-free setting serves as the primary reference for assessing whether locally retrieved
evidence, dynamically acquired evidence, and cross-source adjudication improve upon parametric
reasoning alone.
Retrieval-augmented generation.We use MedRAG Xiong et al. [2024a] as the retrieval-
augmented baseline because it provides an established framework for medical question answering
and supports controlled evaluation with different retrieval methods. We evaluate MedRAG using
two retrieval configurations:MedRAG (BM25), based on sparse lexical retrieval, andMedRAG
(MedCPT), based on dense biomedical retrieval. Both configurations retrieve from the same local
corpus and use the same answer-generation backbone as the corresponding Direct and proposed-
method settings.
These two configurations help assess whether improvements extend beyond those obtained by
changing the local retrieval method. Conceptually, MedRAG represents the local-evidence pathway,
alocal=M(q,R(q, K local)),(20)
whereas CLEAR jointly considers parametric, local, and dynamic pathways and explicitly
adjudicates their agreement and conflicts.
Proposed framework.We evaluate three configurations of the proposed framework:
•CLEAR (default)is the primary configuration. It includes source-specific candidate gen-
eration, the aggregation verifier, the override guard, the challenge audit, bounded follow-up
search, and task-specific response generation.
•CLEAR (dual models)follows the same adjudication procedure but introduces a second
model to assess the relevance and quality of dynamically retrieved evidence. This configu-
ration evaluates whether model diversity during evidence assessment improves cross-source
adjudication.
11

Table 1: Results on Common MCQ Datasets
Method Backbone MedQA PubMedQA NEJM Overall
GPT-based Model
Direct o3-mini 92.38 79.40 84.12
MedRAG (BM25 Setting) o3-mini 92.1482.0080.76
MedRAG (MedCPT Setting) o3-mini 91.59 80.60 80.61
CLEAR o3-mini 92.69 81.6085.04
CLEAR (dual models) o3-mini 92.14 81.80 84.73
CLEAR (Query Planning + Iterative Search) o3-mini92.8580.60 84.43
Qwen-based Model
Direct Qwen-3.5-9B 72.19 78.40 66.56
MedRAG (BM25 Setting) Qwen-3.5-9B 73.8480.0066.87
MedRAG (MedCPT Setting) Qwen-3.5-9B 72.66 79.40 66.87
CLEAR Qwen-3.5-9B 79.97 79.20 73.74
CLEAR (dual models) Qwen-3.5-9B83.5879.60 75.42
CLEAR (Query Planning + Iterative Search) Qwen-3.5-9B 83.19 79.4078.17
•CLEAR (QPIS)extends the primary configuration with Query Planning and Iterative
Search. It decomposes complex queries into targeted subqueries and iteratively acquires
additional evidence when the current evidence bundle remains insufficient. This configuration
is designed for questions requiring broader evidence coverage or multi-step information seeking.
The three configurations share the same cross-source adjudication architecture and differ primarily
in how dynamic evidence is acquired or assessed, including the use of an additional assessment model
or iterative query planning. Unless otherwise stated,CLEAR (default)is treated as the primary
proposed method, whereas the dual-model and QPIS configurations are evaluated as extensions.
3 Results
In this section, we systematically present the main experimental results of CLEAR. We evaluate
the framework separately on multiple-choice question (MCQ) and free-text benchmarks and further
conduct ablation studies on its two key mechanisms. To keep the main text focused, we report the
primary results and analyses here, while additional experimental results are provided in Section A.1.
3.1 Results on MCQ Datasets
As shown in Table 1,CLEARachieves competitive results across the MCQ datasets.
In detail, on MedQA, theCLEAR variantsachieve performance comparable to or better than
the Direct and MedRAG baselines across both the o3-mini and Qwen backbones. On PubMedQA,
although the CLEAR variants do not surpass MedRAG under the BM25 setting, their performance
remains close. Together, these results show that CLEAR remains competitive with the Direct and
MedRAG baselines on commonly used medical QA benchmarks.
NEJM-QA targets more specialized medical knowledge than MedQA and PubMedQA. For overall
NEJM-QA accuracy, allCLEARvariants outperform the evaluated baselines. This advantage is
particularly pronounced with the Qwen backbone, where the best CLEAR configuration improves
accuracy by 11.30 percentage points over the strongest baseline.
12

Table 2: NEJM Detail Results
Method Backbone General Surgery Internal Medicine Psychiatry Pediatrics Obgyn
GPT-based Model
Direct o3-mini 85.82 87.30 83.33 83.84 80.58
MedRAG (BM25 Setting) o3-mini 80.85 80.16 79.33 86.87 78.42
MedRAG (MedCPT Setting) o3-mini 78.72 84.92 82.67 81.82 75.54
CLEAR o3-mini87.9483.3385.33 87.8881.29
CLEAR (dual models) o3-mini 87.23 82.54 84.00 86.8783.45
CLEAR (Query Planning + Iterative Search) o3-mini 85.1188.8982.67 86.87 79.86
Qwen-based Model
Direct Qwen-3.5-9B 62.41 64.29 77.33 62.63 64.03
MedRAG (BM25 Setting) Qwen-3.5-9B 67.38 65.87 73.33 66.67 60.43
MedRAG (MedCPT Setting) Qwen-3.5-9B 63.83 65.08 75.33 64.65 64.03
CLEAR Qwen-3.5-9B 71.63 75.40 81.33 71.72 67.63
CLEAR (dual models) Qwen-3.5-9B75.8974.60 81.33 75.76 69.06
CLEAR (Query Planning + Iterative Search) Qwen-3.5-9B 75.1879.37 84.00 78.79 73.38
As shown in Figure 3, we further report specialty-level results on NEJM-QA. Across the two
backbones, at least oneCLEARconfiguration achieves the highest performance in each evaluated
specialty. The gains are particularly consistent with the Qwen backbone.
In contrast to the results on MedQA and PubMedQA, the MedRAG baseline, which retrieves
from a local corpus, shows lower performance on NEJM-QA under both the BM25 and MedCPT
retriever settings. With o3-mini, MedRAG (BM25) and MedRAG (MedCPT) achieve 80.76% and
80.61%, respectively, compared with 84.12% for Direct inference. A similar pattern is observed with
Qwen-3.5-9B.
3.2 Results on Free-Text Datasets
To assess performance beyond multiple-choice question answering, we further evaluate CLEAR on
MedRBench and HealthBench-Hard.
Table 3 reports results on the free-text benchmarks. On MedRBench, CLEAR scores 82.38
Overall with gpt-o3-mini (82.34 on Diagnosis, 82.46 on Treatment) and 74.74 with qwen-3.5-9B
(76.70, 70.97). QPIS variant scores 79.77 and 73.30 Overall on the two backbones. Direct inference
scores 71.30 and 61.05 Overall; MedRAG scores 71.03 (BM25) and 65.45 (MedCPT) with gpt-o3-mini,
and 58.22 under both settings with qwen-3.5-9B.
HealthBench shows a different pattern from MedRBench. With gpt-o3-mini, CLEAR scores
0.2422 against 0.3334 for Direct inference, and falls below both MedRAG settings, while QPIS
scores 0.3334. With qwen-3.5-9B, CLEAR scores 0.3444 against 0.3179 for Direct inference, and
QPIS scores 0.3179. Both MedRAG settings score below Direct inference under both backbones.
We discuss these backbone- and task-dependent differences in Section 4.
3.3 Ablation Studies
We conduct two ablation studies to examine cross-source adjudication and the targeted follow-up
search mechanism.
First, we compare different evidence configurations on the o3-mini backbone, as reported in
Table 4. Direct uses parametric knowledge only, BM25 uses local retrieval, Online-only uses dynamic
retrieval, and CLEAR combines all three sources through the full framework. Performance varies
across datasets. On MedQA, CLEAR achieves the highest score of 92.69, compared with 92.38 for
13

MedQA
91.5 92 92.5 93
Accuracy (%)  
Direct
MedRAG (BM25)
MedRAG (MedCPT)
CLEAR
CLEAR (dual)
CLEAR (QPIS)best baseline
92.38
92.14
91.59
92.69
92.14
92.85
70 75 80 85
Accuracy (%)  
Direct
MedRAG (BM25)
MedRAG (MedCPT)
CLEAR
CLEAR (dual)
CLEAR (QPIS)best baseline
72.19
73.84
72.66
79.97
83.58
83.19PubMedQA
79 80 81 82 83
Accuracy (%)  
Direct
MedRAG (BM25)
MedRAG (MedCPT)
CLEAR
CLEAR (dual)
CLEAR (QPIS)best baseline
79.40
82.00
80.60
81.60
81.80
80.60
78 79 80
Accuracy (%)  
Direct
MedRAG (BM25)
MedRAG (MedCPT)
CLEAR
CLEAR (dual)
CLEAR (QPIS)best baseline
78.40
80.00
79.40
79.20
79.60
79.40NEJM
80 82 84 86
Accuracy (%)  
Direct
MedRAG (BM25)
MedRAG (MedCPT)
CLEAR
CLEAR (dual)
CLEAR (QPIS)best baseline
84.12
80.76
80.61
85.04
84.73
84.43
65 70 75 80
Accuracy (%)  
Direct
MedRAG (BM25)
MedRAG (MedCPT)
CLEAR
CLEAR (dual)
CLEAR (QPIS)best baseline
66.56
66.87
66.87
73.74
75.42
78.17
GPT backbone Qwen backboneFigure 2: Performance on MedQA, PubMedQA, and NEJM-QA. CLEAR remains competitive
on MedQA and PubMedQA while showing larger gains on NEJM-QA, especially with the Qwen
backbone.
14

General Surgery
75 77.5 80 82.5 85 87.5 90
Accuracy (%)  
Direct
MedRAG (BM25)
MedRAG (MedCPT)
CLEAR
CLEAR (dual)
CLEAR (QPIS)best baseline
85.82
80.85
78.72
87.94
87.23
85.11
60 65 70 75 80 85
Accuracy (%)  
Direct
MedRAG (BM25)
MedRAG (MedCPT)
CLEAR
CLEAR (dual)
CLEAR (QPIS)best baseline
62.41
67.38
63.83
71.63
75.89
75.18Internal Medicine
75 77.5 80 82.5 85 87.5 90
Accuracy (%)  
Direct
MedRAG (BM25)
MedRAG (MedCPT)
CLEAR
CLEAR (dual)
CLEAR (QPIS)best baseline
87.30
80.16
84.92
83.33
82.54
88.89
60 65 70 75 80 85
Accuracy (%)  
Direct
MedRAG (BM25)
MedRAG (MedCPT)
CLEAR
CLEAR (dual)
CLEAR (QPIS)best baseline
64.29
65.87
65.08
75.40
74.60
79.37Psychiatry
75 77.5 80 82.5 85 87.5 90
Accuracy (%)  
Direct
MedRAG (BM25)
MedRAG (MedCPT)
CLEAR
CLEAR (dual)
CLEAR (QPIS)best baseline
83.33
79.33
82.67
85.33
84.00
82.67
60 65 70 75 80 85
Accuracy (%)  
Direct
MedRAG (BM25)
MedRAG (MedCPT)
CLEAR
CLEAR (dual)
CLEAR (QPIS)best baseline
77.33
73.33
75.33
81.33
81.33
84.00Pediatrics
75 77.5 80 82.5 85 87.5 90
Accuracy (%)  
Direct
MedRAG (BM25)
MedRAG (MedCPT)
CLEAR
CLEAR (dual)
CLEAR (QPIS)best baseline
83.84
86.87
81.82
87.88
86.87
86.87
60 65 70 75 80 85
Accuracy (%)  
Direct
MedRAG (BM25)
MedRAG (MedCPT)
CLEAR
CLEAR (dual)
CLEAR (QPIS)best baseline
62.63
66.67
64.65
71.72
75.76
78.79Obgyn
75 77.5 80 82.5 85 87.5 90
Accuracy (%)  
Direct
MedRAG (BM25)
MedRAG (MedCPT)
CLEAR
CLEAR (dual)
CLEAR (QPIS)best baseline
80.58
78.42
75.54
81.29
83.45
79.86
60 65 70 75 80 85
Accuracy (%)  
Direct
MedRAG (BM25)
MedRAG (MedCPT)
CLEAR
CLEAR (dual)
CLEAR (QPIS)best baseline
64.03
60.43
64.03
67.63
69.06
73.38
GPT backbone Qwen backboneFigure 3: Specialty-level results on NEJM-QA. At least one CLEAR configuration achieves the best
performance in each evaluated specialty across both backbone families.
15

Table 3: Results on Free-text Benchmarks
Method Backbone MedRBench Overall MedRBench D MedRBench T HealthBench
GPT-based Model
Direct o3-mini 71.30 71.06 71.77 0.3334
MedRAG (BM25 Setting) o3-mini 71.03 69.70 73.59 0.3051
MedRAG (MedCPT Setting) o3-mini 65.45 62.70 70.77 0.2997
CLEAR o3-mini82.38 82.34 82.460.2422
CLEAR (Query Planning + Iterative Search) o3-mini 79.77 80.46 78.430.3334
Qwen-based Model
Direct Qwen-3.5-9B 61.05 59.35 64.31 0.3179
MedRAG (BM25 Setting) Qwen-3.5-9B 58.22 57.58 59.48 0.2588
MedRAG (MedCPT Setting) Qwen-3.5-9B 58.22 57.58 59.48 0.2434
CLEAR Qwen-3.5-9B74.74 76.70 70.97 0.3444
CLEAR (Query Planning + Iterative Search) Qwen-3.5-9B 73.30 76.18 67.74 0.3179
Table 4: Ablation Study I: Comparison across direct, BM25, online-only, and ours settings.
Dataset Direct BM25 Online-only CLEAR
MedQA 92.38 92.14 87.8292.69
PubMedQA 79.4082.0077.80 81.60
NEJM-QA 84.12 80.76 78.3285.04
All 87.48 86.99 83.2088.34
Direct, 92.14 for BM25, and 87.82 for Online-only. On PubMedQA, BM25 achieves the highest
score of 82.00, followed by CLEAR at 81.60. On NEJM-QA, CLEAR obtains the highest score
of 85.04, compared with 84.12 for Direct, 80.76 for BM25, and 78.32 for Online-only. Aggregated
across the three datasets, CLEAR achieves the highest overall score of 88.34.
Second, we evaluate the effect of targeted follow-up search on the Qwen-3.5-9B backbone. As
shown in Table 5, CLEAR improves the overall score from 77.22 without follow-up search to 78.13
with follow-up search, corresponding to an absolute improvement of 0.91 percentage points. The
improvement is 1.02 percentage points on MedQA and 1.37 percentage points on NEJM-QA, while
the PubMedQA score remains unchanged at 79.20. Across the three datasets, follow-up search
is triggered 123 times, including 87 times on MedQA, 4 times on PubMedQA, and 32 times on
NEJM-QA. Among these triggered searches, 12 result in an accepted revision, corresponding to an
acceptance rate of 9.8%.
Overall, the ablation studies show that the full CLEAR framework achieves stronger aggregate
performance than any individual evidence-source setting, while targeted follow-up search provides
additional gains on a subset of unresolved cases.
3.4 Result Interpretation and Manual Verification
Because all benchmarks evaluated in this study are publicly available, web- or corpus-based retrieval
may potentially recover either benchmark instances themselves or the upstream documents from
which some benchmark items were constructed. We therefore manually audit every retrieval trace
produced during evaluation and explicitly distinguish two cases.
Direct benchmark-item leakagerefers to retrieval of the benchmark instance itself, such as
an exact or near-exact match to the question stem together with its answer choices or reference
answer. We detect such cases by checking for (1) contiguous question-stem matches, (2) high-overlap
16

Table 5: Ablation Study II: Impact and usage statistics of targeted follow-up searc
Setting MedQA PubMedQA NEJM-QA All
Performance
CLEAR79.97 79.20 73.74 78.13
w/o targeted follow-up search 78.9579.2072.37 77.22
Difference +1.02 +0.00 +1.37 +0.91
Second-search statistics
targeted follow-up search count 87 4 32 123
targeted follow-up search accepted 8 1 3 12
matches to the benchmark question, and (3) verbatim recovery of benchmark-specific answer options
or reference answers.
Upstream-source retrieval, by contrast, refers to retrieval of the original publication, abstract,
or clinical case from which a benchmark item was derived, without recovering the benchmark
question, answer options, or reference answer. We identify such cases using source metadata
and identifiers, including article titles, PubMed/PMC identifiers, and other benchmark-provided
provenance information. The same source-identifier checks are also applied to the MedRAG baselines.
As shown in Table 6, we observe no direct benchmark-item leakage for any evaluated benchmark
under either backbone: none of the retrieved documents contains the benchmark question-answer
instance itself. This indicates that we did not identify direct recovery of benchmark items in the
audited retrieval traces.
We next examine upstream-source retrieval. No upstream-source matches are observed for
MedQA, NEJM-QA, or HealthBench. For PubMedQA, the source document is retrieved for 42.4%
and 42.6% of examples with the GPT- and Qwen-based configurations, respectively. For MedRBench,
the corresponding rates are 60.4% and 62.5% on the diagnosis subset and 75.0% and 77.6% on the
treatment subset. The similar rates across backbones are consistent with upstream-source retrieval
being driven primarily by the relationship between benchmark provenance and the retrieval space
rather than by backbone-specific behavior.
Importantly, these upstream-source matches do not contain the benchmark items themselves.
Instead, they correspond to titles, abstracts, or publicly available clinical cases from PubMed and
PMC that served as source material for benchmark construction. Recovering such documents is an
expected behavior of an effective retrieval system, whose objective is to identify documents that
are highly relevant to a given query. For benchmarks derived from publicly indexed literature, the
original source document may therefore be retrieved even when the derived benchmark question,
answer options, and reference answer are not present.
This behavior is also observed in the MedRAG baselines. As shown in Table 7, on PubMedQA,
BM25 retrieves the source abstract for 55.0% of examples and MedCPT for 53.4%, compared with
42.4% for the CLEAR web-search configuration. The source abstract is ranked first for 55.0% and
50.8% of examples with BM25 and MedCPT, respectively. These results indicate that upstream-
source retrieval is not unique to CLEAR and should be distinguished from direct benchmark-item
leakage.
17

Table 6: Manual retrieval audit across benchmarks. Direct benchmark-item leakage denotes
retrieval of the benchmark question-answer instance itself, whereas upstream-source retrieval denotes
recovery of the original source document used to construct the benchmark item without recovering
the benchmark question or answer.
Benchmark Items Direct benchmark-item leakage Upstream-source retrieval
Hits Rate (%) Hits Rate (%)
GPT-based Model
MedQA 1,273 0 0.0 0 0.0
PubMedQA 500 0 0.0 212 42.4
NEJM-QA 655 0 0.0 0 0.0
HealthBench 1,000 0 0.0 0 0.0
MedRBench (diagnosis) 957 0 0.0 578 60.4
MedRBench (treatment) 496 0 0.0 372 75.0
Qwen-based Model
MedQA 1,273 0 0.0 0 0.0
PubMedQA 500 0 0.0 213 42.6
NEJM-QA 655 0 0.0 0 0.0
HealthBench 1,000 0 0.0 0 0.0
MedRBench (diagnosis) 957 0 0.0 598 62.5
MedRBench (treatment) 496 0 0.0 385 77.6
Table 7: Upstream-source retrieval on PubMedQA. Recovery of upstream source documents is
observed with both CLEAR web search and conventional MedRAG retrieval.
Retriever Items Source Retrieved Rate (%) Top-1 (%)
CLEAR (web search) 500 212 42.4 —
MedRAG (BM25) 500 275 55.0 55.0
MedRAG (MedCPT) 500 267 53.4 50.8
18

4 Discussion
4.1 Main findings
External evidence provides greater benefit when parametric knowledge provides weaker
support.A main finding of this study is that the value of external evidence depends on how
well the parametric knowledge of the backbone supports the task. When parametric knowledge is
already strong, additional retrieval may provide limited benefit and can even degrade performance.
For example, with o3-mini, Direct inference already achieves 92.38% accuracy on MedQA and a
rubric score of 0.3334 on HealthBench. CLEAR largely preserves performance on MedQA, whereas
additional retrieved evidence does not improve HealthBench and, under the default configuration,
reduces performance. These results suggest that retrieval should not be treated as uniformly
beneficial when the model already provides a strong answer from its parametric knowledge.
In contrast, larger gains are observed in settings where Direct inference provides a weaker
baseline. This pattern is particularly apparent with the open-weight Qwen-3.5-9B backbone. On
MedQA, Direct inference achieves 72.19% accuracy and the strongest conventional baseline reaches
73.84%, whereas the best CLEAR configuration reaches 83.58%, representing a 9.74-percentage-point
improvement over the strongest baseline. On NEJM-QA, the strongest conventional baseline achieves
66.87%, compared with 78.17% for the best CLEAR configuration, a gain of 11.30 percentage points.
A similar pattern is observed on MedRBench, where CLEAR improves the overall score from
61.05 with Direct inference to 74.74. Importantly, this benefit is not restricted to the open-weight
backbone: with o3-mini, CLEAR improves MedRBench from 71.30 to 82.38. Thus, the benefit of
external evidence appears to be task-dependent as well as model-dependent; even a strong backbone
may benefit substantially in settings where parametric knowledge alone provides limited support.
These observations are consistent with prior studies showing that the utility of retrieval depends on
both the sufficiency of parametric knowledge and the quality of retrieved evidence Kim et al. [2025],
Wang et al. [2025b].
Retrieved evidence should not be trusted by default.More importantly, the need for
additional knowledge does not imply that retrieved evidence should be trusted by default. The
source-ablation results illustrate this distinction. With o3-mini, using dynamic evidence alone
reduces accuracy from 92.38% to 87.82% on MedQA and from 84.12% to 78.32% on NEJM-QA.
Local retrieval is also not uniformly beneficial: MedRAG improves performance on PubMedQA but
reduces performance relative to Direct inference on several other benchmarks. These findings are
consistent with prior evidence that retrieval may introduce irrelevant, misleading, or conflicting
information and that models may fail to use retrieved evidence appropriately even when relevant
information is available Xie et al. [2025b], Wong et al. [2025]. External evidence can therefore
compensate for limitations in parametric knowledge, but it can simultaneously introduce a new
source of uncertainty.
Cross-source adjudication provides a preliminary approach to evidence management.
A third main finding is that explicit cross-source adjudication provides a preliminary approach
to addressing these challenges. CLEAR maintains parametric, local, and dynamic evidence as
separate candidate pathways rather than directly merging them into a single context. It then uses
an aggregation verifier to identify agreement and disagreement among the sources, an override guard
and challenge audit to determine whether a competing conclusion should revise the existing answer,
and targeted follow-up search when the available evidence remains insufficient to resolve the conflict.
Across the ten evaluated benchmarks, CLEAR remains competitive across both multiple-choice
19

and free-text settings, while achieving substantial gains in several settings where parametric or locally
retrieved knowledge alone provides weaker performance. The gains are particularly consistent with
the open-weight backbone. On the three primary MCQ benchmarks, the best CLEAR configurations
improve MedQA by 9.74 percentage points and NEJM-QA by 11.30 percentage points over the
strongest conventional baselines, while remaining close to the best result on PubMedQA. On
MedRBench, the default CLEAR improves the overall score from 61.05 with Direct inference to
74.74 with Qwen-3.5-9B, and from 71.30 to 82.38 with o3-mini. The broader evaluation on five
additional benchmarks shows a similar pattern: with Qwen-3.5-9B, the best CLEAR configurations
outperform the strongest conventional baselines on MedBullets, MedExQA, AfriMedQA, MMLU,
and MMLU-Pro, with gains ranging from 2.94 to 11.36 percentage points.
The two free-text benchmarks behave differently. On MedRBench, CLEAR improves over both
Direct inference and the MedRAG baselines under both backbones, and the improvement holds
on the Diagnosis and Treatment subsets alike; QPIS also performs well, slightly below the default
CLEAR configuration. On HealthBench, the pattern depends on the backbone. Inspection of the
intermediate outputs indicates that the o3-mini Direct baseline is already highly competitive on
this benchmark and often produces a sufficiently strong answer without external grounding. In
this setting, additional local or dynamic evidence contributes limited complementary information
and can interfere with an otherwise correct response, which is consistent with both MedRAG and
the default CLEAR configuration scoring below Direct inference. QPIS recovers to the level of
Direct inference, suggesting that query planning and iterative search integrate external evidence
more selectively. With qwen-3.5-9B the default configuration improves over Direct inference, from
0.3179 to 0.3444. Taken together, these results suggest that the benefit of cross-source adjudication
depends on how far the parametric model already resolves the question on its own, and that retrieval
does not necessarily help when it does.
The ablation results are consistent with a benefit from combining and adjudicating evidence
across sources rather than simply relying on additional retrieved evidence. Across the o3-mini MCQ
benchmarks, Direct inference, local BM25 retrieval, and dynamic evidence alone achieve aggregate
scores of 87.48, 86.99, and 83.20, respectively, whereas CLEAR reaches 88.34 with the full cross-
source framework. The targeted follow-up mechanism also provides additional benefit: removing the
follow-up search step reduces aggregate performance from 78.13 to 77.22 with Qwen-3.5-9B. Notably,
only 12 of 123 triggered follow-up searches ultimately lead to an accepted revision, suggesting that
additional retrieval is used selectively rather than automatically. These results collectively provide
preliminary evidence that preserving source provenance, explicitly identifying disagreement, and
selectively resolving cross-source conflicts can offer advantages over relying on a single knowledge
source or simply introducing more retrieved information.
4.2 Relation to existing literature
CLEAR builds on a growing body of work on retrieval-augmented generation, adaptive retrieval,
and agentic information seeking. RAG provides a general mechanism for supplementing parametric
language models with externally retrieved knowledge Lewis et al. [2020]. In medicine, systematic
evaluations have shown that retrieval quality, corpus selection, and retriever–generator alignment
can substantially influence downstream performance Xiong et al. [2024a]. Medical RAG approaches
have further introduced domain-specific retrieval and reasoning strategies. For example, the
knowledge-graph-based MedRAG framework proposed by Zhao et al. [2025] incorporates knowledge-
graph-elicited reasoning into healthcare question answering, while Self-BioRAG combines retrieval
with self-reflection to improve biomedical reasoning Jeong et al. [2024a]. Collectively, these studies
have established external grounding as an important strategy for improving medical reasoning and
20

have provided a strong foundation for studying how LLMs should interact with external knowledge.
A related line of work has substantially advanced the retrieval process itself by making it adaptive.
Self-RAG learns when to retrieve and critique retrieved passages through self-reflection Asai et al.
[2024]; Adaptive-RAG adjusts retrieval and reasoning strategies according to question complex-
ity Jeong et al. [2024b]; FLARE triggers retrieval during generation when the model anticipates
missing information Jiang et al. [2023]; and corrective RAG evaluates retrieved documents and
acquires additional information when the initial retrieval is insufficient Yan et al. [2024]. Medical
extensions similarly use iterative retrieval, query reformulation, and follow-up questions to improve
evidence acquisition Xiong et al. [2024b], Sohn et al. [2025]. These studies demonstrate that retrieval
need not be a fixed operation and provide important mechanisms for deciding whether, when, and
how additional information should be acquired. CLEAR builds on this insight and focuses on a
complementary question that becomes important once multiple knowledge sources are simultaneously
available: how should a system determine whether an existing conclusion should be preserved,
revised, or subjected to further evidence acquisition when those sources disagree?
Recent agentic systems further extend this paradigm by allowing models to autonomously invoke
tools, rerank evidence, and iteratively acquire information. Jia et al. [2026] combines retrieval,
reranking, evidence grounding, and memory mechanisms within an agentic medical question-
answering workflow, while AgentClinic evaluates the ability of clinical agents to use retrieval,
tools, and reflection in simulated clinical environments Schmidgall et al. [2026]. These studies
represent important steps toward more autonomous and adaptive medical reasoning systems and
demonstrate the value of allowing agents to actively interact with external information. CLEAR
is complementary to these efforts. Rather than focusing primarily on the orchestration of tools
and retrieval actions, it explicitly maintains parametric, local, and dynamic evidence as source-
specific candidate pathways before aggregation. Preserving this provenance makes agreement and
disagreement among heterogeneous sources directly observable and allows these relationships to
inform subsequent adjudication.
The proposed framework is most closely related to recent work on knowledge conflict, which has
directly highlighted the limitations of treating either internal or external knowledge as inherently
reliable. HealthContradict demonstrates that biomedical LLM behavior can change substantially
when supplied evidence is correct, incorrect, or internally contradictory Zhang et al. [2026], providing
an important benchmark for understanding model behavior under conflicting evidence. Astute RAG
further shows that neither internal nor external knowledge is consistently correct under conflict
and develops mechanisms for managing imperfect retrieval and knowledge disagreement Wang et al.
[2025b]. Wu et al. [2026] advance this direction by using disagreement among candidate responses
to guide additional retrieval across successive rounds and move the system toward better-supported
consensus. CLEAR builds on this emerging line of research while considering a somewhat different
evidence setting. Its candidate pathways are heterogeneous by construction: one reflects parametric
knowledge, one is conditioned on a locally curated corpus, and one is grounded in dynamically
acquired evidence. The framework therefore treats disagreement not only as a signal that additional
retrieval may be needed, but also as information about how different evidence sources support
competing conclusions. CLEAR first evaluates whether the available evidence is sufficient to
preserve or revise the current conclusion, uses the override guard and challenge audit to examine
both directions of potential error, and reserves additional search for conflicts that remain unresolved.
4.3 Limitations and future work
This study has several limitations. First, although we systematically evaluated CLEAR on ten
benchmarks spanning both multiple-choice and free-text settings, the limitations of benchmark-
21

based evaluation are well documented Wang et al. [2026a], Chen et al. [2025], Bedi et al. [2025].
In particular, public benchmarks may overlap with data encountered during model pretraining or
post-training Li et al. [2026], Alber et al. [2025]. We carefully audited the retrieved evidence for
direct benchmark-item leakage in this study; however, the training data of the evaluated backbone
models are not fully available, and potential parametric memorization therefore cannot be completely
ruled out. This remains an open challenge for the evaluation of contemporary LLMs. Future work
should prioritize validation on newly collected datasets to further reduce the risk of training-data
overlap and associated memorization effects.
Second, CLEAR is designed as a general, training-free framework that can be applied to different
backbone models without task-specific fine-tuning. Our experiments with both a commercial and an
open-weight backbone provide preliminary evidence for this plug-and-play design, but the current
adjudication mechanisms rely primarily on prompting and structured inference-time decision rules.
The approach could potentially be further improved through post-training strategies, such as
preference optimization or learning from expert adjudication examples. Evaluating such learned
adjudication mechanisms while preserving generalizability across models and tasks represents an
important direction for future work.
Third, our experiments remain offline evaluations and do not establish clinical effectiveness,
safety, or readiness for deployment. Dynamic evidence acquisition introduces additional challenges
in real clinical environments, including variability in search results, provenance and reliability
of external sources, reproducibility over time, institutional access restrictions, privacy, and data-
governance requirements Rahul and Kakade [2026]. These considerations may substantially affect
how dynamic or other external evidence can be integrated into clinical workflows. Future studies
should evaluate the framework in clinically relevant settings, including institutionally governed
evidence environments.
Finally, although we evaluated different backbone models, retrieval configurations, and dynamic
search implementations, broader evaluation across model families, retrievers, search systems, and
application domains will be necessary to establish the generalizability of the findings. We make
the implementation code publicly available to facilitate independent evaluation, extension, and
refinement of the framework by the research community.
Data availability
All datasets used in this study are publicly available. The multiple-choice QA benchmarks include
MedQA, PubMedQA, NEJM-QA, MedBullets, MedExQA, AfriMedQA, MMLU, and MMLU-
Pro. The free-text QA benchmarks include MedRBench and HealthBench. For retrieval-based
experiments, we use the publicly available MedCorp corpus as the local static knowledge base,
following the same corpus setting used in the MedRAG baseline. Online evidence is acquired at
inference time through the web search interface described in the Methods section. No private clinical
records, protected health information, or non-public patient-level data are used in this study.
Code availability
The code used to implement the proposed framework, including multi-path candidate genera-
tion, evidence aggregation, online evidence acquisition, and evaluation scripts, will be available at
https://github.com/Yale-BIDS-Chen-Lab/Agentic-Adjudication-for-Medical-Reasoning. All bench-
marks are publicly available.
22

Acknowledgements
This study is supported by the National Institutes of Health National Library of Medicine under
Award Number R01LM014604.
References
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman,
Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical
report.arXiv preprint arXiv:2303.08774, 2023.
Daniel Alexander Alber, Zihao Yang, Anton Alyakin, Eunice Yang, Sumedha Rai, Aly A Valliani,
Jeff Zhang, Gabriel R Rosenbaum, Ashley K Amend-Thomas, David B Kurland, et al. Medical
large language models are vulnerable to data-poisoning attacks.Nature Medicine, 31(2):618–626,
2025.
Lameck Mbangula Amugongo, Pietro Mascheroni, Steven Brooks, Stefan Doering, and Jan Seidel.
Retrieval augmented generation for large language models in healthcare: A systematic review.
PLOS Digital Health, 4(6):e0000877, 2025.
Rahul K Arora, Jason Wei, Rebecca Soskin Hicks, Preston Bowman, Joaquin Qui˜ nonero-Candela,
Foivos Tsimpourlas, Michael Sharman, Meghan Shah, Andrea Vallone, Alex Beutel, et al. Health-
bench: Evaluating large language models towards improved human health.arXiv preprint
arXiv:2505.08775, 2025.
Yaara Artsi, Vera Sorin, Benjamin S Glicksberg, Panagiotis Korfiatis, Robert Freeman, Girish N
Nadkarni, and Eyal Klang. Challenges of implementing llms in clinical practice: Perspectives.
Journal of Clinical Medicine, 14(17):6169, 2025.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avi Sil, and Hannaneh Hajishirzi. Self-rag: Learning to
retrieve, generate, and critique through self-reflection. InInternational conference on learning
representations, volume 2024, pages 9112–9141, 2024.
Suhana Bedi, Yutong Liu, Lucy Orr-Ewing, Dev Dash, Sanmi Koyejo, Alison Callahan, Jason A
Fries, Michael Wornow, Akshay Swaminathan, Lisa Soleymani Lehmann, et al. Testing and
evaluation of health care applications of large language models: a systematic review.Jama, 333
(4):319–328, 2025.
Elliot Bolton, Abhinav Venigalla, Michihiro Yasunaga, David Hall, Betty Xiong, Tony Lee, Roxana
Daneshjou, Jonathan Frankle, Percy Liang, Michael Carbin, et al. Biomedlm: A 2.7 b parameter
language model trained on biomedical text.arXiv preprint arXiv:2403.18421, 2024.
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are
few-shot learners.Advances in neural information processing systems, 33:1877–1901, 2020.
Qingyu Chen, Yan Hu, Xueqing Peng, Qianqian Xie, Qiao Jin, Aidan Gilson, Maxwell B Singer,
Xuguang Ai, Po-Ting Lai, Zhizheng Wang, et al. Benchmarking large language models for biomed-
ical natural language processing applications and recommendations.Nature communications, 16
(1):3280, 2025.
23

Sully F Chen, Anton Alyakin, Andreas Seas, Eunice Yang, Joanne J Choi, Jin Vivian Lee, Amelia L
Chen, Pranav I Warman, Rochelle T Bitolas, Robert J Steele, et al. Llm-assisted systematic
review of large language models in clinical medicine.Nature medicine, 32(3):1152, 2026.
Zeming Chen, Alejandro Hern´ andez Cano, Angelika Romanou, Antoine Bonnet, Kyle Matoba,
Francesco Salvi, Matteo Pagliardini, Simin Fan, Andreas K¨ opf, Amirkeivan Mohtashami,
et al. Meditron-70b: Scaling medical pretraining for large language models.arXiv preprint
arXiv:2311.16079, 2023.
Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam
Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm:
Scaling language modeling with pathways.Journal of machine learning research, 24(240):1–113,
2023.
Bhuwan Dhingra, Jeremy R Cole, Julian Martin Eisenschlos, Daniel Gillick, Jacob Eisenstein, and
William W Cohen. Time-aware language models as temporal knowledge bases.Transactions of
the Association for Computational Linguistics, 10:257–273, 2022.
Zihan Guan, Qiao Jin, Guangzhi Xiong, Fangyuan Chen, Mengxuan Hu, Qingyu Chen, Yifan Peng,
Zhiyong Lu, and Anil Vullikanti. Large language models lack temporal awareness of medical
knowledge.arXiv preprint arXiv:2605.13045, 2026.
Guideline Central. Clinical practice guidelines published in 2024: Guidelines year in review, 2025.
URLhttps://www.guidelinecentral.com/insights/2024-in-review-guidelines/.
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and
Jacob Steinhardt. Measuring massive multitask language understanding.arXiv preprint
arXiv:2009.03300, 2020.
Minbyul Jeong, Jiwoong Sohn, Mujeen Sung, and Jaewoo Kang. Improving medical reasoning through
retrieval and self-reflection with retrieval-augmented large language models.Bioinformatics, 40
(Supplement 1):i119–i129, 2024a.
Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C Park. Adaptive-rag:
Learning to adapt retrieval-augmented large language models through question complexity.
InProceedings of the 2024 Conference of the North American Chapter of the Association for
Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages
7036–7050, 2024b.
Shuyue Jia, Subhrangshu Bit, Varuna H Jasodanand, Yi Liu, and Vijaya B Kolachalama. Agen-
tic memory-augmented retrieval and evidence grounding for medical question-answering tasks.
International Journal of Medical Informatics, page 106339, 2026.
Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang,
Jamie Callan, and Graham Neubig. Active retrieval augmented generation. InProceedings of the
2023 conference on empirical methods in natural language processing, pages 7969–7992, 2023.
Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. What
disease does this patient have? a large-scale open domain question answering dataset from medical
exams.Applied Sciences, 11(14):6421, 2021.
24

Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William Cohen, and Xinghua Lu. Pubmedqa: A dataset
for biomedical research question answering. InProceedings of the 2019 conference on empirical
methods in natural language processing and the 9th international joint conference on natural
language processing (EMNLP-IJCNLP), pages 2567–2577, 2019.
Hyunjae Kim, Jiwoong Sohn, Aidan Gilson, Nicholas Cochran-Caggiano, Serina Applebaum, Heeju
Jin, Seihee Park, Yujin Park, Jiyeong Park, Seoyoung Choi, et al. Rethinking retrieval-augmented
generation for medicine: A large-scale, systematic expert evaluation and practical insights.arXiv
preprint arXiv:2511.06738, 2025.
Hyunjae Kim, Dain Kim, Pan Xiao, Serina S Applebaum, Younjoon Chung, Xuguang Ai, Yu Yin,
Roy Jiang, Yuexi Du, Yawen Wei, et al. Medpmc: A systematic framework for scaling high-fidelity
medical multimodal data for foundation models.arXiv preprint arXiv:2607.07673, 2026.
Yunsoo Kim, Jinge Wu, Yusuf Abdulle, and Honghan Wu. Medexqa: Medical question answering
benchmark with multiple explanations. InProceedings of the 23rd Workshop on biomedical natural
language processing, pages 167–181, 2024.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich K¨ uttler, Mike Lewis, Wen-tau Yih, Tim Rockt¨ aschel, et al. Retrieval-augmented
generation for knowledge-intensive nlp tasks.Advances in neural information processing systems,
33:9459–9474, 2020.
Anran Li, Lingfei Qian, Mengmeng Du, Yu Yin, Yan Hu, Zihao Sun, Yihang Fu, Hyunjae Kim,
Erica Stutz, Xuguang Ai, et al. Memorization in large language models in medicine prevalence
characteristics and implications.Nature Communications, 2026.
Fenglin Liu, Hongjian Zhou, Boyang Gu, Xinyu Zou, Jinfa Huang, Jinge Wu, Yiru Li, Sam S Chen,
Yining Hua, Peilin Zhou, et al. Application of large language models in medicine.Nature Reviews
Bioengineering, 3(6):445–464, 2025a.
Siru Liu, Allison B McCoy, and Adam Wright. Improving large language model applications in
biomedicine with retrieval-augmented generation: a systematic review, meta-analysis, and clinical
development guidelines.Journal of the American Medical Informatics Association, 32(4):605–615,
2025b.
National Library of Medicine. Medline pubmed production statistics, 2024. URL https://www.nlm.
nih.gov/bsd/medline_pubmed_production_stats.html. Last reviewed April 30, 2024.
Charles Nimo, Tobi Olatunji, Abraham Toluwase Owodunni, Tassallah Abdullahi, Emmanuel
Ayodele, Mardhiyah Sanni, Ezinwanne C Aka, Folafunmi Omofoye, Foutse Yuehgoh, Timothy
Faniran, et al. Afrimed-qa: A pan-african, multi-specialty, medical question-answering benchmark
dataset. InProceedings of the 63rd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 1948–1973, 2025.
Pengcheng Qiu, Chaoyi Wu, Shuyu Liu, Yanjie Fan, Weike Zhao, Zhuoxia Chen, Hongfei Gu,
Chuanjin Peng, Ya Zhang, Yanfeng Wang, et al. Quantifying the reasoning abilities of llms on
clinical cases.Nature Communications, 16(1):9799, 2025.
SS Rahul and Satish V Kakade. Regulatory compliance and data governance in ai-driven healthcare:
Legal and regulatory considerations for ai-driven healthcare solutions.Artificial Intelligence and
Machine Learning in Neurology, 1:79–108, 2026.
25

Samuel Schmidgall, Rojin Ziaei, Carl Harris, Ji Woong Kim, Eduardo Pontes Reis, Jeffrey Jopling,
and Michael Moor. Agentclinic: a multimodal benchmark for tool-using clinical ai agents.npj
Digital Medicine, 2026.
Yunxiao Shi, Xing Zi, Zijing Shi, Haimin Zhang, Qiang Wu, and Min Xu. Enhancing retrieval and
managing retrieval: A four-module synergy for improved quality and efficiency in rag systems.
arXiv preprint arXiv:2407.10670, 2024.
Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan
Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, et al. Large language models encode
clinical knowledge.Nature, 620(7972):172–180, 2023.
Karan Singhal, Tao Tu, Juraj Gottweis, Rory Sayres, Ellery Wulczyn, Mohamed Amin, Le Hou,
Kevin Clark, Stephen R Pfohl, Heather Cole-Lewis, et al. Toward expert-level medical question
answering with large language models.Nature medicine, 31(3):943–950, 2025.
Jiwoong Sohn, Yein Park, Chanwoong Yoon, Sihyeon Park, Hyeon Hwang, Mujeen Sung, Hyunjae
Kim, and Jaewoo Kang. Rationale-guided retrieval augmented generation for medical question
answering. InProceedings of the 2025 Conference of the Nations of the Americas Chapter of
the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long
Papers), pages 12739–12753, 2025.
Yuqiao Tan, Shizhu He, Huanxuan Liao, Jun Zhao, and Kang Liu. Dynamic parametric retrieval
augmented generation for test-time knowledge enhancement.arXiv preprint arXiv:2503.23895,
2025.
Shubo Tian, Qiao Jin, Lana Yeganova, Po-Ting Lai, Qingqing Zhu, Xiuying Chen, Yifan Yang,
Qingyu Chen, Won Kim, Donald C Comeau, et al. Opportunities and challenges for chatgpt and
large language models in biomedicine and health.Briefings in Bioinformatics, 25(1):bbad493,
2024.
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ ee
Lacroix, Baptiste Rozi` ere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand
Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language
models, 2023. URLhttps://arxiv.org/abs/2302.13971.
Bingning Wang, Haizhou Zhao, Huozhi Zhou, Liang Song, Mingyu Xu, Wei Cheng, Xiangrong Zeng,
Yupeng Zhang, Yuqi Huo, Zecheng Wang, et al. Baichuan-m1: Pushing the medical capability of
large language models.arXiv preprint arXiv:2502.12671, 2025a.
Fei Wang, Xingchen Wan, Ruoxi Sun, Jiefeng Chen, and Sercan O Arik. Astute rag: Overcoming
imperfect retrieval augmentation and knowledge conflicts for large language models. InProceedings
of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers), pages 30553–30571, 2025b.
Mingyang Wang, Alisa Stoll, Lukas Lange, Heike Adel, Hinrich Sch¨ utze, and Jannik Str¨ otgen. Bring
your own knowledge: a survey of methods for llm knowledge expansion. InProceedings of the
First Workshop on Large Language Model Memorization (L2M2), pages 150–168, 2025c.
Wenxuan Wang, Zizhan Ma, Guo Yu, Yiu-Fai Cheung, Meidan Ding, Jie Liu, Wenting Chen, and
Linlin Shen. Beyond the leaderboard: Rethinking medical benchmarks for large language models.
InProceedings of the 64th Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), pages 43078–43123, 2026a.
26

Xiaofei Wang, Zhuxin Xiong, Ke Zou, Sahana Srinivasan, Thaddaeus Wai Soon Lo, Yilan Wu,
Minjie Zou, Nan Liu, Fares Antaki, Weizhi Ma, et al. Reasoning-driven large language models in
medicine: opportunities, challenges, and the road ahead.The Lancet Digital Health, 2026b.
Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming
Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging
multi-task language understanding benchmark.Advances in Neural Information Processing
Systems, 37:95266–95290, 2024.
Lionel Wong, Ayman Ali, Raymond Xiong, Shannon Zeijang Shen, Yoon Kim, and Monica
Agrawal. Retrieval-augmented systems can be dangerous medical communicators.arXiv preprint
arXiv:2502.14898, 2025.
Wenhao Wu, Zhentao Tang, Yafu Li, Shixiong Kai, Mingxuan Yuan, Zhenhong Sun, Chunlin Chen,
and Zhi Wang. From conflict to consensus: Boosting medical reasoning via multi-round agentic
rag.arXiv preprint arXiv:2603.03292, 2026.
Qianqian Xie, Qingyu Chen, Aokun Chen, Cheng Peng, Yan Hu, Fongci Lin, Xueqing Peng, Jimin
Huang, Jeffrey Zhang, Vipina Keloth, et al. Medical foundation large language models for
comprehensive text analysis and beyond.NPJ digital medicine, 8(1):141, 2025a.
Weidong Xie, Yushan Fang, Xinxin Zheng, Dongjun Liu, and Zihan Li. When retrieval hurts: A
critical analysis of rag in medical question answering. InProceedings of the 2025 3rd International
Conference on Artificial Intelligence, Systems and Network Security, pages 312–317, 2025b.
Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and Aidong Zhang. Benchmarking retrieval-augmented
generation for medicine. InFindings of the Association for Computational Linguistics: ACL 2024,
pages 6233–6251, 2024a.
Guangzhi Xiong, Qiao Jin, Xiao Wang, Minjia Zhang, Zhiyong Lu, and Aidong Zhang. Improving
retrieval-augmented generation in medicine with iterative follow-up questions. InBiocomputing
2025: Proceedings of the Pacific Symposium, pages 199–214. World Scientific, 2024b.
Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. Corrective retrieval augmented generation,
2024. URLhttps://arxiv.org/abs/2401.15884.
Rui Yang, Matthew Yu Heng Wong, Huitao Li, Xin Li, Wentao Zhu, Jingchi Liao, Kunyu Yu,
Jonathan Chong Kai Liew, Weihao Xuan, Yingjian Chen, et al. Retrieval-augmented generation
in medicine: A scoping review of technical implementations, clinical applications, and ethical
considerations.Cell Reports Medicine, 2026.
Cyril Zakka, Rohan Shad, Akash Chaurasia, Alex R Dalal, Jennifer L Kim, Michael Moor, Robyn
Fong, Curran Phillips, Kevin Alexander, Euan Ashley, et al. Almanac—retrieval-augmented
language models for clinical medicine.Nejm ai, 1(2):AIoa2300068, 2024.
Boya Zhang, Alban Bornet, Rui Yang, Nan Liu, and Douglas Teodoro. Healthcontradict: Evaluating
biomedical knowledge conflicts in language models. npj digital medicine. 2026.
Zihan Zhang, Meng Fang, Ling Chen, Mohammad-Reza Namazi-Rad, and Jun Wang. How do large
language models capture the ever-changing world knowledge? a review of recent advances. In
Proceedings of the 2023 conference on empirical methods in natural language processing, pages
8289–8311, 2023.
27

Table 8: Results on additional MCQ datasets.
Method Backbone MedBullets MedExQA AfriMedQA MMLU MMLU-Pro
GPT-based Model
Direct gpt-o3-mini 81.49 85.45 79.89 92.65 77.02
MedRAG (BM25 Setting) gpt-o3-mini 80.52 84.06 76.44 93.39 75.92
MedRAG (MedCPT Setting) gpt-o3-mini 81.17 85.03 78.74 92.84 74.94
CLEAR gpt-o3-mini 81.8286.2078.74 93.85 76.89
CLEAR (dual models) gpt-o3-mini83.7786.10 79.89 93.76 77.26
CLEAR (Query Planning + Iterative Search) gpt-o3-mini 83.12 85.9979.89 93.85 78.00
Qwen-based Model
Direct qwen-3.5-9B 52.60 79.57 70.69 85.22 65.65
MedRAG (BM25 Setting) qwen-3.5-9B 53.90 79.25 70.11 86.96 66.38
MedRAG (MedCPT Setting) qwen-3.5-9B 54.22 80.86 68.39 86.59 66.50
CLEAR qwen-3.5-9B65.5883.53 75.29 89.7272.00
CLEAR (dual models) qwen-3.5-9B 63.9683.9676.44 89.72 71.52
CLEAR (Query Planning + Iterative Search) qwen-3.5-9B 63.64 83.2177.59 89.9070.78
Xuejiao Zhao, Siyan Liu, Su-Yin Yang, and Chunyan Miao. Medrag: Enhancing retrieval-augmented
generation with knowledge graph-elicited reasoning for healthcare copilot. InProceedings of the
ACM on Web Conference 2025, pages 4442–4457, 2025.
A Additional Experimental Details
A.1 Additional Results
To further examine the stability of CLEAR beyond the datasets used in our main evaluation,
we additionally evaluate the framework on five benchmarks spanning medical question answering
and broader knowledge and reasoning tasks. MedBullets contains 308 USMLE Step 2 and Step
3-style multiple-choice questions with expert-written explanations, collected from the MedBullets
platform Chen et al. [2025]. MedExQA targets five medical specialties underrepresented in existing
benchmarks and pairs each question with multiple reference explanations to support explanation-
quality evaluation Kim et al. [2024]. AfriMed-QA is a large-scale, Pan-African, multi-specialty
medical question-answering dataset spanning more than 60 medical schools across 16 countries Nimo
et al. [2025]. MMLU is a broad multitask benchmark of four-option multiple-choice questions across
57 subjects spanning STEM, humanities, social sciences, and professional domains Hendrycks et al.
[2020]. MMLU-Pro extends MMLU with more reasoning-focused questions, ten answer choices
instead of four, and broader coverage across 14 domains, including health Wang et al. [2024].
A.1.1 Results on Additional Datasets
As shown in Figure 4, CLEAR remains competitive across the five additional benchmarks. With
the Qwen-3.5-9B backbone, the best CLEAR configuration outperforms the strongest evaluated
baseline on all five benchmarks, with gains ranging from 2.94 to 11.36 percentage points. With the
o3-mini backbone, at least one CLEAR configuration matches or exceeds the strongest baseline
on each benchmark. These results provide additional evidence that the framework can be applied
across both medical question-answering tasks and broader knowledge and reasoning benchmarks.
28

gpt-o3-mini
Direct
81.49
85.45
79.89
92.65
77.02
MedRAG (BM25)
80.52
84.06
76.44
93.39
75.92
MedRAG (MedCPT)
81.17
85.03
78.74
92.84
74.94
81.82
86.20
78.74
93.85
76.89
83.77
86.10
79.89
93.76
77.26
83.12
85.99
79.89
93.85
78.00
MedBullets
MedExQA
AfriMedQA
MMLU
MMLU-Pro
qwen-3.5-9B
Direct
52.60
79.57
70.69
85.22
65.65
MedRAG (BM25)
53.90
79.25
70.11
86.96
66.38
MedRAG (MedCPT)
54.22
80.86
68.39
86.59
66.50
65.58
83.53
75.29
89.72
72.00
63.96
83.96
76.44
89.72
71.52
63.64
83.21
77.59
89.90
70.78
MedBullets
MedExQA
AfriMedQA
MMLU
MMLU-Pro
Performance
Higher Accuracy
Lower Accuracy
CLEAR
CLEAR (dual)
CLEAR (QPIS)CLEAR
CLEAR (dual)
CLEAR (QPIS)Figure 4: Overall results on five additional benchmarks. Performance is shown for each method and
backbone, with the color scale normalized separately within each dataset column.
A.2 Implementation Details
Backbone configuration.We instantiate CLEAR with two backbones: o3-mini as a commercial
model and Qwen-3.5-9B as an open-weight model. o3-mini is accessed through the Azure OpenAI
API (version 2024-12-01-preview), with temperature and reasoning effort left at their default settings
and a request timeout of 120 seconds. Qwen-3.5-9B is served through an OpenAI-compatible
server and queried with temperature 0, top- p0.95, and a maximum of 1024 output tokens, with
a request timeout of 300 seconds. In the primary configuration, the same backbone Mperforms
candidate generation across all three pathways, source-quality assessment, aggregation verification,
override-guard and challenge-audit adjudication, and task-specific response generation.
Candidate generation.The parametric pathway queries Mwithout retrieval. The local pathway
uses MedCorp Xiong et al. [2024a] as Klocal, the same corpus used by the MedRAG baselines, so that
local retrieval is compared using identical source material. MedCorp aggregates PubMed (23.9M
documents), 18 medical textbooks, and Wikipedia (6.5M articles). We instantiate Ras BM25 and
retrieve the topk= 16 snippets. The framework is agnostic to the choice ofR, and BM25 is used
here as a controlled sparse-retrieval instantiation. The dynamic pathway acquires evidence through
web search. The primary experiments use GPT-4o’s web-search capability for dynamic evidence
acquisition. As an additional implementation-diversity experiment, we instantiate the dynamic
pathway using Tavily on the NEJM-QA Internal Medicine subset, providing preliminary evidence
that CLEAR can also be implemented with an alternative web-search backend.
Source-quality assessment.The eight retrieved dynamic documents are assessed in a single
backbone call and assigned a categorical quality level of high, medium, or low according to source
authority, recency, and content type. The resulting counts, denoted nhigh,nmedium , and nlow,
constitute the source-quality information Sdynamic used by the aggregation verifier and adjudication
modules. We use categorical rather than continuous quality scores to avoid introducing an additional
manually calibrated threshold.
29

Aggregation verifier.The verifier receives the query, the option set for multiple-choice tasks,
the three candidate records, the local and dynamic evidence, and Sdynamic , and returns a structured
decision containing the preliminary answer, the routing decision z0∈ Z, a rationale, and a confidence
signal.
Override guard.The override guard is invoked when z0=accept dynamic and the dynamic
candidate conflicts with a competing conclusion. In particular, when aparam =alocal̸=adynamic , the
parametric–local consensus is treated as the default; when all three candidates differ, the parametric
candidate is used as the default for the guard assessment.
A targeted follow-up search is issued only when the dynamic evidence includes at least one
high-quality source ( nhigh≥1); otherwise, the proposed override is rejected without further retrieval.
The override is granted only when the follow-up assessment supports the challenging conclusion,
establishes that the default answer is not better supported, confirms that the retrieved evidence is
aligned with the query, provides support at the exact option level when applicable, and leaves no
material unresolved conflict. When these conditions are not satisfied, the system returns the default
answer with routekeep localorkeep parametric, as appropriate.
Challenge audit.The challenge audit is complementary to the override guard and is invoked when
the verifier doesnotinitially accept the dynamic candidate, but the dynamic pathway challenges an
existing parametric–local consensus, aparam =alocal̸=adynamic . In this case, the existing consensus
is treated as the default and the dynamic answer as the challenger.
A targeted follow-up search is used to determine whether sufficient evidence exists to reopen
the decision. The challenge is considered only when the retrieved evidence contains at least one
high-quality source or at least two medium-quality sources ( nhigh≥1 or nmedium ≥2). If this
evidence requirement is not met, or if the follow-up evidence does not clearly support the challenger,
the original parametric–local consensus is preserved.
Task adapters.For multiple-choice tasks, the format gate converts the adjudicated conclusion
into the option representation required by each benchmark and supports both single-answer and
multiple-answer settings, the latter arising in NEJM-QA. For free-text tasks, the response generator
composes the final answer from the adjudicated conclusion, its rationale, and the evidence bundle,
without repeating retrieval or revising the adjudication result.
Implementation diversity.As a complementary implementation-diversity analysis, we instanti-
ate the dynamic-evidence pathway using the Tavily API on the 126-question NEJM-QA Internal
Medicine subset with Qwen-3.5-9B. This experiment is intended to assess whether CLEAR can
be instantiated with an alternative web-search backend rather than to provide a comprehensive
comparison among search systems. No backend-specific performance tuning is applied. The results
are shown in Table 9.
30

Method Accuracy
Direct 84/126 (66.7%)
BM25 80/126 (63.5%)
Dynamic only 85/126 (67.5%)
CLEAR 89/126 (70.6%)
Table 9: Results on the NEJM-QA Internal Medicine subset using Qwen-3.5-9B and Tavily for
dynamic evidence acquisition. The experiment provides preliminary evidence that CLEAR can be
instantiated with an alternative web-search backend.
31