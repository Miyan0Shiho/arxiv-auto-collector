# RECTIFY: An Interactive Workbench for Post-Evaluation RAG Diagnosis, Repair, and Verification

**Authors**: Keerthana Murugaraj, Salima Lamsiyah, Martin Theobald

**Published**: 2026-09-15 07:37:58

**PDF URL**: [https://arxiv.org/pdf/2609.16764v1](https://arxiv.org/pdf/2609.16764v1)

## Abstract
Retrieval-Augmented Generation (RAG) evaluators can identify failures such as weak retrieval, poor grounding, incomplete answers, and unsupported generation, but they rarely help developers decide what to repair next. We present RECTIFY, an interactive Streamlit workbench that turns evaluated RAG cases into auditable repair workflows. RECTIFY filters cases that do not require repair, routes remaining failures into actionable families and finegrained repair slices, and generates editable repair cards that developers can approve, reject, or verify through sandbox reruns. On a controlled RAG benchmark, RECTIFY surfaces interpretable failure profiles across BM25, dense, and hybrid retrieval: BM25 mainly triggers noisy-retrieval repairs, while dense and hybrid retrieval leave smaller sets of multi-part underretrieval and underused-evidence cases. Additional analyses show that pre-filtering reduces unnecessary repair candidates and that slicelevel routing yields more targeted repair cards than broad family-level diagnosis. RECTIFY is publicly available as an open-source Streamlit workbench 1 for helping developers turn evaluation results into inspectable repair decisions.

## Full Text


<!-- PDF content starts -->

RECTIFY: An Interactive Workbench for Post-Evaluation RAG Diagnosis,
Repair, and Verification
Keerthana Murugaraj, Salima Lamsiyah, Martin Theobald
University of Luxembourg
Esch-sur-Alzette, LuxembourgCorrespondence:keerthana.murugaraj@uni.lu
Abstract
Retrieval-Augmented Generation (RAG) eval-
uators can identify failures such as weak re-
trieval, poor grounding, incomplete answers,
and unsupported generation, but they rarely
help developers decide what to repair next.
We present RECTIFY, an interactive Streamlit
workbench that turns evaluated RAG cases into
auditable repair workflows. RECTIFYfilters
cases that do not require repair, routes remain-
ing failures into actionable families and fine-
grained repair slices, and generates editable
repair cards that developers can approve, reject,
or verify through sandbox reruns. On a con-
trolled RAG benchmark, RECTIFYsurfaces in-
terpretable failure profiles across BM25, dense,
and hybrid retrieval: BM25 mainly triggers
noisy-retrieval repairs, while dense and hybrid
retrieval leave smaller sets of multi-part under-
retrieval and underused-evidence cases. Addi-
tional analyses show that pre-filtering reduces
unnecessary repair candidates and that slice-
level routing yields more targeted repair cards
than broad family-level diagnosis. RECTIFYis
publicly available as an open-source Streamlit
workbench1for helping developers turn evalu-
ation results into inspectable repair decisions.
1 Introduction
RAG has become a common design pattern for
building language-model systems that answer ques-
tions over external documents (Lewis et al., 2020).
This makes RAG useful in settings such as enter-
prise question answering, scientific search, legal
and historical archives, and customer-support sys-
tems. (Wiratunga et al., 2024; Murugaraj et al.,
2025; Xu et al., 2024) However, RAG pipelines
remain difficult to debug: a low-quality answer
may arise from failures in retrieval, grounding, an-
swer synthesis, or abstention (Es et al., 2024; Muru-
garaj et al., 2026; Ru et al., 2024). Recent surveys
1/githubCode Repositoryhighlight that RAG evaluation must account for
both retrieval and generation behavior, including
relevance, faithfulness, answer quality, robustness,
and benchmark design aligned with real user needs
(Gao et al., 2023b; Yu et al., 2025). Yet evaluation
alone does not close the debugging loop. Although
recent diagnostic frameworks provide actionable
per-case recommendations (Cohen et al., 2025), de-
velopers are still often left to decide which failures
share a common root cause, which pipeline compo-
nent should be changed, and whether a proposed
change actually improves the affected examples.
This post-evaluation step is especially important
because treating every failed case as an isolated
error makes repair slow, inconsistent, and difficult
to audit.
We present RECTIFY, an interactive workbench
for RAG failure diagnosis, repair, and optional veri-
fication. RECTIFYuses RAGVue (Murugaraj et al.,
2026) as its primary evaluator and turns diagnos-
tic evaluation outputs into a post-evaluation repair
workflow. It takes evaluated cases, groups recur-
ring failures into four actionable families:retrieval,
grounding,generation, andabstention, and maps
them to fine-grained repair slices. Each slice is con-
verted into an editable repair card that a developer
can inspect, approve, or reject before any change
is tested. Approved repairs can optionally be veri-
fied in a sandbox on the affected cases, with before
and after deltas and provenance logs supporting
auditability. Our key contributions are as follows:
•We present RECTIFY, an open-source Stream-
lit workbench for post-evaluation RAG debug-
ging that turns evaluator outputs into case ex-
ploration, failure diagnosis, repair-card review,
provenance logging, and sandbox verification.
•We introduce a deterministic failure-routing
scheme that filters non-actionable cases and
maps remaining failures to four macro fami-
1
arXiv:2609.16764v1  [cs.SE]  15 Sep 2026

1. Unified Case
Representation
answers, contexts, scores2. Failure Families
& Repair Slices
pre-filters; 4 families; 23 slices3. Repair-Card
Generation
template-based repair hypotheses4. Human Approval
& Provenance
inspect, edit, approve/reject5. Optional Sandbox
Verification
rerun cases; report deltas
provenance log
scope, notes, timestamp
Figure 1: RECTIFYpost-evaluation workflow. Evaluated cases are normalized, routed into failure families and
repair slices, converted into editable repair cards, reviewed by a developer, and optionally verified through sandbox
reruns with provenance logging.
lies and 23 fine-grained repair slices, enabling
cluster-level repair decisions rather than case-
by-case inspection.
•We design editable repair cards that connect
each failure slice to inspectable configuration-
level interventions over retrieval, chunking,
reranking, prompting, abstention, and gener-
ation while keeping developers in control of
approval and scope.
•We evaluate RECTIFYon a controlled RAG
benchmark across BM25, dense, and hybrid re-
trieval, showing interpretable retriever-specific
repair agendas, fewer unnecessary repair can-
didates after pre-filtering, and more targeted
cards than family-level routing.
2 Related Work
RAG Evaluation and Benchmarks.Recent
work has developed metrics and benchmarks for
evaluating RAG systems beyond end-task accu-
racy. RAGAS introduced reference-free metrics for
faithfulness, answer relevancy, context precision,
and context recall (Es et al., 2024). ARES trains
lightweight judges for context relevance, answer
faithfulness, and answer relevance using synthetic
data and limited human annotation (Saad-Falcon
et al., 2024). RAGChecker separates retrieval and
generation behavior through fine-grained diagnos-
tic metrics (Ru et al., 2024). RAGVue provides di-
agnostic and explainable reference-free evaluation
across retrieval quality, answer relevance and com-
pleteness, strict faithfulness, and calibration (Mu-
rugaraj et al., 2026). Complementary benchmarks
study hallucination, robustness, citation-supported
generation, and actionable evaluation labels, includ-
ing RAGTruth, RGB, ALCE, and RAGBench (Niu
et al., 2024; Chen et al., 2024; Gao et al., 2023a;
Friel et al., 2024). These works expose important
quality signals, but they do not by themselves de-
fine a human-approved repair workflow.From Evaluation to Guidance and Repair.Re-
cent approaches move from evaluation toward de-
veloper guidance. RAGXplain converts evalua-
tion scores into per-case natural-language expla-
nations for improving RAG pipelines (Cohen et al.,
2025). RAGGY provides composable RAG prim-
itives and an interactive interface for real-time
pipeline debugging (Romero Lauro et al., 2026).
ARAGOG compares RAG configurations such as
reranking, multi-query retrieval, maximal marginal
relevance, HyDE, and sentence-window retrieval
(Eibich et al., 2024). Doctor-RAG studies failure-
aware repair for agentic RAG by localizing fail-
ures in reasoning trajectories and repairing the
diagnosed point (Jiao et al., 2026). These sys-
tems make evaluation more actionable, but they do
not focus on dataset-level post-evaluation failure
grouping, human approval of repair cards, and mea-
sured before–after verification for standard RAG
pipelines.
In contrast to the existing works, RECTIFY
addresses the gap between RAG evaluation and
pipeline revision. It is neither another evaluator nor
an automatic self-repair system: it starts from al-
ready evaluated cases and treats proposed changes
as repair hypotheses that require developer inspec-
tion and approval. Given evaluator outputs, REC-
TIFYgroups recurring failures into repair slices,
generates editable configuration-level repair cards,
optionally verifies approved repairs on the affected
cases, reports before and after deltas, and records
decisions in a provenance log. This positions REC-
TIFYas a post-evaluation workbench for auditable,
human-controlled RAG diagnosis and repair.
3 The RECTIFYSystem
RECTIFYoperates after a RAG pipeline has pro-
duced answers and an evaluator has scored them.
Figure 1 summarizes the post-evaluation workflow.
2

3.1 Unified Case Representation
RECTIFYfirst normalizes each evaluated RAG case
into a shared schema. Each case contains the ques-
tion, generated answer, retrieved contexts, optional
expected answer, evaluator scores, diagnostic fields,
and available metadata. This common representa-
tion allows RECTIFYto apply the same workflow
across cases: failure routing, repair-card generation,
human review, provenance logging, and optional
sandbox verification. In the current implementa-
tion, RAGVue is the primary evaluator because
it provides diagnostic signals for retrieval quality,
grounding, answer completeness, abstention behav-
ior, and response quality. These signals are used
to assign cases to failure families and fine-grained
repair slices.
3.2 Failure Families & Repair Slices
RECTIFYuses a two-layer taxonomy to convert
evaluator outputs into repairable failure patterns.
Before routing begins, two pre-filters remove cases
that do not require repair. First, a correct-abstention
filter excludes unanswerable cases when the model
appropriately refuses to answer. Second, a gold-
answer filter excludes answerable cases when the
generated answer is sufficiently close to the gold
answer and grounded in the retrieved context. The
routing process is illustrated in Figure 2.
Cases not removed by these pre-filters are as-
signed to a primary macro failure family in the fol-
lowing priority order:abstention,retrieval,ground-
ing, andgeneration. This ordering is conservative:
unsupported confident answers are handled first,
and retrieval is checked before grounding because
faithfulness is difficult to interpret when evidence
is weak or noisy. Generation is used when retrieval
and grounding are adequate, but the final answer
remains unsatisfactory. When multiple failure con-
ditions hold, RECTIFYalso records a secondary
family to support compound repairs. Within the
selected macro family, RECTIFYassigns the case
to a fine-grained repair slice. The current taxonomy
defines 23 slices across the four families, with each
slice linked to a repair-card template. This repair
step is more specific than family-level diagnosis
alone: for example, different retrieval failures may
require reranking, broader evidence retrieval, or
chunking changes, while grounding failures may
require more evidence-constrained prompting. The
full failure taxonomy is presented in Table 1.Evaluated RAG case
Correct
abstention?
Gold answer
available?
Answer close
+ grounded?
Any failure
trigger?No repair
needed
Layer 1: Macro family
Abstention→Retrieval→Grounding→Generation
Record secondary family
if another condition also fires
Layer 2: Repair slice
23 slices within selected family
Generate repair cardno
yes
no
yesyes
yes
nono
Figure 2: Failure routing in RECTIFY.
3.3 Repair-Card Generation
For each populated repair slice, RECTIFYcreates
a repair card: a structured, editable proposal that
describes the diagnosed failure, the affected cases,
and the suggested configuration-level intervention.
Card generation is deterministic. Given the as-
signed repair slice and affected case IDs, REC-
TIFYselects the corresponding template and fills
in the target pipeline stage, proposed change, ex-
pected benefit, expected tradeoff, and repair scope.
Repair cards turn recurring failure patterns into
concrete repair hypotheses. For example, a noisy-
retrieval slice may suggest enabling reranking,
while an underused-evidence slice may suggest
a more evidence-constrained prompt. For com-
pound failures, RECTIFYcan combine interven-
tions across stages, such as reranking together with
grounded prompting.
3.4 Human Approval & Provenance
RECTIFYkeeps the human at the decision point.
The user can inspect the repair slice, review af-
3

Macro family Slice Name Short interpretation
AbstentionA1 Confident unsupported answer Model answers confidently despite no supporting evidence.
A2 Partial-evidence overconfidence Model answers fully when evidence only partially supports the claim.
A3 Ambiguous forced answer Model picks one interpretation instead of flagging ambiguity.
RetrievalR1 Partial coverage Retrieved chunks cover only part of what the question requires.
R2 Noisy retrieval Retrieved chunks contain irrelevant or distracting content.
R3 Evidence ignored Relevant chunks are retrieved but not used in the answer.
R4 Fragmented evidence Relevant information is split across chunks, none sufficient alone.
R5 Multi-part under-retrieval Retriever returns chunks for only one part of a multi-faceted question.
R6 Distractor-dominated High-scoring irrelevant chunks crowd out relevant ones.
R7 Sparse evidence Corpus lacks sufficient content to answer the question.
GroundingG1 Temporal misattribution Correct fact assigned to the wrong time period.
G2 Entity substitution Correct relationship stated with the wrong entity name.
G3 Unsupported causal bridge Model infers a causal link not stated in the context.
G4 Omitted qualifier Context is hedged but the answer states the claim as absolute.
G5 Broken multi-hop Error introduced while chaining reasoning steps across chunks.
G6 Unsupported synthesis Model combines facts across chunks in an unsupported way.
G7 Citation drift Answer uses correct content but attributes it to the wrong entity.
GenerationS1 Partial aspect coverage Answer addresses some but not all aspects of the question.
S2 Shallow summarization Answer is too surface-level and misses important context details.
S3 Underused evidence Relevant context is retrieved but not incorporated in the answer.
Q1 Rambling Answer is verbose, unfocused, or contains unnecessary content.
Q2 Poor structure Answer is hard to follow due to disorganized presentation.
Q3 Internal inconsistency Answer contradicts itself within the same response.
Table 1: RECTIFYfailure taxonomy: four macro families and 23 fine-grained repair slices.
fected cases, edit the proposed parameters, choose
the repair scope, and approve or reject the card.
Each approval or rejection is stored in a provenance
log containing the repair-card identifier, slice type,
affected cases, approved parameters, scope, user
notes, timestamp, expected benefit, and expected
tradeoff. This log makes the repair process au-
ditable: it records what was proposed, what was
approved or rejected, which cases were affected,
and under which configuration the repair was con-
sidered or tested.
3.5 Optional Sandbox Verification
After approval, a repair card can be verified in a
sandbox before it is accepted as useful. The sand-
box applies the approved configuration patch to the
affected cases, reruns the RAG pipeline, reevalu-
ates the new outputs, and compares them with the
original evaluation results. In our experiments, this
verification uses RAGVue as the primary evaluator.
When sandbox verification is run, the delta report
summarizes whether each affected case improved,
remained unchanged, or regressed. RECTIFYre-
ports aggregate counts, improvement rate, average
metric deltas, and per-case before and after outputs.
This makes repair proposals empirically checkable
rather than merely plausible.
Local Streamlit Application.For interactive use,
RECTIFYprovides a Python-based Streamlit inter-
face that exposes the main workflow through a local
browser application. Users can clone the repositoryand start the interface with a standard command
such as streamlit run streamlit_app.py . The
local UI supports loading evaluation files, inspect-
ing diagnosed cases, reviewing repair cards, and
generating reports without writing code. This in-
terface is intended for practitioners who prefer a
point-and-click workflow while keeping data and
credentials on their own machine. Selected screen-
shots of the interface are provided in Appendix A,
and the full walkthrough is included in the reposi-
tory and demo video.
4 Evaluation
4.1 Experimental Setup
We evaluate RECTIFYon a controlled 100-question
synthetic RAG benchmark; dataset construction
and question statistics are described in Appendix B.
We test three retriever configurations over the same
corpus and question set: BM25 keyword retrieval
(Robertson and Zaragoza, 2009), dense retrieval
with all-MiniLM-L6-v2 embeddings (Reimers
and Gurevych, 2019; Wang et al., 2020), and a
hybrid BM25+dense retriever. For all configura-
tions, Mistral-7B (Jiang et al., 2023) is used as
the generator model, served locally via Ollama2.
The generated outputs are then evaluated with 12
RAGVue (Murugaraj et al., 2026) metrics using the
same Mistral-7B model as the judge.
2https://ollama.com/
4

Measure BM25 Dense Hybrid
Case accounting
Unanswerable prefiltered 16 16 16
Gold-answer filter 62 75 71
Subtotal: filtered before diagnosis 78 91 87
No repair family triggered 6 4 7
Final repair agenda 1656
Repair-slice breakdown
R2: Noisy retrieval 7 0 0
R5: Multi-part under-retrieval 3 4 1
S3: Underused evidence 6 1 5
Table 2: RECTIFYcase accounting and repair-slice
breakdown across BM25, Dense, and Hybrid retrieval.
The upper block shows filtering and routing outcomes;
the lower block shows the final repair agenda.
4.2 Results
We focus the evaluation on three main questions,
described below. Additional ablations on the pre-
filtering stage and taxonomy depth are reported in
Appendix E.
How do failure profiles differ across retriever
configurations, and does RECTIFYsurface con-
sistent and interpretable diagnostics?Table 2
summarizes the main RECTIFYoutputs for the
three retriever configurations: how cases are fil-
tered before diagnosis, how many cases remain in
the final repair agenda, and which repair slices
are triggered. The results show that retriever
choice substantially changes the repair agenda.
BM25 leaves 16 cases requiring repair, while dense
and hybrid retrieval reduce this to 5 and 6 cases,
respectively. This indicates that many failures
are retrieval-driven: semantic retrieval resolves
cases where BM25 retrieves lexically plausible but
weakly relevant evidence. Hybrid retrieval does
not clearly dominate dense retrieval in the final
repair agenda, but it still preserves complemen-
tary lexical and semantic signals. The repair-slice
breakdown shows why aggregate evaluator scores
(Table 4) alone are not sufficient. Under BM25,
the main repair need is R2 noisy retrieval. With
dense and hybrid retrieval, R2 drops to 0 cases, and
the remaining failures shift toward R5 multi-part
under-retrieval and S3 underused evidence. These
failure types can all lead to weak faithfulness, com-
pleteness, or answer relevance, but they require dif-
ferent repairs: reranking for R2, broader evidence
retrieval for R5, and prompt-level or abstention-Pattern across re-
trieversCount Interpretation
No repair in all retriev-
ers81 The question is handled cor-
rectly across BM25, Dense,
and Hybrid retrieval
Fails in at least one re-
triever19 Questions used to analyze
repair-slice consistency
Same S3 slice in ≥2
retrievers3 Evidence is retrieved but un-
derused by the generator
Same R5 slice in ≥2
retrievers2 The question needs evi-
dence from multiple docu-
ments
R2 only with BM25 7 BM25 introduces noisy
keyword-matched chunks
Other one-off or
mixed failures7 Failure depends on retriever
setting or slice interaction
Table 3: Repair-slice consistency across BM25, Dense,
and Hybrid retrieval.
policy changes for S3. This demonstrates that REC-
TIFYsurfaces interpretable diagnostics that are di-
rectly tied to repair actions.
Are repair slices consistent across retrievers?
We compare how RECTIFYlabels the same 100
questions under BM25, Dense, and Hybrid re-
trieval. If the same question receives the same
repair slice under multiple retrievers, we treat it
as a stable failure pattern. If a slice appears only
under one retriever, the failure is more likely tied
to that retrieval configuration. Table 3 shows that
81 questions need no repair under any retriever.
Among the 19 questions that fail at least once, re-
peated S3 and R5 labels reveal stable problems: S3
means that retrieved evidence is available but un-
derused, while R5 means that the question requires
evidence from multiple documents. In contrast,
seven R2 cases appear only with BM25, indicating
keyword-matching noise that disappears with dense
or hybrid retrieval. This suggests that RECTIFY’s
slices are not arbitrary labels: they help separate
stable failure patterns from retriever-specific errors.
Does RECTIFYreduce debugging effort?We
estimate debugging effort under three workflows.
In a raw metric-scan workflow, a developer inspects
every case where a core metric falls below a thresh-
old. With pre-filtering only, the developer inspects
the post-filter-failing cases individually, without
any grouping. With RECTIFY, the same pre-filters
are applied first, and the remaining failures are then
grouped into repair cards, so the developer reviews
one card per repair slice and makes one repair deci-
sion per cluster. Figure 3 shows that the raw metric
5

BM25 DENSE HYBRID020406080100Decisions for developer89
74 75
16
5 6
3 2 297%
 97%
 97%
Estimated repair decisions across workflows
A: Raw metric scan
B: Pre-filters only
C: Rectify (ours)Figure 3: Estimated debugging effort across three work-
flows: raw metric scan, pre-filtering only, and RECTIFY
with pre-filtering plus repair-slice grouping. Effort is
measured as the number of case-level or card-level re-
pair decisions.
scan requires 74-89 individual case inspections per
retriever configuration. The pre-filters reduce this
to 5-16 post-filter cases. RECTIFY’s taxonomy then
groups those cases into only 2-3 repair cards while
still covering the full final repair agenda. Under
our decision-count estimate, RECTIFYreduces the
number of repair decisions by 97%. This reduc-
tion is important because it changes the debugging
unit. Instead of making many ad hoc case-level
judgements, the developer reviews a small num-
ber of cluster-level repair hypotheses: for example,
enabling reranking for R2 noisy retrieval, increas-
ing top- kfor R5 multi-part under-retrieval, or ad-
justing prompting for S3 underused evidence. Fi-
nally, because sandbox verification is optional in
the workflow, we report detailed sandbox results
separately in Appendix D. These results illustrate
how approved repair cards can be checked through
before and after deltas before a developer accepts
or rejects a repair.
5 Conclusion
We presented RECTIFY, an interactive workbench
prototype for moving from RAG evaluation to
diagnosis, repair, and verification. Our experi-
ments show that RECTIFYsurfaces actionable fail-
ure profiles across retriever configurations. As
retrieval quality improves, the repair agenda be-comes smaller, and the dominant failure modes
shift: BM25 mainly suffers from noisy retrieval,
while dense and hybrid retrieval expose more spe-
cific failures, such as multi-part under-retrieval and
underused evidence. This shift would be difficult
to interpret from aggregate metrics alone, but be-
comes visible through RECTIFY’s slice-level diag-
nosis. The results also show why repair should
be treated as a review-and-verification workflow
rather than an automatic patching step. Pre-filtering
reduces unnecessary repair workload, fine-grained
slicing produces targeted repair cards, and sand-
box verification exposes both successful repairs
and cases where a proposed intervention should be
rejected or revised. Overall, RECTIFYcontributes
a post-evaluation workflow for RAG development:
diagnose recurring failures, propose inspectable
repairs, keep the developer in control, and verify
changes with before and after evidence.
Limitations and Future Work
RECTIFYis a local, interactive workbench for post-
evaluation RAG repair, not an automatic produc-
tion optimizer. Its current implementation uses a
deterministic taxonomy with 23 repair slices. This
design makes routing decisions reproducible and
auditable, while also making the taxonomy easy to
extend as new domains, retrievers, or generation
failure patterns introduce additional repair needs.
The pre-filtering stage can use benchmark annota-
tions such as gold answers and answerability labels
when they are available, but these annotations are
not required for the core workflow: without them,
RECTIFYstill performs diagnosis from evaluator
signals and simply skips annotation-based filter-
ing. Sandbox verification reruns affected cases
and reevaluates the new outputs, so developers can
apply it selectively to high-impact repair cards or
use it as a final check before accepting a proposed
configuration change.
Future work will expand RECTIFYalong three
directions: evaluating it on larger real-world knowl-
edge bases, extending the repair taxonomy to addi-
tional RAG architectures and application domains,
and studying how the workflow transfers across
evaluators with different diagnostic signals. We
also plan to test additional generator-judge com-
binations to further assess the stability of the ob-
served repair profiles.
6

Ethics Statement
RECTIFYis a developer-facing workbench for post-
evaluation RAG diagnosis and repair. It does not
automatically modify or deploy RAG systems; re-
pair cards require human review, approval, and
optional sandbox verification. Our experiments use
a synthetic benchmark with fictional entities and do
not involve private or personally identifiable data.
In real deployments, users should ensure that input
documents, evaluation files, model outputs, and
connected services comply with relevant privacy,
licensing, and data-governance requirements. REC-
TIFY’s suggestions should be treated as decision
support rather than guaranteed fixes, especially in
high-stakes domains.
References
Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun.
2024. Benchmarking large language models in
retrieval-augmented generation. InProceedings of
the Thirty-Eighth AAAI Conference on Artificial In-
telligence and Thirty-Sixth Conference on Innovative
Applications of Artificial Intelligence and Fourteenth
Symposium on Educational Advances in Artificial
Intelligence. AAAI Press.
Dvir Cohen, Lin Burg, and Gilad Barkan. 2025.
RAGXplain: From explainable evaluation to action-
able guidance of RAG pipelines.arXiv preprint
arXiv:2505.13538.
Matouš Eibich, Shivay Nagpal, and Alexander Fred-
Ojala. 2024. ARAGOG: Advanced RAG output grad-
ing.arXiv preprint arXiv:2404.01037.
Shahul Es, Jithin James, Luis Espinosa Anke, and
Steven Schockaert. 2024. RAGAs: Automated evalu-
ation of retrieval augmented generation. InProceed-
ings of the 18th Conference of the European Chap-
ter of the Association for Computational Linguistics:
System Demonstrations, pages 150–158. Association
for Computational Linguistics.
Robert Friel, Masha Belyi, and Atindriyo Sanyal. 2024.
RAGBench: Explainable benchmark for retrieval-
augmented generation systems.arXiv preprint
arXiv:2407.11005.
Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen.
2023a. Enabling large language models to gener-
ate text with citations. InProceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 6465–6488. Association for
Computational Linguistics.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,
and Haofen Wang. 2023b. Retrieval-augmented gen-
eration for large language models: A survey.arXiv
preprint arXiv:2312.10997.Albert Qiaochu Jiang, Alexandre Sablayrolles, Arthur
Mensch, Chris Bamford, Devendra Singh Chap-
lot, Diego de Las Casas, Florian Bressand, Gi-
anna Lengyel, Guillaume Lample, Lucile Saulnier,
Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre
Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang,
Timothée Lacroix, and William El Sayed. 2023.
Mistral-7b.ArXiv, abs/2310.06825.
Shuguang Jiao, Chengkai Huang, Shuhan Qi, Xuan
Wang, Yifan Li, and Lina Yao. 2026. Doctor-RAG:
Failure-aware repair for agentic retrieval-augmented
generation.arXiv preprint arXiv:2604.00865.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive NLP tasks. InProceedings of the 34th
International Conference on Neural Information Pro-
cessing Systems (NIPS’20). Curran Associates Inc.
Keerthana Murugaraj, Salima Lamsiyah, Marten Dur-
ing, and Martin Theobald. 2025. Topic-rag for his-
torical newspapers: Enhancing information retrieval
in humanities research through topic-based retrieval-
augmented generation.Computational Humanities
Research, 1:e15.
Keerthana Murugaraj, Salima Lamsiyah, and Martin
Theobald. 2026. RAGVUE: A diagnostic view for
explainable and automated evaluation of retrieval-
augmented generation. InProceedings of the 19th
Conference of the European Chapter of the Associa-
tion for Computational Linguistics (Volume 3: Sys-
tem Demonstrations), pages 512–526. Association
for Computational Linguistics.
Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu,
KaShun Shum, Randy Zhong, Juntong Song, and
Tong Zhang. 2024. RAGTruth: A hallucination cor-
pus for developing trustworthy retrieval-augmented
language models. InProceedings of the 62nd An-
nual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 10862–
10878. Association for Computational Linguistics.
Nils Reimers and Iryna Gurevych. 2019. Sentence-
BERT: Sentence Embeddings using siamese BERT-
Networks. InProceedings of the 2019 Conference
on Empirical Methods in Natural Language Process-
ing and the 9th International Joint Conference on
Natural Language Processing (EMNLP-IJCNLP’19),
pages 3982–3992.
Stephen Robertson and Hugo Zaragoza. 2009. The
probabilistic relevance framework: Bm25 and be-
yond.Found. Trends Inf. Retr., 3(4):333–389.
Quentin Romero Lauro, Shreya Shankar, Sepanta
Zeighami, and Aditya Parameswaran. 2026. RAG
Without the Lag: Enabling "What-If" Analysis for
Retrieval-Augmented Generation Pipelines. InPro-
ceedings of the 2026 CHI Conference on Human
7

Factors in Computing Systems (CHI’26). Association
for Computing Machinery.
Dongyu Ru, Lin Qiu, Xiangkun Hu, Tianhang Zhang,
Peng Shi, Shuaichen Chang, Cheng Jiayang, Cunxi-
ang Wang, Shichao Sun, Huanyu Li, Zizhao Zhang,
Binjie Wang, Jiarong Jiang, Tong He, Zhiguo Wang,
Pengfei Liu, Yue Zhang, and Zheng Zhang. 2024.
RAGCHECKER: a fine-grained framework for diag-
nosing retrieval-augmented generation. InProceed-
ings of the 38th International Conference on Neural
Information Processing Systems (NIPS’24). Curran
Associates Inc.
Jon Saad-Falcon, Omar Khattab, Christopher Potts, and
Matei Zaharia. 2024. ARES: An automated evalua-
tion framework for retrieval-augmented generation
systems. InProceedings of the 2024 Conference of
the North American Chapter of the Association for
Computational Linguistics: Human Language Tech-
nologies (Volume 1: Long Papers), pages 338–354.
Association for Computational Linguistics.
Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan
Yang, and Ming Zhou. 2020. Minilm: deep self-
attention distillation for task-agnostic compression
of pre-trained transformers. InProceedings of the
34th International Conference on Neural Information
Processing Systems (NIPS’20). Curran Associates
Inc.
Nirmalie Wiratunga, Ramitha Abeyratne, Lasal Jayawar-
dena, Kyle Martin, Stewart Massie, Ikechukwu Nkisi-
Orji, Ruvan Weerasinghe, Anne Liret, and Bruno
Fleisch. 2024. Cbr-rag: case-based reasoning for
retrieval augmented generation in llms for legal ques-
tion answering. InInternational Conference on Case-
Based Reasoning, pages 445–460. Springer.
Zhentao Xu, Mark Jerome Cruz, Matthew Guevara,
Tie Wang, Manasi Deshpande, Xiaofeng Wang, and
Zheng Li. 2024. Retrieval-augmented generation
with knowledge graphs for customer service question
answering. InProceedings of the 47th international
ACM SIGIR conference on research and development
in information retrieval, pages 2905–2909.
Hao Yu, Aoran Gan, Kai Zhang, Shiwei Tong, Qi Liu,
and Zhaofeng Liu. 2025. Evaluation of retrieval-
augmented generation: A survey. InBig Data, pages
102–120. Springer Nature Singapore.
8

A Additional Screenshots
Example UI screenshots are shown in Figure 4, and
the full walkthrough is available in the repository
and the demo video.
B Dataset
We construct a controlled synthetic corpus with
30 short documents about 10 fictional companies
and associated entities. The documents cover com-
pany profiles, persons, products, events, and the-
matic summaries and encode factual relations such
as founding year, headquarters, acquisitions, part-
nerships, and flagship products. We generate 100
questions across eight types: factoid (25), multi-
part (20), multi-hop (16), temporal (13), compari-
son (10), explicitly unanswerable (10), multi-hop
unanswerable (4), and temporal unanswerable (2).
In total, 84 questions are answerable, and 16 are
unanswerable. Answerable questions include gold
answers and known relevant document identifiers;
unanswerable questions test whether the system
correctly abstains when the required evidence is
absent from the corpus.
Why synthetic?We use a small controlled cor-
pus because RECTIFYis evaluated as a repair work-
bench, not as an open-domain QA system. Repair
evaluation requires more control than gold answers
alone: we need known relevant documents, con-
trolled unanswerable cases, and known corpus cov-
erage gaps to distinguish retrieval failures, missing
evidence, poor evidence use, and correct absten-
tion. The synthetic setup also lets us create tar-
geted failure conditions, such as noisy retrieval,
missing multi-hop evidence, underused evidence,
and unsupported answers. We acknowledge that
this corpus is narrower than real enterprise settings,
and future work will evaluate RECTIFYon larger
real-world knowledge bases.
C Full RAGVue Metric Scores
Table 4 reports the full set of RAGVue met-
ric scores used as evaluator signals in the cross-
retriever analysis.
D Sandbox Verification Results
Table 5 illustrates sandbox verification for two rep-
resentative repair slices. For R2 noisy retrieval,
applying reranking to seven BM25 cases improves
all seven. For R5 multi-part under-retrieval, in-
creasing top- kfrom 3 to 6 improves five of eightRAGVue metric BM25 Dense Hybrid
Strict faithfulness 0.6270.7450.741
Retrieval relevance 0.4500.5670.563
Retrieval coverage 0.757 0.8230.830
Answer completeness 0.3910.4740.465
Answer relevance 0.6480.7750.762
Context utilization 0.465 0.5700.596
Multi-hop faithfulness 0.6550.7780.762
Coherence 0.9991.000 1.000
Clarity 0.971 0.9810.987
Negative rejection0.9900.9700.990
Answer conciseness 0.9750.9930.991
Implicit contradiction0.9240.909 0.913
Table 4: Full RAGVue metric scores for BM25, Dense,
and Hybrid retrieval over the same 100-question bench-
mark. Bold marks the best value in each row.
Slice Repair n Improved Unchanged Regressed
R2 Enable reranker 770 0
R5 Increase top-k3→6 8 5 2 1
Table 5: Sandbox verification results for two representa-
tive repair slices.
cases, leaves two unchanged, and regresses one.
These results support the review-and-verification
loop in RECTIFY. Repair cards are testable hy-
potheses, not guaranteed fixes: sandbox deltas help
developers decide whether a proposed repair im-
proves, leaves unchanged, or regresses the affected
cases before accepting it.
E Additional Ablations
E.1 Pre-filters Reduce Unnecessary Repair
Candidates
RECTIFYapplies two pre-filters before failure rout-
ing, as described in Section 3.2. These filters are
not intended as a replacement for evaluation; they
reduce unnecessary repair work before the remain-
ing cases are diagnosed and grouped into repair
slices. Table 6 shows the effect of these filters.
A raw metric scan flags 89, 74, and 75 cases for
BM25, dense, and hybrid retrieval, respectively.
When the failure taxonomy is applied without pre-
filters, the candidate set becomes 68, 54, and 55
cases. With the abstention and gold-answer filters
enabled, the final repair agenda drops to 16, 5, and
6 cases. This corresponds to an 82–93% reduction
compared with raw metric scanning and a 77–91%
reduction compared with taxonomy-based routing
without pre-filters. The number of active failure
slices also decreases from eight to three, two, and
two, producing a smaller and more focused repair
agenda.
9

(a) Home (b) Repair card (c) Delta explorer
Figure 4: Selected screenshots of the local Python-based RECTIFYStreamlit interface.
Measure BM25 Dense Hybrid
Raw metric scan 89 74 75
Family routing, no pre-filters 68 54 55
RECTIFYwith pre-filters16 5 6
Reduction vs. raw scan 82% 93% 92%
Reduction vs. no pre-filters77% 91% 89%
Table 6: Ablation of RECTIFY’s pre-filtering stage. Raw
metric scan counts broadly flagged cases, while family
routing applies the taxonomy before disabling the ab-
stention and gold-answer filters.
Measure BM25 Dense Hybrid
Family-only routing
Repair cards 2 2 2
Avg. cases per card 8 3 3
Cases merged into retrieval card 10 4 1
Cases merged into generation card 6 1 5
Fine-grained slices
Repair cards 3 2 2
Avg. cases per card 5 3 3
R2/R5 separated Yes – –
S3 isolated from retrieval Yes Yes Yes
Table 7: Ablation of taxonomy depth over the post-filter
repair agenda. Fine-grained slices separate failures that
require different repair actions.
E.2 Fine-Grained Slices Produce More
Targeted Repair Cards
We ablate the second layer of RECTIFY’s taxon-
omy by comparing fine-grained repair slices with a
family-only routing baseline. Family-only routing
groups failures into broad families such as retrieval
or generation, while fine-grained routing separates
them into repair slices such as R2 noisy retrieval,
R5 multi-part under-retrieval, and S3 underused
evidence. Table 7 shows the practical effect of
this distinction. Under BM25, family-only routing
merges 10 retrieval-family cases into a single card,
although these cases require different interventions:
R2 points to reranking, while R5 points to broaderevidence retrieval, such as increasing top- k. Fine-
grained slicing separates these cases into more spe-
cific repair cards and keeps S3 underused-evidence
cases separate from retrieval failures. This makes
the repair agenda more actionable: developers re-
view slightly more specific cards, but each card
corresponds to a clearer repair hypothesis.
10