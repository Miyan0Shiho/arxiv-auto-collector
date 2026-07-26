# TAP-RAG: Task-Aware Policy Control for Long-Document Multimodal Question Answering

**Authors**: Zhong Ji, Keqi Jin, Yan Zhang, Jiasheng Li

**Published**: 2026-07-21 09:59:13

**PDF URL**: [https://arxiv.org/pdf/2607.18917v1](https://arxiv.org/pdf/2607.18917v1)

## Abstract
Long-document multimodal question answering requires more than retrieving relevant chunks from a large document. Different queries require different evidence behavior. Existing multimodal RAG systems improve evidence access through text chunks, page images, graph links, or heterogeneous document elements, but they often apply a largely query-agnostic evidence-use strategy. We present TAP-RAG, a task-aware policy-controlled RAG framework for long-document multimodal QA. TAP-RAG contains a main controller, the Task-Aware Policy Controller (TAPC), and two policy-guided evidence executors: Task-Aware Query-Guided Flow Diffusion (TA-QFD) and Task-Aware Visual Enhancement (TAVE). For each query, TAPC predicts the task prior, estimates visual/local/global evidence signals, and produces an executable policy. TA-QFD then expands textual and structural evidence over the multimodal document graph, while TAVE selectively inspects page images when visual or layout evidence is needed. A guarded synthesis stage fuses text, visual, and structural evidence and abstains when support is insufficient. On DocBench and MMLongBench-Doc, TAP-RAG achieves the best overall accuracy among the compared systems, improving over a matched multimodal-RAG baseline by +9.1 points (61.1 to 70.2) and +4.5 points (42.2 to 46.7), respectively.

## Full Text


<!-- PDF content starts -->

TAP-RAG: Task-Aware Policy Control for Long-Document Multimodal
Question Answering
Zhong Ji1, Keqi Jin2, Yan Zhang3,*, Jiasheng Li1,*
1School of Electrical and Information Engineering, Tianjin University
2The International Joint Institute of Tianjin University, Fuzhou, Tianjin University
3Department of Automation, Tsinghua University
jizhong@tju.edu.cn,keqi_jin2002@tju.edu.cn
yzhang1995@tsinghua.edu.cn,li_jiasheng@tju.edu.cn
*Corresponding authors.
Abstract
Long-document multimodal question answer-
ing requires more than retrieving relevant
chunks from a large document. Differ-
ent queries require different evidence behav-
ior. Existing multimodal RAG systems im-
prove evidence access through text chunks,
page images, graph links, or heterogeneous
document elements, but they often apply
a largely query-agnostic evidence-use strat-
egy. We present TAP-RAG, a task-aware
policy-controlled RAG framework for long-
document multimodal QA. TAP-RAG contains
a main controller, the Task-Aware Policy Con-
troller (TAPC), and two policy-guided evi-
dence executors: Task-Aware Query-Guided
Flow Diffusion (TA-QFD) and Task-Aware Vi-
sual Enhancement (TA VE). For each query,
TAPC predicts the task prior, estimates vi-
sual/local/global evidence signals, and pro-
duces an executable policy. TA-QFD then ex-
pands textual and structural evidence over the
multimodal document graph, while TA VE se-
lectively inspects page images when visual or
layout evidence is needed. A guarded syn-
thesis stage fuses text, visual, and structural
evidence and abstains when support is insuf-
ficient. On DocBench and MMLongBench-
Doc, TAP-RAG achieves the best overall ac-
curacy among the compared systems, im-
proving over a matched multimodal-RAG
baseline by +9.1 points ( 61.1→70.2 ) and
+4.5 points ( 42.2→46.7 ), respectively. Code
is available at https://anonymous.4open.
science/r/TAP-RAG.
1 Introduction
Long-document multimodal question answering
(QA) requires evidence selection across paragraphs,
tables, figures, captions, page layout, metadata, and
document structure. A query may ask for a page-
local fact, a table cell, a cross-page aggregation,
a chart-level relation, a document identifier, or an
unsupported claim that should be refused (Ma et al.,
Different types of 
queries need different 
evidences.Generic RAG
Supported answer
Verify: abstain if needed· Underreach
Global: broaden searchLocal: stay near page· Overreach
· Weak supportFailures
Task priorAnswer
Policy
A s s e t s
readingUniform
scopeFixed
TAP-RAGMultimodal ReportFigure 1: Motivating example for policy-controlled mul-
timodal RAG. Generic RAG uses fixed scope and uni-
form reading, which can cause overreach, underreach,
or weak support. TAP-RAG instead uses task priors and
policies to adapt evidence use for supported answers.
2024; Zou et al., 2025). Retrieval-augmented gener-
ation (RAG) improves grounding by providing ex-
ternal evidence to the generator (Lewis et al., 2020).
Recent graph-based and multimodal RAG systems
further expand retrievable evidence through enti-
ties, document units, page images, layout regions,
and structural links (Edge et al., 2024; Guo et al.,
2025b; Wan and Yu, 2025; Guo et al., 2025a). How-
ever, better access to evidence does not automati-
cally determine how that evidence should be used
for a specific query.
Figure 1 illustrates the motivation. Page-local
questions should remain close to page-specific ev-
idence and layout. Table-value questions require
row-column alignment rather than only semantic
similarity. Aggregation questions need broader
traversal across sections or repeated document
structures. Visual questions may require selected
page-image inspection, but visual evidence should
not freely overwrite a well-supported textual an-
swer. Verification and unanswerable questions re-
quire stricter support thresholds and conservative
synthesis. A query-agnostic pipeline can therefore
over-expand local questions, under-explore global
1
arXiv:2607.18917v1  [cs.CV]  21 Jul 2026

ones, over-trust visual guesses, or answer without
sufficient support.
We present TAP-RAG, a Task-Aware Policy
RAG framework for long-document multimodal
QA. TAP-RAG is the overall framework. In-
side TAP-RAG, the Task-Aware Policy Controller
(TAPC) is the main control module. TAPC does
not directly answer the question; instead, it inter-
prets the query, predicts a task prior, estimates
visual/local/global evidence needs, and converts
them into an executable policy. This policy speci-
fies retrieval locality, seed budget, graph diffusion
strength, metadata anchoring, visual acquisition
thresholds, fusion authority, structural operators,
support thresholds, and abstention behavior.
TA-QFD and TA VE are two sibling evidence-
execution modules under TAP-RAG. They are
not independent pipelines and are not subparts of
TAPC. Rather, both execute the policy produced by
TAPC. Task-Aware Query-Guided Flow Diffusion
(TA-QFD) operates on the multimodal document
graph and expands textual and structural candidates
according to the policy’s locality, scope, and seed
controls. Task-Aware Visual Enhancement (TA VE)
selectively inspects page images when the policy in-
dicates visual dependence or when textual evidence
is insufficient. In this way, TAPC decides what
evidence behavior is needed, while TA-QFD and
TA VE acquire the corresponding evidence through
graph and visual channels.
Finally, TAP-RAG applies guarded synthesis to
combine the outputs of these modules. Evidence
from text, visual pages, and structural operators is
fused under the policy’s fusion and support con-
straints. If no candidate answer is sufficiently sup-
ported, the system abstains instead of forcing an
answer. This separation between policy control,
graph execution, visual execution, and final synthe-
sis makes TAP-RAG more explicit and auditable
than a fixed retrieval-generation pipeline.
Contributions.
•We propose TAP-RAG, a task-aware policy-
controlled RAG framework for long-document
multimodal QA, with TAPC as its core control
module for mapping query-level evidence re-
quirements into executable policies.
•We introduce TA-QFD and TA VE as auxiliary
policy-guided modules in TAP-RAG, enabling
graph-based evidence expansion and selective
visual inspection under TAPC’s control.•We evaluate TAP-RAG on DocBench and
MMLongBench-Doc, showing consistent gains
over strong multimodal-RAG baselines and ana-
lyzing module interaction through ablations and
case studies.
2 Related Work
Long-document and multimodal document QA.
Recent benchmarks emphasize cross-page reason-
ing, visual grounding, fine-grained document un-
derstanding, and insufficient-evidence cases (Ma
et al., 2024; Zou et al., 2025). These benchmarks
expose mixed evidence requirements: some ques-
tions can be answered from one local span, while
others require table structure, image layout, or
document-level comparison. TAP-RAG targets this
mixed setting by using one multimodal document
representation while varying query-time evidence
behavior through an explicit policy.
Multimodal representation and document RAG.
RAG grounds language-model outputs in exter-
nal evidence (Lewis et al., 2020). GraphRAG
uses graph communities for query-focused re-
trieval (Edge et al., 2024); LightRAG combines
graph and vector retrieval (Guo et al., 2025b);
MMGraphRAG extends graph retrieval to multi-
modal document QA (Wan and Yu, 2025); and sys-
tems such as ColPali, VisRAG, M3DocRAG, and
RAG-Anything add page-image retrieval or het-
erogeneous document-element handling (Faysse
et al., 2025; Yu et al., 2024; Cho et al., 2024;
Guo et al., 2025a). Orthogonal MLLM-design
work improves modality coordination and image-
text alignment through modality-expert adaptation,
momentum-contrast semantic enhancement, and
keyword-explicit reasoning (Zhang et al., 2024a,b;
Ji et al., 2024). TAP-RAG is complementary:
it focuses on how retrieved evidence should be
expanded, inspected, fused, or rejected for each
query.
Routing and abstention.Self-consistency im-
proves reasoning by aggregating multiple samples
(Wang et al., 2023). Adaptive RAG methods decide
whether retrieval is needed or route among coarse
retrieval-generation paths (Asai et al., 2024; Yan
et al., 2024; Jeong et al., 2024). TAP-RAG uses
sampling inside TAPC, but maps the routed task
prior and evidence signals into field-level controls
over diffusion, visual acquisition, fusion, structural
operators, support checking, and abstention.
2

Input Document
Graph
"Which page shows the
regional revenue chart?"Policy-Contr olled
Evidence Acquisition 
TAVE:V isual
Evidence AcquisitionGuarded Synthesis 
& Response
Guarded
LLM
Unsupported
by evidence
prevents unsupported generation
Local
Neighborhood
Other
Nodes
Structural
Tools
Vision
gateSelect
PagesVLM
EvidencePage 14 -- Regional
Revenue Chart''Validity-
Weighted 
Voting
0.72Task-A ware Policy Contr oller
(TAPC)
Visual Gate Opera tors
0.18
0.10Task Prior π
Evidence Signals a
1.0
0
Confidence CFigure 2: Overview of TAP-RAG. TAPC produces the task prior, evidence signals, confidence, and executable
policy. TA-QFD and TA VE acquire graph and visual evidence under the policy, and guarded synthesis prevents
unsupported generation.
3 Method
Figure 2 summarizes TAP-RAG. The framework
assumes a parsed multimodal document graph and
focuses on query-time control. TAPC first resolves
the evidence behavior required by the query, then
converts it into an executable policy. The policy
controls graph expansion, visual acquisition, fusion
authority, structural operators, and support check-
ing. This design makes the interaction between
modules explicit: TAPC decides the behavior, TA-
QFD and TA VE acquire evidence under that behav-
ior, and guarded synthesis determines whether the
final answer is sufficiently supported.
3.1 Document Graph and Policy
Multimodal document graph.Let qbe a query
andG= (V, E)a multimodal document graph:
V=V p∪Vt∪Vtab∪Vf∪Vm,(1)
where Vp,Vt,Vtab,Vf, and Vmdenote page,
text-unit, table-region, figure/image, and metadata-
anchor nodes. The edge set is
E=E contain ∪E next∪E caption ∪E semantic
∪E layout∪E cross,
(2)
covering containment, sequence, caption, semantic,
layout, and cross-modal relations. The graph there-
fore represents both textual continuity and docu-
ment structure, allowing the controller to choose
between local and broad evidence traversal.Policy definition.A policy is a structured execu-
tion vector
π(q) = (α diff, Kseed, Bscope, Bmeta,
τtext, τacq, τsup, ffusion,
aabs, Ostruct).(3)
Table 1 lists the policy fields. Continuous fields
tune retrieval locality and thresholds; discrete fields
constrain fusion and structural execution. Given
q, TAPC resolves a task prior t, evidence signals
a= (av, al, ag), and confidenceC. The task prior
indexes the Task Policy Table, πt= TPT(t) , and
bounded query control refines it intoπ′(q).
3.2 Task-Aware Policy Controller (TAPC)
TAPC is the control layer of TAP-RAG. It routes
the query to a TPT category, aggregates evidence-
need signals, and converts the result into executable
controls for TA-QFD, TA VE, and guarded synthe-
sis. This separates task interpretation from evi-
dence execution: the model does not merely re-
trieve more context, but decides what kind of evi-
dence behavior is needed before retrieval expansion
and visual inspection.
Task Policy Table (TPT).The TPT contains seven
categories: metadata lookup, evidence localization,
document aggregation, value extraction, visual rea-
soning, Boolean verification, and generic reasoning.
Each category defines a base evidence-use behav-
ior. For example, metadata lookup emphasizes title-
page and header anchors, value extraction favors
local table or numeric evidence, and aggregation
allows broader traversal. Ambiguous queries are
3

Field Type Mut. Controlled stage Runtime role
αdiff cont. yes Graph diffusionSets restart/locality strength: larger values keep evidence near seeds,
while smaller values allow broader traversal.
Kseed int. yes Seed selectionControls how many semantic and structural seed nodes initialize
diffusion.
Bscope cont. yes Scope protection Penalizes cross-document or off-scope evidence.
Bmeta cont. task Metadata anchoring Boosts title-page, header, author, and document-identifying anchors.
τtext cont. yes Text gateDetermines whether text evidence is sufficient without visual
inspection.
τacq cont. yes Visual acquisitionTriggers page-image inspection when visual evidence is likely
needed.
τsup cont. yes Support checking Sets the minimum support required before accepting an answer.
ffusion disc. constr. Cross-modal fusionChooses whether visual evidence validates, supplements, or revises
text evidence.
aabs cont. yes Abstention Raises the refusal prior when evidence support is weak.
Ostruct set constr. Structural executionEnables page lookup, metadata extraction, scoped counting, or
table-cell validation.
Table 1: Policy vector fields used by TAP-RAG. Each component links task interpretation to concrete execution
behavior.
further regulated by evidence signals and routing
confidence instead of relying only on a hard label.
Validity-weighted voting.TAPC samples KLLM
completions si= (t i, zi, ri, hi, ai), where tiis the
TPT category, zistores interpretation fields, riis
a rationale, hicontains optional hints, and ai∈
[0,1]3estimates visual, local, and global evidence
needs. Each sample receives
vi=w sSsem(si) +w fSfield(si)
+wgSground (si), w s+wf+wg= 1.
(4)
Ssemchecks task-answer compatibility, Sfieldmea-
sures agreement with neighboring samples, and
Sground checks whether the predicted exact term is
grounded in the query.
The final task prior and evidence signals are
t= arg max
ℓX
iviI[ti=ℓ],
am=P
iviam
iP
ivi, m∈ {v, l, g}.(5)
Confidence combines vote concentration and signal
consistency:
C=λ tCt+λaCa, λ t+λa= 1,
Ct=max ℓP
iviI[ti=ℓ]P
ivi,
Ca= 1−1
3X
m∈{v,l,g}NormVar({am
i}K
i=1).(6)
NormVar is clipped to [0,1] . High confidence al-
lows stronger query-level refinement, while low
confidence keeps the policy closer to the conserva-
tive TPT default.Bounded query control.Table 2 defines the base
policy row for each TPT category. TAPC computes
π′(q) =π t⊕∆(a, C) , where ⊕denotes clip-then-
substitute composition. For a mutable continuous
fieldj,
π′
j=πt,j+M t,jCclip(η jgj(a, t),
−ϵj, ϵj).(7)
Mt,jis the mutability mask, ηjis a step size, ϵj
bounds perturbation, and gj(a, t) gives the con-
trol direction. Discrete fields choose the highest-
scoring allowed action:
Score(r) =ω 1Compat(r, π t) +ω 2EviFit(r, a)
−ω3Risk(r, t)(1−C).
(8)
The selected fusion action is f′
fusion=
arg max r∈A tScore(r) . Locked fields remain
unchanged, so query-level adaptation cannot
override hard scope or safety constraints.
3.3 Policy-Controlled Evidence Acquisition
Task-Aware Query-Guided Flow Diffusion (TA-
QFD).Task-Aware Query-Guided Flow Diffusion
conditions seed budget, locality, node priors, edge
weights, and scope protection on π′(q). From
semantic-structural seedss, each nodevreceives
rv=µ 1sim(q, v) +µ 2lex(q, v)
+µ3bv(π′(q)).(9)
Here bvboosts nodes such as page-hint neighbor-
hoods, metadata anchors, visual regions, or global
evidence depending on the policy. This allows the
4

Task category Retrieval scope Visual prior Fusion action Structural operators
Metadata lookup anchor-local conditional text-validate metadata and page operators
Evidence localization local/scope conditional conservative page and section operators
Document aggregation broad low/conditional text-first scoped count when complete
Value extraction local conditional/high cross-validate parser operators if structured
Visual reasoning visual-local high visual-gated disabled by default
Boolean verification claim-scope conditional conservative optional if structured
Generic reasoning adaptive conditional balanced disabled by default
Table 2: Compact Task Policy Table (TPT) inside TAPC. Each row records base evidence-use behavior, which
bounded query control refines using visual, local, and global evidence signals.
same graph to support local lookup, table extrac-
tion, and document aggregation without changing
the underlying index.
Edges are reweighted as w′
uv=w uv·Auv·Buv:
Auv= exp
γrel(u) + rel(v)
2−0.3
,
Buv= max 
0.01,1−σI[crossdoc(u, v)]
.
(10)
With row-normalized transitionP′, diffusion is
p(k+1)=α diffs
+ (1−α diff)P′⊤p(k).(11)
Larger αdiffkeeps evidence local; smaller val-
ues allow broader traversal. Thus, TA-QFD im-
proves candidate coverage while preserving the
controller’s decision about whether the query
should remain local or expand globally.
Task-Aware Visual Enhancement (TA VE).Task-
Aware Visual Enhancement first produces a text-
path candidate atand estimates textual sufficiency:
Stext=a1Coverage(q, E t) +a 2Rel(q, E t)
+a3Support(a t, Et).
(12)
The VLM is invoked when Stext< τtextorav>
τacq. Candidate pages are scored by
P(p) =b 1RetrievalPage(p) +b 2HintMatch(p, h)
+b3MetaAnchor(p) +b 4avVisPrior(p).
(13)
For visual candidate av, revision is allowed only
when visual grounding is strong and text support is
weak:
Revise(a t, av) =I[R v> τrel∧Sv> τsup
∧Q(a v)> Q(a t) +ϵ q
∧f′
fusion∈ A revise].(14)
Otherwise, visual evidence validates or supple-
ments the text path without freely overwriting it.
This guarded behavior is important for visually
dense PDFs, where plausible visual interpretations
can conflict with table structure or extracted text.3.4 Guarded Synthesis
For each candidatea, TAP-RAG scores
Q(a) =d 1Ssup(a, E) +d 2Scomp(a, q)
+d3Sspec(a).(15)
Support is channel-aware:
Ssup(a, E) = max{Stext
sup(a, E t), Svis
sup(a, E v),
Sstruct
sup(a, E s)}.
(16)
Text support comes from retrieved spans, visual
support from inspected pages or regions, and struc-
tural support from deterministic operators such as
table-cell or page validation. If no candidate meets
the policy-adjusted support threshold, TAP-RAG
abstains. This final step closes the loop between
policy and execution: retrieval and visual inspec-
tion propose evidence, but the answer is accepted
only when the selected policy’s support require-
ment is satisfied.
4 Experiments
4.1 Experimental Setup
Benchmarks and metrics.We evaluate on
DocBench and MMLongBench-Doc. DocBench
covers academic, financial, government, legal,
and news PDFs with text-only, multimodal,
and unanswerable questions (Zou et al., 2025).
MMLongBench-Doc covers long-context docu-
ment understanding over reports, tutorials, papers,
guidebooks, brochures, administration files, and fi-
nancial reports (Ma et al., 2024). We report answer
accuracy in percent and use the same benchmark
splits and page-range buckets across systems.
Compared systems.The comparison in-
cludes GPT-4o-mini (OpenAI, 2024), Qwen3-
VL-Plus (Qwen Team, 2025), GraphRAG (Edge
et al., 2024), LightRAG (Guo et al., 2025b),
MMGraphRAG (Wan and Yu, 2025), and RAG-
Anything (Guo et al., 2025a). GPT-4o-mini
5

and Qwen3-VL-Plus are direct model base-
lines. GraphRAG uses GPT-4o while lightrag
uses GPT-4o-mini as the generation backbone.
MMGraphRAG is a published-reference result
from the RAG-Anything paper under its GPT-4o-
mini setting. RAG-Anything is evaluated with
Qwen3-VL-Plus as the backbone, and TAP-RAG
uses the same Qwen3-VL-Plus backbone for gen-
eration and page-image vision calls.
Diagnostic view.Domain and length results mea-
sure overall robustness. Since TAP-RAG’s central
claim is query-level evidence control, mechanism-
level evidence is provided by the TAPC/TA-
QFD/TA VE ablation and case studies. This avoids
treating domain labels as a proxy for task type and
instead evaluates whether the controller and execu-
tors interact as intended.
4.2 Domain-Level Accuracy
Table 3 shows that TAP-RAG achieves the best
overall accuracy on both benchmarks. Compared
with the strongest same-backbone baseline, RAG-
Anything with Qwen3-VL-Plus, TAP-RAG im-
proves by +9.1 points on DocBench and +4.5
points on MMLongBench-Doc. The gains span
many domains, especially government, legal, news,
research, tutorial, academic, guidebook, adminis-
tration, and financial documents.
These results support the motivation of TAP-
RAG: long-document multimodal QA requires con-
trolled evidence use, not only more retrieval. TAPC
selects a task-aware policy before evidence acquisi-
tion, adjusting retrieval scope, visual inspection, fu-
sion, and abstention before answer synthesis. The
DocBench multimodal and unanswerable columns
remain challenging, suggesting room for stronger
visual recall and refusal calibration.
4.3 Performance by Document Length
Figure 3 evaluates whether the method remains
useful as documents become longer. TAP-RAG is
competitive across all page ranges and is strongest
in the longest MMLongBench-Doc buckets, where
relevant evidence often spans headings, tables, cap-
tions, and explanatory text. This pattern is consis-
tent with the design of TAPC and TA-QFD: longer
documents require broader but still bounded evi-
dence traversal, while the policy prevents expan-
sion from drifting into irrelevant pages. TA VE
further helps when long reports contain visually
organized tables or page-level cues that are dif-
ficult to recover from extracted text alone. OnDocBench, medium-length documents remain dif-
ficult because they contain enough distractors to
confuse retrieval but fewer repeated structures for
diffusion to reinforce. Exact page-range values are
listed in Appendix Table 9.
4.4 Ablation and Executor Coupling
Table 4 evaluates TAPC and the two policy-guided
executors. The goal is to test interaction rather
than a purely additive module stack: TA-QFD ex-
pands graph candidates, while TA VE verifies se-
lected pages visually. The full system uses both un-
der the same policy and then applies support-gated
synthesis. This setting directly tests whether ex-
ecutor behavior remains beneficial when evidence
expansion, visual checking, and final acceptance
are governed by one shared controller.
TAPC only is already strong, showing that query-
level policy resolution is the main source of control.
It adjusts locality, visual-trigger thresholds, fusion
authority, structural operators, and support require-
ments before generation. However, TAPC alone
cannot recover evidence missing from the initial
candidate set, and it cannot directly inspect layout-
sensitive pages.
TA-QFD improves candidate coverage through
graph diffusion, which benefits text-heavy and
structure-heavy documents. Yet expansion can also
introduce semantically related distractors when the
query is highly local. TA VE provides the com-
plementary behavior by verifying candidate pages
visually, but it depends on the quality of candi-
date pages. If retrieval misses the correct page or
includes visually plausible distractors, visual in-
spection alone can still be misled.
The full model is strongest because the modules
form a controlled loop. TAPC selects the policy,
TA-QFD expands evidence under the selected lo-
cality and scope constraints, TA VE verifies layout
or visual evidence only when needed, and guarded
synthesis prevents noisy graph or visual candidates
from dominating the final answer. This interaction
explains why the two executors are most effective
when coordinated rather than used as independent
add-ons.
4.5 Case Study
The case studies illustrate how TAP-RAG uses one
policy interface for different evidence behaviors.
Figure 4 shows a table-cell extraction case from
case1.pdf. The query asks for Novo Nordisk’s
wages and salaries in 2020. The table contains
6

DocBench
Method Aca. Fin. Gov. Law News Txt. Mm. Una. All
GPT-4o-mini (OpenAI, 2024) 40.3 46.9 60.3 59.2 61.0 61.0 43.8 49.6 51.2
Qwen3-VL-Plus (Qwen Team, 2025) 53.4 45.0 60.0 59.4 62.1 63.9 56.8 43.1 54.5
GraphRAG (Edge et al., 2024) 40.6 27.1 56.8 59.7 75.0 73.5 24.4 76.6 54.7
LightRAG (Guo et al., 2025b) 53.8 56.2 59.5 61.8 65.7 85.0 59.7 46.8 58.4
MMGraphRAG (Wan and Yu, 2025) 64.3 52.8 64.9 40.0 61.5 67.6 66.0 60.5 61.0
RAG-Anything (Guo et al., 2025a)
(Qwen3-VL-Plus backbone)64.1 59.2 55.8 54.0 68.9 72.169.5 87.161.1
TAP-RAG67.1 64.3 75.0 74.7 76.2 87.464.9 71.870.2
MMLongBench-Doc
Method Res. Tut. Acad. Guid. Bro. Adm. Fin. All
GPT-4o-mini (OpenAI, 2024) 35.5 44.0 24.6 33.1 29.5 46.8 31.1 33.5
Qwen3-VL-Plus (Qwen Team, 2025) 39.2 37.1 34.6 38.6 31.7 35.9 34.0 36.5
GraphRAG (Edge et al., 2024) 30.8 27.0 25.0 29.7 24.0 34.4 16.7 27.2
LightRAG (Guo et al., 2025b) 40.8 34.1 36.2 39.441.044.4 38.3 38.9
MMGraphRAG (Wan and Yu, 2025) 40.8 36.5 35.7 35.8 28.2 46.9 38.5 37.7
RAG-Anything (Guo et al., 2025a)
(Qwen3-VL-Plus backbone)46.6 43.5 38.7 43.9 34.0 45.7 43.6 42.2
TAP-RAG50.9 47.4 43.9 48.938.549.7 47.6 46.7
Table 3: Domain-level answer accuracy on DocBench and MMLongBench-Doc. Bold values mark the best result
in each column. MMGraphRAG uses GPT-4o-mini as its backbone, while RAG-Anything and TAP-RAG use
Qwen3-VL-Plus.
RAG-Anything MMGraphRAG TAP-RAG
1–10 11–50 51–100 101–200 200+4050607080
Page RangeAccuracy (%)
(a) DocBench1–10 11–50 51–100 101–200 200+0204060
Page RangeAccuracy (%)
(b) MMLongBench-Doc
Figure 3: Accuracy by document page range for TAP-RAG, RAG-Anything, and MMGraphRAG under the same
page-range buckets.
VariantDocBench MMLongBench-Doc
Aca. Fin. Gov. Law News All Res. Tut. Acad. Guid. Bro. Adm. Fin. All
Baseline 64.1 59.2 55.8 54.0 68.9 61.1 46.6 43.5 38.7 43.9 34.0 45.7 43.6 42.2
TAPC only 66.6 63.9 74.3 73.9 75.0 69.6 50.1 47.1 43.5 48.6 37.7 49.3 47.2 46.6
TAPC + TA-QFD 66.9 63.2 74.9 74.1 74.3 69.1 50.7 46.8 43.8 47.9 36.8 49.0 46.6 46.1
TAPC + TA VE 65.8 64.1 73.3 72.8 75.8 68.7 48.7 45.9 42.8 48.8 38.2 48.4 47.4 45.5
Full TAP-RAG67.1 64.3 75.0 74.7 76.2 70.2 50.9 47.4 43.9 48.9 38.5 49.7 47.6 46.7
Table 4: Domain-level ablation of TAPC and the two policy-guided evidence executors on DocBench and
MMLongBench-Doc. Full TAP-RAG couples TA-QFD candidate expansion, TA VE page verification, and guarded
synthesis.
adjacent values for pension costs, social security
contributions, and total staff costs, so semantic re-
trieval can select a related but wrong number. TAP-
RAG treats the query as value extraction, uses TA-
QFD to retrieve table-centered evidence, and uses
TA VE to verify the row-column intersection. This
prevents nearby numeric distractors from being ac-
cepted merely because they are semantically close
to the query. The final answer, DKK 26,778 mil-lion, is accepted after structural and visual evidence
agree.
Figure 5 shows a section-level aggregation case
from case2.pdf. The query asks which construction
step has the longest textual description, with the
answer Section 2.3, “Evolutionary Question Gen-
eration.” The challenge is comparing neighboring
sections under a bounded scope. TAP-RAG routes
7

Confuses 
adjacent
rows.GPT-4O
2651420192020
26.776
TAP-RAG (Ours)
26,7882020
xRAG-Anything
Irrelevant.
Ambiguous
numbers
LightRAG
3553
2158
1312
Selects correct
intersection.row-column
26,778 million.
Correct cell Row: Wages
and salariesTAP-RAG
Column: 2020
Cell: 26,778DKK*
Final AnswerCase 1
DKK million 2019 2020
Wages and
salaries24,414 26,778
Pensions 3,265 3,553
Social security
contributions
...1,988 2,158
Total staff
costs31,991 34,855of "Wages and Need the value
salaries"in 2020.Graph-guided
Retrieval(TAPC + TAVE 
+ TA-QFD)
1
 2
 3
 5
 4
UnderstandingTask-aware Fused Evidence
SelectionRefer ence answer: DKK 26,788 million.Question: What was Novo Nordisk's total amount spent on wages and salaries in 2020?
(a) (b)Figure 4: Case study on table-cell value extraction from case1.pdf. (a) Baselines are distracted by adjacent rows or
ambiguous numbers. (b) TAP-RAG follows a task-aware evidence path, verifies the row-column intersection, and
returns DKK 26,778 million.
(a) (b)
Question: Which construction step has the longest textual description?
Section: 2.3TAP-RAG
Word Count: 
Max
Answer: 
Evolutionary*
Final AnswerCase 2
Fused Evidence
Selection(TAPC + TAVE 
+ TA-QFD)
1
 3
 5
 4
Selected via bounded
aggregation and task-
aware verification.Evolutionary
GenerationQuestionFind the step 
withthe longest
 textual
description.
Question 
GenerationGraph-guided
RetrievalTask-aware
Understanding
2Refer ence answer: 2.3 Evolutionary Question GenerationNo length 
comparingGPT-4O
TAP-RAG (Ours)
xRAG-AnythingScope drift 
to other  
chunks
No unique 
max-length 
decision
LightRAG
2.2 ...
2.4 ...2.3 ...
2.2 ...
2.3 ...2.1 ...
2.2 ...
2.4 ...2.3 ...(longest)Got the 
right and
complete
section
Figure 5: Case study on section-level textual aggregation from case2.pdf. (a) Baselines retrieve related sections but
do not make a reliable maximum-length decision. (b) TAP-RAG performs bounded aggregation over candidate
sections and selects Section 2.3, “Evolutionary Question Generation.”
the query as aggregation-oriented, expands within
the relevant section range, and selects the answer
after support-aware comparison. Thus, the system
does not stop at the first high-overlap section, but
compares candidate sections according to the re-
quested property.
Together, the cases show why TAP-RAG treats
long-document multimodal QA as controlled evi-
dence use. The first needs local structural precision
and visual validation; the second needs broader but
bounded aggregation. TAPC links both through
policy fields for locality, visual acquisition, fusion
authority, structural operators, and support thresh-
olds, allowing TA-QFD and TA VE to work as coor-
dinated executors rather than independent retrieval
add-ons. The examples also expose distinct base-
line failure modes, making the policy decisions and
evidence paths easier to audit and compare.
5 Conclusion
We presented TAP-RAG, a task-aware policy-
controlled framework for long-document multi-
modal QA. TAPC maps each query into an exe-
cutable policy controlling graph diffusion, visual
acquisition, fusion, structural operators, support
checking, and abstention. This makes RAG be-havior explicit and auditable: questions over the
same document can use different retrieval scopes,
inspection strategies, support thresholds, and re-
fusal rules.
Experiments on DocBench and MMLongBench-
Doc show gains across domains and page ranges,
including long-document settings. The same-
backbone comparison with RAG-Anything shows
that improvements come from query-time evidence
control, not a stronger generator. Ablations show
that TA-QFD and TA VE are most effective with
TAPC and guarded synthesis.
These results suggest treating long-document
multimodal QA as controlled evidence use rather
than retrieval scaling. TAP-RAG offers an inter-
pretable policy interface, while future work can
improve policy calibration, visual region selection,
and task diagnostics for reliable document-level
reasoning. These observations emphasize that TAP-
RAG improves reliability not by adding context,
but by deciding when, where, and how evidence
should contribute safely. This direction supports
stronger human oversight when document evidence
is incomplete or internally conflicting, especially
in high-stakes document reasoning scenarios.
8

Limitations
TAP-RAG is evaluated on DocBench and
MMLongBench-Doc with a fixed parser, judge pro-
tocol, and LLM/VLM backend, so results may vary
with different document parsers, models, or bench-
mark distributions. The task category set covers
common long-document multimodal QA behaviors,
but unusual layouts, languages, or domain conven-
tions may require new categories or recalibration.
This paper reports domain-level, length-level,
ablation, and qualitative analyses. Because the
core claim concerns task-specific evidence behav-
ior, task-type-level accuracy would be a valuable
additional diagnostic when reliable task labels are
available. The current results should therefore be
interpreted together with ablations and case studies
rather than from the domain table alone.
Finally, the policy fields are manually designed
rather than learned end-to-end. This improves in-
terpretability and auditability, but may miss dataset-
specific strategies that a learned controller could
discover.
Ethical Considerations
This work studies document question answering on
benchmark-style documents and does not introduce
new human-subject data. The main ethical con-
cern is unsupported answer generation in document
analysis workflows. TAP-RAG includes evidence
support validation and abstention, but these mech-
anisms are not guarantees in high-stakes settings
such as legal, financial, or medical decision making.
Practical deployments should preserve document
privacy, respect data-use restrictions, avoid storing
secrets in environment files committed to reposi-
tories, and require appropriate human review for
consequential decisions.
Use of AI Assistants.AI assistants were used
only for language polishing, LaTeX formatting, and
checklist-writing assistance. All scientific content,
experiments, results, and final manuscript decisions
were reviewed and approved by the authors.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-RAG: Learning to
retrieve, generate, and critique through self-reflection.
InProceedings of the International Conference on
Learning Representations.
Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie
He, and Mohit Bansal. 2024. M3DocRAG: Multi-modal retrieval is what you need for multi-page
multi-document understanding.Computing Research
Repository, arXiv:2411.04952.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Tru-
itt, and Jonathan Larson. 2024. From local to
global: A graph RAG approach to query-focused
summarization.Computing Research Repository,
arXiv:2404.16130.
Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani,
Gautier Viaud, Céline Hudelot, and Pierre Colombo.
2025. ColPali: Efficient document retrieval with
vision language models. InProceedings of the Inter-
national Conference on Learning Representations.
Zirui Guo, Xubin Ren, Lingrui Xu, Jiahao Zhang, and
Chao Huang. 2025a. RAG-Anything: All-in-one
RAG framework.Computing Research Repository,
arXiv:2510.12323.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao
Huang. 2025b. LightRAG: Simple and fast retrieval-
augmented generation. InFindings of the Associa-
tion for Computational Linguistics: EMNLP 2025,
pages 10746–10761. Association for Computational
Linguistics.
Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju
Hwang, and Jong C. Park. 2024. Adaptive-RAG:
Learning to adapt retrieval-augmented large language
models through question complexity. InProceedings
of the 2024 Conference of the North American Chap-
ter of the Association for Computational Linguistics:
Human Language Technologies, pages 7036–7050.
Association for Computational Linguistics.
Zhong Ji, Changxu Meng, Yan Zhang, Haoran Wang,
Yanwei Pang, and Jungong Han. 2024. Eliminate
before align: A remote sensing image-text retrieval
framework with keyword explicit reasoning. Pro-
ceedings of the 32nd ACM International Conference
on Multimedia (ACM MM 2024), pp. 1662–1671.
doi:10.1145/3664647.3681270.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive NLP tasks. InAdvances in Neural Infor-
mation Processing Systems, volume 33, pages 9459–
9474.
Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen,
Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma,
Xiaoyi Dong, Pan Zhang, Liangming Pan, Yu-Gang
Jiang, Jiaqi Wang, Yixin Cao, and Aixin Sun. 2024.
MMLongBench-Doc: Benchmarking long-context
document understanding with visualizations. InAd-
vances in Neural Information Processing Systems.
OpenAI. 2024. GPT-4o Mini: Advancing cost-efficient
intelligence. Technical report, OpenAI.
9

Qwen Team. 2025. Qwen3-VL: Multimodal foundation
model with extended visual reasoning. Technical
report, Alibaba Group.
Xueyao Wan and Hang Yu. 2025. MMGraphRAG:
Bridging vision and language with interpretable mul-
timodal knowledge graphs.Computing Research
Repository, arXiv:2507.20804.
Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V .
Le, Ed H. Chi, Sharan Narang, Aakanksha Chowd-
hery, and Denny Zhou. 2023. Self-consistency im-
proves chain of thought reasoning in language mod-
els. InProceedings of the International Conference
on Learning Representations.
Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling.
2024. Corrective retrieval-augmented generation.
Computing Research Repository, arXiv:2401.15884.
Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Jun-
hao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang,
Xu Han, Zhiyuan Liu, and Maosong Sun. 2024. Vis-
RAG: Vision-based retrieval-augmented generation
on multi-modality documents.Computing Research
Repository, arXiv:2410.10594.
Yan Zhang, Zhong Ji, Yanwei Pang, Jungong Han,
and Xuelong Li. 2024a. Modality-experts coor-
dinated adaptation for large multimodal models.
Science China Information Sciences, 67:220107.
doi:10.1007/s11432-024-4234-4.
Yan Zhang, Zhong Ji, Di Wang, Yanwei Pang, and
Xuelong Li. 2024b. USER: Unified semantic en-
hancement with momentum contrast for image-text
retrieval. IEEE Transactions on Image Processing,
33:595–609. doi:10.1109/TIP.2023.3348297.
Anni Zou, Wenhao Yu, Hongming Zhang, Kaixin Ma,
Deng Cai, Zhuosheng Zhang, Hai Zhao, and Dong
Yu. 2025. DocBench: A benchmark for evaluating
LLM-based document reading systems. InProceed-
ings of the 4th International Workshop on Knowledge-
Augmented Methods for Natural Language Process-
ing. Association for Computational Linguistics.
A Executable Runtime Configuration
This appendix reports the executable settings used
in the reported TAP-RAG run. The configuration is
derived from the runtime scripts and query-time
policy modules. Environment variables are re-
solved at startup, legacy aliases are normalized,
and required model and embedding credentials are
checked before evaluation. The run uses a fixed lan-
guage/vision backbone, deterministic main genera-
tion, TAPC-based policy control, selective visual
acquisition, and bounded evidence budgets.B Task-Aware Runtime Flow
TAP-RAG processes each query with a task-
aware runtime flow rather than applying one fixed
retrieval-and-generation behavior to all questions.
The pipeline first resolves document scope, nor-
malizes runtime flags, and constructs a task state
containing the predicted task type, target, rela-
tion, expected answer form, page hints, visual re-
quirement, locality signal, globality signal, and
structural-operator permissions.
The first stage is TAPC. Inside TAPC, multi-
ple sampled LLM routers assign the query to one
of seven TPT categories: metadata lookup, evi-
dence localization, document aggregation, value
extraction, visual reasoning, Boolean verification,
or generic reasoning. Each sample also emits aux-
iliary evidence signals that estimate whether the
query requires visual inspection, local page evi-
dence, or broader document traversal. The final
task prior is selected by weighted vote concentra-
tion, while the evidence signals are averaged to
form the query-time policy inputs.
The Task Policy Table provides a stable default
behavior for each category. Metadata lookup em-
phasizes title-page and header evidence; evidence
localization prefers local page-specific evidence;
document aggregation permits broader traversal;
value extraction enables table or numeric opera-
tors; visual reasoning allows selective page-image
inspection; and Boolean verification uses stricter
support requirements. These defaults make the con-
troller interpretable and prevent runtime behavior
from depending only on free-form generation.
Bounded query control then applies constrained
adjustments to the table policy. A high visual sig-
nal lowers the threshold for TA VE page inspection.
A high local signal keeps retrieval close to the most
relevant page or section. A high global signal al-
lows broader TA-QFD evidence expansion. These
adjustments can tune thresholds, budgets, and fu-
sion behavior, but they do not override hard safety
rules such as document scope, support checking, or
abstention.
The evidence acquisition stage combines text
retrieval, structural evidence, optional TA-QFD ex-
pansion, and optional TA VE page inspection. The
final response stage combines text, structural, and
visual evidence under the selected policy. Visual
evidence can validate or supplement the text path,
but it is not allowed to freely overwrite a well-
supported textual answer.
10

Group Setting Value used in the reported run
Backbone LLM/VLM Qwen3-VL-Plus for both language generation and page-image vision calls.
Backbone Embeddingtext-embedding-v3 with 1024 dimensions. The embedding dimension is
fixed to keep indexing and retrieval consistent.
Generation Main temperature0 for the main answer path; sampling is used only inside TAPC voting and
routing.
Parsing ParserMinerU with automatic parsing. Image, table, and equation processing are
enabled to construct multimodal document chunks.
TAPC votingSamples / temperature /
agreementK= 5, sampling temperature 0.7, and agreement threshold 0.6.
TAPC voting Validation switchesStructural validation, field coherence, query-document consistency, and final
task-decision checking are enabled.
TAPC policy Task policy table updateThe Task Policy Table provides task-level defaults, and bounded query
control applies visual, local, and global evidence signals.
TA-QFD Seeds and outputSeed top-k= 6, output top-k= 10, and include seed nodes disabled. The
evidence expansion budget is controlled by the resolved task policy.
Retrieval Mode and budgetsHybrid mode; retrieval top-k= 8, semantic top-k= 8, structural
top-k= 12, and candidate limit 20.
Local context Context budget Top-8 local contexts, maximum 6000 characters, and maximum 12 files.
TA VE Visual budgetMaximum 2 pages, page window 1, and page-image cache enabled. Visual
calls are triggered selectively by the policy.
Synthesis Response behaviorGuarded synthesis produces a concise answer when support is sufficient and
abstains when the document does not support the requested fact.
Batching Runtime Batch size 1, QA concurrency 3, and resume-completed-documents enabled.
Table 5: Executable runtime settings used in the reported TAP-RAG evaluation.
C Framework Prompt Templates
Tables 6–8 summarize the LLM/VLM prompt tem-
plates used by TAP-RAG. Placeholders such as
query, context, doc_name, and page_number are
filled at runtime. Implementation-only strings
used for deterministic keyword matching, regular-
expression checks, file parsing, or environment
loading are omitted.
D Page-Range Values
Table 9 reports the numerical values used for
the page-range accuracy visualization in Figure 3.
Bold values indicate the best result in each page-
range column.
Figure 6 reports the QA-pair distribution across
page-range buckets for the two benchmarks. These
counts contextualize the page-range accuracy val-
ues in Table 9 and Figure 3: DocBench is rela-
tively more balanced across short and medium doc-
uments, while MMLongBench-Doc concentrates
most QA pairs in medium-length documents.
11

Group Template Runtime use
Answer synthesis Final RAG answerThe assistant answers questions about one specific document. It must be
concise, complete, direct, and document-scoped; it starts with the answer, uses
only retrieved evidence from the target document, avoids cross-document
mixing, and returns “Not mentioned” when the requested fact is absent.
Answer synthesisQuestion-format
guidanceThe prompt adapts the answer shape to the query type: counts return the exact
number plus short item names when useful; page questions return page
numbers only; yes/no questions start with “Yes” or “No”; acronym questions
return only the expansion; content questions remain short.
Document scope Evidence filterGiven a target document and candidate evidence, the prompt separates valid
current-document evidence from cross-document contamination. Clearly
unrelated candidates are removed, while uncertain candidates are retained to
avoid discarding useful evidence prematurely.
Support checkingEvidence sufficiency
checkThe prompt checks whether the available evidence supports the requested
answer. It favors supported concise answers when evidence is adequate and
returns an abstention-style response when the document does not contain the
requested fact.
Guarded responseAnswer consistency
guardThe guard rejects changes that alter numbers, page numbers, polarity, author
identity, or document scope without stronger evidence. It prevents visually
plausible but unsupported replacements.
Conciseness Post-processing rewriteThe prompt shortens an answer without adding information or changing
meaning. It removes filler, unnecessary citations, and unnecessary lists, starts
directly with the answer, and leaves already concise answers unchanged.
Page utilitiesPage count and
existenceThe prompts determine the total number of pages or whether a requested page
exists in the current document only. They return only the page count for
total-page queries, or a compact JSON answer for page-existence checks.
Table 6: Answer synthesis, document-scope control, support checking, and guarded-response prompt templates.
Group Template Runtime use
TAPC routing V oting routerThe router classifies each query into exactly one of seven TPT categories:
metadata lookup, evidence localization, document aggregation, value
extraction, visual reasoning, Boolean verification, or generic reasoning. It
returns JSON with the target, relation, exact term, query rewrite, expected
answer type, reasoning, page hints, confidence, and visual/local/global
evidence signals.
TAPC routingDisagreement
arbitrationWhen TAPC voting candidates disagree, an arbitrator examines the query, vote
summary, and candidate reasoning, then returns the final JSON task decision.
The selected task must match the query’s primary evidence requirement.
Metadata lookupFirst-page metadata
extractionThe prompt extracts title, ordered authors, last author, and corresponding
author from first-page context. It returns JSON and avoids guessing when
author order or correspondence is not supported.
Evidence localization Page locatorGiven the target, relation, and candidate page snippets, the prompt identifies
the earliest page that introduces the requested concept. It prefers introduction
pages over later mentions and considers only pages from the current document.
Document
aggregationCount resolverGiven the exact queried term and current-document text, the prompt returns
JSON with the exact occurrence count and a short explanation. It counts only
current-document content and excludes clearly contaminated evidence.
Value extractionNumeric and table
valuesValue-extraction prompts request precise factual values from text or tables.
Numeric questions focus on explicitly stated quantities, percentages, scores, or
dataset sizes, while table-value questions focus on row-column or cell-level
evidence.
Boolean verificationSupport-oriented
judgmentBoolean queries require a direct Yes or No answer when evidence is sufficient.
If the required relation is absent or ambiguous, the prompt favors conservative
refusal rather than unsupported affirmation.
Policy signals Evidence signalsThe routing prompt emits visual, local, and global evidence signals in[0,1].
These signals adjust visual-trigger thresholds, fusion mode, evidence-expansion
behavior, and abstention strength through deterministic policy updates.
Table 7: TAPC routing, structural-operator, and policy-control prompt templates.
12

Group Template Runtime use
Multimodal parsing Image analysisThe image prompt asks an expert visual analyst to produce JSON with a
detailed description and entity summary. It describes layout, objects, text,
visual relationships, colors, actions, and technical details, optionally using
surrounding document context.
Multimodal parsing Table analysisThe table prompt asks for JSON with a detailed table description and entity
summary. It analyzes table structure, column headers, key values, trends,
statistical patterns, relationships among data elements, and the table’s
significance in context.
Multimodal parsing Equation analysisThe equation prompt asks for JSON explaining mathematical meaning,
variable definitions, operations, functions, application domain, theoretical or
physical significance, links to surrounding content, and practical use cases.
Multimodal parsingGeneric content
analysisThe generic-content prompt asks for JSON describing structure, key
information, relationships between components, context, significance, and
retrieval-relevant details for non-image, non-table, and non-equation content.
Chunk construction Multimodal chunk textImage, table, equation, and generic chunk templates serialize modality-specific
metadata and enhanced captions into retrieval text, including paths, captions,
footnotes, table structure, equations, and generated analysis.
Query-time analysis Modality summariesQuery-side prompts briefly describe image content, summarize table data,
explain equations, or analyze generic content so that multimodal evidence can
be incorporated into the answer-generation context.
TA VEVisual router and page
inspectionThe visual router decides whether direct page-image inspection is needed. The
page-inspection prompt examines only the provided page image and returns
JSON with relevance, support score, answer-oriented snippet, page summary,
and evidence strings.
Cross-modal fusionDirect visual answer
and fusionThe direct visual-answer prompt answers from a single page image only,
returning NOT_RELEV ANT if the page lacks evidence. The fusion prompt
merges text-path and visual-path answers, resolves conflicts using more specific
evidence, discards cross-document content, and outputs a concise final answer.
Table 8: Multimodal parsing, TA VE page inspection, and cross-modal fusion prompt templates.
DocBench
Method 1–10 11–50 51–100 101–200 200+
MMGraphRAG 63.1 62.0 58.2 54.6 55.0
RAG-Anything 65.162.4 61.368.2 68.8
TAP-RAG69.3260.93 59.2168.97 72.73
MMLongBench-Doc
Method 1–10 11–50 51–100 101–200 200+
MMGraphRAG 13.8 40.8 31.3 34.2 40.0
RAG-Anything 16.5 44.2 40.6 42.1 60.0
TAP-RAG22.4 48.6 45.9 51.2 65.3
Table 9: Page-range accuracy values used in Figure 3. Bold values mark the best result in each column.
1–10 11–50 51–100 101–200 200+0100200300400
330 340
165150
80
Page RangeQA Pair Count
(a) DocBench1–10 11–50 51–100 101–200 200+0200400600800
15740
150 150110
Page RangeQA Pair Count
(b) MMLongBench-Doc
Figure 6: QA-pair distribution by document page range for DocBench and MMLongBench-Doc. The x-axis labels
are normalized to the same page-range buckets used in Table 9.
13