# Route Before Retrieve: Activating Latent Routing Abilities of LLMs for RAG vs. Long-Context Selection

**Authors**: Yiwen Chen, Kuan Li, Fuzhen Zhuang, Deqing Wang, Zhao Zhang, Liwen Zhang, Yong Jiang, Shuai Wang, Minhao Cheng

**Published**: 2026-05-11 09:10:55

**PDF URL**: [https://arxiv.org/pdf/2605.10235v2](https://arxiv.org/pdf/2605.10235v2)

## Abstract
Recent advances in large language models (LLMs) have expanded the context window to beyond 128K tokens, enabling long-document understanding and multi-source reasoning. A key challenge, however, lies in choosing between retrieval-augmented generation (RAG) and long-context (LC) strategies: RAG is efficient but constrained by retrieval quality, while LC supports global reasoning at higher cost and with position sensitivity. Existing methods such as Self-Route adopt failure-driven fallback from RAG to LC, but remain passive, inefficient, and hard to interpret. We propose Pre-Route, a proactive routing framework that performs structured reasoning before answering. Using lightweight metadata (e.g., document type, length, initial snippet), Pre-Route enables task analysis, coverage estimation, and information-need prediction, producing explainable and cost-efficient routing decisions. Our study shows three key findings: (i) LLMs possess latent routing ability that can be reliably elicited with guidelines, allowing single-sample performance to approach that of multi-sample (Best-of-N) results; (ii) linear probes reveal that structured prompts sharpen the separability of the "optimal routing dimension" in representation space; and (iii) distillation transfers this reasoning structure to smaller models for lightweight deployment. Experiments on LaRA (in-domain) and LongBench-v2 (OOD) confirm that Pre-Route outperforms Always-RAG, Always-LC, and Self-Route baselines, achieving superior overall cost-effectiveness.

## Full Text


<!-- PDF content starts -->

Route Before Retrieve: Activating Latent Routing Abilities of LLMs
for RAG vs. Long-Context Selection
Yiwen Chen1Kuan Li2Fuzhen Zhuang1Deqing Wang1Zhao Zhang1
Liwen Zhang3Yong Jiang3Shuai Wang2Minhao Cheng4
1Beihang University2HKUST3Alibaba Group4Pennsylvania State University
yiwenchen@buaa.edu.cn, likuan@connect.ust.hk, zhuangfuzhen@buaa.edu.cn
Abstract
Recent advances in large language mod-
els (LLMs) have expanded the context win-
dow to beyond 128K tokens, enabling long-
document understanding and multi-source rea-
soning. A key challenge, however, lies in
choosing betweenretrieval-augmented gen-
eration (RAG)andlong-context (LC)strate-
gies: RAG is efficient but constrained by re-
trieval quality, while LC supports global rea-
soning at higher cost and with position sen-
sitivity. Existing methods such asSelf-Route
adopt failure-driven fallback from RAG to LC,
but remain passive, inefficient, and hard to in-
terpret. We proposePre-Route, a proactive
routing framework that performs structured rea-
soningbeforeanswering. Using lightweight
metadata (e.g., document type, length, initial
snippet), Pre-Route enables task analysis, cov-
erage estimation, and information-need predic-
tion, producing explainable and cost-efficient
routing decisions. Our study shows three key
findings: (i) LLMs possess latent routing ability
that can be reliably elicited with guidelines, al-
lowing single-sample performance to approach
that of multi-sample (Best-of-N) results; (ii)
linear probes reveal that structured prompts
sharpen the separability of the “optimal rout-
ing dimension” in representation space; and
(iii) distillation transfers this reasoning struc-
ture to smaller models for lightweight deploy-
ment. Experiments on LaRA (in-domain) and
LongBench-v2 (OOD) confirm that Pre-Route
outperforms Always-RAG, Always-LC, and
Self-Route baselines, achieving superior over-
all cost-effectiveness.
1 Introduction
In recent years, Large Language Models (LLMs)
have shown unprecedented capabilities in pro-
cessing long contexts. The latest models—such
as GPT-5, DeepSeek R1, and Qwen3—support
inputs beyond 128K tokens (OpenAI, 2025;
DeepSeek, 2025; Yang et al., 2025), enablingtasks like long-document comprehension, cross-
segment reasoning, and multi-source synthesis.
Yet, how to efficiently and accurately exploit these
capabilities across tasks remains an open chal-
lenge (Liu et al., 2024). Two mainstream ap-
proaches have emerged:Retrieval-Augmented
Generation (RAG), which retrieves relevant snip-
pets from external sources (Lewis et al., 2020),
andLong Context (LC)processing, where the full
context is provided to the model for end-to-end
reasoning (Kamradt, 2023; Hsieh et al., 2024).
These two pathways differ in suitability, each
with distinct strengths and limitations. Bench-
marks (Li et al., 2025; Bai et al., 2024a; An
et al., 2024) show that LC excels at comparison
and complex reasoning by leveraging a global
view, while RAG is preferred for fact retrieval and
hallucination-sensitive tasks, offering precise ev-
idence and abstention. LC works well for well-
structured texts (e.g., financial reports) but suffers
from positional effects such as lost-in-the-middle
problems (Liu et al., 2024). Conversely, RAG
adapts better to loosely structured texts (e.g., nov-
els) and is less position-sensitive, though its effec-
tiveness depends on retrieval quality. Thus, choos-
ing between RAG and LC is a key challenge for
advancing long-context processing.
Existing strategies such as Self-Route (Li et al.,
2024c) adopt a failure-driven mechanism: attempt
RAG first, and switch to LC only if the model out-
puts "unanswerable". While simple and sometimes
effective, this approach has some drawbacks: it
is passive, relying solely on retrieval failure sig-
nals; it incurs redundant overhead, since RAG re-
trievals must always be executed; it depends on
the model’s self-assessment, which may be either
over-conservative or overconfident; and it offers lit-
tle interpretability regarding routing decisions (see
Appendix A.6). To overcome these issues, we in-
troducePre-Route, a proactive routing framework.
Our core insight is that LLMs possess a latent “rout-
1arXiv:2605.10235v2  [cs.CL]  12 May 2026

0% 20% 40% 60% 80% 100%
Cost (LC selection rate, % of queries)3.333.363.393.423.453.48Performance (QA Score)Always-RAG
Always-LC
Pre-Route (Large)
Pre-Route (Distilled)
Trade-off (Pareto) Frontier(a) LaRA (in-domain)
0% 20% 40% 60% 80% 100%
Cost (LC selection rate, % of queries)0.420.430.440.450.460.47Performance (QA Score)Always-RAG
Always-LC
Pre-Route (Large)
Pre-Route (Distilled on LaRA)
Trade-off (Pareto) Frontier (b) LongBench-v2 (OOD)
Figure 1: Visualization of Pre-Route cost-effectiveness
with Qwen3-235B-A22B: (a) in-domain (LaRA), (b)
out-of-domain (LongBench-v2).
ing potential” dormant in the high-dimensional pa-
rameter space. Pre-Route elicits this ability not by
teaching new knowledge, but by constraining the
decision space through a structured reasoning pro-
cess, aligning internal attention with critical deci-
sion boundaries. Specifically, Pre-Route performs
structured reasoningbeforeanswering using low-
cost metadata such as document type, title, length,
and initial snippets. Furthermore, this process can
be distilled into smaller models for lightweight de-
ployment.
In summary, our contributions are:
1.We propose Pre-Route, an active and cost-
efficient routing framework that introduces
structured reasoning before retrieval and an-
swering, implementing a "plan-then-execute"
paradigm.
2.We systematically verify and uncover the la-
tent routing capabilities of LLMs. Through
BoN experiments and linear probe analysis,
we demonstrate the existence of this ability
and find that guidelines can effectively elicit
and stabilize it, enabling single-sample per-
formance to approach the multi-sample upper
bound and forming a more linearly separable
“optimal route dimension” in the representa-
tion space.
3.We successfully transfer this routing capabil-
ity to smaller models, forging a lightweight,
plug-and-play module. We find that while
smaller models struggle to acquire this ability
through prompting alone, they can effectively
learn the Pre-Route reasoning patterns via dis-
tillation. This paves the way for efficient, low-
cost deployment on edge and client-side de-
vices.
4.We achieve significant advantages in both
in-distribution and out-of-distribution (OOD)
tasks. On the LaRA Benchmark andLongBench-v2 datasets, Pre-Route demon-
strates superior effectiveness and generaliza-
tion by outperforming Self-Route in both per-
formance and efficiency.
2 From Dormant to Active: Unlocking
Latent Routing Potential
Existing methods like Self-Route (Li et al., 2024c)
treat routing as a retrieval fallback—RAG is always
attempted first—implicitly assuming LLMs lack in-
trinsic planning. In this work, we challenge this
assumption and ask a more fundamental question:
Do LLMs already possess a "latent routing abil-
ity"—the internal competence to discern task
requirements and select an appropriate path?
To address this question, we adopt two comple-
mentary perspectives: (i)behavioral experiments
that demonstrate both the existence of such latent
ability and its instability, and (ii)representation
analysisthat probes the model’s internal mecha-
nisms to reveal how this ability can be elicited.
Taken together, our argument is that routing ability
is not absent but rather dormant, and can be reliably
elicited by suitable guidelines. The routing guide-
lines were derived from prior studies and finalized
on the LaRA training set; the test set remained
unseen during prompt construction. All analyses
below are post-hoc verifications conducted after
prompts and models were frozen. Experimental
setup details are in Sec. 4.1.
2.1 Behavioral Evidence: Revealing Latent
Ability via Best-of-N Sampling
To disentangle whether routing failures arise from
a lack of capacity or merely insufficient activation,
we recast routing as a binary classification task
(predicting RAG vs. LC). We adopt theBest-of-
N (BoN) samplingmethodology widely used for
testing latent abilities.
We design three controlled comparisons under
identical metadata: (1) Prompt Paradigm:answer
directly,unconstrained CoT(free-form reasoning
before routing), and our proposedPre-Route(struc-
tured reasoning guidelines). (2) Sampling Strategy:
N∈ {1,2,4,8} . (3) Metrics: QA Score, LC selec-
tion rate and routing accuracy. We reportrouting
accuracy, defined as the proportion of model deci-
sions that match anideal label—the decision rule
that maximizes QA performance while defaulting
to the lower-cost RAG option when performances
are comparable (formalized later in Sec. 3.4).
Experimental results (Fig. 2) clearly reveal the
model’s internal decision capacity. UnderAnswer
2

1 2 4 8
Best-of-/glyph1197 Sampling (/glyph1197)3.23.33.43.53.63.7QA Score (0 4)
(c) Final QA Score (Effectiveness)
Answer Directly
Unconstrained CoT
Pre-route (guidelines)
1 2 4 8
Best-of-/glyph1197 Sampling (/glyph1197)0.20.30.40.50.6LC Selection Rate
(b) LC Selection Rate (Cost)
Answer Directly
Unconstrained CoT
Pre-route (guidelines)
1 2 4 8
Best-of-/glyph1197 Sampling (/glyph1197)0.40.50.60.70.80.9Accuracy
(a) Routing Accuracy
Answer Directly
Unconstrained CoT
Pre-route (guidelines)Figure 2: BoN comparison results tested with answer model Qwen3-235B-A22B.
Table 1: Linear probe accuracy tested with answer
model Qwen3-235B-A22B.
Model Prompt DistilledAccuracy
Ideal Route Doc Task
Qwen3-1.7B Pre-Route✓ 0.6389 0.7986 0.39580.4097
Qwen3-1.7B Pre-Route✗ 0.6250 0.7639 0.33300.4167
Qwen3-8B answer direct.✗ 0.5486 0.6597 0.2986 0.2569
Qwen3-1.7B answer direct.✗ 0.5208 0.6389 0.3194 0.2500
Qwen3-8B unconstr. CoT✗ 0.4653 0.6944 0.3681 0.2500
Qwen3-1.7B unconstr. CoT✗ 0.3958 0.5764 0.3472 0.2639
DirectlyandUnconstrained CoT, routing accuracy
shows a strong positive correlation with N. For
instance, in the Answer Directly setting, accuracy
rises from about 0.53 (N=1) to 0.87 (N=8), indicat-
ing that the knowledge required for correct routing
is present but accessed stochastically: sometimes
the model follows the correct path, other times it
diverges. Increasing the number of samples simply
raises the chance of hitting the right path, thereby
exposing the model’s latent ability.
In sharp contrast, introducing thePre-Route
(guidelines)structured reasoning chain fundamen-
tally changes the picture. Its curve is much
flatter: accuracy already reaches 0.70 at N=1—
substantially higher than the other two settings—
and further climbs to 0.83 at N=4 before saturating
at 0.87 by N=8. This pattern indicates thatstruc-
tured guidance acts as a calibrator and stabi-
lizer. It does not inject new knowledge, but rather
provides a clear reasoning scaffold that reliably
elicits and directs the model’s latent routing capa-
bility, enabling single-shot performance close to
the model’s upper bound.
2.2 Representation Evidence: Probing the
Decision Space
While behavioral results confirm that routing ability
can be elicited, an open question is what changes
occur internally. We hypothesize that structured
prompting not only alters outputs, but also reorga-
nizes the representation space, such that routing
signals become more linearly separable. To in-
vestigate this, we apply linear probes on frozen
representations and examine four prediction tar-
gets: (i) theidealdecision, defined in Sec. 3.4 as
thetask-optimal decision—the choice that maxi-
mizes QA performance while defaulting to RAG
when tied; (ii) the model’s ownroutechoice, re-flecting its internal bias; (iii)doc_type(7 classes),
to test whether routing simply mirrors document
categories; and (iv)task_type(4 classes), to test
whether routing is merely driven by shallow task-
level hints rather than deeper reasoning signals.
Among these, ideal label accuracy is the most im-
portant indicator, since it directly evaluates whether
the model’s latent routing aligns with task-optimal
decisions.
We applylinear probing(Alain and Bengio,
2017): a linear classifier trained on frozen penul-
timate last token embeddings to predict the labels
above. Linear probes are deliberately chosen since
linear separability implies explicit encoding in the
representation. 1)Prompt structure is critical:on
Qwen3-1.7B, ideal-label accuracy improves from
0.396 (unconstr. CoT) to 0.521 (direct) to0.625
(Pre-Route). 2)Distillation further aligns:adding
distillation(see Sec. 3.4) on Pre-Route raises ac-
curacy to 0.639. 3)Design over scale:Qwen3-
8B with direct prompting (0.549) underperforms
Qwen3-1.7B with Pre-Route (0.625). 4)Model’s
own routing decision:under structured guidance,
the model’s representation of its own decisions be-
comes clearer, making probes better at predicting
itsroutechoice (0.58–0.80), though not always
aligned with the task-optimal choice. 5)Not shal-
low heuristics:low accuracy ondoc_typeand
task_typeshows routing is not reducible to trivial
document or task categories. Together, these find-
ings confirm that optimal routing boundaries are al-
ready encoded in the representation space. BoN ex-
periments demonstrate it behaviorally, while prob-
ing confirms it representationally. Crucially, struc-
tured guidance and light supervision reliably elicit
and stabilize this ability—laying the foundation for
the Pre-Route framework introduced next.
3 Methodology
3.1 Problem Formalization and Theoretical
Framework
Decision-theoretic formalization.Given a query
qand long document D, a routing policy πse-
lectsy∈ {RAG, LC} solely from low-cost meta-
information m(title, type, length estimate, leading
snippet, etc.), leveraging the observation that m
3

Step3 · Distilling with 
Rejection SamplingStep1 · Build Prompt with Guidelines
IDEAL LABEL
• Prefer higher QA score
• If scores are equal →prefer RAG (efficiency)
Compare <decision> vs IDEAL LABEL
: keep · 
 : drop
: (Inputs, Guidelines) → (Reasoning, Decision)
where <decision> ==IDEAL LABEL
Accepted Rejected
High -quality 
Training Set 
Pre-Route 
(Distilled)
Qwen3 -1.7BSFT
-query :“Why does Stella repeatedly think thattheir father will'kill'Miles?”
-task_type :reasoning
-document_title :“The Catcher intheRye”
-document_type :novel_fake
-document_length :95ktokens
-answering_max_window :128000 tokens
-document_fits_window :Yes
-answering_model :qwen 3-235b-a22b-instruct -2507
-document_head :“Ifyoureally want tohear about it...”
<step 1>:Identify thequestion &doctype...how does thisaffect context need?
<step 2>:Assess ifinfo islocal orscattered ...how does thisaffect strategy?
<step 3>:Evaluate ifthedocfitswindow ...ifnot,how does thatchange choice?
<step 4>:Ifquery canbesolved bykeywords ...orneeds implicit synthesis ?
<step 5>:Reflect onthemodel used...window size &capacity ,how does it
influence?
<step 6_efficiency> :Compare efficiency ofRAG vsLC...<reflection> :
Summarize above steps ...more suitable? <decision> :State final choice
clearly →RAG orLC
-Ifquality equal/uncertain →prefer RAG (efficiency)
-Prefer LConly iffitsAND needs global synthesisStep2 · Generate Response 
with Reasoning 
<step 1>Reasoning query ona'novel_fake' .Requires
understanding acustom narrative ,not known facts .
</step 1>
<step 2> Answer to 'why repeatedly' is
likely scattered across thetext, needing cross -sectional
synthesis .</step 2>
<step 3>Document (95ktokens) fitswithin themodel's
window (128k).LONG_CONTEXT isfeasible .</step 3>
<step 4>Requires synthesizing implicit logic ,notkeyword
matching .RAG would likely miss narrative buildup .
</step 4>
<step 5>The model (qwen 3-235b)ishigh -capacity
andcapable ofeffective long-context reasoning .</step 5>
<step 6_efficiency> RAG is more efficient, but
theexpected quality drop issignificant .The trade -off
favors quality .</step 6_efficiency>
<reflection> Justification :Global synthesis isrequired ,
thedocument fitsthewindow, and themodel iscapable .
Therefore, LONG_CONTEXT is the superior
choice .</reflection>
<decision> LONG_CONTEXT </decision>
Guidelines Input
 Decision Reasoning
Larger LLMs
Figure 3: Overview of the Pre-Route framework. Step 1 builds structured prompts with meta-infos and guidelines;
Step 2 generates reasoning-based routing decisions; Step 3 performs rejection sampling to filter out suboptimal
decisions, distilling high-quality training data for efficient and interpretable routing with smaller models (Qwen3-
1.7B). (Complete prompt is provided in Fig. 10)
often encodes sufficient cues about information
distribution and task difficulty. Formally, we let
U(y;q,D) denote the expected QA performance of
decision y, and C(y;q,D) denote the correspond-
ing execution cost.
3.2 The Pre-Route Framework
Core hypothesis and idea.LLMs already en-
code task understanding and information-need sens-
ing, yet standard pipelines do not explicitlyacti-
vatethis capacity. Pre-Route triggers the latent
planning ability by performingstructured rea-
soning with decision labelsbeforeanswering:
(i) apre-decisionis made prior to costly full-
document processing; (ii) the decision relies on
meta-information only, avoiding full-document
retrieval overhead; (iii) astructured rationalepro-
vides interpretability. In practice, Pre-Route uses
only low-cost m: user query, task type, document
title/type, document length, answering model, a
short document head, and RAG configuration; no
retriever or answering model is called, markedly re-
ducing planning overhead Croute(see cost analysis
below).
Structured reasoning-chain design.Routing is
conceptually grounded in four broaddecision di-
mensions—(i) task & document characterization,
i.e., how task or document types affect the need
for retrieval vs. long-context; (ii) information dis-
tribution, i.e., whether relevant content is missing,
fragmented, or dispersed across sections; (iii) capa-
bility–constraint matching, i.e., whether retrieval
suffices or long-context is required to avoid po-
sitional or window-size limits; and (iv) decision
with explanation, i.e., outputting the route together
with rationale and fallback considerations. In prac-
tice, these dimensions are instantiated through sixsteps(Fig. 3):(1) task & document character-
ization;(2) distribution pattern judgment;(3)
context-window feasibility;(4) retrieval feasibil-
ity;(5) model capability consideration; and(6)
efficiency trade-off. This layered design aligns
with prior empirical findings (Xu et al., 2024; Li
et al., 2024c, 2025) through a structured pipeline,
highlighting operational and transparent decision
process while yielding interpretability, efficiency,
and robust generalization.
3.3 Cost Analysis
We decompose cost into a routing part and an an-
swering part:
C(y;q,D) =C route(m) +C answer(y;q,D),(1)
where Croute(m) is the cost of making route de-
cision, and Canswer(y;q,D) corresponds to down-
stream execution (retrieval for RAG vs. long-
context processing for LC). Using this decomposi-
tion, the expected total cost of a routing policy with
LC-selection probability p(LC) = Pr(y=LC) is
C=C route+p(LC)·C LC+ 
1−p(LC)
·CRAG,
(2)
where CLCandCRAGdenote the average answering
costs under LC and RAG respectively. In what
follows we analyze CrouteandCanswer separately
and then combine the two perspectives.
3.3.1 Analysis ofC route: Self-Route vs.
Pre-Route
Pre-Route achieves lower routing cost than the Self-
Route baseline once distillation is applied: with
Qwen3-1.7B the cost is reduced to about one-fifth
that of Self-Route (Tab. 2). With Qwen3-235B, the
cost is slightly higher due to the generated rationale,
but this overhead is offset once a smaller router is
4

Table 2: Per-decision routing cost in USD (average
query, computed over all queries in the LaRA bench-
mark).
Method Model Input Toks Output Toks Cost (×10−3USD)
Input OutputTotal
Self-Route Qwen3-235B 2600 27 0.73 0.03 0.76
Pre-Route Qwen3-235B 1205 648 0.34 0.73 1.07
Pre-Route Qwen3-1.7B 1205 670 0.05 0.110.16
used. This demonstrates that planning overhead
can be kept low while retaining interpretability.
3.3.2 Analysis ofC answer : LC vs. RAG
Answering cost is the dominant component, and
it is largely governed by the LC selection proba-
bility p(LC) . A long-context (LC) pass processes
the entire document, whereas retrieval-augmented
generation (RAG) operates only on retrieved pas-
sages. Since CLCis substantially larger than CRAG,
even moderate changes in p(LC) have a significant
impact on total cost. Hence, controlling p(LC) is
central to cost management.
3.3.3 Overall cost comparison
When both components are combined, routing over-
head Croute remains small relative to answering
cost. Even with a large router (235B), planning
cost is below 4% of a single 100k-token LC pass
($0.028); with a small router (1.7B), the propor-
tion falls below 1%. These comparisons indicate
that the total cost is primarily determined by the
answering term, and in particular by the LC se-
lection probability p(LC) . Critically, Pre-Route
providesstructuralsavings over post-retrieval rout-
ing (e.g., Self-Route): since the routing decision
is madebeforedocument processing, the retrieval
overhead (embeddings, vector DB queries, rerank-
ing) is incurred only when RAG is selected, rather
than unconditionally for every query.
3.4 Learning Paradigm: Selective Strategy
Alignment via Distillation
Motivation and cost optimization.The over-
all cost is primarily determined by the answering
component, and in particular by the LC selection
probability p(LC) . Yet Routing overhead Croute
still matters for latency and scalability in practi-
cal deployment. Large models (e.g., Qwen3-235B)
can produce strong routing in zero-shot settings,
yet the planning overhead Crouteis non-trivial. We
distill the planning ability into smaller models (e.g.,
Qwen-7B/1.7B) to reduce cost and latency, im-
prove deployability in constrained environments,
and specialize the student for routing.Learning targets.For fixed meta-information m,
the objective is to maximize expected QA perfor-
mance by choosing the option with higher perfor-
mance. When utilities are comparable or the dif-
ference is marginal (e.g., tied scores in multi-level
or binary evaluation), the lower-cost RAG is pre-
ferred, since CRAG≪C LC. This principle leads
directly to the definition of theideal labelfor each
sample:
ˆyideal=(
LC, U(LC;q,D)> U(RAG;q,D),
RAG,otherwise.
(3)
Intuitively, LC is chosen only when its perfor-
mance advantage over RAG; when utilities are com-
parable, the lower-cost RAG is preferred. Improv-
ing routing accuracy reduces expected performance
loss by avoiding wasteful LC on unhelpful samples
and reallocating budget to cases that truly bene-
fit from LC. We formally prove in Appendix A.2
that optimizing the ideal label ensures minimal ex-
pected cost among all strategies consistent with
performance-optimal decisions.
Stage 1: Rejection sampling.Let the teacher
πTproduce candidates Si= (T i, yi); keep entries
where the teacher decision matches the ideal rule,
Dfiltered ={(m i, Ti, yi)|yi= ˆy ideal,i},(4)
where mi,Ti,yi, and ˆyideal,i denote the metadata,
reasoning trace, decision, and ideal decision for
example i, thus ensuring data quality, correcting
the empirical distribution toward a near-optimal
support, removing teacher errors, and enforcing
consistency with task-optimal supervision.
Stage 2: Path SFT (student fitting on successful
support).Train the student on Dfiltered by mini-
mizing
LSFT(θS) =−E (m,T,y)∼D filtered
logπ S(T, y|m)
,
(5)
which is equivalent to minimizing KL(q∗∥πS)
where q∗is the teacher’ssuccessfulsub-distribution.
Unlike answer distillation, this transfershow to rea-
sonrather than merelywhat to answer.
Data construction strategy.We generate reason-
ing chains with Qwen3-235B-A22B and DeepSeek-
R1; assignˆy idealconsistently; enhance scenario di-
versity by mixing answering-model scales; perform
a global 70/10/20 train/val/test split stratified by
(type, level, context length); then merge and retain
only training entries where the teacher is consistent
with the ideal label.
5

Table 3: LaRA (In-distribution main experiment): Pre-Route vs. Self-Route and fixed baselines; Best and Second
per answer model.Pre-Route Variants:Large: R1 (DeepSeek-R1) or Q235B (Qwen3-235B); Small: Q1.7B
(Qwen3-1.7B); Distilled: D-Q1.7B (Distilled-Qwen3-1.7B).
Red-shaded entries denote our recommendedPre-Route(Large)andPre-Route(Distilled).
Answer Model
Router ModelQwen3-1.7B [N] Qwen3-1.7B [T] Qwen3-4B [N] Qwen3-4B [T] Qwen3-8B [N] Qwen3-8B [T]
QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑
Always-LC (Baseline) 2.13 100.0 0.18 2.29 100.0 0.18 2.89 100.0 0.31 3.14 100.0 0.36 2.98 100.0 0.31 3.23 100.0 0.35
Always-RAG (Baseline) 2.70 0.0 0.82 2.89 0.0 0.82 3.08 0.0 0.69 3.17 0.0 0.64 3.13 0.0 0.69 3.22 0.0 0.65
Self-Route (Baseline) 2.22 33.6 0.49 2.40 31.7 0.48 2.81 24.6 0.53 3.06 30.5 0.47 3.04 28.1 0.55 3.23 30.7 0.57
Pre-Route (R1) [T] 2.70 2.7 0.83 2.88 1.8 0.83 3.09 5.1 0.73 3.20 4.8 0.70 3.15 5.0 0.76 3.23 8.0 0.70
Pre-Route (Q235B) [N] 2.71 4.7 0.83 2.90 4.8 0.82 3.13 21.1 0.73 3.26 26.6 0.70 3.16 22.2 0.73 3.30 26.2 0.70
Pre-Route (Q235B) [T] 2.71 2.3 0.84 2.91 3.0 0.84 3.11 13.1 0.73 3.22 15.7 0.70 3.17 15.9 0.74 3.27 17.9 0.71
Pre-Route (Q1.7B) [N] 2.70 7.9 0.79 2.85 10.8 0.77 3.04 30.1 0.61 3.20 35.9 0.57 3.12 34.0 0.60 3.29 33.2 0.63
Pre-Route (Q1.7B) [T] 2.68 10.8 0.77 2.87 11.1 0.80 3.07 30.8 0.62 3.22 40.1 0.58 3.10 35.6 0.61 3.29 39.9 0.58
Pre-Route (D-Q1.7B) [N] 2.70 3.6 0.82 2.89 3.9 0.83 3.10 17.9 0.71 3.26 21.2 0.71 3.16 21.5 0.73 3.30 21.4 0.71
Pre-Route (D-Q1.7B) [T] 2.71 3.2 0.83 2.89 2.8 0.82 3.10 21.1 0.72 3.26 20.5 0.70 3.19 19.5 0.76 3.28 20.8 0.70
Answer Model
Router ModelQwen3-30B [N] Qwen3-30B [T] Qwen3-235B [N] Qwen3-235B [T] DeepSeek-R1 [T] Qwen-Max [N]
QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑
Always-LC (Baseline) 3.37 100.0 0.40 3.39 100.0 0.39 3.46 100.0 0.34 3.51 100.0 0.40 3.44 100.0 0.35 3.36 100.0 0.39
Always-RAG (Baseline) 3.18 0.0 0.60 3.27 0.0 0.61 3.32 0.0 0.66 3.33 0.0 0.60 3.38 0.0 0.65 3.20 0.0 0.61
Self-Route (Baseline) 3.20 33.9 0.53 3.10 35.5 0.42 3.39 41.1 0.56 3.34 33.9 0.52 3.36 31.4 0.52 3.28 36.5 0.56
Pre-Route (R1) [T] 3.26 14.9 0.69 3.30 9.6 0.69 3.39 14.2 0.73 3.37 10.8 0.68 3.42 10.0 0.70 3.24 10.6 0.67
Pre-Route (Q235B) [N] 3.31 33.8 0.65 3.39 29.4 0.68 3.47 29.1 0.72 3.43 27.2 0.67 3.47 27.2 0.70 3.31 23.1 0.69
Pre-Route (Q235B) [T] 3.27 17.9 0.68 3.32 18.6 0.69 3.42 20.2 0.72 3.40 18.3 0.68 3.45 16.9 0.71 3.29 19.6 0.69
Pre-Route (Q1.7B) [N] 3.26 42.3 0.53 3.37 37.2 0.57 3.36 40.9 0.53 3.41 40.6 0.54 3.44 32.5 0.56 3.25 36.6 0.55
Pre-Route (Q1.7B) [T] 3.27 40.0 0.61 3.36 38.5 0.59 3.41 35.6 0.59 3.41 33.1 0.59 3.49 31.5 0.68 3.26 28.3 0.60
Pre-Route (D-Q1.7B) [N] 3.28 27.7 0.66 3.39 28.0 0.70 3.46 24.6 0.74 3.43 22.7 0.69 3.51 20.6 0.73 3.28 24.7 0.67
Pre-Route (D-Q1.7B) [T] 3.29 26.4 0.65 3.35 24.3 0.67 3.47 26.9 0.73 3.44 26.0 0.67 3.47 20.3 0.72 3.31 24.5 0.69
4 Experiments
4.1 Experimental Setup
Datasets and BenchmarksWe evaluate Pre-
Route on both in-domain and out-of-domain set-
tings. The in-domain dataset isLaRA, used
for training and primary evaluation. To assess
generalization, we further evaluate on the out-of-
domain benchmarkLongBench-v2. ForLaRA,
since knowledge distillation is involved during
data construction (see Sec. 3.4), we adopt strati-
fied random splits bydocument type,task level,
andcontext lengthfor all experiments to en-
sure fair comparison and prevent leakage. For
LongBench-v2, we report results on the full eval-
uation set. Detailed statistics and examples are
provided in Appendix A.3. For retrieval, we
follow the default LaRA configuration, using a
chunk size of 600 with 100-token overlap, the
gte-multilingual-base embedding model
andgte-multilingual-reranker-base
reranker (Zhang et al., 2024a), with vector–rerank
split ratio 0.5 and rerank size 5.
Evaluation MetricsWe report three metrics: (i)
Route Accuracy— the fraction of routing deci-
sions that match theideal label(formalized in
Sec. 3.4). The ideal label selects the decision
y∈ {RAG, LC} that maximizes QA performance
U; in ties, it defaults to RAG to prefer lower
cost. Thus, higher Route Accuracy reflects better
effectiveness–efficiency alignment. (ii)QA Score
— downstream answer quality. Because the origi-
nal LaRA binary metric is too coarse to distinguish
superficial vs. substantively complete answers, we
adopt a 4-point rubric for finer resolution (exam-
ples in Appendix A.5). (iii)LC Selection Rate—the proportion of queries routed to the costly LC
path, used as a proxy for computational cost.
BaselinesWe compare against: (a) Always-
RAG and Always-LC (fixed strategies); (b) Self-
Route (Li et al., 2024c) using the original prompt
(Appendix A.12); and (c) Pre-Route in three
variants that we report separately throughout:
Pre-Route (Large: DeepSeek-R1, Qwen3-235B,
prompt-only), Pre-Route (Small: Qwen3-1.7B,
prompt-only), and Pre-Route (Distilled: Distilled-
Qwen3-1.7B). All Pre-Route prompts are fixed
across models. For all applicable methods we re-
port both thinking[T]and no-thinking[N]modes.
Always-RAG is excluded from LC Rate ranking
(LC=0%), while Always-LC is excluded from QA
ranking due to prohibitive computational costs.
ModelsOur experiments employ the latest avail-
able models from the recent Qwen3 series, span-
ning the 235B, 30B, 8B, 4B, and 1.7B scales.
Qwen3-235B and Qwen3-30B are from the July
2025 version. Our fine-tuned student model is re-
ferred to as Distilled-Qwen3-1.7B. We also include
DeepSeek-R1 (May 2025 version) and the propri-
etary model Qwen-Max (January 2025 version).
4.2 Main Results on LaRA Benchmark
Tab. 3 summarizes the LaRA in-domain results.
Note thattop-two results (bold/underline) consis-
tently fall in our red shaded Pre-Route meth-
ods.Overall, Pre-Route consistently surpasses
Self-Route in QA score and route accuracy across
all settings, while also reducing the reliance on
LC, proving its robustness across various answer
model sizes and thinking modes. All reported im-
provements are statistically significant (paired t-
6

test,p <0.01 , Cohen’s d: 0.19–0.26). We also
provide more case studies on why sometimes Self-
Route fails on LaRA in Appendix A.6.
Large model prompt-only routing.Using
Qwen3-235B and DeepSeek-R1, we find that
Pre-Route (Prompt-only) significantly outperforms
Self-Route across answer models. Structured rea-
soning prompts alone can reliably elicit latent rout-
ing ability, yielding both higher route accuracy
and QA score. This suggests that LLMs implic-
itly learn the comparative strengths of RAG and
LC during pretraining—e.g., when information is
concentrated vs. scattered—and structured prompts
elicit this knowledge proactively rather than relying
on post-hoc failure signals.
Transferability to small models.When directly
applied to Qwen3-1.7B with identical prompts, per-
formance degrades: reasoning chains and routing
decisions become unstable. For example, under
the Qwen3-235B [T] answer model, Qwen3-1.7B
routing lags far behind the large-model routers in
accuracy despite comparable QA scores, highlight-
ing that small routers cannot directly inherit com-
plex planning behavior. Notably, small models’
errors are systematically tilted toward the “safer”
LC option (74.3% of errors), suggesting acorrect
but unstableprior that distillation stabilizes rather
than creates.
Distillation on in-domain data.Using Pre-
Route (Large) as teacher, we distill into a
lightweight Qwen3-1.7B student with reasoning-
chain supervision on LaRA. The distilled model
successfully acquires structured routing logic, ap-
proaching the teacher’s QA performance while
achieving a lower LC selection rate. For instance,
with the strongest answer model Qwen-Max [N],
the distilled router reaches clearly higher accuracy
at substantially reduced LC usage compared to Self-
Route. These results verify that large-model rout-
ing logic can be effectively compressed and trans-
ferred. Notably, the non-thinking distilled router
already performs well, making it the most resource-
efficient configuration. We additionally compare
against traditional ML classifiers in Appendix A.7,
confirming Pre-Route’s gains extend beyond meta-
data signals. A fine-grained breakdown by docu-
ment type and difficulty (Appendix A.8) shows
consistent gains across all categories, with the
largest improvement on hallucination tasks (Acc
0.22→0.75).4.3 Generalization on Out-of-Distribution
Data
Out-of-distribution evaluation.Table 4 reports
representative LongBench-v2 results, compared to
LaRA, LongBench-v2 is out-of-domain not only in
data distribution but also in task format and eval-
uation protocol: LaRA uses open-ended QA with
graded scores (0–4), whereas LongBench-v2 re-
formulates tasks as four-choice multiple-choice
questions(MCQ) with binary correctness. This
makes Self-Route appear more competitive in QA
scores, since its weakness of sometimes routing
partially correct answers (as in Appendix A.6) is
no longer penalized—under binary MCQ evalua-
tion the router only needs to ensure correctness
rather than differentiate answer quality. Neverthe-
less, Pre-Route remains more cost-efficient than
Self-Route with substantially low LC usage, ef-
fectively offloading “easy” samples to maximize
compute efficiency. Note that LC Rate reduction is
significant (p <0.01).
5 7 10
Rerank Size60708090Count72
61
56555864Only LC (count)
Only RAG (count)
Only LC (%)
Only RAG (%)
8101214161820
Proportion (%)
16.7%
13.9%
12.9% 12.8%13.2%14.8%
Figure 4: Qwen3-235B: Number of questions answered
correctlyonlyby LC vs.onlyby RAG.
Robustness to retrieval configuration.Fig. 4
illustrates the LC vs. RAG balance shift as rerank
size increases. Table 4 (rerank size 7) and Ap-
pendix A.4 (sizes 5, 10) demonstrate Pre-Route’s
effectiveness across models. Pre-Route (Distilled)
remains stable across all configurations, showing
minimal sensitivity to retriever changes and con-
sistently achieving higher accuracy with lower LC
usage.
4.4 Pre-Route Guidelines Ablation
To understand why the structured reasoning chain
is effective, we conduct ablations by removing in-
dividual steps. As shown in Table 5, thefull guide-
lineachieves the best balance: highest accuracy
with a moderate LC rate. Removing components
leads either to QA Score drops or inflated LC usage,
both stemming from incorrect route choices.
Decision rules:Dropping them causes the router
to over-rely on LC, greatly increasing cost.Re-
flection:Removal slightly lowers accuracy, sug-
gesting it helps refine borderline cases.Step 1/2
(Task & Distribution):Without them, the router
7

Table 4: LongBench-v2 (OOD, rerank size = 7): Pre-Route results;Bestand Second per answer model.Pre-Route
Variants:Large: R1 (DeepSeek-R1) or Q235B (Qwen3-235B); Small: Q1.7B (Qwen3-1.7B); Distilled: D-Q1.7B
(Distilled-Qwen3-1.7B).
Red-shaded entries denote our recommendedPre-Route(Large)andPre-Route(Distilled).
Answer Model
Router ModelQwen3-1.7B [N] Qwen3-1.7B [T] Qwen3-4B [N] Qwen3-4B [T] Qwen3-8B [N] Qwen3-8B [T]
QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑
Always-LC (Baseline) 0.29 100.0 0.33 0.28 100.0 0.31 0.30 100.0 0.28 0.36 100.0 0.34 0.40 100.0 0.33 0.40 100.0 0.34
Always-RAG (Baseline) 0.34 0.0 0.67 0.34 0.0 0.69 0.33 0.0 0.72 0.38 0.0 0.66 0.36 0.0 0.67 0.39 0.0 0.66
Self-Route (Baseline) 0.3431.1 0.68 0.33 26.0 0.69 0.32 22.0 0.72 0.37 30.6 0.64 0.37 24.8 0.68 0.39 28.6 0.65
Pre-Route (R1) [T] 0.32 24.8 0.71 0.33 22.5 0.72 0.34 24.4 0.69 0.41 25.9 0.69 0.38 19.5 0.71 0.42 22.7 0.71
Pre-Route (Q235B) [N] 0.32 38.1 0.61 0.34 39.1 0.61 0.33 39.4 0.57 0.39 39.6 0.59 0.38 32.2 0.63 0.41 31.3 0.64
Pre-Route (Q235B) [T] 0.32 29.7 0.68 0.33 29.4 0.69 0.34 27.6 0.67 0.39 31.0 0.65 0.38 26.2 0.66 0.41 24.0 0.70
Pre-Route (Q1.7B) [N] 0.3437.7 0.62 0.30 36.4 0.60 0.32 39.4 0.58 0.36 36.8 0.57 0.38 40.1 0.57 0.39 41.3 0.57
Pre-Route (Q1.7B) [T] 0.33 42.2 0.56 0.33 43.9 0.56 0.33 40.5 0.57 0.37 40.0 0.57 0.3945.9 0.55 0.41 43.1 0.55
Pre-Route (D-Q1.7B) [N] 0.34 6.8 0.84 0.33 7.7 0.81 0.34 23.9 0.69 0.37 22.9 0.68 0.37 19.1 0.72 0.40 20.4 0.70
Pre-Route (D-Q1.7B) [T] 0.33 7.0 0.83 0.33 7.7 0.81 0.33 20.3 0.69 0.38 19.7 0.71 0.36 20.4 0.69 0.40 21.5 0.69
Answer Model
Router ModelQwen3-30B [N] Qwen3-30B [T] Qwen3-235B [N] Qwen3-235B [T] DeepSeek-R1 [T] Qwen-Max [N]
QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑
Always-LC (Baseline) 0.40 100.0 0.34 0.36 100.0 0.51 0.47 100.0 0.54 0.52 100.0 0.52 0.53 100.0 0.43 0.48 100.0 0.52
Always-RAG (Baseline) 0.42 0.0 0.66 0.24 0.0 0.49 0.46 0.0 0.46 0.45 0.0 0.48 0.49 0.0 0.57 0.44 0.0 0.48
Self-Route (Baseline) 0.42 29.5 0.66 0.28 40.6 0.54 0.46 57.2 0.46 0.5046.6 0.55 0.54 35.4 0.62 0.46 49.7 0.51
Pre-Route (R1) [T] 0.40 25.2 0.67 0.25 25.5 0.62 0.49 25.3 0.72 0.48 24.0 0.65 0.53 26.2 0.70 0.46 25.8 0.69
Pre-Route (Q235B) [N] 0.40 36.8 0.58 0.28 35.1 0.60 0.49 39.0 0.62 0.50 38.1 0.59 0.55 40.3 0.62 0.47 39.1 0.61
Pre-Route (Q235B) [T] 0.41 29.2 0.66 0.26 28.8 0.60 0.49 29.2 0.68 0.48 28.3 0.63 0.54 36.4 0.62 0.47 29.8 0.67
Pre-Route (Q1.7B) [N] 0.40 48.1 0.51 0.29 42.2 0.53 0.47 36.0 0.61 0.47 35.7 0.57 0.53 41.3 0.56 0.46 42.1 0.56
Pre-Route (Q1.7B) [T] 0.40 44.8 0.52 0.3043.9 0.56 0.46 34.4 0.60 0.5040.5 0.57 0.53 40.8 0.59 0.45 42.3 0.54
Pre-Route (D-Q1.7B) [N] 0.43 26.9 0.67 0.27 28.5 0.62 0.47 26.4 0.67 0.50 28.8 0.61 0.52 18.9 0.67 0.46 25.2 0.69
Pre-Route (D-Q1.7B) [T] 0.42 28.3 0.65 0.28 27.4 0.64 0.46 28.0 0.66 0.48 26.9 0.62 0.52 17.5 0.69 0.44 24.4 0.68
Table 5: Ablation analysis. Removing reasoning steps
reduces accuracy or increases LC rate, highlighting the
necessity of the full Pre-Route design.
Variant QA Score LC Rate (%) Acc
Pre-Route (full) 3.3820.70.68
- No Reflection 3.33 20.8 0.65
- No Decision Rules3.3845.3 0.57
- No Step1 3.3310.1 0.68
- No Step2 3.37 27.0 0.66
- No Step3 3.35 18.2 0.66
- No Step4 3.31 17.2 0.66
- No Step5 3.31 21.2 0.66
- No Step6 3.35 24.3 0.67
misjudges LC necessity—either overusing or un-
derusing it.Step 3–6 (Feasibility & Trade-offs):
Each prevents misallocation; omitting them causes
the router to use LC when RAG suffices or vice
versa.
4.5 Robustness to Metadata
Pre-Route is not built on the assumption of a fully
curated metadata schema. Instead, it deliberately
relies onlow-cost metadatathat are almost always
present in realistic long-context systems (e.g., doc-
ument length, a leading snippet), while coarser at-
tributes such as titles or task tags tend to be missing
or noisy in practice. We therefore treat length and
head as a minimal interface, and focus our analysis
on metadata that may not be present.
Importantly, Pre-Route treats the leading snippet
as asoft priorrather than a hard rule. Decisions
emerge fromjoint reasoningover the query and
structure, enabling the model to look beyond mis-
leading signals. Appendix A.6 demonstrates this
robustness: Pre-Route correctly selects LC based
on distributional cues, even when the head snippet
appears to be a sufficient local fact.
Table 6 compares performance across three meta-
data settings:Full-Meta(all fields),Head-onlyTable 6: Effect of metadata settings (Head-only,
Generated-Meta, Full-Meta) on routing performance
on the LaRA test split.
Answer Model Router Model Setting QA↑LC(%)↓Acc↑
DeepSeek-R1 Pre-Route(Q235B) [N] Head-only 3.42 20.3 0.68
Generated-Meta 3.45 20.6 0.70
Full-Meta 3.47 27.2 0.70
Pre-Route(D-Q1.7B) [N] Head-only 3.43 18.5 0.69
Generated-Meta 3.4715.8 0.73
Full-Meta3.5120.60.73
Self-Route (Baseline) 3.36 31.4 0.52
Qwen3-235B Pre-Route(Q235B) [N] Head-only 3.39 23.0 0.65
Generated-Meta3.4423.20.70
Full-Meta 3.43 27.2 0.67
Pre-Route(D-Q1.7B) [N] Head-only 3.40 21.5 0.66
Generated-Meta 3.4320.4 0.70
Full-Meta 3.43 22.7 0.69
Self-Route (Baseline) 3.34 33.9 0.52
(length/head only), andGenerated-Meta(meta-
data inferred by Qwen3-1.7B). Results confirm
that Pre-Route outperforms the Self-Route baseline
even in theHead-onlysetting, whileGenerated-
Metaeffectively closes the gap toFull-Meta,
demonstrating robust routing using only readily
available or lightweight inferred signals.
5 Conclusion
We introduced PRE-ROUTE, a proactive routing
framework that elicits LLMs’ latent routing ability
to choose between RAG and long-context before
answering. Using lightweight structured reason-
ing, Pre-Route improves routing accuracy and QA
quality while reducing LC usage. Experiments on
LaRA and LongBench-v2 show Pre-Route outper-
forms Self-Route with better accuracy–cost trade-
offs. Through distillation, small models inherit
planning ability for efficient plug-and-play deploy-
ment. These results confirm routing ability is latent
but can be reliably elicited.Future Workincludes
extending to multi-modal or open-domain scenar-
ios, hybrid routing.
8

Limitations
Our work has several limitations. First, Pre-Route’s
effectiveness depends on the quality of metadata
(e.g., document type, task type). While we demon-
strate robustness with minimal metadata (length
+ head only), performance may degrade when
metadata is completely absent or highly unreli-
able. Second, the routing decision is binary (RAG
vs. LC), which may not capture more nuanced
strategies such as hybrid approaches or adaptive
retrieval granularity. Third, our evaluation focuses
on English-language benchmarks; generalization to
other languages and domains requires further inves-
tigation. Fourth, the distillation process requires
a strong teacher model and high-quality filtered
data, which may limit accessibility for some re-
searchers. Finally, while Pre-Route reduces LC
usage, the optimal LC selection threshold may vary
across different deployment scenarios and cost con-
straints.
Ethical Considerations
This work does not involve human subjects, sensi-
tive personal data, or applications that may raise
direct ethical concerns. Our experiments are con-
ducted on publicly available datasets (LaRA and
LongBench-v2), and we adhere to responsible AI
research practices throughout this study. However,
we acknowledge several broader considerations.
First, routing systems like Pre-Route could poten-
tially be used to optimize the cost of automated con-
tent generation at scale, which may impact certain
professional sectors. Second, the system inherits
the biases and limitations present in the underlying
LLMs used for routing and answering. Third, the
retrieval-augmented approach may propagate inac-
curacies if the retrieved sources contain errors or
misinformation. We encourage users of our system
to implement appropriate content filtering and hu-
man oversight for high-stakes applications. Code
and models are released for research purposes only.
References
Guillaume Alain and Yoshua Bengio. 2017. Under-
standing intermediate layers using linear classifier
probes. In5th International Conference on Learning
Representations, ICLR 2017, Toulon, France, April
24-26, 2017, Workshop Track Proceedings. OpenRe-
view.net.
Chenxin An, Shansan Gong, Ming Zhong, Xingjian
Zhao, Mukai Li, Jun Zhang, Lingpeng Kong, andXipeng Qiu. 2024. L-eval: Instituting standardized
evaluation for long context language models. InPro-
ceedings of the 62nd Annual Meeting of the Associa-
tion for Computational Linguistics (Volume 1: Long
Papers), ACL 2024, Bangkok, Thailand, August 11-
16, 2024, pages 14388–14411. Association for Com-
putational Linguistics.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-rag: Learning to
retrieve, generate, and critique through self-reflection.
InThe Twelfth International Conference on Learning
Representations, ICLR 2024, Vienna, Austria, May
7-11, 2024. OpenReview.net.
Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu,
Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao
Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang,
and Juanzi Li. 2024a. Longbench: A bilingual, mul-
titask benchmark for long context understanding. In
Proceedings of the 62nd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), ACL 2024, Bangkok, Thailand, Au-
gust 11-16, 2024, pages 3119–3137. Association for
Computational Linguistics.
Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xi-
aozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu, Lei
Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. 2024b.
Longbench v2: Towards deeper understanding and
reasoning on realistic long-context multitasks.arXiv
preprint arXiv:2412.15204.
Iz Beltagy, Matthew E. Peters, and Arman Cohan. 2020.
Longformer: The long-document transformer.CoRR,
abs/2004.05150.
Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann,
Trevor Cai, Eliza Rutherford, Katie Millican, George
van den Driessche, Jean-Baptiste Lespiau, Bogdan
Damoc, Aidan Clark, Diego de Las Casas, Aurelia
Guy, Jacob Menick, Roman Ring, Tom Hennigan,
Saffron Huang, Loren Maggiore, Chris Jones, Albin
Cassirer, and 9 others. 2022. Improving language
models by retrieving from trillions of tokens. InIn-
ternational Conference on Machine Learning, ICML
2022, 17-23 July 2022, Baltimore, Maryland, USA,
volume 162 ofProceedings of Machine Learning
Research, pages 2206–2240. PMLR.
Lingjiao Chen, Matei Zaharia, and James Zou. 2024.
Frugalgpt: How to use large language models while
reducing cost and improving performance.Trans.
Mach. Learn. Res., 2024.
Qinyuan Cheng, Xiaonan Li, Shimin Li, Qin Zhu,
Zhangyue Yin, Yunfan Shao, Linyang Li, Tianxi-
ang Sun, Hang Yan, and Xipeng Qiu. 2024. Unified
active retrieval for retrieval augmented generation.
InFindings of the Association for Computational
Linguistics: EMNLP 2024, Miami, Florida, USA,
November 12-16, 2024, pages 17153–17166. Associ-
ation for Computational Linguistics.
Gheorghe Comanici, Eric Bieber, Mike Schaekermann,
Ice Pasupat, Noveen Sachdeva, Inderjit S. Dhillon,
9

Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen,
Luke Marris, Sam Petulla, Colin Gaffney, Asaf Aha-
roni, Nathan Lintz, Tiago Cardal Pais, Henrik Ja-
cobsson, Idan Szpektor, Nan-Jiang Jiang, and 17
others. 2025. Gemini 2.5: Pushing the frontier
with advanced reasoning, multimodality, long con-
text, and next generation agentic capabilities.CoRR,
abs/2507.06261.
Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan,
Noah A. Smith, and Matt Gardner. 2021. A dataset
of information-seeking questions and answers an-
chored in research papers. InProceedings of the
2021 Conference of the North American Chapter of
the Association for Computational Linguistics: Hu-
man Language Technologies, NAACL-HLT 2021, On-
line, June 6-11, 2021, pages 4599–4610. Association
for Computational Linguistics.
Inc. DeepSeek. 2025. Deepseek-r1-0528 release.
https://api-docs.deepseek.com/
news/news250528. Accessed: 2025-09-18.
Zican Dong, Tianyi Tang, Junyi Li, Wayne Xin Zhao,
and Ji-Rong Wen. 2024. BAMBOO: A comprehen-
sive benchmark for evaluating long text modeling
capacities of large language models. InProceed-
ings of the 2024 Joint International Conference on
Computational Linguistics, Language Resources and
Evaluation, LREC/COLING 2024, 20-25 May, 2024,
Torino, Italy, pages 2086–2099. ELRA and ICCL.
Mandy Guo, Joshua Ainslie, David C. Uthus, Santi-
ago Ontañón, Jianmo Ni, Yun-Hsuan Sung, and Yin-
fei Yang. 2022. Longt5: Efficient text-to-text trans-
former for long sequences. InFindings of the Associ-
ation for Computational Linguistics: NAACL 2022,
Seattle, WA, United States, July 10-15, 2022, pages
724–736. Association for Computational Linguistics.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat,
and Ming-Wei Chang. 2020. Retrieval augmented
language model pre-training. InProceedings of the
37th International Conference on Machine Learning,
ICML 2020, 13-18 July 2020, Virtual Event, volume
119 ofProceedings of Machine Learning Research,
pages 3929–3938. PMLR.
Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shan-
tanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang,
and Boris Ginsburg. 2024. RULER: what’s the real
context size of your long-context language models?
CoRR, abs/2404.06654.
Cheng-Yu Hsieh, Chun-Liang Li, Chih-Kuan Yeh,
Hootan Nakhost, Yasuhisa Fujii, Alex Ratner, Ranjay
Krishna, Chen-Yu Lee, and Tomas Pfister. 2023. Dis-
tilling step-by-step! outperforming larger language
models with less training data and smaller model
sizes. InFindings of the Association for Compu-
tational Linguistics: ACL 2023, Toronto, Canada,
July 9-14, 2023, pages 8003–8017. Association for
Computational Linguistics.Gautier Izacard and Edouard Grave. 2021. Leveraging
passage retrieval with generative models for open do-
main question answering. InProceedings of the 16th
Conference of the European Chapter of the Associ-
ation for Computational Linguistics: Main Volume,
EACL 2021, Online, April 19 - 23, 2021, pages 874–
880. Association for Computational Linguistics.
Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju
Hwang, and Jong Park. 2024. Adaptive-rag: Learn-
ing to adapt retrieval-augmented large language mod-
els through question complexity. InProceedings of
the 2024 Conference of the North American Chap-
ter of the Association for Computational Linguistics:
Human Language Technologies (Volume 1: Long
Papers), NAACL 2024, Mexico City, Mexico, June
16-21, 2024, pages 7036–7050. Association for Com-
putational Linguistics.
Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng
Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2024.
Longllmlingua: Accelerating and enhancing llms in
long context scenarios via prompt compression. In
Proceedings of the 62nd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), ACL 2024, Bangkok, Thailand, Au-
gust 11-16, 2024, pages 1658–1677. Association for
Computational Linguistics.
Greg Kamradt. 2023. Needle in a haystack - pres-
sure testing llms. https://github.com/
gkamradt/LLMTest_NeedleInAHaystack .
Accessed: 2025-09-18.
Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke
Zettlemoyer, and Mike Lewis. 2020. Generalization
through memorization: Nearest neighbor language
models. In8th International Conference on Learning
Representations, ICLR 2020, Addis Ababa, Ethiopia,
April 26-30, 2020. OpenReview.net.
Mosh Levy, Alon Jacoby, and Yoav Goldberg. 2024.
Same task, more tokens: the impact of input length on
the reasoning performance of large language models.
InProceedings of the 62nd Annual Meeting of the
Association for Computational Linguistics (Volume 1:
Long Papers), ACL 2024, Bangkok, Thailand, August
11-16, 2024, pages 15339–15353. Association for
Computational Linguistics.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive NLP tasks. InAdvances in Neural In-
formation Processing Systems 33: Annual Confer-
ence on Neural Information Processing Systems 2020,
NeurIPS 2020, December 6-12, 2020, virtual.
Jiaqi Li, Mengmeng Wang, Zilong Zheng, and Muhan
Zhang. 2024a. Loogle: Can long-context language
models understand long contexts? InProceedings
of the 62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
10

ACL 2024, Bangkok, Thailand, August 11-16, 2024,
pages 16304–16333. Association for Computational
Linguistics.
Kuan Li, Liwen Zhang, Yong Jiang, Pengjun Xie, Fei
Huang, Shuai Wang, and Minhao Cheng. 2025. Lara:
Benchmarking retrieval-augmented generation and
long-context llms - no silver bullet for LC or RAG
routing.CoRR, abs/2502.09977.
Xiaonan Li, Changtai Zhu, Linyang Li, Zhangyue Yin,
Tianxiang Sun, and Xipeng Qiu. 2024b. Llatrieval:
Llm-verified retrieval for verifiable generation. In
Proceedings of the 2024 Conference of the North
American Chapter of the Association for Computa-
tional Linguistics: Human Language Technologies
(Volume 1: Long Papers), NAACL 2024, Mexico City,
Mexico, June 16-21, 2024, pages 5453–5471. Associ-
ation for Computational Linguistics.
Zhuowan Li, Cheng Li, Mingyang Zhang, Qiaozhu Mei,
and Michael Bendersky. 2024c. Retrieval augmented
generation or long-context llms? A comprehensive
study and hybrid approach. InProceedings of the
2024 Conference on Empirical Methods in Natural
Language Processing: EMNLP 2024 - Industry Track,
Miami, Florida, USA, November 12-16, 2024, pages
881–893. Association for Computational Linguistics.
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024. Lost in the middle: How language
models use long contexts.Trans. Assoc. Comput.
Linguistics, 12:157–173.
OpenAI. 2025. Introducing gpt-5.
https://openai.com/index/
introducing-gpt-5/ . Accessed: 2025-
09-18.
Richard Yuanzhe Pang, Alicia Parrish, Nitish Joshi,
Nikita Nangia, Jason Phang, Angelica Chen, Vishakh
Padmakumar, Johnny Ma, Jana Thompson, He He,
and Samuel R. Bowman. 2022. Quality: Question
answering with long input texts, yes! InProceed-
ings of the 2022 Conference of the North American
Chapter of the Association for Computational Lin-
guistics: Human Language Technologies, NAACL
2022, Seattle, WA, United States, July 10-15, 2022,
pages 5336–5358. Association for Computational
Linguistics.
Tomáš Koˇ ciský, Jonathan Schwarz, Phil Blunsom,
Chris Dyer, Karl Moritz Hermann, Gábor Melis, and
Edward Grefenstette. 2018. The NarrativeQA read-
ing comprehension challenge.Transactions of the As-
sociation for Computational Linguistics, TBD:TBD.
Uri Shaham, Maor Ivgi, Avia Efrat, Jonathan Berant,
and Omer Levy. 2023. Zeroscrolls: A zero-shot
benchmark for long text understanding. InFind-
ings of the Association for Computational Linguis-
tics: EMNLP 2023, Singapore, December 6-10, 2023,
pages 7977–7989. Association for Computational
Linguistics.Weijia Shi, Sewon Min, Michihiro Yasunaga, Min-
joon Seo, Richard James, Mike Lewis, Luke Zettle-
moyer, and Wen-tau Yih. 2024. REPLUG: retrieval-
augmented black-box language models. InProceed-
ings of the 2024 Conference of the North American
Chapter of the Association for Computational Lin-
guistics: Human Language Technologies (Volume 1:
Long Papers), NAACL 2024, Mexico City, Mexico,
June 16-21, 2024, pages 8371–8384. Association for
Computational Linguistics.
Mingyang Song, Mao Zheng, and Xuan Luo. 2025.
Counting-stars: A multi-evidence, position-aware,
and scalable benchmark for evaluating long-context
large language models. InProceedings of the 31st
International Conference on Computational Linguis-
tics, COLING 2025, Abu Dhabi, UAE, January 19-24,
2025, pages 3753–3763. Association for Computa-
tional Linguistics.
Minzheng Wang, Longze Chen, Fu Cheng, Shengyi
Liao, Xinghua Zhang, Bingli Wu, Haiyang Yu, Nan
Xu, Lei Zhang, Run Luo, Yunshui Li, Min Yang, Fei
Huang, and Yongbin Li. 2024. Leave no document
behind: Benchmarking long-context llms with ex-
tended multi-doc QA. InProceedings of the 2024
Conference on Empirical Methods in Natural Lan-
guage Processing, EMNLP 2024, Miami, FL, USA,
November 12-16, 2024, pages 5627–5646. Associa-
tion for Computational Linguistics.
Yile Wang, Peng Li, Maosong Sun, and Yang Liu.
2023. Self-knowledge guided retrieval augmenta-
tion for large language models. InFindings of the
Association for Computational Linguistics: EMNLP
2023, Singapore, December 6-10, 2023, pages 10303–
10315. Association for Computational Linguistics.
Peng Xu, Wei Ping, Xianchao Wu, Lawrence McAfee,
Chen Zhu, Zihan Liu, Sandeep Subramanian, Evelina
Bakhturina, Mohammad Shoeybi, and Bryan Catan-
zaro. 2024. Retrieval meets long context large lan-
guage models. InThe Twelfth International Con-
ference on Learning Representations, ICLR 2024,
Vienna, Austria, May 7-11, 2024. OpenReview.net.
Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling.
2024. Corrective retrieval augmented generation.
CoRR, abs/2401.15884.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao,
Chengen Huang, Chenxu Lv, Chujie Zheng, Day-
iheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao
Ge, Haoran Wei, Huan Lin, Jialong Tang, and 40
others. 2025. Qwen3 technical report.CoRR,
abs/2505.09388.
Xin Zhang, Yanzhao Zhang, Dingkun Long, Wen Xie,
Ziqi Dai, Jialong Tang, Huan Lin, Baosong Yang,
Pengjun Xie, Fei Huang, and 1 others. 2024a. mgte:
Generalized long-context text representation and
reranking models for multilingual text retrieval. In
Proceedings of the 2024 Conference on Empirical
Methods in Natural Language Processing: Industry
Track, pages 1393–1412.
11

Xinrong Zhang, Yingfa Chen, Shengding Hu, Zi-
hang Xu, Junhao Chen, Moo Khai Hao, Xu Han,
Zhen Leng Thai, Shuo Wang, Zhiyuan Liu, and
Maosong Sun. 2024b. ınftybench: Extending long
context evaluation beyond 100k tokens. InProceed-
ings of the 62nd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Pa-
pers), ACL 2024, Bangkok, Thailand, August 11-16,
2024, pages 15262–15277. Association for Computa-
tional Linguistics.
A Appendix
A.1 Related Work
Long-context LLMs.Improving the ability of
LLMs to handle long inputs has long been a key
challenge. Early work reduced computational cost
by modifying attention (Beltagy et al., 2020; Guo
et al., 2022) or compressing inputs (Jiang et al.,
2024), while others explored distillation (Hsieh
et al., 2023) and cascaded mechanisms (Chen
et al., 2024). Recently, frontier models such as
GPT-5 (OpenAI, 2025), Gemini-2.5 (Comanici
et al., 2025), Qwen-3 (Yang et al., 2025), and
DeepSeek (DeepSeek, 2025) have extended context
windows to 128K tokens or way beyond. However,
due to the quadratic complexity of Transformers,
long-context reasoning remains expensive and un-
stable under ultra-long inputs, e.g., the “lost-in-the-
middle” problem (Liu et al., 2024).
Retrieval-Augmented Generation (RAG).An-
other parallel line is RAG (Lewis et al., 2020)),
which augments LLMs with external retrieval.
RAG shows strong performance in language mod-
eling (Khandelwal et al., 2020; Shi et al., 2024) and
open-domain QA (Guu et al., 2020; Izacard and
Grave, 2021), at much lower cost than modeling en-
tire documents (Borgeaud et al., 2022). Subsequent
work proposed answer correction (Yan et al., 2024),
critique (Asai et al., 2024), verification (Li et al.,
2024b), and adaptive retrieval (Wang et al., 2023;
Cheng et al., 2024; Jeong et al., 2024), enhancing
robustness in knowledge-intensive scenarios. Al-
though retrieval introduces some overhead, RAG
pipelines are modular and can be deployed locally.
Benchmarking RAG vs. LC.Evaluation has
drawn attention as well. A number of benchmarks
have been proposed, including both synthetic stress
tests (Kamradt, 2023; Hsieh et al., 2024; Song et al.,
2025) and task-oriented datasets (Bai et al., 2024a;
An et al., 2024; Dasigi et al., 2021; s Koˇ ciský
et al., 2018; Pang et al., 2022). These studies re-
vealed performance degradation as context lengthgrows (Levy et al., 2024; Hsieh et al., 2024) and
phenomena such as lost-in-the-middle (Liu et al.,
2024). Nevertheless, most benchmarks do not ex-
ceed 128K tokens, limiting their ability to fully
assess ultra-long context capabilities. This has mo-
tivated larger-scale benchmarks such asLaRA(Li
et al., 2025),LongBench-v2(Bai et al., 2024b),
and other related efforts (Zhang et al., 2024b; Sha-
ham et al., 2023; Dong et al., 2024; Li et al., 2024a;
Wang et al., 2024), which better match current
LLM capabilities.
The LaRA benchmark.LaRA (Li et al., 2025)
provides a more systematic comparison of RAG
and LC in modern LLMs. It contains test cases
across four tasks—localization, comparison, rea-
soning, hallucination detection—and three types
of natural long documents:novels, papers, reports.
Experiments on7 open-sourceand4 proprietary
models show clear complementarity:
•Model capability:weaker models rely more
on RAG, while stronger ones (e.g., GPT-5,
Gemini 2.5) perform better with LC.
•Task type:LC excels in reasoning and com-
parison, while RAG is more robust for hallu-
cination detection and refusals.
•Text structure:structured texts (e.g., reports,
papers) favor LC, while novels make RAG
more cost-effective.
•Context position & retrieval granularity:
LC suffers from “lost-in-the-middle,” while
RAG can cross-span retrieve, but is sensitive
to chunk size and number.
Routing strategies and Self-Route.To combine
both paradigms, routing-based methods have been
explored. The most representative is Self-Route (Li
et al., 2024c), which adopts afailure-drivenmech-
anism: the model first answers using RAG, then
falls back to LC if it outputsunanswerable. This
avoids explicit task recognition, is simple to im-
plement, and works reasonably well in some cases.
However, it has clear limitations: (1) reliance on
retrieval failure signals without proactive task or
information awareness; (2) extra cost from execut-
ing RAG before fallback; (3) misrouting due to
over- or under-confidence; (4) lack of interpretabil-
ity in decision logic. Optimizingefficiency and
performanceremains a central challenge for future
routing methods.
12

A.2 Optimality of the Ideal Label
Proposition A.1(Optimality of the Ideal Label).
Consider the ideal label defined in Eq. 3, which
prescribes LC if U(LC;q,D)> U(RAG;q,D)
and RAG otherwise (i.e., whenever ULC≤U RAG).
This rule ensures optimal QA performance by con-
struction, and among all decision rules πthat also
achieve optimal performance, the ideal label mini-
mizes the expected cost by defaulting to the lower-
cost RAG in tie cases.
Proof. Step 1 (Performance).By definition
(Eq. 3), the ideal label selects the option with max-
imal performance: LC when ULC> U RAG, and
RAG otherwise. Thus no alternative strategy can
yield higher performance on any instance, so QA
performance is optimal.
Step 2 (Cost).The cost of a single query can be
written as
Canswer =1{y= LC}C LC+1{y= RAG}C RAG,
where CLC> C RAG≥0. Suppose an alternative
strategy π′makes the same decisions as the ideal la-
bel on strictly-better cases, but deviates on a subset
Tof tie cases by choosing LC instead of RAG. For
eachi∈ T , performance remains equal, but cost
strictly increases by ∆Ci=C LC−C RAG >0.
Hence
E[C(π′)−C(π)] =X
i∈Tpi∆Ci>0,
wherep iis the probability of casei.
Conclusion.The ideal label therefore ensures
optimal QA performance by construction, and
among all decision rules πthat also achieve op-
timal performance, the ideal label minimizes the
expected cost by defaulting to the lower-cost RAG
in tie cases.
A.3 Datasets Details
We provide additional details of the datasets used
in our experiments.
Table 7: Statistics of LongBench-v2 datasets.
Category #data Length
I. Single-Document QA 175 51k–96k
II. Multi-Document QA 125 15k–129k
III. Long In-context Learning 81 61k–132k
IV . Dialogue Understanding 39 25k–77k
V . Code Repository 50 167k
VI. Structured Data 33 49k–52k
Total503 up to 167kTable 8: Statistics of LaRA.
Context Location Reasoning Comparison Hallucination
32k length
Novel 25673 25908 25681 25433
Financial 27548 27531 27546 27527
Paper 28078 28088 27708 28081
# of Cases 276 230 151 230
128k length
Novel 96452 96226 95903 96182
Financial 92684 92831 92830 92812
Paper 93911 93818 94731 93890
# of Cases 489 374 198 378
Table 9: Statistics of LaRA distribution of the test set
(0.7:0.1:0.2 split).
Context Location Reasoning Comparison Hallucination
32k length
Novel 16 13 10 14
Financial 33 24 14 20
Paper 10 11 6 12
# of Cases 59 48 30 46
128k length
Novel 50 35 15 32
Financial 19 16 10 20
Paper 27 27 17 27
# of Cases 96 78 42 79
Overall (32k + 128k)
Novel 66 48 25 46
Financial 52 40 24 40
Paper 37 38 23 39
# of Cases 155 126 72 125
A.4 Additional Robustness Results on
LongBench-v2 (rerank size = 5 and 10)
To provide comprehensive evaluation of Pre-
Route’s robustness across different retrieval config-
urations, we present detailed experimental results
for rerank sizes 5 and 10 in this appendix. These re-
sults complement the representative findings shown
in Table 4 and further demonstrate Pre-Route’s con-
sistent performance advantages across varying re-
trieval settings.
13

Table 10: LongBench-v2 (OOD, rerank size = 5): Pre-Route results;Bestand Second per answer model.Pre-Route
Variants:Large: R1 (DeepSeek-R1) or Q235B (Qwen3-235B); Small: Q1.7B (Qwen3-1.7B); Distilled: D-Q1.7B
(Distilled-Qwen3-1.7B).
Answer Model
Router ModelQwen3-1.7B [N] Qwen3-1.7B [T] Qwen3-4B [N] Qwen3-4B [T] Qwen3-8B [N] Qwen3-8B [T]
QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑
Always-LC (Baseline) 0.29 100.0 0.38 0.28 100.0 0.31 0.30 100.0 0.30 0.37 100.0 0.40 0.40 100.0 0.40 0.40 100.0 0.37
Always-RAG (Baseline) 0.33 0.0 0.62 0.34 0.0 0.69 0.31 0.0 0.70 0.34 0.0 0.60 0.34 0.0 0.60 0.37 0.0 0.63
Self-Route (Baseline) 0.3234.9 0.63 0.3328.1 0.68 0.30 22.7 0.69 0.3733.8 0.65 0.3831.3 0.64 0.4132.3 0.67
Pre-Route (R1) [T] 0.30 24.8 0.68 0.33 23.5 0.71 0.31 24.7 0.69 0.35 24.5 0.70 0.37 19.6 0.72 0.39 19.9 0.71
Pre-Route (Q235B) [N] 0.31 39.7 0.58 0.31 37.8 0.60 0.31 39.5 0.59 0.36 38.7 0.59 0.38 31.8 0.64 0.40 33.0 0.63
Pre-Route (Q235B) [T] 0.31 29.7 0.66 0.32 29.4 0.65 0.32 27.0 0.69 0.34 30.1 0.64 0.37 25.8 0.68 0.38 29.6 0.63
Pre-Route (Q1.7B) [N] 0.30 37.6 0.56 0.31 38.9 0.56 0.30 41.2 0.56 0.34 38.2 0.58 0.35 39.8 0.55 0.38 38.1 0.58
Pre-Route (Q1.7B) [T] 0.30 41.7 0.54 0.32 38.7 0.58 0.3242.1 0.58 0.34 40.7 0.55 0.3838.9 0.61 0.37 40.0 0.54
Pre-Route (D-Q1.7B) [N] 0.31 7.3 0.81 0.33 8.0 0.80 0.31 20.8 0.70 0.34 21.1 0.67 0.38 20.7 0.70 0.38 21.9 0.68
Pre-Route (D-Q1.7B) [T] 0.32 6.6 0.82 0.33 7.5 0.80 0.31 20.2 0.71 0.34 20.4 0.66 0.36 18.9 0.69 0.36 21.2 0.68
Answer Model
Router ModelQwen3-30B [N] Qwen3-30B [T] Qwen3-235B [N] Qwen3-235B [T] DeepSeek-R1 [T] Qwen-Max [N]
QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑
Always-LC (Baseline) 0.40 100.0 0.40 0.36 100.0 0.56 0.46 100.0 0.60 0.52 100.0 0.54 0.53 100.0 0.49 0.48 100.0 0.59
Always-RAG (Baseline) 0.39 0.0 0.60 0.22 0.0 0.44 0.42 0.0 0.40 0.43 0.0 0.46 0.45 0.0 0.51 0.40 0.0 0.41
Self-Route (Baseline) 0.40 38.0 0.61 0.2847.8 0.51 0.45 60.9 0.44 0.4952.2 0.52 0.46 44.4 0.53 0.4557.4 0.46
Pre-Route (R1) [T] 0.41 25.9 0.70 0.25 25.2 0.62 0.47 23.5 0.72 0.45 23.8 0.63 0.47 27.6 0.61 0.43 25.8 0.68
Pre-Route (Q235B) [N] 0.42 39.0 0.60 0.27 40.0 0.53 0.47 37.4 0.62 0.47 41.3 0.55 0.51 41.8 0.56 0.44 40.0 0.59
Pre-Route (Q235B) [T] 0.41 29.2 0.67 0.24 28.5 0.58 0.45 27.4 0.67 0.47 28.5 0.61 0.51 31.1 0.65 0.44 29.0 0.65
Pre-Route (Q1.7B) [N] 0.38 39.9 0.57 0.27 35.3 0.58 0.45 36.7 0.61 0.47 37.1 0.55 0.47 42.3 0.53 0.4538.7 0.60
Pre-Route (Q1.7B) [T] 0.40 41.0 0.57 0.27 41.2 0.54 0.44 40.5 0.56 0.46 43.0 0.55 0.47 41.3 0.57 0.44 41.8 0.56
Pre-Route (D-Q1.7B) [N] 0.42 27.6 0.68 0.27 31.2 0.61 0.46 27.9 0.67 0.47 28.7 0.61 0.46 18.4 0.64 0.44 27.3 0.66
Pre-Route (D-Q1.7B) [T] 0.42 28.0 0.67 0.28 29.5 0.62 0.45 26.7 0.66 0.46 29.0 0.58 0.48 21.9 0.66 0.45 27.7 0.66
Table 11: LongBench-v2 (OOD, rerank size = 10): Pre-Route results;Bestand Second per answer model.Pre-Route
Variants:Large: R1 (DeepSeek-R1) or Q235B (Qwen3-235B); Small: Q1.7B (Qwen3-1.7B); Distilled: D-Q1.7B
(Distilled-Qwen3-1.7B).
Answer Model
Router ModelQwen3-1.7B [N] Qwen3-1.7B [T] Qwen3-4B [N] Qwen3-4B [T] Qwen3-8B [N] Qwen3-8B [T]
QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑
Always-LC (Baseline) 0.29 100.0 0.32 0.27 100.0 0.30 0.30 100.0 0.26 0.36 100.0 0.30 0.40 100.0 0.27 0.40 100.0 0.28
Always-RAG (Baseline) 0.32 0.0 0.68 0.31 0.0 0.70 0.33 0.0 0.74 0.39 0.0 0.70 0.43 0.0 0.73 0.43 0.0 0.72
Self-Route (Baseline) 0.3328.2 0.69 0.31 24.2 0.70 0.33 19.5 0.74 0.38 26.00.68 0.41 21.1 0.72 0.41 24.3 0.70
Pre-Route (R1) [T] 0.31 25.9 0.68 0.32 24.8 0.71 0.33 23.6 0.70 0.40 29.5 0.64 0.44 19.3 0.74 0.45 20.9 0.76
Pre-Route (Q235B) [N] 0.31 39.5 0.58 0.31 40.2 0.59 0.33 38.6 0.57 0.39 40.1 0.57 0.45 31.7 0.66 0.46 32.8 0.67
Pre-Route (Q235B) [T] 0.32 29.7 0.66 0.31 27.5 0.68 0.34 29.4 0.66 0.39 30.4 0.64 0.43 27.4 0.66 0.45 27.5 0.70
Pre-Route (Q1.7B) [N] 0.3339.9 0.59 0.31 34.1 0.62 0.32 35.8 0.58 0.36 38.0 0.56 0.41 40.0 0.56 0.42 36.5 0.61
Pre-Route (Q1.7B) [T] 0.31 40.3 0.56 0.31 41.1 0.57 0.33 41.6 0.56 0.38 38.9 0.56 0.43 39.1 0.60 0.43 42.2 0.57
Pre-Route (D-Q1.7B) [N] 0.33 7.6 0.83 0.31 7.5 0.82 0.34 21.2 0.71 0.38 23.7 0.68 0.42 23.4 0.71 0.42 21.6 0.69
Pre-Route (D-Q1.7B) [T] 0.32 6.8 0.83 0.31 9.5 0.80 0.32 20.6 0.70 0.37 22.4 0.66 0.42 22.0 0.72 0.44 20.9 0.71
Answer Model
Router ModelQwen3-30B [N] Qwen3-30B [T] Qwen3-235B [N] Qwen3-235B [T] DeepSeek-R1 [T] Qwen-Max [N]
QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑ QA↑LC(%)↓Acc↑
Always-LC (Baseline) 0.41 100.0 0.31 0.36 100.0 0.46 0.46 100.0 0.50 0.51 100.0 0.49 0.53 100.0 0.38 0.48 100.0 0.47
Always-RAG (Baseline) 0.45 0.0 0.69 0.25 0.0 0.54 0.48 0.0 0.50 0.45 0.0 0.51 0.49 0.0 0.62 0.48 0.0 0.53
Self-Route (Baseline) 0.46 29.0 0.70 0.3035.0 0.60 0.48 54.0 0.49 0.5044.0 0.57 0.50 28.9 0.63 0.48 46.2 0.53
Pre-Route (R1) [T] 0.45 22.8 0.73 0.27 25.0 0.64 0.51 25.6 0.73 0.47 23.3 0.66 0.51 24.5 0.69 0.49 25.8 0.70
Pre-Route (Q235B) [N] 0.46 36.3 0.62 0.29 37.9 0.58 0.51 38.3 0.63 0.48 40.2 0.57 0.51 39.7 0.58 0.50 39.6 0.60
Pre-Route (Q235B) [T] 0.45 29.7 0.67 0.26 31.1 0.58 0.51 26.3 0.71 0.48 31.0 0.61 0.50 32.4 0.61 0.50 30.8 0.67
Pre-Route (Q1.7B) [N] 0.45 48.0 0.53 0.3042.3 0.57 0.50 35.8 0.62 0.5036.7 0.60 0.5341.7 0.59 0.48 38.7 0.59
Pre-Route (Q1.7B) [T] 0.46 43.2 0.57 0.3041.1 0.57 0.50 42.3 0.57 0.49 39.5 0.56 0.50 43.6 0.52 0.48 41.1 0.57
Pre-Route (D-Q1.7B) [N] 0.46 25.9 0.71 0.29 27.6 0.63 0.50 27.5 0.70 0.50 30.0 0.62 0.53 23.0 0.66 0.48 23.9 0.69
Pre-Route (D-Q1.7B) [T] 0.47 25.2 0.72 0.28 26.9 0.61 0.51 27.5 0.70 0.48 29.0 0.61 0.52 19.1 0.69 0.48 25.6 0.69
14

A.5 Case Study on the Necessity of a
Finer-grained Evaluation Metric
To illustrate why binary evaluation is insufficient,
we present a case study highlighting the value of
a 4-point scoring rubric. This example shows how
two answers may both be judged "correct" under
binary metrics, yet differ substantially in complete-
ness and alignment with ground truth. The case
study below (Tab. A.5) concretely demonstrates
how finer-grained evaluation better reflects qualita-
tive differences.
Case Study 1: “Why does Stella repeat-
edly think their father will ‘kill’ Miles?”
1) Initial Analysis: The Query and
RAG’s Scope
Query:"Why does Stella repeatedly think
that their father will ’kill’ Miles?"
The presence of the keywordrepeatedly
is crucial, as it implies that the necessary
context is likely scattered across multiple
passages, requiring a broader understanding
than localized snippets might provide.
RAG Retrieved Snippets Analysis:The
retrieved passages contain phrases like
"Miles was expelled from school"and
"dad will be very angry."These snippets
allow a model to construct a direct causal
chainbut inherently lack the historical
context needed to address the "repeat-
edly" aspect of the query.
Ground Truth:Stella knows that Miles
has been expelled from multiple schools,
and when she realizes he got kicked out
again, she’s convinced their father will be
furious and punish Miles harshly because
it’s an ongoing issue, and they have high
expectations for him.
2) Comparative Analysis and Scoring
Long-Context Answer (Score: 4/4):
"Stella repeatedly thinks their father will
’kill’ Miles because she knows he was ex-
pelledagain. She anticipates not only her
father’s anger but also hisdisappointment,
given thehigh expectationsthey have for
Miles, and thus fears asevere outcome."
RAG Answer (Score: 3/4):"Stella thinks
their father will ’kill’ Miles because Miles
was expelled from school, and she knowstheir father will be very angry and punish
him severely."
Analysis:Under a binary (correct/incor-
rect) system, both answers would likely be
marked "correct" as they identify the core
reason (the expulsion). However, this masks
a substantial quality gap. To properly evalu-
ate them, we compare them against the key
components of the ground truth:
•The expulsion is a repeated event
(again / multiple schools).
•The father hashigh expectationsfor
Miles.
•The father’s reaction includes extreme
anger and disappointment.
•The consequences are expected to be
severe.
TheRAG answercovers the basic causal
chain but fails to address the crucial con-
texts ofrepetitionandhigh expectations.
In contrast, theLong-Context answersuc-
cessfully incorporates these nuances, align-
ing almost perfectly with the ground truth.
Our 4-point rubric is designed to capture
these critical distinctions. By assigning a
higher score to the Long-Context answer,
we can accurately reflect its deeper contex-
tual alignment and superior quality.
Conclusion:The Long-Context strat-
egy is demonstrably superior for
this query ( better_strategy =
long_context).
A.6 Case Studies on Why Self-Route
Sometimes Fails on LaRA
Across our case studies, a consistent pattern
emerges:Self-Route tends to fail at the extremes
of confidence.
•Over-confidence.When retrieved chunks ap-
pear superficially relevant, Self-Route prema-
turely concludes that RAG suffices, even if the
query requiresdistributed reasoningacross mul-
tiple parts of the document. In Case 1, it ignored
the global cue “repeatedly” and thus failed to
capture the need for long-context integration.
•Under-confidence.Conversely, when the evi-
15

dence is present but takes the form of anegative
premise check(“not discussed”) or akeyword-
localized fact(e.g., a single time point), Self-
Route misclassifies the situation as “unanswer-
able.” This leads to unnecessary escalation to
long-context models, wasting cost and provid-
ing no additional benefit (Cases 2 and 3).
In other words, Self-Route lacks a calibrated no-
tion of when retrieved evidence issufficientversus
when it isincomplete. Our router, by relying on
lightweight meta signals (question form, genre, and
evidence distribution cues), avoids these extremes:
it dispatchesdistributedqueries to Long-Context
(Case 1), while keepinglocalizedorpremise-check
queries on RAG (Cases 2 and 3).
Case Study 1: “Why does Stella repeat-
edly think their father will ‘kill’ Miles?”
What was asked?
ExplainwhyStella keeps thinking this (note
the word“repeatedly”).
Where is the evidence?
Scattered across the story: repeated expul-
sions (again), strict family expectations,
and the fear of harsh punishment.
What did the two answering strategies
say?
•RAG (picked by Self-Route):Gave the
short cause: Miles was expelled ⇒father
angry/punish.Missedthe “again” and
"high expectations".Score: 3.
•Long-Context (picked by our router):
Added the missing parts: expelledagain,
family expectations, and "angry +disap-
pointed".Score: 4.
Who routed what? Why?
•Self-RoutechoseRAGbecause re-
trieved chunks looked answerable.
•Our routerchoseLong-Contextbe-
cause“repeatedly”means the proof is
spread out.
Why did Self-Route miss?
It treated “retrieved chunks can answer” as
enough and ignored the global signal“re-
peatedly”.
Why our routing was correct.Using meta
signals only: the question includes“repeat-
edly”, the genre is long-form narrative, andthe document fits the window. These cues
imply evidence isdistributedacross scenes,
soLong-Contextis the appropriate choice.
Case Study 2: “Did the paper discuss
ReLU-at-classification for NLP?”
What was asked?
Check whetherpaper 1talks about using
ReLU as aclassification layerforNLP,
compared with traditional models.
Where is the evidence?
Right in the abstract/introduction: the paper
runs onMNIST, FashionMNIST, WDBC
(image/biomedical),not NLP.
Ground truth.
“Paper 1 does not address the impact . . . on
NLP tasks.”
What did the two answering strategies
say?
•RAG:Said the paperdoes not discuss
NLP (correct).Score: 4.
•Long-Context:Same conclusion (also
correct).Score: 4.
Who routed what? Why?
•Self-Routesent it toLong-Contextafter
treating checking retrieved chunks and
made“not discussed” as “unanswerable”.
•Our routerpickedRAGbecause this is
apremise check; the abstract is enough.
Why did Self-Route miss?
It confused a valid negative answer (“not
discussed”) with “unanswerable”, and used
more context than needed.
Why our routing was correct.Based
solely on meta information: the query is
apremise check("did it discuss . . . ?"), and
such evidence is typicallylocalizedin the
abstract/introduction. Under these cues and
the "prefer RAG on ties" rule,RAGis rec-
ommended for efficiency.
16

Case Study 3: “On April 2 when did the
whole view fall into darkness?”
What was asked?
A single time point on April 2.
Where is the evidence? (verbatim)
“. . . until,at five minutes before seven, the
whole surface in view was enveloped in the
darkness of night.”
What did the two answering strategies
say?
•RAG:“at five minutes before seven
(6:55)”Score: 4.
•Long-Context:“At five minutes before
seven.”Score: 4.
Who routed what? Why?
•Self-Routeescalated toLong-Context
after saying “unanswerable”.
•Our routerchoseRAGbecause this is
akeyword-localizablefact (date/event/-
time).
Why did Self-Route miss?
Even though the answer is in retrieved
chunks, somehow Self-Route failed to give
answer.
Why our routing was correct.From meta-
only cues: the query asks for asingle time
pointwith clear date/time keywords that
usually appears inone sentence. These
signals favorRAGfor precise, keyword-
localizable extraction.A.7 Comparison with Traditional ML
Routers
To isolate the contribution of LLM-based struc-
tured reasoning from the predictive value of meta-
data alone, we train Decision Tree and Random For-
est classifiers using thesame metadata features
as Pre-Route (document length, question length,
task type, document type, difficulty level) on the
LaRA training set. Results are averaged across
3 answer models (qwen3-4b, qwen3-235b-a22b,
deepseek-r1-0528).
Table 12: Pre-Route SFT vs. traditional ML classifiers
on LaRA. All methods use identical metadata features.
Results averaged across 3 answer models.
Method QA Score LC Rate Accuracy
Pre-Route SFT 3.256 0.229 0.708
Random Forest (n=8) 3.106 0.317 0.641
Decision Tree (depth=6) 3.014 0.414 0.572
Pre-Route outperforms ML baselines by +6.7–
13.6% accuracy over Random Forest and Decision
Tree respectively, with lower LC Rate (22.9% vs.
31.7–41.4%). This shows that: (1) Metadata con-
tains useful routing signals (ML achieves 57–64%
accuracy). (2) Distilled knowledge provides ad-
ditional value—the +6.7–13.6% gain comes from
structured reasoning captured from large models.
(3) ML models struggle on LongBench-v2 because
its metadata categories (task_type, doc_type) are
disjoint from LaRA’s, while Pre-Route leverages
LLM semantic understanding for zero-shot trans-
fer.
A.8 Performance Breakdown by Document
Type and Difficulty
By document type.Table 13 shows Pre-Route’s
advantage across document types (Paper, Finan-
cial, Novel). Improvements are consistent, with the
largest QA gains on Paper (+0.16) and Financial
(+0.14) documents.
Table 13: Performance breakdown by document type
(Qwen3-235B answer model, D-Q1.7B router).
Self-Route Pre-Route
Doc Type QA Acc QA Acc∆QA∆Acc
Paper 3.70 0.47 3.86 0.72+0.16 +0.25
Financial 3.33 0.54 3.47 0.78+0.14 +0.24
Novel 3.20 0.54 3.21 0.67 +0.01+0.13
17

Figure 5: Left: QA Score and Route Accuracy breakdown by difficulty level. Right: Radar chart comparing
Self-Route and Pre-Route across task types (Qwen3-235B answer model).
By difficulty level.Table 14 reveals where Pre-
Route’s advantage concentrates. The most striking
finding is on Hallucination tasks: Self-Route’s ac-
curacy drops to 0.22 (vs. Pre-Route’s 0.75), indi-
cating its failure-driven “unanswerable” trigger is
particularly unreliable for hallucination-sensitive
queries. On Location tasks, Pre-Route achieves
0.88 accuracy, confirming that structured reason-
ing correctly identifies keyword-localizable queries
that RAG can handle efficiently.
Table 14: Performance breakdown by difficulty level
(Qwen3-235B answer model, D-Q1.7B router).
Self-Route Pre-Route
Level QA Acc QA Acc∆QA∆Acc
Hallucination 3.38 0.22 3.53 0.75+0.15 +0.53
Location 3.55 0.69 3.63 0.88 +0.08+0.19
Reasoning 3.18 0.53 3.27 0.59 +0.09 +0.06
Complex 3.41 0.64 3.40 0.62 -0.01 -0.02
A.9 Small Model Error Analysis
We analyze samples where the 235B router suc-
ceeds but the prompt-only 1.7B router fails, reveal-
ing the main error patterns for small models:
Over-conservatism (74.3% of errors):For sim-
ple location or factual queries, the 1.7B model is
intimidated by long documents or technical jargon,
safely but inefficiently defaulting to LC when RAG
is perfectly sufficient. By difficulty level, these
over-conservative errors concentrate on Location
(46.7%) and Hallucination (29.4%) tasks.
Root cause:The fundamental gap lies in global
semantic understanding vs. surface heuristics. The
235B model understands the query’s deep struc-
ture and resists surface-level traps, while the 1.7B
model relies on shallow cues (document length,
jargon presence). This capability gap explicitly
justifies our distillation pipeline: it transfers stabledecision boundaries to small models, rather than
relying on their brittle zero-shot reasoning.
A.10 Training Details
The distilled router (Qwen3-1.7B) is trained with
the following hyperparameters:
•Optimizer: AdamW (lr=2e-5,
weight_decay=0.01)
•Schedule: Cosine LR with 10% warmup steps
• Epochs: 6
• Batch size: 8 (gradient accumulation=4)
• Max sequence length: 4000 tokens
• Precision: BF16
A.11 Reproducibility Statement
We have made efforts to ensure reproducibility. The
main paper describes the proposed method, model
configurations, and evaluation protocols. Addi-
tional training details, hyperparameters, and ex-
tended results are provided in the appendix.
A.12 Example Prompts
We provide the full prompts used in our exper-
iments. Figs. 6 and 7 shows the original Self-
Route prompt, while Figs 8 and 9 present evalua-
tion prompts adapted from LaRA for hallucination
detection and general answer comparison. Finally,
Fig. 10 gives the complete Pre-Route prompt.
18

Complete Prompt for Self-Route
Answer the question based on the given passages. Only give me
the answer and do not output any other words. If the
question cannot be answered based on the information in the
article, write "unanswerable". The following are given
passages. {rag_result} Question: {query} Answer:
Figure 6: The complete Self-Route Prompt for LaRA. Taken directly from its original paper (Li et al., 2024c)
Complete Prompt for Self-Route
Read the following text and answer briefly. {rag_chunks} Now,
answer the following question based on the above text, only
give me the answer and do not output any other words. If the
question cannot be answered based on the information in the
article, write "unanswerable". Question: {query} Answer:
Figure 7: The complete Self-Route Prompt for Longbench-v2 multiple-choice. Taken directly from its original
paper (Li et al., 2024c)
19

Evaluation Prompt for Hallucination Detection (Adapted from LaRA)
You will be given a question , a groundtruth answer, and two answers from AI
assistants ( Assistant A and Assistant B).
This question may not be directly answerable from the source text .
Your task :
1. Check whether each assistant is hallucinating . If the answer is **consistent **
with the groundtruth , and the question is **not present in the original text **,
it is considered **not hallucinating **.
2. For each answer, assign a quality score from 1 to 4:
− 4 = Fully correct (not hallucinating and consistent )
− 3 = Mostly correct (minor detail missing or uncertain )
− 2 = Partially correct (some correct points , some hallucination )
− 1 = Hallucinated , irrelevant , or incorrect
3. Write a brief comparative analysis .
4. Decide who is better : "A", "B", or "Tie"
Return JSON in this format:
{
" analysis ": "...",
"score_a ": int ,
"score_b ": int ,
" better ": "A" | "B" | "Tie"
}
[Question]
{query}
[Groundtruth Answer]
{ label }
[ Assistant A’s Answer]
{pred_a}
[ Assistant B’s Answer]
{pred_b}
Start your evaluation :
Figure 8: An adapted prompt from the LaRA benchmark, specifically refined for evaluating hallucination. It
provides a detailed scoring rubric to distinguish between factual consistency and fabrication.
20

Standard Evaluation Prompt for Answer Comparison (Adapted from LaRA)
You w i l l be g i v e n a q u e s t i o n , a g r o u n d t r u t h answer , and two AI
a s s i s t a n t answers .
Your t a s k :
1 . Judge each answer ’ s f a c t u a l a c c u r a c y and c o m p l e t e n e s s
r e l a t i v e t o t h e g r o u n d t r u t h .
2 . For each answer , a s s i g n a s c o r e from 1 t o 4 :
− 4 = F u l l y c o r r e c t ( c o v e r s a l l key p o i n t s , no f a c t u a l or
l o g i c a l f l a w s )
− 3 = Mostly c o r r e c t ( c o r e i s r i g h t , minor f l a w s or
o m i s s i o n s )
− 2 = P a r t i a l l y c o r r e c t ( some a c c u r a t e p a r t s , b u t n o t a b l e
e r r o r s or gaps )
− 1 = I n c o r r e c t or h a l l u c i n a t e d ( s e r i o u s flaws , i r r e l e v a n t ,
f a b r i c a t e d )
3 . P r o v i d e a b r i e f comparison e x p l a i n i n g your s c o r i n g .
4 . Choose which a s s i s t a n t i s b e t t e r : "A" , "B" , or " Tie "
R e t u r n your r e s u l t i n t h e f o l l o w i n g JSON f o r m a t :
{
" a n a l y s i s " : " . . . " ,
" s c o r e _ a " : i n t ,
" s c o r e _ b " : i n t ,
" b e t t e r " : "A" | "B" | " Tie "
}
[ Q u e s t i o n ]
{ query }
[ G r o u n d t r u t h Answer ]
{ l a b e l }
[ A s s i s t a n t A’ s Answer ]
{ p r e d _ a }
[ A s s i s t a n t B’ s Answer ]
{ pred_b }
S t a r t your e v a l u a t i o n :
Figure 9: A general-purpose evaluation prompt adapted from LaRA for detailed comparison. This version focuses
on factual accuracy, completeness, and overall quality against a groundtruth answer.
21

Complete Prompt for Pre-Route
## Inputs
− query: "{query}"
− task_type : {task_type}
− document_title : "{ doc_title }"
− document_type: {doc_type}
− document_length: { length_str } ({doc_len_tokens} tokens)
− answering_model: {model}
− answering_max_window: {max_window_tokens} tokens
− document_fits_window: {document_fits}
− document_head content: "{doc_head }..."
## RAG Configuration
− chunk_size: {chunk_size} tokens
− chunk_overlap: {chunk_overlap} tokens
− embed_model: {embed_model}
− rerank_model: {rerank_model}
− vector_ratio : { vector_ratio }
− rerank_size : { rerank_size }
## Instructions
You are tasked with choosing the most appropriate strategy −− **RAG **,**LONG_CONTEXT **−− for answering the user query, based on the
characteristics of the query, document, and model.
Please complete the ‘< thinking >‘ block below. In each step , provide a clear judgment and explain how it affects your strategy choice .
1. ‘<step1 >‘: Identify the question type (e.g ., factual , reasoning , comparison, judgment, etc .) and the document type (e.g ., book, article ,
report ) . How do these affect the need for deep context understanding or precise retrieval ?
2. ‘<step2 >‘: Assess whether the relevant information is likely concentrated in one part of the document or scattered across multiple
sections . How does this affect strategy selection ?
3. ‘<step3 >‘: Evaluate whether the document can fully fit into the context window, based on ‘document_fits_window‘ above. If not , how does
that impact strategy choice?
4. ‘<step4 >‘: Consider whether the query can be answered through keyword−based retrieval (e.g ., names, dates ) , or requires synthesizing
implicit logic , analogies , or multi−part reasoning .
5. ‘<step5 >‘: Reflect on the model being used (e.g ., {model}). Consider both its context window size and model capacity (parameters) .
Although some models may have large context windows, smaller models may still struggle with effective long−context reasoning due to
limited capacity . How does this influence your strategy recommendation?
6. ‘< step6_efficiency >‘: Compare **expected efficiency **of RAG vs LONG_CONTEXT: expected context size, retrieval selectivity , latency , and
cost . If quality is likely similar , which strategy is more efficient ?
7. ‘< reflection >‘: Based on your reasoning above, state which strategy is more suitable overall −− **RAG **,**LONG_CONTEXT **−− and
explain why.
8. ‘< decision >‘: Write your final strategy choice clearly as either ‘RAG‘, ‘LONG_CONTEXT‘.
### Decision Rules
− If both strategies are **equally suitable **or quality difference is **negligible / uncertain **,**prefer RAG for efficiency **.
− Prefer **LONG_CONTEXT **only if (a) the document **fits**in the window **and**(b) the query requires **global , cross−section
synthesis **that retrieval would likely miss.
− Prefer **RAG **when the document **does not fit **the window, or when **high−precision snippet retrieval **is likely effective (e.g .,
names, dates , localized facts ) , or when **efficiency **is a concern and quality is similar .
## Output Format
<thinking>
<step1 >...</ step1>
<step2 >...</ step2>
<step3 >...</ step3>
<step4 >...</ step4>
<step5 >...</ step5>
< step6_efficiency >...</ step6_efficiency >
< reflection >...</ reflection >
<decision >...</ decision >
</ thinking >
Figure 10: The complete Pre-Route Prompt.
22