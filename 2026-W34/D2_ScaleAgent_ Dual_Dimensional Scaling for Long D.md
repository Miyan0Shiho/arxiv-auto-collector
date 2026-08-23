# D2-ScaleAgent: Dual-Dimensional Scaling for Long Document Understanding

**Authors**: Hao Zhang, Longrong Yang, Lunhao Duan, Ziyang Wang, Qing-Guo Chen, Shanshan Zhao

**Published**: 2026-08-17 11:15:27

**PDF URL**: [https://arxiv.org/pdf/2608.16417v1](https://arxiv.org/pdf/2608.16417v1)

## Abstract
Multi-modal retrieval-augmented generation (RAG) is a key technique for visually rich long document understanding. Existing multi-modal RAG methods are progressively advancing toward multi-agent systems: they first retrieve relevant pages based on a query, and then iteratively understand information within those pages. However, these methods typically rely on fixed workflows and lack the ability to dynamically scale computation at test time, often leading to insufficient evidence. To address this, we propose D2-ScaleAgent, an agentic framework that introduces a dual-dimensional scaling paradigm for retrieval and reasoning. The core of D2-ScaleAgent is a Verifier agent-driven dynamic routing loop based on the intrinsic difficulty of the query, centered around a continuously updated evidence bank that serves as the agent's dynamic working memory: when retrieval needs to be expanded, the agent routes outward (retrieval scaling), decomposing the query into attributes and performing parallel page retrieval, followed by adaptive pruning to ensure comprehensive evidence coverage. When fine-grained reasoning is required, the agent routes inward (reasoning scaling), dynamically selecting sub-agents with varying granularity and count to extract evidence from pages. Finally, D2-ScaleAgent achieves logical closure over the evidence chain. Extensive experiments demonstrate that D2-ScaleAgent is effective on long and visually rich document benchmarks like MMLongBench-Doc, LongDocURL, etc.

## Full Text


<!-- PDF content starts -->

D2-ScaleAgent: Dual-Dimensional Scaling for Long Document
Understanding
Hao Zhang1, Longrong Yang2, Lunhao Duan2, Ziyang Wang3, Qing-Guo Chen2, Shanshan Zhao2
1Zhejiang University
2Alibaba Group3University of Science and Technology of China
Abstract
Multi-modal retrieval-augmented generation
(RAG) is a key technique for visually rich long
document understanding. Existing multi-modal
RAG methods are progressively advancing to-
ward multi-agent systems: they first retrieve
relevant pages based on a query, and then it-
eratively understand information within those
pages. However, these methods typically rely
on fixed workflows and lack the ability to dy-
namically scale computation at test time, often
leading to insufficient evidence. To address
this, we propose D2-ScaleAgent, an agentic
framework that introduces a dual-dimensional
scaling paradigm for retrieval and reasoning.
The core of D2-ScaleAgent is a Verifier agent-
driven dynamic routing loop based on the in-
trinsic difficulty of the query, centered around a
continuously updated evidence bank that serves
as the agent’s dynamic working memory: when
retrieval needs to be expanded, the agent routes
outward (retrieval scaling), decomposing the
query into attributes and performing parallel
page retrieval, followed by adaptive pruning
to ensure comprehensive evidence coverage.
When fine-grained reasoning is required, the
agent routes inward (reasoning scaling), dy-
namically selecting sub-agents with varying
granularity and count to extract evidence from
pages. Finally, D2-ScaleAgent achieves logi-
cal closure over the evidence chain. Extensive
experiments demonstrate that D2-ScaleAgent
is effective on long and visually rich document
benchmarks like MMLongBench-Doc, Long-
DocURL, etc.
1 Introduction
Answering questions from visually rich long docu-
ments, like financial reports and scientific papers,
is critical for many real-world applications (Suri
et al., 2025; Van Landeghem et al., 2023; Wang
et al., 2023; Shorten et al., 2026; Ding et al., 2022).
While Large Vision-Language Models (LVLMs)
have made significant progress in general visual
Figure 1: Architecture and performance of our proposed
method, D2-ScaleAgent. (a) The framework achieves
dual-dimensional scaling through a Verifier agent-driven
dynamic routing loop, addressing evidence insufficiency
by dynamically scaling outward for retrieval and inward
for fine-grained understanding. (b) It shows improve-
ments in comprehensive end-to-end QA accuracy across
multiple long-document benchmarks, consistently out-
performing existing baselines.
understanding (Li et al., 2023; Liu et al., 2024; Bai
et al., 2023; Team et al., 2023; Zhu et al., 2023),
they still struggle with visually rich long docu-
ments. A core conflict exists between the need
1
arXiv:2608.16417v1  [cs.CL]  17 Aug 2026

for fine-grained visual details and the limited con-
text window of LVLMs. This mismatch often leads
to performance drops when processing long doc-
uments (Zong et al., 2024; Xu et al., 2021; Ma
et al., 2024). Crucially, the main challenge of long-
document understanding is not just the document’s
length, but the difficulty of finding "sufficient and
adequate" evidence within limited computational
resources (Wang et al., 2025; Chen et al., 2025;
Chu et al., 2025; Xiao et al., 2025).
To address this challenge, current research typ-
ically employs two primary strategies. The first
involves multi-modal retrieval-augmented genera-
tion (RAG), which filters document content to first
identify relevant pages before performing detailed
reading (Tanaka et al., 2025; Gao et al., 2025). The
second route focuses on multi-agent or iterative
mechanisms, which utilize multi-step analysis and
reasoning to navigate complex document structures
(Wang et al., 2025; Ke et al., 2025). However, as
illustrated in Figure 1(a), these methods are often
constrained by fixed workflows, acting more like
fixed workflows than autonomous agents, lacking
the flexibility to dynamically scale computation
at test time based on the inherent difficulty of the
user’s query (Zhao et al., 2025). Consequently,
many errors arise not because the model’s reason-
ing template is flawed, but because the system fails
to acquire sufficient evidence during generation
(Sivakumar et al., 2026). The fundamental essence
of long-document understanding failure is, there-
fore, an "Evidence Insufficiency Problem".
When human readers encounter insufficient in-
formation in a long document, they do not jump
to conclusions (Pirolli and Card, 1995; Kobayashi
and Kable, 2024). Instead, they distinguish be-
tween a failure to locate the correct page (van der
Sluis, 2025) and a failure to thoroughly process
the relevant section (Li et al., 2024; Qian et al.,
2026). Inspired by this, we categorize evidence
insufficiency into two dimensions: (1) Breadth in-
sufficiency, where incomplete coverage requires
the system to scale outward to broader document
contexts; and (2) Depth insufficiency, where local-
ized but superficial understanding requires the sys-
tem to scale inward for finer granularity and deeper
reasoning. Consequently, rather than following a
fixed path, long document understanding should
dynamically scale across both dimensions based
on detected evidence gaps. To this end, we pro-
pose D2-ScaleAgent, a dual-dimensional evidence-
driven agentic framework.D2-ScaleAgent abandons the typical static
"retrieve-then-read" workflow, establishing instead
a Verifier agent-driven closed-loop agent system
centered around a continuously updated Evidence
Bank, which acts as the agent’s dynamic working
memory. This Evidence Bank maintains the cur-
rently acquired page, region, and atomic evidence,
recording their logical states: whether they sup-
port the answer, conflict, or remain missing. The
core of this closed loop is a Verifier agent-driven
dynamic routing. As shown in Figure 1(a), the Ver-
ifier serves as a trigger for the entire system, which
strictly checks if the current Evidence Bank is suf-
ficient to support a complete response. When an
evidence gap is detected, the Verifier outputs a gap
signal and dynamically routes the compute: When
new source pages are missing, it autonomously
routes outward to retrieval scaling. The retrieval
scaling triggers an attribute decomposition of the
original query, performing parallel candidate re-
trieval from multiple perspectives while applying
rank fusion and adaptive pruning to strictly con-
trol context budgets. When fine-grained details are
missing, it routes inward to reasoning scaling, dy-
namically selecting sub-agents of varying costs and
granularities, including Global Surveyor, Region
Locator, and Fine-grained Extractor. Finally, the
agentic system iteratively completes the evidence
chain under the guidance of the Verifier until logi-
cal closure is achieved: a state where all identified
evidence gaps are resolved to substantiate the final
answer.
In summary, our major contributions are as fol-
lows:
•We identify the fundamental cause of failure
in long document understanding as an "evi-
dence insufficiency problem" and introduce
the "Dual-Dimensional Scaling" paradigm.
•We propose the D2-ScaleAgent framework.
Its primary contribution is a dynamic closed-
loop mechanism, driven by a Verifier agent
and centralized around an Evidence Bank that
serves as the system’s global memory, en-
abling profound synergy and seamless fall-
back between retrieval and reasoning opera-
tions.
•Extensive evaluations on multiple complex,
visually-rich long document benchmarks
(e.g., MMLongBench-Doc (Ma et al., 2024))
demonstrate that D2-ScaleAgent achieves
2

state-of-the-art performance, outperforming
traditional RAG and static-agent workflows,
as shown in Figure 1(b).
2 Related Work
Multi-modal Retrieval-Augmented Generation.
Multi-modal RAG enhances document question-
answering capabilities by supplying models with
externally retrieved visual or textual contexts
(Lewis et al., 2020; Guu et al., 2020; Borgeaud
et al., 2022). Recent studies have achieved signifi-
cant advances in retrieval mechanisms. Represen-
tative examples include M3DocRAG (Cho et al.,
2024), which visualizes PDF pages for end-to-end
visual retrieval; MHier-RAG (Gong et al., 2025),
which enables fine-grained localization within long
documents via intra- and inter-page structural in-
dices; and MoLoRAG (Wu et al., 2025) and
MegaRAG (Hsiao et al., 2025), which utilize page
graphs or multi-modal knowledge graphs to model
cross-page logical and entity relationships. Further-
more, ViDoRAG (Wang et al., 2025) attempts to
move beyond fixed Top-K constraints through dy-
namic retrieval budget allocation, while VisDoM-
RAG (Suri et al., 2025) models textual and visual
evidence in parallel with consistency-constrained
fusion. These works propel multi-modal document
understanding toward joint visual-textual-structural
modeling. However, existing multi-modal RAG
methods typically rely on static retrieval and fixed
dual-channel workflows. This lack of dynamic rout-
ing causesBreadth Insufficiency: failing to recall
relevant information located elsewhere. Our frame-
work addresses this via aRetrieval Scalingmodule.
Instead of blind Top-K expansion (Taguchi et al.,
2025), it decomposes the query into attributes and
parallelizes retrieval from multiple perspectives,
directing outward expansion specifically around
missing evidence.
Multi-Agent Systems.Multi-agent systems
have shown immense potential in handling rea-
soning over complex long documents (Zhao et al.,
2024; Yao et al., 2022; Zhang et al., 2024). These
systems typically utilize specialized sub-agents
with distinct divisions of labor to iteratively un-
derstand document content. For instance, MDocA-
gent (Han et al., 2025) enhances the mining of
text-image details through multi-role collaboration,
while DocAgent (Sun et al., 2025) improves long-
document processing capabilities via outline nav-
igation, interactive reading, reviewer agents, andmemory banks. MACT (Yu et al., 2025) introduces
procedural test-time scaling, and ViDoRAG (Wang
et al., 2025) adopts an iterative seeker-inspector-
answer collaborative workflow. While multi-agent
systems mitigate single-model limitations in com-
plex DocQA, their predetermined workflows and
fixed budgets struggle to dynamically allocate rea-
soning depth for fine-grained visual details. This
rigidity causesDepth Insufficiency: incomplete
evidence resolution. Our framework addresses this
via aVerifier agent-driven dynamic routing loop
centered around anEvidence Bank.
3 Methodology
In this section, we present D2-ScaleAgent, a Dual-
Dimensional Scaling framework that reformulates
long-document understanding from a workflow
into an on-demand compute routing problem, as
illustrated in Figure 2.
3.1 Problem Formulation
We formulate long document understanding as an
evidence-driven exploration. The system maintains
a globalEvidence Bank ( Bt)serving as the unified
epistemic state at global system stept:
Bt={Epage
t, Eregion
t , Eatomic
t , scomp
t, gt}(1)
These five variables respectively track the globally
accumulated page-level, region-level, and atomic-
level evidence, quantify the current evidence com-
pleteness score ( scomp
t), and explicitly record the
current evidence gap ( gt). Guided by this unified
state, the system dynamically allocates computa-
tional budgets to resolve the two primary failure
modes in long-document reasoning: first intro-
ducing the attribute-guided retrieval scaling to re-
solve breadth insufficiency (Section 3.2), and subse-
quently presenting the gap-aware reasoning scaling
to resolve depth insufficiency (Section 3.3).
3.2 Attribute-Guided Retrieval Scaling
Traditional static Top- Kretrieval paradigms are
ill-equipped for complex visually-rich long doc-
uments, as the scope of required evidence varies
drastically across queries. To address this, we for-
malize retrieval scaling as a dynamic, three-stage
process aimed at adaptively expanding the search
boundary until it autonomously converges to a high-
value evidence set. Given a current query intent
qand a page space P, the process operates as fol-
lows:
3

Figure 2: Overview of D2-ScaleAgent. (1) The system first retrieves a candidate page pool and initializes the
Evidence Bank. (2) A Verifier agent-driven loop then routes computation outward for retrieval scaling or inward for
reasoning scaling based on the current evidence gap. (3) Multi-granularity evidence is continuously accumulated
into the Evidence Bank. (4) Once logical closure is achieved, the system generates the final answer from the
saturated evidence state.
Query Attribute Decomposition.In visually-
rich contexts, an information need is rarely singular.
Instead of treating qas a monolithic input, we lever-
age an LLM (whose prompt template is detailed in
Appendix E) to decompose it into a weighted set
of multi-perspective attribute queries:
Q={(q 0, w0),(q 1, w1), . . . ,(q M, wM)}(2)
where q0is the core intent, qmare derived attribute
perspectives, and wmrepresents the confidence
weight (PM
m=0wm= 1).
Multi-round Candidate Accumulation.To
gather complementary evidence, the system exe-
cutes retrieval for the decomposed queries in Q. At
any internal retrieval step j(where 0≤j≤M ),
the system processes the attribute query qjto re-
trieve a local candidate page set R(j). It then up-
dates a monotonically expanding global candidate
page pool, defined as C(j)=Sj
m=0R(m), which
aggregates results from all queries processed up to
stepj. Since raw retrieval scores across different
attribute perspectives are incomparable, we calcu-
late a rank-based weighted fusion score for each
accumulated candidate pagec:
Sj(c) =jX
m=01[c∈ R(m)]·wm
κ+r(m)(c)(3)where r(m)(c)is the local rank of page cin the
m-th query’s results, wmis the query’s confidence
weight, and κis a smoothing constant. This mech-
anism prioritizes high-value evidence consistently
supported across multiple perspectives.
Adaptive Pruning and Convergence.To bal-
ance the expanding boundary with downstream
reasoning costs, we introduce a confidence-aware
pruning mechanism. We define the maximum ac-
cumulated score Smax
jand extract the high-value
evidence set for the current round using a relative
thresholdα∈(0,1):
Eret
j={c∈ C(j)|Sj(c)≥αSmax
j}(4)
To autonomously determine if the retrieval scaling
has saturated, we quantify the cross-round stability
of this evidence set:
Stab j=|Eret
j∩ Eret
j−1|
|Eret
j|(5)
When Stab jconsistently exceeds a predefined
threshold τ, the system infers that the retrieval fron-
tier has converged. It dynamically terminates the
internal expansion and designates this set as the
global retrieval output for the global system step t:
Eret
t← Eret
j. In summary, the terminus of retrieval
scaling is neither a fixed number of rounds nor a
4

rigid Top- Kbudget, but an adaptive stopping state
achieved when the evidence set stabilizes under
multi-perspective scrutiny.
3.3 Gap-Aware Reasoning Scaling
Even when relevant pages are successfully re-
trieved ( Eret
t), LVLMs frequently default to coarse
semantic matching, struggling to parse precise nu-
merical values or complex layouts from visually-
rich contexts. To address this depth insufficiency,
we introduce a Reasoning-Side Scaling mecha-
nism. Rather than indiscriminately forcing pages
through a fixed workflow, this mechanism dynami-
cally invokes a toolkit of cost-stratified cognitive
sub-agents based on explicit evidence gaps, exe-
cuting a coarse-to-fine transition from macroscopic
intuition to auditable atomic facts. The sub-agents’
prompt templates are detailed in Appendix E. The
reasoning toolkit comprises three hierarchical lev-
els of sub-agents, which are adaptively selected by
the downstream routing mechanism:
Global Surveyor: Low-cost Page Understand-
ing.The Surveyor executes a macroscopic scan
over the candidate set Eret
tto establish a directional
semantic prior, preventing the premature alloca-
tion of expensive compute to irrelevant details. It
outputs incremental page-level evidenceepage
t:
epage
t=fsur(q,Eret
t)(6)
Region Locator: Medium-cost Region Focus-
ing.Acting as a dimensionality reduction mecha-
nism, the Locator isolates critical regions from the
broad candidate set and generates structured sum-
maries (e.g., identifying specific tables), outputting
a highly relevant page subset Ekey
tand region-level
evidenceeregion
t :
(Ekey
t, eregion
t ) =f loc(q,Eret
t)(7)
Fine-grained Extractor: High-cost Atomic
Extraction.For the identified critical regions, this
module undertakes high-resolution precision read-
ing based on an explicit extraction specification
(ψt) dynamically generated from current evidence
gaps. It extracts specific, verifiable atomic facts
eatomic
t :
eatomic
t =fext(q,Eret
t, ψt)(8)
Crucially, the multi-granularity evidence ex-
tracted by these adaptive operations is not held in
isolated caches. The incremental evidence ( e∗
t) iscontinuously appended to the globally accumulated
sets (E∗
t):
E∗
t+1=E∗
t∪ {e∗
t}(9)
Consequently, the global epistemic state of the Evi-
dence Bank is explicitly and incrementally updated
at each reasoning step:
Bt+1= Update(B t, epage
t, eregion
t, eatomic
t )(10)
This evidence-driven update mechanism formalizes
the epistemic state, providing the global Verifier
with a unified basis to drive subsequent inward/out-
ward routing decisions.
3.4 Verifier Agent-Driven Routing
The orchestration of this evidence-driven computa-
tion is governed by the Verifier, which functions not
merely as a posterior checker, but as the system’s
explicit stopping and expansion trigger. The Veri-
fier’s prompt template is detailed in Appendix E. At
each inference step t, the Verifier assesses the epis-
temic completeness of the Evidence Bank, identify-
ing specific logical discontinuities or cross-modal
contradictions:
(scomp
t, gt) =f ver(q,B t)(11)
These evaluations are subsequently written
back to update the global state: Bt←
Updatever(Bt, scomp
t, gt). Based on this explicit
gapgt, the system triggers one of two distinct rout-
ing dimensions:
1. Inward Digging (Addressing Depth In-
sufficiency).When the critical source pages are
located but parsed localized facts remain insuffi-
cient, the routing mechanism dynamically allocates
a specific cognitive operation otfrom the reasoning
toolkit (Section 3.3) to extract deeper incremental
evidence:
ifgt∈ Gdepth, o t←Reasoning-Scale(g t,Bt),
ot∈ {f sur, floc, fext}
(12)
2. Outward Expansion (Addressing Breadth
Insufficiency).If the Verifier reveals an unbridge-
able contextual void indicating missing founda-
tional source evidence, the system translates the
explicit gap gtinto a novel retrieval intent to dy-
namically re-expand the candidate boundary via
the retrieval toolkit (Section 3.2):
ifgt∈ Gbreadth , qnew
t←Φ(g t, q),
Eret
t←Retrieval-Scale(qnew
t)
(13)
5

The closed-loop routing mechanism operates
continuously until a rigorous termination condition
is met:
Stop ifscomp
t≥δandg t=∅(14)
where δdefines the strict evidence closure thresh-
old. Once this condition is satisfied, the system
decisively breaks the iterative loop and generates
the final answer based on the logically saturated
Evidence Bank:
a=f ans(q,B t)(15)
By conditioning generation strictly on a verified,
logically closed evidence state rather than forcing
an output under incomplete information, the sys-
tem avoids hallucinating unsupported conclusions.
Further details are provided in Appendix A.
4 Experiments
4.1 Experiment Setup
Datasets: To cover a wide range of real-world ap-
plication scenarios, we selected six multi-modal
long-document benchmarks: MMLongBench-Doc
(Ma et al., 2024), LongDocURL (Deng et al., 2025),
PaperTab (Hui et al., 2024), FetaTab (Hui et al.,
2024), ViDoSeek (Wang et al., 2025) and UniDoc-
Bench (Peng et al., 2025). These datasets span
both open and domain-specific fields, encompass-
ing varying text lengths and rich visual elements to
comprehensively test models’ multi-modal under-
standing capabilities in long contexts.
Baselines: D2-ScaleAgent is benchmarked
against 3 advanced open-source baselines catego-
rized into two mainstream paradigms: (1)Multi-
modal RAG: MoLoRAG (Wu et al., 2025). (2)
Multi-agent Systems: This includes MDocAgent
(Han et al., 2025) and ViDoRAG (Wang et al.,
2025), representing advanced frameworks that op-
erate on fixed workflows.
More implementation details are provided in Ap-
pendix D.
4.2 Main Results
(1) End-to-End QA Performance.Table 1 reports
the end-to-end QA accuracy of D2-ScaleAgent
and all baseline methods on the six benchmarks.
Overall, D2-ScaleAgent achieves the best aver-
age performance among all methods, consistently
outperforming both multi-modal RAG and fixed-
workflow multi-agent baselines across all evalu-
ated models (e.g., 63.7 vs. MDocAgent’s 58.3on GPT-4o). Notably, VQA, which processes the
full document image directly without relying on
explicit retrieval or iterative reasoning stages, out-
performs several retrieval-based baselines in mul-
tiple settings, suggesting that stronger foundation
models can substantially narrow the gap between
holistic VQA and fixed retrieval-and-reasoning
pipelines. This trend is particularly evident with
Gemini-3-flash-preview: VQA even outperforms
D2-ScaleAgent on MMLongBench-Doc (Ma et al.,
2024) and LongDocURL (Deng et al., 2025). We
emphasize, however, that the advantage of D2-
ScaleAgent does not lie in uniformly outperform-
ing VQA in every setting, but in mitigating the
brittleness of fixed retrieval and reasoning work-
flows under evidence-insufficient scenarios. These
gains are particularly prominent on benchmarks
with complex layouts and dispersed evidence.
Figure 3: Retrieval performance comparison on
MMLongBench-Doc and LongDocURL datasets.
(2) Retrieval Performance.We further evalu-
ate retrieval quality on MMLongBench-Doc (Ma
et al., 2024) and LongDocURL (Deng et al., 2025).
As shown in Figure 3, the baseline results are
taken from MoLoRAG (Wu et al., 2025), and
D2-ScaleAgent consistently improves Recall, Pre-
cision, nDCG, and MRR over strong baselines.
These gains indicate that our method not only re-
trieves more relevant evidence, but also ranks it
more effectively. The improvements mainly come
6

Table 1: Performance comparison of different methods across multiple benchmarks. VQA takes the entire document
as image input.
Method MMLongBench-Doc LongDocURL PaperTab FetaTab ViDoSeek UniDoc-Bench Avg
GPT-4o
VQA 43.7 51.1 31.3 75.7 78.8 56.1 56.1
ViDoRAG (Wang et al., 2025) 44.4 50.8 30.6 66.1 70.7 58.0 53.4
MoLoRAG (Wu et al., 2025) 40.7 55.5 29.3 64.7 74.1 57.1 53.6
MDocAgent (Han et al., 2025) 34.7 50.9 45.5 79.8 77.4 61.2 58.3
D2-ScaleAgent52.0 56.0 47.5 82.8 81.8 62.2 63.7
Gemini-3-flash-preview
VQA 65.0 65.4 58.6 72.7 86.8 60.6 68.2
ViDoRAG (Wang et al., 2025)65.360.6 56.5 80.8 83.8 62.9 68.3
MoLoRAG (Wu et al., 2025) 49.2 59.1 48.5 67.7 76.3 66.3 61.2
MDocAgent (Han et al., 2025) 42.6 45.0 54.5 80.8 76.3 64.9 60.7
D2-ScaleAgent 63.064.4 62.6 84.8 85.8 73.5 72.4
Qwen2.5-VL-7B-Instruct
VQA 32.9 44.6 34.3 61.6 70.7 35.7 46.6
ViDoRAG (Wang et al., 2025) 30.5 43.5 33.0 48.0 79.5 38.9 45.6
MoLoRAG (Wu et al., 2025) 38.1 48.6 30.3 55.6 69.9 46.9 48.2
MDocAgent (Han et al., 2025) 32.2 43.9 32.3 66.7 68.851.049.2
D2-ScaleAgent42.1 49.4 43.4 70.7 81.847.055.7
Qwen3-VL-8B-Instruct
VQA 49.0 56.8 49.5 69.0 82.8 53.0 60.0
ViDoRAG (Wang et al., 2025) 44.7 53.0 54.6 68.4 77.6 51.0 58.2
MoLoRAG (Wu et al., 2025) 46.0 55.0 36.4 57.6 72.0 51.0 53.0
MDocAgent (Han et al., 2025) 40.7 52.2 45.5 73.7 76.3 57.1 57.6
D2-ScaleAgent51.1 58.1 55.5 76.7 89.9 63.2 65.8
from the retrieval-side scaling strategy, which per-
forms query attribute decomposition and iterative
evidence expansion rather than relying on a single
static Top-K retrieval step. This allows the system
to better handle questions whose supporting evi-
dence is distributed across multiple pages or only
partially captured by the initial query. Overall, the
results confirm the importance of coupling retrieval
expansion with downstream reasoning.
Table 2: Ablation study of different components on the
MMLongBench-Doc dataset.
Retrieval ScalingReasoning Scaling Verifier-
Driven LoopMMLongBench-Doc
Surveyor Locator Extractor
Evidence✓ ✓ ✓ ✓54.9
✓ ✓ ✓ ✓ ✓52.0
×✓ ✓ ✓ ✓46.8
✓×✓ ✓ ✓46.5
✓ ✓×✓ ✓47.1
✓ ✓ ✓×✓47.5
✓ ✓ ✓ ✓×44.1
✓GPT-4o Answer Directly 45.0
4.3 Ablation Studies
As shown in Table 2, we ablate D2-ScaleAgent on
MMLongBench-Doc to quantify each component’s
contribution. The full model (52.0) outperforms
the naive GPT-4o baseline (45.0) and approaches
the upper bound of 54.9, which is achieved by
replacing the retrieval module with ground-truthevidence. Crucially, disabling the Verifier agent-
driven loop causes the steepest accuracy drop (to
44.1), underscoring the critical role of rigorous
evidence verification and closed-loop routing. Sim-
ilarly, discarding the Retrieval-Side scaling (46.8)
or removing any of the inward-scaling reasoning
agents—Surveyor (46.5), Locator (47.1), and Ex-
tractor (47.5)—consistently degrades performance.
These findings demonstrate that static workflows
are insufficient; dynamic retrieval expansion, multi-
granularity visual parsing, and rigorous evidence
verification must be coordinated to resolve complex
long document understanding tasks. Additional ab-
lation results are provided in Appendix C.
4.4 Compute-cost analysis
We have profiled D2-ScaleAgent on two repre-
sentative benchmarks, MMLongBench-Doc and
ViDoSeek. As shown in Table 3, computa-
tional costs dynamically scale with query diffi-
culty: MMLongBench-Doc requires more verifi-
cation and routing (21.4K tokens, 16.22s) due to
its structural complexity and dispersed evidence,
whereas ViDoSeek exhibits lower overhead (15.9K
tokens, 11.89s) with fewer iterative corrections.
D2-ScaleAgent adaptively allocates computation
based on evidence insufficiency. All reported effi-
7

Figure 4: Case study of D2-ScaleAgent. Given a cross-page compositional question comparing population sizes,
both baseline methods fail due to either breadth or depth insufficiency. In contrast, D2-ScaleAgent, guided by the
Verifier agent-driven dynamic routing loop, identifies that the initial evidence chain is incomplete, then triggers
Attribute-Guided Retrieval Scaling to locate the missing page and Gap-Aware Reasoning Scaling to extract the
required fine-grained numerical evidence.
ciency metrics are averaged per query, and latency
is measured end-to-end using the GPT-4o API.
Table 3: Compute-cost analysis and profiling of D2-
ScaleAgent on MMLongBench-Doc and ViDoSeek us-
ing GPT-4o.
Metric MMLongBench-Doc ViDoSeek
Accuracy 52.0 81.8
Retrieval rounds 1.33 1.12
Reasoning calls (S/L/E) 1.51 / 1.12 / 0.63 1.75 / 0.34 / 0.47
Verifier calls 1.76 1.14
Routing-agent calls 5.02 3.70
Tokens 21.4K 15.9K
E2E latency 16.22 s 11.89 s
4.5 Case Study
Figure 4 presents an example of cross-page ques-
tion answering. The question cannot be resolved
from a single page, as it requires reasoning over
the 2018 group distribution on Page 3 and the de-
mographic composition table on Page 22. Multi-
modal RAG successfully retrieves both pages, but
fails to perform the required cross-page composi-
tion; instead, it compares local ratios and arrives
at the correct answer only by chance. Fixed Multi-
agent exhibits stronger local reasoning ability, yet
remains confined to Page 3 and therefore misses the
critical attribute evidence on Page 22. In contrast,D2-ScaleAgent identifies the incomplete evidence
chain through the Verifier, triggers retrieval scal-
ing to acquire the missing page, and applies fine-
grained reasoning to complete the population-level
comparison, thereby producing the correct answer.
This case highlights that D2-ScaleAgent does not
merely benefit from stronger reasoning or broader
retrieval in isolation, but from its Verifier agent-
driven coordination of the two, which enables the
system to adaptively close the evidence gap and
achieve logical closure over a distributed evidence
chain.
5 Conclusions
In this paper, we tackle the long document under-
standing task by addressing the limitations of prior
methods that rely on fixed workflows and suffer
from evidence insufficiency. By introducing a dual-
dimensional scaling paradigm, our D2-ScaleAgent
utilizes a Verifier agent-driven dynamic routing
loop centered on a continuously updated evidence
bank to perform both outward retrieval scaling
and inward reasoning scaling. Extensive experi-
ments demonstrate that D2-ScaleAgent achieves
logical closure over evidence chains and delivers
SOTA performance on complex benchmarks like
MMLongBench-Doc, LongDocURL, etc.
8

Limitations
While D2-ScaleAgent effectively mitigates evi-
dence insufficiency through dual-dimensional scal-
ing, this paradigm inherently trades computational
efficiency for reasoning accuracy. A primary limi-
tation of our framework is the inevitable increase
in inference latency and computational overhead.
The dynamic routing mechanism, which necessi-
tates multi-round attribute-guided retrieval (out-
ward scaling) and the on-demand invocation of
cost-stratified cognitive sub-agents (inward scal-
ing), results in execution times significantly longer
than standard single-pass inference models. Conse-
quently, the current architecture is computationally
intensive and may be suboptimal for real-time ap-
plications or severely latency-constrained deploy-
ment scenarios. A comprehensive profiling of this
computational efficiency trade-off is detailed in
subsection 4.4.
Ethical Considerations
The datasets utilized in this work do not contain
any personally identifiable or sensitive information,
as all materials were sourced exclusively from pub-
licly available domains. Furthermore, the curation,
processing, and refinement of the data were con-
ducted in strict adherence to applicable copyright
laws and intellectual property guidelines.
The Use of AI assistants
AI assistants (ChatGPT) are used to correct poten-
tial grammatical inaccuracies in the manuscript. AI
assistants do not participate in research ideation.
References
Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang,
Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou,
and Jingren Zhou. 2023. Qwen-vl: A versatile vision-
language model for understanding, localization, text
reading, and beyond.Preprint, arXiv:2308.12966.
Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen,
Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei
Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhi-
fang Guo, Qidong Huang, Jie Huang, Fei Huang,
Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng
Li, and 45 others. 2025. Qwen3-vl technical report.
Preprint, arXiv:2511.21631.
Sebastian Borgeaud, Arthur Mensch, Jordan Hoff-
mann, Trevor Cai, Eliza Rutherford, Katie Milli-
can, George Bm Van Den Driessche, Jean-Baptiste
Lespiau, Bogdan Damoc, Aidan Clark, and 1 others.2022. Improving language models by retrieving from
trillions of tokens. InInternational conference on
machine learning, pages 2206–2240. PMLR.
Bingsen Chen, Shenji Wan, Xi Ye, and Chen Zhao.
2025. Inter-passage verification for multi-evidence
multi-answer qa. InFindings of the Association for
Computational Linguistics: ACL 2025, pages 6811–
6829.
Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie
He, and Mohit Bansal. 2024. M3docrag: Multi-
modal retrieval is what you need for multi-page
multi-document understanding.arXiv preprint
arXiv:2411.04952.
Zheng Chu, Huiming Fan, Jingchang Chen, Qianyu
Wang, Mingda Yang, Jiafeng Liang, Zhongjie Wang,
Hao Li, Guo Tang, Ming Liu, and 1 others. 2025.
Self-critique guided iterative reasoning for multi-hop
question answering. InFindings of the Association
for Computational Linguistics: ACL 2025, pages
2415–2438.
Chao Deng, Jiale Yuan, Pi Bu, Peijie Wang, Zhong-
Zhi Li, Jian Xu, Xiao-Hui Li, Yuan Gao, Jun Song,
Bo Zheng, and 1 others. 2025. Longdocurl: a com-
prehensive multimodal long document benchmark
integrating understanding, reasoning, and locating.
InProceedings of the 63rd Annual Meeting of the
Association for Computational Linguistics (Volume
1: Long Papers), pages 1135–1159.
Yihao Ding, Zhe Huang, Runlin Wang, YanHang Zhang,
Xianru Chen, Yuzhong Ma, Hyunsuk Chung, and
Soyeon Caren Han. 2022. V-doc: Visual questions
answers with documents. InProceedings of the
IEEE/CVF conference on computer vision and pat-
tern recognition, pages 21492–21498.
Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Om-
rani, Gautier Viaud, Céline Hudelot, and Pierre
Colombo. 2024. Colpali: Efficient document re-
trieval with vision language models.arXiv preprint
arXiv:2407.01449.
Yunfan Gao, Yun Xiong, Yijie Zhong, Yuxi Bi, Ming
Xue, and Haofen Wang. 2025. Synergizing rag and
reasoning: A systematic review.arXiv preprint
arXiv:2504.15909.
Ziyu Gong, Chengcheng Mai, and Yihua Huang.
2025. Mhier-rag: Multi-modal rag for visual-
rich document question-answering via hierarchical
and multi-granularity reasoning.arXiv preprint
arXiv:2508.00579.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasu-
pat, and Mingwei Chang. 2020. Retrieval augmented
language model pre-training. InInternational confer-
ence on machine learning, pages 3929–3938. PMLR.
Siwei Han, Peng Xia, Ruiyi Zhang, Tong Sun, Yun Li,
Hongtu Zhu, and Huaxiu Yao. 2025. Mdocagent: A
multi-modal multi-agent framework for document
understanding.arXiv preprint arXiv:2503.13964.
9

Chi-Hsiang Hsiao, Yi-Cheng Wang, Tzung-Sheng
Lin, Yi-Ren Yeh, and Chu-Song Chen. 2025.
Megarag: Multimodal knowledge graph-based re-
trieval augmented generation.arXiv preprint
arXiv:2512.20626.
Yulong Hui, Yao Lu, and Huanchen Zhang. 2024. Uda:
A benchmark suite for retrieval augmented generation
in real-world document analysis.Advances in Neural
Information Processing Systems, 37:67200–67217.
Wenjun Ke, Yifan Zheng, Yining Li, Hengyuan Xu,
Dong Nie, Peng Wang, and Yao He. 2025. Large
language models in document intelligence: A com-
prehensive survey, recent advances, challenges, and
future trends.ACM Transactions on Information Sys-
tems, 44(1):1–64.
Kenji Kobayashi and Joseph W Kable. 2024. Neu-
ral mechanisms of information seeking.Neuron,
112(11):1741–1756.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi.
2023. Blip-2: Bootstrapping language-image pre-
training with frozen image encoders and large lan-
guage models. InInternational conference on ma-
chine learning, pages 19730–19742. PMLR.
Xiaonan Li, Changtai Zhu, Linyang Li, Zhangyue Yin,
Tianxiang Sun, and Xipeng Qiu. 2024. Llatrieval:
Llm-verified retrieval for verifiable generation. In
Proceedings of the 2024 Conference of the North
American Chapter of the Association for Computa-
tional Linguistics: Human Language Technologies
(Volume 1: Long Papers), pages 5453–5471.
Yen-Ting Lin and Yun-Nung Chen. 2023. Llm-eval:
Unified multi-dimensional automatic evaluation for
open-domain conversations with large language mod-
els. InProceedings of the 5th Workshop on NLP
for Conversational AI (NLP4ConvAI 2023), pages
47–58.
Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan
Zhang, Sheng Shen, and Yong Jae Lee. 2024. Llava-
next: Improved reasoning, ocr, and world knowledge.
Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen,
Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma,
Xiaoyi Dong, and 1 others. 2024. Mmlongbench-doc:
Benchmarking long-context document understanding
with visualizations.Advances in Neural Information
Processing Systems, 37:95963–96010.
OpenAI. 2024. Gpt-4o system card.arXiv preprint
arXiv:2410.21276.Xiangyu Peng, Can Qin, Zeyuan Chen, Ran Xu, Caim-
ing Xiong, and Chien-Sheng Wu. 2025. Unidoc-
bench: A unified benchmark for document-centric
multimodal rag.arXiv preprint arXiv:2510.03663.
Peter Pirolli and Stuart Card. 1995. Information forag-
ing in information access environments. InProceed-
ings of the SIGCHI conference on Human factors in
computing systems, pages 51–58.
Deniz Qian, Hung-Ting Chen, and Eunsol Choi. 2026.
Rvr: Retrieve-verify-retrieve for comprehensive ques-
tion answering.arXiv preprint arXiv:2602.18425.
Aymeric Roucher, Albert Villanova del Moral, Thomas
Wolf, Leandro von Werra, and Erik Kaunismäki.
2025. ‘smolagents‘: a smol library to build
great agentic systems. https://github.com/
huggingface/smolagents.
Connor Shorten, Augustas Skaburskas, Daniel M Jones,
Charles Pierse, Roberto Esposito, John Trengrove,
Etienne Dilocker, and Bob van Luijt. 2026. Irpa-
pers: A visual document benchmark for scientific
retrieval and question answering.arXiv preprint
arXiv:2602.17687.
Aswini Sivakumar, Vijayan Sugumaran, and Yao Qiang.
2026. Rag-x: Systematic diagnosis of retrieval-
augmented generation for medical question answer-
ing.arXiv preprint arXiv:2603.03541.
Li Sun, Liu He, Shuyue Jia, Yangfan He, and Chenyu
You. 2025. Docagent: An agentic framework for
multi-modal long-context document understanding.
InProceedings of the 2025 Conference on Empiri-
cal Methods in Natural Language Processing, pages
17712–17727.
Manan Suri, Puneet Mathur, Franck Dernoncourt,
Kanika Goswami, Ryan A Rossi, and Dinesh
Manocha. 2025. Visdom: Multi-document qa with
visually rich elements using multimodal retrieval-
augmented generation. InProceedings of the 2025
Conference of the Nations of the Americas Chap-
ter of the Association for Computational Linguistics:
Human Language Technologies (Volume 1: Long Pa-
pers), pages 6088–6109.
Chihiro Taguchi, Seiji Maekawa, and Nikita Bhutani.
2025. Efficient context selection for long-context qa:
No tuning, no iteration, just adaptive-k. InProceed-
ings of the 2025 Conference on Empirical Methods in
Natural Language Processing, pages 20116–20141.
Ryota Tanaka, Taichi Iki, Taku Hasegawa, Kyosuke
Nishida, Kuniko Saito, and Jun Suzuki. 2025.
Vdocrag: Retrieval-augmented generation over
visually-rich documents. InProceedings of the Com-
puter Vision and Pattern Recognition Conference,
pages 24827–24837.
Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-
Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan
Schalkwyk, Andrew M Dai, Anja Hauth, Katie Mil-
lican, and 1 others. 2023. Gemini: a family of
10

highly capable multimodal models.arXiv preprint
arXiv:2312.11805.
Qwen Team. 2025. Qwen2.5-vl.
Frans van der Sluis. 2025. Wanting information:
Uncertainty and its reduction through search en-
gagement.Information Processing & Management,
62(2):103890.
Jordy Van Landeghem, Rubèn Tito, Łukasz Borchmann,
Michał Pietruszka, Pawel Joziak, Rafal Powalski,
Dawid Jurkiewicz, Mickaël Coustaty, Bertrand Anck-
aert, Ernest Valveny, and 1 others. 2023. Document
understanding dataset and evaluation (dude). InPro-
ceedings of the IEEE/CVF International Conference
on Computer Vision, pages 19528–19540.
Qiuchen Wang, Ruixue Ding, Zehui Chen, Weiqi Wu,
Shihang Wang, Pengjun Xie, and Feng Zhao. 2025.
Vidorag: Visual document retrieval-augmented gener-
ation via dynamic iterative reasoning agents. InPro-
ceedings of the 2025 Conference on Empirical Meth-
ods in Natural Language Processing, pages 9124–
9145.
Zilong Wang, Yichao Zhou, Wei Wei, Chen-Yu Lee, and
Sandeep Tata. 2023. Vrdu: A benchmark for visually-
rich document understanding. InProceedings of the
29th ACM SIGKDD Conference on Knowledge Dis-
covery and Data Mining, pages 5184–5193.
Xixi Wu, Yanchao Tan, Nan Hou, Ruiyang Zhang, and
Hong Cheng. 2025. Molorag: Bootstrapping doc-
ument understanding via multi-modal logic-aware
retrieval. InProceedings of the 2025 Conference on
Empirical Methods in Natural Language Processing,
pages 14035–14056.
Zilin Xiao, Qi Ma, Mengting Gu, Chun-cheng Jason
Chen, Xintao Chen, Vicente Ordonez, and Vijai Mo-
han. 2025. Metaembed: Scaling multimodal re-
trieval at test-time with flexible late interaction.arXiv
preprint arXiv:2509.18095.
Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu
Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha
Zhang, Wanxiang Che, and 1 others. 2021. Lay-
outlmv2: Multi-modal pre-training for visually-rich
document understanding. InProceedings of the 59th
Annual Meeting of the Association for Computational
Linguistics and the 11th International Joint Confer-
ence on Natural Language Processing (Volume 1:
Long Papers), pages 2579–2591.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik R Narasimhan, and Yuan Cao. 2022.
React: Synergizing reasoning and acting in language
models. InThe eleventh international conference on
learning representations.
Xinlei Yu, Chengming Xu, Zhangquan Chen, Yudong
Zhang, Shilin Lu, Cheng Yang, Jiangning Zhang,
Shuicheng Yan, and Xiaobin Hu. 2025. Visual doc-
ument understanding and reasoning: A multi-agent
collaboration framework with agent-wise adaptive
test-time scaling.arXiv preprint arXiv:2508.03404.Yusen Zhang, Ruoxi Sun, Yanfei Chen, Tomas Pfis-
ter, Rui Zhang, and Sercan Ö Arık. 2024. Chain
of agents: Large language models collaborating on
long-context tasks.Advances in Neural Information
Processing Systems, 37:132208–132237.
Jun Zhao, Can Zu, Xu Hao, Yi Lu, Wei He, Yiwen
Ding, Tao Gui, Qi Zhang, and Xuan-Jing Huang.
2024. Longagent: achieving question answering for
128k-token-long documents through multi-agent col-
laboration. InProceedings of the 2024 Conference on
Empirical Methods in Natural Language Processing,
pages 16310–16324.
Zhengyi Zhao, Shubo Zhang, Zezhong Wang, Huimin
Wang, Yutian Zhao, Bin Liang, Yefeng Zheng,
Binyang Li, Kam-Fai Wong, and Xian Wu. 2025.
T2: An adaptive test-time scaling strategy for con-
textual question answering. InProceedings of the
2025 Conference on Empirical Methods in Natural
Language Processing, pages 3731–3756.
Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and
Mohamed Elhoseiny. 2023. Minigpt-4: Enhancing
vision-language understanding with advanced large
language models.arXiv preprint arXiv:2304.10592.
Yongshuo Zong, Ismail Elezi, Yongxin Yang, Jiankang
Deng, and Timothy Hospedales. 2024. Long-context
vision large language models: Empirical insights and
a baseline. InFirst Workshop on Long-Context Foun-
dation Models@ ICML 2024.
11

A Detailed Algorithmic Workflow
Algorithm 1Process ofD2-ScaleAgent
1:Initialize queryq, document spaceP,
Evidence BankB 0;
2:Set stept= 0;
3:Get initial evidenceEret
0via
Retrieval-Scale(q,P)
4:UpdateB 0with initial evidenceEret
0
5:while true do
6:# Verifier assessment
7:Assess completeness and gap:
(scomp
t, gt)←f ver(q,B t)
8:ifscomp
t≥δandg t=∅then
9:break
10:end if
11:# Inward / Outward routing
12:ifg t∈ Gdepth then
13:Route inward: allocate reasoning
operatoro t∈ {f sur, floc, fext}
14:Extract incremental evidencee tvia
reasoning operation
15:else ifg t∈ Gbreadth then
16:Route outward: derive new query
qnew
t←Φ(g t, q)
17:Extract evidencee tvia
Retrieval-Scale(qnew
t)
18:end if
19: # Evidence integration and state transition
20:B t+1←Update(B t, et)
21:t←t+ 1
22:end while
23:returnFinal answera←f ans(q,B t)
The inference process of D2-ScaleAgent is sum-
marized in Algorithm 1. Unlike traditional meth-
ods that rely on fixed retrieval budgets (e.g., rigid
Top-K selection), this algorithm operates through
an adaptive, Verifier agent-driven closed-loop. By
maintaining a global Evidence Bank ( Bt), the algo-
rithm dynamically evaluates the current epistemic
completeness ( scomp
t) and identifies specific evi-
dence gaps ( gt). Leveraging a dual-dimensional
routing mechanism, the traversal is precisely di-
rected: it routes outward (retrieval scaling) to ex-
pand the search space when encountering breadth
insufficiency, and routes inward (reasoning scaling)
to extract fine-grained visual details when facing
depth insufficiency. This on-demand allocation
ensures scalability and avoids the computational
waste of processing irrelevant pages. The itera-tive execution strictly terminates only when logical
chain closure is achieved ( scomp
t≥δandgt=∅).
The final output is a highly accurate answer, di-
rectly grounded in a rigorously verified and suffi-
cient set of multi-granularity evidence.
B Case Study for Execution Traces
As shown in Figure 5, we illustrate the execution
traces of D2-ScaleAgent with a cross-page compo-
sitional QA example. Given the query, the system
first performs initial retrieval and identifies Page 3
as the starting evidence source. The Surveyor then
extracts partial evidence related to party distribu-
tion, while the Verifier determines that the evidence
chain is still incomplete because the key attributes
“Hispanic” and “Male” are missing. Guided by this
gap signal, D2-ScaleAgent activates Retrieval-side
scaling to expand the search boundary and retrieve
Page 22. It then invokes Reasoning-side scaling,
where the Locator identifies the relevant demo-
graphic table and the Extractor obtains the missing
atomic evidence required for cross-page compo-
sition. As the retrieved and extracted evidence is
continuously written into the Evidence Bank, the
Verifier re-evaluates the accumulated support un-
til logical closure is achieved. The system then
performs the final compositional comparison and
produces the correct answer. This case highlights
how D2-ScaleAgent dynamically coordinates re-
trieval expansion and fine-grained reasoning under
Verifier guidance to resolve evidence insufficiency
in long-document understanding.
C Extended Ablation Study on Diverse
Benchmarks
To further validate the effectiveness of each compo-
nent across different document types and evidence
distributions, we extend our ablation study to two
additional benchmarks: PaperTab (a table-centric
benchmark) and UniDoc-Bench (a comprehensive
multi-modal benchmark). Results are summarized
in Table 4.
Note: The oracle-evidence result is unavailable
for PaperTab because it does not provide ground-
truth evidence-page annotations.
D Details
D.1 Implementation Details
For visual embedding, we utilize ColQwen2-v1.0
(Faysse et al., 2024). The D2-ScaleAgent is de-
signed with a plug-and-play architecture for foun-
12

Figure 5: A case study illustrating the dual-dimensional scaling behavior of D2-ScaleAgent. Starting from an
incomplete initial retrieval result, the agent identifies missing evidence through Verifier feedback and dynamically
scales outward to retrieve additional pages, and inward to extract fine-grained attributes from them. The execution
trace shows how the agent progressively gathers cross-page evidence from the evidence bank and finally derives the
correct answer through compositional reasoning.
Table 4: Extended ablation study of different compo-
nents on PaperTab and UniDoc-Bench datasets.
Retrieval
ScalingReasoning Scaling Verifier-
Driven LoopPaperTab UniDoc-Bench
Surveyor Locator Extractor
Evidence✓ ✓ ✓ ✓X 65.1
✓ ✓ ✓ ✓ ✓47.5 62.2
×✓ ✓ ✓ ✓42.4 59.2
✓×✓ ✓ ✓43.3 59.4
✓ ✓×✓ ✓44.3 60.2
✓ ✓ ✓×✓45.6 61.3
✓ ✓ ✓ ✓×42.9 58.7
✓GPT-4o Answer Directly 39.1 57.4
dation models; in our primary experiments, we
default to GPT-4o (OpenAI, 2024) as the core
reasoning engine, which powers all functional
modules including the Surveyor, Locator, Extrac-
tor, and Verifier. Additional foundational models
tested include Gemini-3-flash-preview (Team et al.,
2023), Qwen2.5-VL-7B-Instruct (Team, 2025), and
Qwen3-VL-8B-Instruct (Bai et al., 2025). The en-
tire agentic routing is implemented using the smo-
lagents framework (Roucher et al., 2025).
D.2 Metrics Details
We evaluate performance across generation and re-
trieval dimensions. For MMLongBench-Doc (Ma
et al., 2024) and LongDocURL (Deng et al., 2025),
we employ GPT-4o (OpenAI, 2024) to extract an-
swers and evaluate using Exact Match (EM) andAccuracy. For PaperTab (Hui et al., 2024), FetaTab
(Hui et al., 2024), and UniDoc-Bench (Peng et al.,
2025), we utilize GPT-4o for Binary Correctness
(0/1) scoring. For ViDoSeek (Wang et al., 2025),
we utilize GPT-4o as the judge model to score the
generated final answers against the ground truth
reference on a scale of 1 to 5; an output achieving
a score of 4 or higher is considered correct (Lin
and Chen, 2023). To assess retrieval quality, we
report Recall, Precision, Normalized Discounted
Cumulative Gain (NDCG), and Mean Reciprocal
Rank (MRR).
D.3 Hyperparameters
This section provides the default values of the hy-
perparameters used in D2-ScaleAgent throughout
our experiments, as summarized in Table 5.
E Prompt Templates for D2-ScaleAgent
This section details the specific prompt templates
designed to drive the dual-dimensional scaling pro-
cess in D2-ScaleAgent. Below, we present the
complete prompt for the Query Attribute Decom-
position, Global Surveyor, Region Locator, Fine-
grained Extractor, and the Verifier.
13

The prompt for Query Attribute Decomposition is provided below:
Prompt for Query Attribute Decomposition
# Task Description #
You are an expert query decomposer for visually-rich long document retrieval. In visually-rich
contexts, an information need is rarely singular. Your task is to decompose the user’s question into
a set of multi-perspective attribute queries.
# Guidelines #
•Dynamic Angles:Autonomously define retrieval angles (e.g., entity, visual, comparative,
numerical) based on the question to maximize recall across different document layouts.
•Query Format:Generate short, retrieval-friendly keyword phrases. Avoid pronouns. Include
concrete anchors (entities, time ranges, structural hints).
•Confidence Scores:Assign a confidence score ( conf ) to each query. Scores must be floats
with two decimal places and sum to exactly 1.00.
# Input Format #
{{Question}}
# Response Format #
Please return your answer in JSON format:
{
"queries": [
{"angle": "visual", "query": "...", "conf": 0.40},
{"angle": "...", "query": "...", "conf": 0.60}
]
}
# Example #
Question:From 2014 to 2015, which group had the most significant drop in the percentage of
households claiming their income was falling behind cost of living, and by how much?
Response:
{
"queries": [
{"angle": "visual", "query": "falling behind cost of living 2014 2015
chart figure table", "conf": 0.20},
{"angle": "entity", "query": "households group definitions demographic
segments income categories", "conf": 0.35},
{"angle": "comparative", "query": "largest drop by group 2014 vs 2015
comparison most significant decline", "conf": 0.30},
{"angle": "numerical", "query": "percentage point decrease 2014 2015
by group magnitude difference", "conf": 0.15}
]
}
14

The prompt for the Surveyor is provided below:
Prompt for Global Surveyor
# Task Description #
Automatically analyzes ALL candidate images that were provided at initialization to give a coarse-
grained answer. This agent has been pre-configured with all available images and requires no
additional parameters. Use this agent when you need a direct answer based on all available images.
# Input Format #
{{Question},{Image}}
# Response Format #
Please return your answer in JSON format:
{
"coarse_answer": "Your direct answer to the question"
}
The prompt for the Locator is provided below:
Prompt for Locator
# Task Description #
Analyze all candidate images and select the most relevant images with query-relevant summaries.
This agent examines all candidate images, selects relevant images, and provides detailed summaries
for the selected images explaining their relevance to the question. Use this agent when you need to
identify which images are most relevant and understand their content.
# Input Format #
{{Question},{Image}}
# Response Format #
Please return your answer in JSON format:
{
"choice": List[int],
"outline_level": "Provide summaries for the selected images, explaining what query-
relevant information each contains"
}
# Example #
Example 1: Question: Who is the person playing a musical instrument in a restaurant?
Response to Example 1:
{
"choice": [0, 1, 2],
"outline_level": "Image 0 shows that KFC on Renmin Road has a birthday party on
February 3rd with musical entertainment. Image 1 indicates that Shanghai hotels
have musical instruments playing during meals, suggesting live music performances.
Image 2 shows an invitation letter for a music performance at Qintai Art Museum,
which relates to music events."
}
15

The prompt for the Extractor is provided below:
Prompt for Extractor
# Task Description #
Extract detailed information from the images selected by Locator based on specific extraction
requirements. This agent performs fine-grained reading of the selected images to extract specific
details such as numbers, text, names, locations, dates, times, prices, etc.
# Input Format #
{{Question},{Image},{Extraction Requirement}}
# Response Format #
Please provide the extracted information in JSON format, with image keys as dictionary keys:
{
"img_0": "Detailed description of information found in image 0, including specific
values, text, or other requested details",
"img_1": "Detailed description of information found in image 1, including specific
values, text, or other requested details",
...
}
# Example1 #
Question Context: Where can I find a bookstore that sells rare books?
Extraction Requirement: Extract bookstore names, addresses, and business hours.
Response to Example 1:
{
"img_1": "Rare Finds Bookstore, located at 123 Main Street, open from 9:00 AM to 6:00
PM",
"img_5": "Price list showing rare books: Ancient manuscripts $120, Historical documents
$200, Rare first editions $350"
}
# Example2 #
Question Context: What time is the train departing from Hangzhou to Beijing?
Extraction Requirement: Extract train departure times and destinations.
Response to Example 2:
{
"img_0": "Train ticket: Hangzhou to Beijing, departure time 14:30, seat number 5A, date
March 15",
"img_2": "Train schedule board showing multiple departures: 08:45 to Shanghai, 14:30 to
Beijing, 16:20 to Guangzhou"
}
16

The prompt for the Verifier is provided below:
Prompt for Verifier
# Task Description #
Evaluates whether the current evidence is sufficient to answer the question accurately. Returns
completeness score and gaps.
# Input Format #
{{Question},{Image},{Evidence Bank}}
# Response Format #
Please provide the extracted information in JSON format, with completeness score and gaps as
dictionary keys:
{
"completeness_score": A number from 0-10 (0=insufficient, 10=complete and accurate),
"gaps": "Description of missing information. Empty string if evidence is complete."
}
# Example1 #
Question: What time is the train departing from Hangzhou to Beijing?
Evidence Bank: coarse_answer="14:30", outline_level="Image 0 shows train ticket with departure
14:30", detail_level={"img_0": "Train ticket: Hangzhou to Beijing, departure 14:30, seat 5A, date
March 15"}
Response to Example 1:
{
"completeness_score": 10,
"gaps": "None"
}
# Example2 #
Question: Where can I find a bookstore that sells rare books?
Evidence Bank: coarse_answer="Rare Finds Bookstore", outline_level="Image 1 shows ’Rare
Finds Bookstore’ sign"
Response to Example 2:
{
"completeness_score": 6,
"gaps": "Missing address and business hours information"
}
17

Table 5: Default values of hyperparameters used in D2-
ScaleAgent.
Hyperparameter Default Value
Pruning threshold (α) 0.7
Stability threshold (τ) 0.8
Closure threshold (δ) 8
Maximum decomposed queries 6
Maximum retrieval rounds 5
Maximum reasoning rounds 20
Maximum sub-agent retries 3
LLM token limit per call 8,192
Top-kNone (adaptive)
18