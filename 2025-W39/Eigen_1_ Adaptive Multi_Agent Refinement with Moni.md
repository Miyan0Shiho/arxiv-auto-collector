# Eigen-1: Adaptive Multi-Agent Refinement with Monitor-Based RAG for Scientific Reasoning

**Authors**: Xiangru Tang, Wanghan Xu, Yujie Wang, Zijie Guo, Daniel Shao, Jiapeng Chen, Cixuan Zhang, Ziyi Wang, Lixin Zhang, Guancheng Wan, Wenlong Zhang, Lei Bai, Zhenfei Yin, Philip Torr, Hanrui Wang, Di Jin

**Published**: 2025-09-25 14:05:55

**PDF URL**: [http://arxiv.org/pdf/2509.21193v1](http://arxiv.org/pdf/2509.21193v1)

## Abstract
Large language models (LLMs) have recently shown strong progress on
scientific reasoning, yet two major bottlenecks remain. First, explicit
retrieval fragments reasoning, imposing a hidden "tool tax" of extra tokens and
steps. Second, multi-agent pipelines often dilute strong solutions by averaging
across all candidates. We address these challenges with a unified framework
that combines implicit retrieval and structured collaboration. At its
foundation, a Monitor-based retrieval module operates at the token level,
integrating external knowledge with minimal disruption to reasoning. On top of
this substrate, Hierarchical Solution Refinement (HSR) iteratively designates
each candidate as an anchor to be repaired by its peers, while Quality-Aware
Iterative Reasoning (QAIR) adapts refinement to solution quality. On Humanity's
Last Exam (HLE) Bio/Chem Gold, our framework achieves 48.3\% accuracy -- the
highest reported to date, surpassing the strongest agent baseline by 13.4
points and leading frontier LLMs by up to 18.1 points, while simultaneously
reducing token usage by 53.5\% and agent steps by 43.7\%. Results on SuperGPQA
and TRQA confirm robustness across domains. Error analysis shows that reasoning
failures and knowledge gaps co-occur in over 85\% of cases, while diversity
analysis reveals a clear dichotomy: retrieval tasks benefit from solution
variety, whereas reasoning tasks favor consensus. Together, these findings
demonstrate how implicit augmentation and structured refinement overcome the
inefficiencies of explicit tool use and uniform aggregation. Code is available
at: https://github.com/tangxiangru/Eigen-1.

## Full Text


<!-- PDF content starts -->

EIGEN-1: ADAPTIVEMULTI-AGENTREFINE-
MENT WITHMONITOR-BASEDRAGFORSCIENTIFIC
REASONING
Xiangru Tang1,∗, Wanghan Xu2,∗, Yujie Wang1,∗, Zijie Guo3,∗, Daniel Shao1, Jiapeng Chen1,
Cixuan Zhang1, Ziyi Wang1, Lixin Zhang1, Guancheng Wan4, Wenlong Zhang6, Lei Bai6,
Zhenfei Yin7, Philip Torr7, Hanrui Wang4,8, Di Jin8
1Yale University2Shanghai Jiao Tong University3Fudan University
4University of California, Los Angeles6Shanghai AI Lab7University of Oxford8Eigen AI
ABSTRACT
Large language models (LLMs) have recently shown strong progress on scientific
reasoning, yet two major bottlenecks remain. First, explicit retrieval fragments
reasoning, imposing a hidden “tool tax” of extra tokens and steps. Second, multi-
agent pipelines often dilute strong solutions by averaging across all candidates. We
address these challenges with a unified framework that combines implicit retrieval
and structured collaboration. At its foundation, aMonitor-based retrieval module
operates at the token level, integrating external knowledge with minimal disruption
to reasoning. On top of this substrate,Hierarchical Solution Refinement (HSR)
iteratively designates each candidate as an anchor to be repaired by its peers, while
Quality-Aware Iterative Reasoning (QAIR)adapts refinement to solution quality.
On Humanitys Last Exam (HLE) Bio/Chem Gold, our framework achieves 48.3%
accuracy—the highest reported to date, surpassing the strongest agent baseline by
13.4 points and leading frontier LLMs by up to 18.1 points, while simultaneously
reducing token usage by 53.5% and agent steps by 43.7%. Results on SuperGPQA
and TRQA confirm robustness across domains. Error analysis shows that reasoning
failures and knowledge gaps co-occur in over 85% of cases, while diversity analysis
reveals a clear dichotomy: retrieval tasks benefit from solution variety, whereas
reasoning tasks favor consensus. Together, these findings demonstrate how implicit
augmentation and structured refinement overcome the inefficiencies of explicit tool
use and uniform aggregation. The code is available at https://github.com/
tangxiangru/Eigen-1.
1 INTRODUCTION
0102030405060Eigen-1 (48.3%)
Figure 1:HLE Bio/Chem Gold overall accuracy.On the 149-problem HLE Bio/Chem Gold split
(Pass@1, auto-judged by o3-mini ), our system attains48.3%accuracy, exceeding the strongest
agent baseline (SciMaster) by+13.4points and leading frontier LLMs by up to+18.1points.
∗Equal contribution.
1arXiv:2509.21193v1  [cs.CL]  25 Sep 2025

Quer y : In pop ul a tion g enetic s , ho w c an w e e s tima t e the eff ectiv e
pop ul a tion siz e        fr om s ample polymorphism da t a ?
E r r o r  T y p e  1 :  O v e r  C o n f i d e n t ,  N o  R A G E r r o r  T y p e  2 :  R A G  C a l l i n g  D i s r u p t e d  R e a s o n i n g
A n s w e r :  O k a y ,  I  k n o w  t h i s  i s  a
< p o p u l a t i o n  g e n e t i c s >
p r o b l e m .  E f f e c t i v e  p o p u l a t i o n
s i z e        i s  c a l c u l a t e d  t h r o u g h
f o r m u l a                        .  S o  i f
c a n  b e  e s t i m a t e d  f r o m
p o l y m o r p h i s m  d a t a ,  I  c a n  j u s t
p l u g  i t  i n t o  t h e  f o r m u l a  a n d
g o t :                              . . .  
 A c t u a l l y ,  t h e  c o r r e c t  f o r m u l a
i s                             .  S o  t h e
a n s w e r  i s  [ w r o n g ]  .  
Corr ectne s s:A n s w e r :  T h i s  l o o k s  l i k e  a
< p o p u l a t i o n  g e n e t i c s  e s t i m a t i o n
p r o b l e m > .  I  t h i n k  i t  i n v o l v e s
W a t t e r s o n ’ s  e s t i m a t o r ,  b u t  I  a m  n o t
e n t i r e l y  s u r e  o f  t h e  e x a c t  f o r m u l a .
I  d o n ’ t  w a n t  t o  m a k e  a  m i s t a k e …
m a y b e  I  s h o u l d  c a l l  r e t r i e v a l .
<code>
output = RAG("Watterson's estimator")
</code> 
O k a y ,  s o                         .   B u t  n o w  I  n e e d
t o  r e - a l i g n  m y  r e a s o n i n g :  h o w  t o
c o n n e c t  b a c k  t o  t h e  o r i g i n a l  p r o b l e m
s t a t e m e n t ?
 I n t e r r u p t e d  T h i n k i n g[Ex ec u a tion Err or]Figure 2:Population genetics case with two failure modes.Left (Error Type 1):the model confidently recalls
an incorrect formula ( θ= 2N eµ) and derives Ne=θ/2µ , yielding the wrong answer.Right (Error Type 2):the
model retrieves the correct relation ( θ= 4N eµ) via explicit RAG, but the reasoning flow is disrupted and the
result is not reintegrated into the original problem, illustrating thetool tax. Our Monitor-based RAG avoids this
context suspension by injecting the correct formula directly into the reasoning stream.
Recent advances in large language models have enabled impressive performance on a spectrum of
reasoning benchmarks, from general-purpose evaluations such as MMLU [ 21] and mathematical prob-
lem solving [ 9;43] to domain-specific tasks including ScienceQA [ 38], MedQA [ 29], and GPQA [ 42].
These results indicate that LLMs can already handle factual recall and mid-level reasoning across
diverse domains. However, when moving to more demanding benchmarks such as Humanity’s Last
Exam (HLE) [ 39;47], which targets expert-level biology and chemistry problems, performance de-
grades substantially, and systematic failures persist when problems require deep domain knowledge
and complex multistep reasoning [ 7]. Through comprehensive analysis of error patterns across 149
HLE Bio/Chem problems, we identify two fundamental architectural limitations:(1) the fragmentation
of logical flow through explicit tool invocation, and(2) the inefficiency of democratic multi-agent
collaboration.
Current retrieval-augmented generation systems [ 3;19;33] require explicit interruption to access
external knowledge. Each retrieval breaks the reasoning flow: suspending the logical state, formulating
queries, processing results, and reconstructing the context. Thistool taxcompounds quickly: solving
population genetics problems requires Watterson estimators requires 8-10 such interruptions, doubling
the number of agent steps compared to a baseline without information retrieval (see Table 3) while
reducing coherence. The problem persists in all RAG paradigms: single-round approaches [ 1;45]
cannot adapt to emerging needs, iterative systems [ 27;44] compound interruption costs, and reasoning-
aware methods [51; 55] remain bound by explicit invocation, as shown in Figure 2.
Simultaneously, most current multi-agent systems [ 4;52] employ rigid democratic workflows: genera-
tion, criticism, synthesis, selection, treating all solutions equally regardless of quality. This contradicts
both cognitive science research on hierarchical expert reasoning [ 8;32] and observations of scientific
collaboration where ideas naturally organize into anchors and support [ 14]. Our analysis reveals that
92.8% of the failures involve reasoning errors, while 88.7% involve knowledge gaps, with substantial
overlap, indicating that these challenges are fundamentally intertwined, as shown in Figure 7.
We present EIGEN-1, an efficient agent framework that unifiesMonitor-based RAGeliminatestool
taxthrough implicit augmentation, operating continuously at the token level to detect knowledge gaps
via semantic uncertainty, generate contextual queries, and inject information seamlessly.Hierarchical
Solution Refinement (HSR)rotates each candidate solution as an anchor and applies peer-informed
repair from the remaining candidates, allowing structured cross-solution refinement rather than uniform
averaging.Quality-Aware Iterative Reasoning (QAIR)replaces fixed workflows with adaptive cycles
that respond dynamically to quality trajectories and problem characteristics. While our experiments
focus on integration within a multi-agent reasoning framework, the design of Monitor-based RAG is
2

model-agnostic and can in principle be incorporated into other reasoning systems without architectural
modification.
Our system achieves 48.3% accuracy in Humanity’s Last Exam Bio/Chem Gold, surpassing SciMas-
ter [4] (34.9%) by 13.4 percentage points while reducing token consumption by 53.5% and agent
steps by 43.7%. Solution pattern analysis further validates our framework: retrieval tasks benefit from
diversity, whereas reasoning tasks favor consensus. These results demonstrate that eliminating the tool
tax and embracing hierarchical collaboration enables both superior performance and computational
efficiency, with potential implications that might extend beyond scientific reasoning to any domain
requiring complex knowledge integration with logical inference.
2 RELATEDWORK
2.1 EVOLUTION OFRETRIEVAL-AUGMENTEDGENERATION
The integration of external knowledge into language model reasoning has evolved through
three main paradigms.Single-round RAGsystems [ 19;25;33] employ linear pipelines
(rewrite →retrieve →generate) and are effective for factual queries. REALM [ 19] enabled end-to-
end retrieval training, while RAG [ 33] extended this to knowledge-intensive tasks. More recent variants
such as REPLUG [ 45] and In-Context RALM [ 41] improve robustness via black-box integration, but
they lack adaptivity when knowledge needs emerge mid-reasoning.Iterative RAGintroduces retrieval-
generation loops for dynamic knowledge acquisition. ITER-RETGEN [ 44] alternates retrieval and
generation, Self-RAG [ 1] uses self-reflection to decide retrieval, FLARE [ 27] predicts future content,
and DRAGIN [ 48] updates datastores in real time. These improve grounding but typically incur 35 ×
more API calls.Reasoning-aware RAGembeds retrieval into reasoning itself. Chain-of-Note [ 57]
produces reading notes, RAT [ 51] couples retrieval with thought generation, IRCoT [ 49] interleaves
retrieval with chain-of-thought, and ReAct [ 54] unifies reasoning with action. While more integrated,
they still depend on explicit tool calls, fragmenting reasoning and increasing latency.
Table 1 summarizes these paradigms against our Monitor-based approach. Unlike step-level methods
that pause to query, Monitor-based RAG operates globally at the token level: it monitors uncertainty
signals and implicitly injects evidence into context, reducing retrieval overhead while preserving
reasoning continuity. Moreover, its retrieval granularity is finer, enabling more precise and frequent
evidence integration without overwhelming the reasoning process.
Table 1:RAG paradigms vs. key capabilities.Single-round RAG is efficient but inadaptable; iterative RAG
improves grounding but increases latency; reasoning-aware RAG offers tighter coupling yet still relies on explicit
calls.Monitor-based RAGintegrates evidence implicitly at the token level, improving continuity and efficiency.
System Triggering Fine-grained Continuity Efficiency Adaptivity
Single-round RAG✗ ✗ ✗ ✓ ✗
Iterative RAG✓ ✓ ✗ ✗ ✓
Reasoning RAG✓ ✓ ✓ ✗ ✓
Monitor-based RAG (Ours)✓ ✓ ✓ ✓ ✓
2.2 MULTI-AGENTREASONINGSYSTEMS
Multi-agent frameworks have shown promise through collaborative problem solving, yet many rely on
rigid orchestration assumptions.
Democratic collaboration systemstreat all agents equally. SciMasters [ 4] employs solvercriti-
crewriter pipelines with a selector over candidate solutions, while LLM-Debate [ 13], Debate-Only-
When-Necessary [ 15], and Multi-Agent Debate [ 35] use argumentative dialogue at different scales.
MetaGPT [ 22] assigns role-based responsibilities, and CAMEL [ 34] explores autonomous cooperation.
Table-Critic [ 56] extends these ideas to structured domains such as tabular reasoning. Such approaches,
however, may devote substantial computation to low-quality candidates and do not explicitly capture
hierarchical relationships among solutions.
Structured reasoning systemsexplore non-linear organizations. Tree-of-Thoughts [ 53] enables
branching exploration with backtracking, Graph-of-Thoughts [ 2] allows arbitrary reasoning topologies,
and Everything-of-Thoughts [ 11] combines multiple reasoning patterns. CoMM [ 5] introduces multi-
path prompting, while HM-RAG [ 37] couples hierarchical agents with multimodal retrieval. Although
these methods capture richer reasoning structures, they lack quality-aware adaptation and can rapidly
expand search spaces.
3

c r i t i c a l l y  c h e c k
o n e  b y  o n e
F o r  e a c h  s o l u t i o n
A p p l y  r e p a i r  s t r a t e g i e s :          
-  L o g i c  C o m p l e t i o n
-  N u m e r i c a l  C o r r e c t i o n
-  M e t h o d  R e p l a c e m e n t
-  E x p r e s s i o n  R e f i n e m e n t
S o l u t i o n  Q u a l i t y  R u b r i c :
   L o g i c a l  R e a s o n a b l e
   A n s w e r  C o r r e c t
   E x p l a n a t i o n  C o m p l e t e
   H i g h  C o n f i d e n c e
→   O t h e r w i s e
Quer y
A ns w er
S e l e c t  B e s t
 S o l u t i o n
U n t i l  C o n v e r g e
repaired
solutions
s u g g e s t i o n( b )   F r a m e w o r kC o r r e c t o r P r o p o s e r…G e n e r a t e
i n i t i a l
s o l u t i o n s
a n c h o r
r e f e r e n c e
…
Q u a l i t y - A w a r e  I t e r a t i v e  R e a s o n i n g  
                            ( Q A I R )R a n k e rR e f i n e m e n tMoni t or
R e a s o n i n g
[ O r i g i n a l  C o n t e x t ]
[ A d d i t i o n a l  C o n t e x t ]R A GQ u e r i e r
I n j e c t o rR e s u m e  R e a s o n i n g
O r i g i n a l + A d d i t i o n a l C o n t e x t
Now that I have more
information, I can
continue my reasoning.I f  C o n f u s i o n  D e t e c t e d ,  t r i g g e r s  R A G
H i e r a r c h i c a l  S o l u t i o n  R e f i n e m e n t
( H S R )  …( a)   Moni t or -b as ed R A G
... I know little  about Pi...
By literature, Pi is 3.14 ....To solve this
problem, I first
need to know
what Pi is.
c o r r e c t o rFigure 3:Framework overview.(a)Monitor-based RAGoperates globally during reasoning: the
Monitor detects insufficiency in the reasoning stream, the Querier generates targeted queries, and
the Injector integrates retrieved evidence into context with minimal disruption. (b) Building on this
substrate, theProposergenerates initial candidate solutions. Each candidate is revised individually
by theCorrector, which applies local targeted fixes without access to other solutions. The improved
candidates are then passed toHSR, which enables cross-solution refinement via anchor–reference
relationships. Finally,QAIRevaluates overall quality and may invoke the Corrector again if needed,
while theRankerselects the strongest solution as the final answer.
Recent advancesattempt more flexible or adaptive coordination. AgentVerse [ 6] supports dynamic
team assembly, AutoGen [ 52] enables configurable conversation patterns, and Reflexion [ 46] incorpo-
rates self-improvement signals. Further, evolving orchestration [ 10], intent-propagation strategies [ 40],
RL-enhanced planning with graph-based policies [ 26], and collaborative leaderfollower training [ 16]
highlight the need for adaptive depth and role specialization. Hierarchical orchestration frameworks
such as AgentOrchestra [ 58] and HALO [ 23] exemplify this trend, emphasizing scalable coordination
via layered or logic-oriented control.
In contrast, our HSR and QAIR modules introduce hierarchical refinement and quality-driven iteration.
Rather than following critic–corrector or debate pipelines [ 35;46;52] that operate under democratic
comment–rewrite loops and risk over-investing in weak candidates, HSR organizes solutions into
anchor–reference structures for targeted repair, while QAIR applies quality-thresholded, suggestion-
guided revisions with early stopping. Crucially, both mechanisms operateon top ofmonitor-based
implicitRAG, enabling hierarchical, quality-aware convergence without suspending the reasoning
process, echoing cognitive science findings on expert problem solving [8].
Declarative vs. Procedural Frameworks (DSPy vs. Ours)Declarative frameworks such as DSPy [ 31]
compile tasks into prompt programs and retrieval policies, providing stability but with adaptation largely
at the stage level. Our approach is procedural and run-time: aMonitor,Querier, andInjectoroperate
during inference to adapt reasoning on the fly. This shift from compile-time templates to run-time
control enables finer-grained adaptivity and seamless knowledge infusion.
3 METHOD
Overall workflow.EIGEN-1 integrates global retrieval, role-based reasoning, and higher-level refine-
ment into a unified workflow, as shown in Figure 3 and Algorithm 1. Monitor-based RAG operates
globally during reasoning:Monitordetects insufficiency in the reasoning stream, the Querier formulates
targeted queries, and the Injector seamlessly integrates retrieved evidence back into context. Based on
4

this substrate,Proposergenerates diverse candidate solutions, each of which is individually revised by
Correctorthrough targeted local repairs. The refined candidates are then passed toHierarchical Solution
Refinement (HSR), which introduces cross-solution repair through anchor-reference interactions. Next,
Quality-Aware Iterative Reasoning (QAIR)evaluates overall quality and may invoke the corrector again
for additional improvement. Finally,Rankercompares candidates and selects the strongest as the final
solution. All agents can use web search tool (Serp API [30]) by default.
3.1 MONITOR-BASEDRETRIEVAL-AUGMENTEDGENERATION
Our Monitor-based RAG system augments reasoning implicitly, without fragmenting the workflow
through explicit tool calls. Instead of forcing the LLM to pause, formulate a query, and inject evidence,
the Monitor continuously inspects the reasoning trace, identifies potential knowledge insufficiencies,
and invokes retrieval only when strictly necessary. The construction of the RAG database is shown in
Appendix A.1.
3.1.1 MONITOR: DETECTINGUNCERTAINTY ANDTRIGGERINGRETRIEVAL
The Monitor acts as a sentinel that periodically examines the reasoning trace and determines whether
external knowledge is required:
Monitor(context) =1,if retrieval is required,
0,otherwise.
Here,contextrefers to the partial reasoning sequence. Once insufficiency is detected, the retrieval is
immediately triggered. To balance timeliness and efficiency, the Monitor runs in a streaming setup:
It checks the reasoning at fixed intervals of 512characters with an overlap of 128characters. This
overlapping design ensures that uncertainty markers that cross boundaries are not missed while keeping
latency low. Details of RAG Monitor are in Appendix A.5.
3.1.2 QUERIER: IDENTIFYINGUNCERTAINTY ANDGENERATINGTARGETEDQUERIES
Triggered by the Monitor, the Querier converts the uncertain fragment into one or more retrieval queries:
[query1, . . . ,queryn] =Querier(context) . Here, the Querier maps the reasoning context into one or
more concise, contextually appropriate queries. A key requirement of the Querier is to precisely extract
the minimal set of keywords that capture the essential uncertainty in the reasoning trace. Depending on
the task, this may result in a single keyword or a small collection of terms, each corresponding to a
distinct retrieval perspective. The number and specificity of the generated queries directly determine
the granularity of retrieval, which in turn controls the trade-off between recall and precision in RAG.
By ensuring that queries are as fine-grained as possible, the Querier avoids unnecessary expansion of
the search space while maximizing the relevance of retrieved evidence.
3.1.3 INJECTOR: EVIDENCECOMPRESSION ANDCONTEXTUALINTEGRATION
The Injector first filters and compresses raw RAG outputs into concise, utility-focused snippets to avoid
redundancy and irrelevant noise. Then it rewrites and integrates the selected evidence in the Proposer’s
reasoning context, ensuring coherence and preserving the natural flow of the reasoning narrative. This
two-step design allows the knowledge retrieved to improve accuracy without introducing stylistic or
structural disruptions: additional context=Injector(context,RAG results).
Figure 2 shows a population genetic problem that requires the Watterson estimator. Baseline LLMs
exhibit two characteristic errors: (1) confidently recalling the wrong formula ( θ= 2N eµ) and deriving
an incorrect effective population size, or (2) retrieving the correct formula ( θ= 4N eµ) via explicit RAG
but failing to reintegrate it into the original reasoning chain, a classic case oftool tax. Our Monitor-based
RAG resolves both issues: the Monitor detects semantic uncertainty, the Querier generates a targeted
query, and the Injector seamlessly injects the correct formula into the reasoning stream, allowing the
solution to proceed without disruption and converge to the correct answer, as shown in Figure 4.
3.2 HIERARCHICALSOLUTIONREFINEMENT(HSR)
HSR challenges the assumption that all solutions should contribute equally to the final output. Instead
of democratic averaging, HSR establishes structured relationships among solutions that mirror expert
collaboration patterns. Let S={s 1, . . . , s n}denote the candidate solutions. Each solution is iteratively
designated as the anchor si, while the remaining set R=S \ {s i}provides references. This rotation
ensures that every solution benefits from peer-informed repair, preventing premature convergence to a
single trajectory.
5

S t a r t  f r o m  a  d i p l o i d  a u t o s o m e  t h a t  c o n t a i n s  5  S N P s  f r o m  t w o  i n b r e d  s t r a i n s  o f  t h e  o r g a n i s m ,
a n d  a l l o w i n g  e x a c t l y  o n e  d i r e c t i o n a l  c r o s s o v e r  p e r  g a m e t e  i n  e a c h  g e n e r a t i o n  ( F 1  →  F 2  →  F 3 ) ,
d e t e r m i n e  t h e  n u m b e r  o f  p o s s i b l e  u n i q u e  h a p l o t y p e  s e q u e n c e s  i n  t h e  F 3  g e n e r a t i o n .
 " O k a y ,  t h e n  h a p l o t y p e s  o f  5  S N P s  c a n  b e  r e p r e s e n t e d  a s  b i n a r y
s t r i n g s .  S e q u e n c e s  w i t h  4  c h a n g e - p o i n t s  ( 0 1 0 1 0 ,  1 0 1 0 1 )
c a n n o t  a r i s e  b y  F 3 ,  b u t  a l l  w i t h  0 – 3  c h a n g e - p o i n t s  a r e  p o s s i b l e .
I  s h o u l d  c o m p u t e  h o w  m a n y  s u c h  b i n a r y  s t r i n g s  e x i s t . "
M o n i t o r < D e t e c t e d >  “ U n c e r t a i n t y ”  :   “ M a x i m u m  n u m b e r  o f  r e c o m b i n a t i o n - i n d u c e d  c h a n g e - p o i n t s ”
Q u e r i e r < G e n e r a t e >  “ Q u e r y ” :  “ H o w  m a n y  h a p l o t y p e  r e c o m b i n a t i o n  c h a n g e - p o i n t s  d u r i n g  s i n g l e
c r o s s o v e r  m e i o s i s  i n  p o p u l a t i o n  g e n e t i c s . ”
R A G  T o o l <  R e t r i e v e >  “ K n o w l e d g e ”  :  " D u r i n g  m e i o s i s ,  a  s i n g l e  c r o s s o v e r  i n t r o d u c e s  a t  m o s t  o n e  n e w
b r e a k p o i n t  p e r  g a m e t e .  A c r o s s  s u c c e s s i v e  g e n e r a t i o n s ,  r e c o m b i n a t i o n  p a t t e r n s  a c c u m u l a t e ,  b u t
t y p i c a l l y  r e q u i r e  m o r e  c r o s s o v e r s  t h a n  t w o  r o u n d s  o f  m e i o s i s . . . "
I n j e c t o r < I n j e c t  I n f o r m a t i o n  t o  R e a s o n i n g >  ( 1 )  A t  m o s t  o n e  b r e a k p o i n t  p e r  g a m e t e  ( 2 )  f u l l y  a l t e r n a t i n g
h a p l o t y p e s  c a n n o t  a p p e a r  w i t h i n  a  f e w  g e n e r a t i o n s  a s  F 2  ⟶  F 3Que s tion :
Step 1-1: LLM Begins Reasoning:
" W e  s t a r t  w i t h  t w o  i n b r e d  s t r a i n s  A  a n d  B ,  e a c h  h o m o z y g o u s .  T h e  F 1  g e n e r a t i o n  c a r r i e s  h a p l o t y p e s  f r o m
b o t h  p a r e n t s .  I n  F 2  a n d  F 3 ,  r e c o m b i n a t i o n  c a n  i n t r o d u c e  c h a n g e - p o i n t s ,  b u t  I  a m  u n c e r t a i n  h o w  m a n y
u n i q u e  h a p l o t y p e s  c a n  a r i s e  b y  F 3 "
M o n i t o r - b a s e d  R A G  T r i g g e r e d :
Step 1-2: LLM Resumes Reasoning:
C o d i n g
E x a c u a t i o n
Step 2: Final SummaryE x e c u t o r  O u t p u t :
C o u n t s  b y  c h a n g e - p o i n t s :  { 0 :  2 ,  1 :  8 ,  2 :  1 2 ,  3 :  8 ,  4 :  2 }
T o t a l  v a l i d  h a p l o t y p e s  ( 0 – 3  c h a n g e - p o i n t s ) :  3 0
 "S o , among the 32 po s sible binar y haplotype s of 5 SNP s , only 4 chang e-point s ar e e x c luded.
Ther ef or e , F3 c an pr oduce 30 unique haplotype s . " <ans w er>30</ ans w er>Figure 4:Haplotype counting with single crossovers (F1 →F3).The Proposer exhibits uncertainty about
recombination constraints;Monitordetects insufficiency,Querierissues a targeted query, andInjectorintegrates
two retrieved facts. This enables the reasoning to exclude invalid cases and converge on the correct count of 30
haplotypes.
Formally, the process can be described as s′
i=Refine(s i,R), where Refine(·) denotes the LLM-driven
mechanism that applies multidimensional repairs to the anchor. Specifically, logical completion fills
missing reasoning steps or implicit assumptions, numerical correction resolves arithmetic inaccuracies,
method replacement substitutes stronger strategies for weaker ones, and expression refinement improves
clarity without altering substance. These dimensions ensure that the weaknesses of the anchor are
addressed systematically while preserving its original strengths.
Figure 5 shows a pathway reasoning problem where multiple proposers generate partial but inconsistent
solutions. Baseline multi-agent synthesis averages across candidates, often propagating contradictions
or omitting critical intermediates. Instead, HSR designates one solution as the anchor and integrates
targeted corrections from reference solutions (e.g., filling in missing intermediates or fixing reaction
links). This yields a coherent and biologically valid pathway, demonstrating how HSR consolidates
fragmented contributions into a unified solution. QAIR then evaluates the refined set and can terminate
the process once the quality stabilizes, avoiding unnecessary additional cycles.
3.3 QUALITY-AWAREITERATIVEREASONING(QAIR)
QAIR introduces an evaluation-driven control mechanism to refine candidate solutions after the HSR
stage. Let S′={s′
1, . . . , s′
n}denote the initial set of refined candidate solutions. Each solution s′∈ S′
is evaluated by an LLM-based evaluator on three quality dimensions: logic, answer, and explanation,
and a textual suggestion for improvement is generated. Each dimension is scored on a scale from 0
to 5, and the three quality scores are then combined into a composite score: q(s′) = 0.2·q logic(s′) +
0.6·q answer(s′) + 0.2·q explanation (s′), where the higher weight on the answer dimension emphasizes
the correctness of the final answer while still allowing for logical consistency and explanatory clarity.
Candidates meeting the threshold τ= 3 are retained, while those failing are marked non-passing and
passed to the corrector for targeted revision:˜s=Corrector(s′,suggestion(s′)).
6

T e m  e n t o m o l o g i s t s  c o l l e c t  5 0 0 , 0 0 0  c a m e r a  t r a p  i m a g e s  ( 3 M P ,  1 0  c m  F O V )  o f  p o l l i n a t o r s  o n
S o l i d a g o  a l t i s s i m a .  T h e y  w a n t  t o  i d e n t i f y  a l l  p o l l i n a t o r s  a n d  c o u n t  f l o w e r s  f e d  o n  i n  e a c h  i m a g e .
W h i c h  s i n g l e  m e t h o d  w o u l d  b e  e a s i e s t  t o  p r o c e s s  t h e s e  i m a g e s  f o r  t h e  r e q u i r e d  d a t a ?Que s tion :
D . Manu al, 410h A . EfficientNet, 5 specie s (36h tr ain + 13.8h deplo y)
B . EfficientNet, 500 specie s (126h tr ain + 13.8h deplo y)
C . Re sNet, 500 specie s (128h tr ain + 11.8h deplo y) F . B + CE. A + BAnswer Choice
A n c h o r  S o l u t i o n R e f e r e n c e
S o l u t i o n s
…C .  R e s N e t
 5 0 0  s p e c i e s  EfficientNet (Option A , B) and Re sNet( C) ar e c l as sific a tion
models ⟶   c an ’ t identif y mul tiple pol lina t or s and co unt flo w er s
in the s ame imag e ⟶  Option D  is corr ect
Deployment time:   Option C = 126 + 13.8 = (139.8 hours), less than A: (49.8 hours)
and B: (139.8 hours) ⟶  Option C  is the fastest for deployment
 Al tho ugh EfficientNet is g ener al ly f as t er than Re sNet in pr actice , b u t w e pr oceed
wi th the giv en number . Hence , Option C  is cho s en as the mo s t appr opri a t e .
Numeric corr ection:  Targeted
CorrectnessNumeric corr ection : the comp aris on of deplo yment time is incorr ect;   Option A
is the f as t e s t r a ther than Option C .
Method Corr ection:  Cl as sific a tion models s uch as EfficientNet and Re sNet c annot
simul t aneo us ly r ecogniz e mul tiple ins ect s . Co unting r equir e s ei ther a det ection model (lik e
F as t er R- CNN, Y OL O) or a densi ty map appr o ach. S o , only Option D r emains corr ect.  
L ogic Corr ection : Time c annot be the only cri t erion, e v en tho ugh C has f e w er ho ur s .
Cl arif y Corr ection : E v en if giv en time s s ug g e s t C is f as t e s t, i t c annot be consider ed a s olu tion
bec aus e i t f ails the f undament al r equir ement. Thus , the corr ect ans w er under the sing le-
method r ule is Option D (manu al).  
Figure 5:Illustrative example: HSR.The system rotates anchors among candidate solutions and integrates
targeted corrections from references (e.g., fixing arithmetic mistakes, filling missing steps). Instead of averaging
inconsistent candidates, HSR applies targeted improvements to yield a coherent final answer.
LetF tdenote the set of solutions that fail the evaluation in roundt, andE tdenote the set of solutions
evaluated at round t. Iterative refinement continues exclusively on the subset of failed solutions, forming
the evaluation set for the next round Et+1={˜s|s′∈ Ft}, until all solutions pass or maximum rounds
Tmaxis reached. By coupling structured quality assessment with suggestion-driven repair and avoiding
re-evaluation of already validated candidates, QAIR efficiently converges toward a high-quality solution
set while maintaining logical consistency, answer correctness, and explanatory clarity.
4 EXPERIMENTS
4.1 EXPERIMENTALSETUPTable 2:Benchmark comparison under matched protocol.HLE
Bio/Chem (149 problems; o3-mini judge), SuperGPQA Biology (hard
split), and TRQA Literature (multiple-choice).
Model HLE Bio/Chem SuperGPQA Hard TRQA
Base Models
Kimi K2 6.71 48.91 38.37
DeepSeek V3.1 13.42 66.30 43.60
Claude Opus 4.1 21.48 63.04 42.44
Gemini 2.5 Pro 18.79 65.22 45.93
GPT-5 22.82 61.96 50.58
Grok-4 30.20 66.30 46.51
Agent Systems
SciMaster (GPT 4.1) [4] 9.45 19.78 47.67
Autogen (GPT 4.1) [52] 7.38 29.35 51.74
OpenAI Deep Research (o4-mini) 22.82 39.13 -
Biomni (GPT 4.1) [24] 10.74 43.48 41.09
SciMaster (DeepSeek V3.1) 34.92 66.30 51.74
EIGEN-1 (DeepSeek V3.1, Pass@1) 48.30 69.57 54.65
EIGEN-1 (DeepSeek V3.1, Pass@5) 61.74 78.26 79.07We evaluate our approach
on Humanity’s Last Exam
(HLE) Bio/Chem Gold [ 47]1,
comprising 149 graduate-
level problems in biology,
medicine, and chemistry.
HLE Bio/Chem Gold subset
was manually curated and
corrected by domain experts
to ensure label fidelity. Ad-
ditionally, we test on 92
hard-difficulty problems from
SuperGPQA [ 12] Biology
and 172 problems from
TRQA Literature [ 59]. Our
framework uses DeepSeek-
V3.1 [ 36] as the base model
with temperature 0.5 and 64K
1https://huggingface.co/datasets/futurehouse/hle-gold-bio-chem
7

token limit. Following HLE protocol, we employ o3-mini for automated evaluation (See Appendix A.3).
Beyond accuracy, we log total generated tokens and agent steps in the ablation experiments, as
quantitative measures of the tool tax.
4.2 MAINRESULTS
In the HLE Bio / Chem dataset, our system achieves48.3%accuracy (Pass@1), substantially outper-
forming the strongest baseline Grok-4 (30.2%) by nearly 18 absolute points and more than doubling
the performance of general purpose models such as GPT-5 (22.8%) and Claude Opus 4.1 (21.5%). This
margin is particularly notable, given that HLE problems require domain-specific reasoning rather than
surface-level recall.
In SuperGPQA hard biology, our method reaches69.6%, exceeding all competing large models. The
improvement is consistent across question categories, suggesting that our framework not only boosts
correctness but also improves robustness on especially challenging scientific queries.
Figure 6:Comparison of retrieval
backends within Monitor-based RAG.
Vanilla Vanna HippoRAG LightRAG
RAG Methods010203040AccuracyFinally, in TRQA benchmark, which emphasizes Information
retrieval, integration and reasoning, our method obtains54.7%
Pass@1, surpassing both Grok-4 (46.5%) and Gemini 2.5 Pro
(45.9%). Furthermore, under the more permissive Pass@5,
counting success if any of five attempts is correctaccuracy rises
to79.1%, indicating robustness under a best-of-N setting.
Together, these results establish the advantage of our design
in three heterogeneous benchmarks: biomedical, chemical and
medical, demonstrating not only raw accuracy gains, but also
improved adaptability across domains (Table 2). For more results
of baseline models, please refer to Appendix A.7.
Within the Monitor-based RAG framework, we experimented
with four retrieval backends: Vanilla [ 17], Vanna, Hip-
poRAG [ 28], and LightRAG [ 18]. Among these,HippoRAGachieved the most consistent gains
when coupled with our uncertainty detection. We attribute this to its finer-grained retrieval and
graph-structured indexing, which better capture relevant context fragments without overwhelming the
reasoning stream. Based on these results, we adopt HippoRAG as the default retrieval backend in our
Monitor module (Figure 6).
4.3 COMPONENTANALYSIS
To understand the contribution of each architectural component, we performed systematic analysis
on the full HLE Bio/Chem benchmark (149 problems), considering both incremental build-up and
component ablation (Table 3).
The baseline configuration uses five parallel Proposers with access to a generic web search tool but
without any paper retrieval (no RAG). TheExplicit RAGsetting adds a scientific paper database
queried via embedding-based similarity. Unless otherwise noted, all settings use the same five-Proposer
architecture with CriticCorrector refinement and Ranker selection.
The baseline system without external knowledge achieves 25.3% accuracy, underscoring the limitations
of parametric knowledge alone for graduate-level science problems. Adding an explicit paper database
improves accuracy to 41.4%, but at the cost of a sharp increase in workflow iterations (from 43.4 to 94.8).
This reflects the high overhead of explicit retrieval: each tool call suspends reasoning, requires query
formulation, and forces reintegration of results, fragmenting what should be a continuous reasoning
flow. While the first one or two retrievals may be helpful, repeated interruptions often add little value
and compound this “tool tax.”
Reasoning Process Error 92.78%
Knowledge Application Error 88.66%Execution & Adherence Error 13.40%
Comprehension Error 9.28%
Figure 7:Error type distribution.Analysis of incorrect solution logs shows
reasoning- and knowledge-related errors as dominant. Note that a single
problem may involve multiple error types, so percentages do not sum to 100.Our Monitor-based RAG
mitigates this overhead
through implicit augmen-
tation. By continuously
monitoring generation and
injecting knowledge only
when necessary, it reduces
token consumption by more
than half (from 470.6K to
218.4K) and cuts workflow
8

Table 3:Component analysis from two perspectives on the full HLE Bio/Chem benchmark (149
problems).(a) Incremental build-up: modules are added one by one. (b) Component ablation: each
module is removed from the full system. The baseline configuration uses five parallel Proposers with
web search but without external paper retrieval (no RAG).Steps= agent-level workflow iterations
(not token-level reasoning).
(a) Incremental build-up (b) Component ablation
Configuration Accuracy (%) Tokens (K) Steps Configuration Accuracy (%) Tokens (K) Steps
Baseline (no ext. knowledge & no RAG) 25.3 483.6 43.4 Full system48.3 218.953.4
+ Papers (Explicit RAG) 41.4 470.6 94.8 – (Monitor, Querier, Injector) 48.5 461.3 95.3
+ Monitor only 34.5 218.4 51.3 – Querier 45.9 224.1 53.1
+ Monitor + Querier 36.8 213.0 51.7 – Injector 44.7 202.5 52.1
+ Monitor + Querier + Injector 40.3 229.5 53.1 – HSR 44.8 234.1 53.5
+ Monitor + Querier + Injector + HSR 43.7 214.0 52.9 – QAIR 43.7 214.0 52.9
+ Monitor + Querier + Injector + HSR + QAIR48.3 218.953.4
iterations nearly in half
(from 94.8 to 51.3), while maintaining competitive accuracy (34.5%). Adding the Querier improves
query precision, leading to a modest gain to 36.8%. The limited margin compared to Monitor alone
suggests that the primary bottleneck lies not in query formation but in evidence integration, which is
addressed by the Injector. With the Injector, accuracy rises further to 40.3% with minimal additional
overhead.
Hierarchical Solution Refinement (HSR) then contributes complementary gains, raising accuracy to
43.7%. Instead of naive aggregation, HSR leverages anchorreference interactions to apply targeted
repairs, focusing revisions where they matter most (e.g., filling missing reasoning steps or correcting
arithmetic). This adds some extra reasoning steps but yields proportionally higher accuracy.
Finally, Quality-Aware Iterative Reasoning (QAIR) builds on HSR by selectively invoking the Corrector
when evaluation indicates further refinement is necessary. This yields the best overall result in the
incremental sequence: 48.3% accuracy with 218.9K tokens and 53.4 iterations. Although QAIR
introduces slight additional overhead, it ensures that every revision contributes meaningfully, preventing
uncontrolled exploration or redundant cycles.
The ablation analysis further validates these findings: removing the Monitor results in a significant
increase in the number of tokens and agent steps; and omitting HSR or QAIR lowers final performance
to 44.8% and 43.7%, respectively. Together, these results show that Monitor-based RAG reduces the tool
tax, HSR provides structured cross-solution repair, and QAIR ensures convergence through selective
correction. Their combination achieves both state-of-the-art accuracy and controlled computation.
5 ANALYSIS
5.1 ERRORTYPEDISTRIBUTION
Analysis of failed problems reveals two dominant error modes: reasoning process errors (92.78%) and
knowledge application errors (88.66%), as shown in Figure 7. These frequently co-occur, suggesting
that successful scientific reasoning requires seamless integration of domain knowledge with logical
inference. Execution errors (13.40%) and comprehension errors (9.28%) are comparatively rare,
indicating that the primary challenge lies not in understanding problems or following instructions, but in
maintaining coherent reasoning while accessing relevant knowledge. The strong overlap also suggests
interdependence: missing knowledge often manifests as faulty reasoning steps, and disrupted reasoning
in turn prevents effective incorporation of retrieved facts. For more examples of these different errors,
see Appendix A.6.
5.2 DIVERSITY VS. CONSENSUS INMULTI-AGENTSOLUTIONS
Our framework employs multiple parallelProposersto generate candidate solutions and utilizes a
Rankerto select the final answer. A natural assumption is that higher agreement among Proposers
should correlate with higher accuracy. However, our analysis reveals a more nuanced picture: the
relationship between solution diversity and correctness depends strongly on problem type.
To investigate this, we divide the benchmark into two categories:information retrievaltasks, which
rely heavily on external knowledge, andreasoningtasks, which require longer chains of inference. For
each problem, we measure the level of agreement among Proposers and evaluate how it correlates with
accuracy. Both metrics are scored continuously by an LLM judge on a [0,1] scale: consistency reflects
the pairwise agreement among candidate answers, while accuracy measures the graded alignment
between a candidate and the ground-truth solution (see Appendix A.3). This continuous evaluation
enables fine-grained correlation analysis beyond binary correctness. As shown in Fig 8, retrieval
tasks benefit from diversity (low agreement), whereas reasoning tasks benefit from consensus (high
agreement), with correlation slopes of0.369and0.851, respectively.
9

This dichotomy suggests that different ranking strategies are optimal for different task types. In retrieval-
heavy settings, the Ranker should preserve diversity and aggregate complementary perspectives,
whereas in reasoning-heavy tasks, it should prioritize high-consensus answers as indicators of reliability.
These observations highlight the complementary roles of HSR and QAIR, which operationalize the
transition from diversity to consensus in a task-adaptive manner.
5.3 TOOLTAXQUANTIFICATION
0.0 0.2 0.4 0.6 0.8 1.0
Average Pairwise Consistency Score0.00.20.40.60.81.0Average Accuracy ScoreReasoning Type (n=392)
Trend (r=0.840)
95% ConfidenceCorrelation: 0.840
Slope: 0.851
Samples: 392
0.0 0.2 0.4 0.6 0.8 1.0
Average Pairwise Consistency Score0.00.20.40.60.81.0Average Accuracy ScoreInformation Retrieval Type (n=339)
Trend (r=0.881)
95% ConfidenceCorrelation: 0.881
Slope: 0.369
Samples: 339Consistency vs Accuracy by Question Type
Figure 8:Diversity vs. consensus.Task-dependent effect of solution diversity:
retrieval tasks benefit from variety, while reasoning tasks benefit from agree-
ment. The horizontal axis reports theaverage pairwise consistency scoreamong
Proposers, computed by an LLM-based judge that evaluates semantic overlap
between answers on a 0–1 scale. The vertical axis shows theaverage accuracy
score, also judged by an LLM, which rates the degree of correctness of each an-
swer relative to ground truth on a continuous 0–1 scale (rather than a binary 0/1).
This continuous evaluation enables us to capture fine-grained trends between
diversity and correctness across tasks. The fitted trend lines further highlight
the contrast: retrieval tasks show a relatively flat slope ( ≈0.369 ), whereas
reasoning tasks exhibit a much steeper positive slope ( ≈0.851 ), indicating a
stronger dependence on consensus.The computational burden
of explicit tool invocation
extends beyond simple la-
tency. As shown in Ta-
ble 3, theexplicit RAG
baseline(with Proposers
equipped with paper re-
trieval and web search)
more than doubles agent-
level workflow iterations
compared to the no-IR set-
ting (43.4 →94.8). This
quantifies a hiddentool
taxfrom context switch-
ing between reasoning and
retrieval modes: each call
requires the system to
pause the evolving chain
of thought, formulate a
query, process external
results, and then recon-
struct the local context be-
fore continuingfragment-
ing what should be a con-
tinuous inference process.
Tool Tax
 
T o k e n  D e c r e a s e
A c c u r a c y  I n c r e a s e  Eig en- 1
B as eline+R A G
Figure 9:Quantifying the tool tax.Com-
parison of accuracy and coherence relative
to compute cost, showing the overhead of
explicit retrieval vs. implicit augmentation.
Note that in this analysis, the baseline refers
to the explicit RAG configuration (i.e., Pro-
posers equipped with paper retrieval and web
search), which represents the standard setup
in most existing agent systems.Fig. 9 visualizes this trade-off. Explicit RAG produces
substantially longer traces without commensurate gains,
whereas ourMonitor-based RAGmaintains concise, inter-
pretable reasoning by injecting only the evidence that is
needed, precisely when it is needed. Operating implicitly at
generation time, it delivers comparable knowledge augmen-
tation with markedly fewer tokens and iterations, avoiding
repeated context suspensions.
More broadly, these findings argue forimplicit augmenta-
tionandadaptive tool policiesin agent design: systems
should not treat all retrieval calls equally, but modulate re-
trieval frequency and depth based on emerging uncertainty,
problem structure, and expected utilitypreserving continuity
of reasoning while still accessing external knowledge when
it truly helps.
6 CONCLUSION
Our experiments validate three key architectural innova-
tions in EIGEN-1. First, implicit knowledge augmentation
through Monitor-based RAG substantially reduces explicit
retrieval overhead while preserving reasoning coherence.
Second, HSR improves over uniform multi-agent aggrega-
tion by introducing structured anchor–reference relation-
ships. Third, QAIR adaptively balances exploration and
early stopping, achieving an effective trade-off between di-
versity and consensus. On HLE Bio/Chem, EIGEN-1 reaches 48.3% accuracy under compute-matched
settings, with a 53.5% token reduction, showing that targeted architectural design can enhance both
effectiveness and efficiency. By integrating external knowledge with minimal disruption to reasoning
10

flow, our framework addresses key limitations of prior approaches in complex scientific problem
solving. Future work will extend these principles to additional scientific domains, assess robustness
and transferability, and explore integration into broader scientific workflows.
11

ETHICSSTATEMENT
Our study aims to improve the scientific reasoning capabilities of large language models (LLMs) by
introducing a unified framework that combines implicit retrieval and hierarchical collaboration. All
datasets used in this work, includingHumanity’s Last Exam (HLE),SuperGPQA, andTRQA, are
publicly available under open licenses. No private, sensitive, or human-identifiable data are involved.
All annotations, where applicable, are derived from public benchmarks or generated using synthetic
processes, ensuring that no ethical concerns regarding data privacy or misuse arise. The broader
societal impact of this research lies in its potential to enhance scientific reasoning and complex problem-
solving abilities in AI systems, which can be applied to fields such as education, scientific discovery,
and decision support. Care has been taken to avoid overstating capabilities or drawing misleading
conclusions, and we encourage further research to validate our findings across other high-stakes
domains.
REPRODUCIBILITYSTATEMENT
To ensure reproducibility, we provide detailed descriptions of our dataset preprocessing procedures,
agent prompting strategies, and iterative refinement workflow in the appendix. These include full
pipeline configurations and experimental setups required to reproduce the reported results with high
fidelity.
REFERENCES
[1] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learning
to retrieve, generate, and critique through self-reflection. 2024.
[2]Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas
Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, et al. Graph
of thoughts: Solving elaborate problems with large language models. InProceedings of the AAAI
conference on artificial intelligence, volume 38, pp. 17682–17690, 2024.
[3]Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie
Millican, George Bm Van Den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark,
et al. Improving language models by retrieving from trillions of tokens. InInternational conference
on machine learning, pp. 2206–2240. PMLR, 2022.
[4]Jingyi Chai, Shuo Tang, Rui Ye, Yuwen Du, Xinyu Zhu, Mengcheng Zhou, Yanfeng Wang, Yuzhi
Zhang, Linfeng Zhang, Siheng Chen, et al. Scimaster: Towards general-purpose scientific ai
agents, part i. x-master as foundation: Can we lead on humanity’s last exam?arXiv preprint
arXiv:2507.05241, 2025.
[5]Pei Chen, Boran Han, and Shuai Zhang. Comm: Collaborative multi-agent, multi-reasoning-path
prompting for complex problem solving.arXiv preprint arXiv:2404.17729, 2024.
[6]Weize Chen, Yusheng Su, Jingwei Zuo, Cheng Yang, Chenfei Yuan, Chen Qian, Chi-Min Chan,
Yujia Qin, Yaxi Lu, Ruobing Xie, et al. Agentverse: Facilitating multi-agent collaboration and
exploring emergent behaviors in agents.arXiv preprint arXiv:2308.10848, 2(4):6, 2023.
[7]Ziru Chen, Shijie Chen, Yuting Ning, Qianheng Zhang, Boshi Wang, Botao Yu, Yifei Li, Zeyi
Liao, Chen Wei, Zitong Lu, et al. Scienceagentbench: Toward rigorous assessment of language
agents for data-driven scientific discovery.arXiv preprint arXiv:2410.05080, 2024.
[8]Michelene TH Chi, Paul J Feltovich, and Robert Glaser. Categorization and representation of
physics problems by experts and novices.Cognitive science, 5(2):121–152, 1981.
[9]Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser,
Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve
math word problems.arXiv preprint arXiv:2110.14168, 2021.
[10] Yufan Dang et al. Multi-agent collaboration via evolving orchestration.arXiv preprint
arXiv:2505.19591, 2025.
[11] Ruomeng Ding, Chaoyun Zhang, Lu Wang, Yong Xu, Minghua Ma, Wei Zhang, Si Qin, Saravan
Rajmohan, Qingwei Lin, and Dongmei Zhang. Everything of thoughts: Defying the law of
penrose triangle for thought generation.arXiv preprint arXiv:2311.04254, 2023.
12

[12] Xinrun Du, Yifan Yao, Kaijing Ma, Bingli Wang, Tianyu Zheng, King Zhu, Minghao Liu, Yiming
Liang, Xiaolong Jin, Zhenlin Wei, et al. Supergpqa: Scaling llm evaluation across 285 graduate
disciplines.arXiv preprint arXiv:2502.14739, 2025.
[13] Yilun Du, Shuang Li, Antonio Torralba, Joshua B Tenenbaum, and Igor Mordatch. Improving
factuality and reasoning in language models through multiagent debate. InForty-first International
Conference on Machine Learning, 2023.
[14] Kevin Dunbar. How scientists think: On-line creativity and conceptual change in science. 1997.
[15] Sugyeong Eo, Hyeonseok Moon, Evelyn Hayoon Zi, Chanjun Park, and Heuiseok Lim. Debate
only when necessary: Adaptive multiagent collaboration for efficient llm reasoning.arXiv preprint
arXiv:2504.05047, 2025.
[16] Andrew Estornell et al. How to train a leader: Hierarchical reasoning in multi-agent llms.arXiv
preprint arXiv:2507.08960, 2025.
[17] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun,
Haofen Wang, and Haofen Wang. Retrieval-augmented generation for large language models: A
survey.arXiv preprint arXiv:2312.10997, 2(1), 2023.
[18] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. Lightrag: Simple and fast
retrieval-augmented generation.(2024).arXiv preprint arXiv:2410.05779, 2024.
[19] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. Retrieval aug-
mented language model pre-training. InInternational conference on machine learning, pp.
3929–3938. PMLR, 2020.
[20] Conghui He, Wei Li, Zhenjiang Jin, Chao Xu, Bin Wang, and Dahua Lin. Opendatalab: Em-
powering general artificial intelligence with open datasets.arXiv preprint arXiv:2407.13773,
2024.
[21] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and
Jacob Steinhardt. Measuring massive multitask language understanding.arXiv preprint
arXiv:2009.03300, 2020.
[22] Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Ceyao Zhang, Jinlin
Wang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, et al. Metagpt: Meta programming for a
multi-agent collaborative framework. International Conference on Learning Representations,
ICLR, 2024.
[23] Zhipeng Hou, Junyi Tang, and Yipeng Wang. Halo: Hierarchical autonomous logic-oriented
orchestration for multi-agent llm systems.arXiv preprint arXiv:2505.13516, 2025.
[24] Kexin Huang, Serena Zhang, Hanchen Wang, Yuanhao Qu, Yingzhou Lu, Yusuf Roohani, Ryan
Li, Lin Qiu, Gavin Li, Junze Zhang, et al. Biomni: A general-purpose biomedical ai agent.biorxiv,
2025.
[25] Gautier Izacard and Edouard Grave. Leveraging passage retrieval with generative models for
open domain question answering.arXiv preprint arXiv:2007.01282, 2020.
[26] Ziqi Jia, Junjie Li, Xiaoyang Qu, and Jianzong Wang. Enhancing multi-agent systems via reinforce-
ment learning with llm-based planner and graph-based policy.arXiv preprint arXiv:2503.10049,
2025.
[27] Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang,
Jamie Callan, and Graham Neubig. Active retrieval augmented generation. InProceedings of the
2023 Conference on Empirical Methods in Natural Language Processing, pp. 7969–7992, 2023.
[28] Bernal Jimenez Gutierrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. Hipporag:
Neurobiologically inspired long-term memory for large language models.Advances in Neural
Information Processing Systems, 37:59532–59569, 2024.
[29] Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William W Cohen, and Xinghua Lu. What disease
does this patient have? a large-scale open domain question answering dataset from medical
exams. InProceedings of the Conference on Empirical Methods in Natural Language Processing
(EMNLP), pp. 2397–2407, 2021.
13

[30] Nils Kautto Ernberg. Analyzing google serp: Swedish search queries, 2019.
[31] Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vard-
hamanan, Saiful Haq, Ashutosh Sharma, Thomas T Joshi, Hanna Moazam, et al. Dspy: Compiling
declarative language model calls into self-improving pipelines.arXiv preprint arXiv:2310.03714,
2023.
[32] Jill Larkin, John McDermott, Dorothea P Simon, and Herbert A Simon. Expert and novice
performance in solving physics problems.Science, 208(4450):1335–1342, 1980.
[33] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich K ¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel, et al. Retrieval-augmented
generation for knowledge-intensive nlp tasks.Advances in neural information processing systems,
33:9459–9474, 2020.
[34] Guohao Li, Hasan Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. Camel:
Communicative agents for” mind” exploration of large language model society.Advances in
Neural Information Processing Systems, 36:51991–52008, 2023.
[35] Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Shuming
Shi, and Zhaopeng Tu. Encouraging divergent thinking in large language models through multi-
agent debate.arXiv preprint arXiv:2305.19118, 2023.
[36] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao,
Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report.arXiv preprint
arXiv:2412.19437, 2024.
[37] Pei Liu et al. Hm-rag: Hierarchical multi-agent multimodal retrieval augmented generation.arXiv
preprint arXiv:2504.12330, 2025.
[38] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind
Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought
chains for science question answering.Advances in Neural Information Processing Systems, 35:
2507–2521, 2022.
[39] Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, Josephina Hu, Hugh Zhang, Chen Bo Calvin
Zhang, Mohamed Shaaban, John Ling, Sean Shi, et al. Humanity’s last exam.arXiv preprint
arXiv:2501.14249, 2025.
[40] Xihe Qiu, Haoyu Wang, Xiaoyu Tan, et al. Towards collaborative intelligence: Propagating
intentions and reasoning for multi-agent coordination with large language models.arXiv preprint
arXiv:2407.12532, 2024.
[41] Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown,
and Yoav Shoham. In-context retrieval-augmented language models.Transactions of the Associa-
tion for Computational Linguistics, 11:1316–1331, 2023.
[42] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien
Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a
benchmark. InFirst Conference on Language Modeling, 2024.
[43] David Saxton, Edward Grefenstette, Felix Hill, and Pushmeet Kohli. Analysing mathematical
reasoning abilities of neural models. InInternational Conference on Learning Representations
(ICLR), 2019.
[44] Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, and Weizhu Chen. Enhancing
retrieval-augmented large language models with iterative retrieval-generation synergy.arXiv
preprint arXiv:2305.15294, 2023.
[45] Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke
Zettlemoyer, and Wen-tau Yih. Replug: Retrieval-augmented black-box language models.arXiv
preprint arXiv:2301.12652, 2023.
[46] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Re-
flexion: Language agents with verbal reinforcement learning.Advances in Neural Information
Processing Systems, 36:8634–8652, 2023.
14

[47] Michael Skarlinski, Jon Laurent, Albert Bou, and Andrew White. About 30% of humanitys
last exam chemistry/biology answers are likely wrong. https://www.futurehouse.org/
research-announcements/hle-exam, July 2025. Accessed: 2025-09-23.
[48] Weihang Su, Yichen Tang, Qingyao Ai, Zhijing Wu, and Yiqun Liu. Dragin: dynamic retrieval
augmented generation based on the information needs of large language models.arXiv preprint
arXiv:2403.10081, 2024.
[49] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Interleaving
retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions.arXiv
preprint arXiv:2212.10509, 2022.
[50] Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen
Liu, Yuan Qu, Fukai Shang, et al. Mineru: An open-source solution for precise document content
extraction.arXiv preprint arXiv:2409.18839, 2024.
[51] Zihao Wang, Anji Liu, Haowei Lin, Jiaqi Li, Xiaojian Ma, and Yitao Liang. Rat: Retrieval
augmented thoughts elicit context-aware reasoning in long-horizon generation.arXiv preprint
arXiv:2403.05313, 2024.
[52] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun
Zhang, Shaokun Zhang, Jiale Liu, et al. Autogen: Enabling next-gen llm applications via
multi-agent conversations. InFirst Conference on Language Modeling, 2024.
[53] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik
Narasimhan. Tree of thoughts: Deliberate problem solving with large language models.Advances
in neural information processing systems, 36:11809–11822, 2023.
[54] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao.
React: Synergizing reasoning and acting in language models. InInternational Conference on
Learning Representations (ICLR), 2023.
[55] Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan Berant. Making retrieval-augmented language
models robust to irrelevant context.arXiv preprint arXiv:2310.01558, 2023.
[56] Peiying Yu, Guoxin Chen, and Jingjing Wang. Table-critic: A multi-agent framework for
collaborative criticism and refinement in table reasoning.arXiv preprint arXiv:2502.11799, 2025.
[57] Wenhao Yu, Hongming Zhang, Xiaoman Pan, Kaixin Ma, Hongwei Wang, and Dong Yu.
Chain-of-note: Enhancing robustness in retrieval-augmented language models.arXiv preprint
arXiv:2311.09210, 2023.
[58] Wentao Zhang et al. Agentorchestra: A hierarchical multi-agent framework for general-purpose
task solving.arXiv preprint arXiv:2506.12508, 2025.
[59] Zhongyue Zhang, Zijie Qiu, Yingcheng Wu, Shuya Li, Dingyan Wang, Zhuomin Zhou, Duo
An, Yuhan Chen, Yu Li, Yongbo Wang, et al. Origene: A self-evolving virtual disease biologist
automating therapeutic target discovery.bioRxiv, pp. 2025–06, 2025.
15

A APPENDIX
A.1 RAG DATABASECONSTRUCTION
To construct the RAG dataset, we sourced 10,876 PDF papers in biology and chemistry from Open-
DataLab [ 20]. We then converted these PDFs to plain text using MinerU [ 50] to ensure downstream
readability and analyzability.
Because the raw corpus spans many topics, we designed a two-stage semantic filtering pipeline to focus
the final corpus on biology- and chemistry-centric research.
We defined a set of positive keywords to capture target research areas, and in parallel, we curated
negative keywords to exclude off-topic or tangential materials, as demonstrated in Figure A1.
Positive and Negative Keywords
Positive Keywords
•Biology:biology, DNA Replication, RNA Transcription, Protein Synthesis, Gene
Editing, Viral Infection, Cell Signaling, Nucleic Acid Probes, Genomic Sequenc-
ing, Transgenic Technology, Immune Response, Biomarkers, Cell Culture, CRISPR
Technology, Viral Vectors, RNA Interference, Gene Expression Regulation, Cell Differ-
entiation, Metabolic Pathways, Apoptosis, Bioinformatics
•Chemistry:chemistry, Organic Synthesis, Inorganic Chemistry, Catalysis, Poly-
mer Chemistry, Spectroscopy, Crystallography, Chemical Kinetics, Thermodynamics,
Electrochemistry, Quantum Chemistry
Negative Keywords
•Non-biology:Cosmetics, Food Additives, Drug Advertising, Environmental Pollu-
tion, Ecological Balance, Medical Ethics, Social Sciences, Psychology, Nutrition,
Educational Methods
•Non-academic chemistry:Household Chemicals, Industrial Wastewater, Pesticide
Residues, Fertilizer Application, Chemical Engineering Safety, Petrochemical Produc-
tion
Figure A1:Positive and Negative Keywords.
This design concentrates positives on fundamental research fronts (e.g., gene editing, molecular
signaling, organic synthesis, spectroscopy, thermodynamics) while negatives cover applied or peripheral
themes (e.g., chemical production, pesticide residues, environmental pollution), improving separation
of target papers from irrelevant content.
For filtering, we encoded each paper’s title and abstract with a pretrained Transformer and computed
cosine similarities against the positive and negative keyword sets. Papers were retained only if
cos(E paper, Epositive )>0.2,cos(E paper, Enegative )<0.1,
where Epaper denotes the vector representation of a paper’s title and abstract, and Epositive /Enegative
denote aggregate vectors of the positive/negative keyword sets. This step effectively removed content
unrelated to biology and chemistry. The post-filter distribution is summarized in Table A1.
Table A1: Paper Classification Statistics by Domain (Side-by-Side)
Biology Category (n=2029) % of Domain Chemistry Category (n=359) % of Domain
Molecular and Cell Biology (777) 38.39% Organic Chemistry (172) 47.91%
Immunology and Microbiology
(482)23.76% Physical Chemistry (71) 19.78%
Genetics, Genomics & Computa-
tion (411)20.26% Materials Chemistry (68) 18.94%
Neuroscience (205) 10.11% Analytical Chemistry (37) 10.31%
Ecology and Evolution (149) 7.35% Inorganic Chemistry (11) 3.06%
16

Overall Summary
Total Biology Papers: 2029 (100%) Total Chemistry Papers: 359 (100%)
Total Papers: 2388 (100%)
After semantic filtering, we used a large language model (LLM) to extract structured text suitable for
Retrieval-Augmented Generation (RAG). We designed a paper-specific prompt that guides the LLM to
segment each paper into retrievable knowledge units (e.g., definitions, methods, experimental results,
discussion) with consistent formatting across papers.
Prompt of RAG Bullet-Point Summarization
Role:You are an information synthesis assistant.
General Rules:
•Use ONLY the paper content provided below. No outside knowledge or invented facts.
• Do NOT include verbatim quotes, citations, or page references.
•The final output MUST be a single CSV code block with rows containing ONLY
synthesized, self-contained bullet-point summaries for RAG.
Complete Paper Content:
{paper content}
Objective:Read the entire paper content and internally construct self-contained knowledge
paragraphs. Then derive standalone, self-contained bullet-point summaries from those
paragraphs for retrieval-augmented generation (RAG). The intermediate knowledge paragraphs
are an internal step and MUST NOT be printed in the final output.
Process:
•Phase 1 Internal Knowledge Paragraphs (DO NOT OUTPUT):
–After reading the full content, synthesize a set of self-contained knowledge
paragraphs.
–Each paragraph must be strictly grounded in the provided text, define acronyms
upon first use, include concrete details when available (tasks, datasets, sample
sizes, metrics, effect sizes, confidence intervals, ablations, baselines, hyperpa-
rameters, assumptions, limitations), written in neutral third-person factual style,
and able to stand alone without context from other paragraphs.
•Phase 2 RAG Bullet-Point Summaries (FINAL OUTPUT ONLY):
–Produce around 3 bullet points in total.
–Each bullet must be self-contained, concise (13 sentences), define acronyms upon
first use, include concrete quantitative or methodological details when available,
state scope/assumptions/limitations when given, and use neutral third-person
factual style.
Output Format (CSV ONLY):
• Output EXACTLY ONE CSV code block and NOTHING ELSE.
• Header MUST be:name,year,locator,topic,quote
• For EACH bullet point, create ONE row with:
–name = ”SYNTHESIZED POINT SUMMARY”
–year = N/A
–locator = N/A
–topic = N/A
–quote = the bullets full self-contained text (escape quotes as needed; no line
breaks inside a cell)
• Do NOT include the intermediate knowledge paragraphs in the output.
• Do NOT add extra columns or any prose outside the CSV block.
Begin:Output only the CSV code block.
17

Through this pipeline, we obtained a topic-focused, structurally consistent research corpus that provides
high-quality knowledge support for downstream RAG systems.
A.2 AGENTPROMPT
A.2.1 REFINEMENT
The following shows the prompt we use in the HSR stage.
Prompt of Refinement
Assistant Prefix Prompt
<think>
Okay, I will answer user’s problem by deep reasoning together
with writing python code in <code></code> format. I should
review and check the solution from student first with web
functions to identify errors if exist, then present my
solution and answer. For example
1. If I want to use the function of web search(keywords),
will say:
keywords=...
results=web search(keywords)
print(results)
2. If I want to use the function of web parse(link, query),
will say:
link=...
query=...
results=web parse(link, query)
print(results)
3. If I want to use the function of search local documents(query),
will say:
query="..."
documents=search local documents(query)
print(results)
4. If I want to do computation, I will write code for
accurate result:
a = 123
b = 456
print(a+b)
Now, let me analyze the user’s question.
</think>
User Prompt
Problem
{query}
Anchor Solution
{anchor solution}
Student 1’s Solution
{reference 1}
Student 2’s Solution
{reference 2}
Student 3’s Solution
{reference 3}
Student 4’s Solution
{reference 4}
18

Your Job
You should critically check the Anchor Solution to the
problem, then correct it if needed and write your own answer.
1. Identify its weak points (missing reasoning steps,
calculation errors, unclear logic, etc.).
2. You can refer to other students’ solutions for targeted
improvements relevant to those weak points.
3. Please note that other students’ solutions may have
errors. Please refer to the points worth learning and make
improvements.
4. Apply repair strategies if needed:
- Logic Completion (fill missing reasoning)
- Numerical Correction (fix calculation errors)
- Method Replacement (use a better method if needed)
- Expression Refinement (clarify presentation)
Only use improvements that directly address anchors weak
points. Avoid unnecessary information merging.
You can solve the problem with the help of feedback from a
code executor. Every time you write a piece of code between
<code> and </code>, the code inside will be executed. For
example, when encountering numerical operations, you might
write a piece of code to inteprete the math problem into
python code and print the final result in the code. Based
on the reasoning process and the executor feedback, you could
write code to help answering the question for multiple times
(either for gaining new information or verifying). There are
also several integrated functions that can be used to help you
solve the problem. The available functions are:
1. search local documents(query: str) - this function takes
a query string as input, and the output is a JSON string
containing a list of relevant document snippets from a local,
private knowledge base. This function should be your first
choice for answering questions.
2. web search(keywords) - this function takes keywords
as input, which is a string, and the output is a string
containing several web information. This function will call a
web search engine to return the search results. This function
is especially useful when answering knowledge-based questions.
3. web parse(link:str, query:str) - this function takes the
link and query as input, and the output is a string containing
the answer to the query according to the content in this link.
This function is useful when looking into detail information
of a link.
Your workflow for solving the problem must follow these steps:
- Step 1: Local Document Search (Mandatory First Action):
You must always begin by using search local documents to check
for relevant information in the private knowledge base.
- Step 2: Evaluate and Supplement: After receiving results
from search local documents, evaluate them carefully. Treat
this information as a supplement to your background knowledge,
not as absolute truth. This supplementary context may be
incomplete or require further verification.
- Step 3: Web Search & Parse (Verification & Detail): After
your initial local search, use web search to find relevant
web pages for verification or supplementation. If a specific
link from the search results seems particularly useful, use
19

webparse to extract detailed information from that page.
- You should not be overconfident in your knowledge and
reasoning.
- Each time you write code put the code into <code></code>
snippet. Put your final answer in <answer></answer> with.
The following shows the prompt we use in the QAIR stage.
A.2.2 QUALITYEVALUATOR
Prompt of Quality Evaluator
User Prompt
You are an expert evaluator. Your task is to evaluate the
given solution for the problem from multiple perspectives.
Problem
{query}
Candidate Solution
{solution}
Evaluation Dimensions
1. Logical Reasonableness (0-5): Does the reasoning process
follow valid logic?
2. Answer Correctness (0-5): Is the final answer correct and
reasonable?
3. Explanation Completeness (0-5): Does the solution explain
the reasoning clearly and completely?
Output Format
Return your answer strictly in JSON:
{
"quality scores": [float, float, float], // [logic, answer,
explanation]
"suggestion": "Provide an improvement suggestion for this
solution that could help refine it in the next iteration."
}
The following shows the prompt we use in the RAG Monitor.
A.2.3 RAG MONITORPROMPT
Prompt of RAG Monitor
Role:You are a helpful assistant.
Task:Analyze the following text and determine if responding to it accurately requires
retrieving information from an external source.
Instructions:
•If you find any doubt or uncertainty about a concept or term in the text, consider it
necessary to retrieve information (RAG).
• If retrieval is required, answer:yes.
• If no retrieval is required, answer:no.
20

Text:
{text}
Judgment:
The following shows the prompt we use in the RAG Querier.
A.2.4 RAG QUERIERPROMPT
Prompt of RAG Querier
Role:You are a helpful assistant.
Task:Generate a single, concise, and effective search query for retrieving the information
required by the text below.
Instructions:
• Returnonly the search queryitself.
• Do not include explanations, punctuation, quotation marks, or other text.
• The query should be direct and contain only the most essential keywords.
Text:
{text}
Search Query:
The following shows the prompt we use in the RAG Querier.
A.2.5 RAG INJECTORPROMPT
Prompt of RAG Injector
Role & Core Objective:You are an information integration specialist. Your sole task is
to process the provided RAG (Retrieval-Augmented Generation) output. Maximize the
utilization of all relevant information to substantively support the reasoning, argumentation, or
conclusions presented in the main text. Do not perform additional reasoning or generate new
conclusions.
Content Integration Principles:
•Comprehensive Extraction:Extract all valuable information from the RAG outputs
that enhances the logical depth, robustness, and persuasiveness of the main text’s
arguments.
•Seamless Cohesion and Minimal Completion:Maintain smooth contextual coher-
ence and stylistic consistency. Perform minimal completion only if the main text ends
mid-thought.
•Neutral Representation:Present all information objectively. Do not evaluate, ques-
tion, or add subjective commentary.
Output Specifications:
•Output should follow the template: ”¡main text completion if necessary¿. Wait a
minute, by searching information about ¡rag query¿, I found that ¡rag result¿. Now
that I have more relevant information, I can continue my reasoning.”
• Directly appendable to the end of the original main text.
•Do not include process summaries, headings, bullet points, or labels like ”Supple-
ment:”.
Instruction Recap:Only select, filter, organize, and polish the RAG content. Do not perform
external reasoning or add new information.
21

Main Text:
{text}
RAG Query:
{rag query}
RAG Result:
{rag result}
A.3 EVALUATIONPROTOCOL ANDANSWERSCORINGGUIDELINES
Theo3-mini model was employed as an automatic judge to verify model-generated responses against
the reference answers, following the official HLE Evaluation Prompts.
Prompt of Answer Evaluation
Role:You are an expert evaluator.
Task:Judge whether the following response to a question is correct or not based on the precise
and unambiguous correct answer provided.
Question:
{question}
Response:
{response}
Correct Answer:
{correct answer}
Evaluation Instructions:
•extracted final answer:Extract the final exact answer from the response. If there is
no exact final answer, put ’None’.
•reasoning:Explain why the extracted final answer is correct or incorrect based on the
correct answer. Focus only on differences between the response and correct answer.
Do not comment on background, do not attempt to solve the problem, do not argue
for any alternative answer.
•correct:Answer ’yes’ if extracted final answer matches the correct answer (allow
small margin for numerical problems). Answer ’no’ otherwise (any inconsistency,
ambiguity, or non-equivalency counts as ’no’).
•confidence:Extract the confidence score from the response between 0% and 100%.
If no score is available, put 100.
Output Format:
{
"extracted final answer": "...",
"reasoning": "...",
"correct": "yes/no",
"confidence": "..."
}
In Figure 8, the vertical and horizontal axes represent the scores assigned by the LLM for answer
continuation, with output values ranging from 0 to 1. Below are the prompts used to assess the accuracy
and consistency of the answers.
22

Prompt of Answer Accuracy Evaluation
You are a meticulous grader. Evaluate a set of solver responses (up to five) for one stage of a
medical/biological question by comparing each responses FINAL + full RESP reasoning to the
ground truth GT + official rationale R. Output a continuous accuracy in [0,1] for each response.
Inputs:
• Q: question stem (may or may not have multiple-choice options).
•GT: ground-truth answer. This can be a multiple-choice letter or a short text for
free-response questions.
• R: official rationale (may be empty).
•FINAL[i]: the solver’s extracted final answer from < answer > ... < /answer >
(may be a letter or short text).
• RESP[i]: the solver’s entire assistant message for response i in this stage.
How to grade (read carefully):
Grade one solver’s response at a time, each solver’s grading process should be independent,
and should not rely on anything else except the solver’s response, final answer, Q, R, and GT.
The grading process for one solver:
•Determine the solver’s FINAL answer from FINAL[i]. If missing, infer only if
RESP[i] makes the choice unambiguous; otherwise treat as unanswered.
• Compare against GT:
–For multiple-choice questions, check if the letter matches GT (case-insensitive).
–or free-response questions, check semantic equivalence to GT (normalize word-
ing, allow synonyms or equivalent phrasing).
•Evaluate reasoning quality: Does RESP[i] align with R (key findings, mechanisms,
exclusions)? Does it avoid contradictions, hallucinations, or irrelevant statements?
•Scoring recipe (simple, smooth, continuous); Use a continuous score reflecting BOTH
aspects:
–0.94–1.00 →FINAL matches GT and RESP closely aligns with R with sound,
non–contradictory reasoning.
–0.69-0.94 →FINAL matches GT but RESP shows minor gaps, superficiality, or
small inconsistencies.
–0.34-0.69 →FINAL ̸=GT, yet RESP shows substantial, partially-correct rea-
soning aligned with R (good differential, one key mistake).
–0.00-0.34 →FINAL ̸=GT and RESP shows weak/mostly incorrect reasoning
(some relevant bits).
–0.00→Off-topic, unsupported, self–contradictory, or clearly wrong with no
meaningful alignment to R.
• Penalize confidently wrong statements or contradictions; do not reward verbosity.
Return ONLY valid JSON in the following form:
{{
"items": [
{{"accuracy": <float in [0,1]>,
"reason": "<<= 40 words justification>"}},
{{"accuracy": ..., "reason": ...}},
...
]
}}
Now grade the following batch of responses:
• Q:q
• GT:gt
• R:r
• FINALS:{final items}
• RESPONSES:{resp items}
23

Prompt of Consistency Evaluation
You are an expert biomedical exam grader. Below are two independently generated solutions to
the same question. Your task is to evaluate how consistent these two solutions are.
Instructions:
• Compare the reasoning processes, scientific logic, and final answers.
• Assign a consistency score from 0.00 to 1.00 (two decimal places):
–1.00 = Solutions are highly consistent (nearly identical reasoning and conclusion).
–0.00 = Solutions are completely inconsistent (different reasoning and conclusion).
Solution A:
{solution1}
Solution B:
{solution2}
Please provide your consistency score (e.g., 0.85):
A.4 EXTERNALVALIDATION ANDLIMITATIONS
Primary evaluation.As detailed above, our main results are scored automatically by o3-mini
using the official HLE judging prompts and our continuous scoring rubric (Sec. A.3). No human expert
adjudication is included in the reported metrics.
Risk of bias and robustness.Automatic judging ensures scalability and reproducibility but may
introduce grader-specific biases and failure modes, especially for free-response rationales. To mitigate
this concern and to support future replication, we pre-register a small-scale expert validation protocol
focused on HLE free-response items:
•Sampling.Randomly sample n=20 items from HLE Bio/Chem (stratified by topic and
difficulty), prioritizing free-response questions where judging is more nuanced than multiple
choice.
•Blinding.Two independent domain experts (blinded to model identity and to each other’s
scores) will grade each item using the exact same criteria as our o3-mini rubric (binary
correctness and a continuous accuracy score in[0,1]).
•Outputs.For each response, experts record: (i) extracted final answer, (ii) binary correctness,
(iii) a continuous accuracy in[0,1]with≤40-word justification.
•Agreement metrics.We will report expert–expert agreement (percent agreement, Cohen’s
κfor binary correctness; Pearson/Spearman for continuous accuracy) and expert– o3-mini
agreement (macro-F1 for binary correctness; Pearson/Spearman correlations for continuous
accuracy).
•Release.We will release the sampled IDs, anonymized expert score sheets, and scripts to
recompute all agreement statistics in our artifact package.
Takeaway.While our main findings rely on automatic judging for scale and consistency, the above
protocol provides a concrete path to independently verify fairness and robustness on the subset of
free-response items where grader discretion matters most. We will include the full results of this expert
validation in the camera-ready or artifact release.
A.5 RAG MONITORHYPERPARAMETERSETTINGS
In our implementation of the RAG-enhanced reasoning agent, several key hyperparameters are used.
Table A2 summarizes these hyperparameters and their functions.
The choice of query topk = 3 is motivated by our design of frequent and fine-grained monitoring. Each
retrieval must be highly precise, since too many retrieved documents would unnecessarily lengthen the
context, slow down reasoning, and introduce redundant or noisy information.
24

Hyperparameter Value Description
model gpt-4.1-mini LLM model used in RAG Monitor
query topk3 Maximum number of retrieved documents for each
query.
ragchunk512Text chunk size for RAG monitoring.
ragoverlapping128 Number of overlapping characters between consec-
utive chunks to maintain continuity.
max rag2 Maximum number of RAG insertions allowed in
one reasoning step.
temperature0.5 Controls generation diversity; higher values lead
to more randomness.
Table A2: Main Hyperparameter Settings used in RAG Monitor.
The parameters rag chunk and rag overlapping control the monitoring frequency of the RAG module.
A large rag chunk would make detection too sparse, causing some uncertain reasoning fragments to
miss external knowledge injection. The overlapping setting ensures continuity between consecutive
windows and avoids missing potential triggers.
The parameter max rag limits the maximum number of retrievals that can be inserted within one agent
step. This prevents the monitor from triggering too frequently and ensures that the reasoning process
remains stable and forward-moving.
In practice, the RAG monitor is triggered on average 3.64 times per 10,000 generated characters. Each
trigger adds about 176.17 tokens of new context, resulting in an average of 641.25 additional tokens
per 10,000 characters. Although this introduces extra tokens, it reduces the need for explicit tool calls,
which significantly lowers the tool usage cost. As a result, the overall reasoning process requires fewer
steps and consumes fewer tokens, as shown in Table 3.
A.6 ERRORCASEANALYSIS
Here, we examine three representative failure modes of our model: reasoning-process errors, knowledge-
application errors, and comprehension errors. In the subsections that follow, we present a real case for
each and analyze how HSR and QAIR contributed to the failure.
A.6.1 CASE1: REASONING PROCESS ERROR
HLE Question.Transgenic Arabidopsis lines constitutively expressing wheat proteins AKP1, RIB3,
KIB1, and YKL23 were tested in three assays: (i) luminol-based ROS over 60 min to MAMPs
(flagpep25–40, flagpep140–168, csp192–208), (ii) split-luciferase complementation in tobacco leaves,
and (iii) GFP localization under water vs flagpep140–168. Choose the correct statement.
Answer Choices (A–H):
• A. AKP1 and RIB3 are redundant receptors for pepflag22.
• B. KIB1 is the receptor for flagpep25–40 and flagpep140–168 but not for csp192–208.
• C. RIB3 is the coreceptor of AKP1; KIB1 acts downstream of RIB3.
• D. All tested proteins are transmembrane proteins...
• E. YKL23 acts upstream of KIB1; RIB3 does not act upstream of KIB1.
• F. flagpep25–40 is the ligand for AKP1 and csp192–208 for YKL23.
• G. Tobacco lacks an endogenous homolog of AKP1.
• H. None of the above.
HSR (Hierarchical Solution Refinement)
Anchor s∗(initially correct):From the cross-modal evidence, the solver first forms the anchor
“AKP1 requires RIB3 to sense flagpep140–168; KIB1 acts downstream”⇒favorsC.
25

early pass supporting C:
"... the correct statement is choice C: ’RIB3 is the coreceptor
of AKP1;
KIB1 acts downstream of RIB3.’
- ROS: AKP1+RIB3 -> flagpep140-168 (2e6 RLUs), neither alone
responds.
- Split-luc: AKP1<->RIB3 baseline suggests ligand-dependent
complex;
KIB1<->AKP1 and KIB1<->YKL23 are positive.
- GFP: KIB1 relocalizes under flagpep140-168; AKP1/RIB3/YKL23
stay at PM."
HSR Error Note
Misweighting across modalities.HSR over-weighted split-luc magnitudes and under-weighted
ligand dependence implied by ROS.Missed consistency repair.No reconciliation step to
explain baseline AKP1↔RIB3 (no ligand)vs.positive ROS (with ligand).
Where HSR goes wrong (pivot to a faulty anchor):When a later refinement step over-weights
split-luciferase magnitudes and under-weights ligand dependence, the anchor flips toE. Two faulty
moves are visible in the refinement trace:
•Fault 1 (magnitude ⇒direction):Interprets strong KIB1¡-¿YKL23 (8e5 RLU) asdirectional
upstreamnessof YKL23 over KIB1, even though magnitude does not encode causal order.
•Fault 2 (baseline ⇒absence):Treats AKP1 ↔RIB3 baseline (2e2 RLU)without ligandas
evidence against co-reception, ignoring the ROS gain-of-function with AKP1+RIB3 under
flagpep140–168 (a classic ligand-dependent complex pattern).
refinement pivot toward E:
"... search_local_documents timed out. Proceed from assays.
KIB1<->YKL23 is strong (8e5), AKP1<->RIB3 is baseline (2e2),
so YKL23 likely acts upstream of KIB1 and RIB3 does not.
Final choice: E."
HSR diagnosis:The error isreasoning process, not missing knowledge. All requisite facts (ROS
synergy for AKP1+RIB3 at 140–168; KIB1 relocalization; split-luc positives with KIB1) are
available, but the refinement applies invalid inference rules that overturn the initially correct anchor
C.
QAIR (Quality-Aware Iterative Review)
Checks logged as performed vs. missed:
Performed:
- Parse ROS matrix (AKP1+RIB3 -> 140-168; YKL23 -> csp192-208):
OK
- Parse split-luc matrix (KIB1<->AKP1, KIB1<->YKL23 positive;
AKP1<->RIB3 baseline): OK
- Consistency of the "E" narrative with split-luc magnitudes: OK
Missed:
- Cross-modal reconciliation: ligand-dependent complexes can
yield
baseline split-luc (no ligand) yet positive ROS (with ligand).
- Directionality audit: interaction magnitude != causal
upstreamness.
- Sanity link: GFP relocalization of KIB1 implies downstream
role,
which conflicts with "YKL23 upstream of KIB1" narrative.
26

QAIR Error Note
Missed mandatory audits.QAIR should have enforced:
•Cross-modal reconciliation requires an explanation for baseline split-luc vs. positive ROS
under ligand.
• Directionality audit disallow inferring causal order from interaction magnitude alone.
•Downstream sanity link KIB1s relocalization supports a downstream role; flag conflict with
the YKL23 upstream of KIB1 story.
Skipping these allowed a self-consistent butincorrectE narrative to pass.
Observed plateau:QAIR accepts internal consistency of the E narrative without enforcing cross-
modal reconciliation, so the faulty anchor persists.
finalization kept by QAIR:
"... Based on strong KIB1<->YKL23 luminescence and no
AKP1<->RIB3 signal,
YKL23 acts upstream of KIB1, and RIB3 does not act upstream of
KIB1.
<answer>E</answer>"
A.6.2 CASE2: KNOWLEDGEAPPLICATIONERROR
HLE Question.A university field trip samples bats on an island for one month (methodology assumed
valid). The student computes Simpson’s diversity index using the formula D = 1 - N(N-1)/sum n(n-1)
and obtains D = 0. Which statement best describes this result?
Answer Choices:
• A. Mathematically and ecologically valid
• B. Mathematically inconclusive with the index value provided
• C. Not mathematically valid, but ecologically valid
• D. Mathematically valid, but not ecologically valid
• E. Not mathematically or ecologically valid
HSR (Hierarchical Solution Refinement)
Initial anchor (as formed):Accept the problems formula exactly as stated, derive what D= 0
implies under that formula, then judge mathematical vs. ecological validity.
problem framing and anchor start:
"... the standard Simpson’s diversity index is either (1 -
lambda) or (1/lambda)
where lambda = sum n_i(n_i - 1)/[N(N-1)]. The formula given in
the problem,
D = 1 - N(N-1)/sum n(n-1), is not standard. However, for the
purpose of this
problem, we must work with the formula as stated.
The student obtained a value of 0, which occurs when sum n(n-1)
= N(N-1) ..."
What HSR should have applied (correct knowledge use):
•Ecological validity is evaluated against the observed community in the stated sampling frame
(the one-month survey), not against lifetime site anecdotes.
•Given the solver already decided to use the provided formula as stated, D= 0 is mathematically
valid and, if the months sample indeed shows one species,ecologically valid for that sample ⇒
A.
27

HSR Error Note
Misapplied knowledge at refinement.Instead of auditing ecological validity within the
sampling frame, HSR imported out-of-frame site priors (known island diversity), whichoverrode
the sample-level conclusion.Missed consistency repair.After explicitly deciding to work
with the formula as stated, HSR later allowed a pivot to wrong formula ⇒option 4, without a
reconciliation step.
What actually happened in HSR (knowledge misapplication):The refinement step imported
site-level prior knowledge (known island diversity) to overrule the sample-level ecological judgment
and flipped away fromA.
knowledge misapplication trace:
"- Ecological validity: Ecologically, it is invalid because it
contradicts the
known diversity of the island, as multiple species are known to
exist."
"The student chose D, which is option 3."
refinement concluding to option 4 in this run:
"... using a wrong formula makes it mathematically invalid.
Thus, the correct answer is option 4: Not mathematically or
ecologically valid.
<answer>E</answer>
HSR diagnosis:The error isknowledge application. The solver had all needed facts (including the
decision to use the given formula and the implication of D= 0 ) but applied ecological knowledge
at the wrong level (site history rather than the sampled community), causing the anchor to settle on
DorEinstead ofA.
QAIR (Quality-Aware Iterative Review)
Checks recorded as performed vs. missed (from the run text):
Performed:
- Algebra under the given D-formula: D = 0 <=> sum n(n-1) =
N(N-1) (OK).
- Consistency of "mathematically valid" under the accepted
(given) formula (OK).
Missed:
- Ecological validity audit constrained to the stated sampling
frame
(evaluate representativeness of the one-month sample, not
lifetime site knowledge).
- Consistency check: if the month legitimately observed a
single-species sample,
then both mathematical and ecological validity hold => Choice
A.
QAIR Error Note
Missed mandatory audits.QAIR should have enforced:
• Sampling-frame ecological audit - judge ecological validity only within the months survey.
•Formula-consistency audit - after use the formula as stated, reject later pivots to wrong formula
unless reconciled.
•Gold-aligned sanity check - if the accepted sample contains one species, thenbothmathemati-
cal and ecological validity hold⇒A.
Skipping these allowed a self-consistent butincorrectnarrative (ecologically invalid due to
island history) to pass.
28

Observed plateau:QAIR validated internal consistency of the ecologically invalid due to island
history narrative and did not enforce a sampling-frame ecological audit, so the faulty anchor
persisted.
finalization kept by QAIR in this run:
"... Ecologically ... known diversity on the island ... making
the result
ecologically invalid. Thus ... corresponds to option 3."
<answer>4</answer>
A.6.3 CASE3: COMPREHENSIONERROR
HLE Question.A university field trip samples bats on an island for one month (methodology assumed
valid). The student computes Simpson’s diversity index using the formula D = 1 - N(N-1)/sum n(n-1)
and obtains D = 0. Which statement best describes this result?
Options:
• A. Mathematically and ecologically valid
• B. Mathematically inconclusive with the index value provided
• C. Not mathematically valid, but ecologically valid
• D. Mathematically valid, but not ecologically valid
• E. Not mathematically or ecologically valid
HSR (Hierarchical Solution Refinement)
Anchor s∗(as formed by the solver):correct mechanism up to the enal stage (thermolysis of the
sulfoxide→vinyl ether→[3,3]-Claisen→unsaturated aldehyde).
excerpt from the run (mechanistic anchor):
"... thermal elimination at 180 C gives the vinyl ether
CH2=CH-O-C(CH3)2-CH=CH2,
which undergoes a [3,3]-sigmatropic Claisen rearrangement to an
aldehyde ..."
Where HSR should have repaired comprehension (nomenclature layer):For aldehydes, the
parent chain must (i) include the carbonyl carbon as C1 and (ii) maximize chain length while
assigning the lowest locants jointly to C=O and the C=C. Under the rearranged connectivity, the
correct parent ishex, absorbing one methyl into the main chain; the double bond is at C4 and the
remaining methyl is at C5, yielding5-methylhex-4-enal(Gold).
HSR Error Note
Missed comprehension repair.HSR failed to:
• apply the aldehydeparent-chain rule(chain must include C=O and be maximized);
• apply thelowest-locant rulefor the C=C within that parent;
• re-evaluatepent- *vs.hex- *after the [3,3]-shift mapping.
Result: the naming anchor should have flipped to5-methylhex-4-enal, but did not.
What actually happened in HSR (missed repair; wrong anchor kept):The refinement layer
accepted apent-based chain and locked the name accordingly.
excerpts from the run (mis-naming kept by refinement):
29

"The IUPAC name is derived as follows:
- The longest carbon chain containing the aldehyde group is five
carbons (pentanal).
- The double bond is between carbons 4 and 5 (pent-4-enal).
- Two methyl groups are attached to carbon 3 (3,3-dimethyl).
Thus, the correct name is 3,3-dimethylpent-4-enal.
<answer>
\boxed{3,3-dimethylpent-4-enal}
</answer>"
HSR diagnosis:The failure iscomprehensionof IUPAC chain selection rules (parent-chain
identification when both a C=O and an alkene must be included), not mechanism recall. HSR
improved mechanistic clarity but did not apply a naming-level repair to flip from pent- *to the
hex- *parent required by the Gold answer.
QAIR (Quality-Aware Iterative Review)
Checks recorded (as reflected in the run text):
- Mechanistic plausibility (elimination -> vinyl ether ->
Claisen): PASSED
- Role of NaHCO3 as neutralizing base (sulfenic acid): PASSED
- Internal consistency of proposed names vs. drawn skeleton:
PASSED
- IUPAC audit: parent-chain selection and lowest-locant C=C
(REQUIRED): SKIPPED
QAIR Error Note
Missed mandatory audit.QAIR should have enforced:
• Parent-chain audit (aldehyde rule): chain includes C=O and is maximized;
• Lowest-locant audit (C=C) within that parent chain;
•[3,3]- mapping check to determine which branch becomes part of the main chain.
Skipping these allowed a self-consistent butwrongpent- *narrative to pass.
Observed plateau:QAIR converged on a self-consistentpent-chain narrative and terminated
without running the naming audit that would force re-evaluation of the parent chain under aldehyde
rules.
Conclusion kept by QAIR:
"... The major product is 4,4-dimethylpent-5-enal ...
<answer> \boxed{4,4-dimethylpent-5-enal} </answer>"
A.7 ADDITIONALBENCHMARKRESULTS
A.8 PSEUDO-CODE
A.9 USAGE OFLANGUAGEMODELS
We utilized a large language model (LLM) to aid in the preparation of this manuscript. Its use was
limited to editorial tasks, including proofreading for typographical errors, correcting grammar, and
improving the clarity and readability of the text.
30

Table A3: Benchmark comparison on HLE Bio/Chem (149 problems;o3-minijudge)
Model Acc (%)
LLMs
DeepSeek V3.1 (Non-Think) 6.71
Deekseek R1 10.74
Qwen3 235B A22B 15.38
LLM with Tools
Deekseek R1 with Browsing 16.82
DeepSeek V3.1 with Browsing 11.21
Doubao with Browsing 11.21
Agents
Kimi Researcher 9.35
31

Algorithm 1Eigen-1: High-level Workflow
Require:Queryq, configC(LLM & retriever), #proposersK(e.g.,5), QAIR thresholdτ, max roundsT max
Ensure:Final solutions⋆
1:Init:set up LLM, RETRIEVER, and tool endpoints fromC
2:
Proposer generates initial solutions
3:for alli∈ {1, . . . , K}in parallel do
4:S[i]←GENERATE(q)▷initial solution generation
5:end for
6:
Local correction (optional, per-candidate)
7:for alli∈ {1, . . . , K}in parallel do
8:C[i]←CORRECTOR(S[i])▷targeted fixes without cross-solution access
9:end for
10:
Hierarchical Solution Refinement (HSR)
11:R←∅
12:fora∈ {1, . . . , K}do▷rotate anchors
13:A←C[a],Ref←C\ {C[a]}
14:R[a]←REFINE(A, Ref)▷apply peer-informed repairs: logic, numeric, method, expression
15:end for
16:
Quality-Aware Iterative Reasoning (QAIR)
17:t←0,P←R
18:whilet < T maxdo
19:(parallel)for eachs∈P:(q logic, qans, qexp,suggestion)←EVALUATOR(s)
20:score(s)←0.2·q logic+ 0.6·q ans+ 0.2·q exp
21:Pass← {s∈P|score(s)≥τ};Fail←P\Pass
22:ifFail=∅then break
23:end if
24:(parallel)for eachs∈Fail:˜s←CORRECTOR(s,suggestion)
25:P←Pass∪ {˜s|s∈Fail};t←t+ 1
26:end while
27:
Rank & select
28:s⋆←RANKER.SELECT(P)▷e.g., composite score or pairwise compare
29:returns⋆
Subroutines used in all LLM generation process
30:functionMONITOR-BASEDRAG(q)
31:ctx←INITCONTEXT(q)
32:whilenot DONE(ctx)do
33:ctx←LLM.NEXT(ctx)
34:ifMONITOR(ctx) = 1then▷detect uncertainty/insufficiency on-stream
35:qry←QUERIER(ctx)▷minimal, targeted keywords
36:docs←RETRIEVER(qry)
37:ctx←INJECTOR(ctx, docs)▷compress & integrate evidence seamlessly
38:end if
39:end while
40:returnTRACETOSOLUTION(ctx)
41:end function
32