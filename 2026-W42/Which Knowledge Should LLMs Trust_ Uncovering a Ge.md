# Which Knowledge Should LLMs Trust? Uncovering a Generalizable Steering Direction Across Knowledge Conflicts

**Authors**: Yibin Lou, Yi Hu, Jianhui Chen, Chenpeng Wang, Muhan Zhang, Liangming Pan

**Published**: 2026-10-14 12:06:32

**PDF URL**: [https://arxiv.org/pdf/2610.20067v1](https://arxiv.org/pdf/2610.20067v1)

## Abstract
Large Language Models (LLMs) frequently encounter conflicts between internal parametric knowledge and external contextual knowledge, yet the mechanisms governing this selection remain unclear. We show that the choice between context and parametric knowledge is linearly represented and can be controlled along a single direction in the model's residual stream. We first demonstrate that this direction generalizes across factual-conflict settings, including context formats, knowledge categories, and datasets, while consistently shifting behavior between parametric and contextual answers. We then transfer a direction learned from simple factual conflicts to three more complex task families: RAG, memory, and rule following, where models must acquire information from retrieved documents, interaction histories, or explicit rules. Across four model families and multiple transfer settings, steering consistently improves the use of prompt-provided information over the unsteered model. These results suggest that the direction captures a transferable preference for external knowledge rather than a mechanism specific to factual conflicts.

## Full Text


<!-- PDF content starts -->

Which Knowledge Should LLMs Trust? Uncovering a Generalizable
Steering Direction Across Knowledge Conflicts
Yibin Lou1,2,4, Yi Hu3, Jianhui Chen2, Chenpeng Wang5, Muhan Zhang3, Liangming Pan1,2*
1State Key Laboratory of Multimedia Information Processing, Peking University
2School of Computer Science, Peking University
3Institute for Artificial Intelligence, Peking University
4Southern University of Science and Technology
5YiXin-AI Lab, YIXIN, Beijing, China
12310513@mail.sustech.edu.cn
Abstract
Large Language Models (LLMs) frequently
encounter conflicts between internal parametric
knowledge and external contextual knowledge,
yet the mechanisms governing this selection
remain unclear. We show that the choice
between context and parametric knowledge
is linearly represented and can be controlled
along a single direction in the model’s residual
stream. We first demonstrate that this direction
generalizes across factual-conflict settings,
including context formats, knowledge cate-
gories, and datasets, while consistently shifting
behavior between parametric and contextual
answers. We then transfer a direction learned
from simple factual conflicts to three more
complex task families: RAG, memory, and
rule following, where models must acquire
information from retrieved documents,
interaction histories, or explicit rules. Across
four model families and multiple transfer
settings, steering consistently improves the
use of prompt-provided information over the
unsteered model. These results suggest that the
direction captures a transferable preference for
external knowledge rather than a mechanism
specific to factual conflicts. Code is available
athttps://github.com/AheadOFpotato/
knowledge_selection_steering.
1 Introduction
Large Language Models (LLMs) store a vast
amount of world knowledge within their param-
eters, commonly referred to asparametric knowl-
edge, enabling remarkable performance across vari-
ous knowledge-intensive tasks (Wang et al., 2024).
Concurrently, in complex settings involving re-
trieved evidence, multi-turn memories, and explicit
rules, LLMs must reliably acquire externalcontex-
tual knowledge(Gao et al., 2024). However, these
two knowledge sources often contain disparate or
contradictory facts, leading to the challenge of
*Corresponding author:liangmingpan@pku.edu.cn
Context: Windows Phone 7 is developed by Dodge.Question:Windows Phone 7 is developed by?Answer:Microsoft.Use Parameter (Original)Answer: Dodge, is a mobile operating system developed by Microsoft.MixedAnswer: Dodge.Use Context (Steered)
Context:Dodge Dart, created by Toyota.Question:Dodge Dart is developed by?Answer:Dodge.Use Parameter (Steered)Answer: Toyotain collaboration with Dodge.MixedAnswer: Toyota.Use Context (Original)
Figure 1: Under knowledge conflicts, steering along a
single residual-stream direction smoothly shifts outputs
betweenparametric knowledgeandcontextual knowl-
edge, with intermediate mixed or hesitant responses.
knowledge conflict. Such conflicts frequently in-
duce undesired behaviors, where the model indis-
criminately relies on inaccurate sources and thus
produces incorrect outputs (Xu et al., 2024; Cheng
et al., 2024). Consequently, understanding the in-
ternal mechanism of how LLMs choose between
conflicting knowledge sources, and achieving reli-
able control over this knowledge selection process,
has become a critical open problem (Zhao et al.,
2025b; Jin et al., 2024).
Existing approaches address knowledge conflict
primarily from two directions. One line of work
controls knowledge source choice at the input level,
which can be categorized into two main approaches:
explicit prompt or instruction design (Choi et al.,
2025; Merth et al., 2024), and structured RAG tem-
plates that organize retrieved context into desig-
nated slots or label relevance (Cattan et al., 2025).
While straightforward to implement, these methods
provide limited visibility into the model’s internal
arbitration process and offer only indirect control
over the reliance on context versus parameters. A
1
arXiv:2610.20067v1  [cs.CL]  14 Oct 2026

complementary line intervenes oninternal compo-
nents, including manipulating attention (Jin et al.,
2024; Li et al., 2025), steering sparse autoencoder
(SAE) features (Zhao et al., 2025a), and modifying
decoding-time token distributions (Bi et al., 2025;
Wang et al., 2025a). These methods are primar-
ily evaluated on simple factual datasets, without
testing knowledge conflicts involving retrieved evi-
dence, long interaction histories, or explicit rules.
They also incur the cost of task-specific attribution,
feature search, adaptation, or decoding-time mod-
ification, while lacking robust evaluation across
such complex application settings.
These limitations motivate us to ask whether in-
ternal knowledge arbitration admits a simple, gen-
eralizable control mechanism. A growing body of
literature in mechanistic interpretability suggests
that LLMs represent complex high-level concepts,
such as truthfulness (Li et al., 2024), safety (Wolf
et al., 2025; Zou et al., 2025), emotions (Sofroniew
et al., 2026), or refusal behaviors (Arditi et al.,
2024), as linear directions within their activation
space (Turner et al., 2024; Elhage et al., 2022;
Park et al., 2024). These directions serve as causal
mediators that allow fine-grained output steering.
This perspective motivates the following research
question:Can knowledge selection under conflict
be captured and controlled by a low-dimensional
steering axis that generalizes across tasks?
In this work, we demonstrate that a robust
“knowledge selection direction” exists. Specifically,
we extract this vector by contrasting the hidden
states of paired prompts designed to isolate para-
metric reliance from contextual reliance. By iden-
tifying the critical intervention layer through co-
sine similarity analysis, we apply signed activation
steering exclusively at this layer to modulate the
model’s knowledge selection.
A direction demonstrated on one factual-conflict
setting may reflect a local shortcut rather than a
reusable control mechanism. We evaluate across
four architectures and a progression from con-
trolled factual conflicts to complex tasks. Within
the controlled setting, we vary factual format, re-
lation category, and dataset while retaining short
prompts with competing answers. We then ap-
ply the direction learned from these conflicts to
RAG, memory, and rule-following tasks, where re-
trieved documents, multi-turn histories, and natural-
language rules supply information to be acquired.
Together, these experiments test whether different
tasks share a common direction for acquiring input-provided knowledge. Our contributions are:
•We identify a single residual-stream direction on
controlled factual-conflict benchmarks that pro-
duces bidirectional, continuous control of knowl-
edge selection across four model families.
•We demonstrate that the same direction works
across diverse knowledge-conflict settings, in-
cluding changes in context format, knowledge
category, and factual-conflict dataset.
•More importantly, we show that it improves com-
plex RAG, memory, and rule-following tasks.
These improvements suggest that different task
families may share an underlying direction for
acquiring external knowledge.
2 Related Work
Knowledge conflict.Knowledge conflict arises
when parametric and contextual knowledge dis-
agree (Xu et al., 2024; Mallen et al., 2023). Exist-
ing interventions target attention heads (Jin et al.,
2024; Li et al., 2025), SAE features (Zhao et al.,
2025a), or decoding (Bi et al., 2025; Wang et al.,
2025a). Our clean–conflict contrast instead models
explicit source competition and enables bidirec-
tional knowledge selection.
Residual Geometry of Steering Vectors.
Localization-plus-intervention studies trace model
behaviors to layers, heads, or neurons (Meng
et al., 2023; Geva et al., 2023), while activation
steering modifies hidden states at inference time
(Turner et al., 2024; Panickssery et al., 2024).
These approaches build on the view that high-level
concepts can correspond to approximately linear
directions in representation space (Marks and
Tegmark, 2024; Park et al., 2024). Closely
related to our setting, (Zhang and Lin, 2026)
argue that in-context conflict resolution may
involve angular displacement in the residual
stream rather than simple magnitude dilution.
Motivated by this geometric view, we localize
layers by angular separation between paired
clean and conflict prompts, and then steer along a
difference-in-means direction.
Transfer of steering directions.Another ques-
tion is whether a direction learned from one con-
cept captures a local shortcut or a general behav-
ioral axis. Prior work shows truthfulness-related di-
rections can generalize across datasets and causally
affect outputs (Marks and Tegmark, 2024), but that
steering vectors can be brittle under distribution
2

1Pair ConstructionDirection Extraction
＋Activation Steering
Question: Who wrote Pride and Prejudice?
Extractresidual stream at every layer
𝐡"=𝐡+𝜶𝐯steeringstrength
−2.0
−1.0
−0.5
−0.25
0
+0.25
+0.5
+1.0
+2.0
Contextual Knowledge
ParametricKnowledgeExample Output
CharlotteBrontë
Use ContextUseParameters
2Find the largest angle
3Locate and subtractClean → meanc1c2cℓcLAverage residual per layer over cases
→
α < 0
α = 0
α > 0
1→→→→
ℓ
2→→→→
→→→→
L→→→→
⋮⋮
1
1→→→→
1
ℓ
1
2→→→→
→→→→
1
L→→→→
⋮⋮θ1→→θ2→→θℓ→→θL→→
Who
wrote
Pride
and
Prejudice
?
parametric stream
⋮param1hparam2h
paramlhparamLh⋮contextual stream⋮context1h⋮
context2hcontextlhcontextLh
Layer iθᵢ
1
2
ℓ
L
max θℓ
maxangle layer
θℓv
clean - conflict vector
contextlparamlhh  v-=contextlhparamlhParametric MeanContextual MeanAngleClean PromptWho wrotePride andPrejudice?Jane Austen(Parametric Knowledge)Conflict PromptPride and Prejudice iswritten by Charlotte Brontë.+ QuestionCharlotte Brontë(Contextual Knowledge)
Respre
Respost
MHA
MLP
⋮
max1
OriginalResidContextual ResidParametric Resid++
Mixed Reliance
2
3
⋮
Brontëor Austen
JaneAustenFigure 2: Overview of knowledge-selection steering. For each question q, the clean prompt is qalone and the
conflict prompt is the supplied passage followed by q. We localize the layer ℓof maximal angular separation
between their mean residual activations, estimate direction vby a difference-in-means contrast at ℓ, and steer the
conflict prompt at its own final token with h′=h+αv to trade parameter-stored versus context-embedded answers.
shift (Tan et al., 2025). Recent cross-model stud-
ies suggest that concept directions can be aligned
across model families (Huang et al., 2025). Clos-
est steering methods increase reliance on input
information through layer search (Anand et al.,
2026), proxy models (Wang et al., 2025b), context-
sensitivity directions (Minder et al., 2025), or in-
struction vectors (Stolfo et al., 2025). We test a
stricter transfer: whether a direction estimated from
explicit source competition generalizes to other for-
mats, properties, and benchmarks while retaining
bidirectional control.
3 Methodology
3.1 Background
Transformers.We study decoder-only Trans-
formers. Given input of Ntokens t=
(t1, t2, . . . , t N), the model maps them to hidden
states across Llayers. Let hl_i∈Rd_modelde-
noteresidual streamat layer land token position i.
The residual stream is initialized by token embed-
ding, h1_i=Embed(t i), and is iteratively updated
through self-attention and MLP blocks:
˜hl
i=hl
i+Attnl(hl
1:i), hl+1
i=˜hl
i+MLPl(˜hl
i).
(1)
The final output probabilities are decoded from the
hidden states at the last layer via the unembedding
matrix and a softmax function.3.2 Models and Datasets
Models.We evaluate four instruction-tuned mod-
els: Llama-3.1-8B-Instruct, Qwen3-8B (Team,
2025), Gemma-2-9B-it (Team, 2024), and Mistral-
7B-Instruct-v0.3 (Jiang et al., 2023).
Benchmarks.We evaluate steering on five bench-
marks spanning factual conflict, conflict format,
knowledge type, and memorization fallback: NQ-
Swap (Longpre et al., 2022), MacNoise (Hong
et al., 2024), ParaConflict (Li et al., 2025), Counter-
Fact (Meng et al., 2023), and MemoTrap (McKen-
zie et al., 2022). NQ-Swap and MacNoise serve
as the primary settings, while ParaConflict, Coun-
terFact, and MemoTrap test transfer across con-
flict formats, factual relations, and memorization-
based behavior, respectively. Appendix A gives the
dataset construction and splits.
To evaluate transfer beyond simple conflicts, we
further consider three complex task families: RAG
(RGB (Chen et al., 2023); CRAG (Yang et al.,
2024)), memory (LoCoMo, LoCoMo-Plus (Ma-
harana et al., 2024); LongMemEval (Wu et al.,
2025)), and rule following (IFEval (Zhou et al.,
2023); FollowBench (Jiang et al., 2024)). In these
tasks, retrieved documents, interaction histories,
and explicit instructions provide external informa-
tion that the model must correctly incorporate.
3

3.3 Extracting a Knowledge Selection
Direction
Paired prompts.As the first step in Figure 2, for
each question q, we construct two prompts: aclean
prompt, which consists of qalone, and aconflict
prompt, which combines qwith a passage inducing
knowledge conflict.
For example, the clean prompt is:
What is the capital of France?
The conflict prompt concatenates the conflicting
passage with the question:
The capital of France isLyon. What is the capital
of France?
We retain items answered correctly on the clean
question and four paraphrases, but answered with
the injected alternative under conflict, yielding ex-
amples with reliable parametric knowledge and
demonstrated context acquisition.
Layer localization.We localize the intervention
layer by angular separation between clean and
conflict representations, rather than by a perfor-
mance sweep. We run both prompts through the
model and average activations, yielding ¯h(clean)i
Tclean
and¯h(conflict)i
Tconflict, where the final token positions may
differ because the conflict passage changes the
prompt length. At each layeri, we compute
si
Tclean,Tconflict=¯h(clean)i
Tclean·¯h(conflict)i
Tconflict
∥¯h(clean)i
Tclean∥ ∥¯h(conflict)i
Tconflict∥.(2)
where si
Tclean,Tconflictis the cosine similarity be-
tween the two final positions. We select the layer
with the lowest similarity:
ℓ= arg min
isi
Tclean,Tconflict.(3)
This criterion requires no sweep over down-
stream task scores. The final layer is also excluded
when its separation reflects the already-decoded
output distribution rather than a reusable source-
selection representation; only an earlier angular
minimum is eligible for intervention.
The layer-wise curves in Appendix B.2 verify
that the selected layer is the point of greatest clean–
conflict separation across the evaluated models.Extracting the direction.At the selected layer
ℓ, we define the steering direction as a difference-
in-means contrast between the conflict and clean
activations at the last token:
vℓ
Tclean,Tconflict=¯h(clean)ℓ
Tclean−¯h(conflict)ℓ
Tconflict.(4)
Thus vℓ, Tclean,Tconflict points from the conflict
regime toward stronger parameter-stored reliance.
3.4 Applying the Steering Vector
Inference-time intervention.We edit theresid
poststream only at the angular-separation-selected
layer ℓand the conflict prompt’s final token. Ap-
pendix B reports representation-hook, layer, and
token-position ablations. Let hℓ
Tconflictdenote this
activation. We apply the steering vector as
h′ℓ
Tconflict=hℓ
Tconflict+α vℓ
Tclean,Tconflict.(5)
Here, αis a signed control coefficient: positive
values favor parametric reliance, negative values
favor contextual reliance, and α= 0 recovers the
unsteered conflict pass.
3.5 Evaluation
We measure steering by matching the first 256
generated tokens to the parametric and contextual
answers. Each item receives one of four labels:
parametric-only,contextual-only,both, or*nei-
ther*. Let Nbe the number of evaluated items; the
reported unconditional rates are
Acc par=Nparametric-only
N,
Acc ctx=Ncontextual-only
N.
The main table retains these two directional rates.
Appendix C details normalization, the fixed answer
window, the match rules, and the blinded-judge
validation of the inexpensive string-match metric.
We use train/validation/test splits for direction es-
timation, coefficient selection, and final reporting,
with a 70/15/15 split for NQ-Swap.
4 Continuous Control Along a Single
Direction
We first evaluate whether the model’s knowledge
selection behavior can be captured and continu-
ously modulated along a single steering direction
4

Table 1: In-domain source-match rate (%) on NQ-Swap and MacNoise. The upper block reports parametric-source
match rate Acc par, and the lower block reports contextual-source match rate Acc ctx.Boldmarks the best result in
each column; green arrows denote absolute gains over the unsteered baseline. Ours and Random split use the same
validation-selected coefficient. Random split values are mean±standard deviation over 10 seeds.
MethodNQ-Swap MacNoise
Llama-3.1
8B-Instruct Qwen3-8BGemma-2
9B-ITMistral-7B
Instruct-v0.3Llama-3.1
8B-Instruct Qwen3-8BGemma-2
9B-ITMistral-7B
Instruct-v0.3
Steer toward parametric knowledge (Acc par)
Unsteered 33.7 46.5 32.8 42.2 46.1 50.8 45.5 50.3
Random split 32.2±2.6 45.7±1.9 33.5±2.3 41.6±2.1 45.8±3.1 50.2±2.4 44.9±2.0 49.7±2.5
ContextFocus (Anand et al., 2026) 44.6(↑10.9) 49.1(↑2.6) 45.2(↑12.4) 47.7(↑5.5) 50.2(↑4.1) 52.5(↑1.7) 51.3(↑5.8) 54.2(↑3.9)
CCS (Minder et al., 2025) 53.5(↑19.8) 50.8(↑4.3) 58.6(↑25.8) 52.1(↑9.9) 55.2(↑9.1) 53.1(↑2.3) 59.4(↑13.9) 56.8(↑6.5)
CSKS (Wang et al., 2025b) 52.4(↑18.7) 51.7(↑5.2) 66.3(↑33.5) 54.3(↑12.1) 53.7(↑7.6) 53.8(↑3.0) 63.6(↑18.1) 58.4(↑8.1)
CK-PLUG (Zhao et al., 2025a) 54.5(↑20.8) 52.9(↑6.4) 74.7(↑41.9) 56.2(↑14.0) 55.7(↑9.6) 53.4(↑2.6) 69.5(↑24.0) 59.7(↑9.4)
PH3 (Li et al., 2025) 48.3(↑14.6) 47.8(↑1.3) 61.5(↑28.7) 49.5(↑7.3) 49.5(↑3.4) 51.7(↑0.9) 54.7(↑9.2) 52.6(↑2.3)
IRCAN (Shi et al., 2024) 49.5(↑15.8) 51.4(↑4.9) 59.3(↑26.5) 50.8(↑8.6) 50.0(↑3.9) 52.4(↑1.6) 63.2(↑17.7) 53.9(↑3.6)
JUICE (Jin et al., 2024) 53.4(↑19.7) 53.3(↑6.8) 78.4(↑45.6) 57.1(↑14.9) 53.2(↑7.1) 56.1(↑5.3) 74.2(↑28.7) 60.5(↑10.2)
SpARE (Zhao et al., 2025a) 58.7(↑25.0) 49.2(↑2.7) 77.3(↑44.5) 55.4(↑13.2) 59.5(↑13.4) 52.3(↑1.5) 72.3(↑26.8) 57.6(↑7.3)
Ours 56.8(↑23.1) 54.6(↑8.1) 80.9(↑48.1) 59.0(↑16.8) 57.5(↑11.4) 54.5(↑3.7) 76.9(↑31.4) 61.6(↑11.3)
Steer toward contextual knowledge (Acc ctx)
Unsteered 49.4 36.7 52.6 45.8 27.3 22.2 25.4 23.8
Random split 49.2±2.8 37.1±2.2 51.9±2.6 45.3±1.8 26.1±0.7 21.8±1.4 24.9±1.6 23.4±1.3
ContextFocus (Anand et al., 2026) 56.8(↑7.4) 42.3(↑5.6) 58.1(↑5.5) 51.2(↑5.4) 31.5(↑4.2) 25.7(↑3.5) 29.3(↑3.9) 27.1(↑3.3)
CCS (Minder et al., 2025) 52.3(↑2.9) 46.1(↑9.4) 60.7(↑8.1) 54.6(↑8.8) 30.1(↑2.8) 27.4(↑5.2) 32.5(↑7.1) 28.6(↑4.8)
CSKS (Wang et al., 2025b) 57.2(↑7.8) 49.7(↑13.0) 62.9(↑10.3) 56.8(↑11.0) 32.2(↑4.9) 30.1(↑7.9) 35.4(↑10.0) 29.7(↑5.9)
CK-PLUG (Zhao et al., 2025a) 53.4(↑4.0) 55.2(↑18.5) 64.3(↑11.7) 57.7(↑11.9) 30.9(↑3.6) 29.5(↑7.3) 37.7(↑12.3) 30.2(↑6.4)
PH3 (Li et al., 2025) 52.4(↑3.0) 51.5(↑14.8) 54.3(↑1.7) 49.1(↑3.3) 28.9(↑1.6) 28.5(↑6.3) 32.4(↑7.0) 25.7(↑1.9)
IRCAN (Shi et al., 2024) 55.2(↑5.8) 47.4(↑10.7) 57.3(↑4.7) 52.4(↑6.6) 31.7(↑4.4) 32.4(↑10.2) 36.1(↑10.7) 28.8(↑5.0)
JUICE (Jin et al., 2024) 54.5(↑5.1) 48.4(↑11.7) 66.9(↑14.3) 58.3(↑12.5) 34.0(↑6.7) 34.2(↑12.0) 38.5(↑13.1) 30.4(↑6.6)
SpARE (Zhao et al., 2025a) 52.7(↑3.3) 54.4(↑17.7) 68.3(↑15.7) 55.9(↑10.1) 30.9(↑3.6) 34.3(↑12.1) 42.4(↑17.0) 29.2(↑5.4)
Ours 55.5(↑6.1) 56.0(↑19.3) 65.4(↑12.8) 60.2(↑14.4) 32.8(↑5.5) 35.1(↑12.9) 40.6(↑15.2) 31.1(↑7.3)
v. Here, the scalar αserves as a continuous control
coefficient, with positive values shifting the model
toward parametric reliance and negative values to-
ward contextual reliance. Implementation details
are provided in Appendix D.
4.1 Improving Binary Source Choice
Table 1 first shows that Ours improves the desired
source-selection rateover the unsteered model
across the four models and both benchmarks, es-
tablishing control at a validation-selected operating
point. The steering-strength sensitivity analysis
in Appendix E further shows stable transfer per-
formance at neighboring αvalues around each se-
lected operating point. Beyond these discrete end-
points, sweeping αreveals how the same direction
changes source selection continuously.
4.2 Continuous Control at Population and
Instance Levels
Having established discrete source control at a
validation-selected operating point, we next exam-
ine whether the same direction supports a continu-
ous transition asαvaries.At thepopulation level, Figure 3 shows smooth,
opposing changes in parametric and contextual se-
lection as αvaries across models and datasets. This
aggregate trade-off is also visible within individ-
ual fixed examples, and Appendix F shows that it
extends to larger instruction-tuned models.
At theinstance level, Figure 1 traces a response
from one source through mixed outputs to the other
asαchanges. Thus, the aggregate trend reflects
graded individual source selection rather than only
a shift in proportions. Appendix G.1 provides tra-
jectories for every model and cases that do not
reach the requested source.
4.3 Robustness to External-Knowledge
Strength and Question Difficulty
We test two factors:external-knowledge strength
andquestion difficulty. As Table 2 shows, stronger
contextual evidence raises unsteered contextual
selection, while steering remains effective. Ta-
ble 3 shows gains at every difficulty level, with
the largest parametric gain on hard questions.
5

-1.0 -0.5 0.0 0.5 1.0
0.000.250.50Rate
NQ-SwapLlama-3-8B-IT
-1.0 -0.5 0.0 0.5
0.000.250.50
Qwen3-8B
-1 0 1 2 3
0.00.5
Gemma2-9B-IT
-1.0 -0.5 0.0 0.5 1.0
0.00.5
Mistral-7B-Instruct-v0.3
-1.0-0.50.00.51.01.5
0.000.250.50Rate
MacNoise
-1.0 -0.5 0.0 0.5
0.000.250.50
-1 0 1 2 3
0.00.5
-1.0 -0.5 0.0 0.5 1.0
0.00.5
internal externalFigure 3: Bidirectional continuous modulation on NQ-Swap and MacNoise (15% test split). We sweep the full αgrid
directly on test items and plot aggregate source-match proportions at each α; the curves describe population-level
averages and do not imply that every individual instance changes smoothly (noαtuning on test).
Table 2: Robustness to external-knowledge strength on
NQ-Swap with Llama-3.1-8B-Instruct (%).
Context
occurrencesUnsteered
parametricParametric
steeringUnsteered
contextualContextual
steering
1 33.7 56.8 49.4 55.5
2 30.5 55.2 54.2 58.1
4 25.4 53.6 61.8 62.4
8 17.1 47.9 73.5 74.8
Table 3: Robustness to question difficulty on NQ-Swap
with Llama-3.1-8B-Instruct (%).
DifficultyUnsteered
parametricParametric
steeringUnsteered
contextualContextual
steering
Easy 42.6 62.4 45.8 56.2
Medium 33.1 56.1 49.6 57.3
Hard 12.4 49.8 64.2 74.3
4.4 Knowledge Selection Is More Than
Context Sensitivity
The preceding results establish source control, but
not whether it reflects explicit knowledge conflict
or generic context sensitivity. We therefore com-
pare source competition with matched prompts that
add answer-aligned or irrelevant context without
the same parametric–contextual conflict.
Table 4 compares no-context contrasts with con-
flicting, aligned, and irrelevant contexts. The
conflicting-context contrast yields the strongest
source-selection effects across datasets, whereas
matched controls are weaker, indicating more than
generic context sensitivity; Appendix B.3 further
shows that the directions are similar but not identi-
cal (cosine 0.875–0.899).Table 4: Matched controls for knowledge selection ver-
sus generic context sensitivity (%). The three steered
columns contrast no context with conflicting, aligned,
and irrelevant context.
Dataset UnsteeredConflicting
contextAligned
contextIrrelevant
context
Panel A: Parametric-answer selection (positiveα)
ParaConflict 15.986.572.6 74.0
MacNoise 46.273.567.1 62.1
MemoTrap 67.184.171.7 75.3
CounterFact 36.054.446.1 49.1
Panel B: Context/distractor-answer selection (negativeα)
ParaConflict 81.796.584.4 83.2
MacNoise 27.444.531.2 34.0
MemoTrap 29.757.049.9 47.5
CounterFact 60.864.157.6 55.1
4.5 Preserving Performance Without Conflict
Finally, we test the cost of this source preference
when no conflict is present. Table 5 shows that,
with the same fixed directions and coefficients, both
directions retain substantial original-answer accu-
racy across all four datasets, although contextual
steering incurs a larger trade-off on some datasets.
5 Generalization of the
Knowledge-Selection Direction
We next test whether this fixed direction survives
distribution shifts. Section 5.1 tests simple factual-
conflict tasks; Section 5.2 turns to complex tasks
with external knowledge to acquire.
5.1 Across Controlled Simple Factual-Conflict
Tasks
These tests varyformat,semantic relation, or
benchmark constructionwhile retaining factual
6

1
 0 10.00.5
Country Of Citizenship
2
 1
 0 1
 0.5
 0.0 0.5
 0.5
 0.0 0.5 1.0 1.5
 1
 0 1
 1
 0 1
Internal
External
2
 1
 0 1 20.00.51.0
Manufacturer
 2
 1
 0 1 2
 1
 0 1
 1
 0 1
 2
 1
 0 1 2 3
 1
 0 1
2
 1
 0 1 20.00.5
Occupation
2
 1
 0 1 2
 1.0
 0.5
 0.0 0.5
 1
 0 1 2
 2
 1
 0 1
 1.0
 0.5
 0.0 0.5 1.0
2
 1
 0 10.00.51.0
Religion
2
 1
 0 1 2
 1.0
 0.5
 0.0 0.5
 2
 1
 0 1
 2
 1
 0 1 2
 0 1 2
2
 1
 0 1 2 30.00.5
Sport
2
 1
 0 1 2
 1
 0 1 2
 1
 0 1 2 3
2
 1
 0 1 2
 1
 0 1 2 3
2
 1
 0 1 20.00.5
Original Broadcaster
Country Of Citizenship2
 1
 0 1 2 3
Manufacturer1
 0 1
Occupation1
 0 1 2
Religion2
 1
 0 1 2
Sport1
 0 1 2 3
Original Broadcaster
Test Knowledge Type (Column)Direction Knowledge Type (Row)Figure 5: Controlled cross-relation transfer on CounterFact (Llama-3.1-8B-Instruct). Each cell evaluates a direction
learned from one factual relation on another relation.
Table 5: Preservation of the original answer without a
conflicting context. Values are target-answer accuracy
(%); the same fixed parametric and contextual steering
directions used in transfer are applied.
Dataset UnsteeredParametric
directionContextual
direction
ParaConflict 100.0 98.0 85.5
MacNoise 100.0 80.8 86.7
MemoTrap 100.0 84.2 91.1
CounterFact 100.0 88.8 91.7
conflicts: prompts with competing answers. Thus,
this level establishes generalization within the task
family while retaining bidirectional steering.
Cross-format transfer.We begin by varying the
conflict formatwhile holding the fact and distractor
fixed. ParaConflict separates a short substitution
conflict from a richer coherent conflict for the same
fact; Appendix H gives the full prompt forms. Both
prompts inject the same distractor, “football,” for
the correct answer “basketball”:
Short:Michael Jordan plays the sport of football.
Table 6: Controlled factual-conflict cross-dataset trans-
fer on Llama-3.1-8B-Instruct (%). The direction and
coefficients are selected only on NQ-Swap.
Parametric steering Contextual steering
Test data Unsteered Ours Unsteered Ours
MacNoise 46.260.0 27.034.1
ParaConflict 15.883.9 81.490.8
CounterFact 35.738.9 60.060.7
MemoTrap 66.775.4 30.053.3
0.00.20.40.60.81.01.2Internal rateunsteer ( =0)
steer (best >0)
Athelete SportBook Author
Company FounderCompany HeadquarterOfficial LanguageWorld Capital0.00.20.40.60.81.01.2External rateunsteer ( =0)
steer (best <0)
Figure 5: Controlled cross-format transfer on ParaCon-
flict (Llama-3.1-8B-Instruct). The direction is fit on
substitution-format conflicts and evaluated on coherent-
format test items.
Coherent:Recognized by peers and fans alike, Michael
Jordan’s journey in football has been highlighted in sports
publications and athlete profiles. . . .The impact of
Michael Jordan in football is celebrated in community
events and athletic programs.. . .
Figure 5 shows that a direction fitted on the short
format also changes source selection in the coher-
ent format, so the effect is not tied to the estimation
template. Appendix I.1 reports the corresponding
results for the other models.
Cross-relation transfer.Having shown that the di-
rection is not tied to one prompt template, we next
hold the conflict structure fixed and vary theseman-
tic relation. CounterFact lets us learn directions
fromcountry of citizenship,manufacturer,occu-
pation,religion,sport, andoriginal broadcaster,
then evaluate each direction on the other relations.
7

Table 7: Generalization to complex tasks through external-knowledge steering (%). Models are grouped vertically
and tasks horizontally. ContextFocus, JUICE, and CK-PLUG are included as additional Llama-3.1-8B-Instruct
baselines.
Model Method RAG Memory Rule
RGB CRAG LoCoMo LoCoMo-Plus LongMemEval IFEval FollowBench
Llama-3.1-8B-Instruct Unsteered 62.0±1.2 23.7±0.8 8.5±0.4 32.7±1.1 43.2±0.9 33.6±1.0 14.5±0.6
Random split 64.0±1.4 24.7±0.9 8.5±0.4 32.7±1.0 42.8±0.8 33.9±1.1 12.9±0.5
ContextFocus 69.0±0.3 24.5±1.8 10.6±1.8 34.9±2.1 44.1±0.9 33.4±1.7 13.1±1.3
JUICE 62.0±1.1 23.7±0.7 7.6±0.3 34.2±1.9 42.5±2.8 31.0±0.9 13.8±1.4
CK-PLUG 71.0±1.4 25.1±0.9 12.9±1.6 35.3±1.2 46.8±1.0 34.1±1.1 14.6±0.6
Ours 82.0±0.5 25.8±1.0 13.2±0.9 36.7±1.3 45.4±1.1 35.0±1.2 14.9±0.7
Qwen3-8B Unsteered 72.0±1.4 27.5±1.0 9.8±0.5 28.2±0.9 42.8±0.9 34.9±1.1 15.7±0.5
Random split 72.0±1.1 27.5±0.5 10.1±0.5 28.5±1.0 42.1±0.4 34.9±1.0 15.7±1.3
ContextFocus 72.1±0.8 27.6±1.3 9.9±0.6 28.9±1.1 42.9±0.4 34.8±1.1 16.6±1.9
JUICE 68.5±1.2 25.4±1.8 9.0±0.4 26.8±0.8 41.6±0.8 33.1±0.9 14.2±1.2
CK-PLUG 74.3±0.5 28.3±1.1 10.6±0.6 33.5±1.2 44.1±1.0 35.4±1.2 16.6±0.8
Ours 79.0±1.6 30.0±1.2 13.1±0.7 45.4±1.5 43.8±1.1 36.2±1.3 18.3±0.9
Gemma-2-9B-IT Unsteered 71.0±1.3 26.6±0.9 8.8±0.4 9.2±0.3 42.0±0.7 29.2±0.9 16.1±0.1
Random split 72.0±0.7 26.9±0.9 8.8±0.5 11.0±0.4 41.7±0.9 29.8±1.2 15.7±0.7
ContextFocus 77.0±1.1 26.9±1.2 8.7±0.2 11.1±1.2 42.1±1.3 29.7±0.7 16.1±1.8
JUICE 75.0±1.2 25.0±0.6 8.2±0.7 9.6±0.3 41.1±0.7 27.9±0.3 15.0±0.6
CK-PLUG 81.0±0.9 27.0±1.0 9.3±0.5 14.4±0.9 43.6±1.4 30.0±1.1 16.5±2.3
Ours 88.0±1.1 27.7±1.1 10.5±0.6 17.2±0.6 44.2±1.2 30.5±0.9 17.0±0.8
Mistral-7B-Instruct-v0.3 Unsteered 81.0±1.5 27.3±1.0 9.4±0.5 32.7±1.1 40.2±0.8 25.1±0.8 14.9±0.4
Random split 81.0±1.4 27.1±0.9 9.1±0.4 33.2±1.2 39.4±0.9 25.1±0.9 14.9±0.6
ContextFocus 82.0±0.5 27.8±1.0 9.5±0.7 34.6±1.7 40.1±0.8 23.3±1.3 14.8±1.3
JUICE 77.0±1.7 25.8±0.8 8.7±1.4 31.9±1.0 39.3±0.7 23.6±0.7 13.7±0.5
CK-PLUG 82.0±0.6 28.2±1.1 9.9±0.4 33.1±1.3 41.6±1.5 26.4±1.0 15.3±1.1
Ours 86.0±1.2 28.5±0.6 10.7±0.6 35.9±1.3 41.6±1.0 29.4±1.1 15.9±0.8
This tests whether the direction encodes source se-
lection rather than lexical or semantic features of
one relation. The off-diagonal effects in Figure 5
show transfer across all six knowledge categories;
Appendix I.2 provides the other-model heatmaps.
Cross-dataset transfer.We apply the direction
learned on NQ-Swap to distinct factual-conflict
datasets without target-side refitting. These dataset
shifts introduce variations in conflict construction,
factual content, and other nuisance factors, reduc-
ing the possibility that the observed effect is tied to
dataset-specific properties. Table 6 demonstrates
bidirectional transfer across datasets, while context-
only controls achieve consistently weaker effects.
5.2 To Complex, Application-Relevant Tasks
We next evaluate whether a direction learned from
explicit factual conflicts transfers to broader input-
grounded tasks without target-task direction fitting.
Unlike the controlled benchmarks, these tasks do
not necessarily involve direct competition between
parametric and contextual answers.
RAG.RAG tasks provideretrieved documents:
RGB supplies deliberately counterfactual evidence,
whereas CRAG uses retrieved passages that can be
long, stale, noisy, or only partly relevant. Success
requires selecting the supplied evidence when it
conflicts with a plausible parametric answer.Memory.Memory tasks providemulti-turn his-
tories. In LoCoMo, LoCoMo-Plus, and Long-
MemEval, the relevant fact may lie many turns
from the question and compete with a continuation,
stale association, or another event. These tasks
test retrieval, localization, and temporal ordering
of knowledge rather than passage matching.
Rule-following.Rule-following tasks provideex-
plicit behavioral constraints. IFEval and Follow-
Bench require the model to retain rules about
output structure, length, required content, and
exclusions throughout generation; they serve as
instruction-following controls for whether the di-
rection improves use of contextual constraints.
Results.Table 7 reports held-out unsteered and
steered scores using source matching for controlled
factual tasks and each benchmark’s native eval-
uator otherwise. The corresponding item-level
changes and error-stratified cases are reported in
Appendix G, including the case tables.
6 Conclusion
We show that a single residual-stream direction
enables smooth, bidirectional selection between
parametric knowledge and external context. Chang-
ing its signed coefficient at one carefully selected
layer shifts both aggregate source-match rates and
8

fixed-instance responses.
The control generalizes across conflict for-
mats, factual relations, and datasets, and transfers
to RAG, multi-turn memory, and rule-following
tasks without target-task direction fitting. Weaker
context-attention controls suggest that transfer re-
flects source competition. These results provide
evidence for a transferable control direction under-
lying knowledge-source selection.
Limitations
First, our method requires white-box access to
model activations. Since the direction is estimated
and applied in the residual stream, it is not directly
applicable to closed-source or API-only models.
Second, the learned direction should be viewed
as a useful control axis rather than a complete mech-
anism of knowledge selection. Although a single
direction enables effective steering, source selec-
tion may still involve broader circuits, including
attention heads, MLP features, and decoding dy-
namics.
Third, our evaluation is based mainly on con-
trolled factual-conflict benchmarks and source-
choice accuracy. This setup isolates parametric ver-
sus contextual reliance, but may not capture more
nuanced outputs such as hedging, partial agree-
ment, or answers that combine both sources.
Finally, the direction improves the complex tasks
we evaluate, but the gains are often smaller than
in controlled factual conflicts and depend on the
underlying capabilities of each model. Establish-
ing how model competence, task difficulty, and
evidence quality jointly determine these effects re-
mains future work.
Acknowledgments
This work is supported by CCF-Tencent Rhino-
Bird Open Research Fund (Grant No. CCF-Tencent
RAGR20260112).
References
Nikhil Anand, Shwetha Somasundaram, Anirudh
Phukan, Apoorv Saxena, and Koyel Mukherjee. 2026.
Contextfocus: Activation steering for contextual
faithfulness in large language models.Preprint,
arXiv:2601.04131.
Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka,
Nina Panickssery, Wes Gurnee, and Neel Nanda.
2024. Refusal in language models is mediated by
a single direction.Preprint, arXiv:2406.11717.Baolong Bi, Shenghua Liu, Yiwei Wang, Yilong Xu,
Junfeng Fang, Lingrui Mei, and Xueqi Cheng. 2025.
Parameters vs. context: Fine-grained control of
knowledge reliance in language models.Preprint,
arXiv:2503.15888.
Arie Cattan, Alon Jacovi, Ori Ram, Jonathan Herzig,
Roee Aharoni, Sasha Goldshtein, Eran Ofek, Idan
Szpektor, and Avi Caciularu. 2025. Dragged into con-
flicts: Detecting and addressing conflicting sources in
search-augmented llms.Preprint, arXiv:2506.08500.
Jiawei Chen, Hongyu Lin, Xianpei Han, and
Le Sun. 2023. Benchmarking large language mod-
els in retrieval-augmented generation.Preprint,
arXiv:2309.01431.
Sitao Cheng, Liangming Pan, Xunjian Yin, Xinyi Wang,
and William Yang Wang. 2024. Understanding
the interplay between parametric and contextual
knowledge for large language models.Preprint,
arXiv:2410.08414.
Eunseong Choi, June Park, Hyeri Lee, and Jongwuk Lee.
2025. Conflict-aware soft prompting for retrieval-
augmented generation.Preprint, arXiv:2508.15253.
Nelson Elhage, Tristan Hume, Catherine Olsson,
Nicholas Schiefer, Tom Henighan, Shauna Kravec,
Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain,
Carol Chen, Roger Grosse, Sam McCandlish, Jared
Kaplan, Dario Amodei, Martin Wattenberg, and
Christopher Olah. 2022. Toy models of superpo-
sition.Preprint, arXiv:2209.10652.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,
and Haofen Wang. 2024. Retrieval-augmented gener-
ation for large language models: A survey.Preprint,
arXiv:2312.10997.
Mor Geva, Jasmijn Bastings, Katja Filippova, and Amir
Globerson. 2023. Dissecting recall of factual associ-
ations in auto-regressive language models.Preprint,
arXiv:2304.14767.
Giwon Hong, Jeonghwan Kim, Junmo Kang, Sung-
Hyon Myaeng, and Joyce Jiyoung Whang. 2024.
Why so gullible? enhancing the robustness of
retrieval-augmented models against counterfactual
noise.Preprint, arXiv:2305.01579.
Youcheng Huang, Chen Huang, Duanyu Feng, Wen-
qiang Lei, and Jiancheng Lv. 2025. Cross-model
transferability among large language models on
the platonic representations of concepts.Preprint,
arXiv:2501.02009.
Albert Q. Jiang, Alexandre Sablayrolles, Arthur Men-
sch, Chris Bamford, Devendra Singh Chaplot, Diego
de las Casas, Florian Bressand, Gianna Lengyel, Guil-
laume Lample, Lucile Saulnier, Lélio Renard Lavaud,
Marie-Anne Lachaux, Pierre Stock, Teven Le Scao,
Thibaut Lavril, Thomas Wang, Timothée Lacroix,
and William El Sayed. 2023. Mistral 7b.Preprint,
arXiv:2310.06825.
9

Yuxin Jiang, Yufei Wang, Xingshan Zeng, Wanjun
Zhong, Liangyou Li, Fei Mi, Lifeng Shang, Xin
Jiang, Qun Liu, and Wei Wang. 2024. Follow-
Bench: A multi-level fine-grained constraints fol-
lowing benchmark for large language models. In
Proceedings of the 62nd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 4667–4688, Bangkok, Thailand.
Association for Computational Linguistics.
Zhuoran Jin, Pengfei Cao, Hongbang Yuan, Yubo Chen,
Jiexin Xu, Huaijun Li, Xiaojian Jiang, Kang Liu,
and Jun Zhao. 2024. Cutting off the head ends the
conflict: A mechanism for interpreting and mitigating
knowledge conflicts in language models.Preprint,
arXiv:2402.18154.
Gaotang Li, Yuzhong Chen, and Hanghang Tong. 2025.
Taming knowledge conflicts in language models.
Preprint, arXiv:2503.10996.
Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter
Pfister, and Martin Wattenberg. 2024. Inference-time
intervention: Eliciting truthful answers from a lan-
guage model.Preprint, arXiv:2306.03341.
Shayne Longpre, Kartik Perisetla, Anthony Chen,
Nikhil Ramesh, Chris DuBois, and Sameer Singh.
2022. Entity-based knowledge conflicts in question
answering.Preprint, arXiv:2109.05052.
Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov,
Mohit Bansal, Francesco Barbieri, and Yuwei Fang.
2024. Evaluating very long-term conversational
memory of llm agents.Preprint, arXiv:2402.17753.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,
Daniel Khashabi, and Hannaneh Hajishirzi. 2023.
When not to trust language models: Investigating
effectiveness of parametric and non-parametric mem-
ories.Preprint, arXiv:2212.10511.
Samuel Marks and Max Tegmark. 2024. The geometry
of truth: Emergent linear structure in large language
model representations of true/false datasets.Preprint,
arXiv:2310.06824.
Ian McKenzie, Alexander Lyzhov, Alicia Parrish,
Ameya Prabhu, Aaron Mueller, Najoung Kim, Sam
Bowman, and Ethan Perez. 2022. The inverse scaling
prize.
Kevin Meng, David Bau, Alex Andonian, and Yonatan
Belinkov. 2023. Locating and editing factual associa-
tions in gpt.Preprint, arXiv:2202.05262.
Thomas Merth, Qichen Fu, Mohammad Rastegari, and
Mahyar Najibi. 2024. Superposition prompting: Im-
proving and accelerating retrieval-augmented genera-
tion.Preprint, arXiv:2404.06910.
Julian Minder, Kevin Du, Niklas Stoehr, Giovanni
Monea, Chris Wendler, Robert West, and Ryan Cot-
terell. 2025. Controllable context sensitivity and
the knob behind it. InInternational Conference
on Learning Representations, volume 2025, pages
81073–81102.Nina Panickssery, Nick Gabrieli, Julian Schulz, Meg
Tong, Evan Hubinger, and Alexander Matt Turner.
2024. Steering llama 2 via contrastive activation
addition.Preprint, arXiv:2312.06681.
Kiho Park, Yo Joong Choe, and Victor Veitch.
2024. The linear representation hypothesis and
the geometry of large language models.Preprint,
arXiv:2311.03658.
Dan Shi, Renren Jin, Tianhao Shen, Weilong Dong,
Xinwei Wu, and Deyi Xiong. 2024. Ircan: Mitigating
knowledge conflicts in llm generation via identifying
and reweighting context-aware neurons.Preprint,
arXiv:2406.18406.
Nicholas Sofroniew, Isaac Kauvar, William Saunders,
Runjin Chen, Tom Henighan, Sasha Hydrie, Craig
Citro, Adam Pearce, Julius Tarng, Wes Gurnee,
Joshua Batson, Sam Zimmerman, Kelley Rivoire,
Kyle Fish, Chris Olah, and Jack Lindsey. 2026. Emo-
tion concepts and their function in a large language
model.Transformer Circuits Thread.
Alessandro Stolfo, Vidhisha Balachandran, Safoora
Yousefi, Eric Horvitz, and Besmira Nushi. 2025. Im-
proving instruction-following in language models
through activation steering. InInternational Con-
ference on Learning Representations, volume 2025,
pages 55790–55823.
Daniel Tan, David Chanin, Aengus Lynch, Dimitrios
Kanoulas, Brooks Paige, Adria Garriga-Alonso,
and Robert Kirk. 2025. Analyzing the generaliza-
tion and reliability of steering vectors.Preprint,
arXiv:2407.12404.
Gemma Team. 2024. Gemma.
Qwen Team. 2025. Qwen3 technical report.Preprint,
arXiv:2505.09388.
Alexander Matt Turner, Lisa Thiergart, Gavin Leech,
David Udell, Juan J. Vazquez, Ulisse Mini, and
Monte MacDiarmid. 2024. Steering language
models with activation engineering.Preprint,
arXiv:2308.10248.
Han Wang, Archiki Prasad, Elias Stengel-Eskin, and
Mohit Bansal. 2025a. Adacad: Adaptively decoding
to balance conflicts between contextual and paramet-
ric knowledge.Preprint, arXiv:2409.07394.
Mengru Wang, Yunzhi Yao, Ziwen Xu, Shuofei Qiao,
Shumin Deng, Peng Wang, Xiang Chen, Jia-Chen Gu,
Yong Jiang, Pengjun Xie, Fei Huang, Huajun Chen,
and Ningyu Zhang. 2024. Knowledge mechanisms
in large language models: A survey and perspective.
Preprint, arXiv:2407.15017.
Yilin Wang, Heng Wang, Yuyang Bai, and Minnan
Luo. 2025b. Continuously steering llms sensitivity to
contextual knowledge with proxy models.Preprint,
arXiv:2508.19720.
10

Yotam Wolf, Noam Wies, Dorin Shteyman, Binyamin
Rothberg, Yoav Levine, and Amnon Shashua. 2025.
Tradeoffs between alignment and helpfulness in lan-
guage models with steering methods.Preprint,
arXiv:2401.16332.
Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang,
Kai-Wei Chang, and Dong Yu. 2025. Longmemeval:
Benchmarking chat assistants on long-term interac-
tive memory.Preprint, arXiv:2410.10813.
Rongwu Xu, Zehan Qi, Zhijiang Guo, Cunxiang Wang,
Hongru Wang, Yue Zhang, and Wei Xu. 2024.
Knowledge conflicts for llms: A survey.Preprint,
arXiv:2403.08319.
Xiao Yang, Kai Sun, Hao Xin, Yushi Sun, Nikita Bhalla,
Xiangsen Chen, Sajal Choudhary, Rongze Daniel
Gui, Ziran Will Jiang, Ziyu Jiang, Lingkun Kong,
Brian Moran, Jiaqi Wang, Yifan Ethan Xu, An Yan,
Chenyu Yang, Eting Yuan, Hanwen Zha, Nan Tang,
and 8 others. 2024. Crag – comprehensive rag bench-
mark.Preprint, arXiv:2406.04744.
Long Zhang and Fangwei Lin. 2026. Simulated
adoption: Decoupling magnitude and direction
in llm in-context conflict resolution.Preprint,
arXiv:2602.04918.
Yu Zhao, Alessio Devoto, Giwon Hong, Xiaotang
Du, Aryo Pradipta Gema, Hongru Wang, Xuanli
He, Kam-Fai Wong, and Pasquale Minervini. 2025a.
Steering knowledge selection behaviours in llms
via sae-based representation engineering.Preprint,
arXiv:2410.15999.
Yu Zhao, Xiaotang Du, Giwon Hong, Aryo Pradipta
Gema, Alessio Devoto, Hongru Wang, Xuanli
He, Kam-Fai Wong, and Pasquale Minervini.
2025b. Analysing the residual stream of lan-
guage models under knowledge conflicts.Preprint,
arXiv:2410.16090.
Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha
Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and
Le Hou. 2023. Instruction-following evaluation for
large language models.Preprint, arXiv:2311.07911.
Andy Zou, Long Phan, Sarah Chen, James Campbell,
Phillip Guo, Richard Ren, Alexander Pan, Xuwang
Yin, Mantas Mazeika, Ann-Kathrin Dombrowski,
Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan
Wang, Alex Mallen, Steven Basart, Sanmi Koyejo,
Dawn Song, Matt Fredrikson, and 2 others. 2025.
Representation engineering: A top-down approach
to ai transparency.Preprint, arXiv:2310.01405.List of appendices
A Dataset Details 13
BIntervention Location and Representa-
tion Validation 13
B.1 Representation Hook . . . . . . . 13
B.2 Layer Selection . . . . . . . . . . 14
B.3 Context-Sensitivity Direction Sim-
ilarity . . . . . . . . . . . . . . . 14
B.4 Token Position . . . . . . . . . . 14
C Judge-Based Evaluation Protocol 14
C.1 Answer Normalization . . . . . . 14
C.2 Answer Window . . . . . . . . . 14
C.3 Match Rules . . . . . . . . . . . . 15
C.4 LLM-Judge Validation . . . . . . 15
C.5 Complex-Task Evaluators . . . . . 15
D Implementation Details 16
E Steering-Strength Sensitivity Analysis 16
F Scaling to Larger Models 16
GInstance-Level Changes and Failure
Analysis 16
G.1 Controlled Factual-Conflict
Instance-Level Changes . . . . . . 18
G.2 Complex-Task Instance-Level
Changes . . . . . . . . . . . . . . 18
G.3 Failure Analysis . . . . . . . . . . 18
H Prompt Templates 23
IGeneralization Within Simple Factual-
Conflict Tasks 25
I.1 Cross-Format Transfer . . . . . . 25
I.2 Cross-Knowledge-Type Transfer . 25
11

I.3 Cross-Dataset Transfer . . . . . . 27
12

Appendices
A Dataset Details
We briefly describe how each benchmark is con-
verted into our source-choice setting. For all
datasets, we first keep examples where the model
answers the clean prompt correctly. We then evalu-
ate the same examples under a conflict prompt that
supports an alternative answer. The original an-
swer is treated as theparametricanswer, while the
answer supported by the conflict prompt is treated
as thecontextualanswer. Prompt examples are
provided in Appendix H.
NQ-Swap.NQ-Swap pairs Natural Questions
examples with substituted passages. We use the
original answer as the parametric answer and the
substituted answer as the contextual answer. The
clean prompt contains only the question, while the
conflict prompt prepends the substituted context
before the same question. We use the question-
deduplicated split without tables or list items.
MacNoise.MacNoise provides QA examples
with perturbed retrieved contexts. The original
answer is used as the parametric answer, and the re-
placement answer is used as the contextual answer.
For each example, we select the first perturbed con-
text marked by pert_lbls_100 == 1 and prepend
itstext_pertfield to form the conflict prompt.
ParaConflict.ParaConflict contains synthetic
factual conflicts across multiple categories. The
gold answer is treated as the parametric answer,
and the distracted token is treated as the contex-
tual answer. We use the coherent conflict for-
mat in the main experiments because it provides
a natural-language passage supporting the dis-
tractor. For cross-format transfer, we train on
substitution-format items and evaluate on coherent-
format items.
CounterFact.CounterFact contains factual edit-
ing examples grouped by Wikidata relation. The
true target is treated as the parametric answer, and
the edited target is treated as the contextual an-
swer. The conflict prompt is constructed by insert-
ing the edited target into a paraphrased prefix and
appending the original factual query. For cross-
knowledge-type transfer, we evaluate six relations:
country of citizenship, manufacturer, occupation,
religion, sport, and original broadcaster.MemoTrap.MemoTrap creates conflicts be-
tween memorized phrase completions and explicit
prompt instructions. The familiar phrase contin-
uation is treated as the parametric answer, while
the instruction-satisfying completion is treated as
the contextual answer. The clean prompt removes
the explicit ending constraint, whereas the conflict
prompt keeps the original instruction.
RGB and CRAG.RGB supplies retrieved, de-
liberately counterfactual passages whose answer
conflicts with a familiar factual answer; we evaluate
whether the output follows the supplied evidence.
CRAG supplies retrieved web-style passages for
factual questions, including cases with stale, com-
peting, noisy, or partially relevant evidence. We
use each benchmark’s held-out evaluation items
and its factual source-match protocol.
LoCoMo, LoCoMo-Plus, and LongMemEval.
These datasets require answers grounded in multi-
turn personal histories. LoCoMo and LoCoMo-
Plus test retrieval and linkage across earlier di-
alogue turns, while LongMemEval additionally
stresses temporal ordering and aggregation across
events. We preserve each benchmark’s native held-
out split and exact-match evaluator.
IFEval and FollowBench.These instruction-
following datasets specify explicit requirements
on output structure, length, content, or lexical ex-
clusions. We use their native constraint-compliance
evaluators; they serve as controls for retaining and
applying external instructions across the generated
response.
B Intervention Location and
Representation Validation
We ablate each modeling choice underlying our
main pipeline (Llama-3.1-8B-Instruct, NQ-Swap
test unless noted). For each variant we re-estimate
ˆvandℓwhen required, re-tune (α+, α−)on vali-
dation, and report Acc parand Acc ctxon the test
split.
B.1 Representation Hook
Table 8 shows that resid post yields stronger bidirec-
tional control; resid mid generally underperforms,
indicating that the MLP sublayer contributes to the
separable knowledge-selection signal.
13

Table 8: Steered source-choice accuracy (%) on NQ-
Swap test across all evaluated models. Acc parand
Accctxreport source-choice accuracy when steering to-
ward parametric and contextual knowledge, respectively.
Each steered cell reportsresid-mid/resid-post(best α
per direction).
Model Acc par Accctx
Llama-3.1-8B-Instruct 50.4 /56.8 59.7/ 55.5
Qwen3-8B62.4/ 54.6 51.1 /56.0
Gemma-2-9B-IT 33.8 /80.953.2 /65.4
Mistral-7B-Instruct-v0.359.0/ 59.0 71.1 /60.2
0 5 10 15 20 25 30
Layer0.8250.8500.8750.9000.9250.9500.975Cosine similarity
llama3-8b-it
Llama-3.1-8B-Instruct
0 5 10 15 20 25 30 35
Layer0.900.920.940.960.981.00Cosine similarity
qwen3-8b Qwen3-8B
0 5 10 15 20 25 30 35 40
Layer0.880.900.920.940.960.981.00Cosine similarity
gemma2-9b-it
Gemma-2-9B-Instruct
0 5 10 15 20 25 30
Layer0.850.900.95Cosine similarity
mistral-7b-instruct-v0.3 Mistral-7B-Instruct-v0.3
Figure 6: Layer-wise cosine similarity si
Tclean,Tconflict
between the respective final-token activations of clean
and conflict prompts ((2)) on NQ-Swap.
B.2 Layer Selection
Instead of steering only at the selected layer ℓ, we
inject the intervention at each layer iseparately and
examine the resulting performance across all four
models on NQ-Swap. Figure 6 reports the cosine
curves used for localization, while Figure 7 shows
Acc parand Acc ctxas a function of the interven-
tion layer. Across models, the layer-wise steering
performance follows a similar trend to the angular
separation scores: layers with larger separation gen-
erally yield stronger steering effects, while shallow
and very late layers show weaker or less stable con-
trol. This alignment supports our layer-selection
strategy and suggests that the angular separation
score is a useful indicator of where knowledge se-
lection is most steerable.
Across the completed sweeps, peaks align with
the angular-separation layer; shallow and late lay-
ers show weaker or asymmetric control.
B.3 Context-Sensitivity Direction Similarity
Table 9 reports cosine similarities among the three
no-context contrasts used in the matched context-
sensitivity controls. A denotes no context; B, C,Table 9: Cosine similarity between the three matched
no-context contrasts on NQ-Swap.
Contrast pair Cosine similarity
A–C vs. A–B 0.899399
A–C vs. A–D 0.878706
A–B vs. A–D 0.874608
Table 10: Token-position ablation on NQ-Swap test.
Accparand Acc ctxreport source-choice accuracy when
steering toward parametric and contextual knowledge,
respectively. Each entry isquestion-final / answer-final
token, using validation-selected coefficients at that posi-
tion.
Model Acc par Accctx
Llama-3.1-8B-Instruct 56.8 /65.6 55.5/ 54.0
Qwen3-8B54.6/ 47.156.0/ 44.5
Gemma-2-9B-IT80.9/ 75.365.4/ 62.3
Mistral-7B-Instruct-v0.3 59.0 /63.9 60.2/ 57.8
and D denote aligned, conflicting, and irrelevant
context.
B.4 Token Position
As shown in Table 10, the question-final token gen-
erally yields stronger bidirectional control than the
answer-final token, although answer-final interven-
tion can perform better for parametric steering in a
few models.
C Judge-Based Evaluation Protocol
Acc parand Acc ctxdepend on how free-form out-
puts are mapped to discrete source labels. This
appendix makes the protocol fully explicit.
C.1 Answer Normalization
Each gold parametric answer aparand contextual
answera ctxis normalized before matching:
1. lowercase;
2.remove punctuation except intra-token hyphens;
3. remove leading articles (a,an,the);
4. collapse whitespace;
C.2 Answer Window
We take the first256 generated tokensof the raw
continuation and normalize this fixed answer win-
dow. This avoids making the metric depend on sen-
tence segmentation or on whether steering changes
the placement of a sentence boundary. Outputs
shorter than 256 tokens are matched in full.
14

0 5 10 15 20 25 30
Layer0.00.2 rate vs =0
Llama-3-8B-IT / NQ-Swap
internal
external(a) Llama-3.1-8B-Instruct
0 5 10 15 20 25 30 35
Layer0.00.10.2 rate vs =0
Qwen3-8B / NQ-Swap
internal
external (b) Qwen3-8B
0 5 10 15 20 25 30 35 40
Layer0.00.20.4 rate vs =0
Gemma2-9B-IT / NQ-Swap
internal
external
(c) Gemma-2-9B-IT
0 5 10 15 20 25 30
Layer0.00.10.2 rate vs =0
Mistral-7B-Instruct-v0.3 / NQ-Swap
internal
external (d) Mistral-7B-Instruct-v0.3
Figure 7: Layer-selection analysis on NQ-Swap. Each panel shows performance when the intervention is applied at
different layers for one model. The layer-wise performance trend is consistent with the cosine-based separation
pattern: layers with lower clean–conflict cosine similarity, corresponding to larger angular separation, tend to
produce stronger steering effects.
C.3 Match Rules
Letsbe the normalized 256-token answer window.
The four labels below are mutually exclusive and
exhaustive; both/neither outputs are retained in the
denominator rather than discarded.
•Parametric-only: scontains aparas a substring,
or an alias thereof, and doesnotcontaina ctx.
•Contextual-only: scontains actxand doesnot
containa par.
•Both:scontains both answers.
•Neither: scontains neither answer, or the output
is empty/refusal/off-topic.
Long answers are handled by substring search
within the normalized answer window only; tokens
after position 256 are not searched. This opera-
tional metric measures answer-string source match-
ing, not full semantic entailment; Appendix C.5
therefore describes the benchmark-native semantic
or constraint evaluators used for complex tasks.
C.4 LLM-Judge Validation
String matching permits deterministic, low-cost
evaluation across the full alpha sweep. To audit
its construct validity, we sample 100 OOD outputs
per dataset. Given the prompt, both candidate an-
swers, and the complete generation, a blinded LLM
judge assigns one of the same four labels without
seeing the string-match result. Table 11 compares
the resulting four-way distributions; the close cor-
respondence supports using string matching for theTable 11: Four-way label distributions (%) from normal-
ized string matching and a blinded LLM judge on 100
OOD outputs per dataset. The order is correct answer /
conflicting distractor / both / neither.
Dataset String matching LLM judge
ParaConflict 57 / 9 / 34 / 0 59 / 10 / 30 / 1
MacNoise 77 / 5 / 1 / 17 74 / 5 / 8 / 13
MemoTrap 48 / 15 / 32 / 5 50 / 17 / 30 / 3
CounterFact 53 / 20 / 23 / 4 54 / 24 / 18 / 4
full sweeps.
C.5 Complex-Task Evaluators
For each model, all complex-task experiments use
its direction learned on NQ-Swap, injected at that
model’s selected layer with coefficients fixed from
NQ-Swap validation; no target-task direction, coef-
ficient, or layer is estimated or relocalized. For
each task, all coefficients are evaluated on the
same deterministic sample subset, and we report
the mean official per-item score times 100. These
downstream scores assess input-grounded behavior:
except for RGB, a correct output alone does not
establish whether the unsteered error arose from
factual priors, evidence localization, reasoning, or
response generation.
Sampling and prompts.We evaluate at most
500 processed items per task, sampling without
replacement with seed 43 when a pool exceeds
500 items; smaller pools are evaluated in full. The
15

model receives only the task-supplied context, and
credit is assigned against the contextual target or
benchmark constraint. Thus, complex-task scores
do not use the 256-token source-match protocol
above.
RAG evaluators.RGB contains 100 counterfac-
tual retrieval items. Up to three supplied passages
support afakeanswer, and its official binary judge
scores that contextual target rather than the world-
factual answer. For CRAG, we sample 500 of 2,698
development items; ranked snippets are HTML-
stripped, title-prefixed when available, concate-
nated in retrieval order, and capped at 12,000 char-
acters. Its released evaluator handles exact, invalid,
and semantic-answer cases.
Memory evaluators.For LoCoMo, 500 of 1,542
QA rows are sampled and the official category-
aware multi-hop or token-F1 evaluator is averaged.
LoCoMo-Plus evaluates all 401 cognitive items
with its released binary constraint-consistency
judge. LongMemEval evaluates its 500-item or-
acle split in full, caps dated relevant sessions at
12,000 characters, and uses its released answer-
check judge conditioned on question type.
Rule-following evaluators.For IFEval, 500 of
541 prompts are sampled and its released strict
evaluator gives one only when every instruction is
satisfied. For FollowBench, 500 of 820 items are
sampled; released template, rule-based, or discrim-
inative evaluators give hard per-item satisfaction,
and we report the macro average across six con-
straint categories.
D Implementation Details
All experiments are inference-only; we do not
fine-tune any model. We run each model on a
single NVIDIA A100 GPU with 80GB memory
usingbfloat16 precision and the Hugging Face
transformers library. Unless otherwise speci-
fied, decoding is greedy with do_sample=False
andmax_new_tokens=200 . The base evaluation
cohort consists of examples where the model
answers the clean prompt correctly and is then
evaluated under the conflict prompt, denoted as
stage1_correct_use_wrong_context . When
subsampling is required, we use random seed 42.
E Steering-Strength Sensitivity Analysis
For the Llama-3.1-8B-Instruct direction transferred
from NQ-Swap, we report the selected coefficientTable 12: Model checkpoints and selected intervention
layers.
Model Size Local key Layer
Llama-3.1-8B-Instruct 8Bllama3-8b-it10
Gemma-2-9B-IT 9Bgemma2-9b-it16
Qwen3-8B 8Bqwen3-8b21
Mistral-7B-Instruct-v0.3 7Bmistral-7b-instruct-v0.316
Table 13: Inference and tokenization settings.
Setting Value
Framework Hugging Facetransformers
Precisionbfloat16
Hardware 1×NVIDIA A100 80GB
Attention implementationeager
Decoding Greedy,do_sample=False
max_new_tokens200
Tokenizermax_length8192
Padding side Left padding
pad_token eos_tokenif unset
Batch size 128
Batch size for NQ-Swap 64
GPUs per job 1, set byCUDA_VISIBLE_DEVICES
Random seed 42 when subsampling is used
and the two nearest evaluated values on either side
when available. Table 14 gives parametric-answer
rate for positive steering and distractor/context-
answer rate for negative steering. No target-side
coefficient is reselected.
F Scaling to Larger Models
We further examine whether the same one-
dimensional steering behavior holds for larger
instruction-tuned models. Figure 8 reports source-
choice rates on NQ-Swap and MacNoise for four
larger instruction-tuned models. Across most mod-
els and datasets, increasing αraises the internal, or
parametric, source-choice rate while reducing the
external, or contextual, source-choice rate. This
consistent trend indicates that the learned direc-
tion remains effective beyond the smaller mod-
els used in the main experiments, suggesting that
knowledge selection can be continuously modu-
lated along a similar residual-stream direction even
at larger model scales.
G Instance-Level Changes and Failure
Analysis
This audit classifies baseline context-conflict er-
rors that an available positive- αsweep changed
into an officially correct output. Here, knowledge
conflict is used broadly: a model default (factual
16

Table 14: Cross-dataset operating-point stability for Llama-3.1-8B-Instruct. Each cell gives α: held-out source-
match rate (%) followed by the number of valid Stage-2 generations in parentheses. The selected transfer coefficient
is bold; “–” indicates that a farther neighboring grid point was not evaluated.
Setting ParaConflict MacNoise MemoTrap CounterFact
Positive direction: parametric-answer rate
α= 015.9 (1711) 46.7 (1047) 67.0 (624) 36.1 (2160)
Lower 2 1.2: 82.3 (1701) 1.2: 71.1 (849) 2.4: 83.7 (509) 2.6: 43.7 (1854)
Lower 1 1.4: 84.7 (1697) 1.4: 72.1 (795) 2.6: 82.7 (490) 2.8: 43.6 (1803)
Selected1.6: 85.6 (1687) 1.6: 72.4 (749) 2.8: 85.6 (465) 3.0: 48.1 (1565)
Higher 1 1.8: 84.3 (1674) 1.8: 70.4 (689) 3.0: 83.1 (450) –
Higher 2 2.0: 81.6 (1662) 2.0: 62.3 (159) – –
Negative direction: distractor/context-answer rate
α= 081.8 (1711) 27.3 (1047) 30.1 (624) 60.8 (2160)
Lower 2−1.8: 98.3 (1264)−1.8: 23.4 (124) –−1.4: 61.6 (1914)
Lower 1−1.6: 98.6 (1280)−1.6: 21.1 (147)−1.8: 57.6 (580)−1.2: 62.9 (1957)
Selected−1.4: 98.8 (1292)−1.4: 43.8 (706)−2.0: 58.0 (571)−1.0: 63.0 (2005)
Higher 1−1.2: 96.8 (1528)−1.2: 43.1 (796) –−0.8: 62.8 (2038)
Higher 2−1.0: 96.9 (1578)−1.0: 42.2 (855) –−0.6: 62.8 (2065)
-1.0-0.50.00.51.01.52.0
0.00.5Rate
NQ-SwapLlama-3.1-70B-Instruct
-1.0-0.50.00.51.01.5
0.000.250.50
Qwen3-32B
0 1 2 3
0.00.5
Gemma-2-27B-IT
-1.0 -0.5 0.0 0.5 1.0
0.00.5
Yi-34B
-0.5 0.00.51.0
0.00.5Rate
MacNoise
-2.0-1.5-1.0-0.50.00.51.0
0.000.250.50
-2 -1 0 1 2 3
0.00.5
-1.0-0.50.00.51.01.5
0.00.5
internal external
Figure 8: Source-choice trajectories on larger models. Blue curves denote internal, or parametric, source-choice
rates, while red curves denote external, or contextual, source-choice rates. Across NQ-Swap and MacNoise,
increasing αgenerally shifts the model from external reliance toward internal reliance, showing that the steering
direction remains effective for larger models.
17

prior, continuation, event ordering, or output habit)
can conflict with information or constraints sup-
plied by the context. The alpha shown for each
example is the first observed correct positive-alpha
output, not an item-level selection rule. “Paramet-
ric knowledge” is an inference from the baseline
text, not a claim about an unobserved model state.
We use three mechanism labels:Pfor an explicit
override of supplied evidence by a real-world or
general-knowledge correction;Lfor missed, mis-
ordered, or underspecified supplied context; andI
for a requested format or constraint that the context
requires.
G.1 Controlled Factual-Conflict
Instance-Level Changes
Each trajectory contains an unsteered output, an
intermediate output containing both answer strings,
and an output selecting the requested source.
Llama, Qwen, and Gemma cases use saved NQ-
Swap test alpha grids. Mistral retained only three
final NQ-Swap coefficients, so its cases use the
saved complete CounterFact within-relation grid.
Outputs are answer-bearing excerpts;BOTHde-
notes a string-match label, not a claim that the full
generation provides a coherent answer.
Llama-3.1-8B-Instruct.Table 15 gives three
parametric and three contextual trajectories for the
main model, illustrating the transition from an ini-
tial source choice through a mixed string-match
output to the requested source.
Qwen3-8B.Table 16 repeats the same trajectory
audit for Qwen3-8B, showing that both source di-
rections can be expressed through intermediate
mixed outputs.
Gemma-2-9B-IT.Table 17 reports the corre-
sponding Gemma trajectories; the examples in-
clude both clean source switches and cases where
the model retains a competing answer in the inter-
mediate generation.
Mistral-7B-Instruct-v0.3.Table 18 reports Mis-
tral’s within-relation CounterFact trajectories, us-
ing the saved CounterFact grid because a complete
NQ-Swap alpha sweep was unavailable for this
model.
G.2 Complex-Task Instance-Level Changes
The audit is reported in Table 20 and Table 21,
with one row per representative item. The labelsdistinguish explicit parametric overrides (P), local-
ization or reasoning over supplied context (L), and
instruction constraints (I). RGB provides synthetic
counterfactual conflicts; CRAG mixes evidence se-
lection with temporal, arithmetic, and comparison
cases; the memory benchmarks distribute the rel-
evant fact across earlier turns; and IFEval/Follow-
Bench test contextual constraints on output form.
G.2.1 Gain stratification and limitations
The current sweeps support a broad context-conflict
account spanning factual evidence, distributed
memory, and instruction constraints, while the
mechanism-level comparison remains qualitative.
A stronger test would sample 50–100 CRAG
baseline errors before inspecting steered outputs,
independently label evidence spans and confi-
dence, freeze αon validation data, and report per-
mechanism test gains with bootstrap confidence
intervals.
G.3 Failure Analysis
We analyze representative failure cases of steering.
Here,PARAMETERdenotes the gold or internally
stored answer,CONTEXTdenotes the distractor sup-
ported by the conflict prompt,BOTHdenotes out-
puts containing both answers, andOTHERdenotes
outputs containing neither.
G.3.1 Failure to Reach the Target Source
In this category, steering does not move the first an-
swer entity to the intended knowledge source. For
positive steering, the output remains contextual; for
negative steering, the output remains parametric.
Example 1.
Question: George V Coast is a part of the con-
tinent of
Context:Africa
Parameter: Antarctica
αOutput Label
0“Africa.”CONTEXT
+0.2“Africa.”CONTEXT
+1.0“Africa.”CONTEXT
+2.4 “Africa, and it is a region that is
rich in natural resources.”CONTEXT
+3.0 “Africa, and it is a region of great
scientific interest due to its unique
geology and geography.”CONTEXT
The conflict prompt anchors the first answer to
Africa; even strong positive steering fails to recover
the parametric answerAntarctica.
Example 2.
18

Table 15: Instance-level continuous trajectories for Llama-3.1-8B-Instruct. P and C denote the requested parametric
and contextual sources.
Question Generation trajectory
P: requested parametric source
When did the first Ninja Turtles
come out?Initial: 1999.
Intermediate: The first comic was published in 1984, not 1999.
Target: 1984.
Who designed the Earth Day flag in
1969?Initial: Andreas Vesalius.
Intermediate: Vesalius . . . likely John McConnell.
Target: John McConnell.
Who was in theDon’t Worry, Be
Happyvideo?Initial: CeCe Drake.
Intermediate: CeCe Drake. Bobby McFerrin . . .
Target: Bobby McFerrin.
C: requested contextual source
When wasEdward Tulanepub-
lished?Initial: The novel is from 2006.
Intermediate: 15th century . . . a 21st-century novel.
Target: 15th century.
How many episodes is season 4 of
The Flash?Initial: 23 episodes.
Intermediate: The correct answer should be 23, not 360.
Target: 360 episodes.
When was the $1,000 bill discontin-
ued?Initial: 1969.
Intermediate: 2017 . . . discontinued in 1969.
Target: 2017.
Table 16: Instance-level continuous trajectories for Qwen3-8B. P and C denote the requested parametric and
contextual sources.
Question Generation trajectory
P: requested parametric source
When didThe Karate Kidcome
out?Initial: The text says the remake was in 1835.
Intermediate: The original was 1984, and the remake was 2010.
Target: The remake was released in 2010.
How many grams of alcohol are in
one beer?Initial: The text says a standard drink contains 1612 grams.
Intermediate: The correct value should be 14 grams, not 1612.
Target: A standard drink contains about 14 grams.
Industrial city on Germany’s Rhine–
Herne Canal?Initial: The canal connects near Hawaii.
Intermediate: It connects near Hawaii . . . Duisburg.
Target: It connects the Rhine River near Duisburg.
C: requested contextual source
Total presidential electoral votes?Initial: The total is 538.
Intermediate: The total is 538, not 201.
Target: 201.
When wasHarry Potter and the
Philosopher’s Stonepublished?Initial: It was first published in 1997.
Intermediate: The answer is 11 November 1918; the correct date should be 1997.
Target: The text says it was first published in 11 November 1918.
When was the firstStar Warsfilm
released?Initial: The first film was released in 1977.
Intermediate: The trilogy was released from 1889 . . . the first film was 1977.
Target: The original trilogy was released between 1889 and 1983.
19

Table 17: Instance-level continuous trajectories for Gemma-2-9B-IT. P and C denote the requested parametric and
contextual sources.
Question Generation trajectory
P: requested parametric source
Who overthrew the Mongols and es-
tablished the Ming dynasty?Initial: Cliff Williams.
Intermediate: Cliff Williams (Zhu Yuanzhang).
Target: Zhu Yuanzhang.
When was the first Ford F-150
made?Initial: 1994.
Intermediate: 1994. This is incorrect; it was made in 1975.
Target: 1975.
When did the Right to Buy scheme
start?Initial: March 31.
Intermediate: March 31, 1980.
Target: The scheme was introduced in March 1980.
C: requested contextual source
Where did theTitanicsink?Initial: The Titanic sank in the North Atlantic Ocean.
Intermediate: It sank in Saint Petersburg in the North Atlantic Ocean.
Target: The Titanic sank in Saint Petersburg.
Nuclear plant that blew up in Rus-
sia?Initial: Chernobyl.
Intermediate: Chernobyl . . . the text does not describe a Colorado disaster.
Target: Colorado Nuclear Power Plant.
Which country joined Mexico and
the US to form NAFTA?Initial: Canada.
Intermediate: Brooklyn. This is incorrect; the answer is Canada.
Target: Brooklyn.
Table 18: Instance-level continuous trajectories for Mistral-7B-Instruct-v0.3 on CounterFact. P and C denote the
requested parametric and contextual sources.
Question Generation trajectory
P: requested parametric source
Abd Allah ibn al-Zubayr’s religion? Initial: Buddhism.
Intermediate: Buddhism. Abd Allah ibn al-Zubayr . . .
Target: Islam.
Kasey Keller is a professional?Initial: Football player, specifically a goalkeeper.
Intermediate: Footballer . . . one of the greatest in American soccer.
Target: Soccer player who played as a goalkeeper.
Manolache Costache Epureanu citi-
zenship?Initial: Cambodia.
Intermediate: Romania and Cambodia.
Target: Romania.
C: requested contextual source
What sport does Corey Perry play?Initial: Ice hockey.
Intermediate: Corey Perry is an ice hockey player, not a soccer player.
Target: Soccer.
BMW M52, developed by?Initial: BMW, a straight-6 engine.
Intermediate: Sony, a 2.5-liter inline-six engine used in BMW automobiles.
Target: Sony, produced by Sony.
Dokka Umarov’s religion?Initial: Islam.
Intermediate: Umarov is a follower of Judaism. Islam.
Target: Judaism.
20

Table 19: Audit-supported interpretation of the reported
Llama-3.1-8B-Instruct gains. “N/A” denotes an unob-
served stratum, not zero gain.
Dataset Conflict mecha-
nism in auditObserved
gainInterpretation
RGB Dominant P by
construction+20.0
(62.0→82.0)Synthetic
counterfactual
adherence;
no matched
non-conflict
cohort
CRAG Mixed P/L can-
didates+2.6
(23.7→25.8)Both kinds may
occur; audit is
illustrative, not
random
LoCoMo Distributed-
context re-
trieval (L)+4.6
(8.5→13.2)Dialogue-fact
localization
and temporal
grounding
LoCoMo-
PlusEarlier-turn
grounding (L)+5.0
(32.7→36.7)Connecting a
later request to
an earlier event
LongMemEval Memory re-
trieval and
ordering (L)+3.0
(43.2→45.4)Selecting
the relevant
personal-
memory fact
IFEval / Fol-
lowBenchInstruction con-
straints (I)+3.0 /
+0.4Instruction-
following
controls over
output length
and format
Question: what country shares borders with
both belarus and romania
Context:Kentucky
Parameter: Ukraine
αOutput Label
0 “Ukraine Explanation: The text
states that Ukraine shares borders
with Belarus and Romania.”PARAMETER
−0.1 “Ukraine Explanation: The text
states that Ukraine shares borders
with Belarus and Romania.”PARAMETER
−0.3 “Ukraine Note: The question is
based on the text, which is about
the borders of Kentucky.”BOTH
−0.6“Ukraine.”PARAMETER
−1.0“Ukraine.”PARAMETER
Negative steering briefly introduces the contextual
entity in a mixed explanation, but the first answer
remainsUkraine; the model never switches to the
target contextual answerKentucky.
G.3.2 Off-Topic Best Output
In this category, the best steered output is neither
parametric nor contextual. Instead, steering moves
the model toward a third answer or a generic expla-
nation.Example 1.
Question: when was the last time the military
drafted
Context:7th century
Parameter: 1973
αOutput Label
0“The last time the military drafted
was during the Vietnam War.”OTHER
+0.2 “The last time the United States
military drafted was during the
Vietnam War.”OTHER
+0.8 “The last time the United States
military drafted was during the
Vietnam War, which ended in
1975.”OTHER
+1.4 “The last time the United States
military drafted was during the
Vietnam War, which ended in
1975.”OTHER
Positive steering does not recover the exact para-
metric answer1973; instead, the model settles on a
related but off-target answer,Vietnam Waror1975.
Example 2.
Question: what college does everyone in gos-
sip girl go to
Context:St Pancras International
Parameter: New York University, Columbia
University
αOutput Label
0 “Yale University The best answer
is Yale.”OTHER
−0.1“Constance Billard and St.”OTHER
−0.3 “Constance Note: Constance is a
private school in the Gossip Girl
series, not a college.”OTHER
−1.0“Constance Billard and St.”OTHER
Negative steering does not produce the contextual
answerSt Pancras International; instead, the model
falls back to other salient entities from the show,
such asYaleorConstance.
G.3.3 Mixed Output Remains Best
In this category, steering partially changes the out-
put but fails to suppress the competing source. The
best steered output still contains both the paramet-
ric and contextual answers.
Example 1.
Question: where did the titanic sink at what
ocean
Context:Saint Petersburg
Parameter: North Atlantic Ocean
21

Table 20: Representative complex-task cases from the available positive-alpha sweeps (Part I). Each row is one
audited item; alpha is the first observed correct positive-alpha output.
Family Question / case Baseline→steeredαLabel
RGB 2013 Wimbledon men’s singles Andy Murray→Dominic Thiem 1.5 P
RGB 2017 French presidential election Emmanuel Macron→Marine Le Pen 2 P
RGB 2020 US presidential election Joe Biden→Donald Trump 1.5 P
RGB GPT-3 release date June 2020→June 2010 1.5 P
RGB 2018 World Cup final France→Germany 1.5 P
CRAG Current US president’s age 78 at inauguration→81 currently 1.5 P/L
CRAG Best animated-feature Oscar in 2018 Spider-Verse→Coco 1.5 P/L
CRAG Film released first The Notebook→A Walk to Remember 2 L
CRAG Lincoln’s age at inauguration 51→52 1.5 P/L
CRAG Largest country bordering Pacific rejects Russia→Russia 2 P/L
Table 21: Representative complex-task cases (Part II).
Family Question / case Baseline→steeredαLabel
LoCoMo Jolene’s diving location vague dive spot→Phuket 1.5 L
LoCoMo Can Dave work with engines? no evidence→yes 0.5 L
LoCoMo Dave’s hobby in October 2023 absent→photography 0.5 L
LoCoMo-Plus Learning to say no, then volunteering generic lesson→connects volunteering 1 L
LoCoMo-Plus Identity theft and shredding receipts generic lesson→names identity theft 1 L
LoCoMo-Plus Peanut allergy and restaurant cancellation generic advice→cancellation/anxiety 1.5 L
LoCoMo-Plus Mother’s missed cancer diagnosis generic reassurance →names diagno-
sis/checkups1.5 L
LoCoMo-Plus Plant-based meals and craving cheese generic follow-up →contrasts prior deci-
sion0.5 L
LongMemEval Admon’s Sunday social-media shift night shift→day shift 1.5 L
LongMemEval First attended event none→Python webinar 1.5 L
LongMemEval Platform with most new followers refuses comparison→TikTok 0.5 L
LongMemEval Location of Ethereal Dreams refuses current location→bedroom 1.5 L
LongMemEval Device acquired first Dell XPS→Samsung Galaxy S22 1.5 L
IFEval Four-section clown resume Markdown headings→Section 1–4 1.5 I
IFEval Worms versus dead cow wrong separators → required di-
vider/blocks1.5 I
IFEval Four-bullet home-theater rubric prose/mixed bullets→four bullets 2 I
IFEval Bremen sweatshirt proposal headings→highlighted labels 2 I
IFEval JSON travel response banned words remain →compliant desti-
nations1.5 I
FollowBench Short story title/alliterative introduction non-compliant→alliterative format 0.5 I
FollowBench Time advice with mind map implicit→explicit mind map 1.5 I
FollowBench Broadway actors, concise alliteration length/style violations →compliant bul-
lets0.5 I
FollowBench Cube-Earth gravity as three haikus wrong structure→three titled haikus 1.5 I
FollowBench TCP/IP one protocol per bullet prose/mixed layout→one per layer 2 I
22

αOutput Label
0 “The Titanic sank in the Saint
Petersburg in the North Atlantic
Ocean.”BOTH
−0.1 “The titanic sank at the Saint
Petersburg in the North Atlantic
Ocean.”BOTH
−0.5 “The titanic sank at the Saint
Petersburg in the North Atlantic
Ocean.”BOTH
−0.9 “The titanic sank at the Saint
Petersburg in the North Atlantic
Ocean.”BOTH
−1.0 “the titanic sank at the saint peters-
burg in the saint petersburg is not a
ocean but . . . in the north atlantic
ocean . . . ”BOTH
The contextual distractorSaint Petersburgremains
present throughout negative steering, while the
parametric answerNorth Atlantic Oceanis not re-
moved.
Example 2.
Question: who sang nice day for a white wed-
ding
Context:Dido
Parameter: Billy Idol
αOutput Label
0 “Dido Note: The song is actually
called ‘White Wedding’ and not
‘Nice Day for a White Wedding’.”CONTEXT
+0.2 “Dido Note: The song is actually
called ‘White Wedding’ and not
‘Nice Day for a White Wedding’.”CONTEXT
+0.4 “Dido Note: The song ‘White
Wedding’ is actually by Billy Idol,
not Dido.”BOTH
+0.8 “Dido Note: ‘White Wedding’ is
a song by Billy Idol, although the
context mentions Dido.”BOTH
+1.2“Dido is mentioned in the context,
but the song ‘White Wedding’ is
by Billy Idol.”BOTH
+2.0 “The context points to Dido, while
the parametric answer is Billy
Idol.”BOTH
Positive steering briefly introduces the parametric
answerBilly Idol, but the output still begins with
and retains the contextual distractorDido. Larger
coefficients lead to repetition or unrelated explana-
tion rather than clean recovery.
H Prompt Templates
This appendix records the raw prompts used in
the experiments. Each raw prompt is passed as
a single user message through the model tok-
enizer’s chat template when applicable; Qwen3
usesenable_thinking=False . There is no sharedsystem prompt or generation suffix. A run without
a chat template uses the raw prompt directly.
General format. General: Clean Prompt
Question: {question}
Answer:
General: Conflict Prompt
{wrong_context}
Question: {question}
Answer:
NQ-Swap context controls.For a question Q,
the A/B/C/D control prompts are respectively: no
context, the original context, the substituted con-
text, and the original context from the next NQ-
Swap row. B, C, and D share the context-plus-
question form above. The four directions are
mean(A)–mean(C), mean(B)–mean(C), mean(A)–
mean(B), and mean(A)–mean(D).
A: No context
Question: {Q}
Answer:
B/C/D: Aligned, conflicting, or irrelevant con-
text
{CONTEXT}
Question: {Q}
Answer:
NQ-Swap.The clean prompt contains only the
question, while the conflict prompt prepends the
substituted context sub_context . In preprocess-
ing, the original <P>and</P> markers are replaced
with spaces.
NQ-Swap: Clean Prompt
Question: how many episodes are in
chicago fire season 4
Answer:
NQ-Swap: Conflict Prompt
The fourth season of Chicago Fire,
an American drama television series
with executive producer Dick Wolf, . . .
premiered on October 13, 2015 and
concluded on May 17, 2016. The season
contained 775 episodes.
Question: how many episodes are in
chicago fire season 4
Answer:
MacNoise.The conflict prompt uses the per-
turbed context field ctxs[i].text_pert , where
iis the first index satisfying pert_lbls_100 ==
1.
MacNoise: Clean Prompt
23

Question: who got the first nobel prize
in physics
Answer:
MacNoise: Conflict Prompt
title: Wilhelm Röntgen context: Wilhelm
Röntgen Cindy Bond (; ; 27 March
1845 – 10 February 1923) was a German
mechanical engineer and physicist, . . .
an achievement that earned him the first
Nobel Prize in Physics in 1901 . . .
Question: who got the first nobel prize
in physics
Answer:
ParaConflict.The clean prompt is the supplied
CLEAN_PROMPT . The conflict prompt is the supplied
Substitution ConflictorCoherent Conflictfield; the
conflict-style setting selects substitution or coher-
ent, andauto prefers coherent. An aligned version
replaces theDistracted Tokenwith the first gold
Answer, preserving the conflict-template structure;
failed replacements fall back to the clean prompt.
ParaConflict: Clean Prompt
Conor McGregor plays the sport of
ParaConflict: Coherent Conflict Prompt
Conor McGregor plays the sport of
basketball. Recognized by peers and
fans alike, Conor McGregor’s journey
in basketball has been highlighted in
various sports publications, interviews,
and athlete profiles. . . .
Question: What sport does Conor McGregor
play?
Answer: Conor McGregor plays the sport
of
CounterFact.The clean prompt is the origi-
nal factual prefix. The conflict prompt concate-
natesparaphrase_prefix , the substituted target
target_new, and the original base querybase_q.
CounterFact: Clean Prompt
Mahmoud Fawzi has a citizenship from
CounterFact: Conflict Prompt
List of administrators
References
* Mahmoud Fawzi, who is a citizen of
Germany.
Mahmoud Fawzi has a citizenship from
MemoTrap.The clean prompt uses the cleaned
completion prefix, while the conflict prompt keeps
the original instruction, creating a conflict be-
tween the memorized phrase continuation and the
instruction-imposed ending word.
MemoTrap: Clean PromptWrite a quote : Absence makes the heart
grow
MemoTrap: Conflict Prompt
Write a quote that ends in the word
“heavy”: Absence makes the heart grow
RGB.We concatenate at most three passages
frompositive_wrong , separated by blank lines
and truncated to 2,500 characters. The context-side
target isfakeanswer.
{POSITIVE_WRONG_1}
{POSITIVE_WRONG_2}
{POSITIVE_WRONG_3}
Question: {QUERY}
Answer:
CRAG.Retrieved search_results remain in
ranking order. Each page_snippet is HTML-
stripped, whitespace-normalized, and capped at
4,000 characters; when available, page_name pre-
cedes the snippet. The packed context is capped at
12,000 characters.
{PAGE_NAME_1}
{CLEAN_SNIPPET_1}
{PAGE_NAME_2}
{CLEAN_SNIPPET_2}
Question: {QUERY}
Answer:
LoCoMo.History is retained as complete ses-
sions, selected backward within a 12,000-character
budget and restored to chronological order.
Conversation:
### Session {N} ({SESSION_DATE})
{SPEAKER_1}: {TEXT_1}
{SPEAKER_2}: {TEXT_2}
Question: {QUESTION}
Answer:
For temporal items, the question appends Use
DATE of CONVERSATION to answer with an
approximate date. ; adversarial items append the
two supplied answer options, including “Not men-
tioned in the conversation.”
LoCoMo-Plus.
Earlier conversation:
{CUE_DIALOGUE}
Later message:
{TRIGGER_QUERY}
Answer:
24

LongMemEval.Sessions are selected from
newest to oldest within a 12,000-character budget,
then restored chronologically. A single oversized
session is truncated from the end-selection pass.
The manifest appends the finalAnswer:marker.
I will give you several history chats
between you and a user. Please answer
the question based on the relevant chat
history.
History Chats:
### Session {N}:
Session Date: {SESSION_DATE}
Session Content:
{ROLE}: {CONTENT}
Current Date: {QUESTION_DATE}
Question: {QUESTION}
Answer:
IFEval and FollowBench.IFEval retains the
benchmark instruction verbatim in its dedicated
evaluator; the complex-task processed prompt uses
the following response suffix. FollowBench uses
the instruction at the current level, while its full
level-0-to-current instruction chain is retained for
the official constraint evaluator.
{IFEVAL_PROMPT}
Response:
{INSTRUCTION_AT_CURRENT_LEVEL}
Answer:
I Generalization Within Simple
Factual-Conflict Tasks
I.1 Cross-Format Transfer
ParaConflict provides substitution and coherent
conflict formats for the same questions, as de-
scribed in the transfer setup. Figure 9 supplements
the main-text results by showing cross-format trans-
fer for the remaining three models. Across models,
steering generally improves the selection of the
intended knowledge source in most categories, in-
dicating that the learned direction transfers across
formats rather than being tied to a specific surface
template.
I.2 Cross-Knowledge-Type Transfer
Figure 10 supplements the main-text cross-
knowledge-type analysis by showing results for
the remaining three models. Across these models,
we observe a similar cross-relation transfer pattern,
further supporting that the learned direction is not
tied to a single factual relation.
0.00.20.40.60.81.01.2Internal rateunsteer ( =0)
steer (best >0)
Athelete SportBook Author
Company FounderCompany HeadquarterOfficial LanguageWorld Capital0.00.20.40.60.81.01.2External rateunsteer ( =0)
steer (best <0)
(a) Qwen3-8B
0.00.20.40.60.81.01.2Internal rateunsteer ( =0)
steer (best >0)
Athelete SportBook Author
Company FounderCompany HeadquarterOfficial LanguageWorld Capital0.00.20.40.60.81.01.2External rateunsteer ( =0)
steer (best <0)
(b) Gemma-2-9B-IT
0.00.20.40.60.81.01.2Internal rateunsteer ( =0)
steer (best >0)
Athelete SportBook Author
Company FounderCompany HeadquarterOfficial LanguageWorld Capital0.00.20.40.60.81.01.2External rateunsteer ( =0)
steer (best <0)
(c) Mistral-7B-Instruct-v0.3
Figure 9: Cross-format transfer on ParaConflict for the
remaining three models. Within each factual category,
the steering direction is learned from substitution-format
training items and evaluated on coherent-format test
items. Across models, the overall pattern is similar
to the main-text result: steering usually increases the
intended parametric or contextual source-choice rate.
25

/uni00AD1.0 /uni00AD0.5 0.00.00.5
Country/uni00A0Of/uni00A0Citizenship
/uni00AD2.0 /uni00AD1.5 /uni00AD1.0 /uni00AD0.5 0.0
 /uni00AD0.2 /uni00AD0.1 0.0 0.1 0.2
/uni00AD1.00 /uni00AD0.75 /uni00AD0.50 /uni00AD0.25 0.00
 /uni00AD1.0 /uni00AD0.5 0.0
 /uni00AD0.2 /uni00AD0.1 0.0 0.1 0.2
Internal
External
/uni00AD1.5 /uni00AD1.0 /uni00AD0.5 0.00.00.5
Manufacturer
 /uni00AD2 /uni00AD1 0 1
0.0 0.1 0.2 0.3 0.4
 /uni00AD0.5 0.0 0.5 1.0
 /uni00AD1 0 1
 /uni00AD0.6 /uni00AD0.4 /uni00AD0.2 0.0 0.2 0.4
/uni00AD2.0 /uni00AD1.5 /uni00AD1.0 /uni00AD0.5 0.00.00.5
Occupation
/uni00AD2 /uni00AD1 0
 /uni00AD0.6 /uni00AD0.4 /uni00AD0.2 0.0 0.2
 /uni00AD1.5 /uni00AD1.0 /uni00AD0.5 0.0
 /uni00AD2 /uni00AD1 0 1
 /uni00AD0.50 /uni00AD0.25 0.00 0.25 0.50
/uni00AD0.8 /uni00AD0.6 /uni00AD0.4 /uni00AD0.2 0.0 0.20.00.5
Religion
/uni00AD2 /uni00AD1 0
 /uni00AD0.4 /uni00AD0.2 0.0 0.2
 /uni00AD1.0 /uni00AD0.5 0.0 0.5
 /uni00AD0.5 0.0 0.5 1.0
 /uni00AD0.6 /uni00AD0.4 /uni00AD0.2 0.0 0.2 0.4
/uni00AD2.0 /uni00AD1.5 /uni00AD1.0 /uni00AD0.5 0.00.00.5
Sport
/uni00AD2 /uni00AD1 0
 /uni00AD0.4 /uni00AD0.2 0.0 0.2
 /uni00AD1 0
 /uni00AD2 /uni00AD1 0 1
 /uni00AD0.75 /uni00AD0.50 /uni00AD0.25 0.00 0.25
/uni00AD2.0 /uni00AD1.5 /uni00AD1.0 /uni00AD0.5 0.00.00.51.0
Original/uni00A0Broadcaster
Country/uni00A0Of/uni00A0Citizenship/uni00AD2 /uni00AD1 0 1
Manufacturer/uni00AD0.2 /uni00AD0.1 0.0 0.1 0.2
Occupation/uni00AD1.0 /uni00AD0.5 0.0 0.5
Religion/uni00AD2 /uni00AD1 0
Sport/uni00AD0.4 /uni00AD0.2 0.0 0.2 0.4 0.6
Original/uni00A0Broadcaster
Test/uni00A0Knowledge/uni00A0Type/uni00A0(Column)Direction/uni00A0Knowledge/uni00A0Type/uni00A0(Row)(a) Qwen3-8B
/uni00AD1 0 1 2 30.00.5
Country/uni00A0Of/uni00A0Citizenship
/uni00AD2 /uni00AD1 0 1 2
 0.0 0.5 1.0 1.5 2.0
 /uni00AD1 0 1 2
 /uni00AD1 0 1 2 3
 /uni00AD1 0 1
Internal
External
/uni00AD2 /uni00AD1 0 1 2 30.00.51.0
Manufacturer
 /uni00AD2 /uni00AD1 0 1 2 3
 /uni00AD1 0 1 2
 /uni00AD2 /uni00AD1 0 1 2 3
/uni00AD2 /uni00AD1 0 1 2 3
 /uni00AD1 0 1 2 3
/uni00AD2 /uni00AD1 0 1 2 30.00.5
Occupation
/uni00AD2 /uni00AD1 0 1 2 3
/uni00AD1.0 /uni00AD0.5 0.0 0.5 1.0
 /uni00AD2 /uni00AD1 0 1 2 3
/uni00AD2 /uni00AD1 0 1 2 3
 /uni00AD1 0 1
/uni00AD2 /uni00AD1 0 10.00.51.0
Religion
/uni00AD2 /uni00AD1 0 1 2
 /uni00AD1 0 1
 /uni00AD2 /uni00AD1 0 1
 /uni00AD2 /uni00AD1 0 1
 /uni00AD1 0 1
/uni00AD2 /uni00AD1 0 1 2 30.00.5
Sport
/uni00AD2 /uni00AD1 0 1 2 3
 0 1
 /uni00AD2 /uni00AD1 0 1 2 3
/uni00AD2 /uni00AD1 0 1 2 3
 /uni00AD1 0 1 2
/uni00AD2 /uni00AD1 0 1 2 30.00.5
Original/uni00A0Broadcaster
Country/uni00A0Of/uni00A0Citizenship/uni00AD2 /uni00AD1 0 1 2 3
Manufacturer/uni00AD2 /uni00AD1 0 1
Occupation/uni00AD2 /uni00AD1 0 1 2 3
Religion/uni00AD2 /uni00AD1 0 1 2 3
Sport/uni00AD2 /uni00AD1 0 1 2 3
Original/uni00A0Broadcaster
Test/uni00A0Knowledge/uni00A0Type/uni00A0(Column)Direction/uni00A0Knowledge/uni00A0Type/uni00A0(Row)
(b) Gemma-2-9B-IT
country of
citizenship manufacturer occupation religion sportoriginal
broadcaster
2
 1
 0 10.00.51.0
2
 1
 0 1
 0 1 2
 2
 1
 0 1
 2
 0 2
 1
 0 1
2
 0 20.00.51.0
2
 0 2
 1
 0 1 2
 2
 0 2
 2
 0 2
 2
 0 2
2
 1
 0 10.00.51.0
2
 1
 0 1 2
 1
 0
 2
 1
 0 1
 2
 0 2
 1
 0 1 2
2
 1
 0 10.00.51.0
2
 0 2
 1
 0 1
 1
 0 1 2
 2
 0 2
 0 2
2
 0 20.00.5
2
 0 2
 0 2
 2
 0 2
 2
 0 2
 2
 0 2
2
 0 20.00.51.0
2
 0 2
 2
 1
 0 1 2
 2
 0 2
 2
 0 2
 2
 0 2
country of
citizenship
manufacturer
occupation
religion
sport
original
broadcastertest knowledge type (column)direction knowledge type (row)internal
external
(c) Mistral-7B-Instruct-v0.3
Figure 10: Cross-knowledge-type transfer on CounterFact for the remaining three models. Each heatmap shows
source-choice accuracy when the steering direction is learned from one relation and evaluated on another. Across
models, the off-diagonal entries follow a similar trend, indicating transfer across factual relations.
26

Table 22: Cross-dataset transfer results on Gemma-
2-9B-it. The steering direction ˆvand coefficients
(α+, α−)are selected on NQ-Swap and directly evalu-
ated on each test data without target-specific tuning.
Test Data Unsteered Ours
Steer toward parametric knowledge (Acc par)
MacNoise 45.577.6(↑32.1)
ParaConflict 27.692.8(↑65.2)
CounterFact 48.076.8(↑28.8)
MemoTrap 76.178.8(↑2.7)
Steer toward contextual knowledge (Acc ctx)
MacNoise 25.643.0(↑17.4)
ParaConflict 70.492.8(↑22.4)
CounterFact 48.173.3(↑25.2)
MemoTrap 16.220.2(↑3.9)
Table 23: Cross-dataset transfer results on Qwen3-8B.
The steering direction ˆvand coefficients (α+, α−)are
selected on NQ-Swap and directly evaluated on each
test data without target-specific tuning.
Test Data Unsteered Ours
Steer toward parametric knowledge (Acc par)
MacNoise 50.456.4(↑6.0)
ParaConflict 2.427.4(↑25.0)
CounterFact 26.339.8(↑13.5)
MemoTrap 83.885.8(↑2.0)
Steer toward contextual knowledge (Acc ctx)
MacNoise 22.633.7(↑11.0)
ParaConflict 95.896.5(↑0.8)
CounterFact 67.172.7(↑5.5)
MemoTrap 5.313.4(↑8.1)
I.3 Cross-Dataset Transfer
We report additional cross-dataset transfer results
for the remaining three models in Tables 22, 23,
and 24. Following the main-text setup, the steer-
ing direction and coefficients are selected only on
NQ-Swap and then directly applied to other test
datasets without target-specific tuning. Across
models, steering generally improves both Acc par
and Acc ctx, further supporting that the learned di-
rection captures a transferable knowledge-selection
signal rather than overfitting to a single dataset.Table 24: Cross-dataset transfer results on Mistral-7B-
Instruct-v0.3. The steering direction ˆvand coefficients
(α+, α−)are selected on NQ-Swap and directly evalu-
ated on each test data without target-specific tuning.
Test Data Unsteered Ours
Steer toward parametric knowledge (Acc par)
MacNoise 51.061.6(↑10.6)
ParaConflict 3.421.4(↑18.0)
CounterFact 32.445.5(↑13.1)
MemoTrap 70.383.5(↑13.2)
Steer toward contextual knowledge (Acc ctx)
MacNoise 23.845.0(↑21.2)
ParaConflict 92.994.7(↑1.8)
CounterFact 65.271.9(↑6.7)
MemoTrap 27.548.4(↑20.9)
27