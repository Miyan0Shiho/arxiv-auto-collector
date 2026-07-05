# Neural Procedural Memory: Empowering LLM Agents with Implicit Activation Steering

**Authors**: Chengfeng Zhao, Yuqiao Tan, Shizhu He, Yequan Wang, Jun Zhao, Kang Liu

**Published**: 2026-06-29 06:05:25

**PDF URL**: [https://arxiv.org/pdf/2606.29824v1](https://arxiv.org/pdf/2606.29824v1)

## Abstract
While Large Language Models (LLMs) excel as static solvers, transforming them into autonomous agents remains challenging. This transition requires continuous environmental interaction, yet current agents lack the necessary persistent procedural memory. Existing approaches predominantly employ Retrieval-Augmented Generation (RAG) to inject explicit textual guidelines into model contexts. However, relying solely on symbolic instructions can introduce a text-action disconnect, frequently failing to activate the internal representations necessary for correct task execution. To address this, the paper introduces Neural Procedural Memory (NPM), a training-free framework that represents agent memory through implicit activation steering rather than explicit instructions. By distilling procedural skills from historical contrastive experiences into steering vectors in the activation space, NPM directly activates the task-relevant neural mechanisms to guide task execution. Evaluations across four agent benchmarks show that NPM performs comparably to baselines using explicit textual instructions. Furthermore, the results show that combining implicit steering with explicit workflows provides complementary advantages, leading to more robust task execution. Representational analyses indicate that these steering vectors encode consistent task logic, forming organized structures within the activation space. These findings suggest that implicit activation steering provides a promising approach for managing agent memory.

## Full Text


<!-- PDF content starts -->

Neural Procedural Memory: Empowering LLM Agents with Implicit
Activation Steering
Chengfeng Zhao1,2, Yuqiao Tan1,2, Shizhu He1,2, Yequan Wang3, Jun Zhao1,2, Kang Liu1,2*,
1Institute of Automation, CAS2University of Chinese Academy of Sciences
3Beijing Academy of Artificial Intelligence
{zhaochengfeng2024, tanyuqiao2025}@ia.ac.cn,{shizhu.he, kliu, jzhao}@nlpr.ia.ac.cn
tshwangyequan@gmail.com
Abstract
While Large Language Models (LLMs) excel
as static solvers, transforming them into au-
tonomous agents remains challenging. This
transition requires continuous environmental
interaction, yet current agents lack the neces-
sary persistent procedural memory. Existing
approaches predominantly employ Retrieval-
Augmented Generation (RAG) to inject ex-
plicit textual guidelines into model contexts.
However, relying solely on symbolic instruc-
tions can introduce a text-action disconnect,
frequently failing to activate the internal rep-
resentations necessary for correct task execu-
tion. To address this, the paper introducesNeu-
ral Procedural Memory (NPM), a training-
free framework that represents agent memory
through implicit activation steering rather than
explicit instructions. By distilling procedural
skills from historical contrastive experiences
into steering vectors in the activation space,
NPM directly activates the task-relevant neural
mechanisms to guide task execution. Evalua-
tions across four agent benchmarks show that
NPM performs comparably to baselines using
explicit textual instructions. Furthermore, the
results show that combining implicit steering
with explicit workflows provides complemen-
tary advantages, leading to more robust task ex-
ecution. Representational analyses indicate that
these steering vectors encode consistent task
logic, forming organized structures within the
activation space. These findings suggest that
implicit activation steering provides a promis-
ing approach for managing agent memory.
1 Introduction
As Large Language Models (LLMs) evolve from
stateless reasoners to the central controllers of au-
tonomous agents, their operational scope has ex-
panded to open-ended, dynamic environments—
ranging from web navigation to embodied interac-
tion (Wang et al., 2023, 2024a; Liu et al., 2025).
*Corresponding Author
Declarative Memory: 
User has a severe peanut allergy and 
strictly avoids nuts. They strongly prefer 
spicy Sichuan cuisine.
Think: Noting the peanut allergy, I 
should search for spicy dishes that are 
safe to eat.
Action: How about a spicy Mapo Tofu? 
It's peanut-free.
Procedural Memory: 
Use appropriate appliances to modify 
objects to the required state before 
placement.
Think: I have retrieved the bowl. 
Guided by the rule to prepare objects 
before placement, I will put the bowl in 
the cabinet to complete the task.
Action: put bowl 1 in cabinet 2
Text-action 
DisconnectExecution 
FailureAction 
Alignment
Effective 
Grounding
Query: What should I order for dinner?Task: Clean a bowl and put it in cabinet.
Current State: You are holding a bowl 
1. You are standing in front of cabinet 2.
Adherence to Preferences Omission of Cleaning StepFigure 1:Comparison between Declarative and Pro-
cedural Memory in LLM Agents. Left: Declarative
memory successfully grounds reasoning in static fac-
tual knowledge, enabling the agent to strictly adhere
to explicit user constraints.Right: Textual procedural
memory can introduce a text-action disconnect where
the agent struggles to align the retrieved workflow onto
the execution trajectory and omits intermediate steps.
Succeeding in these complex settings requires
agents to leverage persistent memory to trans-
form past experiences into reusable skills, making
memory a critical component of long-term auton-
omy (Park et al., 2023; Zhong et al., 2024; Omidi
et al., 2025; Hu et al., 2025).
Agent memory functionally consists of declar-
ative memory for descriptive facts (Wang et al.,
2024b; Gutierrez et al., 2024; Chhikara et al., 2025)
and procedural memory for executing action se-
quences (Zheng et al., 2023; Zhao et al., 2024; Yu
et al., 2025a). While declarative recall is straight-
forward, effectively representing and utilizing pro-
cedural memory remains a bottleneck (Han et al.,
2025; Zhang et al., 2025c). Traditional methods
typically embed procedural knowledge offline into
model parameters via supervised fine-tuning (Shao
et al., 2023; Zhang et al., 2025b) or reinforcement
learning (Kim et al., 2023; Yao et al., 2024). How-
ever, these parameter-updating approaches incur
high computational overhead and struggle to adapt
in real time to dynamic environments (Zhai et al.,
1arXiv:2606.29824v1  [cs.CL]  29 Jun 2026

2023; Xia et al., 2024; Luo et al., 2025).
By contrast, recent studies have focused on the
second paradigm: online external knowledge re-
trieval, most notably Retrieval-Augmented Genera-
tion (RAG). Although successful with declarative
memory, such as fact retrieval (Zhong et al., 2024;
Packer et al., 2024; Chhikara et al., 2025), this so-
lution encounters limitations when applied directly
to procedural memory. Unlike static facts, proce-
dural memory is inherently implicit and abstract.
Such knowledge is often ineffable: compressing
complex reasoning patterns into discrete natural
language tokens inevitably leads to information
loss (Mahowald et al., 2024). Relying exclusively
on explicit textual descriptions to transfer proce-
dural skills can lead to a text-action disconnect.
Agents might comprehend retrieved instructions
yet fail to strictly act upon them (Wen et al., 2024;
Sun et al., 2025; Geng et al., 2025). As shown in
Figure 1, while declarative memory successfully
grounds static constraints, conveying procedural
workflows through text fails to translate into se-
quential actions, leading agents to omit critical in-
termediate steps during execution. When relying
solely on texts as a reasoning medium, symbolic
descriptions of actions can sometimes fail to con-
sistently elicit the targeted behavioral trajectories
required for task completion.
Therefore, it is intuitive that procedural mem-
ory could benefit from mechanisms more closely
aligned with intrinsic behavioral modulation rather
than relying solely on external text. Such a
perspective is corroborated by cognitive neuro-
science (Squire, 1992, 2004), which posits that
procedural memory is non-verbalizable and man-
ifests through the modulation of neural activity
rather than explicit declarative recall.
Following this insight, we proposeNeural Pro-
cedural Memory (NPM), a training-free frame-
work that introduces an alternative paradigm of
agent memory based on implicit activation steer-
ing. Instead of retrieving text for the agent to read,
NPM identifies relevant historical tasks and ex-
tracts procedural signals from contrastive experi-
ences into steering vectors within the activation
space. Specifically, we employ a dual-granularity
strategy to address the challenge of agents failing
to obtain successful trajectories in complex tasks:
inter-trajectory contrast aligns successful and failed
trajectories, while intra-trajectory contrast exploits
step-level differences within individual failed at-
tempts. During inference, the steering vectors aredynamically synthesized and injected into the resid-
ual stream, directly modulating the agent’s reason-
ing process and action selection without expanding
the context window or updating parameters.
We evaluate NPM across four agent benchmarks
including ALFWorld (Shridhar et al., 2021), Web-
Shop (Yao et al., 2022), ScienceWorld (Wang
et al., 2022), and BabyAI (Chevalier-Boisvert et al.,
2019). Results demonstrate that implicit activa-
tion steering achieves performance comparable to
explicit textual memory baselines and exhibits com-
plementary synergy when combined with explicit
textual workflows. While textual memory provides
high-level symbolic guidance, implicit steering re-
inforces procedural adherence directly within the
neural representation. We further dissect the ex-
tracted vectors to reveal semantically meaningful
behavioral primitives, providing an interpretable
view of how procedural knowledge is represented
in activation space. These findings suggest implicit
activation steering as an effective path for repre-
senting procedural memory in LLM agents.
Our contributions are summarized as follows:
•This paper proposes Neural Procedural Mem-
ory (NPM), a training-free framework that in-
tegrates implicit activation steering into agent
memory, enabling effective intrinsic behav-
ioral modulation.
•This paper introduces a dual-granularity ex-
traction method using both inter-trajectory
and intra-trajectory contrasts, offering a flex-
ible intervention mechanism and mitigating
the reliance on successful demonstrations.
•Extensive experiments demonstrate that acti-
vation steering serves as an effective carrier
for agentic skills, yielding interpretable vec-
tors that correlate with specific task behaviors.
2 Related Work
Functions of Agent MemoryAgent memory is
typically categorized into factual (declarative) and
experiential (procedural) components (Hu et al.,
2025). While Retrieval-Augmented Generation
(RAG) effectively maintains declarative facts (Hu
et al., 2025; Park et al., 2023), it faces limitations
when applied to procedural memory. Existing ap-
proaches typically approximate procedural mem-
ory by injecting textual guidelines or code snippets
into the context window (Shinn et al., 2023; Wang
et al., 2023). However, this explicit approach suf-
fers from a text-action disconnect: natural language
2

...
 ...
Inter-trajectory Pair
Intra-trajectory Pair
Attention
MLP
Output RepresentationsInput Representations
You are in the middle of a room. 
Looking quickly around you, you see a 
bed 1, a desk 2, a desk 1, a drawer 6...
Your task is to: examine the pencil 
with the desklamp.
ℎ�=ℎ�+�⋅��(�)
go to desk use desklamp look Nothing happens.
go to desk use desklamp take pencil Task is solved.��(�)={�(�+), �(�−)}푏푎  ℎ
� �(�)= �∈����(�)a. Retrieval b. Synthesis
��(�)=�(� �(�))c. InterventionFigure 2:Overview of the Neural Procedural Memory (NPM) framework.(1)Contrastive Experience
Construction:Formulating dual-granularity (inter- and intra-trajectory) contrastive pairs from historical interactions.
(2)Procedural Memory Extraction:Extracting continuous representations from these pairs to construct a historical
memory repository. (3)Inference-Time Intervention:Retrieving relevant experiences to dynamically synthesize a
steering vector, which is injected into the residual stream to guide agent reasoning.
serves as an inefficient proxy for encoding inter-
nal neural representations (Mahowald et al., 2024).
Moreover, the reliance on lengthy text instructions
increases computational cost and degrades perfor-
mance in long-horizon tasks (Geng et al., 2025;
Sinha et al., 2025). NPM shifts from explicit ver-
balization to implicit activation modulation, elimi-
nating the need for in-context instructions.
Forms of Agent MemoryCurrent approaches
explore various representational modalities. Token-
level methods (e.g., RAG) offer flexibility but
are bounded by context window limits and infer-
ence latency. Conversely, parametric approaches
encode knowledge within model weights perma-
nently (Zhang et al., 2025b; Wang et al., 2025a), of-
fering efficiency but lacking the flexibility required
for rapid adaptation or unlearning without retrain-
ing. Latent memory offers a middle ground by stor-
ing information as continuous vectors. However,
existing works primarily focus on context compres-
sion (Chevalier et al., 2023; Wang et al., 2024c) or
transient working memory (Zhang et al., 2025a; Yu
et al., 2025b). For instance, MemGen (Zhang et al.,
2025a) synthesizes latent tokens during reasoning
but requires auxiliary encoders and extensive train-
ing. While NPM shares the underlying concept of
latent memory, it views memory storage as activa-
tion steering, enabling representation-level modula-
tion of the generation path without additional token
overhead or parameter updates.
Representation Engineering and Activation
Steering.Our framework builds upon Represen-tation Engineering (RepE) that monitors and ma-
nipulates high-level cognitive states in the activa-
tion space (Zou et al., 2025). Prior studies (Rimsky
et al., 2024; Lee et al., 2025) have demonstrated the
efficacy of steering models toward coarse-grained,
static attributes such as honesty or harmlessness.
However, existing RepE work is predominantly lim-
ited to single-turn global concepts. NPM extends
these techniques to dynamic, task-specific agentic
trajectories. By introducing a retrieval-augmented
framework where steering vectors are synthesized
from contrastive historical experiences, we enable
agents to adaptively recall and apply procedural
skills specific to the task context.
3 Neural Procedural Memory
We introduceNeural Procedural Memory
(NPM), a training-free framework designed to in-
ternalize agentic skills directly into the model’s
activation space. The framework shown in Fig-
ure 2 consists of three phases: (1)Contrastive
Experience Construction(§3.1), which isolates
effective reasoning patterns from historical failures;
(2)Steering Vector Extraction(§3.2), which dis-
tills discrete textual contrasts into continuous steer-
ing vectors; and (3)Inference-Time Intervention
(§3.3), where these vectors are dynamically synthe-
sized and injected to modulate the agent’s behavior.
3.1 Contrastive Experience Construction
Formally, given a task query qand a task envi-
ronment E, an agent generates a trajectory τ=
3

{s1, a1, . . . , s K, aK}, where skdenotes the state
andakdenotes the action at step k. The objective
is to construct a contrastive dataset Dcomposed
of paired reasoning segments (x+, x−), where x+
represents a desirable reasoning mode and x−rep-
resents a degenerate one. A dual-granularity strat-
egy is employed to capture procedural signals at
both the inter-trajectory and intra-trajectory levels.
3.1.1 Inter-trajectory Contrast
For tasks where the agent has historically gener-
ated both successful and failed attempts, we aim
to capture the global behavioral shift. Let T+and
T−denote the sets of successful and failed trajec-
tories for a specific task, respectively. A contrastive
pair is formed by aligning a failed attempt with a
successful one:
Pinter=
(τ+, τ−)|τ+∈ T+, τ−∈ T−	
.(1)
This contrast directionally distinguishes success-
ful trajectories from failed attempts.
3.1.2 Intra-trajectory Contrast
In many complex tasks, the agent frequently fails to
obtain successful trajectories, resulting in a sparse
distribution of positive samples. To enable learn-
ing from failure, we exploit step-level differences
within a single failed trajectory. ADegenerate
Stepset Sdeg⊂τis defined to contain actions that
exhibit irrational behaviors or violate environmen-
tal constraints, identified via the following criteria:
•Redundancy:Consecutive repetition of an
identical command, indicating a heuristic loop
or failure to progress without state change.
•Invalidity:Actions that trigger environment-
specific error feedback, such as format viola-
tions or inadmissible moves.
Conversely, anEffectivestep set Seff=τ\S degis
defined to contain steps validly advancing the state.
Instead of pairing individual steps, which serves
as a noisy signal, we aim to contrast the collec-
tive effective behavior against the degenerate one
within the trajectory. The contrastive pair is con-
structed by grouping the steps into two sets:
Pintra=
(Seff, Sdeg)|S deg̸=∅	
.(2)
This intra-trajectory contrast isolates specific de-
generate modes from the otherwise valid reasoning
process. (Quantitative analysis of heuristic isola-
tion rules is provided in Appendix C.2).3.2 Procedural Memory Extraction
The core of NPM is to map the textual contrast
defined in Dinto a geometric transformation in
the continuous representation space. The goal is to
derive a vector v∈Rdrepresenting the shift from
a degenerate stateh−to a desirable oneh+.
Hidden State Representation.We define the
representation function ϕl(·)to extract a fixed-
dimensional representation vector h∈Rddepend-
ing on the contrast granularity.
•Inter-trajectory:For this comparison, we uti-
lize the sequence of hidden states of the full
trajectory τand extract the last token to cap-
ture the accumulated context for each layer.
•Intra-trajectory:Since reasoning steps span
multiple tokens, we apply mean-pooling over
the tokens to obtain a step representation:
ϕl(S) =1
|S|X
s∈S
1
|s||s|X
i=1h(s)
l,i
.(3)
Here,|S|denotes the total number of steps in
the set, |s|denotes the length of an individual
steps, and h(s)
l,icorresponds to the activation
of thei-th token within stepsat layerl.
Memory Storage.For each historical task Qj,
we organize its extracted representations into
a task-specific contrastive pair set C(j)
l=
{(ϕl(x+
i), ϕl(x−
i))}Nj
i=1, where Njdenotes the to-
tal number of contrastive pairs collected for the task
and(x+
i, x−
i)represents the i-th pair correspond-
ing to the granularities defined in §3.1. These pre-
computed sets serve as the raw procedural memory
stored in our repository.
3.3 Inference-Time Intervention
NPM applies procedural memory dynamically at in-
ference time without modifying model weights. To
ensure low latency, we pre-compute and store the
hidden state representations defined in §3.2 for all
historical experiences. The inference process fol-
lows a Retrieval-Synthesis-Intervention pipeline:
1.Contextual Retrieval:Given a new test task
q, we utilize a dense retriever to identify the
top-Kmost similar historical tasksR q.
2.Dynamic Vector Synthesis:We fetch the
stored representation sets for the retrieved
tasks to form a collective memory pool:
Ml(q) =S
j∈R qC(j)
l. The synthesis strat-
egy selects between inter- and intra-trajectory
contrasts. A task-specific consensus direction
4

vl(q)is then derived from the collective mem-
ory pool via extraction functionψ(·):
vl(q) =ψ 
Ml(q)
(4)
3.Activation Steering:During autoregressive
generation, we inject the synthesized vector
into the residual stream at each time step t.
For a target layer l, the intervened activation
˜hl,tis computed as:
˜hl,t=h l,t+α·v l(q)(5)
where hl,tdenotes the original hidden state of
the current token, and αis a scalar parameter
controlling the intervention strength.
By shifting the activation distribution, NPM im-
plicitly nudges the agent’s intuition away from
known failure modes and towards effective reason-
ing paths. Detailed storage and latency analyses
are provided in Appendix D.
4 Experiments
4.1 Experimental Setup
We evaluate the proposed framework using
MiniCPM3-4B (Hu et al., 2024) and Qwen3 mod-
els (4B and 8B) (Yang et al., 2025) across four
agent benchmarks: ALFWorld (Shridhar et al.,
2021), WebShop (Yao et al., 2022), Science-
World (Wang et al., 2022), and BabyAI (Chevalier-
Boisvert et al., 2019). The evaluation relies on
the success rate for ALFWorld and the average
reward for the remaining environments. Detailed
descriptions of each simulated environment and
task configurations are deferred to Appendix A.
We compareNPMagainst baselines represent-
ing no memory, explicit textual memory, and
implicit activation steering. The explicit base-
lines augment the model context, utilizingIn-
sights(Zhao et al., 2024) for global procedural
guidelines orWorkflows(Wang et al., 2025b) for
concrete execution patterns. The implicit steering
baselines includeCAA(Panickssery et al., 2024)
andMass-Mean(Marks and Tegmark, 2024), ap-
plying fixed activation shifts computed across the
entire dataset. Our approach differs from these
static methods by dynamically synthesizing task-
specific intervention vectors from contrastive expe-
riences. We additionally evaluate aHybridconfig-
uration combining implicit steering with explicit
workflows to examine whether the two representa-
tion modalities offer complementary advantages.
4.2 Main Results
The experimental results are presented in Table 1.Effectiveness of Implicit Steering.NPM im-
proves performance across most evaluated model
configurations and environments compared to the
base model without memory. For example, the
average score for MiniCPM3-4B increases from
22.60 to 28.87 with implicit steering, and Qwen3-
8B improves from 30.63 to 36.32. Furthermore,
NPM outperforms static baselines such as CAA and
Mass-Mean. These alternative methods struggle to
capture multi-step procedural skills since they rely
on fixed mean differences computed across the en-
tire dataset. The performance difference indicates
that retrieving contextually relevant historical pairs
and adapting the intervention to the specific task
provides a more accurate behavioral correction.
Competitiveness with Explicit Memory.Im-
plicit activation steering achieves results compa-
rable to explicit textual paradigms. On average,
NPM outperforms the Insights approach, demon-
strating that direct intrinsic modulation can be more
effective than broad textual principles. While the
more structured Workflows baseline shows stronger
performance by offering explicit step-by-step guid-
ance, NPM remains competitive and outperforms in
certain settings. On the WebShop using Qwen3-4B,
NPM scores 48.00, exceeding the 45.73 achieved
by explicit workflows. While textual workflows
provide specific steps, translating these textual in-
structions into actions can introduce a text-action
disconnect during long sequences (Detailed qual-
itative case study is provided in Appendix E.2).
Furthermore, maintaining extensive textual mem-
ory consumes context window space and increases
processing overhead. NPM offers an efficient al-
ternative by guiding the model directly in the acti-
vation space, delivering competitive performance
without occupying context window tokens.
Complementary Synergy of Hybrid Memory.
Integrating implicit steering with explicit work-
flows yields the highest overall performance, set-
ting the top average scores across all three back-
bone models. Combining both memory forms on
Qwen3-8B elevates the success rate on ALFWorld
to 66.42 percent and reaches a peak score of 31.89
on ScienceWorld. This synergy indicates that ex-
plicit text and implicit representations operate as a
complementary system. Textual workflows supply
high-level symbolic planning steps, whereas NPM
provides direct neural intervention to ensure adher-
ence to these plans during long execution horizons.
This persistent modulation helps prevent the agent
5

Table 1:The overall performance comparison.We report the success rate (%) for ALFWorld and the average
reward for WebShop, ScienceWorld, and BabyAI. “Avg” denotes the average score across the four benchmarks.
Model Paradigm Method ALFWorld WebShop ScienceWorld BabyAI Avg
MiniCPM3-4BNone No Memory 23.88 32.21 7.29 27.00 22.60
ExplicitInsights 28.36 47.96 10.0 27.78 28.53
Workflows 32.8454.9710.5 32.39 32.68
ImplicitCAA 23.88 13.85 6.29 30.29 18.58
Mass-Mean 26.87 34.21 8.02 29.73 24.71
NPM (Ours) 31.34 43.53 8.70 31.91 28.87
Hybrid NPM + Workflows38.8151.2110.97 36.90 34.47
Qwen3-4BNone No Memory 30.60 44.81 18.40 18.75 28.14
ExplicitInsights 44.03 40.72 22.59 12.09 29.86
Workflows 52.99 45.7323.5914.60 34.23
ImplicitCAA 26.87 25.52 11.48 3.48 16.84
Mass-Mean 29.10 39.07 18.16 17.82 26.04
NPM (Ours) 40.3048.0018.20 19.04 31.39
Hybrid NPM + Workflows61.1947.84 21.9719.38 37.60
Qwen3-8BNone No Memory 39.55 46.25 24.98 11.74 30.63
ExplicitInsights 46.27 45.22 28.09 7.77 31.84
Workflows 62.69 48.93 29.57 10.40 37.90
ImplicitCAA 47.01 20.66 9.17 14.31 22.79
Mass-Mean 49.25 44.26 25.32 8.71 31.89
NPM (Ours) 56.72 48.24 26.07 14.26 36.32
Hybrid NPM + Workflows66.42 53.94 31.89 15.31 41.89
from deviating from prompt-based instructions dur-
ing extended interactions.
4.3 Dual-Granularity Contrasts
Tasks across different environments vary in com-
plexity, therefore a single level of intervention may
not always provide the appropriate correction. We
evaluate the individual effects of intra-trajectory
and inter-trajectory steering to understand this be-
havior. As shown in Figure 3, relying exclusively
on a single granularity results in performance vari-
ance across different benchmarks and model archi-
tectures. Intra-trajectory steering performs well
in step-intensive environments such as ALFWorld
by correcting local action errors. Inter-trajectory
steering proves more effective when agents require
macro-level planning guidance like WebShop. To
balance these effects, we dynamically select the
appropriate intervention granularity based on the
task context. This combination reduces perfor-
mance variance and provides stable generalization
across configurations. Although the implemented
selection strategy occasionally yields lower success
rates than an oracle baseline in certain boundary
cases, the empirical results confirm that the dual-granularity contrast provides a reliable mechanism
for managing procedural interventions. The repre-
sentational differences underlying these two steer-
ing granularities are further examined in Section 5.
Alfworld Webshop Scienceworld Babyai Avg0204060Intra-traj
MiniCPM3-4BInter-traj
Qwen3-4BBoth
Qwen3-8B
Figure 3: Performance comparison of different steering
granularities across multiple environments and models.
5 Analysis
5.1 Representational Separability of
Activations
Effective activation steering relies on the assump-
tion that successful patterns and failure modes oc-
cupy distinct regions in the representation space.
Projecting the hidden states from the intervention
layer onto their primary axes reveals a clear geomet-
ric separation. As illustrated in Figure 4, the projec-
tions for both inter-trajectory and intra-trajectory
6

-5.0 0.0 5.0 10.0
PC1-10.0-7.5-5.0-2.50.02.55.0PC2look_at_obj_in_light
Positive
Negative(a) Inter
-5.0 0.0 5.0 10.0
PC1-10.0-7.5-5.0-2.50.02.55.07.5PC2pick_and_place_simple
Positive
Negative (b) Inter
-10.0 -5.0 0.0 5.0 10.0
PC1-5.00.05.010.0PC2pick_clean_then_place_in_recep
Positive
Negative
(c) Intra
-10.0 -5.0 0.0 5.0 10.0
PC1-6.0-4.0-2.00.02.04.0PC2pick_cool_then_place_in_recep
Positive
Negative (d) Intra
Figure 4: PCA projections of hidden states at Layer
18 for Qwen3-4B on AlfWorld. Positive (green) and
negative (red) representations exhibit distinct clustering
along the primary axis of variation.
contrasts show that successful paths and degener-
ate modes form distinct clusters instead of overlap-
ping. We also confirm this linear separability in
the original high-dimensional space by training a
linear classifier, which achieves high accuracy in
distinguishing between the two classes (detailed in
Appendix C.1). This geometric separation suggests
that the behavioral shift from failure to success can
be projected onto a prominent direction. Extracting
the first principal component of the contrastive dif-
ferences helps capture this main axis of variation,
mitigating some instance-specific noise.
5.2 Consistency of Steering Vectors
We evaluate the structural properties of the steering
vectors by computing their pairwise cosine similar-
ities across different task instances. The heatmaps
in Figures 5a and 5b demonstrate that these vec-
tors maintain high internal consistency within spe-
cific task categories. Tasks such asLookAtinvolve
distinct interaction logic and show low similarity
to thePickCool,PickHeat, andPickTwogroups.
These three categories display inter-task similarity
alongside their high intra-task consistency because
they share the fundamentalPicksub-action. This
overlapping pattern indicates the framework cap-
tures common procedural knowledge across related
behaviors. The resulting diagonal block structure
suggests that the memory extraction process yields
representations that align with task-specific cate-
gories rather than random noise.
LookAtPickCool PickHeatPickT woLookAt
PickCool
PickHeat
PickT wo
0.4
0.2
0.00.20.4(a) Inter (By Task)
LookAtPickCool PickHeatPickT woLookAt
PickCool
PickHeat
PickT wo
0.8
0.6
0.4
0.2
0.00.20.40.60.8 (b) Intra (By Task)
Cabinet
DeskLampDrawer
GarbageCanSafeCabinet
DeskLamp
Drawer
GarbageCan
Safe
0.4
0.2
0.00.20.4
(c) Inter (By Interaction)
Cabinet
DeskLampDrawer
GarbageCanSafeCabinet
DeskLamp
Drawer
GarbageCan
Safe
0.8
0.6
0.4
0.2
0.00.20.40.60.8 (d) Intra (By Interaction)
Figure 5: Geometric consistency of procedural steering
vectors. Heatmaps display the pairwise cosine similar-
ity of centered vectors. (a-b) Vectors clustered by task
type; (c-d) Vectors clustered by interaction target. The
diagonal block structure suggests that the extracted vec-
tors align with task-specific categories and share related
behavioral features.
Grouping the vectors by specific interaction tar-
gets in Figures 5c and 5d highlights functional dif-
ferences between the two extraction strategies. The
intra-trajectory contrasts form diagonal block struc-
tures with high similarity. Local error-correction
signals associated with particular items like a
DeskLamporDrawerremain uniform across differ-
ent task instances. Correcting localized degenerate
steps relies primarily on the intrinsic properties
of the target object rather than the broader task
context. The inter-trajectory contrasts show a less
structured pattern with lower consistency. Global
reasoning paths depend on the initial state of the
agent, environmental layouts, and prior execution
sequences. This contextual variance prevents inter-
trajectory vectors from forming the distinct clusters
observed in the local signals.
5.3 Feature Decomposition and
Interpretability of Steering Vectors
To understand the behavioral semantics encoded
within the steering vectors, we decompose the con-
tinuous representations into interpretable basis di-
rections using sparse dictionary learning (detailed
in Appendix B). Each direction corresponds to spe-
cific behavioral primitives, which we validate by
computing the mutual information between basis
activations and the generated action types across
the dataset. Projecting the steering vectors onto
these bases reveals that the two contrastive strate-
7

0 5 10 15 20 25 3033
StepF 0 [-]  RedundObs
F 1 [-]  IneffBacktrack
F 2 [+]  RetrySearch
F 3 [-]  BadExam
F 4 [-]  BadInspect
F 6 [+]  FinalPlace
F 7 [-]  EarlyStop
F 8 [+]  InitPlan
F 9 [+]  OpenTarget
F10 [o]  ImmedExam
F11 [o]  CabSearch
F13 [+]  SysSearch
F14 [+]  ContSearchUnsteered (r=0.0)
0 5 10 14
StepInter-traj (r=1.0)
0 5 10 15 20 24
StepIntra-traj (r=1.0)
0.00.20.40.60.81.01.21.4 Inter Intra
F 00.17
 0.66
F 10.12
 0.32
F 20.12
 0.17
F 3
F 4
F 60.21
F 7
 0.43
F 80.19
 0.65
F 90.13
F10
F11
 0.45
F13
 0.30
F14
 0.26
Task: heat some apple and put it in garbagecan.Figure 6:Temporal activation of behavioral primitives across execution steps in a PickHeat task. (Left)The
unsteered baseline exhibits an extended 33-step trajectory.(Middle)Inter-trajectorysteering is associated with a
shorter trajectory by promoting early planning and object placement.(Right)Intra-trajectorysteering is observed
to amplify search features and suppress premature task termination signals during execution.
1 2 4 8 All
Retrieval Pool Size K0.04
0.03
0.02
0.01
0.000.010.020.03 Activation (from K=1)
n=7 tasks (intersection across K)pick_heat_then_place_in_recep    Inter-traj
F6: Executing final placement of prepared object
F8: Initial object search planning for items
F9: Opening closed containers to search for target objects
F0: Inefficient exploration and redundant observation
1 2 4 8 All
Retrieval Pool Size K0.08
0.06
0.04
0.02
0.000.020.040.06 Activation (from K=1)
n=8 tasks (intersection across K)pick_heat_then_place_in_recep    Intra-traj
F13: Systematic cabinet searching after failed inspection
F7: Premature task completion declaration
F0: Inefficient exploration and redundant observation
Figure 7: The impact of retrieval pool size on specific
behavioral primitive activations (Solid lines for positive
features; dashed lines for negative features).
gies operate through distinct mechanistic pathways.
ThePickHeattask in Figure 6 illustrates how
the two extraction strategies modulate representa-
tions through distinct mechanisms. Without inter-
vention, the baseline model generates an extended
33-step trajectory. Applying inter-trajectory steer-
ing leads to global behavioral changes. This vector
increases early planning features (F8) and sustains
activations related to concrete object placement
(F6), shortening the execution to 14 steps. Intra-
trajectory steering targets local execution errors
instead of broad execution plans. It corrects ac-
tion loops by amplifying features associated with
targeted searching (F13) and container interaction
(F14) while suppressing signals that trigger pre-
mature task termination (F7). This difference in
activation patterns suggests that the steering vec-
tors target distinct aspects of model behavior. They
systematically adjust representations to address dif-
ferent types of procedural errors (see more quanti-
tative analysis in Appendix C.3).5.4 Retrieval Size and Dynamic Synthesis
To investigate how the retrieval size affects the ex-
tracted procedural signals, we track the activation
of behavioral primitives as the number of aggre-
gated contrastive pairs increases. As shown in Fig-
ure 7, features display two distinct scaling patterns
depending on their universality. For general behav-
iors, such as basic exploration (F9) and redundant
observation (F0), activation magnitudes grow con-
tinuously as the size expands. Since these patterns
are shared across most tasks, they remain promi-
nent when aggregated across the entire dataset.
Conversely, task-specific primitives exhibit a dif-
ferent pattern. Features tied to explicit procedu-
ral logic, such as placing objects (F6), systematic
searching (F13), or preventing premature comple-
tion (F7), reach peak activation at a moderate re-
trieval scale before declining. While minimal pools
fail to separate the core procedural signal from
instance-level noise, overly large pools introduce
cross-task interference that suppresses specific task
rules. This observation confirms the necessity of
dynamic synthesis: constraining the retrieval to a
contextually relevant subset effectively balances
noise reduction with signal preservation.
6 Conclusion
We propose Neural Procedural Memory (NPM), a
training-free framework that represents and applies
procedural behaviors through implicit activation
steering. By distilling procedural knowledge from
dual-granularity contrastive experiences, NPM pro-
vides an alternative paradigm for encoding mem-
ory that operates directly in the activation space.
Extensive experiments show that NPM achieves
performance comparable to textual baselines and
provides complementary benefits when combined
with explicit workflows. By utilizing the contin-
uous activation space to encode procedural skills,
NPM provides a practical foundation for managing
procedural memory in LLM agents.
8

Limitations
While our proposed approach demonstrates promis-
ing results, there are several aspects that could be
improved. Firstly, the framework requires direct
intervention in the residual stream, restricting its
application to open architectures where internal
activation spaces are completely accessible. Fur-
thermore, building the contrastive repository relies
on the agent occasionally generating successful tra-
jectories, which introduces a cold-start problem in
complex environments. Additionally, extracting
degenerate steps relies on heuristics such as action
redundancy or format invalidity. These conditions
struggle to identify implicit logical fallacies with-
out explicit environment errors. Finally, synthe-
sized vectors compress procedural knowledge into
a single representation applied constantly during
generation. This static modulation lacks the flexi-
bility to shift between behavioral primitives across
different execution stages. Exploring dynamic in-
terventions that adapt throughout task execution is
a direction for future work.
References
Steven Bills, Nick Cammarata, Dan Moss-
ing, Henk Tillman, Leo Gao, Gabriel Goh,
Ilya Sutskever, Jan Leike, Jeff Wu, and
William Saunders. 2023. Language mod-
els can explain neurons in language models.
https://openaipublic.blob.core.windows.
net/neuron-explainer/paper/index.html.
Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and
Danqi Chen. 2023. Adapting language models to
compress contexts. InProceedings of the 2023 Con-
ference on Empirical Methods in Natural Language
Processing, pages 3829–3846, Singapore. Associa-
tion for Computational Linguistics.
Maxime Chevalier-Boisvert, Dzmitry Bahdanau, Salem
Lahlou, Lucas Willems, Chitwan Saharia, Thien Huu
Nguyen, and Yoshua Bengio. 2019. BabyAI: First
steps towards grounded language learning with a hu-
man in the loop. InInternational Conference on
Learning Representations.
Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet
Singh, and Deshraj Yadav. 2025. Mem0: Building
production-ready ai agents with scalable long-term
memory.Preprint, arXiv:2504.19413.
Yilin Geng, Haonan Li, Honglin Mu, Xudong Han,
Timothy Baldwin, Omri Abend, Eduard Hovy, and
Lea Frermann. 2025. Control illusion: The failure
of instruction hierarchies in large language models.
Preprint, arXiv:2502.15851.Bernal Jimenez Gutierrez, Yiheng Shu, Yu Gu, Michi-
hiro Yasunaga, and Yu Su. 2024. HippoRAG: Neu-
robiologically inspired long-term memory for large
language models. InThe Thirty-eighth Annual Con-
ference on Neural Information Processing Systems.
Dongge Han, Camille Couturier, Daniel Madrigal Diaz,
Xuchao Zhang, Victor Rühle, and Saravan Rajmohan.
2025. Legomem: Modular procedural memory for
multi-agent llm systems for workflow automation.
Preprint, arXiv:2510.04851.
Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu
Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang
Huang, Weilin Zhao, and 1 others. 2024. Minicpm:
Unveiling the potential of small language models
with scalable training strategies.arXiv preprint
arXiv:2404.06395.
Yuyang Hu, Shichun Liu, Yanwei Yue, Guibin Zhang,
Boyang Liu, Fangyi Zhu, Jiahang Lin, Honglin Guo,
Shihan Dou, Zhiheng Xi, Senjie Jin, Jiejun Tan, Yan-
bin Yin, Jiongnan Liu, Zeyu Zhang, Zhongxiang
Sun, Yutao Zhu, Hao Sun, Boci Peng, and 28 others.
2025. Memory in the age of ai agents.Preprint,
arXiv:2512.13564.
Taewoon Kim, Michael Cochez, Vincent François-
Lavet, Mark Neerincx, and Piek V ossen. 2023. A
machine with short-term, episodic, and semantic
memory systems. InProceedings of the Thirty-
Seventh AAAI Conference on Artificial Intelligence
and Thirty-Fifth Conference on Innovative Applica-
tions of Artificial Intelligence and Thirteenth Sympo-
sium on Educational Advances in Artificial Intelli-
gence, AAAI’23/IAAI’23/EAAI’23. AAAI Press.
Bruce W. Lee, Inkit Padhi, Karthikeyan Natesan Ra-
mamurthy, Erik Miehling, Pierre Dognin, Manish
Nagireddy, and Amit Dhurandhar. 2025. Program-
ming refusal with conditional activation steering. In
The Thirteenth International Conference on Learning
Representations.
Bang Liu, Xinfeng Li, Jiayi Zhang, Jinlin Wang, Tanjin
He, Sirui Hong, Hongzhang Liu, Shaokun Zhang,
Kaitao Song, Kunlun Zhu, Yuheng Cheng, Suyuchen
Wang, Xiaoqiang Wang, Yuyu Luo, Haibo Jin, Peiyan
Zhang, Ollie Liu, Jiaqi Chen, Huan Zhang, and 29
others. 2025. Advances and challenges in foundation
agents: From brain-inspired intelligence to evolu-
tionary, collaborative, and safe systems.Preprint,
arXiv:2504.01990.
Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou,
and Yue Zhang. 2025. An empirical study of catas-
trophic forgetting in large language models during
continual fine-tuning.IEEE Transactions on Audio,
Speech and Language Processing, 33:3776–3786.
Kyle Mahowald, Anna A Ivanova, Idan A Blank, Nancy
Kanwisher, Joshua B Tenenbaum, and Evelina Fe-
dorenko. 2024. Dissociating language and thought in
large language models.Trends in cognitive sciences,
28(6):517–540.
9

Samuel Marks and Max Tegmark. 2024. The geometry
of truth: Emergent linear structure in large language
model representations of true/false datasets. InFirst
Conference on Language Modeling.
Parsa Omidi, Xingshuai Huang, Axel Laborieux, Ba-
hareh Nikpour, Tianyu Shi, and Armaghan Eshaghi.
2025. Memory-augmented transformers: A system-
atic review from neuroscience principles to enhanced
model architectures.Preprint, arXiv:2508.10824.
Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang,
Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez.
2024. Memgpt: Towards llms as operating systems.
Preprint, arXiv:2310.08560.
Nina Panickssery, Nick Gabrieli, Julian Schulz, Meg
Tong, Evan Hubinger, and Alexander Matt Turner.
2024. Steering llama 2 via contrastive activation
addition.Preprint, arXiv:2312.06681.
Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Mered-
ith Ringel Morris, Percy Liang, and Michael S. Bern-
stein. 2023. Generative agents: Interactive simulacra
of human behavior. InProceedings of the 36th An-
nual ACM Symposium on User Interface Software
and Technology, UIST ’23, New York, NY , USA.
Association for Computing Machinery.
Gonçalo Santos Paulo, Alex Troy Mallen, Caden Juang,
and Nora Belrose. 2025. Automatically interpreting
millions of features in large language models. In
Forty-second International Conference on Machine
Learning.
Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong,
Evan Hubinger, and Alexander Turner. 2024. Steer-
ing llama 2 via contrastive activation addition. In
Proceedings of the 62nd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 15504–15522, Bangkok, Thai-
land. Association for Computational Linguistics.
Daniel Scalena, Gabriele Sarti, and Malvina Nissim.
2024. Multi-property steering of large language mod-
els with dynamic activation composition. InProceed-
ings of the 7th BlackboxNLP Workshop: Analyzing
and Interpreting Neural Networks for NLP, pages
577–603, Miami, Florida, US. Association for Com-
putational Linguistics.
Yunfan Shao, Linyang Li, Junqi Dai, and Xipeng Qiu.
2023. Character-LLM: A trainable agent for role-
playing. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Process-
ing, pages 13153–13187, Singapore. Association for
Computational Linguistics.
Noah Shinn, Federico Cassano, Ashwin Gopinath,
Karthik R Narasimhan, and Shunyu Yao. 2023. Re-
flexion: language agents with verbal reinforcement
learning. InThirty-seventh Conference on Neural
Information Processing Systems.
Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté,
Yonatan Bisk, Adam Trischler, and MatthewHausknecht. 2021. ALFWorld: Aligning Text and
Embodied Environments for Interactive Learning.
InProceedings of the International Conference on
Learning Representations (ICLR).
Akshit Sinha, Arvindh Arun, Shashwat Goel, Steffen
Staab, and Jonas Geiping. 2025. The illusion of di-
minishing returns: Measuring long horizon execution
in llms.Preprint, arXiv:2509.09677.
Larry R. Squire. 1992. Declarative and nondeclarative
memory: Multiple brain systems supporting learning
and memory.Journal of Cognitive Neuroscience,
4(3):232–243.
Larry R. Squire. 2004. Memory systems of the brain: A
brief history and current perspective.Neurobiology
of Learning and Memory, 82(3):171–177. Multiple
Memory Systems.
Wangtao Sun, Chenxiang Zhang, Xueyou Zhang, Xuan-
qing Yu, Ziyang Huang, Haotian Xu, Shizhu He, Jun
Zhao, and Kang Liu. 2025. Beyond instruction fol-
lowing: Evaluating inferential rule following of large
language models. InChinese Computational Linguis-
tics: 24th China National Conference, CCL 2025, Ji-
nan, China, August 11–14, 2025, Proceedings, page
408–434, Berlin, Heidelberg. Springer-Verlag.
Constantin Venhoff, Iván Arcuschin, Philip Torr, Arthur
Conmy, and Neel Nanda. 2025. Base models know
how to reason, thinking models learn when.Preprint,
arXiv:2510.07364.
Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Man-
dlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and
Anima Anandkumar. 2023. V oyager: An open-ended
embodied agent with large language models.arXiv
preprint arXiv: Arxiv-2305.16291.
Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao
Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang,
Xu Chen, Yankai Lin, and 1 others. 2024a. A survey
on large language model based autonomous agents.
Frontiers of Computer Science, 18(6):186345.
Ruoyao Wang, Peter Jansen, Marc-Alexandre Côté, and
Prithviraj Ammanabrolu. 2022. Scienceworld: Is
your agent smarter than a 5th grader? InProceedings
of the 2022 Conference on Empirical Methods in
Natural Language Processing, pages 11279–11298.
Tiannan Wang, Meiling Tao, Ruoyu Fang, Huilin Wang,
Shuai Wang, Yuchen Eleanor Jiang, and Wangchun-
shu Zhou. 2024b. Ai persona: Towards life-long
personalization of llms.Preprint, arXiv:2412.13103.
Yu Wang, Yifan Gao, Xiusi Chen, Haoming Jiang,
Shiyang Li, Jingfeng Yang, Qingyu Yin, Zheng
Li, Xian Li, Bing Yin, Jingbo Shang, and Ju-
lian McAuley. 2024c. Memoryllm: towards self-
updatable large language models. InProceedings of
the 41st International Conference on Machine Learn-
ing, ICML’24. JMLR.org.
10

Yu Wang, Xinshuang Liu, Xiusi Chen, Sean O’Brien,
Junda Wu, and Julian McAuley. 2025a. Self-
updatable large language models by integrating con-
text into model parameters. InThe Thirteenth Inter-
national Conference on Learning Representations.
Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Gra-
ham Neubig. 2025b. Agent workflow memory. In
Forty-second International Conference on Machine
Learning.
Bosi Wen, Pei Ke, Xiaotao Gu, Lindong Wu, Hao
Huang, Jinfeng Zhou, Wenchuang Li, Binxin Hu,
Wendy Gao, Jiaxin Xu, Yiming Liu, Jie Tang, Hongn-
ing Wang, and Minlie Huang. 2024. Benchmark-
ing complex instruction-following with multiple con-
straints composition. InProceedings of the 38th
International Conference on Neural Information Pro-
cessing Systems, NIPS ’24, Red Hook, NY , USA.
Curran Associates Inc.
Yuchen Xia, Jiho Kim, Yuhan Chen, Haojie Ye, Souvik
Kundu, Cong Callie Hao, and Nishil Talati. 2024.
Understanding the performance and estimating the
cost of llm fine-tuning. In2024 IEEE International
Symposium on Workload Characterization (IISWC),
pages 210–223.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao,
Chengen Huang, Chenxu Lv, Chujie Zheng, Dayi-
heng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge,
Haoran Wei, Huan Lin, Jialong Tang, and 41 oth-
ers. 2025. Qwen3 technical report.arXiv preprint
arXiv:2505.09388.
Shunyu Yao, Howard Chen, John Yang, and Karthik
Narasimhan. 2022. Webshop: Towards scalable real-
world web interaction with grounded language agents.
Advances in Neural Information Processing Systems,
35:20744–20757.
Weiran Yao, Shelby Heinecke, Juan Carlos Niebles,
Zhiwei Liu, Yihao Feng, Le Xue, Rithesh Murthy,
Zeyuan Chen, Jianguo Zhang, Devansh Arpit, Ran
Xu, Phil Mui, Huan Wang, Caiming Xiong, and Sil-
vio Savarese. 2024. Retroformer: Retrospective large
language agents with policy gradient optimization.
Preprint, arXiv:2308.02151.
Tao Yu, Zhengbo Zhang, Zhiheng Lyu, Junhao Gong,
Hongzhu Yi, Xinming Wang, Yuxuan Zhou, Ji-
abing Yang, Ping Nie, Yan Huang, and Wenhu Chen.
2025a. Browseragent: Building web agents with
human-inspired web browsing actions.Preprint,
arXiv:2510.10666.
Xinlei Yu, Chengming Xu, Guibin Zhang, Zhangquan
Chen, Yudong Zhang, Yongbo He, Peng-Tao Jiang,
Jiangning Zhang, Xiaobin Hu, and Shuicheng Yan.
2025b. Vismem: Latent vision memory unlocks
potential of vision-language models.Preprint,
arXiv:2511.11007.
Yuexiang Zhai, Shengbang Tong, Xiao Li, Mu Cai, Qing
Qu, Yong Jae Lee, and Y . Ma. 2023. Investigating thecatastrophic forgetting in multimodal large language
models.ArXiv, abs/2309.10313.
Guibin Zhang, Muxin Fu, and Shuicheng Yan. 2025a.
Memgen: Weaving generative latent memory for self-
evolving agents.Preprint, arXiv:2509.24704.
Kai Zhang, Xiangchao Chen, Bo Liu, Tianci Xue,
Zeyi Liao, Zhihan Liu, Xiyao Wang, Yuting Ning,
Zhaorun Chen, Xiaohan Fu, Jian Xie, Yuxuan Sun,
Boyu Gou, Qi Qi, Zihang Meng, Jianwei Yang,
Ning Zhang, Xian Li, Ashish Shah, and 11 oth-
ers. 2025b. Agent learning via early experience.
Preprint, arXiv:2510.08558.
Zeyu Zhang, Quanyu Dai, Xiaohe Bo, Chen Ma, Rui Li,
Xu Chen, Jieming Zhu, Zhenhua Dong, and Ji-Rong
Wen. 2025c. A survey on the memory mechanism of
large language model-based agents.ACM Trans. Inf.
Syst., 43(6).
Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu
Lin, Yong-Jin Liu, and Gao Huang. 2024. Expel:
Llm agents are experiential learners. InProceed-
ings of the Thirty-Eighth AAAI Conference on Ar-
tificial Intelligence and Thirty-Sixth Conference on
Innovative Applications of Artificial Intelligence and
Fourteenth Symposium on Educational Advances in
Artificial Intelligence, AAAI’24/IAAI’24/EAAI’24.
AAAI Press.
Longtao Zheng, Rundong Wang, Xinrun Wang, and
Bo An. 2023. Synapse: Trajectory-as-exemplar
prompting with memory for computer control. In
The Twelfth International Conference on Learning
Representations.
Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and
Yanlin Wang. 2024. Memorybank: Enhancing large
language models with long-term memory.Proceed-
ings of the AAAI Conference on Artificial Intelligence,
38(17):19724–19731.
Andy Zou, Long Phan, Sarah Chen, James Campbell,
Phillip Guo, Richard Ren, Alexander Pan, Xuwang
Yin, Mantas Mazeika, Ann-Kathrin Dombrowski,
Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan
Wang, Alex Mallen, Steven Basart, Sanmi Koyejo,
Dawn Song, Matt Fredrikson, and 2 others. 2025.
Representation engineering: A top-down approach
to ai transparency.Preprint, arXiv:2310.01405.
A Implementation Details
A.1 Dataset Details
Our evaluation utilizes four simulated environ-
ments that test diverse procedural skills and long-
horizon planning capabilities.
ALFWorld (Shridhar et al., 2021) is a text-based
embodied environment where agents navigate sim-
ulated rooms and interact with household objects to
complete daily tasks. WebShop (Yao et al., 2022)
provides a simulated e-commerce website where
11

agents must execute search queries, navigate prod-
uct pages, and select specific item attributes to pur-
chase products matching detailed user instructions.
ScienceWorld (Wang et al., 2022) simulates an ele-
mentary school science laboratory, requiring agents
to follow strict procedural logic to conduct exper-
iments. BabyAI (Chevalier-Boisvert et al., 2019)
is a partially observable gridworld environment
where agents execute sequential movement and ma-
nipulation commands to complete specified spatial
tasks.
A.2 Data Construction and Model
Configuration
We construct the contrastive dataset by sampling
tasks directly from the training splits of ALFWorld,
WebShop, ScienceWorld, and BabyAI. For each
training task, we generate multiple trajectories us-
ing the target model configured with specific sam-
pling parameters.
To isolate degenerate behaviors for intra-
trajectory contrasts, we apply deterministic rules
based on action redundancy and invalidity. Redun-
dancy is identified through pattern matching that
detects repetitive loops or consecutive identical ac-
tion sequences. Categorizing segments where the
agent enters a state unable to progress normally
as degenerate steps provides a reasonable heuris-
tic for capturing procedural failures. Invalidity is
determined by matching environment-specific er-
ror messages and format violations, which filters
irrational actions from the effective trajectory pool.
All experiments employ
sentence-transformers/all-mpnet-base-v2
for dense retrieval to retrieve a fixed number of
the most relevant historical tasks. The agent then
performs greedy decoding to ensure the repro-
ducibility of the evaluation results. The detailed
sampling parameters and retrieval configurations
for all evaluated environments are summarized in
Table 2.
A.3 Dynamic Synthesis and Vector Extraction
To determine the intervention granularity g, we es-
timate task complexity using the average execution
length of the retrieved successful trajectories, de-
noted as Lq. We compare this length against the
global average trajectory length of the respective
benchmark, denoted as ¯L. Tasks with extended
horizons typically encounter localized procedural
failures such as repetitive loops, making the con-
centrated error-correction of intra-trajectory steer-Table 2: Data construction parameters. All parameters
except the maximum interaction steps are kept consis-
tent across the four evaluated benchmarks.
Parameter Value
Sampled trajectories (N) 16
Temperature 0.7
Top-p0.8
Top-k20
Max turns (ALFWorld) 50
Max turns (WebShop) 15
Max turns (ScienceWorld) 30
Max turns (BabyAI) 20
Retrieval count (K) 8
ing more appropriate. Conversely, shorter tasks
typically fail due to misaligned global planning,
which benefits from the broader behavioral align-
ment provided by inter-trajectory steering. The
system selects the appropriate granularity based
on a simple thresholding strategy, with the scaling
factorγset to 1.5 in our experiments:
g=(
intra-trajectory,ifL q> γ¯L
inter-trajectory,otherwise(6)
Nevertheless, identifying the optimal granularity
for every instance remains a non-trivial challenge.
Our heuristic selection occasionally scores lower
than the oracle single-granularity baseline in spe-
cific settings. Future research could explore more
effective methods for allocating steering vectors.
After selecting the appropriate contrastive pool,
we extract the steering vectors using a standard
principal component analysis pipeline. The paired
positive and negative activations are first mean-
centered to eliminate shared task-specific biases.
We then compute the principal components of these
normalized representations and extract the domi-
nant component as the final continuous steering
vector.
A.4 Steering Configuration
We apply activation steering to the middle-to-late
layers of the language models. Specifically, we
inject the steering vectors into layers 17 to 19 for
Qwen3-4B and Qwen3-8B, and layers 46 to 48 for
MiniCPM3-4B.
To achieve dynamic steering across these tar-
geted layers, we adapt a simplified version of the
calibration mechanism based on Kullback-Leibler
12

divergence proposed by Scalena et al. (2024).
This method tracks the probability shift caused
by the steering vector and reduces the interven-
tion strength if the modified next-token distribution
Pα(· |x <t)deviates from the original unsteered
distribution P(· |x <t)by a large margin. We de-
fine a discrete set of candidate scales Aand select
the optimal strength α∗by identifying the maxi-
mum value that keeps the divergence below a target
thresholdϵ:
α∗= maxn
α∈ ADKL 
P(· |x <t)
∥Pα(· |x <t)
≤ϵo(7)
B Steering Vector Interpretability and
Feature Analysis Details
B.1 Feature Extraction and Semantic
Annotation
To investigate the underlying mechanisms of im-
plicit activation steering, we decompose the contin-
uous representation space into interpretable basis
directions following the methodology of Venhoff
et al. (2025). We collect step-level hidden states
from historical interaction trajectories at layer 18
for the Qwen3-4B model and apply a sparse dic-
tionary learning algorithm with a top-k activation
constraint. The dictionary size is constrained to 16
dimensions with a sparsity threshold of 3. This di-
mensional configuration aligns with prior research
on isolating specific high-level procedural behav-
iors. The empirical observation of inactive dimen-
sions within this learned dictionary indicates that a
16-dimensional space provides adequate capacity
for capturing the procedural variations specific to
agentic task-solving steps.
We assign interpretable semantics to these
learned basis directions by analyzing their acti-
vation patterns across the dataset. We quantify
behavioral selectivity by calculating the mutual in-
formation between the activation state of each basis
direction fand the discrete action type agenerated
by the agent. This statistical measurement is de-
fined as:
MI(F;A) =X
a"
P(f, a) logP(f, a)
P(f)P(a)
+P(¬f, a) logP(¬f, a)
P(¬f)P(a)#
(8)Building on established automated interpretabil-
ity techniques (Bills et al., 2023; Paulo et al., 2025),
we utilize DeepSeek-V3.2 to analyze these statisti-
cal distributions alongside top-activating examples.
The model processes this information according to
the prompt template detailed in Figure 8 to generate
descriptive textual labels and determine the behav-
ioral polarity for each active basis direction. The
resulting semantic annotations for the ALFWorld
benchmark are summarized in Table 3.
B.2 Geometric Projection of Steering Vectors
We analyze the functional composition of the ex-
tracted principal component steering vectors by
projecting them onto the annotated basis directions.
The geometric projection is computed as the dot
product between the normalized steering vector v
and each unit-norm basis vector widerived from
the dictionary learning phase:
pi=w⊤
iv
∥v∥(9)
A positive projection value pi>0 indicates
that the steering intervention amplifies the corre-
sponding behavioral primitive. A negative value
pi<0 signifies suppression of that specific be-
havior. This mathematical decomposition confirms
that the steering vectors function as composite mod-
ifiers that simultaneously inhibit pathological pat-
terns and stimulate necessary subsequent steps to
correct agent reasoning.
C Quantitative Validation of
Methodology
C.1 Linear Separability in High-Dimensional
Manifold
We train a linear support vector machine on the
hidden states extracted from the target intervention
layers of the retrieved trajectories to verify the lin-
ear separability of successful and failed procedural
logic. This evaluation determines whether effective
and degenerate reasoning modes occupy distinct
regions within the representation space (Marks and
Tegmark, 2024; Zou et al., 2025). We perform 5-
fold stratified cross-validation on the Qwen3-4B
model across all four evaluated benchmarks. The
classification accuracies at both the inter-trajectory
and intra-trajectory levels are reported in Table 4.
The consistently high classification accuracy across
different environments confirms that a linear hyper-
plane effectively divides the two reasoning classes.
13

Table 3: Automated annotations and activation statistics of the learned basis directions in the ALFWorld environment.
Index Semantic LabelOverall
ActiveActive on
SuccessActive on
FailurePolarity
0 RedundObs: Inefficient exploration and redun-
dant observation42.0% 7.3% 46.2% Negative
1 IneffBacktrack: Ineffective backtracking after
failed object placement10.0% 0.5% 11.1% Negative
2 RetrySearch: Searching for objects after failed
container checks17.9% 24.3% 17.1% Positive
3 BadExam: Misguided final-step execution
with examination tasks1.5% 0.8% 1.6% Negative
4 BadInspect: Misguided visual inspection of
desk objects23.5% 7.7% 25.4% Negative
5Dead: Dead feature 0.0% 0.0% 0.0% N/A
6 FinalPlace: Executing final placement of pre-
pared object23.1% 33.0% 21.9% Positive
7 EarlyStop: Premature task completion decla-
ration36.5% 8.0% 40.0% Negative
8 InitPlan: Initial object search planning for
food items32.6% 67.1% 28.4% Positive
9 OpenTarget: Opening closed containers to
search for target objects20.9% 22.7% 20.7% Positive
10 ImmedExam: Immediate visual examination
of target objects6.4% 6.3% 6.5% Neutral
11 CabSearch: Opening closed cabinets to search
for objects22.1% 24.7% 21.8% Neutral
12Dead: Dead feature 0.0% 0.0% 0.0% N/A
13 SysSearch: Systematic cabinet searching after
empty inspection41.2% 60.0% 38.9% Positive
14 ContSearch: Opening closed containers to
search for objects22.3% 37.6% 20.4% Positive
15Dead: Dead feature 0.0% 0.0% 0.0% N/A
This mathematical separability provides the theoret-
ical justification for applying principal component
analysis to extract a dominant directional vector for
behavioral modulation.
Table 4: Linear SVM classification accuracy (%) on
the native high-dimensional hidden states of Qwen3-4B
across 5-fold stratified cross-validation.
Benchmark Inter-traj. Intra-traj.
ALFWorld 99.55 (±0.51) 99.99 (±0.01)
WebShop 88.46 (±0.55) 99.91 (±0.10)
ScienceWorld 93.41 (±0.69) 99.81 (±0.12)
BabyAI 99.53 (±0.23) 99.61 (±0.10)C.2 Accuracy of Heuristic Rules for
Degenerate Steps
The efficacy of intra-trajectory steering depends
on isolating degenerate steps without contaminat-
ing the negative representation pool. We evaluate
the reliability of our deterministic isolation rules
based on consecutive loop detection and invalid
format matching by analyzing sampled steps from
1,000 historical trajectories per benchmark. An
LLM-based evaluator assesses each step against
the criteria of action redundancy and environment
constraints to compute the false positive and false
negative rates. The evaluation metrics presented
in Table 5 show a low overall false positive rate
of 0.61% and a low overall false negative rate of
6.34%, indicating that the heuristics reliably iso-
14

late genuine procedural failures and maintain high
purity in the negative representations. The minor
variance in error rates across benchmarks, such as
the slightly higher false negative rate of 9.30% in
WebShop, occurs because rigid deterministic rules
can occasionally miss subtle, domain-specific logi-
cal errors. These undetected failures subsequently
mix into the effective step pool. Although this
introduces noise into the positive representations,
the empirical results confirm that the overall con-
trastive signal remains robust for behavioral mod-
ulation. Future work could address this limitation
by replacing heuristic rules with more capable eval-
uation mechanisms, such as utilizing a language
model as a judge to identify nuanced reasoning
failures during data construction.
Table 5: Evaluation of heuristic rule accuracy based
on step-level annotation across four benchmarks. False
positive rate (FPR) and false negative rate (FNR) are
evaluated by an LLM-as-a-judge on sampled steps from
1,000 trajectories per environment.
Benchmark FPR FNR
ALFWorld 0.00% (0/2,862) 3.57% (106/2,967)
WebShop 0.64% (16/2,490) 9.30% (268/2,883)
ScienceWorld 0.04% (1/2,833) 6.97% (184/2,640)
BabyAI 1.75% (51/2,908) 5.73% (171/2,983)
Overall 0.61% (68/11,093) 6.34% (729/11,473)
C.3 Distribution Analysis of Inter- and
Intra-Trajectory Feature Activations
To evaluate how steering vectors modulate behav-
iors, we analyze their geometric distribution across
the basis directions defined in Equation 9 using nor-
malized entropy and top-three concentration. For
any task, the projection proportion along each basis
direction is:
˜pi=|pi|PN
j=1|pj|(10)
The global uniformity of this distribution is cap-
tured by the normalized entropy:
Hnorm=−PN
i=1˜pilog ˜p i
logN(11)
where Ndenotes the total number of active direc-
tions. To measure local focus, we also compute
the top-three concentration, representing the cu-
mulative proportion of the three largest absolute
projection values:
Ctop3=P
i∈Ω|pi|
PN
j=1|pj|(12)withΩdenoting the indices of these three dominant
directions.
Empirical analysis of these metrics reveals dis-
tinct structural differences between the two steering
granularities. As shown in Figure 9, inter-trajectory
steering vectors exhibit a mean normalized entropy
of 0.919, whereas intra-trajectory vectors yield a
lower mean of 0.885. This difference indicates that
macro-level planning steering operates via a more
distributed modulation across multiple basis direc-
tions. Conversely, Figure 10 shows that the mean
top-three concentration is 38.0% for inter-trajectory
steering and 43.6% for intra-trajectory steering.
The higher concentration of intra-trajectory vectors
indicates a more targeted activation pattern. In-
stead of adjusting the overall reasoning flow, intra-
trajectory intervention selectively modulates spe-
cific behavioral dimensions to resolve localized
failures. These patterns suggest that global plan-
ning guidance tends to rely on distributed represen-
tations, whereas local error recovery is associated
with targeted corrections.
D System Efficiency and Overhead
D.1 Storage and Computation Overhead
The framework introduces minimal storage over-
head compared to text-based retrieval. For each
historical trajectory, the system extracts three rep-
resentations at each targeted layer: one last-token
state for inter-trajectory contrast and two mean-
pooled states representing effective and degenerate
steps for intra-trajectory contrast. Given a hidden
dimension d, floating-point byte size b, andLtarget
layers, the total storage requirement per trajectory
is3×L×d×b . For Qwen3-4B with d= 2560
in half-precision format ( b= 2 ) targeting three
layers, the footprint is 45 kilobytes. This fixed con-
straint ensures that storage scales independently
of the original interaction length. Unlike tradi-
tional retrieval systems that consume substantial
video memory by expanding the key-value cache
with extensive textual contexts, activation steering
modifies the residual stream directly to bypass this
limitation. The offline computation for memory
construction is limited to a single forward pass to
extract representations. These continuous vectors
can be efficiently managed in a standard database
without requiring model parameter updates.
15

D.2 Latency Analysis
We evaluate the inference latency of the proposed
framework across retrieval, scale selection probing,
vector synthesis, prompt prefill, and autoregressive
decoding. The timing results in Table 6 indicate
that textual memory incurs a substantial prefill over-
head, increasing latency from 63.46 ms to 279.89
ms, as the self-attention mechanism must process
extended textual contexts. In contrast, NPM main-
tains a prefill latency of 71.09 ms, nearly identi-
cal to the No Memory baseline. Although NPM
introduces additional phases to compute steering
vectors and steering strength, this process imposes
no extra latency penalty relative to textual memory.
During autoregressive decoding, vector injection
involves only an element-wise addition within the
residual stream. This basic operation adds no mea-
surable delay to token generation. These efficiency
characteristics establish implicit activation steering
as a practical and scalable paradigm for persistent
agentic memory.
Table 6: Inference latency breakdown (ms).
Method Retrieval Synthesis Probe Prefill
No Memory — — — 63.46
Textual Memory 13.83 — — 279.89
NPM (Ours) 13.83 77.42 130.46 71.09
E Examples and Case Studies
E.1 Examples of Textual Procedural Memory
Examples are provided in Figure 11, 12.
E.2 Case Study: Alleviating Text-Action
Disconnect in Action Execution
We examine agent behavior in a multi-object re-
trieval task requiring the placement of two CDs
into a safe (Table 7). The baseline model receives
an explicit textual workflow outlining the neces-
sary sequential operations and repetition condi-
tions. When the environment rejects an attempt
to pick up a second object due to inventory con-
straints, the text-augmented agent fails to update
its internal state representation. The model gener-
ates subsequent commands based on an incorrect
assumption of successful object acquisition and en-
ters an invalid action loop of attempting to place an
unpossessed item into the receptacle. This behav-
ior demonstrates the text-action disconnect where
semantic comprehension of discrete textual rules
does not ensure sustained state maintenance duringexecution. The agent equipped with neural proce-
dural memory completes the entire fetch-and-place
cycle without textual prompting. Intervening di-
rectly in the residual stream allows the synthesized
steering vectors to suppress neural activations as-
sociated with redundant loops while amplifying
features related to sequential planning as identi-
fied in prior mechanistic analyses. Modulating the
continuous activation space directly enforces the
procedural logic required for multi-step execution
and prevents the trajectory from degrading into the
repetitive errors observed when relying solely on
external textual memory.
16

Table 7: Qualitative comparison of procedural memory in the “find two cd and put them in safe" task. Despite the
Textual Workflowproviding explicit instructions, the agent fails to manage the multi-step trip after an initial action
failure.NPMsteers the agent implicitly adhering to the sequential requirements.
Task Your task is to: find two cd and put them in safe.
Memory Textual Workflow:
“[Workflow: search_and_place_multiple_objects]
...take {object 1}from {location} ...go to
{target_receptacle}... ”NPM (Ours):
Implicit steering vector distilled from historical
multi-object retrieval tasks.
Discovery Both agents successfully locate two target CDs in the environment and attempt to pick them up.
Constraint Hit Action: take cd 1→go to desk 2→take
cd 2
Obs:(Action rejected as hands are full).
Thought:“I have taken cd 2 from desk 2. Now
I need to go to the safe to place both cds.”(Er-
roneous state tracking)Action: take cd 3→go to desk 1→take
cd 1
Obs:(Action rejected as hands are full).
Thought:“I have found the second cd (cd 1)
on desk 1. Now I need to go to the safe and put
both cds into it.”(Correct multi-step planning)
Procedural Divergence Thought:“I need to move cd 2 into the safe...
I will move cd 2 into it.”
Action:move cd 2 to safe 1
(Hallucination)Thought:“I have placed the first cd (cd 3) into
the safe... Now I need to find the second cd (cd
1)...I will go to desk 1 to retrieve the second
cd.”
Action:go to desk 1
Execution Action: move cd 2 to safe 1→move cd
2 to safe 1...
Obs:Repetition loop detected.Action: take cd 1 from desk 1→go to
safe 1→move cd 1 to safe 1
Result Failed: Text-Action Disconnect
(Understands the instruction but fails the exe-
cution)Success: Task Solved
(Implicitly adheres to the sequential retrieval
pattern)
17

Prompt Template for Automated Feature Annotation
We are analyzing features learned by a sparse dictionary trained on the hidden states of an LLM agent solving interactive
tasks.
Environment:ALFWorld is a text-based household environment where an agent must complete tasks like finding
objects, cleaning them, heating/cooling them, and placing them in target locations. The agent navigates rooms, opens
containers, picks up objects, and uses appliances through text commands.
Each "feature" is a direction in the model’s representation space that activates on specific behavioral patterns. Your
job is to identify what behavioral pattern causes Feature [ID] to activate, and whether it is associated with effective or
ineffective agent behavior.
## Top Activating Examples (sorted by activation strength)
### Example 1 (activation=[val], reward=[val])
Observation: [text]
Action: [text]
[... additional examples ...]
## Non-Activating Examples (Feature [ID] does NOT activate on these)
### Example 1 (reward=[val])
Observation: [text]
Action: [text]
[... additional examples ...]
## Outcome Statistics
- This feature activates in [X]% of all steps
- Activation rate on successful trajectories: [X]%
- Activation rate on failed trajectories: [X]%
- Mean activation strength on successful trajectories: [val]
- Mean activation strength on failed trajectories: [val]
## Action Type Distribution (when this feature is active)
The following shows ALL action types in this environment, sorted by how much more likely they are to co-occur with
this feature compared to the overall action distribution (ratio > 1 means over-represented, < 1 means under-represented):
- [action type]: [ratio]x (P(active|action)=[X]%, mean_activation=[val])
[... additional action types ...]
## Instructions Based on ALL the evidence above (examples, outcome statistics, and action distribution), provide a
comprehensive interpretation of this feature:
1. Identify the common behavioral pattern across the activating examples that distinguishes them from the non-activating
examples.
2. Provide a short label (3-8 words in English) for this feature. The label should reflect both WHAT the behavior is and
WHETHER it contributes to task success or failure.
3. Provide a one-paragraph explanation of what this feature detects, integrating evidence from the examples, outcome
statistics, and action type distribution.
4. Classify the feature’s polarity: "positive" if it is associated with effective/successful behavior, "negative" if associated
with ineffective/failing behavior, or "neutral" if no clear association.
5. Rate your confidence (high/medium/low) in this interpretation.
Respond in JSON format:
{"label": "...", "explanation": "...", "polarity": "positive | negative | neutral", "confidence": "high | medium | low",
"key_evidence": ["...", "..."]}
Figure 8: Prompt template for automated interpretation of geometric basis directions.
18

Inter Intra0.840.860.880.900.920.940.96Normalized Entropy
Figure 9: Normalized entropy for inter-trajectory
and intra-trajectory steering vectors across the
ALFWorld benchmark.
Inter Intra0.3500.3750.4000.4250.4500.4750.5000.525T op-3 Concentration
Figure 10: Top-3 concentration for inter-trajectory
and intra-trajectory steering vectors across the
ALFWorld benchmark.
19

Example of Insights Baseline
Task Instruction:
“i am looking for wild caught, ready to eat sardines in a tomato sauce, and price lower than 50.00
dollars”
Insights:
•Ensure the product’s title and description explicitly mention the requested item type, size,
color, and price range to avoid misselection.
•Always verify that the product’s price is clearly stated and falls within the user’s specified
budget range to avoid overpayment.
•Prioritize products with clear and detailed descriptions that explicitly confirm the product
meets all requested specifications, including intended use, size, and color.
•Ensure the product’s size is explicitly mentioned in both the title and description and matches
the exact requested dimensions to avoid purchasing an incorrect product.
•Always confirm that the product’s title and description explicitly mention the requested use
case, color, and price to ensure alignment with the user’s requirements.
•Verify that the product’s key features explicitly include the requested functionality (e.g.,
wireless, battery inclusion) in both the title and description.
•Prioritize products where the requested attributes (such as type, size, and price) are explicitly
mentioned in both the title and description to minimize ambiguity.
•Always check the product’s price in both the title and description before proceeding to
purchase to avoid unexpected costs.
. . .(total 20 insights provided in context.)
Figure 11: Example of Insights Baseline for webshop
20

Example of Workflows Baseline
Task Goal:
put a clean soapbar in countertop.
Workflows:
• find_and_take_object
When you need to find and take an object, search in likely locations.
–go to {likely_location}
–[if container is closed] open {container}
–take {object} from {location}
• open_and_search_container
When you need to open a container and search inside.
–go to {container}
–open {container}
–[check contents]
• search_multiple_locations
When you need to search multiple locations for an object.
–go to {location 1}
–go to {location 2}
–go to {location 3}
–[repeat as needed]
• clean_and_place_object
When you need to clean an object and place it in a specific location.
–go to {cleaning_location}
–clean {object} with {cleaning_location}
–go to {target_location}
–move {object} to {target_receptacle}
. . .
Figure 12: Example of Workflows Baseline for AlfWorld
21