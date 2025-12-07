# VideoMem: Enhancing Ultra-Long Video Understanding via Adaptive Memory Management

**Authors**: Hongbo Jin, Qingyuan Wang, Wenhao Zhang, Yang Liu, Sijie Cheng

**Published**: 2025-12-04 07:42:13

**PDF URL**: [https://arxiv.org/pdf/2512.04540v1](https://arxiv.org/pdf/2512.04540v1)

## Abstract
Ultra long video understanding remains an open challenge, as existing vision language models (VLMs) falter on such content due to limited context length and inefficient long term memory retention. To address this, recent works have attempted to construct external knowledge bases and corresponding retrieval agumented generation (RAG) systems, yet these incur enormous storage and computational overhead. In this paper, we propose VideoMem, a novel framework that pioneers models long video understanding as a sequential generation task via adaptive memory management. Specifically, VideoMem dynamically updates a global memory buffer, which adaptively retains critical information while discarding redundant content across the video timeline. To efficiently train VLMs for such long-term tasks, VideoMem integrates the Progressive Grouped Relative Policy Optimization (PRPO) algorithm, equipped with two core modules: Progressive State Propagation (PSP) adaptively retains valid current states, propagates them to the next rollout step, and gradually narrows the model exploration space. Temporal Cascading Reward (TCR) further alleviates reward sparsity, improving sample utilization and accelerating convergence. Extensive experiments demonstrate that VideoMem significantly outperforms existing open-source models across diverse benchmarks for ultra-long video understanding tasks.

## Full Text


<!-- PDF content starts -->

VideoMem: Enhancing Ultra-Long Video Understanding
via Adaptive Memory Management
Hongbo Jin1Qingyuan Wang1Wenhao Zhang1Yang Liu2Sijie Cheng2*
1School of Electronic and Computer Engineering, Peking University
2Department of Computer Science and Technology, Tsinghua University
Abstract
Ultra-long video understanding remains an open challenge,
as existing vision language models (VLMs) falter on such
content due to limited context length and inefficient long-
term memory retention. To address this, recent works have
attempted to construct external knowledge bases and cor-
responding retrieval agumented generation (RAG) systems,
yet these incur enormous storage and computational over-
head. In this paper, we proposeVideoMem, a novel frame-
work that pioneers models long video understanding as a
sequential generation task via adaptive memory manage-
ment. Specifically, VideoMem dynamically updates a global
memory buffer, which adaptively retains critical informa-
tion while discarding redundant content across the video
timeline. To efficiently train VLMs for such long-term tasks,
VideoMem integrates the Progressive Grouped Relative Pol-
icy Optimization (PRPO) algorithm, equipped with two
core modules: Progressive State Propagation (PSP) adap-
tively retains valid current states, propagates them to the
next rollout step, and gradually narrows the model’s ex-
ploration space. Temporal Cascading Reward (TCR) fur-
ther alleviates reward sparsity, improving sample utiliza-
tion and accelerating convergence. Extensive experiments
demonstrate that VideoMem significantly outperforms ex-
isting open-source models across diverse benchmarks for
ultra-long video understanding tasks.
1. Introduction
Nowadays, Vision-Language Models (VLMs; [1, 13, 14,
18]) have attained remarkable advancements across a wide
range of image-centric downstream tasks [2, 36]. However,
processing video content, particularly ultra-long videos,
remains a prominent challenge for their core architec-
tures [9, 37]. This phenomenon stems from two fundamen-
tal bottlenecks: (i)limited context length[4, 20], restricted
*Corresponding Authorby the quadratic computational cost of attention mecha-
nisms; (ii)inefficient memory modeling[11, 31, 38],
which results in catastrophic forgetting of early video con-
tent. These constraints impede the development of critical
applications such as long-form video summarization and
question-answering over extended timelines, both of which
demand comprehensive and holistic video comprehension.
Existing long-video understanding approaches can be
broadly categorized into three paradigms. Compression-
based methods reduce input dimensionality through frame
sampling or feature compression [12, 15, 30, 35, 41], yet in-
evitably discard fine-grained temporal details. Static mem-
ory mechanisms employ pre-defined rules to manage long-
term memory [11, 26, 31, 38, 39, 45], but lack adaptabil-
ity to accommodate diverse video content and task-specfic
requirements. RAG-based methods [23, 24, 34] construct
external knowledge bases to store full video information
and design corresponding retrieval-augmented generation
(RAG) systems, which however incur substantial storage
and computational overhead.
To address these challenges, we proposeVideoMem, a
framework for ultra-long video understanding via adaptive
memory management. This core component dynamically
retains critical video details to mitigate forgetting while dis-
carding redundant content to control resource consumption.
In Figure 1,VideoMempioneers framing long-video un-
derstanding as a sequential generation task conditioned on
the query. Specifically, the long video is first split into
segments and are then processed iteratively: at each step,
the model dynamically updates a global memory buffer us-
ing the current segment, while reserving key temporal de-
tails from prior segments. After processing the entire long-
video, the model synthesizes all accumulated memories to
generate the final answer.
To train this framework, reinforcement learning (RL) is
a natural fit for such sequential tasks. However, directly ap-
plying prior methods, like GRPO [29], to ultra-long videos
processing is hindered by sparse and delayed reward sig-
nals, as well as prohibitive training computational costs.arXiv:2512.04540v1  [cs.CV]  4 Dec 2025

Figure 1. VideoMem demonstrates amazing ability to capture key information and perform long-term memory.
We thus propose Progressive Grouped Relative Policy Op-
timization (PRPO), an algorithm that enable efficient and
effective RL training for this setting. In detail, PRPO re-
duces training overhead by progressively propagating valid
states and narrowing the exploration space, namely progres-
sive state propagation (PSP). Furthermore, it incorporates
a temporal cascading rewards (TCR) mechanism to pro-
vides denser, stage-aware feedback, effectively mitigating
the sparse reward problem inherent to long-term tasks.
To extensively validate our proposed framework, we
conduct experiments on a comprehensive suite of stan-
dard and challenging long-video benchmarks, includ-
ing VideoMME [9], LongVideoBench [40], MLVU [49],
LVBench [37], LongTimeScope [51]. On one hand, lever-
aging adaptive memory management,VideoMemachieves
state-of-the-art (SOTA) performance among 17 compara-
ble open-source models across all long-video benchmark.
On the other hand, PRPO excels in training efficiency: it
outperforms GRPO by 3.1× in computational efficiency,
cuts convergence steps by 30% (i.e., 0.7× the original), and
yields notably improved training stability.
2. Related Work
2.1. Long Video Understanding
Long video understanding poses challenges that stem from
highly redundant visual information and inefficient tem-
poral dependency modeling. Expanding context length
is an intuitive solution, which directly enlarges model’s
temporal receptive field. Methods such as Gemini[33],
Kangaroo[20], and LongVILA [4] achieve this via efficient
positional encoding or sliding-window. Token compressionapproaches such as VideoAgent [7], LongVU [30], LLaV A-
onevision [13], PLLaV A [42] and VideoXL-2 [27] attempt
to reduce the sequence length via pooling mechanism or
similarity computation, while tend to lose fine-grained de-
tails. Recent studies focus on building RAG systems for
long videos. Ego-R1 [34] constructs a knowledge base in a
multi-level hierarchy, and trains VLMs to retrieve from it.
M3-Agent [23] trains two models separately: one learns to
store while the other learns to retrieve. Different from the
above methods, we enhance long video understanding via
adaptive memory management without external databases
or handcrafted selection rules.
2.2. Reinforcement Learning
RL has become a cornerstone for LLMs to align with hu-
man preferences and enhance reasoning capabilities [25].
PPO [28] optimizes policies through actor-critic frame-
works, balancing exploration and exploitation to improve
generation quality. Recent GRPO [29] further reduces com-
putation overhead by eliminating critic model and estimat-
ing advantages relative to grouped samples. In video do-
main, Video-R1 [8] explores the R1 paradigm [10] to in-
centivize video reasoning. LongVILA-R1 [5] introduces
a full-stack framework that scales up reasoning in long
videos. TinyLLaV A-Video-R1 [47] explored the reasoning
potential of small-scale VLMs. However, standard RL ap-
proaches struggle with exponential exploration space and
reward sparsity in ultra-long video sequences. Our work
incorporates temporal cascading rewards and progressive
state propagation specifically designed for such challenges.

Figure 2. An overview of VideoMem framework. The video are
divided into multiple clips and fed into the model iteratively. The
question, video clips are concatenated with the memory of each
turn’s model output as input for the next turn.
3. VideoMem
In this section, we elaborate on our proposed framework
VideoMem. First, we formulate ultra-long video under-
standing task as a sequential generation task and outline
adaptive memory management, as shown in Figure 2. Sec-
ond, we design an automated pipeline to construct chain-of-
memory data and perform cold start training for enhanced
stability, as illustrated in Figure 3. Finally, we propose
a RL-based training algorithm, Progressive Grouped Rel-
ative Policy Optimization (PRPO), which comprises two
core components: temporal cascading reward (TCR) and
progressive state propagation (PSP).
3.1. Adaptive Memory Management
Ultra-long video understanding grapples with two inher-
ently conflicting bottlenecks: redundant temporal informa-
tion in ultra-long content demands compression to save
computation yet risks losing critical details, while ineffi-
cient long-term memory modeling requires continuously re-
taining key information for holistic comprehension yet ex-
acerbates redundancy and computational burden. To ad-
dress these issues, we frame ultra-long video understand-
ing as a query-conditioned sequential generation task aug-
mented with adaptive memory management. This method
dynamically retains key information, discards redundancy,
and accumulates temporal context across video segments,
enabling the model to holistically comprehend ultra-long
videos without catastrophic forgetting or prohibitive costs.
In Figure 2, we first split a given ultra-long video intoAlgorithm 1Progressive Grouped Relative Policy Opti-
mization (PRPO)
Require:Policyπ θ, video segmentsV={V 1, . . . , V T},
questionQ, ground-truth answer Ans, Group countG, to-
ken budgetL max, coefficientsα,β
Ensure:Updated policyπ θ
1:Initialize memoryM←""
2:Initialize trajectory bufferB ← ∅
3:foreach example in datasetdo
4:Q, V,Ans←example
5:fort= 1toTdo
6:{O i}G
i=1←Rollout(π θ, Q, V t, M){GenerateG
trajectories}
7:{ˆa i}G
i=1 ←ExtractAnswer(Q,{O i}G
i=1)
{Provisional answer from<answer>}
8:ConsReward i←I[ˆa i=Ans]{1if matches Ans,
0otherwise}
9:FmtReward i←I[CheckFormat(O i)]{1if
<memory>···</memory> <answer>
···</answer>,0else}
10:MemPenaltyi ←
max(0,∥ExtractMemory(O i)∥tokens −L max)
11:R i←α·ConsReward i+FmtReward i−β·
MemPenaltyi
12:{A i}G
i=1←ADV({R i}G
i=1){Group-relative ad-
vantage}
13:B ← B ∪ {(O i, Ai)}G
i=1{Reserved for policy up-
dates}
14:ift < Tthen
15:M←ExtractBestMemory({O i}G
i=1,{Ai}G
i=1)
{Samplem j∼pj∝Arel
j}
16:end if
17:end for
18:Updateπ θvia GRPO-style objective usingB
19:end for
a sequence of equally long clips{V 1, V2, ..., V T}. We then
iteratively feed these clips to the VLM one on at a time,
processing clipV tat stept. At each stept, the VLM takes
three types of inputs: the current video clipV t, the user’s
queryQ, and a global memory bufferM t−1. Given these
inputs and the prompt detailed in Appendix??, the VLM
retains critical temporal details (e.g., recurring symbols,
plot transitions) and prunes redundant content (e.g., static
backgrounds) to form the updated global memory buffer
Mt. This adaptive memory management balances current
information with prior context, resolving the “forgetting vs.
redundancy” dilemma. Finally, the global memory buffer
MTadaptively accumulates key information from all clips
{V1, ..., V T}, enhancing the accuracy of the final answer.
This sequential generation task can be naturally formu-

Figure 3. Chain-of-memory data construction pipeline. By combining multiple models, multi-stage generation, and human supervision,
we ensure high-quality of video chain-of-memory pairs.
lated as a reinforcement learning (RL) problem, where the
policyπ θis optimized to generate high-quality trajectories.
However, challenges arise in two key aspects: first, the
ultra-long video setting exacerbates inherent reward spar-
sity and demands improved training efficiency; second, the
model requires robust foundational memory capabilities to
avoid unproductive exploration during RL training. There-
fore, we detail two tailored training components in the fol-
lowing subsections: a cold-start stage and the Progressive
Grouped Relative Policy Optimization algorithm.
3.2. Cold Start Stage
Following Deepseek-R1 [29], we introduce a cold start
stage prior to RL to equip the base model with fundamen-
tal capabilities for compressing ultra-long videos and re-
taining their key information. By pre-aligning the model
with basic memory management heuristics, it further ac-
celerates the RL process, enabling more efficient discovery
of high-performance policies—thereby stabilizing and en-
hancing the subsequent training pipeline.
To automatically construct our chain-of-memory data,
we first sample video–question–answer pairs from Video-
Marathon [17] dataset. VideoMarathon is a comprehensive
open-source long video training dataset, covering a wide
range of video distributions with 1,573k multiple-choice
questions and 1,726k open-ended questions. Each video is
split into multiple equal long clips, which are sequentially
fed intoQwen3-VL-32B-Instructto generate seman-
tic summaries. These summaries capture core narrative or
factual content according to the context of specific video
clips. Then, the clip-wise summaries are fed toQwen3-8B
[43] iteratively to filter out irrelevant information and gen-
erate coherent memories with manual checking. This yields
a high-quality chain-of-memory dataset. Prompt templatesare illurated in Appendix??.
Finally, we fine-tune the base model on these high-
quality chain-of-memory instances using standard super-
vised instruction tuning to initialize its memory manage-
ment ability. This cold start phase equips the model with
the fundamental capability to construct coherent and con-
cise memory representations.
3.3. Progressive Grouped Relative Policy Optimiza-
tion
While Grouped Relative Policy Optimization (GRPO; [29])
lays a solid foundation for policy optimization without step-
wise reward modeling, it encounters notable challenges
when adapted to long-term tasks. The primary issues are
the exponential expansion of the exploration space and ex-
acerbated sparse reward signals inherent to extended tem-
poral sequences. To mitigate these limitations, as illustrated
in Figure 4, we propose Progressive Grouped Relative Pol-
icy Optimization (PRPO). This algorithm is integrated with
two key innovations: temporal cascading reward (TCR) and
progressive state propagation (PSP).
3.3.1. Temporal Cascading Reward
A major challenge in long-term RL is the sparsity and de-
lay of reward signals: A reward for correctly answering a
question grounded in early video content may only be ob-
served after processing the entire sequence, rendering credit
assignment highly non-trival. To address this, the temporal
cascading reward (TCR) mechanism decomposes sparse fi-
nal rewards into denser intermediate rewards at each step,
delivering more frequent feedback to guide memory man-
agement. Specifically, at each stept, the model generates a
provisional answerˆa tconditioned on the current video clip
Vt, memory bufferM t−1and queryQ. The immediate re-

Figure 4. rollout and policy optimization pipeline of PRPO
ward ˆRtfor each trajectory is then computed as the sum of
three components:
ˆRt=α·R cons+R format−β·MemPenalty (1)
•R cons=I[ˆa t=a]: For multiple-choice QA tasks, consis-
tency reward is 1 ifˆa texactly matches the ground-truth
answeraand 0 otherwise.
•R format : A binary reward, where the reward is equal
to 1 if the output conforms to the required for-
mat⟨memory⟩···⟨memory⟩⟨answer⟩···⟨answer⟩,
0 otherwise.
• MemPenalty= max(0,∥M t∥ −L max): Penalizes the
length of memory buffer exceeding thresholdL max to
prevent bloat while preserving efficiency.
Hyperparametersαandβcontrol the relative importance
of consistency and memory efficiency, respectively. These
rewards are used to compute group-relative advantages ˆArel
t
following the same procedure as GRPO for policy update,
detailed in Appendix??. This TCR mechanism can allevi-
ate reward sparsity, improve sample utilization, and accel-
erate convergence.
3.3.2. Progressive State Propagation
Moreover, another key challenge in long-term RL is the
exponential expansion of the exploration space as the se-
quence accumulates. Directly applying GRPO would incur
prohibitive computational costs, especially for ultra-long
video sequences. To mitigate this cost barrier, the Progres-
sive State Propagation (PSP) mechanism acts as a memory-
aware trajectory selection strategy that adaptively narrows
the exploration space while retaining high-reward paths.
LetM t={m 1, m2, . . . , m G}denotes the set of global
memory buffers extracted from theGtrajectories at stept. Instead of retaining allGstates, PRPO selects a sin-
gle memory bufferM∗
tto propagate to the next turn gen-
eration, conditioned on the group-relative advantage ˆArel
iof
each trajectory:
M∗
t=Sample(M t, pj∝ˆArel
j)(2)
The sampling probability is softened with a temperatureτ t
that decreases over time:
pj=exp( ˆArel
j/τt)
PG
k=1exp( ˆArel
k/τt)(3)
whereτ t=τ0·γt(γ <1) ensures high exploration early
and focused exploitation later. The selected stateM∗
tis con-
catenated with the queryQand the upcoming clipV t+1to
form the input for the next turn rollout. This progressive
narrowing yields two key benefits: (i) Computational ef-
ficiency, as prefilling complexity drops fromO(N·G)to
O(N). Since only one memory path is cached per step, ex-
plained in detail in Section 4.8. (ii) Training stability, where
early noise is filtered out and only trajectories with strong
provisional consistency and format compliance propagate,
thereby preventing error accumulation.
4. Experiment
4.1. Implementation Details
We choose Qwen3-VL-8B as our base model. During train-
ing, we divide each video into four equal segments, and
extract 32 frames from each segment with max pixels not
exceeding128∗32∗32. The number of rolloutsGis set to
8. For temporal cascading reward, the hyperparameters are
set toα= 1.0,β= 0.005andL max= 1024. We set global

Table 1. Evaluate results on various long video benchmarks, including VideoMME, LongVideoBench, MLVU, LVBench, LongTimeScope.
All benchmarks adopt percentage accuracy as the evaluation metric. *Scores from our evaluation using 64 uniform frames, to ensure fair
comparisons with same context length. All other results are from their original paper, with at least 64 frames of input.
Models SizeVideoMMELongVideoBench MLVU LVBench LongTimeScopeLong Overall
Duration 30∼60 min 1∼60 min 23sec∼60 min 3∼120 min 30∼90 min 300min+
Proprietary Models
GPT-4V - 56.9 60.7 59.1 49.2 - -
GPT-4o - 65.3 71.9 66.7 64.6 64.4 -
Gemini-2.5-Pro - - 87.0 - 81.2 69.2 -
Seed1.5-VL 8B - 77.9 74.4 82.1 64.6 -
Eagle2.5 8B - 72.4 66.4 77.6 - -
Open Source Models
Video-LLaV A [16] 7B 38.1 40.4 39.1 47.3 - 17.6
ShareGPT4Video [3] 8B 37.9 43.6 39.7 46.4 - -
LLaV A-Next-Video [19] 7B - 46.5 50.5 39.3 - -
VideoLLaMA2 [6] 7B 43.8 46.6 - 48.5 - -
LongV A [46] 7B 47.6 54.3 - 56.3 36.2 -
LLaV A-Onevision [13] 7B 46.7 58.2 56.4 64.7 26.9 30.2
LLaV A-video [48] 7B 50.6 62.6 58.2 70.8 41.5 34.0
Qwen2.5-VL [2] 7B 51.6 65.1 54.7 70.2 45.3 40.7
VideoNSA [32] 7B - - 60.0 51.8 - 44.4
Apollo [50] 7B - 61.3 58.5 70.9 - -
NVILA [22] 8B 54.8 64.2 57.7 70.1 44.0 -
Flow4Agent[21] 7B 54.2 64.7 60.4 71.4 - -
VideoLLaMA3 [44] 7B 54.1 66.2 59.8 73.0 45.3 39.1
Qwen2.5-VL 72B 61.2 73.3 - 74.6 47.3 -
InternVL3.5 [36] 8B - 66.0 62.1 70.2 - -
Video-XL2 [27] 8B - 66.6 61.0 74.8 48.4 -
Qwen3-VL* 8B 58.6 67.9 59.1 71.6 50.6 36.2
VideoMem* 8B 64.2 73.6 63.3 77.4 58.5 45.1
batch size to 16 and use the AdamW optimizer. All ex-
periments are conducted on 16x PPU-ZW810E with 96GB
memory. The complete training process requires 4800 PPU
hours or 3200 A100 GPU hours. During evaluation, the in-
put frame number is all set to 64, and the number of video
segments in VideoMem is 4.
As for data, we sample 26k records from VideoMarathon
dataset, and label them with chain-of-memory as the cold
start training set. During RL training stage, we sample 147k
multi choice QA pairs from VideoMarathon and 49k from
LLaV A-Video-178K [48] dataset as training dataset, em-
phasis on long videos while also considering short videos.
We adopt this data ratio, following the best results described
in VideoMarathon [17].
4.2. Main Results
Table 1 presents our main results across all benchmarks.
VideoMem significantly outperforms various strong base-
lines, achieving new state-of-the-art (SOTA) performance
among comparable open-source models on diverse long-
video benchmarks. Notably, our model delivers the largestperformance gains on the most challenging long-form video
benchmarks–LVBench, VideoMME and LongTimeScope–
outperforming the base model Qwen3-VL-8B by 7.9%,
5.7% and 8.9%, respectively. These results are particularly
revealing. While the base model exhibits strong static visual
comprehension capabilities, it is not inherently optimized
for the long-term reasoning required by ultra-long videos. It
processes extended context as a flat sequence, struggling to
effectively retain salient information over time. In contrast,
VideoMem is explicitly trained via PRPO to act as a state-
ful agent: it learns an adaptive memory management policy
that decides what critical information from the current clip
Vtto integrate into memoryM tand what to discard. The
substantial performance gains provide direct evidence that
our adaptive memory mechanism successfully overcomes
this core limitation of baseline models.
These findings offer a key insight: rather than relying
solely on expanding context windows, which faces com-
putational scaling challenges, a more efficient and effec-
tive paradigm is to reframe long-video understanding as
a sequential generation problem. The fundamental reason

for VideoMem’s success lies in this shift—from training
a model for static, one-shot comprehension to training an
agent with a dynamic policy for information compression
and propagation. Our work demonstrates that VLMs can
be effectively optimized for long-term tasks using tailored
RL strategies like PRPO, which successfully manage the
sparse-reward and large-exploration-space challenges in-
herent to this formulation.
4.3. Core Components Analysis
We ablate the core components of VideoMem in Table 2.
Removing progressive state propagation (PSP) drops per-
formance by 2.1%, confirming its role in stabilizing long-
term training. Removing temporal cascading reward (TCR)
causes even larger degradation (4.2-4.9)%, leading to se-
vere reward sparsity problem. And removing the memory
penalty will also lead to a certain performance degradation.
Table 2. Ablation of VideoMem core components.
Method PSP TCR MP MME LTS
VideoMem ✓ ✓ ✓ 73.6 45.1
w/o PSP. ✓ ✓ 71.5 43.0
w/o TCR ✓ ✓ 69.4 40.3
w/o MP ✓ ✓ 72.0 43.9
SFT 67.5 38.2
4.4. Reward Mechanism Analysis
Table 3 further analyzes the impact of temporal cascad-
ing reward (TCR) compared to the single terminal reward
(TR) used in GRPO. We observe significantly denser and
more stable reward feedback when applying TCR. PRPO’s
temporal reward decomposition increases reward frequency
by 4.0× (number of segments was set to 4), resulting in
smoother policy gradients and faster convergence. We also
notice that temporal cascading rewards stabilize training
and prevent premature policy collapse. Empirically, models
trained with TCR converge in about 70% of the steps re-
quired by TR, highlighting the efficiency of our dense feed-
back design.
Table 3. Ablation of temporal cascading reward mechanism.
Reward Type Reward Density Convergence Steps LVBench
TR 1.0x 1.0x 54.6
TCR 4.0x 0.7x 58.5
4.5. Per-Segment Accuracy Analysis
To verify the performance improvement of VideoMem
truly stems from long-term memory, rather than a “generalTable 4. Seg1-N represents the accuracy of the model’s prediction
after N turns of input.
Method Seg1-1 Seg1-2 Seg1-3 Seg1-4
Qwen3-VL 50.3 48.5 46.2 44.8
VideoMem 51.2 55.4 57.1 58.5
guess” based on early segments, we conducted the follow-
ing experiments on LVBench.
We utilize the Temporal Cascaded Reward (TCR) mech-
anism to generate a provisional answerˆa tat each stept,
and record the corresponding ”provisional accuracy”. As
shown in Table 4, the accuracy of the baseline model de-
creased from 50.3% to 44.8% with the addition of segments.
This demonstrates catastrophic forgetting, where the model
fails to integrate new information to correct early error.
Conversely, VideoMem’s accuracy showed non-trivially in-
creases with each new segment added, steadily increasing
from 51.2% to 58.5%. This strongly demonstrates that
VideoMem continuously and proactively utilizes new seg-
ments to refine and improve its understanding, confirming
the substantial contribution of later fragments.
4.6. Cold Start Effiency
To verify the actual contribution of the cold start phase, we
conducted corresponding ablation experiments in Table 5.
We conducted a comparison on three representative long
video benchmarks: VideoMME (long subset), LVBench,
and LongVideoBench. All other training hyperparameters
(batch size, number of frames, token budget, etc.) are con-
sistent with the main experiment. Results show that cold
start is a necessary training phase that can accelerate model
convergence and avoid invalid rollouts. Detailed diagrams
of the training process are shown in the Appendix.
Table 5. Cold Start Efficiency and Key Baseline Comparison. We
compared five different experimental setups: (1) Qwen3-VL (un-
trained); (2) SFT (fine-tuned on general video instruction data);
(3) PRPO-only (trained using only PRPO); (4) ColdStart-only
(SFT training only on our chain-of-memory data); (5) VideoMem
(trained with both cold start and PRPO).
Method VideoMME LVBench LVB
Qwen3-VL 58.6 50.6 59.1
SFT 58.1 51.7 58.5
PRPO-only 59.8 53.4 59.5
ColdStart-only 61.9 54.4 60.7
VideoMem64.2 58.5 63.3

Figure 5. Inference Scalability
4.7. Inference Scalability
While the number of video segments during training is fixed
toT= 4, to ensure stable RL convergence and balanced re-
ward density, the learned memory policy generalizes seam-
lessly to arbitrary segment counts during inference.
To verify this scalability, we conducted analysis on
LongTimeScope, LongVideoBench and LVBench, shown
in Figure 5. Fix the number of segments to 4 during training
while evaluate its performance during inference with differ-
ent numbers of segmentsT= 1,2,3,4,5,6. Simultane-
ously, we compared the performance of the baseline model
(Qwen3-VL) under the same settings.
The performance of baseline (Qwen3-VL) quickly
reaches a bottleneck aroundT= 3, and shows a rapid de-
cline, indicating that it cannot effectively utilize long term
memory context. In contrast, VideoMem exhibits clear pos-
itive correlation scalability: as the number of segments in-
creases, VideoMem is able to integrate more contextual in-
formation, and its accuracy steadily improves accordingly.
4.8. Computational Efficiency
VideoMem achieves a 3.1x speedup in training compared
to vanilla GRPO while maintaining superior performance,
shown in Table 6. The key lies inPSP: at each turn, a mem-
ory state is selected based on relative strengths to serve as
the starting state for next turn, avoiding the expensive ”pre-
fill” operation for allGpaths. This reduces the number of
pre-filling calls fromO(N∗G)toO(N)(only once per
segment). While decoding (i.e., token-level generation for
each candidate trajectory) still requiresO(N∗G)times,
its computational overhead is far less than pre-filling. The
evaluation results did not decrease but even improved, mak-
ing our approach practical for large-scale deployment.
Table 6. Computational Efficiency
Method Train Speed↑VideoMME LVBench
Qwen3-VL - 67.9 50.6
GRPO 1.0x 72.0 56.7
PRPO 3.1x 73.6 58.55. Discussion
Generalizability of VideoMem to Open-ended Tasks.A
key avenue for future work is to generalize our framework
to open-ended QA tasks. The core sequential processing
and adaptive memory management of VideoMem are task-
agnostic. We hypothesize that by replacing the binary re-
wardR cons within TCR with a score from an LLM-as-a-
judge, PRPO can effectively train VLMs to generate high-
quality, free-form answers, confirming that VideoMem
learns a deep, compressible understanding of video content.
Generalizability of PRPO to Other Long-term Tasks.
PRPO algorithm is a general-purpose optimization strat-
egy for sequential decision-making under sparse rewards.
Its core components, PSP and TCR, address fundamental
challenges in long-term tasks, independent of input modal-
ity. We believe PRPO holds significant promise for other
domains, such as long-document QA, where text chunks
replace video clipsV t. Furthermore, PRPO could be ap-
plied to multi-step agents, where PSP prunes the exponen-
tial exploration space of reasoning paths, and TCR provides
denser feedback on intermediate sub-goals, effectively ex-
tending its application far beyond video understanding.
6. Conclusion
In this paper, we introduced VideoMem, a novel framework
designed to address the significant challenges of ultra-long
video understanding. We pioneeringly model this problem
as a sequential generation task, and train VLMs to progres-
sively compress video information and maintain effective
long-term memory. We also propose Progressive Grouped
Relative Policy Optimization (PRPO) algorithm, a novel re-
inforcement learning strategy tailored for this complex task.
PRPO enhances training efficiency and stability through
two key innovations: Progressive State Propagation (PSP),
which adaptively narrows the model’s exploration space to
reduce computational costs, and Temporal Cascading Re-
ward (TCR), which provides denser, intermediate feedback
to effectively alleviate the reward sparsity problem. Exper-
iments demonstrate that VideoMem achieves new state-of-
the-art performance among comparable open-source mod-
els on a wide range of challenging long-video benchmarks,
showing particularly strong gains on hour-long tasks. Re-

sults validate that our approach offers a promising solution
for VLMs that can genuinely comprehend and reason over
ultra-long video content. Hope that our approach can pro-
vide valuable insights
References
[1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine
Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Men-
sch, Katherine Millican, Malcolm Reynolds, et al. Flamingo:
a visual language model for few-shot learning.Advances
in neural information processing systems, 35:23716–23736,
2022. 1
[2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin
Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun
Tang, et al. Qwen2. 5-vl technical report.arXiv preprint
arXiv:2502.13923, 2025. 1, 6
[3] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang,
Yuhang Zang, Zehui Chen, Haodong Duan, Zhenyu Tang, Li
Yuan, et al. Sharegpt4video: Improving video understand-
ing and generation with better captions.Advances in Neural
Information Processing Systems, 37:19472–19495, 2024. 6
[4] Yukang Chen, Fuzhao Xue, Dacheng Li, Qinghao Hu,
Ligeng Zhu, Xiuyu Li, Yunhao Fang, Haotian Tang, Shang
Yang, Zhijian Liu, et al. Longvila: Scaling long-context vi-
sual language models for long videos. InThe Thirteenth In-
ternational Conference on Learning Representations. 1, 2
[5] Yukang Chen, Wei Huang, Baifeng Shi, Qinghao Hu, Han-
rong Ye, Ligeng Zhu, Zhijian Liu, Pavlo Molchanov, Jan
Kautz, Xiaojuan Qi, et al. Scaling rl to long videos.arXiv
preprint arXiv:2507.07966, 2025. 2
[6] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin
Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang
Luo, Deli Zhao, et al. Videollama 2: Advancing spatial-
temporal modeling and audio understanding in video-llms.
arXiv preprint arXiv:2406.07476, 2024. 6
[7] Yue Fan, Xiaojian Ma, Rujie Wu, Yuntao Du, Jiaqi Li, Zhi
Gao, and Qing Li. Videoagent: A memory-augmented mul-
timodal agent for video understanding. InEuropean Con-
ference on Computer Vision, pages 75–92. Springer, 2024.
2
[8] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo,
Yibing Wang, Tianshuo Peng, Junfei Wu, Xiaoying Zhang,
Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing
video reasoning in mllms.arXiv preprint arXiv:2503.21776,
2025. 2
[9] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai
Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang
Shen, Mengdan Zhang, et al. Video-mme: The first-ever
comprehensive evaluation benchmark of multi-modal llms in
video analysis. InProceedings of the Computer Vision and
Pattern Recognition Conference, pages 24108–24118, 2025.
1, 2
[10] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song,
Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi
Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning
capability in llms via reinforcement learning.arXiv preprint
arXiv:2501.12948, 2025. 2[11] Bo He, Hengduo Li, Young Kyun Jang, Menglin Jia, Xue-
fei Cao, Ashish Shah, Abhinav Shrivastava, and Ser-Nam
Lim. Ma-lmm: Memory-augmented large multimodal model
for long-term video understanding. InProceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 13504–13514, 2024. 1
[12] Peng Jin, Ryuichi Takanobu, Wancai Zhang, Xiaochun Cao,
and Li Yuan. Chat-univi: Unified visual representation em-
powers large language models with image and video un-
derstanding. InProceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 13700–
13710, 2024. 1
[13] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li,
Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Zi-
wei Liu, et al. Llava-onevision: Easy visual task transfer.
arXiv preprint arXiv:2408.03326, 2024. 1, 2, 6
[14] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi.
Blip-2: Bootstrapping language-image pre-training with
frozen image encoders and large language models. InIn-
ternational conference on machine learning, pages 19730–
19742. PMLR, 2023. 1
[15] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid:
An image is worth 2 tokens in large language models. In
European Conference on Computer Vision, pages 323–340.
Springer, 2024. 1
[16] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng
Jin, and Li Yuan. Video-llava: Learning united visual repre-
sentation by alignment before projection. InProceedings of
the 2024 Conference on Empirical Methods in Natural Lan-
guage Processing, pages 5971–5984, 2024. 6
[17] Jingyang Lin, Jialian Wu, Ximeng Sun, Ze Wang, Jiang Liu,
Yusheng Su, Xiaodong Yu, Hao Chen, Jiebo Luo, Zicheng
Liu, and Emad Barsoum. Unleashing hour-scale video train-
ing for long video-language understanding, 2025. 4, 6
[18] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee.
Visual instruction tuning.Advances in neural information
processing systems, 36:34892–34916, 2023. 1
[19] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan
Zhang, Sheng Shen, and Yong Jae Lee. Llavanext: Improved
reasoning, ocr, and world knowledge, 2024. 6
[20] Jiajun Liu, Yibing Wang, Hanghang Ma, Xiaoping Wu, Xi-
aoqi Ma, Xiaoming Wei, Jianbin Jiao, Enhua Wu, and Jie Hu.
Kangaroo: A powerful video-language model supporting
long-context video input.arXiv preprint arXiv:2408.15542,
2024. 1, 2
[21] Ruyang Liu, Shangkun Sun, Haoran Tang, Wei Gao, and Ge
Li. Flow4agent: Long-form video understanding via motion
prior from optical flow. InProceedings of the IEEE/CVF
International Conference on Computer Vision, pages 23817–
23827, 2025. 6
[22] Zhijian Liu, Ligeng Zhu, Baifeng Shi, Zhuoyang Zhang,
Yuming Lou, Shang Yang, Haocheng Xi, Shiyi Cao, Yuxian
Gu, Dacheng Li, et al. Nvila: Efficient frontier visual lan-
guage models. InProceedings of the Computer Vision and
Pattern Recognition Conference, pages 4122–4134, 2025. 6
[23] Lin Long, Yichen He, Wentao Ye, Yiyuan Pan, Yuan Lin,
Hang Li, Junbo Zhao, and Wei Li. Seeing, listening, remem-

bering, and reasoning: A multimodal agent with long-term
memory.arXiv preprint arXiv:2508.09736, 2025. 1, 2
[24] Yongdong Luo, Xiawu Zheng, Xiao Yang, Guilin Li,
Haojia Lin, Jinfa Huang, Jiayi Ji, Fei Chao, Jiebo Luo,
and Rongrong Ji. Video-rag: Visually-aligned retrieval-
augmented long video comprehension.arXiv preprint
arXiv:2411.13093, 2024. 1
[25] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Car-
roll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini
Agarwal, Katarina Slama, Alex Ray, et al. Training language
models to follow instructions with human feedback.Ad-
vances in neural information processing systems, 35:27730–
27744, 2022. 2
[26] Rui Qian, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Shuan-
grui Ding, Dahua Lin, and Jiaqi Wang. Streaming long video
understanding with large language models.Advances in Neu-
ral Information Processing Systems, 37:119336–119360,
2024. 1
[27] Minghao Qin, Xiangrui Liu, Zhengyang Liang, Yan Shu,
Huaying Yuan, Juenjie Zhou, Shitao Xiao, Bo Zhao, and
Zheng Liu. Video-xl-2: Towards very long-video under-
standing through task-aware kv sparsification.arXiv preprint
arXiv:2506.19225, 2025. 2, 6
[28] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Rad-
ford, and Oleg Klimov. Proximal policy optimization algo-
rithms.arXiv preprint arXiv:1707.06347, 2017. 2
[29] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao
Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li,
Yang Wu, et al. Deepseekmath: Pushing the limits of math-
ematical reasoning in open language models.arXiv preprint
arXiv:2402.03300, 2024. 1, 2, 4
[30] Xiaoqian Shen, Yunyang Xiong, Changsheng Zhao, Lemeng
Wu, Jun Chen, Chenchen Zhu, Zechun Liu, Fanyi Xiao, Bal-
akrishnan Varadarajan, Florian Bordes, et al. Longvu: Spa-
tiotemporal adaptive compression for long video-language
understanding. InForty-second International Conference on
Machine Learning. 1, 2
[31] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng
Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo,
Tian Ye, Yanting Zhang, et al. Moviechat: From dense token
to sparse memory for long video understanding. InProceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 18221–18232, 2024. 1
[32] Enxin Song, Wenhao Chai, Shusheng Yang, Ethan Armand,
Xiaojun Shan, Haiyang Xu, Jianwen Xie, and Zhuowen Tu.
Videonsa: Native sparse attention scales video understand-
ing.arXiv preprint arXiv:2510.02295, 2025. 6
[33] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-
Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk,
Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a
family of highly capable multimodal models.arXiv preprint
arXiv:2312.11805, 2023. 2
[34] Shulin Tian, Ruiqi Wang, Hongming Guo, Penghao Wu,
Yuhao Dong, Xiuying Wang, Jingkang Yang, Hao Zhang,
Hongyuan Zhu, and Ziwei Liu. Ego-r1: Chain-of-tool-
thought for ultra-long egocentric video reasoning.arXiv
preprint arXiv:2506.13654, 2025. 1, 2[35] Han Wang, Yuxiang Nie, Yongjie Ye, Yanjie Wang, Shuai
Li, Haiyang Yu, Jinghui Lu, and Can Huang. Dynamic-vlm:
Simple dynamic visual token compression for videollm. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision, pages 20812–20823, 2025. 1
[36] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long
Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong
Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source
multimodal models in versatility, reasoning, and efficiency.
arXiv preprint arXiv:2508.18265, 2025. 1, 6
[37] Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiao-
han Zhang, Ji Qi, Ming Ding, Xiaotao Gu, Shiyu Huang, Bin
Xu, et al. Lvbench: An extreme long video understanding
benchmark. InProceedings of the IEEE/CVF International
Conference on Computer Vision, pages 22958–22967, 2025.
1, 2
[38] Yuxuan Wang, Yiqi Song, Cihang Xie, Yang Liu, and Zi-
long Zheng. Videollamb: Long streaming video understand-
ing with recurrent memory bridges. InProceedings of the
IEEE/CVF International Conference on Computer Vision,
pages 24170–24181, 2025. 1
[39] Yun Wang, Long Zhang, Jingren Liu, Jiaqi Yan, Zhan-
jie Zhang, Jiahao Zheng, Xun Yang, Dapeng Wu, Xi-
angyu Chen, and Xuelong Li. Episodic memory represen-
tation for long-form video understanding.arXiv preprint
arXiv:2508.09486, 2025. 1
[40] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li.
Longvideobench: A benchmark for long-context interleaved
video-language understanding.Advances in Neural Informa-
tion Processing Systems, 37:28828–28857, 2024. 2
[41] Peiran Wu, Zhuorui Yu, Yunze Liu, Chi-Hao Wu, Enmin
Zhou, and Junxiao Shen. Marc: Memory-augmented rl to-
ken compression for efficient video understanding.arXiv
preprint arXiv:2510.07915, 2025. 1
[42] Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See Kiong Ng,
and Jiashi Feng. Pllava: Parameter-free llava extension from
images to videos for video dense captioning.arXiv preprint
arXiv:2404.16994, 2024. 2
[43] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen
Huang, Chenxu Lv, et al. Qwen3 technical report.arXiv
preprint arXiv:2505.09388, 2025. 4
[44] Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu,
Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang,
Hang Zhang, Xin Li, et al. Videollama 3: Frontier multi-
modal foundation models for image and video understand-
ing.arXiv preprint arXiv:2501.13106, 2025. 6
[45] Haoji Zhang, Yiqin Wang, Yansong Tang, Yong Liu, Jiashi
Feng, Jifeng Dai, and Xiaojie Jin. Flash-vstream: Memory-
based real-time understanding for long video streams.arXiv
preprint arXiv:2406.08085, 2024. 1
[46] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng,
Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan,
Chunyuan Li, and Ziwei Liu. Long context transfer from
language to vision.arXiv preprint arXiv:2406.16852, 2024.
6
[47] Xingjian Zhang, Siwei Wen, Wenjun Wu, and Lei Huang.

Tinyllava-video-r1: Towards smaller lmms for video reason-
ing, 2025. 2
[48] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Zi-
wei Liu, and Chunyuan Li. Llava-video: Video instruction
tuning with synthetic data, 2025. 6
[49] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao,
Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and
Zheng Liu. Mlvu: A comprehensive benchmark for multi-
task long video understanding.arXiv e-prints, pages arXiv–
2406, 2024. 2
[50] Orr Zohar, Xiaohan Wang, Yann Dubois, Nikhil Mehta,
Tong Xiao, Philippe Hansen-Estruch, Licheng Yu, Xiaofang
Wang, Felix Juefei-Xu, Ning Zhang, et al. Apollo: An explo-
ration of video understanding in large multimodal models. In
Proceedings of the Computer Vision and Pattern Recognition
Conference, pages 18891–18901, 2025. 6
[51] Orr Zohar, Xiaohan Wang, Rui Li, Andr ´es Marafioti, Miquel
Farr´e, Merve Noyan, Leandro von Werra, Serena Yeung-
Levy, and Thomas Wolf. Apollo2: Exploring the long-video
frontier of large multimodal models. 2025. 2