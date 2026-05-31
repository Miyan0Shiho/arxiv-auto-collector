# WorldMemArena: Evaluating Multimodal Agent Memory Through Action-World Interaction

**Authors**: Chengzhi Liu, Yuzhe Yang, Sophia Xiao Pu, Yepeng Liu, Lin Long, Yichen Guo, Nuo Chen, Zhaotian Weng, Elena Kochkina, Simerjot Kaur, Charese Smiley, Xiaomo Liu, James Zou, Sheng Liu, Yuheng Bu, Songyou Peng, Xin Eric Wang

**Published**: 2026-05-28 04:27:20

**PDF URL**: [https://arxiv.org/pdf/2605.29341v1](https://arxiv.org/pdf/2605.29341v1)

## Abstract
Multimodal large language models are increasingly deployed as long-horizon agents, where memory must do more than recall: it must track an evolving world, revise what has gone stale, and surface the right evidence at decision time. Existing benchmarks measure recall over static dialogue, collapse memory into a single end-of-task accuracy, and reduce visual observations to captions, leaving us unable to localize failures to writing, maintenance, retrieval, or use. The rise of agent harnesses that author their own memory sharpens this gap, since we have no principled way to compare hand-designed pipelines with self-managing alternatives. To close these gaps, we formulate multimodal agent memory as an Action-World Interaction Loop with an observable four-stage lifecycle, and instantiate it in WorldMemArena: 400 multi-session multimodal tasks spanning Lifelong Evolution (evolving personal and task states) and Agentic Execution (memory from real observations, actions, and feedback), annotated with gold memory points, updates, distractors, and evidence chains for stage-level diagnosis. This enables the first head-to-head comparison of long-context, manually designed (RAG and external memory systems), and harness-based memory agents. Results show that: (1) better memory writing and storage do not guarantee better performance; (2) multimodal memory still struggles to fully use visual evidence; (3) systems are unstable across domains and degrade on realistic agentic trajectories; and (4) harness memory is more flexible but remains costly and less reliable.

## Full Text


<!-- PDF content starts -->

WorldMemArena: Evaluating Multimodal Agent
Memory Through Action–World Interaction
Chengzhi Liu*1, Yuzhe Yang*1, Sophia Xiao Pu1, Yepeng Liu1, Lin Long5, Yichen Guo1, Nuo Chen6,
Zhaotian Weng1, Elena Kochkina2, Simerjot Kaur2, Charese Smiley2, Xiaomo Liu2, James Zou4,
Sheng Liu4, Yuheng Bu1, Songyou Peng3, Xin Eric Wang1
1
University of California, Santa Barbara2
J.P. Morgan Chase3
ETH Zurich
4
Stanford University5
Johns Hopkins University6
Carnegie Mellon University
Multimodal large language models are increasingly deployed as long-horizon agents, where memory must do
more than recall: it must track an evolving world, revise what has gone stale, and surface the right evidence
at decision time. Existing benchmarks measure recall over static dialogue, collapse memory into a single
end-of-taskaccuracy, andreducevisualobservationstocaptions, leavingusunabletolocalizefailurestowriting,
maintenance, retrieval, or use. The rise of agent harnesses that author their own memory sharpens this gap,
since we have no principled way to compare hand-designed pipelines with self-managing alternatives. To close
these gaps, we formulate multimodal agent memory as anAction–World Interaction Loopwith an observable
four-stage lifecycle, and instantiate it inWorldMemArena: 400 multi-session multimodal tasks spanning
Lifelong Evolution(evolving personal and task states) andAgentic Execution(memory from real observations,
actions, and feedback), annotated with gold memory points, updates, distractors, and evidence chains for
stage-level diagnosis. This enables the first head-to-head comparison of long-context, manually designed (RAG
and external memory systems), and harness-based memory agents. Results show that:(1)better memory
writing and storage do not guarantee better performance;(2)multimodal memory still struggles to fully use
visual evidence;(3)systems are unstable across domains and degrade on realistic agentic trajectories; and(4)
harness memory is more flexible but remains costly and less reliable.
/envel⌢peCorrespondence:{chengzhi,yuzheyang,ericxwang}@ucsb.edu
/gl⌢beProject Page
 Dataset /githubWorldMemArena
2025-05-042026-04-20............Lifelong EvolutionAgentic Execution 
Session-1Session-2Session-3Session-4
Explore & PlanWrite & Rebuttal
Analyze
Experiment
BasicReasoningMultimodalRobustness0.770.670.690.790.550.320.390.530.480.470.470.330.450.510.520.590.80.60.40.2External MemoryHarness-BasedBase modelRAG
3.Retrieve&Decide
Retrieval
Decision
Tradeoff
2.Update&Consolidate
Ingestion
Update
Consolidate
Discard
1.Observed&Encoder
Image
Text
VideoAudio
Semantic
SpatialTemporal
Episodic
MemoryLibrary
4.Act&Intervene
Feedback
Transition
Execute
NextState
Sourcesofstateandevents
WorldChange by ActionWorld
(A)(B)
(C)
Figure1.(a) WorldMemArena formulates multimodal agent memory as anAction-World Interaction Loop, where agents write
observations, update evolving memory, retrieve evidence for decisions, and act in the world with feedback. (b) It spans two
regimes,Agentic ExecutionandLifelong Evolution, covering real agent trajectories and evolving personal and task states across
sessions. (c) Evaluation covers different memory paradigms across basic, robustness, reasoning, and multimodal capabilities.arXiv:2605.29341v1  [cs.CV]  28 May 2026

WorldMemArena: Evaluating Multimodal Agent Memory Through Action–World Interaction
1. Introduction
MultimodallargelanguagemodelsOpenAI(2026b),QwenTeam(2026),Anthropic(2026a)areturningfrom
question answering systems into agents that act in dynamic environments over long horizons Steinberger
(2025), Anthropic (2026b). In this setting, memory is no longer simply a cache of past text, but a mechanism
for tracking task state, learning from actions, and supporting decisions through real-world interaction. A
capable long horizon agent should not only recall the past, but also write useful information, revise outdated
memories, and retrieve the right evidence for future decisions. How well current memory systems can fulfill
this role remains insufficiently evaluated.
(A) Recall-Only (B) Lacking Lifecycle level Diagnosis
Mon
Project Scope 
Updated.Tue
Budget 
Approved.Wed
Risk List 
Shared.Past Interactions / Memory
Write
Only Final QA TaskMaintain Retrieve Use
   Let’s me check...
No dynamic interactionThe answer is..
No lifecycle  evaluation
(C) Chat-Like, Low Pressure
What did I do 
before xxx?Let me think 
back to the xx
Observe
Act
 Tools
Environment
State 
Changes
No rich multimodal and environmental singals(D) WorldMemArena (ours)
WebpageFileImage
Tools ConversationConstructed world
InteractionObserve
 Act
Lifecycle  
Multimodal
Action-World
Fixed human designed What did I say about the project yesterday?
Figure2.Overview of multimodal agent memory evaluation. (a)
Recall-only evaluation. (b) Missing lifecycle-level diagnosis. (c)
Low-pressure chat-like settings. (d) WorldMemArena evaluates
multimodal memory through action-world interaction.Existing benchmarks fall short of this pic-
ture in three connected ways.(i)They
are often built around long dialogues or ex-
tended contexts Jiayang et al. (2026), test-
ing what models can remember rather than
how they use past experience to guide fu-
ture actions.(Figure 2(a)).(ii)As shown in
Figure 2(b), many evaluations Zhao et al.
(2026), Hu et al. (2026), Liu et al. (2025a)
report only final question answering accu-
racy, without checking whether relevant ev-
idence is written, updated, retrieved, and
used at the right time, making it difficult to
identify where memory failures occur.(iii)
Figure 2(c) shows that existing benchmarks
remain largely text-centric, often converting
images into captions before evaluation, with
limited real interaction and insufficient pres-
sure on multimodal evidence use.
Beyond these evaluation limitations, current benchmarks also miss a deeper shift in how agent memory is
built and used. Agent harness systems such as OpenClaw Steinberger (2025) and Codex OpenAI (2026a)
now let agents author and reorganize their own memory during interaction, blurring the line between the
memory module and the policy that uses it. In the spirit ofSutton’s Bitter Lesson, this invites a question the
field should be asking head-on:
When agents can continuously act in realistic environments and manage their own experience, should
agent memory be evaluated as a predefined write-update-retrieve pipeline, or as a capability formed through
interaction and used to support future decisions over time?
Answering this question requires an evaluation that treats memory as a process rather than a static snapshot.
As shown in Figure 1, we reframe multimodal agent memory as anAction and World Interaction Loop. At
each step, the agent observes a partially visible world, takes an action, receives feedback, and uses memory
to guide future actions and retain useful evidence. Under this view, memory has an observable lifecycle that
covers what is written, how it is maintained as the world changes, what evidence is retrieved, and how the
retrieved evidence is used. As shown in Figure 2(c), each stage can be evaluated using shared trajectory
evidence, rather than inferred from a single accuracy score.
2

WorldMemArena: Evaluating Multimodal Agent Memory Through Action–World Interaction
We instantiate this view inWorldMemArena, a multimodal multi-session benchmark of 400 long-horizon
interaction tasks spanning two complementary regimes.Lifelong Evolutionfocuses on personal and task states
that evolve across sessions, requiring systems to continuously track, update, and reuse long-term memories.
Agentic Executionplaces memory in realistic agent trajectories, where systems must extract reusable evidence
from observations, actions, and feedback rather than relying on pre-organized textual narratives. Each
session is annotated with gold memory points, state updates, distractors, and answer supporting evidence
chains. These annotations support diagnosis across memory writing, maintenance, retrieval, and use, while
providing a shared evidence base for comparing different memory systems.
Under a unified setting, the evaluation covers long-context agents, manually designed memory systems, and
memory agents built on execution harnesses. The results reveal four findings:(1)Storing more correct
memories does not guarantee better performance; the key is whether they can be used correctly at answer
time.(2)multimodal memory remains a major bottleneck, especially for complex visual reasoning tasks;(3)
memory performance varies across domains and degrades on agentic execution tasks, where key information
is distributed across actions, tool feedback, and state changes; and(4)manually designed memory systems
are more structured but less adaptive, while harness based memory agents are more flexible but remain
costly and less reliable. To sum up, our contributions are listed as follows:
•We formulate multimodal agent memory as anAction–World Interaction Loopand define a four stage
lifecycle of writing, maintenance, retrieval, and use.
•We introduceWorldMemArena, a multi-session multimodal benchmark coveringLifelong Evolutionand
Agentic Execution, with annotations for stage level memory diagnosis.
•We conduct a unified comparison of three representative agent memory paradigms, identifying their
respective strengths, failure modes, and implications for future design.
2. Related Works
Memory Benchmarks and Evaluation.Early memory benchmarks such as LoCoMo Maharana et al. (2024),
MemoryAgentBench Hu et al. (2025), and Realme Bian et al. (2026) focus on long-dialogue settings,
measuring whether models can retain and recall historical information. These benchmarks treat memory
as static recall over text and do not capture how memory supports dynamic task execution. More recent
agent-oriented benchmarks He et al. (2026), Zhao et al. (2026), Liu et al. (2024) incorporate tool traces,
environment feedback, and task dependencies, moving closer to realistic agent-environment interaction.
However, evaluation still centers on final success rates or question answering accuracy, making it difficult
to identify where and why memory fails. WorldMemArena differs by decomposing evaluation into writing,
maintenance, retrieval, and use, making it possible to localize where memory failures originate.
Multimodal Memory Mechanisms.Recent multimodal memory systems Long et al. (2025), Liu et al.
(2025b), Zhou et al. (2026), Fu et al. (2026) have demonstrated strong capabilities in visual understanding
and long-term information retention. Their evaluations, however, are largely confined to image and video
comprehension tasks, with limited attention to how memory operates within agent interaction loops. Bench-
marks that incorporate multimodal memory Bei et al. (2026), Lu et al. (2026), Yang et al. (2025), Wang et al.
(2024), Liu et al. (2026a) extend evaluation to images, videos, and dialogues, but cover a narrow range of
scenarios and apply limited evaluation pressure on evidence reuse. WorldMemArena broadens the scope
to multi-session agent interaction, testing whether systems can preserve, update, and reuse multimodal
evidence as tasks and environments evolve.
3

WorldMemArena: Evaluating Multimodal Agent Memory Through Action–World Interaction
3. Problem Formulation
3.1. Memory as an Action-World Interaction Loop
We define each instance as a long horizon agent-world interaction process. Given an initial task context
x, the agent does not directly observe the full world state. At step t, the world has a latent state zt, from
which the agent receives an observation ot. The agent then selects an action atbased on the observation
and its current memory state mt. After the action is executed, the environment updates its state and returns
feedbackf t:
ot=Ω(z t),a t=π(o t,mt),(z t+1,ft)=ℰ(z t,at).
Here,Ωmaps the latent world state to observable inputs,πdenotes the agent policy, andℰrepresents the
environment response, including both state transition and feedback generation. Observations may include
language, visual inputs or logs, while actions may include responses, tool calls, or execution.
Based on the above process, we denote the full trajectory as τ=(x;η 1, . . . ,η T), where each event ηt=
(ot,at,ft)records the observation, action, and feedback at step t. To evaluate long-horizon memory, we
further segment the trajectory into sessions, i.e., τ=τ(1)◦τ(2)◦⋯◦τ(S). Within each session, the agent
only observes local context, while the world state persists and evolves across sessions. This creates a natural
point: later decisions may depend on evidence that is no longer directly visible, and we focus on whether the
agent can recover and use such evidence through memory.
3.2. Memory Lifecycle as a Diagnostic Framework
The Action World Interaction Loop in §3.2 is architecture agnostic. It does not assume where memory is
stored or how it is represented. This allows us to evaluate different memory systems through four observable
phases of writing, maintenance, retrieval, and use. These phases capture the shared lifecycle of preserving
and reusing information across sessions.
♠Observe to Write.This phase evaluateswhether the system can identify future useful evidence from the
current session.Given the previous memory state ms−1and the current session trajectory τ(s), the system
produces a memory delta ∆s=Write(m s−1,τ(s)). The objective is selective retention, keeping information
that may support future responses or actions rather than storing the full trajectory.
♠Update and Consolidate.This phase evaluateshow newly written information is integrated into existing
memory.The system updates its state as ms=Maintain(m s−1,∆s). Since long-horizon interaction is not
purely additive, memory must support revision and consolidation as user preferences, task states, and
environmental evidence evolve.
♠Retrieve for Decision.This phase evaluateswhether the system can access the right evidence when a future
query or decision need arises.For a query q, retrieval returns Rs,q=Retrieve(m s,q). The goal extends
beyond semantic similarity to decision relevance, requiring the retrieved context to contain evidence needed
for the current answer or action.
♠Use and Act.This phase evaluateswhether retrieved memory is faithfully used in the final response or action.
Given qand retrieved evidence Rs,q, the system outputs ˆys,q=Answer(q,R s,q). Failures may still arise when
the system ignores relevant evidence, relies on outdated memory, or fails to translate prior experience into
appropriate action.
4

WorldMemArena: Evaluating Multimodal Agent Memory Through Action–World Interaction
Table1.Comparison ofWorldMemArenawith representative memory benchmarks. ✓= satisfies, ✓✗= partial support,
✗= does not satisfy.MMdenotes multimodal support;Dim.denotes evaluation dimensions;#QAthe number of QA
pairs;Img.the number of images;Sessionthe number of multi-turn sessions;Stepsthe number of interaction steps;
Modethe interaction paradigm. TheLifecyclegroup covers theWrite,Update(Upd.),Retrieve(Ret.), andUsestages of
memory.
LifecycleBenchmark MMDim. Eval. #QAImg.Session Steps ModeWriteUpd.Ret.Use
LoCoMo Maharana et al. (2024)✓ ✗ 5Static1,986 910 272 5,882Dialogue✗ ✗ ✓ ✓
LongMemEval Wu et al. (2024)✗5Static500–23,867 246,750Long-context✗ ✓ ✓ ✓
MemoryAgentBench Hu et al. (2025)✗4Static3,671–146 6,484Long-context✗ ✓ ✓ ✓
MMRC Xue et al. (2025)✓6Static2,105 1,193 457 11,784Dialogue✓ ✗ ✓ ✓ ✓
HaluMem Chen et al. (2026)✗3Static3,467–1,387 60,146Dialogue✓ ✓ ✗ ✓
RealMem Bian et al. (2026)✗4Static1,415–2,055 14,028Dialogue✗ ✓ ✗✓ ✓
Mem-Gallery Bei et al. (2026)✓3Static1,711 1,003 240 7,924Dialogue✓ ✗ ✓ ✓ ✓
AMA-Bench Zhao et al. (2026)✗4Interactive2,496–208 15,244Agent✓ ✗ ✓ ✓✗✓
MEMORYARENA He et al. (2026)✗4Interactive4,850–701 4,850Agent✗ ✗ ✗ ✓
WorldMemArena ✓ 27Interactive 24,258 15,595 8,489 59,858Dialog.+Agent ✓ ✓✓✓
4.WorldMemArena: Agent Memory in Action-World Interaction
Overview.WorldMemArena consists of 400 multi-session multimodal interaction tasks across two regimes
(Lifelong EvolutionandAgentic Execution). Each task is a temporally ordered sequence of sessions, where the
agent receives partial observations and must rely on memory to inform decisions in later sessions. To support
fine-grained diagnosis, every session is annotated with three types of structured labels.Gold memory points
specify the information that should be retained after a session, representing ground-truth memory content.
State updatesmark where previously stored information becomes outdated and must be revised, testing
whether the memory system can maintain temporal consistency.Distractorsintroduce plausible but irrelevant
or superseded information, testing whether the system can distinguish currently valid evidence from noise.
In addition, each question is paired withevidence points, the subset of gold memory points that are necessary
to answer it correctly. These annotations together enable evaluation at each stage of the memory lifecycle.
4.1. Memory Regimes
Agentic Execution.Each instance is derived from a real or realistic agent trajectory containing observations,
actions, and environment feedback. Later steps depend on earlier outcomes, so the agent must convert past
execution experience into reusable memory that informs future decisions.
Lifelong Evolution.Each instance is generated from a hidden world state that evolves across sessions. It
covers two scenarios: (1)lifelong personal evolution, where scattered interactions must be consolidated
into coherent personal memory; and (2)long-horizon projects, where task goals, intermediate results, and
feedback shift across stages, requiring the agent to maintain up-to-date progress memory.
Why both Regimes are Needed.As the Action-World Interaction Loop requires the agent to both observe
an evolving world and act within it, two demands on memory arise:(1) Persistent state trackingrequires
maintaining an accurate representation of an evolving world across sessions, which is evaluated byLifelong
Evolutionthrough controlled state evolution.(2) Action grounded experience reuserequires turning obser-
vations, action outcomes, and feedback into knowledge for later decisions, which is evaluated byAgentic
Executionthrough realistic execution trajectories.
5

WorldMemArena: Evaluating Multimodal Agent Memory Through Action–World Interaction
Synthetic Long-
Horizon Worlds
Persona / 
Project State
Timeline
Attachments
Planned Sessions
Observation-Action 
Trajectory
Tool / UI ActionsScreenshotsReconstructed Real-
Agent Worlds
Segment Window 1
Sessions or Local Windows
1 2 3
⋮Extract Memory 2
Create Gold Memory Points
Update + Deduplicate Construct QA Checkpoints 4
Recall Update
Conflict Non-exist
Visual Cross-step1 2 NMidway + Final QASession N-1
Session N
Session N+1
Session N+2
3
Merge, Revise, Remove 
Redundancy
(A)
Navigation
Household
Minecraft
OmniGibsonMobileImagesDocsWebFile
sExcel
GUI
Embodied
(B)
(1)
AcademicEducationFinanceHealth Software Startup012345
Overall Update Interference 
(2)1600018000
14000
12000
10000
8000
6000
4000
2000
015943
8731
5269
Figure3.Data construction pipeline and benchmark composition. (a) WorldMemArena constructs data around two
task regimes,Lifelong EvolutionandAgentic Execution, by segmenting sessions, extracting and updating gold memory
points, removing redundancy, and constructing midway and final QA checkpoints. (b-1) The benchmark covers both
GUI and embodied interaction settings. (b-2) The upper charts summarize gold memory points across the benchmark,
including update memory points and interference memory points. The lower chart shows the task distribution across
domains inLifelong Evolution.
4.2. Data Collection
As shown in Figure 3(a), WorldMemArena is constructed through a unified automated memory construction
pipeline with four steps.(1)Raw data is segmented into multi-session instances. ForLifelong Evolution,
a hidden world state is first defined and sessions are generated in temporal order, each revealing partial
information about a persona or project. ForAgentic Execution, existing agent trajectories are split at subgoal
boundaries, key feedback points, or state changes.(2)For each session window, gold memory points are
extracted, covering facts to retain, state updates to revise, and evidence required by future questions.(3)
Memory points are merged, revised, and deduplicated across sessions to remove redundancy and ensure
temporal consistency.(4)Question-answer pairs are constructed from the refined gold memory points,
covering 11 question types. Each instance is further reviewed by 2-3 human annotators to ensure quality.
4.3. Data Statics
Dataset Scale and Coverage.Table 1 compares WorldMemArena with existing benchmarks. Prior datasets
typically focus on either long-form dialogue or agentic trajectories, whereas this benchmark covers both
lifelong evolution and agentic execution. It contains 400 multi-session samples, with an average of 18.4
sessions and approximately 9.1K tokens per sample, making it substantially longer than existing multimodal
memory benchmarks. It further provides 24,258 QA pairs and 15,595 images or screenshots, supporting
broader question coverage and richer visual grounding. Most existing benchmarks do not evaluate the full
memory lifecycle; the closest prior work, HaluMem, addresses memory storage and recall but remains limited
to the textual modality.
Domain and Annotations.As shown in Figure 3(b),Lifelong Evolutioncovers 6 domain specific project
types, with each session containing an average of 4 images and 15-20 dialogue turns.Agentic Execution
preserves real agent execution traces and their corresponding visual states, covering 6 GUI subcategories
and 4 Embodied subcategories. Across both regimes, fine-grained lifecycle annotations are provided. Each
session contains an average of 10 key memory points, 3 update points, and 2 interference points. Each
sample further includes staged QA checkpoints with an average of 5 evaluation positions. Each question is
paired with retrieval evidence, where most require 1-2 evidence items and more complex questions require
5-6, covering both textual and visual information.
6

WorldMemArena: Evaluating Multimodal Agent Memory Through Action–World Interaction
4.4. Evaluation Protocol
Following the four lifecycle stages defined in §3.2, we evaluate whether a memory system can correctly
write, maintain, retrieve, and use memory across long horizon interactions. Detailed metric definitions and
settings are provided in the Appendix B.4.
Stage 1.For each session, newly written memory items are matched against the gold memory points
introduced in that session, withmemory recallused as the coverage metric. Each written item is further
assessed by an LLM-as-a-Judge and classified ascorrect, hallucinated, or irrelevant, distinguishing effective
memory writing from noisy or unsupported storage.
Stage 2.For gold memory points marked as updates, the system memory after the corresponding session
is examined to determine whether the new information is preserved and the obsolete version is properly
handled.An update is considered successful only when the revised memory is retained and the old version is
removed or overwritten. This criterion prevents simple accumulation of historical information from being
misclassified as effective memory maintenance.
Stage 3.For each checkpoint question, the retrieved memory items are matched against the annotated gold
evidence. The evidence may be grounded in either textual or visual information, and all evidence types are
evaluated under a unified coverage criterion.Recallmeasures whether the required evidence is retrieved,
while Normalized Discounted Cumulative Gain(NDCG)measures whether relevant evidence is ranked near
the top, thereby separating retrieval quality from final answer correctness.
Stage4.Checkpointquestionsaregroupedintofourcategoriesandtwelvecapabilityaxes:Basiccoversfactual
recall;Robustnesscovers dynamic update, memory boundary, and memory conflict;Reasoningcovers temporal
reasoning, knowledge reasoning, and test-time learning; andMultimodalcovers visual fact recall, visual
search, visual update, and cross-modal reasoning. Each question is jointly evaluated using LLM-as-a-Judge,
F1, and BLEU to reduce biases from any single metric.
5. Experiments
We evaluate three mainstream memory paradigms. Detailed settings are provided in Appendix A.
Long-Context Agents.To test whether frontier models can handle long-horizon memory tasks by relying
solely on context, these agents concatenate the full interaction history into the prompt as in-context memory,
without explicit abstraction, updating, or retrieval. We evaluate GPT-5.4-mini OpenAI (2026c), Qwen3.5
plus Team (2026), Gemini 3 flash Google DeepMind (2026), DeepSeek V4 DeepSeek-AI (2026) and Claude
Haiku 4.5 Anthropic (2025). As no independent memory state is exposed, only final question-answering
performance is measured.
Manually Designed Memory Systems.To assess whether explicitly engineered memory mechanisms can
improvememoryconstruction,maintenance,retrieval,anddownstreamuse,weevaluatetwotypesofsystems.
External memory agentssuch as MemGPT Packer et al. (2024) and Mem0 Chhikara et al. (2025) perform
information abstraction, consolidation, and retrieval through learned or hand-crafted modules.Retrieval-
augmented generation (RAG) systemssuch as UniversalRAG Yeo et al. (2026) store historical information in
an indexed document store and access it via retrieval. To control for backbone differences, all systems use
GPT-5.4-nano OpenAI (2026c) as the base model. Because these systems expose observable memory states
and retrieval outputs, the full memory lifecycle can be evaluated.
Harness-Based Memory Agents.To examine whether agents can autonomously manage memory without
a fixed external module, we evaluate agent harnesses where memory is written, maintained, retrieved,
7

WorldMemArena: Evaluating Multimodal Agent Memory Through Action–World Interaction
and used by the harness itself during interaction. We test OpenClaw Steinberger (2025) paired with GPT-
5.4 OpenAI (2026b) and DeepSeek-V4, and Codex OpenAI (2026a) paired with GPT-5.4, feeding session
contexts sequentially and testing with staged checkpoint QA. Since the internal memory process is difficult
to decompose, we primarily conduct end-to-end evaluation.
Table2.Performance of baselines on memory quality and question answering (QA) quality. All values are reported in
% as the mean across samples.Memory metrics:Recall=Memory Recall,Corr=Memory Correctness,Hallu=Memory
Hallucination,Irrel=Memory Irrelevance,Update=Update Handling, andIntRej=Interference Rejection, averaged
over samples containing interference items.QA metrics:QA-C=QA Correct,QA-H=QA Hallucination,QA-O=QA
Omission, andRC=Retrieval Coverage. In the External Memory Agents, the dashed line separates caption-based text
only systems above from multimodal systems using both images and text below. All judgments are conducted with
GPT-5.4-mini as the evaluator.
Method Memory Quality QA Quality
Recall↑ Corr↑Hallu↓ Irrel↓Update↑ IntRej↑ QA-C↑QA-H↓QA-O↓ RC↑F1↑BLEU-1↑
RAG
Qwen3-VL-Embedding-8B Zhang et al. (2025) 86.22 98.15 1.18 0.67 59.0228.21 51.86 28.0220.1273.44 32.21 17.84
UniversalRAG Yeo et al. (2025) 84.56 96.90 2.42 0.67 57.98 27.34 39.62 31.67 28.70 60.93 27.06 14.16
External Memory
A-Mem Xu et al. (2025) 52.54 96.60 2.57 0.83 58.86 58.9454.63 22.94 22.43 74.19 34.40 19.86
MemGPT Packer et al. (2023) 85.20 96.98 2.28 0.74 58.18 25.4457.8122.05 20.1484.9933.21 18.33
SimpleMem Liu et al. (2026c) 78.84 96.96 1.44 1.35 53.43 24.79 42.93 25.60 31.47 48.03 26.00 12.30
Omni-SimpleMem Liu et al. (2026b) 58.48 72.92 15.95 9.95 52.65 43.22 43.03 32.24 24.72 62.55 25.86 12.52
M2A Feng et al. (2026)86.8397.47 1.25 1.28 56.41 23.42 50.14 29.29 20.57 64.62 31.77 17.54
ViLoMem Bo et al. (2026) 85.96 81.61 10.65 7.74 55.73 24.93 49.77 25.20 25.02 70.71 29.51 15.63
MIRIX Wang and Chen (2025) 64.79 73.50 5.15 1.58 56.97 31.42 44.4620.7934.75 61.90 24.90 12.65
AUGUSTUS Jain et al. (2025) 84.63 96.66 2.63 0.70 57.42 28.85 42.01 32.38 25.61 57.33 27.24 13.87
Best inbold, second-best underlined .
5.1. Main Results
Table 2 reports the overall performance of different human designed systems across the full memory lifecycle.
We identify four main findings. ❶Multimodal memory is still not effectively used.Text-based systems
such as MemoryGPT and A-Mem achieve more stable final answer quality, while multimodal systems such
as ViLoMem and MIRIX show limited downstream gains despite access to visual inputs. This suggests
that current systems still struggle to encode and reuse visual evidence as reliable long term memory. ❷
High memory quality does not necessarily lead to high QA quality.High memory quality does not
necessarily lead to high QA quality. Qwen3-VL-Embedding and M2A perform well in memory storage and
recall, but their final answers remain limited. This indicates that correct memory writing is insufficient;
systems must also retrieve and use the right evidence during answer generation. ❸Retrieval remains a
key bottleneck for final performance.MemoryGPT achieves the strongest evidence retrieval and answer
correctness, while A-Mem uses retrieved information effectively despite lower memory coverage. In contrast,
AUGUSTUS constructs reasonably good memories but fails to surface key evidence at inference time, limiting
its final QA performance. ❹Most systems remain weak in memory updating and distractor rejection.
Nearly all systems are brittle under information changes and interfering content, indicating that they tend to
accumulate memories rather than maintain a consistent long-term state. This suggests that current human
designed memory systems still focus more on how much they remember than on how well they maintain
and update memory over time.
8

WorldMemArena: Evaluating Multimodal Agent Memory Through Action–World Interaction
Table3.QA quality results for all base models and harness agents.
QA-CdenotesQA Correct,QA-HdenotesQA Hallucination, andQA-O
denotesQA Omission.
Method QA Quality
QA-C↑QA-H↓QA-O↓ F1↑BLEU-1↑
Base Model
Qwen3.5 plus 51.05 16.90 32.05 21.04 8.68
Deepseek V469.13 11.4619.41 28.18 13.61
Gemini 3 Flash 51.69 23.69 24.62 22.93 10.32
Claude Haiku 4.5 36.71 25.47 37.83 22.05 10.79
GPT 5.4-mini 58.27 27.8613.8721.31 8.76
Harness
Codex-GPT 5.4-nano53.6220.7625.62 32.5610.12
OpenClaw-DeepSeek V4 50.29 15.57 34.14 28.3818.16
OpenClaw-GPT 5.4-nano 48.3119.5532.13 30.32 15.71Table 3 compares final answer perfor-
mance between long context agents and
harness based memory agents. Most long
context agents perform poorly, with some
falling below dedicated memory systems,
indicating that the benchmark requires
long horizon evidence integration rather
than context extension alone. DeepSeek
V4 benefits mainly from its larger context
window, while standard context models
remain limited. Harness based memory
agents outperform most human designed
memory systems, suggesting that agent
managed memory is more flexible. How-
ever, the same backbone performs differ-
ently across harnesses, showing that native memory design and adaptation mechanisms also affect final
performance.
6. Analysis
[RQ1]Where do memory failures occur in the lifecycle?
Memory failures occur across the full lifecycle and compound over time. (i)Figure 4(a) shows that storing more
memories does not necessarily make them usable; even with high storage coverage, systems may fail to
retrieve the key evidence needed for the current decision.(ii)As illustrated in Figure 4 (b), most systems rely
on append only updates, adding new information when evidence changes rather than revising, removing, or
reorganizing obsolete memories.(iii)Over long trajectories, Figure 4(c) captures a compounding pattern in
which early omissions reduce later evidence availability, while incorrect outputs may contaminate future
memory updates and further induce hallucinated answers.
78.8%21.2%								
			(C)Dialogue session(A)
	

2973427
10546(B)
Basic FactVisual-Experiential FactBaselinerangeAverage
RQ1
Figure4.(a) shows the trend of average QA accu-
racy across dialogue sessions. (b), (c) summarize
memory-point composition in terms of update op-
erations (Append, Revise, Delete, and Merge) and
fact salience.[RQ2]Are memory system designs constrained by
domain-specific data?
Memory performance varies across domains.As shown in
Figure 5(a-b), most systems perform better inLifelong Evo-
lutionthan inAgentic Execution. This suggests that existing
methods are more suited to explicit long-term state evolu-
tion,whileextractingusablememoryfromactiontracesand
environment feedback remains challenging. Performance
also differs across tasks, with long-horizon embodied tasks
such as visual navigation posing greater challenges, sug-
gesting that current systems still struggle to track memory
across sessions and use it for later decisions.
[RQ3]How does multimodal affect the memory lifecycle?
Memory systems still struggle with complex visual memory
tasks.As shown in Figure 5(c), systems perform relatively
stably on simple visual fact recall, but degrade on tasks that
9

WorldMemArena: Evaluating Multimodal Agent Memory Through Action–World Interaction
depend on long interaction histories, such as cross-modal reasoning. This suggests that the core challenge
of multimodal memory is to maintain visual states over time and integrate visual evidence with historical
context when needed.
1'.##5!'-   '-/17'-/17"-.++-0,''-6'.	!-$'&&+.)+-0,''- .+5'12#,'-/17!+/'-
		
	
	


	

	
	








			



%#&'-+%&4%#3+/.+.#.%''#,3*/(36#1'3#1340#,4'-	





	



		






	
	
	
		



			





		

Agentic ExecutionLifelong EvolutionQAAccuracy (%)AverageQAAccuracy (%)

		








		


	
	
		

				
	Agentic ExecutionLifelong EvolutionAverageA-MemM2AMIRIXQwen3-VL-Embedding-8BUniversalRAGMemoryAUGUSTUSMemoryMGMemoryOmini-SimpleMemSimpleMemViLoMem××
	
M2AA-MemQwen3-VL RAGMGMemoryAUGUSTUSMemoryQAAccuracy (%)(A)(B)(C)
VisualSearchVisualUpdateCross-modalReasoning
Figure5.Performance comparison across scenarios and visual QA tasks. (a) Heatmap of baseline performance across
fine-grained task categories in the agent-scenario and long-dialogue settings. (b) Average baseline performance under
the two settings, where long-dialogue tasks show higher overall performance than agent-scenario tasks.(c) Box plots of
baseline performance across three visual QA task types.
[RQ4]What strengths and limitations do different memory systems exhibit across task types?
	

	

	
Recall@k(%)(A)
kNDCG@k(%)(B)
Question typeA-MemM2AQwen3-VL RAGSimpleMem
RQ4
Figure6.(a) Fine-grained QA performance of different base-
lines on individual tasks. (b) Recall@K and NDCG@K (Nor-
malized Discounted Cumulative Gain) trends under different
retrieval cutoffs.Memory performance depends more on system
design than on backbone scale or retrieval vol-
umeAs shown in Table 2, most systems achieve
high memory storage recall and writing quality,
yet their evidence recall at question-answering
time drops substantially, indicating that cor-
rectly stored memories are not effectively sur-
faced when needed. Figure 6(b) further shows
that increasing the retrieval scope does not al-
ways improve answer quality, as longer contexts
may introduce redundant, outdated, or irrele-
vant evidence. This issue is more evident in
multimodal tasks, where long interactions cre-
ate substantial visual redundancy and make key
visual evidence harder to locate and use.
[RQ5]Can agents turn memory into action?
Past memory is not reliably converted into reusable knowledge, and experiential evidence remains fragile.As
shown in Figure 6(a), systems perform worse on reasoning and test-time learning tasks, suggesting that they
are better at storing past information than using it to guide future decisions. Analysis of retrieved memory
points in Figure 4(c) shows that retrieved memories are dominated by explicit textual facts, whereas tool
feedback, failed actions, and visual details are often omitted.
[RQ6]How far do human designed memory systems fall short in agentic memory?
10

WorldMemArena: Evaluating Multimodal Agent Memory Through Action–World Interaction
0.35 0.40 0.45 0.50 0.55 0.60
QA accuracy020k40k60k80k100k120kTotal tokens per session
UniversalRAGMemory
AUGUSTUSMemory
SimpleMemOmni-SimpleMem
MIRIXOpenClaw-GPT
ViLoMemM2AOpenClaw-DeepSeek
Qwen3-VL-Embedding-8BCodex
A-MemMGMemory
Figure7.Token efficiency & QA performance trade-
off among different baselines. Circle size indicates
average inference time, with larger circles denoting
higher time cost.Fixed memory architectures struggle to adapt to dynamic
memory demands.As shown in Figure 7, human de-
signed memory systems perform comparably to harness
based methods on simpler long-horizon tasks, but their
fixed pipelines become limiting in complex agentic set-
tings where memory must adapt to task feedback and
environmental changes. Harness based agentic memory
managers are more flexible because they can record, re-
trieve, and revise memory during interaction. However,
this result also shows that current harness-based mem-
ory remains computationally expensive and framework-
dependent, limiting its stability and transferability.
7. Discussion
The experiments above show that long horizon agent
memory remains fragile. Strong storage signals often fail
to translate into reliable decisions, and multimodal and
interactive settings expose additional failure modes. We
distill four directions for future work.
❖Memory should be shaped through interaction, not fixed as a module.Our results show that higher
storage quality does not necessarily lead to better performance (Table 2), while harness-based agents without
explicit memory modules outperform some manually designed memory pipelines (Table 3). This suggests
that effective memory is better understood as a capability shaped by task pressure, rather than as a module
that can be optimized in isolation. Future work should explore training paradigms that develop memory
through end-to-end interaction objectives.
❖Memory requires consistent state maintenance, not continuous accumulation.Current systems
accumulate information but rarely revise or remove obsolete entries (Figure 4b). Effective memory should
be modeled as mutable state that supports revision, conflict resolution, and selective forgetting. New
architectures and evaluations are needed that reward state consistency rather than raw coverage.
❖Effective use of multimodal memory.Most systems compress visual observations into textual memory,
which often loses spatial, temporal, and procedural details. Our analysis shows that current systems
still perform poorly on complex visual tasks, especially when they need to use visual cues and interaction
experienceforreasoning(Figure5c). Futureworkshoulddeveloparchitecturesthatpreservevisualmemories
in usable forms, with metrics that evaluate whether these memories truly support reasoning and decision-
making.
❖Memory evaluation should focus on learning from experience, not retrospective QA.Current evalua-
tions often rely on checkpoint QA to measure memory, but the ultimate goal of agent memory is not merely
to answer questions about the past, but to improve future behavior. Our experiments show that systems
are better at storing facts than at using them for reasoning or learning (Figure 4a). Future benchmarks
should evaluate whether agents can learn from prior experience and failures, rather than merely retrieve
past information, and improve behavior across sessions.
11

WorldMemArena: Evaluating Multimodal Agent Memory Through Action–World Interaction
8. Conclusion
We presentedWorldMemArena, a multimodal multi-session benchmark that evaluates agent memory
through the lens of an Action World Interaction Loop. By decomposing memory into four observable stages
and annotating each session with gold memory points, updates, and distractors, we enable stage level
diagnosis across long context agents, manually designed memory systems, and harness-based memory agents.
Experiments show that storage quality alone does not predict final performance, that memory maintenance
remains dominated by append only behavior, and that visual evidence is largely reduced to text. These
findings suggest that the field should move beyond optimizing memory as a static module and toward
developing memory as an adaptive capability grounded in interaction.
12

WorldMemArena: Evaluating Multimodal Agent Memory Through Action–World Interaction
References
Anthropic. Introducing claude haiku 4.5. https://www.anthropic.com/news/claude-haiku-4-5 ,
October 2025.
Anthropic. Introducing Claude Opus 4.6. https://www.anthropic.com/news/claude-opus-4-6 ,
2026a.
Anthropic. Claude Code.https://www.anthropic.com/product/claude-code, 2026b.
Yuanchen Bei, Tianxin Wei, Xuying Ning, Yanjun Zhao, Zhining Liu, Xiao Lin, Yada Zhu, Hendrik Hamann,
Jingrui He, and Hanghang Tong. Mem-gallery: Benchmarking multimodal long-term conversational
memory for mllm agents, 2026. URLhttps://arxiv.org/abs/2601.03515.
Haonan Bian, Zhiyuan Yao, Sen Hu, Zishan Xu, Shaolei Zhang, Yifu Guo, Ziliang Yang, Xueran Han, Huacan
Wang, and Ronghao Chen. Realmem: Benchmarking llms in real-world memory-driven interaction, 2026.
URLhttps://arxiv.org/abs/2601.06966.
Weihao Bo, Shan Zhang, Yanpeng Sun, Jingjing Wu, Qunyi Xie, Xiao Tan, Kunbin Chen, Wei He, Xiaofan
Li, Na Zhao, Jingdong Wang, and Zechao Li. Agentic learner with grow-and-refine multimodal semantic
memory, 2026. URLhttps://arxiv.org/abs/2511.21678.
Ding Chen, Simin Niu, Kehang Li, Peng Liu, Xiangping Zheng, Bo Tang, Xinchi Li, Feiyu Xiong, and Zhiyu Li.
Halumem: Evaluating hallucinations in memory systems of agents, 2026. URL https://arxiv.org/
abs/2511.03506.
Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-
ready ai agents with scalable long-term memory.arXiv preprint arXiv:2504.19413, 2025.
DeepSeek-AI. Deepseek-v4: Towards highly efficient million-token context intelligence, 2026.
Junyu Feng, Binxiao Xu, Jiayi Chen, Mengyu Dai, Cenyang Wu, Haodong Li, Bohan Zeng, Yunliu Xie, Hao
Liang, Ming Lu, and Wentao Zhang. M2a: Multimodal memory agent with dual-layer hybrid memory for
long-term personalized interactions, 2026. URLhttps://arxiv.org/abs/2602.07624.
Muxin Fu, Xiangyuan Xue, Yafu Li, Zefeng He, Siyuan Huang, Xiaoye Qu, Yu Cheng, and Yang Yang.
Latentmem: Customizing latent memory for multi-agent systems, 2026. URL https://arxiv.org/
abs/2602.03036.
Google DeepMind. Gemini 3 flash.https://deepmind.google/models/gemini/flash/, 2026.
Zexue He, Yu Wang, Churan Zhi, Yuanzhe Hu, Tzu-Ping Chen, Lang Yin, Ze Chen, Tong Arthur Wu,
Siru Ouyang, Zihan Wang, Jiaxin Pei, Julian McAuley, Yejin Choi, and Alex Pentland. Memoryarena:
Benchmarking agent memory in interdependent multi-session agentic tasks, 2026. URL https://arxiv.
org/abs/2602.16313.
Yuanzhe Hu, Yu Wang, and Julian McAuley. Evaluating memory in llm agents via incremental multi-turn
interactions.arXiv preprint arXiv:2507.05257, 2025.
Yuanzhe Hu, Yu Wang, and Julian McAuley. Evaluating memory in llm agents via incremental multi-turn
interactions, 2026. URLhttps://arxiv.org/abs/2507.05257.
13

WorldMemArena: Evaluating Multimodal Agent Memory Through Action–World Interaction
Jitesh Jain, Shubham Maheshwari, Ning Yu, Wen-mei Hwu, and Humphrey Shi. Augustus: An llm-driven
multimodal agent system with contextualized user memory.arXiv preprint arXiv:2510.15261, 2025.
Cheng Jiayang, Dongyu Ru, Lin Qiu, Yiyang Li, Xuezhi Cao, Yangqiu Song, and Xunliang Cai. Amemgym:
Interactive memory benchmarking for assistants in long-horizon conversations, 2026. URL https://
arxiv.org/abs/2603.01966.
HalidAbdulrahimKadiandKasimTerzić. Agent-arena: Ageneralframeworkforevaluatingcontrolalgorithms,
2025. URLhttps://arxiv.org/abs/2504.06468.
Chengzhi Liu, Zhongxing Xu, Qingyue Wei, Juncheng Wu, James Zou, Xin Eric Wang, Yuyin Zhou, and Sheng
Liu. More thinking, less seeing? assessing amplified hallucination in multimodal reasoning models, 2025a.
URLhttps://arxiv.org/abs/2505.21523.
Chengzhi Liu, Yuzhe Yang, Yue Fan, Qingyue Wei, Sheng Liu, and Xin Eric Wang. Reasoning within the mind:
Dynamic multimodal interleaving in latent space, 2026a. URL https://arxiv.org/abs/2512.12623 .
Jiaqi Liu, Zipeng Ling, Shi Qiu, Yanqing Liu, Siwei Han, Peng Xia, Haoqin Tu, Zeyu Zheng, Cihang Xie,
Charles Fleming, Mingyu Ding, and Huaxiu Yao. Omni-simplemem: Autoresearch-guided discovery of
lifelong multimodal agent memory, 2026b. URLhttps://arxiv.org/abs/2604.01007.
Jiaqi Liu, Yaofeng Su, Peng Xia, Siwei Han, Zeyu Zheng, Cihang Xie, Mingyu Ding, and Huaxiu Yao.
Simplemem: Efficient lifelong memory for llm agents.arXiv preprint arXiv:2601.02553, 2026c.
Junming Liu, Yifei Sun, Weihua Cheng, Haodong Lei, Yirong Chen, Licheng Wen, Xuemeng Yang, Daocheng
Fu, Pinlong Cai, Nianchen Deng, Yi Yu, Shuyue Hu, Botian Shi, and Ding Wang. Memverse: Multimodal
memory for lifelong learning agents, 2025b. URLhttps://arxiv.org/abs/2512.03627.
Xiao Liu, Tianjie Zhang, Yu Gu, Iat Long Iong, Yifan Xu, Xixuan Song, Shudan Zhang, Hanyu Lai, Xinyi
Liu, Hanlin Zhao, Jiadai Sun, Xinyue Yang, Yu Yang, Zehan Qi, Shuntian Yao, Xueqiao Sun, Siyi Cheng,
Qinkai Zheng, Hao Yu, Hanchen Zhang, Wenyi Hong, Ming Ding, Lihang Pan, Xiaotao Gu, Aohan Zeng,
Zhengxiao Du, Chan Hee Song, Yu Su, Yuxiao Dong, and Jie Tang. Visualagentbench: Towards large
multimodal models as visual foundation agents, 2024. URLhttps://arxiv.org/abs/2408.06327.
Lin Long, Yichen He, Wentao Ye, Yiyuan Pan, Yuan Lin, Hang Li, Junbo Zhao, and Wei Li. Seeing, listening,
remembering, and reasoning: A multimodal agent with long-term memory, 2025. URL https://arxiv.
org/abs/2508.09736.
Yihao Lu, Wanru Cheng, Zeyu Zhang, and Hao Tang. Mma: Multimodal memory agent, 2026. URL
https://arxiv.org/abs/2602.16493.
Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang.
Evaluating very long-term conversational memory of llm agents. InProceedings of the 62nd Annual Meeting
of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13851–13870, 2024.
OpenAI. Codex: OpenAI’s Coding Agent.https://developers.openai.com/codex, 2026a.
OpenAI. Gpt-5.4.https://platform.openai.com/docs/models/gpt-5.4, 2026b.
OpenAI. Introducing gpt-5.4 mini and nano. https://openai.com/index/
introducing-gpt-5-4-mini-and-nano/, March 2026c.
14

WorldMemArena: Evaluating Multimodal Agent Memory Through Action–World Interaction
Charles Packer, Vivian Fang, Shishir_G Patil, Kevin Lin, Sarah Wooders, and Joseph_E Gonzalez. Memgpt:
towards llms as operating systems. 2023.
Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez.
Memgpt: Towards llms as operating systems, 2024. URLhttps://arxiv.org/abs/2310.08560.
Qwen Team. Qwen3.5: Towards Native Multimodal Agents. https://qwen.ai/blog?id=qwen3.5 , 2026.
Peter Steinberger. Openclaw: Your own personal AI assistant. https://github.com/openclaw/
openclaw, 2025.
Qwen Team. Qwen3.5: Accelerating productivity with native multimodal agents, February 2026. URL
https://qwen.ai/blog?id=qwen3.5.
Xiyao Wang, Yuhang Zhou, Xiaoyu Liu, Hongjin Lu, Yuancheng Xu, Feihong He, Jaehong Yoon, Taixi
Lu, Gedas Bertasius, Mohit Bansal, Huaxiu Yao, and Furong Huang. Mementos: A comprehensive
benchmark for multimodal large language model reasoning over image sequences, 2024. URL https:
//arxiv.org/abs/2401.10529.
Yu Wang and Xi Chen. Mirix: Multi-agent memory system for llm-based agents, 2025. URL https:
//arxiv.org/abs/2507.07957.
DiWu,HongweiWang,WenhaoYu,YuweiZhang,Kai-WeiChang,andDongYu. Longmemeval: Benchmarking
chat assistants on long-term interactive memory.arXiv preprint arXiv:2410.10813, 2024.
Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-mem: Agentic memory for
llm agents.arXiv preprint arXiv:2502.12110, 2025.
Haochen Xue, Feilong Tang, Ming Hu, Yexin Liu, Qidong Huang, Yulong Li, Chengzhi Liu, Zhongxing Xu,
Chong Zhang, Chun-Mei Feng, et al. Mmrc: A large-scale benchmark for understanding multimodal large
language model in real-world conversation. InProceedings of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), pages 22477–22503, 2025.
Rui Yang, Hanyang Chen, Junyu Zhang, Mark Zhao, Cheng Qian, Kangrui Wang, Qineng Wang, Teja Venkat
Koripella, Marziyeh Movahedi, Manling Li, Heng Ji, Huan Zhang, and Tong Zhang. Embodiedbench:
Comprehensive benchmarking multi-modal large language models for vision-driven embodied agents,
2025. URLhttps://arxiv.org/abs/2502.09560.
Woongyeong Yeo, Kangsan Kim, Soyeong Jeong, Jinheon Baek, and Sung Ju Hwang. Universalrag:
Retrieval-augmented generation over corpora of diverse modalities and granularities.arXiv preprint
arXiv:2504.20734, 2025.
Woongyeong Yeo, Kangsan Kim, Soyeong Jeong, Jinheon Baek, and Sung Ju Hwang. Universalrag: Retrieval-
augmented generation over corpora of diverse modalities and granularities, 2026. URL https://arxiv.
org/abs/2504.20734.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, et al. Qwen3 embedding: Advancing text embedding and reranking through
foundation models.arXiv preprint arXiv:2506.05176, 2025.
Yujie Zhao, Boqin Yuan, Junbo Huang, Haocheng Yuan, Zhongming Yu, Haozhou Xu, Lanxiang Hu, Abhilash
Shankarampeta, Zimeng Huang, Wentao Ni, Yuandong Tian, and Jishen Zhao. Ama-bench: Evaluating
long-horizon memory for agentic applications, 2026. URLhttps://arxiv.org/abs/2602.22769.
15

WorldMemArena: Evaluating Multimodal Agent Memory Through Action–World Interaction
Jinsong Zhou, Yihua Du, Xinli Xu, Luozhou Wang, Zijie Zhuang, Yehang Zhang, Shuaibo Li, Xiaojun Hu,
Bolan Su, and Ying cong Chen. Videomemory: Toward consistent video generation via memory integration,
2026. URLhttps://arxiv.org/abs/2601.03655.
16

WorldMemArena: Evaluating Multimodal Agent Memory Through Action–World Interaction
A. Experimental Setting
Unless stated otherwise, every baseline shares the same backbone and decoding configuration to keep
comparisons fair. The answer-stage and judge LLMs both run with temperature 0.0, a maximum completion
budget of 16,384tokens (which covers reasoning plus output for GPT-5-class models), and a per-call timeout
of300s, with up to 10concurrent requests. Backbone variation is controlled at the model level only: GPT-
5.4-mini, Deepseek-V4, Claude Haiku 4.5, Gemini 3 Flash, and Qwen3.6-plus are evaluated under identical
prompts. Memory adapters that need an embedding model use OpenAI’s text-embedding-3-small
(1,536-dim); multimodal retrievers default to Qwen3-VL-Embedding-8B and the GME Qwen2-VL-2B encoder.
Retrieval is capped at top- K=10 items per query for both text and multimodal paths; the answerer’s
effective context window is 128,000 tokens with an 8,000-token reserve for the system and answer prompt.
Image-augmented QA caps at five images per question and 45MB of merged payload to stay within provider
limits. The LLM judge inherits the answer-stage model and runs with temperature 0.0and up to 5parallel
workers.
B. Evaluation Metrics
B.1. Notation
A single evaluation instance corresponds to one trajectory τ=τ(1)◦⋯◦τ(S)split into Ssessions; every
per-instance metric below is first aggregated within τand then averaged across instances. Within session
τ(s),𝒟scollects theadd/updatememory items the policy πwrites into the memory state mt,𝒢sis the set of
gold memory points the system is expected to remember, and ℐsthe set of goldinterferencepoints it should
reject. Each gold point g∈𝒢 scarries an importance weight wg≥0(default 1). For a QA q,y⋆
qis the gold
answer, ˆyqthe generated answer, and 𝒱qthe gold evidence points the QA relies on. Per-memory and per-QA
labels are produced by an LLM judge.
B.2. Memory metrics
These metrics decompose “did the agent build a useful memory” into two complementary axes:coverage
of what should have been remembered, andpurityof what was actually stored. Lifelong benchmarks also
stress two failure modes outside that simple recall/precision split, namely silently keeping stale facts and
absorbing noise on purpose, so we addUpdateandIntRejto capture them.
•Memory Recall (Recall).Coverage of the gold memory points by the system’s add/update delta. An
LLM judge decides, semantically, which g∈𝒢 sis supported by some item in 𝒟s; let𝒞s⊆𝒢 sbe the covered
subset. Recall is importance-weighted because the gold set mixes high-stakes facts and incidental details,
and we report
Recall s=∑g∈𝒞 swg
∑g∈𝒢 swg,(1)
averaged across sessions with ∣𝒢s∣>0(sessions with no gold are uninformative and dropped). A semantic
judge avoids penalising harmless paraphrases or summarisation by the agent.
•MemoryCorrectness/Hallucination/Irrelevant(Corr,Hallu,Irrel).Recallisblindtogarbage:
an agent that dumps the entire dialogue into memory looks excellent. We classify each stored item
17

WorldMemArena: Evaluating Multimodal Agent Memory Through Action–World Interaction
m∈𝒟 sinto three exclusive labels:correct(overall faithful, minor imprecision allowed),hallucination
(partly right but contradicts the dialogue on a concrete fact), anderror(fundamentally wrong, e.g. an
event that never happened). With per-session countsnC
s,nH
s,nE
s,
Corr s=nC
s
∣𝒟s∣, Hallu s=nH
s
∣𝒟s∣, Irrel s=nE
s
∣𝒟s∣,(2)
averaged across sessions with ∣𝒟s∣>0.HalluandIrrelare reported as “lower is better”: they expose
the price an agent pays for a highRecall.
•Update Handling (Update).Long-horizon memory must overwrite stale facts when the world changes
(e.g., the user moves house). For every gold update we inspect the post-session memory snapshot and
label it asupdated(only the new fact is kept),both(new and old coexist), oroutdated(only the old fact
survives). Pooling counts across the sessions of an instance,
Update=1.0⋅N updated+0.5⋅N both+0.0⋅N outdated
Ntotal.(3)
The half-credit onbothreflects that the agent has the new fact but failed to invalidate the old one;
downstream QA can still surface the wrong answer.
•Interference Rejection (IntRej).Real conversations contain casual remarks, jokes, and corrections
that the agent shouldnotcommit to memory. For every gold interference point g∈ℐ sthe post-session
snapshot is classified asrejectedormemorized, and
IntRej=Nrejected
Nrejected+N memorized.(4)
AhighRecallpairedwithlowIntRejisthesignatureofanindiscriminatewriterthathoardseverything;
the two metrics together separate selective memory from a transcript.
B.3. QA metrics
The memory metrics above audit the memory store directly. The QA metrics measure the downstream effect:
given the memory the agent built, can it answer questions whose evidence is no longer in the local context?
For every QA q, the judge compares ˆyqagainst y⋆
qand the gold evidence list 𝒱qand emits a single label
ℓq∈{Correct, Hallucination, Omission} ; letn⋆be the number of QAs in the instance that received a valid
label.
•QACorrect/Hallucination/Omission(QA-C,QA-H,QA-O).Thethreelabelsseparatethequalitatively
different ways an answer can fail: confident-but-wrong (Hallucination) is treated separately from refusal
or “I don’t know” (Omission), since they imply different failure modes of the memory pipeline.
QA-C=∣{q∶ℓ q=Correct}∣
n⋆ ,(5)
QA-H=∣{q∶ℓ q=Hallucination}∣
n⋆ ,(6)
QA-O=∣{q∶ℓ q=Omission}∣
n⋆ .(7)
18

WorldMemArena: Evaluating Multimodal Agent Memory Through Action–World Interaction
•Answer F 1.The judge label is binary at the QA level; F 1adds a fine-grained surface-form signal that
captures partial overlap on short factual answers. We tokenise both answers with a normaliser that
lowercases, drops the stopwords a/an/the/and, strips punctuation while preserving decimals, and
applies Porter stemming. Writing ̃T(⋅)for the resulting token multiset andC q=̃T(ˆyq)∩̃T(y⋆
q),
Pq=∣Cq∣
∣̃T(ˆyq)∣,R q=∣Cq∣
∣̃T(y⋆q)∣,F 1,q=2PqRq
Pq+R q(0if∣C q∣=0).(8)
Stemming reduces the penalty for harmless inflection (“walk”/“walked”) and is appropriate at the
answer-string level.
•BLEU-1.BLEU-1 (unigram BLEU with add- ϵsmoothing) is reported alongside F 1as a precision-leaning
surface metric: it weights repeated terms and is less generous to padding, so the gap between F 1and
BLEU-1 is informative on its own. Tokenisation uses the same normaliserwithoutPorter stemming, so
BLEU-1 stays comparable to standard implementations.
B.4. Retrieval metrics
Memory and QA quality measure “what was stored” and “what was answered”; the retrieval metrics measure
thebridgebetweenthem, i.e.whethertherelevantpastevidenceisactuallysurfacedwhenaquestionisasked.
ForeveryQA qthesystemreturnsanorderedlistofretrieveditems rq=(r q,1,rq,2, . . .)againstthegoldevidence
set𝒱q. We use a soft match predicate 1{r≈g} that returns 1when (i) the gold memory id is contained in the
retrieveditem’sidentifiers, (ii)thesource-sessionidparsedfromthegoldmatchesthesessionthatcontributed
r, or (iii) the normalised gold content is a substring of, or has ≥0.75token-overlap ratio with, the normalised
retrieved text. These three rules absorb superficial id mismatches between heterogeneous baselines and
avoid awarding credit purely on verbatim string equality. Letcov K(q)={g∈𝒱 q∶∃k≤K,r q,k≈g}.
•Retrieval Coverage (RC).A rank-agnostic, semantic-level check: an LLM judge reads the full top- Klist
and decides how many gold evidence points are supported anywhere in it. Letting ∣Q∣be the QA count
of the instance,
RC=1
∣Q∣∑
q∈Qcovered q
∣𝒱q∣,(9)
where covered qis the judge’s count.RCcaptures retrieval quality without committing to a particular
rank position, since an answer can succeed as long as the evidence is present and the answerer reads the
list.
•Recall@ K.A strict, rank-bounded counterpart ofRCbased on the soft match predicate (no judge,
deterministic). It probes whether the top of the list alone is informative:
Recall@K q=∣cov K(q)∣
∣𝒱q∣,K∈{1, 5, 10}.(10)
The K=1value is the harshest: it rewards retrievers that put the right evidencefirstrather than
somewhere in the top decile.
•NDCG@ K.Recall@ Kignores ranking inside the top- K. NDCG@ Kcloses that gap by discounting later
ranks. We first turn the retrieval list into a binary relevance vector ρqby greedy assignment, so that one
19

WorldMemArena: Evaluating Multimodal Agent Memory Through Action–World Interaction
retrieved item cannot earn credit for two golds:
ρq,k={1,r q,kmatches a gold not yet covered by ranks1∶k−1,
0,otherwise.(11)
The DCG aggregates this vector with a logarithmic rank discount,
DCG K(ρq)=ρ q,1+K
∑
k=2ρq,k
log2(k+1).(12)
The ideal DCG corresponds to all golds appearing as early as possible. Writing K∗=min(∣𝒱 q∣,K)for the
number of golds reachable in the top-K,
IDCG K=1+K∗
∑
k=21
log2(k+1).(13)
NDCG@Kis the ratio of the two, with the convention that QAs with no gold contribute0:
NDCG@K q=DCG K(ρq)
IDCG K(ρq)(0if∣𝒱 q∣=0).(14)
B.5. Per-question-type accuracy
Aggregate accuracy hides systematic strengths and weaknesses, so we also reportQA-Crestricted to QAs of
a single semantic type t. Each gold QA is annotated with one of eleven mutually exclusive types, grouped
along four skill axes summarised in Table 4.
For each axist, the cell value isQA-Ccomputed only over QAs of that type, averaged across instances that
contain at least one QA of type t. TheAvg.column is the unweighted mean of the eleven per-type values per
instance, which prevents types with more QAs from dominating the headline number.
C. Additional dataset details
C.1. Data sources
Our trajectories are sourced from four upstream agent benchmarks, including EmbodiedBench Yang et al.
(2025), VisualAgentBench Liu et al. (2024), the Agent-Arena GUI task collection Kadi and Terzić (2025),
together with an in-house long-horizon dialogue collection that we release alongside this benchmark.
C.2. Quality validation
Each generated session passes through automatic validators (memory point coverage, image caption coverage,
interference detectability, update chain consistency) before being assembled into the dataset. Samples failing
any validator are regenerated up to 3 times.
20

WorldMemArena: Evaluating Multimodal Agent Memory Through Action–World Interaction
Table4.The eleven semantic axes used for per-type QA accuracy. Each axis is a mutually exclusive label assigned to
every gold QA.
Group Abbr. Type What the question tests
BasicFRFact Recall Retrieve a single concrete fact stated earlier in the trajectory.
RobustnessDUDynamic Update The queried fact has been overwritten later; the answer
must reflect the latest version.
MBMemory Boundary The answer is not present in memory; the system must
abstain rather than fabricate.
MCMemory Conflict Two memory items disagree; the system must resolve the
conflict using context.
ReasoningTRTemporal Reasoning The answer requires reasoning about timing, ordering, or
duration of events.
KRKnowledge Reasoning The answer combines stored facts with general world knowl-
edge.
TTLTest-Time Learning The system must apply a rule or skill it was taught earlier in
the trajectory.
MultimodalVFRVisual Fact Recall The gold fact is anchored to a specific image in memory.
VSVisual Search The answer requires locating an object or attribute across
visual memory.
VUVisual Update A previously observed visual state has changed later in the
trajectory; the answer must reflect the most recent observa-
tion.
CMRCross-modal Reasoning The answer combines textual and visual memory.
C.3. Further Introduction to Dataset Domains
Lifelongevolution. LifelongEvolutioninstantiatesthelifelongdimensionofWorldMemArenathroughtwo
complementary domains, specified in the next two paragraphs. In both domains, experience arrives as an
orderedsequenceofsessions(forexample S00,S01, ...), andeachstagemayintroducenewobservationsthat
supersedefacts that previously held. Fine grained supervision comes from staged memory point annotations
(including update flags, importance, and, when applicable, superseded “original” memories). From these
we derive a cumulativegoldmemory state per session for analysis and scoring. Evaluation is interleaved
through qa_checkpoints tied to covered_sessions . The model is examined only after a stretch of
new experience, rather than by replaying the full chat log in a single prompt. The design targetsevolving
personal state(identity, relationships, and preferences revealed in S00and later turns) andevolving task
state(work outcomes, projects, constraints, and domain milestones) under temporal noise and interference.
This is not static persona QA on a single conversation.
Professional verticals domain.The first lifelong domain is organized into six professional verticals (for
example academic, software, health, finance, education, startup), with 18 samples in total. Each trajectory
foregrounds a long arc centered ontasks(research programs, product delivery, clinical or business work-
21

WorldMemArena: Evaluating Multimodal Agent Memory Through Action–World Interaction
flows). Professional artifacts and constraints shift over time. Checkpoint questions may anchor evidence in
multimodalreferences. Besides memory point identifiers, gold references may includeimageidentifiers
tied to per turn attachments, corresponding to documents, interfaces, or scene captures that accompany
narrated actions.
Holistic life course domain.The second lifelong domain adopts a holistic life course setting with 20
trajectories. Each trajectory explicitly separates main arc sessions (career and life goal progression) from side
arc sessions (daily life, family, health), with per session labels for arc role, event type, and whether the session
lies on the primary storyline. Gold QA evidence in this domain is recorded primarily as text memory point
identifiers, emphasizing narrative memory under rich personal context rather than professional domains
stratified by category. The two lifelong domains share the same data shape oriented towardevaluation
(ordered sessions, staged memory points, checkpoint QA), so one lifelong runner and gold state machinery
apply throughout Lifelong Evolution.
Agent domain.The Agent domain covers long horizonagent trajectoriesin WorldMemArena. At each step
the evaluated model receives an observation, internal reasoning, an executed action, environment feedback,
and optional screenshots from diverse simulated or instrumented settings (for example navigation, embodied
manipulation, and desktop GUI tasks). Here the Action World is explicit in the record. State changes are
governed by actions and feedback, not by conversational stance alone, and staged memory point annotations
track evolving quantities such as inventory, location, task phase, and failure or success signals. Probes and
post hoc questions therefore target whether memory captures how the environment changed across steps,
including updates and interference, rather than surface repetition of phrasing. In short, the Agent domain
instantiates the Action World Interaction Loop in its most direct form. The trajectory is already a time
ordered log of acting upon a world and reading consequences back.
Action World Interaction Loop versus pure long dialogue memory.We unify the two lifelong domains
and the Agent domain under an Action World Interaction Loop. In the professional verticals and life course
domains, dialogue between the user and the assistant is thesurface channel. Each session is anchored
toevents in a world(career moves, compliance deadlines, household logistics, health episodes, material
outcomes) that change what is true thereafter, with per turn attachments as observable traces of those events
(forms, screenshots, records). In the Agent domain, the same logic appears without mediation through
narration of a human life in natural language. Observations and screenshots are already traces of an acting
agent coupled to an environment. All three domains require integrating symbolic state evolution with visual
grounding where images appear, rather than only summarizing conversational tone or entity mentions. By
contrast, classical long dialogue benchmarks largely test recall cued bylexical overlapin extended chat. They
seldom commit to a jointly evolving external task state that can be superseded, or to staged interference and
multimodal evidence aligned with what actually happened outside the text channel. Under this loop, success
requires maintaining alatent world modelof consequences and updates across time. The evaluated model
must remember not onlywhat was said, but alsowhat became trueafter actions and outcomes accumulate in
a persistent situation.
22

WorldMemArena: Evaluating Multimodal Agent Memory Through Action–World Interaction
1 10 100 1000
Time per task (seconds)SimpleMem
Omni-SimpleMem
MIRIX
A-Mem
Qwen3-VL-Embedding-8B
AUGUSTUSMemory
ViLoMem
UniversalRAGMemory
MGMemory
M2A  786.3s
  572.6s
  316.6s
  282.2s
  165.2s
  151.0s
  122.8s
  51.0s
  43.8s
  10.0sRetrieval time
Write / Store time
Figure8.Mean per-task wall-clock time on a log-scale axis, split into retrieval (light blue) and write/store (dark blue).
D. Adapter interface
Every memory system implements the seven-method MemoryAdapter interface: reset,ingest_turn ,
end_session ,snapshot_memories ,export_memory_delta ,retrieve ,get_capabilities . This
unifies systems written in Python, hosted via local servers (Qdrant, Neo4j), or wrapped from external
repositories.
E. More Experiment
Latency profile of memory baselines.Figure 8 reports the mean per-task wall-clock time of each memory
method, split into retrieval and write/store phases. Total cost spans almost two orders of magnitude, from
M2A ( 10.0s) to SimpleMem ( 786.3s), and the split between the two phases differs substantially across
designs. Read-heavy methods such as SimpleMem and Omni-SimpleMem spend the bulk of their budget
re-scanning the dialogue at query time, whereas write-heavy methods such as MIRIX and A-Mem front-load
the cost during ingestion and then serve queries in milliseconds; MGMemory pushes this pattern to its limit
by indexing inline, so its write phase is effectively free ( ≈2ms). Write and retrieval time are therefore
largely independent design choices, and the cost frontier is occupied by methods that keepbothsmall (M2A,
MGMemory). In other words, latency on long-horizon traces is dominated by the memory strategy rather
than by raw backbone speed: choosing where to pay, ingestion or query, has a far larger impact than choosing
the LLM.
Retrieval on WorldMemArena.Table ??separates methods by paradigm. Dense RAG with Qwen3-VL-
Embedding-8B achieves the strongest ranking quality at largerK, with the highest NDCG@5 / NDCG@10,
indicating well-ordered candidate lists beyond the first hit. UniversalRAG lags on both recall-oriented metrics
and NDCG, suggesting weaker coverage or ranking on this split. Among agent-memory systems, MemoryGPT
reaches very high Recall@K and NDCG@1, on par with the Base Model rows where R@1=R@5=R@10 .
That saturation pattern is consistent with frequent early retrieval of the relevant unit, while graded relevance
within the top- Klist remains difficult: MemoryGPT does not surpass the embedding baseline on NDCG@5
/ NDCG@10 despite extreme recall. A-Mem occupies a different regime, with moderately high recall and
23

WorldMemArena: Evaluating Multimodal Agent Memory Through Action–World Interaction
second-best NDCG@5 / NDCG@10 among memory methods, which highlights a trade-off between hit rate
and graded relevance across memory designs.
Variance across memory architectures.Beyond the top rows, the agent-memory block exhibits large
spread. SimpleMem and MIRIX are substantially weaker, indicating that lightweight or misaligned memory
indexing fails this retrieval benchmark. Omni-SimpleMem and M2A recover much of the gap toward mid-tier
recall and NDCG, while ViLoMem remains weaker at small Kdespite improving at R@10. AUGUSTUS
tracks A-Mem on recall but does not translate into superior NDCG@ K, reinforcing that high recall alone is
insufficient when evaluation stresses ranking quality.
Backbone competence along 11 types of capabilities.Table ??clarifies how backbone choice can interact
with memory/RAG behavior. Deepseek V4 attains the highest average and leads Fact Recall, Memory
Boundary, Memory Conflict, and most multimodal axes (Visual Fact Recall, Visual Search, Visual Update),
suggesting stronger grounding and visual evidence use under the benchmark definitions. GPT 5.4-mini
is second on average with peaks on Temporal Reasoning, Knowledge Reasoning, Test-Time Learning, and
Cross-modal Reasoning, but it collapses on Memory Boundary, indicating reasoning-centric strength paired
with weak explicit boundary control. Claude Haiku 4.5 excels on Dynamic Update and Test-Time Learning
while suffering on multimodal retrieval scores (Visual Search in particular). Gemini 3 Flash and Qwen3.5 Plus
are more balanced but below the top two on average, with Gemini especially weak on Temporal Reasoning
and Visual Search. Together, the two tables support a systems-level reading: leaderboard differences at
retrieval time reflect both pipeline design and the backbone’s axis-wise strengths, especially when multimodal
alignment or boundary-sensitive memory behavior is required.
F. More Analysis
Due to space limitations, the main text cannot provide a detailed analysis. Here, we provide an extended
analysis following RQ1–RQ6 in the main paper.
❖Long-horizon collapse.Lifecycle failures compound into long-horizon memory collapse.Lifecycle
failures compound as trajectories become longer. Early omissions in writing reduce the evidence available to
later retrieval. Retrieval failures then prevent the model from grounding later answers, and incorrect answers
may further pollute subsequent memory updates. This creates a snowball effect in which later-session
reasoning questions become increasingly difficult, even when the required evidence was present earlier in
the trajectory. The degradation is particularly severe for reconstructed agentic worlds, where the system
must remember not only explicit statements, but also actions, tool outcomes, visual states, and causal
consequences.
❖Agentic trajectories expose domain brittleness.Reconstructed agentic worlds remain challenging for
most systems because therelevant evidence is distributed across dense action sequences, tool feedback,GUI
states, screenshots, and environment transitions. Many existing systemsrely on assumptions that work well
for static conversations or document-likehistories, but break down when memory must be extracted from
interactiveexperience. In agentic trajectories, the system must decide which actionsmattered, which failures
should be remembered, which object states changed,and which tool outcomes should guide future behavior.
As a result, systemsthat perform well on prior memory benchmarks may show a sharp forgetting curveunder
more interactive and causally dense settings.
24

WorldMemArena: Evaluating Multimodal Agent Memory Through Action–World Interaction
❖Retrieval is limited by precision and text bias.Human-designed memory systems show a clear trade-off
between recall and precision. Increasing the retrieval budget can improve the chance of including relevant
evidence, but it does not necessarily improve final answers. Larger retrieved contexts may introduce outdated,
conflicting, or irrelevant memories, making it harder for the model to identify the correct evidence. This
indicates that retrieval quality cannot be reduced to retrieving more items; effective memory systems require
query-aware selection, evidence ranking, and conflict filtering. This limitation is even more pronounced
in multimodal tasks. Many systems store images or screenshots at a surface level, but retrieval still relies
heavily on text proxies such as captions, OCR, or generated summaries. As a result, visual evidence is only
usable if it was correctly textualized during writing. Current multimodal memory therefore remains largely
text-centric, highlighting the need to preserve visual evidence as first-class information rather than reducing
it to incomplete textual descriptions.
❖Past experience is not automatically reusable.A key goal of agent memory is not only to answer
questions about the past, but also to improve future behavior. Our results suggest that this ability remains
limited. Systems can often repeat explicit facts from earlier sessions, but they struggle to convert past
experiences into action-guiding knowledge. This is most evident in reasoning and test-time learning tasks,
where the system must infer a reusable rule, remember a previous failure, or adapt its future decision based
on earlier feedback. In other words, current memory systems are better at recalling past information than at
turning that information into future decisions.
❖Experiential evidence is fragile.Qualitative cases show that tool feedback, failed actions, visual details,
and implicit causal lessons are among the easiest information types to lose. In contrast, explicit textual facts
are much easier to write and retrieve. This asymmetry creates a gap between factual memory and agentic
memory: a system may remember what a user said, while failing to remember what happened when it acted,
why an attempt failed, or which strategy succeeded. The same issue also appears in long dialogue, where
systems often preserve local facts but fail to consolidate them into higher-level user models or stable cognitive
states. These findings suggest that action-oriented memory requires more than storage and retrieval; it
requires transforming experience into reusable policies, constraints, feedback patterns, and decision priors.
❖Limitsofhuman-designedmemorysystems.Human-designed memory systems provide useful structure,
but they also impose fixed assumptions about what should be stored, how memory should be organized,
and how retrieval should operate. These assumptions can work well in narrow settings, yet become limiting
in agentic environments where useful memory depends on the task, tool feedback, visual state, and future
action needs. The main weakness is not only lower absolute performance, but also reduced adaptability:
a memory pipeline tuned for one domain may not know how to reorganize itself when the environment
changes.
25