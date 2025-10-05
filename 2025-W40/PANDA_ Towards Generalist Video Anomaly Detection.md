# PANDA: Towards Generalist Video Anomaly Detection via Agentic AI Engineer

**Authors**: Zhiwei Yang, Chen Gao, Mike Zheng Shou

**Published**: 2025-09-30 15:19:43

**PDF URL**: [http://arxiv.org/pdf/2509.26386v1](http://arxiv.org/pdf/2509.26386v1)

## Abstract
Video anomaly detection (VAD) is a critical yet challenging task due to the
complex and diverse nature of real-world scenarios. Previous methods typically
rely on domain-specific training data and manual adjustments when applying to
new scenarios and unseen anomaly types, suffering from high labor costs and
limited generalization. Therefore, we aim to achieve generalist VAD, i.e.,
automatically handle any scene and any anomaly types without training data or
human involvement. In this work, we propose PANDA, an agentic AI engineer based
on MLLMs. Specifically, we achieve PANDA by comprehensively devising four key
capabilities: (1) self-adaptive scene-aware strategy planning, (2) goal-driven
heuristic reasoning, (3) tool-augmented self-reflection, and (4) self-improving
chain-of-memory. Concretely, we develop a self-adaptive scene-aware RAG
mechanism, enabling PANDA to retrieve anomaly-specific knowledge for anomaly
detection strategy planning. Next, we introduce a latent anomaly-guided
heuristic prompt strategy to enhance reasoning precision. Furthermore, PANDA
employs a progressive reflection mechanism alongside a suite of context-aware
tools to iteratively refine decision-making in complex scenarios. Finally, a
chain-of-memory mechanism enables PANDA to leverage historical experiences for
continual performance improvement. Extensive experiments demonstrate that PANDA
achieves state-of-the-art performance in multi-scenario, open-set, and complex
scenario settings without training and manual involvement, validating its
generalizable and robust anomaly detection capability. Code is released at
https://github.com/showlab/PANDA.

## Full Text


<!-- PDF content starts -->

PANDA: Towards Generalist Video Anomaly
Detection via Agentic AI Engineer
Zhiwei Yang1,2Chen Gao2†Mike Zheng Shou2†
1Xidian University2Show Lab, National University of Singapore
Abstract
Video anomaly detection (V AD) is a critical yet challenging task due to the com-
plex and diverse nature of real-world scenarios. Previous methods typically rely
on domain-specific training data and manual adjustments when applying to new
scenarios and unseen anomaly types, suffering from high labor costs and limited
generalization. Therefore, we aim to achieve generalist V AD,i.e., automatically
handle any scene and any anomaly types without training data or human involve-
ment. In this work, we propose PANDA, an agentic AI engineer based on MLLMs.
Specifically, we achieve PANDA by comprehensively devising four key capabilities:
(1) self-adaptive scene-aware strategy planning, (2) goal-driven heuristic reasoning,
(3) tool-augmented self-reflection, and (4) self-improving chain-of-memory. Con-
cretely, we develop a self-adaptive scene-aware RAG mechanism, enabling PANDA
to retrieve anomaly-specific knowledge for anomaly detection strategy planning.
Next, we introduce a latent anomaly-guided heuristic prompt strategy to enhance
reasoning precision. Furthermore, PANDA employs a progressive reflection mech-
anism alongside a suite of context-aware tools to iteratively refine decision-making
in complex scenarios. Finally, a chain-of-memory mechanism enables PANDA to
leverage historical experiences for continual performance improvement. Exten-
sive experiments demonstrate that PANDA achieves state-of-the-art performance
in multi-scenario, open-set, and complex scenario settings without training and
manual involvement, validating its generalizable and robust anomaly detection
capability. Code is released at https://github.com/showlab/PANDA.
1 Introduction
Video anomaly detection (V AD) [ 1,2,3,4] aims to identify abnormal or suspicious events in
video streams, playing a vital role in a wide range of real-world applications such as intelligent
surveillance [5], traffic monitoring [6], autonomous driving [7], and industrial safety [2].
Existing V AD methods follow a specialist-oriented paradigm and require manual participation when
deploying for new scenarios and anomalies. Broadly, they can be categorized into: training-dependent
and training-free (Fig. 1(a)). Specifically, training-dependent methods rely on newly annotated data
to train models for each target scenario. The manual and training costs make such methods lack
generalization and versatility. Besides, training-free methods typically employ pre-trained large
language models (LLMs) or vision-language models (VLMs) as the backbone, thereby eliminating the
need for model training. However, they still depend heavily on manual engineering when deploying
for new scenarios and anomalies, such as scenario-specific preprocessing steps, handcrafted prompt
templates, rule curation, and post-processing. These static pipelines still lack adaptivity, making them
brittle when confronted with uncertainty, long-term temporal dependencies, or complex, dynamic
scenarios. Moreover, the hand-crafted nature restricts them from towarding generalist V AD.
†Corresponding authors.
39th Conference on Neural Information Processing Systems (NeurIPS 2025).arXiv:2509.26386v1  [cs.CV]  30 Sep 2025

User query
Self-improving 
chain-of-memoryTool-augmented 
self-reflection
Goal-driven 
heuristic reasoningSelf-adaptive 
scene-aware planning
(b) Our generalist PANDA (a) Existing specialized methodsTraining-free VADTask-oriented 
Manual Engineering 
Pre-processing  
LLM or VLM  
Task-oriented 
Manual Engineering
Post-processing  
VAD Model 
Training domain 
scenariosNon-training 
domain scenarios
�
RetrainingAnomaly 
knowledge base
Agentic 
AI Engineer
Training-free
 Manual-free
 Generalist
Any Scenarios 
. . .
Any Anomaly Categories 
. . .
Selection
Labeling
Video Training datasetManual Data Engineering  
Training-dependent VADFigure 1:PANDA vs. Existing specialized methods.When facing arbitrary scenes and anomalies,
PANDA can automatically adapt without the need for training refinements and manual adjustments,
still achieving superior performance.
To overcome the limitations of existing methods and free ourselves from the burden of domain-specific
training and handcrafted pipeline design, our vision is to develop a general-purpose video anomaly
detection method that can be self-adaptive to new scenarios and novel anomaly types without requiring
any training data or manual pipeline assembly. The recent success of Multimodal Large Language
Models (MLLMs) in a wide range of visual understanding tasks offers a promising foundation
for realizing this vision. Therefore, we introduce PANDA, an agentic AI engineer for generalized
V AD (Fig. 1(b)). Drawing inspiration from how human engineers systematically analyze problems,
adapt to complex environments, and iteratively improve through tool use and experience, PANDA
adaptively perceives the enviroment based on user-defined requirements, formulates detecion plans,
perform goal-driven reasoning, invokes external tools to enhance decsion making, and continuously
accumulates expeirence in memory for self-improvement.
Technically, the proposed PANDA is distinguished by the following aspects.(1) self-adaptive scene-
aware strategy planning.Faced with a new scene or user-defined anomaly detection requirements,
PANDA first conducts environment perception and understanding, then retrieves relevant anomaly
rules from an anomaly knowledge database. Based on the environment context information, a
scene-adaptive Retrieval Augmented Generation (RAG) mechanism is designed to construct tailored
anomaly detection plans.(2) Goal-driven heuristic reasoning.PANDA injects task-specific prompts
guided by latent anomaly cues, which steer the reasoning process toward more accurate and focused
decision-making.(3) Tool-augmented self-reflection.PANDA iteratively assesses uncertainty and
activates a suite of curated tools, such as object detection, image retrieve, or web search, to acquire
additional information and resolve ambiguous decision-making.(4) Self-improving chain-of-memory.
PANDA integrates historical experiences to justify current reasoning decisions or self-reflection. By
progressively accumulating contextual cues across temporal spans, it enhances both the stability and
accuracy of its decisions over time. Taken together, PANDA embodies an agentic AI engineer that
proactively perceives diverse environments, formulates adaptive strategies, performs goal-driven
reasoning, and progressively improves through tool-augmented reflection and a chain-of-memory
mechanism, enabling robust and generalizable video anomaly detection across complex real-world
scenarios.
Extensive experiments across multiple challenging benchmarks show that PANDA achieves state-
of-the-art performance in multi-scenario, open-set, and complex scenario settings,without training
and manual involvement. These results highlight PANDA’s strong potential as an autonomous and
general-purpose solution for real-world V AD.
2

2 Related Work
Video anomaly detection (V AD) [ 8,9,10,11], has long been a critical research topic in the computer
vision field due to its wide range of real-world applications. Existing V AD methods are specialist-
oriented and can be broadly categorized into training-dependent and training-free approaches.
Training-dependent V AD.These approaches rely on varying levels of annotated data and typically
fall into three categories: semi-supervised V AD [ 2,5,10,12,13], weakly-supervised V AD [ 14,15,
16,17,18], and instruction-tuned V AD [ 19,20,21,22]. For example, Ristea et al. [ 23] proposed
an efficient anomaly detection model based on a lightweight masked autoencoder. Yang et al. [ 14]
introduced a text prompt-driven pseudo-labeling and self-training framework for weakly-supervised
V AD. Zhang et al. [ 22] presented a model combining an anomaly-focused temporal sampler with an
instruction-tuned MLLM to detect anomalies. While these training-dependent methods often perform
well within the domain of the training data, they typically suffer from sharp performance degradation
when deployed in out-of-distribution environments or faced with novel anomaly types. This limits
their applicability in the open-world scenarios where anomalies are diverse, and context-sensitive.
Training-free V AD.Inspired by the recent success of LLMs [ 24,25] and VLMs [ 26,27,28],
training-free V AD methods [ 29,30] have gained increasing attention. These approaches aim to
leverage the powerful prior knowledge embedded in foundation models without requiring domain-
specific training. For instance, Zanella et al. [ 29] proposed the first language-model-based training-
free V AD framework, which improves anomaly scoring by aligning cross-modal features between
LLMs and VLMs while suppressing noisy captions. Yang et al. [ 30] developed a rule-based anomaly
inference framework by prompting LLMs to perform inductive and deductive reasoning over anomaly
rules. Despite removing the need for training, these methods often rely on static prompting patterns
and require substantial manual engineering (e.g., handcrafted pre/post-processing), which limits their
adaptivity and robustness in complex, real-world scenarios.
Distinct from both paradigms above, PANDA is an agent-based framework that embodies the
characteristics of an agentic AI engineer, which is capable of autonomously performing V AD without
training and manual engineering when faced with various real-world scenarios. By incorporating a
progressive reflection mechanism and a suite of perception-enhancing tools, PANDA can adaptively
refine its predictions through self-reflection and tool invocation. This enables PANDA to dynamically
handle diverse and challenging scenarios in the real world.
3 Method
In this section, we present the core architecture and reasoning process of PANDA, an agentic AI
engineer for generalized V AD. PANDA is designed to dynamically perceive diverse environments and
perform progressive, tool-enhanced reasoning and self-refinement, as shown in Fig. 2. It achieves this
through four synergistic modules: (1) self-adaptive scene-aware strategy planning, (2) goal-driven
heuristic reasoning, (3) tool-augmented self-reflection, and (4) self-improving chain-of-memory.
3.1 Self-adaptive Scene-aware Strategy Planning
To achieve V AD in general and unconstrained environments, it is essential to dynamically perceive the
current video context and construct targeted detection strategies. Given the scene-dependence of many
real-world anomalies and the variability in visual conditions, PANDA first performs self-adaptive
perception of the input video to extract high-level environment contextual information.
Environmental Perception.Given a user-defined detection query User query and an input video se-
quence V={f 1, f2, . . . , f N}containing Nframes, PANDA uniformly samples Mkeyframes
F={f 1, f2, . . . , f M}and constructs a perception prompt combining Fwith the User query.
This prompt is fed to a VLM, which returns structured environmental information EnvInfo
= {Scene Overview ,Potential Anomalies ,Weather Condition ,Video Quality }. Here,
Scene Overview provides a high-level summary of the scene, including location type (e.g., street,
shop, parking lot) and observed activities. Potential Anomalies refers to types of suspicious
behaviors that may plausibly occur in the current scene context. Weather Condition captures at-
tributes such as time of day (day/night) and weather (e.g., sunny, rainy). Video Quality summarizes
resolution and clarity (e.g., low-resolution, blurred, noisy).
3

Please help me detect the following types of abnormal events: Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, Road Accidents, 
Robbery, Shooting, Shoplifting, Stealing, Vandalism.
Anomaly Knowledge
Event Type
Anomaly Rules
Applicable Scenes
VLM Perception
 MLLM Planning
Environmental Information
Scene Overview
•The video depicts a  store with... 
Weather Condition
•Daytime, bright light
Video Quality
•The overall quality of the video is low and blurry.Detection Strategy Planning
Preprocessing
•Image Deblur
Anomaly Rules
•Companion distracts clerk while theft ...
Heuristic Prompt
•Observe customer interactions with ...
MLLM Reflection
Self-Reflection 
Reason Analysis
•Need for further enhancement 
of image details and semantic 
information.
Tools to use
•Image Super-resolution
•Object Detection
MLLM Replanning
Detection Strategy Update
New Anomaly Rules
•Customers repeatedly pick 
up items and look around...
New Heuristic Prompt
•Focusing on customer 
behavior in the store...
Tool InvokeToolkit 
Image Super-resolution
Image Denoise
Object Detection
Web Search
Image Retrive
. . .Brightness Enhance
Image Deblur
Image ZoomReasoning Results
Status: Normal / Abnormal / Insufficient
Score: 0.5
Reason: The video shows two individuals 
browsing items in a store. The scene 
lacks sufficient detail to confidently 
classify it as normal or abnormal. 
MLLM Reflection
Self-Reflection 
Reason Analysis
•More information is needed 
to determine if the woman's 
behavior... 
Tools to use
•Image Retrive
•Web Search
MLLM Replanning
Detection Strategy Update
New Anomaly Rules
•Customers taking items 
without paying...
New Heuristic Prompt
•Focusing on the woman's 
interaction behavior with the 
items...
Tool Invoke
Image RetriveVLM Reasoning
Refined Reasoning Results
Status: Normal / Abnormal / Insufficient
Score: 0.6
Reason: The woman is browsing the 
items with suspicious behavior, but 
based on the current clip it is not clear if 
there is a shoplifting anomaly.
VLM Reasoning
Refined Reasoning Results
Status: Normal / Abnormal / Insufficient
Score: 0.9
Reason: The woman carrying the bag on 
her back without paying, which is a 
typical shoplifting event.
Object Detection
Retrive query: A woman is suspiciously 
browsing items in the store.
Web Search
Search query: What are 
typical shoplifting 
behaviors in a store?Image Super-resolution
...
Potential Anomalies
•Burglary
•Shoplifting
•StealingRAG
VLM Reasoning
1. Customers put 
unpaid items in 
their bags...
2. Customers...Recalling past 
experience ...
First round of reflection
Second round of reflectionRecalling past 
experience ...Short
CoM
Long 
CoMFigure 2:Overview of the proposed PANDA.As shown in the figure, upon receiving a user-
defined query, PANDA first performs environment perception and plans a scene-adaptive detection
strategy. PANDA then executes the plan with goal-driven heuristic reasoning. When encountering
ambiguous cases, PANDA enters a reflection phase, revising its plan and invoking external tools
to refine the decision. Throughout the process, PANDA maintains both short-term and long-term
Chain-of-Memory (CoM), enabling it to accumulate experience and continually improve over time.
RAG-Based Strategy Planning.With the structured environment context in hand, PANDA pro-
ceeds to plan its detection strategy. To avoid hallucinations and improve reliability, this planning is
performed via retrieval-augmented generation (RAG) [ 31], driven by a multimodal large language
model (MLLM). First, based on User query, PANDA constructs a knowledge base prompt Promptknow
and then generates a structured general anomaly knowledge base using the MLLM:
κa=MLLM(User query,Promptknow) ={Event Type,Anomaly Rules,Application Scenes}.
(1)
Here, Event Type indicates anomaly categories specified by the user. Anomaly Rules are detection
rules associated with each anomaly type. Application Scenes are contextual environments where
anomalies are likely to occur.
For each anomaly type, we predefine Hrule-scene pairs to form the knowledge base. PANDA then
uses the perceivedEnvInfoas a query to retrieve the top-kmost relevant anomaly rules:
Rules a=RetrieveTopK(EnvInfo, κ a).(2)
Finally, PANDA integrates the User query,EnvInfo , and Rules ato construct a planning prompt
Promptplan, which is passed to the MLLM to generate the detection strategy plan:
Plan strategy =MLLM(Promptplan)
={Preprocessing,Potential Anomalies,
Heuristic Prompt}.(3)
Here,Preprocessingspecifies optional visual enhancement steps (e.g., brightness adjustment, de-
noising, super-resolution). Potential Anomalies refines the anomaly list based on rule relevance
and scene understanding. Heuristic Prompt includes step-by-step reasoning instructions for each
potential anomaly, enabling the downstream inference module to perform structured, chain-of-thought
analysis.
4

By integrating self-adaptive environment perception and RAG-enhanced strategy planning, PANDA
ensures that subsequent anomaly reasoning is goal-driven and context-aware, significantly improving
robustness in open-world settings.
3.2 Goal-Driven Heuristic Reasoning
The reasoning module serves as the core component of PANDA for analyzing video anomaly events.
PANDA supports both offline and online inference modes. In this section, we focus on the offline
setting, while the implementation details section will describes the online mode settings.
Under the guidance of the detection strategy plan constructed in subsection 3.1, PANDA performs
goal-driven heuristic reasoning using a VLM. Given the User query, a clip-level video segment
Vclip={c 1, c2, . . . , c T}, here each video clip cicontains svideo frames, and the Plan strategy ,
PANDA first applies the preprocessing tools specified in Plan strategy to obtain an enhanced video
clip:
eVclip=Preprocessing(V clip) ={ec 1,ec2, . . . ,ec T}.(4)
Next, PANDA constructs a reasoning prompt based on thePlan strategy :
Prompt reasoning = {Memoryl-steps
text , Potential Anomalies, Anomaly Detection Rules,
Heuristic Prompt, Enhancement and Reflection Info}.
The fields Potential Anomalies ,Anomaly Detection Rules , and Heuristic Prompt are
directly inherited from the planning stage. The Enhancement and Reflection Info field incor-
porates information produced during the self-reflection stage (To be described in subsection 3.3),
including tool-based refinements and updated anomaly rules and heuristic prompts. To enhance
temporal awareness, PANDA equips a short-term memory component Memoryl-steps
text , which records
the past lreasoning steps as textual memory. In addition to textual memory, PANDA also maintains a
corresponding visual memory stream Memoryl-steps
visual , which stores visual frames aligned with the latest
lsteps, allowing the model to access fine-grained visual cues during inference.
Finally, driven by the potential anomaly targets and enriched contextual knowledge, PANDA performs
heuristic reasoning with the following formulation:
Result reasoning =VLM(ec i,Memoryl-steps
visual,Promptreasoning )
={Status:Normal/Abnormal/Insufficient,
Score∈[0,1],Reason:⟨·⟩}.(5)
Here, Status indicates the result of the VLM judgment: Normal indicates the clip is confidently
classified as non-anomalous, Abnormal denotes strong evidence of anomaly, and Insufficient
suggests the current information is inadequate to make a definitive judgment. Score is the probability
of the existence of an abnormal event for the clip corresponding to each status. Reason is the reason
for the status judgment given by the VLM. When the result is Insufficient , PANDA will trigger
the reflection mechanism to gather additional context or observations before re-entering the reasoning
loop.
3.3 Tool-Augmented Self-Reflection
In complex scenarios, PANDA may not be able to make a clear decision on whether a video segment
is normal or abnormal. In such ambiguous cases, it returns an Insufficient status, which triggers
the reflection module. PANDA adopts a tool-augmented self-reflection mechanism enhanced by
a specialized set of tools τ={tool 1,tool 2, . . . ,tool n}for visual content enhancement and
auxiliary analysis, including image deblurring, denoising, brightness enhancement, image retrieval,
object detection, and web search, etc. These tools assist in gathering additional evidence to support
the decision-making process.
Experience-Driven Reflection.Given an Insufficient Reason from the current reasoning step,
PANDA first queries its long chain-of-memory ( Long-CoM , will be introduced in 3.4) to retrieve the
most similar history reflection cases:
Experiencereflection =RetrieveTop1(Insufficient Reason,Long-CoM).(6)
5

... ...
Long CoM
Short CoM
Step t-1:  Visual/Status/Score/Reason
Step t-2:  Visual/Status/Score/Reason
Step t-3:  Visual/Status/Score/Reason
. . .Step 1:  Reasoning Results
•Status / Score / Reason
Step 2:  Reflection Results
•insufficient reason
•Reflection Plan
Tools Used
New Anomaly Rule
New Heuristic Prompt
. . .
Retrieve History Experience
updateNew Detection 
Strategy Planning 
. . .
Self-Reflection ...
Insufficient Normal or Abnormal
New ExperienceStep 3:  Re-reasoning Results
•Status / Score / Reason
Figure 3: Illustration of Self-Improving Chain-of-Memory.
PANDA then constructs a reflection prompt using video context information, including the User query,
EnvInfo ,Plan strategy ,Rules a, short chain-of-memory (short CoM), Insufficient Reason , and
Experiencereflection :Promptreflection ={User query,EnvInfo,Plan strategy ,Rules a,short CoM,
Insufficient Reason,Experiencereflection}.This prompt is fed into the MLLM to analyze the
cause of uncertainty and recommend an appropriate reflection plan:
Result reflection =MLLM(Promptreflection )
={"Insufficient Reason":⟨·⟩,
"Tools to Use": [{tool 1,params}, . . . ,{tool n,params}],
"New Anomaly Rule":⟨·⟩,
"New Heuristic Prompt":⟨·⟩}.(7)
Tool Invocation.PANDA executes the tool functions suggested in the reflection result to enhance
both visual and semantic information. The tool invocation process is formulated as:
Result tool_augmented =ToolInvoke(tool 1, . . . ,tool n)
={Text Enhancement Info,
Visual Enhancement Info=bc j∪ {c k}}.(8)
Here, text enhancement information includes summaries from tool outputs (e.g., detected objects,
web search results), while visual enhancement information includes processed video clip bcjand
retrieved historical keyframes{c k}.
Refined Reasoning.PANDA updates the reasoning prompt with the newly acquired textual cues:
Promptrefined
reasoning =Promptreasoning ∪{Text Enhancement,New Anomaly Rule,New Heuristic Prompt},
and re-reasoning the enhanced video clip input:
Resultreflection
reasoning =VLM(bc j∪ {c k},Memoryl-steps
visual,Promptrefined
reasoning ).(9)
If the returned status is Normal orAbnormal , PANDA resumes reasoning at the next timestep. If
the status remains Insufficient , reflection is re-triggered. To prevent infinite loops, we limit the
number of reflection rounds to r. If after rrounds the result is still Insufficient , PANDA assigns
a default anomaly score corresponding to the "Insufficient" status and skips the current segment and
continues next timestep.
3.4 Self-Improving Chain-of-Memory
To enable PANDA to become increasingly "smarter" over time by accumulating experience through
the iterative cycle of reasoning, reflection, and refined reasoning, PANDA equips a self-improving
chain-of-memory (CoM) mechanism as shown in Fig. 3. This mechanism enhances both long-term
context awareness and consistency in decision-making across video sequences. The CoM comprises
two components: short chain-of-memory (short CoM) and long chain-of-memory (long CoM).
6

Short CoM.In the reasoning stage, short CoM includes both the textual reasoning trace
Memoryl-steps
text and its visual counterpart Memoryl-steps
visual , as described in subsection 3.2. In the re-
flection stage, short CoM is represented by the set of past reflection outputs: Resulthistory
reflection =
{Result1
reflection ,Result2
reflection , . . . ,Resultl
reflection}.
Long CoM.PANDA also maintains a temporally evolving long CoM: Long-CoM=
{M1, M2, . . . , M T},where each memory unit Mtat time step tencapsulates three key outputs:
Mt={Result reasoning ,Result reflection ,Resultrefined
reasoning }.This structure ensures that PANDA retains
a complete trace of all decision stages—initial reasoning, reflective analysis, and post-reflection
decisions. At the start of a video, LongCoM is empty by design, and PANDA relies on ShortCoM’s
local window memory for initial reasoning and reflection. As more clips are processed, LongCoM
gradually accumulates traces, supporting memory-consistent reasoning and reflection planning. With
this self-improving chain-of-memory, PANDA leverages accumulated historical experience to inform
both reasoning and reflection, leading to progressively more stable and accurate anomaly detection
over time.
4 Experiments
4.1 Experiment Setup
Datasets.We evaluate PANDA on four benchmarks: UCF-Crime [ 17], XD-Violence [ 16], UBnor-
mal [ 32], and CSAD, which represent three distinct settings—multi-scenario, open-set, and complex
scenario.UCF-Crimeis a large-scale dataset comprising 1,900 long, untrimmed real-world surveil-
lance videos. It covers 13 types of abnormal events such as fighting, abuse, stealing, arson, robbery,
and traffic accidents. The training set includes 800 normal and 810 abnormal videos, while the test set
consists of 150 normal and 140 abnormal videos.XD-Violenceis another large-scale dataset focused
on violence detection. It contains 4,754 videos collected from surveillance video, movies, and CCTV
sources, encompassing 6 categories of anomaly events. The training and test sets include 3,954 and
800 videos, respectively.UBnormalis a synthetic open-set video anomaly detection dataset with a
total of 543 videos. It defines 7 categories of normal events and 22 types of anomalies. Notably, 12
anomaly categories in the test set are unseen during training, making it a challenging benchmark for
evaluating generalization under open-set conditions.CSADis a complex-scene anomaly detection
benchmark constructed in this work. It consists of 100 videos (50 normal and 50 abnormal), sampled
from UCF-Crime, XD-Violence, and UBnormal. CSAD includes videos with challenging conditions
such as low resolution, poor illumination, high noise levels, and long-range temporal anomalies. It is
designed to assess model robustness in complex and degraded environments.
Evaluation Metrics.Following the previous methods [ 16,17], we report the Area Under the Curve
(AUC) of the frame-level receiver operating characteristic for UCF-Crime, UBnormal, and CSAD.
For XD-Violence, we follow the evaluation criterion of average precision (AP) suggested by the
work [16] to measure the effectiveness of our method.
Implementation Details.We adopt Langgraph [ 33] to build the whole agent framework and all
experiments are implemented using PyTorch [ 34] on the A6000 GPU. We use Qwen2.5VL-7B [ 26] as
the VLM for perception and reasoning stages, and Gemini 2.0 Flash [ 25] as the MLLM for planning
and reflection. To improve the inference efficiency, the input video is 1FPS sampled, and one video
clip containing s= 5 frames is inferred at each time step. PANDA supports both offline and online
reasoning modes. In offline reasoning mode, the perception phase is sampling M= 300 frames
uniformly for the whole video, while only the initial M= 10 frames are sampled in online mode.
The number of knowledge entries for each type of anomalous event in the anomaly knowledge base
isH= 20 . The maximum number of reflection rounds ris set to 3. The short CoM length l= 5
during the reasoning stage. We retrieve the top k= 5 anomaly rules from the anomaly knowledge
base for each user query. More implementation details, including prompt templates and tool usage,
are provided in the supplementary material.
7

Table 1: Comparisons with previous state-of-the-art methods on different datasets. "Expl." stands
for "Explanation", indicating whether the output results include interpretations of the detected
anomalies. Methods categorized as "Semi", "Weak", or "Instru-Tuned" require training data to adapt
to specific scenarios or anomaly types. As a result, their performance degrades significantly when
applied directly to unseen open-set scenarios. On the other hand, training-free specialized methods
rely heavily on manual engineering to accommodate new scenes or anomalies, which limits their
generalization capability.
Multi-Scenario Open-Set Complex ScenarioMethods Supervision Expl. Manual-free Mode
UCF (AUC%) XD (AP%) UB (AUC%) CSAD (AUC%)
Specialized methods
AED-MAE[23][CVPR2024] Semi ✗ ✗ Offline - - 58.50 -
STPAG[35][CVPR2024] Semi ✗ ✗ Offline - - 57.98 -
HL-Net[16][ECCV2020] Weak ✗ ✗ Offline 82.44 73.67 - -
RTFM[36][ICCV2021] Weak ✗ ✗ Offline 84.30 77.81 64.94 -
UR-DMU[37][AAAI2023] Weak ✗ ✗ Offline 86.97 81.66 59.91 -
VadCLIP[15][AAAI2024] Weak ✗ ✗ Offline 88.02 84.51 - -
TPWNG[14][CVPR2024] Weak ✗ ✗ Offline 87.79 83.68 - -
VERA[38][CVPR2025] Weak ✓ ✗ Offline 86.55 70.11 - 64.52
Holmes-V AU[22][CVPR2025] Instru-Tuned ✓ ✗ Offline 88.96 87.68 56.77 72.47
ZS CLIP[26][ICML2021] Training-free ✓ ✗ Offline 53.16 17.83 46.2 32.45
LLA V A-1.5[26][CVPR2024] Training-free ✓ ✗ Offline 72.84 50.26 53.71 47.78
LA V AD [29][CVPR2024] Training-free ✓ ✗ Offline 80.28 62.01 64.23 57.26
AnomalyRuler [30][ECCV2024] Training-free ✓ ✗ Offline - - 71.90 -
Generalized method
PANDA (ours) Training-free ✓ ✓Offline 84.89 70.16 75.78 73.12
Online 82.57 63.57 72.41 71.25
Table 2: We performed key hyperparameter analyses on the UCF-Crime dataset.
(a) Analysis of reflection roundr.
Reflection UCF-Crime
Roundr (AUC%)
1 83.83
3 84.89
5 84.91(b) Analysis of rules numberk.
Rules UCF-Crime
Numberk (AUC%)
1 82.79
5 84.89
9 83.92(c) Analysis of short CoM lengthl.
CoM UCF-Crime
lengthl (AUC%)
1 82.92
5 84.89
9 84.03
4.2 Comparison with State-of-the-Art Methods
Table 1 compares the performance of PANDA against state-of-the-art specialized V AD methods,
including both training-dependent and training-free methods. As shown, PANDA significantly
outperforms all existing training-free baselines across all four datasets, even under online settings.
On the UBnormal dataset, which adopts an open-set evaluation protocol where test-time anomalies
are unseen during training, PANDA surpasses both training-dependent and training-free approaches.
This highlights PANDA’s strong generalization capabilities.
PANDA also exhibits notable advantages on CSAD, the complex-scene benchmark introduced in this
work, where traditional methods tend to fail under low-quality or temporally extended anomalies.
PANDA’s superior performance across diverse datasets and conditions demonstrates its robustness
and effectiveness as a general-purpose solution for real-world video anomaly detection.
4.3 Analytic Results
Analysis of reflection round r.Table 2(a) shows the effect of varying the number of reflection
rounds ron PANDA’s performance. We observe that performance improves gradually when increasing
rfrom 1 to 5. Although r= 5 yields a slight additional improvement compared to r= 3 , it
introduces more computational overhead due to repeated tool invocation and reasoning steps. To
balance efficiency and effectiveness, we adoptr= 3as the default setting in all experiments.
8

...Video Time 00:05  Reasoning Results: {''score'': 0.0, ''status'': ''normal'', ''reason'': ''The scene shows a quiet 
store counter with no visible signs of abnormal activity. There are no people present, and the environment appears 
undisturbed.''}
Video Time 02:25 Reasoning Results: {score'': 0.2, ''status'': ''insufficient'', ''reason'': ''The video shows a static 
view of a store counter with no significant actions or interactions occurring. There is no clear indication of 
abnormal behavior based on the provided rules and prompts.''}
Reflection Results: { "reason": "The primary reason for insufficient information is the lack of activity in the video. 
The VLM is observing a static scene with no discernible events. While the timestamp might suggest early morning 
and low activity, it doesn''t definitively rule out potential subtle anomalies that the low video quality might obscure. 
The VLM also appears to be overly reliant on immediate action, potentially missing precursory behavior that could 
indicate an impending anomaly. Also, the current rules do not account for pre-crime action analysis.",
"tools_to_use": [{ "tool_name": "Image Retrieve", "query": "Similar time of day, static view of store counter, 
identify any instances of suspicious behavior before an anomaly occurs."}],
"new_anomaly_rule": "Security camera shows a still scene in a normally operating business during off-peak 
hours. Requires a check of entrance to the store for loitering for longer than five minutes.",
"new_heuristic_prompt": "Analyze the current scene in the context of past instances with similar conditions. 
Specifically, look for subtle changes in the environment (e.g., an object slightly out of place) or any person 
entering or loitering outside the storefront within the last hour, and determine if these subtle changes are 
suspicious based on existing rules."}
Re-reasoning Results: {"score": 0.2, "status": "insufficient", "reason": "The scene appears to be a typical store 
interior with no significant activity or changes observed. There are no clear indications of abnormal events such as 
forced entry, suspicious behavior, or physical altercations."}
Reflection Results: {"reason": "The primary reason for the insufficient information is the lack of activity and low 
video quality. The VLM is struggling to identify subtle cues that might indicate early stages of potential anomalies 
like shoplifting, burglary, or robbery, especially given the grainy and low-resolution video feed. The historical data 
also indicates a pattern of ''insufficient'' results due to the static nature of the scene in multiple clips. The current 
rules also focus on slightly different scene specifics.",
"tools_to_use": [{"tool_name": "Image Super-Resolution", "query": ""}],
"new_anomaly_rule": "Lack of customer activity combined with repeated glances towards merchandise near 
the entrance in a convenience store setting during non-peak hours may indicate potential shoplifting or pre-
crime surveillance.",
"new_heuristic_prompt": "Focus on identifying subtle changes in the environment, such as items being moved 
or individuals displaying nervous behavior, even if the video quality is low. Pay close attention to hands and 
pockets for potential signs of shoplifting. Consider if the lack of activity is unusual for the time of day (early 
morning, according to historical data)."}
Re-reasoning Results: {"score": 0.7, "status": "abnormal", "reason": "A person enters the store and quickly 
moves behind the counter, which is inconsistent with normal customer behavior. This sudden movement and 
positioning near the cash register raises suspicion for a potential robbery."}
Video Time 05:00 Reasoning Results: {''score'': 0.0, ''status'': ''normal'', ''reason'': ''The scene shows a static 
view of a store counter with no visible movement or interaction that would suggest an abnormal event. The 
environment appears calm and undisturbed.''}...Reflection Round 1
Reflection Round 2
Figure 4:Visualization of qualitative results on UCF-Crime.On the left is a visualization of the
anomaly score curve. The right side shows the specific reasoning and reflection process of PANDA.
Table 3:Ablation study on the capability of PANDA. Plan-
ningrefers to the self-adaptive scene-aware detection strat-
egy planning, which contains detection planning, adaptive
environment perception, and RAG with anomaly knowledge.
Reflectiondenotes the tool-augmented self-reflection mecha-
nism.Memorycorresponds to the chain-of-memory module,
encompassing both short-term and long-term components.
Key Capabilities of PANDA UCF-Crime (AUC%)
Planning Reflection Memory Performance
✗ ✗ ✗ 75.25
✓ ✗ ✗ 80.37 (+5.12%)
✓ ✓ ✗ 82.63 (+2.26%)
✓ ✓ ✓ 84.89 (+2.26%)Analysis of rules number k.Ta-
ble 2(b) analyses the influence of
the number of retrieved rules kused
during RAG-based anomaly strategy
planning. When too few rules are
retrieved (e.g., k= 1 ), the system
lacks diverse contextual cues to sup-
port robust reasoning, resulting in per-
formance degradation. Conversely,
setting ktoo high may introduce noisy
or irrelevant rules that dilute reasoning
quality. Finally, PANDA achieves op-
timal performance when setting k=
5.
Analysis of short CoM length l.Ta-
ble 2(c) analyzes the impact of vary-
ing the short CoM length lduring the
reasoning phase. The best performance is achieved when l= 5 . When the memory length is reduced
tol= 1 , performance drops noticeably due to insufficient temporal information, which limits the
model’s ability to leverage recent reasoning traces. On the other hand, increasing the memory length
tol= 9 also leads to performance degradation, likely because excessive memory introduces historical
noise that distracts from the current decision-making process.
9

Ablation Study.Table 3 presents an ablation study examining the contribution of each core capa-
bility in PANDA, including self-adaptive scene-aware strategy planning (Planning), tool-augmented
self-reflection (Reflection), and chain-of-memory (Memory). The third row serves as the baseline,
where PANDA performs direct reasoning solely based on the user-defined query, without planning,
reflection, and memory modules. As shown, the performance is relatively poor, with an AUC of
75.25%. Equipping PANDA with the Planning capability yields a substantial improvement of +5.12%
in AUC. This demonstrates the effectiveness of scene perception, rule retrieval via RAG, and context-
aware strategy plans in inspiring the potential of PANDA. Adding the reflection module further
improves performance by +2.26%, suggesting that the self-reflection mechanism, enhanced by the
integration of external tools, expands PANDA’s capability to resolve challenging and ambiguous
cases. Finally, incorporating the memory mechanism results in another +2.26% gain, validating the
effectiveness of the chain-of-memory design. This module enables PANDA to accumulate experience
across time and use it to refine decisions. In summary, each of PANDA’s capabilities plays a vital role
in enabling generalizable and reliable anomaly detection. The synergistic integration of all modules
empowers PANDA as a highly capable agentic AI engineer for generalized V AD. For the ablation
study on the key components corresponding to each capability, please refer to the Subsection C.1 in
the supplementary material.
4.4 Qualitative Results
Figure 4 shows a visualized example from the UCF-Crime test set to illustrate PANDA’s reasoning-
reflection process. The left side of the figure shows the anomaly score curve over time. On the
right, we visualize PANDA’s internal reasoning and reflection process. When the model encounters
uncertainty and cannot confidently determine whether an anomaly is present, it transitions into the
reflection phase. PANDA first analyzes the reason behind the insufficient status from the reasoning
stage, and then invokes external tools to acquire complementary information to support decision-
making. This case highlights PANDA’s capacity for progressive self-refinement and dynamic tool
invocation, demonstrating its effectiveness in tackling complex, real-world video anomaly detection
scenarios. For more visualization samples, please refer to the Supplementary Material.
5 Conclusion
In this work, we presented PANDA, an agentic AI engineer for generalized V AD that eliminates the
need for training data or manually crafted pipelines when faced with various real-world scenarios.
PANDA integrates four core capabilities: self-adaptive scene-aware strategy planning, goal-driven
heuristic reasoning, tool-augmented self-reflection, and a self-improving chain-of-memory. These
modules work in concert to enable PANDA to adaptively detect anomalies across diverse, dynamic,
and previously unseen environments. Our extensive experiments across multiple benchmarks, includ-
ing multi-scene, open-set, and complex scenarios, validate PANDA’s strong generalization ability and
robust performance without any training. These findings highlight PANDA’s potential as a generalist
V AD solution for real-world scenes.
Acknowledgement
This research is supported by the National Research Foundation, Singapore under its AI Singapore
Programme (AISG Award No: AISG3-RP-2022-030).
Technical Appendices and Supplementary Material
A Overview of Technical Appendices and Supplementary Material
This technical appendices and supplementary material provide additional information not included in
the main paper. Specifically:
•Section Boffers further clarification on methodological and experimental details.
•Section Cpresents additional ablation studies and parameter analysis experiments.
10

•Section Ddetails the toolset employed by PANDA.
•Section Eshows the prompts used at each stage of the PANDA framework.
•Section Fprovides additional visualizations of qualitative results across various datasets.
•Section Gdiscusses the current limitations, broader impacts, and future work directions.
B Additional Methodological and Experimental Details
B.1 Visualization of the PANDA Detailed Pipeline
Figure 5 presents a visualized case study on a test example from the UCF-Crime dataset, illustrating
the detailed execution process of PANDA across its core components: anomaly knowledge base
construction, self-adaptive scene-aware strategy planning, goal-driven heuristic reasoning, tool-
augmented self-reflection, and the self-improving chain-of-memory mechanism.
B.2 Clarification on Evaluation Modes and SOTA Comparison
In Table 1 of the main paper, we distinguish betweenofflineandonlinesettings based on whether
future information is utilized when reasoning over a given frame or clip. If future information is
accessed, the method is considered offline; otherwise, it falls under the online setting. For offline
evaluation, we follow the SOTA methods AED-MAE, LA V AD, and AnomalyRuler, which are
compared in Table 1 of the main paper, and apply temporal smoothing (mean filtering, window
size=10) on the final anomaly scores.
Among the compared methods, Holmes-V AU is a fine-tuned VLM-based approach that leverages
detailed anomaly annotations via instruction tuning. It was originally evaluated only on UCF-Crime
and XD-Violence. Its results on UBnormal and CSAD are reproduced by us, without any re-training,
using its publicly released model. As seen in Table 1 of the main paper, PANDA significantly
outperforms Holmes-V AU on UBnormal, an open-set dataset that includes unseen anomaly types.
While Holmes-V AU slightly surpasses PANDA in online mode on CSAD, this is primarily due
to CSAD including a large number of videos derived from UCF-Crime and XD-Violence—the
original training sets of Holmes-V AU. Notably, PANDA still achieves superior performance under
the offline setting, underscoring its strong generalization capability. These results demonstrate the
limitations of relying solely on fine-tuned VLMs when facing domain shift and complex real-world
conditions. Additionally, we report new evaluation results for three prominent training-free baselines:
ZS-CLIP, LLaV A-1.5, and LA V AD, on UBnormal and CSAD. PANDA consistently and substantially
outperforms all of them across both datasets. Together, these experimental results reinforce the
strength of PANDA as a generalist, fully automated V AD agent, capable of adapting its reasoning to
scene-specific conditions without supervision or hand-crafted engineering.
C Additional Experiments
C.1 Ablation of Key Component
Table 4 presents the results of an ablation study evaluating the contribution of PANDA’s six core
components. The third row in the table corresponds to a baseline that directly queries the VLM using
only the user-defined anomaly description without leveraging any PANDA modules. This baseline
yields notably poor performance, confirming that naive prompting alone is insufficient. As more
modules are incrementally added—namely, detection strategy planning, adaptive scene perception,
RAG with anomaly knowledge, self-reflection, short CoM, long CoM—the performance steadily
improves. When all six components are combined, PANDA achieves its best overall performance.
These results demonstrate that each individual module contributes positively to the final performance
and validates the effectiveness of our whole PANDA framework design.
C.2 Impact of Different MLLMs
Table 5a compares the performance of PANDA when integrated with different multi-modal large
language models (MLLMs). GPT-4o and Gemini 2 Flash represent proprietary models, while
11

Table 4: Ablation results of PANDA component.
Key Components of PANDA Dataset
Detection Strategy Planning Self-Adaption Scene-Aware RAG with Anomaly Knowledge Self-Reflection Short CoM Long CoM UCF-Crime(AUC%)
✗ ✗ ✗ ✗ ✗ ✗ 75.25
✓ ✗ ✗ ✗ ✗ ✗ 77.01
✓ ✓ ✗ ✗ ✗ ✗ 78.92
✓ ✓ ✓ ✗ ✗ ✗ 80.37
✓ ✓ ✓ ✓ ✗ ✗ 82.63
✓ ✓ ✓ ✓ ✓ ✗ 83.94
✓ ✓ ✓ ✓ ✓ ✓ 84.89
Table 5: Additional Experiments.
(a) Impact of different MLLMs.
Different UCF-Crime
MLLMs (AUC%)
Qwen2.5-72B 84.03
DeepseekV3 84.72
GPT4o 84.97
Gemini 2 Flash 84.89(b) Effect of input clip length.
Input Clip Length UCF-Crime
(Number of frames) (AUC%)
1 84.25
3 84.56
5 84.89
7 83.15(c) Analysis of Inference speed.
Datasets Average speed of inference
Name (FPS)
UCF-Crime 0.82
XD-Violence 0.86
UBnormal 0.79
CSAD 0.53
DeepSeek-V3 and Qwen2.5-72B are open-source alternatives. As shown, GPT-4o achieves the
highest performance. However, we adopt Gemini 2 Flash in our main pipeline due to its strong
trade-off between performance and cost-effectiveness.
Notably, although Qwen2.5-72B yields the lowest performance among the compared models, it
remains significantly superior to prior training-free baselines. Given its open-source nature and ease
of local deployment, it serves as a practical and scalable option for resource-constrained scenarios.
C.3 Effect of Input Clip Length
Table 5b analyzes how varying the number of frames in each input video clip affects PANDA’s
performance. As the input length increases from 1 to 5 frames, detection accuracy steadily improves,
suggesting that short-range temporal cues are beneficial to the reasoning process. However, when the
clip length is extended to 7 frames, performance noticeably drops. We hypothesize that this is due
to the binary labeling strategy used during evaluation—if a clip is anomalous, all frames of the clip
are scored as anomalous. For longer clips that may contain both normal and abnormal frames, this
scoring scheme introduces noise, leading to performance degradation.
C.4 Analysis of Inference Speed
Table 5c reports the average inference speed of PANDA across different datasets. As observed,
PANDA achieves similar inference times on UCF-Crime, XD-Violence, and UBnormal. However, a
noticeable slowdown is observed on the CSAD dataset. This is primarily because CSAD contains
videos with complex conditions and scenarios, leading PANDA to more frequently enter the reflection
stage. Since the reflection stage involves invoking additional tools, it introduces greater computational
overhead. Despite this, the overall average inference speed of PANDA remains acceptable for non-
time-sensitive applications, demonstrating its practical feasibility in real-world deployments where
latency is not a critical constraint.
D Toolset Details
PANDA is equipped with a modular and extensible set of tools designed to enhance video content
analysis, mitigate visual degradation, and provide external contextual information. These tools are
automatically selected and invoked during the self-reflection stage based on the detection context.
Below, we provide a detailed summary of each tool integrated into the PANDA framework.
Object Detection.PANDA employs the YOLOWorld [ 39] model pretrained on a wide set of
open-world concepts. It supports fine-grained category-specific detection including actions like
person hitting another”, person setting something on fire”, and “person stealing”. This enables robust
anomaly-related scene understanding through bounding-box localization and category labels.
12

Image Denoising.PANDA uses OpenCV’s fast non-local means filter for image denoising. It
reduces color and spatial noise in frames using adaptive filtering, helping enhance clarity in low-light
or noisy environments.
Image Deblurring.PANDA applies unsharp masking with Gaussian blur subtraction to sharpen
edge details for motion blur or out-of-focus issues. This lightweight enhancement improves perceptual
clarity without the need for retraining.
Image Brightness Enhancement.PANDA uses OpenCV’s CLAHE (Contrast Limited Adaptive
Histogram Equalization) on the L-channel in LAB color space. This ensures localized brightness
normalization for dimly lit or overexposed frames.
Image Super-Resolution.PANDA integrates the Real-ESRGAN [ 40] model for resolution en-
hancement. It improves detail preservation and restores textures in low-resolution videos using a
deep RRDBNet-based super-resolution pipeline.
Image Retrieval.We use CLIP-based [ 41] visual-textual retrieval to match current queries (e.g.,
“robbery incident”) with previously seen keyframes. Cosine similarity between CLIP embeddings is
used for scoring relevance.
Web Search.PANDA leverages the Tavily Search API [ 42] for querying web content related to
unknown or uncertain anomalies. Search results are parsed into structured summaries that can be
referenced in the reasoning process.
Image Zooming.PANDA leverages a bicubic interpolation-based zooming tool to magnify regions
that require enhanced spatial detail using a specified zoom factor. This is useful when detecting
small-scale interactions or distant activities.
All tools are dynamically invoked during the reflection stage via the MLLM-generated reflection
plan. Each tool outputs enhanced frame sets and structured summaries that are used to augment the
reasoning prompt for follow-up reasoning steps.
E System Prompts
In this section, we present the detailed prompts used by PANDA across its core stages. Figure
6 illustrates the prompt used during anomaly knowledge base construction. Figure 7 shows the
self-adaptive environmental perception prompt. Figure 8 presents the prompt for anomaly detection
strategy planning. Figure 9 demonstrates the goal-driven heuristic reasoning prompt. Figure 10
displays the prompt used during the tool-augmented self-reflection phase.
F Additional Visualization Results
Figures 11 and 12 further show the qualitative results of the samples on the XD-Violence and
UBnormal test sets.
G Discussions
G.1 Limitations
PANDA currently integrates a curated set of commonly used tools for enhancement and reasoning.
While these tools suffice for most general scenarios, expanding the toolkit to accommodate domain-
specific modalities (e.g., thermal imaging) could broaden PANDA’s applicability.
G.2 Broader Impacts
PANDA advances the paradigm of automated video anomaly detection by integrating vision-language
models and decision-time tool augmentation into a unified AI agent framework. This work has the
potential to improve the robustness and interpretability of security monitoring systems, enabling
adaptive deployment across diverse environments without training. Moreover, We also recognize
the importance of ethical considerations: systems like PANDA must be deployed responsibly, with
attention to privacy protection, fairness, and minimizing unintended surveillance harms.
13

G.3 Future Work
Enhancing Real-Time Adaptability.To make PANDA more suitable for practical deployments,
future work could focus on reducing inference latency and optimizing tool invocation paths.
Improving Spatial Reasoning and Localization.Currently, PANDA focuses on frame-level or
clip-level anomaly identification. Incorporating spatial anomaly localization, such as identifying
the precise region or object involved in an abnormal event, could significantly expand its utility in
surveillance systems.
14

web_search_results = web_search(query="body 
language cues for shoplifting")
web_search_results =
"1. 7 Signs Of A Potential Shoplifter - Off Duty Officers
Displaying Avoidance Behaviors
One of the keys to identifying a shoplifter is to have a 
thorough knowledge of body language. ...
https://offdutyofficers.com/7-signs-of-a-potential-
shoplifter/"
2. ... "
  "Scene Overview": "The video depicts a convenience store with shelves 
stocked with various products. The store appears well-lit, with a mix of canned 
goods, snacks, and other items visible. A customer is seen browsing the shelves 
while another person, possibly an employee, is seated at the counter. The 
environment suggests a typical retail setting.",
  "Weather Condition": "The video is shot indoors under artificial lighting, 
which is consistent throughout the frames. There are no indications of weather 
conditions as the scene is entirely inside the store.",
  "Video Quality": "The video quality appears to be that of a standard security 
camera recording, with low clarity and some graininess. The image is blurry, and 
there are no significant distortions or artifacts affecting the overall view.",
  "Potential Anomalies": "Robbery, shoplifting, and stealing are high probability 
anomalies in retail store scenarios."Environmental Information  
"Preprocessing": "Image Denoise -> Image Deblur",
"Potential Anomalies": "Shoplifting, Stealing, Robbery",
"Heuristic Prompts": {
    "Shoplifting": "Analyze customer interactions with products. Does the customer 
conceal an item without paying? Is the customer acting suspiciously near valuable 
items? Check if the customer leaves the store without the item having been 
scanned.",
    "Stealing": "Observe interactions between customers and items. Is someone 
discreetly taking items from shelves or other customers without permission? Pay 
attention to quick movements and attempts to hide objects.",
    "Robbery": "Detect sudden aggressive behavior directed at the cashier or other 
customers. Look for weapons, threats, or demands for money. Analyze if someone 
forcefully takes money or valuable items from the cashier or customers.",Detection Strategy Plan
Video Time 00:05  Reasoning Results: {"score": 0.0, "status": "normal", "reason": "There are no visible signs of abnormal events such as shoplifting, stealing, or robbery in this video clip. The 
scene appears to be a typical retail environment with customers browsing and shopping."}
Video Time 00:30 Reasoning Results: {"score": 0.2, "status": "insufficient", "reason": "The video frames show a general view of a store with people walking around, but no specific actions that 
clearly indicate shoplifting, stealing, or robbery. Due to the lower video quality, the lack of clarity and distinct action makes it difficult to make accurate judgments."}...
Qwen2.5-VL
Gemini2.0 Flash
Reflection Results: {"reason": "The VLM reports 'insufficient information' because the low video 
quality obscure the subtle actions and interactions that might indicate shoplifting or other 
suspicious behavior in a convenience store. With reference to similar historical experiences, the 
focus should be on improving the video quality to better analyze these subtle actions.",
"tools_to_use": [{"tool_name": "Image Super-Resolution","query": ""}],
"new_anomaly_rule": "Enhance image clarity through super-resolution to meticulously analyze 
hand movements and object interactions in retail environments, focusing on identifying 
concealment of items in pockets or bags without payment near shelves and checkout areas.",
"new_heuristic_prompt": "After super-resolving the images, analyze the video with a focus on 
the customer's hand movements, interactions with products on the shelves, and any attempts to 
conceal items. Pay close attention to whether the customer places any items into their pockets or 
bags without visible signs of payment or interaction with the cashier. Consider the customer's 
proximity to the cashier and their overall demeanor. If there's evidence of concealment, classify the 
scene as potentially abnormal; otherwise, classify it as normal.".}Reflection Round 1
Tool InvokeTool Code
Tool Name: 
"Image Super-Resolution" 
Parama: ""Long CoM
Re-reasoning Results: {"score": 0.2, "status": "insufficient", "reason": "The scene appears to be a typical store interior with no significant activity or changes observed. The person in the 
foreground is picking up merchandise with no obvious shoplifting or suspicious behavior."}
Reflection Results: {"reason": "The VLM is reporting insufficient information due to the low 
video quality, which obscures subtle actions that might indicate shoplifting or other anomalies. 
While image super-resolution has been used previously, and there's an overall calm scene, the 
VLM isn't focusing on the *context* of actions. For instance, a person picking up an item and 
placing it in a bag might be normal, but doing so furtively, while looking around suspiciously, is 
not. The system needs to be more sensitive to body language and subtle cues associated with 
potential theft.",
"tools_to_use": [{"tool_name": "Web Search", "query": "body language cues for shoplifting"}],
"new_anomaly_rule": "Analyze individuals' body language and interaction patterns within the 
store, focusing on subtle cues such as furtive glances, nervous movements, concealment attempts, 
and inconsistencies in behavior around merchandise. Consider proximity to checkout and 
interactions with staff to determine the likelihood of shoplifting. ",
"new_heuristic_prompt": "Analyze the video, focusing on subtle cues in body language that 
might indicate shoplifting. Look for furtive glances, nervous movements, attempts to conceal 
items, and inconsistent or hurried actions around merchandise. Consider the context of these 
actions. Are they near the checkout, or further away? Are they interacting with employees, or 
avoiding interaction? If several of these cues are present, classify the scene as potentially 
abnormal; otherwise, classify it as normal.".}Reflection Round 2
Tool InvokeTool Code
Tool Name: "Web Search" 
Parama: 
{"query": "body language 
cues for shoplifting"}
Qwen2.5-VLRe-reasoning Results: {"score": 0.8, "status": "abnormal", "reason": "The person in the foreground appears to be concealing something under their arm, which is a common indicator of shoplifting. 
The individual's body language, including their avoidance of eye contact and nervous movements, aligns with the heuristic prompts for shoplifting."}
Video Time 00:35 Reasoning Results: {''score'': 0.0, ''status'': ''normal'', ''reason'': ''The scene shows a static view of a store counter with no visible movement or interaction that would suggest an 
abnormal event. The environment appears calm and undisturbed.''}
Retrieve
History
Experience"Insufficient Reason": "The video frames show a general view of a store with no clear indication of specific abnormal 
events such as shoplifting or stealing. The lack of detailed actions or interactions makes it difficult ..."
"Reflection Result": { "reason": "The VLM is struggling due to the low video quality and the subtlety of actions associated 
with potential shoplifting or stealing in a busy retail environment. The lack of clear details makes ...",
"tools_to_use": [{"tool_name": "Image Super-Resolution", "query": ""}],
"new_anomaly_rule": "Improve image clarity through super-resolution to meticulously analyze hand movements, object 
interactions, and body language in retail environments, focusing on identifying indicators of shoplifting, such as concealing 
items or suspicious bag handling near shelves and checkout areas. ",
"new_heuristic_prompt": "After super-resolution enhancement, focus on individuals near the checkout counter and shelves, 
paying close attention to their hand movements, body language, and interactions with merchandise.".}History Experience
Long CoM
Retrieve
History
Experience"Insufficient Reason": "The video frames show a general view of a store with minimal activity. There are no clear 
indications of abnormal events such as shoplifting, stealing, or other listed anomalies..."
"Reflection Result": {"reason": "The VLM is reporting 'insufficient information' due to the subtlety of actions associated 
with potential shoplifting in a typical convenience store setting and low video quality. The VLM needs to focus on specific 
actions that, while seemingly innocuous on their own, might indicate shoplifting when combined.",
"tools_to_use": [{"tool_name": "Object Detection", "query": ""}],
"new_anomaly_rule": "Detect and track actions such as concealing items, placing items in bags/pockets, nervous 
behavior, and interactions with store personnel to identify potential shoplifting incidents. ",
"new_heuristic_prompt": "Analyze the video, focusing on the actions of the customer browsing the shelves. Specifically, 
detect and track if they are concealing any items, placing them in a bag or pocket without paying... ".}History Experience
...
Gemini2.0 Flash
Please help me detect the following types of abnormal 
events: Abuse, Arrest, Arson, Assault, Burglary, 
Explosion, Fighting, Road Accidents, Robbery, 
Shooting, Shoplifting, Stealing, Vandalism.Anomaly Knowledge Base
Event Type
Anomaly Rules
Applicable Scenes"Rule ID": "SHO-X001",
"Event Type": "Shoplifting",
"Rule Description": "Customer puts merchandise inside clothing or 
personal bag.",
"Applicable Scenes": "Clothing store, Supermarket, Pharmacy".
...
   
Qwen2.5-VL
RAG
Gemini 2.0 FlashReflection 
Prompt
Qwen2.5-VL
Qwen2.5-VLAnomaly Knowledge Base Construction
Self-adaption Scene-aware Strategy Planning
Goal-Driven Heuristic Reasoning
RAG
Gemini 2.0 FlashReflection 
PromptTool-Augmented Self-Reflection
Refined Reasoning 
Promptimage_super_resolution_results = 
image_super_resolution(frames: list)
Refined Reasoning
Prompt
Qwen2.5-VL
Perception 
Prompt
Reasoning 
PromptRAGRetrieveAnomaly 
Knowledge 
Base Prompt
Planning 
PromptFigure 5:Visualization of the PANDA detailed pipeline.We show a more detailed pipeline for
PANDA by visualizing the results of a test sample on UCF-Crime.
15

Anomaly Knowledge Base Construction Prompt 
You are an expert in designing detection rules for video anomaly detection. Based 
on the user's specified types of abnormal events, your task is to generate 20 
comprehensive and diverse detection rules for each event type.
User Requirement:
{user_query}
Each rule must include the following four fields:
1. Rule ID: A unique identifier for the rule (e.g., "FIG-X001").
2. Event Type: The corresponding abnormal event category (e.g., "Fighting").
3. Rule Description: A concise and clear description of the behavioral pattern 
that defines this event.
4. Applicable Scenes: A list of real-world scenarios where this rule may be 
applicable (e.g., "Street, Shopping mall, School playground").
The rules should cover a wide range of realistic situations for each event type.
Please output your response in the following structured JSON format:
[{
  "Rule ID": "FIG-X001",
  "Event Type": "Fighting",
  "Rule Description": "Pulling hair or grabbing clothes during struggle",
  "Applicable Scenes": "Shopping mall, Playground, Street corner"
},
...
]Figure 6: System prompt for anomaly knowledge base construction.
Self-Adaption Environmental Perception Prompt 
You are an expert in video anomaly perception. Your task is to perform an initial 
understanding and analysis of the provided video frames based on the user's 
specified requirements.
User Requirement: 
{user_query}
Please respond by completing the following four aspects:
1. Scene Overview: Describe the environment shown in the video (e.g., shopping 
mall, office, street, surveillance corridor) and briefly summarize the main 
activities or events observed.
2. Weather Condition: Describe the visual lighting conditions (e.g., 
daytime/nighttime, sunny/overcast, bright/dim).
3. Video Quality:  Comment on the overall quality of the video (e.g., clear, 
blurry, noisy, low frame rate, low light.etc.).
4. Potential Anomalies: Based on the visual content, what types of abnormal events 
are likely to occur in this video (e.g., Fighting, Stealing).
Please return your output strictly in the following JSON format:
 {
  "Scene Overview": "...",
  "Weather Condition": "...",
  "Video Quality": "...",
  "Potential Anomalies": "..."
}    
Figure 7: System prompt for self-adaptive environmental perception.
16

Anomaly Detection Strategy Planning Prompt 
You are a strategic expert in video anomaly detection, specializing in planning 
effective detection strategies based on user-defined requirements, video 
environment information, and scene-specific anomaly rules. Your role is to 
generate an optimal plan that guides the analysis module in accurately detecting 
anomalies in the current video scene.
User Requirement:
{user_query}
Video Environment Information:
1. Scene Overview: {env_info.get("Scene Overview", "Unknown")}
2. Weather Condition: {env_info.get("Weather Condition", "Unknown")}
3. Video Quality: {env_info.get("Video Quality", "Unknown")}
4. Potential Anomalies: {env_info.get("Potential Anomalies", "Unknown")}
Anomaly Detection Rules:
{anomaly_rules}
Based on the user’s requirement, the preliminary video environment information, 
and the provided anomaly rules, please design a strategy tailored for this video 
scenario. Your response must include the following three components:
1. Preprocessing Recommendations and Pipeline
    Suggest a sequence of preprocessing steps (e.g., Image Deblurring, Brightness 
Enhancement, Image Denoising) that can help improve video quality and support 
better anomaly detection, especially if the video is of poor quality.
2. Potential Anomaly Types
    Based on the preliminary video environment information and the given scene-
related anomaly rules, further infer and list the most possible types of anomalies 
in this scenario.
3. Heuristic Prompts for VLM
    Using the anomaly rules as guidance, craft chain-of-thought–style heuristic 
prompts for each potential anomaly type. These prompts are intended to assist 
Visual Language Model in performing accurate anomaly judgments. 
Please return your output strictly in the following JSON format:
{
  "Preprocessing": "Step1 -> Step2 -> ...",
  "Potential Anomalies": "Fighting, Stealing, ...",
  "Heuristic Prompts": {
    "Fighting": "Heuristic prompt with reasoning steps...",
    "Stealing": "Heuristic prompt with reasoning steps...",
    ...
  }
}
Example Output:
{
  "Preprocessing": "Image Brightness Enhancement -> Image Denoising",
  "Potential Anomalies": "Fighting, Stealing",
  "Heuristic Prompts": {
    "Fighting": "Observe the number of people, their movements, and interactions. 
If two individuals are repeatedly making aggressive contact, consider it a 
potential fight.",
    "Stealing": "Identify solitary individuals interacting with objects, 
especially if they conceal items or leave quickly without paying."
  }
}Figure 8: System prompt for anomaly detection strategy planning.
17

Goal-Driven Heuristic Reasoning Prompt
You are a highly skilled expert in video anomaly detection, specializing in 
identifying abnormal events through temporal and spatial analysis of visual 
evidence.
Given a sequence of video frames, user requirements, potential anomalies, anomaly 
detection rules, heuristic prompts, and enhancement/reflection information, your 
task is to assess the likelihood of abnormal events in the current video clip.
You must output:
- A soft anomaly score between **0.0 (clearly normal)** and **1.0 (clearly 
abnormal)**.
- A status label from: **"normal"**, **"abnormal"**, or **"insufficient"**.
- A reason justifying your decision.
User Requirement:
{user_query}
Historical Detection Info:
{history_result_prompt if history_result_prompt.strip() else "No reliable 
historical detection information available."}
Current Video Clip Index:
Clip {index}
Potential Anomalies:
{planning_info.get('potential_anomalies', '')}
Anomaly Detection Rules:
{anomaly_rules}
Heuristic Prompt:
{planning_info.get('heuristic_prompts', '')}
Enhancement and Reflection Information:
{formatted_enhancement_prompt}
Your analysis should follow three steps:
1. Describe the main visible actions and interactions between people or objects in 
the scene.
2. Assess how strongly these actions match any known abnormal event patterns using 
the provided rules and prompts. The provided anomaly rules may not be 
comprehensive, so you also apply your own expert reasoning.
3. Based on your assessment, assign a score and label, and explain your reasoning 
clearly.
4. If the [Enhancement and Reflection Information] section provides additional 
information, you should refer to it emphatically.
Scoring Guidelines:
- A score close to **1.0** indicates clear and confident abnormal behavior.
- A score close to **0.0** indicates clearly normal behavior.
- A score near **0.5** means uncertain, ambiguous behavior or mixed signals.
Examples of valid reasons for "insufficient":
- "The entire scene is too blurry or dark, making it difficult to distinguish any 
actions."
- "All persons are either occluded or out of frame."
- "Only partial limbs are visible and motion cues are unclear."
Please strictly output your response in the following JSON format:
{
      "score": float,  //anomaly score in [0.0, 1.0]
      "status": "normal/abnormal/insufficient",
      "reason": "A detailed explanation of your reasoning..."
    }Figure 9: System prompt for goal-driven heuristic reasoning.
18

Tool-Augmented Self-Reflection Prompt 
You are a reflection assistant within a video anomaly detection system.
The current VLM analysis module has returned "insufficient statu" for determining 
whether an abnormal event occurred in the given video clip.
Your task is to critically analyze the situation based on the provided context and 
recommend solutions.
Here is the contextual information:
- User Requirement: {user_query}
- Video Environment Information:
    **Scene Overview: {env_info.get("Scene Overview", "Unknown")}
    **Weather Condition: {env_info.get("Weather Condition", "Unknown")}
    **Video Quality: {env_info.get("Video Quality", "Unknown")}
- Anomaly Detection Rules:
{anomaly_rules}
- Potential Anomalies: {planning_info.get('Potential Anomalies', '')}
- Historical Detection Results: 
{historical_results}
- Current VLM Output Reason: {reason}
- Information Enhancement tools Already Used: {tools_already_used}
- History Experience: {memory_context}
Based on this information, your tasks are:
1. Analyze and determine the primary reasons for the insufficient information.
2. Recommend which tools from the available options should be used to enhance the 
information for better anomaly detection.
{tool_description_text}
3. For any selected tool that requires a 'query' input (e.g., image_retrieve, 
web_search), generate an appropriate query based on the context; otherwise leave 
the 'query' field empty.
4. Propose a new representative anomaly detection rule derived from the current 
situation to better support future VLM analysis.
5. Propose an additional heuristic prompt based on the context and your analysis 
to better guide the VLM toward an accurate judgment.
Please output your response in the following structured JSON format:
 {
      "reason": "...your analysis of why the information is insufficient...",
      "tools_to_use": [
        {
          "tool_name": "One of the most critical tools.",
          "query": "generated query if needed, otherwise leave empty"
        }
      ],
      },
      "new_anomaly_rule": "...a new representative anomaly rule derived from your 
analysis and context...",
      "new_heuristic_prompt": "...additional guidance to help the VLM make a more 
accurate judgment..."
          }
Important Notes:
- When calling tools, make sure you don't duplicate any of the information 
enhancement tools that have already been applied, and use only one of the most 
critical tools at a time.
- If all available information enhancement tools have been exhausted, you should 
directly suggest in the 'new_heuristic_prompt' how to guide the VLM analysis 
module to make the most reasonable judgment based on incomplete evidence.
- If the current context provides enough information to make a clear judgment, 
please directly guide the VLM analysis module in the 'new_heuristic_prompt' to 
conclude whether the event is abnormal or normal.Figure 10: System prompt for tool-augmented self-reflection.
19

...Video Time 00:05  Reasoning Results: {''score'': 0.0, ''status'': ''normal'', ''reason'': ''The video shows a news anchor reporting live from a studio. 
There are no visible signs of abnormal events such as fighting, shooting, riot, abuse, car accidents, or explosions in the frames provided.''}
Video Time 00:15 Reasoning Results: {score'': 0.2, ''status'': ''insufficient'', ''reason'': ''The video shows a news anchor reporting on an Indonesia 
protest, but the frames do not provide enough visual detail to assess the presence of abnormal events like fighting, shooting, riot, abuse, car accident, 
or explosion. The crowd appears to be gathered, but specific actions or behaviors that would indicate these anomalies are not discernible.''}
Reflection Results: reflection: {"reason": "The VLM output indicates that while the video shows a protest, it lacks the visual detail necessary to 
confirm the presence of specific abnormal events. This suggests that the initial VLM analysis couldn't identify actions or behaviors indicative of 
fighting, shooting, riot, abuse, car accidents, or explosions. The core issue is the granularity of the analysis; it needs to delve deeper into the actions 
of individuals within the crowd to identify potential anomalies based on the provided rules.",
"tools_to_use": [{"tool_name": "Object Detection","query": ""}],
"new_anomaly_rule": "Isolated acts of violence or aggression within a protest crowd, such as pushing, shoving, or physical altercations between 
individuals."，
"new_heuristic_prompt": "Analyze the crowd dynamics, focusing on identifying individuals exhibiting aggressive behavior, wielding weapons, or 
causing disturbances. Check for rapid and chaotic movements indicative of rioting or fleeing from a potential explosion. Also, pay close attention to 
interactions between protesters and authorities to determine if abuse or clashes are occurring."}
Re-reasoning Results: {"score": 0.2, "status": "insufficient", "reason": "The video frames show a news anchor reporting on an Indonesia protest and 
a crowd of people, but no specific actions or behaviors indicating fighting, shooting, riot, abuse, car accident, or explosion are observed. The crowd 
appears to be gathered peacefully, and there is no visible evidence of abnormal events."}
Reflection Results: {"reason": "The primary reason for the insufficient information is the VLM's inability to discern specific actions or behaviors 
within the crowd that would indicate the presence of anomalies like fighting, rioting, abuse, car accidents, or explosions. While the scene context is 
known (protest in Jakarta), the VLM lacks detailed analysis of individual and group activities to confirm or deny the occurrence of abnormal events. 
The 'object_detection' tool was already used, suggesting that basic object recognition didn't provide enough insights. The VLM can only determine 
that there is a crowd but can't figure out any potential fighting or dangerous activity.",
"tools_to_use": [{"tool_name": "Image Retrieve", "query": "riots or protests involving violence or clashes in Jakarta"}],
"new_anomaly_rule": " Sudden, rapid crowd dispersal or localized outbreaks of aggressive behavior (shoving, pushing, object throwing) within a 
protest environment. "，
"new_heuristic_prompt": "Analyze the crowd's movements and interactions closely. Focus on identifying signs of aggression, such as pushing, 
shoving, or individuals being targeted. Look for objects being thrown, signs of property damage, or any indications of conflict with authorities. 
Consider the speed and direction of crowd movement - are they dispersing rapidly, suggesting a potential threat or explosion? Also, search for 
objects being brandished such as sticks, weapons or other harmful objects."}
Re-reasoning Results: {"score": 0.7,"status": "abnormal","reason": "The crowd appears to be engaged in a protest, with some individuals wearing 
masks and holding flags. There was a sense of urgency and movement at the scene, which is typical of a riot."}
Video Time 00:20 Reasoning Results: {"score": 0.8, "status": "abnormal", "reason": "The crowd appears to be engaged in a protest, with some 
individuals raising their hands and others holding flags. The presence of a large number of people and the context of election results suggest 
potential unrest. "}...Reflection Round 1
Reflection Round 2
Figure 11: Visualization of qualitative results for a sample on the XD-Violence test set.
Video Time 00:05  Reasoning Results: {"score": 0.1, "status": "normal", "reason": "The scene shows a quiet street at night with a few pedestrians 
standing around. There are no visible signs of abnormal behavior such as jaywalking, walking drunk, accidents, or running. The pedestrians appear 
to be stationary or moving slowly, which does not match the criteria for any of the listed abnormal events."}
Video Time 00:10 Reasoning Results: {"score": 0.2, "status": "insufficient", "reason": "The video frames show a pedestrian crossing a zebra 
crossing, but the low lighting make it difficult to discern specific details about the pedestrian's actions or the surrounding environment. There is no 
clear indication of abnormal behavior based on the provided rules and prompts."}
Reflection Results: {"reason": "The primary reasons for insufficient information are low lighting, moderate video quality, and the pedestrian's 
actions being difficult to discern. This makes it hard to determine if the pedestrian is exhibiting abnormal behavior such as walking drunk or 
jaywalking. The presence of snow and the nighttime setting further complicate the analysis.",
"tools_to_use": [{"tool_name": "Image Brightness Enhancement", "query": ""}],
"new_anomaly_rule": "Pedestrian exhibits unstable gait or balance while crossing a street, especially under adverse conditions such as low lighting 
or slippery surfaces, indicating possible intoxication or medical issue. ",
"new_heuristic_prompt": "Focus on identifying any indicators of erratic movement, such as swaying, stumbling, or sudden changes in direction. 
Also, check if the pedestrian deviated from the designated crosswalk area. If the pedestrian's movement appears stable and they remain within the 
crosswalk, consider the event normal, despite the low lighting."}
Re-reasoning Results: {"score": 0.2, "status": "insufficient","reason": "The video frames show a pedestrian crossing a street at night, but the foggy 
conditions make it difficult to discern specific actions or behaviors that could indicate abnormal events. There are no clear indications of jaywalking, 
walking drunk, accidents, or other anomalies based on the visible information."}
Reflection Results: {"reason": "The primary reasons for insufficient information are the foggy conditions, and potentially low resolution which 
hinder the VLM's ability to discern specific actions and behaviors. While 'Brightness Enhancement' was already applied, the fog and resolution may 
still be limiting factors. We need to try and improve clarity.",
"tools_to_use": [{"tool_name": "Image Denoising","query": ""}],
"new_anomaly_rule": "Reduced visibility obscures pedestrian activity, making it difficult to identify normal behavior or detect pedestrian-related 
anomalies in nighttime or adverse weather conditions.",
"new_heuristic_prompt": "Focus on identifying changes in pedestrian speed and direction, even if the details are not perfectly clear. Look for 
sudden stops, erratic movements, or collisions, however subtle they might appear. Prioritize detecting deviations from normal walking patterns 
considering the environment of the scene."}
Re-reasoning Results: {"score": 0.7, "status": "abnormal", "reason": "The pedestrian in the center of the frame appears to be walking erratically, 
with a lack of coordination and balance, which matches the description of 'Walking Drunk'. The low visibility due to foggy conditions adds 
uncertainty but does not negate the observed erratic behavior."}
Video Time 00:15 Reasoning Results: {"score": 0.7, "status": "abnormal", "reason": "A pedestrian is crossing the street outside of the designated 
crosswalk area, which matches the rule JAY-X001 for jaywalking. The pedestrian appears to be crossing at a point where there is no marked 
crosswalk nearby, indicating a potential safety risk."}Reflection Round 1
Reflection Round 2
Figure 12: Visualization of qualitative results for a sample on the UBnormal test set.
20

References
[1]Peng Wu, Chengyu Pan, Yuting Yan, Guansong Pang, Peng Wang, and Yanning Zhang. Deep learning for
video anomaly detection: A review.arXiv preprint arXiv:2409.05383, 2024.
[2]Mahmudul Hasan, Jonghyun Choi, Jan Neumann, Amit K Roy-Chowdhury, and Larry S Davis. Learning
temporal regularity in video sequences. InCVPR, pages 733–742, 2016.
[3]Yannick Benezeth, P-M Jodoin, Venkatesh Saligrama, and Christophe Rosenberger. Abnormal events
detection based on spatio-temporal co-occurences. InProc. IEEE Conf. Comput. Vis. Pattern Recognit.,
pages 2458–2465, 2009.
[4]Yang Cong, Junsong Yuan, and Ji Liu. Sparse reconstruction cost for abnormal event detection. InProc.
IEEE Conf. Comput. Vis. Pattern Recognit., pages 3449–3456, 2011.
[5]Cewu Lu, Jianping Shi, and Jiaya Jia. Abnormal event detection at 150 fps in matlab. InCVPR, pages
2720–2727, 2013.
[6]Vijay Mahadevan, Weixin Li, Viral Bhalodia, and Nuno Vasconcelos. Anomaly detection in crowded
scenes. InCVPR, pages 1975–1981, 2010.
[7]Yu Yao, Xizi Wang, Mingze Xu, Zelin Pu, Yuchen Wang, Ella Atkins, and David J Crandall. Dota:
Unsupervised detection of traffic anomaly in driving videos.IEEE transactions on pattern analysis and
machine intelligence, 45(1):444–459, 2022.
[8]Guansong Pang, Cheng Yan, Chunhua Shen, Anton van den Hengel, and Xiao Bai. Self-trained deep
ordinal regression for end-to-end video anomaly detection. InProceedings of the IEEE/CVF conference on
computer vision and pattern recognition, pages 12173–12182, 2020.
[9]Weixin Luo, Wen Liu, and Shenghua Gao. Remembering history with convolutional lstm for anomaly
detection. InICME, pages 439–444, 2017.
[10] Wen Liu, Weixin Luo, Dongze Lian, and Shenghua Gao. Future frame prediction for anomaly detection–a
new baseline. InCVPR, pages 6536–6545, 2018.
[11] Dong Gong, Lingqiao Liu, Vuong Le, Budhaditya Saha, Moussa Reda Mansour, Svetha Venkatesh, and
Anton van den Hengel. Memorizing normality to detect anomaly: Memory-augmented deep autoencoder
for unsupervised anomaly detection. InCVPR, pages 1705–1714, 2019.
[12] Zhiwei Yang, Jing Liu, Zhaoyang Wu, Peng Wu, and Xiaotao Liu. Video event restoration based on
keyframes for video anomaly detection. InCVPR, pages 14592–14601, 2023.
[13] Zhiwei Yang, Peng Wu, Jing Liu, and Xiaotao Liu. Dynamic local aggregation network with adaptive
clusterer for anomaly detection. InECCV, pages 404–421, 2022.
[14] Zhiwei Yang, Jing Liu, and Peng Wu. Text prompt with normality guidance for weakly supervised
video anomaly detection. InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 18899–18908, 2024.
[15] Peng Wu, Xuerong Zhou, Guansong Pang, Lingru Zhou, Qingsen Yan, Peng Wang, and Yanning Zhang.
Vadclip: Adapting vision-language models for weakly supervised video anomaly detection. InProceedings
of the AAAI Conference on Artificial Intelligence, volume 38, pages 6074–6082, 2024.
[16] Peng Wu, Jing Liu, Yujia Shi, Yujia Sun, Fangtao Shao, Zhaoyang Wu, and Zhiwei Yang. Not only look,
but also listen: Learning multimodal violence detection under weak supervision. InECCV, pages 322–339,
2020.
[17] Waqas Sultani, Chen Chen, and Mubarak Shah. Real-world anomaly detection in surveillance videos. In
CVPR, pages 6479–6488, 2018.
[18] Peng Wu, Xuerong Zhou, Guansong Pang, Zhiwei Yang, Qingsen Yan, Peng Wang, and Yanning Zhang.
Weakly supervised video anomaly detection and localization with spatio-temporal prompts. InProceedings
of the 32nd ACM International Conference on Multimedia, pages 9301–9310, 2024.
[19] Hang Du, Sicheng Zhang, Binzhu Xie, Guoshun Nan, Jiayang Zhang, Junrui Xu, Hangyu Liu, Sicong
Leng, Jiangming Liu, Hehe Fan, et al. Uncovering what why and how: A comprehensive benchmark for
causation understanding of video anomaly. InProceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 18793–18803, 2024.
21

[20] Jiaqi Tang, Hao Lu, Ruizheng Wu, Xiaogang Xu, Ke Ma, Cheng Fang, Bin Guo, Jiangbo Lu, Qifeng Chen,
and Yingcong Chen. Hawk: Learning to understand open-world video anomalies.Advances in Neural
Information Processing Systems, 37:139751–139785, 2024.
[21] Huaxin Zhang, Xiaohao Xu, Xiang Wang, Jialong Zuo, Chuchu Han, Xiaonan Huang, Changxin Gao,
Yuehuan Wang, and Nong Sang. Holmes-vad: Towards unbiased and explainable video anomaly detection
via multi-modal llm.arXiv preprint arXiv:2406.12235, 2024.
[22] Huaxin Zhang, Xiaohao Xu, Xiang Wang, Jialong Zuo, Xiaonan Huang, Changxin Gao, Shanjun Zhang,
Li Yu, and Nong Sang. Holmes-vau: Towards long-term video anomaly understanding at any granularity.
arXiv preprint arXiv:2412.06171, 2024.
[23] Nicolae-C Ristea, Florinel-Alin Croitoru, Radu Tudor Ionescu, Marius Popescu, Fahad Shahbaz Khan,
Mubarak Shah, et al. Self-distilled masked auto-encoders are efficient video anomaly detectors. In
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15984–
15995, 2024.
[24] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman,
Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report.arXiv
preprint arXiv:2303.08774, 2023.
[25] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan
Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable
multimodal models.arXiv preprint arXiv:2312.11805, 2023.
[26] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie
Wang, Jun Tang, et al. Qwen2. 5-vl technical report.arXiv preprint arXiv:2502.13923, 2025.
[27] OpenGVLab Team. Internvl2: Better than the best—expanding performance boundaries of open-source
multimodal models with the progressive scaling strategy, 2024.
[28] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united
visual representation by alignment before projection.arXiv preprint arXiv:2311.10122, 2023.
[29] Luca Zanella, Willi Menapace, Massimiliano Mancini, Yiming Wang, and Elisa Ricci. Harnessing large
language models for training-free video anomaly detection. InProceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 18527–18536, 2024.
[30] Yuchen Yang, Kwonjoon Lee, Behzad Dariush, Yinzhi Cao, and Shao-Yuan Lo. Follow the rules: reasoning
for video anomaly detection with large language models. InEuropean Conference on Computer Vision,
pages 304–322. Springer, 2024.
[31] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-
intensive nlp tasks.Advances in neural information processing systems, 33:9459–9474, 2020.
[32] Andra Acsintoae, Andrei Florescu, Mariana-Iuliana Georgescu, Tudor Mare, Paul Sumedrea, Radu Tudor
Ionescu, Fahad Shahbaz Khan, and Mubarak Shah. Ubnormal: New benchmark for supervised open-set
video anomaly detection. InProceedings of the IEEE/CVF conference on computer vision and pattern
recognition, pages 20143–20153, 2022.
[33] AI LANGCHAIN. Langgraph: Multi-agent framework for llms. 2025.URL https://github. com/langchain-
ai/langgraph. Accessed, pages 01–24, 2025.
[34] Sagar Imambi, Kolla Bhanu Prakash, and GR Kanagachidambaresan. Pytorch.Programming with
TensorFlow: solution for edge computing applications, pages 87–104, 2021.
[35] Ayush K Rai, Tarun Krishna, Feiyan Hu, Alexandru Drimbarean, Kevin McGuinness, Alan F Smeaton,
and Noel E O’connor. Video anomaly detection via spatio-temporal pseudo-anomaly generation: A unified
approach. InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
pages 3887–3899, 2024.
[36] Yu Tian, Guansong Pang, Yuanhong Chen, Rajvinder Singh, Johan W Verjans, and Gustavo Carneiro.
Weakly-supervised video anomaly detection with robust temporal feature magnitude learning. InICCV,
pages 4975–4986, 2021.
[37] Hang Zhou, Junqing Yu, and Wei Yang. Dual memory units with uncertainty regulation for weakly
supervised video anomaly detection. InProceedings of the AAAI Conference on Artificial Intelligence,
volume 37, pages 3769–3777, 2023.
22

[38] Muchao Ye, Weiyang Liu, and Pan He. Vera: Explainable video anomaly detection via verbalized learning
of vision-language models. InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pages 8679–8688, June 2025.
[39] Tianheng Cheng, Lin Song, Yixiao Ge, Wenyu Liu, Xinggang Wang, and Ying Shan. Yolo-world: Real-
time open-vocabulary object detection. InProceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 16901–16911, 2024.
[40] Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. Real-esrgan: Training real-world blind super-
resolution with pure synthetic data. InProceedings of the IEEE/CVF international conference on computer
vision, pages 1905–1914, 2021.
[41] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish
Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from
natural language supervision. InInternational conference on machine learning, pages 8748–8763. PmLR,
2021.
[42] Tavily. Tavily search api.https://www.tavily.com.
23